//! Player lifecycle: `open` spawns the decode pipeline, `VideoControl` is
//! the typed live-control handle.
//!
//! Threading model (SPEC V1/V3): `open` owns nothing global — it wires
//! per-player threads together with channels and hands both ends to the
//! caller. The [`VideoPlayer`] carries the receiving side (consumed by
//! [`crate::VideoRenderer`], or polled directly for headless use); the
//! cloneable [`VideoControl`] carries the command senders that the control
//! socket will drive in the integration step.

use std::path::PathBuf;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};

use crate::audio::{AudioInit, AudioLink, CallbackCmd, DecodeCmd};
use crate::decode::{DecodedFrame, Decoder, FRAME_QUEUE_CAP, VideoInfo};
use crate::error::VideoError;
use crate::scaling::ScalingMode;

/// Recycle-queue depth (frame pixel buffers travelling back to the decode
/// thread; a little deeper than the frame queue so nothing bounces).
const RECYCLE_QUEUE_CAP: usize = FRAME_QUEUE_CAP + 4;

/// Commands consumed by the renderer (drained per frame, SPEC V3).
#[derive(Debug, Clone, Copy)]
pub(crate) enum RendererCmd {
    /// Freeze/unfreeze the playback clock.
    Pause(bool),
    /// Playback rate for the wall clock (audio-master speed is handled by
    /// the audio thread).
    Speed(f64),
    /// Live scaling-mode change (control socket `scaling` command).
    Scaling(ScalingMode),
}

/// Initial playback options.
///
/// Volume is on the mpv 0–100 scale (docs/subsystems-misc.md §2.1); the
/// CLI's 0–128 `--volume` maps onto it as `volume * 100 / 128` at the
/// CVideo-equivalent layer (docs/subsystems-misc.md §2.3,
/// docs/compat-cli.md `-v/--volume`).
///
/// There is deliberately **no initial speed** here: in the reference,
/// `setSpeed` before playback start is silently lost because `init()`
/// never re-applies the latched speed (docs/subsystems-misc.md §2.1,
/// GLPlayer.cpp:107-113 quirk). Matching that, speed only takes effect
/// via [`VideoControl::set_speed`] on the live pipeline — which always
/// exists once `open` returns.
#[derive(Debug, Clone, Copy)]
pub struct VideoOptions {
    /// Volume 0–100 (docs/subsystems-misc.md §2.1 volume property).
    pub volume: f64,
    /// Mute, independent of volume (docs/subsystems-misc.md §2.1).
    pub mute: bool,
    /// `--silent` semantics: play with volume forced to 0, **not** paused
    /// (docs/subsystems-misc.md §2.3; docs/compat-cli.md `-s/--silent`).
    /// The audio pipeline keeps running so the audio clock stays master.
    pub silent: bool,
    /// Start paused (docs/subsystems-misc.md §2.1 pause: freeze frame,
    /// keep state; latched values apply from the start).
    pub paused: bool,
    /// Output scaling mode (docs/render-architecture.md §4).
    pub scaling: ScalingMode,
    /// `false` skips the audio pipeline entirely (headless/tests; the
    /// playback clock then falls back to wall clock × speed). `true` is
    /// the mpv-parity behavior.
    pub enable_audio: bool,
}

impl Default for VideoOptions {
    /// mpv-flavored defaults: volume 100, unmuted, playing, default
    /// scaling, audio on.
    fn default() -> Self {
        Self {
            volume: 100.0,
            mute: false,
            silent: false,
            paused: false,
            scaling: ScalingMode::Default,
            enable_audio: true,
        }
    }
}

/// Receiving side of a playing video: decoded frames, playback clock
/// inputs, and pending renderer commands. Consumed by
/// [`crate::VideoRenderer::new`], or polled directly (headless).
pub struct VideoPlayer {
    info: VideoInfo,
    frames_rx: Receiver<DecodedFrame>,
    recycle_tx: Sender<Vec<u8>>,
    commands_rx: Receiver<RendererCmd>,
    audio: Option<AudioLink>,
    scaling: ScalingMode,
    paused: bool,
    shutdown: Sender<()>,
}

/// Internals handed to the renderer.
pub(crate) struct PlayerParts {
    pub frames_rx: Receiver<DecodedFrame>,
    pub recycle_tx: Sender<Vec<u8>>,
    pub commands_rx: Receiver<RendererCmd>,
    pub audio: Option<AudioLink>,
    pub scaling: ScalingMode,
    pub paused: bool,
    pub shutdown: Sender<()>,
}

impl VideoPlayer {
    /// Open `path`, spawn its decode (and audio, if any) threads and
    /// return the player plus its control handle.
    ///
    /// Audio failures degrade to silent wall-clock playback with a
    /// warning instead of failing the whole wallpaper (the mpv reference
    /// keeps video running when audio output is unavailable; V9: typed
    /// errors, no panic).
    pub fn open(path: impl Into<PathBuf>, options: VideoOptions) -> Result<(Self, VideoControl), VideoError> {
        let path = path.into();

        let decoder = Decoder::open(&path)?;
        let info = decoder.info();

        // Bounded frame queue: the only decode-side pacing (SPEC V4).
        let (frames_tx, frames_rx) = bounded(FRAME_QUEUE_CAP);
        let (recycle_tx, recycle_rx) = bounded(RECYCLE_QUEUE_CAP);
        std::thread::Builder::new()
            .name("kirie-video-decode".into())
            .spawn(move || decoder.run(&frames_tx, &recycle_rx))?;

        let (renderer_tx, commands_rx) = unbounded();
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

        let (audio, callback_tx, decode_tx) = if options.enable_audio {
            let (callback_tx, callback_rx) = unbounded();
            let (decode_tx, decode_rx) = unbounded();
            let init = AudioInit {
                volume: options.volume,
                mute: options.mute,
                silent: options.silent,
                paused: options.paused,
            };
            match crate::audio::spawn(path.clone(), init, callback_rx, decode_rx, shutdown_rx) {
                Ok(Some(link)) => (Some(link), Some(callback_tx), Some(decode_tx)),
                Ok(None) => {
                    tracing::debug!(path = %path.display(), "no audio stream; wall clock master");
                    (None, None, None)
                }
                Err(err) => {
                    tracing::warn!(%err, "audio unavailable; playing without sound");
                    (None, None, None)
                }
            }
        } else {
            (None, None, None)
        };

        let player = Self {
            info,
            frames_rx,
            recycle_tx,
            commands_rx,
            audio,
            scaling: options.scaling,
            paused: options.paused,
            shutdown: shutdown_tx,
        };
        let control = VideoControl {
            renderer: renderer_tx,
            audio_callback: callback_tx,
            audio_decode: decode_tx,
        };
        Ok((player, control))
    }

    /// Probed stream properties.
    #[must_use]
    pub fn info(&self) -> VideoInfo {
        self.info
    }

    /// Whether the audio-master clock is active
    /// (docs/subsystems-misc.md §2.1: audio clock when audio present).
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Blocking frame receive for headless consumers and tests. Returns
    /// `None` on timeout or when the decode thread stopped.
    #[must_use]
    pub fn recv_frame_timeout(&self, timeout: Duration) -> Option<DecodedFrame> {
        self.frames_rx.recv_timeout(timeout).ok()
    }

    /// Hand a consumed frame buffer back for reuse (SPEC V5 recycling).
    pub fn recycle_buffer(&self, buffer: Vec<u8>) {
        let _ = self.recycle_tx.try_send(buffer);
    }

    /// Decompose into the renderer's working set.
    pub(crate) fn into_parts(self) -> PlayerParts {
        PlayerParts {
            frames_rx: self.frames_rx,
            recycle_tx: self.recycle_tx,
            commands_rx: self.commands_rx,
            audio: self.audio,
            scaling: self.scaling,
            paused: self.paused,
            shutdown: self.shutdown,
        }
    }
}

/// Typed live-control handle (SPEC V3: every mutation travels as a channel
/// command; clone freely — the control socket and fullscreen detector will
/// each hold one in the integration step).
#[derive(Debug, Clone)]
pub struct VideoControl {
    renderer: Sender<RendererCmd>,
    audio_callback: Option<Sender<CallbackCmd>>,
    audio_decode: Option<Sender<DecodeCmd>>,
}

impl VideoControl {
    /// Pause/resume: freeze frame, keep state
    /// (docs/subsystems-misc.md §2.1 `pause`).
    pub fn set_pause(&self, paused: bool) {
        let _ = self.renderer.send(RendererCmd::Pause(paused));
        if let Some(tx) = &self.audio_callback {
            let _ = tx.send(CallbackCmd::Pause(paused));
        }
    }

    /// Playback rate multiplier; values ≤ 0 (or non-finite) are coerced
    /// to 1.0 (docs/subsystems-misc.md §2.1, GLPlayer.cpp:107-113).
    pub fn set_speed(&self, speed: f64) {
        let speed = if speed > 0.0 && speed.is_finite() {
            speed
        } else {
            1.0
        };
        let _ = self.renderer.send(RendererCmd::Speed(speed));
        if let Some(tx) = &self.audio_decode {
            let _ = tx.send(DecodeCmd::Speed(speed));
        }
    }

    /// Volume on the 0–100 scale, clamped, applied live
    /// (docs/subsystems-misc.md §2.1 volume, §2.3 for the CLI 0–128
    /// mapping done by the caller).
    pub fn set_volume(&self, volume: f64) {
        if let Some(tx) = &self.audio_callback {
            let _ = tx.send(CallbackCmd::Volume(volume));
        }
    }

    /// Mute/unmute, independent of volume (docs/subsystems-misc.md §2.1;
    /// this is what the automute detector toggles, §2.3).
    pub fn set_mute(&self, mute: bool) {
        if let Some(tx) = &self.audio_callback {
            let _ = tx.send(CallbackCmd::Mute(mute));
        }
    }

    /// Live scaling-mode change (control socket `scaling` command,
    /// docs/render-architecture.md §4).
    pub fn set_scaling(&self, mode: ScalingMode) {
        let _ = self.renderer.send(RendererCmd::Scaling(mode));
    }
}
