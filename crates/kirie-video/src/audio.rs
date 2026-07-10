//! Audio decode + resample + cpal output, and the audio-master clock feed.
//!
//! One thread per playing video that has an audio stream. The thread owns
//! the demuxer, decoder, `SwrContext` resampler and the cpal output
//! stream (cpal streams are not `Send`; everything device-related lives
//! and dies on this thread). Decoded audio is resampled to the device's
//! rate/channel-count as packed f32 and pushed into a lock-free SPSC ring
//! (`ringbuf`); the device callback pops, applies gain, converts to the
//! device sample type and publishes consumption snapshots. All
//! cross-thread traffic is channels, the SPSC ring, and immutable
//! `triple_buffer` snapshots (SPEC V3).
//!
//! Behavior contract (docs/subsystems-misc.md §2.1, §2.3):
//! * `volume` is 0–100 (the CLI's 0–128 maps onto it as
//!   `volume * 100 / 128` at the CVideo-equivalent layer, §2.3). The
//!   docs fix the endpoints (0 = silent, 100 = full) but not the curve;
//!   gain is applied linearly here.
//! * `mute` is independent of volume (§2.1).
//! * `--silent` plays the video with volume 0, *not* paused (§2.3) — the
//!   pipeline keeps running so the audio clock stays master.
//! * speed changes rebuild the resampler so the device consumes media
//!   `speed×` faster (output rate = device rate / speed); pitch follows,
//!   like mpv without scaletempo. Speed ≤ 0 coerces to 1.0 (§2.1).
//! * EOF seeks back to 0 and continues (`loop=inf`, §2.1). The wrap
//!   advances by at least the container duration so the independently
//!   demuxed video stays aligned (see [`LoopTimeline`]).
//!
//! Known approximation: this loops the *audio stream* independently of
//! the video stream. mpv restarts the whole file, so a file whose audio
//! track is much shorter than its video would behave differently (audio
//! repeats instead of going silent). Same-length tracks — the corpus
//! case — are unaffected.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::ChannelLayout;
use ffmpeg_next::format::Sample;
use ffmpeg_next::format::sample::Type as SampleType;
use ffmpeg_next::software::resampling;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Producer, Split};

use crate::clock::{ConsumerSnap, ProducerSnap};
use crate::error::VideoError;
use crate::pacing::LoopTimeline;

/// Ring depth in seconds of device-rate audio. Small enough that volume
/// (applied at the callback) reacts instantly and speed-change error is
/// bounded; large enough to ride out decode-thread scheduling hiccups.
const RING_SECONDS: f64 = 0.5;

/// Sleep while the ring is full (decode thread backpressure).
const RING_FULL_BACKOFF: Duration = Duration::from_millis(5);

/// How long `spawn` waits for the audio thread to report its setup result.
const SETUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Commands handled by the device callback (SPEC V3: state changes travel
/// as messages; the callback owns its own gain state).
#[derive(Debug, Clone, Copy)]
pub(crate) enum CallbackCmd {
    /// Set volume, 0–100 (docs/subsystems-misc.md §2.1).
    Volume(f64),
    /// Mute on/off, independent of volume (§2.1).
    Mute(bool),
    /// Pause: emit silence, stop consuming, freeze the audio clock (§2.1).
    Pause(bool),
}

/// Commands handled by the audio decode thread.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DecodeCmd {
    /// Playback rate; resampler is rebuilt (≤ 0 already coerced to 1.0).
    Speed(f64),
}

/// Initial mixer state (latched values applied from the start, matching
/// the reference where volume/mute/pause set before playback take effect
/// — docs/subsystems-misc.md §2.1; speed intentionally excluded, see the
/// quirk note in `player.rs`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AudioInit {
    pub volume: f64,
    pub mute: bool,
    pub silent: bool,
    pub paused: bool,
}

/// What the renderer needs to read the audio-master clock.
pub(crate) struct AudioLink {
    /// Decode-side snapshots (what was pushed).
    pub producer: triple_buffer::Output<ProducerSnap>,
    /// Callback-side snapshots (what was consumed).
    pub consumer: triple_buffer::Output<ConsumerSnap>,
    /// Device sample rate the counters are measured in.
    pub sample_rate: u32,
}

/// Result of the in-thread setup, reported once through a channel.
enum Setup {
    Ready(AudioLink),
    /// The file has no audio stream — caller falls back to wall clock.
    NoStream,
    Failed(VideoError),
}

/// Spawn the audio thread for `path`.
///
/// Returns `Ok(None)` when the file has no audio stream (playback clock
/// falls back to wall clock, docs/subsystems-misc.md §2.1 pacing) and
/// `Err` when audio setup failed — callers may degrade to silent playback
/// rather than abort (mpv keeps video running without audio too).
pub(crate) fn spawn(
    path: PathBuf,
    init: AudioInit,
    callback_rx: Receiver<CallbackCmd>,
    decode_rx: Receiver<DecodeCmd>,
    shutdown_rx: Receiver<()>,
) -> Result<Option<AudioLink>, VideoError> {
    let (setup_tx, setup_rx) = bounded(1);
    std::thread::Builder::new()
        .name("kirie-audio-decode".into())
        .spawn(move || run_thread(&path, init, &callback_rx, &decode_rx, &shutdown_rx, &setup_tx))?;
    match setup_rx.recv_timeout(SETUP_TIMEOUT) {
        Ok(Setup::Ready(link)) => Ok(Some(link)),
        Ok(Setup::NoStream) => Ok(None),
        Ok(Setup::Failed(err)) => Err(err),
        Err(_) => Err(VideoError::AudioOutput(
            "audio setup did not report within timeout".into(),
        )),
    }
}

/// Everything the audio thread does: probe, device bring-up, decode loop.
fn run_thread(
    path: &std::path::Path,
    init: AudioInit,
    callback_rx: &Receiver<CallbackCmd>,
    decode_rx: &Receiver<DecodeCmd>,
    shutdown_rx: &Receiver<()>,
    setup_tx: &Sender<Setup>,
) {
    let setup = || -> Result<Option<(DecodeState, cpal::Stream, AudioLink)>, VideoError> {
        ffmpeg::init()?;
        let input = ffmpeg::format::input(path)?;
        let Some(stream) = input.streams().best(ffmpeg::media::Type::Audio) else {
            return Ok(None);
        };
        let stream_index = stream.index();
        let time_base = f64::from(stream.time_base());
        let start = if stream.start_time() == i64::MIN {
            0.0
        } else {
            stream.start_time() as f64 * time_base
        };
        let duration = if input.duration() > 0 {
            input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
        } else {
            0.0
        };
        let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .audio()?;

        // Device bring-up.
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| VideoError::AudioOutput("no default output device".into()))?;
        let supported = device
            .default_output_config()
            .map_err(|e| VideoError::AudioOutput(e.to_string()))?;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.config();
        let sample_rate = config.sample_rate.0;
        let channels = usize::from(config.channels);
        if sample_rate == 0 || channels == 0 {
            return Err(VideoError::AudioOutput(
                "device reports zero rate/channels".into(),
            ));
        }

        let ring_cap = ((f64::from(sample_rate) * RING_SECONDS) as usize).max(1024) * channels;
        let (ring_prod, ring_cons) = HeapRb::<f32>::new(ring_cap).split();

        let now = Instant::now();
        let (prod_in, prod_out) = triple_buffer::triple_buffer(&ProducerSnap::default());
        let (cons_in, cons_out) = triple_buffer::triple_buffer(&ConsumerSnap::initial(now));

        let callback = CallbackState {
            ring: ring_cons,
            commands: callback_rx.clone(),
            snap: cons_in,
            scratch: vec![0.0; 8192 * channels],
            channels,
            volume: init.volume.clamp(0.0, 100.0),
            mute: init.mute,
            silent: init.silent,
            paused: init.paused,
            consumed: 0,
        };
        let stream = build_stream(&device, &config, sample_format, callback)?;
        stream
            .play()
            .map_err(|e| VideoError::AudioOutput(e.to_string()))?;

        let state = DecodeState {
            input,
            decoder,
            stream_index,
            time_base,
            start,
            timeline: LoopTimeline::new((duration > 0.0).then_some(duration)),
            resampler: None,
            device_rate: sample_rate,
            out_layout: ChannelLayout::default(channels as i32),
            channels,
            speed: 1.0,
            ring: ring_prod,
            snap: prod_in,
            pushed: 0,
            head: 0.0,
            synth_pts: 0.0,
            decoded: ffmpeg::frame::Audio::empty(),
            undecodable: 0,
        };
        let link = AudioLink {
            producer: prod_out,
            consumer: cons_out,
            sample_rate,
        };
        Ok(Some((state, stream, link)))
    };

    match setup() {
        Ok(Some((mut state, stream, link))) => {
            let _ = setup_tx.send(Setup::Ready(link));
            state.run(decode_rx, shutdown_rx);
            // Keep the device stream alive for the whole decode loop.
            drop(stream);
        }
        Ok(None) => {
            let _ = setup_tx.send(Setup::NoStream);
        }
        Err(err) => {
            let _ = setup_tx.send(Setup::Failed(err));
        }
    }
}

/// Gain/consumption state owned by the device callback closure.
struct CallbackState {
    ring: ringbuf::HeapCons<f32>,
    commands: Receiver<CallbackCmd>,
    snap: triple_buffer::Input<ConsumerSnap>,
    scratch: Vec<f32>,
    channels: usize,
    volume: f64,
    mute: bool,
    silent: bool,
    paused: bool,
    consumed: u64,
}

impl CallbackState {
    /// Fill one device buffer. Runs on the cpal audio thread.
    fn fill<T: cpal::SizedSample + cpal::FromSample<f32>>(&mut self, data: &mut [T]) {
        while let Ok(cmd) = self.commands.try_recv() {
            match cmd {
                CallbackCmd::Volume(v) => self.volume = v.clamp(0.0, 100.0),
                CallbackCmd::Mute(m) => self.mute = m,
                CallbackCmd::Pause(p) => self.paused = p,
            }
        }

        let silence = T::from_sample(0.0f32);
        if self.paused {
            // Freeze: emit silence without consuming, so the audio-master
            // clock stops (docs/subsystems-misc.md §2.1 pause).
            data.fill(silence);
            self.snap.write(ConsumerSnap {
                consumed: self.consumed,
                at: Instant::now(),
                paused: true,
            });
            return;
        }

        if self.scratch.len() < data.len() {
            // Device grew its buffer beyond the preallocation; one-time
            // resize outside any render path.
            self.scratch.resize(data.len(), 0.0);
        }
        let got = self.ring.pop_slice(&mut self.scratch[..data.len()]);
        // Linear 0–100 gain; mute/silent force 0 (docs/subsystems-misc.md
        // §2.1 volume/mute, §2.3 silent).
        let gain = if self.mute || self.silent {
            0.0
        } else {
            (self.volume / 100.0) as f32
        };
        for (dst, src) in data.iter_mut().zip(&self.scratch[..got]) {
            *dst = T::from_sample(*src * gain);
        }
        for dst in &mut data[got..] {
            *dst = silence;
        }
        self.consumed += (got / self.channels.max(1)) as u64;
        self.snap.write(ConsumerSnap {
            consumed: self.consumed,
            at: Instant::now(),
            paused: false,
        });
    }
}

/// Build the output stream for whatever sample type the device wants.
fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    state: CallbackState,
) -> Result<cpal::Stream, VideoError> {
    fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        mut state: CallbackState,
    ) -> Result<cpal::Stream, VideoError> {
        device
            .build_output_stream(
                config,
                move |data: &mut [T], _| state.fill(data),
                |err| tracing::warn!(%err, "cpal stream error"),
                None,
            )
            .map_err(|e| VideoError::AudioOutput(e.to_string()))
    }
    match sample_format {
        cpal::SampleFormat::F32 => build::<f32>(device, config, state),
        cpal::SampleFormat::I16 => build::<i16>(device, config, state),
        cpal::SampleFormat::U16 => build::<u16>(device, config, state),
        cpal::SampleFormat::I32 => build::<i32>(device, config, state),
        other => Err(VideoError::UnsupportedSampleFormat(format!("{other:?}"))),
    }
}

/// Demux/decode/resample state for the audio stream.
struct DecodeState {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Audio,
    stream_index: usize,
    time_base: f64,
    start: f64,
    timeline: LoopTimeline,
    resampler: Option<resampling::Context>,
    device_rate: u32,
    out_layout: ChannelLayout,
    channels: usize,
    speed: f64,
    ring: ringbuf::HeapProd<f32>,
    snap: triple_buffer::Input<ProducerSnap>,
    /// Device frames pushed so far.
    pushed: u64,
    /// Playback seconds of the end of pushed data.
    head: f64,
    /// Synthesized raw PTS for streams without timestamps.
    synth_pts: f64,
    decoded: ffmpeg::frame::Audio,
    /// Consecutive undecodable packets skipped, for log throttling (SPEC
    /// V9: a corrupt run must degrade gracefully, not flood the journal).
    undecodable: u64,
}

impl DecodeState {
    /// Output sample rate implementing playback speed: the device drains
    /// at `device_rate`, so producing `device_rate / speed` samples per
    /// media second plays `speed×` faster (pitch shifts with it;
    /// docs/subsystems-misc.md §2.1 defines only that `speed` multiplies
    /// the playback rate).
    fn out_rate(&self) -> u32 {
        ((f64::from(self.device_rate) / self.speed).round() as u32).max(1)
    }

    fn make_resampler(&self) -> Result<resampling::Context, VideoError> {
        let in_layout = if self.decoder.channel_layout().is_empty() {
            ChannelLayout::default(i32::from(self.decoder.channels()))
        } else {
            self.decoder.channel_layout()
        };
        Ok(resampling::Context::get(
            self.decoder.format(),
            in_layout,
            self.decoder.rate(),
            Sample::F32(SampleType::Packed),
            self.out_layout,
            self.out_rate(),
        )?)
    }

    /// Decode/push until shutdown. Mirrors the video loop: EOF → seek 0 →
    /// continue (docs/subsystems-misc.md §2.1 `loop=inf`).
    fn run(&mut self, decode_rx: &Receiver<DecodeCmd>, shutdown_rx: &Receiver<()>) {
        let mut consecutive_read_errors = 0u32;
        loop {
            loop {
                if !self.poll_commands(decode_rx, shutdown_rx) {
                    return;
                }
                let mut packet = ffmpeg::Packet::empty();
                match packet.read(&mut self.input) {
                    Ok(()) => consecutive_read_errors = 0,
                    Err(ffmpeg::Error::Eof) => break,
                    Err(err) => {
                        consecutive_read_errors += 1;
                        if consecutive_read_errors > 1000 {
                            tracing::error!(%err, "audio demux failing persistently; stopping");
                            return;
                        }
                        continue;
                    }
                }
                if packet.stream() != self.stream_index {
                    continue;
                }
                if let Err(err) = self.decoder.send_packet(&packet) {
                    // Corrupt packets are skipped (SPEC V9); throttle the
                    // warning so a corrupt region (re-hit every loop) does
                    // not flood the log.
                    self.undecodable += 1;
                    if self.undecodable.is_power_of_two() {
                        tracing::warn!(%err, count = self.undecodable, "skipping undecodable audio packet(s)");
                    }
                    continue;
                }
                self.undecodable = 0;
                if !self.drain(shutdown_rx) {
                    return;
                }
            }
            let _ = self.decoder.send_eof();
            if !self.drain(shutdown_rx) {
                return;
            }
            if let Err(err) = self.input.seek(0, ..) {
                tracing::error!(%err, "audio loop seek failed; stopping");
                return;
            }
            self.decoder.flush();
            self.timeline.wrap();
            self.synth_pts = 0.0;
        }
    }

    /// Apply pending speed commands; `false` means shut down.
    fn poll_commands(&mut self, decode_rx: &Receiver<DecodeCmd>, shutdown_rx: &Receiver<()>) -> bool {
        if matches!(shutdown_rx.try_recv(), Err(TryRecvError::Disconnected)) {
            return false;
        }
        while let Ok(cmd) = decode_rx.try_recv() {
            match cmd {
                DecodeCmd::Speed(speed) => {
                    if (speed - self.speed).abs() > f64::EPSILON {
                        self.speed = speed;
                        // Rebuild at the new output rate; buffered swr
                        // state is dropped (sub-ring transient, see
                        // clock.rs docs).
                        self.resampler = None;
                    }
                }
            }
        }
        true
    }

    /// Receive decoded audio frames, resample, push. `false` = shut down.
    fn drain(&mut self, shutdown_rx: &Receiver<()>) -> bool {
        loop {
            if self.decoder.receive_frame(&mut self.decoded).is_err() {
                // EAGAIN (needs more input) or EOF (fully drained).
                return true;
            }
            if let Err(err) = self.process_frame(shutdown_rx) {
                match err {
                    ProcessStop::Shutdown => return false,
                    ProcessStop::Error(err) => {
                        tracing::warn!(%err, "dropping unprocessable audio frame");
                    }
                }
            }
        }
    }

    /// Resample `self.decoded` and push it into the ring.
    fn process_frame(&mut self, shutdown_rx: &Receiver<()>) -> Result<(), ProcessStop> {
        let samples = self.decoded.samples();
        if samples == 0 {
            return Ok(());
        }
        let in_rate = self.decoder.rate().max(1);

        if self.resampler.is_none() {
            self.resampler = Some(self.make_resampler().map_err(ProcessStop::Error)?);
        }
        let Some(resampler) = self.resampler.as_mut() else {
            return Ok(());
        };

        // Capacity: converted input + whatever swr still buffers, plus
        // slack — keeps the swr backlog bounded without a flush loop.
        let out_rate = resampler.output().rate;
        let backlog = resampler.delay().map_or(0, |d| d.output.max(0)) as usize;
        let cap = samples * out_rate as usize / in_rate as usize + backlog + 256;
        let mut out = ffmpeg::frame::Audio::new(Sample::F32(SampleType::Packed), cap, self.out_layout);
        resampler
            .run(&self.decoded, &mut out)
            .map_err(|e| ProcessStop::Error(e.into()))?;

        // Raw media time at the end of this frame.
        let raw = match self.decoded.timestamp().or_else(|| self.decoded.pts()) {
            Some(ts) => ts as f64 * self.time_base - self.start,
            None => self.synth_pts,
        };
        let dur = samples as f64 / f64::from(in_rate);
        self.synth_pts = raw + dur;
        let play = self.timeline.map(raw, dur);

        // Push converted samples (packed f32) with backpressure.
        let produced = out.samples() * self.channels;
        if produced > 0 {
            let bytes = &out.data(0)[..produced * size_of::<f32>()];
            let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(bytes) else {
                // AVFrame buffers are 32-byte aligned; this cannot fire in
                // practice, but never panic on data layout (SPEC V9).
                return Err(ProcessStop::Error(VideoError::AudioOutput(
                    "resampler output misaligned".into(),
                )));
            };
            let mut offset = 0;
            while offset < floats.len() {
                offset += self.ring.push_slice(&floats[offset..]);
                if offset < floats.len() {
                    if matches!(shutdown_rx.try_recv(), Err(TryRecvError::Disconnected)) {
                        return Err(ProcessStop::Shutdown);
                    }
                    // Ring full: the wallpaper is ahead; park briefly.
                    std::thread::sleep(RING_FULL_BACKOFF);
                }
            }
            self.pushed += (out.samples()) as u64;
        }

        // Publish the producer snapshot (immutable, SPEC V3). `head` is
        // the media end of what was *decoded*; the ≤ few ms still inside
        // swr are ignored (documented approximation).
        self.head = play + dur;
        self.snap.write(ProducerSnap {
            pushed: self.pushed,
            head: self.head,
            speed: self.speed,
        });
        Ok(())
    }
}

/// Why `process_frame` stopped.
enum ProcessStop {
    /// Owner went away; exit the thread.
    Shutdown,
    /// This frame failed; skip it and continue (SPEC V9).
    Error(VideoError),
}
