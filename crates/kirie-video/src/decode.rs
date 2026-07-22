//! Video decode thread: demux + decode + RGBA conversion.
//!
//! One thread per playing video. It demuxes the file's video stream,
//! decodes on the CPU, converts each frame to tightly-packed RGBA with
//! `sws_scale` at the *native* stream size (the texture is the video's
//! native display size; all scaling happens in composition,
//! docs/subsystems-misc.md §2.2), and sends timestamped frames through a
//! bounded channel (capacity [`FRAME_QUEUE_CAP`]). The bounded send is the
//! only pacing this thread has: when the renderer is behind — or the
//! output is occluded and no frame callbacks arrive — the thread parks in
//! `send` and does zero work (SPEC V4/V6). On EOF it seeks back to 0 and
//! keeps going, matching mpv `loop=inf` (docs/subsystems-misc.md §2.1).
//!
//! Frame pixel buffers are recycled through a return channel so the
//! steady-state loop allocates nothing on either side (SPEC V5 on the
//! render side).
//!
//! Hardware decode (SPEC T11): with the `vaapi` cargo feature, decoder
//! setup first tries a VAAPI hw device (`crate::hw`); decoded frames then
//! arrive as VAAPI surfaces and are downloaded to system memory right
//! before [`Decoder::convert`]'s sws_scale + CPU copy. Every init failure
//! (no render node, no driver, unsupported codec) degrades to this CPU
//! path with an info log, so behavior without a VAAPI stack is unchanged.
//! True zero-copy (export the surface as dma-buf, import as an external
//! wgpu texture) remains follow-up work.

use std::path::{Path, PathBuf};

use crossbeam_channel::{Receiver, Sender};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling;

use crate::error::VideoError;
use crate::pacing::{LoopTimeline, Timed};

/// Bounded frame-queue depth between the decode thread and the renderer.
pub const FRAME_QUEUE_CAP: usize = 4;

/// Fallback frame duration when neither the stream nor the frames expose
/// timing (docs/subsystems-misc.md §2 gives no contract for untimed
/// streams; 30 fps is a neutral guess, flagged in logs).
const FALLBACK_FRAME_DUR: f64 = 1.0 / 30.0;

/// One decoded RGBA frame with its monotonic playback timestamp.
#[derive(Debug)]
pub struct DecodedFrame {
    /// Monotonic playback timestamp in seconds (continuous across loops,
    /// see [`LoopTimeline`]).
    pub play_pts: f64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Tightly packed RGBA8 pixels (`width * height * 4` bytes, row 0 =
    /// top).
    pub data: Vec<u8>,
}

impl Timed for DecodedFrame {
    fn play_pts(&self) -> f64 {
        self.play_pts
    }
}

/// Probed properties of the video stream.
#[derive(Debug, Clone, Copy)]
pub struct VideoInfo {
    /// Native width in pixels (docs/subsystems-misc.md §2.2: texture is
    /// the native display size).
    pub width: u32,
    /// Native height in pixels.
    pub height: u32,
    /// Average frame rate in Hz (0.0 when unknown).
    pub frame_rate: f64,
    /// Container duration of one loop iteration in seconds (0.0 when
    /// unknown).
    pub duration: f64,
}

/// Demuxer + decoder state, moved into the decode thread. Deliberately
/// holds no `SwsContext`: the scaler is not `Send`, so it lives in a
/// thread-local [`Converter`] built inside [`Decoder::run`].
pub(crate) struct Decoder {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    stream_index: usize,
    /// Stream time base in seconds per tick.
    time_base: f64,
    /// Stream start time in seconds (subtracted so raw PTS starts at 0).
    start: f64,
    info: VideoInfo,
    timeline: LoopTimeline,
    decoded: ffmpeg::frame::Video,
    /// Last raw PTS in seconds, for frame-duration estimation.
    last_raw: Option<f64>,
    /// Consecutive undecodable packets skipped, for log throttling (SPEC
    /// V9: a corrupt run must degrade gracefully, not flood the journal).
    undecodable: u64,
    /// Consecutive unconvertible frames dropped, same throttling story
    /// (e.g. a persistently failing VAAPI hw→system download).
    unconvertible: u64,
    /// Estimated per-frame duration in seconds.
    frame_dur: f64,
    /// Synthesized PTS for streams that provide none.
    synth_pts: f64,
}

/// RGBA conversion state; created on (and confined to) the decode thread
/// because `SwsContext` is not `Send`.
struct Converter {
    scaler: Option<scaling::Context>,
    rgb: ffmpeg::frame::Video,
    /// VAAPI surface → system-memory download state; idle (allocating
    /// nothing) unless hardware decode actually engaged.
    #[cfg(feature = "vaapi")]
    hw: crate::hw::HwDownload,
}

impl Converter {
    fn new() -> Self {
        Self {
            scaler: None,
            rgb: ffmpeg::frame::Video::empty(),
            #[cfg(feature = "vaapi")]
            hw: crate::hw::HwDownload::new(),
        }
    }
}

impl Decoder {
    /// Open `path` and locate/open the best video stream.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        ffmpeg::init()?;
        let input = ffmpeg::format::input(path)?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| VideoError::NoVideoStream(PathBuf::from(path)))?;
        let stream_index = stream.index();
        let time_base = f64::from(stream.time_base());
        let start = if stream.start_time() == i64::MIN {
            0.0
        } else {
            stream.start_time() as f64 * time_base
        };
        let frame_rate = f64::from(stream.avg_frame_rate()).max(0.0);
        let decoder = open_video_decoder(&stream)?;
        let (width, height) = (decoder.width(), decoder.height());
        if width == 0 || height == 0 {
            return Err(VideoError::InvalidDimensions { width, height });
        }
        let duration = if input.duration() > 0 {
            input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
        } else {
            0.0
        };
        let info = VideoInfo {
            width,
            height,
            frame_rate,
            duration,
        };
        let frame_dur = if frame_rate > 0.0 {
            1.0 / frame_rate
        } else {
            FALLBACK_FRAME_DUR
        };
        Ok(Self {
            input,
            decoder,
            stream_index,
            time_base,
            start,
            info,
            timeline: LoopTimeline::new((duration > 0.0).then_some(duration)),
            decoded: ffmpeg::frame::Video::empty(),
            last_raw: None,
            undecodable: 0,
            unconvertible: 0,
            frame_dur,
            synth_pts: 0.0,
        })
    }

    /// Probed stream properties.
    pub fn info(&self) -> VideoInfo {
        self.info
    }

    /// Decode forever (mpv `loop` = `inf`, docs/subsystems-misc.md §2.1),
    /// until the frame receiver disconnects.
    pub fn run(mut self, frames: &Sender<DecodedFrame>, recycle: &Receiver<Vec<u8>>) {
        // The sws scaler is not `Send`; it is created here, on the decode
        // thread, and never leaves it.
        let mut converter = Converter::new();
        let mut consecutive_read_errors = 0u32;
        loop {
            // One pass through the file.
            loop {
                let mut packet = ffmpeg::Packet::empty();
                match packet.read(&mut self.input) {
                    Ok(()) => consecutive_read_errors = 0,
                    Err(ffmpeg::Error::Eof) => break,
                    Err(err) => {
                        // Malformed data must not kill playback or panic
                        // (SPEC V9), but a persistently failing source
                        // must not spin either.
                        consecutive_read_errors += 1;
                        if consecutive_read_errors > 1000 {
                            tracing::error!(%err, "video demux failing persistently; stopping");
                            return;
                        }
                        continue;
                    }
                }
                if packet.stream() != self.stream_index {
                    continue;
                }
                if let Err(err) = self.decoder.send_packet(&packet) {
                    // Corrupt packets are skipped (SPEC V9); a whole corrupt
                    // region (or a persistently broken re-looping file) must
                    // not flood the log, so warn on a power-of-two cadence
                    // and carry the running count.
                    self.undecodable += 1;
                    if self.undecodable.is_power_of_two() {
                        tracing::warn!(%err, count = self.undecodable, "skipping undecodable video packet(s)");
                    }
                    continue;
                }
                self.undecodable = 0;
                if !self.drain(&mut converter, frames, recycle) {
                    return;
                }
            }

            // EOF: flush the decoder, then seek back to 0 and continue —
            // infinite seamless loop (docs/subsystems-misc.md §2.1
            // `loop=inf`; same EOF/seek dance as the C++ audio reader,
            // AudioStream.cpp:35-46 via docs/subsystems-misc.md §1.1).
            let _ = self.decoder.send_eof();
            if !self.drain(&mut converter, frames, recycle) {
                return;
            }
            if let Err(err) = self.input.seek(0, ..) {
                tracing::error!(%err, "loop seek to 0 failed; stopping video decode");
                return;
            }
            self.decoder.flush();
            self.timeline.wrap();
            self.last_raw = None;
            self.synth_pts = 0.0;
        }
    }

    /// Receive every frame the decoder has ready and send it converted.
    /// Returns `false` when the receiver hung up.
    fn drain(
        &mut self,
        converter: &mut Converter,
        frames: &Sender<DecodedFrame>,
        recycle: &Receiver<Vec<u8>>,
    ) -> bool {
        loop {
            match self.decoder.receive_frame(&mut self.decoded) {
                Ok(()) => match self.convert(converter, recycle) {
                    Ok(frame) => {
                        self.unconvertible = 0;
                        // Bounded send: blocks when the queue is full,
                        // which is the entire backpressure story (V4/V6).
                        if frames.send(frame).is_err() {
                            return false;
                        }
                    }
                    Err(err) => {
                        // Throttled like undecodable packets (SPEC V9): a
                        // persistent failure — e.g. a VAAPI download that
                        // stops working — must not flood the journal.
                        self.unconvertible += 1;
                        if self.unconvertible.is_power_of_two() {
                            tracing::warn!(%err, count = self.unconvertible, "dropping unconvertible video frame(s)");
                        }
                    }
                },
                // EAGAIN (needs more input) or EOF (fully drained).
                Err(_) => return true,
            }
        }
    }

    /// Convert the frame the decoder just produced to a timestamped RGBA
    /// frame.
    ///
    /// T11 (VAAPI) seam: with the `vaapi` feature, frames decoded in
    /// hardware arrive as VAAPI surfaces and are downloaded to system
    /// memory (typically NV12) here, then take the same sws RGBA path.
    /// Zero-copy dma-buf → wgpu import would replace this download + copy
    /// and is follow-up work.
    fn convert(
        &mut self,
        converter: &mut Converter,
        recycle: &Receiver<Vec<u8>>,
    ) -> Result<DecodedFrame, VideoError> {
        #[cfg(feature = "vaapi")]
        let decoded = match converter.hw.download(&self.decoded)? {
            Some(sw) => sw,
            None => &self.decoded,
        };
        #[cfg(not(feature = "vaapi"))]
        let decoded = &self.decoded;

        let (width, height) = (decoded.width(), decoded.height());
        if width == 0 || height == 0 {
            return Err(VideoError::InvalidDimensions { width, height });
        }

        // (Re)build the scaler only when the source geometry/format
        // changes — the mpv contract resizes output on VIDEO_RECONFIG
        // only (docs/subsystems-misc.md §2.2).
        let needs_scaler = match &converter.scaler {
            None => true,
            Some(s) => {
                s.input().format != decoded.format() || s.input().width != width || s.input().height != height
            }
        };
        if needs_scaler {
            converter.scaler = Some(scaling::Context::get(
                decoded.format(),
                width,
                height,
                Pixel::RGBA,
                width,
                height,
                // Same-size format conversion; FAST_BILINEAR mirrors the
                // mpv `profile=fast` speed-over-quality intent
                // (docs/subsystems-misc.md §2.1).
                scaling::Flags::FAST_BILINEAR,
            )?);
            converter.rgb = ffmpeg::frame::Video::empty();
            if width != self.info.width || height != self.info.height {
                tracing::info!(
                    from = format!("{}x{}", self.info.width, self.info.height),
                    to = format!("{width}x{height}"),
                    "video stream geometry changed"
                );
                self.info.width = width;
                self.info.height = height;
            }
        }
        let Some(scaler) = converter.scaler.as_mut() else {
            // Unreachable by construction; keep V9 (no panic) anyway.
            return Err(VideoError::InvalidDimensions { width, height });
        };
        scaler.run(decoded, &mut converter.rgb)?;

        // Raw PTS in seconds within the file (best-effort timestamp, then
        // pts, then synthesized from the frame rate). Always read off the
        // decoder's own frame: the VAAPI download copies pixels, not props.
        let raw = match self.decoded.timestamp().or_else(|| self.decoded.pts()) {
            Some(ts) => ts as f64 * self.time_base - self.start,
            None => self.synth_pts,
        };
        if let Some(last) = self.last_raw {
            let delta = raw - last;
            if delta > 0.0 && delta < 1.0 {
                self.frame_dur = delta;
            }
        }
        self.last_raw = Some(raw);
        self.synth_pts = raw + self.frame_dur;
        let play_pts = self.timeline.map(raw, self.frame_dur);

        // Copy into a recycled buffer (steady state: no allocation).
        let mut data = recycle.try_recv().unwrap_or_default();
        copy_rgba(&converter.rgb, &mut data);

        Ok(DecodedFrame {
            play_pts,
            width,
            height,
            data,
        })
    }
}

/// Open the stream's video decoder.
///
/// With the `vaapi` feature this first tries a VAAPI hw device (SPEC T11);
/// any init failure — no render node, no driver, codec without VAAPI
/// support, open failure — degrades to the plain CPU decoder with an info
/// log, leaving the no-VAAPI behavior contract untouched.
fn open_video_decoder(
    stream: &ffmpeg::format::stream::Stream<'_>,
) -> Result<ffmpeg::decoder::Video, VideoError> {
    #[cfg(feature = "vaapi")]
    {
        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        match crate::hw::attach_vaapi(&mut context) {
            Ok(()) => match context.decoder().video() {
                Ok(decoder) => {
                    tracing::info!("VAAPI device attached; hardware decode enabled for supported profiles");
                    return Ok(decoder);
                }
                Err(err) => {
                    tracing::info!(%err, "VAAPI decoder open failed; falling back to CPU decode");
                }
            },
            Err(err) => tracing::info!(%err, "VAAPI unavailable; using CPU decode"),
        }
    }
    Ok(
        ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .video()?,
    )
}

/// Copy the RGBA plane into `buf`, dropping any stride padding so the
/// result is exactly `width * 4` bytes per row (what
/// `wgpu::Queue::write_texture` gets fed).
fn copy_rgba(rgb: &ffmpeg::frame::Video, buf: &mut Vec<u8>) {
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let row = width * 4;
    let stride = rgb.stride(0);
    let data = rgb.data(0);
    buf.clear();
    buf.reserve_exact(row * height);
    if stride == row {
        buf.extend_from_slice(&data[..row * height]);
    } else {
        for y in 0..height {
            let start = y * stride;
            buf.extend_from_slice(&data[start..start + row]);
        }
    }
}
