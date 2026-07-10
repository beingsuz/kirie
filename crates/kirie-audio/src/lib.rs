//! kirie-audio — system audio capture + real-FFT spectrum for audio-reactive
//! wallpapers.
//!
//! Pipeline (docs/subsystems-misc.md §1.3, "Visualizer capture path"):
//!
//! ```text
//! PulseAudio monitor (U8 / 44100 / mono)          capture thread
//!        │  pa_stream_read → raw bytes
//!        ▼
//!   ringbuf SPSC (lock-free, V3)
//!        │  pop 1024-sample frames
//!        ▼
//!   real FFT (rustfft) → band reduction            FFT worker thread
//!   → noise gate → move_towards smoothing
//!        │  immutable AudioSpectrum
//!        ▼
//!   arc-swap  ──latest_spectrum()──►  render / uniform packer (never blocks, V4)
//! ```
//!
//! The exact numeric constants (1024-sample window, 0.35·log10 band gain,
//! `boost(x)=2-e^((1-x)-0.5)`, 0.3/frame smoothing, RMS gate 10.0) live in
//! [`dsp`] and are a 1:1 port of the C++ reference — see that module for
//! citations.
//!
//! Failure is never fatal (V9): a missing PulseAudio server, missing monitor
//! source, or `--no-audio-processing` all resolve to a silent (all-zero)
//! spectrum with no panic.
#![forbid(unsafe_code)]

mod automute;
mod capture;
pub mod dsp;
mod spectrum;
mod worker;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use arc_swap::ArcSwap;
use ringbuf::HeapRb;
use ringbuf::traits::Split;

pub use automute::AutoMute;
pub use dsp::{BANDS_16, BANDS_32, BANDS_64, DEFAULT_GATE, SAMPLE_RATE, SMOOTH_RATE, WAVE_BUFFER_SIZE};
pub use spectrum::AudioSpectrum;

/// Ring capacity: ~0.25 s of U8/44100/mono audio. Comfortably absorbs the
/// worker's tick jitter without dropping frames; overflow (worker stalled)
/// simply drops the oldest bytes, underflow yields zero-length drains.
const RING_CAPACITY: usize = SAMPLE_RATE as usize / 4;

/// Errors the capture thread can hit while opening the PulseAudio stream. These
/// are reported (via `status`/tracing) but never propagated as a panic — the
/// spectrum stays silent (V9).
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// The PulseAudio/PipeWire context failed to connect (no server running).
    #[error("failed to connect to PulseAudio server: {0}")]
    Connect(String),
    /// No capture source could be resolved (no default sink monitor and no
    /// `--audio-device` given).
    #[error("no monitor source available")]
    NoMonitor,
    /// The record stream failed to connect to the source.
    #[error("failed to connect record stream to source {source_name:?}: {reason}")]
    StreamConnect {
        /// The source name we tried to record from.
        source_name: String,
        /// Human-readable failure reason.
        reason: String,
    },
    /// The PulseAudio mainloop returned a fatal error while iterating.
    #[error("PulseAudio mainloop error")]
    Mainloop,
}

/// Coarse capture state, published lock-free for callers/tests to poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStatus {
    /// Audio processing disabled (`--no-audio-processing`) — spectrum is always
    /// silent, no threads spawned.
    Disabled,
    /// Threads spawned, stream not yet confirmed connected.
    Starting,
    /// Recording from a monitor source; spectrum is live.
    Running,
    /// Capture failed (see logs); spectrum is silent but the handle is valid.
    Failed,
}

impl CaptureStatus {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Disabled,
            1 => Self::Starting,
            2 => Self::Running,
            _ => Self::Failed,
        }
    }
    fn as_u8(self) -> u8 {
        match self {
            Self::Disabled => 0,
            Self::Starting => 1,
            Self::Running => 2,
            Self::Failed => 3,
        }
    }
}

/// Configuration for [`AudioCapture::start`]. Mirrors the parsed CLI knobs
/// (`--no-audio-processing`, `--audio-device`).
#[derive(Clone, Debug)]
pub struct AudioConfig {
    /// `settings.audio.audioprocessing` — `false` when `--no-audio-processing`
    /// is passed. Disabled → permanent silent spectrum, no threads.
    pub enabled: bool,
    /// `settings.audio.device` — a PulseAudio/PipeWire *source* name. `None`
    /// (or empty) selects the default sink's `<sink>.monitor` (cpp:121-128).
    pub device: Option<String>,
    /// Noise-gate RMS threshold. `None` reads `WPE_AUDIO_GATE` (env) falling
    /// back to [`DEFAULT_GATE`]; `Some(0.0)` disables the gate.
    pub gate: Option<f32>,
    /// Smoother/publish cadence. Defaults to 16 ms (~60 Hz) to mirror the C++
    /// per-render-frame `update()` at which `move_towards` slews.
    pub tick: Duration,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: None,
            gate: None,
            tick: Duration::from_millis(16),
        }
    }
}

impl AudioConfig {
    /// Enabled capture on the given device (`None` = default monitor).
    #[must_use]
    pub fn with_device(device: Option<String>) -> Self {
        Self {
            device: device.filter(|d| !d.is_empty()),
            ..Self::default()
        }
    }

    /// The disabled configuration (`--no-audio-processing`): always silent.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Resolve the effective gate threshold from `gate`/`WPE_AUDIO_GATE`.
    fn resolved_gate(&self) -> f32 {
        if let Some(g) = self.gate {
            return g;
        }
        match std::env::var("WPE_AUDIO_GATE") {
            Ok(v) => v.trim().parse::<f32>().unwrap_or(DEFAULT_GATE),
            Err(_) => DEFAULT_GATE,
        }
    }
}

/// Live handle to the audio pipeline. Holds the latest published spectrum and
/// owns the capture + FFT worker threads (joined on drop).
pub struct AudioCapture {
    shared: Arc<ArcSwap<AudioSpectrum>>,
    status: Arc<AtomicU8>,
    shutdown: Arc<AtomicBool>,
    device: Option<String>,
    capture_thread: Option<JoinHandle<()>>,
    worker_thread: Option<JoinHandle<()>>,
}

impl AudioCapture {
    /// Start the pipeline. Never fails: a disabled config or any capture error
    /// yields a valid handle whose spectrum stays silent (V9). Returns
    /// immediately — the PulseAudio connection is established off-thread.
    #[must_use]
    pub fn start(config: AudioConfig) -> Self {
        let shared = Arc::new(ArcSwap::from_pointee(AudioSpectrum::silent()));
        let shutdown = Arc::new(AtomicBool::new(false));

        if !config.enabled {
            let status = Arc::new(AtomicU8::new(CaptureStatus::Disabled.as_u8()));
            return Self {
                shared,
                status,
                shutdown,
                device: config.device.clone(),
                capture_thread: None,
                worker_thread: None,
            };
        }

        let status = Arc::new(AtomicU8::new(CaptureStatus::Starting.as_u8()));
        let device = config.device.clone();
        let gate = config.resolved_gate();
        let tick = config.tick;

        // SPSC ring: producer → capture thread, consumer → worker thread (V3).
        let (prod, cons) = HeapRb::<u8>::new(RING_CAPACITY).split();

        let worker_thread = {
            let shared = shared.clone();
            let shutdown = shutdown.clone();
            Some(
                std::thread::Builder::new()
                    .name("kirie-audio-fft".into())
                    .spawn(move || {
                        worker::run(cons, shared, shutdown, worker::WorkerParams { gate, tick });
                    })
                    .expect("spawn fft worker"),
            )
        };

        let capture_thread = {
            let status = status.clone();
            let shutdown = shutdown.clone();
            let device = device.clone();
            Some(
                std::thread::Builder::new()
                    .name("kirie-audio-capture".into())
                    .spawn(move || {
                        if let Err(e) = capture::run(device, prod, &status, &shutdown) {
                            status.store(CaptureStatus::Failed.as_u8(), Ordering::Relaxed);
                            tracing::warn!(error = %e, "audio capture unavailable; spectrum silent");
                        }
                    })
                    .expect("spawn capture thread"),
            )
        };

        Self {
            shared,
            status,
            shutdown,
            device,
            capture_thread,
            worker_thread,
        }
    }

    /// A disabled handle (always-silent) — convenience for the
    /// `--no-audio-processing` path.
    #[must_use]
    pub fn disabled() -> Self {
        Self::start(AudioConfig::disabled())
    }

    /// The latest published spectrum. Lock-free, never blocks the render thread
    /// (V4). Returns a shared `Arc` snapshot.
    #[must_use]
    pub fn latest_spectrum(&self) -> Arc<AudioSpectrum> {
        self.shared.load_full()
    }

    /// Current coarse capture state.
    #[must_use]
    pub fn status(&self) -> CaptureStatus {
        CaptureStatus::from_u8(self.status.load(Ordering::Relaxed))
    }

    /// The configured device (`None` = default sink monitor).
    #[must_use]
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.worker_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.capture_thread.take() {
            let _ = h.join();
        }
    }
}
