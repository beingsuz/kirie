//! FFT worker: drains raw U8 bytes from the SPSC ring, assembles 1024-sample
//! frames, runs the real FFT + band reduction, applies the per-frame smoother
//! and publishes an immutable [`AudioSpectrum`] snapshot via `arc-swap`
//! (V3 immutable snapshots, V4 render reads latest lock-free).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use ringbuf::HeapCons;
use ringbuf::traits::Consumer;
use rustfft::{Fft, FftPlanner};

use crate::dsp::{BandTargets, Smoother, WAVE_BUFFER_SIZE};
use crate::spectrum::AudioSpectrum;

/// Assembles the ring's byte stream into 1024-sample frames, keeping only the
/// **latest** complete window when a drain delivers more than one (cpp:61-87).
struct FrameAssembler {
    buf: [u8; WAVE_BUFFER_SIZE],
    len: usize,
}

impl FrameAssembler {
    const fn new() -> Self {
        Self {
            buf: [0; WAVE_BUFFER_SIZE],
            len: 0,
        }
    }

    /// Feed freshly-drained bytes; returns the latest completed frame, if any.
    fn push(&mut self, bytes: &[u8]) -> Option<[u8; WAVE_BUFFER_SIZE]> {
        let mut latest: Option<[u8; WAVE_BUFFER_SIZE]> = None;
        let mut src = bytes;
        while !src.is_empty() {
            let need = WAVE_BUFFER_SIZE - self.len;
            let take = need.min(src.len());
            self.buf[self.len..self.len + take].copy_from_slice(&src[..take]);
            self.len += take;
            src = &src[take..];
            if self.len == WAVE_BUFFER_SIZE {
                latest = Some(self.buf);
                self.len = 0;
            }
        }
        latest
    }
}

/// Parameters handed to the worker loop.
pub(crate) struct WorkerParams {
    pub gate: f32,
    pub tick: Duration,
}

/// Run the FFT worker until `shutdown` is set. Publishes a fresh snapshot into
/// `shared` on every tick (so gated silence visibly decays to zero).
pub(crate) fn run(
    mut cons: HeapCons<u8>,
    shared: Arc<ArcSwap<AudioSpectrum>>,
    shutdown: Arc<AtomicBool>,
    params: WorkerParams,
) {
    let fft: Arc<dyn Fft<f32>> = FftPlanner::<f32>::new().plan_fft_forward(WAVE_BUFFER_SIZE);
    let mut assembler = FrameAssembler::new();
    let mut smoother = Smoother::new();
    // Scratch drain buffer; reused every tick (no steady-state alloc here — the
    // only per-tick heap alloc is the published Arc snapshot, off the render
    // thread).
    let mut drain = [0u8; 8192];

    while !shutdown.load(Ordering::Relaxed) {
        // Drain everything currently available, assembling frames; keep the
        // newest completed window.
        let mut latest: Option<[u8; WAVE_BUFFER_SIZE]> = None;
        loop {
            let n = cons.pop_slice(&mut drain);
            if n == 0 {
                break;
            }
            if let Some(frame) = assembler.push(&drain[..n]) {
                latest = Some(frame);
            }
        }

        if let Some(frame) = latest {
            let targets: BandTargets = crate::dsp::analyze_frame(fft.as_ref(), &frame, params.gate);
            smoother.set_targets(targets);
        }

        smoother.tick();
        shared.store(Arc::new(AudioSpectrum::from(&smoother)));

        std::thread::sleep(params.tick);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single push shorter than a frame yields nothing but retains bytes.
    #[test]
    fn assembler_partial_then_complete() {
        let mut a = FrameAssembler::new();
        assert!(a.push(&[7u8; 500]).is_none());
        assert_eq!(a.len, 500);
        let frame = a.push(&[9u8; WAVE_BUFFER_SIZE - 500 + 3]).expect("frame");
        // First 500 bytes are the earlier 7s, rest are 9s.
        assert_eq!(frame[0], 7);
        assert_eq!(frame[499], 7);
        assert_eq!(frame[500], 9);
        // 3 leftover bytes retained for the next frame.
        assert_eq!(a.len, 3);
    }

    /// When a single drain delivers more than one full frame, only the LATEST
    /// complete window survives (cpp:61-87).
    #[test]
    fn assembler_keeps_latest_when_multiple() {
        let mut a = FrameAssembler::new();
        let mut bytes = vec![1u8; WAVE_BUFFER_SIZE]; // frame A
        bytes.extend_from_slice(&[2u8; WAVE_BUFFER_SIZE]); // frame B
        bytes.extend_from_slice(&[3u8; 10]); // partial C
        let frame = a.push(&bytes).expect("frame");
        // Latest complete frame is the all-2s frame B, not frame A.
        assert!(frame.iter().all(|&b| b == 2));
        assert_eq!(a.len, 10);
    }

    /// Empty input produces no frame and no state change (underflow-safe).
    #[test]
    fn assembler_empty_is_noop() {
        let mut a = FrameAssembler::new();
        assert!(a.push(&[]).is_none());
        assert_eq!(a.len, 0);
    }
}
