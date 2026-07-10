//! Pure DSP: the exact band-reduction / gate / smoothing math from
//! `docs/subsystems-misc.md` §1.3 (Visualizer capture path). No I/O, no
//! threads — every constant here is a 1:1 port of the C++ reference
//! (`Drivers/Recorders/PulseAudioPlaybackRecorder.cpp`) so it can be unit
//! tested against synthetic input.

use rustfft::num_complex::Complex;

/// One FFT frame = 1024 U8 samples ≈ 23.2 ms at 44100 Hz
/// (`WAVE_BUFFER_SIZE`, PulseAudioPlaybackRecorder.h:8).
pub const WAVE_BUFFER_SIZE: usize = 1024;

/// Capture sample rate (Hz). `PA_SAMPLE_U8`, 1 channel (cpp:107-110).
pub const SAMPLE_RATE: u32 = 44100;

/// Real-FFT bin count for a 1024-point transform: `N/2 + 1` = 513 (cpp:149).
pub const FFT_BINS: usize = WAVE_BUFFER_SIZE / 2 + 1;

/// Per-frame slew rate toward the latest FFT result — the ONLY temporal
/// smoothing (`movetowards(.., .., 0.3f)`, cpp:229-240). No attack/decay
/// asymmetry.
pub const SMOOTH_RATE: f32 = 0.3;

/// Default noise-gate RMS threshold (`gate = 10.0`, cpp:248). Overridable via
/// `WPE_AUDIO_GATE` (`0` disables). This gate is a fork addition kept for
/// behavioral parity.
pub const DEFAULT_GATE: f32 = 10.0;

/// Band counts produced, low→high resolution (`audio16/32/64`).
pub const BANDS_16: usize = 16;
/// See [`BANDS_16`].
pub const BANDS_32: usize = 32;
/// See [`BANDS_16`].
pub const BANDS_64: usize = 64;

/// `movetowards(c, t, d) = t if |t-c| <= d else c + sign(t-c)*d`
/// (cpp:9-15). Slews `current` toward `target` by at most `delta` per call.
#[must_use]
#[inline]
pub fn move_towards(current: f32, target: f32, delta: f32) -> f32 {
    let diff = target - current;
    if diff.abs() <= delta {
        target
    } else {
        current + diff.signum() * delta
    }
}

/// Frequency-dependent gain `boost(x) = 2 - e^((1 - x) - 0.5)` (cpp:157).
/// Bass (x→0) attenuated to ≈0.3513, treble (x→1) boosted to ≈1.3935.
#[must_use]
#[inline]
pub fn boost(x: f32) -> f32 {
    2.0 - ((1.0 - x) - 0.5).exp()
}

/// The three destination band arrays (targets the smoother slews toward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandTargets {
    /// 16-band FFT reduction (`dest16`).
    pub b16: [f32; BANDS_16],
    /// 32-band FFT reduction (`dest32`).
    pub b32: [f32; BANDS_32],
    /// 64-band FFT reduction (`dest64`).
    pub b64: [f32; BANDS_64],
}

impl BandTargets {
    /// All-zero targets (silence / gated / no-audio).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            b16: [0.0; BANDS_16],
            b32: [0.0; BANDS_32],
            b64: [0.0; BANDS_64],
        }
    }
}

/// Normalize a raw U8 sample to `[-1, 1)`: `(u8 - 128) / 128.0` (cpp:276-278).
/// **No window function** is applied (raw rectangular window, cpp:145-146).
#[must_use]
#[inline]
pub fn normalize_sample(sample: u8) -> f32 {
    (f32::from(sample) - 128.0) / 128.0
}

/// RMS of `(sample - 128)` over the 1024 U8 samples used by the noise gate
/// (cpp:248-273). Compared against the gate threshold; below it the frame is
/// treated as silence.
#[must_use]
pub fn gate_rms(samples: &[u8; WAVE_BUFFER_SIZE]) -> f32 {
    let mut acc = 0.0f64;
    for &s in samples.iter() {
        let x = f64::from(s) - 128.0;
        acc += x * x;
    }
    (acc / WAVE_BUFFER_SIZE as f64).sqrt() as f32
}

/// Compute the destination bands from one raw 1024-sample U8 frame, applying
/// the noise gate. Returns all-zero targets when `rms < gate`
/// (`gate > 0`); a `gate <= 0` disables the gate entirely (env `WPE_AUDIO_GATE=0`).
///
/// The FFT is a forward **real** transform (bins 0..=512 of a 1024-point DFT),
/// unnormalized — identical to `kiss_fftr` (cpp:147-150). Only even bins
/// `0,2,..,126` are read (band reduction, cpp:285-302).
#[must_use]
pub fn analyze_frame(
    fft: &dyn rustfft::Fft<f32>,
    samples: &[u8; WAVE_BUFFER_SIZE],
    gate: f32,
) -> BandTargets {
    debug_assert_eq!(fft.len(), WAVE_BUFFER_SIZE);

    // Noise gate (cpp:248-273): below threshold → all zeros, stop.
    if gate > 0.0 && gate_rms(samples) < gate {
        return BandTargets::zero();
    }

    // Normalize input (cpp:276-278). rustfft is a full complex FFT; feeding
    // real samples (imag=0) yields bins identical to kiss_fftr for k=0..=N/2.
    let mut buf: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(normalize_sample(s), 0.0))
        .collect();
    fft.process(&mut buf);

    bands_from_spectrum(&buf[..FFT_BINS])
}

/// Band reduction (cpp:285-302) from the first [`FFT_BINS`] complex bins.
/// Split out so tests can drive it with a synthetic spectrum directly.
///
/// ```text
/// idx  = band * 2                        // even bins 0,2,..,126
/// mag2 = re^2 + im^2                      // squared magnitude
/// f1   = mag2 > 0 ? 0.35 * log10(mag2) : 0
/// dest64[band]      = min(1.0, f1 * boost(band/63))
/// dest32[band >> 1] = min(1.0, f1 * boost(band/31))   // overwritten, not averaged
/// dest16[band >> 2] = min(1.0, f1 * boost(band/15))   // overwritten, not averaged
/// ```
///
/// Quirks preserved: only `min(1,·)` — no lower clamp, so bands go negative
/// when `mag2 < 1`; the 32/16 entries keep the value from the *highest* band
/// index in their group (band ≡ 1 mod 2 / 3 mod 4).
#[must_use]
pub fn bands_from_spectrum(spectrum: &[Complex<f32>]) -> BandTargets {
    let mut out = BandTargets::zero();
    for band in 0..BANDS_64 {
        let idx = band * 2;
        let c = spectrum[idx];
        let mag2 = c.re * c.re + c.im * c.im;
        let f1 = if mag2 > 0.0 { 0.35 * mag2.log10() } else { 0.0 };
        let bf = band as f32;
        out.b64[band] = (f1 * boost(bf / 63.0)).min(1.0);
        out.b32[band >> 1] = (f1 * boost(bf / 31.0)).min(1.0);
        out.b16[band >> 2] = (f1 * boost(bf / 15.0)).min(1.0);
    }
    out
}

/// Displayed band arrays that slew toward [`BandTargets`] at [`SMOOTH_RATE`].
/// Owns the persistent `audioN` state; `tick` advances one frame.
#[derive(Clone, Debug)]
pub struct Smoother {
    /// Current displayed 16-band array (`audio16`).
    pub b16: [f32; BANDS_16],
    /// Current displayed 32-band array (`audio32`).
    pub b32: [f32; BANDS_32],
    /// Current displayed 64-band array (`audio64`).
    pub b64: [f32; BANDS_64],
    targets: BandTargets,
}

impl Default for Smoother {
    fn default() -> Self {
        Self::new()
    }
}

impl Smoother {
    /// A fully-decayed (zero) smoother.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            b16: [0.0; BANDS_16],
            b32: [0.0; BANDS_32],
            b64: [0.0; BANDS_64],
            targets: BandTargets::zero(),
        }
    }

    /// Latch new targets (a fresh FFT frame arrived).
    pub fn set_targets(&mut self, targets: BandTargets) {
        self.targets = targets;
    }

    /// Advance one frame: slew every displayed band toward its target by at
    /// most [`SMOOTH_RATE`]. Runs whether or not a new FFT frame exists
    /// (cpp:229-240), so gated silence decays to zero at 0.3/frame.
    pub fn tick(&mut self) {
        for i in 0..BANDS_16 {
            self.b16[i] = move_towards(self.b16[i], self.targets.b16[i], SMOOTH_RATE);
        }
        for i in 0..BANDS_32 {
            self.b32[i] = move_towards(self.b32[i], self.targets.b32[i], SMOOTH_RATE);
        }
        for i in 0..BANDS_64 {
            self.b64[i] = move_towards(self.b64[i], self.targets.b64[i], SMOOTH_RATE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::FftPlanner;

    /// `move_towards` reproduces the exact clamp-to-delta / snap-on-arrival
    /// behavior (cpp:9-15).
    #[test]
    fn move_towards_exact() {
        // Snaps when within delta.
        assert_eq!(move_towards(0.5, 0.6, 0.3), 0.6);
        assert_eq!(move_towards(0.6, 0.5, 0.3), 0.5);
        // Slews by exactly delta when farther.
        assert_eq!(move_towards(0.0, 1.0, 0.3), 0.3);
        assert!((move_towards(1.0, 0.0, 0.3) - 0.7).abs() < 1e-6);
        // Negative targets (bands can go negative — no lower clamp).
        assert!((move_towards(0.0, -1.0, 0.3) + 0.3).abs() < 1e-6);
    }

    /// `normalize_sample` maps U8 to [-1, 1): (u8 - 128)/128 (cpp:276-278).
    #[test]
    fn normalize_exact() {
        assert_eq!(normalize_sample(128), 0.0);
        assert_eq!(normalize_sample(0), -1.0);
        assert_eq!(normalize_sample(255), 127.0 / 128.0);
    }

    /// `boost(x) = 2 - e^((1-x)-0.5)`, endpoints from the doc (cpp:157-166).
    #[test]
    fn boost_endpoints() {
        assert!((boost(0.0) - (2.0 - 0.5f32.exp())).abs() < 1e-6); // ≈0.3513
        assert!((boost(1.0) - (2.0 - (-0.5f32).exp())).abs() < 1e-6); // ≈1.3935
        assert!((boost(0.0) - 0.3513).abs() < 1e-3);
        assert!((boost(1.0) - 1.3935).abs() < 1e-3);
    }

    /// Gate RMS of a full-scale sine passes; silence is below threshold.
    #[test]
    fn gate_rms_behavior() {
        let silent = [128u8; WAVE_BUFFER_SIZE];
        assert_eq!(gate_rms(&silent), 0.0);
        assert!(gate_rms(&silent) < DEFAULT_GATE);

        let mut loud = [0u8; WAVE_BUFFER_SIZE];
        for (i, s) in loud.iter_mut().enumerate() {
            let v = 128.0 + 100.0 * (std::f32::consts::TAU * 40.0 * i as f32 / 1024.0).sin();
            *s = v.round().clamp(0.0, 255.0) as u8;
        }
        // RMS of a 100-amplitude sine ≈ 70.7, well above the gate.
        assert!(gate_rms(&loud) > DEFAULT_GATE);
    }

    /// `bands_from_spectrum` applies `0.35*log10(mag2)*boost` exactly.
    #[test]
    fn bands_formula_exact() {
        let mut spec = vec![Complex::new(0.0f32, 0.0); FFT_BINS];
        // band 1 → idx 2. Choose magnitude so mag2 == 100 → log10 = 2.
        spec[2] = Complex::new(10.0, 0.0);
        let out = bands_from_spectrum(&spec);
        let expected = 0.35 * 100.0f32.log10() * boost(1.0 / 63.0);
        assert!((out.b64[1] - expected).abs() < 1e-5, "got {}", out.b64[1]);
        // band 0 untouched → zero (mag2 == 0 → f1 == 0).
        assert_eq!(out.b64[0], 0.0);
    }

    /// The 32/16 destinations keep the value from the *highest* band index in
    /// their group (overwritten, not averaged) (cpp:167-169).
    #[test]
    fn bands_overwrite_highest_wins() {
        let mut spec = vec![Complex::new(0.0f32, 0.0); FFT_BINS];
        // Group for b32[0] is bands {0,1}; give band 1 (idx 2) energy only.
        spec[2] = Complex::new(10.0, 0.0);
        let out = bands_from_spectrum(&spec);
        // b32[0] is written last by band 1 (band>>1 == 0) → equals band-1 value
        // under the /31 boost, NOT the band-0 (zero) value.
        let expected = 0.35 * 100.0f32.log10() * boost(1.0 / 31.0);
        assert!((out.b32[0] - expected).abs() < 1e-5, "got {}", out.b32[0]);
    }

    /// A pure sine landing exactly on an even bin concentrates energy in one
    /// band; DC and distant bands stay ~0; silence gates to all zeros.
    #[test]
    fn fft_sine_peaks_at_expected_band() {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(WAVE_BUFFER_SIZE);

        // 40 whole cycles across 1024 samples → energy exactly in bin 40 →
        // band = idx/2 = 20.
        let mut samples = [0u8; WAVE_BUFFER_SIZE];
        for (i, s) in samples.iter_mut().enumerate() {
            let v = 128.0 + 100.0 * (std::f32::consts::TAU * 40.0 * i as f32 / 1024.0).sin();
            *s = v.round().clamp(0.0, 255.0) as u8;
        }
        let out = analyze_frame(fft.as_ref(), &samples, DEFAULT_GATE);
        // Band 20 is strong (clamps toward 1.0); DC band 0 is ~0.
        assert!(out.b64[20] > 0.5, "band 20 = {}", out.b64[20]);
        assert!(out.b64[0] <= 0.0, "DC band 0 = {}", out.b64[0]);
        // b64[20] is the (a) maximum.
        let max = out.b64.iter().cloned().fold(f32::MIN, f32::max);
        assert!((out.b64[20] - max).abs() < 1e-6);

        // Silence → gate zeros everything.
        let silent = [128u8; WAVE_BUFFER_SIZE];
        let zero = analyze_frame(fft.as_ref(), &silent, DEFAULT_GATE);
        assert_eq!(zero, BandTargets::zero());
    }

    /// Gate disabled (`0.0`) processes even near-silent frames without zeroing.
    #[test]
    fn gate_disabled_processes_frame() {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(WAVE_BUFFER_SIZE);
        let mut samples = [128u8; WAVE_BUFFER_SIZE];
        // Tiny signal that would fail the gate but passes when disabled.
        for (i, s) in samples.iter_mut().enumerate() {
            if i % 2 == 0 {
                *s = 129;
            }
        }
        let gated = analyze_frame(fft.as_ref(), &samples, DEFAULT_GATE);
        assert_eq!(gated, BandTargets::zero());
        let ungated = analyze_frame(fft.as_ref(), &samples, 0.0);
        // Nyquist-ish content → some band is non-zero (not all zero).
        assert!(ungated != BandTargets::zero());
    }

    /// The smoother slews toward a latched target by at most 0.3/tick and snaps
    /// on arrival; then decays back to zero when the target clears.
    #[test]
    fn smoother_slew_and_decay() {
        let mut sm = Smoother::new();
        let mut target = BandTargets::zero();
        target.b64[0] = 1.0;
        sm.set_targets(target);
        sm.tick();
        assert!((sm.b64[0] - 0.3).abs() < 1e-6);
        sm.tick();
        assert!((sm.b64[0] - 0.6).abs() < 1e-6);
        sm.tick();
        assert!((sm.b64[0] - 0.9).abs() < 1e-6);
        sm.tick();
        assert_eq!(sm.b64[0], 1.0); // snaps (0.1 <= 0.3)
        // Decay to zero.
        sm.set_targets(BandTargets::zero());
        sm.tick();
        assert!((sm.b64[0] - 0.7).abs() < 1e-6);
    }
}
