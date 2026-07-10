//! Animation frame scheduling — the reference's frametime walk
//! (docs/format-tex.md §8.1).

/// Playback schedule over a frame table: pure timing, no pixels.
///
/// Reference semantics (docs/format-tex.md §8.1, `CRenderable.cpp:28-37`,
/// `CPass.cpp:347-378`):
///
/// 1. `total = Σ frametime` (seconds).
/// 2. `t = fmod(renderTimeSeconds, total)` — render time is an *unscaled*
///    wall-clock seconds counter, not the playback-speed-scaled `g_Time`
///    (docs/render-architecture.md §8.2 "Animated texture frame selection").
/// 3. Walk the frame list in file order subtracting each frametime; the
///    first frame that drives `t ≤ 0` is current. A frametime of 0 displays
///    only if it is that first crossing.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSchedule {
    durations: Vec<f32>,
    total: f64,
}

impl FrameSchedule {
    /// Build a schedule from per-frame durations in seconds
    /// (docs/format-tex.md §8: the `frametime` fields in file order).
    #[must_use]
    pub fn new(durations: Vec<f32>) -> Self {
        // f64 accumulator so long tables don't drift; matches the summed
        // "spritesheet duration" of the reference (§8.2).
        let total = durations.iter().map(|&d| f64::from(d)).sum::<f64>();
        Self { durations, total }
    }

    /// Per-frame durations, file order (docs/format-tex.md §8).
    #[must_use]
    pub fn durations(&self) -> &[f32] {
        &self.durations
    }

    /// `Σ frametime` in seconds (docs/format-tex.md §8.1 step 1).
    #[must_use]
    pub fn total_seconds(&self) -> f64 {
        self.total
    }

    /// Whether playback ever leaves frame 0. Single-frame tables are
    /// static, and so are tables whose total duration is ≤ 0 — the
    /// reference would `fmod(t, 0)` into NaN on those, which is malformed
    /// input we refuse to animate instead (SPEC §V9).
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.durations.len() > 1 && self.total > 0.0
    }

    /// Index of the frame displayed at `elapsed` wall-clock seconds since
    /// playback start (docs/format-tex.md §8.1 steps 2-3).
    #[must_use]
    pub fn frame_at(&self, elapsed: f64) -> usize {
        if !self.is_animated() {
            return 0;
        }
        let mut t = elapsed.rem_euclid(self.total);
        for (index, &duration) in self.durations.iter().enumerate() {
            t -= f64::from(duration);
            if t <= 0.0 {
                return index;
            }
        }
        // Float slop past the last boundary wraps to the final frame.
        self.durations.len() - 1
    }

    /// Seconds until the displayed frame next changes, or `None` for static
    /// content (the caller needs no further frames — SPEC §V6 scheduling
    /// hint). `Some(0.0)` means the boundary is exactly now.
    #[must_use]
    pub fn time_until_change(&self, elapsed: f64) -> Option<f64> {
        if !self.is_animated() {
            return None;
        }
        let t = elapsed.rem_euclid(self.total);
        let mut acc = 0.0f64;
        for &duration in &self.durations {
            acc += f64::from(duration);
            // Same boundary rule as `frame_at`: the current frame is the
            // first whose cumulative time reaches t (§8.1 step 3).
            if t <= acc {
                return Some(acc - t);
            }
        }
        Some((self.total - t).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_table_walks_in_file_order() {
        // §8.1 walk over [0.1, 0.2, 0.3], total 0.6.
        let s = FrameSchedule::new(vec![0.1, 0.2, 0.3]);
        assert!(s.is_animated());
        assert_eq!(s.frame_at(0.0), 0);
        assert_eq!(s.frame_at(0.05), 0);
        // Boundary: t == Σ so far still selects the earlier frame (t ≤ 0
        // after subtraction, §8.1 step 3).
        assert_eq!(s.frame_at(f64::from(0.1f32)), 0);
        assert_eq!(s.frame_at(0.15), 1);
        assert_eq!(s.frame_at(0.31), 2);
        assert_eq!(s.frame_at(0.59), 2);
    }

    #[test]
    fn playback_wraps_with_fmod() {
        // §8.1 step 2: t = fmod(renderTime, total).
        let s = FrameSchedule::new(vec![0.25, 0.25]);
        assert_eq!(s.total_seconds(), 0.5);
        assert_eq!(s.frame_at(0.1), 0);
        assert_eq!(s.frame_at(0.3), 1);
        assert_eq!(s.frame_at(0.6), 0); // 0.6 mod 0.5 = 0.1
        assert_eq!(s.frame_at(10.3), 1);
        assert_eq!(s.frame_at(1e6 + 0.3), 1);
    }

    #[test]
    fn zero_frametime_displays_only_as_first_crossing() {
        // docs/format-tex.md §8.1: "A frametime of 0 therefore displays
        // only if it is that first crossing."
        let leading_zero = FrameSchedule::new(vec![0.0, 0.1]);
        assert_eq!(leading_zero.frame_at(0.0), 0); // t=0 → 0-duration frame is the first crossing
        assert_eq!(leading_zero.frame_at(0.05), 1);

        let middle_zero = FrameSchedule::new(vec![0.1, 0.0, 0.2]);
        assert_eq!(middle_zero.frame_at(0.05), 0);
        // At exactly the 0.1 boundary frame 0 still wins (t ≤ 0), so the
        // zero-duration frame 1 is unreachable here.
        assert_eq!(middle_zero.frame_at(f64::from(0.1f32)), 0);
        assert_eq!(middle_zero.frame_at(0.15), 2);
    }

    #[test]
    fn static_and_malformed_tables_never_animate() {
        // Single frame, empty table, and all-zero durations (fmod-by-zero
        // in the reference) are all static frame 0 (SPEC §V9).
        for s in [
            FrameSchedule::new(vec![1.0]),
            FrameSchedule::new(vec![]),
            FrameSchedule::new(vec![0.0, 0.0, 0.0]),
        ] {
            assert!(!s.is_animated());
            assert_eq!(s.frame_at(0.0), 0);
            assert_eq!(s.frame_at(123.4), 0);
            assert_eq!(s.time_until_change(5.0), None);
        }
    }

    #[test]
    fn uniform_39_frame_table_matches_atlas_sample() {
        // The docs/format-tex.md §8.1 real sample: 39 frames, each 1/39 s,
        // total exactly 1 s.
        let dt = 1.0f32 / 39.0;
        let s = FrameSchedule::new(vec![dt; 39]);
        assert!(s.is_animated());
        assert!((s.total_seconds() - 1.0).abs() < 1e-5);
        // Midpoint of every slot selects that slot.
        for k in 0..39usize {
            let midpoint = (k as f64 + 0.5) * f64::from(dt);
            assert_eq!(s.frame_at(midpoint), k, "midpoint of slot {k}");
        }
        // And it wraps around after the full loop.
        assert_eq!(s.frame_at(s.total_seconds() + 0.5 * f64::from(dt)), 0);
    }

    #[test]
    fn time_until_change_counts_down_to_the_boundary() {
        let s = FrameSchedule::new(vec![0.1, 0.2, 0.3]);
        let eps = 1e-9;
        assert!((s.time_until_change(0.0).unwrap() - 0.1).abs() < 1e-6);
        assert!((s.time_until_change(0.05).unwrap() - 0.05).abs() < 1e-6);
        // Inside frame 1 (0.1..0.3): boundary at Σ = 0.1+0.2.
        assert!((s.time_until_change(0.15).unwrap() - 0.15).abs() < 1e-6);
        // Wrapped time.
        assert!((s.time_until_change(0.6 + 0.05).unwrap() - 0.05).abs() < 1e-6);
        assert!(s.time_until_change(0.1 + eps).unwrap() < 0.2);
    }
}
