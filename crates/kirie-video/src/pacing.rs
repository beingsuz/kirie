//! Frame pacing: loop-aware timestamp mapping and due-frame selection.
//!
//! The decode thread maps each raw stream PTS onto a *monotonic playback
//! timeline* that keeps growing across `loop=inf` restarts
//! (docs/subsystems-misc.md §2.1: `loop=inf`, video loops forever,
//! seamlessly). The render side then only ever deals with monotonically
//! increasing timestamps: it pops every frame that is due at the current
//! clock value and presents the newest one, dropping the rest (SPEC V4:
//! render never waits for decode; late frames are discarded, not shown).
//!
//! Everything in this module is pure state-machine logic so the pacing
//! contract is unit-testable without ffmpeg.

/// Anything carrying a monotonic playback timestamp in seconds.
pub trait Timed {
    /// Monotonic playback timestamp (seconds since playback start,
    /// continuous across file loops).
    fn play_pts(&self) -> f64;
}

/// Maps raw in-file timestamps to a monotonic playback timeline across
/// `loop=inf` restarts.
///
/// On EOF the decode thread seeks back to 0 (docs/subsystems-misc.md §2.1)
/// and calls [`LoopTimeline::wrap`]; subsequent raw timestamps are offset
/// so the timeline never runs backwards. When the container advertises a
/// duration, each loop iteration advances by at least that much so that
/// independently-looping streams of the same file (video and audio are
/// demuxed separately here) stay aligned across iterations.
#[derive(Debug, Clone)]
pub struct LoopTimeline {
    /// Offset added to raw timestamps for the current loop iteration.
    base: f64,
    /// Highest `base + raw + duration` observed this iteration.
    end: f64,
    /// Container-advertised duration of one iteration, if known.
    nominal: Option<f64>,
}

impl LoopTimeline {
    /// New timeline starting at zero. `nominal` is the container duration
    /// in seconds when known (used as the minimum loop advance).
    #[must_use]
    pub fn new(nominal: Option<f64>) -> Self {
        Self {
            base: 0.0,
            end: 0.0,
            nominal: nominal.filter(|d| d.is_finite() && *d > 0.0),
        }
    }

    /// Map a raw in-file timestamp (seconds) with the given frame duration
    /// onto the monotonic timeline.
    pub fn map(&mut self, raw_pts: f64, duration: f64) -> f64 {
        let play = self.base + raw_pts;
        let dur = if duration.is_finite() && duration > 0.0 {
            duration
        } else {
            0.0
        };
        self.end = self.end.max(play + dur);
        play
    }

    /// Advance to the next loop iteration (called after the EOF seek-to-0,
    /// docs/subsystems-misc.md §2.1 `loop` = `inf`).
    pub fn wrap(&mut self) {
        let mut next = self.end;
        if let Some(nominal) = self.nominal {
            next = next.max(self.base + nominal);
        }
        self.base = next;
        self.end = next;
    }
}

/// Presented/dropped counters for one playback session.
#[derive(Debug, Clone, Copy, Default)]
pub struct PacerStats {
    /// Frames handed out for presentation.
    pub presented: u64,
    /// Frames that became due but were superseded before presentation
    /// (late frames — dropped, docs/subsystems-misc.md §2.1 pacing intent).
    pub dropped: u64,
}

/// Selects the frame to present at a given clock value.
///
/// Holds at most one read-ahead frame that is not yet due, so the caller's
/// queue never has to be peeked. `select` never blocks (SPEC V4) and does
/// no heap allocation (SPEC V5).
#[derive(Debug)]
pub struct Pacer<T> {
    pending: Option<T>,
    stats: PacerStats,
}

impl<T: Timed> Pacer<T> {
    /// Empty pacer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: None,
            stats: PacerStats::default(),
        }
    }

    /// Counters accumulated so far.
    #[must_use]
    pub fn stats(&self) -> PacerStats {
        self.stats
    }

    /// Return the newest frame due at `now`, pulling frames from `pull`
    /// (non-blocking source). Every older due frame is passed to `recycle`
    /// and counted as dropped. Returns `None` when no new frame is due yet
    /// (the caller keeps showing the previous frame).
    pub fn select(
        &mut self,
        now: f64,
        mut pull: impl FnMut() -> Option<T>,
        mut recycle: impl FnMut(T),
    ) -> Option<T> {
        let mut due: Option<T> = None;
        loop {
            let candidate = match self.pending.take() {
                Some(frame) => frame,
                None => match pull() {
                    Some(frame) => frame,
                    None => break,
                },
            };
            if candidate.play_pts() <= now {
                if let Some(superseded) = due.replace(candidate) {
                    self.stats.dropped += 1;
                    recycle(superseded);
                }
            } else {
                self.pending = Some(candidate);
                break;
            }
        }
        if due.is_some() {
            self.stats.presented += 1;
        }
        due
    }
}

impl<T: Timed> Default for Pacer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{LoopTimeline, Pacer, Timed};

    #[derive(Debug, PartialEq)]
    struct F(f64);

    impl Timed for F {
        fn play_pts(&self) -> f64 {
            self.0
        }
    }

    fn pull_from(frames: &mut Vec<F>) -> impl FnMut() -> Option<F> {
        frames.reverse();
        let mut v = std::mem::take(frames);
        move || v.pop()
    }

    #[test]
    fn selects_newest_due_frame_and_drops_older_ones() {
        let mut pacer = Pacer::new();
        let mut dropped = Vec::new();
        let mut queue = vec![F(0.0), F(0.1), F(0.2), F(0.3)];
        let picked = pacer.select(0.25, pull_from(&mut queue), |f| dropped.push(f.0));
        assert_eq!(picked, Some(F(0.2)));
        assert_eq!(dropped, vec![0.0, 0.1]);
        assert_eq!(pacer.stats().dropped, 2);
        assert_eq!(pacer.stats().presented, 1);
    }

    #[test]
    fn future_frame_is_held_not_dropped() {
        let mut pacer = Pacer::new();
        let mut queue = vec![F(1.0)];
        assert_eq!(pacer.select(0.5, pull_from(&mut queue), |_| {}), None);
        // The held frame comes out once due, without pulling again.
        assert_eq!(pacer.select(1.0, || None, |_| {}), Some(F(1.0)));
        assert_eq!(pacer.stats().dropped, 0);
        assert_eq!(pacer.stats().presented, 1);
    }

    #[test]
    fn empty_source_yields_none() {
        let mut pacer: Pacer<F> = Pacer::new();
        assert_eq!(pacer.select(10.0, || None, |_| {}), None);
        assert_eq!(pacer.stats().presented, 0);
    }

    #[test]
    fn exactly_due_frame_is_presented() {
        let mut pacer = Pacer::new();
        let mut queue = vec![F(0.5)];
        assert_eq!(pacer.select(0.5, pull_from(&mut queue), |_| {}), Some(F(0.5)));
    }

    #[test]
    fn timeline_is_monotonic_across_loop_wrap() {
        // 3 frames at 25fps, then EOF -> seek 0 (mpv loop=inf,
        // docs/subsystems-misc.md §2.1).
        let mut tl = LoopTimeline::new(None);
        assert_eq!(tl.map(0.00, 0.04), 0.00);
        assert_eq!(tl.map(0.04, 0.04), 0.04);
        assert_eq!(tl.map(0.08, 0.04), 0.08);
        tl.wrap();
        // Second iteration continues right after the last frame's end.
        let second = tl.map(0.00, 0.04);
        assert!(
            (second - 0.12).abs() < 1e-9,
            "second iteration starts at {second}"
        );
        assert!(tl.map(0.04, 0.04) > second);
    }

    #[test]
    fn timeline_wrap_uses_nominal_duration_when_longer() {
        // Container says 1.0s but the last decoded frame ends at 0.96s
        // (e.g. audio stream slightly shorter than the container): both
        // demuxers wrap by the same nominal amount so streams stay aligned.
        let mut tl = LoopTimeline::new(Some(1.0));
        tl.map(0.92, 0.04);
        tl.wrap();
        assert_eq!(tl.map(0.0, 0.04), 1.0);
    }

    #[test]
    fn timeline_wrap_uses_observed_end_when_past_nominal() {
        let mut tl = LoopTimeline::new(Some(1.0));
        tl.map(1.06, 0.04);
        tl.wrap();
        assert!((tl.map(0.0, 0.04) - 1.1).abs() < 1e-9);
    }

    #[test]
    fn timeline_ignores_bogus_durations() {
        let mut tl = LoopTimeline::new(Some(f64::NAN));
        tl.map(0.5, f64::INFINITY);
        tl.wrap();
        assert_eq!(tl.map(0.0, 0.04), 0.5);
    }
}
