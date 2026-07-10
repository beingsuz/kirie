//! Playback clocks.
//!
//! Two clock sources drive frame selection, mirroring the mpv behavior
//! contract (docs/subsystems-misc.md §2.1):
//!
//! * **audio master** — when the file has an audio stream, video follows
//!   the audio device's consumption position (A/V sync the way mpv does it
//!   by default);
//! * **wall clock × speed** — when there is no audio, media time advances
//!   with real time scaled by the playback speed.
//!
//! Both support live `pause` (freeze, keep state) and `speed` changes
//! (docs/subsystems-misc.md §2.1 `pause`/`speed` properties; speed ≤ 0 is
//! coerced to 1.0, GLPlayer.cpp:107-113 via §2.1).
//!
//! Cross-thread rule (SPEC V3): the audio side publishes immutable
//! [`ProducerSnap`]/[`ConsumerSnap`] snapshots through `triple_buffer`;
//! nothing here shares mutable state between threads.

use std::time::Instant;

/// Wall-clock playback time with pause and speed, used when the video has
/// no audio stream (docs/subsystems-misc.md §2.1: `speed` multiplies the
/// playback rate).
#[derive(Debug, Clone)]
pub(crate) struct WallClock {
    /// Media seconds at `anchor`.
    base_media: f64,
    /// Wall instant the current segment started.
    anchor: Instant,
    /// Playback rate multiplier (> 0).
    speed: f64,
    paused: bool,
}

impl WallClock {
    /// Clock starting at media time 0 from `now`.
    pub fn new(now: Instant, paused: bool) -> Self {
        Self {
            base_media: 0.0,
            anchor: now,
            speed: 1.0,
            paused,
        }
    }

    /// Media seconds at wall instant `now`.
    pub fn now(&self, now: Instant) -> f64 {
        if self.paused {
            self.base_media
        } else {
            self.base_media + now.saturating_duration_since(self.anchor).as_secs_f64() * self.speed
        }
    }

    /// Freeze/unfreeze media time (docs/subsystems-misc.md §2.1 `pause`:
    /// freeze frame, keep state).
    pub fn set_paused(&mut self, paused: bool, now: Instant) {
        if self.paused == paused {
            return;
        }
        self.base_media = self.now(now);
        self.anchor = now;
        self.paused = paused;
    }

    /// Change the playback rate. Values ≤ 0 are coerced to 1.0
    /// (docs/subsystems-misc.md §2.1, GLPlayer.cpp:107-113).
    pub fn set_speed(&mut self, speed: f64, now: Instant) {
        self.base_media = self.now(now);
        self.anchor = now;
        self.speed = if speed > 0.0 && speed.is_finite() {
            speed
        } else {
            1.0
        };
    }
}

/// Snapshot published by the audio *decode* thread after each ring-buffer
/// push (SPEC V3: immutable snapshot via triple_buffer).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProducerSnap {
    /// Total device frames (sample groups) pushed into the ring so far.
    pub pushed: u64,
    /// Monotonic playback time (seconds) of the *end* of the pushed data.
    pub head: f64,
    /// Playback rate the pushed data was resampled for.
    pub speed: f64,
}

impl Default for ProducerSnap {
    fn default() -> Self {
        Self {
            pushed: 0,
            head: 0.0,
            speed: 1.0,
        }
    }
}

/// Snapshot published by the audio *device callback* after each buffer
/// fill (SPEC V3).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConsumerSnap {
    /// Total device frames consumed from the ring so far.
    pub consumed: u64,
    /// Wall instant this snapshot was taken.
    pub at: Instant,
    /// Whether playback was paused during this callback.
    pub paused: bool,
}

impl ConsumerSnap {
    /// Initial snapshot (nothing consumed yet).
    pub fn initial(now: Instant) -> Self {
        Self {
            consumed: 0,
            at: now,
            paused: false,
        }
    }
}

/// Current playback position of the audio-mastered clock, in monotonic
/// playback seconds.
///
/// `position = head − buffered·speed/rate`, extrapolated by the wall time
/// since the consumer snapshot while playing, clamped to `[0, head]` so a
/// still-priming (or underrun) ring never runs the clock ahead of decoded
/// data. Ring contents produced just before a speed change are accounted
/// at the *new* speed — the error is bounded by the ring length and decays
/// as the ring turns over (documented approximation; the mpv contract does
/// not constrain sub-ring-latency speed transitions,
/// docs/subsystems-misc.md §2.1).
pub(crate) fn audio_position(
    prod: &ProducerSnap,
    cons: &ConsumerSnap,
    sample_rate: u32,
    now: Instant,
) -> f64 {
    if sample_rate == 0 {
        return 0.0;
    }
    let buffered = prod.pushed.saturating_sub(cons.consumed) as f64 / f64::from(sample_rate) * prod.speed;
    let mut pos = prod.head - buffered;
    if !cons.paused {
        pos += now.saturating_duration_since(cons.at).as_secs_f64() * prod.speed;
    }
    pos.clamp(0.0, prod.head.max(0.0))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{ConsumerSnap, ProducerSnap, WallClock, audio_position};

    #[test]
    fn wall_clock_advances_with_time() {
        let t0 = Instant::now();
        let clock = WallClock::new(t0, false);
        assert_eq!(clock.now(t0), 0.0);
        assert!((clock.now(t0 + Duration::from_secs(2)) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn wall_clock_pause_freezes_and_resume_continues() {
        let t0 = Instant::now();
        let mut clock = WallClock::new(t0, false);
        let t1 = t0 + Duration::from_secs(1);
        clock.set_paused(true, t1);
        // Frozen while paused (docs/subsystems-misc.md §2.1 pause).
        assert!((clock.now(t1 + Duration::from_secs(5)) - 1.0).abs() < 1e-9);
        let t2 = t1 + Duration::from_secs(5);
        clock.set_paused(false, t2);
        assert!((clock.now(t2 + Duration::from_secs(1)) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn wall_clock_starts_paused_when_asked() {
        let t0 = Instant::now();
        let clock = WallClock::new(t0, true);
        assert_eq!(clock.now(t0 + Duration::from_secs(3)), 0.0);
    }

    #[test]
    fn wall_clock_speed_scales_time() {
        let t0 = Instant::now();
        let mut clock = WallClock::new(t0, false);
        let t1 = t0 + Duration::from_secs(1);
        clock.set_speed(2.0, t1);
        assert!((clock.now(t1 + Duration::from_secs(2)) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn wall_clock_coerces_nonpositive_speed_to_one() {
        // docs/subsystems-misc.md §2.1: values <= 0 are coerced to 1.0.
        let t0 = Instant::now();
        let mut clock = WallClock::new(t0, false);
        clock.set_speed(0.0, t0);
        assert!((clock.now(t0 + Duration::from_secs(1)) - 1.0).abs() < 1e-9);
        clock.set_speed(-3.0, t0);
        assert!((clock.now(t0 + Duration::from_secs(1)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn audio_position_accounts_for_ring_backlog() {
        let now = Instant::now();
        let prod = ProducerSnap {
            pushed: 48_000,
            head: 2.0,
            speed: 1.0,
        };
        let cons = ConsumerSnap {
            consumed: 24_000,
            at: now,
            paused: true,
        };
        // 0.5s of data still in the ring -> position is 0.5s behind head.
        assert!((audio_position(&prod, &cons, 48_000, now) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn audio_position_extrapolates_while_playing() {
        let at = Instant::now();
        let prod = ProducerSnap {
            pushed: 48_000,
            head: 2.0,
            speed: 1.0,
        };
        let cons = ConsumerSnap {
            consumed: 24_000,
            at,
            paused: false,
        };
        let pos = audio_position(&prod, &cons, 48_000, at + Duration::from_millis(100));
        assert!((pos - 1.6).abs() < 1e-9);
    }

    #[test]
    fn audio_position_clamped_to_decoded_head() {
        // Priming: nothing pushed yet, callback running -> position must
        // stay at 0, not extrapolate into undecoded time.
        let at = Instant::now();
        let prod = ProducerSnap::default();
        let cons = ConsumerSnap {
            consumed: 0,
            at,
            paused: false,
        };
        assert_eq!(
            audio_position(&prod, &cons, 48_000, at + Duration::from_secs(1)),
            0.0
        );
    }

    #[test]
    fn audio_position_zero_rate_is_safe() {
        let now = Instant::now();
        assert_eq!(
            audio_position(&ProducerSnap::default(), &ConsumerSnap::initial(now), 0, now),
            0.0
        );
    }
}
