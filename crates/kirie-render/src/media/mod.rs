//! MPRIS media integration (SPEC §T24, docs/subsystems-misc.md §5).
//!
//! Media-reactive wallpapers want the current now-playing track: title/artist/
//! album, playback state, timeline position, and cover art. The C++ reference
//! sources this over D-Bus (`DBusMediaSource`, session bus) and hands it to web
//! wallpapers (the `wallpaperRegisterMedia*Listener` bridge, §3.5) and to the
//! `$mediaThumbnail` virtual asset for scene shaders (§5).
//!
//! This module is the Rust port of that source:
//!
//! ```text
//!   session D-Bus (org.mpris.MediaPlayer2.*)        MPRIS worker thread
//!        │  detect player → Metadata / PlaybackStatus / Position
//!        │  decode album art (file:// | bare path | data:)
//!        ▼
//!   immutable MediaState  ──latest()──►  render / script (never blocks, V4)
//!        via arc-swap (V3 immutable snapshots)
//! ```
//!
//! Design guarantees:
//!
//! * **§V1** — no globals; all state is owned by a [`MediaSource`] handle.
//! * **§V3** — the D-Bus worker is the sole writer; consumers read immutable
//!   [`MediaState`] snapshots through `arc-swap`.
//! * **§V4** — [`MediaSource::latest`] is a lock-free `arc-swap` load; the
//!   blocking D-Bus calls live entirely on the worker thread.
//! * **§V9** — no session bus, no player, a vanished player, or malformed
//!   metadata all resolve to the empty snapshot with no panic.
//! * **§V2** — the crate is `#![forbid(unsafe_code)]`; this module adds no
//!   `unsafe`.
//!
//! ```no_run
//! use kirie_render::media::{MediaSource, MediaConfig};
//!
//! let media = MediaSource::start(MediaConfig::default());
//! let state = media.latest();
//! if state.available {
//!     println!("{} — {}", state.metadata.artist, state.metadata.title);
//! }
//! ```

mod art;
mod metadata;
mod state;
mod worker;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use arc_swap::ArcSwap;

pub use art::{AlbumArt, MediaPlaybackEvent, load_art};
pub use metadata::parse_metadata;
pub use state::{MediaState, PlaybackState, TrackMetadata};

/// Coarse worker state, published lock-free for callers/tests to poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaStatus {
    /// Media integration disabled — worker never spawned, state stays empty.
    Disabled,
    /// Worker spawned; the session bus connection is not yet confirmed.
    Starting,
    /// Connected to the session bus and polling for players.
    Connected,
    /// The session bus could not be reached (headless / no D-Bus). State is
    /// empty; the handle is still valid (V9).
    Failed,
}

impl MediaStatus {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Disabled,
            1 => Self::Starting,
            2 => Self::Connected,
            _ => Self::Failed,
        }
    }
    fn as_u8(self) -> u8 {
        match self {
            Self::Disabled => 0,
            Self::Starting => 1,
            Self::Connected => 2,
            Self::Failed => 3,
        }
    }
}

/// Configuration for [`MediaSource::start`].
#[derive(Clone, Debug)]
pub struct MediaConfig {
    /// Whether to spawn the D-Bus worker at all. `false` yields an always-empty
    /// handle (no thread, no bus connection).
    pub enabled: bool,
    /// Poll cadence — how often the worker re-detects the player and re-reads
    /// metadata/position. Defaults to 1 s; the C++ reference re-fetches every
    /// 2 s (`update` interval, docs/subsystems-misc.md §5) but a 1 s tick keeps
    /// the timeline position fresher.
    pub tick: Duration,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tick: Duration::from_secs(1),
        }
    }
}

impl MediaConfig {
    /// The disabled configuration: always-empty state, no worker thread.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Set the poll cadence.
    #[must_use]
    pub fn with_tick(mut self, tick: Duration) -> Self {
        self.tick = tick;
        self
    }
}

/// Parameters handed to the worker loop.
struct WorkerParams {
    tick: Duration,
}

/// Live handle to the MPRIS media pipeline.
///
/// Holds the latest published [`MediaState`] and owns the D-Bus worker thread
/// (joined on drop). Cheap to keep around; reading is lock-free (§V4).
pub struct MediaSource {
    shared: Arc<ArcSwap<MediaState>>,
    status: Arc<AtomicU8>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl MediaSource {
    /// Start the pipeline. Never fails: a disabled config or an unreachable
    /// session bus both yield a valid handle whose state stays empty (§V9).
    /// Returns immediately — the D-Bus connection is established off-thread.
    #[must_use]
    pub fn start(config: MediaConfig) -> Self {
        let shared = Arc::new(ArcSwap::from_pointee(MediaState::empty()));
        let shutdown = Arc::new(AtomicBool::new(false));

        if !config.enabled {
            return Self {
                shared,
                status: Arc::new(AtomicU8::new(MediaStatus::Disabled.as_u8())),
                shutdown,
                worker: None,
            };
        }

        let status = Arc::new(AtomicU8::new(MediaStatus::Starting.as_u8()));
        let worker = {
            let shared = shared.clone();
            let status = status.clone();
            let shutdown = shutdown.clone();
            let params = WorkerParams { tick: config.tick };
            Some(
                std::thread::Builder::new()
                    .name("kirie-mpris".into())
                    .spawn(move || {
                        worker::run(shared, status, shutdown, params);
                    })
                    .expect("spawn mpris worker"),
            )
        };

        Self {
            shared,
            status,
            shutdown,
            worker,
        }
    }

    /// A disabled handle (always-empty) — convenience for the no-media path.
    #[must_use]
    pub fn disabled() -> Self {
        Self::start(MediaConfig::disabled())
    }

    /// The latest published snapshot. Lock-free, never blocks (§V4). Always a
    /// valid [`MediaState`] — the empty snapshot when no player is present.
    #[must_use]
    pub fn latest(&self) -> Arc<MediaState> {
        self.shared.load_full()
    }

    /// The latest snapshot projected into the script/web-facing
    /// [`MediaPlaybackEvent`] (seconds-based timeline + derived palette).
    #[must_use]
    pub fn event(&self) -> MediaPlaybackEvent {
        MediaPlaybackEvent::from_state(&self.latest())
    }

    /// Current coarse worker state.
    #[must_use]
    pub fn status(&self) -> MediaStatus {
        MediaStatus::from_u8(self.status.load(Ordering::Relaxed))
    }
}

impl Drop for MediaSource {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_source_is_empty_and_never_spawns() {
        let media = MediaSource::disabled();
        assert_eq!(media.status(), MediaStatus::Disabled);
        let state = media.latest();
        assert!(!state.available);
        assert!(state.metadata.is_empty());
        // Event projection of the empty state.
        let ev = media.event();
        assert!(!ev.available);
        assert_eq!(ev.state, PlaybackState::Stopped.as_i32());
        assert!(ev.primary_color.is_none());
    }

    #[test]
    fn config_builders() {
        let c = MediaConfig::default().with_tick(Duration::from_millis(250));
        assert!(c.enabled);
        assert_eq!(c.tick, Duration::from_millis(250));
        assert!(!MediaConfig::disabled().enabled);
    }

    /// Live-gated: connect to the real session bus and read whatever player is
    /// present (or none) without panicking. Opt in with `KIRIE_MPRIS_LIVE=1`.
    #[test]
    fn live_session_bus_no_panic() {
        if std::env::var("KIRIE_MPRIS_LIVE").as_deref() != Ok("1") {
            eprintln!("skipping live MPRIS test (set KIRIE_MPRIS_LIVE=1 to run)");
            return;
        }
        let media = MediaSource::start(MediaConfig::default().with_tick(Duration::from_millis(200)));
        // Give the worker a moment to connect and poll once.
        std::thread::sleep(Duration::from_millis(600));
        let state = media.latest();
        // Whatever the result, it must be a coherent, panic-free snapshot.
        eprintln!(
            "live media: status={:?} available={} player={:?} playback={:?} title={:?} artist={:?} pos={:.2}s/{:.2}s art={:?}",
            media.status(),
            state.available,
            state.player,
            state.playback,
            state.metadata.title,
            state.metadata.artist,
            state.position_secs(),
            state.duration_secs(),
            state.art.as_ref().map(|a| (a.width, a.height)),
        );
        assert!(matches!(
            media.status(),
            MediaStatus::Connected | MediaStatus::Failed
        ));
        if !state.available {
            assert!(state.metadata.is_empty());
        }
        // The event projection must also be panic-free.
        let _ = media.event();
    }
}
