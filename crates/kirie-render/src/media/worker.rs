//! The MPRIS D-Bus worker thread.
//!
//! Runs a blocking `zbus` session-bus client on its own thread (SPEC §V3 — the
//! render/script side only ever reads immutable snapshots). Each tick it
//! re-detects the active `org.mpris.MediaPlayer2.*` player, reads its
//! `Metadata` / `PlaybackStatus` / `Position`, decodes album art when the art
//! URL changes, and publishes a fresh [`MediaState`] via `arc-swap`
//! (docs/subsystems-misc.md §5, `DBusMediaSource.cpp`).
//!
//! Player selection mirrors the reference `detectPlayer()`
//! (`DBusMediaSource.cpp:259-320`): a `Playing` player wins; otherwise the
//! first `Paused` one; otherwise no player is adopted (`available = false`).
//!
//! Nothing here can take down the process: a bus that fails to connect, a
//! player that vanishes mid-read, or an unreadable property all resolve to the
//! empty snapshot (SPEC §V9). The blocking D-Bus calls never touch the render
//! thread (SPEC §V4).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use zbus::zvariant::OwnedValue;

use super::art::{AlbumArt, load_art};
use super::metadata::parse_metadata;
use super::state::{MediaState, PlaybackState};
use super::{MediaStatus, WorkerParams};

/// MPRIS well-known-name prefix.
const MPRIS_PREFIX: &str = "org.mpris.MediaPlayer2.";
/// The standard MPRIS object path all players expose.
const OBJECT_PATH: &str = "/org/mpris/MediaPlayer2";
/// The player interface carrying `Metadata` / `PlaybackStatus` / `Position`.
const PLAYER_IFACE: &str = "org.mpris.MediaPlayer2.Player";

/// Small cache so album art is decoded only when the art URL actually changes
/// (decoding is the one heavy step; re-running it every tick would waste CPU).
struct ArtCache {
    url: Option<String>,
    art: Option<Arc<AlbumArt>>,
}

impl ArtCache {
    const fn new() -> Self {
        Self { url: None, art: None }
    }

    /// Return the decoded art for `url`, decoding (and caching) on change.
    fn resolve(&mut self, url: Option<&str>) -> Option<Arc<AlbumArt>> {
        if self.url.as_deref() == url {
            return self.art.clone();
        }
        self.url = url.map(str::to_owned);
        self.art = url.and_then(load_art).map(Arc::new);
        self.art.clone()
    }
}

/// Worker entry point. Loops until `shutdown` is set.
pub(super) fn run(
    shared: Arc<ArcSwap<MediaState>>,
    status: Arc<AtomicU8>,
    shutdown: Arc<AtomicBool>,
    params: WorkerParams,
) {
    let conn = match zbus::blocking::Connection::session() {
        Ok(c) => {
            status.store(MediaStatus::Connected.as_u8(), Ordering::Relaxed);
            c
        }
        Err(e) => {
            // No session bus (e.g. headless) → stay empty, never panic (V9).
            tracing::info!(error = %e, "no D-Bus session bus; media state stays empty");
            status.store(MediaStatus::Failed.as_u8(), Ordering::Relaxed);
            shared.store(Arc::new(MediaState::empty()));
            return;
        }
    };

    let mut art_cache = ArtCache::new();

    while !shutdown.load(Ordering::Relaxed) {
        let state = poll_once(&conn, &mut art_cache);
        shared.store(Arc::new(state));
        sleep_interruptible(&shutdown, params.tick);
    }
}

/// One detect → read → build cycle. Returns the empty snapshot when no player
/// is adopted or the bus read fails.
fn poll_once(conn: &zbus::blocking::Connection, art_cache: &mut ArtCache) -> MediaState {
    let Some(player) = detect_player(conn) else {
        return MediaState::empty();
    };

    let Ok(proxy) = player_proxy(conn, &player) else {
        return MediaState::empty();
    };

    let playback = proxy
        .get_property::<String>("PlaybackStatus")
        .map(|s| PlaybackState::from_mpris(&s))
        .unwrap_or_default();

    let metadata = proxy
        .get_property::<HashMap<String, OwnedValue>>("Metadata")
        .map(|d| parse_metadata(&d))
        .unwrap_or_default();

    // Position is optional in MPRIS; a missing/erroring property is 0.
    let position_us = proxy.get_property::<i64>("Position").unwrap_or(0);

    let art = art_cache.resolve(metadata.art_url.as_deref());

    MediaState {
        available: true,
        player: Some(player),
        playback,
        metadata,
        position_us,
        art,
    }
}

/// Build a blocking player proxy at the standard MPRIS object path.
fn player_proxy<'a>(
    conn: &zbus::blocking::Connection,
    player: &str,
) -> zbus::Result<zbus::blocking::Proxy<'a>> {
    zbus::blocking::Proxy::new(conn, player.to_owned(), OBJECT_PATH, PLAYER_IFACE)
}

/// Detect the active player: first `Playing`, else first `Paused`, else `None`
/// (mirrors `DBusMediaSource::detectPlayer`, docs/subsystems-misc.md §5).
fn detect_player(conn: &zbus::blocking::Connection) -> Option<String> {
    let dbus = zbus::blocking::fdo::DBusProxy::new(conn).ok()?;
    let names = dbus.list_names().ok()?;

    let mut first_paused: Option<String> = None;
    for name in names {
        let name = name.as_str();
        if !name.starts_with(MPRIS_PREFIX) {
            continue;
        }
        let Ok(proxy) = player_proxy(conn, name) else {
            continue;
        };
        match proxy.get_property::<String>("PlaybackStatus").as_deref() {
            Ok("Playing") => return Some(name.to_owned()),
            Ok("Paused") if first_paused.is_none() => {
                first_paused = Some(name.to_owned());
            }
            _ => {}
        }
    }
    first_paused
}

/// Sleep for `dur`, waking early (in ≤50 ms slices) if shutdown is requested,
/// so `Drop` joins promptly.
fn sleep_interruptible(shutdown: &AtomicBool, dur: Duration) {
    let slice = Duration::from_millis(50);
    let mut remaining = dur;
    while remaining > Duration::ZERO {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        let step = remaining.min(slice);
        std::thread::sleep(step);
        remaining = remaining.saturating_sub(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn art_cache_decodes_once_per_url() {
        let mut cache = ArtCache::new();
        // Unknown/remote URL → None, but the URL is remembered so a repeat is
        // a cheap no-op.
        assert!(cache.resolve(Some("https://example.com/a.jpg")).is_none());
        assert_eq!(cache.url.as_deref(), Some("https://example.com/a.jpg"));
        // Same URL again: still cached None.
        assert!(cache.resolve(Some("https://example.com/a.jpg")).is_none());
        // Clearing the URL resets the cache.
        assert!(cache.resolve(None).is_none());
        assert_eq!(cache.url, None);
    }
}
