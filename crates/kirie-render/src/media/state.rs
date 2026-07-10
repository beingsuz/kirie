//! Immutable now-playing snapshot types.
//!
//! These mirror the C++ reference `MediaInfo` / `PlaybackState`
//! (docs/subsystems-misc.md §5, `MediaSource.h:11-26`). A [`MediaState`] is an
//! immutable value published lock-free via `arc-swap` so the render / script
//! thread reads the latest snapshot without ever blocking the D-Bus worker
//! (SPEC §V3 immutable snapshots, §V4 render never blocks).

use std::sync::Arc;

use super::art::AlbumArt;

/// Playback status, integer-compatible with the MPRIS → page contract.
///
/// The discriminants are the exact integers `__wpMediaPlayback` delivers to web
/// wallpapers and SceneScript listeners (docs/subsystems-misc.md §5,
/// `MediaSource.h:11-26`): `Stopped=0`, `Playing=1`, `Paused=2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PlaybackState {
    /// No media loaded / playback stopped.
    #[default]
    Stopped = 0,
    /// Media is actively playing.
    Playing = 1,
    /// Media is loaded but paused.
    Paused = 2,
}

impl PlaybackState {
    /// The MPRIS `PlaybackStatus` string → state mapping
    /// (docs/subsystems-misc.md §5, `DBusMediaSource.cpp` `parsePlaybackStatus`).
    /// Any unrecognized value maps to [`PlaybackState::Stopped`] (V9: never
    /// panics on malformed input).
    #[must_use]
    pub fn from_mpris(status: &str) -> Self {
        match status {
            "Playing" => Self::Playing,
            "Paused" => Self::Paused,
            // "Stopped" and anything else.
            _ => Self::Stopped,
        }
    }

    /// The integer the page/script contract expects (`0`/`1`/`2`).
    #[must_use]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Parsed MPRIS track metadata (the `xesam:*` / `mpris:*` dict, decoded).
///
/// Field sources (docs/subsystems-misc.md §5, `DBusMediaSource.cpp:87-184`):
/// `xesam:title` → [`title`](Self::title), `xesam:artist` (array, first entry)
/// → [`artist`](Self::artist), `xesam:album` → [`album`](Self::album),
/// `mpris:artUrl` → [`art_url`](Self::art_url), `mpris:length` (int64 µs) →
/// [`length_us`](Self::length_us).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrackMetadata {
    /// `xesam:title`. Empty when absent.
    pub title: String,
    /// First entry of the `xesam:artist` string array (some players expose a
    /// bare string; both are accepted). Empty when absent.
    pub artist: String,
    /// `xesam:album`. Empty when absent.
    pub album: String,
    /// `mpris:artUrl` — `file://`, `http(s)://`, `data:` or a bare absolute
    /// path. `None` when absent or emptied.
    pub art_url: Option<String>,
    /// `mpris:length` in **microseconds** (MPRIS native unit). `None` when
    /// absent. Consumers convert to seconds for the page contract.
    pub length_us: Option<i64>,
}

impl TrackMetadata {
    /// True when every field is empty/absent — i.e. no track information.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.title.is_empty()
            && self.artist.is_empty()
            && self.album.is_empty()
            && self.art_url.is_none()
            && self.length_us.is_none()
    }
}

/// An immutable now-playing snapshot.
///
/// [`MediaState::empty`] is the graceful "no player present" value (V9): every
/// field default, `available == false`. The render/script side can always call
/// [`super::MediaSource::latest`] and get a valid snapshot with no panic and no
/// blocking, whether or not any MPRIS player exists.
#[derive(Clone, Debug, Default)]
pub struct MediaState {
    /// Whether an MPRIS player is currently adopted (C++ `available =
    /// currentPlayer.has_value()`, docs/subsystems-misc.md §5). When `false`
    /// all other fields are defaults.
    pub available: bool,
    /// The bus name of the adopted player (e.g.
    /// `org.mpris.MediaPlayer2.spotify`), when one is present.
    pub player: Option<String>,
    /// Current playback status.
    pub playback: PlaybackState,
    /// Decoded track metadata.
    pub metadata: TrackMetadata,
    /// Current playback position in **microseconds** (MPRIS `Position`).
    pub position_us: i64,
    /// Decoded album art (RGBA), when [`TrackMetadata::art_url`] pointed at a
    /// loadable local (`file://` / bare-path) or `data:` image. Remote
    /// (`http(s)://`) art is left as a URL only (not fetched here) — the field
    /// is `None` in that case. Shared so cloning a snapshot is cheap (V4).
    pub art: Option<Arc<AlbumArt>>,
}

impl MediaState {
    /// The empty snapshot: no player, stopped, no metadata. This is what a
    /// caller sees when no session bus / no player is present (V9).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Playback position in seconds (`position_us / 1e6`), the unit the page /
    /// script timeline contract uses (docs/subsystems-misc.md §3.5).
    #[must_use]
    pub fn position_secs(&self) -> f64 {
        self.position_us as f64 / 1_000_000.0
    }

    /// Track duration in seconds (`mpris:length / 1e6`), or `0.0` when unknown.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.metadata.length_us.unwrap_or(0) as f64 / 1_000_000.0
    }
}
