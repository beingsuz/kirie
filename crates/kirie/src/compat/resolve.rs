//! Background value resolution and wallpaper-type classification.
//!
//! [`translate_background`] is the C++ `translateBackground` (doc §3.4): a
//! value containing `/` is a filesystem path used verbatim, otherwise it is a
//! bare Steam Workshop id probed under the four Steam roots. Resolution runs
//! *during parse* (doc §3.4), so a bad id fails before anything else.
//!
//! [`classify`] maps a resolved background to what kirie can actually run in
//! P3 — video (kirie-video), a plain image/gif/.tex file (kirie-render), or a
//! not-yet-supported scene/web wallpaper.

use std::path::{Path, PathBuf};

use kirie_formats::project::{Project, WallpaperType};

use crate::compat::args::{ParseError, WORKSHOP_APP_ID};

/// The four Steam workshop roots probed for a bare id, relative to `$HOME`
/// (doc §3.4, Steam/FileSystem/FileSystem.cpp:16-53), in priority order.
const STEAM_WORKSHOP_ROOTS: [&str; 4] = [
    ".local/share/Steam/steamapps/workshop/content",
    ".steam/steam/steamapps/workshop/content",
    ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/workshop/content",
    "snap/steam/common/.local/share/Steam/steamapps/workshop/content",
];

/// The C++ `translateBackground` (doc §3.4).
///
/// * A value containing `/` is used as a filesystem path verbatim (relative
///   allowed; the existence is not checked here — the C++ likewise defers it).
/// * A value with no `/` is a bare workshop id: probe `$HOME/<root>/431960/<id>`
///   for each root in priority order; the first existing directory wins. None
///   exists → fatal `Cannot find workshop directory ...` (doc §3.4 [observed]).
pub fn translate_background(value: &str) -> Result<String, ParseError> {
    if value.contains('/') {
        return Ok(value.to_owned());
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| fatal("Cannot find home directory, please set the HOME environment variable"))?;
    let home = PathBuf::from(home);
    for root in STEAM_WORKSHOP_ROOTS {
        let candidate = home.join(root).join(WORKSHOP_APP_ID).join(value);
        if candidate.is_dir() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
    }
    Err(fatal(format!(
        "Cannot find workshop directory for steam app {WORKSHOP_APP_ID} and content {value}"
    )))
}

/// A doubled [`ParseError`] (doc §4.7) for a resolution failure.
fn fatal(message: impl Into<String>) -> ParseError {
    ParseError {
        message: message.into(),
        doubled: true,
    }
}

/// What a resolved background is, for run-mode dispatch (task scope: video →
/// kirie-video, image/gif file → kirie-render, scene/web → not yet supported).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wallpaper {
    /// A video wallpaper: the media file to open with kirie-video.
    Video {
        /// Absolute path to the media file (`<dir>/<project.file>` for a
        /// workshop dir, or the file itself for a direct path).
        media: PathBuf,
    },
    /// An image/gif/`.tex` wallpaper: the content file for kirie-render.
    Image {
        /// Path to the image/gif/`.tex` file.
        file: PathBuf,
    },
    /// A scene wallpaper: the workshop item directory (holding `scene.pkg` and
    /// `project.json`) for kirie-render's scene renderer (P4).
    Scene {
        /// Workshop item directory containing `scene.pkg` + `project.json`.
        dir: PathBuf,
    },
    /// A web wallpaper: an HTML/CSS/JS bundle whose entry page is `<dir>/<file>`
    /// (typically `index.html`), rendered by an embedded browser (kirie-web).
    /// Whether it is *runnable* depends on which `web-*` feature this binary was
    /// built with (see `compat/run.rs`); classification is feature-agnostic.
    Web {
        /// Workshop item directory (or the entry page's parent for a direct
        /// `.html` background).
        dir: PathBuf,
        /// The `project.json` `"file"` value: an entry filename (`index.html`)
        /// or, rarely, a bare `http(s)://` URL (docs/format-project-json.md §3.2).
        file: String,
    },
    /// A wallpaper kirie refuses to run. `kind == "application"` is *reference
    /// parity*, not a kirie gap: app wallpapers are Windows `.exe` items driven
    /// by DLL injection with no Linux equivalent, and the C++ engine refuses
    /// them too — `WallpaperParser::parse` throws
    /// `"Application wallpapers are not supported on this platform"`
    /// (WallpaperParser.cpp:22-24). `kind == "unknown"` covers kirie's
    /// direct-file extension fallthrough (the reference only takes workshop
    /// dirs, so it has no equivalent).
    Unsupported {
        /// Which type it resolved to, for the stderr message.
        kind: &'static str,
    },
    /// A published Wallpaper Engine *asset* (an effect/preset, `project.json`
    /// `category == "Asset"`) — not a wallpaper at all, so non-renderable by
    /// design rather than merely unimplemented (docs/corpus.md §7).
    Asset,
}

impl Wallpaper {
    /// A short human phrase for why a non-runnable background cannot be shown,
    /// or `None` if it is runnable. Used for the per-screen run-mode notice.
    #[must_use]
    pub fn unrunnable_reason(&self) -> Option<String> {
        match self {
            Wallpaper::Video { .. } | Wallpaper::Image { .. } | Wallpaper::Scene { .. } => None,
            // Web runnability is feature-dependent; `compat/run.rs` owns the
            // precise per-build message. This default covers the no-web build.
            Wallpaper::Web { .. } => Some(
                "web wallpapers need a web build (rebuild with --features web-cef or --features web-webview)"
                    .to_owned(),
            ),
            // Application items reproduce the reference's exact refusal
            // (WallpaperParser.cpp:22-24); other kinds (kirie's direct-file
            // unknown-extension case) keep the kirie phrasing.
            Wallpaper::Unsupported { kind: "application" } => {
                Some("Application wallpapers are not supported on this platform".to_owned())
            }
            Wallpaper::Unsupported { kind } => {
                Some(format!("{kind} wallpapers are not yet supported by kirie"))
            }
            Wallpaper::Asset => {
                Some("is a Wallpaper Engine asset (effect preset), not a renderable wallpaper".to_owned())
            }
        }
    }
}

/// Steam roots (relative to `$HOME`) whose sibling `common/wallpaper_engine`
/// install holds the shared builtin assets (`genericimage2`, effect shaders,
/// builtin materials) that scene `.pkg`s reference but do not bundle
/// (docs/render-architecture.md §10 asset lookup).
const STEAM_COMMON_ROOTS: [&str; 4] = [
    ".local/share/Steam/steamapps/common/wallpaper_engine/assets",
    ".steam/steam/steamapps/common/wallpaper_engine/assets",
    ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/wallpaper_engine/assets",
    "snap/steam/common/.local/share/Steam/steamapps/common/wallpaper_engine/assets",
];

/// Locate the shared Wallpaper Engine builtin-assets directory, or `None` if it
/// is not installed. `KIRIE_WE_ASSETS` overrides the probe (docs/corpus.md).
///
/// Scenes that reference only self-contained pkg assets resolve without it;
/// scenes using builtin shaders/materials need it or they degrade to their
/// clear color (best-effort, SPEC.md §V9).
#[must_use]
pub fn we_assets_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("KIRIE_WE_ASSETS") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    STEAM_COMMON_ROOTS
        .iter()
        .map(|root| home.join(root))
        .find(|candidate| candidate.is_dir())
}

/// Media/image extensions recognized on a direct-file background (no
/// `project.json`). Video first so `.mp4`/etc. route to kirie-video.
const VIDEO_EXTS: [&str; 6] = ["mp4", "webm", "mkv", "avi", "mov", "m4v"];
const IMAGE_EXTS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "gif", "tex"];

/// Classify a resolved background path into a runnable [`Wallpaper`].
///
/// A directory is treated as a workshop item: its `project.json` decides the
/// type (doc §3.4). A file is classified by extension (kirie's direct-file
/// behavior; the C++ fork only takes workshop dirs). A web item becomes
/// [`Wallpaper::Web`]; an application item or anything unclassifiable is
/// [`Wallpaper::Unsupported`].
pub fn classify(background: &str) -> Result<Wallpaper, ClassifyError> {
    let path = Path::new(background);
    if path.is_dir() {
        return classify_dir(path);
    }
    if path.is_file() {
        return Ok(classify_file(path));
    }
    Err(ClassifyError::NotFound {
        path: path.to_path_buf(),
    })
}

/// Classify a workshop item directory by its `project.json` (doc §3.4).
fn classify_dir(dir: &Path) -> Result<Wallpaper, ClassifyError> {
    let manifest = dir.join("project.json");
    let project = Project::from_path(&manifest).map_err(|source| ClassifyError::Project {
        path: manifest.clone(),
        reason: source.to_string(),
    })?;
    // An `Asset` item (effect/preset) resolves by extension to a Scene, but its
    // main file is an effect manifest, not a scene — it is not a wallpaper and
    // must be reported as non-renderable, not attempted (docs/corpus.md §7).
    if project.is_asset() {
        return Ok(Wallpaper::Asset);
    }
    match project.resolved_type {
        WallpaperType::Video => Ok(Wallpaper::Video {
            media: dir.join(&project.file),
        }),
        WallpaperType::Image => Ok(Wallpaper::Image {
            file: dir.join(&project.file),
        }),
        WallpaperType::Scene => Ok(Wallpaper::Scene {
            dir: dir.to_path_buf(),
        }),
        WallpaperType::Web => Ok(Wallpaper::Web {
            dir: dir.to_path_buf(),
            file: project.file.clone(),
        }),
        WallpaperType::Application => Ok(Wallpaper::Unsupported { kind: "application" }),
    }
}

/// Classify a direct file path by its extension.
fn classify_file(file: &Path) -> Wallpaper {
    let ext = file
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if VIDEO_EXTS.contains(&ext.as_str()) {
        Wallpaper::Video {
            media: file.to_path_buf(),
        }
    } else if IMAGE_EXTS.contains(&ext.as_str()) {
        Wallpaper::Image {
            file: file.to_path_buf(),
        }
    } else if matches!(ext.as_str(), "html" | "htm") {
        // A direct `.html` background is a self-contained web wallpaper: its
        // parent directory is the bundle root and the file itself is the entry.
        Wallpaper::Web {
            dir: file.parent().unwrap_or_else(|| Path::new(".")).to_path_buf(),
            file: file
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        }
    } else {
        Wallpaper::Unsupported { kind: "unknown" }
    }
}

/// Build the entry-page URL for a [`Wallpaper::Web`] item.
///
/// A `file` that already carries an `http`/`https`/`file` scheme (a bare-URL
/// web wallpaper, docs/format-project-json.md §3.2) is used verbatim.
/// Otherwise the entry is the local file `<dir>/<file>`, canonicalized to an
/// absolute path and rendered as a percent-encoded `file://` URL so the page's
/// relative `css`/`js`/`img` references resolve against its own directory
/// (docs/subsystems-misc.md §3.4).
#[must_use]
pub fn web_entry_url(dir: &Path, file: &str) -> String {
    let lower = file.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("file://") {
        return file.to_owned();
    }
    let path = dir.join(file);
    let abs = std::fs::canonicalize(&path).unwrap_or(path);
    file_url(&abs)
}

/// Render an absolute filesystem path as a percent-encoded `file://` URL,
/// leaving `/` as the path separator and RFC 3986 unreserved bytes untouched.
fn file_url(path: &Path) -> String {
    use std::path::Component;

    let mut url = String::from("file://");
    for comp in path.components() {
        match comp {
            Component::Normal(seg) => {
                url.push('/');
                for &b in seg.to_string_lossy().as_bytes() {
                    if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
                        url.push(b as char);
                    } else {
                        url.push('%');
                        url.push(hex_digit(b >> 4));
                        url.push(hex_digit(b & 0x0f));
                    }
                }
            }
            Component::ParentDir => url.push_str("/.."),
            // RootDir/CurDir/Prefix: no own segment (leading `/` is emitted per
            // Normal segment; a Linux absolute path has no Prefix).
            _ => {}
        }
    }
    if url == "file://" {
        url.push('/');
    }
    url
}

/// One hex digit (upper-case) for a 0..=15 nibble.
fn hex_digit(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'A' + (n - 10)) as char,
    }
}

/// Errors classifying a background at run time (SPEC V9: typed, no panic).
#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    /// The background path exists as neither a directory nor a file.
    #[error("background path does not exist: {path}")]
    NotFound {
        /// The missing path.
        path: PathBuf,
    },
    /// The workshop item's `project.json` failed to parse.
    #[error("cannot load {path}: {reason}")]
    Project {
        /// The manifest path.
        path: PathBuf,
        /// The underlying parse error message.
        reason: String,
    },
}
