//! Backend selection facade.
//!
//! [`Platform`] is a thin dispatcher over the two presentation backends —
//! [`WaylandPlatform`] (layer-shell, docs/render-architecture.md §2.3) and
//! [`X11Platform`] (root-window/desktop, docs/render-architecture.md §2.2) —
//! so callers (`kirie`, the examples, the tests) get one type with one
//! `connect`/`run`/`output_count` surface regardless of session type
//! (SPEC T24). The backend is chosen by the environment, mirroring the C++
//! driver dispatch which picks the GLFW/X11 or Wayland-EGL video driver at
//! startup (docs/render-architecture.md §2.1).

use std::time::Duration;

use crate::error::PlatformError;
use crate::platform::WaylandPlatform;
use crate::renderer::RendererFactory;
use crate::x11::{X11Mode, X11Platform};

/// Presentation options shared across backends.
///
/// Threaded into [`Platform::connect_with`] so the compat CLI can make kirie a
/// drop-in for the wallpaper daemon (`~/.config/hypr/wallpaper-daemon`):
///
/// - `layer_namespace` sets the wlr-layer-shell surface namespace. The
///   daemon's watchdog (`wallpaperengine.sh`, `engine_layer_ok()`) decides a
///   monitor still has a live wallpaper by grepping that monitor's layer
///   namespaces:
///   `… any(.[][]?; .namespace|test("wallpaperengine"))` — so the namespace
///   MUST contain the substring `wallpaperengine` or the watchdog concludes
///   the wallpaper is gone and kill-restarts the engine every ~45s. Default
///   `"linux-wallpaperengine"` matches; keep any custom value containing
///   `wallpaperengine`.
/// - `screen_roots` are the output/monitor names (`--screen-root` values,
///   e.g. `HDMI-A-1`) to place wallpaper surfaces on. **Empty means every
///   output.** Any output whose name is not listed gets no surface at all, so
///   the user's other monitors are left untouched instead of being blacked out
///   by an unconfigured wallpaper surface (Wayland backend; SPEC V6 — a
///   skipped output costs zero render work).
#[derive(Debug, Clone)]
pub struct PresentOptions {
    /// wlr-layer-shell surface namespace (Wayland only; ignored by X11).
    /// Must contain `wallpaperengine` for the daemon watchdog to match.
    pub layer_namespace: String,
    /// Output/monitor names to place surfaces on. Empty = all outputs.
    pub screen_roots: Vec<String>,
    /// `--fps` cap: skip rendering on frame callbacks that arrive early,
    /// re-requesting the next callback so pacing stays compositor-driven
    /// (the reference paces its GL swap the same way). `None`/0 = uncapped.
    pub fps: Option<u32>,
    /// `--playback-speed`/`--clock`: scales every scene's frame delta — the
    /// reference multiplies its clock (`g_Time`) by this
    /// (WallpaperApplication.cpp:908), so animations run faster/slower
    /// without changing the render FPS. Videos apply their own rate.
    pub playback_speed: f64,
}

impl Default for PresentOptions {
    fn default() -> Self {
        Self {
            layer_namespace: "linux-wallpaperengine".to_string(),
            screen_roots: Vec::new(),
            fps: None,
            playback_speed: 1.0,
        }
    }
}

/// Which presentation backend to bring up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Wayland `wlr-layer-shell` (docs/render-architecture.md §2.3).
    Wayland,
    /// X11 root-window / desktop background (docs/render-architecture.md
    /// §2.2).
    X11,
}

impl Backend {
    /// Pick a backend from the environment: Wayland when `$WAYLAND_DISPLAY`
    /// is set (the compositor is the native session), else X11 when
    /// `$DISPLAY` is set. Defaults to Wayland when neither is set so the
    /// error surfaced is the (more common) wayland connect failure.
    #[must_use]
    pub fn from_env() -> Self {
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            Backend::Wayland
        } else if std::env::var_os("DISPLAY").is_some() {
            Backend::X11
        } else {
            Backend::Wayland
        }
    }
}

/// The presentation layer: owns whichever backend was selected. All variants
/// drive the same [`crate::Renderer`] contract (SPEC V1: state owned here,
/// nothing global).
// One long-lived instance per process; the variant size skew is irrelevant.
#[allow(clippy::large_enum_variant)]
pub enum Platform {
    /// Wayland layer-shell backend.
    Wayland(WaylandPlatform),
    /// X11 root-window / desktop backend.
    X11(X11Platform),
}

impl Platform {
    /// Connect using the backend implied by the environment
    /// ([`Backend::from_env`]). On X11 this uses the desktop (behind-windows)
    /// wallpaper mode; use [`Platform::connect_x11`] to force a mode.
    pub fn connect(make_renderer: RendererFactory) -> Result<Self, PlatformError> {
        Self::connect_backend(Backend::from_env(), make_renderer)
    }

    /// Connect using an explicit backend with default [`PresentOptions`]
    /// (namespace `linux-wallpaperengine`, all outputs). On X11 this uses the
    /// desktop wallpaper mode ([`X11Mode::Desktop`]).
    ///
    /// Forcing a backend matters when both `$WAYLAND_DISPLAY` and `$DISPLAY`
    /// are set (an Xwayland session under a wayland compositor): the CLI's
    /// `--window`/desktop selection and the X11 live test both need to
    /// bypass the env heuristic.
    pub fn connect_backend(backend: Backend, make_renderer: RendererFactory) -> Result<Self, PlatformError> {
        Self::connect_with(backend, PresentOptions::default(), make_renderer)
    }

    /// Connect using an explicit backend and explicit [`PresentOptions`].
    ///
    /// This is the drop-in entry point the compat CLI uses: it carries the
    /// `--screen-root` selection (so only the requested monitors get a
    /// wallpaper surface) and the layer-shell namespace the daemon watchdog
    /// greps for. The X11 backend ignores both fields today (its per-CRTC
    /// desktop windows are already scoped to real monitors and have no
    /// layer-shell namespace); see [`PresentOptions`].
    pub fn connect_with(
        backend: Backend,
        options: PresentOptions,
        make_renderer: RendererFactory,
    ) -> Result<Self, PlatformError> {
        match backend {
            Backend::Wayland => Ok(Self::Wayland(WaylandPlatform::connect_with(
                make_renderer,
                options,
            )?)),
            Backend::X11 => Self::connect_x11(X11Mode::Desktop, make_renderer),
        }
    }

    /// Connect the X11 backend with an explicit window mode
    /// (desktop-background vs a normal `--window`,
    /// docs/render-architecture.md §2.2).
    pub fn connect_x11(mode: X11Mode, make_renderer: RendererFactory) -> Result<Self, PlatformError> {
        Ok(Self::X11(X11Platform::connect(mode, make_renderer)?))
    }

    /// Number of outputs/monitors that currently have a surface.
    #[must_use]
    pub fn output_count(&self) -> usize {
        match self {
            Self::Wayland(p) => p.output_count(),
            Self::X11(p) => p.output_count(),
        }
    }

    /// Render-command sender for live `bg`/`preload` swaps. `Some` on Wayland;
    /// `None` on X11 (not wired there yet — X11 relaunches per switch).
    #[must_use]
    pub fn command_sender(&self) -> Option<crate::renderer::CommandSender> {
        match self {
            Self::Wayland(p) => Some(p.command_sender()),
            Self::X11(_) => None,
        }
    }

    /// Drive the backend's render loop until `duration` elapses (`None` = run
    /// forever). Wayland blocks in the compositor event loop between frame
    /// callbacks; X11 runs the vsync-paced present loop
    /// (docs/render-architecture.md §2.2–§2.3).
    pub fn run(&mut self, duration: Option<Duration>) -> Result<(), PlatformError> {
        match self {
            Self::Wayland(p) => p.run(duration),
            Self::X11(p) => p.run(duration),
        }
    }
}
