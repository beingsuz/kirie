//! The `linux-wallpaperengine` compat flag surface: the parsed model
//! ([`CompatArgs`]), the hand-rolled parser ([`parse`]), the exact `--help`
//! synopsis, and typed parse errors — all per docs/compat-cli.md.
//!
//! The parser is hand-rolled rather than clap-driven because spec fidelity
//! wins over ergonomics: argument *order* is load-bearing for the per-screen
//! options (doc §3.1), unknown flags must be silently ignored (doc §4.1),
//! `--flag=value` must split at the first `=` (doc §4.2), repeating a
//! non-repeatable flag is fatal (doc §4.4), and the mutually-exclusive
//! `-w`/`-r` and `-v`/`-s` groups are *not* enforced (doc §4.3) — only the
//! hand-coded window/background mode conflict is (doc §3.1, §3.2).

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use crate::compat::resolve;

/// Wallpaper Engine's Steam Workshop app id (doc §3.4, ApplicationContext.cpp:19).
pub const WORKSHOP_APP_ID: &str = "431960";

/// The three window modes (doc §3.3, ApplicationContext.h:34-41).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowMode {
    /// `NORMAL_WINDOW`: preview window, no `-w`/`-r` given.
    #[default]
    Normal,
    /// `DESKTOP_BACKGROUND`: at least one `-r`/`--screen-span`.
    DesktopBackground,
    /// `EXPLICIT_WINDOW`: `-w`/`--window` given.
    ExplicitWindow,
}

/// Wayland `wlr-layer-shell` layer, CLI `--layer` (doc §2, default `bottom`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layer {
    /// `background` layer (use on niri with `place-within-backdrop`).
    Background,
    /// `bottom` layer — the C++ default (doc §2).
    #[default]
    Bottom,
    /// `top` layer.
    Top,
    /// `overlay` layer.
    Overlay,
}

impl Layer {
    /// The CLI spelling of this layer (doc §2 choices).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Background => "background",
            Self::Bottom => "bottom",
            Self::Top => "top",
            Self::Overlay => "overlay",
        }
    }
}

/// Output scaling mode, CLI `--scaling` (doc §2, §3.1: `stretch`/`fit`/
/// `fill`/`default`). Compat-local so the parser is self-contained; mapped to
/// the `kirie-render`/`kirie-video` enums at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// `default` → `DefaultUVs` (the CLI default, doc §2).
    #[default]
    Default,
    /// `fit` → `ZoomFitUVs`.
    Fit,
    /// `fill` → `ZoomFillUVs`.
    Fill,
    /// `stretch` → `StretchUVs`.
    Stretch,
}

impl ScalingMode {
    /// The CLI spelling (doc §2 choices).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Fit => "fit",
            Self::Fill => "fill",
            Self::Stretch => "stretch",
        }
    }
}

/// UV clamp mode, CLI `--clamp` (doc §2, §3.1: `clamp`/`border`/`repeat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClampMode {
    /// `clamp` → `TextureFlags_ClampUVs` (the CLI default, doc §2).
    #[default]
    Clamp,
    /// `border` → `TextureFlags_ClampUVsBorder`.
    Border,
    /// `repeat` → `TextureFlags_NoFlags`.
    Repeat,
}

impl ClampMode {
    /// The CLI spelling (doc §2 choices).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clamp => "clamp",
            Self::Border => "border",
            Self::Repeat => "repeat",
        }
    }
}

/// `--window XxYxWxH` geometry (doc §3.2): X/Y position, width, height,
/// `strtol` base-10 parsed (garbage → 0, no range validation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowGeometry {
    /// X position.
    pub x: i64,
    /// Y position.
    pub y: i64,
    /// Width.
    pub w: i64,
    /// Height.
    pub h: i64,
}

/// One `--render-debug` switch (doc §3.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderDebug {
    /// `base-only`.
    BaseOnly,
    /// `no-solid-final`.
    NoSolidFinal,
    /// `pass-log`.
    PassLog,
    /// `object=<id>`.
    Object(i64),
    /// `skip-object=<id>`.
    SkipObject(i64),
    /// `skip-effect=<id>`.
    SkipEffect(i64),
}

/// One `-r`/`--screen-root` or `--screen-span` output, in declaration order,
/// with the per-screen options that followed it (doc §3.1).
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenConfig {
    /// Registration key: the `-r` output name, or `span:<raw value>` for a
    /// span group (doc §3.1).
    pub name: String,
    /// Whether this is a `--screen-span` group.
    pub is_span: bool,
    /// Member output names (span groups only).
    pub members: Vec<String>,
    /// Background for this screen (`translateBackground`-resolved, doc §3.4);
    /// `None` means inherit `default_background` at load time (doc §3.1).
    pub background: Option<String>,
    /// Scaling mode for this screen (inherits the window default at `-r`
    /// time, doc §3.1).
    pub scaling: ScalingMode,
    /// Clamp mode for this screen (inherits the window default at `-r` time).
    pub clamp: ClampMode,
    /// Per-screen playlist name, if a `--playlist` followed this `-r`
    /// (doc §3.5; playlist *resolution* is not implemented — see run.rs).
    pub playlist: Option<String>,
}

/// The fully parsed compat command line (doc §2 flag table).
#[derive(Debug, Clone, PartialEq)]
pub struct CompatArgs {
    /// Every argv element verbatim, for the `Running with:` banner (doc §1.2)
    /// and the `--help` error suffix (doc §4.7).
    pub argv: Vec<String>,
    /// `-h`/`--help` seen.
    pub help: bool,
    /// Declared screens/spans in order (doc §3.1).
    pub screens: Vec<ScreenConfig>,
    /// `-w`/`--window` geometry (doc §3.2).
    pub window: Option<WindowGeometry>,
    /// Window/global default scaling — the value inherited by later `-r`
    /// screens and used in window mode (doc §3.1).
    pub window_scaling: ScalingMode,
    /// Window/global default clamp (doc §3.1).
    pub window_clamp: ClampMode,
    /// Window-mode default playlist (doc §3.5), if `--playlist` was given
    /// before any `-r`.
    pub window_playlist: Option<String>,
    /// `general.defaultBackground`: the last `--bg`/positional wins (doc §3.1,
    /// §3.4-resolved). `None` when never set → fatal at validation (doc §4.8).
    pub default_background: Option<String>,
    /// The resolved window mode (doc §3.3).
    pub mode: WindowMode,
    /// `--layer` (doc §2).
    pub layer: Layer,
    /// `-f`/`--fps` (doc §2, default 30).
    pub fps: i64,
    /// `--playback-speed`/`--clock` (doc §2, default 1.0).
    pub playback_speed: f64,
    /// `--render-scale` (doc §2, default 1.0; clamped [0.5, 2.0] at use).
    pub render_scale: f64,
    /// `--control-socket` path (doc §2, `None` = disabled).
    pub control_socket: Option<PathBuf>,
    /// `--audio-device` (doc §2, `None` = default monitor).
    pub audio_device: Option<String>,
    /// `--no-fullscreen-pause` (doc §2).
    pub no_fullscreen_pause: bool,
    /// `--fullscreen-pause-only-active` (doc §2).
    pub fullscreen_pause_only_active: bool,
    /// `--fullscreen-pause-ignore-appid` values (doc §2, repeatable; empty
    /// values discarded, doc §2 note).
    pub fullscreen_pause_ignore_appid: Vec<String>,
    /// `-v`/`--volume` (doc §2, default 15; clamped [0,128] at validation).
    pub volume: i64,
    /// `-s`/`--silent` (doc §2).
    pub silent: bool,
    /// `--noautomute` (doc §2).
    pub noautomute: bool,
    /// `--no-audio-processing` (doc §2).
    pub no_audio_processing: bool,
    /// `--screenshot` path (doc §2/§3.6, `None` = off).
    pub screenshot: Option<PathBuf>,
    /// `--screenshot-delay` frames (doc §2, default 5; clamped [0,600]).
    pub screenshot_delay: u32,
    /// `--assets-dir` (doc §2, `None` = auto-detect).
    pub assets_dir: Option<PathBuf>,
    /// `--disable-particles` (doc §2).
    pub disable_particles: bool,
    /// `--disable-mouse` (doc §2).
    pub disable_mouse: bool,
    /// `--disable-parallax` (doc §2).
    pub disable_parallax: bool,
    /// `-l`/`--list-properties` (doc §3.8).
    pub list_properties: bool,
    /// `--list-properties-json` (doc §3.8).
    pub list_properties_json: bool,
    /// `--set-property`/`--property` `(key, value)` pairs (doc §3.10, order
    /// preserved; bare key → value `"1"`).
    pub set_properties: Vec<(String, String)>,
    /// `-z`/`--dump-structure` (doc §2).
    pub dump_structure: bool,
    /// `--render-debug` switches (doc §3.9).
    pub render_debug: Vec<RenderDebug>,
}

impl Default for CompatArgs {
    fn default() -> Self {
        Self {
            argv: Vec::new(),
            help: false,
            screens: Vec::new(),
            window: None,
            window_scaling: ScalingMode::Default,
            window_clamp: ClampMode::Clamp,
            window_playlist: None,
            default_background: None,
            mode: WindowMode::Normal,
            layer: Layer::Bottom,
            fps: 30,
            playback_speed: 1.0,
            render_scale: 1.0,
            control_socket: None,
            audio_device: None,
            no_fullscreen_pause: false,
            fullscreen_pause_only_active: false,
            fullscreen_pause_ignore_appid: Vec::new(),
            volume: 15,
            silent: false,
            noautomute: false,
            no_audio_processing: false,
            screenshot: None,
            screenshot_delay: 5,
            assets_dir: None,
            disable_particles: false,
            disable_mouse: false,
            disable_parallax: false,
            list_properties: false,
            list_properties_json: false,
            set_properties: Vec::new(),
            dump_structure: false,
            render_debug: Vec::new(),
        }
    }
}

/// A fatal parse error (doc §4). `doubled` controls the doc §4.7 stderr
/// doubling: `sLog.exception` errors print once bare, then again with the
/// `. Use <argv0> --help for more information` suffix; argparse scan/choices
/// errors print once (the suffix is embedded in `message` where present).
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// The primary message (matches the C++ text reasonably closely).
    pub message: String,
    /// Whether to reproduce the doc §4.7 doubling.
    pub doubled: bool,
}

impl ParseError {
    fn doubled(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            doubled: true,
        }
    }

    fn single(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            doubled: false,
        }
    }
}

/// The exact `--help` synopsis the real binary prints (doc §2 [observed]),
/// rendered as a single Usage line followed by the grouped detail listing.
///
/// The real binary prints the synopsis on one physical line; this wraps it
/// only for readability of the leading `Usage:` block, matching the doc's
/// verbatim capture in fixtures/cpp-help-capture.txt closely enough for
/// tooling (no daemon script parses `--help`).
pub const HELP_TEXT: &str = concat!(
    "Usage: linux-wallpaperengine [--help] [[--window VAR]...|[--screen-root VAR]...] ",
    "[--screen-span VAR]... [--bg VAR]... [--playlist VAR]... [--scaling VAR]... ",
    "[--clamp VAR]... [--layer VAR] [--fps VAR] [--playback-speed VAR] ",
    "[--render-scale VAR] [--control-socket VAR] [--audio-device VAR] ",
    "[--no-fullscreen-pause] [--fullscreen-pause-only-active] ",
    "[--fullscreen-pause-ignore-appid VAR]... [[--volume VAR]|[--silent]] ",
    "[--noautomute] [--no-audio-processing] [--screenshot VAR] ",
    "[--screenshot-delay VAR] [--assets-dir VAR] [--disable-particles] ",
    "[--disable-mouse] [--disable-parallax] [--list-properties] ",
    "[--list-properties-json] [--set-property VAR]... [--dump-structure] ",
    "[--render-debug VAR]... background id\n",
);

/// Parse an integer with C `strtol` base-10 semantics used by `--window`
/// (doc §3.2): optional sign, longest decimal-digit prefix, garbage → 0. No
/// range validation (the C++ stores whatever `strtol` returns).
fn strtol(s: &str) -> i64 {
    let bytes = s.as_bytes();
    let mut i = 0;
    let neg = match bytes.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let start = i;
    let mut v: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        v = v.saturating_mul(10).saturating_add(i64::from(bytes[i] - b'0'));
        i += 1;
    }
    if i == start {
        return 0;
    }
    if neg { -v } else { v }
}

/// Parse `--window XxYxWxH` (doc §3.2): at least three `x` delimiters
/// (≥ 4 components) required; each component's `strtol` stops at the next
/// non-digit, so extra `x`s are harmless (`1x2x3x4x5` → x=1 y=2 w=3 h=4).
fn parse_geometry(value: &str) -> Result<WindowGeometry, ParseError> {
    let parts: Vec<&str> = value.split('x').collect();
    if parts.len() < 4 {
        return Err(ParseError::doubled(
            "Window geometry must be in the format: XxYxWxH",
        ));
    }
    Ok(WindowGeometry {
        x: strtol(parts[0]),
        y: strtol(parts[1]),
        w: strtol(parts[2]),
        h: strtol(parts[3]),
    })
}

/// Longest-prefix decimal parse for `--fps`/`--volume`/`--screenshot-delay`
/// (argparse `.scan<'i', int>()`): a bare integer. A value that is not a
/// clean integer is a fatal scan error (doc §4.5).
fn scan_int(flag: &str, value: &str) -> Result<i64, ParseError> {
    value
        .trim()
        .parse::<i64>()
        .map_err(|_| ParseError::single(format!("Invalid numeric value '{value}' for {flag}")))
}

/// Float parse for `--playback-speed`/`--render-scale` (argparse
/// `.scan<'g', double>()`).
fn scan_float(flag: &str, value: &str) -> Result<f64, ParseError> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| ParseError::single(format!("Invalid numeric value '{value}' for {flag}")))
}

/// Validate a `.choices(...)` value (doc §4.6), producing the C++ message
/// shape on rejection.
fn choice<T: Copy>(argv0: &str, flag_value: &str, allowed: &[(&str, T)]) -> Result<T, ParseError> {
    if let Some((_, v)) = allowed.iter().find(|(s, _)| *s == flag_value) {
        return Ok(*v);
    }
    let names: Vec<&str> = allowed.iter().map(|(s, _)| *s).collect();
    Err(ParseError::single(format!(
        "Invalid argument \"{flag_value}\" - allowed options: {{{}}}. Use {argv0} --help for more information",
        names.join(", ")
    )))
}

/// A `render-debug` numeric argument (doc §3.9); non-numeric is fatal.
fn render_debug_int(rest: &str) -> Result<i64, ParseError> {
    rest.parse::<i64>()
        .map_err(|_| ParseError::doubled(format!("Invalid numeric value for --render-debug: {rest}")))
}

/// Parse one `--render-debug MODE` value (doc §3.9 table).
fn parse_render_debug(value: &str) -> Result<RenderDebug, ParseError> {
    match value {
        "base-only" => Ok(RenderDebug::BaseOnly),
        "no-solid-final" => Ok(RenderDebug::NoSolidFinal),
        "pass-log" => Ok(RenderDebug::PassLog),
        _ => {
            if let Some(rest) = value.strip_prefix("object=") {
                Ok(RenderDebug::Object(render_debug_int(rest)?))
            } else if let Some(rest) = value.strip_prefix("skip-object=") {
                Ok(RenderDebug::SkipObject(render_debug_int(rest)?))
            } else if let Some(rest) = value.strip_prefix("skip-effect=") {
                Ok(RenderDebug::SkipEffect(render_debug_int(rest)?))
            } else {
                Err(ParseError::doubled(format!("Invalid render debug mode: {value}")))
            }
        }
    }
}

/// Which flags may appear only once (doc §2 "Rep?" = no, doc §4.4). A second
/// use is a fatal `Duplicate argument --X` error.
fn is_non_repeatable(canonical: &str) -> bool {
    matches!(
        canonical,
        "--layer"
            | "--fps"
            | "--playback-speed"
            | "--render-scale"
            | "--control-socket"
            | "--audio-device"
            | "--no-fullscreen-pause"
            | "--fullscreen-pause-only-active"
            | "--volume"
            | "--silent"
            | "--noautomute"
            | "--no-audio-processing"
            | "--screenshot"
            | "--screenshot-delay"
            | "--assets-dir"
            | "--disable-particles"
            | "--disable-mouse"
            | "--disable-parallax"
            | "--list-properties"
            | "--list-properties-json"
            | "--dump-structure"
    )
}

/// Whether a flag consumes a following value (or the inline `--flag=value`).
/// Flag-only options (booleans, `--help`) take no value (doc §2 table).
fn flag_takes_value(canonical: &str) -> bool {
    matches!(
        canonical,
        "--window"
            | "--screen-root"
            | "--screen-span"
            | "--bg"
            | "--playlist"
            | "--scaling"
            | "--clamp"
            | "--layer"
            | "--fps"
            | "--playback-speed"
            | "--render-scale"
            | "--control-socket"
            | "--audio-device"
            | "--fullscreen-pause-ignore-appid"
            | "--volume"
            | "--screenshot"
            | "--screenshot-delay"
            | "--assets-dir"
            | "--set-property"
            | "--render-debug"
    )
}

/// The parse cursor tracking the current per-screen target (doc §3.1
/// `lastScreen`): options apply to the window defaults until the first `-r`,
/// then to the most recently declared screen/span.
enum Cursor {
    Window,
    Screen(usize),
}

/// Parse the compat command line (`args[0]` is the program name).
///
/// Total on any input (SPEC V9): every failure is a typed [`ParseError`],
/// never a panic. Semantics mirror docs/compat-cli.md §3-§4.
pub fn parse(args: &[OsString]) -> Result<CompatArgs, ParseError> {
    let argv0 = args
        .first()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "linux-wallpaperengine".to_owned());
    let mut out = CompatArgs {
        argv: args.iter().map(|a| a.to_string_lossy().into_owned()).collect(),
        ..CompatArgs::default()
    };
    let mut cursor = Cursor::Window;
    let mut seen: Vec<String> = Vec::new();

    // Value-fetch helper state: we iterate manually so a flag can consume the
    // next argv element as its value (or the inline `--flag=value` form).
    let mut i = 1;
    while i < args.len() {
        let raw = &args[i];
        let token = raw.to_string_lossy();

        // A bare positional (no leading '-', or the lone "-"): the background
        // id (doc §2 positional, nargs 0..1). First non-empty wins; extras are
        // ignored like argparse's discarded extra positionals (doc §4.1).
        if !token.starts_with('-') || token.as_ref() == "-" {
            if !token.is_empty() {
                out.default_background = Some(resolve::translate_background(&token)?);
            }
            i += 1;
            continue;
        }

        // Split the `--flag=value` inline form at the first '=' (doc §4.2).
        // Short flags (`-x`) are never given inline values here.
        let (name, inline): (String, Option<String>) = if token.starts_with("--") {
            match token.split_once('=') {
                Some((n, v)) => (n.to_owned(), Some(v.to_owned())),
                None => (token.into_owned(), None),
            }
        } else {
            (token.into_owned(), None)
        };

        let canonical = canonical_flag(&name);
        let Some(canonical) = canonical else {
            // Unknown flag (incl. CEF `--type=...`): silently ignored, and we
            // do NOT consume a following value (doc §4.1, §1.1).
            i += 1;
            continue;
        };

        if is_non_repeatable(canonical) && seen.iter().any(|s| s == canonical) {
            return Err(ParseError::single(format!(
                "Duplicate argument {canonical}. Use {argv0} --help for more information"
            )));
        }
        seen.push(canonical.to_owned());

        // Pre-fetch this flag's value (inline `--flag=v` form, else the next
        // argv element) so the borrow of `consumed_next` closes before we act
        // on it below. Flags that take no value never consume the next token.
        let mut consumed_next = false;
        let fetched: Result<String, ParseError> = if flag_takes_value(canonical) {
            if let Some(v) = inline {
                Ok(v)
            } else {
                match args.get(i + 1) {
                    Some(v) => {
                        consumed_next = true;
                        Ok(v.to_string_lossy().into_owned())
                    }
                    None => Err(ParseError::single(format!("{canonical}: expected one argument"))),
                }
            }
        } else {
            Ok(String::new())
        };
        // A no-op for value-less flags; value-taking flags read via `value()`.
        let value = || fetched.clone();

        match canonical {
            "--help" => out.help = true,
            "--window" => {
                if out.mode == WindowMode::DesktopBackground {
                    return Err(ParseError::doubled(
                        "Cannot run in both background and window mode",
                    ));
                }
                if out.window.is_some() {
                    return Err(ParseError::doubled("Only one window at a time can be specified"));
                }
                out.window = Some(parse_geometry(&value()?)?);
                out.mode = WindowMode::ExplicitWindow;
            }
            "--screen-root" => {
                let name = value()?;
                apply_screen_root(&mut out, &mut cursor, name)?;
            }
            "--screen-span" => {
                let raw = value()?;
                apply_screen_span(&mut out, &mut cursor, raw)?;
            }
            "--bg" => {
                let resolved = resolve::translate_background(&value()?)?;
                out.default_background = Some(resolved.clone());
                if let Cursor::Screen(idx) = cursor {
                    out.screens[idx].background = Some(resolved);
                }
            }
            "--playlist" => {
                let name = value()?;
                match cursor {
                    Cursor::Screen(idx) => out.screens[idx].playlist = Some(name),
                    Cursor::Window => out.window_playlist = Some(name),
                }
            }
            "--scaling" => {
                let mode = choice(
                    &argv0,
                    &value()?,
                    &[
                        ("stretch", ScalingMode::Stretch),
                        ("fit", ScalingMode::Fit),
                        ("fill", ScalingMode::Fill),
                        ("default", ScalingMode::Default),
                    ],
                )?;
                match cursor {
                    Cursor::Screen(idx) => out.screens[idx].scaling = mode,
                    Cursor::Window => out.window_scaling = mode,
                }
            }
            "--clamp" => {
                let mode = choice(
                    &argv0,
                    &value()?,
                    &[
                        ("clamp", ClampMode::Clamp),
                        ("border", ClampMode::Border),
                        ("repeat", ClampMode::Repeat),
                    ],
                )?;
                match cursor {
                    Cursor::Screen(idx) => out.screens[idx].clamp = mode,
                    Cursor::Window => out.window_clamp = mode,
                }
            }
            "--layer" => {
                out.layer = choice(
                    &argv0,
                    &value()?,
                    &[
                        ("background", Layer::Background),
                        ("bottom", Layer::Bottom),
                        ("top", Layer::Top),
                        ("overlay", Layer::Overlay),
                    ],
                )?;
            }
            "--fps" => out.fps = scan_int("--fps", &value()?)?,
            "--playback-speed" => {
                out.playback_speed = scan_float("--playback-speed", &value()?)?;
            }
            "--render-scale" => {
                out.render_scale = scan_float("--render-scale", &value()?)?;
            }
            "--control-socket" => out.control_socket = Some(PathBuf::from(value()?)),
            "--audio-device" => out.audio_device = Some(value()?),
            "--no-fullscreen-pause" => out.no_fullscreen_pause = true,
            "--fullscreen-pause-only-active" => out.fullscreen_pause_only_active = true,
            "--fullscreen-pause-ignore-appid" => {
                let v = value()?;
                // Empty values are discarded (doc §2 note).
                if !v.is_empty() {
                    out.fullscreen_pause_ignore_appid.push(v);
                }
            }
            "--volume" => out.volume = scan_int("--volume", &value()?)?,
            "--silent" => out.silent = true,
            "--noautomute" => out.noautomute = true,
            "--no-audio-processing" => out.no_audio_processing = true,
            "--screenshot" => out.screenshot = Some(PathBuf::from(value()?)),
            "--screenshot-delay" => {
                let n = scan_int("--screenshot-delay", &value()?)?;
                out.screenshot_delay = n.clamp(0, u32::MAX as i64) as u32;
            }
            "--assets-dir" => out.assets_dir = Some(PathBuf::from(value()?)),
            "--disable-particles" => out.disable_particles = true,
            "--disable-mouse" => out.disable_mouse = true,
            "--disable-parallax" => out.disable_parallax = true,
            "--list-properties" => out.list_properties = true,
            "--list-properties-json" => out.list_properties_json = true,
            "--set-property" => {
                let kv = value()?;
                out.set_properties.push(split_property(&kv));
            }
            "--dump-structure" => out.dump_structure = true,
            "--render-debug" => {
                let dbg = parse_render_debug(&value()?)?;
                out.render_debug.push(dbg);
            }
            other => unreachable!("unmapped canonical flag {other}"),
        }

        i += if consumed_next { 2 } else { 1 };
    }

    Ok(out)
}

/// `-b/--bg`/`--set-property` value grammar (doc §3.10): split at the first
/// `=`; a bare key with no `=` stores value `"1"`.
fn split_property(kv: &str) -> (String, String) {
    match kv.split_once('=') {
        Some((k, v)) => (k.to_owned(), v.to_owned()),
        None => (kv.to_owned(), "1".to_owned()),
    }
}

/// `-r`/`--screen-root` action (doc §3.1): mode → DESKTOP_BACKGROUND, register
/// the screen inheriting the current window defaults, make it current.
fn apply_screen_root(out: &mut CompatArgs, cursor: &mut Cursor, name: String) -> Result<(), ParseError> {
    if out.mode == WindowMode::ExplicitWindow {
        return Err(ParseError::doubled(
            "Cannot run in both background and window mode",
        ));
    }
    if out.screens.iter().any(|s| !s.is_span && s.name == name) {
        return Err(ParseError::doubled(
            "Cannot specify the same screen more than once",
        ));
    }
    if out.screens.iter().any(|s| s.is_span && s.members.contains(&name)) {
        return Err(ParseError::doubled(format!(
            "Screen {name} is already part of a span group"
        )));
    }
    out.mode = WindowMode::DesktopBackground;
    out.screens.push(ScreenConfig {
        name,
        is_span: false,
        members: Vec::new(),
        background: None,
        scaling: out.window_scaling,
        clamp: out.window_clamp,
        playlist: None,
    });
    *cursor = Cursor::Screen(out.screens.len() - 1);
    Ok(())
}

/// `--screen-span` action (doc §3.1): split on `,`, ≥ 2 members, each unique
/// across individual screens and other groups.
fn apply_screen_span(out: &mut CompatArgs, cursor: &mut Cursor, raw: String) -> Result<(), ParseError> {
    if out.mode == WindowMode::ExplicitWindow {
        return Err(ParseError::doubled(
            "Cannot run in both background and window mode",
        ));
    }
    let members: Vec<String> = raw
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    if members.len() < 2 {
        return Err(ParseError::doubled(
            "A span requires at least two comma-separated screen names",
        ));
    }
    for m in &members {
        if out.screens.iter().any(|s| !s.is_span && s.name == *m) {
            return Err(ParseError::doubled(format!(
                "Screen {m} is already configured individually"
            )));
        }
        if out
            .screens
            .iter()
            .any(|s| s.is_span && s.members.iter().any(|x| x == m))
        {
            return Err(ParseError::doubled(format!(
                "Screen {m} is already part of a span group"
            )));
        }
        // Duplicate within this same group.
        if members.iter().filter(|x| *x == m).count() > 1 {
            return Err(ParseError::doubled(format!(
                "Screen {m} is duplicated in the span group"
            )));
        }
    }
    out.mode = WindowMode::DesktopBackground;
    out.screens.push(ScreenConfig {
        name: format!("span:{raw}"),
        is_span: true,
        members,
        background: None,
        scaling: out.window_scaling,
        clamp: out.window_clamp,
        playlist: None,
    });
    *cursor = Cursor::Screen(out.screens.len() - 1);
    Ok(())
}

/// Map an argv flag spelling (long or short, incl. aliases) to its canonical
/// long name, or `None` for an unknown flag (doc §2 aliases:
/// `--clock`≡`--playback-speed`, `--property`≡`--set-property`).
fn canonical_flag(name: &str) -> Option<&'static str> {
    Some(match name {
        "-h" | "--help" => "--help",
        "-w" | "--window" => "--window",
        "-r" | "--screen-root" => "--screen-root",
        "--screen-span" => "--screen-span",
        "-b" | "--bg" => "--bg",
        "--playlist" => "--playlist",
        "--scaling" => "--scaling",
        "--clamp" => "--clamp",
        "--layer" => "--layer",
        "-f" | "--fps" => "--fps",
        "--playback-speed" | "--clock" => "--playback-speed",
        "--render-scale" => "--render-scale",
        "--control-socket" => "--control-socket",
        "--audio-device" => "--audio-device",
        "--no-fullscreen-pause" => "--no-fullscreen-pause",
        "--fullscreen-pause-only-active" => "--fullscreen-pause-only-active",
        "--fullscreen-pause-ignore-appid" => "--fullscreen-pause-ignore-appid",
        "-v" | "--volume" => "--volume",
        "-s" | "--silent" => "--silent",
        "--noautomute" => "--noautomute",
        "--no-audio-processing" => "--no-audio-processing",
        "--screenshot" => "--screenshot",
        "--screenshot-delay" => "--screenshot-delay",
        "--assets-dir" => "--assets-dir",
        "--disable-particles" => "--disable-particles",
        "--disable-mouse" => "--disable-mouse",
        "--disable-parallax" => "--disable-parallax",
        "-l" | "--list-properties" => "--list-properties",
        "--list-properties-json" => "--list-properties-json",
        "--set-property" | "--property" => "--set-property",
        "-z" | "--dump-structure" => "--dump-structure",
        "--render-debug" => "--render-debug",
        _ => return None,
    })
}

/// Post-parse validation (doc §4.8) that does not require I/O: the missing
/// background check, the volume clamp, and the screenshot-delay clamp. The
/// screenshot-extension check lives in run.rs (it only applies when the mode
/// runs). Returns the validated args or the fatal error.
pub fn validate(mut args: CompatArgs) -> Result<CompatArgs, ParseError> {
    // (1) defaultBackground empty → fatal (doc §4.8). A positional or any
    // `--bg` satisfies this; playlists would too but are unimplemented.
    if args.default_background.is_none() {
        return Err(ParseError::doubled(
            "At least one background ID must be specified",
        ));
    }
    // (2) volume clamp [0, 128] (doc §4.8).
    args.volume = args.volume.clamp(0, 128);
    // (3) screenshot-delay clamp [0, 600] (doc §4.8).
    args.screenshot_delay = args.screenshot_delay.min(600);
    Ok(args)
}

/// Screenshot path extension validation (doc §3.6): the extension must be one
/// of `.bmp`/`.png`/`.jpeg`/`.jpg` (case-sensitive, lowercase). Any other
/// extension is fatal.
pub fn validate_screenshot_ext(path: &OsStr) -> Result<(), ParseError> {
    let name = path.to_string_lossy();
    let ok =
        name.ends_with(".bmp") || name.ends_with(".png") || name.ends_with(".jpeg") || name.ends_with(".jpg");
    if ok {
        Ok(())
    } else {
        Err(ParseError::doubled(format!(
            "Cannot determine screenshot format, unknown extension for {name}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn strtol_semantics() {
        assert_eq!(strtol("1920"), 1920);
        assert_eq!(strtol("-5"), -5);
        assert_eq!(strtol("3junk"), 3);
        assert_eq!(strtol("junk"), 0);
        assert_eq!(strtol(""), 0);
    }

    #[test]
    fn geometry_parses_and_ignores_extra_x() {
        assert_eq!(
            parse_geometry("0x0x1920x1080").unwrap(),
            WindowGeometry {
                x: 0,
                y: 0,
                w: 1920,
                h: 1080
            }
        );
        // doc §3.2: 1x2x3x4x5 → x=1 y=2 w=3 h=4, trailing x5 ignored.
        assert_eq!(
            parse_geometry("1x2x3x4x5").unwrap(),
            WindowGeometry {
                x: 1,
                y: 2,
                w: 3,
                h: 4
            }
        );
        // Fewer than three delimiters → fatal.
        assert!(parse_geometry("1x2").is_err());
        assert!(parse_geometry("1920x1080x0").is_err());
    }

    #[test]
    fn help_flag_parses() {
        let args = parse(&os(&["linux-wallpaperengine", "--help"])).unwrap();
        assert!(args.help);
    }

    #[test]
    fn unknown_flags_are_ignored() {
        // doc §4.1: unknown flags do not error; parse fails later only on the
        // missing background.
        let args = parse(&os(&["kirie", "--bogus-flag", "--type=zygote"])).unwrap();
        assert!(args.default_background.is_none());
        assert!(validate(args).is_err());
    }

    #[test]
    fn duplicate_non_repeatable_is_fatal() {
        let err = parse(&os(&["kirie", "--fps", "30", "--fps", "60"])).unwrap_err();
        assert!(err.message.contains("Duplicate argument --fps"));
    }

    #[test]
    fn repeatable_flags_accumulate() {
        let args = parse(&os(&[
            "kirie",
            "--set-property",
            "a=1",
            "--set-property",
            "b=2",
            "/tmp/x",
        ]))
        .unwrap();
        assert_eq!(
            args.set_properties,
            vec![("a".into(), "1".into()), ("b".into(), "2".into())]
        );
    }

    #[test]
    fn bare_property_key_defaults_to_one() {
        let args = parse(&os(&["kirie", "--set-property", "bloom", "/tmp/x"])).unwrap();
        assert_eq!(args.set_properties, vec![("bloom".into(), "1".into())]);
    }

    #[test]
    fn window_and_screen_root_conflict() {
        let err = parse(&os(&[
            "kirie",
            "--window",
            "0x0x100x100",
            "--screen-root",
            "HDMI-A-1",
        ]))
        .unwrap_err();
        assert!(
            err.message
                .contains("Cannot run in both background and window mode")
        );
        // Order-reversed conflict too.
        let err = parse(&os(&[
            "kirie",
            "--screen-root",
            "HDMI-A-1",
            "--window",
            "0x0x100x100",
        ]))
        .unwrap_err();
        assert!(
            err.message
                .contains("Cannot run in both background and window mode")
        );
    }

    #[test]
    fn same_screen_twice_is_fatal() {
        let err = parse(&os(&[
            "kirie",
            "--screen-root",
            "HDMI-A-1",
            "--screen-root",
            "HDMI-A-1",
        ]))
        .unwrap_err();
        assert!(
            err.message
                .contains("Cannot specify the same screen more than once")
        );
    }

    #[test]
    fn bad_choice_is_fatal_with_message() {
        let err = parse(&os(&["kirie", "--scaling", "wrong", "/tmp/x"])).unwrap_err();
        assert!(err.message.contains("allowed options"));
        assert!(err.message.contains("stretch"));
    }

    #[test]
    fn clock_alias_maps_to_playback_speed() {
        let args = parse(&os(&["kirie", "--clock", "0.5", "/tmp/x"])).unwrap();
        assert_eq!(args.playback_speed, 0.5);
    }

    #[test]
    fn property_alias_maps_to_set_property() {
        let args = parse(&os(&["kirie", "--property", "a=b", "/tmp/x"])).unwrap();
        assert_eq!(args.set_properties, vec![("a".into(), "b".into())]);
    }

    #[test]
    fn inline_equals_form_parses() {
        // doc §4.2: --fps=30 and --set-property=foo=bar.
        let args = parse(&os(&["kirie", "--fps=30", "--set-property=foo=bar", "/tmp/x"])).unwrap();
        assert_eq!(args.fps, 30);
        assert_eq!(args.set_properties, vec![("foo".into(), "bar".into())]);
    }

    #[test]
    fn per_screen_scaling_before_r_is_the_inherited_default() {
        // doc §3.1: --scaling before any -r sets the window default, which
        // every later -r screen inherits.
        let args = parse(&os(&[
            "kirie",
            "--scaling",
            "fill",
            "--screen-root",
            "HDMI-A-1",
            "--screen-root",
            "DP-1",
            "/tmp/x",
        ]))
        .unwrap();
        assert_eq!(args.window_scaling, ScalingMode::Fill);
        assert_eq!(args.screens[0].scaling, ScalingMode::Fill);
        assert_eq!(args.screens[1].scaling, ScalingMode::Fill);
    }

    #[test]
    fn per_screen_scaling_after_r_is_local() {
        let args = parse(&os(&[
            "kirie",
            "--screen-root",
            "HDMI-A-1",
            "--scaling",
            "fill",
            "--screen-root",
            "DP-1",
            "/tmp/x",
        ]))
        .unwrap();
        assert_eq!(args.window_scaling, ScalingMode::Default);
        assert_eq!(args.screens[0].scaling, ScalingMode::Fill);
        assert_eq!(args.screens[1].scaling, ScalingMode::Default);
    }

    #[test]
    fn last_bg_wins_as_default_background() {
        let args = parse(&os(&[
            "kirie",
            "--screen-root",
            "HDMI-A-1",
            "--bg",
            "/a",
            "--screen-root",
            "DP-1",
            "--bg",
            "/b",
        ]))
        .unwrap();
        assert_eq!(args.screens[0].background.as_deref(), Some("/a"));
        assert_eq!(args.screens[1].background.as_deref(), Some("/b"));
        assert_eq!(args.default_background.as_deref(), Some("/b"));
    }

    #[test]
    fn the_exact_live_cmdline_parses_to_the_expected_model() {
        // fixtures/cpp-live-cmdline.txt (doc §8.1 [observed]): the daemon's
        // real launch argv. Socket path swapped for the task's test socket.
        let argv = os(&[
            "linux-wallpaperengine",
            "--control-socket",
            "/tmp/claude-1000/kirie-test.sock",
            "--screen-root",
            "HDMI-A-1",
            "--bg",
            "/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375",
            "--scaling",
            "fill",
            "--clamp",
            "clamp",
            "--fps",
            "30",
            "--render-scale",
            "1.06",
            "--volume",
            "0",
            "--set-property",
            "fov=48.333333333333336",
            "--set-property",
            "bloom=true",
            "--set-property",
            "radialblur=false",
            "--set-property",
            "huespeed=0.10555555555555556",
            "--set-property",
            "coloring1=2",
            "--set-property",
            "newproperty=0.025",
            "--set-property",
            "schemecolor=0.00000 0.00000 0.00000",
            "--set-property",
            "outline=0.36585 0.04268 0.43902",
            "--set-property",
            "bloomstrength=1.7916666666666665",
            "--set-property",
            "color1=0.00000 0.00000 1.00000",
            "--set-property",
            "color2=0.46951 0.00000 0.77439",
        ]);
        let args = validate(parse(&argv).unwrap()).unwrap();

        assert_eq!(args.mode, WindowMode::DesktopBackground);
        assert_eq!(
            args.control_socket.as_deref(),
            Some(std::path::Path::new("/tmp/claude-1000/kirie-test.sock"))
        );
        assert_eq!(args.screens.len(), 1);
        assert_eq!(args.screens[0].name, "HDMI-A-1");
        assert_eq!(
            args.screens[0].background.as_deref(),
            Some("/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375")
        );
        assert_eq!(args.screens[0].scaling, ScalingMode::Fill);
        assert_eq!(args.screens[0].clamp, ClampMode::Clamp);
        assert_eq!(args.fps, 30);
        assert!((args.render_scale - 1.06).abs() < 1e-12);
        assert_eq!(args.volume, 0);

        // 11 set-property pairs, values preserved verbatim (spaces intact).
        assert_eq!(args.set_properties.len(), 11);
        assert_eq!(
            args.set_properties[0],
            ("fov".to_owned(), "48.333333333333336".to_owned())
        );
        assert_eq!(
            args.set_properties[6],
            ("schemecolor".to_owned(), "0.00000 0.00000 0.00000".to_owned())
        );
        assert_eq!(
            args.set_properties[7],
            ("outline".to_owned(), "0.36585 0.04268 0.43902".to_owned())
        );
        assert_eq!(
            args.set_properties[10],
            ("color2".to_owned(), "0.46951 0.00000 0.77439".to_owned())
        );

        // default_background follows the (only) --bg (doc §3.1).
        assert_eq!(
            args.default_background.as_deref(),
            Some("/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375")
        );
    }
}
