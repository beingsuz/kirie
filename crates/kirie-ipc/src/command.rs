//! Request-line parsing: raw bytes → typed [`Command`] (docs/compat-socket.md §2, §4).
//!
//! The parser mirrors the C++ `ControlSocket::handle` dispatch
//! (ControlSocket.cpp:80-155, per docs/compat-socket.md) at the byte level:
//! same tokenization (`istringstream >>` semantics, doc §2 grammar), same
//! rest-of-line extraction (exactly one leading space stripped, doc §2), same
//! numeric-coercion quirks (doc §4.3-§4.6).
//!
//! Where the C++ splits *input validation* between `ControlSocket` and
//! `WallpaperApplication` (`set` key table, `scaling`/`clamp` mode strings,
//! `speed` ≤ 0 coercion), this parser folds the validation in so that
//! [`Command`] is fully typed. The wire-visible responses are unchanged
//! (doc §3/§4): inputs the C++ app would reject with `error\n` parse to
//! [`Request::Rejected`] here and never reach the app.

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

/// One parsed request line, before any app involvement.
///
/// Produced by [`parse_request`]; the server maps each variant to the fixed
/// response vocabulary of doc §3.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Zero-byte request line → no response bytes at all (doc §2 step 5).
    Empty,
    /// `ping` → `pong\n`, answered by the socket layer itself
    /// (doc §4.1, ControlSocket.cpp:94-96).
    Ping,
    /// `status` → multi-line snapshot reply (doc §4.2).
    Status,
    /// `getproperties [screen]` → single-line JSON property-schema reply
    /// (docs/compat-socket.md §11). A **kirie extension**, not part of the C++
    /// protocol (a real engine answers `unknown command\n`). The optional
    /// argument selects a registered screen; absent ⇒ the app's default screen.
    GetProperties {
        /// Screen key to report, or `None` for the default screen.
        screen: Option<String>,
    },
    /// A recognized command, forwarded to the app as an [`crate::IpcEvent`].
    Command(Command),
    /// Recognized command with an argument the C++ app would reject with
    /// `error\n` before touching any state: unknown `set` key (doc §4.6),
    /// invalid `scaling`/`clamp` mode string (doc §4.10/§4.11).
    Rejected,
    /// Unrecognized command token → `unknown command\n` (doc §3,
    /// ControlSocket.cpp:154). Also produced for whitespace-only lines,
    /// whose command-token extraction fails exactly as in C++.
    Unknown,
}

/// A typed control-socket command, ready for the app (SPEC V3: sent over a
/// channel, owned data only).
///
/// Argument coercions specified per command in docs/compat-socket.md §4 are
/// already applied; the app receives clean values.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// `speed <float>` — global playback-speed multiplier (doc §4.3).
    /// Missing arg → 1.0, non-numeric → 0 → coerced to 1.0, `v ≤ 0` → 1.0
    /// (coercion per WallpaperApplication.cpp:1277-1280 via doc §4.3).
    /// Always acknowledged with `ok\n`.
    Speed(f32),
    /// `volume <int>` — audio volume on the 0–128 Wallpaper Engine scale
    /// (doc §4.4). Missing arg → 128, non-numeric → 0. Deliberately **not**
    /// clamped: the C++ socket path forwards out-of-range values as-is.
    /// Always acknowledged with `ok\n`.
    Volume(i32),
    /// `mute <int>` — nonzero mutes, zero (or missing/non-numeric arg)
    /// unmutes (doc §4.5). Always acknowledged with `ok\n`.
    Mute(bool),
    /// `set <key> <value>` — live global engine option (doc §4.6). Key and
    /// value are pre-validated/coerced into [`SetOption`]; recognized keys
    /// always acknowledge `ok\n`.
    Set(SetOption),
    /// `bg <screen> <path>` — live wallpaper swap (doc §4.7). `path` is
    /// rest-of-line (spaces allowed), a directory containing `project.json`.
    /// The app decides `ok\n`/`error\n`. Note the C++ engine performs no
    /// screen-name validation (doc §4.7): unknown screens still load + `ok`.
    Bg {
        /// Registration key: output name, `default`, or `span:<screen>`
        /// (doc §4).
        screen: String,
        /// Wallpaper directory (raw bytes preserved; may contain spaces).
        path: PathBuf,
    },
    /// `preload <path>` — warm the resident cache (doc §4.8). Always
    /// acknowledged `ok\n`, even if the app-side load fails
    /// (ControlSocket.cpp:128-132 via doc §4.8).
    Preload {
        /// Wallpaper directory, rest-of-line.
        path: PathBuf,
    },
    /// `property <screen> <key> <value>` — set a wallpaper user property
    /// (doc §4.9). `value` is rest-of-line (color triples contain spaces).
    /// The app decides `ok\n`/`error\n`. Value typing depends on the loaded
    /// wallpaper's `project.json`, so it stays a raw string here.
    Property {
        /// Screen key the wallpaper is registered under.
        screen: String,
        /// Property name from `project.json` `general.properties`.
        key: String,
        /// Raw property value, rest-of-line.
        value: String,
    },
    /// `scaling <screen> <mode>` — per-screen texture scaling + rebuild
    /// (doc §4.10). Mode is pre-validated; the app decides `ok\n`/`error\n`
    /// (no recorded background / rebuild failure). Per doc §4.10 the C++ app
    /// records the mode *before* the screen lookup — the app side must
    /// reproduce that quirk.
    Scaling {
        /// Screen key.
        screen: String,
        /// Validated scaling mode.
        mode: ScalingMode,
    },
    /// `clamp <screen> <mode>` — per-screen texture addressing + rebuild
    /// (doc §4.11). Same shape and quirks as [`Command::Scaling`].
    Clamp {
        /// Screen key.
        screen: String,
        /// Validated clamp mode.
        mode: ClampMode,
    },
    /// `screenshot <path>` — capture the current frame (doc §4.12). `path`
    /// is rest-of-line. The app decides `ok\n`/`error\n` (empty path or no
    /// wallpaper rendering → `error\n`).
    Screenshot {
        /// Destination file, rest-of-line. Extension rules per doc §4.12.
        path: PathBuf,
    },
}

/// `set` option keys with their coerced values (doc §4.6 table).
#[derive(Debug, Clone, PartialEq)]
pub enum SetOption {
    /// `fps <int>` → `max(1, atoi(value))` (doc §4.6; non-numeric → 1).
    /// Upstream `atoi` overflow is UB; this parser saturates instead.
    Fps(i32),
    /// `noautomute <bool>` → automute disabled when true.
    NoAutomute(bool),
    /// `disablemouse <bool>` → mouse input disabled when true.
    DisableMouse(bool),
    /// `disableparallax <bool>` → parallax disabled when true.
    DisableParallax(bool),
    /// `nofullscreenpause <bool>` → fullscreen pause disabled when true.
    NoFullscreenPause(bool),
    /// `renderscale <float>` → `atof` then clamped to `[0.5, 2.0]`
    /// (non-numeric → 0 → 0.5) per doc §4.6.
    RenderScale(f32),
    /// `audiodevice <string>` → PulseAudio source name, rest-of-line; the
    /// literal `default` maps to the empty string (doc §4.6).
    AudioDevice(String),
}

/// `scaling` mode strings (doc §4.10 table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    /// `stretch` → `TextureUVsScaling::StretchUVs`.
    Stretch,
    /// `fit` → `TextureUVsScaling::ZoomFitUVs`.
    Fit,
    /// `fill` → `TextureUVsScaling::ZoomFillUVs`.
    Fill,
    /// `default` → `TextureUVsScaling::DefaultUVs`.
    Default,
}

impl ScalingMode {
    /// The exact wire token for this mode (doc §4.10).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stretch => "stretch",
            Self::Fit => "fit",
            Self::Fill => "fill",
            Self::Default => "default",
        }
    }
}

/// `clamp` mode strings (doc §4.11 table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClampMode {
    /// `clamp` → `TextureFlags_ClampUVs`.
    Clamp,
    /// `border` → `TextureFlags_ClampUVsBorder`.
    Border,
    /// `repeat` → `TextureFlags_NoFlags`.
    Repeat,
}

impl ClampMode {
    /// The exact wire token for this mode (doc §4.11).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clamp => "clamp",
            Self::Border => "border",
            Self::Repeat => "repeat",
        }
    }
}

impl Command {
    /// Whether the C++ socket layer maps the handler result onto
    /// `ok\n`/`error\n` (`true`), or replies `ok\n` unconditionally
    /// (`false`) — doc §4 summary table, ControlSocket.cpp:94-154.
    pub fn is_fallible(&self) -> bool {
        matches!(
            self,
            Self::Bg { .. }
                | Self::Property { .. }
                | Self::Scaling { .. }
                | Self::Clamp { .. }
                | Self::Screenshot { .. }
        )
    }

    /// Render this command back into a canonical wire request line (no
    /// trailing `\n`).
    ///
    /// Round-trip contract (SPEC V13): for every command in the
    /// post-parse domain (coerced values, non-empty screen/key tokens, no
    /// embedded newlines), `parse_request(&cmd.to_request_line())` yields
    /// `Request::Command(cmd)` again.
    pub fn to_request_line(&self) -> Vec<u8> {
        fn join(parts: &[&[u8]]) -> Vec<u8> {
            let mut out = Vec::new();
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(b' ');
                }
                out.extend_from_slice(p);
            }
            out
        }
        let bool_str = |b: bool| if b { "true" } else { "false" };
        match self {
            Self::Speed(v) => format!("speed {v}").into_bytes(),
            Self::Volume(v) => format!("volume {v}").into_bytes(),
            Self::Mute(m) => format!("mute {}", i32::from(*m)).into_bytes(),
            Self::Set(opt) => match opt {
                SetOption::Fps(n) => format!("set fps {n}"),
                SetOption::NoAutomute(b) => format!("set noautomute {}", bool_str(*b)),
                SetOption::DisableMouse(b) => format!("set disablemouse {}", bool_str(*b)),
                SetOption::DisableParallax(b) => format!("set disableparallax {}", bool_str(*b)),
                SetOption::NoFullscreenPause(b) => format!("set nofullscreenpause {}", bool_str(*b)),
                SetOption::RenderScale(v) => format!("set renderscale {v}"),
                SetOption::AudioDevice(s) => format!("set audiodevice {s}"),
            }
            .into_bytes(),
            Self::Bg { screen, path } => join(&[b"bg", screen.as_bytes(), path.as_os_str().as_bytes()]),
            Self::Preload { path } => join(&[b"preload", path.as_os_str().as_bytes()]),
            Self::Property { screen, key, value } => {
                join(&[b"property", screen.as_bytes(), key.as_bytes(), value.as_bytes()])
            }
            Self::Scaling { screen, mode } => {
                join(&[b"scaling", screen.as_bytes(), mode.as_str().as_bytes()])
            }
            Self::Clamp { screen, mode } => join(&[b"clamp", screen.as_bytes(), mode.as_str().as_bytes()]),
            Self::Screenshot { path } => join(&[b"screenshot", path.as_os_str().as_bytes()]),
        }
    }
}

/// Byte-level cursor reproducing `istringstream` extraction over one request
/// line (doc §2 grammar). The line must not contain the terminating `\n`.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// C "C-locale" `isspace` set; `>>` skips these and they delimit tokens.
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// `istringstream >>` into a string: skip whitespace, take the non-ws
    /// run. Leaves the cursor *on* the terminating whitespace byte, matching
    /// the C++ stream position after `>>` — this is what makes the
    /// rest-of-line rule of doc §2 come out right.
    fn token(&mut self) -> Option<&'a [u8]> {
        while self.pos < self.bytes.len() && is_ws(self.bytes[self.pos]) {
            self.pos += 1;
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && !is_ws(self.bytes[self.pos]) {
            self.pos += 1;
        }
        (self.pos > start).then(|| &self.bytes[start..self.pos])
    }

    /// Rest-of-line argument (doc §2, ControlSocket.cpp:85-92): everything
    /// after the previous token to end of line, with **exactly one** leading
    /// space stripped. A second space (or a `\t`/`\r` separator) stays part
    /// of the value — bug-compatible by contract.
    fn rest(&mut self) -> &'a [u8] {
        let mut r = &self.bytes[self.pos..];
        self.pos = self.bytes.len();
        if r.first() == Some(&b' ') {
            r = &r[1..];
        }
        r
    }
}

/// Longest decimal-float prefix, per the `strtof`/`num_get` grammar the C++
/// extraction uses (doc §4.3: `12abc` parses as 12): optional sign, digits
/// and/or fraction, optional exponent. Returns `None` when no float prefix
/// exists. Hex floats and `inf`/`nan` words are not accepted (never produced
/// by any known client; doc examples are all decimal).
fn scan_float(t: &[u8]) -> Option<f32> {
    let mut i = usize::from(matches!(t.first(), Some(b'+' | b'-')));
    let d0 = i;
    while i < t.len() && t[i].is_ascii_digit() {
        i += 1;
    }
    let int_len = i - d0;
    let mut frac_len = 0;
    if t.get(i) == Some(&b'.') {
        let mut j = i + 1;
        while j < t.len() && t[j].is_ascii_digit() {
            j += 1;
        }
        frac_len = j - (i + 1);
        if int_len > 0 || frac_len > 0 {
            i = j;
        }
    }
    if int_len + frac_len == 0 {
        return None;
    }
    if matches!(t.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(t.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let e0 = j;
        while j < t.len() && t[j].is_ascii_digit() {
            j += 1;
        }
        if j > e0 {
            i = j;
        }
    }
    // The matched prefix is pure ASCII, and Rust's f32 parser accepts the
    // full strtof decimal grammar with identical rounding.
    std::str::from_utf8(&t[..i]).ok()?.parse::<f32>().ok()
}

/// Longest decimal-integer prefix with C++11 `num_get` out-of-range
/// saturation (failbit + `INT_MAX`/`INT_MIN` stored). Returns `None` when no
/// digits are present.
fn scan_int(t: &[u8]) -> Option<i32> {
    let mut i = 0usize;
    let neg = match t.first() {
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
    let d0 = i;
    let mut v: i64 = 0;
    while i < t.len() && t[i].is_ascii_digit() {
        v = v.saturating_mul(10).saturating_add(i64::from(t[i] - b'0'));
        i += 1;
    }
    if i == d0 {
        return None;
    }
    if neg {
        v = -v;
    }
    Some(v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
}

/// `istringstream >> float` semantics (doc §4.3): missing token → `missing`
/// (failed sentry leaves the pre-initialized default untouched); present but
/// non-numeric → 0.0 (C++11 `num_get` writes 0 on parse failure).
fn extract_f32(token: Option<&[u8]>, missing: f32) -> f32 {
    match token {
        None => missing,
        Some(t) => scan_float(t).unwrap_or(0.0),
    }
}

/// `istringstream >> int` semantics (doc §4.4/§4.5): missing token →
/// `missing`; present but non-numeric → 0.
fn extract_i32(token: Option<&[u8]>, missing: i32) -> i32 {
    match token {
        None => missing,
        Some(t) => scan_int(t).unwrap_or(0),
    }
}

fn skip_leading_ws(b: &[u8]) -> &[u8] {
    let n = b.iter().take_while(|&&c| is_ws(c)).count();
    &b[n..]
}

/// C `atoi` semantics for `set fps` (doc §4.6): skip leading whitespace,
/// longest integer prefix, 0 when none. Overflow saturates (upstream UB).
fn atoi(value: &[u8]) -> i32 {
    scan_int(skip_leading_ws(value)).unwrap_or(0)
}

/// C `atof` semantics for `set renderscale` (doc §4.6): skip leading
/// whitespace, longest float prefix, 0.0 when none.
fn atof(value: &[u8]) -> f32 {
    scan_float(skip_leading_ws(value)).unwrap_or(0.0)
}

/// `set` boolean values (doc §4.6): the exact strings `true` or `1` mean on;
/// **anything else** (including `TRUE`, `yes`, or a value with a trailing
/// space — the value is rest-of-line) means off.
fn set_bool(value: &[u8]) -> bool {
    value == b"true" || value == b"1"
}

fn token_string(cur: &mut Cursor<'_>) -> String {
    cur.token()
        .map(|t| String::from_utf8_lossy(t).into_owned())
        .unwrap_or_default()
}

fn rest_string(cur: &mut Cursor<'_>) -> String {
    String::from_utf8_lossy(cur.rest()).into_owned()
}

fn rest_path(cur: &mut Cursor<'_>) -> PathBuf {
    // Paths are raw bytes on Linux; preserve them exactly (doc §2: no
    // quoting/escaping exists, values are raw bytes).
    PathBuf::from(OsStr::from_bytes(cur.rest()).to_os_string())
}

fn parse_set(cur: &mut Cursor<'_>) -> Request {
    // Missing key ⇒ C++ extracts "" ⇒ handleSet falls through the key table
    // ⇒ `error\n` (doc §4.6).
    let Some(key) = cur.token() else {
        return Request::Rejected;
    };
    let opt = match key {
        b"fps" => SetOption::Fps(atoi(cur.rest()).max(1)),
        b"noautomute" => SetOption::NoAutomute(set_bool(cur.rest())),
        b"disablemouse" => SetOption::DisableMouse(set_bool(cur.rest())),
        b"disableparallax" => SetOption::DisableParallax(set_bool(cur.rest())),
        b"nofullscreenpause" => SetOption::NoFullscreenPause(set_bool(cur.rest())),
        b"renderscale" => SetOption::RenderScale(atof(cur.rest()).clamp(0.5, 2.0)),
        b"audiodevice" => {
            let v = rest_string(cur);
            SetOption::AudioDevice(if v == "default" { String::new() } else { v })
        }
        _ => return Request::Rejected,
    };
    Request::Command(Command::Set(opt))
}

/// Parse one request line (the bytes up to, excluding, the first `\n`) into a
/// typed [`Request`]. Total: never panics on any input (SPEC V9).
///
/// Dispatch order and byte comparisons match `ControlSocket::handle`
/// (doc §4 summary table); command tokens are case-sensitive.
pub fn parse_request(line: &[u8]) -> Request {
    if line.is_empty() {
        // doc §2 step 5: empty request ⇒ no response bytes.
        return Request::Empty;
    }
    let mut cur = Cursor::new(line);
    // Whitespace-only line: `>>` fails, command stays "" ⇒ `unknown command`.
    let Some(cmd) = cur.token() else {
        return Request::Unknown;
    };
    match cmd {
        b"ping" => Request::Ping,
        b"status" => Request::Status,
        // kirie extension (docs/compat-socket.md §11): optional screen token.
        b"getproperties" => Request::GetProperties {
            screen: cur.token().map(|t| String::from_utf8_lossy(t).into_owned()),
        },
        b"speed" => {
            let v = extract_f32(cur.token(), 1.0);
            // WallpaperApplication.cpp:1277-1280 (doc §4.3): v ≤ 0 ⇒ 1.0.
            Request::Command(Command::Speed(if v <= 0.0 { 1.0 } else { v }))
        }
        b"volume" => Request::Command(Command::Volume(extract_i32(cur.token(), 128))),
        b"mute" => Request::Command(Command::Mute(extract_i32(cur.token(), 0) != 0)),
        b"set" => parse_set(&mut cur),
        b"bg" => {
            let screen = token_string(&mut cur);
            let path = rest_path(&mut cur);
            Request::Command(Command::Bg { screen, path })
        }
        b"preload" => Request::Command(Command::Preload {
            path: rest_path(&mut cur),
        }),
        b"property" => {
            let screen = token_string(&mut cur);
            let key = token_string(&mut cur);
            let value = rest_string(&mut cur);
            Request::Command(Command::Property { screen, key, value })
        }
        b"scaling" => {
            let screen = token_string(&mut cur);
            let mode = match cur.token() {
                Some(b"stretch") => ScalingMode::Stretch,
                Some(b"fit") => ScalingMode::Fit,
                Some(b"fill") => ScalingMode::Fill,
                Some(b"default") => ScalingMode::Default,
                // doc §4.10: any other mode string ⇒ `error\n`, nothing stored.
                _ => return Request::Rejected,
            };
            Request::Command(Command::Scaling { screen, mode })
        }
        b"clamp" => {
            let screen = token_string(&mut cur);
            let mode = match cur.token() {
                Some(b"clamp") => ClampMode::Clamp,
                Some(b"border") => ClampMode::Border,
                Some(b"repeat") => ClampMode::Repeat,
                // doc §4.11: same error semantics as `scaling`.
                _ => return Request::Rejected,
            };
            Request::Command(Command::Clamp { screen, mode })
        }
        b"screenshot" => Request::Command(Command::Screenshot {
            path: rest_path(&mut cur),
        }),
        _ => Request::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(line: &[u8]) -> Command {
        match parse_request(line) {
            Request::Command(c) => c,
            other => panic!(
                "expected Command for {:?}, got {other:?}",
                String::from_utf8_lossy(line)
            ),
        }
    }

    #[test]
    fn empty_and_whitespace_lines() {
        assert_eq!(parse_request(b""), Request::Empty);
        // Whitespace-only is NOT empty: C++ empty-check is on the raw buffer,
        // then command extraction fails ⇒ unknown command (doc §2/§9).
        assert_eq!(parse_request(b"   "), Request::Unknown);
        assert_eq!(parse_request(b"\r"), Request::Unknown);
    }

    #[test]
    fn ping_and_status() {
        assert_eq!(parse_request(b"ping"), Request::Ping);
        assert_eq!(parse_request(b"ping with extra args"), Request::Ping);
        assert_eq!(parse_request(b"ping\r"), Request::Ping); // \r is stream ws (doc §2)
        assert_eq!(parse_request(b"  ping"), Request::Ping); // leading ws skipped by >>
        assert_eq!(parse_request(b"status"), Request::Status);
        assert_eq!(parse_request(b"PING"), Request::Unknown); // case-sensitive
        assert_eq!(parse_request(b"frobnicate"), Request::Unknown); // doc §9
    }

    #[test]
    fn getproperties_optional_screen() {
        // kirie extension (docs/compat-socket.md §11): bare form ⇒ default
        // screen; a token selects a screen. Extra tokens are ignored.
        assert_eq!(
            parse_request(b"getproperties"),
            Request::GetProperties { screen: None }
        );
        assert_eq!(
            parse_request(b"getproperties HDMI-A-1"),
            Request::GetProperties {
                screen: Some("HDMI-A-1".into())
            }
        );
        assert_eq!(
            parse_request(b"getproperties DP-2 extra"),
            Request::GetProperties {
                screen: Some("DP-2".into())
            }
        );
        // Case-sensitive like every other command token.
        assert_eq!(parse_request(b"GetProperties"), Request::Unknown);
    }

    #[test]
    fn speed_coercions() {
        // doc §4.3: missing → default 1.0; non-numeric → 0 → 1.0; ≤0 → 1.0.
        assert_eq!(cmd(b"speed"), Command::Speed(1.0));
        assert_eq!(cmd(b"speed abc"), Command::Speed(1.0));
        assert_eq!(cmd(b"speed 0"), Command::Speed(1.0));
        assert_eq!(cmd(b"speed -2.5"), Command::Speed(1.0));
        assert_eq!(cmd(b"speed 0.5"), Command::Speed(0.5));
        assert_eq!(cmd(b"speed 2"), Command::Speed(2.0));
        assert_eq!(cmd(b"speed 1.5x"), Command::Speed(1.5)); // longest prefix
        assert_eq!(cmd(b"speed .5"), Command::Speed(0.5));
        assert_eq!(cmd(b"speed 1e1"), Command::Speed(10.0));
    }

    #[test]
    fn volume_coercions() {
        // doc §4.4: missing → 128; non-numeric → 0; no clamping.
        assert_eq!(cmd(b"volume"), Command::Volume(128));
        assert_eq!(cmd(b"volume abc"), Command::Volume(0));
        assert_eq!(cmd(b"volume 15"), Command::Volume(15));
        assert_eq!(cmd(b"volume -5"), Command::Volume(-5));
        assert_eq!(cmd(b"volume 500"), Command::Volume(500));
        // Out-of-range saturates like C++11 num_get.
        assert_eq!(cmd(b"volume 99999999999999999999"), Command::Volume(i32::MAX));
        assert_eq!(cmd(b"volume -99999999999999999999"), Command::Volume(i32::MIN));
    }

    #[test]
    fn mute_coercions() {
        // doc §4.5: missing → 0 (unmute); any nonzero → mute.
        assert_eq!(cmd(b"mute"), Command::Mute(false));
        assert_eq!(cmd(b"mute 0"), Command::Mute(false));
        assert_eq!(cmd(b"mute 1"), Command::Mute(true));
        assert_eq!(cmd(b"mute 2"), Command::Mute(true));
        assert_eq!(cmd(b"mute -1"), Command::Mute(true));
        assert_eq!(cmd(b"mute x"), Command::Mute(false));
    }

    #[test]
    fn set_keys_and_coercions() {
        assert_eq!(parse_request(b"set"), Request::Rejected);
        assert_eq!(parse_request(b"set bogus 1"), Request::Rejected);
        // fps: max(1, atoi) — doc §4.6.
        assert_eq!(cmd(b"set fps 30"), Command::Set(SetOption::Fps(30)));
        assert_eq!(cmd(b"set fps abc"), Command::Set(SetOption::Fps(1)));
        assert_eq!(cmd(b"set fps -5"), Command::Set(SetOption::Fps(1)));
        assert_eq!(cmd(b"set fps 0"), Command::Set(SetOption::Fps(1)));
        // atoi skips leading ws: two spaces after key leave " 60" as value.
        assert_eq!(cmd(b"set fps  60"), Command::Set(SetOption::Fps(60)));
        // bools: exact `true`/`1` only — doc §4.6.
        assert_eq!(
            cmd(b"set disablemouse true"),
            Command::Set(SetOption::DisableMouse(true))
        );
        assert_eq!(
            cmd(b"set disablemouse 1"),
            Command::Set(SetOption::DisableMouse(true))
        );
        assert_eq!(
            cmd(b"set disablemouse TRUE"),
            Command::Set(SetOption::DisableMouse(false))
        );
        assert_eq!(
            cmd(b"set disablemouse yes"),
            Command::Set(SetOption::DisableMouse(false))
        );
        // Rest-of-line keeps a trailing space ⇒ "true " ≠ "true" ⇒ off.
        assert_eq!(
            cmd(b"set disablemouse true "),
            Command::Set(SetOption::DisableMouse(false))
        );
        assert_eq!(
            cmd(b"set noautomute true"),
            Command::Set(SetOption::NoAutomute(true))
        );
        assert_eq!(
            cmd(b"set disableparallax 1"),
            Command::Set(SetOption::DisableParallax(true))
        );
        assert_eq!(
            cmd(b"set nofullscreenpause true"),
            Command::Set(SetOption::NoFullscreenPause(true))
        );
        // renderscale: atof, clamp [0.5, 2.0] — doc §4.6.
        assert_eq!(
            cmd(b"set renderscale 1.06"),
            Command::Set(SetOption::RenderScale(1.06))
        );
        assert_eq!(
            cmd(b"set renderscale 5"),
            Command::Set(SetOption::RenderScale(2.0))
        );
        assert_eq!(
            cmd(b"set renderscale 0.1"),
            Command::Set(SetOption::RenderScale(0.5))
        );
        assert_eq!(
            cmd(b"set renderscale abc"),
            Command::Set(SetOption::RenderScale(0.5))
        );
        // audiodevice: rest-of-line; `default` → "".
        assert_eq!(
            cmd(b"set audiodevice default"),
            Command::Set(SetOption::AudioDevice(String::new()))
        );
        assert_eq!(
            cmd(b"set audiodevice alsa_output.pci 0000"),
            Command::Set(SetOption::AudioDevice("alsa_output.pci 0000".into()))
        );
    }

    #[test]
    fn rest_of_line_semantics() {
        // Exactly one leading space stripped (doc §2): a second space is
        // part of the value.
        assert_eq!(
            cmd(b"bg HDMI-A-1 /path/with spaces"),
            Command::Bg {
                screen: "HDMI-A-1".into(),
                path: PathBuf::from("/path/with spaces")
            }
        );
        assert_eq!(
            cmd(b"bg HDMI-A-1  /padded"),
            Command::Bg {
                screen: "HDMI-A-1".into(),
                path: PathBuf::from(" /padded")
            }
        );
        // \r is KEPT in rest-of-line args (doc §2).
        assert_eq!(
            cmd(b"bg HDMI-A-1 /a\r"),
            Command::Bg {
                screen: "HDMI-A-1".into(),
                path: PathBuf::from("/a\r")
            }
        );
        // Missing args degrade to empty strings, exactly like failed C++
        // stream extractions; the app then answers error.
        assert_eq!(
            cmd(b"bg"),
            Command::Bg {
                screen: String::new(),
                path: PathBuf::new()
            }
        );
    }

    #[test]
    fn property_values() {
        assert_eq!(
            cmd(b"property HDMI-A-1 outline 0.36585 0.04268 0.43902"),
            Command::Property {
                screen: "HDMI-A-1".into(),
                key: "outline".into(),
                value: "0.36585 0.04268 0.43902".into(),
            }
        );
        assert_eq!(
            cmd(b"property HDMI-A-1 bloom true"),
            Command::Property {
                screen: "HDMI-A-1".into(),
                key: "bloom".into(),
                value: "true".into()
            }
        );
        assert_eq!(
            cmd(b"property"),
            Command::Property {
                screen: String::new(),
                key: String::new(),
                value: String::new()
            }
        );
    }

    #[test]
    fn scaling_and_clamp_modes() {
        for (s, m) in [
            ("stretch", ScalingMode::Stretch),
            ("fit", ScalingMode::Fit),
            ("fill", ScalingMode::Fill),
            ("default", ScalingMode::Default),
        ] {
            assert_eq!(
                cmd(format!("scaling HDMI-A-1 {s}").as_bytes()),
                Command::Scaling {
                    screen: "HDMI-A-1".into(),
                    mode: m
                }
            );
        }
        for (s, m) in [
            ("clamp", ClampMode::Clamp),
            ("border", ClampMode::Border),
            ("repeat", ClampMode::Repeat),
        ] {
            assert_eq!(
                cmd(format!("clamp HDMI-A-1 {s}").as_bytes()),
                Command::Clamp {
                    screen: "HDMI-A-1".into(),
                    mode: m
                }
            );
        }
        // doc §9 verified live: bogus mode ⇒ error.
        assert_eq!(parse_request(b"scaling HDMI-A-1 bogusmode"), Request::Rejected);
        assert_eq!(parse_request(b"scaling HDMI-A-1"), Request::Rejected);
        assert_eq!(parse_request(b"clamp HDMI-A-1 nope"), Request::Rejected);
        assert_eq!(parse_request(b"clamp"), Request::Rejected);
    }

    #[test]
    fn preload_and_screenshot() {
        assert_eq!(
            cmd(b"preload /w/dir"),
            Command::Preload {
                path: PathBuf::from("/w/dir")
            }
        );
        assert_eq!(
            cmd(b"screenshot /tmp/a b.png"),
            Command::Screenshot {
                path: PathBuf::from("/tmp/a b.png")
            }
        );
        // Empty path is delivered; the app answers error (doc §4.12).
        assert_eq!(cmd(b"screenshot"), Command::Screenshot { path: PathBuf::new() });
    }

    #[test]
    fn non_utf8_paths_survive() {
        // Raw (non-UTF-8) path bytes must round-trip into the PathBuf (V9:
        // malformed input never panics, and paths are bytes on Linux).
        let line = b"bg HDMI-A-1 /weird/\xff\xfe/dir";
        let Command::Bg { path, .. } = cmd(line) else {
            panic!("expected bg")
        };
        assert_eq!(path.as_os_str().as_bytes(), b"/weird/\xff\xfe/dir");
    }

    /// SPEC V13 adapted: every `Command` variant (and every `SetOption`
    /// variant) round-trips `parse(to_request_line(c)) == c`.
    #[test]
    fn round_trip_every_variant() {
        let all = [
            Command::Speed(0.5),
            Command::Speed(1.0),
            Command::Speed(2.25),
            Command::Volume(0),
            Command::Volume(64),
            Command::Volume(-3),
            Command::Volume(500),
            Command::Mute(true),
            Command::Mute(false),
            Command::Set(SetOption::Fps(30)),
            Command::Set(SetOption::NoAutomute(true)),
            Command::Set(SetOption::NoAutomute(false)),
            Command::Set(SetOption::DisableMouse(true)),
            Command::Set(SetOption::DisableParallax(false)),
            Command::Set(SetOption::NoFullscreenPause(true)),
            Command::Set(SetOption::RenderScale(1.06)),
            Command::Set(SetOption::AudioDevice(String::new())),
            Command::Set(SetOption::AudioDevice("alsa_output.pci 0000_00.analog".into())),
            Command::Bg {
                screen: "HDMI-A-1".into(),
                path: PathBuf::from("/path/with spaces/dir"),
            },
            Command::Bg {
                screen: "span:DP-1".into(),
                path: PathBuf::from(" /leading-space"),
            },
            Command::Preload {
                path: PathBuf::from("/w/431960/3047596375"),
            },
            Command::Property {
                screen: "HDMI-A-1".into(),
                key: "outline".into(),
                value: "0.36585 0.04268 0.43902".into(),
            },
            Command::Property {
                screen: "default".into(),
                key: "bloom".into(),
                value: String::new(),
            },
            Command::Scaling {
                screen: "HDMI-A-1".into(),
                mode: ScalingMode::Stretch,
            },
            Command::Scaling {
                screen: "HDMI-A-1".into(),
                mode: ScalingMode::Fit,
            },
            Command::Scaling {
                screen: "HDMI-A-1".into(),
                mode: ScalingMode::Fill,
            },
            Command::Scaling {
                screen: "HDMI-A-1".into(),
                mode: ScalingMode::Default,
            },
            Command::Clamp {
                screen: "HDMI-A-1".into(),
                mode: ClampMode::Clamp,
            },
            Command::Clamp {
                screen: "HDMI-A-1".into(),
                mode: ClampMode::Border,
            },
            Command::Clamp {
                screen: "HDMI-A-1".into(),
                mode: ClampMode::Repeat,
            },
            Command::Screenshot {
                path: PathBuf::from("/tmp/shot.png"),
            },
        ];
        for c in all {
            let line = c.to_request_line();
            assert_eq!(
                parse_request(&line),
                Request::Command(c.clone()),
                "round-trip failed for line {:?}",
                String::from_utf8_lossy(&line)
            );
        }
    }
}
