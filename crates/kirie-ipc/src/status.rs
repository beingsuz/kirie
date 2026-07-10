//! `status` response formatting (docs/compat-socket.md §4.2).

use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

/// Immutable snapshot of the app state the `status` command reports
/// (doc §4.2). Built by the app, sent over the reply channel by value —
/// never shared (SPEC V3).
#[derive(Debug, Clone, PartialEq)]
pub struct StatusSnapshot {
    /// Current global playback speed (doc §4.3). The app stores the
    /// already-coerced value delivered by [`crate::Command::Speed`].
    pub speed: f32,
    /// One entry per registered background, in any order — the formatter
    /// emits them lexicographically by key bytes, matching the C++
    /// `std::map` iteration order (doc §4.2).
    pub screens: Vec<ScreenStatus>,
}

/// One `screen=<key> bg=<path>` line of the `status` response (doc §4.2).
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenStatus {
    /// Registration key: output name (`HDMI-A-1`), `default` (windowed), or
    /// `span:<first-screen>` (doc §4).
    pub screen: String,
    /// Recorded background path for the key; `None` (or empty) renders as
    /// `bg=` followed by the newline (doc §4.2).
    pub bg: Option<PathBuf>,
}

/// Render the full multi-line `status` body (doc §4.2):
///
/// ```text
/// speed=<float>\n
/// screen=<key> bg=<path>\n     (per screen, lexicographic by key)
/// ```
///
/// Every line is LF-terminated; there is no `ok` trailer — connection close
/// ends the response.
pub(crate) fn format_status(snapshot: &StatusSnapshot) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"speed=");
    out.extend_from_slice(format_speed(snapshot.speed).as_bytes());
    out.push(b'\n');
    let mut screens: Vec<&ScreenStatus> = snapshot.screens.iter().collect();
    // std::map<std::string, ...> iterates lexicographically by byte
    // comparison; Rust's byte-slice Ord is the same relation (doc §4.2).
    screens.sort_by(|a, b| a.screen.as_bytes().cmp(b.screen.as_bytes()));
    for sc in screens {
        out.extend_from_slice(b"screen=");
        out.extend_from_slice(sc.screen.as_bytes());
        out.extend_from_slice(b" bg=");
        if let Some(p) = &sc.bg {
            out.extend_from_slice(p.as_os_str().as_bytes());
        }
        out.push(b'\n');
    }
    out
}

/// Default `std::stringstream <<` float formatting (doc §4.2): printf `%g`
/// with precision 6 — six significant digits, trailing zeros stripped, no
/// forced decimal point, scientific notation outside `[1e-4, 1e6)`.
/// `1.0f` → `1`, `0.5f` → `0.5`.
pub(crate) fn format_speed(value: f32) -> String {
    let v = f64::from(value); // C++ inserts float promoted to double
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    // %g picks the style from the %e exponent at precision P-1 = 5.
    let sci = format!("{v:.5e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let x: i32 = exp.parse().unwrap_or(0);
    if (-4..6).contains(&x) {
        let prec = (5 - x).max(0) as usize;
        trim_g(format!("{v:.prec$}"))
    } else {
        let m = trim_g(mantissa.to_string());
        format!("{m}e{}{:02}", if x < 0 { '-' } else { '+' }, x.abs())
    }
}

/// `%g` trailing cleanup: drop trailing zeros after the point, then a bare
/// trailing point.
fn trim_g(mut s: String) -> String {
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_matches_cpp_stream_default_format() {
        // doc §4.2: 1.0f → "1", 0.5f → "0.5" (live-verified fixture values).
        assert_eq!(format_speed(1.0), "1");
        assert_eq!(format_speed(0.5), "0.5");
        assert_eq!(format_speed(0.25), "0.25");
        assert_eq!(format_speed(2.0), "2");
        assert_eq!(format_speed(1.75), "1.75");
        assert_eq!(format_speed(100.0), "100");
        assert_eq!(format_speed(0.1), "0.1"); // %g rounds the f32 0.1 back to 0.1
        assert_eq!(format_speed(1.0 / 3.0), "0.333333"); // 6 significant digits
        assert_eq!(format_speed(123456.0), "123456");
        assert_eq!(format_speed(1_000_000.0), "1e+06");
        assert_eq!(format_speed(0.0001), "0.0001");
        assert_eq!(format_speed(0.00001), "1e-05");
        assert_eq!(format_speed(0.0), "0");
    }

    #[test]
    fn status_body_single_screen_matches_live_capture() {
        // fixtures/socket-live-capture.txt, byte-verified (doc §9).
        let snap = StatusSnapshot {
            speed: 1.0,
            screens: vec![ScreenStatus {
                screen: "HDMI-A-1".into(),
                bg: Some(PathBuf::from(
                    "/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375",
                )),
            }],
        };
        assert_eq!(
            format_status(&snap),
            b"speed=1\nscreen=HDMI-A-1 bg=/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375\n"
        );
    }

    #[test]
    fn status_screens_sorted_lexicographically_by_bytes() {
        // "DP-10" < "DP-2" byte-wise ('1' < '2'), like std::map (doc §4.2).
        let snap = StatusSnapshot {
            speed: 0.5,
            screens: vec![
                ScreenStatus {
                    screen: "HDMI-A-1".into(),
                    bg: Some(PathBuf::from("/c")),
                },
                ScreenStatus {
                    screen: "DP-2".into(),
                    bg: Some(PathBuf::from("/b")),
                },
                ScreenStatus {
                    screen: "DP-10".into(),
                    bg: None,
                },
            ],
        };
        assert_eq!(
            format_status(&snap),
            b"speed=0.5\nscreen=DP-10 bg=\nscreen=DP-2 bg=/b\nscreen=HDMI-A-1 bg=/c\n"
        );
    }
}
