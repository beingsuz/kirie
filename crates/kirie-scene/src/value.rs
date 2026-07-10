//! Scalar/vector/color value encodings shared by every scene field.
//!
//! Spec: docs/format-scene-json.md §2. Wallpaper Engine stores vectors as
//! space-separated strings (§2.1), colors in several spellings (§2.2), and
//! bridges loose scalar typing through `coerce<T>` (§2.3). The `strtof`-family
//! prefix parsers here match the C++ `VectorBuilder`/`coerce<T>` byte-for-byte
//! and are the same fuzz-hardened routines used by `kirie-formats::project`.

use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use thiserror::Error;

/// A 2-component vector (`"x y"`), stored `[x, y]`.
pub type Vec2 = [f32; 2];
/// A 3-component vector (`"x y z"`), stored `[x, y, z]`.
pub type Vec3 = [f32; 3];
/// An RGBA color, each channel nominally 0..1 (docs/format-scene-json.md §2.2).
pub type Color = [f32; 4];

/// Error parsing a vector string (docs/format-scene-json.md §2.1).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VecError {
    /// A fixed-length vector was requested but the string had a different
    /// component count — a load error in the C++ (`VectorBuilder.h:81–104`).
    #[error("vector has {found} components (expected {expected})")]
    ComponentCount {
        /// Components requested by the field.
        expected: usize,
        /// Components actually found (space count + 1, capped at 4).
        found: usize,
    },
}

/// Error parsing a color value (docs/format-scene-json.md §2.2).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ColorError {
    /// A `#`-hex digit string was not 3, 4, 6 or 8 digits (`ColorBuilder.cpp:39`).
    #[error("unsupported hex color length {len} (expected 3, 4, 6 or 8 digits)")]
    HexLength {
        /// Number of hex digits after `#`.
        len: usize,
    },
    /// A `#`-hex string held non-hex characters (`std::stoi` throws in the C++).
    #[error("invalid hex digits in color {digits:?}")]
    HexDigits {
        /// The offending digit string.
        digits: String,
    },
    /// A space-separated color had a component count other than 3 or 4
    /// (`Invalid color value`, `ColorBuilder.cpp:52–56`).
    #[error("color has {count} components (expected 3 or 4)")]
    ComponentCount {
        /// Number of components found.
        count: usize,
    },
}

/// Parse a space-separated vector string into up to four components
/// (docs/format-scene-json.md §2.1, `VectorBuilder.h:34–117`).
///
/// The separator is exactly one ASCII space; the component count is the space
/// count plus one, capped at four. Consecutive spaces therefore yield empty
/// tokens that `strtof` reads as `0.0`, exactly like the C++ reader.
pub fn parse_vec_components(s: &str) -> Vec<f32> {
    s.split(' ').take(4).map(strtof).collect()
}

/// Parse a fixed-length vector, validating the component count strictly
/// (docs/format-scene-json.md §2.1: too few/too many is a load error).
pub fn parse_vec<const N: usize>(s: &str) -> Result<[f32; N], VecError> {
    let parts = parse_vec_components(s);
    let mut out = [0.0; N];
    if parts.len() != N {
        return Err(VecError::ComponentCount {
            expected: N,
            found: parts.len(),
        });
    }
    out.copy_from_slice(&parts);
    Ok(out)
}

/// Parse a color value in any of the spellings of docs/format-scene-json.md §2.2.
///
/// `alpha` supplies the alpha of 3-component values; `force_float` disables the
/// int-0..255 path (project-property colors force it, scene fields do not —
/// §2.2 pitfall). Deliberate clean-implementation deviations the spec
/// prescribes: `#rrggbb` gains an `ff` alpha byte (the C++ mis-splits it), and
/// the hex parse is 64-bit (the C++ `std::stoi` throws for values ≥ `0x80000000`).
pub fn parse_color(s: &str, alpha: f32, force_float: bool) -> Result<Color, ColorError> {
    // §2.2: commas are replaced by spaces first (`ColorBuilder.cpp:18–21`).
    let normalized = s.replace(',', " ");

    // §2.2: `#`-prefixed CSS hex (`ColorBuilder.cpp:24–50`).
    if let Some(digits) = normalized.strip_prefix('#') {
        let expanded: String = match digits.len() {
            3 => {
                let mut e: String = digits.chars().flat_map(|c| [c, c]).collect();
                e.push_str("ff");
                e
            }
            4 => digits.chars().flat_map(|c| [c, c]).collect(),
            6 => format!("{digits}ff"),
            8 => digits.to_owned(),
            len => return Err(ColorError::HexLength { len }),
        };
        let v = u32::from_str_radix(&expanded, 16).map_err(|_| ColorError::HexDigits {
            digits: digits.to_owned(),
        })?;
        let byte = |shift: u32| ((v >> shift) & 0xff) as f32 / 255.0;
        return Ok([byte(24), byte(16), byte(8), byte(0)]);
    }

    // §2.2: space-separated 3- or 4-component vector.
    let parts: Vec<&str> = normalized.split(' ').collect();
    // §2.2 pitfall: int vs float is decided by "does the string contain a `.`".
    let is_float = force_float || normalized.contains('.');
    let scale = |raw: &str| -> f32 {
        if is_float {
            strtof(raw)
        } else {
            strtoi(raw) as f32 / 255.0
        }
    };
    match parts.as_slice() {
        [r, g, b] => Ok([scale(r), scale(g), scale(b), alpha]),
        [r, g, b, a] => Ok([scale(r), scale(g), scale(b), scale(a)]),
        other => Err(ColorError::ComponentCount { count: other.len() }),
    }
}

/// The canonical §2.2 constant White = (1,1,1,1).
pub const WHITE: Color = [1.0, 1.0, 1.0, 1.0];
/// The canonical §2.2 constant Black = (0,0,0,1).
pub const BLACK: Color = [0.0, 0.0, 0.0, 1.0];

/// A decoded dynamic value literal (docs/format-scene-json.md §2.4).
///
/// This is the one canonical type a user-setting literal decodes to; the render
/// side reads it as any type via the conversions in [`DynamicValue`]'s methods
/// (vecN→float takes `.x`, float→bool is `!= 0`, color→int is `255*r`, …).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "v", rename_all = "lowercase")]
pub enum DynamicValue {
    /// A boolean literal.
    Bool(bool),
    /// An integer literal.
    Int(i64),
    /// A floating-point literal.
    Float(f64),
    /// A string literal that did not parse as a single float.
    Str(String),
    /// A 2/3/4-component vector literal (`.len()` is 2, 3 or 4).
    Vec(Vec<f32>),
    /// A color literal (RGBA, `color`-typed field).
    Color(Color),
    /// A JSON `null` literal.
    Null,
}

impl DynamicValue {
    /// Decode a raw JSON value literal by type (docs/format-scene-json.md §2.4).
    ///
    /// `color_expected` routes single strings through the §2.2 color parser;
    /// otherwise a 1-token string is a float when the *whole* token parses via
    /// `stof`, else a plain string, and multi-token strings become vectors.
    pub fn decode(value: &Value, color_expected: bool) -> Self {
        match value {
            Value::Null => DynamicValue::Null,
            Value::Bool(b) => DynamicValue::Bool(*b),
            Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    DynamicValue::Int(n.as_i64().unwrap_or(0))
                } else {
                    DynamicValue::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            Value::String(s) => {
                if color_expected {
                    return parse_color(s, 1.0, false)
                        .map_or_else(|_| DynamicValue::Str(s.clone()), DynamicValue::Color);
                }
                let tokens = s.split(' ').count();
                match tokens {
                    0 | 1 => match parse_whole_f32(s) {
                        Some(f) => DynamicValue::Float(f64::from(f)),
                        None => DynamicValue::Str(s.clone()),
                    },
                    _ => DynamicValue::Vec(parse_vec_components(s)),
                }
            }
            // Arrays/objects are not value literals in this position; keep the
            // string form so nothing is silently dropped.
            other => DynamicValue::Str(other.to_string()),
        }
    }

    /// Read this value as a float (docs/format-scene-json.md §2.4: vecN→`.x`,
    /// bool→0/1, color→`.r`, string→`stod` prefix or 0).
    pub fn as_f32(&self) -> f32 {
        match self {
            DynamicValue::Bool(b) => f32::from(*b),
            DynamicValue::Int(i) => *i as f32,
            DynamicValue::Float(f) => *f as f32,
            DynamicValue::Str(s) => strtof(s),
            DynamicValue::Vec(v) => v.first().copied().unwrap_or(0.0),
            DynamicValue::Color(c) => c[0],
            DynamicValue::Null => 0.0,
        }
    }

    /// Read this value as a bool (docs/format-scene-json.md §2.4: float→`!= 0`).
    pub fn as_bool(&self) -> bool {
        match self {
            DynamicValue::Bool(b) => *b,
            DynamicValue::Null => false,
            DynamicValue::Str(s) => matches!(s.as_str(), "1" | "true" | "True" | "TRUE"),
            other => other.as_f32() != 0.0,
        }
    }

    /// Read this value as its string form.
    pub fn as_string(&self) -> String {
        match self {
            DynamicValue::Str(s) => s.clone(),
            DynamicValue::Bool(b) => b.to_string(),
            DynamicValue::Int(i) => i.to_string(),
            DynamicValue::Float(f) => f.to_string(),
            DynamicValue::Vec(v) => v.iter().map(f32::to_string).collect::<Vec<_>>().join(" "),
            DynamicValue::Color(c) => format!("{} {} {} {}", c[0], c[1], c[2], c[3]),
            DynamicValue::Null => String::new(),
        }
    }
}

// ---- §2.3 loose scalar coercion (`coerce<T>`, JSON.h:41–79) ----------------

/// §2.3 coercion, bool target: bool as-is; number ≠ 0; string ∈
/// {"1","true","True","TRUE"}. `None` = not coercible (caller keeps the default).
pub fn coerce_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => Some(n.as_f64().unwrap_or(f64::NAN) != 0.0),
        Value::String(s) => Some(matches!(s.as_str(), "1" | "true" | "True" | "TRUE")),
        _ => None,
    }
}

/// §2.3 coercion, float target: number as-is; bool → 0/1; string → `stod`
/// prefix with failure → 0. `None` = not coercible.
pub fn coerce_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            let prefix = float_prefix(skip_c_whitespace(s));
            let parsed = prefix.parse::<f64>().unwrap_or(0.0);
            Some(if parsed.is_infinite() && !is_infinity_token(prefix) {
                0.0
            } else {
                parsed
            })
        }
        _ => None,
    }
}

/// §2.3 coercion, integer target: number truncated; bool → 0/1; string →
/// `stoll` prefix with failure → 0. `None` = not coercible.
pub fn coerce_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => Some(n.as_i64().unwrap_or_else(|| n.as_f64().map_or(0, |f| f as i64))),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::String(s) => Some(int_prefix(skip_c_whitespace(s)).parse().unwrap_or(0)),
        _ => None,
    }
}

/// A JSON `Number` for a `u32` unsigned target (`maxcount`, `flags`, …). Numbers
/// truncate; bools map 0/1; strings go through `stoll` then saturate to `u32`.
pub fn coerce_u32(v: &Value) -> Option<u32> {
    coerce_i64(v).map(|i| i.clamp(0, i64::from(u32::MAX)) as u32)
}

// ---- C `strtof`/`strtoll` prefix parsers (VectorBuilder.h:128, JSON.h) ------

/// C `strtof` semantics: skip leading whitespace, convert the longest valid
/// float prefix, yield `0.0` when none converts. Out-of-range values saturate
/// to ±inf like C `strtof`.
pub fn strtof(s: &str) -> f32 {
    float_prefix(skip_c_whitespace(s)).parse().unwrap_or(0.0)
}

/// C `strtol` semantics for an integer field (base 10).
fn strtoi(s: &str) -> i64 {
    int_prefix(skip_c_whitespace(s)).parse().unwrap_or(0)
}

/// `stof` on the *whole* token (docs/format-scene-json.md §2.4 single-token
/// rule): `Some` only when the entire trimmed string is a float literal.
fn parse_whole_f32(s: &str) -> Option<f32> {
    let trimmed = skip_c_whitespace(s);
    let prefix = float_prefix(trimmed);
    if !prefix.is_empty() && prefix.len() == trimmed.trim_end().len() {
        prefix.parse().ok()
    } else {
        None
    }
}

/// Skip the C-locale `isspace` set (space, `\t`, `\n`, `\v`, `\f`, `\r`).
fn skip_c_whitespace(s: &str) -> &str {
    s.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r'])
}

/// Longest prefix of `s` that the C `strtof`/`strtod` grammar converts. Single
/// O(n) scan — a try-every-length loop is quadratic on adversarial input
/// (SPEC.md §V9).
fn float_prefix(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    for token in ["infinity", "inf", "nan"] {
        if s.len() >= i + token.len() && s[i..i + token.len()].eq_ignore_ascii_case(token) {
            return &s[..i + token.len()];
        }
    }
    let mut valid = 0;
    let int_start = i;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    let has_int_digits = i > int_start;
    if has_int_digits {
        valid = i;
    }
    if bytes.get(i) == Some(&b'.') {
        let frac_start = i + 1;
        let mut j = frac_start;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if has_int_digits || j > frac_start {
            valid = j;
            i = j;
        }
    }
    if valid > 0 && matches!(bytes.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(bytes.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let exp_digits_start = j;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if j > exp_digits_start {
            valid = j;
        }
    }
    &s[..valid]
}

/// Longest prefix `strtoll` (base 10) converts: optional sign then digits.
fn int_prefix(s: &str) -> &str {
    let bytes = s.as_bytes();
    let sign = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mut i = sign;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    if i == sign { "" } else { &s[..i] }
}

/// Whether a [`float_prefix`] result is an infinity token (a valid conversion),
/// as opposed to a finite literal that overflowed to ±inf.
fn is_infinity_token(prefix: &str) -> bool {
    let rest = prefix.trim_start_matches(['+', '-']);
    rest.eq_ignore_ascii_case("inf") || rest.eq_ignore_ascii_case("infinity")
}

/// A JSON `Number` from an `f32`, widened losslessly, mapping ±∞ to the
/// largest-magnitude finite double (round-trips back to the same `f32`) and NaN
/// to 0 (docs/format-scene-json.md V13 NaN exception).
pub fn f32_number(v: f32) -> Value {
    let widened = if v == f32::INFINITY {
        f64::MAX
    } else if v == f32::NEG_INFINITY {
        f64::MIN
    } else {
        f64::from(v)
    };
    Value::Number(Number::from_f64(widened).unwrap_or_else(|| Number::from(0)))
}
