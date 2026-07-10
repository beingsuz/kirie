//! Parsing of the two magic annotation-comment forms in the Wallpaper Engine
//! shader dialect (docs/shader-pipeline.md §2.1, §2.2):
//!
//! - `// [COMBO] {json}` — declares a combo macro with a default and optional
//!   `require` chain (`ShaderUnit::parseComboConfiguration`,
//!   `ShaderUnit.cpp:445-496`).
//! - `uniform T name; // {json}` — declares a bindable parameter or a
//!   texture/sampler slot (`parseParameterConfiguration`, `ShaderUnit.cpp:540-696`).
//!
//! Per SPEC.md §V9 nothing here panics on malformed input: JSON errors and
//! violated invariants surface as typed [`AnnotationError`], mirroring the
//! reference's "error log + skip" behavior where the C++ merely logs
//! (docs/shader-pipeline.md §2.1).

use std::collections::BTreeMap;

use serde_json::Value;
use thiserror::Error;

use crate::reflect::{ParamDefault, ParamType};

/// The exact prefix the reference scans for (docs/shader-pipeline.md §2.1). The
/// variants `[COMBO_OFF]`/`[COMBO_DISABLED]` are deliberately **not** matched —
/// they are comment-out conventions (`ShaderUnit.cpp` searches the literal
/// `"// [COMBO] "`).
pub const COMBO_PREFIX: &str = "// [COMBO] ";

/// Errors from annotation parsing (docs/shader-pipeline.md §2.1, §2.2).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AnnotationError {
    /// The annotation JSON did not parse. The reference logs and skips
    /// (`ShaderUnit.cpp:446-451`); we surface it typed.
    #[error("malformed annotation JSON: {0}")]
    BadJson(String),
    /// A `[COMBO]` object was valid JSON but lacked the required `combo` key —
    /// a hard error in the reference (`JSON.require` throws,
    /// docs/shader-pipeline.md §2.1).
    #[error("[COMBO] annotation missing required \"combo\" key")]
    MissingCombo,
    /// A `[COMBO]` `default` was a float or string where the reference requires
    /// an integer (`ShaderUnit.cpp:486-491`, docs/shader-pipeline.md §2.1).
    #[error("[COMBO] \"default\" must be an integer")]
    NonIntComboDefault,
}

/// A parsed `// [COMBO]` annotation (docs/shader-pipeline.md §2.1). Only the
/// keys the C++ consumes are retained; editor-only keys are ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComboAnnotation {
    /// The macro name to define (uppercased at emission time, §4.3). Stored as
    /// written; the `combo` key value (docs/shader-pipeline.md §2.1).
    pub combo: String,
    /// The default value used when nothing overrides it; absent ⇒ 0
    /// (`ShaderUnit.cpp:485`, docs/shader-pipeline.md §2.1).
    pub default: i32,
    /// `require` chain: if this combo is enabled (non-zero) each listed combo is
    /// forced to the given value (docs/shader-pipeline.md §2.1, §3.4).
    pub require: BTreeMap<String, i32>,
}

/// Parse a single line as a `// [COMBO]` annotation, or `Ok(None)` if the line
/// contains no such annotation (docs/shader-pipeline.md §2.1).
pub fn parse_combo_line(line: &str) -> Result<Option<ComboAnnotation>, AnnotationError> {
    let Some(idx) = line.find(COMBO_PREFIX) else {
        return Ok(None);
    };
    let json = line[idx + COMBO_PREFIX.len()..].trim();
    let value: Value = serde_json::from_str(json).map_err(|e| AnnotationError::BadJson(e.to_string()))?;

    let combo = value
        .get("combo")
        .and_then(Value::as_str)
        .ok_or(AnnotationError::MissingCombo)?
        .to_string();

    // `default`: integer only; absent ⇒ 0 (docs/shader-pipeline.md §2.1).
    let default = match value.get("default") {
        None | Some(Value::Null) => 0,
        Some(Value::Number(n)) => n
            .as_i64()
            .filter(|_| n.as_f64().is_some_and(|f| f.fract() == 0.0))
            .ok_or(AnnotationError::NonIntComboDefault)? as i32,
        Some(_) => return Err(AnnotationError::NonIntComboDefault),
    };

    let mut require = BTreeMap::new();
    if let Some(obj) = value.get("require").and_then(Value::as_object) {
        for (k, v) in obj {
            if let Some(i) = v.as_i64() {
                require.insert(k.clone(), i as i32);
            }
        }
    }

    Ok(Some(ComboAnnotation {
        combo,
        default,
        require,
    }))
}

/// The outcome of parsing a `uniform … // {json}` line (docs/shader-pipeline.md §2.2).
#[derive(Debug, Clone, PartialEq)]
pub enum UniformAnnotation {
    /// A non-sampler bindable parameter (`material` link + typed default).
    Parameter {
        /// Uniform name.
        name: String,
        /// Declared type.
        ty: ParamType,
        /// `material` link, if present (only then does the reference register
        /// it, `ShaderUnit.cpp:690-695`).
        material: Option<String>,
        /// Parsed default, if present.
        default: Option<ParamDefault>,
    },
    /// A `sampler2D`/`sampler2DComparison` texture slot.
    Sampler {
        /// Uniform name.
        name: String,
        /// Default texture name (docs/shader-pipeline.md §2.2 `default`).
        default_texture: Option<String>,
        /// Combo macro gated by this slot (docs/shader-pipeline.md §2.2 `combo`).
        combo: Option<String>,
        /// `require` gating conditions when the slot is empty.
        require: BTreeMap<String, i32>,
        /// `requireany` flag (docs/shader-pipeline.md §2.2).
        require_any: bool,
    },
}

/// Detect and parse a `uniform <type> <name>; // {json}` annotation on a line.
///
/// Detection matches the reference (`ShaderUnit.cpp:105-132`,
/// docs/shader-pipeline.md §2.2): the line must contain `"uniform "`, `"// "`,
/// and `';'`, with the `;` positioned **before** the `//` (so commented-out
/// declarations are skipped). Returns `Ok(None)` when the line is not an
/// annotated uniform. Unknown parameter types are ignored (`Ok(None)`),
/// mirroring "Unknown parameter type" (`ShaderUnit.cpp:685-687`).
pub fn parse_uniform_line(line: &str) -> Result<Option<UniformAnnotation>, AnnotationError> {
    let Some(comment_at) = line.find("// ") else {
        return Ok(None);
    };
    let Some(semi_at) = line.find(';') else {
        return Ok(None);
    };
    if semi_at >= comment_at {
        return Ok(None); // `;` must come before `//`.
    }
    let decl = &line[..semi_at];
    let Some(after_uniform) = decl.find("uniform ") else {
        return Ok(None);
    };
    // tokens between `uniform` and `;`: … <type> <name>
    let tokens: Vec<&str> = decl[after_uniform + "uniform ".len()..]
        .split_whitespace()
        .collect();
    if tokens.len() < 2 {
        return Ok(None);
    }
    // name = token before `;`; type = token before the name (docs §2.2).
    let name = tokens[tokens.len() - 1];
    // strip any array suffix from the name token for typing purposes.
    let name = name.split('[').next().unwrap_or(name).to_string();
    let type_tok = tokens[tokens.len() - 2];

    let json = line[comment_at + 3..].trim();
    let value: Value = serde_json::from_str(json).map_err(|e| AnnotationError::BadJson(e.to_string()))?;

    match type_tok {
        "sampler2D" | "sampler2DComparison" => {
            let default_texture = value.get("default").and_then(Value::as_str).map(str::to_string);
            let combo = value.get("combo").and_then(Value::as_str).map(str::to_string);
            let mut require = BTreeMap::new();
            if let Some(obj) = value.get("require").and_then(Value::as_object) {
                for (k, v) in obj {
                    if let Some(i) = v.as_i64() {
                        require.insert(k.clone(), i as i32);
                    }
                }
            }
            let require_any = value.get("requireany").and_then(Value::as_bool).unwrap_or(false);
            Ok(Some(UniformAnnotation::Sampler {
                name,
                default_texture,
                combo,
                require,
                require_any,
            }))
        }
        other => {
            let Some(ty) = param_type(other) else {
                return Ok(None); // unknown type ignored (docs §2.2).
            };
            let material = value.get("material").and_then(Value::as_str).map(str::to_string);
            let default = parse_param_default(ty, value.get("default"));
            Ok(Some(UniformAnnotation::Parameter {
                name,
                ty,
                material,
                default,
            }))
        }
    }
}

fn param_type(tok: &str) -> Option<ParamType> {
    Some(match tok {
        "float" => ParamType::Float,
        "int" => ParamType::Int,
        "vec2" => ParamType::Vec2,
        "vec3" => ParamType::Vec3,
        "vec4" => ParamType::Vec4,
        _ => return None,
    })
}

/// Parse a `default` value for a non-sampler parameter (docs/shader-pipeline.md §2.2).
/// Vectors come as space-separated strings; scalars as JSON numbers or strings.
/// The reference's `std::stoi` string→int truncation quirk is replicated for the
/// integer case (docs/shader-pipeline.md §2.2: `"0.5"` ⇒ `0`).
fn parse_param_default(ty: ParamType, raw: Option<&Value>) -> Option<ParamDefault> {
    let raw = raw?;
    match ty {
        ParamType::Float => match raw {
            Value::Number(n) => Some(ParamDefault::Scalar(n.as_f64()?)),
            Value::String(s) => s.parse::<f64>().ok().map(ParamDefault::Scalar),
            _ => None,
        },
        ParamType::Int => match raw {
            Value::Number(n) => Some(ParamDefault::Scalar(n.as_i64()? as f64)),
            // std::stoi longest-prefix: "0.5" -> 0 (docs §2.2 quirk).
            Value::String(s) => Some(ParamDefault::Scalar(stoi_prefix(s) as f64)),
            _ => None,
        },
        ParamType::Vec2 | ParamType::Vec3 | ParamType::Vec4 => {
            let s = raw.as_str()?;
            let comps: Vec<f32> = s.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            let want = match ty {
                ParamType::Vec2 => 2,
                ParamType::Vec3 => 3,
                _ => 4,
            };
            // VectorBuilder throws on too few components (docs §2.2); we require
            // at least `want` and truncate extras (vec4's 5th is ignored).
            if comps.len() >= want {
                Some(ParamDefault::Vector(comps[..want].to_vec()))
            } else {
                None
            }
        }
    }
}

/// `std::stoi`-style leading-integer parse: consume an optional sign then digits,
/// stop at the first non-digit (docs/shader-pipeline.md §2.2, matches B3 in
/// SPEC.md §B for stoll semantics on the formats side).
fn stoi_prefix(s: &str) -> i64 {
    let s = s.trim_start();
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        neg = bytes[i] == b'-';
        i += 1;
    }
    let start = i;
    let mut acc: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        acc = acc.saturating_mul(10).saturating_add((bytes[i] - b'0') as i64);
        i += 1;
    }
    if i == start {
        return 0;
    }
    if neg { -acc } else { acc }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_basic_and_default() {
        // docs/shader-pipeline.md §2.1 stock chroma4.frag example.
        let line =
            r#"// [COMBO] {"material":"ui_editor_properties_lighting","combo":"LIGHTING","default":1}"#;
        let c = parse_combo_line(line).unwrap().unwrap();
        assert_eq!(c.combo, "LIGHTING");
        assert_eq!(c.default, 1);
        assert!(c.require.is_empty());
    }

    #[test]
    fn combo_absent_default_is_zero_and_require_chain() {
        let line = r#"// [COMBO] {"combo":"RIMLIGHTING","require":{"LIGHTING":1}}"#;
        let c = parse_combo_line(line).unwrap().unwrap();
        assert_eq!(c.default, 0); // §2.1: absent ⇒ 0
        assert_eq!(c.require.get("LIGHTING"), Some(&1));
    }

    #[test]
    fn combo_missing_key_is_hard_error() {
        // §2.1: valid JSON but no `combo` ⇒ hard error.
        let line = r#"// [COMBO] {"material":"x"}"#;
        assert_eq!(parse_combo_line(line), Err(AnnotationError::MissingCombo));
    }

    #[test]
    fn combo_non_int_default_rejected() {
        // §2.1: float/string default ⇒ hard error.
        assert_eq!(
            parse_combo_line(r#"// [COMBO] {"combo":"X","default":0.5}"#),
            Err(AnnotationError::NonIntComboDefault)
        );
        assert_eq!(
            parse_combo_line(r#"// [COMBO] {"combo":"X","default":"1"}"#),
            Err(AnnotationError::NonIntComboDefault)
        );
    }

    #[test]
    fn combo_off_variants_not_matched() {
        // docs/shader-pipeline.md §2: [COMBO_OFF]/[COMBO_DISABLED] are comment-out
        // conventions, never matched.
        assert_eq!(parse_combo_line(r#"// [COMBO_OFF] {"combo":"X"}"#), Ok(None));
        assert_eq!(parse_combo_line(r#"// [COMBO_DISABLED] {"combo":"X"}"#), Ok(None));
    }

    #[test]
    fn combo_malformed_json_is_typed_error() {
        assert!(matches!(
            parse_combo_line("// [COMBO] {not json}"),
            Err(AnnotationError::BadJson(_))
        ));
    }

    #[test]
    fn uniform_parameter_with_material_and_default() {
        // docs/shader-pipeline.md §2.2 non-sampler example.
        let line = r#"uniform float g_Brightness; // {"material":"Brightness","default":1,"range":[0,2]}"#;
        let u = parse_uniform_line(line).unwrap().unwrap();
        match u {
            UniformAnnotation::Parameter {
                name,
                ty,
                material,
                default,
            } => {
                assert_eq!(name, "g_Brightness");
                assert_eq!(ty, ParamType::Float);
                assert_eq!(material.as_deref(), Some("Brightness"));
                assert_eq!(default, Some(ParamDefault::Scalar(1.0)));
            }
            _ => panic!("expected parameter"),
        }
    }

    #[test]
    fn uniform_vec_default_space_separated() {
        let line = r#"uniform vec3 g_Tint; // {"material":"tint","default":"1 0.5 0"}"#;
        let u = parse_uniform_line(line).unwrap().unwrap();
        match u {
            UniformAnnotation::Parameter { default, .. } => {
                assert_eq!(default, Some(ParamDefault::Vector(vec![1.0, 0.5, 0.0])));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn uniform_int_string_default_truncates() {
        // docs/shader-pipeline.md §2.2 std::stoi quirk: "0.5" -> 0.
        let line = r#"uniform int g_Mode; // {"material":"mode","default":"0.5"}"#;
        let u = parse_uniform_line(line).unwrap().unwrap();
        match u {
            UniformAnnotation::Parameter { default, .. } => {
                assert_eq!(default, Some(ParamDefault::Scalar(0.0)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn uniform_sampler_with_combo_and_default() {
        // docs/shader-pipeline.md §2.2 genericimage2.frag example.
        let line = r#"uniform sampler2D g_Texture1; // {"combo":"NORMALMAP","default":"util/black","requireany":true,"require":{"LIGHTING":1}}"#;
        let u = parse_uniform_line(line).unwrap().unwrap();
        match u {
            UniformAnnotation::Sampler {
                name,
                default_texture,
                combo,
                require_any,
                require,
            } => {
                assert_eq!(name, "g_Texture1");
                assert_eq!(default_texture.as_deref(), Some("util/black"));
                assert_eq!(combo.as_deref(), Some("NORMALMAP"));
                assert!(require_any);
                assert_eq!(require.get("LIGHTING"), Some(&1));
            }
            _ => panic!("expected sampler"),
        }
    }

    #[test]
    fn uniform_commented_out_is_skipped() {
        // §2.2: `;` must come before `//`. A `//`-first line is not an annotation.
        assert_eq!(parse_uniform_line("// uniform float g_Dead; {}").unwrap(), None);
    }

    #[test]
    fn uniform_unknown_type_ignored() {
        // §2.2: unknown parameter type ignored.
        assert_eq!(
            parse_uniform_line(r#"uniform mat4 g_M; // {"material":"m"}"#).unwrap(),
            None
        );
    }
}
