//! The `"user"` indirection — property-bound / script-driven leaf fields.
//!
//! Spec: docs/format-scene-json.md §3. Nearly every leaf of a scene object is a
//! *user setting*: a plain literal, or an object with a required `"value"` plus
//! an optional `"user"` property binding and/or an optional `"script"`
//! (SceneScript). This module models that wrapper and the typed readers that
//! pull each field off a JSON object with the correct default.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::value::{Color, Vec2, Vec3, coerce_bool, coerce_f64, coerce_i64, parse_color, parse_vec};

/// A property binding target (docs/format-scene-json.md §3.1/§3.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserRef {
    /// `"user": "name"` — the property's value overwrites the setting's value
    /// on load and on change (§3.2).
    Name(String),
    /// `"user": { "name": N, "condition": C }` — the setting becomes the
    /// boolean `property == C` (§3.3, plain string equality).
    Conditional {
        /// The bound property name.
        name: String,
        /// The literal string the property's value is compared against.
        condition: String,
    },
}

impl UserRef {
    /// The property name this binding reads, regardless of form.
    pub fn name(&self) -> &str {
        match self {
            UserRef::Name(n) | UserRef::Conditional { name: n, .. } => n,
        }
    }
}

/// A script-driven setting (docs/format-scene-json.md §3.1: SceneScript source
/// plus its own recursively-parsed properties). Semantics belong to the script
/// engine; the parser only captures the source and property blobs verbatim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScriptBinding {
    /// JavaScript source string.
    pub source: String,
    /// `scriptproperties` — each value is itself a user setting; preserved raw
    /// for the SceneScript engine to interpret.
    pub properties: Map<String, Value>,
}

/// A leaf field: an initial/fallback value plus optional bindings
/// (docs/format-scene-json.md §3.1).
///
/// After [`crate::resolve`], `value` holds the current resolved value; the
/// bindings stay attached so a later `setProperty` can re-resolve the field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserSetting<T> {
    /// The literal in `"value"` (or the field default) — the initial/fallback
    /// value and, after resolution, the current value (§3.2).
    pub value: T,
    /// Optional project-property binding.
    pub user: Option<UserRef>,
    /// Optional SceneScript driver.
    pub script: Option<ScriptBinding>,
}

impl<T> UserSetting<T> {
    /// A plain literal setting with no bindings.
    pub fn literal(value: T) -> Self {
        UserSetting {
            value,
            user: None,
            script: None,
        }
    }

    /// Whether this field is bound to a property or a script (so resolution and
    /// live `setProperty` must consider it).
    pub fn is_bound(&self) -> bool {
        self.user.is_some() || self.script.is_some()
    }
}

/// Parse the `"user"` / `"script"` bindings off a user-setting object
/// (docs/format-scene-json.md §3.1, `UserSettingParser.cpp:15–36`).
fn parse_bindings(obj: &Map<String, Value>) -> (Option<UserRef>, Option<ScriptBinding>) {
    let user = match obj.get("user") {
        Some(Value::String(name)) => Some(UserRef::Name(name.clone())),
        Some(Value::Object(o)) => {
            // §3.1: object form requires `name` and `condition` strings.
            match (o.get("name"), o.get("condition")) {
                (Some(Value::String(name)), Some(cond)) => Some(UserRef::Conditional {
                    name: name.clone(),
                    condition: cond.as_str().map(str::to_owned).unwrap_or_default(),
                }),
                _ => None,
            }
        }
        _ => None,
    };
    let script = match obj.get("script") {
        Some(Value::String(source)) => Some(ScriptBinding {
            source: source.clone(),
            properties: match obj.get("scriptproperties") {
                Some(Value::Object(p)) => p.clone(),
                _ => Map::new(),
            },
        }),
        _ => None,
    };
    (user, script)
}

/// Read a user setting off `map[key]` with a value parser and default.
///
/// docs/format-scene-json.md §3.1: an object form takes its literal from the
/// required `"value"` member (a missing/`null` key or a non-object literal that
/// the parser rejects falls back to `default`). Plain literals are parsed
/// directly. `null`/absent ⇒ default (§2.3).
pub fn read_user<T>(
    map: &Map<String, Value>,
    key: &str,
    default: T,
    parse: impl Fn(&Value) -> Option<T>,
) -> UserSetting<T> {
    match map.get(key) {
        None | Some(Value::Null) => UserSetting::literal(default),
        Some(Value::Object(obj)) => {
            let (user, script) = parse_bindings(obj);
            let value = obj.get("value").and_then(&parse).unwrap_or(default);
            UserSetting { value, user, script }
        }
        Some(other) => UserSetting {
            value: parse(other).unwrap_or(default),
            user: None,
            script: None,
        },
    }
}

/// User bool field, §2.3-coerced, with the given default.
pub fn user_bool(map: &Map<String, Value>, key: &str, default: bool) -> UserSetting<bool> {
    read_user(map, key, default, |v| coerce_bool(v).or(Some(default)))
}

/// User float field, §2.3-coerced, with the given default.
pub fn user_f32(map: &Map<String, Value>, key: &str, default: f32) -> UserSetting<f32> {
    read_user(map, key, default, |v| coerce_f64(v).map(|f| f as f32))
}

/// User int field, §2.3-coerced, with the given default.
pub fn user_i64(map: &Map<String, Value>, key: &str, default: i64) -> UserSetting<i64> {
    read_user(map, key, default, coerce_i64)
}

/// User string field with the given default (non-strings keep the default).
pub fn user_string(map: &Map<String, Value>, key: &str, default: &str) -> UserSetting<String> {
    read_user(map, key, default.to_owned(), |v| v.as_str().map(str::to_owned))
}

/// User vec2 field (`"x y"`) with the given default.
pub fn user_vec2(map: &Map<String, Value>, key: &str, default: Vec2) -> UserSetting<Vec2> {
    read_user(map, key, default, |v| {
        v.as_str().and_then(|s| parse_vec::<2>(s).ok())
    })
}

/// User vec3 field (`"x y z"`) with the given default.
pub fn user_vec3(map: &Map<String, Value>, key: &str, default: Vec3) -> UserSetting<Vec3> {
    read_user(map, key, default, |v| {
        v.as_str().and_then(|s| parse_vec::<3>(s).ok())
    })
}

/// User color field with the given default (scene-side: int-0..255 path active
/// unless the string carries a `.`; docs/format-scene-json.md §2.2).
pub fn user_color(map: &Map<String, Value>, key: &str, default: Color) -> UserSetting<Color> {
    read_user(map, key, default, |v| {
        v.as_str().and_then(|s| parse_color(s, 1.0, false).ok())
    })
}

/// A `constantshadervalues` map: uniform-constant name → user setting
/// (docs/format-scene-json.md §10.3). Values are decoded as
/// [`crate::value::DynamicValue`] user settings so any spelling is preserved.
pub type ConstantValues = BTreeMap<String, UserSetting<crate::value::DynamicValue>>;

/// Parse a `constantshadervalues` object (docs/format-scene-json.md §10.3,
/// `ShaderConstantParser.cpp:9–21`). A non-object degrades to empty (§17.6).
pub fn parse_constant_values(value: Option<&Value>) -> ConstantValues {
    let mut out = BTreeMap::new();
    if let Some(Value::Object(obj)) = value {
        for (name, raw) in obj {
            out.insert(
                name.clone(),
                read_user(&single(name, raw), name, crate::value::DynamicValue::Null, |v| {
                    Some(crate::value::DynamicValue::decode(v, false))
                }),
            );
        }
    }
    out
}

/// Wrap a `(key, value)` pair as a one-entry map so [`read_user`] can process a
/// constant's user-setting object (which itself may carry `value`/`user`).
fn single(key: &str, value: &Value) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(key.to_owned(), value.clone());
    m
}
