//! [`ScriptValue`] — the single marshalled value type crossing the JS ↔ host
//! boundary, plus the QuickJS/JSON conversions.
//!
//! docs/scripting-api.md §5.1 (`dynamicToJs` / `jsToDynamicValue`). The C++
//! defect that dropped `z`/`w` when reading an object return value
//! (docs §5.1 `DO-NOT-PORT`) is fixed here: all present components are read.

use rquickjs::{Array, Ctx, Function, Object, Value};
use serde::ser::{Serialize, SerializeMap, Serializer};

/// A value marshalled between a script and the host. Mirrors the reference
/// `DynamicValue` underlying types (docs §5.1).
#[derive(Clone, Debug, PartialEq)]
pub enum ScriptValue {
    /// JS `null`/`undefined`/uninitialized — property set to the Null type
    /// (docs §5.1: returning nothing is *not* a no-op).
    Null,
    /// A boolean.
    Bool(bool),
    /// A 32-bit integer.
    Int(i64),
    /// A double.
    Float(f64),
    /// A string.
    Str(String),
    /// A 2-component vector.
    Vec2([f32; 2]),
    /// A 3-component vector.
    Vec3([f32; 3]),
    /// A 4-component vector.
    Vec4([f32; 4]),
}

impl ScriptValue {
    /// Build a JS value for `self` in `ctx`. Vectors become real `VecN`
    /// instances (so scripts may call methods and mutate components).
    pub fn to_js<'js>(&self, ctx: &Ctx<'js>) -> rquickjs::Result<Value<'js>> {
        Ok(match self {
            ScriptValue::Null => Value::new_null(ctx.clone()),
            ScriptValue::Bool(b) => Value::new_bool(ctx.clone(), *b),
            ScriptValue::Int(i) => Value::new_number(ctx.clone(), *i as f64),
            ScriptValue::Float(f) => Value::new_number(ctx.clone(), *f),
            ScriptValue::Str(s) => rquickjs::String::from_str(ctx.clone(), s)?.into_value(),
            ScriptValue::Vec2(v) => construct_vec(ctx, 2, [v[0], v[1], 0.0, 0.0])?,
            ScriptValue::Vec3(v) => construct_vec(ctx, 3, [v[0], v[1], v[2], 0.0])?,
            ScriptValue::Vec4(v) => construct_vec(ctx, 4, [v[0], v[1], v[2], v[3]])?,
        })
    }

    /// Decode a JS value returned by `update()` (docs §5.1 `jsToDynamicValue`,
    /// with the z/w-drop defect fixed). Objects are read by `x/y/z/w`.
    pub fn from_js(value: &Value<'_>) -> Self {
        if value.is_undefined() || value.is_null() || value.type_of() == rquickjs::Type::Uninitialized {
            return ScriptValue::Null;
        }
        if let Some(b) = value.as_bool() {
            return ScriptValue::Bool(b);
        }
        if value.is_int() {
            return ScriptValue::Int(value.as_int().unwrap_or(0) as i64);
        }
        if value.is_float() {
            return ScriptValue::Float(value.as_float().unwrap_or(0.0));
        }
        if value.is_number() {
            return ScriptValue::Float(value.as_number().unwrap_or(0.0));
        }
        if let Some(s) = value.as_string() {
            return match s.to_string() {
                Ok(s) => ScriptValue::Str(s),
                Err(_) => ScriptValue::Null,
            };
        }
        if let Some(obj) = value.as_object() {
            let comp = |k: &str| obj.get::<_, f64>(k).ok().filter(|f| f.is_finite());
            match (comp("x"), comp("y")) {
                (Some(x), Some(y)) => {
                    return match (comp("z"), comp("w")) {
                        (Some(z), Some(w)) => ScriptValue::Vec4([x as f32, y as f32, z as f32, w as f32]),
                        (Some(z), None) => ScriptValue::Vec3([x as f32, y as f32, z as f32]),
                        _ => ScriptValue::Vec2([x as f32, y as f32]),
                    };
                }
                // An object without numeric x/y is not a value (docs §5.1 treats
                // the malformed case as ignore — we map to Null, never throw).
                _ => return ScriptValue::Null,
            }
        }
        ScriptValue::Null
    }
}

/// Construct a global `VecN` instance from components (via the `__mkVec` helper
/// — `rquickjs::Function` has no `construct`, that lives on `Constructor`).
fn construct_vec<'js>(ctx: &Ctx<'js>, n: i32, c: [f32; 4]) -> rquickjs::Result<Value<'js>> {
    let mk: Function = ctx.globals().get("__mkVec")?;
    mk.call((n, c[0], c[1], c[2], c[3]))
}

// docs §5.1: userProperties expose vectors as `{x,y,z}`-shaped values; the plain
// object form is enough for component reads inside `engine.userProperties`.
impl Serialize for ScriptValue {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ScriptValue::Null => s.serialize_none(),
            ScriptValue::Bool(b) => s.serialize_bool(*b),
            ScriptValue::Int(i) => s.serialize_i64(*i),
            ScriptValue::Float(f) => s.serialize_f64(*f),
            ScriptValue::Str(v) => s.serialize_str(v),
            ScriptValue::Vec2(v) => {
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("x", &v[0])?;
                m.serialize_entry("y", &v[1])?;
                m.end()
            }
            ScriptValue::Vec3(v) => {
                let mut m = s.serialize_map(Some(3))?;
                m.serialize_entry("x", &v[0])?;
                m.serialize_entry("y", &v[1])?;
                m.serialize_entry("z", &v[2])?;
                m.end()
            }
            ScriptValue::Vec4(v) => {
                let mut m = s.serialize_map(Some(4))?;
                m.serialize_entry("x", &v[0])?;
                m.serialize_entry("y", &v[1])?;
                m.serialize_entry("z", &v[2])?;
                m.serialize_entry("w", &v[3])?;
                m.end()
            }
        }
    }
}

/// Convert a [`serde_json::Value`] into a QuickJS [`Value`]. Used to inject the
/// per-tick host snapshot (`__host`) as plain JS data (SPEC.md §V3: typed
/// messages, never shared memory).
pub fn json_to_js<'js>(ctx: &Ctx<'js>, v: &serde_json::Value) -> rquickjs::Result<Value<'js>> {
    use serde_json::Value as J;
    Ok(match v {
        J::Null => Value::new_null(ctx.clone()),
        J::Bool(b) => Value::new_bool(ctx.clone(), *b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::new_number(ctx.clone(), i as f64)
            } else {
                Value::new_number(ctx.clone(), n.as_f64().unwrap_or(0.0))
            }
        }
        J::String(s) => rquickjs::String::from_str(ctx.clone(), s)?.into_value(),
        J::Array(a) => {
            let arr = Array::new(ctx.clone())?;
            for (i, e) in a.iter().enumerate() {
                arr.set(i, json_to_js(ctx, e)?)?;
            }
            arr.into_value()
        }
        J::Object(o) => {
            let obj = Object::new(ctx.clone())?;
            for (k, e) in o {
                obj.set(k.as_str(), json_to_js(ctx, e)?)?;
            }
            obj.into_value()
        }
    })
}
