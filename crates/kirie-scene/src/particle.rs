//! Particle system definition (docs/format-scene-json.md §14).
//!
//! A particle scene object carries transform + instance overrides; the system
//! definition lives in a separate file (string path) or inline (object). This
//! module models the definition — emitters, initializers, operators, renderers,
//! control points, child systems — and the scene-side `instanceoverride`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::user::{UserSetting, read_user, user_f32};
use crate::value::{DynamicValue, Vec2, Vec3, coerce_f64, coerce_i64, coerce_u32, parse_vec};

/// Parse a broadcastable vec3 (docs/format-scene-json.md §14.3): a `"x y z"`
/// string, an `[x, y, z]` array, or a single number broadcast to all three.
pub fn parse_bvec3(value: Option<&Value>, default: Vec3) -> Vec3 {
    match value {
        Some(Value::String(s)) => parse_vec::<3>(s).unwrap_or(default),
        Some(Value::Array(a)) => {
            let mut out = default;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(v) = a.get(i).and_then(coerce_f64) {
                    *slot = v as f32;
                }
            }
            out
        }
        Some(Value::Number(_)) => {
            let n = coerce_f64(value.unwrap()).unwrap_or(0.0) as f32;
            [n, n, n]
        }
        _ => default,
    }
}

/// Parse an ivec3 `sign` field — array form only (docs/format-scene-json.md
/// §14.3, `:718–727`).
fn parse_ivec3(value: Option<&Value>, default: [i32; 3]) -> [i32; 3] {
    match value {
        Some(Value::Array(a)) => {
            let mut out = default;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(v) = a.get(i).and_then(coerce_i64) {
                    *slot = v as i32;
                }
            }
            out
        }
        _ => default,
    }
}

/// Parse a vec2 that accepts a `"x y"` string or `[x, y]` array but no number
/// broadcast (docs/format-scene-json.md §14.3 `audioprocessingbounds`).
fn parse_bvec2(value: Option<&Value>, default: Vec2) -> Vec2 {
    match value {
        Some(Value::String(s)) => parse_vec::<2>(s).unwrap_or(default),
        Some(Value::Array(a)) => {
            let mut out = default;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(v) = a.get(i).and_then(coerce_f64) {
                    *slot = v as f32;
                }
            }
            out
        }
        _ => default,
    }
}

/// An emitter (docs/format-scene-json.md §14.3). Defaults per `ObjectParser.cpp:744–770`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    /// `id`, default -1.
    pub id: i64,
    /// `name` — shape name (`boxrandom`, `sphererandom`, …), default empty.
    pub name: String,
    /// `directions`, default (1,1,0).
    pub directions: Vec3,
    /// `distancemin`, default (0,0,0) (broadcastable).
    pub distancemin: Vec3,
    /// `distancemax`, default (256,256,0) (broadcastable).
    pub distancemax: Vec3,
    /// `origin`, default (0,0,0).
    pub origin: Vec3,
    /// `sign` ivec3 (array form only), default (0,0,0).
    pub sign: [i32; 3],
    /// `instantaneous`, default 0.
    pub instantaneous: u32,
    /// `speedmin`, default 0.
    pub speedmin: f32,
    /// `speedmax`, default 0.
    pub speedmax: f32,
    /// `rate`, default 10.0.
    pub rate: f32,
    /// `controlpoint`, default 0.
    pub controlpoint: i64,
    /// `flags`, default 0.
    pub flags: u32,
    /// `cone`, default 0.0.
    pub cone: f32,
    /// `delay`, default 0.0.
    pub delay: f32,
    /// `duration`, default 0.0.
    pub duration: f32,
    /// `audioprocessingbounds`, default (0.8, 1.0).
    pub audioprocessingbounds: Vec2,
    /// `audioprocessingexponent`, default 2.
    pub audioprocessingexponent: i64,
    /// `audioprocessingfrequencystart`, default 0.
    pub audioprocessingfrequencystart: i64,
    /// `audioprocessingfrequencyend`, default 1.
    pub audioprocessingfrequencyend: i64,
    /// `audioprocessingmode`, default 0.
    pub audioprocessingmode: i64,
    /// `minperiodicdelay`, default 1.0.
    pub minperiodicdelay: f32,
    /// `maxperiodicdelay`, default 2.0.
    pub maxperiodicdelay: f32,
    /// `minperiodicduration`, default 2.0.
    pub minperiodicduration: f32,
    /// `maxperiodicduration`, default 3.0.
    pub maxperiodicduration: f32,
}

impl Emitter {
    /// Parse one emitter object (docs/format-scene-json.md §14.3).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        let f = |k: &str, d: f32| obj.get(k).and_then(coerce_f64).map_or(d, |v| v as f32);
        let i = |k: &str, d: i64| obj.get(k).and_then(coerce_i64).unwrap_or(d);
        let u = |k: &str, d: u32| obj.get(k).and_then(coerce_u32).unwrap_or(d);
        Emitter {
            id: i("id", -1),
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            directions: parse_bvec3(obj.get("directions"), [1.0, 1.0, 0.0]),
            distancemin: parse_bvec3(obj.get("distancemin"), [0.0, 0.0, 0.0]),
            distancemax: parse_bvec3(obj.get("distancemax"), [256.0, 256.0, 0.0]),
            origin: parse_bvec3(obj.get("origin"), [0.0, 0.0, 0.0]),
            sign: parse_ivec3(obj.get("sign"), [0, 0, 0]),
            instantaneous: u("instantaneous", 0),
            speedmin: f("speedmin", 0.0),
            speedmax: f("speedmax", 0.0),
            rate: f("rate", 10.0),
            controlpoint: i("controlpoint", 0),
            flags: u("flags", 0),
            cone: f("cone", 0.0),
            delay: f("delay", 0.0),
            duration: f("duration", 0.0),
            audioprocessingbounds: parse_bvec2(obj.get("audioprocessingbounds"), [0.8, 1.0]),
            audioprocessingexponent: i("audioprocessingexponent", 2),
            audioprocessingfrequencystart: i("audioprocessingfrequencystart", 0),
            audioprocessingfrequencyend: i("audioprocessingfrequencyend", 1),
            audioprocessingmode: i("audioprocessingmode", 0),
            minperiodicdelay: f("minperiodicdelay", 1.0),
            maxperiodicdelay: f("maxperiodicdelay", 2.0),
            minperiodicduration: f("minperiodicduration", 2.0),
            maxperiodicduration: f("maxperiodicduration", 3.0),
        }
    }
}

/// A particle initializer or operator (docs/format-scene-json.md §14.4/§14.5).
///
/// Both are dispatched on `name` render-side with per-name parameter defaults;
/// this model captures the name plus every parameter as a
/// [`DynamicValue`] user setting so property-bound params survive resolution.
/// Unknown names are preserved (a compatible reader keeps everything, §7.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedStage {
    /// `id`, default -1.
    pub id: i64,
    /// The dispatch name (`colorrandom`, `movement`, `vortex`, …).
    pub name: String,
    /// Every remaining member as a user setting (property-bindable, §14.4).
    pub params: std::collections::BTreeMap<String, UserSetting<DynamicValue>>,
}

impl NamedStage {
    /// Parse an initializer/operator object (docs/format-scene-json.md §14.4/§14.5).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        let mut params = std::collections::BTreeMap::new();
        for (k, v) in obj {
            if k == "id" || k == "name" {
                continue;
            }
            params.insert(
                k.clone(),
                read_user(obj, k, DynamicValue::Null, |val| {
                    Some(DynamicValue::decode(val, false))
                }),
            );
            // `read_user(obj, k, …)` above reads `obj[k]`; keep it uniform.
            let _ = v;
        }
        NamedStage {
            id: obj.get("id").and_then(coerce_i64).unwrap_or(-1),
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            params,
        }
    }
}

/// A particle renderer (docs/format-scene-json.md §14.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Renderer {
    /// `name`, default `"sprite"` (`sprite`/`rope`/`ropetrail`).
    pub name: String,
    /// `length`, default 0.05 (1.0 for `ropetrail`).
    pub length: f32,
    /// `maxlength`, default 10.0.
    pub maxlength: f32,
    /// `minlength`, default 0.0.
    pub minlength: f32,
    /// `subdivision`, default 1.0 (4.0 for `rope`).
    pub subdivision: f32,
    /// `segments`, default 4.0.
    pub segments: f32,
    /// `uvscale`, default 1.0.
    pub uvscale: f32,
    /// `uvscrolling`, default false.
    pub uvscrolling: bool,
    /// `uvsmoothing`, default true.
    pub uvsmoothing: bool,
    /// `fadealpha`, default false.
    pub fadealpha: bool,
    /// `fadesize`, default false.
    pub fadesize: bool,
}

impl Renderer {
    /// The default sprite renderer (docs/format-scene-json.md §14.6, `:546–562`).
    pub fn default_sprite() -> Self {
        Renderer {
            name: "sprite".to_owned(),
            length: 0.05,
            maxlength: 10.0,
            minlength: 0.0,
            subdivision: 1.0,
            segments: 4.0,
            uvscale: 1.0,
            uvscrolling: false,
            uvsmoothing: true,
            fadealpha: false,
            fadesize: false,
        }
    }

    /// Parse one renderer object (docs/format-scene-json.md §14.6).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("sprite")
            .to_owned();
        // Name-dependent defaults (§14.6).
        let length_default = if name == "ropetrail" { 1.0 } else { 0.05 };
        let subdivision_default = if name == "rope" { 4.0 } else { 1.0 };
        let f = |k: &str, d: f32| obj.get(k).and_then(coerce_f64).map_or(d, |v| v as f32);
        let b = |k: &str, d: bool| obj.get(k).and_then(crate::value::coerce_bool).unwrap_or(d);
        Renderer {
            length: f("length", length_default),
            maxlength: f("maxlength", 10.0),
            minlength: f("minlength", 0.0),
            subdivision: f("subdivision", subdivision_default),
            segments: f("segments", 4.0),
            uvscale: f("uvscale", 1.0),
            uvscrolling: b("uvscrolling", false),
            uvsmoothing: b("uvsmoothing", true),
            fadealpha: b("fadealpha", false),
            fadesize: b("fadesize", false),
            name,
        }
    }
}

/// A control point (docs/format-scene-json.md §14.6, `:940–966`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPoint {
    /// `id`, default -1.
    pub id: i64,
    /// `flags`, default 0.
    pub flags: u32,
    /// `offset` vec3 (string form only in the C++), default (0,0,0).
    pub offset: Vec3,
    /// `locktopointer`, default false — bind the point to the cursor.
    pub locktopointer: bool,
}

impl ControlPoint {
    /// Parse one control point object (docs/format-scene-json.md §14.6).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        ControlPoint {
            id: obj.get("id").and_then(coerce_i64).unwrap_or(-1),
            flags: obj.get("flags").and_then(coerce_u32).unwrap_or(0),
            offset: obj
                .get("offset")
                .and_then(Value::as_str)
                .and_then(|s| parse_vec::<3>(s).ok())
                .unwrap_or([0.0, 0.0, 0.0]),
            locktopointer: obj
                .get("locktopointer")
                .and_then(crate::value::coerce_bool)
                .unwrap_or(false),
        }
    }
}

/// A child particle system (docs/format-scene-json.md §14.6, `:968–1018`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChildSystem {
    /// `type`, default `"static"`.
    pub kind: String,
    /// `name`, default empty.
    pub name: String,
    /// `maxcount`, default 20.
    pub maxcount: u32,
    /// `controlpointstartindex`, default 0.
    pub controlpointstartindex: i64,
    /// `probability`, default 1.0.
    pub probability: f32,
    /// `angles`, default (0,0,0).
    pub angles: Vec3,
    /// `origin`, default (0,0,0).
    pub origin: Vec3,
    /// `scale`, default (1,1,1).
    pub scale: Vec3,
    /// `particle` path to the child particle file.
    pub particle: Option<String>,
}

impl ChildSystem {
    /// Parse one child system object (docs/format-scene-json.md §14.6).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        ChildSystem {
            kind: obj
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("static")
                .to_owned(),
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            maxcount: obj.get("maxcount").and_then(coerce_u32).unwrap_or(20),
            controlpointstartindex: obj
                .get("controlpointstartindex")
                .and_then(coerce_i64)
                .unwrap_or(0),
            probability: obj
                .get("probability")
                .and_then(coerce_f64)
                .map_or(1.0, |v| v as f32),
            angles: parse_bvec3(obj.get("angles"), [0.0, 0.0, 0.0]),
            origin: parse_bvec3(obj.get("origin"), [0.0, 0.0, 0.0]),
            scale: parse_bvec3(obj.get("scale"), [1.0, 1.0, 1.0]),
            particle: obj.get("particle").and_then(Value::as_str).map(str::to_owned),
        }
    }
}

/// The particle system definition (docs/format-scene-json.md §14.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct ParticleSystem {
    /// `material` path (the wrapped material .json), if any.
    pub material: Option<String>,
    /// The loaded material, filled during resolution.
    pub resolved_material: Option<crate::material::Material>,
    /// `animationmode`, default `"sequence"`.
    pub animationmode: String,
    /// `sequencemultiplier`, default 1.0.
    pub sequencemultiplier: f32,
    /// `maxcount`, default 100.
    pub maxcount: u32,
    /// `starttime`, default 0.
    pub starttime: u32,
    /// `flags`, default 0.
    pub flags: u32,
    /// `emitter[]` (note singular key name).
    pub emitters: Vec<Emitter>,
    /// `initializer[]`.
    pub initializers: Vec<NamedStage>,
    /// `operator[]`.
    pub operators: Vec<NamedStage>,
    /// `renderer[]` (a default sprite renderer when empty).
    pub renderers: Vec<Renderer>,
    /// `controlpoint[]`.
    pub controlpoints: Vec<ControlPoint>,
    /// `children[]`.
    pub children: Vec<ChildSystem>,
}

impl ParticleSystem {
    /// Parse a particle definition value (docs/format-scene-json.md §14.2,
    /// `ObjectParser.cpp:503–658`). Note the singular array key names.
    pub fn from_value(value: &Value) -> Self {
        let obj = value.as_object().cloned().unwrap_or_default();
        let arr = |k: &str| -> Vec<Map<String, Value>> {
            match obj.get(k) {
                Some(Value::Array(a)) => a.iter().filter_map(Value::as_object).cloned().collect(),
                _ => Vec::new(),
            }
        };
        let mut renderers: Vec<Renderer> = arr("renderer").iter().map(Renderer::parse).collect();
        if renderers.is_empty() {
            renderers.push(Renderer::default_sprite());
        }
        ParticleSystem {
            material: obj.get("material").and_then(Value::as_str).map(str::to_owned),
            resolved_material: None,
            animationmode: obj
                .get("animationmode")
                .and_then(Value::as_str)
                .unwrap_or("sequence")
                .to_owned(),
            sequencemultiplier: obj
                .get("sequencemultiplier")
                .and_then(coerce_f64)
                .map_or(1.0, |v| v as f32),
            maxcount: obj.get("maxcount").and_then(coerce_u32).unwrap_or(100),
            starttime: obj.get("starttime").and_then(coerce_u32).unwrap_or(0),
            flags: obj.get("flags").and_then(coerce_u32).unwrap_or(0),
            emitters: arr("emitter").iter().map(Emitter::parse).collect(),
            initializers: arr("initializer").iter().map(NamedStage::parse).collect(),
            operators: arr("operator").iter().map(NamedStage::parse).collect(),
            renderers,
            controlpoints: arr("controlpoint").iter().map(ControlPoint::parse).collect(),
            children: arr("children").iter().map(ChildSystem::parse).collect(),
        }
    }
}

/// The scene-side `instanceoverride` (docs/format-scene-json.md §14.7).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstanceOverride {
    /// `enabled`, default true.
    pub enabled: UserSetting<bool>,
    /// `alpha` multiplier, default 1.0.
    pub alpha: UserSetting<f32>,
    /// `size` multiplier, default 1.0.
    pub size: UserSetting<f32>,
    /// `lifetime` multiplier, default 1.0.
    pub lifetime: UserSetting<f32>,
    /// `rate` multiplier, default 1.0.
    pub rate: UserSetting<f32>,
    /// `speed` multiplier, default 1.0.
    pub speed: UserSetting<f32>,
    /// `count` multiplier, default 1.0.
    pub count: UserSetting<f32>,
    /// `color` — replaces particle color, default (1,1,1).
    pub color: UserSetting<Vec3>,
    /// `colorn` — multiplies particle color, default (1,1,1).
    pub colorn: UserSetting<Vec3>,
}

impl Default for InstanceOverride {
    fn default() -> Self {
        InstanceOverride {
            enabled: UserSetting::literal(true),
            alpha: UserSetting::literal(1.0),
            size: UserSetting::literal(1.0),
            lifetime: UserSetting::literal(1.0),
            rate: UserSetting::literal(1.0),
            speed: UserSetting::literal(1.0),
            count: UserSetting::literal(1.0),
            color: UserSetting::literal([1.0, 1.0, 1.0]),
            colorn: UserSetting::literal([1.0, 1.0, 1.0]),
        }
    }
}

impl InstanceOverride {
    /// Parse an `instanceoverride` object (docs/format-scene-json.md §14.7).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        use crate::user::{user_bool, user_vec3};
        InstanceOverride {
            enabled: user_bool(obj, "enabled", true),
            alpha: user_f32(obj, "alpha", 1.0),
            size: user_f32(obj, "size", 1.0),
            lifetime: user_f32(obj, "lifetime", 1.0),
            rate: user_f32(obj, "rate", 1.0),
            speed: user_f32(obj, "speed", 1.0),
            count: user_f32(obj, "count", 1.0),
            color: user_vec3(obj, "color", [1.0, 1.0, 1.0]),
            colorn: user_vec3(obj, "colorn", [1.0, 1.0, 1.0]),
        }
    }
}
