//! The scene root: `camera`, `general`, `objects` (docs/format-scene-json.md
//! §4, §5, §6) and the top-level parse entry point.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::SceneError;
use crate::object::Object;
use crate::user::{UserSetting, user_bool, user_color, user_f32};
use crate::value::{BLACK, Color, Vec3, WHITE, coerce_bool, coerce_f64, coerce_i64, parse_vec};

/// The camera projection (docs/format-scene-json.md §6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Projection {
    /// `general.orthogonalprojection = { width, height }` in scene pixels.
    Orthogonal {
        /// Projection width in scene pixels (missing member ⇒ 0).
        width: i64,
        /// Projection height in scene pixels.
        height: i64,
    },
    /// `{ "auto": true }`, explicit `null`, or a missing key ⇒ auto-size at
    /// render time (§6.2).
    Auto,
}

/// The `camera` section (docs/format-scene-json.md §6).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    /// `eye` (required vec3).
    pub eye: Vec3,
    /// `center` (required vec3).
    pub center: Vec3,
    /// `up` (required vec3).
    pub up: Vec3,
    /// Near plane, default 0.1 (§6.2, clamped `>0` else 0.1 at render time).
    pub nearz: f32,
    /// Far plane, default 10000 (§6.2, clamped `>nearz` else 10000).
    pub farz: f32,
    /// Field of view, default 50 (`general.fov` preferred, else `camera.fov`).
    pub fov: f32,
    /// Resolved projection (§6.2).
    pub projection: Projection,
}

impl Camera {
    /// Parse the camera from the `camera` and `general` sections
    /// (docs/format-scene-json.md §6). `nearz`/`farz` are read from `camera`
    /// first, then `general` (a documented faithful extension — real scenes
    /// declare them under `general`, §6.2 caveat).
    fn parse(camera: &Map<String, Value>, general: &Map<String, Value>) -> Result<Self, SceneError> {
        let vec = |field: &'static str| -> Result<Vec3, SceneError> {
            let s = camera
                .get(field)
                .and_then(Value::as_str)
                .ok_or(SceneError::MissingCameraField(field))?;
            parse_vec::<3>(s).map_err(|source| SceneError::CameraVec { field, source })
        };
        let eye = vec("eye")?;
        let center = vec("center")?;
        let up = vec("up")?;

        // §6.2: nearz/farz — camera first, then general.
        let read_f = |key: &str, default: f32| -> f32 {
            camera
                .get(key)
                .and_then(coerce_f64)
                .or_else(|| general.get(key).and_then(coerce_f64))
                .map_or(default, |v| v as f32)
        };
        let nearz = read_f("nearz", 0.1);
        let farz = read_f("farz", 10000.0);
        // §6.2/§5: fov — general preferred, else camera.
        let fov = general
            .get("fov")
            .and_then(coerce_f64)
            .or_else(|| camera.get("fov").and_then(coerce_f64))
            .map_or(50.0, |v| v as f32);

        // §6.2: orthogonalprojection object / null / missing ⇒ auto.
        let projection = match general.get("orthogonalprojection") {
            Some(Value::Object(o)) => {
                if o.get("auto").and_then(coerce_bool).unwrap_or(false) {
                    Projection::Auto
                } else {
                    Projection::Orthogonal {
                        width: o.get("width").and_then(coerce_i64).unwrap_or(0),
                        height: o.get("height").and_then(coerce_i64).unwrap_or(0),
                    }
                }
            }
            _ => Projection::Auto,
        };

        Ok(Camera {
            eye,
            center,
            up,
            nearz,
            farz,
            fov,
            projection,
        })
    }
}

/// The `general` scene-wide settings (docs/format-scene-json.md §5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct General {
    /// `ambientcolor`, default (0,0,0).
    pub ambientcolor: UserSetting<Color>,
    /// `skylightcolor`, default (0,0,0).
    pub skylightcolor: UserSetting<Color>,
    /// `clearcolor`, default (1,1,1).
    pub clearcolor: UserSetting<Color>,
    /// `camerafade`, default false.
    pub camerafade: UserSetting<bool>,
    /// `camerapreview` (plain optional), default false.
    pub camerapreview: bool,
    /// `bloom`, default false.
    pub bloom: UserSetting<bool>,
    /// `bloomstrength`, default 0.0.
    pub bloomstrength: UserSetting<f32>,
    /// `bloomthreshold`, default 0.0.
    pub bloomthreshold: UserSetting<f32>,
    /// `cameraparallax`, default false.
    pub cameraparallax: UserSetting<bool>,
    /// `cameraparallaxamount`, default 1.0.
    pub cameraparallaxamount: UserSetting<f32>,
    /// `cameraparallaxdelay`, default 0.0.
    pub cameraparallaxdelay: UserSetting<f32>,
    /// `cameraparallaxmouseinfluence`, default 1.0.
    pub cameraparallaxmouseinfluence: UserSetting<f32>,
    /// `camerashake`, default false.
    pub camerashake: UserSetting<bool>,
    /// `camerashakeamplitude`, default 0.0.
    pub camerashakeamplitude: UserSetting<f32>,
    /// `camerashakeroughness`, default 0.0.
    pub camerashakeroughness: UserSetting<f32>,
    /// `camerashakespeed`, default 0.0.
    pub camerashakespeed: UserSetting<f32>,
    /// `customsortorder` (plain optional), default false.
    pub customsortorder: bool,
    /// Every other `general.*` key, preserved verbatim (§5 unparsed list).
    pub extra: Map<String, Value>,
}

impl General {
    /// Parse the `general` section (docs/format-scene-json.md §5).
    fn parse(map: &Map<String, Value>) -> Self {
        General {
            ambientcolor: user_color(map, "ambientcolor", BLACK),
            skylightcolor: user_color(map, "skylightcolor", BLACK),
            clearcolor: user_color(map, "clearcolor", WHITE),
            camerafade: user_bool(map, "camerafade", false),
            camerapreview: map.get("camerapreview").and_then(coerce_bool).unwrap_or(false),
            bloom: user_bool(map, "bloom", false),
            bloomstrength: user_f32(map, "bloomstrength", 0.0),
            bloomthreshold: user_f32(map, "bloomthreshold", 0.0),
            cameraparallax: user_bool(map, "cameraparallax", false),
            cameraparallaxamount: user_f32(map, "cameraparallaxamount", 1.0),
            cameraparallaxdelay: user_f32(map, "cameraparallaxdelay", 0.0),
            cameraparallaxmouseinfluence: user_f32(map, "cameraparallaxmouseinfluence", 1.0),
            camerashake: user_bool(map, "camerashake", false),
            camerashakeamplitude: user_f32(map, "camerashakeamplitude", 0.0),
            camerashakeroughness: user_f32(map, "camerashakeroughness", 0.0),
            camerashakespeed: user_f32(map, "camerashakespeed", 0.0),
            customsortorder: map.get("customsortorder").and_then(coerce_bool).unwrap_or(false),
            extra: map.clone(),
        }
    }
}

/// A parsed `scene.json` (docs/format-scene-json.md §4).
///
/// This is the *unresolved* model: leaf fields keep their [`UserSetting`]
/// bindings. Call [`Scene::resolve`] with a [`crate::property::PropertyBag`] to
/// collapse bindings into concrete values for the renderer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// The camera (§6).
    pub camera: Camera,
    /// Scene-wide settings (§5).
    pub general: General,
    /// The scene objects in file order (§7).
    pub objects: Vec<Object>,
}

impl Scene {
    /// Parse a `scene.json` from raw bytes (docs/format-scene-json.md §1/§4).
    pub fn from_slice(bytes: &[u8]) -> Result<Self, SceneError> {
        let value: Value = serde_json::from_slice(bytes).map_err(|e| SceneError::Json(e.to_string()))?;
        Self::from_value(&value)
    }

    /// Parse a `scene.json` from an already-decoded JSON value
    /// (docs/format-scene-json.md §4).
    pub fn from_value(value: &Value) -> Result<Self, SceneError> {
        let root = value.as_object().ok_or(SceneError::NotAnObject)?;

        // §4: camera, general, objects all required.
        let camera_map = root
            .get("camera")
            .and_then(Value::as_object)
            .ok_or(SceneError::MissingSection("camera"))?;
        let general_map = root
            .get("general")
            .and_then(Value::as_object)
            .ok_or(SceneError::MissingSection("general"))?;
        let objects_arr = root
            .get("objects")
            .and_then(Value::as_array)
            .ok_or(SceneError::MissingSection("objects"))?;

        let camera = Camera::parse(camera_map, general_map)?;
        let general = General::parse(general_map);
        // §7: unparseable objects (non-object entries) are skipped, not fatal.
        let objects = objects_arr.iter().filter_map(Object::parse).collect();

        Ok(Scene {
            camera,
            general,
            objects,
        })
    }
}
