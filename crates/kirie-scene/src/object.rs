//! Scene objects — base fields, kind dispatch, and every object kind.
//!
//! Spec: docs/format-scene-json.md §7 (common model + dispatch), §8 (image),
//! §11 (effects on images), §12 (sound), §13 (text), §14.1 (particle scene
//! side), §15 (model). Kind is chosen by which discriminator key exists, in the
//! §7 order.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::material::{Combos, TextureSlots, parse_combos, parse_textures};
use crate::particle::{InstanceOverride, ParticleSystem};
use crate::user::{
    ConstantValues, UserSetting, parse_constant_values, user_bool, user_color, user_f32, user_i64,
    user_string, user_vec2, user_vec3,
};
use crate::value::{Color, Vec2, Vec3, WHITE, coerce_f64, coerce_i64};

/// One keyframe of an angle-animation channel (docs/format-scene-json.md §7.4).
/// Only `frame`+`value` are read; tangent/lock handles are ignored (linear
/// interpolation only).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    /// Frame index (numeric).
    pub frame: f32,
    /// Channel value at that frame (radians for angles).
    pub value: f32,
}

/// Loop mode of an animation track (docs/format-scene-json.md §7.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AnimMode {
    /// `"mirror"` ping-pongs.
    Mirror,
    /// Anything else loops.
    #[default]
    Loop,
}

/// A per-component keyframe track on `angles` (docs/format-scene-json.md §7.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct AnimationTrack {
    /// `c0` — X channel keyframes.
    pub c0: Vec<Keyframe>,
    /// `c1` — Y channel keyframes.
    pub c1: Vec<Keyframe>,
    /// `c2` — Z channel keyframes.
    pub c2: Vec<Keyframe>,
    /// `options.fps`, default 30.0.
    pub fps: f32,
    /// `options.length`, default 0 (track disabled when 0).
    pub length: f32,
    /// `options.mode`.
    pub mode: AnimMode,
    /// `relative`, default true (result added to base; false replaces).
    pub relative: bool,
}

impl AnimationTrack {
    /// Parse an `angles.animation` object (docs/format-scene-json.md §7.4,
    /// `ObjectParser.cpp:26–76`).
    fn parse(obj: &Map<String, Value>) -> Self {
        let channel = |k: &str| -> Vec<Keyframe> {
            match obj.get(k) {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|kf| {
                        let o = kf.as_object()?;
                        Some(Keyframe {
                            frame: coerce_f64(o.get("frame")?)? as f32,
                            value: coerce_f64(o.get("value")?)? as f32,
                        })
                    })
                    .collect(),
                _ => Vec::new(),
            }
        };
        let options = obj.get("options").and_then(Value::as_object);
        let fps = options
            .and_then(|o| o.get("fps"))
            .and_then(coerce_f64)
            .map_or(30.0, |v| v as f32);
        let length = options
            .and_then(|o| o.get("length"))
            .and_then(coerce_f64)
            .map_or(0.0, |v| v as f32);
        let mode = match options.and_then(|o| o.get("mode")).and_then(Value::as_str) {
            Some("mirror") => AnimMode::Mirror,
            _ => AnimMode::Loop,
        };
        let relative = obj
            .get("relative")
            .and_then(crate::value::coerce_bool)
            .unwrap_or(true);
        AnimationTrack {
            c0: channel("c0"),
            c1: channel("c1"),
            c2: channel("c2"),
            fps,
            length,
            mode,
            relative,
        }
    }
}

/// Base fields common to every object (docs/format-scene-json.md §7.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaseObject {
    /// `id` (required; salvaged to -1 when non-numeric).
    pub id: i64,
    /// `name` (required; numeric names stringified, "unknown" fallback).
    pub name: String,
    /// `sortorder`, default 0 (consulted only with `general.customsortorder`).
    pub sortorder: i64,
    /// `dependencies` — ids that must render first (may self-reference).
    pub dependencies: Vec<i64>,
    /// `parent` id for transform inheritance (§7.3).
    pub parent: Option<i64>,
    /// `origin`, default (0,0,0).
    pub origin: UserSetting<Vec3>,
    /// `scale`, default (1,1,1).
    pub scale: UserSetting<Vec3>,
    /// `angles`, default (0,0,0), radians (§7.2).
    pub angles: UserSetting<Vec3>,
    /// `angles.animation` keyframe track, if present (§7.4).
    pub angles_animation: Option<AnimationTrack>,
    /// `visible`, default true.
    pub visible: UserSetting<bool>,
}

impl BaseObject {
    /// Parse the base fields off an object (docs/format-scene-json.md §7.1).
    fn parse(obj: &Map<String, Value>) -> Self {
        // §7.1: `id` salvages to -1 when not a number.
        let id = obj.get("id").and_then(coerce_i64).unwrap_or(-1);
        // §7.1: `name` may be numeric → stringify; "unknown" fallback.
        let name = match obj.get("name") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => "unknown".to_owned(),
        };
        let dependencies = match obj.get("dependencies") {
            Some(Value::Array(a)) => a.iter().filter_map(coerce_i64).collect(),
            _ => Vec::new(),
        };
        // §7.4: `angles` may be an object carrying an `animation` track; the
        // static base still comes from the normal user-setting `value` path.
        let angles_animation = obj
            .get("angles")
            .and_then(Value::as_object)
            .and_then(|o| o.get("animation"))
            .and_then(Value::as_object)
            .map(AnimationTrack::parse);
        BaseObject {
            id,
            name,
            sortorder: obj.get("sortorder").and_then(coerce_i64).unwrap_or(0),
            dependencies,
            parent: obj.get("parent").and_then(coerce_i64),
            origin: user_vec3(obj, "origin", [0.0, 0.0, 0.0]),
            scale: user_vec3(obj, "scale", [1.0, 1.0, 1.0]),
            angles: user_vec3(obj, "angles", [0.0, 0.0, 0.0]),
            angles_animation,
            visible: user_bool(obj, "visible", true),
        }
    }
}

/// A scene-side effect entry on an image object (docs/format-scene-json.md §11.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    /// `file` (required) — path to the effect.json.
    pub file: String,
    /// `id`, default -1.
    pub id: i64,
    /// `name`, default `"Effect without name"`.
    pub name: String,
    /// `visible`, default true — property-bindable per-effect toggle.
    pub visible: UserSetting<bool>,
    /// `passes` — per-pass overrides applied by array position (§11.3).
    pub passes: Vec<PassOverride>,
    /// The loaded effect file, filled during resolution.
    pub resolved: Option<crate::material::EffectFile>,
}

impl Effect {
    /// Parse one effect entry (docs/format-scene-json.md §11.1). Returns `None`
    /// when `file` is absent (required — the C++ skips such entries).
    fn parse(obj: &Map<String, Value>) -> Option<Self> {
        let file = obj.get("file").and_then(Value::as_str)?.to_owned();
        let passes = match obj.get("passes") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_object)
                .map(PassOverride::parse)
                .collect(),
            _ => Vec::new(),
        };
        Some(Effect {
            file,
            id: obj.get("id").and_then(coerce_i64).unwrap_or(-1),
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("Effect without name")
                .to_owned(),
            visible: user_bool(obj, "visible", true),
            passes,
            resolved: None,
        })
    }
}

/// A per-pass override (docs/format-scene-json.md §11.3), applied by position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassOverride {
    /// `id`, default -1.
    pub id: i64,
    /// `combos` merged over the material pass combos.
    pub combos: Combos,
    /// `constantshadervalues` uniform overrides.
    pub constantshadervalues: ConstantValues,
    /// `textures` slot overrides (§10.2).
    pub textures: TextureSlots,
    /// `usertextures` slot overrides.
    pub usertextures: TextureSlots,
}

impl PassOverride {
    /// Parse one pass override (docs/format-scene-json.md §11.3).
    fn parse(obj: &Map<String, Value>) -> Self {
        PassOverride {
            id: obj.get("id").and_then(coerce_i64).unwrap_or(-1),
            combos: parse_combos(obj.get("combos")),
            constantshadervalues: parse_constant_values(obj.get("constantshadervalues")),
            textures: parse_textures(obj.get("textures")),
            usertextures: parse_textures(obj.get("usertextures")),
        }
    }
}

/// A puppet-warp animation layer (docs/format-scene-json.md §8.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationLayer {
    /// `id` (required).
    pub id: i64,
    /// `rate`, default 1.0.
    pub rate: UserSetting<f32>,
    /// `visible`, default false.
    pub visible: UserSetting<bool>,
    /// `blend`, default 1.0.
    pub blend: UserSetting<f32>,
    /// `animation` index in the puppet `.mdl`, default 0.
    pub animation: UserSetting<i64>,
}

impl AnimationLayer {
    /// Parse one animation layer (docs/format-scene-json.md §8.2). `None` when
    /// `id` is absent (required).
    fn parse(obj: &Map<String, Value>) -> Option<Self> {
        let id = obj.get("id").and_then(coerce_i64)?;
        Some(AnimationLayer {
            id,
            rate: user_f32(obj, "rate", 1.0),
            visible: user_bool(obj, "visible", false),
            blend: user_f32(obj, "blend", 1.0),
            animation: user_i64(obj, "animation", 0),
        })
    }
}

/// A per-object first-pass texture override (docs/format-scene-json.md §8.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Instance {
    /// `instance.textures`.
    pub textures: TextureSlots,
    /// `instance.usertextures`.
    pub usertextures: TextureSlots,
}

/// An image object (docs/format-scene-json.md §8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageObject {
    /// `image` (required) — path to the model .json (§9).
    pub image: String,
    /// The loaded model file, filled during resolution.
    pub model: Option<crate::material::ModelFile>,
    /// The loaded material (from the model's `material`), filled during resolution.
    pub material: Option<crate::material::Material>,
    /// `scale`, default (1,1,1).
    pub scale: UserSetting<Vec3>,
    /// `angles`, default (0,0,0).
    pub angles: UserSetting<Vec3>,
    /// `visible`, default true.
    pub visible: UserSetting<bool>,
    /// `alpha`, default 1.0.
    pub alpha: UserSetting<f32>,
    /// `color`, default white.
    pub color: UserSetting<Color>,
    /// `alignment`/`horizontalalign`, default `"center"` (`horizontalalign` wins).
    pub alignment: String,
    /// `size` vec2, default (0,0) — the user wrapper is unwrapped (§8, not live).
    pub size: Vec2,
    /// `parallaxDepth` (camelCase), default (0,0).
    pub parallax_depth: UserSetting<Vec2>,
    /// `colorBlendMode` (camelCase), default 0.
    pub color_blend_mode: UserSetting<i64>,
    /// `brightness`, default 1.0.
    pub brightness: UserSetting<f32>,
    /// `effects` (§11).
    pub effects: Vec<Effect>,
    /// `animationlayers` (§8.2).
    pub animationlayers: Vec<AnimationLayer>,
    /// `instance` first-pass override (§8.3).
    pub instance: Option<Instance>,
}

/// A text object (docs/format-scene-json.md §13).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextObject {
    /// `text` (required) — commonly script-driven (clock/date wallpapers).
    pub text: UserSetting<String>,
    /// `font`, default empty.
    pub font: String,
    /// `pointsize`, default 32.0.
    pub pointsize: UserSetting<f32>,
    /// `size` vec2 bounding box, default (0,0).
    pub size: Vec2,
    /// `scale`, default (1,1,1).
    pub scale: UserSetting<Vec3>,
    /// `color`, default white.
    pub color: UserSetting<Color>,
    /// `alpha`, default 1.0.
    pub alpha: UserSetting<f32>,
    /// `visible`, default true.
    pub visible: UserSetting<bool>,
    /// `horizontalalign`/`alignment`, default `"center"`.
    pub horizontalalign: String,
    /// `verticalalign`, default `"center"`.
    pub verticalalign: String,
    /// `padding`, default 0.
    pub padding: i64,
}

/// A sound object (docs/format-scene-json.md §12).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoundObject {
    /// `sound` file paths (required).
    pub sound: Vec<String>,
    /// `playbackmode` (optional); `"loop"` repeats, else plays once.
    pub playbackmode: Option<String>,
}

/// A 3D model object (docs/format-scene-json.md §15).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelObject {
    /// `model` — path to the binary `.mdl` (mesh parsing is a formats concern).
    pub model: String,
}

/// The kind-specific payload of an object (docs/format-scene-json.md §7 dispatch).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectKind {
    /// Image object (`image` is a string) — §8.
    Image(Box<ImageObject>),
    /// Sound object (`sound` is an array) — §12.
    Sound(SoundObject),
    /// Particle object (`particle` present) — §14.
    Particle(Box<ParticleObject>),
    /// Text object (`text` present) — §13.
    Text(Box<TextObject>),
    /// 3D model object (`model` is a string) — §15.
    Model(ModelObject),
    /// Light — unsupported by the reference impl; fields preserved raw (§7).
    Light(Map<String, Value>),
    /// Volume light (`shape`) — unsupported; fields preserved raw (§7).
    Shape(Map<String, Value>),
    /// Plain group/transform object — no typed discriminator (§7).
    Group,
}

/// A particle scene object (docs/format-scene-json.md §14.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleObject {
    /// `scale`, default (1,1,1).
    pub scale: UserSetting<Vec3>,
    /// `angles`, default (0,0,0).
    pub angles: UserSetting<Vec3>,
    /// `visible`, default true.
    pub visible: UserSetting<bool>,
    /// `parallaxDepth`, default (0,0).
    pub parallax_depth: UserSetting<Vec2>,
    /// `particle` file path, when the definition is external.
    pub particle_file: Option<String>,
    /// The particle system definition (inline, or loaded from `particle_file`).
    pub system: ParticleSystem,
    /// `instanceoverride` (§14.7).
    pub instanceoverride: InstanceOverride,
}

/// A fully parsed scene object: base fields plus the kind payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Object {
    /// Common base fields (§7.1).
    pub base: BaseObject,
    /// Kind-specific payload (§7 dispatch).
    pub kind: ObjectKind,
    /// Fields not read for this kind, preserved verbatim (§7.5: `solid`,
    /// `copybackground`, `castshadow`, `perspective`, …).
    pub extra: Map<String, Value>,
}

impl Object {
    /// Parse one scene object, dispatching on the §7 discriminator order.
    pub fn parse(value: &Value) -> Option<Object> {
        let obj = value.as_object()?;
        let base = BaseObject::parse(obj);

        // §7 dispatch order. `image`/`model` must be *strings*; `sound` an
        // array; the rest by presence. `null`-valued keys fall through.
        let kind = if obj.get("image").is_some_and(Value::is_string) {
            ObjectKind::Image(Box::new(parse_image(obj)))
        } else if obj.get("sound").is_some_and(Value::is_array) {
            parse_sound(obj)
        } else if present(obj, "particle") {
            ObjectKind::Particle(Box::new(parse_particle(obj)))
        } else if present(obj, "text") {
            ObjectKind::Text(Box::new(parse_text(obj)))
        } else if obj.get("model").is_some_and(Value::is_string) {
            ObjectKind::Model(ModelObject {
                model: obj
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            })
        } else if present(obj, "light") {
            ObjectKind::Light(obj.clone())
        } else if present(obj, "shape") {
            ObjectKind::Shape(obj.clone())
        } else {
            ObjectKind::Group
        };

        Some(Object {
            base,
            kind,
            extra: obj.clone(),
        })
    }
}

/// Whether a key is present and not JSON `null` (docs/format-scene-json.md §2.3:
/// null ≡ absent for dispatch).
fn present(obj: &Map<String, Value>, key: &str) -> bool {
    !matches!(obj.get(key), None | Some(Value::Null))
}

/// Parse the image-specific fields (docs/format-scene-json.md §8).
fn parse_image(obj: &Map<String, Value>) -> ImageObject {
    // §8: `alignment`/`horizontalalign`, `horizontalalign` wins.
    let alignment = obj
        .get("horizontalalign")
        .and_then(Value::as_str)
        .or_else(|| obj.get("alignment").and_then(Value::as_str))
        .unwrap_or("center")
        .to_owned();
    // §8: `size` user-setting is immediately unwrapped to a plain vec2.
    let size = user_vec2(obj, "size", [0.0, 0.0]).value;
    let effects = match obj.get("effects") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_object)
            .filter_map(Effect::parse)
            .collect(),
        _ => Vec::new(),
    };
    let animationlayers = match obj.get("animationlayers") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_object)
            .filter_map(AnimationLayer::parse)
            .collect(),
        _ => Vec::new(),
    };
    let instance = obj.get("instance").and_then(Value::as_object).map(|o| Instance {
        textures: parse_textures(o.get("textures")),
        usertextures: parse_textures(o.get("usertextures")),
    });
    ImageObject {
        image: obj
            .get("image")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        model: None,
        material: None,
        scale: user_vec3(obj, "scale", [1.0, 1.0, 1.0]),
        angles: user_vec3(obj, "angles", [0.0, 0.0, 0.0]),
        visible: user_bool(obj, "visible", true),
        alpha: user_f32(obj, "alpha", 1.0),
        color: user_color(obj, "color", WHITE),
        alignment,
        size,
        parallax_depth: user_vec2(obj, "parallaxDepth", [0.0, 0.0]),
        color_blend_mode: user_i64(obj, "colorBlendMode", 0),
        brightness: user_f32(obj, "brightness", 1.0),
        effects,
        animationlayers,
        instance,
    }
}

/// Parse the text-specific fields (docs/format-scene-json.md §13).
fn parse_text(obj: &Map<String, Value>) -> TextObject {
    let horizontalalign = obj
        .get("horizontalalign")
        .and_then(Value::as_str)
        .or_else(|| obj.get("alignment").and_then(Value::as_str))
        .unwrap_or("center")
        .to_owned();
    TextObject {
        text: user_string(obj, "text", ""),
        font: obj
            .get("font")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        pointsize: user_f32(obj, "pointsize", 32.0),
        size: user_vec2(obj, "size", [0.0, 0.0]).value,
        scale: user_vec3(obj, "scale", [1.0, 1.0, 1.0]),
        color: user_color(obj, "color", WHITE),
        alpha: user_f32(obj, "alpha", 1.0),
        visible: user_bool(obj, "visible", true),
        horizontalalign,
        verticalalign: obj
            .get("verticalalign")
            .and_then(Value::as_str)
            .unwrap_or("center")
            .to_owned(),
        padding: obj.get("padding").and_then(coerce_i64).unwrap_or(0),
    }
}

/// Parse the sound-specific fields (docs/format-scene-json.md §12).
fn parse_sound(obj: &Map<String, Value>) -> ObjectKind {
    let sound = match obj.get("sound") {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
        _ => Vec::new(),
    };
    ObjectKind::Sound(SoundObject {
        sound,
        playbackmode: obj.get("playbackmode").and_then(Value::as_str).map(str::to_owned),
    })
}

/// Parse the particle-scene-object fields (docs/format-scene-json.md §14.1).
fn parse_particle(obj: &Map<String, Value>) -> ParticleObject {
    let (particle_file, system) = match obj.get("particle") {
        Some(Value::String(s)) => (Some(s.clone()), ParticleSystem::default()),
        Some(v @ Value::Object(_)) => (None, ParticleSystem::from_value(v)),
        _ => (None, ParticleSystem::default()),
    };
    let instanceoverride = obj
        .get("instanceoverride")
        .and_then(Value::as_object)
        .map_or_else(InstanceOverride::default, InstanceOverride::parse);
    ParticleObject {
        scale: user_vec3(obj, "scale", [1.0, 1.0, 1.0]),
        angles: user_vec3(obj, "angles", [0.0, 0.0, 0.0]),
        visible: user_bool(obj, "visible", true),
        parallax_depth: user_vec2(obj, "parallaxDepth", [0.0, 0.0]),
        particle_file,
        system,
        instanceoverride,
    }
}
