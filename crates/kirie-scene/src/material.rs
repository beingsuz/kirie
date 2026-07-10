//! Material, effect and model-file parsers — the effect/material/pass chain.
//!
//! Spec: docs/format-scene-json.md §9 (model files), §10 (materials + passes +
//! textures + combos + constant values), §11 (effects, effect files, FBOs, pass
//! overrides). These are separate `.json` files referenced by path from scene
//! objects; a full scene resolution loads and parses each of them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::user::{ConstantValues, parse_constant_values};
use crate::value::coerce_i64;

/// Pass blend mode (docs/format-scene-json.md §10, `Material.h:12–17`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Blending {
    /// `"normal"` = 1 (default; unknown strings fall back here with a log).
    #[default]
    Normal,
    /// `"translucent"` = 2.
    Translucent,
    /// `"additive"` = 3.
    Additive,
}

impl Blending {
    /// Parse a blending string, defaulting on unknown values (§17.4).
    pub fn parse(s: &str) -> Self {
        match s {
            "translucent" => Blending::Translucent,
            "additive" => Blending::Additive,
            _ => Blending::Normal,
        }
    }
}

/// Face-cull mode (docs/format-scene-json.md §10, `Material.h:19`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CullMode {
    /// `"nocull"` = Disable (default).
    #[default]
    NoCull,
    /// `"normal"` = Normal cull.
    Normal,
}

impl CullMode {
    /// Parse a cull-mode string, defaulting on unknown values (§17.4).
    pub fn parse(s: &str) -> Self {
        match s {
            "normal" => CullMode::Normal,
            _ => CullMode::NoCull,
        }
    }
}

/// Depth-test / depth-write toggle (docs/format-scene-json.md §10,
/// `Material.h:21–31`). Both `depthtest` and `depthwrite` default `"disabled"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DepthMode {
    /// `"disabled"` = 1 (default).
    #[default]
    Disabled,
    /// `"enabled"` = 2.
    Enabled,
}

impl DepthMode {
    /// Parse a depth string, defaulting on unknown values (§17.4).
    pub fn parse(s: &str) -> Self {
        match s {
            "enabled" => DepthMode::Enabled,
            _ => DepthMode::Disabled,
        }
    }
}

/// One texture slot of a pass (docs/format-scene-json.md §10.2). The array
/// index *is* the binding slot; a `None` is a `null`/empty slot that keeps
/// whatever the shader default or previous pass provides.
pub type TextureSlots = Vec<Option<String>>;

/// Parse a `textures`/`usertextures` array (docs/format-scene-json.md §10.2,
/// `TextureParser.cpp:125–154`). Non-array degrades to empty (§17.6).
pub fn parse_textures(value: Option<&Value>) -> TextureSlots {
    let Some(Value::Array(arr)) = value else {
        return Vec::new();
    };
    arr.iter()
        .map(|entry| match entry {
            // null → empty slot; index still advances.
            Value::Null => None,
            // empty string is skipped (treated as empty slot).
            Value::String(s) if s.is_empty() => None,
            Value::String(s) => Some(s.clone()),
            // object entry → its `name` member.
            Value::Object(o) => o.get("name").and_then(Value::as_str).map(str::to_owned),
            _ => None,
        })
        .collect()
}

/// A combo map: preprocessor define name → integer (docs/format-scene-json.md
/// §10, `MaterialParser.cpp:59–71`).
pub type Combos = BTreeMap<String, i64>;

/// Parse a `combos` object (docs/format-scene-json.md §10). Non-object degrades
/// to empty (§17.6); values are §2.3-coerced to int.
pub fn parse_combos(value: Option<&Value>) -> Combos {
    let mut out = BTreeMap::new();
    if let Some(Value::Object(obj)) = value {
        for (k, v) in obj {
            out.insert(k.clone(), coerce_i64(v).unwrap_or(0));
        }
    }
    out
}

/// A material pass (docs/format-scene-json.md §10).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pass {
    /// `blending`, default `normal`.
    pub blending: Blending,
    /// `cullmode`, default `nocull`.
    pub cullmode: CullMode,
    /// `depthtest`, default `disabled`.
    pub depthtest: DepthMode,
    /// `depthwrite`, default `disabled`.
    pub depthwrite: DepthMode,
    /// `shader` base name (required) — `shaders/<name>.vert/.frag`.
    pub shader: String,
    /// `textures` slots (§10.2).
    pub textures: TextureSlots,
    /// `usertextures` slots (§10, same encoding).
    pub usertextures: TextureSlots,
    /// `combos` preprocessor defines (§10).
    pub combos: Combos,
    /// `constantshadervalues` uniform overrides (§10.3).
    pub constantshadervalues: ConstantValues,
}

impl Pass {
    /// Parse one pass object (docs/format-scene-json.md §10,
    /// `MaterialParser.cpp:39–57`). `shader` defaults to empty when absent so a
    /// malformed pass never aborts the whole load (§17.6).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        Pass {
            blending: obj
                .get("blending")
                .and_then(Value::as_str)
                .map_or(Blending::Normal, Blending::parse),
            cullmode: obj
                .get("cullmode")
                .and_then(Value::as_str)
                .map_or(CullMode::NoCull, CullMode::parse),
            depthtest: obj
                .get("depthtest")
                .and_then(Value::as_str)
                .map_or(DepthMode::Disabled, DepthMode::parse),
            depthwrite: obj
                .get("depthwrite")
                .and_then(Value::as_str)
                .map_or(DepthMode::Disabled, DepthMode::parse),
            shader: obj
                .get("shader")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            textures: parse_textures(obj.get("textures")),
            usertextures: parse_textures(obj.get("usertextures")),
            combos: parse_combos(obj.get("combos")),
            constantshadervalues: parse_constant_values(obj.get("constantshadervalues")),
        }
    }
}

/// A material file (docs/format-scene-json.md §10): `{"passes": [ … ]}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Material {
    /// The material's passes; a non-array `passes` is tolerated as empty
    /// (§10, `MaterialParser.cpp:28–30`).
    pub passes: Vec<Pass>,
}

impl Material {
    /// Parse a material file's JSON value (docs/format-scene-json.md §10).
    pub fn from_value(value: &Value) -> Self {
        let passes = match value.get("passes") {
            Some(Value::Array(arr)) => arr.iter().filter_map(Value::as_object).map(Pass::parse).collect(),
            _ => Vec::new(),
        };
        Material { passes }
    }
}

/// A model file (`models/*.json`, docs/format-scene-json.md §9) referenced by an
/// image object's `image` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelFile {
    /// `material` path (required) — the material .json this model draws with.
    pub material: String,
    /// `solidlayer`, default false.
    pub solidlayer: bool,
    /// `fullscreen`, default false.
    pub fullscreen: bool,
    /// `passthrough`, default false.
    pub passthrough: bool,
    /// `autosize`, default false.
    pub autosize: bool,
    /// `nopadding`, default false.
    pub nopadding: bool,
    /// `width` (optional).
    pub width: Option<i64>,
    /// `height` (optional).
    pub height: Option<i64>,
    /// `puppet` path (optional) — puppet-warp `.mdl`.
    pub puppet: Option<String>,
}

/// Error parsing a model file (docs/format-scene-json.md §9).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelFileError {
    /// `material` key absent — required (`ModelParser.cpp:19–33`).
    #[error("model file: required key `material` missing")]
    MaterialMissing,
}

impl ModelFile {
    /// Parse a model file's JSON value (docs/format-scene-json.md §9).
    pub fn from_value(value: &Value) -> Result<Self, ModelFileError> {
        let material = value
            .get("material")
            .and_then(Value::as_str)
            .ok_or(ModelFileError::MaterialMissing)?
            .to_owned();
        let flag = |k: &str| value.get(k).and_then(crate::value::coerce_bool).unwrap_or(false);
        let int = |k: &str| value.get(k).and_then(coerce_i64);
        Ok(ModelFile {
            material,
            solidlayer: flag("solidlayer"),
            fullscreen: flag("fullscreen"),
            passthrough: flag("passthrough"),
            autosize: flag("autosize"),
            nopadding: flag("nopadding"),
            width: int("width"),
            height: int("height"),
            puppet: value.get("puppet").and_then(Value::as_str).map(str::to_owned),
        })
    }
}

/// A buffer command on an effect pass (docs/format-scene-json.md §11.2,
/// `Effect.h:9`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassCommand {
    /// `"copy"` → Copy the source FBO to the target.
    Copy,
    /// Anything else present → Swap.
    Swap,
}

/// A texture-slot bind on an effect pass (docs/format-scene-json.md §11.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bind {
    /// FBO / named source; `"previous"` = the previous pass output (§11.2).
    pub name: String,
    /// Texture slot index the source binds to.
    pub index: i64,
}

/// One pass of an effect file (docs/format-scene-json.md §11.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectPass {
    /// `material` path (loaded recursively into [`EffectPass::resolved`]).
    pub material: Option<String>,
    /// The loaded material, filled during resolution (§11.2).
    pub resolved: Option<Material>,
    /// `bind` entries — slot ← FBO/named source (§11.2).
    pub bind: Vec<Bind>,
    /// `command` — `copy`/swap buffer command instead of a draw (§11.2).
    pub command: Option<PassCommand>,
    /// `source` FBO name (required when `command` present) (§11.2).
    pub source: Option<String>,
    /// `target` FBO name (render into this instead of the chain) (§11.2).
    pub target: Option<String>,
}

impl EffectPass {
    /// Parse one effect-file pass (docs/format-scene-json.md §11.2,
    /// `EffectParser.cpp:48–97`).
    pub fn parse(obj: &Map<String, Value>) -> Self {
        let bind = match obj.get("bind") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|b| {
                    let o = b.as_object()?;
                    Some(Bind {
                        name: o.get("name").and_then(Value::as_str)?.to_owned(),
                        index: coerce_i64(o.get("index")?)?,
                    })
                })
                .collect(),
            _ => Vec::new(),
        };
        let command = obj.get("command").and_then(Value::as_str).map(|c| {
            if c == "copy" {
                PassCommand::Copy
            } else {
                PassCommand::Swap
            }
        });
        EffectPass {
            material: obj.get("material").and_then(Value::as_str).map(str::to_owned),
            resolved: None,
            bind,
            command,
            source: obj.get("source").and_then(Value::as_str).map(str::to_owned),
            target: obj.get("target").and_then(Value::as_str).map(str::to_owned),
        }
    }
}

/// An FBO declaration in an effect file (docs/format-scene-json.md §11.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fbo {
    /// `name` (required) — referenced by `bind`/`target`, always `_rt_…`.
    pub name: String,
    /// `format`, default `"rgba8888"`.
    pub format: String,
    /// `scale` — a **divisor** vs the image render size (2 = half res).
    pub scale: f32,
    /// `unique`, default false.
    pub unique: bool,
}

impl Fbo {
    /// Parse one FBO declaration (docs/format-scene-json.md §11.2,
    /// `EffectParser.cpp:99–118`).
    pub fn parse(obj: &Map<String, Value>) -> Option<Self> {
        let name = obj.get("name").and_then(Value::as_str)?.to_owned();
        Some(Fbo {
            name,
            format: obj
                .get("format")
                .and_then(Value::as_str)
                .unwrap_or("rgba8888")
                .to_owned(),
            scale: obj.get("scale").and_then(crate::value::coerce_f64).unwrap_or(1.0) as f32,
            unique: obj
                .get("unique")
                .and_then(crate::value::coerce_bool)
                .unwrap_or(false),
        })
    }
}

/// An effect file (`effect.json`, docs/format-scene-json.md §11.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct EffectFile {
    /// `name` metadata, default empty.
    pub name: String,
    /// `description` metadata, default empty.
    pub description: String,
    /// `group` metadata, default empty.
    pub group: String,
    /// `preview` metadata, default empty.
    pub preview: String,
    /// `dependencies` file list (editor metadata).
    pub dependencies: Vec<String>,
    /// `passes` (required); a non-array degrades to empty (§17.6).
    pub passes: Vec<EffectPass>,
    /// `fbos` declarations, default empty.
    pub fbos: Vec<Fbo>,
}

impl EffectFile {
    /// Parse an effect file's JSON value (docs/format-scene-json.md §11.2,
    /// `EffectParser.cpp:19–32`).
    pub fn from_value(value: &Value) -> Self {
        let str_field = |k: &str| {
            value
                .get(k)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let dependencies = match value.get("dependencies") {
            Some(Value::Array(arr)) => arr.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
            _ => Vec::new(),
        };
        let passes = match value.get("passes") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(Value::as_object)
                .map(EffectPass::parse)
                .collect(),
            _ => Vec::new(),
        };
        let fbos = match value.get("fbos") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(Value::as_object)
                .filter_map(Fbo::parse)
                .collect(),
            _ => Vec::new(),
        };
        EffectFile {
            name: str_field("name"),
            description: str_field("description"),
            group: str_field("group"),
            preview: str_field("preview"),
            dependencies,
            passes,
            fbos,
        }
    }
}
