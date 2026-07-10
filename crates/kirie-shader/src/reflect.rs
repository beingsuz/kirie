//! Reflection metadata produced alongside a translated shader module.
//!
//! The renderer binds against this table by **source name** (docs/shader-pipeline.md
//! §8.2, §9.1.3): Wallpaper Engine performs all uniform/attribute binding through
//! `glGetUniformLocation`/`glGetAttribLocation`, so the wgpu port must carry the
//! original names forward. This module records the name→binding maps captured at
//! preprocessing time, when names and slots are still authoritative
//! (docs/shader-pipeline.md §9.3).

use std::collections::BTreeMap;

/// A scalar/vector uniform default drawn from a `// {json}` annotation
/// (docs/shader-pipeline.md §2.2). Encodes the parsed default value.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamDefault {
    /// `float`/`int` scalar default (docs/shader-pipeline.md §2.2 number/string case).
    Scalar(f64),
    /// `vec2`/`vec3`/`vec4` default parsed from a space-separated string
    /// (docs/shader-pipeline.md §2.2 `VectorBuilder`).
    Vector(Vec<f32>),
}

/// The declared GLSL type of a bindable parameter (docs/shader-pipeline.md §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    /// `float`
    Float,
    /// `int`
    Int,
    /// `vec2`
    Vec2,
    /// `vec3`
    Vec3,
    /// `vec4`
    Vec4,
}

/// A non-sampler bindable parameter: a `uniform` with a `// {json}` annotation
/// carrying a `material` link (docs/shader-pipeline.md §2.2). Only uniforms with a
/// `material` key are registered by the reference (`ShaderUnit.cpp:690-695`).
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    /// The GLSL uniform name, e.g. `g_Brightness`.
    pub name: String,
    /// The constant identifier this uniform binds to (the annotation `material`
    /// key), matched against pass `constantshadervalues` and effect overrides
    /// (docs/shader-pipeline.md §2.2, §8.2).
    pub material: String,
    /// Declared type (docs/shader-pipeline.md §2.2).
    pub ty: ParamType,
    /// Parsed default value, if the annotation supplied one.
    pub default: Option<ParamDefault>,
}

/// A texture/sampler slot synthesized from a `uniform sampler2D g_Texture<N>`
/// declaration (docs/shader-pipeline.md §2.2, §8.2). The combined GL sampler is
/// split into a separate texture + sampler binding pair for wgpu
/// (docs/shader-pipeline.md §9.2.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplerSlot {
    /// GLSL uniform name, e.g. `g_Texture0`.
    pub name: String,
    /// Wallpaper Engine texture slot index (`name[9] - '0'`, slots 0-9;
    /// docs/shader-pipeline.md §2.2). `None` for samplers not named
    /// `g_Texture<digit>`.
    pub slot: Option<u32>,
    /// wgpu binding assigned to the split `texture2D` global.
    pub texture_binding: u32,
    /// wgpu binding assigned to the split `sampler` global.
    pub sampler_binding: u32,
    /// Default texture name from the annotation `default` key, if any
    /// (docs/shader-pipeline.md §2.2; may be a `materials/...` path or an
    /// `_rt_`/`_alias_` FBO reference).
    pub default_texture: Option<String>,
    /// Combo macro this slot gates when populated, from the annotation `combo`
    /// key (docs/shader-pipeline.md §2.2).
    pub combo: Option<String>,
}

/// A vertex attribute input (`attribute` in the raw dialect; docs/shader-pipeline.md
/// §2, §8.2). Wallpaper Engine binds exactly `a_Position`/`a_TexCoord`
/// (`CPass.cpp:715-718`) but the reflection records whatever the unit declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexAttribute {
    /// Attribute name, e.g. `a_Position`.
    pub name: String,
    /// Assigned `@location`.
    pub location: u32,
}

/// Reflection captured during translation, keyed by source name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reflection {
    /// Loose engine/parameter uniforms packed into the synthesized `_WEGlobals`
    /// uniform block (docs/shader-pipeline.md §9.2.2), in declaration order. The
    /// name is the member name; the renderer writes it by name→offset.
    pub globals_block: Vec<String>,
    /// Bindable material parameters (docs/shader-pipeline.md §2.2).
    pub parameters: Vec<Parameter>,
    /// Texture/sampler slots (docs/shader-pipeline.md §2.2, §8.2).
    pub samplers: Vec<SamplerSlot>,
    /// Vertex attributes (vertex stage only).
    pub attributes: Vec<VertexAttribute>,
    /// The final combo macro table emitted for this program, uppercased
    /// (docs/shader-pipeline.md §4.3), name→value.
    pub active_combos: BTreeMap<String, i32>,
}
