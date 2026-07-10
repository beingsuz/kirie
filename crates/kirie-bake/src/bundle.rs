//! The baked-bundle payload: an rkyv archive of everything the renderer needs to
//! start a scene without re-parsing, re-translating, or re-decoding
//! (SPEC.md §V8; the "prebaked scene bundle" of the task).
//!
//! Layout (all fields rkyv-native so the archive validates with bytecheck and
//! loads zero-copy via mmap):
//!
//! - [`BakedBundle::header`] — magic + versions + source digest, checked on load.
//! - [`BakedBundle::scene_json`] — the resolved [`kirie_scene::SceneModel`] as
//!   serde JSON bytes. `SceneModel` is `serde`, not `rkyv`; storing its JSON keeps
//!   kirie-bake decoupled from the scene model's internal shape while preserving
//!   the §V13 round-trip. It is small relative to textures/shaders.
//! - [`BakedBundle::shaders`] — translated shader units: SPIR-V words (so warm
//!   load skips the expensive glslang pass), the modernized GLSL, and reflection.
//! - [`BakedBundle::textures`] — GPU-ready texture payloads: BCn kept *compressed*
//!   with its mip layout, or decoded RGBA8 where the format needs CPU decode. The
//!   rule is "store what avoids re-decode".
//! - [`BakedBundle::tables`] — arbitrary precomputed byte tables (e.g. particle
//!   LUTs) the renderer wants memoized.

/// Magic word at the head of every [`BakedBundle::header`] ("KBAK" LE).
pub const BUNDLE_MAGIC: u32 = 0x4b41_424b;

/// Fixed metadata at the head of a bundle, validated on load.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BundleHeader {
    /// [`BUNDLE_MAGIC`]. A mismatch means the file is not a kirie bundle.
    pub magic: u32,
    /// [`crate::BAKE_FORMAT_VERSION`] at write time (SPEC.md §V8).
    pub format_version: u32,
    /// [`kirie_shader::TRANSLATOR_VERSION`] at write time (SPEC.md §V8).
    pub translator_version: u32,
    /// blake3 of the source bytes this bundle was baked from (SPEC.md §V8). The
    /// cache directory is keyed by [`crate::BundleKey`] over the same source plus
    /// versions; this copy lets a consumer double-check provenance.
    pub source_hash: [u8; 32],
}

/// One of the two shader stages, stored as a byte (mirrors [`kirie_shader::Stage`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub enum BakedStage {
    /// Vertex unit.
    Vertex,
    /// Fragment unit.
    Fragment,
}

impl From<kirie_shader::Stage> for BakedStage {
    fn from(s: kirie_shader::Stage) -> Self {
        match s {
            kirie_shader::Stage::Vertex => BakedStage::Vertex,
            kirie_shader::Stage::Fragment => BakedStage::Fragment,
        }
    }
}

impl From<BakedStage> for kirie_shader::Stage {
    fn from(s: BakedStage) -> Self {
        match s {
            BakedStage::Vertex => kirie_shader::Stage::Vertex,
            BakedStage::Fragment => kirie_shader::Stage::Fragment,
        }
    }
}

/// A translated shader unit, ready to hand to wgpu without re-running the
/// glslang/naga translation pipeline (the expensive bake step).
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedShader {
    /// Which stage this unit is.
    pub stage: BakedStage,
    /// Source unit name for diagnostics/keying (e.g. `effects/foo.frag`).
    pub name: String,
    /// SPIR-V words. Empty if SPIR-V emission was unavailable at bake time; the
    /// consumer then falls back to re-translating [`Self::glsl`].
    pub spirv: Vec<u32>,
    /// The final modernized GLSL fed to the frontend (kept for debugging and as
    /// a fallback translation input).
    pub glsl: String,
    /// Reflection captured during translation (binding metadata by source name).
    pub reflection: BakedReflection,
}

impl BakedShader {
    /// Build from a [`kirie_shader::TranslatedShader`]. Emits SPIR-V from the
    /// validated naga module when possible; on failure the [`Self::spirv`] field
    /// is left empty and the consumer re-translates [`Self::glsl`].
    #[must_use]
    pub fn from_translated(
        stage: kirie_shader::Stage,
        name: impl Into<String>,
        ts: &kirie_shader::TranslatedShader,
    ) -> Self {
        BakedShader {
            stage: stage.into(),
            name: name.into(),
            spirv: emit_spirv(&ts.module).unwrap_or_default(),
            glsl: ts.glsl.clone(),
            reflection: BakedReflection::from(&ts.reflection),
        }
    }
}

/// Emit SPIR-V words from a translated naga module (best-effort). Re-validates
/// with permissive capabilities before writing; any failure returns `None` and
/// the shader is baked GLSL-only.
fn emit_spirv(module: &naga::Module) -> Option<Vec<u32>> {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(module)
        .ok()?;
    let opts = naga::back::spv::Options::default();
    naga::back::spv::write_vec(module, &info, &opts, None).ok()
}

/// Serializable mirror of [`kirie_shader::reflect::ParamType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub enum BakedParamType {
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

impl From<kirie_shader::reflect::ParamType> for BakedParamType {
    fn from(t: kirie_shader::reflect::ParamType) -> Self {
        use kirie_shader::reflect::ParamType as P;
        match t {
            P::Float => BakedParamType::Float,
            P::Int => BakedParamType::Int,
            P::Vec2 => BakedParamType::Vec2,
            P::Vec3 => BakedParamType::Vec3,
            P::Vec4 => BakedParamType::Vec4,
        }
    }
}

/// Serializable mirror of [`kirie_shader::reflect::ParamDefault`].
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum BakedParamDefault {
    /// Scalar `float`/`int` default.
    Scalar(f64),
    /// Vector default (2–4 components).
    Vector(Vec<f32>),
}

impl From<&kirie_shader::reflect::ParamDefault> for BakedParamDefault {
    fn from(d: &kirie_shader::reflect::ParamDefault) -> Self {
        use kirie_shader::reflect::ParamDefault as D;
        match d {
            D::Scalar(v) => BakedParamDefault::Scalar(*v),
            D::Vector(v) => BakedParamDefault::Vector(v.clone()),
        }
    }
}

/// Serializable mirror of [`kirie_shader::reflect::Parameter`].
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedParameter {
    /// GLSL uniform name.
    pub name: String,
    /// Constant identifier this uniform binds to.
    pub material: String,
    /// Declared type.
    pub ty: BakedParamType,
    /// Parsed default, if any.
    pub default: Option<BakedParamDefault>,
}

/// Serializable mirror of [`kirie_shader::reflect::SamplerSlot`].
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedSampler {
    /// GLSL uniform name.
    pub name: String,
    /// Wallpaper Engine texture slot index (0–9), if the name matched.
    pub slot: Option<u32>,
    /// wgpu binding of the split `texture2D` global.
    pub texture_binding: u32,
    /// wgpu binding of the split `sampler` global.
    pub sampler_binding: u32,
    /// Default texture name from the annotation, if any.
    pub default_texture: Option<String>,
    /// Combo macro gated by this slot, if any.
    pub combo: Option<String>,
}

/// Serializable mirror of [`kirie_shader::reflect::VertexAttribute`].
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedAttribute {
    /// Attribute name.
    pub name: String,
    /// Assigned `@location`.
    pub location: u32,
}

/// One entry of the combo macro table.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedCombo {
    /// Uppercased combo name.
    pub name: String,
    /// Combo value.
    pub value: i32,
}

/// Serializable mirror of [`kirie_shader::Reflection`].
#[derive(Debug, Clone, Default, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedReflection {
    /// Loose uniforms packed into the `_WEGlobals` block, in declaration order.
    pub globals_block: Vec<String>,
    /// Bindable material parameters.
    pub parameters: Vec<BakedParameter>,
    /// Texture/sampler slots.
    pub samplers: Vec<BakedSampler>,
    /// Vertex attributes (vertex stage only).
    pub attributes: Vec<BakedAttribute>,
    /// The final combo macro table, name→value.
    pub active_combos: Vec<BakedCombo>,
}

impl From<&kirie_shader::Reflection> for BakedReflection {
    fn from(r: &kirie_shader::Reflection) -> Self {
        BakedReflection {
            globals_block: r.globals_block.clone(),
            parameters: r
                .parameters
                .iter()
                .map(|p| BakedParameter {
                    name: p.name.clone(),
                    material: p.material.clone(),
                    ty: p.ty.into(),
                    default: p.default.as_ref().map(BakedParamDefault::from),
                })
                .collect(),
            samplers: r
                .samplers
                .iter()
                .map(|s| BakedSampler {
                    name: s.name.clone(),
                    slot: s.slot,
                    texture_binding: s.texture_binding,
                    sampler_binding: s.sampler_binding,
                    default_texture: s.default_texture.clone(),
                    combo: s.combo.clone(),
                })
                .collect(),
            attributes: r
                .attributes
                .iter()
                .map(|a| BakedAttribute {
                    name: a.name.clone(),
                    location: a.location,
                })
                .collect(),
            active_combos: r
                .active_combos
                .iter()
                .map(|(name, value)| BakedCombo {
                    name: name.clone(),
                    value: *value,
                })
                .collect(),
        }
    }
}

/// A single mip level within a [`BakedTexture`] payload (offset/length into
/// [`BakedTexture::data`]).
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedMip {
    /// Mip width in texels.
    pub width: u32,
    /// Mip height in texels.
    pub height: u32,
    /// Byte offset of this mip's payload within [`BakedTexture::data`].
    pub offset: u64,
    /// Byte length of this mip's payload.
    pub len: u64,
}

/// A GPU-ready texture: either a compressed BCn payload with its mip layout, or
/// decoded RGBA8 (the choice made to avoid re-decode at load time).
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedTexture {
    /// Asset name/path the scene references this texture by.
    pub name: String,
    /// The wgpu-facing texture format, encoded as its numeric tag by the
    /// producer. `0` denotes "decoded RGBA8" (see [`Self::is_rgba8`]); non-zero
    /// values carry a caller-defined compressed-format tag.
    pub format_tag: u32,
    /// Real (usable) width in texels.
    pub width: u32,
    /// Real (usable) height in texels.
    pub height: u32,
    /// `true` if [`Self::data`] is uncompressed RGBA8 (mip 0 only); `false` if it
    /// is a compressed payload described by [`Self::mips`].
    pub rgba8: bool,
    /// Mip layout into [`Self::data`]. For RGBA8 this is a single entry.
    pub mips: Vec<BakedMip>,
    /// The contiguous texture bytes (compressed blocks kept as-is, or RGBA8).
    pub data: Vec<u8>,
}

impl BakedTexture {
    /// Bake a decoded RGBA8 image (single mip). Use for formats that must be CPU
    /// decoded (FreeImage path, etc.) so the renderer uploads directly.
    #[must_use]
    pub fn rgba8(name: impl Into<String>, width: u32, height: u32, pixels: Vec<u8>) -> Self {
        let len = pixels.len() as u64;
        BakedTexture {
            name: name.into(),
            format_tag: 0,
            width,
            height,
            rgba8: true,
            mips: vec![BakedMip {
                width,
                height,
                offset: 0,
                len,
            }],
            data: pixels,
        }
    }

    /// `true` if this texture is stored decoded RGBA8.
    #[must_use]
    pub fn is_rgba8(&self) -> bool {
        self.rgba8
    }
}

/// An arbitrary precomputed byte table (e.g. particle initializer LUTs) the
/// renderer wants memoized in the bundle.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedTable {
    /// Table name/key.
    pub name: String,
    /// Table bytes.
    pub data: Vec<u8>,
}

/// The complete on-disk bundle (rkyv archive root). Built by the cache from a
/// [`BundleContent`] plus a computed [`BundleHeader`].
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BakedBundle {
    /// Fixed metadata, validated on load.
    pub header: BundleHeader,
    /// Resolved [`kirie_scene::SceneModel`] as serde JSON bytes.
    pub scene_json: Vec<u8>,
    /// Translated shader units.
    pub shaders: Vec<BakedShader>,
    /// GPU-ready textures.
    pub textures: Vec<BakedTexture>,
    /// Precomputed tables.
    pub tables: Vec<BakedTable>,
}

/// The producer-facing payload assembled before a bake. The cache stamps the
/// [`BundleHeader`] and serializes it into a [`BakedBundle`].
#[derive(Debug, Clone, Default)]
pub struct BundleContent {
    /// Resolved scene model as serde JSON bytes.
    pub scene_json: Vec<u8>,
    /// Translated shader units.
    pub shaders: Vec<BakedShader>,
    /// GPU-ready textures.
    pub textures: Vec<BakedTexture>,
    /// Precomputed tables.
    pub tables: Vec<BakedTable>,
}

impl BundleContent {
    /// Start an empty content builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the resolved [`kirie_scene::SceneModel`] as serde JSON (SPEC.md
    /// §V13 round-trip). Returns a [`crate::BakeError::Serialize`] if the model
    /// cannot be serialized (should not happen for a resolved model).
    pub fn set_scene_model(
        &mut self,
        model: &kirie_scene::SceneModel,
    ) -> Result<&mut Self, crate::BakeError> {
        self.scene_json =
            serde_json::to_vec(model).map_err(|e| crate::BakeError::Serialize(e.to_string()))?;
        Ok(self)
    }

    /// Add a translated shader, emitting SPIR-V where possible.
    pub fn add_translated_shader(
        &mut self,
        stage: kirie_shader::Stage,
        name: impl Into<String>,
        ts: &kirie_shader::TranslatedShader,
    ) -> &mut Self {
        self.shaders.push(BakedShader::from_translated(stage, name, ts));
        self
    }

    /// Add an already-built shader entry.
    pub fn add_shader(&mut self, shader: BakedShader) -> &mut Self {
        self.shaders.push(shader);
        self
    }

    /// Add a decoded RGBA8 texture.
    pub fn add_rgba8_texture(
        &mut self,
        name: impl Into<String>,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> &mut Self {
        self.textures
            .push(BakedTexture::rgba8(name, width, height, pixels));
        self
    }

    /// Add an already-built texture entry.
    pub fn add_texture(&mut self, texture: BakedTexture) -> &mut Self {
        self.textures.push(texture);
        self
    }

    /// Add a precomputed table.
    pub fn add_table(&mut self, name: impl Into<String>, data: Vec<u8>) -> &mut Self {
        self.tables.push(BakedTable {
            name: name.into(),
            data,
        });
        self
    }

    /// Stamp the header and materialize the archive-root [`BakedBundle`].
    #[must_use]
    pub(crate) fn into_bundle(self, source: &[u8]) -> BakedBundle {
        BakedBundle {
            header: BundleHeader {
                magic: BUNDLE_MAGIC,
                format_version: crate::BAKE_FORMAT_VERSION,
                translator_version: kirie_shader::TRANSLATOR_VERSION,
                source_hash: *blake3::hash(source).as_bytes(),
            },
            scene_json: self.scene_json,
            shaders: self.shaders,
            textures: self.textures,
            tables: self.tables,
        }
    }
}
