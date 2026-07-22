//! kirie-shader — translate the Wallpaper Engine shader dialect into wgpu-ready
//! shader modules.
//!
//! The reference pipeline (docs/shader-pipeline.md) takes HLSL-flavored legacy
//! GLSL through a text preprocessor, a patched glslang → SPIR-V step, and a
//! SPIRV-Cross → GLSL 330 re-emission for its GL 3.3 renderer. This crate keeps
//! the content-visible layers (docs/shader-pipeline.md §9.1) — include
//! resolution, combo discovery/expansion, the macro prelude, annotation
//! reflection — and replaces the GL-era tail (docs/shader-pipeline.md §9.2) with
//! a wgpu-native path:
//!
//! ```text
//! raw .vert/.frag
//!   │ preprocess()   §3-§4: includes, #require, combos, prelude, gl_FragColor
//!   │ modernize()    §9.2: UBO-pack loose uniforms, split combined samplers,
//!   │                      rename GLSL-reserved identifiers, flatten array varyings
//!   │ translate()    naga glsl-in (pure Rust) first, else shaderc → SPIR-V → naga spv-in
//!   ▼
//! naga::Module (+ Reflection)  → wgpu::ShaderModule via ShaderSource::Naga
//! ```
//!
//! # Why shaderc is (usually) required
//!
//! Empirically (see the corpus test), naga's pure-Rust GLSL frontend rejects
//! several constructs that pervade real workshop shaders even after
//! best-effort modernization, so the crate falls back to the system
//! `libshaderc` (stock glslang) → SPIR-V → naga's SPIR-V frontend for those.
//! [`TranslatePath`] records which route each shader took.
//!
//! # Determinism (SPEC.md §V8)
//!
//! Translation is deterministic for a fixed input and [`TRANSLATOR_VERSION`],
//! which participates in the bake bundle key (SPEC.md §V8: `blake3(source) ⊕
//! fmt-ver ⊕ translator-ver`). Bump it whenever translation output changes.

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::collections::BTreeMap;
use std::path::PathBuf;

use thiserror::Error;

pub mod annotation;
pub mod coerce;
pub mod modernize;
pub mod preprocess;
pub mod reflect;
pub mod translate;

pub use reflect::Reflection;

/// Shader-translator version, mixed into the bake bundle key (SPEC.md §V8). Any
/// change to preprocessing, modernization, or translation output that would
/// alter a produced module must bump this so stale bundles are re-baked.
pub const TRANSLATOR_VERSION: u32 = 3;

/// The two stage kinds in the dialect — vertex and fragment only; there is no
/// geometry/compute support (docs/shader-pipeline.md §Pipeline, `GLSLContext.h:15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// `.vert` unit — `attribute` inputs, `varying` outputs.
    Vertex,
    /// `.frag` unit — `varying` inputs, writes `gl_FragColor`.
    Fragment,
}

impl Stage {
    /// Map to the corresponding naga shader stage.
    #[must_use]
    pub fn naga(self) -> naga::ShaderStage {
        match self {
            Stage::Vertex => naga::ShaderStage::Vertex,
            Stage::Fragment => naga::ShaderStage::Fragment,
        }
    }

    /// The source-file extension the reference derives for this stage
    /// (docs/shader-pipeline.md §1.2).
    #[must_use]
    pub fn ext(self) -> &'static str {
        match self {
            Stage::Vertex => "vert",
            Stage::Fragment => "frag",
        }
    }
}

/// Which frontend produced the module for a given shader (reported per SPEC/task
/// requirement so the render phase knows the translation route).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TranslatePath {
    /// naga's pure-Rust GLSL frontend accepted the modernized source directly.
    NagaGlsl,
    /// Fell back to system `libshaderc` (glslang) → SPIR-V → naga's SPIR-V
    /// frontend (docs/shader-pipeline.md §9.1.2 option (a)).
    Shaderc,
}

/// Resolves `#include "name"` directives to shader-header source
/// (docs/shader-pipeline.md §1.1, §3.1). The reference maps the include name to
/// `shaders/<base>.h` and searches mounts in order (workshop files shadow
/// `scene.pkg`, which shadows stock assets); a miss is **not** an error
/// (docs/shader-pipeline.md §3.1).
pub trait IncludeResolver {
    /// Return the header source for `include_name` (already `.h`-normalized by
    /// the caller), or `None` if no mount contains it.
    fn resolve(&self, include_name: &str) -> Option<String>;
}

/// A filesystem include resolver searching a list of `shaders/` roots in order
/// (docs/shader-pipeline.md §1.1 mount order). Typically the wallpaper's
/// extracted `shaders/` dir followed by the stock-assets `shaders/` dir.
#[derive(Debug, Clone)]
pub struct FsIncludeResolver {
    /// Directories searched in order; the first containing the header wins.
    pub roots: Vec<PathBuf>,
}

impl FsIncludeResolver {
    /// Build from a list of `shaders/` root directories, highest priority first.
    #[must_use]
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }
}

impl IncludeResolver for FsIncludeResolver {
    fn resolve(&self, include_name: &str) -> Option<String> {
        for root in &self.roots {
            let path = root.join(include_name);
            if let Ok(text) = std::fs::read_to_string(&path) {
                return Some(text);
            }
        }
        None
    }
}

/// An in-memory include resolver backed by a name→source map (e.g. headers read
/// out of a `scene.pkg` via kirie-formats). Keys are `.h`-normalized include
/// names such as `common.h` (docs/shader-pipeline.md §1.2).
#[derive(Debug, Clone, Default)]
pub struct MapIncludeResolver {
    /// Header name → source. Consulted before any fallback.
    pub headers: BTreeMap<String, String>,
    /// Optional fallback consulted when `headers` misses (e.g. stock assets).
    pub fallback: Option<FsIncludeResolver>,
}

impl IncludeResolver for MapIncludeResolver {
    fn resolve(&self, include_name: &str) -> Option<String> {
        if let Some(s) = self.headers.get(include_name) {
            return Some(s.clone());
        }
        self.fallback.as_ref().and_then(|f| f.resolve(include_name))
    }
}

/// Material/scene-supplied inputs that steer combo emission and sampler gating
/// (docs/shader-pipeline.md §4.5). All optional; an empty value uses only the
/// shader's own discovered combo defaults (docs/shader-pipeline.md §2.1).
#[derive(Debug, Clone, Default)]
pub struct ShaderInputs {
    /// Material/pass combos: `passes[i].combos` (docs/shader-pipeline.md §4.5,
    /// `MaterialParser.cpp`). Keys may be lower-case; uppercased at emission.
    pub combos: BTreeMap<String, i32>,
    /// Effect-pass override combos: `effects[j].passes[k].combos`
    /// (docs/shader-pipeline.md §4.5, `ObjectParser.cpp`). Highest precedence
    /// among externally supplied combos (docs/shader-pipeline.md §4.3).
    pub override_combos: BTreeMap<String, i32>,
    /// Texture slots (0-9) that the pass/override actually populates
    /// (docs/shader-pipeline.md §2.2 rule 1: a populated slot forces its combo).
    pub populated_texture_slots: std::collections::BTreeSet<u32>,
}

/// A fully translated shader: the naga module plus reflection and route.
#[derive(Debug, Clone)]
pub struct TranslatedShader {
    /// The validated naga IR module, ready for `wgpu::ShaderSource::Naga`.
    pub module: naga::Module,
    /// Name-keyed reflection for renderer binding (docs/shader-pipeline.md §8.2).
    pub reflection: Reflection,
    /// Which frontend produced the module.
    pub path: TranslatePath,
    /// The final modernized GLSL fed to the frontend (kept for debugging/bake).
    pub glsl: String,
}

/// Errors from the translation pipeline (SPEC.md §V9: typed, no panics on
/// malformed input).
#[derive(Debug, Error)]
pub enum TranslateError {
    /// An annotation failed to parse (docs/shader-pipeline.md §2).
    #[error("annotation error in {file}: {source}")]
    Annotation {
        /// Unit filename for context.
        file: String,
        /// The underlying annotation error.
        source: annotation::AnnotationError,
    },
    /// The unit contains no `main` entry point — fatal in the reference
    /// (docs/shader-pipeline.md §3.1 `ShaderUnit.cpp:311-313`).
    #[error("no `main` entry point found in {file}")]
    NoMain {
        /// Unit filename for context.
        file: String,
    },
    /// Both the naga GLSL frontend and the shaderc fallback rejected the
    /// modernized source. Carries both diagnostics.
    #[error("translation failed for {file}:\n  naga glsl-in: {naga}\n  shaderc: {shaderc}")]
    Compile {
        /// Unit filename for context.
        file: String,
        /// naga GLSL frontend diagnostic.
        naga: String,
        /// shaderc / SPIR-V-frontend diagnostic.
        shaderc: String,
    },
    /// The produced module failed naga validation (the same validation wgpu runs
    /// before creating a module).
    #[error("naga validation failed for {file}: {diag}")]
    Validate {
        /// Unit filename for context.
        file: String,
        /// Validation diagnostic.
        diag: String,
    },
}

/// Translate one raw `.vert`/`.frag` unit into a wgpu-ready module + reflection.
///
/// `filename` is the source name used for diagnostics (docs/shader-pipeline.md
/// §1.2). `resolver` supplies `#include` bodies; `inputs` supplies material combos
/// and populated texture slots. See the module docs for the pipeline shape.
pub fn translate(
    stage: Stage,
    filename: &str,
    source: &str,
    resolver: &dyn IncludeResolver,
    inputs: &ShaderInputs,
) -> Result<TranslatedShader, TranslateError> {
    // Warm path: a unit cached against the RAW inputs skips preprocessing,
    // modernization AND translation. Include bodies are verified against the
    // hashes recorded at build time (see translate::UnitEntry), so an edited
    // header is a clean miss, never a stale hit.
    let unit_key = translate::unit_cache_key(stage, source, inputs);
    if let Some(ts) = translate::unit_cache_load(&unit_key, resolver) {
        return Ok(ts);
    }

    // Cold path — record every include body the preprocessor pulls so the
    // stored unit can be verified on later loads.
    let recording = RecordingResolver {
        inner: resolver,
        seen: std::cell::RefCell::new(Vec::new()),
    };
    let assembled = preprocess::preprocess(stage, filename, source, &recording, inputs)?;
    let (glsl, reflection) = modernize::modernize(stage, assembled);
    let out = translate::translate(stage, filename, glsl, reflection)?;
    translate::unit_cache_store(&unit_key, recording.seen.into_inner(), &out);
    Ok(out)
}

/// Wraps an [`IncludeResolver`], recording `(name, blake3(body))` for every
/// successful resolve — the dependency set the unit cache verifies on load.
struct RecordingResolver<'a> {
    inner: &'a dyn IncludeResolver,
    seen: std::cell::RefCell<Vec<(String, [u8; 32])>>,
}

impl IncludeResolver for RecordingResolver<'_> {
    fn resolve(&self, include_name: &str) -> Option<String> {
        let body = self.inner.resolve(include_name)?;
        let mut seen = self.seen.borrow_mut();
        if !seen.iter().any(|(n, _)| n == include_name) {
            seen.push((include_name.to_owned(), *blake3::hash(body.as_bytes()).as_bytes()));
        }
        Some(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoIncludes;
    impl IncludeResolver for NoIncludes {
        fn resolve(&self, _: &str) -> Option<String> {
            None
        }
    }

    #[test]
    fn translate_minimal_fragment_with_sampler_and_uniform() {
        // Exercises the whole pipeline end-to-end (no GPU): combined sampler
        // split, loose-uniform packing, gl_FragColor rewrite, translation +
        // validation.
        let src = "\
uniform sampler2D g_Texture0; // {\"default\":\"util/white\"}\n\
uniform float g_Brightness; // {\"material\":\"Brightness\",\"default\":1}\n\
varying vec2 v_TexCoord;\n\
void main() {\n\
    gl_FragColor = texSample2D(g_Texture0, v_TexCoord) * g_Brightness;\n\
}\n";
        let ts = translate(
            Stage::Fragment,
            "test.frag",
            src,
            &NoIncludes,
            &ShaderInputs::default(),
        )
        .expect("translation should succeed");
        // The module has a fragment entry point named `main`.
        assert!(ts.module.entry_points.iter().any(|e| e.name == "main"));
        // Reflection captured the sampler slot and the parameter.
        assert_eq!(ts.reflection.samplers.len(), 1);
        assert_eq!(ts.reflection.samplers[0].slot, Some(0));
        assert_eq!(ts.reflection.parameters.len(), 1);
        assert_eq!(ts.reflection.parameters[0].material, "Brightness");
        assert_eq!(ts.reflection.globals_block, vec!["g_Brightness"]);
    }

    #[test]
    fn translate_minimal_vertex() {
        let src = "\
attribute vec3 a_Position;\n\
attribute vec2 a_TexCoord;\n\
varying vec2 v_TexCoord;\n\
uniform mat4 g_ModelViewProjectionMatrix;\n\
void main() {\n\
    v_TexCoord = a_TexCoord;\n\
    gl_Position = mul(g_ModelViewProjectionMatrix, vec4(a_Position, 1.0));\n\
}\n";
        let ts = translate(
            Stage::Vertex,
            "test.vert",
            src,
            &NoIncludes,
            &ShaderInputs::default(),
        )
        .expect("vertex translation should succeed");
        assert!(ts.module.entry_points.iter().any(|e| e.name == "main"));
        // Attributes were reflected.
        let names: Vec<&str> = ts.reflection.attributes.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"a_Position"));
        assert!(names.contains(&"a_TexCoord"));
    }

    #[test]
    fn translator_version_is_stable() {
        assert_eq!(TRANSLATOR_VERSION, 3);
    }
}
