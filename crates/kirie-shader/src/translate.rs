//! Translate modernized GLSL into a validated naga module (docs/shader-pipeline.md
//! §9.1.2, §9.3).
//!
//! Route, per shader:
//! 1. Run the modernized source through `libshaderc`'s **preprocessor only** to
//!    resolve the macro prelude and combo `#if`s, then flatten constant-indexed
//!    array varyings (docs/shader-pipeline.md §9.2.3) — arrays are not valid
//!    entry-point IO for wgpu.
//! 2. Try naga's pure-Rust GLSL frontend on the flat source
//!    ([`crate::TranslatePath::NagaGlsl`]).
//! 3. On failure, fall back to `libshaderc` (stock glslang) → SPIR-V → naga's
//!    SPIR-V frontend ([`crate::TranslatePath::Shaderc`]).
//!
//! The resulting module is validated with the same validator wgpu runs before
//! `create_shader_module`, so a returned [`crate::TranslatedShader`] is
//! GPU-loadable.

use naga::front::glsl::{Frontend as GlslFrontend, Options as GlslOptions};
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::reflect::Reflection;
use crate::{Stage, TranslateError, TranslatePath, TranslatedShader};

/// Translate modernized GLSL for `stage` into a validated module + reflection.
pub fn translate(
    stage: Stage,
    filename: &str,
    modernized: String,
    reflection: Reflection,
) -> Result<TranslatedShader, TranslateError> {
    // Step 1: preprocess (macro/`#if` expansion) + array-varying flatten, then
    // reproduce the patched-glslang shape/type leniencies (docs/shader-pipeline.md
    // §7.1, §7.2) that stock glslang and naga reject.
    let flat = preprocess_and_flatten(stage, filename, &modernized);
    let flat = crate::coerce::coerce_shapes(&flat);

    // Step 2: naga pure-Rust GLSL frontend, then validate. A parse *or*
    // validation failure falls through to the shaderc route.
    let naga_diag = match try_naga_glsl(stage, &flat).and_then(validate) {
        Ok(module) => {
            return Ok(TranslatedShader {
                module,
                reflection,
                path: TranslatePath::NagaGlsl,
                glsl: flat,
            });
        }
        Err(e) => e,
    };

    // Step 3: shaderc → SPIR-V → naga SPIR-V frontend, then validate.
    let shaderc_diag = match try_shaderc(stage, filename, &flat).and_then(validate) {
        Ok(module) => {
            return Ok(TranslatedShader {
                module,
                reflection,
                path: TranslatePath::Shaderc,
                glsl: flat,
            });
        }
        Err(e) => e,
    };

    Err(TranslateError::Compile {
        file: filename.to_string(),
        naga: naga_diag,
        shaderc: shaderc_diag,
    })
}

/// Validate a module with wgpu-equivalent settings; returns the module or a
/// diagnostic string so the caller can fall back or aggregate.
fn validate(module: naga::Module) -> Result<naga::Module, String> {
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    match validator.validate(&module) {
        Ok(_) => Ok(module),
        Err(e) => Err(format!("validation: {:?}", e.as_inner())),
    }
}

/// Build shaderc options matching the reference environment as closely as a
/// stock toolchain allows (docs/shader-pipeline.md §5: OpenGL 4.5 client,
/// auto-mapped locations + bindings). We target Vulkan SPIR-V because that is
/// what naga's SPIR-V frontend and wgpu consume.
fn shaderc_options() -> Option<shaderc::CompileOptions<'static>> {
    let mut opts = shaderc::CompileOptions::new().ok()?;
    opts.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
    opts.set_target_spirv(shaderc::SpirvVersion::V1_3);
    // autoMapBindings / autoMapLocations (docs/shader-pipeline.md §5).
    opts.set_auto_bind_uniforms(true);
    opts.set_auto_map_locations(true);
    Some(opts)
}

/// Run shaderc's preprocessor over the modernized source and flatten array
/// varyings. On preprocessor failure, fall back to the unpreprocessed source
/// (the frontends run their own preprocessing then).
fn preprocess_and_flatten(stage: Stage, filename: &str, modernized: &str) -> String {
    let Some(compiler) = shaderc::Compiler::new().ok() else {
        return modernized.to_string();
    };
    let Some(opts) = shaderc_options() else {
        return modernized.to_string();
    };
    let _ = stage;
    match compiler.preprocess(modernized, filename, "main", Some(&opts)) {
        Ok(pp) => flatten_array_varyings(&pp.as_text()),
        Err(_) => modernized.to_string(),
    }
}

/// Attempt naga's GLSL frontend; returns the module or a diagnostic string.
fn try_naga_glsl(stage: Stage, src: &str) -> Result<naga::Module, String> {
    let mut frontend = GlslFrontend::default();
    let options = GlslOptions::from(stage.naga());
    frontend.parse(&options, src).map_err(|e| e.emit_to_string(src))
}

/// Attempt the shaderc → SPIR-V → naga SPIR-V path; returns the module or a
/// diagnostic string.
fn try_shaderc(stage: Stage, filename: &str, src: &str) -> Result<naga::Module, String> {
    let compiler = shaderc::Compiler::new().map_err(|e| e.to_string())?;
    let opts = shaderc_options().ok_or_else(|| "shaderc options unavailable".to_string())?;
    let kind = match stage {
        Stage::Vertex => shaderc::ShaderKind::Vertex,
        Stage::Fragment => shaderc::ShaderKind::Fragment,
    };
    let artifact = compiler
        .compile_into_spirv(src, kind, filename, "main", Some(&opts))
        .map_err(|e| first_error_line(&e.to_string()))?;
    // wgpu/naga does not flip clip space here; we keep coordinates as emitted.
    let spv_opts = naga::front::spv::Options {
        adjust_coordinate_space: false,
        ..Default::default()
    };
    naga::front::spv::parse_u8_slice(artifact.as_binary_u8(), &spv_opts).map_err(|e| format!("{e:?}"))
}

/// Extract the first `: error:` line from a shaderc diagnostic for concision.
fn first_error_line(msg: &str) -> String {
    msg.lines()
        .find(|l| l.contains(": error:"))
        .unwrap_or_else(|| msg.lines().last().unwrap_or(msg))
        .trim()
        .to_string()
}

/// A discovered array varying: `in|out TYPE NAME[COUNT];`.
struct ArrayVarying {
    /// Base name (e.g. `v_TexCoord`).
    name: String,
    /// Element type (e.g. `vec2`, `vec4`).
    ty: String,
    /// Fixed element count from the declaration.
    count: usize,
    /// `true` for an `out` (vertex→fragment producer), `false` for an `in`.
    is_out: bool,
    /// `true` if any use indexes the array with a non-literal expression.
    dynamic: bool,
}

/// Flatten array varyings so they can cross the wgpu entry-point IO boundary
/// (docs/shader-pipeline.md §9.2.3): wgpu / naga cannot use an array as
/// entry-point IO, so a declaration like `in vec2 v_TexCoord[13];` is split into
/// 13 scalar varyings `v_TexCoord_0 … v_TexCoord_12`. Inter-stage matching stays
/// by name, so paired vertex/fragment units flatten identically.
///
/// Two use patterns are handled:
///
/// * **Constant-indexed** (`v_TexCoord[3]`): each literal index is rewritten to
///   its scalar `v_TexCoord_3`. This covers the blur/gaussian downsample family
///   that only ever indexes with compile-time constants.
/// * **Dynamic-indexed** (`v_TexCoord[i]`, `audioValue[k][j]`): a scalar array
///   cannot be indexed by a runtime value, so the transported scalars are bridged
///   through a function-local array reconstructed inside `main` — for an `in`
///   varying the scalars are copied *into* the local at entry; for an `out`
///   varying the local is copied *out* to the scalars before return. Every
///   dynamic (and constant) index then operates on the local array, which naga /
///   wgpu index normally. This lifts the loop-summed oscilloscope, light-map, and
///   dynamically-indexed downsample shaders (docs/shader-pipeline.md §7.2 float
///   indices flow through `int(...)` on emission).
fn flatten_array_varyings(src: &str) -> String {
    let arrays = discover_array_varyings(src);
    if arrays.is_empty() {
        return src.to_string();
    }

    // Rewrite every array declaration line into `count` scalar varyings.
    let mut out = String::with_capacity(src.len() + 256);
    'lines: for line in src.lines() {
        let t = line.trim();
        for a in &arrays {
            let pat = format!("{}[{}]", a.name, a.count);
            if (t.starts_with("in ") || t.starts_with("out ")) && t.contains(&pat) {
                let kw = if t.starts_with("in ") { "in" } else { "out" };
                for i in 0..a.count {
                    out.push_str(&format!("{kw} {} {}_{i};\n", a.ty, a.name));
                }
                continue 'lines;
            }
        }
        out.push_str(line);
        out.push('\n');
    }

    // Constant-only arrays: rewrite each literal-index use to its scalar.
    let mut result = out;
    for a in &arrays {
        if a.dynamic {
            continue;
        }
        for i in 0..a.count {
            result = result.replace(&format!("{}[{i}]", a.name), &format!("{}_{i}", a.name));
        }
    }

    // Dynamic arrays: bridge the scalars through a reconstructed local array.
    let dyn_arrays: Vec<&ArrayVarying> = arrays.iter().filter(|a| a.dynamic).collect();
    if dyn_arrays.is_empty() {
        return result;
    }
    reconstruct_dynamic_arrays(&result, &dyn_arrays)
}

/// Discover `in|out TYPE NAME[COUNT];` array varyings and classify each as
/// constant- or dynamic-indexed by scanning its uses.
fn discover_array_varyings(src: &str) -> Vec<ArrayVarying> {
    let mut arrays: Vec<ArrayVarying> = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        let (is_out, rest) = if let Some(r) = t.strip_prefix("in ") {
            (false, r)
        } else if let Some(r) = t.strip_prefix("out ") {
            (true, r)
        } else {
            continue;
        };
        let Some(decl) = rest.strip_suffix(';') else {
            continue;
        };
        let Some(open) = decl.find('[') else { continue };
        let Some(close) = decl.find(']') else { continue };
        // Malformed `]`-before-`[` must not panic-slice (SPEC.md §V9): skip it.
        if close <= open {
            continue;
        }
        let count: usize = decl[open + 1..close].trim().parse().unwrap_or(0);
        let head = decl[..open].trim();
        let mut parts = head.split_whitespace();
        let (Some(ty), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let dynamic = array_has_dynamic_index(src, name);
        arrays.push(ArrayVarying {
            name: name.to_string(),
            ty: ty.to_string(),
            count,
            is_out,
            dynamic,
        });
    }
    arrays
}

/// True if `name[` is ever indexed by something other than a plain integer
/// literal (a whole-token match of `name`, so `foo` never matches `myfoo`).
fn array_has_dynamic_index(src: &str, name: &str) -> bool {
    let bytes = src.as_bytes();
    let mut search = 0;
    while let Some(rel) = src[search..].find(name) {
        let start = search + rel;
        let end = start + name.len();
        search = end;
        // Whole-token boundary check.
        let before_ok = start
            .checked_sub(1)
            .map(|b| !is_ident_byte(bytes[b]))
            .unwrap_or(true);
        if !before_ok || bytes.get(end) != Some(&b'[') {
            continue;
        }
        // Extract the bracketed index text.
        let Some(close_rel) = src[end + 1..].find(']') else {
            continue;
        };
        let inner = src[end + 1..end + 1 + close_rel].trim();
        if inner.is_empty() {
            // A bare `name[]` (declaration form) — not an index use.
            continue;
        }
        if !inner.bytes().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}

/// True if `b` can appear inside a GLSL identifier.
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Reconstruct dynamically-indexed array varyings as a `main`-local array bridged
/// to the transported scalar varyings (see [`flatten_array_varyings`]).
fn reconstruct_dynamic_arrays(src: &str, arrays: &[&ArrayVarying]) -> String {
    let Some((body_open, body_close)) = main_brace_span(src) else {
        // No locatable `main` body — leave the source for the frontend to reject.
        return src.to_string();
    };

    let mut decls = String::new();
    let mut copy_in = String::new();
    let mut copy_out = String::new();
    for a in arrays {
        decls.push_str(&format!("{} {}[{}];\n", a.ty, a.name, a.count));
        if a.is_out {
            for i in 0..a.count {
                copy_out.push_str(&format!("{}_{i} = {}[{i}];\n", a.name, a.name));
            }
        } else {
            for i in 0..a.count {
                copy_in.push_str(&format!("{}[{i}] = {}_{i};\n", a.name, a.name));
            }
        }
    }

    let mut s = String::with_capacity(src.len() + decls.len() + copy_in.len() + copy_out.len() + 4);
    s.push_str(&src[..body_open]);
    s.push('\n');
    s.push_str(&decls);
    s.push_str(&copy_in);
    s.push_str(&src[body_open..body_close]);
    s.push_str(&copy_out);
    s.push_str(&src[body_close..]);
    s
}

/// Locate the `main` function body: returns `(index just after its opening `{`,
/// index of the matching `}`)`. Brace-matched so nested blocks are handled.
fn main_brace_span(src: &str) -> Option<(usize, usize)> {
    let bytes = src.as_bytes();
    // Find a `main` token followed (past spaces) by `(`.
    let mut search = 0;
    let main_at = loop {
        let rel = src[search..].find("main")?;
        let start = search + rel;
        let end = start + 4;
        search = end;
        let before_ok = start
            .checked_sub(1)
            .map(|b| !is_ident_byte(bytes[b]))
            .unwrap_or(true);
        let after_paren = src[end..].trim_start().starts_with('(');
        if before_ok && after_paren {
            break start;
        }
    };
    let open = src[main_at..].find('{')? + main_at;
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((open + 1, i));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}
