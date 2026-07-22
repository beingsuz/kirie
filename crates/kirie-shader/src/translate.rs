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
///
/// The expensive work (preprocess → naga-GLSL / glslang frontend → validate —
/// seconds on a big scene) is served from an on-disk cache of the **serialized
/// naga module**, keyed by the input GLSL. A warm load deserializes the cached
/// module (a few ms) instead of retranslating, and it's **lossless** — serde
/// round-trips the *exact* module the cold path produced (unlike re-emitting
/// SPIR-V from the module, which recompiles and changes the shader). wgpu
/// re-validates on `create_shader_module`, so skipping our own validate on the
/// warm path is safe. Cache I/O is best-effort — any miss/corruption falls
/// through to a full translate, so correctness never depends on the cache.
pub fn translate(
    stage: Stage,
    filename: &str,
    modernized: String,
    reflection: Reflection,
) -> Result<TranslatedShader, TranslateError> {
    let key = shader_cache_key(stage, &modernized);
    if let Some(module) = cache_load_module(&key) {
        return Ok(TranslatedShader {
            module,
            reflection,
            path: TranslatePath::Shaderc,
            glsl: String::new(),
        });
    }

    // Step 1: preprocess (macro/`#if` expansion) + array-varying flatten, then
    // reproduce the patched-glslang shape/type leniencies (docs/shader-pipeline.md
    // §7.1, §7.2) that stock glslang and naga reject.
    let flat = preprocess_and_flatten(stage, filename, &modernized);
    let flat = crate::coerce::coerce_shapes(&flat);

    // Step 2: naga pure-Rust GLSL frontend, then validate. On success cache the
    // validated module; on failure fall through to the shaderc route.
    let naga_diag = match try_naga_glsl(stage, &flat).and_then(validate) {
        Ok(module) => {
            cache_store_module(&key, &module);
            return Ok(TranslatedShader {
                module,
                reflection,
                path: TranslatePath::NagaGlsl,
                glsl: flat,
            });
        }
        Err(e) => e,
    };

    // Step 3: shaderc → SPIR-V → naga SPIR-V frontend, then validate + cache.
    let shaderc_diag = match try_shaderc(stage, filename, &flat).and_then(validate) {
        Ok(module) => {
            cache_store_module(&key, &module);
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
/// diagnostic string. The glslang compile (the expensive step, seconds on big
/// shaders) is served from an on-disk SPIR-V cache when possible, so the same
/// shader across loads/scenes only compiles once — this is what turns a heavy
/// scene's ~18s first frame into a warm-load in well under a second.
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

/// A fully translated unit cached against the RAW inputs: skipping not just
/// glslang/naga (the module cache below) but preprocessing + modernization +
/// validation entirely on warm loads. `includes` records every `#include`
/// body's content hash observed at build; a hit re-resolves each and compares,
/// so an edited header is a clean miss (the depfile approach).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct UnitEntry {
    includes: Vec<(String, [u8; 32])>,
    glsl: String,
    path: crate::TranslatePath,
    reflection: crate::Reflection,
    module: naga::Module,
}

/// Cache key over the RAW translate() inputs: source, stage, canonicalized
/// [`crate::ShaderInputs`] and the translator/format versions. Include bodies
/// are deliberately NOT keyed (unknowable before preprocessing) — they are
/// verified per entry via the recorded hashes.
pub(crate) fn unit_cache_key(stage: Stage, source: &str, inputs: &crate::ShaderInputs) -> String {
    const UNIT_FORMAT: u32 = 1;
    let mut h = blake3::Hasher::new();
    h.update(source.as_bytes());
    h.update(&[match stage {
        Stage::Vertex => 0u8,
        Stage::Fragment => 1u8,
    }]);
    for (k, v) in &inputs.combos {
        h.update(k.as_bytes());
        h.update(&v.to_le_bytes());
        h.update(&[0xfe]);
    }
    h.update(&[0xfd]);
    for (k, v) in &inputs.override_combos {
        h.update(k.as_bytes());
        h.update(&v.to_le_bytes());
        h.update(&[0xfe]);
    }
    h.update(&[0xfd]);
    for slot in &inputs.populated_texture_slots {
        h.update(&slot.to_le_bytes());
    }
    h.update(&crate::TRANSLATOR_VERSION.to_le_bytes());
    h.update(&UNIT_FORMAT.to_le_bytes());
    h.finalize().to_hex().to_string()
}

/// Load a cached unit for `key`, verifying every recorded include body still
/// resolves to the same content. `None` ⇒ full translate.
pub(crate) fn unit_cache_load(
    key: &str,
    resolver: &dyn crate::IncludeResolver,
) -> Option<crate::TranslatedShader> {
    let bytes = std::fs::read(spirv_cache_dir()?.join(format!("{key}.tng"))).ok()?;
    let entry: UnitEntry = bincode::deserialize(&bytes).ok()?;
    for (name, hash) in &entry.includes {
        let body = resolver.resolve(name)?;
        if blake3::hash(body.as_bytes()).as_bytes() != hash {
            return None;
        }
    }
    Some(crate::TranslatedShader {
        module: entry.module,
        reflection: entry.reflection,
        path: entry.path,
        glsl: entry.glsl,
    })
}

/// Store a translated unit for `key` (best-effort, atomic).
pub(crate) fn unit_cache_store(
    key: &str,
    includes: Vec<(String, [u8; 32])>,
    ts: &crate::TranslatedShader,
) {
    let entry = UnitEntry {
        includes,
        glsl: ts.glsl.clone(),
        path: ts.path,
        reflection: ts.reflection.clone(),
        module: ts.module.clone(),
    };
    if let (Ok(bytes), Some(dir)) = (bincode::serialize(&entry), spirv_cache_dir()) {
        write_cache_atomic(&dir.join(format!("{key}.tng")), &bytes);
        maybe_prune_cache(&dir);
    }
}

/// Cache key for a shader's serialized naga module: `blake3(modernized ‖ stage ‖
/// TRANSLATOR_VERSION ‖ CACHE_FORMAT)`. Bumping [`crate::TRANSLATOR_VERSION`] or
/// the format tag invalidates every entry; a different shader or stage never
/// collides.
fn shader_cache_key(stage: Stage, modernized: &str) -> String {
    // Bump when the cached representation changes (currently: bincode naga::Module).
    const CACHE_FORMAT: u32 = 3;
    let mut h = blake3::Hasher::new();
    h.update(modernized.as_bytes());
    h.update(&[match stage {
        Stage::Vertex => 0u8,
        Stage::Fragment => 1u8,
    }]);
    h.update(&crate::TRANSLATOR_VERSION.to_le_bytes());
    h.update(&CACHE_FORMAT.to_le_bytes());
    h.finalize().to_hex().to_string()
}

/// Deserialize the cached naga module for `key`, if present and valid. `None`
/// (miss/corrupt) ⇒ full translate.
fn cache_load_module(key: &str) -> Option<naga::Module> {
    let bytes = std::fs::read(spirv_cache_dir()?.join(format!("{key}.nga"))).ok()?;
    bincode::deserialize(&bytes).ok()
}

/// Serialize a validated naga module for `key` (best-effort, atomic).
fn cache_store_module(key: &str, module: &naga::Module) {
    if let (Ok(bytes), Some(dir)) = (bincode::serialize(module), spirv_cache_dir()) {
        write_cache_atomic(&dir.join(format!("{key}.nga")), &bytes);
        maybe_prune_cache(&dir);
    }
}

/// Byte cap for a shader-module cache dir; oldest `.nga` files are evicted past
/// it. Modules are small (a bincode naga module), so this holds thousands.
const SHADER_CACHE_CAP_BYTES: u64 = 128 * 1024 * 1024;

/// Prune `dir` under [`SHADER_CACHE_CAP_BYTES`], oldest-first — at most once per
/// dir per process. Without this, bumping `TRANSLATOR_VERSION`/`CACHE_FORMAT`
/// orphans every key and the cache grows unbounded. Best-effort.
fn maybe_prune_cache(dir: &std::path::Path) {
    use std::sync::{Mutex, OnceLock};
    static SWEPT: OnceLock<Mutex<std::collections::HashSet<std::path::PathBuf>>> = OnceLock::new();
    let swept = SWEPT.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let Ok(mut guard) = swept.lock() else { return };
    if !guard.insert(dir.to_path_buf()) {
        return; // already swept this dir this process
    }
    drop(guard); // release the lock before the filesystem sweep
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::path::PathBuf, u64, std::time::SystemTime)> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "nga" || x == "tng"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((e.path(), m.len(), m.modified().ok()?))
        })
        .collect();
    let mut remaining: u64 = files.iter().map(|f| f.1).sum();
    if remaining <= SHADER_CACHE_CAP_BYTES {
        return;
    }
    files.sort_by(|a, b| a.2.cmp(&b.2)); // oldest first
    for (path, len, _) in files {
        if remaining <= SHADER_CACHE_CAP_BYTES {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            remaining -= len;
        }
    }
}

thread_local! {
    /// Per-build cache directory override (e.g. a folder beside the wallpaper),
    /// used in preference to the global cache. Set for the current thread at the
    /// start of a scene load; the worker thread that builds a preload/swap sets
    /// its own.
    static CACHE_DIR_OVERRIDE: std::cell::RefCell<Option<std::path::PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Set a per-build shader-cache directory for the current thread, or `None` to
/// clear it (falling back to the global cache). A scene loader sets this to a
/// folder beside the wallpaper so its compiled shaders persist with it and never
/// rebuild, independent of `~/.cache`.
pub fn set_cache_dir(dir: Option<std::path::PathBuf>) {
    CACHE_DIR_OVERRIDE.with(|c| *c.borrow_mut() = dir);
}

/// The active shader-cache directory: the per-build override if set, else
/// `$XDG_CACHE_HOME/kirie/shaders` (or `$HOME/.cache/kirie/shaders`). `None`
/// disables the cache.
fn spirv_cache_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = CACHE_DIR_OVERRIDE.with(|c| c.borrow().clone()) {
        return Some(dir);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("kirie").join("shaders"))
}

/// Write `bytes` to `path` atomically (temp + rename), best-effort — a failure to
/// create the dir or write is ignored so the cache never breaks a build.
fn write_cache_atomic(path: &std::path::Path, bytes: &[u8]) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("x");
    let tmp = dir.join(format!(".tmp-{}-{stem}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
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
