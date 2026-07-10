//! The `ShaderUnit` preprocessing + assembly stages (docs/shader-pipeline.md
//! §3, §4): `#include` inlining, `#require` module synthesis, annotation
//! scanning, combo discovery + require-chain resolution + precedence emission,
//! `gl_FragColor` rewriting, and final source assembly with the HLSL-compat
//! macro prelude.
//!
//! Deviations from the reference tail are deliberate and documented inline; the
//! content-visible behavior of §3-§4 is reproduced (docs/shader-pipeline.md
//! §9.1.1). Nothing panics on malformed input (SPEC.md §V9).

use std::collections::BTreeMap;

use crate::annotation::{self, UniformAnnotation};
use crate::reflect::{Parameter, Reflection, SamplerSlot, VertexAttribute};
use crate::{IncludeResolver, ShaderInputs, Stage, TranslateError};

/// The HLSL-compat macro prelude, byte-for-byte from `SHADER_HEADER`
/// (docs/shader-pipeline.md §4.1, `ShaderUnit.cpp:22-54`) **minus** the leading
/// `#version` line, which [`crate::modernize`] re-emits at the wgpu-required
/// version. Semantics preserved verbatim: `mul`/`max` operand swaps, `ddy`
/// negation, HLSL trunc-remainder `fmod` (docs/shader-pipeline.md §4.1 notes).
pub const PRELUDE_MACROS: &str = r#"precision highp float;
#define mul(x, y) ((y) * (x))
#define max(x, y) max (y, x)
#define lerp mix
#define frac fract
#define CAST2(x) (vec2(x))
#define CAST3(x) (vec3(x))
#define CAST4(x) (vec4(x))
#define CAST3X3(x) (mat3(x))
#define CASTF(x) (float(x))
#define CASTU(x) (uint(x))
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define saturate(x) (clamp(x, 0.0, 1.0))
#define texSample2D texture
#define texSample2DLod textureLod
#define log10(x) (log2(x) * 0.301029995663981)
#define atan2 atan
#define fmod(x, y) ((x)-(y)*trunc((x)/(y)))
#define ddx dFdx
#define ddy(x) dFdy(-(x))
#define GLSL 1
"#;

/// The `#require LightingV1` stub (docs/shader-pipeline.md §3.2,
/// `ShaderUnit.cpp:369-380`): official WE generates this from scene lights; the
/// linux fork emits a no-light stub, which we reproduce.
const LIGHTING_V1_STUB: &str = "vec3 PerformLighting_V1(vec3 worldPos, vec3 albedo, vec3 normal, vec3 viewDir, vec3 specularTint, vec3 baseReflectance, float roughness, float metallic) { return vec3(0.0); }\n";

/// Output of preprocessing: the assembled legacy-GLSL source (still using
/// `uniform sampler2D`, loose uniforms, and `varying`/`attribute` keywords) plus
/// the reflection table captured from annotations.
#[derive(Debug, Clone)]
pub struct Assembled {
    /// Assembled source: prelude macros + per-stage defines + combo `#define`s +
    /// the include-inlined, `gl_FragColor`-rewritten body. No `#version` line
    /// (added by [`crate::modernize`]).
    pub source: String,
    /// Reflection captured during the annotation scan (docs/shader-pipeline.md §8.2).
    pub reflection: Reflection,
}

/// Normalize an `#include "F"` name to the header path the reference loads:
/// replace the extension with `.h` (docs/shader-pipeline.md §1.2,
/// `AssetLocator.cpp:56-62`). In practice `F` already ends in `.h`.
fn header_name(raw: &str) -> String {
    match raw.rsplit_once('.') {
        Some((base, _ext)) => format!("{base}.h"),
        None => format!("{raw}.h"),
    }
}

/// Inline `#include "x"` directives (docs/shader-pipeline.md §3.1). Missing
/// includes become a comment, never an error (`ShaderUnit.cpp:161-165`). There
/// is no include-once guard, matching the reference; a recursion cap guards
/// against pathological cycles (SPEC.md §V9) rather than reproducing a stack
/// overflow.
fn resolve_includes(src: &str, resolver: &dyn IncludeResolver, depth: usize) -> String {
    const MAX_DEPTH: usize = 32;
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#include")
            && let Some(name) = extract_quoted(rest)
        {
            let header = header_name(&name);
            if let Some(content) = (depth < MAX_DEPTH).then(|| resolver.resolve(&header)).flatten() {
                out.push_str(&format!("// begin of include from file {header}\n"));
                out.push_str(&resolve_includes(&content, resolver, depth + 1));
                out.push_str(&format!("\n// end of included from file {header}\n"));
                continue;
            }
            // Non-failing miss (docs/shader-pipeline.md §3.1).
            out.push_str(&format!("// tried including file {name} but was not found\n"));
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Expand `#require <Module>` (docs/shader-pipeline.md §3.2). Only `LightingV1`
/// is known; unknown modules expand to nothing (reference logs an error).
fn resolve_requires(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#require") {
            let module = rest.trim();
            if module == "LightingV1" {
                out.push_str(LIGHTING_V1_STUB);
            }
            // Unknown module → nothing (docs/shader-pipeline.md §3.2).
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Extract the first double-quoted substring from `s`, if any.
fn extract_quoted(s: &str) -> Option<String> {
    let start = s.find('"')? + 1;
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

/// Uppercase a combo name (docs/shader-pipeline.md §4.3: combos are uppercased
/// at emission; corpus materials carry lower-case keys).
fn upper(name: &str) -> String {
    name.to_uppercase()
}

/// Slot index for a `g_Texture<N>` sampler: the single digit after `g_Texture`
/// (docs/shader-pipeline.md §2.2, `ShaderUnit.cpp:595-600`). Slots 0-9 only.
fn sampler_slot(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("g_Texture")?;
    let c = rest.as_bytes().first().copied()?;
    if c.is_ascii_digit() {
        Some((c - b'0') as u32)
    } else {
        None
    }
}

/// Run the full preprocessing pipeline (docs/shader-pipeline.md §3, §4).
pub fn preprocess(
    stage: Stage,
    filename: &str,
    source: &str,
    resolver: &dyn IncludeResolver,
    inputs: &ShaderInputs,
) -> Result<Assembled, TranslateError> {
    // §3.1, §3.2: include + require expansion happen before the annotation scan
    // so annotations inside headers are honored (docs/shader-pipeline.md §2.1).
    let included = resolve_includes(source, resolver, 0);
    let expanded = resolve_requires(&included);

    // §3.1: a unit with no `main` is fatal in the reference.
    if !contains_main(&expanded) {
        return Err(TranslateError::NoMain {
            file: filename.to_string(),
        });
    }

    // §3.3: annotation scan over the post-include text.
    let mut discovered: BTreeMap<String, i32> = BTreeMap::new();
    let mut combo_requires: BTreeMap<String, BTreeMap<String, i32>> = BTreeMap::new();
    let mut parameters: Vec<Parameter> = Vec::new();
    let mut samplers: Vec<SamplerSlot> = Vec::new();
    let mut attributes: Vec<VertexAttribute> = Vec::new();
    let mut next_attr_loc = 0u32;

    for line in expanded.lines() {
        // [COMBO] annotations (docs/shader-pipeline.md §2.1).
        match annotation::parse_combo_line(line) {
            Ok(Some(combo)) => {
                let name = upper(&combo.combo);
                discovered.entry(name.clone()).or_insert(combo.default);
                if !combo.require.is_empty() {
                    let reqs = combo.require.iter().map(|(k, v)| (upper(k), *v)).collect();
                    combo_requires.insert(name, reqs);
                }
            }
            Ok(None) => {}
            Err(source) => {
                return Err(TranslateError::Annotation {
                    file: filename.to_string(),
                    source,
                });
            }
        }

        // Annotated uniforms (docs/shader-pipeline.md §2.2).
        if let Ok(Some(uni)) = annotation::parse_uniform_line(line) {
            match uni {
                UniformAnnotation::Parameter {
                    name,
                    ty,
                    material,
                    default,
                } => {
                    // Only uniforms with a `material` link are registered as
                    // bindable parameters (docs/shader-pipeline.md §2.2).
                    if let Some(material) = material {
                        parameters.push(Parameter {
                            name,
                            material,
                            ty,
                            default,
                        });
                    }
                }
                UniformAnnotation::Sampler {
                    name,
                    default_texture,
                    combo,
                    ..
                } => {
                    let slot = sampler_slot(&name);
                    // Sampler combo gating (docs/shader-pipeline.md §2.2 rule 1):
                    // a populated slot forces its combo to 1. We treat a slot as
                    // populated when the inputs list it or the annotation carries
                    // a `default` texture (the common case: util/white etc.).
                    // The `require`/`requireany` edge cases (rules 2-3) are noted
                    // UNVERIFIED in the spec and left to material context.
                    if let Some(ref combo_name) = combo {
                        let populated = slot
                            .map(|s| inputs.populated_texture_slots.contains(&s))
                            .unwrap_or(false)
                            || default_texture.is_some();
                        if populated {
                            discovered.entry(upper(combo_name)).or_insert(1);
                        }
                    }
                    samplers.push(SamplerSlot {
                        name,
                        slot,
                        // Bindings assigned during modernization.
                        texture_binding: 0,
                        sampler_binding: 0,
                        default_texture,
                        combo,
                    });
                }
            }
        }
    }

    // §3.4 + §4.3: resolve final combo values with require chains + precedence.
    let active = resolve_combos(&discovered, &combo_requires, inputs);

    // Vertex attributes (reflection only; docs/shader-pipeline.md §8.2). Scanned
    // *after* combo resolution and with `#if`/`#ifdef` awareness so that a
    // skinning/model attribute declared inside an inactive branch (e.g.
    // `attribute vec4 a_BlendIndices;` under `#if SKINNING` with SKINNING off) is
    // NOT registered — the frontend strips that block, so the compiled module has
    // no such input and the location numbering must match (the reference resolves
    // `#if` before this scan; a stale attribute otherwise fails pipeline build).
    // I/O an active branch *declares* but never *consumes* must be dropped from
    // both the reflection and the compiled source. `flat.vert` declares
    // `attribute vec4 a_Color;` and `flat.frag` declares `varying vec4 v_Color;`
    // unconditionally, yet each is used only under `#ifdef VERTEXCOLOR`, which the
    // solid-layer material leaves off. Left in, the GLSL frontend still emits a
    // live vertex input (`a_Color`, unfeedable by the 2D VAO) and a live fragment
    // input (`v_Color`, at a location the vertex stage never writes) — either one
    // makes the whole `flat` pass fail to build, so every solid-color layer
    // renders nothing and the scene clear-color bleeds through
    // (docs/render-architecture.md §7.1, SPEC.md §V9). Eliminating the unused I/O
    // matches a GL driver whose linker drops it (`glGetAttribLocation` → -1).
    let active_defs: std::collections::HashMap<String, Option<i64>> = active
        .iter()
        .map(|(k, v)| (k.clone(), Some(i64::from(*v))))
        .collect();
    // Identifiers referenced in active, non-declaration lines — an attribute or
    // varying is "used" iff its name appears here.
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cond_stack: Vec<crate::modernize::Tri> = Vec::new();
    for line in expanded.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            crate::modernize::update_cond_stack(&mut cond_stack, trimmed, &active_defs);
            continue;
        }
        if cond_stack.contains(&crate::modernize::Tri::False) {
            continue;
        }
        if parse_attribute(line).is_some() || parse_varying(line).is_some() {
            continue;
        }
        for tok in line.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            if !tok.is_empty() {
                used.insert(tok.to_string());
            }
        }
    }
    // Names of unused declared I/O to strip from the assembled source.
    let mut unused_io: Vec<String> = Vec::new();
    // Varyings in declaration order, with their used flag. glsl locations are
    // auto-mapped *by declaration order*, and a varying's location must agree
    // across the two stages — so an unused varying can only be dropped when doing
    // so shifts nothing, i.e. every varying declared after it is also unused
    // (`flat.frag`'s lone `v_Color`). Dropping a non-trailing unused one (e.g.
    // `composelayer.frag`'s `v_TexCoord`, followed by the used `v_ScreenCoord`)
    // would renumber the survivor and break the VS/FS interface. An unused *input*
    // that stays is harmless (wgpu allows a fragment input the vertex feeds but
    // the fragment ignores).
    let mut varyings: Vec<(String, bool)> = Vec::new();
    cond_stack.clear();
    for line in expanded.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            crate::modernize::update_cond_stack(&mut cond_stack, trimmed, &active_defs);
            continue;
        }
        if cond_stack.contains(&crate::modernize::Tri::False) {
            continue;
        }
        // Vertex attributes are additionally registered in the reflection (in
        // declaration order), so the used/unused split must happen here too. They
        // carry no cross-stage interface, so any unused one is safely dropped and
        // the survivors renumber consistently with the frontend.
        if let Some(attr) = parse_attribute(line) {
            if stage == Stage::Vertex {
                if used.contains(&attr) {
                    attributes.push(VertexAttribute {
                        name: attr,
                        location: next_attr_loc,
                    });
                    next_attr_loc += 1;
                } else {
                    unused_io.push(attr);
                }
            }
        } else if let Some(v) = parse_varying(line) {
            let is_used = used.contains(&v);
            varyings.push((v, is_used));
        }
    }
    // Strip only the maximal trailing run of unused varyings (no location shift).
    for (name, is_used) in varyings.iter().rev() {
        if *is_used {
            break;
        }
        unused_io.push(name.clone());
    }

    // §3.5: gl_FragColor → out_FragColor (blind string replace, both stages).
    let body = expanded.replace("gl_FragColor", "out_FragColor");
    // Drop the declaration lines of the unused I/O identified above so the GLSL
    // frontend emits no dangling vertex/fragment interface variable for them.
    let body = if unused_io.is_empty() {
        body
    } else {
        body.lines()
            .filter(
                |line| match parse_attribute(line).or_else(|| parse_varying(line)) {
                    Some(name) => !unused_io.contains(&name),
                    None => true,
                },
            )
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Engine (`g_`-prefixed) uniforms are bound by name and always available
    // (docs/shader-pipeline.md §8.2). Some workshop shaders reference one the unit
    // never declares — e.g. scene 3118949804's `pulse.frag` uses `g_PulseThresholds`
    // etc. in its non-audio branch after stripping the declarations that stock
    // `effects/pulse/…/pulse.frag` carries. Synthesize a loose declaration for each
    // undeclared engine uniform (typed from its widest swizzle) so it packs into
    // the globals block and binds by name like any other. This can only ever add a
    // declaration the unit lacked, so a unit that already compiled is untouched.
    let synth = synth_missing_engine_uniforms(&body);
    let body = if synth.is_empty() {
        body
    } else {
        format!("{synth}{body}")
    };

    // §4: assemble prelude macros + per-stage defines + combo defines + body.
    // A unit may define its own function whose name collides with a function-like
    // prelude macro (e.g. tone_mapping.frag's `float log10(float x) { … }` vs the
    // prelude's `#define log10(x) …`): the macro would corrupt the definition into
    // `float (log2(float x) …) { … }`. The reference tolerates this because such
    // redefinitions sit behind `#if GLSL`; we reproduce the intent by dropping any
    // prelude function-like macro the body redefines as a function.
    let mut source_out = String::new();
    source_out.push_str(&filtered_prelude(&body));
    source_out.push_str(stage_defines(stage));
    for (name, value) in &active {
        source_out.push_str(&format!("#define {name} {value}\n"));
    }
    source_out.push('\n');
    source_out.push_str(&body);

    Ok(Assembled {
        source: source_out,
        reflection: Reflection {
            globals_block: Vec::new(),
            parameters,
            samplers,
            attributes,
            active_combos: active,
        },
    })
}

/// Synthesize `uniform TYPE g_X;` declarations for engine (`g_`-prefixed)
/// uniforms that `body` references but never declares (see the call site). The
/// type is inferred from the widest swizzle the code applies (`g_X.zw` ⇒ `vec3`
/// component index ⇒ at least `vec3`); an unswizzled use defaults to `float`.
/// Returns the joined declaration lines (empty when nothing is missing).
fn synth_missing_engine_uniforms(body: &str) -> String {
    use std::collections::{BTreeMap, BTreeSet};

    let mut declared: BTreeSet<String> = BTreeSet::new();
    let mut widths: BTreeMap<String, u8> = BTreeMap::new();

    for raw in body.lines() {
        // Ignore comment tails when scanning.
        let line = raw.split("//").next().unwrap_or("");
        let trimmed = line.trim_start();
        let is_decl = trimmed.starts_with("uniform ")
            || trimmed.starts_with("attribute ")
            || trimmed.starts_with("varying ");
        if is_decl {
            if let Some(name) = decl_name(line) {
                declared.insert(name);
            }
            // A declaration line's own name is not a "use"; skip scanning it.
            continue;
        }
        scan_engine_uses(line, &mut widths);
    }

    let mut out = String::new();
    for (name, width) in &widths {
        if declared.contains(name) {
            continue;
        }
        let ty = match width {
            2 => "vec2",
            3 => "vec3",
            4 => "vec4",
            _ => "float",
        };
        out.push_str(&format!("uniform {ty} {name};\n"));
    }
    out
}

/// Extract the declared identifier from a `uniform/attribute/varying TYPE name…;`
/// line: the last identifier before `;`/`[`/`=`.
fn decl_name(line: &str) -> Option<String> {
    let code = line.split("//").next().unwrap_or("").trim();
    let code = code.strip_suffix(';').unwrap_or(code);
    // Trim a trailing array suffix / initializer.
    let head = code.split(['[', '=']).next().unwrap_or(code).trim();
    let name = head.split_whitespace().next_back()?;
    name.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        .then(|| name.to_string())
}

/// Record `g_*` identifier uses in `line`, tracking the widest swizzle width seen
/// per name (a bare use registers width 1).
fn scan_engine_uses(line: &str, widths: &mut std::collections::BTreeMap<String, u8>) {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) && (i == 0 || !is_ident_byte(bytes[i - 1])) {
            let start = i;
            while i < bytes.len() && is_ident_byte(bytes[i]) {
                i += 1;
            }
            let name = &line[start..i];
            if name.starts_with("g_") {
                let mut width = 1u8;
                // Following `.swizzle` widens the inferred type.
                if bytes.get(i) == Some(&b'.') {
                    let sw_start = i + 1;
                    let mut j = sw_start;
                    while j < bytes.len() && is_ident_byte(bytes[j]) {
                        j += 1;
                    }
                    let sw = &line[sw_start..j];
                    if !sw.is_empty() && sw.bytes().all(|c| b"xyzwrgbastpq".contains(&c)) {
                        width = sw.bytes().map(swizzle_component).max().unwrap_or(0) + 1;
                    }
                }
                let e = widths.entry(name.to_string()).or_insert(1);
                *e = (*e).max(width);
            }
            continue;
        }
        i += 1;
    }
}

/// 0-based component index of a swizzle character (`x/r/s`→0 … `w/a/q`→3).
fn swizzle_component(c: u8) -> u8 {
    match c {
        b'x' | b'r' | b's' => 0,
        b'y' | b'g' | b't' => 1,
        b'z' | b'b' | b'p' => 2,
        _ => 3,
    }
}

/// True if `b` can start a GLSL identifier.
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// True if `b` can appear inside a GLSL identifier.
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// GLSL vector/matrix/scalar type keywords that can head a function definition's
/// return type — used to distinguish a user function definition `TYPE name(…)`
/// from an ordinary call `… = name(…)`.
const TYPE_KEYWORDS: &[&str] = &[
    "void", "float", "int", "uint", "bool", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2",
    "uvec3", "uvec4", "bvec2", "bvec3", "bvec4", "mat2", "mat3", "mat4",
];

/// Emit the HLSL-compat prelude with any function-like `#define NAME(…)` removed
/// when `body` defines a function of the same name (see the call site). Keeps the
/// prelude byte-identical for the overwhelmingly common no-collision case.
fn filtered_prelude(body: &str) -> String {
    let mut out = String::with_capacity(PRELUDE_MACROS.len());
    for line in PRELUDE_MACROS.lines() {
        if let Some(name) = function_macro_name(line)
            && body_defines_function(body, name)
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// If `line` is a function-like macro definition `#define NAME(…) …`, return
/// `NAME`; otherwise `None` (object-like macros have a space before any `(`).
fn function_macro_name(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#define ")?;
    let paren = rest.find('(')?;
    let name = &rest[..paren];
    // Function-like macros have no whitespace between name and `(`.
    (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')).then_some(name)
}

/// True if `body` contains a function *definition* whose name is `name`, i.e.
/// `<type-keyword> name (` — a call site (`x = name(…)`) has no type keyword
/// immediately before the name and so does not match.
fn body_defines_function(body: &str, name: &str) -> bool {
    let bytes = body.as_bytes();
    let mut search = 0;
    while let Some(rel) = body[search..].find(name) {
        let start = search + rel;
        let end = start + name.len();
        search = end;
        // Whole-token match followed by `(` (allowing spaces).
        let before_ok = start
            .checked_sub(1)
            .map(|b| !(bytes[b].is_ascii_alphanumeric() || bytes[b] == b'_'))
            .unwrap_or(true);
        if !before_ok || !body[end..].trim_start().starts_with('(') {
            continue;
        }
        // Preceding token must be a return-type keyword for this to be a def.
        let prefix = body[..start].trim_end();
        let prev_tok = prefix
            .rsplit(|c: char| c.is_whitespace() || c == ';' || c == '}')
            .next();
        if prev_tok.is_some_and(|t| TYPE_KEYWORDS.contains(&t)) {
            return true;
        }
    }
    false
}

/// Per-stage IO remap defines (docs/shader-pipeline.md §4.2,
/// `ShaderUnit.cpp:55-60`).
fn stage_defines(stage: Stage) -> &'static str {
    match stage {
        Stage::Fragment => "out vec4 out_FragColor;\n#define varying in\n",
        Stage::Vertex => "#define attribute in\n#define varying out\n",
    }
}

/// True if the text contains a `main` entry point (docs/shader-pipeline.md §3.1
/// match: `" main"` followed by ` ` or `(`).
fn contains_main(src: &str) -> bool {
    let bytes = src.as_bytes();
    let mut i = 0;
    while let Some(pos) = src[i..].find("main") {
        let at = i + pos;
        let before = at.checked_sub(1).map(|b| bytes[b]);
        let after = bytes.get(at + 4).copied();
        let word_before = before.is_none_or(|c| !c.is_ascii_alphanumeric() && c != b'_');
        let ok_after = matches!(
            after,
            Some(b'(') | Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') | Some(b'{')
        );
        if word_before && ok_after {
            return true;
        }
        i = at + 4;
    }
    false
}

/// Parse an `attribute <type> <name>;` declaration, returning the name.
fn parse_attribute(line: &str) -> Option<String> {
    let code = line.split("//").next().unwrap_or("").trim();
    let rest = code.strip_prefix("attribute ")?;
    let decl = rest.strip_suffix(';')?;
    let name = decl.split_whitespace().next_back()?;
    Some(name.split('[').next().unwrap_or(name).to_string())
}

/// Parse a legacy `varying <type> <name>;` declaration, returning the varying
/// name. Used to elide interface varyings an active branch declares but never
/// references (e.g. `flat.frag`'s `v_Color` with `VERTEXCOLOR` off), which would
/// otherwise leave a fragment input the vertex stage never writes.
fn parse_varying(line: &str) -> Option<String> {
    let code = line.split("//").next().unwrap_or("").trim();
    let rest = code.strip_prefix("varying ")?;
    let decl = rest.strip_suffix(';')?;
    let name = decl.split_whitespace().next_back()?;
    Some(name.split('[').next().unwrap_or(name).to_string())
}

/// Resolve final combo values: discovered defaults, overlaid by material then
/// override combos, then require-chain promotion to a fixed point
/// (docs/shader-pipeline.md §3.4, §4.3). Higher-precedence sources win.
fn resolve_combos(
    discovered: &BTreeMap<String, i32>,
    combo_requires: &BTreeMap<String, BTreeMap<String, i32>>,
    inputs: &ShaderInputs,
) -> BTreeMap<String, i32> {
    // Base: discovered defaults (lowest precedence, docs §4.3 item 4).
    let mut values: BTreeMap<String, i32> = discovered.clone();
    // Material/pass combos (docs §4.3 item 3).
    for (k, v) in &inputs.combos {
        values.insert(upper(k), *v);
    }
    // Override combos (docs §4.3 item 2).
    for (k, v) in &inputs.override_combos {
        values.insert(upper(k), *v);
    }
    // §3.4: fixed-point require-chain promotion (≤16 rounds). If a combo's
    // effective value is non-zero, force each required combo to its value.
    for _ in 0..16 {
        let mut changed = false;
        let snapshot = values.clone();
        for (name, value) in &snapshot {
            if *value == 0 {
                continue;
            }
            if let Some(reqs) = combo_requires.get(name) {
                for (req_name, req_val) in reqs {
                    if values.get(req_name) != Some(req_val) {
                        values.insert(req_name.clone(), *req_val);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IncludeResolver;
    use std::collections::BTreeMap;

    struct MapResolver(BTreeMap<String, String>);
    impl IncludeResolver for MapResolver {
        fn resolve(&self, name: &str) -> Option<String> {
            self.0.get(name).cloned()
        }
    }

    fn run(stage: Stage, src: &str, headers: &[(&str, &str)]) -> Assembled {
        let map = headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        preprocess(stage, "unit", src, &MapResolver(map), &ShaderInputs::default()).unwrap()
    }

    #[test]
    fn include_is_inlined_before_main() {
        // docs/shader-pipeline.md §3.1: include bodies are inlined.
        let a = run(
            Stage::Fragment,
            "#include \"common.h\"\nvoid main() { float x = HELPER; }",
            &[("common.h", "#define HELPER 3.0\n")],
        );
        assert!(a.source.contains("#define HELPER 3.0"));
        assert!(a.source.contains("begin of include from file common.h"));
    }

    #[test]
    fn missing_include_is_not_error() {
        // docs/shader-pipeline.md §3.1: a miss becomes a comment, never fatal.
        let a = run(Stage::Fragment, "#include \"nope.h\"\nvoid main() {}", &[]);
        assert!(a.source.contains("tried including file nope.h but was not found"));
    }

    #[test]
    fn unused_trailing_varying_and_attribute_are_elided() {
        // `flat` with VERTEXCOLOR off: `a_Color` (vertex) and `v_Color`
        // (fragment) are declared but only used under the inactive combo. Both
        // must be dropped so the 2D solid-layer pass builds instead of being
        // rejected on an unfeedable attribute / unmatched varying.
        let vs = run(
            Stage::Vertex,
            "attribute vec3 a_Position;\nattribute vec4 a_Color;\n#ifdef VERTEXCOLOR\nvarying vec4 v_Color;\n#endif\nvoid main() {\n gl_Position = vec4(a_Position, 1.0);\n#ifdef VERTEXCOLOR\n v_Color = a_Color;\n#endif\n}",
            &[],
        );
        let attrs: Vec<&str> = vs.reflection.attributes.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            attrs,
            vec!["a_Position"],
            "unused a_Color dropped from reflection"
        );
        // The declaration is removed (its only use sits in the inactive combo
        // block, which the frontend strips — so no live `a_Color` input remains).
        assert!(
            !vs.source.contains("attribute vec4 a_Color"),
            "unused a_Color declaration removed from source"
        );

        let fs = run(
            Stage::Fragment,
            "varying vec4 v_Color;\nvoid main() {\n gl_FragColor = vec4(1.0);\n#ifdef VERTEXCOLOR\n gl_FragColor *= v_Color;\n#endif\n}",
            &[],
        );
        assert!(
            !fs.source.contains("varying vec4 v_Color"),
            "unused trailing v_Color declaration removed"
        );
    }

    #[test]
    fn unused_non_trailing_varying_is_retained() {
        // `composelayer.frag`: `v_TexCoord` is unused but is declared *before*
        // the used `v_ScreenCoord`. Dropping it would renumber the survivor and
        // break the VS/FS interface, so it must stay (an unused fragment input
        // the vertex still feeds is harmless).
        let fs = run(
            Stage::Fragment,
            "varying vec2 v_TexCoord;\nvarying vec3 v_ScreenCoord;\nvoid main() {\n gl_FragColor = vec4(v_ScreenCoord, 1.0);\n}",
            &[],
        );
        assert!(
            fs.source.contains("v_TexCoord"),
            "non-trailing unused varying kept to preserve locations"
        );
        assert!(fs.source.contains("v_ScreenCoord"));
    }

    #[test]
    fn combo_discovered_default_emitted_uppercased() {
        // docs/shader-pipeline.md §2.1 + §4.3: discovered default, uppercased.
        let a = run(
            Stage::Fragment,
            "// [COMBO] {\"combo\":\"lighting\",\"default\":2}\nvoid main() {}",
            &[],
        );
        assert!(a.source.contains("#define LIGHTING 2"));
        assert_eq!(a.reflection.active_combos.get("LIGHTING"), Some(&2));
    }

    #[test]
    fn require_chain_promotes_dependency() {
        // docs/shader-pipeline.md §3.4: RIMLIGHTING=1 forces LIGHTING=1.
        let src = "// [COMBO] {\"combo\":\"LIGHTING\",\"default\":0}\n\
                   // [COMBO] {\"combo\":\"RIMLIGHTING\",\"default\":1,\"require\":{\"LIGHTING\":1}}\n\
                   void main() {}";
        let a = run(Stage::Fragment, src, &[]);
        assert_eq!(a.reflection.active_combos.get("LIGHTING"), Some(&1));
    }

    #[test]
    fn gl_fragcolor_rewritten() {
        // docs/shader-pipeline.md §3.5.
        let a = run(Stage::Fragment, "void main() { gl_FragColor = vec4(1.0); }", &[]);
        assert!(a.source.contains("out_FragColor = vec4(1.0)"));
        assert!(!a.source.contains("gl_FragColor"));
        assert!(a.source.contains("out vec4 out_FragColor;"));
    }

    #[test]
    fn stage_defines_differ() {
        // docs/shader-pipeline.md §4.2.
        let f = run(Stage::Fragment, "void main() {}", &[]);
        assert!(f.source.contains("#define varying in"));
        let v = run(Stage::Vertex, "void main() {}", &[]);
        assert!(v.source.contains("#define attribute in"));
        assert!(v.source.contains("#define varying out"));
    }

    #[test]
    fn require_module_lightingv1_stub() {
        // docs/shader-pipeline.md §3.2.
        let a = run(Stage::Fragment, "#require LightingV1\nvoid main() {}", &[]);
        assert!(a.source.contains("PerformLighting_V1"));
    }

    #[test]
    fn no_main_is_error() {
        // docs/shader-pipeline.md §3.1.
        let map = BTreeMap::new();
        let err = preprocess(
            Stage::Fragment,
            "u",
            "float x = 1.0;",
            &MapResolver(map),
            &ShaderInputs::default(),
        );
        assert!(matches!(err, Err(TranslateError::NoMain { .. })));
    }

    #[test]
    fn sampler_slot_and_param_reflected() {
        let src = "uniform sampler2D g_Texture0; // {\"default\":\"util/white\"}\n\
                   uniform float g_Brightness; // {\"material\":\"Brightness\",\"default\":1}\n\
                   void main() {}";
        let a = run(Stage::Fragment, src, &[]);
        assert_eq!(a.reflection.samplers.len(), 1);
        assert_eq!(a.reflection.samplers[0].slot, Some(0));
        assert_eq!(a.reflection.parameters.len(), 1);
        assert_eq!(a.reflection.parameters[0].material, "Brightness");
    }

    #[test]
    fn material_combo_overrides_discovered_default() {
        // docs/shader-pipeline.md §4.3 precedence: material > discovered.
        let mut inputs = ShaderInputs::default();
        inputs.combos.insert("lighting".into(), 0);
        let map = BTreeMap::new();
        let a = preprocess(
            Stage::Fragment,
            "u",
            "// [COMBO] {\"combo\":\"LIGHTING\",\"default\":1}\nvoid main() {}",
            &MapResolver(map),
            &inputs,
        )
        .unwrap();
        assert_eq!(a.reflection.active_combos.get("LIGHTING"), Some(&0));
    }
}
