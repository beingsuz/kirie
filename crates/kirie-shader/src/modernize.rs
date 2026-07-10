//! GLSL modernization: turn the assembled legacy WE source (docs/shader-pipeline.md
//! §4 output) into GLSL a modern Vulkan-flavored frontend accepts, performing the
//! structural changes docs/shader-pipeline.md §9.2 requires for a wgpu target:
//!
//! 1. Prepend `#version 450` (the reference's `#version 330` is a GL-era choice;
//!    naga's GLSL frontend and Vulkan glslang require 4.5+, docs §9.2).
//! 2. Rename GLSL-4.x reserved keywords used as identifiers by GL-3.3-era shaders
//!    (e.g. `sample`) — otherwise the Vulkan frontends reject them.
//! 3. Aggregate loose default-block uniforms into one uniform block (`_WEGlobals`)
//!    — loose uniforms are invalid for Vulkan SPIR-V / wgpu (docs §9.2.2).
//! 4. Split every combined `uniform sampler2D g_TexN` into a separate
//!    `texture2D` + `sampler` binding pair, re-paired at every use site by a
//!    `#define g_TexN sampler2D(img, smp)` macro — combined samplers are
//!    unsupported by both naga frontends and by wgpu's binding model (docs §9.2.4).
//!
//! Array-varying flattening (docs §9.2.3, the blur/gaussian family) is done later
//! in [`crate::translate`] after macro/`#if` expansion, when only the active
//! array declaration remains.

use crate::Stage;
use crate::preprocess::Assembled;
use crate::reflect::{Reflection, SamplerSlot};

/// The wgpu-required GLSL version (docs/shader-pipeline.md §9.2). glslang's
/// OpenGL client caps at 450; naga accepts 440/450/460.
const VERSION_LINE: &str = "#version 450\n";

/// The synthesized uniform-block name holding aggregated loose uniforms
/// (docs/shader-pipeline.md §9.2.2). An **unnamed** block exposes its members by
/// name into global scope, so no reference rewriting is needed.
const GLOBALS_BLOCK: &str = "_WEGlobals";

/// GLSL keywords reserved in 4.x that GL-3.3-era WE shaders use as ordinary
/// identifiers (docs/shader-pipeline.md §2: the dialect predates these). Renamed
/// with a `_we` suffix by whole-token substitution. `sample` is by far the most
/// common in the corpus; the rest are included defensively from the GLSL 4.x
/// reserved list.
const RESERVED: &[&str] = &[
    "sample",
    "filter",
    "input",
    "output",
    "active",
    "partition",
    "common",
    "superp",
    "resource",
    "patch",
];

/// Modernize the assembled source and finalize sampler bindings + the globals
/// block in the reflection. Returns the modernized GLSL (still containing
/// `#define`s / `#if`s for the frontend's preprocessor) and the updated
/// reflection.
pub fn modernize(_stage: Stage, assembled: Assembled) -> (String, Reflection) {
    let Assembled {
        source,
        mut reflection,
    } = assembled;

    let renamed = rename_reserved(&source);

    // Object-like `#define`s in the assembled source (combo values emitted by
    // preprocess, e.g. `#define LIGHTING 0`, plus any in-source defines). Used to
    // evaluate simple `#if`/`#ifdef` guards below.
    let defines = collect_defines(&renamed);

    // Split declarations from body: pull loose uniforms into a block and split
    // combined samplers. Everything else is preserved in order.
    let mut block_members: Vec<String> = Vec::new();
    let mut sampler_decls = String::new();
    let mut body = String::new();
    // Binding 0 is reserved for the globals block; samplers start at 1.
    let mut next_binding = 1u32;
    // Preprocessor-conditional nesting: the tri-state of each open branch. A
    // loose uniform / combined sampler inside a branch that is *known inactive*
    // (e.g. `uniform mat4x3 g_Bones[BONECOUNT];` inside `#if SKINNING` with
    // SKINNING undefined) must NOT be hoisted into the unconditional globals
    // block — that would leak an undefined array-size macro and fail translation.
    // The reference resolves `#if` before this modernization; we keep the
    // declaration inline (guarded) so the frontend's preprocessor drops it. We
    // only suppress on *definitely* inactive branches; active/unknown branches
    // hoist as before, so no shader that already compiled regresses (§9.2.2).
    let mut cond_stack: Vec<Tri> = Vec::new();

    for line in renamed.lines() {
        // Track `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif` nesting. The
        // directive line itself passes through to the body unchanged.
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            update_cond_stack(&mut cond_stack, trimmed, &defines);
            body.push_str(line);
            body.push('\n');
            continue;
        }
        let inactive = cond_stack.contains(&Tri::False);

        if !inactive && let Some(decl) = parse_uniform_decl(line) {
            match decl {
                UniformDecl::Sampler(name) => {
                    let tex_b = next_binding;
                    let smp_b = next_binding + 1;
                    next_binding += 2;
                    sampler_decls.push_str(&format!(
                        "layout(set = 0, binding = {tex_b}) uniform texture2D {name}_img;\n\
                         layout(set = 0, binding = {smp_b}) uniform sampler {name}_smp;\n\
                         #define {name} sampler2D({name}_img, {name}_smp)\n"
                    ));
                    // Record the split bindings in the reflection slot. An
                    // *unannotated* `uniform sampler2D g_TextureN;` (no `// {json}`
                    // trailer, so preprocess never registered it) still needs a
                    // reflection entry: the reference binds `g_Texture0..7` by name
                    // regardless of annotation (docs/shader-pipeline.md §8.2), and
                    // the renderer wires textures from this table by name/slot.
                    // Without it the split bindings exist in the module but the
                    // texture is never bound → the pass samples an unbound texture
                    // (silent black/garbage). Synthesize a slot from the name digit.
                    if let Some(slot) = reflection.samplers.iter_mut().find(|s| s.name == name) {
                        slot.texture_binding = tex_b;
                        slot.sampler_binding = smp_b;
                    } else {
                        reflection.samplers.push(SamplerSlot {
                            slot: sampler_slot(&name),
                            name,
                            texture_binding: tex_b,
                            sampler_binding: smp_b,
                            default_texture: None,
                            combo: None,
                        });
                    }
                    continue;
                }
                UniformDecl::Loose(member) => {
                    block_members.push(member);
                    continue;
                }
                UniformDecl::Other => { /* fall through, keep the line */ }
            }
        }
        body.push_str(line);
        body.push('\n');
    }

    // Assemble: version, globals block, split samplers, then the body.
    let mut out = String::with_capacity(renamed.len() + 256);
    out.push_str(VERSION_LINE);
    if !block_members.is_empty() {
        out.push_str(&format!(
            "layout(std140, set = 0, binding = 0) uniform {GLOBALS_BLOCK} {{\n"
        ));
        for m in &block_members {
            out.push_str("    ");
            out.push_str(m);
            out.push_str(";\n");
        }
        out.push_str("};\n");
    }
    out.push_str(&sampler_decls);
    out.push_str(&body);

    reflection.globals_block = block_members.iter().map(|m| member_name(m).to_string()).collect();

    (out, reflection)
}

/// Tri-state truth of a preprocessor branch: known-true, known-false, or
/// unknown (a condition we do not evaluate, e.g. `#if (A || B) && C == 0`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tri {
    True,
    False,
    Unknown,
}

/// Collect object-like `#define NAME VALUE` macros from the assembled source,
/// mapping each to its integer value when it has one (function-like macros and
/// non-numeric bodies map to `None` = "defined but not a known integer"). Used
/// only to evaluate simple `#if`/`#ifdef` guards for uniform-hoisting.
fn collect_defines(src: &str) -> std::collections::HashMap<String, Option<i64>> {
    let mut out = std::collections::HashMap::new();
    for line in src.lines() {
        let Some(rest) = line.trim_start().strip_prefix("#define ") else {
            continue;
        };
        let rest = rest.trim_start();
        // First whitespace-delimited token is the macro name; a `(` in it (no
        // space before the paren) marks a function-like macro — skip those.
        let name_end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
        let name = &rest[..name_end];
        if name.is_empty() || name.contains('(') {
            continue;
        }
        let body = rest[name_end..].split("//").next().unwrap_or("").trim();
        out.insert(name.to_string(), body.parse::<i64>().ok());
    }
    out
}

/// Evaluate a `#if`/`#ifdef`/`#ifndef` directive to a [`Tri`], recognizing the
/// simple forms that gate uniform declarations in the corpus: a bare macro
/// (`#if SKINNING`), an integer literal, `defined(X)`, and `#ifdef`/`#ifndef`.
/// Anything else is [`Tri::Unknown`], leaving the current hoisting behavior.
fn eval_directive(directive: &str, defines: &std::collections::HashMap<String, Option<i64>>) -> Tri {
    let d = directive.split("//").next().unwrap_or("").trim();
    let defined = |name: &str| defines.contains_key(name);
    if let Some(name) = d.strip_prefix("#ifdef ") {
        return tri(defined(name.trim()));
    }
    if let Some(name) = d.strip_prefix("#ifndef ") {
        return tri(!defined(name.trim()));
    }
    let Some(expr) = d.strip_prefix("#if ").or_else(|| d.strip_prefix("#elif ")) else {
        return Tri::Unknown;
    };
    let expr = expr.trim();
    if let Ok(n) = expr.parse::<i64>() {
        return tri(n != 0);
    }
    if let Some(inner) = expr.strip_prefix("defined(").and_then(|s| s.strip_suffix(')')) {
        return tri(defined(inner.trim()));
    }
    // A single bare identifier: undefined ⇒ 0 (false); defined ⇒ its int value,
    // or unknown when the body is not a plain integer.
    if !expr.is_empty() && expr.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return match defines.get(expr) {
            None => Tri::False,
            Some(Some(v)) => tri(*v != 0),
            Some(None) => Tri::Unknown,
        };
    }
    Tri::Unknown
}

fn tri(b: bool) -> Tri {
    if b { Tri::True } else { Tri::False }
}

/// Apply one preprocessor directive line to the conditional-nesting stack.
pub(crate) fn update_cond_stack(
    stack: &mut Vec<Tri>,
    directive: &str,
    defines: &std::collections::HashMap<String, Option<i64>>,
) {
    let d = directive.trim_start();
    if d.starts_with("#ifdef") || d.starts_with("#ifndef") || d.starts_with("#if ") || d == "#if" {
        stack.push(eval_directive(d, defines));
    } else if d.starts_with("#elif") {
        // Conservative: once any branch could be taken we cannot cheaply prove
        // this one dead, so treat an `#elif` region as unknown (never suppress).
        if let Some(top) = stack.last_mut() {
            *top = Tri::Unknown;
        }
    } else if d.starts_with("#else") {
        if let Some(top) = stack.last_mut() {
            *top = match *top {
                Tri::True => Tri::False,
                Tri::False => Tri::True,
                Tri::Unknown => Tri::Unknown,
            };
        }
    } else if d.starts_with("#endif") {
        stack.pop();
    }
}

/// Classification of a uniform declaration line.
enum UniformDecl {
    /// `uniform sampler2D NAME;` — carries the sampler name.
    Sampler(String),
    /// A loose non-opaque uniform — carries the block-member text (type + name,
    /// e.g. `float g_Time;` or `float g_AudioSpectrum16Left[16];`).
    Loose(String),
    /// A uniform we do not transform (e.g. a uniform block, or a sampler type we
    /// don't split); left untouched.
    Other,
}

/// Parse a `uniform …;` declaration line, ignoring any trailing `//` comment.
/// Returns `None` if the line is not a plain uniform declaration.
fn parse_uniform_decl(line: &str) -> Option<UniformDecl> {
    let code = line.split("//").next().unwrap_or("").trim();
    let rest = code.strip_prefix("uniform ")?;
    let inner = rest.strip_suffix(';')?;
    // Skip block openers (`uniform Foo {`) and initialized decls.
    if inner.contains('{') || inner.contains('=') {
        return Some(UniformDecl::Other);
    }
    let tokens: Vec<&str> = inner.split_whitespace().collect();
    if tokens.len() < 2 {
        return Some(UniformDecl::Other);
    }
    let ty = tokens[0];
    if ty == "sampler2D" {
        let name = tokens[tokens.len() - 1];
        // Only split simple (non-array) sampler declarations.
        if name.contains('[') {
            return Some(UniformDecl::Other);
        }
        return Some(UniformDecl::Sampler(name.to_string()));
    }
    if ty.starts_with("sampler") || ty.starts_with("texture") || ty.starts_with("image") {
        // Other opaque types (e.g. sampler2DComparison): leave for the frontend
        // / fallback to handle (docs/shader-pipeline.md §2.2 lists these as rare).
        return Some(UniformDecl::Other);
    }
    // Non-opaque loose uniform → block member (docs/shader-pipeline.md §9.2.2).
    Some(UniformDecl::Loose(inner.trim().to_string()))
}

/// Slot index for a `g_Texture<N>` sampler: the single digit after `g_Texture`
/// (docs/shader-pipeline.md §2.2, `ShaderUnit.cpp:595-600`); slots 0-9 only.
/// Mirrors `preprocess::sampler_slot` so unannotated samplers reflect the same
/// slot the annotation path would have assigned.
fn sampler_slot(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("g_Texture")?;
    let c = rest.as_bytes().first().copied()?;
    c.is_ascii_digit().then(|| u32::from(c - b'0'))
}

/// Extract the member name from a block-member declaration like
/// `float g_AudioSpectrum16Left[16]` → `g_AudioSpectrum16Left`.
fn member_name(member: &str) -> &str {
    let name = member.split_whitespace().next_back().unwrap_or(member);
    name.split('[').next().unwrap_or(name)
}

/// Rename GLSL-4.x reserved identifiers via whole-token substitution
/// (docs/shader-pipeline.md §9.2). Only identifier tokens are considered, so
/// substrings inside longer identifiers (`sampleCount`) are untouched. Comments
/// and preprocessor lines pass through the same tokenizer harmlessly.
fn rename_reserved(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 64);
    let mut word = String::new();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    for ch in src.chars() {
        if is_word(ch) {
            word.push(ch);
        } else {
            flush_word(&mut word, &mut out);
            out.push(ch);
        }
    }
    flush_word(&mut word, &mut out);
    out
}

/// Emit an accumulated identifier, applying the reserved-word rename.
fn flush_word(word: &mut String, out: &mut String) {
    if word.is_empty() {
        return;
    }
    if RESERVED.contains(&word.as_str()) {
        out.push_str(word);
        out.push_str("_we");
    } else {
        out.push_str(word);
    }
    word.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::Assembled;
    use crate::reflect::{Reflection, SamplerSlot};

    fn asm(source: &str, samplers: Vec<SamplerSlot>) -> Assembled {
        Assembled {
            source: source.to_string(),
            reflection: Reflection {
                samplers,
                ..Reflection::default()
            },
        }
    }

    fn slot(name: &str) -> SamplerSlot {
        SamplerSlot {
            name: name.to_string(),
            slot: Some(0),
            texture_binding: 0,
            sampler_binding: 0,
            default_texture: None,
            combo: None,
        }
    }

    #[test]
    fn version_line_prepended() {
        let (src, _) = modernize(Stage::Fragment, asm("void main() {}", vec![]));
        assert!(src.starts_with("#version 450"));
    }

    #[test]
    fn loose_uniforms_packed_with_semicolons() {
        // docs/shader-pipeline.md §9.2.2: block members must be well-formed.
        let (src, refl) = modernize(
            Stage::Fragment,
            asm(
                "uniform float g_Time;\nuniform vec2 g_TexelSize;\nvoid main() {}",
                vec![],
            ),
        );
        assert!(src.contains("uniform _WEGlobals {"));
        assert!(src.contains("    float g_Time;"));
        assert!(src.contains("    vec2 g_TexelSize;"));
        assert_eq!(refl.globals_block, vec!["g_Time", "g_TexelSize"]);
        // No loose uniform survives outside the block.
        assert!(!src.contains("\nuniform float g_Time;"));
    }

    #[test]
    fn array_uniform_packed_member_name() {
        let (_, refl) = modernize(
            Stage::Fragment,
            asm("uniform float g_AudioSpectrum16Left[16];\nvoid main() {}", vec![]),
        );
        assert_eq!(refl.globals_block, vec!["g_AudioSpectrum16Left"]);
    }

    #[test]
    fn combined_sampler_split_with_pairing_macro() {
        // docs/shader-pipeline.md §9.2.4.
        let (src, refl) = modernize(
            Stage::Fragment,
            asm(
                "uniform sampler2D g_Texture0;\nvoid main() {}",
                vec![slot("g_Texture0")],
            ),
        );
        assert!(src.contains("uniform texture2D g_Texture0_img;"));
        assert!(src.contains("uniform sampler g_Texture0_smp;"));
        assert!(src.contains("#define g_Texture0 sampler2D(g_Texture0_img, g_Texture0_smp)"));
        assert_eq!(refl.samplers[0].texture_binding, 1);
        assert_eq!(refl.samplers[0].sampler_binding, 2);
    }

    #[test]
    fn unannotated_sampler_is_reflected() {
        // Regression: a `uniform sampler2D g_TextureN;` with no `// {json}`
        // annotation (corpus 2395163768 tint.frag) is never registered by
        // preprocess, but the reference binds g_Texture0..7 by name regardless
        // (docs/shader-pipeline.md §8.2). modernize must still record the split
        // bindings + slot so the renderer can wire the texture.
        let (src, refl) = modernize(
            Stage::Fragment,
            asm(
                "uniform sampler2D g_Texture0;\nuniform sampler2D g_Texture1;\nvoid main() {}",
                // No annotated slots — reflection starts empty, as from preprocess.
                vec![],
            ),
        );
        assert!(src.contains("uniform texture2D g_Texture0_img;"));
        assert!(src.contains("uniform texture2D g_Texture1_img;"));
        assert_eq!(refl.samplers.len(), 2);
        assert_eq!(refl.samplers[0].name, "g_Texture0");
        assert_eq!(refl.samplers[0].slot, Some(0));
        assert_eq!(refl.samplers[0].texture_binding, 1);
        assert_eq!(refl.samplers[0].sampler_binding, 2);
        assert_eq!(refl.samplers[1].name, "g_Texture1");
        assert_eq!(refl.samplers[1].slot, Some(1));
        assert_eq!(refl.samplers[1].texture_binding, 3);
        assert_eq!(refl.samplers[1].sampler_binding, 4);
    }

    #[test]
    fn reserved_word_renamed() {
        // docs/shader-pipeline.md §9.2: `sample` reserved in GLSL 4.x.
        let (src, _) = modernize(
            Stage::Fragment,
            asm("void main() { float sample = 1.0; }", vec![]),
        );
        assert!(src.contains("float sample_we = 1.0;"));
        // A longer identifier containing the reserved word is untouched.
        let (src2, _) = modernize(
            Stage::Fragment,
            asm("void main() { int sampleCount = 0; }", vec![]),
        );
        assert!(src2.contains("sampleCount"));
        assert!(!src2.contains("sampleCount_we"));
    }
}
