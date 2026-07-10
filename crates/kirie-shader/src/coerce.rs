//! Shape/type coercion — a text-level reimplementation of the patched-glslang
//! leniencies (docs/shader-pipeline.md §7.1, §7.2) that stock glslang and naga
//! both reject. Runs on the flat, macro/`#if`-expanded GLSL just before the
//! frontends, so all declarations and the single active `#if` branch are visible.
//!
//! Reproduced, conservatively (only when both sides' types are confidently
//! inferred and actually mismatch — a shader that already compiles has no such
//! mismatch, so passing shaders are untouched):
//!
//! * **§7.1 scalar→vector splat / wider→narrower truncation on assignment**:
//!   `vec2 a = vec4Expr;` ⇒ `vec2 a = (vec4Expr).xy;`, `vec3 t = 0.5;` ⇒
//!   `vec3(0.5)`, and scalar `float`↔`int` via a wrapping cast.
//! * **§7.2 lax overload argument shapes**: a call argument whose vector width
//!   differs from the parameter is truncated/splatted to fit — for `texture`
//!   (2-D coordinate forced to `vec2`), the `mix` built-in (operands unified to
//!   the narrower width), and any user function whose signature is visible in the
//!   flat source (e.g. `rotateVec2(vec2, float)`).
//!
//! Anything the inferencer cannot type is left verbatim (SPEC.md §V9: no panic,
//! no guessing); the frontend then reports it as before.

use std::collections::HashMap;

/// Scalar base class of a GLSL numeric type. `mat*`/opaque types are not tracked
/// (they never participate in the coercions above) and yield `None` on parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Base {
    Float,
    Int,
    Uint,
    Bool,
}

/// A tracked GLSL numeric type: a scalar (`size == 1`) or a vector (`2..=4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ty {
    base: Base,
    size: u8,
}

impl Ty {
    /// The GLSL spelling used when synthesizing a splat constructor.
    fn ctor(self) -> &'static str {
        match (self.base, self.size) {
            (Base::Float, 1) => "float",
            (Base::Float, 2) => "vec2",
            (Base::Float, 3) => "vec3",
            (Base::Float, 4) => "vec4",
            (Base::Int, 1) => "int",
            (Base::Int, 2) => "ivec2",
            (Base::Int, 3) => "ivec3",
            (Base::Int, 4) => "ivec4",
            (Base::Uint, 1) => "uint",
            (Base::Uint, 2) => "uvec2",
            (Base::Uint, 3) => "uvec3",
            (Base::Uint, 4) => "uvec4",
            (Base::Bool, 1) => "bool",
            (Base::Bool, 2) => "bvec2",
            (Base::Bool, 3) => "bvec3",
            _ => "vec4",
        }
    }
}

/// Parse a GLSL type keyword into a tracked [`Ty`] (`None` for `mat*`, samplers,
/// and anything not a scalar/vector numeric type).
fn parse_ty(tok: &str) -> Option<Ty> {
    let (base, size) = match tok {
        "float" => (Base::Float, 1),
        "vec2" => (Base::Float, 2),
        "vec3" => (Base::Float, 3),
        "vec4" => (Base::Float, 4),
        "int" => (Base::Int, 1),
        "ivec2" => (Base::Int, 2),
        "ivec3" => (Base::Int, 3),
        "ivec4" => (Base::Int, 4),
        "uint" => (Base::Uint, 1),
        "uvec2" => (Base::Uint, 2),
        "uvec3" => (Base::Uint, 3),
        "uvec4" => (Base::Uint, 4),
        "bool" => (Base::Bool, 1),
        "bvec2" => (Base::Bool, 2),
        "bvec3" => (Base::Bool, 3),
        "bvec4" => (Base::Bool, 4),
        _ => return None,
    };
    Some(Ty { base, size })
}

/// Type tables gathered from the flat source: variable names → type, function
/// names → (parameter types, return type).
struct TypeEnv {
    vars: HashMap<String, Ty>,
    funcs: HashMap<String, (Vec<Option<Ty>>, Option<Ty>)>,
}

/// The coercion entry point. Builds the type environment, then rewrites the
/// tractable mismatches. Idempotent-ish: a second pass finds nothing to change.
pub fn coerce_shapes(src: &str) -> String {
    let env = build_env(src);
    // Line-oriented: every mismatch in the corpus is a single-line statement.
    let mut out = String::with_capacity(src.len() + 64);
    for line in src.lines() {
        let rewritten = coerce_line(line, &env);
        out.push_str(&rewritten);
        out.push('\n');
    }
    out
}

/// True if `b` can appear inside a GLSL identifier.
fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Collect variable and function-signature types from top-level declarations,
/// uniform-block members, and local `TYPE name` declarations.
fn build_env(src: &str) -> TypeEnv {
    let mut vars: HashMap<String, Ty> = HashMap::new();
    let mut funcs: HashMap<String, (Vec<Option<Ty>>, Option<Ty>)> = HashMap::new();

    for raw in src.lines() {
        let line = raw.trim();
        // Function definition/prototype: `RET name(params)` with a body/`;`.
        if let Some((name, sig, ret)) = parse_function_sig(line) {
            funcs.entry(name).or_insert((sig, ret));
            // Register the parameters as locals too (best effort).
            continue;
        }
        // A declaration line, possibly prefixed by `in`/`out`/`uniform`/
        // qualifiers, possibly a block member `    TYPE name;`.
        register_var_decl(line, &mut vars);
    }
    TypeEnv { vars, funcs }
}

/// Register a `TYPE name` declaration (optionally IO/uniform-qualified, optionally
/// with an `= initializer`) into `vars`. Only the simple single-name form is
/// tracked; multi-name lists, arrays, and non-numeric types are ignored.
fn register_var_decl(line: &str, vars: &mut HashMap<String, Ty>) {
    let mut s = line.trim();
    for q in [
        "in ", "out ", "uniform ", "flat ", "smooth ", "const ", "highp ", "lowp ", "mediump ",
    ] {
        while let Some(rest) = s.trim_start().strip_prefix(q) {
            s = rest.trim_start();
        }
    }
    let s = s.trim().trim_end_matches(';').trim();
    // Cut off any initializer: the declared name sits before the first `=`.
    let head = s.split('=').next().unwrap_or(s).trim();
    // A plain `TYPE name` head has no call/index/block syntax.
    if head.contains('(') || head.contains('{') || head.contains('[') {
        return;
    }
    let mut it = head.split_whitespace();
    let (Some(ty_tok), Some(name)) = (it.next(), it.next()) else {
        return;
    };
    if it.next().is_some() {
        return; // more than `TYPE name` (e.g. struct member lists) — skip.
    }
    if let Some(ty) = parse_ty(ty_tok)
        && name.bytes().all(is_ident)
    {
        vars.insert(name.to_string(), ty);
    }
}

/// Parse a function definition/prototype header `RET name(T a, T b, …)`.
fn parse_function_sig(line: &str) -> Option<(String, Vec<Option<Ty>>, Option<Ty>)> {
    let open = line.find('(')?;
    let head = line[..open].trim();
    // head = `RET name`
    let mut ht = head.split_whitespace();
    let ret_tok = ht.next()?;
    let name = ht.next()?;
    if ht.next().is_some() || !name.bytes().all(is_ident) {
        return None;
    }
    let ret = parse_ty(ret_tok);
    // Require this to look like a signature: `)` then `{` or `;` on the line.
    let close = line[open..].find(')')? + open;
    let tail = line[close + 1..].trim_start();
    if !(tail.starts_with('{') || tail.starts_with(';') || tail.is_empty()) {
        return None;
    }
    let params_str = &line[open + 1..close];
    let mut params = Vec::new();
    if params_str.trim() != "void" && !params_str.trim().is_empty() {
        for p in params_str.split(',') {
            // Skip leading parameter qualifiers (`const`, `in`, `highp`, …).
            let ptok = p.split_whitespace().find(|t| {
                !matches!(
                    *t,
                    "const" | "in" | "out" | "inout" | "highp" | "lowp" | "mediump"
                )
            });
            params.push(ptok.and_then(parse_ty));
        }
    }
    Some((name.to_string(), params, ret))
}

/// Infer the type of a *simple* expression: numeric-free atoms only — an
/// identifier, an identifier with a `.swizzle`, a leading unary `+`/`-` of those,
/// a whole `texture(...)` call (vec4), or a call to a known-return user function.
fn infer(expr: &str, env: &TypeEnv) -> Option<Ty> {
    let e = expr.trim();
    let e = e.strip_prefix(['-', '+']).unwrap_or(e).trim();
    // Whole-expression function call `name( ... )` with matched outer parens.
    if let Some(open) = e.find('(')
        && e.ends_with(')')
        && paren_balanced_span(e, open)
    {
        let callee = e[..open].trim();
        if callee == "texture" || callee == "textureLod" {
            return Some(Ty {
                base: Base::Float,
                size: 4,
            });
        }
        if let Some((_, ret)) = env.funcs.get(callee) {
            return *ret;
        }
        return None;
    }
    // `ident` or `ident.swizzle`.
    let (base_ident, swizzle) = match e.split_once('.') {
        Some((a, b)) => (a.trim(), Some(b.trim())),
        None => (e, None),
    };
    if !base_ident.bytes().all(is_ident) || base_ident.is_empty() {
        return None;
    }
    let ty = *env.vars.get(base_ident)?;
    match swizzle {
        None => Some(ty),
        Some(sw) => {
            if !sw.bytes().all(|c| b"xyzwrgbastpq".contains(&c)) || sw.is_empty() || sw.len() > 4 {
                return None;
            }
            Some(Ty {
                base: ty.base,
                size: sw.len() as u8,
            })
        }
    }
}

/// True if the `(` at `open` closes exactly at the end of `e` (one outer group).
fn paren_balanced_span(e: &str, open: usize) -> bool {
    let bytes = e.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i == bytes.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

/// Coerce a single source line (assignment truncation/splat + call-argument
/// shape fixes). Preserves leading indentation.
fn coerce_line(line: &str, env: &TypeEnv) -> String {
    // Skip preprocessor and comment lines untouched.
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') || trimmed.starts_with("//") {
        return line.to_string();
    }
    let indent_len = line.len() - trimmed.len();
    let (indent, code) = line.split_at(indent_len);
    let with_calls = coerce_calls(code, env);
    let with_assign = coerce_for_init(&coerce_assignment(&with_calls, env), env);
    format!("{indent}{with_assign}")
}

/// Coerce the initializer clause of a `for (INIT; COND; STEP)` header — the one
/// assignment site the line-oriented [`coerce_assignment`] cannot reach because a
/// `for` line carries several `;`. Handles the `int i = -floatExpr` case
/// (docs/shader-pipeline.md §7.1 float→int) seen in `blur_gaussian.frag`.
fn coerce_for_init(code: &str, env: &TypeEnv) -> String {
    let Some(fpos) = code.find("for") else {
        return code.to_string();
    };
    // Whole-token `for` immediately (past spaces) followed by `(`.
    let before_ok = fpos
        .checked_sub(1)
        .map(|b| !is_ident(code.as_bytes()[b]))
        .unwrap_or(true);
    let after = code[fpos + 3..].trim_start();
    if !before_ok || !after.starts_with('(') {
        return code.to_string();
    }
    let open = fpos + 3 + (code[fpos + 3..].find('(').unwrap());
    let Some(semi_rel) = code[open + 1..].find(';') else {
        return code.to_string();
    };
    let init_start = open + 1;
    let init_end = init_start + semi_rel;
    let init = &code[init_start..init_end];
    let Some(eq) = find_plain_assign(init) else {
        return code.to_string();
    };
    let lhs = init[..eq].trim();
    let rhs = init[eq + 1..].trim();
    let (Some(lt), Some(rt)) = (lhs_type(lhs, env), infer(rhs, env)) else {
        return code.to_string();
    };
    if lt == rt {
        return code.to_string();
    }
    let Some(new_rhs) = coerce_expr_to(rhs, rt, lt) else {
        return code.to_string();
    };
    format!("{}{lhs} = {new_rhs}{}", &code[..init_start], &code[init_end..])
}

/// Coerce a simple assignment/declaration `LHS = RHS;` when the two sides' widths
/// or scalar bases mismatch. Only plain `=` (not `==`/`+=`/… ) is handled.
fn coerce_assignment(code: &str, env: &TypeEnv) -> String {
    let Some(semi) = code.rfind(';') else {
        return code.to_string();
    };
    let stmt = &code[..semi];
    let after = &code[semi..];
    let Some(eq) = find_plain_assign(stmt) else {
        return code.to_string();
    };
    let lhs = stmt[..eq].trim();
    let rhs = stmt[eq + 1..].trim();
    if rhs.is_empty() {
        return code.to_string();
    }

    // LHS type: either a declaration `TYPE name` or an existing lvalue.
    let lhs_ty = lhs_type(lhs, env);
    let Some(lt) = lhs_ty else {
        return code.to_string();
    };
    let Some(rt) = infer(rhs, env) else {
        return code.to_string();
    };
    if lt == rt {
        return code.to_string();
    }

    let new_rhs = coerce_expr_to(rhs, rt, lt);
    let Some(new_rhs) = new_rhs else {
        return code.to_string();
    };
    format!("{lhs} = {new_rhs}{after}")
}

/// Determine the type an assignment's LHS expects: a declaration `TYPE name`
/// yields `TYPE`; a bare lvalue `name`/`name.swizzle` is looked up/measured.
fn lhs_type(lhs: &str, env: &TypeEnv) -> Option<Ty> {
    let mut it = lhs.split_whitespace();
    let first = it.next()?;
    if let Some(ty) = parse_ty(first) {
        // Declaration form `TYPE name` (ignore any `name` here).
        return Some(ty);
    }
    // Lvalue form (possibly with swizzle).
    infer(lhs, env)
}

/// Produce an expression converting `expr` (typed `from`) to `to`, or `None` when
/// the conversion is not one of the reproduced leniencies.
fn coerce_expr_to(expr: &str, from: Ty, to: Ty) -> Option<String> {
    if from.size == to.size && from.base != to.base && from.size == 1 {
        // Scalar base cast (float↔int↔uint), §7.1.
        return Some(format!("{}({})", to.ctor(), expr));
    }
    if from.base != to.base {
        return None; // vector base mismatch — not reproduced.
    }
    if from.size > to.size {
        // Wider→narrower vector: truncate via swizzle.
        let sw = &"xyzw"[..to.size as usize];
        return Some(format!("({expr}).{sw}"));
    }
    if from.size == 1 && to.size > 1 {
        // Scalar→vector splat.
        return Some(format!("{}({})", to.ctor(), expr));
    }
    None
}

/// Find the byte index of a top-level plain `=` assignment operator in `stmt`,
/// skipping `==`, `<=`, `>=`, `!=`, `+=`, and any `=` inside parentheses.
fn find_plain_assign(stmt: &str) -> Option<usize> {
    let bytes = stmt.as_bytes();
    let mut depth = 0i32;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'=' if depth == 0 => {
                let prev = i.checked_sub(1).map(|p| bytes[p]);
                let next = bytes.get(i + 1).copied();
                if next == Some(b'=') {
                    continue;
                }
                if matches!(prev, Some(b'=' | b'!' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/')) {
                    continue;
                }
                return Some(i);
            }
            _ => {}
        }
    }
    None
}

/// Coerce arguments of recognised calls (`texture`, `mix`, and user functions) so
/// each argument's width matches the callee's expectation. Operates left-to-right
/// on `code`, rebuilding it.
fn coerce_calls(code: &str, env: &TypeEnv) -> String {
    let bytes = code.as_bytes();
    let mut out = String::with_capacity(code.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        // Identify an identifier token immediately followed by `(`.
        if is_ident(bytes[i]) && (i == 0 || !is_ident(bytes[i - 1])) {
            let start = i;
            let mut j = i;
            while j < bytes.len() && is_ident(bytes[j]) {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' {
                let name = &code[start..j];
                if let Some((args_span, end)) = arg_list_span(code, j) {
                    let handled = rewrite_call(name, &code[args_span.0..args_span.1], env);
                    if let Some(new_args) = handled {
                        out.push_str(name);
                        out.push('(');
                        // Recurse into rewritten args' own nested calls.
                        out.push_str(&coerce_calls(&new_args, env));
                        out.push(')');
                        i = end;
                        continue;
                    }
                }
            }
            out.push_str(&code[start..j]);
            i = j;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Given `(` at `open`, return `((args_start, args_end), index-past-`)`)`.
fn arg_list_span(code: &str, open: usize) -> Option<((usize, usize), usize)> {
    let mut depth = 0i32;
    for (i, &b) in code.as_bytes().iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(((open + 1, i), i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Rewrite a call's argument string, coercing each argument to its expected
/// width. Returns `None` when the callee is unknown or nothing needs changing.
fn rewrite_call(name: &str, args: &str, env: &TypeEnv) -> Option<String> {
    let parts = split_top_commas(args);
    let want: Vec<Option<Ty>> = match name {
        "texture" | "textureLod" => {
            // 2-D sampler coordinate is arg index 1 → vec2.
            let mut w = vec![None; parts.len()];
            if parts.len() >= 2 {
                w[1] = Some(Ty {
                    base: Base::Float,
                    size: 2,
                });
            }
            w
        }
        "mix" => {
            // Unify the two operands to the narrower vector width.
            let t0 = parts.first().and_then(|a| infer(a, env));
            let t1 = parts.get(1).and_then(|a| infer(a, env));
            if let (Some(a), Some(b)) = (t0, t1)
                && a.base == b.base
                && a.size != b.size
            {
                let target = Ty {
                    base: a.base,
                    size: a.size.min(b.size),
                };
                let mut w = vec![None; parts.len()];
                w[0] = Some(target);
                w[1] = Some(target);
                w
            } else {
                return None;
            }
        }
        _ => {
            let (sig, _) = env.funcs.get(name)?;
            sig.iter()
                .cloned()
                .chain(std::iter::repeat(None))
                .take(parts.len())
                .collect()
        }
    };

    let mut changed = false;
    let mut rebuilt: Vec<String> = Vec::with_capacity(parts.len());
    for (idx, part) in parts.iter().enumerate() {
        let expected = want.get(idx).copied().flatten();
        if let Some(exp) = expected
            && let Some(cur) = infer(part, env)
            && cur != exp
            && let Some(fixed) = coerce_expr_to(part.trim(), cur, exp)
        {
            rebuilt.push(fixed);
            changed = true;
        } else {
            rebuilt.push(part.trim().to_string());
        }
    }
    changed.then(|| rebuilt.join(", "))
}

/// Split a call-argument string on top-level commas (ignoring commas nested in
/// parentheses/brackets).
fn split_top_commas(args: &str) -> Vec<String> {
    if args.trim().is_empty() {
        return Vec::new();
    }
    let bytes = args.as_bytes();
    let mut depth = 0i32;
    let mut parts = Vec::new();
    let mut last = 0;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(args[last..i].to_string());
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(args[last..].to_string());
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_vec4_to_vec2_truncates() {
        // cloudmotion.vert: `v_NoiseCoord = v_TexCoord;` (vec2 = vec4).
        let src =
            "out vec4 v_TexCoord;\nout vec2 v_NoiseCoord;\nvoid main() {\nv_NoiseCoord = v_TexCoord;\n}\n";
        let got = coerce_shapes(src);
        assert!(got.contains("v_NoiseCoord = (v_TexCoord).xy;"), "{got}");
    }

    #[test]
    fn scalar_float_to_int_wraps() {
        // blur_gaussian.frag: `int i = -iterations;` (iterations is float).
        let src = "void main() {\nfloat iterations = 3.0;\nint i = -iterations;\n}\n";
        let got = coerce_shapes(src);
        assert!(got.contains("int i = int(-iterations);"), "{got}");
    }

    #[test]
    fn mix_unifies_operand_widths() {
        // hue_shift.frag: mix(vec4, vec3, float) → operands to vec3.
        let src = "void main() {\nvec4 albedo = vec4(0.0);\nvec3 newAlbedo = vec3(0.0);\nfloat mask = 1.0;\nalbedo.rgb = mix(albedo, newAlbedo, mask);\n}\n";
        let got = coerce_shapes(src);
        assert!(got.contains("mix((albedo).xyz, newAlbedo, mask)"), "{got}");
    }

    #[test]
    fn user_func_arg_truncated() {
        // shimmer.frag: rotateVec2(vec4, float) with a vec2 first param.
        let src = "vec2 rotateVec2(vec2 v, float a) { return v; }\nvec4 v_TexCoord;\nvoid main() {\nvec2 c = rotateVec2(v_TexCoord, 1.0);\n}\n";
        let got = coerce_shapes(src);
        assert!(got.contains("rotateVec2((v_TexCoord).xy, 1.0)"), "{got}");
    }

    #[test]
    fn texture_coord_forced_to_vec2() {
        let src = "vec4 v_TexCoord;\nvoid main() {\nvec4 c = texture(g_Tex, v_TexCoord);\n}\n";
        let got = coerce_shapes(src);
        assert!(got.contains("texture(g_Tex, (v_TexCoord).xy)"), "{got}");
    }

    #[test]
    fn matching_types_untouched() {
        let src = "void main() {\nvec3 a = vec3(0.0);\nvec3 b = a;\n}\n";
        let got = coerce_shapes(src);
        assert!(got.contains("vec3 b = a;"), "{got}");
    }
}
