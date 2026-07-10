//! Column-major 4×4 matrix math for the scene camera and per-object transforms.
//!
//! Layout matches glm / GLSL / wgpu: 16 floats in **column-major** order
//! (`col0.xyzw, col1.xyzw, …`), so a [`Mat4`] uploads directly into a `mat4`
//! uniform. The reference builds every matrix with glm
//! (docs/render-architecture.md §7.1, §9); the one deliberate deviation is the
//! orthographic projection, which uses the wgpu **zero-to-one** depth range
//! instead of GL's `[-1, 1]` so flat layers at `z = 0` are not clipped by the
//! near plane (docs/render-architecture.md §9 "near = 0 … flat layers sit at
//! z=0"; the GL `[-1,1]` range would put them at clip-z −1 and cull them under
//! wgpu's `0 ≤ z ≤ w` rule).

/// A column-major 4×4 matrix (`[col][row]` flattened, 16 floats).
pub type Mat4 = [f32; 16];

/// The 4×4 identity.
pub const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// `a · b` (column-major, so the result applies `b` first then `a`, matching
/// `glm`'s `a * b`).
#[must_use]
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

/// Right-multiply `m` by a translation (matches `glm::translate(m, t)`).
#[must_use]
pub fn translate(m: &Mat4, t: [f32; 3]) -> Mat4 {
    mul(m, &translation(t))
}

/// A pure translation matrix.
#[must_use]
pub fn translation(t: [f32; 3]) -> Mat4 {
    let mut out = IDENTITY;
    out[12] = t[0];
    out[13] = t[1];
    out[14] = t[2];
    out
}

/// A pure non-uniform scale matrix.
#[must_use]
pub fn scale(s: [f32; 3]) -> Mat4 {
    let mut out = IDENTITY;
    out[0] = s[0];
    out[5] = s[1];
    out[10] = s[2];
    out
}

/// Rotation about Z by `radians` (right-handed, column-major).
#[must_use]
pub fn rotation_z(radians: f32) -> Mat4 {
    let (s, c) = radians.sin_cos();
    let mut out = IDENTITY;
    out[0] = c;
    out[1] = s;
    out[4] = -s;
    out[5] = c;
    out
}

/// Rotation about Y by `radians`.
#[must_use]
pub fn rotation_y(radians: f32) -> Mat4 {
    let (s, c) = radians.sin_cos();
    let mut out = IDENTITY;
    out[0] = c;
    out[2] = -s;
    out[8] = s;
    out[10] = c;
    out
}

/// Rotation about X by `radians`.
#[must_use]
pub fn rotation_x(radians: f32) -> Mat4 {
    let (s, c) = radians.sin_cos();
    let mut out = IDENTITY;
    out[5] = c;
    out[6] = s;
    out[9] = -s;
    out[10] = c;
    out
}

/// Orthographic projection with the wgpu zero-to-one depth range
/// (docs/render-architecture.md §9; see the module note on the deviation from
/// GL's `[-1, 1]`). Equivalent to `glm::orthoRH_ZO`.
#[must_use]
pub fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let rl = right - left;
    let tb = top - bottom;
    let fnr = far - near;
    if rl == 0.0 || tb == 0.0 || fnr == 0.0 {
        return IDENTITY;
    }
    let mut out = [0.0f32; 16];
    out[0] = 2.0 / rl;
    out[5] = 2.0 / tb;
    out[10] = -1.0 / fnr;
    out[12] = -(right + left) / rl;
    out[13] = -(top + bottom) / tb;
    out[14] = -near / fnr;
    out[15] = 1.0;
    out
}

/// Right-handed perspective projection with the wgpu zero-to-one depth range
/// (`glm::perspectiveRH_ZO`, docs/render-architecture.md §9). Used by 3D MODEL
/// objects (`CModel.cpp::render` builds `glm::perspective` then renders into the
/// scene FBO). Unlike the reference — whose scene FBO is Y-down, so it negates
/// `m[1][1]` to flip clip-space Y — kirie's scene FBO is Y-up throughout (the 2D
/// layers build Y-up quads through [`ortho`] with no flip), so no Y flip is
/// applied here: a point above the camera maps to +Y clip → the top of the
/// target, upright, consistent with the 2D layers (see [`CModel`] winding note
/// in `model.rs`). `fov_y_radians` is the vertical field of view.
#[must_use]
pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let t = (fov_y_radians / 2.0).tan();
    if t == 0.0 || aspect == 0.0 || (near - far) == 0.0 {
        return IDENTITY;
    }
    let mut out = [0.0f32; 16];
    out[0] = 1.0 / (aspect * t); // c0r0
    out[5] = 1.0 / t; // c1r1
    out[10] = far / (near - far); // c2r2
    out[11] = -1.0; // c2r3
    out[14] = -(far * near) / (far - near); // c3r2
    out
}

/// Right-handed look-at view matrix (matches `glm::lookAt`).
#[must_use]
pub fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize(sub(center, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        s[0],
        u[0],
        -f[0],
        0.0, //
        s[1],
        u[1],
        -f[1],
        0.0, //
        s[2],
        u[2],
        -f[2],
        0.0, //
        -dot(s, eye),
        -dot(u, eye),
        dot(f, eye),
        1.0,
    ]
}

/// General 4×4 inverse; returns [`IDENTITY`] when the matrix is singular (so
/// the `*Inverse` builtin uniforms never produce NaNs, SPEC.md §V9).
#[must_use]
pub fn inverse(m: &Mat4) -> Mat4 {
    let mut inv = [0.0f32; 16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
        + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11]
        - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
        - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11]
        + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
        + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11]
        - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
        - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10]
        + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
        - m[9] * m[3] * m[14]
        - m[13] * m[2] * m[11]
        + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
        + m[8] * m[3] * m[14]
        + m[12] * m[2] * m[11]
        - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
        - m[8] * m[3] * m[13]
        - m[12] * m[1] * m[11]
        + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
        + m[8] * m[2] * m[13]
        + m[12] * m[1] * m[10]
        - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
        + m[5] * m[3] * m[14]
        + m[13] * m[2] * m[7]
        - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
        - m[4] * m[3] * m[14]
        - m[12] * m[2] * m[7]
        + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
        + m[4] * m[3] * m[13]
        + m[12] * m[1] * m[7]
        - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
        - m[4] * m[2] * m[13]
        - m[12] * m[1] * m[6]
        + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
        - m[5] * m[3] * m[10]
        - m[9] * m[2] * m[7]
        + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
        + m[4] * m[3] * m[10]
        + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
        - m[4] * m[3] * m[9]
        - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
        + m[4] * m[2] * m[9]
        + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];

    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if det == 0.0 || !det.is_finite() {
        return IDENTITY;
    }
    let inv_det = 1.0 / det;
    for v in &mut inv {
        *v *= inv_det;
    }
    inv
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &Mat4, b: &Mat4) {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "elem {i}: {x} vs {y}");
        }
    }

    /// Transform a point (column vector) by a column-major matrix.
    fn apply(m: &Mat4, p: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            for k in 0..4 {
                out[row] += m[k * 4 + row] * p[k];
            }
        }
        out
    }

    #[test]
    fn identity_multiplies_to_self() {
        approx(&mul(&IDENTITY, &IDENTITY), &IDENTITY);
    }

    #[test]
    fn mul_is_left_applied_last() {
        // translate then scale: mul(scale, translate) applies translate first.
        let t = translation([1.0, 2.0, 3.0]);
        let s = scale([2.0, 2.0, 2.0]);
        let m = mul(&s, &t);
        let p = apply(&m, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!([p[0], p[1], p[2]], [2.0, 4.0, 6.0]);
    }

    #[test]
    fn ortho_maps_center_and_depth() {
        // A centered ortho maps origin → NDC origin and near (z=0) → clip-z 0
        // (the wgpu ZO convention the module documents).
        let m = ortho(-960.0, 960.0, -540.0, 540.0, 0.0, 1000.0);
        let c = apply(&m, [0.0, 0.0, 0.0, 1.0]);
        assert!((c[0]).abs() < 1e-6 && (c[1]).abs() < 1e-6);
        assert!((c[2]).abs() < 1e-6, "z=0 (near) maps to clip-z 0, not -1");
        // Right edge maps to x=+1.
        let r = apply(&m, [960.0, 540.0, 0.0, 1.0]);
        assert!((r[0] - 1.0).abs() < 1e-6 && (r[1] - 1.0).abs() < 1e-6);
        // Right-handed ortho (glm::orthoRH_ZO, docs §9): the view looks down -Z,
        // so the visible depth span is z ∈ [-far, -near]. The far plane at
        // z = -1000 maps to clip-z 1; the near plane (z=0) to clip-z 0 above.
        let f = apply(&m, [0.0, 0.0, -1000.0, 1.0]);
        assert!((f[2] - 1.0).abs() < 1e-6, "z=-far maps to clip-z 1 (RH_ZO)");
    }

    #[test]
    fn look_at_identity_camera() {
        // Eye on +Z looking at origin with +Y up is a pure -Z translation.
        let m = look_at([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let p = apply(&m, [0.0, 0.0, 0.0, 1.0]);
        assert!((p[0]).abs() < 1e-5 && (p[1]).abs() < 1e-5);
        assert!((p[2] + 10.0).abs() < 1e-5, "origin sits 10 in front of camera");
    }

    #[test]
    fn perspective_zo_maps_near_and_far() {
        // A RH_ZO perspective maps a point on the near plane (view z = -near) to
        // clip-z 0 and a point on the far plane (view z = -far) to clip-z w (→
        // NDC-z 1 after divide), the wgpu depth convention (docs §9). Points are
        // in *view* space (camera at origin looking down -Z).
        let p = perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 10.0);
        let near = apply(&p, [0.0, 0.0, -0.1, 1.0]);
        assert!((near[2] / near[3]).abs() < 1e-4, "near plane → NDC-z 0");
        let far = apply(&p, [0.0, 0.0, -10.0, 1.0]);
        assert!((far[2] / far[3] - 1.0).abs() < 1e-4, "far plane → NDC-z 1");
        // No Y flip: a point above the axis stays at +Y clip (upright).
        let up = apply(&p, [0.0, 1.0, -1.0, 1.0]);
        assert!(up[1] > 0.0, "world +Y maps to +Y clip (no flip)");
    }

    #[test]
    fn inverse_round_trips() {
        let m = mul(
            &translation([3.0, -2.0, 5.0]),
            &mul(&rotation_z(0.7), &scale([2.0, 0.5, 1.5])),
        );
        approx(&mul(&m, &inverse(&m)), &IDENTITY);
        approx(&mul(&inverse(&m), &m), &IDENTITY);
    }

    #[test]
    fn singular_inverse_is_identity_not_nan() {
        // A zero-scale (rank-deficient) matrix must not yield NaNs (SPEC §V9).
        let m = scale([0.0, 1.0, 1.0]);
        let inv = inverse(&m);
        assert!(inv.iter().all(|v| v.is_finite()));
        assert_eq!(inv, IDENTITY);
    }

    #[test]
    fn rotation_z_quarter_turn() {
        let m = rotation_z(std::f32::consts::FRAC_PI_2);
        let p = apply(&m, [1.0, 0.0, 0.0, 1.0]);
        assert!((p[0]).abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6);
    }
}
