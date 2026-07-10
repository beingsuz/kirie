//! Curl noise for the `turbulence` operator and `turbulentvelocityrandom`
//! initializer (docs/render-architecture.md §7.3).
//!
//! The reference samples a curl-noise field to get a divergence-free direction
//! that pushes particles along smooth swirls. The *exact* noise basis in the
//! C++ engine is UNVERIFIED (not part of any format), so we use a classic
//! Perlin gradient noise and take the analytic curl of a vector potential built
//! from three shifted samples — a standard, divergence-free construction. It is
//! deterministic (a pure function of position), which is what the operator
//! semantics require ("curl noise at `(pos + phase + t*timescale) * scale*2`").

use kirie_scene::value::Vec3;

use super::math;

/// Classic Perlin permutation (Ken Perlin's reference table, doubled).
const PERM: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
    37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27,
    166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
    102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119,
    248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110,
    79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179,
    162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176,
    115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78,
    66, 215, 61, 156, 180,
];

#[inline]
fn perm(i: i32) -> i32 {
    i32::from(PERM[(i & 255) as usize])
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn grad(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

/// Perlin noise in roughly `[-1, 1]` at a point.
#[must_use]
pub fn perlin(p: Vec3) -> f32 {
    let (x, y, z) = (p[0], p[1], p[2]);
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let (xf, yf, zf) = (x - x.floor(), y - y.floor(), z - z.floor());
    let (u, v, w) = (fade(xf), fade(yf), fade(zf));

    let a = perm(xi) + yi;
    let aa = perm(a) + zi;
    let ab = perm(a + 1) + zi;
    let b = perm(xi + 1) + yi;
    let ba = perm(b) + zi;
    let bb = perm(b + 1) + zi;

    let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
    let x1 = lerp(u, grad(perm(aa), xf, yf, zf), grad(perm(ba), xf - 1.0, yf, zf));
    let x2 = lerp(
        u,
        grad(perm(ab), xf, yf - 1.0, zf),
        grad(perm(bb), xf - 1.0, yf - 1.0, zf),
    );
    let y1 = lerp(v, x1, x2);
    let x3 = lerp(
        u,
        grad(perm(aa + 1), xf, yf, zf - 1.0),
        grad(perm(ba + 1), xf - 1.0, yf, zf - 1.0),
    );
    let x4 = lerp(
        u,
        grad(perm(ab + 1), xf, yf - 1.0, zf - 1.0),
        grad(perm(bb + 1), xf - 1.0, yf - 1.0, zf - 1.0),
    );
    let y2 = lerp(v, x3, x4);
    lerp(w, y1, y2)
}

/// A divergence-free curl-noise direction at `p`.
///
/// Builds a vector potential `Ψ = (perlin(p), perlin(p+o1), perlin(p+o2))` from
/// three decorrelated Perlin samples and returns `∇ × Ψ` via central
/// differences. The result is not normalized (callers scale it); it is a
/// deterministic pure function of `p` (SPEC §V13 stable behavior).
#[must_use]
pub fn curl(p: Vec3) -> Vec3 {
    const EPS: f32 = 1e-2;
    // Decorrelating offsets so the three potential components differ.
    const O1: Vec3 = [123.4, 56.7, 89.1];
    const O2: Vec3 = [-45.6, 78.9, -12.3];
    let psi = |q: Vec3| -> Vec3 { [perlin(q), perlin(math::add(q, O1)), perlin(math::add(q, O2))] };
    let dx = math::sub(psi([p[0] + EPS, p[1], p[2]]), psi([p[0] - EPS, p[1], p[2]]));
    let dy = math::sub(psi([p[0], p[1] + EPS, p[2]]), psi([p[0], p[1] - EPS, p[2]]));
    let dz = math::sub(psi([p[0], p[1], p[2] + EPS]), psi([p[0], p[1], p[2] - EPS]));
    let inv = 1.0 / (2.0 * EPS);
    // curl = (dPsiz/dy - dPsiy/dz, dPsix/dz - dPsiz/dx, dPsiy/dx - dPsix/dy)
    [
        (dy[2] - dz[1]) * inv,
        (dz[0] - dx[2]) * inv,
        (dx[1] - dy[0]) * inv,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_is_deterministic_and_bounded() {
        let p = [1.3, -2.7, 0.9];
        assert_eq!(perlin(p), perlin(p));
        for i in 0..1000 {
            let f = i as f32 * 0.137;
            let n = perlin([f, f * 0.5, -f]);
            assert!((-1.2..=1.2).contains(&n), "n={n}");
        }
    }

    #[test]
    fn curl_is_deterministic() {
        let p = [0.5, 1.5, -0.3];
        assert_eq!(curl(p), curl(p));
    }
}
