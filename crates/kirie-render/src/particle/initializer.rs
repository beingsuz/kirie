//! Particle initializers — run once per particle at spawn, in declaration order
//! (docs/render-architecture.md §7.3 initializer table; behavior
//! `CParticle.cpp:725-1020`). Parameter names and per-name defaults match the
//! spec table exactly (SPEC §V10).

use kirie_scene::particle::NamedStage;
use kirie_scene::value::Vec3;

use super::math;
use super::noise;
use super::param::Params;
use super::rng::Rng;
use super::state::{Overrides, Particle};

/// Context threaded into every initializer at spawn time.
pub struct SpawnCtx<'a> {
    /// The scene-object instance overrides.
    pub overrides: &'a Overrides,
    /// The particle system's `flags & 4` (perspective / 3D) bit
    /// (docs/render-architecture.md §7.3): controls whether Z is kept.
    pub perspective: bool,
    /// Control-point positions in system-local space (index 0 = CP0).
    pub control_points: &'a [Vec3],
}

/// A compiled initializer with resolved parameters. Variant field names mirror
/// the JSON parameter names in the docs §7.3 initializer table.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum Initializer {
    ColorRandom {
        min: Vec3,
        max: Vec3,
    },
    SizeRandom {
        min: f32,
        max: f32,
        exponent: f32,
    },
    AlphaRandom {
        min: f32,
        max: f32,
    },
    LifetimeRandom {
        min: f32,
        max: f32,
    },
    VelocityRandom {
        min: Vec3,
        max: Vec3,
    },
    RotationRandom {
        min: Vec3,
        max: Vec3,
    },
    AngularVelocityRandom {
        min: Vec3,
        max: Vec3,
        exponent: f32,
    },
    TurbulentVelocityRandom {
        speedmin: f32,
        speedmax: f32,
        scale: f32,
        offset: f32,
        forward: Vec3,
        right: Vec3,
        phasemin: f32,
        phasemax: f32,
    },
    MapSequenceAroundControlPoint {
        controlpoint: usize,
        count: i64,
        speedmin: Vec3,
        speedmax: Vec3,
        /// Round-robin spawn counter (advances every spawn).
        counter: i64,
    },
    /// An initializer whose `name` the reference does not implement — preserved
    /// but a no-op (docs/render-architecture.md §7.3: unknown names logged and
    /// ignored). Carries the name for diagnostics.
    Unknown(String),
}

#[inline]
fn biased(rng: &mut Rng, min: f32, max: f32, exponent: f32) -> f32 {
    min + rng.unit().powf(exponent) * (max - min)
}

impl Initializer {
    /// Compile one scene [`NamedStage`] into a typed initializer.
    #[must_use]
    pub fn compile(stage: &NamedStage) -> Self {
        let p = Params::new(stage);
        match stage.name.as_str() {
            "colorrandom" => Initializer::ColorRandom {
                min: p.color3("min", [0.0, 0.0, 0.0]),
                max: p.color3("max", [1.0, 1.0, 1.0]),
            },
            "sizerandom" => Initializer::SizeRandom {
                min: p.f32("min", 0.0),
                max: p.f32("max", 20.0),
                exponent: p.f32("exponent", 1.0),
            },
            "alpharandom" => Initializer::AlphaRandom {
                min: p.f32("min", 0.05),
                max: p.f32("max", 1.0),
            },
            "lifetimerandom" => Initializer::LifetimeRandom {
                min: p.f32("min", 0.0),
                max: p.f32("max", 1.0),
            },
            "velocityrandom" => Initializer::VelocityRandom {
                min: p.vec3("min", [-32.0, -32.0, -32.0]),
                max: p.vec3("max", [32.0, 32.0, 32.0]),
            },
            "rotationrandom" => Initializer::RotationRandom {
                min: p.vec3("min", [0.0, 0.0, 0.0]),
                max: p.vec3("max", [0.0, 0.0, std::f32::consts::TAU]),
            },
            "angularvelocityrandom" => Initializer::AngularVelocityRandom {
                min: p.vec3("min", [0.0, 0.0, -5.0]),
                max: p.vec3("max", [0.0, 0.0, 5.0]),
                exponent: p.f32("exponent", 1.0),
            },
            "turbulentvelocityrandom" => Initializer::TurbulentVelocityRandom {
                speedmin: p.f32("speedmin", 100.0),
                speedmax: p.f32("speedmax", 250.0),
                scale: p.f32("scale", 1.0),
                offset: p.f32("offset", 0.0),
                forward: p.vec3("forward", [0.0, 1.0, 0.0]),
                right: p.vec3("right", [0.0, 0.0, 1.0]),
                phasemin: p.f32("phasemin", 0.0),
                phasemax: p.f32("phasemax", 0.1),
            },
            "mapsequencearoundcontrolpoint" => Initializer::MapSequenceAroundControlPoint {
                controlpoint: p.i64("controlpoint", 0).max(0) as usize,
                count: p.i64("count", 1).max(1),
                speedmin: p.vec3("speedmin", [0.0, 0.0, 0.0]),
                speedmax: p.vec3("speedmax", [0.0, 0.0, 0.0]),
                counter: 0,
            },
            other => Initializer::Unknown(other.to_owned()),
        }
    }

    /// Apply this initializer to a freshly spawned particle.
    pub fn apply(&mut self, pt: &mut Particle, ctx: &SpawnCtx<'_>, rng: &mut Rng) {
        let o = ctx.overrides;
        match self {
            Initializer::ColorRandom { min, max } => {
                let c = [
                    rng.range(min[0], max[0]),
                    rng.range(min[1], max[1]),
                    rng.range(min[2], max[2]),
                ];
                pt.color = math::mul_comp(c, o.colorn);
                pt.initial.color = pt.color;
            }
            Initializer::SizeRandom { min, max, exponent } => {
                pt.size = biased(rng, *min, *max, *exponent) * o.size / 2.0;
                pt.initial.size = pt.size;
            }
            Initializer::AlphaRandom { min, max } => {
                pt.alpha = rng.range(*min, *max) * o.alpha;
                pt.initial.alpha = pt.alpha;
            }
            Initializer::LifetimeRandom { min, max } => {
                pt.lifetime = rng.range(*min, *max) * o.lifetime;
            }
            Initializer::VelocityRandom { min, max } => {
                let v = [
                    rng.range(min[0], max[0]),
                    rng.range(min[1], max[1]),
                    rng.range(min[2], max[2]),
                ];
                pt.velocity = math::add(pt.velocity, math::flip_y(v));
            }
            Initializer::RotationRandom { min, max } => {
                pt.rotation = [
                    rng.range(min[0], max[0]),
                    rng.range(min[1], max[1]),
                    rng.range(min[2], max[2]),
                ];
            }
            Initializer::AngularVelocityRandom { min, max, exponent } => {
                pt.angular_velocity = [
                    biased(rng, min[0], max[0], *exponent),
                    biased(rng, min[1], max[1], *exponent),
                    biased(rng, min[2], max[2], *exponent),
                ];
            }
            Initializer::TurbulentVelocityRandom {
                speedmin,
                speedmax,
                scale,
                offset,
                forward,
                right,
                phasemin,
                phasemax,
            } => {
                // UNVERIFIED noise basis (docs/render-architecture.md §7.3): a
                // curl-noise direction cone-limited toward `forward` by `scale/2`
                // and tilted by `offset` around `right`, z zeroed for ortho.
                let phase = rng.range(*phasemin, *phasemax);
                let sample = noise::curl(math::mul(math::add(pt.position, [phase; 3]), 0.01));
                let mut dir = math::add(*forward, math::mul(sample, *scale * 0.5));
                dir = math::add(dir, math::mul(*right, *offset));
                if !ctx.perspective {
                    dir[2] = 0.0;
                }
                let dir = math::normalize_or(dir, *forward);
                pt.velocity = math::mul_add(pt.velocity, dir, rng.range(*speedmin, *speedmax));
            }
            Initializer::MapSequenceAroundControlPoint {
                controlpoint,
                count,
                speedmin,
                speedmax,
                counter,
            } => {
                let cp = ctx.control_points.get(*controlpoint).copied().unwrap_or([0.0; 3]);
                pt.position = cp;
                let base = [
                    rng.range(speedmin[0], speedmax[0]),
                    rng.range(speedmin[1], speedmax[1]),
                    rng.range(speedmin[2], speedmax[2]),
                ];
                let idx = counter.rem_euclid((*count).max(1));
                let angle = std::f32::consts::TAU * (idx as f32) / (*count as f32);
                let (s, c) = angle.sin_cos();
                pt.velocity = math::add(
                    pt.velocity,
                    [base[0] * c - base[1] * s, base[0] * s + base[1] * c, base[2]],
                );
                *counter += 1;
            }
            Initializer::Unknown(_) => {}
        }
    }
}
