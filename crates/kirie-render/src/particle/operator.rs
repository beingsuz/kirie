//! Particle operators — run every frame after aging, in declaration order
//! (docs/render-architecture.md §7.3 operator table; behavior
//! `CParticle.cpp:1024-1697`). Parameter names and per-name defaults match the
//! spec table exactly (SPEC §V10).
//!
//! "Randomized once per operator" state (turbulence phase / speed) is drawn at
//! compile time from the system RNG. "Randomized once per particle" state
//! (oscillator frequency / phase / scale) is derived on the fly from the
//! particle's seed and the operator's salt, so it is stable for the particle's
//! whole life without per-particle storage (SPEC §V5).

use kirie_scene::particle::NamedStage;
use kirie_scene::value::Vec3;

use super::math;
use super::noise;
use super::param::Params;
use super::rng::{self, Rng};
use super::state::{Overrides, Particle};

/// Context threaded into every operator each step.
pub struct StepCtx<'a> {
    /// Step delta (seconds, already capped).
    pub dt: f32,
    /// Absolute simulated time (seconds).
    pub time: f32,
    /// The scene-object instance overrides (supplies `speed`).
    pub overrides: &'a Overrides,
    /// Control-point positions in system-local space.
    pub control_points: &'a [Vec3],
}

/// Linear ramp of a normalized life position through `[start, end]` returning a
/// multiplier lerped `startvalue → endvalue` (clamped outside the window).
#[inline]
fn ramp(lp: f32, start: f32, end: f32, sv: f32, ev: f32) -> f32 {
    if end <= start {
        return if lp >= end { ev } else { sv };
    }
    let t = ((lp - start) / (end - start)).clamp(0.0, 1.0);
    sv + (ev - sv) * t
}

/// A compiled operator with resolved parameters. Variant field names mirror the
/// JSON parameter names in the docs §7.3 operator table.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum Operator {
    Movement {
        drag: f32,
        gravity: Vec3,
    },
    AngularMovement {
        drag: f32,
        force: Vec3,
    },
    AlphaFade {
        fadeintime: f32,
        fadeouttime: f32,
    },
    SizeChange {
        starttime: f32,
        endtime: f32,
        startvalue: f32,
        endvalue: f32,
    },
    AlphaChange {
        starttime: f32,
        endtime: f32,
        startvalue: f32,
        endvalue: f32,
    },
    ColorChange {
        startvalue: Vec3,
        endvalue: Vec3,
    },
    Turbulence {
        scale: f32,
        timescale: f32,
        mask: Vec3,
        /// Randomized once per operator (docs §7.3).
        phase: Vec3,
        turb_speed: f32,
    },
    Vortex {
        controlpoint: usize,
        axis: Vec3,
        offset: Vec3,
        distanceinner: f32,
        distanceouter: f32,
        speedinner: f32,
        speedouter: f32,
        centerforce: f32,
    },
    ControlPointAttract {
        controlpoint: usize,
        scale: f32,
        threshold: f32,
    },
    OscillateAlpha {
        frequencymin: f32,
        frequencymax: f32,
        scalemin: f32,
        scalemax: f32,
        phasemin: f32,
        phasemax: f32,
        salt: u32,
    },
    OscillateSize {
        frequencymin: f32,
        frequencymax: f32,
        scalemin: f32,
        scalemax: f32,
        phasemin: f32,
        phasemax: f32,
        salt: u32,
    },
    OscillatePosition {
        frequencymax: f32,
        scalemax: f32,
        mask: Vec3,
        salt: u32,
    },
    /// An unimplemented operator name — preserved but a no-op (docs §7.3).
    Unknown(String),
}

impl Operator {
    /// Compile one scene [`NamedStage`] into a typed operator. `salt` seeds the
    /// per-operator randomization; `rng` supplies "once per operator" values.
    #[must_use]
    pub fn compile(stage: &NamedStage, salt: u32, rng: &mut Rng) -> Self {
        let p = Params::new(stage);
        match stage.name.as_str() {
            "movement" => Operator::Movement {
                drag: p.f32("drag", 0.0),
                gravity: math::flip_y(p.vec3("gravity", [0.0, 0.0, 0.0])),
            },
            "angularmovement" => Operator::AngularMovement {
                drag: p.f32("drag", 0.0),
                force: p.vec3("force", [0.0, 0.0, 0.0]),
            },
            "alphafade" => Operator::AlphaFade {
                fadeintime: p.f32("fadeintime", 0.5),
                fadeouttime: p.f32("fadeouttime", 0.5),
            },
            "sizechange" => Operator::SizeChange {
                starttime: p.f32("starttime", 0.0),
                endtime: p.f32("endtime", 1.0),
                startvalue: p.f32("startvalue", 1.0),
                endvalue: p.f32("endvalue", 0.0),
            },
            "alphachange" => Operator::AlphaChange {
                starttime: p.f32("starttime", 0.0),
                endtime: p.f32("endtime", 1.0),
                startvalue: p.f32("startvalue", 1.0),
                endvalue: p.f32("endvalue", 0.0),
            },
            "colorchange" => Operator::ColorChange {
                startvalue: p.vec3("startvalue", [1.0, 1.0, 1.0]),
                endvalue: p.vec3("endvalue", [1.0, 1.0, 1.0]),
            },
            "turbulence" => {
                let phasemin = p.f32("phasemin", 0.0);
                let phasemax = p.f32("phasemax", 0.0);
                let speedmin = p.f32("speedmin", 500.0);
                let speedmax = p.f32("speedmax", 1000.0);
                Operator::Turbulence {
                    scale: p.f32("scale", 0.005),
                    timescale: p.f32("timescale", 0.01),
                    mask: p.vec3("mask", [1.0, 1.0, 0.0]),
                    phase: [
                        rng.range(phasemin, phasemax),
                        rng.range(phasemin, phasemax),
                        rng.range(phasemin, phasemax),
                    ],
                    turb_speed: rng.range(speedmin, speedmax),
                }
            }
            "vortex" | "vortex_v2" => Operator::Vortex {
                controlpoint: p.i64("controlpoint", 0).max(0) as usize,
                axis: p.vec3("axis", [0.0, 0.0, 1.0]),
                offset: p.vec3("offset", [0.0, 0.0, 0.0]),
                distanceinner: p.f32("distanceinner", 500.0),
                distanceouter: p.f32("distanceouter", 650.0),
                speedinner: p.f32("speedinner", 2500.0),
                speedouter: p.f32("speedouter", 0.0),
                centerforce: p.f32("centerforce", 1.0),
            },
            "controlpointattract" => Operator::ControlPointAttract {
                controlpoint: p.i64("controlpoint", 0).max(0) as usize,
                scale: p.f32("scale", 100.0),
                threshold: p.f32("threshold", 1000.0),
            },
            "oscillatealpha" => Operator::OscillateAlpha {
                frequencymin: p.f32("frequencymin", 0.0),
                frequencymax: p.f32("frequencymax", 10.0),
                scalemin: p.f32("scalemin", 0.0),
                scalemax: p.f32("scalemax", 1.0),
                phasemin: p.f32("phasemin", 0.0),
                phasemax: p.f32("phasemax", std::f32::consts::TAU),
                salt,
            },
            "oscillatesize" => Operator::OscillateSize {
                frequencymin: p.f32("frequencymin", 0.0),
                frequencymax: p.f32("frequencymax", 10.0),
                scalemin: p.f32("scalemin", 0.8),
                scalemax: p.f32("scalemax", 1.2),
                phasemin: p.f32("phasemin", 0.0),
                phasemax: p.f32("phasemax", std::f32::consts::TAU),
                salt,
            },
            "oscillateposition" => Operator::OscillatePosition {
                frequencymax: p.f32("frequencymax", 5.0),
                scalemax: p.f32("scalemax", 10.0),
                mask: p.vec3("mask", [1.0, 1.0, 0.0]),
                salt,
            },
            other => Operator::Unknown(other.to_owned()),
        }
    }

    /// Apply this operator to one live particle for a step of `ctx.dt`.
    pub fn apply(&self, pt: &mut Particle, ctx: &StepCtx<'_>) {
        let dt = ctx.dt;
        let speed = ctx.overrides.speed;
        match self {
            Operator::Movement { drag, gravity } => {
                pt.position = math::mul_add(pt.position, pt.velocity, dt);
                pt.velocity = math::mul_add(pt.velocity, *gravity, dt * speed);
                pt.velocity = math::mul(pt.velocity, (1.0 - drag * dt).max(0.0));
            }
            Operator::AngularMovement { drag, force } => {
                pt.rotation = math::mul_add(pt.rotation, pt.angular_velocity, dt);
                pt.angular_velocity = math::mul_add(pt.angular_velocity, *force, dt * speed);
                pt.angular_velocity = math::mul(pt.angular_velocity, (1.0 - drag * dt).max(0.0));
                pt.rotation = math::wrap_pi(pt.rotation);
            }
            Operator::AlphaFade {
                fadeintime,
                fadeouttime,
            } => {
                let lp = pt.life_pos();
                let mut a = pt.initial.alpha;
                if *fadeintime > 0.0 && lp < *fadeintime {
                    a *= lp / *fadeintime;
                }
                if *fadeouttime > 0.0 && lp > 1.0 - *fadeouttime {
                    a *= (1.0 - lp) / *fadeouttime;
                }
                pt.alpha = a;
            }
            Operator::SizeChange {
                starttime,
                endtime,
                startvalue,
                endvalue,
            } => {
                pt.size = pt.initial.size * ramp(pt.life_pos(), *starttime, *endtime, *startvalue, *endvalue);
            }
            Operator::AlphaChange {
                starttime,
                endtime,
                startvalue,
                endvalue,
            } => {
                pt.alpha =
                    pt.initial.alpha * ramp(pt.life_pos(), *starttime, *endtime, *startvalue, *endvalue);
            }
            Operator::ColorChange { startvalue, endvalue } => {
                let lp = pt.life_pos();
                let m = [
                    startvalue[0] + (endvalue[0] - startvalue[0]) * lp,
                    startvalue[1] + (endvalue[1] - startvalue[1]) * lp,
                    startvalue[2] + (endvalue[2] - startvalue[2]) * lp,
                ];
                pt.color = math::mul_comp(pt.initial.color, m);
            }
            Operator::Turbulence {
                scale,
                timescale,
                mask,
                phase,
                turb_speed,
            } => {
                let coord = math::mul(
                    math::add(math::add(pt.position, *phase), [ctx.time * timescale; 3]),
                    scale * 2.0,
                );
                let dir = noise::curl(coord);
                let step = math::mul_comp(math::mul(dir, *turb_speed * dt * speed), *mask);
                pt.velocity = math::add(pt.velocity, step);
            }
            Operator::Vortex {
                controlpoint,
                axis,
                offset,
                distanceinner,
                distanceouter,
                speedinner,
                speedouter,
                centerforce,
            } => {
                let cp = ctx.control_points.get(*controlpoint).copied().unwrap_or([0.0; 3]);
                let center = math::add(cp, *offset);
                let radial = math::sub(pt.position, center);
                let dist = math::length(radial);
                let axis_n = math::normalize_or(*axis, [0.0, 0.0, 1.0]);
                let tangent = math::normalize_or(math::cross(axis_n, radial), [0.0, 0.0, 0.0]);
                let t = if *distanceouter > *distanceinner {
                    ((dist - distanceinner) / (distanceouter - distanceinner)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let spd = speedinner + (speedouter - speedinner) * t;
                pt.velocity = math::mul_add(pt.velocity, tangent, spd * dt * speed);
                // Radial pull toward the axis center (centerforce, UNVERIFIED
                // exact scaling; docs §7.3 "maintain distance"/"centerforce").
                if *centerforce != 0.0 {
                    let inward = math::normalize_or(math::mul(radial, -1.0), [0.0; 3]);
                    pt.velocity = math::mul_add(pt.velocity, inward, centerforce * dt * speed);
                }
            }
            Operator::ControlPointAttract {
                controlpoint,
                scale,
                threshold,
            } => {
                let cp = ctx.control_points.get(*controlpoint).copied().unwrap_or([0.0; 3]);
                let to = math::sub(cp, pt.position);
                if math::length(to) < threshold / 2.0 {
                    let dir = math::normalize_or(to, [0.0; 3]);
                    pt.velocity = math::mul_add(pt.velocity, dir, scale * dt * speed);
                }
            }
            Operator::OscillateAlpha {
                frequencymin,
                frequencymax,
                scalemin,
                scalemax,
                phasemin,
                phasemax,
                salt,
            } => {
                let m = oscillate(
                    pt,
                    *salt,
                    *frequencymin,
                    *frequencymax,
                    *scalemin,
                    *scalemax,
                    *phasemin,
                    *phasemax,
                );
                pt.alpha *= m;
            }
            Operator::OscillateSize {
                frequencymin,
                frequencymax,
                scalemin,
                scalemax,
                phasemin,
                phasemax,
                salt,
            } => {
                let m = oscillate(
                    pt,
                    *salt,
                    *frequencymin,
                    *frequencymax,
                    *scalemin,
                    *scalemax,
                    *phasemin,
                    *phasemax,
                );
                pt.size *= m;
            }
            Operator::OscillatePosition {
                frequencymax,
                scalemax,
                mask,
                salt,
            } => {
                let mut r = rng::derived(pt.seed, *salt);
                let age = pt.age;
                for (pos, &m) in pt.position.iter_mut().zip(mask.iter()) {
                    if m == 0.0 {
                        continue;
                    }
                    let freq = r.range(0.0, *frequencymax);
                    let scl = r.range(0.0, *scalemax);
                    let phase = r.range(0.0, std::f32::consts::TAU);
                    *pos += -scl * freq * (freq * age + phase).sin() * dt * m * speed;
                }
            }
            Operator::Unknown(_) => {}
        }
    }
}

/// The shared randomized-cosine multiplier for `oscillatealpha`/`oscillatesize`
/// (docs/render-architecture.md §7.3): a per-particle frequency / phase / scale
/// window driving a cosine that oscillates the current (base) value.
#[allow(clippy::too_many_arguments)]
#[inline]
fn oscillate(
    pt: &Particle,
    salt: u32,
    fmin: f32,
    fmax: f32,
    smin: f32,
    smax: f32,
    pmin: f32,
    pmax: f32,
) -> f32 {
    let mut r = rng::derived(pt.seed, salt);
    let freq = r.range(fmin, fmax);
    let phase = r.range(pmin, pmax);
    let smin_r = smin.min(smax);
    let smax_r = smin.max(smax);
    let unit = 0.5 * ((freq * pt.age + phase).cos() + 1.0);
    smin_r + (smax_r - smin_r) * unit
}
