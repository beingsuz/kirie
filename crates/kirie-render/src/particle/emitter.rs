//! Particle emitters — shape + rate/burst scheduling
//! (docs/render-architecture.md §7.3: `boxrandom`, `sphererandom`; anything
//! else is logged and ignored). Fields and defaults come from the scene-side
//! [`Emitter`] model (`ObjectParser.cpp:745-775`).
//!
//! Emission accumulates `dt * rate * instanceOverride.rate * audioFactor`
//! (audio adds, never gates; we run with `audioFactor = 1` headless). `flags`
//! bit 0x2 limits one spawn per frame; bit 0x4 drives random periodic
//! emission windows (`min/maxperiodicdelay`, `min/maxperiodicduration`).

use kirie_scene::particle::Emitter;
use kirie_scene::value::Vec3;

use super::math;
use super::rng::Rng;

/// The two emitter shapes the reference implements (docs §7.3). Unknown names
/// compile to [`Shape::Unsupported`] and emit nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    BoxRandom,
    SphereRandom,
    Unsupported,
}

const FLAG_ONE_PER_FRAME: u32 = 0x2;
const FLAG_PERIODIC: u32 = 0x4;

/// A compiled emitter with its live scheduling state.
#[derive(Clone, Debug)]
pub struct CompiledEmitter {
    shape: Shape,
    directions: Vec3,
    distancemin: Vec3,
    distancemax: Vec3,
    origin: Vec3,
    sign: [i32; 3],
    instantaneous: bool,
    speedmin: f32,
    speedmax: f32,
    rate: f32,
    controlpoint: usize,
    flags: u32,
    delay: f32,
    duration: f32,
    minperiodicdelay: f32,
    maxperiodicdelay: f32,
    minperiodicduration: f32,
    maxperiodicduration: f32,

    // ---- live state ----
    /// Fractional accumulator of pending spawns (SPEC §V5: no alloc).
    accumulator: f32,
    /// Whether the emitter is inside an active window this step.
    active: bool,
    /// Whether the current burst (`instantaneous`) has fired this window.
    burst_fired: bool,
    /// Absolute time the next periodic toggle happens (`FLAG_PERIODIC`).
    next_toggle: f32,
    /// Whether the periodic scheduler has been primed.
    primed: bool,
}

impl CompiledEmitter {
    /// Compile a scene [`Emitter`]. Its origin/directions are Y-flipped to the
    /// centered coordinate convention (docs §7.3).
    #[must_use]
    pub fn compile(e: &Emitter) -> Self {
        let shape = match e.name.as_str() {
            "boxrandom" => Shape::BoxRandom,
            "sphererandom" => Shape::SphereRandom,
            _ => Shape::Unsupported,
        };
        CompiledEmitter {
            shape,
            directions: math::flip_y(e.directions),
            distancemin: e.distancemin,
            distancemax: e.distancemax,
            origin: math::flip_y(e.origin),
            sign: e.sign,
            instantaneous: e.instantaneous != 0,
            speedmin: e.speedmin,
            speedmax: e.speedmax,
            rate: e.rate,
            controlpoint: e.controlpoint.max(0) as usize,
            flags: e.flags,
            delay: e.delay,
            duration: e.duration,
            minperiodicdelay: e.minperiodicdelay,
            maxperiodicdelay: e.maxperiodicdelay,
            minperiodicduration: e.minperiodicduration,
            maxperiodicduration: e.maxperiodicduration,
            accumulator: 0.0,
            active: false,
            burst_fired: false,
            next_toggle: 0.0,
            primed: false,
        }
    }

    /// Whether this emitter can ever spawn (a known shape).
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.shape != Shape::Unsupported
    }

    /// Advance scheduling by `dt` at absolute `time` and return how many
    /// particles to spawn this step. `rate_override` is `instanceOverride.rate`;
    /// `audio_factor` is `1 + 3·sampleAudio` (`1.0` headless).
    pub fn tick(&mut self, dt: f32, time: f32, rate_override: f32, audio_factor: f32, rng: &mut Rng) -> u32 {
        if self.shape == Shape::Unsupported {
            return 0;
        }

        let was_active = self.active;
        self.active = self.compute_active(time, rng);
        if self.active && !was_active {
            // Entered an active window: re-arm the instantaneous burst.
            self.burst_fired = false;
        }
        if !self.active {
            return 0;
        }

        if self.instantaneous {
            if self.burst_fired {
                return 0;
            }
            self.burst_fired = true;
            return (self.rate * rate_override).round().max(0.0) as u32;
        }

        self.accumulator += dt * self.rate * rate_override * audio_factor;
        let mut n = self.accumulator.floor().max(0.0);
        self.accumulator -= n;
        if self.flags & FLAG_ONE_PER_FRAME != 0 {
            n = n.min(1.0);
        }
        n as u32
    }

    /// Whether the emitter is active at `time` (delay/duration window, or the
    /// random periodic scheduler when `FLAG_PERIODIC` is set).
    fn compute_active(&mut self, time: f32, rng: &mut Rng) -> bool {
        if self.flags & FLAG_PERIODIC != 0 {
            if !self.primed {
                self.primed = true;
                // Start inactive; first toggle after a periodic delay + delay.
                self.active = false;
                self.next_toggle = self.delay + rng.range(self.minperiodicdelay, self.maxperiodicdelay);
                return false;
            }
            if time >= self.next_toggle {
                let now_active = !self.active;
                let span = if now_active {
                    rng.range(self.minperiodicduration, self.maxperiodicduration)
                } else {
                    rng.range(self.minperiodicdelay, self.maxperiodicdelay)
                };
                self.next_toggle = time + span.max(0.0);
                return now_active;
            }
            return self.active;
        }
        if time < self.delay {
            return false;
        }
        self.duration <= 0.0 || time <= self.delay + self.duration
    }

    /// Compute one spawn's system-local position and velocity. Called `n` times
    /// per step by the simulation, which then builds the particle and runs
    /// initializers over it.
    #[must_use]
    pub fn spawn(&self, control_points: &[Vec3], perspective: bool, rng: &mut Rng) -> (Vec3, Vec3) {
        let base = math::add(
            self.origin,
            control_points.get(self.controlpoint).copied().unwrap_or([0.0; 3]),
        );
        match self.shape {
            Shape::BoxRandom => {
                // Uniform per axis in [distancemin, distancemax] with random
                // sign, times `directions` (docs §7.3).
                let off = [
                    rng.range(self.distancemin[0], self.distancemax[0]) * rng.sign() * self.directions[0],
                    rng.range(self.distancemin[1], self.distancemax[1]) * rng.sign() * self.directions[1],
                    rng.range(self.distancemin[2], self.distancemax[2]) * rng.sign() * self.directions[2],
                ];
                (math::add(base, off), [0.0, 0.0, 0.0])
            }
            Shape::SphereRandom => {
                // Radius range from the X components (broadcastable field).
                // UNVERIFIED: the reference's exact radius source is not
                // documented; the X component matches the broadcast default.
                let rmin = self.distancemin[0];
                let rmax = self.distancemax[0];
                let u = rng.unit();
                let mut dir;
                let radius;
                if perspective {
                    // 3D shell, volume-uniform via cbrt (docs §7.3).
                    radius = (lerp(rmin.powi(3), rmax.powi(3), u)).cbrt();
                    let z = rng.range(-1.0, 1.0);
                    let phi = rng.range(0.0, std::f32::consts::TAU);
                    let s = (1.0 - z * z).max(0.0).sqrt();
                    dir = [s * phi.cos(), s * phi.sin(), z];
                } else {
                    // 2D annulus, area-uniform via sqrt (docs §7.3).
                    radius = lerp(rmin * rmin, rmax * rmax, u).sqrt();
                    let theta = rng.range(0.0, std::f32::consts::TAU);
                    dir = [theta.cos(), theta.sin(), 0.0];
                }
                // `sign` ivec3 forces per-axis sign (sphere only, docs §7.3).
                for (d, &s) in dir.iter_mut().zip(self.sign.iter()) {
                    if s != 0 {
                        *d = d.abs() * (s.signum() as f32);
                    }
                }
                let pos = math::add(base, math::mul(dir, radius));
                let vel = math::mul(dir, rng.range(self.speedmin, self.speedmax));
                (pos, vel)
            }
            Shape::Unsupported => (base, [0.0, 0.0, 0.0]),
        }
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
