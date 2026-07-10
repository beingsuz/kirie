//! [`ParticleSim`] — the CPU particle simulation for one particle system
//! (docs/render-architecture.md §7.3, `CParticle::update`).
//!
//! The update order matches the reference: run emitters (spawn), age particles,
//! run operators, compute the spritesheet frame, then order-preservingly
//! compact dead particles (index 0 stays the oldest — the rope renderer relies
//! on it). `dt` is capped at 0.1 s.
//!
//! SPEC §V5: the particle pool is allocated once at construction (`maxcount ×
//! instanceOverride.count`); steady-state stepping pushes into spare capacity
//! and compacts in place — no per-frame heap allocation. SPEC §V1: no globals;
//! all state is owned by the `ParticleSim`.

use kirie_scene::particle::{InstanceOverride, ParticleSystem};
use kirie_scene::value::Vec3;

use super::emitter::CompiledEmitter;
use super::initializer::{Initializer, SpawnCtx};
use super::operator::{Operator, StepCtx};
use super::rng::Rng;
use super::state::{Initial, Overrides, Particle, SpriteInstance};

/// The maximum `dt` a single update advances (docs §7.3: `g_Time` delta capped
/// at 0.1 s so a stalled frame cannot teleport particles).
pub const MAX_DT: f32 = 0.1;

/// The default particle size before any `sizerandom` initializer (docs §7.3:
/// "size base 20").
const BASE_SIZE: f32 = 20.0;

/// Hard cap on the pool so a malformed `maxcount` cannot request a huge
/// allocation (SPEC §V9: never trust input sizing).
const MAX_POOL: usize = 1_000_000;

/// Spritesheet frame-advance mode (docs §7.3: `randomframe`, `once`, else loop).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FrameMode {
    /// Advance and wrap (`frame mod frames`).
    #[default]
    Loop,
    /// Advance and clamp at the last frame.
    Once,
    /// A frozen random frame chosen per particle.
    RandomFrame,
}

/// Spritesheet timing for frame computation (from the material texture's
/// spritesheet grid; supplied by the integrator since texture decoding lives
/// elsewhere).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteSheet {
    /// Total frames in the sheet (`>= 1`).
    pub frames: u32,
    /// Seconds per frame.
    pub frame_duration: f32,
    /// Frame-advance mode.
    pub mode: FrameMode,
}

/// Per-simulation configuration.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimConfig {
    /// RNG seed (deterministic runs / tests).
    pub seed: u64,
    /// Optional spritesheet timing (none → all particles stay on frame 0).
    pub sheet: Option<SpriteSheet>,
}

/// The CPU simulation for a single particle system instance.
pub struct ParticleSim {
    particles: Vec<Particle>,
    capacity: usize,
    emitters: Vec<CompiledEmitter>,
    initializers: Vec<Initializer>,
    operators: Vec<Operator>,
    control_points: Vec<Vec3>,
    overrides: Overrides,
    perspective: bool,
    sequence_multiplier: f32,
    sheet: Option<SpriteSheet>,
    rng: Rng,
    time: f32,
    next_seed: u32,
    total_spawned: u64,
}

impl ParticleSim {
    /// Build a simulation from a resolved particle system + its scene-object
    /// `instanceoverride`.
    #[must_use]
    pub fn new(system: &ParticleSystem, overrides: &InstanceOverride, config: SimConfig) -> Self {
        let ov = Overrides::from_scene(overrides);
        let capacity = pool_size(system.maxcount, ov.count);

        let emitters: Vec<CompiledEmitter> = system.emitters.iter().map(CompiledEmitter::compile).collect();
        let initializers: Vec<Initializer> = system.initializers.iter().map(Initializer::compile).collect();

        let mut rng = Rng::new(config.seed);
        let operators: Vec<Operator> = system
            .operators
            .iter()
            .enumerate()
            .map(|(i, s)| Operator::compile(s, 0x9E37_79B9u32.wrapping_mul(i as u32 + 1), &mut rng))
            .collect();

        // Control points (system-local; mouse-linked ones would update per
        // frame — static without a pointer). At least CP0 at the origin.
        let mut control_points: Vec<Vec3> = system
            .controlpoints
            .iter()
            .map(|cp| super::math::flip_y(cp.offset))
            .collect();
        if control_points.is_empty() {
            control_points.push([0.0, 0.0, 0.0]);
        }

        let perspective = system.flags & 0x4 != 0;

        ParticleSim {
            particles: Vec::with_capacity(capacity),
            capacity,
            emitters,
            initializers,
            operators,
            control_points,
            overrides: ov,
            perspective,
            sequence_multiplier: system.sequencemultiplier,
            sheet: config.sheet,
            rng,
            time: 0.0,
            next_seed: 0x1234_5678,
            total_spawned: 0,
        }
    }

    /// Advance the simulation by `dt` seconds (capped at [`MAX_DT`]).
    pub fn update(&mut self, dt: f32) {
        if !self.overrides.enabled {
            return;
        }
        let dt = dt.clamp(0.0, MAX_DT);
        self.time += dt;

        self.run_emitters(dt);
        self.age_particles(dt);
        self.run_operators(dt);
        self.compute_frames();
        self.compact();
    }

    fn run_emitters(&mut self, dt: f32) {
        for e_idx in 0..self.emitters.len() {
            let n = self.emitters[e_idx].tick(dt, self.time, self.overrides.rate, 1.0, &mut self.rng);
            for _ in 0..n {
                if self.particles.len() >= self.capacity {
                    break;
                }
                let (position, velocity) =
                    self.emitters[e_idx].spawn(&self.control_points, self.perspective, &mut self.rng);
                let mut pt = self.new_particle(position, velocity);
                let spawn_ctx = SpawnCtx {
                    overrides: &self.overrides,
                    perspective: self.perspective,
                    control_points: &self.control_points,
                };
                for init in &mut self.initializers {
                    init.apply(&mut pt, &spawn_ctx, &mut self.rng);
                }
                self.particles.push(pt);
                self.total_spawned += 1;
            }
        }
    }

    /// A fresh particle before initializers run (docs §7.3: color = colorn
    /// override; alpha/size/lifetime = 1 × override with size base 20).
    fn new_particle(&mut self, position: Vec3, velocity: Vec3) -> Particle {
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(0x9E37_79B9);
        let color = self.overrides.colorn;
        let alpha = self.overrides.alpha;
        let size = BASE_SIZE * self.overrides.size;
        let lifetime = self.overrides.lifetime;
        Particle {
            position,
            velocity,
            acceleration: [0.0; 3],
            rotation: [0.0; 3],
            angular_velocity: [0.0; 3],
            color,
            alpha,
            size,
            lifetime,
            age: 0.0,
            frame: 0.0,
            initial: Initial { color, alpha, size },
            seed,
        }
    }

    fn age_particles(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.age += dt;
        }
    }

    fn run_operators(&mut self, dt: f32) {
        let ctx = StepCtx {
            dt,
            time: self.time,
            overrides: &self.overrides,
            control_points: &self.control_points,
        };
        for op in &self.operators {
            for p in &mut self.particles {
                op.apply(p, &ctx);
            }
        }
    }

    fn compute_frames(&mut self) {
        let Some(sheet) = self.sheet else { return };
        let frames = sheet.frames.max(1);
        if frames == 1 || sheet.frame_duration <= 0.0 {
            return;
        }
        let fps = self.sequence_multiplier / sheet.frame_duration;
        for p in &mut self.particles {
            let raw = p.age * fps;
            p.frame = match sheet.mode {
                FrameMode::RandomFrame => {
                    let mut r = super::rng::derived(p.seed, 0xF00D);
                    (r.unit() * frames as f32).floor().min((frames - 1) as f32)
                }
                FrameMode::Once => raw.floor().min((frames - 1) as f32),
                FrameMode::Loop => raw.floor().rem_euclid(frames as f32),
            };
        }
    }

    /// Order-preserving compaction: drop particles whose age reached their
    /// lifetime, keeping the oldest at index 0 (docs §7.3). `Vec::retain` keeps
    /// order and does not allocate.
    fn compact(&mut self) {
        self.particles.retain(|p| p.lifetime > 0.0 && p.age < p.lifetime);
    }

    /// The live particles, oldest first.
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// The number of live particles.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.particles.len()
    }

    /// The pool capacity (`maxcount × instanceOverride.count`, clamped).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total particles spawned over the sim's lifetime (diagnostics).
    #[must_use]
    pub fn total_spawned(&self) -> u64 {
        self.total_spawned
    }

    /// Whether any emitter has a shape the reference implements. A system with
    /// only unsupported emitters can never spawn (docs §7.3).
    #[must_use]
    pub fn has_supported_emitter(&self) -> bool {
        self.emitters.iter().any(CompiledEmitter::is_supported)
    }

    /// Elapsed simulated time.
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Fill `out` with one [`SpriteInstance`] per live particle for the
    /// instanced-quad renderer. `out` is cleared and refilled in place so a
    /// warm buffer never reallocates (SPEC §V5).
    pub fn write_sprites(&self, out: &mut Vec<SpriteInstance>) {
        out.clear();
        let frames = self.sheet.map_or(1u32, |s| s.frames.max(1));
        for p in &self.particles {
            let norm_frame = if frames > 1 { p.frame / frames as f32 } else { 0.0 };
            out.push(SpriteInstance {
                position_size: [p.position[0], p.position[1], p.position[2], p.size],
                color: [p.color[0], p.color[1], p.color[2], p.alpha],
                rotation_frame: [p.rotation[0], p.rotation[1], p.rotation[2], norm_frame],
                velocity: [p.velocity[0], p.velocity[1], p.velocity[2], 0.0],
            });
        }
    }
}

/// The pool size: `maxcount × instanceOverride.count`, at least 1, clamped to
/// [`MAX_POOL`] (docs §7.3 pool; SPEC §V9 malformed-size guard).
#[must_use]
fn pool_size(maxcount: u32, count_override: f32) -> usize {
    let raw = (maxcount as f32 * count_override.max(0.0)).ceil();
    (raw as usize).clamp(1, MAX_POOL)
}
