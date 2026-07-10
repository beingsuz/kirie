//! The per-particle state record and the instance-override snapshot.
//!
//! Particle state mirrors the reference struct (docs/render-architecture.md
//! §7.3, `CParticle.h`): position / velocity / acceleration, rotation /
//! angular velocity, color / alpha / size, lifetime / age / frame, plus an
//! `initial.*` snapshot that fade/ramp operators multiply against each frame.

use kirie_scene::particle::InstanceOverride;
use kirie_scene::value::Vec3;

/// The immutable-per-particle spawn snapshot that ramp/fade operators read
/// (docs/render-architecture.md §7.3: `alphafade`/`sizechange`/`colorchange`
/// multiply `initial.*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Initial {
    /// Spawn color (post-initializer).
    pub color: Vec3,
    /// Spawn alpha (post-initializer).
    pub alpha: f32,
    /// Spawn size (post-initializer).
    pub size: f32,
}

/// One live particle (system-local coordinates, relative to the system origin
/// — the reference stores positions relative to the origin, §7.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// Position relative to the system origin.
    pub position: Vec3,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Linear acceleration (reserved; operators integrate velocity directly).
    pub acceleration: Vec3,
    /// Euler rotation (radians); Z is the sprite billboard spin.
    pub rotation: Vec3,
    /// Angular velocity (radians/s).
    pub angular_velocity: Vec3,
    /// Current color (rgb).
    pub color: Vec3,
    /// Current alpha.
    pub alpha: f32,
    /// Current size (full quad edge length).
    pub size: f32,
    /// Total lifetime in seconds; the particle dies when `age >= lifetime`.
    pub lifetime: f32,
    /// Seconds since spawn.
    pub age: f32,
    /// Current spritesheet frame index (float; `0.0` when no sheet).
    pub frame: f32,
    /// Spawn snapshot that fade/ramp operators multiply against.
    pub initial: Initial,
    /// A per-particle random seed — the basis for "randomized once per particle"
    /// operator state (oscillators) without storing per-operator fields on the
    /// particle (SPEC §V5).
    pub seed: u32,
}

impl Particle {
    /// The normalized position in the particle's lifetime, in `[0, 1]`
    /// (`age / lifetime`; `0` when lifetime is non-positive).
    #[inline]
    #[must_use]
    pub fn life_pos(&self) -> f32 {
        if self.lifetime > 0.0 {
            (self.age / self.lifetime).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// A snapshot of the per-scene-object `instanceoverride` multipliers, resolved
/// to plain floats/vectors (docs/format-scene-json.md §14.7). Applied
/// multiplicatively at spawn and by movement/turbulence operators
/// (docs/render-architecture.md §7.3 `instanceOverride`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overrides {
    /// Whether the system renders at all.
    pub enabled: bool,
    /// Spawn-alpha multiplier.
    pub alpha: f32,
    /// Spawn-size multiplier.
    pub size: f32,
    /// Spawn-lifetime multiplier.
    pub lifetime: f32,
    /// Emission-rate multiplier.
    pub rate: f32,
    /// Force/gravity/velocity multiplier applied by operators.
    pub speed: f32,
    /// Pool-size multiplier.
    pub count: f32,
    /// Replaces the spawn color.
    pub color: Vec3,
    /// Multiplies the spawn / initializer color.
    pub colorn: Vec3,
}

impl Overrides {
    /// Snapshot the resolved override values (`.value` after §3.2 resolution).
    #[must_use]
    pub fn from_scene(o: &InstanceOverride) -> Self {
        Overrides {
            enabled: o.enabled.value,
            alpha: o.alpha.value,
            size: o.size.value,
            lifetime: o.lifetime.value,
            rate: o.rate.value,
            speed: o.speed.value,
            count: o.count.value,
            color: o.color.value,
            colorn: o.colorn.value,
        }
    }
}

impl Default for Overrides {
    fn default() -> Self {
        Overrides {
            enabled: true,
            alpha: 1.0,
            size: 1.0,
            lifetime: 1.0,
            rate: 1.0,
            speed: 1.0,
            count: 1.0,
            color: [1.0, 1.0, 1.0],
            colorn: [1.0, 1.0, 1.0],
        }
    }
}

/// GPU per-particle instance for the instanced-quad sprite renderer
/// (docs/render-architecture.md §7.3 sprite vertex format, adapted to
/// instancing: the shader expands the quad from the center, so we upload one
/// record per particle instead of four corners). std140-friendly 64-byte
/// stride, `bytemuck`-safe.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteInstance {
    /// Particle center in system-local space; `.w` = size (quad half-extent ×2).
    pub position_size: [f32; 4],
    /// rgb + alpha.
    pub color: [f32; 4],
    /// rotation.xyz + normalized spritesheet frame `(frame / frames)`.
    pub rotation_frame: [f32; 4],
    /// velocity.xyz (for rope/stretch renderers) + spare.
    pub velocity: [f32; 4],
}
