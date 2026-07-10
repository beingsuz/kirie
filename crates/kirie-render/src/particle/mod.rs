//! CPU particle simulation + instanced-quad rendering
//! (docs/render-architecture.md §7.3, `CParticle`).
//!
//! A [`ParticleSim`] owns one particle system's pooled state and advances it on
//! the CPU: emitters spawn into spare pool capacity, initializers set spawn
//! state, operators integrate each frame, dead particles are compacted in place
//! (no per-frame allocation — SPEC §V5). [`ParticleRenderer`] draws the sim's
//! [`SpriteInstance`]s as instanced billboards with the material's blend mode.
//!
//! The pieces are decoupled: the simulation has no GPU dependency (unit-testable
//! headless), and the renderer takes only instance data + a matrix. The scene
//! renderer skips particle objects today and exposes no hook, so this is a
//! standalone surface the integrator wires in later (see [`render`]).
//!
//! Semantics — emitter shapes, every initializer and operator kind, and their
//! parameter names/defaults — follow the render-architecture spec tables
//! exactly (SPEC §V10). Where the reference's exact numeric basis is not part of
//! any format (RNG stream, curl-noise field), we use a deterministic stand-in
//! and mark it `UNVERIFIED`; distributions and update math match.

pub mod emitter;
pub mod initializer;
mod math;
pub mod noise;
pub mod operator;
mod param;
mod render;
mod rng;
mod state;
mod system;

pub use emitter::CompiledEmitter;
pub use initializer::{Initializer, SpawnCtx};
pub use operator::{Operator, StepCtx};
pub use render::ParticleRenderer;
pub use rng::Rng;
pub use state::{Initial, Overrides, Particle, SpriteInstance};
pub use system::{FrameMode, MAX_DT, ParticleSim, SimConfig, SpriteSheet};
