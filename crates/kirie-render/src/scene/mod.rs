//! The Wallpaper Engine **scene** renderer (docs/render-architecture.md).
//!
//! Renders a resolved [`kirie_scene::SceneModel`] — image layers, per-pass
//! wgpu pipelines built from [`kirie_shader`]-translated modules, the builtin
//! uniform set, and the effect FBO ping-pong chain — into a scene FBO, then
//! blits that FBO to the output surface exactly like the reference two-stage
//! frame (docs/render-architecture.md §2.5).
//!
//! Module map:
//! - [`blend`] — pass GL-state → wgpu blend/cull/depth (docs §8.1).
//! - [`matrix`] — column-major 4×4 camera/transform math (docs §7.1, §9).
//! - [`uniforms`] — the builtin set + std140 `_WEGlobals` packing (docs §8.3).
//! - [`plan`] — the per-image pass list + ping-pong FBO wiring (docs §7.1).
//! - [`pipeline`] — pipelines + bind-group layouts from translated shaders.
//! - [`texture`] — `.tex` upload, the default white texture, name resolution.
//! - [`extras`] — non-image object compositing: particles + text (docs §7.3-4).
//! - [`model`] — 3D MODEL object compositing: `.mdl` meshes (docs §7.2).
//! - [`fbo`] — the resize-stable FBO pool (docs §6; SPEC.md §V5).
//! - [`renderer`] — [`SceneRenderer`], the per-frame compositor.

pub mod blend;
pub mod error;
pub mod extras;
pub mod matrix;
pub mod plan;
pub mod uniforms;

pub mod fbo;
pub mod load;
pub mod model;
pub mod pipeline;
pub mod renderer;
pub mod scripting;
pub mod text;
pub mod texture;

pub use error::SceneError;
pub use load::{SceneLoadError, load_workshop_scene};
pub use renderer::{SceneOptions, SceneRenderer};
