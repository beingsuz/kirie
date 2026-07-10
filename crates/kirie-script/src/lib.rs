//! `kirie-script` — Wallpaper Engine **SceneScript** (QuickJS-embedded JS
//! scripting) for kirie, implemented on [`rquickjs`].
//!
//! Behavior truth: `docs/scripting-api.md` (the C++ reference reversed +
//! corpus-verified). This crate reproduces that surface with the reference's
//! `DO-NOT-PORT` defects fixed (docs §13 "Fix" list): the vector methods use
//! real prototypes (docs §9.2), operand orders are corrected, `z`/`w` are read
//! on object returns (docs §5.1), and the timer canceller cancels its own timer
//! (docs §5.4).
//!
//! # Architecture (SPEC.md §V3)
//!
//! One QuickJS `Runtime` + `Context` per scene. The runtime is `!Send`, so it
//! lives on its own dedicated thread ([`world::World`]); the public
//! [`ScriptEngine`] handle drives it over a bounded `crossbeam-channel`.
//! **JS never touches engine memory.** Each tick the integrator marshals an
//! immutable [`HostFrame`] snapshot *in* (all `engine`/`thisScene`/`thisLayer`/
//! `input` reads are served from it) and reads typed [`SceneOp`]s + property
//! results *out* ([`TickOutput`]) — host calls are pure typed messages.
//!
//! # Property-script contract (docs §5.1)
//!
//! A property's inline script is an ES module exporting `update(value)`; its
//! return value is applied to that property each tick (surfaced in
//! [`TickOutput::property_results`]). `init(value)` runs once, deferred to the
//! first tick. `applyUserProperties(changed)` fires on user edits
//! ([`ScriptEngine::dispatch_user_property`]).
//!
//! # JS global surface
//!
//! `engine`, `input`, `thisScene`, `thisLayer`, `console`, `shared`,
//! `createScriptProperties`, `Vec2`/`Vec3`/`Vec4`, `Mat3`/`Mat4`,
//! `localStorage`, `MediaPlaybackEvent`, and the import-only modules
//! `WEMath`/`WEColor`/`WEVector`. See the crate `tests/` and `docs/scripting-api.md`
//! §12 for the implemented-vs-stubbed matrix.
//!
//! # SPEC.md §V9
//!
//! A malformed or throwing script surfaces as a typed [`ScriptError`] (the
//! script is disabled/logged) and never panics or crashes the engine.

#![forbid(unsafe_code)]

mod engine;
mod error;
mod frame;
mod value;
mod world;

pub use engine::{API_VERSION, ScriptEngine, TRANSLATOR_VERSION};
pub use error::ScriptError;
pub use frame::{AudioBuffers, CameraState, HostFrame, LayerState, LogLine, SceneOp, SceneState, TickOutput};
pub use value::ScriptValue;
