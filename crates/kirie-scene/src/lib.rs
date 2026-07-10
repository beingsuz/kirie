//! `kirie-scene` — Wallpaper Engine `scene.json` object model + resolution.
//!
//! Spec: `docs/format-scene-json.md` (the byte/behavior truth) and
//! `docs/format-project-json.md` §4/§6 (user-property binding). Parsing mirrors
//! the C++ reference reader exactly: null ≡ absent, loose scalar coercion,
//! per-field defaults, and tolerant degradation of malformed collections
//! (docs/format-scene-json.md §17; SPEC.md §V9 — typed errors, no panics).
//!
//! Two layers:
//! - [`Scene`] — the parsed, *unresolved* graph. Every leaf field is a
//!   [`UserSetting`] that may bind to a project property (§3) or a SceneScript.
//! - [`SceneModel`] — the resolved, immutable snapshot the renderer consumes
//!   (SPEC.md §V3). [`SceneModel::resolve`] collapses property bindings against
//!   a [`PropertyBag`]; [`SceneModel::load_assets`] loads the referenced
//!   material / effect / model / particle files.
//!
//! Every model type is `Clone + Serialize + Deserialize` for cross-thread
//! snapshots and the bake cache; the serde form round-trips
//! (SPEC.md §V13, NaN excepted).

#![forbid(unsafe_code)]

pub mod error;
pub mod material;
pub mod object;
pub mod particle;
pub mod property;
pub mod resolve;
pub mod scene;
pub mod user;
pub mod value;

pub use error::SceneError;
pub use property::{PropertyBag, PropertyValue};
pub use resolve::{AssetProblem, AssetSource, SceneModel};
pub use scene::{Camera, General, Projection, Scene};
pub use user::UserSetting;
pub use value::{Color, DynamicValue, Vec2, Vec3};
