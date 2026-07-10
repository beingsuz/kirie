//! Typed scene-parse errors (docs/format-scene-json.md §17; SPEC.md §V9).

use thiserror::Error;

use crate::value::VecError;

/// An error loading or parsing a `scene.json` (docs/format-scene-json.md §4/§6).
#[derive(Debug, Error, PartialEq)]
pub enum SceneError {
    /// The bytes are not valid JSON (docs/format-scene-json.md §1: strict JSON).
    #[error("invalid JSON: {0}")]
    Json(String),
    /// The scene root is not a JSON object.
    #[error("scene.json root is not a JSON object")]
    NotAnObject,
    /// A required top-level section is absent (docs/format-scene-json.md §4:
    /// `camera`, `general`, `objects` are all required).
    #[error("required section `{0}` missing")]
    MissingSection(&'static str),
    /// A required `camera` vector field is absent (docs/format-scene-json.md
    /// §6.1: `center`/`eye`/`up` are required vec3 strings).
    #[error("camera field `{0}` missing")]
    MissingCameraField(&'static str),
    /// A `camera` vector field failed to parse (docs/format-scene-json.md §6.1).
    #[error("camera field `{field}`: {source}")]
    CameraVec {
        /// Which camera field failed.
        field: &'static str,
        /// The underlying vector error.
        source: VecError,
    },
}
