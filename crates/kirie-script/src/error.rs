//! Typed errors (SPEC.md §V9: a malformed or throwing script must surface as a
//! typed error — script disabled and logged — never a panic or engine crash).

use thiserror::Error;

/// An error from the SceneScript engine.
#[derive(Clone, Debug, Error)]
pub enum ScriptError {
    /// A property script failed to compile or its top-level body threw. The
    /// script is dropped (never registered); the engine keeps running.
    #[error("script {key:?} failed to load: {message}")]
    Load {
        /// The module key (`<prop>_<objectId>`), used as the QuickJS filename.
        key: String,
        /// The JS exception text (`toString` + stack) or QuickJS error.
        message: String,
    },

    /// A runtime exception escaped an `init`/`update`/event handler. The write
    /// back is skipped for that tick; the script stays loaded.
    #[error("script {key:?} threw in {phase}: {message}")]
    Runtime {
        /// The module key.
        key: String,
        /// Which lifecycle call threw (`init`, `update`, `applyUserProperties`, …).
        phase: &'static str,
        /// The JS exception text.
        message: String,
    },

    /// The dedicated script thread has stopped (its `Runtime` was dropped) so no
    /// further commands can be serviced.
    #[error("script engine thread is not running")]
    ThreadGone,

    /// An internal QuickJS/host-marshalling error not attributable to script
    /// source (e.g. the embedded builtins failed to evaluate — a bug, not user
    /// input).
    #[error("internal script error: {0}")]
    Internal(String),
}
