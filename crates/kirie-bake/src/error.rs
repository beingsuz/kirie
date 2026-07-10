//! Typed errors for the bake cache (SPEC.md §V9: no panic on malformed input —
//! including a corrupt on-disk bundle, which is hash-verified *and* structurally
//! validated on load).

use std::path::PathBuf;

use thiserror::Error;

/// Everything that can go wrong baking, loading, or reaping a bundle.
#[derive(Debug, Error)]
pub enum BakeError {
    /// A filesystem operation failed. Carries the offending path for context.
    #[error("io error at {path}: {source}")]
    Io {
        /// The path the operation targeted.
        path: PathBuf,
        /// The underlying `std::io` error.
        source: std::io::Error,
    },

    /// rkyv serialization of a [`crate::BakedBundle`] failed. Should not happen
    /// for well-formed content; surfaced rather than panicked (§V9).
    #[error("bundle serialization failed: {0}")]
    Serialize(String),

    /// The on-disk bundle failed structural validation (rkyv bytecheck) — it is
    /// corrupt or was written by an incompatible layout. Never a panic (§V9).
    #[error("corrupt bundle at {path}: {reason}")]
    Corrupt {
        /// The bundle file that failed validation.
        path: PathBuf,
        /// Human-readable validation diagnostic.
        reason: String,
    },

    /// The bundle's content checksum did not match its recorded digest — the
    /// payload was truncated or tampered after write. Treated as corruption.
    #[error("bundle checksum mismatch at {path} (expected {expected}, got {actual})")]
    ChecksumMismatch {
        /// The bundle file whose checksum failed.
        path: PathBuf,
        /// The digest recorded in the sidecar at write time.
        expected: String,
        /// The digest computed from the payload on load.
        actual: String,
    },

    /// A field the renderer needs could not be decoded from a validated bundle
    /// (e.g. the embedded scene JSON is not a valid [`kirie_scene::SceneModel`]).
    #[error("bundle field decode failed ({field}): {reason}")]
    Decode {
        /// Which logical field failed to decode.
        field: &'static str,
        /// Human-readable diagnostic.
        reason: String,
    },

    /// Setting up the filesystem watcher failed (background baker only).
    #[error("watcher error: {0}")]
    Watch(String),
}

impl BakeError {
    /// Build an [`BakeError::Io`] tagged with the path it happened at.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        BakeError::Io {
            path: path.into(),
            source,
        }
    }
}
