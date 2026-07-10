//! Typed errors for the control socket (SPEC V9: no panics, callers decide
//! policy — the C++ engine logs bind failures and continues without a
//! socket, docs/compat-socket.md §1; the app crate can reproduce that by
//! logging this error and dropping it).

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Errors surfaced when creating the control socket.
#[derive(Debug, Error)]
pub enum IpcError {
    /// Binding the listener failed (path unwritable, directory missing,
    /// path too long, …).
    ///
    /// Divergence note (doc §1): the C++ engine `strncpy`-truncates paths
    /// ≥ 108 bytes and silently binds the truncated path; we surface the
    /// OS error instead of reproducing that footgun.
    #[error("failed to bind control socket at {path}: {source}")]
    Bind {
        /// The requested socket path, verbatim (no derivation — doc §1).
        path: PathBuf,
        /// Underlying OS error.
        source: io::Error,
    },

    /// Spawning the dedicated socket thread failed.
    #[error("failed to spawn the control-socket thread: {0}")]
    Spawn(#[source] io::Error),
}
