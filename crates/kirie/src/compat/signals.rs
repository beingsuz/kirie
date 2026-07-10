//! Graceful `SIGTERM`/`SIGINT` handling so the control socket file is unlinked
//! on signal-driven shutdown, matching the clean-exit path.
//!
//! The daemon stops a running engine with `pkill -f -- "--screen-root <mon> "`
//! (default `SIGTERM`) before removing the socket itself
//! (`~/.config/hypr/wallpaper-daemon/wallpaperengine.sh`). With no handler the
//! process dies on the default `SIGTERM` disposition **without** running
//! [`kirie_ipc::ControlSocket`]'s `Drop`, so the `lwe-<mon>.sock` file is left
//! stale (a lingering socket path clients then fail to connect to). On the
//! normal exit path [`crate::compat::run`] drops the socket, which unlinks the
//! file (docs/compat-socket.md §1 clean teardown); this module makes the signal
//! path do the same.
//!
//! Implementation: a dedicated thread built on [`signal_hook`] (the crate
//! forbids `unsafe`, so a hand-rolled `sigaction` is out). The thread waits on
//! the self-pipe [`Signals`] iterator — its body therefore runs in an ordinary
//! thread context, not an async-signal context, so [`std::fs::remove_file`] is
//! safe to call directly. After unlinking it re-raises the signal with the
//! default disposition so the process still terminates *as if* killed by the
//! signal (same externally observable exit as the no-handler case, only now the
//! socket is gone first). If installing the handler fails, shutdown is simply
//! left to the OS default (V9: never a panic).

use std::path::PathBuf;

use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use signal_hook::low_level;

/// Install the `SIGTERM`/`SIGINT` cleanup handler on a dedicated thread.
///
/// `socket_path` is the control-socket file to unlink on shutdown (`None` when
/// no `--control-socket` was given — the handler then just re-raises so a
/// socket-less run still terminates on the signal without the default handler
/// being suppressed by nothing at all; installing it is harmless and keeps the
/// `Ctrl-C` path identical between the two cases).
pub fn install_cleanup(socket_path: Option<PathBuf>) {
    let mut signals = match Signals::new([SIGTERM, SIGINT]) {
        Ok(s) => s,
        Err(err) => {
            // No handler ⇒ default disposition still terminates the process; the
            // daemon's own `rm -f "$sock"` remains the backstop (V9).
            tracing::warn!(%err, "could not install SIGTERM handler; socket cleanup on signal disabled");
            return;
        }
    };

    let spawn = std::thread::Builder::new()
        .name("kirie-signals".into())
        .spawn(move || {
            // `forever()` blocks until a registered signal arrives; the first one
            // wins and we tear down.
            if let Some(signal) = signals.forever().next() {
                if let Some(path) = &socket_path {
                    match std::fs::remove_file(path) {
                        Ok(()) => tracing::info!(path = %path.display(), signal, "signal received; control socket unlinked"),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => tracing::warn!(path = %path.display(), error = %e, "failed to unlink control socket on signal"),
                    }
                } else {
                    tracing::info!(signal, "signal received; shutting down");
                }
                // Re-raise with the default disposition so the process exits as
                // if killed by the signal (correct 128+signo status). Falls back
                // to a bare libc exit if emulation cannot run.
                if low_level::emulate_default_handler(signal).is_err() {
                    low_level::exit(128 + signal);
                }
                // `emulate_default_handler` for a terminating signal does not
                // return; the explicit exit is a belt-and-braces guarantee.
                low_level::exit(128 + signal);
            }
        });

    if let Err(err) = spawn {
        tracing::warn!(%err, "could not spawn signal-handler thread; socket cleanup on signal disabled");
    }
}
