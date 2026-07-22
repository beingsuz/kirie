//! Global pointer position via the Hyprland IPC socket (T26).
//!
//! A wallpaper layer surface has an empty input region — the compositor never
//! sends it `wl_pointer` events — so, like the reference's `WaylandMouseInput`,
//! the cursor is polled out-of-band from Hyprland's control socket
//! (`$XDG_RUNTIME_DIR/hypr/<HIS>/.socket.sock`, request `cursorpos`, reply
//! `"x, y"` in global logical coordinates). A background thread polls at
//! ~60 Hz into a lock-guarded slot the render thread reads per frame (SPEC
//! §V4: the render thread never blocks on the socket).
//!
//! On non-Hyprland compositors the poller never starts and the position stays
//! `None`; consumers fall back to the centered pointer they used before.

use std::io::{Read, Write};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Shared global cursor position (logical coordinates), `None` until the first
/// successful poll (or forever, off Hyprland).
#[derive(Clone, Default)]
pub struct PointerPoll {
    pos: Arc<RwLock<Option<(f64, f64)>>>,
}

impl PointerPoll {
    /// Start the poller if a Hyprland instance is reachable. Always returns a
    /// handle; it just stays `None` when there is nothing to poll.
    #[must_use]
    pub fn start() -> Self {
        let handle = PointerPoll::default();
        let Some(sock) = hypr_socket_path() else {
            return handle;
        };
        let slot = handle.pos.clone();
        // The engine runs for the process lifetime; the poller thread parks on
        // sleep and exits with the process (detached by design, like audio).
        let _ = std::thread::Builder::new()
            .name("kirie-pointer-poll".into())
            .spawn(move || {
                loop {
                    let read = query_cursorpos(&sock);
                    if let Ok(mut w) = slot.write() {
                        *w = read;
                    }
                    std::thread::sleep(Duration::from_millis(16));
                }
            });
        handle
    }

    /// The latest global cursor position, if known.
    #[must_use]
    pub fn get(&self) -> Option<(f64, f64)> {
        self.pos.read().ok().and_then(|g| *g)
    }
}

/// `$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock`.
fn hypr_socket_path() -> Option<std::path::PathBuf> {
    let run = std::env::var_os("XDG_RUNTIME_DIR")?;
    let his = std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE")?;
    let p = std::path::PathBuf::from(run)
        .join("hypr")
        .join(his)
        .join(".socket.sock");
    p.exists().then_some(p)
}

/// One `cursorpos` round-trip. Any failure ⇒ `None` (treated as unknown).
fn query_cursorpos(sock: &std::path::Path) -> Option<(f64, f64)> {
    let mut s = std::os::unix::net::UnixStream::connect(sock).ok()?;
    s.set_read_timeout(Some(Duration::from_millis(50))).ok()?;
    s.set_write_timeout(Some(Duration::from_millis(50))).ok()?;
    s.write_all(b"cursorpos").ok()?;
    let mut buf = String::new();
    s.read_to_string(&mut buf).ok()?;
    let (x, y) = buf.trim().split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}
