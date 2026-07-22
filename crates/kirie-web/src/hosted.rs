//! Out-of-process web backend: the Chromium runtime lives in a spawned
//! `kirie-webhost` child, frames arrive through shared memory, and dropping
//! the backend kills the child — the kernel then reclaims *everything*
//! (threads, zygotes, V8 heaps) deterministically. In-process CEF could not
//! guarantee that: `cef_shutdown` returns with Chromium threads still alive,
//! leaving hundreds of MB resident after a web→scene switch. As a bonus the
//! engine binary no longer links `libcef.so` at all, so scene/video-only runs
//! never map the browser stack.
//!
//! ## Protocol (engine ↔ child)
//!
//! * Child stdout, line-based status:
//!   `shm <path> <bytes>` announces the frame buffer (a `memfd` republished
//!   via `/proc/<pid>/fd/<fd>`, same-uid open); `ready` after the browser is
//!   up. Anything else is ignored (forward-compatible).
//! * Child stdin, line-based commands: `resize <w> <h>`, `pointer <x> <y>
//!   <down>`, `mute <0|1>`, `props <single-line-json>`, `quit`.
//! * Frame buffer, seqlock layout: `[seq u64][w u32][h u32][fmt u32][pad u32]`
//!   then pixels. The writer increments `seq` to odd before writing and to
//!   even after; a reader retries until it sees a stable even value.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::{Duration, Instant};

use crate::backend::{FrameBuffer, PixelFormat, PointerState, WebBackend, WebError, WebFrameRef, WebSize};

/// Seqlock header size (see module docs).
pub const SHM_HEADER: usize = 24;
/// Frame capacity: up to 4096×2304 BGRA — pages materialize only when written
/// (memfd is sparse), so the virtual size costs nothing on smaller outputs.
pub const SHM_PIXELS: usize = 4096 * 2304 * 4;

/// Locate the `kirie-webhost` binary: `KIRIE_WEBHOST` override, else beside
/// the current executable (the packed runtime ships it next to the engine).
fn webhost_path() -> Result<std::path::PathBuf, WebError> {
    if let Some(p) = std::env::var_os("KIRIE_WEBHOST") {
        return Ok(std::path::PathBuf::from(p));
    }
    let exe = std::env::current_exe().map_err(|_| WebError::Init("current_exe".into()))?;
    let dir = exe.parent().ok_or_else(|| WebError::Init("exe dir".into()))?;
    let candidate = dir.join("kirie-webhost");
    if candidate.is_file() {
        Ok(candidate)
    } else {
        Err(WebError::Init("kirie-webhost binary not found".into()))
    }
}

/// The out-of-process backend handle. `Send`: it holds no browser objects,
/// just a child process, a pipe and a read-only mapping.
pub struct HostedBackend {
    child: Child,
    stdin: ChildStdin,
    /// Read-only mapping of the child's frame memfd.
    shm: Box<dyn AsRef<[u8]> + Send + Sync>,
    /// Last consumed seqlock value (even = stable).
    last_seq: u64,
    /// Latest copied-out frame, reused across ticks (no per-frame alloc once
    /// the size settles).
    cached: Option<FrameBuffer>,
    /// Lines from the child's stdout (drained non-blocking on tick).
    status_rx: Receiver<String>,
}

impl HostedBackend {
    fn send_line(&mut self, line: &str) {
        // A dead child means the wallpaper is being torn down anyway.
        let _ = writeln!(self.stdin, "{line}");
        let _ = self.stdin.flush();
    }

    /// Seqlock read: copy the latest stable frame out of the shm, if newer
    /// than the last consumed one.
    fn poll_frame(&mut self) {
        let shm = (*self.shm).as_ref();
        if shm.len() < SHM_HEADER {
            return;
        }
        for _ in 0..3 {
            let seq0 = u64::from_le_bytes(shm[0..8].try_into().unwrap_or_default());
            if seq0 % 2 != 0 || seq0 == self.last_seq {
                // Mid-write: the writer finishes within microseconds — catch
                // it next tick rather than spinning. Equal: nothing new.
                return;
            }
            let w = u32::from_le_bytes(shm[8..12].try_into().unwrap_or_default());
            let h = u32::from_le_bytes(shm[12..16].try_into().unwrap_or_default());
            let len = (w as usize) * (h as usize) * 4;
            if w == 0 || h == 0 || SHM_HEADER + len > shm.len() {
                return;
            }
            let pixels = &shm[SHM_HEADER..SHM_HEADER + len];
            let buf = match self.cached.as_mut() {
                Some(b) => {
                    b.data.clear();
                    b.data.extend_from_slice(pixels);
                    b.width = w;
                    b.height = h;
                    b.format = PixelFormat::Bgra8;
                    None
                }
                None => Some(FrameBuffer {
                    data: pixels.to_vec(),
                    width: w,
                    height: h,
                    format: PixelFormat::Bgra8,
                }),
            };
            // Confirm the frame was stable across the copy.
            let seq1 = u64::from_le_bytes(shm[0..8].try_into().unwrap_or_default());
            if seq1 == seq0 {
                if let Some(b) = buf {
                    self.cached = Some(b);
                }
                self.last_seq = seq0;
                return;
            }
            // Torn read — retry with the newer frame.
            if let Some(b) = buf {
                self.cached = Some(b);
            }
        }
    }
}

impl WebBackend for HostedBackend {
    fn new(url: &str, size: WebSize) -> Result<Self, WebError> {
        let host = webhost_path()?;
        let mut child = Command::new(&host)
            .arg("--url")
            .arg(url)
            .arg("--width")
            .arg(size.width.to_string())
            .arg("--height")
            .arg(size.height.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr inherits → the child's tracing lands in the engine log.
            .spawn()
            .map_err(|_| WebError::Init("webhost spawn".into()))?;
        let stdin = child.stdin.take().ok_or_else(|| WebError::Init("webhost pipes".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| WebError::Init("webhost pipes".into()))?;

        // Status reader thread: forwards stdout lines over a channel so `new`
        // can wait for the shm announcement and `tick` can drain the rest
        // without blocking.
        let (tx, status_rx) = channel();
        std::thread::Builder::new()
            .name("kirie-webhost-io".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            })
            .map_err(|_| WebError::Init("webhost spawn".into()))?;

        // Wait for the frame-buffer announcement (browser init dominates).
        let deadline = Instant::now() + Duration::from_secs(20);
        let shm = loop {
            match status_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(line) => {
                    let mut parts = line.split_whitespace();
                    if parts.next() == Some("shm")
                        && let Some(path) = parts.next()
                        && let Ok(map) = kirie_bake::map_readonly(std::path::Path::new(path))
                    {
                        break map;
                    }
                }
                Err(_) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(WebError::Init("webhost shm announcement timeout".into()));
                    }
                }
            }
        };

        tracing::info!(host = %host.display(), pid = child.id(), "web host process started");
        Ok(Self {
            child,
            stdin,
            shm,
            last_seq: 0,
            cached: None,
            status_rx,
        })
    }

    fn tick(&mut self, _dt: f32) {
        // Drain child status lines (ignored today; keeps the pipe from
        // filling) and pick up the newest stable frame.
        loop {
            match self.status_rx.try_recv() {
                Ok(_) => {}
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        self.poll_frame();
    }

    fn latest_frame(&self) -> Option<WebFrameRef<'_>> {
        self.cached.as_ref().map(|f| WebFrameRef {
            data: &f.data,
            width: f.width,
            height: f.height,
            format: f.format,
        })
    }

    fn resize(&mut self, size: WebSize) {
        let s = size.clamped();
        self.send_line(&format!("resize {} {}", s.width, s.height));
    }

    fn send_pointer(&mut self, pointer: PointerState) {
        self.send_line(&format!(
            "pointer {} {} {} {}",
            pointer.x,
            pointer.y,
            u8::from(pointer.left),
            u8::from(pointer.right)
        ));
    }

    fn set_muted(&mut self, muted: bool) {
        self.send_line(&format!("mute {}", u8::from(muted)));
    }

    fn apply_properties(&mut self, json: &str) {
        // The batch is single-line JSON by construction (serde output).
        if !json.contains('\n') {
            self.send_line(&format!("props {json}"));
        }
    }

    fn shutdown(&mut self) {
        // Ask nicely, then make it certain: process death is the whole point
        // of this backend — the kernel reclaims every page and thread.
        self.send_line("quit");
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
        tracing::info!("web host process stopped; browser runtime fully reclaimed");
    }
}

impl Drop for HostedBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}
