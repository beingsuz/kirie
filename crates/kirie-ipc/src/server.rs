//! The control-socket server thread (docs/compat-socket.md §1, §2, §5).
//!
//! One dedicated thread owns the `UnixListener` and serves connections
//! strictly in accept order — the same full serialization the C++ engine
//! gets from draining every pending connection in one render-thread poll
//! pass (doc §5) — but decoupled from rendering entirely (SPEC V4).

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{fs, thread};

use crossbeam_channel::{Sender, bounded};

use crate::command::{Request, parse_request};
use crate::error::IpcError;
use crate::event::{CommandOutcome, IpcEvent};
use crate::status::format_status;

/// Per-`read()` timeout, the C++ `SO_RCVTIMEO` of 50 ms
/// (doc §2, ControlSocket.cpp:52-53). A client that connects and sends
/// nothing holds the thread at most this long; its (empty) partial buffer is
/// then processed as the request.
const READ_TIMEOUT: Duration = Duration::from_millis(50);

/// Read chunk size, matching the C++ accumulation loop (doc §2 step 3).
const READ_CHUNK: usize = 1024;

/// Total per-connection read deadline. The C++ timeout is per-`read()` only,
/// so a client trickling ≥ 1 byte per 50 ms without ever sending `\n` stalls
/// the engine's render thread indefinitely (doc §5). Adaptation required by
/// the port: the read phase is capped so one stalled client cannot block
/// other clients for more than this long. Well-behaved clients (all daemon
/// scripts) send the whole line in one write and are unaffected.
const CONNECTION_DEADLINE: Duration = Duration::from_secs(2);

const RESP_PONG: &[u8] = b"pong\n";
const RESP_OK: &[u8] = b"ok\n";
const RESP_ERROR: &[u8] = b"error\n";
const RESP_UNKNOWN: &[u8] = b"unknown command\n";

/// The listening control socket. Owns the dedicated socket thread; dropping
/// (or [`ControlSocket::shutdown`]) stops the thread and unlinks the socket
/// file, like the C++ destructor on clean teardown (doc §1).
#[derive(Debug)]
pub struct ControlSocket {
    path: PathBuf,
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ControlSocket {
    /// Bind the control socket at exactly `path` (the `--control-socket`
    /// value is used verbatim; no derivation — doc §1) and spawn the socket
    /// thread.
    ///
    /// A pre-existing socket file is unlinked unconditionally first, exactly
    /// like the C++ engine (doc §1: callers must serialize engine launches
    /// externally; the daemon uses a per-monitor `flock`).
    ///
    /// Parsed requests are delivered on `events`; see [`IpcEvent`] for the
    /// reply contract. On bind failure the caller decides policy — the C++
    /// engine logs and continues without a socket (doc §1).
    pub fn bind(path: impl Into<PathBuf>, events: Sender<IpcEvent>) -> Result<Self, IpcError> {
        let path = path.into();
        // unlink(path) unconditionally, errors ignored (doc §1,
        // ControlSocket.cpp:20); a failure surfaces as AddrInUse from bind.
        if let Err(e) = fs::remove_file(&path)
            && e.kind() != ErrorKind::NotFound
        {
            tracing::debug!(path = %path.display(), error = %e, "stale socket unlink failed");
        }
        let listener = UnixListener::bind(&path).map_err(|source| IpcError::Bind {
            path: path.clone(),
            source,
        })?;
        // Backlog divergence (doc §1): C++ uses listen(fd, 8); std hardcodes
        // 128. Strictly more permissive — behavior beyond 8 pending
        // connections is explicitly unverified upstream (doc §10) — and the
        // workspace has no crate (socket2/libc) that could set it exactly.
        tracing::info!(path = %path.display(), "ControlSocket listening");
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread = thread::Builder::new()
            .name("kirie-ipc".into())
            .spawn({
                let shutdown = Arc::clone(&shutdown);
                move || serve(&listener, &events, shutdown.as_ref())
            })
            .map_err(IpcError::Spawn)?;
        Ok(Self {
            path,
            shutdown,
            thread: Some(thread),
        })
    }

    /// The socket path this server is bound to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stop the socket thread and unlink the socket file (doc §1 clean
    /// teardown). Idempotent; also invoked by `Drop`.
    pub fn shutdown(&mut self) {
        let Some(handle) = self.thread.take() else {
            return;
        };
        self.shutdown.store(true, Ordering::Release);
        // Wake the (blocking) accept with a throwaway connection. If the
        // path was stolen/unlinked externally this can fail; the thread then
        // stays parked in accept until process exit — leak it rather than
        // hang the caller. (C++ has the mirror-image flaw: its exception
        // exit path leaks the socket file, doc §1.)
        match UnixStream::connect(&self.path) {
            Ok(stream) => {
                drop(stream);
                let _ = handle.join();
            }
            Err(e) => {
                tracing::warn!(path = %self.path.display(), error = %e,
                    "could not wake control-socket thread; detaching it");
            }
        }
        let _ = fs::remove_file(&self.path);
    }
}

impl Drop for ControlSocket {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Accept loop: connections served strictly in accept order (doc §5).
fn serve(listener: &UnixListener, events: &Sender<IpcEvent>, shutdown: &AtomicBool) {
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                if shutdown.load(Ordering::Acquire) {
                    break; // wake-up connection (or late client during teardown)
                }
                handle_connection(stream, events);
            }
            Err(e) => {
                if shutdown.load(Ordering::Acquire) {
                    break;
                }
                tracing::warn!(error = %e, "control-socket accept failed");
                // Guard against a hot spin on persistent errors (e.g. EMFILE).
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// One connection = one request, one response, close (doc §2).
fn handle_connection(mut stream: UnixStream, events: &Sender<IpcEvent>) {
    if stream.set_read_timeout(Some(READ_TIMEOUT)).is_err() {
        // Without the timeout a half-open client could park the thread
        // forever; refuse the connection instead.
        return;
    }
    let deadline = Instant::now() + CONNECTION_DEADLINE;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; READ_CHUNK];
    loop {
        match stream.read(&mut chunk) {
            // EOF/half-close terminates the request (doc §2 step 3-4:
            // clients may terminate by newline OR by half-closing).
            Ok(0) => break,
            Ok(n) => {
                let has_newline = chunk[..n].contains(&b'\n');
                buf.extend_from_slice(&chunk[..n]);
                if has_newline {
                    break;
                }
            }
            // The 50 ms silence timeout: whatever accumulated is the request
            // (doc §2 step 3; live-verified `printf 'ping'` + EOF → pong).
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break, // reset/etc. — treat like EOF, as C++ read() ≤ 0
        }
        if Instant::now() >= deadline {
            break; // trickle-client cap; see CONNECTION_DEADLINE
        }
    }
    // Request = bytes up to (excluding) the FIRST '\n'; everything after is
    // discarded (doc §2 step 4). No '\n' ⇒ the whole buffer is the request.
    let line = match buf.iter().position(|&b| b == b'\n') {
        Some(i) => &buf[..i],
        None => &buf[..],
    };
    if let Some(response) = respond(line, events) {
        // Single write, failure logged and ignored (doc §5); the C++ single
        // unlooped write() is upgraded to write_all (doc §10 short-write
        // open item resolved conservatively).
        if let Err(e) = stream.write_all(&response) {
            tracing::debug!(error = %e, "control-socket response write failed");
        }
    }
    // Connection closed on drop; EOF signals end-of-response (doc §2 step 6).
}

/// Map one request line to its response bytes. `None` ⇒ close with zero
/// response bytes (empty request, doc §2 step 5 — or the app is gone, which
/// is exactly the protocol's dead-engine signal, doc §3).
fn respond(line: &[u8], events: &Sender<IpcEvent>) -> Option<Vec<u8>> {
    match parse_request(line) {
        Request::Empty => None,
        Request::Ping => Some(RESP_PONG.to_vec()),
        Request::Unknown => Some(RESP_UNKNOWN.to_vec()),
        Request::Rejected => Some(RESP_ERROR.to_vec()),
        Request::Status => {
            let (tx, rx) = bounded(1);
            events.send(IpcEvent::Status { reply: tx }).ok()?;
            let snapshot = rx.recv().ok()?;
            Some(format_status(&snapshot))
        }
        Request::GetProperties { screen } => {
            // kirie extension (docs/compat-socket.md §11): the app returns the
            // JSON schema body; we frame it with exactly one trailing newline.
            let (tx, rx) = bounded(1);
            events.send(IpcEvent::GetProperties { screen, reply: tx }).ok()?;
            let mut body = rx.recv().ok()?.into_bytes();
            body.push(b'\n');
            Some(body)
        }
        Request::Command(command) => {
            let fallible = command.is_fallible();
            let (tx, rx) = bounded(1);
            events.send(IpcEvent::Command { command, reply: tx }).ok()?;
            // Blocks until the app has applied the command — the same
            // synchronous semantics clients get from the C++ engine, where
            // blocking commands stall the reply for their full duration
            // (doc §5). Bounded only by the app's own progress.
            let outcome = rx.recv().ok()?;
            Some(match (fallible, outcome) {
                // ok\n unconditionally for speed/volume/mute/set/preload
                // (ControlSocket.cpp:100-132 per doc §4).
                (false, _) | (true, CommandOutcome::Ok) => RESP_OK.to_vec(),
                (true, CommandOutcome::Error) => RESP_ERROR.to_vec(),
            })
        }
    }
}
