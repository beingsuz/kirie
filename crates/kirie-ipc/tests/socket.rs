//! Full command-matrix integration tests over a real unix socket
//! (docs/compat-socket.md; expected bytes cross-checked against
//! fixtures/socket-live-capture.txt where the live capture covers them).

use std::io::{ErrorKind, Read, Write};
use std::net::Shutdown;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{env, fs, process, thread};

use crossbeam_channel::{Receiver, unbounded};
use kirie_ipc::{
    ClampMode, Command, CommandOutcome, ControlSocket, IpcEvent, ScalingMode, ScreenStatus, SetOption,
    StatusSnapshot,
};

/// The live-captured workshop item path (fixtures/socket-live-capture.txt).
const LIVE_BG: &str = "/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375";

// ---------------------------------------------------------------------------
// harness

/// Unique per-test temp dir (no external tempdir crate in the workspace).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = env::temp_dir().join(format!("kirie-ipc-{}-{name}", process::id()));
        fs::create_dir_all(&dir).expect("create tempdir");
        Self(dir)
    }
    fn sock(&self) -> PathBuf {
        self.0.join("ctl.sock")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Mock app loop: owns the "engine state" (speed + screens), captures every
/// delivered command, and answers fallible commands via `on_command`
/// (SPEC V3: everything crosses via channels; the snapshot is built fresh
/// per status request).
struct MockApp {
    screens: Vec<ScreenStatus>,
    on_command: Box<dyn FnMut(&Command) -> CommandOutcome + Send>,
}

impl MockApp {
    fn doc_semantics() -> Self {
        // Default oracle mirroring doc §4: bg fails on paths under /bad,
        // property fails for key "nosuchkey123" (doc §9 live capture),
        // scaling/clamp fail for unregistered screens, screenshot fails on
        // an empty path.
        Self {
            screens: vec![ScreenStatus {
                screen: "HDMI-A-1".into(),
                bg: Some(PathBuf::from(LIVE_BG)),
            }],
            on_command: Box::new(|cmd| match cmd {
                Command::Bg { path, .. } if path.starts_with("/bad") => CommandOutcome::Error,
                Command::Property { key, .. } if key == "nosuchkey123" => CommandOutcome::Error,
                Command::Scaling { screen, .. } | Command::Clamp { screen, .. } if screen != "HDMI-A-1" => {
                    CommandOutcome::Error
                }
                Command::Screenshot { path } if path.as_os_str().is_empty() => CommandOutcome::Error,
                _ => CommandOutcome::Ok,
            }),
        }
    }

    /// Spawn the app loop; returns the captured-command stream. The loop
    /// exits when the server (sole sender) drops the event channel.
    fn spawn(mut self, rx: Receiver<IpcEvent>) -> (JoinHandle<()>, Receiver<Command>) {
        let (cap_tx, cap_rx) = unbounded();
        let handle = thread::spawn(move || {
            let mut speed = 1.0f32;
            // Post-override property store, keyed by name (docs/compat-socket.md
            // §4.9 records the override; §11 reads it back).
            let mut props: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
            for event in rx {
                match event {
                    IpcEvent::Status { reply } => {
                        let _ = reply.send(StatusSnapshot {
                            speed,
                            screens: self.screens.clone(),
                        });
                    }
                    IpcEvent::GetProperties { screen: _, reply } => {
                        // Serialize the recorded overrides as a single-line JSON
                        // array `[{"key":..,"value":..}, ..]` — the same byte
                        // shape the real applier emits (docs/compat-socket.md §11).
                        let body: String = props
                            .iter()
                            .map(|(k, v)| format!(r#"{{"key":"{k}","value":"{v}"}}"#))
                            .collect::<Vec<_>>()
                            .join(",");
                        let _ = reply.send(format!("[{body}]"));
                    }
                    IpcEvent::Command { command, reply } => {
                        if let Command::Speed(v) = command {
                            speed = v;
                        }
                        if let Command::Property { key, value, .. } = &command {
                            props.insert(key.clone(), value.clone());
                        }
                        let outcome = (self.on_command)(&command);
                        let _ = cap_tx.send(command);
                        let _ = reply.send(outcome);
                    }
                }
            }
        });
        (handle, cap_rx)
    }
}

struct Server {
    _dir: TempDir,
    sock: PathBuf,
    server: ControlSocket,
    captured: Receiver<Command>,
    app: Option<JoinHandle<()>>,
}

impl Server {
    fn start(name: &str, mock: MockApp) -> Self {
        let dir = TempDir::new(name);
        let sock = dir.sock();
        let (tx, rx) = unbounded();
        let server = ControlSocket::bind(&sock, tx).expect("bind control socket");
        let (app, captured) = mock.spawn(rx);
        Self {
            _dir: dir,
            sock,
            server,
            captured,
            app: Some(app),
        }
    }

    fn request(&self, bytes: &[u8]) -> Vec<u8> {
        request_at(&self.sock, bytes)
    }

    /// Next captured command (the mock echoes them in delivery order).
    fn captured(&self) -> Command {
        self.captured
            .recv_timeout(Duration::from_secs(5))
            .expect("mock captured a command")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.server.shutdown();
        if let Some(app) = self.app.take() {
            let _ = app.join();
        }
    }
}

fn connect(sock: &Path) -> UnixStream {
    let stream = UnixStream::connect(sock).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    stream
}

/// One request/response cycle: write, then read to EOF (doc §2: clients must
/// read to EOF; the server close ends the response).
fn request_at(sock: &Path, bytes: &[u8]) -> Vec<u8> {
    let mut stream = connect(sock);
    stream.write_all(bytes).expect("write request");
    let mut response = Vec::new();
    stream.read_to_end(&mut response).expect("read response");
    response
}

// ---------------------------------------------------------------------------
// fixed vocabulary + framing (doc §2, §3, §9)

#[test]
fn ping_pong() {
    let s = Server::start("ping", MockApp::doc_semantics());
    assert_eq!(s.request(b"ping\n"), b"pong\n");
    // Args after the command token are irrelevant for ping.
    assert_eq!(s.request(b"ping whatever\n"), b"pong\n");
    // '\r' is stream whitespace for token args (doc §2).
    assert_eq!(s.request(b"ping\r\n"), b"pong\n");
}

#[test]
fn ping_without_newline_terminated_by_half_close() {
    // doc §9 verified live: `printf 'ping'` + EOF → pong\n.
    let s = Server::start("ping-eof", MockApp::doc_semantics());
    let mut stream = connect(&s.sock);
    stream.write_all(b"ping").unwrap();
    stream.shutdown(Shutdown::Write).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    assert_eq!(response, b"pong\n");
}

#[test]
fn ping_without_newline_terminated_by_read_timeout() {
    // doc §2 step 3: after 50 ms of silence the partial buffer is the
    // request — no EOF required.
    let s = Server::start("ping-timeout", MockApp::doc_semantics());
    let mut stream = connect(&s.sock);
    stream.write_all(b"ping").unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    assert_eq!(response, b"pong\n");
}

#[test]
fn second_line_discarded() {
    // doc §9: `ping\nstatus\n` in one connection → pong\n only.
    let s = Server::start("two-lines", MockApp::doc_semantics());
    assert_eq!(s.request(b"ping\nstatus\n"), b"pong\n");
}

#[test]
fn empty_line_gets_zero_response_bytes() {
    // doc §2 step 5 / §9: bare "\n" → 0 bytes, connection closed.
    let s = Server::start("empty", MockApp::doc_semantics());
    assert_eq!(s.request(b"\n"), b"");
    // Whitespace-only is NOT empty: command extraction fails → unknown.
    assert_eq!(s.request(b"  \n"), b"unknown command\n");
}

#[test]
fn unknown_command() {
    let s = Server::start("unknown", MockApp::doc_semantics());
    assert_eq!(s.request(b"frobnicate\n"), b"unknown command\n"); // doc §9
    assert_eq!(s.request(b"PING\n"), b"unknown command\n"); // case-sensitive
    assert_eq!(s.request(b"quit\n"), b"unknown command\n"); // doc §6: no quit
}

#[test]
fn oversized_request_line_is_served() {
    // doc §2 step 3: no cap on request size other than memory. 1 MiB path
    // must arrive intact at the app.
    let s = Server::start("oversized", MockApp::doc_semantics());
    let long = "a".repeat(1024 * 1024);
    let mut line = format!("bg HDMI-A-1 /{long}").into_bytes();
    line.push(b'\n');
    assert_eq!(s.request(&line), b"ok\n");
    match s.captured() {
        Command::Bg { screen, path } => {
            assert_eq!(screen, "HDMI-A-1");
            assert_eq!(path.as_os_str().len(), 1 + long.len());
        }
        other => panic!("expected bg, got {other:?}"),
    }
}

#[test]
fn half_open_client_times_out_without_blocking_others() {
    let s = Server::start("half-open", MockApp::doc_semantics());
    // Client A connects and sends nothing (no EOF either).
    let mut idle = connect(&s.sock);
    // Client B must still be served promptly (A costs the server ≤ 50 ms).
    let started = Instant::now();
    assert_eq!(s.request(b"ping\n"), b"pong\n");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "idle client stalled the server"
    );
    // A's connection: empty request → zero response bytes, then EOF.
    let mut response = Vec::new();
    idle.read_to_end(&mut response).unwrap();
    assert_eq!(response, b"");
}

#[test]
fn late_bytes_after_read_timeout_are_ignored() {
    // Per-read 50 ms timeout (doc §2): a pause mid-request ends it; the
    // partial buffer "pi" is the request → unknown command.
    let s = Server::start("late-bytes", MockApp::doc_semantics());
    let mut stream = connect(&s.sock);
    stream.write_all(b"pi").unwrap();
    thread::sleep(Duration::from_millis(150));
    let _ = stream.write_all(b"ng\n"); // may hit EPIPE; irrelevant
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    assert_eq!(response, b"unknown command\n");
}

// ---------------------------------------------------------------------------
// status (doc §4.2, fixtures)

#[test]
fn status_matches_live_capture_bytes() {
    let s = Server::start("status-live", MockApp::doc_semantics());
    let expected = format!("speed=1\nscreen=HDMI-A-1 bg={LIVE_BG}\n");
    assert_eq!(s.request(b"status\n"), expected.as_bytes());
}

#[test]
fn fixture_file_pairs_are_byte_exact() {
    // A live C++ engine capture: every recorded request must yield the recorded
    // response bytes (inlined; the standalone capture file was removed).
    let fixture = "\
=== request: status ===
speed=1
screen=HDMI-A-1 bg=/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375
=== request: speed ===
ok
=== request: volume ===
ok
";
    let mut pairs: Vec<(String, String)> = Vec::new();
    for line in fixture.lines() {
        if let Some(name) = line
            .strip_prefix("=== request: ")
            .and_then(|r| r.strip_suffix(" ==="))
        {
            pairs.push((name.to_string(), String::new()));
        } else if let Some((_, response)) = pairs.last_mut() {
            response.push_str(line);
            response.push('\n');
        }
    }
    assert!(!pairs.is_empty(), "fixture parsed no request/response pairs");
    let s = Server::start("fixture", MockApp::doc_semantics());
    for (request, expected) in pairs {
        let actual = s.request(format!("{request}\n").as_bytes());
        assert_eq!(
            actual,
            expected.as_bytes(),
            "response mismatch for fixture request {request:?}"
        );
    }
}

#[test]
fn status_multi_screen_ordering_and_empty_bg() {
    // Screens delivered unsorted; response must be lexicographic by key
    // bytes ("DP-10" < "DP-2" < "HDMI-A-1"), std::map order (doc §4.2).
    let mock = MockApp {
        screens: vec![
            ScreenStatus {
                screen: "HDMI-A-1".into(),
                bg: Some(PathBuf::from("/w/c")),
            },
            ScreenStatus {
                screen: "DP-2".into(),
                bg: Some(PathBuf::from("/w/has space")),
            },
            ScreenStatus {
                screen: "DP-10".into(),
                bg: None,
            },
        ],
        on_command: Box::new(|_| CommandOutcome::Ok),
    };
    let s = Server::start("status-multi", mock);
    assert_eq!(
        s.request(b"status\n"),
        b"speed=1\nscreen=DP-10 bg=\nscreen=DP-2 bg=/w/has space\nscreen=HDMI-A-1 bg=/w/c\n"
    );
}

#[test]
fn status_reflects_speed_commands() {
    let s = Server::start("status-speed", MockApp::doc_semantics());
    assert_eq!(s.request(b"speed 0.5\n"), b"ok\n");
    let _ = s.captured();
    assert!(s.request(b"status\n").starts_with(b"speed=0.5\n"));
    // ≤ 0 / non-numeric coerce back to 1 (doc §4.3).
    assert_eq!(s.request(b"speed 0\n"), b"ok\n");
    let _ = s.captured();
    assert!(s.request(b"status\n").starts_with(b"speed=1\n"));
}

// ---------------------------------------------------------------------------
// getproperties read-back (docs/compat-socket.md §11, kirie extension)

#[test]
fn getproperties_reflects_property_overrides_over_the_socket() {
    // The daemon's list->edit->save->apply loop: push a `property` set, then
    // read it back with `getproperties`; the current value must reflect the
    // override (docs/compat-socket.md §11). Response is a single-line JSON
    // array terminated by exactly one '\n'.
    let s = Server::start("getprops", MockApp::doc_semantics());
    // Empty schema before any override, byte-clean.
    assert_eq!(s.request(b"getproperties\n"), b"[]\n");
    // Set two properties (fallible ok on the registered screen).
    assert_eq!(s.request(b"property HDMI-A-1 bloom true\n"), b"ok\n");
    let _ = s.captured();
    assert_eq!(s.request(b"property HDMI-A-1 outline 0.5 0.25 0.75\n"), b"ok\n");
    let _ = s.captured();
    // Read back: both overrides present, single line + one trailing newline.
    let body = s.request(b"getproperties HDMI-A-1\n");
    assert_eq!(
        body,
        br#"[{"key":"bloom","value":"true"},{"key":"outline","value":"0.5 0.25 0.75"}]"#
            .iter()
            .chain(b"\n")
            .copied()
            .collect::<Vec<u8>>()
            .as_slice()
    );
    // Exactly one trailing newline; no embedded newline in the JSON payload.
    assert_eq!(body.iter().filter(|&&b| b == b'\n').count(), 1);
    assert_eq!(body.last(), Some(&b'\n'));
}

#[test]
fn getproperties_is_unknown_absent_an_app_arm() {
    // If the app drops the reply (e.g. shutting down) the read-back yields the
    // dead-engine signal: zero response bytes (docs/compat-socket.md §3, §11).
    let dir = TempDir::new("getprops-dead");
    let sock = dir.sock();
    let (tx, rx) = unbounded();
    drop(rx);
    let _server = ControlSocket::bind(&sock, tx).expect("bind");
    assert_eq!(request_at(&sock, b"getproperties\n"), b"");
}

// ---------------------------------------------------------------------------
// always-ok commands (doc §4.3-§4.6, §4.8)

#[test]
fn bare_speed_and_volume_reply_ok_like_live_capture() {
    // fixtures/socket-live-capture.txt: bare `speed` / `volume` → ok\n.
    let s = Server::start("bare-args", MockApp::doc_semantics());
    assert_eq!(s.request(b"speed\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Speed(1.0));
    assert_eq!(s.request(b"volume\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Volume(128));
    assert_eq!(s.request(b"mute\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Mute(false));
}

#[test]
fn volume_is_not_clamped() {
    // doc §4.4: out-of-range values forwarded as-is.
    let s = Server::start("volume-raw", MockApp::doc_semantics());
    assert_eq!(s.request(b"volume 500\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Volume(500));
    assert_eq!(s.request(b"volume -7\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Volume(-7));
    assert_eq!(s.request(b"volume abc\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Volume(0));
}

#[test]
fn mute_nonzero_semantics() {
    let s = Server::start("mute", MockApp::doc_semantics());
    assert_eq!(s.request(b"mute 1\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Mute(true));
    assert_eq!(s.request(b"mute 0\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Mute(false));
    assert_eq!(s.request(b"mute 2\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Mute(true));
}

#[test]
fn set_recognized_keys_ok_unknown_key_error() {
    let s = Server::start("set", MockApp::doc_semantics());
    assert_eq!(s.request(b"set fps 30\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Set(SetOption::Fps(30)));
    assert_eq!(s.request(b"set renderscale 5\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Set(SetOption::RenderScale(2.0))); // clamped
    assert_eq!(s.request(b"set audiodevice default\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Set(SetOption::AudioDevice(String::new())));
    assert_eq!(s.request(b"set disablemouse TRUE\n"), b"ok\n");
    assert_eq!(s.captured(), Command::Set(SetOption::DisableMouse(false))); // exact-string bool
    // Unknown key → error\n (doc §4.6); never reaches the app.
    assert_eq!(s.request(b"set bogus 1\n"), b"error\n");
    assert_eq!(s.request(b"set\n"), b"error\n");
    assert!(s.captured.is_empty(), "rejected set leaked to the app");
}

#[test]
fn preload_replies_ok_even_when_the_app_fails() {
    // doc §4.8 / ControlSocket.cpp:128-132: ok\n unconditionally, failures
    // only logged. The mock reports Error; the wire must still say ok.
    let mock = MockApp {
        screens: vec![],
        on_command: Box::new(|_| CommandOutcome::Error),
    };
    let s = Server::start("preload", mock);
    assert_eq!(s.request(b"preload /definitely/not/a/wallpaper\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Preload {
            path: PathBuf::from("/definitely/not/a/wallpaper")
        }
    );
}

// ---------------------------------------------------------------------------
// fallible commands (doc §4.7, §4.9-§4.12)

#[test]
fn bg_ok_and_error_paths() {
    let s = Server::start("bg", MockApp::doc_semantics());
    let line = format!("bg HDMI-A-1 {LIVE_BG}\n");
    assert_eq!(s.request(line.as_bytes()), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Bg {
            screen: "HDMI-A-1".into(),
            path: PathBuf::from(LIVE_BG)
        }
    );
    // Unloadable path → error\n, prior wallpaper keeps running (doc §7).
    assert_eq!(s.request(b"bg HDMI-A-1 /bad/dir\n"), b"error\n");
    let _ = s.captured();
    // No screen-name validation in the engine (doc §4.7): bogus screen with
    // a loadable path still returns ok.
    assert_eq!(s.request(b"bg BOGUS /w/fine\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Bg {
            screen: "BOGUS".into(),
            path: PathBuf::from("/w/fine")
        }
    );
    // Path with spaces survives rest-of-line extraction (doc §2).
    assert_eq!(s.request(b"bg HDMI-A-1 /w/dir with spaces\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Bg {
            screen: "HDMI-A-1".into(),
            path: PathBuf::from("/w/dir with spaces")
        }
    );
}

#[test]
fn property_ok_error_and_value_fidelity() {
    let s = Server::start("property", MockApp::doc_semantics());
    // doc §4.9 example corpus values.
    assert_eq!(s.request(b"property HDMI-A-1 bloom true\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Property {
            screen: "HDMI-A-1".into(),
            key: "bloom".into(),
            value: "true".into()
        }
    );
    // Color triple: value is rest-of-line WITH spaces, delivered intact.
    assert_eq!(
        s.request(b"property HDMI-A-1 outline 0.36585 0.04268 0.43902\n"),
        b"ok\n"
    );
    assert_eq!(
        s.captured(),
        Command::Property {
            screen: "HDMI-A-1".into(),
            key: "outline".into(),
            value: "0.36585 0.04268 0.43902".into(),
        }
    );
    // doc §9 verified live: undeclared key → error\n.
    assert_eq!(s.request(b"property HDMI-A-1 nosuchkey123 1\n"), b"error\n");
}

#[test]
fn scaling_and_clamp_modes_and_errors() {
    let s = Server::start("scaling", MockApp::doc_semantics());
    assert_eq!(s.request(b"scaling HDMI-A-1 fill\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Scaling {
            screen: "HDMI-A-1".into(),
            mode: ScalingMode::Fill
        }
    );
    // doc §9 verified live: bogus mode → error\n, and it must NOT reach the
    // app (nothing stored, doc §4.10).
    assert_eq!(s.request(b"scaling HDMI-A-1 bogusmode\n"), b"error\n");
    assert!(s.captured.is_empty(), "invalid scaling mode leaked to the app");
    // Valid mode + unregistered screen → app-side error (doc §4.10).
    assert_eq!(s.request(b"scaling DP-9 fit\n"), b"error\n");
    assert_eq!(
        s.captured(),
        Command::Scaling {
            screen: "DP-9".into(),
            mode: ScalingMode::Fit
        }
    );

    assert_eq!(s.request(b"clamp HDMI-A-1 border\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Clamp {
            screen: "HDMI-A-1".into(),
            mode: ClampMode::Border
        }
    );
    assert_eq!(s.request(b"clamp HDMI-A-1 nope\n"), b"error\n");
    assert!(s.captured.is_empty(), "invalid clamp mode leaked to the app");
}

#[test]
fn screenshot_ok_and_empty_path_error() {
    let s = Server::start("screenshot", MockApp::doc_semantics());
    assert_eq!(s.request(b"screenshot /tmp/kirie test.png\n"), b"ok\n");
    assert_eq!(
        s.captured(),
        Command::Screenshot {
            path: PathBuf::from("/tmp/kirie test.png")
        }
    );
    assert_eq!(s.request(b"screenshot\n"), b"error\n"); // doc §4.12
}

// ---------------------------------------------------------------------------
// lifecycle + architecture

#[test]
fn stale_socket_file_is_unlinked_on_bind() {
    // doc §1: unlink(path) unconditionally before bind — a leftover socket
    // file from an abnormal exit must not prevent startup.
    let dir = TempDir::new("stale");
    let sock = dir.sock();
    fs::write(&sock, b"stale").unwrap(); // plain file squatting on the path
    let (tx, rx) = unbounded();
    let server = ControlSocket::bind(&sock, tx).expect("bind over stale file");
    let (app, _cap) = MockApp::doc_semantics().spawn(rx);
    assert_eq!(request_at(&sock, b"ping\n"), b"pong\n");
    drop(server);
    let _ = app.join();
    // Clean teardown unlinks the socket file (doc §1).
    assert!(!sock.exists(), "socket file left behind after shutdown");
}

#[test]
fn app_gone_yields_dead_engine_signal() {
    // If the app side is gone, commands answer with zero bytes — exactly
    // the protocol's dead/orphaned-socket signal (doc §3). ping still works:
    // the socket layer answers it itself (doc §4.1).
    let dir = TempDir::new("app-gone");
    let sock = dir.sock();
    let (tx, rx) = unbounded();
    drop(rx);
    let _server = ControlSocket::bind(&sock, tx).expect("bind");
    assert_eq!(request_at(&sock, b"ping\n"), b"pong\n");
    assert_eq!(request_at(&sock, b"speed 1\n"), b"");
    assert_eq!(request_at(&sock, b"status\n"), b"");
}

#[test]
fn concurrent_clients_are_all_served() {
    // doc §4.9 latency note: the daemon fires N property sets concurrently;
    // all must be answered (serialized in accept order server-side).
    let s = Server::start("concurrent", MockApp::doc_semantics());
    let sock = s.sock.clone();
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let sock = sock.clone();
            thread::spawn(move || request_at(&sock, format!("property HDMI-A-1 fov {i}\n").as_bytes()))
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), b"ok\n");
    }
    for _ in 0..10 {
        let _ = s.captured();
    }
}

#[test]
fn non_utf8_bg_path_reaches_app_byte_exact() {
    // Paths are raw bytes on Linux; the parser must not mangle them (V9).
    let s = Server::start("non-utf8", MockApp::doc_semantics());
    let mut line = b"bg HDMI-A-1 /weird/\xff\xfe/dir".to_vec();
    line.push(b'\n');
    assert_eq!(s.request(&line), b"ok\n");
    match s.captured() {
        Command::Bg { path, .. } => assert_eq!(path.as_os_str().as_bytes(), b"/weird/\xff\xfe/dir"),
        other => panic!("expected bg, got {other:?}"),
    }
}

/// Round-trip every `Command` variant through the full stack: canonical wire
/// line in, identical typed command captured at the app, correct wire
/// response out (SPEC V13 adapted to the socket protocol).
#[test]
fn round_trip_every_command_variant_over_the_socket() {
    let s = Server::start("roundtrip", MockApp::doc_semantics());
    let all = [
        Command::Speed(0.5),
        Command::Volume(64),
        Command::Mute(true),
        Command::Set(SetOption::Fps(30)),
        Command::Set(SetOption::NoAutomute(true)),
        Command::Set(SetOption::DisableMouse(false)),
        Command::Set(SetOption::DisableParallax(true)),
        Command::Set(SetOption::NoFullscreenPause(false)),
        Command::Set(SetOption::RenderScale(1.06)),
        Command::Set(SetOption::AudioDevice("alsa_output.pci 0000_00.analog".into())),
        Command::Bg {
            screen: "HDMI-A-1".into(),
            path: PathBuf::from("/path/with spaces/dir"),
        },
        Command::Preload {
            path: PathBuf::from("/w/431960/3047596375"),
        },
        Command::Property {
            screen: "HDMI-A-1".into(),
            key: "outline".into(),
            value: "0.36585 0.04268 0.43902".into(),
        },
        Command::Scaling {
            screen: "HDMI-A-1".into(),
            mode: ScalingMode::Stretch,
        },
        Command::Clamp {
            screen: "HDMI-A-1".into(),
            mode: ClampMode::Repeat,
        },
        Command::Screenshot {
            path: PathBuf::from("/tmp/shot.png"),
        },
    ];
    for cmd in all {
        let mut line = cmd.to_request_line();
        line.push(b'\n');
        let response = s.request(&line);
        assert_eq!(response, b"ok\n", "unexpected response for {cmd:?}");
        assert_eq!(s.captured(), cmd, "command mangled in transit");
    }
}

#[test]
fn read_timeout_close_is_observable_quickly() {
    // The half-open timeout must be on the order of the C++ 50 ms
    // SO_RCVTIMEO, not seconds: measure EOF latency for a silent client.
    let s = Server::start("timeout-latency", MockApp::doc_semantics());
    let mut stream = connect(&s.sock);
    let started = Instant::now();
    let mut sink = Vec::new();
    stream.read_to_end(&mut sink).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(sink, b"");
    assert!(
        elapsed < Duration::from_millis(1500),
        "silent client held open for {elapsed:?}"
    );
}

#[test]
fn shutdown_is_idempotent_and_unbinds() {
    let dir = TempDir::new("shutdown");
    let sock = dir.sock();
    let (tx, rx) = unbounded();
    let mut server = ControlSocket::bind(&sock, tx).expect("bind");
    assert_eq!(server.path(), sock.as_path());
    let (app, _cap) = MockApp::doc_semantics().spawn(rx);
    assert_eq!(request_at(&sock, b"ping\n"), b"pong\n");
    server.shutdown();
    server.shutdown(); // second call is a no-op
    let _ = app.join();
    assert!(!sock.exists());
    match UnixStream::connect(&sock) {
        Err(e) => assert!(
            matches!(e.kind(), ErrorKind::NotFound | ErrorKind::ConnectionRefused),
            "unexpected error kind {e:?}"
        ),
        Ok(_) => panic!("socket still accepting after shutdown"),
    }
}
