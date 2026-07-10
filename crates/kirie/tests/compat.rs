//! Integration tests for the `linux-wallpaperengine` compat surface
//! (docs/compat-cli.md, docs/compat-socket.md).
//!
//! Two tiers:
//!
//! * **arg-parse table tests** call the library parser directly
//!   ([`kirie::compat::args`]) — the exact daemon launch line
//!   (fixtures/cpp-live-cmdline.txt) and the doc §3/§4 edge cases;
//! * **live e2e** (gated on `$WAYLAND_DISPLAY` + the workshop corpus) drives
//!   the built binary against the real compositor: a video wallpaper in
//!   `--window` for a few seconds, an offscreen `--screenshot`, and the
//!   control socket with byte-exact responses (doc §9). The live scenarios
//!   run sequentially inside one test to avoid GPU/compositor contention.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kirie::compat::args::{self, ClampMode, ScalingMode, WindowMode};

/// The corpus video item (SPEC §C: the one `"type":"video"` workshop item).
const CORPUS_VIDEO: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960/3600453929";
/// The corpus scene item used by the live daemon (fixtures/cpp-live-cmdline.txt).
const CORPUS_SCENE: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960/3047596375";

fn os(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
}

// ---- arg-parse table tests (corpus-independent) ---------------------------

#[test]
fn exact_live_cmdline_parses_to_expected_model() {
    // fixtures/cpp-live-cmdline.txt (doc §8.1 [observed]) with the task's test
    // socket path substituted for the daemon's.
    let argv = os(&[
        "linux-wallpaperengine",
        "--control-socket",
        "/tmp/claude-1000/kirie-test.sock",
        "--screen-root",
        "HDMI-A-1",
        "--bg",
        CORPUS_SCENE,
        "--scaling",
        "fill",
        "--clamp",
        "clamp",
        "--fps",
        "30",
        "--render-scale",
        "1.06",
        "--volume",
        "0",
        "--set-property",
        "fov=48.333333333333336",
        "--set-property",
        "bloom=true",
        "--set-property",
        "radialblur=false",
        "--set-property",
        "huespeed=0.10555555555555556",
        "--set-property",
        "coloring1=2",
        "--set-property",
        "newproperty=0.025",
        "--set-property",
        "schemecolor=0.00000 0.00000 0.00000",
        "--set-property",
        "outline=0.36585 0.04268 0.43902",
        "--set-property",
        "bloomstrength=1.7916666666666665",
        "--set-property",
        "color1=0.00000 0.00000 1.00000",
        "--set-property",
        "color2=0.46951 0.00000 0.77439",
    ]);
    let parsed = args::validate(args::parse(&argv).expect("parse")).expect("validate");

    assert_eq!(parsed.mode, WindowMode::DesktopBackground);
    assert_eq!(
        parsed.control_socket.as_deref(),
        Some(Path::new("/tmp/claude-1000/kirie-test.sock"))
    );
    assert_eq!(parsed.screens.len(), 1);
    assert_eq!(parsed.screens[0].name, "HDMI-A-1");
    assert_eq!(parsed.screens[0].background.as_deref(), Some(CORPUS_SCENE));
    assert_eq!(parsed.screens[0].scaling, ScalingMode::Fill);
    assert_eq!(parsed.screens[0].clamp, ClampMode::Clamp);
    assert_eq!(parsed.fps, 30);
    assert!((parsed.render_scale - 1.06).abs() < 1e-12);
    assert_eq!(parsed.volume, 0);
    assert_eq!(parsed.set_properties.len(), 11);
    assert_eq!(
        parsed.set_properties[6],
        ("schemecolor".to_owned(), "0.00000 0.00000 0.00000".to_owned())
    );
    assert_eq!(parsed.default_background.as_deref(), Some(CORPUS_SCENE));
}

#[test]
fn geometry_parse_cases() {
    let g = |s: &str| args::parse(&os(&["kirie", "--window", s, "/tmp/x"])).map(|a| a.window);
    // Full geometry.
    let w = g("100x200x1920x1080").unwrap().unwrap();
    assert_eq!((w.x, w.y, w.w, w.h), (100, 200, 1920, 1080));
    // doc §3.2: extra x components are ignored.
    let w = g("1x2x3x4x5").unwrap().unwrap();
    assert_eq!((w.x, w.y, w.w, w.h), (1, 2, 3, 4));
    // doc §3.2: garbage components strtol to 0.
    let w = g("axbx1920x1080").unwrap().unwrap();
    assert_eq!((w.x, w.y, w.w, w.h), (0, 0, 1920, 1080));
    // Fewer than three delimiters is fatal.
    assert!(g("1920x1080").is_err());
}

#[test]
fn help_flag_exits_zero() {
    let out = Command::new(env!("CARGO_BIN_EXE_kirie"))
        .arg("--help")
        .output()
        .expect("spawn kirie --help");
    assert!(out.status.success(), "--help must exit 0 (doc §5)");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("Usage: linux-wallpaperengine"),
        "help synopsis missing:\n{stdout}"
    );
}

#[test]
fn unknown_flag_is_ignored_then_missing_background_fails() {
    // doc §4.1: an unknown flag is not itself an error; the run fails only on
    // the missing background (doc §4.8).
    let out = Command::new(env!("CARGO_BIN_EXE_kirie"))
        .arg("--bogus-flag")
        .output()
        .expect("spawn kirie --bogus-flag");
    assert!(!out.status.success(), "missing background must fail (doc §4.8)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("At least one background ID must be specified"),
        "expected the missing-background fatal, got:\n{stderr}"
    );
}

#[test]
fn bad_choice_exits_nonzero() {
    // doc §4.6: a bad --scaling choice is fatal with the allowed-options message.
    let out = Command::new(env!("CARGO_BIN_EXE_kirie"))
        .args(["--scaling", "nope", "/tmp/x"])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("allowed options"), "got:\n{stderr}");
}

#[test]
fn info_subcommand_still_works() {
    // The existing subcommand surface must keep working (task: extend, don't
    // break). `info` on a missing path fails cleanly.
    let out = Command::new(env!("CARGO_BIN_EXE_kirie"))
        .args(["info", "/nonexistent/kirie-compat-test"])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    assert!(!out.stderr.is_empty());
}

// ---- live e2e (gated on $WAYLAND_DISPLAY + the corpus) ---------------------

/// The socket path the task specifies; a unique suffix avoids stale-file races.
fn socket_path() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    dir.join(format!("kirie-test-{}.sock", std::process::id()))
}

fn corpus_video() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("KIRIE_CORPUS_VIDEO").unwrap_or_else(|| CORPUS_VIDEO.into()));
    dir.join("project.json").is_file().then_some(dir)
}

fn have_wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
}

/// Wait up to `timeout` for `child` to exit; kill it if it overruns.
fn wait_or_kill(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => return None,
        }
    }
}

/// One request/response round trip (doc §2: one line, one response, close).
fn socket_roundtrip(path: &Path, request: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut stream = UnixStream::connect(path)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(request)?;
    // Half-close so the server also sees EOF (doc §2 step 3-4).
    let _ = stream.shutdown(Shutdown::Write);
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    Ok(buf)
}

#[test]
fn live_e2e_video_window_screenshot_and_socket() {
    if !have_wayland() {
        eprintln!("skipping live e2e: WAYLAND_DISPLAY unset");
        return;
    }
    let Some(video) = corpus_video() else {
        eprintln!("skipping live e2e: corpus video {CORPUS_VIDEO} not installed");
        return;
    };
    let video_str = video.to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_kirie");

    // (1) Video wallpaper in --window mode for 5s → clean exit 0. --window
    // falls back to a layer-shell surface (kirie-platform exposes no xdg
    // toplevel — reported as a blocker); the run is bounded by
    // KIRIE_RUN_SECONDS so it exits on its own.
    {
        let mut child = Command::new(bin)
            .args(["--window", "0x0x1280x720", "--bg", &video_str])
            .env("KIRIE_RUN_SECONDS", "5")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn --window");
        let status =
            wait_or_kill(&mut child, Duration::from_secs(25)).expect("--window run did not exit within 25s");
        assert!(
            status.success(),
            "--window video run exited with {status:?} (expected 0)"
        );
    }

    // (2) Offscreen --screenshot → a non-black PNG (unlocks the P4 SSIM gate).
    {
        let shot = std::env::temp_dir().join(format!("kirie-shot-{}.png", std::process::id()));
        let _ = std::fs::remove_file(&shot);
        let mut child = Command::new(bin)
            .args(["--bg", &video_str, "--screenshot"])
            .arg(&shot)
            .args(["--screenshot-delay", "5"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn --screenshot");
        let status = wait_or_kill(&mut child, Duration::from_secs(30))
            .expect("--screenshot did not finish within 30s");
        assert!(status.success(), "--screenshot exited with {status:?}");
        let img = image::open(&shot)
            .unwrap_or_else(|e| panic!("screenshot {} did not decode: {e}", shot.display()))
            .to_rgba8();
        let total = img.pixels().len();
        let lit = img
            .pixels()
            .filter(|p| p.0[0] > 8 || p.0[1] > 8 || p.0[2] > 8)
            .count();
        assert!(
            lit * 100 > total * 5,
            "screenshot is essentially black: {lit}/{total} pixels lit"
        );
        let _ = std::fs::remove_file(&shot);
    }

    // (3) Control socket: byte-exact responses (doc §9) + clean shutdown.
    {
        let sock = socket_path();
        let _ = std::fs::remove_file(&sock);
        let mut child = Command::new(bin)
            .args(["--control-socket"])
            .arg(&sock)
            .args(["--screen-root", "HDMI-A-1", "--bg", &video_str, "--volume", "0"])
            .env("KIRIE_RUN_SECONDS", "5")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn --control-socket");

        // Daemon readiness: the socket file must appear within ~5s (doc §8.3).
        let deadline = Instant::now() + Duration::from_secs(5);
        while !sock.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(sock.exists(), "control socket never appeared (doc §8.3)");
        // Give the applier a moment to seed its screen registry.
        std::thread::sleep(Duration::from_millis(200));

        let want_status = format!("speed=1\nscreen=HDMI-A-1 bg={video_str}\n").into_bytes();

        assert_eq!(socket_roundtrip(&sock, b"ping\n").unwrap(), b"pong\n");
        assert_eq!(socket_roundtrip(&sock, b"status\n").unwrap(), want_status);
        assert_eq!(socket_roundtrip(&sock, b"speed 0.5\n").unwrap(), b"ok\n");
        assert_eq!(socket_roundtrip(&sock, b"volume 30\n").unwrap(), b"ok\n");
        assert_eq!(socket_roundtrip(&sock, b"mute 1\n").unwrap(), b"ok\n");
        // Speed change is reflected in status (doc §4.2/§4.3).
        let after = format!("speed=0.5\nscreen=HDMI-A-1 bg={video_str}\n").into_bytes();
        assert_eq!(socket_roundtrip(&sock, b"status\n").unwrap(), after);
        // Fixed vocabulary (doc §9).
        assert_eq!(
            socket_roundtrip(&sock, b"frobnicate\n").unwrap(),
            b"unknown command\n"
        );
        assert_eq!(
            socket_roundtrip(&sock, b"scaling HDMI-A-1 bogusmode\n").unwrap(),
            b"error\n"
        );
        // Empty request → zero response bytes (doc §2 step 5).
        assert_eq!(socket_roundtrip(&sock, b"\n").unwrap(), Vec::<u8>::new());

        // Clean shutdown: the bounded run exits 0 and unlinks the socket file
        // on the clean teardown path (doc §1). No `stop` command exists in the
        // protocol (doc §6) — the engine stops on the run bound / a signal.
        let status =
            wait_or_kill(&mut child, Duration::from_secs(10)).expect("socket run did not exit within 10s");
        assert!(status.success(), "socket run exited with {status:?}");
        assert!(
            !sock.exists(),
            "socket file not unlinked on clean shutdown (doc §1)"
        );
        let _ = std::fs::remove_file(&sock);
    }
}
