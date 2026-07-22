//! `kirie-webhost` — the out-of-process web wallpaper host.
//!
//! Owns the CEF context and one windowless browser, publishes frames into a
//! `memfd` seqlock buffer the engine maps read-only (announced as
//! `shm /proc/<pid>/fd/<fd> <bytes>` on stdout), and takes line commands on
//! stdin (`resize`/`pointer`/`mute`/`props`/`quit`). The engine kills this
//! process to tear the browser down — the kernel then reclaims every thread,
//! zygote and heap deterministically, which in-process `cef_shutdown` never
//! guaranteed. See `kirie_web::hosted` for the protocol/layout.

use std::io::BufRead;
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::mpsc::{TryRecvError, channel};
use std::time::{Duration, Instant};

use kirie_web::backend::{PointerState, WebBackend, WebSize};
use kirie_web::cef::CefBackend;
use kirie_web::hosted::{SHM_HEADER, SHM_PIXELS};

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(a) = args.next() {
        if a == name {
            return args.next();
        }
    }
    None
}

fn main() {
    // Child logs to stderr, which the engine inherits into its own log.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let Some(url) = arg("--url") else {
        eprintln!("kirie-webhost: missing --url");
        std::process::exit(2);
    };
    let width: u32 = arg("--width").and_then(|v| v.parse().ok()).unwrap_or(1920);
    let height: u32 = arg("--height").and_then(|v| v.parse().ok()).unwrap_or(1080);

    // Frame buffer: an anonymous memfd, republished to the parent via procfs
    // (same-uid open of /proc/<pid>/fd/<fd>). Sparse — pages materialize only
    // for the frames actually written.
    let shm_len = SHM_HEADER + SHM_PIXELS;
    // SAFETY: plain syscalls creating and sizing an anonymous fd we own.
    let (shm_file, shm_fd) = unsafe {
        let fd = libc::memfd_create(c"kirie-web-frames".as_ptr(), 0);
        if fd < 0 {
            eprintln!("kirie-webhost: memfd_create failed");
            std::process::exit(1);
        }
        if libc::ftruncate(fd, shm_len as libc::off_t) != 0 {
            eprintln!("kirie-webhost: ftruncate failed");
            std::process::exit(1);
        }
        (std::fs::File::from_raw_fd(fd), fd)
    };
    // SAFETY: writable shared mapping of our own memfd; only this process maps
    // it writable (the engine maps the fd read-only).
    let mut shm = match unsafe { memmap2::MmapMut::map_mut(&shm_file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("kirie-webhost: mmap failed: {e}");
            std::process::exit(1);
        }
    };
    println!("shm /proc/{}/fd/{} {}", std::process::id(), shm_fd, shm_len);

    // stdin command reader → channel (the pump loop must never block on IO).
    let (tx, rx) = channel::<String>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
        // Engine hung up (crash?) — no reason to outlive it.
        let _ = tx;
    });

    let mut backend = match CefBackend::new(&url, WebSize { width, height }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("kirie-webhost: browser start failed: {e}");
            std::process::exit(1);
        }
    };
    println!("ready");

    let mut seq: u64 = 0;
    let mut last_pub: (*const u8, u32, u32) = (std::ptr::null(), 0, 0);
    let frame_dt = Duration::from_millis(8);
    let mut last = Instant::now();
    'run: loop {
        let tick_start = Instant::now();
        loop {
            match rx.try_recv() {
                Ok(line) => {
                    let mut p = line.split_whitespace();
                    match p.next() {
                        Some("resize") => {
                            let w = p.next().and_then(|v| v.parse().ok()).unwrap_or(width);
                            let h = p.next().and_then(|v| v.parse().ok()).unwrap_or(height);
                            backend.resize(WebSize { width: w, height: h });
                        }
                        Some("pointer") => {
                            let x = p.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                            let y = p.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                            let left = p.next() == Some("1");
                            let right = p.next() == Some("1");
                            backend.send_pointer(PointerState { x, y, left, right });
                        }
                        Some("mute") => backend.set_muted(p.next() == Some("1")),
                        Some("props") => {
                            if let Some(rest) = line.strip_prefix("props ") {
                                backend.apply_properties(rest);
                            }
                        }
                        Some("quit") => break 'run,
                        _ => {}
                    }
                }
                Err(TryRecvError::Empty) => break,
                // Engine gone: exit rather than render to nowhere.
                Err(TryRecvError::Disconnected) => break 'run,
            }
        }

        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f32();
        last = now;
        backend.tick(dt);

        if let Some(frame) = backend.latest_frame() {
            let key = (frame.data.as_ptr(), frame.width, frame.height);
            let len = frame.data.len();
            if key != last_pub && SHM_HEADER + len <= shm.len() {
                last_pub = key;
                // Seqlock publish: odd while writing, even when stable.
                seq += 1;
                shm[0..8].copy_from_slice(&seq.to_le_bytes());
                shm[8..12].copy_from_slice(&frame.width.to_le_bytes());
                shm[12..16].copy_from_slice(&frame.height.to_le_bytes());
                shm[16..20].copy_from_slice(&0u32.to_le_bytes());
                shm[SHM_HEADER..SHM_HEADER + len].copy_from_slice(frame.data);
                seq += 1;
                shm[0..8].copy_from_slice(&seq.to_le_bytes());
            }
        }

        if let Some(rem) = frame_dt.checked_sub(tick_start.elapsed()) {
            std::thread::sleep(rem);
        }
    }

    // Hard exit backstop: if CEF's teardown wedges, the engine's kill covers
    // us, but don't let a hang keep the GPU/audio alive meanwhile.
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(3));
        std::process::exit(0);
    });
    drop(backend);
    std::process::exit(0);
}
