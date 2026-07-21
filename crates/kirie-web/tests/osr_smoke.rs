//! CEF off-screen-rendering smoke test (feature `cef` only).
//!
//! Drives a real windowless CEF browser over one corpus web wallpaper's
//! `index.html`, pumps its message loop until a non-blank frame is painted,
//! and writes it to a PNG for inspection. This is the end-to-end proof that
//! the OSR paint path (docs/subsystems-misc.md §3.5) actually renders.
//!
//! Requires the CEF runtime files next to the test binary (the `cef-dll-sys`
//! build script copies them into `target/<profile>/`) and `libcef.so` on the
//! loader path — run with:
//!
//! ```text
//! LD_LIBRARY_PATH=target/debug \
//!   cargo test -p kirie-web --features cef --test osr_smoke -- --nocapture --ignored
//! ```
//!
//! Marked `#[ignore]` so the default `cargo test` (which cannot link libcef on
//! a box without it) never runs it.
#![cfg(feature = "cef")]

use std::time::{Duration, Instant};

use kirie_web::WebBackend;
use kirie_web::backend::WebSize;
use kirie_web::cef::CefBackend;

const CORPUS_INDEX: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960/2968833989/index.html";

#[test]
#[ignore = "needs libcef runtime; run explicitly with LD_LIBRARY_PATH set"]
fn osr_renders_corpus_index() {
    let url = format!("file://{CORPUS_INDEX}");
    let size = WebSize {
        width: 1280,
        height: 720,
    };

    let mut backend = CefBackend::new(&url, size).expect("create CEF backend");

    // Pump for up to ~12s waiting for a non-blank paint.
    let deadline = Instant::now() + Duration::from_secs(12);
    let mut best: Option<(Vec<u8>, u32, u32)> = None;
    let mut non_blank = false;

    while Instant::now() < deadline {
        backend.tick(1.0 / 60.0);
        if let Some(frame) = backend.latest_frame() {
            let owned = frame.data.to_vec();
            let blank = is_blank(&owned);
            best = Some((owned, frame.width, frame.height));
            if !blank {
                non_blank = true;
                // Keep pumping a little longer so animations settle, then stop.
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(16));
    }

    backend.shutdown();

    let (data, w, h) = best.expect("no frame was ever painted");

    // Convert BGRA (CEF native) -> RGBA for PNG.
    let mut rgba = data.clone();
    for px in rgba.as_chunks_mut::<4>().0 {
        px.swap(0, 2);
    }
    let out = std::env::temp_dir().join("kirie-web-osr.png");
    image::save_buffer(&out, &rgba, w, h, image::ColorType::Rgba8).expect("write png");
    println!("wrote OSR frame {w}x{h} to {}", out.display());

    assert!(
        non_blank,
        "browser painted only blank frames ({w}x{h}); see {}",
        out.display()
    );
}

/// Two backends share one CEF context: both paint concurrently (the
/// multi-monitor web case), and closing one leaves the other painting.
#[test]
#[ignore = "needs libcef runtime; run explicitly with LD_LIBRARY_PATH set"]
fn osr_two_browsers_share_one_context() {
    let url = format!("file://{CORPUS_INDEX}");
    let size_a = WebSize {
        width: 1280,
        height: 720,
    };
    let size_b = WebSize {
        width: 800,
        height: 600,
    };

    let mut a = CefBackend::new(&url, size_a).expect("create first CEF backend");
    // The second backend must join the same context instead of failing on the
    // old process-singleton gate.
    let mut b = CefBackend::new(&url, size_b).expect("create second CEF backend");

    let deadline = Instant::now() + Duration::from_secs(12);
    let (mut a_painted, mut b_painted) = (false, false);
    while Instant::now() < deadline && !(a_painted && b_painted) {
        a.tick(1.0 / 60.0);
        b.tick(1.0 / 60.0);
        if let Some(f) = a.latest_frame() {
            a_painted = !is_blank(f.data);
            assert_eq!((f.width, f.height), (1280, 720), "A paints at its own size");
        }
        if let Some(f) = b.latest_frame() {
            b_painted = !is_blank(f.data);
            assert_eq!((f.width, f.height), (800, 600), "B paints at its own size");
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    assert!(a_painted, "first browser never painted a non-blank frame");
    assert!(b_painted, "second browser never painted a non-blank frame");

    // Closing A must not tear the shared context down: B keeps painting.
    a.shutdown();
    let mut b_after = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !b_after {
        b.tick(1.0 / 60.0);
        if let Some(f) = b.latest_frame() {
            b_after = !is_blank(f.data);
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    assert!(b_after, "second browser stopped painting after the first shut down");
    b.shutdown();
}

/// A frame is "blank" if every pixel is identical (uniform colour, e.g. the
/// transparent/white default before the page renders anything).
fn is_blank(data: &[u8]) -> bool {
    if data.len() < 8 {
        return true;
    }
    let first = &data[0..4];
    data.as_chunks::<4>().0.iter().all(|px| px == first)
}
