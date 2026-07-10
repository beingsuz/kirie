//! The `webview` web-wallpaper backend: wry + the system `webkit2gtk-4.1`.
//!
//! # Rendering model — native surface, not off-screen
//!
//! The CEF backend ([`crate::cef`]) renders **off-screen**: Chromium paints
//! into a CPU pixel buffer that kirie uploads to a wgpu texture and composites
//! like any other wallpaper (the [`crate::backend::WebBackend`] trait is shaped
//! around that — `new(url, size)` + `latest_frame()`).
//!
//! wry cannot do that. On Linux wry is a thin wrapper over `webkit2gtk`, which
//! renders with its **own** GTK/EGL context straight into a native window and
//! exposes **no** CPU pixel read-back (there is no OSR path, and no synchronous
//! snapshot API on wry 0.55). A `wry::WebView` is also `!Send` — it must be
//! created and used on the GTK main thread. Both facts make it impossible to
//! implement the off-screen, `Send`, frame-publishing `WebBackend` trait.
//!
//! So this backend uses the only model webkit2gtk supports for a full-surface
//! HTML wallpaper: **it fills the background window directly.** The host
//! (kirie-platform) creates the background surface — on Wayland a layer-shell
//! surface (a GTK window promoted to the background layer via `gtk-layer-shell`),
//! on X11 the desktop window — and hands the backend its
//! [`raw_window_handle`] handles via [`SurfaceTarget`]. webkit renders into it;
//! there is no wgpu compositing step and no per-frame `latest_frame()` upload
//! for web wallpapers on this backend. This matches how WE web wallpapers are
//! authored: standalone full-surface `index.html` documents.
//!
//! Everything else is kept identical to the CEF backend: the same WE JS bridge
//! shims ([`crate::shim`]) are injected, audio is pushed to
//! `wallpaperRegisterAudioListener`, properties flow through
//! `wallpaperPropertyListener`, and audio can be muted (via JavaScript, since
//! wry exposes no native mute).
//!
//! # Build requirement
//!
//! The `webview` feature pulls `wry`, whose Linux backend links
//! `webkit2gtk-4.1` **and** `libsoup-3.0` (+ GTK 3) via `pkg-config` at
//! **compile** time. On a box without `webkit2gtk-4.1` installed this crate
//! cannot be compiled with `--features webview` — that is expected; CI installs
//! the package. The default build enables no web feature and never references
//! wry, so it stays green on such machines (SPEC invariant).
//!
//! # Runtime preconditions (host responsibility)
//!
//! wry 0.55's `WebViewBuilder::build` on Linux **panics** if `gtk::init()` was
//! not called on the current thread, and a `wry::WebView` is `!Send` — it must
//! be created and driven on the one GTK main thread. So kirie-platform must,
//! on the thread that owns the web wallpaper: call `gtk::init()` once, create
//! the background GTK/gtk-layer-shell window, then build the
//! [`WebviewBackend`] from its handles and run the GTK main loop (that loop —
//! not [`WebviewBackend::tick`] — is what actually paints webkit). These are
//! host misconfigurations, not page input, so this crate documents rather than
//! defends against them (SPEC V9 concerns malformed *wallpaper* input).

mod backend;
mod surface;

pub use backend::WebviewBackend;
pub use surface::SurfaceTarget;

/// Turn a filesystem path to a wallpaper entry page into a `file://` URL.
///
/// webkit2gtk resolves the page's relative asset references against the URL's
/// directory, so the entry page must be given as an absolute `file://` URL
/// (docs/subsystems-misc.md §3.4). The path is percent-encoded per RFC 3986
/// for the bytes that are unsafe in a URL path, leaving `/` as the separator.
/// The path is **not** canonicalized here (that is I/O the caller may have
/// already done); pass an absolute path.
#[must_use]
pub fn file_url(path: &std::path::Path) -> String {
    use std::path::Component;

    let mut url = String::from("file://");
    for comp in path.components() {
        match comp {
            Component::RootDir => { /* leading '/' emitted below per-segment */ }
            Component::Prefix(_) => { /* Windows prefixes: not a target platform */ }
            Component::CurDir => {}
            Component::ParentDir => {
                url.push('/');
                url.push_str("..");
            }
            Component::Normal(seg) => {
                url.push('/');
                encode_segment(&seg.to_string_lossy(), &mut url);
            }
        }
    }
    if url == "file://" {
        // Root path.
        url.push('/');
    }
    url
}

/// Percent-encode one path segment into `out`, leaving RFC 3986 unreserved
/// characters (and a small safe set) untouched.
fn encode_segment(seg: &str, out: &mut String) {
    for &b in seg.as_bytes() {
        let safe = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~');
        if safe {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_digit(b >> 4));
            out.push(hex_digit(b & 0x0f));
        }
    }
}

fn hex_digit(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'A' + (n - 10)) as char,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn encodes_spaces_and_keeps_slashes() {
        let url = file_url(Path::new("/home/a b/My Wallpaper/index.html"));
        assert_eq!(url, "file:///home/a%20b/My%20Wallpaper/index.html");
    }

    #[test]
    fn keeps_unreserved() {
        let url = file_url(Path::new("/a-b_c.d~e/index.html"));
        assert_eq!(url, "file:///a-b_c.d~e/index.html");
    }
}
