//! kirie-web — web wallpapers via an off-screen (headless) browser.
//!
//! A "web" wallpaper is an HTML/CSS/JS bundle (`project.json` `"type":"web"`,
//! entry `index.html`) rendered by an embedded browser in **windowless /
//! off-screen** mode: the browser paints into a CPU buffer, which the GPU side
//! uploads to a texture and blits fullscreen. See docs/subsystems-misc.md §3
//! (WebBrowser / CEF) for the reference C++ behaviour this ports.
//!
//! # Layout
//!
//! * [`backend`] — the engine-neutral [`WebBackend`] trait plus its frame /
//!   pointer / error types. Shared by every browser backend.
//! * [`renderer`] — [`WebRenderer`], a [`kirie_platform::Renderer`] that
//!   presents whatever a [`WebBackend`] paints.
//! * `cef` (feature `cef`) — the Chromium Embedded Framework OSR backend
//!   ([`cef::CefBackend`]) and the JS shim WE web wallpapers expect. Heavy
//!   (downloads + cmake-builds libcef); **compiled only with `--features
//!   cef`** so the default `cargo build --workspace` stays green on machines
//!   without libcef.
//!
//! # Safety (SPEC §V2)
//!
//! The default build carries `#![forbid(unsafe_code)]`. The `cef` module
//! cannot: CEF is a C ABI and every callback/handoff is `unsafe`. That module
//! locally relaxes the ban and `// SAFETY`-annotates each FFI touch; the ban
//! stays in force for the rest of the crate.

#![cfg_attr(not(any(feature = "cef", feature = "webview")), forbid(unsafe_code))]

pub mod backend;
pub mod renderer;
pub mod shim;

#[cfg(feature = "cef")]
pub mod cef;

/// The wry + system-`webkit2gtk` native-surface backend (feature `webview`).
///
/// Unlike [`cef`], this backend renders straight into a host-provided
/// background window instead of an off-screen buffer, so it does *not*
/// implement the [`WebBackend`] trait (a `wry::WebView` is `!Send` and yields
/// no CPU frame); see [`webview`] for the model and its `webkit2gtk-4.1` build
/// requirement. The `unsafe` it needs (borrowing raw window handles) is why the
/// crate-level `forbid(unsafe_code)` is relaxed for this feature too.
#[cfg(feature = "webview")]
pub mod webview;

pub use backend::{FrameBuffer, PixelFormat, PointerState, WebBackend, WebError, WebFrameRef, WebSize};
pub use renderer::WebRenderer;
