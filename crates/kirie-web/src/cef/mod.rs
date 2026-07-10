//! Chromium Embedded Framework off-screen-rendering backend (feature `cef`).
//!
//! Ports the reference engine's `WebBrowser` subsystem
//! (docs/subsystems-misc.md §3): a windowless (OSR) CEF browser paints the
//! wallpaper's `index.html` into a BGRA buffer that [`CefBackend`] publishes
//! for the GPU side to blit. The JS bridge WE web wallpapers expect lives in
//! the backend-neutral [`crate::shim`] module and is injected here in the
//! render process.
//!
//! # Modules
//!
//! * [`app`] — the process-wide [`cef::App`]: Chromium command-line flags
//!   (browser process) and the shim-injecting render-process handler.
//! * [`client`] — the OSR [`cef::Client`] + [`cef::RenderHandler`] whose
//!   `on_paint` publishes frames into the lock-free slot.
//! * [`backend`] — [`CefBackend`], the [`crate::WebBackend`] that owns the CEF
//!   thread, the message-loop pump, and browser lifecycle.
//!
//! # Safety (SPEC §V2)
//!
//! CEF is a C ABI: constructing settings, initializing, and every handler
//! callback cross an `unsafe` boundary. Each is isolated behind a small
//! function and carries a `// SAFETY` note; nothing outside this module needs
//! `unsafe`.

pub mod app;
pub mod backend;
pub mod client;

pub use backend::CefBackend;
