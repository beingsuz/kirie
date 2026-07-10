//! kirie-platform — wayland + X11 presentation layer.
//!
//! Two backends sit behind one [`Platform`] facade and drive the same
//! [`Renderer`] contract over a shared output/surface model:
//!
//! - **Wayland** ([`platform::WaylandPlatform`]): ports the C++
//!   `WaylandOpenGLDriver`/`WaylandOutputViewport` model
//!   (docs/render-architecture.md §2.3) — see below.
//! - **X11** ([`x11::X11Platform`]): ports the C++ GLFW/X11 root-window path
//!   (docs/render-architecture.md §2.2) — one override-redirect window per
//!   RANDR CRTC, a `_NET_WM_WINDOW_TYPE_DESKTOP` background for the wallpaper
//!   (behind normal windows) or a plain window for `--window` (SPEC T24).
//!
//! [`Platform::connect`] picks the backend from the environment (Wayland when
//! `$WAYLAND_DISPLAY` is set, else X11); [`Platform::connect_backend`] forces
//! one. The Wayland model below is unchanged:
//!
//!
//! - one `zwlr_layer_shell_v1` surface per output (layer `background`,
//!   anchored to all edges, exclusive zone -1, no keyboard interactivity,
//!   full output size assigned via configure),
//! - a single shared wgpu device with one swapchain per output,
//! - rendering driven exclusively by `wl_surface.frame` callbacks — no
//!   busy loop (SPEC V6 groundwork), each output at its own
//!   compositor-driven cadence,
//! - output hotplug via `wl_output` global add/remove.
//!
//! The app supplies a [`Renderer`] per output through a
//! [`RendererFactory`]; [`TestPattern`] is a built-in renderer proving the
//! full stack (used by `examples/layer_clear.rs`).
//!
//! SPEC V2: `unsafe` is allowed in this crate solely for raw-window-handle
//! surface creation; it is confined to one function per backend
//! (`src/gpu.rs` for Wayland, `src/x11.rs` for X11) and denied everywhere
//! else.

#![deny(unsafe_code)]

mod backend;
mod error;
mod gpu;
mod output;
mod platform;
mod renderer;
mod test_pattern;
mod x11;

pub use backend::{Backend, Platform, PresentOptions};
pub use error::PlatformError;
pub use renderer::{RenderTarget, Renderer, RendererFactory, SurfaceSize};
pub use test_pattern::TestPattern;
pub use x11::X11Mode;
