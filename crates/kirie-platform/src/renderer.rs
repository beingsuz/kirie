//! The renderer contract the presentation layer drives.
//!
//! Mirrors the C++ split between the video driver (owns surfaces, decides
//! *when* to draw) and the wallpaper renderer (decides *what* to draw):
//! `RenderContext::render(viewport)` dispatches into the wallpaper per
//! frame callback (docs/render-architecture.md §2.3, §2.4). Here the driver
//! is [`crate::Platform`] and the wallpaper is a [`Renderer`] supplied by
//! the app, so the P3/P4 renderers can plug in without touching this crate
//! (SPEC V1: no globals, state passed explicitly).

/// Physical size of a surface in pixels.
///
/// The C++ wayland viewport is `{0, 0, w*scale, h*scale}` — physical
/// pixels, logical size multiplied by the integer output scale
/// (docs/render-architecture.md §2.3, WaylandOutputViewport.cpp:19-27).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceSize {
    /// Width in physical pixels (logical width × buffer scale).
    pub width: u32,
    /// Height in physical pixels (logical height × buffer scale).
    pub height: u32,
}

/// Everything a [`Renderer`] needs to build its GPU resources for one
/// output surface.
///
/// One renderer instance is created per output; the device/queue are shared
/// across outputs (docs/render-architecture.md §2.3 "wgpu:" note — one
/// shared device, one present pass per monitor surface).
pub struct RenderTarget<'a> {
    /// Shared wgpu device (cheap to clone; internally reference-counted).
    pub device: &'a wgpu::Device,
    /// Shared wgpu queue.
    pub queue: &'a wgpu::Queue,
    /// The swapchain texture format render pipelines must target.
    pub format: wgpu::TextureFormat,
    /// Compositor-reported output name (e.g. `DP-1`), if known.
    pub output_name: &'a str,
}

/// A per-output frame producer driven by compositor frame callbacks.
///
/// `render` is called exactly once per `wl_surface` frame callback — never
/// from a busy loop (docs/render-architecture.md §2.3: rendering is
/// frame-callback driven per output; SPEC V6 groundwork). Implementations
/// encode and submit their own command buffers; presentation is handled by
/// the caller after `render` returns.
pub trait Renderer {
    /// Draw one frame into `view`.
    ///
    /// `size` is the current physical size of the surface texture and `dt`
    /// is the seconds elapsed since this output's previous frame (`0.0` on
    /// the first frame). Per-output timing mirrors the C++ driver, where
    /// each monitor renders at its own compositor-driven cadence
    /// (docs/render-architecture.md §2.3).
    fn render(&mut self, view: &wgpu::TextureView, size: SurfaceSize, dt: f32);
}

/// Factory invoked once per output surface to build its [`Renderer`].
pub type RendererFactory = Box<dyn FnMut(&RenderTarget<'_>) -> Box<dyn Renderer>>;
