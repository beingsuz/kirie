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

    /// Apply a live property override (socket `property <key> <value>`, doc
    /// §4.9): update the running wallpaper in place so the change shows on the
    /// next frame — no reload. `value` is the raw string; the renderer parses it
    /// to the property's declared type. Default: no-op (renderers with no live
    /// properties, e.g. image/video/black, ignore it).
    fn set_property(&mut self, _key: &str, _value: &str) {}
}

/// Factory invoked once per output surface to build its [`Renderer`].
pub type RendererFactory = Box<dyn FnMut(&RenderTarget<'_>) -> Box<dyn Renderer>>;

/// Builds a renderer off the render thread, given a cloned GPU device+queue and
/// the target output's surface format + name. `Send` so a worker thread can run
/// it; the produced renderer is `Box<dyn Renderer + Send>` so it can be handed
/// back to the render thread. The app supplies this (it owns the build logic);
/// the platform just runs it on a worker and swaps the result in.
pub type BuildFn = Box<
    dyn FnOnce(&wgpu::Device, &wgpu::Queue, wgpu::TextureFormat, &str) -> Box<dyn Renderer + Send>
        + Send,
>;

/// Builds a renderer **on the render thread**, given a cloned GPU device+queue
/// and the target output's surface format + name. The closure is `Send` (so it
/// can ride the command channel) but its output need not be — this is for
/// `!Send` backends like the CEF web renderer, which must be created and live on
/// the render thread and therefore can't use the off-thread [`BuildFn`]. Running
/// it blocks the render loop for the build's duration (a brief hitch), so it's
/// reserved for backends that genuinely cannot build off-thread.
pub type BuildLocalFn = Box<
    dyn FnOnce(&wgpu::Device, &wgpu::Queue, wgpu::TextureFormat, &str) -> Box<dyn Renderer> + Send,
>;

/// Captures the current frame of an already-built, warm renderer to disk, on the
/// render thread. Given the shared device+queue, the live output's current
/// renderer, its physical size and surface format, the app renders one frame to
/// an offscreen texture (pipelines match because the format matches the live
/// surface), reads it back and writes the image. `Send` so it can ride the
/// command channel from the IPC applier; the app supplies it (it owns the
/// readback/encode logic and the `image` dependency).
pub type CaptureFn = Box<
    dyn FnOnce(&wgpu::Device, &wgpu::Queue, &mut dyn Renderer, SurfaceSize, wgpu::TextureFormat)
        + Send,
>;

/// Clone-able sender for [`RenderCommand`]s into the render thread's channel.
pub type CommandSender =
    smithay_client_toolkit::reexports::calloop::channel::Sender<RenderCommand>;

/// A command delivered to the render thread over the platform's command channel
/// (sent by another thread, e.g. the IPC applier). Applied between frames on the
/// render thread — no lock, no surface sharing.
pub enum RenderCommand {
    /// Build a new wallpaper for output `screen` on a worker thread, then install
    /// it (async `bg` swap). The current wallpaper keeps rendering until ready.
    /// `stash` = `Some(key)` preloads it (stash, don't show) for a later instant
    /// [`RenderCommand::Swap`]; `None` shows it as soon as it's built.
    Build {
        /// Target output name (`--screen-root`); `"*"` = every output.
        screen: String,
        /// `Some(key)` = preload (stash); `None` = install when built.
        stash: Option<String>,
        /// The app's off-thread builder.
        build: BuildFn,
    },
    /// Show a wallpaper: if one is stashed under `key` for `screen` (a preload
    /// hit) install it instantly — a sub-100ms pointer swap. On a miss (nothing
    /// preloaded, or the preload is still building), fall back to building
    /// `build` off-thread and installing it when ready.
    Swap {
        /// Target output name; `"*"` = every output.
        screen: String,
        /// Preload key to look up.
        key: String,
        /// Fallback builder used on a preload miss.
        build: BuildFn,
    },
    /// Internal: a worker finished building — install (swap) it on `screen`.
    Install {
        /// Target output name; `"*"` = every output.
        screen: String,
        /// Preload key if this was a preload build (stash instead of show).
        stash: Option<String>,
        /// The freshly built renderer.
        renderer: Box<dyn Renderer + Send>,
    },
    /// Capture the current frame of `screen`'s live renderer to disk (socket
    /// `screenshot`, doc §4.12). Runs `capture` on the render thread with the
    /// output's warm renderer, physical size and surface format — a one-shot
    /// offscreen re-render + readback, so the palette source matches what is on
    /// screen (property overrides included), not the workshop preview.
    Screenshot {
        /// Target output name; `"*"` = the first output (per-monitor daemon).
        screen: String,
        /// The app's render-thread capture-to-disk closure.
        capture: CaptureFn,
    },
    /// Build a renderer on the render thread and swap it in — for `!Send`
    /// backends (CEF web) that must be created and live on the render thread and
    /// so can't use the off-thread [`RenderCommand::Swap`]. The build runs inline
    /// on the render thread: a brief hitch on the current wallpaper while it
    /// initializes (e.g. CEF's first browser init), then the new one installs.
    /// No black gap, no process relaunch.
    SwapLocal {
        /// Target output name; `"*"` = the first output.
        screen: String,
        /// Render-thread builder producing the (possibly `!Send`) renderer.
        build_local: BuildLocalFn,
    },
    /// Apply a live property override to `screen`'s running renderer (socket
    /// `property`, doc §4.9): call [`Renderer::set_property`] and repaint, so the
    /// change shows next frame — no reload.
    SetProperty {
        /// Target output name; `"*"` = the first output.
        screen: String,
        /// Property key.
        key: String,
        /// Raw value string (the renderer parses it to the declared type).
        value: String,
    },
}
