//! X11 root-window / desktop presentation backend.
//!
//! Ports the C++ GLFW/X11 driver path (docs/render-architecture.md §2.2):
//! monitors come from XRandR CRTC geometry (one viewport per connected CRTC,
//! X11Output.cpp:111-159), and the wallpaper is drawn *behind* normal windows.
//!
//! The C++ path renders into a hidden full-screen GL surface, reads the pixels
//! back with `glReadPixels`, `XPutImage`s them into a root-sized pixmap and
//! points `_XROOTPMAP_ID`/`ESETROOT_PMAP_ID` at it (X11Output.cpp:226-246).
//! That readback-to-pixmap dance exists because GLFW cannot present GL to the
//! root window directly. wgpu can present straight to an X drawable via
//! `VK_KHR_xcb_surface`, so kirie instead creates one real
//! **override-redirect window per CRTC**, typed `_NET_WM_WINDOW_TYPE_DESKTOP`
//! and lowered to the bottom of the stack, and presents GPU frames into it.
//! Net effect matches the C++ intent — a per-monitor background behind every
//! window — without the CPU readback (docs/render-architecture.md §2.2; this
//! is the documented translation, SPEC V10). `--window` mode drops the desktop
//! typing/lowering and maps a single ordinary override-redirect window.
//!
//! SPEC V2: the sole `unsafe` here is the raw-handle wgpu surface creation in
//! [`create_x11_surface`], mirroring the Wayland path in `src/gpu.rs`.

use std::num::NonZeroU32;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use raw_window_handle::{RawDisplayHandle, RawWindowHandle, XcbDisplayHandle, XcbWindowHandle};
use x11rb::connection::Connection as _;
use x11rb::protocol::Event;
use x11rb::protocol::randr::ConnectionExt as _;
use x11rb::protocol::xproto::{
    AtomEnum, ConfigureWindowAux, ConnectionExt as _, CreateWindowAux, EventMask, PropMode, StackMode,
    Window, WindowClass,
};
use x11rb::wrapper::ConnectionExt as _;
use x11rb::xcb_ffi::XCBConnection;

use crate::error::PlatformError;
use crate::gpu::Gpu;
use crate::renderer::{RenderTarget, Renderer, RendererFactory, SurfaceSize};

/// How the X11 wallpaper window is presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X11Mode {
    /// One `_NET_WM_WINDOW_TYPE_DESKTOP` override-redirect window per CRTC,
    /// lowered behind normal windows — the wallpaper mode
    /// (docs/render-architecture.md §2.2).
    Desktop,
    /// A single ordinary override-redirect window of the given size, for the
    /// `--window` preview mode (docs/compat-cli.md `--window`).
    Window {
        /// Window width in pixels.
        width: u32,
        /// Window height in pixels.
        height: u32,
    },
}

/// A monitor rectangle in the X screen's pixel coordinate space
/// (docs/render-architecture.md §2.2: `GLFWOutputViewport{crtc.x, crtc.y,
/// crtc.width, crtc.height}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MonitorGeometry {
    /// Left edge in root coordinates.
    pub x: i32,
    /// Top edge in root coordinates.
    pub y: i32,
    /// Width in pixels (always ≥ 1).
    pub width: u32,
    /// Height in pixels (always ≥ 1).
    pub height: u32,
}

/// Minimal projection of a RANDR `GetCrtcInfo` reply — just the fields that
/// decide whether a CRTC is an active monitor and where it sits. Kept
/// separate from x11rb's reply type so [`active_monitors`] is a pure function
/// unit-testable with synthetic input (no live X server).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawCrtc {
    /// CRTC top-left x (root coords).
    pub x: i16,
    /// CRTC top-left y (root coords).
    pub y: i16,
    /// CRTC width in pixels.
    pub width: u16,
    /// CRTC height in pixels.
    pub height: u16,
    /// Active mode id; `0` means the CRTC is disabled.
    pub mode: u32,
    /// Number of outputs (monitors) driven by this CRTC; `0` means nothing is
    /// connected.
    pub connected_outputs: u32,
}

/// Reduce raw CRTC info to the set of active monitor rectangles
/// (docs/render-architecture.md §2.2: one viewport per *connected* CRTC).
///
/// A CRTC is active iff it has a mode set, at least one connected output, and
/// non-zero extent. Results are sorted by `(x, y)` so window/output ordering
/// is deterministic across runs (RANDR returns CRTCs in arbitrary order).
pub(crate) fn active_monitors(crtcs: &[RawCrtc]) -> Vec<MonitorGeometry> {
    let mut monitors: Vec<MonitorGeometry> = crtcs
        .iter()
        .filter(|c| c.mode != 0 && c.connected_outputs != 0 && c.width != 0 && c.height != 0)
        .map(|c| MonitorGeometry {
            x: i32::from(c.x),
            y: i32::from(c.y),
            width: u32::from(c.width),
            height: u32::from(c.height),
        })
        .collect();
    monitors.sort_by_key(|m| (m.x, m.y));
    monitors
}

/// Everything owned per X11 monitor: its window, swapchain surface, and lazily
/// built renderer. Field order is load-bearing — `wgpu_surface` (which borrows
/// the raw xcb connection + window) is declared before everything so it drops
/// before the connection closes (see [`X11Platform`] field order).
struct X11Output {
    /// wgpu swapchain surface over this window (`None` if creation failed).
    wgpu_surface: Option<wgpu::Surface<'static>>,
    /// The X11 window id backing the surface.
    window: Window,
    /// Human-readable name for logs (`X11-0`, …).
    name: String,
    /// Current swapchain extent in pixels.
    physical_size: SurfaceSize,
    /// Swapchain configured at least once.
    configured: bool,
    /// App-supplied frame producer; built lazily once the GPU exists.
    renderer: Option<Box<dyn Renderer>>,
    /// Timestamp of the previous presented frame, for per-monitor `dt`.
    last_frame: Option<Instant>,
    /// Whether the first frame was logged.
    first_frame_presented: bool,
}

/// The X11 presentation layer: owns the xcb connection, one window+surface per
/// monitor, and the shared GPU context (SPEC V1: owned here, nothing global).
///
/// Field order is load-bearing for drop safety: `outputs` (holding
/// `wgpu::Surface`s over raw xcb pointers) and `gpu` are declared before
/// `conn`, so every surface is destroyed before the xcb connection closes.
/// The X server frees the client's windows automatically when `conn` drops, so
/// no explicit window teardown is needed.
pub struct X11Platform {
    outputs: Vec<X11Output>,
    gpu: Gpu,
    conn: XCBConnection,
    make_renderer: RendererFactory,
    /// Target ~60 FPS present cadence, matching the C++ `usleep` FPS cap
    /// (docs/render-architecture.md §2.2 step 7).
    frame_interval: Duration,
}

impl X11Platform {
    /// Connect to `$DISPLAY`, enumerate monitors via RANDR, and bring up a
    /// window + wgpu surface per monitor (or a single window in
    /// [`X11Mode::Window`]).
    pub(crate) fn connect(mode: X11Mode, make_renderer: RendererFactory) -> Result<Self, PlatformError> {
        let (conn, screen_num) =
            XCBConnection::connect(None).map_err(|e| PlatformError::X11Connect(e.to_string()))?;

        let setup = conn.setup();
        let screen = setup
            .roots
            .get(screen_num)
            .ok_or_else(|| PlatformError::X11Connect(format!("no screen {screen_num}")))?;
        let root = screen.root;
        let depth = screen.root_depth;
        let visual = screen.root_visual;
        let black = screen.black_pixel;

        // Monitor geometry: RANDR CRTCs for desktop mode, a single synthetic
        // rectangle for --window (docs/render-architecture.md §2.2).
        let geometries = match mode {
            X11Mode::Desktop => query_monitors(&conn, root)?,
            X11Mode::Window { width, height } => vec![MonitorGeometry {
                x: 0,
                y: 0,
                width: width.max(1),
                height: height.max(1),
            }],
        };
        if geometries.is_empty() {
            return Err(PlatformError::NoCrtcs);
        }

        // Create every window up front (windows only need the connection);
        // the GPU is brought up against the first window's surface.
        let mut windows = Vec::with_capacity(geometries.len());
        for geom in &geometries {
            let wid = create_window(&conn, root, depth, visual, black, *geom, mode)?;
            windows.push((wid, *geom));
        }
        conn.flush()
            .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

        // Bring up wgpu against the first window, then create surfaces for the
        // rest with the same instance (a surface is only compatible with
        // adapters from the instance that made it — same rule as Wayland,
        // src/gpu.rs).
        let (gpu, first_surface) = bring_up_gpu(&conn, screen_num, windows[0].0)?;
        let mut first_surface = Some(first_surface);

        let mut outputs = Vec::with_capacity(windows.len());
        for (idx, (wid, geom)) in windows.iter().enumerate() {
            let wgpu_surface = if idx == 0 {
                first_surface.take()
            } else {
                match create_x11_surface(&gpu.instance, &conn, screen_num, *wid) {
                    Ok(s) => Some(s),
                    Err(err) => {
                        tracing::error!(window = wid, %err, "x11 surface creation failed");
                        None
                    }
                }
            };
            outputs.push(X11Output {
                wgpu_surface,
                window: *wid,
                name: format!("X11-{idx}"),
                physical_size: SurfaceSize {
                    width: geom.width.max(1),
                    height: geom.height.max(1),
                },
                configured: false,
                renderer: None,
                last_frame: None,
                first_frame_presented: false,
            });
        }

        let mut platform = Self {
            outputs,
            gpu,
            conn,
            make_renderer,
            frame_interval: Duration::from_micros(16_666),
        };

        for index in 0..platform.outputs.len() {
            platform.configure_swapchain(index);
        }
        platform
            .conn
            .flush()
            .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

        tracing::info!(monitors = platform.outputs.len(), ?mode, "x11 backend up");
        Ok(platform)
    }

    /// Number of monitors that currently have a window.
    #[must_use]
    pub(crate) fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// (Re)configure the swapchain of `outputs[index]` for its current size.
    fn configure_swapchain(&mut self, index: usize) {
        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        let Some(surface) = &ctx.wgpu_surface else {
            return;
        };
        let Some(mut config) = surface.get_default_config(
            &self.gpu.adapter,
            ctx.physical_size.width,
            ctx.physical_size.height,
        ) else {
            tracing::error!(output = %ctx.name, "adapter cannot present to this surface");
            return;
        };
        // Fifo is universally supported and vsync-paces the present loop,
        // standing in for the C++ usleep FPS cap
        // (docs/render-architecture.md §2.2 step 7).
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&self.gpu.device, &config);
        ctx.configured = true;
    }

    /// Render + present one frame for `outputs[index]`.
    fn draw(&mut self, index: usize) {
        let (device, queue) = (self.gpu.device.clone(), self.gpu.queue.clone());

        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        if !ctx.configured {
            return;
        }
        let Some(wgpu_surface) = &ctx.wgpu_surface else {
            return;
        };

        let mut texture = wgpu_surface.get_current_texture();
        if matches!(
            texture,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost
        ) {
            tracing::debug!(output = %ctx.name, "swapchain outdated/lost; reconfiguring");
            self.configure_swapchain(index);
            let Some(ctx) = self.outputs.get(index) else {
                return;
            };
            let Some(wgpu_surface) = &ctx.wgpu_surface else {
                return;
            };
            texture = wgpu_surface.get_current_texture();
        }

        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        let texture = match texture {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                tracing::debug!(output = %ctx.name, status = ?other, "skipping frame");
                return;
            }
        };

        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let renderer = ctx.renderer.get_or_insert_with(|| {
            (self.make_renderer)(&RenderTarget {
                device: &device,
                queue: &queue,
                format: texture.texture.format(),
                output_name: &ctx.name,
            })
        });

        let now = Instant::now();
        let dt = ctx
            .last_frame
            .map(|prev| now.duration_since(prev).as_secs_f32())
            .unwrap_or(0.0);
        ctx.last_frame = Some(now);

        renderer.render(&view, ctx.physical_size, dt);
        queue.present(texture);

        if !ctx.first_frame_presented {
            ctx.first_frame_presented = true;
            tracing::info!(
                output = %ctx.name,
                width = ctx.physical_size.width,
                height = ctx.physical_size.height,
                "first frame presented"
            );
        }
    }

    /// Handle one X event: keep window sizes in sync with the server.
    fn handle_event(&mut self, event: Event) {
        if let Event::ConfigureNotify(ev) = event {
            let new = SurfaceSize {
                width: u32::from(ev.width).max(1),
                height: u32::from(ev.height).max(1),
            };
            if let Some(index) = self.outputs.iter().position(|o| o.window == ev.window)
                && self.outputs[index].physical_size != new
            {
                self.outputs[index].physical_size = new;
                self.configure_swapchain(index);
            }
        }
    }

    /// Drive the present loop until `duration` elapses (`None` = forever).
    ///
    /// Unlike Wayland's frame-callback model, X11 has no compositor-driven
    /// cadence, so this mirrors the C++ GLFW/X11 driver: an unthrottled loop
    /// with an FPS cap (docs/render-architecture.md §2.2). Fifo present vsync-
    /// paces each surface; the trailing sleep enforces the cap when present
    /// returns early (e.g. no monitor attached).
    pub(crate) fn run(&mut self, duration: Option<Duration>) -> Result<(), PlatformError> {
        let deadline = duration.map(|d| Instant::now() + d);

        loop {
            let frame_start = Instant::now();

            while let Some(event) = self
                .conn
                .poll_for_event()
                .map_err(|e| PlatformError::X11Protocol(e.to_string()))?
            {
                self.handle_event(event);
            }

            if let Some(deadline) = deadline
                && Instant::now() >= deadline
            {
                break;
            }

            for index in 0..self.outputs.len() {
                self.draw(index);
            }

            let elapsed = frame_start.elapsed();
            if elapsed < self.frame_interval {
                std::thread::sleep(self.frame_interval - elapsed);
            }
        }

        tracing::info!("x11 run deadline reached; tearing down windows");
        Ok(())
    }
}

/// Enumerate active monitors from RANDR (docs/render-architecture.md §2.2:
/// XRandR CRTCs, X11Output.cpp:111-159).
fn query_monitors(conn: &XCBConnection, root: Window) -> Result<Vec<MonitorGeometry>, PlatformError> {
    let resources = conn
        .randr_get_screen_resources_current(root)
        .map_err(|e| PlatformError::X11Protocol(e.to_string()))?
        .reply()
        .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

    let mut raw = Vec::with_capacity(resources.crtcs.len());
    for crtc in resources.crtcs {
        let info = match conn
            .randr_get_crtc_info(crtc, resources.config_timestamp)
            .map_err(|e| PlatformError::X11Protocol(e.to_string()))?
            .reply()
        {
            Ok(info) => info,
            // A CRTC can vanish between the resource list and the info query;
            // skip it rather than fail the whole enumeration (SPEC V9).
            Err(err) => {
                tracing::debug!(crtc, %err, "skipping crtc with no info");
                continue;
            }
        };
        raw.push(RawCrtc {
            x: info.x,
            y: info.y,
            width: info.width,
            height: info.height,
            mode: info.mode,
            connected_outputs: u32::try_from(info.outputs.len()).unwrap_or(u32::MAX),
        });
    }

    Ok(active_monitors(&raw))
}

/// Create one wallpaper window for `geom` (docs/render-architecture.md §2.2).
fn create_window(
    conn: &XCBConnection,
    root: Window,
    depth: u8,
    visual: u32,
    background: u32,
    geom: MonitorGeometry,
    mode: X11Mode,
) -> Result<Window, PlatformError> {
    let wid = conn
        .generate_id()
        .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

    let aux = CreateWindowAux::new()
        .background_pixel(background)
        // Override-redirect so no WM decorates or repositions the wallpaper
        // (both desktop and --window; the C++ GLFW window is likewise
        // unmanaged, docs/render-architecture.md §2.2).
        .override_redirect(1u32)
        .event_mask(EventMask::EXPOSURE | EventMask::STRUCTURE_NOTIFY);

    conn.create_window(
        depth,
        wid,
        root,
        i16::try_from(geom.x).unwrap_or(0),
        i16::try_from(geom.y).unwrap_or(0),
        u16::try_from(geom.width).unwrap_or(u16::MAX),
        u16::try_from(geom.height).unwrap_or(u16::MAX),
        0,
        WindowClass::INPUT_OUTPUT,
        visual,
        &aux,
    )
    .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

    // WM_NAME so the window is identifiable in tooling.
    conn.change_property8(
        PropMode::REPLACE,
        wid,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"kirie",
    )
    .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

    if matches!(mode, X11Mode::Desktop) {
        // _NET_WM_WINDOW_TYPE = _NET_WM_WINDOW_TYPE_DESKTOP so EWMH-aware
        // compositors keep the wallpaper behind everything (the modern
        // equivalent of the C++ root-pixmap behavior,
        // docs/render-architecture.md §2.2).
        let type_atom = intern(conn, b"_NET_WM_WINDOW_TYPE")?;
        let desktop_atom = intern(conn, b"_NET_WM_WINDOW_TYPE_DESKTOP")?;
        conn.change_property32(PropMode::REPLACE, wid, type_atom, AtomEnum::ATOM, &[desktop_atom])
            .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;
    }

    conn.map_window(wid)
        .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;

    if matches!(mode, X11Mode::Desktop) {
        // Lower to the bottom of the stack as a belt-and-braces guarantee for
        // WMs that ignore the DESKTOP type hint.
        conn.configure_window(wid, &ConfigureWindowAux::new().stack_mode(StackMode::BELOW))
            .map_err(|e| PlatformError::X11Protocol(e.to_string()))?;
    }

    Ok(wid)
}

/// Intern an atom, returning it as a plain `u32`.
fn intern(conn: &XCBConnection, name: &[u8]) -> Result<u32, PlatformError> {
    Ok(conn
        .intern_atom(false, name)
        .map_err(|e| PlatformError::X11Protocol(e.to_string()))?
        .reply()
        .map_err(|e| PlatformError::X11Protocol(e.to_string()))?
        .atom)
}

/// Bring up wgpu against the first window and return it plus that window's
/// swapchain surface. Vulkan preferred, falling back to all backends — the
/// same policy as the Wayland path (src/gpu.rs).
fn bring_up_gpu(
    conn: &XCBConnection,
    screen_num: usize,
    window: Window,
) -> Result<(Gpu, wgpu::Surface<'static>), PlatformError> {
    let mut last_err: Option<PlatformError> = None;

    for backends in [wgpu::Backends::VULKAN, wgpu::Backends::all()] {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = match create_x11_surface(&instance, conn, screen_num, window) {
            Ok(surface) => surface,
            Err(err) => {
                tracing::warn!(?backends, %err, "x11 surface creation failed on backend set");
                last_err = Some(err);
                continue;
            }
        };

        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..wgpu::RequestAdapterOptions::default()
        })) {
            Ok(adapter) => {
                let info = adapter.get_info();
                tracing::info!(backend = %info.backend, adapter = %info.name, "selected gpu adapter");
                let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("kirie-platform-x11"),
                    ..wgpu::DeviceDescriptor::default()
                }))?;
                let gpu = Gpu {
                    instance,
                    adapter,
                    device,
                    queue,
                };
                return Ok((gpu, surface));
            }
            Err(err) => {
                tracing::warn!(?backends, %err, "no adapter for backend set");
                last_err = Some(err.into());
            }
        }
    }

    Err(last_err.unwrap_or(PlatformError::NoCrtcs))
}

/// The single `unsafe` of the X11 backend: wrap
/// [`wgpu::Instance::create_surface_unsafe`] over the libxcb connection
/// pointer and an X window id (SPEC V2, mirroring `src/gpu.rs`).
#[allow(unsafe_code)]
fn create_x11_surface(
    instance: &wgpu::Instance,
    conn: &XCBConnection,
    screen_num: usize,
    window: Window,
) -> Result<wgpu::Surface<'static>, PlatformError> {
    let raw = conn.get_raw_xcb_connection();
    let connection = NonNull::new(raw).ok_or(PlatformError::NullXcbConnection)?;
    let window = NonZeroU32::new(window)
        .ok_or_else(|| PlatformError::X11Protocol("window id was zero".to_string()))?;

    let screen = i32::try_from(screen_num).unwrap_or(0);
    let raw_display_handle = RawDisplayHandle::Xcb(XcbDisplayHandle::new(Some(connection), screen));
    let raw_window_handle = RawWindowHandle::Xcb(XcbWindowHandle::new(window));

    // SAFETY: `create_surface_unsafe` requires both raw handles to be valid
    // and to stay valid until the returned `Surface` is dropped.
    // - Validity: `connection` is the live `*mut xcb_connection_t` owned by
    //   `conn` (an `XCBConnection`, libxcb-backed) and was null-checked;
    //   `window` is a live X11 window id created against that same connection
    //   earlier in `connect`.
    // - Lifetime: the returned surface is stored in an `X11Output`, and
    //   `X11Platform` declares `outputs` before `conn`, so every surface is
    //   dropped before the xcb connection closes; the X server frees the
    //   windows when the connection closes, strictly after their surfaces are
    //   gone. A surface that never reaches an `X11Output` (the backend-
    //   fallback path in `bring_up_gpu` when no adapter is found) dies inside
    //   that function, strictly within the caller's borrow of `conn`. The
    //   `'static` return lifetime is sound under that ownership discipline.
    let surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_display_handle),
            raw_window_handle,
        })
    }?;

    Ok(surface)
}

#[cfg(test)]
mod tests {
    use super::{MonitorGeometry, RawCrtc, active_monitors};

    fn crtc(x: i16, y: i16, w: u16, h: u16, mode: u32, outputs: u32) -> RawCrtc {
        RawCrtc {
            x,
            y,
            width: w,
            height: h,
            mode,
            connected_outputs: outputs,
        }
    }

    #[test]
    fn single_active_crtc() {
        let got = active_monitors(&[crtc(0, 0, 1920, 1080, 42, 1)]);
        assert_eq!(
            got,
            vec![MonitorGeometry {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080
            }]
        );
    }

    #[test]
    fn disabled_crtcs_are_dropped() {
        let got = active_monitors(&[
            crtc(0, 0, 1920, 1080, 1, 1),    // active
            crtc(0, 0, 0, 0, 0, 0),          // unused (mode 0)
            crtc(1920, 0, 2560, 1440, 7, 0), // mode set but no output connected
            crtc(0, 0, 1280, 1024, 3, 1),    // active but zero-origin dup position
        ]);
        // Only the two CRTCs with a mode AND a connected output survive.
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|m| m.width > 0 && m.height > 0));
    }

    #[test]
    fn zero_extent_crtc_is_dropped() {
        // Mode + output set but zero pixels: not a real monitor.
        let got = active_monitors(&[crtc(0, 0, 0, 1080, 5, 1)]);
        assert!(got.is_empty());
    }

    #[test]
    fn multi_monitor_sorted_left_to_right() {
        // RANDR order is arbitrary; result must be deterministic by (x, y).
        let got = active_monitors(&[
            crtc(2560, 0, 1920, 1080, 1, 1),
            crtc(0, 0, 2560, 1440, 1, 1),
            crtc(4480, 0, 1080, 1920, 1, 1),
        ]);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].x, 0);
        assert_eq!(got[1].x, 2560);
        assert_eq!(got[2].x, 4480);
    }

    #[test]
    fn negative_origin_preserved() {
        // A monitor left-of-primary has negative x in root coordinates.
        let got = active_monitors(&[crtc(0, 0, 1920, 1080, 1, 1), crtc(-1080, 0, 1080, 1920, 1, 1)]);
        assert_eq!(got[0].x, -1080);
        assert_eq!(got[1].x, 0);
    }
}
