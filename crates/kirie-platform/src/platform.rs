//! Wayland presentation driver: output enumeration + hotplug, one
//! layer-shell surface per output, frame-callback-driven rendering
//! (docs/render-architecture.md §2.3).

use std::time::{Duration, Instant};

use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState, Region};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::calloop::EventLoop;
use smithay_client_toolkit::reexports::calloop::channel::{
    Event as CalloopEvent, Sender as CmdSender, channel,
};
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use std::collections::HashMap;

use crate::renderer::RenderCommand;
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface, LayerSurfaceConfigure,
};
use smithay_client_toolkit::{
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, registry_handlers,
};
use wayland_client::globals::registry_queue_init;
use wayland_client::protocol::{wl_output, wl_surface};
use wayland_client::{Connection, QueueHandle};

use crate::error::PlatformError;
use crate::gpu::Gpu;
use crate::output::OutputContext;
use crate::renderer::{RenderTarget, RendererFactory, SurfaceSize};

/// The wayland presentation layer: owns the compositor connection, all
/// per-output surfaces, and the shared GPU context (SPEC V1: everything
/// owned here, nothing global).
///
/// Selected by [`crate::Platform`] when a wayland session is present; the
/// X11 sibling is [`crate::x11::X11Platform`]. Both drive the same
/// [`crate::Renderer`] contract behind the shared output/surface model.
pub struct WaylandPlatform {
    event_loop: EventLoop<'static, PlatformState>,
    state: PlatformState,
    /// Clone-able sender for [`RenderCommand`]s applied on the render thread
    /// (live `bg` swap / preload); handed to the IPC applier.
    cmd_tx: CmdSender<RenderCommand>,
}

/// Handler state driven by the wayland event queue.
///
/// Field order is load-bearing for drop safety: `outputs` (which hold
/// `wgpu::Surface`s over raw libwayland pointers) and `gpu` are declared
/// before `conn`, so all surfaces are destroyed before the display
/// connection closes (see the SAFETY discussion in src/gpu.rs).
struct PlatformState {
    outputs: Vec<OutputContext>,
    gpu: Option<Gpu>,
    conn: Connection,
    qh: QueueHandle<PlatformState>,

    registry_state: RegistryState,
    output_state: OutputState,
    compositor_state: CompositorState,
    layer_shell: LayerShell,

    make_renderer: RendererFactory,
    /// Which layer to create surfaces on. The C++ driver defaults to
    /// `bottom` with the layer selectable via CLI
    /// (docs/render-architecture.md §2.3); kirie's presentation layer
    /// defaults to `background` and will expose the CLI selection in the
    /// compat layer (docs/compat-cli.md).
    layer: Layer,
    /// wlr-layer-shell surface namespace assigned to every surface. Must
    /// contain `wallpaperengine` so the daemon watchdog
    /// (`wallpaperengine.sh`, `engine_layer_ok()`:
    /// `any(.[][]?; .namespace|test("wallpaperengine"))`) recognises the
    /// live wallpaper and does not kill-restart the engine
    /// (see [`crate::PresentOptions`]).
    namespace: String,
    /// Output names (`--screen-root` values) that should get a wallpaper
    /// surface. Empty means every output. Any output not listed is left
    /// alone (no surface), so unconfigured monitors are not blacked out
    /// (SPEC V6: skipped outputs cost zero render work).
    screen_roots: Vec<String>,
    /// Minimum frame interval from `PresentOptions::fps` (`None` = uncapped).
    /// Live-updated by [`RenderCommand::SetFps`].
    min_frame: Option<std::time::Duration>,
    /// Playback-speed clock scale applied to every frame delta
    /// (`PresentOptions::playback_speed`; live-updated by
    /// [`RenderCommand::SetSpeed`]).
    playback_speed: f32,
    /// Set when the compositor closed the last layer surface — treated as
    /// abnormal, mirroring WaylandOpenGLDriver.cpp:234-274
    /// (docs/render-architecture.md §2.3).
    all_surfaces_closed: bool,
    /// Sender for the render thread's own command channel — build workers send
    /// `Install` back through it.
    cmd_tx: CmdSender<RenderCommand>,
    /// Preloaded renderers awaiting an instant [`RenderCommand::Swap`], keyed by
    /// (output name, preload key), stored with the format they were built for.
    preloaded: HashMap<(String, String), (wgpu::TextureFormat, Box<dyn crate::renderer::Renderer + Send>)>,
    /// Global cursor poller (T26; Hyprland IPC — inert elsewhere).
    pointer: crate::pointer::PointerPoll,
}

impl WaylandPlatform {
    /// Connect to the wayland compositor named by `$WAYLAND_DISPLAY`, bind
    /// the required globals (`wl_compositor`, `zwlr_layer_shell_v1`,
    /// `wl_output`/`xdg_output`), and prepare the event loop.
    ///
    /// Output surfaces appear as `wl_output` globals are announced during
    /// [`Platform::run`] — the same path handles both initial enumeration
    /// and hotplug (docs/render-architecture.md §2.3: per requested output
    /// a viewport with its own surface is created).
    pub fn connect(make_renderer: RendererFactory) -> Result<Self, PlatformError> {
        Self::connect_with(make_renderer, crate::PresentOptions::default())
    }

    /// Connect with explicit [`crate::PresentOptions`] — the drop-in path.
    ///
    /// `options.layer_namespace` is stamped on every layer surface (the
    /// daemon watchdog greps it) and `options.screen_roots` restricts which
    /// outputs get a surface (empty = all). Both take effect for the initial
    /// enumeration and for any hotplugged output, since surface creation for
    /// every output flows through [`PlatformState::add_output`].
    pub fn connect_with(
        make_renderer: RendererFactory,
        options: crate::PresentOptions,
    ) -> Result<Self, PlatformError> {
        let conn = Connection::connect_to_env()?;
        let (globals, event_queue) = registry_queue_init::<PlatformState>(&conn)?;
        let qh = event_queue.handle();

        let compositor_state = CompositorState::bind(&globals, &qh)?;
        let layer_shell = LayerShell::bind(&globals, &qh)?;
        let output_state = OutputState::new(&globals, &qh);
        let registry_state = RegistryState::new(&globals);

        let event_loop = EventLoop::try_new()?;
        WaylandSource::new(conn.clone(), event_queue)
            .insert(event_loop.handle())
            .map_err(|err| PlatformError::EventLoopRegister(err.to_string()))?;

        // Command channel: another thread (the IPC applier) sends RenderCommands;
        // they are applied on THIS (render) thread between frames via the calloop
        // source callback — no lock, no surface sharing.
        let (cmd_tx, cmd_rx) = channel::<RenderCommand>();
        event_loop
            .handle()
            .insert_source(cmd_rx, |event, _, state: &mut PlatformState| {
                if let CalloopEvent::Msg(cmd) = event {
                    state.handle_command(cmd);
                }
            })
            .map_err(|err| PlatformError::EventLoopRegister(err.to_string()))?;

        Ok(Self {
            event_loop,
            cmd_tx: cmd_tx.clone(),
            state: PlatformState {
                outputs: Vec::new(),
                gpu: None,
                conn,
                qh,
                registry_state,
                output_state,
                compositor_state,
                layer_shell,
                make_renderer,
                layer: Layer::Background,
                namespace: options.layer_namespace,
                screen_roots: options.screen_roots,
                min_frame: options.fps.filter(|f| *f > 0).map(|f| std::time::Duration::from_secs_f64(1.0 / f64::from(f))),
                playback_speed: if options.playback_speed > 0.0 {
                    options.playback_speed as f32
                } else {
                    1.0
                },
                all_surfaces_closed: false,
                cmd_tx,
                preloaded: HashMap::new(),
                pointer: crate::pointer::PointerPoll::start(),
            },
        })
    }

    /// A clone-able sender for [`RenderCommand`]s. Hand this to the IPC applier so
    /// `bg`/`preload` build renderers off-thread and swap them in on the render
    /// thread (live switch, no relaunch).
    #[must_use]
    pub fn command_sender(&self) -> CmdSender<RenderCommand> {
        self.cmd_tx.clone()
    }

    /// Number of outputs that currently have a surface.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.state.outputs.len()
    }

    /// Dispatch compositor events — and therefore render frames — until
    /// `duration` elapses (`None` = run forever).
    ///
    /// Rendering happens exclusively from `wl_surface.frame` callbacks and
    /// configure events; between events this blocks in the event loop with
    /// zero CPU work (docs/render-architecture.md §2.3:
    /// `wl_display_dispatch` blocks when nothing needs redrawing; SPEC V6
    /// groundwork).
    pub fn run(&mut self, duration: Option<Duration>) -> Result<(), PlatformError> {
        let deadline = duration.map(|d| Instant::now() + d);

        loop {
            if self.state.all_surfaces_closed {
                return Err(PlatformError::AllSurfacesClosed);
            }

            let timeout = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    Some(deadline - now)
                }
                None => None,
            };

            self.event_loop.dispatch(timeout, &mut self.state)?;
        }

        tracing::info!("run deadline reached; tearing down surfaces");
        Ok(())
    }
}

impl PlatformState {
    /// Apply a [`RenderCommand`] on the render thread (called from the calloop
    /// channel source, between frames — no lock, surface untouched here).
    fn handle_command(&mut self, cmd: RenderCommand) {
        match cmd {
            RenderCommand::Build { screen, stash, build } => {
                let Some(gpu) = &self.gpu else { return };
                let Some(ctx) = self.output_for(&screen) else { return };
                let Some(format) = ctx.format else { return }; // no frame drawn yet
                let name = ctx.name.clone();
                let size = (ctx.physical_size.width, ctx.physical_size.height);
                let device = gpu.device.clone();
                let queue = gpu.queue.clone();
                let tx = self.cmd_tx.clone();
                // Build off the render thread — the current wallpaper keeps
                // rendering. The worker sends the result back as `Install`.
                std::thread::spawn(move || {
                    let renderer = build(&device, &queue, format, &name, size);
                    let _ = tx.send(RenderCommand::Install { screen: name, stash, renderer });
                });
            }
            RenderCommand::Install { screen, stash, renderer } => {
                let Some(idx) = self.output_index(&screen) else { return };
                match stash {
                    Some(key) => {
                        let name = self.outputs[idx].name.clone();
                        if let Some(format) = self.outputs[idx].format {
                            // Cap the stash at one preloaded wallpaper per
                            // output: each entry holds a fully-built renderer
                            // (GPU textures + CPU state), and the disk caches
                            // (bundle + shaders) make a cold rebuild cheap —
                            // hoarding old builds is RAM/VRAM the compositor
                            // never sees again. Newest wins.
                            self.preloaded.retain(|(n, _), _| *n != name);
                            self.preloaded.insert((name, key), (format, renderer));
                            kirie_bake::trim_heap();
                        }
                    }
                    None => self.install_renderer(idx, renderer),
                }
            }
            RenderCommand::Swap { screen, key, build } => {
                let Some(idx) = self.output_index(&screen) else { return };
                let name = self.outputs[idx].name.clone();
                let hit = self.preloaded.remove(&(name, key)).and_then(|(format, r)| {
                    // Only a format match is a real hit (surface may have
                    // reconfigured since the preload).
                    (self.outputs[idx].format == Some(format)).then_some(r)
                });
                match hit {
                    // Preload hit → instant pointer swap (sub-100ms).
                    Some(renderer) => {
                        tracing::info!(%screen, "preload hit — instant swap");
                        self.install_renderer(idx, renderer);
                    }
                    // Miss → build off-thread + install when ready (like Build).
                    None => {
                        tracing::info!(%screen, "preload miss — building off-thread");
                        self.handle_command(RenderCommand::Build {
                            screen,
                            stash: None,
                            build,
                        });
                    }
                }
            }
            RenderCommand::SetProperty { screen, key, value, structural } => {
                // Live property change: update the output's renderer in place and
                // repaint so it shows next frame (no reload). No-op if the output
                // or its renderer isn't up yet.
                let Some(idx) = self.output_index(&screen) else { return };
                let ctx = &mut self.outputs[idx];
                if let Some(renderer) = ctx.renderer.as_mut() {
                    // The flag starts `true` (assume structural) so a debounce
                    // that fires before this command was processed still
                    // rebuilds; an explicit Live verdict clears it.
                    let impact = renderer.set_property(&key, &value);
                    structural.store(
                        impact == crate::renderer::PropertyImpact::NeedsRebuild,
                        std::sync::atomic::Ordering::SeqCst,
                    );
                    if ctx.configured && !ctx.frame_pending {
                        let qh = self.qh.clone();
                        ctx.wl_surface().frame(&qh, ctx.wl_surface().clone());
                        ctx.frame_pending = true;
                        ctx.wl_surface().commit();
                    }
                }
            }
            RenderCommand::SwapLocal { screen, build_local } => {
                // Render-thread build (CEF web is !Send). Blocks the loop for the
                // build's duration — a brief hitch on the current wallpaper — then
                // installs. Needs the GPU + a drawn output (format known).
                let Some(gpu) = &self.gpu else { return };
                let Some(idx) = self.output_index(&screen) else { return };
                let Some(format) = self.outputs[idx].format else { return };
                let name = self.outputs[idx].name.clone();
                let size = (
                    self.outputs[idx].physical_size.width,
                    self.outputs[idx].physical_size.height,
                );
                let device = gpu.device.clone();
                let queue = gpu.queue.clone();
                let renderer = build_local(&device, &queue, format, &name, size);
                self.install_renderer(idx, renderer);
            }
            RenderCommand::SetFps(fps) => {
                self.min_frame = fps
                    .filter(|f| *f > 0)
                    .map(|f| std::time::Duration::from_secs_f64(1.0 / f64::from(f)));
            }
            RenderCommand::SetSpeed(speed) => {
                self.playback_speed = if speed > 0.0 { speed } else { 1.0 };
            }
            RenderCommand::Screenshot { screen, capture } => {
                // Capture the live frame on the render thread: the warm renderer
                // re-renders one frame to an offscreen texture (format matches the
                // surface, so its pipelines fit) and the app reads it back + writes
                // the file. Needs the GPU, a drawn output (format known) and a
                // renderer; any missing → drop (the daemon then falls back to the
                // workshop preview, which is why no error is surfaced here).
                let Some(gpu) = &self.gpu else { return };
                let Some(idx) = self.output_index(&screen) else { return };
                let ctx = &mut self.outputs[idx];
                let Some(format) = ctx.format else { return };
                let size = ctx.physical_size;
                if let Some(renderer) = ctx.renderer.as_mut() {
                    capture(&gpu.device, &gpu.queue, renderer.as_mut(), size, format);
                }
            }
        }
    }

    /// Swap `outputs[idx]`'s renderer (the old one drops here) and request a
    /// repaint so the new wallpaper paints on the next frame. Takes a plain
    /// `Box<dyn Renderer>` (no `Send` bound) so it serves both the off-thread
    /// build (whose output is `Send`, coerced here) and the render-thread
    /// [`RenderCommand::SwapLocal`] build (whose output may be `!Send`, e.g. CEF).
    fn install_renderer(&mut self, idx: usize, renderer: Box<dyn crate::renderer::Renderer>) {
        let qh = self.qh.clone();
        let ctx = &mut self.outputs[idx];
        ctx.renderer = Some(renderer); // previous renderer drops here
        ctx.last_frame = None; // reset per-output dt for the fresh renderer
        if ctx.configured && !ctx.frame_pending {
            ctx.wl_surface().frame(&qh, ctx.wl_surface().clone());
            ctx.frame_pending = true;
            ctx.wl_surface().commit();
        }
        // The old renderer just freed its CPU-side state (decoded assets,
        // script heaps, staging buffers) — return those pages to the kernel
        // now rather than letting glibc hoard them until the next build. GPU
        // resources already unmapped via the wgpu drop chain above.
        kirie_bake::trim_heap();
        tracing::info!(output = %ctx.name, "wallpaper swapped in");
    }

    /// The output matching `screen` (`"*"` = the first output, for window mode).
    fn output_for(&self, screen: &str) -> Option<&OutputContext> {
        if screen == "*" {
            self.outputs.first()
        } else {
            self.outputs.iter().find(|c| c.name == screen)
        }
    }

    /// Index of the output matching `screen` (`"*"` = the first).
    fn output_index(&self, screen: &str) -> Option<usize> {
        if screen == "*" {
            (!self.outputs.is_empty()).then_some(0)
        } else {
            self.outputs.iter().position(|c| c.name == screen)
        }
    }

    /// Create the layer surface + swapchain for a newly announced output
    /// (docs/render-architecture.md §2.3: per requested output one
    /// `wl_surface` + layer surface anchored to all four edges with
    /// exclusive zone -1; size is left at 0×0 so the compositor assigns
    /// the full output size via configure).
    fn add_output(&mut self, qh: &QueueHandle<Self>, wl_output: wl_output::WlOutput) {
        let info = self.output_state.info(&wl_output);
        let name = info.as_ref().and_then(|i| i.name.clone()).unwrap_or_default();
        let scale = info
            .as_ref()
            .map(|i| u32::try_from(i.scale_factor.max(1)).unwrap_or(1))
            .unwrap_or(1);
        // Global logical position, for mapping the polled global cursor to
        // surface-local pointer coordinates (T26).
        let position = info.as_ref().map_or((0, 0), |i| (i.location.0, i.location.1));

        // Output selection: when a --screen-root list was supplied, only the
        // listed outputs get a surface. Every other output is left entirely
        // alone (no layer surface, no swapchain), so an unconfigured monitor
        // is never blacked out by a stray wallpaper surface. Empty list =
        // every output (matches the C++ engine's no-`--screen-root` default).
        // Applies to hotplugged outputs too, since they arrive here as well
        // (SPEC V6: a skipped output costs zero render work).
        if !self.screen_roots.is_empty() && !self.screen_roots.iter().any(|r| r == &name) {
            tracing::info!(
                output = %name,
                requested = ?self.screen_roots,
                "output not in requested --screen-root list; leaving it alone"
            );
            return;
        }

        tracing::info!(output = %name, scale, namespace = %self.namespace, "new output; creating layer surface");

        let surface = self.compositor_state.create_surface(qh);
        let layer = self.layer_shell.create_layer_surface(
            qh,
            surface,
            self.layer,
            Some(self.namespace.clone()),
            Some(&wl_output),
        );
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.set_size(0, 0);
        // Initial commit (no buffer) requests the first configure.
        layer.wl_surface().commit();

        // Bring up the shared GPU context on the first surface; reuse the
        // instance for later outputs (docs/render-architecture.md §2.3
        // "wgpu:" note — shared device, per-monitor present pass).
        let wgpu_surface = match &self.gpu {
            Some(gpu) => gpu.create_surface(&self.conn, layer.wl_surface()),
            None => Gpu::new_for_surface(&self.conn, layer.wl_surface()).map(|(gpu, surface)| {
                self.gpu = Some(gpu);
                surface
            }),
        };

        let wgpu_surface = match wgpu_surface {
            Ok(surface) => Some(surface),
            Err(err) => {
                tracing::error!(output = %name, %err, "gpu surface creation failed");
                None
            }
        };

        self.outputs.push(OutputContext {
            wgpu_surface,
            layer,
            wl_output,
            name,
            scale,
            logical_size: (0, 0),
            physical_size: SurfaceSize { width: 1, height: 1 },
            configured: false,
            frame_pending: false,
            first_frame_presented: false,
            renderer: None,
            last_frame: None,
            format: None,
            position,
        });
    }

    /// (Re)configure the swapchain of `outputs[index]` for its current
    /// physical size and mark the full surface opaque, as the C++ driver
    /// does (docs/render-architecture.md §2.3: opaque region
    /// full-surface).
    fn configure_swapchain(&mut self, index: usize) {
        let Some(gpu) = &self.gpu else { return };
        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        let Some(surface) = &ctx.wgpu_surface else {
            return;
        };

        let Some(mut config) =
            surface.get_default_config(&gpu.adapter, ctx.physical_size.width, ctx.physical_size.height)
        else {
            tracing::error!(output = %ctx.name, "adapter cannot present to this surface");
            return;
        };
        // Fifo is universally supported and compositor-paced, matching the
        // frame-callback driven model (docs/render-architecture.md §2.3).
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&gpu.device, &config);

        if let Ok(region) = Region::new(&self.compositor_state) {
            region.add(
                0,
                0,
                i32::try_from(ctx.logical_size.0).unwrap_or(i32::MAX),
                i32::try_from(ctx.logical_size.1).unwrap_or(i32::MAX),
            );
            ctx.wl_surface().set_opaque_region(Some(region.wl_region()));
        }

        ctx.configured = true;
    }

    /// Render one frame for the output backing `surface` and present it.
    ///
    /// Called from exactly two places, mirroring the C++ driver: the first
    /// configure (kick-start, WaylandOpenGLDriver.cpp:405-440) and each
    /// `wl_surface.frame` callback (WaylandOutputViewport.cpp:94-105) —
    /// never from a busy loop (docs/render-architecture.md §2.3).
    fn draw(&mut self, surface: &wl_surface::WlSurface) {
        let Some(index) = self.outputs.iter().position(|ctx| ctx.wl_surface() == surface) else {
            return;
        };

        // Split borrows: take what we need without holding &mut self.
        let Some(gpu) = &self.gpu else { return };
        let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        if !ctx.configured {
            return;
        }

        // `--fps` pacing MUST run before the swapchain acquire: an early frame
        // callback re-requests the next callback and commits WITHOUT rendering.
        // Acquiring first and early-returning drops the SurfaceTexture without
        // presenting it — Vulkan has no un-acquire, so a monitor whose frame
        // callbacks arrive faster than the cap (144Hz vs --fps 123) exhausts
        // the swapchain within ~3 frames; every later acquire times out and
        // the output freezes while the callback/commit cycle spins at 100%
        // CPU (the live-desktop freeze this fixes).
        if let (Some(min), Some(prev)) = (self.min_frame, ctx.last_frame)
            && prev.elapsed() < min
        {
            if !ctx.frame_pending {
                let qh = self.qh.clone();
                ctx.wl_surface().frame(&qh, ctx.wl_surface().clone());
                ctx.frame_pending = true;
                ctx.wl_surface().commit();
            }
            return;
        }

        let Some(wgpu_surface) = &ctx.wgpu_surface else {
            return;
        };

        let mut texture = wgpu_surface.get_current_texture();

        // Outdated/Lost: reconfigure once and retry (wgpu 30 contract on
        // `CurrentSurfaceTexture`).
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
            // Acquire failed (Timeout/Occluded, or Outdated/Lost that
            // survived one reconfigure). This is a swapchain stall, not
            // compositor throttling: the frame callback for this round has
            // already fired (`frame_pending` was cleared), so returning
            // without re-arming would freeze this output forever — no
            // event would ever call `draw` again. Re-request the callback
            // and commit (no buffer attach) so the chain stays alive; this
            // still satisfies SPEC V6 because an occluded/DPMS-off output
            // simply never gets the callback delivered, costing zero work.
            other => {
                tracing::debug!(output = %ctx.name, status = ?other, "skipping frame; re-arming callback");
                if !ctx.frame_pending {
                    ctx.wl_surface().frame(&self.qh, ctx.wl_surface().clone());
                    ctx.frame_pending = true;
                }
                ctx.wl_surface().commit();
                return;
            }
        };

        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        ctx.format = Some(texture.texture.format());

        let renderer = ctx.renderer.get_or_insert_with(|| {
            (self.make_renderer)(&RenderTarget {
                device: &device,
                queue: &queue,
                format: texture.texture.format(),
                output_name: &ctx.name,
                size: (ctx.physical_size.width, ctx.physical_size.height),
            })
        });

        // Per-output dt, seconds; 0 on the first frame
        // (docs/render-architecture.md §2.1 step 3, §2.3 per-output
        // cadence). Scaled by the playback-speed clock — the reference scales
        // g_Time the same way (WallpaperApplication.cpp:908).
        let now = Instant::now();
        let dt = ctx
            .last_frame
            .map(|prev| now.duration_since(prev).as_secs_f32())
            .unwrap_or(0.0)
            * self.playback_speed;
        ctx.last_frame = Some(now);

        // Pointer (T26): map the polled global cursor into this surface's
        // normalized [0,1] coords (top-left origin). Unknown cursor / zero
        // size ⇒ don't call — the renderer keeps its centered default.
        if let Some((gx, gy)) = self.pointer.get() {
            let (lw, lh) = ctx.logical_size;
            if lw > 0 && lh > 0 {
                let nx = ((gx - f64::from(ctx.position.0)) / f64::from(lw)).clamp(0.0, 1.0);
                let ny = ((gy - f64::from(ctx.position.1)) / f64::from(lh)).clamp(0.0, 1.0);
                renderer.set_pointer(nx as f32, ny as f32);
            }
        }

        renderer.render(&view, ctx.physical_size, dt);

        // Request the next frame callback *before* presenting so it rides
        // the same commit — the C++ driver's swapOutput does the same
        // (request frame callback, then swap/commit,
        // WaylandOutputViewport.cpp:263-273;
        // docs/render-architecture.md §2.3).
        if !ctx.frame_pending {
            ctx.wl_surface().frame(&self.qh, ctx.wl_surface().clone());
            ctx.frame_pending = true;
        }

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

    /// Apply a scale change to the output backing `surface`
    /// (docs/render-architecture.md §2.3: buffer scale is re-asserted per
    /// frame in C++; here we reconfigure when it actually changes).
    fn apply_scale(&mut self, surface: &wl_surface::WlSurface, new_scale: u32) {
        let Some(index) = self.outputs.iter().position(|ctx| ctx.wl_surface() == surface) else {
            return;
        };
        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        if ctx.scale == new_scale {
            return;
        }
        tracing::info!(output = %ctx.name, scale = new_scale, "output scale changed");
        ctx.scale = new_scale;
        ctx.update_physical_size();
        if ctx.configured {
            ctx.wl_surface()
                .set_buffer_scale(i32::try_from(new_scale).unwrap_or(1));
            self.configure_swapchain(index);
            let surface = surface.clone();
            self.draw(&surface);
        }
    }
}

impl CompositorHandler for PlatformState {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        self.apply_scale(surface, u32::try_from(new_factor.max(1)).unwrap_or(1));
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
        // Buffer transform optimization not implemented; the compositor
        // handles rotation of untransformed buffers.
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        // The compositor wants a new frame for this output
        // (docs/render-architecture.md §2.3: frame callback fires →
        // render this viewport again).
        if let Some(ctx) = self.outputs.iter_mut().find(|ctx| ctx.wl_surface() == surface) {
            ctx.frame_pending = false;
        }
        self.draw(surface);
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for PlatformState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(&mut self, _conn: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
        self.add_output(qh, output);
    }

    fn update_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
        let Some(info) = self.output_state.info(&output) else {
            return;
        };
        let new_scale = u32::try_from(info.scale_factor.max(1)).unwrap_or(1);
        if let Some(ctx) = self.outputs.iter().find(|ctx| ctx.wl_output == output) {
            let surface = ctx.wl_surface().clone();
            self.apply_scale(&surface, new_scale);
        }
    }

    fn output_destroyed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
        // Hotplug removal: drop the whole per-output context (swapchain
        // first, then layer surface — field order in OutputContext).
        let before = self.outputs.len();
        self.outputs.retain(|ctx| ctx.wl_output != output);
        if self.outputs.len() != before {
            tracing::info!("output removed; destroyed its layer surface");
        }
    }
}

impl LayerShellHandler for PlatformState {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, layer: &LayerSurface) {
        self.outputs.retain(|ctx| &ctx.layer != layer);
        tracing::warn!("compositor closed a layer surface");
        if self.outputs.is_empty() {
            // Abnormal: supervisor should relaunch
            // (docs/render-architecture.md §2.3,
            // WaylandOpenGLDriver.cpp:234-274).
            self.all_surfaces_closed = true;
        }
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let Some(index) = self.outputs.iter().position(|ctx| &ctx.layer == layer) else {
            return;
        };

        // Compositor-suggested size in surface-local (logical)
        // coordinates. Anchored to all four edges with exclusive zone -1
        // this is the full output size (docs/render-architecture.md §2.3).
        // A zero dimension means "pick your own"; fall back to the
        // output's logical size, else skip until a real size arrives.
        let (mut width, mut height) = configure.new_size;
        if width == 0 || height == 0 {
            let logical = self
                .outputs
                .get(index)
                .and_then(|ctx| self.output_state.info(&ctx.wl_output))
                .and_then(|info| info.logical_size);
            match logical {
                Some((w, h)) if w > 0 && h > 0 => {
                    width = u32::try_from(w).unwrap_or(1);
                    height = u32::try_from(h).unwrap_or(1);
                }
                _ => {
                    tracing::warn!("configure with zero size and no logical size; waiting");
                    return;
                }
            }
        }

        let Some(ctx) = self.outputs.get_mut(index) else {
            return;
        };
        let first_configure = !ctx.configured;
        let previous_physical = ctx.physical_size;
        ctx.logical_size = (width, height);
        ctx.update_physical_size();
        // Integer buffer scale, as the C++ driver sets per swap
        // (docs/render-architecture.md §2.3:
        // `wl_surface_set_buffer_scale(scale)`).
        ctx.wl_surface()
            .set_buffer_scale(i32::try_from(ctx.scale).unwrap_or(1));

        tracing::info!(
            output = %ctx.name,
            logical_width = width,
            logical_height = height,
            scale = ctx.scale,
            physical_width = ctx.physical_size.width,
            physical_height = ctx.physical_size.height,
            first_configure,
            "layer surface configured"
        );

        // Only rebuild the swapchain when the size actually changed (or on
        // the first / a previously failed configure): wgpu's
        // `Surface::configure` waits for the GPU to go idle before
        // recreating the swapchain, so doing it on a spurious same-size
        // configure would stall the pipeline for nothing.
        if first_configure || ctx.physical_size != previous_physical {
            self.configure_swapchain(index);
        }

        // Kick-start rendering: the first frame is drawn from the
        // configure, after which frame callbacks take over
        // (docs/render-architecture.md §2.3,
        // WaylandOpenGLDriver.cpp:405-440). Redraws on later configures
        // pick up the new size immediately.
        let surface = layer.wl_surface().clone();
        self.draw(&surface);
    }
}

impl ProvidesRegistryState for PlatformState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers![OutputState];
}

delegate_compositor!(PlatformState);
delegate_output!(PlatformState);
delegate_layer!(PlatformState);
delegate_registry!(PlatformState);
