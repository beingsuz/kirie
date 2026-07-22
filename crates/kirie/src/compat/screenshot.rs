//! `--screenshot` capture (docs/compat-cli.md §3.6).
//!
//! Renders the resolved wallpaper offscreen with a headless wgpu device and
//! reads the framebuffer back with a `copy_texture_to_buffer` + buffer map,
//! then writes the image file. Unlike the running engine (which keeps
//! rendering and captures the live surface, doc §3.6) kirie takes the shot on
//! a throwaway device — enough to unlock the P4 SSIM gate, which only needs a
//! faithful frame written to disk.
//!
//! Both P3 wallpaper types are supported: video (kirie-video) and image/gif/
//! `.tex` (kirie-render). The render target is `Rgba8UnormSrgb`, so the bytes
//! read back are already sRGB-encoded — exactly the stored bytes a PNG wants.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use kirie_audio::AudioCapture;
use kirie_platform::{RenderTarget, Renderer, SurfaceSize};
use kirie_render::{ImageContent, ImageOptions, ImageRenderer};
use kirie_video::{VideoOptions, VideoPlayer};

use crate::compat::args::{ClampMode, ScalingMode};
use crate::compat::resolve::Wallpaper;

/// Fallback offscreen capture size when nothing better is known (video / web /
/// image / auto-projection scenes). The engine composites at the output size ×
/// render-scale (doc §3.6); with no compositor surface here kirie falls back to
/// a fixed 720p canvas.
///
/// A **fixed** canvas was the whole bug (SPEC T16c): the oracle (and the live
/// display) render at the scene's *native projection aspect*, e.g. a tall/portrait
/// scene, so forcing every shot to 16:9 1280×720 horizontally stretches
/// portrait/non-16:9 content — wrong framing and a tanked SSIM even when the
/// pixels are right. [`resolve_capture_size`] now derives the canvas from the
/// scene's orthogonal projection (honoring its aspect) and honors an explicit
/// override, only landing here when neither applies.
const DEFAULT_CAPTURE_SIZE: SurfaceSize = SurfaceSize {
    width: 1280,
    height: 720,
};

/// Longest edge of the projection-derived canvas. The scene renderer draws into
/// its own projection-sized FBOs and blits to this canvas applying the output
/// scaling mode (doc §4); all we must preserve for a faithful screenshot is the
/// projection *aspect*, so we cap the long edge here to keep the readback cheap
/// (a 5160×2160 ultrawide projection would otherwise allocate a ~45 MB target).
/// `KIRIE_SCREENSHOT_SIZE=WxH` overrides both the aspect and this bound.
const CAPTURE_MAX_EDGE: u32 = 1280;

/// Parse a `WxH` size string (e.g. `1280x720`, `634x692`). Both dimensions must
/// be positive integers; separator is a literal `x` (ASCII).
fn parse_size(raw: &str) -> Option<SurfaceSize> {
    let (w, h) = raw.trim().split_once(['x', 'X'])?;
    let width: u32 = w.trim().parse().ok()?;
    let height: u32 = h.trim().parse().ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    Some(SurfaceSize { width, height })
}

/// Explicit `KIRIE_SCREENSHOT_SIZE=WxH` override — the escape hatch for pinning
/// the shot to the exact output size the daemon/oracle used on a given machine
/// (SPEC T16c: the fidelity harness can force byte-for-byte matching dims).
fn size_override() -> Option<SurfaceSize> {
    let raw = std::env::var("KIRIE_SCREENSHOT_SIZE").ok()?;
    match parse_size(&raw) {
        Some(sz) => Some(sz),
        None => {
            tracing::warn!(value = %raw, "ignoring malformed KIRIE_SCREENSHOT_SIZE (want WxH, e.g. 634x692)");
            None
        }
    }
}

/// Scene projection size in scene pixels from `general.orthogonalprojection`
/// (docs/format-scene-json.md §6.2): an object with positive `width`/`height`.
/// `null`, `{ "auto": true }`, a missing key, or non-positive dims ⇒ `None`
/// (auto-projection is resolved from image extents at render time and is not
/// reproduced here — such scenes fall back to the default canvas).
fn scene_projection_dims(scene_json: &[u8]) -> Option<(u32, u32)> {
    let root: serde_json::Value = serde_json::from_slice(scene_json).ok()?;
    let op = root.get("general")?.get("orthogonalprojection")?;
    if op.is_null() || op.get("auto").and_then(serde_json::Value::as_bool) == Some(true) {
        return None;
    }
    let dim = |key: &str| -> Option<u32> {
        let v = op.get(key)?;
        // Tolerate both JSON numbers and stringified numbers, matching the
        // permissive scene parser (kirie_scene::coerce_i64).
        let n = v
            .as_u64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))?;
        u32::try_from(n).ok().filter(|n| *n > 0)
    };
    Some((dim("width")?, dim("height")?))
}

/// Read the projection dims straight out of a scene's `scene.pkg` (`scene.json`
/// entry). Best-effort: any read/parse failure ⇒ `None` (default canvas).
fn read_scene_projection(dir: &Path) -> Option<(u32, u32)> {
    let pkg = kirie_formats::pkg::OwnedPkg::from_path(dir.join("scene.pkg")).ok()?;
    let bytes = pkg.read_name(b"scene.json").ok()?;
    scene_projection_dims(bytes)
}

/// Scale `w×h` so its longest edge is at most `max_edge`, preserving aspect
/// (never upscales — small projections are kept verbatim). Each result dim is
/// clamped to ≥ 1 so a degenerate projection still yields a valid target.
fn fit_aspect(w: u32, h: u32, max_edge: u32) -> SurfaceSize {
    let max_edge = max_edge.max(1);
    let longest = w.max(h);
    if longest <= max_edge {
        return SurfaceSize {
            width: w.max(1),
            height: h.max(1),
        };
    }
    let scale = f64::from(max_edge) / f64::from(longest);
    let round = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
    SurfaceSize {
        width: round(w),
        height: round(h),
    }
}

/// Resolve the offscreen canvas size for this shot (SPEC T16c).
///
/// Priority: explicit `KIRIE_SCREENSHOT_SIZE` override → the scene's native
/// orthogonal-projection aspect (bounded by [`CAPTURE_MAX_EDGE`]) → the fixed
/// [`DEFAULT_CAPTURE_SIZE`] fallback (video/web/image/auto-projection). Honoring
/// the projection aspect is what stops portrait/non-16:9 scenes from being
/// horizontally stretched into a 16:9 frame.
fn resolve_capture_size(wallpaper: &Wallpaper) -> SurfaceSize {
    resolve_capture_size_with(size_override(), wallpaper)
}

/// Core resolution with the override supplied explicitly — split out so tests
/// exercise the priority order without mutating the process-global env (the
/// crate is `#![forbid(unsafe_code)]`, and `std::env::set_var` is `unsafe` on
/// edition 2024).
fn resolve_capture_size_with(override_size: Option<SurfaceSize>, wallpaper: &Wallpaper) -> SurfaceSize {
    if let Some(sz) = override_size {
        return sz;
    }
    if let Wallpaper::Scene { dir } = wallpaper
        && let Some((w, h)) = read_scene_projection(dir)
    {
        return fit_aspect(w, h, CAPTURE_MAX_EDGE);
    }
    DEFAULT_CAPTURE_SIZE
}

/// Row alignment required by `copy_texture_to_buffer` (wgpu/WebGPU: buffer
/// row pitch must be a multiple of 256 bytes).
const ROW_ALIGN: u32 = 256;

// --- Scene settle heuristic (SPEC T14) ----------------------------------
//
// Image/video/web paint a complete frame the instant they have decoded data, so
// the first non-black readback is a faithful shot. **Scene** wallpapers do not:
// a model/effect/generative scene streams textures + meshes on background
// threads, warms up its SceneScript host, and animates its composite in over
// several frames (confirmed: 3047596375 Starscape is black for the first few
// seconds, then figure + ripples + planets fade in). Capturing the first
// non-black frame there yields a half-built composite, and a short all-black
// deadline gives up before any content exists at all. So for scenes we keep
// pumping frames past the first non-black one until the composite *settles* —
// its lit fraction stops changing — or a generous cap.

/// Minimum lit fraction (see [`lit_fraction`]) for a scene frame to count as
/// having *content* at all. Deliberately far below the 5% image/video gate: a
/// dark scene like Starscape (a black-mesh figure on near-black space with faint
/// ripples + a few bright planets) legitimately lights only ~1.5% of pixels, yet
/// is plainly not black — its brightest pixels hit 255. A genuinely-black
/// loading frame sits at ~0% and never clears this floor, so it still times out
/// (capped) rather than being captured as content.
const SCENE_CONTENT_FLOOR: f64 = 0.005;

/// A scene is "settled" once its lit fraction stops changing between readbacks.
/// The tolerance is `max(EPS_ABS, prev·EPS_REL)`: the absolute floor keeps a
/// near-zero-lit dark scene from looking stable while still fading in, and the
/// relative term tolerates the small frame-to-frame jitter of steady-state
/// animation (drifting ripples/planets) on brighter scenes.
const SETTLE_LIT_EPS_ABS: f64 = 0.002;
/// Relative component of the settle tolerance (fraction of the previous lit
/// fraction). See [`SETTLE_LIT_EPS_ABS`].
const SETTLE_LIT_EPS_REL: f64 = 0.05;

/// Consecutive stable readbacks required before a scene is declared settled —
/// guards against a lull mid fade-in being mistaken for the final composite.
const SETTLE_STREAK: u32 = 3;

/// Minimum non-black readbacks to observe after content first appears before
/// accepting, even if the lit fraction looks stable immediately (a scene that
/// is non-black on frame one still gets a few frames to finish compositing).
const SETTLE_MIN_EXTRA: u32 = 8;

/// Hard cap on non-black readbacks spent settling, so a scene whose lit fraction
/// never plateaus (e.g. a continuously pulsing composite) still captures a
/// fully-rendered frame promptly instead of burning the whole wall-clock budget.
const SETTLE_MAX_EXTRA: u32 = 150;

/// Tracks a scene's settle state across readbacks. Pure (no GPU), so the decision
/// logic is unit-testable without a device.
struct SceneSettle {
    prev_lit: Option<f64>,
    stable_streak: u32,
    extra: u32,
}

impl SceneSettle {
    fn new() -> Self {
        Self {
            prev_lit: None,
            stable_streak: 0,
            extra: 0,
        }
    }

    /// Feed the lit fraction of the latest **non-black** readback. Returns `true`
    /// once the scene's composite has settled enough to capture (a stable-lit
    /// streak past the minimum, or the settle cap).
    fn observe(&mut self, lit: f64) -> bool {
        self.extra = self.extra.saturating_add(1);
        let stable = self.prev_lit.is_some_and(|prev| {
            let tol = SETTLE_LIT_EPS_ABS.max(prev * SETTLE_LIT_EPS_REL);
            (lit - prev).abs() <= tol
        });
        if stable {
            self.stable_streak += 1;
        } else {
            self.stable_streak = 0;
        }
        self.prev_lit = Some(lit);
        (self.stable_streak >= SETTLE_STREAK && self.extra >= SETTLE_MIN_EXTRA)
            || self.extra >= SETTLE_MAX_EXTRA
    }
}

/// A headless wgpu context (no surface).
struct Headless {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
}

impl Headless {
    /// Bring up a headless device: Vulkan preferred, then any backend (SPEC
    /// §G wgpu/Vulkan), mirroring kirie-platform's adapter policy.
    fn new() -> Result<Self> {
        let mut last: Option<anyhow::Error> = None;
        for backends in [wgpu::Backends::VULKAN, wgpu::Backends::all()] {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
                Ok(adapter) => {
                    let info = adapter.get_info();
                    tracing::info!(backend = %info.backend, adapter = %info.name, "screenshot gpu");
                    let (device, queue) =
                        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                            label: Some("kirie-screenshot"),
                            required_features: adapter.features()
                                & wgpu::Features::PIPELINE_CACHE,
                            ..wgpu::DeviceDescriptor::default()
                        }))
                        .context("request headless wgpu device")?;
                    // Warm/persist the driver pipeline cache here too, so
                    // `--screenshot` runs share the engine's compiled binaries.
                    kirie_platform::attach_pipeline_cache(&device, &adapter);
                    return Ok(Self { device, queue, adapter });
                }
                Err(err) => last = Some(anyhow!("no adapter on {backends:?}: {err}")),
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("no wgpu adapter for screenshot")))
    }
}

/// Capture `wallpaper` to `out_path` after at least `delay` frames.
///
/// `scaling`/`clamp` are the resolved per-target modes (doc §3.1). `delay` is
/// the `--screenshot-delay` **minimum settle** in frames (default 5): the loop
/// always renders at least this many frames before it will accept a shot. Image
/// / video / web are then captured on their first non-black frame (they paint a
/// complete frame immediately). A **scene** streams assets and animates its
/// composite in over several frames, so it keeps pumping past the first
/// non-black frame until the composite settles (its lit fraction plateaus) or a
/// generous hard cap, so model/effect/generative scenes reliably capture their
/// real content instead of an early half-built (or all-black) frame (SPEC T14).
/// The last readback is always written; if none was ever non-black a warning is
/// logged.
/// `audio` is the shared capture handle whose spectrum feeds a scene's
/// `g_AudioSpectrum*` uniforms (docs §8.3); `None` ⇒ a silent spectrum (the
/// screenshot of an audio-reactive scene then shows its rest state).
#[allow(clippy::too_many_arguments)]
pub fn capture(
    wallpaper: &Wallpaper,
    scaling: ScalingMode,
    clamp: ClampMode,
    delay: u32,
    out_path: &Path,
    audio: Option<Arc<AudioCapture>>,
    properties: &[(String, String)],
) -> Result<()> {
    let gpu = Headless::new()?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    // T16c: render at the scene's native projection aspect (or an explicit
    // override), not a hardcoded 1280×720, so portrait/non-16:9 scenes aren't
    // horizontally stretched.
    let capture_size = resolve_capture_size(wallpaper);
    tracing::info!(
        width = capture_size.width,
        height = capture_size.height,
        "screenshot canvas"
    );

    let target_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kirie-screenshot-target"),
        size: wgpu::Extent3d {
            width: capture_size.width,
            height: capture_size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let render_target = RenderTarget {
        device: &gpu.device,
        queue: &gpu.queue,
        format,
        output_name: "screenshot",
        size: (capture_size.width, capture_size.height),
    };

    let mut renderer: Box<dyn Renderer> = match wallpaper {
        Wallpaper::Video { media } => {
            let options = VideoOptions {
                scaling: super::run::to_video_scaling(scaling),
                // Headless: skip the audio pipeline so the wall clock paces
                // frame selection and no device is opened.
                enable_audio: false,
                ..VideoOptions::default()
            };
            let (player, _control) = VideoPlayer::open(media, options)
                .with_context(|| format!("opening video {}", media.display()))?;
            Box::new(kirie_video::VideoRenderer::new(&render_target, player))
        }
        Wallpaper::Image { file } => {
            let content =
                ImageContent::from_path(file).with_context(|| format!("loading image {}", file.display()))?;
            let options = ImageOptions {
                scaling: super::run::to_render_scaling(scaling),
                clamp: super::run::to_render_clamp(clamp),
            };
            Box::new(
                ImageRenderer::new(&render_target, &content, options).context("building image renderer")?,
            )
        }
        Wallpaper::Scene { dir } => {
            let options = kirie_render::SceneOptions {
                render_scale: 1.0,
                scaling: super::run::to_render_scaling(scaling),
                clamp: super::run::to_render_clamp(clamp),
                disable_parallax: false,
            };
            kirie_render::load_workshop_scene(
                &render_target,
                dir,
                super::resolve::we_assets_dir().as_deref(),
                options,
                audio,
                properties,
            )
            .with_context(|| format!("building scene renderer for {}", dir.display()))?
        }
        #[cfg(feature = "web-cef")]
        Wallpaper::Web { dir, file } => {
            use kirie_web::{WebBackend, WebRenderer, WebSize, hosted::HostedBackend};
            let url = super::resolve::web_entry_url(dir, file);
            let size = WebSize {
                width: capture_size.width,
                height: capture_size.height,
            };
            let backend = <HostedBackend as WebBackend>::new(&url, size)
                .map_err(|e| anyhow!("starting web backend for {url}: {e}"))?;
            Box::new(WebRenderer::new(&render_target, Box::new(backend)))
        }
        #[cfg(not(feature = "web-cef"))]
        Wallpaper::Web { .. } => {
            bail!(
                "cannot screenshot a web wallpaper: this build has no off-screen web backend \
                 (rebuild with --features web-cef)"
            );
        }
        Wallpaper::Unsupported { kind } => {
            bail!("cannot screenshot a {kind} wallpaper: not yet supported by kirie");
        }
        Wallpaper::Asset => {
            bail!(
                "cannot screenshot this item: it is a Wallpaper Engine asset (effect preset), not a renderable wallpaper"
            );
        }
    };

    let deadline = Instant::now() + capture_budget(wallpaper);
    let dt = 1.0 / 60.0;
    // `--screenshot-delay` is a *minimum* settle: render at least this many
    // frames before accepting any shot (default 5, clamped 0..600 upstream).
    let min_frames = delay.max(1);
    // Only scenes need the multi-frame settle; image/video/web paint a complete
    // frame at once and keep the fast first-non-black path.
    let settle_scene = matches!(wallpaper, Wallpaper::Scene { .. });
    // Scenes count as having content at a much lower lit fraction than the 5%
    // image/video gate — a dark scene (Starscape) lights only ~1.5% of pixels
    // yet is plainly not black. Image/video/web fill the frame, so they keep the
    // 5% gate (matching the e2e corpus test's own threshold).
    let content_floor = if settle_scene { SCENE_CONTENT_FLOOR } else { 0.05 };
    let mut pixels = vec![0u8; (capture_size.width * capture_size.height * 4) as usize];
    let mut frame: u32 = 0;
    let mut captured_nonblack = false;
    let mut settle = SceneSettle::new();
    loop {
        renderer.render(&view, capture_size, dt);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("gpu poll after render: {e}"))?;
        frame += 1;

        let timed_out = Instant::now() >= deadline;

        // Read back once past the minimum settle, or on the final (timed-out)
        // frame so the last frame is always written.
        if frame >= min_frames || timed_out {
            pixels = readback(&gpu.device, &gpu.queue, &target_tex, capture_size)?;
            let lit = lit_fraction(&pixels);
            if lit > content_floor {
                captured_nonblack = true;
                if !settle_scene {
                    break; // image/video/web: first painted frame is the shot.
                }
                // Scene: keep pumping until the composite settles (or its cap).
                if settle.observe(lit) {
                    break;
                }
            }
        }

        if timed_out {
            break; // generous hard cap; the last readback (above) is written.
        }
        std::thread::sleep(Duration::from_millis(16));
    }

    kirie_platform::persist_pipeline_cache(&gpu.adapter);
    write_image(out_path, capture_size.width, capture_size.height, &pixels)?;
    if !captured_nonblack {
        tracing::warn!(
            path = %out_path.display(),
            "screenshot frame was all black (wallpaper produced no visible frame in time)"
        );
    }
    Ok(())
}

/// Capture the current frame of an already-built, warm `renderer` to `path`,
/// rendering one frame into an offscreen texture whose format matches the live
/// surface `format` (so the renderer's pipelines — built for that format — fit)
/// and reading it straight back.
///
/// Distinct from [`capture`]: no throwaway device, no renderer rebuild, no
/// settle loop. The renderer is already warm on the render thread and has been
/// presenting, so a single re-render of its current state (`dt = 0`, no
/// animation advance) reproduces the on-screen frame. This backs the socket
/// `screenshot` command (doc §4.12), whose caller (the daemon's theming) wants
/// the palette to match what is actually displayed — property overrides and all
/// — rather than the workshop preview.
pub fn capture_live(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut dyn Renderer,
    size: SurfaceSize,
    format: wgpu::TextureFormat,
    path: &Path,
) -> Result<()> {
    let size = SurfaceSize {
        width: size.width.max(1),
        height: size.height.max(1),
    };
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kirie-socket-screenshot"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    renderer.render(&view, size, 0.0);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| anyhow!("gpu poll after live-frame render: {e}"))?;

    let mut pixels = readback(device, queue, &target_tex, size)?;
    // The surface may be BGRA (common on Vulkan); write_image reads the first
    // three bytes as R,G,B, so reorder BGRA→RGBA. RGBA formats need no swap.
    if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    write_image(path, size.width, size.height, &pixels)
}

/// Wall-clock safety budget for the capture loop.
///
/// Image/video paints its first frame immediately and the loop returns the
/// instant a non-black frame appears, so 6s is ample worst-case slack. A
/// **scene** streams assets + warms its SceneScript host + fades its composite
/// in over several seconds (confirmed: 3047596375 is black for ~3s before the
/// figure/ripples/planets appear), and its settle loop pumps past first content,
/// so it gets a much more generous budget. A **web** wallpaper boots a whole
/// headless browser and may stream large media (the corpus MV wallpaper pulls in
/// 100–250 MB `.webm` clips) before its first visible paint, so it gets the
/// largest. `KIRIE_SCREENSHOT_TIMEOUT_SECS` overrides all for extra headroom on
/// slow machines / especially heavy pages.
fn capture_budget(wallpaper: &Wallpaper) -> Duration {
    let default_secs = match wallpaper {
        Wallpaper::Web { .. } => 45,
        Wallpaper::Scene { .. } => 20,
        _ => 6,
    };
    let secs = std::env::var("KIRIE_SCREENSHOT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(default_secs);
    Duration::from_secs(secs)
}

/// Copy the target texture to a mappable buffer and return tightly packed
/// RGBA8/BGRA8 (`width·height·4` bytes, row padding stripped) — the byte order
/// is the texture's own (the caller reorders per format).
fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: SurfaceSize,
) -> Result<Vec<u8>> {
    let width = size.width;
    let height = size.height;
    let unpadded = width * 4;
    let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kirie-screenshot-readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("kirie-screenshot-copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = crossbeam_channel::bounded(1);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| anyhow!("gpu poll for map: {e}"))?;
    rx.recv()
        .map_err(|_| anyhow!("readback map channel closed"))?
        .map_err(|e| anyhow!("buffer map failed: {e}"))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| anyhow!("mapping readback buffer: {e}"))?;
    let mut out = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    Ok(out)
}

/// Fraction (0.0..=1.0) of pixels that are visibly lit — any channel above the
/// black-floor of 8. The settle heuristic watches this value plateau; a scene's
/// content appearing swings it sharply, while steady-state animation barely
/// moves it.
fn lit_fraction(rgba: &[u8]) -> f64 {
    let total = rgba.len() / 4;
    if total == 0 {
        return 0.0;
    }
    let mut lit = 0usize;
    let (pixels, _) = rgba.as_chunks::<4>();
    for px in pixels {
        if px[0] > 8 || px[1] > 8 || px[2] > 8 {
            lit += 1;
        }
    }
    lit as f64 / total as f64
}

/// Write RGB (alpha dropped, RGB-8 like the engine's screenshot, doc §3.6) to
/// `path`; the `image` crate picks the encoder from the extension (validated
/// to `.bmp`/`.png`/`.jpeg`/`.jpg` at parse, doc §3.6).
fn write_image(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    let (pixels, _) = rgba.as_chunks::<4>();
    for px in pixels {
        rgb.extend_from_slice(&px[0..3]);
    }
    let img = image::RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| anyhow!("screenshot buffer size mismatch"))?;
    img.save(path)
        .with_context(|| format!("writing screenshot {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// `lit_fraction` counts any pixel with a channel above the black-floor of 8,
    /// and a dark scene (few lit pixels) still clears `SCENE_CONTENT_FLOOR` while
    /// staying under the 5% image/video gate — the crux of the T14 fix.
    #[test]
    fn lit_fraction_counts_lit_pixels_and_floors_empty() {
        assert_eq!(lit_fraction(&[]), 0.0);
        // 1000 pixels, 20 lit (2%) — a Starscape-like dark scene.
        let mut buf = vec![0u8; 1000 * 4];
        for px in buf.chunks_mut(4).take(20) {
            px[0] = 255;
        }
        let frac = lit_fraction(&buf);
        assert!((frac - 0.02).abs() < 1e-9, "expected 2% lit, got {frac}");
        assert!(
            frac > SCENE_CONTENT_FLOOR,
            "dark scene must clear the scene floor"
        );
        assert!(frac < 0.05, "but stays under the image/video 5% gate");
        // Black-floor: channel == 8 is not lit.
        let mut floor = vec![0u8; 4];
        floor[0] = 8;
        assert_eq!(lit_fraction(&floor), 0.0);
    }

    /// A scene that is fully lit from the first readback still gets a few settle
    /// frames, then captures once the streak clears the minimum.
    #[test]
    fn scene_settle_accepts_stable_scene_after_min_extra() {
        let mut s = SceneSettle::new();
        // Constant lit fraction: first observe seeds prev (not stable), the rest
        // build the streak. It must not accept before SETTLE_MIN_EXTRA frames.
        for i in 1..SETTLE_MIN_EXTRA {
            assert!(!s.observe(0.42), "accepted too early at extra={i}");
        }
        assert!(s.observe(0.42), "must accept once min extra reached and stable");
    }

    /// A composite that fades in (lit fraction climbing past the epsilon each
    /// frame) is not declared settled until it plateaus.
    #[test]
    fn scene_settle_waits_through_fade_in() {
        let mut s = SceneSettle::new();
        // Rising lit fraction, each step well above SETTLE_LIT_EPS: streak keeps
        // resetting, so no accept during the fade.
        let mut lit = 0.10;
        for _ in 0..20 {
            assert!(!s.observe(lit), "must not settle mid fade-in at lit={lit}");
            lit += 0.03;
        }
        // Now it plateaus; after a stable streak it settles.
        let plateau = lit;
        let mut settled = false;
        for _ in 0..SETTLE_STREAK + 1 {
            if s.observe(plateau) {
                settled = true;
                break;
            }
        }
        assert!(settled, "must settle once the composite plateaus");
    }

    /// A composite whose lit fraction never plateaus still captures at the cap
    /// rather than looping forever.
    #[test]
    fn scene_settle_caps_when_never_stable() {
        let mut s = SceneSettle::new();
        let mut accepted_at = None;
        for i in 1..=SETTLE_MAX_EXTRA {
            // Alternate far apart so the streak can never build.
            let lit = if i % 2 == 0 { 0.20 } else { 0.60 };
            if s.observe(lit) {
                accepted_at = Some(i);
                break;
            }
        }
        assert_eq!(
            accepted_at,
            Some(SETTLE_MAX_EXTRA),
            "unstable composite must accept exactly at the settle cap"
        );
    }

    #[test]
    fn parse_size_accepts_wxh_and_rejects_junk() {
        assert_eq!(
            parse_size("1280x720"),
            Some(SurfaceSize {
                width: 1280,
                height: 720
            })
        );
        assert_eq!(
            parse_size(" 634X692 "),
            Some(SurfaceSize {
                width: 634,
                height: 692
            })
        );
        assert_eq!(parse_size("0x100"), None);
        assert_eq!(parse_size("100x0"), None);
        assert_eq!(parse_size("1280"), None);
        assert_eq!(parse_size("axb"), None);
        assert_eq!(parse_size(""), None);
    }

    #[test]
    fn fit_aspect_preserves_orientation_and_bounds_long_edge() {
        // Landscape projections shrink to the 1280 long-edge, staying landscape.
        assert_eq!(
            fit_aspect(1920, 1080, 1280),
            SurfaceSize {
                width: 1280,
                height: 720
            }
        );
        assert_eq!(
            fit_aspect(2560, 1440, 1280),
            SurfaceSize {
                width: 1280,
                height: 720
            }
        );
        // A tall/portrait projection stays portrait — the whole point of T16c:
        // width < height in, width < height out (no 16:9 stretch).
        let tall = fit_aspect(634, 692, 1280);
        assert!(tall.width < tall.height, "portrait must stay portrait: {tall:?}");
        assert_eq!(
            tall,
            SurfaceSize {
                width: 634,
                height: 692
            }
        );
        // A large portrait projection is bounded by its long (height) edge and
        // still portrait.
        let big_tall = fit_aspect(1500, 3000, 1280);
        assert_eq!(
            big_tall,
            SurfaceSize {
                width: 640,
                height: 1280
            }
        );
        assert!(big_tall.width < big_tall.height);
        // Non-16:9 landscape (3609007632: 2474×1856) keeps its ~4:3 aspect
        // rather than being squashed to 16:9.
        assert_eq!(
            fit_aspect(2474, 1856, 1280),
            SurfaceSize {
                width: 1280,
                height: 960
            }
        );
    }

    #[test]
    fn scene_projection_dims_reads_orthogonalprojection() {
        // Explicit portrait projection.
        let portrait = br#"{"general":{"orthogonalprojection":{"width":634,"height":692}}}"#;
        assert_eq!(scene_projection_dims(portrait), Some((634, 692)));
        // Explicit landscape projection.
        let landscape = br#"{"general":{"orthogonalprojection":{"width":1920,"height":1080}}}"#;
        assert_eq!(scene_projection_dims(landscape), Some((1920, 1080)));
        // Stringified numbers (permissive, like the scene parser).
        let strnums = br#"{"general":{"orthogonalprojection":{"width":"800","height":"600"}}}"#;
        assert_eq!(scene_projection_dims(strnums), Some((800, 600)));
        // Auto / null / missing / zero ⇒ None (default canvas).
        assert_eq!(
            scene_projection_dims(br#"{"general":{"orthogonalprojection":{"auto":true}}}"#),
            None
        );
        assert_eq!(
            scene_projection_dims(br#"{"general":{"orthogonalprojection":null}}"#),
            None
        );
        assert_eq!(scene_projection_dims(br#"{"general":{}}"#), None);
        assert_eq!(
            scene_projection_dims(br#"{"general":{"orthogonalprojection":{"width":0,"height":100}}}"#),
            None
        );
        assert_eq!(scene_projection_dims(b"not json"), None);
    }

    /// The headline T16c assertion tying the pieces together: a portrait scene
    /// resolves to a portrait canvas, a landscape scene to a landscape one, and
    /// neither is the stretched 16:9 default when a projection is present.
    #[test]
    fn projection_derived_canvas_is_not_stretched() {
        let portrait =
            scene_projection_dims(br#"{"general":{"orthogonalprojection":{"width":634,"height":692}}}"#)
                .map(|(w, h)| fit_aspect(w, h, CAPTURE_MAX_EDGE))
                .unwrap();
        assert!(
            portrait.width < portrait.height,
            "portrait scene must screenshot portrait, got {portrait:?}"
        );
        assert_ne!(portrait, DEFAULT_CAPTURE_SIZE, "must not be the 1280x720 default");

        let landscape =
            scene_projection_dims(br#"{"general":{"orthogonalprojection":{"width":1920,"height":1080}}}"#)
                .map(|(w, h)| fit_aspect(w, h, CAPTURE_MAX_EDGE))
                .unwrap();
        assert!(landscape.width > landscape.height, "landscape stays landscape");
        // 1920×1080 folds exactly onto the historical 1280×720 canvas, so
        // already-matching landscape scenes do not regress.
        assert_eq!(landscape, DEFAULT_CAPTURE_SIZE);
    }

    /// `resolve_capture_size_with` priority: override → projection → default.
    #[test]
    fn resolve_capture_size_priority() {
        // No override, non-scene wallpaper ⇒ default canvas.
        let video = Wallpaper::Video {
            media: PathBuf::from("/nonexistent.mp4"),
        };
        assert_eq!(resolve_capture_size_with(None, &video), DEFAULT_CAPTURE_SIZE);

        // A scene dir with no readable pkg ⇒ default (best-effort).
        let bad_scene = Wallpaper::Scene {
            dir: PathBuf::from("/nonexistent-scene-dir"),
        };
        assert_eq!(resolve_capture_size_with(None, &bad_scene), DEFAULT_CAPTURE_SIZE);

        // Explicit override wins over everything, even a non-scene wallpaper.
        let over = SurfaceSize {
            width: 500,
            height: 900,
        };
        assert_eq!(resolve_capture_size_with(Some(over), &video), over);

        // A real corpus scene (if present) resolves to its projection aspect,
        // bounded — never the stretched default. 3609007632 is 2474×1856.
        let corpus = PathBuf::from("/home/aiko/.steam/steam/steamapps/workshop/content/431960/3609007632");
        if corpus.join("scene.pkg").is_file() {
            let sz = resolve_capture_size_with(None, &Wallpaper::Scene { dir: corpus });
            assert_eq!(sz, fit_aspect(2474, 1856, CAPTURE_MAX_EDGE));
            assert_ne!(
                sz, DEFAULT_CAPTURE_SIZE,
                "non-16:9 scene must not use the 1280x720 default"
            );
        }
    }
}
