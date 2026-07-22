//! Corpus-gated headless GPU test for the **scene** renderer.
//!
//! For every `scene.pkg` in the workshop corpus (docs/corpus.md) this resolves
//! the scene against its `project.json` properties, builds a [`SceneRenderer`],
//! renders one frame into an offscreen `Rgba8Unorm` target on a real adapter,
//! reads it back, and classifies the result (rendered non-black / clear-color
//! only / build error). It asserts the pipeline produces plausible pixels for
//! at least one of the simplest scenes and prints a per-item table so the ones
//! that still error (unsupported shaders, effect graphs, model objects) are
//! visible.
//!
//! Skipped (with a note, never failed) when either the corpus or a wgpu adapter
//! is absent — so this is inert in CI and live on the RTX 4080.

use std::path::{Path, PathBuf};

use kirie_formats::pkg::OwnedPkg;
use kirie_formats::project::Project;
use kirie_platform::{RenderTarget, Renderer, SurfaceSize};
use kirie_render::{ClampMode, ScalingMode, SceneOptions, SceneRenderer};
use kirie_scene::object::ObjectKind;
use kirie_scene::resolve::AssetSource;
use kirie_scene::{PropertyBag, Scene, SceneModel};

/// Default corpus location (docs/corpus.md); override with `KIRIE_CORPUS`.
const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
/// Wallpaper Engine's shared builtin assets (`genericimage2`, effect shaders,
/// builtin materials) — referenced by scene materials but *not* bundled in the
/// scene.pkg, exactly as the C++ engine loads them from its install
/// (docs/render-architecture.md §10 asset lookup). Override with
/// `KIRIE_WE_ASSETS`.
const ASSETS_DIR: &str = "/home/aiko/.steam/steam/steamapps/common/wallpaper_engine/assets";

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const OUT_W: u32 = 640;
const OUT_H: u32 = 360;

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("KIRIE_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CORPUS_DIR));
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "skipping scene corpus test: {} not found (set KIRIE_CORPUS to override)",
            dir.display()
        );
        None
    }
}

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        Ok(adapter) => adapter,
        Err(err) => {
            eprintln!("skipping scene corpus test: no adapter ({err})");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kirie-scene-corpus"),
        ..wgpu::DeviceDescriptor::default()
    })) {
        Ok((device, queue)) => Some((device, queue)),
        Err(err) => {
            eprintln!("skipping scene corpus test: no device ({err})");
            None
        }
    }
}

/// An [`AssetSource`] that resolves a path against the scene's `scene.pkg`
/// first (byte-exact entry name, docs/format-pkg.md §2), then falls back to the
/// shared builtin assets directory on disk — mirroring the C++ engine, which
/// reads scene-local assets from the container and builtin shaders/materials
/// from its install (docs/render-architecture.md §10).
struct CompositeSource<'a> {
    pkg: &'a OwnedPkg,
    assets: Option<PathBuf>,
}

impl AssetSource for CompositeSource<'_> {
    fn load(&self, path: &str) -> Option<Vec<u8>> {
        if let Ok(bytes) = self.pkg.read_name(path.as_bytes()) {
            return Some(bytes.to_vec());
        }
        let base = self.assets.as_ref()?;
        std::fs::read(base.join(path)).ok()
    }
}

/// Render one frame with `renderer` into a fresh offscreen target and read back
/// tightly-packed RGBA8 rows.
fn render_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut dyn Renderer,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-corpus-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // A dt=0 frame establishes the blit window, then a burst of fixed-step
    // frames advances time-driven shaders *and* populates particle sims (an
    // emitter at ~10/s needs seconds of simulation before its sprites show).
    renderer.render(&view, SurfaceSize { width, height }, 0.0);
    for _ in 0..120 {
        renderer.render(&view, SurfaceSize { width, height }, 1.0 / 30.0);
    }

    let padded_row = (width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene-corpus-readback"),
        size: u64::from(padded_row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scene-corpus-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let mapped = buffer.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded_row) as usize;
        pixels.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

/// Classification of a rendered frame.
struct Stats {
    /// Fraction of pixels whose max RGB channel differs from the frame's modal
    /// (background) color by more than a small threshold — i.e. actual content.
    non_background: f32,
    /// Whether any pixel has non-zero RGB at all.
    any_lit: bool,
}

fn classify(pixels: &[u8]) -> Stats {
    // The clear color fills most of a simple scene; treat the top-left pixel as
    // the background reference and count pixels that deviate from it.
    let bg = [pixels[0], pixels[1], pixels[2]];
    let mut differ = 0usize;
    let mut lit = false;
    let total = pixels.len() / 4;
    for px in pixels.as_chunks::<4>().0 {
        if px[0] > 4 || px[1] > 4 || px[2] > 4 {
            lit = true;
        }
        let d = (i32::from(px[0]) - i32::from(bg[0])).abs()
            + (i32::from(px[1]) - i32::from(bg[1])).abs()
            + (i32::from(px[2]) - i32::from(bg[2])).abs();
        if d > 12 {
            differ += 1;
        }
    }
    Stats {
        non_background: differ as f32 / total.max(1) as f32,
        any_lit: lit,
    }
}

/// Build a resolved [`SceneModel`] for `item_dir` (its `scene.pkg` +
/// `project.json`). Returns the model, the owned pkg (kept alive so its
/// [`PkgSource`] stays valid), and the property bag.
fn load_model(
    item_dir: &Path,
    assets: &Option<PathBuf>,
) -> Result<(OwnedPkg, SceneModel, PropertyBag), String> {
    let pkg = OwnedPkg::from_path(item_dir.join("scene.pkg")).map_err(|e| format!("pkg: {e}"))?;
    let bag = Project::from_path(item_dir.join("project.json"))
        .map(|p| PropertyBag::from_project(&p))
        .unwrap_or_default();

    // Parse scene.json out of the pkg. Borrow the pkg only inside this scope.
    let scene = {
        let bytes = pkg
            .read_name(b"scene.json")
            .map_err(|e| format!("scene.json: {e}"))?;
        Scene::from_slice(bytes).map_err(|e| format!("scene.json parse: {e}"))?
    };
    let mut model = SceneModel::resolve(scene, &bag);
    let source = CompositeSource {
        pkg: &pkg,
        assets: assets.clone(),
    };
    let _problems = model.load_assets(&source, &bag);
    drop(source);
    Ok((pkg, model, bag))
}

#[test]
fn scene_corpus_renders_simplest_scenes_non_black() {
    let Some(dir) = corpus_dir() else { return };
    let Some((device, queue)) = gpu() else { return };
    // Surface per-object/per-pass skip reasons (shader build + texture decode
    // failures) by setting `KIRIE_TRACE=1`; quiet otherwise.
    if std::env::var_os("KIRIE_TRACE").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
    }

    let assets = std::env::var_os("KIRIE_WE_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(ASSETS_DIR));
    let assets = assets.is_dir().then_some(assets);
    if assets.is_none() {
        eprintln!("note: builtin WE assets not found; scenes using builtin shaders will skip");
    }

    let mut items: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("scene.pkg").is_file())
        .collect();
    items.sort();
    assert!(!items.is_empty(), "no scene.pkg items under {}", dir.display());

    let mut rendered = 0usize; // non-background content present
    let mut clear_only = 0usize; // built + drew but frame is flat background
    let mut errored = 0usize; // could not build a renderer
    let mut lines = Vec::new();

    // Non-image compositing coverage (SPEC §T16b): how many particle/text
    // objects are wired in across the corpus and whether they populate.
    let mut total_particle_objs = 0usize; // particle objects present in the models
    let mut wired_particle_items = 0usize; // particle systems the renderer built
    let mut wired_text_items = 0usize;
    let mut total_live_particles = 0usize; // live particles after warm-up
    let mut particle_scenes_lit = 0usize; // scenes with particles that render content

    for item in &items {
        let id = item.file_name().unwrap().to_string_lossy().into_owned();
        let (_pkg, model, _bag) = match load_model(item, &assets) {
            Ok(m) => m,
            Err(e) => {
                errored += 1;
                lines.push(format!("  {id:<12} ERROR  {e}"));
                continue;
            }
        };
        let scene_particle_objs = model
            .scene
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::Particle(_)))
            .count();
        total_particle_objs += scene_particle_objs;
        let target = RenderTarget {
            device: &device,
            queue: &queue,
            format: FORMAT,
            output_name: "corpus",
            size: (1920, 1080),
        };
        let mut renderer = match SceneRenderer::new(
            &target,
            &model,
            &CompositeSource {
                pkg: &_pkg,
                assets: assets.clone(),
            },
            SceneOptions {
                render_scale: 1.0,
                scaling: ScalingMode::Fill,
                clamp: ClampMode::Clamp,
                disable_parallax: false,
            },
            None,
            &[],
        ) {
            Ok(r) => r,
            Err(e) => {
                errored += 1;
                lines.push(format!("  {id:<12} ERROR  build: {e}"));
                continue;
            }
        };
        let (pw, ph) = renderer.projection_size();
        let pixels = render_and_read(&device, &queue, &mut renderer, OUT_W, OUT_H);
        let stats = classify(&pixels);

        // Sampled after the warm-up burst: proves particle sims populated.
        let particles = renderer.debug_particle_count();
        let texts = renderer.debug_text_count();
        let live = renderer.debug_live_particles();
        wired_particle_items += particles;
        wired_text_items += texts;
        total_live_particles += live;

        let extra = if particles > 0 || texts > 0 {
            format!(" [{particles} particle ({live} live), {texts} text]")
        } else {
            String::new()
        };
        if stats.non_background > 0.01 && stats.any_lit {
            rendered += 1;
            if particles > 0 {
                particle_scenes_lit += 1;
            }
            lines.push(format!(
                "  {id:<12} OK     {pw}x{ph} proj, {:.1}% content{extra}",
                stats.non_background * 100.0
            ));
        } else {
            clear_only += 1;
            lines.push(format!(
                "  {id:<12} FLAT   {pw}x{ph} proj, bg=({},{},{}) passes={}{extra}",
                pixels[0],
                pixels[1],
                pixels[2],
                renderer.debug_pass_count()
            ));
        }
    }

    eprintln!(
        "\nscene corpus: {} items — {rendered} rendered, {clear_only} flat, {errored} errored\n\
         non-image compositing: {wired_particle_items} particle items wired \
         ({total_particle_objs} particle objects in corpus), {total_live_particles} live \
         particles after warm-up, {wired_text_items} text items, \
         {particle_scenes_lit} particle scenes rendered content\n{}",
        items.len(),
        lines.join("\n")
    );

    // The pipeline must render at least one of the simplest scenes to plausible,
    // non-background pixels end-to-end on a real adapter.
    assert!(
        rendered >= 1,
        "no corpus scene produced non-background content ({} built-but-flat, {} errored)",
        clear_only,
        errored
    );

    // SPEC §T16b: particle objects (docs count 58 across the corpus) must now be
    // composited, not skipped — the renderer builds a sim per particle object
    // and those sims populate over the warm-up burst (the biggest visual win).
    if total_particle_objs > 0 {
        assert!(
            wired_particle_items > 0,
            "corpus has {total_particle_objs} particle objects but the scene renderer wired in none"
        );
        assert!(
            total_live_particles > 0,
            "particle systems were wired in ({wired_particle_items}) but none spawned any \
             particles over the warm-up burst"
        );
    }
}
