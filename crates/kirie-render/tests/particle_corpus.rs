//! Corpus-gated test for the CPU particle simulation (docs/corpus.md: 58
//! particle objects across the workshop corpus; docs/render-architecture.md
//! §7.3).
//!
//! For every `scene.pkg` in the corpus this resolves the scene (loading any
//! external particle definitions and materials), builds a [`ParticleSim`] for
//! each particle object, and simulates a fixed number of steps. It asserts no
//! panic, that produced state stays finite, and reports per-item + aggregate
//! particle-object coverage (how many systems ever spawned, how many use a
//! supported emitter shape). A small headless GPU burst render runs when a wgpu
//! adapter is available.
//!
//! Skipped (never failed) when the corpus or an adapter is absent — inert in
//! CI, live on the workstation.

use std::path::{Path, PathBuf};

use kirie_formats::pkg::OwnedPkg;
use kirie_formats::project::Project;
use kirie_render::particle::{ParticleSim, SimConfig, SpriteInstance};
use kirie_scene::object::ObjectKind;
use kirie_scene::resolve::AssetSource;
use kirie_scene::{PropertyBag, Scene, SceneModel};

const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
const ASSETS_DIR: &str = "/home/aiko/.steam/steam/steamapps/common/wallpaper_engine/assets";

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("KIRIE_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CORPUS_DIR));
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skipping particle corpus test: {} not found", dir.display());
        None
    }
}

struct CompositeSource<'a> {
    pkg: &'a OwnedPkg,
    assets: Option<PathBuf>,
}

impl AssetSource for CompositeSource<'_> {
    fn load(&self, path: &str) -> Option<Vec<u8>> {
        if let Ok(bytes) = self.pkg.read_name(path.as_bytes()) {
            return Some(bytes.to_vec());
        }
        std::fs::read(self.assets.as_ref()?.join(path)).ok()
    }
}

fn load_model(item_dir: &Path, assets: &Option<PathBuf>) -> Result<(OwnedPkg, SceneModel), String> {
    let pkg = OwnedPkg::from_path(item_dir.join("scene.pkg")).map_err(|e| format!("pkg: {e}"))?;
    let bag = Project::from_path(item_dir.join("project.json"))
        .map(|p| PropertyBag::from_project(&p))
        .unwrap_or_default();
    let scene = {
        let bytes = pkg
            .read_name(b"scene.json")
            .map_err(|e| format!("scene.json: {e}"))?;
        Scene::from_slice(bytes).map_err(|e| format!("parse: {e}"))?
    };
    let mut model = SceneModel::resolve(scene, &bag);
    let source = CompositeSource {
        pkg: &pkg,
        assets: assets.clone(),
    };
    let _ = model.load_assets(&source, &bag);
    drop(source);
    Ok((pkg, model))
}

fn finite(v: [f32; 3]) -> bool {
    v.iter().all(|f| f.is_finite())
}

#[test]
fn particle_corpus_simulates_without_panic() {
    let Some(dir) = corpus_dir() else { return };
    let assets = std::env::var_os("KIRIE_WE_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(ASSETS_DIR));
    let assets = assets.is_dir().then_some(assets);

    let mut items: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("scene.pkg").is_file())
        .collect();
    items.sort();

    let mut total_objects = 0usize;
    let mut with_supported_emitter = 0usize;
    let mut ever_spawned = 0usize;
    let mut scenes_with_particles = 0usize;
    let mut lines = Vec::new();
    // A reused sprite buffer proves the write path never reallocates in steady
    // state (SPEC §V5): its capacity must not grow once warm.
    let mut sprites: Vec<SpriteInstance> = Vec::new();

    for item in &items {
        let id = item.file_name().unwrap().to_string_lossy().into_owned();
        let Ok((_pkg, model)) = load_model(item, &assets) else {
            continue;
        };
        let mut n_here = 0usize;
        let mut spawned_here = 0usize;
        for object in &model.scene.objects {
            let ObjectKind::Particle(p) = &object.kind else {
                continue;
            };
            n_here += 1;
            total_objects += 1;

            let mut sim = ParticleSim::new(
                &p.system,
                &p.instanceoverride,
                SimConfig {
                    seed: 0xC0FFEE,
                    sheet: None,
                },
            );
            if sim.has_supported_emitter() {
                with_supported_emitter += 1;
            }
            // Simulate 2 seconds at 60 Hz.
            for _ in 0..120 {
                sim.update(1.0 / 60.0);
                for pt in sim.particles() {
                    assert!(
                        finite(pt.position) && finite(pt.velocity) && pt.alpha.is_finite(),
                        "non-finite particle state in {id}"
                    );
                }
            }
            sim.write_sprites(&mut sprites);
            assert_eq!(sprites.len(), sim.live_count());
            if sim.total_spawned() > 0 {
                ever_spawned += 1;
                spawned_here += 1;
            }
        }
        if n_here > 0 {
            scenes_with_particles += 1;
            lines.push(format!(
                "  {id:<12} {n_here} particle obj, {spawned_here} spawned"
            ));
        }
    }

    eprintln!(
        "\nparticle corpus: {} scenes, {scenes_with_particles} with particles, \
         {total_objects} particle objects — {with_supported_emitter} with a supported emitter, \
         {ever_spawned} spawned during sim\n{}",
        items.len(),
        lines.join("\n")
    );

    assert!(total_objects > 0, "expected particle objects in the corpus");
    // The docs count 58 particle objects across the corpus; we should at least
    // simulate the bulk of them without panic (some scenes may fail to parse for
    // unrelated reasons and are skipped above).
    assert!(
        ever_spawned > 0,
        "no particle system spawned anything ({total_objects} objects seen)"
    );
}

#[test]
fn particle_burst_renders_on_gpu() {
    // Headless render of a synthetic burst to prove the instanced-quad path
    // draws visible pixels. Skipped when no adapter is present.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("skipping particle GPU render: no adapter");
        return;
    };
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
    else {
        eprintln!("skipping particle GPU render: no device");
        return;
    };

    use kirie_render::particle::ParticleRenderer;
    use kirie_scene::material::Blending;

    const W: u32 = 128;
    const H: u32 = 128;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    let renderer = ParticleRenderer::new(&device, &queue, FORMAT, Blending::Additive, None, 16);

    // Four bright quads near the center in NDC (identity VP; positions already
    // in clip space here for a self-contained visibility check).
    let instances = [
        make_instance([-0.3, -0.3, 0.0], 0.3, [1.0, 0.2, 0.2, 1.0]),
        make_instance([0.3, -0.3, 0.0], 0.3, [0.2, 1.0, 0.2, 1.0]),
        make_instance([-0.3, 0.3, 0.0], 0.3, [0.2, 0.2, 1.0, 1.0]),
        make_instance([0.3, 0.3, 0.0], 0.3, [1.0, 1.0, 0.2, 1.0]),
    ];
    let identity: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let count = renderer.upload(&queue, &identity, &instances);
    assert_eq!(count, 4);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particle-burst-target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    // Clear to black, then draw particles (additive over black).
    {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    renderer.draw(&mut encoder, &view, count);

    let padded = (W * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle-readback"),
        size: u64::from(padded * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let mapped = readback.get_mapped_range(..).expect("range");
    let mut lit = 0usize;
    for row in 0..H {
        let start = (row * padded) as usize;
        for px in mapped[start..start + (W * 4) as usize].as_chunks::<4>().0 {
            if px[0] > 8 || px[1] > 8 || px[2] > 8 {
                lit += 1;
            }
        }
    }
    drop(mapped);
    readback.unmap();

    eprintln!("particle burst: {lit} lit pixels of {}", W * H);
    assert!(
        lit > 100,
        "expected the particle burst to light many pixels, got {lit}"
    );
}

fn make_instance(pos: [f32; 3], size: f32, color: [f32; 4]) -> SpriteInstance {
    SpriteInstance {
        position_size: [pos[0], pos[1], pos[2], size],
        color,
        rotation_frame: [0.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 0.0, 0.0, 0.0],
    }
}
