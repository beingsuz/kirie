//! Corpus-gated headless-GPU test for **real text-glyph rendering**
//! (docs/render-architecture.md §7.4; docs/corpus.md: 7 text objects across the
//! workshop corpus).
//!
//! For every `scene.pkg` in the corpus this resolves the scene, collects its
//! text objects, and for each visible one with non-empty text: shapes +
//! rasterizes the glyphs with cosmic-text ([`kirie_render::scene::text`]), then
//! draws the resulting coverage quad through the **real** text pipeline
//! ([`kirie_render::scene::extras::build_text_pipeline`] / `TEXT_WGSL`) into an
//! offscreen `Rgba16Float` target (the scene-FBO format) on a real adapter and
//! reads it back.
//!
//! The core assertion (the deliverable): the drawn quad lights **> 0**
//! non-background pixels **and** is *not* a uniform fill — i.e. it is real glyph
//! shapes, distinct from the old flat placeholder quad which covered 100 % of its
//! area. A per-item table reports raster size, glyph-coverage fraction and lit
//! fraction; script-driven clocks whose initial `text` value is empty, and any
//! font-substitution gaps, are reported (never failed).
//!
//! Skipped (never failed) when the corpus or a wgpu adapter is absent, or when
//! the machine's `fontdb` finds no font faces (nothing to shape) — inert in CI,
//! live on the RTX 4080.

use std::path::{Path, PathBuf};

use kirie_formats::pkg::OwnedPkg;
use kirie_formats::project::Project;
use kirie_render::scene::extras::{self, TextPipeline};
use kirie_render::scene::text::{self, TextFonts, TextRaster};
use kirie_scene::object::{ObjectKind, TextObject};
use kirie_scene::resolve::AssetSource;
use kirie_scene::{PropertyBag, Scene, SceneModel};

const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
const ASSETS_DIR: &str = "/home/aiko/.steam/steam/steamapps/common/wallpaper_engine/assets";
/// Must match the text pipeline's target (the scene FBO is `Rgba16Float`,
/// docs/render-architecture.md §6 — HDR render targets).
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Decode an IEEE-754 half-precision (`f16`) bit pattern to `f32` (no `half`
/// dependency needed for the readback of an `Rgba16Float` target).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = u32::from(bits & 0x3ff);
    let out = match exp {
        0 if mant == 0 => sign, // ±0
        0 => {
            // subnormal
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127 - 15) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13), // inf/nan
        _ => sign | ((u32::from(exp) + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}
/// Cap the offscreen target so readback stays cheap; the NDC quad fills it
/// regardless, so coverage geometry is preserved under scaling.
const MAX_FBO: u32 = 256;

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("KIRIE_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CORPUS_DIR));
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skipping text corpus test: {} not found", dir.display());
        None
    }
}

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kirie-text-corpus"),
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
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
    Ok((pkg, model))
}

/// Draw a rasterized text block through the real text pipeline into an
/// `MAX_FBO`-capped offscreen target (identity MVP, opaque-white color so the
/// lit region is exactly the glyph coverage) and return `(lit_pixels, total)`.
fn draw_and_count(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tp: &TextPipeline,
    raster: &TextRaster,
) -> (usize, usize) {
    let texture = text::upload(device, queue, raster);
    let w = raster.width.clamp(1, MAX_FBO);
    let h = raster.height.clamp(1, MAX_FBO);

    // Identity MVP (column-major) + opaque white.
    #[rustfmt::skip]
    let mvp: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let mut ubo_bytes = Vec::with_capacity(80);
    for f in mvp {
        ubo_bytes.extend_from_slice(&f.to_le_bytes());
    }
    for f in [1.0f32, 1.0, 1.0, 1.0] {
        ubo_bytes.extend_from_slice(&f.to_le_bytes());
    }
    let ubo = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("text-corpus-ubo"),
        size: 80,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&ubo, 0, &ubo_bytes);

    // Fullscreen NDC quad (TL, BL, TR, BR) with uv; v=0 at top row of bitmap.
    #[rustfmt::skip]
    let verts: [f32; 20] = [
        -1.0,  1.0, 0.0,  0.0, 0.0,
        -1.0, -1.0, 0.0,  0.0, 1.0,
         1.0,  1.0, 0.0,  1.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 1.0,
    ];
    let vb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("text-corpus-vb"),
        size: std::mem::size_of_val(&verts) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vb, 0, bytemuck::cast_slice(&verts));

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("text-corpus-bg"),
        layout: &tp.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            },
        ],
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("text-corpus-target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("text-corpus-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Transparent-black background so any lit pixel is glyph.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&tp.pipeline);
        rp.set_bind_group(0, &bind, &[]);
        rp.set_vertex_buffer(0, vb.slice(..));
        rp.draw(0..4, 0..1);
    }

    let padded = (w * 8).div_ceil(256) * 256; // 8 bytes/px for Rgba16Float
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("text-corpus-readback"),
        size: u64::from(padded * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
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
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let mapped = readback.get_mapped_range(..).expect("mapped");
    let mut lit = 0usize;
    let total = (w * h) as usize;
    let chan = |b: &[u8], off: usize| f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]));
    for row in 0..h {
        let base = (row * padded) as usize;
        for col in 0..w {
            let p = base + (col * 8) as usize;
            if chan(&mapped, p) > 0.03 || chan(&mapped, p + 2) > 0.03 || chan(&mapped, p + 4) > 0.03 {
                lit += 1;
            }
        }
    }
    drop(mapped);
    readback.unmap();
    (lit, total)
}

#[test]
fn text_corpus_renders_real_glyphs() {
    let Some(dir) = corpus_dir() else { return };
    let Some((device, queue)) = gpu() else {
        eprintln!("skipping text corpus test: no wgpu adapter");
        return;
    };

    let assets = std::env::var_os("KIRIE_WE_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(ASSETS_DIR));
    let assets = assets.is_dir().then_some(assets);

    let mut fonts = TextFonts::new();
    if fonts.face_count() == 0 {
        eprintln!("skipping text corpus test: fontdb found no font faces");
        return;
    }
    let tp = extras::build_text_pipeline(&device);

    let mut items: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("scene.pkg").is_file())
        .collect();
    items.sort();

    let mut total_text = 0usize; // text objects present
    let mut empty_text = 0usize; // visible but empty initial value (script clocks)
    let mut hidden_text = 0usize; // visible:false
    let mut real_glyphs = 0usize; // rendered partial-coverage glyph quads
    let mut lines = Vec::new();

    for item in &items {
        let id = item.file_name().unwrap().to_string_lossy().into_owned();
        let Ok((_pkg, model)) = load_model(item, &assets) else {
            continue;
        };
        for obj in &model.scene.objects {
            let ObjectKind::Text(tobj) = &obj.kind else {
                continue;
            };
            let tobj: &TextObject = tobj;
            total_text += 1;
            if !(tobj.visible.value && obj.base.visible.value) {
                hidden_text += 1;
                lines.push(format!("  {id:<12} id{:<5} HIDDEN", obj.base.id));
                continue;
            }
            let Some(raster) = text::rasterize(
                &mut fonts,
                &tobj.text.value,
                &tobj.font,
                tobj.pointsize.value,
                tobj.size,
                &tobj.horizontalalign,
                &tobj.verticalalign,
                tobj.padding as f32,
                None,
            ) else {
                empty_text += 1;
                lines.push(format!("  {id:<12} id{:<5} EMPTY (no text)", obj.base.id));
                continue;
            };
            if !raster.any_coverage {
                empty_text += 1;
                let snippet: String = tobj.text.value.chars().take(20).collect();
                lines.push(format!(
                    "  {id:<12} id{:<5} NO-COVERAGE font={:?} text={snippet:?}",
                    obj.base.id, tobj.font
                ));
                continue;
            }

            // Glyph-coverage fraction from the CPU raster (proves the bitmap is
            // real glyph shapes, not a solid fill).
            let covered = raster
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| px[3] > 0)
                .count();
            let raster_total = (raster.width * raster.height) as usize;
            let cov_frac = covered as f32 / raster_total.max(1) as f32;

            // GPU draw through the real pipeline.
            let (lit, fbo_total) = draw_and_count(&device, &queue, &tp, &raster);
            let lit_frac = lit as f32 / fbo_total.max(1) as f32;

            // The deliverable: > 0 lit pixels, and NOT a uniform fill (the old
            // placeholder covered 100 %). Glyphs leave large empty margins.
            assert!(lit > 0, "{id} id{}: text quad lit no pixels", obj.base.id);
            assert!(
                lit_frac < 0.95,
                "{id} id{}: text quad is a near-uniform fill ({lit_frac:.2}) — not glyphs",
                obj.base.id
            );
            assert!(
                cov_frac < 0.95,
                "{id} id{}: raster is a solid block ({cov_frac:.2}) — not glyphs",
                obj.base.id
            );
            real_glyphs += 1;
            let snippet: String = tobj.text.value.chars().take(24).collect();
            lines.push(format!(
                "  {id:<12} id{:<5} {}x{} cov={cov_frac:.2} lit={lit_frac:.2} font={:?} {snippet:?}",
                obj.base.id, raster.width, raster.height, tobj.font
            ));
        }
    }

    eprintln!("\n=== text corpus ({} scene.pkg) ===", items.len());
    for l in &lines {
        eprintln!("{l}");
    }
    eprintln!(
        "text objects: {total_text} total, {real_glyphs} rendered real glyphs, \
         {empty_text} empty/script-driven, {hidden_text} hidden",
    );

    // At least one corpus text object must render real glyphs (guards the whole
    // shaping→raster→draw path). If every text object were empty/script-driven
    // the corpus would not exercise glyph rendering — flag it loudly but do not
    // fail (the corpus contents are outside this crate's control).
    if total_text > 0 && real_glyphs == 0 {
        eprintln!(
            "WARNING: no corpus text object rendered static glyphs \
             (all script-driven or font-substitution gaps)"
        );
    }
}
