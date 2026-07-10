//! 3D MODEL object rendering — the `.mdl` mesh path (docs/render-architecture.md
//! §7.2; the reference `Render/Objects/CModel.cpp`).
//!
//! A model object names a binary `.mdl` (parsed by [`kirie_formats::model`]).
//! Each sub-mesh carries a material path whose first pass is a `generic3`
//! program; this module builds one wgpu pipeline per mesh (via
//! [`super::pipeline::build_model_pass`], which accepts the full 48-byte vertex
//! layout the 2D path rejects), uploads the mesh's vertex/index buffers, and
//! draws every mesh in `.mdl` declaration order into the scene FBO under a
//! **perspective** camera with a private depth buffer — exactly the reference's
//! `CModel::render`.
//!
//! # The two hard-won gotchas (user notes; `CModel.cpp` comments)
//!
//! 1. **Winding / cull.** The `.mdl` triangles are authored for the reference,
//!    which flips clip-space Y for its Y-down scene FBO (`projection[1][1] *=
//!    -1`) and therefore declares CW front-facing so back-face culling keeps the
//!    real front faces. kirie's scene FBO is Y-up (the 2D layers build Y-up quads
//!    through [`super::matrix::ortho`] with no flip), so [`super::matrix::perspective`]
//!    applies **no** Y flip and the natural winding is CCW-front — matching
//!    kirie's default [`wgpu::FrontFace::Ccw`]. Cull mode comes from the material
//!    (`normal` ⇒ cull back). Get the flip and the winding out of sync and the
//!    figure is culled to invisibility.
//! 2. **Depth-clip half-coloring.** With no depth buffer the sub-meshes composite
//!    in draw order and the translucent shell whitens the opaque body; with the
//!    wrong depth range half the mesh is near/far-clipped. The perspective is
//!    wgpu zero-to-one (`perspectiveRH_ZO`) and the model gets its own
//!    `Depth24Plus` buffer cleared to 1.0 each frame so `LessEqual` occlusion is
//!    correct and nothing is clipped.
//!
//! The Starscape figure's material renders it a near-black silhouette (its tint
//! is black / the shell is unlit) — that dark shape is correct, not a bug.

use std::collections::HashMap;

use kirie_audio::AudioSpectrum;
use kirie_scene::material::Material;
use kirie_scene::object::{ModelObject, Object};
use kirie_scene::resolve::AssetSource;
use kirie_shader::IncludeResolver;

use super::fbo::{FBO_FORMAT, Fbo};
use super::matrix::{self, Mat4};
use super::pipeline::{self, BuiltPass};
use super::renderer::{
    build_bind_group, create_buffer_init, create_ubo, is_scene_rt, resolve_params, tex_res,
};
use super::texture::TextureRegistry;
use super::uniforms::{Builtins, GlobalsLayout, pack_globals};

/// The model's private depth attachment format (the reference allocates
/// `GL_DEPTH_COMPONENT24`, `CModel::ensureDepthBuffer`). `Depth24Plus` is the
/// portable wgpu equivalent and matches [`super::pipeline::build_model_pass`].
pub(super) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

/// The neutral camera clamps the reference applies before building the model
/// projection (`CModel::render`): fov ∈ [1, 170] else 50; near > 0 else 0.1;
/// far > near else 10000.
fn clamp_camera(fov: f32, near: f32, far: f32) -> (f32, f32, f32) {
    let fov = if (1.0..=170.0).contains(&fov) { fov } else { 50.0 };
    let near = if near > 0.0 { near } else { 0.1 };
    let far = if far > near { far } else { 10000.0 };
    (fov, near, far)
}

/// One drawable sub-mesh: its pipeline, static bind groups, per-frame UBOs and
/// geometry buffers.
struct MeshGpu {
    pipeline: wgpu::RenderPipeline,
    g0_bind: wgpu::BindGroup,
    g1_bind: wgpu::BindGroup,
    vs_ubo: Option<wgpu::Buffer>,
    fs_ubo: Option<wgpu::Buffer>,
    vs_globals: GlobalsLayout,
    fs_globals: GlobalsLayout,
    vs_params: std::collections::BTreeMap<String, Vec<f32>>,
    fs_params: std::collections::BTreeMap<String, Vec<f32>>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// `g_TextureNResolution` per slot for this mesh's material.
    tex_resolution: [[f32; 4]; 8],
}

/// One renderable 3D model object: its sub-meshes plus the object transform the
/// per-frame model matrix is built from (`CModel::computeModelMatrix`).
pub(super) struct ModelGpu {
    meshes: Vec<MeshGpu>,
    /// Object origin, world space (`origin`).
    origin: [f32; 3],
    /// Object scale (`scale`).
    scale: [f32; 3],
    /// Static base angles in radians (`angles`; the keyframe animation is a
    /// documented gap — at t≈0 it evaluates to these base angles anyway).
    angles: [f32; 3],
    /// Live visibility (`visible`; false ⇒ the whole model is skipped).
    pub(super) visible: bool,
    /// True when a mesh's material samples `_rt_FullFrameBuffer` /
    /// `_rt_MipMappedFrameBuffer` (generic3 REFLECTION): the scene composited so
    /// far is snapshotted before the model draws so the read never aliases the
    /// write (docs §6/§11 shadow-copy; `CModel::render` blits the scene FBO).
    pub(super) reads_scene: bool,
}

/// Build a 3D model object's GPU resources, or `None` when nothing is drawable
/// (missing/invalid `.mdl`, no buildable mesh — SPEC.md §V9 skip-and-continue).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_model(
    device: &wgpu::Device,
    object: &Object,
    model_object: &ModelObject,
    scene_size: (u32, u32),
    source: &dyn AssetSource,
    resolver: &dyn IncludeResolver,
    registry: &mut TextureRegistry,
    fbo_sampler: &wgpu::Sampler,
    scene_snapshot: &Fbo,
) -> Option<ModelGpu> {
    let bytes = source.load(&model_object.model)?;
    let model = match kirie_formats::model::Model::parse(&bytes) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(model = %model_object.model, error = %e, "model .mdl parse failed; skipped");
            return None;
        }
    };

    let mut reads_scene = false;
    let mut meshes = Vec::new();
    // Draw in `.mdl` declaration order — load-bearing for depth (the opaque body
    // must draw before the translucent shell; `CModel::setup`).
    for (mi, mesh) in model.meshes.iter().enumerate() {
        // Load and parse the mesh's material JSON; take its first pass.
        let Some(mat_bytes) = source.load(&mesh.material_ref) else {
            tracing::debug!(material = %mesh.material_ref, "model material missing; mesh skipped");
            continue;
        };
        let Ok(mat_value) = serde_json::from_slice::<serde_json::Value>(&mat_bytes) else {
            tracing::debug!(material = %mesh.material_ref, "model material invalid JSON; mesh skipped");
            continue;
        };
        let material = Material::from_value(&mat_value);
        let Some(raw_pass) = material.passes.first().cloned() else {
            tracing::debug!(material = %mesh.material_ref, "model material has no pass; mesh skipped");
            continue;
        };

        // Translate + build the mesh pipeline (generic3, 48-byte vertex layout,
        // triangle list, depth, cull per material — `build_model_pass`).
        let vs_name = format!("shaders/{}.vert", raw_pass.shader);
        let fs_name = format!("shaders/{}.frag", raw_pass.shader);
        let (Some(vs_bytes), Some(fs_bytes)) = (source.load(&vs_name), source.load(&fs_name)) else {
            tracing::debug!(shader = %raw_pass.shader, "model shader source missing; mesh skipped");
            continue;
        };
        let (Ok(vs_src), Ok(fs_src)) = (String::from_utf8(vs_bytes), String::from_utf8(fs_bytes)) else {
            continue;
        };
        let built = match pipeline::build_model_pass(
            device,
            FBO_FORMAT,
            DEPTH_FORMAT,
            &raw_pass,
            &vs_src,
            &fs_src,
            resolver,
        ) {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(mesh = mi, shader = %raw_pass.shader, error = %e, "model mesh pipeline failed; skipped");
                continue;
            }
        };

        // Geometry buffers: the raw interleaved vertex bytes upload verbatim
        // (`CModel::setupMesh`), the u16 index list as-is.
        let vertex_buffer = create_buffer_init(
            device,
            "kirie-model-vb",
            &mesh.vertex_data,
            wgpu::BufferUsages::VERTEX,
        );
        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
        let index_buffer =
            create_buffer_init(device, "kirie-model-ib", index_bytes, wgpu::BufferUsages::INDEX);
        let index_count = mesh.indices.len() as u32;

        // The albedo/base input (`g_Texture0`, default `util/white`); the model
        // materials declare `textures:[null]`, so slot 0 is the neutral white the
        // reference resolves too (`CModel::setup`).
        let input = registry.white();

        // `_rt_` reflection binds resolve to the scene snapshot; generic3's
        // `g_Texture3` defaults to `_rt_MipMappedFrameBuffer` and REFLECTION
        // samples `_rt_FullFrameBuffer` (`CModel::setup` aliases both to a shadow
        // copy). Any such bind flags the model as reading the scene.
        let mut named: HashMap<&str, (&wgpu::TextureView, &wgpu::Sampler)> = HashMap::new();
        named.insert("_rt_FullFrameBuffer", (&scene_snapshot.view, fbo_sampler));
        named.insert("_rt_MipMappedFrameBuffer", (&scene_snapshot.view, fbo_sampler));
        if built
            .fs_samplers
            .iter()
            .chain(built.vs_samplers.iter())
            .any(|s| s.default_texture.as_deref().is_some_and(is_scene_rt))
            || raw_pass.textures.iter().flatten().any(|n| is_scene_rt(n))
        {
            reads_scene = true;
        }

        // Per-stage UBOs sized to the shader's `_WEGlobals` block.
        let vs_ubo = (!built.vs_globals.is_empty()).then(|| create_ubo(device, built.vs_globals.size));
        let fs_ubo = (!built.fs_globals.is_empty()).then(|| create_ubo(device, built.fs_globals.size));

        let g0_bind = build_bind_group(
            device,
            &built.g0_layout,
            vs_ubo.as_ref(),
            &built.g0_bindings,
            &built.vs_samplers,
            &input.view,
            &input.sampler,
            registry,
            source,
            &raw_pass,
            (&scene_snapshot.view, fbo_sampler),
            &named,
        );
        let g1_bind = build_bind_group(
            device,
            &built.g1_layout,
            fs_ubo.as_ref(),
            &built.g1_bindings,
            &built.fs_samplers,
            &input.view,
            &input.sampler,
            registry,
            source,
            &raw_pass,
            (&scene_snapshot.view, fbo_sampler),
            &named,
        );

        let tex_resolution =
            build_tex_resolution(&built, &raw_pass, scene_size, registry, source, input.as_ref());

        let vs_params = resolve_params(&built.vs_params, &raw_pass);
        let fs_params = resolve_params(&built.fs_params, &raw_pass);

        let BuiltPass {
            pipeline,
            vs_globals,
            fs_globals,
            ..
        } = built;

        meshes.push(MeshGpu {
            pipeline,
            g0_bind,
            g1_bind,
            vs_ubo,
            fs_ubo,
            vs_globals,
            fs_globals,
            vs_params,
            fs_params,
            vertex_buffer,
            index_buffer,
            index_count,
            tex_resolution,
        });
    }

    if meshes.is_empty() {
        return None;
    }

    let origin = object.base.origin.value;
    let scale = object.base.scale.value;
    let angles = object.base.angles.value;
    let visible = object.base.visible.value;
    tracing::debug!(
        id = object.base.id,
        model = %model_object.model,
        meshes = meshes.len(),
        "built 3D model object"
    );
    Some(ModelGpu {
        meshes,
        origin,
        scale,
        angles,
        visible,
        reads_scene,
    })
}

/// Compute `g_TextureNResolution` per slot for a mesh material (docs §8.3): slot
/// 0 is the albedo input; slots 1.. resolve from the pass textures / sampler
/// defaults. FBO/composite refs are clean render targets at the scene size; real
/// `.tex` assets carry their own (padded) size so shaders can crop NPOT padding.
fn build_tex_resolution(
    built: &BuiltPass,
    pass: &kirie_scene::material::Pass,
    scene_size: (u32, u32),
    registry: &mut TextureRegistry,
    source: &dyn AssetSource,
    input: &super::texture::GpuTexture,
) -> [[f32; 4]; 8] {
    let scene_res = [
        scene_size.0 as f32,
        scene_size.1 as f32,
        scene_size.0 as f32,
        scene_size.1 as f32,
    ];
    let mut out = [scene_res; 8];
    out[0] = tex_res(input);
    for slot in &built.fs_samplers {
        let Some(i) = slot.slot else { continue };
        let i = i as usize;
        if i == 0 || i >= 8 {
            continue;
        }
        let name = pass
            .textures
            .get(i)
            .and_then(|s| s.clone())
            .or_else(|| slot.default_texture.clone());
        out[i] = match name {
            Some(n) if n.starts_with("_rt_") || n.starts_with("_alias_") => scene_res,
            Some(n) => tex_res(&registry.get(&n, source)),
            None => scene_res,
        };
    }
    out
}

/// Draw one model's sub-meshes into the scene FBO under the perspective camera
/// (`CModel::render`). `depth_view` is the shared model depth buffer; it is
/// cleared to 1.0 at pass begin so `LessEqual` occlusion is correct.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_model(
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    model: &ModelGpu,
    scene_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    camera: &kirie_scene::scene::Camera,
    aspect: f32,
    ambient: [f32; 3],
    skylight: [f32; 3],
    time: f32,
    texel: [f32; 2],
    audio: Option<&AudioSpectrum>,
) {
    let (fov, near, far) = clamp_camera(camera.fov, camera.nearz, camera.farz);
    let projection = matrix::perspective(fov.to_radians(), aspect, near, far);
    let view = matrix::look_at(camera.eye, camera.center, camera.up);
    let view_projection = matrix::mul(&projection, &view);
    let model_matrix = compute_model_matrix(model.origin, model.angles, model.scale);
    let mvp = matrix::mul(&view_projection, &model_matrix);

    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("kirie-model-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: scene_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // Composite onto the scene so far (like every other kind).
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    for mesh in &model.meshes {
        // Neutral CRenderable identity values; the material constants style the
        // mesh (`CModel.h`: brightness/alpha = 1, color = white).
        let builtins = Builtins {
            time,
            daytime: 0.0,
            brightness: 1.0,
            alpha: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            ambient,
            skylight,
            pointer: [0.5, 0.5],
            pointer_last: [0.5, 0.5],
            texel_size: texel,
            mvp,
            model: model_matrix,
            view_projection,
            eye: camera.eye,
            texture0_translation: [0.0, 0.0],
            texture0_rotation: [0.0, 0.0, 0.0, 0.0],
            texture_resolution: mesh.tex_resolution,
            audio16: audio.map_or([0.0; 16], |a| a.audio16),
            audio32: audio.map_or([0.0; 32], |a| a.audio32),
            audio64: audio.map_or([0.0; 64], |a| a.audio64),
        };
        if let Some(ubo) = &mesh.vs_ubo {
            queue.write_buffer(
                ubo,
                0,
                &pack_globals(&mesh.vs_globals, &builtins, &mesh.vs_params),
            );
        }
        if let Some(ubo) = &mesh.fs_ubo {
            queue.write_buffer(
                ubo,
                0,
                &pack_globals(&mesh.fs_globals, &builtins, &mesh.fs_params),
            );
        }
        rp.set_pipeline(&mesh.pipeline);
        rp.set_bind_group(0, &mesh.g0_bind, &[]);
        rp.set_bind_group(1, &mesh.g1_bind, &[]);
        rp.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rp.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..mesh.index_count, 0, 0..1);
    }
}

/// The object model matrix (`CModel::computeModelMatrix`):
/// `translate(origin) · rotZ · rotY · rotX · scale`.
fn compute_model_matrix(origin: [f32; 3], angles: [f32; 3], scale: [f32; 3]) -> Mat4 {
    let mut m = matrix::translation(origin);
    m = matrix::mul(&m, &matrix::rotation_z(angles[2]));
    m = matrix::mul(&m, &matrix::rotation_y(angles[1]));
    m = matrix::mul(&m, &matrix::rotation_x(angles[0]));
    matrix::mul(&m, &matrix::scale(scale))
}

/// Allocate the model's private depth buffer (the reference's
/// `ensureDepthBuffer`, `CModel::render`): a `Depth24Plus` render target at the
/// scene size, built once and reused every frame (SPEC.md §V5).
pub(super) fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kirie-model-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_clamps_match_reference() {
        // In-range values pass through; out-of-range fall back (CModel::render).
        assert_eq!(clamp_camera(50.0, 0.01, 11.0), (50.0, 0.01, 11.0));
        assert_eq!(clamp_camera(0.0, 0.0, -1.0), (50.0, 0.1, 10000.0));
        assert_eq!(clamp_camera(200.0, 0.01, 0.005), (50.0, 0.01, 10000.0));
    }

    #[test]
    fn model_matrix_places_origin() {
        // A pure origin translation moves the world origin to `origin`.
        let m = compute_model_matrix([1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Column-major: translation lives in the 4th column (indices 12,13,14).
        assert_eq!([m[12], m[13], m[14]], [1.0, -2.0, 3.0]);
    }

    /// Isolated headless diagnostic (ignored): build and draw ONLY the Starscape
    /// model onto a magenta `Rgba16F` target (bypassing the rest-of-scene compose
    /// so a black background can't hide it), then report where the mesh drew. Any
    /// pixel that differs from pure magenta is geometry the model produced —
    /// independent of shading — so this proves the winding/transform/depth path.
    #[test]
    #[ignore = "heavy GPU + corpus diagnostic; run manually with --ignored"]
    fn model_only_on_magenta() {
        use kirie_scene::object::ObjectKind;
        use kirie_scene::resolve::AssetSource;
        use kirie_scene::{PropertyBag, Scene, SceneModel};
        use kirie_shader::IncludeResolver;

        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("skip: no adapter");
            return;
        };
        let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("model-only-diag"),
            ..wgpu::DeviceDescriptor::default()
        })) else {
            eprintln!("skip: no device");
            return;
        };

        let scene_dir =
            std::path::Path::new("/home/aiko/.steam/steam/steamapps/workshop/content/431960/3047596375");
        let assets = std::path::PathBuf::from(
            "/home/aiko/.local/share/Steam/steamapps/common/wallpaper_engine/assets",
        );
        let pkg = match kirie_formats::pkg::OwnedPkg::from_path(scene_dir.join("scene.pkg")) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: corpus absent");
                return;
            }
        };

        struct Src {
            pkg: kirie_formats::pkg::OwnedPkg,
            assets: std::path::PathBuf,
        }
        impl AssetSource for Src {
            fn load(&self, path: &str) -> Option<Vec<u8>> {
                if let Ok(b) = self.pkg.read_name(path.as_bytes()) {
                    return Some(b.to_vec());
                }
                std::fs::read(self.assets.join(path)).ok()
            }
        }
        struct Inc<'a>(&'a dyn AssetSource);
        impl IncludeResolver for Inc<'_> {
            fn resolve(&self, name: &str) -> Option<String> {
                String::from_utf8(self.0.load(&format!("shaders/{name}"))?).ok()
            }
        }

        let scene_bytes = pkg.read_name(b"scene.json").expect("scene.json").to_vec();
        let scene = Scene::from_slice(&scene_bytes).expect("parse scene");
        let model = SceneModel::resolve(scene, &PropertyBag::default());
        let (obj, mo) = model
            .scene
            .objects
            .iter()
            .find_map(|o| match &o.kind {
                ObjectKind::Model(m) => Some((o, m)),
                _ => None,
            })
            .expect("a model object");

        let src = Src { pkg, assets };
        let resolver = Inc(&src);
        let mut registry = TextureRegistry::new(&device, &queue);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        // Oracle-matching frame (oracle-3047596375.png is 636×692).
        let (w, h) = (636u32, 692u32);
        let snapshot = Fbo::new(&device, "diag-snap", w, h);
        let mg = build_model(
            &device,
            obj,
            mo,
            (w, h),
            &src,
            &resolver,
            &mut registry,
            &sampler,
            &snapshot,
        )
        .expect("build_model returned None");
        eprintln!("model built: {} mesh(es)", mg.meshes.len());

        let color = Fbo::new(&device, "diag-color", w, h);
        let depth = create_depth_texture(&device, w, h);
        // Clear to magenta, then draw the model on top.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let _c = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("diag-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        let aspect = w as f32 / h as f32;
        draw_model(
            &mut enc,
            &queue,
            &mg,
            &color.view,
            &depth,
            &model.scene.camera,
            aspect,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0,
            [1.0 / w as f32, 1.0 / h as f32],
            None,
        );
        queue.submit(Some(enc.finish()));

        // Read back the Rgba16F target (8 bytes/texel, f16 channels).
        let padded = (w * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diag-rb"),
            size: u64::from(padded * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
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
        queue.submit(Some(enc.finish()));
        buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mapped = buffer.get_mapped_range(..).expect("range");

        let f16 = |bytes: &[u8]| -> f32 {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            let sign = (bits >> 15) & 1;
            let exp = (bits >> 10) & 0x1f;
            let man = bits & 0x3ff;
            let val = if exp == 0 {
                f32::from(man) * 2f32.powi(-24)
            } else if exp == 31 {
                f32::INFINITY
            } else {
                (1.0 + f32::from(man) / 1024.0) * 2f32.powi(i32::from(exp) - 15)
            };
            if sign == 1 { -val } else { val }
        };

        // Save an 8-bit PNG so the figure/water placement can be eyeballed
        // against the oracle (magenta = untouched background).
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let off = (y * padded + x * 8) as usize;
                let px = ((y * w + x) * 4) as usize;
                for c in 0..3 {
                    let v = f16(&mapped[off + c * 2..]).clamp(0.0, 1.0);
                    rgba[px + c] = (v * 255.0) as u8;
                }
                rgba[px + 3] = 255;
            }
        }
        let png_path = std::env::temp_dir().join("kirie-model-only.png");
        let _ = image::save_buffer(&png_path, &rgba, w, h, image::ColorType::Rgba8);
        eprintln!("wrote {}", png_path.display());

        let (mut minx, mut miny, mut maxx, mut maxy) = (w, h, 0u32, 0u32);
        let mut drew = 0u64;
        let cell = w / 48;
        let mut grid = String::new();
        for gy in 0..48u32 {
            for gx in 0..48u32 {
                let px = (gx * cell + cell / 2).min(w - 1);
                let py = (gy * cell + cell / 2).min(h - 1);
                let off = (py * padded + px * 8) as usize;
                let (r, g, b) = (
                    f16(&mapped[off..]),
                    f16(&mapped[off + 2..]),
                    f16(&mapped[off + 4..]),
                );
                let diff = (r - 1.0).abs() + g.abs() + (b - 1.0).abs();
                grid.push(if diff > 0.1 { '#' } else { ' ' });
            }
            grid.push('\n');
        }
        for y in 0..h {
            for x in 0..w {
                let off = (y * padded + x * 8) as usize;
                let (r, g, b) = (
                    f16(&mapped[off..]),
                    f16(&mapped[off + 2..]),
                    f16(&mapped[off + 4..]),
                );
                let diff = (r - 1.0).abs() + g.abs() + (b - 1.0).abs();
                if diff > 0.1 {
                    drew += 1;
                    minx = minx.min(x);
                    miny = miny.min(y);
                    maxx = maxx.max(x);
                    maxy = maxy.max(y);
                }
            }
        }
        eprintln!("model-drawn pixels (differ from magenta): {drew} / {}", w * h);
        if drew > 0 {
            eprintln!("bbox: x[{minx}..{maxx}] y[{miny}..{maxy}]  (image is {w}x{h})");
        }
        eprintln!("{grid}");
        drop(mapped);
        buffer.unmap();
    }

    #[test]
    fn model_matrix_scales_then_translates() {
        // Scale applies innermost, then translate: a unit +X point → origin + s.
        let m = compute_model_matrix([0.0, -0.84, 0.0], [0.0, 0.0, 0.0], [0.003, 0.003, 0.003]);
        // Column 0 carries the X scale.
        assert!((m[0] - 0.003).abs() < 1e-6);
        assert_eq!([m[12], m[13], m[14]], [0.0, -0.84, 0.0]);
    }
}
