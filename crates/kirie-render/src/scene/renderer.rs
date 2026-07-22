//! [`SceneRenderer`] — the per-frame scene compositor
//! (docs/render-architecture.md §2.5, §5.1, §7.1).
//!
//! Build phase (`new`): resolve the projection size and camera matrices; then,
//! in render order (§5.7), build one [`SceneItem`] per drawable object —
//! image layers plan their pass chain (§7.1), allocate ping-pong FBOs, upload
//! textures and prebuild static bind groups; particle and text objects wire in
//! through [`super::extras`] (§7.3-§7.4), 3D models through [`super::model`]
//! (§7.2). The only per-frame GPU writes are the packed `_WEGlobals` UBO, the
//! particle instance/VP buffers and the model MVP UBOs (SPEC.md §V5). Light /
//! shape / sound / group objects are not composited — they are transform groups
//! the reference does not draw (§5.6, §7.2).
//!
//! Frame phase (`render`): clear the scene FBO to the scene clear color
//! (§5.1); for each item in render order composite into the scene FBO — image
//! passes ping-pong through image FBOs then draw the final pass, particles
//! advance their sim and draw instanced sprites, text draws its placeholder
//! quad; then blit the scene FBO to the output surface through the
//! output-scaling UV window (§2.5, §4). Item order is the cross-kind z-order.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use kirie_audio::{AudioCapture, AudioSpectrum};
use kirie_platform::{RenderTarget, Renderer, SurfaceSize};
use kirie_scene::SceneModel;
use kirie_scene::material::Blending;
use kirie_scene::object::{ImageObject, Object, ObjectKind};
use kirie_scene::resolve::AssetSource;
use kirie_scene::scene::Projection;
use kirie_scene::value::DynamicValue;
use kirie_shader::reflect::{ParamDefault, Parameter};
use kirie_shader::{IncludeResolver, reflect::SamplerSlot};

use crate::particle::SpriteInstance;
use crate::scaling::{ClampMode, ScalingMode};

use super::extras::{self, ParticleGpu, TextGpu, TextPipeline};
use super::fbo::{FBO_FORMAT, Fbo};
use super::matrix::{self, Mat4};
use super::pipeline::{self, BindKind, BuiltPass, ModuleBinding};
use super::plan::{self, Geometry, PassOutput};
use super::scripting::{PropTarget, PropUpdate, ScriptHost, as_f32, as_rgb};
use super::text::TextFonts;
use super::texture::TextureRegistry;
use super::uniforms::{Builtins, GlobalsLayout, pack_globals};

/// Presentation options for a scene wallpaper (same surface as image/video).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SceneOptions {
    /// Output scaling mode (docs/render-architecture.md §4).
    pub scaling: ScalingMode,
    /// Out-of-window UV behavior (docs/render-architecture.md §4).
    pub clamp: ClampMode,
}

/// An include resolver backed by an [`AssetSource`]: `#include "x.h"` →
/// `shaders/x.h` in the scene container (docs/shader-pipeline.md §1.1).
struct SourceIncludes<'a>(&'a dyn AssetSource);

impl IncludeResolver for SourceIncludes<'_> {
    fn resolve(&self, include_name: &str) -> Option<String> {
        let bytes = self.0.load(&format!("shaders/{include_name}"))?;
        String::from_utf8(bytes).ok()
    }
}

/// One built pass: pipeline, static bind groups, per-frame UBOs, geometry.
struct PassGpu {
    pipeline: wgpu::RenderPipeline,
    g0_bind: wgpu::BindGroup,
    g1_bind: wgpu::BindGroup,
    vs_ubo: Option<wgpu::Buffer>,
    fs_ubo: Option<wgpu::Buffer>,
    vs_globals: GlobalsLayout,
    fs_globals: GlobalsLayout,
    vs_params: BTreeMap<String, Vec<f32>>,
    fs_params: BTreeMap<String, Vec<f32>>,
    vertex_buffer: wgpu::Buffer,
    /// Puppet-warp index buffer (`u16` triangle list). `Some` only for a puppet
    /// base pass, whose draw is `draw_indexed` over the mesh instead of the
    /// 4-vertex quad strip.
    puppet_indices: Option<wgpu::Buffer>,
    /// Index count for [`Self::puppet_indices`] (0 when absent).
    puppet_index_count: u32,
    output: PassOutput,
    geometry: Geometry,
    model_matrix: Mat4,
    blending: Blending,
    /// `g_TextureNResolution` per slot: `(texW, texH, realW, realH)` of the
    /// texture bound at slot N (docs §8.3). Shaders crop NPOT padding with
    /// `realSize / texSize = (z/x, w/y)` — masks live in oversized POT pages,
    /// so a flat projection-size resolution leaks their padding as a solid
    /// block (docs/format-tex.md §8.1; docs/render-architecture.md §7.1).
    tex_resolution: [[f32; 4]; 8],
    /// Re-resolution inputs for a live `setProperty` (docs §4.9): the material
    /// pass (its `constantshadervalues` keep their property bindings) plus the
    /// vs/fs shader-parameter reflection. On a property change the pass's
    /// constants are re-resolved and `{vs,fs}_params` recomputed, so the next
    /// frame's `_WEGlobals` pack (which reads `{vs,fs}_params`) shows the new
    /// value — no rebuild.
    ///
    /// `material_pass` stays per-pass (it is mutated in place by
    /// [`SceneRenderer::set_property`]), trimmed to the `constantshadervalues`
    /// entries some reflection parameter actually reads. The reflection itself
    /// is immutable after build, so it is `Arc`-shared across every pass built
    /// from the same shader (shaders are heavily reused across objects) instead
    /// of retaining one deep copy per pass.
    material_pass: kirie_scene::material::Pass,
    params_vs: Arc<Vec<Parameter>>,
    params_fs: Arc<Vec<Parameter>>,
}

/// One renderable image object with its FBOs and passes.
struct ObjectGpu {
    /// Scene-object id — the target key for script property updates (docs §8).
    id: i64,
    /// Parent object id (docs §7.1) — for walking the ancestor chain to gate
    /// this layer off when an ancestor group/image is hidden.
    parent: Option<i64>,
    passes: Vec<PassGpu>,
    fbos: [Option<Fbo>; 2],
    /// Per-effect scratch FBOs (§11.2 `fbos`), keyed by declared name and
    /// referenced by a pass's [`PassOutput::Named`] target.
    named_fbos: std::collections::HashMap<String, Fbo>,
    alpha: f32,
    brightness: f32,
    color: [f32; 4],
    /// Live visibility; a script may toggle it per frame (V6: false ⇒ no draw).
    visible: bool,
    /// True when this layer samples `_rt_FullFrameBuffer` (post-process layers).
    /// The scene FBO is copied into the snapshot before this object draws so the
    /// sample reads the composite-so-far without aliasing the write (docs §11).
    reads_scene: bool,
    /// Dependency donor (docs §5.6): draws its image-space composite into the
    /// ping-pong FIRST each frame — even when invisible — and never to the
    /// scene; dependents bind `_rt_imageLayerComposite_<id>_a/b` from it.
    offscreen_donor: bool,
    /// Per-layer parallax depth (`parallaxdepth`, docs §7.1): the layer's mvp
    /// shifts by `(depth + amount) · displacement · scene_w` (CImage.cpp:1173).
    parallax_depth: [f32; 2],
    // (video-backed textures live on SceneRenderer, not per object — one .tex
    // may be shared by several layers.)
    /// Ping-pong index holding the final composite after the full chain (the
    /// view dependents bind). `None` when the object has no composite FBOs.
    final_front: Option<usize>,
    /// The layer texture's animated atlas, when the base material's slot-0
    /// `.tex` is multi-frame. The first (layer-sampling) pass drives its
    /// `g_Texture0Translation`/`g_Texture0Rotation` builtins from this per
    /// frame — the reference's per-pass texture-animation state
    /// (`CPass.cpp:287-306`). Page streaming lives on [`SceneRenderer`].
    atlas: Option<Arc<super::texture::AtlasTexture>>,
}

/// One animated atlas plus the page currently uploaded into its bound texture
/// (see [`SceneRenderer::atlas_textures`]).
struct AtlasSlot {
    atlas: Arc<super::texture::AtlasTexture>,
    uploaded_page: usize,
}

/// A script-created runtime layer (`thisScene.createLayer`, docs §6.2) — the
/// audio-visualizer bar pattern: solid quads whose transform/color the script
/// drives every frame. Rendered as flat translucent rects on top of the scene
/// composite (model files are solid-pixel quads for this pattern; textured
/// runtime layers are a tracked extension).
struct RuntimeLayer {
    origin: [f32; 3],
    scale: [f32; 3],
    angles: [f32; 3],
    color: [f32; 3],
    alpha: f32,
    visible: bool,
}

impl Default for RuntimeLayer {
    fn default() -> Self {
        RuntimeLayer {
            origin: [0.0; 3],
            scale: [1.0; 3],
            angles: [0.0; 3],
            color: [1.0; 3],
            alpha: 1.0,
            visible: true,
        }
    }
}

/// One renderable scene object, in render order. Kinds composite into the same
/// scene FBO so cross-kind z-ordering is just this vector's order (docs §5.7,
/// §7.1-§7.4). Non-drawn kinds (light / shape / sound / group) never
/// produce an item — see [`super::extras`] for the rationale.
enum SceneItem {
    /// A 2D image layer with its effect-pass chain (docs §7.1).
    Image(Box<ObjectGpu>),
    /// A particle system: CPU sim + instanced sprites (docs §7.3).
    Particle(Box<ParticleGpu>),
    /// A real-glyph text quad + its coverage texture (docs §7.4).
    Text(Box<TextGpu>),
    /// A 3D model: `.mdl` sub-meshes drawn under the perspective camera with a
    /// private depth buffer (docs §7.2).
    Model(Box<super::model::ModelGpu>),
}

/// The scene wallpaper renderer.
pub struct SceneRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    proj_w: u32,
    proj_h: u32,
    clear_color: wgpu::Color,
    screen_mvp: Mat4,
    items: Vec<SceneItem>,
    /// Reused per-frame particle sprite scratch — cleared and refilled in place
    /// so steady-state stepping never reallocates (SPEC.md §V5).
    sprite_scratch: Vec<SpriteInstance>,
    /// Reused byte buffer for packing each pass's `_WEGlobals` block per frame
    /// (SPEC §V5: no per-frame allocation — capacity is retained across frames).
    pack_scratch: Vec<u8>,
    /// Live video-backed `.tex` textures (docs §10): each keeps its decoder
    /// playing; the newest ready frame streams into the bound GPU texture every
    /// render tick (the reference plays these — a frozen first frame was the
    /// 3445942378 divergence).
    video_textures: Vec<super::texture::VideoTexture>,
    /// Animated `.tex` atlases (docs/format-tex.md §8-§9): per render tick the
    /// current frame is selected by the reference's frametime walk
    /// (`CPass.cpp:348-378`) and, for multi-page (gif-style) textures, the
    /// frame's page streams into the one texture every pass bound — the wgpu
    /// equivalent of the reference's `textureID[frameNumber]` bind
    /// (`CPass.cpp:380-387`). `uploaded_page` tracks the page currently in the
    /// texture so unchanged frames upload nothing (SPEC.md §V5).
    atlas_textures: Vec<AtlasSlot>,
    /// Surface-normalized pointer, top-left origin (T26; platform-fed). Centered
    /// until the platform knows the cursor — the old hardcoded default.
    pointer: [f32; 2],
    /// Previous frame's pointer (`g_PointerPositionLast`).
    pointer_last: [f32; 2],
    /// Script-created runtime layers keyed by their synthetic (negative) id.
    runtime_layers: std::collections::HashMap<i64, RuntimeLayer>,
    /// Lazily built solid-quad pipeline + growable vertex buffer for them.
    runtime_pipeline: Option<(wgpu::RenderPipeline, wgpu::Buffer, usize)>,
    /// Eased camera-parallax displacement (`CScene::m_parallaxDisplacement`):
    /// `mix(disp, (pointer-0.5)·amount·influence, clamp(delay·dt, 0, 1))`.
    parallax_disp: [f32; 2],
    /// The shared text pipeline, built only when the scene has drawable text.
    text_pipeline: Option<TextPipeline>,
    scene_fbo: Fbo,
    /// Persistent snapshot of `scene_fbo` bound wherever a layer samples
    /// `_rt_FullFrameBuffer` (docs §6/§11). Refreshed by a GPU copy immediately
    /// before each post-process layer draws, so the read never aliases the
    /// write. Same size as `scene_fbo`. `None` when no object samples the scene
    /// FBO and bloom is off — a proj-size `RGBA16F` target (16–66 MB) not worth
    /// keeping for a scene that never reads it back.
    scene_snapshot: Option<Fbo>,
    /// Camera bloom post-process, present when `general.bloom` is enabled
    /// (docs §5). Runs on the composited scene FBO just before the blit.
    bloom: Option<super::bloom::Bloom>,
    // Final blit stage-2.
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind: wgpu::BindGroup,
    blit_window: wgpu::Buffer,
    options: SceneOptions,
    elapsed: f64,
    window_for: Option<SurfaceSize>,
    ambient: [f32; 3],
    skylight: [f32; 3],
    /// Whether the output surface is sRGB (drives the blit's encode-cancel).
    blit_srgb: bool,
    /// Shared system-audio capture; its latest spectrum feeds the
    /// `g_AudioSpectrum*` uniforms each frame (docs §8.3). `None` ⇒ silent.
    audio: Option<Arc<AudioCapture>>,
    /// Per-scene SceneScript host, if the scene has driveable property scripts
    /// (docs/scripting-api.md §3; SPEC.md §V3). Ticked once per frame.
    script: Option<ScriptHost>,
    /// The scene camera (eye/center/up/fov/near/far) — the 3D MODEL objects
    /// build their perspective from it each frame (docs §7.2). `fov` is a
    /// property-bound `UserSetting`, re-resolved on a live `setProperty`.
    camera: kirie_scene::scene::Camera,
    /// The property bag (declared user props + `--set-property` overrides),
    /// retained so a live `setProperty` (docs §4.9) can update a value and
    /// re-resolve the affected shader params / camera / general in place.
    bag: kirie_scene::PropertyBag,
    /// The resolved `general` block, re-resolved on `setProperty` to update
    /// bloom / ambient / skylight / clearcolor live.
    general: kirie_scene::scene::General,
    /// The 3D models' shared private depth buffer, allocated once at the scene
    /// size when the scene has any model object (SPEC.md §V5). `None` ⇒ no model.
    model_depth: Option<wgpu::TextureView>,
    /// `object id → parent id` over every object (docs §7.1). Used to gate a
    /// layer off when any ancestor group/image is hidden.
    parent_by_id: HashMap<i64, Option<i64>>,
    /// `object id → live visibility` over every object (incl. non-drawn groups),
    /// kept current by script visibility updates (docs §7.1 ancestor gating).
    visible_by_id: HashMap<i64, bool>,
}

impl SceneRenderer {
    /// Build the renderer from a resolved [`SceneModel`]. `source` supplies
    /// shader sources, `#include` headers and `.tex` bytes (the same asset
    /// source used to resolve the model). Returns an error only when the scene
    /// yields no drawable object or a degenerate projection (SPEC.md §V9).
    pub fn new(
        target: &RenderTarget<'_>,
        model: &SceneModel,
        source: &dyn AssetSource,
        options: SceneOptions,
        audio: Option<Arc<AudioCapture>>,
        user_props: &[(String, kirie_scene::PropertyValue)],
    ) -> Result<Self, super::SceneError> {
        let device = target.device;
        let queue = target.queue;
        let scene = &model.scene;

        // Retain the property bag (the resolved user-property snapshot) + the
        // general block so a live `setProperty` can re-resolve in place. The
        // snapshot carries every declared property's current value, which is what
        // a re-resolution reads.
        let mut bag = kirie_scene::PropertyBag::new();
        for (name, value) in user_props {
            bag.insert(name.clone(), value.clone());
        }
        let general = scene.general.clone();

        let (proj_w, proj_h) = projection_size(model);
        if proj_w == 0 || proj_h == 0 {
            return Err(super::SceneError::BadProjection {
                width: proj_w,
                height: proj_h,
            });
        }

        // Camera: screen MVP = translate(ortho, eye) * lookAt, conjugated by
        // the Y-mirror (see `screen_camera_mvp` — the X/Y camera tilt port).
        let cam = &scene.camera;
        let screen_mvp = screen_camera_mvp((proj_w, proj_h), cam.eye, cam.center, cam.up, cam.farz);

        let clear = scene.general.clearcolor.value;
        let clear_color = wgpu::Color {
            r: f64::from(clear[0]),
            g: f64::from(clear[1]),
            b: f64::from(clear[2]),
            a: 1.0,
        };

        let mut registry = TextureRegistry::new(device, queue);
        let fbo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("kirie-scene-fbo-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });

        // Parent/visibility maps over *every* object (drawn or not, incl. the
        // transform groups that never become a `SceneItem`). A layer is hidden
        // when any ancestor group/image is hidden (docs §7.1 `CImage::render`
        // "walk parents and skip if any ancestor is hidden") — but the ancestors
        // are often `Group`/solid objects that aren't items, so their visibility
        // must be tracked separately. `visible_by_id` seeds from each object's
        // resolved `visible` (property/TOG-conditional bindings already collapsed,
        // docs §3.3) and is kept live by script visibility updates.
        let parent_by_id: HashMap<i64, Option<i64>> =
            scene.objects.iter().map(|o| (o.base.id, o.base.parent)).collect();
        // Per-object local 2D transform, for parent-chain composition (docs
        // §7.3 transform inheritance): a child's origin/scale/angle are given in
        // its parent group's frame, so a group at the canvas centre spreads its
        // children across the scene. Without this, parented character layers
        // (the layer-switcher groups) all collapse onto one oversized sprite.
        let local_xf: HashMap<i64, LocalXf> = scene
            .objects
            .iter()
            .map(|o| {
                (
                    o.base.id,
                    LocalXf {
                        origin: [o.base.origin.value[0], o.base.origin.value[1]],
                        scale: [o.base.scale.value[0], o.base.scale.value[1]],
                        angle_z: o.base.angles.value[2],
                        parent: o.base.parent,
                    },
                )
            })
            .collect();
        let mut visible_by_id: HashMap<i64, bool> = scene
            .objects
            .iter()
            .map(|o| (o.base.id, o.base.visible.value))
            .collect();
        // Fold the image-level `visible` into the object entry so an ancestor
        // image hidden via its own image field also gates descendants.
        for o in &scene.objects {
            if let ObjectKind::Image(img) = &o.kind
                && !img.visible.value
            {
                visible_by_id.insert(o.base.id, false);
            }
        }

        // Render order across ALL object kinds (docs §5.7): declaration order,
        // stable-sorted by `sortorder` when `general.customsortorder` is set.
        // (Dependency hoisting, docs §5.6, is a documented gap.) Every kind
        // composites into the one scene FBO, so this order *is* the z-order.
        let mut order: Vec<usize> = (0..scene.objects.len()).collect();
        if scene.general.customsortorder {
            order.sort_by_key(|&i| scene.objects[i].base.sortorder);
        }

        // Scene FBO + its snapshot are built up front: a post-process layer
        // whose base texture is `_rt_FullFrameBuffer` (compose/project/fullscreen
        // util layers) samples the scene composited so far. We bind a persistent
        // snapshot texture (allocated once, refreshed by a GPU copy before each
        // such layer draws) so sampling never reads the target being written —
        // the reference's shadow-copy trick, docs §6/§11 (SPEC.md §V5: no
        // per-frame alloc). Both share the projection size.
        let scene_fbo = Fbo::new(device, "kirie-scene-fbo", proj_w, proj_h);
        let scene_snapshot = Fbo::new(device, "kirie-scene-snapshot", proj_w, proj_h);

        // Camera bloom (docs §5): when enabled, glow the composited scene with
        // the reproduced WE `camerabloom` effect just before the blit. Strength +
        // threshold come from the resolved `general.bloomstrength`/`bloomthreshold`
        // (matching WE `CScene.cpp:160`). The combine reuses `scene_snapshot`.
        let bloom = scene.general.bloom.value.then(|| {
            // `KIRIE_BLOOM_STRENGTH`/`KIRIE_BLOOM_THRESHOLD` override the scene
            // values (diagnostics / tuning against the reference).
            let env_f = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok());
            super::bloom::Bloom::new(
                device,
                queue,
                proj_w,
                proj_h,
                &scene_fbo.view,
                &scene_snapshot.view,
                env_f("KIRIE_BLOOM_STRENGTH").unwrap_or(scene.general.bloomstrength.value),
                env_f("KIRIE_BLOOM_THRESHOLD").unwrap_or(scene.general.bloomthreshold.value),
            )
        });

        let resolver = SourceIncludes(source);
        let mut items = Vec::new();
        let mut text_pipeline: Option<TextPipeline> = None;
        // The font stack is scanned once, the first time a drawable text object
        // is seen, and reused for the rest (system-font discovery is costly).
        let mut text_fonts: Option<TextFonts> = None;

        // Dependency hoisting (docs §5.6): objects listed in another object's
        // `dependencies` render their IMAGE-SPACE composite into their ping-pong
        // FBOs even when invisible, and dependents bind it cross-object as
        // `_rt_imageLayerComposite_<id>_a/b` (2155933185's hidden "planet N
        // texture" donors). Donors build FIRST so dependents' bind groups can
        // reference their composite views; they draw first each frame too.
        // A donor is an image object referenced by a DIFFERENT object's
        // `dependencies` — self-references are legal render-order no-ops
        // (object.rs: "may self-reference") and must NOT strip a visible
        // layer's scene draw (1388331347/1627026721 regressed exactly so).
        let donor_ids: std::collections::HashSet<i64> = scene
            .objects
            .iter()
            .flat_map(|o| {
                o.base
                    .dependencies
                    .iter()
                    .copied()
                    .filter(move |dep| *dep != o.base.id)
            })
            .filter(|id| {
                scene
                    .objects
                    .iter()
                    .any(|o| o.base.id == *id && matches!(o.kind, ObjectKind::Image(_)))
            })
            .collect();
        let mut donor_built: std::collections::HashMap<usize, ObjectGpu> =
            std::collections::HashMap::new();
        let mut cross: std::collections::HashMap<String, wgpu::TextureView> =
            std::collections::HashMap::new();
        // Build-scoped reflection interning (dropped with `new`; the shared
        // tables live on inside the passes that reference them).
        let mut param_cache = ParamCache::new();
        for &oi in &order {
            let object = &scene.objects[oi];
            if !donor_ids.contains(&object.base.id) {
                continue;
            }
            if let ObjectKind::Image(image) = &object.kind {
                let world = world_xf(object.base.id, &local_xf);
                if let Some(obj) = build_object(
                    device,
                    object,
                    image,
                    (proj_w, proj_h),
                    &screen_mvp,
                    source,
                    &resolver,
                    &mut registry,
                    &fbo_sampler,
                    &scene_snapshot,
                    world,
                    true,
                    &cross,
                    &mut param_cache,
                ) {
                    if let Some(front) = obj.final_front
                        && let Some(fbo) = obj.fbos[front].as_ref()
                    {
                        let id = object.base.id;
                        cross.insert(format!("_rt_imageLayerComposite_{id}_a"), fbo.view.clone());
                        cross.insert(format!("_rt_imageLayerComposite_{id}_b"), fbo.view.clone());
                    }
                    donor_built.insert(oi, obj);
                }
            }
        }

        for &oi in &order {
            let object = &scene.objects[oi];
            if let Some(obj) = donor_built.remove(&oi) {
                items.push(SceneItem::Image(Box::new(obj)));
                continue;
            }
            match &object.kind {
                ObjectKind::Image(image) => {
                    let world = world_xf(object.base.id, &local_xf);
                    if let Some(obj) = build_object(
                        device,
                        object,
                        image,
                        (proj_w, proj_h),
                        &screen_mvp,
                        source,
                        &resolver,
                        &mut registry,
                        &fbo_sampler,
                        &scene_snapshot,
                        world,
                        false,
                        &cross,
                        &mut param_cache,
                    ) {
                        items.push(SceneItem::Image(Box::new(obj)));
                    }
                }
                ObjectKind::Particle(pobj) => {
                    if let Some(pg) = extras::build_particle(
                        device,
                        queue,
                        object,
                        pobj,
                        (proj_w, proj_h),
                        &screen_mvp,
                        source,
                        &mut registry,
                    ) {
                        items.push(SceneItem::Particle(Box::new(pg)));
                    }
                }
                ObjectKind::Text(tobj) => {
                    let tp = text_pipeline.get_or_insert_with(|| extras::build_text_pipeline(device));
                    let fonts = text_fonts.get_or_insert_with(TextFonts::new);
                    if let Some(tg) = extras::build_text(
                        device,
                        queue,
                        tp,
                        fonts,
                        object,
                        tobj,
                        (proj_w, proj_h),
                        &screen_mvp,
                        source,
                    ) {
                        items.push(SceneItem::Text(Box::new(tg)));
                    }
                }
                ObjectKind::Model(mobj) => {
                    if let Some(mg) = super::model::build_model(
                        device,
                        object,
                        mobj,
                        (proj_w, proj_h),
                        source,
                        &resolver,
                        &mut registry,
                        &fbo_sampler,
                        &scene_snapshot,
                    ) {
                        items.push(SceneItem::Model(Box::new(mg)));
                    }
                }
                // Light / shape / sound / group are transform groups the
                // reference does not composite (docs §5.6, §7.2) — drawing a
                // stand-in would diverge from the C++ oracle.
                other => {
                    tracing::debug!(id = object.base.id, kind = ?std::mem::discriminant(other), "non-drawn object skipped (docs §5.6, §7.2)");
                }
            }
        }

        if items.is_empty() {
            return Err(super::SceneError::NoRenderableObjects);
        }

        // Keep the scene snapshot only if a post-process layer / reflection model
        // samples `_rt_FullFrameBuffer`, or bloom is enabled; otherwise it is dead
        // VRAM. The build above already bound it wherever needed, so dropping it
        // when nothing references it is safe (frees 16–66 MB per plain scene).
        let scene_snapshot = (bloom.is_some()
            || items.iter().any(|it| match it {
                SceneItem::Image(o) => o.reads_scene,
                SceneItem::Model(m) => m.reads_scene,
                _ => false,
            }))
        .then_some(scene_snapshot);

        let (blit_pipeline, blit_bind, blit_window) =
            build_blit(device, target.format, &scene_fbo, &fbo_sampler);

        // Build the SceneScript host from the resolved model (docs §3). `None`
        // when the scene has no driveable property script (the common case).
        let script = ScriptHost::build(model, (proj_w, proj_h), user_props);

        // Allocate the models' shared depth buffer once, only when the scene has
        // a model object (SPEC.md §V5: no per-frame alloc; §7.2).
        let model_depth = items
            .iter()
            .any(|it| matches!(it, SceneItem::Model(_)))
            .then(|| super::model::create_depth_texture(device, proj_w, proj_h));

        Ok(SceneRenderer {
            device: device.clone(),
            queue: queue.clone(),
            proj_w,
            proj_h,
            clear_color,
            screen_mvp,
            items,
            sprite_scratch: Vec::new(),
            pack_scratch: Vec::new(),
            video_textures: registry.take_videos(),
            atlas_textures: registry
                .take_atlases()
                .into_iter()
                .map(|atlas| AtlasSlot {
                    atlas,
                    // `load` uploaded page 0 (the first page) at build time.
                    uploaded_page: 0,
                })
                .collect(),
            pointer: [0.5, 0.5],
            pointer_last: [0.5, 0.5],
            parallax_disp: [0.0, 0.0],
            runtime_layers: std::collections::HashMap::new(),
            runtime_pipeline: None,
            text_pipeline,
            scene_fbo,
            scene_snapshot,
            bloom,
            bag,
            general,
            blit_pipeline,
            blit_bind,
            blit_window,
            options,
            elapsed: 0.0,
            window_for: None,
            ambient: [
                scene.general.ambientcolor.value[0],
                scene.general.ambientcolor.value[1],
                scene.general.ambientcolor.value[2],
            ],
            skylight: [
                scene.general.skylightcolor.value[0],
                scene.general.skylightcolor.value[1],
                scene.general.skylightcolor.value[2],
            ],
            blit_srgb: target.format.is_srgb(),
            audio,
            script,
            camera: scene.camera.clone(),
            model_depth,
            parent_by_id,
            visible_by_id,
        })
    }

    /// The scene's projection (native content) size — the output-scaling
    /// "projection" the blit window is computed against (docs §4).
    #[must_use]
    pub fn projection_size(&self) -> (u32, u32) {
        (self.proj_w, self.proj_h)
    }

    /// Total built draw passes across all image objects (diagnostics only).
    #[doc(hidden)]
    #[must_use]
    pub fn debug_pass_count(&self) -> usize {
        self.items
            .iter()
            .map(|it| match it {
                SceneItem::Image(o) => o.passes.len(),
                _ => 0,
            })
            .sum()
    }

    /// Number of wired-in particle objects (diagnostics only).
    #[doc(hidden)]
    #[must_use]
    pub fn debug_particle_count(&self) -> usize {
        self.items
            .iter()
            .filter(|it| matches!(it, SceneItem::Particle(_)))
            .count()
    }

    /// Number of text placeholder objects (diagnostics only).
    #[doc(hidden)]
    #[must_use]
    pub fn debug_text_count(&self) -> usize {
        self.items
            .iter()
            .filter(|it| matches!(it, SceneItem::Text(_)))
            .count()
    }

    /// Total live particles across every particle sim right now (diagnostics —
    /// proves systems spawned after warm-up frames).
    #[doc(hidden)]
    #[must_use]
    pub fn debug_live_particles(&self) -> usize {
        self.items
            .iter()
            .map(|it| match it {
                SceneItem::Particle(p) => p.sim.live_count(),
                _ => 0,
            })
            .sum()
    }
}

/// Scene-build interning table for shader-parameter reflection: shader name →
/// the distinct `(vs, fs)` parameter tables seen under that name. Shaders are
/// heavily reused across objects (every image base pass is `genericimage2`,
/// effect shaders repeat per instance), so interning collapses the per-pass
/// retained reflection ([`PassGpu::params_vs`]/`params_fs`) to one allocation
/// per distinct table. Keyed by name only as an index — reuse is decided by
/// comparing the tables themselves, so combo-divergent variants of one shader
/// never alias and correctness never rests on the key.
type ParamCache = HashMap<String, Vec<(Arc<Vec<Parameter>>, Arc<Vec<Parameter>>)>>;

/// Intern one pass's reflection tables through the [`ParamCache`]: return the
/// shared copy when an identical `(vs, fs)` pair was already seen for this
/// shader, else store and share this one.
fn intern_params(
    cache: &mut ParamCache,
    shader: &str,
    vs: Vec<Parameter>,
    fs: Vec<Parameter>,
) -> (Arc<Vec<Parameter>>, Arc<Vec<Parameter>>) {
    let variants = cache.entry(shader.to_owned()).or_default();
    if let Some((v, f)) = variants.iter().find(|(v, f)| **v == vs && **f == fs) {
        return (Arc::clone(v), Arc::clone(f));
    }
    let entry = (Arc::new(vs), Arc::new(fs));
    variants.push(entry.clone());
    entry
}

/// Build one image object's GPU resources, or `None` if it plans nothing / has
/// no buildable pass (SPEC.md §V9 skip-and-continue).
#[allow(clippy::too_many_arguments)]
fn build_object(
    device: &wgpu::Device,
    object: &Object,
    image: &ImageObject,
    scene_size: (u32, u32),
    screen_mvp: &Mat4,
    source: &dyn AssetSource,
    resolver: &dyn IncludeResolver,
    registry: &mut TextureRegistry,
    fbo_sampler: &wgpu::Sampler,
    scene_snapshot: &Fbo,
    world: WorldXf,
    offscreen_donor: bool,
    cross: &std::collections::HashMap<String, wgpu::TextureView>,
    param_cache: &mut ParamCache,
) -> Option<ObjectGpu> {
    // A dependency donor plans its full chain even when invisible — hoisting
    // renders its composite RT regardless of visibility (docs §5.6); only the
    // scene draw is suppressed (donors never emit a Scene pass).
    let visible = offscreen_donor || (image.visible.value && object.base.visible.value);
    // The `colorBlendMode` compatibility material, loaded at setup like the
    // reference's `MaterialParser::load(project, "materials/util/
    // effectpassthrough.json")` (`CImage.cpp:770-788`). A missing builtin
    // degrades to no extra pass (SPEC.md §V9 skip-and-continue).
    let color_blend = (image.color_blend_mode.value > 0)
        .then(|| source.load(plan::COLOR_BLEND_MATERIAL))
        .flatten()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .map(|v| kirie_scene::material::Material::from_value(&v));
    let chain = plan::plan_image(image, visible, offscreen_donor, color_blend.as_ref());
    if chain.passes.is_empty() {
        return None;
    }

    // Fullscreen post-process layers (compose/project/fullscreen util layers)
    // composite their final pass with the NDC quad + identity MVP, bypassing the
    // scene camera — otherwise a 3D/off-center camera skews a full-frame overlay
    // (docs §7.1 corner depth-clip fix, `CImage.cpp:847-853`).
    let fullscreen = image.model.as_ref().is_some_and(|m| m.fullscreen);

    // Puppet-warp mesh: an image whose model declares a `puppet` `.mdl` is a
    // deformable character (girl/guy/cat in scene 3428443753), not a flat quad.
    // Load + parse the mesh here; its first pass then draws the mesh instead of
    // the copy/scene quad (`CImage::loadPuppetMesh` + `setupPasses`
    // `m_hasPuppetMesh`, `CImage.cpp:426-528, 823-834`). A malformed/absent
    // puppet falls back to the flat quad (SPEC.md §V9). Fullscreen post-process
    // layers never carry a puppet, so they are excluded.
    let puppet = if fullscreen {
        None
    } else {
        image
            .model
            .as_ref()
            .and_then(|m| m.puppet.as_ref())
            .and_then(|path| {
                let bytes = source.load(path)?;
                match kirie_formats::model::PuppetMesh::parse(&bytes) {
                    Ok(mesh) if !mesh.indices.is_empty() => {
                        tracing::debug!(
                            id = object.base.id,
                            %path,
                            verts = mesh.vertices.len(),
                            indices = mesh.indices.len(),
                            "loaded puppet mesh"
                        );
                        Some(mesh)
                    }
                    Ok(_) => None,
                    Err(e) => {
                        tracing::debug!(id = object.base.id, %path, error = %e, "puppet mesh parse failed; flat quad");
                        None
                    }
                }
            })
    };

    // Image size: explicit size, else scene size (fullscreen/solid, docs §7.1).
    let (mut iw, mut ih) = (image.size[0] as u32, image.size[1] as u32);
    // World-space transform: local origin/scale/angle already composed through
    // the parent chain (docs §7.3), so a group's translation/scale reaches its
    // children.
    let (world_origin, world_scale, world_angle_z) = world;
    let mut origin = world_origin;
    if iw == 0 || ih == 0 {
        iw = scene_size.0;
        ih = scene_size.1;
        // Fullscreen fallback centers the layer (docs §7.1).
        origin = [scene_size.0 as f32 / 2.0, scene_size.1 as f32 / 2.0];
    }

    // The layer texture: the base material's texture slot 0 (docs §7.1). Falls
    // back to white if absent. When slot 0 is `_rt_FullFrameBuffer` (post-process
    // compose/project/fullscreen layers) the first pass samples the scene
    // composited so far — bound from the snapshot, not the white fallback.
    let layer_tex = base_layer_texture(image, source, registry);
    let layer_reads_scene = base_layer_name(image).as_deref().is_some_and(is_scene_rt);
    let mut reads_scene = layer_reads_scene;
    // The layer texture's animated atlas, if `load` registered one for its
    // name (multi-frame `.tex`). Drives the first pass's `g_Texture0*` frame
    // placement builtins per frame (`CPass.cpp:287-306`, docs §8.3).
    let layer_atlas = base_layer_name(image)
        .filter(|n| !n.starts_with("_rt_") && !n.starts_with("_alias_"))
        .and_then(|n| registry.atlas_for(&n));

    // The layer's own transform (docs §7.1: the `sceneSpacePosition` quad is
    // built at the *scaled* size, rotated about its center by the negated Z
    // angle, then translated to origin). Parent-chain resolution and parallax
    // are documented gaps; X/Y tilt is near-zero in the corpus.
    let scale = world_scale;
    let angle_z = world_angle_z;
    let scene_quad = scene_space_quad(origin, (iw, ih), [scale[0], scale[1]], angle_z, scene_size);
    let model_matrix = matrix::ortho(0.0, iw as f32, 0.0, ih as f32, 0.0, 1.0);

    // Phase A: translate + build every pass pipeline, dropping the ones whose
    // shaders are missing or untranslatable (SPEC.md §V9). Wiring is deferred to
    // Phase B so a dropped effect pass never leaves a broken FBO chain — the
    // surviving passes are re-wired into a valid ping-pong ending at the scene.
    // Each survivor carries its material pass plus its effect FBO routing
    // (`target`/`binds`) so Phase B can render effect scratch passes into their
    // own named FBOs and keep the composite intact for a combine pass (§11.2).
    struct Survivor {
        built: BuiltPass,
        raw: kirie_scene::material::Pass,
        /// The pass's shader-parameter reflection, taken out of [`BuiltPass`]
        /// and interned through the scene-build [`ParamCache`] so identical
        /// tables (same shader, same combos) are one shared allocation.
        params_vs: Arc<Vec<Parameter>>,
        params_fs: Arc<Vec<Parameter>>,
        target: Option<String>,
        binds: Vec<(u32, String)>,
        /// True for the base (first) material pass of a puppet object — the pass
        /// whose geometry the puppet mesh replaces (`CImage.cpp:823-834`, the mesh
        /// always lives on the first pass). Its pipeline is a triangle *list*.
        is_puppet_base: bool,
    }
    let mut built: Vec<Survivor> = Vec::new();
    for (ci, plan_pass) in chain.passes.iter().enumerate() {
        // `commands/copy` is the reference's VFS-registered blit for effect
        // `command:"copy"` passes (`WallpaperApplication.cpp:165-182`) — it
        // exists in no asset container, so bind the embedded sources instead.
        let (vs_src, fs_src) = if plan_pass.shader == plan::COPY_COMMAND_SHADER {
            (COPY_COMMAND_VERT.to_owned(), COPY_COMMAND_FRAG.to_owned())
        } else {
            let vs_name = format!("shaders/{}.vert", plan_pass.shader);
            let fs_name = format!("shaders/{}.frag", plan_pass.shader);
            let (Some(vs_bytes), Some(fs_bytes)) = (source.load(&vs_name), source.load(&fs_name)) else {
                tracing::debug!(shader = %plan_pass.shader, "missing shader source; pass skipped");
                continue;
            };
            let (Ok(vs_src), Ok(fs_src)) = (String::from_utf8(vs_bytes), String::from_utf8(fs_bytes)) else {
                continue;
            };
            (vs_src, fs_src)
        };
        // The base pass of a puppet draws the deformable mesh (indexed triangle
        // list); every other pass keeps the 4-vertex triangle strip.
        let is_puppet_base = puppet.is_some() && ci == 0;
        let topology = if is_puppet_base {
            wgpu::PrimitiveTopology::TriangleList
        } else {
            wgpu::PrimitiveTopology::TriangleStrip
        };
        // Force depthless: the 2D scene renderer allocates no depth attachment.
        match pipeline::build_pass(
            device,
            FBO_FORMAT,
            effective_blending(is_puppet_base, plan_pass.blending),
            plan_pass.cull,
            kirie_scene::material::DepthMode::Disabled,
            kirie_scene::material::DepthMode::Disabled,
            topology,
            &plan_pass.pass,
            &vs_src,
            &fs_src,
            resolver,
        ) {
            Ok(mut b) => {
                let (params_vs, params_fs) = intern_params(
                    param_cache,
                    &plan_pass.shader,
                    std::mem::take(&mut b.vs_params),
                    std::mem::take(&mut b.fs_params),
                );
                // Retain only the constants some reflection parameter reads:
                // the retained copy exists solely for `set_property`'s
                // `resolve_constants` + `resolve_params` re-resolution, and
                // `resolve_params` looks up `constantshadervalues` by each
                // parameter's `material` key — entries no parameter names are
                // never read again (the full authored pass already resolved
                // the initial `{vs,fs}_params` below and stays in the model).
                let mut raw = plan_pass.pass.clone();
                raw.constantshadervalues
                    .retain(|k, _| params_vs.iter().chain(params_fs.iter()).any(|p| p.material == *k));
                built.push(Survivor {
                    built: b,
                    raw,
                    params_vs,
                    params_fs,
                    target: plan_pass.target.clone(),
                    binds: plan_pass.binds.clone(),
                    is_puppet_base,
                });
            }
            Err(e) => {
                tracing::debug!(shader = %plan_pass.shader, error = %e, "pass shader failed to build; skipped");
            }
        }
    }
    if built.is_empty() {
        return None;
    }

    // Ping-pong composite FBOs only when more than one pass survived (docs §7.1;
    // V5: allocated once here, reused every frame). These are the image
    // composite `_rt_imageLayerComposite_<id>_a/_b`; effect scratch passes render
    // elsewhere so the composite keeps the pre-effect image (§11.2).
    let n = built.len();
    // A dependency donor keeps its FULL chain in the ping-pong (its last pass
    // composites instead of drawing to scene), so it always needs the pair.
    let fbos = if n > 1 || offscreen_donor {
        [
            Some(Fbo::new(device, "kirie-image-fbo-a", iw, ih)),
            Some(Fbo::new(device, "kirie-image-fbo-b", iw, ih)),
        ]
    } else {
        [None, None]
    };

    // Per-effect scratch FBOs (§11.2 `fbos`): a pass with a named `target`
    // renders here instead of the composite, and later `bind`s sample them by
    // name. `scale` is a divisor vs the image size (2 = half-res godrays blur).
    let mut named_fbos: std::collections::HashMap<String, Fbo> = std::collections::HashMap::new();
    for decl in &chain.named_fbos {
        let s = if decl.scale > 0.0 { decl.scale } else { 1.0 };
        let w = ((iw as f32 / s).round() as u32).max(1);
        let h = ((ih as f32 / s).round() as u32).max(1);
        named_fbos.insert(decl.name.clone(), Fbo::new(device, "kirie-effect-fbo", w, h));
    }
    // The two composite reference names for this object's id (both resolve to the
    // current composite front — the reference's `_a`/`_b` ping-pong pair).
    let comp_a = format!("_rt_imageLayerComposite_{}_a", object.base.id);
    let comp_b = format!("_rt_imageLayerComposite_{}_b", object.base.id);

    // A pass "composites" (advances/ends the ping-pong) when it has no named
    // target FBO; a target that names a declared scratch FBO renders aside.
    let is_composite = |target: &Option<String>| match target {
        None => true,
        Some(t) => !named_fbos.contains_key(t),
    };
    let last_comp = built
        .iter()
        .rposition(|s| is_composite(&s.target))
        .unwrap_or(n - 1);

    // Phase B: render the base/composite passes through the ping-pong (last →
    // scene), effect scratch passes into their named FBOs, threading the
    // composite front so `previous`/`_rt_imageLayerComposite` binds sample the
    // real prior output instead of the 1×1 white default (§11.2).
    let mut passes = Vec::with_capacity(n);
    // `comp_front`: the composite buffer holding the latest composite (None until
    // the first composite pass writes; before that the layer texture is "it").
    let mut comp_front: Option<usize> = None;
    for (i, sv) in built.into_iter().enumerate() {
        let Survivor {
            built: built_pass,
            raw: raw_pass,
            params_vs,
            params_fs,
            target,
            binds,
            is_puppet_base,
        } = sv;
        // A donor never draws to the scene: its final composite stays in the
        // ping-pong (image space) for dependents to sample (docs §5.6).
        let is_scene = i == last_comp && !offscreen_donor;
        let composite = is_composite(&target);

        // Effective texture slots: overlay the effect `bind`s onto the material's
        // slots without clobbering an authored (scene-json) override (§11.2 — the
        // bind fills only empty slots; the position override wins where present).
        let mut raw_pass = raw_pass;
        for (slot, name) in &binds {
            let idx = *slot as usize;
            if idx >= raw_pass.textures.len() {
                raw_pass.textures.resize(idx + 1, None);
            }
            if raw_pass.textures[idx].is_none() {
                raw_pass.textures[idx] = Some(name.clone());
            }
        }

        let geometry = if is_puppet_base {
            // The puppet mesh replaces this pass's quad. Single-pass puppets draw
            // the mesh straight into the scene FBO (scene space, screen MVP);
            // multi-pass puppets draw it into the copy FBO (local space, ortho
            // MVP) so the effect chain then processes the deformed character
            // (`CImage.cpp:823-834`).
            if is_scene {
                Geometry::Puppet
            } else {
                Geometry::PuppetCopy
            }
        } else if is_scene {
            // Fullscreen post-process layers composite full-frame in NDC with an
            // identity MVP (bypass the camera); ordinary layers use the scene
            // quad + screen MVP (docs §7.1).
            if fullscreen {
                Geometry::Pass
            } else {
                Geometry::Scene
            }
        } else if i == 0 {
            Geometry::Copy
        } else {
            Geometry::Pass
        };

        // The first pass samples the base layer texture and must crop NPOT
        // padding exactly like the reference's `texcoordCopy = realSize /
        // textureSize` (docs §7.1). Later passes read real-sized FBOs (0..1). The
        // scene snapshot is already real-sized, so it is never UV-cropped.
        let reads_layer = i == 0;
        let uv_crop = if reads_layer && !layer_reads_scene {
            layer_tex.uv_crop
        } else {
            [1.0, 1.0]
        };
        // Build the geometry: a puppet base uploads the deformable mesh (indexed
        // triangle list); everything else uploads the 4-vertex quad strip.
        let (vertex_buffer, puppet_indices) = match geometry {
            Geometry::Puppet | Geometry::PuppetCopy => {
                let verts = puppet_copy_vertices(
                    puppet.as_ref().expect("puppet base has a mesh"),
                    (iw, ih),
                    uv_crop,
                );
                (
                    create_buffer_init(
                        device,
                        "kirie-puppet-vb",
                        bytemuck::cast_slice(&verts),
                        wgpu::BufferUsages::VERTEX,
                    ),
                    Some(create_puppet_index_buffer(device, puppet.as_ref().unwrap())),
                )
            }
            _ => {
                let mut verts = match geometry {
                    Geometry::Scene => scene_quad,
                    _ => ndc_quad(1.0, 1.0),
                };
                if uv_crop != [1.0, 1.0] {
                    apply_uv_crop(&mut verts, uv_crop);
                }
                (
                    create_vertex_buffer(device, &verts, built_pass.uv_location.is_some()),
                    None,
                )
            }
        };
        let puppet_index_count = puppet
            .as_ref()
            .filter(|_| puppet_indices.is_some())
            .map_or(0, |m| m.indices.len() as u32);

        let vs_params = resolve_params(&params_vs, &raw_pass);
        let fs_params = resolve_params(&params_fs, &raw_pass);

        // The pass input (slot-0 default and `previous`): the scene snapshot when
        // the first pass samples `_rt_FullFrameBuffer`, else the layer texture
        // (first pass) or the current composite front (later passes).
        let (input_view, input_sampler): (&wgpu::TextureView, &wgpu::Sampler) = if reads_layer {
            if layer_reads_scene {
                (&scene_snapshot.view, fbo_sampler)
            } else {
                (&layer_tex.view, &layer_tex.sampler)
            }
        } else {
            match comp_front {
                Some(k) => (fbos[k].as_ref().map_or(&layer_tex.view, |f| &f.view), fbo_sampler),
                None => (&layer_tex.view, &layer_tex.sampler),
            }
        };

        // Name→FBO resolution for this pass's `_rt_` binds: `previous` and the
        // composite references resolve to the composite front; declared scratch
        // FBOs resolve to their target (§11.2).
        let comp_view: &wgpu::TextureView = match comp_front {
            Some(k) => fbos[k].as_ref().map_or(input_view, |f| &f.view),
            None => input_view,
        };
        let mut named: std::collections::HashMap<&str, (&wgpu::TextureView, &wgpu::Sampler)> =
            std::collections::HashMap::new();
        // Cross-object donor composites first (docs §5.6) — own names below win
        // on any collision (a donor id can never equal this object's id).
        for (name, view) in cross {
            named.insert(name.as_str(), (view, fbo_sampler));
        }
        named.insert("previous", (comp_view, fbo_sampler));
        named.insert(comp_a.as_str(), (comp_view, fbo_sampler));
        named.insert(comp_b.as_str(), (comp_view, fbo_sampler));
        for (name, fbo) in &named_fbos {
            named.insert(name.as_str(), (&fbo.view, fbo_sampler));
        }

        // Any `_rt_FullFrameBuffer` slot needs a per-frame scene snapshot copy.
        if raw_pass.textures.iter().flatten().any(|n| is_scene_rt(n)) {
            reads_scene = true;
        }

        let vs_ubo =
            (!built_pass.vs_globals.is_empty()).then(|| create_ubo(device, built_pass.vs_globals.size));
        let fs_ubo =
            (!built_pass.fs_globals.is_empty()).then(|| create_ubo(device, built_pass.fs_globals.size));

        let g0_bind = build_bind_group(
            device,
            &built_pass.g0_layout,
            vs_ubo.as_ref(),
            &built_pass.g0_bindings,
            &built_pass.vs_samplers,
            input_view,
            input_sampler,
            registry,
            source,
            &raw_pass,
            (&scene_snapshot.view, fbo_sampler),
            &named,
        );
        let g1_bind = build_bind_group(
            device,
            &built_pass.g1_layout,
            fs_ubo.as_ref(),
            &built_pass.g1_bindings,
            &built_pass.fs_samplers,
            input_view,
            input_sampler,
            registry,
            source,
            &raw_pass,
            (&scene_snapshot.view, fbo_sampler),
            &named,
        );

        // `g_TextureNResolution` per slot. Slot 0 is the pass input; slots 1.. are
        // the pass's named textures/binds. FBO/composite refs are clean render
        // targets (realSize == texSize ⇒ img_res); real `.tex` assets carry NPOT
        // padding, so shaders read `realSize/texSize` to crop it (docs §7.1).
        let img_res = [iw as f32, ih as f32, iw as f32, ih as f32];
        let mut tex_resolution = [img_res; 8];
        tex_resolution[0] = if reads_layer && !layer_reads_scene {
            tex_res(&layer_tex)
        } else {
            img_res
        };
        for (si, slot) in raw_pass.textures.iter().enumerate().take(8).skip(1) {
            let Some(name) = slot else { continue };
            tex_resolution[si] = if name.starts_with("_rt_") || name.starts_with("_alias_") {
                img_res
            } else {
                tex_res(&registry.get(name, source))
            };
        }

        // Output routing: a named-target scratch pass renders into its FBO (the
        // composite front is untouched); a composite pass writes the other
        // ping-pong buffer (or the scene FBO if it is the final composite).
        let output = if is_scene {
            PassOutput::Scene
        } else if composite {
            let dst = match comp_front {
                Some(k) => 1 - k,
                None => 0,
            };
            comp_front = Some(dst);
            PassOutput::Fbo(dst)
        } else {
            // Safe: `composite == false` ⇒ the target names a declared FBO.
            PassOutput::Named(target.clone().unwrap_or_default())
        };

        let BuiltPass {
            pipeline,
            vs_globals,
            fs_globals,
            ..
        } = built_pass;

        passes.push(PassGpu {
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
            puppet_indices,
            puppet_index_count,
            output,
            geometry,
            model_matrix,
            blending: effective_blending(is_puppet_base, raw_pass.blending),
            tex_resolution,
            params_vs,
            params_fs,
            material_pass: raw_pass,
        });
    }
    let _ = screen_mvp; // screen MVP applied per-frame via builtins.
    tracing::trace!(target: "kirie_render::ptrdbg",
        id = object.base.id,
        n_passes = passes.len(),
        geoms = ?passes.iter().map(|p| format!("{:?}", p.geometry)).collect::<Vec<_>>(),
        "object built");
    Some(ObjectGpu {
        id: object.base.id,
        parent: object.base.parent,
        passes,
        fbos,
        named_fbos,
        alpha: image.alpha.value,
        brightness: image.brightness.value,
        color: image.color.value,
        visible: true,
        reads_scene,
        offscreen_donor,
        final_front: comp_front,
        parallax_depth: image.parallax_depth.value,
        atlas: layer_atlas,
    })
}

/// Whether every ancestor of a layer (walking `parent` up the chain) is visible
/// (docs §7.1 `CImage::render`: a layer is skipped if any ancestor group/image
/// is hidden). An unknown parent id (dangling reference) is treated as visible.
/// A cycle guard caps the walk (SPEC.md §V9 — malformed graphs never hang).
fn ancestors_visible(
    parent_by_id: &HashMap<i64, Option<i64>>,
    visible_by_id: &HashMap<i64, bool>,
    start: Option<i64>,
) -> bool {
    let mut cur = start;
    for _ in 0..64 {
        let Some(id) = cur else { return true };
        if !visible_by_id.get(&id).copied().unwrap_or(true) {
            return false;
        }
        cur = parent_by_id.get(&id).copied().flatten();
    }
    true
}

impl Renderer for SceneRenderer {
    fn render(&mut self, view: &wgpu::TextureView, size: SurfaceSize, dt: f32) {
        self.elapsed += f64::from(dt);
        let time = self.elapsed as f32;
        let texel = [1.0 / self.proj_w as f32, 1.0 / self.proj_h as f32];

        // Audio: lock-free snapshot of the latest published spectrum (V4 — never
        // blocks the render thread). `None`/silent capture ⇒ all-zero bands, the
        // exact state an AUDIOPROCESSING shader sees with audio off (docs §8.3).
        let spectrum = self.audio.as_ref().map(|a| a.latest_spectrum());

        // SceneScript: tick once per frame and apply the typed property updates
        // to the live objects *before* drawing (SPEC.md §V3 — typed ops only,
        // JS never touches these buffers). Frame-callback driven, so an occluded
        // output that gets no callbacks never ticks (V6 groundwork).
        if let Some(script) = &mut self.script {
            let updates = script.tick(dt, spectrum.as_deref(), self.pointer);
            for (id, path) in script.take_created() {
                tracing::debug!(id, %path, "runtime layer created by script");
                self.runtime_layers.entry(id).or_default();
            }
            if !updates.is_empty() {
                // Keep the ancestor-visibility map live so a script hiding/showing
                // a group also gates/ungates its descendants (docs §7.1).
                for u in &updates {
                    if matches!(u.target, PropTarget::Visible)
                        && let kirie_script::ScriptValue::Bool(v) = &u.value
                    {
                        self.visible_by_id.insert(u.object_id, *v);
                    }
                }
                apply_runtime_updates(&mut self.runtime_layers, &updates);
                apply_script_updates(&mut self.items, &updates);
            }
        }

        // Recompute the blit UV window on resize (docs §4; cached like the
        // reference WallpaperState).
        if self.window_for != Some(size) {
            let window = self
                .options
                .scaling
                .uv_window((self.proj_w, self.proj_h), (size.width, size.height));
            let clamp_mode = match self.options.clamp {
                ClampMode::Clamp => 0u32,
                ClampMode::Border => 1u32,
                ClampMode::Repeat => 2u32,
            };
            self.queue.write_buffer(
                &self.blit_window,
                0,
                bytemuck::bytes_of(&BlitWindow {
                    rect: [window.u0, window.v0, window.u1, window.v1],
                    clamp_mode,
                    srgb: u32::from(self.blit_srgb),
                    _pad: [0; 2],
                }),
            );
            self.window_for = Some(size);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kirie-scene-encoder"),
            });

        // Stage 1a: clear the scene FBO to the scene clear color (docs §5.1).
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kirie-scene-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_fbo.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        // Stage 1b: every object in render order, dispatched by kind. All kinds
        // composite into the same scene FBO with `LoadOp::Load`, so the item
        // order set at build time *is* the cross-kind z-order (docs §5.7, §7).
        let scene_view = &self.scene_fbo.view;
        let scene_tex = &self.scene_fbo.texture;
        // Present only when a layer reads the scene FBO / bloom is on; a
        // `reads_scene` object always implies `Some` here (both set at build).
        let snap_tex = self.scene_snapshot.as_ref().map(|s| &s.texture);
        let copy_extent = wgpu::Extent3d {
            width: self.proj_w,
            height: self.proj_h,
            depth_or_array_layers: 1,
        };
        let audio = spectrum.as_deref();
        // Ancestor-visibility maps (disjoint from the mut items borrow below):
        // a layer is skipped when any ancestor group/image is hidden (docs §7.1).
        let parent_by_id = &self.parent_by_id;
        let visible_by_id = &self.visible_by_id;
        // Reused UBO-pack buffer (disjoint field from `items` below), so no pass
        // allocates its `_WEGlobals` bytes each frame (SPEC §V5).
        let pack_scratch = &mut self.pack_scratch;
        // Pointer + parallax snapshots for the draw loops (disjoint fields).
        let pointer = self.pointer;
        let pointer_last = self.pointer_last;
        let parallax = (
            self.parallax_disp,
            self.general.cameraparallaxamount.value,
            self.proj_w as f32,
        );

        // Pointer rollover (`g_PointerPositionLast`) + camera parallax easing
        // (CScene::renderFrame): `disp = mix(disp, (mouse-0.5)·amount·influence,
        // clamp(delay·dt, 0, 1))` — `delay` is a rate despite the name.
        self.pointer_last = self.pointer;
        if self.general.cameraparallax.value {
            let amount = self.general.cameraparallaxamount.value;
            let influence = self.general.cameraparallaxmouseinfluence.value;
            let t = (self.general.cameraparallaxdelay.value * dt).clamp(0.0, 1.0);
            for axis in 0..2 {
                let target = (self.pointer[axis] - 0.5) * amount * influence;
                self.parallax_disp[axis] += (target - self.parallax_disp[axis]) * t;
            }
        }

        // Stream video-backed `.tex` frames (docs §10): drain what the decoder
        // has ready and upload only the NEWEST (no backlog), into the same
        // texture object every pass bound. The decoder paces itself to the
        // video clock and loops seamlessly; nothing ready ⇒ keep last frame.
        for vt in &self.video_textures {
            let mut newest = None;
            while let Some(f) = vt.player.recv_frame_timeout(std::time::Duration::ZERO) {
                newest = Some(f);
            }
            if let Some(f) = newest
                && (f.width, f.height) == vt.size
            {
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vt.gpu.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &f.data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * f.width),
                        rows_per_image: Some(f.height),
                    },
                    wgpu::Extent3d {
                        width: f.width,
                        height: f.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        // Advance animated `.tex` atlases (docs/format-tex.md §8-§9): select
        // the frame with the reference's `fmod(renderTime, Σ frametime)` walk
        // (`CPass.cpp:348-378`) and, when the frame lives on another page
        // (gif-style multi-image .tex), stream that page into the one texture
        // every pass bound — the wgpu stand-in for the reference's
        // `glBindTexture(textureID[frameNumber])` (`CPass.cpp:380-387`).
        // Single-page spritesheets upload nothing here; their placement rides
        // the `g_Texture0Translation`/`g_Texture0Rotation` builtins instead.
        for slot in &mut self.atlas_textures {
            let frame = slot.atlas.placement_at(self.elapsed);
            if frame.page == slot.uploaded_page {
                continue;
            }
            let Some(page) = slot.atlas.pages.get(frame.page) else {
                continue; // single-page atlas (no CPU pages retained)
            };
            let gpu = &slot.atlas.gpu;
            // Registration guarantees uniform page dims; guard anyway (V9).
            if (page.width, page.height) != (gpu.width, gpu.height) {
                continue;
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &gpu.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &page.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * page.width),
                    rows_per_image: Some(page.height),
                },
                wgpu::Extent3d {
                    width: page.width,
                    height: page.height,
                    depth_or_array_layers: 1,
                },
            );
            slot.uploaded_page = frame.page;
        }

        // Sweep 0 — dependency donors (docs §5.6): render their image-space
        // composites FIRST, unconditionally (they are invisible by design), so
        // dependents sample this frame's content. Their passes only write their
        // own ping-pong/scratch FBOs, never the scene.
        for item in &mut self.items {
            if let SceneItem::Image(object) = item
                && object.offscreen_donor
            {
                draw_image_object(
                    &mut encoder,
                    &self.queue,
                    object,
                    scene_view,
                    self.screen_mvp,
                    self.ambient,
                    self.skylight,
                    time,
                    self.elapsed,
                    texel,
                    audio,
                    pack_scratch,
                    pointer,
                    pointer_last,
                    parallax,
                );
            }
        }

        for item in &mut self.items {
            match item {
                // Donors drew in sweep 0; a script may have hidden this object
                // this frame (V6: skip its whole pass chain — zero GPU work).
                // Also skip when any ancestor group/image is hidden (§7.1).
                SceneItem::Image(object)
                    if object.offscreen_donor
                        || !object.visible
                        || !ancestors_visible(parent_by_id, visible_by_id, object.parent) => {}
                SceneItem::Image(object) => {
                    // Post-process layers sample `_rt_FullFrameBuffer`: snapshot
                    // the composite-so-far so the read never aliases the write
                    // (docs §11 shadow-copy; the copy sees every earlier object
                    // because encoder passes run in order).
                    if object.reads_scene && let Some(snap_tex) = snap_tex {
                        encoder.copy_texture_to_texture(
                            scene_tex.as_image_copy(),
                            snap_tex.as_image_copy(),
                            copy_extent,
                        );
                    }
                    draw_image_object(
                        &mut encoder,
                        &self.queue,
                        object,
                        scene_view,
                        self.screen_mvp,
                        self.ambient,
                        self.skylight,
                        time,
                        self.elapsed,
                        texel,
                        audio,
                        pack_scratch,
                        pointer,
                        pointer_last,
                        parallax,
                    );
                }
                SceneItem::Particle(pg) => {
                    // Advance the CPU sim, refill the shared scratch (no realloc
                    // once warm — SPEC.md §V5), upload + draw instanced sprites.
                    pg.sim.update(dt);
                    pg.sim.write_sprites(&mut self.sprite_scratch);
                    let n = pg
                        .renderer
                        .upload(&self.queue, &pg.view_projection, &self.sprite_scratch);
                    pg.renderer.draw(&mut encoder, scene_view, n);
                }
                SceneItem::Text(tg) => {
                    if let Some(tp) = &self.text_pipeline {
                        extras::draw_text(&mut encoder, tp, tg, scene_view);
                    }
                }
                // A script may hide the model this frame (V6: skip entirely).
                SceneItem::Model(mg) if !mg.visible => {}
                SceneItem::Model(mg) => {
                    // REFLECTION meshes sample `_rt_FullFrameBuffer`: snapshot the
                    // composite-so-far first so the read never aliases the write
                    // (docs §6/§11 shadow-copy; `CModel::render` blits the scene).
                    if mg.reads_scene && let Some(snap_tex) = snap_tex {
                        encoder.copy_texture_to_texture(
                            scene_tex.as_image_copy(),
                            snap_tex.as_image_copy(),
                            copy_extent,
                        );
                    }
                    // The model needs a depth buffer; it is always allocated when
                    // the scene has a model item (built above). Skip defensively.
                    if let Some(depth_view) = self.model_depth.as_ref() {
                        let aspect = if self.proj_h > 0 {
                            self.proj_w as f32 / self.proj_h as f32
                        } else {
                            16.0 / 9.0
                        };
                        super::model::draw_model(
                            &mut encoder,
                            &self.queue,
                            mg,
                            scene_view,
                            depth_view,
                            &self.camera,
                            aspect,
                            self.ambient,
                            self.skylight,
                            time,
                            texel,
                            audio,
                            pack_scratch,
                            pointer,
                            pointer_last,
                        );
                    }
                }
            }
        }

        // Stage 1b2: script-created runtime layers (visualizer bars) — solid
        // translucent quads on top of the composite, transforms in the same
        // JSON→scene→NDC space as scene_quad_verts.
        if self.runtime_layers.values().any(|l| l.visible && l.alpha > 0.0) {
            let (sw, sh) = (self.proj_w as f32, self.proj_h as f32);
            let mut verts: Vec<f32> = Vec::with_capacity(self.runtime_layers.len() * 36);
            for l in self.runtime_layers.values() {
                if !l.visible || l.alpha <= 0.0 {
                    continue;
                }
                let cx = l.origin[0] - sw / 2.0;
                let cy = l.origin[1] - sh / 2.0;
                let (hw, hh) = (l.scale[0] / 2.0, l.scale[1] / 2.0);
                let (sn, cs) = (-l.angles[2].to_radians()).sin_cos();
                let corner = |dx: f32, dy: f32| {
                    [
                        (cx + dx * cs - dy * sn) / (sw / 2.0),
                        (cy + dx * sn + dy * cs) / (sh / 2.0),
                    ]
                };
                let tl = corner(-hw, hh);
                let bl = corner(-hw, -hh);
                let tr = corner(hw, hh);
                let br = corner(hw, -hh);
                let (r, g, b, a) = (l.color[0], l.color[1], l.color[2], l.alpha);
                for v in [tl, bl, tr, tr, bl, br] {
                    verts.extend_from_slice(&[v[0], v[1], r, g, b, a]);
                }
            }
            if !verts.is_empty() {
                let needed = verts.len() * 4;
                let rebuild = match &self.runtime_pipeline {
                    Some((_, _, cap)) => *cap < needed,
                    None => true,
                };
                if rebuild {
                    let pipeline = self
                        .runtime_pipeline
                        .take()
                        .map(|(p, _, _)| p)
                        .unwrap_or_else(|| build_runtime_pipeline(&self.device));
                    let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("kirie-runtime-layer-verts"),
                        size: (needed.max(4096)) as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let cap = needed.max(4096);
                    self.runtime_pipeline = Some((pipeline, buf, cap));
                }
                if let Some((pipeline, buf, _)) = &self.runtime_pipeline {
                    self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&verts));
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("kirie-runtime-layers"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: scene_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rp.set_pipeline(pipeline);
                    rp.set_vertex_buffer(0, buf.slice(..));
                    rp.draw(0..(verts.len() / 6) as u32, 0..1);
                }
            }
        }

        // Stage 1c: camera bloom — glow the composited scene in place (docs §5),
        // so the blit below picks up the bloomed result unchanged. Bloom always
        // keeps the snapshot alive (the gate at build includes `bloom.is_some()`).
        if let (Some(bloom), Some(snap)) = (&self.bloom, &self.scene_snapshot) {
            bloom.run(&mut encoder, &self.scene_fbo, snap);
        }

        // Stage 2: blit the scene FBO to the surface (docs §2.5).
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kirie-scene-blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            rp.set_pipeline(&self.blit_pipeline);
            rp.set_bind_group(0, &self.blit_bind, &[]);
            rp.draw(0..4, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Live `setProperty` (doc §4.9): parse `value` to the property's declared
    /// type, update the bag, then re-resolve the affected shader parameters,
    /// camera fov and general (bloom/ambient/skylight/clearcolor) in place — all
    /// read per-frame, so the next frame shows the change with no rebuild.
    ///
    /// Object visibility/transform bindings are not yet re-resolved live (they
    /// still apply on the next scene load); the render-uniform properties users
    /// tweak (colors, shader constants, fov, bloom) are covered.
    fn set_property(&mut self, key: &str, value: &str) {
        if !self.bag.set_from_str(key, value) {
            return; // unknown key or unparseable — nothing changed
        }
        // Material constants → shader params (the bulk: colors, hue, coloring…).
        for item in &mut self.items {
            if let SceneItem::Image(o) = item {
                for pass in &mut o.passes {
                    kirie_scene::resolve::resolve_constants(
                        &mut pass.material_pass.constantshadervalues,
                        &self.bag,
                    );
                    pass.vs_params = resolve_params(&pass.params_vs, &pass.material_pass);
                    pass.fs_params = resolve_params(&pass.params_fs, &pass.material_pass);
                }
            }
        }
        // Camera fov (model perspective, rebuilt each frame from camera.fov).
        self.camera.reresolve(&self.bag);
        // General: ambient / skylight / clearcolor + bloom, all read per-frame.
        self.general.resolve(&self.bag);
        self.ambient = [
            self.general.ambientcolor.value[0],
            self.general.ambientcolor.value[1],
            self.general.ambientcolor.value[2],
        ];
        self.skylight = [
            self.general.skylightcolor.value[0],
            self.general.skylightcolor.value[1],
            self.general.skylightcolor.value[2],
        ];
        let clear = self.general.clearcolor.value;
        self.clear_color = wgpu::Color {
            r: f64::from(clear[0]),
            g: f64::from(clear[1]),
            b: f64::from(clear[2]),
            a: 1.0,
        };
        if let Some(bloom) = &self.bloom {
            bloom.set_params(
                &self.queue,
                self.general.bloomstrength.value,
                self.general.bloomthreshold.value,
            );
        }
        // Script-driven properties (a SceneScript's `applyUserProperties` — e.g. a
        // `coloring` combo that recolors the scene, or a layer-switcher combo):
        // fire the change handler and apply its typed updates live (docs §5.3).
        let value = self.bag.get(key).cloned();
        let updates = match (self.script.as_mut(), value.as_ref()) {
            (Some(script), Some(v)) => script.apply_user_property(key, v),
            _ => Vec::new(),
        };
        for u in &updates {
            if matches!(u.target, PropTarget::Visible)
                && let kirie_script::ScriptValue::Bool(v) = &u.value
            {
                self.visible_by_id.insert(u.object_id, *v);
            }
        }
        for (id, path) in self.script.as_mut().map(|s| s.take_created()).unwrap_or_default() {
            tracing::debug!(id, %path, "runtime layer created by script");
            self.runtime_layers.entry(id).or_default();
        }
        if !updates.is_empty() {
            apply_runtime_updates(&mut self.runtime_layers, &updates);
            apply_script_updates(&mut self.items, &updates);
        }
    }

    /// Platform-fed pointer (T26): drives `g_PointerPosition*`, camera parallax
    /// and SceneScript `pointer_screen` on the following frames.
    fn set_pointer(&mut self, x: f32, y: f32) {
        self.pointer = [x, y];
    }
}

// ---- helpers ---------------------------------------------------------------

/// Solid translucent pipeline for runtime layers: NDC position + straight RGBA
/// per vertex, standard alpha blend into the RGBA16F scene FBO.
fn build_runtime_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    const SRC: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> };
@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.color = color;
    return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> { return i.color; }
"#;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirie-runtime-layer-shader"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kirie-runtime-layer-layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("kirie-runtime-layer-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 24,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: super::fbo::FBO_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Apply a frame's SceneScript property updates to the live image objects
/// (docs/scripting-api.md §5.1/§8). Only the fields flowing into per-object
/// builtin uniforms (`alpha`/`brightness`/`color`) and visibility are driven;
/// text/transform targets are tracked in the script layer snapshot but not
/// composited here (see [`super::scripting`] module docs).
fn apply_script_updates(items: &mut [SceneItem], updates: &[PropUpdate]) {
    for u in updates {
        for item in items.iter_mut() {
            let SceneItem::Image(object) = item else { continue };
            if object.id != u.object_id {
                continue;
            }
            match u.target {
                PropTarget::Alpha => {
                    if let Some(a) = as_f32(&u.value) {
                        object.alpha = a;
                    }
                }
                PropTarget::Brightness => {
                    if let Some(b) = as_f32(&u.value) {
                        object.brightness = b;
                    }
                }
                PropTarget::Color => {
                    if let Some(c) = as_rgb(&u.value) {
                        object.color = [c[0], c[1], c[2], object.color[3]];
                    }
                }
                PropTarget::Visible => {
                    if let kirie_script::ScriptValue::Bool(v) = &u.value {
                        object.visible = *v;
                    }
                }
                // Text handled by the text pipeline; transforms only drive
                // runtime layers (applied in `apply_runtime_updates`).
                PropTarget::Text | PropTarget::Origin | PropTarget::Scale | PropTarget::Angles => {}
            }
        }
    }
}

/// Apply script updates addressed to runtime layers (synthetic ids from
/// `createLayer`): transform, color, alpha and visibility all drive the solid
/// quad drawn for the layer each frame.
fn apply_runtime_updates(
    layers: &mut std::collections::HashMap<i64, RuntimeLayer>,
    updates: &[PropUpdate],
) {
    use super::scripting::as_vec3;
    for u in updates {
        let Some(l) = layers.get_mut(&u.object_id) else { continue };
        match u.target {
            PropTarget::Origin => {
                if let Some(v) = as_vec3(&u.value) {
                    l.origin = v;
                }
            }
            PropTarget::Scale => {
                if let Some(v) = as_vec3(&u.value) {
                    l.scale = v;
                }
            }
            PropTarget::Angles => {
                if let Some(v) = as_vec3(&u.value) {
                    l.angles = v;
                }
            }
            PropTarget::Color => {
                if let Some(c) = as_rgb(&u.value) {
                    l.color = c;
                }
            }
            PropTarget::Alpha => {
                if let Some(a) = as_f32(&u.value) {
                    l.alpha = a;
                }
            }
            PropTarget::Visible => {
                if let kirie_script::ScriptValue::Bool(v) = &u.value {
                    l.visible = *v;
                }
            }
            PropTarget::Brightness | PropTarget::Text => {}
        }
    }
}

/// Draw one image object's pass chain into the scene FBO (docs §7.1): effect
/// passes ping-pong through the object's image FBOs (clearing each to
/// transparent), the final pass composites into `scene_view` (`LoadOp::Load`).
/// Factored out of [`SceneRenderer::render`] so the item loop can borrow the
/// object mutably while reading the renderer's other fields.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn draw_image_object(
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    object: &ObjectGpu,
    scene_view: &wgpu::TextureView,
    screen_mvp: Mat4,
    ambient: [f32; 3],
    skylight: [f32; 3],
    time: f32,
    elapsed: f64,
    texel: [f32; 2],
    audio: Option<&AudioSpectrum>,
    scratch: &mut Vec<u8>,
    pointer: [f32; 2],
    pointer_last: [f32; 2],
    parallax: ([f32; 2], f32, f32),
) {
    // Camera parallax (CImage.cpp:1173): this layer's scene-space shift is
    // `(depth + amount) · displacement · scene_width` per axis. Zero whenever
    // parallax is off (displacement stays zero).
    let (disp, amount, ref_size) = parallax;
    let px = (object.parallax_depth[0] + amount) * disp[0] * ref_size;
    let py = (object.parallax_depth[1] + amount) * disp[1] * ref_size;
    let parallax_mvp = if px != 0.0 || py != 0.0 {
        matrix::mul(&screen_mvp, &matrix::translation([px, py, 0.0]))
    } else {
        screen_mvp
    };
    // Animated-atlas frame placement for the layer texture (the reference's
    // per-pass `resolveTextureAnimationState`, `CPass.cpp:287-306, 348-378`):
    // `g_Texture0Translation = frame origin / page dims`, `g_Texture0Rotation =
    // frame axes / page dims`. Only the pass that samples the layer texture
    // (the first pass — every later pass's texture0 is a composite FBO, which
    // the reference reports as not-animated) receives the state; SPRITESHEET
    // shaders remap `v_TexCoord` from it (`genericimage2.vert:99-101`).
    let atlas_anim = object.atlas.as_ref().map(|a| {
        let f = a.placement_at(elapsed);
        (f.translation, f.axes)
    });
    for (pass_index, pass) in object.passes.iter().enumerate() {
        let (t0_translation, t0_rotation) = match (pass_index, &atlas_anim) {
            (0, Some((t, r))) => (*t, *r),
            _ => ([0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
        };
        // Per-frame builtins for this pass.
        let mvp = match pass.geometry {
            // Scene-space geometry (flat quad or scene-space puppet mesh) maps
            // through the screen MVP; a copy-space puppet mesh maps through the
            // image's own ortho model matrix into its copy FBO; copy/pass quads
            // are already in NDC.
            Geometry::Scene | Geometry::Puppet => parallax_mvp,
            Geometry::PuppetCopy => pass.model_matrix,
            Geometry::Copy | Geometry::Pass => matrix::IDENTITY,
        };
        // Shaders unprojecting the pointer (xray: `mul(ndc, MVPInverse)` then
        // `× 1/g_Texture0Resolution`) need the REFERENCE's inverse, which maps
        // NDC into the pass's local space:
        // - Scene/Puppet passes: the reference renders the image quad under
        //   `ortho × model`, so its inverse is `inverse(screen_mvp × model)` —
        //   kirie bakes the model into the vertices and feeds a pure-ortho
        //   forward MVP, so the plain `inverse(mvp)` misses the model part.
        // - Copy/Pass effect quads: pre-baked NDC with identity MVP; the
        //   reference's ortho inverse maps NDC → image pixels of tex0.
        let mvp_inverse = match pass.geometry {
            Geometry::Scene | Geometry::Puppet => Some(matrix::inverse(&matrix::mul(
                &parallax_mvp,
                &pass.model_matrix,
            ))),
            Geometry::Copy | Geometry::Pass => {
                let (tw, th) = (pass.tex_resolution[0][0], pass.tex_resolution[0][1]);
                (tw > 0.0 && th > 0.0).then(|| {
                    let mut m = matrix::IDENTITY;
                    m[0] = tw / 2.0; // x: (ndc+1)·w/2
                    m[5] = th / 2.0; // y: (ndc+1)·h/2 (top-left UV space)
                    m[12] = tw / 2.0;
                    m[13] = th / 2.0;
                    m
                })
            }
            Geometry::PuppetCopy => None,
        };
        let builtins = Builtins {
            time,
            daytime: 0.0,
            brightness: object.brightness,
            alpha: object.alpha,
            color: object.color,
            ambient,
            skylight,
            pointer,
            pointer_last,
            texel_size: texel,
            mvp,
            mvp_inverse,
            model: pass.model_matrix,
            view_projection: matrix::IDENTITY,
            eye: [0.0, 0.0, 1000.0],
            texture0_translation: t0_translation,
            texture0_rotation: t0_rotation,
            texture_resolution: pass.tex_resolution,
            // Live audio bands (mono; Left == Right, filled by `components`).
            // Silent (zeros) when no capture is running (docs §8.3).
            audio16: audio.map_or([0.0; 16], |a| a.audio16),
            audio32: audio.map_or([0.0; 32], |a| a.audio32),
            audio64: audio.map_or([0.0; 64], |a| a.audio64),
        };
        if let Some(ubo) = &pass.vs_ubo {
            pack_globals(scratch, &pass.vs_globals, &builtins, &pass.vs_params);
            queue.write_buffer(ubo, 0, scratch);
        }
        if let Some(ubo) = &pass.fs_ubo {
            pack_globals(scratch, &pass.fs_globals, &builtins, &pass.fs_params);
            queue.write_buffer(ubo, 0, scratch);
        }

        let (target_view, load) = match &pass.output {
            PassOutput::Scene => (scene_view, wgpu::LoadOp::Load),
            PassOutput::Fbo(i) => (
                object.fbos[*i].as_ref().map_or(scene_view, |f| &f.view),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            ),
            PassOutput::Named(name) => (
                object.named_fbos.get(name).map_or(scene_view, |f| &f.view),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            ),
        };
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("kirie-scene-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&pass.pipeline);
        rp.set_bind_group(0, &pass.g0_bind, &[]);
        rp.set_bind_group(1, &pass.g1_bind, &[]);
        rp.set_vertex_buffer(0, pass.vertex_buffer.slice(..));
        if let Some(indices) = &pass.puppet_indices {
            // Puppet base pass: deformable mesh as an indexed triangle list.
            rp.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
            rp.draw_indexed(0..pass.puppet_index_count, 0, 0..1);
        } else {
            rp.draw(0..4, 0..1);
        }
        let _ = pass.blending;
    }
}

/// std140 blit window uniform: scaling UV window + clamp mode + sRGB-cancel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlitWindow {
    rect: [f32; 4],
    clamp_mode: u32,
    /// 1 when the output surface is an sRGB format: the blit sRGB-*decodes* its
    /// sampled color so wgpu's automatic linear→sRGB encode on store cancels
    /// out, writing the raw scene-FBO bytes to the surface. This reproduces the
    /// reference's gamma-naive blit (raw `Rgba16F` → 8-bit default framebuffer,
    /// no `GL_FRAMEBUFFER_SRGB`; docs/render-architecture.md §2.5, §6). Without
    /// it the encode brightens the whole frame (clearcolor, shader output, and
    /// every layer) and tanks SSIM against the oracle.
    srgb: u32,
    _pad: [u32; 2],
}

/// The scene screen view-projection for the 2D compositor — the reference's
/// `Camera::setOrthogonalProjection` + `lookAt` pair (`Camera.cpp:76-91`:
/// `ortho(-w/2..w/2, -h/2..h/2, 0, max(farz, 1000))`, then
/// `projection = translate(projection, eye)`; `Camera.cpp:14`:
/// `lookat = lookAt(eye, center, up)`; every image/text draw uses
/// `projection * lookat`), **conjugated by the Y-mirror**.
///
/// The conjugation is the X/Y camera-tilt port: the reference composites in a
/// Y-mirrored GL space and presents the frame vertically flipped
/// (`WaylandOutput.cpp:34` `renderVFlip = true`; see the text
/// `scene_space_quad` note in `extras.rs`), while kirie builds Y-up geometry
/// and presents unflipped. Feeding the reference matrix `M` to mirrored
/// vertices is only correct when `M` commutes with the mirror `F =
/// diag(1,-1,1)` — true for the corpus-dominant centered camera (eye/center on
/// the z-axis, up `+Y`, where the folded eye and the lookAt translation cancel
/// and flat z=0 layers map straight through the ortho), but wrong for an
/// off-axis eye/center: a vertical tilt (an `eye→center` Y displacement
/// rotating about X) would bend the scene the mirrored way. `F · M · F`
/// re-expresses the camera in kirie's space — X tilt (rotation about Y, xz
/// terms) is mirror-even and unchanged; Y tilt and camera roll pick up the
/// correctly-signed direction.
fn screen_camera_mvp(proj: (u32, u32), eye: [f32; 3], center: [f32; 3], up: [f32; 3], farz: f32) -> Mat4 {
    let far = farz.max(1000.0);
    let ortho = matrix::ortho(
        -(proj.0 as f32) / 2.0,
        proj.0 as f32 / 2.0,
        -(proj.1 as f32) / 2.0,
        proj.1 as f32 / 2.0,
        0.0,
        far,
    );
    let proj_eye = matrix::translate(&ortho, eye);
    let look = matrix::look_at(eye, center, up);
    let reference = matrix::mul(&proj_eye, &look);
    let flip = matrix::scale([1.0, -1.0, 1.0]);
    matrix::mul(&flip, &matrix::mul(&reference, &flip))
}

/// Compute the scene projection size (docs §5): explicit ortho size, else an
/// auto size from image extents, else a 1080p fallback.
fn projection_size(model: &SceneModel) -> (u32, u32) {
    match model.scene.camera.projection {
        Projection::Orthogonal { width, height } if width > 0 && height > 0 => (width as u32, height as u32),
        _ => auto_projection(model),
    }
}

/// Auto projection: `2 × max(|origin| + size/2)` over image objects
/// (docs §5, `CScene.cpp:44-75`); falls back to 1920×1080 when empty.
fn auto_projection(model: &SceneModel) -> (u32, u32) {
    let mut ext_w = 0.0f32;
    let mut ext_h = 0.0f32;
    for object in &model.scene.objects {
        if let ObjectKind::Image(img) = &object.kind {
            let ox = object.base.origin.value[0].abs();
            let oy = object.base.origin.value[1].abs();
            ext_w = ext_w.max(ox + img.size[0] / 2.0);
            ext_h = ext_h.max(oy + img.size[1] / 2.0);
        }
    }
    let w = (ext_w * 2.0).round() as u32;
    let h = (ext_h * 2.0).round() as u32;
    if w == 0 || h == 0 { (1920, 1080) } else { (w, h) }
}

/// The base material's first-pass texture-slot-0 name (docs §7.1), if any.
fn base_layer_name(image: &ImageObject) -> Option<String> {
    image
        .material
        .as_ref()
        .and_then(|m| m.passes.first())
        .and_then(|p| p.textures.first())
        .and_then(|slot| slot.clone())
}

/// Whether a texture name refers to the scene FBO (`_rt_FullFrameBuffer` or its
/// mip alias) — a post-process layer's read of the composite-so-far (docs §6).
pub(super) fn is_scene_rt(name: &str) -> bool {
    name == "_rt_FullFrameBuffer" || name == "_rt_MipMappedFrameBuffer"
}

/// The base layer texture: the base material's first pass, texture slot 0
/// (docs §7.1). White when absent or when the slot is an FBO reference (the
/// scene-FBO case is handled by the snapshot in [`build_object`]).
fn base_layer_texture(
    image: &ImageObject,
    source: &dyn AssetSource,
    registry: &mut TextureRegistry,
) -> std::sync::Arc<super::texture::GpuTexture> {
    match base_layer_name(image) {
        Some(n) if !n.starts_with("_rt_") && !n.starts_with("_alias_") => registry.get(&n, source),
        _ => registry.white(),
    }
}

/// A 4-vertex triangle-strip fullscreen NDC quad (TL, BL, TR, BR) with uv
/// `0..ucrop × 0..vcrop`.
fn ndc_quad(ucrop: f32, vcrop: f32) -> [[f32; 5]; 4] {
    [
        [-1.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0, 0.0, vcrop],
        [1.0, 1.0, 0.0, ucrop, 0.0],
        [1.0, -1.0, 0.0, ucrop, vcrop],
    ]
}

/// A single object's local 2D transform, used to compose the parent chain
/// (docs §7.3). Only translation/scale/Z-rotation participate — the corpus's
/// group transforms are planar.
#[derive(Clone, Copy)]
struct LocalXf {
    origin: [f32; 2],
    scale: [f32; 2],
    angle_z: f32,
    parent: Option<i64>,
}

/// A composed world transform `(origin, scale, angle_z)` in JSON pixel space
/// (Y-down), ready for [`scene_space_quad`].
type WorldXf = ([f32; 2], [f32; 2], f32);

/// Compose an object's world transform by walking its parent chain root-first
/// (docs §7.3): each level applies its own scale + Z-rotation + translation to
/// the accumulated frame. A missing/dangling parent stops the walk; a cycle is
/// capped (SPEC.md §V9). Top-level objects (no parent) return their own local
/// transform unchanged, so unparented layers render exactly as before.
fn world_xf(id: i64, locals: &HashMap<i64, LocalXf>) -> WorldXf {
    // Collect node → … → root.
    let mut chain: Vec<LocalXf> = Vec::new();
    let mut cur = Some(id);
    for _ in 0..64 {
        let Some(c) = cur else { break };
        let Some(l) = locals.get(&c) else { break };
        chain.push(*l);
        cur = l.parent;
    }
    let (mut ox, mut oy) = (0.0f32, 0.0f32);
    let (mut sx, mut sy) = (1.0f32, 1.0f32);
    let mut ang = 0.0f32;
    // Apply root → node so a parent's transform frames its children.
    for l in chain.iter().rev() {
        let (lx, ly) = (l.origin[0] * sx, l.origin[1] * sy);
        let (s, c) = ang.sin_cos();
        ox += lx * c - ly * s;
        oy += lx * s + ly * c;
        sx *= l.scale[0];
        sy *= l.scale[1];
        ang += l.angle_z;
    }
    ([ox, oy], [sx, sy], ang)
}

/// The scene-space quad for an image (docs §7.1): centered Y-up scene coords.
fn scene_space_quad(
    origin: [f32; 2],
    size: (u32, u32),
    scale: [f32; 2],
    angle_z: f32,
    scene: (u32, u32),
) -> [[f32; 5]; 4] {
    let (sw, sh) = (scene.0 as f32, scene.1 as f32);
    // Scaled half-extents (docs §7.1: the quad is built at the scaled size).
    let hw = size.0 as f32 / 2.0 * scale[0];
    let hh = size.1 as f32 / 2.0 * scale[1];
    // Layer center in centered scene space. scene.json origin is Y-UP, so the
    // prior `sh/2 - origin.y` reflected every off-center layer about the scene
    // mid-line — masked for Y-centered content, but it flipped e.g. 3428443753's
    // off-center 男涂鸦 graffiti (origin.y=470) UP over the character's face.
    let cx = origin[0] - sw / 2.0;
    let cy = origin[1] - sh / 2.0;
    // In-plane rotation about the quad center by the negated Z angle
    // (CImage.cpp `updateScreenSpacePosition`, docs §7.1).
    let (s, c) = (-angle_z).sin_cos();
    // Corner offsets in Y-up (top = +hh), matching the UVs below.
    let corner = |dx: f32, dy: f32| [cx + dx * c - dy * s, cy + dx * s + dy * c, 0.0];
    let tl = corner(-hw, hh);
    let bl = corner(-hw, -hh);
    let tr = corner(hw, hh);
    let br = corner(hw, -hh);
    [
        [tl[0], tl[1], 0.0, 0.0, 0.0],
        [bl[0], bl[1], 0.0, 0.0, 1.0],
        [tr[0], tr[1], 0.0, 1.0, 0.0],
        [br[0], br[1], 0.0, 1.0, 1.0],
    ]
}

/// The blending a pass actually renders with. The base pass of a *loaded*
/// puppet mesh is forced Translucent — `CImage::setupPasses` does
/// `pass->setBlendingMode (BlendingMode_Translucent)` on the first pass when
/// `m_hasPuppetMesh` (`CImage.cpp:832-834`), *after* the §7.1 first→last blend
/// relocation (`CImage.cpp:789-795`) already set it to Normal. The force is
/// applied here (not in the pure plan) because it is gated on the puppet `.mdl`
/// actually parsing, exactly like `m_hasPuppetMesh`: a corrupt puppet falls
/// back to the flat quad *and* keeps the relocated Normal, as the reference
/// would. Overlapping puppet triangles need it — Normal (replace) would punch
/// alpha-0 holes where a transparent margin overdraws an opaque texel.
fn effective_blending(
    is_puppet_base: bool,
    planned: kirie_scene::material::Blending,
) -> kirie_scene::material::Blending {
    if is_puppet_base {
        kirie_scene::material::Blending::Translucent
    } else {
        planned
    }
}

/// Interleaved `[x, y, z, u, v]` local-space puppet vertices in the reference's
/// `updatePuppetPositionBuffer` space (`size/2 ± p`, `CImage.cpp:536-540`). Both
/// puppet paths upload these same local vertices; only the MVP differs. A
/// multi-pass character (`Geometry::PuppetCopy`) draws them through the image's
/// own ortho model matrix into its copy FBO, so the effect chain then processes
/// the deformed character and the final flat composite places it at the object
/// origin. A single-pass character (`Geometry::Puppet`) draws them through the
/// *screen* MVP with no origin translation — exactly the reference's puppet
/// geometry callback + `m_modelViewProjectionScreen` override (`CImage.cpp:832,
/// 855-856`), which lands the mesh near the scene center where later objects
/// occlude it (this is why the fortune-cat in scene 3428443753 is hidden behind
/// the two characters drawn after it, matching the oracle). UVs are the mesh
/// UVs, NPOT-cropped like the flat copy quad.
fn puppet_copy_vertices(
    mesh: &kirie_formats::model::PuppetMesh,
    size: (u32, u32),
    _uv_crop: [f32; 2],
) -> Vec<f32> {
    let (hw, hh) = (size.0 as f32 / 2.0, size.1 as f32 / 2.0);
    let mut out = Vec::with_capacity(mesh.vertices.len() * 5);
    for v in &mesh.vertices {
        // `+y` (not the reference's `size/2 - y`): the reference's copy projection
        // flips clip-space Y for its Y-down FBO, kirie's ortho does not, so the
        // sign is absorbed here to keep the character upright once the composite
        // samples the copy FBO through the Y-up scene quad (cf. the model winding
        // note in `model.rs`).
        //
        // Puppet texcoords are used RAW — the reference pushes the mesh's `u,v`
        // straight into `m_puppetTexCoord` (`CImage.cpp:491-492`) and never applies
        // the `texcoordCopy = realSize/textureSize` NPOT crop that the flat-quad
        // copy path uses. The mesh author already baked the atlas-page coordinates
        // into the UVs, so scaling them by `uv_crop` (~0.99) shifts every sample
        // ~1% toward the atlas origin, misregistering the character and reading the
        // transparent margin at feature/alpha boundaries — the girl's "hole over the
        // eye" that let the LOGO layer behind bleed through (§ girl eye fix).
        out.extend_from_slice(&[hw + v.position[0], hh + v.position[1], 0.0, v.uv[0], v.uv[1]]);
    }
    out
}

/// Upload a puppet mesh's `u16` triangle-list index buffer (`CImage.cpp:512-514`).
/// An odd index count is padded with one trailing index so the byte length is a
/// multiple of wgpu's 4-byte `COPY_BUFFER_ALIGNMENT` (the pad index is never
/// drawn — the draw range stays at the real count).
fn create_puppet_index_buffer(
    device: &wgpu::Device,
    mesh: &kirie_formats::model::PuppetMesh,
) -> wgpu::Buffer {
    let mut indices = mesh.indices.clone();
    if !indices.len().is_multiple_of(2) {
        indices.push(0);
    }
    create_buffer_init(
        device,
        "kirie-puppet-ib",
        bytemuck::cast_slice(&indices),
        wgpu::BufferUsages::INDEX,
    )
}

/// Scale a quad's UV columns (indices 3, 4) into a texture's real sub-rect —
/// the reference's `texcoordCopy = realSize / textureSize` NPOT-padding crop
/// (docs §7.1). Base UVs run 0..1; after this they run 0..crop, so sampling the
/// padded `.tex` page hits only the real image and the padding never composites.
/// `g_TextureNResolution` value for a texture: `(texW, texH, realW, realH)`,
/// where `real` is the logical content size — the NPOT header crop for stills,
/// `gifWidth/gifHeight` for animated atlases — exactly the reference's
/// `CTexture::setupResolution` (`CTexture.cpp:149-153`; docs/format-tex.md
/// §8.1). Shaders derive the padding crop as `real/tex`.
pub(super) fn tex_res(t: &super::texture::GpuTexture) -> [f32; 4] {
    [t.width as f32, t.height as f32, t.real_size[0], t.real_size[1]]
}

/// Scale a quad's UV columns (indices 3, 4) into a texture's real sub-rect —
/// the reference's `texcoordCopy = realSize / textureSize` NPOT-padding crop
/// (docs §7.1). Base UVs run 0..1; after this they run 0..crop, so sampling the
/// padded `.tex` page hits only the real image and the padding never composites.
fn apply_uv_crop(verts: &mut [[f32; 5]; 4], crop: [f32; 2]) {
    for v in verts.iter_mut() {
        v[3] *= crop[0];
        v[4] *= crop[1];
    }
}

/// Upload a 4-vertex strip; `with_uv` decides whether the uv columns matter
/// (they are uploaded regardless — the vertex layout drops them when unused).
pub(super) fn create_vertex_buffer(
    device: &wgpu::Device,
    verts: &[[f32; 5]; 4],
    _with_uv: bool,
) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity(4 * 20);
    for v in verts {
        for f in v {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    }
    create_buffer_init(device, "kirie-scene-vb", &bytes, wgpu::BufferUsages::VERTEX)
}

pub(super) fn create_ubo(device: &wgpu::Device, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kirie-scene-ubo"),
        size: size.max(16) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(super) fn create_buffer_init(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("freshly mapped buffer");
        view.slice(..data.len()).copy_from_slice(data);
    }
    buffer.unmap();
    buffer
}

/// Resolve a stage's material parameters to global-member values by uniform
/// name (docs §8.3): the pass `constantshadervalues` keyed by the annotation
/// `material` name, else the annotation default.
pub(super) fn resolve_params(
    params: &[Parameter],
    pass: &kirie_scene::material::Pass,
) -> BTreeMap<String, Vec<f32>> {
    let mut out = BTreeMap::new();
    for p in params {
        let value = pass
            .constantshadervalues
            .get(&p.material)
            .map(|us| dynamic_components(&us.value, p))
            .or_else(|| p.default.as_ref().map(default_components));
        if let Some(v) = value {
            out.insert(p.name.clone(), v);
        }
    }
    out
}

fn dynamic_components(dv: &DynamicValue, p: &Parameter) -> Vec<f32> {
    match dv {
        DynamicValue::Vec(v) => v.clone(),
        DynamicValue::Color(c) => c.to_vec(),
        _ => vec![dv.as_f32()],
    }
    .into_iter()
    .chain(std::iter::repeat(0.0))
    .take(param_len(p))
    .collect()
}

fn default_components(d: &ParamDefault) -> Vec<f32> {
    match d {
        ParamDefault::Scalar(s) => vec![*s as f32],
        ParamDefault::Vector(v) => v.clone(),
    }
}

fn param_len(p: &Parameter) -> usize {
    use kirie_shader::reflect::ParamType;
    match p.ty {
        ParamType::Float | ParamType::Int => 1,
        ParamType::Vec2 => 2,
        ParamType::Vec3 => 3,
        ParamType::Vec4 => 4,
    }
}

/// Build a stage bind group from the module's actual `bindings` (ground truth,
/// so it matches the layout exactly). The UBO fills its slot; each texture /
/// sampler binding is resolved via the annotated `samplers` (slot 0 = the pass
/// input; others = named/default), falling back to the white texture and its
/// sampler for any binding the reflection does not describe (an un-annotated
/// sampler the shader still declares — docs/render-architecture.md §8.2, §8.5).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    ubo: Option<&wgpu::Buffer>,
    bindings: &[ModuleBinding],
    samplers: &[SamplerSlot],
    input_view: &wgpu::TextureView,
    input_sampler: &wgpu::Sampler,
    registry: &mut TextureRegistry,
    source: &dyn AssetSource,
    pass: &kirie_scene::material::Pass,
    scene: (&wgpu::TextureView, &wgpu::Sampler),
    named: &std::collections::HashMap<&str, (&wgpu::TextureView, &wgpu::Sampler)>,
) -> wgpu::BindGroup {
    // How a sampler slot resolves: the pass input, the scene snapshot (a
    // `_rt_FullFrameBuffer` read), a named effect/composite FBO (§11.2 `bind`),
    // or a concrete uploaded texture.
    enum Slot<'a> {
        Input,
        Scene,
        Named((&'a wgpu::TextureView, &'a wgpu::Sampler)),
        Tex(std::sync::Arc<super::texture::GpuTexture>),
    }
    let resolved: Vec<Slot> = samplers
        .iter()
        .map(|slot| {
            // The bound name for this slot: an authored/bind texture override, or
            // the annotation default. A named FBO / composite / `previous` bind
            // wins over the slot-0 input default so a combine pass samples the
            // real prior output rather than the 1×1 white (§11.2).
            let name = slot
                .slot
                .and_then(|i| pass.textures.get(i as usize))
                .and_then(|s| s.clone())
                .or_else(|| slot.default_texture.clone());
            // A named FBO / composite / `previous` bind overrides any slot,
            // including slot 0, so a combine pass reads the right scratch buffer.
            if let Some(hit) = name.as_deref().and_then(|n| named.get(n)) {
                return Slot::Named(*hit);
            }
            // A `_rt_FullFrameBuffer` name resolves to the scene snapshot on
            // any slot, slot 0 included (a `command:"copy"` source may be the
            // scene): the reference resolves FBO names before falling back to
            // the pass input (`CPass.cpp` name-based texture resolution). For
            // the base pass this is the same view the Input arm binds.
            if name.as_deref().is_some_and(is_scene_rt) {
                return Slot::Scene;
            }
            // Slot 0 otherwise samples the pass input view (the layer texture or
            // composite front), whose contents already are slot 0's texture.
            if slot.slot == Some(0) {
                return Slot::Input;
            }
            match name {
                Some(n) if !n.starts_with("_rt_") && !n.starts_with("_alias_") => {
                    Slot::Tex(registry.get(&n, source))
                }
                _ => Slot::Tex(registry.white()),
            }
        })
        .collect();
    let white = registry.white();

    let mut entries = Vec::with_capacity(bindings.len());
    for mb in bindings {
        let resource = match mb.kind {
            BindKind::Ubo => match ubo {
                Some(u) => u.as_entire_binding(),
                // The layout only declares a UBO when the module does, so this
                // is unreachable; skip defensively rather than panic (§V9).
                None => continue,
            },
            BindKind::Texture => {
                let view = samplers
                    .iter()
                    .position(|s| s.texture_binding == mb.binding)
                    .map_or(&white.view, |i| match &resolved[i] {
                        Slot::Tex(t) => &t.view,
                        Slot::Scene => scene.0,
                        Slot::Named((v, _)) => v,
                        Slot::Input => input_view,
                    });
                wgpu::BindingResource::TextureView(view)
            }
            BindKind::Sampler => {
                let samp = samplers
                    .iter()
                    .position(|s| s.sampler_binding == mb.binding)
                    .map_or(&white.sampler, |i| match &resolved[i] {
                        Slot::Tex(t) => &t.sampler,
                        Slot::Scene => scene.1,
                        Slot::Named((_, s)) => s,
                        Slot::Input => input_sampler,
                    });
                wgpu::BindingResource::Sampler(samp)
            }
        };
        entries.push(wgpu::BindGroupEntry {
            binding: mb.binding,
            resource,
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kirie-scene-bg"),
        layout,
        entries: &entries,
    })
}

/// Build the final blit pipeline sampling the scene FBO through a UV window.
fn build_blit(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    scene_fbo: &Fbo,
    sampler: &wgpu::Sampler,
) -> (wgpu::RenderPipeline, wgpu::BindGroup, wgpu::Buffer) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirie-scene-blit-shader"),
        source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
    });
    let window = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kirie-scene-blit-window"),
        size: std::mem::size_of::<BlitWindow>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("kirie-scene-blit-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kirie-scene-blit-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: window.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&scene_fbo.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kirie-scene-blit-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("kirie-scene-blit-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bind, window)
}

/// The final-blit shader: fullscreen strip sampling the scene FBO through the
/// scaling UV window (docs §2.5, §4). Clamp mode 0 = clamp, 1 = border
/// (transparent black), 2 = repeat.
const BLIT_WGSL: &str = r#"
struct Window { rect: vec4<f32>, clamp_mode: u32, srgb: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> win: Window;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // TL, BL, TR, BR — matches UvWindow::strip_corners ordering.
    var xs = array<f32, 4>(-1.0, -1.0, 1.0, 1.0);
    var ys = array<f32, 4>(1.0, -1.0, 1.0, -1.0);
    var us = array<f32, 4>(0.0, 0.0, 1.0, 1.0);
    var vs = array<f32, 4>(0.0, 1.0, 0.0, 1.0);
    var o: VsOut;
    o.pos = vec4<f32>(xs[i], ys[i], 0.0, 1.0);
    let u = mix(win.rect.x, win.rect.z, us[i]);
    let v = mix(win.rect.y, win.rect.w, vs[i]);
    o.uv = vec2<f32>(u, v);
    return o;
}

// Linear→sRGB inverse (sRGB decode), per channel. Applied before store when the
// surface is sRGB so wgpu's automatic linear→sRGB encode cancels it, writing the
// raw scene-FBO bytes to the surface — the reference's gamma-naive blit.
fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c <= vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, cutoff);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var uv = in.uv;
    if (win.clamp_mode == 2u) {
        uv = fract(uv);
    } else if (win.clamp_mode == 1u) {
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    } else {
        uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    }
    var c = textureSample(scene, samp, uv);
    if (win.srgb == 1u) {
        c = vec4<f32>(srgb_decode(c.rgb), c.a);
    }
    return c;
}
"#;

/// The `commands/copy` blit vertex shader. The reference registers this pair
/// in its virtual asset filesystem instead of shipping files
/// (`WallpaperApplication.cpp:165-182`), so no [`AssetSource`] can resolve
/// them; ported verbatim to the WE GLSL dialect the shader pipeline consumes
/// (`attribute`/`varying`, `gl_FragColor`, `texSample2D`). The pass-space NDC
/// quad (`gl_Position = a_Position`) drawn into the `target` FBO while
/// sampling the `source` at slot 0 IS the copy (`CImage.cpp:704-718`).
const COPY_COMMAND_VERT: &str = "\
attribute vec3 a_Position;\n\
attribute vec2 a_TexCoord;\n\
varying vec2 v_TexCoord;\n\
void main() {\n\
gl_Position = vec4(a_Position, 1.0);\n\
v_TexCoord = a_TexCoord;\n\
}\n";

/// The `commands/copy` blit fragment shader — samples the `source` FBO the
/// plan bound at texture slot 0 (`WallpaperApplication.cpp:165-171`).
const COPY_COMMAND_FRAG: &str = "\
uniform sampler2D g_Texture0;\n\
varying vec2 v_TexCoord;\n\
void main() {\n\
gl_FragColor = texSample2D(g_Texture0, v_TexCoord);\n\
}\n";

#[cfg(test)]
mod tests {
    use super::*;

    /// Transform a point (column vector) by a column-major matrix.
    fn apply(m: &Mat4, p: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            for k in 0..4 {
                out[row] += m[k * 4 + row] * p[k];
            }
        }
        out
    }

    /// The reference's raw screen VP (`Camera.cpp:76-91` + `Camera.cpp:14`),
    /// without kirie's mirror conjugation.
    fn reference_mvp(proj: (u32, u32), eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
        let ortho = matrix::ortho(
            -(proj.0 as f32) / 2.0,
            proj.0 as f32 / 2.0,
            -(proj.1 as f32) / 2.0,
            proj.1 as f32 / 2.0,
            0.0,
            1000.0,
        );
        matrix::mul(&matrix::translate(&ortho, eye), &matrix::look_at(eye, center, up))
    }

    #[test]
    fn centered_camera_mvp_is_mirror_invariant() {
        // The corpus-dominant camera (eye/center on the z-axis, up +Y) is
        // Y-even, so the conjugation must be a no-op — no regression for
        // every already-verified scene.
        let (eye, center, up) = ([0.0, 0.0, 1000.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let conj = screen_camera_mvp((1920, 1080), eye, center, up, 1000.0);
        let plain = reference_mvp((1920, 1080), eye, center, up);
        for (i, (a, b)) in conj.iter().zip(plain.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "elem {i}: {a} vs {b}");
        }
    }

    #[test]
    fn tilted_camera_matches_the_flipped_reference() {
        // An off-axis center tilts the view (rotation about X — the "X/Y
        // camera tilt"). The reference draws GL-space vertices vR under its
        // raw MVP and presents the frame vertically flipped
        // (WaylandOutput.cpp:34); kirie draws the mirrored vertex F·vR under
        // the conjugated MVP and presents unflipped. Both must land every
        // point on the same clip position: conj · (F·vR) == F_ndc · (ref · vR).
        let (eye, center, up) = ([0.0, 0.0, 1000.0], [0.0, 300.0, 0.0], [0.0, 1.0, 0.0]);
        let conj = screen_camera_mvp((1920, 1080), eye, center, up, 1000.0);
        let reference = reference_mvp((1920, 1080), eye, center, up);
        for v in [
            [0.0f32, 0.0, 0.0, 1.0],
            [100.0, 200.0, 0.0, 1.0],
            [-50.0, -120.0, 0.0, 1.0],
        ] {
            let kirie = apply(&conj, [v[0], -v[1], v[2], v[3]]);
            let mut expected = apply(&reference, v);
            expected[1] = -expected[1];
            for (i, (a, b)) in kirie.iter().zip(expected.iter()).enumerate() {
                assert!((a - b).abs() < 1e-4, "clip {i}: {a} vs {b} for {v:?}");
            }
        }
        // And the conjugation has teeth here: the unconjugated matrix would
        // send off-center points to the mirrored (wrong-signed) tilt.
        let p = [0.0f32, 200.0, 0.0, 1.0];
        let old = apply(&reference, p);
        let new = apply(&conj, p);
        assert!((old[1] - new[1]).abs() > 1e-3, "tilt must not be mirror-even");
    }

    #[test]
    fn puppet_base_forces_translucent_blending() {
        // `CImage.cpp:832-834`: a loaded puppet's first pass is forced
        // Translucent regardless of what relocation/material left there; every
        // other pass keeps its planned blending.
        use kirie_scene::material::Blending;
        for planned in [Blending::Normal, Blending::Translucent, Blending::Additive] {
            assert_eq!(effective_blending(true, planned), Blending::Translucent);
            assert_eq!(effective_blending(false, planned), planned);
        }
    }

    #[test]
    fn embedded_copy_command_shader_translates() {
        // The `commands/copy` pair never touches an asset container, so a
        // translation regression would silently drop every copy pass (§V9
        // skip). Translate both stages exactly as `pipeline::build_pass` does
        // (CPU-only; no GPU device involved).
        struct NoIncludes;
        impl IncludeResolver for NoIncludes {
            fn resolve(&self, _: &str) -> Option<String> {
                None
            }
        }
        let inputs = kirie_shader::ShaderInputs::default();
        kirie_shader::translate(
            kirie_shader::Stage::Vertex,
            "copy.vert",
            COPY_COMMAND_VERT,
            &NoIncludes,
            &inputs,
        )
        .expect("commands/copy vertex stage must translate");
        kirie_shader::translate(
            kirie_shader::Stage::Fragment,
            "copy.frag",
            COPY_COMMAND_FRAG,
            &NoIncludes,
            &inputs,
        )
        .expect("commands/copy fragment stage must translate");
    }

    #[test]
    fn uv_crop_scales_only_uv_columns_into_real_subrect() {
        // A page padded to 2× the real height (v_crop = 0.5) must map the base
        // 0..1 V range into 0..0.5 so the padding below never samples (docs §7.1
        // texcoordCopy = realSize/textureSize). X/Y/Z position columns untouched.
        let mut q = scene_space_quad([960.0, 540.0], (1920, 1080), [1.0, 1.0], 0.0, (1920, 1080));
        let pos_before: Vec<[f32; 3]> = q.iter().map(|v| [v[0], v[1], v[2]]).collect();
        apply_uv_crop(&mut q, [0.9375, 0.5]);
        for (v, p) in q.iter().zip(&pos_before) {
            assert_eq!([v[0], v[1], v[2]], *p, "position columns must not move");
        }
        // Corner UVs: (0,0),(0,1),(1,0),(1,1) -> scaled by crop.
        assert_eq!([q[0][3], q[0][4]], [0.0, 0.0]);
        assert_eq!([q[1][3], q[1][4]], [0.0, 0.5]);
        assert_eq!([q[2][3], q[2][4]], [0.9375, 0.0]);
        assert_eq!([q[3][3], q[3][4]], [0.9375, 0.5]);
    }
}
