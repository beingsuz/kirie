//! Integration glue: build a scene renderer straight from a Wallpaper Engine
//! workshop item directory (`scene.pkg` + `project.json`).
//!
//! This is the single entry the kirie binary's compat surface calls to turn a
//! resolved `scene`-type background into a live [`Renderer`]. All the actual
//! per-frame render logic lives in [`super::renderer`]; this module only wires
//! the format/scene loaders together the same way the corpus test does:
//!
//! 1. Open the `scene.pkg` container ([`OwnedPkg`]).
//! 2. Build the [`PropertyBag`] from `project.json` (user-property defaults).
//! 3. Consult the kirie-bake bundle cache ([`super::bundle`], Phase 3.1) — a
//!    hit yields the resolved [`SceneModel`] directly and skips steps 4–5.
//! 4. Parse `scene.json` out of the pkg and [`SceneModel::resolve`] it.
//! 5. Load referenced material/effect/model/particle assets against a
//!    [`CompositeSource`] — the pkg first, then the shared WE builtin-assets
//!    dir (docs/render-architecture.md §10 asset lookup) — then bake the
//!    resolved model into the bundle cache (inline, best-effort).
//! 6. Build the [`SceneRenderer`].
//!
//! Best-effort per SPEC.md §V9 and the P4 corpus-render gate: a scene whose
//! objects all skip (unsupported object kinds, all-invisible after property
//! resolution, or per-object shader/texture build failures) does **not** sink
//! the wallpaper. Instead of erroring it degrades to a [`ClearColorRenderer`]
//! that fills the surface with the scene's `clearcolor`, so the output is the
//! scene's own background rather than black.

use std::path::Path;
use std::sync::Arc;

use kirie_audio::AudioCapture;
use kirie_formats::pkg::OwnedPkg;
use kirie_formats::project::Project;
use kirie_platform::{RenderTarget, Renderer, SurfaceSize};
use kirie_scene::resolve::AssetSource;
use kirie_scene::{PropertyBag, PropertyValue, Scene, SceneModel};

use super::error::SceneError;
use super::renderer::{SceneOptions, SceneRenderer};

/// Parse a `--set-property` raw string into the declared property's type and
/// apply it to the bag (docs/format-scene-json.md §3.2). The target type is
/// taken from the property's current value, so a color stays a color, a slider
/// a number, etc.; unknown keys are ignored (reference `setProperty` semantics).
fn apply_property_override(bag: &mut PropertyBag, name: &str, raw: &str) -> bool {
    let Some(current) = bag.get(name) else {
        return false;
    };
    let parsed = match current {
        PropertyValue::Bool(_) => {
            let t = matches!(raw.trim(), "1" | "true" | "True" | "TRUE");
            PropertyValue::Bool(t)
        }
        PropertyValue::Number(_) => match raw.trim().parse::<f64>() {
            Ok(n) => PropertyValue::Number(n),
            Err(_) => return false,
        },
        PropertyValue::Color(_) => {
            // "r g b" (or "r g b a"), space-separated floats (§3.4 color form).
            let mut c = [0.0f32; 4];
            c[3] = 1.0;
            let mut any = false;
            for (i, tok) in raw.split_whitespace().take(4).enumerate() {
                match tok.parse::<f32>() {
                    Ok(v) => {
                        c[i] = v;
                        any = true;
                    }
                    Err(_) => return false,
                }
            }
            if !any {
                return false;
            }
            PropertyValue::Color(c)
        }
        // Combos compare by their selected option string (§3.3); text is verbatim.
        PropertyValue::Combo(_) => PropertyValue::Combo(raw.trim().to_owned()),
        PropertyValue::Text(_) => PropertyValue::Text(raw.to_owned()),
    };
    bag.set(name, parsed)
}

/// Everything that can go wrong turning a workshop directory into a renderer
/// *before* the per-object best-effort kicks in (SPEC.md §V9: typed, no panic).
#[derive(Debug, thiserror::Error)]
pub enum SceneLoadError {
    /// The `scene.pkg` container could not be opened/parsed.
    #[error("cannot open scene.pkg: {0}")]
    Pkg(String),
    /// `scene.json` was missing from the container or could not be read.
    #[error("cannot read scene.json from scene.pkg: {0}")]
    SceneJson(String),
    /// `scene.json` failed to parse into the object model.
    #[error("cannot parse scene.json: {0}")]
    Parse(String),
    /// The renderer could not be built even after the best-effort fallback
    /// (e.g. a degenerate projection with no valid render target).
    #[error("cannot build scene renderer: {0}")]
    Build(#[from] SceneError),
}

/// An [`AssetSource`] resolving a path against the scene's `scene.pkg` first
/// (byte-exact entry name, docs/format-pkg.md §2), then the shared builtin WE
/// assets directory on disk — mirroring the C++ engine, which reads
/// scene-local assets from the container and builtin shaders/materials from its
/// install (docs/render-architecture.md §10).
struct CompositeSource<'a> {
    pkg: &'a OwnedPkg,
    assets: Option<&'a Path>,
}

impl AssetSource for CompositeSource<'_> {
    fn load(&self, path: &str) -> Option<Vec<u8>> {
        if let Ok(bytes) = self.pkg.read_name(path.as_bytes()) {
            return Some(bytes.to_vec());
        }
        std::fs::read(self.assets?.join(path)).ok()
    }
}

/// Load a workshop `scene`-type item (`scene.pkg` + `project.json`) from
/// `scene_dir` and build its renderer.
///
/// `assets_dir` is the shared WE builtin-assets directory (used to satisfy
/// builtin shader/material references not bundled in the pkg); pass `None` if it
/// is unavailable — scenes that only use builtin shaders will then degrade to
/// the clear-color fallback rather than erroring.
///
/// Returns a boxed [`Renderer`] ready to hand to the presentation/screenshot
/// layer. See the module docs for the best-effort clear-color degradation.
///
/// `audio` is the shared system-audio capture handle whose latest spectrum
/// feeds the scene's `g_AudioSpectrum*` uniforms (docs §8.3); pass `None` (or a
/// disabled handle) for a permanently silent spectrum.
pub fn load_workshop_scene(
    target: &RenderTarget<'_>,
    scene_dir: &Path,
    assets_dir: Option<&Path>,
    options: SceneOptions,
    audio: Option<Arc<AudioCapture>>,
    properties: &[(String, String)],
) -> Result<Box<dyn Renderer + Send>, SceneLoadError> {
    // Cache this wallpaper's compiled shaders in a folder beside it
    // (`<dir>/.kirie-cache`), so they persist with the wallpaper and never
    // rebuild — independent of `~/.cache`. Falls back to the global cache if the
    // folder isn't writable (best-effort). Set per-thread, so the off-thread
    // preload/swap build uses it too.
    kirie_shader::translate::set_cache_dir(Some(scene_dir.join(".kirie-cache")));
    // mmap the pkg instead of reading it onto the heap: the bytes stay in the
    // page cache (evictable, shared, no RSS spike for multi-hundred-MB
    // packages) and the blake3 key hash streams straight over the mapping.
    // Falls back to the heap read if the mmap fails (exotic filesystems).
    let pkg_path = scene_dir.join("scene.pkg");
    let pkg = match kirie_bake::map_readonly(&pkg_path) {
        Ok(map) => OwnedPkg::from_external(map),
        Err(_) => OwnedPkg::from_path(&pkg_path),
    }
    .map_err(|e| SceneLoadError::Pkg(e.to_string()))?;

    // Missing/unreadable project.json → empty bag (property defaults), matching
    // the corpus loader and the C++ tolerance of a scene without user props.
    // Keep the parsed project so we can enumerate its declared property names
    // for `engine.userProperties` (SceneScript §6.1) — the bag has no iterator.
    // The raw bytes are kept too: they are one input of the bundle-cache key.
    let project_bytes = std::fs::read(scene_dir.join("project.json")).ok();
    let project = project_bytes.as_deref().and_then(|bytes| {
        let value = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
        Project::from_value(value).ok()
    });
    let mut bag = project
        .as_ref()
        .map(PropertyBag::from_project)
        .unwrap_or_default();

    // Fold in `--set-property` overrides *before* resolution so user colors,
    // combos and sliders drive the render (docs/format-scene-json.md §3.2;
    // T33: a `--set-property` change must visibly change output). Each raw
    // string is parsed into the declared property's type; unknown keys are
    // ignored, matching the reference's `setProperty` (property.rs `set`).
    for (name, raw) in properties {
        apply_property_override(&mut bag, name, raw);
    }

    // Snapshot the resolved user-property values (name → value, overrides
    // folded in) for SceneScript's `engine.userProperties` (docs §6.1). A
    // layer-switcher script branches on these (`mode_combo`, `style_left`, …)
    // to decide which layer group is visible; without them every group would
    // read `undefined` and the wrong (or no) group would show.
    let user_props: Vec<(String, PropertyValue)> = project
        .as_ref()
        .map(|p| {
            p.general
                .properties
                .keys()
                .filter_map(|name| bag.get(name).map(|v| (name.clone(), v.clone())))
                .collect()
        })
        .unwrap_or_default();

    let source = CompositeSource {
        pkg: &pkg,
        assets: assets_dir,
    };

    // Phase 3.1: consult the prebaked bundle cache before doing any scene work.
    // The key folds in the pkg + project.json content only — deliberately NOT
    // the property values. The bundle stores the DEFAULTS-resolved model (all
    // bindings retained), and the loader applies the caller's actual values
    // via `SceneModel::reresolve` below — the same resolution pass re-run. So
    // a property change (fov, colors, x-ray combos, …) reuses the one baked
    // bundle instead of re-baking per value combination. A hit skips
    // scene.json parse and every asset-JSON load; a miss builds against the
    // DEFAULT bag, bakes, and then reresolves. Shader translation and texture
    // decode still run in SceneRenderer::new either way (kirie-shader's own
    // content-addressed `.kirie-cache` keeps shaders warm).
    let bundle_cache = kirie_bake::Cache::open_default().ok();
    let bundle_src = super::bundle::bundle_source(pkg.as_bytes(), project_bytes.as_deref(), assets_dir);
    let baked = bundle_cache
        .as_ref()
        .and_then(|cache| super::bundle::try_load_model(cache, &bundle_src));

    let mut model = if let Some(model) = baked {
        model
    } else {
        let scene = {
            let bytes = pkg
                .read_name(b"scene.json")
                .map_err(|e| SceneLoadError::SceneJson(e.to_string()))?;
            Scene::from_slice(bytes).map_err(|e| SceneLoadError::Parse(e.to_string()))?
        };

        // Resolve + load against the project DEFAULTS (no user overrides), so
        // the baked artifact is property-independent.
        let defaults = project
            .as_ref()
            .map(PropertyBag::from_project)
            .unwrap_or_default();
        let mut model = SceneModel::resolve(scene, &defaults);
        // Asset load problems are non-fatal (missing textures/shaders degrade to
        // per-object skips inside the renderer); surface them at trace level.
        let problems = model.load_assets(&source, &defaults);
        for p in &problems {
            tracing::debug!(path = %p.path, reason = %p.reason, "scene asset problem");
        }
        if let Some(cache) = &bundle_cache {
            super::bundle::store_model(cache, &bundle_src, &model);
        }
        model
    };
    // Apply the actual property values (defaults + `--set-property` overrides)
    // over the defaults-resolved model — bindings persist, so this is exactly
    // the resolution the direct path used to do with the full bag.
    model.reresolve(&bag);

    match SceneRenderer::new(target, &model, &source, options, audio, &user_props) {
        Ok(renderer) => Ok(Box::new(renderer)),
        // Best-effort: a scene with no drawable object still presents its own
        // background instead of black (SPEC.md §V9; P4 corpus-render gate).
        Err(SceneError::NoRenderableObjects) => {
            tracing::warn!(
                dir = %scene_dir.display(),
                "scene has no renderable objects; presenting clear color"
            );
            Ok(Box::new(ClearColorRenderer::new(
                target,
                model.scene.general.clearcolor.value,
            )))
        }
        Err(e) => Err(SceneLoadError::Build(e)),
    }
}

/// Start the background pre-baker over a workshop root: every scene item gets
/// its defaults-resolved bundle baked ahead of time (idle single-thread pool;
/// new/updated items via the directory watcher), so the FIRST switch to a
/// never-seen wallpaper already hits the bundle cache. Baking uses the exact
/// same descriptor + defaults-resolution as [`load_workshop_scene`]'s miss
/// path, so keys always line up. Returns `None` (silently) when the cache or
/// root is unavailable — pre-baking is a pure accelerator, never a dependency.
///
/// Keep the returned handle alive for the engine's lifetime; dropping it
/// stops the pool.
pub fn start_background_prebake(
    workshop_root: &Path,
    assets_dir: Option<&Path>,
) -> Option<kirie_bake::BackgroundBaker> {
    let cache = kirie_bake::Cache::open_default().ok()?;
    let assets_a: Option<std::path::PathBuf> = assets_dir.map(std::path::Path::to_path_buf);
    let assets_b = assets_a.clone();

    let source_fn: kirie_bake::SourceFn = std::sync::Arc::new(move |item: &Path| {
        let pkg_path = item.join("scene.pkg");
        let pkg = kirie_bake::map_readonly(&pkg_path).map_err(|e| kirie_bake::BakeError::Io {
            path: pkg_path.clone(),
            source: e,
        })?;
        let project = std::fs::read(item.join("project.json")).ok();
        Ok(super::bundle::bundle_source(
            (*pkg).as_ref(),
            project.as_deref(),
            assets_a.as_deref(),
        ))
    });

    let content_fn: kirie_bake::ContentFn = std::sync::Arc::new(move |item: &Path, _src: &[u8]| {
        let pkg_path = item.join("scene.pkg");
        let map = kirie_bake::map_readonly(&pkg_path).map_err(|e| kirie_bake::BakeError::Io {
            path: pkg_path.clone(),
            source: e,
        })?;
        let pkg =
            OwnedPkg::from_external(map).map_err(|e| kirie_bake::BakeError::Serialize(e.to_string()))?;
        // Shader translation warms per-wallpaper caches; keep the same
        // per-item cache dir convention as the real load.
        kirie_shader::translate::set_cache_dir(Some(item.join(".kirie-cache")));
        let scene = {
            let bytes = pkg
                .read_name(b"scene.json")
                .map_err(|e| kirie_bake::BakeError::Serialize(e.to_string()))?;
            Scene::from_slice(bytes).map_err(|e| kirie_bake::BakeError::Serialize(e.to_string()))?
        };
        let project = std::fs::read(item.join("project.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|value| Project::from_value(value).ok());
        let defaults = project
            .as_ref()
            .map(PropertyBag::from_project)
            .unwrap_or_default();
        let mut model = SceneModel::resolve(scene, &defaults);
        let source = CompositeSource {
            pkg: &pkg,
            assets: assets_b.as_deref(),
        };
        let _ = model.load_assets(&source, &defaults);
        let mut content = kirie_bake::BundleContent::new();
        content
            .set_scene_model(&model)
            .map_err(|e| kirie_bake::BakeError::Serialize(e.to_string()))?;
        Ok(content)
    });

    let mut baker =
        kirie_bake::BackgroundBaker::start(kirie_bake::BakerConfig::new(cache, source_fn, content_fn));
    let mut queued = 0usize;
    if let Ok(entries) = std::fs::read_dir(workshop_root) {
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir.join("scene.pkg").is_file() {
                baker.enqueue(&dir);
                queued += 1;
            }
        }
    }
    let _ = baker.watch(workshop_root);
    tracing::info!(root = %workshop_root.display(), queued, "background pre-bake started");
    Some(baker)
}

/// A minimal renderer that clears its surface to a fixed color — the best-effort
/// fallback for a scene whose objects all skipped. Reuses no per-frame heap
/// allocation (SPEC.md §V5): a single clear pass per frame.
struct ClearColorRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    color: wgpu::Color,
}

impl ClearColorRenderer {
    /// Build from a linear RGBA `clearcolor` (docs/format-scene-json.md §5.1).
    fn new(target: &RenderTarget<'_>, clear: [f32; 4]) -> Self {
        Self {
            device: target.device.clone(),
            queue: target.queue.clone(),
            color: wgpu::Color {
                r: f64::from(clear[0]),
                g: f64::from(clear[1]),
                b: f64::from(clear[2]),
                a: 1.0,
            },
        }
    }
}

impl Renderer for ClearColorRenderer {
    fn render(&mut self, view: &wgpu::TextureView, _size: SurfaceSize, _dt: f32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kirie-scene-clear-encoder"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kirie-scene-clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
