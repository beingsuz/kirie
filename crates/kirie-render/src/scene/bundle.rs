//! Phase 3.1 glue: the [`kirie_bake`] prebaked-bundle cache in the scene load
//! path.
//!
//! [`super::load::load_workshop_scene`] consults the cache before doing any
//! scene work. On a hit the resolved [`SceneModel`] is deserialized straight
//! out of the mmapped bundle, skipping `scene.json` extraction + parse,
//! property resolution, and every referenced asset-JSON load/parse
//! (model/material/effect/particle files). On a miss the model is built the
//! normal way and then baked inline, best-effort — a cache failure never fails
//! a load.
//!
//! ## Key composition (CORRECTNESS OVER SPEED)
//!
//! kirie-bake keys a bundle by `blake3(source) ⊕ BAKE_FORMAT_VERSION ⊕
//! TRANSLATOR_VERSION` over opaque `source` bytes. The bundle stores the
//! **DEFAULTS-resolved** model — property bindings collapsed against the
//! project's declared defaults, with every binding retained (`UserSetting`
//! keeps its `user` ref). The loader then applies the caller's actual
//! property values via [`SceneModel::reresolve`], which is the same
//! resolution pass re-run. So the key pins only what the *bake* reads, and a
//! property change NEVER re-bakes (one bundle per wallpaper, not one per
//! property combination):
//!
//! - the full `scene.pkg` content (scene.json + all pkg-borne asset JSON) —
//!   folded in as its blake3 digest, not the raw bytes, so composing the
//!   descriptor never copies a multi-hundred-MB package;
//! - the `project.json` content (declares the properties, their types AND the
//!   default values the bake resolves against);
//! - the builtin-assets directory *path* (presence + location). `load_assets`
//!   falls back to it for asset JSON not present in the pkg. Its contents are
//!   treated as immutable per WE install — the same assumption the shader
//!   translate cache makes; an in-place edit of a builtin asset JSON is the one
//!   staleness this key does not see.
//!
//! Every field is length-prefixed so the encoding is injective (no
//! concatenation ambiguity).
//!
//! ## What the bundle does / does not cover
//!
//! Covered on a hit: pkg-entry scan for `scene.json`, `Scene::from_slice`,
//! `SceneModel::resolve`, `load_assets` (all asset-JSON parses + the post-load
//! re-resolve). Not covered: texture decode and shader translation — both run
//! inside `SceneRenderer::new` against the still-open pkg. Shader translation
//! stays warm through kirie-shader's own content-addressed `.kirie-cache`
//! (Phase 3.4 note in `load.rs`).

use std::path::Path;
use std::time::Instant;

use kirie_bake::{BundleContent, Cache};
use kirie_scene::SceneModel;

/// Domain tag + descriptor-layout version. Bump the trailing byte whenever the
/// descriptor encoding below changes so every existing bundle keys to a
/// different directory (mirrors kirie-bake's no-migration §V8 stance).
/// `\x02`: property values dropped from the key — the bundle stores the
/// defaults-resolved model and the loader reresolves.
const DESCRIPTOR_TAG: &[u8] = b"kirie-scene-bundle-src\x02";

/// Append a length-prefixed byte string (u32 LE length).
fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&u32::try_from(bytes.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Compose the canonical bundle-source descriptor for a workshop scene — the
/// opaque `source` bytes kirie-bake keys on. See the module docs for what each
/// component pins and why. Deliberately property-independent: one bundle
/// serves every override combination via [`SceneModel::reresolve`].
pub(crate) fn bundle_source(
    pkg_bytes: &[u8],
    project_bytes: Option<&[u8]>,
    assets_dir: Option<&Path>,
) -> Vec<u8> {
    let mut src = Vec::with_capacity(128);
    src.extend_from_slice(DESCRIPTOR_TAG);
    src.extend_from_slice(blake3::hash(pkg_bytes).as_bytes());
    match project_bytes {
        Some(bytes) => {
            src.push(1);
            src.extend_from_slice(blake3::hash(bytes).as_bytes());
        }
        None => src.push(0),
    }
    match assets_dir {
        Some(dir) => {
            src.push(1);
            push_bytes(&mut src, dir.as_os_str().as_encoded_bytes());
        }
        None => src.push(0),
    }
    src
}

/// Try to satisfy a scene load from the bundle cache. Returns the resolved
/// model on a validated hit (mmap + blake3 + rkyv bytecheck all inside
/// [`Cache::load`]); `None` on a miss or any cache problem. An undecodable
/// scene payload inside an otherwise-valid bundle evicts it (self-healing, so
/// the next load rebakes) — a cache failure never fails the load.
pub(crate) fn try_load_model(cache: &Cache, source: &[u8]) -> Option<SceneModel> {
    let start = Instant::now();
    match cache.load(source) {
        Ok(Some(bundle)) => match bundle.scene_model() {
            Ok(model) => {
                tracing::info!(
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    size_bytes = bundle.size_bytes(),
                    "scene bundle hit; skipping parse/resolve/asset load"
                );
                Some(model)
            }
            Err(e) => {
                tracing::warn!(error = %e, "baked scene payload undecodable; evicting bundle");
                let _ = cache.remove(source);
                None
            }
        },
        Ok(None) => None,
        Err(e) => {
            tracing::debug!(error = %e, "bundle cache unavailable; loading directly");
            None
        }
    }
}

/// Bake the freshly-built resolved model into the cache (the miss path),
/// inline and best-effort: any failure is logged and swallowed, never
/// propagated into the load.
pub(crate) fn store_model(cache: &Cache, source: &[u8], model: &SceneModel) {
    let start = Instant::now();
    let mut content = BundleContent::new();
    if let Err(e) = content.set_scene_model(model) {
        tracing::warn!(error = %e, "scene model not serializable; skipping bake");
        return;
    }
    match cache.bake(source, content) {
        Ok(path) => tracing::info!(
            elapsed_ms = start.elapsed().as_millis() as u64,
            path = %path.display(),
            "scene bundle baked"
        ),
        Err(e) => tracing::warn!(error = %e, "scene bundle bake failed"),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use kirie_scene::{PropertyBag, PropertyValue, Scene};

    use super::*;

    /// A unique scratch directory removed on drop (no tempfile dep; same
    /// pattern as kirie-bake's integration tests).
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let p =
                std::env::temp_dir().join(format!("kirie-render-bundle-{}-{tag}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            TmpDir(p)
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A synthetic scene: one image object whose visibility is property-bound
    /// and whose model/material asset JSON comes from an in-memory source — so
    /// the baked model exercises resolution AND filled `resolved` asset slots.
    fn synthetic_scene_json() -> &'static [u8] {
        br#"{
            "camera": { "eye": "0 0 1", "center": "0 0 0", "up": "0 1 0" },
            "general": { "clearcolor": "0.1 0.2 0.3" },
            "objects": [
                {
                    "image": "models/test.json",
                    "origin": "1 2 3",
                    "visible": { "value": true, "user": "show" }
                }
            ]
        }"#
    }

    /// In-memory asset source for the synthetic scene.
    fn asset_source(path: &str) -> Option<Vec<u8>> {
        match path {
            "models/test.json" => Some(br#"{ "material": "materials/test.json" }"#.to_vec()),
            "materials/test.json" => Some(
                br#"{ "passes": [ { "shader": "genericimage2", "blending": "translucent" } ] }"#.to_vec(),
            ),
            _ => None,
        }
    }

    /// Build the model the direct way: parse → resolve → load_assets.
    fn build_direct(bag: &PropertyBag) -> SceneModel {
        let scene = Scene::from_slice(synthetic_scene_json()).expect("synthetic scene parses");
        let mut model = SceneModel::resolve(scene, bag);
        let problems = model.load_assets(&asset_source, bag);
        assert!(problems.is_empty(), "synthetic assets all resolve: {problems:?}");
        model
    }

    /// Determinism guard: a model loaded via the bundle cache is IDENTICAL to
    /// the directly-built one — structurally (PartialEq) and as serialized
    /// JSON — including collapsed property bindings and filled asset slots.
    #[test]
    fn bundle_roundtrip_equals_direct_load() {
        let tmp = TmpDir::new("roundtrip");
        let cache = Cache::with_root(&tmp.0);

        let mut bag = PropertyBag::new();
        bag.insert("show", PropertyValue::Bool(false));
        let _props = [("show".to_owned(), PropertyValue::Bool(false))];

        let direct = build_direct(&bag);
        // Sanity: resolution actually collapsed the binding (value true→false)
        // and the asset slots were filled — otherwise the guard proves nothing.
        assert!(!direct.scene.objects[0].base.visible.value, "binding resolved");
        match &direct.scene.objects[0].kind {
            kirie_scene::object::ObjectKind::Image(img) => {
                assert!(img.model.is_some(), "model file loaded");
                assert!(img.material.is_some(), "material loaded");
            }
            other => panic!("expected image object, got {other:?}"),
        }

        let source = bundle_source(b"pkg-bytes", Some(b"project-bytes"), None);
        store_model(&cache, &source, &direct);
        let baked = try_load_model(&cache, &source).expect("bundle hit after bake");

        assert_eq!(baked, direct, "bundle round-trip is structurally identical");
        assert_eq!(
            serde_json::to_value(&baked).unwrap(),
            serde_json::to_value(&direct).unwrap(),
            "bundle round-trip is JSON-identical"
        );
    }

    /// THE determinism guard for the property-independent key: a model baked
    /// against the DEFAULTS bag and then `reresolve`d against the override bag
    /// is identical to one resolved directly against the override bag — so one
    /// bundle can serve every property combination.
    #[test]
    fn defaults_bake_plus_reresolve_equals_direct() {
        let tmp = TmpDir::new("reresolve");
        let cache = Cache::with_root(&tmp.0);

        // The real loader's bag always carries every project-declared default;
        // mirror that (a bag MISSING a property leaves the current value, so
        // an empty bag cannot restore anything).
        let mut defaults = PropertyBag::new();
        defaults.insert("show", PropertyValue::Bool(true));
        let baked_model = build_direct(&defaults);
        assert!(baked_model.scene.objects[0].base.visible.value, "default visible");
        let source = bundle_source(b"pkg", Some(b"proj"), None);
        store_model(&cache, &source, &baked_model);

        // Load the same bundle and apply an override.
        let mut overridden = PropertyBag::new();
        overridden.insert("show", PropertyValue::Bool(false));
        let mut from_bundle = try_load_model(&cache, &source).expect("hit");
        from_bundle.reresolve(&overridden);

        let direct = build_direct(&overridden);
        assert_eq!(from_bundle, direct, "defaults-bake + reresolve == direct resolve");
        assert!(
            !from_bundle.scene.objects[0].base.visible.value,
            "override applied"
        );

        // And back again — bindings survive any number of re-resolutions.
        let mut back = from_bundle;
        back.reresolve(&defaults);
        assert_eq!(back, baked_model, "reresolve to defaults round-trips");
    }

    /// The builtin-assets dir identity (presence + path) is part of the key:
    /// `load_assets` may read asset JSON from it, so a different (or missing)
    /// assets dir must never serve the other's resolved model.
    #[test]
    fn assets_dir_identity_is_part_of_the_key() {
        let none = bundle_source(b"p", None, None);
        let a = bundle_source(b"p", None, Some(Path::new("/opt/we/assets")));
        let b = bundle_source(b"p", None, Some(Path::new("/mnt/we/assets")));
        assert_ne!(none, a);
        assert_ne!(a, b);
    }

    /// pkg / project.json content changes flow into the descriptor.
    #[test]
    fn source_content_is_part_of_the_key() {
        assert_ne!(
            bundle_source(b"pkg-1", Some(b"proj"), None),
            bundle_source(b"pkg-2", Some(b"proj"), None)
        );
        assert_ne!(
            bundle_source(b"pkg", Some(b"proj-1"), None),
            bundle_source(b"pkg", Some(b"proj-2"), None)
        );
        assert_ne!(
            bundle_source(b"pkg", Some(b"proj"), None),
            bundle_source(b"pkg", None, None)
        );
    }
}
