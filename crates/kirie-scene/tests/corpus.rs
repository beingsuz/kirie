//! Corpus gate (SPEC.md §V11): load + fully resolve the `scene.json` of every
//! `scene.pkg` in the workshop corpus, assert the docs/format-scene-json.md
//! object-count tables, no parse errors, and that every referenced material /
//! effect / model / particle file resolves.

use std::collections::BTreeMap;
use std::path::PathBuf;

use kirie_formats::pkg::OwnedPkg;
use kirie_formats::project::Project;
use kirie_scene::object::ObjectKind;
use kirie_scene::{AssetSource, PropertyBag, Scene, SceneModel};

/// Default corpus location (SPEC.md §C); override with `KIRIE_CORPUS`.
const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var("KIRIE_CORPUS").map_or_else(|_| PathBuf::from(CORPUS_DIR), PathBuf::from);
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skipping corpus test: {} not present", dir.display());
        None
    }
}

/// Default WE shared-assets directory (the reference locator's last fallback:
/// pkg → project dir → shared `assets/`). Override with `KIRIE_WE_ASSETS`.
const WE_ASSETS_DIR: &str = "/home/aiko/.local/share/Steam/steamapps/common/wallpaper_engine/assets";

fn we_assets_dir() -> Option<PathBuf> {
    let dir = std::env::var("KIRIE_WE_ASSETS").map_or_else(|_| PathBuf::from(WE_ASSETS_DIR), PathBuf::from);
    dir.is_dir().then_some(dir)
}

/// An [`AssetSource`] resolving `scene.pkg` entries first, then the WE shared
/// `assets/` directory (builtin models/materials like `util/fullscreenlayer`).
struct PkgSource {
    pkg: OwnedPkg,
    assets: Option<PathBuf>,
}

impl AssetSource for PkgSource {
    fn load(&self, path: &str) -> Option<Vec<u8>> {
        if let Ok(bytes) = self.pkg.read_name(path.as_bytes()) {
            return Some(bytes.to_vec());
        }
        let assets = self.assets.as_ref()?;
        std::fs::read(assets.join(path)).ok()
    }
}

/// The 19 scene.pkg items documented in docs/format-scene-json.md's corpus
/// tables. Exact per-kind object counts are asserted only over this snapshot;
/// newly subscribed scenes are parsed + resolved but excluded from the tallies.
const DOC_SCENE_IDS: &[&str] = &[
    "1388331347",
    "1627026721",
    "2082653325",
    "2085292947",
    "2155933185",
    "2395163768",
    "3047596375",
    "3118949804",
    "3293156956",
    "3421423611",
    "3428443753",
    "3445942378",
    "3576956643",
    "3585875739",
    "3587565260",
    "3609007632",
    "3611478368",
    "3631634316",
    "3738467344",
];

/// The dispatch-kind tag for count aggregation (docs/format-scene-json.md §7).
fn kind_tag(kind: &ObjectKind) -> &'static str {
    match kind {
        ObjectKind::Image(_) => "image",
        ObjectKind::Sound(_) => "sound",
        ObjectKind::Particle(_) => "particle",
        ObjectKind::Text(_) => "text",
        ObjectKind::Model(_) => "model",
        ObjectKind::Light(_) => "light",
        ObjectKind::Shape(_) => "shape",
        ObjectKind::Group => "group",
    }
}

#[test]
fn corpus_load_resolve_and_count() {
    let Some(dir) = corpus_dir() else { return };

    let assets = we_assets_dir();
    if assets.is_none() {
        eprintln!("note: WE shared assets dir absent — builtin util/* assets may not resolve");
    }

    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut scene_count = 0;
    let mut problems_total = 0;

    for entry in std::fs::read_dir(&dir).expect("read corpus dir") {
        let item = entry.expect("dir entry").path();
        let pkg_path = item.join("scene.pkg");
        if !pkg_path.is_file() {
            continue;
        }
        let id = item.file_name().unwrap().to_string_lossy().into_owned();
        // Live corpus grows; the exact per-kind object tallies below describe the
        // documented 19-scene snapshot. Newly subscribed scenes are still fully
        // parsed + resolved (§V9/asset invariants apply to them), but excluded
        // from the snapshot counts so a new subscription can't break the gate.
        let documented = DOC_SCENE_IDS.contains(&id.as_str());

        let pkg = OwnedPkg::from_path(&pkg_path).expect("open scene.pkg");
        let scene_bytes = pkg
            .read_name(b"scene.json")
            .expect("scene.json entry present")
            .to_vec();

        // §V9: parse must succeed with no panic on real data.
        let scene =
            Scene::from_slice(&scene_bytes).unwrap_or_else(|e| panic!("scene {id} parse failed: {e}"));
        if documented {
            scene_count += 1;
            for object in &scene.objects {
                *counts.entry(kind_tag(&object.kind)).or_default() += 1;
            }
        }

        // Build the property bag from the item's project.json (§3.4).
        let project = Project::from_path(item.join("project.json")).expect("project.json parses");
        let bag = PropertyBag::from_project(&project);

        // Resolve bindings, then load + resolve every referenced asset file.
        let mut model = SceneModel::resolve(scene, &bag);
        let source = PkgSource {
            pkg,
            assets: assets.clone(),
        };
        let problems = model.load_assets(&source, &bag);
        assert!(problems.is_empty(), "scene {id}: unresolved assets: {problems:?}");
        problems_total += problems.len();

        // Every image material and every effect pass material must have loaded.
        for object in &model.scene.objects {
            if let ObjectKind::Image(img) = &object.kind {
                assert!(
                    img.material.is_some(),
                    "scene {id} obj {}: image material did not resolve",
                    object.base.id
                );
                for effect in &img.effects {
                    let file = effect
                        .resolved
                        .as_ref()
                        .unwrap_or_else(|| panic!("scene {id}: effect {} did not resolve", effect.file));
                    for (i, pass) in file.passes.iter().enumerate() {
                        if pass.material.is_some() {
                            assert!(
                                pass.resolved.is_some(),
                                "scene {id} effect {} pass {i}: material did not resolve",
                                effect.file
                            );
                        }
                    }
                }
            }
        }

        // §V13: the resolved snapshot round-trips through serde (bake format).
        let json = serde_json::to_value(&model).expect("serialize model");
        let back: SceneModel = serde_json::from_value(json).expect("deserialize model");
        assert_eq!(model, back, "scene {id}: serde round-trip mismatch");
    }

    // docs/format-scene-json.md corpus tables: 19 scenes, 166 objects.
    assert_eq!(scene_count, 19, "expected 19 corpus scenes");
    assert_eq!(counts.get("image").copied().unwrap_or(0), 87, "image count");
    assert_eq!(counts.get("particle").copied().unwrap_or(0), 58, "particle count");
    assert_eq!(counts.get("sound").copied().unwrap_or(0), 7, "sound count");
    assert_eq!(counts.get("text").copied().unwrap_or(0), 7, "text count");
    assert_eq!(counts.get("model").copied().unwrap_or(0), 1, "model count");
    assert_eq!(counts.get("light").copied().unwrap_or(0), 1, "light count");
    assert_eq!(counts.get("shape").copied().unwrap_or(0), 1, "shape count");
    assert_eq!(counts.get("group").copied().unwrap_or(0), 4, "group count");
    assert_eq!(counts.values().sum::<usize>(), 166, "total object count");
    assert_eq!(problems_total, 0);
}
