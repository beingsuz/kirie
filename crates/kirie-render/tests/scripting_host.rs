//! SceneScript host integration (docs/scripting-api.md §3/§5; SPEC.md §V3).
//!
//! GPU-free: builds a resolved [`SceneModel`] from an inline scene.json with a
//! scripted property, drives [`ScriptHost`] over several ticks, and asserts the
//! property value evolves — the "a scripted scene's property changes over ticks"
//! gate. Also asserts a script-free scene spawns no engine.

use kirie_render::scene::scripting::{PropTarget, ScriptHost};
use kirie_scene::{PropertyBag, Scene, SceneModel};

/// Resolve an inline scene.json into a [`SceneModel`] (no assets loaded — the
/// host only reads property/script bindings).
fn model(json: &str) -> SceneModel {
    let scene = Scene::from_slice(json.as_bytes()).expect("parse scene.json");
    SceneModel::resolve(scene, &PropertyBag::default())
}

#[test]
fn scripted_alpha_changes_over_ticks() {
    // An image object whose `alpha` is script-driven: `update` returns the
    // engine runtime, so the applied value grows every frame (docs §5.1).
    let json = r#"{
        "camera": { "eye": "0 0 100", "center": "0 0 0", "up": "0 1 0" },
        "general": { "orthogonalprojection": { "width": 128, "height": 128 } },
        "objects": [
            {
                "id": 7,
                "name": "layer",
                "image": "models/x.json",
                "alpha": {
                    "value": 1.0,
                    "script": "export function update(v) { return engine.runtime; }"
                }
            }
        ]
    }"#;
    let model = model(json);
    let mut host = ScriptHost::build(&model, (128, 128), &[]).expect("scene has a driveable script");

    let mut last = -1.0_f32;
    let mut saw_update = false;
    for _ in 0..4 {
        let updates = host.tick(0.5, None);
        for u in updates {
            if u.object_id == 7 && u.target == PropTarget::Alpha {
                let v = kirie_render::scene::scripting::as_f32(&u.value).expect("alpha is a scalar");
                assert!(v > last, "scripted alpha must increase each tick: {v} !> {last}");
                last = v;
                saw_update = true;
            }
        }
    }
    assert!(saw_update, "the alpha script produced no property update");
    assert!(
        last > 0.5,
        "runtime-driven alpha advanced past the first frame: {last}"
    );
}

#[test]
fn scene_without_scripts_spawns_no_host() {
    let json = r#"{
        "camera": { "eye": "0 0 100", "center": "0 0 0", "up": "0 1 0" },
        "general": { "orthogonalprojection": { "width": 64, "height": 64 } },
        "objects": [
            { "id": 1, "name": "plain", "image": "models/x.json", "alpha": { "value": 0.5 } }
        ]
    }"#;
    let model = model(json);
    assert!(
        ScriptHost::build(&model, (64, 64), &[]).is_none(),
        "no script binding ⇒ no engine thread (V9 best-effort)"
    );
}

#[test]
fn throwing_script_does_not_panic_and_leaves_value_alone() {
    // A script that throws inside update surfaces as a typed error, never a
    // panic; the tick returns no update for it (SPEC.md §V9).
    let json = r#"{
        "camera": { "eye": "0 0 100", "center": "0 0 0", "up": "0 1 0" },
        "general": { "orthogonalprojection": { "width": 64, "height": 64 } },
        "objects": [
            {
                "id": 3,
                "name": "boom",
                "image": "models/x.json",
                "alpha": {
                    "value": 1.0,
                    "script": "export function update(v) { throw new Error('boom'); }"
                }
            }
        ]
    }"#;
    let model = model(json);
    let mut host = ScriptHost::build(&model, (64, 64), &[]).expect("script loads even if it throws at tick");
    // Several ticks, no panic; a throwing update yields no applied value.
    for _ in 0..3 {
        let updates = host.tick(0.016, None);
        assert!(
            updates.iter().all(|u| u.object_id != 3),
            "a throwing update must not apply a value"
        );
    }
}
