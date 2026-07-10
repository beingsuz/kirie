//! Synthetic scenes exercising every object kind, value encoding, the user
//! binding/resolution machinery, and the serde round-trip contract (SPEC.md
//! §V13). Behavior citations are to docs/format-scene-json.md.

use kirie_scene::material::{Blending, CullMode, DepthMode};
use kirie_scene::object::ObjectKind;
use kirie_scene::property::PropertyValue;
use kirie_scene::scene::Projection;
use kirie_scene::value::{DynamicValue, parse_color, parse_vec};
use kirie_scene::{PropertyBag, Scene, SceneModel};
use serde_json::{Value, json};

/// Wrap objects in a minimal valid scene root (§4: camera/general/objects).
fn scene_with(objects: Value) -> Value {
    json!({
        "camera": { "eye": "0 0 0", "center": "0 0 -1", "up": "0 1 0" },
        "general": {},
        "objects": objects,
    })
}

fn parse(objects: Value) -> Scene {
    Scene::from_value(&scene_with(objects)).expect("scene parses")
}

/// Assert the serde form round-trips (SPEC.md §V13, the bake/snapshot format).
fn round_trip(scene: &Scene) {
    let v = serde_json::to_value(scene).expect("serialize");
    let back: Scene = serde_json::from_value(v).expect("deserialize");
    assert_eq!(scene, &back, "serde round-trip must be lossless");
}

// ---- §2 value encodings ----------------------------------------------------

#[test]
fn vec_parsing_strict_count() {
    assert_eq!(parse_vec::<3>("1 2 3").unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(parse_vec::<2>("0.85968 0.84985").unwrap(), [0.85968, 0.84985]);
    // §2.1: wrong component count is a load error.
    assert!(parse_vec::<3>("1 2").is_err());
    assert!(parse_vec::<3>("1 2 3 4").is_err());
}

#[test]
fn color_float_vs_int_path() {
    // §2.2: a `.` anywhere ⇒ float path.
    assert_eq!(
        parse_color("0.3 0.3 0.3", 1.0, false).unwrap(),
        [0.3, 0.3, 0.3, 1.0]
    );
    // §2.2: no `.` ⇒ int-0..255 path (scene-side).
    let c = parse_color("255 128 0", 1.0, false).unwrap();
    assert_eq!(c, [1.0, 128.0 / 255.0, 0.0, 1.0]);
    // §2.2 pitfall: "1 1 1" as ints ≈ black.
    assert_eq!(parse_color("1 1 1", 1.0, false).unwrap()[0], 1.0 / 255.0);
    // force_float disables that.
    assert_eq!(parse_color("1 1 1", 1.0, true).unwrap(), [1.0, 1.0, 1.0, 1.0]);
    // §2.2: 4-component carries its own alpha.
    assert_eq!(
        parse_color("0.1 0.2 0.3 0.4", 1.0, false).unwrap(),
        [0.1, 0.2, 0.3, 0.4]
    );
}

#[test]
fn color_hex_forms() {
    // §2.2: 3-digit expands with the alpha byte.
    assert_eq!(parse_color("#fff", 1.0, false).unwrap(), [1.0, 1.0, 1.0, 1.0]);
    // clean-impl deviation: 6-digit gains ff alpha.
    assert_eq!(parse_color("#ff0000", 1.0, false).unwrap(), [1.0, 0.0, 0.0, 1.0]);
    assert!(parse_color("#zz", 1.0, false).is_err());
}

#[test]
fn dynamic_value_decode() {
    // §2.4: single token that parses as float → Float.
    assert_eq!(
        DynamicValue::decode(&json!("1.5"), false),
        DynamicValue::Float(1.5)
    );
    // §2.4: single non-float token → Str.
    assert_eq!(
        DynamicValue::decode(&json!("hello"), false),
        DynamicValue::Str("hello".into())
    );
    // §2.4: 3 tokens → Vec.
    assert_eq!(
        DynamicValue::decode(&json!("1 2 3"), false),
        DynamicValue::Vec(vec![1.0, 2.0, 3.0])
    );
    // §2.4: color expected → Color.
    assert_eq!(
        DynamicValue::decode(&json!("0.5 0.5 0.5"), true),
        DynamicValue::Color([0.5, 0.5, 0.5, 1.0])
    );
    assert_eq!(
        DynamicValue::decode(&json!(true), false),
        DynamicValue::Bool(true)
    );
    assert_eq!(DynamicValue::decode(&json!(7), false), DynamicValue::Int(7));
}

// ---- §7 base fields + dispatch --------------------------------------------

#[test]
fn base_defaults_and_salvage() {
    let s = parse(json!([{ "particle": "p/x.json" }]));
    let o = &s.objects[0];
    // §7.1: id salvages to -1, name to "unknown".
    assert_eq!(o.base.id, -1);
    assert_eq!(o.base.name, "unknown");
    assert_eq!(o.base.scale.value, [1.0, 1.0, 1.0]);
    assert!(o.base.visible.value);
    round_trip(&s);
}

#[test]
fn numeric_name_stringified() {
    // §7.1: a numeric name is stringified (particle objects in the wild).
    let s = parse(json!([{ "particle": "p.json", "name": 42, "id": 3 }]));
    assert_eq!(s.objects[0].base.name, "42");
}

#[test]
fn dispatch_image_null_falls_through_to_particle() {
    // §7 note: `image: null` + `particle` ⇒ particle (is_string guard).
    let s = parse(json!([{ "id": 97, "name": "Sakura", "image": null, "model": null,
                           "particle": "particles/presets/leaves5.json" }]));
    assert!(matches!(s.objects[0].kind, ObjectKind::Particle(_)));
}

#[test]
fn all_kinds_dispatch() {
    let s = parse(json!([
        { "id": 1, "name": "img", "image": "models/cave.json" },
        { "id": 2, "name": "snd", "sound": ["sounds/a.mp3"], "playbackmode": "loop" },
        { "id": 3, "name": "par", "particle": "particles/x.json" },
        { "id": 4, "name": "txt", "text": "hi" },
        { "id": 5, "name": "mdl", "model": "models/x.mdl" },
        { "id": 6, "name": "lgt", "light": true, "radius": 100 },
        { "id": 7, "name": "shp", "shape": "sphere" },
        { "id": 8, "name": "grp", "solid": true },
    ]));
    let kinds: Vec<_> = s.objects.iter().map(|o| &o.kind).collect();
    assert!(matches!(kinds[0], ObjectKind::Image(_)));
    assert!(matches!(kinds[1], ObjectKind::Sound(_)));
    assert!(matches!(kinds[2], ObjectKind::Particle(_)));
    assert!(matches!(kinds[3], ObjectKind::Text(_)));
    assert!(matches!(kinds[4], ObjectKind::Model(_)));
    assert!(matches!(kinds[5], ObjectKind::Light(_)));
    assert!(matches!(kinds[6], ObjectKind::Shape(_)));
    assert!(matches!(kinds[7], ObjectKind::Group));
    round_trip(&s);
}

// ---- §8 image + §10 material + §11 effects --------------------------------

#[test]
fn image_fields_and_alignment() {
    let s = parse(json!([{
        "id": 12, "name": "cave", "image": "models/cave.json",
        "size": "1920 1080", "parallaxDepth": "1 1", "colorBlendMode": 2,
        "brightness": 0.8, "alpha": 0.5, "alignment": "left", "horizontalalign": "right",
        "effects": [{ "file": "effects/waterwaves/effect.json", "visible": true,
                      "passes": [{ "combos": { "NOISE": 1 },
                                   "textures": [null, "masks/m.tex"],
                                   "constantshadervalues": { "speed": 2.5 } }] }]
    }]));
    let ObjectKind::Image(img) = &s.objects[0].kind else {
        panic!()
    };
    assert_eq!(img.size, [1920.0, 1080.0]);
    assert_eq!(img.color_blend_mode.value, 2);
    // §8: horizontalalign wins over alignment.
    assert_eq!(img.alignment, "right");
    assert_eq!(img.effects.len(), 1);
    let pass = &img.effects[0].passes[0];
    assert_eq!(pass.combos.get("NOISE"), Some(&1));
    // §10.2: null slot preserved, index advances.
    assert_eq!(pass.textures, vec![None, Some("masks/m.tex".to_owned())]);
    round_trip(&s);
}

#[test]
fn material_pass_enums() {
    use kirie_scene::material::Material;
    let m = Material::from_value(&json!({
        "passes": [{ "blending": "translucent", "cullmode": "normal",
                     "depthtest": "enabled", "depthwrite": "disabled",
                     "shader": "genericimage2", "textures": ["cave", null],
                     "combos": { "VERTICAL": 1 } }]
    }));
    let p = &m.passes[0];
    assert_eq!(p.blending, Blending::Translucent);
    assert_eq!(p.cullmode, CullMode::Normal);
    assert_eq!(p.depthtest, DepthMode::Enabled);
    assert_eq!(p.depthwrite, DepthMode::Disabled);
    assert_eq!(p.shader, "genericimage2");
    assert_eq!(p.textures, vec![Some("cave".to_owned()), None]);
    // §17.4: unknown enum → default, not fatal.
    let m2 = Material::from_value(&json!({ "passes": [{ "blending": "bogus", "shader": "x" }] }));
    assert_eq!(m2.passes[0].blending, Blending::Normal);
}

// ---- §13 text / §12 sound / §14 particle ----------------------------------

#[test]
fn text_defaults() {
    let s = parse(json!([{ "id": 1, "name": "t", "text": { "value": "12:00",
        "script": "return time();", "scriptproperties": { "p": 1 } },
        "font": "fonts/VCR.ttf", "pointsize": 48, "verticalalign": "top" }]));
    let ObjectKind::Text(t) = &s.objects[0].kind else {
        panic!()
    };
    assert_eq!(t.text.value, "12:00");
    assert!(t.text.script.is_some());
    assert_eq!(t.pointsize.value, 48.0);
    assert_eq!(t.verticalalign, "top");
    assert_eq!(t.horizontalalign, "center"); // default
    round_trip(&s);
}

#[test]
fn particle_instanceoverride_and_emitter() {
    let s = parse(json!([{
        "id": 97, "name": "leaves", "particle": {
            "material": "materials/presets/leaves.json", "maxcount": 500,
            "emitter": [{ "name": "sphererandom", "distancemax": 25, "rate": 20, "id": 10 }],
            "initializer": [{ "name": "colorrandom", "min": "255 255 255", "max": "255 192 248" }],
            "operator": [{ "name": "movement", "gravity": "0 -10 0" }]
        },
        "instanceoverride": { "count": 0.2, "lifetime": 0.25, "speed": 0.2 }
    }]));
    let ObjectKind::Particle(p) = &s.objects[0].kind else {
        panic!()
    };
    assert_eq!(p.system.maxcount, 500);
    // §14.3: number broadcast to all vec3 components.
    assert_eq!(p.system.emitters[0].distancemax, [25.0, 25.0, 25.0]);
    assert_eq!(p.system.emitters[0].name, "sphererandom");
    assert_eq!(p.instanceoverride.count.value, 0.2);
    // §14.6: empty renderer array ⇒ one default sprite renderer.
    assert_eq!(p.system.renderers.len(), 1);
    assert_eq!(p.system.renderers[0].name, "sprite");
    round_trip(&s);
}

// ---- §5 general / §6 camera -----------------------------------------------

#[test]
fn general_defaults_and_colors() {
    let scene = Scene::from_value(&json!({
        "camera": { "eye": "0 0 0", "center": "0 0 -1", "up": "0 1 0" },
        "general": { "ambientcolor": "0.3 0.3 0.3", "bloom": true, "bloomstrength": 2.0,
                     "clearcolor": "0.7 0.7 0.7", "clearenabled": null },
        "objects": []
    }))
    .unwrap();
    assert_eq!(scene.general.ambientcolor.value, [0.3, 0.3, 0.3, 1.0]);
    assert!(scene.general.bloom.value);
    assert_eq!(scene.general.bloomstrength.value, 2.0);
    // default clearcolor is white but here overridden.
    assert_eq!(scene.general.clearcolor.value, [0.7, 0.7, 0.7, 1.0]);
    round_trip(&scene);
}

#[test]
fn camera_projection_forms() {
    let ortho = Scene::from_value(&json!({
        "camera": { "eye": "84 -248 0", "center": "84 -248 -1", "up": "0 1 0" },
        "general": { "orthogonalprojection": { "width": 1920, "height": 1080 } },
        "objects": []
    }))
    .unwrap();
    assert_eq!(
        ortho.camera.projection,
        Projection::Orthogonal {
            width: 1920,
            height: 1080
        }
    );
    assert_eq!(ortho.camera.eye, [84.0, -248.0, 0.0]);

    // null / missing / {auto:true} ⇒ Auto.
    for general in [
        json!({ "orthogonalprojection": null }),
        json!({}),
        json!({ "orthogonalprojection": { "auto": true } }),
    ] {
        let s = Scene::from_value(&json!({
            "camera": { "eye": "0 0 0", "center": "0 0 -1", "up": "0 1 0" },
            "general": general, "objects": []
        }))
        .unwrap();
        assert_eq!(s.camera.projection, Projection::Auto);
    }
}

#[test]
fn missing_sections_error() {
    // §4: required sections.
    assert!(Scene::from_value(&json!({ "general": {}, "objects": [] })).is_err());
    assert!(
        Scene::from_value(
            &json!({ "camera": { "eye": "0 0 0", "center": "0 0 0", "up": "0 1 0" },
                                       "objects": [] })
        )
        .is_err()
    );
    // §6.1: missing required camera field.
    assert!(
        Scene::from_value(&json!({ "camera": { "eye": "0 0 0" }, "general": {}, "objects": [] })).is_err()
    );
}

// ---- §3 user bindings + resolution ----------------------------------------

#[test]
fn resolve_name_binding_overwrites() {
    // §3.2: a bound property overwrites the literal value.
    let s = parse(json!([{ "id": 1, "name": "i", "image": "m.json",
        "alpha": { "value": 1.0, "user": "opacity" } }]));
    let mut bag = PropertyBag::new();
    bag.insert("opacity", PropertyValue::Number(0.25));
    let model = SceneModel::resolve(s, &bag);
    let ObjectKind::Image(img) = &model.scene.objects[0].kind else {
        panic!()
    };
    assert_eq!(img.alpha.value, 0.25);
}

#[test]
fn resolve_conditional_binding_is_equality() {
    // §3.3: conditional binding ⇒ boolean property == condition.
    let make = || {
        parse(json!([{ "id": 1, "name": "i", "image": "m.json",
        "visible": { "value": true, "user": { "name": "style", "condition": "2" } } }]))
    };

    let mut bag = PropertyBag::new();
    bag.insert("style", PropertyValue::Combo("2".to_owned()));
    let model = SceneModel::resolve(make(), &bag);
    let ObjectKind::Image(img) = &model.scene.objects[0].kind else {
        panic!()
    };
    assert!(img.visible.value);

    let mut bag2 = PropertyBag::new();
    bag2.insert("style", PropertyValue::Combo("1".to_owned()));
    let model2 = SceneModel::resolve(make(), &bag2);
    let ObjectKind::Image(img2) = &model2.scene.objects[0].kind else {
        panic!()
    };
    assert!(!img2.visible.value);
}

#[test]
fn setproperty_only_touches_declared_keys() {
    let mut bag = PropertyBag::new();
    bag.insert("opacity", PropertyValue::Number(1.0));
    assert!(bag.set("opacity", PropertyValue::Number(0.5)));
    assert!(!bag.set("undeclared", PropertyValue::Number(0.5)));
    assert_eq!(bag.get("opacity"), Some(&PropertyValue::Number(0.5)));
}

#[test]
fn unbound_field_keeps_literal_when_property_absent() {
    let s = parse(json!([{ "id": 1, "name": "i", "image": "m.json",
        "alpha": { "value": 0.9, "user": "missing_prop" } }]));
    let model = SceneModel::resolve(s, &PropertyBag::new());
    let ObjectKind::Image(img) = &model.scene.objects[0].kind else {
        panic!()
    };
    assert_eq!(img.alpha.value, 0.9); // literal kept
}
