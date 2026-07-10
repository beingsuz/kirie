//! End-to-end tests: real JS through the embedded QuickJS runtime.
//! docs/scripting-api.md is the behavior oracle.

use kirie_script::{AudioBuffers, HostFrame, LayerState, SceneOp, ScriptEngine, ScriptValue};

fn num(v: &ScriptValue) -> f64 {
    match v {
        ScriptValue::Int(i) => *i as f64,
        ScriptValue::Float(f) => *f,
        other => panic!("expected number, got {other:?}"),
    }
}

// ---- builtins: Vec/Mat math (docs §9 fixed + §10) -------------------------

#[test]
fn vec_math_correct_and_operand_order_fixed() {
    let e = ScriptEngine::new().unwrap();
    // add is commutative; subtract/divide/cross/mix use fixed operand order.
    assert_eq!(
        e.eval("new Vec3(5,5,5).subtract(new Vec3(1,2,3)).x").unwrap(),
        "4"
    ); // this - v
    assert_eq!(e.eval("new Vec2(10,10).divide(new Vec2(2,5)).y").unwrap(), "2"); // this / v
    assert_eq!(e.eval("new Vec3(1,0,0).cross(new Vec3(0,1,0)).z").unwrap(), "1"); // this × v
    assert_eq!(
        e.eval("new Vec3(0,0,0).mix(new Vec3(10,10,10), 0.5).x").unwrap(),
        "5"
    );
    assert_eq!(e.eval("new Vec2(3,4).length()").unwrap(), "5");
    assert_eq!(e.eval("new Vec2(3,4).lengthSqr()").unwrap(), "25"); // not aliased to length
    assert_eq!(
        e.eval("new Vec3(1,2,3).add(new Vec3(1,1,1)).toString()").unwrap(),
        "2.000000, 3.000000, 4.000000"
    );
}

#[test]
fn mat4_transform_and_compose() {
    let e = ScriptEngine::new().unwrap();
    assert_eq!(
        e.eval("Mat4.fromTranslation(new Vec3(5,6,7)).transformPoint(new Vec3(0,0,0)).x")
            .unwrap(),
        "5"
    );
    assert_eq!(
        e.eval("Mat4.fromScale(2).transformPoint(new Vec3(3,0,0)).x")
            .unwrap(),
        "6"
    );
    // 90° about Z maps +X to +Y.
    assert_eq!(
        e.eval("Math.round(Mat4.fromRotation(90, new Vec3(0,0,1)).transformDirection(new Vec3(1,0,0)).y)")
            .unwrap(),
        "1"
    );
}

// ---- console + localStorage (docs §6.5 / §10.3) ---------------------------

#[test]
fn console_and_localstorage() {
    let e = ScriptEngine::new().unwrap();
    // localStorage round-trips; missing key => null.
    assert_eq!(
        e.eval("localStorage.set('k','v'); localStorage.get('k')")
            .unwrap(),
        "v"
    );
    assert_eq!(e.eval("localStorage.get('missing')").unwrap(), "null");
    assert_eq!(
        e.eval("localStorage.set('n', 42); localStorage.get('n')")
            .unwrap(),
        "42"
    );
    // MediaPlaybackEvent constants present.
    assert_eq!(e.eval("MediaPlaybackEvent.PLAYBACK_PLAYING").unwrap(), "1");
}

// ---- property script contract (docs §5.1) ---------------------------------

#[test]
fn update_return_applied_to_property() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "text_1",
        "export function update(value){ return 'hello ' + value; }",
        None,
        ScriptValue::Str("world".into()),
        serde_json::json!({}),
    )
    .unwrap();
    let out = e.tick(HostFrame::default(), vec![]).unwrap();
    assert_eq!(out.property_results.len(), 1);
    assert_eq!(out.property_results[0].0, "text_1");
    assert_eq!(out.property_results[0].1, ScriptValue::Str("hello world".into()));
    assert!(out.errors.is_empty());
}

#[test]
fn init_runs_once_before_update() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "alpha_2",
        "let n = 0; export function init(v){ n = 100; } export function update(v){ n += 1; return n; }",
        None,
        ScriptValue::Int(0),
        serde_json::json!({}),
    )
    .unwrap();
    let a = e.tick(HostFrame::default(), vec![]).unwrap();
    let b = e.tick(HostFrame::default(), vec![]).unwrap();
    assert_eq!(num(&a.property_results[0].1), 101.0); // init(100) then +1
    assert_eq!(num(&b.property_results[0].1), 102.0); // init did NOT run again
}

#[test]
fn script_properties_from_json_only() {
    // docs §5.5: createScriptProperties descriptors are ignored; values come
    // from JSON scriptproperties. `==` string/number coercion (corpus).
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "text_9",
        "export var scriptProperties = createScriptProperties().addCombo({name:'monthFormat',value:'99'}).finish();\
         export function update(v){ return (scriptProperties.monthFormat == 1) ? 'numeric' : 'other'; }",
        None,
        ScriptValue::Str(String::new()),
        serde_json::json!({ "monthFormat": "1" }),
    )
    .unwrap();
    let out = e.tick(HostFrame::default(), vec![]).unwrap();
    assert_eq!(out.property_results[0].1, ScriptValue::Str("numeric".into()));
}

// ---- V9: throwing script yields a typed error, never a panic --------------

#[test]
fn throwing_update_is_typed_error_not_panic() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "color_3",
        "export function update(v){ throw new Error('boom'); }",
        None,
        ScriptValue::Int(0),
        serde_json::json!({}),
    )
    .unwrap();
    let out = e.tick(HostFrame::default(), vec![]).unwrap();
    assert!(out.property_results.is_empty(), "write-back skipped on throw");
    assert_eq!(out.errors.len(), 1);
    assert!(matches!(
        &out.errors[0],
        kirie_script::ScriptError::Runtime { phase: "update", .. }
    ));
    // Engine still alive.
    assert_eq!(e.eval("1+1").unwrap(), "2");
}

#[test]
fn malformed_source_is_load_error_not_panic() {
    let e = ScriptEngine::new().unwrap();
    let r = e.load_property_script(
        "broken_4",
        "export function update(v { this is not valid",
        None,
        ScriptValue::Null,
        serde_json::json!({}),
    );
    assert!(matches!(r, Err(kirie_script::ScriptError::Load { .. })));
    // A dropped script does not tick.
    let out = e.tick(HostFrame::default(), vec![]).unwrap();
    assert!(out.property_results.is_empty());
}

// ---- events: applyUserProperties (docs §5.3) ------------------------------

#[test]
fn apply_user_properties_fires_on_every_module() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "rate_5",
        "export function applyUserProperties(changed){ if ('foo' in changed) console.log('upd:' + changed.foo); }\
         export function update(v){ return v; }",
        None,
        ScriptValue::Int(0),
        serde_json::json!({}),
    )
    .unwrap();
    let out = e.dispatch_user_property("foo", ScriptValue::Int(7)).unwrap();
    assert!(
        out.logs.iter().any(|l| l.message == "upd:7"),
        "logs: {:?}",
        out.logs
    );
}

// ---- importable modules (docs §6.6) ---------------------------------------

#[test]
fn we_modules_import_and_compute() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "a_6",
        "import * as WEMath from 'WEMath';\
         import * as WEColor from 'WEColor';\
         export function update(v){ return WEMath.mix(0, 10, 0.5) + WEColor.hsv2rgb(new Vec3(0,1,1)).x; }",
        None,
        ScriptValue::Int(0),
        serde_json::json!({}),
    )
    .unwrap();
    let out = e.tick(HostFrame::default(), vec![]).unwrap();
    assert_eq!(num(&out.property_results[0].1), 6.0); // 5 + red.x(1)
}

// ---- thisLayer writes become typed scene ops (docs §8) --------------------

#[test]
fn this_layer_write_records_scene_op() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "visible_42",
        "export function update(v){ thisLayer.visible = false; return v; }",
        Some(42),
        ScriptValue::Bool(true),
        serde_json::json!({}),
    )
    .unwrap();
    let frame = HostFrame {
        layers: vec![LayerState {
            id: 42,
            name: "L".into(),
            visible: Some(true),
            ..Default::default()
        }],
        ..Default::default()
    };
    let out = e.tick(frame, vec![]).unwrap();
    assert!(
        out.ops.iter().any(|op| matches!(op,
        SceneOp::SetProperty { layer_id: 42, name, value: ScriptValue::Bool(false) } if name == "visible")),
        "ops: {:?}",
        out.ops
    );
}

// ---- timers (docs §5.4, canceller bug fixed) ------------------------------

#[test]
fn engine_interval_fires_by_frame_clock() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "alpha_7",
        "export function update(v){ if (v === 0) { engine.setInterval(function(){ console.log('fire'); }, 100); } return 1; }",
        None,
        ScriptValue::Int(0),
        serde_json::json!({}),
    )
    .unwrap();
    let t1 = e
        .tick(
            HostFrame {
                now: 0.0,
                ..Default::default()
            },
            vec![],
        )
        .unwrap();
    assert!(!t1.logs.iter().any(|l| l.message == "fire"));
    // current value fed back = 1, so no re-register; now past 100ms → fires.
    let t2 = e
        .tick(
            HostFrame {
                now: 200.0,
                ..Default::default()
            },
            vec![],
        )
        .unwrap();
    assert!(t2.logs.iter().any(|l| l.message == "fire"), "logs: {:?}", t2.logs);
}

// ---- text-layer scripts (docs §7) -----------------------------------------

#[test]
fn text_layer_script_ticks_and_reads_text() {
    let e = ScriptEngine::new().unwrap();
    let h = e
        .create_layer_script(
            "'use strict';\nexport function update(value){ return 'T:' + Math.floor(thisScene.time); }",
            serde_json::json!({}),
            "placeholder",
        )
        .unwrap();
    assert!(h > 0);
    e.tick_layer(h, 5.0, 0.016, 60.0).unwrap();
    assert_eq!(e.layer_text(h).unwrap(), "T:5");
    e.destroy_layer(h).unwrap();
    assert_eq!(e.layer_text(h).unwrap(), ""); // invalid handle after destroy
}

// ---- audio buffers (docs §6.1) --------------------------------------------

/// `engine.registerAudioBuffers(res).average` must read the *matching* audioN
/// reduction — the 16-band getter returns `audio16`, not the first 16 entries
/// of `audio64` (docs/scripting-api.md §6.1). Distinct fill values per
/// resolution prove the correct array is selected.
#[test]
fn register_audio_buffers_reads_matching_resolution() {
    let e = ScriptEngine::new().unwrap();
    // Sum the whole requested buffer so the count *and* the source array matter.
    for (res, key) in [(16, "a16"), (32, "a32"), (64, "a64")] {
        e.load_property_script(
            format!("alpha_{res}"),
            format!(
                "export function update(v){{ var a = engine.registerAudioBuffers({res}).average; \
                 var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length * 1000 + s; }}"
            ),
            None,
            ScriptValue::Float(0.0),
            serde_json::json!({}),
        )
        .unwrap();
        let _ = key;
    }
    let audio = AudioBuffers {
        audio16: vec![0.5; 16],  // sum 8, len 16 → 16008
        audio32: vec![0.25; 32], // sum 8, len 32 → 32008
        audio64: vec![0.1; 64],  // sum 6.4, len 64 → 64006.4
    };
    let out = e
        .tick(
            HostFrame {
                audio: Some(audio),
                ..Default::default()
            },
            vec![],
        )
        .unwrap();
    let get = |k: &str| -> f64 {
        num(&out
            .property_results
            .iter()
            .find(|(key, _)| key == k)
            .expect("result")
            .1)
    };
    assert!(
        (get("alpha_16") - 16_008.0).abs() < 1e-3,
        "16-band read wrong array"
    );
    assert!(
        (get("alpha_32") - 32_008.0).abs() < 1e-3,
        "32-band read wrong array"
    );
    assert!(
        (get("alpha_64") - 64_006.4).abs() < 1e-2,
        "64-band read wrong array"
    );
}

/// Missing audio (`None`) yields a zero-filled buffer of the requested length,
/// never a crash (V9).
#[test]
fn register_audio_buffers_silent_is_zeroed() {
    let e = ScriptEngine::new().unwrap();
    e.load_property_script(
        "alpha_1",
        "export function update(v){ var a = engine.registerAudioBuffers(32).average; \
         var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length * 1000 + s; }",
        None,
        ScriptValue::Float(0.0),
        serde_json::json!({}),
    )
    .unwrap();
    let out = e.tick(HostFrame::default(), vec![]).unwrap();
    assert_eq!(
        num(&out.property_results[0].1),
        32_000.0,
        "silent 32-band all zeros"
    );
}

// ---- version surface ------------------------------------------------------

#[test]
fn version_constants_exposed() {
    assert_eq!(kirie_script::API_VERSION, "2.8");
    assert_eq!(kirie_script::TRANSLATOR_VERSION, 1);
}
