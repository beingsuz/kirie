//! Corpus-gated: load a real workshop scene script and tick it end-to-end.
//!
//! docs/scripting-api.md §2: ~8 of the 19 scene items ship `export function
//! update`. This extracts a script directly from a `scene.pkg`, loads it into
//! the engine, and asserts `update()` runs and returns a value — proving the
//! module + `createScriptProperties` + `Date` surface works on real content.
//! Skips (does not fail) when the corpus is not installed.

use kirie_script::{HostFrame, ScriptEngine, ScriptValue};
use serde_json::Value;

const CORPUS: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";

/// Recursively find the first user-setting object carrying a `"script"` whose
/// source contains `export function update`, returning `(source, scriptprops)`.
fn find_script(v: &Value) -> Option<(String, Value)> {
    if let Value::Object(map) = v {
        if let Some(Value::String(src)) = map.get("script")
            && src.contains("export function update")
        {
            let props = map
                .get("scriptproperties")
                .cloned()
                .unwrap_or(Value::Object(Default::default()));
            return Some((src.clone(), props));
        }
        for child in map.values() {
            if let Some(r) = find_script(child) {
                return Some(r);
            }
        }
    } else if let Value::Array(a) = v {
        for child in a {
            if let Some(r) = find_script(child) {
                return Some(r);
            }
        }
    }
    None
}

/// Flatten a `scriptproperties` map to the effective JSON values the engine sees
/// (each entry may be a `{ "value": ... }` user setting).
fn flatten_props(props: &Value) -> Value {
    let mut out = serde_json::Map::new();
    if let Value::Object(map) = props {
        for (k, v) in map {
            let val = match v {
                Value::Object(o) => o.get("value").cloned().unwrap_or(v.clone()),
                other => other.clone(),
            };
            out.insert(k.clone(), val);
        }
    }
    Value::Object(out)
}

#[test]
fn corpus_script_ticks_end_to_end() {
    let root = std::path::Path::new(CORPUS);
    if !root.exists() {
        eprintln!("corpus not installed at {CORPUS}; skipping");
        return;
    }

    let mut found = None;
    for entry in std::fs::read_dir(root).unwrap().flatten() {
        let pkg = entry.path().join("scene.pkg");
        if !pkg.exists() {
            continue;
        }
        let Ok(package) = kirie_formats::pkg::OwnedPkg::from_path(&pkg) else {
            continue;
        };
        let Ok(bytes) = package.read_name(b"scene.json") else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<Value>(bytes) else {
            continue;
        };
        if let Some((src, props)) = find_script(&json) {
            found = Some((entry.file_name().to_string_lossy().into_owned(), src, props));
            break;
        }
    }

    let Some((id, src, props)) = found else {
        eprintln!("no scripted corpus scene found; skipping");
        return;
    };
    eprintln!("loading script from corpus item {id}");

    let engine = ScriptEngine::new().unwrap();
    engine
        .load_property_script(
            "text_0",
            src,
            None,
            ScriptValue::Str(String::new()),
            flatten_props(&props),
        )
        .expect("real corpus script must compile");

    // Tick a few frames; a throwing script would surface as a typed error, not
    // a panic (SPEC.md §V9).
    let mut applied = None;
    for f in 0..3 {
        let out = engine
            .tick(
                HostFrame {
                    runtime: f as f64,
                    now: f as f64 * 16.0,
                    ..Default::default()
                },
                vec![],
            )
            .unwrap();
        assert!(out.errors.is_empty(), "corpus script errored: {:?}", out.errors);
        if let Some((_, v)) = out.property_results.first() {
            applied = Some(v.clone());
        }
    }

    // Scripted corpus properties are clock/date text (string) or color/audio
    // cycles (vector/number); any non-null applied value proves update() ran and
    // its return marshalled back.
    match applied {
        Some(ScriptValue::Str(s)) => assert!(!s.is_empty(), "expected non-empty text"),
        Some(ScriptValue::Null) | None => panic!("update() produced no applied value"),
        Some(_) => {} // numeric / vector result is fine
    }
}
