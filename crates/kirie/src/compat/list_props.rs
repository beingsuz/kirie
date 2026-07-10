//! `-l` / `--list-properties-json` (docs/compat-cli.md §3.8) and the shared
//! property-schema serializer used by the `getproperties` control-socket
//! read-back (docs/compat-socket.md §11).
//!
//! Loads a background's `project.json` and prints (or returns) its user
//! properties as a stable JSON schema: each entry carries
//! `key/type/default/value/min/max/step/options/order/text`, where present for
//! the type. `default` is the value declared in `project.json`; `value` is that
//! default folded with any active override (`--set-property` at launch, or a
//! live `property` socket command). For a plain `--list-properties` with no
//! overrides `default == value`.
//!
//! Source resolution ([`load_source`]) accepts a **workshop directory**, a
//! **`project.json` path**, or a **`scene.pkg` path** (its sibling
//! `project.json` — the properties live beside the package, not inside it;
//! resolve.rs, docs/format-project-json.md §3). The CLI's background token is
//! passed through verbatim when it contains a `/` (resolve.rs), so any of the
//! three forms reaches here.

use std::collections::BTreeMap;
use std::path::Path;

use kirie_formats::project::{ComboOption, Project, PropertyEntry, PropertyKind};
use serde_json::{Map, Value, json};

use crate::compat::args::CompatArgs;

/// Print the property list for the default background (doc §3.8). Returns a
/// short error string on load failure (SPEC V9: typed-ish, no panic).
pub fn run(args: &CompatArgs) -> Result<(), String> {
    let overrides = overrides_map(&args.set_properties);
    let project = match args.default_background.as_deref().and_then(load_source) {
        Some(p) => p,
        None => {
            // No workshop project.json (e.g. a direct image/video file): the
            // C++ would fail to build a wallpaper, but there are simply no
            // user properties to list. Emit the empty forms.
            if args.list_properties_json {
                println!("[]");
            }
            return Ok(());
        }
    };

    let views = property_views(&project, &overrides);
    if args.list_properties_json {
        // Single line, matching the C++ compact array output (doc §3.8).
        println!("{}", Value::Array(views.iter().map(PropView::to_json).collect()));
    } else {
        for v in &views {
            v.print_human();
        }
    }
    Ok(())
}

/// Serialize a source's property schema (with overrides folded into `value`)
/// to a single-line JSON array string, for the `getproperties` socket
/// read-back (docs/compat-socket.md §11). Load failure yields `"[]"` — a valid,
/// byte-clean empty schema.
///
/// `source` is a workshop directory, a `project.json` path, or a `scene.pkg`
/// path (see [`load_source`]); `overrides` is keyed by property name with the
/// raw override string (post-override current value).
pub fn properties_json_string(source: &Path, overrides: &BTreeMap<String, String>) -> String {
    match load_source(source) {
        Some(project) => {
            let array: Vec<Value> = property_views(&project, overrides)
                .iter()
                .map(PropView::to_json)
                .collect();
            Value::Array(array).to_string()
        }
        None => "[]".to_string(),
    }
}

/// Load the `project.json` for a background reference. Accepts:
/// - a **directory** → `<dir>/project.json`,
/// - a **`project.json`** (or any `.json`) file → parsed directly,
/// - a **`scene.pkg`** file → its sibling `<parent>/project.json`.
///
/// Returns `None` for a direct media file, a missing/undecodable manifest, or
/// an unrecognized path (SPEC V9: never panics).
pub fn load_source(source: impl AsRef<Path>) -> Option<Project> {
    let path = source.as_ref();
    if path.is_dir() {
        return Project::from_path(path.join("project.json")).ok();
    }
    if path.is_file() {
        let name = path.file_name().and_then(|n| n.to_str());
        let ext = path.extension().and_then(|e| e.to_str());
        // A `.pkg` (scene.pkg) does not itself hold the manifest — the sibling
        // project.json does (resolve.rs, docs/format-project-json.md §3).
        if ext == Some("pkg") {
            return Project::from_path(path.parent()?.join("project.json")).ok();
        }
        // A `project.json` (or any `.json`) is the manifest itself.
        if name == Some("project.json") || ext == Some("json") {
            return Project::from_path(path).ok();
        }
    }
    None
}

/// The active override map (last-wins), keyed by property name. Shared by the
/// CLI (`--set-property`) and the socket read-back.
fn overrides_map(pairs: &[(String, String)]) -> BTreeMap<String, String> {
    pairs.iter().cloned().collect()
}

/// The real (non-separator) properties, sorted by `order` then key (doc §3.8),
/// each rendered into a [`PropView`] with its override folded in.
fn property_views(project: &Project, overrides: &BTreeMap<String, String>) -> Vec<PropView> {
    let mut props: Vec<(&String, &kirie_formats::project::Property)> = project
        .general
        .properties
        .iter()
        .filter_map(|(k, e)| match e {
            PropertyEntry::Property(p) => Some((k, p)),
            PropertyEntry::Group(_) | PropertyEntry::Unrecognized(_) => None,
        })
        .collect();
    props.sort_by(|a, b| a.1.order.cmp(&b.1.order).then_with(|| a.0.cmp(b.0)));
    props
        .into_iter()
        .map(|(key, prop)| PropView::new(key, prop, overrides.get(key).map(String::as_str)))
        .collect()
}

/// A resolved property row with a typed `default`, an override-folded `value`,
/// and the type-specific extras — the stable JSON/human shape (doc §3.8;
/// docs/compat-socket.md §11).
struct PropView {
    key: String,
    text: String,
    order: i64,
    type_tag: &'static str,
    /// Declared value (`project.json`), typed per `type_tag`.
    default: Value,
    /// Current value = default folded with the active override, same JSON type.
    value: Value,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    options: Option<Vec<Value>>,
}

impl PropView {
    fn new(key: &str, prop: &kirie_formats::project::Property, over: Option<&str>) -> Self {
        let type_tag = prop.kind.type_tag();
        let (default, min, max, step, options) = match &prop.kind {
            PropertyKind::Bool { value } => (json!(value), None, None, None, None),
            PropertyKind::Slider {
                value,
                min,
                max,
                step,
            } => (
                json!(value),
                Some(f64::from(*min)),
                Some(f64::from(*max)),
                Some(f64::from(*step)),
                None,
            ),
            PropertyKind::Color { value } => (json!(format_color(value)), None, None, None, None),
            PropertyKind::Combo { options, value } => {
                (json!(value), None, None, None, Some(combo_options(options)))
            }
            // The `text` kind is a UI separator with no value (doc §3.8).
            PropertyKind::Text => (json!(""), None, None, None, None),
            PropertyKind::TextInput { value }
            | PropertyKind::UserShortcut { value }
            | PropertyKind::File { value }
            | PropertyKind::Directory { value }
            | PropertyKind::SceneTexture { value } => (json!(value), None, None, None, None),
        };
        let value = match over {
            Some(raw) => fold_override(&prop.kind, &default, raw),
            None => default.clone(),
        };
        Self {
            key: key.to_owned(),
            text: prop.text.clone(),
            order: prop.order,
            type_tag,
            default,
            value,
            min,
            max,
            step,
            options,
        }
    }

    /// One JSON object with the stable field set (doc §3.8; socket §11): always
    /// `key/type/order/text/default/value`, plus `min/max/step` (slider) or
    /// `options` (combo) where applicable.
    fn to_json(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("key".into(), json!(self.key));
        obj.insert("type".into(), json!(self.type_tag));
        obj.insert("order".into(), json!(self.order));
        obj.insert("text".into(), json!(self.text));
        obj.insert("default".into(), self.default.clone());
        obj.insert("value".into(), self.value.clone());
        if let (Some(min), Some(max), Some(step)) = (self.min, self.max, self.step) {
            obj.insert("min".into(), json!(min));
            obj.insert("max".into(), json!(max));
            obj.insert("step".into(), json!(step));
        }
        if let Some(options) = &self.options {
            obj.insert("options".into(), Value::Array(options.clone()));
        }
        Value::Object(obj)
    }

    /// Print the human `-l` dump (doc §3.8): `key - type` plus indented details.
    fn print_human(&self) {
        println!("{} - {}", self.key, self.type_tag);
        if !self.text.is_empty() {
            println!("    Text: {}", self.text);
        }
        println!("    Value: {}", scalar_str(&self.value));
        if self.value != self.default {
            println!("    Default: {}", scalar_str(&self.default));
        }
        if let (Some(min), Some(max), Some(step)) = (self.min, self.max, self.step) {
            println!("    Min: {min}  Max: {max}  Step: {step}");
        }
        if let Some(options) = &self.options {
            for o in options {
                if let (Some(label), Some(value)) = (o.get("label"), o.get("value")) {
                    println!("    - {} = {}", scalar_str(label), scalar_str(value));
                }
            }
        }
    }
}

/// Coerce a raw override string into the property's typed JSON value so
/// `value` keeps the same JSON type as `default` (docs/format-project-json.md
/// §8 value coercion, socket §11): bool → `bool`, slider → `number`, everything
/// else keeps the raw string (color triples, combo option values, paths, text).
fn fold_override(kind: &PropertyKind, default: &Value, raw: &str) -> Value {
    match kind {
        PropertyKind::Bool { .. } => {
            let on = raw == "true" || raw.parse::<f64>().map(|n| n != 0.0).unwrap_or(false);
            json!(on)
        }
        PropertyKind::Slider { .. } => match raw.parse::<f64>() {
            Ok(n) => json!(n),
            // Non-numeric override: keep the declared default (no silent 0).
            Err(_) => default.clone(),
        },
        _ => json!(raw),
    }
}

/// Combo `options` array `[{label, value}]` (doc §3.8).
fn combo_options(options: &[ComboOption]) -> Vec<Value> {
    options
        .iter()
        .map(|o| json!({ "label": o.label, "value": o.value }))
        .collect()
}

/// Render a scalar JSON value for the human dump without JSON quoting.
fn scalar_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Format an RGB color triple as space-separated `%f` (6 decimals), the C++
/// color value shape (doc §3.8 `"0.000000 0.000000 0.000000"`).
fn format_color(rgb: &[f32; 3]) -> String {
    format!("{:.6} {:.6} {:.6}", rgb[0], rgb[1], rgb[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn project(props: Value) -> Project {
        Project::from_value(json!({
            "title": "t",
            "file": "scene.json",
            "type": "scene",
            "general": { "properties": props },
        }))
        .expect("valid project")
    }

    #[test]
    fn schema_has_stable_fields_and_typed_default() {
        let p = project(json!({
            "bloom": { "text": "Bloom", "order": 2, "type": "bool", "value": true },
            "fov": {
                "text": "FOV", "order": 1, "type": "slider",
                "value": 45.0, "min": 10.0, "max": 90.0, "step": 1.0
            },
        }));
        let views = property_views(&p, &BTreeMap::new());
        // Sorted by order: fov (1) before bloom (2).
        assert_eq!(views[0].key, "fov");
        let fov = views[0].to_json();
        assert_eq!(fov["type"], json!("slider"));
        assert_eq!(fov["default"], json!(45.0));
        assert_eq!(fov["value"], json!(45.0));
        assert_eq!(fov["min"], json!(10.0));
        assert_eq!(fov["max"], json!(90.0));
        assert_eq!(fov["step"], json!(1.0));
        assert_eq!(fov["order"], json!(1));
        assert_eq!(fov["text"], json!("FOV"));
        let bloom = views[1].to_json();
        assert_eq!(bloom["type"], json!("bool"));
        assert_eq!(bloom["default"], json!(true));
        assert!(bloom.get("min").is_none(), "bool must not carry min");
    }

    #[test]
    fn override_folds_into_value_typed() {
        let p = project(json!({
            "bloom": { "type": "bool", "value": false },
            "fov": { "type": "slider", "value": 45.0, "min": 0.0, "max": 90.0, "step": 1.0 },
            "outline": { "type": "color", "value": "0 0 0" },
        }));
        let mut over = BTreeMap::new();
        over.insert("bloom".to_string(), "1".to_string());
        over.insert("fov".to_string(), "70".to_string());
        over.insert("outline".to_string(), "0.5 0.25 0.75".to_string());
        let by_key: BTreeMap<_, _> = property_views(&p, &over)
            .into_iter()
            .map(|v| (v.key.clone(), v.to_json()))
            .collect();
        // bool default vs override-folded value.
        assert_eq!(by_key["bloom"]["default"], json!(false));
        assert_eq!(by_key["bloom"]["value"], json!(true));
        // slider coerces to number.
        assert_eq!(by_key["fov"]["default"], json!(45.0));
        assert_eq!(by_key["fov"]["value"], json!(70.0));
        // color keeps the raw triple string.
        assert_eq!(by_key["outline"]["default"], json!("0.000000 0.000000 0.000000"));
        assert_eq!(by_key["outline"]["value"], json!("0.5 0.25 0.75"));
    }

    #[test]
    fn json_string_is_single_line_and_empty_on_missing() {
        let p = project(json!({ "a": { "type": "bool", "value": true } }));
        let s = Value::Array(
            property_views(&p, &BTreeMap::new())
                .iter()
                .map(PropView::to_json)
                .collect(),
        )
        .to_string();
        assert!(!s.contains('\n'), "schema JSON must be single-line");
        assert!(s.starts_with('[') && s.ends_with(']'));
        // Missing source → byte-clean empty array.
        assert_eq!(
            properties_json_string(Path::new("/definitely/not/a/thing"), &BTreeMap::new()),
            "[]"
        );
    }

    #[test]
    fn non_numeric_slider_override_keeps_default() {
        let p = project(json!({
            "fov": { "type": "slider", "value": 45.0, "min": 0.0, "max": 90.0, "step": 1.0 }
        }));
        let mut over = BTreeMap::new();
        over.insert("fov".to_string(), "garbage".to_string());
        let v = property_views(&p, &over).remove(0).to_json();
        assert_eq!(v["value"], json!(45.0));
    }
}
