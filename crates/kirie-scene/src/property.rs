//! Resolved project-property values and the bag scene bindings resolve against.
//!
//! Spec: docs/format-scene-json.md §3.2–§3.4. `project.json →
//! general.properties` supplies named values; scene `"user"` bindings connect a
//! field to a property so the property's current value overwrites the field
//! (§3.2). This module turns the typed [`kirie_formats::project`] properties
//! into a flat name→value bag, supports `setProperty` overrides, and defines the
//! [`Resolvable`] conversions each leaf type uses.

use std::collections::BTreeMap;

use kirie_formats::project::{Project, PropertyEntry, PropertyKind};

use crate::value::{Color, DynamicValue, strtof};

/// A concrete resolved property value (docs/format-scene-json.md §3.4 types).
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyValue {
    /// A `bool` property.
    Bool(bool),
    /// A `slider` property (numeric).
    Number(f64),
    /// A `color` property (RGBA, alpha forced to 1.0 by project parsing).
    Color(Color),
    /// A `combo` property — its selected option value (§3.3 compares this).
    Combo(String),
    /// A textual property (`text`/`textinput`/`file`/…).
    Text(String),
}

impl PropertyValue {
    /// The string form used for §3.3 conditional-binding equality.
    pub fn as_condition_string(&self) -> String {
        match self {
            PropertyValue::Bool(b) => {
                if *b {
                    "true".to_owned()
                } else {
                    "false".to_owned()
                }
            }
            PropertyValue::Number(n) => n.to_string(),
            PropertyValue::Color([r, g, b, a]) => format!("{r} {g} {b} {a}"),
            PropertyValue::Combo(s) | PropertyValue::Text(s) => s.clone(),
        }
    }
}

/// A flat map of property name → current resolved value, plus overrides
/// (docs/format-scene-json.md §3.4). Scene fields resolve against this.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PropertyBag {
    values: BTreeMap<String, PropertyValue>,
}

impl PropertyBag {
    /// An empty bag — every bound field falls back to its literal.
    pub fn new() -> Self {
        PropertyBag::default()
    }

    /// Build the bag from a parsed `project.json` (docs/format-scene-json.md
    /// §3.4). Group/unrecognized entries contribute nothing.
    pub fn from_project(project: &Project) -> Self {
        let mut values = BTreeMap::new();
        for (name, entry) in &project.general.properties {
            if let PropertyEntry::Property(p) = entry {
                let value = match &p.kind {
                    PropertyKind::Bool { value } => PropertyValue::Bool(*value),
                    PropertyKind::Slider { value, .. } => PropertyValue::Number(f64::from(*value)),
                    PropertyKind::Color { value: [r, g, b] } => PropertyValue::Color([*r, *g, *b, 1.0]),
                    PropertyKind::Combo { value, .. } => PropertyValue::Combo(value.clone()),
                    PropertyKind::Text => PropertyValue::Text(p.text.clone()),
                    PropertyKind::TextInput { value }
                    | PropertyKind::UserShortcut { value }
                    | PropertyKind::File { value }
                    | PropertyKind::Directory { value }
                    | PropertyKind::SceneTexture { value } => PropertyValue::Text(value.clone()),
                };
                values.insert(name.clone(), value);
            }
        }
        PropertyBag { values }
    }

    /// The current value of a property, if declared.
    pub fn get(&self, name: &str) -> Option<&PropertyValue> {
        self.values.get(name)
    }

    /// Apply a `setProperty` override (docs/format-scene-json.md §3.2; SPEC.md
    /// §I setProperty). Only overrides existing keys — the reference rejects
    /// keys the wallpaper does not declare (returns whether it applied).
    pub fn set(&mut self, name: &str, value: PropertyValue) -> bool {
        if let Some(slot) = self.values.get_mut(name) {
            *slot = value;
            true
        } else {
            false
        }
    }

    /// Insert or replace a property value unconditionally (for tests / synthetic
    /// bags where no `project.json` exists).
    pub fn insert(&mut self, name: impl Into<String>, value: PropertyValue) {
        self.values.insert(name.into(), value);
    }

    /// Parse a raw `setProperty` / `--set-property` string into `name`'s declared
    /// type and set it (docs/format-scene-json.md §3.2/§3.4). The target type is
    /// taken from the current value, so a color stays a color, a slider a number,
    /// etc.; unknown keys and unparseable values are rejected (returns whether it
    /// applied). Mirrors the reference `setProperty` typing.
    pub fn set_from_str(&mut self, name: &str, raw: &str) -> bool {
        let Some(current) = self.values.get(name) else {
            return false;
        };
        let parsed = match current {
            PropertyValue::Bool(_) => {
                PropertyValue::Bool(matches!(raw.trim(), "1" | "true" | "True" | "TRUE"))
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
            // Combos compare by their selected option string (§3.3); text verbatim.
            PropertyValue::Combo(_) => PropertyValue::Combo(raw.trim().to_owned()),
            PropertyValue::Text(_) => PropertyValue::Text(raw.to_owned()),
        };
        self.set(name, parsed)
    }

    /// Number of declared properties.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the bag is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// A leaf type that a bound property value can be converted into
/// (docs/format-scene-json.md §2.4 conversions, §3.3 conditional → bool).
pub trait Resolvable: Sized {
    /// Convert a property value to this type.
    fn from_property(value: &PropertyValue) -> Self;
    /// Convert a §3.3 conditional result (a bool) to this type.
    fn from_bool(b: bool) -> Self;
}

/// PropertyValue → f32 (`.x`/`.r` for aggregates, string via `strtof`).
fn prop_f32(v: &PropertyValue) -> f32 {
    match v {
        PropertyValue::Bool(b) => f32::from(*b),
        PropertyValue::Number(n) => *n as f32,
        PropertyValue::Color([r, ..]) => *r,
        PropertyValue::Combo(s) | PropertyValue::Text(s) => strtof(s),
    }
}

impl Resolvable for bool {
    fn from_property(value: &PropertyValue) -> Self {
        match value {
            PropertyValue::Bool(b) => *b,
            PropertyValue::Combo(s) | PropertyValue::Text(s) => {
                matches!(s.as_str(), "1" | "true" | "True" | "TRUE")
            }
            other => prop_f32(other) != 0.0,
        }
    }
    fn from_bool(b: bool) -> Self {
        b
    }
}

impl Resolvable for f32 {
    fn from_property(value: &PropertyValue) -> Self {
        prop_f32(value)
    }
    fn from_bool(b: bool) -> Self {
        f32::from(b)
    }
}

impl Resolvable for i64 {
    fn from_property(value: &PropertyValue) -> Self {
        prop_f32(value) as i64
    }
    fn from_bool(b: bool) -> Self {
        i64::from(b)
    }
}

impl Resolvable for String {
    fn from_property(value: &PropertyValue) -> Self {
        value.as_condition_string()
    }
    fn from_bool(b: bool) -> Self {
        b.to_string()
    }
}

impl Resolvable for [f32; 2] {
    fn from_property(value: &PropertyValue) -> Self {
        match value {
            PropertyValue::Color([r, g, ..]) => [*r, *g],
            other => {
                let x = prop_f32(other);
                [x, x]
            }
        }
    }
    fn from_bool(b: bool) -> Self {
        let x = f32::from(b);
        [x, x]
    }
}

impl Resolvable for [f32; 3] {
    fn from_property(value: &PropertyValue) -> Self {
        match value {
            PropertyValue::Color([r, g, b, _]) => [*r, *g, *b],
            other => {
                let x = prop_f32(other);
                [x, x, x]
            }
        }
    }
    fn from_bool(b: bool) -> Self {
        let x = f32::from(b);
        [x, x, x]
    }
}

impl Resolvable for [f32; 4] {
    fn from_property(value: &PropertyValue) -> Self {
        match value {
            PropertyValue::Color(c) => *c,
            other => {
                let x = prop_f32(other);
                [x, x, x, 1.0]
            }
        }
    }
    fn from_bool(b: bool) -> Self {
        let x = f32::from(b);
        [x, x, x, 1.0]
    }
}

impl Resolvable for DynamicValue {
    fn from_property(value: &PropertyValue) -> Self {
        match value {
            PropertyValue::Bool(b) => DynamicValue::Bool(*b),
            PropertyValue::Number(n) => DynamicValue::Float(*n),
            PropertyValue::Color(c) => DynamicValue::Color(*c),
            PropertyValue::Combo(s) | PropertyValue::Text(s) => DynamicValue::Str(s.clone()),
        }
    }
    fn from_bool(b: bool) -> Self {
        DynamicValue::Bool(b)
    }
}
