//! SceneScript integration for the scene renderer (docs/scripting-api.md §3/§5;
//! SPEC.md §V3).
//!
//! A scene's inline property scripts (`"script"` bindings on object leaf
//! fields, docs/format-scene-json.md §3.1) are driven here. Following SPEC.md
//! §V3, the QuickJS runtime lives on its own thread inside
//! [`kirie_script::ScriptEngine`]; this host never touches JS memory. Each frame
//! it marshals an immutable [`HostFrame`] snapshot *in* and reads typed
//! [`kirie_script::SceneOp`]s + property results *out*, translating them into
//! [`PropUpdate`]s the render loop applies to its live objects.
//!
//! # Scope (v1 slice)
//!
//! Only the object-leaf properties that flow into the per-object builtin
//! uniforms (`g_Alpha`, `g_Brightness`, `g_Color`) and object visibility, plus
//! text content, are wired back. Transform ops (`origin`/`scale`/`angles`),
//! camera transforms, particle rate and `createLayer`/`sortLayer` are collected
//! but not yet applied — they require a geometry/scene rebuild the 2D
//! compositor does not do per-frame (tracked as a gap, see the crate report).
//! An input pointer position from the platform is not delivered to the
//! [`kirie_platform::Renderer`] trait yet (`render(dt)` only), so scripts read a
//! centered pointer until that hook exists (T26).

use std::collections::BTreeMap;

use kirie_audio::AudioSpectrum;
use kirie_scene::object::ObjectKind;
use kirie_scene::user::ScriptBinding;
use kirie_scene::{PropertyValue, SceneModel};
use kirie_script::{AudioBuffers, HostFrame, LayerState, SceneOp, SceneState, ScriptEngine, ScriptValue};
use serde_json::{Map, Value};

/// Which live render value a scripted property (or a script `SetProperty` op)
/// drives. Only these flow into the per-object builtin uniforms / visibility /
/// text this compositor tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropTarget {
    /// `g_Alpha` — object alpha.
    Alpha,
    /// `g_Brightness` — object brightness.
    Brightness,
    /// `g_Color` / `g_Color4` — object color.
    Color,
    /// Object visibility (drops the object's draw when false).
    Visible,
    /// Text object content.
    Text,
}

impl PropTarget {
    /// Map a scene-object leaf field name to a target, if we drive it.
    fn from_field(name: &str) -> Option<Self> {
        Some(match name {
            "alpha" => Self::Alpha,
            "brightness" => Self::Brightness,
            "color" => Self::Color,
            "visible" => Self::Visible,
            "text" => Self::Text,
            _ => return None,
        })
    }
}

/// One property script bound to a scene object leaf, keyed by its module key.
struct ScriptedProp {
    /// `"<field>_<objectId>"` module key (matches [`kirie_script`] results).
    key: String,
    /// The owning object id.
    object_id: i64,
    /// The live value it drives.
    target: PropTarget,
}

/// A typed property update the render loop applies to its live objects
/// (docs/scripting-api.md §5.1 / §8).
#[derive(Clone, Debug, PartialEq)]
pub struct PropUpdate {
    /// Target object id.
    pub object_id: i64,
    /// Which live value to write.
    pub target: PropTarget,
    /// The new value (as returned by the script).
    pub value: ScriptValue,
}

/// The per-scene script host: the engine plus the frame snapshot it re-marshals
/// each tick.
pub struct ScriptHost {
    engine: ScriptEngine,
    props: Vec<ScriptedProp>,
    /// Every scene object as a `thisScene` layer, in render order, kept updated
    /// as ops/results apply so next frame's `thisLayer`/`getLayer` reads reflect
    /// the mutations (docs §4.1/§6.2). A script that iterates `getLayerCount()`
    /// and toggles group visibility (the layer-switcher pattern) needs the whole
    /// scene here, not just the scripted leaves.
    layers: Vec<LayerState>,
    scene: SceneState,
    /// `engine.userProperties` — resolved project-property values (combo/slider/
    /// bool/…) the scripts branch on (docs §6.1). Populated from the property bag
    /// so `mode_combo`/`style_*`/… drive the same visibility the oracle shows.
    user_props: BTreeMap<String, ScriptValue>,
    res: [f32; 2],
    elapsed: f64,
}

impl ScriptHost {
    /// Build a host for `model` if it has any driveable property script;
    /// `None` when the scene has no scripts (the common case) or the engine
    /// fails to start (SPEC.md §V9: scripts are best-effort, never fatal).
    #[must_use]
    pub fn build(
        model: &SceneModel,
        res: (u32, u32),
        user_props: &[(String, PropertyValue)],
    ) -> Option<Self> {
        // Collect every driveable scripted leaf first (source + initial value +
        // flattened props), so an empty set skips the QuickJS thread spawn.
        // Every object is also snapshotted as a `thisScene` layer (in render
        // order) so scripts can enumerate/toggle layers they do not directly
        // drive (docs §6.2 `getLayer`/`getLayerCount`).
        let mut pending: Vec<Pending> = Vec::new();
        let mut layers: Vec<LayerState> = Vec::with_capacity(model.scene.objects.len());
        for object in &model.scene.objects {
            let id = object.base.id;
            layers.push(layer_state(object));
            match &object.kind {
                ObjectKind::Image(img) => {
                    collect(&mut pending, id, "alpha", &img.alpha.script, || {
                        ScriptValue::Float(f64::from(img.alpha.value))
                    });
                    collect(&mut pending, id, "brightness", &img.brightness.script, || {
                        ScriptValue::Float(f64::from(img.brightness.value))
                    });
                    collect(&mut pending, id, "color", &img.color.script, || {
                        color_value(img.color.value)
                    });
                    collect(&mut pending, id, "visible", &img.visible.script, || {
                        ScriptValue::Bool(img.visible.value)
                    });
                }
                ObjectKind::Text(txt) => {
                    collect(&mut pending, id, "text", &txt.text.script, || {
                        ScriptValue::Str(txt.text.value.clone())
                    });
                    collect(&mut pending, id, "alpha", &txt.alpha.script, || {
                        ScriptValue::Float(f64::from(txt.alpha.value))
                    });
                    collect(&mut pending, id, "color", &txt.color.script, || {
                        color_value(txt.color.value)
                    });
                    collect(&mut pending, id, "visible", &txt.visible.script, || {
                        ScriptValue::Bool(txt.visible.value)
                    });
                }
                _ => {}
            }
        }

        if pending.is_empty() {
            return None;
        }

        let engine = match ScriptEngine::new() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "script engine failed to start; scene runs without scripts");
                return None;
            }
        };

        let mut props = Vec::with_capacity(pending.len());
        for p in pending {
            match engine.load_property_script(
                p.prop.key.clone(),
                p.source,
                Some(p.prop.object_id),
                p.initial,
                p.script_props,
            ) {
                Ok(()) => props.push(p.prop),
                Err(e) => {
                    tracing::warn!(key = %p.prop.key, error = %e, "property script failed to load; skipped");
                }
            }
        }

        if props.is_empty() {
            return None;
        }
        // One-shot per scene load (not per frame): INFO like the audio-capture
        // startup line, so a live run shows the script surface came up.
        tracing::info!(scripts = props.len(), "scene script host started");

        Some(ScriptHost {
            engine,
            props,
            layers,
            scene: scene_state(model),
            user_props: user_props
                .iter()
                .map(|(k, v)| (k.clone(), prop_to_script(v)))
                .collect(),
            res: [res.0 as f32, res.1 as f32],
            elapsed: 0.0,
        })
    }

    /// Advance one frame: marshal a [`HostFrame`], tick the engine, and return
    /// the typed property updates to apply. `dt` is seconds since last frame;
    /// `audio` is the latest mono spectrum. All three band reductions are
    /// marshalled so `registerAudioBuffers(16|32|64)` each read their matching
    /// array (docs/scripting-api.md §6.1). Never blocks the render thread beyond
    /// the bounded channel round-trip (SPEC.md §V3/§V4) and never panics on a
    /// throwing script (§V9).
    pub fn tick(&mut self, dt: f32, audio: Option<&AudioSpectrum>) -> Vec<PropUpdate> {
        self.elapsed += f64::from(dt);
        let frame = HostFrame {
            runtime: self.elapsed,
            frametime: f64::from(dt),
            now: self.elapsed * 1000.0,
            res_x: f64::from(self.res[0]),
            res_y: f64::from(self.res[1]),
            // No platform pointer at render() yet (T26): a centered pointer.
            pointer_screen: [0.5, 0.5],
            audio: audio.map(|s| AudioBuffers {
                audio16: s.audio16.to_vec(),
                audio32: s.audio32.to_vec(),
                audio64: s.audio64.to_vec(),
            }),
            scene: self.scene.clone(),
            layers: self.layers.clone(),
            user_props: self.user_props.clone(),
            ..HostFrame::default()
        };

        let output = match self.engine.tick(frame, Vec::new()) {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!(error = %e, "script tick failed; leaving properties unchanged");
                return Vec::new();
            }
        };
        for err in &output.errors {
            tracing::debug!(error = %err, "script runtime error (script stays loaded, V9)");
        }
        for log in &output.logs {
            if log.error {
                tracing::debug!(target: "kirie_script::console", "{}", log.message);
            } else {
                tracing::trace!(target: "kirie_script::console", "{}", log.message);
            }
        }

        let mut updates = Vec::new();
        // Property-script results (docs §5.1: update()'s return drives the prop).
        for (key, value) in output.property_results {
            if let Some((object_id, target)) = self
                .props
                .iter()
                .find(|p| p.key == key)
                .map(|p| (p.object_id, p.target))
            {
                self.record_layer(object_id, target, &value);
                updates.push(PropUpdate {
                    object_id,
                    target,
                    value,
                });
            }
        }
        // Imperative scene ops (docs §6.2/§8): only the leaf writes we drive.
        for op in output.ops {
            if let SceneOp::SetProperty {
                layer_id,
                name,
                value,
            } = op
                && let Some(target) = PropTarget::from_field(&name)
            {
                self.record_layer(layer_id, target, &value);
                updates.push(PropUpdate {
                    object_id: layer_id,
                    target,
                    value,
                });
            }
            // Transform / camera / createLayer / sortLayer ops are collected by
            // the engine but not applied by the 2D compositor yet (see module
            // docs); dropping them keeps the scene stable rather than diverging.
        }
        updates
    }

    /// Keep the cached layer snapshot in step with an applied value so next
    /// frame's `thisLayer` reads reflect it (docs §4.1).
    fn record_layer(&mut self, id: i64, target: PropTarget, value: &ScriptValue) {
        let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) else {
            return;
        };
        match target {
            PropTarget::Alpha => {
                if let Some(a) = as_f32(value) {
                    layer.alpha = Some(a);
                }
            }
            PropTarget::Color => {
                if let Some(c) = as_rgb(value) {
                    layer.color = Some(c);
                }
            }
            PropTarget::Visible => {
                if let ScriptValue::Bool(b) = value {
                    layer.visible = Some(*b);
                }
            }
            PropTarget::Text => {
                if let ScriptValue::Str(s) = value {
                    layer.text = Some(s.clone());
                }
            }
            // Brightness is not a registered `thisLayer` property (docs §4.1),
            // so there is nothing to mirror.
            PropTarget::Brightness => {}
        }
    }
}

/// Flatten a `scriptproperties` map to the effective values the engine's
/// `createScriptProperties().finish()` reads: each entry may be a
/// `{ "value": ... }` user setting (docs/scripting-api.md §5.5).
fn flatten_props(props: &Map<String, Value>) -> Value {
    let mut out = Map::new();
    for (k, v) in props {
        let val = match v {
            Value::Object(o) => o.get("value").cloned().unwrap_or_else(|| v.clone()),
            other => other.clone(),
        };
        out.insert(k.clone(), val);
    }
    Value::Object(out)
}

/// A scripted leaf staged before the engine is spawned.
struct Pending {
    prop: ScriptedProp,
    source: String,
    initial: ScriptValue,
    script_props: Value,
}

/// Stage a scripted leaf field if it carries a `"script"` binding we drive.
/// `initial` is evaluated lazily so unscripted fields cost nothing.
fn collect(
    out: &mut Vec<Pending>,
    id: i64,
    field: &str,
    binding: &Option<ScriptBinding>,
    initial: impl FnOnce() -> ScriptValue,
) {
    let (Some(target), Some(b)) = (PropTarget::from_field(field), binding.as_ref()) else {
        return;
    };
    out.push(Pending {
        prop: ScriptedProp {
            key: format!("{field}_{id}"),
            object_id: id,
            target,
        },
        source: b.source.clone(),
        initial: initial(),
        script_props: flatten_props(&b.properties),
    });
}

/// Snapshot an object's registered `thisLayer` properties (docs §4.1).
fn layer_state(object: &kirie_scene::object::Object) -> LayerState {
    let base = &object.base;
    let mut ls = LayerState {
        id: base.id,
        name: base.name.clone(),
        parent: base.parent,
        origin: Some(base.origin.value),
        scale: Some(base.scale.value),
        angles: Some(base.angles.value),
        visible: Some(base.visible.value),
        ..LayerState::default()
    };
    match &object.kind {
        ObjectKind::Image(img) => {
            ls.color = Some([img.color.value[0], img.color.value[1], img.color.value[2]]);
            ls.alpha = Some(img.alpha.value);
            ls.visible = Some(img.visible.value && base.visible.value);
        }
        ObjectKind::Text(txt) => {
            ls.color = Some([txt.color.value[0], txt.color.value[1], txt.color.value[2]]);
            ls.alpha = Some(txt.alpha.value);
            ls.visible = Some(txt.visible.value && base.visible.value);
            ls.point_size = Some(txt.pointsize.value);
            ls.text = Some(txt.text.value.clone());
        }
        _ => {}
    }
    ls
}

/// Snapshot scene-level read-only members (docs §6.2 `thisScene`).
fn scene_state(model: &SceneModel) -> SceneState {
    let g = &model.scene.general;
    SceneState {
        clearcolor: [
            g.clearcolor.value[0],
            g.clearcolor.value[1],
            g.clearcolor.value[2],
        ],
        ambientcolor: [
            g.ambientcolor.value[0],
            g.ambientcolor.value[1],
            g.ambientcolor.value[2],
        ],
        skylightcolor: [
            g.skylightcolor.value[0],
            g.skylightcolor.value[1],
            g.skylightcolor.value[2],
        ],
        bloom: g.bloom.value,
        // SceneState reports these as ints (docs §6.2 `thisScene.bloomstrength`).
        bloomstrength: g.bloomstrength.value as i64,
        bloomthreshold: g.bloomthreshold.value as i64,
        ..SceneState::default()
    }
}

/// A scene `Color` ([r,g,b,a]) as the `Vec3` a color script expects.
fn color_value(c: [f32; 4]) -> ScriptValue {
    ScriptValue::Vec3([c[0], c[1], c[2]])
}

/// A resolved project property as the `engine.userProperties.<name>` value a
/// script reads (docs §6.1): combos/text as strings (`==`-compared),
/// sliders as numbers, bools as bools, colors as an RGB vec3.
fn prop_to_script(v: &PropertyValue) -> ScriptValue {
    match v {
        PropertyValue::Bool(b) => ScriptValue::Bool(*b),
        PropertyValue::Number(n) => ScriptValue::Float(*n),
        PropertyValue::Color([r, g, b, _]) => ScriptValue::Vec3([*r, *g, *b]),
        PropertyValue::Combo(s) | PropertyValue::Text(s) => ScriptValue::Str(s.clone()),
    }
}

/// Coerce a script value to a scalar (alpha/brightness).
pub fn as_f32(v: &ScriptValue) -> Option<f32> {
    match v {
        ScriptValue::Float(f) => Some(*f as f32),
        ScriptValue::Int(i) => Some(*i as f32),
        ScriptValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Coerce a script value to an RGB triple (color).
pub fn as_rgb(v: &ScriptValue) -> Option<[f32; 3]> {
    match v {
        ScriptValue::Vec3(c) => Some(*c),
        ScriptValue::Vec4(c) => Some([c[0], c[1], c[2]]),
        ScriptValue::Vec2(c) => Some([c[0], c[1], 0.0]),
        ScriptValue::Float(f) => Some([*f as f32; 3]),
        _ => None,
    }
}
