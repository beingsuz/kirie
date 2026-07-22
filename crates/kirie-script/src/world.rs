//! The single-threaded script world: one QuickJS `Runtime` + `Context` per scene
//! (docs/scripting-api.md §1; SPEC.md §V3 — this lives on its own dedicated
//! thread, driven only by [`crate::engine`] commands).
//!
//! All JS handles (module namespaces, timers, text layers, `localStorage`) stay
//! inside the JS heap (`__host.*`), so no Rust-side `Persistent` bookkeeping is
//! needed; Rust keeps only lightweight per-module metadata.

use std::collections::BTreeMap;

use rquickjs::loader::{BuiltinLoader, BuiltinResolver};
use rquickjs::{Array, CatchResultExt, Context, Ctx, Function, Module, Object, Runtime, Value};

use crate::error::ScriptError;
use crate::frame::{HostFrame, LogLine, SceneOp, TickOutput};
use crate::value::{ScriptValue, json_to_js};

const BUILTINS_JS: &str = include_str!("js/builtins.js");
const HOST_JS: &str = include_str!("js/host.js");
const WE_MATH_JS: &str = include_str!("js/we_math.js");
const WE_COLOR_JS: &str = include_str!("js/we_color.js");
const WE_VECTOR_JS: &str = include_str!("js/we_vector.js");

/// Snapshot of one module's tick inputs `(key, owner_id, inited, current value,
/// workshop id)`, cloned out of [`World::modules`] before entering `ctx.with`.
type ModuleTickState = (String, Option<i64>, bool, ScriptValue, Option<String>);

/// Per-module bookkeeping (docs §4.2/§5.1).
struct ModuleMeta {
    /// The layer object this module drives (for `thisLayer` binding); `None`
    /// when the property has no scriptable owner.
    owner_id: Option<i64>,
    /// Whether deferred `init()` has run (first tick only, docs §4.2 step 6).
    inited: bool,
    /// The property's current value (fed as `update`'s argument; updated from
    /// the previous tick's return so scripts see their own writes, docs §5.1).
    current: ScriptValue,
    /// `__workshopId` export, for `createLayer` path resolution (docs §2).
    workshop_id: Option<String>,
}

/// The owned script world. Not `Send` (QuickJS `Runtime` is `!Send`); created and
/// used only on the dedicated script thread.
pub struct World {
    _runtime: Runtime,
    context: Context,
    /// Loaded property-script modules, keyed `"<prop>_<objectId>"`; iteration is
    /// lexicographic (docs §4.1 tick order).
    modules: BTreeMap<String, ModuleMeta>,
    /// Next text-layer handle (docs §7; positive ints, 0 = invalid).
    next_layer_handle: u32,
}

impl World {
    /// Build a fresh world: runtime + context, module loader for the three
    /// importable modules, then the embedded builtins + host bridge.
    pub fn new() -> Result<Self, ScriptError> {
        let runtime = Runtime::new().map_err(|e| ScriptError::Internal(e.to_string()))?;
        let resolver = BuiltinResolver::default()
            .with_module("WEMath")
            .with_module("WEColor")
            .with_module("WEVector");
        let loader = BuiltinLoader::default()
            .with_module("WEMath", WE_MATH_JS)
            .with_module("WEColor", WE_COLOR_JS)
            .with_module("WEVector", WE_VECTOR_JS);
        runtime.set_loader(resolver, loader);
        let context = Context::full(&runtime).map_err(|e| ScriptError::Internal(e.to_string()))?;

        context
            .with(|ctx| -> Result<(), ScriptError> {
                eval_global(&ctx, "<builtins>", BUILTINS_JS)?;
                eval_global(&ctx, "<host>", HOST_JS)?;
                Ok(())
            })
            .map_err(|e| match e {
                ScriptError::Load { message, .. } => ScriptError::Internal(message),
                other => other,
            })?;

        Ok(World {
            _runtime: runtime,
            context,
            modules: BTreeMap::new(),
            next_layer_handle: 0,
        })
    }

    /// Load (compile + evaluate) a property script as an ES module (docs §4.2).
    /// `key` is the module key; `owner_id` the scriptable layer it drives;
    /// `initial` the property's initial value; `script_properties` the JSON
    /// `scriptproperties` bag (docs §5.5, descriptors ignored).
    ///
    /// A compile/eval failure yields [`ScriptError::Load`] and the module is not
    /// registered (SPEC.md §V9 — script disabled, engine unharmed).
    pub fn load_property_script(
        &mut self,
        key: &str,
        source: &str,
        owner_id: Option<i64>,
        initial: ScriptValue,
        script_properties: &serde_json::Value,
    ) -> Result<(), ScriptError> {
        if self.modules.contains_key(key) {
            // docs §4.1: queueScript dedupes by key — first registration wins.
            return Ok(());
        }
        let key_owned = key.to_owned();
        let workshop_id = self.context.with(|ctx| -> Result<Option<String>, ScriptError> {
            // docs §5.5: expose this module's scriptproperties before its body
            // runs, so top-level createScriptProperties().finish() reads them.
            let host: Object = global(&ctx, "__host")?;
            host.set("scriptProps", json_to_js(&ctx, script_properties).internal()?)
                .internal()?;

            let module = Module::declare(ctx.clone(), key_owned.clone(), source)
                .catch(&ctx)
                .map_err(|e| ScriptError::Load {
                    key: key_owned.clone(),
                    message: e.to_string(),
                })?;
            let (module, _promise) = module.eval().catch(&ctx).map_err(|e| ScriptError::Load {
                key: key_owned.clone(),
                message: e.to_string(),
            })?;
            drain_jobs(&ctx);
            let namespace = module.namespace().internal()?;
            // Register the namespace in the JS heap so exports outlive this call.
            let register: Function = global(&ctx, "__registerModule")?;
            register
                .call::<_, ()>((key_owned.clone(), namespace.clone()))
                .internal()?;
            let workshop_id: Option<String> = namespace.get("__workshopId").ok();
            Ok(workshop_id)
        })?;

        self.modules.insert(
            key.to_owned(),
            ModuleMeta {
                owner_id,
                inited: false,
                current: initial,
                workshop_id,
            },
        );
        Ok(())
    }

    /// Run one frame (docs §3.2): fire due timers, then for each module in key
    /// order bind `thisLayer`, run deferred `init` once, call `update(value)`
    /// and collect its return. Drains recorded scene ops and console output.
    pub fn tick(&mut self, frame: &HostFrame, overrides: &[(String, ScriptValue)]) -> TickOutput {
        // Apply integrator-pushed value changes (user edits) before the tick.
        for (k, v) in overrides {
            if let Some(m) = self.modules.get_mut(k) {
                m.current = v.clone();
            }
        }
        let metas: Vec<ModuleTickState> = self
            .modules
            .iter()
            .map(|(k, m)| {
                (
                    k.clone(),
                    m.owner_id,
                    m.inited,
                    m.current.clone(),
                    m.workshop_id.clone(),
                )
            })
            .collect();

        let (results, mut out) = self.context.with(|ctx| {
            let mut out = TickOutput::default();
            if let Err(e) = apply_frame(&ctx, frame) {
                out.errors.push(e);
            }
            // docs §3.2.a: fire due engine timers (self-catching in JS).
            let _ = call_void(&ctx, "__tickTimers", ());

            let mut results: Vec<(String, ScriptValue, bool)> = Vec::new();
            for (key, owner, inited, current, workshop) in &metas {
                bind_this_layer(&ctx, *owner);
                set_workshop_id(&ctx, workshop.as_deref());
                let arg = match current.to_js(&ctx) {
                    Ok(v) => v,
                    Err(e) => {
                        out.errors.push(ScriptError::Internal(e.to_string()));
                        continue;
                    }
                };
                if !inited && let Err(msg) = call_export(&ctx, key, "init", arg.clone()) {
                    out.errors.push(ScriptError::Runtime {
                        key: key.clone(),
                        phase: "init",
                        message: msg,
                    });
                }
                // `update(value)` must receive the property's CURRENT value —
                // including writes made by this module's own `init()` (e.g. the
                // visualizer template's `thisLayer.visible = false`) or another
                // script, which the host-side `current` cache cannot see. The
                // layer snapshot in the JS world is the live truth for
                // layer-backed props; fall back to the cache when the layer
                // doesn't carry the prop (module keys are "<prop>_<objectId>").
                let arg = match (*owner, key.rsplit_once('_')) {
                    (Some(id), Some((prop, _))) => match call_ret2(&ctx, "__getLayerProp", (id, prop)) {
                        Ok(v) if !v.is_undefined() => v,
                        _ => arg,
                    },
                    _ => arg,
                };
                match call_export_ret(&ctx, key, "update", arg) {
                    Ok(Some(ret)) => {
                        results.push((key.clone(), ScriptValue::from_js(&ret), true));
                    }
                    Ok(None) => { /* no update export — leave value untouched */ }
                    Err(msg) => {
                        // docs §5.1: exception from update skips the write-back.
                        out.errors.push(ScriptError::Runtime {
                            key: key.clone(),
                            phase: "update",
                            message: msg,
                        });
                    }
                }
            }
            drain_side_effects(&ctx, &mut out);
            (results, out)
        });

        for (key, value, applied) in results {
            if let Some(m) = self.modules.get_mut(&key) {
                m.inited = true;
                if applied {
                    m.current = value.clone();
                    out.property_results.push((key, value));
                }
            }
        }
        // Mark inited even for modules with no update export.
        for m in self.modules.values_mut() {
            m.inited = true;
        }
        out
    }

    /// Dispatch `applyUserProperties({key: value})` to every module (docs §5.3).
    pub fn dispatch_user_property(&mut self, key: &str, value: &ScriptValue) -> TickOutput {
        let keys: Vec<(String, Option<i64>)> = self
            .modules
            .iter()
            .map(|(k, m)| (k.clone(), m.owner_id))
            .collect();
        self.context.with(|ctx| {
            let mut out = TickOutput::default();
            let payload = match build_single(&ctx, key, value) {
                Ok(p) => p,
                Err(e) => {
                    out.errors.push(e);
                    return out;
                }
            };
            for (mkey, owner) in &keys {
                bind_this_layer(&ctx, *owner);
                if let Err(msg) = call_export(&ctx, mkey, "applyUserProperties", payload.clone()) {
                    out.errors.push(ScriptError::Runtime {
                        key: mkey.clone(),
                        phase: "applyUserProperties",
                        message: msg,
                    });
                }
            }
            drain_side_effects(&ctx, &mut out);
            out
        })
    }

    /// Create a text-layer script (docs §7): returns a positive handle, or 0 on
    /// evaluation failure.
    pub fn create_layer_script(
        &mut self,
        source: &str,
        script_properties: &serde_json::Value,
        initial_text: &str,
    ) -> u32 {
        self.next_layer_handle += 1;
        let handle = self.next_layer_handle;
        let ok = self.context.with(|ctx| -> bool {
            let props = match json_to_js(&ctx, script_properties) {
                Ok(v) => v,
                Err(_) => return false,
            };
            let create: Function = match global(&ctx, "__createLayerScript") {
                Ok(f) => f,
                Err(_) => return false,
            };
            create
                .call::<_, bool>((handle, source, props, initial_text))
                .catch(&ctx)
                .unwrap_or(false)
        });
        if ok { handle } else { 0 }
    }

    /// Tick a text layer (docs §7.2): runs its deferred `init` then `update`.
    pub fn tick_layer(&mut self, handle: u32, time: f64, dt: f64, fps: f64) -> Vec<LogLine> {
        self.context.with(|ctx| {
            let _ = call_void(&ctx, "__tickLayer", (handle, time, dt, fps));
            let mut out = TickOutput::default();
            drain_side_effects(&ctx, &mut out);
            out.logs
        })
    }

    /// Read a text layer's current rendered text (docs §7.2); `""` for an
    /// invalid handle.
    pub fn layer_text(&self, handle: u32) -> String {
        self.context.with(|ctx| {
            global(&ctx, "__layerText")
                .and_then(|f: Function| f.call::<_, String>((handle,)).internal())
                .unwrap_or_default()
        })
    }

    /// Destroy a text layer (docs §7.2): calls `destroy()` if present.
    pub fn destroy_layer(&mut self, handle: u32) {
        self.context.with(|ctx| {
            let _ = call_void(&ctx, "__destroyLayer", (handle,));
        });
    }

    /// Evaluate an arbitrary global script and return its value as a string
    /// (test/diagnostic helper).
    pub fn eval_to_string(&self, source: &str) -> Result<String, ScriptError> {
        self.context.with(|ctx| {
            ctx.eval::<Value, _>(source)
                .catch(&ctx)
                .map_err(|e| ScriptError::Runtime {
                    key: String::new(),
                    phase: "eval",
                    message: e.to_string(),
                })
                .map(|v| stringify(&ctx, &v))
        })
    }
}

// ---- helpers --------------------------------------------------------------

fn eval_global(ctx: &Ctx<'_>, name: &str, src: &str) -> Result<(), ScriptError> {
    ctx.eval::<(), _>(src).catch(ctx).map_err(|e| ScriptError::Load {
        key: name.to_owned(),
        message: e.to_string(),
    })
}

fn global<'js, T: rquickjs::FromJs<'js>>(ctx: &Ctx<'js>, name: &str) -> Result<T, ScriptError> {
    ctx.globals().get(name).internal()
}

fn drain_jobs(ctx: &Ctx<'_>) {
    // docs §4.2 step 4: drain the pending-job queue after module evaluation.
    while ctx.execute_pending_job() {}
}

fn call_void<'js, A: rquickjs::function::IntoArgs<'js>>(
    ctx: &Ctx<'js>,
    name: &str,
    args: A,
) -> Result<(), ScriptError> {
    let f: Function = global(ctx, name)?;
    f.call::<_, Value>(args)
        .catch(ctx)
        .map(|_| ())
        .map_err(|e| ScriptError::Internal(e.to_string()))
}

/// Call a global function returning its raw JS value (for `__getLayerProp`).
fn call_ret2<'js, A: rquickjs::function::IntoArgs<'js>>(
    ctx: &Ctx<'js>,
    name: &str,
    args: A,
) -> Result<Value<'js>, ScriptError> {
    let f: Function = global(ctx, name)?;
    f.call::<_, Value>(args)
        .catch(ctx)
        .map_err(|e| ScriptError::Internal(e.to_string()))
}

fn bind_this_layer(ctx: &Ctx<'_>, owner: Option<i64>) {
    if let Ok(f) = global::<Function>(ctx, "__bindThisLayer") {
        let arg = match owner {
            Some(id) => Value::new_number(ctx.clone(), id as f64),
            None => Value::new_null(ctx.clone()),
        };
        let _ = f.call::<_, ()>((arg,));
    }
}

fn set_workshop_id(ctx: &Ctx<'_>, id: Option<&str>) {
    if let Ok(host) = global::<Object>(ctx, "__host") {
        let _ = match id {
            Some(s) => {
                rquickjs::String::from_str(ctx.clone(), s).map(|v| host.set("workshopId", v.into_value()))
            }
            None => Ok(host.set("workshopId", Value::new_null(ctx.clone()))),
        };
    }
}

/// Call `module.export(arg)` ignoring any return; a missing export is not an
/// error (docs §4.2: missing/non-function exports are silently skipped).
fn call_export<'js>(ctx: &Ctx<'js>, key: &str, name: &str, arg: Value<'js>) -> Result<(), String> {
    call_export_ret(ctx, key, name, arg).map(|_| ())
}

/// Call `module.export(arg)` via `__callExport`, returning its value. `Ok(None)`
/// = the export was missing/non-function; `Err` = the call threw.
fn call_export_ret<'js>(
    ctx: &Ctx<'js>,
    key: &str,
    name: &str,
    arg: Value<'js>,
) -> Result<Option<Value<'js>>, String> {
    let f: Function = ctx.globals().get("__callExport").map_err(|e| e.to_string())?;
    let ret: Object = f.call((key, name, arg)).catch(ctx).map_err(|e| e.to_string())?;
    if ret.get::<_, bool>("__missing").unwrap_or(false) {
        return Ok(None);
    }
    Ok(Some(ret.get("value").map_err(|e| e.to_string())?))
}

/// Inject the frame snapshot into `__host` (only the data fields; JS-owned state
/// such as timers/textLayers/modules is preserved).
fn apply_frame(ctx: &Ctx<'_>, frame: &HostFrame) -> Result<(), ScriptError> {
    let host: Object = global(ctx, "__host")?;
    let json =
        serde_json::to_value(frame).map_err(|e| ScriptError::Internal(format!("frame serialize: {e}")))?;
    if let serde_json::Value::Object(map) = json {
        for (k, v) in &map {
            host.set(k.as_str(), json_to_js(ctx, v).internal()?).internal()?;
        }
    }
    Ok(())
}

fn build_single<'js>(ctx: &Ctx<'js>, key: &str, value: &ScriptValue) -> Result<Value<'js>, ScriptError> {
    let obj = Object::new(ctx.clone()).internal()?;
    obj.set(key, value.to_js(ctx).internal()?).internal()?;
    Ok(obj.into_value())
}

/// Drain `__host.ops` (typed scene mutations) and `__host.console`, then reset
/// both to empty arrays for the next tick.
fn drain_side_effects(ctx: &Ctx<'_>, out: &mut TickOutput) {
    let host: Object = match global(ctx, "__host") {
        Ok(h) => h,
        Err(e) => {
            out.errors.push(e);
            return;
        }
    };
    if let Ok(ops) = host.get::<_, Array>("ops") {
        for i in 0..ops.len() {
            if let Ok(v) = ops.get::<Value>(i)
                && let Some(op) = parse_op(&v)
            {
                out.ops.push(op);
            }
        }
    }
    if let Ok(console) = host.get::<_, Array>("console") {
        for i in 0..console.len() {
            if let Ok(s) = console.get::<String>(i) {
                // Each line is tagged with a leading 'I' (log) or 'E' (error).
                let error = s.starts_with('E');
                out.logs.push(LogLine {
                    error,
                    message: s.get(1..).unwrap_or("").to_owned(),
                });
            }
        }
    }
    if let Ok(empty) = Array::new(ctx.clone()) {
        let _ = host.set("ops", empty);
    }
    if let Ok(empty) = Array::new(ctx.clone()) {
        let _ = host.set("console", empty);
    }
}

/// Parse one recorded op object (`{op: "...", ...}`) into a [`SceneOp`].
fn parse_op(v: &Value<'_>) -> Option<SceneOp> {
    let obj = v.as_object()?;
    let op: String = obj.get("op").ok()?;
    match op.as_str() {
        "setProp" => Some(SceneOp::SetProperty {
            layer_id: obj.get::<_, f64>("id").ok()? as i64,
            name: obj.get("name").ok()?,
            value: op_value(&obj.get::<_, Value>("value").ok()?),
        }),
        "setParent" => Some(SceneOp::SetParent {
            layer_id: obj.get::<_, f64>("id").ok()? as i64,
            parent: obj
                .get::<_, Option<f64>>("parent")
                .ok()
                .flatten()
                .map(|f| f as i64),
        }),
        "setCameraTransforms" => Some(SceneOp::SetCameraTransforms {
            eye: get_vec3(obj, "eye"),
            center: get_vec3(obj, "center"),
            up: get_vec3(obj, "up"),
            fov: obj.get::<_, f64>("fov").ok().map(|f| f as f32),
        }),
        "createLayer" => Some(SceneOp::CreateLayer {
            layer_id: obj.get::<_, f64>("id").ok()? as i64,
            path: obj.get("path").ok()?,
            workshop_id: obj.get::<_, Option<String>>("workshopId").ok().flatten(),
        }),
        "sortLayer" => Some(SceneOp::SortLayer {
            layer_id: obj.get::<_, f64>("id").ok()? as i64,
            index: obj.get::<_, f64>("index").ok()? as i64,
        }),
        _ => None,
    }
}

fn get_vec3(obj: &Object<'_>, key: &str) -> Option<[f32; 3]> {
    let arr: Array = obj.get(key).ok()?;
    if arr.len() < 3 {
        return None;
    }
    Some([
        arr.get::<f64>(0).ok()? as f32,
        arr.get::<f64>(1).ok()? as f32,
        arr.get::<f64>(2).ok()? as f32,
    ])
}

/// Decode an op's `value` field, which is a scalar, a bool, or a `[x,y,z]`
/// array (vector properties).
fn op_value(v: &Value<'_>) -> ScriptValue {
    if v.is_array()
        && let Some(arr) = v.as_array()
    {
        let comps: Vec<f32> = (0..arr.len())
            .filter_map(|i| arr.get::<f64>(i).ok().map(|f| f as f32))
            .collect();
        return match comps.len() {
            2 => ScriptValue::Vec2([comps[0], comps[1]]),
            3 => ScriptValue::Vec3([comps[0], comps[1], comps[2]]),
            4 => ScriptValue::Vec4([comps[0], comps[1], comps[2], comps[3]]),
            _ => ScriptValue::Null,
        };
    }
    ScriptValue::from_js(v)
}

fn stringify(ctx: &Ctx<'_>, v: &Value<'_>) -> String {
    if let Some(s) = v.as_string() {
        return s.to_string().unwrap_or_default();
    }
    let _ = ctx;
    match ScriptValue::from_js(v) {
        ScriptValue::Null => "null".to_owned(),
        ScriptValue::Bool(b) => b.to_string(),
        ScriptValue::Int(i) => i.to_string(),
        ScriptValue::Float(f) => f.to_string(),
        ScriptValue::Str(s) => s,
        ScriptValue::Vec2(v) => format!("{}, {}", v[0], v[1]),
        ScriptValue::Vec3(v) => format!("{}, {}, {}", v[0], v[1], v[2]),
        ScriptValue::Vec4(v) => format!("{}, {}, {}, {}", v[0], v[1], v[2], v[3]),
    }
}

/// Map a raw `rquickjs::Result` to [`ScriptError::Internal`].
trait Internalize<T> {
    fn internal(self) -> Result<T, ScriptError>;
}
impl<T> Internalize<T> for rquickjs::Result<T> {
    fn internal(self) -> Result<T, ScriptError> {
        self.map_err(|e| ScriptError::Internal(e.to_string()))
    }
}
