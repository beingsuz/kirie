//! Typed per-tick host snapshot ([`HostFrame`]) fed to the script world, and the
//! typed outputs ([`TickOutput`], [`SceneOp`]) drained back out.
//!
//! docs/scripting-api.md §3/§6/§8. SPEC.md §V3: the script `Runtime` lives on its
//! own thread and never touches engine memory; the integrator marshals an
//! immutable frame in and reads typed mutation ops out. All host-visible reads
//! (`engine.*`, `thisScene.*`, `thisLayer.*`, `input.*`) are served from this
//! frame; every scene mutation a script performs is recorded as a [`SceneOp`].

use std::collections::BTreeMap;

use serde::Serialize;

use crate::value::ScriptValue;

/// A scriptable layer's state (docs §4.1 registered properties). `None` fields
/// are omitted so a script's `name in thisLayer` check matches the C++ (only
/// the properties actually registered for that object type are present).
///
/// `Clone` is hand-written so `clone_from` reuses the `name`/`text` string
/// heap: the integrator refreshes a retained [`HostFrame`]'s `layers` every
/// tick via `Vec::clone_from` (which clones element-wise through
/// `Clone::clone_from`), and the derived impl would re-allocate both strings
/// per layer per frame.
#[derive(Debug, Default, Serialize)]
pub struct LayerState {
    /// Object id (`getLayerByID`, op targeting).
    pub id: i64,
    /// Object name (`thisLayer.name`, `getLayer("name")`).
    pub name: String,
    /// Parent object id, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<i64>,
    /// `origin` (translation), scene units.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<[f32; 3]>,
    /// `scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 3]>,
    /// `angles` (degrees, Y-X-Z order).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angles: Option<[f32; 3]>,
    /// `color` (RGB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 3]>,
    /// `alpha`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// `visible`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    /// `parallaxDepth`.
    #[serde(rename = "parallaxDepth", skip_serializing_if = "Option::is_none")]
    pub parallax_depth: Option<f32>,
    /// `pointSize` (text objects).
    #[serde(rename = "pointSize", skip_serializing_if = "Option::is_none")]
    pub point_size: Option<f32>,
    /// `text` (text objects).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        LayerState {
            id: self.id,
            name: self.name.clone(),
            parent: self.parent,
            origin: self.origin,
            scale: self.scale,
            angles: self.angles,
            color: self.color,
            alpha: self.alpha,
            visible: self.visible,
            parallax_depth: self.parallax_depth,
            point_size: self.point_size,
            text: self.text.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        // Exhaustive destructure: adding a field without updating this copy is
        // a compile error, so the reuse path can never silently drop state.
        let LayerState {
            id,
            name,
            parent,
            origin,
            scale,
            angles,
            color,
            alpha,
            visible,
            parallax_depth,
            point_size,
            text,
        } = source;
        self.id = *id;
        // `String::clone_from` (and `Option`'s Some→Some forwarding) reuse the
        // existing capacity when it suffices — the whole point of this impl.
        self.name.clone_from(name);
        self.parent = *parent;
        self.origin = *origin;
        self.scale = *scale;
        self.angles = *angles;
        self.color = *color;
        self.alpha = *alpha;
        self.visible = *visible;
        self.parallax_depth = *parallax_depth;
        self.point_size = *point_size;
        self.text.clone_from(text);
    }
}

/// Camera transforms (docs §6.2 `getCameraTransforms`): the base pre-override
/// camera, so scripts recompute absolutely each frame.
#[derive(Clone, Debug, Serialize)]
pub struct CameraState {
    /// Eye position.
    pub eye: [f32; 3],
    /// Look-at center.
    pub center: [f32; 3],
    /// Up vector.
    pub up: [f32; 3],
    /// Vertical field of view (degrees).
    pub fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        CameraState {
            eye: [0.0, 0.0, 0.0],
            center: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
        }
    }
}

/// Scene-level properties (docs §6.2 `thisScene` read-only members). The two
/// reference copy-paste defects (`clearenabled`, `skylightcolor`) are *not*
/// reproduced — the integrator supplies the correct sources.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SceneState {
    /// `bloom`.
    pub bloom: bool,
    /// `bloomstrength`.
    pub bloomstrength: i64,
    /// `bloomthreshold`.
    pub bloomthreshold: i64,
    /// `clearenabled`.
    pub clearenabled: bool,
    /// `clearcolor` (RGB).
    pub clearcolor: [f32; 3],
    /// `ambientcolor` (RGB).
    pub ambientcolor: [f32; 3],
    /// `skylightcolor` (RGB).
    pub skylightcolor: [f32; 3],
    /// `fov`.
    pub fov: f32,
    /// `nearz`.
    pub nearz: f32,
    /// `farz`.
    pub farz: f32,
    /// `camerafade`.
    pub camerafade: bool,
    /// `camerashake`.
    pub camerashake: bool,
    /// `camerashakespeed`.
    pub camerashakespeed: f32,
    /// `camerashakeamplitude`.
    pub camerashakeamplitude: f32,
    /// `camerashakeroughness`.
    pub camerashakeroughness: f32,
    /// `cameraparallax`.
    pub cameraparallax: bool,
    /// `cameraparallaxamount`.
    pub cameraparallaxamount: f32,
    /// `cameraparallaxdelay`.
    pub cameraparallaxdelay: f32,
    /// `cameraparallaxmouseinfluence`.
    pub cameraparallaxmouseinfluence: f32,
    /// Base camera transforms.
    pub camera: CameraState,
}

/// The three FFT band reductions a script may request via
/// `engine.registerAudioBuffers(16|32|64)`. Each resolution's `average` getter
/// returns its *matching* array (docs/scripting-api.md §6.1) — NOT a slice of
/// the 64-band array — so all three must be marshalled every tick. Serialized
/// under keys `a16`/`a32`/`a64` on `__host.audio` for the JS bridge.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AudioBuffers {
    /// 16-band reduction (`registerAudioBuffers(16)`).
    #[serde(rename = "a16")]
    pub audio16: Vec<f32>,
    /// 32-band reduction (`registerAudioBuffers(32)`).
    #[serde(rename = "a32")]
    pub audio32: Vec<f32>,
    /// 64-band reduction (`registerAudioBuffers(64)`, the default/fallback).
    #[serde(rename = "a64")]
    pub audio64: Vec<f32>,
}

/// The immutable per-tick host snapshot (docs §3). Everything a script can read
/// this frame. Field names match the injected `__host` object exactly.
#[derive(Clone, Debug, Serialize)]
pub struct HostFrame {
    /// `engine.runtime` — seconds since start × playback speed.
    pub runtime: f64,
    /// `engine.frametime` — seconds since previous frame.
    pub frametime: f64,
    /// `engine.timeOfDay` — wall-clock day fraction ∈ [0,1).
    #[serde(rename = "timeOfDay")]
    pub time_of_day: f64,
    /// Monotonic milliseconds, for `engine.setInterval`/`setTimeout` firing.
    pub now: f64,
    /// `engine.screenResolution.x` (scene units).
    #[serde(rename = "resX")]
    pub res_x: f64,
    /// `engine.screenResolution.y` (scene units).
    #[serde(rename = "resY")]
    pub res_y: f64,
    /// `engine.userProperties` — project property name → current value.
    #[serde(rename = "userProps")]
    pub user_props: BTreeMap<String, ScriptValue>,
    /// `input.cursorScreenPosition` ∈ [0,1].
    #[serde(rename = "pointerScreen")]
    pub pointer_screen: [f32; 2],
    /// `input.cursorWorldPosition`.
    #[serde(rename = "pointerWorld")]
    pub pointer_world: [f32; 3],
    /// `input.cursorLeftDown`.
    #[serde(rename = "pointerLeftDown")]
    pub pointer_left_down: bool,
    /// FFT band arrays (`engine.registerAudioBuffers(16|32|64).average`); `None`
    /// yields zeros. Each resolution reads its *own* reduction — the 16-band
    /// getter must return `audio16`, not the first 16 of `audio64`
    /// (docs/scripting-api.md §6.1 "the matching `audioN` array").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioBuffers>,
    /// Scene-level properties.
    pub scene: SceneState,
    /// Scriptable layers in render order (index = `getLayer(i)`).
    pub layers: Vec<LayerState>,
    /// `__workshopId` fallback for `thisScene.createLayer` path resolution.
    #[serde(rename = "workshopId", skip_serializing_if = "Option::is_none")]
    pub workshop_id: Option<String>,
}

impl Default for HostFrame {
    fn default() -> Self {
        HostFrame {
            runtime: 0.0,
            frametime: 0.0,
            time_of_day: 0.0,
            now: 0.0,
            res_x: 1920.0,
            res_y: 1080.0,
            user_props: BTreeMap::new(),
            pointer_screen: [0.0, 0.0],
            pointer_world: [0.0, 0.0, 0.0],
            pointer_left_down: false,
            audio: None,
            scene: SceneState::default(),
            layers: Vec::new(),
            workshop_id: None,
        }
    }
}

/// A typed scene mutation a script requested this tick (docs §6.2/§8). The
/// integrator applies these against the live scene on the update thread.
#[derive(Clone, Debug, PartialEq)]
pub enum SceneOp {
    /// `thisLayer.<prop> = v` / `rotateObjectSpace` / `lookAt*` — write a
    /// registered property on a layer.
    SetProperty {
        /// Target layer id.
        layer_id: i64,
        /// Registered property name (`origin`, `scale`, `angles`, `color`,
        /// `alpha`, `visible`, `parallaxDepth`, `pointSize`).
        name: String,
        /// New value.
        value: ScriptValue,
    },
    /// `thisLayer.setParent(...)` — reparent a layer (`None` = detach).
    SetParent {
        /// Target layer id.
        layer_id: i64,
        /// New parent id, or `None`.
        parent: Option<i64>,
    },
    /// `thisScene.setCameraTransforms({...})` — runtime camera override.
    SetCameraTransforms {
        /// New eye, if provided.
        eye: Option<[f32; 3]>,
        /// New center, if provided.
        center: Option<[f32; 3]>,
        /// New up, if provided.
        up: Option<[f32; 3]>,
        /// New fov, if provided.
        fov: Option<f32>,
    },
    /// `thisScene.createLayer(path|{file})` — instantiate a runtime image layer.
    CreateLayer {
        /// The script-world synthetic layer id later `SetProperty` ops target.
        layer_id: i64,
        /// Model path or asset file.
        path: String,
        /// Workshop id for relative-path resolution, if known.
        workshop_id: Option<String>,
    },
    /// `thisScene.sortLayer(layer, index)` — z-order move.
    SortLayer {
        /// Target layer id.
        layer_id: i64,
        /// New scriptable index.
        index: i64,
    },
}

/// A single console line drained from a tick.
#[derive(Clone, Debug, PartialEq)]
pub struct LogLine {
    /// `true` for `console.error`, `false` for `console.log`.
    pub error: bool,
    /// The concatenated message text.
    pub message: String,
}

/// Everything produced by one [`crate::ScriptEngine::tick`].
#[derive(Clone, Debug, Default)]
pub struct TickOutput {
    /// Per property-script result: module key → value to apply to that property
    /// (docs §5.1: `update()`'s return value drives the property). `Null` means
    /// the property is set to the Null type (not a no-op).
    pub property_results: Vec<(String, ScriptValue)>,
    /// Scene mutations recorded this tick, in the order scripts issued them.
    pub ops: Vec<SceneOp>,
    /// `console.log`/`error` output this tick.
    pub logs: Vec<LogLine>,
    /// Non-fatal runtime errors (a script threw); the script stays loaded.
    pub errors: Vec<crate::ScriptError>,
}
