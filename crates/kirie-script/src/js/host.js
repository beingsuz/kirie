// kirie-script host bridge — the JS side of the engine/scene/layer/input API.
//
// docs/scripting-api.md §6/§8. In the C++ these are native adapters reading
// live engine memory. Here (SPEC.md §V3: the QuickJS runtime lives on its own
// thread and MUST NOT touch engine memory) they read a per-tick immutable
// snapshot `__host`, injected by Rust before each tick, and record scene
// mutations as typed ops into `__host.ops`, drained back to Rust as typed
// messages. Reads are snapshot-consistent within a tick.
'use strict';

// The injected frame snapshot. Rust replaces its fields each tick; the adapter
// objects below close over `globalThis.__host` by name so they always see the
// current frame.
globalThis.__host = {
  runtime: 0, frametime: 0, timeOfDay: 0, now: 0,
  resX: 1920, resY: 1080,
  userProps: {},
  pointerScreen: [0, 0], pointerWorld: [0, 0, 0], pointerLeftDown: false,
  audio: null,
  scene: { bloom: false, bloomstrength: 0, bloomthreshold: 0, clearenabled: false, clearcolor: [0, 0, 0], ambientcolor: [0, 0, 0], skylightcolor: [0, 0, 0], fov: 45, nearz: 0.1, farz: 10000, camerafade: false, camerashake: false, camerashakespeed: 0, camerashakeamplitude: 0, camerashakeroughness: 0, cameraparallax: false, cameraparallaxamount: 0, cameraparallaxdelay: 0, cameraparallaxmouseinfluence: 0, camera: { eye: [0, 0, 0], center: [0, 0, -1], up: [0, 1, 0], fov: 45 } },
  layers: [],
  ops: [],
  console: [],
  workshopId: null,
  scriptProps: {},
  timers: [],
  timerSeq: 1,
  textLayers: Object.create(null),
  modules: Object.create(null),
};

globalThis.shared = {};

// ---- console (docs §6.5): concatenate args, no separator, 0 args = no-op. ---
globalThis.console = {
  log: function () { if (arguments.length === 0) return; var s = ''; for (var i = 0; i < arguments.length; i++) s += String(arguments[i]); __host.console.push('I' + s); },
  error: function () { if (arguments.length === 0) return; var s = ''; for (var i = 0; i < arguments.length; i++) s += String(arguments[i]); __host.console.push('E' + s); },
};

// ---- engine (docs §6.1) ---------------------------------------------------
globalThis.engine = {
  get frametime() { return __host.frametime; },
  get runtime() { return __host.runtime; },
  get timeOfDay() { return __host.timeOfDay; },
  get screenResolution() { return { x: __host.resX, y: __host.resY }; },
  get canvasSize() { return { x: __host.resX, y: __host.resY }; },
  get userProperties() { var o = {}; for (var k in __host.userProps) o[k] = __host.userProps[k]; return o; },
  AUDIO_RESOLUTION_16: 16, AUDIO_RESOLUTION_32: 32, AUDIO_RESOLUTION_64: 64,
  isWallpaper: function () { return true; },
  isScreensaver: function () { return false; },
  isDesktopDevice: function () { return true; },
  isMobileDevice: function () { return false; },
  isRunningInEditor: function () { return false; },
  isPortrait: function () { return __host.resY > __host.resX; },
  isLandscape: function () { return __host.resX >= __host.resY; },
  openUserShortcut: function () { },
  registerAsset: function (file) { return (typeof file === 'string') ? { file: file } : undefined; },
  registerAudioBuffers: function (resolution) {
    var res = (resolution === 16 || resolution === 32 || resolution === 64) ? resolution : 64;
    // docs/scripting-api.md §6.1: read the *matching* audioN reduction, not the
    // first `res` entries of the 64-band array.
    var key = 'a' + res;
    return { get average() { var a = new Array(res); var buf = __host.audio; var src = (buf && buf[key]) || []; for (var i = 0; i < res; i++) a[i] = (i < src.length) ? src[i] : 0; return a; } };
  },
  // docs §5.4 (canceller-arg defect fixed): returns a zero-arg canceller that
  // cancels its own timer.
  setInterval: function (fn, ms) {
    if (typeof fn !== 'function') throw new TypeError('engine.setInterval: first argument must be a function');
    ms = ms | 0; var id = __host.timerSeq++;
    __host.timers.push({ id: id, fn: fn, ms: ms, next: __host.now + ms, repeat: true });
    return function () { __host.timers = __host.timers.filter(function (t) { return t.id !== id; }); };
  },
  setTimeout: function (fn, ms) {
    if (typeof fn !== 'function') throw new TypeError('engine.setTimeout: first argument must be a function');
    ms = ms | 0; var id = __host.timerSeq++;
    __host.timers.push({ id: id, fn: fn, ms: ms, next: __host.now + ms, repeat: false });
    return function () { __host.timers = __host.timers.filter(function (t) { return t.id !== id; }); };
  },
};

// Fire due timers (docs §3.2.a; called at the start of each tick from Rust).
globalThis.__tickTimers = function () {
  var now = __host.now;
  var due = __host.timers.filter(function (t) { return now >= t.next; });
  for (var i = 0; i < due.length; i++) {
    var t = due[i];
    try { t.fn.call(null); } catch (e) { __host.console.push('E' + String(e && e.stack || e)); }
    if (t.repeat) { t.next = now + t.ms; } else { __host.timers = __host.timers.filter(function (x) { return x.id !== t.id; }); }
  }
};

// ---- input (docs §6.4) ----------------------------------------------------
globalThis.input = {
  get cursorWorldPosition() { return new Vec3(__host.pointerWorld[0], __host.pointerWorld[1], __host.pointerWorld[2]); },
  get cursorScreenPosition() { return new Vec2(__host.pointerScreen[0], __host.pointerScreen[1]); },
  get cursorLeftDown() { return __host.pointerLeftDown; },
};

// ---- ILayer adapter (docs §8) ---------------------------------------------
function __layerById(id) { var L = __host.layers; for (var i = 0; i < L.length; i++) if (L[i].id === id) return L[i]; return null; }
function __recordProp(id, name, value) { __host.ops.push({ op: 'setProp', id: id, name: name, value: value }); }

var __VEC3_PROPS = { origin: 1, scale: 1, angles: 1, color: 1 };
var __VEC2_PROPS = {};
var __NUM_PROPS = { alpha: 1, parallaxDepth: 1, pointSize: 1 };
var __BOOL_PROPS = { visible: 1 };
var __STR_PROPS = { text: 1 };

function __makeLayer(id) {
  var self = {
    __id: id,
    get name() { var l = __layerById(id); return l ? l.name : ''; },
    getParent: function () { var l = __layerById(id); if (!l || l.parent == null) return undefined; return __makeLayer(l.parent); },
    getChildren: function () { return __host.layers.filter(function (x) { return x.parent === id; }).map(function (x) { return __makeLayer(x.id); }); },
    getLayerIndex: function () { for (var i = 0; i < __host.layers.length; i++) if (__host.layers[i].id === id) return i; return -1; },
    rotateObjectSpace: function (delta) {
      var l = __layerById(id); if (!l) return;
      var a = l.angles || [0, 0, 0];
      l.angles = [a[0] + (delta.x || 0), a[1] + (delta.y || 0), a[2] + (delta.z || 0)];
      __recordProp(id, 'angles', l.angles.slice());
    },
    getTransformMatrix: function () {
      var l = __layerById(id) || {};
      var o = l.origin || [0, 0, 0], a = l.angles || [0, 0, 0], s = l.scale || [1, 1, 1];
      // T(origin) * Ry * Rx * Rz * S(scale) — docs §8 Y-X-Z order.
      return Mat4.fromTranslation(o)
        .multiply(Mat4.fromRotation(a[1], new Vec3(0, 1, 0)))
        .multiply(Mat4.fromRotation(a[0], new Vec3(1, 0, 0)))
        .multiply(Mat4.fromRotation(a[2], new Vec3(0, 0, 1)))
        .multiply(Mat4.fromScale(new Vec3(s[0], s[1], s[2])));
    },
    lookAt: function (target) {
      var l = __layerById(id); if (!l) return;
      var o = l.origin || [0, 0, 0];
      var dir = new Vec3(target).subtract(new Vec3(o[0], o[1], o[2])).normalize();
      var yaw = Math.atan2(dir.x, -dir.z) * 57.29577951308232;
      var pitch = Math.asin(Math.max(-1, Math.min(1, dir.y))) * 57.29577951308232;
      var a = l.angles || [0, 0, 0];
      l.angles = [pitch, yaw, a[2]];
      __recordProp(id, 'angles', l.angles.slice());
    },
    lookAtYaw: function (target) {
      var l = __layerById(id); if (!l) return;
      var o = l.origin || [0, 0, 0];
      var dir = new Vec3(target).subtract(new Vec3(o[0], o[1], o[2])).normalize();
      var yaw = Math.atan2(dir.x, -dir.z) * 57.29577951308232;
      var a = l.angles || [0, 0, 0];
      l.angles = [0, yaw, a[2]];
      __recordProp(id, 'angles', l.angles.slice());
    },
    setParent: function (p) {
      var pid = null;
      if (p == null) pid = null;
      else if (typeof p === 'number') pid = p;
      else if (typeof p === 'string') { for (var i = 0; i < __host.layers.length; i++) if (__host.layers[i].name === p) { pid = __host.layers[i].id; break; } }
      else if (p.__id !== undefined) pid = p.__id;
      if (pid === id) return; // self-parent ignored
      var l = __layerById(id); if (l) l.parent = pid;
      __host.ops.push({ op: 'setParent', id: id, parent: pid });
    },
  };
  var names = Object.keys(__VEC3_PROPS).concat(Object.keys(__VEC2_PROPS), Object.keys(__NUM_PROPS), Object.keys(__BOOL_PROPS), Object.keys(__STR_PROPS));
  names.forEach(function (name) {
    Object.defineProperty(self, name, {
      enumerable: true,
      get: function () {
        var l = __layerById(id); if (!l || !(name in l)) return undefined;
        var v = l[name];
        if (__VEC3_PROPS[name]) return new Vec3(v[0], v[1], v[2]);
        if (__VEC2_PROPS[name]) return new Vec2(v[0], v[1]);
        return v;
      },
      set: function (val) {
        var l = __layerById(id); if (!l) return;
        var stored;
        if (__VEC3_PROPS[name]) { if (typeof val === 'number') stored = [val, val, val]; else stored = [val.x || 0, val.y || 0, val.z || 0]; }
        else if (__VEC2_PROPS[name]) { if (typeof val === 'number') stored = [val, val]; else stored = [val.x || 0, val.y || 0]; }
        else if (__BOOL_PROPS[name]) stored = !!val;
        else if (__NUM_PROPS[name]) stored = +val;
        else stored = val; // string props ignored by C++ ILayer set; we store for reads
        l[name] = stored;
        if (!__STR_PROPS[name]) __recordProp(id, name, (stored && stored.slice) ? stored.slice() : stored);
      },
    });
  });
  return self;
}
globalThis.__makeLayer = __makeLayer;
globalThis.__bindThisLayer = function (id) { globalThis.thisLayer = (id == null) ? undefined : __makeLayer(id); };

// ---- thisScene (docs §6.2) ------------------------------------------------
function __sceneVecGetter(key) { return function () { var v = __host.scene[key]; return new Vec3(v[0], v[1], v[2]); }; }
var __scene = {
  getLayer: function (arg) {
    if (typeof arg === 'number') { var l = __host.layers[arg]; return l ? __makeLayer(l.id) : undefined; }
    if (typeof arg === 'string') { for (var i = 0; i < __host.layers.length; i++) if (__host.layers[i].name === arg) return __makeLayer(__host.layers[i].id); throw new Error('thisScene.getLayer: no layer named ' + arg); }
    throw new Error('thisScene.getLayer: invalid argument');
  },
  getLayerCount: function () { return __host.layers.length; },
  getLayerIndex: function (layer) { if (!layer || layer.__id === undefined) return -1; for (var i = 0; i < __host.layers.length; i++) if (__host.layers[i].id === layer.__id) return i; return -1; },
  getLayerByID: function (idString) { var id = parseInt(idString, 10); var l = __layerById(id); return l ? __makeLayer(id) : undefined; },
  enumerateLayers: function () { return __host.layers.map(function (l) { return __makeLayer(l.id); }); },
  getCameraTransforms: function () { var c = __host.scene.camera; return { eye: new Vec3(c.eye[0], c.eye[1], c.eye[2]), center: new Vec3(c.center[0], c.center[1], c.center[2]), up: new Vec3(c.up[0], c.up[1], c.up[2]), fov: c.fov }; },
  setCameraTransforms: function (t) {
    var op = { op: 'setCameraTransforms' };
    if (t) {
      if (t.eye) op.eye = [t.eye.x || 0, t.eye.y || 0, t.eye.z || 0];
      if (t.center) op.center = [t.center.x || 0, t.center.y || 0, t.center.z || 0];
      if (t.up) op.up = [t.up.x || 0, t.up.y || 0, t.up.z || 0];
      if (typeof t.fov === 'number') op.fov = t.fov;
    }
    __host.ops.push(op);
  },
  // docs §6.2: the new layer is not visible in this frame's snapshot; return
  // undefined and let the integrator create it (deviation, documented).
  createLayer: function (arg) { var path = (typeof arg === 'string') ? arg : (arg && arg.file); if (!path) return undefined; __host.ops.push({ op: 'createLayer', path: path, workshopId: __host.workshopId }); return undefined; },
  sortLayer: function (layer, index) { if (layer && layer.__id !== undefined) __host.ops.push({ op: 'sortLayer', id: layer.__id, index: index }); },
};
['bloom', 'bloomstrength', 'bloomthreshold', 'clearenabled', 'fov', 'nearz', 'farz', 'camerafade', 'camerashake', 'camerashakespeed', 'camerashakeamplitude', 'camerashakeroughness', 'cameraparallax', 'cameraparallaxamount', 'cameraparallaxdelay', 'cameraparallaxmouseinfluence'].forEach(function (k) {
  Object.defineProperty(__scene, k, { enumerable: true, get: function () { return __host.scene[k]; } });
});
['clearcolor', 'ambientcolor', 'skylightcolor'].forEach(function (k) { Object.defineProperty(__scene, k, { enumerable: true, get: __sceneVecGetter(k) }); });
globalThis.thisScene = __scene;

// ---- createScriptProperties (docs §5.5, property-script variant) ----------
// Descriptors are ignored; values come only from the module's JSON
// scriptproperties (injected as __host.scriptProps before module evaluation).
globalThis.createScriptProperties = function () {
  var captured = __host.scriptProps || {};
  var builder = {
    addSlider: function () { return builder; },
    addCheckbox: function () { return builder; },
    addCombo: function () { return builder; },
    addColor: function () { return builder; },
    addText: function () { return builder; },
    finish: function () { var o = {}; for (var k in captured) o[k] = captured[k]; return o; },
  };
  return builder;
};

// ---- property-script module dispatch --------------------------------------
// Rust stores each evaluated module namespace under its key; these helpers keep
// all JS handles inside the JS heap (no Rust-side Persistent needed).
globalThis.__registerModule = function (key, ns) { __host.modules[key] = ns; };
globalThis.__callExport = function (key, name, arg) {
  var ns = __host.modules[key];
  if (!ns) return { __missing: true };
  var fn = ns[name];
  if (typeof fn !== 'function') return { __missing: true };
  return { value: fn.call(ns, arg) };
};
globalThis.__moduleWorkshopId = function (key) { var ns = __host.modules[key]; return ns && typeof ns.__workshopId === 'string' ? ns.__workshopId : null; };

// ---- text-layer scripts (docs §7) -----------------------------------------
globalThis.__createLayerScript = function (handle, source, props, text) {
  // §7.1 source transform: strip 'use strict'; / "use strict"; / 'export '.
  var stripped = source.split("'use strict';").join('').split('"use strict";').join('').split('export ').join('');
  var wrapper = '(function(){\n'
    + 'var __id = ' + handle + ';\n'
    + 'var __props = Object.assign({}, globalThis.__layerSeedProps || {});\n'
    + 'var thisLayer = { text: String(globalThis.__layerSeedText || "") };\n'
    + 'var thisScene = { get time(){ var c = globalThis.__sceneCtx; return c?c.time:0; }, get currentTime(){ var c = globalThis.__sceneCtx; return c?c.time:0; }, get dt(){ var c = globalThis.__sceneCtx; return c?c.dt:0; }, get fps(){ var c = globalThis.__sceneCtx; return c?c.fps:60; } };\n'
    + 'var engine = { get frametime(){ var c = globalThis.__sceneCtx; return c?c.dt:0; }, get time(){ var c = globalThis.__sceneCtx; return c?c.time:0; } };\n'
    + 'function createScriptProperties(){ var b = { addSlider:add, addCheckbox:add, addCombo:add, addColor:add, addText:add, finish:function(){ return __props; } }; function add(o){ if (o && !(o.name in __props)) __props[o.name] = o.value; return b; } return b; }\n'
    + stripped + '\n'
    + 'globalThis.__host.textLayers[__id] = { thisLayer: thisLayer, thisScene: thisScene, _init: (typeof init === "function")?init:null, _destroy: (typeof destroy === "function")?destroy:null, _tick: (typeof update === "function")?function(){ var r = update(thisLayer.text); if (typeof r === "string") thisLayer.text = r; }:null, _inited:false };\n'
    + '})();';
  globalThis.__layerSeedProps = props || {};
  globalThis.__layerSeedText = text || '';
  try { (0, eval)(wrapper); } finally { globalThis.__layerSeedProps = undefined; globalThis.__layerSeedText = undefined; }
  return !!__host.textLayers[handle];
};
globalThis.__tickLayer = function (handle, time, dt, fps) {
  globalThis.__sceneCtx = { time: time, dt: dt, fps: fps };
  var e = __host.textLayers[handle]; if (!e) return;
  if (!e._inited) { e._inited = true; if (e._init) { try { e._init.call(e); } catch (ex) { __host.console.push('E' + String(ex && ex.stack || ex)); } } }
  if (e._tick) { try { e._tick.call(e); } catch (ex) { __host.console.push('E' + String(ex && ex.stack || ex)); } }
};
globalThis.__layerText = function (handle) { var e = __host.textLayers[handle]; return e ? String(e.thisLayer.text) : ''; };
globalThis.__destroyLayer = function (handle) { var e = __host.textLayers[handle]; if (e && e._destroy) { try { e._destroy.call(e); } catch (ex) { __host.console.push('E' + String(ex && ex.stack || ex)); } } delete __host.textLayers[handle]; };
