// kirie-script builtins — pure-JS layer, evaluated once as global code at
// context init. Port of linux-wallpaperengine
// `src/WallpaperEngine/Scripting/resources/builtins.js` (docs/scripting-api.md
// §10), with the DO-NOT-PORT defects fixed:
//   * Vec2/Vec3/Vec4 are real JS classes (docs §9.2: the C++ exotic getter made
//     every method throw). All operand-order reversals in §9 are corrected.
//   * lengthSqr actually squares (§9 defect: aliased length).
// Angles are degrees throughout, matching the reference API.
'use strict';

var DEG2RAD = 0.017453292519943295;
var RAD2DEG = 57.29577951308232;
var EPS = 1e-6;

function __clampNum(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
function __fractNum(x) { return x - Math.floor(x); }
function __modNum(x, y) { return x - y * Math.floor(x / y); } // GLSL mod
function __smoothStepNum(e0, e1, x) {
  var t = __clampNum((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}
function __isNum(v) { return typeof v === 'number' && isFinite(v); }

// ---------------------------------------------------------------------------
// Vec2 / Vec3 / Vec4 — docs §9 (fixed) + §10.1 additions.
// ---------------------------------------------------------------------------

function __comp(v, i, d) {
  // Read component i of a Vec-like arg: number broadcasts, {x,y,z,w} object,
  // or a native VecN. Missing → d.
  if (typeof v === 'number') return v;
  if (v == null) return d;
  var k = ['x', 'y', 'z', 'w'][i];
  var c = v[k];
  return (typeof c === 'number') ? c : d;
}

class Vec2 {
  constructor(x, y) {
    if (arguments.length === 0) { this.x = 0; this.y = 0; return; }
    if (arguments.length === 1) {
      if (typeof x === 'number') { this.x = x; this.y = x; return; }
      this.x = __comp(x, 0, 0); this.y = __comp(x, 1, 0); return;
    }
    this.x = +x || 0; this.y = +y || 0;
  }
  copy() { return new Vec2(this.x, this.y); }
  clone() { return this.copy(); }
  equals(o) { return o instanceof Vec2 && this.x === o.x && this.y === o.y; }
  length() { return Math.sqrt(this.x * this.x + this.y * this.y); }
  lengthSqr() { return this.x * this.x + this.y * this.y; }
  normalize() { var l = this.length(); return l < EPS ? new Vec2(0, 0) : new Vec2(this.x / l, this.y / l); }
  add(v) { return new Vec2(this.x + __comp(v, 0, 0), this.y + __comp(v, 1, 0)); }
  subtract(v) { return new Vec2(this.x - __comp(v, 0, 0), this.y - __comp(v, 1, 0)); }
  multiply(v) { return new Vec2(this.x * __comp(v, 0, 1), this.y * __comp(v, 1, 1)); }
  divide(v) { return new Vec2(this.x / __comp(v, 0, 1), this.y / __comp(v, 1, 1)); }
  dot(v) { return this.x * __comp(v, 0, 0) + this.y * __comp(v, 1, 0); }
  min(v) { return new Vec2(Math.min(this.x, __comp(v, 0, 0)), Math.min(this.y, __comp(v, 1, 0))); }
  max(v) { return new Vec2(Math.max(this.x, __comp(v, 0, 0)), Math.max(this.y, __comp(v, 1, 0))); }
  mix(v, t) { return new Vec2(this.x + (__comp(v, 0, 0) - this.x) * t, this.y + (__comp(v, 1, 0) - this.y) * t); }
  abs() { return new Vec2(Math.abs(this.x), Math.abs(this.y)); }
  sign() { return new Vec2(Math.sign(this.x), Math.sign(this.y)); }
  round() { return new Vec2(Math.round(this.x), Math.round(this.y)); }
  floor() { return new Vec2(Math.floor(this.x), Math.floor(this.y)); }
  ceil() { return new Vec2(Math.ceil(this.x), Math.ceil(this.y)); }
  negate() { return new Vec2(-this.x, -this.y); }
  isFinite() { return isFinite(this.x) && isFinite(this.y); }
  distance(o) { return this.subtract(o).length(); }
  distanceSqr(o) { return this.subtract(o).lengthSqr(); }
  project(v) { var d = __comp(v, 0, 0) * __comp(v, 0, 0) + __comp(v, 1, 0) * __comp(v, 1, 0); if (d < EPS * EPS) return new Vec2(0, 0); var s = this.dot(v) / d; return new Vec2(__comp(v, 0, 0) * s, __comp(v, 1, 0) * s); }
  reflect(n) { var d = 2 * this.dot(n); return new Vec2(this.x - d * __comp(n, 0, 0), this.y - d * __comp(n, 1, 0)); }
  perpendicular() { return new Vec2(-this.y, this.x); }
  angle() { return Math.atan2(this.y, this.x) * RAD2DEG; }
  angleBetween(v) { return Math.atan2(this.x * __comp(v, 1, 0) - this.y * __comp(v, 0, 0), this.dot(v)) * RAD2DEG; }
  rotate(deg) { var r = deg * DEG2RAD, c = Math.cos(r), s = Math.sin(r); return new Vec2(this.x * c - this.y * s, this.x * s + this.y * c); }
  clamp(lo, hi) { return new Vec2(__clampNum(this.x, __comp(lo, 0, 0), __comp(hi, 0, 0)), __clampNum(this.y, __comp(lo, 1, 0), __comp(hi, 1, 0))); }
  mod(v) { return new Vec2(__modNum(this.x, __comp(v, 0, 1)), __modNum(this.y, __comp(v, 1, 1))); }
  step(e) { return new Vec2(this.x < __comp(e, 0, 0) ? 0 : 1, this.y < __comp(e, 1, 0) ? 0 : 1); }
  smoothStep(a, b) { return new Vec2(__smoothStepNum(__comp(a, 0, 0), __comp(b, 0, 0), this.x), __smoothStepNum(__comp(a, 1, 0), __comp(b, 1, 0), this.y)); }
  fract() { return new Vec2(__fractNum(this.x), __fractNum(this.y)); }
  toString() { return this.x.toFixed(6) + ', ' + this.y.toFixed(6); }
}

class Vec3 {
  constructor(x, y, z) {
    if (arguments.length === 0) { this.x = 0; this.y = 0; this.z = 0; return; }
    if (arguments.length === 1) {
      if (typeof x === 'number') { this.x = x; this.y = x; this.z = x; return; }
      this.x = __comp(x, 0, 0); this.y = __comp(x, 1, 0); this.z = __comp(x, 2, 0); return;
    }
    this.x = +x || 0; this.y = +y || 0; this.z = +z || 0;
  }
  copy() { return new Vec3(this.x, this.y, this.z); }
  clone() { return this.copy(); }
  equals(o) { return o instanceof Vec3 && this.x === o.x && this.y === o.y && this.z === o.z; }
  length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
  lengthSqr() { return this.x * this.x + this.y * this.y + this.z * this.z; }
  normalize() { var l = this.length(); return l < EPS ? new Vec3(0, 0, 0) : new Vec3(this.x / l, this.y / l, this.z / l); }
  add(v) { return new Vec3(this.x + __comp(v, 0, 0), this.y + __comp(v, 1, 0), this.z + __comp(v, 2, 0)); }
  subtract(v) { return new Vec3(this.x - __comp(v, 0, 0), this.y - __comp(v, 1, 0), this.z - __comp(v, 2, 0)); }
  multiply(v) { return new Vec3(this.x * __comp(v, 0, 1), this.y * __comp(v, 1, 1), this.z * __comp(v, 2, 1)); }
  divide(v) { return new Vec3(this.x / __comp(v, 0, 1), this.y / __comp(v, 1, 1), this.z / __comp(v, 2, 1)); }
  dot(v) { return this.x * __comp(v, 0, 0) + this.y * __comp(v, 1, 0) + this.z * __comp(v, 2, 0); }
  cross(v) {
    var vx = __comp(v, 0, 0), vy = __comp(v, 1, 0), vz = __comp(v, 2, 0);
    return new Vec3(this.y * vz - this.z * vy, this.z * vx - this.x * vz, this.x * vy - this.y * vx);
  }
  min(v) { return new Vec3(Math.min(this.x, __comp(v, 0, 0)), Math.min(this.y, __comp(v, 1, 0)), Math.min(this.z, __comp(v, 2, 0))); }
  max(v) { return new Vec3(Math.max(this.x, __comp(v, 0, 0)), Math.max(this.y, __comp(v, 1, 0)), Math.max(this.z, __comp(v, 2, 0))); }
  mix(v, t) { return new Vec3(this.x + (__comp(v, 0, 0) - this.x) * t, this.y + (__comp(v, 1, 0) - this.y) * t, this.z + (__comp(v, 2, 0) - this.z) * t); }
  abs() { return new Vec3(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z)); }
  sign() { return new Vec3(Math.sign(this.x), Math.sign(this.y), Math.sign(this.z)); }
  round() { return new Vec3(Math.round(this.x), Math.round(this.y), Math.round(this.z)); }
  floor() { return new Vec3(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z)); }
  ceil() { return new Vec3(Math.ceil(this.x), Math.ceil(this.y), Math.ceil(this.z)); }
  negate() { return new Vec3(-this.x, -this.y, -this.z); }
  isFinite() { return isFinite(this.x) && isFinite(this.y) && isFinite(this.z); }
  distance(o) { return this.subtract(o).length(); }
  distanceSqr(o) { return this.subtract(o).lengthSqr(); }
  project(v) { var d = new Vec3(v).lengthSqr(); if (d < EPS * EPS) return new Vec3(0, 0, 0); var s = this.dot(v) / d; return new Vec3(__comp(v, 0, 0) * s, __comp(v, 1, 0) * s, __comp(v, 2, 0) * s); }
  reflect(n) { var d = 2 * this.dot(n); return new Vec3(this.x - d * __comp(n, 0, 0), this.y - d * __comp(n, 1, 0), this.z - d * __comp(n, 2, 0)); }
  refract(n, eta) {
    var ni = this.dot(n);
    var k = 1 - eta * eta * (1 - ni * ni);
    if (k < 0) return new Vec3(0, 0, 0);
    var f = eta * ni + Math.sqrt(k);
    return new Vec3(eta * this.x - f * __comp(n, 0, 0), eta * this.y - f * __comp(n, 1, 0), eta * this.z - f * __comp(n, 2, 0));
  }
  angleBetween(v) { var vv = new Vec3(v); var d = this.dot(vv) / (this.length() * vv.length()); return Math.acos(__clampNum(d, -1, 1)) * RAD2DEG; }
  toSpherical() { var r = this.length(); if (r < EPS) return new Vec3(0, 0, 0); return new Vec3(r, Math.acos(__clampNum(this.y / r, -1, 1)) * RAD2DEG, Math.atan2(this.z, this.x) * RAD2DEG); }
  clamp(lo, hi) { return new Vec3(__clampNum(this.x, __comp(lo, 0, 0), __comp(hi, 0, 0)), __clampNum(this.y, __comp(lo, 1, 0), __comp(hi, 1, 0)), __clampNum(this.z, __comp(lo, 2, 0), __comp(hi, 2, 0))); }
  mod(v) { return new Vec3(__modNum(this.x, __comp(v, 0, 1)), __modNum(this.y, __comp(v, 1, 1)), __modNum(this.z, __comp(v, 2, 1))); }
  step(e) { return new Vec3(this.x < __comp(e, 0, 0) ? 0 : 1, this.y < __comp(e, 1, 0) ? 0 : 1, this.z < __comp(e, 2, 0) ? 0 : 1); }
  smoothStep(a, b) { return new Vec3(__smoothStepNum(__comp(a, 0, 0), __comp(b, 0, 0), this.x), __smoothStepNum(__comp(a, 1, 0), __comp(b, 1, 0), this.y), __smoothStepNum(__comp(a, 2, 0), __comp(b, 2, 0), this.z)); }
  fract() { return new Vec3(__fractNum(this.x), __fractNum(this.y), __fractNum(this.z)); }
  toString() { return this.x.toFixed(6) + ', ' + this.y.toFixed(6) + ', ' + this.z.toFixed(6); }
}
Vec3.fromSpherical = function (r, thetaDeg, phiDeg) {
  var t = thetaDeg * DEG2RAD, p = phiDeg * DEG2RAD;
  return new Vec3(r * Math.sin(t) * Math.cos(p), r * Math.cos(t), r * Math.sin(t) * Math.sin(p));
};

class Vec4 {
  constructor(x, y, z, w) {
    if (arguments.length === 0) { this.x = 0; this.y = 0; this.z = 0; this.w = 0; return; }
    if (arguments.length === 1) {
      if (typeof x === 'number') { this.x = x; this.y = x; this.z = x; this.w = x; return; }
      this.x = __comp(x, 0, 0); this.y = __comp(x, 1, 0); this.z = __comp(x, 2, 0); this.w = __comp(x, 3, 0); return;
    }
    this.x = +x || 0; this.y = +y || 0; this.z = +z || 0; this.w = +w || 0;
  }
  copy() { return new Vec4(this.x, this.y, this.z, this.w); }
  clone() { return this.copy(); }
  equals(o) { return o instanceof Vec4 && this.x === o.x && this.y === o.y && this.z === o.z && this.w === o.w; }
  length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w); }
  lengthSqr() { return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w; }
  normalize() { var l = this.length(); return l < EPS ? new Vec4(0, 0, 0, 0) : new Vec4(this.x / l, this.y / l, this.z / l, this.w / l); }
  add(v) { return new Vec4(this.x + __comp(v, 0, 0), this.y + __comp(v, 1, 0), this.z + __comp(v, 2, 0), this.w + __comp(v, 3, 0)); }
  subtract(v) { return new Vec4(this.x - __comp(v, 0, 0), this.y - __comp(v, 1, 0), this.z - __comp(v, 2, 0), this.w - __comp(v, 3, 0)); }
  multiply(v) { return new Vec4(this.x * __comp(v, 0, 1), this.y * __comp(v, 1, 1), this.z * __comp(v, 2, 1), this.w * __comp(v, 3, 1)); }
  divide(v) { return new Vec4(this.x / __comp(v, 0, 1), this.y / __comp(v, 1, 1), this.z / __comp(v, 2, 1), this.w / __comp(v, 3, 1)); }
  dot(v) { return this.x * __comp(v, 0, 0) + this.y * __comp(v, 1, 0) + this.z * __comp(v, 2, 0) + this.w * __comp(v, 3, 0); }
  min(v) { return new Vec4(Math.min(this.x, __comp(v, 0, 0)), Math.min(this.y, __comp(v, 1, 0)), Math.min(this.z, __comp(v, 2, 0)), Math.min(this.w, __comp(v, 3, 0))); }
  max(v) { return new Vec4(Math.max(this.x, __comp(v, 0, 0)), Math.max(this.y, __comp(v, 1, 0)), Math.max(this.z, __comp(v, 2, 0)), Math.max(this.w, __comp(v, 3, 0))); }
  mix(v, t) { return new Vec4(this.x + (__comp(v, 0, 0) - this.x) * t, this.y + (__comp(v, 1, 0) - this.y) * t, this.z + (__comp(v, 2, 0) - this.z) * t, this.w + (__comp(v, 3, 0) - this.w) * t); }
  abs() { return new Vec4(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z), Math.abs(this.w)); }
  negate() { return new Vec4(-this.x, -this.y, -this.z, -this.w); }
  isFinite() { return isFinite(this.x) && isFinite(this.y) && isFinite(this.z) && isFinite(this.w); }
  distance(o) { return this.subtract(o).length(); }
  distanceSqr(o) { return this.subtract(o).lengthSqr(); }
  floor() { return new Vec4(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z), Math.floor(this.w)); }
  ceil() { return new Vec4(Math.ceil(this.x), Math.ceil(this.y), Math.ceil(this.z), Math.ceil(this.w)); }
  round() { return new Vec4(Math.round(this.x), Math.round(this.y), Math.round(this.z), Math.round(this.w)); }
  fract() { return new Vec4(__fractNum(this.x), __fractNum(this.y), __fractNum(this.z), __fractNum(this.w)); }
  clamp(lo, hi) { return new Vec4(__clampNum(this.x, __comp(lo, 0, 0), __comp(hi, 0, 0)), __clampNum(this.y, __comp(lo, 1, 0), __comp(hi, 1, 0)), __clampNum(this.z, __comp(lo, 2, 0), __comp(hi, 2, 0)), __clampNum(this.w, __comp(lo, 3, 0), __comp(hi, 3, 0))); }
  toString() { return this.x.toFixed(6) + ', ' + this.y.toFixed(6) + ', ' + this.z.toFixed(6) + ', ' + this.w.toFixed(6); }
}

globalThis.Vec2 = Vec2;
globalThis.Vec3 = Vec3;
globalThis.Vec4 = Vec4;
// Host-side constructor helper (Rust marshals property vectors through this).
globalThis.__mkVec = function (n, x, y, z, w) { if (n === 2) return new Vec2(x, y); if (n === 3) return new Vec3(x, y, z); return new Vec4(x, y, z, w); };

// ---------------------------------------------------------------------------
// Mat3 / Mat4 — column-major, degrees API (docs §10.2). Minimal but correct
// subset covering the transform-matrix + compose/decompose surface.
// ---------------------------------------------------------------------------

class Mat4 {
  constructor(src) {
    if (src && src.length === 16) { this.m = src.slice(); }
    else { this.m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; }
  }
  static identity() { return new Mat4(); }
  static fromTranslation(v) { var m = new Mat4(); m.m[12] = __comp(v, 0, 0); m.m[13] = __comp(v, 1, 0); m.m[14] = __comp(v, 2, 0); return m; }
  static fromScale(v) { var m = new Mat4(); if (typeof v === 'number') { m.m[0] = v; m.m[5] = v; m.m[10] = v; } else { m.m[0] = __comp(v, 0, 1); m.m[5] = __comp(v, 1, 1); m.m[10] = __comp(v, 2, 1); } return m; }
  static fromRotation(angleDeg, axis) {
    var ax = new Vec3(axis); var l = ax.length(); if (l < EPS) return new Mat4();
    var x = ax.x / l, y = ax.y / l, z = ax.z / l;
    var r = angleDeg * DEG2RAD, c = Math.cos(r), s = Math.sin(r), t = 1 - c;
    var m = new Mat4();
    m.m[0] = t * x * x + c; m.m[1] = t * x * y + s * z; m.m[2] = t * x * z - s * y;
    m.m[4] = t * x * y - s * z; m.m[5] = t * y * y + c; m.m[6] = t * y * z + s * x;
    m.m[8] = t * x * z + s * y; m.m[9] = t * y * z - s * x; m.m[10] = t * z * z + c;
    return m;
  }
  static fromEuler(x, y, z) {
    if (typeof x === 'object') { var v = x; x = __comp(v, 0, 0); y = __comp(v, 1, 0); z = __comp(v, 2, 0); }
    // Rz * Ry * Rx
    return Mat4.fromRotation(z, new Vec3(0, 0, 1))
      .multiply(Mat4.fromRotation(y, new Vec3(0, 1, 0)))
      .multiply(Mat4.fromRotation(x, new Vec3(1, 0, 0)));
  }
  static compose(t, rEuler, s) { return Mat4.fromTranslation(t).multiply(Mat4.fromEuler(rEuler)).multiply(Mat4.fromScale(s)); }
  add(o) { var r = new Mat4(); for (var i = 0; i < 16; i++) r.m[i] = this.m[i] + o.m[i]; return r; }
  subtract(o) { var r = new Mat4(); for (var i = 0; i < 16; i++) r.m[i] = this.m[i] - o.m[i]; return r; }
  multiply(o) {
    if (typeof o === 'number') { var rs = new Mat4(); for (var k = 0; k < 16; k++) rs.m[k] = this.m[k] * o; return rs; }
    if (o instanceof Mat4) {
      var a = this.m, b = o.m, r = new Mat4();
      for (var col = 0; col < 4; col++) {
        for (var row = 0; row < 4; row++) {
          var sum = 0;
          for (var i = 0; i < 4; i++) sum += a[i * 4 + row] * b[col * 4 + i];
          r.m[col * 4 + row] = sum;
        }
      }
      return r;
    }
    // vec4
    var x = __comp(o, 0, 0), y = __comp(o, 1, 0), z = __comp(o, 2, 0), w = __comp(o, 3, 1), mm = this.m;
    return new Vec4(
      mm[0] * x + mm[4] * y + mm[8] * z + mm[12] * w,
      mm[1] * x + mm[5] * y + mm[9] * z + mm[13] * w,
      mm[2] * x + mm[6] * y + mm[10] * z + mm[14] * w,
      mm[3] * x + mm[7] * y + mm[11] * z + mm[15] * w);
  }
  translation(pos) { if (pos !== undefined) { this.m[12] = __comp(pos, 0, 0); this.m[13] = __comp(pos, 1, 0); this.m[14] = __comp(pos, 2, 0); return this; } return new Vec3(this.m[12], this.m[13], this.m[14]); }
  right() { return new Vec3(this.m[0], this.m[1], this.m[2]); }
  up() { return new Vec3(this.m[4], this.m[5], this.m[6]); }
  forward() { return new Vec3(this.m[8], this.m[9], this.m[10]); }
  transformPoint(v) { var r = this.multiply(new Vec4(__comp(v, 0, 0), __comp(v, 1, 0), __comp(v, 2, 0), 1)); var w = r.w || 1; return new Vec3(r.x / w, r.y / w, r.z / w); }
  transformDirection(v) { var r = this.multiply(new Vec4(__comp(v, 0, 0), __comp(v, 1, 0), __comp(v, 2, 0), 0)); return new Vec3(r.x, r.y, r.z); }
  transpose() { var m = this.m, r = new Mat4(); for (var c = 0; c < 4; c++) for (var ro = 0; ro < 4; ro++) r.m[ro * 4 + c] = m[c * 4 + ro]; return r; }
  copy() { return new Mat4(this.m); }
  equals(o) { for (var i = 0; i < 16; i++) if (Math.abs(this.m[i] - o.m[i]) > EPS) return false; return true; }
  toString() { return 'mat4(' + this.m.join(', ') + ')'; }
}

class Mat3 {
  constructor(src) { if (src && src.length === 9) this.m = src.slice(); else this.m = [1, 0, 0, 0, 1, 0, 0, 0, 1]; }
  static identity() { return new Mat3(); }
  static fromMat4(m4) { var m = m4.m; return new Mat3([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]); }
  static fromRotation(angleDeg) { var r = angleDeg * DEG2RAD, c = Math.cos(r), s = Math.sin(r); return new Mat3([c, s, 0, -s, c, 0, 0, 0, 1]); }
  static fromTranslation(v) { var m = new Mat3(); m.m[6] = __comp(v, 0, 0); m.m[7] = __comp(v, 1, 0); return m; }
  static fromScale(v) { var m = new Mat3(); if (typeof v === 'number') { m.m[0] = v; m.m[4] = v; } else { m.m[0] = __comp(v, 0, 1); m.m[4] = __comp(v, 1, 1); } return m; }
  multiply(o) {
    if (o instanceof Mat3) { var a = this.m, b = o.m, r = new Mat3(); for (var col = 0; col < 3; col++) for (var row = 0; row < 3; row++) { var sum = 0; for (var i = 0; i < 3; i++) sum += a[i * 3 + row] * b[col * 3 + i]; r.m[col * 3 + row] = sum; } return r; }
    var x = __comp(o, 0, 0), y = __comp(o, 1, 0), z = __comp(o, 2, 1), mm = this.m;
    return new Vec3(mm[0] * x + mm[3] * y + mm[6] * z, mm[1] * x + mm[4] * y + mm[7] * z, mm[2] * x + mm[5] * y + mm[8] * z);
  }
  transformPoint(v) { var r = this.multiply(new Vec3(__comp(v, 0, 0), __comp(v, 1, 0), 1)); var w = r.z || 1; return new Vec2(r.x / w, r.y / w); }
  copy() { return new Mat3(this.m); }
  toString() { return 'mat3(' + this.m.join(', ') + ')'; }
}

globalThis.Mat4 = Mat4;
globalThis.Mat3 = Mat3;

// ---------------------------------------------------------------------------
// MediaPlaybackEvent (docs §6.3) + localStorage (docs §10.3, in-memory).
// ---------------------------------------------------------------------------
globalThis.MediaPlaybackEvent = { PLAYBACK_STOPPED: 0, PLAYBACK_PLAYING: 1, PLAYBACK_PAUSED: 2 };

(function () {
  var buckets = Object.create(null);
  function bucket(loc) { loc = loc || 'global'; if (!buckets[loc]) buckets[loc] = Object.create(null); return buckets[loc]; }
  globalThis.localStorage = {
    LOCATION_GLOBAL: 'global',
    LOCATION_SCREEN: 'screen',
    set: function (k, v, loc) { bucket(loc)[String(k)] = String(v); },
    get: function (k, loc) { var b = bucket(loc); var key = String(k); return (key in b) ? b[key] : null; },
    delete: function (k, loc) { delete bucket(loc)[String(k)]; },
    remove: function (k, loc) { delete bucket(loc)[String(k)]; },
    clear: function (loc) { if (loc === undefined) { buckets = Object.create(null); } else { buckets[loc] = Object.create(null); } },
  };
})();
