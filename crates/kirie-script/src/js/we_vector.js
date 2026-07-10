// WEVector module (docs/scripting-api.md §6.6). Import-only. Degree conventions.
'use strict';
const RAD2DEG = 57.29577951308232;
const DEG2RAD = 0.017453292519943295;
export function vectorAngle2(v) {
  if (v == null || typeof v !== 'object') throw new TypeError('WEVector.vectorAngle2 expects a vector');
  return Math.atan2(v.y || 0, v.x || 0) * RAD2DEG;
}
export function angleVector2(deg) {
  if (typeof deg !== 'number') throw new TypeError('WEVector.angleVector2 expects a number');
  const r = deg * DEG2RAD;
  return new Vec2(Math.cos(r), Math.sin(r));
}
