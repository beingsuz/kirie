// WEMath module (docs/scripting-api.md §6.6). Import-only:
//   import * as WEMath from 'WEMath';
'use strict';
export const deg2rad = 0.017453292519943295;
export const rad2deg = 57.29577951308232;
export function smoothStep(edge0, edge1, x) {
  if (typeof edge0 !== 'number' || typeof edge1 !== 'number' || typeof x !== 'number') throw new TypeError('WEMath.smoothStep expects 3 numbers');
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
export function mix(a, b, t) {
  if (typeof a !== 'number' || typeof b !== 'number' || typeof t !== 'number') throw new TypeError('WEMath.mix expects 3 numbers');
  return a * (1 - t) + b * t;
}
