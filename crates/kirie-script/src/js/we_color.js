// WEColor module (docs/scripting-api.md §6.6). Import-only. All take one
// {x,y,z} and return a fresh Vec3. Hue is normalized to 0..1 and wraps.
'use strict';
function req(v, name) { if (v == null || typeof v !== 'object') throw new TypeError('WEColor.' + name + ' expects a color object'); }
export function rgb2hsv(v) {
  req(v, 'rgb2hsv');
  const r = v.x || 0, g = v.y || 0, b = v.z || 0;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  let h = 0;
  if (d > 1e-9) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60; if (h < 0) h += 360;
  }
  const s = max <= 0 ? 0 : d / max;
  return new Vec3(h / 360, s, max); // hue in 0..1 (docs §6.6)
}
export function hsv2rgb(v) {
  req(v, 'hsv2rgb');
  let h = v.x || 0; h = h - Math.floor(h); // fractional wrap
  const s = Math.max(0, Math.min(1, v.y || 0)), val = Math.max(0, Math.min(1, v.z || 0));
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = val * (1 - s), q = val * (1 - f * s), t = val * (1 - (1 - f) * s);
  let r, g, b;
  switch (i % 6) {
    case 0: r = val; g = t; b = p; break;
    case 1: r = q; g = val; b = p; break;
    case 2: r = p; g = val; b = t; break;
    case 3: r = p; g = q; b = val; break;
    case 4: r = t; g = p; b = val; break;
    default: r = val; g = p; b = q; break;
  }
  return new Vec3(r, g, b);
}
export function normalizeColor(v) { req(v, 'normalizeColor'); return new Vec3((v.x || 0) / 255, (v.y || 0) / 255, (v.z || 0) / 255); }
export function expandColor(v) { req(v, 'expandColor'); return new Vec3((v.x || 0) * 255, (v.y || 0) * 255, (v.z || 0) * 255); }
