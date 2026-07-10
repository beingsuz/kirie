//! Typed reads over a [`NamedStage`]'s parameter bag.
//!
//! Initializers and operators are modeled scene-side as a `name` plus a
//! `BTreeMap<String, UserSetting<DynamicValue>>` (docs/format-scene-json.md
//! §14.4/§14.5, `kirie_scene::particle::NamedStage`). This module reads those
//! `DynamicValue`s as the scalar / vector types the simulation needs, applying
//! the per-name defaults from the render-architecture tables
//! (docs/render-architecture.md §7.3). Resolution has already folded any
//! property binding into each `UserSetting::value` (§3.2), so we read `.value`.

use kirie_scene::particle::NamedStage;
use kirie_scene::value::{DynamicValue, Vec3};

/// A read-only view over one stage's parameters with defaulted typed accessors.
pub struct Params<'a>(pub &'a NamedStage);

impl<'a> Params<'a> {
    #[must_use]
    pub fn new(stage: &'a NamedStage) -> Self {
        Params(stage)
    }

    fn get(&self, key: &str) -> Option<&DynamicValue> {
        self.0.params.get(key).map(|us| &us.value)
    }

    /// A scalar float, or `default` when absent/`null`.
    #[must_use]
    pub fn f32(&self, key: &str, default: f32) -> f32 {
        match self.get(key) {
            Some(DynamicValue::Null) | None => default,
            Some(v) => v.as_f32(),
        }
    }

    /// An integer, or `default` when absent/`null`.
    #[must_use]
    pub fn i64(&self, key: &str, default: i64) -> i64 {
        match self.get(key) {
            Some(DynamicValue::Null) | None => default,
            Some(v) => v.as_f32() as i64,
        }
    }

    /// A `Vec3`, or `default` when absent/`null`.
    ///
    /// A vector literal takes its first three components (missing → 0). A scalar
    /// literal broadcasts to all three axes, matching the reference's
    /// broadcastable-vec3 fields (docs/format-scene-json.md §14.3).
    #[must_use]
    pub fn vec3(&self, key: &str, default: Vec3) -> Vec3 {
        match self.get(key) {
            Some(DynamicValue::Null) | None => default,
            Some(DynamicValue::Vec(v)) => {
                let c = |i: usize| v.get(i).copied().unwrap_or(0.0);
                [c(0), c(1), c(2)]
            }
            Some(DynamicValue::Color(c)) => [c[0], c[1], c[2]],
            Some(other) => {
                let s = other.as_f32();
                [s, s, s]
            }
        }
    }

    /// A `Vec3` COLOR, normalized to 0..1. Particle color fields (`colorrandom`
    /// min/max, `colorchange`, …) are authored as 0-255 integer triples in the
    /// scene JSON — the reference funnels them through `ColorBuilder`, which
    /// divides an integer-notation color by 255 (`ColorBuilder.cpp:62`,
    /// `it.color("min"/"max")` in `ObjectParser.cpp:783`). kirie decodes the bag
    /// generically as a raw `Vec`, so `"255 0 0"` arrives as `[255,0,0]`; used
    /// as-is it blows every channel past 1.0, saturating the sprite to a uniform
    /// bright blob (the Yellow_Splatter systems all rendered as yellow instead of
    /// the red→yellow `colorrandom` spread). A `Color`-typed literal is already
    /// normalized; a raw `Vec` with any channel > 1 is 0-255 encoding and gets
    /// /255 (a genuine 0..1 float color never exceeds 1, so this is unambiguous).
    #[must_use]
    pub fn color3(&self, key: &str, default: Vec3) -> Vec3 {
        match self.get(key) {
            Some(DynamicValue::Null) | None => default,
            Some(DynamicValue::Color(c)) => [c[0], c[1], c[2]],
            Some(DynamicValue::Vec(v)) => {
                let c = |i: usize| v.get(i).copied().unwrap_or(0.0);
                let raw = [c(0), c(1), c(2)];
                if raw.iter().any(|&x| x > 1.0) {
                    [raw[0] / 255.0, raw[1] / 255.0, raw[2] / 255.0]
                } else {
                    raw
                }
            }
            Some(other) => {
                let s = other.as_f32();
                let s = if s > 1.0 { s / 255.0 } else { s };
                [s, s, s]
            }
        }
    }
}
