//! Property resolution and asset loading → the immutable [`SceneModel`].
//!
//! Spec: docs/format-scene-json.md §3.2 (bindings connect a field to a
//! property), §3.3 (conditional bindings become `property == C`), §9–§11 (asset
//! files loaded by path). [`SceneModel`] is the resolved snapshot the renderer
//! consumes (SPEC.md §V3: `Clone + Send`, serde-serializable for bake).

use serde::{Deserialize, Serialize};

use crate::material::{EffectFile, Material, ModelFile};
use crate::object::{Effect, ImageObject, Object, ObjectKind, ParticleObject};
use crate::particle::ParticleSystem;
use crate::property::{PropertyBag, Resolvable};
use crate::scene::{General, Scene};
use crate::user::{ConstantValues, UserRef, UserSetting};

/// Resolve one user setting against the bag (docs/format-scene-json.md §3.2/§3.3).
///
/// A `Name` binding overwrites the value with the property's value (kept if the
/// property is undeclared); a `Conditional` binding sets the value to the §3.3
/// boolean `property == condition`. Script drivers are left for the runtime.
fn resolve_us<T: Resolvable + Clone>(us: &mut UserSetting<T>, bag: &PropertyBag) {
    match &us.user {
        Some(UserRef::Name(name)) => {
            if let Some(v) = bag.get(name) {
                us.value = T::from_property(v);
            }
        }
        Some(UserRef::Conditional { name, condition }) => {
            let matches = bag
                .get(name)
                .is_some_and(|v| &v.as_condition_string() == condition);
            us.value = T::from_bool(matches);
        }
        None => {}
    }
}

/// Resolve a `constantshadervalues` map in place.
fn resolve_constants(constants: &mut ConstantValues, bag: &PropertyBag) {
    for us in constants.values_mut() {
        resolve_us(us, bag);
    }
}

impl General {
    /// Resolve every property-bound `general` field against the bag.
    fn resolve(&mut self, bag: &PropertyBag) {
        resolve_us(&mut self.ambientcolor, bag);
        resolve_us(&mut self.skylightcolor, bag);
        resolve_us(&mut self.clearcolor, bag);
        resolve_us(&mut self.camerafade, bag);
        resolve_us(&mut self.bloom, bag);
        resolve_us(&mut self.bloomstrength, bag);
        resolve_us(&mut self.bloomthreshold, bag);
        resolve_us(&mut self.cameraparallax, bag);
        resolve_us(&mut self.cameraparallaxamount, bag);
        resolve_us(&mut self.cameraparallaxdelay, bag);
        resolve_us(&mut self.cameraparallaxmouseinfluence, bag);
        resolve_us(&mut self.camerashake, bag);
        resolve_us(&mut self.camerashakeamplitude, bag);
        resolve_us(&mut self.camerashakeroughness, bag);
        resolve_us(&mut self.camerashakespeed, bag);
    }
}

impl Object {
    /// Resolve every property-bound field of this object against the bag.
    fn resolve(&mut self, bag: &PropertyBag) {
        resolve_us(&mut self.base.origin, bag);
        resolve_us(&mut self.base.scale, bag);
        resolve_us(&mut self.base.angles, bag);
        resolve_us(&mut self.base.visible, bag);
        match &mut self.kind {
            ObjectKind::Image(img) => img.resolve(bag),
            ObjectKind::Particle(p) => p.resolve(bag),
            ObjectKind::Text(t) => {
                resolve_us(&mut t.text, bag);
                resolve_us(&mut t.pointsize, bag);
                resolve_us(&mut t.scale, bag);
                resolve_us(&mut t.color, bag);
                resolve_us(&mut t.alpha, bag);
                resolve_us(&mut t.visible, bag);
            }
            ObjectKind::Sound(_)
            | ObjectKind::Model(_)
            | ObjectKind::Light(_)
            | ObjectKind::Shape(_)
            | ObjectKind::Group => {}
        }
    }
}

impl ImageObject {
    fn resolve(&mut self, bag: &PropertyBag) {
        resolve_us(&mut self.scale, bag);
        resolve_us(&mut self.angles, bag);
        resolve_us(&mut self.visible, bag);
        resolve_us(&mut self.alpha, bag);
        resolve_us(&mut self.color, bag);
        resolve_us(&mut self.parallax_depth, bag);
        resolve_us(&mut self.color_blend_mode, bag);
        resolve_us(&mut self.brightness, bag);
        if let Some(material) = &mut self.material {
            resolve_material(material, bag);
        }
        for effect in &mut self.effects {
            effect.resolve(bag);
        }
        for layer in &mut self.animationlayers {
            resolve_us(&mut layer.rate, bag);
            resolve_us(&mut layer.visible, bag);
            resolve_us(&mut layer.blend, bag);
            resolve_us(&mut layer.animation, bag);
        }
    }
}

impl Effect {
    fn resolve(&mut self, bag: &PropertyBag) {
        resolve_us(&mut self.visible, bag);
        for pass in &mut self.passes {
            resolve_constants(&mut pass.constantshadervalues, bag);
        }
        if let Some(file) = &mut self.resolved {
            for pass in &mut file.passes {
                if let Some(material) = &mut pass.resolved {
                    resolve_material(material, bag);
                }
            }
        }
    }
}

impl ParticleObject {
    fn resolve(&mut self, bag: &PropertyBag) {
        resolve_us(&mut self.scale, bag);
        resolve_us(&mut self.angles, bag);
        resolve_us(&mut self.visible, bag);
        resolve_us(&mut self.parallax_depth, bag);
        let ov = &mut self.instanceoverride;
        resolve_us(&mut ov.enabled, bag);
        resolve_us(&mut ov.alpha, bag);
        resolve_us(&mut ov.size, bag);
        resolve_us(&mut ov.lifetime, bag);
        resolve_us(&mut ov.rate, bag);
        resolve_us(&mut ov.speed, bag);
        resolve_us(&mut ov.count, bag);
        resolve_us(&mut ov.color, bag);
        resolve_us(&mut ov.colorn, bag);
        for stage in self
            .system
            .initializers
            .iter_mut()
            .chain(&mut self.system.operators)
        {
            resolve_constants(&mut stage.params, bag);
        }
        if let Some(material) = &mut self.system.resolved_material {
            resolve_material(material, bag);
        }
    }
}

/// Resolve every pass's `constantshadervalues` in a material.
fn resolve_material(material: &mut Material, bag: &PropertyBag) {
    for pass in &mut material.passes {
        resolve_constants(&mut pass.constantshadervalues, bag);
    }
}

/// A source of asset-file bytes by relative path (pkg entry, loose dir, …).
pub trait AssetSource {
    /// Load the raw bytes of `path`, or `None` if it is not available.
    fn load(&self, path: &str) -> Option<Vec<u8>>;
}

impl<F: Fn(&str) -> Option<Vec<u8>>> AssetSource for F {
    fn load(&self, path: &str) -> Option<Vec<u8>> {
        self(path)
    }
}

/// An asset that could not be loaded or parsed during resolution
/// (docs/format-scene-json.md §9–§11).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetProblem {
    /// The referenced asset path.
    pub path: String,
    /// Why it failed (missing, invalid JSON, required key absent).
    pub reason: String,
}

/// Load and parse a JSON asset, recording a problem on failure.
fn load_json(
    source: &dyn AssetSource,
    path: &str,
    problems: &mut Vec<AssetProblem>,
) -> Option<serde_json::Value> {
    let Some(bytes) = source.load(path) else {
        problems.push(AssetProblem {
            path: path.to_owned(),
            reason: "asset not found".to_owned(),
        });
        return None;
    };
    match serde_json::from_slice(&bytes) {
        Ok(v) => Some(v),
        Err(e) => {
            problems.push(AssetProblem {
                path: path.to_owned(),
                reason: format!("invalid JSON: {e}"),
            });
            None
        }
    }
}

/// The resolved, immutable scene snapshot the renderer consumes.
///
/// Holds a fully-resolved [`Scene`] (property bindings collapsed, asset files
/// loaded) and the [`PropertyBag`] it resolved against so a later `setProperty`
/// can produce a fresh snapshot. `Clone + Send + Serialize` per SPEC.md §V3.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneModel {
    /// The resolved scene graph.
    pub scene: Scene,
}

impl SceneModel {
    /// Resolve a parsed [`Scene`] against a [`PropertyBag`]
    /// (docs/format-scene-json.md §3.2). Collapses every property binding into
    /// its current value; asset files are loaded separately via
    /// [`SceneModel::load_assets`].
    pub fn resolve(mut scene: Scene, bag: &PropertyBag) -> Self {
        scene.general.resolve(bag);
        for object in &mut scene.objects {
            object.resolve(bag);
        }
        SceneModel { scene }
    }

    /// Load every referenced material / effect / model / particle file from
    /// `source`, filling the `resolved` slots (docs/format-scene-json.md
    /// §9–§11, §14.2), then re-resolve the newly-loaded materials against
    /// `bag`. Returns the assets that could not be loaded/parsed (empty on a
    /// fully-resolvable scene). Builtin shared assets (WE `assets/`) are not
    /// referenced by these JSON files, so a self-contained pkg resolves fully.
    pub fn load_assets(&mut self, source: &dyn AssetSource, bag: &PropertyBag) -> Vec<AssetProblem> {
        let mut problems = Vec::new();
        for object in &mut self.scene.objects {
            match &mut object.kind {
                ObjectKind::Image(img) => load_image_assets(img, source, &mut problems),
                ObjectKind::Particle(p) => load_particle_assets(p, source, &mut problems),
                _ => {}
            }
        }
        // Newly-loaded materials carry their own constantshadervalues bindings.
        for object in &mut self.scene.objects {
            object.resolve(bag);
        }
        problems
    }
}

/// Load an image object's model file, its material, and every effect file/pass
/// material (docs/format-scene-json.md §8/§9/§10/§11).
fn load_image_assets(img: &mut ImageObject, source: &dyn AssetSource, problems: &mut Vec<AssetProblem>) {
    if let Some(value) = load_json(source, &img.image, problems) {
        match ModelFile::from_value(&value) {
            Ok(model) => {
                if let Some(value) = load_json(source, &model.material, problems) {
                    img.material = Some(Material::from_value(&value));
                }
                img.model = Some(model);
            }
            Err(e) => problems.push(AssetProblem {
                path: img.image.clone(),
                reason: e.to_string(),
            }),
        }
    }
    for effect in &mut img.effects {
        if let Some(value) = load_json(source, &effect.file, problems) {
            let mut file = EffectFile::from_value(&value);
            for pass in &mut file.passes {
                if let Some(mat_path) = pass.material.clone()
                    && let Some(mat_value) = load_json(source, &mat_path, problems)
                {
                    pass.resolved = Some(Material::from_value(&mat_value));
                }
            }
            effect.resolved = Some(file);
        }
    }
}

/// Load a particle object's external definition (if any) and its material
/// (docs/format-scene-json.md §14.2).
fn load_particle_assets(p: &mut ParticleObject, source: &dyn AssetSource, problems: &mut Vec<AssetProblem>) {
    if let Some(path) = p.particle_file.clone()
        && let Some(value) = load_json(source, &path, problems)
    {
        p.system = ParticleSystem::from_value(&value);
    }
    if let Some(mat_path) = p.system.material.clone()
        && let Some(value) = load_json(source, &mat_path, problems)
    {
        p.system.resolved_material = Some(Material::from_value(&value));
    }
}
