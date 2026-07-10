//! The per-image pass list and its ping-pong FBO wiring (the pure planning of
//! docs/render-architecture.md §7.1, `CImage::setup`/`setupPasses`).
//!
//! This computes the *order* and *routing* of an image's draws without touching
//! the GPU, so the chain topology is unit-testable. Given the base material and
//! the visible effects it produces an ordered [`PlanPass`] list where each pass
//! knows its input texture (the previous output, or the layer texture for the
//! first pass), its render target (a ping-pong image FBO or the scene FBO for
//! the last visible pass), and its geometry role (copy-space, pass-space, or
//! scene-space — §7.1's `copySpacePosition`/`passSpacePosition`/
//! `sceneSpacePosition`).
//!
//! Effect per-pass FBO routing IS modeled (docs §11.2): each pass carries its
//! `target` scratch-FBO name and `bind` sources, and the effects' declared
//! `fbos` are surfaced on the plan, so a combine pass samples the composite
//! (`_rt_imageLayerComposite_<id>_a/_b`) and its own scratch buffers
//! (`_rt_HalfCompoBuffer*`) instead of the 1×1 white default. The renderer
//! allocates the named FBOs and threads the composite front (§11.2).
//!
//! Simplifications vs the reference, each a documented seam (SPEC.md §V10):
//! effect `command:"copy"`/swap buffer commands are not yet modeled — an effect
//! pass with no material is skipped with a trace note. Puppet meshes and
//! `colorBlendMode` extra passes are likewise deferred to the renderer's
//! object-skip path.

use kirie_scene::material::{Blending, CullMode, DepthMode, Pass};
use kirie_scene::object::{ImageObject, PassOverride};

/// Where a planned pass reads its `g_Texture0` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassInput {
    /// The image's base layer texture (the first pass).
    Layer,
    /// One of the two ping-pong image FBOs (`_rt_imageLayerComposite_<id>_a/_b`).
    Fbo(usize),
}

/// Where a planned pass renders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassOutput {
    /// A ping-pong image FBO (index 0 or 1).
    Fbo(usize),
    /// A per-effect scratch FBO by declared name (§11.2 `target`).
    Named(String),
    /// The scene FBO — the composite-into-scene draw (the last visible pass).
    Scene,
}

/// The geometry + MVP role of a pass (docs/render-architecture.md §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// `copySpacePosition` + copy matrices — first pass (texture → layer FBO).
    Copy,
    /// `passSpacePosition` + identity — intermediate effect passes.
    Pass,
    /// `sceneSpacePosition` + screen MVP — composite into the scene FBO.
    Scene,
    /// A puppet-warp deformable mesh drawn straight into the scene FBO in
    /// scene space (screen MVP) — the single-pass puppet character path
    /// (`CImage::setupPasses` `m_hasPuppetMesh`, `CImage.cpp:823-834`). The
    /// vertices/indices come from the puppet `.mdl`, not the 4-vertex quad.
    Puppet,
    /// A puppet-warp deformable mesh drawn into the image's own copy FBO in
    /// local `[0..size]` space (the ortho model matrix as MVP) — the first
    /// pass of a *multi-pass* puppet character, so the effect chain then
    /// processes the deformed character exactly as the reference does
    /// (`CImage.cpp:823-825`, `m_modelViewProjectionCopy`).
    PuppetCopy,
}

/// One fully-resolved, wired pass ready for pipeline creation and drawing.
#[derive(Debug, Clone)]
pub struct PlanPass {
    /// The pass's shader base name (`shaders/<name>.vert/.frag`).
    pub shader: String,
    /// Blending after §7.1 relocation.
    pub blending: Blending,
    /// Cull mode.
    pub cull: CullMode,
    /// Depth test.
    pub depthtest: DepthMode,
    /// Depth write.
    pub depthwrite: DepthMode,
    /// The merged material pass (combos/textures/constants, override applied).
    pub pass: Pass,
    /// Texture input source.
    pub input: PassInput,
    /// Render target.
    pub output: PassOutput,
    /// Geometry / MVP role.
    pub geometry: Geometry,
    /// The effect pass's own render `target` FBO name (docs/format-scene-json.md
    /// §11.2), or `None` for the base material and effect passes that render
    /// back into the image composite ping-pong. A named target routes the draw
    /// into a per-effect scratch FBO (e.g. `_rt_HalfCompoBuffer1`) instead of the
    /// composite, so the composite keeps the pre-effect image for a later
    /// combine pass to sample as `_rt_imageLayerComposite_<id>_a/_b` (§11.2,
    /// `CImage.cpp`/`CPass.cpp` FBO routing — the reference never overwrites the
    /// composite with an effect's intermediate work).
    pub target: Option<String>,
    /// The effect pass's `bind` entries — `(slot, source-name)` where the source
    /// is `previous` (the composite front), a named effect FBO, or an
    /// `_rt_imageLayerComposite_<id>_*` composite reference (§11.2, `CPass.cpp`
    /// getInput/"previous"). Filled into empty texture slots by the renderer so
    /// each `g_TextureN` samples the right prior output instead of the 1×1 white.
    pub binds: Vec<(u32, String)>,
}

/// The planned draw chain for one image object.
#[derive(Debug, Clone, Default)]
pub struct ImagePlan {
    /// Passes in draw order (empty ⇒ the image is skipped, §7.1 early-out).
    pub passes: Vec<PlanPass>,
    /// The union of the visible effects' declared scratch FBOs (§11.2 `fbos`),
    /// allocated per object and referenced by pass `target`/`binds`.
    pub named_fbos: Vec<kirie_scene::material::Fbo>,
}

/// Merge a per-position effect [`PassOverride`] onto a base material [`Pass`]
/// (docs/render-architecture.md §8.1, §8.3 priority: override wins).
fn apply_override(mut pass: Pass, ov: &PassOverride) -> Pass {
    for (k, v) in &ov.combos {
        pass.combos.insert(k.clone(), *v);
    }
    for (k, v) in &ov.constantshadervalues {
        pass.constantshadervalues.insert(k.clone(), v.clone());
    }
    // Texture-slot overrides win by index where present and non-empty.
    for (i, slot) in ov.textures.iter().enumerate() {
        if slot.is_some() {
            if i >= pass.textures.len() {
                pass.textures.resize(i + 1, None);
            }
            pass.textures[i] = slot.clone();
        }
    }
    pass
}

/// A material pass paired with its effect FBO routing (`target`/`bind`); the
/// base material carries `None`/empty (it renders into the composite).
struct SrcPass {
    pass: Pass,
    target: Option<String>,
    binds: Vec<(u32, String)>,
}

/// The base material's passes (the layer's own draw, before any effect).
fn base_passes(image: &ImageObject) -> Vec<SrcPass> {
    image
        .material
        .as_ref()
        .map(|m| {
            m.passes
                .iter()
                .cloned()
                .map(|pass| SrcPass {
                    pass,
                    target: None,
                    binds: Vec::new(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The passes contributed by the image's *visible* effects, in effect order,
/// each with its per-position [`PassOverride`] applied plus its effect-file
/// `target`/`bind` routing preserved (docs §7.1, §8.1, §11.2).
fn effect_passes(image: &ImageObject) -> Vec<SrcPass> {
    let mut out = Vec::new();
    for effect in &image.effects {
        if !effect.visible.value {
            continue; // §7.1: `visible:false` effects skipped.
        }
        let Some(file) = &effect.resolved else { continue };
        for (i, epass) in file.passes.iter().enumerate() {
            let Some(mat) = &epass.resolved else {
                if epass.command.is_some() {
                    tracing::debug!(effect = %effect.file, "effect command pass not yet modeled; skipped");
                }
                continue;
            };
            let ov = effect.passes.get(i);
            // The effect-file `bind`s route named FBOs / the composite `previous`
            // into texture slots (§11.2). Only the first material pass of a multi
            // -pass material inherits them (the reference binds per effect pass).
            let binds: Vec<(u32, String)> = epass
                .bind
                .iter()
                .filter_map(|b| u32::try_from(b.index).ok().map(|i| (i, b.name.clone())))
                .collect();
            for (mi, mpass) in mat.passes.iter().enumerate() {
                out.push(SrcPass {
                    pass: match ov {
                        Some(o) => apply_override(mpass.clone(), o),
                        None => mpass.clone(),
                    },
                    target: epass.target.clone(),
                    binds: if mi == 0 { binds.clone() } else { Vec::new() },
                });
            }
        }
    }
    out
}

/// The union of the visible effects' declared scratch FBOs (§11.2 `fbos`).
fn effect_fbos(image: &ImageObject) -> Vec<kirie_scene::material::Fbo> {
    let mut out: Vec<kirie_scene::material::Fbo> = Vec::new();
    for effect in &image.effects {
        if !effect.visible.value {
            continue;
        }
        let Some(file) = &effect.resolved else { continue };
        for fbo in &file.fbos {
            if !out.iter().any(|f| f.name == fbo.name) {
                out.push(fbo.clone());
            }
        }
    }
    out
}

/// Build the draw plan for an image (docs/render-architecture.md §7.1).
///
/// `visible` is the image's resolved visibility (a hidden image still plans
/// nothing). `passthrough` is the model's `passthrough` flag: a passthrough
/// image whose passes are all trivial is the §7.1 early-out.
#[must_use]
pub fn plan_image(image: &ImageObject, visible: bool) -> ImagePlan {
    if !visible {
        return ImagePlan::default();
    }
    // §7.1 passthrough early-out (`CImage.cpp:606-624`): a `passthrough` layer
    // whose visible effects contribute no passes is an identity copy of the
    // scene FBO onto itself — the reference skips it entirely. Rendering it
    // anyway samples `_rt_FullFrameBuffer` and blits the scene straight back,
    // which (before the scene-snapshot wiring) composited a solid block. These
    // compose/project/fullscreen util layers exist only to *host* effects.
    let passthrough = image.model.as_ref().is_some_and(|m| m.passthrough);
    let effects = effect_passes(image);
    if passthrough && effects.is_empty() {
        return ImagePlan::default();
    }
    let mut passes = base_passes(image);
    passes.extend(effects);
    if passes.is_empty() {
        return ImagePlan::default();
    }

    // §7.1 blend-mode relocation: with >1 pass, the first pass's blending moves
    // to the last pass and the first becomes Normal (layer blending happens
    // when compositing into the scene, not when copying into the layer FBO).
    //
    // Exception — puppet-mesh base: a flat-quad copy pass writes each destination
    // texel exactly once, so Normal (replace) into the transparent layer FBO is
    // correct. A puppet base instead draws an *indexed mesh whose triangles
    // overlap*, and Normal blending makes a later transparent-margin triangle
    // REPLACE an already-opaque texel with alpha 0 — punching holes (the girl 女's
    // eye socket, which then let the LOGO layer bleed through as a red mark). The
    // mesh must composite over itself, so a puppet base keeps its translucent
    // blend (paired with blend.rs's coverage-correct alpha factor). The relocated
    // layer blend still lands on the last pass for the scene composite.
    if passes.len() > 1 {
        let first_blend = passes[0].pass.blending;
        let puppet_base = image.model.as_ref().is_some_and(|m| m.puppet.is_some());
        if !puppet_base {
            passes[0].pass.blending = Blending::Normal;
        }
        let last = passes.len() - 1;
        passes[last].pass.blending = first_blend;
    }

    // Wire inputs/outputs: ping-pong across two image FBOs, last pass → scene.
    // These composite fields are the linear fallback (`target: None` passes); the
    // renderer re-derives the true routing from `target`/`binds` so effect
    // scratch passes (a named `target`) render into per-effect FBOs and keep the
    // composite intact for a combine pass (docs §11.2).
    let n = passes.len();
    let mut wired = Vec::with_capacity(n);
    let mut input = PassInput::Layer;
    let mut cur_out = 0usize;
    for (i, src) in passes.into_iter().enumerate() {
        let is_last = i == n - 1;
        let (output, geometry) = if is_last {
            (PassOutput::Scene, Geometry::Scene)
        } else if i == 0 {
            (PassOutput::Fbo(cur_out), Geometry::Copy)
        } else {
            (PassOutput::Fbo(cur_out), Geometry::Pass)
        };
        // A single visible pass composites straight into the scene from the
        // layer texture (copy-space geometry keeps the correct MVP path).
        let geometry = if n == 1 { Geometry::Scene } else { geometry };
        let SrcPass { pass, target, binds } = src;
        wired.push(PlanPass {
            shader: pass.shader.clone(),
            blending: pass.blending,
            cull: pass.cullmode,
            depthtest: pass.depthtest,
            depthwrite: pass.depthwrite,
            pass,
            input,
            output,
            geometry,
            target,
            binds,
        });
        if !is_last {
            input = PassInput::Fbo(cur_out);
            cur_out = 1 - cur_out;
        }
    }
    ImagePlan {
        passes: wired,
        named_fbos: effect_fbos(image),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kirie_scene::material::Material;
    use kirie_scene::object::ImageObject;
    use kirie_scene::user::UserSetting;
    use kirie_scene::value::WHITE;

    fn pass(shader: &str, blending: Blending) -> Pass {
        Pass {
            blending,
            cullmode: CullMode::NoCull,
            depthtest: DepthMode::Disabled,
            depthwrite: DepthMode::Disabled,
            shader: shader.to_string(),
            textures: vec![],
            usertextures: vec![],
            combos: Default::default(),
            constantshadervalues: Default::default(),
        }
    }

    fn image(passes: Vec<Pass>) -> ImageObject {
        ImageObject {
            image: "img.json".into(),
            model: None,
            material: Some(Material { passes }),
            scale: UserSetting::literal([1.0, 1.0, 1.0]),
            angles: UserSetting::literal([0.0, 0.0, 0.0]),
            visible: UserSetting::literal(true),
            alpha: UserSetting::literal(1.0),
            color: UserSetting::literal(WHITE),
            alignment: "center".into(),
            size: [0.0, 0.0],
            parallax_depth: UserSetting::literal([0.0, 0.0]),
            color_blend_mode: UserSetting::literal(0),
            brightness: UserSetting::literal(1.0),
            effects: vec![],
            animationlayers: vec![],
            instance: None,
        }
    }

    fn passthrough_model() -> kirie_scene::material::ModelFile {
        kirie_scene::material::ModelFile {
            material: "materials/util/fullscreenlayer.json".into(),
            solidlayer: false,
            fullscreen: true,
            passthrough: true,
            autosize: false,
            nopadding: false,
            width: None,
            height: None,
            puppet: None,
        }
    }

    #[test]
    fn hidden_image_plans_nothing() {
        let img = image(vec![pass("effect", Blending::Normal)]);
        assert!(plan_image(&img, false).passes.is_empty());
    }

    #[test]
    fn passthrough_layer_without_visible_effects_is_skipped() {
        // §7.1 early-out: a `passthrough` util layer (compose/fullscreen) whose
        // base pass just samples `_rt_FullFrameBuffer` and copies it back is an
        // identity — the reference never renders it (would blit a solid block).
        let mut img = image(vec![pass("passthrough", Blending::Translucent)]);
        img.model = Some(passthrough_model());
        assert!(plan_image(&img, true).passes.is_empty());
    }

    #[test]
    fn passthrough_layer_with_visible_effect_renders() {
        // The same util layer *with* a visible effect hosts real work — it must
        // render: base pass reads the scene, the effect pass composites back.
        use kirie_scene::material::{EffectFile, EffectPass};
        use kirie_scene::object::Effect;
        use kirie_scene::user::UserSetting;
        let mut img = image(vec![pass("passthrough", Blending::Translucent)]);
        img.model = Some(passthrough_model());
        img.effects = vec![Effect {
            file: "effects/tint/effect.json".into(),
            id: -1,
            name: "tint".into(),
            visible: UserSetting::literal(true),
            passes: vec![],
            resolved: Some(EffectFile {
                name: String::new(),
                description: String::new(),
                group: String::new(),
                preview: String::new(),
                dependencies: vec![],
                fbos: vec![],
                passes: vec![EffectPass {
                    material: Some("materials/effects/tint.json".into()),
                    resolved: Some(Material {
                        passes: vec![pass("effects/tint", Blending::Normal)],
                    }),
                    bind: vec![],
                    command: None,
                    source: None,
                    target: None,
                }],
            }),
        }];
        assert_eq!(plan_image(&img, true).passes.len(), 2);
    }

    #[test]
    fn image_without_material_plans_nothing() {
        let mut img = image(vec![]);
        img.material = None;
        assert!(plan_image(&img, true).passes.is_empty());
    }

    #[test]
    fn single_pass_composites_into_scene_from_layer() {
        let img = image(vec![pass("passthrough", Blending::Translucent)]);
        let plan = plan_image(&img, true);
        assert_eq!(plan.passes.len(), 1);
        let p = &plan.passes[0];
        assert_eq!(p.input, PassInput::Layer);
        assert_eq!(p.output, PassOutput::Scene);
        assert_eq!(p.geometry, Geometry::Scene);
        // A single pass keeps its blending (no relocation happens).
        assert_eq!(p.blending, Blending::Translucent);
    }

    #[test]
    fn multi_pass_ping_pongs_and_ends_at_scene() {
        let img = image(vec![
            pass("a", Blending::Additive),
            pass("b", Blending::Normal),
            pass("c", Blending::Normal),
        ]);
        let plan = plan_image(&img, true);
        assert_eq!(plan.passes.len(), 3);

        // First pass reads the layer, copy-space, writes fbo 0.
        assert_eq!(plan.passes[0].input, PassInput::Layer);
        assert_eq!(plan.passes[0].output, PassOutput::Fbo(0));
        assert_eq!(plan.passes[0].geometry, Geometry::Copy);

        // Middle pass reads fbo 0, pass-space, writes fbo 1.
        assert_eq!(plan.passes[1].input, PassInput::Fbo(0));
        assert_eq!(plan.passes[1].output, PassOutput::Fbo(1));
        assert_eq!(plan.passes[1].geometry, Geometry::Pass);

        // Last pass reads fbo 1 and composites into the scene.
        assert_eq!(plan.passes[2].input, PassInput::Fbo(1));
        assert_eq!(plan.passes[2].output, PassOutput::Scene);
        assert_eq!(plan.passes[2].geometry, Geometry::Scene);
    }

    #[test]
    fn blend_relocation_moves_first_to_last() {
        // docs/render-architecture.md §7.1: first pass's blending moves to the
        // last; the first becomes Normal.
        let img = image(vec![
            pass("a", Blending::Additive),
            pass("b", Blending::Translucent),
        ]);
        let plan = plan_image(&img, true);
        assert_eq!(plan.passes[0].blending, Blending::Normal, "first forced Normal");
        assert_eq!(
            plan.passes[1].blending,
            Blending::Additive,
            "first's blend relocated to last"
        );
    }
}
