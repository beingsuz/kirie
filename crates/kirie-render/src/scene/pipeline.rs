//! Build a wgpu render pipeline for one pass from [`kirie_shader`]-translated
//! vertex + fragment modules (docs/render-architecture.md §8, §8.5;
//! docs/shader-pipeline.md §9).
//!
//! The shader crate translates each stage independently, and each assigns its
//! own `set = 0` binding numbers (binding 0 = the `_WEGlobals` UBO, samplers
//! from binding 1). Two stages therefore both claim `@group(0) @binding(0)`,
//! which wgpu would merge into a single shared buffer — wrong, since the vertex
//! globals (MVP matrices) and fragment globals (colors/time) differ. So the
//! fragment module's resources are **remapped to group 1**; the pipeline layout
//! is then `[group0 = vertex, group1 = fragment]`, each with its own globals
//! UBO and samplers. Binding numbers within a stage are preserved.
//!
//! Only `a_Position`/`a_TexCoord` vertex attributes are supported (the 2D image
//! layers, `CPass.cpp:715-718`); a pass whose vertex shader declares any other
//! attribute (3D models) is rejected so the renderer skips it (SPEC.md §V9).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use kirie_scene::material::{Blending, CullMode, DepthMode, Pass};
use kirie_shader::reflect::{Parameter, Reflection, SamplerSlot};
use kirie_shader::{IncludeResolver, ShaderInputs, Stage, TranslateError, translate};

use super::blend;
use super::uniforms::{GlType, GlobalsLayout, Member, builtin_type};

/// Why a pass pipeline could not be built (SPEC.md §V9 — a skippable, typed
/// failure, never a panic). `build_object` logs the reason and drops the pass.
#[derive(Debug, thiserror::Error)]
pub enum PassBuildError {
    /// A stage's GLSL failed to translate/validate (docs/shader-pipeline.md §9).
    #[error(transparent)]
    Translate(#[from] TranslateError),
    /// The fragment stage reads an inter-stage `@location` the vertex stage does
    /// not write. GL tolerates this (undefined varying reads); wgpu rejects the
    /// pipeline, so the pass is skipped rather than crashing the wallpaper
    /// (docs/render-architecture.md §8.5, the strict stage-interface rule).
    #[error("VS/FS interface mismatch: fragment reads location {0} the vertex stage does not write")]
    InterfaceMismatch(u32),
    /// An inter-stage `@location` exceeds the device's
    /// `max_inter_stage_shader_variables` limit. WE shaders (e.g. audio
    /// oscilloscopes) can declare more varyings than a downlevel GL/Vulkan
    /// target allows; wgpu makes this a *fatal* pipeline error, so the pass is
    /// skipped rather than crashing the wallpaper (SPEC.md §V9).
    #[error("inter-stage location {0} exceeds the device limit ({1})")]
    TooManyVaryings(u32, u32),
}

/// A compiled pass pipeline plus everything needed to bind it each frame.
pub struct BuiltPass {
    /// The render pipeline.
    pub pipeline: wgpu::RenderPipeline,
    /// Group 0 (vertex-stage) bind-group layout.
    pub g0_layout: wgpu::BindGroupLayout,
    /// Group 1 (fragment-stage) bind-group layout.
    pub g1_layout: wgpu::BindGroupLayout,
    /// Vertex `_WEGlobals` std140 layout (may be empty).
    pub vs_globals: GlobalsLayout,
    /// Fragment `_WEGlobals` std140 layout (may be empty).
    pub fs_globals: GlobalsLayout,
    /// Vertex sampler slots (usually none for 2D layers).
    pub vs_samplers: Vec<SamplerSlot>,
    /// Fragment sampler slots.
    pub fs_samplers: Vec<SamplerSlot>,
    /// The vertex module's actual `group 0` resource bindings (ground truth for
    /// the bind group — a superset of `vs_samplers` when the shader declares
    /// un-annotated samplers).
    pub g0_bindings: Vec<ModuleBinding>,
    /// The fragment module's actual `group 1` resource bindings.
    pub g1_bindings: Vec<ModuleBinding>,
    /// Vertex material parameters (for resolving global values).
    pub vs_params: Vec<Parameter>,
    /// Fragment material parameters.
    pub fs_params: Vec<Parameter>,
    /// `@location` of `a_Position`.
    pub pos_location: u32,
    /// `@location` of `a_TexCoord` (position stays at [`Self::pos_location`]).
    pub uv_location: Option<u32>,
}

/// The interleaved vertex layout the renderer uploads: `pos.xyz` then `uv.xy`.
pub const VERTEX_STRIDE: u64 = 20; // 3*4 + 2*4

/// Build a pass pipeline. `vs_src`/`fs_src` are the raw `.vert`/`.frag` sources;
/// `resolver` supplies `#include` bodies; `pass` supplies combos/textures;
/// `target_format` is the render-target format (the scene/effect FBO format).
/// `depth` is the optional depth-stencil state (`None` for 2D layers).
#[allow(clippy::too_many_arguments)]
pub fn build_pass(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    blending: Blending,
    cull: CullMode,
    depthtest: DepthMode,
    depthwrite: DepthMode,
    topology: wgpu::PrimitiveTopology,
    pass: &Pass,
    vs_src: &str,
    fs_src: &str,
    resolver: &dyn IncludeResolver,
) -> Result<BuiltPass, PassBuildError> {
    let base_inputs = shader_inputs(pass);
    // Compat shim (docs/shader-pipeline.md §9.2): WE effect shaders authored
    // against lenient desktop GL sometimes use constructs the strict wgpu
    // translation rejects, which would skip the whole pass and blank the effect
    // (SPEC.md §V9). Normalize the two seen in the corpus before translating:
    // a `const` function-return qualifier (illegal GLSL, silently accepted by
    // some drivers) and a fragment-stage `gl_Position` read (undefined in GL,
    // hard-rejected by naga) — see [`sanitize_glsl`].
    let vs_src = sanitize_glsl(vs_src, Stage::Vertex);
    let fs_src = sanitize_glsl(fs_src, Stage::Fragment);

    // First attempt: translate each stage independently, exactly as before. The
    // overwhelming majority of corpus passes resolve identical combos and declare
    // their varyings in the same order in both stages, so this path is unchanged
    // for them — the reconciliation below is a *fallback* that never runs for a
    // pass that already links cleanly, so working scenes are untouched.
    let mut vs = translate(Stage::Vertex, "pass.vert", &vs_src, resolver, &base_inputs)?;
    let mut fs = translate(Stage::Fragment, "pass.frag", &fs_src, resolver, &base_inputs)?;

    // wgpu links VS→FS by `@location`, not by name; the two stages are translated
    // independently, so they can disagree when a fragment reads a varying the
    // vertex never writes. Two corpus sources of that disagreement bite the 2D
    // effect shaders (both exemplified by `effects/blend`):
    //
    // 1. Per-stage combo resolution. A sampler annotation `combo:"X"` with a
    //    `default` texture force-enables X in whichever stage *declares that
    //    sampler* (docs/shader-pipeline.md §2.2; `preprocess.rs`). `blend.frag`
    //    declares `g_Texture7` (`combo:OPACITYMASK`, `default:util/white`) so its
    //    fragment promotes `OPACITYMASK=1` and reads `v_TexCoordOpacity`, while
    //    `blend.vert` only guards a *resolution uniform* under `#if OPACITYMASK==1`
    //    and never declares the sampler — so its vertex stays `OPACITYMASK=0` and
    //    never writes that varying. GL resolves a program's combos *once* for both
    //    stages; reproduce that by unioning the active combos (a non-zero value
    //    from either stage wins) and forcing that program-wide set on both.
    // 2. Declaration order. Even with equal combos, `blend.vert`/`blend.frag`
    //    declare their varyings in a different order, so auto-mapping still
    //    disagrees. Pin every non-array varying to a stable name-derived
    //    `layout(location=N)` in both sources so the stages agree by construction.
    //    Skipped when either stage declares an *array* varying: pinning cannot
    //    annotate those (the array-flattening pass rejects a `layout(...)` prefix),
    //    so pinned scalars would collide with the array auto-mapped from location 0
    //    (`light_map.vert`'s `v_TexCoord[4]` → `BindingCollision`).
    //
    // This only runs when the independent translation genuinely mismatched, so a
    // pass that already builds is never perturbed (SPEC.md §V9; no scene regresses).
    let vs_out0 = io_locations(&vs.module, IoDir::Output);
    let mismatch = io_locations(&fs.module, IoDir::Input)
        .iter()
        .any(|loc| !vs_out0.contains(loc));
    if mismatch {
        // First drop fragment `varying` declarations the vertex never writes AND
        // the fragment body never reads (declaration is the only occurrence).
        // GL's linker eliminates such dead varyings (WE's stock waterripple.frag
        // declares `varying vec2 v_Scroll;` and never uses it); wgpu links
        // byte-strictly and would reject the whole pass below.
        let fs_src_stripped = strip_dead_fs_varyings(&vs_src, &fs_src);
        let (vs_src, fs_src) = if has_array_varying(&vs_src) || has_array_varying(&fs_src_stripped) {
            (vs_src.clone(), fs_src_stripped)
        } else {
            pin_varying_locations(&vs_src, &fs_src_stripped)
        };
        let vs0 = translate(Stage::Vertex, "pass.vert", &vs_src, resolver, &base_inputs)?;
        let fs0 = translate(Stage::Fragment, "pass.frag", &fs_src, resolver, &base_inputs)?;
        let mut merged: BTreeMap<String, i32> = vs0.reflection.active_combos.clone();
        for (name, value) in &fs0.reflection.active_combos {
            let slot = merged.entry(name.clone()).or_insert(*value);
            if *value != 0 {
                *slot = *value;
            }
        }
        let inputs = ShaderInputs {
            combos: base_inputs.combos.clone(),
            override_combos: merged,
            populated_texture_slots: base_inputs.populated_texture_slots.clone(),
        };
        vs = translate(Stage::Vertex, "pass.vert", &vs_src, resolver, &inputs)?;
        fs = translate(Stage::Fragment, "pass.frag", &fs_src, resolver, &inputs)?;
    }

    // Reject anything but the 2D attribute set (models are skipped, §V9). The
    // shader crate already elides attributes an active branch declares but never
    // consumes (e.g. `flat.vert`'s `a_Color`, read only under an inactive
    // `#ifdef VERTEXCOLOR`), so a surviving unknown attribute is one the shader
    // genuinely needs and the 2D VAO can't feed.
    let mut pos_location = 0u32;
    let mut uv_location = None;
    for attr in &vs.reflection.attributes {
        match attr.name.as_str() {
            "a_Position" => pos_location = attr.location,
            "a_TexCoord" => uv_location = Some(attr.location),
            other => {
                return Err(TranslateError::NoMain {
                    file: format!("unsupported vertex attribute {other}"),
                }
                .into());
            }
        }
    }

    // wgpu links VS→FS by `@location`, byte-strictly: every fragment input must
    // be a vertex output. The two stages are translated independently, so a
    // fragment reading a varying the vertex never writes (legal-but-undefined
    // under GL) would panic pipeline creation. Detect it and skip the pass
    // instead (docs/render-architecture.md §8.5; SPEC.md §V9).
    let vs_outputs = io_locations(&vs.module, IoDir::Output);
    for loc in io_locations(&fs.module, IoDir::Input) {
        if !vs_outputs.contains(&loc) {
            return Err(PassBuildError::InterfaceMismatch(loc));
        }
    }

    // Guard the inter-stage variable limit before `create_render_pipeline`,
    // which panics on overflow (wgpu's default fatal error handler). A varying
    // at location N needs N < the limit; skip the pass otherwise (SPEC.md §V9).
    let max_varyings = device.limits().max_inter_stage_shader_variables;
    let fs_inputs = io_locations(&fs.module, IoDir::Input);
    if let Some(&loc) = vs_outputs.iter().chain(fs_inputs.iter()).max()
        && loc >= max_varyings
    {
        return Err(PassBuildError::TooManyVaryings(loc, max_varyings));
    }

    // Remap the fragment module's resources to group 1.
    let mut fs_module = fs.module;
    for (_h, gv) in fs_module.global_variables.iter_mut() {
        if let Some(binding) = &mut gv.binding {
            binding.group = 1;
        }
    }

    // Read the exact std140 layout of each stage's `_WEGlobals` UBO from the
    // translated module (member offsets + block size), so per-frame writes land
    // where the GPU expects and the bound buffer is exactly the shader's size —
    // no hand-rolled std140 that could drift (docs/render-architecture.md §8.3).
    let vs_globals = globals_layout(
        &vs.module,
        &vs.reflection.globals_block,
        &param_types(&vs.reflection),
    );
    let fs_globals = globals_layout(
        &fs_module,
        &fs.reflection.globals_block,
        &param_types(&fs.reflection),
    );

    // The bind-group layouts are built from the modules' *actual* resource
    // bindings, not from reflection alone — a shader may declare more samplers
    // than the annotation reflection lists, and every declared binding must
    // appear in the pipeline layout or wgpu rejects the pipeline (docs §8.5).
    let g0_bindings = module_bindings(&vs.module);
    let g1_bindings = module_bindings(&fs_module);

    let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirie-scene-vs"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(vs.module)),
    });
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirie-scene-fs"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(fs_module)),
    });

    let g0_layout = stage_layout(device, "kirie-scene-g0", wgpu::ShaderStages::VERTEX, &g0_bindings);
    let g1_layout = stage_layout(
        device,
        "kirie-scene-g1",
        wgpu::ShaderStages::FRAGMENT,
        &g1_bindings,
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kirie-scene-pipeline-layout"),
        bind_group_layouts: &[Some(&g0_layout), Some(&g1_layout)],
        immediate_size: 0,
    });

    // Vertex buffer attributes: only those the shader actually declares.
    let mut attrs: Vec<wgpu::VertexAttribute> = vec![wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: pos_location,
    }];
    if let Some(uv) = uv_location {
        attrs.push(wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 12,
            shader_location: uv,
        });
    }
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attrs,
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("kirie-scene-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(vertex_layout)],
        },
        primitive: wgpu::PrimitiveState {
            topology,
            cull_mode: blend::cull_mode(cull),
            front_face: wgpu::FrontFace::Ccw,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: blend::depth_stencil_state(depthtest, depthwrite, wgpu::TextureFormat::Depth24Plus),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(blend::blend_state(blending)),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: kirie_platform::pipeline_cache(),
    });

    Ok(BuiltPass {
        pipeline,
        g0_layout,
        g1_layout,
        vs_globals,
        fs_globals,
        vs_samplers: vs.reflection.samplers,
        fs_samplers: fs.reflection.samplers,
        g0_bindings,
        g1_bindings,
        vs_params: vs.reflection.parameters,
        fs_params: fs.reflection.parameters,
        pos_location,
        uv_location,
    })
}

/// The interleaved 48-byte `.mdl` vertex stride (`CModel.cpp:23`,
/// `kirie_formats::model::VERTEX_STRIDE`): `pos[3] normal[3] tangent[4] uv[2]`.
const MODEL_VERTEX_STRIDE: u64 = 48;

/// Build a pass pipeline for a 3D MODEL mesh (`CModel::setupMesh`,
/// docs/render-architecture.md §7.2). Unlike [`build_pass`] this accepts the
/// full `.mdl` attribute set (`a_Position`/`a_Normal`/`a_Tangent4`/`a_TexCoord`)
/// bound at their fixed 48-byte offsets, draws a **triangle list** (models are
/// indexed lists, not the 2D strip), and carries a real depth-stencil state (the
/// model gets a private depth buffer for sub-mesh occlusion, `CModel::render`).
/// Winding matches kirie's Y-up scene FBO: front faces are CCW, cull per the
/// material's `cullmode` (the reference flips clip-Y for its Y-down FBO and so
/// declares CW front — kirie applies no flip, see [`super::matrix::perspective`]
/// and `model.rs`). A skinning attribute (`a_BlendIndices`/`a_BlendWeights`, the
/// unsupported `MDLV0023` puppets) is rejected so the mesh is skipped (§V9).
#[allow(clippy::too_many_arguments)]
pub fn build_model_pass(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    pass: &Pass,
    vs_src: &str,
    fs_src: &str,
    resolver: &dyn IncludeResolver,
) -> Result<BuiltPass, PassBuildError> {
    let base_inputs = shader_inputs(pass);

    // kirie-shader assigns explicit `@location`s to vertex *attributes* but lets
    // glslang auto-map *varyings* per stage by declaration order. The 2D effect
    // shaders declare their varyings in the same order in `.vert` and `.frag`, so
    // that lines up — but `generic3.vert` and `generic3.frag` declare theirs in
    // *different* orders (e.g. `v_ViewDir` before vs after `v_TexCoord`), so the
    // auto-mapped locations disagree and wgpu rejects the pipeline on a type
    // mismatch at a shared location (GL links varyings by name, wgpu by location).
    // Pin each varying to a stable name-derived `layout(location = N)` in both
    // sources first, so the stages agree by construction (docs/shader-pipeline.md
    // §9.2.3 inter-stage matching). Scoped to the model path — the 2D path is
    // untouched.
    let (vs_src, fs_src) = pin_varying_locations(vs_src, fs_src);

    // GL resolves a program's combos ONCE and compiles both stages with that one
    // set; kirie-shader resolves them per stage. A require chain declared in only
    // one stage then splits the two — `generic3.frag` declares
    // `RIMLIGHTING → require LIGHTING=1`, so a material with `RIMLIGHTING=1,
    // LIGHTING=0` promotes `LIGHTING=1` in the *fragment* (which then reads
    // `v_WorldNormal`/`v_WorldPos`) but leaves the *vertex* at `LIGHTING=0` (which
    // never writes them) — an inter-stage mismatch wgpu rejects. Resolve each
    // stage once, union their active combos (an active/non-zero value from either
    // stage wins the promotion), and force that program-wide set on both stages so
    // their varyings agree, reproducing GL's single whole-program resolution.
    let vs0 = translate(Stage::Vertex, "model.vert", &vs_src, resolver, &base_inputs)?;
    let fs0 = translate(Stage::Fragment, "model.frag", &fs_src, resolver, &base_inputs)?;
    let mut merged: BTreeMap<String, i32> = vs0.reflection.active_combos.clone();
    for (name, value) in &fs0.reflection.active_combos {
        let slot = merged.entry(name.clone()).or_insert(*value);
        if *value != 0 {
            *slot = *value;
        }
    }
    let inputs = ShaderInputs {
        combos: base_inputs.combos,
        override_combos: merged,
        populated_texture_slots: base_inputs.populated_texture_slots,
    };
    let vs = translate(Stage::Vertex, "model.vert", &vs_src, resolver, &inputs)?;
    let fs = translate(Stage::Fragment, "model.frag", &fs_src, resolver, &inputs)?;

    // Map the declared attributes onto the fixed .mdl layout (`CModel.cpp:24-27`);
    // only the ones the compiled shader actually uses are bound, exactly like the
    // reference's `glGetAttribLocation >= 0` guard.
    let mut pos_location = 0u32;
    let mut uv_location = None;
    let mut attrs: Vec<wgpu::VertexAttribute> = Vec::new();
    for attr in &vs.reflection.attributes {
        let (format, offset) = match attr.name.as_str() {
            "a_Position" => {
                pos_location = attr.location;
                (wgpu::VertexFormat::Float32x3, 0)
            }
            "a_Normal" => (wgpu::VertexFormat::Float32x3, 12),
            "a_Tangent4" => (wgpu::VertexFormat::Float32x4, 24),
            "a_TexCoord" => {
                uv_location = Some(attr.location);
                (wgpu::VertexFormat::Float32x2, 40)
            }
            other => {
                return Err(TranslateError::NoMain {
                    file: format!("unsupported model vertex attribute {other}"),
                }
                .into());
            }
        };
        attrs.push(wgpu::VertexAttribute {
            format,
            offset,
            shader_location: attr.location,
        });
    }
    attrs.sort_by_key(|a| a.shader_location);

    // Same strict VS→FS interface + varying-limit guards as the 2D path (§8.5).
    let vs_outputs = io_locations(&vs.module, IoDir::Output);
    for loc in io_locations(&fs.module, IoDir::Input) {
        if !vs_outputs.contains(&loc) {
            return Err(PassBuildError::InterfaceMismatch(loc));
        }
    }
    let max_varyings = device.limits().max_inter_stage_shader_variables;
    let fs_inputs = io_locations(&fs.module, IoDir::Input);
    if let Some(&loc) = vs_outputs.iter().chain(fs_inputs.iter()).max()
        && loc >= max_varyings
    {
        return Err(PassBuildError::TooManyVaryings(loc, max_varyings));
    }

    // Remap the fragment module's resources to group 1 (see `build_pass`).
    let mut fs_module = fs.module;
    for (_h, gv) in fs_module.global_variables.iter_mut() {
        if let Some(binding) = &mut gv.binding {
            binding.group = 1;
        }
    }

    let vs_globals = globals_layout(
        &vs.module,
        &vs.reflection.globals_block,
        &param_types(&vs.reflection),
    );
    let fs_globals = globals_layout(
        &fs_module,
        &fs.reflection.globals_block,
        &param_types(&fs.reflection),
    );
    let g0_bindings = module_bindings(&vs.module);
    let g1_bindings = module_bindings(&fs_module);

    let vs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirie-model-vs"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(vs.module)),
    });
    let fs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kirie-model-fs"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(fs_module)),
    });

    let g0_layout = stage_layout(device, "kirie-model-g0", wgpu::ShaderStages::VERTEX, &g0_bindings);
    let g1_layout = stage_layout(
        device,
        "kirie-model-g1",
        wgpu::ShaderStages::FRAGMENT,
        &g1_bindings,
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kirie-model-pipeline-layout"),
        bind_group_layouts: &[Some(&g0_layout), Some(&g1_layout)],
        immediate_size: 0,
    });

    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: MODEL_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attrs,
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("kirie-model-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(vertex_layout)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: blend::cull_mode(pass.cullmode),
            front_face: wgpu::FrontFace::Ccw,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: blend::depth_stencil_state(pass.depthtest, pass.depthwrite, depth_format),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &fs_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(blend::blend_state(pass.blending)),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: kirie_platform::pipeline_cache(),
    });

    Ok(BuiltPass {
        pipeline,
        g0_layout,
        g1_layout,
        vs_globals,
        fs_globals,
        vs_samplers: vs.reflection.samplers,
        fs_samplers: fs.reflection.samplers,
        g0_bindings,
        g1_bindings,
        vs_params: vs.reflection.parameters,
        fs_params: fs.reflection.parameters,
        pos_location,
        uv_location,
    })
}

/// Pin every (non-array) `varying` in a `.vert`/`.frag` pair to the same
/// `layout(location = N)` in both stages, derived from a stable sort of the
/// union of varying names across both sources. This makes the two independently
/// translated stages agree on inter-stage locations by name (see the call site).
///
/// Array varyings are left untouched: kirie-shader flattens them into scalars
/// later by a regex that a `layout(...)` prefix would defeat, and the model
/// materials in the corpus declare none — so they fall back to auto-mapping.
fn pin_varying_locations(vs_src: &str, fs_src: &str) -> (String, String) {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for src in [vs_src, fs_src] {
        for line in src.lines() {
            if let Some(name) = varying_decl_name(line) {
                names.insert(name.to_string());
            }
        }
    }
    let locations: BTreeMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let rewrite = |src: &str| -> String {
        let mut out = String::with_capacity(src.len() + names.len() * 24);
        for (i, line) in src.lines().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            match varying_decl_name(line).and_then(|n| locations.get(n).copied()) {
                Some(loc) => {
                    let indent = line.len() - line.trim_start().len();
                    out.push_str(&line[..indent]);
                    out.push_str(&format!("layout(location = {loc}) "));
                    out.push_str(line.trim_start());
                }
                None => out.push_str(line),
            }
        }
        out
    };
    (rewrite(vs_src), rewrite(fs_src))
}

/// Drop fragment `varying` declarations that are dead: not declared by the
/// vertex stage AND never mentioned again in the fragment source (the
/// declaration line is the identifier's only occurrence). Mirrors the GL
/// linker's dead-varying elimination so a stock shader with a stray declaration
/// (waterripple.frag's `v_Scroll`) doesn't fail wgpu's strict VS/FS interface
/// match and lose the whole pass. Conservative: an identifier that appears
/// anywhere else (even a comment) is kept.
fn strip_dead_fs_varyings(vs_src: &str, fs_src: &str) -> String {
    let vs_names: BTreeSet<&str> = vs_src.lines().filter_map(varying_decl_name).collect();
    let dead: Vec<&str> = fs_src
        .lines()
        .filter_map(varying_decl_name)
        .filter(|n| !vs_names.contains(n))
        .filter(|n| fs_src.matches(n).count() == 1)
        .collect();
    if dead.is_empty() {
        return fs_src.to_string();
    }
    fs_src
        .lines()
        .filter(|line| !varying_decl_name(line).is_some_and(|n| dead.contains(&n)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether any line declares an *array* varying (`varying TYPE name[N];`).
/// [`pin_varying_locations`] cannot annotate these (the array-flattening pass
/// rejects a `layout(...)` prefix), so a shader that mixes an array varying with
/// pinned scalars would collide at a shared location — the caller skips pinning
/// entirely for such shaders (e.g. `light_map.vert`'s `v_TexCoord[4]`).
fn has_array_varying(src: &str) -> bool {
    src.lines().any(|line| {
        line.trim_start()
            .strip_prefix("varying ")
            .is_some_and(|rest| rest.contains('['))
    })
}

/// The identifier of a plain (non-array) `varying TYPE name;` declaration line,
/// else `None`. Array varyings (`varying TYPE name[N];`) return `None` so they
/// are left for the array-flattening pass (see [`pin_varying_locations`]).
fn varying_decl_name(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("varying ")?;
    // An array varying (or any bracketed form) is left alone.
    if rest.contains('[') {
        return None;
    }
    // `TYPE name;` — take the second token, trim the terminator.
    let name = rest.split_whitespace().nth(1)?.trim_end_matches(';');
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(name)
}

/// The kind of a shader resource binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindKind {
    /// A `layout(std140) uniform` block — the `_WEGlobals` UBO.
    Ubo,
    /// A `texture2D` (the split half of a combined sampler).
    Texture,
    /// A `sampler` (the split half of a combined sampler).
    Sampler,
}

/// One resource binding a translated module actually declares (within its
/// group). Ground truth for both the bind-group layout and the bind group.
#[derive(Clone, Copy, Debug)]
pub struct ModuleBinding {
    /// Binding number within the stage's bind group.
    pub binding: u32,
    /// What kind of resource it is.
    pub kind: BindKind,
}

/// Enumerate a translated module's resource bindings, classified by type and
/// sorted by binding number (docs/shader-pipeline.md §9.1.3). Only uniform
/// buffers, sampled images, and samplers appear in WE shaders; anything else is
/// ignored.
fn module_bindings(module: &naga::Module) -> Vec<ModuleBinding> {
    let mut out = Vec::new();
    for (_h, gv) in module.global_variables.iter() {
        let Some(rb) = &gv.binding else { continue };
        let kind = match gv.space {
            naga::AddressSpace::Uniform => BindKind::Ubo,
            naga::AddressSpace::Handle => match module.types[gv.ty].inner {
                naga::TypeInner::Image { .. } => BindKind::Texture,
                naga::TypeInner::Sampler { .. } => BindKind::Sampler,
                _ => continue,
            },
            _ => continue,
        };
        out.push(ModuleBinding {
            binding: rb.binding,
            kind,
        });
    }
    out.sort_by_key(|b| b.binding);
    out
}

/// Which side of an entry point's inter-stage interface to collect.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IoDir {
    /// The entry point's result (vertex outputs).
    Output,
    /// The entry point's arguments (fragment inputs).
    Input,
}

/// Collect the set of inter-stage `@location` numbers on one side of a module's
/// first entry point (docs/render-architecture.md §8.5). Builtins (`position`,
/// etc.) carry no `Location` binding and are ignored; struct-typed IO is walked
/// member by member.
fn io_locations(module: &naga::Module, dir: IoDir) -> BTreeSet<u32> {
    let mut locs = BTreeSet::new();
    let Some(ep) = module.entry_points.first() else {
        return locs;
    };
    let mut collect = |binding: Option<&naga::Binding>, ty: naga::Handle<naga::Type>| {
        match binding {
            Some(naga::Binding::Location { location, .. }) => {
                locs.insert(*location);
            }
            _ => {
                // Unbound IO is a struct: each member carries its own binding.
                if let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner {
                    for m in members {
                        if let Some(naga::Binding::Location { location, .. }) = &m.binding {
                            locs.insert(*location);
                        }
                    }
                }
            }
        }
    };
    match dir {
        IoDir::Output => {
            if let Some(res) = &ep.function.result {
                collect(res.binding.as_ref(), res.ty);
            }
        }
        IoDir::Input => {
            for arg in &ep.function.arguments {
                collect(arg.binding.as_ref(), arg.ty);
            }
        }
    }
    locs
}

/// Normalize GL-lenient constructs that strict wgpu translation rejects, so a
/// workshop effect shader authored against desktop GL still builds instead of
/// being skipped whole (SPEC.md §V9; docs/shader-pipeline.md §9.2). Two corpus
/// cases, both no-ops on well-formed shaders:
///
/// 1. **`const` function-return qualifier** (e.g. `const float fract(float x)`).
///    Illegal GLSL that some drivers accept; shaderc/glslang rejects it. The
///    leading `const` is dropped from any line that is a function *definition*
///    (a `(` reached before any `=`/`;`). A `const` *global* (`const float PI =
///    …`) reaches `=` first and is left untouched.
/// 2. **Fragment-stage `gl_Position` read** (undefined in GL, so it reads
///    garbage; naga hard-rejects it as an invalid fragment built-in). Rewritten
///    to `gl_FragCoord`, the valid per-fragment screen position — the intent
///    wherever it is (mis)used, e.g. a per-pixel noise seed. Applied only to the
///    fragment stage; `gl_Position` is a legitimate vertex output.
fn sanitize_glsl(src: &str, stage: Stage) -> String {
    let mut out = String::with_capacity(src.len());
    for (i, line) in src.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("const ")
            && is_const_return_function(rest)
        {
            let indent = &line[..line.len() - trimmed.len()];
            out.push_str(indent);
            out.push_str(rest);
        } else {
            out.push_str(line);
        }
    }
    if stage == Stage::Fragment && out.contains("gl_Position") {
        out = out.replace("gl_Position", "gl_FragCoord");
    }
    out
}

/// Whether `rest` (the text after a leading `const `) begins a function
/// definition — a `(` occurs before any `=` or `;`, i.e. `TYPE NAME(…)` rather
/// than a `const` variable declaration.
fn is_const_return_function(rest: &str) -> bool {
    for c in rest.chars() {
        match c {
            '(' => return true,
            '=' | ';' => return false,
            _ => {}
        }
    }
    false
}

/// Assemble [`ShaderInputs`] from a resolved pass: merged combos + the set of
/// populated texture slots (slot 0 = the chain input, always populated;
/// docs/shader-pipeline.md §2.2 rule 1).
fn shader_inputs(pass: &Pass) -> ShaderInputs {
    let mut combos = BTreeMap::new();
    for (k, v) in &pass.combos {
        combos.insert(k.clone(), *v as i32);
    }
    let mut slots = std::collections::BTreeSet::new();
    slots.insert(0u32);
    for (i, slot) in pass.textures.iter().enumerate() {
        if slot.is_some() {
            slots.insert(i as u32);
        }
    }
    ShaderInputs {
        combos,
        override_combos: BTreeMap::new(),
        populated_texture_slots: slots,
    }
}

/// Read a stage's `_WEGlobals` UBO layout straight from the translated module:
/// exact member byte offsets and the block's total `span`, as naga computed
/// them (docs/render-architecture.md §8.3). Member names come from the module
/// when preserved, else positionally from `globals_block`; each member's
/// write-stride type is inferred by name (builtin, else material parameter,
/// else `float`). Falls back to the name-only std140 computation when the
/// module declares no uniform block.
fn globals_layout(
    module: &naga::Module,
    globals_block: &[String],
    param_types: &BTreeMap<String, GlType>,
) -> GlobalsLayout {
    let uniform_ty = module
        .global_variables
        .iter()
        .find_map(|(_h, gv)| (gv.space == naga::AddressSpace::Uniform).then_some(gv.ty));
    let Some(ty) = uniform_ty else {
        return GlobalsLayout::build(globals_block, param_types);
    };
    let naga::TypeInner::Struct { members, span } = &module.types[ty].inner else {
        return GlobalsLayout::build(globals_block, param_types);
    };
    let members = members
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let name = m
                .name
                .clone()
                .or_else(|| globals_block.get(i).cloned())
                .unwrap_or_default();
            let ty = builtin_type(&name)
                .or_else(|| param_types.get(&name).copied())
                .unwrap_or(GlType::Float);
            Member {
                name,
                ty,
                offset: m.offset as usize,
            }
        })
        .collect();
    GlobalsLayout {
        members,
        size: *span as usize,
    }
}

/// Map a stage's reflected material parameters to their global-member types.
fn param_types(reflection: &Reflection) -> BTreeMap<String, GlType> {
    reflection
        .parameters
        .iter()
        .map(|p| (p.name.clone(), GlType::from_param(p.ty)))
        .collect()
}

/// Build the bind-group layout for one stage directly from the module's actual
/// resource bindings, so it can never diverge from what the pipeline declares
/// (docs §8.5). Every UBO is `Uniform`, every texture a filterable 2D float
/// image, every sampler `Filtering`.
fn stage_layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
    bindings: &[ModuleBinding],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .map(|b| wgpu::BindGroupLayoutEntry {
            binding: b.binding,
            visibility,
            ty: match b.kind {
                BindKind::Ubo => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindKind::Texture => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                BindKind::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}
