//! Per-pass GL fixed-function state → wgpu render-pipeline state.
//!
//! Exact port of the reference `setupRenderFramebuffer` mapping
//! (docs/render-architecture.md §8.1, `CPass.cpp:123-181`). The blend equation
//! is always `FUNC_ADD`; color and alpha use the same factors; "Normal" is an
//! opaque replace (`ONE, ZERO`) done with blending *enabled* — equivalent to
//! disabled blending, but reproduced faithfully so the mapping is exhaustively
//! testable (SPEC.md §V10 cites the doc table).

use kirie_scene::material::{Blending, CullMode, DepthMode};

/// The wgpu blend state for a pass blending mode (docs/render-architecture.md
/// §8.1). Color and alpha carry identical factors, matching the reference's
/// `glBlendFuncSeparate(f, g, f, g)` calls.
#[must_use]
pub fn blend_state(mode: Blending) -> wgpu::BlendState {
    let component = |src, dst| wgpu::BlendComponent {
        src_factor: src,
        dst_factor: dst,
        operation: wgpu::BlendOperation::Add,
    };
    let (src, dst) = match mode {
        // Normal: `glBlendFuncSeparate(ONE, ZERO, ...)` — replace.
        Blending::Normal => (wgpu::BlendFactor::One, wgpu::BlendFactor::Zero),
        // Translucent: `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` (note: *not* the
        // premultiplied-friendly ONE/ONE_MINUS_SRC_ALPHA — replicated as-is,
        // docs/render-architecture.md §8.1).
        Blending::Translucent => (wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::OneMinusSrcAlpha),
        // Additive: `SRC_ALPHA, ONE`.
        Blending::Additive => (wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::One),
    };
    // Alpha channel: the reference issues `glBlendFuncSeparate(f,g,f,g)` (color
    // factors reused for alpha) and gets away with it because it composites
    // straight onto the opaque scene framebuffer, whose accumulated alpha is
    // never read back. kirie renders each image-with-effects layer into an
    // ISOLATED FBO (cleared to alpha 0) and later alpha-composites that FBO into
    // the scene — so its alpha channel MUST accumulate as a proper straight-alpha
    // "over". For Translucent, `SRC_ALPHA` as the *alpha* source factor squares
    // the coverage of every overlapping semi-transparent triangle
    // (`outA = srcA² + dstA(1−srcA)`), eroding a fully-covered region below 1.0
    // wherever the puppet mesh layers translucent triangles (the girl 女's eye:
    // the eroded alpha let the LOGO layer bleed through as a red mark). Use the
    // coverage-correct alpha factor `ONE` there so `outA = srcA + dstA(1−srcA)`
    // reaches 1.0. Color is unchanged, so layers drawn straight onto the opaque
    // scene look identical to the reference. Normal (replace) and Additive keep
    // reference-identical alpha.
    let alpha = match mode {
        Blending::Translucent => component(wgpu::BlendFactor::One, wgpu::BlendFactor::OneMinusSrcAlpha),
        _ => component(src, dst),
    };
    wgpu::BlendState {
        color: component(src, dst),
        alpha,
    }
}

/// The wgpu cull face for a pass cull mode (docs/render-architecture.md §8.1).
///
/// `Normal` → cull back faces (GL default: cull back, CCW front, so the wgpu
/// front face stays [`wgpu::FrontFace::Ccw`]); `NoCull` → no culling.
#[must_use]
pub fn cull_mode(mode: CullMode) -> Option<wgpu::Face> {
    match mode {
        CullMode::Normal => Some(wgpu::Face::Back),
        CullMode::NoCull => None,
    }
}

/// The wgpu depth-stencil state for a pass, or `None` when depth testing is
/// disabled (docs/render-architecture.md §8.1). `depthtest Enabled` →
/// `LEQUAL`; `depthwrite Enabled` → depth mask on. A depthless pass (the 2D
/// image layers, corpus-observed default) has no depth attachment at all.
#[must_use]
pub fn depth_stencil_state(
    depthtest: DepthMode,
    depthwrite: DepthMode,
    format: wgpu::TextureFormat,
) -> Option<wgpu::DepthStencilState> {
    match depthtest {
        DepthMode::Disabled => None,
        DepthMode::Enabled => Some(wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(matches!(depthwrite, DepthMode::Enabled)),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(src: wgpu::BlendFactor, dst: wgpu::BlendFactor) -> wgpu::BlendComponent {
        wgpu::BlendComponent {
            src_factor: src,
            dst_factor: dst,
            operation: wgpu::BlendOperation::Add,
        }
    }

    #[test]
    fn normal_is_replace() {
        // docs/render-architecture.md §8.1: Normal = ONE, ZERO.
        let b = blend_state(Blending::Normal);
        assert_eq!(b.color, comp(wgpu::BlendFactor::One, wgpu::BlendFactor::Zero));
        assert_eq!(b.alpha, comp(wgpu::BlendFactor::One, wgpu::BlendFactor::Zero));
    }

    #[test]
    fn translucent_uses_srcalpha_oneminus() {
        let b = blend_state(Blending::Translucent);
        assert_eq!(
            b.color,
            comp(wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::OneMinusSrcAlpha)
        );
        // Alpha uses coverage-correct `ONE, ONE_MINUS_SRC_ALPHA` so an isolated
        // FBO accumulates a proper straight-alpha "over" (overlapping translucent
        // puppet triangles must not erode a fully-covered region below alpha 1.0).
        assert_eq!(
            b.alpha,
            comp(wgpu::BlendFactor::One, wgpu::BlendFactor::OneMinusSrcAlpha)
        );
    }

    #[test]
    fn additive_uses_srcalpha_one() {
        let b = blend_state(Blending::Additive);
        let expect = comp(wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::One);
        assert_eq!(b.color, expect);
        assert_eq!(b.alpha, expect);
    }

    #[test]
    fn every_mode_maps_to_add_operation() {
        for mode in [Blending::Normal, Blending::Translucent, Blending::Additive] {
            let b = blend_state(mode);
            assert_eq!(b.color.operation, wgpu::BlendOperation::Add);
            assert_eq!(b.alpha.operation, wgpu::BlendOperation::Add);
        }
    }

    #[test]
    fn cull_mapping() {
        assert_eq!(cull_mode(CullMode::NoCull), None);
        assert_eq!(cull_mode(CullMode::Normal), Some(wgpu::Face::Back));
    }

    #[test]
    fn depth_disabled_has_no_state() {
        assert!(
            depth_stencil_state(
                DepthMode::Disabled,
                DepthMode::Disabled,
                wgpu::TextureFormat::Depth24Plus
            )
            .is_none()
        );
        // depthwrite is irrelevant when the test is disabled.
        assert!(
            depth_stencil_state(
                DepthMode::Disabled,
                DepthMode::Enabled,
                wgpu::TextureFormat::Depth24Plus
            )
            .is_none()
        );
    }

    #[test]
    fn depth_enabled_is_lequal_with_write_flag() {
        let fmt = wgpu::TextureFormat::Depth24Plus;
        let on = depth_stencil_state(DepthMode::Enabled, DepthMode::Enabled, fmt).unwrap();
        assert_eq!(on.depth_compare, Some(wgpu::CompareFunction::LessEqual));
        assert_eq!(on.depth_write_enabled, Some(true));

        let off = depth_stencil_state(DepthMode::Enabled, DepthMode::Disabled, fmt).unwrap();
        assert_eq!(off.depth_compare, Some(wgpu::CompareFunction::LessEqual));
        assert_eq!(off.depth_write_enabled, Some(false));
    }
}
