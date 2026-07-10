// Image wallpaper present pass: fullscreen triangle-strip quad sampling one
// content page through a scaling UV window and a per-frame atlas placement.
//
// The window carries the output-scaling crop/overscan
// (docs/render-architecture.md §4); the frame transform is the reference's
// g_Texture0Translation / g_Texture0Rotation pair (docs/format-tex.md §8.1
// step 5: pageUV = translation + uv.x·axes.xz + uv.y·axes.yw).

struct Window {
    // u0, v0, u1, v1 — UVs at the viewport's top-left / bottom-right.
    rect: vec4<f32>,
    // 0 = clamp (GL_CLAMP_TO_EDGE), 1 = border (GL_CLAMP_TO_BORDER,
    // transparent-black border), 2 = repeat (GL_REPEAT)
    // (docs/compat-cli.md §3.1; docs/render-architecture.md §4).
    clamp_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct FrameXform {
    // xy = frame origin / page dims (g_Texture0Translation).
    translation: vec4<f32>,
    // (width1/w, width2/w, height2/h, height1/h) (g_Texture0Rotation).
    axes: vec4<f32>,
}

@group(0) @binding(0) var<uniform> window: Window;
@group(0) @binding(1) var<uniform> frame: FrameXform;
@group(0) @binding(2) var page: texture_2d<f32>;
@group(0) @binding(3) var page_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // 4-vertex strip: TL, BL, TR, BR (matches UvWindow::strip_corners).
    let x = f32(index >> 1u); // 0, 0, 1, 1
    let y = f32(index & 1u); // 0, 1, 0, 1
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(
        mix(window.rect.x, window.rect.z, x),
        mix(window.rect.y, window.rect.w, y),
    );
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Resolve out-of-window UVs in *content* space before the atlas
    // transform, so overscan never bleeds into neighboring atlas frames
    // (the reference wraps on its content-sized FBO for the same effect,
    // docs/render-architecture.md §4). `clamp_mode` is uniform, so control
    // flow stays uniform for textureSample below.
    var uv = in.uv;
    var mask = 1.0;
    if window.clamp_mode == 2u {
        // repeat → GL_REPEAT.
        uv = fract(uv);
    } else if window.clamp_mode == 1u {
        // border → GL_CLAMP_TO_BORDER with the GL default border color,
        // transparent black (docs/render-architecture.md §4,
        // CFBO.cpp:31-33). Masking after an edge clamp gives a hard border
        // instead of GL's half-texel blend — accepted deviation.
        let inside = step(vec2<f32>(0.0), uv) * step(uv, vec2<f32>(1.0));
        mask = inside.x * inside.y;
        uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    } else {
        // clamp → GL_CLAMP_TO_EDGE.
        uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    }

    // docs/format-tex.md §8.1: atlasUV = translation + uv.x·axes.xz +
    // uv.y·axes.yw (identity/crop placement for still images).
    let atlas_uv = frame.translation.xy
        + uv.x * vec2<f32>(frame.axes.x, frame.axes.z)
        + uv.y * vec2<f32>(frame.axes.y, frame.axes.w);
    return textureSample(page, page_sampler, atlas_uv) * mask;
}
