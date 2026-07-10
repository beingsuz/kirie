//! The scene / effect render-target pool (docs/render-architecture.md §6).
//!
//! Every `CFBO` is an `RGBA16F` color target so shaders may emit HDR values and
//! the final 8-bit blit clamps (docs §6). Targets are allocated **once** at
//! build time — the scene FBO at the projection size, per-image ping-pong FBOs
//! at the image size — and reused every frame; the projection/image sizes do
//! not depend on the output surface, so a surface resize recreates nothing
//! (SPEC.md §V5). Every FBO is cleared to transparent black `(0,0,0,0)` at
//! creation (docs §5.2, `CFBO.cpp:60-65`), which under wgpu is just the first
//! render pass's `Clear` load op.

/// The internal format of every scene render target (docs §6: `GL_RGBA16F`).
pub const FBO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// One offscreen color target: a texture, its default view, and its size.
#[derive(Debug)]
pub struct Fbo {
    /// The color texture (`RGBA16F`, docs §6).
    pub texture: wgpu::Texture,
    /// Its full view, used as both a render attachment and a sampled input.
    pub view: wgpu::TextureView,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

impl Fbo {
    /// Allocate a fresh `RGBA16F` target of `width`×`height` (both clamped to at
    /// least 1 so a degenerate size never panics, SPEC.md §V9).
    #[must_use]
    pub fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FBO_FORMAT,
            // COPY_SRC/DST so the scene FBO can be snapshotted into a sibling
            // target for `_rt_FullFrameBuffer` reads (feedback-safe post-process
            // layers, docs §6/§11 shadow-copy; the reference blits the scene FBO
            // before an effect samples it). Cheap on every target, needed on two.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Fbo {
            texture,
            view,
            width,
            height,
        }
    }
}
