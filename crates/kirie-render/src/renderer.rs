//! The still-image / animated-gif wallpaper renderer
//! ([`kirie_platform::Renderer`] implementation).
//!
//! Upload model (SPEC §V5): every content page is uploaded to its own GPU
//! texture once at construction; per frame only a 32-byte uniform write
//! happens, and only when the displayed frame actually changes. Static
//! content re-encodes the same cached draw with zero uploads.

use std::time::Duration;

use kirie_platform::{RenderTarget, Renderer, SurfaceSize};

use crate::content::ImageContent;
use crate::error::RenderError;
use crate::scaling::{ClampMode, ScalingMode, UvWindow};
use crate::schedule::FrameSchedule;

/// Presentation options for one output, from the CLI compat surface
/// (docs/compat-cli.md §2: `--scaling` default `default`, `--clamp` default
/// `clamp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageOptions {
    /// Output scaling mode (docs/render-architecture.md §4).
    pub scaling: ScalingMode,
    /// Out-of-window UV behavior (docs/render-architecture.md §4).
    pub clamp: ClampMode,
}

/// std140-compatible window uniform: scaling UV window + clamp mode.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WindowUniform {
    rect: [f32; 4],
    clamp_mode: u32,
    _pad: [u32; 3],
}

/// std140-compatible frame uniform: the §8.1 atlas placement.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    translation: [f32; 4],
    axes: [f32; 4],
}

/// Renders decoded [`ImageContent`] to one output surface.
pub struct ImageRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    window_buffer: wgpu::Buffer,
    frame_buffer: wgpu::Buffer,
    /// One bind group per content page; switching frames switches bind
    /// groups (prebuilt — no steady-state allocation, SPEC §V5).
    bind_groups: Vec<wgpu::BindGroup>,
    /// Page index per frame, parallel to `schedule`.
    frame_pages: Vec<usize>,
    /// Atlas placement uniform per frame, parallel to `schedule`.
    frame_uniforms: Vec<FrameUniform>,
    schedule: FrameSchedule,
    options: ImageOptions,
    content_size: (u32, u32),
    /// Wall-clock seconds accumulated from per-frame `dt` — the unscaled
    /// render-time counter the reference feeds to `fmod`
    /// (docs/format-tex.md §8.1 step 2).
    elapsed: f64,
    current_frame: usize,
    /// Surface size the current window uniform was computed for.
    window_for: Option<SurfaceSize>,
}

impl ImageRenderer {
    /// Upload `content` and build the present pipeline for one output.
    pub fn new(
        target: &RenderTarget<'_>,
        content: &ImageContent,
        options: ImageOptions,
    ) -> Result<Self, RenderError> {
        let device = target.device;
        let max_dim = device.limits().max_texture_dimension_2d;

        // Match the surface's sRGB-ness so stored bytes survive the
        // sample→write round trip unchanged, like the reference's
        // gamma-naive GL_RGBA8 → default-framebuffer path
        // (docs/render-architecture.md §10).
        let texture_format = if target.format.is_srgb() {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kirie-image-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let window_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kirie-image-window-uniform"),
            size: size_of::<WindowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kirie-image-frame-uniform"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // docs/format-tex.md §6.1: NoInterpolation → nearest else linear;
        // ClampUVs → clamp-to-edge else repeat. The scaling window's own
        // wrap (--clamp) is resolved in content space by the shader, so
        // this sampler only governs filtering at page edges.
        let filter = if content.sampler.nearest {
            wgpu::FilterMode::Nearest
        } else {
            wgpu::FilterMode::Linear
        };
        let address = if content.sampler.clamp_uvs {
            wgpu::AddressMode::ClampToEdge
        } else {
            wgpu::AddressMode::Repeat
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("kirie-image-sampler"),
            address_mode_u: address,
            address_mode_v: address,
            mag_filter: filter,
            min_filter: filter,
            ..wgpu::SamplerDescriptor::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kirie-image-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Upload every page once; animation only ever rebinds
        // (docs/format-tex.md §9: the reference likewise keeps one GL
        // texture per image and binds textureID[frameNumber]).
        let mut bind_groups = Vec::with_capacity(content.pages.len());
        for page in &content.pages {
            if page.width > max_dim || page.height > max_dim {
                return Err(RenderError::TextureTooLarge {
                    width: page.width,
                    height: page.height,
                    max: max_dim,
                });
            }
            if page.width == 0 || page.height == 0 {
                return Err(RenderError::InvalidDimensions {
                    width: page.width,
                    height: page.height,
                });
            }
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("kirie-image-page"),
                size: wgpu::Extent3d {
                    width: page.width,
                    height: page.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            target.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &page.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * page.width),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: page.width,
                    height: page.height,
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kirie-image-bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: window_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: frame_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }));
        }
        if bind_groups.is_empty() {
            return Err(RenderError::NoImages);
        }

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kirie-image-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("kirie-image-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.format,
                    // The reference's final blit runs with blending
                    // disabled (docs/render-architecture.md §2.5).
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let frame_pages: Vec<usize> = content.frames.iter().map(|f| f.page).collect();
        let frame_uniforms: Vec<FrameUniform> = content
            .frames
            .iter()
            .map(|f| FrameUniform {
                translation: [f.translation[0], f.translation[1], 0.0, 0.0],
                axes: f.axes,
            })
            .collect();

        // Seed the frame uniform with frame 0; the window uniform is
        // written on the first `render` when the real surface size is
        // known.
        target
            .queue
            .write_buffer(&frame_buffer, 0, bytemuck::bytes_of(&frame_uniforms[0]));

        Ok(Self {
            device: target.device.clone(),
            queue: target.queue.clone(),
            pipeline,
            window_buffer,
            frame_buffer,
            bind_groups,
            frame_pages,
            frame_uniforms,
            schedule: content.schedule(),
            options,
            content_size: content.content_size(),
            elapsed: 0.0,
            current_frame: 0,
            window_for: None,
        })
    }

    /// Whether the content ever changes frames. Static content needs no
    /// further redraws after the first presented frame (SPEC §V6) — the
    /// presentation layer can stop scheduling frame callbacks based on
    /// this + [`ImageRenderer::time_until_frame_change`].
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.schedule.is_animated()
    }

    /// Time until the displayed frame next changes, or `None` for static
    /// content (redraw-scheduling hint, SPEC §V6; the wall-clock walk of
    /// docs/format-tex.md §8.1).
    #[must_use]
    pub fn time_until_frame_change(&self) -> Option<Duration> {
        self.schedule
            .time_until_change(self.elapsed)
            .map(Duration::from_secs_f64)
    }

    /// The UV window currently in effect for `size`
    /// (docs/render-architecture.md §4).
    #[must_use]
    pub fn uv_window_for(&self, size: SurfaceSize) -> UvWindow {
        self.options
            .scaling
            .uv_window(self.content_size, (size.width, size.height))
    }
}

impl Renderer for ImageRenderer {
    fn render(&mut self, view: &wgpu::TextureView, size: SurfaceSize, dt: f32) {
        // Unscaled wall-clock accumulation (docs/format-tex.md §8.1 step 2:
        // frame selection uses render time, not the playback-speed-scaled
        // g_Time).
        self.elapsed += f64::from(dt);

        // Recompute the scaling window only when the surface size changes
        // (the reference caches UVs the same way,
        // docs/render-architecture.md §4, WallpaperState.cpp:11-17).
        if self.window_for != Some(size) {
            let window = self.uv_window_for(size);
            let clamp_mode = match self.options.clamp {
                ClampMode::Clamp => 0u32,
                ClampMode::Border => 1u32,
                ClampMode::Repeat => 2u32,
            };
            self.queue.write_buffer(
                &self.window_buffer,
                0,
                bytemuck::bytes_of(&WindowUniform {
                    rect: [window.u0, window.v0, window.u1, window.v1],
                    clamp_mode,
                    _pad: [0; 3],
                }),
            );
            self.window_for = Some(size);
        }

        // Animated content: pick the frame by the §8.1 walk and upload the
        // 32-byte placement only when it changes (SPEC §V5 — the atlas
        // pages themselves were uploaded once at construction).
        if self.schedule.is_animated() {
            let frame = self.schedule.frame_at(self.elapsed);
            if frame != self.current_frame {
                self.current_frame = frame;
                self.queue.write_buffer(
                    &self.frame_buffer,
                    0,
                    bytemuck::bytes_of(&self.frame_uniforms[frame]),
                );
            }
        }

        let bind_group = &self.bind_groups[self.frame_pages[self.current_frame]];

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kirie-image-encoder"),
            });
        {
            // The quad covers the whole surface; the clear only shows
            // through border-mode transparency. Opaque black matches the
            // GL default border color composited onto an opaque surface
            // (docs/render-architecture.md §4, §5.2).
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kirie-image-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
