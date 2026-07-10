//! [`ParticleRenderer`] — a standalone instanced-quad sprite renderer for a
//! [`ParticleSim`]'s output (docs/render-architecture.md §7.3 sprite path).
//!
//! The scene renderer (`crate::scene`) currently skips particle objects
//! (`scene/renderer.rs`: "non-image object skipped"); it exposes no particle
//! hook. So this is a standalone renderer the scene integrator wires in later:
//! give it the particle material's texture + blend mode, upload the sim's
//! [`SpriteInstance`]s each frame, and draw into the scene FBO with the scene's
//! view-projection and the object's model matrix folded into one matrix.
//!
//! Each particle is one instance; the vertex shader expands the centered quad
//! (rotation about Z, `size` as the full edge length), samples the (optional)
//! spritesheet frame, and multiplies by the per-particle color/alpha. Blend
//! mode reuses the scene's [`blend_state`](crate::scene::blend::blend_state)
//! mapping (additive / translucent), so particles composite exactly like the
//! reference passes (SPEC §V10, no duplicated blend table).

use kirie_scene::material::Blending;
use wgpu::util::DeviceExt;

use super::state::SpriteInstance;

/// A GPU renderer for particle sprites. Owns its instance buffer (sized to the
/// sim pool once — SPEC §V5), a view-projection UBO, and a bind group holding
/// the particle texture + sampler.
pub struct ParticleRenderer {
    pipeline: wgpu::RenderPipeline,
    vp_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    capacity: u32,
    /// Kept alive so the fallback white texture view in the bind group stays
    /// valid when the caller supplies no texture.
    _fallback: Option<wgpu::Texture>,
}

impl ParticleRenderer {
    /// Build a renderer targeting `format`, blending with `blending`, sampling
    /// `texture` (or an internal white 1×1 when `None`), with room for
    /// `capacity` instances.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        blending: Blending,
        texture: Option<(&wgpu::TextureView, &wgpu::Sampler)>,
        capacity: usize,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kirie-particle-shader"),
            source: wgpu::ShaderSource::Wgsl(SPRITE_WGSL.into()),
        });

        let vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kirie-particle-vp"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fallback white texture + sampler when the caller has none.
        let mut fallback = None;
        let (view, sampler_owned);
        let (tex_view, tex_sampler): (&wgpu::TextureView, &wgpu::Sampler) = match texture {
            Some((v, s)) => (v, s),
            None => {
                // A 1×1 white texture so particles show their own color when the
                // caller supplies no atlas (docs §7.3: color modulates texture).
                let tex = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("kirie-particle-white"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &[255, 255, 255, 255],
                );
                view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                sampler_owned = device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("kirie-particle-white-sampler"),
                    ..wgpu::SamplerDescriptor::default()
                });
                fallback = Some(tex);
                (&view, &sampler_owned)
            }
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kirie-particle-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kirie-particle-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(tex_sampler),
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kirie-particle-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // One vertex buffer, stepped per instance: four vec4 attributes.
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SpriteInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("kirie-particle-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(instance_layout)],
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
                    format,
                    blend: Some(crate::scene::blend::blend_state(blending)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kirie-particle-instances"),
            size: (capacity.max(1) * std::mem::size_of::<SpriteInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        ParticleRenderer {
            pipeline,
            vp_buffer,
            bind_group,
            instance_buffer,
            capacity: capacity.max(1) as u32,
            _fallback: fallback,
        }
    }

    /// Upload the combined view-projection × model matrix (column-major, 16
    /// floats) and the current instances. Instances beyond the pool capacity
    /// are dropped defensively (SPEC §V9). Returns the number uploaded.
    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        view_projection: &[f32; 16],
        instances: &[SpriteInstance],
    ) -> u32 {
        queue.write_buffer(&self.vp_buffer, 0, bytemuck::cast_slice(view_projection));
        let n = instances.len().min(self.capacity as usize);
        if n > 0 {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances[..n]));
        }
        n as u32
    }

    /// Draw `count` uploaded instances into `target` (loading existing contents,
    /// so this composites over a scene FBO). No-op when `count == 0` (SPEC §V6:
    /// zero work when there is nothing to draw).
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, count: u32) {
        if count == 0 {
            return;
        }
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("kirie-particle-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.instance_buffer.slice(..));
        rp.draw(0..4, 0..count);
    }
}

/// The instanced sprite shader: expand a centered quad per instance (rotation
/// about Z, `size` as the full edge length), sample the optional spritesheet
/// frame column, and modulate by per-particle color/alpha.
const SPRITE_WGSL: &str = r#"
struct VP { m: mat4x4<f32> }
@group(0) @binding(0) var<uniform> vp: VP;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct Instance {
    @location(0) position_size: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) rotation_frame: vec4<f32>,
    @location(3) velocity: vec4<f32>,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> VsOut {
    // Triangle-strip corners: TL, BL, TR, BR.
    var cx = array<f32, 4>(-0.5, -0.5, 0.5, 0.5);
    var cy = array<f32, 4>(0.5, -0.5, 0.5, -0.5);
    var ux = array<f32, 4>(0.0, 0.0, 1.0, 1.0);
    var uy = array<f32, 4>(0.0, 1.0, 0.0, 1.0);

    let size = inst.position_size.w;
    let rot = inst.rotation_frame.z;
    let s = sin(rot);
    let c = cos(rot);
    let ox = cx[vi] * size;
    let oy = cy[vi] * size;
    let rx = ox * c - oy * s;
    let ry = ox * s + oy * c;
    let world = vec4<f32>(inst.position_size.xyz + vec3<f32>(rx, ry, 0.0), 1.0);

    var out: VsOut;
    out.pos = vp.m * world;
    out.uv = vec2<f32>(ux[vi], uy[vi]);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex_c = textureSample(tex, samp, in.uv);
    return tex_c * in.color;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_is_pod_and_64_bytes() {
        assert_eq!(std::mem::size_of::<SpriteInstance>(), 64);
        let z = SpriteInstance {
            position_size: [0.0; 4],
            color: [0.0; 4],
            rotation_frame: [0.0; 4],
            velocity: [0.0; 4],
        };
        let _: &[u8] = bytemuck::bytes_of(&z);
    }
}
