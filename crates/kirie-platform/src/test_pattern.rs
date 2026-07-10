//! Built-in test-pattern renderer proving the full presentation stack:
//! surface acquire → WGSL pipeline → submit → present.

use crate::renderer::{RenderTarget, Renderer, SurfaceSize};

/// Hue-animated clear color plus a centered gradient quad drawn by a tiny
/// WGSL pipeline. Exists so `examples/layer_clear.rs` (and later e2e
/// harnesses) can validate the wayland/wgpu stack without any wallpaper
/// content.
pub struct TestPattern {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    /// Seconds accumulated from per-frame `dt`, mirroring the C++ driver
    /// clock (seconds since driver start,
    /// docs/render-architecture.md §2.1 step 3).
    time: f32,
}

/// Uniform block for the gradient quad. 16 bytes to satisfy uniform buffer
/// alignment without a layout override.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    time: f32,
    _pad: [f32; 3],
}

/// Centered gradient quad: 4-vertex triangle strip generated from
/// `vertex_index`, no vertex buffers.
const SHADER: &str = r#"
struct Uniforms {
    time: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // Quad covering the middle 50% of NDC, centered at the origin.
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5),
    );
    let c = corners[index];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = c + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pulse = 0.5 + 0.5 * sin(u.time);
    return vec4<f32>(in.uv.x, in.uv.y, pulse, 1.0);
}
"#;

impl TestPattern {
    /// Build the pipeline against the surface `format` of one output.
    #[must_use]
    pub fn new(target: &RenderTarget<'_>) -> Self {
        let device = target.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kirie-test-pattern"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kirie-test-pattern-uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kirie-test-pattern-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kirie-test-pattern-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kirie-test-pattern-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("kirie-test-pattern-pipeline"),
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
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            device: target.device.clone(),
            queue: target.queue.clone(),
            pipeline,
            bind_group,
            uniforms,
            time: 0.0,
        }
    }
}

impl Renderer for TestPattern {
    fn render(&mut self, view: &wgpu::TextureView, _size: SurfaceSize, dt: f32) {
        self.time += dt;

        self.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&Uniforms {
                time: self.time,
                _pad: [0.0; 3],
            }),
        );

        // Slow hue sweep for the clear color; one full revolution every 10s.
        let (r, g, b) = hsv_to_rgb(f64::from(self.time) * 0.1 % 1.0, 0.65, 0.35);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kirie-test-pattern-encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kirie-test-pattern-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..4, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

/// Convert HSV (`h` in `[0, 1)`, `s`/`v` in `[0, 1]`) to RGB in `[0, 1]`.
///
/// Standard sextant formulation; used only for the test-pattern clear
/// color, not for any wallpaper-format color math.
pub(crate) fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let sextant = h.floor().min(5.0);
    let f = h - sextant;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    // `sextant` is clamped to [0, 5]; `as` cast is exact for these values.
    match sextant as u8 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::hsv_to_rgb;

    #[test]
    fn hsv_stays_in_unit_range() {
        let mut i = 0u32;
        while i < 1000 {
            let h = f64::from(i) / 1000.0;
            let (r, g, b) = hsv_to_rgb(h, 0.65, 0.35);
            for c in [r, g, b] {
                assert!((0.0..=1.0).contains(&c), "h={h} produced {c}");
            }
            i += 1;
        }
    }

    #[test]
    fn hsv_primary_hues() {
        // h=0 → red-dominant, h=1/3 → green-dominant, h=2/3 → blue-dominant.
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!(r > g && r > b);
        let (r, g, b) = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(g > r && g > b);
        let (r, g, b) = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert!(b > r && b > g);
    }

    #[test]
    fn hsv_wraps_and_handles_negative_hue() {
        let a = hsv_to_rgb(0.25, 0.5, 0.5);
        let b = hsv_to_rgb(1.25, 0.5, 0.5);
        let c = hsv_to_rgb(-0.75, 0.5, 0.5);
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}
