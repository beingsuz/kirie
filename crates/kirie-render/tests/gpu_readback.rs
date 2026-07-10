//! Headless GPU verification of the image pipeline: render synthetic
//! content into an offscreen target and assert the exact pixels — quad
//! orientation, atlas placement + frame advance (docs/format-tex.md §8.1),
//! and border-mode letterboxing (docs/render-architecture.md §4).
//!
//! Skipped (with a note) when no wgpu adapter is available.

use kirie_platform::{RenderTarget, Renderer, SurfaceSize};
use kirie_render::{
    ClampMode, FramePlacement, ImageContent, ImageOptions, ImagePage, ImageRenderer, SamplerSpec, ScalingMode,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        Ok(adapter) => adapter,
        Err(err) => {
            eprintln!("skipping gpu test: no adapter ({err})");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kirie-render-test"),
        ..wgpu::DeviceDescriptor::default()
    })) {
        Ok((device, queue)) => Some((device, queue)),
        Err(err) => {
            eprintln!("skipping gpu test: no device ({err})");
            None
        }
    }
}

/// Render one frame with `renderer` into a fresh `width`×`height` offscreen
/// target and read back the RGBA rows.
fn render_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut ImageRenderer,
    width: u32,
    height: u32,
    dt: f32,
) -> Vec<u8> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("readback-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    renderer.render(&view, SurfaceSize { width, height }, dt);

    // 256-byte row alignment for texture→buffer copies.
    let padded_row = 256u32;
    assert!(width * 4 <= padded_row);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-buffer"),
        size: u64::from(padded_row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    buffer.map_async(wgpu::MapMode::Read, .., |result| result.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

    let mapped = buffer.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded_row) as usize;
        pixels.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

fn content(pages: Vec<ImagePage>, frames: Vec<FramePlacement>, size: (u32, u32)) -> ImageContent {
    ImageContent {
        pages,
        frames,
        sampler: SamplerSpec {
            nearest: true,
            clamp_uvs: true,
        },
        content_width: size.0,
        content_height: size.1,
    }
}

const RED: [u8; 4] = [255, 0, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const WHITE: [u8; 4] = [255, 255, 255, 255];

#[test]
fn stretch_renders_content_upright() {
    let Some((device, queue)) = gpu() else { return };
    // 1x2 page: red on top, blue below. UvWindow v0 is the viewport's top
    // edge (reference Wayland vflip convention,
    // docs/render-architecture.md §2.4, §4).
    let page = ImagePage {
        width: 1,
        height: 2,
        pixels: [RED, BLUE].concat(),
    };
    let frame = FramePlacement {
        page: 0,
        duration: 0.0,
        translation: [0.0, 0.0],
        axes: [1.0, 0.0, 0.0, 1.0],
    };
    let target = RenderTarget {
        device: &device,
        queue: &queue,
        format: FORMAT,
        output_name: "test",
    };
    let mut renderer = ImageRenderer::new(
        &target,
        &content(vec![page], vec![frame], (1, 2)),
        ImageOptions {
            scaling: ScalingMode::Stretch,
            clamp: ClampMode::Clamp,
        },
    )
    .expect("renderer");

    let pixels = render_and_read(&device, &queue, &mut renderer, 2, 2, 0.0);
    // Top row red, bottom row blue — not upside down.
    assert_eq!(&pixels[0..4], &RED, "top-left");
    assert_eq!(&pixels[4..8], &RED, "top-right");
    assert_eq!(&pixels[8..12], &BLUE, "bottom-left");
    assert_eq!(&pixels[12..16], &BLUE, "bottom-right");
}

#[test]
fn atlas_frames_advance_on_schedule() {
    let Some((device, queue)) = gpu() else { return };
    // 2x1 atlas: frame 0 = left (red) texel, frame 1 = right (blue) texel,
    // 0.5 s each (§8.1 placements over a single page).
    let page = ImagePage {
        width: 2,
        height: 1,
        pixels: [RED, BLUE].concat(),
    };
    let axes = [0.5, 0.0, 0.0, 1.0];
    let frames = vec![
        FramePlacement {
            page: 0,
            duration: 0.5,
            translation: [0.0, 0.0],
            axes,
        },
        FramePlacement {
            page: 0,
            duration: 0.5,
            translation: [0.5, 0.0],
            axes,
        },
    ];
    let target = RenderTarget {
        device: &device,
        queue: &queue,
        format: FORMAT,
        output_name: "test",
    };
    let mut renderer = ImageRenderer::new(
        &target,
        &content(vec![page], frames, (1, 1)),
        ImageOptions {
            scaling: ScalingMode::Stretch,
            clamp: ClampMode::Clamp,
        },
    )
    .expect("renderer");
    assert!(renderer.is_animated());

    // t = 0 → frame 0 (red fills the target).
    let pixels = render_and_read(&device, &queue, &mut renderer, 4, 4, 0.0);
    for (i, px) in pixels.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(px, &RED, "pixel {i} at t=0");
    }
    // t = 0.6 → frame 1 (§8.1 walk: 0.6-0.5 > 0, next frame) — the frame
    // uniform is rewritten and blue fills the target.
    let pixels = render_and_read(&device, &queue, &mut renderer, 4, 4, 0.6);
    for (i, px) in pixels.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(px, &BLUE, "pixel {i} at t=0.6");
    }
    // Half a second later the total elapsed is 1.1 s, which the §8.1 fmod
    // wraps to 0.1 s: back to red.
    let pixels = render_and_read(&device, &queue, &mut renderer, 4, 4, 0.5);
    for (i, px) in pixels.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(px, &RED, "pixel {i} at t=1.1 (wrapped to 0.1)");
    }
}

#[test]
fn fit_with_border_letterboxes_transparent_black() {
    let Some((device, queue)) = gpu() else { return };
    // Square white content on a 2:1 viewport, fit + border: u window is
    // [-0.5, 1.5] (docs/render-architecture.md §4), so the outer quarters
    // sample the border — GL default transparent black (CFBO.cpp:31-33).
    let page = ImagePage {
        width: 1,
        height: 1,
        pixels: WHITE.to_vec(),
    };
    let frame = FramePlacement {
        page: 0,
        duration: 0.0,
        translation: [0.0, 0.0],
        axes: [1.0, 0.0, 0.0, 1.0],
    };
    let target = RenderTarget {
        device: &device,
        queue: &queue,
        format: FORMAT,
        output_name: "test",
    };
    let mut renderer = ImageRenderer::new(
        &target,
        &content(vec![page], vec![frame], (1, 1)),
        ImageOptions {
            scaling: ScalingMode::Fit,
            clamp: ClampMode::Border,
        },
    )
    .expect("renderer");

    let pixels = render_and_read(&device, &queue, &mut renderer, 4, 2, 0.0);
    for row in 0..2usize {
        let base = row * 16;
        assert_eq!(&pixels[base..base + 4], &[0, 0, 0, 0], "row {row} left bar");
        assert_eq!(&pixels[base + 4..base + 8], &WHITE, "row {row} content left");
        assert_eq!(&pixels[base + 8..base + 12], &WHITE, "row {row} content right");
        assert_eq!(
            &pixels[base + 12..base + 16],
            &[0, 0, 0, 0],
            "row {row} right bar"
        );
    }
}
