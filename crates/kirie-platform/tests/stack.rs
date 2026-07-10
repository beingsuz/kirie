//! Environment-dependent integration tests.
//!
//! This crate has no corpus-dependent behavior (it renders, it does not
//! parse wallpaper formats), so instead of the corpus guard the tests
//! guard on what they actually need — a GPU adapter or a live wayland
//! session — and skip (eprintln + return) when absent so CI stays green.

use std::time::Duration;

use kirie_platform::{Backend, Platform, RenderTarget, Renderer, SurfaceSize, TestPattern};

/// Headless GPU bring-up: adapter → device without any surface. Skips when
/// the environment has no usable adapter (e.g. bare CI).
fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        Ok(adapter) => adapter,
        Err(err) => {
            eprintln!("skipping: no gpu adapter available ({err})");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kirie-platform-test"),
        ..wgpu::DeviceDescriptor::default()
    })) {
        Ok(pair) => Some(pair),
        Err(err) => {
            eprintln!("skipping: gpu device creation failed ({err})");
            None
        }
    }
}

/// The TestPattern pipeline must build and render two frames into an
/// offscreen target without validation errors — proves the WGSL shader,
/// pipeline state, and encode/submit path independent of any compositor.
#[test]
fn test_pattern_renders_headless() {
    let Some((device, queue)) = headless_device() else {
        return;
    };

    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let mut pattern = TestPattern::new(&RenderTarget {
        device: &device,
        queue: &queue,
        format,
        output_name: "offscreen-test",
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kirie-platform-test-target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let size = SurfaceSize {
        width: 64,
        height: 32,
    };

    // Two frames: first with dt = 0 (first-frame contract), then an
    // animated step, mirroring how the platform drives renderers.
    pattern.render(&view, size, 0.0);
    pattern.render(&view, size, 1.0 / 60.0);

    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "validation error: {error:?}");

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
}

/// On a live wayland session, connecting and binding the required globals
/// (wl_compositor, zwlr_layer_shell_v1) must succeed. Skips when no
/// session is available. Does not run the event loop, so no surfaces are
/// ever committed and nothing flashes on screen.
#[test]
fn platform_connects_on_live_session() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: WAYLAND_DISPLAY not set (no wayland session)");
        return;
    }

    let platform = Platform::connect(Box::new(|target| {
        Box::new(TestPattern::new(target)) as Box<dyn Renderer>
    }));
    match platform {
        Ok(platform) => {
            // No dispatch has run yet, so no outputs can have surfaces.
            assert_eq!(platform.output_count(), 0);
        }
        Err(err) => panic!("connect failed on live session: {err}"),
    }
}

/// On a live X11 (or Xwayland) session, the X11 backend must connect,
/// enumerate at least one RANDR monitor, create a desktop window + wgpu
/// surface per monitor, and drive the [`TestPattern`] for ~2 seconds before
/// exiting cleanly. Forces `Backend::X11` because this machine also exposes a
/// wayland session, which the env heuristic would otherwise prefer. Skips when
/// `$DISPLAY` is unset or no GPU adapter can present to an xcb surface.
#[test]
fn x11_backend_renders_live() {
    if std::env::var_os("DISPLAY").is_none() {
        eprintln!("skipping: DISPLAY not set (no X11/Xwayland session)");
        return;
    }

    let platform = Platform::connect_backend(
        Backend::X11,
        Box::new(|target| Box::new(TestPattern::new(target)) as Box<dyn Renderer>),
    );

    let mut platform = match platform {
        Ok(platform) => platform,
        Err(err) => {
            // No adapter / no CRTC in this environment is a skip, not a
            // failure — CI without a GPU or without RANDR must stay green.
            eprintln!("skipping: X11 backend bring-up unavailable ({err})");
            return;
        }
    };

    assert!(
        platform.output_count() >= 1,
        "expected at least one X11 monitor window"
    );

    platform
        .run(Some(Duration::from_secs(2)))
        .expect("X11 render loop should run for the deadline and exit 0");
}
