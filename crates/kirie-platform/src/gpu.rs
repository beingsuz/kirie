//! wgpu instance/adapter/device bring-up and raw-handle surface creation.
//!
//! This module contains the **only** `unsafe` in the crate (SPEC V2:
//! kirie-platform may use unsafe for raw-window-handle surface creation
//! only), isolated in [`create_wgpu_surface`].

use std::ptr::NonNull;

use raw_window_handle::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Proxy};

use crate::error::PlatformError;

/// Shared GPU context: one instance/adapter/device/queue for every output
/// surface (docs/render-architecture.md §2.3 "wgpu:" note — the portable
/// model is one shared device with one present pass per monitor surface;
/// the C++ driver likewise shares a single EGL context across all
/// per-output EGL window surfaces, WaylandOpenGLDriver.cpp:140-224).
pub(crate) struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Bring up wgpu against the first output's `wl_surface`, returning the
    /// context plus that surface's swapchain handle.
    ///
    /// Backend policy: Vulkan preferred, fall back to `Backends::all()` if
    /// no Vulkan adapter can present to the surface (SPEC §G: wgpu/Vulkan
    /// renderer). Each attempt creates a fresh instance because a surface
    /// is only compatible with adapters from the instance that created it.
    pub fn new_for_surface(
        conn: &Connection,
        wl_surface: &WlSurface,
    ) -> Result<(Self, wgpu::Surface<'static>), PlatformError> {
        let mut last_err: Option<PlatformError> = None;

        for backends in [wgpu::Backends::VULKAN, wgpu::Backends::all()] {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });

            let surface = match create_wgpu_surface(&instance, conn, wl_surface) {
                Ok(surface) => surface,
                Err(err) => {
                    tracing::warn!(?backends, %err, "surface creation failed on backend set");
                    last_err = Some(err);
                    continue;
                }
            };

            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..wgpu::RequestAdapterOptions::default()
            })) {
                Ok(adapter) => {
                    let info = adapter.get_info();
                    tracing::info!(
                        backend = %info.backend,
                        adapter = %info.name,
                        "selected gpu adapter"
                    );
                    let (device, queue) =
                        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                            label: Some("kirie-platform"),
                            ..wgpu::DeviceDescriptor::default()
                        }))?;
                    return Ok((
                        Self {
                            instance,
                            adapter,
                            device,
                            queue,
                        },
                        surface,
                    ));
                }
                Err(err) => {
                    tracing::warn!(?backends, %err, "no adapter for backend set");
                    last_err = Some(err.into());
                }
            }
        }

        // Both attempts recorded an error; report the most recent one.
        Err(last_err.unwrap_or(PlatformError::NullDisplayPointer))
    }

    /// Create a swapchain surface for an additional output using the
    /// already-selected instance.
    pub fn create_surface(
        &self,
        conn: &Connection,
        wl_surface: &WlSurface,
    ) -> Result<wgpu::Surface<'static>, PlatformError> {
        create_wgpu_surface(&self.instance, conn, wl_surface)
    }
}

/// The single unsafe entry point of the crate: wraps
/// [`wgpu::Instance::create_surface_unsafe`] over the raw libwayland
/// pointers of the connection and one `wl_surface` (SPEC V2).
#[allow(unsafe_code)]
fn create_wgpu_surface(
    instance: &wgpu::Instance,
    conn: &Connection,
    wl_surface: &WlSurface,
) -> Result<wgpu::Surface<'static>, PlatformError> {
    // Both pointer accessors are safe; they only exist because the
    // `client_system` (libwayland) backend is compiled in.
    let display =
        NonNull::new(conn.backend().display_ptr().cast()).ok_or(PlatformError::NullDisplayPointer)?;
    let surface = NonNull::new(wl_surface.id().as_ptr().cast()).ok_or(PlatformError::NullSurfacePointer)?;

    let raw_display_handle = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(display));
    let raw_window_handle = RawWindowHandle::Wayland(WaylandWindowHandle::new(surface));

    // SAFETY: `create_surface_unsafe` requires both raw handles to be valid
    // objects and to remain valid until the returned `Surface` is dropped.
    // - Validity: `display` is the live `*mut wl_display` of `conn`'s
    //   libwayland backend and `surface` is the live `*mut wl_proxy` of
    //   `wl_surface`; both were null-checked above, and a non-null
    //   `ObjectId::as_ptr` means the proxy has not been destroyed.
    // - Lifetime: the returned surface is stored in an `OutputContext`
    //   whose field order drops the `wgpu::Surface` before the sctk
    //   `LayerSurface` that owns (and on drop destroys) the `wl_surface`
    //   (src/output.rs), and `PlatformState` declares its output list
    //   before the `Connection`, so every surface is dropped before the
    //   display connection closes (src/platform.rs). The `'static`
    //   lifetime on the return type is sound under that ownership
    //   discipline. Two supporting facts:
    //   - A surface that never reaches an `OutputContext` (the backend
    //     fallback path in `Gpu::new_for_surface` drops the Vulkan-instance
    //     surface when no adapter is found) dies inside this call, strictly
    //     within the caller's borrows of `conn` and `wl_surface`.
    //   - `Connection` is reference-counted: the `wl_display` stays alive
    //     while *any* clone exists (`PlatformState.conn`, the calloop
    //     `WaylandSource`), so the field-order argument is a conservative
    //     lower bound on the display pointer's validity.
    let surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_display_handle),
            raw_window_handle,
        })
    }?;

    Ok(surface)
}
