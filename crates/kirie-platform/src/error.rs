//! Typed errors for the wayland presentation layer (SPEC V9: no panics on
//! external input — compositor behavior included).

use thiserror::Error;

/// Errors surfaced by the wayland presentation layer.
///
/// Every fallible interaction with the compositor or the GPU maps to a
/// variant here; handlers that cannot propagate errors log them instead
/// (SPEC V9).
#[derive(Debug, Error)]
pub enum PlatformError {
    /// Connecting to the wayland display failed (e.g. no `$WAYLAND_DISPLAY`).
    #[error("failed to connect to the wayland display: {0}")]
    Connect(#[from] wayland_client::ConnectError),

    /// Initial registry/global enumeration failed.
    #[error("failed to enumerate wayland globals: {0}")]
    Globals(#[from] wayland_client::globals::GlobalError),

    /// A required global (`wl_compositor`, `zwlr_layer_shell_v1`, …) is
    /// missing or too old. The layer-shell requirement mirrors the C++
    /// wayland driver, which binds `wlr-layer-shell` unconditionally
    /// (docs/render-architecture.md §2.3).
    #[error("required wayland global unavailable: {0}")]
    Bind(#[from] wayland_client::globals::BindError),

    /// The libwayland `wl_display` pointer was null. Only possible if the
    /// system (libwayland) backend is not active.
    #[error("wl_display pointer is null; the libwayland client backend is required")]
    NullDisplayPointer,

    /// The `wl_proxy` pointer for a `wl_surface` was null (surface already
    /// destroyed).
    #[error("wl_surface pointer is null; the surface was already destroyed")]
    NullSurfacePointer,

    /// wgpu could not create a presentable surface from the raw handles.
    #[error("failed to create wgpu surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),

    /// No adapter accepted the surface on any attempted backend set
    /// (Vulkan preferred, then all backends).
    #[error("no compatible wgpu adapter (tried Vulkan, then all backends): {0}")]
    NoAdapter(#[from] wgpu::RequestAdapterError),

    /// Device creation on the selected adapter failed.
    #[error("failed to create wgpu device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    /// The adapter reports no valid swapchain configuration for a surface.
    #[error("adapter reports no supported configuration for output {output:?}")]
    UnsupportedSurface {
        /// Compositor-reported output name (e.g. `DP-1`), if known.
        output: String,
    },

    /// The calloop event loop failed.
    #[error("event loop error: {0}")]
    EventLoop(#[from] smithay_client_toolkit::reexports::calloop::Error),

    /// Registering the wayland event source in the event loop failed.
    #[error("failed to register wayland source in the event loop: {0}")]
    EventLoopRegister(String),

    /// The compositor closed every layer surface. The C++ reference treats
    /// losing the last layer surface as an abnormal exit so a supervisor can
    /// relaunch (docs/render-architecture.md §2.3,
    /// WaylandOpenGLDriver.cpp:234-274).
    #[error("all layer surfaces were closed by the compositor")]
    AllSurfacesClosed,

    // ── X11 backend (docs/render-architecture.md §2.2) ──────────────────
    /// Connecting to the X server named by `$DISPLAY` failed.
    #[error("failed to connect to the X display: {0}")]
    X11Connect(String),

    /// The libxcb `xcb_connection_t` pointer was null; wgpu's Vulkan
    /// `VK_KHR_xcb_surface` path needs a live connection pointer.
    #[error("xcb_connection_t pointer is null; the libxcb (xcb_ffi) backend is required")]
    NullXcbConnection,

    /// An X11 request/reply round-trip failed (protocol error, connection
    /// dropped, or ID exhaustion). Kept as a string so the crate does not
    /// depend on x11rb's error enum shape (SPEC V9: typed at this boundary).
    #[error("X11 protocol error: {0}")]
    X11Protocol(String),

    /// RANDR reported no usable CRTC, so there is no monitor to place a
    /// wallpaper window on (docs/render-architecture.md §2.2: one viewport
    /// per connected CRTC, X11Output.cpp:111-159).
    #[error("no active RANDR CRTC found; nothing to render on")]
    NoCrtcs,
}
