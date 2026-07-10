//! Per-output render context: one layer-shell surface plus one wgpu
//! swapchain per requested output, mirroring the C++
//! `WaylandOutputViewport` (one `wl_surface` + layer surface + EGL window
//! surface per output, docs/render-architecture.md §2.3).

use std::time::Instant;

use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::wlr_layer::LayerSurface;
use wayland_client::protocol::wl_output::WlOutput;
use wayland_client::protocol::wl_surface::WlSurface;

use crate::renderer::{Renderer, SurfaceSize};

/// Everything owned per output (SPEC V1: state owned and passed explicitly,
/// no globals; structured for reuse by the P3/P4 renderers).
///
/// Field order is load-bearing: `wgpu_surface` is declared before `layer`
/// so the swapchain (which borrows the raw `wl_surface` pointer) is dropped
/// before the sctk `LayerSurface` destroys the `wl_surface`. The layer-shell
/// protocol in turn requires destroying the role object before the surface,
/// which sctk's `LayerSurface` drop handles.
pub(crate) struct OutputContext {
    /// wgpu swapchain surface. `None` until the wgpu instance exists.
    /// Must precede `layer` (drop order, see above).
    pub wgpu_surface: Option<wgpu::Surface<'static>>,
    /// Layer-shell surface; owns the underlying `wl_surface`.
    pub layer: LayerSurface,
    /// The output this surface is pinned to.
    pub wl_output: WlOutput,
    /// Compositor-reported name (e.g. `DP-1`); empty string when unknown.
    pub name: String,
    /// Integer buffer scale from `wl_output`
    /// (docs/render-architecture.md §2.3: buffers are `w*scale × h*scale`
    /// physical pixels with `wl_surface_set_buffer_scale(scale)`).
    pub scale: u32,
    /// Logical size from the latest layer-surface configure, in
    /// surface-local coordinates.
    pub logical_size: (u32, u32),
    /// `logical_size × scale`; the swapchain extent.
    pub physical_size: SurfaceSize,
    /// The first configure has arrived and the swapchain is configured;
    /// only then may buffers be attached (layer-shell requires an
    /// acked configure before the first commit with a buffer).
    pub configured: bool,
    /// A `wl_surface.frame` callback is in flight; don't request another
    /// until it fires (one render per callback,
    /// docs/render-architecture.md §2.3).
    pub frame_pending: bool,
    /// Whether the first frame for this output has been presented (logged
    /// once).
    pub first_frame_presented: bool,
    /// App-supplied frame producer; built lazily once the GPU exists.
    pub renderer: Option<Box<dyn Renderer>>,
    /// Timestamp of the previous presented frame, for per-output `dt`
    /// (each monitor renders at its own compositor-driven cadence,
    /// docs/render-architecture.md §2.3).
    pub last_frame: Option<Instant>,
}

impl OutputContext {
    /// The `wl_surface` backing this output's layer surface.
    pub fn wl_surface(&self) -> &WlSurface {
        self.layer.wl_surface()
    }

    /// Recompute `physical_size` from `logical_size` and `scale` with
    /// saturating math (compositor-provided values; SPEC V9 discipline:
    /// no unchecked arithmetic on externally supplied sizes).
    pub fn update_physical_size(&mut self) {
        self.physical_size = SurfaceSize {
            width: self.logical_size.0.saturating_mul(self.scale).max(1),
            height: self.logical_size.1.saturating_mul(self.scale).max(1),
        };
    }
}
