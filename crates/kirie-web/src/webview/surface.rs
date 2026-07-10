//! The native surface a [`WebviewBackend`](super::WebviewBackend) renders into.
//!
//! Unlike the CEF backend (which paints off-screen into a CPU buffer), wry +
//! webkit2gtk draws straight into a real window. The host (kirie-platform)
//! therefore hands the backend the background window it should fill: on
//! Wayland that is a layer-shell surface (in practice a GTK window promoted to
//! the background layer via `gtk-layer-shell`), on X11 the desktop/root-child
//! window. Both are described to us through `raw-window-handle` handles.

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle,
};

/// The background surface to render the web wallpaper into.
///
/// The handles must come from a window the host keeps alive for at least as
/// long as the [`WebviewBackend`](super::WebviewBackend) — wry attaches its
/// native web view as a child of it and does not take ownership.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceTarget {
    window: RawWindowHandle,
    display: RawDisplayHandle,
}

impl SurfaceTarget {
    /// Wrap raw window + display handles for the background surface.
    ///
    /// # Safety
    ///
    /// The caller guarantees both handles are valid and refer to a window that
    /// outlives every use of the resulting [`SurfaceTarget`] and of any
    /// [`WebviewBackend`](super::WebviewBackend) built from it. On Linux the
    /// window must be GTK-backed (webkit2gtk cannot attach to a bare Wayland
    /// or X11 surface); kirie-platform is responsible for creating that
    /// GTK/gtk-layer-shell window.
    #[must_use]
    pub unsafe fn new(window: RawWindowHandle, display: RawDisplayHandle) -> Self {
        Self { window, display }
    }
}

// SAFETY: `SurfaceTarget::new` is `unsafe` precisely so the caller upholds the
// validity/lifetime contract these two impls rely on. We only ever hand back
// borrowed handles for the duration of `&self`, and the wrapped raw handles are
// `Copy` plain data.
impl HasWindowHandle for SurfaceTarget {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        // SAFETY: validity + lifetime are guaranteed by the caller of
        // `SurfaceTarget::new`; the borrow is scoped to `&self`.
        Ok(unsafe { WindowHandle::borrow_raw(self.window) })
    }
}

impl HasDisplayHandle for SurfaceTarget {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // SAFETY: same contract as `window_handle`; the borrow is scoped to
        // `&self` and the raw display handle is caller-guaranteed valid.
        Ok(unsafe { DisplayHandle::borrow_raw(self.display) })
    }
}
