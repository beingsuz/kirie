//! The CEF [`Client`] and its [`RenderHandler`].
//!
//! Windowless (OSR) rendering only supplies a render handler
//! (docs/subsystems-misc.md §3.5, "BrowserClient only supplies the render
//! handler"). CEF calls `get_view_rect` to learn the offscreen size and
//! `on_paint` with a fresh BGRA buffer whenever the page repaints. `on_paint`
//! copies that buffer into an owned [`FrameBuffer`] and publishes it through a
//! lock-free [`arc_swap`] slot the render thread reads — the browser thread
//! never blocks on the GPU and vice-versa (SPEC §V4).

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use cef::{
    Browser, Client, ImplClient, ImplLifeSpanHandler, ImplRenderHandler, LifeSpanHandler,
    PaintElementType, Rect, RenderHandler, WrapClient, WrapLifeSpanHandler, WrapRenderHandler,
    rc::Rc, wrap_client, wrap_life_span_handler, wrap_render_handler,
};

use crate::backend::{FrameBuffer, FrameSlot, PixelFormat};

/// Off-screen dimensions shared between the backend (writer, on resize) and
/// the render handler (reader, in `get_view_rect`).
#[derive(Debug)]
pub struct SharedSize {
    width: AtomicI32,
    height: AtomicI32,
}

impl SharedSize {
    /// New shared size, clamped to a non-zero view rect.
    #[must_use]
    pub fn new(width: i32, height: i32) -> Arc<Self> {
        Arc::new(Self {
            width: AtomicI32::new(width.max(1)),
            height: AtomicI32::new(height.max(1)),
        })
    }

    /// Update the size the next `get_view_rect` will report.
    pub fn set(&self, width: i32, height: i32) {
        self.width.store(width.max(1), Ordering::Relaxed);
        self.height.store(height.max(1), Ordering::Relaxed);
    }

    /// Current width in pixels.
    #[must_use]
    pub fn width(&self) -> i32 {
        self.width.load(Ordering::Relaxed)
    }

    /// Current height in pixels.
    #[must_use]
    pub fn height(&self) -> i32 {
        self.height.load(Ordering::Relaxed)
    }
}

// OSR paint sink: reports the shared view size and publishes each paint. The
// `wrap_*` macros reject attributes on the struct/fields, so this is a plain
// comment and the struct is private (built via `make_client`).
wrap_render_handler! {
    struct KirieRenderHandler {
        slot: FrameSlot,
        size: Arc<SharedSize>,
    }

    impl RenderHandler {
        fn view_rect(&self, _browser: Option<&mut Browser>, rect: Option<&mut Rect>) {
            if let Some(rect) = rect {
                rect.x = 0;
                rect.y = 0;
                rect.width = self.size.width();
                rect.height = self.size.height();
            }
        }

        fn on_paint(
            &self,
            _browser: Option<&mut Browser>,
            type_: PaintElementType,
            _dirty_rects: Option<&[Rect]>,
            buffer: *const u8,
            width: ::std::os::raw::c_int,
            height: ::std::os::raw::c_int,
        ) {
            // Only the main view; ignore popup (dropdown) paints.
            if type_ != PaintElementType::VIEW {
                return;
            }
            if buffer.is_null() || width <= 0 || height <= 0 {
                return;
            }
            let len = (width as usize) * (height as usize) * 4;
            // SAFETY: CEF guarantees `buffer` points to `width * height * 4`
            // contiguous BGRA bytes for the duration of this callback
            // (docs/subsystems-misc.md §3.5). We copy them out immediately so
            // the borrow does not outlive the call.
            let data = unsafe { std::slice::from_raw_parts(buffer, len) }.to_vec();
            let frame = FrameBuffer {
                data,
                width: width as u32,
                height: height as u32,
                format: PixelFormat::Bgra8,
            };
            self.slot.store(Some(Arc::new(frame)));
        }
    }
}

/// Browsers created and not yet fully closed (`OnAfterCreated` →
/// `OnBeforeClose`). `cef_shutdown` with a browser still alive hangs
/// Chromium's thread teardown — the whole runtime (threads, zygotes, V8
/// heaps) then survives a web→scene switch. The pump drains to zero before
/// shutting the context down.
pub static LIVE_BROWSERS: AtomicUsize = AtomicUsize::new(0);

// Lifespan tracker: counts real browser lifetimes for the shutdown drain.
wrap_life_span_handler! {
    struct KirieLifeSpan;

    impl LifeSpanHandler {
        fn on_after_created(&self, _browser: Option<&mut Browser>) {
            LIVE_BROWSERS.fetch_add(1, Ordering::SeqCst);
        }

        fn on_before_close(&self, _browser: Option<&mut Browser>) {
            LIVE_BROWSERS.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

// The OSR browser client — render handler + lifespan tracker. Private for the
// same macro-attribute reason; the public entry point is `make_client`.
wrap_client! {
    struct KirieClient {
        handler: RenderHandler,
        life: LifeSpanHandler,
    }

    impl Client {
        fn render_handler(&self) -> Option<RenderHandler> {
            Some(self.handler.clone())
        }

        fn life_span_handler(&self) -> Option<LifeSpanHandler> {
            Some(self.life.clone())
        }
    }
}

/// Build a client wired to publish paints into `slot` at `size`.
#[must_use]
pub fn make_client(slot: FrameSlot, size: Arc<SharedSize>) -> Client {
    let handler = KirieRenderHandler::new(slot, size);
    KirieClient::new(handler, KirieLifeSpan::new())
}
