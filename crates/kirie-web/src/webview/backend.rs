//! The wry + webkit2gtk web-wallpaper backend.
//!
//! See the module docs ([`super`]) for the rendering model. This file owns the
//! live [`wry::WebView`], drives the WE JS bridge each frame, and forwards
//! pointer input as synthetic DOM events.

use wry::dpi::{PhysicalPosition, PhysicalSize};
use wry::{Rect, WebView, WebViewBuilder};

use crate::backend::{PointerState, WebError, WebFrameRef, WebSize};
use crate::shim;

use super::surface::SurfaceTarget;

/// webkit2gtk exposes no pixel read-back and wry no OSR path, so muting is done
/// in JavaScript. This init script installs a bridge that (1) keeps every
/// `<audio>`/`<video>` element's `muted` flag in sync with `window.__wpMuteState`
/// (existing and future, via a `MutationObserver`), and (2) routes every
/// `AudioContext.destination` through a gain node so Web Audio output (used by
/// visualiser wallpapers, which bypass media-element muting) can be silenced
/// too. The literal `MUTESTATE` token is replaced with the initial bool.
const MUTE_INIT: &str = r#"(function(){
  if (window.__wpMuteInit) { window.__wpMuteState = MUTESTATE; return; }
  window.__wpMuteInit = true;
  window.__wpMuteState = MUTESTATE;
  function applyEl(el){ try { el.muted = window.__wpMuteState; } catch(e){} }
  window.__wpSweepMute = function(){
    var m = document.querySelectorAll('audio,video');
    for (var i = 0; i < m.length; i++) { applyEl(m[i]); }
  };
  document.addEventListener('DOMContentLoaded', window.__wpSweepMute);
  try {
    new MutationObserver(window.__wpSweepMute)
      .observe(document.documentElement, { childList: true, subtree: true });
  } catch(e) {}
  var AC = window.AudioContext || window.webkitAudioContext;
  if (AC && !AC.__wpPatched) {
    AC.__wpPatched = true;
    var desc = Object.getOwnPropertyDescriptor(AC.prototype, 'destination');
    if (desc && desc.get) {
      Object.defineProperty(AC.prototype, 'destination', {
        configurable: true,
        get: function () {
          if (!this.__wpMuteGain) {
            var real = desc.get.call(this);
            var g = this.createGain();
            g.connect(real);
            this.__wpMuteGain = g;
          }
          this.__wpMuteGain.gain.value = window.__wpMuteState ? 0 : 1;
          return this.__wpMuteGain;
        }
      });
    }
  }
})();"#;

/// Build the combined initialization script: WE bridge + mute bridge.
fn init_script(muted: bool) -> String {
    let mute = MUTE_INIT.replace("MUTESTATE", if muted { "true" } else { "false" });
    format!("{}\n{}", shim::BRIDGE_INIT, mute)
}

/// A live web wallpaper rendered by wry into a native background surface.
///
/// This type is intentionally **not** the object-safe, `Send` `WebBackend`
/// trait from [`crate::backend`]: a `wry::WebView`/webkit2gtk object is `!Send`
/// (it must live on the GTK main thread) and produces no CPU frame, so the
/// off-screen `WebBackend` contract does not apply. The method set below mirrors
/// that trait as closely as the native-surface model allows.
pub struct WebviewBackend {
    webview: Option<WebView>,
    size: WebSize,
    muted: bool,
    last_pointer: PointerState,
}

impl WebviewBackend {
    /// Launch a web wallpaper on `url`, rendering into `surface`.
    ///
    /// `url` should be a `file://` URL to the entry page (see
    /// [`file_url`]) so the page's relative asset references resolve against
    /// its own directory. `size` is the initial surface size; `muted` sets the
    /// starting audio state (honouring `--silent`).
    ///
    /// # Errors
    ///
    /// Returns [`WebError::BrowserCreation`] if wry fails to construct the web
    /// view (most often: no GTK/webkit2gtk runtime, or an invalid surface
    /// handle).
    pub fn with_surface(
        url: &str,
        size: WebSize,
        surface: &SurfaceTarget,
        muted: bool,
    ) -> Result<Self, WebError> {
        let size = size.clamped();
        // Make the model unmistakable at runtime: this backend paints its own
        // native surface and can never composite through the wgpu presentation
        // layer (wry/webkit2gtk has no off-screen path — see the module docs;
        // won't-fix upstream). The CEF backend is the composited one.
        tracing::warn!(
            url,
            "webview (wry/webkit2gtk) backend: native-surface fallback only; it cannot \
             render off-screen (upstream wry/webkit2gtk limitation) — build with the \
             `cef` feature (kirie: --features web-cef) for composited web wallpapers"
        );
        let webview = build_webview(url, surface, muted)?;
        let backend = Self {
            webview: Some(webview),
            size,
            muted,
            last_pointer: PointerState::default(),
        };
        // On X11 wry never auto-resizes; set the initial bounds explicitly so
        // the page fills the surface from the first frame.
        backend.apply_bounds();
        Ok(backend)
    }

    /// Push `self.size` to the web view as its bounds. Required on X11 (wry does
    /// not auto-resize there); a no-op-effect on Wayland/GTK where the view
    /// tracks its parent, but harmless.
    fn apply_bounds(&self) {
        let Some(webview) = self.webview.as_ref() else {
            return;
        };
        let rect = Rect {
            position: PhysicalPosition::<i32>::new(0, 0).into(),
            size: PhysicalSize::new(self.size.width, self.size.height).into(),
        };
        if let Err(e) = webview.set_bounds(rect) {
            tracing::debug!(error = %e, "webview set_bounds failed");
        }
    }

    /// Always [`None`]: the webview renders directly into its surface and never
    /// produces a CPU frame (native-surface model, see [`super`]).
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn latest_frame(&self) -> Option<WebFrameRef<'_>> {
        None
    }

    /// Advance one presentation step. `dt` is unused: webkit2gtk renders on the
    /// host's GTK/event loop, not on a tick we pump. Kept for interface parity.
    #[allow(clippy::unused_self)]
    pub fn tick(&mut self, _dt: f32) {}

    /// Resize the surface. On X11 wry does not auto-resize, so the new size is
    /// pushed to the web view via `set_bounds`; on Wayland/GTK the view already
    /// tracks its parent and the call is harmless.
    pub fn resize(&mut self, size: WebSize) {
        self.size = size.clamped();
        self.apply_bounds();
    }

    /// Push one audio frame to the page's registered audio listeners.
    pub fn push_audio(&mut self, bands: &[f32]) {
        self.evaluate(&shim::audio_call(bands));
    }

    /// Apply user properties (`{name: {value: ...}}` JSON) to the page.
    pub fn apply_user_properties(&mut self, json: &str) {
        self.evaluate(&shim::apply_user_properties_call(json));
    }

    /// Apply general/engine properties to the page.
    pub fn apply_general_properties(&mut self, json: &str) {
        self.evaluate(&shim::apply_general_properties_call(json));
    }

    /// Mute or unmute page audio. Flips `window.__wpMuteState` and re-sweeps
    /// media elements; Web Audio gains pick the new value up on next access.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        let state = if muted { "true" } else { "false" };
        self.evaluate(&format!(
            "window.__wpMuteState={state};if(window.__wpSweepMute){{window.__wpSweepMute();}}"
        ));
    }

    /// Forward a pointer sample as synthetic DOM events.
    ///
    /// A background layer/desktop window rarely holds pointer focus, so real
    /// webkit pointer events usually never arrive; dispatching synthetic
    /// `MouseEvent`s lets pages that listen on `document`/`window` still react
    /// (docs/subsystems-misc.md §3.5: position every frame, click on state
    /// change — no keyboard, no scroll).
    pub fn send_pointer(&mut self, pointer: PointerState) {
        self.evaluate(&mouse_move_call(pointer.x, pointer.y));
        if pointer.left != self.last_pointer.left {
            self.evaluate(&mouse_button_call(pointer.x, pointer.y, 0, pointer.left));
        }
        if pointer.right != self.last_pointer.right {
            self.evaluate(&mouse_button_call(pointer.x, pointer.y, 2, pointer.right));
        }
        self.last_pointer = pointer;
    }

    /// Tear the web view down. Idempotent.
    pub fn shutdown(&mut self) {
        // Dropping the `WebView` destroys the native webkit view.
        self.webview = None;
    }

    /// The initial (or last-set) audio mute state.
    #[must_use]
    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Evaluate `js` in the page, logging (never propagating) any error so a
    /// broken page or a torn-down view cannot take the wallpaper down (V9).
    fn evaluate(&self, js: &str) {
        let Some(webview) = self.webview.as_ref() else {
            return;
        };
        if let Err(e) = webview.evaluate_script(js) {
            tracing::debug!(error = %e, "webview evaluate_script failed");
        }
    }
}

impl Drop for WebviewBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Construct the wry web view attached to `surface`.
fn build_webview(url: &str, surface: &SurfaceTarget, muted: bool) -> Result<WebView, WebError> {
    let builder = WebViewBuilder::new()
        .with_url(url)
        .with_initialization_script(init_script(muted))
        // WE pages start audio/video with no user gesture
        // (docs/subsystems-misc.md §3.3: `--autoplay-policy=no-user-gesture-required`).
        .with_autoplay(true)
        // A wallpaper is opaque; an opaque black background avoids the
        // compositor showing through before first paint.
        .with_transparent(false)
        .with_background_color((0, 0, 0, 255));

    // `SurfaceTarget` supplies the window (and display) handle `build` needs.
    // The shared `WebError::BrowserCreation` carries no context, so log the
    // wry detail before mapping to it.
    builder.build(surface).map_err(|e| {
        tracing::error!(error = %e, url, "wry WebView build failed");
        WebError::BrowserCreation
    })
}

/// Build a synthetic `mousemove` dispatch at `(x, y)` browser pixels.
fn mouse_move_call(x: i32, y: i32) -> String {
    format!(
        "(function(){{var t=document.elementFromPoint({x},{y})||document;\
t.dispatchEvent(new MouseEvent('mousemove',\
{{clientX:{x},clientY:{y},bubbles:true,cancelable:true,view:window}}));}})();"
    )
}

/// Build a synthetic mouse button dispatch. `button` is the DOM button index
/// (0 = left, 2 = right); `down` selects `mousedown`+`click` vs `mouseup`.
fn mouse_button_call(x: i32, y: i32, button: i32, down: bool) -> String {
    let mut js = format!(
        "(function(){{var t=document.elementFromPoint({x},{y})||document;\
t.dispatchEvent(new MouseEvent('{ev}',\
{{clientX:{x},clientY:{y},button:{button},bubbles:true,cancelable:true,view:window}}));",
        ev = if down { "mousedown" } else { "mouseup" },
    );
    if !down {
        // A release completes a click.
        js.push_str(&format!(
            "t.dispatchEvent(new MouseEvent('click',\
{{clientX:{x},clientY:{y},button:{button},bubbles:true,cancelable:true,view:window}}));"
        ));
    }
    js.push_str("})();");
    js
}
