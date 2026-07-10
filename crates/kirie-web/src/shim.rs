//! The Wallpaper Engine web JS API bridge, shared by every web backend.
//!
//! WE web wallpapers call a small set of `window.wallpaper*` functions to
//! receive audio, user properties and MPRIS media data. In the reference
//! engine the *renderer-process* half of the bridge is injected once per V8
//! context (`SubprocessApp::OnContextCreated`) and the *browser-process* half
//! pushes data into it via `frame->ExecuteJavaScript`
//! (docs/subsystems-misc.md §3.5). kirie reproduces both halves here in a
//! backend-neutral way:
//!
//! - [`BRIDGE_INIT`] is the renderer-side shim, injected as an *initialization
//!   script* so `wallpaperRegisterAudioListener` &c. exist before the page's
//!   own scripts run.
//! - the `*_call` builders produce the one-line JavaScript statements a
//!   backend evaluates each frame (CEF via `ExecuteJavaScript`, the webview via
//!   `WebView::evaluate_script`) to fire those listeners.
//!
//! Both backends import this module so the shim string and the call encodings
//! are defined exactly once (no duplication). Everything here is pure `std`
//! and compiled in the default build (SPEC V9: string building only, never
//! panics on odd audio/property input).

use std::fmt::Write as _;

/// The renderer-side bridge, injected before page scripts.
///
/// Defines the `wallpaperRegister*Listener` registration functions plus the
/// `__wp*` entry points the backend drives, guarded by `window.__wpBridge` so
/// a double injection is a no-op. Mirrors `SubprocessApp::OnContextCreated`
/// (docs/subsystems-misc.md §3.5): audio + the four MPRIS media listeners, and
/// `__wpApplyProps` / `__wpApplyGeneral` which forward to the page's
/// `wallpaperPropertyListener`.
pub const BRIDGE_INIT: &str = r#"(function(){
  if (window.__wpBridge) { return; }
  window.__wpBridge = true;
  var lists = {};
  function register(name, key) {
    lists[key] = [];
    window[name] = function (cb) { if (typeof cb === 'function') { lists[key].push(cb); } };
  }
  function fire(key, data) {
    var cbs = lists[key] || [];
    for (var i = 0; i < cbs.length; i++) {
      try { cbs[i](data); } catch (e) { /* a broken page listener must not break the bridge */ }
    }
  }
  register('wallpaperRegisterAudioListener', 'audio');
  register('wallpaperRegisterMediaPropertiesListener', 'mprops');
  register('wallpaperRegisterMediaPlaybackListener', 'mplayback');
  register('wallpaperRegisterMediaThumbnailListener', 'mthumb');
  register('wallpaperRegisterMediaTimelineListener', 'mtimeline');
  register('wallpaperRegisterMediaStatusListener', 'mstatus');
  window.__wpAudio = function (d) { fire('audio', d); };
  window.__wpMediaProps = function (d) { fire('mprops', d); };
  window.__wpMediaPlayback = function (d) { fire('mplayback', d); };
  window.__wpMediaThumb = function (d) { fire('mthumb', d); };
  window.__wpMediaTimeline = function (d) { fire('mtimeline', d); };
  window.__wpMediaStatus = function (d) { fire('mstatus', d); };
  window.wallpaperRequestRandomFileForProperty = function (name, cb) {
    if (typeof cb === 'function') { try { cb(name, ''); } catch (e) {} }
  };
  window.__wpApplyProps = function (p) {
    var l = window.wallpaperPropertyListener;
    if (l && typeof l.applyUserProperties === 'function') { l.applyUserProperties(p); }
  };
  window.__wpApplyGeneral = function (p) {
    var l = window.wallpaperPropertyListener;
    if (l && typeof l.applyGeneralProperties === 'function') { l.applyGeneralProperties(p); }
  };
})();"#;

/// Build the per-frame `__wpAudio([...])` call from FFT magnitudes.
///
/// WE delivers **128** floats — 64 bands duplicated as identical left+right
/// channels, each formatted `"%.4f"` (docs/subsystems-misc.md §1.3, §3.5). A
/// 64-length `bands` slice is mirrored to 128; any other length is used
/// verbatim (padded/truncated to at least keep valid JS), so malformed audio
/// input can never panic (SPEC V9).
#[must_use]
pub fn audio_call(bands: &[f32]) -> String {
    // Reproduce the reference layout: 64 bands, twice.
    let mirror = bands.len() == 64;
    let count = if mirror { 128 } else { bands.len() };
    let mut js = String::with_capacity(count * 8 + 16);
    js.push_str("window.__wpAudio([");
    for i in 0..count {
        let v = if mirror { bands[i % 64] } else { bands[i] };
        if i != 0 {
            js.push(',');
        }
        // `{:.4}` matches the reference "%.4f"; guard non-finite to 0.
        let v = if v.is_finite() { v } else { 0.0 };
        let _ = write!(js, "{v:.4}");
    }
    js.push_str("]);");
    js
}

/// Build the one-shot `__wpApplyProps({...})` call.
///
/// `json` must already be a serialized JSON object of the shape
/// `{name: {value: ...}}` (the caller performs the typed color/bool/slider
/// serialization described in docs/subsystems-misc.md §3.5). It is spliced in
/// verbatim.
#[must_use]
pub fn apply_user_properties_call(json: &str) -> String {
    format!("window.__wpApplyProps({json});")
}

/// Build the `__wpApplyGeneral({...})` call (engine/general properties).
#[must_use]
pub fn apply_general_properties_call(json: &str) -> String {
    format!("window.__wpApplyGeneral({json});")
}

/// Build the `__wpMediaProps({title, artist, album})` call from serialized JSON.
#[must_use]
pub fn media_properties_call(json: &str) -> String {
    format!("window.__wpMediaProps({json});")
}

/// Build the `__wpMediaPlayback({state})` call from serialized JSON.
#[must_use]
pub fn media_playback_call(json: &str) -> String {
    format!("window.__wpMediaPlayback({json});")
}

/// Build the `__wpMediaTimeline({position, duration})` call from serialized JSON.
#[must_use]
pub fn media_timeline_call(json: &str) -> String {
    format!("window.__wpMediaTimeline({json});")
}

/// Build the `__wpMediaThumb({thumbnail, primaryColor, ...})` call from JSON.
#[must_use]
pub fn media_thumbnail_call(json: &str) -> String {
    format!("window.__wpMediaThumb({json});")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_mirrors_64_bands_to_128() {
        let bands: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let js = audio_call(&bands);
        // 128 comma-separated values → 127 commas between them.
        assert_eq!(js.matches(',').count(), 127);
        assert!(js.starts_with("window.__wpAudio(["));
        assert!(js.ends_with("]);"));
    }

    #[test]
    fn audio_handles_non_finite_without_panic() {
        let bands = [f32::NAN, f32::INFINITY, -1.0, 2.0];
        let js = audio_call(&bands);
        assert!(js.contains("0.0000"));
        // Non-64 length is used verbatim: 4 values, 3 commas.
        assert_eq!(js.matches(',').count(), 3);
    }

    #[test]
    fn property_calls_splice_json() {
        assert_eq!(
            apply_user_properties_call("{\"a\":{\"value\":1}}"),
            "window.__wpApplyProps({\"a\":{\"value\":1}});"
        );
    }
}
