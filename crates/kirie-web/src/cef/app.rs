//! The CEF [`App`] — process-wide handlers.
//!
//! One `App` type serves both processes (docs/subsystems-misc.md §3.1):
//!
//! * **Browser process** (`cef::initialize`): `on_before_command_line_processing`
//!   sets the Chromium flags the reference uses for offscreen Wayland
//!   rendering plus the `file://` access relaxations this port needs
//!   (docs/subsystems-misc.md §3.3).
//! * **Render process** (`cef::execute_process` in the helper): returns a
//!   [`RenderProcessHandler`] whose `on_context_created` injects the WE JS
//!   bridge shim before page scripts run (docs/subsystems-misc.md §3.5).

use cef::{
    App, Browser, CefString, CommandLine, Frame, ImplApp, ImplCommandLine, ImplFrame,
    ImplRenderProcessHandler, RenderProcessHandler, V8Context, WrapApp, WrapRenderProcessHandler, rc::Rc,
    wrap_app, wrap_render_process_handler,
};

use crate::shim::BRIDGE_INIT;

/// Whether the running session is Wayland (adds ozone flags, §3.3).
fn is_wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
        || std::env::var("XDG_SESSION_TYPE")
            .map(|v| v == "wayland")
            .unwrap_or(false)
}

fn switch(cmd: &CommandLine, name: &str) {
    let name = CefString::from(name);
    cmd.append_switch(Some(&name));
}

fn switch_val(cmd: &CommandLine, name: &str, value: &str) {
    let name = CefString::from(name);
    let value = CefString::from(value);
    cmd.append_switch_with_value(Some(&name), Some(&value));
}

// Injects the WE JS bridge shim into every V8 context as it is created. The
// `wrap_*` macros do not accept attributes on the generated struct, so this is
// a plain comment and the struct is private (documented via `make_app`).
wrap_render_process_handler! {
    struct ShimRenderProcessHandler {}

    impl RenderProcessHandler {
        fn on_context_created(
            &self,
            _browser: Option<&mut Browser>,
            frame: Option<&mut Frame>,
            _context: Option<&mut V8Context>,
        ) {
            if let Some(frame) = frame {
                let code = CefString::from(BRIDGE_INIT);
                // start_line 0; no script url (internal).
                frame.execute_java_script(Some(&code), None, 0);
            }
        }
    }
}

// The kirie CEF application object (browser + render process). Private for the
// same macro-attribute reason as above; the public entry point is `make_app`.
wrap_app! {
    struct KirieApp {}

    impl App {
        fn on_before_command_line_processing(
            &self,
            _process_type: Option<&CefString>,
            command_line: Option<&mut CommandLine>,
        ) {
            let Some(cmd) = command_line else { return; };

            // --- offscreen / trust / throttling (docs §3.3) ---------------
            switch_val(cmd, "disable-features",
                "IsolateOrigins,HardwareMediaKeyHandling,WebContentsOcclusion,\
                 RendererCodeIntegrityEnabled,site-per-process");
            switch(cmd, "disable-gpu-shader-disk-cache");
            switch(cmd, "disable-site-isolation-trials");
            switch(cmd, "disable-web-security");
            switch_val(cmd, "remote-allow-origins", "*");
            switch_val(cmd, "autoplay-policy", "no-user-gesture-required");
            switch(cmd, "disable-background-timer-throttling");
            switch(cmd, "disable-backgrounding-occluded-windows");
            switch(cmd, "disable-background-media-suspend");
            switch(cmd, "disable-renderer-backgrounding");
            switch(cmd, "disable-breakpad");
            switch(cmd, "disable-field-trial-config");
            switch(cmd, "no-experiments");

            // Local `file://` wallpapers: let the entry page fetch/XHR/module
            // its own sibling assets (WE uses the secure `wp://` scheme; the
            // port loads `file://`, so it must relax the file-origin rules).
            // `disable-web-security` is already appended above.
            switch(cmd, "allow-file-access-from-files");

            // Standalone GPU process crashes on Wayland offscreen; keep it
            // in-process unless the escape hatch is set (docs §3.3).
            if std::env::var_os("WPE_CEF_NO_IPG").is_none() {
                switch(cmd, "in-process-gpu");
            }

            if is_wayland() {
                let ozone = std::env::var("WPE_CEF_OZONE").unwrap_or_else(|_| "wayland".into());
                switch_val(cmd, "ozone-platform", &ozone);
                switch_val(cmd, "enable-features", "UseOzonePlatform");
                match std::env::var("WPE_CEF_ANGLE").as_deref() {
                    Ok("skip") => {}
                    Ok(v) => switch_val(cmd, "use-angle", v),
                    Err(_) => switch_val(cmd, "use-angle", "gl-egl"),
                }
            }
        }

        fn render_process_handler(&self) -> Option<RenderProcessHandler> {
            Some(ShimRenderProcessHandler::new())
        }
    }
}

/// Construct a fresh [`App`] for either process.
#[must_use]
pub fn make_app() -> App {
    KirieApp::new()
}
