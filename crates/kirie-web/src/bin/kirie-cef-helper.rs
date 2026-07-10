//! `kirie-cef-helper` — the CEF subprocess entrypoint.
//!
//! CEF's multi-process model relaunches a helper executable for its render,
//! GPU, utility and zygote processes (docs/subsystems-misc.md §3.1). The main
//! `kirie` binary points `CefSettings.browser_subprocess_path` at this helper
//! so those child processes run here instead of re-entering the wallpaper
//! engine's own `main`.
//!
//! The one hard rule (docs §3.1): the handoff to `cef_execute_process` must
//! happen **before any file IO** — CEF passes inherited file descriptors
//! (notably the ICU data fd) to the child, and touching the filesystem first
//! can close them and abort the child. So `main` does nothing but build the
//! args, construct the shared [`kirie_web::cef::app`] `App` (which registers the
//! render-process JS-shim handler), and hand off. The browser process itself
//! never runs this binary, so `execute_process` here is safe (it must never be
//! called in the browser process).

fn main() {
    // Negotiate the CEF API version before building any CEF object (must match
    // the browser process, else libcef rejects the app with "invalid version").
    let _ = cef::api_hash(cef::sys::CEF_API_VERSION_LAST, 0);

    // No logging, no file IO before the handoff (docs §3.1).
    let args = cef::args::Args::new();
    let mut app = kirie_web::cef::app::make_app();

    // Returns the child process exit code, or < 0 if this is not a recognised
    // CEF subprocess invocation (which should not happen for the helper).
    let code = cef::execute_process(Some(args.as_main_args()), Some(&mut app), std::ptr::null_mut());

    std::process::exit(if code >= 0 { code } else { 0 });
}
