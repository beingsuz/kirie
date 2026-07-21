//! Build script: bake an `$ORIGIN` rpath into the `kirie-cef-helper` subprocess
//! binary for `cef` builds, and surface the `webview` feature's limitation at
//! compile time.
//!
//! The helper is a separate executable CEF launches as its browser/render
//! subprocess (`browser_subprocess_path`), and it links `libcef.so` as a
//! load-time (`DT_NEEDED`) dependency just like the main binary. It sits beside
//! `libcef.so` in the install, so an `$ORIGIN` runpath lets it find the library
//! without `LD_LIBRARY_PATH` — mirroring the main `kirie` binary's rpath so the
//! whole CEF runtime is self-contained in one directory.
//!
//! Gated on the `cef` feature so non-CEF builds of this crate are unaffected.

fn main() {
    if std::env::var_os("CARGO_FEATURE_CEF").is_some() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    if std::env::var_os("CARGO_FEATURE_WEBVIEW").is_some() {
        // Compile-time-visible notice (src/webview/mod.rs has the evidence):
        // wry/webkit2gtk has no off-screen rendering path, so this feature can
        // only ever be a native-surface fallback — won't-fix upstream.
        println!(
            "cargo:warning=kirie-web `webview` feature: wry/webkit2gtk cannot render \
             off-screen (upstream limitation, won't-fix); this backend paints a native \
             surface only — use `--features cef` (kirie: `web-cef`) for composited web \
             wallpapers"
        );
    }
}
