//! `kirie-bake` — the hash-keyed prebaked scene-bundle cache (SPEC.md §V7/§V8).
//!
//! A *bundle* is an [`rkyv`] archive of everything the renderer needs to start a
//! scene without re-parsing, re-translating shaders, or re-decoding textures:
//! the resolved [`kirie_scene::SceneModel`], translated shader units (SPIR-V +
//! reflection), GPU-ready textures, and precomputed tables (see [`bundle`]).
//! Bundles are content-addressed by a [`BundleKey`] = `blake3(source) ⊕
//! bake-format-version ⊕ shader-translator-version` (SPEC.md §V8) and live at
//! `~/.cache/kirie/bundles/<blake3-hex>/`.
//!
//! ## Load path (warm start)
//!
//! [`Cache::load`] mmaps the bundle and validates it (blake3 checksum + rkyv
//! bytecheck), then hands back a [`LoadedBundle`] whose fields are read
//! zero-copy from the mapping. A key mismatch is a clean miss → rebake; a corrupt
//! bundle is a typed error, never a panic (SPEC.md §V9).
//!
//! ## Bake path (cold / background)
//!
//! [`Cache::bake`] writes a [`BundleContent`] atomically. The
//! [`BackgroundBaker`] watches a workshop directory and bakes new/stale items on
//! an idle-priority pool that pauses under a fullscreen app (SPEC.md §V7), with
//! LRU [`gc`] keeping the cache under a size cap.
//!
//! ## §V2 note
//!
//! The task orders `memmap2` zero-copy loading; mmap requires one `unsafe` call,
//! so this crate cannot `#![forbid(unsafe_code)]`. The two `unsafe` blocks (map +
//! post-validation `access_unchecked`) are documented with `// SAFETY:` in
//! [`cache`]. SPEC.md §V2's exception list should be extended to include
//! kirie-bake; that is a spec-owner amendment, flagged in the task report.

pub mod baker;
pub mod bundle;
pub mod cache;
pub mod error;
pub mod gc;
pub mod key;

pub use baker::{BackgroundBaker, BakeOutcome, BakerConfig, ContentFn, PauseFn, SourceFn, never_pause};

/// Cap glibc's malloc arena count (`mallopt(M_ARENA_MAX, n)`).
///
/// Every wallpaper build runs on a fresh worker thread; glibc hands each new
/// thread a (possibly new) arena, up to 8×cores — and [`trim_heap`] only
/// reliably releases the main arena. Capping the count early keeps transient
/// build allocations in a couple of arenas the trims actually reach, so RSS
/// doesn't ratchet up across wallpaper switches. Call once at startup.
pub fn limit_malloc_arenas(n: i32) {
    // SAFETY: mallopt sets an allocator tuning knob; no pointers involved.
    unsafe {
        libc::mallopt(libc::M_ARENA_MAX, n.max(1));
    }
}

/// Return freed heap pages to the kernel (`malloc_trim(0)`).
///
/// Wallpaper builds allocate large transient buffers (texture decode, shader
/// translation, scene JSON) that glibc's arenas retain after free — tens of
/// MB of dead RSS per build. Callers invoke this once after a build/swap
/// completes; it is a no-op-safe hint, never required for correctness. Lives
/// here for the same §V2 reason as [`map_readonly`] (the one crate allowed
/// `unsafe`; `malloc_trim` is a foreign call).
pub fn trim_heap() {
    // SAFETY: malloc_trim(0) only releases free arena memory back to the OS;
    // it takes no pointers and cannot invalidate live allocations.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Memory-map a file read-only, boxed as opaque bytes.
///
/// This lives here because kirie-bake is the one crate with the §V2 `unsafe`
/// exception for `memmap2` (see the module docs) — `forbid(unsafe_code)`
/// callers (kirie-formats/kirie-render) use it to back large read-only inputs
/// (a multi-hundred-MB `scene.pkg`) with the page cache instead of a heap
/// `Vec`, so the bytes are evictable and never counted as process RSS.
///
/// SAFETY of the map itself: the file is opened read-only and the mapping is
/// private; kirie treats workshop content as immutable while an engine runs —
/// the same assumption the bundle/shader caches already make. An external
/// truncation during use would fault, exactly like the cache mmaps above.
pub fn map_readonly(
    path: &std::path::Path,
) -> std::io::Result<Box<dyn AsRef<[u8]> + Send + Sync>> {
    let f = std::fs::File::open(path)?;
    // SAFETY: read-only mapping of a file kirie never writes while mapped
    // (workshop content is immutable per the crate contract above).
    let map = unsafe { memmap2::Mmap::map(&f) }?;
    Ok(Box::new(map))
}
pub use bundle::{
    BUNDLE_MAGIC, BakedBundle, BakedMip, BakedReflection, BakedShader, BakedStage, BakedTable, BakedTexture,
    BundleContent, BundleHeader,
};
pub use cache::{Cache, LoadedBundle};
pub use error::BakeError;
pub use gc::{DEFAULT_CAP_BYTES, GcReport, gc};
pub use key::{BAKE_FORMAT_VERSION, BundleKey};
