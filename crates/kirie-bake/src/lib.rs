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
pub use bundle::{
    BUNDLE_MAGIC, BakedBundle, BakedMip, BakedReflection, BakedShader, BakedStage, BakedTable, BakedTexture,
    BundleContent, BundleHeader,
};
pub use cache::{Cache, LoadedBundle};
pub use error::BakeError;
pub use gc::{DEFAULT_CAP_BYTES, GcReport, gc};
pub use key::{BAKE_FORMAT_VERSION, BundleKey};
