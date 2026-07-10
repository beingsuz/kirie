//! Synthetic (no-corpus, no-GPU) tests for the bake cache: round-trip,
//! key invalidation, LRU GC eviction, and corrupt-bundle rejection
//! (SPEC.md §V8/§V9, task §K test list).

use std::path::{Path, PathBuf};

use kirie_bake::{BakeError, BundleContent, BundleKey, Cache, LoadedBundle};
use kirie_scene::{PropertyBag, Scene, SceneModel};

/// A unique scratch directory removed on drop (no tempfile dep).
struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("kirie-bake-{}-{}-{n}", std::process::id(), tag));
        std::fs::create_dir_all(&p).unwrap();
        TmpDir(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A minimal but real resolved scene model.
fn synthetic_model() -> SceneModel {
    let json = br#"{
        "camera": { "eye": "0 0 1", "center": "0 0 0", "up": "0 1 0" },
        "general": {},
        "objects": []
    }"#;
    let scene = Scene::from_slice(json).expect("minimal scene parses");
    SceneModel::resolve(scene, &PropertyBag::new())
}

fn content_for(model: &SceneModel) -> BundleContent {
    let mut c = BundleContent::new();
    c.set_scene_model(model).unwrap();
    c.add_rgba8_texture("bg", 2, 2, vec![255u8; 2 * 2 * 4]);
    c.add_table("lut", vec![1, 2, 3, 4]);
    c
}

#[test]
fn round_trip_bake_then_load() {
    let tmp = TmpDir::new("rt");
    let cache = Cache::with_root(tmp.path());
    let model = synthetic_model();
    let source = b"synthetic-source-A";

    let path = cache.bake(source, content_for(&model)).unwrap();
    assert!(path.exists(), "bundle file written");

    let loaded = cache.load(source).unwrap().expect("bundle present");
    // Scene model round-trips (SPEC.md §V13).
    assert_eq!(loaded.scene_model().unwrap(), model);
    // Texture + table survived the archive.
    assert_eq!(loaded.texture_count(), 1);
    assert_eq!(loaded.texture_data(0).unwrap(), &vec![255u8; 16][..]);
    // Header records the current versions (SPEC.md §V8).
    let full = loaded.deserialize().unwrap();
    assert_eq!(full.header.format_version, kirie_bake::BAKE_FORMAT_VERSION);
    assert_eq!(full.header.translator_version, kirie_shader::TRANSLATOR_VERSION);
    assert_eq!(full.tables[0].data, vec![1, 2, 3, 4]);
}

#[test]
fn key_miss_on_source_change_triggers_rebake() {
    let tmp = TmpDir::new("keymiss");
    let cache = Cache::with_root(tmp.path());
    let model = synthetic_model();

    cache.bake(b"source-A", content_for(&model)).unwrap();

    // Same source → hit; different source → miss (SPEC.md §V8: rebake).
    assert!(cache.load(b"source-A").unwrap().is_some());
    assert!(cache.load(b"source-B").unwrap().is_none());

    // Distinct sources key to distinct directories.
    assert_ne!(cache.bundle_dir(b"source-A"), cache.bundle_dir(b"source-B"));
    assert_ne!(BundleKey::compute(b"source-A"), BundleKey::compute(b"source-B"));
}

#[test]
fn lru_gc_evicts_oldest_accessed_first() {
    let tmp = TmpDir::new("gc");
    let cache = Cache::with_root(tmp.path());
    let model = synthetic_model();

    // Three bundles; each ~big-ish via a table payload.
    let big = |n: u8| {
        let mut c = BundleContent::new();
        c.set_scene_model(&model).unwrap();
        c.add_table("payload", vec![n; 300_000]);
        c
    };
    cache.bake(b"item-old", big(1)).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    cache.bake(b"item-mid", big(2)).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    cache.bake(b"item-new", big(3)).unwrap();

    // Touch the two newer ones so "old" is the LRU victim.
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(cache.load(b"item-mid").unwrap().is_some());
    assert!(cache.load(b"item-new").unwrap().is_some());

    // Cap that forces evicting ~one bundle (each ≥300KB; cap 500KB).
    let report = kirie_bake::gc(&cache.bundles_dir(), 500_000).unwrap();
    assert!(report.evicted >= 1, "at least one eviction: {report:?}");
    assert!(report.total_after <= 500_000, "under cap: {report:?}");
    // The least-recently-accessed bundle is the one gone.
    assert!(cache.load(b"item-old").unwrap().is_none(), "oldest evicted first");
    assert!(cache.load(b"item-new").unwrap().is_some(), "newest kept");
}

#[test]
fn corrupt_bundle_is_rejected_not_panicked() {
    let tmp = TmpDir::new("corrupt");
    let cache = Cache::with_root(tmp.path());
    let model = synthetic_model();
    let source = b"corruptible";

    let path = cache.bake(source, content_for(&model)).unwrap();

    // Overwrite the payload with garbage and update the checksum sidecar so the
    // checksum gate passes and rkyv's structural bytecheck is what rejects it.
    let garbage = vec![0xABu8; 4096];
    std::fs::write(&path, &garbage).unwrap();
    let sidecar = path.with_file_name("bundle.b3");
    std::fs::write(&sidecar, blake3::hash(&garbage).to_hex().as_bytes()).unwrap();

    // Low-level open surfaces the typed corruption error (no panic, SPEC.md §V9).
    let err = LoadedBundle::open(&path).unwrap_err();
    assert!(
        matches!(err, BakeError::Corrupt { .. }),
        "expected Corrupt, got {err:?}"
    );

    // Self-healing load treats it as a miss and removes the bad dir.
    assert!(cache.load(source).unwrap().is_none());
    assert!(!path.exists(), "corrupt bundle dir removed");
}

#[test]
fn checksum_mismatch_is_rejected() {
    let tmp = TmpDir::new("checksum");
    let cache = Cache::with_root(tmp.path());
    let model = synthetic_model();
    let source = b"tamperable";
    let path = cache.bake(source, content_for(&model)).unwrap();

    // Truncate the payload but leave the original checksum sidecar in place.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&path, &bytes).unwrap();

    let err = LoadedBundle::open(&path).unwrap_err();
    assert!(
        matches!(
            err,
            BakeError::ChecksumMismatch { .. } | BakeError::Corrupt { .. }
        ),
        "expected checksum/corrupt rejection, got {err:?}"
    );
}
