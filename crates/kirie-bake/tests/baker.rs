//! Background-baker behavior: staleness skip, pause gating (SPEC.md §V7), and
//! queue draining onto the pool.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use kirie_bake::{BackgroundBaker, BakeOutcome, BakerConfig, BundleContent, Cache, PauseFn};
use kirie_scene::{PropertyBag, Scene, SceneModel};

struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("kirie-baker-{}-{}-{n}", std::process::id(), tag));
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

fn model() -> SceneModel {
    let scene = Scene::from_slice(
        br#"{"camera":{"eye":"0 0 1","center":"0 0 0","up":"0 1 0"},"general":{},"objects":[]}"#,
    )
    .unwrap();
    SceneModel::resolve(scene, &PropertyBag::new())
}

/// source_fn/content_fn that count how many times content is built.
fn config(cache: Cache, pause: PauseFn, built: Arc<AtomicUsize>) -> BakerConfig {
    let source_fn = Arc::new(|item: &Path| Ok(item.as_os_str().as_encoded_bytes().to_vec()));
    let content_fn = Arc::new(move |_item: &Path, _src: &[u8]| {
        built.fetch_add(1, Ordering::Relaxed);
        let mut c = BundleContent::new();
        c.set_scene_model(&model())?;
        Ok(c)
    });
    let mut cfg = BakerConfig::new(cache, source_fn, content_fn);
    cfg.should_pause = pause;
    cfg
}

#[test]
fn bakes_then_skips_when_fresh() {
    let tmp = TmpDir::new("fresh");
    let cache = Cache::with_root(tmp.path());
    let built = Arc::new(AtomicUsize::new(0));
    let baker = BackgroundBaker::start(config(cache, kirie_bake::never_pause(), built.clone()));

    let item = tmp.path().join("item-1");
    let first = baker.bake_item_now(&item).unwrap();
    assert!(matches!(first, BakeOutcome::Baked(_)), "first bakes: {first:?}");

    let second = baker.bake_item_now(&item).unwrap();
    assert_eq!(second, BakeOutcome::Fresh, "second is a fresh hit");
    assert_eq!(built.load(Ordering::Relaxed), 1, "content built exactly once");
}

#[test]
fn paused_baker_skips_work() {
    let tmp = TmpDir::new("pause");
    let cache = Cache::with_root(tmp.path());
    let built = Arc::new(AtomicUsize::new(0));
    let gate = Arc::new(AtomicBool::new(true));
    let g = gate.clone();
    let pause: PauseFn = Arc::new(move || g.load(Ordering::Relaxed));
    let baker = BackgroundBaker::start(config(cache, pause, built.clone()));

    let item = tmp.path().join("item-2");
    // Paused: no work, no bundle.
    assert_eq!(baker.bake_item_now(&item).unwrap(), BakeOutcome::Paused);
    assert_eq!(built.load(Ordering::Relaxed), 0);
    assert!(baker.is_paused());

    // Release the gate → bakes.
    gate.store(false, Ordering::Relaxed);
    assert!(matches!(
        baker.bake_item_now(&item).unwrap(),
        BakeOutcome::Baked(_)
    ));
    assert_eq!(built.load(Ordering::Relaxed), 1);
}

#[test]
fn enqueue_drains_onto_pool() {
    let tmp = TmpDir::new("queue");
    let cache = Cache::with_root(tmp.path());
    let built = Arc::new(AtomicUsize::new(0));
    let baker = BackgroundBaker::start(config(cache.clone(), kirie_bake::never_pause(), built.clone()));

    for i in 0..3 {
        baker.enqueue(tmp.path().join(format!("q-{i}")));
    }

    // Poll for completion (coordinator + pool are async).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while built.load(Ordering::Relaxed) < 3 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(built.load(Ordering::Relaxed), 3, "all queued items baked");
    for i in 0..3 {
        let src = tmp
            .path()
            .join(format!("q-{i}"))
            .into_os_string()
            .into_encoded_bytes();
        assert!(cache.load(&src).unwrap().is_some());
    }
}
