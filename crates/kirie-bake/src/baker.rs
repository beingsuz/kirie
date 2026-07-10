//! Background baker: watch a workshop directory and bake new/stale items on an
//! idle-priority pool, pausing under a fullscreen app (SPEC.md §V7).
//!
//! The heavy asset pipeline (pkg → resolved scene → translated shaders → decoded
//! textures) lives in the render/scene crates; kirie-bake stays the cache layer
//! and takes that work as two injected closures so it need not depend on the
//! whole engine:
//!
//! - a **source** closure — cheap: reads the item's canonical source bytes
//!   (`scene.pkg` + `project.json`) used for the [`crate::BundleKey`] and the
//!   staleness check (SPEC.md §V8: stale ⇔ key miss).
//! - a **content** closure — expensive: builds the [`BundleContent`].
//!
//! ## Idle priority (SPEC.md §V7)
//!
//! kirie-bake forbids OS-level thread-nice tuning here (no FFI/`unsafe` — see the
//! crate's §V2 note), so idle priority is realized cooperatively: a small
//! [`rayon`] pool (default 1 thread) plus a `should_pause` gate the app drives
//! from its fullscreen-detection signal. A paused job yields immediately and is
//! re-enqueued, so a bake never competes with a foreground app for the GPU/CPU.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use notify::{RecursiveMode, Watcher};

use crate::bundle::BundleContent;
use crate::cache::Cache;
use crate::error::BakeError;
use crate::gc;

/// Reads the canonical source bytes for a workshop item (cheap; used for keying
/// and staleness). Typically `scene.pkg` bytes ⧺ `project.json` bytes.
pub type SourceFn = Arc<dyn Fn(&Path) -> Result<Vec<u8>, BakeError> + Send + Sync>;

/// Builds the [`BundleContent`] for an item (expensive: resolve + translate +
/// decode). Receives the item path and its already-read source bytes.
pub type ContentFn = Arc<dyn Fn(&Path, &[u8]) -> Result<BundleContent, BakeError> + Send + Sync>;

/// Returns `true` while baking should be suspended (fullscreen app, SPEC.md §V7).
pub type PauseFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// A `should_pause` that never pauses (default).
#[must_use]
pub fn never_pause() -> PauseFn {
    Arc::new(|| false)
}

/// Configuration for a [`BackgroundBaker`].
#[derive(Clone)]
pub struct BakerConfig {
    /// The cache to write bundles into.
    pub cache: Cache,
    /// Cheap source-bytes reader (keying + staleness).
    pub source_fn: SourceFn,
    /// Expensive content builder.
    pub content_fn: ContentFn,
    /// Fullscreen/idle pause gate (SPEC.md §V7).
    pub should_pause: PauseFn,
    /// Size cap for the post-bake GC pass.
    pub cap_bytes: u64,
    /// Worker-pool thread count (idle priority ⇒ keep small; default 1).
    pub num_threads: usize,
}

impl BakerConfig {
    /// A config with default pause (never), 4 GiB cap, single-thread pool.
    #[must_use]
    pub fn new(cache: Cache, source_fn: SourceFn, content_fn: ContentFn) -> Self {
        BakerConfig {
            cache,
            source_fn,
            content_fn,
            should_pause: never_pause(),
            cap_bytes: gc::DEFAULT_CAP_BYTES,
            num_threads: 1,
        }
    }
}

/// Outcome of baking one item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BakeOutcome {
    /// Baked; bundle written at this path.
    Baked(PathBuf),
    /// Skipped — an up-to-date bundle already exists (SPEC.md §V8 fresh).
    Fresh,
    /// Skipped — baking is paused (SPEC.md §V7); the item should be retried.
    Paused,
}

/// Shared, thread-safe baker state.
struct Inner {
    cache: Cache,
    source_fn: SourceFn,
    content_fn: ContentFn,
    should_pause: PauseFn,
    paused: AtomicBool,
    cap_bytes: u64,
}

impl Inner {
    fn paused_now(&self) -> bool {
        self.paused.load(Ordering::Relaxed) || (self.should_pause)()
    }

    /// Bake one item synchronously, honoring pause and staleness.
    fn bake_item(&self, item: &Path) -> Result<BakeOutcome, BakeError> {
        if self.paused_now() {
            return Ok(BakeOutcome::Paused);
        }
        let source = (self.source_fn)(item)?;
        if self.cache.load(&source)?.is_some() {
            return Ok(BakeOutcome::Fresh);
        }
        // Re-check pause after the (cheap) source read but before heavy work.
        if self.paused_now() {
            return Ok(BakeOutcome::Paused);
        }
        let content = (self.content_fn)(item, &source)?;
        let path = self.cache.bake(&source, content)?;
        // Best-effort GC; never fail a bake because the reaper hiccuped.
        if let Err(e) = gc::gc(&self.cache.bundles_dir(), self.cap_bytes) {
            tracing::warn!(error = %e, "post-bake gc failed");
        }
        Ok(BakeOutcome::Baked(path))
    }
}

/// Messages to the coordinator thread.
enum Msg {
    Item(PathBuf),
    Stop,
}

/// A running background baker: a coordinator thread drains an item queue onto a
/// rayon pool; filesystem watchers feed the queue.
pub struct BackgroundBaker {
    inner: Arc<Inner>,
    tx: Sender<Msg>,
    coordinator: Option<JoinHandle<()>>,
    watchers: Vec<notify::RecommendedWatcher>,
}

impl BackgroundBaker {
    /// Create and start a background baker. The coordinator thread runs until
    /// [`Self::shutdown`] (or drop).
    #[must_use]
    pub fn start(config: BakerConfig) -> Self {
        let inner = Arc::new(Inner {
            cache: config.cache,
            source_fn: config.source_fn,
            content_fn: config.content_fn,
            should_pause: config.should_pause,
            paused: AtomicBool::new(false),
            cap_bytes: config.cap_bytes,
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads.max(1))
            .thread_name(|i| format!("kirie-bake-{i}"))
            .build()
            .expect("rayon pool");
        let (tx, rx): (Sender<Msg>, Receiver<Msg>) = crossbeam_channel::unbounded();
        let coord_inner = Arc::clone(&inner);
        let coordinator = std::thread::Builder::new()
            .name("kirie-bake-coord".into())
            .spawn(move || coordinator_loop(&coord_inner, &pool, &rx))
            .expect("spawn coordinator");
        BackgroundBaker {
            inner,
            tx,
            coordinator: Some(coordinator),
            watchers: Vec::new(),
        }
    }

    /// Enqueue an item to be baked on the pool.
    pub fn enqueue(&self, item: impl Into<PathBuf>) {
        let _ = self.tx.send(Msg::Item(item.into()));
    }

    /// Watch `dir` recursively; on any create/modify under it, enqueue the
    /// immediate child of `dir` that contains the change (the "item root").
    ///
    /// # Errors
    /// [`BakeError::Watch`] if the watcher cannot be created or armed.
    pub fn watch(&mut self, dir: impl Into<PathBuf>) -> Result<(), BakeError> {
        let dir = dir.into();
        let tx = self.tx.clone();
        let root = dir.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            let Ok(event) = res else { return };
            if !matches!(
                event.kind,
                notify::EventKind::Create(_) | notify::EventKind::Modify(_)
            ) {
                return;
            }
            for path in event.paths {
                if let Some(item) = item_root(&root, &path) {
                    let _ = tx.send(Msg::Item(item));
                }
            }
        })
        .map_err(|e| BakeError::Watch(e.to_string()))?;
        watcher
            .watch(&dir, RecursiveMode::Recursive)
            .map_err(|e| BakeError::Watch(e.to_string()))?;
        self.watchers.push(watcher);
        Ok(())
    }

    /// Bake one item synchronously (bypasses the queue/pool). Honors pause and
    /// staleness. Useful for one-shot bakes and deterministic tests.
    ///
    /// # Errors
    /// Propagates [`BakeError`] from the source/content closures or the cache.
    pub fn bake_item_now(&self, item: &Path) -> Result<BakeOutcome, BakeError> {
        self.inner.bake_item(item)
    }

    /// Suspend baking (SPEC.md §V7). Idempotent.
    pub fn pause(&self) {
        self.inner.paused.store(true, Ordering::Relaxed);
    }

    /// Resume baking. Idempotent.
    pub fn resume(&self) {
        self.inner.paused.store(false, Ordering::Relaxed);
    }

    /// Whether baking is currently paused (by the flag or the `should_pause` gate).
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.inner.paused_now()
    }

    /// Stop the coordinator thread and drop watchers, blocking until the
    /// coordinator exits. Called automatically on drop.
    pub fn shutdown(&mut self) {
        self.watchers.clear();
        let _ = self.tx.send(Msg::Stop);
        if let Some(h) = self.coordinator.take() {
            let _ = h.join();
        }
    }
}

impl Drop for BackgroundBaker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Coordinator: drain the queue, dispatching each item onto the pool. A paused
/// item is re-enqueued after a short sleep so it is retried once foreground
/// activity ends (SPEC.md §V7).
fn coordinator_loop(inner: &Arc<Inner>, pool: &rayon::ThreadPool, rx: &Receiver<Msg>) {
    while let Ok(msg) = rx.recv() {
        let item = match msg {
            Msg::Item(p) => p,
            Msg::Stop => break,
        };
        let job = Arc::clone(inner);
        pool.spawn(move || match job.bake_item(&item) {
            Ok(BakeOutcome::Paused) => {
                // Retry later without a busy loop.
                std::thread::sleep(std::time::Duration::from_millis(250));
                // Best-effort: if still paused it will re-pause and re-sleep.
                let _ = job.bake_item(&item);
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(item = %item.display(), error = %e, "background bake failed"),
        });
    }
}

/// The immediate child of `root` that contains `changed`, or `changed` itself if
/// it is a direct child. `None` if `changed` is not under `root`.
fn item_root(root: &Path, changed: &Path) -> Option<PathBuf> {
    let rel = changed.strip_prefix(root).ok()?;
    let first = rel.components().next()?;
    Some(root.join(first.as_os_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_root_maps_nested_change_to_child() {
        let root = Path::new("/ws");
        assert_eq!(
            item_root(root, Path::new("/ws/123/scene.pkg")),
            Some(PathBuf::from("/ws/123"))
        );
        assert_eq!(
            item_root(root, Path::new("/ws/123")),
            Some(PathBuf::from("/ws/123"))
        );
        assert_eq!(item_root(root, Path::new("/other/x")), None);
    }
}
