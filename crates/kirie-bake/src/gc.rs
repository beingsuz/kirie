//! Cache garbage collection: evict least-recently-used bundles to stay under a
//! size cap (task §K GC: LRU by atime, default 4 GB cap, configurable).
//!
//! "Access time" is the mtime of each bundle directory's `.atime` marker, which
//! [`crate::Cache::load`] refreshes on every load (mtime is reliable across
//! filesystems, unlike POSIX atime under `relatime`). Directories without a
//! marker fall back to the bundle file's mtime.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::BakeError;

/// Default cache size cap: 4 GiB.
pub const DEFAULT_CAP_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Marker filename whose mtime records last access (kept in sync with [`crate::cache`]).
const ATIME_FILE: &str = ".atime";
const BUNDLE_FILE: &str = "bundle.rkyv";

/// One bundle directory's GC-relevant stats.
#[derive(Debug, Clone)]
struct Entry {
    dir: PathBuf,
    size: u64,
    accessed: SystemTime,
}

/// What a GC pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Total on-disk size of all bundles before eviction.
    pub total_before: u64,
    /// Total on-disk size after eviction.
    pub total_after: u64,
    /// Number of bundle directories evicted.
    pub evicted: usize,
    /// Bytes reclaimed.
    pub reclaimed: u64,
}

/// Evict least-recently-used bundle directories under `bundles_dir` until the
/// total size is at most `cap_bytes`. Missing dir → empty report.
///
/// # Errors
/// [`BakeError::Io`] on an unexpected filesystem failure while scanning; a
/// removal that fails is skipped (its bytes stay counted) rather than aborting.
pub fn gc(bundles_dir: &Path, cap_bytes: u64) -> Result<GcReport, BakeError> {
    let mut entries = match scan(bundles_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(GcReport::default()),
        Err(e) => return Err(BakeError::io(bundles_dir, e)),
    };

    let total_before: u64 = entries.iter().map(|e| e.size).sum();
    let mut total = total_before;
    let mut report = GcReport {
        total_before,
        total_after: total_before,
        ..Default::default()
    };
    if total <= cap_bytes {
        return Ok(report);
    }

    // Oldest access first → evicted first (LRU).
    entries.sort_by_key(|e| e.accessed);
    for e in entries {
        if total <= cap_bytes {
            break;
        }
        match fs::remove_dir_all(&e.dir) {
            Ok(()) => {
                total = total.saturating_sub(e.size);
                report.evicted += 1;
                report.reclaimed += e.size;
            }
            Err(err) => {
                tracing::warn!(dir = %e.dir.display(), error = %err, "gc: eviction failed");
            }
        }
    }
    report.total_after = total;
    Ok(report)
}

/// Scan the bundles tree into per-directory entries.
fn scan(bundles_dir: &Path) -> std::io::Result<Vec<Entry>> {
    let mut out = Vec::new();
    for dent in fs::read_dir(bundles_dir)? {
        let dent = dent?;
        let dir = dent.path();
        if !dir.is_dir() {
            continue;
        }
        let size = dir_size(&dir);
        let accessed = access_time(&dir);
        out.push(Entry { dir, size, accessed });
    }
    Ok(out)
}

/// Sum the byte sizes of all regular files directly under `dir` (bundles are
/// flat: payload + checksum + marker).
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = fs::read_dir(dir) {
        for dent in rd.flatten() {
            if let Ok(md) = dent.metadata()
                && md.is_file()
            {
                total += md.len();
            }
        }
    }
    total
}

/// The access time for LRU ranking: the `.atime` marker mtime if present, else
/// the bundle file mtime, else the Unix epoch (evict first).
fn access_time(dir: &Path) -> SystemTime {
    for name in [ATIME_FILE, BUNDLE_FILE] {
        if let Ok(md) = fs::metadata(dir.join(name))
            && let Ok(t) = md.modified()
        {
            return t;
        }
    }
    SystemTime::UNIX_EPOCH
}
