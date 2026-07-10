//! The on-disk bundle cache: `~/.cache/kirie/bundles/<blake3-hex>/`.
//!
//! - [`Cache::bake`] writes a [`BundleContent`] as an rkyv archive under the
//!   [`BundleKey`] directory, atomically (temp + rename), with a blake3 sidecar.
//! - [`Cache::load`] mmaps the bundle for the given source and returns it, or
//!   `None` on a key miss (SPEC.md §V8: mismatch → rebake, no migration). A
//!   *corrupt* on-disk bundle is detected (checksum + rkyv bytecheck) and, for
//!   the self-healing `load` path, treated as a miss after removal; the lower
//!   level [`LoadedBundle::open`] surfaces the typed error instead (SPEC.md §V9).
//! - Pipeline/GPU caches live in a sibling `pipelines/<adapter>/` tree, keyed
//!   additionally by adapter id.

use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::bundle::{ArchivedBakedBundle, BUNDLE_MAGIC, BakedBundle, BundleContent};
use crate::error::BakeError;
use crate::key::BundleKey;

/// Bundle payload filename inside a key directory.
const BUNDLE_FILE: &str = "bundle.rkyv";
/// blake3 checksum sidecar filename.
const CHECKSUM_FILE: &str = "bundle.b3";
/// Access-time marker; its mtime is the LRU key (updated on every load).
const ATIME_FILE: &str = ".atime";

/// A handle to the bundle cache rooted at some base directory (default
/// `~/.cache/kirie`). Holds no global state — construct one and pass it around
/// (SPEC.md §V1).
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Open the cache at the default base directory: `$XDG_CACHE_HOME/kirie`, or
    /// `$HOME/.cache/kirie`. Does not touch the filesystem until a bake/load.
    ///
    /// # Errors
    /// Returns [`BakeError::Io`] if neither `XDG_CACHE_HOME` nor `HOME` is set.
    pub fn open_default() -> Result<Self, BakeError> {
        let base = default_cache_base()?;
        Ok(Self::with_root(base))
    }

    /// Open the cache rooted at an explicit base directory (used in tests).
    #[must_use]
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Cache { root: root.into() }
    }

    /// The base directory (`.../kirie`).
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The `bundles/` subtree.
    #[must_use]
    pub fn bundles_dir(&self) -> PathBuf {
        self.root.join("bundles")
    }

    /// The key directory for `source` (whether or not it exists).
    #[must_use]
    pub fn bundle_dir(&self, source: &[u8]) -> PathBuf {
        self.bundles_dir().join(BundleKey::compute(source).to_hex())
    }

    /// The GPU pipeline-cache directory for an adapter and source, keyed
    /// additionally by adapter id so pipeline blobs never cross GPUs. The
    /// renderer owns the contents; kirie-bake only vends the path.
    #[must_use]
    pub fn pipeline_cache_dir(&self, adapter_id: &str, source: &[u8]) -> PathBuf {
        self.root
            .join("pipelines")
            .join(sanitize(adapter_id))
            .join(BundleKey::compute(source).to_hex())
    }

    /// Bake `content` for `source` into the cache, returning the bundle file
    /// path. Overwrites any existing bundle at the key atomically.
    ///
    /// # Errors
    /// [`BakeError::Serialize`] if rkyv encoding fails; [`BakeError::Io`] on any
    /// filesystem failure.
    pub fn bake(&self, source: &[u8], content: BundleContent) -> Result<PathBuf, BakeError> {
        let dir = self.bundle_dir(source);
        fs::create_dir_all(&dir).map_err(|e| BakeError::io(&dir, e))?;

        let bundle: BakedBundle = content.into_bundle(source);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
            .map_err(|e| BakeError::Serialize(e.to_string()))?;
        let checksum = blake3::hash(&bytes);

        let file = dir.join(BUNDLE_FILE);
        write_atomic(&file, &bytes)?;
        write_atomic(&dir.join(CHECKSUM_FILE), checksum.to_hex().as_bytes())?;
        touch(&dir.join(ATIME_FILE))?;
        Ok(file)
    }

    /// Load the bundle for `source`, or `None` on a key miss (SPEC.md §V8). A
    /// corrupt bundle is removed and reported as a miss so the caller rebakes
    /// (self-healing); the load never panics (SPEC.md §V9).
    ///
    /// # Errors
    /// [`BakeError::Io`] only for unexpected filesystem failures (a missing
    /// directory is a clean `Ok(None)`).
    pub fn load(&self, source: &[u8]) -> Result<Option<LoadedBundle>, BakeError> {
        let dir = self.bundle_dir(source);
        let file = dir.join(BUNDLE_FILE);
        if !file.exists() {
            return Ok(None);
        }
        match LoadedBundle::open(&file) {
            Ok(b) => {
                // Refresh the LRU access marker; ignore marker write failures.
                let _ = touch(&dir.join(ATIME_FILE));
                Ok(Some(b))
            }
            Err(BakeError::Corrupt { .. } | BakeError::ChecksumMismatch { .. }) => {
                // Self-heal: drop the bad directory so the next bake refills it.
                let _ = fs::remove_dir_all(&dir);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Remove the cached bundle for `source`, if present.
    ///
    /// # Errors
    /// [`BakeError::Io`] if removal fails for a reason other than absence.
    pub fn remove(&self, source: &[u8]) -> Result<(), BakeError> {
        let dir = self.bundle_dir(source);
        match fs::remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(BakeError::io(&dir, e)),
        }
    }
}

/// A memory-mapped, validated bundle. Zero-copy: field access reads directly out
/// of the mapped file (SPEC.md §V8 warm load). Validated once at [`Self::open`].
pub struct LoadedBundle {
    mmap: Mmap,
    path: PathBuf,
}

impl std::fmt::Debug for LoadedBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedBundle")
            .field("path", &self.path)
            .field("size_bytes", &self.mmap.len())
            .finish()
    }
}

impl LoadedBundle {
    /// Open and validate a bundle file: mmap it, verify the blake3 checksum
    /// sidecar, and run rkyv's structural bytecheck. Any failure is a typed
    /// error, never a panic (SPEC.md §V9).
    ///
    /// # Errors
    /// [`BakeError::Io`], [`BakeError::ChecksumMismatch`], or [`BakeError::Corrupt`].
    pub fn open(file: &Path) -> Result<Self, BakeError> {
        let f = fs::File::open(file).map_err(|e| BakeError::io(file, e))?;
        // SAFETY: the bundle file is opened read-only and this process never
        // holds a writable handle to it. Bundles are only ever *replaced* via
        // temp-file + `rename` (a fresh inode) and *removed* via unlink
        // (`remove_dir_all` on GC or `load` self-heal) — never written in place.
        // A live mapping pins its backing inode: unlinking the directory entry or
        // renaming a new inode over the path leaves the mapped bytes intact and
        // immutable for the lifetime of the returned `LoadedBundle` (POSIX
        // unlink-after-mmap semantics), even if GC reaps the directory or another
        // thread rebakes the same key concurrently. Task orders mandate memmap2
        // zero-copy load; this is the sole unavoidable `unsafe` for it (§V2 note).
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| BakeError::io(file, e))?;

        // Checksum gate: detect truncation/tamper before trusting the bytes.
        let sidecar = file.with_file_name(CHECKSUM_FILE);
        if let Ok(expected_hex) = fs::read_to_string(&sidecar) {
            let expected = expected_hex.trim();
            let actual = blake3::hash(&mmap).to_hex();
            if !expected.is_empty() && expected != actual.as_str() {
                return Err(BakeError::ChecksumMismatch {
                    path: file.to_path_buf(),
                    expected: expected.to_string(),
                    actual: actual.to_string(),
                });
            }
        }

        // Structural validation (rkyv bytecheck). Corrupt archive → typed error.
        let archived = rkyv::access::<ArchivedBakedBundle, rkyv::rancor::Error>(&mmap).map_err(|e| {
            BakeError::Corrupt {
                path: file.to_path_buf(),
                reason: e.to_string(),
            }
        })?;
        if archived.header.magic.to_native() != BUNDLE_MAGIC {
            return Err(BakeError::Corrupt {
                path: file.to_path_buf(),
                reason: format!(
                    "bad magic 0x{:08x} (expected 0x{BUNDLE_MAGIC:08x})",
                    archived.header.magic.to_native()
                ),
            });
        }

        Ok(LoadedBundle {
            mmap,
            path: file.to_path_buf(),
        })
    }

    /// The validated archive root, borrowed zero-copy from the mmap.
    #[must_use]
    pub fn archived(&self) -> &ArchivedBakedBundle {
        // SAFETY: the same bytes were validated with the checked `access` in
        // `open` and the mmap is immutable for `self`'s lifetime, so the archive
        // is well-formed. Avoids re-validating O(n) on every accessor.
        unsafe { rkyv::access_unchecked::<ArchivedBakedBundle>(&self.mmap) }
    }

    /// The embedded resolved [`kirie_scene::SceneModel`], deserialized from its
    /// JSON payload (SPEC.md §V13 round-trip).
    ///
    /// # Errors
    /// [`BakeError::Decode`] if the JSON is not a valid model.
    pub fn scene_model(&self) -> Result<kirie_scene::SceneModel, BakeError> {
        serde_json::from_slice(self.scene_json_bytes()).map_err(|e| BakeError::Decode {
            field: "scene_json",
            reason: e.to_string(),
        })
    }

    /// Zero-copy view of the embedded scene-model JSON bytes.
    #[must_use]
    pub fn scene_json_bytes(&self) -> &[u8] {
        &self.archived().scene_json
    }

    /// Number of baked shaders.
    #[must_use]
    pub fn shader_count(&self) -> usize {
        self.archived().shaders.len()
    }

    /// Number of baked textures.
    #[must_use]
    pub fn texture_count(&self) -> usize {
        self.archived().textures.len()
    }

    /// Zero-copy payload bytes of texture `i`, if present.
    #[must_use]
    pub fn texture_data(&self, i: usize) -> Option<&[u8]> {
        self.archived().textures.get(i).map(|t| t.data.as_slice())
    }

    /// The mapped size in bytes (bundle-on-disk size).
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.mmap.len()
    }

    /// The file this bundle was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Fully deserialize the bundle into owned Rust values (copies out of the
    /// mmap). Prefer the zero-copy accessors on the hot path.
    ///
    /// # Errors
    /// [`BakeError::Corrupt`] if deserialization fails.
    pub fn deserialize(&self) -> Result<BakedBundle, BakeError> {
        rkyv::deserialize::<BakedBundle, rkyv::rancor::Error>(self.archived()).map_err(|e| {
            BakeError::Corrupt {
                path: self.path.clone(),
                reason: e.to_string(),
            }
        })
    }
}

/// Resolve the default cache base directory.
fn default_cache_base() -> Result<PathBuf, BakeError> {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME")
        && !x.is_empty()
    {
        return Ok(PathBuf::from(x).join("kirie"));
    }
    if let Some(home) = std::env::var_os("HOME")
        && !home.is_empty()
    {
        return Ok(PathBuf::from(home).join(".cache").join("kirie"));
    }
    Err(BakeError::io(
        PathBuf::from("~/.cache/kirie"),
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "neither XDG_CACHE_HOME nor HOME is set",
        ),
    ))
}

/// Sanitize an adapter id into a single path component.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Write `bytes` to `path` atomically: write a sibling temp file, then rename.
///
/// The temp filename is unique per writer — process id **and** thread id — so two
/// threads in *this* process baking the same source (e.g. the background baker and
/// an on-demand foreground bake, or duplicate watcher events fanned onto a
/// multi-thread pool) never share a temp path. A shared temp path let concurrent
/// writers interleave `write`s and race `rename`s, producing a torn bundle that
/// then failed bytecheck on load (self-healed away → cache miss). Distinct sources
/// already key to distinct directories; this closes the *same*-source case.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), BakeError> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    // Numeric-ish thread tag (e.g. "ThreadId(3)" → "ThreadId3"); filename-safe and
    // distinct across concurrent writers without any process-global counter (§V1).
    let tid: String = format!("{:?}", std::thread::current().id())
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    let tmp = dir.join(format!(
        ".{}.tmp.{}.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("bundle"),
        std::process::id(),
        tid,
    ));
    fs::write(&tmp, bytes).map_err(|e| BakeError::io(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        BakeError::io(path, e)
    })
}

/// Create/truncate a marker file so its mtime is "now" (LRU access stamp).
fn touch(path: &Path) -> Result<(), BakeError> {
    fs::write(path, []).map_err(|e| BakeError::io(path, e))
}
