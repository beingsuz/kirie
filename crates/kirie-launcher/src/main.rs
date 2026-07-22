//! kirie self-extracting launcher stub.
//!
//! Prepended to a zstd-compressed tar of the CEF runtime (see [`kirie_launcher`]
//! for the layout). On launch it finds the runtime in the cache — extracting it
//! from its own trailing blob on first run — then `exec`s the real web-cef
//! `kirie` engine with the original arguments. `exec` replaces this process, so
//! the pid, argv (the daemon's `--screen-root`/`--control-socket` matchers) and
//! the supervisor all see the real engine.
//!
//! Non-web wallpapers run on the same web-cef engine (CEF is only initialized
//! when a web wallpaper loads), so a single file serves every wallpaper type and
//! keeps live web `bg` swaps working (the running process has CEF available).

use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use kirie_launcher::{KEY_LEN, MAGIC, TRAILER_LEN};

fn main() -> std::process::ExitCode {
    match run() {
        // `exec` only returns on failure; a success value is unreachable.
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("kirie: {err}");
            std::process::ExitCode::from(127)
        }
    }
}

fn run() -> io::Result<()> {
    let exe = std::env::current_exe()?;
    let dir = ensure_extracted(&exe)?;
    let target = dir.join("kirie");
    // Replace this process with the real engine, forwarding every argument.
    // `exec` returns only if it failed (e.g. the extracted binary is missing).
    let err = Command::new(&target).args(std::env::args_os().skip(1)).exec();
    Err(io::Error::other(format!(
        "cannot exec extracted engine {}: {err}",
        target.display()
    )))
}

/// Ensure the CEF runtime is extracted under the cache and return its directory.
/// Fast path: the trailer's cache key already has a `.complete` marker → return
/// it without touching the (large) blob. Slow path (first run / new build):
/// extract the appended `zstd(tar(...))` into a temp dir and atomically rename
/// it into place.
fn ensure_extracted(exe: &Path) -> io::Result<PathBuf> {
    let mut f = File::open(exe)?;
    let size = f.metadata()?.len();
    if size < TRAILER_LEN as u64 {
        return Err(io::Error::other(
            "executable too small to be a kirie self-extracting binary",
        ));
    }

    let mut trailer = [0u8; TRAILER_LEN];
    f.seek(SeekFrom::End(-(TRAILER_LEN as i64)))?;
    f.read_exact(&mut trailer)?;
    if &trailer[..8] != MAGIC {
        return Err(io::Error::other(
            "not a kirie self-extracting binary (bad trailer magic)",
        ));
    }
    let blob_len = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
    let key = std::str::from_utf8(&trailer[16..16 + KEY_LEN])
        .map_err(|_| io::Error::other("bad trailer cache key"))?;

    let root = cache_root()?;
    let dir = root.join(key);
    if dir.join(".complete").is_file() {
        return Ok(dir);
    }

    // Extract into a per-pid temp dir, then rename into place so a partial
    // extraction is never mistaken for a complete one (and concurrent launches
    // don't corrupt each other).
    fs::create_dir_all(&root)?;
    let tmp = root.join(format!(".tmp.{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp)?;

    let blob_off = size - TRAILER_LEN as u64 - blob_len;
    f.seek(SeekFrom::Start(blob_off))?;
    let blob = BufReader::new(f.take(blob_len));
    let decoder = zstd::Decoder::new(blob)?;
    let mut archive = tar::Archive::new(decoder);
    archive.set_preserve_permissions(true);
    archive.unpack(&tmp)?;
    File::create(tmp.join(".complete"))?;

    match fs::rename(&tmp, &dir) {
        Ok(()) => {
            // A distinct build extracts a new ~1.5 GB runtime; prune old ones so
            // they don't accumulate one-per-build forever.
            prune_old_runtimes(&root, key);
            Ok(dir)
        }
        // Lost a race with a concurrent launch that already populated `dir`.
        Err(_) if dir.join(".complete").is_file() => {
            let _ = fs::remove_dir_all(&tmp);
            Ok(dir)
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp);
            Err(e)
        }
    }
}

/// Keep the current runtime + the single most-recently-modified other under
/// `root`, removing older `<key>` dirs (each ~1.5 GB). Best-effort: skips the
/// current key and any in-progress `.tmp.*` dir; a runtime a running engine
/// still uses keeps working through its open inodes even once unlinked.
fn prune_old_runtimes(root: &Path, keep: &str) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut dirs: Vec<(PathBuf, std::time::SystemTime)> = entries
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| {
            let n = e.file_name();
            let n = n.to_string_lossy();
            !n.starts_with('.') && n != keep
        })
        .filter_map(|e| {
            let m = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((e.path(), m))
        })
        .collect();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.1)); // newest first
    for (path, _) in dirs.into_iter().skip(1) {
        let _ = fs::remove_dir_all(path);
    }
}

/// Cache root for extracted runtimes: `$XDG_CACHE_HOME/kirie/rt` (or
/// `$HOME/.cache/kirie/rt`).
fn cache_root() -> io::Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or_else(|| io::Error::other("neither XDG_CACHE_HOME nor HOME is set"))?;
    Ok(base.join("kirie").join("rt"))
}
