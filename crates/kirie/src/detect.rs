//! Input-file kind detection shared by the `info` and `extract` subcommands.

use std::path::Path;

/// What kind of file an `info`/`extract` input is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    /// `project.json` wallpaper manifest (docs/format-project-json.md §1).
    Project,
    /// `scene.pkg` flat-file archive (docs/format-pkg.md §1).
    Pkg,
    /// `.tex` texture container (docs/format-tex.md §1).
    Tex,
}

/// Classify an input file, by name first and by content as a fallback.
///
/// Name rules: `*.json` → manifest, `*.pkg` → archive, `*.tex` → texture.
/// Content sniffing when the name is inconclusive:
///
/// * a pkg opens with an `sstr` magic — a `u32` length then bytes starting
///   `PKGV`, so bytes 4..8 are always `PKGV` (docs/format-pkg.md §3–§4);
/// * a tex opens with a 9-byte `TEXV...` magic at offset 0
///   (docs/format-tex.md §2–§3);
/// * a manifest is strict JSON whose first non-whitespace byte is `{`
///   (docs/format-project-json.md §1: object root).
pub fn detect(path: &Path, bytes: &[u8]) -> Option<FileKind> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.ends_with(".json") {
        return Some(FileKind::Project);
    }
    if name.ends_with(".pkg") {
        return Some(FileKind::Pkg);
    }
    if name.ends_with(".tex") {
        return Some(FileKind::Tex);
    }
    if bytes.get(4..8) == Some(b"PKGV".as_slice()) {
        return Some(FileKind::Pkg);
    }
    if bytes.get(..4) == Some(b"TEXV".as_slice()) {
        return Some(FileKind::Tex);
    }
    if bytes.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{') {
        return Some(FileKind::Project);
    }
    None
}
