//! `kirie extract <scene.pkg|.tex> [-o DIR]` — write a pkg's entries to
//! disk preserving entry paths, or decode a `.tex` to PNG(s) (SPEC.md §I).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use image::RgbaImage;
use kirie_formats::pkg::Pkg;
use kirie_formats::tex::{Frame, Tex};

use crate::detect::{self, FileKind};

/// Run the `extract` subcommand: `path` is a `scene.pkg` or `.tex` file,
/// `out_dir` the destination directory, `tex_to_png` additionally converts
/// every `.tex` entry of a pkg to PNG(s).
pub fn run(path: &Path, out_dir: &Path, tex_to_png: bool) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    match detect::detect(path, &bytes) {
        Some(FileKind::Pkg) => extract_pkg(path, &bytes, out_dir, tex_to_png),
        Some(FileKind::Tex) => extract_tex_file(path, &bytes, out_dir),
        Some(FileKind::Project) => bail!(
            "{} looks like a project.json manifest; extract takes a scene.pkg or .tex",
            path.display()
        ),
        None => bail!(
            "cannot determine the type of {} (expected a scene.pkg or .tex)",
            path.display()
        ),
    }
}

/// Write every archive entry below `out_dir`, preserving entry paths
/// (docs/format-pkg.md §5: names are `/`-separated relative paths).
fn extract_pkg(path: &Path, bytes: &[u8], out_dir: &Path, tex_to_png: bool) -> Result<()> {
    let pkg = Pkg::parse(bytes).with_context(|| format!("parsing {}", path.display()))?;
    std::fs::create_dir_all(out_dir).with_context(|| format!("cannot create {}", out_dir.display()))?;
    let mut written = 0usize;
    for entry in pkg.entries() {
        // Entry names are opaque bytes, conventionally UTF-8
        // (docs/format-pkg.md §2); writing to disk requires valid UTF-8.
        let name = entry.name_str().ok_or_else(|| {
            anyhow!(
                "entry name {:?} is not valid UTF-8",
                String::from_utf8_lossy(entry.name)
            )
        })?;
        let rel = sanitize_entry_path(name)?;
        let payload = pkg.read(entry).with_context(|| format!("reading entry {name}"))?;
        let dest = out_dir.join(&rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
        }
        std::fs::write(&dest, payload).with_context(|| format!("cannot write {}", dest.display()))?;
        println!("{}", dest.display());
        written += 1;

        if tex_to_png && name.to_ascii_lowercase().ends_with(".tex") {
            let tex = Tex::parse(payload).with_context(|| format!("parsing texture {name}"))?;
            if tex.is_video() {
                // Video textures hold a verbatim MP4 stream, not decodable
                // pixels (docs/format-tex.md §7.3) — skip, keep going.
                eprintln!("skipping {name}: video texture (docs/format-tex.md §7.3)");
                continue;
            }
            // foo/bar.tex → foo/bar[.frameNNN|.imageN].png
            let stem = rel.with_extension("");
            for png in
                write_tex_pngs(&tex, out_dir, &stem).with_context(|| format!("decoding texture {name}"))?
            {
                println!("{}", png.display());
            }
        }
    }
    println!("extracted {written} entries to {}", out_dir.display());
    Ok(())
}

/// Decode a standalone `.tex` file to PNG(s) in `out_dir`.
fn extract_tex_file(path: &Path, bytes: &[u8], out_dir: &Path) -> Result<()> {
    let tex = Tex::parse(bytes).with_context(|| format!("parsing {}", path.display()))?;
    if tex.is_video() {
        // docs/format-tex.md §7.3: the payload is a whole MP4 file, not
        // decodable pixels.
        bail!(
            "{} is a video texture (docs/format-tex.md §7.3): it stores an MP4 \
             stream, not decodable pixels",
            path.display()
        );
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("cannot create {}", out_dir.display()))?;
    let stem = path
        .file_stem()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("texture"));
    let pngs =
        write_tex_pngs(&tex, out_dir, &stem).with_context(|| format!("decoding {}", path.display()))?;
    for png in &pngs {
        println!("{}", png.display());
    }
    println!("wrote {} PNG(s) to {}", pngs.len(), out_dir.display());
    Ok(())
}

/// Turn an archive entry name into a safe relative path.
///
/// Entry names are `/`-separated relative paths without a leading slash
/// (docs/format-pkg.md §5). Anything that could escape `out_dir` — absolute
/// paths, `.`/`..` components, empty components, backslashes, NULs — is
/// rejected with an error rather than remapped.
fn sanitize_entry_path(name: &str) -> Result<PathBuf> {
    ensure!(!name.is_empty(), "empty entry name");
    ensure!(!name.starts_with('/'), "absolute entry path {name:?}");
    let mut out = PathBuf::new();
    for component in name.split('/') {
        ensure!(!component.is_empty(), "empty path component in entry {name:?}");
        ensure!(
            component != "." && component != "..",
            "path traversal in entry {name:?}"
        );
        ensure!(
            !component.contains('\\') && !component.contains('\0'),
            "unsafe character in entry {name:?}"
        );
        out.push(component);
    }
    Ok(out)
}

/// Decode `tex` to PNG files under `out_dir`, named after `rel_stem`
/// (a sanitized relative path without the `.tex` extension).
///
/// Non-animated textures produce mip 0 of each image (docs/format-tex.md §7:
/// mip levels are largest first); `<stem>.png` for the single-image case,
/// `<stem>.imageN.png` otherwise (`imageCount > 1` is UNVERIFIED, §9).
/// Animated textures (`flags & IsGif`, §8) instead produce one
/// `<stem>.frameNNN.png` per animation frame, cropped from mip 0 of the
/// image selected by the frame's `frameNumber` (§8, §8.1).
fn write_tex_pngs(tex: &Tex<'_>, out_dir: &Path, rel_stem: &Path) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    if let Some(anim) = &tex.animation {
        // All frames of a page share its decode; cache per image index.
        let mut atlases: HashMap<u32, RgbaImage> = HashMap::new();
        for (index, frame) in anim.frames.iter().enumerate() {
            let atlas = match atlases.entry(frame.frame_number) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    // §8: frameNumber indexes the imageCount images.
                    let image_index = usize::try_from(frame.frame_number)
                        .with_context(|| format!("frame {index} frameNumber overflow"))?;
                    entry.insert(decode_mip0(tex, image_index)?)
                }
            };
            let cropped = crop_frame(atlas, frame, index)?;
            written.push(save_png(
                cropped,
                out_dir,
                rel_stem,
                &format!(".frame{index:03}"),
            )?);
        }
    } else {
        for index in 0..tex.images.len() {
            let image = decode_mip0(tex, index)?;
            let suffix = if tex.images.len() == 1 {
                String::new()
            } else {
                format!(".image{index}")
            };
            written.push(save_png(image, out_dir, rel_stem, &suffix)?);
        }
    }
    Ok(written)
}

/// Decode mip 0 (the largest level, docs/format-tex.md §7) of one image to
/// an RGBA8 buffer.
fn decode_mip0(tex: &Tex<'_>, image_index: usize) -> Result<RgbaImage> {
    let decoded = tex.decode_rgba8(image_index, 0)?;
    RgbaImage::from_raw(decoded.width, decoded.height, decoded.pixels)
        .ok_or_else(|| anyhow!("image {image_index}: decoded pixel buffer size mismatch"))
}

/// Crop one animation frame out of its atlas page.
///
/// The frame rect is origin `(x, y)` with extents `width1` × `height1` in
/// atlas texels (docs/format-tex.md §8). Nonzero `width2`/`height2` encode
/// frames stored rotated in the atlas; their exact sampling is UNVERIFIED
/// (§8.1), so they are rejected with an error rather than guessed at.
fn crop_frame(atlas: &RgbaImage, frame: &Frame, index: usize) -> Result<RgbaImage> {
    ensure!(
        frame.width2 == 0.0 && frame.height2 == 0.0,
        "frame {index} is stored rotated in the atlas (width2/height2 != 0), \
         which is UNVERIFIED (docs/format-tex.md §8.1)"
    );
    let x = texel(frame.x, "x", index)?;
    let y = texel(frame.y, "y", index)?;
    let width = texel(frame.width1, "width1", index)?;
    let height = texel(frame.height1, "height1", index)?;
    ensure!(
        width > 0 && height > 0,
        "frame {index} has an empty rect {width}x{height}"
    );
    let end_x = x
        .checked_add(width)
        .ok_or_else(|| anyhow!("frame {index} rect overflows"))?;
    let end_y = y
        .checked_add(height)
        .ok_or_else(|| anyhow!("frame {index} rect overflows"))?;
    ensure!(
        end_x <= atlas.width() && end_y <= atlas.height(),
        "frame {index} rect {width}x{height}@{x},{y} exceeds its {}x{} atlas page",
        atlas.width(),
        atlas.height()
    );
    Ok(image::imageops::crop_imm(atlas, x, y, width, height).to_image())
}

/// Convert a frame-rect field (f32 texels, docs/format-tex.md §8) to a texel
/// count, rejecting non-finite or negative values.
fn texel(value: f32, what: &str, index: usize) -> Result<u32> {
    ensure!(
        value.is_finite() && value >= 0.0 && value <= u32::MAX as f32,
        "frame {index} field {what} = {value} is not a valid texel coordinate"
    );
    // In-range by the check above; float→int `as` casts also saturate,
    // so this cannot wrap or panic.
    Ok(value.round() as u32)
}

/// Save `image` as `<out_dir>/<rel_stem><suffix>.png` and return the path.
fn save_png(image: RgbaImage, out_dir: &Path, rel_stem: &Path, suffix: &str) -> Result<PathBuf> {
    let mut name = rel_stem
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "texture".to_owned());
    name.push_str(suffix);
    name.push_str(".png");
    let dest = out_dir
        .join(rel_stem.parent().unwrap_or_else(|| Path::new("")))
        .join(name);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    image
        .save_with_format(&dest, image::ImageFormat::Png)
        .with_context(|| format!("cannot write {}", dest.display()))?;
    Ok(dest)
}
