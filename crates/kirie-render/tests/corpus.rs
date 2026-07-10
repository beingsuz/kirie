//! Corpus-gated content tests (skipped when the workshop corpus is
//! absent): the single animated `.tex` in the corpus — workshop item
//! 3585875739, a 39-frame TEXS0003 atlas (docs/format-tex.md §8.1 real
//! sample, §10.3) — must decode into the documented placements and
//! schedule.

use std::path::{Path, PathBuf};

use kirie_formats::pkg::OwnedPkg;
use kirie_formats::tex::Tex;
use kirie_render::ImageContent;

/// Default corpus location (docs/corpus.md); override with `KIRIE_CORPUS`.
const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
/// The corpus item holding the animated texture (docs/format-tex.md §10.3).
const ANIMATED_ITEM: &str = "3585875739";

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("KIRIE_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CORPUS_DIR));
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "skipping corpus test: {} not found (set KIRIE_CORPUS to override)",
            dir.display()
        );
        None
    }
}

/// Find the animated (`flags & IsGif`) texture payload in the item's
/// scene.pkg and hand it to `visit`.
fn with_animated_tex(dir: &Path, visit: impl FnOnce(&str, &[u8])) {
    let pkg_path = dir.join(ANIMATED_ITEM).join("scene.pkg");
    let pkg = OwnedPkg::from_path(&pkg_path).expect("scene.pkg parses");
    for entry in pkg.entries() {
        let Some(name) = entry.name_str() else { continue };
        if !name.ends_with(".tex") {
            continue;
        }
        let payload = pkg.read(&entry).expect("entry reads");
        let tex = Tex::parse(payload).expect("tex parses");
        if tex.flags.is_gif() {
            visit(name, payload);
            return;
        }
    }
    panic!("no animated .tex found in {}", pkg_path.display());
}

#[test]
fn corpus_animated_tex_decodes_to_documented_frame_schedule() {
    let Some(dir) = corpus_dir() else { return };

    with_animated_tex(&dir, |name, payload| {
        let content = ImageContent::from_tex_bytes(payload).unwrap_or_else(|e| panic!("{name}: {e}"));

        // docs/format-tex.md §8.1 sample: 39 frames on a single
        // 1608x1005 PNG-backed page, logical frame 201x201.
        assert_eq!(content.pages.len(), 1, "{name}: imageCount");
        let page = &content.pages[0];
        assert_eq!((page.width, page.height), (1608, 1005), "{name}: atlas dims");
        assert_eq!(page.pixels.len(), 1608 * 1005 * 4, "{name}: decoded RGBA size");
        assert_eq!(content.frames.len(), 39, "{name}: frame count");
        assert_eq!(content.content_size(), (201, 201), "{name}: gifWidth/gifHeight");

        // Frames laid out left-to-right, top-to-bottom in an 8x5 grid of
        // 201x201 cells; every frameNumber = 0, no rotated frames
        // (docs/format-tex.md §8.1, §10.3).
        let (w, h) = (1608.0f32, 1005.0f32);
        for (i, frame) in content.frames.iter().enumerate() {
            assert_eq!(frame.page, 0, "{name}: frame {i} page");
            let col = (i % 8) as f32;
            let row = (i / 8) as f32;
            assert_eq!(
                frame.translation,
                [col * 201.0 / w, row * 201.0 / h],
                "{name}: frame {i} origin"
            );
            assert_eq!(
                frame.axes,
                [201.0 / w, 0.0, 0.0, 201.0 / h],
                "{name}: frame {i} axes"
            );
            // Every frametime = 1/39 s (docs/format-tex.md §8.1).
            assert!(
                (frame.duration - 1.0 / 39.0).abs() < 1e-6,
                "{name}: frame {i} duration {}",
                frame.duration
            );
        }

        // Schedule semantics over the real table: total exactly 1 s, the
        // §8.1 walk hits every slot at its midpoint and wraps.
        let schedule = content.schedule();
        assert!(schedule.is_animated(), "{name}");
        assert!(
            (schedule.total_seconds() - 1.0).abs() < 1e-4,
            "{name}: total {}",
            schedule.total_seconds()
        );
        let slot = schedule.total_seconds() / 39.0;
        for k in 0..39usize {
            let midpoint = (k as f64 + 0.5) * slot;
            assert_eq!(schedule.frame_at(midpoint), k, "{name}: slot {k}");
            // One full loop later, same frame (fmod wrap, §8.1 step 2).
            assert_eq!(
                schedule.frame_at(midpoint + schedule.total_seconds()),
                k,
                "{name}: wrapped slot {k}"
            );
        }
        // The frame after slot k's boundary is k+1.
        assert_eq!(schedule.frame_at(1.5 * slot), 1, "{name}");
        assert_eq!(schedule.frame_at(38.5 * slot), 38, "{name}");
    });
}

#[test]
fn corpus_preview_jpg_loads_as_static_content() {
    let Some(dir) = corpus_dir() else { return };
    let path = dir.join(ANIMATED_ITEM).join("preview.jpg");
    let content = ImageContent::from_path(&path).expect("preview.jpg decodes");
    assert_eq!(content.pages.len(), 1);
    assert_eq!(content.frames.len(), 1);
    assert!(!content.schedule().is_animated());
    assert_eq!(content.frames[0].translation, [0.0, 0.0]);
    assert_eq!(content.frames[0].axes, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        content.content_size(),
        (content.pages[0].width, content.pages[0].height)
    );
    assert!(content.pages[0].width > 0 && content.pages[0].height > 0);
}
