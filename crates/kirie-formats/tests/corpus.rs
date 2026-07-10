//! P1 corpus e2e gate — THE definition of P1-done (SPEC.md §T7, §V11).
//!
//! Every assertion pins an exact literal number taken from the docs/ specs:
//! docs/corpus.md (item inventory, per-pkg composition) and docs/format-tex.md
//! (tex census). One `#[test]` per numbered gate for clear failure
//! attribution:
//!
//! 1. all 24 `project.json` parse; type split matches the inventory
//!    (docs/corpus.md §1, §3)
//! 2. all 19 `scene.pkg` parse; per-item entry counts match; every entry
//!    reads; every `scene.json` entry is valid JSON (docs/corpus.md §2–§4, §9)
//! 3. all 190 embedded `.tex` parse; every non-mp4 top mip decodes to RGBA8
//!    with `len == 4·w·h`; animated frame tables parse
//!    (docs/corpus.md §4; docs/format-tex.md §6.1, §7.3, §8.1)
//! 4. Asset item 3347128360 is identified as non-renderable
//!    (docs/corpus.md §1, §6.3, §8)
//! 5. read-everything smoke pass: every pkg extracts > 0 bytes
//!
//! Corpus-gated: when the corpus directory is absent, each test skips
//! (eprintln + return) so CI without the Steam corpus stays green. Override
//! the corpus location with the `KIRIE_CORPUS` environment variable.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use kirie_formats::pkg::OwnedPkg;
use kirie_formats::project::{Project, WallpaperType};
use kirie_formats::tex::{AnimationVersion, Tex, TexError};

// ---- corpus location (skip-gating) ----------------------------------------

/// Default corpus location (docs/corpus.md: corpus root); override with
/// `KIRIE_CORPUS`.
const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";

/// Resolve the corpus directory, or `None` (with an eprintln) to skip the
/// calling test when the corpus is not installed.
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

// ---- expected inventory (exact literals from docs/corpus.md) ---------------

/// docs/corpus.md §1: 24 items total.
const ITEM_COUNT: usize = 24;
/// docs/corpus.md §1 class counts: 19 scene.pkg, 3 web, 1 video, 1 asset.
const SCENE_COUNT: usize = 19;
/// docs/corpus.md §1.
const WEB_COUNT: usize = 3;
/// docs/corpus.md §1.
const VIDEO_COUNT: usize = 1;
/// docs/corpus.md §1.
const ASSET_COUNT: usize = 1;

/// docs/format-tex.md (header): 190 real `.tex` files embedded in the 19
/// scene.pkg archives.
const TEX_TOTAL: usize = 190;
/// docs/format-tex.md §6.1: corpus flag value 34 (Video|ClampUVs) ×3 — the
/// three mp4-payload textures (§7.3).
const TEX_VIDEO_COUNT: usize = 3;
/// docs/format-tex.md §6.1: corpus flag value 6 (IsGif|ClampUVs) ×1 — exactly
/// one animated texture (§8: TEXS0003, item 3585875739).
const TEX_ANIMATED_COUNT: usize = 1;

/// docs/format-tex.md §7.3: all corpus video payloads start with the 12-byte
/// MP4 signature `00 00 00 20 66 74 79 70 69 73 6F 6D` (`ftyp isom`).
const MP4_FTYP_ISOM: &[u8] = &[
    0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d,
];

/// Per-item expectations from the docs/corpus.md §3 inventory table.
struct ItemExpect {
    /// Workshop id == corpus directory name (docs/corpus.md §3 notes: the
    /// directory name is the authoritative id).
    id: &'static str,
    /// Declared `"type"` string, case-preserved; `None` = key absent
    /// (docs/corpus.md §1, §3 "type (declared)" column).
    declared: Option<&'static str>,
    /// Type resolved by the main-file extension/URL rule
    /// (docs/corpus.md §1 resolution table).
    resolved: WallpaperType,
    /// `project.json:file` (docs/corpus.md §3 "main file" column; for scenes
    /// the value is `scene.json`, which the loader maps to the pkg entry).
    file: &'static str,
    /// Size in bytes of the on-disk file backing `file` (docs/corpus.md §3
    /// "main size" column; for scenes that is `scene.pkg`).
    main_size: u64,
    /// Whether `project.json` carries a `workshopid` key (docs/corpus.md §3
    /// notes list the 7 items lacking it).
    has_workshopid: bool,
    /// `category == "Asset"` non-renderable item (docs/corpus.md §1, §6.3).
    asset: bool,
}

/// The full docs/corpus.md §3 inventory (24 rows, id-ascending).
const ITEMS: &[ItemExpect] = &[
    ItemExpect {
        id: "1388331347",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 4_124_099,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "1627026721",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 1_983_520,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "2082653325",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 34_976_397,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "2085292947",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 35_404_091,
        has_workshopid: false,
        asset: false,
    },
    ItemExpect {
        id: "2155933185",
        declared: Some("Scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 107_698_572,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "2395163768",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 675_812,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "2968833989",
        declared: Some("Web"),
        resolved: WallpaperType::Web,
        file: "index.html",
        main_size: 4349,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3047596375",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 6_923_422,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3118949804",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 9_082_034,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3293156956",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 13_428_067,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        // docs/corpus.md §1 edge case: no `type` key, category=Asset; by the
        // extension rule `effect.json` resolves to Scene, but the item is an
        // effect preset, not a wallpaper (§6.3).
        //
        // DOC ERRATUM: the §3 row prints `main size` 4741, but effect.json is
        // 876 bytes on disk — and the same row's `item size` 111637 sums
        // exactly from the live files (876 + 195 + 7822 + 11375 + 90082 +
        // 1287), so the corpus matches the documented snapshot and the 4741
        // cell is internally inconsistent. Asserting the real byte count;
        // flagged for spec backprop into docs/corpus.md.
        id: "3347128360",
        declared: None,
        resolved: WallpaperType::Scene,
        file: "effects/gradient_generator/effect.json",
        main_size: 876,
        has_workshopid: true,
        asset: true,
    },
    ItemExpect {
        id: "3421423611",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 33_861_818,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3428443753",
        declared: Some("Scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 23_872_566,
        has_workshopid: false,
        asset: false,
    },
    ItemExpect {
        id: "3445942378",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 40_861_619,
        has_workshopid: false,
        asset: false,
    },
    ItemExpect {
        id: "3551997868",
        declared: Some("web"),
        resolved: WallpaperType::Web,
        file: "index.html",
        main_size: 908,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3576956643",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 12_061_757,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3585875739",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 89_384_418,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3587565260",
        declared: Some("Scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 1_480_416,
        has_workshopid: false,
        asset: false,
    },
    ItemExpect {
        id: "3600453929",
        declared: Some("video"),
        resolved: WallpaperType::Video,
        file: "冷冰冰的誓言.mp4",
        main_size: 30_092_496,
        has_workshopid: false,
        asset: false,
    },
    ItemExpect {
        id: "3609007632",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 7_155_575,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3611478368",
        declared: Some("Scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 16_882_275,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3631634316",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 4_213_530,
        has_workshopid: false,
        asset: false,
    },
    ItemExpect {
        id: "3679122549",
        declared: Some("web"),
        resolved: WallpaperType::Web,
        file: "index.html",
        main_size: 33_548,
        has_workshopid: true,
        asset: false,
    },
    ItemExpect {
        id: "3738467344",
        declared: Some("scene"),
        resolved: WallpaperType::Scene,
        file: "scene.json",
        main_size: 6_357_714,
        has_workshopid: false,
        asset: false,
    },
];

/// Per-pkg expectations from the docs/corpus.md §4 composition table
/// ("pkg ver", "entries", "tex" columns) plus the §3 "main size" column
/// (for scenes = the scene.pkg byte size).
struct PkgExpect {
    /// Workshop id / directory name (docs/corpus.md §4).
    id: &'static str,
    /// Full pkg magic incl. version suffix (docs/corpus.md §4 "pkg ver").
    magic: &'static [u8],
    /// Entry-table row count (docs/corpus.md §4 "entries").
    entries: usize,
    /// Number of `.tex` entries (docs/corpus.md §4 "tex").
    tex: usize,
    /// scene.pkg file size in bytes (docs/corpus.md §3 "main size").
    pkg_size: u64,
}

/// The full docs/corpus.md §4 table (19 rows, id-ascending).
const PKGS: &[PkgExpect] = &[
    PkgExpect {
        id: "1388331347",
        magic: b"PKGV0001",
        entries: 44,
        tex: 9,
        pkg_size: 4_124_099,
    },
    PkgExpect {
        id: "1627026721",
        magic: b"PKGV0002",
        entries: 47,
        tex: 8,
        pkg_size: 1_983_520,
    },
    PkgExpect {
        id: "2082653325",
        magic: b"PKGV0006",
        entries: 11,
        tex: 2,
        pkg_size: 34_976_397,
    },
    PkgExpect {
        id: "2085292947",
        magic: b"PKGV0006",
        entries: 11,
        tex: 2,
        pkg_size: 35_404_091,
    },
    PkgExpect {
        id: "2155933185",
        magic: b"PKGV0009",
        entries: 103,
        tex: 22,
        pkg_size: 107_698_572,
    },
    PkgExpect {
        id: "2395163768",
        magic: b"PKGV0012",
        entries: 25,
        tex: 8,
        pkg_size: 675_812,
    },
    PkgExpect {
        id: "3047596375",
        magic: b"PKGV0022",
        entries: 39,
        tex: 1,
        pkg_size: 6_923_422,
    },
    PkgExpect {
        id: "3118949804",
        magic: b"PKGV0019",
        entries: 48,
        tex: 15,
        pkg_size: 9_082_034,
    },
    PkgExpect {
        id: "3293156956",
        magic: b"PKGV0021",
        entries: 54,
        tex: 15,
        pkg_size: 13_428_067,
    },
    PkgExpect {
        id: "3421423611",
        magic: b"PKGV0022",
        entries: 96,
        tex: 18,
        pkg_size: 33_861_818,
    },
    PkgExpect {
        id: "3428443753",
        magic: b"PKGV0022",
        entries: 66,
        tex: 19,
        pkg_size: 23_872_566,
    },
    PkgExpect {
        id: "3445942378",
        magic: b"PKGV0022",
        entries: 25,
        tex: 3,
        pkg_size: 40_861_619,
    },
    PkgExpect {
        id: "3576956643",
        magic: b"PKGV0023",
        entries: 52,
        tex: 11,
        pkg_size: 12_061_757,
    },
    PkgExpect {
        id: "3585875739",
        magic: b"PKGV0023",
        entries: 34,
        tex: 3,
        pkg_size: 89_384_418,
    },
    PkgExpect {
        id: "3587565260",
        magic: b"PKGV0023",
        entries: 5,
        tex: 1,
        pkg_size: 1_480_416,
    },
    PkgExpect {
        id: "3609007632",
        magic: b"PKGV0023",
        entries: 18,
        tex: 3,
        pkg_size: 7_155_575,
    },
    PkgExpect {
        id: "3611478368",
        magic: b"PKGV0024",
        entries: 74,
        tex: 41,
        pkg_size: 16_882_275,
    },
    PkgExpect {
        id: "3631634316",
        magic: b"PKGV0023",
        entries: 37,
        tex: 8,
        pkg_size: 4_213_530,
    },
    PkgExpect {
        id: "3738467344",
        magic: b"PKGV0024",
        entries: 11,
        tex: 1,
        pkg_size: 6_357_714,
    },
];

// ---- shared helpers ---------------------------------------------------------

/// docs/corpus.md §9: pkg-embedded JSON files carry UTF-8 BOMs — strip a
/// leading `EF BB BF` before JSON parsing (`utf-8-sig` semantics).
fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes)
}

/// Load and parse one item's scene.pkg, checking its on-disk size against the
/// docs/corpus.md §3 "main size" column first.
fn load_pkg(dir: &Path, exp: &PkgExpect) -> Result<OwnedPkg> {
    let path = dir.join(exp.id).join("scene.pkg");
    let meta =
        std::fs::metadata(&path).with_context(|| format!("{}: scene.pkg missing from corpus", exp.id))?;
    ensure!(
        meta.len() == exp.pkg_size,
        "{}: scene.pkg is {} bytes, docs/corpus.md §3 says {}",
        exp.id,
        meta.len(),
        exp.pkg_size,
    );
    OwnedPkg::from_path(&path).with_context(|| format!("{}: scene.pkg failed to parse", exp.id))
}

// ---- gate 1: project.json ---------------------------------------------------

/// P1 gate 1: all 24 `project.json` parse and the type split matches the
/// docs/corpus.md §1/§3 inventory exactly (19 scene + 3 web + 1 video +
/// 1 asset; per-item declared string, resolved type, main file, workshopid
/// presence).
#[test]
fn gate1_all_24_project_json_parse_and_type_split_matches_inventory() -> Result<()> {
    let Some(dir) = corpus_dir() else {
        return Ok(());
    };

    ensure!(ITEMS.len() == ITEM_COUNT, "inventory table must hold 24 rows");

    // The corpus is a LIVE Steam directory: the user can subscribe to new
    // wallpapers at any time, so the inventory table is a documented *subset*,
    // not an exact snapshot. Invariant (SPEC §V11): every documented item is
    // still installed, AND every extra installed item also parses — corpus
    // growth must never regress "∀ installed wallpaper works", only the
    // documented rows below carry exact per-item assertions.
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .with_context(|| format!("cannot list corpus dir {}", dir.display()))?
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    on_disk.sort();
    let expected_ids: Vec<&str> = ITEMS.iter().map(|i| i.id).collect();
    let missing: Vec<&&str> = expected_ids
        .iter()
        .filter(|id| !on_disk.iter().any(|d| d == **id))
        .collect();
    ensure!(
        missing.is_empty(),
        "documented corpus items no longer installed: {missing:?} (docs/corpus.md §3 needs a re-scan)",
    );
    // Extra (newly subscribed) items must still parse — the real §V11 gate.
    for id in &on_disk {
        if expected_ids.iter().any(|e| e == id) {
            continue;
        }
        Project::from_path(dir.join(id).join("project.json"))
            .with_context(|| format!("newly installed corpus item {id}: project.json failed to parse"))?;
    }

    let (mut scene, mut web, mut video, mut asset) = (0usize, 0usize, 0usize, 0usize);
    for item in ITEMS {
        let project = Project::from_path(dir.join(item.id).join("project.json"))
            .with_context(|| format!("{}: project.json failed to parse", item.id))?;

        let declared = project.declared_type.as_ref().map(|d| d.0.as_str());
        ensure!(
            declared == item.declared,
            "{}: declared type {declared:?} != inventory {:?} (docs/corpus.md §3)",
            item.id,
            item.declared,
        );
        ensure!(
            project.resolved_type == item.resolved,
            "{}: resolved type {:?} != inventory {:?} (docs/corpus.md §1)",
            item.id,
            project.resolved_type,
            item.resolved,
        );
        ensure!(
            project.file == item.file,
            "{}: main file {:?} != inventory {:?} (docs/corpus.md §3)",
            item.id,
            project.file,
            item.file,
        );
        ensure!(
            project.workshopid.is_some() == item.has_workshopid,
            "{}: workshopid presence {} != inventory {} (docs/corpus.md §3 notes)",
            item.id,
            project.workshopid.is_some(),
            item.has_workshopid,
        );
        ensure!(
            project.is_asset() == item.asset,
            "{}: is_asset() {} != inventory {} (docs/corpus.md §1)",
            item.id,
            project.is_asset(),
            item.asset,
        );

        // For non-scene classes the main file is a loose file on disk whose
        // exact size the §3 inventory records (scene sizes are checked in
        // gate 2 against scene.pkg, which backs `file: "scene.json"`).
        if item.file != "scene.json" {
            let main = dir.join(item.id).join(item.file);
            let meta = std::fs::metadata(&main)
                .with_context(|| format!("{}: main file {} missing", item.id, item.file))?;
            ensure!(
                meta.len() == item.main_size,
                "{}: main file is {} bytes, docs/corpus.md §3 says {}",
                item.id,
                meta.len(),
                item.main_size,
            );
        }

        // Class split (docs/corpus.md §1): the Asset item counts as its own
        // class, not as one of the 19 renderable scenes.
        if item.asset {
            asset += 1;
        } else {
            match item.resolved {
                WallpaperType::Scene => scene += 1,
                WallpaperType::Web => web += 1,
                WallpaperType::Video => video += 1,
                other => anyhow::bail!("{}: unexpected class {other:?}", item.id),
            }
        }
    }

    ensure!(
        (scene, web, video, asset) == (SCENE_COUNT, WEB_COUNT, VIDEO_COUNT, ASSET_COUNT),
        "class split (scene={scene}, web={web}, video={video}, asset={asset}) != \
         docs/corpus.md §1 (19, 3, 1, 1)",
    );
    Ok(())
}

// ---- gate 2: scene.pkg ------------------------------------------------------

/// P1 gate 2: all 19 `scene.pkg` parse, per-item entry counts and pkg
/// versions match the docs/corpus.md §4 table, every entry payload reads
/// fully, and every embedded `scene.json` is valid JSON (BOM-tolerant per
/// docs/corpus.md §9).
#[test]
fn gate2_all_19_scene_pkg_parse_counts_match_and_scene_json_is_valid() -> Result<()> {
    let Some(dir) = corpus_dir() else {
        return Ok(());
    };

    ensure!(PKGS.len() == SCENE_COUNT, "pkg table must hold 19 rows");

    for exp in PKGS {
        let pkg = load_pkg(&dir, exp)?;
        ensure!(
            pkg.magic() == exp.magic,
            "{}: pkg magic {:?} != docs/corpus.md §4 {:?}",
            exp.id,
            String::from_utf8_lossy(pkg.magic()),
            String::from_utf8_lossy(exp.magic),
        );
        ensure!(
            pkg.entry_count() == exp.entries,
            "{}: {} entries != docs/corpus.md §4 count {}",
            exp.id,
            pkg.entry_count(),
            exp.entries,
        );

        // Every entry readable: proves every payload range is in-bounds
        // (docs/format-pkg.md §3 payload addressing).
        for entry in pkg.entries() {
            let payload = pkg.read(&entry).with_context(|| {
                format!(
                    "{}: entry {:?} unreadable",
                    exp.id,
                    String::from_utf8_lossy(entry.name)
                )
            })?;
            ensure!(
                payload.len() == entry.len as usize,
                "{}: entry {:?} read {} bytes, table says {}",
                exp.id,
                String::from_utf8_lossy(entry.name),
                payload.len(),
                entry.len,
            );
        }

        // Every scene.pkg carries a scene.json entry (docs/corpus.md §3: the
        // loader maps `file: "scene.json"` to the pkg entry), and it is
        // valid JSON after BOM stripping (§9).
        let scene = pkg
            .read_name(b"scene.json")
            .with_context(|| format!("{}: no readable scene.json entry", exp.id))?;
        let value: serde_json::Value = serde_json::from_slice(strip_bom(scene))
            .with_context(|| format!("{}: scene.json is not valid JSON", exp.id))?;
        ensure!(
            value.is_object(),
            "{}: scene.json root is not a JSON object",
            exp.id,
        );
    }
    Ok(())
}

// ---- gate 3: .tex -----------------------------------------------------------

/// P1 gate 3: all 190 embedded `.tex` parse (per-item counts per
/// docs/corpus.md §4); every non-mp4 texture's top mip decodes to RGBA8 with
/// `pixels.len() == 4·w·h`; the 3 video textures expose their verbatim mp4
/// payload (docs/format-tex.md §7.3); the single animated texture's frame
/// table parses to the exact §8.1 sample values.
#[test]
fn gate3_all_190_tex_parse_and_non_mp4_top_mips_decode_rgba8() -> Result<()> {
    let Some(dir) = corpus_dir() else {
        return Ok(());
    };

    let (mut total, mut videos, mut animated) = (0usize, 0usize, 0usize);
    for exp in PKGS {
        let pkg = load_pkg(&dir, exp)?;
        let mut item_tex = 0usize;
        for entry in pkg.entries() {
            if !entry.name.ends_with(b".tex") {
                continue;
            }
            item_tex += 1;
            let name = String::from_utf8_lossy(entry.name).into_owned();
            let bytes = pkg
                .read(&entry)
                .with_context(|| format!("{}: {name}: unreadable", exp.id))?;
            let tex =
                Tex::parse(bytes).with_context(|| format!("{}: {name}: .tex failed to parse", exp.id))?;

            if tex.is_video() {
                // docs/format-tex.md §7.3: single image/mip, payload is the
                // whole mp4 file verbatim, starting `ftyp isom`.
                videos += 1;
                let payload = tex
                    .video_payload()
                    .with_context(|| format!("{}: {name}: video payload unreadable", exp.id))?;
                ensure!(
                    payload.starts_with(MP4_FTYP_ISOM),
                    "{}: {name}: video payload lacks the docs/format-tex.md §7.3 \
                     `ftyp isom` signature",
                    exp.id,
                );
                // Pixel decoding of an mp4 payload must refuse with the typed
                // error, not panic (docs/format-tex.md §7.3; SPEC.md §V9).
                ensure!(
                    matches!(tex.decode_rgba8(0, 0), Err(TexError::IsVideo)),
                    "{}: {name}: decode_rgba8 on a video must return TexError::IsVideo",
                    exp.id,
                );
            } else {
                // Top mip (image 0, mip 0) decodes to tightly packed RGBA8
                // (docs/format-tex.md §5, §7.1/§7.2).
                let img = tex
                    .decode_rgba8(0, 0)
                    .with_context(|| format!("{}: {name}: top mip failed to decode", exp.id))?;
                ensure!(
                    img.width > 0 && img.height > 0,
                    "{}: {name}: decoded to degenerate {}x{}",
                    exp.id,
                    img.width,
                    img.height,
                );
                let expected_len = 4 * img.width as usize * img.height as usize;
                ensure!(
                    img.pixels.len() == expected_len,
                    "{}: {name}: decoded {} bytes for {}x{}, want 4*w*h = {expected_len}",
                    exp.id,
                    img.pixels.len(),
                    img.width,
                    img.height,
                );
            }

            if let Some(anim) = &tex.animation {
                animated += 1;
                // docs/format-tex.md §8.1 real TEXS0003 sample: workshop item
                // 3585875739, 39 frames, every frametime = 1/39 s (total
                // exactly 1.0 s), every frameNumber = 0, gifWidth =
                // gifHeight = 201.
                ensure!(
                    exp.id == "3585875739",
                    "{}: {name}: unexpected animated texture (docs/format-tex.md §8.1 \
                     documents exactly one, in 3585875739)",
                    exp.id,
                );
                ensure!(
                    anim.version == AnimationVersion::Texs0003,
                    "{}: {name}: animation version {:?} != TEXS0003",
                    exp.id,
                    anim.version,
                );
                ensure!(
                    anim.frames.len() == 39,
                    "{}: {name}: {} frames != 39 (docs/format-tex.md §8.1)",
                    exp.id,
                    anim.frames.len(),
                );
                ensure!(
                    anim.gif_width == 201 && anim.gif_height == 201,
                    "{}: {name}: gif dims {}x{} != 201x201 (docs/format-tex.md §8.1)",
                    exp.id,
                    anim.gif_width,
                    anim.gif_height,
                );
                for (i, frame) in anim.frames.iter().enumerate() {
                    ensure!(
                        frame.frame_number == 0,
                        "{}: {name}: frame {i} frameNumber {} != 0 (docs/format-tex.md §8.1)",
                        exp.id,
                        frame.frame_number,
                    );
                    ensure!(
                        frame.frametime == 1.0_f32 / 39.0_f32,
                        "{}: {name}: frame {i} frametime {} != 1/39 s (docs/format-tex.md §8.1)",
                        exp.id,
                        frame.frametime,
                    );
                }
                let sum: f32 = anim.frames.iter().map(|f| f.frametime).sum();
                ensure!(
                    sum == 1.0,
                    "{}: {name}: total animation time {sum} != exactly 1.0 s \
                     (docs/format-tex.md §8.1)",
                    exp.id,
                );
            }
        }
        ensure!(
            item_tex == exp.tex,
            "{}: {item_tex} .tex entries != docs/corpus.md §4 count {}",
            exp.id,
            exp.tex,
        );
        total += item_tex;
    }

    ensure!(
        total == TEX_TOTAL,
        "{total} .tex files across the corpus != docs/format-tex.md census 190",
    );
    ensure!(
        videos == TEX_VIDEO_COUNT,
        "{videos} video .tex != docs/format-tex.md §6.1 count 3",
    );
    ensure!(
        animated == TEX_ANIMATED_COUNT,
        "{animated} animated .tex != docs/format-tex.md §6.1 count 1",
    );
    Ok(())
}

// ---- gate 4: the Asset item ---------------------------------------------------

/// P1 gate 4: the Asset item 3347128360 parses without panic and is
/// identified as non-renderable exactly as docs/corpus.md §1/§6.3/§8
/// prescribe: no `type` key, `category == "Asset"`, extension-resolved Scene,
/// rendering not attempted; its loose `effect.json` parses as JSON
/// (editor-only `gizmos` tolerated).
#[test]
fn gate4_asset_item_3347128360_identified_non_renderable() -> Result<()> {
    let Some(dir) = corpus_dir() else {
        return Ok(());
    };
    let item = dir.join("3347128360");

    let project =
        Project::from_path(item.join("project.json")).context("3347128360: project.json failed to parse")?;

    // docs/corpus.md §1: item 3347128360 has no `type` key...
    ensure!(
        project.declared_type.is_none(),
        "3347128360: expected no declared type, got {:?}",
        project.declared_type,
    );
    // ... and `"category": "Asset"`, which is what marks it non-renderable
    // (docs/corpus.md §8: "classified as non-renderable via category ==
    // Asset"; docs/format-project-json.md §3.3).
    ensure!(
        project.category() == Some("Asset"),
        "3347128360: category {:?} != Some(\"Asset\")",
        project.category(),
    );
    ensure!(project.is_asset(), "3347128360: is_asset() must be true");
    // docs/corpus.md §1: by the extension rule its effect.json main file
    // resolves to Scene — the Asset category, not the resolved type, is the
    // non-renderable signal.
    ensure!(
        project.resolved_type == WallpaperType::Scene,
        "3347128360: resolved type {:?} != Scene (docs/corpus.md §1)",
        project.resolved_type,
    );
    ensure!(
        project.file == "effects/gradient_generator/effect.json",
        "3347128360: main file {:?} != inventory (docs/corpus.md §3)",
        project.file,
    );
    // docs/format-project-json.md §2.1: WE's playlist preflight rejects
    // manifests lacking a `type` key, so this item never reaches rendering.
    ensure!(
        !project.passes_preflight(),
        "3347128360: an item without a `type` key must fail preflight",
    );

    // docs/corpus.md §6.3/§8: the effect manifest itself parses, tolerating
    // editor-only keys such as the `gizmos` array. Size asserted against the
    // real bytes (876): the §3 "main size" cell prints 4741, but the same
    // row's item size 111637 only sums from the live files with effect.json
    // = 876 — doc erratum, see the ITEMS entry for this id.
    let effect_path = item.join(&project.file);
    let bytes = std::fs::read(&effect_path)
        .with_context(|| format!("3347128360: cannot read {}", effect_path.display()))?;
    ensure!(
        bytes.len() == 876,
        "3347128360: effect.json is {} bytes, expected 876 (docs/corpus.md §3 \
         erratum: prints 4741, contradicted by its own item-size column)",
        bytes.len(),
    );
    let value: serde_json::Value =
        serde_json::from_slice(strip_bom(&bytes)).context("3347128360: effect.json is not valid JSON")?;
    ensure!(
        value.get("gizmos").is_some_and(serde_json::Value::is_array),
        "3347128360: effect.json lacks the `gizmos` array docs/corpus.md §6.3 documents",
    );
    Ok(())
}

// ---- gate 5: read-everything smoke pass ---------------------------------------

/// P1 gate 5: smoke read-everything pass — extracting every entry of every
/// scene.pkg yields a strictly positive byte total per pkg (SPEC.md §T7).
#[test]
fn gate5_every_pkg_extracts_more_than_zero_bytes() -> Result<()> {
    let Some(dir) = corpus_dir() else {
        return Ok(());
    };

    for exp in PKGS {
        let pkg = load_pkg(&dir, exp)?;
        let mut extracted: u64 = 0;
        for entry in pkg.entries() {
            let payload = pkg.read(&entry).with_context(|| {
                format!(
                    "{}: entry {:?} unreadable",
                    exp.id,
                    String::from_utf8_lossy(entry.name)
                )
            })?;
            extracted += payload.len() as u64;
        }
        ensure!(
            extracted > 0,
            "{}: extracted 0 bytes from scene.pkg (smoke pass failed)",
            exp.id,
        );
    }
    Ok(())
}
