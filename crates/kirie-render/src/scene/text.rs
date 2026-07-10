//! Real glyph shaping + rasterization for TEXT-layer objects
//! (docs/render-architecture.md §7.4 `CText`; docs/format-scene-json.md §13).
//!
//! A text object is shaped once with [`cosmic_text`] (font family/size/alignment
//! taken from the [`TextObject`](kirie_scene::object::TextObject)) using system
//! fonts discovered through the bundled `fontdb`, then every glyph is rasterized
//! into a single tightly-packed coverage bitmap (RGB = 0xFF, alpha = glyph
//! coverage). The bitmap is uploaded once as a [`GpuTexture`] and drawn as one
//! textured quad at the object's scene-space transform; the fragment shader
//! multiplies the coverage by the object's `color`/`alpha` and translucent-blends
//! into the scene FBO (blend factors from docs §7.4:
//! `GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA`).
//!
//! **V5** (no steady-state render alloc): shaping + rasterization happen at build
//! time (here), not per frame — the per-frame path only re-draws the cached quad.
//! **V9** (no panic on malformed input): an empty string, a font that resolves to
//! nothing, or a zero-coverage layout all return [`None`]/an empty raster and the
//! caller simply skips the object; every bitmap write is bounds-clamped.
//!
//! Font resolution: the WE `font` field is a *file path* or `systemfont_<name>`
//! token (docs §13). A file path is first tried as a **scene-bundled face** —
//! its bytes are read from the scene [`AssetSource`] and loaded into `fontdb`
//! ([`TextFonts::bundled_family`]), so a wallpaper shipping a custom display font
//! (e.g. `Anurati-Regular.otf`) shapes in that exact face. Only when the field is
//! not a loadable font file do we fall back to a [`Family`] name hint (the file
//! stem or `systemfont_` suffix) and let `fontdb` substitute the nearest
//! installed face.

use std::collections::HashMap;
use std::sync::Arc;

use cosmic_text::fontdb;
use cosmic_text::{
    Align, Attrs, Buffer, Color as CtColor, Family, FontSystem, Metrics, Shaping, SwashCache, Wrap,
};
use kirie_scene::resolve::AssetSource;

use super::texture::GpuTexture;

/// Line-height as a multiple of the font point size (cosmic-text default ratio;
/// WE's `spacing` override is unparsed, docs §13).
const LINE_HEIGHT_RATIO: f32 = 1.2;

/// Upper bound on a rasterized text bitmap edge, in pixels, so a pathological
/// `pointsize`/`size` cannot request a multi-gigabyte allocation (V9).
const MAX_EDGE: u32 = 4096;

/// The shared font stack for a scene's text objects: a [`FontSystem`] (system
/// fonts, scanned once) and a [`SwashCache`] (rasterized-glyph cache). Built the
/// first time a drawable text object is encountered, reused for the rest.
pub struct TextFonts {
    font_system: FontSystem,
    swash: SwashCache,
    /// `font` path → real family name of the scene-bundled face loaded from it
    /// (`None` = tried and not a loadable bundled font). Dedupes reloads and
    /// records the family name to shape with (docs §13 bundled-font path).
    bundled: HashMap<String, Option<String>>,
}

impl TextFonts {
    /// Scan the system font database once (`fontdb`) and build an empty glyph
    /// cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            swash: SwashCache::new(),
            bundled: HashMap::new(),
        }
    }

    /// Number of faces the font database discovered (diagnostics / test gating).
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.font_system.db().len()
    }

    /// Resolve a text object's `font` field to a scene-bundled face, loading it
    /// into the font DB on first use (docs §13). A WE text layer's `font` is the
    /// exact container path of the packaged face (e.g.
    /// `fonts/workshop/2981960200/Anurati-Regular.otf`); we read those bytes from
    /// the scene [`AssetSource`] and register them so shaping uses the real
    /// wallpaper font instead of a system substitute.
    ///
    /// Returns the loaded face's family name to shape with, or `None` when the
    /// field is not a font-file path or the bytes are absent/unparsable — the
    /// caller then falls back to the [`family_hint`] system substitution.
    pub fn bundled_family(&mut self, font: &str, source: &dyn AssetSource) -> Option<String> {
        let font = font.trim();
        let lower = font.to_ascii_lowercase();
        if !(lower.ends_with(".ttf") || lower.ends_with(".otf") || lower.ends_with(".ttc")) {
            return None;
        }
        if let Some(cached) = self.bundled.get(font) {
            return cached.clone();
        }
        let family = source.load(font).and_then(|bytes| self.load_face(bytes));
        if family.is_none() {
            tracing::debug!(font, "bundled font not found/loadable; using system fallback");
        }
        self.bundled.insert(font.to_owned(), family.clone());
        family
    }

    /// Load raw font bytes into the DB and return the first face's family name.
    fn load_face(&mut self, bytes: Vec<u8>) -> Option<String> {
        let db = self.font_system.db_mut();
        let ids = db.load_font_source(fontdb::Source::Binary(Arc::new(bytes)));
        let id = *ids.first()?;
        db.face(id)?.families.first().map(|(name, _)| name.clone())
    }
}

impl Default for TextFonts {
    fn default() -> Self {
        Self::new()
    }
}

/// A shaped + rasterized text block: an RGBA8 coverage bitmap (RGB 0xFF, alpha =
/// per-pixel glyph coverage) plus its pixel size and the laid-out line count.
pub struct TextRaster {
    /// Tightly-packed RGBA8, `width * height * 4` bytes.
    pub pixels: Vec<u8>,
    /// Bitmap width in pixels (= the box width when a box is set, else measured).
    pub width: u32,
    /// Bitmap height in pixels.
    pub height: u32,
    /// Number of laid-out visual lines (explicit newlines + soft wraps).
    pub line_count: usize,
    /// Whether any pixel received non-zero glyph coverage (false → nothing to
    /// draw, e.g. whitespace-only text or no font faces available).
    pub any_coverage: bool,
}

/// Horizontal alignment from the WE `horizontalalign` string (docs §13:
/// `left|center|right`, default center).
fn h_align(s: &str) -> Align {
    match s {
        "left" => Align::Left,
        "right" => Align::Right,
        _ => Align::Center,
    }
}

/// Vertical-alignment factor from the WE `verticalalign` string (docs §13:
/// `top|center|bottom`, default center): 0 = top, 0.5 = center, 1 = bottom.
fn v_align_factor(s: &str) -> f32 {
    match s {
        "top" => 0.0,
        "bottom" => 1.0,
        _ => 0.5,
    }
}

/// Map the WE `font` field to a `fontdb` [`Family`] name hint (docs §13):
/// `systemfont_arial` → `arial`; `fonts/VCR_OSD_MONO.ttf` → `VCR_OSD_MONO`;
/// empty → [`None`] (sans-serif default). The returned owned string backs the
/// borrowed [`Family::Name`] in [`rasterize`].
fn family_hint(font: &str) -> Option<String> {
    let font = font.trim();
    if font.is_empty() {
        return None;
    }
    if let Some(name) = font.strip_prefix("systemfont_") {
        let name = name.trim();
        return (!name.is_empty()).then(|| name.to_owned());
    }
    // A path like `fonts/Name.ttf` or a workshop path — take the file stem
    // (filename without directory or extension).
    let file = font.rsplit(['/', '\\']).next().unwrap_or(font);
    let stem = file.rsplit_once('.').map_or(file, |(base, _ext)| base);
    let stem = stem.trim();
    (!stem.is_empty()).then(|| stem.to_owned())
}

/// Shape `text` and rasterize its glyphs into a coverage bitmap (docs §7.4).
/// Returns [`None`] when `text` is empty after trimming trailing control chars
/// (V9: nothing to render). `box_size` is the WE `size` bounding box in scene
/// pixels — a positive width enables word wrap + horizontal alignment across the
/// box; a positive height enables vertical alignment. `padding` insets the glyphs
/// on every edge.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn rasterize(
    fonts: &mut TextFonts,
    text: &str,
    font: &str,
    point_size: f32,
    box_size: [f32; 2],
    horizontalalign: &str,
    verticalalign: &str,
    padding: f32,
    bundled_family: Option<&str>,
) -> Option<TextRaster> {
    if text.is_empty() {
        return None;
    }
    let point_size = point_size.max(1.0);
    let pad = padding.max(0.0);
    let metrics = Metrics::new(point_size, point_size * LINE_HEIGHT_RATIO);

    let has_box_w = box_size[0] > 1.0;
    let has_box_h = box_size[1] > 1.0;
    let inner_w = has_box_w.then(|| (box_size[0] - 2.0 * pad).max(1.0));

    // A scene-bundled face (loaded by name) wins over the system-substitution
    // hint derived from the raw `font` field (docs §13).
    let hint = family_hint(font);
    let family_name = bundled_family.or(hint.as_deref());
    let family = family_name.map_or(Family::SansSerif, Family::Name);
    let attrs = Attrs::new().family(family);

    let mut buffer = {
        let fs = &mut fonts.font_system;
        let mut buffer = Buffer::new(fs, metrics);
        // Bound the width for wrapping/alignment; leave height unbounded so no
        // line is scrolled out of the layout we measure and draw.
        buffer.set_size(inner_w, None);
        buffer.set_wrap(if has_box_w { Wrap::WordOrGlyph } else { Wrap::None });
        buffer.set_text(text, &attrs, Shaping::Advanced, Some(h_align(horizontalalign)));
        buffer.shape_until_scroll(fs, false);
        buffer
    };

    // Measure the laid-out block: widest run + total stacked height.
    let mut text_w = 0.0f32;
    let mut text_h = 0.0f32;
    let mut line_count = 0usize;
    for run in buffer.layout_runs() {
        line_count += 1;
        text_w = text_w.max(run.line_w);
        text_h = text_h.max(run.line_top + run.line_height);
    }
    if line_count == 0 {
        return None;
    }

    let out_w = ceil_clamp(if has_box_w {
        box_size[0]
    } else {
        text_w + 2.0 * pad
    });
    let out_h = ceil_clamp(if has_box_h {
        box_size[1]
    } else {
        text_h + 2.0 * pad
    });

    // Horizontal alignment is already applied within `inner_w`; only the padding
    // inset remains. Vertical alignment distributes the leftover box height.
    let x_off = pad.round() as i32;
    let avail = out_h as f32 - 2.0 * pad - text_h;
    let y_off = (pad
        + if avail > 0.0 {
            avail * v_align_factor(verticalalign)
        } else {
            0.0
        })
    .round() as i32;

    let mut pixels = vec![0u8; (out_w as usize) * (out_h as usize) * 4];
    let mut any_coverage = false;
    buffer.draw(
        &mut fonts.font_system,
        &mut fonts.swash,
        // Opaque white default → the per-pixel callback alpha is pure coverage
        // (the object color is applied later in the fragment shader).
        CtColor::rgba(0xFF, 0xFF, 0xFF, 0xFF),
        |gx, gy, w, h, color| {
            let a = color.a();
            if a == 0 {
                return;
            }
            // `LegacyRenderer::glyph` emits 1×1 pixels, but honor w/h defensively.
            for dy in 0..h as i32 {
                for dx in 0..w as i32 {
                    let px = gx + dx + x_off;
                    let py = gy + dy + y_off;
                    if px < 0 || py < 0 || px >= out_w as i32 || py >= out_h as i32 {
                        continue;
                    }
                    let idx = ((py as u32 * out_w + px as u32) * 4) as usize;
                    // Keep the strongest coverage where glyphs overlap.
                    if a > pixels[idx + 3] {
                        pixels[idx] = 0xFF;
                        pixels[idx + 1] = 0xFF;
                        pixels[idx + 2] = 0xFF;
                        pixels[idx + 3] = a;
                        any_coverage = true;
                    }
                }
            }
        },
    );

    Some(TextRaster {
        pixels,
        width: out_w,
        height: out_h,
        line_count,
        any_coverage,
    })
}

/// Round up to at least 1 and clamp to [`MAX_EDGE`] (V9 allocation guard).
fn ceil_clamp(v: f32) -> u32 {
    if !v.is_finite() || v <= 0.0 {
        return 1;
    }
    (v.ceil() as u32).clamp(1, MAX_EDGE)
}

/// Upload a [`TextRaster`] coverage bitmap as a linear-filtered, clamp-to-edge
/// `Rgba8Unorm` texture (docs §7.4 — text samples with linear filtering for
/// anti-aliasing; clamp so the quad edges do not wrap).
#[must_use]
pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, raster: &TextRaster) -> GpuTexture {
    let width = raster.width.max(1);
    let height = raster.height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("kirie-scene-text-atlas"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let need = (width * height * 4) as usize;
    if raster.pixels.len() >= need {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &raster.pixels[..need],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("kirie-scene-text-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..wgpu::SamplerDescriptor::default()
    });
    GpuTexture {
        texture,
        view,
        sampler,
        width,
        height,
        uv_crop: [1.0, 1.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Explicit newlines produce exactly that many laid-out lines, independent of
    /// the installed font set (line breaking is font-agnostic).
    #[test]
    fn multiline_line_count_matches_newlines() {
        let mut fonts = TextFonts::new();
        let r = rasterize(
            &mut fonts,
            "line one\nline two\nline three",
            "",
            32.0,
            [0.0, 0.0],
            "left",
            "top",
            0.0,
            None,
        )
        .expect("non-empty text rasterizes");
        assert_eq!(r.line_count, 3, "three newline-separated lines");
        // Three 32pt lines at 1.2 line-height ≈ 115px tall — always > one line.
        assert!(
            r.height >= (32.0 * LINE_HEIGHT_RATIO * 2.0) as u32,
            "height {} spans multiple lines",
            r.height
        );
        assert!(r.width > 0 && r.height > 0, "non-degenerate bounds");
    }

    /// A single line is exactly one laid-out line and a positive box.
    #[test]
    fn single_line_bounds() {
        let mut fonts = TextFonts::new();
        let r = rasterize(
            &mut fonts,
            "Hello",
            "",
            24.0,
            [0.0, 0.0],
            "center",
            "center",
            0.0,
            None,
        )
        .expect("non-empty text rasterizes");
        assert_eq!(r.line_count, 1);
        assert!(r.height > 0);
        // Width can only be measured when a font face is available; when the CI
        // image ships no fonts, `fontdb` finds nothing and the run is empty.
        if fonts.face_count() > 0 {
            assert!(r.width > 1, "measured a non-empty advance");
            assert!(r.any_coverage, "rasterized real glyph coverage");
        }
    }

    /// A fixed bounding box drives the bitmap size regardless of the text extent.
    #[test]
    fn box_size_sets_bitmap_dims() {
        let mut fonts = TextFonts::new();
        let r = rasterize(
            &mut fonts,
            "x",
            "",
            16.0,
            [200.0, 120.0],
            "center",
            "center",
            0.0,
            None,
        )
        .expect("rasterizes");
        assert_eq!(r.width, 200);
        assert_eq!(r.height, 120);
    }

    /// Empty text renders nothing (V9: caller skips it, no panic).
    #[test]
    fn empty_text_is_none() {
        let mut fonts = TextFonts::new();
        assert!(
            rasterize(
                &mut fonts,
                "",
                "any",
                32.0,
                [0.0, 0.0],
                "center",
                "center",
                0.0,
                None
            )
            .is_none()
        );
    }

    /// A pathological point size cannot request an unbounded allocation.
    #[test]
    fn oversize_is_clamped() {
        assert_eq!(ceil_clamp(f32::INFINITY), 1);
        assert_eq!(ceil_clamp(-5.0), 1);
        assert_eq!(ceil_clamp(1_000_000.0), MAX_EDGE);
        assert_eq!(ceil_clamp(63.2), 64);
    }

    /// Font-field → family-hint mapping (docs §13 spellings).
    #[test]
    fn family_hint_mapping() {
        assert_eq!(family_hint(""), None);
        assert_eq!(family_hint("   "), None);
        assert_eq!(family_hint("systemfont_arial").as_deref(), Some("arial"));
        assert_eq!(
            family_hint("fonts/VCR_OSD_MONO.ttf").as_deref(),
            Some("VCR_OSD_MONO")
        );
        assert_eq!(
            family_hint("workshop/123/My Font.otf").as_deref(),
            Some("My Font")
        );
    }
}
