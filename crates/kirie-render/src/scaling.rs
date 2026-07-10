//! Output-scaling and clamping modes shared by every wallpaper type.
//!
//! Ports the reference `WallpaperState` UV math verbatim
//! (docs/render-architecture.md §4; `WallpaperState.cpp:20-131`): the
//! wallpaper content is presented through a *UV window* over a fullscreen
//! quad — the window crops (values inside `[0, 1]`) or over-scans (values
//! outside `[0, 1]`, resolved by the clamp mode) the content.
//!
//! This module is THE shared scaling implementation: `kirie-video` and the
//! scene compositor consume the same [`ScalingMode`]/[`ClampMode`] enums and
//! [`ScalingMode::uv_window`] math.

use crate::error::RenderError;

/// Output scaling mode, CLI `--scaling` (docs/compat-cli.md §2, §3.1:
/// `stretch→StretchUVs`, `fit→ZoomFitUVs`, `fill→ZoomFillUVs`,
/// `default→DefaultUVs`; enum `WallpaperState.h:17-22`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// `default`: crop U or V only when the viewport/content orientations
    /// disagree (docs/render-architecture.md §4; the CLI default,
    /// docs/compat-cli.md §2).
    #[default]
    Default,
    /// `fit`: scale by `min(vw/pw, vh/ph)`; the short axis leaves the UV
    /// window outside `[0, 1]`, resolved by the clamp mode
    /// (docs/render-architecture.md §4).
    Fit,
    /// `fill`: scale by `max(vw/pw, vh/ph)`, crop the overflowing axis
    /// (docs/render-architecture.md §4).
    Fill,
    /// `stretch`: full `[0, 1]` window, distorts
    /// (docs/render-architecture.md §4).
    Stretch,
}

impl ScalingMode {
    /// Parse a CLI `--scaling` value (docs/compat-cli.md §2: choices are
    /// `stretch|fit|fill|default`, anything else is rejected).
    pub fn from_cli(value: &str) -> Result<Self, RenderError> {
        match value {
            "default" => Ok(Self::Default),
            "fit" => Ok(Self::Fit),
            "fill" => Ok(Self::Fill),
            "stretch" => Ok(Self::Stretch),
            other => Err(RenderError::BadScalingMode(other.to_owned())),
        }
    }

    /// The CLI spelling of this mode (docs/compat-cli.md §2).
    #[must_use]
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Fit => "fit",
            Self::Fill => "fill",
            Self::Stretch => "stretch",
        }
    }

    /// Compute the UV window for `content` (the wallpaper's native size —
    /// "projection" in the reference) presented on a `viewport` of physical
    /// pixels (docs/render-architecture.md §4, `WallpaperState.cpp:20-131`).
    ///
    /// Zero dimensions on either side yield the full window (the reference
    /// would divide by zero; malformed input must not panic, SPEC §V9).
    #[must_use]
    pub fn uv_window(self, content: (u32, u32), viewport: (u32, u32)) -> UvWindow {
        let (cw, ch) = (content.0 as f32, content.1 as f32);
        let (vw, vh) = (viewport.0 as f32, viewport.1 as f32);
        if cw <= 0.0 || ch <= 0.0 || vw <= 0.0 || vh <= 0.0 {
            return UvWindow::FULL;
        }

        // Cross-multiplied aspect comparison: `vh*cw > vw*ch ⇔ vh/ch >
        // vw/cw ⇔ m2 > m1` with m1/m2 the fill/fit scale candidates
        // (docs/render-architecture.md §4; WallpaperState.cpp:81-91 —
        // the reference scales integer dims by max/min(m1, m2) and picks
        // the axis whose scaled size differs from the viewport; the
        // products below decide the same axis without the intermediate
        // truncation).
        let wide = vh * cw; // ∝ m2 = vh/ch
        let tall = vw * ch; // ∝ m1 = vw/cw

        match self {
            // §4: stretch = full range (distorts).
            Self::Stretch => UvWindow::FULL,
            // §4: fill = max scale; the axis that overflows is cropped
            // (WallpaperState.cpp:74-93).
            Self::Fill => {
                if wide > tall {
                    UvWindow::with_u(u_range(cw, ch, vw, vh))
                } else if tall > wide {
                    UvWindow::with_v(v_range(cw, ch, vw, vh))
                } else {
                    UvWindow::FULL
                }
            }
            // §4: fit = min scale; the short axis goes outside [0, 1] and
            // relies on the wrap/clamp mode (WallpaperState.cpp:95-114).
            Self::Fit => {
                if wide < tall {
                    UvWindow::with_u(u_range(cw, ch, vw, vh))
                } else if tall < wide {
                    UvWindow::with_v(v_range(cw, ch, vw, vh))
                } else {
                    UvWindow::FULL
                }
            }
            // §4: default = adjust U and/or V per the orientation rules of
            // WallpaperState.cpp:116-131 (portrait viewport + landscape
            // content adjusts U, etc.; a square viewport adjusts neither).
            Self::Default => {
                let mut window = UvWindow::FULL;
                if (vh > vw && cw >= ch) || (vw > vh && ch > cw) {
                    (window.u0, window.u1) = u_range(cw, ch, vw, vh);
                }
                if (vw > vh && cw >= ch) || (vh > vw && ch > cw) {
                    (window.v0, window.v1) = v_range(cw, ch, vw, vh);
                }
                window
            }
        }
    }
}

/// UV clamping mode for out-of-window sampling, CLI `--clamp`
/// (docs/compat-cli.md §2, §3.1: `clamp→TextureFlags_ClampUVs`,
/// `border→TextureFlags_ClampUVsBorder`, `repeat→TextureFlags_NoFlags`;
/// GL wrap semantics per docs/render-architecture.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClampMode {
    /// `clamp` → `GL_CLAMP_TO_EDGE` (the CLI default, docs/compat-cli.md §2).
    #[default]
    Clamp,
    /// `border` → `GL_CLAMP_TO_BORDER` with the GL default border color,
    /// transparent black (docs/render-architecture.md §4, CFBO.cpp:28-37).
    Border,
    /// `repeat` → `GL_REPEAT` (docs/render-architecture.md §4).
    Repeat,
}

impl ClampMode {
    /// Parse a CLI `--clamp` value (docs/compat-cli.md §2: choices are
    /// `clamp|border|repeat`, anything else is rejected).
    pub fn from_cli(value: &str) -> Result<Self, RenderError> {
        match value {
            "clamp" => Ok(Self::Clamp),
            "border" => Ok(Self::Border),
            "repeat" => Ok(Self::Repeat),
            other => Err(RenderError::BadClampMode(other.to_owned())),
        }
    }

    /// The CLI spelling of this mode (docs/compat-cli.md §2).
    #[must_use]
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::Clamp => "clamp",
            Self::Border => "border",
            Self::Repeat => "repeat",
        }
    }
}

/// The UV window of one presented frame: `(u0, v0)` is sampled at the
/// viewport's top-left corner, `(u1, v1)` at its bottom-right, with V
/// increasing downward (image row order).
///
/// This matches the reference *Wayland* present path (vflip == true:
/// `resetUVs` puts vstart at 0, docs/render-architecture.md §2.4, §4,
/// `WallpaperState.cpp:20-31`); the X11 readback path's flipped V is a GL
/// framebuffer artifact that does not exist under wgpu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvWindow {
    /// U at the viewport's left edge.
    pub u0: f32,
    /// V at the viewport's top edge.
    pub v0: f32,
    /// U at the viewport's right edge.
    pub u1: f32,
    /// V at the viewport's bottom edge.
    pub v1: f32,
}

impl UvWindow {
    /// The identity window: content maps `[0, 1]²` onto the viewport.
    pub const FULL: Self = Self {
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
    };

    fn with_u((u0, u1): (f32, f32)) -> Self {
        Self { u0, u1, ..Self::FULL }
    }

    fn with_v((v0, v1): (f32, f32)) -> Self {
        Self { v0, v1, ..Self::FULL }
    }

    /// Per-vertex UVs for a 4-vertex triangle-strip fullscreen quad in the
    /// order top-left, bottom-left, top-right, bottom-right — the same
    /// corner layout the reference uploads per frame
    /// (docs/render-architecture.md §2.5: tex coords are the
    /// `ustart/uend/vstart/vend` corners).
    #[must_use]
    pub fn strip_corners(&self) -> [[f32; 2]; 4] {
        [
            [self.u0, self.v0],
            [self.u0, self.v1],
            [self.u1, self.v0],
            [self.u1, self.v1],
        ]
    }
}

/// U range so the content, scaled to match the viewport height, is centered
/// horizontally (docs/render-architecture.md §4; `WallpaperState.cpp:34-47`:
/// `newW = vh/ch*cw; ustart = (newW/2 - vw/2)/newW; uend = (newW/2 +
/// vw/2)/newW`). Computed as `0.5 ∓ vw·ch/(2·vh·cw)` — algebraically the
/// same expression, but the integer-exact f32 products make the division the
/// only rounding step (the reference truncates `newW` to int, which this
/// deliberately does not reproduce; the doc formula is the pinned behavior).
fn u_range(cw: f32, ch: f32, vw: f32, vh: f32) -> (f32, f32) {
    let half = (vw * ch) / (2.0 * vh * cw);
    (0.5 - half, 0.5 + half)
}

/// V range so the content, scaled to match the viewport width, is centered
/// vertically (docs/render-architecture.md §4; `WallpaperState.cpp:50-67`,
/// vflip == true branch — see [`UvWindow`] for the V convention).
fn v_range(cw: f32, ch: f32, vw: f32, vh: f32) -> (f32, f32) {
    let half = (vh * cw) / (2.0 * vw * ch);
    (0.5 - half, 0.5 + half)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HD: (u32, u32) = (1920, 1080); // landscape content
    const HD_PORTRAIT: (u32, u32) = (1080, 1920); // portrait content
    const SQUARE: (u32, u32) = (1024, 1024); // square content

    fn window(mode: ScalingMode, content: (u32, u32), viewport: (u32, u32)) -> UvWindow {
        mode.uv_window(content, viewport)
    }

    // ---- stretch ---------------------------------------------------------

    #[test]
    fn stretch_is_always_full_window() {
        for content in [HD, HD_PORTRAIT, SQUARE] {
            for viewport in [(1920, 1080), (1080, 1920), (1000, 1000), (2560, 1080)] {
                assert_eq!(window(ScalingMode::Stretch, content, viewport), UvWindow::FULL);
            }
        }
    }

    // ---- fill (docs/render-architecture.md §4: crop overflowing axis) ----

    #[test]
    fn fill_matching_aspect_is_full() {
        assert_eq!(window(ScalingMode::Fill, HD, (3840, 2160)), UvWindow::FULL);
        assert_eq!(window(ScalingMode::Fill, SQUARE, (512, 512)), UvWindow::FULL);
    }

    #[test]
    fn fill_wider_viewport_crops_v() {
        // 1920x1080 on 2560x1080: scale = 2560/1920, newH = 1440.
        // v = 0.5 ∓ 1080/2880 = 0.125 / 0.875 (exact in f32).
        let w = window(ScalingMode::Fill, HD, (2560, 1080));
        assert_eq!((w.u0, w.u1), (0.0, 1.0));
        assert_eq!((w.v0, w.v1), (0.125, 0.875));
    }

    #[test]
    fn fill_portrait_viewport_crops_u() {
        // 1920x1080 on 1080x1920: newW = 1920/1080*1920 = 3413.3…;
        // u = 0.5 ∓ 1080²/(2·1920²) = 0.5 ∓ 81/512 (dyadic → exact).
        let w = window(ScalingMode::Fill, HD, (1080, 1920));
        assert_eq!((w.u0, w.u1), (0.5 - 81.0 / 512.0, 0.5 + 81.0 / 512.0));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
    }

    #[test]
    fn fill_square_viewport_on_landscape_crops_u() {
        // 1920x1080 on 1000x1000: u = 0.5 ∓ 1000·1080/(2·1000·1920)
        //                            = 0.5 ∓ 0.28125 (exact).
        let w = window(ScalingMode::Fill, HD, (1000, 1000));
        assert_eq!((w.u0, w.u1), (0.21875, 0.78125));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
    }

    #[test]
    fn fill_square_viewport_on_portrait_crops_v() {
        let w = window(ScalingMode::Fill, HD_PORTRAIT, (1000, 1000));
        assert_eq!((w.u0, w.u1), (0.0, 1.0));
        assert_eq!((w.v0, w.v1), (0.21875, 0.78125));
    }

    // ---- fit (docs/render-architecture.md §4: short axis leaves [0,1]) ---

    #[test]
    fn fit_matching_aspect_is_full() {
        assert_eq!(window(ScalingMode::Fit, HD, (3840, 2160)), UvWindow::FULL);
    }

    #[test]
    fn fit_wider_viewport_overscans_u() {
        // 1920x1080 on 2560x1080: scale = 1, newW = 1920;
        // u = 0.5 ∓ 2560/3840 = 0.5 ∓ f32(2/3) → outside [0, 1].
        let w = window(ScalingMode::Fit, HD, (2560, 1080));
        let two_thirds = 2.0f32 / 3.0;
        assert_eq!((w.u0, w.u1), (0.5 - two_thirds, 0.5 + two_thirds));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
        assert!(w.u0 < 0.0 && w.u1 > 1.0);
    }

    #[test]
    fn fit_portrait_viewport_overscans_v() {
        // 1920x1080 on 1080x1920: v = 0.5 ∓ 1920²/(2·1080²)
        //                           = 0.5 ∓ f32(1.5802469…).
        let w = window(ScalingMode::Fit, HD, (1080, 1920));
        let half = (1920.0f32 * 1920.0) / (2.0 * 1080.0 * 1080.0);
        assert_eq!((w.u0, w.u1), (0.0, 1.0));
        assert_eq!((w.v0, w.v1), (0.5 - half, 0.5 + half));
        assert!(w.v0 < 0.0 && w.v1 > 1.0);
    }

    #[test]
    fn fit_square_content_on_landscape_overscans_u() {
        // 1024² on 2048x1024: u = 0.5 ∓ 2048/2048 = -0.5 / 1.5 (exact).
        let w = window(ScalingMode::Fit, SQUARE, (2048, 1024));
        assert_eq!((w.u0, w.u1), (-0.5, 1.5));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
    }

    // ---- default (docs/render-architecture.md §4 orientation rules) ------

    #[test]
    fn default_landscape_viewport_landscape_content_adjusts_v() {
        // vw>vh, cw>=ch → V from newH = vw·ch/cw. Same-aspect case is the
        // identity: newH == vh → 0/1 exactly (WallpaperState.cpp:116-131).
        assert_eq!(window(ScalingMode::Default, HD, (1920, 1080)), UvWindow::FULL);

        // 21:9 viewport: newH = 2560·1080/1920 = 1440 → crop V like fill.
        let w = window(ScalingMode::Default, HD, (2560, 1080));
        assert_eq!((w.u0, w.u1), (0.0, 1.0));
        assert_eq!((w.v0, w.v1), (0.125, 0.875));

        // 4:3 viewport on 16:9 content: newH = 1024·1080/1920 = 576 < 768
        // → V overscans (fit-like), u untouched: v = 0.5 ∓ 768/1152 =
        // 0.5 ∓ f32(2/3).
        let w = window(ScalingMode::Default, HD, (1024, 768));
        let two_thirds = 2.0f32 / 3.0;
        assert_eq!((w.u0, w.u1), (0.0, 1.0));
        assert_eq!((w.v0, w.v1), (0.5 - two_thirds, 0.5 + two_thirds));
    }

    #[test]
    fn default_portrait_viewport_landscape_content_adjusts_u() {
        // vh>vw, cw>=ch → U from newW: u = 0.5 ∓ 81/512 (fill-like crop).
        let w = window(ScalingMode::Default, HD, (1080, 1920));
        assert_eq!((w.u0, w.u1), (0.5 - 81.0 / 512.0, 0.5 + 81.0 / 512.0));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
    }

    #[test]
    fn default_landscape_viewport_portrait_content_adjusts_u() {
        // vw>vh, ch>cw → U: newW = vh·cw/ch = 1080·1080/1920 = 607.5;
        // u = 0.5 ∓ 1920·1920/(2·1080·1080) = 0.5 ∓ f32(1.5802469…).
        let w = window(ScalingMode::Default, HD_PORTRAIT, (1920, 1080));
        let half = (1920.0f32 * 1920.0) / (2.0 * 1080.0 * 1080.0);
        assert_eq!((w.u0, w.u1), (0.5 - half, 0.5 + half));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
    }

    #[test]
    fn default_portrait_viewport_portrait_content_adjusts_v() {
        // vh>vw, ch>cw → V from newH = vw·ch/cw (identity when aspects
        // match).
        assert_eq!(
            window(ScalingMode::Default, HD_PORTRAIT, (1080, 1920)),
            UvWindow::FULL
        );

        // Taller viewport (9:21) on 9:16 content: newH = 1080·1920/1080 =
        // 1920 < 2520 → V overscans: v = 0.5 ∓ 2520/3840 = 0.5 ∓ 0.65625.
        let w = window(ScalingMode::Default, HD_PORTRAIT, (1080, 2520));
        assert_eq!((w.u0, w.u1), (0.0, 1.0));
        assert_eq!((w.v0, w.v1), (0.5 - 0.65625, 0.5 + 0.65625));
    }

    #[test]
    fn default_square_viewport_touches_nothing() {
        // Neither WallpaperState.cpp:116-131 condition holds when vw == vh.
        for content in [HD, HD_PORTRAIT, SQUARE] {
            assert_eq!(
                window(ScalingMode::Default, content, (1000, 1000)),
                UvWindow::FULL
            );
        }
    }

    #[test]
    fn default_square_content_counts_as_landscape() {
        // cw >= ch includes square content (WallpaperState.cpp:123, 128).
        // Portrait viewport → U adjusted: u = 0.5 ∓ 1080·1024/(2·1920·1024)
        //                                   = 0.5 ∓ 0.28125.
        let w = window(ScalingMode::Default, SQUARE, (1080, 1920));
        assert_eq!((w.u0, w.u1), (0.5 - 0.28125, 0.5 + 0.28125));
        assert_eq!((w.v0, w.v1), (0.0, 1.0));
    }

    // ---- degenerate inputs (SPEC §V9: never panic) ------------------------

    #[test]
    fn zero_dimensions_yield_full_window() {
        for mode in [
            ScalingMode::Default,
            ScalingMode::Fit,
            ScalingMode::Fill,
            ScalingMode::Stretch,
        ] {
            assert_eq!(mode.uv_window((0, 0), (1920, 1080)), UvWindow::FULL);
            assert_eq!(mode.uv_window(HD, (0, 0)), UvWindow::FULL);
            assert_eq!(mode.uv_window((1920, 0), (0, 1080)), UvWindow::FULL);
        }
    }

    // ---- quad corners ------------------------------------------------------

    #[test]
    fn strip_corners_map_window_to_quad() {
        let w = UvWindow {
            u0: 0.125,
            v0: 0.25,
            u1: 0.875,
            v1: 0.75,
        };
        assert_eq!(
            w.strip_corners(),
            [[0.125, 0.25], [0.125, 0.75], [0.875, 0.25], [0.875, 0.75]]
        );
    }

    // ---- CLI parsing (docs/compat-cli.md §2) -------------------------------

    #[test]
    fn cli_scaling_round_trip() {
        for (s, mode) in [
            ("default", ScalingMode::Default),
            ("fit", ScalingMode::Fit),
            ("fill", ScalingMode::Fill),
            ("stretch", ScalingMode::Stretch),
        ] {
            assert_eq!(ScalingMode::from_cli(s).unwrap(), mode);
            assert_eq!(mode.as_cli_str(), s);
        }
        assert!(matches!(
            ScalingMode::from_cli("Fit"),
            Err(RenderError::BadScalingMode(_))
        ));
        assert!(matches!(
            ScalingMode::from_cli(""),
            Err(RenderError::BadScalingMode(_))
        ));
    }

    #[test]
    fn cli_clamp_round_trip() {
        for (s, mode) in [
            ("clamp", ClampMode::Clamp),
            ("border", ClampMode::Border),
            ("repeat", ClampMode::Repeat),
        ] {
            assert_eq!(ClampMode::from_cli(s).unwrap(), mode);
            assert_eq!(mode.as_cli_str(), s);
        }
        assert!(matches!(
            ClampMode::from_cli("edge"),
            Err(RenderError::BadClampMode(_))
        ));
    }

    #[test]
    fn cli_defaults_match_compat_cli() {
        // docs/compat-cli.md §2: --scaling default = "default",
        // --clamp default = "clamp".
        assert_eq!(ScalingMode::default(), ScalingMode::Default);
        assert_eq!(ClampMode::default(), ClampMode::Clamp);
    }
}
