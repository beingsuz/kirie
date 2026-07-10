//! Output scaling UV math — fill / fit / stretch / default.
//!
//! Port of the C++ `WallpaperState` UV computation
//! (docs/render-architecture.md §4; WallpaperState.cpp:20-131). kirie-render
//! is still a placeholder crate, so the math lives here for now and should
//! move to (or be re-exported from) kirie-render once the scene renderer
//! exists — noted for the integration step.
//!
//! Semantics per docs/render-architecture.md §4:
//! * **stretch** — full 0..1 range (distorts);
//! * **fill** — scale by `max(vw/pw, vh/ph)`, crop the overflowing axis;
//! * **fit** — same with `min(...)`; UVs go *outside* [0, 1] and the
//!   sampler/shader decides what shows there (this crate letterboxes with
//!   black in the fragment shader);
//! * **default** — crop U or V only when viewport/projection orientations
//!   disagree.
//!
//! The reference computes intermediate sizes as C++ `int` (truncating
//! float→int conversions, WallpaperState.cpp:37,53,86-88,107-109); this
//! port reproduces those truncations for pixel-exact parity.
//!
//! Orientation note: the C++ default (`vflip == false`) V range is `1..0`
//! because its wallpapers sample a GL FBO whose row 0 is the bottom
//! (docs/render-architecture.md §4 resetUVs). This renderer samples a CPU
//! upload whose row 0 is the *top*, so it uses the `vflip == true` branch
//! (`0..1`) — same picture on screen, no double flip.

/// Output scaling mode (docs/render-architecture.md §4 scaling enum and
/// CLI strings: `default`, `fit`, `fill`, `stretch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// `DefaultUVs`: crop only on orientation mismatch.
    #[default]
    Default,
    /// `ZoomFillUVs`: cover the viewport, crop overflow.
    Fill,
    /// `ZoomFitUVs`: contain in the viewport, letterbox.
    Fit,
    /// `StretchUVs`: full range, distorts.
    Stretch,
}

impl ScalingMode {
    /// Parse the CLI string form (docs/render-architecture.md §4:
    /// `"default"`, `"fit"`, `"fill"`, `"stretch"`).
    #[must_use]
    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "fit" => Some(Self::Fit),
            "fill" => Some(Self::Fill),
            "stretch" => Some(Self::Stretch),
            _ => None,
        }
    }
}

/// Texture-space sampling rectangle. Values may lie outside `[0, 1]` in
/// `Fit` mode (docs/render-architecture.md §4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvRect {
    /// U at the left edge of the viewport.
    pub ustart: f32,
    /// U at the right edge of the viewport.
    pub uend: f32,
    /// V at the top edge of the viewport.
    pub vstart: f32,
    /// V at the bottom edge of the viewport.
    pub vend: f32,
}

impl UvRect {
    /// Full texture, no crop (resetUVs, vflip branch — see module docs).
    #[must_use]
    pub const fn full() -> Self {
        Self {
            ustart: 0.0,
            uend: 1.0,
            vstart: 0.0,
            vend: 1.0,
        }
    }
}

/// Crop U for a projection scaled to `proj_w × proj_h`
/// (WallpaperState.cpp:34-47 `updateUs`, including the `int newWidth`
/// truncation).
fn update_us(viewport_w: f32, viewport_h: f32, proj_w: i32, proj_h: i32, uv: &mut UvRect) {
    if proj_h == 0 {
        return;
    }
    let new_width = (viewport_h / proj_h as f32 * proj_w as f32) as i32;
    if new_width == 0 {
        return;
    }
    let new_center = new_width as f32 / 2.0;
    let viewport_center = viewport_w / 2.0;
    uv.ustart = (new_center - viewport_center) / new_width as f32;
    uv.uend = (new_center + viewport_center) / new_width as f32;
}

/// Crop V (WallpaperState.cpp:50-67 `updateVs`, vflip branch — module
/// docs explain why).
fn update_vs(viewport_w: f32, viewport_h: f32, proj_w: i32, proj_h: i32, uv: &mut UvRect) {
    if proj_w == 0 {
        return;
    }
    let new_height = (viewport_w / proj_w as f32 * proj_h as f32) as i32;
    if new_height == 0 {
        return;
    }
    let new_center = new_height as f32 / 2.0;
    let viewport_center = viewport_h / 2.0;
    uv.vstart = (new_center - viewport_center) / new_height as f32;
    uv.vend = (new_center + viewport_center) / new_height as f32;
}

/// Compute the UV crop for one video frame in one viewport
/// (docs/render-architecture.md §4; WallpaperState.cpp:70-131).
///
/// `video_w`/`video_h` play the role of the projection size: the video
/// texture is kept at its native size and all scaling happens here, in
/// composition (docs/subsystems-misc.md §2.2).
#[must_use]
pub fn compute_uvs(
    mode: ScalingMode,
    viewport_w: u32,
    viewport_h: u32,
    video_w: u32,
    video_h: u32,
) -> UvRect {
    let mut uv = UvRect::full();
    if viewport_w == 0 || viewport_h == 0 || video_w == 0 || video_h == 0 {
        return uv;
    }
    let (vw, vh) = (viewport_w as f32, viewport_h as f32);
    let (pw, ph) = (video_w as i32, video_h as i32);

    match mode {
        ScalingMode::Stretch => {}
        ScalingMode::Fill | ScalingMode::Fit => {
            let m1 = vw / pw as f32;
            let m2 = vh / ph as f32;
            let m = if mode == ScalingMode::Fill {
                m1.max(m2)
            } else {
                m1.min(m2)
            };
            // `projectionWidth *= m` on C++ ints truncates
            // (WallpaperState.cpp:86-88,107-109).
            let scaled_w = (pw as f32 * m) as i32;
            let scaled_h = (ph as f32 * m) as i32;
            if scaled_w != viewport_w as i32 {
                update_us(vw, vh, scaled_w, scaled_h, &mut uv);
            } else if scaled_h != viewport_h as i32 {
                update_vs(vw, vh, scaled_w, scaled_h, &mut uv);
            }
        }
        ScalingMode::Default => {
            let (vw_i, vh_i) = (viewport_w as i32, viewport_h as i32);
            // Orientation-mismatch crops (WallpaperState.cpp:115-131).
            if (vh_i > vw_i && pw >= ph) || (vw_i > vh_i && ph > pw) {
                update_us(vw, vh, pw, ph, &mut uv);
            }
            if (vw_i > vh_i && pw >= ph) || (vh_i > vw_i && ph > pw) {
                update_vs(vw, vh, pw, ph, &mut uv);
            }
        }
    }
    uv
}

#[cfg(test)]
mod tests {
    use super::{ScalingMode, UvRect, compute_uvs};

    fn assert_uv(actual: UvRect, expected: (f32, f32, f32, f32)) {
        let (us, ue, vs, ve) = expected;
        assert!(
            (actual.ustart - us).abs() < 1e-6
                && (actual.uend - ue).abs() < 1e-6
                && (actual.vstart - vs).abs() < 1e-6
                && (actual.vend - ve).abs() < 1e-6,
            "got {actual:?}, expected ({us}, {ue}, {vs}, {ve})"
        );
    }

    #[test]
    fn same_aspect_is_identity_in_every_mode() {
        for mode in [
            ScalingMode::Default,
            ScalingMode::Fill,
            ScalingMode::Fit,
            ScalingMode::Stretch,
        ] {
            assert_uv(compute_uvs(mode, 1920, 1080, 1920, 1080), (0.0, 1.0, 0.0, 1.0));
        }
    }

    #[test]
    fn stretch_is_always_full_range() {
        assert_uv(
            compute_uvs(ScalingMode::Stretch, 1000, 1000, 2000, 1000),
            (0.0, 1.0, 0.0, 1.0),
        );
    }

    #[test]
    fn fill_crops_the_wide_axis() {
        // Square viewport, 2:1 video: fill scales by max(0.5, 1.0) = 1.0,
        // width overflows -> centered U crop of the middle half
        // (WallpaperState.cpp:74-93 via docs/render-architecture.md §4).
        assert_uv(
            compute_uvs(ScalingMode::Fill, 1000, 1000, 2000, 1000),
            (0.25, 0.75, 0.0, 1.0),
        );
    }

    #[test]
    fn fill_crops_the_tall_axis() {
        assert_uv(
            compute_uvs(ScalingMode::Fill, 1000, 1000, 1000, 2000),
            (0.0, 1.0, 0.25, 0.75),
        );
    }

    #[test]
    fn fit_letterboxes_with_out_of_range_uvs() {
        // Square viewport, 2:1 video: fit scales by min(0.5, 1.0) = 0.5,
        // height underflows -> V range extends outside [0,1]
        // (docs/render-architecture.md §4: fit relies on what is sampled
        // outside the texture).
        assert_uv(
            compute_uvs(ScalingMode::Fit, 1000, 1000, 2000, 1000),
            (0.0, 1.0, -0.5, 1.5),
        );
    }

    #[test]
    fn default_landscape_viewport_landscape_video_crops_v() {
        // 16:9 viewport, 4:3 video, orientations agree (landscape) ->
        // updateVs path (WallpaperState.cpp:123-127): newHeight =
        // 1920/1440*1080 = 1440 -> V in [0.125, 0.875].
        assert_uv(
            compute_uvs(ScalingMode::Default, 1920, 1080, 1440, 1080),
            (0.0, 1.0, 0.125, 0.875),
        );
    }

    #[test]
    fn default_portrait_viewport_landscape_video_crops_u() {
        // Portrait viewport + landscape video -> updateUs
        // (WallpaperState.cpp:119-121): newWidth = 1920/1080*1920 = 3413,
        // U = center ± 540 over 3413.
        let uv = compute_uvs(ScalingMode::Default, 1080, 1920, 1920, 1080);
        let new_width = 3413.0f32;
        let expected_us = (new_width / 2.0 - 540.0) / new_width;
        let expected_ue = (new_width / 2.0 + 540.0) / new_width;
        assert_uv(uv, (expected_us, expected_ue, 0.0, 1.0));
    }

    #[test]
    fn zero_sizes_return_full_range() {
        assert_uv(
            compute_uvs(ScalingMode::Fill, 0, 1080, 1920, 1080),
            (0.0, 1.0, 0.0, 1.0),
        );
        assert_uv(
            compute_uvs(ScalingMode::Fill, 1920, 1080, 0, 0),
            (0.0, 1.0, 0.0, 1.0),
        );
    }

    #[test]
    fn cli_strings_parse() {
        assert_eq!(ScalingMode::from_cli("default"), Some(ScalingMode::Default));
        assert_eq!(ScalingMode::from_cli("fit"), Some(ScalingMode::Fit));
        assert_eq!(ScalingMode::from_cli("fill"), Some(ScalingMode::Fill));
        assert_eq!(ScalingMode::from_cli("stretch"), Some(ScalingMode::Stretch));
        assert_eq!(ScalingMode::from_cli("cover"), None);
    }
}
