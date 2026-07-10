//! CPU-side image wallpaper content: decoded RGBA pages plus per-frame
//! atlas placements and timing, independent of any GPU state so the frame
//! schedule and atlas math are unit-testable.

use std::fs;
use std::io::BufReader;
use std::path::Path;

use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;
use kirie_formats::tex::Tex;

use crate::error::RenderError;
use crate::schedule::FrameSchedule;

/// One decoded RGBA8 texture page (a whole atlas image for animated
/// textures, or the single still image).
#[derive(Debug, Clone)]
pub struct ImagePage {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Tightly packed RGBA8 rows, `4·width·height` bytes.
    pub pixels: Vec<u8>,
}

/// Where one displayed frame lives: which page, for how long, and the
/// affine UV placement inside the page.
///
/// `translation`/`axes` are exactly the reference's per-frame atlas
/// uniforms (docs/format-tex.md §8.1 step 5): the quad UV `uv ∈ [0,1]²`
/// maps to `pageUV = translation + uv.x·(axes.x, axes.z) + uv.y·(axes.y,
/// axes.w)`, i.e. `g_Texture0Translation` and `g_Texture0Rotation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramePlacement {
    /// Index into [`ImageContent::pages`] (the TEXS `frameNumber`,
    /// docs/format-tex.md §8, §9).
    pub page: usize,
    /// Display duration in seconds (TEXS `frametime`, docs/format-tex.md
    /// §8; 0 for still images).
    pub duration: f32,
    /// Frame origin over the page, normalized by the page's mip-0 dims —
    /// `g_Texture0Translation = (x/texW, y/texH)` (docs/format-tex.md §8.1).
    pub translation: [f32; 2],
    /// Frame axis vectors, normalized — `g_Texture0Rotation = (width1/texW,
    /// width2/texW, height2/texH, height1/texH)`; nonzero `width2/height2`
    /// encode 90°-rotated atlas frames (docs/format-tex.md §8, §8.1).
    pub axes: [f32; 4],
}

/// Sampler state derived from the texture (docs/format-tex.md §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerSpec {
    /// `NoInterpolation` (0x1): nearest filtering instead of linear
    /// (docs/format-tex.md §6.1, CTexture.cpp:185-191).
    pub nearest: bool,
    /// `ClampUVs` (0x2): clamp-to-edge wrap instead of repeat
    /// (docs/format-tex.md §6.1, CTexture.cpp:177-183).
    pub clamp_uvs: bool,
}

/// Fully decoded image-wallpaper content, ready for GPU upload.
#[derive(Debug, Clone)]
pub struct ImageContent {
    /// Decoded texture pages (one per `.tex` image, docs/format-tex.md §4;
    /// one per composited gif frame for plain animated gifs).
    pub pages: Vec<ImagePage>,
    /// Displayed frames in playback order. Always at least one.
    pub frames: Vec<FramePlacement>,
    /// Sampler behavior for the pages (docs/format-tex.md §6.1).
    pub sampler: SamplerSpec,
    /// Logical content width used for output scaling: TEXS `gifWidth`, the
    /// `.tex` real (crop) width, or the plain image width
    /// (docs/format-tex.md §3, §8; docs/render-architecture.md §4 uses the
    /// wallpaper's native size as the projection).
    pub content_width: u32,
    /// Logical content height; see [`ImageContent::content_width`].
    pub content_height: u32,
}

impl ImageContent {
    /// Load content from a file: `.tex` goes through the Wallpaper Engine
    /// container (docs/format-tex.md), anything else through the `image`
    /// crate (png/jpg/bmp/gif). Animated gifs keep all frames.
    ///
    /// Plain image files have no C++ reference behavior — the reference
    /// rejects image wallpapers outright (docs/render-architecture.md §3)
    /// — so the plain path defines kirie behavior: frames composited by the
    /// gif decoder, delays taken verbatim.
    pub fn from_path(path: &Path) -> Result<Self, RenderError> {
        let is_tex = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("tex"));
        if is_tex {
            let bytes = fs::read(path).map_err(|source| RenderError::Io {
                path: path.to_owned(),
                source,
            })?;
            return Self::from_tex_bytes(&bytes);
        }

        let file = fs::File::open(path).map_err(|source| RenderError::Io {
            path: path.to_owned(),
            source,
        })?;
        let reader = image::ImageReader::new(BufReader::new(file))
            .with_guessed_format()
            .map_err(|source| RenderError::Io {
                path: path.to_owned(),
                source,
            })?;

        if reader.format() == Some(image::ImageFormat::Gif) {
            Self::from_gif(reader.into_inner())
        } else {
            let rgba = reader.decode()?.into_rgba8();
            let (width, height) = rgba.dimensions();
            Self::from_single_rgba8(width, height, rgba.into_raw())
        }
    }

    /// Parse and decode a raw `.tex` byte buffer (e.g. an entry read from
    /// `scene.pkg`); see [`ImageContent::from_tex`].
    pub fn from_tex_bytes(bytes: &[u8]) -> Result<Self, RenderError> {
        let tex = Tex::parse(bytes)?;
        Self::from_tex(&tex)
    }

    /// Decode a parsed `.tex` into displayable content.
    ///
    /// * Video textures are refused (kirie-video owns them,
    ///   docs/format-tex.md §7.3).
    /// * Every image's mip 0 becomes a page (multi-image gifs bind
    ///   `textureID[frameNumber]` per frame, docs/format-tex.md §9).
    /// * With a TEXS block, frames carry the §8.1 atlas placement; without
    ///   one, a single still frame crops the NPOT padding via `realSize /
    ///   textureSize` exactly like the reference's `texcoordCopy` buffer
    ///   (docs/render-architecture.md §7.1).
    pub fn from_tex(tex: &Tex<'_>) -> Result<Self, RenderError> {
        if tex.is_video() || tex.fif.is_mp4() {
            return Err(RenderError::VideoTex);
        }
        if tex.images.is_empty() {
            return Err(RenderError::NoImages);
        }

        let mut pages = Vec::with_capacity(tex.images.len());
        for (index, tex_image) in tex.images.iter().enumerate() {
            if tex_image.mipmaps.is_empty() {
                return Err(RenderError::NoMipmaps { image: index });
            }
            let decoded = tex.decode_rgba8(index, 0)?;
            if decoded.width == 0 || decoded.height == 0 {
                return Err(RenderError::InvalidDimensions {
                    width: decoded.width,
                    height: decoded.height,
                });
            }
            pages.push(ImagePage {
                width: decoded.width,
                height: decoded.height,
                pixels: decoded.pixels,
            });
        }

        // docs/format-tex.md §6.1: NoInterpolation → nearest, ClampUVs →
        // clamp-to-edge; other flag bits don't affect sampling.
        let sampler = SamplerSpec {
            nearest: tex.flags.no_interpolation(),
            clamp_uvs: tex.flags.clamp_uvs(),
        };

        match &tex.animation {
            Some(animation) => {
                if animation.frames.is_empty() {
                    return Err(RenderError::EmptyAnimation);
                }
                let mut frames = Vec::with_capacity(animation.frames.len());
                for (index, frame) in animation.frames.iter().enumerate() {
                    let page = frame.frame_number as usize;
                    let Some(atlas) = pages.get(page) else {
                        return Err(RenderError::FramePageOutOfRange {
                            frame: index,
                            page,
                            pages: pages.len(),
                        });
                    };
                    // §8.1 step 5: normalize by the mip-0 dims of the
                    // *selected image*.
                    let w = atlas.width as f32;
                    let h = atlas.height as f32;
                    frames.push(FramePlacement {
                        page,
                        duration: frame.frametime,
                        translation: [frame.x / w, frame.y / h],
                        axes: [
                            frame.width1 / w,
                            frame.width2 / w,
                            frame.height2 / h,
                            frame.height1 / h,
                        ],
                    });
                }
                // §8: gifWidth/gifHeight is the logical frame size (stored
                // for TEXS0003, back-filled from frame 0 otherwise).
                if animation.gif_width == 0 || animation.gif_height == 0 {
                    return Err(RenderError::InvalidDimensions {
                        width: animation.gif_width,
                        height: animation.gif_height,
                    });
                }
                Ok(Self {
                    pages,
                    frames,
                    sampler,
                    content_width: animation.gif_width,
                    content_height: animation.gif_height,
                })
            }
            None => {
                // Still image: one frame over page 0. The header's real
                // (crop) dims trim NPOT padding — `texcoordCopy = realSize /
                // textureSize` (docs/render-architecture.md §7.1; header
                // fields per docs/format-tex.md §3). FreeImage-path decodes
                // already come out at real size, making the ratio 1.
                let page = &pages[0];
                if tex.width == 0 || tex.height == 0 {
                    return Err(RenderError::InvalidDimensions {
                        width: tex.width,
                        height: tex.height,
                    });
                }
                let u_crop = (tex.width as f32 / page.width as f32).min(1.0);
                let v_crop = (tex.height as f32 / page.height as f32).min(1.0);
                Ok(Self {
                    pages,
                    frames: vec![FramePlacement {
                        page: 0,
                        duration: 0.0,
                        translation: [0.0, 0.0],
                        axes: [u_crop, 0.0, 0.0, v_crop],
                    }],
                    sampler,
                    content_width: tex.width,
                    content_height: tex.height,
                })
            }
        }
    }

    /// Wrap an already-decoded still RGBA8 buffer as single-frame content.
    pub fn from_single_rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        Ok(Self {
            pages: vec![ImagePage {
                width,
                height,
                pixels,
            }],
            frames: vec![FramePlacement {
                page: 0,
                duration: 0.0,
                translation: [0.0, 0.0],
                axes: [1.0, 0.0, 0.0, 1.0],
            }],
            // Plain files carry no .tex flags; linear filtering and edge
            // clamping are the safe defaults for a single full page (no
            // reference behavior exists, docs/render-architecture.md §3).
            sampler: SamplerSpec {
                nearest: false,
                clamp_uvs: true,
            },
            content_width: width,
            content_height: height,
        })
    }

    /// Decode a plain animated gif: the `image` crate composites each frame
    /// onto the logical canvas (disposal handled by its gif frame
    /// iterator), so every frame becomes one full page with an identity
    /// placement and its own delay.
    fn from_gif<R: std::io::BufRead + std::io::Seek>(reader: R) -> Result<Self, RenderError> {
        let decoder = GifDecoder::new(reader)?;
        let frames = decoder.into_frames().collect_frames()?;

        let mut pages = Vec::with_capacity(frames.len());
        let mut placements = Vec::with_capacity(frames.len());
        let mut canvas: Option<(u32, u32)> = None;

        for frame in frames {
            let (numer_ms, denom_ms) = frame.delay().numer_denom_ms();
            let duration = if denom_ms == 0 {
                0.0
            } else {
                numer_ms as f32 / denom_ms as f32 / 1000.0
            };
            let buffer = frame.into_buffer();
            let (width, height) = buffer.dimensions();
            match canvas {
                None => canvas = Some((width, height)),
                Some((cw, ch)) if (cw, ch) != (width, height) => {
                    return Err(RenderError::FrameSizeMismatch {
                        width: cw,
                        height: ch,
                        got_width: width,
                        got_height: height,
                    });
                }
                Some(_) => {}
            }
            placements.push(FramePlacement {
                page: pages.len(),
                duration,
                translation: [0.0, 0.0],
                axes: [1.0, 0.0, 0.0, 1.0],
            });
            pages.push(ImagePage {
                width,
                height,
                pixels: buffer.into_raw(),
            });
        }

        let Some((width, height)) = canvas else {
            return Err(RenderError::EmptyAnimation);
        };
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        Ok(Self {
            pages,
            frames: placements,
            sampler: SamplerSpec {
                nearest: false,
                clamp_uvs: true,
            },
            content_width: width,
            content_height: height,
        })
    }

    /// Playback schedule over [`ImageContent::frames`]
    /// (docs/format-tex.md §8.1).
    #[must_use]
    pub fn schedule(&self) -> FrameSchedule {
        FrameSchedule::new(self.frames.iter().map(|f| f.duration).collect())
    }

    /// Logical content size, the scaling-math projection
    /// (docs/render-architecture.md §4).
    #[must_use]
    pub fn content_size(&self) -> (u32, u32) {
        (self.content_width, self.content_height)
    }
}

#[cfg(test)]
mod tests {
    use kirie_formats::tex::{
        Animation, AnimationVersion, Compression, ContainerVersion, Frame, FreeImageFormat, Mipmap, Tex,
        TexImage, TextureFlags, TextureFormat,
    };

    use super::*;

    /// A synthetic in-memory `.tex` model (fields are all public, so no
    /// byte-level round trip is needed to exercise `from_tex`).
    fn synthetic_tex<'a>(
        payload: &'a [u8],
        width: u32,
        height: u32,
        real_w: u32,
        real_h: u32,
        flags: u32,
        animation: Option<Animation>,
    ) -> Tex<'a> {
        Tex {
            format: TextureFormat::Argb8888,
            flags: TextureFlags(flags),
            texture_width: width,
            texture_height: height,
            width: real_w,
            height: real_h,
            unknown: 0,
            container: ContainerVersion::Texb0003,
            fif: FreeImageFormat::UNKNOWN,
            is_video_mp4: false,
            images: vec![TexImage {
                mipmaps: vec![Mipmap {
                    width,
                    height,
                    compression: Compression::Stored,
                    uncompressed_size: payload.len(),
                    payload,
                }],
            }],
            animation,
        }
    }

    fn rgba_page(width: u32, height: u32) -> Vec<u8> {
        (0..width * height * 4).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn static_tex_crops_npot_padding() {
        // 8x8 stored payload, 6x5 real size → texcoordCopy = real/texture
        // (docs/render-architecture.md §7.1).
        let payload = rgba_page(8, 8);
        let tex = synthetic_tex(&payload, 8, 8, 6, 5, 0, None);
        let content = ImageContent::from_tex(&tex).unwrap();
        assert_eq!(content.pages.len(), 1);
        assert_eq!(content.frames.len(), 1);
        let frame = content.frames[0];
        assert_eq!(frame.page, 0);
        assert_eq!(frame.duration, 0.0);
        assert_eq!(frame.translation, [0.0, 0.0]);
        assert_eq!(frame.axes, [0.75, 0.0, 0.0, 0.625]);
        assert_eq!(content.content_size(), (6, 5));
        assert!(!content.schedule().is_animated());
        // No flags → linear + repeat (docs/format-tex.md §6.1).
        assert_eq!(
            content.sampler,
            SamplerSpec {
                nearest: false,
                clamp_uvs: false
            }
        );
    }

    #[test]
    fn tex_flags_drive_the_sampler_spec() {
        let payload = rgba_page(4, 4);
        // NoInterpolation | ClampUVs = 0x3 (docs/format-tex.md §6.1).
        let tex = synthetic_tex(&payload, 4, 4, 4, 4, 0x3, None);
        let content = ImageContent::from_tex(&tex).unwrap();
        assert_eq!(
            content.sampler,
            SamplerSpec {
                nearest: true,
                clamp_uvs: true
            }
        );
    }

    #[test]
    fn animated_tex_builds_atlas_placements() {
        // 8x4 atlas holding two 4x4 frames side by side; TEXS0003-style
        // float frame records (docs/format-tex.md §8).
        let payload = rgba_page(8, 4);
        let animation = Animation {
            version: AnimationVersion::Texs0003,
            gif_width: 4,
            gif_height: 4,
            frames: vec![
                Frame {
                    frame_number: 0,
                    frametime: 0.25,
                    x: 0.0,
                    y: 0.0,
                    width1: 4.0,
                    width2: 0.0,
                    height2: 0.0,
                    height1: 4.0,
                },
                Frame {
                    frame_number: 0,
                    frametime: 0.5,
                    x: 4.0,
                    y: 0.0,
                    width1: 4.0,
                    width2: 0.0,
                    height2: 0.0,
                    height1: 4.0,
                },
            ],
        };
        let tex = synthetic_tex(&payload, 8, 4, 8, 4, TextureFlags::IS_GIF, Some(animation));
        let content = ImageContent::from_tex(&tex).unwrap();
        assert_eq!(content.content_size(), (4, 4));
        assert_eq!(content.frames.len(), 2);
        // §8.1 step 5: translation = origin/texDims, axes normalized the
        // same way.
        assert_eq!(content.frames[0].translation, [0.0, 0.0]);
        assert_eq!(content.frames[0].axes, [0.5, 0.0, 0.0, 1.0]);
        assert_eq!(content.frames[1].translation, [0.5, 0.0]);
        assert_eq!(content.frames[1].axes, [0.5, 0.0, 0.0, 1.0]);

        let schedule = content.schedule();
        assert!(schedule.is_animated());
        assert_eq!(schedule.durations(), &[0.25, 0.5]);
        assert_eq!(schedule.frame_at(0.1), 0);
        assert_eq!(schedule.frame_at(0.3), 1);
        assert_eq!(schedule.frame_at(0.8), 0); // wraps at 0.75
    }

    #[test]
    fn rotated_frames_keep_cross_axes() {
        // Nonzero width2/height2 = 90°-rotated atlas frame
        // (docs/format-tex.md §8: interleaved width1, width2, height2,
        // height1 order).
        let payload = rgba_page(8, 8);
        let animation = Animation {
            version: AnimationVersion::Texs0002,
            gif_width: 4,
            gif_height: 4,
            frames: vec![Frame {
                frame_number: 0,
                frametime: 0.1,
                x: 2.0,
                y: 4.0,
                width1: 0.0,
                width2: 4.0,
                height2: 4.0,
                height1: 0.0,
            }],
        };
        let tex = synthetic_tex(&payload, 8, 8, 8, 8, TextureFlags::IS_GIF, Some(animation));
        let content = ImageContent::from_tex(&tex).unwrap();
        assert_eq!(content.frames[0].translation, [0.25, 0.5]);
        assert_eq!(content.frames[0].axes, [0.0, 0.5, 0.5, 0.0]);
    }

    #[test]
    fn malformed_tex_content_yields_typed_errors() {
        // frameNumber past imageCount (docs/format-tex.md §8) — SPEC §V9.
        let payload = rgba_page(4, 4);
        let animation = Animation {
            version: AnimationVersion::Texs0003,
            gif_width: 4,
            gif_height: 4,
            frames: vec![Frame {
                frame_number: 3,
                frametime: 0.1,
                x: 0.0,
                y: 0.0,
                width1: 4.0,
                width2: 0.0,
                height2: 0.0,
                height1: 4.0,
            }],
        };
        let tex = synthetic_tex(&payload, 4, 4, 4, 4, TextureFlags::IS_GIF, Some(animation));
        assert!(matches!(
            ImageContent::from_tex(&tex),
            Err(RenderError::FramePageOutOfRange {
                frame: 0,
                page: 3,
                pages: 1
            })
        ));

        // Empty frame table.
        let empty = Animation {
            version: AnimationVersion::Texs0003,
            gif_width: 4,
            gif_height: 4,
            frames: vec![],
        };
        let tex = synthetic_tex(&payload, 4, 4, 4, 4, TextureFlags::IS_GIF, Some(empty));
        assert!(matches!(
            ImageContent::from_tex(&tex),
            Err(RenderError::EmptyAnimation)
        ));

        // No images at all.
        let mut no_images = synthetic_tex(&payload, 4, 4, 4, 4, 0, None);
        no_images.images.clear();
        assert!(matches!(
            ImageContent::from_tex(&no_images),
            Err(RenderError::NoImages)
        ));

        // Video flag (0x20) → kirie-video's job (docs/format-tex.md §7.3).
        let video = synthetic_tex(&payload, 4, 4, 4, 4, TextureFlags::VIDEO, None);
        assert!(matches!(
            ImageContent::from_tex(&video),
            Err(RenderError::VideoTex)
        ));
    }

    #[test]
    fn plain_animated_gif_keeps_frames_and_delays() {
        // Encode a 2-frame 4x4 gif in memory, then decode through the
        // public path (no reference behavior exists for plain files;
        // docs/render-architecture.md §3).
        let dir = std::env::temp_dir().join("kirie-render-gif-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two-frame.gif");
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut encoder = image::codecs::gif::GifEncoder::new(file);
            encoder.set_repeat(image::codecs::gif::Repeat::Infinite).unwrap();
            let red = image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255]));
            let blue = image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 255, 255]));
            let delay = image::Delay::from_numer_denom_ms(100, 1);
            encoder
                .encode_frame(image::Frame::from_parts(red, 0, 0, delay))
                .unwrap();
            encoder
                .encode_frame(image::Frame::from_parts(blue, 0, 0, delay))
                .unwrap();
        }

        let content = ImageContent::from_path(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(content.pages.len(), 2);
        assert_eq!(content.frames.len(), 2);
        assert_eq!(content.content_size(), (4, 4));
        for (index, frame) in content.frames.iter().enumerate() {
            assert_eq!(frame.page, index);
            assert_eq!(frame.translation, [0.0, 0.0]);
            assert_eq!(frame.axes, [1.0, 0.0, 0.0, 1.0]);
            assert!((frame.duration - 0.1).abs() < 1e-6, "delay {}", frame.duration);
        }
        // First frame stays red after gif palette quantization.
        assert_eq!(&content.pages[0].pixels[0..4], &[255, 0, 0, 255]);
        assert!(content.schedule().is_animated());
    }

    #[test]
    fn zero_sized_rgba_is_rejected() {
        assert!(matches!(
            ImageContent::from_single_rgba8(0, 4, vec![]),
            Err(RenderError::InvalidDimensions { .. })
        ));
    }
}
