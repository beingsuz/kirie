//! Typed errors for wallpaper content loading and renderer construction
//! (SPEC §V9: no panics on malformed input).

use std::path::PathBuf;

use kirie_formats::tex::TexError;

/// Everything that can go wrong loading image content or building an
/// [`crate::ImageRenderer`].
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// Filesystem error reading a content file.
    #[error("reading {path}: {source}")]
    Io {
        /// The file being read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// `.tex` parse or decode failure (docs/format-tex.md).
    #[error(transparent)]
    Tex(#[from] TexError),

    /// Plain image (png/jpg/bmp/gif) decode failure.
    #[error(transparent)]
    Image(#[from] image::ImageError),

    /// The `.tex` is a video container (`flags & Video` or mp4 alias,
    /// docs/format-tex.md §7.3) — not image content; kirie-video owns it.
    #[error("video .tex is not image content (docs/format-tex.md §7.3)")]
    VideoTex,

    /// The texture has no images at all (`imageCount == 0`).
    #[error("texture contains no images")]
    NoImages,

    /// An image has no mip levels to decode.
    #[error("texture image {image} has no mip levels")]
    NoMipmaps {
        /// Index of the empty image.
        image: usize,
    },

    /// A TEXS frame's `frameNumber` points past `imageCount`
    /// (docs/format-tex.md §8: it indexes the image holding the frame's
    /// atlas).
    #[error(
        "animation frame {frame} references image {page}, but only {pages} exist (docs/format-tex.md §8)"
    )]
    FramePageOutOfRange {
        /// Index of the offending frame record.
        frame: usize,
        /// The out-of-range `frameNumber`.
        page: usize,
        /// Number of images actually present.
        pages: usize,
    },

    /// `flags & IsGif` was set but the TEXS block holds zero frames —
    /// nothing to schedule (docs/format-tex.md §8).
    #[error("animated texture has an empty frame table (docs/format-tex.md §8)")]
    EmptyAnimation,

    /// A decoded page or the logical content size is zero-sized.
    #[error("zero-sized image content ({width}x{height})")]
    InvalidDimensions {
        /// Decoded width.
        width: u32,
        /// Decoded height.
        height: u32,
    },

    /// Animated gif frames must all share the canvas size (the `image`
    /// crate composites frames onto the logical screen).
    #[error("gif frame is {got_width}x{got_height}, expected canvas {width}x{height}")]
    FrameSizeMismatch {
        /// Canvas width.
        width: u32,
        /// Canvas height.
        height: u32,
        /// Offending frame width.
        got_width: u32,
        /// Offending frame height.
        got_height: u32,
    },

    /// A page exceeds what the wgpu device can allocate.
    #[error("image page {width}x{height} exceeds the device texture limit {max}")]
    TextureTooLarge {
        /// Page width.
        width: u32,
        /// Page height.
        height: u32,
        /// `max_texture_dimension_2d` of the device.
        max: u32,
    },

    /// Unknown `--scaling` value (docs/compat-cli.md §2 choices).
    #[error("unknown scaling mode {0:?} (expected stretch|fit|fill|default, docs/compat-cli.md §2)")]
    BadScalingMode(String),

    /// Unknown `--clamp` value (docs/compat-cli.md §2 choices).
    #[error("unknown clamp mode {0:?} (expected clamp|border|repeat, docs/compat-cli.md §2)")]
    BadClampMode(String),
}
