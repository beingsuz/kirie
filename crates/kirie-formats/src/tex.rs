//! `.tex` texture container parser + decoder. Spec: docs/format-tex.md
//!
//! A `.tex` file is Wallpaper Engine's universal texture container
//! (docs/format-tex.md §1). It holds one of:
//!
//! 1. raw pixel data (ARGB8888 / R8 / RG88 / DXTn ...), optionally
//!    LZ4-compressed per mip level, with a full mipmap chain (§7.1);
//! 2. whole encoded image files (PNG/JPEG/... bytes verbatim, one complete
//!    file per mip level; the "FreeImage path", §7.2);
//! 3. a whole MP4 video file stored verbatim as the single mip payload
//!    (§7.3);
//! 4. optionally an appended GIF/spritesheet animation frame table (§8).
//!
//! Layout: `"TEXV0005\0"` + `"TEXI0001\0"` magics, a 28-byte header,
//! a `TEXB000N` image container, `imageCount` images of `mipmapCount`
//! mipmaps each, then an optional `TEXS000N` animation block
//! (docs/format-tex.md §1). All multi-byte integers are little-endian
//! (§2); all magic strings are exactly 9 bytes including the trailing
//! NUL (§2).
//!
//! Two layers live here:
//!
//! * **Container parse** — [`Tex::parse`] reads header, container, and
//!   mip/frame tables; mip payloads are lazy zero-copy slices of the
//!   input ([`Mipmap::payload`]).
//! * **Decode** — [`Tex::decode_rgba8`] converts a mip level to tightly
//!   packed RGBA8 pixels; [`Tex::video_payload`] hands out an MP4
//!   payload as opaque bytes (videos are not decoded here).
//!
//! Per SPEC.md §V9 this parser never panics on malformed input: every
//! read is bounds-checked, every offset/size computation uses checked
//! arithmetic, and all failures are typed [`TexError`]s.

use std::borrow::Cow;

use thiserror::Error;

/// Errors produced by the `.tex` parser and decoder (docs/format-tex.md).
#[derive(Debug, Error)]
pub enum TexError {
    /// A magic string did not match its only accepted value. The reference
    /// accepts only `TEXV0005` + `TEXI0001` (docs/format-tex.md §3,
    /// `TextureParser.cpp:185-193`); unknown `TEXB`/`TEXS` magics are hard
    /// errors too (§4, §8).
    #[error("bad {what} magic: expected {expected:?}, got {found:?}")]
    BadMagic {
        /// Which magic was being checked.
        what: &'static str,
        /// The accepted value (without the trailing NUL).
        expected: &'static str,
        /// The rejected bytes, decoded lossily for display.
        found: String,
    },

    /// The input ended before an expected field. The reference reads through
    /// an exception-throwing cursor; any header/table/payload read hitting
    /// EOF is a hard error.
    #[error(
        "truncated texture: need {needed} byte(s) for {what} at offset {offset}, \
         only {available} available"
    )]
    Truncated {
        /// Which field was being read when the data ran out.
        what: &'static str,
        /// Byte offset in the texture where the read started.
        offset: usize,
        /// Bytes required by the field.
        needed: usize,
        /// Bytes actually remaining.
        available: usize,
    },

    /// The header `format` word is not one of the defined enum values.
    /// Values 3 and 5 are not defined and the reference rejects them along
    /// with everything else outside the table (docs/format-tex.md §5,
    /// `TextureParser.cpp:156-178`).
    #[error("unknown texture format {value} (0x{value:08x})")]
    UnknownFormat {
        /// The rejected `format` word.
        value: u32,
    },

    /// The container `freeImageFormat` word is outside the gapless accepted
    /// range −1..=36 (docs/format-tex.md §6.2, `TextureParser.cpp:313-358`).
    #[error("unknown FreeImage format id {value}")]
    UnknownFif {
        /// The rejected FIF id.
        value: i32,
    },

    /// A mip `compression` word other than 0 (stored) or 1 (LZ4). The
    /// reference never validates this and would read `uncompressedSize`
    /// bytes as stored data, silently desyncing the stream on mismatched
    /// size fields (docs/format-tex.md §7 rule 3); only 0 and 1 exist in
    /// the corpus, so we refuse the desync footgun with a typed error
    /// instead (strictness per SPEC.md §V9).
    #[error("unsupported mip compression mode {value} (only 0=stored, 1=LZ4 exist)")]
    UnsupportedCompression {
        /// The rejected `compression` word.
        value: u32,
    },

    /// A size field declared as `i32` in the reference (docs/format-tex.md
    /// §7: `uncompressedSize`/`compressedSize` are read via `nextInt`) is
    /// negative.
    #[error("negative mip {what}: {value}")]
    NegativeSize {
        /// Which size field was negative.
        what: &'static str,
        /// The rejected value.
        value: i32,
    },

    /// An LZ4-compressed mip payload failed to decompress as a single raw
    /// LZ4 block (docs/format-tex.md §7 rule 2: block format, not the frame
    /// format, `LZ4_decompress_safe` semantics).
    #[error("LZ4 decompression failed: {source}")]
    Lz4 {
        /// Underlying lz4_flex block decoder error.
        #[source]
        source: lz4_flex::block::DecompressError,
    },

    /// An LZ4 mip decompressed to a different length than the mip header's
    /// `uncompressedSize`; the reference requires an exact match
    /// (docs/format-tex.md §7 rule 2). Also raised before decompression for
    /// a declared size beyond LZ4's mathematical 255× expansion bound, so a
    /// hostile header cannot force a huge allocation (SPEC.md §V9).
    #[error("LZ4 mip decompressed to {actual} byte(s), header says {expected}")]
    Lz4SizeMismatch {
        /// `uncompressedSize` from the mip header.
        expected: usize,
        /// Actual decompressed length (or the 255× bound that was exceeded).
        actual: usize,
    },

    /// A raw-path mip payload length does not match the verified size
    /// formula for its format (docs/format-tex.md §7.1: 0 mismatches over
    /// all raw corpus mips).
    #[error(
        "payload of {format:?} mip {width}x{height} is {actual} byte(s), \
         format rule says {expected} (docs/format-tex.md §7.1)"
    )]
    WrongPayloadSize {
        /// Header texture format.
        format: TextureFormat,
        /// Mip width in texels.
        width: u32,
        /// Mip height in texels.
        height: u32,
        /// Expected byte length per the §7.1 size rule.
        expected: usize,
        /// Actual payload byte length after (optional) LZ4 decode.
        actual: usize,
    },

    /// RGBA8 decoding is not implemented for this format. Only
    /// [`TextureFormat::Unknown`] (0xFFFFFFFF) hits this: it is accepted by
    /// the parser but carries no payload semantics at all, so decode refuses
    /// instead of guessing (SPEC.md §V10). The other originally
    /// "parser-accepted only" formats (RGB888/RGB565/RG1616f/R16f/BC7/
    /// RGBA1010102/RGBA16161616f/RGB161616f) now decode per SPEC.md §T11.
    #[error("no RGBA8 decoder for texture format {format:?}")]
    UnsupportedFormat {
        /// The undecodable format.
        format: TextureFormat,
    },

    /// Mip dimensions whose pixel/byte count overflows `usize`
    /// (SPEC.md §V9 checked arithmetic; not reachable from sane files).
    #[error("mip dimensions {width}x{height} overflow the address space")]
    Oversized {
        /// Mip width in texels.
        width: u32,
        /// Mip height in texels.
        height: u32,
    },

    /// [`Tex::decode_rgba8`] was called on a video texture. MP4 payloads
    /// are opaque bytes for a video player, not decodable pixels
    /// (docs/format-tex.md §7.3); use [`Tex::video_payload`] instead.
    #[error("texture is a video; use video_payload() for the raw MP4 bytes")]
    IsVideo,

    /// [`Tex::video_payload`] was called on a non-video texture
    /// (docs/format-tex.md §7.3 detection: `isVideoMp4 || flags & Video`).
    #[error("texture is not a video (docs/format-tex.md §7.3)")]
    NotVideo,

    /// Image index out of range (`imageCount` images exist,
    /// docs/format-tex.md §4).
    #[error("no image {index} (texture has {count})")]
    NoSuchImage {
        /// Requested image index.
        index: usize,
        /// Number of images present.
        count: usize,
    },

    /// Mipmap index out of range for the selected image
    /// (docs/format-tex.md §7).
    #[error("no mipmap {index} (image has {count})")]
    NoSuchMipmap {
        /// Requested mip index.
        index: usize,
        /// Number of mip levels present.
        count: usize,
    },

    /// A FreeImage-path payload failed to decode as an embedded image file
    /// (docs/format-tex.md §7.2: each mip is a complete PNG/JPEG/... file).
    #[error("embedded image decode failed: {source}")]
    ImageDecode {
        /// Underlying `image` crate error.
        #[source]
        source: Box<image::ImageError>,
    },

    /// A DXTn block decode failed (buffer size mismatch inside the block
    /// decoder; guarded by the §7.1 size check, so not normally reachable).
    #[error("{format:?} block decode failed: {reason}")]
    BlockDecode {
        /// The block-compressed format being decoded.
        format: TextureFormat,
        /// Error string from the block decoder.
        reason: &'static str,
    },
}

/// Texture pixel/block format (docs/format-tex.md §5, exact numeric values
/// from `Texture.h:70-86`). Values 3 and 5 are not defined; the reference
/// rejects them and every other value outside this set
/// (`TextureParser.cpp:156-178`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// `0xFFFFFFFF` — accepted by the parser, not renderable (§5).
    Unknown,
    /// 0 — 4 bytes/texel; despite the name the payload bytes are R,G,B,A
    /// per texel, uploaded directly as `GL_RGBA`/`GL_UNSIGNED_BYTE` (§5).
    Argb8888,
    /// 1 — 3 bytes/texel (R,G,B), opaque alpha; decoded per SPEC.md §T11
    /// (§5; UNVERIFIED — no corpus sample).
    Rgb888,
    /// 2 — 2 bytes/texel, packed little-endian `RRRRRGGGGGGBBBBB` u16,
    /// bit-replicated to 8-bit; decoded per SPEC.md §T11 (§5; UNVERIFIED).
    Rgb565,
    /// 4 — 16 bytes per 4×4 block, `GL_COMPRESSED_RGBA_S3TC_DXT5_EXT` (§5).
    Dxt5,
    /// 6 — 16 bytes per 4×4 block, `GL_COMPRESSED_RGBA_S3TC_DXT3_EXT` (§5).
    Dxt3,
    /// 7 — 8 bytes per 4×4 block, `GL_COMPRESSED_RGBA_S3TC_DXT1_EXT` (§5).
    Dxt1,
    /// 8 — 2 bytes/texel (R,G), uploaded as `GL_RG` (§5).
    Rg88,
    /// 9 — 1 byte/texel, uploaded as `GL_RED` with unpack alignment 1;
    /// rows are byte-packed with no padding (§5).
    R8,
    /// 10 — 4 bytes/texel, two IEEE half-floats (R,G); tone-mapped to
    /// 8-bit by clamp-to-`[0,1]` per SPEC.md §T11 (§5; UNVERIFIED).
    Rg1616f,
    /// 11 — 2 bytes/texel, one IEEE half-float; decoded per SPEC.md §T11
    /// (§5; UNVERIFIED).
    R16f,
    /// 12 — BC7 block-compressed, 16 bytes per 4×4 block; decoded via
    /// `texture2ddecoder` per SPEC.md §T11 (§5; UNVERIFIED — no corpus
    /// sample).
    Bc7,
    /// 13 — 4 bytes/texel, packed little-endian DXGI `R10G10B10A2` u32;
    /// decoded per SPEC.md §T11 (§5; UNVERIFIED).
    Rgba1010102,
    /// 14 — 8 bytes/texel, four IEEE half-floats (R,G,B,A); tone-mapped to
    /// 8-bit by clamp-to-`[0,1]` per SPEC.md §T11 (§5; UNVERIFIED).
    Rgba16161616f,
    /// 15 — 6 bytes/texel, three IEEE half-floats (R,G,B), opaque alpha;
    /// decoded per SPEC.md §T11 (§5; UNVERIFIED).
    Rgb161616f,
}

impl TextureFormat {
    /// Map the header `format` word to the enum (docs/format-tex.md §5).
    /// Unknown values (including the undefined 3 and 5) are a hard error,
    /// matching `TextureParser.cpp:156-178`.
    fn from_u32(value: u32) -> Result<Self, TexError> {
        Ok(match value {
            0xFFFF_FFFF => Self::Unknown,
            0 => Self::Argb8888,
            1 => Self::Rgb888,
            2 => Self::Rgb565,
            4 => Self::Dxt5,
            6 => Self::Dxt3,
            7 => Self::Dxt1,
            8 => Self::Rg88,
            9 => Self::R8,
            10 => Self::Rg1616f,
            11 => Self::R16f,
            12 => Self::Bc7,
            13 => Self::Rgba1010102,
            14 => Self::Rgba16161616f,
            15 => Self::Rgb161616f,
            other => return Err(TexError::UnknownFormat { value: other }),
        })
    }
}

/// Header `flags` bitfield (docs/format-tex.md §6.1, `Texture.h:88-98`).
///
/// The reference validation is sloppy (accepts any value strictly below
/// `TextureFlags_All` = 524335, `TextureParser.cpp:305-311`); per §6.1
/// reimplementations should treat unknown bits as ignorable, so no value
/// is rejected here and the raw word is preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextureFlags(pub u32);

impl TextureFlags {
    /// 0x1 — GL `NEAREST` filtering instead of `LINEAR` (§6.1).
    pub const NO_INTERPOLATION: u32 = 0x1;
    /// 0x2 — GL wrap `CLAMP_TO_EDGE` instead of `REPEAT` (§6.1).
    pub const CLAMP_UVS: u32 = 0x2;
    /// 0x4 — texture is animated; a TEXS block follows the mip data (§6.1).
    pub const IS_GIF: u32 = 0x4;
    /// 0x8 — `CLAMP_TO_BORDER` on effect render targets; ignored for `.tex`
    /// textures by the reference (§6.1).
    pub const CLAMP_UVS_BORDER: u32 = 0x8;
    /// 0x20 — mip-0 payload is a whole video file (§6.1, §7.3).
    pub const VIDEO: u32 = 0x20;
    /// 0x80000 — defined as "RG88/R8 format where alpha is in G/R channel"
    /// but never read by the reference; semantics UNVERIFIED (§6.1).
    pub const ALPHA_CHANNEL_PRIORITY: u32 = 0x8_0000;

    /// `NoInterpolation` (0x1): nearest-neighbour filtering (§6.1).
    #[must_use]
    pub fn no_interpolation(self) -> bool {
        self.0 & Self::NO_INTERPOLATION != 0
    }

    /// `ClampUVs` (0x2): clamp-to-edge wrapping (§6.1).
    #[must_use]
    pub fn clamp_uvs(self) -> bool {
        self.0 & Self::CLAMP_UVS != 0
    }

    /// `IsGif` (0x4): an animation block follows the mip data (§6.1, §8).
    #[must_use]
    pub fn is_gif(self) -> bool {
        self.0 & Self::IS_GIF != 0
    }

    /// `ClampUVsBorder` (0x8): clamp-to-border (effect render targets only,
    /// §6.1).
    #[must_use]
    pub fn clamp_uvs_border(self) -> bool {
        self.0 & Self::CLAMP_UVS_BORDER != 0
    }

    /// `Video` (0x20): the payload is a whole video file (§6.1, §7.3).
    #[must_use]
    pub fn video(self) -> bool {
        self.0 & Self::VIDEO != 0
    }

    /// `AlphaChannelPriority` (0x80000): defined but never read by the
    /// reference (§6.1).
    #[must_use]
    pub fn alpha_channel_priority(self) -> bool {
        self.0 & Self::ALPHA_CHANNEL_PRIORITY != 0
    }
}

/// FreeImage format id identifying the codec of an embedded image file
/// (docs/format-tex.md §6.2). Signed; every value in the gapless range
/// −1..=36 is defined and accepted, anything else is a hard error
/// (`TextureParser.cpp:313-358`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeImageFormat(pub i32);

impl FreeImageFormat {
    /// −1: no embedded image file — the payload is raw pixel data (§6.2).
    pub const UNKNOWN: Self = Self(-1);
    /// 2: JPEG (§6.2; 4 corpus files).
    pub const JPEG: Self = Self(2);
    /// 13: PNG (§6.2; 28 corpus files).
    pub const PNG: Self = Self(13);
    /// 35: MP4 — aliases FIF_WEBP; a genuine WEBP id 35 is also treated as
    /// MP4 by the reference (§4, `Texture.h:66`).
    pub const MP4: Self = Self(35);

    /// Map the on-disk id, rejecting values outside −1..=36 (§6.2).
    fn from_i32(value: i32) -> Result<Self, TexError> {
        if (-1..=36).contains(&value) {
            Ok(Self(value))
        } else {
            Err(TexError::UnknownFif { value })
        }
    }

    /// True when the payload is raw pixel data, not an image file (§6.2).
    #[must_use]
    pub fn is_raw(self) -> bool {
        self == Self::UNKNOWN
    }

    /// True for the MP4/WEBP alias id 35 (§4).
    #[must_use]
    pub fn is_mp4(self) -> bool {
        self == Self::MP4
    }
}

/// `TEXB` image-container version (docs/format-tex.md §4, `Texture.h:12-18`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerVersion {
    /// `TEXB0001` — mips lack compression/uncompressedSize fields (§7);
    /// UNVERIFIED (none in corpus).
    Texb0001,
    /// `TEXB0002` — adds per-mip compression fields (§7); UNVERIFIED.
    Texb0002,
    /// `TEXB0003` — adds the container `freeImageFormat` field (§4).
    Texb0003,
    /// `TEXB0004` — adds `isVideoMp4` and (only when effectively mp4) an
    /// extra per-mip prefix; non-mp4 TEXB0004 downgrades to the TEXB0003
    /// layout (§4).
    Texb0004,
}

/// `TEXS` animation-block version (docs/format-tex.md §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationVersion {
    /// `TEXS0001` — integer frame coords, fields 5–6 unused; UNVERIFIED (§8).
    Texs0001,
    /// `TEXS0002` — float frame records; UNVERIFIED (§8).
    Texs0002,
    /// `TEXS0003` — adds stored `gifWidth`/`gifHeight` (§8).
    Texs0003,
}

/// Per-mip payload encoding (docs/format-tex.md §7: `compression` word,
/// 0 = stored, 1 = one raw LZ4 block). Other values are rejected at parse
/// time ([`TexError::UnsupportedCompression`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// `compression == 0`: payload bytes are the data itself (§7 rule 1).
    Stored,
    /// `compression == 1`: payload is a single raw LZ4 block — block
    /// format, no magic/headers — expanding to exactly
    /// [`Mipmap::uncompressed_size`] bytes (§7 rule 2).
    Lz4,
}

/// One mip level (docs/format-tex.md §7). The payload is a lazy zero-copy
/// slice of the parsed input; call [`Mipmap::data`] to get the stored or
/// LZ4-expanded bytes.
#[derive(Debug, Clone)]
pub struct Mipmap<'a> {
    /// Mip width in texels (§7). For DXT levels these are the true texel
    /// dims, not block-rounded (§7.1).
    pub width: u32,
    /// Mip height in texels (§7).
    pub height: u32,
    /// Payload encoding (§7). `TEXB0001` mips have no compression field and
    /// are treated as stored (§7).
    pub compression: Compression,
    /// Expected byte length after decompression. For stored mips this is
    /// aliased to the payload length — the on-disk `uncompressedSize` word
    /// is ignored when `compression == 0` (§7 rule 1).
    pub uncompressed_size: usize,
    /// Raw on-disk payload bytes, exactly `compressedSize` long (§7).
    pub payload: &'a [u8],
}

impl<'a> Mipmap<'a> {
    /// The mip's data bytes: the payload itself for stored mips, or the
    /// LZ4-expanded bytes (docs/format-tex.md §7 rule 2: one raw LZ4 block,
    /// `LZ4_decompress_safe` semantics, result must be exactly
    /// [`Mipmap::uncompressed_size`] bytes).
    pub fn data(&self) -> Result<Cow<'a, [u8]>, TexError> {
        match self.compression {
            Compression::Stored => Ok(Cow::Borrowed(self.payload)),
            Compression::Lz4 => {
                // A raw LZ4 block cannot expand a sequence by more than
                // 255× its input; reject implausible declared sizes before
                // allocating so a hostile header cannot force a huge
                // allocation (SPEC.md §V9). Real sizes are far below this.
                let bound = self.payload.len().saturating_mul(255).saturating_add(64);
                if self.uncompressed_size > bound {
                    return Err(TexError::Lz4SizeMismatch {
                        expected: self.uncompressed_size,
                        actual: bound,
                    });
                }
                let out = lz4_flex::block::decompress(self.payload, self.uncompressed_size)
                    .map_err(|source| TexError::Lz4 { source })?;
                if out.len() != self.uncompressed_size {
                    return Err(TexError::Lz4SizeMismatch {
                        expected: self.uncompressed_size,
                        actual: out.len(),
                    });
                }
                Ok(Cow::Owned(out))
            }
        }
    }
}

/// One independent image: its own mip chain (and its own GL texture object
/// in the reference). `imageCount > 1` occurs only for multi-image GIFs
/// (docs/format-tex.md §4, §9; corpus: always 1).
#[derive(Debug, Clone)]
pub struct TexImage<'a> {
    /// Mip levels, largest first, each successive level halving both dims
    /// (integer floor); the chain need not reach 1×1 (docs/format-tex.md §7).
    pub mipmaps: Vec<Mipmap<'a>>,
}

/// One animation frame (docs/format-tex.md §8). TEXS0002/3 store floats in
/// the interleaved order width1, width2, height2, height1; TEXS0001 stores
/// integer coords with the two middle fields unused (`width2`/`height2`
/// stay 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// Index of the *image* (of `imageCount`) holding this frame's atlas
    /// (§8).
    pub frame_number: u32,
    /// Display duration of this frame, seconds (§8, §8.1).
    pub frametime: f32,
    /// Frame origin X in the atlas, texels (§8).
    pub x: f32,
    /// Frame origin Y in the atlas, texels (§8).
    pub y: f32,
    /// Frame X-axis extent along atlas X, texels (§8).
    pub width1: f32,
    /// Frame X-axis extent along atlas Y; ≠0 for rotated frames (§8).
    pub width2: f32,
    /// Frame Y-axis extent along atlas X; ≠0 for rotated frames (§8).
    pub height2: f32,
    /// Frame Y-axis extent along atlas Y, texels (§8).
    pub height1: f32,
}

/// Parsed `TEXS` animation block (docs/format-tex.md §8). Present iff
/// `flags & IsGif`.
#[derive(Debug, Clone)]
pub struct Animation {
    /// On-disk TEXS version (§8).
    pub version: AnimationVersion,
    /// Logical frame width. Stored on disk for TEXS0003; back-filled from
    /// frame 0's `width1` for TEXS0001/2 (0 when there are no frames)
    /// (§8, `TextureParser.cpp:269-273`).
    pub gif_width: u32,
    /// Logical frame height; same rules as [`Animation::gif_width`] (§8).
    pub gif_height: u32,
    /// Frame records in file order — playback walks this list linearly
    /// subtracting frametimes (§8.1).
    pub frames: Vec<Frame>,
}

/// A decoded mip level: tightly packed RGBA8, `pixels.len() == 4·w·h`.
#[derive(Debug, Clone)]
pub struct Rgba8Image {
    /// Width in texels. Raw path: the mip record's width; FreeImage path:
    /// the embedded decoder's width, which overrides the record fields
    /// (docs/format-tex.md §7.2).
    pub width: u32,
    /// Height in texels (same source rules as `width`).
    pub height: u32,
    /// R,G,B,A byte quadruplets, row-major, no padding.
    pub pixels: Vec<u8>,
}

/// A parsed `.tex` texture borrowing the input bytes (zero-copy: mip
/// payloads are lazy slices). See docs/format-tex.md.
#[derive(Debug, Clone)]
pub struct Tex<'a> {
    /// Pixel/block format of raw payloads (header +0x12, docs/format-tex.md
    /// §3, §5). Ignored by the reference when
    /// [`fif`](Self::fif) ≠ −1 (§5: FIF payloads decode to RGBA8).
    pub format: TextureFormat,
    /// Flags bitfield (header +0x16, §3, §6.1).
    pub flags: TextureFlags,
    /// Width of the stored mip-0 payload for raw formats (header +0x1a,
    /// §3). Not necessarily a power of two; unreliable for FreeImage-path
    /// textures (§3).
    pub texture_width: u32,
    /// Height of the stored mip-0 payload for raw formats (header +0x1e,
    /// §3).
    pub texture_height: u32,
    /// Real (usable/crop) image width (header +0x22, §3).
    pub width: u32,
    /// Real (usable/crop) image height (header +0x26, §3).
    pub height: u32,
    /// Header word at +0x2a, read and discarded by the reference. Corpus
    /// looks like a dominant/average `0xAARRGGBB` color (0 in videos);
    /// semantics UNVERIFIED (§3).
    pub unknown: u32,
    /// On-disk `TEXB` container version (§4). The *effective* version after
    /// the TEXB0004 downgrade rule is [`Tex::effective_container`].
    pub container: ContainerVersion,
    /// FreeImage codec id, after the §4 alias step: a TEXB0004 container
    /// with `freeImageFormat == −1` and `isVideoMp4 == 1` reads back as
    /// [`FreeImageFormat::MP4`]. −1 for TEXB0001/2 which store no such
    /// field (§4).
    pub fif: FreeImageFormat,
    /// TEXB0004 `isVideoMp4` word == 1 (§4). Always false in the corpus —
    /// real workshop MP4s use the `Video` header flag instead (§4, §7.3).
    pub is_video_mp4: bool,
    /// `imageCount` images, each with its own mip chain (§4, §7).
    pub images: Vec<TexImage<'a>>,
    /// Animation block, present iff `flags & IsGif` (§8).
    pub animation: Option<Animation>,
}

impl<'a> Tex<'a> {
    /// Parse a `.tex` file from a byte slice (container layer only; mip
    /// payloads stay as lazy slices). Layout per docs/format-tex.md §1:
    /// `TEXV0005`/`TEXI0001` magics + header (§3), `TEXB000N` container
    /// (§4), images/mipmaps (§7), optional `TEXS000N` animation block (§8).
    pub fn parse(data: &'a [u8]) -> Result<Self, TexError> {
        let mut r = Reader { data, pos: 0 };

        // §3: only TEXV0005 + TEXI0001 are accepted; anything else is a
        // hard error (TextureParser.cpp:185-193). Magics are 9 bytes
        // including the NUL (§2).
        r.expect_magic(b"TEXV0005\0", "outer container", "TEXV0005")?;
        r.expect_magic(b"TEXI0001\0", "header sub-block", "TEXI0001")?;

        // §3 header: 7 × u32 at +0x12..+0x2e.
        let format = TextureFormat::from_u32(r.read_u32("format")?)?;
        let flags = TextureFlags(r.read_u32("flags")?);
        let texture_width = r.read_u32("textureWidth")?;
        let texture_height = r.read_u32("textureHeight")?;
        let width = r.read_u32("width")?;
        let height = r.read_u32("height")?;
        let unknown = r.read_u32("header word +0x2a")?;

        // §4 container block at +0x2e.
        let magic = r.take(9, "TEXB magic")?;
        let container = match magic {
            b"TEXB0001\0" => ContainerVersion::Texb0001,
            b"TEXB0002\0" => ContainerVersion::Texb0002,
            b"TEXB0003\0" => ContainerVersion::Texb0003,
            b"TEXB0004\0" => ContainerVersion::Texb0004,
            other => {
                // §4: unknown container magic is a hard error
                // (TextureParser.cpp:233-235).
                return Err(TexError::BadMagic {
                    what: "image container",
                    expected: "TEXB0001..TEXB0004",
                    found: String::from_utf8_lossy(other).into_owned(),
                });
            }
        };

        // §4: imageCount is read *before* the version branch
        // (TextureParser.cpp:211).
        let image_count = r.read_u32("imageCount")?;

        // §4: freeImageFormat exists in TEXB0003/0004, isVideoMp4 in
        // TEXB0004 only; TEXB0001/2 leave fif at −1.
        let mut fif = FreeImageFormat::UNKNOWN;
        if matches!(container, ContainerVersion::Texb0003 | ContainerVersion::Texb0004) {
            fif = FreeImageFormat::from_i32(r.read_i32("freeImageFormat")?)?;
        }
        let mut is_video_mp4 = false;
        if container == ContainerVersion::Texb0004 {
            is_video_mp4 = r.read_u32("isVideoMp4")? == 1;
            // §4 downgrade rule step 1: FIF_UNKNOWN + isVideoMp4 → FIF_MP4
            // (TextureParser.cpp:213-225).
            if fif.is_raw() && is_video_mp4 {
                fif = FreeImageFormat::MP4;
            }
        }
        // §4 downgrade rule step 2: a TEXB0004 whose resulting fif is not
        // MP4 is treated as TEXB0003 from here on (changes the mip layout,
        // §7). All 89 corpus TEXB0004 files take this path.
        let effective = if container == ContainerVersion::Texb0004 && !fif.is_mp4() {
            ContainerVersion::Texb0003
        } else {
            container
        };

        // §7: imageCount × { mipmapCount: u32; mipmap[mipmapCount] }.
        // Preallocation is capped by what could physically fit so a hostile
        // count cannot force a huge allocation (SPEC.md §V9); each image
        // needs ≥ 4 bytes, each mip ≥ 12.
        let remaining = data.len().saturating_sub(r.pos);
        let mut images = Vec::with_capacity((image_count as usize).min(remaining / 4));
        for _ in 0..image_count {
            let mip_count = r.read_u32("mipmapCount")?;
            let remaining = data.len().saturating_sub(r.pos);
            let mut mipmaps = Vec::with_capacity((mip_count as usize).min(remaining / 12));
            for _ in 0..mip_count {
                mipmaps.push(parse_mipmap(&mut r, effective)?);
            }
            images.push(TexImage { mipmaps });
        }

        // §8: animation block present iff flags & IsGif, immediately after
        // the last mip payload of the last image.
        let animation = if flags.is_gif() {
            Some(parse_animation(&mut r)?)
        } else {
            None
        };

        Ok(Self {
            format,
            flags,
            texture_width,
            texture_height,
            width,
            height,
            unknown,
            container,
            fif,
            is_video_mp4,
            images,
            animation,
        })
    }

    /// The container version that governed the per-mip layout: TEXB0004
    /// downgrades to TEXB0003 unless its (post-alias) fif is MP4
    /// (docs/format-tex.md §4).
    #[must_use]
    pub fn effective_container(&self) -> ContainerVersion {
        if self.container == ContainerVersion::Texb0004 && !self.fif.is_mp4() {
            ContainerVersion::Texb0003
        } else {
            self.container
        }
    }

    /// Video detection exactly as the reference renderer:
    /// `isVideoMp4 || (flags & Video)` (docs/format-tex.md §7.3,
    /// `CTexture.cpp:20`). Real workshop MP4s set only the flag (§4).
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.is_video_mp4 || self.flags.video()
    }

    /// The whole video file stored verbatim as image 0 / mip 0
    /// (docs/format-tex.md §7.3), returned as opaque bytes — videos are
    /// *not* decoded here. Errors with [`TexError::NotVideo`] on non-video
    /// textures.
    pub fn video_payload(&self) -> Result<Cow<'a, [u8]>, TexError> {
        if !self.is_video() {
            return Err(TexError::NotVideo);
        }
        let image = self
            .images
            .first()
            .ok_or(TexError::NoSuchImage { index: 0, count: 0 })?;
        let mip = image
            .mipmaps
            .first()
            .ok_or(TexError::NoSuchMipmap { index: 0, count: 0 })?;
        mip.data()
    }

    /// Decode one mip level to tightly packed RGBA8.
    ///
    /// * Video textures (including the fif-35 MP4 alias, docs/format-tex.md
    ///   §4) error with [`TexError::IsVideo`]; use [`Tex::video_payload`].
    /// * FreeImage path (`fif != −1`, §7.2): the payload is a complete
    ///   encoded image file; `format` is ignored (§5) and the decoder's own
    ///   dims override the mip record fields.
    /// * Raw path (§7.1): the payload (after optional LZ4 expansion, §7)
    ///   must match the format's size rule exactly, then is converted per
    ///   the reference's GL upload semantics (§5).
    pub fn decode_rgba8(&self, image_index: usize, mip_index: usize) -> Result<Rgba8Image, TexError> {
        if self.is_video() || self.fif.is_mp4() {
            return Err(TexError::IsVideo);
        }
        let image = self.images.get(image_index).ok_or(TexError::NoSuchImage {
            index: image_index,
            count: self.images.len(),
        })?;
        let mip = image.mipmaps.get(mip_index).ok_or(TexError::NoSuchMipmap {
            index: mip_index,
            count: image.mipmaps.len(),
        })?;
        let data = mip.data()?;

        if !self.fif.is_raw() {
            // §7.2: each mip payload is a whole encoded image file; the
            // reference decodes with stb_image at desired_channels=4 and
            // takes dims from the decoder, not the mip record
            // (CTexture.cpp:63-69,84-86).
            let decoded = image::load_from_memory(&data)
                .map_err(|source| TexError::ImageDecode {
                    source: Box::new(source),
                })?
                .to_rgba8();
            let (width, height) = (decoded.width(), decoded.height());
            return Ok(Rgba8Image {
                width,
                height,
                pixels: decoded.into_raw(),
            });
        }

        decode_raw_rgba8(self.format, mip.width, mip.height, &data)
    }
}

// ---- raw-format RGBA8 decoding (docs/format-tex.md §5, §7.1) --------------

/// Decode an IEEE-754 binary16 (half) to `f32`. Total and panic-free: every
/// bit pattern maps to a finite/inf/NaN `f32` (docs/format-tex.md §5 float
/// formats; SPEC.md §V9). Subnormals and specials are handled exactly.
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    match exp {
        // Subnormal / zero: value = mant · 2^-24 (= 2^-14 / 2^10).
        0 => sign * mant * 2.0f32.powi(-24),
        // Inf / NaN.
        0x1f => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        // Normal: (1 + mant/1024) · 2^(exp-15).
        _ => sign * (1.0 + mant / 1024.0) * 2.0f32.powi(exp as i32 - 15),
    }
}

/// Tone-map a half-float channel to an 8-bit UNORM by clamping to `[0, 1]`
/// (HDR values > 1 saturate; NaN → 0). Float texture formats have no fixed
/// LDR mapping in the reference (it uploads them as `GL_*16F` and samples in
/// shader, docs/format-tex.md §5 / §13), so decode-to-RGBA8 picks the
/// conventional clamp-and-scale (SPEC.md §T11). Panic-free: `as u8`
/// saturates and maps NaN → 0 (SPEC.md §V9).
fn half_to_unorm8(h: u16) -> u8 {
    let v = half_to_f32(h);
    let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    (c * 255.0 + 0.5) as u8
}

/// Number of pixels as `usize` with overflow checking (SPEC.md §V9).
fn pixel_count(width: u32, height: u32) -> Result<usize, TexError> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(TexError::Oversized { width, height })
}

/// Number of 4×4 blocks covering `width`×`height` texels (BC standard;
/// docs/format-tex.md §7.1 `ceil(w/4)·ceil(h/4)`).
fn block_count(width: u32, height: u32) -> Result<usize, TexError> {
    (width as usize)
        .div_ceil(4)
        .checked_mul((height as usize).div_ceil(4))
        .ok_or(TexError::Oversized { width, height })
}

/// Expected raw payload byte length per format (docs/format-tex.md §7.1;
/// verified with 0 mismatches over all raw corpus mips — DXT1/DXT3 use the
/// BC-standard rule stated in the §5/§7.1 tables, UNVERIFIED in corpus).
fn expected_payload_len(format: TextureFormat, width: u32, height: u32) -> Result<usize, TexError> {
    let px = pixel_count(width, height)?;
    let len = match format {
        // §7.1: ARGB8888 = w·h·4.
        TextureFormat::Argb8888 => px.checked_mul(4),
        // §5: RGB888 = w·h·3, one byte per R,G,B channel (SPEC.md §T11;
        // UNVERIFIED — no corpus sample, standard 24-bit packing).
        TextureFormat::Rgb888 => px.checked_mul(3),
        // §5: RGB565 = w·h·2, one packed little-endian u16 per texel
        // (SPEC.md §T11; UNVERIFIED — standard 16-bit packing).
        TextureFormat::Rgb565 => px.checked_mul(2),
        // §7.1: RG88 = w·h·2.
        TextureFormat::Rg88 => px.checked_mul(2),
        // §7.1: R8 = w·h, byte-packed rows, width need not be 4-aligned.
        TextureFormat::R8 => Some(px),
        // §5: RG1616f = w·h·4, two IEEE half-floats (R,G) per texel
        // (SPEC.md §T11; UNVERIFIED — standard 2×f16 packing).
        TextureFormat::Rg1616f => px.checked_mul(4),
        // §5: R16f = w·h·2, one IEEE half-float per texel (SPEC.md §T11;
        // UNVERIFIED — standard f16 packing).
        TextureFormat::R16f => px.checked_mul(2),
        // §5: RGBA1010102 = w·h·4, one packed little-endian u32 per texel
        // (SPEC.md §T11; UNVERIFIED — standard DXGI R10G10B10A2 packing).
        TextureFormat::Rgba1010102 => px.checked_mul(4),
        // §5: RGBA16161616f = w·h·8, four IEEE half-floats (R,G,B,A)
        // (SPEC.md §T11; UNVERIFIED — standard 4×f16 packing).
        TextureFormat::Rgba16161616f => px.checked_mul(8),
        // §5: RGB161616f = w·h·6, three IEEE half-floats (R,G,B)
        // (SPEC.md §T11; UNVERIFIED — standard 3×f16 packing).
        TextureFormat::Rgb161616f => px.checked_mul(6),
        // §7.1: DXT5/DXT3 = ceil(w/4)·ceil(h/4)·16.
        TextureFormat::Dxt5 | TextureFormat::Dxt3 => block_count(width, height)?.checked_mul(16),
        // §5: BC7 = ceil(w/4)·ceil(h/4)·16 (SPEC.md §T11; UNVERIFIED —
        // standard BC7 block size).
        TextureFormat::Bc7 => block_count(width, height)?.checked_mul(16),
        // §7.1: DXT1 = ceil(w/4)·ceil(h/4)·8 (BC standard, UNVERIFIED).
        TextureFormat::Dxt1 => block_count(width, height)?.checked_mul(8),
        // §5: UNKNOWN (0xFFFFFFFF) is accepted by the parser but has no
        // payload semantics at all — decode refuses (SPEC.md §V10).
        TextureFormat::Unknown => return Err(TexError::UnsupportedFormat { format }),
    };
    len.ok_or(TexError::Oversized { width, height })
}

/// Convert a raw-path mip payload to RGBA8 (docs/format-tex.md §5, §7.1).
fn decode_raw_rgba8(
    format: TextureFormat,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<Rgba8Image, TexError> {
    let expected = expected_payload_len(format, width, height)?;
    if data.len() != expected {
        // §7.1: the size rules held exactly for every raw corpus mip.
        return Err(TexError::WrongPayloadSize {
            format,
            width,
            height,
            expected,
            actual: data.len(),
        });
    }
    let px = pixel_count(width, height)?;
    let out_len = px.checked_mul(4).ok_or(TexError::Oversized { width, height })?;

    let pixels = match format {
        // §5: ARGB8888 bytes are already R,G,B,A per texel, tightly packed
        // (uploaded directly as GL_RGBA/GL_UNSIGNED_BYTE).
        TextureFormat::Argb8888 => data.to_vec(),
        // §5: RG88 uploads as GL_RG8 — sampled as (R, G, 0, 1).
        TextureFormat::Rg88 => {
            let mut out = Vec::with_capacity(out_len);
            for &[r, g] in data.as_chunks::<2>().0 {
                out.extend_from_slice(&[r, g, 0, 255]);
            }
            out
        }
        // §5: R8 uploads as GL_R8/GL_RED — sampled as (R, 0, 0, 1); rows
        // are byte-packed with no padding (unpack alignment 1).
        TextureFormat::R8 => {
            let mut out = Vec::with_capacity(out_len);
            for &r in data {
                out.extend_from_slice(&[r, 0, 0, 255]);
            }
            out
        }
        // §5: RGB888 bytes are R,G,B per texel (same channel order as
        // ARGB8888's real RGBA byte layout); opaque A. Size validated to
        // w·h·3 above (SPEC.md §T11).
        TextureFormat::Rgb888 => {
            let mut out = Vec::with_capacity(out_len);
            for &[r, g, b] in data.as_chunks::<3>().0 {
                out.extend_from_slice(&[r, g, b, 255]);
            }
            out
        }
        // §5: RGB565 packs one little-endian u16 per texel, bits
        // R[15:11] G[10:5] B[4:0]; each channel bit-replicated up to 8 bits
        // (the standard exact-endpoint expansion). Opaque A (SPEC.md §T11).
        TextureFormat::Rgb565 => {
            let mut out = Vec::with_capacity(out_len);
            for &[b0, b1] in data.as_chunks::<2>().0 {
                let v = u16::from_le_bytes([b0, b1]);
                let r5 = ((v >> 11) & 0x1f) as u8;
                let g6 = ((v >> 5) & 0x3f) as u8;
                let b5 = (v & 0x1f) as u8;
                let r = (r5 << 3) | (r5 >> 2);
                let g = (g6 << 2) | (g6 >> 4);
                let b = (b5 << 3) | (b5 >> 2);
                out.extend_from_slice(&[r, g, b, 255]);
            }
            out
        }
        // §5: RG1616f — two half-floats (R,G) per texel; tone-mapped to
        // 8-bit by clamping to [0,1] (SPEC.md §T11). Sampled as (R,G,0,1).
        TextureFormat::Rg1616f => {
            let mut out = Vec::with_capacity(out_len);
            for &[b0, b1, b2, b3] in data.as_chunks::<4>().0 {
                let r = half_to_unorm8(u16::from_le_bytes([b0, b1]));
                let g = half_to_unorm8(u16::from_le_bytes([b2, b3]));
                out.extend_from_slice(&[r, g, 0, 255]);
            }
            out
        }
        // §5: R16f — one half-float per texel; sampled as (R,0,0,1)
        // (SPEC.md §T11).
        TextureFormat::R16f => {
            let mut out = Vec::with_capacity(out_len);
            for &[b0, b1] in data.as_chunks::<2>().0 {
                let r = half_to_unorm8(u16::from_le_bytes([b0, b1]));
                out.extend_from_slice(&[r, 0, 0, 255]);
            }
            out
        }
        // §5: RGBA1010102 packs one little-endian u32 per texel in DXGI
        // R10G10B10A2 order: R[9:0] G[19:10] B[29:20] A[31:30]. 10-bit
        // channels take the top 8 bits; the 2-bit alpha bit-replicates to
        // 8 bits (0,85,170,255) (SPEC.md §T11).
        TextureFormat::Rgba1010102 => {
            let mut out = Vec::with_capacity(out_len);
            for &[b0, b1, b2, b3] in data.as_chunks::<4>().0 {
                let v = u32::from_le_bytes([b0, b1, b2, b3]);
                let r = ((v >> 2) & 0xff) as u8;
                let g = ((v >> 12) & 0xff) as u8;
                let b = ((v >> 22) & 0xff) as u8;
                let a2 = ((v >> 30) & 0x3) as u8;
                let a = a2 * 0x55; // 2-bit → 8-bit replication
                out.extend_from_slice(&[r, g, b, a]);
            }
            out
        }
        // §5: RGBA16161616f — four half-floats (R,G,B,A) per texel; all
        // channels tone-mapped to 8-bit by clamping to [0,1] (SPEC.md §T11).
        TextureFormat::Rgba16161616f => {
            let mut out = Vec::with_capacity(out_len);
            for &[b0, b1, b2, b3, b4, b5, b6, b7] in data.as_chunks::<8>().0 {
                let r = half_to_unorm8(u16::from_le_bytes([b0, b1]));
                let g = half_to_unorm8(u16::from_le_bytes([b2, b3]));
                let b = half_to_unorm8(u16::from_le_bytes([b4, b5]));
                let a = half_to_unorm8(u16::from_le_bytes([b6, b7]));
                out.extend_from_slice(&[r, g, b, a]);
            }
            out
        }
        // §5: RGB161616f — three half-floats (R,G,B) per texel; opaque A;
        // channels tone-mapped to 8-bit by clamping to [0,1] (SPEC.md §T11).
        TextureFormat::Rgb161616f => {
            let mut out = Vec::with_capacity(out_len);
            for &[b0, b1, b2, b3, b4, b5] in data.as_chunks::<6>().0 {
                let r = half_to_unorm8(u16::from_le_bytes([b0, b1]));
                let g = half_to_unorm8(u16::from_le_bytes([b2, b3]));
                let b = half_to_unorm8(u16::from_le_bytes([b4, b5]));
                out.extend_from_slice(&[r, g, b, 255]);
            }
            out
        }
        // §5: DXTn / BC7 4×4 blocks; mip dims are true texel dims (§7.1).
        TextureFormat::Dxt1 | TextureFormat::Dxt3 | TextureFormat::Dxt5 | TextureFormat::Bc7 => {
            let mut words = vec![0u32; px];
            let (w, h) = (width as usize, height as usize);
            let result = match format {
                TextureFormat::Dxt1 => texture2ddecoder::decode_bc1(data, w, h, &mut words),
                TextureFormat::Dxt3 => texture2ddecoder::decode_bc2(data, w, h, &mut words),
                TextureFormat::Dxt5 => texture2ddecoder::decode_bc3(data, w, h, &mut words),
                // §5: BC7 (SPEC.md §T11; UNVERIFIED — no corpus sample).
                _ => texture2ddecoder::decode_bc7(data, w, h, &mut words),
            };
            result.map_err(|reason| TexError::BlockDecode { format, reason })?;
            // texture2ddecoder packs colors as u32::from_le_bytes([b,g,r,a]).
            let mut out = Vec::with_capacity(out_len);
            for word in words {
                let [b, g, r, a] = word.to_le_bytes();
                out.extend_from_slice(&[r, g, b, a]);
            }
            out
        }
        // Unreachable: expected_payload_len already rejected UNKNOWN (§5,
        // no payload semantics).
        other => return Err(TexError::UnsupportedFormat { format: other }),
    };

    Ok(Rgba8Image {
        width,
        height,
        pixels,
    })
}

// ---- parse internals -------------------------------------------------------

/// Parse one mip record per the effective container version
/// (docs/format-tex.md §7 table + rules).
fn parse_mipmap<'a>(r: &mut Reader<'a>, effective: ContainerVersion) -> Result<Mipmap<'a>, TexError> {
    if effective == ContainerVersion::Texb0004 {
        // §7: effective-TEXB0004 (mp4-flagged) mips carry an ignored prefix
        // of 2 × u32 + JSON cstr + u32 (TextureParser.cpp:43-54) —
        // UNVERIFIED against real bytes (no corpus sample).
        r.read_u32("mip editor int 1")?;
        r.read_u32("mip editor int 2")?;
        r.read_cstr("mip JSON string")?;
        r.read_u32("mip editor int 3")?;
    }

    // §7: width/height are present in all versions.
    let width = r.read_u32("mip width")?;
    let height = r.read_u32("mip height")?;

    // §7: compression + uncompressedSize exist in TEXB0002/3/4; absent in
    // TEXB0001 → treated as compression 0 (UNVERIFIED, none in corpus).
    let (compression_word, uncompressed_field) = if effective == ContainerVersion::Texb0001 {
        (0u32, 0i32)
    } else {
        (
            r.read_u32("mip compression")?,
            r.read_i32("mip uncompressedSize")?,
        )
    };

    // §7: compressedSize is present in all versions and is the stored
    // payload byte length (i32 via nextInt, §2).
    let compressed_size = r.read_i32("mip compressedSize")?;
    let compressed_size = usize::try_from(compressed_size).map_err(|_| TexError::NegativeSize {
        what: "compressedSize",
        value: compressed_size,
    })?;

    let (compression, uncompressed_size) = match compression_word {
        // §7 rule 1: compression == 0 → uncompressedSize := compressedSize;
        // the on-disk uncompressedSize word is ignored.
        0 => (Compression::Stored, compressed_size),
        // §7 rule 2: one raw LZ4 block expanding to exactly
        // uncompressedSize bytes.
        1 => {
            let size = usize::try_from(uncompressed_field).map_err(|_| TexError::NegativeSize {
                what: "uncompressedSize",
                value: uncompressed_field,
            })?;
            (Compression::Lz4, size)
        }
        // §7 rule 3: never validated by the reference (which would desync);
        // rejected here — see TexError::UnsupportedCompression.
        other => return Err(TexError::UnsupportedCompression { value: other }),
    };

    let payload = r.take(compressed_size, "mip payload")?;
    Ok(Mipmap {
        width,
        height,
        compression,
        uncompressed_size,
        payload,
    })
}

/// Parse the `TEXS000N` animation block (docs/format-tex.md §8).
fn parse_animation(r: &mut Reader<'_>) -> Result<Animation, TexError> {
    let magic = r.take(9, "TEXS magic")?;
    let version = match magic {
        b"TEXS0001\0" => AnimationVersion::Texs0001,
        b"TEXS0002\0" => AnimationVersion::Texs0002,
        b"TEXS0003\0" => AnimationVersion::Texs0003,
        other => {
            // §8: other magics are a hard error.
            return Err(TexError::BadMagic {
                what: "animation block",
                expected: "TEXS0001..TEXS0003",
                found: String::from_utf8_lossy(other).into_owned(),
            });
        }
    };

    let frame_count = r.read_u32("frameCount")?;

    // §8: gifWidth/gifHeight are stored for TEXS0003 only.
    let (mut gif_width, mut gif_height) = (0u32, 0u32);
    if version == AnimationVersion::Texs0003 {
        gif_width = r.read_u32("gifWidth")?;
        gif_height = r.read_u32("gifHeight")?;
    }

    // §8: 32 bytes per frame; preallocation capped by remaining bytes
    // (SPEC.md §V9).
    let remaining = r.data.len().saturating_sub(r.pos);
    let mut frames = Vec::with_capacity((frame_count as usize).min(remaining / 32));
    for _ in 0..frame_count {
        frames.push(match version {
            AnimationVersion::Texs0001 => parse_frame_v1(r)?,
            _ => parse_frame(r)?,
        });
    }

    // §8: for TEXS0001/0002 the gif dims are back-filled from frame 0's
    // width1/height1 after parsing (TextureParser.cpp:269-273). The f32→u32
    // `as` casts saturate (NaN → 0), so this cannot panic (SPEC.md §V9).
    if version != AnimationVersion::Texs0003
        && let Some(frame0) = frames.first()
    {
        gif_width = frame0.width1 as u32;
        gif_height = frame0.height1 as u32;
    }

    Ok(Animation {
        version,
        gif_width,
        gif_height,
        frames,
    })
}

/// Parse a TEXS0002/TEXS0003 frame record: 32 bytes, floats in the
/// interleaved order width1, width2, height2, height1 (docs/format-tex.md
/// §8, `TextureParser.cpp:110-123`).
fn parse_frame(r: &mut Reader<'_>) -> Result<Frame, TexError> {
    Ok(Frame {
        frame_number: r.read_u32("frame frameNumber")?,
        frametime: r.read_f32("frame frametime")?,
        x: r.read_f32("frame x")?,
        y: r.read_f32("frame y")?,
        width1: r.read_f32("frame width1")?,
        width2: r.read_f32("frame width2")?,
        height2: r.read_f32("frame height2")?,
        height1: r.read_f32("frame height1")?,
    })
}

/// Parse a TEXS0001 frame record: 32 bytes, integer coords converted to
/// float, the two middle fields ignored, `width2`/`height2` left at 0
/// (docs/format-tex.md §8, `TextureParser.cpp:95-108`). UNVERIFIED (no
/// corpus sample).
fn parse_frame_v1(r: &mut Reader<'_>) -> Result<Frame, TexError> {
    let frame_number = r.read_u32("frame frameNumber")?;
    let frametime = r.read_f32("frame frametime")?;
    let x = r.read_u32("frame x")?;
    let y = r.read_u32("frame y")?;
    let width1 = r.read_u32("frame width1")?;
    r.read_u32("frame unused field 5")?;
    r.read_u32("frame unused field 6")?;
    let height1 = r.read_u32("frame height1")?;
    // u32 → f32 is lossy above 2^24 but total and panic-free; the reference
    // performs the same integer-to-float conversion on load (§8).
    Ok(Frame {
        frame_number,
        frametime,
        x: x as f32,
        y: y as f32,
        width1: width1 as f32,
        width2: 0.0,
        height2: 0.0,
        height1: height1 as f32,
    })
}

/// Bounds-checked little-endian cursor (docs/format-tex.md §2: all
/// multi-byte integers little-endian; f32 read as raw 4 bytes).
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Consume exactly `len` bytes or fail with [`TexError::Truncated`].
    fn take(&mut self, len: usize, what: &'static str) -> Result<&'a [u8], TexError> {
        let truncated = || TexError::Truncated {
            what,
            offset: self.pos,
            needed: len,
            available: self.data.len().saturating_sub(self.pos),
        };
        let end = self.pos.checked_add(len).ok_or_else(truncated)?;
        let bytes = self.data.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(bytes)
    }

    /// Read a 9-byte magic (8 ASCII chars + NUL, docs/format-tex.md §2) and
    /// require an exact match.
    fn expect_magic(
        &mut self,
        magic: &'static [u8; 9],
        what: &'static str,
        expected: &'static str,
    ) -> Result<(), TexError> {
        let bytes = self.take(9, what)?;
        if bytes == magic {
            Ok(())
        } else {
            Err(TexError::BadMagic {
                what,
                expected,
                found: String::from_utf8_lossy(bytes).into_owned(),
            })
        }
    }

    /// Read a `u32`, little-endian unconditionally (docs/format-tex.md §2).
    fn read_u32(&mut self, what: &'static str) -> Result<u32, TexError> {
        Ok(u32::from_le_bytes(self.read_4(what)?))
    }

    /// Read an `i32` (the reference reads sizes via `nextInt`,
    /// docs/format-tex.md §2).
    fn read_i32(&mut self, what: &'static str) -> Result<i32, TexError> {
        Ok(i32::from_le_bytes(self.read_4(what)?))
    }

    /// Read an IEEE-754 f32 stored as raw little-endian bytes
    /// (docs/format-tex.md §2).
    fn read_f32(&mut self, what: &'static str) -> Result<f32, TexError> {
        Ok(f32::from_le_bytes(self.read_4(what)?))
    }

    fn read_4(&mut self, what: &'static str) -> Result<[u8; 4], TexError> {
        let offset = self.pos;
        let bytes = self.take(4, what)?;
        match bytes.first_chunk::<4>() {
            Some(arr) => Ok(*arr),
            // `take` returned exactly 4 bytes, so this arm is unreachable;
            // kept as a typed error instead of a panic path (SPEC.md §V9).
            None => Err(TexError::Truncated {
                what,
                offset,
                needed: 4,
                available: bytes.len(),
            }),
        }
    }

    /// Read a NUL-terminated byte string (docs/format-tex.md §2 `cstr`,
    /// `BinaryReader.cpp:45-53`); the NUL is consumed but not returned.
    fn read_cstr(&mut self, what: &'static str) -> Result<&'a [u8], TexError> {
        let start = self.pos;
        let rest = self.data.get(start..).unwrap_or(&[]);
        match rest.iter().position(|&b| b == 0) {
            Some(n) => {
                let bytes = rest.get(..n).unwrap_or(&[]);
                // start + n + 1 is within data.len() by construction.
                self.pos = start.saturating_add(n).saturating_add(1);
                Ok(bytes)
            }
            None => Err(TexError::Truncated {
                what,
                offset: start,
                needed: rest.len().saturating_add(1),
                available: rest.len(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    // ---- synthetic texture builders (docs/format-tex.md §3, §4, §7) ------

    /// TEXV0005 + TEXI0001 + 7 × u32 header (docs/format-tex.md §3).
    fn header(format: u32, flags: u32, tex_w: u32, tex_h: u32, w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"TEXV0005\0");
        v.extend_from_slice(b"TEXI0001\0");
        for word in [format, flags, tex_w, tex_h, w, h, 0xFF00_0000] {
            v.extend_from_slice(&word.to_le_bytes());
        }
        v
    }

    /// TEXB0002/3-layout mip record (docs/format-tex.md §7).
    fn mip_v3(w: u32, h: u32, compression: u32, uncompressed: i32, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&w.to_le_bytes());
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&compression.to_le_bytes());
        v.extend_from_slice(&uncompressed.to_le_bytes());
        v.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        v.extend_from_slice(payload);
        v
    }

    /// Minimal single-image TEXB0003 texture with one stored raw mip.
    fn simple_tex(format: u32, flags: u32, w: u32, h: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = header(format, flags, w, h, w, h);
        v.extend_from_slice(b"TEXB0003\0");
        v.extend_from_slice(&1u32.to_le_bytes()); // imageCount
        v.extend_from_slice(&(-1i32).to_le_bytes()); // fif = raw
        v.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount
        v.extend_from_slice(&mip_v3(w, h, 0, 0, payload));
        v
    }

    // ---- synthetic parse tests -------------------------------------------

    #[test]
    fn parses_minimal_argb_texture() {
        let payload: Vec<u8> = (0..16).collect(); // 2×2 RGBA
        let data = simple_tex(0, 2, 2, 2, &payload);
        let tex = Tex::parse(&data).unwrap();
        assert_eq!(tex.format, TextureFormat::Argb8888);
        assert_eq!(tex.flags.0, 2);
        assert!(tex.flags.clamp_uvs());
        assert!(!tex.flags.is_gif());
        assert_eq!((tex.texture_width, tex.texture_height), (2, 2));
        assert_eq!((tex.width, tex.height), (2, 2));
        assert_eq!(tex.unknown, 0xFF00_0000);
        assert_eq!(tex.container, ContainerVersion::Texb0003);
        assert_eq!(tex.effective_container(), ContainerVersion::Texb0003);
        assert!(tex.fif.is_raw());
        assert!(!tex.is_video());
        assert!(tex.animation.is_none());
        assert_eq!(tex.images.len(), 1);
        let mip = &tex.images[0].mipmaps[0];
        assert_eq!((mip.width, mip.height), (2, 2));
        assert_eq!(mip.compression, Compression::Stored);
        assert_eq!(mip.uncompressed_size, 16);
        assert_eq!(mip.payload, &payload[..]);
        assert_eq!(mip.data().unwrap().as_ref(), &payload[..]);

        // §5: ARGB8888 payload bytes are already RGBA order.
        let img = tex.decode_rgba8(0, 0).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(img.pixels, payload);
    }

    #[test]
    fn lz4_mip_roundtrips() {
        // §7 rule 2: one raw LZ4 block with exact uncompressedSize.
        let raw: Vec<u8> = std::iter::repeat_n([1u8, 2, 3, 4], 4).flatten().collect();
        let compressed = lz4_flex::block::compress(&raw);
        let mut data = header(0, 0, 2, 2, 2, 2);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(2, 2, 1, raw.len() as i32, &compressed));

        let tex = Tex::parse(&data).unwrap();
        let mip = &tex.images[0].mipmaps[0];
        assert_eq!(mip.compression, Compression::Lz4);
        assert_eq!(mip.uncompressed_size, 16);
        assert_eq!(mip.data().unwrap().as_ref(), &raw[..]);
        assert_eq!(tex.decode_rgba8(0, 0).unwrap().pixels, raw);
    }

    #[test]
    fn rejects_bad_outer_and_inner_magic() {
        // §3: only TEXV0005 and TEXI0001 are accepted.
        let mut data = simple_tex(0, 0, 1, 1, &[0; 4]);
        data[0..9].copy_from_slice(b"TEXV0004\0");
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::BadMagic {
                what: "outer container",
                ..
            })
        ));

        let mut data = simple_tex(0, 0, 1, 1, &[0; 4]);
        data[9..18].copy_from_slice(b"TEXI0002\0");
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::BadMagic {
                what: "header sub-block",
                ..
            })
        ));
    }

    #[test]
    fn rejects_unknown_formats() {
        // §5: values 3 and 5 are undefined; everything outside the table
        // is a hard error.
        for bad in [3u32, 5, 16, 0xFFFF_FFFE] {
            let data = simple_tex(bad, 0, 1, 1, &[0; 4]);
            assert!(
                matches!(Tex::parse(&data), Err(TexError::UnknownFormat { value }) if value == bad),
                "format {bad} must be rejected"
            );
        }
        // All defined values parse (payload sized for the 1×1 rules where
        // decodable; parse itself never checks payload size).
        for good in [0u32, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0xFFFF_FFFF] {
            let data = simple_tex(good, 0, 1, 1, &[0; 4]);
            assert!(Tex::parse(&data).is_ok(), "format {good} must parse");
        }
    }

    #[test]
    fn rejects_unknown_container_magic() {
        // §4: unknown TEXB magic is a hard error.
        let mut data = header(0, 0, 1, 1, 1, 1);
        data.extend_from_slice(b"TEXB0005\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::BadMagic {
                what: "image container",
                ..
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_fif() {
        // §6.2: gapless accepted range is −1..=36.
        for bad in [-2i32, 37, i32::MIN, i32::MAX] {
            let mut data = header(0, 0, 1, 1, 1, 1);
            data.extend_from_slice(b"TEXB0003\0");
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&bad.to_le_bytes());
            assert!(
                matches!(Tex::parse(&data), Err(TexError::UnknownFif { value }) if value == bad),
                "fif {bad} must be rejected"
            );
        }
    }

    #[test]
    fn never_panics_and_errors_on_any_truncation() {
        // SPEC.md §V9: every prefix of a valid file must produce a typed
        // error, never a panic.
        let payload: Vec<u8> = (0..16).collect();
        let data = simple_tex(0, 0, 2, 2, &payload);
        for len in 0..data.len() {
            let prefix = data.get(..len).unwrap();
            assert!(Tex::parse(prefix).is_err(), "prefix of {len} bytes must fail");
        }
        assert!(Tex::parse(&data).is_ok());
    }

    #[test]
    fn truncated_mip_payload_is_a_typed_error() {
        // §7: compressedSize bytes of payload must be present.
        let mut data = header(0, 0, 4, 4, 4, 4);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // width
        data.extend_from_slice(&4u32.to_le_bytes()); // height
        data.extend_from_slice(&0u32.to_le_bytes()); // compression
        data.extend_from_slice(&0i32.to_le_bytes()); // uncompressedSize
        data.extend_from_slice(&100i32.to_le_bytes()); // compressedSize
        data.extend_from_slice(&[0u8; 10]); // ...but only 10 bytes present
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::Truncated {
                what: "mip payload",
                needed: 100,
                available: 10,
                ..
            })
        ));
    }

    #[test]
    fn negative_mip_sizes_are_typed_errors() {
        // §7: sizes are i32 on the wire; negative values are nonsense.
        let mut data = header(0, 0, 1, 1, 1, 1);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(1, 1, 0, 0, &[]));
        let n = data.len();
        data[n - 4..].copy_from_slice(&(-1i32).to_le_bytes()); // compressedSize = -1
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::NegativeSize {
                what: "compressedSize",
                value: -1
            })
        ));

        let mut data = header(0, 0, 1, 1, 1, 1);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(1, 1, 1, -5, &[0; 4]));
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::NegativeSize {
                what: "uncompressedSize",
                value: -5
            })
        ));
    }

    #[test]
    fn unsupported_compression_is_a_typed_error() {
        // §7 rule 3: compression ≥ 2 would desync the reference; we refuse.
        let mut data = header(0, 0, 1, 1, 1, 1);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(1, 1, 2, 4, &[0; 4]));
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::UnsupportedCompression { value: 2 })
        ));
    }

    #[test]
    fn lz4_size_mismatch_and_corruption_are_typed_errors() {
        let raw = [7u8; 64];
        let compressed = lz4_flex::block::compress(&raw);

        // Header claims 128 uncompressed bytes but the block expands to 64
        // (§7 rule 2: result must be exactly uncompressedSize).
        let mut data = header(0, 0, 8, 8, 8, 8);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(8, 8, 1, 128, &compressed));
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.images[0].mipmaps[0].data(),
            Err(TexError::Lz4SizeMismatch {
                expected: 128,
                actual: 64
            })
        ));

        // Header claims fewer bytes than the block expands to → decoder
        // error surfaces as TexError::Lz4.
        let mut data = header(0, 0, 8, 8, 8, 8);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(8, 8, 1, 32, &compressed));
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.images[0].mipmaps[0].data(),
            Err(TexError::Lz4 { .. })
        ));

        // Implausible declared size (over the 255× expansion bound) is
        // rejected before allocating (SPEC.md §V9).
        let mut data = header(0, 0, 8, 8, 8, 8);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(-1i32).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(8, 8, 1, i32::MAX, &[0u8; 8]));
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.images[0].mipmaps[0].data(),
            Err(TexError::Lz4SizeMismatch { .. })
        ));
    }

    #[test]
    fn texb0004_without_mp4_downgrades_to_texb0003_layout() {
        // §4 downgrade rule: fif == −1 and isVideoMp4 == 0 → TEXB0003 mip
        // layout (the corpus-universal case, 89/89).
        let payload: Vec<u8> = (0..16).collect();
        let mut data = header(0, 0, 2, 2, 2, 2);
        data.extend_from_slice(b"TEXB0004\0");
        data.extend_from_slice(&1u32.to_le_bytes()); // imageCount
        data.extend_from_slice(&(-1i32).to_le_bytes()); // fif
        data.extend_from_slice(&0u32.to_le_bytes()); // isVideoMp4 = 0
        data.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount
        data.extend_from_slice(&mip_v3(2, 2, 0, 0, &payload));

        let tex = Tex::parse(&data).unwrap();
        assert_eq!(tex.container, ContainerVersion::Texb0004);
        assert_eq!(tex.effective_container(), ContainerVersion::Texb0003);
        assert!(tex.fif.is_raw());
        assert!(!tex.is_video());
        assert_eq!(tex.images[0].mipmaps[0].payload, &payload[..]);
    }

    #[test]
    fn texb0004_mp4_uses_v4_mip_layout_and_is_video() {
        // §4: fif == −1 + isVideoMp4 == 1 → fif = FIF_MP4 (35), effective
        // TEXB0004 → §7 v4 mip prefix (2 × u32 + JSON cstr + u32).
        let mp4 = b"\x00\x00\x00\x20ftypisom-fake-video-bytes";
        let mut data = header(0, 0, 2, 2, 2, 2);
        data.extend_from_slice(b"TEXB0004\0");
        data.extend_from_slice(&1u32.to_le_bytes()); // imageCount
        data.extend_from_slice(&(-1i32).to_le_bytes()); // fif
        data.extend_from_slice(&1u32.to_le_bytes()); // isVideoMp4 = 1
        data.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount
        data.extend_from_slice(&0u32.to_le_bytes()); // v4 editor int 1
        data.extend_from_slice(&0u32.to_le_bytes()); // v4 editor int 2
        data.extend_from_slice(b"{}\0"); // v4 JSON cstr
        data.extend_from_slice(&0u32.to_le_bytes()); // v4 editor int 3
        data.extend_from_slice(&mip_v3(2, 2, 0, 0, mp4));

        let tex = Tex::parse(&data).unwrap();
        assert_eq!(tex.container, ContainerVersion::Texb0004);
        assert_eq!(tex.effective_container(), ContainerVersion::Texb0004);
        assert!(tex.fif.is_mp4());
        assert!(tex.is_video_mp4);
        assert!(tex.is_video());
        // §7.3: video payload is opaque bytes; RGBA decode refuses.
        assert_eq!(tex.video_payload().unwrap().as_ref(), &mp4[..]);
        assert!(matches!(tex.decode_rgba8(0, 0), Err(TexError::IsVideo)));
    }

    #[test]
    fn video_flag_marks_texture_as_video() {
        // §7.3 + §4 checklist 4: real workshop MP4s carry only header flag
        // 0x20 and go through the TEXB0003 layout.
        let mp4 = b"\x00\x00\x00\x20ftypisomvideo";
        let data = simple_tex(0, 0x22, 2, 2, mp4); // Video|ClampUVs = 34
        let tex = Tex::parse(&data).unwrap();
        assert!(tex.flags.video());
        assert!(!tex.is_video_mp4);
        assert!(tex.is_video());
        assert_eq!(tex.video_payload().unwrap().as_ref(), &mp4[..]);
        assert!(matches!(tex.decode_rgba8(0, 0), Err(TexError::IsVideo)));

        // Non-video textures refuse video_payload (§7.3).
        let data = simple_tex(0, 2, 1, 1, &[0; 4]);
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(tex.video_payload(), Err(TexError::NotVideo)));
    }

    #[test]
    fn texb0001_mip_layout_has_no_compression_fields() {
        // §7 table: TEXB0001 mips are width/height/compressedSize only,
        // treated as stored (UNVERIFIED; none in corpus).
        let payload = [9u8; 4];
        let mut data = header(0, 0, 1, 1, 1, 1);
        data.extend_from_slice(b"TEXB0001\0");
        data.extend_from_slice(&1u32.to_le_bytes()); // imageCount
        data.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount
        data.extend_from_slice(&1u32.to_le_bytes()); // width
        data.extend_from_slice(&1u32.to_le_bytes()); // height
        data.extend_from_slice(&4i32.to_le_bytes()); // compressedSize
        data.extend_from_slice(&payload);

        let tex = Tex::parse(&data).unwrap();
        assert_eq!(tex.container, ContainerVersion::Texb0001);
        assert!(tex.fif.is_raw()); // §4: TEXB0001/2 store no fif
        let mip = &tex.images[0].mipmaps[0];
        assert_eq!(mip.compression, Compression::Stored);
        assert_eq!(mip.data().unwrap().as_ref(), &payload[..]);
    }

    // ---- animation block tests (docs/format-tex.md §8) --------------------

    /// A raw 2×2 texture with the IsGif flag and an appended TEXS block.
    fn gif_tex(texs: &[u8]) -> Vec<u8> {
        let mut data = simple_tex(0, TextureFlags::IS_GIF, 2, 2, &[0u8; 16]);
        data.extend_from_slice(texs);
        data
    }

    /// Serialize a TEXS0002/3 frame record in on-disk field order:
    /// frameNumber, frametime, x, y, width1, width2, height2, height1 (§8).
    fn frame_v23(frame: &Frame) -> Vec<u8> {
        let mut v = frame.frame_number.to_le_bytes().to_vec();
        for f in [
            frame.frametime,
            frame.x,
            frame.y,
            frame.width1,
            frame.width2,
            frame.height2,
            frame.height1,
        ] {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    }

    /// Shorthand for an unrotated frame (`width2`/`height2` = 0).
    fn plain_frame(frame_number: u32, frametime: f32, x: f32, y: f32, w: f32, h: f32) -> Frame {
        Frame {
            frame_number,
            frametime,
            x,
            y,
            width1: w,
            width2: 0.0,
            height2: 0.0,
            height1: h,
        }
    }

    #[test]
    fn parses_texs0003_animation_block() {
        // §8: TEXS0003 stores gifWidth/gifHeight before the frames.
        let mut texs = b"TEXS0003\0".to_vec();
        texs.extend_from_slice(&2u32.to_le_bytes()); // frameCount
        texs.extend_from_slice(&201u32.to_le_bytes()); // gifWidth
        texs.extend_from_slice(&201u32.to_le_bytes()); // gifHeight
        texs.extend_from_slice(&frame_v23(&plain_frame(0, 0.5, 0.0, 0.0, 201.0, 201.0)));
        texs.extend_from_slice(&frame_v23(&plain_frame(0, 0.5, 201.0, 0.0, 201.0, 201.0)));

        let data = gif_tex(&texs);
        let tex = Tex::parse(&data).unwrap();
        let anim = tex.animation.as_ref().unwrap();
        assert_eq!(anim.version, AnimationVersion::Texs0003);
        assert_eq!((anim.gif_width, anim.gif_height), (201, 201));
        assert_eq!(anim.frames.len(), 2);
        // §8: interleaved field order width1, width2, height2, height1.
        assert_eq!(anim.frames[1], plain_frame(0, 0.5, 201.0, 0.0, 201.0, 201.0));
    }

    #[test]
    fn texs0002_backfills_gif_dims_from_frame_zero() {
        // §8: TEXS0001/2 store no gif dims; back-filled from frame 0.
        let mut texs = b"TEXS0002\0".to_vec();
        texs.extend_from_slice(&1u32.to_le_bytes());
        texs.extend_from_slice(&frame_v23(&Frame {
            frame_number: 3,
            frametime: 0.1,
            x: 4.0,
            y: 8.0,
            width1: 64.0,
            width2: 1.0,
            height2: 2.0,
            height1: 32.0,
        }));
        let data = gif_tex(&texs);
        let tex = Tex::parse(&data).unwrap();
        let anim = tex.animation.as_ref().unwrap();
        assert_eq!(anim.version, AnimationVersion::Texs0002);
        assert_eq!((anim.gif_width, anim.gif_height), (64, 32));
        assert_eq!(anim.frames[0].frame_number, 3);
        assert_eq!(anim.frames[0].width2, 1.0);
        assert_eq!(anim.frames[0].height2, 2.0);
    }

    #[test]
    fn texs0001_frames_use_integer_coords_with_unused_middle_fields() {
        // §8: TEXS0001 records are u32 coords; fields 5–6 ignored;
        // width2/height2 stay 0.
        let mut texs = b"TEXS0001\0".to_vec();
        texs.extend_from_slice(&1u32.to_le_bytes()); // frameCount
        texs.extend_from_slice(&7u32.to_le_bytes()); // frameNumber
        texs.extend_from_slice(&0.25f32.to_le_bytes()); // frametime
        for word in [10u32, 20, 30, 999, 888, 40] {
            texs.extend_from_slice(&word.to_le_bytes());
        }
        let data = gif_tex(&texs);
        let tex = Tex::parse(&data).unwrap();
        let anim = tex.animation.as_ref().unwrap();
        assert_eq!(anim.version, AnimationVersion::Texs0001);
        assert_eq!(
            anim.frames[0],
            Frame {
                frame_number: 7,
                frametime: 0.25,
                x: 10.0,
                y: 20.0,
                width1: 30.0,
                width2: 0.0,
                height2: 0.0,
                height1: 40.0,
            }
        );
        assert_eq!((anim.gif_width, anim.gif_height), (30, 40));
    }

    #[test]
    fn rejects_bad_or_missing_texs_block() {
        // §8: other magics are a hard error.
        let mut texs = b"TEXS0004\0".to_vec();
        texs.extend_from_slice(&0u32.to_le_bytes());
        let data = gif_tex(&texs);
        assert!(matches!(
            Tex::parse(&data),
            Err(TexError::BadMagic {
                what: "animation block",
                ..
            })
        ));

        // IsGif flag set but no TEXS block at all → truncation.
        let data = simple_tex(0, TextureFlags::IS_GIF, 2, 2, &[0u8; 16]);
        assert!(matches!(Tex::parse(&data), Err(TexError::Truncated { .. })));
    }

    // ---- decode tests (docs/format-tex.md §5, §7.1, §7.2) -----------------

    #[test]
    fn expands_rg88_and_r8() {
        // §5: RG88 → GL_RG (R, G, 0, 1); R8 → GL_RED (R, 0, 0, 1).
        let data = simple_tex(8, 0, 1, 2, &[1, 2, 3, 4]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels, [1, 2, 0, 255, 3, 4, 0, 255]);

        let data = simple_tex(9, 0, 3, 1, &[10, 20, 30]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels, [10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255]);
    }

    #[test]
    fn decodes_dxt_blocks() {
        // §7.1: DXT1 = 8 bytes per 4×4 block. Both endpoint colors 0xFFFF
        // (white), all indices 0 → uniform opaque white.
        let dxt1_white = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        let data = simple_tex(7, 0, 4, 4, &dxt1_white);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 64);
        assert!(
            img.pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [255, 255, 255, 255])
        );

        // §7.1: DXT5 = 16 bytes per block: 8-byte interpolated-alpha block
        // (a0 = a1 = 255, indices 0) + the white BC1 color block.
        let mut dxt5_white = vec![0xFF, 0xFF, 0, 0, 0, 0, 0, 0];
        dxt5_white.extend_from_slice(&dxt1_white);
        let data = simple_tex(4, 0, 4, 4, &dxt5_white);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert!(
            img.pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [255, 255, 255, 255])
        );

        // §7.1: DXT3 = 16 bytes per block: 8 bytes of 4-bit explicit alpha
        // (all 0xF → 255) + the white BC1 color block.
        let mut dxt3_white = vec![0xFF; 8];
        dxt3_white.extend_from_slice(&dxt1_white);
        let data = simple_tex(6, 0, 4, 4, &dxt3_white);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert!(
            img.pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [255, 255, 255, 255])
        );
    }

    #[test]
    fn decodes_fif_png_payload() {
        // §7.2: each mip payload is a complete encoded image file; decoder
        // dims override the mip record fields.
        let mut png = Vec::new();
        let rgba = image::RgbaImage::from_fn(3, 2, |x, y| image::Rgba([x as u8 * 10, y as u8 * 10, 7, 255]));
        image::DynamicImage::ImageRgba8(rgba.clone())
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();

        let mut data = header(0, 0, 4, 4, 3, 2);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&13i32.to_le_bytes()); // fif = PNG
        data.extend_from_slice(&1u32.to_le_bytes());
        // §7.2: record dims deliberately wrong (9×9) — decoder wins.
        data.extend_from_slice(&mip_v3(9, 9, 0, 0, &png));

        let tex = Tex::parse(&data).unwrap();
        assert_eq!(tex.fif, FreeImageFormat::PNG);
        let img = tex.decode_rgba8(0, 0).unwrap();
        assert_eq!((img.width, img.height), (3, 2));
        assert_eq!(img.pixels, rgba.into_raw());

        // Garbage payload under a fif claim → typed decode error.
        let mut data = header(0, 0, 4, 4, 3, 2);
        data.extend_from_slice(b"TEXB0003\0");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&13i32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&mip_v3(1, 1, 0, 0, b"not a png"));
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.decode_rgba8(0, 0),
            Err(TexError::ImageDecode { .. })
        ));
    }

    #[test]
    fn decode_rejects_wrong_sizes_and_unsupported_formats() {
        // §7.1: payload length must match the size rule exactly.
        let data = simple_tex(0, 0, 2, 2, &[0u8; 15]);
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.decode_rgba8(0, 0),
            Err(TexError::WrongPayloadSize {
                expected: 16,
                actual: 15,
                ..
            })
        ));

        // §5/§V10: UNKNOWN (0xFFFFFFFF) has no payload semantics, so decode
        // refuses instead of guessing (the other formerly parser-accepted
        // formats now decode — see `decodes_all_added_raw_formats`).
        let data = simple_tex(0xFFFF_FFFF, 0, 1, 1, &[0u8; 4]);
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.decode_rgba8(0, 0),
            Err(TexError::UnsupportedFormat {
                format: TextureFormat::Unknown
            })
        ));

        // Out-of-range indices are typed errors.
        let data = simple_tex(0, 0, 1, 1, &[0u8; 4]);
        let tex = Tex::parse(&data).unwrap();
        assert!(matches!(
            tex.decode_rgba8(1, 0),
            Err(TexError::NoSuchImage { index: 1, count: 1 })
        ));
        assert!(matches!(
            tex.decode_rgba8(0, 9),
            Err(TexError::NoSuchMipmap { index: 9, count: 1 })
        ));
    }

    #[test]
    fn half_to_unorm8_handles_normals_subnormals_and_specials() {
        // IEEE binary16 → 8-bit UNORM clamp-and-scale (SPEC.md §T11, §V9).
        assert_eq!(half_to_unorm8(0x0000), 0); // +0.0
        assert_eq!(half_to_unorm8(0x8000), 0); // -0.0
        assert_eq!(half_to_unorm8(0x3C00), 255); // 1.0
        assert_eq!(half_to_unorm8(0x3800), 128); // 0.5
        assert_eq!(half_to_unorm8(0x3400), 64); // 0.25
        assert_eq!(half_to_unorm8(0x7C00), 255); // +inf → clamped to 1.0
        assert_eq!(half_to_unorm8(0xFC00), 0); // -inf → clamped to 0.0
        assert_eq!(half_to_unorm8(0x7E00), 0); // NaN → 0
        assert_eq!(half_to_unorm8(0xBC00), 0); // -1.0 → clamped to 0.0
        assert_eq!(half_to_unorm8(0x0001), 0); // smallest subnormal → 0
    }

    #[test]
    fn decodes_all_added_raw_formats() {
        // SPEC.md §T11: each newly-added raw format decodes to tightly
        // packed RGBA8 (len == 4·w·h) with the documented channel order and
        // normalization. Payloads carry a known pixel to pin the mapping.

        // §5: RGB888 (fmt 1) — 3 bytes/texel R,G,B, opaque A. 2×1 texels.
        let data = simple_tex(1, 0, 2, 1, &[10, 20, 30, 40, 50, 60]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 4 * 2);
        assert_eq!(img.pixels, [10, 20, 30, 255, 40, 50, 60, 255]);

        // §5: RGB565 (fmt 2) — packed u16, bit-replicated. 0xF800 = pure red.
        let data = simple_tex(2, 0, 1, 1, &[0x00, 0xF8]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels, [255, 0, 0, 255]);
        // 0x07E0 = pure green (6-bit 63 → 255); 0x001F = pure blue.
        let data = simple_tex(2, 0, 2, 1, &[0xE0, 0x07, 0x1F, 0x00]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels, [0, 255, 0, 255, 0, 0, 255, 255]);

        // §5: RG1616f (fmt 10) — two f16 (R,G); 1.0=0x3C00, 0.5=0x3800.
        let data = simple_tex(10, 0, 1, 1, &[0x00, 0x3C, 0x00, 0x38]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels, [255, 128, 0, 255]);

        // §5: R16f (fmt 11) — one f16; 1.0 → (255,0,0,255).
        let data = simple_tex(11, 0, 1, 1, &[0x00, 0x3C]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels, [255, 0, 0, 255]);

        // §5: BC7 (fmt 12) — a hand-built mode-6 solid-white block
        // (16 bytes / 4×4 block). All color+p bits set, all indices 0 →
        // every texel opaque white, robust to endpoint field ordering.
        let bc7_white = [
            0xC0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0, 0, 0, 0, 0, 0, 0,
        ];
        let data = simple_tex(12, 0, 4, 4, &bc7_white);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 4 * 16);
        assert!(
            img.pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [255, 255, 255, 255]),
            "BC7 mode-6 solid block must decode to opaque white"
        );

        // §5: RGBA1010102 (fmt 13) — DXGI R10G10B10A2 u32. R=1023,A=3 →
        // (255,0,0,255). 0xC00003FF little-endian.
        let data = simple_tex(13, 0, 1, 1, &[0xFF, 0x03, 0x00, 0xC0]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels, [255, 0, 0, 255]);
        // R=0,G=1023,B=0,A=0: v = 1023<<10 = 0x000FFC00.
        let data = simple_tex(13, 0, 1, 1, &[0x00, 0xFC, 0x0F, 0x00]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels, [0, 255, 0, 0]);

        // §5: RGBA16161616f (fmt 14) — four f16 (R,G,B,A):
        // R=1.0,G=0.5,B=0.0,A=1.0.
        let data = simple_tex(14, 0, 1, 1, &[0x00, 0x3C, 0x00, 0x38, 0x00, 0x00, 0x00, 0x3C]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels, [255, 128, 0, 255]);

        // §5: RGB161616f (fmt 15) — three f16 (R,G,B), opaque A:
        // R=1.0,G=0.5,B=0.25.
        let data = simple_tex(15, 0, 1, 1, &[0x00, 0x3C, 0x00, 0x38, 0x00, 0x34]);
        let img = Tex::parse(&data).unwrap().decode_rgba8(0, 0).unwrap();
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels, [255, 128, 64, 255]);
    }

    #[test]
    fn added_formats_reject_wrong_payload_sizes() {
        // §7.1: the size rule is enforced for the added formats too — a
        // short payload is a typed error, not a panic (SPEC.md §V9).
        for (value, w, h, good_len) in [
            (1u32, 2u32, 1u32, 6usize), // RGB888 = w·h·3
            (2, 2, 1, 4),               // RGB565 = w·h·2
            (10, 1, 1, 4),              // RG1616f = w·h·4
            (11, 1, 1, 2),              // R16f = w·h·2
            (12, 4, 4, 16),             // BC7 = 1 block · 16
            (13, 1, 1, 4),              // RGBA1010102 = w·h·4
            (14, 1, 1, 8),              // RGBA16161616f = w·h·8
            (15, 1, 1, 6),              // RGB161616f = w·h·6
        ] {
            let data = simple_tex(value, 0, w, h, &vec![0u8; good_len - 1]);
            let tex = Tex::parse(&data).unwrap();
            assert!(
                matches!(tex.decode_rgba8(0, 0), Err(TexError::WrongPayloadSize { .. })),
                "format {value} must reject a short payload"
            );
        }
    }

    #[test]
    fn flag_accessors_match_bit_values() {
        // §6.1 bit table.
        let f = TextureFlags(0x1 | 0x2 | 0x4 | 0x8 | 0x20 | 0x8_0000);
        assert!(f.no_interpolation());
        assert!(f.clamp_uvs());
        assert!(f.is_gif());
        assert!(f.clamp_uvs_border());
        assert!(f.video());
        assert!(f.alpha_channel_priority());
        let none = TextureFlags(0x10); // undefined bit: ignorable, no error
        assert!(!none.no_interpolation() && !none.video() && !none.is_gif());
    }

    // ---- corpus tests (skipped when the corpus is absent) ----------------

    /// Default corpus location (docs/format-tex.md §1); override with
    /// `KIRIE_CORPUS`.
    const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
    /// docs/format-tex.md §11: 190 embedded .tex files across 19 scene.pkg.
    const CORPUS_TEX_COUNT: usize = 190;
    /// docs/format-tex.md §7.3/§11: 3 video textures (flags & 0x20).
    const CORPUS_VIDEO_COUNT: usize = 3;

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

    fn corpus_scene_pkgs(dir: &Path) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|item| item.path().join("scene.pkg"))
            .filter(|p| p.is_file())
            .collect();
        paths.sort();
        paths
    }

    /// Visit every embedded `.tex` payload in every corpus scene.pkg via
    /// the pkg module (crate-internal use).
    fn for_each_corpus_tex(dir: &Path, mut visit: impl FnMut(&Path, &str, &[u8])) {
        for path in corpus_scene_pkgs(dir) {
            let pkg =
                crate::pkg::OwnedPkg::from_path(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            for entry in pkg.entries() {
                let Some(name) = entry.name_str() else { continue };
                if !name.ends_with(".tex") {
                    continue;
                }
                let payload = pkg
                    .read(&entry)
                    .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
                visit(&path, name, payload);
            }
        }
    }

    #[test]
    fn corpus_every_tex_parses_with_spec_distributions() {
        let Some(dir) = corpus_dir() else { return };

        let mut total = 0usize;
        let mut videos = 0usize;
        let mut containers: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut formats: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut fifs: BTreeMap<i32, usize> = BTreeMap::new();
        let mut flags: BTreeMap<u32, usize> = BTreeMap::new();
        let mut chain_lengths: BTreeMap<usize, usize> = BTreeMap::new();
        let mut animations = Vec::new();

        for_each_corpus_tex(&dir, |path, name, payload| {
            let tex = Tex::parse(payload).unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            total += 1;

            // §11: imageCount == 1 in all 190 files.
            assert_eq!(tex.images.len(), 1, "{name}: imageCount != 1");

            *containers
                .entry(match tex.container {
                    ContainerVersion::Texb0001 => "TEXB0001",
                    ContainerVersion::Texb0002 => "TEXB0002",
                    ContainerVersion::Texb0003 => "TEXB0003",
                    ContainerVersion::Texb0004 => "TEXB0004",
                })
                .or_default() += 1;
            // §11: all 89 TEXB0004 downgrade to the TEXB0003 layout.
            assert_eq!(tex.effective_container(), ContainerVersion::Texb0003, "{name}");
            assert!(!tex.is_video_mp4, "{name}: §4 — no corpus file sets isVideoMp4");

            *formats
                .entry(match tex.format {
                    TextureFormat::Argb8888 => "ARGB8888",
                    TextureFormat::R8 => "R8",
                    TextureFormat::Dxt5 => "DXT5",
                    TextureFormat::Rg88 => "RG88",
                    other => panic!("{name}: unexpected corpus format {other:?}"),
                })
                .or_default() += 1;
            *fifs.entry(tex.fif.0).or_default() += 1;
            *flags.entry(tex.flags.0).or_default() += 1;
            for image in &tex.images {
                *chain_lengths.entry(image.mipmaps.len()).or_default() += 1;
            }

            if tex.is_video() {
                videos += 1;
                // §7.3: whole MP4 file verbatim, `ftyp isom` signature.
                let bytes = tex.video_payload().unwrap();
                assert_eq!(bytes.get(4..12), Some(&b"ftypisom"[..]), "{name}");
            }

            // §8: animation block present iff flags & IsGif.
            assert_eq!(tex.animation.is_some(), tex.flags.is_gif(), "{name}");
            if let Some(anim) = tex.animation.clone() {
                animations.push(anim);
            }
        });

        assert_eq!(
            total, CORPUS_TEX_COUNT,
            "corpus .tex count changed vs docs/format-tex.md §11"
        );
        assert_eq!(videos, CORPUS_VIDEO_COUNT, "video texture count vs §7.3");

        // §11 corpus survey table.
        assert_eq!(containers, BTreeMap::from([("TEXB0003", 101), ("TEXB0004", 89)]));
        assert_eq!(
            formats,
            BTreeMap::from([("ARGB8888", 79), ("R8", 60), ("DXT5", 26), ("RG88", 25)])
        );
        assert_eq!(fifs, BTreeMap::from([(-1, 158), (13, 28), (2, 4)]));
        assert_eq!(
            flags,
            BTreeMap::from([(2, 171), (0, 11), (3, 4), (34, 3), (6, 1)])
        );
        assert_eq!(
            chain_lengths,
            BTreeMap::from([
                (1, 128),
                (2, 2),
                (3, 1),
                (4, 27),
                (5, 12),
                (6, 5),
                (8, 4),
                (9, 6),
                (11, 5)
            ])
        );

        // §8.1: exactly one animated texture — TEXS0003, 39 frames of
        // 1/39 s each (total exactly 1 s), all frameNumber 0, 201×201 grid
        // cells.
        assert_eq!(animations.len(), 1, "§11: exactly one TEXS block in corpus");
        let anim = animations.first().unwrap();
        assert_eq!(anim.version, AnimationVersion::Texs0003);
        assert_eq!((anim.gif_width, anim.gif_height), (201, 201));
        assert_eq!(anim.frames.len(), 39);
        let total_time: f32 = anim.frames.iter().map(|f| f.frametime).sum();
        assert!((total_time - 1.0).abs() < 1e-4, "total {total_time}");
        for frame in &anim.frames {
            assert_eq!(frame.frame_number, 0);
            assert!((frame.frametime - 1.0 / 39.0).abs() < 1e-6);
            assert_eq!((frame.width1, frame.height1), (201.0, 201.0));
            assert_eq!((frame.width2, frame.height2), (0.0, 0.0));
        }
    }

    #[test]
    fn corpus_top_mip_of_every_non_video_tex_decodes_to_rgba8() {
        let Some(dir) = corpus_dir() else { return };

        let mut decoded = 0usize;
        let mut skipped_videos = 0usize;
        for_each_corpus_tex(&dir, |path, name, payload| {
            let tex = Tex::parse(payload).unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            if tex.is_video() {
                skipped_videos += 1;
                return;
            }
            let img = tex
                .decode_rgba8(0, 0)
                .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            // Decoded byte length must be exactly 4·w·h.
            assert_eq!(
                img.pixels.len(),
                4 * img.width as usize * img.height as usize,
                "{name}: pixel byte length"
            );
            assert!(img.width > 0 && img.height > 0, "{name}: empty decode");
            let mip = &tex.images[0].mipmaps[0];
            if tex.fif.is_raw() {
                // §7.1: raw-path decode dims are the mip record dims, and
                // mip 0 matches the header payload dims.
                assert_eq!((img.width, img.height), (mip.width, mip.height), "{name}");
                assert_eq!(
                    (mip.width, mip.height),
                    (tex.texture_width, tex.texture_height),
                    "{name}: §7.1 mip-0 dims == textureWidth/Height"
                );
            } else {
                // §7.2: the embedded decoder's dims override the record's;
                // in the corpus both agree.
                assert_eq!((img.width, img.height), (mip.width, mip.height), "{name}");
            }
            decoded += 1;
        });

        assert_eq!(decoded + skipped_videos, CORPUS_TEX_COUNT);
        assert_eq!(skipped_videos, CORPUS_VIDEO_COUNT);
    }

    #[test]
    fn corpus_every_mip_of_every_non_video_tex_has_consistent_sizes() {
        let Some(dir) = corpus_dir() else { return };

        let mut raw_mips = 0usize;
        for_each_corpus_tex(&dir, |path, name, payload| {
            let tex = Tex::parse(payload).unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            if tex.is_video() || !tex.fif.is_raw() {
                return;
            }
            for image in &tex.images {
                for mip in &image.mipmaps {
                    // §7.1 size rules: uncompressed size must match the
                    // format formula for every level, not just mip 0.
                    let expected = expected_payload_len(tex.format, mip.width, mip.height)
                        .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
                    assert_eq!(
                        mip.uncompressed_size, expected,
                        "{name}: {}x{} {:?}",
                        mip.width, mip.height, tex.format
                    );
                    raw_mips += 1;
                }
            }
        });
        // §7.1: 135 ARGB + 25 RG88 + 60 R8 + 111 DXT5 raw mips checked.
        assert_eq!(raw_mips, 331, "raw corpus mip count vs docs/format-tex.md §7.1");
    }

    #[test]
    fn corpus_any_added_format_tex_decodes_non_uniform() {
        // SPEC.md §T11: for every corpus `.tex` whose raw format is one of
        // the newly-added decoders (RGB888/RGB565/RG1616f/R16f/BC7/
        // RGBA1010102/RGBA16161616f/RGB161616f), the top mip must decode to
        // 4·w·h bytes of non-uniform pixels (a real texture is never a
        // single flat color — this catches an all-white/all-black misdecode
        // like the white fallback these formats previously produced).
        //
        // Reality of the installed corpus (20 scene.pkg / 190 .tex): the
        // format distribution is exactly {ARGB8888, DXT5, RG88, R8}
        // (docs/format-tex.md §11) — no added format is present, so this
        // test currently exercises 0 textures. It is kept so the assertion
        // fires automatically if such a texture ever enters the corpus.
        // (The task's "fmt-13 cave.tex in 1388331347" is a misread: that
        // texture is format 0 / fif 13 — a PNG-backed FreeImage texture,
        // not a raw RGBA1010102 payload.)
        let Some(dir) = corpus_dir() else { return };

        let added = |f: TextureFormat| {
            matches!(
                f,
                TextureFormat::Rgb888
                    | TextureFormat::Rgb565
                    | TextureFormat::Rg1616f
                    | TextureFormat::R16f
                    | TextureFormat::Bc7
                    | TextureFormat::Rgba1010102
                    | TextureFormat::Rgba16161616f
                    | TextureFormat::Rgb161616f
            )
        };

        let mut checked = 0usize;
        for_each_corpus_tex(&dir, |path, name, payload| {
            let tex = Tex::parse(payload).unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            if tex.is_video() || !tex.fif.is_raw() || !added(tex.format) {
                return;
            }
            let img = tex
                .decode_rgba8(0, 0)
                .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            assert_eq!(
                img.pixels.len(),
                4 * img.width as usize * img.height as usize,
                "{name}: pixel byte length"
            );
            let first = img.pixels.as_chunks::<4>().0.first().copied();
            assert!(
                img.pixels.as_chunks::<4>().0.iter().any(|px| Some(*px) != first),
                "{name}: {:?} decoded to a uniform color — likely a misdecode",
                tex.format
            );
            checked += 1;
        });
        eprintln!("corpus_any_added_format_tex_decodes_non_uniform: checked {checked} texture(s)");
    }
}
