//! `.mdl` 3D-model / mesh container parser (Wallpaper Engine `MDLV` format).
//!
//! A `.mdl` file is a binary mesh container referenced by a scene object's
//! `mesh` field. Its geometry is what a [`crate::project`] model layer draws.
//! Unlike the JSON `.mdl` *wrapper* some tools emit, the binary payload stored
//! inside `scene.pkg` begins with a NUL-terminated `MDLV00NN` magic.
//!
//! # Byte layout (little-endian throughout)
//!
//! This reproduces the reference parser
//! `WallpaperEngine/Data/Parsers/ObjectParser.cpp::parseModel`
//! (`ObjectParser.cpp:210-295`) exactly:
//!
//! ```text
//! cstring  version        e.g. "MDLV0017\0" (must start with "MDLV")
//! i32      header0        "unknown (15)"  — a bitfield; 0x0000000F on 0017
//! i32      header1        "unknown (1)"
//! i32      meshCount
//! meshCount × Mesh:
//!     cstring  materialRef    e.g. "materials/models/space boi/diffuse_0.json\0"
//!     i32      _reserved      "unknown (0)"
//!     f32[6]   bbox           (minX,minY,minZ, maxX,maxY,maxZ)
//!     i32      flags
//!     i32      vertexBytes    > 0 and a multiple of 48 (the vertex stride)
//!     u8[vertexBytes]  vertexData
//!     i32      indexBytes     > 0 and a multiple of 2 (u16 indices)
//!     u16[indexBytes/2] indices
//! ```
//!
//! # Vertex layout
//!
//! Each vertex is a fixed 48-byte record, matching the attribute bindings the
//! reference renderer uses when uploading `vertexData` verbatim to the GPU
//! (`Render/Objects/CModel.cpp:22-27`):
//!
//! ```text
//! offset 0  : f32[3] position
//! offset 12 : f32[3] normal
//! offset 24 : f32[4] tangent   (xyz + handedness w)
//! offset 40 : f32[2] texcoord
//! ```
//!
//! Decode a mesh's records with [`Mesh::vertices`]; the raw bytes remain
//! available via [`Mesh::vertex_data`] for direct GPU upload.
//!
//! # Version handling (`MDLV0017` vs `MDLV0023`)
//!
//! The reference makes **no** structural distinction between versions: it
//! validates only the `MDLV` prefix (`ObjectParser.cpp:243`) and then reads
//! the layout above with a **fixed 48-byte stride** (`ObjectParser.cpp:258`).
//! This parser matches that behaviour exactly, and thus treats the corpus's
//! two versions identically:
//!
//! * `MDLV0017` (e.g. Starscape's `space boi.mdl`, the live wallpaper) is a
//!   plain 48-byte-vertex mesh; `header0 == 0x0F`, and every vertex block is a
//!   multiple of 48. Parses fully.
//! * `MDLV0023` ("puppet" models) carries a larger `header0` bitfield and, on
//!   inspection, an **80-byte** vertex stride (the 48-byte geometry record plus
//!   an extra 32 bytes of skinning data), followed by skinning/bone tables
//!   after the last mesh. Because the reference hard-codes 48, any `MDLV0023`
//!   whose `vertexBytes` is not also a multiple of 48 is *rejected* by the
//!   reference (it logs "Invalid vertex block" and stops) — so this parser
//!   returns [`ModelError::InvalidVertexBlock`] for it too, deliberately
//!   mirroring the reference rather than guessing the unverified skinning
//!   layout. (Only a puppet whose byte count happens to be a common multiple
//!   of 48 and 80 slips through, and then its 80-byte records are
//!   mis-split — a latent reference bug we reproduce, not fix.)
//!
//! We surface the raw `version`, `header0`, and `header1` so callers can
//! observe the variant. Any bytes after the last mesh are ignored, exactly as
//! the reference ignores them.
//!
//! # Error contract
//!
//! Per SPEC.md §V9 this parser never panics on malformed input: every read is
//! bounds-checked and every size is validated, yielding a typed [`ModelError`].
//! (The reference logs-and-continues with a partial mesh list; a library
//! returns a hard typed error instead, matching the crate's other parsers.)

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Size in bytes of one vertex record (`ObjectParser.cpp:258`,
/// `CModel.cpp:23`). Vertex blocks must be a whole multiple of this.
pub const VERTEX_STRIDE: usize = 48;

/// Byte offset of the `position` attribute within a vertex (`CModel.cpp:24`).
pub const POSITION_OFFSET: usize = 0;
/// Byte offset of the `normal` attribute within a vertex (`CModel.cpp:25`).
pub const NORMAL_OFFSET: usize = 12;
/// Byte offset of the `tangent` attribute within a vertex (`CModel.cpp:26`).
pub const TANGENT_OFFSET: usize = 24;
/// Byte offset of the `texcoord` attribute within a vertex (`CModel.cpp:27`).
pub const UV_OFFSET: usize = 40;

/// Errors produced by the `.mdl` parser (`ObjectParser.cpp:210-295`).
#[derive(Debug, Error)]
pub enum ModelError {
    /// The input ended before an expected field. The reference reads through a
    /// throwing cursor (`ObjectParser.cpp:222-224`); any header/cstring/block
    /// read hitting EOF is a hard error here.
    #[error(
        "truncated model: need {needed} byte(s) for {what} at offset {offset}, only {available} available"
    )]
    UnexpectedEof {
        /// Human-readable name of the field being read.
        what: &'static str,
        /// Stream offset at which the read began.
        offset: usize,
        /// Bytes the read required.
        needed: usize,
        /// Bytes actually remaining from `offset` to end of input.
        available: usize,
    },

    /// The version string did not start with `MDLV` (`ObjectParser.cpp:243`).
    #[error("unsupported model header {header:?} (expected \"MDLV\" prefix)")]
    BadMagic {
        /// The rejected version string, decoded lossily for display.
        header: String,
    },

    /// `meshCount` was negative — a corrupt header. The reference's
    /// `for i < meshCount` loop would simply skip; a library rejects it.
    #[error("invalid mesh count {count} (must be >= 0)")]
    InvalidMeshCount {
        /// The rejected count as read from the file.
        count: i32,
    },

    /// A vertex block was non-positive, not a multiple of [`VERTEX_STRIDE`],
    /// or ran past the end of the data (`ObjectParser.cpp:258-262`).
    #[error("mesh {index}: invalid vertex block: {vertex_bytes} byte(s) (must be > 0, a multiple of {stride}, and in bounds)", stride = VERTEX_STRIDE)]
    InvalidVertexBlock {
        /// Zero-based mesh index.
        index: usize,
        /// The rejected block size as read from the file.
        vertex_bytes: i32,
    },

    /// An index block was non-positive, odd (indices are `u16`), or ran past
    /// the end of the data (`ObjectParser.cpp:270-274`).
    #[error("mesh {index}: invalid index block: {index_bytes} byte(s) (must be > 0, even, and in bounds)")]
    InvalidIndexBlock {
        /// Zero-based mesh index.
        index: usize,
        /// The rejected block size as read from the file.
        index_bytes: i32,
    },
}

/// One decoded vertex record (48 bytes; see the module docs and
/// `CModel.cpp:22-27`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    /// Object-space position (`a_Position`).
    pub position: [f32; 3],
    /// Object-space normal (`a_Normal`).
    pub normal: [f32; 3],
    /// Tangent frame: `xyz` tangent direction, `w` handedness (`a_Tangent4`).
    pub tangent: [f32; 4],
    /// Texture coordinate (`a_TexCoord`).
    pub uv: [f32; 2],
}

/// One sub-mesh: a material reference plus a vertex block and a `u16` index
/// buffer (`ObjectParser.cpp:250-286`). Sub-meshes are drawn in declaration
/// order, which is load-bearing for depth (`CModel.cpp:108-114`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    /// Path of the material JSON this mesh is drawn with, e.g.
    /// `materials/models/space boi/diffuse_0.json` (`ObjectParser.cpp:251`).
    pub material_ref: String,
    /// Axis-aligned bounding-box minimum corner `(minX, minY, minZ)`
    /// (`ObjectParser.cpp:253`).
    pub bbox_min: [f32; 3],
    /// Axis-aligned bounding-box maximum corner `(maxX, maxY, maxZ)`
    /// (`ObjectParser.cpp:253`).
    pub bbox_max: [f32; 3],
    /// Per-mesh flags word (`ObjectParser.cpp:254`); meaning undocumented by
    /// the reference, preserved verbatim.
    pub flags: i32,
    /// Raw interleaved vertex bytes, [`VERTEX_STRIDE`] per vertex. Kept as raw
    /// bytes so callers can upload them straight to the GPU as the reference
    /// does (`CModel.cpp:138-140`). Decode with [`Mesh::vertices`].
    pub vertex_data: Vec<u8>,
    /// Triangle-list indices into this mesh's vertices
    /// (`ObjectParser.cpp:275-276`).
    pub indices: Vec<u16>,
}

impl Mesh {
    /// Number of vertices in this mesh (`vertex_data.len() / 48`).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertex_data.len() / VERTEX_STRIDE
    }

    /// Iterate the decoded [`Vertex`] records. Zero-allocation; decodes each
    /// 48-byte record on demand from [`Self::vertex_data`].
    pub fn vertices(&self) -> impl Iterator<Item = Vertex> + '_ {
        self.vertex_data
            .as_chunks::<VERTEX_STRIDE>()
            .0
            .iter()
            .map(decode_vertex)
    }

    /// The material path as raw bytes (always valid UTF-8 here since it was
    /// parsed from a Rust `String`, but exposed for symmetry with other
    /// parsers).
    #[must_use]
    pub fn material_ref(&self) -> &str {
        &self.material_ref
    }
}

/// A parsed `.mdl` model: header metadata plus its sub-meshes
/// (`ObjectParser.cpp:210-295`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// The full version string including the `MDLV` prefix, e.g. `MDLV0017`
    /// or `MDLV0023` (`ObjectParser.cpp:242`).
    pub version: String,
    /// First post-magic header word the reference reads and discards as
    /// "unknown (15)" (`ObjectParser.cpp:246`). A bitfield: `0x0000000F` on
    /// observed `MDLV0017`, a larger value on `MDLV0023`. Preserved so callers
    /// can distinguish variants.
    pub header0: i32,
    /// Second post-magic header word, read and discarded by the reference as
    /// "unknown (1)" (`ObjectParser.cpp:247`). Preserved verbatim.
    pub header1: i32,
    /// The sub-meshes, in declaration order (`ObjectParser.cpp:250`).
    pub meshes: Vec<Mesh>,
}

impl Model {
    /// Parse a binary `.mdl` payload (the bytes stored in `scene.pkg`, i.e. the
    /// `MDLV`-prefixed form, not the JSON wrapper).
    ///
    /// Mirrors `ObjectParser::parseModel` (`ObjectParser.cpp:210-295`): reads
    /// the magic, two header words, `meshCount`, then each mesh's material
    /// reference, bounding box, flags, vertex block, and index block. Trailing
    /// bytes after the last mesh (e.g. `MDLV0023` skinning tables) are ignored,
    /// exactly as the reference ignores them.
    ///
    /// # Errors
    ///
    /// Returns a typed [`ModelError`] on any malformed or truncated input;
    /// never panics (SPEC.md §V9).
    pub fn parse(data: &[u8]) -> Result<Self, ModelError> {
        let mut cur = Cursor::new(data);

        let version = cur.read_cstring("version")?;
        if !version.starts_with("MDLV") {
            return Err(ModelError::BadMagic { header: version });
        }

        let header0 = cur.read_i32("header0")?;
        let header1 = cur.read_i32("header1")?;
        let mesh_count = cur.read_i32("meshCount")?;
        if mesh_count < 0 {
            return Err(ModelError::InvalidMeshCount { count: mesh_count });
        }

        // Do NOT pre-allocate from the untrusted `mesh_count`: a corrupt header
        // with a huge count (e.g. 2e9) would make `Vec::with_capacity` abort the
        // process on the failed allocation — a §V9 panic on malformed input. The
        // reference never reserves; it push_backs in a loop bounded by real data
        // (`ObjectParser.cpp:250`), and each per-mesh read below is bounds-checked
        // and hits `UnexpectedEof` almost immediately for a truncated file.
        let mut meshes = Vec::new();
        for index in 0..mesh_count as usize {
            let material_ref = cur.read_cstring("materialRef")?;
            let _reserved = cur.read_i32("mesh reserved word")?;
            let bbox = cur.read_f32x6("bbox")?;
            let flags = cur.read_i32("flags")?;

            // Vertex block: > 0, a multiple of the 48-byte stride, and in
            // bounds (ObjectParser.cpp:258-262).
            let vertex_bytes = cur.read_i32("vertexBytes")?;
            if vertex_bytes <= 0
                || !(vertex_bytes as usize).is_multiple_of(VERTEX_STRIDE)
                || cur.remaining() < vertex_bytes as usize
            {
                return Err(ModelError::InvalidVertexBlock { index, vertex_bytes });
            }
            let vertex_data = cur.take(vertex_bytes as usize).to_vec();

            // Index block: > 0, even (u16 indices), and in bounds
            // (ObjectParser.cpp:270-274).
            let index_bytes = cur.read_i32("indexBytes")?;
            if index_bytes <= 0 || (index_bytes % 2) != 0 || cur.remaining() < index_bytes as usize {
                return Err(ModelError::InvalidIndexBlock { index, index_bytes });
            }
            let index_slice = cur.take(index_bytes as usize);
            let indices = index_slice
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[lo, hi]| u16::from_le_bytes([lo, hi]))
                .collect();

            meshes.push(Mesh {
                material_ref,
                bbox_min: [bbox[0], bbox[1], bbox[2]],
                bbox_max: [bbox[3], bbox[4], bbox[5]],
                flags,
                vertex_data,
                indices,
            });
        }

        Ok(Self {
            version,
            header0,
            header1,
            meshes,
        })
    }

    /// Total vertex count across all sub-meshes.
    #[must_use]
    pub fn total_vertices(&self) -> usize {
        self.meshes.iter().map(Mesh::vertex_count).sum()
    }

    /// Total index count across all sub-meshes.
    #[must_use]
    pub fn total_indices(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len()).sum()
    }
}

/// Decode one 48-byte vertex record.
fn decode_vertex(chunk: &[u8; VERTEX_STRIDE]) -> Vertex {
    let f = |off: usize| f32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]]);
    Vertex {
        position: [f(POSITION_OFFSET), f(POSITION_OFFSET + 4), f(POSITION_OFFSET + 8)],
        normal: [f(NORMAL_OFFSET), f(NORMAL_OFFSET + 4), f(NORMAL_OFFSET + 8)],
        tangent: [
            f(TANGENT_OFFSET),
            f(TANGENT_OFFSET + 4),
            f(TANGENT_OFFSET + 8),
            f(TANGENT_OFFSET + 12),
        ],
        uv: [f(UV_OFFSET), f(UV_OFFSET + 4)],
    }
}

/// Bounds-checked forward-only byte cursor. Every read either advances or
/// returns [`ModelError::UnexpectedEof`]; it never panics (SPEC.md §V9).
struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    /// Advance past and return the next `n` bytes. Callers must have verified
    /// `self.remaining() >= n` (used for the already-bounds-checked blocks).
    fn take(&mut self, n: usize) -> &'a [u8] {
        let start = self.offset;
        self.offset += n;
        &self.data[start..start + n]
    }

    fn need(&self, what: &'static str, n: usize) -> Result<(), ModelError> {
        if self.remaining() < n {
            return Err(ModelError::UnexpectedEof {
                what,
                offset: self.offset,
                needed: n,
                available: self.remaining(),
            });
        }
        Ok(())
    }

    fn read_i32(&mut self, what: &'static str) -> Result<i32, ModelError> {
        self.need(what, 4)?;
        let b = self.take(4);
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32x6(&mut self, what: &'static str) -> Result<[f32; 6], ModelError> {
        self.need(what, 24)?;
        let b = self.take(24);
        let mut out = [0.0f32; 6];
        for (i, slot) in out.iter_mut().enumerate() {
            let o = i * 4;
            *slot = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        }
        Ok(out)
    }

    /// Read a NUL-terminated string, consuming the terminator. Matches the
    /// reference `readCString` (`ObjectParser.cpp:230-240`): it scans to the
    /// first NUL or end of data. If no NUL is found before end of data the
    /// remaining bytes are returned and the cursor lands at the end — the
    /// subsequent field read then reports the truncation.
    fn read_cstring(&mut self, what: &'static str) -> Result<String, ModelError> {
        // At least one byte must remain to read a (possibly empty) string.
        self.need(what, 1)?;
        let start = self.offset;
        while self.offset < self.data.len() && self.data[self.offset] != 0 {
            self.offset += 1;
        }
        let bytes = &self.data[start..self.offset];
        let s = String::from_utf8_lossy(bytes).into_owned();
        if self.offset < self.data.len() {
            self.offset += 1; // skip the NUL terminator
        }
        Ok(s)
    }
}

// ===========================================================================
// Puppet-warp meshes (`MDLV0021` / `MDLV0023`, 80-byte skinned stride)
// ===========================================================================
//
// A model JSON's optional `puppet` field points at a *puppet* `.mdl`: the
// deformable mesh a WE "puppet warp" character (girl/guy/cat in scene
// 3428443753) is drawn with. Its container is unlike the plain `MDLV0017`
// meshes [`Model::parse`] reads:
//
// * The version marker is `MDLV0021` or `MDLV0023` (8 chars + NUL = 9 bytes;
//   `CImage.cpp:437-443`).
// * The usable vertex/index block is **not** at a fixed header offset. The
//   reference *scans* for it (`findPuppetMeshBlock`, `CImage.cpp:75-105`):
//   starting just past the marker and stopping at the `MDLS` marker (or EOF),
//   it looks for an offset whose `[+4]` `u32` `vertexBytes` is a positive
//   multiple of the 80-byte stride and is followed, after `vertexBytes` bytes,
//   by a `u32` `indexBytes` that is a positive multiple of `u16*3` (whole
//   triangles) — all in bounds. The first such offset wins.
// * A "mesh header" of two `u32`s precedes the vertices; the reference only
//   uses the second word (`vertexBytes`) and starts vertices at
//   `headerOffset + 8` (`CImage.cpp:466`). After the vertex block a single
//   `u32` `indexBytes` precedes the `u16` triangle indices.
//
// # 80-byte vertex stride
//
// The reference only reads `position` (offset 0) and `uv` (offset 72) — it
// treats the puppet as a static mesh and never applies the bones
// (`docs/render-architecture.md` §7.1: "Bones/animation are not implemented").
// The remaining 60 bytes are the geometry frame plus the per-vertex skinning
// the animation would use. Inspecting scene 3428443753's puppets (every vertex:
// normal `(0,0,1)`, tangent `(1,~0,0,1)`, bone indices `(4,0,0,0)` as `u32`,
// weights `(1,0,0,0)` as `f32`) fixes the full layout:
//
// ```text
// offset  0 : f32[3] position   (CImage.cpp:439 positionOffset = 0)
// offset 12 : f32[3] normal      (empirical — reference does not read it)
// offset 24 : f32[4] tangent     (empirical)
// offset 40 : u32[4] boneIndices (empirical — the 32-byte skinning half)
// offset 56 : f32[4] boneWeights (empirical)
// offset 72 : f32[2] uv          (CImage.cpp:440 uvOffset = 72)
// ```
//
// i.e. 48 bytes of geometry (pos+normal+tangent+uv) interleaved with 32 bytes
// of skinning (4 bone indices + 4 weights). This parser exposes the full
// [`PuppetVertex`] — positions/uv/indices as the reference reads them, plus the
// skinning the reference discards — so the renderer can apply the animation
// layers the reference cannot. Bytes after the index block (bones, the `MDLS`
// animation tables) are left for a later pass, exactly as the reference leaves
// them here.
//
// V9: every read is bounds-checked; malformed puppets yield a typed
// [`PuppetError`], never a panic.

/// Puppet vertex stride in bytes (`CImage.cpp:438` `vertexStride = 80`).
pub const PUPPET_VERTEX_STRIDE: usize = 80;
/// Bytes of the `MDLV00NN\0` marker skipped before the block scan
/// (`CImage.cpp:436` `markerSize = 9`).
const PUPPET_MARKER_SIZE: usize = 9;
/// Two `u32`s precede the vertex data; vertices start at `headerOffset + 8`
/// (`CImage.cpp:437` `meshHeaderSize = sizeof(uint32_t) * 2`).
const PUPPET_MESH_HEADER_SIZE: usize = 8;
/// Byte offset of `position` within a puppet vertex (`CImage.cpp:439`).
pub const PUPPET_POSITION_OFFSET: usize = 0;
/// Byte offset of `normal` (empirical; see module docs).
pub const PUPPET_NORMAL_OFFSET: usize = 12;
/// Byte offset of `tangent` (empirical).
pub const PUPPET_TANGENT_OFFSET: usize = 24;
/// Byte offset of the four `u32` bone indices (empirical; skinning half).
pub const PUPPET_BONE_INDEX_OFFSET: usize = 40;
/// Byte offset of the four `f32` bone weights (empirical).
pub const PUPPET_BONE_WEIGHT_OFFSET: usize = 56;
/// Byte offset of `uv` within a puppet vertex (`CImage.cpp:440` `uvOffset`).
pub const PUPPET_UV_OFFSET: usize = 72;

/// Errors produced by the puppet `.mdl` parser (`CImage.cpp:426-528`). Mirrors
/// the reference's failure points (each a `return false` there) as typed values;
/// never panics on malformed input (SPEC.md §V9).
#[derive(Debug, Error)]
pub enum PuppetError {
    /// The version marker was not `MDLV0021` or `MDLV0023` (`CImage.cpp:441`).
    #[error("unsupported puppet model header {header:?} (expected \"MDLV0021\" or \"MDLV0023\")")]
    BadMagic {
        /// The rejected marker, decoded lossily (empty when the file is < 9 bytes).
        header: String,
    },

    /// No offset before the `MDLS` marker held a valid vertex+index block
    /// (`findPuppetMeshBlock` returned `nullopt`, `CImage.cpp:461-465`).
    #[error("no usable puppet mesh block found before the MDLS marker")]
    NoMeshBlock,

    /// A triangle index referenced a vertex outside the mesh
    /// (`CImage.cpp:498-501`).
    #[error("invalid puppet mesh index {index} (>= vertex count {vertex_count})")]
    InvalidIndex {
        /// The out-of-range index value.
        index: u16,
        /// The mesh's vertex count.
        vertex_count: usize,
    },
}

/// One decoded puppet vertex (80 bytes; see the module docs). Carries the
/// geometry the reference reads (`position`, `uv`) plus the `normal`/`tangent`
/// frame and the per-vertex skinning (`bone_indices` + `bone_weights`) the
/// reference discards, so a renderer can apply the puppet animation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PuppetVertex {
    /// Puppet-space position (`CImage.cpp:481-484`, offset 0).
    pub position: [f32; 3],
    /// Vertex normal (empirical, offset 12).
    pub normal: [f32; 3],
    /// Tangent frame `xyz` + handedness `w` (empirical, offset 24).
    pub tangent: [f32; 4],
    /// Four bone indices this vertex is skinned to (empirical, offset 40).
    pub bone_indices: [u32; 4],
    /// Matching bone weights, summing to ~1 (empirical, offset 56).
    pub bone_weights: [f32; 4],
    /// Texture coordinate (`CImage.cpp:486-488`, offset 72).
    pub uv: [f32; 2],
}

/// A parsed puppet-warp mesh: the located vertex block and its `u16` triangle
/// index buffer (`CImage.cpp:466-503`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PuppetMesh {
    /// The version marker, `MDLV0021` or `MDLV0023` (`CImage.cpp:437`).
    pub version: String,
    /// Decoded vertices, in file order.
    pub vertices: Vec<PuppetVertex>,
    /// Triangle-list indices into [`Self::vertices`] (`CImage.cpp:496-503`).
    pub indices: Vec<u16>,
}

impl PuppetMesh {
    /// Parse a puppet `.mdl` payload (the bytes of the model's `puppet` file).
    ///
    /// Mirrors `CImage::loadPuppetMesh` + `findPuppetMeshBlock`
    /// (`CImage.cpp:75-105, 426-528`): validates the marker, finds the `MDLS`
    /// boundary, scans for the first valid vertex/index block, then decodes the
    /// 80-byte vertices and `u16` indices (rejecting any index past the vertex
    /// count). Skinning is decoded in addition to the reference's pos/uv.
    ///
    /// # Errors
    ///
    /// Returns a typed [`PuppetError`] on any malformed input; never panics
    /// (SPEC.md §V9).
    pub fn parse(data: &[u8]) -> Result<Self, PuppetError> {
        // Marker: 8 chars + NUL. `CImage.cpp:441` compares the first 8 bytes.
        let version = if data.len() >= PUPPET_MARKER_SIZE {
            String::from_utf8_lossy(&data[..8]).into_owned()
        } else {
            String::new()
        };
        if version != "MDLV0021" && version != "MDLV0023" {
            return Err(PuppetError::BadMagic { header: version });
        }

        // The block scan stops at the `MDLS` marker, or EOF if absent
        // (`CImage.cpp:448-457`).
        let mdls_offset = find_mdls(data);
        let block = find_puppet_mesh_block(data, mdls_offset).ok_or(PuppetError::NoMeshBlock)?;

        let vertex_count = block.vertex_bytes / PUPPET_VERTEX_STRIDE;
        let vertices_offset = block.header_offset + PUPPET_MESH_HEADER_SIZE;
        let indices_offset = vertices_offset + block.vertex_bytes + 4;
        let index_count = block.index_bytes / 2;

        // `find_puppet_mesh_block` guaranteed both blocks lie within
        // `mdls_offset <= data.len()`, so these reads are in bounds.
        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let base = vertices_offset + i * PUPPET_VERTEX_STRIDE;
            let rec = &data[base..base + PUPPET_VERTEX_STRIDE];
            vertices.push(decode_puppet_vertex(rec));
        }

        let mut indices = Vec::with_capacity(index_count);
        for i in 0..index_count {
            let o = indices_offset + i * 2;
            let index = u16::from_le_bytes([data[o], data[o + 1]]);
            if index as usize >= vertex_count {
                return Err(PuppetError::InvalidIndex { index, vertex_count });
            }
            indices.push(index);
        }

        Ok(Self {
            version,
            vertices,
            indices,
        })
    }

    /// Number of vertices in the mesh.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

/// Located puppet mesh block: the header offset and its two block sizes
/// (`struct PuppetMeshBlock`, `CImage.cpp:74-78`).
struct PuppetMeshBlock {
    header_offset: usize,
    vertex_bytes: usize,
    index_bytes: usize,
}

/// Offset of the first `MDLS` marker at or after the version marker, else the
/// end of `data` (`CImage.cpp:448-457`).
fn find_mdls(data: &[u8]) -> usize {
    let mut offset = PUPPET_MARKER_SIZE;
    while offset + 4 <= data.len() {
        if &data[offset..offset + 4] == b"MDLS" {
            return offset;
        }
        offset += 1;
    }
    data.len()
}

/// Scan for the first offset before `mdls_offset` whose `[+4]` `vertexBytes`
/// (multiple of the 80-byte stride) and trailing `indexBytes` (multiple of
/// `u16*3`) describe an in-bounds vertex+index block (`CImage.cpp:80-104`).
fn find_puppet_mesh_block(data: &[u8], mdls_offset: usize) -> Option<PuppetMeshBlock> {
    let read_u32 =
        |off: usize| -> u32 { u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) };
    let mut offset = PUPPET_MARKER_SIZE;
    // Loop bound guarantees the `[+4]` u32 read below is in bounds and inside
    // the MDLS region (`CImage.cpp:84`).
    while offset + PUPPET_MESH_HEADER_SIZE + 4 < mdls_offset {
        let vertex_bytes = read_u32(offset + 4) as usize;
        let vertices_offset = offset + PUPPET_MESH_HEADER_SIZE;
        let index_length_offset = vertices_offset + vertex_bytes;

        if vertex_bytes == 0
            || !vertex_bytes.is_multiple_of(PUPPET_VERTEX_STRIDE)
            || index_length_offset + 4 > mdls_offset
        {
            offset += 1;
            continue;
        }

        let index_bytes = read_u32(index_length_offset) as usize;
        let indices_offset = index_length_offset + 4;
        if index_bytes == 0
            || !index_bytes.is_multiple_of(2 * 3)
            || indices_offset + index_bytes > mdls_offset
        {
            offset += 1;
            continue;
        }

        return Some(PuppetMeshBlock {
            header_offset: offset,
            vertex_bytes,
            index_bytes,
        });
    }
    None
}

/// Decode one 80-byte puppet vertex record (see the module docs).
fn decode_puppet_vertex(chunk: &[u8]) -> PuppetVertex {
    let f = |off: usize| f32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]]);
    let u = |off: usize| u32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]]);
    PuppetVertex {
        position: [
            f(PUPPET_POSITION_OFFSET),
            f(PUPPET_POSITION_OFFSET + 4),
            f(PUPPET_POSITION_OFFSET + 8),
        ],
        normal: [
            f(PUPPET_NORMAL_OFFSET),
            f(PUPPET_NORMAL_OFFSET + 4),
            f(PUPPET_NORMAL_OFFSET + 8),
        ],
        tangent: [
            f(PUPPET_TANGENT_OFFSET),
            f(PUPPET_TANGENT_OFFSET + 4),
            f(PUPPET_TANGENT_OFFSET + 8),
            f(PUPPET_TANGENT_OFFSET + 12),
        ],
        bone_indices: [
            u(PUPPET_BONE_INDEX_OFFSET),
            u(PUPPET_BONE_INDEX_OFFSET + 4),
            u(PUPPET_BONE_INDEX_OFFSET + 8),
            u(PUPPET_BONE_INDEX_OFFSET + 12),
        ],
        bone_weights: [
            f(PUPPET_BONE_WEIGHT_OFFSET),
            f(PUPPET_BONE_WEIGHT_OFFSET + 4),
            f(PUPPET_BONE_WEIGHT_OFFSET + 8),
            f(PUPPET_BONE_WEIGHT_OFFSET + 12),
        ],
        uv: [f(PUPPET_UV_OFFSET), f(PUPPET_UV_OFFSET + 4)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal well-formed single-mesh MDLV payload for unit tests.
    fn synth_model(version: &str, verts: u32, tris: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(version.as_bytes());
        b.push(0);
        b.extend_from_slice(&15i32.to_le_bytes()); // header0
        b.extend_from_slice(&1i32.to_le_bytes()); // header1
        b.extend_from_slice(&1i32.to_le_bytes()); // meshCount
        // mesh 0
        b.extend_from_slice(b"materials/test.json");
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes()); // reserved
        for v in [-1.0f32, -2.0, -3.0, 4.0, 5.0, 6.0] {
            b.extend_from_slice(&v.to_le_bytes()); // bbox
        }
        b.extend_from_slice(&15i32.to_le_bytes()); // flags
        let vbytes = verts * VERTEX_STRIDE as u32;
        b.extend_from_slice(&(vbytes as i32).to_le_bytes());
        for i in 0..verts {
            // 12 f32 per vertex; encode the vertex index into position.x
            b.extend_from_slice(&(i as f32).to_le_bytes());
            for _ in 1..12 {
                b.extend_from_slice(&0.0f32.to_le_bytes());
            }
        }
        let ibytes = tris * 3 * 2;
        b.extend_from_slice(&(ibytes as i32).to_le_bytes());
        for t in 0..tris {
            for k in 0..3u16 {
                b.extend_from_slice(&((t as u16 + k) % verts as u16).to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn parses_synthetic_model() {
        let bytes = synth_model("MDLV0017", 4, 2);
        let m = Model::parse(&bytes).expect("parse");
        assert_eq!(m.version, "MDLV0017");
        assert_eq!(m.header0, 15);
        assert_eq!(m.header1, 1);
        assert_eq!(m.meshes.len(), 1);
        let mesh = &m.meshes[0];
        assert_eq!(mesh.material_ref, "materials/test.json");
        assert_eq!(mesh.bbox_min, [-1.0, -2.0, -3.0]);
        assert_eq!(mesh.bbox_max, [4.0, 5.0, 6.0]);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.indices.len(), 6);
        assert_eq!(m.total_vertices(), 4);
        assert_eq!(m.total_indices(), 6);
        // Vertex decode: position.x carries the vertex index we wrote.
        let positions: Vec<f32> = mesh.vertices().map(|v| v.position[0]).collect();
        assert_eq!(positions, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synth_model("MDLV0017", 1, 1);
        bytes[0..4].copy_from_slice(b"XXXX");
        match Model::parse(&bytes) {
            Err(ModelError::BadMagic { header }) => assert!(header.starts_with("XXXX")),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_is_typed_error() {
        let bytes = synth_model("MDLV0017", 1, 1);
        // Cut off inside the header words (right after the magic + a couple bytes).
        let truncated = &bytes[..11];
        assert!(matches!(
            Model::parse(truncated),
            Err(ModelError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn truncated_vertex_block_is_typed_error() {
        let full = synth_model("MDLV0017", 8, 4);
        // Drop the tail so the declared vertex block runs past the data.
        let truncated = &full[..full.len() - 100];
        match Model::parse(truncated) {
            Err(ModelError::InvalidVertexBlock { index, .. }) => assert_eq!(index, 0),
            Err(ModelError::UnexpectedEof { .. }) => {}
            other => panic!("expected typed truncation error, got {other:?}"),
        }
    }

    #[test]
    fn odd_index_block_is_rejected() {
        let mut bytes = synth_model("MDLV0017", 4, 2);
        // The last i32 written before the index payload is `ibytes`. Locate it:
        // it sits right before the final `tris*3*2` bytes. Corrupt to an odd
        // value by rewriting the index-byte-count field to 5.
        // ibytes field position = total_len - ibytes_payload - 4.
        let ibytes_payload = 2u32 * 3 * 2;
        let pos = bytes.len() - ibytes_payload as usize - 4;
        bytes[pos..pos + 4].copy_from_slice(&5i32.to_le_bytes());
        assert!(matches!(
            Model::parse(&bytes),
            Err(ModelError::InvalidIndexBlock { .. })
        ));
    }

    #[test]
    fn huge_mesh_count_does_not_panic() {
        // A corrupt header claiming ~2e9 meshes must NOT abort via a giant
        // `Vec::with_capacity`; the truncated body yields a typed error (§V9).
        let mut bytes = synth_model("MDLV0017", 1, 1);
        let pos = 8 + 1 + 4 + 4; // meshCount field offset
        bytes[pos..pos + 4].copy_from_slice(&2_000_000_000i32.to_le_bytes());
        // The first mesh's reads run past EOF long before 2e9 iterations.
        assert!(matches!(
            Model::parse(&bytes),
            Err(ModelError::UnexpectedEof { .. }) | Err(ModelError::InvalidVertexBlock { .. })
        ));
    }

    #[test]
    fn negative_mesh_count_is_rejected() {
        let mut bytes = synth_model("MDLV0017", 1, 1);
        // meshCount is the i32 at offset: magic("MDLV0017")=8 + NUL=1 + 4 + 4 = 17.
        let pos = 8 + 1 + 4 + 4;
        bytes[pos..pos + 4].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(matches!(
            Model::parse(&bytes),
            Err(ModelError::InvalidMeshCount { count: -1 })
        ));
    }

    // ---- corpus tests (skipped when the corpus is absent) ----------------

    use crate::pkg::Pkg;
    use std::path::PathBuf;

    /// Default corpus location; override with `KIRIE_CORPUS`.
    const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";

    fn corpus_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("KIRIE_CORPUS")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(CORPUS_DIR));
        if dir.is_dir() {
            Some(dir)
        } else {
            eprintln!("skipping corpus test: {} not found", dir.display());
            None
        }
    }

    /// Every entry in a `scene.pkg` whose payload starts with the `MDLV` magic.
    fn models_in_pkg(pkg: &Pkg<'_>) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        for entry in pkg.entries() {
            if let Ok(payload) = pkg.read(entry)
                && payload.len() >= 4
                && &payload[..4] == b"MDLV"
            {
                let name = entry.name_str().unwrap_or("<non-utf8>").to_owned();
                out.push((name, payload.to_vec()));
            }
        }
        out
    }

    #[test]
    fn parses_real_starscape_model() {
        let Some(dir) = corpus_dir() else { return };
        let pkg_path = dir.join("3047596375").join("scene.pkg");
        if !pkg_path.is_file() {
            eprintln!("skipping: {} not present", pkg_path.display());
            return;
        }
        let bytes = std::fs::read(&pkg_path).unwrap();
        let pkg = Pkg::parse(&bytes).expect("parse pkg");
        let models = models_in_pkg(&pkg);
        assert_eq!(models.len(), 1, "Starscape has exactly one .mdl");
        let (name, payload) = &models[0];
        assert_eq!(name, "models/space boi/space boi.mdl");

        let model = Model::parse(payload).expect("parse Starscape model");
        assert_eq!(model.version, "MDLV0017");
        assert_eq!(model.meshes.len(), 2);
        assert_eq!(model.total_vertices(), 61296);
        assert_eq!(model.total_indices(), 241992);

        for mesh in &model.meshes {
            // Non-empty geometry.
            assert!(mesh.vertex_count() > 0);
            assert!(!mesh.indices.is_empty());
            // Indices stay within the vertex range.
            let max_index = mesh.indices.iter().copied().max().unwrap();
            assert!((max_index as usize) < mesh.vertex_count());
            // Sane bounding box: min <= max on every axis, non-degenerate.
            for axis in 0..3 {
                assert!(mesh.bbox_min[axis] <= mesh.bbox_max[axis]);
            }
            // Every declared material path is a `.json` under materials/.
            assert!(mesh.material_ref.starts_with("materials/"));
            // A decoded position must fall inside (a small epsilon of) the bbox.
            let first = mesh.vertices().next().unwrap();
            for axis in 0..3 {
                assert!(
                    first.position[axis] >= mesh.bbox_min[axis] - 1.0
                        && first.position[axis] <= mesh.bbox_max[axis] + 1.0,
                    "vertex axis {axis} out of bbox"
                );
            }
        }
    }

    #[test]
    fn corpus_every_model_scans_like_the_reference() {
        let Some(dir) = corpus_dir() else { return };
        let mut items_with_models = 0usize;
        let mut total_models = 0usize;
        let mut parsed = 0usize;
        // MDLV0023 puppets with an 80-byte stride are rejected by the fixed
        // 48-byte reference reader; we reproduce that (see module docs).
        let mut ref_rejected = 0usize;
        // version-string -> (parsed, rejected)
        let mut versions: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

        for item in std::fs::read_dir(&dir).unwrap().filter_map(Result::ok) {
            let pkg_path = item.path().join("scene.pkg");
            if !pkg_path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&pkg_path).unwrap();
            let Ok(pkg) = Pkg::parse(&bytes) else { continue };
            let models = models_in_pkg(&pkg);
            if models.is_empty() {
                continue;
            }
            items_with_models += 1;
            for (name, payload) in &models {
                total_models += 1;
                // Every payload starts with an 8-byte MDLV version tag.
                let version = String::from_utf8_lossy(&payload[..8]).into_owned();
                match Model::parse(payload) {
                    Ok(model) => {
                        assert!(!model.meshes.is_empty(), "model {name} has no meshes");
                        assert!(model.total_vertices() > 0, "model {name} has no vertices");
                        assert_eq!(model.version, version);
                        parsed += 1;
                        versions.entry(version).or_default().0 += 1;
                    }
                    // The only tolerated failure mirrors the reference's own
                    // "Invalid vertex block" reject of 80-byte-stride puppets.
                    Err(ModelError::InvalidVertexBlock { .. }) => {
                        ref_rejected += 1;
                        versions.entry(version).or_default().1 += 1;
                    }
                    Err(e) => panic!("model {name} failed unexpectedly: {e}"),
                }
            }
        }

        eprintln!(
            "corpus: {total_models} model(s) across {items_with_models} item(s); \
             {parsed} parsed, {ref_rejected} reference-rejected; per-version (parsed,rejected) {versions:?}"
        );
        // Starscape's MDLV0017 model must always parse.
        assert!(parsed >= 1, "expected at least the MDLV0017 model to parse");
    }

    // ---- puppet-mesh tests -----------------------------------------------

    /// Build a minimal well-formed puppet payload: `MDLV0023\0`, some padding,
    /// a two-`u32` mesh header, `verts` × 80-byte records, an `indexBytes`
    /// `u32`, `tris` triangles of `u16` indices, then an `MDLS` marker.
    fn synth_puppet(version: &str, verts: u32, tris: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(version.as_bytes());
        b.push(0); // marker NUL (markerSize = 9)
        // A few bytes of pre-block padding the scanner must skip over.
        b.extend_from_slice(&[0xAB; 12]);
        // Mesh header: first u32 unused, second is vertexBytes.
        b.extend_from_slice(&0u32.to_le_bytes());
        let vbytes = verts * PUPPET_VERTEX_STRIDE as u32;
        b.extend_from_slice(&vbytes.to_le_bytes());
        for i in 0..verts {
            // position.x carries the index; uv encodes it too; bone_indices[0]=i.
            let mut rec = [0u8; PUPPET_VERTEX_STRIDE];
            rec[PUPPET_POSITION_OFFSET..PUPPET_POSITION_OFFSET + 4]
                .copy_from_slice(&(i as f32).to_le_bytes());
            rec[PUPPET_BONE_INDEX_OFFSET..PUPPET_BONE_INDEX_OFFSET + 4].copy_from_slice(&i.to_le_bytes());
            rec[PUPPET_BONE_WEIGHT_OFFSET..PUPPET_BONE_WEIGHT_OFFSET + 4]
                .copy_from_slice(&1.0f32.to_le_bytes());
            rec[PUPPET_UV_OFFSET..PUPPET_UV_OFFSET + 4].copy_from_slice(&(i as f32 * 0.5).to_le_bytes());
            b.extend_from_slice(&rec);
        }
        let ibytes = tris * 3 * 2;
        b.extend_from_slice(&ibytes.to_le_bytes());
        for t in 0..tris {
            for k in 0..3u16 {
                b.extend_from_slice(&((t as u16 + k) % verts as u16).to_le_bytes());
            }
        }
        b.extend_from_slice(b"MDLS");
        b.extend_from_slice(&[0u8; 8]);
        b
    }

    #[test]
    fn parses_synthetic_puppet() {
        let bytes = synth_puppet("MDLV0023", 6, 4);
        let m = PuppetMesh::parse(&bytes).expect("parse puppet");
        assert_eq!(m.version, "MDLV0023");
        assert_eq!(m.vertex_count(), 6);
        assert_eq!(m.indices.len(), 12);
        let v = &m.vertices[3];
        assert_eq!(v.position[0], 3.0);
        assert_eq!(v.bone_indices[0], 3);
        assert_eq!(v.bone_weights[0], 1.0);
        assert_eq!(v.uv[0], 1.5);
        // Every index stays within the vertex range.
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertex_count()));
    }

    #[test]
    fn puppet_accepts_mdlv0021() {
        let bytes = synth_puppet("MDLV0021", 3, 1);
        assert!(PuppetMesh::parse(&bytes).is_ok());
    }

    #[test]
    fn puppet_rejects_bad_magic() {
        let mut bytes = synth_puppet("MDLV0023", 3, 1);
        bytes[5] = b'X';
        match PuppetMesh::parse(&bytes) {
            Err(PuppetError::BadMagic { .. }) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
        // A plain (non-puppet) MDLV0017 header is also rejected here.
        let short = b"MDLV0017\0".to_vec();
        assert!(matches!(
            PuppetMesh::parse(&short),
            Err(PuppetError::BadMagic { .. })
        ));
    }

    #[test]
    fn puppet_rejects_out_of_range_index() {
        let mut bytes = synth_puppet("MDLV0023", 3, 1);
        // The first index sits right after the index-length u32. Locate the
        // index block: header(9+12) + meshHeader(8) + 3*80 verts + 4 (ibytes).
        let idx0 = 9 + 12 + 8 + 3 * PUPPET_VERTEX_STRIDE + 4;
        bytes[idx0..idx0 + 2].copy_from_slice(&99u16.to_le_bytes());
        match PuppetMesh::parse(&bytes) {
            Err(PuppetError::InvalidIndex {
                index: 99,
                vertex_count: 3,
            }) => {}
            other => panic!("expected InvalidIndex, got {other:?}"),
        }
    }

    #[test]
    fn puppet_no_block_before_mdls_is_typed_error() {
        // MDLV0023 marker immediately followed by MDLS — no room for a block.
        let mut bytes = b"MDLV0023\0".to_vec();
        bytes.extend_from_slice(b"MDLS");
        bytes.extend_from_slice(&[0u8; 16]);
        assert!(matches!(PuppetMesh::parse(&bytes), Err(PuppetError::NoMeshBlock)));
    }

    #[test]
    fn puppet_never_panics_on_random_input() {
        // A "MDLV0023"-prefixed blob of garbage must yield a typed error, never
        // a panic (§V9).
        let mut bytes = b"MDLV0023\0".to_vec();
        bytes.extend(
            std::iter::successors(Some(1u8), |n| Some(n.wrapping_mul(31).wrapping_add(7))).take(4096),
        );
        let _ = PuppetMesh::parse(&bytes); // must not panic
    }

    #[test]
    fn corpus_parses_scene_3428443753_puppets() {
        let Some(dir) = corpus_dir() else { return };
        let pkg_path = dir.join("3428443753").join("scene.pkg");
        if !pkg_path.is_file() {
            eprintln!("skipping: {} not present", pkg_path.display());
            return;
        }
        let bytes = std::fs::read(&pkg_path).unwrap();
        let pkg = Pkg::parse(&bytes).expect("parse pkg");
        // The three puppet .mdl files (girl/guy/cat).
        let mut puppets = Vec::new();
        for entry in pkg.entries() {
            let name = entry.name_str().unwrap_or("");
            if name.ends_with("_puppet.mdl")
                && let Ok(payload) = pkg.read(entry)
            {
                puppets.push((name.to_owned(), PuppetMesh::parse(payload).expect("parse puppet")));
            }
        }
        assert_eq!(puppets.len(), 3, "scene 3428443753 has three puppet meshes");
        for (name, mesh) in &puppets {
            assert_eq!(mesh.version, "MDLV0023", "{name}");
            assert!(mesh.vertex_count() > 0, "{name} has vertices");
            assert!(!mesh.indices.is_empty(), "{name} has indices");
            assert!(mesh.indices.len().is_multiple_of(3), "{name} whole triangles");
            let max = mesh.indices.iter().copied().max().unwrap();
            assert!((max as usize) < mesh.vertex_count(), "{name} indices in range");
            // Skinning weights on the first vertex are sane (sum ~ 1).
            let w: f32 = mesh.vertices[0].bone_weights.iter().sum();
            assert!(
                (w - 1.0).abs() < 0.01,
                "{name} first-vertex weights sum ~1 (got {w})"
            );
        }
    }
}
