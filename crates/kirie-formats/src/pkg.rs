//! `scene.pkg` archive parser. Spec: docs/format-pkg.md
//!
//! `scene.pkg` is an uncompressed, unencrypted flat-file archive: an `sstr`
//! magic (e.g. `"PKGV0001"`), a `u32` entry count, `count` × { `sstr` name,
//! `u32` offset, `u32` length } table entries, then a data region of
//! concatenated payloads (docs/format-pkg.md §1, §3). All multi-byte integers
//! are little-endian **unconditionally**, regardless of host endianness
//! (docs/format-pkg.md §2). An `sstr` is a `u32` byte length followed by
//! exactly that many raw bytes, with no NUL terminator (§2).
//!
//! Entry payload `i` occupies bytes
//! `[baseOffset + offset_i, baseOffset + offset_i + length_i)` where
//! `baseOffset` is the stream position immediately after the last table entry
//! (§3). Entry names are opaque, conventionally-UTF-8 byte strings matched
//! byte-exactly and case-sensitively (§2, §5).
//!
//! Per SPEC.md §V9 this parser never panics on malformed input: every read is
//! bounds-checked and every offset computation uses checked arithmetic,
//! returning typed [`PkgError`]s.

use std::ops::Range;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// Errors produced by the `scene.pkg` parser (docs/format-pkg.md §9).
#[derive(Debug, Error)]
pub enum PkgError {
    /// The magic sized string does not start with the 4 bytes `PKGV`
    /// (docs/format-pkg.md §4; reference check `PackageParser.cpp:18-19`).
    #[error("expected magic to start with \"PKGV\", got {found:?}")]
    BadMagic {
        /// The rejected magic string, decoded lossily for display.
        found: String,
    },

    /// The header or entry table ended before the expected field. The
    /// reference parser does not detect truncation, but the sane model —
    /// which docs/format-pkg.md §9 prescribes for reimplementations — is a
    /// hard error for any header/table read hitting EOF, including an `sstr`
    /// length exceeding the remaining bytes (§4 sanity bound).
    #[error(
        "truncated package: need {needed} byte(s) for {what} at offset {offset}, \
         only {available} available"
    )]
    Truncated {
        /// Which field was being read when the data ran out.
        what: &'static str,
        /// Byte offset in the package where the read started.
        offset: usize,
        /// Bytes required by the field.
        needed: usize,
        /// Bytes actually remaining.
        available: usize,
    },

    /// An entry payload extends past the end of the package
    /// (`baseOffset + offset + length > file size`). The current C++
    /// reference silently returns **zero-filled** data for the unreadable
    /// bytes (docs/format-pkg.md §9: the seek+read failure never surfaces and
    /// the value-initialized buffer stays zeroed). We refuse instead: silently
    /// fabricated zero bytes would corrupt downstream consumers (e.g. a
    /// half-zeroed `.tex`), and §5/§9 direct reimplementations to treat this
    /// as a hard error, as the pre-rewrite `CPackage` did ("Cannot read file
    /// <name> contents from package").
    #[error(
        "entry {name:?} payload out of bounds: base offset {base_offset} + entry offset \
         {offset} + length {length} exceeds package size {package_size}"
    )]
    PayloadOutOfBounds {
        /// Entry name, decoded lossily for display.
        name: String,
        /// Computed base offset of the data region (docs/format-pkg.md §3).
        base_offset: usize,
        /// Entry offset relative to the data region.
        offset: u32,
        /// Entry payload length in bytes.
        length: u32,
        /// Total package size in bytes.
        package_size: usize,
    },

    /// No entry with the requested name exists in the table. Mirrors the
    /// reference "Cannot find file" error on `open()` of an unknown name
    /// (docs/format-pkg.md §9, `Adapters/Package.cpp:22-23`).
    #[error("no entry named {name:?} in package")]
    EntryNotFound {
        /// The looked-up name, decoded lossily for display.
        name: String,
    },

    /// Reading the package file from disk failed (owned loader only).
    #[error("failed to read package file `{}`: {source}", path.display())]
    Io {
        /// Path passed to [`OwnedPkg::from_path`].
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

/// One file-entry-table row: `{ name: sstr, offset: u32, length: u32 }`
/// (docs/format-pkg.md §5). The name borrows the parsed input (zero-copy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry<'a> {
    /// Archive-relative path: `/`-separated, no leading slash, matched
    /// byte-exactly and case-sensitively. Opaque bytes, conventionally UTF-8
    /// (docs/format-pkg.md §2, §5).
    pub name: &'a [u8],
    /// Payload byte offset **relative to `baseOffset`**, not absolute in the
    /// file (docs/format-pkg.md §5).
    pub offset: u32,
    /// Payload byte count (docs/format-pkg.md §5).
    pub len: u32,
}

impl<'a> Entry<'a> {
    /// The entry name as UTF-8, or `None` if it is not valid UTF-8. Names
    /// are conventionally UTF-8 but must be treated as opaque bytes
    /// (docs/format-pkg.md §2).
    #[must_use]
    pub fn name_str(&self) -> Option<&'a str> {
        std::str::from_utf8(self.name).ok()
    }
}

/// A parsed `scene.pkg` archive borrowing the input bytes (zero-copy).
///
/// Parsing reads only the header and entry table; payloads are sliced lazily
/// on [`Pkg::read`], matching the reference's lazy `open()`
/// (docs/format-pkg.md §8).
#[derive(Debug, Clone)]
pub struct Pkg<'a> {
    data: &'a [u8],
    magic: &'a [u8],
    base_offset: usize,
    entries: Vec<Entry<'a>>,
}

impl<'a> Pkg<'a> {
    /// Parse a `scene.pkg` archive from a byte slice.
    ///
    /// Layout per docs/format-pkg.md §3: `sstr` magic (must start with
    /// `PKGV`, §4), `u32` count, `count` × entry, then the data region.
    /// Payload bounds are *not* validated here: per §5 a reader must tolerate
    /// arbitrary entry order, gaps, and overlaps (all accepted by the
    /// reference); out-of-bounds payloads are rejected at [`Pkg::read`] time.
    pub fn parse(data: &'a [u8]) -> Result<Self, PkgError> {
        let raw = parse_raw(data)?;
        let entries = raw
            .entries
            .iter()
            .map(|e| Entry {
                name: slice(data, &e.name),
                offset: e.offset,
                len: e.len,
            })
            .collect();
        Ok(Self {
            data,
            magic: slice(data, &raw.magic),
            base_offset: raw.base_offset,
            entries,
        })
    }

    /// The full magic/version string, e.g. `b"PKGV0001"`. Guaranteed to start
    /// with `PKGV`; the suffix is the version and gates no behavior
    /// (docs/format-pkg.md §4, §10.1).
    #[must_use]
    pub fn magic(&self) -> &'a [u8] {
        self.magic
    }

    /// The version suffix of the magic after `PKGV`, e.g. `b"0001"`.
    /// Recorded but never used to gate behavior (docs/format-pkg.md §4).
    #[must_use]
    pub fn version(&self) -> &'a [u8] {
        // Magic is guaranteed >= 4 bytes ("PKGV" prefix checked in parse).
        self.magic.get(4..).unwrap_or(&[])
    }

    /// Start of the data region: the stream position immediately after the
    /// last table entry — `baseOffset = 4 + len(magic) + 4 + Σ over entries
    /// (4 + len(name) + 4 + 4)` (docs/format-pkg.md §3).
    #[must_use]
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Number of entries in the file table (docs/format-pkg.md §3 `count`).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The entry table, in on-disk order (docs/format-pkg.md §5).
    #[must_use]
    pub fn entries(&self) -> &[Entry<'a>] {
        &self.entries
    }

    /// Look up an entry by exact, case-sensitive byte equality of its name.
    /// Linear scan; the first occurrence wins for duplicate names, matching
    /// the reference `std::ranges::find_if` (docs/format-pkg.md §5).
    #[must_use]
    pub fn get(&self, name: &[u8]) -> Option<Entry<'a>> {
        self.entries.iter().find(|e| e.name == name).copied()
    }

    /// Read an entry's payload: bytes
    /// `[baseOffset + offset, baseOffset + offset + len)`
    /// (docs/format-pkg.md §3). Returns [`PkgError::PayloadOutOfBounds`] if
    /// the range exceeds the package instead of zero-filling like the
    /// reference (docs/format-pkg.md §9; see the error's docs for rationale).
    pub fn read(&self, entry: &Entry<'_>) -> Result<&'a [u8], PkgError> {
        read_payload(self.data, self.base_offset, entry.name, entry.offset, entry.len)
    }

    /// Look up `name` and read its payload in one step. Returns
    /// [`PkgError::EntryNotFound`] for unknown names, mirroring the reference
    /// "Cannot find file" (docs/format-pkg.md §9).
    pub fn read_name(&self, name: &[u8]) -> Result<&'a [u8], PkgError> {
        let entry = self.get(name).ok_or_else(|| PkgError::EntryNotFound {
            name: String::from_utf8_lossy(name).into_owned(),
        })?;
        self.read(&entry)
    }
}

/// A parsed `scene.pkg` archive that owns its bytes. Convenience wrapper for
/// loading from a path; same semantics as [`Pkg`].
#[derive(Debug, Clone)]
pub struct OwnedPkg {
    data: Vec<u8>,
    raw: RawPkg,
}

impl OwnedPkg {
    /// Read and parse a `scene.pkg` file from disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, PkgError> {
        let path = path.as_ref();
        let data = std::fs::read(path).map_err(|source| PkgError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_vec(data)
    }

    /// Parse a `scene.pkg` archive from an owned byte buffer.
    pub fn from_vec(data: Vec<u8>) -> Result<Self, PkgError> {
        let raw = parse_raw(&data)?;
        Ok(Self { data, raw })
    }

    /// The raw bytes of the whole package.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// The full magic/version string (docs/format-pkg.md §4). See
    /// [`Pkg::magic`].
    #[must_use]
    pub fn magic(&self) -> &[u8] {
        slice(&self.data, &self.raw.magic)
    }

    /// The version suffix after `PKGV` (docs/format-pkg.md §4). See
    /// [`Pkg::version`].
    #[must_use]
    pub fn version(&self) -> &[u8] {
        self.magic().get(4..).unwrap_or(&[])
    }

    /// Start of the data region (docs/format-pkg.md §3). See
    /// [`Pkg::base_offset`].
    #[must_use]
    pub fn base_offset(&self) -> usize {
        self.raw.base_offset
    }

    /// Number of entries in the file table (docs/format-pkg.md §3 `count`).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.raw.entries.len()
    }

    /// Iterate the entry table in on-disk order (docs/format-pkg.md §5).
    pub fn entries(&self) -> impl Iterator<Item = Entry<'_>> {
        self.raw.entries.iter().map(|e| Entry {
            name: slice(&self.data, &e.name),
            offset: e.offset,
            len: e.len,
        })
    }

    /// Look up an entry by exact name (docs/format-pkg.md §5). See
    /// [`Pkg::get`].
    #[must_use]
    pub fn get(&self, name: &[u8]) -> Option<Entry<'_>> {
        self.entries().find(|e| e.name == name)
    }

    /// Read an entry's payload (docs/format-pkg.md §3, §9). See
    /// [`Pkg::read`].
    pub fn read(&self, entry: &Entry<'_>) -> Result<&[u8], PkgError> {
        read_payload(
            &self.data,
            self.raw.base_offset,
            entry.name,
            entry.offset,
            entry.len,
        )
    }

    /// Look up `name` and read its payload in one step. See
    /// [`Pkg::read_name`].
    pub fn read_name(&self, name: &[u8]) -> Result<&[u8], PkgError> {
        let entry = self.get(name).ok_or_else(|| PkgError::EntryNotFound {
            name: String::from_utf8_lossy(name).into_owned(),
        })?;
        // Re-borrow the payload directly to decouple its lifetime from `entry`.
        read_payload(
            &self.data,
            self.raw.base_offset,
            entry.name,
            entry.offset,
            entry.len,
        )
    }
}

/// Parsed structure as byte ranges into the source buffer, so it can back
/// both the borrowed [`Pkg`] and the owning [`OwnedPkg`] without
/// self-referential borrows.
#[derive(Debug, Clone)]
struct RawPkg {
    magic: Range<usize>,
    base_offset: usize,
    entries: Vec<RawEntry>,
}

#[derive(Debug, Clone)]
struct RawEntry {
    name: Range<usize>,
    offset: u32,
    len: u32,
}

/// Parse header + entry table per docs/format-pkg.md §3 (normative summary
/// §10 steps 1-3).
fn parse_raw(data: &[u8]) -> Result<RawPkg, PkgError> {
    let mut r = Reader { data, pos: 0 };

    // §4: the magic is itself a sized string; accept any sstr starting with
    // the 4 bytes "PKGV" (any suffix, any length — reference prefix rule,
    // PackageParser.cpp:18). §4 sanity bound: an sstr length > remaining
    // bytes is rejected (as Truncated) instead of blindly allocated.
    let magic = r.read_sstr("magic length", "magic bytes")?;
    let magic_bytes = slice(data, &magic);
    if !magic_bytes.starts_with(b"PKGV") {
        return Err(PkgError::BadMagic {
            found: String::from_utf8_lossy(magic_bytes).into_owned(),
        });
    }

    // §3: u32 count, then count × { sstr name, u32 offset, u32 length }.
    let count = r.read_u32("entry count")?;
    // Each entry is at least 12 bytes on the wire (4 name-length + 4 offset
    // + 4 length), so cap the preallocation by what could actually fit —
    // a hostile count cannot force a huge allocation (SPEC.md §V9).
    let remaining = data.len().saturating_sub(r.pos);
    let mut entries = Vec::with_capacity((count as usize).min(remaining / 12));
    for _ in 0..count {
        let name = r.read_sstr("entry name length", "entry name bytes")?;
        let offset = r.read_u32("entry offset")?;
        let len = r.read_u32("entry length")?;
        entries.push(RawEntry { name, offset, len });
    }

    // §3: baseOffset is *computed* as the stream position immediately after
    // the last table entry (PackageParser.cpp:26-27); there is no stored
    // header size, data pointer, or EOF marker.
    Ok(RawPkg {
        magic,
        base_offset: r.pos,
        entries,
    })
}

/// Bounds-checked little-endian cursor (docs/format-pkg.md §2: decode
/// little-endian unconditionally, regardless of host).
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Consume exactly `len` bytes or fail with [`PkgError::Truncated`].
    fn take(&mut self, len: usize, what: &'static str) -> Result<&'a [u8], PkgError> {
        let truncated = || PkgError::Truncated {
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

    /// Read a `u32`, little-endian unconditionally (docs/format-pkg.md §2).
    fn read_u32(&mut self, what: &'static str) -> Result<u32, PkgError> {
        let offset = self.pos;
        let bytes = self.take(4, what)?;
        match bytes.first_chunk::<4>() {
            Some(arr) => Ok(u32::from_le_bytes(*arr)),
            // `take` returned exactly 4 bytes, so this arm is unreachable;
            // kept as a typed error instead of a panic path (SPEC.md §V9).
            None => Err(PkgError::Truncated {
                what,
                offset,
                needed: 4,
                available: bytes.len(),
            }),
        }
    }

    /// Read an `sstr`: `u32` byte length, then exactly that many raw bytes,
    /// no NUL terminator (docs/format-pkg.md §2). Returns the byte range of
    /// the string content within the input.
    fn read_sstr(
        &mut self,
        what_len: &'static str,
        what_bytes: &'static str,
    ) -> Result<Range<usize>, PkgError> {
        let len = self.read_u32(what_len)?;
        let start = self.pos;
        self.take(len as usize, what_bytes)?;
        Ok(start..self.pos)
    }
}

/// Slice `data` by a range produced by [`parse_raw`]. Such ranges are
/// in-bounds by construction; the empty-slice fallback exists only to avoid
/// a panic path (SPEC.md §V9), it is unreachable for validated ranges.
fn slice<'d>(data: &'d [u8], range: &Range<usize>) -> &'d [u8] {
    data.get(range.clone()).unwrap_or(&[])
}

/// Slice payload bytes `[base_offset + offset, base_offset + offset + length)`
/// (docs/format-pkg.md §3). Rejects out-of-bounds payloads with a typed error
/// where the reference zero-fills (docs/format-pkg.md §9; rationale on
/// [`PkgError::PayloadOutOfBounds`]). All arithmetic is checked (SPEC.md §V9).
fn read_payload<'d>(
    data: &'d [u8],
    base_offset: usize,
    name: &[u8],
    offset: u32,
    length: u32,
) -> Result<&'d [u8], PkgError> {
    let oob = || PkgError::PayloadOutOfBounds {
        name: String::from_utf8_lossy(name).into_owned(),
        base_offset,
        offset,
        length,
        package_size: data.len(),
    };
    let start = (base_offset as u64)
        .checked_add(u64::from(offset))
        .ok_or_else(oob)?;
    let end = start.checked_add(u64::from(length)).ok_or_else(oob)?;
    if end > data.len() as u64 {
        return Err(oob());
    }
    // end <= data.len() <= usize::MAX, so both conversions succeed.
    let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
        return Err(oob());
    };
    data.get(start..end).ok_or_else(oob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ---- synthetic archive builder -------------------------------------

    /// Encode an `sstr` (docs/format-pkg.md §2).
    fn sstr(s: &[u8]) -> Vec<u8> {
        let mut v = (s.len() as u32).to_le_bytes().to_vec();
        v.extend_from_slice(s);
        v
    }

    /// Build a synthetic archive per docs/format-pkg.md §3.
    fn build_pkg(magic: &[u8], entries: &[(&[u8], u32, u32)], payload: &[u8]) -> Vec<u8> {
        let mut v = sstr(magic);
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, offset, len) in entries {
            v.extend_from_slice(&sstr(name));
            v.extend_from_slice(&offset.to_le_bytes());
            v.extend_from_slice(&len.to_le_bytes());
        }
        v.extend_from_slice(payload);
        v
    }

    // ---- synthetic tests ------------------------------------------------

    #[test]
    fn happy_path_two_entries() {
        let data = build_pkg(
            b"PKGV0001",
            &[(b"scene.json", 0, 5), (b"a/b.txt", 5, 3)],
            b"helloabc",
        );
        let pkg = Pkg::parse(&data).unwrap();

        assert_eq!(pkg.magic(), b"PKGV0001");
        assert_eq!(pkg.version(), b"0001");
        // baseOffset formula (docs/format-pkg.md §3):
        // 4 + 8 (magic) + 4 (count) + (4+10+4+4) + (4+7+4+4) = 57
        assert_eq!(pkg.base_offset(), 57);
        assert_eq!(pkg.base_offset(), data.len() - 8);
        assert_eq!(pkg.entry_count(), 2);

        let e0 = pkg.entries()[0];
        assert_eq!(e0.name, b"scene.json");
        assert_eq!(e0.name_str(), Some("scene.json"));
        assert_eq!((e0.offset, e0.len), (0, 5));
        assert_eq!(pkg.read(&e0).unwrap(), b"hello");

        let e1 = pkg.get(b"a/b.txt").unwrap();
        assert_eq!((e1.offset, e1.len), (5, 3));
        assert_eq!(pkg.read(&e1).unwrap(), b"abc");
        assert_eq!(pkg.read_name(b"scene.json").unwrap(), b"hello");

        // Lookup is exact and case-sensitive (docs/format-pkg.md §5).
        assert!(pkg.get(b"Scene.json").is_none());
        assert!(pkg.get(b"missing").is_none());
        assert!(matches!(
            pkg.read_name(b"missing"),
            Err(PkgError::EntryNotFound { .. })
        ));
    }

    #[test]
    fn empty_archive() {
        let data = build_pkg(b"PKGV0009", &[], b"");
        let pkg = Pkg::parse(&data).unwrap();
        assert_eq!(pkg.entry_count(), 0);
        assert_eq!(pkg.base_offset(), data.len());
        assert!(pkg.get(b"anything").is_none());
    }

    #[test]
    fn non_pkgv_magic_rejected() {
        // Wrong prefix (docs/format-pkg.md §4).
        let data = build_pkg(b"NOPE0001", &[], b"");
        assert!(matches!(
            Pkg::parse(&data),
            Err(PkgError::BadMagic { found }) if found == "NOPE0001"
        ));
        // Magic shorter than the 4-byte prefix cannot match.
        let data = build_pkg(b"PKG", &[], b"");
        assert!(matches!(Pkg::parse(&data), Err(PkgError::BadMagic { .. })));
        let data = build_pkg(b"", &[], b"");
        assert!(matches!(Pkg::parse(&data), Err(PkgError::BadMagic { .. })));
    }

    #[test]
    fn magic_prefix_rule_accepts_any_suffix_and_length() {
        // §4: accept any sstr starting with "PKGV"; suffix = version.
        for (magic, version) in [
            (b"PKGVabcd".as_slice(), b"abcd".as_slice()),
            (b"PKGV000123".as_slice(), b"000123".as_slice()),
            (b"PKGV".as_slice(), b"".as_slice()),
        ] {
            let data = build_pkg(magic, &[], b"");
            let pkg = Pkg::parse(&data).unwrap();
            assert_eq!(pkg.magic(), magic);
            assert_eq!(pkg.version(), version);
        }
    }

    #[test]
    fn truncated_header() {
        // Empty input: can't even read the magic length.
        assert!(matches!(
            Pkg::parse(&[]),
            Err(PkgError::Truncated {
                what: "magic length",
                ..
            })
        ));
        // Partial magic length u32.
        assert!(matches!(
            Pkg::parse(&[0x08, 0x00]),
            Err(PkgError::Truncated {
                what: "magic length",
                ..
            })
        ));
        // Magic sstr length exceeds remaining bytes (§4 sanity bound).
        let mut data = 100u32.to_le_bytes().to_vec();
        data.extend_from_slice(b"PKGV0001");
        assert!(matches!(
            Pkg::parse(&data),
            Err(PkgError::Truncated {
                what: "magic bytes",
                ..
            })
        ));
        // Missing entry count after a valid magic.
        assert!(matches!(
            Pkg::parse(&sstr(b"PKGV0001")),
            Err(PkgError::Truncated {
                what: "entry count",
                ..
            })
        ));
    }

    #[test]
    fn truncated_table() {
        // count = 2 but only one entry present.
        let full = build_pkg(b"PKGV0001", &[(b"a", 0, 1)], b"x");
        let mut data = full.clone();
        // Patch count (u32 right after the 12-byte magic sstr) to 2.
        data[12..16].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            Pkg::parse(&data),
            Err(PkgError::Truncated {
                what: "entry name length",
                ..
            })
        ));

        // Entry name bytes cut short: name length says 4, only 2 present.
        let mut data = sstr(b"PKGV0001");
        data.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        data.extend_from_slice(&4u32.to_le_bytes()); // name length = 4
        data.extend_from_slice(b"ab"); // ...but only 2 bytes follow
        assert!(matches!(
            Pkg::parse(&data),
            Err(PkgError::Truncated {
                what: "entry name bytes",
                ..
            })
        ));

        // Entry missing its offset u32.
        let mut data = sstr(b"PKGV0001");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&sstr(b"scene.json"));
        assert!(matches!(
            Pkg::parse(&data),
            Err(PkgError::Truncated {
                what: "entry offset",
                ..
            })
        ));

        // Entry missing its length u32.
        let mut data = sstr(b"PKGV0001");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&sstr(b"scene.json"));
        data.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            Pkg::parse(&data),
            Err(PkgError::Truncated {
                what: "entry length",
                ..
            })
        ));
    }

    #[test]
    fn hostile_entry_count_fails_fast_without_huge_alloc() {
        let mut data = sstr(b"PKGV0001");
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(Pkg::parse(&data), Err(PkgError::Truncated { .. })));
    }

    #[test]
    fn payload_out_of_range_is_typed_error() {
        // Entry claims 100 bytes but only 5 exist after the table
        // (docs/format-pkg.md §9: reference zero-fills; we reject).
        let data = build_pkg(b"PKGV0001", &[(b"big", 0, 100)], b"hello");
        let pkg = Pkg::parse(&data).unwrap(); // parse tolerates it (§5)
        let entry = pkg.get(b"big").unwrap();
        assert!(matches!(
            pkg.read(&entry),
            Err(PkgError::PayloadOutOfBounds { length: 100, .. })
        ));

        // Zero-length entry pointing past EOF is still out of bounds
        // (baseOffset + offset + length > file size, §5).
        let data = build_pkg(b"PKGV0001", &[(b"z", 100, 0)], b"hello");
        let pkg = Pkg::parse(&data).unwrap();
        let entry = pkg.get(b"z").unwrap();
        assert!(matches!(
            pkg.read(&entry),
            Err(PkgError::PayloadOutOfBounds { .. })
        ));

        // Zero-length entry exactly at EOF is fine (end == file size).
        let data = build_pkg(b"PKGV0001", &[(b"end", 5, 0)], b"hello");
        let pkg = Pkg::parse(&data).unwrap();
        let entry = pkg.get(b"end").unwrap();
        assert_eq!(pkg.read(&entry).unwrap(), b"");

        // u32::MAX offset+length must not overflow anything.
        let data = build_pkg(b"PKGV0001", &[(b"max", u32::MAX, u32::MAX)], b"hello");
        let pkg = Pkg::parse(&data).unwrap();
        let entry = pkg.get(b"max").unwrap();
        assert!(matches!(
            pkg.read(&entry),
            Err(PkgError::PayloadOutOfBounds { .. })
        ));
    }

    #[test]
    fn duplicate_names_first_occurrence_wins() {
        // §5: lookups are a linear scan; first occurrence wins.
        let data = build_pkg(b"PKGV0001", &[(b"dup", 0, 3), (b"dup", 3, 3)], b"aaabbb");
        let pkg = Pkg::parse(&data).unwrap();
        let entry = pkg.get(b"dup").unwrap();
        assert_eq!(pkg.read(&entry).unwrap(), b"aaa");
    }

    #[test]
    fn unordered_gapped_overlapping_entries_tolerated() {
        // §5: a reader tolerates arbitrary order, gaps, and overlaps.
        let data = build_pkg(
            b"PKGV0002",
            &[(b"late", 6, 2), (b"early", 0, 4), (b"overlap", 2, 4)],
            b"01234567",
        );
        let pkg = Pkg::parse(&data).unwrap();
        assert_eq!(pkg.read_name(b"late").unwrap(), b"67");
        assert_eq!(pkg.read_name(b"early").unwrap(), b"0123");
        assert_eq!(pkg.read_name(b"overlap").unwrap(), b"2345");
    }

    #[test]
    fn utf8_names_matched_byte_exactly() {
        // §2: sstr length counts bytes, not characters; "models/背景.json"
        // is 18 bytes (7 + 3·2 + 5).
        let name = "models/背景.json".as_bytes();
        assert_eq!(name.len(), 18);
        let data = build_pkg(b"PKGV0022", &[(name, 0, 2)], b"{}");
        let pkg = Pkg::parse(&data).unwrap();
        let entry = pkg.get(name).unwrap();
        assert_eq!(entry.name, name);
        assert_eq!(pkg.read(&entry).unwrap(), b"{}");
    }

    #[test]
    fn entries_are_zero_copy_views_of_input() {
        let data = build_pkg(b"PKGV0001", &[(b"scene.json", 0, 5)], b"hello");
        let (entry, payload) = {
            let pkg = Pkg::parse(&data).unwrap();
            let entry = pkg.get(b"scene.json").unwrap();
            let payload = pkg.read(&entry).unwrap();
            (entry, payload)
        };
        // Entry and payload outlive the dropped Pkg: they borrow `data`.
        // Name bytes start right after magic sstr (12) + count (4) + name
        // length (4) = offset 20.
        assert_eq!(entry.name.as_ptr(), data[20..].as_ptr());
        assert_eq!(payload.as_ptr(), data[data.len() - 5..].as_ptr());
    }

    // ---- owned loader ----------------------------------------------------

    static TEMP_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(bytes: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "kirie-pkg-test-{}-{}.pkg",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn owned_pkg_from_path_round_trip() {
        let bytes = build_pkg(
            b"PKGV0024",
            &[(b"scene.json", 0, 5), (b"a/b.txt", 5, 3)],
            b"helloabc",
        );
        let path = temp_file(&bytes);
        let pkg = OwnedPkg::from_path(&path).unwrap();
        assert_eq!(pkg.magic(), b"PKGV0024");
        assert_eq!(pkg.version(), b"0024");
        assert_eq!(pkg.entry_count(), 2);
        assert_eq!(pkg.base_offset(), bytes.len() - 8);
        assert_eq!(pkg.as_bytes(), bytes.as_slice());
        let names: Vec<Vec<u8>> = pkg.entries().map(|e| e.name.to_vec()).collect();
        assert_eq!(names, [b"scene.json".to_vec(), b"a/b.txt".to_vec()]);
        let entry = pkg.get(b"a/b.txt").unwrap();
        assert_eq!(pkg.read(&entry).unwrap(), b"abc");
        assert_eq!(pkg.read_name(b"scene.json").unwrap(), b"hello");
        assert!(matches!(
            pkg.read_name(b"missing"),
            Err(PkgError::EntryNotFound { .. })
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn owned_pkg_from_path_missing_file_is_io_error() {
        let err = OwnedPkg::from_path("/nonexistent/kirie-no-such-file.pkg").unwrap_err();
        assert!(matches!(err, PkgError::Io { .. }));
    }

    #[test]
    fn owned_pkg_from_vec_rejects_malformed() {
        assert!(matches!(
            OwnedPkg::from_vec(vec![]),
            Err(PkgError::Truncated { .. })
        ));
        assert!(matches!(
            OwnedPkg::from_vec(build_pkg(b"XXXX0001", &[], b"")),
            Err(PkgError::BadMagic { .. })
        ));
    }

    // ---- corpus tests (skipped when the corpus is absent) ----------------

    /// Default corpus location (docs/format-pkg.md §7); override with
    /// `KIRIE_CORPUS`.
    const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
    /// docs/format-pkg.md §7: 19 scene.pkg archives in the corpus.
    const CORPUS_SCENE_PKG_COUNT: usize = 19;
    /// Sum of the per-item entry counts in the docs/format-pkg.md §7 table.
    const CORPUS_TOTAL_ENTRIES: usize = 800;

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

    #[test]
    fn corpus_every_archive_parses_and_every_entry_reads() {
        let Some(dir) = corpus_dir() else { return };
        let paths = corpus_scene_pkgs(&dir);
        // Live Steam corpus grows; the documented count is a floor. Every
        // installed archive (documented or newly subscribed) must still parse
        // and read every entry in-bounds — that is the real invariant.
        assert!(
            paths.len() >= CORPUS_SCENE_PKG_COUNT,
            "corpus scene.pkg count {} fell below docs/format-pkg.md §7 floor {CORPUS_SCENE_PKG_COUNT}",
            paths.len()
        );

        let mut total_entries = 0usize;
        for path in &paths {
            let pkg = OwnedPkg::from_path(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(pkg.magic().starts_with(b"PKGV"), "{}: bad magic", path.display());
            assert!(pkg.entry_count() > 0, "{}: no entries", path.display());
            total_entries += pkg.entry_count();
            for entry in pkg.entries() {
                // Reading every entry proves every payload is in-bounds.
                let payload = pkg
                    .read(&entry)
                    .unwrap_or_else(|e| panic!("{}: entry {:?}: {e}", path.display(), entry.name_str()));
                assert_eq!(payload.len(), entry.len as usize);
            }
        }
        assert!(
            total_entries >= CORPUS_TOTAL_ENTRIES,
            "total corpus entry count {total_entries} fell below docs/format-pkg.md §7 floor {CORPUS_TOTAL_ENTRIES}"
        );
    }

    #[test]
    fn corpus_item_1388331347_matches_spec_hexdump() {
        // Byte-anchored cross-check against docs/format-pkg.md §6/§7.
        let Some(dir) = corpus_dir() else { return };
        let path = dir.join("1388331347/scene.pkg");
        if !path.is_file() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let pkg = OwnedPkg::from_path(&path).unwrap();
        assert_eq!(pkg.as_bytes().len(), 4_124_099); // §6 file size
        assert_eq!(pkg.magic(), b"PKGV0001");
        assert_eq!(pkg.version(), b"0001");
        assert_eq!(pkg.entry_count(), 44); // §7
        assert_eq!(pkg.base_offset(), 0x891); // §7

        // §6: entry[0] = shaders/effects/waterflow.vert, offset 0, len 449,
        // payload begins "\r\nuniform mat4 g_ModelViewProje…".
        let e0 = pkg.entries().next().unwrap();
        assert_eq!(e0.name, b"shaders/effects/waterflow.vert");
        assert_eq!((e0.offset, e0.len), (0, 449));
        let payload = pkg.read(&e0).unwrap();
        assert!(payload.starts_with(b"\r\nuniform mat4 g_ModelViewProje"));

        // §6: entry[1] = scene.json, offset 449, len 5050, payload is JSON
        // starting "{\r\n\t\"camera\"".
        let scene = pkg.get(b"scene.json").unwrap();
        assert_eq!((scene.offset, scene.len), (449, 5050));
        let payload = pkg.read(&scene).unwrap();
        assert!(payload.starts_with(b"{\r\n\t\"camera\""));
    }
}
