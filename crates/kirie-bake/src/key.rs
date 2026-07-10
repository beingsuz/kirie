//! Bundle cache keys (SPEC.md §V8).
//!
//! > V8: bundle key = blake3(source) ⊕ bake-format ver ⊕ shader-translator ver.
//! > Key mismatch → rebake. ⊥ migration code.
//!
//! We realize the `⊕`-of-versions as domain-separated hasher updates: the key is
//! `blake3(source ‖ BAKE_FORMAT_VERSION_le ‖ TRANSLATOR_VERSION_le)`. Any change
//! to the source bytes, the bundle layout, or the shader translator yields a
//! fresh digest → a different cache directory → a guaranteed miss → a rebake,
//! with no on-disk migration path (§V8). This is stronger than an integer XOR
//! (which could alias) while serving the same invariant.

use std::fmt;

/// On-disk bundle layout version. Bump whenever the [`crate::BakedBundle`] shape
/// or its encoding changes so every prior bundle keys to a different directory
/// and is transparently re-baked (SPEC.md §V8 — no migration).
pub const BAKE_FORMAT_VERSION: u32 = 1;

/// The 256-bit content-addressed key for a bundle (SPEC.md §V8). Its lowercase
/// hex form names the cache subdirectory.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BundleKey([u8; 32]);

impl BundleKey {
    /// Compute the key for `source` — the raw `scene.pkg` / `project.json` bytes
    /// that define the wallpaper — mixed with [`BAKE_FORMAT_VERSION`] and
    /// [`kirie_shader::TRANSLATOR_VERSION`] (SPEC.md §V8).
    #[must_use]
    pub fn compute(source: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(source);
        // Domain-separate the two versions so bumping either changes the key.
        h.update(&BAKE_FORMAT_VERSION.to_le_bytes());
        h.update(&kirie_shader::TRANSLATOR_VERSION.to_le_bytes());
        BundleKey(*h.finalize().as_bytes())
    }

    /// The raw 32-byte digest.
    #[must_use]
    pub fn bytes(&self) -> [u8; 32] {
        self.0
    }

    /// Lowercase hex, used as the cache directory name.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

impl fmt::Debug for BundleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BundleKey({})", self.to_hex())
    }
}

impl fmt::Display for BundleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_source_same_key() {
        assert_eq!(BundleKey::compute(b"abc"), BundleKey::compute(b"abc"));
    }

    #[test]
    fn source_change_changes_key() {
        assert_ne!(BundleKey::compute(b"abc"), BundleKey::compute(b"abd"));
    }

    #[test]
    fn hex_is_64_chars() {
        assert_eq!(BundleKey::compute(b"x").to_hex().len(), 64);
    }
}
