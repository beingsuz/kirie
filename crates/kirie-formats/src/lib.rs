//! Parsers for Wallpaper Engine file formats.
//!
//! - [`pkg`]: `scene.pkg` archive container
//! - [`tex`]: `.tex` texture container (TEXV/TEXB/TEXI blocks)
//! - [`project`]: `project.json` wallpaper manifest
//! - [`model`]: `.mdl` 3D-model / mesh container (MDLV format)
//!
//! Byte-level format specs live in `docs/` at the repo root; parsers here
//! must match the reference C++ implementation exactly (see SPEC.md §V10).
//! All parsers return typed errors and never panic on malformed input (§V9).

#![forbid(unsafe_code)]

pub mod model;
pub mod pkg;
pub mod project;
pub mod tex;
