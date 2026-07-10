//! Typed errors for scene-renderer construction (SPEC.md §V9: no panics on
//! malformed input; per-object/per-pass failures degrade to a skip with a
//! trace note rather than aborting the whole wallpaper).

/// Everything that can go wrong building a [`super::SceneRenderer`] from a
/// resolved [`kirie_scene::SceneModel`].
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    /// The resolved scene has no drawable object (all skipped / unsupported) —
    /// the renderer would present only the clear color.
    #[error("scene has no renderable objects")]
    NoRenderableObjects,

    /// The scene's computed projection size is degenerate (zero on an axis),
    /// so no valid render target can be allocated
    /// (docs/render-architecture.md §5).
    #[error("scene projection size is degenerate ({width}x{height})")]
    BadProjection {
        /// Computed projection width.
        width: u32,
        /// Computed projection height.
        height: u32,
    },
}
