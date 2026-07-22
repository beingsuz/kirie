//! kirie-render — wallpaper renderers on wgpu.
//!
//! First slice: still-image and animated-gif wallpapers.
//!
//! * [`ImageContent`] decodes plain image files (png/jpg/bmp/gif via the
//!   `image` crate) and Wallpaper Engine `.tex` containers — including
//!   animated `TEXS` atlases with per-frame rects, frametime pacing and
//!   fmod looping (docs/format-tex.md §8) — into RGBA pages + frame
//!   placements.
//! * [`FrameSchedule`] is the reference's frametime walk
//!   (docs/format-tex.md §8.1).
//! * [`ScalingMode`]/[`ClampMode`]/[`UvWindow`] are THE shared
//!   output-scaling implementation (docs/render-architecture.md §4,
//!   docs/compat-cli.md §3.1) — kirie-video reuses these.
//! * [`ImageRenderer`] implements [`kirie_platform::Renderer`]: pages are
//!   uploaded once, animation switches prebuilt bind groups and rewrites a
//!   32-byte uniform only on frame change (SPEC §V5); static content
//!   reports "no further frames" via
//!   [`ImageRenderer::time_until_frame_change`] (SPEC §V6 hint for the
//!   presentation layer's redraw scheduling).
//!
//! Loading returns typed [`RenderError`]s and never panics on malformed
//! input (SPEC §V9). No globals; every piece of state is owned and passed
//! explicitly (SPEC §V1).

#![forbid(unsafe_code)]

mod content;
mod error;
pub mod media;
pub mod particle;
mod renderer;
mod scaling;
pub mod scene;
mod schedule;

pub use content::{FramePlacement, ImageContent, ImagePage, SamplerSpec};
pub use error::RenderError;
pub use media::{
    AlbumArt, MediaConfig, MediaPlaybackEvent, MediaSource, MediaState, MediaStatus, PlaybackState,
    TrackMetadata,
};
pub use renderer::{ImageOptions, ImageRenderer};
pub use scaling::{ClampMode, ScalingMode, UvWindow};
pub use scene::{
    SceneError, SceneLoadError, SceneOptions, SceneRenderer, load_workshop_scene, start_background_prebake,
};
pub use schedule::FrameSchedule;
