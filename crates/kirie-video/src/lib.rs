//! kirie-video — ffmpeg-based video wallpaper playback (SPEC T10).
//!
//! Reimplements the *behavior contract* of the C++ reference's libmpv
//! player (docs/subsystems-misc.md §2, "VideoPlayback") without mpv:
//!
//! * infinite seamless loop (`loop=inf`, §2.1) — EOF seeks back to 0 on a
//!   monotonic playback timeline;
//! * native-size RGBA8 frame texture that resizes only when the stream
//!   geometry changes (§2.2);
//! * live volume (0–100) / mute / pause; `--silent` plays with volume 0
//!   (§2.1, §2.3); speed with the reference's "pre-start speed is lost"
//!   quirk documented at [`VideoOptions`];
//! * audio clock is playback master when the file has audio, wall clock ×
//!   speed otherwise (§2.1 pacing);
//! * output scaling fill/fit/stretch/default per
//!   docs/render-architecture.md §4.
//!
//! Architecture (SPEC V1/V3/V4/V5/V6): per video, one video-decode thread
//! and (optionally) one audio thread, connected to the render side only by
//! bounded channels, an SPSC sample ring and immutable `triple_buffer`
//! snapshots. The renderer ([`VideoRenderer`], implementing
//! [`kirie_platform::Renderer`]) never blocks on decode, drops late
//! frames, recycles pixel buffers (no steady-state allocation) and does
//! zero work while the compositor withholds frame callbacks.
//!
//! CPU decode only for now: hardware decode (VAAPI dma-buf import) is
//! SPEC T11 and slots into the seam marked in `decode.rs`.

#![deny(unsafe_code)] // SPEC V2: unsafe reserved for the future dma-buf import (T11).

mod audio;
mod clock;
mod decode;
mod error;
mod pacing;
mod player;
mod renderer;
mod scaling;

pub use decode::{DecodedFrame, FRAME_QUEUE_CAP, VideoInfo};
pub use error::VideoError;
pub use pacing::{LoopTimeline, Pacer, PacerStats, Timed};
pub use player::{VideoControl, VideoOptions, VideoPlayer};
pub use renderer::VideoRenderer;
pub use scaling::{ScalingMode, UvRect, compute_uvs};
