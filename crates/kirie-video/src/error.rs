//! Typed errors for the video playback subsystem (SPEC V9: no panics on
//! malformed input — media files are external input).

use std::path::PathBuf;

use thiserror::Error;

/// Errors surfaced while opening or playing a video wallpaper.
#[derive(Debug, Error)]
pub enum VideoError {
    /// Any libav* level failure (demux, decode, scale, resample, seek).
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    /// The container holds no video stream at all.
    #[error("no video stream in {0}")]
    NoVideoStream(PathBuf),

    /// The stream reports impossible geometry (zero width/height).
    #[error("invalid video dimensions {width}x{height}")]
    InvalidDimensions {
        /// Reported frame width in pixels.
        width: u32,
        /// Reported frame height in pixels.
        height: u32,
    },

    /// No usable audio output device / stream configuration.
    #[error("audio output unavailable: {0}")]
    AudioOutput(String),

    /// The audio device wants a sample format we do not convert to.
    #[error("unsupported audio device sample format: {0}")]
    UnsupportedSampleFormat(String),

    /// Spawning a decode worker thread failed.
    #[error("failed to spawn decode thread: {0}")]
    ThreadSpawn(#[from] std::io::Error),
}
