//! VAAPI hardware decode plumbing (SPEC T11), compiled only with the
//! `vaapi` cargo feature.
//!
//! Scope: hardware *decode* plus hw-surface → system-memory download
//! ([`HwDownload`]), feeding the existing sws RGBA path in `decode.rs`.
//! True zero-copy (VAAPI surface → dma-buf → imported wgpu texture) is
//! deliberate follow-up work; the win here is taking the decode itself off
//! the CPU.
//!
//! # Why this module holds unsafe code
//!
//! ffmpeg-next 8 exposes **no safe surface** for `AVCodecContext::
//! hw_device_ctx`, `av_hwdevice_ctx_create` or `av_hwframe_transfer_data`
//! (its safe API stops at software scaling/resampling). The crate-level
//! `deny(unsafe_code)` — whose lib.rs comment reserves unsafe for exactly
//! this seam — is therefore relaxed to an `allow` for this module alone;
//! every unsafe block carries a SAFETY comment and touches nothing but the
//! four FFI calls above plus the raw-pointer accessors they require.
//!
//! # How decode engages
//!
//! [`attach_vaapi`] runs against the *not yet opened* codec context. With
//! `hw_device_ctx` set, FFmpeg's default `get_format` (the generic-hwaccel
//! path, libavcodec ≥ 3.4) selects `AV_PIX_FMT_VAAPI` when the codec +
//! driver support the stream's profile — and falls back to a software
//! format *inside FFmpeg* when hwaccel setup fails at decode time, so a
//! device that turns out not to handle the profile still plays via the
//! CPU decoder without any action on our side.

#![allow(unsafe_code)] // Sole exception to the crate's deny; see module docs.

use ffmpeg_next as ffmpeg;
use ffmpeg_next::ffi;
use ffmpeg_next::format::Pixel;

/// Why VAAPI could not be attached. Only ever logged at info level: every
/// variant degrades to the existing CPU decode path (SPEC V9).
#[derive(Debug, thiserror::Error)]
pub(crate) enum HwAttachError {
    /// No decoder registered for the stream's codec id (the CPU open will
    /// surface the real `DecoderNotFound` right after).
    #[error("no decoder for {0:?}")]
    DecoderNotFound(ffmpeg::codec::Id),
    /// The decoder does not advertise VAAPI via the hw-device-ctx method.
    #[error("codec {0} has no VAAPI hw-device support")]
    Unsupported(String),
    /// `av_hwdevice_ctx_create` failed (no render node, no driver, …).
    #[error("VAAPI device creation failed: {0}")]
    Device(ffmpeg::Error),
}

/// Create a VAAPI hw device on the default DRM render node and attach it
/// to the not-yet-opened codec context, so `avcodec_open2` + the default
/// `get_format` enable the VAAPI hwaccel (see module docs).
///
/// On any `Err` the context is left untouched and fully usable for the
/// plain CPU open.
pub(crate) fn attach_vaapi(ctx: &mut ffmpeg::codec::context::Context) -> Result<(), HwAttachError> {
    let id = ctx.id();
    // Same lookup `Context::decoder().video()` performs, so the hw-config
    // check below inspects the codec that will actually be opened.
    let codec = ffmpeg::codec::decoder::find(id).ok_or(HwAttachError::DecoderNotFound(id))?;

    // Attaching a device to a codec without a matching hw config would be
    // silently ignored; check first so the fallback log tells the truth.
    let mut supported = false;
    for index in 0.. {
        // SAFETY: `codec.as_ptr()` is the valid, program-lifetime AVCodec
        // returned by avcodec_find_decoder; avcodec_get_hw_config is
        // documented to return NULL once `index` passes the last config,
        // and any non-NULL result points at static data owned by
        // libavcodec, valid to dereference for the program's lifetime.
        let config = unsafe { ffi::avcodec_get_hw_config(codec.as_ptr(), index).as_ref() };
        let Some(config) = config else { break };
        if config.device_type == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI
            && config.methods & (ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0
        {
            supported = true;
            break;
        }
    }
    if !supported {
        return Err(HwAttachError::Unsupported(codec.name().to_owned()));
    }

    let mut device: *mut ffi::AVBufferRef = std::ptr::null_mut();
    // SAFETY: `&mut device` is a valid out-pointer; NULL device path + NULL
    // options ask libavutil to open the default DRM render node. On failure
    // the out-pointer is left NULL, so there is nothing to free.
    let ret = unsafe {
        ffi::av_hwdevice_ctx_create(
            &mut device,
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        )
    };
    if ret < 0 {
        return Err(HwAttachError::Device(ffmpeg::Error::from(ret)));
    }

    // SAFETY: `ctx` wraps a live, not-yet-opened AVCodecContext (owned by
    // ffmpeg-next, freed via avcodec_free_context). Storing our sole
    // AVBufferRef in `hw_device_ctx` transfers ownership to the codec
    // context — avcodec_free_context unrefs it — so no reference leaks and
    // no raw pointer outlives the context on our side.
    unsafe {
        (*ctx.as_mut_ptr()).hw_device_ctx = device;
    }
    Ok(())
}

/// Per-decode-thread hw → system-memory download state: the reusable
/// destination frame plus one-shot "hw decode actually engaged" logging.
pub(crate) struct HwDownload {
    /// System-memory destination (typically NV12). Its buffer is reused
    /// across frames while the geometry holds, keeping the steady-state
    /// decode loop free of per-frame allocation on our side.
    frame: ffmpeg::frame::Video,
    /// Whether the one-time "hardware decode active" log fired. Attach
    /// success alone does not prove engagement — FFmpeg may still fall
    /// back to software for an unsupported profile — so the proof is the
    /// first frame that arrives as a VAAPI surface.
    announced: bool,
}

impl HwDownload {
    /// Empty state; allocates nothing until the first VAAPI frame.
    pub(crate) fn new() -> Self {
        Self {
            frame: ffmpeg::frame::Video::empty(),
            announced: false,
        }
    }

    /// If `src` is a VAAPI surface, download it to system memory and
    /// return the software frame; `Ok(None)` when `src` is already a
    /// system-memory frame (hwaccel never engaged, or FFmpeg fell back).
    ///
    /// The download copies pixels only — the caller keeps reading
    /// timestamps off `src`, so `av_frame_copy_props` is not needed.
    pub(crate) fn download(
        &mut self,
        src: &ffmpeg::frame::Video,
    ) -> Result<Option<&ffmpeg::frame::Video>, ffmpeg::Error> {
        if src.format() != Pixel::VAAPI {
            return Ok(None);
        }
        if !self.announced {
            tracing::info!("VAAPI hardware decode active");
            self.announced = true;
        }

        // Reuse the destination buffer only while the geometry matches;
        // on a change, return to the clean state so FFmpeg reallocates at
        // the new size (mirrors the scaler-rebuild policy in decode.rs).
        if self.frame.width() != src.width() || self.frame.height() != src.height() {
            // SAFETY: `self.frame` is a valid owned AVFrame; av_frame_unref
            // returns it to the clean (bufferless) state the allocating
            // transfer path expects.
            unsafe { ffi::av_frame_unref(self.frame.as_mut_ptr()) };
        }
        // SAFETY: dst is a valid owned AVFrame — either clean (FFmpeg
        // allocates it and picks the transfer format) or filled by a
        // previous call with identical geometry/format (direct-copy path);
        // src is a valid decoded VAAPI frame that stays alive across the
        // call. Both pointers come from ffmpeg-next-owned frames.
        let mut ret = unsafe { ffi::av_hwframe_transfer_data(self.frame.as_mut_ptr(), src.as_ptr(), 0) };
        if ret < 0 {
            // A reused destination can go stale (e.g. the driver switched
            // transfer formats); retry once from a clean frame before
            // reporting the frame unconvertible.
            // SAFETY: same as the av_frame_unref above.
            unsafe { ffi::av_frame_unref(self.frame.as_mut_ptr()) };
            // SAFETY: same as the transfer above, with dst now clean.
            ret = unsafe { ffi::av_hwframe_transfer_data(self.frame.as_mut_ptr(), src.as_ptr(), 0) };
        }
        if ret < 0 {
            return Err(ffmpeg::Error::from(ret));
        }
        Ok(Some(&self.frame))
    }
}
