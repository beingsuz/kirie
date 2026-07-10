//! The immutable spectrum snapshot the renderer binds to `g_AudioSpectrum*`.

use crate::dsp::{BANDS_16, BANDS_32, BANDS_64, Smoother};

/// One frame of normalized FFT bands, published lock-free via `arc-swap` and
/// read by the render/uniform packer without blocking (V4).
///
/// The source is mono (PulseAudio monitor, 1 channel), so Left == Right: the
/// uniform packer feeds `audio16` to both `g_AudioSpectrum16Left` and
/// `g_AudioSpectrum16Right`, and likewise 32/64 (docs/render-architecture.md
/// §8.3; subsystems-misc.md §1.3 "Consumers").
#[derive(Clone, Debug, PartialEq)]
pub struct AudioSpectrum {
    /// 16-band spectrum (`g_AudioSpectrum16Left/Right`).
    pub audio16: [f32; BANDS_16],
    /// 32-band spectrum (`g_AudioSpectrum32Left/Right`).
    pub audio32: [f32; BANDS_32],
    /// 64-band spectrum (`g_AudioSpectrum64Left/Right`).
    pub audio64: [f32; BANDS_64],
}

impl AudioSpectrum {
    /// The silent / no-audio fallback: all bands zero (the exact state a
    /// wallpaper sees when audio processing is off — cpp arrays stay zeroed,
    /// subsystems-misc.md §1.3 porting note).
    #[must_use]
    pub const fn silent() -> Self {
        Self {
            audio16: [0.0; BANDS_16],
            audio32: [0.0; BANDS_32],
            audio64: [0.0; BANDS_64],
        }
    }

    /// Fetch the mono band array for a resolution (16/32/64). Any other value
    /// falls back to 64 to mirror the C++ `registerAudioBuffers` default.
    #[must_use]
    pub fn bands(&self, resolution: usize) -> &[f32] {
        match resolution {
            16 => &self.audio16,
            32 => &self.audio32,
            _ => &self.audio64,
        }
    }
}

impl Default for AudioSpectrum {
    fn default() -> Self {
        Self::silent()
    }
}

impl From<&Smoother> for AudioSpectrum {
    fn from(s: &Smoother) -> Self {
        Self {
            audio16: s.b16,
            audio32: s.b32,
            audio64: s.b64,
        }
    }
}
