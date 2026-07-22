//! Texture upload and the pass texture-name → GPU resource resolution
//! (docs/render-architecture.md §6 name rule, §10 `.tex` upload).
//!
//! A bare pass texture name `X` resolves to the file `materials/X.tex` in the
//! scene container (docs §10, `AssetLocator.cpp:72-79`); names prefixed `_rt_`
//! or `_alias_` are FBO references resolved by the renderer instead
//! (docs §6). Each `.tex` is decoded via the shared [`crate::ImageContent`]
//! path and uploaded once; a multi-frame animated `.tex` additionally
//! registers an [`AtlasTexture`] so the renderer can advance its frames per
//! tick exactly like the reference (`CPass.cpp:348-378`
//! `resolveTextureAnimationState`). A shader sampler with no bound/resolvable
//! texture falls back to the built-in 1×1 white texture (the reference's
//! `util/white` default, docs §8.2).

use std::collections::HashMap;

use kirie_scene::resolve::AssetSource;

use crate::content::{FramePlacement, ImageContent, ImagePage};
use crate::error::RenderError;
use crate::schedule::FrameSchedule;

/// One uploaded texture with its sampler and mip-0 dimensions.
#[derive(Debug)]
pub struct GpuTexture {
    /// The uploaded color texture.
    pub texture: wgpu::Texture,
    /// Its default view.
    pub view: wgpu::TextureView,
    /// Its sampler (filter/wrap from the `.tex` flags, docs §10).
    pub sampler: wgpu::Sampler,
    /// mip-0 width (the uploaded/padded texture width).
    pub width: u32,
    /// mip-0 height (the uploaded/padded texture height).
    pub height: u32,
    /// The frame-0 UV crop `(realW/texW, realH/texH)` that trims NPOT padding
    /// baked into the `.tex` page — the reference's `texcoordCopy = realSize /
    /// textureSize` for the layer's first pass (docs/render-architecture.md
    /// §7.1; docs/format-tex.md §8.1). `[1, 1]` when the page is already at real
    /// size (FreeImage-path decodes, the white fallback, or an atlas the still
    /// crop does not apply to). Sampling the layer texture 0..1 without this
    /// leak the padding region (a solid block) into the composited layer.
    pub uv_crop: [f32; 2],
    /// The logical (real) content size reported as `g_TextureNResolution.zw`:
    /// the `.tex` header crop for stills, `gifWidth/gifHeight` for animated
    /// atlases — exactly the reference's `CTexture::setupResolution`
    /// (`CTexture.cpp:149-153`: animated → `{texW, texH, gifW, gifH}`).
    /// Defaults to the uploaded page size.
    pub real_size: [f32; 2],
}

/// A name-keyed cache of uploaded pass textures plus the fallback white texture
/// (docs §6, §8.2). One registry per scene build.
/// A live video-backed `.tex`: the decode thread keeps producing frames and the
/// renderer streams the newest into `gpu.texture` each frame (the reference
/// plays these; a frozen first frame was the 3445942378 divergence).
pub struct VideoTexture {
    /// The playing decoder (silent, wall-clock paced, seamless loop). Dropping
    /// it stops the decode thread.
    pub player: kirie_video::VideoPlayer,
    /// The sampled texture every pass bound — updated in place.
    pub gpu: std::sync::Arc<GpuTexture>,
    /// Frame dimensions at allocation (upload guard).
    pub size: (u32, u32),
}

/// A multi-frame animated `.tex` (spritesheet atlas or gif-style multi-page,
/// docs/format-tex.md §8-§9). The reference animates these per pass: it picks
/// the frame with the `fmod(renderTime, Σ frametime)` walk, binds the frame's
/// page (`bindTextureUnit(0, texture, frame.frameNumber)`) and feeds the
/// frame's placement to `g_Texture0Translation` / `g_Texture0Rotation`
/// (`CPass.cpp:287-306, 348-378`; `CRenderable.cpp:31-36`). kirie mirrors the
/// [`VideoTexture`] pattern: all passes keep binding one [`GpuTexture`]; the
/// renderer streams the current frame's page into it on page change and drives
/// the placement builtins from [`AtlasTexture::placement_at`] each frame.
pub struct AtlasTexture {
    /// Playback frames in file order (docs/format-tex.md §8.1).
    pub frames: Vec<FramePlacement>,
    /// The §8.1 frametime walk over `frames` (the reference's
    /// `resolveTextureAnimationState` selection, `CPass.cpp:355-365`).
    pub schedule: FrameSchedule,
    /// Decoded CPU pages for multi-page (gif-style) textures, streamed into
    /// `gpu` when the displayed frame's page changes — the wgpu equivalent of
    /// the reference's `glBindTexture(textureID[frameNumber])` page switch
    /// (docs/format-tex.md §9). Empty for single-page spritesheets, whose
    /// animation is placement-only.
    pub pages: Vec<ImagePage>,
    /// The texture every pass bound — holds the current frame's page.
    pub gpu: std::sync::Arc<GpuTexture>,
}

impl AtlasTexture {
    /// The frame displayed at `elapsed` wall-clock seconds — the reference's
    /// `fmod(renderTime, animationTime)` frametime walk (`CPass.cpp:355-365`).
    #[must_use]
    pub fn placement_at(&self, elapsed: f64) -> &FramePlacement {
        // `frame_at` is always in range for a non-empty table; registration
        // guarantees at least two frames (SPEC.md §V9 guard regardless).
        let index = self.schedule.frame_at(elapsed).min(self.frames.len() - 1);
        &self.frames[index]
    }
}

/// A name-keyed cache of uploaded pass textures plus the fallback white texture
/// (docs §6, §8.2). One registry per scene build.
pub struct TextureRegistry {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Name → per-entry once-cell. Shared-`&self` so the object-build loop can
    /// run in parallel: the brief map lock only hands out the cell; the decode
    /// + upload happen inside `OnceLock::get_or_init`, so two threads wanting
    /// the SAME texture dedupe (one loads, the other waits) while different
    /// textures load fully concurrently.
    cache: std::sync::Mutex<
        HashMap<String, std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<GpuTexture>>>>>,
    >,
    white: std::sync::Arc<GpuTexture>,
    /// Video-backed textures kept playing; the renderer takes these after build
    /// ([`Self::take_videos`]) and streams frames per render tick.
    videos: std::sync::Mutex<Vec<VideoTexture>>,
    /// Animated atlases by texture name; objects look up their layer's atlas
    /// ([`Self::atlas_for`]) and the renderer takes the list after build
    /// ([`Self::take_atlases`]) to advance pages per render tick.
    atlases: std::sync::Mutex<HashMap<String, std::sync::Arc<AtlasTexture>>>,
}

impl TextureRegistry {
    /// Build an empty registry and its 1×1 white fallback.
    ///
    /// Textures upload as raw `Rgba8Unorm` (no sRGB decode on sample), matching
    /// the reference's gamma-naive `GL_RGBA8` path (docs/render-architecture.md
    /// §10): the whole scene pipeline works in raw/gamma space — raw sample →
    /// `Rgba16Float` FBO → raw blit — and the final blit cancels the output
    /// surface's sRGB encode (see `renderer::build_blit`) so bytes round-trip
    /// unchanged and multi-pass blends composite in the same space as the
    /// oracle. Decoding here instead would blend effect passes in *linear*
    /// space and diverge from the reference on every additive/translucent pass.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let white = std::sync::Arc::new(upload_rgba8(
            device,
            queue,
            "kirie-white",
            1,
            1,
            &[255, 255, 255, 255],
            true,
            true,
        ));
        TextureRegistry {
            device: device.clone(),
            queue: queue.clone(),
            cache: std::sync::Mutex::new(HashMap::new()),
            videos: std::sync::Mutex::new(Vec::new()),
            atlases: std::sync::Mutex::new(HashMap::new()),
            white,
        }
    }

    /// The fallback white texture.
    #[must_use]
    pub fn white(&self) -> std::sync::Arc<GpuTexture> {
        self.white.clone()
    }

    /// Resolve a bare texture `name` to an uploaded texture, loading it from
    /// `source` on first use. Returns the white fallback when the file is
    /// absent or fails to decode (never an error, docs §8.2 fallback chain).
    pub fn get(&self, name: &str, source: &dyn AssetSource) -> std::sync::Arc<GpuTexture> {
        let slot = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(name.to_string())
            .or_default()
            .clone();
        slot.get_or_init(|| self.load(name, source))
            .clone()
            .unwrap_or_else(|| self.white.clone())
    }

    /// Resolve a **particle sprite** texture, cropped to spritesheet frame 0.
    ///
    /// WE particle materials routinely point at a multi-frame atlas (e.g.
    /// `particle/bubbles/bubble3` is an 8×8 grid of 128² frames). The instanced
    /// sprite renderer samples the whole bound texture `0..1`, so binding the
    /// full atlas makes every particle draw the entire grid — a static block of
    /// tiny sprites instead of one bubble. Per-frame animation is a documented
    /// seam (particles hold frame 0 here and in the still oracle screenshot), so
    /// uploading just frame 0's sub-rect is the faithful still. A non-atlas or
    /// rotated-frame texture returns `None`, and the caller falls back to the
    /// ordinary [`Self::get`] upload.
    pub fn get_sprite_frame0(&self, name: &str, source: &dyn AssetSource) -> std::sync::Arc<GpuTexture> {
        let key = format!("\u{0}f0:{name}");
        let slot = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_default()
            .clone();
        match slot.get_or_init(|| self.load_frame0(name, source)) {
            Some(t) => t.clone(),
            // Not an atlas (or undecodable): reuse the ordinary upload path.
            None => self.get(name, source),
        }
    }

    /// Decode `name`'s `.tex` and, if it is a multi-frame atlas with an
    /// axis-aligned frame 0, upload just that frame's sub-rect. `None` means
    /// "not a croppable atlas" — the caller should take the normal path.
    fn load_frame0(&self, name: &str, source: &dyn AssetSource) -> Option<std::sync::Arc<GpuTexture>> {
        let path = format!("materials/{name}.tex");
        let bytes = source.load(&path)?;
        let content = ImageContent::from_tex_bytes(&bytes).ok()?;
        if content.frames.len() <= 1 {
            return None; // a plain still — ordinary path applies the NPOT crop.
        }
        let page = content.pages.first()?;
        let fr = content.frames.first()?;
        // Only axis-aligned frames (no 90° atlas rotation) crop trivially.
        if fr.axes[1].abs() > 1e-4 || fr.axes[2].abs() > 1e-4 {
            return None;
        }
        let (tw, th) = (page.width as i64, page.height as i64);
        let fx = (fr.translation[0] * tw as f32).round() as i64;
        let fy = (fr.translation[1] * th as f32).round() as i64;
        let fw = ((fr.axes[0] * tw as f32).round() as i64).max(1);
        let fh = ((fr.axes[3] * th as f32).round() as i64).max(1);
        if fx < 0 || fy < 0 || fx + fw > tw || fy + fh > th {
            return None;
        }
        let (fx, fy, fw, fh) = (fx as usize, fy as usize, fw as usize, fh as usize);
        let stride = page.width as usize * 4;
        let mut cropped = Vec::with_capacity(fw * fh * 4);
        for row in 0..fh {
            let start = (fy + row) * stride + fx * 4;
            cropped.extend_from_slice(&page.pixels[start..start + fw * 4]);
        }
        // Clamp so linear filtering never bleeds neighbouring atlas cells.
        let gpu = upload_rgba8(
            &self.device,
            &self.queue,
            name,
            fw as u32,
            fh as u32,
            &cropped,
            content.sampler.nearest,
            true,
        );
        Some(std::sync::Arc::new(gpu))
    }

    fn load(&self, name: &str, source: &dyn AssetSource) -> Option<std::sync::Arc<GpuTexture>> {
        let path = format!("materials/{name}.tex");
        let bytes = source.load(&path)?;
        let content = match ImageContent::from_tex_bytes(&bytes) {
            Ok(c) => c,
            // A `.tex` wrapping an MP4 (docs/format-tex.md §7.3): the scene
            // renderer can't animate it, but a still frame beats the 1×1 white
            // fallback that would paint the whole layer white. Decode frame 0.
            Err(RenderError::VideoTex) => return self.load_video_first_frame(name, &bytes),
            Err(e) => {
                tracing::debug!(texture = %name, error = %e, "texture decode failed; using white");
                return None;
            }
        };
        let page = content.pages.first()?;
        // The frame-0 UV crop trims NPOT padding for a still image (the
        // reference's `texcoordCopy = realSize / textureSize`, docs §7.1). Only
        // a single still frame is cropped here; animated atlases keep 0..1
        // exactly like the reference ("animations should be copied completely",
        // `CImage.cpp:308-316`) — their per-frame placement is applied by the
        // shader via `g_Texture0Translation`/`g_Texture0Rotation` instead.
        // `axes = [realW/texW, 0, 0, realH/texH]` for a still.
        let uv_crop = match content.frames.as_slice() {
            [only] => [only.axes[0], only.axes[3]],
            _ => [1.0, 1.0],
        };
        let mut gpu = upload_rgba8(
            &self.device,
            &self.queue,
            name,
            page.width,
            page.height,
            &page.pixels,
            content.sampler.nearest,
            content.sampler.clamp_uvs,
        );
        gpu.uv_crop = uv_crop;
        // `g_TextureNResolution.zw` is the logical content size: the header
        // crop for stills, `gifWidth/gifHeight` for animated atlases
        // (`CTexture.cpp:149-153` `setupResolution`).
        gpu.real_size = [content.content_width as f32, content.content_height as f32];
        let gpu = std::sync::Arc::new(gpu);
        self.register_atlas(name, content, &gpu);
        Some(gpu)
    }

    /// Register a decoded multi-frame `.tex` for per-tick animation (the
    /// reference's `CPass` texture-animation state, `CPass.cpp:348-378`).
    /// Multi-page (gif-style) content keeps its CPU pages for page streaming;
    /// pages whose dimensions differ from page 0 cannot stream into the one
    /// bound texture, so such content stays a static frame 0 (SPEC.md §V9 —
    /// malformed input degrades, never breaks).
    fn register_atlas(&self, name: &str, content: ImageContent, gpu: &std::sync::Arc<GpuTexture>) {
        let Some(multi_page) = atlas_animates(&content) else {
            if content.frames.len() > 1 {
                tracing::debug!(texture = %name, "animated .tex not streamable; keeping static frame 0");
            }
            return;
        };
        let schedule = content.schedule();
        self.atlases.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(
            name.to_string(),
            std::sync::Arc::new(AtlasTexture {
                frames: content.frames,
                schedule,
                // Single-page spritesheets never re-upload; drop their pixels.
                pages: if multi_page { content.pages } else { Vec::new() },
                gpu: gpu.clone(),
            }),
        );
    }

    /// The animated atlas registered for texture `name`, if any. Objects whose
    /// layer texture is animated hold this to drive the per-pass
    /// `g_Texture0Translation`/`g_Texture0Rotation` builtins (docs §8.3).
    #[must_use]
    pub fn atlas_for(&self, name: &str) -> Option<std::sync::Arc<AtlasTexture>> {
        self.atlases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
            .cloned()
    }

    /// Decode the **first frame** of an MP4 video texture and upload it as a
    /// static still (docs/format-tex.md §7.3). Per-frame playback is a
    /// documented seam; a real frame is still vastly closer to the reference
    /// than the 1×1 white fallback, which turns a video base layer into a
    /// full-screen white sheet (the dominant cause of "all-white" scene
    /// renders in the corpus). Any failure returns `None` → white fallback, so
    /// this branch can never render worse than before.
    fn load_video_first_frame(&self, name: &str, tex_bytes: &[u8]) -> Option<std::sync::Arc<GpuTexture>> {
        use std::time::Duration;

        let tex = kirie_formats::tex::Tex::parse(tex_bytes).ok()?;
        let payload = tex.video_payload().ok()?;
        // kirie-video decodes from a path; stage the embedded MP4 in a temp
        // file keyed by a hash of its bytes so concurrent scenes never collide.
        let mut key: u64 = 0xcbf2_9ce4_8422_2325;
        for b in &*payload {
            key = (key ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let file = std::env::temp_dir().join(format!("kirie-vtex-{key:016x}.mp4"));
        std::fs::write(&file, &payload).ok()?;

        let opened = kirie_video::VideoPlayer::open(
            &file,
            kirie_video::VideoOptions {
                enable_audio: false,
                silent: true,
                ..kirie_video::VideoOptions::default()
            },
        );
        let (player, frame) = match opened {
            Ok((player, _control)) => {
                // The decode thread fills the bounded queue independently of the
                // clock; frame 0 arrives promptly.
                let frame = player.recv_frame_timeout(Duration::from_secs(5));
                (Some(player), frame)
            }
            Err(e) => {
                tracing::debug!(texture = %name, error = %e, "video texture open failed; using white");
                (None, None)
            }
        };
        // The decoder holds an open fd; unlinking the staged file is safe and
        // keeps the temp dir clean even while playback continues.
        let _ = std::fs::remove_file(&file);
        let frame = frame?;
        if frame.width == 0 || frame.height == 0 {
            return None;
        }

        // Frame pixels are top-row-first RGBA8 (kirie-video's converter), the
        // same layout `upload_rgba8` expects. A video frame is already at real
        // size, so no NPOT crop; honor the `.tex` sampler flags for wrap/filter.
        let gpu = std::sync::Arc::new(upload_rgba8(
            &self.device,
            &self.queue,
            name,
            frame.width,
            frame.height,
            &frame.data,
            tex.flags.no_interpolation(),
            tex.flags.clamp_uvs(),
        ));
        // Keep the player: the renderer streams later frames into this same
        // texture every tick (the reference PLAYS video .tex, docs §10).
        if let Some(player) = player {
            self.videos.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(VideoTexture {
                player,
                gpu: gpu.clone(),
                size: (frame.width, frame.height),
            });
        }
        Some(gpu)
    }

    /// Hand the live video textures to the renderer (called once after build;
    /// the registry is discarded afterwards).
    pub fn take_videos(&mut self) -> Vec<VideoTexture> {
        std::mem::take(
            &mut *self.videos.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Hand the animated atlases to the renderer (called once after build; the
    /// registry is discarded afterwards). The renderer advances each atlas per
    /// render tick and streams multi-page frames into the bound texture.
    pub fn take_atlases(&mut self) -> Vec<std::sync::Arc<AtlasTexture>> {
        std::mem::take(
            &mut *self.atlases.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        )
        .into_values()
        .collect()
    }
}

/// Whether decoded content animates as an atlas, and how: `Some(false)` — a
/// single-page spritesheet (placement-only animation, no page uploads);
/// `Some(true)` — a multi-page (gif-style) texture whose uniform-size pages
/// can stream into the one bound texture (the reference's per-frame
/// `textureID[frameNumber]` bind, docs/format-tex.md §9); `None` — static
/// (one frame, a zero-duration table the reference would `fmod` into NaN on,
/// or pages of differing sizes that cannot stream — SPEC.md §V9 degrade).
fn atlas_animates(content: &ImageContent) -> Option<bool> {
    if content.frames.len() <= 1 || !content.schedule().is_animated() {
        return None;
    }
    let multi_page = content.frames.iter().any(|f| f.page != 0);
    if multi_page {
        let (w0, h0) = (content.pages[0].width, content.pages[0].height);
        if content.pages.iter().any(|p| (p.width, p.height) != (w0, h0)) {
            return None;
        }
    }
    Some(multi_page)
}

/// Upload a tightly-packed RGBA8 buffer as a sampled texture with a matching
/// sampler (docs §10: `NoInterpolation` → nearest; `ClampUVs` → clamp-to-edge
/// else repeat).
#[allow(clippy::too_many_arguments)]
fn upload_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    pixels: &[u8],
    nearest: bool,
    clamp: bool,
) -> GpuTexture {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Guard against a short buffer (SPEC.md §V9): only upload what fits.
    let need = (width * height * 4) as usize;
    if pixels.len() >= need {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels[..need],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let filter = if nearest {
        wgpu::FilterMode::Nearest
    } else {
        wgpu::FilterMode::Linear
    };
    let address = if clamp {
        wgpu::AddressMode::ClampToEdge
    } else {
        wgpu::AddressMode::Repeat
    };
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: filter,
        min_filter: filter,
        ..wgpu::SamplerDescriptor::default()
    });
    GpuTexture {
        texture,
        view,
        sampler,
        width,
        height,
        uv_crop: [1.0, 1.0],
        real_size: [width as f32, height as f32],
    }
}

#[cfg(test)]
mod tests {
    use crate::content::SamplerSpec;

    use super::*;

    fn content(pages: Vec<(u32, u32)>, frames: Vec<(usize, f32)>) -> ImageContent {
        ImageContent {
            pages: pages
                .into_iter()
                .map(|(width, height)| ImagePage {
                    width,
                    height,
                    pixels: vec![0; (width * height * 4) as usize],
                })
                .collect(),
            frames: frames
                .into_iter()
                .map(|(page, duration)| FramePlacement {
                    page,
                    duration,
                    translation: [0.0, 0.0],
                    axes: [1.0, 0.0, 0.0, 1.0],
                })
                .collect(),
            sampler: SamplerSpec {
                nearest: false,
                clamp_uvs: true,
            },
            content_width: 4,
            content_height: 4,
        }
    }

    #[test]
    fn spritesheets_animate_without_page_streaming() {
        // All frames on one page: placement-only animation (the reference's
        // g_Texture0Translation/Rotation path, CPass.cpp:287-306).
        let c = content(vec![(8, 8)], vec![(0, 0.1), (0, 0.1)]);
        assert_eq!(atlas_animates(&c), Some(false));
    }

    #[test]
    fn uniform_multi_page_gifs_stream_pages() {
        // Frames on different equal-size pages: the reference's
        // textureID[frameNumber] bind (docs/format-tex.md §9).
        let c = content(vec![(4, 4), (4, 4)], vec![(0, 0.1), (1, 0.1)]);
        assert_eq!(atlas_animates(&c), Some(true));
    }

    #[test]
    fn static_and_malformed_content_never_animates() {
        // One frame: a still.
        let single = content(vec![(4, 4)], vec![(0, 0.0)]);
        assert_eq!(atlas_animates(&single), None);
        // All-zero durations: the reference would fmod(t, 0) into NaN (V9).
        let zero = content(vec![(4, 4)], vec![(0, 0.0), (0, 0.0)]);
        assert_eq!(atlas_animates(&zero), None);
        // Mismatched page sizes cannot stream into one texture (V9 degrade).
        let mismatched = content(vec![(4, 4), (8, 8)], vec![(0, 0.1), (1, 0.1)]);
        assert_eq!(atlas_animates(&mismatched), None);
    }
}
