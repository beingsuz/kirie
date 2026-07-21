//! Texture upload and the pass texture-name → GPU resource resolution
//! (docs/render-architecture.md §6 name rule, §10 `.tex` upload).
//!
//! A bare pass texture name `X` resolves to the file `materials/X.tex` in the
//! scene container (docs §10, `AssetLocator.cpp:72-79`); names prefixed `_rt_`
//! or `_alias_` are FBO references resolved by the renderer instead
//! (docs §6). Each `.tex` is decoded via the shared [`crate::ImageContent`]
//! path (its first page — animated-atlas frame advance is a documented seam
//! here) and uploaded once. A shader sampler with no bound/resolvable texture
//! falls back to the built-in 1×1 white texture (the reference's `util/white`
//! default, docs §8.2).

use std::collections::HashMap;

use kirie_scene::resolve::AssetSource;

use crate::content::ImageContent;
use crate::error::RenderError;

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

pub struct TextureRegistry {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cache: HashMap<String, Option<std::sync::Arc<GpuTexture>>>,
    white: std::sync::Arc<GpuTexture>,
    /// Video-backed textures kept playing; the renderer takes these after build
    /// ([`Self::take_videos`]) and streams frames per render tick.
    videos: Vec<VideoTexture>,
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
            cache: HashMap::new(),
            videos: Vec::new(),
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
    pub fn get(&mut self, name: &str, source: &dyn AssetSource) -> std::sync::Arc<GpuTexture> {
        if let Some(entry) = self.cache.get(name) {
            return entry.clone().unwrap_or_else(|| self.white.clone());
        }
        let loaded = self.load(name, source);
        self.cache.insert(name.to_string(), loaded.clone());
        loaded.unwrap_or_else(|| self.white.clone())
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
    pub fn get_sprite_frame0(&mut self, name: &str, source: &dyn AssetSource) -> std::sync::Arc<GpuTexture> {
        let key = format!("\u{0}f0:{name}");
        if let Some(entry) = self.cache.get(&key) {
            return entry.clone().unwrap_or_else(|| self.white.clone());
        }
        let cropped = self.load_frame0(name, source);
        if let Some(t) = cropped {
            self.cache.insert(key, Some(t.clone()));
            return t;
        }
        // Not an atlas (or undecodable): reuse the ordinary upload path.
        self.get(name, source)
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

    fn load(&mut self, name: &str, source: &dyn AssetSource) -> Option<std::sync::Arc<GpuTexture>> {
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
        // a single still frame is cropped here; animated atlases keep 0..1 (the
        // per-frame placement is a documented seam, not applied to the layer
        // geometry). `axes = [realW/texW, 0, 0, realH/texH]` for a still.
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
        Some(std::sync::Arc::new(gpu))
    }

    /// Decode the **first frame** of an MP4 video texture and upload it as a
    /// static still (docs/format-tex.md §7.3). Per-frame playback is a
    /// documented seam; a real frame is still vastly closer to the reference
    /// than the 1×1 white fallback, which turns a video base layer into a
    /// full-screen white sheet (the dominant cause of "all-white" scene
    /// renders in the corpus). Any failure returns `None` → white fallback, so
    /// this branch can never render worse than before.
    fn load_video_first_frame(&mut self, name: &str, tex_bytes: &[u8]) -> Option<std::sync::Arc<GpuTexture>> {
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
            self.videos.push(VideoTexture {
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
        std::mem::take(&mut self.videos)
    }
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
    }
}
