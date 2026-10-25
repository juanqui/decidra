//! Strict image ingestion matching the Python d3 runtime's server contract.
//!
//! Data URLs only, PNG/JPEG/WebP MIME types, 8 MB encoded, 16 MP source
//! pixels, aspect ratio <= 200, RGB conversion, Qwen2VL smart resize to
//! 65,536..1,638,400 pixels on a 32-pixel grid, bicubic resize, and the exact
//! spatial-merge patch order consumed by the Qwen3.5 vision tower.

use anyhow::{Context, Result, bail};
use base64::Engine;
use half::bf16;
use image::{DynamicImage, ImageReader};

pub const PATCH: usize = 16;
pub const MERGE: usize = 2;
pub const TEMPORAL: usize = 2;
pub const MIN_PIXELS: usize = 65_536;
pub const MAX_PIXELS: usize = 1_638_400;
pub const MAX_ENCODED: usize = 8_000_000;
pub const MAX_SOURCE_PIXELS: u64 = 16_000_000;
pub const MAX_ASPECT: f64 = 200.0;
pub const MAX_IMAGES: usize = 16;
pub const MAX_IMAGE_TOKENS: usize = 8_192;

#[derive(Debug, Clone)]
pub struct ImageMedia {
    pub width: u32,
    pub height: u32,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Normalized patch pixels in spatial-merge order, `[patches, 1536]`.
    pub patches: Vec<bf16>,
    /// Digest of the decoded RGB source, used as the image cache identity.
    pub digest: [u8; 32],
}

impl ImageMedia {
    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }
    pub fn tokens(&self) -> usize {
        self.patches() / (MERGE * MERGE)
    }
    pub fn patch_dim(&self) -> usize {
        3 * TEMPORAL * PATCH * PATCH
    }
}

pub fn decode_data_url(value: &str) -> Result<ImageMedia> {
    let (subtype, payload) = split_data_url(value)?;
    if !matches!(subtype.as_str(), "png" | "jpeg" | "jpg" | "webp") {
        bail!("image data URLs take the types png, jpeg, jpg, webp, got {subtype:?}");
    }
    if payload.len() > 4 * MAX_ENCODED.div_ceil(3) {
        bail!("the encoded image exceeds the {MAX_ENCODED} byte limit");
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .context("the image data URL does not hold valid base64")?;
    if bytes.len() > MAX_ENCODED {
        bail!(
            "the decoded image has {} bytes, over the {MAX_ENCODED} byte limit",
            bytes.len()
        );
    }
    decode_bytes(&bytes)
}

fn split_data_url(value: &str) -> Result<(String, &str)> {
    let (head, payload) = value
        .split_once(',')
        .context("expected a base64 data URL: data:image/<type>;base64,<data>")?;
    let head = head.trim().to_ascii_lowercase();
    let rest = head
        .strip_prefix("data:image/")
        .and_then(|h| h.strip_suffix(";base64"))
        .context("expected a base64 data URL: data:image/<type>;base64,<data>")?;
    Ok((rest.to_owned(), payload))
}

pub fn decode_bytes(bytes: &[u8]) -> Result<ImageMedia> {
    let mut reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("the image could not be identified")?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SOURCE_PIXELS as u32);
    limits.max_image_height = Some(MAX_SOURCE_PIXELS as u32);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode().context("the image could not be decoded")?;
    from_dynamic(image)
}

fn check_source_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("the image has no pixels");
    }
    if u64::from(width) * u64::from(height) > MAX_SOURCE_PIXELS {
        bail!(
            "the image has {} pixels, over the {MAX_SOURCE_PIXELS} pixel source limit",
            u64::from(width) * u64::from(height)
        );
    }
    if width.max(height) as f64 / width.min(height) as f64 > MAX_ASPECT {
        bail!("absolute aspect ratio must be smaller than 200");
    }
    Ok(())
}

pub fn from_dynamic(image: DynamicImage) -> Result<ImageMedia> {
    let (width, height) = (image.width(), image.height());
    check_source_dimensions(width, height)?;
    let rgb = image.to_rgb8();
    let mut digest_hasher = blake3::Hasher::new();
    digest_hasher.update(&width.to_le_bytes());
    digest_hasher.update(&height.to_le_bytes());
    digest_hasher.update(rgb.as_raw());
    let digest = *digest_hasher.finalize().as_bytes();
    let (resized_w, resized_h) = smart_resize(height as usize, width as usize);
    let src = fast_image_resize::images::Image::from_vec_u8(
        width,
        height,
        rgb.into_raw(),
        fast_image_resize::PixelType::U8x3,
    )
    .context("creating the resize source")?;
    let mut dst = fast_image_resize::images::Image::new(
        resized_w as u32,
        resized_h as u32,
        fast_image_resize::PixelType::U8x3,
    );
    let options = fast_image_resize::ResizeOptions::new().resize_alg(
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom),
    );
    fast_image_resize::Resizer::new()
        .resize(&src, &mut dst, Some(&options))
        .context("resizing the image")?;
    let grid_h = resized_h / PATCH;
    let grid_w = resized_w / PATCH;
    if grid_h % MERGE != 0 || grid_w % MERGE != 0 {
        bail!("resized grid {grid_h}x{grid_w} is not divisible by the merge size");
    }
    let patch_dim = 3 * TEMPORAL * PATCH * PATCH;
    let mut patches = vec![bf16::ZERO; grid_h * grid_w * patch_dim];
    let blocks_h = grid_h / MERGE;
    let blocks_w = grid_w / MERGE;
    let mut out = 0usize;
    for bh in 0..blocks_h {
        for bw in 0..blocks_w {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    let py = (bh * MERGE + mh) * PATCH;
                    let px = (bw * MERGE + mw) * PATCH;
                    for c in 0..3 {
                        for t in 0..TEMPORAL {
                            let _ = t; // still images duplicate the same frame
                            for y in 0..PATCH {
                                for x in 0..PATCH {
                                    let p = dst.buffer()[((py + y) * resized_w + px + x) * 3 + c];
                                    let v = (f32::from(p) / 127.5) - 1.0;
                                    patches[out] = bf16::from_f32(v);
                                    out += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ImageMedia {
        width: resized_w as u32,
        height: resized_h as u32,
        grid_t: 1,
        grid_h,
        grid_w,
        patches,
        digest,
    })
}

/// Python/Transformers `smart_resize`: ties-to-even rounding, then the
/// min/max pixel correction with floor/ceil in the reference's order.
/// Returns `(width, height)`; the Python helper returns `(height, width)`.
pub fn smart_resize(height: usize, width: usize) -> (usize, usize) {
    let factor = PATCH * MERGE;
    let round = |v: f64| v.round_ties_even() as usize;
    let mut h_bar = round(height as f64 / factor as f64) * factor;
    let mut w_bar = round(width as f64 / factor as f64) * factor;
    let pixels = h_bar.saturating_mul(w_bar);
    if pixels > MAX_PIXELS {
        let beta = ((height * width) as f64 / MAX_PIXELS as f64).sqrt();
        h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
        w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
    } else if pixels < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
    }
    (w_bar, h_bar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use image::{ImageBuffer, Rgb, Rgba};

    fn png(width: u32, height: u32, rgba: bool) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        if rgba {
            let image = ImageBuffer::from_pixel(width, height, Rgba([200u8, 30, 40, 128]));
            image.write_to(&mut out, image::ImageFormat::Png).unwrap();
        } else {
            let image = ImageBuffer::from_pixel(width, height, Rgb([200u8, 30, 40]));
            image.write_to(&mut out, image::ImageFormat::Png).unwrap();
        }
        out.into_inner()
    }

    #[test]
    fn smart_resize_matches_qwen_rounding() {
        assert_eq!(smart_resize(60, 80), (320, 224));
        assert_eq!(smart_resize(1, 1), (256, 256));
        assert_eq!(smart_resize(10, 2000), (3_648, 32));
        assert_eq!(smart_resize(2000, 10), (32, 3_648));
    }

    #[test]
    fn data_url_is_strict_about_type_and_base64() {
        let bytes = png(16, 16, false);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        for (url, needle) in [
            ("hello".to_owned(), "data URL"),
            ("data:image/gif;base64,AAAA".to_owned(), "png, jpeg"),
            ("data:image/png;base64,!!!".to_owned(), "valid base64"),
            (format!("data:image/png;base64,{encoded}"), ""),
        ] {
            let result = decode_data_url(&url);
            if needle.is_empty() {
                assert!(result.is_ok(), "{result:?}");
            } else {
                assert!(result.unwrap_err().to_string().contains(needle));
            }
        }
    }

    #[test]
    fn decoded_alpha_is_dropped_and_digest_includes_dimensions() {
        let image = decode_bytes(&png(8, 8, true)).unwrap();
        assert_eq!((image.width, image.height), (256, 256));
        assert_eq!(image.patches(), image.grid_h * image.grid_w);
        assert_eq!(image.tokens(), image.patches() / 4);
        let mut raw = Vec::new();
        for pixel in &image.patches[..3] {
            raw.extend_from_slice(&pixel.to_bits().to_le_bytes());
        }
        // A same-size RGBA image with a different alpha channel must have the
        // same RGB identity, while a transposed image must not.
        let same = decode_bytes(&png(8, 8, false)).unwrap();
        assert_eq!(image.digest, same.digest);
        let transposed = decode_bytes(&png(4, 16, false)).unwrap();
        assert_ne!(image.digest, transposed.digest);
        // Different source dimensions with the same pixel count must not share
        // an image-cache identity.
        assert_ne!(image.digest, transposed.digest);
    }
}
