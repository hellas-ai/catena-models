use super::VisionConfig;
use anyhow::{Context, Result, bail, ensure};
use std::{fs::File, io::BufReader, path::Path};

/// Spatial-merge ordered patches; each row is [RGB channel, patch y, patch x].
/// Still-image temporal duplication is applied by the two projection slices.
pub struct ImagePatches {
    pub pixels: Vec<f32>,
    pub grid: [usize; 3],
    pub positions: Vec<u64>,
    pub bilinear_indices: Vec<u64>,
    pub bilinear_weights: Vec<f32>,
    pub resized_rgb: Vec<u8>,
    pub resized_size: [usize; 2],
}
impl ImagePatches {
    pub fn open(path: &Path, config: &VisionConfig) -> Result<Self> {
        let (rgb, w, h) = decode(path).with_context(|| format!("decoding {}", path.display()))?;
        Self::from_rgb(&rgb, w, h, config)
    }
    pub fn from_rgb(rgb: &[u8], w: usize, h: usize, config: &VisionConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            w.checked_mul(h).and_then(|n| n.checked_mul(3)) == Some(rgb.len()),
            "invalid RGB image size"
        );
        let (rh, rw) = smart_resize(
            h,
            w,
            config.patch_size * config.merge_size,
            config.min_pixels,
            config.max_pixels,
        )?;
        let rgb = resize_rgb(rgb, w, h, rw, rh);
        let gh = rh / config.patch_size;
        let gw = rw / config.patch_size;
        let p = config.patch_size;
        let m = config.merge_size;
        let mut pixels = Vec::with_capacity(rgb.len());
        let mut positions = Vec::with_capacity(gh * gw * 2);
        let mut bilinear_indices = Vec::with_capacity(gh * gw * 4);
        let mut bilinear_weights = Vec::with_capacity(gh * gw * 4);
        for by in 0..gh / m {
            for bx in 0..gw / m {
                for dy in 0..m {
                    for dx in 0..m {
                        let y = by * m + dy;
                        let x = bx * m + dx;
                        positions.extend([y as u64, x as u64]);
                        for c in 0..3 {
                            for py in 0..p {
                                for px in 0..p {
                                    let value =
                                        rgb[((y * p + py) * rw + x * p + px) * 3 + c] as f32;
                                    pixels.push((value / 255. - config.mean[c]) / config.std[c]);
                                }
                            }
                        }
                        let sy =
                            y as f32 * (config.position_side - 1) as f32 / (gh - 1).max(1) as f32;
                        let sx =
                            x as f32 * (config.position_side - 1) as f32 / (gw - 1).max(1) as f32;
                        let y0 = sy.floor() as usize;
                        let x0 = sx.floor() as usize;
                        let y1 = (y0 + 1).min(config.position_side - 1);
                        let x1 = (x0 + 1).min(config.position_side - 1);
                        let fy = sy - y0 as f32;
                        let fx = sx - x0 as f32;
                        bilinear_indices.extend(
                            [
                                y0 * config.position_side + x0,
                                y0 * config.position_side + x1,
                                y1 * config.position_side + x0,
                                y1 * config.position_side + x1,
                            ]
                            .map(|v| v as u64),
                        );
                        bilinear_weights.extend([
                            (1. - fy) * (1. - fx),
                            (1. - fy) * fx,
                            fy * (1. - fx),
                            fy * fx,
                        ]);
                    }
                }
            }
        }
        Ok(Self {
            pixels,
            grid: [1, gh, gw],
            positions,
            bilinear_indices,
            bilinear_weights,
            resized_rgb: rgb,
            resized_size: [rh, rw],
        })
    }
}

pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min: usize,
    max: usize,
) -> Result<(usize, usize)> {
    ensure!(
        h > 0 && w > 0 && factor > 0 && min > 0 && min <= max,
        "invalid resize dimensions or limits"
    );
    ensure!(
        h.max(w) as f64 / h.min(w) as f64 <= 200.,
        "image aspect ratio exceeds 200"
    );
    ensure!(
        max >= factor * factor,
        "maximum image pixels is smaller than one merge block"
    );
    // Python round uses ties-to-even.
    let mut rh = (h as f64 / factor as f64).round_ties_even() as usize * factor;
    let mut rw = (w as f64 / factor as f64).round_ties_even() as usize * factor;
    let area = h as f64 * w as f64;
    if rh as f64 * rw as f64 > max as f64 {
        let beta = (area / max as f64).sqrt();
        rh = ((h as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        rw = ((w as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if (rh as f64 * rw as f64) < min as f64 {
        let beta = (min as f64 / area).sqrt();
        rh = (h as f64 * beta / factor as f64).ceil() as usize * factor;
        rw = (w as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    ensure!(
        rh.checked_mul(rw).and_then(|n| n.checked_mul(3)).is_some(),
        "resized image size overflow"
    );
    Ok((rh, rw))
}
fn decode(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let mut magic = [0; 8];
    std::io::Read::read_exact(&mut File::open(path)?, &mut magic)?;
    if magic.starts_with(&[0xff, 0xd8]) {
        let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(File::open(path)?));
        let bytes = decoder.decode()?;
        let info = decoder.info().context("missing JPEG header")?;
        let rgb = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => bytes,
            jpeg_decoder::PixelFormat::L8 => bytes.into_iter().flat_map(|v| [v; 3]).collect(),
            _ => bail!("only RGB/grayscale JPEG images are supported"),
        };
        return Ok((rgb, info.width as usize, info.height as usize));
    }
    ensure!(magic == *b"\x89PNG\r\n\x1a\n", "expected PNG or JPEG image");
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut bytes = vec![
        0;
        reader
            .output_buffer_size()
            .context("PNG dimensions overflow")?
    ];
    let info = reader.next_frame(&mut bytes)?;
    let bytes = &bytes[..info.buffer_size()];
    let rgb = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|v| [v[0], v[1], v[2]])
            .collect(),
        png::ColorType::Grayscale => bytes.iter().flat_map(|v| [*v; 3]).collect(),
        png::ColorType::GrayscaleAlpha => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|v| [v[0]; 3])
            .collect(),
        _ => bail!("unsupported PNG color type"),
    };
    Ok((rgb, info.width as usize, info.height as usize))
}
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        ((-0.5 * x + 2.5) * x - 4.) * x + 2.
    } else {
        0.
    }
}
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let support = 2. * filter_scale;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<_> = (start..end)
                .map(|j| cubic((j as f64 - center + 0.5) / filter_scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            (
                start,
                weights
                    .into_iter()
                    .map(|v| (v / sum * 4194304.).round() as i64)
                    .collect(),
            )
        })
        .collect()
}
/// Pillow-compatible antialiased bicubic resizing, including 8-bit intermediate rounding.
fn resize_rgb(rgb: &[u8], w: usize, h: usize, rw: usize, rh: usize) -> Vec<u8> {
    let horizontal = if w == rw {
        rgb.to_vec()
    } else {
        let coeff = coefficients(w, rw);
        let mut out = vec![0; rw * h * 3];
        for y in 0..h {
            for (x, (start, weights)) in coeff.iter().enumerate() {
                for c in 0..3 {
                    let sum = weights
                        .iter()
                        .enumerate()
                        .fold(1_i64 << 21, |a, (j, weight)| {
                            a + rgb[(y * w + start + j) * 3 + c] as i64 * weight
                        });
                    out[(y * rw + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    if h == rh {
        return horizontal;
    }
    let coeff = coefficients(h, rh);
    let mut out = vec![0; rw * rh * 3];
    for (y, (start, weights)) in coeff.iter().enumerate() {
        for x in 0..rw {
            for c in 0..3 {
                let sum = weights
                    .iter()
                    .enumerate()
                    .fold(1_i64 << 21, |a, (j, weight)| {
                        a + horizontal[((start + j) * rw + x) * 3 + c] as i64 * weight
                    });
                out[(y * rw + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resize_limits() {
        assert_eq!(
            smart_resize(256, 256, 32, 65536, 1048576).unwrap(),
            (256, 256)
        );
        assert_eq!(
            smart_resize(100, 200, 32, 65536, 1048576).unwrap(),
            (192, 384)
        );
        assert!(smart_resize(1, 1000, 32, 65536, 1048576).is_err());
        assert!(smart_resize(1, 1, 32, 1, 1).is_err());
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    #[test]
    fn preprocessing_matches_pinned_transformers_pillow_fixtures() -> Result<()> {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/qwen35-images.json"))?;
        let config = VisionConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            output_size: 1024,
            layers: 12,
            heads: 12,
            patch_size: 16,
            merge_size: 2,
            position_side: 48,
            eps: 1e-6,
            min_pixels: 65536,
            max_pixels: 1048576,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        for case in reference["cases"].as_array().unwrap() {
            let w = case["width"].as_u64().unwrap() as usize;
            let h = case["height"].as_u64().unwrap() as usize;
            let rgb: Vec<u8> = (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        [
                            ((x * 7 + y * 3) % 256) as u8,
                            ((x * 11 + y * 5) % 256) as u8,
                            ((x * 13 + y * 17) % 256) as u8,
                        ]
                    })
                })
                .collect();
            let patches = ImagePatches::from_rgb(&rgb, w, h, &config)?;
            assert_eq!(serde_json::json!(patches.grid), case["grid"]);
            assert_eq!(
                serde_json::json!(patches.resized_size),
                case["resized_size"]
            );
            let hash = patches
                .resized_rgb
                .iter()
                .fold(14695981039346656037_u64, |h, &b| {
                    (h ^ b as u64).wrapping_mul(1099511628211)
                });
            assert_eq!(
                hash.to_string(),
                case["rgb_fnv"].as_str().unwrap(),
                "resized RGB differs for {}",
                case["name"]
            );
            for (i, index) in [0, 1, 15, 16, 255, 256, 511, 512, 767, 768, 769, 1536]
                .into_iter()
                .enumerate()
            {
                assert!(
                    (patches.pixels[index] - case["patch_sample"][i].as_f64().unwrap() as f32)
                        .abs()
                        < 2e-7,
                    "patch packing differs for {}",
                    case["name"]
                );
            }
        }
        Ok(())
    }
}
