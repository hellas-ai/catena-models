use super::{
    text::{TextEncoder, tokenize},
    transformer::Transformer,
    vae::VaeDecoder,
};
use anyhow::{Context, Result, ensure};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::{fs::File, io::BufWriter, path::PathBuf, time::Instant};
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub backend: super::ops::Backend,
    pub prompt: String,
    pub output: PathBuf,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub dump_dir: Option<PathBuf>,
    pub noise: Option<PathBuf>,
}
impl GenerationOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0 && self.height > 0 && self.width % 16 == 0 && self.height % 16 == 0,
            "image dimensions must be positive multiples of 16"
        );
        ensure!(self.steps > 0, "steps must be positive");
        if matches!(self.backend, super::ops::Backend::Legacy) {
            ensure!(
                self.width
                    .checked_mul(self.height)
                    .is_some_and(|area| area <= 1024 * 1024),
                "legacy backend supports at most 1,048,576 pixels (existing softmax limit: 16,384 keys)"
            );
        }
        self.width
            .checked_mul(self.height)
            .and_then(|v| v.checked_mul(128))
            .context("image size overflow")?;
        Ok(())
    }
}
pub struct FluxPipeline {
    snapshot: PathBuf,
}
impl FluxPipeline {
    pub fn new(snapshot: PathBuf) -> Result<Self> {
        let index: serde_json::Value = super::config::read(snapshot.join("model_index.json"))?;
        ensure!(
            index["_class_name"] == "Flux2KleinPipeline" && index["is_distilled"] == true,
            "expected distilled Flux2 Klein snapshot"
        );
        let scheduler: serde_json::Value =
            super::config::read(snapshot.join("scheduler/scheduler_config.json"))?;
        ensure!(
            scheduler["_class_name"] == "FlowMatchEulerDiscreteScheduler"
                && scheduler["use_dynamic_shifting"] == true
                && scheduler["time_shift_type"] == "exponential"
                && scheduler["invert_sigmas"] == false
                && scheduler["shift_terminal"].is_null()
                && scheduler["stochastic_sampling"] == false,
            "unsupported scheduler configuration"
        );
        for key in [
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        ] {
            ensure!(scheduler[key] == false, "unsupported scheduler {key}");
        }
        Ok(Self { snapshot })
    }
    pub fn generate(&self, o: &GenerationOptions) -> Result<()> {
        o.validate()?;
        let total = Instant::now();
        let dump = o.dump_dir.as_deref();
        let tokens = tokenize(&self.snapshot, &o.prompt)?;
        if let Some(dir) = dump {
            std::fs::create_dir_all(dir)?;
            std::fs::write(
                dir.join("tokens.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"ids":tokens.ids,"valid":tokens.valid}),
                )?,
            )?;
        }
        let t = Instant::now();
        let encoder = TextEncoder::load_with_backend(&self.snapshot, tokens.valid, o.backend)?;
        eprintln!("text load/compile: {:.2}s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let conditioning = encoder.encode(&tokens, dump)?;
        eprintln!("text encode: {:.2}s", t.elapsed().as_secs_f64());
        drop(encoder);
        let t = Instant::now();
        let transformer =
            Transformer::load_with_backend(&self.snapshot, o.height, o.width, o.backend)?;
        eprintln!(
            "transformer load/compile: {:.2}s",
            t.elapsed().as_secs_f64()
        );
        let rows = o.width / 16 * (o.height / 16);
        let count = rows * 128;
        let noise = if let Some(path) = &o.noise {
            super::debug::noise(path, count)?
        } else {
            let mut rng = ChaCha8Rng::seed_from_u64(o.seed);
            (0..count)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect::<Vec<f32>>()
        };
        let bits: Vec<_> = noise.into_iter().map(f32_to_bf16).collect();
        let mut latents = transformer.ops.runtime.mem_u16(&bits)?;
        super::debug::dump(dump, "initial_noise", &latents, &[1, rows, 128], true)?;
        let sigmas = schedule(rows, o.steps);
        if let Some(dir) = dump {
            std::fs::write(
                dir.join("schedule.json"),
                serde_json::to_vec_pretty(&sigmas)?,
            )?;
        }
        let t = Instant::now();
        for step in 0..o.steps {
            let st = Instant::now();
            let velocity = transformer.predict_velocity(
                latents.as_ref(),
                conditioning.as_ref(),
                sigmas[step],
                step,
                dump,
            )?;
            latents = transformer.ops.run(
                "flux.euler",
                vec![
                    latents.as_ref().into(),
                    velocity.as_ref().into(),
                    (count as u64).into(),
                    (sigmas[step + 1] - sigmas[step]).into(),
                ],
            )?;
            super::debug::dump(
                dump,
                &format!("step_{step}_latents"),
                &latents,
                &[1, rows, 128],
                true,
            )?;
            eprintln!(
                "diffusion step {}/{}: {:.2}s",
                step + 1,
                o.steps,
                st.elapsed().as_secs_f64()
            );
        }
        eprintln!("denoise: {:.2}s", t.elapsed().as_secs_f64());
        drop(transformer);
        drop(conditioning);
        let t = Instant::now();
        let vae = VaeDecoder::load_with_backend(&self.snapshot, o.backend)?;
        eprintln!("VAE load/compile: {:.2}s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let rgb = vae.decode(&latents, o.height, o.width, dump)?;
        let values = rgb.try_to_f32_vec()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "VAE produced non-finite pixels"
        );
        let pixels = o.width * o.height;
        let mut bytes = Vec::with_capacity(pixels * 3);
        for pixel in 0..pixels {
            for channel in 0..3 {
                bytes.push(
                    ((values[channel * pixels + pixel] / 2. + 0.5).clamp(0., 1.) * 255.)
                        .round_ties_even() as u8,
                );
            }
        }
        let mut png = png::Encoder::new(
            BufWriter::new(
                File::create(&o.output)
                    .with_context(|| format!("creating {}", o.output.display()))?,
            ),
            o.width.try_into()?,
            o.height.try_into()?,
        );
        png.set_color(png::ColorType::Rgb);
        png.set_depth(png::BitDepth::Eight);
        png.write_header()?.write_image_data(&bytes)?;
        eprintln!(
            "decode/write: {:.2}s; total: {:.2}s; output: {}",
            t.elapsed().as_secs_f64(),
            total.elapsed().as_secs_f64(),
            o.output.display()
        );
        Ok(())
    }
}
/// Diffusers v0.37 Flux2 empirical shift + FlowMatch Euler exponential time shift.
pub fn schedule(image_seq_len: usize, steps: usize) -> Vec<f32> {
    let n = image_seq_len as f64;
    let m200 = 0.00016927 * n + 0.45666666;
    let mu = if image_seq_len > 4300 {
        m200
    } else {
        let m10 = 8.73809524e-5 * n + 1.89833333;
        let a = (m200 - m10) / 190.;
        a * steps as f64 + m200 - 200. * a
    };
    let e = mu.exp();
    let mut sigmas = (0..steps)
        .map(|i| {
            let s = 1. - i as f64 / steps as f64;
            (e / (e + (1. / s - 1.))) as f32
        })
        .collect::<Vec<_>>();
    sigmas.push(0.);
    sigmas
}
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7fffffff > 0x7f800000 {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scheduler_is_monotone_and_ends_at_zero() {
        for rows in [256, 1024, 4096, 5000] {
            for steps in [1, 4, 20] {
                let s = schedule(rows, steps);
                assert_eq!(s.len(), steps + 1);
                assert_eq!(s[0], 1.);
                assert_eq!(s[steps], 0.);
                assert!(s.windows(2).all(|w| w[0] > w[1]));
            }
        }
    }
    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_to_f32(f32_to_bf16(1.)), 1.);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f818000)), 0x3f82);
    }
}
