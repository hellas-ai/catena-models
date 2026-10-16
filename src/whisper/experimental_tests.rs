//! Independent BF16 CPU oracle and end-to-end GPU generation checks.
use super::{Config, bf16, graph, host};
use anyhow::{Result, ensure};
use catena_lang::runtime::{GpuDialect, MemOwn, Runtime, RuntimeModule, Value};
use std::collections::BTreeMap;
fn round(x: f32) -> f32 {
    f32::from_bits((bf16(&[x])[0] as u32) << 16)
}
fn add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(a, b)| round(a + b)).collect()
}
fn gelu(x: f32) -> f32 {
    let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let p = ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    x * 0.5 * (1.0 + (1.0 - p * (-z * z).exp()).copysign(x))
}
struct Oracle<'a> {
    cfg: &'a Config,
    w: &'a BTreeMap<String, Vec<f32>>,
}
impl Oracle<'_> {
    fn linear(&self, x: &[f32], name: &str, ci: usize, co: usize) -> Vec<f32> {
        let w = &self.w[name];
        x.chunks_exact(ci)
            .flat_map(|r| {
                (0..co).map(move |j| {
                    round(
                        r.iter()
                            .zip(&w[j * ci..(j + 1) * ci])
                            .map(|(a, b)| a * b)
                            .sum(),
                    )
                })
            })
            .collect()
    }
    fn bias(&self, x: &[f32], name: &str, activate: bool) -> Vec<f32> {
        let b = &self.w[name];
        x.iter()
            .enumerate()
            .map(|(i, x)| {
                let v = x + b[i % b.len()];
                round(if activate { gelu(v) } else { v })
            })
            .collect()
    }
    fn norm(&self, x: &[f32], prefix: &str) -> Vec<f32> {
        let w = &self.w[&format!("{prefix}.weight")];
        let b = &self.w[&format!("{prefix}.bias")];
        x.chunks_exact(w.len())
            .flat_map(|r| {
                let mean = r.iter().sum::<f32>() / r.len() as f32;
                let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / r.len() as f32;
                r.iter()
                    .enumerate()
                    .map(move |(i, x)| round((x - mean) / (var + 1e-5).sqrt() * w[i] + b[i]))
            })
            .collect()
    }
    fn proj(
        &self,
        x: &[f32],
        prefix: &str,
        ci: usize,
        co: usize,
        bias: bool,
        activate: bool,
    ) -> Vec<f32> {
        let x = self.linear(x, &format!("{prefix}.weight"), ci, co);
        if bias {
            self.bias(&x, &format!("{prefix}.bias"), activate)
        } else {
            x
        }
    }
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], valid: usize, heads: usize) -> Vec<f32> {
        let d = self.cfg.d_model;
        let dim = d / heads;
        let mut out = vec![0.; q.len()];
        for r in 0..q.len() / d {
            for head in 0..heads {
                let base = r * d + head * dim;
                let mut scores = (0..valid)
                    .map(|j| {
                        (0..dim)
                            .map(|i| q[base + i] * k[j * d + head * dim + i])
                            .sum::<f32>()
                            / (dim as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let den = scores.iter().sum::<f32>();
                for s in &mut scores {
                    *s /= den;
                }
                for i in 0..dim {
                    out[base + i] = round(
                        (0..valid)
                            .map(|j| scores[j] * v[j * d + head * dim + i])
                            .sum(),
                    );
                }
            }
        }
        out
    }
    fn ff(&self, x: &[f32], p: &str, ff: usize) -> Vec<f32> {
        let n = self.norm(x, &format!("{p}.final_layer_norm"));
        let up = self.proj(&n, &format!("{p}.fc1"), self.cfg.d_model, ff, true, true);
        let down = self.proj(&up, &format!("{p}.fc2"), ff, self.cfg.d_model, true, false);
        add(x, &down)
    }
    fn conv(&self, x: &[f32], frames: usize, ci: usize, stride: usize, p: &str) -> Vec<f32> {
        let rows = frames.div_ceil(stride);
        let mut patches = vec![0.; rows * ci * 3];
        for r in 0..rows {
            for channel in 0..ci {
                for tap in 0..3 {
                    let f = (r * stride + tap) as isize - 1;
                    if f >= 0 && (f as usize) < frames {
                        patches[(r * ci + channel) * 3 + tap] = x[f as usize * ci + channel];
                    }
                }
            }
        }
        self.proj(&patches, p, ci * 3, self.cfg.d_model, true, true)
    }
    fn encode(&self, features: &[f32]) -> Vec<f32> {
        let c = self.cfg;
        let d = c.d_model;
        let x = self.conv(
            features,
            c.max_source_positions * 2,
            c.num_mel_bins,
            1,
            "model.encoder.conv1",
        );
        let x = self.conv(&x, c.max_source_positions * 2, d, 2, "model.encoder.conv2");
        let mut x = add(&x, &self.w["model.encoder.embed_positions.weight"]);
        for l in 0..c.encoder_layers {
            let p = format!("model.encoder.layers.{l}");
            let n = self.norm(&x, &format!("{p}.self_attn_layer_norm"));
            let q = self.proj(&n, &format!("{p}.self_attn.q_proj"), d, d, true, false);
            let k = self.proj(&n, &format!("{p}.self_attn.k_proj"), d, d, false, false);
            let v = self.proj(&n, &format!("{p}.self_attn.v_proj"), d, d, true, false);
            let a = self.attention(
                &q,
                &k,
                &v,
                c.max_source_positions,
                c.encoder_attention_heads,
            );
            x = add(
                &x,
                &self.proj(&a, &format!("{p}.self_attn.out_proj"), d, d, true, false),
            );
            x = self.ff(&x, &p, c.encoder_ffn_dim);
        }
        self.norm(&x, "model.encoder.layer_norm")
    }
    fn cross(&self, encoder: &[f32]) -> Vec<Vec<f32>> {
        (0..self.cfg.decoder_layers)
            .flat_map(|l| {
                let p = format!("model.decoder.layers.{l}.encoder_attn");
                [
                    self.proj(
                        encoder,
                        &format!("{p}.k_proj"),
                        self.cfg.d_model,
                        self.cfg.d_model,
                        false,
                        false,
                    ),
                    self.proj(
                        encoder,
                        &format!("{p}.v_proj"),
                        self.cfg.d_model,
                        self.cfg.d_model,
                        true,
                        false,
                    ),
                ]
            })
            .collect()
    }
    fn decode(
        &self,
        token: usize,
        pos: usize,
        cache: &mut [Vec<f32>],
        cross: &[Vec<f32>],
    ) -> Vec<f32> {
        let c = self.cfg;
        let d = c.d_model;
        let mut x = add(
            &self.w["model.decoder.embed_tokens.weight"][token * d..(token + 1) * d],
            &self.w["model.decoder.embed_positions.weight"][pos * d..(pos + 1) * d],
        );
        for l in 0..c.decoder_layers {
            let p = format!("model.decoder.layers.{l}");
            let n = self.norm(&x, &format!("{p}.self_attn_layer_norm"));
            let q = self.proj(&n, &format!("{p}.self_attn.q_proj"), d, d, true, false);
            let k = self.proj(&n, &format!("{p}.self_attn.k_proj"), d, d, false, false);
            let v = self.proj(&n, &format!("{p}.self_attn.v_proj"), d, d, true, false);
            cache[2 * l][pos * d..(pos + 1) * d].copy_from_slice(&k);
            cache[2 * l + 1][pos * d..(pos + 1) * d].copy_from_slice(&v);
            let a = self.attention(
                &q,
                &cache[2 * l],
                &cache[2 * l + 1],
                pos + 1,
                c.decoder_attention_heads,
            );
            x = add(
                &x,
                &self.proj(&a, &format!("{p}.self_attn.out_proj"), d, d, true, false),
            );
            let n = self.norm(&x, &format!("{p}.encoder_attn_layer_norm"));
            let q = self.proj(&n, &format!("{p}.encoder_attn.q_proj"), d, d, true, false);
            let a = self.attention(
                &q,
                &cross[2 * l],
                &cross[2 * l + 1],
                c.max_source_positions,
                c.decoder_attention_heads,
            );
            x = add(
                &x,
                &self.proj(&a, &format!("{p}.encoder_attn.out_proj"), d, d, true, false),
            );
            x = self.ff(&x, &p, c.decoder_ffn_dim);
        }
        let x = self.norm(&x, "model.decoder.layer_norm");
        self.linear(&x, "model.decoder.embed_tokens.weight", d, c.vocab_size)
    }
}
fn tensors(c: &Config) -> BTreeMap<String, Vec<f32>> {
    let mut shapes = BTreeMap::from([
        (
            "model.encoder.conv1.weight".into(),
            c.d_model * c.num_mel_bins * 3,
        ),
        ("model.encoder.conv1.bias".into(), c.d_model),
        (
            "model.encoder.conv2.weight".into(),
            c.d_model * c.d_model * 3,
        ),
        ("model.encoder.conv2.bias".into(), c.d_model),
        (
            "model.encoder.embed_positions.weight".into(),
            c.max_source_positions * c.d_model,
        ),
        (
            "model.decoder.embed_tokens.weight".into(),
            c.vocab_size * c.d_model,
        ),
        (
            "model.decoder.embed_positions.weight".into(),
            c.max_target_positions * c.d_model,
        ),
    ]);
    for kind in ["encoder", "decoder"] {
        for s in ["weight", "bias"] {
            shapes.insert(format!("model.{kind}.layer_norm.{s}"), c.d_model);
        }
        let layers = if kind == "encoder" {
            c.encoder_layers
        } else {
            c.decoder_layers
        };
        for l in 0..layers {
            let mut names = graph::layer_names(kind, l, graph::ATTN);
            names.extend(graph::layer_names(kind, l, graph::FF));
            if kind == "decoder" {
                names.extend(graph::layer_names(kind, l, graph::CROSS));
                names.extend(graph::layer_names(
                    kind,
                    l,
                    &[
                        "encoder_attn.k_proj.weight",
                        "encoder_attn.v_proj.weight",
                        "encoder_attn.v_proj.bias",
                    ],
                ));
            }
            for n in names {
                let ff = if kind == "encoder" {
                    c.encoder_ffn_dim
                } else {
                    c.decoder_ffn_dim
                };
                let size = if n.ends_with("fc1.weight") || n.ends_with("fc2.weight") {
                    c.d_model * ff
                } else if n.ends_with("fc1.bias") {
                    ff
                } else if n.ends_with("bias") || n.contains("layer_norm") {
                    c.d_model
                } else {
                    c.d_model * c.d_model
                };
                shapes.insert(n, size);
            }
        }
    }
    shapes
        .iter()
        .enumerate()
        .map(|(i, (name, n))| {
            let values = (0..*n)
                .map(|j| {
                    let gamma = name.contains("layer_norm") && name.ends_with("weight");
                    let value = (((i * 17 + j * 13) % 19) as i32 - 9) as f32 / 128.;
                    round(value + if gamma { 1.0 } else { 0.0 })
                })
                .collect();
            (name.clone(), values)
        })
        .collect()
}
fn memories(outputs: Vec<Value<'_>>) -> Vec<MemOwn> {
    outputs
        .into_iter()
        .map(|v| match v {
            Value::MemOwn(m) => m,
            _ => panic!("unexpected scalar output"),
        })
        .collect()
}
fn compare(a: &[f32], b: &[f32], label: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "shape mismatch: {label}");
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    eprintln!("{label}: max error {error}");
    ensure!(
        a.iter().all(|x| x.is_finite()) && error <= 0.004,
        "{label}: error {error}"
    );
    Ok(())
}
fn args<'a>(weights: &'a BTreeMap<String, MemOwn>, names: &[String]) -> Vec<Value<'a>> {
    names.iter().map(|n| weights[n].as_ref().into()).collect()
}

pub(super) fn exercise(module: RuntimeModule, c: &Config, debug: bool) -> Result<()> {
    if let Some(dir) = std::env::var_os("CATENA_WHISPER_TEST_ARTIFACT_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(format!("tiny-{debug}.cpp")), &module.source)?;
    }
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(module)?;
    let mut weights = tensors(c);
    let mut device = weights
        .iter()
        .map(|(n, w)| Ok((n.clone(), runtime.mem_u16(&bf16(w))?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let features = (0..c.max_source_positions * 2 * c.num_mel_bins)
        .map(|i| round((i as i32 % 13 - 6) as f32 / 16.))
        .collect::<Vec<_>>();
    let oracle = Oracle {
        cfg: c,
        w: &weights,
    };
    let expected_encoder = oracle.encode(&features);
    let expected_cross = oracle.cross(&expected_encoder);
    let mut a = vec![runtime.mem_u16(&bf16(&features))?.into()];
    a.extend(args(&device, &graph::encoder_names(c)));
    let encoder = memories(artifact.exec_values("whisper.encode", a)?).remove(0);
    compare(&host(&encoder)?, &expected_encoder, "encoder")?;
    let mut a = vec![encoder.into()];
    a.extend(args(&device, &graph::cross_names(c)));
    let cross = memories(artifact.exec_values("whisper.prepare-cross-kv", a)?);
    for (i, (a, b)) in cross.iter().zip(&expected_cross).enumerate() {
        compare(&host(a)?, b, &format!("cross {i}"))?;
    }
    let mut cache = (0..2 * c.decoder_layers)
        .map(|_| runtime.mem_u16(&vec![0x7fc0; c.max_target_positions * c.d_model]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut cpu_cache =
        vec![vec![f32::NAN; c.max_target_positions * c.d_model]; 2 * c.decoder_layers];
    for (pos, token) in [12, 13, 15, 16, 3, 5].into_iter().enumerate() {
        let expected = oracle.decode(token, pos, &mut cpu_cache, &expected_cross);
        let mut a = vec![(token as u64).into(), (pos as u64).into()];
        for (l, k) in cache.into_iter().enumerate() {
            a.push(k.into());
            if l % 2 == 1 {
                a.extend([cross[l - 1].as_ref().into(), cross[l].as_ref().into()]);
            }
        }
        a.extend(args(&device, &graph::decoder_names(c)));
        let mut out = memories(artifact.exec_values("whisper.decode-step", a)?);
        let logits = out.remove(0);
        compare(&host(&logits)?, &expected, &format!("decoder pos {pos}"))?;
        cache = out;
        for (i, k) in cache.iter().enumerate() {
            let values = host(k)?;
            compare(
                &values[..(pos + 1) * c.d_model],
                &cpu_cache[i][..(pos + 1) * c.d_model],
                "self cache",
            )?;
            ensure!(
                values[(pos + 1) * c.d_model..].iter().all(|x| x.is_nan()),
                "overwrote future cache"
            );
        }
    }
    let df = runtime.mem_f32(&features)?;
    // Compare device-side generation with ordinary CPU decoding before replacing
    // the tied embeddings for the isolated control tests below.
    for override_language in [0_u64, 14] {
        let mut cpu_cache =
            vec![vec![0.; c.max_target_positions * c.d_model]; 2 * c.decoder_layers];
        let logits = oracle.decode(12, 0, &mut cpu_cache, &expected_cross);
        let detected = if logits[14] > logits[13] { 14 } else { 13 };
        let language = if override_language == 0 {
            detected
        } else {
            override_language as usize
        };
        oracle.decode(language, 1, &mut cpu_cache, &expected_cross);
        oracle.decode(15, 2, &mut cpu_cache, &expected_cross);
        let mut logits = oracle.decode(16, 3, &mut cpu_cache, &expected_cross);
        let mut expected = vec![];
        let mut trace = vec![];
        let mut status = 0_u64;
        for i in 0..3 {
            trace.extend_from_slice(&logits);
            let mut best = 0;
            let mut maximum = f32::NEG_INFINITY;
            for (j, &value) in logits.iter().enumerate().take(12) {
                if j != 4 && (i != 0 || j != 3) && value > maximum {
                    best = j;
                    maximum = value;
                }
            }
            if best >= 11 {
                status = 1;
                break;
            }
            expected.push(best as u64);
            if i < 2 {
                logits = oracle.decode(best, 4 + i, &mut cpu_cache, &expected_cross);
            }
        }
        let mut language_mask = vec![f32::NEG_INFINITY; c.vocab_size];
        language_mask[13] = 0.;
        language_mask[14] = 0.;
        let mut text = vec![0.; c.vocab_size];
        text[12..].fill(f32::NEG_INFINITY);
        text[4] = f32::NEG_INFINITY;
        let mut begin = vec![0.; c.vocab_size];
        begin[3] = f32::NEG_INFINITY;
        let dl = runtime.mem_f32(&language_mask)?;
        let dt = runtime.mem_f32(&text)?;
        let db = runtime.mem_f32(&begin)?;
        let mut a = vec![
            df.as_ref().into(),
            3_u64.into(),
            override_language.into(),
            12_u64.into(),
            15_u64.into(),
            16_u64.into(),
            11_u64.into(),
            dl.as_ref().into(),
            dt.as_ref().into(),
            db.as_ref().into(),
        ];
        a.extend(args(&device, &graph::encoder_names(c)));
        a.extend(args(&device, &graph::cross_names(c)));
        a.extend(args(&device, &graph::decoder_names(c)));
        let out = memories(artifact.exec_values("whisper.transcribe", a)?);
        let values = out[0].try_to_u64_vec()?;
        let mut expected_result = vec![expected.len() as u64, language as u64, status];
        expected_result.extend(expected);
        assert_eq!(
            &values[..expected_result.len()],
            expected_result.as_slice(),
            "device generation differs from CPU decoding"
        );
        if debug {
            compare(&host(&out[2])?[..trace.len()], &trace, "generation logits")?;
        }
    }
    // Zero tied embeddings make every logit equal, isolating generation control:
    // lowest-ID ties, first-token suppression, language override, EOS and invalid masks.
    weights
        .get_mut("model.decoder.embed_tokens.weight")
        .unwrap()
        .fill(0.);
    device.insert(
        "model.decoder.embed_tokens.weight".into(),
        runtime.mem_u16(&vec![0; c.vocab_size * c.d_model])?,
    );
    for (mode, override_language) in [(0, 0_u64), (1, 14), (2, 0), (0, 14)] {
        let mut language = vec![f32::NEG_INFINITY; c.vocab_size];
        language[13] = 0.;
        language[14] = 0.;
        let mut text = vec![f32::NEG_INFINITY; c.vocab_size];
        if mode == 0 {
            text[3] = 0.;
            text[5] = 0.;
        } else if mode == 1 {
            text[11] = 0.;
        }
        let mut begin = vec![0.; c.vocab_size];
        begin[3] = f32::NEG_INFINITY;
        let dl = runtime.mem_f32(&language)?;
        let dt = runtime.mem_f32(&text)?;
        let db = runtime.mem_f32(&begin)?;
        let mut a = vec![
            df.as_ref().into(),
            3_u64.into(),
            override_language.into(),
            12_u64.into(),
            15_u64.into(),
            16_u64.into(),
            11_u64.into(),
            dl.as_ref().into(),
            dt.as_ref().into(),
            db.as_ref().into(),
        ];
        a.extend(args(&device, &graph::encoder_names(c)));
        a.extend(args(&device, &graph::cross_names(c)));
        a.extend(args(&device, &graph::decoder_names(c)));
        let out = memories(artifact.exec_values("whisper.transcribe", a)?);
        ensure!(out.len() == if debug { 3 } else { 1 }, "wrong result ABI");
        let values = out[0].try_to_u64_vec()?;
        let lang = if override_language == 0 {
            13
        } else {
            override_language
        };
        match mode {
            0 => {
                assert_eq!(&values[..6], &[3, lang, 0, 5, 3, 3]);
            }
            1 => assert_eq!(&values[..3], &[0, lang, 1]),
            _ => assert_eq!(&values[..3], &[0, lang, 2]),
        }
        if debug {
            compare(&host(&out[1])?, &expected_encoder, "encoder debug capture")?;
            ensure!(
                host(&out[2])?.iter().all(|x| x.is_finite()),
                "nonfinite debug logits"
            );
        }
    }
    Ok(())
}
