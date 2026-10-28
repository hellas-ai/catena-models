use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use catena_models::{
    catena::{InferenceOptions, KernelKind, generated_model_source_for_options},
    checkpoint::read_gguf_model_config,
    hfhub::{download_gguf, download_mmproj},
    model::Model,
    tokenizer::QwenTokenizer,
    vision::{PreparedMultimodalPrompt, VisionModel},
};
use clap::{CommandFactory, FromArgMatches, Parser};

#[derive(Debug, Parser)]
#[command(about = "Qwen text and Qwen3.5 image generation with native Catena GPU execution")]
struct Args {
    /// Hex kernel implementation (reference is available for Qwen3 dense).
    #[arg(long, value_enum, default_value_t = KernelKind::Fused)]
    kernels: KernelKind,

    /// Use experimental RDNA3 tiled attention and expert prefill kernels.
    #[arg(long)]
    wmma_prefill: bool,

    /// Text prefix, or user-message text when images are supplied.
    #[arg(long, required_unless_present_any = ["generate_entrypoint", "bench", "image"])]
    prompt: Option<String>,

    /// Write the generated model-specific Hex source under target/, then exit.
    #[arg(long, conflicts_with = "prompt")]
    generate_entrypoint: bool,

    /// Run llama-style prompt-processing and token-generation benchmarks.
    #[arg(long, conflicts_with = "generate_entrypoint")]
    bench: bool,

    /// Prompt-processing token count used by --bench; zero skips this test.
    #[arg(long, default_value_t = 512)]
    pp: usize,

    /// Token-generation count used by --bench; zero skips this test.
    #[arg(long, default_value_t = 128)]
    tg: usize,

    /// Prompt context used for warmed decode benchmarking.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
    tg_context: usize,

    /// Number of measured runs per enabled benchmark test.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
    bench_runs: usize,

    /// Hugging Face model as owner/repo[:variant].
    #[arg(long, value_name = "MODEL", required = true)]
    model: String,

    /// Still images, in order, preceding the text in one Qwen3.5 user message.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["bench", "generate_entrypoint"])]
    image: Vec<PathBuf>,

    /// Optional HF repository for the companion projector (defaults to --model).
    #[arg(long, value_name = "MODEL", requires = "image")]
    mmproj_model: Option<String>,

    /// Override the projector's maximum resized image pixel area.
    #[arg(long, value_parser = parse_positive_usize, requires = "image")]
    image_max_pixels: Option<usize>,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 20)]
    max_new_tokens: usize,

    /// Continue through EOS to produce a fixed-length benchmark.
    #[arg(long)]
    ignore_eos: bool,
}

fn inference_options(args: &Args) -> Result<InferenceOptions> {
    let dialect = catena_models::model::configured_gpu_dialect()?;
    ensure!(
        !args.wmma_prefill || dialect == catena_lang::runtime::GpuDialect::Hip,
        "--wmma-prefill requires the HIP backend on gfx11"
    );
    Ok(InferenceOptions::default()
        .with_kernels(args.kernels)
        .with_wmma_prefill(args.wmma_prefill))
}

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches)?;
    let options = inference_options(&args)?;
    let checkpoint = download_gguf(&args.model)?;
    eprintln!(
        "selected {} GGUF file(s) from the Hugging Face cache",
        checkpoint.files.len()
    );
    let selected_gguf = checkpoint.primary;
    if args.generate_entrypoint {
        let config = read_gguf_model_config(&selected_gguf)?;
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/catena");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("generated-qwen3.hex");
        std::fs::write(&path, generated_model_source_for_options(&config, options)?)?;
        println!("generated {}", path.display());
        return Ok(());
    }
    if args.bench {
        ensure!(
            args.pp > 0 || args.tg > 0,
            "at least one of --pp or --tg must be non-zero"
        );
        let load_started = Instant::now();
        let model = Model::load_gguf_with_options(&selected_gguf, options)?;
        eprintln!(
            "loaded Catena runtime and weights in {:.2?}",
            load_started.elapsed()
        );
        return run_benchmark(&model, args.pp, args.tg, args.tg_context, args.bench_runs);
    }

    let prompt = args.prompt.as_deref().unwrap_or_default();
    let gguf_path = selected_gguf;
    let tokenizer = QwenTokenizer::from_gguf(&gguf_path)?
        .ok_or_else(|| anyhow::anyhow!("HF checkpoint has no embedded tokenizer"))?;
    let mut tokens = tokenizer.encode(prompt)?;
    ensure!(
        !tokens.is_empty() || !args.image.is_empty(),
        "the prompt encoded to an empty token prefix"
    );

    let projector = if !args.image.is_empty() {
        let config = read_gguf_model_config(&gguf_path)?;
        ensure!(
            config.is_qwen35(),
            "image inputs require a Qwen3.5 dense or MoE model"
        );
        let path = download_mmproj(args.mmproj_model.as_deref().unwrap_or(&args.model))?;
        let vision_config = catena_models::vision::VisionConfig::from_gguf(&path)?;
        ensure!(
            vision_config.output_size == config.hidden_size,
            "projector output width does not match language model"
        );
        Some(path)
    } else {
        None
    };
    let load_started = Instant::now();
    let model = if projector.is_some() {
        Model::load_gguf_with_multimodal_options(&gguf_path, options)?
    } else {
        Model::load_gguf_with_options(&gguf_path, options)?
    };
    let eos_token_id = model.config().eos_token_id;
    eprintln!(
        "loaded Catena runtime and weights in {:.2?}",
        load_started.elapsed()
    );
    let multimodal = if let Some(path) = projector {
        let vision = VisionModel::load_gguf(&path, model.gpu_dialect(), args.image_max_pixels)?;
        let images = args
            .image
            .iter()
            .map(|path| vision.encode_image(path))
            .collect::<Result<Vec<_>>>()?;
        let prepared = PreparedMultimodalPrompt::new(&tokenizer, prompt, images)?;
        tokens = prepared.tokens().to_vec();
        Some(prepared)
    } else {
        None
    };
    if let Some(fifo_path) = std::env::var_os("CATENA_QWEN3_PROFILE_FIFO") {
        let pid_path = std::env::var_os("CATENA_QWEN3_PROFILE_PID_FILE").ok_or_else(|| {
            anyhow::anyhow!(
                "CATENA_QWEN3_PROFILE_PID_FILE is required with CATENA_QWEN3_PROFILE_FIFO"
            )
        })?;
        // This path is explicitly enabled only by the local profiling script.
        // Permit its sibling rocprofv3 process to attach under Yama ptrace_scope.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0);
        }
        std::fs::write(&pid_path, std::process::id().to_string())?;
        eprintln!(
            "weights ready; waiting for profiler handshake on {} (PID {})",
            PathBuf::from(&fifo_path).display(),
            std::process::id()
        );
        // Opening the FIFO blocks until the script has started rocprofv3. A
        // one-byte read then provides an explicit release signal.
        let mut fifo = File::open(&fifo_path)?;
        let mut release = [0_u8; 1];
        std::io::Read::read_exact(&mut fifo, &mut release)?;
        eprintln!("profiler handshake complete; starting inference");
    }
    if let Some(delay) = std::env::var_os("CATENA_QWEN3_START_DELAY_SECS") {
        let seconds = delay
            .to_string_lossy()
            .parse::<u64>()
            .map_err(|error| anyhow::anyhow!("invalid CATENA_QWEN3_START_DELAY_SECS: {error}"))?;
        // Permit a sibling rocprofv3 process to attach on Linux systems with
        // Yama ptrace_scope enabled. This path is opt-in via the delay variable.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0);
        }
        eprintln!(
            "waiting {seconds}s before inference for profiler attachment (PID {})",
            std::process::id()
        );
        std::thread::sleep(std::time::Duration::from_secs(seconds));
    }

    let mut stdout = io::stdout().lock();
    let mut decode_stream = tokenizer.decode_stream(true);
    if multimodal.is_some() {
        writeln!(stdout, "{prompt}")?;
    } else {
        for &token in &tokens {
            if let Some(chunk) = decode_stream
                .step(u32::try_from(token)?)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
            {
                write!(stdout, "{chunk}")?;
            }
        }
    }
    stdout.flush()?;

    if args.max_new_tokens == 0 {
        writeln!(stdout)?;
        return Ok(());
    }

    let capacity = tokens
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| anyhow::anyhow!("generation capacity overflowed"))?;
    let mut session = model.session(capacity)?;
    let generation_started = Instant::now();
    let prompt_started = Instant::now();
    let mut next_token = if let Some(prepared) = &multimodal {
        session.prefill_multimodal_next_token(prepared)?
    } else {
        session.prefill_next_token(&tokens)?
    };
    let prompt_elapsed = prompt_started.elapsed();
    let mut eval_elapsed = std::time::Duration::ZERO;
    let mut generated = 0_usize;
    for index in 0..args.max_new_tokens {
        let token = if index == 0 {
            next_token
        } else {
            let eval_started = Instant::now();
            next_token = session.decode_next_token(next_token)?;
            eval_elapsed += eval_started.elapsed();
            next_token
        };
        tokens.push(token);
        generated += 1;
        if let Some(chunk) = decode_stream
            .step(u32::try_from(token)?)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
        {
            write!(stdout, "{chunk}")?;
            stdout.flush()?;
        }
        if Some(token) == eos_token_id && !args.ignore_eos {
            break;
        }
    }
    writeln!(stdout)?;
    stdout.flush()?;
    let elapsed = generation_started.elapsed();
    let tokens_per_second = if elapsed.is_zero() {
        f64::INFINITY
    } else {
        generated as f64 / elapsed.as_secs_f64()
    };
    let prompt_tokens_per_second =
        tokens.len().saturating_sub(generated) as f64 / prompt_elapsed.as_secs_f64();
    let eval_tokens = generated.saturating_sub(1);
    let eval_tokens_per_second = if eval_elapsed.is_zero() {
        f64::INFINITY
    } else {
        eval_tokens as f64 / eval_elapsed.as_secs_f64()
    };
    eprintln!(
        "prompt eval: {} token(s) in {prompt_elapsed:.2?} ({prompt_tokens_per_second:.2} tok/s)",
        tokens.len().saturating_sub(generated)
    );
    eprintln!(
        "cached eval: {eval_tokens} token(s) in {eval_elapsed:.2?} ({eval_tokens_per_second:.2} tok/s)"
    );
    eprintln!("generated {generated} token(s) in {elapsed:.2?} ({tokens_per_second:.2} tok/s)");
    Ok(())
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("invalid positive integer: {error}"))?;
    if value == 0 {
        Err("value must be greater than zero".into())
    } else {
        Ok(value)
    }
}

#[derive(Debug, Clone)]
struct BenchRow {
    test: String,
    mean_tokens_per_second: f64,
    stddev_tokens_per_second: Option<f64>,
}

fn run_benchmark(
    model: &Model,
    pp: usize,
    tg: usize,
    tg_context: usize,
    runs: usize,
) -> Result<()> {
    let mut rows = Vec::with_capacity(2);
    if pp > 0 {
        let tokens = synthetic_tokens(pp, model.config().vocabulary_size);
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let mut session = model.session(pp)?;
            let started = Instant::now();
            session.prefill_next_token(&tokens)?;
            samples.push((pp, started.elapsed()));
        }
        let (mean, stddev) = tokens_per_second_stats(&samples);
        rows.push(BenchRow {
            test: format!("pp{pp}"),
            mean_tokens_per_second: mean,
            stddev_tokens_per_second: stddev,
        });
    }
    if tg > 0 {
        let total = tg_context
            .checked_add(tg)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| anyhow::anyhow!("benchmark capacity overflow"))?;
        let tokens = synthetic_tokens(total, model.config().vocabulary_size);
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let mut session = model.session(total)?;
            session.prefill_next_token(&tokens[..tg_context])?;
            for &token in &tokens[tg_context..tg_context + 2] {
                session.decode_next_token(token)?;
            }
            let started = Instant::now();
            for &token in &tokens[tg_context + 2..] {
                session.decode_next_token(token)?;
            }
            samples.push((tg, started.elapsed()));
        }
        let (mean, stddev) = tokens_per_second_stats(&samples);
        rows.push(BenchRow {
            test: format!("tg{tg}/ctx{tg_context}"),
            mean_tokens_per_second: mean,
            stddev_tokens_per_second: stddev,
        });
    }
    print_benchmark_table(model, &rows);
    Ok(())
}

fn synthetic_tokens(count: usize, vocabulary_size: usize) -> Vec<u64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state % vocabulary_size as u64
        })
        .collect()
}

fn tokens_per_second_stats(samples: &[(usize, Duration)]) -> (f64, Option<f64>) {
    let rates = samples
        .iter()
        .map(|(tokens, elapsed)| *tokens as f64 / elapsed.as_secs_f64())
        .collect::<Vec<_>>();
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let stddev = (rates.len() > 1).then(|| {
        let variance =
            rates.iter().map(|rate| (rate - mean).powi(2)).sum::<f64>() / (rates.len() - 1) as f64;
        variance.sqrt()
    });
    (mean, stddev)
}

fn print_benchmark_table(model: &Model, rows: &[BenchRow]) {
    let model_width = 40_usize.max(model.display_name().chars().count());
    println!(
        "| {:model_width$} | {:10} | {:10} | {:>15} | {:>20} |",
        "model", "size", "params", "test", "t/s"
    );
    println!(
        "| {} | {} | {} | {} | {} |",
        "-".repeat(model_width),
        "-".repeat(10),
        "-".repeat(10),
        "-".repeat(15),
        "-".repeat(20)
    );
    let size = format_byte_size(model.weight_byte_len());
    let params = format_parameter_count(model.parameter_count());
    for row in rows {
        let rate = row.stddev_tokens_per_second.map_or_else(
            || format!("{:.2}", row.mean_tokens_per_second),
            |stddev| format!("{:.2} ± {stddev:.2}", row.mean_tokens_per_second),
        );
        println!(
            "| {:model_width$} | {:>10} | {:>10} | {:>15} | {:>20} |",
            model.display_name(),
            size,
            params,
            row.test,
            rate
        );
    }
}

fn format_byte_size(bytes: usize) -> String {
    const GIB: f64 = (1_u64 << 30) as f64;
    const MIB: f64 = (1_u64 << 20) as f64;
    if bytes >= 1 << 30 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.2} MiB", bytes as f64 / MIB)
    }
}

fn format_parameter_count(parameters: usize) -> String {
    if parameters >= 1_000_000_000 {
        format!("{:.2} B", parameters as f64 / 1_000_000_000.0)
    } else {
        format!("{:.2} M", parameters as f64 / 1_000_000.0)
    }
}
