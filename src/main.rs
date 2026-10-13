use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use catena_lang::codegen::CodegenKind;
use catena_qwen3::{
    catena::{
        ExperimentalKernelKind, InferenceOptions, InferencePrecision, MatrixCoreMode,
        generated_model_source_for_options,
    },
    checkpoint::read_gguf_model_config,
    hfhub::{download_gguf, download_mmproj},
    model::Model,
    tokenizer::QwenTokenizer,
    vision::{PreparedMultimodalPrompt, VisionModel},
};
use clap::{CommandFactory, FromArgMatches, Parser};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CodegenArg {
    Default,
    Experimental,
}
impl From<CodegenArg> for CodegenKind {
    fn from(arg: CodegenArg) -> Self {
        match arg {
            CodegenArg::Default => Self::Default,
            CodegenArg::Experimental => Self::Experimental,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Qwen text and Qwen3.5 image generation with native Catena GPU execution")]
struct Args {
    /// Compiler backend (experimental supports BF16 dense Qwen3 and dense/MoE Qwen3.5 on HIP gfx11).
    #[arg(long, value_enum, default_value_t = CodegenArg::Default)]
    codegen: CodegenArg,

    /// Experimental Hex kernels (reference retains the original scalar implementation).
    #[arg(long, value_enum, default_value_t = ExperimentalKernelKind::Fused)]
    kernels: ExperimentalKernelKind,

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

    /// Use the native mixed-BF16 graph (Qwen3 dense and Qwen3.5/3.6 dense or MoE).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true",
        require_equals = true)]
    bf16: bool,

    /// Use BF16 WMMA matrix cores for supported Qwen prefill (experimental).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true",
        require_equals = true)]
    matrix_cores: bool,

    /// Use the platform BLAS library for BF16 GEMM during supported Qwen prefill.
    #[arg(long, conflicts_with = "matrix_cores")]
    blas: bool,

    /// Use the experimental fused BF16 cached-attention kernel.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true",
        require_equals = true)]
    fused_attention: bool,
}

fn inference_options(args: &Args, matches: &clap::ArgMatches) -> Result<InferenceOptions> {
    let precision = if args.bf16 {
        InferencePrecision::Bf16
    } else {
        InferencePrecision::F32
    };
    let matrix_cores = if args.blas {
        MatrixCoreMode::Blas
    } else if args.matrix_cores {
        MatrixCoreMode::Wmma
    } else {
        MatrixCoreMode::Disabled
    };
    let options = InferenceOptions::new(precision)
        .with_matrix_cores(matrix_cores)
        .with_fused_attention(args.fused_attention);
    let mut options = options
        .with_codegen(args.codegen.into())
        .with_experimental_kernels(args.kernels);
    ensure!(
        args.codegen == CodegenArg::Experimental
            || matches.value_source("kernels") != Some(clap::parser::ValueSource::CommandLine),
        "--kernels requires --codegen experimental"
    );
    if args.codegen == CodegenArg::Experimental {
        ensure!(
            args.bf16,
            "experimental Qwen3-family inference requires BF16 (--bf16)"
        );
        ensure!(
            args.image.is_empty(),
            "experimental codegen does not yet support image inputs"
        );
        ensure!(!args.blas, "experimental codegen does not support --blas");
        for (name, enabled) in [
            ("matrix_cores", args.matrix_cores),
            ("fused_attention", args.fused_attention),
        ] {
            ensure!(
                !(enabled
                    && matches.value_source(name) == Some(clap::parser::ValueSource::CommandLine)),
                "experimental codegen does not support --{}; omit it or set it to false",
                name.replace('_', "-")
            );
        }
        options.matrix_cores = MatrixCoreMode::Disabled;
        options.fused_attention = false;
        ensure!(
            catena_qwen3::model::configured_gpu_dialect()? == catena_lang::runtime::GpuDialect::Hip,
            "experimental BF16 Qwen3 requires HIP gfx11; CUDA is not supported"
        );
    }
    Ok(options)
}

fn main() -> Result<()> {
    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches)?;
    let options = inference_options(&args, &matches)?;
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
        let vision_config = catena_qwen3::vision::VisionConfig::from_gguf(&path)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use catena_lang::codegen::CodegenKind;

    fn options_from_cli(extra: &[&str]) -> Result<InferenceOptions> {
        let mut argv = vec![
            "catena-qwen3",
            "--model",
            "unsloth/Qwen3.5-0.8B-GGUF:BF16",
            "--prompt",
            "Gravity is",
        ];
        argv.extend_from_slice(extra);
        let matches = Args::command().try_get_matches_from(argv)?;
        inference_options(&Args::from_arg_matches(&matches)?, &matches)
    }

    #[test]
    fn compiler_selection_preserves_default_and_disables_legacy_acceleration() -> Result<()> {
        let default = options_from_cli(&[])?;
        assert_eq!(default.codegen, CodegenKind::Default);
        assert_eq!(default.matrix_cores, MatrixCoreMode::Wmma);
        assert!(default.fused_attention);
        let experimental = options_from_cli(&["--codegen", "experimental"])?;
        assert_eq!(experimental.codegen, CodegenKind::Experimental);
        assert_eq!(experimental.precision, InferencePrecision::Bf16);
        assert_eq!(experimental.matrix_cores, MatrixCoreMode::Disabled);
        assert!(!experimental.fused_attention);
        assert!(options_from_cli(&["--codegen", "unknown"]).is_err());
        Ok(())
    }

    #[test]
    fn experimental_kernel_selection_requires_experimental_codegen() -> Result<()> {
        assert_eq!(
            options_from_cli(&["--codegen", "experimental"])?.experimental_kernels,
            ExperimentalKernelKind::Fused
        );
        assert_eq!(
            options_from_cli(&["--codegen", "experimental", "--kernels", "reference"])?
                .experimental_kernels,
            ExperimentalKernelKind::Reference
        );
        assert!(options_from_cli(&["--kernels", "fused"]).is_err());
        assert!(options_from_cli(&["--codegen", "experimental", "--kernels", "unknown"]).is_err());
        assert!(parse_args(["catena-qwen3", "--bench", "--tg-context", "0"]).is_err());
        Ok(())
    }

    #[test]
    fn experimental_rejects_explicit_legacy_options_and_f32() {
        for option in [
            "--matrix-cores",
            "--fused-attention",
            "--blas",
            "--bf16=false",
        ] {
            assert!(
                options_from_cli(&["--codegen", "experimental", option]).is_err(),
                "{option}"
            );
        }
        assert!(options_from_cli(&["--codegen", "experimental", "--image", "image.png"]).is_err());
        assert!(
            options_from_cli(&[
                "--codegen",
                "experimental",
                "--matrix-cores=false",
                "--fused-attention=false"
            ])
            .is_ok()
        );
    }

    #[test]
    fn benchmark_cli_defaults_do_not_require_a_prompt() {
        let args = parse_args(["catena-qwen3", "--bench"]).unwrap();
        assert!(args.bench);
        assert_eq!(args.pp, 512);
        assert_eq!(args.tg, 128);
        assert_eq!(args.bench_runs, 1);
        assert!(args.prompt.is_none());
        assert!(args.bf16);
        assert!(args.matrix_cores);
        assert!(args.fused_attention);
    }

    #[test]
    fn bf16_graph_is_explicitly_selected() {
        let args = parse_args([
            "catena-qwen3",
            "--bf16",
            "--model",
            "unsloth/Qwen3-0.6B-GGUF:BF16",
            "--prompt",
            "hello",
        ])
        .unwrap();
        assert!(args.bf16);
    }

    #[test]
    fn acceleration_defaults_do_not_require_explicit_bf16() {
        let args = parse_args(["catena-qwen3", "--prompt", "hello"]).unwrap();
        assert_eq!(args.prompt.as_deref(), Some("hello"));
        assert!(parse_args(["catena-qwen3", "hello"]).is_err());
        assert!(parse_args(["catena-qwen3"]).is_err());
        assert!(args.bf16);
        assert!(args.matrix_cores);
        assert!(args.fused_attention);

        let explicit = parse_args([
            "catena-qwen3",
            "--matrix-cores",
            "--fused-attention",
            "--prompt",
            "hello",
        ])
        .unwrap();
        assert!(explicit.bf16);
    }

    #[test]
    fn blas_overrides_the_default_and_conflicts_with_explicit_matrix_cores() {
        let args = parse_args(["catena-qwen3", "--blas", "--prompt", "hello"]).unwrap();
        assert!(args.blas);
        assert!(
            parse_args([
                "catena-qwen3",
                "--bf16",
                "--matrix-cores",
                "--blas",
                "--prompt",
                "hello"
            ])
            .is_err()
        );
    }

    #[test]
    fn benchmark_prompt_is_accepted_and_ignored_by_the_mode() {
        let args = parse_args([
            "catena-qwen3",
            "--bench",
            "--pp",
            "8",
            "--prompt",
            "ignored",
        ])
        .unwrap();
        assert_eq!(args.prompt.as_deref(), Some("ignored"));
        assert_eq!(args.pp, 8);
    }

    #[test]
    fn benchmark_runs_must_be_positive() {
        assert!(parse_args(["catena-qwen3", "--bench", "--bench-runs", "0"]).is_err());
    }

    #[test]
    fn benchmark_and_entrypoint_generation_conflict() {
        assert!(parse_args(["catena-qwen3", "--bench", "--generate-entrypoint"]).is_err());
    }

    #[test]
    fn hf_model_is_required_and_local_model_flags_are_rejected() {
        assert!(Args::try_parse_from(["catena-qwen3", "--prompt", "hello"]).is_err());
        assert!(
            parse_args([
                "catena-qwen3",
                "--model",
                "unsloth/Qwen3.5-0.8B-GGUF:BF16",
                "--gguf",
                "model.gguf",
                "--prompt",
                "prompt",
            ])
            .is_err()
        );
    }

    #[test]
    fn benchmark_helpers_are_deterministic_and_format_units() {
        assert_eq!(synthetic_tokens(4, 100), synthetic_tokens(4, 100));
        assert_eq!(format_byte_size(3 << 30), "3.00 GiB");
        assert_eq!(format_parameter_count(30_532_122_624), "30.53 B");
        let samples = [(10, Duration::from_secs(2)), (10, Duration::from_secs(1))];
        let (mean, stddev) = tokens_per_second_stats(&samples);
        assert_eq!(mean, 7.5);
        assert_eq!(stddev, Some(2.5_f64 * 2.0_f64.sqrt()));
        assert_eq!(tokens_per_second_stats(&samples[..1]), (5.0, None));
    }
}

#[cfg(test)]
mod image_cli_tests {
    use super::*;
    #[test]
    fn repeated_images_and_boolean_overrides_parse() {
        let args = parse_args([
            "catena-qwen3",
            "--image",
            "one.png",
            "--image",
            "two.jpg",
            "--mmproj-model",
            "owner/vision",
            "--bf16=false",
            "--matrix-cores=false",
            "--fused-attention=false",
            "--prompt",
            "Describe these.",
        ])
        .unwrap();
        assert_eq!(args.image.len(), 2);
        assert!(!args.bf16 && !args.matrix_cores && !args.fused_attention);
        let image_only = parse_args(["catena-qwen3", "--image", "one.png"]).unwrap();
        assert!(image_only.prompt.is_none());
    }
    #[test]
    fn images_conflict_with_bench_and_require_companion_options() {
        assert!(parse_args(["catena-qwen3", "--bench", "--image", "one.png"]).is_err());
        assert!(
            parse_args([
                "catena-qwen3",
                "--generate-entrypoint",
                "--image",
                "one.png"
            ])
            .is_err()
        );
        assert!(
            parse_args([
                "catena-qwen3",
                "--mmproj-model",
                "owner/vision",
                "--prompt",
                "Describe."
            ])
            .is_err()
        );
        assert!(
            parse_args([
                "catena-qwen3",
                "--image-max-pixels",
                "0",
                "--prompt",
                "Describe."
            ])
            .is_err()
        );
    }
}

#[cfg(test)]
fn parse_args<I, T>(args: I) -> std::result::Result<Args, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let mut args = args
        .into_iter()
        .map(Into::into)
        .collect::<Vec<std::ffi::OsString>>();
    if !args.iter().any(|a| a == "--model") {
        args.extend(["--model".into(), "unsloth/Qwen3.5-0.8B-GGUF:BF16".into()]);
    }
    Args::try_parse_from(args)
}
