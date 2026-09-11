use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use catena_qwen3::{
    catena::{
        InferenceOptions, InferencePrecision, MatrixCoreMode, generated_model_source_for_options,
    },
    checkpoint::read_gguf_model_config,
    hfhub::download_gguf,
    model::Model,
    tokenizer::QwenTokenizer,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Greedy BF16 text generation with dense or MoE Qwen3 on Catena/HIP")]
struct Args {
    /// Text prefix to continue.
    #[arg(required_unless_present_any = ["generate_entrypoint", "bench"])]
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

    /// Number of measured runs per enabled benchmark test.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
    bench_runs: usize,

    /// Fallback directory for tokenizer.json and model-bf16.gguf.
    #[arg(long, default_value = "unsloth/Qwen3-30B-A3B-Base")]
    model_dir: PathBuf,

    /// Tokenizer JSON override (defaults to GGUF metadata, then <model-dir>/tokenizer.json).
    #[arg(long, value_name = "PATH")]
    tokenizer: Option<PathBuf>,

    /// GGUF file, shard, or shard directory (defaults to <model-dir>/model-bf16.gguf).
    #[arg(long, conflicts_with = "model")]
    gguf: Option<PathBuf>,

    /// Hugging Face model as owner/repo[:variant], e.g. unsloth/Qwen3.5-0.8B-GGUF:BF16.
    #[arg(long, value_name = "MODEL", conflicts_with = "gguf")]
    model: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 20)]
    max_new_tokens: usize,

    /// Continue through EOS to produce a fixed-length benchmark.
    #[arg(long)]
    ignore_eos: bool,

    /// Use the native mixed-BF16 graph (Qwen3 dense and Qwen3.5/3.6 dense or MoE).
    #[arg(long)]
    bf16: bool,

    /// Use BF16 WMMA matrix cores for supported Qwen prefill (experimental).
    #[arg(long, requires = "bf16")]
    matrix_cores: bool,

    /// Use the platform BLAS library for BF16 GEMM during supported Qwen prefill.
    #[arg(long, requires = "bf16", conflicts_with = "matrix_cores")]
    blas: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let precision = if args.bf16 {
        InferencePrecision::Bf16
    } else {
        InferencePrecision::F32
    };
    let matrix_cores = if args.matrix_cores {
        MatrixCoreMode::Wmma
    } else if args.blas {
        MatrixCoreMode::Blas
    } else {
        MatrixCoreMode::Disabled
    };
    let options = InferenceOptions::new(precision).with_matrix_cores(matrix_cores);
    let hub_gguf = args.model.as_deref().map(download_gguf).transpose()?;
    let selected_gguf = match (&args.gguf, &hub_gguf) {
        (Some(path), None) => path.clone(),
        (None, Some(checkpoint)) => {
            eprintln!(
                "selected {} GGUF file(s) from the Hugging Face cache",
                checkpoint.files.len()
            );
            checkpoint.primary.clone()
        }
        (None, None) => args.model_dir.join("model-bf16.gguf"),
        (Some(_), Some(_)) => unreachable!("clap rejects --gguf with --model"),
    };
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
        return run_benchmark(&model, args.pp, args.tg, args.bench_runs);
    }

    let prompt = args
        .prompt
        .as_deref()
        .expect("clap requires a prompt unless entrypoint generation was selected");
    let gguf_path = selected_gguf;
    let tokenizer = if let Some(tokenizer_path) = args.tokenizer {
        QwenTokenizer::open(tokenizer_path)?
    } else if let Some(tokenizer) = QwenTokenizer::from_gguf(&gguf_path)? {
        eprintln!("loaded tokenizer from GGUF metadata");
        tokenizer
    } else {
        QwenTokenizer::open(args.model_dir.join("tokenizer.json"))?
    };
    let mut tokens = tokenizer.encode(prompt)?;
    ensure!(
        !tokens.is_empty(),
        "the prompt encoded to an empty token prefix"
    );

    let load_started = Instant::now();
    let model = Model::load_gguf_with_options(&gguf_path, options)?;
    let eos_token_id = model.config().eos_token_id;
    eprintln!(
        "loaded Catena runtime and weights in {:.2?}",
        load_started.elapsed()
    );
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
    for &token in &tokens {
        if let Some(chunk) = decode_stream
            .step(u32::try_from(token)?)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
        {
            write!(stdout, "{chunk}")?;
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
    let mut next_token = session.prefill_next_token(&tokens)?;
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
    tokens_per_second: f64,
}

fn run_benchmark(model: &Model, pp: usize, tg: usize, runs: usize) -> Result<()> {
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
        rows.push(BenchRow {
            test: format!("pp{pp}"),
            tokens_per_second: mean_tokens_per_second(&samples),
        });
    }
    if tg > 0 {
        let tokens = synthetic_tokens(tg, model.config().vocabulary_size);
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let mut session = model.session(tg)?;
            let started = Instant::now();
            session.prefill_next_token(&tokens[..1])?;
            for &token in &tokens[1..] {
                session.decode_next_token(token)?;
            }
            samples.push((tg, started.elapsed()));
        }
        rows.push(BenchRow {
            test: format!("tg{tg}"),
            tokens_per_second: mean_tokens_per_second(&samples),
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

fn mean_tokens_per_second(samples: &[(usize, Duration)]) -> f64 {
    samples
        .iter()
        .map(|(tokens, elapsed)| *tokens as f64 / elapsed.as_secs_f64())
        .sum::<f64>()
        / samples.len() as f64
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
        println!(
            "| {:model_width$} | {:>10} | {:>10} | {:>15} | {:>20.2} |",
            model.display_name(),
            size,
            params,
            row.test,
            row.tokens_per_second
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
    use clap::Parser;

    #[test]
    fn benchmark_cli_defaults_do_not_require_a_prompt() {
        let args = Args::try_parse_from(["catena-qwen3", "--bench"]).unwrap();
        assert!(args.bench);
        assert_eq!(args.pp, 512);
        assert_eq!(args.tg, 128);
        assert_eq!(args.bench_runs, 1);
        assert!(args.prompt.is_none());
        assert!(!args.bf16);
    }

    #[test]
    fn bf16_graph_is_explicitly_selected() {
        let args = Args::try_parse_from([
            "catena-qwen3",
            "--bf16",
            "--model",
            "unsloth/Qwen3-0.6B-GGUF:BF16",
            "hello",
        ])
        .unwrap();
        assert!(args.bf16);
    }

    #[test]
    fn matrix_cores_require_bf16() {
        assert!(Args::try_parse_from(["catena-qwen3", "--matrix-cores", "hello"]).is_err());
        let args =
            Args::try_parse_from(["catena-qwen3", "--bf16", "--matrix-cores", "hello"]).unwrap();
        assert!(args.matrix_cores);
    }

    #[test]
    fn blas_requires_bf16_and_conflicts_with_matrix_cores() {
        assert!(Args::try_parse_from(["catena-qwen3", "--blas", "hello"]).is_err());
        let args = Args::try_parse_from(["catena-qwen3", "--bf16", "--blas", "hello"]).unwrap();
        assert!(args.blas);
        assert!(
            Args::try_parse_from([
                "catena-qwen3",
                "--bf16",
                "--matrix-cores",
                "--blas",
                "hello"
            ])
            .is_err()
        );
    }

    #[test]
    fn benchmark_prompt_is_accepted_and_ignored_by_the_mode() {
        let args =
            Args::try_parse_from(["catena-qwen3", "--bench", "--pp", "8", "ignored"]).unwrap();
        assert_eq!(args.prompt.as_deref(), Some("ignored"));
        assert_eq!(args.pp, 8);
    }

    #[test]
    fn benchmark_runs_must_be_positive() {
        assert!(Args::try_parse_from(["catena-qwen3", "--bench", "--bench-runs", "0"]).is_err());
    }

    #[test]
    fn benchmark_and_entrypoint_generation_conflict() {
        assert!(
            Args::try_parse_from(["catena-qwen3", "--bench", "--generate-entrypoint"]).is_err()
        );
    }

    #[test]
    fn hub_model_and_local_gguf_conflict() {
        assert!(
            Args::try_parse_from([
                "catena-qwen3",
                "--model",
                "unsloth/Qwen3.5-0.8B-GGUF:BF16",
                "--gguf",
                "model.gguf",
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
        assert_eq!(mean_tokens_per_second(&samples), 7.5);
    }
}
