use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Result, ensure};
use catena_qwen3::{
    catena::generated_model_source_for, checkpoint::read_gguf_model_config, model::Model,
    tokenizer::QwenTokenizer,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Greedy text generation with Qwen3-30B-A3B-Base BF16 on Catena/HIP")]
struct Args {
    /// Text prefix to continue.
    #[arg(required_unless_present = "generate_entrypoint")]
    prompt: Option<String>,

    /// Regenerate the explicit entrypoint in src/catena/qwen3.hex, then exit.
    #[arg(long, conflicts_with = "prompt")]
    generate_entrypoint: bool,

    /// Fallback directory for tokenizer.json and model-bf16.gguf.
    #[arg(long, default_value = "unsloth/Qwen3-30B-A3B-Base")]
    model_dir: PathBuf,

    /// Tokenizer JSON override (defaults to GGUF metadata, then <model-dir>/tokenizer.json).
    #[arg(long, value_name = "PATH")]
    tokenizer: Option<PathBuf>,

    /// GGUF file, shard, or shard directory (defaults to <model-dir>/model-bf16.gguf).
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 20)]
    max_new_tokens: usize,

    /// Continue through EOS to produce a fixed-length benchmark.
    #[arg(long)]
    ignore_eos: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let selected_gguf = args
        .gguf
        .clone()
        .unwrap_or_else(|| args.model_dir.join("model-bf16.gguf"));
    if args.generate_entrypoint {
        let config = read_gguf_model_config(&selected_gguf)?;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/catena/qwen3.hex");
        std::fs::write(&path, generated_model_source_for(&config)?)?;
        println!("regenerated {}", path.display());
        return Ok(());
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
    let model = Model::load_gguf(&gguf_path)?;
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
