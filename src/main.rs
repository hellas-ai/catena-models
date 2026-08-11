use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Result, ensure};
use catena_qwen3::{model::Model, spec::EOS_TOKEN_ID, tokenizer::QwenTokenizer};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Greedy text generation with Qwen3-30B-A3B-Base BF16 on Catena/HIP")]
struct Args {
    /// Text prefix to continue.
    prompt: String,

    /// Directory containing model.safetensors and tokenizer.json.
    #[arg(long, default_value = "models/Qwen3-30B-A3B-Base")]
    model_dir: PathBuf,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 20)]
    max_new_tokens: usize,

    /// Continue through EOS to produce a fixed-length benchmark.
    #[arg(long)]
    ignore_eos: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer = QwenTokenizer::open(args.model_dir.join("tokenizer.json"))?;
    let mut tokens = tokenizer.encode(&args.prompt)?;
    ensure!(
        !tokens.is_empty(),
        "the prompt encoded to an empty token prefix"
    );

    let load_started = Instant::now();
    let model = Model::load(&args.model_dir)?;
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
        if token == EOS_TOKEN_ID && !args.ignore_eos {
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
