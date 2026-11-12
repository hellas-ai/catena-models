# Catena inference

Native Catena GPU inference for Qwen3 dense and Qwen3.5/3.6 dense or MoE text
models, LFM3 text and images, Qwen3.5 images, Whisper ASR, Flux2 Klein, and MMS English TTS.
All applications use one compiler backend and explicit Hex GPU kernels.

Model selection uses Hugging Face names. The Hub client manages downloads,
shards, revisions, and caches. `HF_TOKEN`, `HF_HOME`, and `HF_HUB_CACHE` retain
their normal meanings. Text tokenizers and model dimensions come from GGUF.
BF16/F32 checkpoint weights retain their native storage.

## GPU selection

HIP is the default. Set `CATENA_GPU_DIALECT=cuda` to use CUDA for any application,
including Qwen images, Flux, Whisper, and MMS TTS:

```bash
CATENA_GPU_DIALECT=cuda cargo run -r -- \
  --model unsloth/Qwen3.5-0.8B-GGUF:BF16 --prompt 'Gravity is'
CATENA_GPU_DIALECT=cuda cargo run -r --bin whisper-asr -- speech.wav
```

## Qwen text and images

If `--model` is omitted, CLI and server modes use
`unsloth/Qwen3.6-35B-A3B-GGUF:BF16`.

```bash
cargo run -r -- --model unsloth/Qwen3.5-2B-GGUF:BF16 --prompt 'Gravity is'
cargo run -r -- --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 --prompt 'Gravity is'
cargo run -r -- --model unsloth/Qwen3.5-0.8B-GGUF:BF16 \
  --image photo.jpg --prompt 'Describe this image.'
```

Use `--serve` to keep a Qwen model loaded and expose text Chat Completions and
Responses APIs, including streaming, reasoning output, and client-executed tools:

```bash
cargo run -r -- --model unsloth/Qwen3.5-0.8B-GGUF:BF16 --serve --port 8000
```

See [serving and coding-client setup](SERVING.md) for remote binding,
authentication, request limits, tool calls, protocol logging with
`RUST_LOG=catena_models::server::protocol=debug`, and the supported API subset.

WMMA prefill is enabled by default in both the Qwen CLI and server for the
Qwen3.5/3.6 path. `--wmma-prefill` remains accepted; use `--no-wmma-prefill` to
select SIMT explicitly. Qwen3 models require `--no-wmma-prefill` because their
prefill implementation does not support this WMMA path.

On RDNA3 HIP and NVIDIA sm_80+ CUDA GPUs, prefill uses tiled attention and BF16
matrix-core projections. CUDA uses native `mma.sync` instructions; HIP uses WMMA.
Dense models also use WMMA gate/up/down FFN projections; MoE models use grouped
expert projections. DeltaNet's QKV, Z, alpha, and beta input projections use WMMA
during prefill; token decode and the vocabulary projection retain SIMT arithmetic.
Dense HIP prefill selects larger block projections and an eight-lane DeltaNet
scan; small or unaligned input shapes and CUDA retain the existing projection fallback.
Dense vocabulary selection uses the shared parallel argmax helper. MoE kernel
selection is unchanged. See [dense performance measurements](QWEN35_DENSE_PERFORMANCE.md).
WMMA changes accumulation order and can change logits and generated tokens.
Set `CATENA_WMMA_COMPONENTS=safe` to use the exact tiled attention variant and,
for MoE, exact experts. Dense FFNs retain SIMT arithmetic in this safe selection.
See [prefill measurements](PREFILL_BENCHMARKS.md) for numerical checks and
component ablations. The model graph and reduction loops share Hex code; small
CUDA/HIP Hex helpers select native fragment loads, MMA operations, and output coordinates.
`CATENA_WMMA_EMULATE=1` remains a HIP arithmetic diagnostic.

All GPU model exponentials use the shared `stdlib.models.math.exp` Hex implementation,
including DeltaNet decay and online attention. It uses fixed F32 arithmetic,
nearest-even subnormal rounding, and canonical F32 NaNs; its accuracy target is
at most 1 ULP with round-to-nearest arithmetic and FMA contraction disabled.
This changes results previously produced by native `expf` or the older Hex
approximation. Packed-dot arithmetic and host preprocessing still require
separate work for complete cross-platform bitwise agreement.

Shared Hex helpers live under `stdlib.models.math`, `stdlib.models.memory`,
`stdlib.models.resources`, and `stdlib.models.coop`. The `stdlib.` prefix keeps
these helpers private and lets source pruning retain only reachable definitions.

The `:variant` suffix selects an unambiguous checkpoint, including split BF16
checkpoints. Image projectors come from the same repository or an explicit
`--mmproj-model`. Multiple `--image` arguments preserve their order;
`--image-max-pixels` limits resized image area.

The Rust source renderers live under `src/qwen`; shared and model-specific Hex
kernels live under `src/catena`. Hex owns allocation, scheduling, layer execution,
cache updates, and generation. Rust passes opaque model-owned session state
through the `qwen3.init` and text/image entrypoints. BF16 cache boundaries and
F32 recurrent workspace are preserved.

[Image handling](docs/qwen35-images.md),
[kernel details and historical measurements](docs/qwen-kernels.md), and
[prefill measurements](PREFILL_BENCHMARKS.md).

## LFM3 text and images

The BF16 text checkpoint uses the same CLI and chat server:

```bash
cargo run -r -- --model LiquidAI/LFM3-1.7B-GGUF:BF16 --prompt 'The capital of France is' -n 32
cargo run -r -- --model LiquidAI/LFM3-1.7B-GGUF:BF16 --serve --ctx-size 4096 --port 8000
```

HIP WMMA accelerates batched projections and attention. Single-token decode
uses portable F32 reductions, without packed BF16 dot instructions. Use
`--no-wmma-prefill` for the SIMT comparison path. The model reads layer types,
sliding windows, RoPE factors, tokenizer, and chat template from GGUF. Sessions
retain convolution history and attention caches across appended prompts and
checkpoint/resume. Audio, omni, and CUDA optimization are deferred.

The CLI takes a raw text prefix; server requests use the embedded chat template,
including its BOS token and optional thinking. The server's existing greedy
sampling policy applies.

For still-image understanding, use the VL checkpoint and the existing image options:

```bash
cargo run -r -- --model LiquidAI/LFM3-VL-2B-GGUF:BF16 \
  --image photo.jpg --prompt 'Describe this image.' -n 128
```

The companion BF16 projector downloads automatically. Multiple `--image` arguments
preserve image order; `--image-max-pixels` can reduce vision encoding cost.
The CLI and `PreparedMultimodalPrompt` API use the checkpoint's image chat template.
See [LFM3 image support](LFM3_VISION.md) for implementation and validation details.

To run the focused checkpoint/compiler and GPU checks, set `LFM3_GGUF` to the
cached BF16 file and run `cargo test -r --test lfm3 -- --ignored --nocapture`.

## Benchmarks and validation

```bash
cargo run -r -- --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 \
  --bench --pp 0 --tg 128 --tg-context 32 --bench-runs 5
cargo test --all-targets
```

The default tests need no GPU or model downloads. Run the example applications
above and below to exercise model compilation and GPU execution.

External-reference checks are ignored by default. For Qwen images, set
`QWEN35_REFERENCE` to a reference directory and `QWEN35_MODEL` to an HF model
selector; optionally set `QWEN35_MMPROJ_MODEL` for a separate projector repository.
The native inference check requires a supported HIP or CUDA GPU; the tokenizer
check requires no GPU.

```bash
cargo test -r --test qwen35_images -- --ignored --nocapture
```

For Flux, set `FLUX_TEST_REFERENCE` to a reference dump directory and
`FLUX_TEST_DUMP` to an output directory. This check requires a HIP or CUDA GPU
and the model weights, and enforces VAE accuracy; transformer error is diagnostic only.

```bash
cargo test -r --lib flux::kernel_tests::cached_components_match_reference -- --ignored --nocapture
```

Explicitly invoked reference checks fail if required inputs or hardware are
unavailable.

The shared compiler adapter caches generated GPU C++ and its complete public ABI
before native compilation. Repeated Qwen loads skip Hex parsing, checking and
closure conversion. Cache keys include ordered source contents, compiler and
stdlib changes (including uncommitted edits), backend, GPU dialect, and the
adapter's source-pruning policy. Native GPU compilation still runs on each load.
[Compilation measurements](COMPILATION_BENCHMARKS.md) compare kernel deduplication
and cache reuse.

`CATENA_COMPILE_TIMINGS=1` reports module-cache hits/misses and Hex compilation
time. `CATENA_HEX_CACHE=0` disables the module cache; `CATENA_GPU_CACHE=0` also
disables it. Override its directory with `CATENA_GPU_CACHE_DIR` (entries go under
`hex/`); the default is `$XDG_CACHE_HOME/catena/gpu/hex`, falling back to
`$HOME/.cache/catena/gpu/hex`. Invalid entries are rebuilt, and unavailable caches
fall back to normal compilation.

`CATENA_QWEN3_CPP` optionally exports generated C++ for diagnostics.
`--generate-entrypoint` exports Qwen Hex without initializing a GPU.

## Whisper ASR

```bash
cargo run -r --bin whisper-asr -- speech.wav
cargo run -r --bin whisper-asr -- --model openai/whisper-small speech.wav
```

Whisper accepts 16 kHz WAV input up to 30 seconds and defaults to Whisper small.
Its model, kernels, cache updates, and bounded generation use the same compiler
adapter as the other applications. `--language`, `--max-new-tokens`, `--timings`,
and `--dump-dir` control transcription and diagnostics. `--inspect`,
`--features-only`, and `--generate-entrypoint PATH` support offline inspection.

[Whisper validation](docs/whisper-asr.md).

## Flux2 Klein

```bash
cargo run -r --bin flux2-klein -- \
  --prompt 'A red fox sitting in a snowy forest' --output fox.png
```

Flux uses native BF16 matrix cores on gfx11 HIP and sm_80+ CUDA GPUs. Set
`CATENA_GPU_DIALECT=cuda` for CUDA; HIP remains the default. The text encoder and
transformer use tiled projections and fused attention. The VAE uses tiled
convolutions, projections, and 512-channel attention, with F32 accumulation,
normalization, residuals, and RGB output. CUDA kernels compile offline with
nvcc; CUDA execution and performance have not yet been verified on hardware.
See [WMMA implementation and measurements](FLUX_WMMA.md).
`--width`, `--height`, `--steps`, `--seed`, and `--noise` control generation.
`--dump-dir` exports intermediate tensors and `--startup-timings` reports startup
costs. `--generate-entrypoint DIR` exports the three complete Hex stage assets.
Compile exported assets through this repository's compiler adapter, which supplies
Catena's core library and the selected HIP/CUDA tile helpers. Regenerate WMMA kernels
with `python3 tools/generate-flux-wmma-linear.py` and
`python3 tools/generate-flux-wmma-attention.py`; VAE kernels use
`python3 tools/generate-flux-vae-wmma.py` and
`python3 tools/generate-flux-vae-attention.py`. The linear generator also emits
the VAE's F32-interface projections. Timings separate VAE decode, RGB
readback, and PNG conversion/write.

## MMS English TTS

```bash
cargo run -r --bin mms-tts -- --output speech.wav 'Hello from Catena.'
```

The model defaults to `facebook/mms-tts-eng`; use `--model` to override it.

MMS uses BF16 learned tensors and F32 numerical calculations, producing 16 kHz
WAV audio. Seed, speaking rate, noise scales, tensor dumps, inspection, and
entrypoint export remain available. Regenerate its core GPU kernels and weight
bindings with `python3 scripts/generate-tts-hex.py`.

Synthesis makes one Hex call. Hex preallocates a reusable decoder workspace
(about 152 MiB), predicts durations, validates the frame count on the GPU, and
generates the waveform. Kernels read the live length from device memory and
skip work beyond it; convolutions gather patches directly without im2col
buffers. Frame counts must be integral and within 1–1875 (30 seconds).
Invalid lengths produce zero valid samples and status 1, disabling decoder
arithmetic; success returns status 0. This uses upstream Catena operations,
without HostRead or an intermediate device-to-host transfer. Rust handles
tokenization, weight loading, result readback, and WAV conversion without
coordinating inference stages.

Exported bundles expose `tts.synthesize` and `tts.synthesize-debug`, each taking
token IDs, token count, synthesis options, and prepared weight references.
Preparation is private: callers no longer pass intermediate buffers or a frame
count. Normal results are a waveform buffer and a two-U64 device buffer
containing `[valid_sample_count, status]`. Only the waveform's valid prefix is
audio. Debug results append durations, stats, duration noise, latent noise, and
flow output; captures also use their live shape's prefix. Previously exported
bundles must be regenerated for this result ABI. Tokenization and checkpoint
weight preparation remain outside this inference change.

`tests/mms_single.rs` checks compilation with upstream operations, and has
ignored GPU checks for device-side length validation, convolution bounds,
direct artifact execution, normal/debug agreement, and exact parity with a
saved short fixed-seed baseline.

## Compiler integration

`src/catena.rs` contains the sole compiler adapter. It uses the compiler's
`CodegenKind::Default` backend and `core_sources()` standard library helpers;
applications have no backend choice.
