# Catena inference

Native Catena GPU inference for Qwen3 dense and Qwen3.5/3.6 dense or MoE text
models, Qwen3.5 images, Whisper ASR, Flux2 Klein, and MMS English TTS.
All applications use one compiler backend and explicit Hex GPU kernels.

Model selection uses Hugging Face names. The Hub client manages downloads,
shards, revisions, and caches. `HF_TOKEN`, `HF_HOME`, and `HF_HUB_CACHE` retain
their normal meanings. Qwen tokenizers and model dimensions come from GGUF.
BF16/F32 checkpoint weights retain their native storage.

## Qwen text and images

```bash
cargo run -r -- --model unsloth/Qwen3.5-2B-GGUF:BF16 --prompt 'Gravity is'
cargo run -r -- --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 --prompt 'Gravity is'
cargo run -r -- --model unsloth/Qwen3.5-0.8B-GGUF:BF16 \
  --image photo.jpg --prompt 'Describe this image.'
```

The `:variant` suffix selects an unambiguous checkpoint, including split BF16
checkpoints. Image projectors come from the same repository or an explicit
`--mmproj-model`. Multiple `--image` arguments preserve their order;
`--image-max-pixels` limits resized image area.

Qwen always uses BF16 on HIP gfx11. Qwen3 dense supports `--kernels fused`
(the default) and `--kernels reference`; Qwen3.5/3.6 requires fused kernels.
The older Qwen3 MoE architecture and Qwen CUDA execution are unsupported.
Compiler, precision, WMMA, BLAS, and legacy attention selectors have been removed.

The Rust source renderers live under `src/qwen`; shared and model-specific Hex
kernels live under `src/catena`. Hex owns allocation, scheduling, layer execution,
cache updates, and generation. Rust passes opaque model-owned session state
through the `qwen3.init` and text/image entrypoints. BF16 cache boundaries and
F32 recurrent workspace are preserved.

[Image handling](docs/qwen35-images.md),
[kernel details and historical measurements](docs/qwen-kernels.md), and
[prefill measurements](PREFILL_BENCHMARKS.md).

## Benchmarks and validation

```bash
cargo run -r -- --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 \
  --bench --pp 0 --tg 128 --tg-context 32 --bench-runs 5
cargo test --all-targets
CATENA_QWEN3_GPU_TESTS=1 cargo test -r --lib qwen:: -- --nocapture
```

GPU tests are opt-in. Checkpoint tests accept `CATENA_QWEN3_MODEL`,
`CATENA_QWEN35_MODEL`, and `CATENA_QWEN36_MODEL` as HF model selectors.
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

The text encoder and transformer use BF16 kernels; the VAE uses F32.
`--width`, `--height`, `--steps`, `--seed`, and `--noise` control generation.
`--dump-dir` exports intermediate tensors and `--startup-timings` reports startup
costs. `--generate-entrypoint DIR` exports the three complete Hex stage assets.
There is no backend selector. Exported assets require Catena's matching core
library. Regenerate kernels with `python3 tools/generate-flux-kernels.py` and
network compositions with `python3 scripts/generate-flux-networks.py`.

## MMS English TTS

```bash
cargo run -r --bin mms-tts -- --output speech.wav 'Hello from Catena.'
```

The model defaults to `facebook/mms-tts-eng`; use `--model` to override it.

MMS uses BF16 learned tensors and F32 numerical calculations, producing 16 kHz
WAV audio. Seed, speaking rate, noise scales, tensor dumps, inspection, and
entrypoint export remain available. Regenerate its core GPU kernels and weight
bindings with `python3 scripts/generate-tts-hex.py`. Preparation returns a one-element
device length header; Rust validates the frame count before invoking the Hex
decoder. The public synthesis API and normal/debug result layouts are preserved.

## Compiler integration

`src/catena.rs` contains the sole compiler adapter. It uses the compiler's
`CodegenKind::Default` backend and `core_sources()` standard library helpers;
applications have no backend choice.
