# Qwen3-30B-A3B-Base reference runners

Local BF16 inference for the base (non-instruct) `Qwen3-30B-A3B-Base`
checkpoint through Transformers, llama.cpp, and an exploratory Catena GPU
implementation. The Catena path executes all 48 layers, uses a persistent KV
cache, and performs greedy generation directly from a GPU-registered BF16
GGUF mapping.

The Rust/Catena runner also accepts metadata-compatible dense Qwen3 GGUFs,
including the tied-embedding 0.6B and 1.7B layouts, dense hybrid Qwen3.5 GGUFs
such as `unsloth/Qwen3.5-0.8B-GGUF:BF16`, and hybrid MoE checkpoints such as
`unsloth/Qwen3.6-35B-A3B-GGUF:BF16`. Qwen3.5/3.6 selects full-attention or
Gated DeltaNet per layer from GGUF metadata and uses partial RoPE, gated
attention, depthwise causal convolution, and recurrent DeltaNet state. MoE
layers combine normalized top-k routed experts with their sigmoid-gated shared
expert. The conversion and measured performance notes below remain specific
to Qwen3-30B-A3B.

Reduction and softmax launch limits are owned by the selected Catena compiler
and target GPU. The runner does not duplicate those limits or reject model
shapes before compilation.

The scripts assume the Nix development environment; invoke them with
`nix develop --command` as shown below.

## Setup

The local compiler supports a shared [HIP artifact cache](docs/gpu-artifact-cache.md)
for Qwen, Flux, Whisper and MMS TTS. Warm runs reuse compiled libraries while still
lowering Hex to C++.

The Catena compiler dependency is pinned to the private
`hellas-ai/exploratory-catena` repository. Configure an SSH key with access to
that GitHub repository before fetching or building dependencies:

```bash
nix develop --command cargo fetch
```

Cargo fetches the compiler automatically; the local `smollm2-catena-fork`
symlink used during development is not required.

Download the original BF16 Hugging Face checkpoint (about 61 GB):

```bash
mkdir -p models/Qwen3-30B-A3B-Base
nix develop --command hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir models/Qwen3-30B-A3B-Base
```

Convert it to an unquantized BF16 GGUF used by both llama.cpp and Catena
(about another 61 GB):

```bash
nix develop --command ./setup-llama-cpp.sh
```

## How to run

### Transformers

```bash
nix develop --command ./run-transformers.py "The capital of France is" -n 20
```

Transformers uses BF16 on the CPU by default. A ROCm-enabled PyTorch build can
instead use `--device cuda` (PyTorch calls the ROCm device `cuda`).

### llama.cpp

```bash
nix develop --command env QWEN3_TOKENS=20 \
  ./run-llama-cpp.sh "The capital of France is"
```

The llama.cpp runner offloads every layer to ROCm. This machine is configured
with a 104 GiB AMDGPU GTT aperture so the 61 GB model and compute buffers fit in
unified GPU memory. Set `QWEN3_GPU_LAYERS` to override the layer count. Both
runners use raw text completion because this is a base model, not an
instruction-tuned model.

## Numbers comparison

BF16, greedy decoding, eight tokens from `"The capital of France is"` on the
Ryzen AI Max+ Pro 395 / Radeon 8060S:

| Implementation | Device | Load time | Prompt evaluation | Cached token evaluation |
| --- | --- | ---: | ---: | ---: |
| llama.cpp 9190 | Radeon 8060S, all layers | 4.59 s | 31.83 tok/s | 20.32 tok/s |
| Transformers 5.5.4 | CPU | 74.15 s | 10.03 tok/s | 4.21 tok/s |

Both produced the same continuation:

```text
The capital of France is Paris. Which of the following options correctly
```

Transformers reported generated token IDs
`[12095, 13, 15920, 315, 279, 2701, 2606, 12440]`. Its first, colder model
load took 125.41 seconds; the table records the repeated run used for the split
prompt/cached timings. The llama.cpp run used full ROCm offload with
`amdgpu.gttsize=98304`; Transformers used the flake's CPU-only PyTorch 2.11.0.

### Catena

```bash
nix develop --command cargo run --release -- \
  "The capital of France is" --max-new-tokens 20
```

HIP is the default backend. Select CUDA for compilation, runtime allocation,
and GGUF registration with `CATENA_GPU_DIALECT=cuda`, for example:

```bash
CATENA_GPU_DIALECT=cuda cargo run --release -- \
  --bf16 --blas --bench --pp 512 --tg 128
```

Catena validates and privately maps `model-bf16.gguf`, registers that single
mapping with the selected GPU runtime, and passes every tensor subrange
directly to the generated kernels in its native GGUF representation. The 338
large tensors remain BF16;
the 241 norms/router tensors emitted by llama.cpp's converter remain F32. The
loader validates this dtype contract and performs no weight repacking,
conversion, upload, or per-tensor allocation. Measured warm loads reached
weight readiness in 2.72--2.84 seconds.

Use a GGUF outside the model directory with `--gguf /path/to/model.gguf`.
For split checkpoints, `--gguf` can also name a shard directory or any one
standard `-00001-of-00002.gguf` shard. By default the tokenizer is reconstructed
from the GGUF's `tokenizer.ggml.*` metadata. Use
`--tokenizer /path/to/tokenizer.json` as an explicit override; when the GGUF has
no embedded tokenizer, `<model-dir>/tokenizer.json` remains the fallback.

Alternatively, let the runner list and cache a checkpoint from Hugging Face:

```bash
nix develop --command cargo run --release -- \
  --model unsloth/Qwen3.5-0.8B-GGUF:BF16 "The capital of France is"
```

Qwen3 dense and Qwen3.5/3.6 dense or MoE models have an opt-in native-BF16
execution graph. It keeps activations, attention probabilities, logits, and
persistent caches in BF16. Dense and routed expert projections use packed BF16
dot instructions with F32 register accumulation on gfx11. Router selection and
probability normalization, RMSNorm, softmax, RoPE, and the internal Gated
DeltaNet recurrence retain F32 calculations; DeltaNet inputs and recurrent
state are stored as BF16 between calls. The original F32-activation graph
remains the default so the two paths can be compared directly. Both graphs
update the persistent KV cache in place and use a tiled, coalesced cached-value
kernel for the exact attention context sum; global attention still scans the
existing history for each decoded token:

```bash
nix develop --command cargo run --release -- \
  --bf16 --model unsloth/Qwen3-0.6B-GGUF:BF16 \
  "The capital of France is"

nix develop --command cargo run --release -- \
  --bf16 --model unsloth/Qwen3-0.6B-GGUF:BF16 \
  --bench --pp 512 --tg 128 --bench-runs 5

nix develop --command cargo run --release -- \
  --bf16 --model unsloth/Qwen3.5-0.8B-GGUF:BF16 \
  --bench --pp 512 --tg 128 --bench-runs 5

nix develop --command cargo run --release -- \
  --bf16 --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 \
  --bench --pp 512 --tg 128 --bench-runs 5
```

For prefill experiments, add `--matrix-cores` or `--blas` alongside
`--bf16`. Both modes accelerate ordinary dense projections for Qwen3 dense,
Qwen3.5 dense, and Qwen3.5/3.6 MoE. In MoE models they also accelerate the
large shared-expert gate/up pair; dynamically routed experts and the narrow
F32 routers retain their native SIMT paths. Projections below 64 input rows
fall back to packed-BF16 SIMT, leaving token-at-a-time decode unchanged. The
BLAS mode uses hipBLAS on HIP and cuBLAS on CUDA. `--matrix-cores` and
`--blas` are mutually exclusive.

The BF16 graph currently rejects the older Qwen3 MoE architecture, which
continues to use the existing F32-activation graph.

The optional suffix after `:` selects a quantization or variant by directory
or filename. It is required when a repository contains more than one GGUF
checkpoint and selection would otherwise be ambiguous. Every member of a
standard sharded checkpoint is downloaded; `--model` and `--gguf` are mutually
exclusive. Normal Hugging Face environment variables such as `HF_TOKEN`,
`HF_HOME`, and `HF_HUB_CACHE` are honored.

Run llama.cpp-style prompt-processing and token-generation measurements with:

```bash
nix develop --command cargo run --release -- \
  --gguf /path/to/model.gguf --bench --pp 512 --tg 128
```

`--pp` and `--tg` default to 512 and 128; set either one to zero to skip that
row. `--bench-runs N` reports the mean throughput across `N` measured runs
(default 1). Benchmark mode ignores any positional prompt and uses deterministic
token IDs directly, so it does not load or require a tokenizer.

RoPE is also part of the Catena program rather than a model-specific Rust
input. Each invocation materializes only its current absolute-position range,
using exact F32 inverse-frequency constants and deterministic range-reduced
sine/cosine polynomials, then reuses that owned table across all 48 layers.
The generated table differs from the previous host `sinf`/`cosf` values by at
most `4.92e-7` across the complete 32,768-position context.

Normal inference compiles the dimension-free helpers in
`src/catena/qwen3_helpers.hex` together with model-specific Hex rendered from
`src/catena/qwen3_model.hex` and, for Qwen3.5, the hybrid layer definitions in
`src/catena/qwen35_model.hex`. Rust fills its embedding, normalization,
projection, RoPE, cache, attention, DeltaNet, and dense/MoE feed-forward
dimensions from GGUF metadata and appends a linear-use entrypoint with the
metadata-derived layer types and tensor counts. Qwen3.5 prompt processing is
currently deliberately sequential, one token per graph invocation, so the
first implementation favors a directly testable recurrent transition over a
parallel scan. To inspect the exact generated model source for the selected
GGUF, write it under
`target/catena/generated-qwen3.hex` with:

```bash
nix develop --command cargo run --release -- --generate-entrypoint
```

For a fixed-length performance run that continues through EOS:

```bash
nix develop --command cargo run --release -- \
  "The capital of France is" --max-new-tokens 1000 --ignore-eos
```

The current exploratory compiler fork produces the known continuation. The
registered-GGUF path reached 10.92 tok/s over 99 cached tokens, versus the
previous safetensors loader's 10.95 tok/s sample, and a 999-cached-token GGUF
validation run reached 9.83 tok/s. Three earlier
999-cached-token safetensors runs reached
9.90, 9.81, and 9.88 tok/s, a 9.88 tok/s median. That is 51.9% of the recorded
matched llama.cpp 19.03 tok/s baseline. A later llama.cpp sample reached 20.74
tok/s. The retained techniques are summarized in
[`PERFORMANCE_SUMMARY.md`](PERFORMANCE_SUMMARY.md); detailed measurements,
rejected experiments, and the EOS protocol caveat are documented in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## FLUX.2-klein text to image

The separate `flux2-klein` app loads the cached FLUX.2-klein-4B safetensors
checkpoint and writes one PNG. Missing model files are downloaded through the
shared Rust `hf-hub` loader. See
[setup, component review order and validation](docs/flux2-klein.md).
Use `--backend legacy` to run the separate implementation with existing compiler
primitives; see [comparison and benchmarks](docs/flux2-klein-legacy.md).

```bash
cargo run --offline --bin flux2-klein -- --prompt 'A fox in a snowy forest' --output fox.png
```

## Whisper ASR

The separate `whisper-asr` binary transcribes 16 kHz WAV clips up to 30 seconds
with `openai/whisper-small`, using the current local compiler without new
primitives. Model execution is composed in Hex, with persistent decoder caches.
The Rust `hf-hub` API downloads missing files to the standard Hugging Face cache
and reuses cached files on subsequent runs. Model resolution uses the Hub default
branch without a pinned revision. See
[setup, compiler alternatives and validation](docs/whisper-asr.md).

```bash
cargo run --release --bin whisper-asr -- recording.wav
```

## MMS English text to speech

The `mms-tts` app synthesizes a mono 16 kHz WAV with `facebook/mms-tts-eng`.
It shares unpinned Hub file resolution and safetensors loading with Whisper and
Flux, and composes inference and seeded sampling in one Catena/Hex entrypoint.
See [usage and validation](docs/mms-tts.md).

```bash
cargo run --release --bin mms-tts -- 'Hello, world.' --output speech.wav
```
