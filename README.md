# Qwen3-30B-A3B-Base reference runners

Local BF16 inference for the base (non-instruct) `Qwen3-30B-A3B-Base`
checkpoint through Transformers, llama.cpp, and an exploratory Catena/HIP
implementation. The Catena path executes all 48 layers, uses a persistent KV
cache, and performs greedy generation directly from a HIP-registered BF16
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

The pinned Catena code generator currently limits ordinary and paired F32
reductions to 4,096 terms and borrowed F32 reductions to 8,192 terms because
their complete term arrays live in per-block shared memory. Consequently,
Qwen3-4B and 8B exceed the borrowed down-projection limit with intermediate
widths 9,728 and 12,288, while 14B and 32B also exceed the ordinary projection
limit with hidden width 5,120. The runner rejects these shapes before compiling
the GPU program. Attention history is likewise limited to 8,192 columns by the
current softmax primitive.

The scripts assume the Nix development environment; invoke them with
`nix develop --command` as shown below.

## Setup

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

Catena validates and read-only-maps `model-bf16.gguf`, registers that single
mapping with HIP, and passes every tensor subrange directly to the generated
kernels in its native GGUF representation. The 338 large tensors remain BF16;
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
