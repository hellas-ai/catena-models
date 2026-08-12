# Qwen3-30B-A3B-Base reference runners

Local BF16 inference for the base (non-instruct) `Qwen3-30B-A3B-Base`
checkpoint through Transformers, llama.cpp, and an exploratory Catena/HIP
implementation. The Catena path executes all 48 layers, uses a persistent KV
cache, and performs greedy generation directly from a HIP-registered BF16
GGUF mapping.

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
mapping with HIP, and passes BF16 tensor subranges directly to the generated
kernels. It neither repacks the 128 experts nor copies the 56.85 GiB BF16
payload into separate device allocations. llama.cpp's converter stores 48.80
MiB of small BF16-derived tensors as F32; the loader verifies those values are
exact BF16 expansions and restores only that subset into 24.40 MiB of device
memory. Measured loads reached weight readiness in 2.72--2.84 seconds.

Use a GGUF outside the model directory with `--gguf /path/to/model.gguf`.
`--model-dir` still selects the directory containing `tokenizer.json`.

RoPE is also part of the Catena program rather than a model-specific Rust
input. Each invocation materializes only its current absolute-position range,
using exact F32 inverse-frequency constants and deterministic range-reduced
sine/cosine polynomials, then reuses that owned table across all 48 layers.
The generated table differs from the previous host `sinf`/`cosf` values by at
most `4.92e-7` across the complete 32,768-position context.

Normal inference compiles the complete checked-in `src/catena/qwen3.hex` and
does not generate its 48-layer entrypoint in Rust. If the layer/weight ABI is
changed, explicitly regenerate that entrypoint once and review the resulting
source diff:

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
