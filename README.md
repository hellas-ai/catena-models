# Qwen3-30B-A3B-Base reference runners

Local BF16 inference for the base (non-instruct) `Qwen3-30B-A3B-Base`
checkpoint through Transformers and llama.cpp. There is deliberately no Catena
implementation yet.

The scripts assume the Nix development environment; invoke them with
`nix develop --command` as shown below.

## Setup

Download the original BF16 Hugging Face checkpoint (about 61 GB):

```bash
mkdir -p models/Qwen3-30B-A3B-Base
nix develop --command hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir models/Qwen3-30B-A3B-Base
```

Convert it to an unquantized BF16 GGUF for llama.cpp (about another 61 GB):

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
with a 96 GiB AMDGPU GTT aperture so the 61 GB model and compute buffers fit in
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
