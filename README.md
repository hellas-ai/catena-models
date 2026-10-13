# Catena Qwen inference

Native Catena GPU inference for Qwen3 dense, Qwen3 MoE, and Qwen3.5/3.6 dense
and MoE text models, with a Qwen3.5 image path. Model dimensions and tensor
layouts come from GGUF metadata. BF16/F32 weights retain their native storage.

All model selection uses Hugging Face names on the command line. `hf_hub`
resolves repositories, caches files, and downloads every selected GGUF shard.
`HF_TOKEN`, `HF_HOME`, and `HF_HUB_CACHE` retain their normal meanings. Cache
paths and snapshot revisions are managed by the Hub client. The removed
`--model-dir`, `--gguf`, and `--tokenizer` options have no replacement local-path
mode. Qwen tokenizers come from the selected GGUF metadata.

```bash
cargo run -r -- --model unsloth/Qwen3.5-2B-GGUF:BF16 --prompt 'Gravity is'
cargo run -r -- --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 --prompt 'Gravity is'
```

The `:variant` suffix selects an unambiguous checkpoint, including split BF16
checkpoints. Companion vision projectors are resolved from the same repository,
or from an explicit HF name using `--mmproj-model`:

```bash
cargo run -r -- --model unsloth/Qwen3.5-0.8B-GGUF:BF16 \
  --image photo.jpg --prompt 'Describe this image.'
```

[Image handling and precision](docs/qwen35-images.md).

## Compiler and kernel selection

The default compiler enables BF16 matrix cores and fused attention. Use
`--matrix-cores=false` to select SIMT projections. Fused cached attention has a
SIMT implementation and continues working without WMMA. The older Qwen3 MoE
architecture uses the F32 activation graph (`--bf16=false`, with matrix cores
and fused attention disabled).

```bash
cargo run -r -- --codegen default --matrix-cores=false \
  --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 --prompt 'Gravity is'
cargo run -r -- --codegen experimental \
  --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 --prompt 'Gravity is'
```

The experimental compiler supports BF16 Qwen3 dense and Qwen3.5/3.6 dense or
MoE text inference on HIP gfx11. It uses Hex kernels under
`src/catena/experimental`, with no new compiler primitives. Its CLI automatically
disables legacy matrix-core, BLAS, and attention switches. `--kernels fused`
is the default; `--kernels reference` is available for Qwen3 dense only.
Images and older Qwen3 MoE use the default compiler.

Experimental Qwen3.5 MoE combines routed/shared expert gate/up activations and
down projections, keeps routing scratch per session, and uses parallel argmax.
Its full-attention layers use SIMT online cached attention during decode,
including context output gating, without score/probability allocations.
Multi-token prefill retains the validated staged BF16 attention path.
Cooperative per-head L2 and gated RMS normalization preserve the existing
scalar sum order. DeltaNet
prediction, recurrent update, and output execute in one kernel without a delta
allocation. Prefill keeps recurrent state F32 until the call boundary, then
persists BF16 cache state. Prefill currently processes tokens sequentially.

[Kernel details, validation, and measurements](docs/experimental-qwen.md).

## Benchmarks and validation

```bash
cargo run -r -- --codegen experimental \
  --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 \
  --bench --pp 0 --tg 128 --tg-context 32 --bench-runs 5
cargo run -r -- --matrix-cores=false --fused-attention=true \
  --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 \
  --bench --pp 0 --tg 128 --tg-context 32 --bench-runs 5
cargo test -r --lib --bins
CATENA_QWEN3_EXPERIMENTAL_GPU_TESTS=1 cargo test -r --lib \
  experimental_qwen35 -- --nocapture --test-threads=1
CATENA_QWEN36_EXPERIMENTAL_MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:BF16 \
  cargo test -r --lib experimental_qwen35_moe_checkpoint_parity_on_gpu -- --nocapture
```

Repeat warmed decode with `--tg-context 256`. Measurements exclude model
loading, compilation, prefill, and two warmup decode calls. Each checkpoint
comparison loads compiler paths sequentially to avoid pinning two copies of a
large checkpoint. Optional real-checkpoint tests accept HF selectors through
`CATENA_QWEN3_EXPERIMENTAL_MODEL`, `CATENA_QWEN35_EXPERIMENTAL_MODEL`, and
`CATENA_QWEN36_EXPERIMENTAL_MODEL`.

`--generate-entrypoint` exports the selected model's Hex under `target/catena`.
`CATENA_QWEN3_EXPERIMENTAL_CPP` optionally exports experimental generated C++.
Library callers can use `Model::load_hf_with_options(model_name, options)`;
Session APIs and internal cache layouts are shared by both compilers.

## Other model applications

Their model selection also uses HF names, with compatible architecture/config
validation before execution:

```bash
cargo run -r --bin whisper-asr -- --model openai/whisper-small speech.wav
cargo run -r --bin mms-tts -- --model facebook/mms-tts-eng 'Hello, world.'
cargo run -r --bin flux2-klein -- --model black-forest-labs/FLUX.2-klein-4B \
  --prompt 'A red fox sitting in a snowy forest' --output image.png
```

[Whisper](docs/whisper-asr.md), [MMS TTS](docs/mms-tts.md),
[Flux2 Klein](docs/flux2-klein.md).

The Transformers comparison runner takes `--model owner/repo`.
`setup-llama-cpp.sh owner/repo output.gguf` resolves its source on the Hub before
converting it. `evict-gguf-cache.sh owner/repo:variant` operates on resolved Hub
cache files rather than a fixed model directory.

`run-llama-cpp.sh owner/repository:variant 'Gravity is'` delegates HF model
resolution to llama.cpp. `profile-qwen3.sh owner/repository:variant` resolves
its checkpoint through the same Hub loader as the Qwen runner.
