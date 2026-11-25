# Prototype Qwen3.5 prefill experiments

Model: `unsloth/Qwen3.6-35B-A3B-GGUF:BF16`, loaded through `hf_hub`.
Compiler: the existing Cargo.lock revision `af89a96`; HIP `gfx1151`, BF16,
no WMMA, matrix cores, BLAS, or compiler modifications. Runtime Rust is unchanged.

Each model benchmark loads once, warms up once, and measures three new sessions
per prompt length. Prompts use the same deterministic token sequence. Session
capacity is prompt length + 66. Decode measures 64 supplied tokens after two
warm-up tokens at a 32-token context. Compilation/loading are outside timing.

## Ablations

Prefill throughput, tokens/s; additions are cumulative unless stated otherwise.

| Variant | 32 tokens | 128 tokens | 256 tokens |
| --- | ---: | ---: | ---: |
| Starting point, register-retained single-token DeltaNet | 22.75 | 22.80 | 22.73 |
| Batched linear projections, convolution, gated norm, and recurrent scan | 23.35 | 23.76 | 23.95 |
| Also batch MoE routing and expert FFNs | 25.85 | ~25.72* | 25.70 |
| Also reuse projection weights across 4 prompt tokens | 34.09 | 32.43 | 31.85 |
| Use 8 prompt tokens per projection tile | 35.05 | 33.85 | 33.36 |
| Batch full-layer preparation/FFNs with online attention (rejected numerical change) | 44.11 | 41.11 | 40.39 |
| Also share K/V tiles across 8 queries (rejected) | 43.99 | 41.18 | 40.43 |
| Batch full-layer preparation/FFNs with existing staged BF16 attention (retained) | **41.81** | **39.57** | **38.75** |
| Default compiler, fused SIMT reference | 54.10 | 55.84 | 55.81 |

*The MoE-only 128-token row uses two clean runs; the third overlapped a separate
GPU test and is excluded. Its 256-token measurements were clean.

The shared-K/V variant is excluded from production: its whole-model improvement
was only 0.1–0.2%, despite improving the isolated attention kernel by 30–35% at
128–256 tokens. Per-query scalar maximum/sum folds beat the existing block
reductions in that experiment, but did not produce a material model-level gain.

For comparison, enabling the existing single-query online attention inside the
old token-major prefill loop alone gave only about 1%. The earlier layer-major
schedule using row copies and single-token kernels regressed by 2.5–5% and remains
reverted. The retained layer-major path requires actual batched kernels; these
measurements do not justify scheduling changes alone.

The retained result is **1.84× / 1.74× / 1.70×** the starting prefill throughput
at 32 / 128 / 256 tokens. Measured standard deviations were 0.85 / 0.05 / 0.10
tokens/s. Decode measured **20.24 ± 0.04 tokens/s**, compared with about 20.26
at the starting point. Its single-token graph and persistent scratch are retained.

## Implementation and numerical behavior

The new Hex kernels batch independent operations and retain recurrent state in
registers over the GPU token loop. MoE batch workspace excludes the recurrent
state prefix, avoiding a copy of that large prefix for every prompt token.
Eight-token projection tiles preserve the 256-lane dot accumulation tree,
BF16 activation/residual rounding, and masked tail stores.

The online-prefill experiments keep scores/normalization in F32. The independent
query variant differed from default fused attention by 12% in decoded BF16 cache
RMS on the 121-token prompt; the shared-K/V variant also changed a greedy token
on the short prompt. Neither variant is retained. An earlier diagnostic compared
BF16 bit patterns as integers; those invalid RMS numbers were discarded and the
checks rerun with float conversion.

The retained variant batches full-layer projection, normalization, KV preparation,
and FFNs while executing the existing staged attention operations per query in
Hex. It preserves their BF16 rounding and accumulation order. Its real-model
prefill cache matches the existing default staged path bit for bit; subsequent
20-token greedy decoding matches the default fused-decode reference. The opaque
session ABI and Rust text/image harness are unchanged.

## Validation

- `cargo check --lib --bins` passed.
- `cargo test --release --lib`: 93 passed, one ignored.
- CLI unit tests: 14 passed.
- GPU projection, convolution/history, DeltaNet scan, gated normalization, and
  MoE FFN tests compare with existing per-token kernels, including tail batches
  and nonzero cache positions. Projection and recurrent-state checks are exact.
- GPU hybrid session tests pass for dense and MoE: repeated independent sessions,
  chunked prefill, continued decoding, image rows, and MRoPE. Dense coverage uses
  a 96-wide hidden state to exercise non-power-of-two normalization.
- `unsloth/Qwen3.5-4B-GGUF:BF16` runs through the CLI and completes “Gravity is”
  coherently for 20 generated tokens.
- Real 35B parity passed on 2-token and 121-token prompts: exact BF16 cache
  equality, logit RMS within 5%, and 20 matching greedy tokens per prompt.
  Device argmax and independent sessions are checked too.

## RDNA3 Hex prefill kernels (2026-10-27)

The new compiler primitives expose wave32 16×16×16 BF16 inputs and F32 WMMA
accumulators. Hex owns routing, tile scheduling, allocations, epilogues, and
attention. Generated gfx1100/gfx1151 code contains
`v_wmma_f32_16x16x16_bf16`. Checked U64-to-U32 narrowing replaces the old
32-step loop-count conversion fold without changing its bounds check.

`--wmma-prefill` selects the experimental native path for dense and MoE
Qwen3.5/3.6 on HIP gfx11. It includes grouped routed/shared expert gate/up and down projections,
full-attention Q/K/V and output projections, linear output projections, and
attention with 64-query tiles. The default `all` selection also uses WMMA for the
four recurrent inputs (QKV, Z, alpha, and beta); diagnostic selections retain
their SIMT projections. Attention uses three launches per tile (QK, softmax, PV),
keeps the BF16 score/probability/context boundaries, and bounds temporary storage
by 64 query rows rather than the complete query sequence. This is not a
single-launch fused-attention kernel.

The flag-off default remains SIMT. The historical native WMMA measurements below
failed the agreed full-model accuracy gate: at most 5% relative logit RMS and
matching 20-token greedy continuations.

### Profiles and throughput

The measurements below predate enabling WMMA for the four recurrent input
projections; they do not measure the current default `all` selection.

On `unsloth/Qwen3.6-35B-A3B-GGUF:BF16`, gfx1151, the starting PP512 profile
spent 42.96% in expert down, 35.54% in expert gate/up, and 5.33% in output
projections. At PP4096 the corresponding shares were 26.86%, 21.70%, and 11.11%;
QK and context took a further 24.95% and 6.75%. This explains why accelerating
only seven input projections produced a small improvement.

| Variant | PP512 | PP4096 |
| --- | ---: | ---: |
| SIMT, current compiler | 39.25 ± 0.21 tok/s (three runs) | 25.51 tok/s (one run) |
| Initial input-projection WMMA, traced single run | 43.84 tok/s | 27.45 tok/s |
| Grouped experts, intermediate build, three runs | 132.52 ± 0.61 tok/s | — |
| Integrated native path, three runs | 131.92 ± 2.59 tok/s | 135.22 ± 0.28 tok/s |
| Exact tiled experts and attention, three runs | 51.27 ± 0.46 tok/s | 41.87 ± 0.17 tok/s |

Compilation and checkpoint loading are outside benchmark timing. Runs use the
CLI's deterministic synthetic tokens and new sessions. The traced starting
measurements include profiler overhead and should not be treated as a matched
unprofiled speed ratio. The integrated native PP512 measurement is 3.36× the
three-run SIMT baseline, but does not pass accuracy acceptance.
The exact tiled path passes accuracy acceptance and improves PP512 by 1.31×;
its PP4096 throughput is 1.64× the single-run baseline. It does not achieve the
requested 3–4× speedup. Native PP4096 is 5.30× that single-run baseline but fails
accuracy acceptance.

A bit-identical 32-token recurrent input projection tile was also tested. It
regressed integrated PP512 to 123.51 ± 2.14 tok/s and was removed.

### Accuracy ablations

`CATENA_WMMA_COMPONENTS` selects diagnostic variants when `--wmma-prefill` is
set. The generated source includes the choice in its module-cache key.

| Value | Accelerated operations |
| --- | --- |
| `all` (default for the flag) | Native grouped experts, tiled WMMA QK/PV, dense/output projections |
| `experts` | Native grouped experts only |
| `attention` | Tiled native WMMA QK/PV only |
| `exact-experts` | Original gate/up arithmetic with expert-major weight reuse; native WMMA down |
| `exact-attention` | Native WMMA QK; original eight-part F32 context accumulation |
| `safe-experts` | Original arithmetic with 16-route gate/up and 16-token down tiling |
| `safe-attention` | Packed QK loads and original QK/context arithmetic in bounded query tiles |
| `safe` | Both fully exact tiled variants |
| `none` | Original SIMT path, for control comparisons |

The original projection-only prototype measured 44.58 ± 0.07 tok/s at PP32
and 45.35 ± 0.69 at PP128, versus SIMT's 41.88 ± 0.04 and 40.68 ± 0.08.
Its 32-token logit error was already 5.70%. Extending native WMMA coverage
improved speed but increased error on long prompts:

| Variant | Synthetic32 RMS | Short text RMS | Text512 RMS | Text2048 RMS |
| --- | ---: | ---: | ---: | ---: |
| Native all | 7.99% | 6.07% | 45.99% | 39.71% |
| Native experts only | 6.97% | 1.64% | 52.38% | 41.26% |
| Native attention only | 2.83% | 1.61% | 38.53% | 36.55% |
| Exact gate/up, WMMA down | 4.85% | 2.52% | 44.52% | 30.56% |
| WMMA QK, exact context | 2.49% | 0% | 42.85% | 45.21% |
| Exact tiled experts only | 0% | 0% | 0% | 0% |
| Exact tiled experts and attention | 0% | 0% | 0% | 0% |
| SIMT control, independent model loads | 0% | 0% | 0% | 0% |

All native variants above changed at least one 20-token continuation. The
control and both fully exact variants matched every logit and generated token
exactly, including all 20-token continuations. The fully exact
expert kernels match the original GPU kernels bit for bit at the actual
256-expert, top-8, 2048×512 projection dimensions, including partial batches.
The fully exact attention variant passes the CPU comparison at 256-wide heads,
16-to-2 GQA, nonzero cache positions, and partial query/key tiles.
The combined exact variant also passed the 4096-token text prompt with zero
bitwise logit differences and all 20 continuation tokens matching. Its final
model test checks bitwise equality for exact component modes in addition to the
5%/token gate used for native modes.

Native MoE kernel tests compare gate activations and down projection separately
against independent CPU math at the existing 1% tolerance (with an absolute
floor of 0.01). The down reference consumes the measured BF16 activations, so
first-stage rounding is not confused with down-projection error. The exact
variant additionally requires bitwise equality with both original GPU kernels.
These kernel tolerances do not replace the full-model accuracy gate.

### Validation commands

Reproduce the native benchmark below. Prefix it with
`CATENA_WMMA_COMPONENTS=safe` for the exact tiled variant, or omit
`--wmma-prefill` for SIMT. Use `--pp 4096` for the longer prompt.

```bash
HF_HUB_OFFLINE=1 cargo run --release --offline -- \
  --model unsloth/Qwen3.6-35B-A3B-GGUF:BF16 --wmma-prefill \
  --bench --pp 512 --tg 0 --bench-runs 3
```

```bash
cargo test --all-targets
cargo test --release --test numeric_narrowing -- --ignored --nocapture
cargo test --release --test qwen35_wmma_moe -- --ignored --nocapture
cargo test --release --test qwen35_wmma_attention -- --ignored --nocapture
HF_HUB_OFFLINE=1 CATENA_WMMA_COMPONENTS=safe \
  cargo test --release --test qwen35_wmma_model wmma_prefill_matches_simt_model \
  -- --ignored --nocapture
```

`QWEN35_MODEL` overrides the cached BF16 MoE selector. `QWEN35_PARITY_LENGTHS`
controls additional text prompt lengths (default `512,2048,4096`); the synthetic32
and short-text checks always run. Native component modes currently fail the
model accuracy test; the test intentionally retains the 5%/token requirements.

Regenerate the explicit Hex sources in this order:

```bash
python3 tools/generate-qwen35-wmma-moe.py
python3 tools/generate-qwen35-exact-moe.py
python3 tools/generate-qwen35-exact-down.py
python3 tools/generate-qwen35-wmma-attention.py
python3 tools/generate-qwen35-exact-attention.py
python3 tools/generate-qwen35-wmma-epilogues.py
```


## Dense Qwen3.5 WMMA prefill

The same opt-in flag now supports dense Qwen3.5. Full-attention Q/K/V,
attention/output projections, and the gate/up/down FFN projections in both
full-attention and DeltaNet layers use the existing WMMA primitives. Dense
FFNs store gate and up projections as BF16, round SiLU to BF16, and multiply
with a BF16 result, preserving the existing intermediate precision boundaries.
The four recurrent DeltaNet inputs use WMMA with the default `all` selection.
Decode, diagnostic projection selectors, and the default flag-off graph are
unchanged.

`CATENA_WMMA_COMPONENTS=all` is the flag default. Attention-only selections
apply to dense models; `safe` keeps dense FFNs and other projections on SIMT
and selects safe tiled attention. Expert-only selections (`experts`,
`exact-experts`, `safe-experts`) require MoE and are rejected for dense models.

The adjacent compiler's emulator must stay out of line: HIP otherwise promotes
its private fragment arrays into a fixed 64 KiB LDS reservation even in native
mode, leaving no room for the projection kernel's dynamic LDS. The compiler
fix preserves emulator arithmetic and passed the existing native/emulated HIP
stress suite with zero mismatches across 17,892,864 output positions.

```bash
HF_HUB_OFFLINE=1 cargo run --release -- \
  --model unsloth/Qwen3.5-0.8B-GGUF:BF16 --wmma-prefill \
  --prompt 'The capital of France is' -n 20
HF_HUB_OFFLINE=1 cargo test --release --test qwen35_wmma_model \
  dense_wmma_prefill_diagnostics -- --ignored --nocapture
```

The diagnostic model check uses one short prompt and one 512-token prompt,
then 20 decode tokens each. It reports numerical/continuation differences
rather than enforcing exact native WMMA parity. Focused projection and dense
FFN GPU checks cover partial tiles and BF16 rounding boundaries.

### Dense 0.8B validation on gfx1151

One run on the cached `unsloth/Qwen3.5-0.8B-GGUF:BF16`, with native WMMA
and the adjacent compiler's out-of-line emulator fix:

| Prompt | SIMT prefill | WMMA prefill | Relative logit RMS | 20-token greedy continuation |
| --- | ---: | ---: | ---: | --- |
| Short text, 5 tokens | 0.049841 s | 0.048389 s | 0.8269% | identical |
| Text, 512 tokens | 2.005946 s (255.24 tok/s) | 0.961360 s (532.58 tok/s) | 0.9345% | identical |

PP512 improved **2.09×** in this single comparison. Prefill timings exclude
model loading, compilation, and session construction; they include logit
readback. The complete diagnostic test took 358.86 s including cold compilation
and both models' decode checks. These two matching continuations do not
establish parity for other prompts or dense model sizes. WMMA remains opt-in.

All library tests passed (30 passed, one ignored), as did all-target compilation,
source-generation checks, focused GPU projection/FFN checks, and the compiler's
native/emulated HIP stress check. No model inference was moved into Rust.

## Qwen3 FP8 prefill on NVIDIA CUDA

`--prefill-dtype fp8` selects native E4M3 projections for dense Qwen3 FP8
safetensors. `--prefill-dtype bf16` remains the default. The initial FP8 path
covers Q/K/V, attention output, and gate/up/down projections. It fuses residual
addition and gate/up SiLU multiplication while retaining the existing BF16
rounding boundaries. Attention, KV storage, normalization, vocabulary output,
and single-token execution use their existing kernels.

The checkpoint used here is `unsloth/Qwen3-0.6B-FP8`, revision
`877ed919128bc5b13ab7f30878c6cf1ae78f4531`. Its 196 projection matrices are E4M3
with 128×128 weight blocks and BF16 `weight_scale_inv` tensors. The loader also
accepts F32 scales and explicit `quant_type: "float"`; this revision omits that
field. Despite the scale tensor's name, reconstruction multiplies by it:
`W[n,k] = decode_e4m3(bits[n,k]) * weight_scale_inv[n/128,k/128]`.

Activations are dynamically quantized per token and 128-element K group using
`amax / 448`, with scale 1 for an all-zero group. Packed conversion uses CUDA's
round-to-nearest and finite saturation. GEMM accumulates each K group in FP32
with `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, then applies that
group's activation and weight scales before adding to the final sum. Applying
one scale after the entire K reduction would be incorrect.

Kernels use four warps and 16×64×128 or 64×64×128 tiles, with shared-memory rows
padded to avoid bank conflicts. Row tails are masked. Projection widths must be
multiples of 128. Unquantized matrices in mixed checkpoints use BF16 kernels.
Both BF16 decode weights and native FP8 prefill weights are retained: measured
weight storage is 1,192,230,912 bytes for BF16 and 1,632,740,352 bytes for FP8 mode.

FP8 opt-in assumes a suitable CUDA toolkit and SM89+ hardware. FP8 modules
compile for SM89 with forward-compatible PTX. There is no hardware capability
probe and no hardware version in compiler cache keys. The existing development
cache policy applies; clear caches manually when changing hardware or toolkit.

Reproduce generation or compare the CLI benchmark modes:

```bash
CATENA_GPU_DIALECT=cuda cargo run --release -- \
  --model unsloth/Qwen3-0.6B-FP8 --prefill-dtype fp8 \
  --prompt 'The capital of France is' -n 8
CATENA_GPU_DIALECT=cuda cargo run --release -- \
  --model unsloth/Qwen3-0.6B-FP8 --prefill-dtype bf16 \
  --bench --pp 512 --tg 0 --bench-runs 9
CATENA_GPU_DIALECT=cuda cargo run --release -- \
  --model unsloth/Qwen3-0.6B-FP8 --prefill-dtype fp8 \
  --bench --pp 512 --tg 0 --bench-runs 9
```

The dedicated measurements warm each shape twice and report the median of nine
runs. Model loading, compilation, and session allocation are outside timing.
Projection measurements include activation quantization and scratch allocation;
input/weight uploads are outside timing. Run performance tests without competing
GPU workloads:

```bash
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8 \
  fp8_prefill_benchmark -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8 \
  fp8_projection_benchmark -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8 \
  fp8_kernels_match_independent_blockwise_reference -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8 \
  unsloth_fp8_prefill_matches_bf16_and_continues_decode -- --ignored --nocapture
```

Regenerate the Hex kernels with `python3 tools/generate-qwen3-fp8.py`.
The compiler primitives and CUDA target selection are implemented in the sibling
`catena-lang` checkout selected by `.cargo/config.toml`.

### Initial projection-only measurements, 2026-11-25

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU (SM120, 8 GiB), CUDA toolkit 13.2,
driver 595.71.05. Release builds, isolated GPU execution, two warmups and nine
measured runs per shape. End-to-end timings use the real checkpoint weights and
deterministic synthetic tokens. The initial implementation with fused projection epilogues and BF16 attention measured:

| Prompt tokens | BF16 median | FP8 median | BF16 tokens/s | FP8 tokens/s | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 25.025 ms | 9.071 ms | 679.31 | 1,874.17 | 2.76× |
| 128 | 91.508 ms | 24.860 ms | 1,398.78 | 5,148.81 | 3.68× |
| 512 | 378.096 ms | 103.384 ms | 1,354.15 | 4,952.41 | 3.66× |
| 1024 | 759.008 ms | 241.799 ms | 1,349.13 | 4,234.92 | 3.14× |

Projection microbenchmarks use representable unit weights and constant BF16
inputs. Timings include a separate activation quantization for each FP8 GEMM;
the model shares quantized inputs across Q/K/V and computes gate/up together.
Selected projection medians:

| M | N | K | BF16 | FP8 including quantization |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2048 | 1024 | 1.529 ms | 0.254 ms |
| 512 | 3072 | 1024 | 2.143 ms | 0.314 ms |
| 512 | 1024 | 3072 | 2.273 ms | 0.399 ms |
| 1024 | 2048 | 1024 | 2.868 ms | 0.439 ms |
| 1024 | 3072 | 1024 | 4.121 ms | 0.553 ms |
| 1024 | 1024 | 3072 | 4.186 ms | 0.579 ms |

Independent CPU comparisons passed for both tile sizes, fused residual and
SiLU/gate/up epilogues, multiple nonuniform scale blocks, zero activation groups,
row tails, FP8 subnormals, and maximum finite E4M3 values. Full-model BF16/FP8
logit cosine similarity was 0.994013–0.999288 across multi-token synthetic inputs
of lengths 2, 17, 128, 512, 1024 and four text prompts. All logits were finite.
One-token prefill and three subsequent teacher-forced decode steps matched BF16
exactly. A 17+111-token chunked prefill passed the final-logit comparison against
an uninterrupted 128-token prefill, and decode continued successfully from FP8
prefilled caches.

First graph compilation can take several minutes; subsequent runs use the
existing compiler and native caches. Performance comparisons exclude that cost.

## Fused FP8 attention and warp preparation

These Qwen3 measurements predate mandatory FP8 attention. The current FP8 option
also supports dense Qwen3.5 and uses FP8 attention for every batched prefill; the
old attention toggle and short-batch BF16 fallback have been removed.

The measured version used fused causal attention for batches of at least 64 tokens and
head dimensions 32, 64, 128, or 256. The historical comparison mode retained the previous
BF16 attention and preparation path while keeping FP8 projections, allowing an
isolated comparison against the initial implementation above.

Both QK and probability–value products use native E4M3 MMA with FP32
accumulation. Online maximum, denominator, exponential, and context rescaling
stay FP32. Softmax uses CUDA’s fast FP32 exponential; normalization and RoPE
retain their existing math. The kernel processes 64 queries and 64 keys per tile without writing
a score matrix or probability matrix to global memory. Query and key snapshots
use one dynamic scale per head vector; value snapshots use one scale per KV head
and 64-key tile. Unnormalized probabilities lie in [0,1] and use FP8 scale 1/448.
Transient snapshots are released after attention. The persistent KV cache stays
BF16, including cached prefixes and values consumed by subsequent decode.

A warp preparation kernel handles full-head Q/K RMSNorm, RoPE, and BF16 KV append.
Warp reductions replace shared-memory reduction stages, and RoPE partners stay
in registers. Partial rotary heads retain the existing preparation kernel.
Single-token execution and decode are unchanged. No GPU capability probing or
hardware-specific cache keys were introduced.

Before this change, a CUDA trace at 1024 tokens attributed approximately 53% of
GPU kernel time to the separate QK/softmax/PV attention passes and 16% to Q/K
normalization/RoPE preparation. After implementation, independent FP32 causal
attention comparisons passed for head sizes 32/64/128/256, grouped query heads,
nonzero prefix positions, row/key tails, and sharper scores. Relative RMS errors
were below 2.6% and cosine similarity exceeded 0.9996 across 24 cases including zero Q/K heads.
Preparation checks cover absolute RoPE, unchanged cache prefixes, and exact V
copies; neither transformation changes persistent KV storage format.

Additional component checks and timing:

```bash
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8_attention \
  fp8_attention_matches_independent_causal_reference -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8_attention \
  warp_prepare_preserves_bf16_kv_and_absolute_rope -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8_attention \
  fp8_attention_benchmark -- --ignored --nocapture
```

Regenerate these kernels with `python3 tools/generate-qwen3-fp8-attention.py`.
The generator also regenerates its shared dense FP8 helpers. CUDA warp shuffle
support lives in the sibling `catena-lang` compiler.

### Final attention/preparation measurements, 2026-11-25

Same RTX 5050 Laptop GPU and checkpoint revision as above. Each path is loaded
once, warmed twice per shape, and measured nine times; medians exclude loading,
compilation, and session allocation. `FP8 projections` uses
the former projection-only comparison mode; `Full FP8` uses `--prefill-dtype fp8`.

| Tokens | BF16 median | FP8 projections | Full FP8 | Gain over FP8 projections | Gain over BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 25.080 ms | 9.014 ms | 8.535 ms | 1.06× | 2.94× |
| 128 | 92.232 ms | 26.354 ms | 20.040 ms | 1.32× | 4.60× |
| 512 | 379.470 ms | 102.733 ms | 57.007 ms | 1.80× | 6.66× |
| 1024 | 760.396 ms | 244.734 ms | 119.711 ms | 2.04× | 6.35× |

Full FP8 throughput was 8,981 tokens/s at 512 tokens and 8,554 tokens/s at 1024.
Weight storage remains 1,632,740,352 bytes; attention snapshots are transient.
The 17-token gain comes from warp preparation, since attention below 64 tokens
retains the BF16 kernels.

An isolated attention comparison uses the real model's head geometry (16 query
heads, 8 KV heads, head dimension 128), including FP8 snapshot preparation,
allocation, and cleanup:

| Query tokens | BF16 attention | Fused FP8 attention |
| ---: | ---: | ---: |
| 64 | 0.091 ms | 0.062 ms |
| 128 | 0.238 ms | 0.150 ms |
| 512 | 1.596 ms | 0.547 ms |
| 1024 | 4.430 ms | 1.075 ms |

The final fast-exponential variant passed all 24 independent attention cases
with the same reported error bounds. Full-model logit cosine similarity against
BF16 was 0.994013–0.999288 across the synthetic lengths and four text prompts.
The 128/512/1024-token cosines were 0.996423, 0.995099, and 0.998219. All logits
were finite. Chunked 17+111-token prefill and subsequent decode passed; a
single-token prefill and three subsequent decode steps still matched BF16 exactly.

A follow-up CUDA trace confirms the targeted kernel changes. These are GPU
kernel-only averages across 84 layer calls (28 layers × 3 PP1024 runs), excluding
host allocation and API overhead:

| Component | Previous FP8-projection path | New path | Kernel speedup |
| --- | ---: | ---: | ---: |
| Attention, including new Q/K/V quantization kernels | 3.757 ms | 0.705 ms | 5.33× |
| Q/K normalization, RoPE, and KV append | 1.117 ms | 0.082 ms | 13.68× |

Attention kernel launches per layer fell from 48 to 4 at 1024 tokens. The final
CLI generation check completed successfully; a warm runtime/weight load took
about 0.8 seconds in that run. Profile artifacts were captured under
`/tmp/qwen-fp8-before.nsys-rep` and `/tmp/qwen-fp8-after.nsys-rep`.

## Dense Qwen3.5 FP8 prefill

`--prefill-dtype fp8` supports dynamic blockwise FP8 dense Qwen3.5 text
checkpoints. Validation uses `surogate/Qwen3.5-0.8B-FP8` at revision
`1423a80c469215a6045bd4cdb9f342fcc293d0ea`, whose text matrices
use E4M3 bytes and F32 `weight_scale_inv` tensors for 128×128 blocks. The loader
accepts quantization metadata at the model root or inside `text_config`, and
handles both plain and `model.language_model` tensor names.

Full-attention layers use native FP8 Q/G, K, V, output, and MLP projections.
Fused FP8 attention applies the per-head sigmoid output gate after the same
BF16 context rounding as the existing attention kernel. Partial RoPE and the
canonical BF16 KV layout are preserved. Linear-attention layers use FP8 QKV,
Z, output, and MLP projections, sharing one activation quantization across
QKV/Z. Alpha/beta projections, convolution, recurrent updates, and gated
normalization retain their existing arithmetic. Native value-head permutations
reorder the FP8 bytes and their scale blocks together without requantization;
the BF16 decode copies use the corresponding element permutation.

Selecting FP8 also selects FP8 full attention for every batched prefill, including
short batches. The separate attention option and the former 64-token cutoff
have been removed for both Qwen3 and Qwen3.5. Single-token execution continues
to use the existing decode path. Unsupported attention head sizes are rejected.
The supported sizes are 32, 64, 128, and 256, and Qwen3.5 currently requires
128-wide linear key heads. There are no hardware probes or changes to compiler
cache key policy.

Independent attention references cover 24 ungated and 24 gated cases: all four
head sizes, GQA, cached prefixes, causal masks, partial tiles, zero Q/K, sharper
scores, and NaN poison beyond valid cache positions. Gated cosine similarity
is at least 0.999679; ungated similarity is at least 0.999681. Native head
permutation tests verify identical bytes and dequantized BF16 values, including
nonuniform scale blocks, a QKV prefix, and column permutations.

```bash
CATENA_GPU_DIALECT=cuda cargo run --release -- \
  --model surogate/Qwen3.5-0.8B-FP8 --prefill-dtype fp8 \
  --prompt 'The capital of France is' -n 16
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen35_fp8 \
  dense_fp8_prefill_matches_bf16_and_continues_decode -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen35_fp8 \
  dense_fp8_prefill_benchmark -- --ignored --nocapture
CATENA_GPU_DIALECT=cuda cargo test --release --test qwen3_fp8_attention \
  fp8_gated_attention_matches_independent_causal_reference -- --ignored --nocapture
```

The end-to-end benchmark uses two warmups and seven measurements per shape,
reports medians, and excludes checkpoint loading, compilation, and session
allocation. `QWEN35_FP8_MODEL` selects another compatible checkpoint for these
tests. `QWEN35_FP8_LAYERS=4` runs the parity check on a truncated model for faster
debugging; final full-model validation leaves this unset.

Full 24-layer checkpoint parity passes at 1, 2, 17, 65, 128, and 512 tokens,
plus a natural-language prompt. Logit cosine similarities against the BF16
path are respectively 1.00000000, 0.99730419, 0.99937044, 0.99931703,
0.99926902, and 0.99886207; the five-token prompt gives 0.99951654. A 17+111
chunked prefill versus a single 128-token prefill gives 0.99839956. All cases
continue through three forced decode tokens with finite logits; single-token
prefill and its forced decode continuation match BF16 exactly.

The cold full-model FP8 Hex compile took 598.8 seconds on this development
machine. Subsequent runs use the existing module and native caches. Compilation
and loading are excluded from the performance measurements below.

RTX 5050 Laptop GPU, release build, two warmups and seven measurements per
shape, with medians excluding loading, compilation, and session allocation:

| Tokens | BF16 median | FP8 median | FP8 throughput | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 29.627 ms | 12.427 ms | 1,367.95 tok/s | 2.38× |
| 128 | 110.747 ms | 29.234 ms | 4,378.39 tok/s | 3.79× |
| 512 | 397.101 ms | 95.666 ms | 5,351.96 tok/s | 4.15× |
| 1024 | 769.944 ms | 183.717 ms | 5,573.80 tok/s | 4.19× |

Text weight storage is 1,505,783,040 bytes in BF16 mode and 2,002,929,408 bytes
in FP8 mode, including both BF16 decode operands and native FP8 prefill operands.

### Preserving Qwen3 short-batch accuracy with mandatory FP8 attention

Removing the old BF16 fallback exposed a Qwen3 accuracy regression at short
sequence lengths: single-pass FP8 attention gave logit cosine 0.98360806 at
17 tokens. Qwen3 batches below 64 tokens therefore use residual E4M3 operands
for Q, K, V, and probabilities. Power-of-two Q/K/V scales avoid introducing
extra rounding before splitting the BF16 operands. For each normalized operand,
the second packet stores sixteen times the residual after the first E4M3
rounding. QK and PV use four FP8 MMA products, rescaling the primary/residual
cross products by 1/16 and the residual/residual product by 1/256.

The short-batch path preserves the existing BF16 score boundary. For a single
64-key tile it also preserves the BF16 normalized-probability boundary, using
power-of-two probability scaling and already-normalized probabilities in PV.
All matrix products remain FP8; this introduces no BF16 attention path or
independent attention option. Longer Qwen3 batches and all Qwen3.5 batches keep
the single-pass FP8 attention kernel measured above.

Temporary Q/K/V packets and the shared probability tile have a second packet
plane; persistent KV storage is unchanged.

The residual kernel passes 24 independent reference cases across head sizes
32, 64, 128, and 256, including the baseline score/probability boundaries.
Maximum normalized RMS error is 0.001714; minimum reported cosine is
0.999999.

The final Qwen3 checkpoint regression passes with mandatory FP8 attention.
At 2 and 17 tokens, logit cosine versus BF16 is 0.99401264 and 0.99851949.
At 128, 512, and 1024 tokens it remains 0.99642262, 0.99509936, and 0.99821934.
Natural-language prompts, chunked prefill, and decode continuation also pass;
single-token prefill and forced decode remain identical to BF16.
