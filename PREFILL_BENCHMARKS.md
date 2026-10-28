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

`--wmma-prefill` selects the experimental native path for Qwen3.5/3.6 MoE on
HIP gfx11. It includes grouped routed/shared expert gate/up and down projections,
full-attention Q/K/V and output projections, linear output projections, and
attention with 64-query tiles. The four recurrent input projections retain the
original SIMT tree. Attention uses three launches per tile (QK, softmax, PV),
keeps the BF16 score/probability/context boundaries, and bounds temporary storage
by 64 query rows rather than the complete query sequence. This is not a
single-launch fused-attention kernel.

The default remains SIMT because native WMMA currently fails the agreed
full-model accuracy gate: at most 5% relative logit RMS and matching 20-token
greedy continuations.

### Profiles and throughput

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
