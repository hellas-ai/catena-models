# Experimental Qwen3.5 prefill experiments

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
