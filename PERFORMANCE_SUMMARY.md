# Language features added to achieve performance

## Experimental Catena dependency delta

This repository uses the local `../exploratory-catena/catena-lang` path
dependency rather than the compiler at `origin/master`. Its current `master`
is exactly five commits ahead of `origin/master`:

| Commit | Compiler addition | Use in this repository |
| --- | --- | --- |
| `187d12e` | Raises the exact-reduction and softmax limits | Accommodates Qwen3.5/Qwen3.6 dimensions and longer attention rows without changing the specified reduction tree |
| `135a19e` | Adds `materializec.borrow-gated-delta-net-f32` | Implements the recurrent gated DeltaNet layers in the F32 Qwen3.5 dense and Qwen3.6 MoE graphs |
| `1a1229e` | Adds native-BF16 GEMV/argmax support and parameterizes reductions and softmax over F32/BF16 storage | Keeps dense Qwen activations, projections, attention probabilities, and logits in BF16 while accumulating sensitive arithmetic in F32 |
| `90e3372` | Adds `materializec.borrow-gated-delta-net-bf16` | Runs the Qwen3.5/Qwen3.6 recurrent scan with BF16 persistent state and inputs but F32 register arithmetic |
| `e9e0659` | Adds tiled cached-context kernels and native-BF16 routed-expert kernels | Speeds exact cached attention and removes F32 activation detours from Qwen3.6 sparse MoE gate/up/down projections |

The additions are compiler operations, not opaque model-side foreign kernels:
their Catena types preserve ownership of persistent buffers, validation checks
the tensor capacities and layout parameters, and the GPU backend emits HIP or
CUDA source for the selected storage type.

## `materializec.bf16-gemv`

Computes row-major BF16 matrix products with F32 accumulation and one final
BF16 rounding. On AMD gfx11 it loads packed BF16 pairs and emits
`v_dot2_f32_bf16`, accumulates per thread in registers, and reduces only 256
partial F32 sums per output instead of materializing every product in shared
memory. A scalar BF16/F32 fallback keeps other GPU targets compilable.

Dense Qwen3 uses this primitive for Q/K/V, gate/up, attention-output, MLP-down,
and vocabulary projections. On the local gfx1151 system, matched checkpoints
measured:

| Model | Test | F32 graph | Native BF16 | Change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | pp512 | 94.70 tok/s | 154.40 tok/s | +63.0% |
| Qwen3-0.6B | tg128 | 51.89 tok/s | 81.81 tok/s | +57.7% |
| Qwen3-4B | pp512 | 11.81 tok/s | 24.01 tok/s | +103.3% |
| Qwen3-4B | tg128 | 9.91 tok/s | 20.30 tok/s | +104.8% |

## Experimental BF16 accelerated prefill

`--bf16 --matrix-cores` selects `materializec.bf16-gemm-wmma` for ordinary
dense projections in Qwen3 dense, Qwen3.5 dense, and Qwen3.5/3.6 MoE graphs.
`--bf16 --blas` selects the matching `materializec.bf16-gemm-blas`
backend. The typed API uses
`InferenceOptions::new(InferencePrecision::Bf16).with_matrix_cores(...)` with
`MatrixCoreMode::Wmma` or `MatrixCoreMode::Blas`. F32 and the older Qwen3 MoE
graph are rejected.

On HIP gfx11, WMMA projections with at least 64 input rows use 16×16×16
rocWMMA BF16 matrix operations with F32 accumulators; CUDA uses
`nvcuda::wmma`. The BLAS mode uses `hipblasGemmEx` on HIP and `cublasGemmEx`
on CUDA, with BF16 input/output and F32 accumulation. Smaller
inputs use the packed-dot SIMT kernel in either mode, so cached single-token
decode is unchanged. Both modes are opt-in because accelerated GEMM does not
retain the deterministic SIMT reduction order.

For Qwen3.5/3.6 MoE, the large dense shared-expert gate/up pair uses matching
paired WMMA or BLAS projections. Dynamically routed expert gate/up/down
operations remain on their native-BF16 SIMT kernels, and the narrow F32 router
and shared-router projections remain unchanged.

Matched three-run measurements on the local gfx1151 system and the cached
Qwen3-0.6B BF16 checkpoint were:

| Test | Default native BF16 | BF16 matrix cores | Change |
| --- | ---: | ---: | ---: |
| pp64 | 311.35 tok/s | 425.57 tok/s | +36.7% |
| pp512 | 223.57 tok/s | 381.94 tok/s | +70.8% |
| tg16 | 76.73 tok/s | 77.16 tok/s | +0.6% |

A 112-token prompt produced the same next token in both modes while prompt
processing improved from 312.03 to 524.81 tok/s. Compiler integration tests
execute single and paired 64×16 products through the SIMT, WMMA, and BLAS
primitives. A standalone gfx1151 code-object check also confirms emission of
`v_wmma_f32_16x16x16_bf16`.

## BF16-capable reductions and softmax

`materializec.reduce-f32` and `materializec.borrow-reduce-f32` now retain their
specified F32 term and adjacent-pair accumulation trees while allowing the
epilogue and output buffer to be either F32 or BF16. The BF16 path therefore
rounds only the final value rather than every partial sum. Likewise,
`materializec.softmax-f32` accepts F32 or BF16 storage, widens BF16 values for
the maximum, exponential, and denominator calculations, and writes the
normalized row back in its original storage type.

The same compiler delta raises the supported reduction length to 16,384 for
`reduce-f32` and `borrow-reduce-f32`, the paired reduction limit to 8192, and
the softmax row width to 16,384. These are capacity changes only: the logical
tree and deterministic result are unchanged.

## `Runtime::mem_bf16_zeroed`

The runtime now allocates and zero-initializes an application-owned device
buffer using two bytes per element. `Model::session` selects it for BF16
inference when allocating the combined persistent attention and recurrent
cache, rather than allocating an F32 cache or staging a host-side BF16 zero
buffer.

## `materializec.borrow-argmax-bf16`

This is the native-storage counterpart of `borrow-argmax-f32`. It compares
BF16 values directly using their total-order keys, returns the logits owner,
and materializes only the selected U64 index. The BF16 entrypoint uses it so a
151,936-element vocabulary projection neither widens to F32 nor transfers to
the host merely to select the next token.

## Fused gated DeltaNet scans

`materializec.borrow-gated-delta-net-f32` runs the complete token scan for one
layer in a single GPU kernel. It consumes and returns the persistent recurrent
state together with Q, K, V, gate, and beta owners; the output is laid out like
V. Each workgroup keeps a state row in registers across the token loop, uses
warp/wave shuffles for prediction and output dot products, updates the state
in place at a validated layer offset, and avoids materializing every recurrent
step as a separate Catena operation.

`materializec.borrow-gated-delta-net-bf16` shares that implementation through
a storage-type parameter. BF16 inputs and recurrent state are widened on
load, the state recurrence, exponentials, dot products, and scaling remain
F32 in registers for the full scan, and only output and final-state stores are
rounded to BF16. Qwen3.5 dense and Qwen3.6 MoE select the F32 or BF16 primitive
according to the inference graph.

## Tiled cached-attention context primitives

`materializec.borrow-cached-context-f32` and
`materializec.borrow-cached-context-bf16` implement the probability-times-V
half of grouped-query cached attention. They understand the persistent cache
layout `[layer, K-or-V, token, KV-head, channel]`, the probability layout
`[query-position, query-head, key-position]`, and return both input owners with
the new context buffer.

The generated kernel uses a `32 x 8` block: 32 lanes read contiguous value
channels while eight lanes partition the history dimension. Partial F32 sums
use a fixed `8 x 32` shared-memory tile (1 KiB per block). This replaces the
generic one-block-per-output-channel reduction, its strided V reads, and
history-sized dynamic shared memory. The BF16 variant widens loads, accumulates
in F32, and rounds once per output channel.

This remains exact global attention and therefore still reads every prior V
entry for every decoded token; the primitive removes avoidable layout and
reduction overhead but does not change the linear per-token dependence on
context length.

## Native-BF16 routed expert primitives

`materializec.borrow-routed-native-bf16-gemv-pair` computes sparse-expert gate
and up projections together from BF16 activations and weights. Selected-expert
lookup, row addressing, and packed BF16 pair loads are shared; each projection
has an independent F32 accumulator and is rounded once to BF16.

`materializec.borrow-routed-native-bf16-gemv-residual` performs the routed down
projection from materialized BF16 expert activations, accumulates across all
selected expert slots in F32, adds the BF16 residual in F32, and rounds the
result once. Both operations return their owned activation, selected-expert,
and residual buffers so the Qwen3.6 graph retains linear ownership without
F32 conversion buffers.

## `materializec.into`

Consumes an owned typed buffer, materializes into a bounds-checked contiguous
range, and returns the owner. Qwen uses this to append keys and values to a
persistent GPU KV cache without reallocating or copying prior positions.

## `materializec.borrow`

Scopes a read-only buffer borrow to one materialization, then returns the owner
with the result. Persistent GPU state can be consumed by later kernels without
cloning it or allowing an alias to escape.

## `materializec.reduce-f32`

Assigns one output to each GPU block and cooperatively reduces terms in shared
memory. Its adjacent-pair F32 tree is exactly specified for bitwise
determinism; a supplied F32- or BF16-valued epilogue consumes the sum. Qwen
uses it for RMSNorm, fused projections, residual projections, and final
vocabulary logits.

## `materializec.borrow-reduce-f32`

Runs the same canonical cooperative reduction while scoping a read-only borrow
of an owned source buffer. It returns the owner with the result and supports
reductions through length 16,384, covering Qwen's wide projections without
changing the canonical reduction order.

## `materializec.reduce-f32-pair`

Computes two independent F32 reductions in one block. Gate and up projections
share index setup, term loading, launch overhead, and synchronization, while
retaining separate products, shared-memory arrays, and exact adjacent-pair
trees. The shared term function returns an F32 pair so common routed-expert
address calculations are evaluated once.

## `materializec.borrow-routed-bf16-gemv-pair`

Specializes the paired reduction for Qwen's sparse experts. It accepts an owned
F32 input, two referenced BF16 matrices, and an owned selected-expert buffer;
it returns both owners with gate and up outputs. Expert lookup and weight-base
calculation occur once per output block instead of once per reduction term.
BF16 conversion, F32 multiplication, and both reduction trees remain separate
and bitwise specified.

## `materializec.borrow-topk-f32`

Selects stable row-wise top-k indices entirely on the GPU while returning the
source owner. Rows through 128 columns use a synchronized 128-thread bitonic
sort; wider rows use a serial stable-insertion fallback. Sort keys define
finite and NaN ordering, canonicalize signed zero for comparison, and break
ties by lower source index. Qwen's 128-way top-8 router became about 9.6 times
faster per layer.

## `materializec.softmax-f32`

Normalizes contiguous F32 or BF16 rows in place with F32 arithmetic. Each block
performs specified maximum and denominator trees; masked negative infinities
become zero without evaluating the supplied exponential. Attention
probabilities therefore reuse the score allocation and retain the graph's
storage precision.

## `materializec.borrow-argmax-f32`

Computes deterministic argmax on the GPU while returning the source owner.
Only one U64 token ID is transferred to the host rather than the full 151,936
element vocabulary logits buffer.

## Deterministic reduction launch tuning

Physical launch width is selected independently of the logical reduction
tree. Borrowed reductions longer than 4096 terms use 512 threads; Qwen's
2048-term paired expert projection remains at its measured optimum of 256.

For tree levels with `stride >= 256`, the maximum supported reduction length
leaves at most 32 active lanes. Those levels use explicit warp/wave barriers
instead of whole-block barriers, without changing any term, addition operand,
logical index, or parenthesization. GPU tests compare every output bit with the
canonical CPU tree at lengths 257, 2048, 6144, 8192, and 16,384.

## Stream-ordered GPU execution

Materializations enqueue asynchronous allocation, kernel execution, and free
operations on one stream. This removes synchronization between model stages
and lets the GPU memory pool reuse intermediates. `Runtime::exec` synchronizes
once at the Catena program boundary.

# Architecture/model implementation changes made to achieve performance

## Registered GGUF weight views

The Qwen runner validates and read-only-maps the offline-packed BF16 GGUF once,
registers the complete mapping with HIP, and passes all 579 tensor ranges as
borrowed `MemRef`s in their native GGUF storage. This removes the model-sized
safetensors repack, upload, and duplicate allocation. The Catena consumers use
BF16 for the 338 large tensors and F32 for the 241 norms/router tensors, so the
remaining 48.80 MiB F32-to-BF16 setup conversion and its 24.40 MiB of device
allocations are also gone. The loader validates the per-role dtype contract.

Observed warm weight readiness remained 2.72--2.84 seconds. The eight-token
known continuation was unchanged, and one 99-cached-token native-dtype run
measured 10.33 tok/s, or 54.3% of the matched 19.03 tok/s llama.cpp baseline.
The preceding conversion-based loader measured 10.92 tok/s over 99 cached
tokens and 9.83 tok/s over 999; a matched long native-dtype median has not yet
been recorded.

## Persistent KV cache

A session preallocates the complete layered key/value cache. Prefill and decode
append RoPE-applied keys and raw values in place, and decode evaluates only the
new token.

## Catena-owned RoPE table

RoPE is no longer constructed with model-specific Rust `powf`, `sin`, and
`cos` calls or passed as an additional model input. The Catena entrypoint
materializes only the current absolute-position range once per call and reuses
that owned table across all 48 layers. Exact F32 inverse-frequency constants,
a three-part Cody-Waite range reduction, and fixed degree-13/14 sine/cosine
polynomials make the operation graph explicit. Across all 32,768 supported
positions, the GPU table's maximum absolute difference from the previous host
table was `4.92e-7` for sine and `4.17e-7` for cosine. The known continuation
was unchanged; three 99-cached-token samples reached 10.42, 10.43, and 10.35
tok/s (10.42 tok/s median).

## Cooperative fused projections

RMSNorm, Q/K/V projections, attention output, residual connections, MoE down,
and vocabulary projection use cooperative exact-tree reductions. Normalization
is fused into projection terms and residual addition into epilogues, avoiding
standalone normalized or residual buffers where reuse is unnecessary.

## Staged cached attention

Each QK score is materialized once, normalized in place, and reused across all
value channels. The dedicated cached-context primitive maps contiguous value
channels across neighboring threads and partitions history across eight key
lanes, then reduces the eight partial F32 sums in a fixed 1 KiB shared tile.

In a controlled Qwen3-1.7B BF16 comparison on the same local GPU/runtime, this
changed TG512 from 17.59 to 18.70 tok/s (+6.3%) and TG2048 from 13.99 to 15.37
tok/s (+9.9%). Exact global attention still scales linearly with existing
context for each decoded token, so the gain grows with context but does not
remove the underlying slope.

## Materialized sparse-MoE activation

Qwen activates eight of 128 experts per token. `silu(gate) * up *
routing_weight` is materialized once for each of the 6144 active intermediate
values and reused across all 2048 down-projection outputs. This avoids
recomputing the exponential and routing products inside every down term.

## Paired routed gate/up projection

Gate and up weights for each selected expert are read by one typed BF16 kernel.
The expert ID, row, channel, and input address are shared while the two
arithmetic paths and deterministic sums remain independent.

## GPU routing and token selection

Stable top-8 expert routing and final greedy argmax both remain on the GPU.
The host receives only the selected token ID during generation; expert IDs and
router weights stay device-resident.

## Final normalization reuse

The final RMSNorm statistic is computed once per sequence row and reused by
the tied-embedding vocabulary projection instead of being recomputed for every
logit.

## Explicit intermediate lifetimes

K/V projections, scores, probabilities, contexts, normalization statistics,
router buffers, gate/up outputs, and MoE activations are freed immediately
after their last queued consumer. Stream ordering makes these frees safe and
enables allocator reuse.

## Sustained result

For 1000 generated tokens from `"The capital of France is"`, the final BF16
Catena build measured 9.90, 9.81, and 9.88 cached tok/s: a 9.88 tok/s median.
The initial Catena implementation measured 7.79 tok/s, so the retained changes
improved sustained generation by 26.8%.

The recorded matched llama.cpp BF16 run measured 19.03 tok/s, making Catena
51.9% as fast. A later llama.cpp sample reached 20.74 tok/s, against which the
same Catena median is 47.6%. At 99 cached tokens, before the runners' different
ignore-EOS behavior changes their streams, Catena measured 10.95 tok/s versus
llama.cpp's 19.42 tok/s, or 56.4%. Detailed measurements, determinism notes,
and rejected experiments are retained in `EXPERIMENT_LOG.md`.

After replacing the safetensors upload with registered GGUF views, one full
validation run reached 9.83 tok/s over 999 cached tokens. This is 51.7% of the
matched 19.03 tok/s baseline and confirms the loading change retains the
requested sustained-performance threshold; it is not a replacement for the
three-run median above.
