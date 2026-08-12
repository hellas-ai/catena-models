# Language features added to achieve performance

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
determinism; a supplied epilogue consumes the sum. Qwen uses it for RMSNorm,
dense projections, residual projections, and final vocabulary logits.

## `materializec.borrow-reduce-f32`

Runs the same canonical cooperative reduction while scoping a read-only borrow
of an owned source buffer. It returns the owner with the result and supports
reductions through length 8192, covering Qwen's 6144-term MoE down projection
and cache-length attention context.

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

Normalizes contiguous F32 rows in place. Each block performs specified maximum
and denominator trees; masked negative infinities become zero without
evaluating the supplied exponential. Attention probabilities therefore reuse
the score allocation.

## `materializec.borrow-argmax-f32`

Computes deterministic argmax on the GPU while returning the source owner.
Only one U64 token ID is transferred to the host rather than the full 151,936
element vocabulary logits buffer.

## Deterministic reduction launch tuning

Physical launch width is selected independently of the logical reduction
tree. Borrowed reductions longer than 4096 terms use 512 threads; Qwen's
2048-term paired expert projection remains at its measured optimum of 256.

For tree levels with `stride >= 256`, the maximum supported reduction length
leaves at most 16 active lanes. Those levels use explicit warp/wave barriers
instead of whole-block barriers, without changing any term, addition operand,
logical index, or parenthesization. GPU tests compare every output bit with the
canonical CPU tree at lengths 257, 2048, and 6144.

## Stream-ordered GPU execution

Materializations enqueue asynchronous allocation, kernel execution, and free
operations on one stream. This removes synchronization between model stages
and lets the GPU memory pool reuse intermediates. `Runtime::exec` synchronizes
once at the Catena program boundary.

# Architecture/model implementation changes made to achieve performance

## Registered GGUF weight views

The Qwen runner validates and read-only-maps the offline-packed BF16 GGUF once,
registers the complete mapping with HIP, and passes its BF16 tensor ranges as
borrowed `MemRef`s. This removes the model-sized safetensors repack, upload, and
duplicate allocation. Only 48.80 MiB of GGUF tensors promoted exactly to F32
are verified and restored into 24.40 MiB of BF16 device allocations. Observed
weight readiness fell to 2.72--2.84 seconds, while the known continuation and
short cached throughput were retained (10.92 tok/s over 99 cached tokens versus
the earlier 10.95 tok/s sample). A 999-cached-token validation reached 9.83
tok/s, or 51.7% of the matched 19.03 tok/s llama.cpp baseline.

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
value channels. Context accumulation cooperatively reduces the cache-length
dimension rather than traversing history serially for every output channel.

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
