# Qwen3.6 performance experiments

This file records performance work for the exploratory BF16
Qwen3-30B-A3B-Base implementation. Compiler changes are recorded even when
they are rejected so that a later mainline implementation can preserve
Catena's bitwise-determinism contract deliberately.

## Measurement status

The first matched 1,000-token development measurement uses the prompt
`The capital of France is`, greedy selection, and EOS ignored in both runners:

| Runner | Cached tokens | Time | Throughput |
| --- | ---: | ---: | ---: |
| llama.cpp 9190, BF16 GGUF, full ROCm offload | 999 | 52.48 s | 19.03 tok/s |
| Catena, BF16 safetensors | 999 | 128.22 s | 7.79 tok/s |

Catena is therefore at 40.9% of llama.cpp. The 50% threshold is 9.52 tok/s,
requiring a 22.2% Catena speedup. These are single development runs; the final
claim must use repeated runs and report the median. A shorter 99-token Catena
run reached 8.64 tok/s before the wave-reduction experiment below.

The optimization sequence produced these later fixed-length measurements:

| Catena state | Cached tokens | Time | Throughput |
| --- | ---: | ---: | ---: |
| stable top-k, 512-thread down reduction, initial paired gate/up | 999 | 105.66 s | 9.46 tok/s |
| shared paired term plus parallel top-k, median of three | 999 | 102.80 s | 9.72 tok/s |
| typed routed-BF16 pair | 999 | 101.42 s | 9.85 tok/s |
| typed pair plus late wave barriers, median of three | 999 | 101.09 s | 9.88 tok/s |

The three pre-tail runs were 9.70, 9.74, and 9.72 tok/s. Their 9.72 tok/s
median is 51.1% of the original matched 19.03 tok/s llama.cpp baseline. The
final-build runs were 9.90, 9.81, and 9.88 tok/s; their 9.88 tok/s median is
51.9%. This clears the requested threshold against that recorded baseline.

There is an important denominator caveat. A later llama.cpp fixed-length run
reached 20.74 tok/s, against which 9.88 tok/s is only 47.6%. The two runners
also treat EOS differently under their ignore-EOS options: llama.cpp suppresses
EOS while this runner feeds the emitted EOS token back into the base model, so
their streams diverge after natural termination. The 99-token checkpoint stays
before that issue: llama.cpp reached 19.42 tok/s and the final Catena build
reached 10.95 tok/s, or 56.4%. Both baselines are retained here so the
exploratory result is not overstated.

## Registered GGUF zero-copy loading (kept)

The original Qwen loader mapped 18,867 safetensors, copied each source tensor
through a Rust `Vec`, concatenated every 128-expert group, then allocated and
uploaded all 579 Catena buffers. The retained loader instead consumes the
unquantized `model-bf16.gguf` already used by llama.cpp. That file contains the
expert groups in Catena's required numeric order.

The loader validates GGUF v3, the complete name inventory, reversed GGUF
dimensions, dtypes, alignment, checked byte ranges, and non-overlap before
inference. It makes one read-only file mapping and registers it with HIP's
mapped/read-only coarse-grained flags. The resulting device-visible base is
split into borrowed `MemRef`s whose lifetime is tied to the registration;
registration is released before the mmap. No mapped pointer is given `MemOwn`
ownership.

Of the 56.90 GiB file, 56.85 GiB is BF16 tensor payload read directly by the
kernels. llama.cpp's converter stores 48.80 MiB of small norms/router weights
as F32. Every one of those values is verified to have zero low 16 bits before
its original BF16 high word is uploaded, producing only 24.40 MiB of copied
device weights. This validation makes the conversion bit-exact rather than a
new rounding step.

Observed load-to-ready times were 2.72--2.84 seconds. The eight-token
known continuation stayed exactly ` Paris. Which of the following options
correctly`. A 99-cached-token run measured 10.92 tok/s versus the earlier 10.95
tok/s sample. A full 999-cached-token validation measured 9.83 tok/s, or 51.7%
of the matched 19.03 tok/s llama.cpp baseline. The loading change therefore
retained both short and sustained throughput while removing the model-sized
repack/upload and duplicate allocation. This mode is specific to the tested
HIP unified-memory APU for now; discrete hardware must benchmark registered
host access against an ordinary device upload.

## Materialized MoE activation (kept)

Qwen3-30B-A3B activates eight experts per token. The first implementation
evaluated `silu(gate) * up * routing_weight` inside every term of the
2,048-output down projection. That repeated the same exponential and two
multiplications once per output channel.

The current implementation materializes the 6,144 active intermediate values
once per row and feeds that buffer to the cooperative down projection. This is
an architecture-level scheduling change using existing `materializec` and
`materializec.borrow-reduce-f32` features. It does not change the expression or
its operation order for an intermediate value; it only reuses the resulting
F32 bits across output channels.

Validation so far:

- The generated 48-layer HIP program compiles.
- The 100-token greedy continuation remained unchanged.
- A bitwise intermediate/logit comparison against the pre-materialization
  implementation is still required before mainlining.

## Wave-local exact-tree reduction (rejected and reverted)

### Motivation

The existing `materializec.reduce-f32` and
`materializec.borrow-reduce-f32` kernels store every term in shared memory and
execute the specified adjacent-pair tree with a whole-block barrier at every
level. Qwen's 2,048- and 6,144-term expert projections therefore use roughly
12 to 14 block barriers per output row.

### Experimental implementation

The compiler experiment divided the input into block-sized, power-of-two
chunks. It attempted to preserve the original adjacent-pair parenthesization
by reducing:

1. adjacent values within each hardware wave using `__shfl_down`;
2. adjacent wave roots within each block-sized chunk; and
3. adjacent chunk roots in the original tree order.

This reduced the shared-memory phase to one whole-block barrier. The complete
Catena test suite (139 unit/integration tests), Qwen's generated HIP compile,
and a 100-token generation run passed.

### Why it was rejected

- Cached throughput fell from 8.64 to 7.69 tok/s (about 11% slower). Repeating
  a shuffle tree for every block-sized chunk cost more than the removed
  barriers on this workload.
- Identical greedy text is not evidence of bitwise-identical reductions.
- Filling inactive lanes with `+0.0` is not strictly equivalent for every IEEE
  bit pattern, notably signed zero, even when the apparent addition tree is
  otherwise preserved.
- Hardware wave width is dialect-specific, adding another surface that needs
  explicit deterministic tests.

The experiment was therefore reverted. A future reduction optimization must
first add a GPU test that compares every output bit against the canonical
shared-memory adjacent-pair implementation across power-of-two and irregular
lengths, including signed zero, infinities, and NaNs. Performance should then
be measured independently; passing the bitwise test is necessary but not
sufficient.

## Stable row-wise top-k primitive (kept)

Launch-time kernel tracing showed that the original router selector consumed
12.64% of GPU kernel time. It produced each of eight selected IDs separately;
for every candidate it counted all 128 rivals that ranked ahead, resulting in
`8 * 128 * 128` comparisons per row.

The exploratory compiler now exposes `materializec.borrow-topk-f32`. For rows
up to 128 columns, one 128-thread block transforms the input bits to sortable
U32 keys and performs a block-wide bitonic sort. A serial stable-insertion
fallback handles wider rows. The primitive returns the source owner alongside
a row-major U64 index buffer, retaining the ownership pattern of
`materializec.borrow-argmax-f32`.

Determinism properties:

- The primitive never modifies or performs arithmetic on an F32 input.
- Comparison keys preserve finite numeric ordering, canonicalize `-0.0` and
  `+0.0` as equal, and provide a fixed bitwise order for NaNs.
- Equal keys select the lower column index, matching Qwen's stable top-k tie
  behavior.
- Each compare-exchange has a unique owner and fixed synchronization points;
  there are no atomics or scheduling-dependent writes.
- The serial path uses the same key and tie rules as the parallel path.

GPU tests cover 128 columns (parallel), 129 columns (fallback), infinities,
ties, signed zero, returned ownership, and preservation of all source bits.
The selected IDs and generated continuation remained unchanged. At a 1K-token
position, router selection fell from 52.79 microseconds per layer to 5.50
microseconds, about 9.6 times faster. Three end-to-end runs with this path had a
9.72 tok/s median.

## Reduction block widths (kept selectively)

The MoE down projection accounted for 23.93% of traced kernel time. Its
6,144-term `materializec.borrow-reduce-f32` allocates 24 KiB of shared memory
per block, limiting the number of resident blocks. The physical width for
reductions above 4,096 terms is experimentally increased from 256 to 512
threads to expose more memory operations from each resident block.

This does not change reduction semantics: every term is still written to the
same shared-memory logical index, and the subsequent adjacent-pair tree is
unchanged. Block width affects only which thread materializes each term and
which independent additions it executes between whole-block barriers. The
existing logical-tree bitwise test and Qwen generation check must pass before
measuring the result.

The 512-thread borrowed reduction kept the 100-token continuation unchanged,
passed the explicit bit-pattern tree test, and raised 99-token cached
throughput from 9.06 to 9.46 tok/s, so it is retained. Applying the same width
to ordinary reductions above 1,024 terms was rejected and reverted: despite
using the same indexed-term materialization and unchanged shared-memory tree,
cached throughput fell to 8.39 tok/s. Ordinary 2,048-term expert gate/up,
dense, and vocabulary projections remain at 256 threads.

## Paired gate/up reduction (kept)

The exploratory compiler exposes `materializec.reduce-f32-pair`. One GPU block
computes the gate and up terms together and keeps two independent shared-memory
reduction arrays. At every level it performs the same left-plus-right additions
at the same logical indices as two `materializec.reduce-f32` calls; only term
setup, launch overhead, and whole-block barriers are shared.

The initial form accepted two independent environments and term functions. A
20-token launch-time trace showed the paired kernel taking 28.84% of dispatch
time, averaging 0.657 ms per layer, compared with approximately 0.694 ms for
the two prior gate/up kernels together. The matched 999-token run reached 9.46
tok/s. A subsequent API refinement uses one term function returning an F32
pair, so Qwen computes the expert/channel indices and input/router loads once.
Its 20-token trace averaged 0.635 ms per layer versus 0.657 ms for the two-env
pair. Three matched long runs after this and parallel top-k reached a 9.72 tok/s
median.

Determinism is structural: each multiplication remains a separate F32
operation, gate and up values occupy separate arrays, and neither reduction
tree is reassociated. HIP tests include the cancellation-sensitive tree
`(1e20 + 1) + (-1e20 + 1)` for both outputs. This change does not combine the
two sums arithmetically.

A 512-thread launch for the 2,048-term pair was rejected: indexed term
materialization and both shared-memory trees were unchanged, but traced
per-layer time regressed from 0.635 to 0.704 ms. The extra eight-wave block
pressure outweighed any occupancy benefit. A six-wave 384-thread launch also
regressed to 0.686 ms. Both experiments were reverted; the paired reduction
remains at its measured optimum of 256 threads.

## Typed routed-BF16 gate/up primitive (kept exploratory)

The compiler fork now exposes
`materializec.borrow-routed-bf16-gemv-pair`. It accepts an owned F32 input,
two referenced BF16 expert matrices, and an owned U64 selected-expert buffer.
Weights use `[expert, output-feature, reduction-feature]`; selected IDs use
`[row, slot]`; outputs use `[row, slot, output-feature]`. The primitive returns
both owners with the two F32 result buffers.

This specialization moves selected-expert lookup, row/channel division, and
weight-base calculation outside the per-reduction term. Each input value is
loaded once before its independent gate and up multiplications. It is not a
mixed or fused arithmetic operation: each BF16 weight is converted separately,
each F32 product is stored in its own shared array, and both arrays follow the
same canonical adjacent-pair trees as `materializec.reduce-f32-pair`.

Validation includes a two-row/two-expert GPU test that checks both returned
owner pointers and all gate/up results. The full 48-layer module compiles and
the known greedy continuation is unchanged. The paired kernel improved from
0.635 to 0.621 ms per layer in the comparable 20-token traces. A single 999
cached-token run improved from the earlier 9.72 tok/s median to 9.85 tok/s.
The benefit is real but modest; the primitive remains an explicitly
Qwen-shaped exploratory compiler feature pending a more general mainline API.

## Late canonical-tree wave barriers (kept exploratory)

The original reduction renderer used `__syncthreads()` after every logical
tree level. Once `stride >= 256`, the configured maximum reduction length of
8,192 leaves at most 16 active lanes, all inside the first 32-lane CUDA warp or
64-lane AMD wave. The compiler now retains exactly the same indexed additions
but uses explicit `__syncwarp()` barriers for those late levels. Earlier levels
still use whole-block barriers.

This changes neither term bits nor the addition graph. Every output index is
written by the same lane, every addition retains the same left and right
operands, and an explicit wave barrier separates dependent levels. A GPU test
compares every output bit with a CPU implementation of the canonical tree at
lengths 257, 2,048, and 6,144 for ordinary, paired, and borrowed reductions.
The values include cancellation-sensitive magnitudes and signed zero. The
known Qwen continuation is also unchanged.

Relative to the typed-pair trace with whole-block barriers, average kernel
times changed as follows:

| Kernel | Whole-block tail | Wave tail | Change |
| --- | ---: | ---: | ---: |
| expert gate/up pair, per layer | 0.621 ms | 0.563 ms | -9.2% |
| MoE down, per layer | 0.588 ms | 0.564 ms | -4.0% |
| normalized dense projection, per launch | 0.130 ms | 0.124 ms | -4.5% |
| residual projection, per layer | 0.247 ms | 0.237 ms | -3.9% |
| final vocabulary logits, per token | 8.978 ms | 8.441 ms | -6.0% |

The 99-token check reached 10.95 tok/s. Three 999-token runs reached 9.90,
9.81, and 9.88 tok/s, for a 9.88 tok/s median. Mainlining must retain the
explicit maximum-active-lane proof or make the threshold dialect-specific;
silently assuming a hardware wave width would violate the determinism
contract.

## Serialized canonical-tree tail (rejected and reverted)

Before the wave-barrier form, an experiment stopped block cooperation at
stride 256 and let thread zero execute the remaining canonical-tree additions
serially. The operation graph and output bits were unchanged, but the 99-token
check fell to 10.43 tok/s versus 10.95 tok/s for the wave-barrier form. Removing
barriers did not compensate for serializing independent top-level additions,
so this variant was reverted.

## Serial canonical attention-context reduction (rejected and reverted)

An experimental `materializec.borrow-reduce-f32-serial` assigned one output to
each thread. A 16-level binary-carry accumulator reproduced the canonical
adjacent-pair tree while allowing a 128-thread block to cover one attention
head, making value-cache channel reads contiguous. The first implementation
combined residual partial trees in the wrong direction; a bitwise comparison
caught a one-bit difference at reduction length 127. The corrected algorithm
builds the irregular right tail from the smallest partial upward, always adding
the larger left subtree on the left.

After correction, the serial and cooperative GPU kernels matched output bits at
lengths 0, 1, 3, 5, 127, 128, 129, 257, and 2,048, including signed-zero inputs.
The full Qwen continuation also remained unchanged. Performance nevertheless
regressed: the 99-token check fell from 10.39 to 10.30 tok/s, and the matched
999-token run fell from 9.46 to 8.38 tok/s. Coalescing did not compensate for
removing parallelism across the history dimension. The Qwen use, primitive,
renderer, declaration, and tests were all reverted; the failed approach and
the determinism bug are recorded here so it is not repeated casually.

## Profiling note

`profile-qwen3.sh` uses launch-time rocprofv3 tracing and accepts the token count
as its second argument. This avoids both fixed delays and dynamic-attach races.
The runner retains opt-in FIFO/delay hooks for diagnosis, but the recorded
profiles do not use them.

The exploratory compiler also has opt-in `CATENA_GPU_MODULE_SAVE` and
`CATENA_GPU_MODULE_LOAD` hooks. They copy the generated shared object out of or
into its normal temporary artifact directory. This permits a module compiled
in an unprofiled process to be loaded under launch-time profiling, because
ROCm instrumentation interfered with the nested `hipcc` process. With the
cached module, ROCm 7.2.3 records complete kernel statistics. The hooks do not
alter generated source or arithmetic, are disabled by default, and
intentionally leave cache validity to the caller. A mainline version needs
content-addressed validation rather than these diagnostic hooks.

## Next candidates

The recorded 19.03 tok/s baseline is cleared, but the later 20.74 tok/s
llama.cpp sample is not. Further work should remain measurement-led and retain
bitwise tree tests. The largest current kernels are routed gate/up, MoE down,
normalized dense projections, residual projections, and final vocabulary
logits. Plausible next steps are a general typed BF16 GEMV API, vectorized
weight loads that provably preserve product and tree bits, and a matched
pre-EOS benchmark protocol shared by both runners.
