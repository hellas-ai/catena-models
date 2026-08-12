# RoPE performance

## Retained implementation

The current Qwen implementation constructs RoPE inside Catena for every model
invocation. `qwen3.rope-table(start, sequence)` allocates and materializes only
the rows needed by that invocation:

```text
prefill:  sequence * 128 F32 values
decode:   1 * 128 F32 values
layout:   [cos(sequence, 64), sin(sequence, 64)]
```

The resulting owned buffer is shared by all 48 decoder layers in that call and
freed after final logits have been produced. Rust no longer calculates or
uploads RoPE. Absolute positions are incorporated by `rope-table-at` using
`start + local_position`, so consumers index the call-local table from zero.

This design is deliberately retained for now because it is simple, keeps the
KV cache layout unchanged, and keeps all RoPE arithmetic in Catena. Its cost is
that every one-token decode performs a small device allocation, launches a RoPE
materialization kernel, and frees the allocation. The generated work is tiny,
so launch and allocator overhead can dominate it.

With the registered BF16 GGUF and the prompt `"The capital of France is"`, the
per-call implementation measured 10.42 tok/s median over three 99-cached-token
runs and 9.39 tok/s over one 999-cached-token run. The earlier registered-GGUF
implementation with host-provided RoPE measured 9.83 tok/s over 999 cached
tokens. These are development measurements rather than a controlled attribution
of the entire difference, but they make the per-call RoPE lifecycle worth
removing eventually.

## Proposed fixed table

RoPE depends only on model constants and absolute position. Qwen therefore does
not need a new table for each session or decode call. A future implementation
can materialize the entire table once:

```text
MAX_POSITION_EMBEDDINGS * HEAD_DIM
= 32,768 * 128
= 4,194,304 F32 values
= 16 MiB

layout:
[cos(32,768, 64), sin(32,768, 64)]
```

The initialization should still be a Catena program. During `Model::load`, Rust
would execute something equivalent to:

```text
qwen3.rope-table(0, MAX_POSITION_EMBEDDINGS)
```

Rust would retain the returned owner for the lifetime of `Model`, but would not
calculate or copy any table values. Each model invocation would pass a borrowed
`cap.ref mem` view of that allocation. Attention and key-cache producers only
read RoPE, so their RoPE parameters should likewise become `cap.ref mem`; they
do not need ownership.

Indexing a full table differs from indexing the current call-local fragment.
For local position `p` and frequency pair `i`, consumers would use:

```text
absolute_position = start + p
cos_index = absolute_position * 64 + i
sin_index = MAX_POSITION_EMBEDDINGS * 64 + cos_index
```

This preserves the current non-interleaved plane layout and the same
deterministic Catena approximation. The existing exhaustive GPU test should
continue comparing all 4,194,304 generated values against the former host
reference. Its current maximum absolute errors are approximately
`4.92e-7` for sine and `4.17e-7` for cosine.

### Required implementation changes

1. Add a `rope: MemOwn` field to `Model` and construct it once with the Catena
   `qwen3.rope-table` entrypoint after the runtime is created.
2. Add one `(cap.ref mem)` RoPE argument to the generated Qwen model entrypoint.
3. Pass `model.rope.as_ref()` on every prefill and decode invocation.
4. Change RoPE parameters through the attention call graph from `cap.own mem`
   to `cap.ref mem`.
5. Change `qwen3.rope-value` to index the full table using absolute position and
   the fixed 32,768-position sine-plane offset.
6. Remove per-call `qwen3.rope-table` construction and the corresponding final
   RoPE free from the generated model entrypoint.
7. Re-run determinism, known-token, 99-token, and 999-token validation.

This proposal needs no compiler primitive and must not combine RoPE with the KV
cache. Keeping two allocations preserves the existing zero-based KV layout and
avoids adding a RoPE prefix offset to history-scaled attention loads.

## Alternative: persistent partial materialization

A fixed 16 MiB allocation could instead be filled incrementally with
`materializec.into` using:

```text
offset = start * 128
count  = sequence * 128
```

That avoids allocation and freeing but still launches a kernel during every
decode unless rows are generated ahead of use. It also makes the RoPE buffer
mutable session state. Since the complete table is only 16 MiB and is identical
for every session, materializing it once at model initialization is simpler and
removes all steady-state RoPE generation work.
