# Serving Qwen

The Qwen application can keep a model loaded and serve remote text inference
through the OpenAI Chat Completions and Responses protocols:

```bash
CATENA_API_KEY=local-secret cargo run -r -- \
  --model unsloth/Qwen3.5-0.8B-GGUF:BF16 \
  --serve --host 0.0.0.0 --port 8000 --ctx-size 4096
```

Without `--model`, both CLI and server modes select
`unsloth/Qwen3.6-35B-A3B-GGUF:BF16`. An explicit `--model` overrides this default.

The default address is `127.0.0.1:8000`. Set `--host` to the interface on which
clients should connect. `CATENA_API_KEY` (or `--api-key`) optionally requires a
bearer token on every endpoint. Without a configured key, authentication is
disabled. HTTP is served directly; TLS can be supplied by a reverse proxy.
Model downloads, GPU selection, `--kernels`, and `--wmma-prefill` work as in
one-shot mode. WMMA prefill is enabled by default; pass `--no-wmma-prefill` to
select SIMT (required for Qwen3 models). CUDA requires `CATENA_WMMA_EMULATE=1`
for the default WMMA path, or the SIMT opt-out. Startup loads the tokenizer, embedded GGUF chat template, compiled
kernels, and weights before opening the listener. No catena-lang changes are
required.

## Protocol logging

Set the protocol logger to `debug` to see incoming request bodies, outgoing JSON
responses and errors, and each outgoing SSE message (including usage and
`[DONE]`). Logs go to stderr and include a `request_id` linking each exchange:

```bash
RUST_LOG=catena_models::server::protocol=debug cargo run -r -- --serve
```

Payload logging is off by default. It includes full prompts, tool arguments,
reasoning, and generated text; request and response headers are not logged.
Captures are limited to 8 MiB per body or SSE message, with `truncated=true`
when clipped. Disconnects or body errors mark incomplete captures `partial=true`.
Logging observes delivery without buffering the stream sent to clients.

## Requests

```bash
curl http://localhost:8000/v1/models -H 'Authorization: Bearer local-secret'
curl http://localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3.5-0.8B-GGUF:BF16",
       "messages":[{"role":"user","content":"Explain Rust ownership briefly."}],
       "temperature":0,"max_completion_tokens":128,"stream":true,
       "stream_options":{"include_usage":true}}'
```

`GET /health` reports whether the inference worker is available. `/v1/models`
advertises exactly the supplied `--model` selector; send that same ID in requests.
Chat requests support:

- Multi-turn `system`, `developer`, `user`, `assistant`, and `tool` messages.
  Content may be a string or an array of text parts. System/developer instructions
  are combined into the initial system message, in their original order.
- Greedy decoding only: omit `temperature` or set it to `0`. `top_p=1`, `n=1`,
  and zero frequency/presence penalties are accepted. Other values are rejected.
  `seed` is accepted but has no effect because decoding has no random state.
- `max_completion_tokens` or `max_tokens`; if both appear they must agree.
  The default is the smaller of 32,768 and the remaining context, including
  reasoning and tool-call tokens. Responses uses the same default when
  `max_output_tokens` is omitted. Explicit output
  limits plus prompt tokens must fit `--ctx-size`; history is never truncated.
  Server mode does not use the one-shot `--max-new-tokens` or `--ignore-eos` flags.
- Up to four nonempty `stop` strings. Stop text and model end markers are omitted
  from returned content. Completion usage includes generated reasoning/tool
  tokens and terminating model tokens.
- `stream=true` returns incremental SSE chunks, a finish reason, optional final
  usage, and `[DONE]`. Non-streaming responses return the assembled assistant
  message and usage.
- `chat_template_kwargs: {"enable_thinking": true}` enables thinking. It defaults
  to false. Reasoning is returned separately as `message.reasoning_content` or
  streamed `delta.reasoning_content`, without `<think>` delimiters.

Send the full conversation on every turn, including prior tool calls and their
matching results. The server reuses matching prompt checkpoints as described
below. GPU inference is
serialized on one owning thread; up to eight requests can wait. A full queue
returns `429`. Dropping a request cancels inference between GPU operations and
releases its session. HTTP model/health requests remain responsive during
inference. Request bodies are limited to 8 MiB. Ctrl-C shuts the server down.

## Prompt prefix caching

Prefix caching is enabled by default for both APIs. It keeps at most two GPU
checkpoints on the inference worker: a stable prefix through the latest user
message, and a recent prompt prefix for tool continuations. Matching compares
the rendered token IDs exactly. A changed system prompt, tool definition, or
history edit can cause a miss; Qwen's removal of old reasoning can invalidate
the recent checkpoint while leaving the stable one usable. This is a small
shared cache, not a separate cache for each client or conversation.

Checkpoints contain all session state, including attention KV, DeltaNet
recurrence, and convolution history. Generated output is not cached. Clients
must still replay history, and Responses `previous_response_id` remains
unsupported. `prompt_cache_key` does not determine a hit.

Serving prefills use fixed 1,024-token blocks from position zero, with a final
partial block when needed. Checkpoints are saved only at full-block boundaries;
prompts shorter than 1,024 tokens do not populate the cache. `--no-prefix-cache`
disables reuse and checkpoint storage but retains this schedule, so cold and
warm execution use the same arithmetic order. This schedule can produce
different bits from older single-batch serving and from one-shot CLI prefill.

Session capacity is rounded up to a power of two covering prompt plus output
budget, capped at `--ctx-size`. Reuse requires the same capacity bucket; growing
past a bucket boundary causes a cold prefill. The cache can retain two full
session allocations in addition to the working session. Identical checkpoints
share one allocation. Short unrelated requests do not evict useful checkpoints.

Usage reports reused tokens in Chat Completions
`prompt_tokens_details.cached_tokens` and Responses
`input_tokens_details.cached_tokens`. Responses also reports
`input_tokens_details.cache_write_tokens`: newly computed prompt tokens covered
by saved checkpoints. Counts are summed across structured-output correction
attempts; total prompt usage still includes all input tokens.

Use `RUST_LOG=catena_models::server::prefix_cache=debug` to see hits, reused-token
counts, capacity, GPU copy time, and suffix prefill time.

GPU validation and benchmarks accept a local BF16 GGUF or HF selector:

```bash
export CATENA_PREFIX_MODEL=unsloth/Qwen3.5-0.8B-GGUF:BF16
cargo test -r --test session_checkpoint -- --ignored --nocapture
cargo test -r --bin catena-models serving_prefix_cache_bitwise_parity -- --ignored --nocapture
RUST_LOG=catena_models::server::prefix_cache=debug \
  cargo test -r --bin catena-models serving_prefix_cache_benchmark -- --ignored --nocapture
```

Repeat with a MoE model to exercise its additional state. Set
`CATENA_PREFIX_WMMA=0` for SIMT, or `CATENA_GPU_DIALECT=cuda` with
`CATENA_WMMA_EMULATE=1` for emulated WMMA. The raw device-copy test uses HIP.
`CATENA_PREFIX_LENGTHS=4096,16384,32768` controls benchmark prefix lengths;
each case appends 32 tokens and reports cold and warm prefill latency.

## Function tools and coding clients

Select an **OpenAI-compatible Chat Completions** provider in the client and set:

- Base URL: `http://SERVER:8000/v1`
- API key: the configured bearer token, or a placeholder if authentication is off
- Model: the exact ID returned by `/v1/models`
- Temperature: `0`
- Context/output limits: within the configured `--ctx-size`

Function definitions use the standard `tools` array. The server supports
`tool_choice="auto"` (default) or `"none"`, and `parallel_tool_calls`. It translates
Qwen3 JSON and Qwen3.5 XML-style calls into OpenAI `tool_calls` with IDs and
JSON-string `function.arguments`. Multiple calls are returned with stable indices
in streaming responses. Tool blocks are buffered until complete; surrounding
text and reasoning still stream incrementally.

**The client executes tools.** The server never runs shell commands or accesses
client files. Append the returned assistant message and a `role="tool"` message
for each call, using its `tool_call_id`, then make the next request. For example,
this OpenAI Python SDK loop can be used by a client that has a local `read_file`
implementation:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://SERVER:8000/v1", api_key="local-secret")
model = "unsloth/Qwen3.5-0.8B-GGUF:BF16"
messages = [{"role": "user", "content": "Read src/main.rs and summarize it."}]
tools = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file in the client workspace",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}, "required": ["path"]}}}]

for _ in range(8):
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        temperature=0, max_completion_tokens=512)
    message = response.choices[0].message
    messages.append(message.model_dump(exclude_none=True))
    if not message.tool_calls:
        print(message.content)
        break
    for call in message.tool_calls:
        arguments = json.loads(call.function.arguments)
        # Supply your harness's own tool implementation and access policy.
        result = read_file(arguments["path"])
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": result})
```

Tool names and JSON object arguments are checked, including required arguments
and basic declared property types. This is not a constrained JSON-schema
sampler. Tool-definition `strict=true`, forced/required tool choices, non-function tools,
Chat Completions structured-output controls, logprobs, and non-neutral logits/sampling controls are
rejected explicitly. A malformed generated tool call returns an error rather
than exposing internal tool markup as answer text.

## Responses API

`POST /v1/responses` uses the same loaded model, greedy decoding, queue,
authentication, context limits, cancellation, and protocol logging as Chat
Completions. It accepts a string `input`, or a replayed array of text messages,
reasoning items, function/custom calls, and their matching outputs:

```bash
curl http://localhost:8000/v1/responses \
  -H 'Authorization: Bearer local-secret' -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3.5-0.8B-GGUF:BF16", "store":false,
       "instructions":"Answer concisely.", "input":"Explain Rust ownership.",
       "temperature":0, "max_output_tokens":128, "stream":true}'
```

JSON responses contain `object="response"`, output items with stable IDs,
`status`, and input/output token usage. The OpenAI Python SDK exposes message
text through `response.output_text`. Streaming uses named Responses SSE events
with increasing `sequence_number` values:

- `response.created`, `response.in_progress`;
- item and content-part added events, then incremental output text or reasoning
  summary text deltas and corresponding done events;
- function argument or custom-tool input delta/done events with matching item IDs
  and output indices;
- terminal `response.completed`, `response.incomplete`, or `response.failed`.

There is no Chat Completions `[DONE]` sentinel on this endpoint. Reaching the
token limit produces `status="incomplete"` with
`incomplete_details.reason="max_output_tokens"`; clients must handle that
terminal event rather than assume every response completes successfully.
Generation errors after streaming begins produce an error and a failed response,
with no successful completion event.

This server is **stateless**. `store` defaults to false; `store=true` and
`previous_response_id` are rejected. To continue, resend the prior input plus
**all** prior output items and any matching tool results. There are no response
retrieval/deletion, WebSocket, background, compaction, or Conversations endpoints.

### Responses function tools

Function declarations use top-level `name`, `description`, `parameters`, and
`strict`, rather than Chat Completions' nested `function` object. The same
`auto`/`none`, greedy, non-constrained restrictions apply. The server returns
`function_call` items with JSON-string `arguments`. Clients execute tools and
send `function_call_output` items linked by **call_id**, not the output item ID:

```python
from openai import OpenAI

client = OpenAI(base_url="http://SERVER:8000/v1", api_key="local-secret")
model = "unsloth/Qwen3.5-0.8B-GGUF:BF16"
tools = [{"type": "function", "name": "read_file",
          "description": "Read a file in the client workspace",
          "parameters": {"type": "object", "properties": {
              "path": {"type": "string"}}, "required": ["path"]},
          "strict": False}]
history = [{"role": "user", "content": "Read src/main.rs and summarize it."}]

for _ in range(8):
    response = client.responses.create(
        model=model, input=history, tools=tools, store=False,
        temperature=0, max_output_tokens=512)
    history.extend(item.model_dump(exclude_none=True) for item in response.output)
    calls = [item for item in response.output if item.type == "function_call"]
    if not calls:
        print(response.output_text)
        break
    for call in calls:
        import json
        args = json.loads(call.arguments)
        result = read_file(args["path"])  # Your client's implementation and policy.
        history.append({"type": "function_call_output",
                        "call_id": call.call_id, "output": result})
```

Custom tools (`{"type":"custom","name":"apply_patch",...}`) are also supported.
They are presented to Qwen as a function with one string `input` parameter, then
translated back to `custom_tool_call` items containing free-form `input`. Return
results as `custom_tool_call_output` with the matching `call_id`. Text formats
and Lark/regex grammar descriptions are accepted as prompt guidance; the server
does not enforce a grammar during decoding. Tools execute on the client.
Hosted tools such as web search and code interpreter are rejected.

### Reasoning and client configuration

`reasoning.effort="none"` disables thinking; `minimal`, `low`, `medium`, `high`,
and `xhigh` enable it. These levels toggle Qwen's thinking mode; they do not set
separate compute budgets. `chat_template_kwargs.enable_thinking` remains
available; conflicting settings are rejected. The default is off.

Qwen's visible reasoning is returned as plaintext `reasoning` output items with
`summary_text` parts and corresponding summary streaming events. Summary options
are accepted, but the returned text is the model's reasoning, without a separate
summarization pass. Replay these items alongside messages and calls. An
`include=["reasoning.encrypted_content"]` hint is accepted for client
compatibility, but no encrypted payload is produced; opaque encrypted reasoning
from another service cannot be replayed here. `text.verbosity` is supported as a
concise/normal/detailed prompt instruction. `text.format` accepts plain text and
JSON Schema outputs, as described below.

### JSON Schema answers

Responses accepts `text.format.type="json_schema"`, including Codex's strict
conversation-title request:

```json
{"text":{"format":{"type":"json_schema","strict":true,
 "schema":{"type":"object","properties":{"title":{"type":"string",
 "minLength":1,"maxLength":36}},"required":["title"],
 "additionalProperties":false},"name":"codex_output_schema"}}}
```

The schema is included in the model's instructions. Complete answers are parsed
as JSON and validated against it, including required fields, additional
properties, nested types, and string length bounds. Invalid schemas return `400`
before inference. Schemas must be self-contained; internal references are
supported, and external network/file references are never fetched.

This uses **prompt guidance and server-side validation**, rather than token-level
constrained decoding. Structured responses, including reasoning, are buffered
while candidates are validated. A nonconforming completed answer returns an error
(`500` for JSON, `response.failed` for SSE), rather than an invalid successful
response. On schema failure the server makes up to two corrective attempts,
sharing the original output-token budget; invalid candidates are never emitted.
Usage includes tokens from all attempts. Token exhaustion still
returns `response.incomplete` and may contain partial JSON. On a tool-call turn,
commentary and calls are returned normally; the answer schema applies to the
final text answer. Ordinary plain-text responses keep streaming incrementally.

The supplied schema format is preserved in the response, and OpenAI SDK
`responses.parse(..., text_format=YourPydanticModel)` requests are supported.

For a Codex custom HTTP provider, use the Responses wire protocol, for example:

```toml
model_provider = "catena"
model = "unsloth/Qwen3.6-35B-A3B-GGUF:BF16"

[model_providers.catena]
name = "Catena"
base_url = "http://SERVER:8000/v1"
wire_api = "responses"
env_key = "CATENA_API_KEY"
requires_openai_auth = false
```

Set the client's `CATENA_API_KEY` to the server's key. Clients must use stateless
HTTP replay and supported text/function/custom tool inputs. Provider configuration
does not enable unsupported hosted tools, token-level grammar constraints,
WebSockets, server-side continuation, or encrypted reasoning. The HTTP protocol
and tool loops are tested with the OpenAI Python SDK; full Codex client behavior
has not been qualified.

Images and legacy `/v1/completions` remain unsupported.
See the official [Responses streaming event reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)
and [gateway requirements](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility).
