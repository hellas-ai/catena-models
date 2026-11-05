# Serving Qwen

The Qwen application can keep a model loaded and serve remote text inference
through the OpenAI Chat Completions protocol:

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
one-shot mode. Startup loads the tokenizer, embedded GGUF chat template, compiled
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
  The default is the smaller of 512 and the remaining context. Explicit output
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

Each request creates a fresh GPU session. Send the full conversation on every
turn, including prior tool calls and their matching results. GPU inference is
serialized on one owning thread; up to eight requests can wait. A full queue
returns `429`. Dropping a request cancels inference between GPU operations and
releases its session. HTTP model/health requests remain responsive during
inference. Request bodies are limited to 8 MiB. Ctrl-C shuts the server down.

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
sampler. `strict=true`, forced/required tool choices, non-function tools,
structured-output modes, logprobs, and non-neutral logits/sampling controls are
rejected explicitly. A malformed generated tool call returns an error rather
than exposing internal tool markup as answer text.

Images, legacy `/v1/completions`, `/v1/responses`, and persistent conversation
storage are not implemented. **Codex compatibility is deferred**: Codex requires
the Responses protocol, including its tool-call/result and streaming events.
A working Chat Completions endpoint alone does not establish that compatibility.
See the [official gateway requirements](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility).
