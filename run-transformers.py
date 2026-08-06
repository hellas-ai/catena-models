#!/usr/bin/env python3
"""Run greedy BF16 generation from the local Qwen3 base checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy Qwen3-30B-A3B-Base generation through Transformers"
    )
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/Qwen3-30B-A3B-Base")
    )
    parser.add_argument("-n", "--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="PyTorch device; ROCm builds use the 'cuda' device name",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rate(count: int, seconds: float) -> float:
    return float("inf") if seconds == 0 else count / seconds


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm PyTorch is not available in this environment")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(device)
    load_seconds = time.perf_counter() - load_started

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    if input_ids.shape[1] == 0:
        raise SystemExit("the prompt encoded to an empty token prefix")

    print(
        f"loaded Transformers weights on {device} as {model.dtype} "
        f"in {load_seconds:.3f}s",
        file=sys.stderr,
    )
    print(f"prompt token IDs: {input_ids[0].tolist()}", file=sys.stderr)

    generated: list[int] = []
    decode_seconds = 0.0
    generation_started = time.perf_counter()
    with torch.inference_mode():
        synchronize(device)
        prefill_started = time.perf_counter()
        output = model(input_ids=input_ids, use_cache=True)
        synchronize(device)
        prefill_seconds = time.perf_counter() - prefill_started
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values

        for step in range(args.max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id or step + 1 == args.max_new_tokens:
                break

            synchronize(device)
            decode_started = time.perf_counter()
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            synchronize(device)
            decode_seconds += time.perf_counter() - decode_started
            logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values

    generation_seconds = time.perf_counter() - generation_started

    print(
        args.prompt
        + tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    print(f"generated token IDs: {generated}", file=sys.stderr)
    print(
        f"prompt eval: {input_ids.shape[1]} token(s) in {prefill_seconds:.4f}s "
        f"({rate(input_ids.shape[1], prefill_seconds):.2f} tok/s)",
        file=sys.stderr,
    )
    decode_count = max(len(generated) - 1, 0)
    if decode_count:
        print(
            f"cached eval: {decode_count} token(s) in {decode_seconds:.4f}s "
            f"({rate(decode_count, decode_seconds):.2f} tok/s)",
            file=sys.stderr,
        )
    print(
        f"generated {len(generated)} token(s) in {generation_seconds:.4f}s "
        f"({rate(len(generated), generation_seconds):.2f} tok/s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
