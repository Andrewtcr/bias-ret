"""OpenAI batch judge primitives shared across experiments.

Provides token estimates and response parsing. Experiment scripts construct
requests and manage batch submission, polling, and downloaded outputs.
"""
from __future__ import annotations

import json
import math
from typing import Any


def try_tiktoken_encoding(model_name: str):
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def estimate_batch_tokens(records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    enc = try_tiktoken_encoding(model_name)
    input_tokens = 0
    for record in records:
        body_text = json.dumps(record["body"], ensure_ascii=True, sort_keys=True)
        if enc is None:
            input_tokens += math.ceil(len(body_text) / 4)
        else:
            input_tokens += len(enc.encode(body_text))
    return {
        "input_tokens": input_tokens,
        "estimation_method": "tiktoken" if enc is not None else "chars_div_4",
    }


def extract_response_text(response_body: dict[str, Any]) -> str:
    if not isinstance(response_body, dict):
        return ""

    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = response_body.get("output") or []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


__all__ = [
    "estimate_batch_tokens",
    "extract_response_text",
    "try_tiktoken_encoding",
]
