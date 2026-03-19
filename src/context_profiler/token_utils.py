"""Token counting utilities using tiktoken."""

from __future__ import annotations

import tiktoken

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding (used by GPT-4, Claude tokenizers are similar)."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))
