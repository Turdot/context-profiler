"""Turn-to-turn context diff evidence for agent sessions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from context_profiler.models import APIRequest, BlockType, ContentBlock, Message, Session

_PREVIEW_LIMIT = 240
_LARGE_ADDITION_TOKENS = 2_000
_RATIO_ADDITION_MIN_TOKENS = 500
_LARGE_ADDITION_RATIO = 0.25
_HIGH_TOOL_USE_MIN_TOKENS = 1_000
_HIGH_TOOL_USE_RATIO = 0.5
_MAX_HINTS = 40


@dataclass(frozen=True)
class ContextBlock:
    id: str
    request_index: int
    message_index: int
    block_index: int
    role: str
    kind: str
    tokens: int
    hash: str
    preview: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    artifact_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_context_diff(session: Session | None) -> dict[str, Any]:
    if session is None or len(session.requests) < 2:
        return {
            "diff_summary": {
                "transition_count": 0,
                "max_added_tokens": 0,
                "max_removed_tokens": 0,
                "total_added_tokens": 0,
                "total_removed_tokens": 0,
            },
            "diff_hints": [],
            "transitions": [],
        }

    transitions = []
    all_blocks_by_request = [_blocks_for_request(req) for req in session.requests]
    for prev_req, curr_req, prev_blocks, curr_blocks in zip(
        session.requests,
        session.requests[1:],
        all_blocks_by_request,
        all_blocks_by_request[1:],
    ):
        transitions.append(_diff_transition(prev_req, curr_req, prev_blocks, curr_blocks))

    hints = _build_hints(transitions)
    summary = {
        "transition_count": len(transitions),
        "max_added_tokens": max((t["added_tokens"] for t in transitions), default=0),
        "max_removed_tokens": max((t["removed_tokens"] for t in transitions), default=0),
        "total_added_tokens": sum(t["added_tokens"] for t in transitions),
        "total_removed_tokens": sum(t["removed_tokens"] for t in transitions),
        "max_delta_tokens": max((t["delta_tokens"] for t in transitions), default=0),
        "min_delta_tokens": min((t["delta_tokens"] for t in transitions), default=0),
    }
    return {
        "diff_summary": summary,
        "diff_hints": hints,
        "transitions": transitions,
    }


def _blocks_for_request(request: APIRequest) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    for message in request.messages:
        for block_index, block in enumerate(message.blocks):
            kind = block.block_type.value
            artifact_key = _artifact_key(block)
            content_hash = _content_hash(block.text, kind, block.tool_name, artifact_key)
            blocks.append(ContextBlock(
                id=f"r{request.request_index}:m{message.index}:b{block_index}",
                request_index=request.request_index,
                message_index=message.index,
                block_index=block_index,
                role=message.role.value,
                kind=kind,
                tokens=block.token_count,
                hash=content_hash,
                preview=block.text[:_PREVIEW_LIMIT],
                tool_name=block.tool_name,
                tool_call_id=block.tool_call_id,
                artifact_key=artifact_key,
            ))
    return blocks


def _diff_transition(
    prev_req: APIRequest,
    curr_req: APIRequest,
    prev_blocks: list[ContextBlock],
    curr_blocks: list[ContextBlock],
) -> dict[str, Any]:
    prev_counts = Counter(block.hash for block in prev_blocks)
    curr_counts = Counter(block.hash for block in curr_blocks)
    retained_hashes = prev_counts & curr_counts

    retained_remaining = Counter(retained_hashes)
    added_blocks = []
    retained_tokens = 0
    for block in curr_blocks:
        if retained_remaining[block.hash] > 0:
            retained_remaining[block.hash] -= 1
            retained_tokens += block.tokens
        else:
            added_blocks.append(block)

    added_remaining = Counter(curr_counts - prev_counts)
    removed_blocks = []
    for block in prev_blocks:
        if added_remaining[block.hash] > 0:
            continue
        if curr_counts[block.hash] < prev_counts[block.hash]:
            removed_blocks.append(block)
            prev_counts[block.hash] -= 1

    prev_total = prev_req.total_input_tokens
    curr_total = curr_req.total_input_tokens
    added_tokens = sum(block.tokens for block in added_blocks)
    removed_tokens = sum(block.tokens for block in removed_blocks)
    top_added = _top_blocks(added_blocks)
    top_removed = _top_blocks(removed_blocks)
    by_added_kind = _sum_by(added_blocks, "kind")
    by_removed_kind = _sum_by(removed_blocks, "kind")

    return {
        "request_index": curr_req.request_index,
        "prev_request_index": prev_req.request_index,
        "prev_total_tokens": prev_total,
        "total_tokens": curr_total,
        "delta_tokens": curr_total - prev_total,
        "growth_ratio": (curr_total - prev_total) / prev_total if prev_total else None,
        "added_tokens": added_tokens,
        "removed_tokens": removed_tokens,
        "retained_tokens": retained_tokens,
        "added_by_kind": by_added_kind,
        "removed_by_kind": by_removed_kind,
        "top_added": top_added,
        "top_removed": top_removed,
        "top_added_tool": _top_tool(added_blocks),
        "top_added_artifact": _top_artifact(added_blocks),
    }


def _build_hints(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for transition in transitions:
        growth_ratio = transition["growth_ratio"] or 0
        if (
            transition["added_tokens"] >= _LARGE_ADDITION_TOKENS
            or (
                transition["added_tokens"] >= _RATIO_ADDITION_MIN_TOKENS
                and growth_ratio >= _LARGE_ADDITION_RATIO
            )
        ):
            hints.append({
                "type": "large_addition",
                "confidence": "medium",
                "request_index": transition["request_index"],
                "evidence": {
                    "added_tokens": transition["added_tokens"],
                    "delta_tokens": transition["delta_tokens"],
                    "growth_ratio": transition["growth_ratio"],
                    "top_added_tool": transition["top_added_tool"],
                    "top_added_artifact": transition["top_added_artifact"],
                },
                "reason": "This turn added substantially more visible context than the previous turn.",
            })

        tool_use_tokens = transition["added_by_kind"].get(BlockType.TOOL_USE.value, 0)
        if (
            tool_use_tokens >= _HIGH_TOOL_USE_MIN_TOKENS
            and transition["added_tokens"]
            and tool_use_tokens / transition["added_tokens"] >= _HIGH_TOOL_USE_RATIO
        ):
            hints.append({
                "type": "high_tool_use_addition",
                "confidence": "medium",
                "request_index": transition["request_index"],
                "evidence": {
                    "tool_use_added_tokens": tool_use_tokens,
                    "added_tokens": transition["added_tokens"],
                    "top_added_tool": transition["top_added_tool"],
                },
                "reason": "Most newly added visible context in this turn came from tool inputs.",
            })

    hints.extend(_artifact_churn_hints(transitions))
    return hints[:_MAX_HINTS]


def _artifact_churn_hints(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    occurrences: dict[str, list[ContextBlock]] = defaultdict(list)
    for transition in transitions:
        seen_in_request = set()
        for block_data in transition["top_added"]:
            block = ContextBlock(**block_data)
            if not block.artifact_key or block.kind != BlockType.TOOL_USE.value:
                continue
            key = (block.artifact_key, block.request_index)
            if key in seen_in_request:
                continue
            seen_in_request.add(key)
            occurrences[block.artifact_key].append(block)

    hints = []
    for artifact_key, blocks in occurrences.items():
        request_indices = sorted({block.request_index for block in blocks})
        if len(request_indices) < 2:
            continue
        hints.append({
            "type": "possible_artifact_churn",
            "confidence": "medium",
            "evidence": {
                "artifact_key": artifact_key,
                "occurrences": len(request_indices),
                "request_indices": request_indices[:10],
                "tokens": sum(block.tokens for block in blocks),
            },
            "reason": "The same artifact appears in multiple tool inputs across turns.",
        })
    return hints


def _top_blocks(blocks: list[ContextBlock], limit: int = 5) -> list[dict[str, Any]]:
    return [
        block.to_dict()
        for block in sorted(blocks, key=lambda b: b.tokens, reverse=True)[:limit]
    ]


def _top_tool(blocks: list[ContextBlock]) -> str | None:
    totals: dict[str, int] = defaultdict(int)
    for block in blocks:
        if block.tool_name:
            totals[block.tool_name] += block.tokens
    if not totals:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def _top_artifact(blocks: list[ContextBlock]) -> str | None:
    totals: dict[str, int] = defaultdict(int)
    for block in blocks:
        if block.artifact_key:
            totals[block.artifact_key] += block.tokens
    if not totals:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def _sum_by(blocks: list[ContextBlock], field: str) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for block in blocks:
        totals[str(getattr(block, field))] += block.tokens
    return dict(totals)


def _content_hash(text: str, kind: str, tool_name: str | None, artifact_key: str | None) -> str:
    payload = {
        "kind": kind,
        "tool_name": tool_name,
        "artifact_key": artifact_key,
        "text": _normalize_text(text),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)


def _artifact_key(block: ContentBlock) -> str | None:
    if block.tool_input:
        direct = _artifact_from_tool_input(block.tool_input)
        if direct:
            return direct
    if block.tool_name == "ApplyPatch":
        patch_target = _artifact_from_patch(block.text)
        if patch_target:
            return patch_target
    return _artifact_from_text(block.text)


def _artifact_from_tool_input(tool_input: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "target_file", "target_notebook"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    patch = tool_input.get("patch")
    if isinstance(patch, str):
        return _artifact_from_patch(patch)
    command = tool_input.get("command")
    if isinstance(command, str):
        return _artifact_from_text(command)
    return None


def _artifact_from_patch(text: str) -> str | None:
    match = re.search(r"^\*\*\* (?:Update|Add) File: (.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _artifact_from_text(text: str) -> str | None:
    match = re.search(
        r"[\w./-]+\.(?:jsonl|ipynb|tsx|jsx|yaml|yml|json|html|css|txt|md|py|ts|js)",
        text,
    )
    if match:
        return match.group(0)
    return None
