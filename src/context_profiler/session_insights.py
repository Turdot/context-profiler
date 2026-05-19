"""Session-level context insights built on deterministic block evidence."""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from context_profiler.analyzers.content_repeat import _jaccard_similarity, _ngram_set
from context_profiler.context_diff import ContextBlock, _blocks_for_request
from context_profiler.models import BlockType, Session

_MAX_ITEMS = 20
_CARRYOVER_MIN_TOKENS = 1_000
_PROPAGATION_MIN_TOKENS = 500
_SPAWN_SIMILARITY = 0.65
_MAX_PROPAGATION_LINKS = 120
_ARTIFACT_DUPLICATION_MIN_REDUNDANT_TOKENS = 500
_BUDGET_THRESHOLDS = (32_000, 64_000, 128_000, 200_000)
_BUDGET_PRESSURE_RATIO = 0.8
_COMPRESSION_DROP_MIN_TOKENS = 5_000
_COMPRESSION_DROP_RATIO = 0.15


def analyze_session_insights(session: Session | None) -> dict[str, Any]:
    """Summarize session-level carryover, budget, and artifact lifecycle signals."""
    if session is None or not session.requests:
        return {
            "carryover_hotspots": [],
            "budget_events": [],
            "artifact_lifecycles": [],
            "artifact_duplications": [],
            "propagation": {"nodes": [], "links": []},
            "hints": [],
        }

    blocks_by_request = [_blocks_for_request(req) for req in session.requests]
    carryover = _carryover_hotspots(blocks_by_request)
    budget_events = _budget_events(session)
    artifacts = _artifact_lifecycles(blocks_by_request)
    artifact_duplications = _artifact_duplications(session)
    propagation = _propagation_graph(blocks_by_request)
    hints = _build_hints(carryover, budget_events, artifacts, artifact_duplications)

    return {
        "carryover_hotspots": carryover,
        "budget_events": budget_events,
        "artifact_lifecycles": artifacts,
        "artifact_duplications": artifact_duplications,
        "propagation": propagation,
        "hints": hints,
    }


def _carryover_hotspots(blocks_by_request: list[list[ContextBlock]]) -> list[dict[str, Any]]:
    occurrences: dict[str, list[ContextBlock]] = defaultdict(list)
    for blocks in blocks_by_request:
        seen_hashes: set[str] = set()
        for block in blocks:
            if block.tokens <= 0 or block.hash in seen_hashes:
                continue
            seen_hashes.add(block.hash)
            occurrences[block.hash].append(block)

    hotspots = []
    for content_hash, blocks in occurrences.items():
        request_indices = sorted({block.request_index for block in blocks})
        if len(request_indices) < 2:
            continue
        first = min(blocks, key=lambda block: block.request_index)
        if first.role == "system":
            continue
        carried_blocks = [block for block in blocks if block.request_index != first.request_index]
        carried_tokens = sum(block.tokens for block in carried_blocks)
        if carried_tokens < _CARRYOVER_MIN_TOKENS:
            continue
        artifact_key = first.artifact_key
        if artifact_key and _is_external_asset_artifact(artifact_key):
            artifact_key = None
        hotspots.append({
            "hash": content_hash,
            "first_request_index": first.request_index,
            "last_request_index": request_indices[-1],
            "request_indices": request_indices[:20],
            "carried_request_count": len(request_indices) - 1,
            "block_tokens": first.tokens,
            "carried_tokens": carried_tokens,
            "kind": first.kind,
            "role": first.role,
            "tool_name": first.tool_name,
            "artifact_key": artifact_key,
            "source_block_id": first.id,
            "label": _flow_label(first),
            "preview": first.preview,
        })

    return sorted(hotspots, key=lambda item: item["carried_tokens"], reverse=True)[:_MAX_ITEMS]


def _budget_events(session: Session) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_total = 0
    for req in session.requests:
        total = req.total_input_tokens
        threshold = _next_budget_threshold(total)
        if threshold and total >= threshold * _BUDGET_PRESSURE_RATIO:
            events.append({
                "type": "budget_pressure",
                "request_index": req.request_index,
                "total_tokens": total,
                "threshold": threshold,
                "ratio": total / threshold,
                "reason": "Visible context is approaching a common model context budget.",
            })

        drop = previous_total - total
        if previous_total and drop >= _COMPRESSION_DROP_MIN_TOKENS and drop / previous_total >= _COMPRESSION_DROP_RATIO:
            events.append({
                "type": "compression_opportunity",
                "request_index": req.request_index,
                "previous_total_tokens": previous_total,
                "total_tokens": total,
                "dropped_tokens": drop,
                "drop_ratio": drop / previous_total,
                "reason": "Context dropped substantially here; similar earlier growth may be compressible.",
            })
        previous_total = total
    return events[:_MAX_ITEMS]


def _next_budget_threshold(total_tokens: int) -> int | None:
    for threshold in _BUDGET_THRESHOLDS:
        if total_tokens <= threshold:
            return threshold
    return _BUDGET_THRESHOLDS[-1]


def _artifact_lifecycles(blocks_by_request: list[list[ContextBlock]]) -> list[dict[str, Any]]:
    occurrences: dict[str, list[ContextBlock]] = defaultdict(list)
    for blocks in blocks_by_request:
        seen_in_request: set[str] = set()
        for block in blocks:
            if not block.artifact_key:
                continue
            if _is_external_asset_artifact(block.artifact_key):
                continue
            key = f"{block.artifact_key}:{block.request_index}"
            if key in seen_in_request:
                continue
            seen_in_request.add(key)
            occurrences[block.artifact_key].append(block)

    lifecycles = []
    for artifact_key, blocks in occurrences.items():
        request_indices = sorted({block.request_index for block in blocks})
        if len(request_indices) < 2:
            continue
        tool_names = sorted({block.tool_name for block in blocks if block.tool_name})
        first = min(blocks, key=lambda block: block.request_index)
        lifecycles.append({
            "artifact_key": artifact_key,
            "request_indices": request_indices[:20],
            "occurrences": len(request_indices),
            "tokens": sum(block.tokens for block in blocks),
            "tool_names": tool_names,
            "first_request_index": request_indices[0],
            "last_request_index": request_indices[-1],
            "source_block_id": first.id,
            "label": f"artifact: {artifact_key}",
        })

    return sorted(lifecycles, key=lambda item: (item["occurrences"], item["tokens"]), reverse=True)[:_MAX_ITEMS]


def _artifact_duplications(session: Session) -> list[dict[str, Any]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for req in session.requests:
        for message in req.messages:
            for block_index, block in enumerate(message.blocks):
                if block.block_type != BlockType.TOOL_RESULT or block.token_count <= 0:
                    continue
                artifact_key = _artifact_identity_from_text(block.text)
                if not artifact_key:
                    continue
                occurrences[artifact_key].append({
                    "block_id": f"r{req.request_index}:m{message.index}:b{block_index}",
                    "request_index": req.request_index,
                    "message_index": message.index,
                    "block_index": block_index,
                    "tool_name": block.tool_name,
                    "tokens": block.token_count,
                    "text": block.text,
                    "preview": block.text[:240],
                })

    duplications = []
    for artifact_key, blocks in occurrences.items():
        request_indices = sorted({block["request_index"] for block in blocks})
        if len(blocks) < 2:
            continue
        largest_tokens = max(block["tokens"] for block in blocks)
        redundant_tokens = sum(block["tokens"] for block in blocks) - largest_tokens
        if redundant_tokens < _ARTIFACT_DUPLICATION_MIN_REDUNDANT_TOKENS:
            continue
        base = max(blocks, key=lambda block: block["tokens"])
        base_ngrams = _ngram_set(base["text"])
        similarities = [
            _jaccard_similarity(base_ngrams, _ngram_set(block["text"]))
            for block in blocks
            if block is not base
        ]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        tools = sorted({block["tool_name"] for block in blocks if block["tool_name"]})
        duplications.append({
            "artifact_key": artifact_key,
            "occurrences": len(blocks),
            "request_indices": request_indices[:20],
            "tools": tools,
            "total_tokens": sum(block["tokens"] for block in blocks),
            "redundant_tokens": redundant_tokens,
            "avg_similarity": round(avg_similarity, 3),
            "source_block_id": min(blocks, key=lambda block: block["request_index"])["block_id"],
            "blocks": [
                {
                    "block_id": block["block_id"],
                    "request_index": block["request_index"],
                    "tool_name": block["tool_name"],
                    "tokens": block["tokens"],
                    "preview": block["preview"],
                }
                for block in sorted(blocks, key=lambda block: (block["request_index"], block["block_id"]))
            ][:20],
        })

    return sorted(duplications, key=lambda item: item["redundant_tokens"], reverse=True)[:_MAX_ITEMS]


def _propagation_graph(blocks_by_request: list[list[ContextBlock]]) -> dict[str, list[dict[str, Any]]]:
    blocks = [
        block
        for request_blocks in blocks_by_request
        for block in request_blocks
        if _eligible_propagation_block(block)
    ]
    links: list[dict[str, Any]] = []
    exact_groups: dict[str, list[ContextBlock]] = defaultdict(list)
    for block in blocks:
        exact_groups[block.hash].append(block)

    for group in exact_groups.values():
        request_indices = sorted({block.request_index for block in group})
        if len(request_indices) < 2:
            continue
        first = min(group, key=lambda block: block.request_index)
        last = max(group, key=lambda block: block.request_index)
        links.append(_propagation_link(first, last, "carry", 1.0, repeats=len(request_indices) - 1))

    for i, source in enumerate(blocks):
        source_ngrams = None
        for target in blocks[i + 1:]:
            if target.request_index <= source.request_index:
                continue
            if source.hash == target.hash:
                continue

            if not _related_for_spawn(source, target):
                continue
            source_ngrams = source_ngrams or _ngram_set(source.preview)
            similarity = _jaccard_similarity(source_ngrams, _ngram_set(target.preview))
            if similarity < _SPAWN_SIMILARITY:
                continue
            links.append(_propagation_link(source, target, "spawn", similarity))

    carry_links = sorted(
        [link for link in links if link["type"] == "carry"],
        key=lambda link: link["value"],
        reverse=True,
    )[: _MAX_PROPAGATION_LINKS // 2]
    accumulation_nodes, accumulation_links = _accumulation_links(links)
    spawn_links = sorted(
        [link for link in links if link["type"] == "spawn"],
        key=lambda link: link["value"],
        reverse=True,
    )[: _MAX_PROPAGATION_LINKS // 2]
    accumulation_budget = max(0, _MAX_PROPAGATION_LINKS - len(carry_links) - len(spawn_links))
    accumulation_links = sorted(
        accumulation_links,
        key=lambda link: link["value"],
        reverse=True,
    )[:accumulation_budget]
    links = sorted(
        carry_links + spawn_links + accumulation_links,
        key=lambda link: link["value"],
        reverse=True,
    )
    node_ids = {link["source"] for link in links} | {link["target"] for link in links}
    block_by_id = {block.id: block for block in blocks if block.id in node_ids}
    nodes = [
        {
            "id": block.id,
            "name": f"#{block.request_index} {_flow_label(block)}",
            "request_index": block.request_index,
            "block_id": block.id,
            "message_index": _message_index(block.id),
            "block_index": _block_index(block.id),
            "tool_name": block.tool_name,
            "artifact_key": None
            if block.artifact_key and _is_external_asset_artifact(block.artifact_key)
            else block.artifact_key,
            "kind": block.kind,
            "tokens": block.tokens,
            "preview": block.preview,
        }
        for block in sorted(block_by_id.values(), key=lambda item: (item.request_index, item.id))
    ]
    nodes.extend(node for node in accumulation_nodes if node["id"] in node_ids)
    return {"nodes": nodes, "links": links}


def _accumulation_links(links: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        if link["type"] != "spawn":
            continue
        groups[(link["source"], link["target_request_index"])].append(link)

    nodes: list[dict[str, Any]] = []
    acc_links: list[dict[str, Any]] = []
    for (source_id, request_index), group in groups.items():
        member_ids = sorted({link["target_block_id"] for link in group})
        if len(member_ids) < 2:
            continue
        first = max(group, key=lambda link: link["value"])
        node_id = f"acc:{source_id}:r{request_index}"
        total_value = sum(link["value"] for link in group)
        avg_similarity = sum(link["similarity"] for link in group) / len(group)
        nodes.append({
            "id": node_id,
            "name": f"#{request_index} accumulation · {len(member_ids)} copies",
            "request_index": request_index,
            "block_id": member_ids[0],
            "message_index": min(_message_index(block_id) for block_id in member_ids),
            "block_index": min(_block_index(block_id) for block_id in member_ids),
            "kind": "accumulation",
            "tokens": total_value,
            "copies": len(member_ids),
            "member_block_ids": member_ids,
            "source_block_id": source_id,
            "label": f"{len(member_ids)} copies / {total_value:,} propagated tokens",
        })
        acc_links.append({
            "source": source_id,
            "target": node_id,
            "type": "accumulation",
            "value": total_value,
            "similarity": round(avg_similarity, 3),
            "repeats": len(member_ids),
            "source_request_index": first["source_request_index"],
            "target_request_index": request_index,
            "source_block_id": source_id,
            "target_block_id": member_ids[0],
            "member_block_ids": member_ids,
            "source_label": first["source_label"],
            "target_label": f"{len(member_ids)} accumulated copies",
            "tool_name": first.get("tool_name"),
            "artifact_key": first.get("artifact_key"),
        })

    return nodes, acc_links


def _eligible_propagation_block(block: ContextBlock) -> bool:
    return block.role != "system" and block.tokens >= _PROPAGATION_MIN_TOKENS


def _related_for_spawn(source: ContextBlock, target: ContextBlock) -> bool:
    if source.artifact_key and target.artifact_key and source.artifact_key == target.artifact_key:
        return not _is_external_asset_artifact(source.artifact_key)
    if source.tool_name and target.tool_name and source.tool_name == target.tool_name:
        return True
    if source.kind == "tool_result" and target.kind == "tool_use":
        return True
    return False


def _propagation_link(
    source: ContextBlock,
    target: ContextBlock,
    link_type: str,
    similarity: float,
    repeats: int = 1,
) -> dict[str, Any]:
    propagated_tokens = int(min(source.tokens, target.tokens) * similarity * max(1, repeats))
    return {
        "source": source.id,
        "target": target.id,
        "type": link_type,
        "value": propagated_tokens,
        "similarity": round(similarity, 3),
        "repeats": repeats,
        "source_request_index": source.request_index,
        "target_request_index": target.request_index,
        "source_block_id": source.id,
        "target_block_id": target.id,
        "source_label": _flow_label(source),
        "target_label": _flow_label(target),
        "tool_name": target.tool_name or source.tool_name,
        "artifact_key": None
        if source.artifact_key and _is_external_asset_artifact(source.artifact_key)
        else source.artifact_key,
    }


def _message_index(block_id: str) -> int:
    match = re.match(r"r\d+:m(\d+):b\d+", block_id)
    return int(match.group(1)) if match else 0


def _block_index(block_id: str) -> int:
    match = re.match(r"r\d+:m\d+:b(\d+)", block_id)
    return int(match.group(1)) if match else 0


def _build_hints(
    carryover: list[dict[str, Any]],
    budget_events: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    artifact_duplications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for item in carryover[:5]:
        hints.append({
            "type": "token_carryover_hotspot",
            "confidence": "medium",
            "request_index": item["first_request_index"],
            "evidence": item,
            "reason": "A large context block is retained across multiple later requests.",
        })
    for event in budget_events[:8]:
        hints.append({
            "type": event["type"],
            "confidence": "medium",
            "request_index": event["request_index"],
            "evidence": event,
            "reason": event["reason"],
        })
    for item in artifacts[:5]:
        hints.append({
            "type": "possible_artifact_lifecycle_churn",
            "confidence": "medium",
            "request_index": item["first_request_index"],
            "evidence": item,
            "reason": "The same artifact appears across multiple requests in the session.",
        })
    for item in artifact_duplications[:5]:
        hints.append({
            "type": "tool_result_artifact_duplication",
            "confidence": "medium",
            "request_index": item["request_indices"][0],
            "evidence": item,
            "reason": "Multiple tool results return substantially similar content for the same artifact.",
        })
    return hints[:_MAX_ITEMS]


def _flow_label(block: ContextBlock) -> str:
    if block.role == "system":
        return "system prompt"
    if block.tool_name:
        if block.kind == "tool_result":
            return f"tool result: {block.tool_name}"
        if block.kind == "tool_use":
            return f"tool call: {block.tool_name}"
        return f"tool: {block.tool_name}"
    if block.artifact_key:
        return f"artifact: {block.artifact_key}"
    return block.kind


def _is_external_asset_artifact(artifact_key: str) -> bool:
    lowered = artifact_key.lower()
    return (
        lowered.startswith("//")
        or lowered.startswith("http://")
        or lowered.startswith("https://")
        or "cdn" in lowered
    )


def _artifact_identity_from_text(text: str) -> str | None:
    stripped = text.strip()
    parsed: Any | None = None
    if stripped and stripped[0] in "{[":
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None

    if parsed is not None:
        value = _find_first_key(parsed, ("component_id", "interactive_component_id", "artifact_id"))
        if isinstance(value, str) and value:
            return f"component:{value}"

    match = re.search(r"[\w./-]+\.(?:jsonl|ipynb|tsx|jsx|yaml|yml|json|html|css|txt|md|py|ts|js)", text)
    if match:
        artifact = match.group(0)
        if not _is_external_asset_artifact(artifact):
            return artifact
    return None


def _find_first_key(value: Any, keys: tuple[str, ...]) -> Any | None:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for child in value.values():
            found = _find_first_key(child, keys)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_first_key(child, keys)
            if found is not None:
                return found
    return None
