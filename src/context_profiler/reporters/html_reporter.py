"""Interactive HTML report generator.

Builds a self-contained HTML file with:
- Icicle chart (D3 partition layout) showing token distribution
- Split-pane detail panel with content preview
- Search and color scheme toggling
- Intra-request content similarity heatmap
- Session timeline (stacked area chart)

The icicle tree mirrors the raw JSON structure of the API request,
recursively profiling tokens at every nesting level.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from context_profiler.models import APIRequest, BlockType, Session
from context_profiler.profiler import ProfileResult
from context_profiler.token_utils import count_tokens

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "report.html"

CONTENT_PREVIEW_LIMIT = 3000
MAX_TREE_DEPTH = 12


def _try_parse_json(s: str) -> Any | None:
    """Try to parse a string as JSON. Returns parsed object or None."""
    s = s.strip()
    if not s or (s[0] not in ('{', '[', '"')):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def _infer_node_type(key: str, value: Any, parent_type: str) -> str:
    """Infer a semantic type for color-coding in the icicle chart."""
    if parent_type == "root" and key == "tools":
        return "tool_defs"
    if parent_type == "root" and key == "messages":
        return "messages"

    if isinstance(value, dict):
        role = value.get("role")
        if role == "system":
            return "msg_system"
        if role == "user":
            return "msg_user"
        if role == "assistant":
            return "msg_assistant"
        if role == "tool":
            return "msg_tool"
        if "function" in value and "name" in value.get("function", {}):
            return "tool_use"

    if parent_type == "tool_defs":
        return "tool_def"
    if key in ("tool_calls", "function"):
        return "tool_use"
    if key == "arguments":
        return "tool_use"
    if key == "content" and parent_type == "msg_tool":
        return "tool_result"
    if key == "content":
        return "text"

    return "field"


def _profile_node(
    obj: Any,
    name: str,
    node_type: str = "field",
    depth: int = 0,
) -> dict[str, Any]:
    """Recursively profile any JSON value, building a tree with token counts.

    Every JSON key at every nesting level gets its own node.
    JSON strings that contain parseable JSON are expanded recursively.
    """
    node: dict[str, Any] = {
        "name": name,
        "type": node_type,
        "children": [],
        "content": "",
    }

    if depth > MAX_TREE_DEPTH:
        text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
        node["tokens"] = count_tokens(text)
        node["content"] = text[:CONTENT_PREVIEW_LIMIT]
        return node

    if isinstance(obj, dict):
        children = []
        for key, val in obj.items():
            child_type = _infer_node_type(key, val, node_type)
            child = _profile_node(val, key, child_type, depth + 1)
            children.append(child)
        node["children"] = children
        node["tokens"] = sum(c["tokens"] for c in children) if children else 0

        role = obj.get("role")
        if role:
            node["role"] = role

        func = obj.get("function", {})
        if isinstance(func, dict) and func.get("name"):
            node["tool_name"] = func["name"]

        tool_name = obj.get("name")
        if role == "tool" and tool_name:
            node["tool_name"] = tool_name

    elif isinstance(obj, list):
        children = []
        for i, item in enumerate(obj):
            label = _list_item_label(item, i, node_type)
            child_type = _infer_node_type(str(i), item, node_type)
            child = _profile_node(item, label, child_type, depth + 1)
            children.append(child)
        node["children"] = children
        node["tokens"] = sum(c["tokens"] for c in children) if children else 0

    elif isinstance(obj, str):
        parsed = _try_parse_json(obj) if depth < MAX_TREE_DEPTH else None
        if parsed is not None and isinstance(parsed, (dict, list)):
            inner = _profile_node(parsed, name, node_type, depth + 1)
            node["children"] = inner["children"]
            node["tokens"] = inner["tokens"]
            node["content"] = obj[:CONTENT_PREVIEW_LIMIT]
            if inner.get("role"):
                node["role"] = inner["role"]
            if inner.get("tool_name"):
                node["tool_name"] = inner["tool_name"]
        else:
            node["tokens"] = count_tokens(obj)
            node["content"] = obj[:CONTENT_PREVIEW_LIMIT]

    elif obj is None:
        node["tokens"] = 0
        node["content"] = "null"

    else:
        text = str(obj)
        node["tokens"] = count_tokens(text)
        node["content"] = text[:CONTENT_PREVIEW_LIMIT]

    return node


def _list_item_label(item: Any, index: int, parent_type: str) -> str:
    """Generate a readable label for a list item."""
    if isinstance(item, dict):
        role = item.get("role")
        if role:
            tool_calls = item.get("tool_calls", [])
            tool_call_id = item.get("tool_call_id", "")
            name = item.get("name", "")
            if role == "assistant" and tool_calls:
                fn = tool_calls[0].get("function", {}).get("name", "")
                return f"[{index}] assistant → {fn}"
            if role == "tool" and name:
                return f"[{index}] tool: {name}"
            if role == "tool" and tool_call_id:
                return f"[{index}] tool ({tool_call_id[:20]})"
            return f"[{index}] {role}"

        func_name = item.get("function", {}).get("name") if isinstance(item.get("function"), dict) else None
        if func_name:
            return f"[{index}] {func_name}"

        name = item.get("name")
        if name:
            return f"[{index}] {name}"

        item_type = item.get("type")
        if item_type:
            return f"[{index}] {item_type}"

    return f"[{index}]"


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize a value for hashing: re-parse JSON strings so formatting doesn't matter."""
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed  # will be re-serialized with consistent formatting by json.dumps
        except (json.JSONDecodeError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    return obj


def _hash_message(msg: dict) -> str:
    """Hash a raw message dict for diff comparison.

    Normalizes JSON string fields (like tool_calls arguments) so that
    semantically identical content with different formatting hashes the same.
    """
    raw = {k: msg.get(k) for k in ("role", "content", "tool_calls", "tool_call_id", "name")
           if msg.get(k) is not None}
    normalized = _normalize_for_hash(raw)
    key_parts = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key_parts.encode()).hexdigest()


def _compute_diff_statuses(
    prev_raw_messages: list[dict] | None,
    curr_raw_messages: list[dict],
) -> tuple[list[str], int, int, list[dict]]:
    """Compute per-message diff status between two consecutive requests.

    Uses LCS (Longest Common Subsequence) to respect message ordering.
    Returns (statuses, removed_count, removed_tokens, removed_messages) where
    statuses[i] is 'unchanged' | 'added' for each message in curr_raw_messages,
    and removed_messages are the actual prev messages no longer present.
    """
    if not prev_raw_messages:
        return ["added"] * len(curr_raw_messages), 0, 0, []

    prev_hashes = [_hash_message(m) for m in prev_raw_messages]
    curr_hashes = [_hash_message(m) for m in curr_raw_messages]

    n, m = len(prev_hashes), len(curr_hashes)

    # Build LCS table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if prev_hashes[i - 1] == curr_hashes[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find which curr messages are in the LCS (unchanged)
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    i, j = n, m
    while i > 0 and j > 0:
        if prev_hashes[i - 1] == curr_hashes[j - 1]:
            matched_prev.add(i - 1)
            matched_curr.add(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    statuses = [
        "unchanged" if j in matched_curr else "added"
        for j in range(m)
    ]

    # Removed = prev messages not matched by LCS
    removed_messages: list[dict] = []
    removed_tokens = 0
    for i in range(n):
        if i not in matched_prev:
            removed_messages.append(prev_raw_messages[i])
            text = json.dumps(prev_raw_messages[i], ensure_ascii=False)
            removed_tokens += count_tokens(text)

    removed_count = len(removed_messages)
    return statuses, removed_count, removed_tokens, removed_messages


def _tag_tree_diff(tree_node: dict, diff_status: str) -> None:
    """Recursively tag a tree node and all descendants with diff_status."""
    tree_node["diff"] = diff_status
    for child in tree_node.get("children", []):
        _tag_tree_diff(child, diff_status)


def _apply_diff_to_tree(
    tree: dict[str, Any],
    diff_statuses: list[str],
    removed_count: int,
    removed_tokens: int,
    removed_messages: list[dict] | None = None,
) -> None:
    """Apply diff statuses to the messages node in an icicle tree.

    Tags each message child and adds a Removed summary node if needed.
    """
    # Find the messages node
    messages_node = None
    for child in tree.get("children", []):
        if child.get("type") == "messages" or child.get("name") == "messages":
            messages_node = child
            break

    if not messages_node:
        return

    msg_children = messages_node.get("children", [])

    # Tag each message node with its diff status
    for i, child in enumerate(msg_children):
        if i < len(diff_statuses):
            _tag_tree_diff(child, diff_statuses[i])
        else:
            _tag_tree_diff(child, "added")

    # Tag non-message children (tools, system) as unchanged by default
    for child in tree.get("children", []):
        if child is not messages_node:
            _tag_tree_diff(child, "unchanged")

    # Tag root
    tree["diff"] = "context"

    # Add Removed summary node at start of messages if any were removed
    if removed_count > 0:
        # Build child nodes for each removed message
        removed_children = []
        for rm in (removed_messages or []):
            label = _list_item_label(rm, len(removed_children), "messages")
            child = _profile_node(rm, label, _infer_node_type("", rm, "messages"))
            _tag_tree_diff(child, "removed")
            removed_children.append(child)

        removed_node = {
            "name": f"Removed [{removed_count} msgs, {removed_tokens:,} tok]",
            "type": "removed_group",
            "diff": "removed",
            "tokens": removed_tokens,
            "children": removed_children,
            "content": f"{removed_count} messages were pruned from the previous request (token budget truncation)",
        }
        messages_node["children"] = [removed_node] + msg_children
        messages_node["tokens"] = messages_node.get("tokens", 0) + removed_tokens

    # Tag the messages node itself
    messages_node["diff"] = "context"


def _split_messages_by_history(
    messages_node: dict[str, Any],
    history_boundary: int,
) -> None:
    """Split a messages node's children into History and New groups.

    history_boundary is the number of messages from the previous request,
    i.e. children[:boundary] are carried-over history, children[boundary:] are new.
    """
    children = messages_node.get("children", [])
    if not children or history_boundary <= 0:
        return

    history_children = children[:history_boundary]
    new_children = children[history_boundary:]

    history_tokens = sum(c.get("tokens", 0) for c in history_children)
    new_tokens = sum(c.get("tokens", 0) for c in new_children)

    history_node = {
        "name": f"History [{len(history_children)} msgs, {history_tokens:,} tok]",
        "type": "history_group",
        "tokens": history_tokens,
        "children": history_children,
        "content": "",
    }

    result = [history_node]
    if new_children:
        new_node = {
            "name": f"New [{len(new_children)} msgs, {new_tokens:,} tok]",
            "type": "new_group",
            "tokens": new_tokens,
            "children": new_children,
            "content": "",
        }
        result.append(new_node)

    messages_node["children"] = result


def _build_tool_analysis(request: APIRequest) -> dict[str, Any]:
    """Build per-tool token analysis from an API request.

    Aggregates definition/call/result tokens grouped by tool name,
    with per-invocation detail for drill-down.
    """
    # Build tool_call_id → tool_name map from TOOL_USE blocks
    call_id_to_name: dict[str, str] = {}
    for msg in request.messages:
        for block in msg.blocks:
            if block.block_type == BlockType.TOOL_USE and block.tool_call_id and block.tool_name:
                call_id_to_name[block.tool_call_id] = block.tool_name

    # Per-tool accumulator
    tool_stats: dict[str, dict[str, Any]] = {}

    def _get_tool(name: str) -> dict[str, Any]:
        if name not in tool_stats:
            tool_stats[name] = {
                "name": name,
                "definition_tokens": 0,
                "call_tokens": 0,
                "result_tokens": 0,
                "call_count": 0,
                "calls": [],
            }
        return tool_stats[name]

    # Aggregate definition tokens from request.tools
    for td in request.tools:
        t = _get_tool(td.name)
        t["definition_tokens"] = td.token_count
        t["raw_definition"] = td.raw_json[:CONTENT_PREVIEW_LIMIT] if td.raw_json else None

    # Track per-invocation pairs: tool_call_id → {call_tokens, result_tokens, ...}
    invocation_map: dict[str, dict[str, Any]] = {}

    # Aggregate call tokens (TOOL_USE blocks)
    for msg in request.messages:
        for block in msg.blocks:
            if block.block_type == BlockType.TOOL_USE and block.tool_name:
                tool = _get_tool(block.tool_name)
                tool["call_tokens"] += block.token_count
                tool["call_count"] += 1
                call_id = block.tool_call_id or f"_anon_{id(block)}"
                # Capture raw call content
                raw_call = None
                if block.tool_input is not None:
                    raw_call = block.tool_input
                elif block.text:
                    raw_call = block.text[:CONTENT_PREVIEW_LIMIT]
                invocation_map[call_id] = {
                    "index": msg.index,
                    "call_tokens": block.token_count,
                    "result_tokens": 0,
                    "tool_name": block.tool_name,
                    "raw_call": raw_call,
                    "raw_result": None,
                }

    # Aggregate result tokens (TOOL_RESULT blocks)
    for msg in request.messages:
        for block in msg.blocks:
            if block.block_type == BlockType.TOOL_RESULT:
                # Resolve tool name: from block directly, or via call_id map
                name = block.tool_name
                if not name and block.tool_call_id:
                    name = call_id_to_name.get(block.tool_call_id)
                if not name:
                    name = "_unknown"

                tool = _get_tool(name)
                tool["result_tokens"] += block.token_count

                # Capture raw result content
                raw_result = block.text[:CONTENT_PREVIEW_LIMIT] if block.text else None

                # Link to invocation
                call_id = block.tool_call_id
                if call_id and call_id in invocation_map:
                    invocation_map[call_id]["result_tokens"] += block.token_count
                    invocation_map[call_id]["raw_result"] = raw_result
                else:
                    # Orphan result — create standalone invocation record
                    inv_id = f"_result_{id(block)}"
                    invocation_map[inv_id] = {
                        "index": msg.index,
                        "call_tokens": 0,
                        "result_tokens": block.token_count,
                        "tool_name": name,
                        "raw_call": None,
                        "raw_result": raw_result,
                    }

    # Attach invocations to their tools and compute totals
    tools_list = []
    for name, stats in tool_stats.items():
        # Gather invocations for this tool
        calls = [
            inv for inv in invocation_map.values()
            if inv["tool_name"] == name
        ]
        calls.sort(key=lambda c: c["call_tokens"] + c["result_tokens"], reverse=True)
        stats["calls"] = [
            {
                "index": c["index"],
                "call_tokens": c["call_tokens"],
                "result_tokens": c["result_tokens"],
                "raw_call": c.get("raw_call"),
                "raw_result": c.get("raw_result"),
            }
            for c in calls
        ]
        stats["total_tokens"] = (
            stats["definition_tokens"] + stats["call_tokens"] + stats["result_tokens"]
        )
        stats["avg_call_tokens"] = (
            round(stats["call_tokens"] / stats["call_count"])
            if stats["call_count"] > 0 else 0
        )
        stats["avg_result_tokens"] = (
            round(stats["result_tokens"] / stats["call_count"])
            if stats["call_count"] > 0 else 0
        )
        tools_list.append(stats)

    # Sort by total tokens descending
    tools_list.sort(key=lambda t: t["total_tokens"], reverse=True)

    return {
        "tools": tools_list,
        "total_context_tokens": request.total_input_tokens,
    }


def _build_raw_icicle_tree(request: APIRequest) -> dict[str, Any]:
    """Build icicle tree from the raw API request JSON, preserving original order."""
    raw = request.raw_input
    if not raw:
        return _build_fallback_tree(request)

    root = _profile_node(raw, "API Request", "root")
    return root


def _build_fallback_tree(request: APIRequest) -> dict[str, Any]:
    """Fallback when raw_input is not available: build from canonical models."""
    root: dict[str, Any] = {
        "name": "API Request",
        "type": "root",
        "tokens": request.total_input_tokens,
        "children": [],
        "content": "",
    }

    if request.tools:
        defs_node: dict[str, Any] = {
            "name": "tools",
            "type": "tool_defs",
            "tokens": request.tool_definition_tokens,
            "children": [],
            "content": "",
        }
        for td in request.tools:
            defs_node["children"].append({
                "name": td.name,
                "type": "tool_def",
                "tokens": td.token_count,
                "children": [],
                "content": td.raw_json[:CONTENT_PREVIEW_LIMIT],
            })
        root["children"].append(defs_node)

    for m in request.messages:
        msg_node: dict[str, Any] = {
            "name": f"[{m.index}] {m.role.value}",
            "type": f"msg_{m.role.value}",
            "role": m.role.value,
            "tokens": m.total_tokens,
            "children": [],
            "content": m.text_content[:CONTENT_PREVIEW_LIMIT],
        }
        root["children"].append(msg_node)

    return root


def _build_timeline_data(result: ProfileResult) -> list[dict[str, Any]]:
    if not result.session_timeline:
        return []
    return result.session_timeline


def _build_metrics(result: ProfileResult) -> dict[str, Any]:
    tc = result.analyzer_results.get("token_counter")

    metrics: dict[str, Any] = {
        "mode": result.mode,
        "source": result.source,
    }

    if tc:
        s = tc.summary
        metrics["total_input_tokens"] = s.get("total_input_tokens", 0)
        metrics["system_prompt_tokens"] = s.get("system_prompt_tokens", 0)
        metrics["tool_definition_tokens"] = s.get("tool_definition_tokens", 0)
        metrics["by_role"] = s.get("by_role", {})
        metrics["by_content_type"] = s.get("by_content_type", {})
        metrics["top_tools"] = s.get("top_tools_by_tokens", [])

    return metrics


def _build_report_data(
    result: ProfileResult,
    session: Session | None = None,
) -> dict[str, Any]:
    if session and session.requests:
        last_req = session.requests[-1]
    else:
        last_req = None

    icicle_tree = (
        _build_raw_icicle_tree(last_req) if last_req
        else {"name": "empty", "tokens": 0, "children": []}
    )

    session_trees: list[dict[str, Any]] = []
    if session and session.requests:
        prev_raw_messages = None
        for i, req in enumerate(session.requests):
            tree = _build_raw_icicle_tree(req)
            tree["request_index"] = req.request_index
            tree["trace_index"] = req.trace_index

            # Compute diff against previous request
            curr_raw_messages = (req.raw_input or {}).get("messages", [])
            if curr_raw_messages:
                diff_statuses, removed_count, removed_tokens, removed_messages = _compute_diff_statuses(
                    prev_raw_messages, curr_raw_messages,
                )
                _apply_diff_to_tree(tree, diff_statuses, removed_count, removed_tokens, removed_messages)
                prev_raw_messages = curr_raw_messages

            session_trees.append(tree)

    turn_boundaries = []
    if session and session.metadata:
        turn_boundaries = session.metadata.get("turn_boundaries", [])

    tool_analysis = _build_tool_analysis(last_req) if last_req else {"tools": [], "total_context_tokens": 0}

    return {
        "metrics": _build_metrics(result),
        "icicle": icicle_tree,
        "timeline": _build_timeline_data(result),
        "session_trees": session_trees,
        "turn_boundaries": turn_boundaries,
        "warnings": result.all_warnings,
        "tool_analysis": tool_analysis,
    }


def export_html(
    result: ProfileResult,
    output_path: Path,
    session: Session | None = None,
) -> Path:
    """Render an interactive HTML report."""
    data = _build_report_data(result, session)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace(
        "/*__DATA__*/null",
        json.dumps(data, ensure_ascii=False, default=str),
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path
