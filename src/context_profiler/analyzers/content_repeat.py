"""Content repeat detector — find exact and near-exact duplicate content blocks.

Scans all content blocks in the messages array, hashes them for exact match
detection, and uses n-gram Jaccard similarity for near-duplicate detection.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from context_profiler.analyzers.base import AnalyzerResult, BaseAnalyzer
from context_profiler.models import APIRequest, ContentBlock

_MIN_TOKENS_TO_CHECK = 50


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode(errors="replace")).hexdigest()[:16]


def _ngram_set(text: str, n: int = 4) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {text}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class ContentRepeatAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "content_repeat"

    def analyze(self, request: APIRequest) -> AnalyzerResult:
        chunks: list[dict[str, Any]] = []
        for msg in request.messages:
            for block in msg.blocks:
                if block.token_count < _MIN_TOKENS_TO_CHECK:
                    continue
                chunks.append({
                    "msg_index": msg.index,
                    "role": msg.role.value,
                    "block_type": block.block_type.value,
                    "tool_name": block.tool_name,
                    "text": block.text,
                    "tokens": block.token_count,
                    "hash": _content_hash(block.text),
                    "ngrams": _ngram_set(block.text),
                })

        # Phase 1: exact duplicates (same hash)
        hash_groups: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            hash_groups[chunk["hash"]].append(chunk)

        exact_duplicates: list[dict[str, Any]] = []
        exact_wasted_tokens = 0
        for h, group in hash_groups.items():
            if len(group) < 2:
                continue
            wasted = sum(c["tokens"] for c in group[1:])
            exact_wasted_tokens += wasted
            exact_duplicates.append({
                "type": "exact",
                "count": len(group),
                "tokens_each": group[0]["tokens"],
                "wasted_tokens": wasted,
                "preview": group[0]["text"][:200],
                "locations": [
                    {
                        "msg_index": c["msg_index"],
                        "role": c["role"],
                        "block_type": c["block_type"],
                        "tool_name": c["tool_name"],
                    }
                    for c in group
                ],
            })

        # Phase 2: near-duplicates (Jaccard > threshold) among non-exact chunks
        seen_hashes = {h for h, g in hash_groups.items() if len(g) >= 2}
        non_exact = [c for c in chunks if c["hash"] not in seen_hashes]

        near_duplicates: list[dict[str, Any]] = []
        near_wasted_tokens = 0
        matched: set[int] = set()

        for i in range(len(non_exact)):
            if i in matched:
                continue
            cluster = [non_exact[i]]
            for j in range(i + 1, len(non_exact)):
                if j in matched:
                    continue
                sim = _jaccard_similarity(non_exact[i]["ngrams"], non_exact[j]["ngrams"])
                if sim > 0.8:
                    cluster.append(non_exact[j])
                    matched.add(j)

            if len(cluster) >= 2:
                matched.add(i)
                wasted = sum(c["tokens"] for c in cluster[1:])
                near_wasted_tokens += wasted
                near_duplicates.append({
                    "type": "near_duplicate",
                    "count": len(cluster),
                    "avg_tokens": sum(c["tokens"] for c in cluster) // len(cluster),
                    "wasted_tokens": wasted,
                    "preview": cluster[0]["text"][:200],
                    "locations": [
                        {
                            "msg_index": c["msg_index"],
                            "role": c["role"],
                            "block_type": c["block_type"],
                            "tool_name": c["tool_name"],
                        }
                        for c in cluster
                    ],
                })

        total_tokens = request.total_input_tokens
        total_wasted = exact_wasted_tokens + near_wasted_tokens

        summary = {
            "exact_duplicate_groups": len(exact_duplicates),
            "near_duplicate_groups": len(near_duplicates),
            "exact_wasted_tokens": exact_wasted_tokens,
            "near_wasted_tokens": near_wasted_tokens,
            "total_wasted_tokens": total_wasted,
            "waste_ratio": total_wasted / total_tokens if total_tokens > 0 else 0,
        }

        details = sorted(
            exact_duplicates + near_duplicates,
            key=lambda x: x["wasted_tokens"],
            reverse=True,
        )

        warnings = []
        if summary["waste_ratio"] > 0.1:
            warnings.append(
                f"Content duplication: {total_wasted:,} redundant tokens "
                f"({summary['waste_ratio'] * 100:.1f}% of total)"
            )

        return AnalyzerResult(
            analyzer_name=self.name,
            summary=summary,
            details=details,
            warnings=warnings,
        )
