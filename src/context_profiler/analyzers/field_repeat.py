"""Field-level repeat detector for tool call arguments.

Groups tool_use blocks by (tool_name, field_name), compares field values
across invocations, and flags fields with high repetition.

This is the core differentiator — catches the PR #123 scenario where
generate_canvas_component.requirements was repeated 17 times (110K tokens).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from context_profiler.analyzers.base import AnalyzerResult, BaseAnalyzer
from context_profiler.analyzers.content_repeat import _jaccard_similarity, _ngram_set
from context_profiler.models import APIRequest, BlockType
from context_profiler.token_utils import count_tokens

_MIN_FIELD_TOKENS = 20


class FieldRepeatAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "field_repeat"

    def analyze(self, request: APIRequest) -> AnalyzerResult:
        # Collect all tool_use blocks with their parsed arguments
        tool_calls: list[dict[str, Any]] = []
        for msg in request.messages:
            for block in msg.blocks:
                if block.block_type != BlockType.TOOL_USE:
                    continue
                if not block.tool_input or not block.tool_name:
                    continue
                tool_calls.append({
                    "msg_index": msg.index,
                    "tool_name": block.tool_name,
                    "tool_input": block.tool_input,
                })

        # Group by (tool_name, field_name) and collect field values
        field_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for tc in tool_calls:
            for field_name, field_value in tc["tool_input"].items():
                value_str = str(field_value) if not isinstance(field_value, str) else field_value
                tokens = count_tokens(value_str)
                if tokens < _MIN_FIELD_TOKENS:
                    continue
                field_groups[(tc["tool_name"], field_name)].append({
                    "msg_index": tc["msg_index"],
                    "value": value_str,
                    "tokens": tokens,
                    "ngrams": _ngram_set(value_str),
                })

        repeated_fields: list[dict[str, Any]] = []
        total_wasted = 0

        for (tool_name, field_name), occurrences in field_groups.items():
            if len(occurrences) < 2:
                continue

            # Compare all pairs to the first occurrence to compute avg similarity
            base = occurrences[0]
            similarities = []
            for occ in occurrences[1:]:
                sim = _jaccard_similarity(base["ngrams"], occ["ngrams"])
                similarities.append(sim)

            avg_similarity = sum(similarities) / len(similarities) if similarities else 0

            if avg_similarity < 0.5:
                continue

            total_tokens = sum(o["tokens"] for o in occurrences)
            unique_tokens = occurrences[0]["tokens"]
            wasted = total_tokens - unique_tokens
            total_wasted += wasted

            repeated_fields.append({
                "tool_name": tool_name,
                "field_name": field_name,
                "occurrences": len(occurrences),
                "avg_similarity": round(avg_similarity, 3),
                "tokens_per_occurrence": unique_tokens,
                "total_tokens": total_tokens,
                "wasted_tokens": wasted,
                "savings_ratio": round(wasted / total_tokens, 3) if total_tokens > 0 else 0,
                "preview": base["value"][:200],
                "msg_indices": [o["msg_index"] for o in occurrences],
            })

        repeated_fields.sort(key=lambda x: x["wasted_tokens"], reverse=True)

        total_input = request.total_input_tokens
        summary = {
            "repeated_field_groups": len(repeated_fields),
            "total_wasted_tokens": total_wasted,
            "waste_ratio": total_wasted / total_input if total_input > 0 else 0,
            "top_offenders": [
                {
                    "field": f"{r['tool_name']}.{r['field_name']}",
                    "occurrences": r["occurrences"],
                    "similarity": r["avg_similarity"],
                    "wasted_tokens": r["wasted_tokens"],
                }
                for r in repeated_fields[:5]
            ],
        }

        warnings = []
        for r in repeated_fields:
            if r["wasted_tokens"] > 5000:
                warnings.append(
                    f"{r['tool_name']}.{r['field_name']}: "
                    f"{r['occurrences']} occurrences, "
                    f"{r['avg_similarity'] * 100:.0f}% similar, "
                    f"{r['wasted_tokens']:,} tokens redundant"
                )

        return AnalyzerResult(
            analyzer_name=self.name,
            summary=summary,
            details=repeated_fields,
            warnings=warnings,
        )
