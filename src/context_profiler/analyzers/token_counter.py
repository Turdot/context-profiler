"""Token counting analyzer — breakdown by role, content type, and tool name."""

from __future__ import annotations

from collections import defaultdict

from context_profiler.analyzers.base import AnalyzerResult, BaseAnalyzer
from context_profiler.models import APIRequest, BlockType, Role


class TokenCounterAnalyzer(BaseAnalyzer):

    @property
    def name(self) -> str:
        return "token_counter"

    def analyze(self, request: APIRequest) -> AnalyzerResult:
        by_role: dict[str, int] = defaultdict(int)
        by_content_type: dict[str, int] = defaultdict(int)
        by_tool_name: dict[str, int] = defaultdict(int)
        per_message: list[dict] = []
        total_tokens = 0

        for msg in request.messages:
            msg_tokens = msg.total_tokens
            total_tokens += msg_tokens
            by_role[msg.role.value] += msg_tokens

            for block in msg.blocks:
                by_content_type[block.block_type.value] += block.token_count
                if block.tool_name:
                    by_tool_name[block.tool_name] += block.token_count

            per_message.append({
                "index": msg.index,
                "role": msg.role.value,
                "tokens": msg_tokens,
                "block_types": [b.block_type.value for b in msg.blocks],
            })

        tool_def_tokens = request.tool_definition_tokens
        total_tokens += tool_def_tokens

        top_messages = sorted(per_message, key=lambda x: x["tokens"], reverse=True)[:10]
        top_tools = sorted(by_tool_name.items(), key=lambda x: x[1], reverse=True)[:10]

        tool_defs_detail = [
            {"name": t.name, "tokens": t.token_count}
            for t in sorted(request.tools, key=lambda t: t.token_count, reverse=True)
        ]

        summary = {
            "total_input_tokens": total_tokens,
            "message_tokens": total_tokens - tool_def_tokens,
            "tool_definition_tokens": tool_def_tokens,
            "system_prompt_tokens": request.system_prompt_tokens,
            "by_role": dict(by_role),
            "by_content_type": dict(by_content_type),
            "top_tools_by_tokens": top_tools,
            "tool_definitions": tool_defs_detail,
        }

        warnings = []
        if tool_def_tokens > total_tokens * 0.3:
            warnings.append(
                f"Tool definitions consume {tool_def_tokens:,} tokens "
                f"({tool_def_tokens / total_tokens * 100:.1f}% of total)"
            )
        if request.system_prompt_tokens > total_tokens * 0.3:
            warnings.append(
                f"System prompt consumes {request.system_prompt_tokens:,} tokens "
                f"({request.system_prompt_tokens / total_tokens * 100:.1f}% of total)"
            )

        return AnalyzerResult(
            analyzer_name=self.name,
            summary=summary,
            details=top_messages,
            warnings=warnings,
        )
