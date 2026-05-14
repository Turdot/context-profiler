"""Known input format metadata for agent-readable CLI discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FormatSpec:
    name: str
    description: str
    status: str
    input_kind: str
    confidence: str
    required_signals: list[str]
    common_sources: list[str]
    analysis_scope: list[str]
    limitations: list[str]
    agent_conversion_guidance: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORMAT_REGISTRY: dict[str, FormatSpec] = {
    "openai": FormatSpec(
        name="openai",
        description="OpenAI-compatible chat completion request with messages and optional tools.",
        status="supported",
        input_kind="provider-request",
        confidence="exact",
        required_signals=["messages[] with role/content", "optional tools[] with function schemas"],
        common_sources=["OpenAI API logs", "Azure OpenAI logs", "OpenAI-compatible gateways"],
        analysis_scope=[
            "Exact visible request token distribution",
            "Tool definition size",
            "Tool call arguments and tool result content",
        ],
        limitations=["Does not include hidden provider-side prompt or runtime state."],
        agent_conversion_guidance="If data is an OpenAI-compatible request, pass it directly. Otherwise map messages/tools/model into ContextTrace or OpenAI-compatible shape.",
        notes=["Tool calls are read from assistant message tool_calls."],
    ),
    "anthropic": FormatSpec(
        name="anthropic",
        description="Anthropic Messages API request with typed content blocks.",
        status="supported",
        input_kind="provider-request",
        confidence="exact",
        required_signals=["messages[]", "content blocks with type", "optional tools[] with input_schema"],
        common_sources=["Anthropic API logs", "Claude-compatible request captures"],
        analysis_scope=[
            "Exact visible request token distribution",
            "System prompt and tool schema size",
            "Typed tool_use/tool_result content",
        ],
        limitations=["Does not include hidden provider-side prompt or runtime state."],
        agent_conversion_guidance="Map Anthropic Messages API fields to system/messages/tools with typed content blocks.",
        notes=["System prompt may be top-level system string or blocks."],
    ),
    "langfuse": FormatSpec(
        name="langfuse",
        description="Langfuse trace export containing observations and GENERATION inputs.",
        status="supported",
        input_kind="observability-trace",
        confidence="high",
        required_signals=[
            "observations[]",
            "GENERATION observations with input.messages, input {role, content}, or string input",
        ],
        common_sources=["Langfuse UI export", "Langfuse public API"],
        analysis_scope=[
            "Per-generation context growth",
            "Visible generation input messages",
            "Tool input/output if captured in generation inputs",
        ],
        limitations=["Only GENERATION observations with analyzable input content are profiled in the current adapter."],
        agent_conversion_guidance="Use the Langfuse public API via curl to fetch traces/observations, then pass the exported trace JSON directly. Avoid langfuse-cli for trace analysis because it may omit nested observation or generation fields.",
        notes=["Current adapter extracts generation inputs and delegates to OpenAI parsing."],
    ),
    "cursor-jsonl": FormatSpec(
        name="cursor-jsonl",
        description="Cursor agent transcript JSONL with role/message/content and tool_use blocks.",
        status="supported",
        input_kind="agent-transcript",
        confidence="partial",
        required_signals=["one JSON object per line", "top-level role", "message.content[] blocks"],
        common_sources=["Cursor agent transcript exports", "Cursor workspace agent-transcripts"],
        analysis_scope=[
            "Visible agent loop growth",
            "Tool use and tool result repetition",
            "Turn-to-turn content evolution in recorded transcript",
        ],
        limitations=[
            "Transcript may omit hidden system prompts, rules, tool definitions, and provider request compaction.",
            "Token counts reflect visible transcript content only.",
        ],
        agent_conversion_guidance="If Cursor stores a compatible JSONL transcript, pass it directly. If not, convert events into JSONL lines with role and message.content blocks.",
        notes=["Parsed as cumulative session snapshots so existing HTML timeline remains available."],
    ),
    "claude-code-jsonl": FormatSpec(
        name="claude-code-jsonl",
        description="Claude Code transcript JSONL event stream with user/assistant messages and tool blocks.",
        status="supported",
        input_kind="agent-transcript",
        confidence="partial",
        required_signals=["one JSON object per line", "type=user|assistant", "message.content"],
        common_sources=["~/.claude/projects/** session transcripts", "Claude Code subagent transcripts"],
        analysis_scope=[
            "Visible agent loop growth",
            "Tool use and tool result repetition",
            "Main-session and subagent transcript analysis as separate runs",
        ],
        limitations=[
            "Transcript may omit hidden system prompts, skills, tool definitions, MCP schemas, and raw provider requests.",
            "Subagent linkage is not yet modeled as a single graph.",
        ],
        agent_conversion_guidance="Use Claude Code JSONL session files directly. Analyze subagent JSONL files separately until linked-run graph support lands.",
        notes=["Subagent directories can be analyzed as separate transcript files in the first phase."],
    ),
    "otel": FormatSpec(
        name="otel",
        description="OpenTelemetry or OpenInference span tree for LLM and agent operations.",
        status="planned",
        input_kind="observability-trace",
        confidence="high",
        required_signals=["trace_id", "span_id", "parent span relation", "span attributes"],
        common_sources=["Phoenix", "Braintrust", "CrewAI", "Pydantic AI", "OpenAI Agents SDK exporters"],
        analysis_scope=[
            "Span tree structure",
            "LLM inputs/outputs when captured",
            "Tool call spans and parent-child relationships",
        ],
        limitations=["Content capture may be disabled for privacy; span attributes vary by exporter."],
        agent_conversion_guidance="Map OTel/OpenInference spans into ContextTrace events preserving trace_id/span_id/parent_id and LLM/tool attributes.",
        notes=["Initial version should focus on schema documentation before full span ingestion."],
    ),
    "agent-trace": FormatSpec(
        name="agent-trace",
        description="Multi-turn agent traces with llm_steps and tool spans.",
        status="supported",
        input_kind="benchmark-trajectory",
        confidence="dataset-dependent",
        required_signals=["llm_steps[]", "spans[]", "tool_input/tool_output"],
        common_sources=["pagarsky/agent-trace"],
        analysis_scope=[
            "LLM step to tool span relationships",
            "Tool input/output growth",
            "Turn-level trajectory evolution",
        ],
        limitations=["Dataset rows may store nested structures as JSON strings that must be decoded first."],
        agent_conversion_guidance="Decode llm_steps_json/spans_json/metadata_json, then map LLM steps and tool spans into ContextTrace events.",
        notes=["Strong candidate for Context Event Graph demos because LLM steps and tool spans are explicit."],
    ),
    "agent-trajectories": FormatSpec(
        name="agent-trajectories",
        description="Large multi-turn academic agent trajectory records across multiple benchmarks.",
        status="planned",
        input_kind="benchmark-trajectory",
        confidence="dataset-dependent",
        required_signals=["conversation messages", "benchmark metadata", "reward/evaluation fields"],
        common_sources=["cx-cmu/agent_trajectories"],
        analysis_scope=[
            "Long multi-turn conversation growth",
            "Benchmark outcome correlated with context pathologies",
            "Tool/observation repetition when represented in messages",
        ],
        limitations=["The dataset spans multiple benchmarks with heterogeneous message conventions."],
        agent_conversion_guidance="Use the dataset schema to extract messages and metadata, then normalize each trajectory into a ContextTrace run.",
        notes=["Best fit for turn-to-turn context evolution analysis across long trajectories."],
    ),
    "swe-agent-traj": FormatSpec(
        name="swe-agent-traj",
        description="SWE-agent trajectory files with thought/action/observation steps and LM queries.",
        status="planned",
        input_kind="benchmark-trajectory",
        confidence="dataset-dependent",
        required_signals=["trajectory steps", "query/messages", "action", "observation"],
        common_sources=["SWE-agent .traj files", "nebius/SWE-agent-trajectories"],
        analysis_scope=[
            "Coding-agent action/observation loops",
            "Terminal output and test feedback growth",
            "Patch churn and repeated modification patterns in later graph analyzers",
        ],
        limitations=["Different SWE-agent versions use different trajectory field names."],
        agent_conversion_guidance="Prefer query/messages when available; otherwise map thought/action/observation turns into ContextTrace events.",
        notes=["Useful for coding-agent patch churn, terminal output, and test feedback loop analysis."],
    ),
}


def list_formats() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in FORMAT_REGISTRY.values()]


def describe_format(name: str) -> dict[str, Any]:
    try:
        return FORMAT_REGISTRY[name].to_dict()
    except KeyError as exc:
        available = ", ".join(sorted(FORMAT_REGISTRY))
        raise ValueError(f"Unknown format '{name}'. Available formats: {available}") from exc
