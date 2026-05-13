"""JSON Schemas exposed by the CLI for agent-readable contracts."""

from __future__ import annotations

from typing import Any


TRACE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ContextTrace",
    "type": "object",
    "required": ["schema_version", "runs"],
    "properties": {
        "schema_version": {"type": "string", "const": "0.1"},
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["run_id", "events"],
                "properties": {
                    "run_id": {"type": "string"},
                    "source_format": {"type": "string"},
                    "metadata": {"type": "object"},
                    "events": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    },
}


DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ContextDiagnosis",
    "type": "object",
    "required": ["schema_version", "issues", "summary"],
    "properties": {
        "schema_version": {"type": "string", "const": "0.1"},
        "summary": {"type": "object"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["code", "severity", "message"],
                "properties": {
                    "code": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    "message": {"type": "string"},
                    "evidence": {"type": "object"},
                    "recommendation": {"type": "string"},
                },
            },
        },
    },
}


def get_schema(name: str) -> dict[str, Any]:
    if name == "trace":
        return TRACE_SCHEMA
    if name == "diagnosis":
        return DIAGNOSIS_SCHEMA
    raise ValueError("Unknown schema. Available schemas: trace, diagnosis")
