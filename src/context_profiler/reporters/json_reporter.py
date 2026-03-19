"""JSON reporter — machine-readable output."""

from __future__ import annotations

import json
from pathlib import Path

from context_profiler.profiler import ProfileResult


def export_json(result: ProfileResult, output_path: Path | None = None) -> str:
    """Export the profile result as JSON.

    Returns the JSON string. If output_path is provided, also writes to file.
    """
    data = result.to_dict()
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    if output_path:
        output_path.write_text(json_str, encoding="utf-8")

    return json_str
