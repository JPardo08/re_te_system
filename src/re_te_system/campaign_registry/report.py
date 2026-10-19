"""Deterministic report rendering. No clock and no host identity."""

from __future__ import annotations

import json
from typing import Any, Mapping


def render_report(report: Mapping[str, Any]) -> str:
    """Serialize a report with sorted keys and a trailing newline."""
    return json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
