"""Read one unambiguous terminal run object from a CLI stdout log."""

from __future__ import annotations

import json
import sys
from pathlib import Path


class TerminalRunJsonError(ValueError):
    """The stdout stream does not end in one unambiguous run object."""


def parse_terminal_run_json(stdout: str) -> dict[str, object]:
    """Accept logs followed by one JSON object that owns ``run_dir``.

    Earlier non-JSON lines are logs. Another object that also contains
    ``run_dir``, a malformed terminal line, an empty stream, or a terminal
    object whose ``run_dir`` is not a non-empty string is rejected.
    """
    content = [line for line in stdout.splitlines() if line.strip()]
    if not content:
        raise TerminalRunJsonError("CLI stdout is empty")
    candidates: list[tuple[int, dict[str, object]]] = []
    for index, line in enumerate(content):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "run_dir" in value:
            candidates.append((index, value))
    if len(candidates) != 1 or candidates[0][0] != len(content) - 1:
        raise TerminalRunJsonError(
            "CLI stdout must end with exactly one JSON object that contains run_dir"
        )
    run_dir = candidates[0][1]["run_dir"]
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise TerminalRunJsonError("run_dir must be a non-empty string")
    return candidates[0][1]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("FAIL: usage: terminal_run_json.py STDOUT_FILE", file=sys.stderr)
        return 2
    try:
        payload = parse_terminal_run_json(Path(args[0]).read_text(encoding="utf-8"))
    except (OSError, TerminalRunJsonError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(payload["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
