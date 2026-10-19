"""Load the declarative catalogs. This module does not interpret scientific rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CatalogLoadError(ValueError):
    """A catalog file is missing or is not a JSON object."""


REQUIRED_FILES = (
    "catalog/systems.json",
    "catalog/datasets.json",
    "catalog/protocols.json",
    "plans/sota.json",
    "plans/conditioning.json",
    "resources/cesvima_profiles.json",
    "schemas/task.schema.json",
    "schemas/leaderboard_entry.schema.json",
)


def load_catalog(catalog_root: Path) -> dict[str, Any]:
    """Read the eight declarative files under ``catalog_root``."""
    root = Path(catalog_root)
    loaded = {relative: _read_object(root / relative) for relative in REQUIRED_FILES}
    entries_path = root / "leaderboard" / "entries.json"
    entries: list[Any] = []
    if entries_path.exists():
        document = _read_object(entries_path)
        raw_entries = document.get("entries")
        if not isinstance(raw_entries, list):
            raise CatalogLoadError("leaderboard/entries.json needs an entries array")
        entries = raw_entries
    sota = loaded["plans/sota.json"]
    conditioning = loaded["plans/conditioning.json"]
    return {
        "catalog_root": str(root),
        "systems": _as_list(loaded["catalog/systems.json"], "systems"),
        "datasets": _as_list(loaded["catalog/datasets.json"], "datasets"),
        "protocols": _as_list(loaded["catalog/protocols.json"], "protocols"),
        "profiles": _as_list(loaded["resources/cesvima_profiles.json"], "profiles"),
        "profile_document": loaded["resources/cesvima_profiles.json"],
        "tasks": _as_list(sota, "tasks") + _as_list(conditioning, "tasks"),
        "condition_policy": conditioning.get("condition_policy", {}),
        "shell": conditioning.get("shell", {}),
        "notes": list(sota.get("notes", [])) + list(conditioning.get("notes", [])),
        "leaderboard_entries": entries,
        "task_schema": loaded["schemas/task.schema.json"],
        "leaderboard_schema": loaded["schemas/leaderboard_entry.schema.json"],
        "sota_authority": sota.get("authority"),
        "submit_all_is_not_authorization": sota.get("submit_all_is_not_authorization"),
    }


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CatalogLoadError(f"missing catalog file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogLoadError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CatalogLoadError(f"{path} must contain a JSON object")
    return document


def _as_list(document: dict[str, Any], key: str) -> list[Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise CatalogLoadError(f"{key} must be an array")
    return value
