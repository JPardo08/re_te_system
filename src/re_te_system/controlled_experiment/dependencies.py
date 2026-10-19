"""Resolve sibling Paper 3 packages without a hard-coded machine path.

Resolution order for ``re_te_benchmark`` and ``re_te_evaluation``:

1. Use the package if it is already importable.
2. Use ``RE_TE_BENCHMARK_SRC`` or ``RE_TE_EVALUATION_SRC`` when set. The value
   is the ``src`` directory that contains the package, not a checkout root.
3. Use a sibling checkout of this repository:
   ``<parent of re_te_system>/re_te_benchmark/src``. This is a layout
   convention of the workspace, and the manifest records it as
   ``sibling_checkout`` so the resolution is not silent.

A missing package raises ``ImportError``. The harness does not vendor payload
copies and does not invent a repository path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SYSTEM_REPO = Path(__file__).resolve().parents[3]


def _insert_src(src: Path) -> None:
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)


def ensure_package(package: str, repo_name: str) -> dict[str, str]:
    try:
        module = __import__(package)
    except ImportError:
        module = None
    if module is not None:
        return _located(package, module, "already_importable")
    env_name = "RE_TE_" + repo_name.removeprefix("re_te_").upper() + "_SRC"
    configured = os.environ.get(env_name)
    if configured:
        src = Path(configured)
        if not src.is_dir():
            raise ImportError(f"{env_name} does not point at a directory: {configured}")
        _insert_src(src)
        module = __import__(package)
        return _located(package, module, "environment_src")
    sibling_src = SYSTEM_REPO.parent / repo_name / "src"
    if (sibling_src / package).is_dir():
        _insert_src(sibling_src)
        module = __import__(package)
        return _located(package, module, "sibling_checkout")
    raise ImportError(
        f"Cannot import {package}. Install it, export {env_name} as its src "
        "directory, or place the checkout next to re_te_system."
    )


def _located(package: str, module: Any, method: str) -> dict[str, str]:
    path = getattr(module, "__file__", None)
    return {
        "method": method,
        "module_file": None if path is None else str(path),
        "package": package,
    }


def repository_root_from_package(package: str) -> Path | None:
    module = __import__(package)
    path = getattr(module, "__file__", None)
    if path is None:
        return None
    candidate = Path(path).resolve().parents[2]
    if (candidate / "src" / package).is_dir():
        return candidate
    return None


def git_revision(root: Path | None) -> dict[str, Any]:
    """Read HEAD and porcelain. A dirty tree is not identified by its SHA alone."""
    if root is None or not root.is_dir():
        return {
            "commit_identifies_tree": False,
            "dirty": None,
            "head": "UNRESOLVED",
            "root_resolved": False,
        }
    try:
        head = _git(root, "rev-parse", "HEAD")
        porcelain = _git(root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit_identifies_tree": False,
            "dirty": None,
            "head": "UNRESOLVED",
            "root_resolved": True,
        }
    dirty = bool(porcelain.strip())
    return {
        "commit_identifies_tree": not dirty,
        "dirty": dirty,
        "head": head.strip(),
        "root_resolved": True,
    }


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
