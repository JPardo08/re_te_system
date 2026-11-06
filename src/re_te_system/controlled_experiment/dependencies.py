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

import importlib
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


def ensure_conditioning_runtime() -> dict[str, str]:
    """Import the checked-in C0/C1 shell. Do not vendor a second copy.

    ``re_te_benchmark`` on ``main`` (5fe30ca) has the specification and the C1
    label file, but not ``conditioning/shell.py``. The renderer
    ``common-instruction-shell-v2`` is on ``feat/paper3-conditioning`` at
    commit 36d6203.
    """
    located = ensure_package("re_te_benchmark", "re_te_benchmark")
    required = (
        "re_te_benchmark.conditioning",
        "re_te_benchmark.conditioning.shell",
        "re_te_benchmark.conditioning.c2",
    )
    missing: list[str] = []
    first_error: ModuleNotFoundError | None = None
    for name in required:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing.append(name)
            if first_error is None:
                first_error = exc
    if missing:
        revision = git_revision(repository_root_from_package("re_te_benchmark"))
        raise ImportError(
            "re_te_benchmark was imported "
            f"({located['method']}, {located['module_file']}, HEAD {revision['head']}) "
            "but does not provide " + ", ".join(missing) + ". "
            "common-instruction-shell-v2 is checked in on branch "
            "feat/paper3-conditioning at commit 36d6203 and is absent from main "
            "5fe30ca. This runtime does not vendor a second shell."
        ) from first_error
    shell = importlib.import_module("re_te_benchmark.conditioning.shell")
    if getattr(shell, "SHELL_ID", None) != "common-instruction-shell-v2":
        raise ImportError(
            "re_te_benchmark.conditioning.shell.SHELL_ID is "
            f"{getattr(shell, 'SHELL_ID', None)!r}, not 'common-instruction-shell-v2'."
        )
    return located


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
