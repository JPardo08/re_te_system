"""Pinned Hugging Face snapshot helpers. Callers decide local-only versus download."""

from __future__ import annotations

from pathlib import Path

from cesvima_common import fail, require_sha, write_json


def snapshot(repo_id: str, revision: str, *, local_files_only: bool) -> Path:
    require_sha(revision, repo_id)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        fail(f"huggingface_hub is not importable: {exc}")
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    resolved = Path(path).resolve()
    scratch = Path(__import__("os").environ["SCRATCH_ROOT"]).resolve()
    if scratch not in resolved.parents:
        fail(f"snapshot resolved outside SCRATCH_ROOT: {resolved}")
    return resolved


def cache_or_download(repo_id: str, revision: str, pointer: Path) -> Path:
    """Reuse a complete pinned snapshot. Download only when it is absent."""
    try:
        path = snapshot(repo_id, revision, local_files_only=True)
        source = "cache"
    except Exception as first_error:
        print(f"cache miss for {repo_id}@{revision}: {first_error}", flush=True)
        try:
            path = snapshot(repo_id, revision, local_files_only=False)
        except Exception as second_error:
            fail(f"could not cache {repo_id}@{revision}: {second_error}")
        source = "downloaded"
    write_json(
        pointer,
        {
            "local_path": str(path),
            "repository": repo_id,
            "revision": revision,
            "source": source,
        },
    )
    print(f"PASS snapshot {repo_id}@{revision} via {source}", flush=True)
    return path
