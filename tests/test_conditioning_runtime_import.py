"""Import the real benchmark shell in a clean interpreter.

Local Gemma tests passed before job 1369033 because this workspace checks out
re_te_benchmark on feat/paper3-conditioning, which contains conditioning/.
CESVIMA main at 5fe30ca does not. These tests do not mock that package.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_SRC = ROOT / "src"


def _clean_env(benchmark_src: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SYSTEM_SRC)
    env.pop("RE_TE_BENCHMARK_SRC", None)
    if benchmark_src is not None:
        env["RE_TE_BENCHMARK_SRC"] = str(benchmark_src)
    return env


def _run(script: str, benchmark_src: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=_clean_env(benchmark_src),
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_process_imports_the_real_shell_v2() -> None:
    completed = _run(
        """
        import sys
        assert "re_te_benchmark" not in sys.modules
        from re_te_system.controlled_experiment.dependencies import ensure_conditioning_runtime
        from re_te_system.controlled_experiment.gemma4 import launch_g04, smoke_requests

        located = ensure_conditioning_runtime()
        import re_te_benchmark.conditioning.shell as shell
        from re_te_benchmark.conditioning.c2 import load_c1_labels

        assert shell.SHELL_ID == "common-instruction-shell-v2"
        assert shell.conditioning_for("C0") == ""
        labels = load_c1_labels()
        assert labels == ["Duty", "Right", "Privilege", "NoRight"]
        slot = shell.conditioning_for("C1")
        assert slot == "relation_labels:\\n- Duty\\n- Right\\n- Privilege\\n- NoRight\\n"
        assert "Gold" not in slot and "Silver" not in slot
        text = "Ava notifies Blake about one invented duty."
        assert shell.render_condition("C1", text).replace(slot, "", 1) == shell.render_condition("C0", text)
        requests = smoke_requests()
        assert [(item["document_id"], item["condition"]) for item in requests] == [
            ("invented-gemma-01", "C0"),
            ("invented-gemma-01", "C1"),
            ("invented-gemma-02", "C0"),
            ("invented-gemma-02", "C1"),
        ]
        assert requests[0]["rendered_prompt_sha256"] != requests[1]["rendered_prompt_sha256"]
        assert all(item["uses_gold"] is False for item in requests)
        try:
            launch_g04({})
        except Exception as exc:
            assert "g04_phase_lock_closed" in str(exc)
        else:
            raise AssertionError("G04 opened")
        assert located["method"] == "sibling_checkout"
        assert shell.__file__.endswith("conditioning/shell.py")
        """
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_missing_conditioning_package_fails_explicitly(tmp_path: Path) -> None:
    package = tmp_path / "src" / "re_te_benchmark"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Benchmark checkout without the shell."""\n', encoding="utf-8")
    completed = _run(
        """
        from re_te_system.controlled_experiment.dependencies import ensure_conditioning_runtime
        try:
            ensure_conditioning_runtime()
        except ImportError as exc:
            message = str(exc)
            assert "re_te_benchmark.conditioning" in message
            assert "feat/paper3-conditioning" in message
            assert "36d6203" in message
            assert "does not vendor a second shell" in message
        else:
            raise AssertionError("a checkout without conditioning was accepted")
        """,
        benchmark_src=tmp_path / "src",
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


if __name__ == "__main__":
    import tempfile

    test_clean_process_imports_the_real_shell_v2()
    with tempfile.TemporaryDirectory() as directory:
        test_missing_conditioning_package_fails_explicitly(Path(directory))
    print("conditioning runtime import tests passed")
