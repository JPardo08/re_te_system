"""Synthetic J17 stdout parsing and gated extraction traceback. No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from re_te_system.contracts import InputRecord
from re_te_system.runner.run import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

import terminal_run_json  # noqa: E402


DATASET = {"hash": "abc", "id": "test", "version": "1"}
SOURCE = {"documents_sha256": "abc", "manifest_sha256": "def"}
GENERATION = {
    "do_sample": False,
    "max_new_tokens": None,
    "num_beams": 1,
    "seed": None,
}
FAILURE_KEYS = {
    "error_message",
    "error_type",
    "example_id",
    "recoverable",
    "segment_id",
    "stage",
}
TERMINAL = '{"run_dir": "/scratch/gollie/native-smoke/run-1", "run_id": "run-1"}'


class ShapeProbe:
    model_name = "probe/gollie"
    model_revision = "probe"

    def extract(self, text: str, context: object) -> object:
        missing = None
        return missing.shape


def test_terminal_json_keeps_logs_and_one_run_object() -> None:
    stdout = "\n".join(
        [
            "FlashAttention initialized",
            "Flash rotary embedding initialized",
            '{"status": "loading"}',
            TERMINAL,
            "",
        ]
    )
    payload = terminal_run_json.parse_terminal_run_json(stdout)
    assert payload["run_dir"] == "/scratch/gollie/native-smoke/run-1"
    assert payload["run_id"] == "run-1"


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "FlashAttention initialized\n",
        "FlashAttention initialized\n{not json}\n",
        'FlashAttention initialized\n{"ok": true}\n',
        f"{TERMINAL}\n{TERMINAL}\n",
        TERMINAL + "\nFlashAttention initialized\n",
        TERMINAL + " trailing\n",
        '{"run_dir": ""}\n',
        '{"run_dir": 7}\n',
    ],
)
def test_terminal_json_rejects_ambiguous_or_malformed_stdout(stdout: str) -> None:
    with pytest.raises(terminal_run_json.TerminalRunJsonError):
        terminal_run_json.parse_terminal_run_json(stdout)


def test_terminal_json_cli_prints_only_the_run_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "cli-stdout.txt"
    log.write_text("library ready\n" + TERMINAL + "\n", encoding="utf-8")
    assert terminal_run_json.main([str(log)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "/scratch/gollie/native-smoke/run-1\n"
    assert captured.err == ""
    assert terminal_run_json.main([str(tmp_path / "missing.txt")]) == 1
    assert capsys.readouterr().err.startswith("FAIL:")


def _run_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grant: str | None) -> Path:
    monkeypatch.delenv("GOLLIE_J17_TRACE_AUTHORIZED", raising=False)
    if grant is not None:
        monkeypatch.setenv("GOLLIE_J17_TRACE_AUTHORIZED", grant)
    return run_pipeline(
        inputs=[InputRecord("gollie-1", "gollie-1", "Ana visited Mary.")],
        extractor=ShapeProbe(),
        output_root=tmp_path,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )


def test_extraction_traceback_stays_closed_without_the_exact_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _run_probe(tmp_path, monkeypatch, None)
    captured = capsys.readouterr()
    statistics = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
    failure = json.loads((run_dir / "failures.jsonl").read_text(encoding="utf-8"))
    assert statistics["raw_outputs"] == 0
    assert statistics["inputs_failed"] == 1
    assert set(failure) == FAILURE_KEYS
    assert failure["stage"] == "extraction"
    assert failure["error_type"] == "AttributeError"
    assert "shape" in failure["error_message"]
    assert "traceback" not in failure
    assert not (run_dir / "extraction_traceback.txt").exists()
    assert "Traceback" not in captured.err

    again = _run_probe(tmp_path / "other", monkeypatch, "true")
    assert not (again / "extraction_traceback.txt").exists()
    assert "Traceback" not in capsys.readouterr().err


def test_authorized_extraction_traceback_preserves_the_failure_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _run_probe(tmp_path, monkeypatch, "1")
    captured = capsys.readouterr()
    failure = json.loads((run_dir / "failures.jsonl").read_text(encoding="utf-8"))
    trace = (run_dir / "extraction_traceback.txt").read_text(encoding="utf-8")
    statistics = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["inputs_succeeded"] == 0
    assert statistics["raw_outputs"] == 0
    assert set(failure) == FAILURE_KEYS
    assert failure["error_type"] == "AttributeError"
    for text in (trace, captured.err):
        assert "Traceback (most recent call last)" in text
        assert "AttributeError" in text
        assert "shape" in text
        assert "in extract" in text
        assert "missing.shape" in text


def test_j17_trace_job_is_individual_and_stdout_parsing_is_terminal() -> None:
    smoke = (SCRIPTS / "j17_gollie_native_smoke.sh").read_text(encoding="utf-8")
    normal = (ROOT / "jobs" / "cesvima" / "j17_gollie_native_smoke.sbatch").read_text(encoding="utf-8")
    trace = (ROOT / "jobs" / "cesvima" / "j17_gollie_trace.sbatch").read_text(encoding="utf-8")
    submitter = (SCRIPTS / "submit_train_first.sh").read_text(encoding="utf-8")
    assert "json.load(sys.stdin)" not in smoke
    assert "terminal_run_json.py" in smoke
    assert "official_re_documents.jsonl" in smoke
    assert "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9" in smoke
    assert "--gollie-weight-snapshot" in smoke
    assert "unset GOLLIE_J17_TRACE_AUTHORIZED" in normal
    assert "GOLLIE_J17_SMOKE_AUTHORIZED=1" not in normal
    assert "GOLLIE_J17_TRACE_AUTHORIZED=1" not in trace
    assert "GOLLIE_J17_SMOKE_AUTHORIZED=1" not in trace
    assert 'saved_trace="${GOLLIE_J17_TRACE_AUTHORIZED:-}"' in trace
    assert 'export GOLLIE_J17_TRACE_AUTHORIZED="${saved_trace}"' in trace
    assert "unset GOLLIE_J17_TRACE_AUTHORIZED" not in trace
    assert "--partition=standard-gpu" in trace
    assert "--gres=gpu:l40s:1" in trace
    assert "j17_gollie_native_smoke.sh" in trace
    assert "j17_gollie_trace.sbatch" not in submitter
    assert "submit_train_first.sh --all" not in trace
    assert "--all" not in trace
