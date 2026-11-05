"""Local G00-G02 checks. No weights are downloaded or loaded."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from re_te_system.controlled_experiment.campaign import render_controlled_prompt
from re_te_system.controlled_experiment.gemma4 import (
    APPROVED_REVISIONS,
    BASE_MODEL_ID,
    CANDIDATE_REVISION,
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    ESTIMATED_BF16_WEIGHT_BYTES,
    G01_ENV,
    G02_ENV,
    MODEL_ID,
    GemmaAcquireNotAuthorized,
    GemmaContextBudgetError,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaModelIdentityError,
    GemmaRevisionNotApproved,
    GemmaSilentTruncationRefused,
    GemmaThinkingNotAuthorized,
    GemmaWorkerTimeout,
    acquire_checkpoint,
    assert_instruction_tuned,
    assert_revision_approved,
    assess_prompt_budget,
    classify_runtime_failure,
    execute_bounded_preflight,
    preflight_decoding_contract,
    preflight_plan,
    publish_snapshot,
    readiness_report,
    render_serialized_turn,
    run_bounded_command,
    scientific_decoding_contract,
    separate_official_channels,
    serialize_text_turn,
    technical_manifest,
    transport_and_parse,
    verify_snapshot_directory,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
SUBMITTER = ROOT / "scripts" / "cesvima" / "submit_train_first.sh"


class _Tokens:
    def __init__(self, truncated: bool = False) -> None:
        self.truncated = truncated
        self.saw_truncation_true = False

    def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False, **_kwargs):
        del text, add_special_tokens
        if truncation:
            self.saw_truncation_true = True
            raise AssertionError("production tokenization must not request truncation")
        if self.truncated:
            return {"input_ids": [1], "truncation": True}
        return {"input_ids": [1, 2, 3, 4]}


def _snapshot(root: Path, *, name: str = MODEL_ID, with_bin: bool = False) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": name,
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "published-template"}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"not-a-real-weight")
    if with_bin:
        (root / "pytorch_model.bin").write_bytes(b"pickle")
    return root


def _restore_approval(previous: frozenset[str]) -> None:
    import re_te_system.controlled_experiment.gemma4 as gemma4

    gemma4.APPROVED_REVISIONS = previous


def test_identity_rejects_base_and_unapproved_candidate() -> None:
    assert_instruction_tuned(MODEL_ID)
    try:
        assert_instruction_tuned(BASE_MODEL_ID)
    except GemmaModelIdentityError:
        pass
    else:
        raise AssertionError("base checkpoint was accepted")
    assert CANDIDATE_REVISION not in APPROVED_REVISIONS
    try:
        assert_revision_approved(CANDIDATE_REVISION)
    except GemmaRevisionNotApproved as exc:
        assert "candidate" in str(exc)
    else:
        raise AssertionError("candidate revision was approved")
    report = readiness_report()
    assert report["model_id"] == MODEL_ID
    assert report["candidate_revision_approved"] is False
    assert report["gold_inference_authorized"] is False
    assert report["ready_for_gpu_reservation"] is False
    assert report["weight_estimate"]["status"] == "estimate_not_measurement"
    assert report["weight_estimate"]["bf16_weight_bytes"] == ESTIMATED_BF16_WEIGHT_BYTES
    assert report["dtype_policy"]["quantization"] == "refused"


def test_acquisition_does_not_call_a_downloader_for_the_candidate() -> None:
    called: list[str] = []

    def downloader(model_id: str, revision: str) -> Path:
        called.append(revision)
        raise AssertionError(model_id)

    os.environ[G01_ENV] = "1"
    try:
        acquire_checkpoint(MODEL_ID, CANDIDATE_REVISION, Path("/tmp/unused"), downloader)
    except GemmaAcquireNotAuthorized:
        pass
    else:
        raise AssertionError("candidate acquisition was permitted")
    finally:
        os.environ.pop(G01_ENV, None)
    assert called == []


def test_publish_is_refused_twice_and_bins_are_not_loadable(tmp_path: Path = None) -> None:
    import re_te_system.controlled_experiment.gemma4 as gemma4

    if tmp_path is None:
        raise AssertionError("tmp_path missing")
    source = _snapshot(tmp_path / "source")
    refused = verify_snapshot_directory(_snapshot(tmp_path / "base", name=BASE_MODEL_ID))
    assert refused["verified"] is False
    binary = verify_snapshot_directory(_snapshot(tmp_path / "bin", with_bin=True))
    assert any("pytorch_model.bin" in item for item in binary["errors"])
    previous = gemma4.APPROVED_REVISIONS
    gemma4.APPROVED_REVISIONS = frozenset({"abc"})
    try:
        first = publish_snapshot(source, tmp_path / "published" / "abc", "abc")
        assert first["verification"]["verified"] is True
        assert (tmp_path / "published" / "accepted_revision.txt").read_text(encoding="utf-8").strip() == "abc"
        try:
            publish_snapshot(source, tmp_path / "published" / "second", "abc")
        except GemmaDuplicatePublication:
            pass
        else:
            raise AssertionError("second publication was accepted")
    finally:
        _restore_approval(previous)


def test_channel_separation_and_tsv_do_not_repair_output() -> None:
    separated = transport_and_parse(CHANNEL_OPEN + "hidden\ttriple\tok" + CHANNEL_CLOSE + "subj\trel\tobj")
    assert separated["triples"] == [{"subject": "subj", "relation": "rel", "object": "obj"}]
    assert "hidden" not in json.dumps(separated["triples"])
    malformed = transport_and_parse(CHANNEL_OPEN + "subj\trel\tobj")
    assert malformed["triples"] == []
    assert malformed["issues"][0]["code"] == "CHANNEL_NOT_SEPARATED"
    invalid = transport_and_parse("only-two\tfields")
    assert invalid["triples"] == []
    truncated = transport_and_parse(
        "subj\trel\tobj",
        generation_metadata={"output_reached_limit": True},
    )
    assert truncated["triples"] == []
    assert truncated["truncation"] == "TRUNCATED_GENERATION"
    assert separate_official_channels("subj\trel\tobj")["status"] == "answer_only"


def test_serialization_uses_shell_without_thinking_or_truncation() -> None:
    text = "Invented preflight sentence."
    rendered = render_controlled_prompt("C0", text)
    serialized = serialize_text_turn(rendered, enable_thinking=False)
    assert rendered in serialized
    assert serialized.endswith(CHANNEL_OPEN + CHANNEL_CLOSE)
    assert render_serialized_turn("C1", text).count(text) == 1
    try:
        serialize_text_turn(rendered, enable_thinking=True)
    except GemmaThinkingNotAuthorized:
        pass
    else:
        raise AssertionError("thinking mode was accepted")
    try:
        render_serialized_turn("C0", "datasets/teresia_hohfeld/canonical/documents.jsonl")
    except GemmaGoldAccessRefused:
        pass
    else:
        raise AssertionError("gold path was rendered")
    counter = _Tokens()
    from re_te_system.controlled_experiment.gemma4 import count_tokens_untruncated

    assert count_tokens_untruncated(counter, "abcd") == 4
    assert counter.saw_truncation_true is False
    try:
        count_tokens_untruncated(_Tokens(truncated=True), "abcd")
    except GemmaSilentTruncationRefused:
        pass
    else:
        raise AssertionError("truncated tokenization was accepted")


def test_budget_timeout_and_runtime_classification() -> None:
    try:
        assess_prompt_budget(
            "prompt",
            count_tokens=lambda _text: 100,
            effective_context_limit=40,
            reserved_output_tokens=32,
            tokenizer_id=MODEL_ID,
            tokenizer_revision=None,
            counter_id="test-counter",
        )
    except GemmaContextBudgetError:
        pass
    else:
        raise AssertionError("over-budget prompt was accepted")
    fitted = assess_prompt_budget(
        "prompt",
        count_tokens=lambda _text: 4,
        effective_context_limit=64,
        reserved_output_tokens=32,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=None,
        counter_id="test-counter",
    )
    assert fitted["fits"] is True
    try:
        run_bounded_command([sys.executable, "-c", "import time; time.sleep(30)"], 0.2)
    except GemmaWorkerTimeout:
        pass
    else:
        raise AssertionError("timeout was not raised")
    failure = classify_runtime_failure(RuntimeError("CUDA out of memory"))
    assert failure["kind"] == "external_infrastructure_error"
    assert failure["parsed_as_triple"] == "false"
    assert scientific_decoding_contract().blockers()
    assert preflight_decoding_contract().blockers() == ()
    assert preflight_decoding_contract().eos_token_id is None
    plan = preflight_plan()
    assert plan["weights_loaded"] is False
    assert plan["fixture"]["uses_gold"] is False
    manifest = technical_manifest(plan)
    assert manifest["manifest_sha256"] == technical_manifest(plan)["manifest_sha256"]
    assert manifest["gold_inference_authorized"] is False


def test_preflight_loader_is_not_called_without_approval(tmp_path: Path = None) -> None:
    if tmp_path is None:
        raise AssertionError("tmp_path missing")
    called: list[bool] = []

    def loader(**_kwargs):
        called.append(True)
        raise AssertionError("loader")

    os.environ[G02_ENV] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        execute_bounded_preflight(
            revision=CANDIDATE_REVISION,
            allocation_text="gpu:l40s:1",
            loader=loader,
            snapshot_dir=_snapshot(tmp_path / "snap"),
        )
    except GemmaRevisionNotApproved:
        pass
    else:
        raise AssertionError("unapproved preflight loaded weights")
    finally:
        os.environ.pop(G02_ENV, None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    assert called == []
    os.environ[G02_ENV] = "1"
    try:
        execute_bounded_preflight(
            revision="a" * 40,
            allocation_text="gpu:1",
            loader=loader,
            snapshot_dir=tmp_path / "missing",
        )
    except GemmaRevisionNotApproved:
        pass
    finally:
        os.environ.pop(G02_ENV, None)
    assert called == []
    import re_te_system.controlled_experiment.gemma4 as gemma4

    approved = "b" * 40
    previous = gemma4.APPROVED_REVISIONS
    gemma4.APPROVED_REVISIONS = frozenset({approved})
    os.environ[G02_ENV] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    accepted: list[dict] = []

    def accepted_loader(**kwargs):
        accepted.append(kwargs)
        return {"device": "not-loaded-in-test"}

    try:
        loaded = execute_bounded_preflight(
            revision=approved,
            allocation_text="gpu:l40s:1",
            loader=accepted_loader,
            snapshot_dir=tmp_path / "snap",
        )
    finally:
        gemma4.APPROVED_REVISIONS = previous
        os.environ.pop(G02_ENV, None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    assert accepted[0]["local_files_only"] is True
    assert accepted[0]["quantization"] is None
    assert loaded["weights_loaded"] is True


def _fake_gemma_venv(directory: Path, *, version: str | None, status: int) -> Path:
    python = directory / "bin" / "python"
    python.parent.mkdir(parents=True)
    if version is None:
        python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    else:
        python.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\nexit {status}\n", encoding="utf-8")
    python.chmod(0o755)
    return directory


def test_g00_separates_runtime_compatibility_from_scientific_gaps(tmp_path: Path = None) -> None:
    if tmp_path is None:
        raise AssertionError("tmp_path missing")
    saved = os.environ.get("GEMMA_VENV")
    os.environ.pop("GEMMA_VENV", None)
    try:
        absent = readiness_report()
        assert absent["ready_for_gpu_reservation"] is False
        assert absent["runtime_blockers"] == ["gemma_venv_absent"]
        assert "candidate_revision_not_approved" in absent["scientific_blockers"]

        os.environ["GEMMA_VENV"] = str(_fake_gemma_venv(tmp_path / "old", version="4.57.6", status=0))
        old = readiness_report()
        assert old["ready_for_gpu_reservation"] is False
        assert old["runtime_blockers"] == ["isolated_transformers_incompatible"]

        os.environ["GEMMA_VENV"] = str(_fake_gemma_venv(tmp_path / "missing-class", version=None, status=1))
        missing = readiness_report()
        assert missing["ready_for_gpu_reservation"] is False
        assert missing["runtime_blockers"] == ["gemma4_unified_class_not_importable"]

        os.environ["GEMMA_VENV"] = str(_fake_gemma_venv(tmp_path / "compatible", version="5.10.1", status=0))
        compatible = readiness_report()
        assert compatible["ready_for_gpu_reservation"] is True
        assert compatible["runtime_blockers"] == []
        assert compatible["scientific_ready"] is False
        assert compatible["candidate_revision_approved"] is False
        assert compatible["eos_token_id"] is None and compatible["pad_token_id"] is None
        assert compatible["vram_bytes_measured"] is None
        assert "vram_unverified" in compatible["scientific_blockers"]
        assert "eos_pad_ids_unverified" in compatible["scientific_blockers"]
        assert compatible["authorizes_later_stages"] == {
            "G01": False,
            "G02": False,
            "G03": False,
            "G04": False,
        }
        from re_te_system.controlled_experiment.gemma4 import acquisition_plan, launch_g04

        assert acquisition_plan(MODEL_ID, CANDIDATE_REVISION)["acquire_permitted"] is False
        try:
            launch_g04({})
        except Exception as exc:
            assert "g04_phase_lock_closed" in str(exc)
        else:
            raise AssertionError("compatible G00 opened G04")
        sys.path.insert(0, str(SCRIPTS))
        import g00_gemma_readiness

        os.environ["SCRATCH_ROOT"] = str(tmp_path)
        os.environ["GEMMA_G00_READINESS_AUTHORIZED"] = "1"
        exit_code = g00_gemma_readiness.main(["--output", str(tmp_path / "compatible.json")])
        assert exit_code == 0
        written = json.loads((tmp_path / "compatible.json").read_text(encoding="utf-8"))
        assert written["ready_for_gpu_reservation"] is True
        assert written["scientific_ready"] is False
        assert written["authorizes_later_stages"]["G04"] is False
        os.environ.pop("GEMMA_G00_READINESS_AUTHORIZED", None)
    finally:
        if saved is None:
            os.environ.pop("GEMMA_VENV", None)
        else:
            os.environ["GEMMA_VENV"] = saved


def test_commands_and_registry_stay_closed() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import g00_gemma_readiness
    import g01_gemma_acquire
    import g02_gemma_preflight

    submitter = SUBMITTER.read_text(encoding="utf-8")
    for name in ("g00_gemma_readiness.sbatch", "g01_gemma_acquire.sbatch", "g02_gemma_preflight.sbatch"):
        text = (ROOT / "jobs" / "cesvima" / name).read_text(encoding="utf-8")
        assert "--gres=gpu:1" not in text
        assert name not in submitter
    assert "gpu:l40s:1" in (ROOT / "jobs" / "cesvima" / "g02_gemma_preflight.sbatch").read_text(encoding="utf-8")
    scratch = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT", "/tmp")) 
    # The caller passes a real temporary directory through the function argument when present.
    del scratch
    catalog = json.loads((ROOT / "campaigns" / "exp3c" / "plans" / "conditioning.json").read_text(encoding="utf-8"))
    tasks = {item["id"]: item for item in catalog["tasks"]}
    assert tasks["G00"]["execution_permitted"] is False
    assert tasks["G01"]["preparation_status"] == "BLOCKED"
    assert tasks["G02"]["execution_permitted"] is False
    assert tasks["G02"]["dependencies"] == []
    assert "snapshot_download" not in Path(g00_gemma_readiness.__file__).read_text(encoding="utf-8")
    assert "Gemma4UnifiedForConditionalGeneration" in Path(g02_gemma_preflight.__file__).read_text(encoding="utf-8")
    g01_source = Path(g01_gemma_acquire.__file__).read_text(encoding="utf-8")
    assert g01_source.index("acquisition_plan") < g01_source.index("snapshot_download")


def test_cli_refuses_gold_and_candidate_download(tmp_path: Path = None) -> None:
    if tmp_path is None:
        raise AssertionError("tmp_path missing")
    sys.path.insert(0, str(SCRIPTS))
    import g00_gemma_readiness
    import g01_gemma_acquire
    import g02_gemma_preflight

    os.environ["SCRATCH_ROOT"] = str(tmp_path)
    os.environ.pop("GEMMA_VENV", None)
    os.environ["GEMMA_G00_READINESS_AUTHORIZED"] = "1"
    try:
        try:
            g00_gemma_readiness.main(
                ["--output", str(tmp_path / "g00.json"), "--documents", "datasets/teresia_hohfeld/canonical/documents.jsonl"]
            )
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("gold documents were accepted by G00")
        g00_gemma_readiness.main(["--output", str(tmp_path / "ready.json")])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("G00 reported GPU readiness")
    report = json.loads((tmp_path / "ready.json").read_text(encoding="utf-8"))
    assert report["candidate_revision"] == CANDIDATE_REVISION
    assert report["ready_for_gpu_reservation"] is False
    called: list[str] = []

    def downloader(model_id: str, revision: str) -> Path:
        called.append(model_id)
        raise AssertionError(revision)

    os.environ[G01_ENV] = "1"
    try:
        g01_gemma_acquire.main(
            [
                "--model-id",
                MODEL_ID,
                "--revision",
                CANDIDATE_REVISION,
                "--destination",
                str(tmp_path / "dest"),
                "--report",
                str(tmp_path / "g01.json"),
            ],
            downloader=downloader,
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("G01 downloaded the candidate")
    assert called == []
    try:
        g02_gemma_preflight.main(["--output", str(tmp_path / "g02.json"), "--revision", CANDIDATE_REVISION])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("G02 executed without --execute")
    plan = json.loads((tmp_path / "g02.json").read_text(encoding="utf-8"))
    assert plan["executed"] is False
    assert plan["weights_loaded"] is False
    os.environ.pop("GEMMA_G00_READINESS_AUTHORIZED", None)
    os.environ.pop(G01_ENV, None)
    os.environ.pop("SCRATCH_ROOT", None)


def _run_all() -> None:
    import tempfile

    os.environ.pop("GEMMA_VENV", None)
    test_identity_rejects_base_and_unapproved_candidate()
    test_acquisition_does_not_call_a_downloader_for_the_candidate()
    test_channel_separation_and_tsv_do_not_repair_output()
    test_serialization_uses_shell_without_thinking_or_truncation()
    test_budget_timeout_and_runtime_classification()
    test_commands_and_registry_stay_closed()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("publish", "preflight", "cli"):
            (root / name).mkdir()
        test_publish_is_refused_twice_and_bins_are_not_loadable(root / "publish")
        test_preflight_loader_is_not_called_without_approval(root / "preflight")
        test_g00_separates_runtime_compatibility_from_scientific_gaps(root / "g00")
        test_cli_refuses_gold_and_candidate_download(root / "cli")


if __name__ == "__main__":
    _run_all()
    print("gemma g00-g02 tests passed")
