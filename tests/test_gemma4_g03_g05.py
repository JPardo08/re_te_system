"""G03-G05 local checks. No GPU, no Hohfeld files, no weight load."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from re_te_system.controlled_experiment.gemma4 import (
    CANDIDATE_REVISION,
    EXPECTED_DOCUMENT_COUNT,
    G03_ENV,
    G04_ENV,
    SERIALIZATION_ID,
    GemmaAllocationError,
    GemmaChannelError,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaGoldNotAuthorized,
    GemmaPreflightNotAuthorized,
    GemmaRevisionNotApproved,
    GemmaWorkerTimeout,
    persist_g03,
    run_invented_smoke,
    accept_document_records,
    assert_paired_cohort,
    classify_smoke_output,
    contrast_c1_minus_c0,
    evaluation_table,
    g04_blockers,
    launch_g04,
    paired_condition_identity,
    same_retry_policy,
    scientific_plan,
    score_levels,
    smoke_requests,
    smoke_technical_pass,
    write_smoke_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"


def _gold(document_id: str, subject: str, relation: str, object_: str, identifier: str, *, eligible: bool = True) -> dict:
    return {
        "document_id": document_id,
        "eligibility": {"complete_triplet_strict_inline": {"eligible": eligible, "reason": "synthetic"}},
        "id": identifier,
        "object": object_,
        "relation": relation,
        "subject": subject,
    }


def _prediction(document_id: str, subject: str, relation: str, object_: str, identifier: str) -> dict:
    return {
        "document_id": document_id,
        "id": identifier,
        "object": object_,
        "relation": relation,
        "subject": subject,
    }


def attempts_for_persist() -> list[dict]:
    return [
        _attempt(document_id, condition, "Ava\tDuty\tBlake")
        for document_id in ("invented-gemma-01", "invented-gemma-02")
        for condition in ("C0", "C1")
    ]


def _attempt(document_id: str, condition: str, raw: str, *, state: str = "completed", within: bool = True) -> dict:
    return {
        "condition": condition,
        "document_id": document_id,
        "exception": None if state == "completed" else state,
        "raw": raw,
        "rendered_prompt_sha256": "prompt",
        "serialized_sha256": "serialized",
        "state": state,
        "truncated": False,
        "within_budget": within,
    }


def test_shell_slots_and_paired_documents() -> None:
    text = "Ava notifies Blake about one invented duty."
    identity = paired_condition_identity(text)
    assert identity["shell_id"] == "common-instruction-shell-v2"
    assert identity["labels"] == ["Duty", "Right", "Privilege", "NoRight"]
    assert identity["c0_slot_sha256"] != identity["c1_slot_sha256"]
    assert identity["same_decoding_for_both_conditions"] is True
    requests = smoke_requests()
    c0 = [item["document_id"] for item in requests if item["condition"] == "C0"]
    c1 = [item["document_id"] for item in requests if item["condition"] == "C1"]
    assert_paired_cohort(c0, c1)
    assert all(item["uses_gold"] is False for item in requests)
    try:
        assert_paired_cohort(["a", "b"], ["b", "a"])
    except GemmaGoldAccessRefused:
        pass
    else:
        raise AssertionError("reordered cohort was accepted")


def test_parser_channels_budget_and_retries() -> None:
    valid = classify_smoke_output("Ava\tDuty\tBlake")
    assert valid["valid_prediction"] is True
    assert valid["parsed"]["triples"][0]["relation"] == "Duty"
    empty = classify_smoke_output("")
    assert empty["valid_prediction"] is False
    assert empty["failure"]["kind"] == "empty_model_output"
    malformed = classify_smoke_output("<|channel>thought\nAva\tDuty\tBlake")
    assert malformed["parsed"]["triples"] == []
    timed = classify_smoke_output("Ava\tDuty\tBlake", state="timeout")
    oom = classify_smoke_output("Ava\tDuty\tBlake", state="external_infrastructure_error")
    assert timed["failure"]["kind"] == "timeout"
    assert oom["failure"]["kind"] == "external_infrastructure_error"
    assert timed["parsed"]["triples"] == []
    assert same_retry_policy("C0", "timeout", 1) is True
    assert same_retry_policy("C1", "timeout", 1) is True
    assert same_retry_policy("C0", "timeout", 2) is False
    assert same_retry_policy("C1", "completed", 1) is False
    from re_te_system.controlled_experiment.gemma4 import GemmaContextBudgetError, assess_prompt_budget

    try:
        assess_prompt_budget(
            "prompt",
            count_tokens=lambda _text: 100,
            effective_context_limit=10,
            reserved_output_tokens=32,
            tokenizer_id="google/gemma-4-12B-it",
            tokenizer_revision=None,
            counter_id="test-counter",
        )
    except GemmaContextBudgetError:
        pass
    else:
        raise AssertionError("short context was accepted")


def test_smoke_artifacts_and_g04_closure(tmp_path: Path) -> None:
    attempts = [
        _attempt(document_id, condition, "Ava\tDuty\tBlake")
        for document_id in ("invented-gemma-01", "invented-gemma-02")
        for condition in ("C0", "C1")
    ]
    assert smoke_technical_pass(attempts, 1) is True
    assert smoke_technical_pass(attempts, 2) is False
    written = write_smoke_artifacts(tmp_path / "smoke", attempts, load_count=1)
    assert written["statistics"]["technical_pass"] is True
    assert written["statistics"]["scientific_result"] is False
    assert written["statistics"]["g04_authorized"] is False
    for name in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        assert (tmp_path / "smoke" / name).is_file()
    try:
        write_smoke_artifacts(tmp_path / "smoke", attempts, load_count=1)
    except GemmaDuplicatePublication:
        pass
    else:
        raise AssertionError("smoke artifacts were overwritten")
    try:
        accept_document_records([{"document_id": "a", "text": "x", "annotations": []}])
    except GemmaGoldAccessRefused:
        pass
    else:
        raise AssertionError("annotations were accepted as generation input")
    plan = scientific_plan()
    assert plan["documents_read"] is False
    assert plan["document_count_expected"] == EXPECTED_DOCUMENT_COUNT == 36
    assert plan["gold_inference_authorized"] is False
    called: list[bool] = []

    def loader():
        called.append(True)

    os.environ[G04_ENV] = "1"
    try:
        launch_g04({"human_authorization": {}, "output_safe": False}, loader=loader)
    except GemmaGoldNotAuthorized as exc:
        assert "human_authorization_missing" in str(exc)
        assert "g04_phase_lock_closed" in str(exc)
        assert "execution_grant_missing" not in str(exc)
    else:
        raise AssertionError("environment grant launched G04")
    finally:
        os.environ.pop(G04_ENV, None)
    assert called == []
    blockers = g04_blockers(
        {
            "frozen_decoding": {"c0": "same", "c1": "same"},
            "g02_report": {"status": "PREFLIGHT_PASS"},
            "g03_report": {"g04_authorized": False, "scientific_result": False, "technical_pass": True},
            "human_authorization": {"actor": "reviewer", "decision": "authorize_g04", "revision": "a" * 40},
            "output_safe": True,
            "revision_approved": True,
            "runtime_compatible": True,
            "snapshot_verified": True,
        }
    )
    assert "g04_phase_lock_closed" in blockers
    assert "human_authorization_missing" in blockers


def test_relation_triplet_and_withheld_contrast() -> None:
    gold = [_gold("d1", "Ava", "Duty", "Blake", "g1")]
    exact = [_prediction("d1", "Ava", "Duty", "Blake", "p1")]
    levels = score_levels(gold, exact)
    assert levels["experimental_f1"] is False
    assert levels["relation"]["tp"] == 1 and levels["triplet"]["tp"] == 1
    assert levels["relation"]["fp"] == 0 and levels["triplet"]["fn"] == 0
    duplicated = exact + [_prediction("d1", "Ava", "Duty", "Blake", "p2")]
    extra = score_levels(gold, duplicated)
    assert extra["triplet"]["tp"] == 1 and extra["triplet"]["fp"] == 1
    missing = score_levels(gold, [])
    assert missing["triplet"]["fn"] == 1 and missing["triplet"]["tp"] == 0
    false_positive = score_levels([], exact)
    assert false_positive["triplet"]["fp"] == 1 and false_positive["triplet"]["fn"] == 0
    ineligible = score_levels([_gold("d1", "Ava", "Duty", "Blake", "g1", eligible=False)], exact)
    assert ineligible["triplet"]["fn"] == 0
    assert ineligible["triplet"]["fp"] == 1
    row = evaluation_table(
        "google/gemma-4-12B-it",
        "C0",
        levels,
        estimand="conservative",
        completeness="not_a_scientific_campaign",
        coverage={"documents": 1},
    )
    assert row["estimand"] == "conservative_operational_f1"
    assert row["relation_precision"] == 1
    assert row["triplet_precision"] == 1
    assert row["experimental_f1"] is False
    same_condition = contrast_c1_minus_c0(row, row)
    assert same_condition["published"] is False
    assert "one C0 row and one C1 row" in same_condition["reason"]
    contrast = contrast_c1_minus_c0(row, {**row, "condition": "C1"})
    assert contrast["published"] is False
    other = evaluation_table(
        "google/gemma-4-12B-it",
        "C1",
        levels,
        estimand="paired",
        completeness="not_a_scientific_campaign",
        coverage={"documents": 1},
    )
    assert other["estimand"] == "paired_secondary_f1"
    assert other["estimand"] != row["estimand"]


def test_commands_do_not_launch(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import g03_gemma_invented_smoke
    import g04_gemma_campaign
    import g05_gemma_evaluate

    os.environ["SCRATCH_ROOT"] = str(tmp_path)
    called: list[bool] = []
    try:
        g03_gemma_invented_smoke.main(["--output", str(tmp_path / "g03.json")])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("G03 executed without a request")
    plan = json.loads((tmp_path / "g03.json").read_text(encoding="utf-8"))
    assert plan["weights_loaded"] is False and plan["g04_authorized"] is False
    os.environ[G03_ENV] = "1"
    try:
        g03_gemma_invented_smoke.main(
            ["--execute", "--output-dir", str(tmp_path / "denied"), "--revision", "a" * 40],
            loader=lambda: called.append(True),
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("unapproved G03 loaded weights")
    finally:
        os.environ.pop(G03_ENV, None)
    assert called == []
    import re_te_system.controlled_experiment.gemma4 as gemma4

    previous = gemma4.APPROVED_REVISIONS
    gemma4.APPROVED_REVISIONS = frozenset({"b" * 40})
    os.environ[G03_ENV] = "1"
    try:
        g03_gemma_invented_smoke.main(
            [
                "--execute",
                "--output-dir",
                str(tmp_path / "g03-run"),
                "--revision",
                "b" * 40,
            ],
            loader=lambda: {
                "attempts": attempts_for_persist(),
                "g04_authorized": False,
                "load_count": 1,
                "scientific_result": False,
            },
        )
    finally:
        gemma4.APPROVED_REVISIONS = previous
        os.environ.pop(G03_ENV, None)
    manifest = json.loads((tmp_path / "g03-run" / "manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((tmp_path / "g03-run" / "statistics.json").read_text(encoding="utf-8"))
    assert manifest["report"]["g04_authorized"] is False
    assert statistics["g04_authorized"] is False
    assert statistics["technical_pass"] is True
    assert (tmp_path / "g03-run" / "predictions.jsonl").is_file()
    try:
        g04_gemma_campaign.main(
            ["--output", str(tmp_path / "g04.json"), "--annotations", str(tmp_path / "annotations.jsonl")]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("annotations were accepted")
    assert not (tmp_path / "g04.json").exists()
    try:
        g04_gemma_campaign.main(["--output", str(tmp_path / "g04.json")], loader=lambda: called.append(True))
    except SystemExit as exc:
        assert exc.code == 2
    refusal = json.loads((tmp_path / "g04.json").read_text(encoding="utf-8"))
    assert refusal["documents_read"] is False
    assert "g04_phase_lock_closed" in refusal["refusal"]
    assert called == []
    gold = tmp_path / "synthetic-gold.jsonl"
    predictions = tmp_path / "synthetic-predictions.jsonl"
    gold.write_text(json.dumps(_gold("d1", "Ava", "Duty", "Blake", "g1")) + "\n", encoding="utf-8")
    predictions.write_text(
        json.dumps(_prediction("d1", "Ava", "Duty", "Blake", "p1")) + "\n",
        encoding="utf-8",
    )
    g05_gemma_evaluate.main(
        [
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--condition",
            "C0",
            "--output",
            str(tmp_path / "g05.json"),
        ]
    )
    table = json.loads((tmp_path / "g05.json").read_text(encoding="utf-8"))
    assert table["experimental_f1"] is False
    assert table["contrast"]["published"] is False
    assert "single condition" in table["contrast"]["reason"]
    try:
        g05_gemma_evaluate.main(
            [
                "--gold",
                "datasets/teresia_hohfeld/canonical/annotations.jsonl",
                "--predictions",
                str(predictions),
                "--condition",
                "C0",
                "--output",
                str(tmp_path / "blocked.json"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Hohfeld annotations were scored")
    submitter = (ROOT / "scripts" / "cesvima" / "submit_train_first.sh").read_text(encoding="utf-8")
    for name in (
        "g03_gemma_invented_smoke.sbatch",
        "g04_gemma_campaign.sbatch",
        "g05_gemma_evaluate.sbatch",
    ):
        text = (ROOT / "jobs" / "cesvima" / name).read_text(encoding="utf-8")
        assert "--gres=gpu:1\n" not in text and "--gres=gpu:1 " not in text
        assert name not in submitter
    assert "gpu:l40s:1" in (ROOT / "jobs" / "cesvima" / "g03_gemma_invented_smoke.sbatch").read_text(encoding="utf-8")
    assert "--gres=" not in (ROOT / "jobs" / "cesvima" / "g04_gemma_campaign.sbatch").read_text(encoding="utf-8")
    os.environ.pop("SCRATCH_ROOT", None)


L40S_ALLOCATION = (
    "AllocTRES=billing=82000,cpu=8,gres/gpu:l40s=1,gres/gpu=1,mem=64G,node=1 "
    "ReqTRES=billing=82000,cpu=8,gres/gpu:l40s=1,gres/gpu=1,mem=64G,node=1 "
    "TresPerNode=gres:gpu:l40s:1"
)


def _accepted_snapshot(root: Path, revision: str) -> Path:
    snapshot = root / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "google/gemma-4-12B-it",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "published-template"}),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"synthetic-not-a-weight")
    (root / "accepted_revision.txt").write_text(revision + "\n", encoding="utf-8")
    return snapshot


class _FakeSession:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.generated = 0
        self.policies: list[tuple[bool, int]] = []
        self.bos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        from re_te_system.controlled_experiment.gemma4 import serialize_text_turn

        assert kwargs.get("tokenize") is False
        assert kwargs.get("add_generation_prompt") is True
        assert kwargs.get("enable_thinking") is False
        return serialize_text_turn(messages[0]["content"], enable_thinking=False)

    def __call__(self, *, text: str, add_special_tokens: bool = True, truncation: bool = False, **_kwargs):
        assert add_special_tokens is False
        assert truncation is False
        assert text.startswith("<bos><|turn>user\n")
        assert text.count("<bos>") == 1
        return {"input_ids": [[2, 3, 4, 5]]}

    def generate(self, encoded, *, do_sample: bool, max_new_tokens: int):
        self.generated += 1
        self.policies.append((do_sample, max_new_tokens))
        assert encoded["input_ids"][0][0] == 2 and encoded["input_ids"][0][1] != 2
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _smoke_env() -> None:
    os.environ[G03_ENV] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _clear_smoke_env() -> None:
    os.environ.pop(G03_ENV, None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_invented_smoke_loads_once_and_keeps_failures(tmp_path: Path) -> None:
    snapshot = _accepted_snapshot(tmp_path / "models", CANDIDATE_REVISION)
    loads: list[int] = []

    def factory(session: _FakeSession):
        def load():
            loads.append(1)
            return session

        return load

    _smoke_env()
    try:
        session = _FakeSession([{"raw": "Ava\tDuty\tBlake", "truncated": False}] * 4)
        result = run_invented_smoke(
            revision=CANDIDATE_REVISION,
            snapshot_dir=snapshot,
            allocation_text=L40S_ALLOCATION,
            session_factory=factory(session),
            confirm_device=lambda: "NVIDIA L40S",
        )
        assert loads == [1]
        assert session.generated == 4
        assert session.policies == [(False, 32)] * 4
        assert [(item["document_id"], item["condition"]) for item in result["attempts"]] == [
            ("invented-gemma-01", "C0"),
            ("invented-gemma-01", "C1"),
            ("invented-gemma-02", "C0"),
            ("invented-gemma-02", "C1"),
        ]
        assert result["load_count"] == 1
        assert result["g04_authorized"] is False
        assert result["scientific_result"] is False
        assert result["serialization_id"] == SERIALIZATION_ID
        assert result["decoding"] == {
            "do_sample": False,
            "max_new_tokens": 32,
            "same_for_c0_and_c1": True,
        }
        written = persist_g03(tmp_path / "smoke", result)
        assert written["statistics"]["technical_pass"] is True
        assert written["statistics"]["g04_authorized"] is False
        assert written["manifest"]["report"]["technical_smoke"] is True
        assert written["manifest"]["report"]["revision"] == CANDIDATE_REVISION
        try:
            persist_g03(tmp_path / "smoke", result)
        except GemmaDuplicatePublication:
            pass
        else:
            raise AssertionError("smoke artifacts were overwritten")

        loads.clear()
        empty = _FakeSession(
            [{"raw": "", "truncated": False}]
            + [{"raw": "Ava\tDuty\tBlake", "truncated": False}] * 3
        )
        empty_result = run_invented_smoke(
            revision=CANDIDATE_REVISION,
            snapshot_dir=snapshot,
            allocation_text=L40S_ALLOCATION,
            session_factory=factory(empty),
            confirm_device=lambda: "NVIDIA L40S",
        )
        persist_g03(tmp_path / "empty", empty_result)
        failures = [
            json.loads(line)
            for line in (tmp_path / "empty" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert failures[0]["kind"] == "empty_model_output"
        assert failures[0]["document_id"] == "invented-gemma-01"
        predictions = [
            json.loads(line)
            for line in (tmp_path / "empty" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert predictions[0]["valid_prediction"] is False
        assert predictions[0]["parsed_triples"] == []

        limited = _FakeSession([{"raw": "Ava\tDuty\tBlake", "truncated": True}] * 4)
        limited_result = run_invented_smoke(
            revision=CANDIDATE_REVISION,
            snapshot_dir=snapshot,
            allocation_text=L40S_ALLOCATION,
            session_factory=factory(limited),
            confirm_device=lambda: "NVIDIA L40S",
        )
        persist_g03(tmp_path / "limit", limited_result)
        limit_failures = [
            json.loads(line)
            for line in (tmp_path / "limit" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {item["kind"] for item in limit_failures} == {"token_limit"}
        assert all(item["valid_prediction"] is False for item in [
            json.loads(line)
            for line in (tmp_path / "limit" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ])

        broken = _FakeSession(
            [{"raw": "Ava\tDuty\tBlake", "truncated": False}] * 2
            + [RuntimeError("boom")]
            + [{"raw": "Ava\tDuty\tBlake", "truncated": False}]
        )
        broken_result = run_invented_smoke(
            revision=CANDIDATE_REVISION,
            snapshot_dir=snapshot,
            allocation_text=L40S_ALLOCATION,
            session_factory=factory(broken),
            confirm_device=lambda: "NVIDIA L40S",
        )
        assert broken.generated == 4
        assert broken_result["attempts"][2]["state"] == "model_execution_error"
        assert broken_result["attempts"][2]["exception"] == "boom"
        persist_g03(tmp_path / "broken", broken_result)
        broken_failures = [
            json.loads(line)
            for line in (tmp_path / "broken" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert broken_failures[0]["kind"] == "model_execution_error"
        assert json.loads((tmp_path / "broken" / "statistics.json").read_text(encoding="utf-8"))[
            "technical_pass"
        ] is False

        retried = _FakeSession(
            [GemmaWorkerTimeout("budget")]
            + [{"raw": "Ava\tDuty\tBlake", "truncated": False}] * 4
        )
        retried_result = run_invented_smoke(
            revision=CANDIDATE_REVISION,
            snapshot_dir=snapshot,
            allocation_text=L40S_ALLOCATION,
            session_factory=factory(retried),
            confirm_device=lambda: "NVIDIA L40S",
        )
        assert retried.generated == 5
        assert retried_result["attempts"][0]["attempt_count"] == 2
        assert retried_result["attempts"][0]["prior_failure"]["state"] == "timeout"
        assert retried_result["load_count"] == 1
        refused = factory(_FakeSession([]))
        for revision, allocation, device, marker in (
            ("a" * 40, L40S_ALLOCATION, "NVIDIA L40S", CANDIDATE_REVISION),
            (CANDIDATE_REVISION, "AllocTRES=cpu=8,gres/gpu:a100=1", "NVIDIA L40S", CANDIDATE_REVISION),
            (CANDIDATE_REVISION, "AllocTRES=cpu=8,gres/gpu=1", "NVIDIA L40S", CANDIDATE_REVISION),
            (CANDIDATE_REVISION, L40S_ALLOCATION, "NVIDIA A100-SXM4-40GB", CANDIDATE_REVISION),
            (CANDIDATE_REVISION, L40S_ALLOCATION, "NVIDIA H100", CANDIDATE_REVISION),
        ):
            loads.clear()
            if marker != CANDIDATE_REVISION:
                (snapshot.parent / "accepted_revision.txt").write_text(marker + "\n", encoding="utf-8")
            try:
                run_invented_smoke(
                    revision=revision,
                    snapshot_dir=snapshot,
                    allocation_text=allocation,
                    session_factory=refused,
                    confirm_device=lambda name=device: name,
                )
            except (GemmaAllocationError, GemmaRevisionNotApproved):
                pass
            else:
                raise AssertionError(f"smoke accepted {revision} on {device}")
            assert loads == []
        (snapshot.parent / "accepted_revision.txt").write_text("other-revision\n", encoding="utf-8")
        try:
            run_invented_smoke(
                revision=CANDIDATE_REVISION,
                snapshot_dir=snapshot,
                allocation_text=L40S_ALLOCATION,
                session_factory=refused,
                confirm_device=lambda: "NVIDIA L40S",
            )
        except GemmaRevisionNotApproved as exc:
            assert "acceptance marker" in str(exc)
        else:
            raise AssertionError("wrong acceptance marker was loaded")
        assert loads == []
        os.environ.pop(G03_ENV, None)
        try:
            run_invented_smoke(
                revision=CANDIDATE_REVISION,
                snapshot_dir=snapshot,
                allocation_text=L40S_ALLOCATION,
                session_factory=refused,
                confirm_device=lambda: "NVIDIA L40S",
            )
        except GemmaPreflightNotAuthorized as exc:
            assert "grant" in str(exc)
        else:
            raise AssertionError("missing grant loaded weights")
        os.environ[G03_ENV] = "1"
        os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            run_invented_smoke(
                revision=CANDIDATE_REVISION,
                snapshot_dir=snapshot,
                allocation_text=L40S_ALLOCATION,
                session_factory=refused,
                confirm_device=lambda: "NVIDIA L40S",
            )
        except GemmaPreflightNotAuthorized as exc:
            assert "HF_HUB_OFFLINE" in str(exc)
        else:
            raise AssertionError("online smoke was accepted")
        os.environ["HF_HUB_OFFLINE"] = "1"
        (snapshot.parent / "accepted_revision.txt").write_text(CANDIDATE_REVISION + "\n", encoding="utf-8")

        class _Mismatch(_FakeSession):
            def apply_chat_template(self, messages, **kwargs):
                del messages, kwargs
                return "not-the-official-template"

        loads.clear()
        try:
            run_invented_smoke(
                revision=CANDIDATE_REVISION,
                snapshot_dir=snapshot,
                allocation_text=L40S_ALLOCATION,
                session_factory=factory(_Mismatch([])),
                confirm_device=lambda: "NVIDIA L40S",
            )
        except GemmaChannelError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("template mismatch generated text")
        assert loads == [1]
    finally:
        _clear_smoke_env()
    sys.path.insert(0, str(SCRIPTS))
    import g03_gemma_invented_smoke

    os.environ["SCRATCH_ROOT"] = str(tmp_path)
    try:
        g03_gemma_invented_smoke.main(
            [
                "--output",
                str(tmp_path / "gold.json"),
                "--documents",
                "datasets/teresia_hohfeld/canonical/documents.jsonl",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("gold documents were accepted by G03")
    sbatch = (ROOT / "jobs" / "cesvima" / "g03_gemma_invented_smoke.sbatch").read_text(encoding="utf-8")
    assert "GEMMA_SNAPSHOT_DIR" in sbatch
    assert "--snapshot-dir" in sbatch
    assert "GEMMA_G04_RUN_AUTHORIZED=1" not in sbatch
    os.environ.pop("SCRATCH_ROOT", None)
    try:
        launch_g04({})
    except GemmaGoldNotAuthorized as exc:
        assert "g04_phase_lock_closed" in str(exc)
    else:
        raise AssertionError("G03 smoke opened G04")


def _run_all() -> None:
    test_shell_slots_and_paired_documents()
    test_parser_channels_budget_and_retries()
    test_relation_triplet_and_withheld_contrast()
    with tempfile.TemporaryDirectory() as directory:
        test_smoke_artifacts_and_g04_closure(Path(directory) / "smoke")
        test_commands_do_not_launch(Path(directory) / "cli")
        test_invented_smoke_loads_once_and_keeps_failures(Path(directory) / "invented")


if __name__ == "__main__":
    _run_all()
    print("gemma g03-g05 tests passed")
