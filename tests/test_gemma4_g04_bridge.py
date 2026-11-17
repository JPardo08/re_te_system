"""Synthetic G04 to Evaluation Bridge contract. No Gold release, no weights, no GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from re_te_system.controlled_experiment.dependencies import ensure_package

ensure_package("re_te_benchmark", "re_te_benchmark")
ensure_package("re_te_evaluation", "re_te_evaluation")

from re_te_evaluation import (  # noqa: E402
    ContextMode,
    EligibilityPolicy,
    EvaluationConfig,
    MetricFamily,
    evaluate_triplets,
)
from re_te_evaluation.eligibility import read_records
from re_te_system.controlled_experiment.contracts import sha256_text
from re_te_system.controlled_experiment.gemma4 import (
    CANDIDATE_REVISION,
    G04_PHASE_LOCK,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaGoldNotAuthorized,
    GemmaWorkerTimeout,
    contrast_c1_minus_c0,
    g04_provenance,
    launch_g04,
    load_g04_cohort,
    require_g04_decoding,
)
from re_te_system.controlled_experiment.gemma4_bridge import (
    attach_g04_bridge_view,
    evaluate_g04_synthetic,
    weights_are_verified,
)
from re_te_system.controlled_experiment.persistence import atomic_write_json
from re_te_system.manifests.run_manifest import sha256_file


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


def _cohort_rows(count: int) -> list[dict[str, str]]:
    return [
        {"document_id": f"articulo_{index}", "source_text": f"Synthetic statute {index}."}
        for index in range(count, 0, -1)
    ]


def _write_cohort(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "documents.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _clean_repositories(head: str) -> dict[str, dict[str, object]]:
    return {
        name: {"commit_identifies_tree": True, "dirty": False, "head": head}
        for name in ("re_te_benchmark", "re_te_evaluation", "re_te_system")
    }


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


ROOT = Path(__file__).resolve().parents[1]


def _gold(document_id: str, number: int) -> dict:
    return {
        "document_id": document_id,
        "eligibility": {"complete_triplet_strict_inline": {"eligible": True, "reason": "synthetic"}},
        "id": f"gold-{number}",
        "object": f"O{number}",
        "relation": "Rel",
        "subject": f"S{number}",
    }


def _line(number: int, *, marker: bool = True) -> dict:
    raw = f"S{number}\tRel\tO{number}"
    if marker:
        raw += "<turn|>"
    return {"raw": raw, "truncated": False}


def _outcomes() -> list:
    rows = []
    for number in range(1, 37):
        good = _line(number)
        if number == 21:
            rows.extend([{"raw": "", "truncated": False}, good])
        elif number == 22:
            rows.extend([{"raw": "not tsv", "truncated": False}, good])
        elif number == 23:
            rows.extend([{"raw": f"S{number}\tRel\tO{number}", "truncated": True}, good])
        elif number == 24:
            rows.extend([GemmaWorkerTimeout("budget"), good, good])
        elif number == 25:
            rows.extend([RuntimeError("parse internally"), good])
        else:
            rows.extend([good, good])
    return rows


def _factory(session: _FakeSession, loads: list[int]):
    def load() -> _FakeSession:
        loads.append(1)
        return session

    return load


def _p0(gold: list[dict], predictions: list) -> dict:
    scored = evaluate_triplets(
        gold,
        predictions,
        EvaluationConfig(MetricFamily.TRIPLET, ContextMode.DOCUMENT, EligibilityPolicy.STRICT_INLINE),
    )
    return scored.summary


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _state(campaign: Path, document_id: str, condition: str) -> dict:
    matches = [
        row
        for row in _rows(campaign / "document_states.jsonl")
        if row["document_id"] == document_id and row["condition"] == condition
    ]
    assert len(matches) == 1
    return matches[0]


def test_bridge_scores_a_synthetic_paired_cohort_without_opening_g04(tmp_path: Path) -> None:
    assert G04_PHASE_LOCK is True
    sbatch = (ROOT / "jobs" / "cesvima" / "g04_gemma_campaign.sbatch").read_text(encoding="utf-8")
    entry = (ROOT / "scripts" / "cesvima" / "g04_gemma_campaign.py").read_text(encoding="utf-8")
    assert "unset GEMMA_G04_RUN_AUTHORIZED" in sbatch
    assert "run_g04_paired_core" not in entry
    assert "evaluate_g04_synthetic" not in entry
    assert "attach_g04_bridge_view" not in entry
    try:
        launch_g04({})
    except GemmaGoldNotAuthorized as exc:
        assert "g04_phase_lock_closed" in str(exc)
    else:
        raise AssertionError("phase lock allowed production launch")

    documents = _write_cohort(tmp_path / "cohort", _cohort_rows(36))
    digest = sha256_file(documents)
    snapshot = _accepted_snapshot(tmp_path / "models", CANDIDATE_REVISION)
    loads: list[int] = []
    session = _FakeSession(_outcomes())
    clean = _clean_repositories("a" * 40)
    result = run_campaign(
        documents,
        snapshot,
        tmp_path / "campaign",
        session,
        loads,
        repositories=clean,
        declared=digest,
    )
    campaign = tmp_path / "campaign"
    assert loads == [1]
    assert session.generated == 73
    assert session.policies == [(False, 40)] * 73
    assert result["statistics"]["request_count"] == 72
    assert result["statistics"]["experimental_f1"] is False
    assert result["statistics"]["g04_authorized"] is False
    assert result["manifest"]["scientific_result"] is False
    assert result["bridge"]["campaign_scientific_id"] != result["manifest"]["run_id"]
    assert weights_are_verified(result["manifest"]["identity"]["snapshot_identity"]) is False
    assert "model_weights_sha256_not_computed" in result["bridge"]["manifest"]["operational"]["freeze_gaps"]

    producer = _rows(campaign / "predictions.jsonl")
    assert [(row["document_id"], row["condition"]) for row in producer][0] == ("articulo_1", "C0")
    assert len(producer) == 72
    success = producer[0]
    assert success["raw"].endswith("<turn|>")
    assert success["valid_prediction"] is True
    assert success["parsed_triples"][0]["object"] == "O1"
    assert "<turn|>" not in (campaign / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8")
    attempts = _rows(campaign / "attempts.jsonl")
    first = next(row for row in attempts if row["document_id"] == "articulo_1" and row["condition"] == "C0")
    assert first["raw"].endswith("<turn|>")
    assert first["model_output"] == "S1\tRel\tO1"
    assert first["removed_terminal_marker"] == "<turn|>"
    assert first["transport_policy"] == "gemma4-one-terminal-turn-end"

    empty = _state(campaign, "articulo_21", "C0")
    assert empty["principal_state"] == "completed"
    assert empty["diagnostics"]["completed_empty"] is True
    assert "external_infrastructure_error" not in empty["principal_state"]
    empty_row = next(row for row in producer if row["document_id"] == "articulo_21" and row["condition"] == "C0")
    assert empty_row["generation_state"] == "empty_model_output"
    assert empty_row["valid_prediction"] is False
    assert empty_row["raw"] == ""

    malformed = _state(campaign, "articulo_22", "C0")
    assert malformed["principal_state"] == "completed"
    assert malformed["diagnostics"]["fully_malformed"] is True
    assert malformed["parsed_triples"] == 0

    truncated = _state(campaign, "articulo_23", "C0")
    truncated_row = next(row for row in producer if row["document_id"] == "articulo_23" and row["condition"] == "C0")
    assert truncated["principal_state"] == "completed"
    assert truncated["diagnostics"]["output_truncated"] is True
    assert truncated["parsed_triples"] == 1
    assert truncated_row["generation_state"] == "truncated"
    assert truncated_row["valid_prediction"] is False
    assert truncated_row["protocol_principal_state"] == "completed"
    assert truncated_row["raw"] == "S23\tRel\tO23"
    exported_23 = [
        row
        for row in _rows(campaign / "conditions" / "C0" / "evaluation_predictions.jsonl")
        if row["document_id"] == "articulo_23"
    ]
    assert exported_23[0]["subject"] == "S23"
    assert "<turn|>" not in exported_23[0]["object"]
    producer_coverage = json.loads((campaign / "coverage.json").read_text(encoding="utf-8"))
    assert producer_coverage["articulo_23"]["C0"]["triple_count"] == 1
    assert producer_coverage["articulo_23"]["C0"]["triple_count"] == len(truncated_row["parsed_triples"])
    assert producer_coverage["articulo_23"]["C0"]["triple_count"] == len(exported_23)
    assert producer_coverage["articulo_23"]["C0"]["generation_state"] == "truncated"
    assert producer_coverage["articulo_23"]["C0"]["valid_prediction"] is False

    retried = [row for row in attempts if row["document_id"] == "articulo_24" and row["condition"] == "C0"]
    assert [row["principal_state"] for row in retried] == ["timeout", "completed"]
    assert _state(campaign, "articulo_24", "C0")["also_observed"][0]["principal_state"] == "timeout"
    failed = _state(campaign, "articulo_25", "C0")
    assert failed["principal_state"] == "model_execution_error"
    assert "diagnostics" not in failed

    bytes_claim = result["bridge"]["manifest"]["operational"]["document_bytes"]
    derived = campaign / "bridge_documents.jsonl"
    assert bytes_claim["canonical_sha256"] == digest
    assert bytes_claim["derived_sha256"] == sha256_file(derived)
    assert bytes_claim["canonical_sha256"] != bytes_claim["derived_sha256"]
    assert bytes_claim["texts_identical"] is True
    source = {json.loads(line)["document_id"]: json.loads(line)["source_text"] for line in documents.read_text(encoding="utf-8").splitlines()}
    derived_rows = _rows(derived)
    assert [row["document_id"] for row in derived_rows] == [f"articulo_{number}" for number in range(1, 37)]
    assert all(row["text"] == source[row["document_id"]] for row in derived_rows)
    assert "source_text" not in derived.read_text(encoding="utf-8")

    gold = [_gold(f"articulo_{number}", number) for number in range(1, 31)]
    assert "articulo_31" not in {row["document_id"] for row in gold}
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold), encoding="utf-8")
    report = evaluate_g04_synthetic(
        campaign,
        gold_path=gold_path,
        output_dir=tmp_path / "bridge-report",
        documents_path=derived,
        protocol_path=Path(result["bridge"]["protocol_path"]),
    )
    assert report["blockers"] == [], report["blockers"]
    eligibility = report["publication_eligibility"]
    assert eligibility["scientific_result"] is False
    assert eligibility["experimental_f1"] is False
    assert eligibility["comparability"] == "not_comparable"
    text = (tmp_path / "bridge-report" / "campaign_evaluation_report.json").read_text(encoding="utf-8")
    assert '"experimental_f1":true' not in text
    assert report["estimand_selection"]["selected_estimand"] is None
    coverage = report["coverage"]["conditions"]
    assert coverage["C0"]["completed"] == 35
    assert coverage["C0"]["principal_states"]["model_execution_error"] == 1
    assert coverage["C0"]["empty_generations"] == 1
    assert coverage["C0"]["fully_malformed"] == 1
    assert coverage["C0"]["truncated_outputs"] == 1
    assert coverage["C1"]["completed"] == 36
    assert report["coverage"]["same_cohort_across_conditions"] is True

    conservative = report["estimands"]["conservative"]
    conditional = report["estimands"]["conditional"]
    paired = report["estimands"]["paired"]
    assert conservative["publication"] == "synthetic_contract_only"
    assert conservative["not_the_cohort"] is False
    assert conditional["publication"] == "synthetic_contract_only"
    assert conditional["not_the_cohort"] is True
    assert paired["publication"] == "synthetic_contract_only"
    assert paired["not_the_cohort"] is True
    assert conservative["conditions"]["C0"]["document_count"] == 36
    assert conditional["conditions"]["C0"]["document_count"] == 35
    assert "articulo_25" not in conditional["conditions"]["C0"]["document_ids"]
    assert conditional["conditions"]["C1"]["document_count"] == 36
    assert paired["document_ids"] == conditional["conditions"]["C0"]["document_ids"]
    assert any(
        item["document_id"] == "articulo_25" and "C0:model_execution_error" in item["reasons"]
        for item in paired["exclusions"]
    )
    assert "slurm_job_id" not in json.dumps(result["bridge"]["manifest"]["identity"])
    for estimand in (conservative, conditional, paired):
        assert estimand["experimental_f1"] is False
        for condition in estimand["conditions"].values():
            assert condition["experimental_f1"] is False
            assert condition["publication"] == "synthetic_contract_only"
    c0_export = read_records(campaign / "conditions" / "C0" / "evaluation_predictions.jsonl")
    direct = _p0(gold, c0_export)
    assert conservative["conditions"]["C0"]["counts"] == direct["counts"]
    assert conservative["conditions"]["C0"]["micro"] == direct["micro"]
    selected = set(conditional["conditions"]["C0"]["document_ids"])
    conditional_direct = _p0(
        [row for row in gold if row["document_id"] in selected],
        [row for row in c0_export if row["document_id"] in selected],
    )
    assert conditional["conditions"]["C0"]["counts"] == conditional_direct["counts"]
    assert conditional["conditions"]["C0"]["counts"] != conservative["conditions"]["C0"]["counts"]
    withheld = contrast_c1_minus_c0(
        {
            "comparability": "not_comparable",
            "completeness": "not_a_scientific_campaign",
            "condition": "C0",
            "estimand": "conservative_operational_f1",
            "experimental_f1": False,
            "model": "google/gemma-4-12B-it",
        },
        {
            "comparability": "not_comparable",
            "completeness": "not_a_scientific_campaign",
            "condition": "C1",
            "estimand": "conservative_operational_f1",
            "experimental_f1": False,
            "model": "google/gemma-4-12B-it",
        },
    )
    assert withheld["published"] is False

    contaminated = tmp_path / "contaminated"
    shutil.copytree(campaign, contaminated)
    export = contaminated / "conditions" / "C0" / "evaluation_predictions.jsonl"
    mutated = _rows(export)
    mutated[0]["subject"] = "Contaminated"
    export.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in mutated), encoding="utf-8")
    digest_c0 = sha256_text(export.read_text(encoding="utf-8"))
    manifest = json.loads((contaminated / "campaign_manifest.json").read_text(encoding="utf-8"))
    manifest["operational"]["export_sha256"]["C0"] = digest_c0
    summary = json.loads((contaminated / "synthetic_evaluation_summary.json").read_text(encoding="utf-8"))
    for payload in summary["estimands"].values():
        payload["conditions"]["C0"]["export_sha256"] = digest_c0
    atomic_write_json(contaminated / "campaign_manifest.json", manifest)
    atomic_write_json(contaminated / "synthetic_evaluation_summary.json", summary)
    contaminated_report = evaluate_g04_synthetic(
        contaminated,
        gold_path=gold_path,
        output_dir=tmp_path / "contaminated-report",
        documents_path=contaminated / "bridge_documents.jsonl",
        protocol_path=Path(result["bridge"]["protocol_path"]),
    )
    assert "exporter_contract_violation" in contaminated_report["blockers"]
    assert contaminated_report["publication_eligibility"]["scientific_result"] is False
    assert contaminated_report["estimands"]["conservative"]["publication"] == "withheld_integrity"
    assert contaminated_report["estimands"]["conservative"]["conditions"]["C0"]["micro"] is None


def test_bridge_retains_retries_dirt_and_closed_directories(tmp_path: Path) -> None:
    documents = _write_cohort(tmp_path / "one", _cohort_rows(1))
    snapshot = _accepted_snapshot(tmp_path / "models", CANDIDATE_REVISION)
    clean = _clean_repositories("c" * 40)
    loads: list[int] = []
    timeout = _FakeSession(
        [GemmaWorkerTimeout("budget"), GemmaWorkerTimeout("budget"), {"raw": "S1\tRel\tO1", "truncated": False}]
    )
    timeout_result = run_campaign(
        documents,
        snapshot,
        tmp_path / "timeout",
        timeout,
        loads,
        repositories=clean,
        expected_count=1,
    )
    assert timeout.generated == 3
    states = _rows(tmp_path / "timeout" / "document_states.jsonl")
    assert states[0]["principal_state"] == "timeout"
    assert states[0]["attempt_count"] == 2
    assert [row["principal_state"] for row in _rows(tmp_path / "timeout" / "attempts.jsonl")[:2]] == ["timeout", "timeout"]
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(_gold("articulo_1", 1)) + "\n", encoding="utf-8")
    timeout_report = evaluate_g04_synthetic(
        tmp_path / "timeout",
        gold_path=gold,
        output_dir=tmp_path / "timeout-report",
        documents_path=tmp_path / "timeout" / "bridge_documents.jsonl",
        protocol_path=Path(timeout_result["bridge"]["protocol_path"]),
    )
    assert timeout_report["blockers"] == [], timeout_report["blockers"]
    assert timeout_report["campaign_status"]["recomputed"]["status"] == "incomplete_external_or_interrupted"
    assert timeout_report["estimands"]["conservative"]["publication"] == "withheld_campaign_incomplete"
    assert timeout_report["estimands"]["conditional"]["publication"] == "withheld_campaign_incomplete"
    assert timeout_report["estimands"]["paired"]["publication"] == "withheld_campaign_incomplete"
    assert timeout_report["estimands"]["conservative"]["conditions"]["C0"]["micro"] is None

    external = _FakeSession(
        [
            RuntimeError("cuda error: synthetic"),
            RuntimeError("cuda error: synthetic"),
            {"raw": "S1\tRel\tO1", "truncated": False},
        ]
    )
    external_result = run_campaign(
        documents,
        snapshot,
        tmp_path / "external",
        external,
        [],
        repositories=clean,
        expected_count=1,
    )
    assert _state(tmp_path / "external", "articulo_1", "C0")["principal_state"] == "external_infrastructure_error"
    external_report = evaluate_g04_synthetic(
        tmp_path / "external",
        gold_path=gold,
        output_dir=tmp_path / "external-report",
        documents_path=tmp_path / "external" / "bridge_documents.jsonl",
        protocol_path=Path(external_result["bridge"]["protocol_path"]),
    )
    assert external_report["estimands"]["paired"]["publication"] == "withheld_campaign_incomplete"
    assert external_report["publication_eligibility"]["experimental_f1"] is False

    dirty = _clean_repositories("d" * 40)
    dirty["re_te_system"] = {"commit_identifies_tree": False, "dirty": True, "head": "d" * 40}
    dirty_result = run_campaign(
        documents,
        snapshot,
        tmp_path / "dirty",
        _FakeSession([{"raw": "S1\tRel\tO1<turn|>", "truncated": False}] * 2),
        [],
        repositories=dirty,
        expected_count=1,
    )
    dirty_report = evaluate_g04_synthetic(
        tmp_path / "dirty",
        gold_path=gold,
        output_dir=tmp_path / "dirty-report",
        documents_path=tmp_path / "dirty" / "bridge_documents.jsonl",
        protocol_path=Path(dirty_result["bridge"]["protocol_path"]),
    )
    assert any("re_te_system_not_a_clean_tree" in reason for reason in dirty_report["scientific_reasons"])
    assert dirty_report["publication_eligibility"]["scientific_result"] is False
    other = run_campaign(
        documents,
        snapshot,
        tmp_path / "other-commit",
        _FakeSession([{"raw": "S1\tRel\tO1<turn|>", "truncated": False}] * 2),
        [],
        repositories=_clean_repositories("e" * 40),
        expected_count=1,
    )
    assert other["bridge"]["campaign_scientific_id"] != dirty_result["bridge"]["campaign_scientific_id"]
    assert other["manifest"]["run_id"] != dirty_result["manifest"]["run_id"]

    partial = tmp_path / "partial"
    partial.mkdir()
    note = partial / "notes.txt"
    note.write_text("interrupted\n", encoding="utf-8")
    calls = {"n": 0}

    def counting() -> _FakeSession:
        calls["n"] += 1
        return _FakeSession([])

    try:
        run_campaign(
            documents,
            snapshot,
            partial,
            _FakeSession([]),
            [],
            repositories=clean,
            expected_count=1,
            session_factory=counting,
        )
    except GemmaDuplicatePublication as exc:
        assert "partial" in str(exc)
    else:
        raise AssertionError("partial campaign directory was reused")
    assert calls["n"] == 0
    assert note.read_text(encoding="utf-8") == "interrupted\n"
    try:
        run_campaign(
            documents,
            snapshot,
            tmp_path / "timeout",
            _FakeSession([]),
            [],
            repositories=clean,
            expected_count=1,
            session_factory=counting,
        )
    except GemmaDuplicatePublication as exc:
        assert "complete" in str(exc)
    else:
        raise AssertionError("complete campaign directory was overwritten")
    assert calls["n"] == 0

    cohort = load_g04_cohort(documents, expected_count=1)
    decoding = require_g04_decoding(max_new_tokens=40)
    provenance = g04_provenance(
        cohort=cohort,
        decoding=decoding,
        revision=CANDIDATE_REVISION,
        snapshot_dir=snapshot,
        repositories=clean,
    )
    omitted = tmp_path / "omitted"
    attach_g04_bridge_view(
        omitted,
        attempts=[
            {
                "attempt_count": 1,
                "condition": "C0",
                "document_id": "articulo_1",
                "raw": "S1\tRel\tO1",
                "state": "completed",
                "truncated": False,
            }
        ],
        cohort=cohort,
        decoding=decoding,
        documents_source=documents,
        provenance=provenance,
        revision=CANDIDATE_REVISION,
    )
    assert _state(omitted, "articulo_1", "C1")["principal_state"] == "not_attempted"
    omitted_report = evaluate_g04_synthetic(
        omitted,
        gold_path=gold,
        output_dir=tmp_path / "omitted-report",
        documents_path=omitted / "bridge_documents.jsonl",
        protocol_path=Path(timeout_result["bridge"]["protocol_path"]),
    )
    assert omitted_report["blockers"] == [], omitted_report["blockers"]
    assert omitted_report["estimands"]["conservative"]["publication"] == "withheld_campaign_not_publishable"
    assert omitted_report["estimands"]["conditional"]["conditions"]["C1"]["publication"] == "withheld_empty_subset"
    assert omitted_report["estimands"]["paired"]["publication"] == "withheld_empty_intersection"
    assert omitted_report["publication_eligibility"]["scientific_result"] is False
    try:
        evaluate_g04_synthetic(
            omitted,
            gold_path=Path("datasets/teresia_hohfeld/canonical/annotations.jsonl"),
            output_dir=tmp_path / "real-gold",
            documents_path=omitted / "bridge_documents.jsonl",
        )
    except GemmaGoldAccessRefused:
        pass
    else:
        raise AssertionError("real Gold path was accepted")


def test_channels_partial_parse_and_bridge_is_published_once(tmp_path: Path) -> None:
    from re_te_system.controlled_experiment.gemma4 import CHANNEL_CLOSE, CHANNEL_OPEN, TURN_END

    documents = _write_cohort(tmp_path / "one", _cohort_rows(1))
    snapshot = _accepted_snapshot(tmp_path / "models", CANDIDATE_REVISION)
    clean = _clean_repositories("f" * 40)
    partial_raw = f"Ava\tDuty\tBlake\nnot tsv{TURN_END}"
    channel_raw = f"{CHANNEL_OPEN}secret thought\n{CHANNEL_CLOSE}Ava\tDuty\tBlake{TURN_END}"
    broken_channel = f"{CHANNEL_OPEN}{CHANNEL_OPEN}Ava\tDuty\tBlake{TURN_END}"
    campaign = tmp_path / "shapes"
    result = run_campaign(
        documents,
        snapshot,
        campaign,
        _FakeSession(
            [
                {"raw": partial_raw, "truncated": False},
                {"raw": channel_raw, "truncated": False},
            ]
        ),
        [],
        repositories=clean,
        expected_count=1,
    )
    partial = _state(campaign, "articulo_1", "C0")
    assert partial["principal_state"] == "completed"
    assert partial["diagnostics"]["partially_malformed"] is True
    assert partial["parsed_triples"] == 1
    partial_attempt = next(
        row
        for row in _rows(campaign / "attempts.jsonl")
        if row["document_id"] == "articulo_1" and row["condition"] == "C0"
    )
    assert partial_attempt["raw"] == partial_raw
    assert partial_attempt["model_output"] == "Ava\tDuty\tBlake\nnot tsv"
    assert partial_attempt["raw"] != partial_attempt["model_output"]
    exported_partial = _rows(campaign / "conditions" / "C0" / "evaluation_predictions.jsonl")
    assert [(row["subject"], row["relation"], row["object"]) for row in exported_partial] == [
        ("Ava", "Duty", "Blake")
    ]
    coverage = json.loads((campaign / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["articulo_1"]["C0"]["triple_count"] == len(exported_partial)

    separated = _state(campaign, "articulo_1", "C1")
    assert separated["principal_state"] == "completed"
    assert separated["parsed_triples"] == 1
    separated_attempt = next(
        row
        for row in _rows(campaign / "attempts.jsonl")
        if row["document_id"] == "articulo_1" and row["condition"] == "C1"
    )
    assert separated_attempt["raw"] == channel_raw
    assert separated_attempt["model_output"] == "Ava\tDuty\tBlake"
    assert CHANNEL_OPEN in separated_attempt["raw"]
    assert CHANNEL_OPEN not in separated_attempt["model_output"]
    exported_channel = (campaign / "conditions" / "C1" / "evaluation_predictions.jsonl").read_text(
        encoding="utf-8"
    )
    assert "Ava" in exported_channel
    assert "secret thought" not in exported_channel
    assert CHANNEL_OPEN not in exported_channel
    assert CHANNEL_CLOSE not in exported_channel
    assert coverage["articulo_1"]["C1"]["triple_count"] == 1

    manifest_hash = sha256_file(campaign / "campaign_manifest.json")
    export_hash = sha256_file(campaign / "conditions" / "C1" / "evaluation_predictions.jsonl")
    try:
        attach_g04_bridge_view(
            campaign,
            attempts=[],
            cohort=load_g04_cohort(documents, expected_count=1),
            decoding=require_g04_decoding(max_new_tokens=40),
            documents_source=documents,
            provenance=g04_provenance(
                cohort=load_g04_cohort(documents, expected_count=1),
                decoding=require_g04_decoding(max_new_tokens=40),
                revision=CANDIDATE_REVISION,
                snapshot_dir=snapshot,
                repositories=clean,
            ),
            revision=CANDIDATE_REVISION,
        )
    except GemmaDuplicatePublication as exc:
        assert "bridge publication" in str(exc)
    else:
        raise AssertionError("a second bridge publication replaced the first")
    assert sha256_file(campaign / "campaign_manifest.json") == manifest_hash
    assert sha256_file(campaign / "conditions" / "C1" / "evaluation_predictions.jsonl") == export_hash

    partial_dir = tmp_path / "partial-bridge"
    partial_dir.mkdir()
    evidence = partial_dir / "attempts.jsonl"
    evidence.write_text("keep-this-evidence\n", encoding="utf-8")
    evidence_hash = sha256_file(evidence)
    try:
        attach_g04_bridge_view(
            partial_dir,
            attempts=[],
            cohort=load_g04_cohort(documents, expected_count=1),
            decoding=require_g04_decoding(max_new_tokens=40),
            documents_source=documents,
            provenance=g04_provenance(
                cohort=load_g04_cohort(documents, expected_count=1),
                decoding=require_g04_decoding(max_new_tokens=40),
                revision=CANDIDATE_REVISION,
                snapshot_dir=snapshot,
                repositories=clean,
            ),
            revision=CANDIDATE_REVISION,
        )
    except GemmaDuplicatePublication as exc:
        assert "attempts.jsonl" in str(exc)
    else:
        raise AssertionError("a partial bridge directory was adopted")
    assert evidence.read_text(encoding="utf-8") == "keep-this-evidence\n"
    assert sha256_file(evidence) == evidence_hash
    assert not (partial_dir / "campaign_manifest.json").exists()

    forbidden = tmp_path / "synthetic-marker" / "teresia_hohfeld" / "documents.jsonl"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"\xff")
    calls = {"n": 0}

    def counting() -> _FakeSession:
        calls["n"] += 1
        return _FakeSession([])

    try:
        run_campaign(
            forbidden,
            snapshot,
            tmp_path / "forbidden-out",
            _FakeSession([]),
            [],
            repositories=clean,
            expected_count=1,
            session_factory=counting,
        )
    except GemmaGoldAccessRefused as exc:
        assert "Hohfeld" in str(exc)
    else:
        raise AssertionError("a forbidden document path was read")
    assert calls["n"] == 0
    assert not (tmp_path / "forbidden-out").exists()
    assert forbidden.read_bytes() == b"\xff"

    refused = tmp_path / "malformed-channel"
    run_campaign(
        documents,
        snapshot,
        refused,
        _FakeSession(
            [
                {"raw": broken_channel, "truncated": False},
                {"raw": "Ava\tDuty\tBlake", "truncated": False},
            ]
        ),
        [],
        repositories=clean,
        expected_count=1,
    )
    refused_attempt = next(
        row
        for row in _rows(refused / "attempts.jsonl")
        if row["condition"] == "C0"
    )
    assert refused_attempt["raw"] == broken_channel
    assert refused_attempt["model_output"] == ""
    assert any(issue["code"] == "CHANNEL_NOT_SEPARATED" for issue in refused_attempt["parse_issues"])
    refused_export = (refused / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(
        encoding="utf-8"
    )
    assert refused_export == ""
    assert "Ava" not in refused_export
    assert CHANNEL_OPEN not in refused_export
    assert result["statistics"]["g04_authorized"] is False
    assert result["manifest"]["gold_inference_authorized"] is False
    assert result["manifest"]["scientific_result"] is False


def run_campaign(documents, snapshot, output, session, loads, **kwargs):
    from re_te_system.controlled_experiment.gemma4 import run_g04_paired_core

    factory = kwargs.pop("session_factory", None)
    return run_g04_paired_core(
        documents_path=documents,
        snapshot_dir=snapshot,
        output_dir=output,
        max_new_tokens=40,
        session_factory=factory or _factory(session, loads),
        repositories=kwargs.pop("repositories"),
        declared_documents_sha256=kwargs.pop("declared", None),
        expected_count=kwargs.pop("expected_count", 36),
    )


def _run_all() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_bridge_scores_a_synthetic_paired_cohort_without_opening_g04(root / "main")
        test_bridge_retains_retries_dirt_and_closed_directories(root / "edges")
        test_channels_partial_parse_and_bridge_is_published_once(root / "shapes")


if __name__ == "__main__":
    _run_all()
    print("gemma g04 bridge tests passed")
