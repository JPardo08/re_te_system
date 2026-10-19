"""EXP.3a synthetic harness tests.

Fixtures are invented. This module does not read Hohfeld Gold, does not call a
model, and does not treat a scripted TSV string as a model prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

from re_te_system.controlled_experiment.dependencies import ensure_package

ensure_package("re_te_benchmark", "re_te_benchmark")
ensure_package("re_te_evaluation", "re_te_evaluation")

from re_te_benchmark.conditioning.c3 import C2_PAYLOAD_SHA256  # noqa: E402
from re_te_benchmark.conditioning.shell import (
    CONDITIONS,
    SHELL_ID,
    conditioning_for,
    render_condition,
    shell_template,
)
from re_te_system.controlled_experiment.campaign import (
    CampaignConfig,
    SyntheticDocument,
    authorize_campaign,
    render_controlled_prompt,
    run_controlled_campaign,
)
from re_te_system.controlled_experiment.contracts import (
    REAL_C3_CONDITIONS,
    SYNTHETIC_CONTRACT_TEST,
    CampaignError,
    PreflightNotAssessed,
    RealC3CampaignRejected,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import repository_root_from_package
from re_te_system.controlled_experiment.preflight import PreflightRequest, assess_preflight
from re_te_system.controlled_experiment.synthetic import (
    SYNTHETIC_COUNTER_ID,
    ScriptedTsvExtractor,
    ScriptedTurn,
    SyntheticTokenCounter,
)


ALPHA = "Alpha supplies Beta and Alpha supplies Gamma."
TWO_LINES = "Alpha\tsupplies\tBeta\nAlpha\tsupplies\tGamma\n"
REPEATED = "Alpha supplies Beta. Alpha supplies Beta."


def _counter() -> SyntheticTokenCounter:
    return SyntheticTokenCounter()


def _document(document_id: str, text: str = ALPHA) -> SyntheticDocument:
    return SyntheticDocument(document_id, text)


def _extractor(scripts: dict[tuple[str, str, int], ScriptedTurn]) -> ScriptedTsvExtractor:
    return ScriptedTsvExtractor(scripts)


def _output(text: str, **flags: bool) -> ScriptedTurn:
    return ScriptedTurn("output", output=text, **flags)


def _config(extractor: ScriptedTsvExtractor, **overrides: object) -> CampaignConfig:
    documents = overrides.pop("cohort", (_document("doc-alpha"),))
    conditions = overrides.pop("conditions", ("C0",))
    prompts = [
        render_condition(condition, document.text)
        for condition in conditions
        for document in documents
    ]
    reserved = int(overrides.pop("reserved_output_tokens", 8))
    limit = overrides.pop("effective_context_limit", max(len(prompt) for prompt in prompts) + reserved)
    counter = _counter()
    values = dict(
        conditions=conditions,
        cohort=documents,
        extractor=extractor,
        count_tokens=counter.count_tokens,
        counter_id=counter.counter_id,
        tokenizer_id=counter.tokenizer_id,
        tokenizer_revision=counter.tokenizer_revision,
        effective_context_limit=limit,
        reserved_output_tokens=reserved,
    )
    values.update(overrides)
    return CampaignConfig(**values)


def _states(result: dict, condition: str) -> dict[str, dict]:
    return {
        row["document_id"]: row
        for row in result["document_states"]
        if row["condition"] == condition
    }


def _eligible(document_id: str, subject: str, relation: str, object_: str, identifier: str) -> dict:
    return {
        "id": identifier,
        "document_id": document_id,
        "subject": subject,
        "relation": relation,
        "object": object_,
        "eligibility": {
            "complete_triplet_strict_inline": {
                "eligible": True,
                "reason": "synthetic_fixture",
            }
        },
    }


def test_shell_identity_and_five_condition_slots() -> None:
    assert SHELL_ID == "common-instruction-shell-v2"
    assert CONDITIONS == ("C0", "C1", "C2", "C3", "C3-NL")
    bodies = set()
    slot_hashes = {}
    for condition in CONDITIONS:
        rendered = render_controlled_prompt(condition, ALPHA)
        direct = render_condition(condition, ALPHA)
        assert rendered == direct
        slot = conditioning_for(condition)
        slot_hashes[condition] = sha256_text(slot)
        assert sha256_text(rendered) != slot_hashes[condition]
        if slot:
            assert slot in rendered
            bodies.add(rendered.replace(slot, "[[SLOT]]", 1))
        else:
            bodies.add(rendered.replace("\n\nInput text:\n", "\n[[SLOT]]\nInput text:\n", 1))
        assert rendered.count(ALPHA) == 1
    assert len(bodies) == 1
    assert "relation_labels:" not in render_condition("C0", ALPHA)
    assert slot_hashes["C2"] == C2_PAYLOAD_SHA256
    assert len({*slot_hashes.values()}) == 5


def test_readiness_files_are_not_modified() -> None:
    root = repository_root_from_package("re_te_benchmark")
    assert root is not None
    conditions_path = root / "schemas" / "conditioning_conditions.json"
    package_path = root / "schemas" / "c3" / "c3_ontology_v1.json"
    before = (conditions_path.read_bytes(), package_path.read_bytes())
    result = run_controlled_campaign(
        _config(_extractor({("*", "doc-alpha", 1): _output(TWO_LINES)}), conditions=CONDITIONS)
    )
    assert (conditions_path.read_bytes(), package_path.read_bytes()) == before
    readiness = result["manifest"]["observed_readiness"]
    assert readiness["C2"] == "PARTIAL"
    assert readiness["C3"] == "NOT_READY"
    assert readiness["C3_package_experimental_condition_status"] == "NOT_EXECUTABLE"
    assert result["manifest"]["readiness_mutated"] is False


def test_real_c3_campaign_is_rejected() -> None:
    for condition in REAL_C3_CONDITIONS:
        try:
            authorize_campaign("scientific", (condition,))
        except RealC3CampaignRejected:
            continue
        raise AssertionError(condition)
    authorize_campaign(SYNTHETIC_CONTRACT_TEST, CONDITIONS)
    try:
        authorize_campaign("scientific", ("C0",))
    except CampaignError as exc:
        assert not isinstance(exc, RealC3CampaignRejected)
    else:
        raise AssertionError("a non-synthetic mode must be refused")


def test_preflight_boundaries_and_missing_measurement() -> None:
    counter = _counter()
    prompt = "Alpha supplies Beta."
    exact = assess_preflight(
        PreflightRequest(
            prompt,
            counter.count_tokens,
            len(prompt) + 4,
            4,
            counter.tokenizer_id,
            counter_id=counter.counter_id,
        )
    )
    assert exact["fits"] is True
    assert exact["assurance"] == "synthetic_contract"
    assert exact["model_compatibility"] is False
    over = assess_preflight(
        PreflightRequest(
            prompt,
            counter.count_tokens,
            len(prompt) + 4 - 1,
            4,
            counter.tokenizer_id,
            counter_id=counter.counter_id,
        )
    )
    assert over["fits"] is False
    assert over["required_tokens"] == len(prompt) + 4
    reserve = assess_preflight(
        PreflightRequest(
            prompt,
            counter.count_tokens,
            len(prompt),
            len(prompt),
            counter.tokenizer_id,
            counter_id=counter.counter_id,
        )
    )
    assert reserve["fits"] is False
    assert reserve["reserved_output_tokens"] == len(prompt)
    missing_tokenizer = assess_preflight(
        PreflightRequest(prompt, counter.count_tokens, 100, 4, None, counter_id=counter.counter_id)
    )
    assert missing_tokenizer["assurance"] == "not_assessed"
    assert "fits" not in missing_tokenizer
    assert "tokenizer_id" in missing_tokenizer["missing"]
    missing_counter = assess_preflight(
        PreflightRequest(prompt, None, 100, 4, "tokenizer", counter_id=None)
    )
    assert missing_counter["assurance"] == "not_assessed"
    assert "token_counter" in missing_counter["missing"]
    measured = assess_preflight(
        PreflightRequest(prompt, counter.count_tokens, len(prompt) + 1, 1, "org/tokenizer", counter_id="org/tokenizer")
    )
    assert measured["assurance"] == "measured"
    assert measured["model_compatibility"] is False


def test_context_block_does_not_call_or_retry(tmp_path: Path) -> None:
    extractor = _extractor({("*", "doc-alpha", 1): _output(TWO_LINES)})
    result = run_controlled_campaign(
        _config(extractor, effective_context_limit=1, reserved_output_tokens=1, output_dir=tmp_path)
    )
    assert extractor.calls == []
    state = _states(result, "C0")["doc-alpha"]
    assert state["principal_state"] == "context_blocked"
    assert state["attempt_count"] == 1
    assert "also_observed" not in state
    assert "diagnostics" not in state
    attempt = result["attempts"][0]
    assert attempt["preflight"]["fits"] is False
    assert attempt["prompt_sha256"] == sha256_text(render_condition("C0", ALPHA))
    assert result["prediction_records"] == []
    assert (tmp_path / "evaluation_predictions.jsonl").read_text(encoding="utf-8") == ""


def test_unassessed_preflight_is_not_a_pass() -> None:
    extractor = _extractor({("*", "doc-alpha", 1): _output(TWO_LINES)})
    try:
        run_controlled_campaign(_config(extractor, tokenizer_id=None))
    except PreflightNotAssessed:
        pass
    else:
        raise AssertionError("missing tokenizer must not pass preflight")
    assert extractor.calls == []


def test_generation_states_diagnostics_and_retries(tmp_path: Path) -> None:
    cohort = (
        _document("doc-ok"),
        _document("doc-empty", "Alpha keeps no relation."),
        _document("doc-bad", "Alpha names no object."),
        _document("doc-partial", "Alpha supplies Beta and then fails."),
        _document("doc-trunc", "Alpha supplies Beta."),
        _document("doc-both", "Alpha supplies Beta badly."),
        _document("doc-model", "Alpha fails closed."),
        _document("doc-infra", "Alpha waits once."),
        _document("doc-timeout-ok", "Alpha times out once."),
        _document("doc-timeout-fail", "Alpha times out twice."),
        _document("doc-cut", "Alpha must not lose its prompt."),
        _document("doc-held", "Alpha is not submitted."),
    )
    scripts = {
        ("*", "doc-ok", 1): _output(TWO_LINES),
        ("*", "doc-empty", 1): _output(""),
        ("*", "doc-bad", 1): _output("not a triple\n"),
        ("*", "doc-partial", 1): _output("Alpha\tsupplies\tBeta\nnot a triple\n"),
        ("*", "doc-trunc", 1): _output("Alpha\tsupplies\tBeta\n", output_reached_limit=True),
        ("*", "doc-both", 1): _output("not a triple\n", output_reached_limit=True),
        ("*", "doc-model", 1): ScriptedTurn("model_error", message="extractor raised"),
        ("*", "doc-model", 2): _output("SHOULD\tNOT\tRUN\n"),
        ("*", "doc-infra", 1): ScriptedTurn("infrastructure", message="disk unavailable"),
        ("*", "doc-infra", 2): _output("Alpha\tsupplies\tBeta\n"),
        ("*", "doc-timeout-ok", 1): ScriptedTurn("timeout", message="limit reached"),
        ("*", "doc-timeout-ok", 2): _output("Alpha\tsupplies\tGamma\n"),
        ("*", "doc-timeout-fail", 1): ScriptedTurn("timeout", message="first limit"),
        ("*", "doc-timeout-fail", 2): ScriptedTurn("timeout", message="second limit"),
        ("*", "doc-cut", 1): _output("Alpha\tsupplies\tBeta\n", truncated_input=True),
    }
    result = run_controlled_campaign(
        _config(
            _extractor(scripts),
            cohort=cohort,
            submitted_ids=frozenset(item.document_id for item in cohort if item.document_id != "doc-held"),
            output_dir=tmp_path,
        )
    )
    states = _states(result, "C0")
    assert states["doc-ok"]["principal_state"] == "completed"
    assert "diagnostics" not in states["doc-ok"]
    assert states["doc-empty"]["diagnostics"] == {"completed_empty": True}
    assert "fully_malformed" not in states["doc-empty"]["diagnostics"]
    assert states["doc-bad"]["diagnostics"] == {"fully_malformed": True}
    assert "completed_empty" not in states["doc-bad"]["diagnostics"]
    assert states["doc-partial"]["diagnostics"] == {"partially_malformed": True}
    assert states["doc-trunc"]["diagnostics"] == {"output_truncated": True}
    assert states["doc-both"]["diagnostics"] == {
        "fully_malformed": True,
        "output_truncated": True,
    }
    assert states["doc-model"]["principal_state"] == "model_execution_error"
    assert states["doc-model"]["attempt_count"] == 1
    assert "diagnostics" not in states["doc-model"]
    model_attempt = next(row for row in result["attempts"] if row["document_id"] == "doc-model")
    assert model_attempt["exception_type"] == "SimulatedModelError"
    assert "model_output" not in model_attempt
    assert states["doc-infra"]["principal_state"] == "completed"
    assert states["doc-infra"]["also_observed"] == [
        {"attempt_index": 1, "principal_state": "external_infrastructure_error"}
    ]
    assert states["doc-timeout-ok"]["principal_state"] == "completed"
    assert states["doc-timeout-ok"]["also_observed"][0]["principal_state"] == "timeout"
    assert states["doc-timeout-fail"]["principal_state"] == "timeout"
    assert states["doc-timeout-fail"]["attempt_count"] == 2
    timeout_attempts = [row for row in result["attempts"] if row["document_id"] == "doc-timeout-fail"]
    assert {row["exception_type"] for row in timeout_attempts} == {"SimulatedTimeout"}
    assert {row["prompt_sha256"] for row in timeout_attempts} == {timeout_attempts[0]["prompt_sha256"]}
    assert states["doc-cut"]["diagnostics"]["truncated_input"] is True
    assert states["doc-cut"]["diagnostics"]["silent_input_truncation"] == "operational_incident"
    assert states["doc-held"]["principal_state"] == "not_attempted"
    assert states["doc-held"]["attempt_count"] == 0
    assert "doc-held" not in {row["document_id"] for row in result["prediction_records"]}
    assert "SHOULD" not in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8")
    assert not any(row["document_id"] == "doc-held" and row.get("model_output") is not None for row in result["attempts"])
    infra_attempts = [row for row in result["attempts"] if row["document_id"] == "doc-infra"]
    assert len(infra_attempts) == 2
    assert infra_attempts[0]["prompt_sha256"] == infra_attempts[1]["prompt_sha256"]
    assert "model_output" not in infra_attempts[0]
    assert "Alpha\tsupplies\tBeta\n" == infra_attempts[1]["model_output"]
    exported_ids = [
        json.loads(line)["document_id"]
        for line in (tmp_path / "evaluation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert exported_ids.count("doc-infra") == 1
    assert "doc-timeout-fail" not in exported_ids
    assert "doc-model" not in exported_ids
    assert "doc-empty" not in exported_ids
    coverage = result["coverage"]["conditions"]["C0"]
    assert coverage["denominator"] == len(cohort)
    assert sum(coverage["principal_states"].values()) == len(cohort)
    assert coverage["not_attempted"] == 1


def test_tsv_export_and_synthetic_p0(tmp_path: Path) -> None:
    gold = (
        {
            "id": "syn-multi",
            "document_id": "doc-alpha",
            "relation": "supplies",
            "eligibility": {
                "complete_triplet_strict_inline": {
                    "eligible": True,
                    "reason": "synthetic_fixture",
                }
            },
            "arguments": {
                "e1": {"text": "Alpha"},
                "e2": [{"text": "Beta"}, {"text": "Gamma"}],
            },
        },
        _eligible("doc-repeat", "Alpha", "supplies", "Beta", "syn-repeat"),
        _eligible("doc-extra", "Alpha", "supplies", "Beta", "syn-extra"),
        _eligible("doc-missing", "Alpha", "supplies", "Beta", "syn-missing-1"),
        _eligible("doc-missing", "Alpha", "supplies", "Gamma", "syn-missing-2"),
        _eligible("doc-space", "Alpha", "supplies", "Beta", "syn-space"),
        _eligible("doc-open", "Alpha", "supplies", "Beta", "syn-open"),
    )
    cohort = (
        _document("doc-alpha"),
        _document("doc-repeat", REPEATED),
        _document("doc-extra", "Alpha supplies Beta and owns Dock."),
        _document("doc-missing", ALPHA),
        _document("doc-space", "Alpha has a blank field."),
        _document("doc-open", "Alpha supplies Beta."),
    )
    scripts = {
        ("*", "doc-alpha", 1): _output(TWO_LINES),
        ("*", "doc-repeat", 1): _output("Alpha\tsupplies\tBeta\nAlpha\tsupplies\tBeta\n"),
        ("*", "doc-extra", 1): _output("Alpha\tsupplies\tBeta\nAlpha\towns\tDock\n"),
        ("*", "doc-missing", 1): _output("Alpha\tsupplies\tBeta\n"),
        ("*", "doc-space", 1): _output("Alpha\t \tBeta\n"),
        ("*", "doc-open", 1): _output("Alpha\tsupplies\tBeta\n"),
    }
    result = run_controlled_campaign(
        _config(
            _extractor(scripts),
            cohort=cohort,
            synthetic_gold=gold,
            output_dir=tmp_path,
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_doc: dict[str, list[dict]] = {}
    for row in rows:
        by_doc.setdefault(row["document_id"], []).append(row)
    assert [(row["subject"], row["relation"], row["object"]) for row in by_doc["doc-alpha"]] == [
        ("Alpha", "supplies", "Beta"),
        ("Alpha", "supplies", "Gamma"),
    ]
    assert len(by_doc["doc-repeat"]) == 2
    assert by_doc["doc-open"][0]["relation"] == "supplies"
    assert "Duty" not in by_doc["doc-open"][0]["relation"]
    assert "doc-space" not in by_doc
    assert _states(result, "C0")["doc-space"]["diagnostics"] == {"fully_malformed": True}
    summary = result["evaluation_summary"]
    assert summary["experimental_f1"] is False
    assert summary["result_role"] == "synthetic_contract_test_not_experimental_f1"
    assert summary["counts"]["gold_eligible"] == 7
    assert summary["counts"]["gold_eligible_occurrences"] == 8
    assert summary["counts"]["tp"] == 6
    assert summary["counts"]["fp"] == 2
    assert summary["counts"]["fn"] == 2


def test_same_script_across_conditions_and_stable_identity(tmp_path: Path) -> None:
    other = tmp_path / "other"
    extractor = _extractor({("*", "doc-alpha", 1): _output(TWO_LINES)})
    first = run_controlled_campaign(
        _config(extractor, conditions=CONDITIONS, output_dir=tmp_path / "first")
    )
    second = run_controlled_campaign(
        _config(
            _extractor({("*", "doc-alpha", 1): _output(TWO_LINES)}),
            conditions=CONDITIONS,
            output_dir=other,
        )
    )
    assert first["scientific_id"] == second["scientific_id"]
    assert first["attempts"] == second["attempts"]
    bodies = set()
    slots = set()
    for condition in CONDITIONS:
        state_rows = [row for row in first["document_states"] if row["condition"] == condition]
        assert len(state_rows) == 1
        assert state_rows[0]["principal_state"] == "completed"
        attempt = next(row for row in first["attempts"] if row["condition"] == condition)
        slots.add(attempt["slot_sha256"])
        rendered = render_condition(condition, ALPHA)
        assert attempt["prompt_sha256"] == sha256_text(rendered)
        slot = conditioning_for(condition)
        bodies.add(rendered.replace(slot, "[[SLOT]]", 1) if slot else rendered.replace("\n\nInput text:\n", "\n[[SLOT]]\nInput text:\n", 1))
    assert len(bodies) == 1
    assert len(slots) == 5
    changed_budget = run_controlled_campaign(
        _config(
            _extractor({("*", "doc-alpha", 1): _output(TWO_LINES)}),
            conditions=CONDITIONS,
            reserved_output_tokens=9,
        )
    )
    assert changed_budget["scientific_id"] != first["scientific_id"]
    identity = first["manifest"]["identity"]
    assert identity["repositories"]["re_te_system"]["commit_identifies_tree"] is False
    assert identity["repositories"]["re_te_system"]["head"] == "89cc7fed7399ff22bfb29b4b6c3ff2a18828c08a"
    assert identity["repositories"]["re_te_benchmark"]["head"] == "005cb78a117216e0f2d4ab221c6e1bc7e8baf16f"
    assert identity["repositories"]["re_te_benchmark"]["commit_identifies_tree"] is True
    assert identity["repositories"]["re_te_evaluation"]["head"] == "11bc1b99499061e7b5bbfdab1a6103c62a393c58"
    assert identity["freeze_status"] == "incomplete"
    assert identity["shell"]["sha256"] == sha256_text(shell_template())
    assert "duration" in first["manifest"]["identity_excludes"]
    files = {
        "campaign_manifest.json",
        "attempts.jsonl",
        "document_states.jsonl",
        "predictions.jsonl",
        "evaluation_predictions.jsonl",
        "coverage.json",
        "synthetic_evaluation_summary.json",
    }
    assert files <= {path.name for path in (tmp_path / "first").iterdir()}
    assert json.loads((tmp_path / "first" / "coverage.json").read_text(encoding="utf-8"))["denominator_meaning"] == (
        "cohort_size_not_f1_denominator"
    )


def test_campaign_does_not_call_the_baseline_pipeline() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "re_te_system" / "controlled_experiment" / "campaign.py"
    text = source.read_text(encoding="utf-8")
    assert "run_pipeline(" not in text


def test_contract_abort_withholds_score() -> None:
    from re_te_system.controlled_experiment.reporting import evaluation_summary

    summary = evaluation_summary(
        (_eligible("doc-alpha", "Alpha", "supplies", "Beta", "syn"),),
        [{"document_id": "doc-alpha", "subject": " ", "relation": "supplies", "object": "Beta"}],
        True,
    )
    assert summary["publication"] == "withheld_evaluation_contract_abort"
    assert summary["micro"] is None
    assert summary["experimental_f1"] is False
