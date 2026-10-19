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
from re_te_system.contracts import ExtractionContext
from re_te_system.controlled_experiment.contracts import (
    REAL_C3_CONDITIONS,
    SYNTHETIC_CONTRACT_TEST,
    CampaignError,
    ExternalInfrastructureError,
    PreflightNotAssessed,
    RealC3CampaignRejected,
    SimulatedInfrastructureError,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import (
    SYSTEM_REPO,
    repository_root_from_package,
)
from re_te_system.controlled_experiment.fingerprint import (
    collect_implementation_parts,
    implementation_fingerprint,
)
from re_te_system.controlled_experiment.reporting import contract_violation, evaluation_summary
from re_te_system.runner.run import export_evaluation
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
    assert (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8") == ""


def test_unassessed_preflight_is_not_a_pass(tmp_path: Path) -> None:
    extractor = _extractor({("*", "doc-alpha", 1): _output(TWO_LINES)})
    try:
        run_controlled_campaign(_config(extractor, tokenizer_id=None, output_dir=tmp_path))
    except PreflightNotAssessed:
        pass
    else:
        raise AssertionError("missing tokenizer must not pass preflight")
    assert extractor.calls == []
    states = [
        json.loads(line)
        for line in (tmp_path / "document_states.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(states) == 1
    assert states[0]["harness_incident"] == "not_started_configuration_abort"
    assert "principal_state" not in states[0]
    status = json.loads((tmp_path / "campaign_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "aborted_configuration"
    summary = json.loads((tmp_path / "synthetic_evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["estimands"]["conservative"]["micro"] is None
    assert summary["experimental_f1"] is False


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
        for line in (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
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
    assert coverage["also_observed"] == 3


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
    export_path = tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl"
    rows = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines() if line]
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
    scored = summary["estimands"]["conservative"]["conditions"]["C0"]
    assert scored["scored_row_ids"] == [row["id"] for row in rows]
    assert scored["counts"]["gold_eligible"] == 7
    assert scored["counts"]["gold_eligible_occurrences"] == 8
    assert scored["counts"]["tp"] == 6
    assert scored["counts"]["fp"] == 2
    assert scored["counts"]["fn"] == 2


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
    operational = first["manifest"]["operational"]
    assert "repositories" not in identity
    assert "implementation_fingerprint" not in identity
    assert operational["repositories"]["re_te_system"]["commit_identifies_tree"] is False
    assert operational["repositories"]["re_te_system"]["dirty"] is True
    assert operational["repositories"]["re_te_system"]["head"] == "4996f32d809084a7c6b7e19f5e2649b523b24641"
    assert operational["repositories"]["re_te_benchmark"]["head"] == "005cb78a117216e0f2d4ab221c6e1bc7e8baf16f"
    assert operational["repositories"]["re_te_benchmark"]["commit_identifies_tree"] is True
    assert operational["repositories"]["re_te_evaluation"]["head"] == "11bc1b99499061e7b5bbfdab1a6103c62a393c58"
    assert operational["repositories"]["re_te_evaluation"]["commit_identifies_tree"] is True
    assert operational["freeze_status"] == "incomplete"
    assert operational["created_at"] is None
    assert identity["shell"]["sha256"] == sha256_text(shell_template())
    assert first["implementation_fingerprint"] == second["implementation_fingerprint"]
    assert "duration" in first["manifest"]["identity_excludes"]
    files = {
        "campaign_manifest.json",
        "attempts.jsonl",
        "document_states.jsonl",
        "predictions.jsonl",
        "coverage.json",
        "synthetic_evaluation_summary.json",
        "campaign_status.json",
    }
    assert files <= {path.name for path in (tmp_path / "first").iterdir()}
    assert (tmp_path / "first" / "conditions" / "C0" / "evaluation_predictions.jsonl").is_file()
    assert not (tmp_path / "first" / "evaluation_predictions.jsonl").exists()
    assert json.loads((tmp_path / "first" / "coverage.json").read_text(encoding="utf-8"))["denominator_meaning"] == (
        "cohort_size_not_f1_denominator"
    )


def test_campaign_does_not_call_the_baseline_pipeline() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "re_te_system" / "controlled_experiment" / "campaign.py"
    text = source.read_text(encoding="utf-8")
    assert "run_pipeline(" not in text


def test_contract_abort_withholds_score(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    exported = tmp_path / "evaluation_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "document_id": "doc-alpha",
                "example_id": "doc-alpha",
                "run_id": "synthetic-run",
                "validated": {
                    "triples": [
                        {
                            "subject": " ",
                            "relation": "supplies",
                            "object": "Beta",
                            "segment_id": None,
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    export_evaluation(predictions, exported)
    rows = [json.loads(line) for line in exported.read_text(encoding="utf-8").splitlines() if line]
    assert contract_violation(rows) is not None
    summary = evaluation_summary(
        (_eligible("doc-alpha", "Alpha", "supplies", "Beta", "syn"),),
        rows,
        True,
    )
    assert summary["publication"] == "withheld_evaluation_contract_abort"
    assert summary["micro"] is None
    assert summary["experimental_f1"] is False


class _RaisingExtractor(ScriptedTsvExtractor):
    def __init__(self, error: BaseException) -> None:
        super().__init__({})
        self.error = error

    def extract(self, text: str, context: ExtractionContext):
        self.calls.append((context.condition, context.document_id, int(context.input_id), text))
        raise self.error


def test_second_document_failure_keeps_the_first(tmp_path: Path) -> None:
    def count_tokens(prompt: str) -> int:
        if "STOPHERE" in prompt:
            raise ValueError("counter failed on purpose")
        return len(prompt)

    cohort = (
        _document("doc-kept", "Alpha supplies Beta."),
        _document("doc-stop", "STOPHERE supplies Beta."),
        _document("doc-later", "Gamma supplies Beta."),
    )
    result = run_controlled_campaign(
        _config(
            _extractor(
                {
                    ("*", "doc-kept", 1): _output("Alpha\tsupplies\tBeta\n"),
                    ("*", "doc-stop", 1): _output("SHOULD\tNOT\tRUN\n"),
                    ("*", "doc-later", 1): _output("Gamma\tsupplies\tBeta\n"),
                }
            ),
            cohort=cohort,
            count_tokens=count_tokens,
            synthetic_gold=(
                _eligible("doc-kept", "Alpha", "supplies", "Beta", "syn-kept"),
                _eligible("doc-stop", "STOPHERE", "supplies", "Beta", "syn-stop"),
                _eligible("doc-later", "Gamma", "supplies", "Beta", "syn-later"),
            ),
            output_dir=tmp_path,
        )
    )
    states = _states(result, "C0")
    assert states["doc-kept"]["principal_state"] == "completed"
    assert states["doc-stop"]["harness_incident"] == "token_counter_error"
    assert "principal_state" not in states["doc-stop"]
    assert states["doc-later"]["harness_incident"] == "not_reached_after_interruption"
    assert result["campaign_status"]["status"] == "incomplete_external_or_interrupted"
    assert result["evaluation_summary"]["estimands"]["conservative"]["conditions"]["C0"]["micro"] is None
    assert "doc-stop" not in {row["document_id"] for row in result["prediction_records"]}
    stored = [
        json.loads(line)
        for line in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["document_id"] for row in stored] == [row["document_id"] for row in result["attempts"]]
    assert sum(1 for row in stored if row["document_id"] == "doc-kept") == 1
    assert stored[0]["model_output"] == "Alpha\tsupplies\tBeta\n"
    assert "SHOULD" not in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8")
    exported = (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8")
    assert "doc-kept" in exported
    assert "doc-stop" not in exported
    assert "doc-later" not in exported


def test_missing_script_is_a_harness_incident(tmp_path: Path) -> None:
    extractor = _extractor({("*", "doc-kept", 1): _output("Alpha\tsupplies\tBeta\n")})
    cohort = (
        _document("doc-kept", "Alpha supplies Beta."),
        _document("doc-gap", "Alpha has no script."),
        _document("doc-later", "Gamma supplies Beta."),
    )
    result = run_controlled_campaign(
        _config(
            extractor,
            cohort=cohort,
            synthetic_gold=(
                _eligible("doc-kept", "Alpha", "supplies", "Beta", "syn-kept"),
                _eligible("doc-gap", "Alpha", "supplies", "Beta", "syn-gap"),
            ),
            output_dir=tmp_path,
        )
    )
    states = _states(result, "C0")
    assert states["doc-kept"]["principal_state"] == "completed"
    assert states["doc-gap"]["harness_incident"] == "script_missing"
    assert states["doc-gap"].get("principal_state") != "model_execution_error"
    assert states["doc-later"]["harness_incident"] == "not_reached_after_interruption"
    assert [call[1] for call in extractor.calls] == ["doc-kept", "doc-gap"]
    assert result["campaign_status"]["status"] == "incomplete_external_or_interrupted"
    assert result["evaluation_summary"]["estimands"]["conservative"]["conditions"]["C0"]["micro"] is None
    assert "doc-kept" in (tmp_path / "document_states.jsonl").read_text(encoding="utf-8")


def test_infrastructure_and_timeout_retry_once_but_model_errors_do_not(tmp_path: Path) -> None:
    assert issubclass(SimulatedInfrastructureError, ExternalInfrastructureError)
    assert not issubclass(SimulatedInfrastructureError, OSError)
    cohort = (
        _document("doc-socket", "Alpha opens a socket."),
        _document("doc-os", "Alpha touches a file."),
    )
    socket_extractor = _RaisingExtractor(ConnectionError("connection refused"))
    socket_result = run_controlled_campaign(
        _config(socket_extractor, cohort=(_document("doc-socket", "Alpha opens a socket."),), output_dir=tmp_path / "socket")
    )
    assert _states(socket_result, "C0")["doc-socket"]["principal_state"] == "model_execution_error"
    assert _states(socket_result, "C0")["doc-socket"]["attempt_count"] == 1
    assert socket_extractor.calls and len(socket_extractor.calls) == 1
    os_extractor = _RaisingExtractor(OSError("internal buffer"))
    os_result = run_controlled_campaign(
        _config(os_extractor, cohort=(_document("doc-os", "Alpha touches a file."),), output_dir=tmp_path / "os")
    )
    assert _states(os_result, "C0")["doc-os"]["principal_state"] == "model_execution_error"
    assert _states(os_result, "C0")["doc-os"]["attempt_count"] == 1
    assert "also_observed" not in _states(os_result, "C0")["doc-os"]
    assert len(os_extractor.calls) == 1
    assert cohort[0].document_id == "doc-socket"


def test_truncated_input_is_traced_and_not_scored(tmp_path: Path) -> None:
    extractor = _extractor(
        {
            ("*", "doc-cut", 1): _output("Alpha\tsupplies\tBeta\n", truncated_input=True),
            ("*", "doc-cut", 2): _output("SHOULD\tNOT\tRETRY\n"),
        }
    )
    result = run_controlled_campaign(
        _config(
            extractor,
            cohort=(_document("doc-cut", "Alpha must not lose its prompt."),),
            synthetic_gold=(_eligible("doc-cut", "Alpha", "supplies", "Beta", "syn-cut"),),
            output_dir=tmp_path,
        )
    )
    state = _states(result, "C0")["doc-cut"]
    assert state["principal_state"] == "completed"
    assert state["attempt_count"] == 1
    assert state["diagnostics"]["truncated_input"] is True
    assert len(extractor.calls) == 1
    attempt = result["attempts"][0]
    assert attempt["model_output"] == "Alpha\tsupplies\tBeta\n"
    withheld = [
        json.loads(line)
        for line in (tmp_path / "conditions" / "C0" / "withheld_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert withheld[0]["validated"]["triples"][0]["object"] == "Beta"
    exported = (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8")
    assert exported == ""
    assert result["campaign_status"]["status"] == "finished_publication_blocked"
    assert result["campaign_status"]["publishable_under_protocol"] is False
    for name in ("conservative", "conditional"):
        assert result["evaluation_summary"]["estimands"][name]["conditions"]["C0"]["micro"] is None
    assert result["evaluation_summary"]["estimands"]["paired"]["publication"].startswith("withheld")
    assert result["evaluation_summary"]["estimands"]["conditional"]["conflict_with_protocol"]
    assert result["coverage"]["conditions"]["C0"]["truncated_inputs"] == 1
    assert result["evaluation_summary"]["experimental_f1"] is False
    assert result["manifest"]["operational"]["input_truncation_attestation"] == "not_verified_no_backbone_adapter"


def test_output_truncation_keeps_a_partial_parse(tmp_path: Path) -> None:
    result = run_controlled_campaign(
        _config(
            _extractor(
                {
                    ("*", "doc-partial", 1): _output(
                        "Alpha\tsupplies\tBeta\nnot a triple\n",
                        output_reached_limit=True,
                    )
                }
            ),
            cohort=(_document("doc-partial", "Alpha supplies Beta and then fails."),),
            synthetic_gold=(_eligible("doc-partial", "Alpha", "supplies", "Beta", "syn-partial"),),
            output_dir=tmp_path,
        )
    )
    diagnostics = _states(result, "C0")["doc-partial"]["diagnostics"]
    assert diagnostics["output_truncated"] is True
    assert diagnostics["partially_malformed"] is True
    assert "truncated_input" not in diagnostics
    rows = [
        json.loads(line)
        for line in (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert [(row["subject"], row["relation"], row["object"]) for row in rows] == [("Alpha", "supplies", "Beta")]
    coverage = result["coverage"]["conditions"]["C0"]
    assert coverage["invalid_lines"] == 1
    assert coverage["parsed_triples"] == 1
    assert coverage["triples_exported_to_p0"] == 1
    scored = result["evaluation_summary"]["estimands"]["conservative"]["conditions"]["C0"]
    assert scored["counts"]["tp"] == 1
    assert scored["counts"]["fp"] == 0
    assert scored["counts"]["fn"] == 0
    assert result["campaign_status"]["status"] == "operationally_complete_publishable"


def test_estimands_are_not_interchangeable(tmp_path: Path) -> None:
    cohort = (
        _document("doc-a", "Alpha supplies Beta."),
        _document("doc-b", "Alpha supplies Gamma."),
    )
    result = run_controlled_campaign(
        _config(
            _extractor(
                {
                    ("C0", "doc-a", 1): _output("Alpha\tsupplies\tBeta\n"),
                    ("C0", "doc-b", 1): ScriptedTurn("model_error", message="closed"),
                    ("C1", "doc-a", 1): _output("Alpha\tsupplies\tBeta\n"),
                    ("C1", "doc-b", 1): _output("Alpha\tsupplies\tGamma\n"),
                }
            ),
            cohort=cohort,
            conditions=("C0", "C1"),
            synthetic_gold=(
                _eligible("doc-a", "Alpha", "supplies", "Beta", "syn-a"),
                _eligible("doc-b", "Alpha", "supplies", "Gamma", "syn-b"),
            ),
            output_dir=tmp_path,
        )
    )
    summary = result["evaluation_summary"]
    conservative = summary["estimands"]["conservative"]["conditions"]
    conditional = summary["estimands"]["conditional"]["conditions"]
    paired = summary["estimands"]["paired"]
    assert conservative["C0"]["micro"]["f1"] != conditional["C0"]["micro"]["f1"]
    assert conditional["C0"]["document_ids"] == ["doc-a"]
    assert conditional["C0"]["not_the_cohort"] is True
    assert conditional["C0"]["counts"]["tp"] == 1
    assert conditional["C0"]["counts"]["fn"] == 0
    assert conservative["C0"]["counts"]["fn"] == 1
    assert conservative["C1"]["micro"]["f1"] == 1.0
    assert paired["document_ids"] == ["doc-a"]
    assert paired["exclusions"][0]["document_id"] == "doc-b"
    assert paired["not_the_cohort"] is True
    assert paired["conditions"]["C0"]["counts"]["fn"] == 0
    assert summary["experimental_f1"] is False
    assert result["campaign_status"]["status"] == "operationally_complete_publishable"
    c0_rows = [
        json.loads(line)["id"]
        for line in (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert conservative["C0"]["scored_row_ids"] == c0_rows
    assert all("condition" not in json.loads(line) for line in (tmp_path / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line)


def test_context_block_remains_visible_in_the_paired_exclusion(tmp_path: Path) -> None:
    text = "Alpha supplies Beta."
    reserved = 8
    limit = len(render_condition("C0", text)) + reserved
    assert len(render_condition("C1", text)) + reserved > limit
    result = run_controlled_campaign(
        _config(
            _extractor({("*", "doc-alpha", 1): _output("Alpha\tsupplies\tBeta\n")}),
            cohort=(_document("doc-alpha", text),),
            conditions=("C0", "C1"),
            effective_context_limit=limit,
            reserved_output_tokens=reserved,
            synthetic_gold=(_eligible("doc-alpha", "Alpha", "supplies", "Beta", "syn"),),
            output_dir=tmp_path,
        )
    )
    assert _states(result, "C0")["doc-alpha"]["principal_state"] == "completed"
    assert _states(result, "C1")["doc-alpha"]["principal_state"] == "context_blocked"
    paired = result["evaluation_summary"]["estimands"]["paired"]
    assert paired["document_ids"] == []
    assert paired["publication"] == "withheld_empty_intersection"
    assert "context_blocked" in paired["exclusions"][0]["reasons"][0]
    assert result["evaluation_summary"]["estimands"]["conservative"]["conditions"]["C1"]["micro"]["f1"] == 0.0
    assert result["evaluation_summary"]["estimands"]["conservative"]["conditions"]["C0"]["micro"]["f1"] == 1.0
    assert result["campaign_status"]["status"] == "operationally_complete_publishable"


def test_fingerprint_tracks_parser_bytes_and_ignores_directory(tmp_path: Path) -> None:
    benchmark = repository_root_from_package("re_te_benchmark")
    evaluation = repository_root_from_package("re_te_evaluation")
    assert benchmark is not None and evaluation is not None
    parts = collect_implementation_parts(
        system_root=SYSTEM_REPO,
        benchmark_root=benchmark,
        evaluation_root=evaluation,
        shell_text=shell_template(),
        slots={condition: conditioning_for(condition) for condition in ("C0",)},
        protocol_text=(benchmark / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8"),
    )
    repositories = {"re_te_system": {"head": "abc", "dirty": True, "commit_identifies_tree": False}}
    original = implementation_fingerprint(parts, repositories)
    parser_changed = dict(parts)
    parser_changed["re_te_system/src/re_te_system/parsing/tsv.py"] += "\n# synthetic marker\n"
    exporter_changed = dict(parts)
    exporter_changed["re_te_system/src/re_te_system/runner/run.py"] += "\n# synthetic marker\n"
    assert implementation_fingerprint(parser_changed, repositories) != original
    assert implementation_fingerprint(exporter_changed, repositories) != original
    assert "def parse_tsv" in parts["re_te_system/src/re_te_system/parsing/tsv.py"]
    assert "def export_evaluation" in parts["re_te_system/src/re_te_system/runner/run.py"]
    assert parts["re_te_benchmark/conditioning/shell-v2-template"] == shell_template()
    assert all(not key.startswith("/") for key in parts)
    assert "tests/test_controlled_experiment.py" not in parts
    moved = implementation_fingerprint(parts, repositories)
    assert moved == original
    assert "/Users" not in original
    first = run_controlled_campaign(
        _config(_extractor({("*", "doc-alpha", 1): _output(TWO_LINES)}), output_dir=tmp_path / "one")
    )
    second = run_controlled_campaign(
        _config(_extractor({("*", "doc-alpha", 1): _output(TWO_LINES)}), output_dir=tmp_path / "two")
    )
    assert first["scientific_id"] == second["scientific_id"]
    assert first["implementation_fingerprint"] == second["implementation_fingerprint"]
    assert first["manifest"]["operational"]["output_dir"] != second["manifest"]["operational"]["output_dir"]


def test_disk_error_propagates(tmp_path: Path, monkeypatch) -> None:
    def fail_write(path: Path, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "re_te_system.controlled_experiment.persistence.atomic_write_text",
        fail_write,
    )
    try:
        run_controlled_campaign(
            _config(_extractor({("*", "doc-alpha", 1): _output(TWO_LINES)}), output_dir=tmp_path)
        )
    except OSError as exc:
        assert "disk full" in str(exc)
    else:
        raise AssertionError("a failed checkpoint must not look like a saved campaign")
    assert not (tmp_path / "synthetic_evaluation_summary.json").exists()
