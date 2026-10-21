"""Offline evaluation bridge tests.

Fixtures are invented. These tests do not read Hohfeld Gold, do not load a
model, and do not treat a scripted TSV string as a scientific prediction.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from re_te_system.controlled_experiment.dependencies import ensure_package

ensure_package("re_te_benchmark", "re_te_benchmark")
ensure_package("re_te_evaluation", "re_te_evaluation")

from re_te_benchmark.conditioning.shell import render_condition  # noqa: E402
from re_te_evaluation import (  # noqa: E402
    ContextMode,
    EligibilityPolicy,
    EvaluationConfig,
    MetricFamily,
    evaluate_triplets,
)
from re_te_evaluation.cli import main as evaluation_cli  # noqa: E402
from re_te_evaluation.contracts import NormalizationPolicy  # noqa: E402
from re_te_evaluation.eligibility import read_records  # noqa: E402
from re_te_evaluation.reporting.writers import REPORT_FILENAMES  # noqa: E402
from re_te_system.controlled_experiment.campaign import (  # noqa: E402
    CampaignConfig,
    SyntheticDocument,
    normalize_controlled_triples,
    run_controlled_campaign,
)
from re_te_system.controlled_experiment.contracts import sha256_text  # noqa: E402
from re_te_system.controlled_experiment.dependencies import (  # noqa: E402
    repository_root_from_package,
)
from re_te_system.controlled_experiment.synthetic import (  # noqa: E402
    ScriptedTsvExtractor,
    ScriptedTurn,
    SyntheticTokenCounter,
)
from re_te_system.evaluation_bridge import BridgeRequest, evaluate_campaign  # noqa: E402
from re_te_system.evaluation_bridge.__main__ import main as bridge_main  # noqa: E402
from re_te_system.normalization.basic import normalize_text  # noqa: E402
from re_te_system.parsing.tsv import parse_tsv  # noqa: E402


DOC_A = "Alpha supplies Beta."
DOC_B = "Gamma supplies Delta."
LINE_A = "Alpha\tsupplies\tBeta\n"
LINE_B = "Gamma\tsupplies\tDelta\n"


def _output(text: str, **flags: bool) -> ScriptedTurn:
    return ScriptedTurn("output", output=text, **flags)


def _eligible(document_id: str, subject: str, relation: str, object_: str, identifier: str) -> dict:
    return {
        "document_id": document_id,
        "eligibility": {
            "complete_triplet_strict_inline": {"eligible": True, "reason": "synthetic_fixture"}
        },
        "id": identifier,
        "object": object_,
        "relation": relation,
        "subject": subject,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _campaign(
    tmp_path: Path,
    *,
    documents: list[tuple[str, str]],
    scripts: dict,
    gold: list[dict],
    conditions: tuple[str, ...] = ("C0",),
    submitted_ids: frozenset[str] | None = None,
):
    cohort = tuple(SyntheticDocument(document_id, text) for document_id, text in documents)
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompts = [render_condition(condition, document.text) for condition in conditions for document in cohort]
    counter = SyntheticTokenCounter()
    campaign = tmp_path / "campaign"
    gold_path = tmp_path / "gold.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(
        documents_path,
        [{"document_id": document_id, "text": text} for document_id, text in documents],
    )
    result = run_controlled_campaign(
        CampaignConfig(
            conditions=conditions,
            cohort=cohort,
            count_tokens=counter.count_tokens,
            counter_id=counter.counter_id,
            effective_context_limit=max(len(prompt) for prompt in prompts) + 8,
            extractor=ScriptedTsvExtractor(scripts),
            output_dir=campaign,
            reserved_output_tokens=8,
            submitted_ids=submitted_ids,
            synthetic_gold=tuple(gold),
            tokenizer_id=counter.tokenizer_id,
            tokenizer_revision=counter.tokenizer_revision,
        )
    )
    protocol = repository_root_from_package("re_te_benchmark")
    assert protocol is not None
    return {
        "campaign": campaign,
        "documents": documents_path,
        "gold": gold_path,
        "gold_rows": gold,
        "protocol": protocol / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md",
        "result": result,
    }


def _evaluate(paths: dict, output: Path) -> dict:
    return evaluate_campaign(
        BridgeRequest(
            campaign_dir=paths["campaign"],
            documents_path=paths["documents"],
            gold_path=paths["gold"],
            output_dir=output,
            protocol_path=paths["protocol"],
        )
    )


def _p0(gold: list[dict], predictions: list[dict]):
    return evaluate_triplets(
        gold,
        predictions,
        EvaluationConfig(MetricFamily.TRIPLET, ContextMode.DOCUMENT, EligibilityPolicy.STRICT_INLINE),
    )


def _condition(report: dict, estimand: str, condition: str = "C0") -> dict:
    return report["estimands"][estimand]["conditions"][condition]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_export_hashes(campaign: Path, digest: str) -> None:
    path = campaign / "synthetic_evaluation_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "export_sha256" in node:
                node["export_sha256"] = digest
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(summary)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path = campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claims = manifest.setdefault("operational", {}).get("export_sha256")
    if isinstance(claims, dict):
        for condition in claims:
            claims[condition] = digest
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def _replace_export_subject(campaign: Path, subject: str) -> None:
    path = campaign / "conditions" / "C0" / "evaluation_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["subject"] = subject
    _write_jsonl(path, rows)
    _rewrite_export_hashes(campaign, sha256_text(path.read_text(encoding="utf-8")))


def test_complete_campaign_matches_official_p0_and_is_not_scientific(tmp_path: Path) -> None:
    documents = [("doc-a", DOC_A), ("doc-b", DOC_B)]
    gold = [
        _eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a"),
        _eligible("doc-b", "Gamma", "supplies", "Delta", "gold-b"),
    ]
    paths = _campaign(
        tmp_path,
        documents=documents,
        conditions=("C0", "C1"),
        gold=gold,
        scripts={
            ("*", "doc-a", 1): _output(LINE_A),
            ("*", "doc-b", 1): _output(LINE_B),
        },
    )
    before = _snapshot(paths["campaign"])
    report = _evaluate(paths, tmp_path / "bridge")
    assert _snapshot(paths["campaign"]) == before
    assert report["publication_eligibility"]["scientific_result"] is False
    assert report["publication_eligibility"]["experimental_f1"] is False
    assert report["publication_eligibility"]["comparability"] == "not_comparable"
    assert report["publication_eligibility"]["registry_protocol_id"] == "SYNTHETIC_CONTRACT"
    assert report["publication_eligibility"]["technically_scorable"] is False
    assert report["estimand_selection"]["selected_after_observing_scores"] is False
    assert report["estimand_selection"]["selected_estimand"] is None
    text = (tmp_path / "bridge" / "campaign_evaluation_report.json").read_text(encoding="utf-8")
    assert '"experimental_f1":true' not in text
    for condition in ("C0", "C1"):
        scored = _condition(report, "conservative", condition)
        export = paths["campaign"] / "conditions" / condition / "evaluation_predictions.jsonl"
        direct = _p0(gold, read_records(export))
        assert scored["counts"] == direct.summary["counts"]
        assert scored["micro"] == direct.summary["micro"]
        assert scored["gold_support"] == 2
        assert scored["prediction_support"] == 2
        assert scored["experimental_f1"] is False
        assert scored["publication"] == "synthetic_contract_only"
        assert scored["micro"]["f1"] == 1.0
        cli_dir = tmp_path / f"cli-{condition}"
        assert evaluation_cli(
            [
                "--gold",
                str(paths["gold"]),
                "--predictions",
                str(export),
                "--metric",
                "triplet",
                "--context-mode",
                "document",
                "--eligibility-policy",
                "strict_inline",
                "--output-dir",
                str(cli_dir),
            ]
        ) == 0
        cli_summary = json.loads((cli_dir / "summary.json").read_text(encoding="utf-8"))
        assert scored["counts"] == cli_summary["counts"]
        assert scored["micro"] == cli_summary["micro"]
        assert scored["config_id"] == cli_summary["config_id"]
        bridge_summary = tmp_path / "bridge" / scored["p0_output"] / "summary.json"
        for name in (*REPORT_FILENAMES, "publication.json"):
            assert (bridge_summary.parent / name).is_file()
        subset_dir = tmp_path / f"cli-subset-{condition}"
        assert evaluation_cli(
            [
                "--gold",
                str(bridge_summary.parent / "gold_subset.jsonl"),
                "--predictions",
                str(bridge_summary.parent / "predictions_subset.jsonl"),
                "--metric",
                "triplet",
                "--context-mode",
                "document",
                "--eligibility-policy",
                "strict_inline",
                "--output-dir",
                str(subset_dir),
            ]
        ) == 0
        for name in REPORT_FILENAMES:
            if name == "summary.json":
                continue
            assert (bridge_summary.parent / name).read_bytes() == (subset_dir / name).read_bytes()
        bridge_disk = json.loads(bridge_summary.read_text(encoding="utf-8"))
        subset_summary = json.loads((subset_dir / "summary.json").read_text(encoding="utf-8"))
        for key, value in subset_summary.items():
            if key == "provenance":
                continue
            assert bridge_disk[key] == value
        assert bridge_disk["provenance"]["code_commit"] == subset_summary["provenance"]["code_commit"]
        assert bridge_disk["provenance"]["evaluation_package"] == "re_te_evaluation"
        assert bridge_disk["provenance"]["scoring_interface"] == "read_records+evaluate_triplets+write_report"
        assert bridge_disk["config"] == subset_summary["config"]
        publication = json.loads((bridge_summary.parent / "publication.json").read_text(encoding="utf-8"))
        assert publication["experimental_f1"] is False
        assert publication["scientific_result"] is False
    assert report["estimands"]["paired"]["document_count"] == 2
    assert report["coverage"]["conditions"]["C0"]["denominator"] == 2
    harness = paths["result"]["evaluation_summary"]["estimands"]["conservative"]["conditions"]["C0"]
    assert _condition(report, "conservative")["micro"] == harness["micro"]


def test_incomplete_cohort_withholds_conservative_only(tmp_path: Path) -> None:
    gold = [
        _eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a"),
        _eligible("doc-b", "Gamma", "supplies", "Delta", "gold-b"),
    ]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A), ("doc-b", DOC_B)],
        gold=gold,
        scripts={("*", "doc-a", 1): _output(LINE_A), ("*", "doc-b", 1): _output(LINE_B)},
        submitted_ids=frozenset({"doc-a"}),
    )
    report = _evaluate(paths, tmp_path / "bridge")
    assert _condition(report, "conservative")["micro"] is None
    assert _condition(report, "conservative")["publication"] == "withheld_campaign_not_publishable"
    conditional = _condition(report, "conditional")
    assert conditional["publication"] == "synthetic_contract_only"
    assert conditional["gold_support"] == 1
    assert conditional["micro"]["f1"] == 1.0
    assert conditional["counts"] == _p0([gold[0]], read_records(
        tmp_path / "bridge" / conditional["p0_output"] / "predictions_subset.jsonl"
    )).summary["counts"]
    assert report["coverage"]["conditions"]["C0"]["not_attempted"] == 1
    assert report["publication_eligibility"]["scientific_result"] is False


def test_duplicate_document_state_is_blocked(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    states = paths["campaign"] / "document_states.jsonl"
    states.write_text(states.read_text(encoding="utf-8") + states.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    report = _evaluate(paths, tmp_path / "bridge")
    assert "duplicate_document_state" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None
    assert report["estimands"]["conservative"]["publication"] == "withheld_integrity"


def test_unexpected_document_id_is_blocked(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    states = paths["campaign"] / "document_states.jsonl"
    row = json.loads(states.read_text(encoding="utf-8").splitlines()[0])
    row["document_id"] = "intruder"
    with states.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    report = _evaluate(paths, tmp_path / "bridge")
    assert "unexpected_document_id" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None


def test_incorrect_input_hash_is_blocked(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    _write_jsonl(paths["documents"], [{"document_id": "doc-a", "text": "different text"}])
    report = _evaluate(paths, tmp_path / "bridge")
    assert "incorrect_input_hash" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None
    assert report["publication_eligibility"]["scientific_result"] is False


def test_incorrect_export_hash_is_blocked(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    _rewrite_export_hashes(paths["campaign"], "0" * 64)
    report = _evaluate(paths, tmp_path / "bridge")
    assert "incorrect_export_hash" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None


def test_missing_gold_is_blocked(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    paths["gold"] = tmp_path / "absent-gold.jsonl"
    report = _evaluate(paths, tmp_path / "bridge")
    assert "missing_gold" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None


def test_invalid_principal_state_is_blocked(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    path = paths["campaign"] / "document_states.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["principal_state"] = "not_a_state"
    _write_jsonl(path, rows)
    report = _evaluate(paths, tmp_path / "bridge")
    assert "invalid_principal_state" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None


def test_protocol_and_condition_mismatches_are_blocked(tmp_path: Path) -> None:
    policy = _ready(tmp_path / "policy")
    manifest_path = policy["campaign"] / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["evaluation_policy"]["normalization_policy"] = "unicode_nfc+outer_strip+lowercase_true"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    policy_report = _evaluate(policy, tmp_path / "policy-out")
    assert "protocol_mismatch" in policy_report["blockers"]
    assert _condition(policy_report, "conservative")["micro"] is None

    wrong_protocol = tmp_path / "protocol.md"
    wrong_protocol.write_text("not the frozen protocol\n", encoding="utf-8")
    hashed = _ready(tmp_path / "hash")
    hashed["protocol"] = wrong_protocol
    hash_report = _evaluate(hashed, tmp_path / "hash-out")
    assert "incorrect_protocol_hash" in hash_report["blockers"]
    assert _condition(hash_report, "conservative")["micro"] is None

    condition = _ready(tmp_path / "condition")
    condition_manifest_path = condition["campaign"] / "campaign_manifest.json"
    condition_manifest = json.loads(condition_manifest_path.read_text(encoding="utf-8"))
    condition_manifest["identity"]["conditions"] = ["C1"]
    condition_manifest_path.write_text(
        json.dumps(condition_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    condition_report = _evaluate(condition, tmp_path / "condition-out")
    assert "condition_mismatch" in condition_report["blockers"]
    assert condition_report["estimands"]["conservative"]["publication"] == "withheld_integrity"


def test_controlled_export_follows_frozen_p0_normalization(tmp_path: Path) -> None:
    assert normalize_text("Alpha  Beta") == "Alpha Beta"
    assert NormalizationPolicy().normalize("Alpha  Beta") == "Alpha  Beta"
    parsed = parse_tsv("Alpha  Beta\tlinks\tGamma\n", None)
    controlled = normalize_controlled_triples(parsed.triples)
    assert controlled[0].subject == "Alpha  Beta"
    assert len(controlled) == 1
    outer = normalize_controlled_triples(parse_tsv("  Alpha  \tsupplies\tBeta\n", None).triples)
    assert outer[0].subject == "Alpha"
    assert outer[0].relation == "supplies"
    gold = [_eligible("doc-a", "Alpha  Beta", "links", "Gamma", "gold-space")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", "A synthetic sentence.")],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output("Alpha  Beta\tlinks\tGamma\n")},
    )
    export_rows = read_records(paths["campaign"] / "conditions" / "C0" / "evaluation_predictions.jsonl")
    assert export_rows[0]["subject"] == "Alpha  Beta"
    report = _evaluate(paths, tmp_path / "bridge")
    assert "normalization_asymmetry" not in report["blockers"]
    assert "exporter_contract_violation" not in report["blockers"]
    assert report["normalization"]["asymmetry_observed"] is False
    assert report["normalization"]["controlled_export_policy"] == report["normalization"]["p0_policy"]
    assert report["normalization"]["policies_changed"] is False
    scored = _condition(report, "conservative")
    assert scored["counts"]["tp"] == 1
    assert scored["micro"] == _p0(gold, export_rows).summary["micro"]
    assert scored["publication"] == "synthetic_contract_only"
    assert report["publication_eligibility"]["scientific_result"] is False
    folded = _campaign(
        tmp_path / "case",
        documents=[("doc-a", "A synthetic sentence.")],
        gold=[_eligible("doc-a", "Alpha", "links", "Gamma", "gold-case")],
        scripts={("C0", "doc-a", 1): _output("ALPHA\tlinks\tGamma\n")},
    )
    folded_report = _evaluate(folded, tmp_path / "case-out")
    assert _condition(folded_report, "conservative")["counts"]["tp"] == 0


def test_collapsed_export_cannot_replace_the_frozen_policy(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha  Beta", "links", "Gamma", "gold-space")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", "A synthetic sentence.")],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output("Alpha  Beta\tlinks\tGamma\n")},
    )
    _replace_export_subject(paths["campaign"], "Alpha Beta")
    report = _evaluate(paths, tmp_path / "bridge")
    assert "exporter_contract_violation" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None
    assert report["publication_eligibility"]["scientific_result"] is False
    assert report["estimand_selection"]["selected_estimand"] is None


def test_normalization_divergence_still_withholds_both_candidate_scores(tmp_path: Path) -> None:
    from re_te_system.evaluation_bridge.integrity import Inspection, _verify_normalization
    from re_te_system.evaluation_bridge.scoring import score_inspection

    gold = [_eligible("doc-a", "Alpha  Beta", "links", "Gamma", "gold-space")]
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, gold)
    found = Inspection(
        conditions=["C0"],
        document_ids=["doc-a"],
        exports={
            "C0": [
                {
                    "document_id": "doc-a",
                    "object": "Gamma",
                    "relation": "links",
                    "subject": "Alpha Beta",
                }
            ]
        },
        final_attempts={("C0", "doc-a"): {"model_output": "Alpha  Beta\tlinks\tGamma\n"}},
        gold=gold,
        model_identity={"name": "synthetic-tsv-script", "revision": "exp3a-script-v1"},
        recomputed_status={"status": "operationally_complete_publishable"},
        states=[
            {
                "condition": "C0",
                "diagnostics": {},
                "document_id": "doc-a",
                "principal_state": "completed",
            }
        ],
    )
    _verify_normalization(found)
    assert found.normalization["asymmetry_observed"] is True
    assert "normalization_asymmetry" in found.blockers
    request = BridgeRequest(
        campaign_dir=tmp_path / "campaign",
        gold_path=gold_path,
        output_dir=tmp_path / "bridge",
    )
    estimands = score_inspection(request, found)
    assert estimands["conservative"]["publication"] == "withheld_normalization_asymmetry"
    assert estimands["conservative"]["conditions"]["C0"]["micro"] is None
    diagnostic = found.normalization["conditions"]["C0"]
    assert diagnostic["export_p0"]["counts"]["tp"] == 0
    assert diagnostic["protocol_p0"]["counts"]["tp"] == 1
    assert diagnostic["export_p0"]["publication"] == "non_publishable_normalization_diagnostic"


def test_empty_completed_generation_is_scored_and_distinguished(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output("")},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    scored = _condition(report, "conservative")
    assert report["coverage"]["conditions"]["C0"]["empty_generations"] == 1
    assert report["coverage"]["conditions"]["C0"]["fully_malformed"] == 0
    assert report["coverage"]["conditions"]["C0"]["principal_states"]["completed"] == 1
    assert scored["prediction_support"] == 0
    assert scored["counts"]["fn"] == 1
    assert scored["micro"]["f1"] == 0.0
    assert scored["publication"] == "synthetic_contract_only"
    assert report["campaign_status"]["recomputed"]["status"] == "operationally_complete_publishable"


def test_fully_malformed_generation_is_scored_and_distinguished(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output("this is not tsv\n")},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    coverage = report["coverage"]["conditions"]["C0"]
    assert coverage["fully_malformed"] == 1
    assert coverage["empty_generations"] == 0
    assert _condition(report, "conservative")["prediction_support"] == 0
    assert _condition(report, "conservative")["counts"]["fn"] == 1
    assert "UNPARSEABLE_CHUNK" in report["preservation"]["parse_issue_counts"]


def test_partially_malformed_generation_keeps_valid_lines(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output("Alpha\tsupplies\tBeta\nstill not tsv\n")},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    scored = _condition(report, "conservative")
    assert report["coverage"]["conditions"]["C0"]["partially_malformed"] == 1
    assert scored["counts"]["tp"] == 1
    assert scored["prediction_support"] == 1
    assert scored["micro"] == _p0(gold, read_records(
        paths["campaign"] / "conditions" / "C0" / "evaluation_predictions.jsonl"
    )).summary["micro"]
    assert report["preservation"]["parse_issue_counts"]["UNPARSEABLE_CHUNK"] == 1


def test_truncated_input_withholds_scores(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output(LINE_A, truncated_input=True)},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    assert _condition(report, "conservative")["micro"] is None
    assert _condition(report, "conservative")["publication"] == "withheld_truncated_input"
    assert _condition(report, "conditional")["micro"] is None
    withheld = paths["campaign"] / "conditions" / "C0" / "withheld_predictions.jsonl"
    assert "Alpha" in withheld.read_text(encoding="utf-8")
    assert report["preservation"]["raw_output_count"] == 1


def test_truncated_output_scores_valid_lines(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output(LINE_A, output_reached_limit=True)},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    scored = _condition(report, "conservative")
    assert report["coverage"]["conditions"]["C0"]["truncated_outputs"] == 1
    assert scored["counts"]["tp"] == 1
    assert scored["publication"] == "synthetic_contract_only"
    assert scored["experimental_f1"] is False


def test_unresolved_external_failure_is_incomplete(tmp_path: Path) -> None:
    gold = [_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={
            ("C0", "doc-a", 1): ScriptedTurn("timeout"),
            ("C0", "doc-a", 2): ScriptedTurn("timeout"),
        },
    )
    report = _evaluate(paths, tmp_path / "bridge")
    assert report["campaign_status"]["recomputed"]["status"] == "incomplete_external_or_interrupted"
    assert report["coverage"]["conditions"]["C0"]["principal_states"]["timeout"] == 1
    assert _condition(report, "conservative")["micro"] is None
    assert _condition(report, "conservative")["publication"] == "withheld_campaign_incomplete"
    assert report["publication_eligibility"]["scientific_result"] is False


def test_model_execution_error_stays_in_the_conservative_estimand(tmp_path: Path) -> None:
    gold = [
        _eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a"),
        _eligible("doc-b", "Gamma", "supplies", "Delta", "gold-b"),
    ]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A), ("doc-b", DOC_B)],
        gold=gold,
        scripts={
            ("C0", "doc-a", 1): ScriptedTurn("model_error"),
            ("C0", "doc-b", 1): _output(LINE_B),
        },
    )
    report = _evaluate(paths, tmp_path / "bridge")
    coverage = report["coverage"]["conditions"]["C0"]
    assert coverage["principal_states"]["model_execution_error"] == 1
    assert coverage["empty_generations"] == 0
    scored = _condition(report, "conservative")
    assert scored["counts"]["tp"] == 1
    assert scored["counts"]["fn"] == 1
    assert scored["prediction_support"] == 1
    assert scored["publication"] == "synthetic_contract_only"
    assert scored["micro"] == _p0(
        gold,
        read_records(paths["campaign"] / "conditions" / "C0" / "evaluation_predictions.jsonl"),
    ).summary["micro"]
    assert report["publication_eligibility"]["experimental_f1"] is False


def test_incomplete_paired_cohort_reports_the_intersection(tmp_path: Path) -> None:
    gold = [
        _eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a"),
        _eligible("doc-b", "Gamma", "supplies", "Delta", "gold-b"),
    ]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A), ("doc-b", DOC_B)],
        conditions=("C0", "C1"),
        gold=gold,
        scripts={
            ("C0", "doc-a", 1): _output(LINE_A),
            ("C0", "doc-b", 1): _output(LINE_B),
            ("C1", "doc-a", 1): _output(LINE_A),
            ("C1", "doc-b", 1): ScriptedTurn("model_error"),
        },
    )
    report = _evaluate(paths, tmp_path / "bridge")
    paired = report["estimands"]["paired"]
    assert paired["document_count"] == 1
    assert paired["document_ids"] == ["doc-a"]
    assert paired["exclusions"] == [{"document_id": "doc-b", "reasons": ["C1:model_execution_error"]}]
    assert paired["not_the_cohort"] is True
    for condition in ("C0", "C1"):
        scored = _condition(report, "paired", condition)
        assert scored["gold_support"] == 1
        assert scored["counts"]["tp"] == 1
        assert scored["counts"]["fn"] == 0
        assert scored["experimental_f1"] is False
    conservative_c1 = _condition(report, "conservative", "C1")
    assert conservative_c1["counts"]["fn"] == 1
    assert report["estimand_selection"]["selected_estimand"] is None


def test_duplicate_triplets_keep_multiplicity(tmp_path: Path) -> None:
    gold = [
        _eligible("doc-a", "Alpha", "supplies", "Beta", "gold-1"),
        _eligible("doc-a", "Alpha", "supplies", "Beta", "gold-2"),
    ]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output(LINE_A + LINE_A)},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    scored = _condition(report, "conservative")
    direct = _p0(gold, read_records(paths["campaign"] / "conditions" / "C0" / "evaluation_predictions.jsonl"))
    assert scored["counts"] == direct.summary["counts"]
    assert scored["counts"]["tp"] == 2
    assert scored["counts"]["fp"] == 0
    assert scored["counts"]["fn"] == 0
    assert scored["prediction_support"] == 2
    assert scored["gold_support"] == 2


def test_bridge_report_is_byte_stable(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    first = _evaluate(paths, tmp_path / "one")
    second = _evaluate(paths, tmp_path / "two")
    assert first == second
    assert (tmp_path / "one" / "campaign_evaluation_report.json").read_bytes() == (
        tmp_path / "two" / "campaign_evaluation_report.json"
    ).read_bytes()
    relative = _condition(first, "conservative")["p0_output"]
    assert (tmp_path / "one" / relative / "summary.json").read_bytes() == (
        tmp_path / "two" / relative / "summary.json"
    ).read_bytes()
    assert (tmp_path / "one" / relative / "matches.jsonl").read_bytes() == (
        tmp_path / "two" / relative / "matches.jsonl"
    ).read_bytes()
    assert str(tmp_path) not in (tmp_path / "one" / "campaign_evaluation_report.json").read_text(encoding="utf-8")


def test_parser_or_exporter_contract_violation_blocks_publication(tmp_path: Path) -> None:
    empty = _ready(tmp_path / "empty")
    _replace_export_subject(empty["campaign"], "")
    empty_report = _evaluate(empty, tmp_path / "empty-out")
    assert "evaluation_contract_abort" in empty_report["blockers"]
    assert _condition(empty_report, "conservative")["micro"] is None
    assert evaluation_cli(
        [
            "--gold",
            str(empty["gold"]),
            "--predictions",
            str(empty["campaign"] / "conditions" / "C0" / "evaluation_predictions.jsonl"),
            "--metric",
            "triplet",
            "--context-mode",
            "document",
            "--eligibility-policy",
            "strict_inline",
            "--output-dir",
            str(tmp_path / "cli-empty"),
        ]
    ) == 2

    tampered = _ready(tmp_path / "tampered")
    _replace_export_subject(tampered["campaign"], "Tampered")
    tampered_report = _evaluate(tampered, tmp_path / "tampered-out")
    assert "exporter_contract_violation" in tampered_report["blockers"]
    assert _condition(tampered_report, "conservative")["micro"] is None
    assert tampered_report["publication_eligibility"]["scientific_result"] is False


def test_gold_without_document_id_uses_annotation_id(tmp_path: Path) -> None:
    gold = [
        {
            "annotation_id": "doc-a#1",
            "eligibility": {
                "complete_triplet_strict_inline": {"eligible": True, "reason": "synthetic_fixture"}
            },
            "object": "Beta",
            "relation": "supplies",
            "subject": "Alpha",
        }
    ]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output(LINE_A)},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    direct = _p0(gold, read_records(paths["campaign"] / "conditions" / "C0" / "evaluation_predictions.jsonl"))
    for estimand in ("conservative", "conditional"):
        scored = _condition(report, estimand)
        assert scored["counts"] == direct.summary["counts"]
        assert scored["gold_support"] == 1
        assert scored["counts"]["tp"] == 1
        assert scored["experimental_f1"] is False
    harness = paths["result"]["evaluation_summary"]["estimands"]["conditional"]["conditions"]["C0"]
    assert harness["gold_support"] == 1
    assert harness["counts"]["tp"] == 1
    paired = paths["result"]["evaluation_summary"]["estimands"]["paired"]["conditions"]["C0"]
    assert paired["gold_support"] == 1


def test_explicit_document_id_wins_over_annotation_prefix(tmp_path: Path) -> None:
    gold = [
        {
            "annotation_id": "other-doc#9",
            "document_id": "doc-a",
            "eligibility": {
                "complete_triplet_strict_inline": {"eligible": True, "reason": "synthetic_fixture"}
            },
            "object": "Beta",
            "relation": "supplies",
            "subject": "Alpha",
        }
    ]
    paths = _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=gold,
        scripts={("C0", "doc-a", 1): _output(LINE_A)},
    )
    report = _evaluate(paths, tmp_path / "bridge")
    assert "gold_document_id_unresolved" not in report["blockers"]
    assert _condition(report, "conditional")["counts"]["tp"] == 1
    harness = paths["result"]["evaluation_summary"]["estimands"]["conditional"]["conditions"]["C0"]
    assert harness["gold_support"] == 1


def test_missing_export_hash_claim_blocks_the_campaign(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    (paths["campaign"] / "synthetic_evaluation_summary.json").unlink()
    report = _evaluate(paths, tmp_path / "bridge")
    assert "missing_evaluation_summary" in report["blockers"]
    assert _condition(report, "conservative")["micro"] is None
    assert report["publication_eligibility"]["scientific_result"] is False


def test_benchmark_manifest_provenance_reaches_the_p0_summary(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    documents_hash = hashlib.sha256(paths["documents"].read_bytes()).hexdigest()
    gold_hash = hashlib.sha256(paths["gold"].read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": paths["documents"].name,
                        "role": "canonical_documents",
                        "sha256": documents_hash,
                    },
                    {
                        "relative_path": paths["gold"].name,
                        "role": "canonical_annotations",
                        "sha256": gold_hash,
                    },
                ],
                "dataset_hash": "synthetic-dataset-hash",
                "dataset_id": "synthetic-fixture",
                "dataset_version": "0",
                "schema": {"id": "synthetic-schema", "sha256": "synthetic-schema-hash", "version": "0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = evaluate_campaign(
        BridgeRequest(
            benchmark_manifest_path=manifest_path,
            campaign_dir=paths["campaign"],
            documents_path=paths["documents"],
            gold_path=paths["gold"],
            output_dir=tmp_path / "bridge",
            protocol_path=paths["protocol"],
        )
    )
    assert "incorrect_gold_hash" not in report["blockers"]
    assert "incorrect_documents_hash" not in report["blockers"]
    assert "benchmark_manifest_unverified" not in report["scientific_reasons"]
    provenance = json.loads(
        (tmp_path / "bridge" / _condition(report, "conservative")["p0_output"] / "summary.json").read_text(
            encoding="utf-8"
        )
    )["provenance"]
    assert provenance["dataset_id"] == "synthetic-fixture"
    assert provenance["dataset_version"] == "0"
    assert provenance["dataset_hash"] == "synthetic-dataset-hash"
    assert provenance["schema_id"] == "synthetic-schema"
    assert provenance["schema_version"] == "0"
    assert provenance["schema_hash"] == "synthetic-schema-hash"
    assert provenance["gold_artifact_hash"] == gold_hash
    assert provenance["gold_artifact_role"] == "canonical_annotations"
    assert provenance["code_commit"]
    assert report["publication_eligibility"]["scientific_result"] is False


def test_module_entry_point_reads_a_finished_campaign(tmp_path: Path) -> None:
    paths = _ready(tmp_path)
    output = tmp_path / "module-out"
    arguments = [
        "--campaign-dir",
        str(paths["campaign"]),
        "--gold",
        str(paths["gold"]),
        "--documents",
        str(paths["documents"]),
        "--protocol",
        str(paths["protocol"]),
        "--output-dir",
        str(output),
    ]
    assert bridge_main(arguments) == 0
    report = json.loads((output / "campaign_evaluation_report.json").read_text(encoding="utf-8"))
    assert _condition(report, "conservative")["counts"]["tp"] == 1
    assert report["publication_eligibility"]["scientific_result"] is False
    assert bridge_main(
        [
            "--campaign-dir",
            str(paths["campaign"]),
            "--gold",
            str(paths["gold"]),
            "--output-dir",
            str(paths["campaign"] / "inside"),
        ]
    ) == 2
    root = repository_root_from_package("re_te_system")
    assert root is not None
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "re_te_system.evaluation_bridge", *arguments[:8], "--output-dir", str(tmp_path / "process")],
        check=False,
        capture_output=True,
        cwd=root,
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    process_report = json.loads(
        (tmp_path / "process" / "campaign_evaluation_report.json").read_text(encoding="utf-8")
    )
    assert _condition(process_report, "conservative")["counts"]["tp"] == 1


def _ready(tmp_path: Path) -> dict:
    return _campaign(
        tmp_path,
        documents=[("doc-a", DOC_A)],
        gold=[_eligible("doc-a", "Alpha", "supplies", "Beta", "gold-a")],
        scripts={("C0", "doc-a", 1): _output(LINE_A)},
    )
