"""Local checks for the declarative campaign registry."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

from re_te_system.campaign_registry.contracts import REGISTRY_CONTRACT_VERSION
from re_te_system.campaign_registry.report import render_report
from re_te_system.campaign_registry.validator import (
    comparison_key,
    entries_rankable,
    validate_catalog,
    validate_documents,
)
from re_te_system.controlled_experiment.contracts import CAMPAIGN_CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "campaigns" / "exp3c"
SUBMITTER = ROOT / "scripts" / "cesvima" / "submit_train_first.sh"


def _profile(profile_id: str = "cpu", gpu_count: int = 0, vram: object = None) -> dict:
    return {
        "id": profile_id,
        "gpu_count": gpu_count,
        "vram": vram,
        "enabled": True,
    }


def _task(task_id: str = "T", **overrides: object) -> dict:
    task = {
        "id": task_id,
        "program": "SOTA",
        "system_id": None,
        "dataset_id": None,
        "split": None,
        "condition": None,
        "protocol_id": None,
        "summary": "synthetic",
        "inputs": [],
        "outputs": [],
        "dependencies": [],
        "resource_profile_id": "cpu",
        "resources": {"process_ram": "1G", "vram": None},
        "executor": {"kind": "none", "path": None, "invoked_by": None},
        "gates": ["synthetic gate"],
        "acceptance_criteria": ["synthetic acceptance"],
        "preparation_status": "AWAITING_AUTHORIZATION",
        "authorization": "not_authorized",
        "technical_block": False,
        "block_reasons": [],
        "training": False,
        "included_in_submit_all": False,
        "evaluation": None,
        "retry_policy": "none",
        "execution_permitted": False,
        "leaderboard_class": None,
        "requires_checkpoint_identity": False,
        "requires_tokenizer": False,
        "requires_real_preflight": False,
        "real_preflight_done": False,
        "requires_no_truncation": False,
        "requires_frozen_output_budget": False,
        "output_budget_frozen": False,
        "requires_channel_separator": False,
    }
    task.update(overrides)
    return task


def _documents(**overrides: object) -> dict:
    documents = {
        "systems": [],
        "datasets": [],
        "protocols": [],
        "profiles": [_profile()],
        "tasks": [_task()],
        "condition_policy": {
            "C0": {"scientifically_executable": False},
            "C1": {"scientifically_executable": False},
            "C2": {"scientifically_executable": False},
            "C3": {"scientifically_executable": False},
            "C3-NL": {"scientifically_executable": False},
        },
        "task_schema": {},
        "leaderboard_schema": {},
        "leaderboard_entries": [],
        "notes": [],
        "submit_all_is_not_authorization": True,
    }
    documents.update(overrides)
    return documents


def _codes(report: dict, field: str = "errors") -> set[str]:
    return {item["code"] for item in report[field]}


def test_shipped_catalog_is_structurally_valid_and_not_ready() -> None:
    report = validate_catalog(CATALOG)
    assert report["catalog_contract"] == REGISTRY_CONTRACT_VERSION
    assert report["structural_status"] == "STRUCTURALLY_VALID"
    assert report["readiness_status"] == "NOT_READY"
    assert report["overall_status"] == "NOT_READY"
    assert report["errors"] == []
    assert report["executed_external_commands"] == []
    assert report["counts"]["prepared"] == 0
    assert report["counts"]["tasks"] == report["counts"]["structurally_valid_tasks"]
    assert report["counts"]["by_program"] == {"SOTA": 23, "CONDITIONING": 18}
    assert report["submit_all_is_not_authorization"] is True
    assert report["comparison_groups"] == []
    assert "J06" in report["blocked_tasks"]
    assert "PYTHIA" in report["blocked_tasks"]
    assert "GENIE-INFERENCE" in report["blocked_tasks"]
    assert "QWEN-C2" in report["blocked_tasks"]
    assert "QWEN-C3" in report["blocked_tasks"]
    assert "GEMMA-C3-NL" in report["blocked_tasks"]


def test_shipped_report_is_deterministic() -> None:
    first = render_report(validate_catalog(CATALOG))
    second = render_report(validate_catalog(CATALOG))
    assert first == second
    json.loads(first)


def test_dag_matches_versioned_submitter() -> None:
    report = validate_catalog(CATALOG)
    edges = {item["id"]: item["depends_on"] for item in report["dependencies"]}
    submitted: dict[str, list[str]] = {}
    for line in SUBMITTER.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("submit_job "):
            continue
        parts = stripped.split()
        submitted[parts[1]] = parts[3:]
    for task_id, dependencies in submitted.items():
        assert edges[task_id] == sorted(dependencies)
    assert "J06" not in submitted
    assert "PYTHIA" not in submitted
    assert "GENIE-INFERENCE" not in submitted
    plan = json.loads((CATALOG / "plans" / "sota.json").read_text(encoding="utf-8"))
    by_id = {task["id"]: task for task in plan["tasks"]}
    assert by_id["J06"]["included_in_submit_all"] is False
    assert by_id["J07"]["included_in_submit_all"] is False
    assert by_id["J07"]["leaderboard_class"] == "diagnostic_only"
    training = [task["id"] for task in plan["tasks"] if task["training"] is True]
    assert training == ["J13"]
    assert by_id["J13"]["dataset_id"] == "conll04_uie"
    assert by_id["J00"]["executor"]["path"] == "jobs/cesvima/j00_verify_infer_env.sbatch"


def test_shell_hashes_match_benchmark_without_storing_the_instruction() -> None:
    benchmark_src = ROOT.parent / "re_te_benchmark" / "src"
    sys.path.insert(0, str(benchmark_src))
    from re_te_benchmark.conditioning.shell import conditioning_for, shell_template

    plan = json.loads((CATALOG / "plans" / "conditioning.json").read_text(encoding="utf-8"))
    shell = plan["shell"]
    assert shell["instruction_text_stored"] is False
    assert shell["id"] == "common-instruction-shell-v2"
    template = shell_template()
    assert hashlib.sha256(template.encode("utf-8")).hexdigest() == shell["template_sha256"]
    assert "[[CONDITIONING]]" not in json.dumps(plan["tasks"])
    for condition, slot in shell["slots"].items():
        text = conditioning_for(condition)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == slot["sha256"]
        assert len(text) == slot["characters"]
    assert plan["harness_contract"] == CAMPAIGN_CONTRACT_VERSION


def test_registry_package_does_not_invoke_execution_apis() -> None:
    package = ROOT / "src" / "re_te_system" / "campaign_registry"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for banned in ("subprocess", "os.system", "Popen", "from_pretrained", "nvidia_smi"):
        assert banned not in text


def test_duplicate_ids_and_missing_references_are_invalid() -> None:
    duplicated = _documents(tasks=[_task("T"), _task("T")])
    missing = _documents(tasks=[_task("T", system_id="missing-system")])
    assert "duplicate_id" in _codes(validate_documents(duplicated))
    assert "missing_system" in _codes(validate_documents(missing))
    assert validate_documents(duplicated)["structural_status"] == "INVALID"


def test_dependency_cycle_is_invalid() -> None:
    documents = _documents(
        tasks=[
            _task("A", dependencies=["B"]),
            _task("B", dependencies=["A"]),
        ]
    )
    report = validate_documents(documents)
    assert "dependency_cycle" in _codes(report)
    assert report["overall_status"] == "INVALID"
    assert {item["id"] for item in report["dependencies"]} == {"A", "B"}


def test_gold_training_is_prohibited() -> None:
    documents = _documents(
        datasets=[
            {
                "id": "teresia_hohfeld",
                "evaluation_only": True,
                "independent_cohort": True,
                "synthetic": False,
                "prohibited_uses": ["training"],
                "splits": [{"id": "canonical_documents"}],
            }
        ],
        tasks=[
            _task(
                "TRAIN-GOLD",
                dataset_id="teresia_hohfeld",
                split="canonical_documents",
                training=True,
            )
        ],
    )
    assert "gold_training_prohibited" in _codes(validate_documents(documents))


def test_incompatible_and_diagnostic_rows_are_not_ranked_together() -> None:
    hohfeld = {
        "entry_id": "hohfeld",
        "program": "CONDITIONING",
        "protocol_id": "CONDITIONING_PRIMARY",
        "dataset_id": "teresia_hohfeld",
        "split": "canonical_documents",
        "language": "es",
        "task": "conditioned_te",
        "reference_ontology": "hohfeld",
        "prediction_ontology": "hohfeld",
        "ontology_alignment": "same_schema",
        "matching": "exact_multiset_v1",
        "normalization": "unicode_nfc+outer_strip+lowercase_false",
        "eligibility_policy": "strict_inline",
        "estimand": "conservative_operational_f1",
        "comparability": "scientific_comparable",
        "metrics": {"f1": 0.1},
    }
    other = deepcopy(hohfeld)
    other["entry_id"] = "other"
    other["prediction_ontology"] = "wikidata"
    other["metrics"] = {"f1": 0.9}
    diagnostic = deepcopy(hohfeld)
    diagnostic["entry_id"] = "diagnostic"
    diagnostic["protocol_id"] = "CROSS_LINGUAL_DIAGNOSTIC"
    diagnostic["comparability"] = "scientific_comparable"
    synthetic = deepcopy(hohfeld)
    synthetic["entry_id"] = "synthetic"
    synthetic["protocol_id"] = "SYNTHETIC_CONTRACT"
    synthetic["dataset_id"] = "controlled_experiment_synthetic"
    documents = _documents(
        protocols=[
            {
                "id": "CONDITIONING_PRIMARY",
                "comparability": "scientific_comparable",
                "schema_mismatch": False,
                "estimands": ["conservative_operational_f1"],
            },
            {
                "id": "CROSS_LINGUAL_DIAGNOSTIC",
                "comparability": "diagnostic_only",
                "schema_mismatch": True,
                "estimands": ["diagnostic_exact_complete_triplet_f1"],
            },
            {
                "id": "SYNTHETIC_CONTRACT",
                "comparability": "not_comparable",
                "estimands": ["contract_test_not_f1"],
            },
        ],
        datasets=[
            {"id": "teresia_hohfeld", "synthetic": False, "splits": []},
            {"id": "controlled_experiment_synthetic", "synthetic": True, "splits": []},
        ],
        leaderboard_entries=[other, hohfeld, diagnostic, synthetic],
    )
    report = validate_documents(documents)
    assert "unmapped_ontology_ranked" in _codes(report)
    assert "diagnostic_promoted" in _codes(report)
    assert "synthetic_in_scientific_ranking" in _codes(report)
    assert entries_rankable(hohfeld, other) is False
    twin = deepcopy(hohfeld)
    twin["entry_id"] = "twin"
    twin["metrics"] = {"f1": 0.99}
    assert entries_rankable(hohfeld, twin) is True
    assert comparison_key(hohfeld) == comparison_key(twin)
    grouped = validate_documents(
        _documents(leaderboard_entries=[twin, hohfeld])
    )["comparison_groups"]
    assert grouped[0]["entry_ids"] == ["hohfeld", "twin"]


def test_locked_conditions_checkpoint_vram_and_acceptance() -> None:
    locked = _documents(
        tasks=[_task("ARM", program="CONDITIONING", condition="C2", preparation_status="PREPARED")]
    )
    assert "condition_must_remain_blocked" in _codes(validate_documents(locked))
    unresolved = _documents(
        systems=[
            {
                "id": "qwen",
                "programs": ["CONDITIONING"],
                "parser": "tsv",
                "checkpoint_identity": {"kind": "unresolved", "value": None},
                "tokenizer_revision": None,
                "truncates_input": None,
            }
        ],
        tasks=[
            _task(
                "GEN",
                program="CONDITIONING",
                system_id="qwen",
                preparation_status="PREPARED",
                authorization="authorized",
                requires_checkpoint_identity=True,
            )
        ],
    )
    assert "checkpoint_sha_unknown" in _codes(validate_documents(unresolved))
    vram = _documents(
        profiles=[_profile("gpu", gpu_count=1, vram=None)],
        tasks=[
            _task(
                "GPU",
                resource_profile_id="gpu",
                preparation_status="PREPARED",
                authorization="authorized",
            )
        ],
    )
    assert "vram_unverified" in _codes(validate_documents(vram))
    empty = _documents(tasks=[_task("BARE", acceptance_criteria=[])])
    assert "missing_acceptance_criteria" in _codes(validate_documents(empty))


def test_blocked_task_remains_visible_and_is_not_prepared() -> None:
    documents = _documents(
        tasks=[
            _task(
                "HOLD",
                preparation_status="BLOCKED",
                technical_block=True,
                block_reasons=[{"code": "BLOCKED_FIXTURE", "evidence": "missing fixture"}],
            )
        ]
    )
    report = validate_documents(documents)
    assert report["structural_status"] == "STRUCTURALLY_VALID"
    assert report["blocked_tasks"] == ["HOLD"]
    assert report["counts"]["prepared"] == 0
    assert any(item["id"] == "HOLD" for item in report["dependencies"])
