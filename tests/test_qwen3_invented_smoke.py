"""Invented-smoke authorization. Fixtures are not real shards.

A mocked constructor shows that the permit can reach from_pretrained.
It does not show that Qwen3 loads on a GPU. Gold and the scientific
campaign stay closed.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from re_te_system.controlled_experiment.qwen3 import (
    QWEN3_CONTROLLED_EOS_TOKEN_IDS,
    QWEN3_CONTROLLED_PAD_TOKEN_ID,
    QWEN3_INVENTED_SMOKE_DOCUMENTS,
    QWEN3_INVENTED_SMOKE_RESULT_ROLE,
    QWEN3_OBSERVED_DEFAULT_REVISION,
    QWEN3_SMOKE_INPUT_SCOPE,
    QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
    Qwen3ConfigurationError,
    Qwen3ExecutionGrant,
    Qwen3ExecutionNotAuthorized,
    Qwen3InventedSmokePermit,
    Qwen3SmokeAuthorization,
    Qwen3WeightsNotAuthorized,
    claim_fresh_smoke_directory,
    construct_invented_smoke_model,
    construct_qwen3_causal_lm,
    expected_invented_input_ids,
    greedy_continuation,
    handle_invented_smoke_message,
    invented_smoke_authorization,
    invented_smoke_generation_contract,
    invented_smoke_technical_pass,
    issue_invented_smoke_permit,
    load_qwen3_tokenizer,
    load_qwen3_weights,
    official_weight_manifest,
    paired_c0_c1_scientific_plan,
    prepare_invented_smoke_requests,
    proposed_smoke_contract,
    tokenizer_assets_verified,
    tokenizer_directory,
)
from re_te_system.evaluation_bridge.bridge import evaluate_campaign
from re_te_system.evaluation_bridge.contracts import BridgeRequest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cesvima" / "j21_qwen3_invented_smoke.py"
PROTOCOL = ROOT.parent / "re_te_benchmark" / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md"
UNIVERSE_WITHOUT_STRICT_GOLD = (
    "articulo_6",
    "articulo_9",
    "articulo_17",
    "articulo_28",
    "articulo_34",
    "articulo_50",
    "articulo_65",
)


def _verification(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "errors": [],
        "index_sha256": "ab" * 32,
        "index_sha256_approved": False,
        "revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "revision_file_used": False,
        "verified": True,
    }
    payload.update(overrides)
    return payload


def _require_tokenizer():
    if not tokenizer_assets_verified():
        pytest.skip("Official tokenizer assets are required.")
    return load_qwen3_tokenizer()


def test_phase_lock_and_scientific_contract_stay_closed() -> None:
    assert QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is True
    assert official_weight_manifest()["index_sha256"] is None
    proposed = proposed_smoke_contract()
    assert proposed.device_policy is None
    assert proposed.dtype_policy is None
    assert proposed.timeout_seconds is None
    assert proposed.test_only is False
    smoke = invented_smoke_generation_contract()
    assert smoke.do_sample is False
    assert smoke.temperature is None
    assert smoke.max_new_tokens == 256
    assert smoke.seed == 42
    assert smoke.device_policy == "cuda:0"
    assert smoke.dtype_policy == "bfloat16"
    with pytest.raises(Qwen3WeightsNotAuthorized):
        load_qwen3_weights()


def test_permit_rejects_unbounded_or_gold_shaped_authorization() -> None:
    verification = _verification()
    permit = issue_invented_smoke_permit(verification)
    assert permit.index_sha256_approved is False
    assert permit.canonical_document_count == 36
    assert permit.document_ids == tuple(item["document_id"] for item in QWEN3_INVENTED_SMOKE_DOCUMENTS)
    assert all(not document_id.startswith("articulo_") for document_id in permit.document_ids)
    assert set(permit.document_text_sha256).isdisjoint(permit.canonical_text_sha256)
    from re_te_system.controlled_experiment.qwen3 import _require_smoke_authorization

    with pytest.raises(Qwen3WeightsNotAuthorized, match="test grant"):
        _require_smoke_authorization(Qwen3ExecutionGrant(True, True, False, "test"))  # type: ignore[arg-type]
    with pytest.raises(Qwen3ExecutionNotAuthorized, match="bounds"):
        _require_smoke_authorization(
            Qwen3SmokeAuthorization(
                True,
                QWEN3_OBSERVED_DEFAULT_REVISION,
                QWEN3_SMOKE_INPUT_SCOPE,
                device_policy="pending_environment",
                dtype_policy="bfloat16",
                load_timeout_seconds=1200.0,
                generation_timeout_seconds=180.0,
            )
        )
    with pytest.raises(Qwen3ConfigurationError, match="not an approved pin"):
        issue_invented_smoke_permit(_verification(index_sha256_approved=True))
    with pytest.raises(Qwen3ConfigurationError, match="revision.txt"):
        issue_invented_smoke_permit(_verification(revision_file_used=True))
    forged = Qwen3InventedSmokePermit(
        authorization=invented_smoke_authorization(),
        document_ids=("articulo_6",),
        document_text_sha256=permit.document_text_sha256,
        canonical_document_count=36,
        canonical_text_sha256=permit.canonical_text_sha256,
        index_sha256=str(verification["index_sha256"]),
        index_sha256_approved=False,
        verification_revision=QWEN3_OBSERVED_DEFAULT_REVISION,
    )
    with pytest.raises(Qwen3ExecutionNotAuthorized, match="corpus"):
        construct_invented_smoke_model(Path("/absent"), forged, verification)


def test_permit_reaches_mocked_constructor_once_and_old_entry_stays_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification = _verification()
    permit = issue_invented_smoke_permit(verification)
    calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.training = True

        def to(self, device: str) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> "FakeModel":
            self.training = False
            return self

    def fake_from_pretrained(*_args: object, **kwargs: object) -> FakeModel:
        calls.append(dict(kwargs))
        return FakeModel()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_from_pretrained, raising=False)
    model = construct_invented_smoke_model(tmp_path, permit, verification)
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert "temperature" not in calls[0]
    assert "device_map" not in calls[0]
    assert model.device == "cuda:0"
    assert model.training is False
    assert QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is True
    with pytest.raises(Qwen3WeightsNotAuthorized, match="locked"):
        construct_qwen3_causal_lm(
            tmp_path,
            Qwen3SmokeAuthorization(
                True,
                QWEN3_OBSERVED_DEFAULT_REVISION,
                QWEN3_SMOKE_INPUT_SCOPE,
                device_policy="pending_environment",
                dtype_policy="pending_environment",
            ),
            invented_smoke_generation_contract(),
            {"verified": True},
        )
    assert len(calls) == 1


def test_worker_loads_once_and_rejects_foreign_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer = _require_tokenizer()
    verification = _verification()
    constructions: list[object] = []

    class FakeModel:
        device = "cpu"

        def to(self, _device: str) -> "FakeModel":
            return self

        def eval(self) -> "FakeModel":
            return self

        def generate(self, tensor: object, **kwargs: object) -> object:
            assert kwargs["do_sample"] is False
            assert "temperature" not in kwargs
            import torch

            row = tensor[0].tolist()  # type: ignore[index]
            return torch.tensor([row + [QWEN3_CONTROLLED_EOS_TOKEN_IDS[0]]])

    def fake_from_pretrained(*_args: object, **_kwargs: object) -> FakeModel:
        constructions.append("loaded")
        return FakeModel()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_from_pretrained, raising=False)
    state: dict[str, object] = {"load_count": 0, "model": None, "tokenizer": None}
    loaded = handle_invented_smoke_message(
        state,
        {"checkpoint_dir": str(tmp_path), "op": "load", "tokenizer_dir": str(tokenizer_directory())},
        verify=lambda _path: verification,
    )
    again = handle_invented_smoke_message(
        state,
        {"checkpoint_dir": str(tmp_path), "op": "load", "tokenizer_dir": str(tokenizer_directory())},
        verify=lambda _path: verification,
    )
    assert loaded["load_count"] == 1
    assert again["load_count"] == 1
    assert constructions == ["loaded"]
    assert loaded["weights_executed"] is True
    assert loaded["index_sha256_approved"] is False
    document_id = QWEN3_INVENTED_SMOKE_DOCUMENTS[0]["document_id"]
    input_ids = expected_invented_input_ids(tokenizer, document_id, "C0")
    kwargs = {
        "do_sample": False,
        "eos_token_id": list(QWEN3_CONTROLLED_EOS_TOKEN_IDS),
        "max_new_tokens": 256,
        "pad_token_id": QWEN3_CONTROLLED_PAD_TOKEN_ID,
    }
    generated = handle_invented_smoke_message(
        state,
        {
            "condition": "C0",
            "document_id": document_id,
            "input_ids": input_ids,
            "model_generate_kwargs": kwargs,
            "op": "generate",
        },
    )
    assert generated["sequence_ids"][: len(input_ids)] == input_ids
    assert generated["load_count"] == 1
    assert constructions == ["loaded"]
    with pytest.raises(Qwen3ExecutionNotAuthorized, match="not the invented"):
        handle_invented_smoke_message(
            state,
            {
                "condition": "C0",
                "document_id": document_id,
                "input_ids": [1, 2, 3],
                "model_generate_kwargs": kwargs,
                "op": "generate",
            },
        )
    with pytest.raises(Qwen3ConfigurationError, match="does not accept text"):
        handle_invented_smoke_message(state, {"gold": "no", "op": "load"})
    fresh = tmp_path / "fresh-smoke"
    claim_fresh_smoke_directory(fresh)
    with pytest.raises(Qwen3ConfigurationError, match="directory already exists"):
        claim_fresh_smoke_directory(fresh)


def test_greedy_helper_refuses_sampling_keys() -> None:
    class FakeModel:
        device = "cpu"

        def generate(self, tensor: object, **kwargs: object) -> object:
            import torch

            return torch.tensor([tensor[0].tolist() + [151645]])  # type: ignore[index]

    sequence = greedy_continuation(
        FakeModel(),
        [4, 5],
        {
            "do_sample": False,
            "eos_token_id": [151645],
            "max_new_tokens": 256,
            "pad_token_id": 151643,
        },
        seed=42,
    )
    assert sequence == [4, 5, 151645]
    with pytest.raises(Qwen3ConfigurationError, match="sampling parameters"):
        greedy_continuation(
            FakeModel(),
            [4, 5],
            {"do_sample": False, "temperature": 0.7, "eos_token_id": [151645], "max_new_tokens": 256, "pad_token_id": 151643},
            seed=42,
        )


def test_paired_plan_is_predeclared_and_gold_blind() -> None:
    tokenizer = _require_tokenizer()
    plan = paired_c0_c1_scientific_plan()
    encoded = json.dumps(plan, ensure_ascii=False, sort_keys=True)
    assert plan["authorization"]["executable"] is False
    assert plan["authorization"]["execution_permitted"] is False
    assert plan["authorization"]["gold_inference_authorized"] is False
    assert plan["authorization"]["real_execution_authorized"] is False
    assert plan["cohort"]["document_count"] == 36
    assert plan["cohort"]["annotations_used_in_prompts"] is False
    assert plan["cohort"]["excluded_record_fields"] == ["annotations", "provenance"]
    assert plan["conditions"]["C0"]["target_inventory"] is None
    assert plan["conditions"]["C1"]["relation_labels"] == ["Duty", "Right", "Privilege", "NoRight"]
    assert plan["conditions"]["C0"]["slot_sha256"] != plan["conditions"]["C1"]["slot_sha256"]
    assert plan["estimands"]["predeclared"] == ["conservative", "conditional", "paired"]
    assert plan["estimands"]["selected_estimand"] is None
    assert plan["estimands"]["selected_after_observing_scores"] is False
    assert plan["estimands"]["primary_definition"] == "conservative"
    assert plan["evaluation"]["policy"]["matching_policy"] == "exact_multiset_v1"
    assert plan["evaluation"]["policy"]["normalization_policy"] == "unicode_nfc+outer_strip+lowercase_false"
    assert plan["decoding"]["do_sample"] is False
    assert plan["decoding"]["temperature"] is None
    assert plan["decoding"]["approved_as_scientific_freeze"] is False
    assert plan["identity"]["shell_id"] == "common-instruction-shell-v2"
    assert plan["identity"]["index_sha256_approved"] is False
    assert plan["training"] is False
    ids = [item["document_id"] for item in plan["cohort"]["documents"]]
    assert len(ids) == len(set(ids)) == 36
    assert set(UNIVERSE_WITHOUT_STRICT_GOLD).issubset(ids)
    assert all("text" not in item and "source_text" not in item and "annotations" not in item for item in plan["cohort"]["documents"])
    assert "Se prohíbe" not in encoded
    assert "annotated_text" not in encoded
    requests = prepare_invented_smoke_requests(tokenizer)
    assert len(requests) == 4
    assert all(item["fits"] is True for item in requests)
    c0 = {item["document_id"]: item["rendered_prompt_sha256"] for item in requests if item["condition"] == "C0"}
    c1 = {item["document_id"]: item["rendered_prompt_sha256"] for item in requests if item["condition"] == "C1"}
    assert c0.keys() == c1.keys()
    assert all(c0[key] != c1[key] for key in c0)
    assert not invented_smoke_technical_pass([], 0)
    assert invented_smoke_technical_pass(
        [
            {"condition": condition, "document_id": spec["document_id"], "fits": True, "within_budget": True}
            for spec in QWEN3_INVENTED_SMOKE_DOCUMENTS
            for condition in ("C0", "C1")
        ],
        1,
    )


def test_smoke_directory_is_not_a_scientific_bridge_result(tmp_path: Path) -> None:
    campaign = tmp_path / "smoke"
    campaign.mkdir()
    (campaign / "invented_smoke_report.json").write_text(
        json.dumps({"technical_pass": True, "scientific_result": False}),
        encoding="utf-8",
    )
    gold = tmp_path / "gold.jsonl"
    gold.write_text("{}\n", encoding="utf-8")
    documents = tmp_path / "documents.jsonl"
    documents.write_text("{}\n", encoding="utf-8")
    report = evaluate_campaign(
        BridgeRequest(
            campaign_dir=campaign,
            gold_path=gold,
            output_dir=tmp_path / "bridge",
            documents_path=documents,
            protocol_path=PROTOCOL,
        )
    )
    assert report["publication_eligibility"]["scientific_result"] is False


def test_cli_dry_run_and_refused_scientific_command(tmp_path: Path) -> None:
    if not tokenizer_assets_verified():
        pytest.skip("Official tokenizer assets are required.")
    env = dict(**{key: value for key, value in __import__("os").environ.items() if key != "SCRATCH_ROOT"})
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os_pathsep()}{ROOT / 'scripts' / 'cesvima'}"
    dry = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert dry.returncode == 0, dry.stderr
    payload = json.loads(dry.stdout)
    assert payload["local_contract_coherent"] is True
    assert payload["invented_smoke_on_this_machine"] == "NOT_READY"
    assert payload["c0_c1_scientific_execution"] == "NOT_READY"
    assert payload["execute_on_this_machine"] is False
    assert payload["weights_executed"] is False
    assert payload["gold_inference_authorized"] is False
    assert payload["phase_lock"] is True
    assert payload["commands"]["execute_c0_c1"] == "refused"
    assert payload["commands"]["submit_smoke_not_from_cursor"].startswith(
        "sbatch /home/y064/SCRATCH/y064644/paper3/isolated/qwen3-j21/"
    )
    sbatch = (ROOT / "jobs" / "cesvima" / "j21_qwen3_invented_smoke.sbatch").read_text(encoding="utf-8")
    assert "source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh" in sbatch
    assert 'J21_ROOT=/home/y064/SCRATCH/y064644/paper3/isolated/qwen3-j21/re_te_system' in sbatch
    assert 'PYTHONPATH="${J21_ROOT}/src:${J21_ROOT}/scripts/cesvima"' in sbatch
    assert 'RE_TE_BENCHMARK_SRC="${BENCHMARK_ROOT}/src"' in sbatch
    assert '${SYSTEM_ROOT}/scripts/cesvima/j21_qwen3_invented_smoke.py' not in sbatch
    assert "j21" not in (ROOT / "scripts" / "cesvima" / "submit_train_first.sh").read_text(encoding="utf-8")
    refused = subprocess.run(
        [sys.executable, str(SCRIPT), "--execute-c0-c1"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert refused.returncode != 0
    assert "not authorized" in refused.stdout + refused.stderr
    report = tmp_path / "invented_smoke_report.json"
    report.write_text(
        json.dumps(
            {
                "gold_inference_authorized": False,
                "index_sha256": "ab" * 32,
                "index_sha256_approved": False,
                "input_scope": QWEN3_SMOKE_INPUT_SCOPE,
                "load_count": 1,
                "real_execution_authorized": False,
                "result_role": QWEN3_INVENTED_SMOKE_RESULT_ROLE,
                "scientific_result": False,
                "technical_pass": True,
                "weights_executed": True,
            }
        ),
        encoding="utf-8",
    )
    prepared = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prepare-c0-c1",
            "--smoke-report",
            str(report),
            "--output-dir",
            str(tmp_path / "plan"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert prepared.returncode == 0, prepared.stderr
    written = json.loads((tmp_path / "plan" / "c0_c1_paired_plan.json").read_text(encoding="utf-8"))
    assert written["authorization"]["execution_permitted"] is False
    assert written["observed_smoke"]["index_sha256_approved"] is False
    marker = json.loads((tmp_path / "plan" / "NOT_AUTHORIZED").read_text(encoding="utf-8"))
    assert marker["gold_inference_authorized"] is False
    assert not (tmp_path / "plan" / "conditions").exists()
    rejected = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prepare-c0-c1",
            "--smoke-report",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "other"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert rejected.returncode != 0


def os_pathsep() -> str:
    return __import__("os").pathsep
