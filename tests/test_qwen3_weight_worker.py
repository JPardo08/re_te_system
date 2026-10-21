"""Checkpoint integrity and the locked smoke worker.

Fixtures are small invented files. They do not stand in for the real shards.
A mocked constructor does not establish that Qwen3 loads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest

from re_te_system.controlled_experiment.campaign import SyntheticDocument, authorize_campaign
from re_te_system.controlled_experiment.contracts import CampaignError
from re_te_system.controlled_experiment.qwen3 import (
    QWEN3_CONTROLLED_EOS_TOKEN_IDS,
    QWEN3_CONTROLLED_PAD_TOKEN_ID,
    QWEN3_EOS_POLICY_ID,
    QWEN3_OBSERVED_DEFAULT_REVISION,
    QWEN3_REAL_MODE,
    QWEN3_SMOKE_INPUT_SCOPE,
    QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
    QWEN3_WEIGHT_SHARDS,
    PersistentJsonWorker,
    Qwen3ConfigurationError,
    Qwen3ExecutionGrant,
    Qwen3GenerationContract,
    Qwen3SmokeAuthorization,
    Qwen3SmokeWorker,
    Qwen3WeightsNotAuthorized,
    Qwen3WorkerTimeout,
    construct_qwen3_causal_lm,
    execution_blockers,
    inspect_qwen3_checkpoint,
    official_weight_manifest,
    proposed_smoke_contract,
    run_qwen3_contract_campaign,
    verify_qwen3_checkpoint,
)
from re_te_system.evaluation_bridge.bridge import evaluate_campaign
from re_te_system.evaluation_bridge.contracts import BridgeRequest

ENGLISH = "Alpha supplies Beta in this invented sentence."
GRANT = Qwen3ExecutionGrant(True, True, False, "injected_contract_test")
PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "re_te_benchmark"
    / "docs"
    / "CONTROLLED_EXPERIMENT_PROTOCOL.md"
)


def _spec(name: str, data: bytes) -> dict[str, object]:
    return {"byte_size": len(data), "filename": name, "sha256": hashlib.sha256(data).hexdigest()}


def _fixture(root: Path, shards: list[tuple[str, bytes]], *, total_size: int | None = None, omit: str | None = None) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in shards:
        if name != omit:
            (root / name).write_bytes(data)
        written.append(_spec(name, data))
    mapping = {f"weight.{name}": name for name, _data in shards}
    total = sum(len(data) for _name, data in shards) if total_size is None else total_size
    index = {"metadata": {"total_size": total}, "weight_map": mapping}
    if omit != "model.safetensors.index.json":
        (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    config = b"config-fixture"
    generation = b"generation-fixture"
    (root / "config.json").write_bytes(config)
    (root / "generation_config.json").write_bytes(generation)
    (root / "revision.txt").write_text("main\n", encoding="utf-8")
    return {
        "config": _spec("config.json", config),
        "generation_config": _spec("generation_config.json", generation),
        "index_filename": "model.safetensors.index.json",
        "index_sha256": None,
        "revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "shards": written,
    }


def _shards() -> list[tuple[str, bytes]]:
    return [
        ("model-00001-of-00003.safetensors", b"shard-one"),
        ("model-00002-of-00003.safetensors", b"shard-two"),
        ("model-00003-of-00003.safetensors", b"shard-three"),
    ]


def _contract(**overrides: object) -> Qwen3GenerationContract:
    values: dict[str, object] = {
        "pinned_checkpoint_revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "pinned_tokenizer_revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "do_sample": False,
        "eos_policy_id": QWEN3_EOS_POLICY_ID,
        "pad_token_id": QWEN3_CONTROLLED_PAD_TOKEN_ID,
        "max_new_tokens": 32,
        "reserved_output_tokens": 32,
        "effective_context_limit": 4096,
        "seed": 42,
        "device_policy": "test_only_unassigned",
        "dtype_policy": "test_only_unassigned",
        "timeout_seconds": 0.4,
        "test_only": True,
    }
    values.update(overrides)
    return Qwen3GenerationContract(**values)  # type: ignore[arg-type]


def _smoke(revision: str = QWEN3_OBSERVED_DEFAULT_REVISION) -> Qwen3SmokeAuthorization:
    return Qwen3SmokeAuthorization(
        granted=True,
        revision=revision,
        input_scope=QWEN3_SMOKE_INPUT_SCOPE,
        device_policy=None,
        dtype_policy=None,
    )


def test_official_manifest_matches_the_observed_shard_identities() -> None:
    manifest = official_weight_manifest()
    assert manifest["index_sha256"] is None
    assert manifest["index_byte_size"] == 32819
    assert manifest["revision"] == QWEN3_OBSERVED_DEFAULT_REVISION
    observed = {(item["filename"], item["byte_size"], item["sha256"]) for item in manifest["shards"]}
    expected = {(item["filename"], item["byte_size"], item["sha256"]) for item in QWEN3_WEIGHT_SHARDS}
    assert observed == expected
    assert QWEN3_WEIGHT_SHARDS[0]["sha256"] == "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6"
    assert QWEN3_WEIGHT_SHARDS[1]["byte_size"] == 3987450520
    assert QWEN3_WEIGHT_SHARDS[2]["byte_size"] == 99630640
    assert QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is True


def test_checkpoint_verifier_fail_closed_on_fixtures(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    manifest = _fixture(root, _shards())
    report = verify_qwen3_checkpoint(root, manifest)
    assert report["verified"] is True
    assert report["revision_file_used"] is False
    assert report["index_sha256_approved"] is False
    assert isinstance(report["index_sha256"], str)
    assert (root / "revision.txt").read_text(encoding="utf-8").strip() == "main"

    with pytest.raises(Qwen3ConfigurationError, match="checkpoint integrity"):
        verify_qwen3_checkpoint(root)

    missing = tmp_path / "missing"
    incomplete = _fixture(missing, _shards(), omit="model-00002-of-00003.safetensors")
    with pytest.raises(Qwen3ConfigurationError, match="missing required checkpoint file"):
        verify_qwen3_checkpoint(missing, incomplete)

    corrupt = tmp_path / "corrupt"
    corrupt_manifest = _fixture(corrupt, _shards())
    shard = corrupt / "model-00003-of-00003.safetensors"
    shard.write_bytes(b"shard-three-mutated")
    with pytest.raises(Qwen3ConfigurationError, match="sha256 or size mismatch"):
        verify_qwen3_checkpoint(corrupt, corrupt_manifest)

    extra = tmp_path / "extra"
    extra_manifest = _fixture(extra, _shards())
    (extra / "model.safetensors").write_bytes(b"extra")
    with pytest.raises(Qwen3ConfigurationError, match="unexpected weight file"):
        verify_qwen3_checkpoint(extra, extra_manifest)

    inconsistent = tmp_path / "index"
    bad_manifest = _fixture(inconsistent, _shards(), total_size=1)
    inspected = inspect_qwen3_checkpoint(inconsistent, bad_manifest)
    assert inspected["verified"] is False
    assert any("total_size" in item for item in inspected["errors"])

    unmapped = tmp_path / "unmapped"
    unmapped_manifest = _fixture(unmapped, _shards())
    index_path = unmapped / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    del payload["weight_map"]["weight.model-00001-of-00003.safetensors"]
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Qwen3ConfigurationError, match="does not reference"):
        verify_qwen3_checkpoint(unmapped, unmapped_manifest)

    foreign = tmp_path / "foreign"
    foreign_manifest = _fixture(foreign, _shards())
    foreign_index = foreign / "model.safetensors.index.json"
    foreign_payload = json.loads(foreign_index.read_text(encoding="utf-8"))
    foreign_payload["weight_map"]["weight.model-00001-of-00003.safetensors"] = "model-00009-of-00003.safetensors"
    foreign_index.write_text(json.dumps(foreign_payload), encoding="utf-8")
    with pytest.raises(Qwen3ConfigurationError, match="unexpected shard"):
        verify_qwen3_checkpoint(foreign, foreign_manifest)


def test_worker_reuses_one_load_and_keeps_the_constructor_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkpoint"
    manifest = _fixture(root, _shards())
    called: list[str] = []

    def boom(*_args: object, **_kwargs: object) -> None:
        called.append("called")
        raise AssertionError("from_pretrained was called")

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", boom, raising=False)
    with pytest.raises(Qwen3WeightsNotAuthorized, match="test-only grant"):
        Qwen3SmokeWorker(GRANT, _contract(), root, manifest=manifest)  # type: ignore[arg-type]
    with pytest.raises(Qwen3ConfigurationError, match="immutable"):
        Qwen3SmokeWorker(_smoke("main"), _contract(pinned_checkpoint_revision="main"), root, manifest=manifest)
    with pytest.raises(Qwen3ConfigurationError, match="reserved output is smaller"):
        Qwen3SmokeWorker(_smoke(), _contract(reserved_output_tokens=8, max_new_tokens=32), root, manifest=manifest)
    with pytest.raises(Qwen3ConfigurationError, match="device and dtype"):
        Qwen3SmokeWorker(_smoke(), _contract(), root, manifest=manifest).load()
    with pytest.raises(Qwen3WeightsNotAuthorized, match="locked"):
        Qwen3SmokeWorker(
            Qwen3SmokeAuthorization(
                True,
                QWEN3_OBSERVED_DEFAULT_REVISION,
                QWEN3_SMOKE_INPUT_SCOPE,
                device_policy="pending_environment",
                dtype_policy="pending_environment",
            ),
            _contract(),
            root,
            manifest=manifest,
        ).load()
    with pytest.raises(Qwen3WeightsNotAuthorized, match="locked"):
        construct_qwen3_causal_lm(
            root,
            Qwen3SmokeAuthorization(
                True,
                QWEN3_OBSERVED_DEFAULT_REVISION,
                QWEN3_SMOKE_INPUT_SCOPE,
                device_policy="pending_environment",
                dtype_policy="pending_environment",
            ),
            _contract(),
            {"verified": True},
        )
    assert called == []

    loads: list[Mapping[str, object]] = []
    seen: list[list[int]] = []

    def loader(report: Mapping[str, object]) -> dict[str, bool]:
        loads.append(report)
        return {"double": True}

    def generator(model: object, input_ids: list[int], kwargs: Mapping[str, object]) -> list[int]:
        assert model == {"double": True}
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs
        assert kwargs["eos_token_id"] == list(QWEN3_CONTROLLED_EOS_TOKEN_IDS)
        assert kwargs["pad_token_id"] == QWEN3_CONTROLLED_PAD_TOKEN_ID
        seen.append(list(input_ids))
        return list(input_ids) + [151645]

    worker = Qwen3SmokeWorker(_smoke(), _contract(), root, manifest=manifest, loader=loader, generator=generator)
    first_ids = [11, 22, 33]
    second_ids = [11, 22, 44]
    kwargs = {
        "do_sample": False,
        "eos_token_id": list(QWEN3_CONTROLLED_EOS_TOKEN_IDS),
        "max_new_tokens": 32,
        "pad_token_id": QWEN3_CONTROLLED_PAD_TOKEN_ID,
    }
    first = worker.generate({"input_ids": first_ids, "model_generate_kwargs": kwargs})
    second = worker.generate({"input_ids": second_ids, "model_generate_kwargs": kwargs})
    assert worker.load_count == 1
    assert len(loads) == 1
    assert loads[0]["verified"] is True
    assert loads[0]["index_sha256_approved"] is False
    assert first["weights_executed"] is False
    assert first["sequence_ids"][:3] == first_ids
    assert second["sequence_ids"][:3] == second_ids
    assert seen == [first_ids, second_ids]
    assert called == []
    with pytest.raises(Qwen3ConfigurationError, match="does not accept text"):
        worker.generate({"input_ids": first_ids, "text": "Alpha", "model_generate_kwargs": kwargs})
    with pytest.raises(Qwen3ConfigurationError, match="sampling parameters"):
        worker.generate({"input_ids": first_ids, "model_generate_kwargs": {**kwargs, "temperature": 0.7}})

    def mismatched(_model: object, _input_ids: list[int], _kwargs: Mapping[str, object]) -> list[int]:
        return [0]

    worker.generator = mismatched
    with pytest.raises(Qwen3ConfigurationError, match="does not start with the preflight ids"):
        worker.generate({"input_ids": first_ids, "model_generate_kwargs": kwargs})


def test_persistent_session_loads_once_and_timeout_is_not_completed(tmp_path: Path) -> None:
    worker_path = tmp_path / "session.py"
    worker_path.write_text(
        "import json, sys, time\n"
        "loaded = 0\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    if message['op'] == 'load':\n"
        "        if message.get('sleep'):\n"
        "            time.sleep(30)\n"
        "        loaded += 1\n"
        "        sys.stdout.write(json.dumps({'ok': True, 'load_count': loaded}) + '\\n')\n"
        "    elif message['op'] == 'generate':\n"
        "        if loaded != 1:\n"
        "            sys.stdout.write(json.dumps({'ok': False}) + '\\n')\n"
        "        else:\n"
        "            ids = message['input_ids']\n"
        "            sys.stdout.write(json.dumps({'ok': True, 'sequence_ids': ids + [151645], 'load_count': loaded}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    import sys

    session = PersistentJsonWorker([sys.executable, str(worker_path)])
    try:
        loaded = session.request({"op": "load"}, 1)
        first = session.request({"op": "generate", "input_ids": [1, 2]}, 1)
        second = session.request({"op": "generate", "input_ids": [1, 3]}, 1)
    finally:
        session.close()
    assert loaded["load_count"] == 1
    assert first["load_count"] == 1
    assert second["load_count"] == 1
    assert first["sequence_ids"] == [1, 2, 151645]
    assert second["sequence_ids"] == [1, 3, 151645]

    sleeping = PersistentJsonWorker([sys.executable, str(worker_path)])
    with pytest.raises(Qwen3WorkerTimeout):
        sleeping.request({"op": "load", "sleep": True}, 0.3)
    assert sleeping.process.poll() is not None
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(sleeping.pids[0], 0)


def test_proposed_smoke_is_not_a_scientific_campaign(tmp_path: Path) -> None:
    contract = proposed_smoke_contract()
    assert contract.seed == 42
    assert contract.max_new_tokens == 256
    assert contract.reserved_output_tokens == 256
    assert contract.effective_context_limit == 4096
    assert contract.do_sample is False
    assert contract.test_only is False
    assert contract.device_policy is None
    assert contract.dtype_policy is None
    assert "test_only_required_for_unexecuted_runtime" in execution_blockers(contract)
    assert "device_policy_unapproved" in contract.blockers()
    assert "reserved_output_smaller_than_max_new_tokens" in execution_blockers(
        _contract(reserved_output_tokens=8, max_new_tokens=32)
    )
    backend_calls: list[object] = []

    class Backend:
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            backend_calls.append(request)
            return {"sequence_ids": request["input_ids"], "termination": "eos"}

    with pytest.raises(Qwen3ConfigurationError, match="test_only_required"):
        run_qwen3_contract_campaign(
            contract,
            (SyntheticDocument("doc-en", ENGLISH),),
            tmp_path / "campaign",
            tokenizer=None,
            backend=Backend(),
            grant=GRANT,
        )
    assert backend_calls == []
    with pytest.raises(CampaignError, match="only SYNTHETIC_CONTRACT_TEST"):
        authorize_campaign(QWEN3_REAL_MODE, ("C0", "C1"))


def test_injected_worker_reaches_the_bridge_without_weights(tmp_path: Path) -> None:
    from re_te_system.controlled_experiment.qwen3 import inspect_tokenizer_assets, load_qwen3_tokenizer

    record = inspect_tokenizer_assets()
    if not record["verified"]:
        pytest.skip("Official tokenizer assets are required for the campaign bridge check.")
    tokenizer = load_qwen3_tokenizer()
    root = tmp_path / "checkpoint"
    manifest = _fixture(root, _shards())
    contract = _contract()
    calls: list[list[int]] = []

    def generator(_model: object, input_ids: list[int], _kwargs: Mapping[str, object]) -> list[int]:
        calls.append(list(input_ids))
        return list(input_ids) + [151645]

    worker = Qwen3SmokeWorker(
        _smoke(),
        contract,
        root,
        manifest=manifest,
        loader=lambda _report: {"double": True},
        generator=generator,
    )
    campaign = tmp_path / "campaign"
    result = run_qwen3_contract_campaign(
        contract,
        (SyntheticDocument("doc-en", ENGLISH),),
        campaign,
        tokenizer=tokenizer,
        backend=worker,
        grant=GRANT,
    )
    assert worker.load_count == 1
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert result["attempts"][0]["diagnostics"]["input_ids"] == calls[0]
    assert result["attempts"][1]["diagnostics"]["input_ids"] == calls[1]
    assert result["attempts"][0]["model_output"] == ""
    assert result["attempts"][0]["diagnostics"]["decoded_raw_output"] == "<|im_end|>"
    assert result["campaign_status"]["real_execution_authorized"] is False
    assert not (campaign / "synthetic_evaluation_summary.json").exists()
    documents = tmp_path / "documents.jsonl"
    gold = tmp_path / "gold.jsonl"
    documents.write_text(
        json.dumps({"document_id": "doc-en", "text": ENGLISH}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {"document_id": "doc-en", "id": "doc-en#1", "subject": "Alpha", "relation": "supplies", "object": "Beta"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
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
    assert report["publication_eligibility"]["technically_scorable"] is not True
    assert "synthetic_summary_on_nonsynthetic_campaign" not in report["blockers"]

    def fail(_model: object, _input_ids: list[int], _kwargs: Mapping[str, object]) -> list[int]:
        raise RuntimeError("model failed")

    failing = Qwen3SmokeWorker(
        _smoke(),
        contract,
        root,
        manifest=manifest,
        loader=lambda _report: {"double": True},
        generator=fail,
    )
    failed = run_qwen3_contract_campaign(
        contract,
        (SyntheticDocument("doc-en", ENGLISH),),
        tmp_path / "failed",
        tokenizer=tokenizer,
        backend=failing,
        grant=GRANT,
    )
    assert {item["principal_state"] for item in failed["attempts"]} == {"model_execution_error"}
    assert failed["campaign_status"]["real_execution_authorized"] is False
    assert not any(item["principal_state"] == "completed" for item in failed["attempts"])
