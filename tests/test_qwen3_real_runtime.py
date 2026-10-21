"""Injected Qwen3 runtime contract. These tests do not load model weights.

The official tokenizer is required for serialization and token ids. A missing
cache skips those tests with that prerequisite stated. Subprocess doubles
stand in for the inference worker. They do not start CUDA and they do not
validate generation quality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from re_te_system.controlled_experiment.campaign import (
    SyntheticDocument,
    authorize_campaign,
    render_controlled_prompt,
)
from re_te_system.controlled_experiment.contracts import CampaignError, sha256_payload, sha256_text
from re_te_system.controlled_experiment.qwen3 import (
    QWEN3_CONTROLLED_EOS_TOKEN_IDS,
    QWEN3_CONTROLLED_PAD_TOKEN_ID,
    QWEN3_EOS_POLICY_ID,
    QWEN3_MODEL_ID,
    QWEN3_OBSERVED_DEFAULT_REVISION,
    QWEN3_REAL_MODE,
    QWEN3_RESULT_ROLE,
    QWEN3_RUNTIME_STATUS,
    QWEN3_SPECIAL_TOKEN_IDS,
    QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
    Qwen3ConfigurationError,
    Qwen3ExecutionGrant,
    Qwen3ExecutionNotAuthorized,
    Qwen3GenerationContract,
    Qwen3WeightsNotAuthorized,
    ScriptedGenerationBackend,
    SubprocessGenerationBackend,
    execute_controlled_document,
    generation_request,
    inspect_tokenizer_assets,
    load_qwen3_model,
    load_qwen3_tokenizer,
    run_qwen3_contract_campaign,
    execution_blockers,
    tokenizer_directory,
)
from re_te_system.controlled_experiment.dependencies import ensure_package
from re_te_system.evaluation_bridge.bridge import evaluate_campaign
from re_te_system.evaluation_bridge.contracts import BridgeRequest

ensure_package("re_te_benchmark", "re_te_benchmark")
from re_te_benchmark.conditioning.shell import conditioning_for

ENGLISH = "Alpha supplies Beta in this invented sentence."
SPANISH = "El artículo concede un derecho de subvención a la entidad."
LINE = "Alpha\tsupplies\tBeta\n"
SPANISH_LINE = "Entidad\tconcede\tDerecho\n"
C1_SLOT = "relation_labels:\n- Duty\n- Right\n- Privilege\n- NoRight\n"
GRANT = Qwen3ExecutionGrant(True, True, False, "injected_contract_test")
WEIGHT_GRANT = Qwen3ExecutionGrant(True, False, True, "must_not_load")
PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "re_te_benchmark"
    / "docs"
    / "CONTROLLED_EXPERIMENT_PROTOCOL.md"
)


def _require_tokenizer():
    record = inspect_tokenizer_assets()
    if not record["verified"]:
        pytest.skip(
            "Official Qwen3 tokenizer assets are required at "
            + record["directory"]
            + ". A mock counter is not a substitute."
        )
    return load_qwen3_tokenizer()


def _contract(**overrides: object) -> Qwen3GenerationContract:
    values: dict[str, object] = {
        "pinned_checkpoint_revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "pinned_tokenizer_revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "do_sample": False,
        "eos_policy_id": QWEN3_EOS_POLICY_ID,
        "pad_token_id": QWEN3_CONTROLLED_PAD_TOKEN_ID,
        "max_new_tokens": 32,
        "seed": 0,
        "effective_context_limit": 100000,
        "reserved_output_tokens": 32,
        "device_policy": "test_only_unassigned",
        "dtype_policy": "test_only_unassigned",
        "timeout_seconds": 0.4,
        "test_only": True,
    }
    values.update(overrides)
    return Qwen3GenerationContract(**values)  # type: ignore[arg-type]


def _ids(tokenizer, text: str, *terminal: int) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False)) + list(terminal)


def _output(tokenizer, document_id: str, text: str) -> dict[str, object]:
    return {
        "kind": "output",
        "continuation_ids": _ids(tokenizer, text, QWEN3_SPECIAL_TOKEN_IDS["<|im_end|>"]),
    }


def test_revision_and_weight_loader_stay_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is True
    moving = Qwen3GenerationContract(pinned_checkpoint_revision="main", pinned_tokenizer_revision="main")
    assert "pinned_checkpoint_revision_not_immutable" in moving.blockers()
    assert "pinned_tokenizer_revision_not_immutable" in moving.blockers()
    assert "checkpoint_revision_not_the_validated_candidate" in execution_blockers(moving)
    with pytest.raises(ValueError, match="sampling is not authorized"):
        Qwen3GenerationContract(do_sample=True)
    with pytest.raises(ValueError, match="does not accept sampling"):
        Qwen3GenerationContract(do_sample=False, temperature=0.7)
    with pytest.raises(Qwen3WeightsNotAuthorized, match="not authorized"):
        load_qwen3_model(_contract(), tmp_path, None)
    with pytest.raises(Qwen3WeightsNotAuthorized, match="not authorized"):
        load_qwen3_model(_contract(), tmp_path, GRANT)
    called: list[str] = []

    def boom(*_args: object, **_kwargs: object) -> None:
        called.append("called")
        raise AssertionError("model constructor was called")

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", boom, raising=False)
    with pytest.raises(Qwen3ConfigurationError, match="missing required checkpoint file"):
        load_qwen3_model(_contract(), tmp_path, WEIGHT_GRANT)
    record = inspect_tokenizer_assets()
    if record["verified"]:
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_bytes((tokenizer_directory() / "config.json").read_bytes())
        (checkpoint / "revision.txt").write_text(QWEN3_OBSERVED_DEFAULT_REVISION + "\n", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"not-a-weight")
        with pytest.raises(Qwen3ConfigurationError, match="checkpoint integrity"):
            load_qwen3_model(_contract(), checkpoint, WEIGHT_GRANT)
    assert called == []
    backend = ScriptedGenerationBackend({})
    with pytest.raises(Qwen3ExecutionNotAuthorized):
        run_qwen3_contract_campaign(
            _contract(),
            (SyntheticDocument("doc-en", ENGLISH),),
            tmp_path / "campaign",
            tokenizer=None,
            backend=backend,
            grant=None,
        )
    assert backend.calls == []
    with pytest.raises(CampaignError, match="only SYNTHETIC_CONTRACT_TEST"):
        authorize_campaign(QWEN3_REAL_MODE, ("C0", "C1"))


def test_c0_c1_chain_reaches_the_bridge_without_a_scientific_result(tmp_path: Path) -> None:
    tokenizer = _require_tokenizer()
    assert tokenizer.decode(_ids(tokenizer, LINE), skip_special_tokens=False) == LINE
    assert tokenizer.decode(_ids(tokenizer, SPANISH_LINE), skip_special_tokens=False) == SPANISH_LINE
    documents = (
        SyntheticDocument("doc-en", ENGLISH),
        SyntheticDocument("doc-es", SPANISH),
    )
    scripts = {}
    for document, line in ((documents[0], LINE), (documents[1], SPANISH_LINE)):
        for condition in ("C0", "C1"):
            scripts[(condition, document.document_id, 1)] = _output(tokenizer, document.document_id, line)
    backend = ScriptedGenerationBackend(scripts)
    first = tmp_path / "campaign-a"
    second = tmp_path / "campaign-b"
    result = run_qwen3_contract_campaign(
        _contract(),
        documents,
        first,
        tokenizer=tokenizer,
        backend=backend,
        grant=GRANT,
    )
    repeated = run_qwen3_contract_campaign(
        _contract(),
        documents,
        second,
        tokenizer=tokenizer,
        backend=ScriptedGenerationBackend(scripts),
        grant=GRANT,
    )
    assert result["manifest"]["identity"] == repeated["manifest"]["identity"]
    assert result["scientific_id"] == repeated["scientific_id"]
    for condition in ("C0", "C1"):
        assert (first / "conditions" / condition / "evaluation_predictions.jsonl").read_text(encoding="utf-8") == (
            second / "conditions" / condition / "evaluation_predictions.jsonl"
        ).read_text(encoding="utf-8")
    identity = result["manifest"]["identity"]
    assert identity["mode"] == QWEN3_REAL_MODE
    assert identity["evaluation_policy"]["result_role"] == QWEN3_RESULT_ROLE
    assert identity["decoding"]["do_sample"] is False
    assert identity["decoding"]["temperature"] is None
    assert identity["decoding"]["eos_token_id"] == [151645]
    assert identity["decoding"]["bitwise_determinism"] is False
    assert identity["context_budget"]["approved_effective_context_limit"] is None
    assert identity["tokenizer"]["revision_role"] == "technical_validation_not_scientific_pin"
    assert result["campaign_status"]["real_execution_authorized"] is False
    assert result["manifest"]["operational"]["runtime_status"] == QWEN3_RUNTIME_STATUS
    assert not (first / "synthetic_evaluation_summary.json").exists()
    assert result["manifest"]["operational"]["export_sha256"]["C0"] == sha256_text(
        (first / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8")
    )
    kwargs = [call["model_generate_kwargs"] for call in backend.calls]
    assert kwargs
    assert all(item == kwargs[0] for item in kwargs)
    assert "temperature" not in kwargs[0]
    assert "top_p" not in kwargs[0]
    assert "top_k" not in kwargs[0]
    assert kwargs[0]["eos_token_id"] == list(QWEN3_CONTROLLED_EOS_TOKEN_IDS)
    assert kwargs[0]["pad_token_id"] == QWEN3_CONTROLLED_PAD_TOKEN_ID
    assert QWEN3_CONTROLLED_PAD_TOKEN_ID not in kwargs[0]["eos_token_id"]
    for attempt, call in zip(result["attempts"], backend.calls, strict=True):
        assert attempt["diagnostics"]["input_ids"] == call["input_ids"]
        assert attempt["preflight"]["prompt_tokens"] == len(call["input_ids"])
        assert attempt["diagnostics"]["input_ids_sha256"] == sha256_payload(attempt["diagnostics"]["input_ids"])
        assert attempt["model_output"] == attempt["diagnostics"]["parser_input"]
        assert attempt["diagnostics"]["decoded_raw_output"].endswith("<|im_end|>")
        assert not attempt["model_output"].endswith("<|im_end|>")
        assert attempt["diagnostics"]["removed_terminal_ids"] == [151645]
        assert attempt["diagnostics"]["input_ids"].count(151644) == 2
        assert attempt["diagnostics"]["input_ids"].count(151645) == 1
        decoded_prompt = tokenizer.decode(attempt["diagnostics"]["input_ids"], skip_special_tokens=False)
        if attempt["document_id"] == "doc-en":
            assert ENGLISH in decoded_prompt
        else:
            assert SPANISH in decoded_prompt
    prompts = {
        (attempt["condition"], attempt["document_id"]): tokenizer.decode(
            attempt["diagnostics"]["input_ids"], skip_special_tokens=False
        )
        for attempt in result["attempts"]
    }
    assert prompts[("C1", "doc-en")].replace(C1_SLOT, "") == prompts[("C0", "doc-en")]
    assert prompts[("C1", "doc-es")].replace(C1_SLOT, "") == prompts[("C0", "doc-es")]
    assert conditioning_for("C1") == C1_SLOT
    c0 = render_controlled_prompt("C0", ENGLISH)
    c1 = render_controlled_prompt("C1", ENGLISH)
    assert c1.replace(C1_SLOT, "") == c0
    preflight_tokens = {attempt["preflight"]["prompt_tokens"] for attempt in result["attempts"]}
    assert len(preflight_tokens) == 4
    assert all(attempt["preflight"]["gpu_ready"] is False for attempt in result["attempts"])
    assert all(attempt["preflight"]["model_ready"] is False for attempt in result["attempts"])
    documents_path = tmp_path / "documents.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    documents_path.write_text(
        "".join(
            json.dumps({"document_id": document.document_id, "text": document.text}, ensure_ascii=False) + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps(
            {"document_id": "doc-en", "id": "doc-en#1", "subject": "Alpha", "relation": "supplies", "object": "Beta"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = evaluate_campaign(
        BridgeRequest(
            campaign_dir=first,
            gold_path=gold_path,
            output_dir=tmp_path / "bridge",
            documents_path=documents_path,
            protocol_path=PROTOCOL,
        )
    )
    assert "incorrect_export_hash" not in report["blockers"]
    assert "export_hash_unrecorded" not in report["blockers"]
    assert "synthetic_summary_on_nonsynthetic_campaign" not in report["blockers"]
    assert report["publication_eligibility"]["scientific_result"] is False
    assert report["publication_eligibility"]["experimental_f1"] is False
    assert report["publication_eligibility"]["technically_scorable"] is not True
    assert list(first.rglob("*.safetensors")) == []


def test_context_block_and_continuation_failures(tmp_path: Path) -> None:
    tokenizer = _require_tokenizer()
    document = SyntheticDocument("doc-en", ENGLISH)
    blocked_backend = ScriptedGenerationBackend({})
    blocked = execute_controlled_document(
        _contract(effective_context_limit=10, reserved_output_tokens=32),
        document,
        "C0",
        tokenizer=tokenizer,
        backend=blocked_backend,
        grant=GRANT,
    )
    assert blocked["backend_calls"] == 0
    assert blocked_backend.calls == []
    assert blocked["state"]["principal_state"] == "context_blocked"
    assert ENGLISH in blocked["prepared"].serialized_input
    assert len(blocked["prepared"].serialized_input) > len(ENGLISH)

    def run(script: dict[tuple[str, str, int], dict[str, object]]) -> dict[str, object]:
        return execute_controlled_document(
            _contract(),
            document,
            "C0",
            tokenizer=tokenizer,
            backend=ScriptedGenerationBackend(script),
            grant=GRANT,
        )

    empty = run({("C0", "doc-en", 1): {"kind": "output", "continuation_ids": [151645]}})
    assert empty["state"]["principal_state"] == "completed"
    assert empty["state"]["diagnostics"]["completed_empty"] is True
    assert empty["attempts"][-1]["diagnostics"]["decoded_raw_output"] == "<|im_end|>"
    assert empty["attempts"][-1]["model_output"] == ""

    malformed = run({("C0", "doc-en", 1): {"kind": "output", "continuation_ids": _ids(tokenizer, "this is not tsv\n", 151645)}})
    assert malformed["state"]["diagnostics"]["fully_malformed"] is True

    partial = run(
        {
            (
                "C0",
                "doc-en",
                1,
            ): {"kind": "output", "continuation_ids": _ids(tokenizer, "Alpha\tsupplies\tBeta\nnot a row\n", 151645)}
        }
    )
    assert partial["state"]["diagnostics"]["partially_malformed"] is True
    assert partial["state"]["parsed_triples"] == 1

    duplicate = run(
        {("C0", "doc-en", 1): {"kind": "output", "continuation_ids": _ids(tokenizer, LINE + LINE, 151645)}}
    )
    assert duplicate["state"]["parsed_triples"] == 2

    limited = run(
        {("C0", "doc-en", 1): {"kind": "output", "continuation_ids": _ids(tokenizer, LINE), "termination": "length"}}
    )
    assert limited["state"]["diagnostics"]["output_truncated"] is True
    assert limited["state"]["parsed_triples"] == 1
    assert "truncated_input" not in limited["state"]["diagnostics"]

    padded = run({("C0", "doc-en", 1): {"kind": "output", "continuation_ids": _ids(tokenizer, LINE, 151643)}})
    assert padded["attempts"][-1]["diagnostics"]["removed_terminal_ids"] == [151643]
    assert padded["attempts"][-1]["diagnostics"]["decoded_raw_output"].endswith("<|endoftext|>")
    assert padded["attempts"][-1]["model_output"] == LINE
    assert padded["attempts"][-1]["diagnostics"]["parser_input"] == LINE

    embedded_ids = _ids(tokenizer, LINE, 151645) + _ids(tokenizer, "Gamma\tlinks\tDelta\n", 151645)
    embedded = run({("C0", "doc-en", 1): {"kind": "output", "continuation_ids": embedded_ids}})
    assert embedded["attempts"][-1]["diagnostics"]["removed_terminal_ids"] == [151645]
    assert "<|im_end|>" in embedded["attempts"][-1]["model_output"]
    assert 151645 in embedded["attempts"][-1]["diagnostics"]["continuation_ids"][:-1]

    failed = run({("C0", "doc-en", 1): {"kind": "error", "message": "backend failed"}})
    assert failed["backend_calls"] == 1
    assert failed["state"]["principal_state"] == "model_execution_error"
    assert "model_output" not in failed["attempts"][-1]

    class BadPrefix:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, request: dict[str, object]) -> dict[str, object]:
            self.calls.append(request)
            return {"sequence_ids": [1], "termination": "eos"}

    bad = BadPrefix()
    mismatched = execute_controlled_document(
        _contract(),
        document,
        "C0",
        tokenizer=tokenizer,
        backend=bad,
        grant=GRANT,
    )
    assert mismatched["state"]["principal_state"] == "model_execution_error"
    assert bad.calls[0]["input_ids"] == mismatched["input_ids"]

    recovered = run(
        {
            ("C0", "doc-en", 1): {"kind": "infrastructure", "message": "disk"},
            ("C0", "doc-en", 2): {"kind": "output", "continuation_ids": _ids(tokenizer, LINE, 151645)},
        }
    )
    assert recovered["backend_calls"] == 2
    assert recovered["state"]["principal_state"] == "completed"
    assert recovered["state"]["also_observed"][0]["principal_state"] == "external_infrastructure_error"
    assert recovered["attempts"][-1]["model_output"] == LINE


def test_subprocess_timeout_is_enforced_without_a_model(tmp_path: Path) -> None:
    tokenizer = _require_tokenizer()
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys, time\n"
        "payload = json.loads(sys.stdin.read())\n"
        "if payload['attempt_index'] == 1:\n"
        "    time.sleep(30)\n"
        "print(json.dumps({'sequence_ids': payload['input_ids'], 'termination': 'eos'}))\n",
        encoding="utf-8",
    )
    backend = SubprocessGenerationBackend([os.environ.get("PYTHON", "") or __import__("sys").executable, str(worker)])
    document = SyntheticDocument("doc-en", ENGLISH)
    outcome = execute_controlled_document(
        _contract(),
        document,
        "C1",
        tokenizer=tokenizer,
        backend=backend,
        grant=GRANT,
    )
    assert outcome["backend_calls"] == 2
    assert outcome["state"]["also_observed"][0]["principal_state"] == "timeout"
    assert outcome["state"]["principal_state"] == "completed"
    assert outcome["attempts"][0]["timeout_enforcement"] == "subprocess_process_group"
    assert outcome["attempts"][0]["worker_reaped"] is True
    assert outcome["state"]["diagnostics"]["completed_empty"] is True
    with pytest.raises(ProcessLookupError):
        os.kill(backend.pids[0], 0)
    assert backend.calls[0]["model_generate_kwargs"] == backend.calls[1]["model_generate_kwargs"]
    assert backend.calls[0]["condition"] == "C1"

    sleeping = tmp_path / "sleep.py"
    sleeping.write_text(
        "import json, sys, time\njson.loads(sys.stdin.read())\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    exhausted = SubprocessGenerationBackend([__import__("sys").executable, str(sleeping)])
    timed_out = execute_controlled_document(
        _contract(),
        document,
        "C0",
        tokenizer=tokenizer,
        backend=exhausted,
        grant=GRANT,
    )
    assert timed_out["state"]["principal_state"] == "timeout"
    assert timed_out["backend_calls"] == 2
    assert "model_output" not in timed_out["attempts"][-1]
    for pid in exhausted.pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    campaign = tmp_path / "timeout-campaign"
    recorded = run_qwen3_contract_campaign(
        _contract(),
        (document,),
        campaign,
        tokenizer=tokenizer,
        backend=SubprocessGenerationBackend([__import__("sys").executable, str(sleeping)]),
        grant=GRANT,
    )
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    assert status["real_execution_authorized"] is False
    assert status["status"] != "operationally_complete_publishable"
    assert recorded["document_states"][0]["principal_state"] == "timeout"
    assert recorded["document_states"][1]["principal_state"] == "timeout"
    assert not (campaign / "synthetic_evaluation_summary.json").exists()
    attempt_text = (campaign / "attempts.jsonl").read_text(encoding="utf-8")
    assert '"principal_state":"timeout"' in attempt_text
    assert '"principal_state":"completed"' not in attempt_text

    crashing = tmp_path / "crash.py"
    crashing.write_text("import sys\nsys.stdin.read()\nsys.exit(3)\n", encoding="utf-8")
    crashed = SubprocessGenerationBackend([__import__("sys").executable, str(crashing)])
    failed = execute_controlled_document(
        _contract(),
        document,
        "C0",
        tokenizer=tokenizer,
        backend=crashed,
        grant=GRANT,
    )
    assert failed["backend_calls"] == 1
    assert failed["state"]["principal_state"] == "model_execution_error"


def test_preflight_ids_match_the_request_and_native_limits_are_not_copied() -> None:
    tokenizer = _require_tokenizer()
    request = generation_request(
        _contract(),
        [151644, 872],
        condition="C0",
        document_id="doc-en",
        attempt_index=1,
    )
    assert request["input_ids"] == [151644, 872]
    assert request["model_generate_kwargs"]["do_sample"] is False
    assert "temperature" not in request["model_generate_kwargs"]
    missing = execution_blockers(Qwen3GenerationContract())
    assert "effective_context_limit_unapproved" in missing
    assert "process_timeout_budget_unapproved" in missing
    assert "greedy_decoding_not_selected" in missing
    with pytest.raises(Qwen3ConfigurationError, match="effective_context_limit_copies_native_context"):
        execute_controlled_document(
            _contract(effective_context_limit=262144),
            SyntheticDocument("doc-en", ENGLISH),
            "C0",
            tokenizer=tokenizer,
            backend=ScriptedGenerationBackend({}),
            grant=GRANT,
        )
