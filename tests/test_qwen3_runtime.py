"""Qwen3 controlled-runtime contract tests.

Fixtures are invented. These tests do not load weights, do not read Hohfeld
Gold, and do not treat an injected string as a Qwen3 generation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from re_te_system.controlled_experiment.campaign import (
    CampaignConfig,
    SyntheticDocument,
    render_controlled_prompt,
    run_controlled_campaign,
)
from re_te_system.controlled_experiment.contracts import (
    SYNTHETIC_CONTRACT_TEST,
    CampaignError,
    ExternalInfrastructureError,
)
from re_te_system.controlled_experiment.dependencies import (
    ensure_package,
    repository_root_from_package,
)
from re_te_system.controlled_experiment.qwen3 import (
    QWEN3_CONTRACT_TEST,
    QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH,
    QWEN3_MODEL_ID,
    Qwen3Extractor,
    Qwen3GenerationContract,
    Qwen3Runtime,
    Qwen3WeightsNotAuthorized,
    load_qwen3_weights,
    observe_installed_libraries,
    readiness,
    serialize_rendered_prompt,
    upstream_facts,
)
from re_te_system.controlled_experiment.synthetic import (
    ScriptedTsvExtractor,
    ScriptedTurn,
    SyntheticTokenCounter,
)
from re_te_system.evaluation_bridge import BridgeRequest, evaluate_campaign
from re_te_system.controlled_experiment.campaign import authorize_campaign

ensure_package("re_te_benchmark", "re_te_benchmark")
from re_te_benchmark.conditioning.shell import conditioning_for, shell_template  # noqa: E402


TEXT = "Alpha supplies Beta in this invented sentence."
LINE = "Alpha\tsupplies\tBeta\n"


def _counter() -> tuple[list[str], Callable[[str], int]]:
    seen: list[str] = []

    def count(prompt: str) -> int:
        seen.append(prompt)
        return len(prompt)

    return seen, count


def _contract(**overrides: object) -> Qwen3GenerationContract:
    values: dict[str, object] = {
        "do_sample": False,
        "effective_context_limit": 100000,
        "max_new_tokens": 32,
        "reserved_output_tokens": 32,
        "seed": 0,
        "test_only": True,
    }
    values.update(overrides)
    return Qwen3GenerationContract(**values)  # type: ignore[arg-type]


def _runtime(
    complete,
    *,
    limit: int = 100000,
    reserved: int = 32,
    test_only: bool = True,
    count: bool = True,
):
    seen, counter = _counter()
    runtime = Qwen3Runtime(
        _contract(
            effective_context_limit=limit,
            reserved_output_tokens=reserved,
            test_only=test_only,
        ),
        count_tokens=counter if count else None,
        counter_id="qwen3-test-character-counter",
        complete=complete,
    )
    return seen, runtime


def test_upstream_identity_is_unpinned_and_weights_stay_unauthorized() -> None:
    facts = upstream_facts()
    assert facts["model_id"] == QWEN3_MODEL_ID
    assert facts["declared_native_context_length"] == QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH == 262144
    assert facts["pinned_checkpoint_revision"] is None
    assert facts["pinned_tokenizer_revision"] is None
    assert facts["token_id_to_string_mapping"] == {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }
    assert facts["pinned_checkpoint_revision"] is None
    assert facts["decoding_recommendation"]["approved"] is False
    assert facts["tokenizer_declared_model_max_length"] != facts["declared_native_context_length"]
    assert facts["chat_template_contains_think_tag"] is False
    assert facts["non_thinking_mode_declared"] is True
    assert facts["upstream_generation_config"]["approved_for_campaign"] is False
    assert facts["weights_authorized"] is False
    report = readiness()
    assert report["executable"] is False
    assert report["real_model_validated"] is False
    assert report["ready_for_gold"] is False
    assert report["ready_for_cesvima"] is False
    assert "pinned_checkpoint_revision_unresolved" in report["blockers"]
    assert "process_timeout_budget_unapproved" in report["blockers"]
    assert report["real_runtime_implemented"] is True
    assert report["real_runtime_executed"] is False
    assert report["status"] == "REAL_RUNTIME_IMPLEMENTED_NOT_EXECUTED"
    assert report["weight_execution_phase_lock"] is True
    assert "weights_not_authorized" in report["blockers"]
    try:
        load_qwen3_weights()
    except Qwen3WeightsNotAuthorized:
        pass
    else:
        raise AssertionError("weight loading must be refused")
    observed = observe_installed_libraries()
    assert observed["weights_loaded"] is False
    try:
        authorize_campaign(QWEN3_CONTRACT_TEST, ("C0", "C1"))
    except CampaignError as exc:
        assert "only SYNTHETIC_CONTRACT_TEST" in str(exc)
    else:
        raise AssertionError("Qwen3 contract test mode must not be authorized")
    try:
        Qwen3Extractor().extract("invented", None)  # type: ignore[arg-type]
    except Qwen3WeightsNotAuthorized:
        pass
    else:
        raise AssertionError("the extractor facade must not generate")


def test_c0_and_c1_share_shell_and_wrapper_and_change_only_the_slot() -> None:
    slot_c0 = conditioning_for("C0")
    slot_c1 = conditioning_for("C1")
    assert slot_c0 == ""
    assert slot_c1 == "relation_labels:\n- Duty\n- Right\n- Privilege\n- NoRight\n"
    template = shell_template()
    rendered_c0 = render_controlled_prompt("C0", TEXT)
    rendered_c1 = render_controlled_prompt("C1", TEXT)
    assert template
    assert rendered_c0 != rendered_c1
    assert TEXT in rendered_c0 and TEXT in rendered_c1
    assert slot_c1 in rendered_c1
    assert slot_c1 not in rendered_c0
    first = serialize_rendered_prompt(rendered_c0)
    second = serialize_rendered_prompt(rendered_c0)
    assert first == second
    wrapped_c1 = serialize_rendered_prompt(rendered_c1)
    prefix = "<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    assert first.startswith(prefix) and first.endswith(suffix)
    assert wrapped_c1.startswith(prefix) and wrapped_c1.endswith(suffix)
    assert first != wrapped_c1
    assert slot_c1 in wrapped_c1
    assert "<think>" not in first and "<think>" not in wrapped_c1


def test_preflight_counts_the_serialized_input_and_blocks_without_measurement() -> None:
    seen, runtime = _runtime(lambda _text, _attempt: {"text": LINE})
    prepared = runtime.prepare("C0", TEXT)
    assert prepared.preflight["serialized_input_counted"] is True
    assert prepared.preflight["real_tokenizer_measured"] is False
    assert prepared.preflight["assurance"] == "test_only_not_model_measurement"
    assert seen == [prepared.serialized_input]
    assert seen[0] != prepared.rendered_prompt
    unmeasured = Qwen3Runtime(Qwen3GenerationContract())
    blocked = unmeasured.prepare("C1", TEXT)
    assert blocked.preflight["assurance"] == "not_assessed"
    assert blocked.preflight["fits"] is None
    assert "tokenizer_assets" in blocked.preflight["missing"]
    outcome = unmeasured.run_attempt("C1", TEXT)
    assert outcome.harness_incident == "preflight_not_assessed"
    assert outcome.generator_calls == 0
    assert outcome.eligible_for_export is False
    assert QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH != outcome.preflight.get("effective_context_limit")


def test_context_block_does_not_truncate_or_call_the_generator() -> None:
    calls: list[str] = []

    def complete(text: str, _attempt: int) -> dict:
        calls.append(text)
        return {"text": LINE}

    seen, runtime = _runtime(complete, limit=1, reserved=1)
    c0 = runtime.run_attempt("C0", TEXT)
    c1 = runtime.run_attempt("C1", TEXT)
    assert calls == []
    assert c0.generator_calls == 0 and c1.generator_calls == 0
    assert c0.final_principal_state == "context_blocked"
    assert c1.final_principal_state == "context_blocked"
    assert c0.preflight["fits"] is False and c1.preflight["fits"] is False
    assert seen[0] == c0.serialized_input
    assert seen[1] == c1.serialized_input
    assert conditioning_for("C1") in c1.serialized_input
    assert len(c1.serialized_input) > len(c0.serialized_input)
    assert c0.contract_sha256 == c1.contract_sha256


def test_injected_completions_preserve_parser_and_retry_semantics() -> None:
    def sequence(outputs: list[object]):
        cursor = {"index": 0}

        def complete(text: str, _attempt: int) -> dict:
            assert text.startswith("<|im_start|>user\n")
            item = outputs[cursor["index"]]
            cursor["index"] += 1
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, dict)
            return item

        return complete

    _seen, completed = _runtime(sequence([{"text": LINE}]))
    done = completed.run_attempt("C0", TEXT)
    assert done.final_principal_state == "completed"
    assert done.raw_output == LINE
    assert done.validated_triples == (("Alpha", "supplies", "Beta"),)
    assert done.eligible_for_export is True
    assert done.attempts[0]["model_output"] == LINE
    assert done.test_only is True

    _seen, empty = _runtime(sequence([{"text": ""}]))
    empty_result = empty.run_attempt("C0", TEXT)
    assert empty_result.final_principal_state == "completed"
    assert empty_result.raw_output == ""
    assert empty_result.validated_triples == ()
    assert empty_result.attempts[0]["diagnostics"]["completed_empty"] is True

    _seen, malformed = _runtime(sequence([{"text": "not a triple\n"}]))
    malformed_result = malformed.run_attempt("C0", TEXT)
    assert malformed_result.validated_triples == ()
    assert malformed_result.attempts[0]["diagnostics"]["fully_malformed"] is True
    assert malformed_result.raw_output == "not a triple\n"

    _seen, partial = _runtime(sequence([{"text": LINE + "still not tsv\n"}]))
    partial_result = partial.run_attempt("C0", TEXT)
    assert partial_result.validated_triples == (("Alpha", "supplies", "Beta"),)
    assert partial_result.attempts[0]["diagnostics"]["partially_malformed"] is True

    _seen, truncated = _runtime(sequence([{"text": LINE, "output_reached_limit": True}]))
    truncated_result = truncated.run_attempt("C0", TEXT)
    assert truncated_result.validated_triples == (("Alpha", "supplies", "Beta"),)
    assert truncated_result.attempts[0]["diagnostics"]["output_truncated"] is True
    assert truncated_result.raw_output == LINE

    _seen, duplicates = _runtime(sequence([{"text": LINE + LINE}]))
    duplicate_result = duplicates.run_attempt("C0", TEXT)
    assert duplicate_result.validated_triples == (
        ("Alpha", "supplies", "Beta"),
        ("Alpha", "supplies", "Beta"),
    )

    _seen, spaced = _runtime(sequence([{"text": "Alpha  Beta\tlinks\tGamma\n"}]))
    spaced_result = spaced.run_attempt("C0", TEXT)
    assert spaced_result.validated_triples == (("Alpha  Beta", "links", "Gamma"),)

    _seen, model_error = _runtime(sequence([RuntimeError("injected model failure")]))
    error = model_error.run_attempt("C0", TEXT)
    assert error.final_principal_state == "model_execution_error"
    assert error.generator_calls == 1
    assert error.raw_output is None
    assert error.attempts[0]["exception_message"] == "injected model failure"
    assert error.eligible_for_export is False

    _seen, external = _runtime(
        sequence([ExternalInfrastructureError("disk"), {"text": LINE}])
    )
    recovered = external.run_attempt("C1", TEXT)
    assert recovered.generator_calls == 2
    assert [item["principal_state"] for item in recovered.attempts] == [
        "external_infrastructure_error",
        "completed",
    ]
    assert recovered.final_principal_state == "completed"

    _seen, timed = _runtime(sequence([TimeoutError("limit"), {"text": LINE}]))
    timed_result = timed.run_attempt("C0", TEXT)
    assert timed_result.attempts[0]["principal_state"] == "timeout"
    assert timed_result.attempts[0]["timeout_enforcement"] == "not_implemented"
    assert timed_result.final_principal_state == "completed"
    assert timed_result.attempts[0]["exception_type"] != "RuntimeError"

    _seen, cut = _runtime(sequence([{"text": LINE, "truncated_input": True}]))
    cut_result = cut.run_attempt("C0", TEXT)
    assert cut_result.final_principal_state == "completed"
    assert cut_result.attempts[0]["diagnostics"]["truncated_input"] is True
    assert cut_result.eligible_for_export is False
    assert cut_result.raw_output == LINE


def test_decoding_configuration_is_identical_across_conditions_and_runs() -> None:
    def complete(_text: str, _attempt: int) -> dict:
        return {"text": LINE}

    _seen, runtime = _runtime(complete)
    first = runtime.run_attempt("C0", TEXT)
    second = runtime.run_attempt("C1", TEXT)
    third = runtime.run_attempt("C0", TEXT)
    assert first.contract_sha256 == second.contract_sha256 == third.contract_sha256
    assert first.serialized_input == third.serialized_input
    assert first.validated_triples == third.validated_triples
    assert first.attempts[0]["seed"] == second.attempts[0]["seed"] == 0
    assert "test_only_configuration" in first.contract_blockers
    assert "weights_not_authorized" in first.contract_blockers
    unauthorized = Qwen3Runtime(
        _contract(test_only=False),
        count_tokens=lambda prompt: len(prompt),
        counter_id="qwen3-test-character-counter",
        complete=complete,
    )
    refused = unauthorized.run_attempt("C0", TEXT)
    assert refused.harness_incident == "preflight_not_assessed"
    assert refused.generator_calls == 0
    assert refused.preflight["assurance"] == "not_assessed"
    missing_generator = Qwen3Runtime(
        _contract(),
        count_tokens=lambda prompt: len(prompt),
        counter_id="qwen3-test-character-counter",
        complete=None,
    )
    missing = missing_generator.run_attempt("C0", TEXT)
    assert missing.harness_incident == "generator_not_authorized"
    assert missing.generator_calls == 0


def test_adapter_output_matches_campaign_export_and_bridge_hash(tmp_path: Path) -> None:
    def complete(_text: str, _attempt: int) -> dict:
        return {"text": LINE}

    _seen, runtime = _runtime(complete)
    produced = runtime.run_attempt("C0", TEXT)
    assert produced.eligible_for_export is True
    campaign = tmp_path / "campaign"
    counter = SyntheticTokenCounter()
    prompt = render_controlled_prompt("C0", TEXT)
    result = run_controlled_campaign(
        CampaignConfig(
            conditions=("C0",),
            cohort=(SyntheticDocument("doc-a", TEXT),),
            count_tokens=counter.count_tokens,
            counter_id=counter.counter_id,
            effective_context_limit=len(prompt) + 8,
            extractor=ScriptedTsvExtractor({("C0", "doc-a", 1): ScriptedTurn("output", output=produced.raw_output)}),
            output_dir=campaign,
            reserved_output_tokens=8,
            synthetic_gold=(
                {
                    "document_id": "doc-a",
                    "eligibility": {
                        "complete_triplet_strict_inline": {"eligible": True, "reason": "synthetic_fixture"}
                    },
                    "id": "gold-a",
                    "object": "Beta",
                    "relation": "supplies",
                    "subject": "Alpha",
                },
            ),
            tokenizer_id=counter.tokenizer_id,
            tokenizer_revision=counter.tokenizer_revision,
        )
    )
    export = (campaign / "conditions" / "C0" / "evaluation_predictions.jsonl").read_text(encoding="utf-8")
    assert '"subject":"Alpha"' in export
    assert "Alpha  Beta" not in export
    claims = result["manifest"]["operational"]["export_sha256"]
    assert claims["C0"]
    summary = json.loads((campaign / "synthetic_evaluation_summary.json").read_text(encoding="utf-8"))
    summary_claim = summary["estimands"]["conservative"]["conditions"]["C0"]["export_sha256"]
    assert summary_claim == claims["C0"]
    gold = tmp_path / "gold.jsonl"
    documents = tmp_path / "documents.jsonl"
    gold.write_text(
        json.dumps(result["evaluation_summary"] and {
            "document_id": "doc-a",
            "eligibility": {
                "complete_triplet_strict_inline": {"eligible": True, "reason": "synthetic_fixture"}
            },
            "id": "gold-a",
            "object": "Beta",
            "relation": "supplies",
            "subject": "Alpha",
        }, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    documents.write_text(
        json.dumps({"document_id": "doc-a", "text": TEXT}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    protocol = repository_root_from_package("re_te_benchmark")
    assert protocol is not None
    report = evaluate_campaign(
        BridgeRequest(
            campaign_dir=campaign,
            documents_path=documents,
            gold_path=gold,
            output_dir=tmp_path / "bridge",
            protocol_path=protocol / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md",
        )
    )
    assert "incorrect_export_hash" not in report["blockers"]
    assert "missing_evaluation_summary" not in report["blockers"]
    assert report["publication_eligibility"]["scientific_result"] is False
    assert report["estimands"]["conservative"]["conditions"]["C0"]["counts"]["tp"] == 1
    manifest_path = campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["mode"] = QWEN3_CONTRACT_TEST
    manifest["identity"]["evaluation_policy"]["result_role"] = "qwen3_contract_test_not_experimental_f1"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (campaign / "synthetic_evaluation_summary.json").unlink()
    real_style = evaluate_campaign(
        BridgeRequest(
            campaign_dir=campaign,
            documents_path=documents,
            gold_path=gold,
            output_dir=tmp_path / "real-style",
            protocol_path=protocol / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md",
        )
    )
    assert "missing_evaluation_summary" not in real_style["blockers"]
    assert "export_hash_unrecorded" not in real_style["blockers"]
    assert "incorrect_export_hash" not in real_style["blockers"]
    assert "synthetic_summary_on_nonsynthetic_campaign" not in real_style["blockers"]
    assert real_style["publication_eligibility"]["scientific_result"] is False
    assert real_style["estimands"]["conservative"]["conditions"]["C0"]["experimental_f1"] is False
