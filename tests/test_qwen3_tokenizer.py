"""Official Qwen3 tokenizer checks. These tests do not load model weights.

The revision, file hashes, and token counts belong to the candidate revision
``cdbee75f17c01a7cc42f958dc650907174af0554``. That revision is not an approved
scientific checkpoint. Tests that load the tokenizer skip, with this
prerequisite stated, when the gitignored local assets are absent or do not
match. They do not fall back to the character counter.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from re_te_system.controlled_experiment.campaign import authorize_campaign, render_controlled_prompt
from re_te_system.controlled_experiment.contracts import CampaignError
from re_te_system.controlled_experiment.qwen3 import (
    QWEN3_CONTRACT_TEST,
    QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH,
    QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256,
    QWEN3_OBSERVED_DEFAULT_REVISION,
    QWEN3_SPECIAL_TOKEN_IDS,
    QWEN3_TOKENIZER_ASSETS,
    QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH,
    Qwen3GenerationContract,
    Qwen3Runtime,
    Qwen3TokenizerAssetsError,
    Qwen3WeightsNotAuthorized,
    inspect_tokenizer_assets,
    load_qwen3_tokenizer,
    load_qwen3_weights,
    official_chat_serialization,
    official_input_token_ids,
    readiness,
    serialize_rendered_prompt,
    trace_continuation,
)
from re_te_system.controlled_experiment.dependencies import ensure_package

ensure_package("re_te_benchmark", "re_te_benchmark")
from re_te_benchmark.conditioning.shell import conditioning_for

ENGLISH = "Alpha supplies Beta in this invented sentence."
SPANISH = "El artículo concede un derecho de subvención a la entidad."
C1_SLOT = "relation_labels:\n- Duty\n- Right\n- Privilege\n- NoRight\n"
# Official token counts for ENGLISH at the candidate revision.
ENGLISH_C0_TOKENS = 286
ENGLISH_C1_TOKENS = 303


def _require_tokenizer():
    record = inspect_tokenizer_assets()
    if not record["verified"]:
        pytest.skip(
            "Official Qwen3 tokenizer assets are required at "
            + record["directory"]
            + " for revision "
            + QWEN3_OBSERVED_DEFAULT_REVISION
            + ". A mock counter is not a substitute."
        )
    return load_qwen3_tokenizer()


def test_missing_tokenizer_assets_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(Qwen3TokenizerAssetsError, match="missing or do not match"):
        load_qwen3_tokenizer(tmp_path)


def test_incomplete_or_incorrect_assets_fail_closed(tmp_path: Path) -> None:
    for name, (_digest, size) in QWEN3_TOKENIZER_ASSETS.items():
        (tmp_path / name).write_bytes(b"x" * min(size, 32))
    with pytest.raises(Qwen3TokenizerAssetsError, match="missing or do not match"):
        load_qwen3_tokenizer(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not-a-weight")
    with pytest.raises(Qwen3TokenizerAssetsError, match="model weights"):
        load_qwen3_tokenizer(tmp_path)


def test_revision_identity_and_chat_template() -> None:
    tokenizer = _require_tokenizer()
    record = inspect_tokenizer_assets()
    assert record["revision"] == QWEN3_OBSERVED_DEFAULT_REVISION
    assert record["revision_role"] == "technical_validation_not_scientific_pin"
    assert record["weight_files"] == []
    names = {item["filename"] for item in record["files"]}
    assert names == set(QWEN3_TOKENIZER_ASSETS)
    assert all(item["sha256_matches"] is True for item in record["files"])
    assert type(tokenizer).__name__ == "Qwen2TokenizerFast"
    digest = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    assert digest == QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256
    assert tokenizer.model_max_length == QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH
    assert tokenizer.model_max_length != QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH
    directory = Path(record["directory"])
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((directory / "generation_config.json").read_text(encoding="utf-8"))
    assert config["eos_token_id"] == 151645
    assert config["bos_token_id"] == 151643
    assert "pad_token_id" not in config
    assert config["max_position_embeddings"] == QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH
    assert generation["do_sample"] is True
    assert generation["temperature"] == 0.7
    assert generation["top_p"] == 0.8
    assert generation["top_k"] == 20
    assert generation["eos_token_id"] == [151645, 151643]
    assert generation["pad_token_id"] == 151643
    assert generation["bos_token_id"] == 151643


def test_official_serialization_matches_the_checked_reconstruction() -> None:
    tokenizer = _require_tokenizer()
    for text in (ENGLISH, SPANISH, "Alpha  Beta\tlinks\tGamma", "A\u0301rbol"):
        rendered = render_controlled_prompt("C0", text)
        official = official_chat_serialization(tokenizer, rendered)
        assert official == serialize_rendered_prompt(rendered)
        assert official.count("<|im_start|>") == 2
        assert official.count("<|im_end|>") == 1
        assert "<|endoftext|>" not in official
        assert text in official
        ids = official_input_token_ids(tokenizer, official)
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            tokenize=True,
            add_generation_prompt=True,
        )
        assert ids == list(templated)
        without_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert without_prompt != official
        assert len(official) != len(ids)


def test_c0_and_c1_token_counts_are_reproducible() -> None:
    tokenizer = _require_tokenizer()
    runtime = Qwen3Runtime(
        Qwen3GenerationContract(effective_context_limit=100000, reserved_output_tokens=32),
        tokenizer=tokenizer,
    )
    prepared = {condition: runtime.prepare(condition, ENGLISH) for condition in ("C0", "C1")}
    c0 = prepared["C0"]
    c1 = prepared["C1"]
    assert c0.rendered_prompt != c1.rendered_prompt
    assert c1.rendered_prompt.replace(C1_SLOT, "") == c0.rendered_prompt
    assert conditioning_for("C1") == C1_SLOT
    assert c0.serialization_source == "official_apply_chat_template"
    assert c0.serialization_matches_reconstruction is True
    assert c1.serialization_matches_reconstruction is True
    assert c0.preflight["prompt_tokens"] == ENGLISH_C0_TOKENS
    assert c1.preflight["prompt_tokens"] == ENGLISH_C1_TOKENS
    assert c0.preflight["prompt_tokens"] == len(official_input_token_ids(tokenizer, c0.serialized_input))
    assert c1.preflight["prompt_tokens"] == len(official_input_token_ids(tokenizer, c1.serialized_input))
    repeated = runtime.prepare("C0", ENGLISH)
    assert repeated.serialized_input == c0.serialized_input
    assert repeated.preflight["prompt_tokens"] == ENGLISH_C0_TOKENS
    spanish = runtime.prepare("C0", SPANISH)
    assert SPANISH in spanish.serialized_input
    assert spanish.preflight["prompt_tokens"] != len(spanish.serialized_input)


def test_special_tokens_and_context_boundary_do_not_truncate_or_generate() -> None:
    tokenizer = _require_tokenizer()
    for token, expected in QWEN3_SPECIAL_TOKEN_IDS.items():
        assert tokenizer.convert_tokens_to_ids(token) == expected
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.pad_token_id == 151643
    assert tokenizer.bos_token_id is None
    assert tokenizer.eos_token_id != tokenizer.pad_token_id

    runtime = Qwen3Runtime(
        Qwen3GenerationContract(effective_context_limit=ENGLISH_C0_TOKENS + 8, reserved_output_tokens=8),
        tokenizer=tokenizer,
        complete=lambda _text, _index: {"text": "Alpha\tsupplies\tBeta\n"},
    )
    fitting = runtime.run_attempt("C0", ENGLISH)
    assert fitting.preflight["assurance"] == "tokenizer_measured_not_model_ready"
    assert fitting.preflight["fits"] is True
    assert fitting.preflight["gpu_ready"] is False
    assert fitting.preflight["model_ready"] is False
    assert fitting.preflight["model_compatibility"] is False
    assert fitting.preflight["approved_effective_context_limit"] is None
    assert fitting.preflight["native_context_length"] == 262144
    assert fitting.preflight["tokenizer_declared_model_max_length"] == 1010000
    assert fitting.generator_calls == 0
    assert fitting.harness_incident == "generator_not_authorized"
    assert runtime.serialized_calls == []

    blocked = Qwen3Runtime(
        Qwen3GenerationContract(effective_context_limit=ENGLISH_C0_TOKENS + 7, reserved_output_tokens=8),
        tokenizer=tokenizer,
        complete=lambda _text, _index: {"text": "Alpha\tsupplies\tBeta\n"},
    ).run_attempt("C0", ENGLISH)
    assert blocked.preflight["fits"] is False
    assert blocked.preflight["required_tokens"] == ENGLISH_C0_TOKENS + 8
    assert blocked.final_principal_state == "context_blocked"
    assert blocked.generator_calls == 0
    assert blocked.serialized_input == fitting.serialized_input
    assert len(blocked.serialized_input) == len(fitting.serialized_input)

    missing = Qwen3Runtime(Qwen3GenerationContract(), tokenizer=tokenizer).run_attempt("C0", ENGLISH)
    assert missing.preflight["assurance"] == "not_assessed"
    assert missing.preflight["fits"] is None
    assert "effective_context_limit" in missing.preflight["missing"]
    assert missing.generator_calls == 0


def test_unicode_whitespace_and_continuation_trace() -> None:
    tokenizer = _require_tokenizer()
    nfd = "A\u0301rbol  conserva espacios"
    nfc = unicodedata.normalize("NFC", nfd)
    assert nfd != nfc
    for text in (nfd, nfc):
        rendered = render_controlled_prompt("C0", text)
        official = official_chat_serialization(tokenizer, rendered)
        assert text in official
        assert "  " in official
        ids = official_input_token_ids(tokenizer, official)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert "  " in decoded
        assert unicodedata.normalize("NFC", text) in decoded
    nfd_serialized = official_chat_serialization(tokenizer, render_controlled_prompt("C0", nfd))
    nfc_serialized = official_chat_serialization(tokenizer, render_controlled_prompt("C0", nfc))
    assert nfd in nfd_serialized
    assert nfd not in nfc_serialized
    nfd_ids = official_input_token_ids(tokenizer, nfd_serialized)
    nfc_ids = official_input_token_ids(tokenizer, nfc_serialized)
    assert nfd_ids == nfc_ids
    assert tokenizer.decode(nfd_ids, skip_special_tokens=False) == nfc_serialized

    rendered = render_controlled_prompt("C0", ENGLISH)
    serialized = official_chat_serialization(tokenizer, rendered)
    input_ids = official_input_token_ids(tokenizer, serialized)
    continuation = tokenizer.encode("Alpha\tsupplies\tBeta\n<|im_end|>", add_special_tokens=False)
    traced = trace_continuation(tokenizer, input_ids, [*input_ids, *continuation])
    assert traced["continuation_ids"] == continuation
    assert traced["decoded_raw_output"] == traced["parser_input"]
    assert traced["skip_special_tokens"] is False
    assert "Alpha\tsupplies\tBeta\n" in traced["parser_input"]
    assert "<|im_end|>" in traced["parser_input"]
    with pytest.raises(ValueError, match="does not start"):
        trace_continuation(tokenizer, input_ids, continuation)


def test_tokenizer_validation_does_not_authorize_weights_or_publication() -> None:
    _require_tokenizer()
    report = readiness(Qwen3GenerationContract(effective_context_limit=100000, reserved_output_tokens=32))
    assert report["executable"] is False
    assert report["real_model_validated"] is False
    assert report["ready_for_gold"] is False
    assert report["ready_for_cesvima"] is False
    assert report["weights_authorized"] is False
    assert "weights_not_authorized" in report["blockers"]
    assert "pinned_checkpoint_revision_unresolved" in report["blockers"]
    assert "tokenizer_assets_not_local" not in report["blockers"]
    with pytest.raises(Qwen3WeightsNotAuthorized):
        load_qwen3_weights()
    with pytest.raises(CampaignError, match="only SYNTHETIC_CONTRACT_TEST"):
        authorize_campaign(QWEN3_CONTRACT_TEST, ("C0", "C1"))
