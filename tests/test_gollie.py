from __future__ import annotations

from dataclasses import replace
from functools import partial
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from re_te_system.cli import main as cli_main
from re_te_system.conditioning.gollie_schema import (
    GOLLIE_BLACK_TARGET_VERSION,
    GOLLIE_PROMPT_SERIALIZER_VERSION,
    GOLLIE_SCHEMA_VERSION,
    GoLLIEArgument,
    GoLLIEClass,
    GoLLIESchema,
    KIND_ENTITY,
    KIND_EVENT,
    KIND_RELATION,
    KIND_TEMPLATE,
    OFFICIAL_RE_ANA_MARY_TEXT,
    official_re_ana_mary_schema,
)
from re_te_system.contracts import ExtractionContext, InputRecord
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.gollie import (
    CANONICAL_GOLLIE_BASE_MODEL,
    CANONICAL_GOLLIE_MODEL_ID,
    CANONICAL_GOLLIE_REVISION,
    GOLLIE_DOCUMENTED_LANGUAGE,
    GOLLIE_MERGED_FULL_MODEL,
    GOLLIE_SCIENTIFIC_ROLE,
    GOLLIE_WEIGHT_LICENSE,
    REAL_GOLLIE_NATIVE_SMOKE,
    GoLLIEConfig,
    GoLLIEExtractor,
    _require_cuda_flash_attention,
    causal_effective_input_limit,
)
from re_te_system.extractors.mock import MockGoLLIEExtractor
from re_te_system.parsing.gollie import (
    align_gollie_records,
    parse_gollie,
    project_gollie_relations,
)
from re_te_system.runner.run import export_evaluation, run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "gollie"
DATASET = {"hash": "gollie-data", "id": "technical", "version": "1"}
SOURCE = {"documents_sha256": "gollie-data", "manifest_sha256": None}
SCHEMA = official_re_ana_mary_schema()
OFFICIAL_RAW = (FIXTURES / "official_re_raw.txt").read_text(encoding="utf-8")
HALLUCINATED_RAW = (FIXTURES / "hallucinated_raw.txt").read_text(encoding="utf-8")
UNSAFE_RAW = (FIXTURES / "unsafe_raw.txt").read_text(encoding="utf-8").strip()
EXPECTED_PROMPT = (FIXTURES / "official_re_expected_prompt.txt").read_text(encoding="utf-8")
ENTITY_SCHEMA = GoLLIESchema(
    schema_id="synthetic-entity-fixture",
    classes=(
        GoLLIEClass(
            name="Person",
            kind=KIND_ENTITY,
            base="Entity",
            definition="A person mention used only to test typed entity parsing.",
            arguments=(GoLLIEArgument(name="span", type_name="str"),),
        ),
    ),
)
EVENT_SCHEMA = GoLLIESchema(
    schema_id="synthetic-event-fixture",
    classes=(
        GoLLIEClass(
            name="Occurrence",
            kind=KIND_EVENT,
            base="Event",
            definition="A synthetic event used only to prove events are not projected.",
            arguments=(
                GoLLIEArgument(name="mention", type_name="str"),
                GoLLIEArgument(name="place", type_name="str"),
            ),
        ),
    ),
)
TEMPLATE_SCHEMA = GoLLIESchema(
    schema_id="synthetic-template-fixture",
    classes=(
        GoLLIEClass(
            name="Summary",
            kind=KIND_TEMPLATE,
            base="Template",
            definition="A synthetic template used only to prove templates are not projected.",
            arguments=(GoLLIEArgument(name="text", type_name="str"),),
        ),
    ),
)
GOLLIE_MODULE_PATHS = (
    REPO_ROOT / "src/re_te_system/conditioning/gollie_schema.py",
    REPO_ROOT / "src/re_te_system/extractors/gollie.py",
    REPO_ROOT / "src/re_te_system/parsing/gollie.py",
)


def smoke_main(argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(
        "run_gollie_native_smoke",
        Path(__file__).resolve().parents[1] / "scripts" / "run_gollie_native_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_gollie(
    root: Path,
    *,
    raw: str = OFFICIAL_RAW,
    text: str = OFFICIAL_RE_ANA_MARY_TEXT,
    schema: GoLLIESchema = SCHEMA,
    fail_on: set[str] | None = None,
    input_language: str = "es",
    language_status: str = "out_of_documented_training_scope",
) -> Path:
    extractor = MockGoLLIEExtractor(
        outputs={"gollie-1": raw},
        fail_on=fail_on or set(),
        schema=schema,
    )
    return run_pipeline(
        inputs=[InputRecord("gollie-1", "gollie-1", text)],
        extractor=extractor,
        output_root=root,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GoLLIEConfig().generation_config(),
        model_metadata={
            "base_model": CANONICAL_GOLLIE_BASE_MODEL,
            "constraint_decoding": False,
            "custom_modeling": True,
            "definitions_guidelines": True,
            "documented_language": GOLLIE_DOCUMENTED_LANGUAGE,
            "dynamic_runtime_schema": True,
            "flash_attention_required": True,
            "formal_ontology": False,
            "interface_accepts_custom_schema": True,
            "merged_full_model": GOLLIE_MERGED_FULL_MODEL,
            "parser_policy": "safe_ast_gollie_v1",
            "quantization": "none",
            "schema": schema.manifest_metadata(),
            "scientific_role": GOLLIE_SCIENTIFIC_ROLE,
            "tokenizer": "mock",
            "weight_license": GOLLIE_WEIGHT_LICENSE,
            "zero_shot_unseen_schema_supported": "documented_with_limitations",
        },
        parse_output=partial(parse_gollie, schema=schema),
        record_aligner=align_gollie_records,
        record_projector=project_gollie_relations,
        run_role="baseline",
        model_family="gollie",
        target_schema_knowledge="structural_schema+definitions_guidelines",
        native_schema={
            "definitions_guidelines": True,
            "dynamic_runtime_schema": True,
            "formal_ontology": False,
            "id": schema.schema_id,
            "inherited_from_model": False,
            "schema_hash": schema.schema_hash(),
            "schema_version": schema.schema_version,
            "scope": "dynamic_runtime_guideline_schema",
        },
        input_language=input_language,
        language_status=language_status,
        task_class="UNIVERSAL_IE",
        controlled_experiment_condition="not_applicable",
        code_commit="test-commit",
    )


def test_gollie_adapter_protocol_and_pinned_configuration_without_weights() -> None:
    adapter = GoLLIEExtractor.__new__(GoLLIEExtractor)
    adapter.model_name = CANONICAL_GOLLIE_MODEL_ID
    adapter.model_revision = CANONICAL_GOLLIE_REVISION
    assert isinstance(adapter, Extractor)
    config = GoLLIEConfig()
    assert config.model_name == CANONICAL_GOLLIE_MODEL_ID
    assert config.revision == CANONICAL_GOLLIE_REVISION
    assert config.local_files_only is True
    assert config.flash_attention is True
    assert config.quantization == "none"
    assert config.do_sample is False
    assert CANONICAL_GOLLIE_BASE_MODEL == "codellama/CodeLlama-7b-hf"
    assert GOLLIE_WEIGHT_LICENSE == "llama2"
    assert GOLLIE_MERGED_FULL_MODEL is True
    assert GOLLIE_DOCUMENTED_LANGUAGE == "en"
    assert GOLLIE_SCIENTIFIC_ROLE == "GUIDELINE_FOLLOWING_UIE_BASELINE"
    assert REAL_GOLLIE_NATIVE_SMOKE == "BLOCKED_GPU"


def test_config_rejects_non_cuda_and_non_flash_attention_variants() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        GoLLIEConfig(device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        GoLLIEConfig(device="mps")
    with pytest.raises(ValueError, match="FlashAttention"):
        GoLLIEConfig(flash_attention=False)
    with pytest.raises(ValueError, match="quantization"):
        GoLLIEConfig(quantization="4bit")


def test_real_extractor_fails_clearly_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        _require_cuda_flash_attention()


def test_schema_preserves_separate_structural_and_guideline_fields() -> None:
    item = SCHEMA.class_by_name()["PersonalSocialRelation"]
    assert item.name == "PersonalSocialRelation"
    assert item.kind == KIND_RELATION
    assert item.base == "Relation"
    assert item.definition.startswith("The Personal-Social Relation describe")
    assert [argument.name for argument in item.arguments] == ["arg1", "arg2"]
    assert [argument.type_name for argument in item.arguments] == ["str", "str"]
    assert item.field_guidelines() == {}
    assert item.examples == ()
    metadata = SCHEMA.manifest_metadata()
    assert metadata["target_schema_knowledge_components"] == [
        "structural_schema",
        "definitions_guidelines",
    ]
    assert metadata["structural_schema"]["classes"][1]["name"] == "PersonalSocialRelation"
    assert "definition" not in metadata["structural_schema"]["classes"][1]
    assert metadata["guidelines"]["classes"][1]["definition"] == item.definition
    assert metadata["black_policy"] == "explicit_black_serialization_contract"
    assert metadata["black_target_version"] == GOLLIE_BLACK_TARGET_VERSION
    assert metadata["prompt_serializer_version"] == GOLLIE_PROMPT_SERIALIZER_VERSION
    assert metadata["schema_version"] == GOLLIE_SCHEMA_VERSION


def test_schema_and_guideline_hashes_are_separated_and_deterministic() -> None:
    first = official_re_ana_mary_schema()
    second = official_re_ana_mary_schema()
    assert first.schema_hash() == second.schema_hash()
    assert first.guideline_hash() == second.guideline_hash()
    definition_changed = GoLLIESchema(
        schema_id=first.schema_id,
        classes=(
            replace(first.classes[0], definition="Altered Physical Relation guideline."),
            first.classes[1],
        ),
    )
    assert definition_changed.schema_hash() == first.schema_hash()
    assert definition_changed.guideline_hash() != first.guideline_hash()
    examples_changed = GoLLIESchema(
        schema_id=first.schema_id,
        classes=(
            replace(first.classes[0], examples=("Ana is near Mary.",)),
            first.classes[1],
        ),
    )
    assert examples_changed.schema_hash() == first.schema_hash()
    assert examples_changed.guideline_hash() != first.guideline_hash()
    structure_changed = GoLLIESchema(
        schema_id=first.schema_id,
        classes=(
            replace(
                first.classes[0],
                arguments=(
                    GoLLIEArgument(name="arg1", type_name="Person"),
                    GoLLIEArgument(name="arg2", type_name="str"),
                ),
            ),
            first.classes[1],
        ),
    )
    assert structure_changed.schema_hash() != first.schema_hash()


def test_prompt_serialization_is_deterministic_and_matches_official_shape() -> None:
    first = SCHEMA.serialize_prompt(OFFICIAL_RE_ANA_MARY_TEXT)
    second = SCHEMA.serialize_prompt(OFFICIAL_RE_ANA_MARY_TEXT)
    assert first == second
    assert first == EXPECTED_PROMPT.rstrip("\n")
    assert first.endswith("result =")
    assert "PhysicalRelation" in first
    assert "PersonalSocialRelation" in first
    assert "The Personal-Social Relation describe" in first
    metadata = SCHEMA.manifest_metadata()
    assert metadata["black_version"] != "unknown"
    assert metadata["schema_hash"] == SCHEMA.schema_hash()
    assert metadata["guideline_hash"] == SCHEMA.guideline_hash()


def test_prompt_is_generated_from_schema_objects_not_runtime_source() -> None:
    for path in GOLLIE_MODULE_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "inspect.getsource" not in text
        assert re.search(r"(?<![\w.])eval\(", text) is None
        assert re.search(r"(?<![\w.])exec\(", text) is None


def test_valid_relation_parse_preserves_typed_arguments() -> None:
    parsed = parse_gollie(OFFICIAL_RAW, "seg-0", schema=SCHEMA)
    assert parsed.triples == ()
    assert parsed.gollie_records[0].class_name == "PersonalSocialRelation"
    assert parsed.gollie_records[0].kind == KIND_RELATION
    assert parsed.gollie_records[0].schema_known is True
    assert [
        (argument.name, argument.value) for argument in parsed.gollie_records[0].arguments
    ] == [("arg1", "Ana"), ("arg2", "Mary")]
    assert all(issue.code != "UNKNOWN_CLASS" for issue in parsed.issues)


def test_valid_entity_parse_is_supported_without_triple_projection() -> None:
    parsed = parse_gollie('[Person(span="Ana")]', schema=ENTITY_SCHEMA)
    assert parsed.gollie_records[0].class_name == "Person"
    assert parsed.gollie_records[0].kind == KIND_ENTITY
    assert parsed.gollie_records[0].schema_known is True
    assert project_gollie_relations(parsed.gollie_records) == ()


def test_unknown_class_is_retained_and_not_projected() -> None:
    parsed = parse_gollie(HALLUCINATED_RAW, schema=SCHEMA)
    assert parsed.gollie_records[0].class_name == "UnknownRelation"
    assert parsed.gollie_records[0].schema_known is False
    assert parsed.gollie_records[0].kind == "unknown"
    assert any(issue.code == "UNKNOWN_CLASS" for issue in parsed.issues)
    assert project_gollie_relations(parsed.gollie_records) == ()


def test_missing_and_unknown_arguments_are_explicit_violations() -> None:
    missing = parse_gollie('[PersonalSocialRelation(arg1="Ana")]', schema=SCHEMA)
    assert missing.gollie_records[0].class_name == "PersonalSocialRelation"
    assert any(issue.code == "MISSING_REQUIRED_ARGUMENT" for issue in missing.issues)
    unknown = parse_gollie(
        '[PersonalSocialRelation(arg1="Ana", arg2="Mary", extra="x")]',
        schema=SCHEMA,
    )
    assert unknown.gollie_records[0].class_name == "PersonalSocialRelation"
    assert any(issue.code == "UNKNOWN_ARGUMENT" for issue in unknown.issues)


def test_argument_type_mismatch_is_explicit() -> None:
    parsed = parse_gollie('[PersonalSocialRelation(arg1=1, arg2="Mary")]', schema=SCHEMA)
    assert parsed.gollie_records[0].arguments[0].value == 1
    assert any(issue.code == "ARGUMENT_TYPE_MISMATCH" for issue in parsed.issues)


def test_unsafe_ast_is_rejected_and_never_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("os.system", lambda command: called.append(command))
    parsed = parse_gollie(UNSAFE_RAW, schema=SCHEMA)
    assert parsed.gollie_records == ()
    assert any(issue.code == "UNSAFE_AST_NODE" for issue in parsed.issues)
    assert called == []


def test_star_args_imports_and_expressions_are_rejected() -> None:
    fixtures = (
        '[PersonalSocialRelation(*["Ana", "Mary"])]',
        '[PersonalSocialRelation(arg1="Ana", **{"arg2": "Mary"})]',
        '[PersonalSocialRelation(arg1="A" + "na", arg2="Mary")]',
        "[lambda: None]",
        "[[x for x in ['Ana']]]",
    )
    for raw in fixtures:
        parsed = parse_gollie(raw, schema=SCHEMA)
        assert parsed.gollie_records == ()
        assert any(
            issue.code in {"UNSAFE_AST_NODE", "INVALID_GOLLIE_SYNTAX"}
            for issue in parsed.issues
        )


def test_invalid_and_empty_outputs_are_explicit() -> None:
    invalid = parse_gollie("not a constructor list", schema=SCHEMA)
    assert invalid.gollie_records == ()
    assert invalid.issues[0].code == "INVALID_GOLLIE_SYNTAX"
    empty = parse_gollie("[]", schema=SCHEMA)
    assert empty.gollie_records == ()
    assert empty.issues[0].code == "EMPTY_STRUCTURE"


def test_alignment_is_exact_ambiguous_or_not_found() -> None:
    parsed = parse_gollie(OFFICIAL_RAW, schema=SCHEMA)
    exact, exact_issues = align_gollie_records(parsed.gollie_records, "Ana visited Mary.")
    assert exact[0].alignment_status == "exact"
    assert exact[0].arguments[0].span_start == 0
    assert exact[0].arguments[1].span_start == 12
    assert exact_issues == ()
    ambiguous, ambiguous_issues = align_gollie_records(
        parsed.gollie_records,
        OFFICIAL_RE_ANA_MARY_TEXT,
    )
    assert ambiguous[0].alignment_status == "ambiguous"
    assert ambiguous[0].arguments[0].span_start is None
    assert ambiguous[0].arguments[1].span_start is None
    assert any(issue.code == "UNKNOWN_SPAN" for issue in ambiguous_issues)
    missing, missing_issues = align_gollie_records(
        parsed.gollie_records,
        "Bob visited Carol.",
    )
    assert missing[0].alignment_status == "not_found"
    assert missing[0].arguments[0].span_start is None
    assert any(issue.code == "UNKNOWN_SPAN" for issue in missing_issues)


def test_relation_projection_uses_class_name_and_arg_values() -> None:
    parsed = parse_gollie(OFFICIAL_RAW, schema=SCHEMA)
    aligned, _issues = align_gollie_records(parsed.gollie_records, "Ana visited Mary.")
    projected = project_gollie_relations(aligned)
    assert [(item.subject, item.relation, item.object) for item in projected] == [
        ("Ana", "PersonalSocialRelation", "Mary")
    ]
    assert projected[0].subject_span == (0, 3)
    assert projected[0].object_span == (12, 16)


def test_entities_events_and_templates_are_not_fabricated_into_triples() -> None:
    entity = parse_gollie('[Person(span="Ana")]', schema=ENTITY_SCHEMA)
    event = parse_gollie('[Occurrence(mention="visit", place="home")]', schema=EVENT_SCHEMA)
    template = parse_gollie('[Summary(text="Ana and Mary")]', schema=TEMPLATE_SCHEMA)
    unknown = parse_gollie(HALLUCINATED_RAW, schema=SCHEMA)
    malformed = parse_gollie('[PersonalSocialRelation(arg1="Ana")]', schema=SCHEMA)
    assert project_gollie_relations(entity.gollie_records) == ()
    assert project_gollie_relations(event.gollie_records) == ()
    assert project_gollie_relations(template.gollie_records) == ()
    assert project_gollie_relations(unknown.gollie_records) == ()
    assert project_gollie_relations(malformed.gollie_records) == ()


def test_duplicates_are_preserved_and_validated(tmp_path: Path) -> None:
    raw = (
        "[\n"
        '    PersonalSocialRelation(arg1="Ana", arg2="Mary"),\n'
        '    PersonalSocialRelation(arg1="Ana", arg2="Mary")\n'
        "]"
    )
    run_dir = run_gollie(tmp_path, raw=raw, text="Ana visited Mary.")
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert len(prediction["validated"]["triples"]) == 2
    assert any(
        issue["code"] == "DUPLICATE_TRIPLE"
        for issue in prediction["validated"]["violations"]
    )
    assert [
        record["class_name"] for record in prediction["parsed_gollie_records"]
    ] == ["PersonalSocialRelation", "PersonalSocialRelation"]


def test_pipeline_preserves_exact_raw_prompt_and_unknown_classes(tmp_path: Path) -> None:
    official = records(run_gollie(tmp_path / "official") / "predictions.jsonl")[0]
    assert official["raw"]["model_output"] == OFFICIAL_RAW
    assert official["raw"]["segments"][0]["model_output"] == OFFICIAL_RAW
    assert official["raw"]["segments"][0]["generation_metadata"]["exact_prompt"] == EXPECTED_PROMPT.rstrip("\n")
    hallucinated = records(
        run_gollie(tmp_path / "hallucinated", raw=HALLUCINATED_RAW) / "predictions.jsonl"
    )[0]
    assert hallucinated["parsed_gollie_records"][0]["class_name"] == "UnknownRelation"
    assert hallucinated["parsed_gollie_records"][0]["schema_known"] is False
    assert hallucinated["validated"]["triples"] == []
    assert any(
        issue["code"] == "PARSE_UNKNOWN_CLASS"
        for issue in hallucinated["validated"]["violations"]
    )


def test_model_failure_is_recorded_and_raw_is_not_invented(tmp_path: Path) -> None:
    run_dir = run_gollie(tmp_path, fail_on={"gollie-1"})
    prediction = records(run_dir / "predictions.jsonl")[0]
    failures = records(run_dir / "failures.jsonl")
    assert prediction["raw"]["model_output"] == ""
    assert any(
        issue["code"] == "PARSE_MODEL_FAILURE"
        for issue in prediction["validated"]["violations"]
    )
    assert failures[0]["stage"] == "extraction"


def test_official_text_alignment_is_ambiguous_not_first_match(tmp_path: Path) -> None:
    prediction = records(run_gollie(tmp_path) / "predictions.jsonl")[0]
    aligned = prediction["aligned_gollie_records"][0]
    assert aligned["alignment_status"] == "ambiguous"
    assert aligned["arguments"][0]["span_start"] is None
    assert aligned["arguments"][1]["span_start"] is None
    assert prediction["validated"]["triples"][0]["subject_span"] is None
    assert prediction["validated"]["triples"][0]["object_span"] is None


def test_relation_projection_is_evaluation_export_compatible(tmp_path: Path) -> None:
    run_dir = run_gollie(tmp_path, text="Ana visited Mary.")
    export = export_evaluation(
        run_dir / "predictions.jsonl",
        tmp_path / "relation-evaluation.jsonl",
    )
    row = records(export)[0]
    assert (row["subject"], row["relation"], row["object"]) == (
        "Ana",
        "PersonalSocialRelation",
        "Mary",
    )


def test_manifest_identifies_gollie_outside_controlled_conditions(tmp_path: Path) -> None:
    manifest = json.loads((run_gollie(tmp_path) / "manifest.json").read_text())
    assert manifest["run_role"] == "baseline"
    assert manifest["model_family"] == "gollie"
    assert manifest["task_class"] == "UNIVERSAL_IE"
    assert manifest["target_schema_knowledge"] == "structural_schema+definitions_guidelines"
    assert manifest["controlled_experiment_condition"] == "not_applicable"
    assert manifest["input_language"] == "es"
    assert manifest["language_status"] == "out_of_documented_training_scope"
    assert manifest["native_schema"]["dynamic_runtime_schema"] is True
    assert manifest["native_schema"]["definitions_guidelines"] is True
    assert manifest["native_schema"]["formal_ontology"] is False
    assert manifest["model"]["scientific_role"] == GOLLIE_SCIENTIFIC_ROLE
    assert manifest["model"]["constraint_decoding"] is False
    assert manifest["model"]["flash_attention_required"] is True
    assert manifest["model"]["merged_full_model"] is True
    assert manifest["model"]["weight_license"] == "llama2"
    assert manifest["model"]["documented_language"] == "en"
    assert "C2" not in json.dumps(manifest)


def test_gollie_cli_marks_english_supported_and_spanish_out_of_scope(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        json.dumps(
            {
                "document_id": "official-re-ana-mary",
                "source_text": OFFICIAL_RE_ANA_MARY_TEXT,
            }
        )
        + "\n"
    )
    schema_file = FIXTURES / "official_re_schema.json"
    english = tmp_path / "english"
    spanish = tmp_path / "spanish"
    assert (
        cli_main(
            [
                "run",
                "--extractor",
                "gollie-mock",
                "--documents",
                str(documents),
                "--output-root",
                str(english),
                "--gollie-schema-file",
                str(schema_file),
                "--input-language",
                "en",
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "run",
                "--extractor",
                "gollie-mock",
                "--documents",
                str(documents),
                "--output-root",
                str(spanish),
                "--gollie-schema-file",
                str(schema_file),
                "--input-language",
                "es",
            ]
        )
        == 0
    )
    english_manifest = json.loads(next(english.glob("run-*/manifest.json")).read_text())
    spanish_manifest = json.loads(next(spanish.glob("run-*/manifest.json")).read_text())
    assert english_manifest["input_language"] == "en"
    assert english_manifest["language_status"] == "supported"
    assert spanish_manifest["input_language"] == "es"
    assert spanish_manifest["language_status"] == "out_of_documented_training_scope"
    assert english_manifest["task_class"] == "UNIVERSAL_IE"
    assert english_manifest["controlled_experiment_condition"] == "not_applicable"


def test_documentation_does_not_claim_c2_or_hohfeld() -> None:
    text = (REPO_ROOT / "docs/GOLLIE_BASELINE.md").read_text(encoding="utf-8")
    assert "GUIDELINE_FOLLOWING_UIE_BASELINE" in text
    assert "GoLLIE is not Controlled C2" in text
    assert "not part of same-model causal contrast" in text
    assert "REAL_GOLLIE_NATIVE_SMOKE = BLOCKED_GPU" in text
    assert "No Hohfeld guideline definitions" in text


def test_causal_context_budget_is_model_max_minus_generation() -> None:
    assert causal_effective_input_limit(16, 4) == 12
    assert causal_effective_input_limit(16384, 128) == 16256
    with pytest.raises(ValueError, match="max_new_tokens"):
        causal_effective_input_limit(8, 8)
    with pytest.raises(ValueError, match="max_new_tokens"):
        causal_effective_input_limit(8, 9)


def test_prompt_is_truncated_to_leave_generation_budget() -> None:
    import torch

    model_max_length = 16
    max_new_tokens = 4
    untruncated_tokens = 20

    class FakeTokenizer:
        eos_token_id = 2

        def __call__(self, text: str, *, add_special_tokens: bool = True):
            return {"input_ids": list(range(untruncated_tokens)) + [self.eos_token_id]}

        def decode(self, ids, **kwargs) -> str:
            return OFFICIAL_RAW

    class FakeModel:
        def generate(self, **kwargs):
            prompt_len = kwargs["input_ids"].shape[-1]
            continuation = list(range(kwargs["max_new_tokens"]))
            return SimpleNamespace(
                sequences=torch.tensor([[0] * prompt_len + continuation])
            )

    adapter = GoLLIEExtractor.__new__(GoLLIEExtractor)
    adapter.schema = SCHEMA
    adapter.config = GoLLIEConfig(max_new_tokens=max_new_tokens)
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeModel()
    adapter.device = torch.device("cpu")
    adapter.model_max_length = model_max_length
    adapter.effective_input_limit = causal_effective_input_limit(
        model_max_length,
        max_new_tokens,
    )
    result = adapter.extract(OFFICIAL_RE_ANA_MARY_TEXT, ExtractionContext("x", "x", None))
    metadata = result.generation_metadata
    assert adapter.effective_input_limit == model_max_length - max_new_tokens
    assert metadata["model_max_length"] == model_max_length
    assert metadata["max_new_tokens"] == max_new_tokens
    assert metadata["effective_input_limit"] == 12
    assert metadata["untruncated_input_token_count"] == untruncated_tokens
    assert metadata["truncated_input"] is True
    assert metadata["input_token_count"] == 12
    assert metadata["input_token_count"] + metadata["max_new_tokens"] <= metadata["model_max_length"]


def test_real_extractor_metadata_and_truncation_without_loading_weights() -> None:
    import torch

    class FakeTokenizer:
        model_max_length = 16384
        eos_token_id = 2

        def __call__(self, text: str, *, add_special_tokens: bool = True):
            return {"input_ids": list(range(10)) + [self.eos_token_id]}

        def decode(self, ids, **kwargs) -> str:
            return OFFICIAL_RAW

    class FakeModel:
        def generate(self, **kwargs):
            prompt_len = kwargs["input_ids"].shape[-1]
            return SimpleNamespace(
                sequences=torch.tensor([[0] * prompt_len + [3, 4, 5]])
            )

    adapter = GoLLIEExtractor.__new__(GoLLIEExtractor)
    adapter.schema = SCHEMA
    adapter.config = GoLLIEConfig(max_new_tokens=2)
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeModel()
    adapter.device = torch.device("cpu")
    adapter.model_max_length = 16384
    adapter.effective_input_limit = 4
    result = adapter.extract(OFFICIAL_RE_ANA_MARY_TEXT, ExtractionContext("x", "x", None))
    assert result.model_output == OFFICIAL_RAW
    assert result.generation_metadata["exact_prompt"] == EXPECTED_PROMPT.rstrip("\n")
    assert result.generation_metadata["truncated_input"] is True
    assert result.generation_metadata["untruncated_input_token_count"] == 10
    assert result.generation_metadata["input_token_count"] == 4
    assert result.generation_metadata["model_max_length"] == 16384
    assert result.generation_metadata["max_new_tokens"] == 2
    assert result.generation_metadata["effective_input_limit"] == 4
    assert result.generation_metadata["output_token_count"] == 3
    assert result.generation_metadata["schema_hash"] == SCHEMA.schema_hash()
    assert result.generation_metadata["guideline_hash"] == SCHEMA.guideline_hash()
    assert result.generation_metadata["prompt_serializer_version"] == GOLLIE_PROMPT_SERIALIZER_VERSION
    assert result.generation_metadata["quantization"] == "none"
    assert result.generation_metadata["flash_attention"] is True


def test_gollie_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = run_gollie(tmp_path / "first")
    second = run_gollie(tmp_path / "second")
    assert first.name == second.name
    for filename in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_native_smoke_script_prepares_command_without_loading_weights(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        smoke_main(
            [
                "--documents",
                str(FIXTURES / "official_re_documents.jsonl"),
                "--gollie-schema-file",
                str(FIXTURES / "official_re_schema.json"),
                "--output-root",
                str(tmp_path / "blocked"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr().out
    assert "REAL_GOLLIE_NATIVE_SMOKE = BLOCKED_GPU" in captured
    assert "not executed" in captured
    assert "--extractor gollie" in captured
    assert "--local-files-only" in captured
