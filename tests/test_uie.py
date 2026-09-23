from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from re_te_system.conditioning.uie_schema import (
    NULL_SPAN,
    SPAN_START,
    TEXT_START,
    TYPE_END,
    TYPE_START,
    UIESchema,
)
from re_te_system.cli import main as cli_main
from re_te_system.contracts import ExtractionContext, InputRecord
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.mock import MockUIEExtractor
from re_te_system.extractors.uie import (
    CANONICAL_UIE_MODEL_ID,
    CANONICAL_UIE_REVISION,
    UIEConfig,
    UIEExtractor,
)
from re_te_system.parsing.sel import (
    align_spot_structures,
    parse_sel,
    project_binary_relations,
)
from re_te_system.runner.run import export_evaluation, run_pipeline


DATASET = {"hash": "uie-data", "id": "technical", "version": "1"}
SOURCE = {"documents_sha256": "uie-data", "manifest_sha256": None}
SCHEMA = UIESchema(
    schema_id="native-relation-fixture",
    spot_labels=("person", "organization"),
    association_labels=("works for",),
    spot_to_association={"person": ("works for",), "organization": ()},
)

ENTITY_SEL = (
    "<pad><extra_id_0>"
    "<extra_id_0> person <extra_id_5> Alice <extra_id_1>"
    "<extra_id_1></s>"
)
RELATION_SEL = (
    "<pad><extra_id_0>"
    "<extra_id_0> person <extra_id_5> Alice "
    "<extra_id_0> works for <extra_id_5> Acme <extra_id_1>"
    "<extra_id_1>"
    "<extra_id_0> organization <extra_id_5> Acme <extra_id_1>"
    "<extra_id_1></s>"
)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_uie(
    root: Path,
    *,
    raw: str = RELATION_SEL,
    text: str = "Alice works for Acme.",
    schema: UIESchema = SCHEMA,
) -> Path:
    extractor = MockUIEExtractor(outputs={"uie-1": raw})
    return run_pipeline(
        inputs=[InputRecord("uie-1", "uie-1", text)],
        extractor=extractor,
        output_root=root,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=UIEConfig().generation_config(),
        model_metadata={
            "base_model": "t5-v1_1-base",
            "constraint_decoding": False,
            "interface_accepts_custom_schema": True,
            "parser_policy": "conservative_sel_v1",
            "schema": schema.manifest_metadata(),
            "tokenizer": "mock",
            "zero_shot_unseen_schema_supported": "unknown",
        },
        parse_output=partial(
            parse_sel,
            spot_labels=schema.spot_labels,
            association_labels=schema.association_labels,
        ),
        structure_aligner=align_spot_structures,
        structure_projector=project_binary_relations,
        run_role="baseline",
        model_family="uie",
        target_schema_knowledge="structural_schema",
        native_schema={
            "id": schema.schema_id,
            "inherited_from_model": False,
            "scope": "dynamic_runtime_structural_schema",
            "schema_hash": schema.stable_hash(),
            "schema_version": schema.schema_version,
        },
        input_language="es",
        language_status="out_of_documented_training_scope",
        task_class="UNIVERSAL_IE",
        controlled_experiment_condition="not_applicable",
        code_commit="test-commit",
    )


def test_uie_adapter_protocol_and_pinned_configuration_without_weights() -> None:
    adapter = UIEExtractor.__new__(UIEExtractor)
    adapter.model_name = CANONICAL_UIE_MODEL_ID
    adapter.model_revision = CANONICAL_UIE_REVISION
    assert isinstance(adapter, Extractor)
    config = UIEConfig()
    assert config.model_name == CANONICAL_UIE_MODEL_ID
    assert config.revision == CANONICAL_UIE_REVISION
    assert config.local_files_only is True
    assert config.constraint_decoding is False


def test_schema_preserves_order_and_builds_deterministic_ssi() -> None:
    schema = UIESchema(
        schema_id="ordered",
        spot_labels=("zeta", "alpha"),
        association_labels=("second", "first"),
        spot_to_association={"zeta": ("second",), "alpha": ("first",)},
    )
    expected = (
        "<spot> zeta<spot> alpha"
        "<asoc> second<asoc> first"
        "<extra_id_2> "
    )
    assert schema.build_ssi() == expected
    assert schema.build_ssi() == expected
    assert schema.manifest_metadata()["ordering_policy"] == "schema_order"
    assert schema.manifest_metadata()["serialized_ssi"] == expected


def test_schema_hash_includes_optional_spot_association_metadata() -> None:
    first = UIESchema("x", ("entity",), ("r",), {"entity": ("r",)})
    second = UIESchema("x", ("entity",), ("r",), {"entity": ()})
    assert first.stable_hash() != second.stable_hash()


def test_canonical_special_tokens_are_explicit() -> None:
    assert (TYPE_START, TYPE_END, TEXT_START, SPAN_START, NULL_SPAN) == (
        "<extra_id_0>",
        "<extra_id_1>",
        "<extra_id_2>",
        "<extra_id_5>",
        "<extra_id_6>",
    )


def test_valid_entity_sel_preserves_typed_structure_without_triple() -> None:
    parsed = parse_sel(
        ENTITY_SEL,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.issues == ()
    assert parsed.structures[0].label == "person"
    assert parsed.structures[0].span == "Alice"
    assert parsed.structures[0].associations == ()
    assert project_binary_relations(parsed.structures) == ()


def test_valid_relation_sel_parses_and_projects_binary_relation() -> None:
    parsed = parse_sel(
        RELATION_SEL,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.issues == ()
    assert len(parsed.structures) == 2
    assert parsed.structures[0].associations[0].label == "works for"
    projected = project_binary_relations(parsed.structures)
    assert [(item.subject, item.relation, item.object) for item in projected] == [
        ("Alice", "works for", "Acme")
    ]


def test_malformed_sel_is_not_silently_repaired() -> None:
    raw = "<extra_id_0><extra_id_0> person Alice <extra_id_1><extra_id_1>"
    parsed = parse_sel(
        raw,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.structures == ()
    assert parsed.issues[0].code == "MALFORMED_SEL"
    assert parsed.issues[0].severity == "hard"


def test_truncated_sel_is_not_auto_closed() -> None:
    raw = (
        "<extra_id_0><extra_id_0> person <extra_id_5> Alice "
        "<extra_id_1>"
    )
    parsed = parse_sel(
        raw,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert len(parsed.structures) == 1
    assert any(issue.code == "TRUNCATED_SEL" for issue in parsed.issues)
    assert any(issue.severity == "hard" for issue in parsed.issues)


def test_unknown_label_is_retained_and_surfaced() -> None:
    raw = (
        "<extra_id_0><extra_id_0> unseen type <extra_id_5> Alice "
        "<extra_id_1><extra_id_1>"
    )
    parsed = parse_sel(
        raw,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.structures[0].label == "unseen type"
    assert any(issue.code == "UNKNOWN_LABEL" for issue in parsed.issues)


def test_unknown_span_is_retained_and_surfaced_during_alignment() -> None:
    parsed = parse_sel(
        ENTITY_SEL,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    aligned, issues = align_spot_structures(
        parsed.structures,
        "Bob works here.",
    )
    assert aligned[0].span == "Alice"
    assert aligned[0].alignment_status == "not_found"
    assert aligned[0].span_start is None
    assert issues[0].code == "UNKNOWN_SPAN"


def test_ambiguous_span_is_not_forced_to_first_occurrence() -> None:
    parsed = parse_sel(
        ENTITY_SEL,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    aligned, issues = align_spot_structures(
        parsed.structures,
        "Alice met Alice.",
    )
    assert aligned[0].alignment_status == "ambiguous"
    assert aligned[0].span_start is None
    assert issues[0].code == "UNKNOWN_SPAN"


def test_unk_span_is_not_repaired() -> None:
    raw = (
        "<extra_id_0><extra_id_0> person <extra_id_5> Al<unk>ce "
        "<extra_id_1><extra_id_1>"
    )
    parsed = parse_sel(
        raw,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.structures[0].span == "Al <unk> ce"
    assert any(issue.code == "UNKNOWN_SPAN" for issue in parsed.issues)


def test_empty_structure_is_explicit_issue() -> None:
    parsed = parse_sel(
        "<pad><extra_id_0><extra_id_1></s>",
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.structures == ()
    assert parsed.issues[0].code == "EMPTY_STRUCTURE"


def test_null_span_is_preserved_structurally_but_not_projected() -> None:
    raw = (
        "<extra_id_0><extra_id_0> person <extra_id_5> <extra_id_6> "
        "<extra_id_1><extra_id_1>"
    )
    parsed = parse_sel(
        raw,
        spot_labels=SCHEMA.spot_labels,
        association_labels=SCHEMA.association_labels,
    )
    assert parsed.structures[0].span == "<extra_id_6>"
    aligned, issues = align_spot_structures(parsed.structures, "Alice")
    assert issues == ()
    assert aligned[0].alignment_status == "null"
    assert project_binary_relations(aligned) == ()


def test_duplicate_relations_are_preserved_and_validated(tmp_path: Path) -> None:
    duplicate = RELATION_SEL.replace(
        "<extra_id_1></s>",
        (
            "<extra_id_0> person <extra_id_5> Alice "
            "<extra_id_0> works for <extra_id_5> Acme <extra_id_1>"
            "<extra_id_1><extra_id_1></s>"
        ),
        1,
    )
    run_dir = run_uie(tmp_path, raw=duplicate)
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert len(prediction["validated"]["triples"]) == 2
    assert any(
        issue["code"] == "DUPLICATE_TRIPLE"
        for issue in prediction["validated"]["violations"]
    )


def test_pipeline_preserves_exact_raw_and_special_tokens(tmp_path: Path) -> None:
    run_dir = run_uie(tmp_path)
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert prediction["raw"]["model_output"] == RELATION_SEL
    assert prediction["raw"]["segments"][0]["model_output"] == RELATION_SEL
    assert "<pad>" in prediction["raw"]["model_output"]
    assert "<extra_id_0>" in prediction["raw"]["model_output"]
    assert "</s>" in prediction["raw"]["model_output"]


def test_pipeline_preserves_parsed_and_aligned_structures(tmp_path: Path) -> None:
    prediction = records(run_uie(tmp_path) / "predictions.jsonl")[0]
    assert prediction["parsed_structures"][0]["alignment_status"] == "not_attempted"
    assert prediction["aligned_structures"][0]["alignment_status"] == "exact"
    assert prediction["aligned_structures"][0]["span_start"] == 0
    assert prediction["aligned_structures"][0]["associations"][0]["span_start"] == 16


def test_entity_only_output_is_not_fabricated_for_evaluation(tmp_path: Path) -> None:
    run_dir = run_uie(tmp_path, raw=ENTITY_SEL, text="Alice arrived.")
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert prediction["validated"]["triples"] == []
    export = export_evaluation(
        run_dir / "predictions.jsonl",
        tmp_path / "entity-evaluation.jsonl",
    )
    assert records(export) == []


def test_relation_projection_is_evaluation_export_compatible(tmp_path: Path) -> None:
    run_dir = run_uie(tmp_path)
    export = export_evaluation(
        run_dir / "predictions.jsonl",
        tmp_path / "relation-evaluation.jsonl",
    )
    row = records(export)[0]
    assert (row["subject"], row["relation"], row["object"]) == (
        "Alice",
        "works for",
        "Acme",
    )


def test_manifest_identifies_uie_outside_controlled_conditions(tmp_path: Path) -> None:
    manifest = json.loads((run_uie(tmp_path) / "manifest.json").read_text())
    assert manifest["run_role"] == "baseline"
    assert manifest["model_family"] == "uie"
    assert manifest["task_class"] == "UNIVERSAL_IE"
    assert manifest["target_schema_knowledge"] == "structural_schema"
    assert manifest["controlled_experiment_condition"] == "not_applicable"
    assert manifest["language_status"] == "out_of_documented_training_scope"
    assert manifest["native_schema"]["scope"] == "dynamic_runtime_structural_schema"
    assert manifest["model"]["constraint_decoding"] is False


def test_uie_cli_marks_native_english_as_supported(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        json.dumps({"document_id": "native", "source_text": "Native English text."})
        + "\n"
    )
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(
        json.dumps(
            {
                "schema_id": "native",
                "spot_labels": ["entity"],
                "association_labels": [],
                "spot_to_association": {"entity": []},
            }
        )
    )
    output = tmp_path / "output"
    assert (
        cli_main(
            [
                "run",
                "--extractor",
                "uie-mock",
                "--documents",
                str(documents),
                "--output-root",
                str(output),
                "--uie-schema-file",
                str(schema_file),
                "--input-language",
                "en",
            ]
        )
        == 0
    )
    manifest_path = next(output.glob("run-*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["input_language"] == "en"
    assert manifest["language_status"] == "supported"


def test_custom_hohfeld_labels_are_interface_fixture_not_capability_claim(
    tmp_path: Path,
) -> None:
    schema = UIESchema(
        schema_id="synthetic-hohfeld-interface-only",
        spot_labels=("generic source", "generic target"),
        association_labels=("Right", "Duty", "Privilege", "NoRight"),
        spot_to_association={
            "generic source": ("Right", "Duty", "Privilege", "NoRight"),
            "generic target": (),
        },
    )
    assert all(label in schema.build_ssi() for label in schema.association_labels)
    manifest = json.loads(
        (run_uie(tmp_path, raw="<extra_id_0><extra_id_1>", schema=schema) / "manifest.json").read_text()
    )
    assert manifest["model"]["interface_accepts_custom_schema"] is True
    assert manifest["model"]["zero_shot_unseen_schema_supported"] == "unknown"


def test_real_extractor_metadata_and_truncation_without_loading_weights() -> None:
    import torch

    class FakeTokenizer:
        model_max_length = 512

        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool = True,
            max_length: int | None = None,
            padding: bool = False,
            truncation: bool = False,
            return_tensors: str | None = None,
        ):
            ids = list(range(10))
            if truncation and max_length is not None:
                ids = ids[:max_length]
            if return_tensors == "pt":
                tensor = torch.tensor([ids])
                return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
            return {"input_ids": ids}

        def decode(self, ids, **kwargs) -> str:
            return "<pad><extra_id_0><extra_id_1></s>"

    class FakeModel:
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=torch.tensor([[0, 32099, 32098, 1]]))

    adapter = UIEExtractor.__new__(UIEExtractor)
    adapter.schema = SCHEMA
    adapter.config = UIEConfig(max_source_tokens=4, max_target_tokens=8)
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeModel()
    adapter.device = torch.device("cpu")
    adapter.model_max_length = 512
    adapter.effective_input_limit = 4
    adapter.ssi = SCHEMA.build_ssi()
    result = adapter.extract("text", ExtractionContext("x", "x", None))
    assert result.model_output == "<pad><extra_id_0><extra_id_1></s>"
    assert result.generation_metadata["decoded_with_special_tokens"] is True
    assert result.generation_metadata["truncated_input"] is True
    assert result.generation_metadata["input_token_count"] == 4
    assert result.generation_metadata["output_token_count"] == 4


def test_uie_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = run_uie(tmp_path / "first")
    second = run_uie(tmp_path / "second")
    assert first.name == second.name
    for filename in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
