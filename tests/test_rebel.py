from __future__ import annotations

import json
from pathlib import Path

from re_te_system.contracts import InputRecord
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.mock import MockRebelExtractor
from re_te_system.extractors.rebel import RebelConfig, RebelExtractor
from re_te_system.parsing.rebel import parse_rebel
from re_te_system.runner.run import (
    WindowingConfig,
    export_evaluation,
    run_pipeline,
)


DATASET = {"hash": "rebel-data", "id": "test", "version": "1"}
SOURCE = {"documents_sha256": "rebel-data", "manifest_sha256": "manifest"}


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_rebel(
    root: Path,
    *,
    inputs: list[InputRecord] | None = None,
    extractor: MockRebelExtractor | None = None,
    windowing: WindowingConfig = WindowingConfig(),
) -> Path:
    return run_pipeline(
        inputs=inputs or [InputRecord("d1", "d1", "Empresa informa trabajador.")],
        extractor=extractor or MockRebelExtractor(),
        output_root=root,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=RebelConfig().generation_config(),
        windowing=windowing,
        model_metadata={
            "device": "none",
            "dtype": "none",
            "language": {"input": "es", "language_tokens": "none"},
            "tokenizer": "none",
        },
        parse_output=parse_rebel,
        run_role="baseline",
        model_family="rebel",
        target_schema_knowledge="none",
        native_schema={"id": "wikidata_like", "inherited_from_model": True},
        input_language="es",
        language_status="out_of_primary_model_scope",
        code_commit="test-commit",
    )


def test_rebel_adapter_conforms_to_shared_interface_without_loading_model() -> None:
    adapter = RebelExtractor.__new__(RebelExtractor)
    adapter.model_name = "Babelscape/rebel-large"
    adapter.model_revision = "test"
    assert isinstance(adapter, Extractor)


def test_rebel_config_is_deterministic_and_historically_explicit() -> None:
    assert RebelConfig().generation_config() == {
        "do_sample": False,
        "early_stopping": False,
        "length_penalty": 0.0,
        "max_length": 512,
        "num_beams": 3,
        "num_return_sequences": 1,
        "seed": 42,
        "temperature": None,
    }
    assert RebelConfig().model_name == "Babelscape/rebel-large"


def test_rebel_parser_one_and_multiple_triples() -> None:
    one = parse_rebel("<s><triplet> London <subj> England <obj> capital of</s>")
    many = parse_rebel(
        "<triplet> London <subj> England <obj> capital of "
        "<triplet> England <subj> Europe <obj> part of"
    )
    assert (one.triples[0].subject, one.triples[0].object, one.triples[0].relation) == (
        "London",
        "England",
        "capital of",
    )
    assert len(many.triples) == 2


def test_rebel_parser_malformed_truncated_and_empty() -> None:
    malformed = parse_rebel("<triplet> A <obj> B <subj> r")
    truncated = parse_rebel("<triplet> A <subj> B")
    empty = parse_rebel("<pad> </s>")
    assert malformed.issues[0].code == "MALFORMED_CONTROL_ORDER"
    assert truncated.issues[0].code == "TRUNCATED_TRIPLE"
    assert empty.issues[0].code == "EMPTY_MODEL_OUTPUT"


def test_rebel_parser_and_pipeline_preserve_duplicates_and_raw(tmp_path: Path) -> None:
    raw = (
        "<s> <triplet> A <subj> B <obj> native relation "
        "<triplet> A <subj> B <obj> native relation </s>"
    )
    run_dir = run_rebel(tmp_path, extractor=MockRebelExtractor(outputs={"d1": raw}))
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert prediction["raw"]["model_output"] == raw
    assert len(prediction["parsed"]) == 2
    assert len(prediction["normalized"]) == 2
    assert len(prediction["validated"]["triples"]) == 2
    assert prediction["validated"]["violations"][0]["code"] == "DUPLICATE_TRIPLE"


def test_rebel_manifest_has_role_schema_and_language_metadata(tmp_path: Path) -> None:
    manifest = json.loads((run_rebel(tmp_path) / "manifest.json").read_text())
    assert manifest["run_role"] == "baseline"
    assert manifest["model_family"] == "rebel"
    assert manifest["target_condition"] == {
        "id": "C0",
        "target_schema_knowledge": "none",
    }
    assert manifest["native_schema"] == {
        "id": "wikidata_like",
        "inherited_from_model": True,
    }
    assert manifest["input_language"] == "es"
    assert manifest["language_status"] == "out_of_primary_model_scope"


def test_rebel_has_no_target_schema_leakage_or_rescue(tmp_path: Path) -> None:
    native = "<triplet> A <subj> B <obj> country of citizenship"
    run_dir = run_rebel(tmp_path, extractor=MockRebelExtractor(outputs={"d1": native}))
    prediction = records(run_dir / "predictions.jsonl")[0]
    serialized = json.dumps(prediction, ensure_ascii=False)
    assert prediction["normalized"][0]["relation"] == "country of citizenship"
    assert all(label not in serialized for label in ("Duty", "Right", "Privilege", "NoRight"))
    assert set(prediction["input"]) == {"text"}


def test_rebel_document_aggregation_and_evaluation_export(tmp_path: Path) -> None:
    extractor = MockRebelExtractor(
        outputs={
            "d1:window-0000": "<triplet> A <subj> B <obj> r1",
            "d1:window-0001": "<triplet> C <subj> D <obj> r2",
        }
    )
    run_dir = run_rebel(
        tmp_path,
        inputs=[InputRecord("d1", "d1", "abcdefgh")],
        extractor=extractor,
        windowing=WindowingConfig("CHARACTER", 4, 0, "character_sliding_window"),
    )
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert prediction["document_id"] == "d1"
    assert [triple["segment_id"] for triple in prediction["normalized"]] == [
        "window-0000",
        "window-0001",
    ]
    export = export_evaluation(
        run_dir / "predictions.jsonl", tmp_path / "evaluation.jsonl"
    )
    exported = records(export)
    assert [(row["relation"], row["document_id"]) for row in exported] == [
        ("r1", "d1"),
        ("r2", "d1"),
    ]


def test_rebel_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = run_rebel(tmp_path / "first")
    second = run_rebel(tmp_path / "second")
    assert first.name == second.name
    for filename in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
