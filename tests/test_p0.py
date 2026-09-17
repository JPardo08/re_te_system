from __future__ import annotations

import json
from pathlib import Path

from re_te_system.contracts import ExtractionContext, InputRecord, ParsedTriple
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.mock import MockExtractor
from re_te_system.extractors.mrebel import LEGACY_REL_ALLOWED, MRebelConfig
from re_te_system.manifests.run_manifest import build_manifest
from re_te_system.normalization.basic import align_triples, normalize_triples
from re_te_system.parsing.mrebel import parse_mrebel
from re_te_system.runner.run import (
    WindowingConfig,
    export_evaluation,
    load_hohfeld_documents,
    run_pipeline,
    segment_document,
)
from re_te_system.validation.structural import validate_structural


DATASET = {"hash": "abc", "id": "test", "version": "1"}
SOURCE = {"documents_sha256": "abc", "manifest_sha256": "def"}
GENERATION = {
    "do_sample": False,
    "max_new_tokens": None,
    "num_beams": 1,
    "seed": None,
}


def run_mock(tmp_path: Path, extractor: MockExtractor | None = None) -> Path:
    return run_pipeline(
        inputs=[InputRecord("d1", "d1", "Empresa contrata trabajador.")],
        extractor=extractor or MockExtractor(),
        output_root=tmp_path,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_extractor_protocol_and_context() -> None:
    extractor = MockExtractor()
    assert isinstance(extractor, Extractor)
    result = extractor.extract("a b", ExtractionContext("x", "d", None))
    assert result.model_output.startswith("<triplet>")


def test_c0_loader_is_gold_blind(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    source.write_text(
        json.dumps(
            {
                "document_id": "article-1",
                "source_text": "Only this text",
                "annotations": [
                    {
                        "relation": "Duty",
                        "subject": "secret",
                        "signature": "LegalAgent-LegalAgent",
                        "eligibility": {"document_level_te": True},
                    }
                ],
            }
        )
        + "\n"
    )
    payload = load_hohfeld_documents(source)[0].extractor_payload()
    assert payload == {
        "document_id": "article-1",
        "example_id": "article-1",
        "text": "Only this text",
    }
    forbidden = {"relation", "subject", "object", "signature", "types", "eligibility"}
    assert forbidden.isdisjoint(payload)


def test_parser_multiple_triples_and_surface_preservation() -> None:
    raw = (
        "<s> tp_XX <triplet> La Empresa <ORG> el trabajador <PER> employer "
        "<triplet> trabajador <PER> salario mínimo <CONCEPT> part of </s>"
    )
    parsed = parse_mrebel(raw)
    assert [(t.subject, t.relation, t.object) for t in parsed.triples] == [
        ("La Empresa", "employer", "el trabajador"),
        ("trabajador", "part of", "salario mínimo"),
    ]


def test_parser_preserves_duplicate_occurrences() -> None:
    raw = "<triplet> A <x> B <y> r <triplet> A <x> B <y> r"
    assert len(parse_mrebel(raw).triples) == 2


def test_parser_angle_and_pipe_fallbacks() -> None:
    angle = parse_mrebel("<triplet> <subj> Acme Corp <obj> New York <rel> headquarters")
    pipe = parse_mrebel("<triplet> aspirin | has part | salicylic acid")
    assert (angle.triples[0].subject, angle.triples[0].relation, angle.triples[0].object) == (
        "Acme Corp",
        "headquarters",
        "New York",
    )
    assert (pipe.triples[0].subject, pipe.triples[0].relation, pipe.triples[0].object) == (
        "aspirin",
        "has part",
        "salicylic acid",
    )


def test_parser_malformed_is_deterministic() -> None:
    first = parse_mrebel("<triplet> truncated garbage")
    second = parse_mrebel("<triplet> truncated garbage")
    assert first == second
    assert first.triples == ()
    assert first.issues[0].code == "UNPARSEABLE_CHUNK"


def test_normalization_is_representational_and_duplicate_preserving() -> None:
    triples = (
        ParsedTriple("  Cafe\u0301  legal", "part   of", " X "),
        ParsedTriple("  Cafe\u0301  legal", "part   of", " X "),
    )
    normalized = normalize_triples(triples)
    assert normalized[0].subject == "Café legal"
    assert normalized[0].relation == "part of"
    assert len(normalized) == 2


def test_posthoc_alignment_is_document_based_and_whitespace_tolerant() -> None:
    aligned = align_triples(
        (ParsedTriple("empresa pública", "r", "trabajador"),),
        "La empresa\n  pública informa al trabajador.",
        segment_start=100,
    )
    assert aligned[0].subject_span == (103, 120)
    assert aligned[0].object_span == (132, 142)
    assert aligned[0].span_source == "posthoc_string_alignment"


def test_structural_validation_flags_empty_and_duplicates_without_deletion() -> None:
    duplicate = ParsedTriple("s", "r", "o")
    result = validate_structural((duplicate, duplicate, ParsedTriple("s", "r", "")))
    assert len(result.triples) == 3
    assert {item.code for item in result.violations} == {
        "DUPLICATE_TRIPLE",
        "EMPTY_OBJECT",
    }
    assert result.status == "hard_fail"


def test_mock_end_to_end_preserves_raw_and_contract(tmp_path: Path) -> None:
    exact = "<s> tp_XX <triplet> Empresa <ORG> trabajador <PER> employer </s>"
    run_dir = run_mock(tmp_path, MockExtractor(outputs={"d1": exact}))
    record = read_jsonl(run_dir / "predictions.jsonl")[0]
    assert record["raw"]["model_output"] == exact
    assert record["raw"]["segments"][0]["model_output"] == exact
    assert record["parsed"][0]["relation"] == "employer"
    assert record["validated"]["status"] == "ok"
    assert json.loads((run_dir / "manifest.json").read_text())["condition"] == "C0"


def test_windowing_and_document_aggregation(tmp_path: Path) -> None:
    extractor = MockExtractor(
        outputs={
            "d1:window-0000": "<triplet> A <x> B <y> r",
            "d1:window-0001": "<triplet> C <x> D <y> r",
        }
    )
    run_dir = run_pipeline(
        inputs=[InputRecord("d1", "d1", "abcdefgh")],
        extractor=extractor,
        output_root=tmp_path,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        windowing=WindowingConfig("CHARACTER", 4, 0, "character_sliding_window"),
        code_commit="test-commit",
    )
    record = read_jsonl(run_dir / "predictions.jsonl")[0]
    assert record["document_id"] == "d1"
    assert record["segment_id"] is None
    assert [item["segment_id"] for item in record["normalized"]] == [
        "window-0000",
        "window-0001",
    ]
    assert [(s["segment_start"], s["segment_end"]) for s in record["raw"]["segments"]] == [
        (0, 4),
        (4, 8),
    ]


def test_token_windowing_requires_explicit_counter() -> None:
    config = WindowingConfig("TOKEN", 3, 0, "tokenizer_aware")
    segments = segment_document("abcdef", config, lambda text: len(text))
    assert [segment.text for segment in segments] == ["abc", "def"]


def test_failure_isolation(tmp_path: Path) -> None:
    run_dir = run_pipeline(
        inputs=[InputRecord("bad", "bad", "x"), InputRecord("good", "good", "a b")],
        extractor=MockExtractor(fail_on={"bad"}),
        output_root=tmp_path,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )
    statistics = json.loads((run_dir / "statistics.json").read_text())
    assert statistics["inputs_total"] == 2
    assert statistics["inputs_failed"] == 1
    assert statistics["inputs_succeeded"] == 1
    assert read_jsonl(run_dir / "failures.jsonl")[0]["example_id"] == "bad"
    assert len(read_jsonl(run_dir / "predictions.jsonl")) == 2


def test_manifest_complete_and_run_id_deterministic() -> None:
    kwargs = dict(
        dataset=DATASET,
        model={"name": "mock", "resolved_revision": "v1"},
        condition="C0",
        generation=GENERATION,
        windowing={"strategy": "NONE"},
        benchmark_source=SOURCE,
        code_commit="abc123",
    )
    first = build_manifest(**kwargs)
    second = build_manifest(**kwargs)
    assert first == second
    assert first["run_id"].startswith("run-")
    assert first["conditioning"] == "none"
    assert first["prediction_contract_version"] == "1.0"
    assert first["operational_metadata"]["created_at"] is None


def test_artifacts_and_order_are_deterministic(tmp_path: Path) -> None:
    first = run_mock(tmp_path / "one")
    second = run_mock(tmp_path / "two")
    assert first.name == second.name
    for filename in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_evaluation_export_contract(tmp_path: Path) -> None:
    run_dir = run_mock(tmp_path)
    target = export_evaluation(run_dir / "predictions.jsonl", tmp_path / "evaluation.jsonl")
    row = read_jsonl(target)[0]
    assert set(row) == {
        "document_id",
        "id",
        "object",
        "prediction_id",
        "relation",
        "segment_id",
        "subject",
    }
    assert row["document_id"] == "d1"


def test_mrebel_spanish_and_generation_configuration() -> None:
    config = MRebelConfig()
    assert config.src_lang == "es_XX"
    assert config.target_token == "tp_XX"
    assert config.generation_config() == {
        "do_sample": False,
        "early_stopping": True,
        "max_new_tokens": 512,
        "num_beams": 3,
        "num_return_sequences": 1,
        "seed": 42,
        "temperature": None,
    }


def test_legacy_relation_filter_is_not_in_default_config() -> None:
    assert "part_of" in LEGACY_REL_ALLOWED
    assert "legacy_relation_filter" not in MRebelConfig().manifest_config()


def test_sources_contain_no_absolute_user_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = [root / "README.md", *sorted((root / "src").rglob("*.py")), *sorted((root / "docs").glob("*.md"))]
    assert all("/Users/" not in path.read_text(encoding="utf-8") for path in checked)
