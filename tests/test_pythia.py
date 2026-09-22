from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from re_te_system.contracts import (
    ExtractionContext,
    InputRecord,
    RawExtractionResult,
)
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.mock import MockPythiaExtractor
from re_te_system.extractors.pythia import (
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
    PythiaConfig,
    PythiaSpaceKBPExtractor,
)
from re_te_system.parsing.turtle import parse_turtle
from re_te_system.runner.run import export_evaluation, run_pipeline


DATASET = {"hash": "space-data", "id": "technical", "version": "1"}
SOURCE = {"documents_sha256": "space-data", "manifest_sha256": None}


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_pythia(
    root: Path,
    *,
    extractor: MockPythiaExtractor | None = None,
) -> Path:
    return run_pipeline(
        inputs=[InputRecord("space-1", "space-1", "A mission carries an instrument.")],
        extractor=extractor or MockPythiaExtractor(),
        output_root=root,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=PythiaConfig().generation_config(),
        model_metadata={
            "base_model": "EleutherAI/pythia-1b-deduped",
            "remote_download_authorized": False,
            "tokenizer": "mock",
        },
        parse_output=parse_turtle,
        run_role="baseline",
        model_family="pythia_spacekbp",
        target_schema_knowledge="none",
        native_schema={
            "id": "space_kbp_space_ontology",
            "inherited_from_model": True,
            "scope": "fixed_domain_ontology",
        },
        input_language="en",
        language_status="native_domain",
        task_class="KBP_DOMAIN_BASELINE",
        controlled_experiment_condition="not_applicable",
        code_commit="test-commit",
    )


def test_pythia_adapter_conforms_to_protocol_without_loading_weights() -> None:
    adapter = PythiaSpaceKBPExtractor.__new__(PythiaSpaceKBPExtractor)
    adapter.model_name = CANONICAL_MODEL_ID
    adapter.model_revision = CANONICAL_MODEL_REVISION
    assert isinstance(adapter, Extractor)


def test_pythia_checkpoint_is_pinned_and_remote_loading_is_disabled() -> None:
    config = PythiaConfig()
    assert config.model_name == CANONICAL_MODEL_ID
    assert config.revision == CANONICAL_MODEL_REVISION
    assert config.local_files_only is True
    assert config.quantization == "none"
    assert config.prompt_profile == "basic"


def test_pythia_mock_generation_is_deterministic() -> None:
    extractor = MockPythiaExtractor()
    context = ExtractionContext("space-1", "space-1", None)
    first = extractor.extract("mission", context)
    second = extractor.extract("mission", context)
    assert first == second


def test_valid_turtle_preserves_uri_terms_and_emission_order() -> None:
    raw = (
        "<https://example.org/s> <https://example.org/p2> <https://example.org/o2> .\n"
        "<https://example.org/s> <https://example.org/p1> <https://example.org/o1> ."
    )
    result = parse_turtle(raw)
    assert [triple.relation for triple in result.triples] == [
        "https://example.org/p2",
        "https://example.org/p1",
    ]
    assert all(triple.subject_is_uri for triple in result.triples)
    assert all(triple.predicate_is_uri for triple in result.triples)
    assert all(triple.object_is_uri for triple in result.triples)


def test_turtle_literals_preserve_datatype_and_language() -> None:
    result = parse_turtle(
        """
        <https://example.org/s> <https://example.org/name> "Misión"@es .
        <https://example.org/s> <https://example.org/year>
            "2026"^^<http://www.w3.org/2001/XMLSchema#integer> .
        """
    )
    assert result.triples[0].object == "Misión"
    assert result.triples[0].object_is_literal is True
    assert result.triples[0].language == "es"
    assert result.triples[1].object == "2026"
    assert result.triples[1].datatype == "http://www.w3.org/2001/XMLSchema#integer"


def test_external_prefix_context_does_not_change_raw_text() -> None:
    raw = "ex:s ex:p ex:o ."
    result = parse_turtle(raw, prefixes={"ex": "https://example.org/"})
    assert len(result.triples) == 1
    assert raw == "ex:s ex:p ex:o ."


def test_invalid_turtle_is_not_repaired_and_records_violation(tmp_path: Path) -> None:
    raw = "ex:subject ex:predicate ex:object"
    parsed = parse_turtle(raw)
    assert parsed.triples == ()
    assert parsed.issues[0].code in {"INVALID_TURTLE", "PREFIX_RESOLUTION_FAILURE"}

    run_dir = run_pythia(
        tmp_path,
        extractor=MockPythiaExtractor(outputs={"space-1": raw}),
    )
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert prediction["raw"]["model_output"] == raw
    assert prediction["parsed"] == []
    assert prediction["normalized"] == []
    assert prediction["validated"]["status"] == "hard_fail"
    assert prediction["validated"]["violations"][0]["code"] in {
        "PARSE_INVALID_TURTLE",
        "PARSE_PREFIX_RESOLUTION_FAILURE",
    }
    statistics = json.loads((run_dir / "statistics.json").read_text())
    assert statistics["parse_failures"] == 1


def test_duplicate_turtle_occurrences_are_preserved_and_reported(tmp_path: Path) -> None:
    triple = "<https://example.org/s> <https://example.org/p> <https://example.org/o> ."
    run_dir = run_pythia(
        tmp_path,
        extractor=MockPythiaExtractor(outputs={"space-1": f"{triple}\n{triple}"}),
    )
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert len(prediction["parsed"]) == 2
    assert len(prediction["validated"]["triples"]) == 2
    assert any(
        item["code"] == "DUPLICATE_TRIPLE"
        for item in prediction["validated"]["violations"]
    )


def test_raw_turtle_is_preserved_exactly_through_pipeline(tmp_path: Path) -> None:
    raw = (
        "<https://example.org/s> <https://example.org/p> "
        '"value with  spaces" .\n'
    )
    run_dir = run_pythia(
        tmp_path,
        extractor=MockPythiaExtractor(outputs={"space-1": raw}),
    )
    prediction = records(run_dir / "predictions.jsonl")[0]
    assert prediction["raw"]["model_output"] == raw
    assert prediction["raw"]["segments"][0]["model_output"] == raw
    assert prediction["parsed"][0]["object"] == "value with  spaces"
    assert prediction["normalized"][0]["object"] == "value with  spaces"


def test_manifest_identifies_domain_baseline_outside_controlled_conditions(
    tmp_path: Path,
) -> None:
    manifest = json.loads((run_pythia(tmp_path) / "manifest.json").read_text())
    assert manifest["run_role"] == "baseline"
    assert manifest["model_family"] == "pythia_spacekbp"
    assert manifest["task_class"] == "KBP_DOMAIN_BASELINE"
    assert manifest["controlled_experiment_condition"] == "not_applicable"
    assert manifest["target_schema_knowledge"] == "none"
    assert manifest["native_schema"]["id"] == "space_kbp_space_ontology"
    assert manifest["condition"] == "C0"


def test_pythia_artifacts_contain_no_hohfeld_knowledge_or_legacy_aliases(
    tmp_path: Path,
) -> None:
    run_dir = run_pythia(tmp_path)
    serialized = (run_dir / "manifest.json").read_text() + (
        run_dir / "predictions.jsonl"
    ).read_text()
    assert all(
        forbidden not in serialized
        for forbidden in ("Duty", "Right", "Privilege", "NoRight", "relation_aliases")
    )


def test_pythia_sources_have_no_legacy_prompt_alias_or_namespace() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "re_te_system"
    source = (
        (source_root / "extractors" / "pythia.py").read_text()
        + (source_root / "parsing" / "turtle.py").read_text()
    )
    assert all(
        forbidden not in source
        for forbidden in ("Aqua", "relation_aliases", "spaceontology.org")
    )


def test_generation_limit_and_model_failure_are_descriptive_violations(
    tmp_path: Path,
) -> None:
    class LimitedExtractor(MockPythiaExtractor):
        def extract(
            self, text: str, context: ExtractionContext
        ) -> RawExtractionResult:
            result = super().extract(text, context)
            return RawExtractionResult(
                result.model_output,
                {**result.generation_metadata, "output_reached_limit": True},
            )

    limited = records(
        run_pythia(tmp_path / "limited", extractor=LimitedExtractor())
        / "predictions.jsonl"
    )[0]
    assert any(
        item["code"] == "PARSE_TRUNCATED_GENERATION"
        for item in limited["validated"]["violations"]
    )

    failed = records(
        run_pythia(
            tmp_path / "failed",
            extractor=MockPythiaExtractor(fail_on={"space-1"}),
        )
        / "predictions.jsonl"
    )[0]
    assert any(
        item["code"] == "PARSE_MODEL_FAILURE"
        for item in failed["validated"]["violations"]
    )


def test_real_adapter_reports_input_truncation_without_loading_weights() -> None:
    import torch

    class FakeTokenizer:
        eos_token_id = 0

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
            return "RAW Turtle "

    class FakeModel:
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=torch.tensor([[0, 1, 2, 3, 8, 9]]))

    adapter = PythiaSpaceKBPExtractor.__new__(PythiaSpaceKBPExtractor)
    adapter.config = PythiaConfig(max_input_tokens=4, max_new_tokens=2)
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeModel()
    adapter.device = torch.device("cpu")
    adapter.model_max_length = 6
    adapter.effective_input_limit = 4
    result = adapter.extract("text", ExtractionContext("x", "x", None))
    assert result.model_output == "RAW Turtle "
    assert result.generation_metadata["truncated_input"] is True
    assert result.generation_metadata["untruncated_input_token_count"] == 10
    assert result.generation_metadata["effective_input_limit"] == 4


def test_pythia_evaluation_export_projects_rdf_terms(tmp_path: Path) -> None:
    run_dir = run_pythia(tmp_path)
    target = export_evaluation(
        run_dir / "predictions.jsonl",
        tmp_path / "evaluation.jsonl",
    )
    row = records(target)[0]
    assert row["subject"] == "https://example.org/mission"
    assert row["relation"] == "https://example.org/hasDescription"
    assert row["object"] == "A mission carries an instrument."


def test_pythia_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = run_pythia(tmp_path / "first")
    second = run_pythia(tmp_path / "second")
    assert first.name == second.name
    for filename in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
