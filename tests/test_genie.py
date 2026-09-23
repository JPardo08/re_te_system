from __future__ import annotations

from functools import partial
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from re_te_system.cli import main as cli_main
from re_te_system.conditioning.genie_constraints import (
    NON_NATIVE_ABLATION,
    NON_NATIVE_RELATION_ONLY,
    OFFICIAL_TRIE_DESERIALIZATION,
    PROFILE_CUSTOM_FULL,
    PROFILE_LARGE,
    PROFILE_SMALL,
    PROFILE_UNCONSTRAINED,
    STATE_OBJECT,
    STATE_OUTSIDE,
    STATE_RELATION,
    STATE_SUBJECT,
    StructuralCodes,
    TokenPrefixTrie,
    allowed_next_token_ids,
    closed_schema_spec,
    named_schema_spec,
    official_string_inventories_available,
    official_string_inventory_paths,
    structural_state,
    unconstrained_spec,
)
from re_te_system.contracts import ExtractionContext, InputRecord, ParsedGenieOccurrence
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.genie import (
    CANONICAL_CHECKPOINT_MD5,
    CANONICAL_CHECKPOINT_NAME,
    CANONICAL_CHECKPOINT_SIZE_BYTES,
    CANONICAL_TOKENIZER_ID,
    CANONICAL_TOKENIZER_REVISION,
    CHECKPOINT_LOADING_FORM,
    CHECKPOINT_LOADING_STATUS,
    CONTROL_TAG_BPE,
    EVALUATOR_BEAM_POLICY,
    GENIE_DOCUMENTED_LANGUAGE,
    GENIE_SCIENTIFIC_ROLE,
    GENIE_TASK_CLASS,
    MODERNIZED_LOADING_STATUS,
    OFFICIAL_NOTEBOOK_INPUT,
    OFFICIAL_NOTEBOOK_LARGE_RAW,
    OFFICIAL_NOTEBOOK_SMALL_RAW,
    OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW,
    OUTPUT_SOURCE_OFFICIAL_NOTEBOOK,
    REAL_GENIE_NATIVE_SMOKE,
    GenIEConfig,
    GenIEExtractor,
)
from re_te_system.extractors.mock import MockGenieExtractor
from re_te_system.parsing.genie import map_genie_ids, parse_genie, project_genie_occurrences
from re_te_system.runner.run import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
LEGACY_DATA = WORKSPACE_ROOT / "_legacy" / "GenIE" / "data"
DATASET = {"hash": "genie-data", "id": "technical", "version": "1"}
SOURCE = {"documents_sha256": "genie-data", "manifest_sha256": None}
DUPLICATE_RAW = " <sub> A <rel> r <obj> B <et> <sub> A <rel> r <obj> B <et>"
TINY_ENTITIES = ("A", "B")
TINY_RELATIONS = ("r",)


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.ids = {"A": 20, "B": 21, "r": 30}

    def encode(self, text: str) -> list[int]:
        parts = text.strip().split()
        return [self.bos_token_id] + [self.ids.get(part, 99) for part in parts] + [self.eos_token_id]

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        ids = self.encode(text)
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": SimpleNamespace(shape=(1, len(ids))),
                "attention_mask": None,
            }
        return {"input_ids": ids}

    def decode(self, sequence: object, skip_special_tokens: bool = True) -> str:
        if isinstance(sequence, str):
            return sequence
        return " <sub> A <rel> r <obj> B <et>"


class FakeGenieModel:
    def generate(self, **kwargs: object) -> dict[str, object]:
        return {
            "sequences": [
                " <sub> A <rel> r <obj> B <et>",
                " <sub> B <rel> r <obj> A <et>",
            ],
            "sequences_scores": [-0.1, -0.4],
        }


CODES = StructuralCodes(
    bos_token_id=0,
    eos_token_id=2,
    start_of_tag=10,
    end_of_tag=11,
    subject_token=12,
    relation_token=13,
    object_token=14,
    end_of_triple_token=15,
)


def smoke_main(argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(
        "run_genie_native_smoke",
        REPO_ROOT / "scripts" / "run_genie_native_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_genie(
    root: Path,
    *,
    raw: str = OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW,
    text: str = OFFICIAL_NOTEBOOK_INPUT,
    profile: str = PROFILE_UNCONSTRAINED,
    fail_on: set[str] | None = None,
    input_language: str = "en",
    language_status: str = "supported",
) -> Path:
    if profile == PROFILE_UNCONSTRAINED:
        spec = unconstrained_spec()
    elif profile in {PROFILE_SMALL, PROFILE_LARGE}:
        spec = named_schema_spec(profile)
    else:
        spec = closed_schema_spec(TINY_ENTITIES, TINY_RELATIONS, profile=profile)
    extractor = MockGenieExtractor(
        outputs={"genie-1": raw},
        fail_on=fail_on or set(),
        constraint_spec=spec,
    )
    return run_pipeline(
        inputs=[InputRecord("genie-1", "genie-1", text)],
        extractor=extractor,
        output_root=root,
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GenIEConfig(constraint_profile=spec.profile).generation_config(),
        model_metadata={
            "constraint": spec.manifest_metadata(),
            "documented_language": GENIE_DOCUMENTED_LANGUAGE,
            "evaluator_beam_policy": EVALUATOR_BEAM_POLICY,
            "formal_ontology": False,
            "parser_policy": "conservative_genie_v1",
            "scientific_role": GENIE_SCIENTIFIC_ROLE,
            "tokenizer": CANONICAL_TOKENIZER_ID,
        },
        parse_output=parse_genie,
        run_role="baseline",
        model_family="genie",
        target_schema_knowledge="fixed_native_schema+kb_constraints",
        native_schema={
            "constraint_profile": spec.profile,
            "entity_inventory_required": spec.entity_inventory_required,
            "formal_ontology": False,
            "id": "wikidata_closed_schema",
            "inherited_from_model": True,
            "relation_inventory_required": spec.relation_inventory_required,
            "scope": "fixed_native_schema+kb_constraints",
        },
        input_language=input_language,
        language_status=language_status,
        task_class=GENIE_TASK_CLASS,
        controlled_experiment_condition="not_applicable",
        code_commit="test-commit",
    )


def test_genie_adapter_protocol_and_checkpoint_provenance_without_weights() -> None:
    adapter = GenIEExtractor.__new__(GenIEExtractor)
    adapter.model_name = CANONICAL_CHECKPOINT_NAME
    adapter.model_revision = CANONICAL_CHECKPOINT_MD5
    assert isinstance(adapter, Extractor)
    config = GenIEConfig()
    assert config.checkpoint_name == CANONICAL_CHECKPOINT_NAME
    assert config.checkpoint_md5 == CANONICAL_CHECKPOINT_MD5
    assert config.checkpoint_size_bytes == CANONICAL_CHECKPOINT_SIZE_BYTES
    assert config.tokenizer_name == CANONICAL_TOKENIZER_ID
    assert config.tokenizer_revision == CANONICAL_TOKENIZER_REVISION
    assert config.local_files_only is True
    assert CHECKPOINT_LOADING_FORM == "LIGHTNING_CHECKPOINT_NATIVE"
    assert CHECKPOINT_LOADING_STATUS == "not_real_smoked"
    assert MODERNIZED_LOADING_STATUS == "unverified"
    assert REAL_GENIE_NATIVE_SMOKE == "BLOCKED_RESOURCE"
    assert GENIE_SCIENTIFIC_ROLE == "CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE"
    assert CONTROL_TAG_BPE[" <sub>"] == ["<s>", "Ġ<", "sub", ">", "</s>"]


def test_real_extractor_does_not_load_or_download_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="does not load genie_r.ckpt"):
        GenIEExtractor()


def test_injected_extractor_preserves_beams_and_selects_top_score() -> None:
    extractor = GenIEExtractor(
        model=FakeGenieModel(),
        tokenizer=FakeTokenizer(),
    )
    result = extractor.extract(
        "A met B.",
        ExtractionContext("ex", "doc", None),
    )
    assert result.model_output == " <sub> A <rel> r <obj> B <et>"
    beams = list(result.generation_metadata["beams"])
    assert len(beams) == 2
    assert beams[0]["beam_rank"] == 0
    assert beams[0]["raw"] == result.model_output
    assert beams[1]["raw"] == " <sub> B <rel> r <obj> A <et>"
    assert result.generation_metadata["selected_beam_policy"] == EVALUATOR_BEAM_POLICY
    assert result.generation_metadata["selected_beam_rank"] == 0
    assert result.generation_metadata["checkpoint"]["loading_status"] == CHECKPOINT_LOADING_STATUS


def test_official_notebook_raw_is_labeled_and_parsed() -> None:
    unconstrained = parse_genie(OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW)
    small = parse_genie(OFFICIAL_NOTEBOOK_SMALL_RAW)
    large = parse_genie(OFFICIAL_NOTEBOOK_LARGE_RAW)
    assert OFFICIAL_NOTEBOOK_LARGE_RAW == OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW
    assert OUTPUT_SOURCE_OFFICIAL_NOTEBOOK == "OFFICIAL_NOTEBOOK_RECORDED_OUTPUT"
    assert unconstrained.triples[0].subject == "KSAZ-TV"
    assert unconstrained.triples[0].relation == "headquarters location"
    assert unconstrained.triples[0].object == "Phoenix, Arizona"
    assert [item.subject_text for item in small.genie_occurrences] == [
        "Phoenix, Arizona",
        "Arizona",
    ]
    assert large.triples == unconstrained.triples


def test_parser_preserves_order_duplicates_and_exact_raw_fields() -> None:
    parsed = parse_genie(DUPLICATE_RAW)
    assert len(parsed.genie_occurrences) == 2
    assert len(parsed.triples) == 2
    assert parsed.triples[0] == parsed.triples[1]
    assert parsed.triples[0].subject == "A"
    assert parsed.triples[0].relation == "r"
    assert parsed.triples[0].object == "B"
    assert any(issue.code == "DUPLICATE_OCCURRENCE" for issue in parsed.issues)
    projected = project_genie_occurrences(parsed.genie_occurrences)
    assert projected == parsed.triples


def test_malformed_and_incomplete_outputs_are_reported() -> None:
    missing_et = parse_genie(" <sub> A <rel> r <obj> B")
    assert any(issue.code == "INCOMPLETE_TRIPLET" for issue in missing_et.issues)
    assert missing_et.genie_occurrences[0].complete is False
    assert missing_et.triples == ()

    missing_obj = parse_genie(" <sub> A <rel> r <et>")
    assert {issue.code for issue in missing_obj.issues} >= {
        "UNEXPECTED_CONTROL_TOKEN",
        "INCOMPLETE_TRIPLET",
    }

    unexpected = parse_genie(" <rel> r <sub> A <obj> B <et>")
    assert any(issue.code == "UNEXPECTED_CONTROL_TOKEN" for issue in unexpected.issues)

    trailing = parse_genie(" <sub> A <rel> r <obj> B <et> <sub> C")
    assert len(trailing.triples) == 1
    assert trailing.genie_occurrences[-1].complete is False
    assert trailing.genie_occurrences[-1].subject_text == "C"

    plain = parse_genie("hello world")
    assert any(issue.code == "MALFORMED_GENIE" for issue in plain.issues)
    empty = parse_genie("   ")
    assert any(issue.code == "EMPTY_STRUCTURE" for issue in empty.issues)


def test_inventory_status_is_optional_and_ids_are_separated() -> None:
    parsed = parse_genie(
        DUPLICATE_RAW,
        entity_inventory=TINY_ENTITIES,
        relation_inventory=TINY_RELATIONS,
    )
    first = parsed.genie_occurrences[0]
    assert first.subject_inventory_status == "in_inventory"
    assert first.relation_inventory_status == "in_inventory"
    assert first.subject_id is None
    mapped = map_genie_ids(
        parsed.genie_occurrences,
        {"A": ("Q1", "Q9"), "B": ("Q2",)},
        {"r": ("P1",)},
    )
    assert mapped[0].subject_text == "A"
    assert mapped[0].subject_id is None
    assert mapped[0].subject_id_status == "ambiguous"
    assert mapped[0].relation_id == "P1"
    assert mapped[0].relation_id_status == "resolved"
    assert mapped[0].object_id == "Q2"
    unresolved = map_genie_ids(
        (ParsedGenieOccurrence("X", "y", "Z", 0),),
        {},
        {},
    )
    assert unresolved[0].subject_id_status == "unresolved"
    assert unresolved[0].subject_text == "X"


def test_tiny_trie_and_state_machine() -> None:
    tokenizer = FakeTokenizer()
    entity_trie = TokenPrefixTrie.from_strings(TINY_ENTITIES, tokenizer)
    relation_trie = TokenPrefixTrie.from_strings(TINY_RELATIONS, tokenizer)
    assert entity_trie.get([]) == [20, 21]
    assert relation_trie.get([30]) == [2]
    outside = [0]
    assert allowed_next_token_ids(outside, CODES) == [0]
    after_bos = [0, 0]
    assert structural_state(after_bos, CODES) == STATE_OUTSIDE
    assert set(allowed_next_token_ids(after_bos, CODES)) == {10, 2}
    after_sub = [0, 0, 10, 12, 11]
    assert structural_state(after_sub, CODES) == STATE_SUBJECT
    assert allowed_next_token_ids(
        after_sub,
        CODES,
        entity_trie=entity_trie,
        relation_trie=relation_trie,
    ) == [20, 21]
    after_rel_tag = [0, 0, 10, 12, 11, 20, 2, 10, 13, 11]
    # after complete sub+entity+rel tags the state is relation?
    # complete tags: sub, rel -> STATE_RELATION? Wait
    # tags: (10,12,11)=sub, (10,13,11)=rel -> 2 tags -> STATE_RELATION
    assert structural_state(after_rel_tag, CODES) == STATE_RELATION
    allowed = allowed_next_token_ids(
        after_rel_tag,
        CODES,
        entity_trie=entity_trie,
        relation_trie=relation_trie,
    )
    assert 30 in allowed
    with pytest.raises(ValueError, match="both entity and relation"):
        allowed_next_token_ids(after_sub, CODES, relation_trie=relation_trie)


def test_no_canonical_relation_only_profile() -> None:
    with pytest.raises(ValueError, match=NON_NATIVE_ABLATION):
        closed_schema_spec(TINY_ENTITIES, TINY_RELATIONS, profile=NON_NATIVE_RELATION_ONLY)
    with pytest.raises(ValueError, match="both an entity"):
        closed_schema_spec((), TINY_RELATIONS)
    small = named_schema_spec(PROFILE_SMALL)
    large = named_schema_spec(PROFILE_LARGE)
    assert small.entity_inventory_required is True
    assert large.relation_inventory_required is True
    assert small.ready_for_constrained_decode is False
    unconstrained = unconstrained_spec()
    assert unconstrained.entity_inventory_required is False


def test_official_string_inventories_are_preferred_over_pickle() -> None:
    assert OFFICIAL_TRIE_DESERIALIZATION == "RECONSTRUCT_FROM_STRINGS_PREFERRED"
    if not LEGACY_DATA.exists():
        pytest.skip("legacy GenIE data cache is not present")
    assert official_string_inventories_available(LEGACY_DATA) is True
    paths = official_string_inventory_paths(LEGACY_DATA)
    assert all(path.is_file() for path in paths.values())
    assert not str(paths["small_entity"]).endswith(".pickle")


def test_manifest_and_prediction_contract(tmp_path: Path) -> None:
    run_dir = run_genie(tmp_path, raw=DUPLICATE_RAW, text="A related B.")
    prediction = records(run_dir / "predictions.jsonl")[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert prediction["raw"]["model_output"] == DUPLICATE_RAW
    assert len(prediction["parsed"]) == 2
    assert len(prediction["parsed_genie_occurrences"]) == 2
    assert prediction["parsed_genie_occurrences"][0]["occurrence_index"] == 0
    beams = prediction["raw"]["segments"][0]["generation_metadata"]["beams"]
    assert beams[0]["raw"] == DUPLICATE_RAW
    assert manifest["run_role"] == "baseline"
    assert manifest["model_family"] == "genie"
    assert manifest["task_class"] == "KBP_CLOSED_IE"
    assert manifest["target_schema_knowledge"] == "fixed_native_schema+kb_constraints"
    assert manifest["controlled_experiment_condition"] == "not_applicable"
    assert manifest["native_schema"]["formal_ontology"] is False
    assert manifest["model"]["scientific_role"] == GENIE_SCIENTIFIC_ROLE


def test_model_failure_and_language_scope(tmp_path: Path) -> None:
    failed = run_genie(tmp_path / "fail", fail_on={"genie-1"})
    prediction = records(failed / "predictions.jsonl")[0]
    assert any(
        item["code"] == "PARSE_MODEL_FAILURE"
        for item in prediction["validated"]["violations"]
    )
    english = json.loads((run_genie(tmp_path / "en") / "manifest.json").read_text())
    spanish = json.loads(
        (
            run_genie(
                tmp_path / "es",
                input_language="es",
                language_status="out_of_documented_training_scope",
            )
            / "manifest.json"
        ).read_text()
    )
    assert english["language_status"] == "supported"
    assert spanish["input_language"] == "es"
    assert spanish["language_status"] == "out_of_documented_training_scope"


def test_cli_marks_english_supported_and_spanish_out_of_scope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    documents = tmp_path / "docs.jsonl"
    documents.write_text(
        json.dumps(
            {"document_id": "genie-cli", "source_text": OFFICIAL_NOTEBOOK_INPUT},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    cli_main(
        [
            "run",
            "--documents",
            str(documents),
            "--output-root",
            str(tmp_path / "en"),
            "--extractor",
            "genie-mock",
            "--input-language",
            "en",
        ]
    )
    cli_main(
        [
            "run",
            "--documents",
            str(documents),
            "--output-root",
            str(tmp_path / "es"),
            "--extractor",
            "genie-mock",
            "--input-language",
            "es",
        ]
    )
    english = json.loads(next((tmp_path / "en").glob("*/manifest.json")).read_text())
    spanish = json.loads(next((tmp_path / "es").glob("*/manifest.json")).read_text())
    assert english["language_status"] == "supported"
    assert spanish["language_status"] == "out_of_documented_training_scope"
    assert english["task_class"] == "KBP_CLOSED_IE"


def test_no_gold_or_hohfeld_leakage_in_genie_modules() -> None:
    forbidden = ("Duty", "Privilege", "NoRight", "gold_entities")
    paths = (
        REPO_ROOT / "src/re_te_system/conditioning/genie_constraints.py",
        REPO_ROOT / "src/re_te_system/extractors/genie.py",
        REPO_ROOT / "src/re_te_system/parsing/genie.py",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "Hohfeld" not in text
        for token in forbidden:
            assert token not in text


def test_smoke_script_does_not_download(capsys: pytest.CaptureFixture[str]) -> None:
    code = smoke_main(
        [
            "--documents",
            "docs.jsonl",
            "--output-root",
            "runs",
            "--constraint-profile",
            "small",
        ]
    )
    captured = capsys.readouterr().out
    assert code == 0
    assert "BLOCKED_RESOURCE" in captured
    assert "--extractor genie-mock" in captured
    assert "does not download" in captured.lower() or "never downloads" in captured.lower() or "P0 does not download" in captured


def test_docs_state_closed_world_and_role() -> None:
    baseline = (REPO_ROOT / "docs/GENIE_BASELINE.md").read_text(encoding="utf-8")
    assert "KBP_CLOSED_IE" in baseline
    assert "CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE" in baseline
    assert "entity inventory" in baseline
    assert "relation inventory" in baseline
    assert "NON_NATIVE_ABLATION" in baseline
    assert "out_of_documented_training_scope" in baseline
    assert "OFFICIAL_NOTEBOOK_RECORDED_OUTPUT" in baseline
    assert "not_real_smoked" in baseline
