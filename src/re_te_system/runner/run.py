"""Gold-blind loading, windowing, pipeline execution, and file exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

from re_te_system.contracts import (
    ExtractionContext,
    InputRecord,
    ParseResult,
    PREDICTION_CONTRACT_VERSION,
    Segment,
    triple_to_dict,
)
from re_te_system.extractors.base import Extractor
from re_te_system.extractors.mrebel import LEGACY_REL_ALLOWED
from re_te_system.manifests.run_manifest import build_manifest, canonical_json, sha256_file
from re_te_system.normalization.basic import align_triples, normalize_triples
from re_te_system.parsing.mrebel import parse_mrebel
from re_te_system.validation.structural import validate_structural


Parser = Callable[[str, str | None], ParseResult]


@dataclass(frozen=True)
class WindowingConfig:
    strategy: str = "NONE"
    max_units: int | None = None
    overlap: int = 0
    segmentation_mechanism: str = "whole_document"

    def __post_init__(self) -> None:
        if self.strategy not in {"NONE", "CHARACTER", "TOKEN"}:
            raise ValueError("windowing strategy must be NONE, CHARACTER, or TOKEN")
        if self.strategy != "NONE" and (self.max_units is None or self.max_units <= 0):
            raise ValueError("windowed strategies require positive max_units")
        if self.overlap < 0 or (
            self.max_units is not None and self.overlap >= self.max_units
        ):
            raise ValueError("overlap must be non-negative and smaller than max_units")


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def load_hohfeld_documents(path: str | Path) -> list[InputRecord]:
    """Read only identifiers and source text; annotation fields never escape."""
    records: list[InputRecord] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            document_id = value.get("document_id")
            text = value.get("source_text")
            if not isinstance(document_id, str) or not isinstance(text, str):
                raise ValueError(f"invalid document contract at line {line_number}")
            records.append(InputRecord(document_id, document_id, text))
    return sorted(records, key=lambda item: _natural_key(item.document_id))


def _character_segments(text: str, maximum: int, overlap: int) -> list[Segment]:
    segments: list[Segment] = []
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        segments.append(Segment(f"window-{len(segments):04d}", text[start:end], start, end))
        if end == len(text):
            break
        start = end - overlap
    return segments


def _token_segments(
    text: str,
    maximum: int,
    overlap: int,
    count_tokens: Callable[[str], int],
) -> list[Segment]:
    segments: list[Segment] = []
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        best = start
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(text[start:middle]) <= maximum:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best == start:
            raise ValueError("tokenizer cannot fit any text in configured window")
        end = best
        segments.append(Segment(f"window-{len(segments):04d}", text[start:end], start, end))
        if end == len(text):
            break
        if overlap:
            rewind = end
            while rewind > start and count_tokens(text[rewind:end]) < overlap:
                rewind -= 1
            start = max(start + 1, rewind)
        else:
            start = end
    return segments


def segment_document(
    text: str,
    config: WindowingConfig,
    count_tokens: Callable[[str], int] | None = None,
) -> list[Segment]:
    if config.strategy == "NONE":
        return [Segment(None, text, 0, len(text))]
    if config.strategy == "CHARACTER":
        return _character_segments(text, config.max_units or 1, config.overlap)
    if count_tokens is None:
        raise ValueError("TOKEN windowing requires extractor token counting")
    return _token_segments(text, config.max_units or 1, config.overlap, count_tokens)


def _legacy_relation_selected(relation: str) -> bool:
    key = re.sub(r"\s+", "_", relation.strip().lower().replace("-", " "))
    return key in LEGACY_REL_ALLOWED


def _jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(canonical_json(record) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def run_pipeline(
    *,
    inputs: list[InputRecord],
    extractor: Extractor,
    output_root: str | Path,
    dataset: Mapping[str, Any],
    benchmark_source: Mapping[str, Any],
    generation: Mapping[str, Any],
    windowing: WindowingConfig = WindowingConfig(),
    model_metadata: Mapping[str, Any] | None = None,
    condition: str = "C0",
    legacy_relation_filter: bool = False,
    parse_output: Parser = parse_mrebel,
    run_role: str = "baseline",
    model_family: str = "mrebel",
    target_schema_knowledge: str = "none",
    native_schema: Mapping[str, Any] | None = None,
    input_language: str = "es",
    language_status: str = "supported",
    code_commit: str | None = None,
) -> Path:
    model = {
        "name": extractor.model_name,
        "requested_revision": getattr(getattr(extractor, "config", None), "revision", None),
        "resolved_revision": extractor.model_revision,
        **dict(model_metadata or {}),
    }
    window_manifest = asdict(windowing)
    window_manifest["relation_filter"] = (
        "LEGACY_REL_ALLOWED_DERIVED_VIEW" if legacy_relation_filter else "none"
    )
    manifest = build_manifest(
        dataset=dataset,
        model=model,
        condition=condition,
        generation=generation,
        windowing=window_manifest,
        benchmark_source=benchmark_source,
        run_role=run_role,
        model_family=model_family,
        target_schema_knowledge=target_schema_knowledge,
        native_schema=native_schema
        or {"id": "wikidata_like", "inherited_from_model": True},
        input_language=input_language,
        language_status=language_status,
        code_commit=code_commit,
    )
    run_dir = Path(output_root) / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stats = {
        "alignment_failures": 0,
        "duplicates": 0,
        "inputs_failed": 0,
        "inputs_succeeded": 0,
        "inputs_total": len(inputs),
        "parse_failures": 0,
        "parsed_triples": 0,
        "raw_outputs": 0,
        "valid_triples": 0,
    }
    count_tokens = getattr(extractor, "count_tokens", None)
    for item in sorted(inputs, key=lambda value: _natural_key(value.document_id)):
        raw_segments: list[dict[str, Any]] = []
        parsed_all = []
        parse_issues = []
        example_failed = False
        try:
            segments = segment_document(item.text, windowing, count_tokens)
        except Exception as exc:
            failures.append(
                {
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                    "example_id": item.example_id,
                    "recoverable": False,
                    "stage": "windowing",
                }
            )
            stats["inputs_failed"] += 1
            continue
        for segment in segments:
            context = ExtractionContext(
                item.example_id if segment.segment_id is None else f"{item.example_id}:{segment.segment_id}",
                item.document_id,
                segment.segment_id,
                condition,
                segment.start,
                segment.end,
            )
            try:
                raw = extractor.extract(segment.text, context)
                stats["raw_outputs"] += 1
                raw_segments.append(
                    {
                        "generation_metadata": dict(raw.generation_metadata),
                        "model_output": raw.model_output,
                        "segment_end": segment.end,
                        "segment_id": segment.segment_id,
                        "segment_start": segment.start,
                        "text": segment.text,
                    }
                )
                parsed = parse_output(raw.model_output, segment.segment_id)
                aligned = align_triples(parsed.triples, segment.text, segment.start)
                parsed_all.extend(aligned)
                parse_issues.extend(parsed.issues)
            except Exception as exc:
                example_failed = True
                failures.append(
                    {
                        "error_message": str(exc),
                        "error_type": type(exc).__name__,
                        "example_id": item.example_id,
                        "recoverable": True,
                        "segment_id": segment.segment_id,
                        "stage": "extraction",
                    }
                )
        parsed_tuple = tuple(parsed_all)
        normalized = normalize_triples(parsed_tuple)
        validated = validate_structural(normalized, tuple(parse_issues))
        stats["parsed_triples"] += len(parsed_tuple)
        stats["parse_failures"] += sum(
            issue.code in {"UNPARSEABLE_CHUNK", "EMPTY_MODEL_OUTPUT"} for issue in parse_issues
        )
        stats["duplicates"] += sum(
            item.code == "DUPLICATE_TRIPLE" for item in validated.violations
        )
        stats["alignment_failures"] += sum(
            triple.subject_span is None or triple.object_span is None for triple in normalized
        )
        stats["valid_triples"] += sum(
            bool(triple.subject and triple.relation and triple.object) for triple in normalized
        )
        prediction: dict[str, Any] = {
            "condition": condition,
            "document_id": item.document_id,
            "example_id": item.example_id,
            "input": {"text": item.text},
            "parsed": [triple_to_dict(triple) for triple in parsed_tuple],
            "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
            "raw": {
                "model_output": "\n".join(value["model_output"] for value in raw_segments),
                "segments": raw_segments,
            },
            "normalized": [triple_to_dict(triple) for triple in normalized],
            "run_id": manifest["run_id"],
            "segment_id": None,
            "validated": {
                "status": validated.status,
                "triples": [triple_to_dict(triple) for triple in validated.triples],
                "violations": [asdict(value) for value in validated.violations],
            },
        }
        if legacy_relation_filter:
            prediction["derived"] = {
                "legacy_filtered": [
                    triple_to_dict(triple)
                    for triple in normalized
                    if _legacy_relation_selected(triple.relation)
                ]
            }
        predictions.append(prediction)
        if example_failed:
            stats["inputs_failed"] += 1
        else:
            stats["inputs_succeeded"] += 1
    _jsonl(run_dir / "predictions.jsonl", predictions)
    _jsonl(run_dir / "failures.jsonl", failures)
    (run_dir / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    (run_dir / "statistics.json").write_text(canonical_json(stats) + "\n", encoding="utf-8")
    return run_dir


def export_evaluation(predictions_path: str | Path, output_path: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    with Path(predictions_path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            for index, triple in enumerate(record["validated"]["triples"]):
                rows.append(
                    {
                        "document_id": record["document_id"],
                        "id": f'{record["run_id"]}:{record["example_id"]}:{index:06d}',
                        "object": triple["object"],
                        "prediction_id": f'{record["run_id"]}:{record["example_id"]}:{index:06d}',
                        "relation": triple["relation"],
                        "segment_id": triple.get("segment_id"),
                        "subject": triple["subject"],
                    }
                )
    target = Path(output_path)
    _jsonl(target, rows)
    return target


def benchmark_identity(documents: str | Path, manifest: str | Path | None) -> tuple[dict, dict]:
    dataset = {
        "hash": sha256_file(documents),
        "id": "UNKNOWN",
        "version": "UNKNOWN",
    }
    source = {"documents_sha256": dataset["hash"], "manifest_sha256": None}
    if manifest:
        manifest_value = json.loads(Path(manifest).read_text(encoding="utf-8"))
        dataset["id"] = manifest_value.get("benchmark_id", "UNKNOWN")
        dataset["version"] = manifest_value.get("benchmark_version", "UNKNOWN")
        source["manifest_sha256"] = sha256_file(manifest)
    return dataset, source
