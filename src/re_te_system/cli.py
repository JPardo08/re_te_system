"""File-contract CLI for extraction and evaluation export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from re_te_system.extractors.mock import MockExtractor, MockRebelExtractor
from re_te_system.extractors.mrebel import MRebelConfig, MRebelExtractor
from re_te_system.extractors.rebel import RebelConfig, RebelExtractor
from re_te_system.parsing.mrebel import parse_mrebel
from re_te_system.parsing.rebel import parse_rebel
from re_te_system.runner.run import (
    WindowingConfig,
    benchmark_identity,
    export_evaluation,
    load_hohfeld_documents,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="re-te-system")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--documents", required=True)
    run.add_argument("--benchmark-manifest")
    run.add_argument("--output-root", required=True)
    run.add_argument(
        "--extractor",
        choices=("mock", "rebel-mock", "mrebel", "rebel"),
        default="mock",
    )
    run.add_argument("--model-name")
    run.add_argument("--model-revision")
    run.add_argument("--src-lang", default="es_XX")
    run.add_argument("--target-token", default="tp_XX")
    run.add_argument("--device", default="auto")
    run.add_argument("--dtype", default="float32")
    run.add_argument("--max-input-tokens", type=int, default=256)
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--max-length", type=int, default=512)
    run.add_argument("--num-beams", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--window-strategy", choices=("NONE", "CHARACTER", "TOKEN"))
    run.add_argument("--window-max-units", type=int)
    run.add_argument("--window-overlap", type=int, default=0)
    run.add_argument("--legacy-relation-filter", action="store_true")
    export = commands.add_parser("export-evaluation")
    export.add_argument("--predictions", required=True)
    export.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export-evaluation":
        output = export_evaluation(args.predictions, args.output)
        print(json.dumps({"output": str(output)}, sort_keys=True))
        return 0
    if args.legacy_relation_filter and args.extractor not in {"mock", "mrebel"}:
        raise SystemExit("--legacy-relation-filter is mREBEL-specific")

    inputs = load_hohfeld_documents(args.documents)
    dataset, benchmark_source = benchmark_identity(
        args.documents, args.benchmark_manifest
    )
    if args.extractor == "mock":
        extractor = MockExtractor()
        parse_output = parse_mrebel
        generation = {
            "do_sample": False,
            "early_stopping": True,
            "max_new_tokens": None,
            "num_beams": 1,
            "seed": None,
            "temperature": None,
        }
        strategy = args.window_strategy or "NONE"
        model_metadata = {
            "device": "none",
            "dtype": "none",
            "language": {"source": "es_XX", "target_token": "tp_XX"},
            "tokenizer": "none",
        }
        model_family = "mrebel"
        language_status = "supported"
    elif args.extractor == "rebel-mock":
        extractor = MockRebelExtractor()
        parse_output = parse_rebel
        generation = {
            "do_sample": False,
            "early_stopping": False,
            "length_penalty": 0.0,
            "max_length": None,
            "num_beams": 1,
            "seed": None,
            "temperature": None,
        }
        strategy = args.window_strategy or "NONE"
        model_metadata = {
            "device": "none",
            "dtype": "none",
            "language": {"input": "es", "language_tokens": "none"},
            "tokenizer": "none",
        }
        model_family = "rebel"
        language_status = "out_of_primary_model_scope"
    elif args.extractor == "mrebel":
        config = MRebelConfig(
            model_name=args.model_name or "Babelscape/mrebel-large",
            revision=args.model_revision,
            src_lang=args.src_lang,
            target_token=args.target_token,
            device=args.device,
            dtype=args.dtype,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        extractor = MRebelExtractor(config)
        parse_output = parse_mrebel
        generation = config.generation_config()
        strategy = args.window_strategy or "TOKEN"
        model_metadata = {
            "device": str(extractor.device),
            "dtype": config.dtype,
            "language": {"source": config.src_lang, "target_token": config.target_token},
            "tokenizer": config.tokenizer_name or config.model_name,
        }
        model_family = "mrebel"
        language_status = "supported"
    else:
        config = RebelConfig(
            model_name=args.model_name or "Babelscape/rebel-large",
            revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            max_input_tokens=args.max_input_tokens,
            max_length=args.max_length,
            num_beams=args.num_beams,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        extractor = RebelExtractor(config)
        parse_output = parse_rebel
        generation = config.generation_config()
        strategy = args.window_strategy or "TOKEN"
        model_metadata = {
            "device": str(extractor.device),
            "dtype": config.dtype,
            "language": {"input": "es", "language_tokens": "none"},
            "tokenizer": config.tokenizer_name or config.model_name,
        }
        model_family = "rebel"
        language_status = "out_of_primary_model_scope"
    maximum = args.window_max_units
    if strategy == "TOKEN" and maximum is None:
        maximum = args.max_input_tokens
    windowing = WindowingConfig(
        strategy=strategy,
        max_units=maximum,
        overlap=args.window_overlap,
        segmentation_mechanism=(
            "whole_document" if strategy == "NONE" else f"{strategy.lower()}_sliding_window"
        ),
    )
    run_dir = run_pipeline(
        inputs=inputs,
        extractor=extractor,
        output_root=args.output_root,
        dataset=dataset,
        benchmark_source=benchmark_source,
        generation=generation,
        windowing=windowing,
        model_metadata=model_metadata,
        legacy_relation_filter=args.legacy_relation_filter,
        parse_output=parse_output,
        run_role="baseline",
        model_family=model_family,
        target_schema_knowledge="none",
        native_schema={"id": "wikidata_like", "inherited_from_model": True},
        input_language="es",
        language_status=language_status,
    )
    print(json.dumps({"run_dir": str(run_dir), "run_id": run_dir.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
