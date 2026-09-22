"""File-contract CLI for extraction and evaluation export."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from re_te_system.conditioning.uie_schema import UIESchema
from re_te_system.extractors.mock import (
    MockExtractor,
    MockPythiaExtractor,
    MockRebelExtractor,
    MockUIEExtractor,
)
from re_te_system.extractors.mrebel import MRebelConfig, MRebelExtractor
from re_te_system.extractors.pythia import PythiaConfig, PythiaSpaceKBPExtractor
from re_te_system.extractors.rebel import RebelConfig, RebelExtractor
from re_te_system.extractors.uie import UIEConfig, UIEExtractor
from re_te_system.parsing.mrebel import parse_mrebel
from re_te_system.parsing.rebel import parse_rebel
from re_te_system.parsing.sel import (
    align_spot_structures,
    parse_sel,
    project_binary_relations,
)
from re_te_system.parsing.turtle import parse_turtle
from re_te_system.runner.run import (
    WindowingConfig,
    benchmark_identity,
    export_evaluation,
    load_hohfeld_documents,
    run_pipeline,
)


def _prefixes(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        prefix, separator, namespace = value.partition("=")
        if not separator or not prefix or not namespace:
            raise SystemExit("--turtle-prefix requires PREFIX=URI")
        result[prefix] = namespace
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="re-te-system")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--documents", required=True)
    run.add_argument("--benchmark-manifest")
    run.add_argument("--output-root", required=True)
    run.add_argument(
        "--extractor",
        choices=(
            "mock",
            "rebel-mock",
            "pythia-mock",
            "uie-mock",
            "mrebel",
            "rebel",
            "pythia",
            "uie",
        ),
        default="mock",
    )
    run.add_argument("--model-name")
    run.add_argument("--model-revision")
    run.add_argument("--src-lang", default="es_XX")
    run.add_argument("--target-token", default="tp_XX")
    run.add_argument("--device", default="auto")
    run.add_argument("--dtype", default="float32")
    run.add_argument("--input-language", default="es")
    run.add_argument("--max-input-tokens", type=int, default=256)
    run.add_argument("--pythia-max-input-tokens", type=int)
    run.add_argument("--uie-max-source-tokens", type=int, default=256)
    run.add_argument("--uie-max-target-tokens", type=int, default=192)
    run.add_argument("--uie-num-beams", type=int, default=1)
    run.add_argument("--uie-schema-file")
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--max-length", type=int, default=512)
    run.add_argument("--num-beams", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--window-strategy", choices=("NONE", "CHARACTER", "TOKEN"))
    run.add_argument("--window-max-units", type=int)
    run.add_argument("--window-overlap", type=int, default=0)
    run.add_argument("--prompt-profile", choices=("basic",), default="basic")
    run.add_argument("--turtle-prefix", action="append", default=[])
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
    task_class = None
    controlled_experiment_condition = None
    structure_aligner = None
    structure_projector = None
    target_schema_knowledge = "none"
    native_schema = {"id": "wikidata_like", "inherited_from_model": True}
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
    elif args.extractor == "pythia-mock":
        prefixes = _prefixes(args.turtle_prefix)
        extractor = MockPythiaExtractor()
        parse_output = partial(parse_turtle, prefixes=prefixes)
        generation = PythiaConfig().generation_config()
        strategy = args.window_strategy or "NONE"
        model_metadata = {
            "base_model": "EleutherAI/pythia-1b-deduped",
            "device": "none",
            "dtype": "none",
            "effective_input_limit": 1536,
            "model_max_length": 2048,
            "prompt_profile": args.prompt_profile,
            "quantization": "none",
            "remote_download_authorized": False,
            "tokenizer": "mock",
            "turtle_prefix_context": prefixes,
        }
        model_family = "pythia_spacekbp"
        language_status = (
            "native_domain" if args.input_language == "en" else "out_of_native_domain"
        )
        task_class = "KBP_DOMAIN_BASELINE"
        controlled_experiment_condition = "not_applicable"
        native_schema = {
            "id": "space_kbp_space_ontology",
            "inherited_from_model": True,
            "scope": "fixed_domain_ontology",
        }
    elif args.extractor == "uie-mock":
        if not args.uie_schema_file:
            raise SystemExit("--uie-schema-file is required for UIE extractors")
        schema = UIESchema.read_json(args.uie_schema_file)
        extractor = MockUIEExtractor()
        parse_output = partial(
            parse_sel,
            spot_labels=schema.spot_labels,
            association_labels=schema.association_labels,
        )
        structure_aligner = align_spot_structures
        structure_projector = project_binary_relations
        generation = UIEConfig().generation_config()
        strategy = args.window_strategy or "NONE"
        model_metadata = {
            "base_model": "t5-v1_1-base",
            "constraint_decoding": False,
            "device": "none",
            "dtype": "none",
            "interface_accepts_custom_schema": True,
            "parser_policy": "conservative_sel_v1",
            "schema": schema.manifest_metadata(),
            "tokenizer": "mock",
            "zero_shot_unseen_schema_supported": "unknown",
        }
        model_family = "uie"
        language_status = "out_of_documented_training_scope"
        task_class = "UNIVERSAL_IE"
        controlled_experiment_condition = "not_applicable"
        target_schema_knowledge = "structural_schema"
        native_schema = {
            "id": schema.schema_id,
            "inherited_from_model": False,
            "scope": "dynamic_runtime_structural_schema",
            "schema_hash": schema.stable_hash(),
            "schema_version": schema.schema_version,
        }
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
    elif args.extractor == "rebel":
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
    elif args.extractor == "pythia":
        prefixes = _prefixes(args.turtle_prefix)
        config = PythiaConfig(
            model_name=args.model_name or PythiaConfig().model_name,
            revision=args.model_revision or PythiaConfig().revision,
            prompt_profile=args.prompt_profile,
            device=args.device,
            dtype=args.dtype,
            max_input_tokens=args.pythia_max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            local_files_only=True,
        )
        extractor = PythiaSpaceKBPExtractor(config)
        parse_output = partial(parse_turtle, prefixes=prefixes)
        generation = config.generation_config()
        strategy = args.window_strategy or "TOKEN"
        model_metadata = {
            "base_model": "EleutherAI/pythia-1b-deduped",
            "device": str(extractor.device),
            "dtype": config.dtype,
            "effective_input_limit": extractor.effective_input_limit,
            "model_max_length": extractor.model_max_length,
            "prompt_profile": config.prompt_profile,
            "prompt_profile_version": "basic-v1",
            "quantization": config.quantization,
            "remote_download_authorized": False,
            "tokenizer": config.tokenizer_name or config.model_name,
            "turtle_prefix_context": prefixes,
        }
        model_family = "pythia_spacekbp"
        language_status = (
            "native_domain" if args.input_language == "en" else "out_of_native_domain"
        )
        task_class = "KBP_DOMAIN_BASELINE"
        controlled_experiment_condition = "not_applicable"
        native_schema = {
            "id": "space_kbp_space_ontology",
            "inherited_from_model": True,
            "scope": "fixed_domain_ontology",
        }
    else:
        if not args.uie_schema_file:
            raise SystemExit("--uie-schema-file is required for UIE extractors")
        schema = UIESchema.read_json(args.uie_schema_file)
        config = UIEConfig(
            model_name=args.model_name or UIEConfig().model_name,
            revision=args.model_revision or UIEConfig().revision,
            device=args.device,
            dtype=args.dtype,
            max_source_tokens=args.uie_max_source_tokens,
            max_target_tokens=args.uie_max_target_tokens,
            num_beams=args.uie_num_beams,
            seed=args.seed,
            local_files_only=True,
        )
        extractor = UIEExtractor(schema=schema, config=config)
        parse_output = partial(
            parse_sel,
            spot_labels=schema.spot_labels,
            association_labels=schema.association_labels,
        )
        structure_aligner = align_spot_structures
        structure_projector = project_binary_relations
        generation = config.generation_config()
        strategy = args.window_strategy or "TOKEN"
        model_metadata = {
            "base_model": "t5-v1_1-base",
            "constraint_decoding": False,
            "device": str(extractor.device),
            "dtype": config.dtype,
            "effective_input_limit": extractor.effective_input_limit,
            "interface_accepts_custom_schema": True,
            "model_max_length": extractor.model_max_length,
            "parser_policy": "conservative_sel_v1",
            "schema": schema.manifest_metadata(),
            "tokenizer": extractor.tokenizer_name,
            "zero_shot_unseen_schema_supported": "unknown",
        }
        model_family = "uie"
        language_status = "out_of_documented_training_scope"
        task_class = "UNIVERSAL_IE"
        controlled_experiment_condition = "not_applicable"
        target_schema_knowledge = "structural_schema"
        native_schema = {
            "id": schema.schema_id,
            "inherited_from_model": False,
            "scope": "dynamic_runtime_structural_schema",
            "schema_hash": schema.stable_hash(),
            "schema_version": schema.schema_version,
        }
    maximum = args.window_max_units
    if strategy == "TOKEN" and maximum is None:
        maximum = getattr(extractor, "effective_input_limit", args.max_input_tokens)
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
        structure_aligner=structure_aligner,
        structure_projector=structure_projector,
        run_role="baseline",
        model_family=model_family,
        target_schema_knowledge=target_schema_knowledge,
        native_schema=native_schema,
        input_language=args.input_language,
        language_status=language_status,
        task_class=task_class,
        controlled_experiment_condition=controlled_experiment_condition,
    )
    print(json.dumps({"run_dir": str(run_dir), "run_id": run_dir.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
