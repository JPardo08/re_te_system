"""File-contract CLI for extraction and evaluation export."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from re_te_system.conditioning.genie_constraints import (
    PROFILE_CUSTOM_FULL,
    PROFILE_LARGE,
    PROFILE_SMALL,
    PROFILE_UNCONSTRAINED,
    GenIEConstraintSpec,
    closed_schema_spec,
    named_schema_spec,
    unconstrained_spec,
)
from re_te_system.conditioning.gollie_schema import GoLLIESchema
from re_te_system.conditioning.uie_schema import UIESchema
from re_te_system.extractors.genie import (
    CANONICAL_ARCHITECTURE,
    CANONICAL_CHECKPOINT_MD5,
    CANONICAL_CHECKPOINT_NAME,
    CANONICAL_TOKENIZER_ID,
    CANONICAL_TOKENIZER_REVISION,
    CHECKPOINT_LOADING_FORM,
    CHECKPOINT_LOADING_STATUS,
    EVALUATOR_BEAM_POLICY,
    GENIE_DOCUMENTED_LANGUAGE,
    GENIE_SCIENTIFIC_ROLE,
    GENIE_TARGET_SCHEMA_KNOWLEDGE,
    GENIE_TASK_CLASS,
    MODERNIZED_LOADING_STATUS,
    GenIEConfig,
    GenIEExtractor,
)
from re_te_system.extractors.gollie import (
    GOLLIE_DOCUMENTED_LANGUAGE,
    GOLLIE_MERGED_FULL_MODEL,
    GOLLIE_SCIENTIFIC_ROLE,
    GOLLIE_WEIGHT_LICENSE,
    GoLLIEConfig,
    GoLLIEExtractor,
)
from re_te_system.extractors.mock import (
    MockExtractor,
    MockGenieExtractor,
    MockGoLLIEExtractor,
    MockPythiaExtractor,
    MockRebelExtractor,
    MockUIEExtractor,
)
from re_te_system.extractors.mrebel import MRebelConfig, MRebelExtractor
from re_te_system.extractors.pythia import PythiaConfig, PythiaSpaceKBPExtractor
from re_te_system.extractors.rebel import RebelConfig, RebelExtractor
from re_te_system.extractors.uie import UIEConfig, UIEExtractor
from re_te_system.parsing.genie import parse_genie
from re_te_system.parsing.gollie import (
    align_gollie_records,
    parse_gollie,
    project_gollie_relations,
)
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


def _inventory_strings(path: str | None) -> tuple[str, ...]:
    if not path:
        return ()
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item:
            continue
        if item.startswith('"') and item.endswith('"'):
            item = json.loads(item)
        values.append(item)
    return tuple(values)


def _genie_constraint_spec(args: argparse.Namespace) -> GenIEConstraintSpec:
    profile = args.genie_constraint_profile
    if profile == PROFILE_UNCONSTRAINED:
        return unconstrained_spec()
    if profile in {PROFILE_SMALL, PROFILE_LARGE}:
        return named_schema_spec(profile)
    entities = _inventory_strings(args.genie_entity_inventory)
    relations = _inventory_strings(args.genie_relation_inventory)
    if not entities or not relations:
        raise SystemExit(
            "custom_full GenIE constraints require --genie-entity-inventory "
            "and --genie-relation-inventory"
        )
    return closed_schema_spec(entities, relations)


def _genie_language_status(input_language: str) -> str:
    return (
        "supported"
        if input_language == "en"
        else "out_of_documented_training_scope"
    )


def _genie_model_metadata(
    constraint_spec: GenIEConstraintSpec,
    *,
    device: str,
    dtype: str,
    tokenizer: str,
) -> dict:
    return {
        "architecture": CANONICAL_ARCHITECTURE,
        "checkpoint": {
            "loading_form": CHECKPOINT_LOADING_FORM,
            "loading_status": CHECKPOINT_LOADING_STATUS,
            "md5": CANONICAL_CHECKPOINT_MD5,
            "modernized_loading_status": MODERNIZED_LOADING_STATUS,
            "name": CANONICAL_CHECKPOINT_NAME,
        },
        "constraint": constraint_spec.manifest_metadata(),
        "constraint_decoding": constraint_spec.constrained,
        "device": device,
        "documented_language": GENIE_DOCUMENTED_LANGUAGE,
        "dtype": dtype,
        "evaluator_beam_policy": EVALUATOR_BEAM_POLICY,
        "formal_ontology": False,
        "parser_policy": "conservative_genie_v1",
        "scientific_role": GENIE_SCIENTIFIC_ROLE,
        "selected_beam_policy": EVALUATOR_BEAM_POLICY,
        "tokenizer": tokenizer,
        "tokenizer_revision": CANONICAL_TOKENIZER_REVISION,
    }


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
            "gollie-mock",
            "genie-mock",
            "mrebel",
            "rebel",
            "pythia",
            "uie",
            "gollie",
            "genie",
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
    run.add_argument("--gollie-schema-file")
    run.add_argument("--gollie-max-new-tokens", type=int, default=128)
    run.add_argument(
        "--genie-constraint-profile",
        choices=(
            PROFILE_UNCONSTRAINED,
            PROFILE_SMALL,
            PROFILE_LARGE,
            PROFILE_CUSTOM_FULL,
        ),
        default=PROFILE_UNCONSTRAINED,
    )
    run.add_argument("--genie-entity-inventory")
    run.add_argument("--genie-relation-inventory")
    run.add_argument("--genie-checkpoint-path")
    run.add_argument("--genie-num-beams", type=int, default=10)
    run.add_argument("--authorize-checkpoint-load", action="store_true")
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
    record_aligner = None
    record_projector = None
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
        language_status = (
            "supported"
            if args.input_language == "en"
            else "out_of_documented_training_scope"
        )
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
    elif args.extractor == "gollie-mock":
        if not args.gollie_schema_file:
            raise SystemExit("--gollie-schema-file is required for GoLLIE extractors")
        schema = GoLLIESchema.read_json(args.gollie_schema_file)
        extractor = MockGoLLIEExtractor(schema=schema)
        parse_output = partial(parse_gollie, schema=schema)
        record_aligner = align_gollie_records
        record_projector = project_gollie_relations
        generation = GoLLIEConfig().generation_config()
        strategy = args.window_strategy or "NONE"
        model_metadata = {
            "base_model": "codellama/CodeLlama-7b-hf",
            "constraint_decoding": False,
            "custom_modeling": True,
            "definitions_guidelines": True,
            "device": "none",
            "documented_language": GOLLIE_DOCUMENTED_LANGUAGE,
            "dtype": "none",
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
        }
        model_family = "gollie"
        language_status = (
            "supported"
            if args.input_language == "en"
            else "out_of_documented_training_scope"
        )
        task_class = "UNIVERSAL_IE"
        controlled_experiment_condition = "not_applicable"
        target_schema_knowledge = "structural_schema+definitions_guidelines"
        native_schema = {
            "definitions_guidelines": True,
            "dynamic_runtime_schema": True,
            "formal_ontology": False,
            "id": schema.schema_id,
            "inherited_from_model": False,
            "schema_hash": schema.schema_hash(),
            "schema_version": schema.schema_version,
            "scope": "dynamic_runtime_guideline_schema",
        }
    elif args.extractor == "genie-mock":
        constraint_spec = _genie_constraint_spec(args)
        extractor = MockGenieExtractor(constraint_spec=constraint_spec)
        parse_output = parse_genie
        generation = GenIEConfig(
            constraint_profile=constraint_spec.profile,
            num_beams=args.genie_num_beams,
            num_return_sequences=min(args.genie_num_beams, 2),
            seed=args.seed,
        ).generation_config()
        strategy = args.window_strategy or "NONE"
        model_metadata = _genie_model_metadata(
            constraint_spec,
            device="none",
            dtype="none",
            tokenizer="mock",
        )
        model_family = "genie"
        language_status = _genie_language_status(args.input_language)
        task_class = GENIE_TASK_CLASS
        controlled_experiment_condition = "not_applicable"
        target_schema_knowledge = GENIE_TARGET_SCHEMA_KNOWLEDGE
        native_schema = {
            "constraint_profile": constraint_spec.profile,
            "entity_inventory_required": constraint_spec.entity_inventory_required,
            "formal_ontology": False,
            "id": "wikidata_closed_schema",
            "inherited_from_model": True,
            "relation_inventory_required": constraint_spec.relation_inventory_required,
            "scope": "fixed_native_schema+kb_constraints",
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
    elif args.extractor == "uie":
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
        language_status = (
            "supported"
            if args.input_language == "en"
            else "out_of_documented_training_scope"
        )
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
    elif args.extractor == "gollie":
        if not args.gollie_schema_file:
            raise SystemExit("--gollie-schema-file is required for GoLLIE extractors")
        schema = GoLLIESchema.read_json(args.gollie_schema_file)
        config = GoLLIEConfig(
            model_name=args.model_name or GoLLIEConfig().model_name,
            revision=args.model_revision or GoLLIEConfig().revision,
            device=args.device if args.device in {"auto", "cuda"} else "auto",
            dtype=args.dtype if args.dtype != "float32" else "bfloat16",
            max_new_tokens=args.gollie_max_new_tokens,
            seed=args.seed,
            local_files_only=True,
        )
        extractor = GoLLIEExtractor(schema=schema, config=config)
        parse_output = partial(parse_gollie, schema=schema)
        record_aligner = align_gollie_records
        record_projector = project_gollie_relations
        generation = config.generation_config()
        strategy = args.window_strategy or "TOKEN"
        model_metadata = {
            "base_model": "codellama/CodeLlama-7b-hf",
            "constraint_decoding": False,
            "custom_modeling": True,
            "definitions_guidelines": True,
            "device": str(extractor.device),
            "documented_language": GOLLIE_DOCUMENTED_LANGUAGE,
            "dtype": config.dtype,
            "dynamic_runtime_schema": True,
            "effective_input_limit": extractor.effective_input_limit,
            "flash_attention_required": True,
            "formal_ontology": False,
            "interface_accepts_custom_schema": True,
            "merged_full_model": GOLLIE_MERGED_FULL_MODEL,
            "model_max_length": extractor.model_max_length,
            "parser_policy": "safe_ast_gollie_v1",
            "quantization": config.quantization,
            "schema": schema.manifest_metadata(),
            "scientific_role": GOLLIE_SCIENTIFIC_ROLE,
            "tokenizer": extractor.tokenizer_name,
            "weight_license": GOLLIE_WEIGHT_LICENSE,
            "zero_shot_unseen_schema_supported": "documented_with_limitations",
        }
        model_family = "gollie"
        language_status = (
            "supported"
            if args.input_language == "en"
            else "out_of_documented_training_scope"
        )
        task_class = "UNIVERSAL_IE"
        controlled_experiment_condition = "not_applicable"
        target_schema_knowledge = "structural_schema+definitions_guidelines"
        native_schema = {
            "definitions_guidelines": True,
            "dynamic_runtime_schema": True,
            "formal_ontology": False,
            "id": schema.schema_id,
            "inherited_from_model": False,
            "schema_hash": schema.schema_hash(),
            "schema_version": schema.schema_version,
            "scope": "dynamic_runtime_guideline_schema",
        }
    else:
        constraint_spec = _genie_constraint_spec(args)
        config = GenIEConfig(
            checkpoint_path=args.genie_checkpoint_path,
            constraint_profile=constraint_spec.profile,
            num_beams=args.genie_num_beams,
            num_return_sequences=args.genie_num_beams,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            local_files_only=True,
            authorize_checkpoint_load=args.authorize_checkpoint_load,
        )
        extractor = GenIEExtractor(config=config, constraint_spec=constraint_spec)
        parse_output = parse_genie
        generation = config.generation_config()
        strategy = args.window_strategy or "TOKEN"
        model_metadata = _genie_model_metadata(
            constraint_spec,
            device=str(extractor.device),
            dtype=config.dtype,
            tokenizer=extractor.tokenizer_name,
        )
        model_family = "genie"
        language_status = _genie_language_status(args.input_language)
        task_class = GENIE_TASK_CLASS
        controlled_experiment_condition = "not_applicable"
        target_schema_knowledge = GENIE_TARGET_SCHEMA_KNOWLEDGE
        native_schema = {
            "constraint_profile": constraint_spec.profile,
            "entity_inventory_required": constraint_spec.entity_inventory_required,
            "formal_ontology": False,
            "id": "wikidata_closed_schema",
            "inherited_from_model": True,
            "relation_inventory_required": constraint_spec.relation_inventory_required,
            "scope": "fixed_native_schema+kb_constraints",
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
        record_aligner=record_aligner,
        record_projector=record_projector,
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
