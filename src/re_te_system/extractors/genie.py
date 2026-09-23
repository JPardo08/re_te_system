"""Clean-room GenIE extractor. P0 does not load the 4.9 GB Lightning checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from re_te_system.conditioning.genie_constraints import (
    OFFICIAL_TRIE_DESERIALIZATION,
    PROFILE_UNCONSTRAINED,
    GenIEConstraintSpec,
    unconstrained_spec,
)
from re_te_system.contracts import ExtractionContext, RawExtractionResult


CANONICAL_CHECKPOINT_NAME = "genie_r.ckpt"
CANONICAL_CHECKPOINT_SIZE_BYTES = 4_879_373_206
CANONICAL_CHECKPOINT_MD5 = "c214da56b6e5d5bd259e0cbe826d92f7"
CANONICAL_CHECKPOINT_SOURCE = "zenodo:6139236"
CANONICAL_CHECKPOINT_URL = "https://zenodo.org/record/6139236/files/genie_r.ckpt"
CANONICAL_TRAINING = "REBEL"
CANONICAL_INITIALIZATION = "random"
CANONICAL_ARCHITECTURE = "facebook/bart-large"
CANONICAL_TOKENIZER_ID = "martinjosifoski/genie-rw"
CANONICAL_TOKENIZER_REVISION = "81eb2ccca714fbb62a65283b65c9c153469ee409"
GENIE_CODE_LICENSE = "MIT"
GENIE_CHECKPOINT_LICENSE = "CC-BY-4.0"
GENIE_SCIENTIFIC_ROLE = "CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE"
GENIE_DOCUMENTED_LANGUAGE = "en"
GENIE_TASK_CLASS = "KBP_CLOSED_IE"
GENIE_TARGET_SCHEMA_KNOWLEDGE = "fixed_native_schema+kb_constraints"
CHECKPOINT_LOADING_FORM = "LIGHTNING_CHECKPOINT_NATIVE"
CHECKPOINT_LOADING_STATUS = "not_real_smoked"
MODERNIZED_LOADING_STATUS = "unverified"
REAL_GENIE_NATIVE_SMOKE = "BLOCKED_RESOURCE"
OUTPUT_SOURCE_OFFICIAL_NOTEBOOK = "OFFICIAL_NOTEBOOK_RECORDED_OUTPUT"
EVALUATOR_BEAM_POLICY = "top_score_beam0"

OFFICIAL_NOTEBOOK_INPUT = (
    "Prior to KTRK, Carson was an anchor for KSAZ in Phoenix, Arizona."
)
OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW = (
    " <sub> KSAZ-TV <rel> headquarters location <obj> Phoenix, Arizona <et>"
)
OFFICIAL_NOTEBOOK_SMALL_RAW = (
    " <sub> Phoenix, Arizona <rel> capital of <obj> Arizona <et> "
    "<sub> Arizona <rel> capital <obj> Phoenix, Arizona <et>"
)
OFFICIAL_NOTEBOOK_LARGE_RAW = OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW
OFFICIAL_NOTEBOOK_UNCONSTRAINED_LOG_PROB = -0.1262681782245636
OFFICIAL_NOTEBOOK_SMALL_LOG_PROB = -0.30889713764190674
OFFICIAL_NOTEBOOK_LARGE_LOG_PROB = OFFICIAL_NOTEBOOK_UNCONSTRAINED_LOG_PROB

ORIGINAL_RUNTIME = {
    "hydra": "1.1",
    "python": "3.8",
    "pytorch": "1.8",
    "pytorch_lightning": "1.4.9",
    "transformers": "4.10",
}
FUTURE_INFERENCE_TARGET = "direct_bart_transformers_if_scientifically_equivalent"

CONTROL_TAG_BPE = {
    " <sub>": ["<s>", "Ġ<", "sub", ">", "</s>"],
    " <rel>": ["<s>", "Ġ<", "rel", ">", "</s>"],
    " <obj>": ["<s>", "Ġ<", "obj", ">", "</s>"],
    " <et>": ["<s>", "Ġ<", "et", ">", "</s>"],
}


@dataclass(frozen=True)
class GenIEConfig:
    checkpoint_name: str = CANONICAL_CHECKPOINT_NAME
    checkpoint_path: str | None = None
    checkpoint_size_bytes: int = CANONICAL_CHECKPOINT_SIZE_BYTES
    checkpoint_md5: str = CANONICAL_CHECKPOINT_MD5
    checkpoint_source: str = CANONICAL_CHECKPOINT_SOURCE
    tokenizer_name: str = CANONICAL_TOKENIZER_ID
    tokenizer_revision: str = CANONICAL_TOKENIZER_REVISION
    constraint_profile: str = PROFILE_UNCONSTRAINED
    max_input_length: int = 256
    max_output_length: int = 256
    num_beams: int = 10
    num_return_sequences: int = 10
    do_sample: bool = False
    seed: int = 123
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = True
    authorize_checkpoint_load: bool = False

    def __post_init__(self) -> None:
        if self.max_input_length <= 0 or self.max_output_length <= 0:
            raise ValueError("GenIE input and output lengths must be positive")
        if self.num_beams <= 0 or self.num_return_sequences <= 0:
            raise ValueError("beam settings must be positive")
        if self.num_return_sequences > self.num_beams:
            raise ValueError("num_return_sequences cannot exceed num_beams")
        if self.do_sample:
            raise ValueError("P0 GenIE requires deterministic do_sample=False")
        if self.checkpoint_md5 != CANONICAL_CHECKPOINT_MD5:
            raise ValueError("P0 freezes genie_r.ckpt MD5 identity")
        if self.checkpoint_size_bytes != CANONICAL_CHECKPOINT_SIZE_BYTES:
            raise ValueError("P0 freezes genie_r.ckpt size identity")

    def generation_config(self) -> dict[str, Any]:
        return {
            "constraint_profile": self.constraint_profile,
            "do_sample": self.do_sample,
            "early_stopping": False,
            "length_penalty": 1.0,
            "max_length": self.max_output_length,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "seed": self.seed,
            "selected_beam_policy": EVALUATOR_BEAM_POLICY,
            "temperature": 1.0,
        }

    def manifest_config(self) -> dict[str, Any]:
        return asdict(self)

    def checkpoint_identity(self) -> dict[str, Any]:
        return {
            "architecture": CANONICAL_ARCHITECTURE,
            "initialization": CANONICAL_INITIALIZATION,
            "loading_form": CHECKPOINT_LOADING_FORM,
            "loading_status": CHECKPOINT_LOADING_STATUS,
            "md5": self.checkpoint_md5,
            "modernized_loading_status": MODERNIZED_LOADING_STATUS,
            "name": self.checkpoint_name,
            "native_default_constraints": "large",
            "path": self.checkpoint_path,
            "size_bytes": self.checkpoint_size_bytes,
            "source": self.checkpoint_source,
            "training": CANONICAL_TRAINING,
            "url": CANONICAL_CHECKPOINT_URL,
        }


def _decode_sequence(tokenizer: Any, sequence: Any) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(sequence, skip_special_tokens=True)
    if isinstance(sequence, str):
        return sequence
    return str(sequence)


class GenIEExtractor:
    """Generate exact decoded GenIE markup. Real Lightning loading is unverified."""

    def __init__(
        self,
        config: GenIEConfig = GenIEConfig(),
        constraint_spec: GenIEConstraintSpec | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.constraint_spec = constraint_spec or unconstrained_spec()
        self.model_name = config.checkpoint_name
        self.model_revision = config.checkpoint_md5
        self.tokenizer_name = config.tokenizer_name
        self.model_max_length = config.max_input_length
        self.effective_input_limit = config.max_input_length
        self.device = config.device
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            return
        raise RuntimeError(
            "P0 does not load genie_r.ckpt and does not auto-download 4.9 GB. "
            f"checkpoint_loading_status={CHECKPOINT_LOADING_STATUS}; "
            f"modernized_loading_status={MODERNIZED_LOADING_STATUS}; "
            f"loading_form={CHECKPOINT_LOADING_FORM}. "
            "Inject a model and tokenizer, or use the mock extractor. "
            "The Hugging Face tokenizer-only repo martinjosifoski/genie-rw "
            "does not contain weights."
        )

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=True)
        return len(list(encoded.get("input_ids", [])))

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
        )
        input_ids = encoded["input_ids"]
        untruncated = self.tokenizer(text, add_special_tokens=True)
        untruncated_count = len(list(untruncated.get("input_ids", [])))
        truncated_input = untruncated_count > self.effective_input_limit
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": encoded.get("attention_mask"),
            "num_beams": self.config.num_beams,
            "num_return_sequences": self.config.num_return_sequences,
            "max_length": self.config.max_output_length,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        generated = self.model.generate(**generate_kwargs)
        sequences = list(generated["sequences"])
        scores = list(generated.get("sequences_scores", [None] * len(sequences)))
        beams = []
        for rank, (sequence, score) in enumerate(zip(sequences, scores)):
            raw = _decode_sequence(self.tokenizer, sequence)
            beams.append(
                {
                    "beam_rank": rank,
                    "log_prob": None if score is None else float(score),
                    "raw": raw,
                }
            )
        beams.sort(key=lambda item: float("-inf") if item["log_prob"] is None else item["log_prob"], reverse=True)
        for index, beam in enumerate(beams):
            beam["beam_rank"] = index
        selected = beams[0] if beams else {"beam_rank": 0, "log_prob": None, "raw": ""}
        return RawExtractionResult(
            model_output=selected["raw"],
            generation_metadata=self._metadata(
                text=text,
                beams=beams,
                input_token_count=int(getattr(input_ids, "shape", [1, untruncated_count])[-1])
                if hasattr(input_ids, "shape")
                else untruncated_count,
                untruncated_input_token_count=untruncated_count,
                truncated_input=truncated_input,
                output_token_count=len(selected["raw"].split()),
            ),
        )

    def _metadata(
        self,
        *,
        text: str,
        beams: list[dict[str, Any]],
        input_token_count: int,
        untruncated_input_token_count: int,
        truncated_input: bool,
        output_token_count: int,
    ) -> dict[str, Any]:
        return {
            "architecture": CANONICAL_ARCHITECTURE,
            "beams": beams,
            "checkpoint": self.config.checkpoint_identity(),
            "constraint": self.constraint_spec.manifest_metadata(),
            "constraint_decoding": self.constraint_spec.constrained,
            "control_tags_are_added_special_tokens": False,
            "deterministic": True,
            "documented_language": GENIE_DOCUMENTED_LANGUAGE,
            "effective_input_limit": self.effective_input_limit,
            "evaluator_beam_policy": EVALUATOR_BEAM_POLICY,
            "future_inference_target": FUTURE_INFERENCE_TARGET,
            "input_token_count": input_token_count,
            "max_input_length": self.config.max_input_length,
            "max_output_length": self.config.max_output_length,
            "model_max_length": self.model_max_length,
            "number_of_sequences": len(beams),
            "official_trie_deserialization": OFFICIAL_TRIE_DESERIALIZATION,
            "original_runtime": dict(ORIGINAL_RUNTIME),
            "output_reached_limit": False,
            "output_token_count": output_token_count,
            "scientific_role": GENIE_SCIENTIFIC_ROLE,
            "seed": self.config.seed,
            "selected_beam_policy": EVALUATOR_BEAM_POLICY,
            "selected_beam_rank": 0,
            "tokenizer": self.tokenizer_name,
            "tokenizer_revision": self.config.tokenizer_revision,
            "truncated_input": truncated_input,
            "untruncated_input_token_count": untruncated_input_token_count,
        }


def official_notebook_raw_for_profile(profile: str) -> str:
    if profile == "small":
        return OFFICIAL_NOTEBOOK_SMALL_RAW
    if profile == "large":
        return OFFICIAL_NOTEBOOK_LARGE_RAW
    return OFFICIAL_NOTEBOOK_UNCONSTRAINED_RAW


def official_notebook_log_prob_for_profile(profile: str) -> float:
    if profile == "small":
        return OFFICIAL_NOTEBOOK_SMALL_LOG_PROB
    if profile == "large":
        return OFFICIAL_NOTEBOOK_LARGE_LOG_PROB
    return OFFICIAL_NOTEBOOK_UNCONSTRAINED_LOG_PROB


def local_checkpoint_present(path: str | Path | None) -> bool:
    return bool(path) and Path(path).is_file()
