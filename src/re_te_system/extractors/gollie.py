"""Clean-room extractor for the official guideline-following GoLLIE checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from re_te_system.conditioning.gollie_schema import (
    GOLLIE_PROMPT_SERIALIZER_VERSION,
    GoLLIESchema,
)
from re_te_system.contracts import ExtractionContext, RawExtractionResult


CANONICAL_GOLLIE_MODEL_ID = "HiTZ/GoLLIE-7B"
CANONICAL_GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"
CANONICAL_GOLLIE_BASE_MODEL = "codellama/CodeLlama-7b-hf"
GOLLIE_WEIGHT_LICENSE = "llama2"
GOLLIE_CODE_LICENSE = "Apache-2.0"
GOLLIE_SCIENTIFIC_ROLE = "GUIDELINE_FOLLOWING_UIE_BASELINE"
GOLLIE_DOCUMENTED_LANGUAGE = "en"
GOLLIE_MERGED_FULL_MODEL = True
REAL_GOLLIE_NATIVE_SMOKE = "BLOCKED_GPU"


@dataclass(frozen=True)
class GoLLIEConfig:
    model_name: str = CANONICAL_GOLLIE_MODEL_ID
    revision: str = CANONICAL_GOLLIE_REVISION
    tokenizer_name: str | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    do_sample: bool = False
    num_beams: int = 1
    seed: int = 42
    quantization: str = "none"
    flash_attention: bool = True
    local_files_only: bool = True
    weight_snapshot: str | None = None

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("GoLLIE requires an immutable model revision")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.do_sample:
            raise ValueError("P0 GoLLIE requires deterministic do_sample=False")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.quantization != "none":
            raise ValueError("P0 supports standard loading only; quantization must be none")
        if not self.flash_attention:
            raise ValueError("P0 GoLLIE requires FlashAttention; no alternative backend")
        if self.device not in {"auto", "cuda"}:
            raise ValueError("P0 GoLLIE supports only CUDA; CPU/MPS are not implemented")

    def generation_config(self) -> dict[str, Any]:
        return {
            "constraint_decoding": False,
            "do_sample": self.do_sample,
            "flash_attention": self.flash_attention,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 0,
            "num_beams": self.num_beams,
            "num_return_sequences": 1,
            "quantization": self.quantization,
            "seed": self.seed,
            "temperature": None,
        }

    def manifest_config(self) -> dict[str, Any]:
        return asdict(self)


def causal_effective_input_limit(model_max_length: int, max_new_tokens: int) -> int:
    """Reserve generation tokens inside the decoder-only context window."""
    if max_new_tokens >= model_max_length:
        raise ValueError(
            "max_new_tokens must be smaller than model_max_length so the "
            "decoder-only context budget leaves room for the prompt"
        )
    limit = model_max_length - max_new_tokens
    if limit <= 0:
        raise ValueError("effective_input_limit must be positive")
    return limit


def assert_safetensors_snapshot(snapshot: Path) -> None:
    """Refuse a GoLLIE directory that could fall back to pytorch bins."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("GoLLIE safetensors snapshot is not a real directory")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("GoLLIE safetensors index is absent")
    for candidate in snapshot.rglob("*"):
        if candidate.name.endswith(".bin") or candidate.name == "pytorch_model.bin.index.json":
            raise ValueError(f"refusing pytorch weights in the GoLLIE snapshot: {candidate.name}")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("GoLLIE safetensors index is malformed") from exc
    mapping = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("GoLLIE safetensors index is empty")
    for name in mapping.values():
        if not isinstance(name, str) or not name.endswith(".safetensors") or name != Path(name).name:
            raise ValueError("GoLLIE safetensors index is malformed")
        weight = snapshot / name
        if weight.is_symlink() or not weight.is_file() or weight.stat().st_size == 0:
            raise ValueError(f"GoLLIE safetensors weight is absent: {name}")
    config = snapshot / "config.json"
    if config.is_symlink() or not config.is_file() or config.stat().st_size == 0:
        raise ValueError("GoLLIE config.json is absent")


def resolve_gollie_weights(config: GoLLIEConfig) -> tuple[str, dict[str, object]]:
    """Load only a verified local Safetensors snapshot.

    The scientific identity stays HiTZ/GoLLIE-7B at the pinned revision. The
    hub repository is not a load path: Transformers 4.57.6 refuses its .bin
    shards on torch 2.5.1, and this call does not fall back to them.
    """
    if config.model_name != CANONICAL_GOLLIE_MODEL_ID or config.revision != CANONICAL_GOLLIE_REVISION:
        raise ValueError("GoLLIE scientific identity is HiTZ/GoLLIE-7B at the pinned revision")
    if not config.local_files_only:
        raise ValueError("GoLLIE refuses a network weight load")
    snapshot = (config.weight_snapshot or "").strip()
    if not snapshot:
        raise ValueError("GoLLIE requires the verified safetensors snapshot; refusing pytorch bins")
    path = Path(snapshot)
    assert_safetensors_snapshot(path)
    return str(path), {
        "local_files_only": True,
        "trust_remote_code": True,
        "use_safetensors": True,
    }


def _require_cuda_flash_attention() -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError('Install optional dependencies with "pip install -e .[gollie]"') from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GoLLIE-7B requires an NVIDIA CUDA GPU and official FlashAttention. "
            "CPU and MPS fallbacks are not implemented in P0."
        )
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GoLLIE-7B requires FlashAttention. Alternative attention backends "
            "are not implemented in P0."
        ) from exc


class GoLLIEExtractor:
    """Generate exact decoded GoLLIE continuations for one schema-conditioned text."""

    def __init__(
        self,
        schema: GoLLIESchema,
        config: GoLLIEConfig = GoLLIEConfig(),
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'Install optional dependencies with "pip install -e .[gollie]"'
            ) from exc

        _require_cuda_flash_attention()
        dtype = getattr(torch, config.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported torch dtype: {config.dtype}")

        self.schema = schema
        self.config = config
        self.model_name = config.model_name
        self.tokenizer_name = config.tokenizer_name or config.model_name
        source, load_options = resolve_gollie_weights(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            add_eos_token=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=dtype,
            **load_options,
        ).eval()
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model_revision = config.revision
        tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", 16384))
        self.model_max_length = tokenizer_limit if tokenizer_limit < 1_000_000 else 16384
        self.effective_input_limit = causal_effective_input_limit(
            self.model_max_length,
            config.max_new_tokens,
        )

    def _prompt(self, text: str) -> str:
        return self.schema.serialize_prompt(text)

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(self._prompt(text), add_special_tokens=True)
        ids = list(encoded.get("input_ids", []))
        if ids and ids[-1] == getattr(self.tokenizer, "eos_token_id", None):
            ids = ids[:-1]
        return len(ids)

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        import torch

        prompt = self._prompt(text)
        untruncated = self.tokenizer(prompt, add_special_tokens=True)
        input_ids = list(untruncated.get("input_ids", []))
        if input_ids and input_ids[-1] == getattr(self.tokenizer, "eos_token_id", None):
            input_ids = input_ids[:-1]
        untruncated_input_tokens = len(input_ids)
        truncated_input = untruncated_input_tokens > self.effective_input_limit
        if truncated_input:
            input_ids = input_ids[: self.effective_input_limit]
        tensor = torch.tensor([input_ids], device=self.device)
        attention = torch.ones_like(tensor)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=tensor,
                attention_mask=attention,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=0,
                num_beams=self.config.num_beams,
                num_return_sequences=1,
                return_dict_in_generate=True,
            )
        sequence = generated.sequences[0]
        continuation = sequence[tensor.shape[-1] :]
        model_output = self.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_token_count = int(continuation.shape[-1])
        return RawExtractionResult(
            model_output=model_output,
            generation_metadata={
                "base_model": CANONICAL_GOLLIE_BASE_MODEL,
                "constraint_decoding": False,
                "custom_modeling": True,
                "deterministic": True,
                "documented_language": GOLLIE_DOCUMENTED_LANGUAGE,
                "merged_full_model": GOLLIE_MERGED_FULL_MODEL,
                "scientific_role": GOLLIE_SCIENTIFIC_ROLE,
                "weight_license": GOLLIE_WEIGHT_LICENSE,
                "effective_input_limit": self.effective_input_limit,
                "exact_prompt": prompt,
                "flash_attention": True,
                "flash_attention_required": True,
                "guideline_hash": self.schema.guideline_hash(),
                "input_token_count": int(tensor.shape[-1]),
                "max_new_tokens": self.config.max_new_tokens,
                "model_max_length": self.model_max_length,
                "number_of_sequences": int(generated.sequences.shape[0]),
                "output_reached_limit": output_token_count >= self.config.max_new_tokens,
                "output_token_count": output_token_count,
                "prompt_serializer_version": GOLLIE_PROMPT_SERIALIZER_VERSION,
                "quantization": self.config.quantization,
                "schema_hash": self.schema.schema_hash(),
                "schema_id": self.schema.schema_id,
                "truncated_input": truncated_input,
                "untruncated_input_token_count": untruncated_input_tokens,
            },
        )
