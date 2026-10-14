"""Clean-room extractor for the original schema-guided UIE checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from re_te_system.conditioning.uie_schema import (
    UIESchema,
    UIE_SSI_VERSION,
)
from re_te_system.contracts import ExtractionContext, RawExtractionResult
from re_te_system.manifests.run_manifest import sha256_file


CANONICAL_UIE_MODEL_ID = "luyaojie/uie-base-en"
CANONICAL_UIE_REVISION = "966f8b1fc4c74e94ab552081605913ad5133cc41"
_WEIGHT_SUFFIXES = {".bin", ".safetensors"}


def _is_local_checkpoint(model_name: str) -> bool:
    return Path(model_name).expanduser().is_dir()


def local_uie_weight_identity(checkpoint: Path) -> dict[str, Any]:
    """Content identity of a local checkpoint. This is not a Git revision."""
    root = checkpoint.expanduser().resolve()
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in _WEIGHT_SUFFIXES
        and "optimizer" not in path.name
        and "scheduler" not in path.name
        and path.stat().st_size > 0
    ]
    config_path = root / "config.json"
    if config_path.is_file():
        files = [config_path, *files]
    if not any(path.suffix in _WEIGHT_SUFFIXES for path in files):
        raise ValueError(f"local UIE checkpoint has no weight file: {root}")
    records: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        records[relative] = {"sha256": file_hash, "size": path.stat().st_size}
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
    weights_identity = f"local-weights:{digest.hexdigest()}"
    if weights_identity == CANONICAL_UIE_REVISION:
        raise ValueError("local weight identity collided with the base revision")
    return {
        "checkpoint_path": str(root),
        "weight_files": records,
        "weights_identity": weights_identity,
    }


def read_uie_training_run(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"UIE training manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "best_metric": payload.get("best_metric"),
        "best_model_checkpoint": payload.get("best_model_checkpoint"),
        "best_model_metric": payload.get("best_model_metric"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_dir": payload.get("output_dir"),
        "slurm": payload.get("slurm"),
    }


def local_uie_provenance(config: "UIEConfig") -> dict[str, Any]:
    checkpoint = Path(config.model_name).expanduser().resolve()
    identity = local_uie_weight_identity(checkpoint)
    if config.training_manifest is None:
        raise ValueError("a local UIE checkpoint requires a training manifest")
    return {
        "base_model_id": config.base_model_id,
        "base_revision": config.base_revision,
        "kind": "local_finetuned_checkpoint",
        "training_run": read_uie_training_run(config.training_manifest),
        **identity,
    }


def resolve_uie_weight_revision(
    config: "UIEConfig",
    loaded_commit_hash: str | None,
) -> str:
    """Hub loads keep the Git SHA. Local weights never inherit the base SHA."""
    if _is_local_checkpoint(config.model_name):
        identity = local_uie_weight_identity(Path(config.model_name))["weights_identity"]
        if identity == CANONICAL_UIE_REVISION or identity == loaded_commit_hash:
            raise ValueError("refusing to identify local UIE weights with the base revision")
        return identity
    resolved = loaded_commit_hash or config.revision
    if not resolved:
        raise ValueError("UIE hub load did not resolve an immutable revision")
    return resolved


@dataclass(frozen=True)
class UIEConfig:
    model_name: str = CANONICAL_UIE_MODEL_ID
    revision: str | None = CANONICAL_UIE_REVISION
    tokenizer_name: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    max_source_tokens: int = 256
    max_target_tokens: int = 192
    do_sample: bool = False
    num_beams: int = 1
    seed: int = 42
    constraint_decoding: bool = False
    local_files_only: bool = True
    base_model_id: str = CANONICAL_UIE_MODEL_ID
    base_revision: str = CANONICAL_UIE_REVISION
    training_manifest: str | None = None

    def __post_init__(self) -> None:
        if _is_local_checkpoint(self.model_name):
            if self.revision:
                raise ValueError(
                    "a local UIE checkpoint cannot carry a Git revision; "
                    f"refusing {self.revision!r}. Record base_revision separately."
                )
            if not self.training_manifest:
                raise ValueError("a local UIE checkpoint requires training_manifest")
        elif not self.revision:
            raise ValueError("UIE requires an immutable model revision")
        if self.base_revision != CANONICAL_UIE_REVISION:
            raise ValueError("UIE base_revision is the frozen uie-base-en SHA")
        if self.base_model_id != CANONICAL_UIE_MODEL_ID:
            raise ValueError("UIE base_model_id is luyaojie/uie-base-en")
        if self.max_source_tokens <= 0 or self.max_target_tokens <= 0:
            raise ValueError("UIE source and target token limits must be positive")
        if self.do_sample:
            raise ValueError("P0 UIE requires deterministic do_sample=False")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.constraint_decoding:
            raise ValueError(
                "P0 records constrained decoding as a future optional profile only"
            )

    def generation_config(self) -> dict[str, Any]:
        return {
            "constraint_decoding": self.constraint_decoding,
            "do_sample": self.do_sample,
            "max_length": self.max_target_tokens,
            "num_beams": self.num_beams,
            "num_return_sequences": 1,
            "seed": self.seed,
            "temperature": None,
        }

    def manifest_config(self) -> dict[str, Any]:
        return asdict(self)


class UIEExtractor:
    """Generate exact decoded SEL for text conditioned by one UIE schema."""

    def __init__(
        self,
        schema: UIESchema,
        config: UIEConfig = UIEConfig(),
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'Install optional dependencies with "pip install -e .[uie]"'
            ) from exc

        dtype = getattr(torch, config.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported torch dtype: {config.dtype}")

        self.schema = schema
        self.config = config
        self.model_name = config.model_name
        self.tokenizer_name = config.tokenizer_name or config.model_name
        local_checkpoint = _is_local_checkpoint(config.model_name)
        load_options: dict[str, Any] = {"local_files_only": config.local_files_only}
        if not local_checkpoint:
            load_options["revision"] = config.revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            **load_options,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            **load_options,
        ).eval()
        if config.device == "auto":
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                selected_device = "mps"
            else:
                selected_device = "cpu"
        else:
            selected_device = config.device
        self.device = torch.device(selected_device)
        self.model.to(self.device)
        loaded_commit_hash = getattr(self.model.config, "_commit_hash", None)
        self.model_revision = resolve_uie_weight_revision(config, loaded_commit_hash)
        self.checkpoint_provenance = (
            local_uie_provenance(config) if local_checkpoint else None
        )
        if self.checkpoint_provenance is not None:
            if self.model_revision != self.checkpoint_provenance["weights_identity"]:
                raise ValueError("local UIE weight identity diverged from model_revision")
            if self.model_revision == CANONICAL_UIE_REVISION:
                raise ValueError("local UIE weights were labeled with the base revision")
        tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", 512))
        self.model_max_length = (
            tokenizer_limit if tokenizer_limit < 1_000_000 else 512
        )
        self.effective_input_limit = min(
            config.max_source_tokens,
            self.model_max_length,
        )
        self.ssi = schema.build_ssi()

    def _model_input(self, text: str) -> str:
        return self.ssi + text

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer(
                self._model_input(text),
                add_special_tokens=True,
            ).get("input_ids", [])
        )

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        import torch

        model_input = self._model_input(text)
        untruncated_input_tokens = len(
            self.tokenizer(model_input, add_special_tokens=True).get("input_ids", [])
        )
        encoded = self.tokenizer(
            model_input,
            max_length=self.effective_input_limit,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_token_count = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_length=self.config.max_target_tokens,
                num_beams=self.config.num_beams,
                num_return_sequences=1,
                return_dict_in_generate=True,
            )
        sequence = generated.sequences[0]
        model_output = self.tokenizer.decode(
            sequence,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        output_token_count = int(sequence.shape[-1])
        return RawExtractionResult(
            model_output=model_output,
            generation_metadata={
                "constraint_decoding": False,
                "decoded_with_special_tokens": True,
                "deterministic": True,
                "effective_input_limit": self.effective_input_limit,
                "input_token_count": input_token_count,
                "max_target_tokens": self.config.max_target_tokens,
                "model_max_length": self.model_max_length,
                "number_of_sequences": int(generated.sequences.shape[0]),
                "output_reached_limit": (
                    output_token_count >= self.config.max_target_tokens
                ),
                "output_token_count": output_token_count,
                "schema_hash": self.schema.stable_hash(),
                "schema_id": self.schema.schema_id,
                "ssi_version": UIE_SSI_VERSION,
                "truncated_input": (
                    untruncated_input_tokens > self.effective_input_limit
                ),
                "untruncated_input_token_count": untruncated_input_tokens,
            },
        )
