"""Optional Hugging Face adapter for the historical monolingual REBEL baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from re_te_system.contracts import ExtractionContext, RawExtractionResult


@dataclass(frozen=True)
class RebelConfig:
    model_name: str = "Babelscape/rebel-large"
    revision: str | None = None
    tokenizer_name: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    max_input_tokens: int = 256
    max_length: int = 512
    num_beams: int = 3
    do_sample: bool = False
    early_stopping: bool = False
    length_penalty: float = 0.0
    seed: int = 42
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0 or self.max_length <= 0 or self.num_beams <= 0:
            raise ValueError("token, generation, and beam limits must be positive")
        if self.do_sample:
            raise ValueError("P0.5 REBEL baseline requires deterministic do_sample=False")

    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "length_penalty": self.length_penalty,
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "num_return_sequences": 1,
            "seed": self.seed,
            "temperature": None,
        }

    def manifest_config(self) -> dict[str, Any]:
        return asdict(self)


class RebelExtractor:
    """REBEL receives source text directly, with no mREBEL language tokens."""

    def __init__(self, config: RebelConfig = RebelConfig()) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install optional dependencies with "pip install -e .[rebel]"') from exc

        dtype = getattr(torch, config.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported torch dtype: {config.dtype}")
        self.config = config
        self.model_name = config.model_name
        tokenizer_name = config.tokenizer_name or config.model_name
        load_options = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **load_options)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            **load_options,
        ).eval()
        selected_device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        self.device = torch.device(selected_device)
        self.model.to(self.device)
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or config.revision
            or "UNKNOWN_UNPINNED"
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True).get("input_ids", []))

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        import torch

        encoded = self.tokenizer(
            text,
            max_length=self.config.max_input_tokens,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                decoder_start_token_id=self.decoder_start_token_id,
                do_sample=self.config.do_sample,
                early_stopping=self.config.early_stopping,
                length_penalty=self.config.length_penalty,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                num_return_sequences=1,
                return_dict_in_generate=True,
            )
        sequence = generated.sequences[0]
        output = self.tokenizer.decode(sequence, skip_special_tokens=False)
        return RawExtractionResult(
            output,
            {
                "decoder_start_token_id": self.decoder_start_token_id,
                "deterministic": True,
                "input_token_count": int(encoded["input_ids"].shape[-1]),
                "number_of_sequences": int(generated.sequences.shape[0]),
                "output_reached_limit": int(sequence.shape[-1]) >= self.config.max_length,
                "output_token_count": int(sequence.shape[-1]),
                "truncated_input": self.count_tokens(text) > self.config.max_input_tokens,
            },
        )
