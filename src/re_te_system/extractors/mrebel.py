"""Optional Hugging Face mREBEL C0_OPEN extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from re_te_system.contracts import ExtractionContext, RawExtractionResult


LEGACY_REL_ALLOWED = frozenset(
    {
        "subclass_of",
        "part_of",
        "instance_of",
        "facet_of",
        "different_from",
        "has_cause",
        "field_of_work",
    }
)


@dataclass(frozen=True)
class MRebelConfig:
    model_name: str = "Babelscape/mrebel-large"
    revision: str | None = None
    tokenizer_name: str | None = None
    src_lang: str = "es_XX"
    target_token: str = "tp_XX"
    device: str = "auto"
    dtype: str = "float32"
    max_input_tokens: int = 256
    max_new_tokens: int = 512
    num_beams: int = 3
    do_sample: bool = False
    early_stopping: bool = True
    temperature: float | None = None
    seed: int = 42
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.src_lang != "es_XX":
            raise ValueError("Paper 3 P0 Spanish configuration requires src_lang=es_XX")
        if self.target_token != "tp_XX":
            raise ValueError("mREBEL relation extraction requires target_token=tp_XX")
        if self.max_input_tokens <= 0 or self.max_new_tokens <= 0 or self.num_beams <= 0:
            raise ValueError("token and beam limits must be positive")
        if not self.do_sample and self.temperature is not None:
            raise ValueError("temperature is inactive when do_sample=False; use null")

    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "num_return_sequences": 1,
            "seed": self.seed,
            "temperature": self.temperature,
        }

    def manifest_config(self) -> dict[str, Any]:
        return asdict(self)


class MRebelExtractor:
    """Loads ML dependencies lazily and receives text plus runtime context only."""

    def __init__(self, config: MRebelConfig = MRebelConfig()) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install optional dependencies with "pip install -e .[mrebel]"') from exc

        self.config = config
        self.model_name = config.model_name
        tokenizer_name = config.tokenizer_name or config.model_name
        dtype = getattr(torch, config.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported torch dtype: {config.dtype}")
        load_options = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            src_lang=config.src_lang,
            tgt_lang=config.target_token,
            **load_options,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            **load_options,
        ).eval()
        if config.device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = config.device
        self.device = torch.device(selected_device)
        self.model.to(self.device)
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(config.target_token)
        if self.decoder_start_token_id in (None, self.tokenizer.unk_token_id):
            raise ValueError(f"tokenizer does not support target token {config.target_token}")
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
        options = {
            key: value
            for key, value in self.config.generation_config().items()
            if key not in {"seed", "temperature"} and value is not None
        }
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                decoder_start_token_id=self.decoder_start_token_id,
                return_dict_in_generate=True,
                **options,
            )
        sequence = generated.sequences[0]
        output = self.tokenizer.decode(sequence, skip_special_tokens=False)
        return RawExtractionResult(
            output,
            {
                "decoder_start_token_id": self.decoder_start_token_id,
                "deterministic": not self.config.do_sample,
                "input_token_count": int(encoded["input_ids"].shape[-1]),
                "number_of_sequences": int(generated.sequences.shape[0]),
                "output_token_count": int(sequence.shape[-1]),
                "output_reached_limit": int(sequence.shape[-1])
                >= self.config.max_new_tokens + 1,
                "truncated_input": self.count_tokens(text) > self.config.max_input_tokens,
            },
        )
