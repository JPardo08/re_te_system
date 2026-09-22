"""Clean-room adapter for the Pythia model fine-tuned on SPACE-KBP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from re_te_system.contracts import ExtractionContext, RawExtractionResult


CANONICAL_MODEL_ID = "expertailab/pythia-1B-deduped-turtle-only-response-best"
CANONICAL_MODEL_REVISION = "8362933633d3584b2e572dd2c31f5190d91f8d87"
PROMPT_PROFILE_VERSION = "basic-v1"


def _basic_prompt(text: str) -> str:
    return (
        "Represent the information stated in the following text as RDF Turtle.\n\n"
        f"Text:\n{text}\n\n"
        "Turtle:\n"
    )


@dataclass(frozen=True)
class PythiaConfig:
    model_name: str = CANONICAL_MODEL_ID
    revision: str = CANONICAL_MODEL_REVISION
    tokenizer_name: str | None = None
    prompt_profile: str = "basic"
    device: str = "auto"
    dtype: str = "float32"
    max_input_tokens: int | None = None
    max_new_tokens: int = 512
    do_sample: bool = False
    seed: int = 42
    quantization: str = "none"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("Pythia requires an immutable model revision")
        if self.prompt_profile != "basic":
            raise ValueError("P0 supports only the clean-room basic prompt profile")
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive or null")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.do_sample:
            raise ValueError("P0 Pythia requires deterministic do_sample=False")
        if self.quantization != "none":
            raise ValueError("P0 supports standard loading only; quantization must be none")

    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 1,
            "num_return_sequences": 1,
            "prompt_profile": self.prompt_profile,
            "prompt_profile_version": PROMPT_PROFILE_VERSION,
            "quantization": self.quantization,
            "seed": self.seed,
            "temperature": None,
        }

    def manifest_config(self) -> dict[str, Any]:
        return asdict(self)


class PythiaSpaceKBPExtractor:
    """Generate exact decoded Turtle continuations for one input segment."""

    def __init__(self, config: PythiaConfig = PythiaConfig()) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install optional dependencies with "pip install -e .[pythia]"') from exc

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
        self.model = AutoModelForCausalLM.from_pretrained(
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
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None) or config.revision
        )
        self.model_max_length = int(self.model.config.max_position_embeddings)
        available_input = self.model_max_length - config.max_new_tokens
        if available_input <= 0:
            raise ValueError("max_new_tokens leaves no model context for input")
        self.effective_input_limit = min(
            config.max_input_tokens or available_input,
            available_input,
        )

    def _prompt(self, text: str) -> str:
        return _basic_prompt(text)

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer(
                self._prompt(text),
                add_special_tokens=True,
            ).get("input_ids", [])
        )

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        import torch

        prompt = self._prompt(text)
        untruncated_input_tokens = len(
            self.tokenizer(prompt, add_special_tokens=True).get("input_ids", [])
        )
        encoded = self.tokenizer(
            prompt,
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
                max_new_tokens=self.config.max_new_tokens,
                num_beams=1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        generated_ids = generated.sequences[0][input_token_count:]
        model_output = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_token_count = int(generated_ids.shape[-1])
        return RawExtractionResult(
            model_output=model_output,
            generation_metadata={
                "decoded_with_special_tokens": False,
                "deterministic": True,
                "effective_input_limit": self.effective_input_limit,
                "input_token_count": input_token_count,
                "max_new_tokens": self.config.max_new_tokens,
                "model_max_length": self.model_max_length,
                "number_of_sequences": int(generated.sequences.shape[0]),
                "output_reached_limit": output_token_count >= self.config.max_new_tokens,
                "output_token_count": output_token_count,
                "prompt_profile": self.config.prompt_profile,
                "prompt_profile_version": PROMPT_PROFILE_VERSION,
                "quantization": self.config.quantization,
                "truncated_input": untruncated_input_tokens > self.effective_input_limit,
                "untruncated_input_token_count": untruncated_input_tokens,
            },
        )
