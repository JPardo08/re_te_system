"""Controlled runtime contract for Qwen/Qwen3-4B-Instruct-2507.

This module does not download weights, load a checkpoint, or authorize a
campaign mode. ``authorize_campaign`` still accepts only
``SYNTHETIC_CONTRACT_TEST``. A test double may exercise serialization and
state transitions. That double is not a Qwen3 measurement.

Verified upstream facts are constants below. Unapproved scientific choices
stay ``None`` and are readiness blockers. The native context length declared
by the model card is not copied into ``effective_context_limit``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from typing import Any, Callable, Mapping, Sequence

from re_te_system.contracts import ExtractionContext, RawExtractionResult
from re_te_system.controlled_experiment.campaign import (
    SyntheticDocument,
    _completed_attempt,
    render_controlled_prompt,
)
from re_te_system.controlled_experiment.contracts import (
    RETRYABLE_STATES,
    ExternalInfrastructureError,
    sha256_payload,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import SYSTEM_REPO
from re_te_system.controlled_experiment.preflight import PreflightRequest, assess_preflight


QWEN3_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN3_OBSERVED_DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256 = "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"
QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH = 262144
QWEN3_SERIALIZATION_ID = "qwen3-instruct-2507-single-user-add-generation-prompt"
QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"
QWEN3_MODEL_TYPE = "qwen3"
QWEN3_DECLARED_TORCH_DTYPE = "bfloat16"
QWEN3_CONFIG_TRANSFORMERS_VERSION = "4.51.0"
QWEN3_EOS_TOKEN = "<|im_end|>"
QWEN3_PAD_TOKEN = "<|endoftext|>"
QWEN3_CONTRACT_TEST = "QWEN3_CONTRACT_TEST"
QWEN3_OFFICIAL_COUNTER_ID = "qwen3-official-tokenizer"
QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH = 1010000
QWEN3_OBSERVED_TOKENIZER_CLASS = "Qwen2TokenizerFast"
# Measured from the official tokenizer at the candidate revision. Not a scientific pin.
QWEN3_SPECIAL_TOKEN_IDS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
}
# config.json and generation_config.json do not use these ids in the same roles.
QWEN3_SPECIAL_TOKEN_MISMATCHES = (
    {
        "config_pad_token_id": None,
        "generation_bos_token_id": 151643,
        "generation_eos_token_id": [151645, 151643],
        "generation_pad_token_id": 151643,
        "model_bos_token_id": 151643,
        "model_eos_token_id": 151645,
        "note": (
            "151645 is <|im_end|> and the tokenizer EOS. 151643 is <|endoftext|> and the "
            "tokenizer pad. config.json names 151643 as bos_token_id and has no pad_token_id "
            "field. generation_config.json uses 151643 as bos, pad, and a second EOS id. "
            "Those roles are not interchangeable."
        ),
    },
)
# Preferred initial scientific candidate. Not frozen and not written into contract defaults.
QWEN3_DECODING_RECOMMENDATION = {
    "approved": False,
    "candidate": "greedy",
    "do_sample": False,
    "reason": (
        "C0 and C1 must share one decoding policy. The published generation_config.json "
        "samples with do_sample true, temperature 0.7, top_p 0.8, and top_k 20. Sampling "
        "makes repeated runs differ. Greedy decoding is the preferred first candidate. "
        "max_new_tokens, device, dtype, and the effective context limit stay unapproved."
    ),
}
# Controlled stop policy. The upstream EOS list [151645, 151643] is not used.
# 151645 stops generation. 151643 remains pad and is not a stop token.
QWEN3_EOS_POLICY_ID = "stop_on_im_end_only"
QWEN3_CONTROLLED_EOS_TOKEN_IDS = (151645,)
QWEN3_CONTROLLED_PAD_TOKEN_ID = 151643
QWEN3_TRANSPORT_POLICY_ID = "strip_trailing_im_end_or_endoftext"
QWEN3_TRANSPORT_TOKEN_IDS = frozenset({151645, 151643})
QWEN3_REAL_MODE = "QWEN3_REAL_RUNTIME_CONTRACT"
QWEN3_RESULT_ROLE = "qwen3_real_runtime_not_scientific_f1"
QWEN3_RUNTIME_STATUS = "REAL_RUNTIME_IMPLEMENTED_NOT_EXECUTED"
# Phase lock. The model constructor below stays unreachable until a later
# explicit change. A grant, a path, or an environment variable cannot open it.
QWEN3_WEIGHT_EXECUTION_PHASE_LOCK = True
# sha256 and byte size of the allowlisted files at the candidate revision.
QWEN3_TOKENIZER_ASSETS = {
    "config.json": ("5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba", 727),
    "generation_config.json": ("835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48", 238),
    "merges.txt": ("599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3", 1671839),
    "tokenizer.json": ("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", 11422654),
    "tokenizer_config.json": ("a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3", 9377),
    "vocab.json": ("ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2776833),
}
_ASSET_INSPECTION_CACHE: dict[tuple[object, ...], dict[str, Any]] = {}

# generation_config.json at the observed default revision. Not an approved campaign setting.
QWEN3_UPSTREAM_GENERATION_CONFIG = {
    "approved_for_campaign": False,
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "source": "generation_config.json",
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
}
# Model-card recommendation. Not copied into the campaign contract.
QWEN3_UPSTREAM_RECOMMENDED_MAX_NEW_TOKENS = 16384


class Qwen3WeightsNotAuthorized(RuntimeError):
    """Weight download and model load are outside this phase."""


class Qwen3TokenizerAssetsError(RuntimeError):
    """The local tokenizer directory is missing, incomplete, or not the candidate revision."""


class Qwen3ConfigurationError(RuntimeError):
    """A real-run contract is missing an explicit gate or selects a rejected policy."""


class Qwen3ExecutionNotAuthorized(RuntimeError):
    """Generation was requested without a test grant. Weights stay locked."""


class Qwen3WorkerTimeout(TimeoutError):
    """The inference worker exceeded its budget and the process group was stopped."""

    def __init__(self, timeout_seconds: float, pid: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.pid = pid
        super().__init__(f"inference worker {pid} exceeded {timeout_seconds}s and was terminated")


class Qwen3WorkerFailed(RuntimeError):
    """The inference worker exited without a generation result."""


@dataclass(frozen=True)
class Qwen3GenerationContract:
    """Declarative generation settings. ``None`` means the choice is not approved.

    ``test_only`` never promotes these values to a scientific freeze. Upstream
    sampling defaults are recorded separately and are not the field defaults.
    """

    model_id: str = QWEN3_MODEL_ID
    observed_default_revision: str = QWEN3_OBSERVED_DEFAULT_REVISION
    pinned_checkpoint_revision: str | None = None
    tokenizer_id: str = QWEN3_MODEL_ID
    pinned_tokenizer_revision: str | None = None
    chat_template_sha256: str = QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256
    serialization_id: str = QWEN3_SERIALIZATION_ID
    declared_native_context_length: int = QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH
    effective_context_limit: int | None = None
    reserved_output_tokens: int | None = None
    special_token_allowance: int = 0
    max_new_tokens: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    dtype_policy: str | None = None
    device_policy: str | None = None
    timeout_seconds: float | None = None
    test_only: bool = False
    eos_policy_id: str | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.do_sample is True:
            raise ValueError("sampling is not authorized for this runtime")
        if self.do_sample is False and (
            self.temperature is not None or self.top_p is not None or self.top_k is not None
        ):
            raise ValueError("greedy decoding does not accept sampling parameters")
        if self.eos_policy_id is not None and self.eos_policy_id != QWEN3_EOS_POLICY_ID:
            raise ValueError("eos_policy_id is not the controlled stop policy")
        if self.pad_token_id is not None and self.pad_token_id != QWEN3_CONTROLLED_PAD_TOKEN_ID:
            raise ValueError("pad_token_id is not the verified pad id")

    def blockers(self) -> tuple[str, ...]:
        blocked = [
            _revision_blocker(self.pinned_checkpoint_revision, "pinned_checkpoint_revision"),
            _revision_blocker(self.pinned_tokenizer_revision, "pinned_tokenizer_revision"),
            "tokenizer_assets_not_local" if not tokenizer_assets_verified() else None,
            "effective_context_limit_unapproved" if self.effective_context_limit is None else None,
            "reserved_output_tokens_unapproved" if self.reserved_output_tokens is None else None,
            "max_new_tokens_unapproved" if self.max_new_tokens is None else None,
            "decoding_unapproved" if self.do_sample is None else None,
            "eos_policy_unapproved" if self.eos_policy_id is None else None,
            "pad_policy_unapproved" if self.pad_token_id is None else None,
            "seed_unapproved" if self.seed is None else None,
            "dtype_policy_unapproved" if self.dtype_policy is None else None,
            "device_policy_unapproved" if self.device_policy is None else None,
            "process_timeout_budget_unapproved" if self.timeout_seconds is None else None,
            (
                "reserved_output_smaller_than_max_new_tokens"
                if (
                    self.reserved_output_tokens is not None
                    and self.max_new_tokens is not None
                    and self.reserved_output_tokens < self.max_new_tokens
                )
                else None
            ),
            "weights_not_authorized",
        ]
        if self.test_only:
            blocked.append("test_only_configuration")
        return tuple(item for item in blocked if item is not None)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "chat_template_sha256": self.chat_template_sha256,
            "declared_native_context_length": self.declared_native_context_length,
            "device_policy": self.device_policy,
            "do_sample": self.do_sample,
            "dtype_policy": self.dtype_policy,
            "bitwise_determinism": False,
            "effective_context_limit": self.effective_context_limit,
            "eos_policy_id": self.eos_policy_id,
            "max_new_tokens": self.max_new_tokens,
            "model_id": self.model_id,
            "observed_default_revision": self.observed_default_revision,
            "pad_token_id": self.pad_token_id,
            "pinned_checkpoint_revision": self.pinned_checkpoint_revision,
            "pinned_tokenizer_revision": self.pinned_tokenizer_revision,
            "process_timeout": (
                "subprocess_process_group" if self.timeout_seconds is not None else "budget_unapproved"
            ),
            "reserved_output_tokens": self.reserved_output_tokens,
            "seed": self.seed,
            "serialization_id": self.serialization_id,
            "special_token_allowance": self.special_token_allowance,
            "temperature": self.temperature,
            "test_only": self.test_only,
            "timeout_seconds": self.timeout_seconds,
            "tokenizer_id": self.tokenizer_id,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "upstream_generation_config_approved": False,
        }

    def identity_sha256(self) -> str:
        return sha256_payload(self.identity_payload())


def upstream_facts() -> dict[str, Any]:
    """Facts read from the Hub on 2026-10-21, separated from unapproved choices."""
    return {
        "approved_effective_context_limit": None,
        "approved_max_new_tokens": None,
        "architecture": QWEN3_ARCHITECTURE,
        "bos_token": None,
        "chat_template_contains_think_tag": False,
        "chat_template_sha256": QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256,
        "config_transformers_version": QWEN3_CONFIG_TRANSFORMERS_VERSION,
        "declared_native_context_length": QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH,
        "declared_torch_dtype": QWEN3_DECLARED_TORCH_DTYPE,
        "eos_token": QWEN3_EOS_TOKEN,
        "license": "apache-2.0",
        "loader": "transformers.AutoModelForCausalLM",
        "model_id": QWEN3_MODEL_ID,
        "model_type": QWEN3_MODEL_TYPE,
        "non_thinking_mode_declared": True,
        "observed_default_revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "pad_token": QWEN3_PAD_TOKEN,
        "pinned_checkpoint_revision": None,
        "pinned_tokenizer_revision": None,
        "rope_scaling": None,
        "controlled_eos_policy": QWEN3_EOS_POLICY_ID,
        "controlled_eos_token_ids": list(QWEN3_CONTROLLED_EOS_TOKEN_IDS),
        "controlled_pad_token_id": QWEN3_CONTROLLED_PAD_TOKEN_ID,
        "decoding_recommendation": dict(QWEN3_DECODING_RECOMMENDATION),
        "parser_transport_policy": QWEN3_TRANSPORT_POLICY_ID,
        "process_timeout": "subprocess_process_group",
        "upstream_eos_token_ids_not_adopted": [151645, 151643],
        "special_token_mismatches": list(QWEN3_SPECIAL_TOKEN_MISMATCHES),
        "tokenizer_assets_local": tokenizer_assets_verified(),
        "tokenizer_class": QWEN3_OBSERVED_TOKENIZER_CLASS,
        "tokenizer_declared_model_max_length": QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH,
        "tokenizer_validation_revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "token_id_to_string_mapping": dict(QWEN3_SPECIAL_TOKEN_IDS),
        "upstream_generation_config": dict(QWEN3_UPSTREAM_GENERATION_CONFIG),
        "upstream_recommended_max_new_tokens": QWEN3_UPSTREAM_RECOMMENDED_MAX_NEW_TOKENS,
        "weights_authorized": False,
    }


def readiness(contract: Qwen3GenerationContract | None = None) -> dict[str, Any]:
    """Local implementation status. This is not real-model validation."""
    selected = contract or Qwen3GenerationContract()
    return {
        "blockers": list(selected.blockers()),
        "executable": False,
        "ready_for_cesvima": False,
        "ready_for_gold": False,
        "real_model_validated": False,
        "real_runtime_executed": False,
        "real_runtime_implemented": True,
        "status": QWEN3_RUNTIME_STATUS,
        "weights_authorized": False,
        "weight_execution_phase_lock": QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
    }


def serialize_rendered_prompt(rendered_prompt: str) -> str:
    """Return the checked single-user reconstruction.

    For no tools and one user message with ``add_generation_prompt=True``,
    the official template emits this exact string. The official tokenizer
    remains the authority: ``official_chat_serialization`` is what a loaded
    tokenizer uses. This reconstruction is retained because it matched that
    call on the candidate revision. It is not a second template.
    """
    if not isinstance(rendered_prompt, str) or rendered_prompt == "":
        raise ValueError("rendered prompt must be a non-empty string")
    return "<|im_start|>user\n" + rendered_prompt + "<|im_end|>\n<|im_start|>assistant\n"


def tokenizer_directory() -> Path:
    """Project-local cache for the candidate revision. The directory is gitignored."""
    return (
        SYSTEM_REPO
        / ".cache"
        / "huggingface"
        / "qwen3-tokenizer"
        / QWEN3_OBSERVED_DEFAULT_REVISION
    )


def inspect_tokenizer_assets(directory: Path | None = None) -> dict[str, Any]:
    """Hash the allowlisted files. This does not download or load a model."""
    selected = Path(directory) if directory is not None else tokenizer_directory()
    signature = _asset_signature(selected)
    cached = _ASSET_INSPECTION_CACHE.get(signature)
    if cached is not None:
        return cached
    files: list[dict[str, Any]] = []
    for name, (expected_sha256, expected_size) in QWEN3_TOKENIZER_ASSETS.items():
        path = selected / name
        if not path.is_file():
            files.append({"filename": name, "present": False, "sha256_matches": False})
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        files.append(
            {
                "byte_size": len(data),
                "expected_byte_size": expected_size,
                "expected_sha256": expected_sha256,
                "filename": name,
                "present": True,
                "sha256": digest,
                "sha256_matches": digest == expected_sha256 and len(data) == expected_size,
            }
        )
    weight_files = sorted(path.name for path in selected.glob("*.safetensors"))
    verified = all(item.get("sha256_matches") is True for item in files) and not weight_files
    record = {
        "directory": str(selected),
        "files": files,
        "revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "revision_role": "technical_validation_not_scientific_pin",
        "verified": verified,
        "weight_files": weight_files,
    }
    _ASSET_INSPECTION_CACHE[signature] = record
    return record


def tokenizer_assets_verified(directory: Path | None = None) -> bool:
    return bool(inspect_tokenizer_assets(directory)["verified"])


def load_qwen3_tokenizer(directory: Path | None = None) -> Any:
    """Load ``AutoTokenizer`` from a verified local directory.

    ``local_files_only=True`` prevents a Hub fetch. Weight files in the same
    directory are refused. The returned object is a tokenizer, not a model.
    """
    selected = Path(directory) if directory is not None else tokenizer_directory()
    record = inspect_tokenizer_assets(selected)
    if record["weight_files"]:
        raise Qwen3TokenizerAssetsError("tokenizer directory contains model weights")
    if not record["verified"]:
        missing = [item["filename"] for item in record["files"] if item.get("sha256_matches") is not True]
        raise Qwen3TokenizerAssetsError("tokenizer assets are missing or do not match the candidate revision: " + ",".join(missing))
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise Qwen3TokenizerAssetsError("transformers is not installed") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(selected), local_files_only=True)
    _check_loaded_tokenizer(tokenizer)
    return tokenizer


def official_chat_serialization(tokenizer: Any, rendered_prompt: str) -> str:
    """Render one user turn with the loaded tokenizer's chat template."""
    if not isinstance(rendered_prompt, str) or rendered_prompt == "":
        raise ValueError("rendered prompt must be a non-empty string")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": rendered_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or rendered == "":
        raise Qwen3TokenizerAssetsError("apply_chat_template did not return text")
    return rendered


def official_input_token_ids(tokenizer: Any, serialized: str) -> list[int]:
    """Tokenize the exact serialized string that would be sent to generation.

    ``add_special_tokens=True`` must not change the ids. A change would mean
    a control token was added or removed after the chat template.
    """
    plain = list(tokenizer.encode(serialized, add_special_tokens=False))
    marked = list(tokenizer.encode(serialized, add_special_tokens=True))
    if plain != marked:
        raise Qwen3TokenizerAssetsError("special-token handling changed the serialized input ids")
    return plain


def trace_continuation(
    tokenizer: Any,
    input_ids: Sequence[int],
    sequence_ids: Sequence[int],
) -> dict[str, Any]:
    """Separate a continuation from its input ids without calling a model.

    The parser input is the raw decode. Special tokens are not stripped and
    the text is not rewritten.
    """
    prefix = [int(item) for item in input_ids]
    sequence = [int(item) for item in sequence_ids]
    if sequence[: len(prefix)] != prefix:
        raise ValueError("sequence_ids does not start with the serialized input ids")
    continuation = sequence[len(prefix) :]
    decoded = tokenizer.decode(continuation, skip_special_tokens=False)
    if not isinstance(decoded, str):
        raise Qwen3TokenizerAssetsError("decoded continuation must be a string")
    return {
        "continuation_ids": continuation,
        "decoded_raw_output": decoded,
        "input_token_count": len(prefix),
        "parser_input": decoded,
        "skip_special_tokens": False,
    }


def _asset_signature(directory: Path) -> tuple[object, ...]:
    parts: list[object] = [str(directory)]
    for name in QWEN3_TOKENIZER_ASSETS:
        path = directory / name
        if not path.is_file():
            parts.append((name, None))
            continue
        stat = path.stat()
        parts.append((name, stat.st_size, stat.st_mtime_ns))
    weight_names = tuple(sorted(path.name for path in directory.glob("*.safetensors")))
    parts.append(weight_names)
    return tuple(parts)


def _check_loaded_tokenizer(tokenizer: Any) -> None:
    if type(tokenizer).__name__ != QWEN3_OBSERVED_TOKENIZER_CLASS:
        raise Qwen3TokenizerAssetsError("loaded tokenizer class is not the verified Qwen class")
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise Qwen3TokenizerAssetsError("loaded tokenizer has no chat template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256:
        raise Qwen3TokenizerAssetsError("loaded chat template does not match the verified hash")
    if "<think>" in template or "enable_thinking" in template:
        raise Qwen3TokenizerAssetsError("loaded chat template contains a thinking control")
    for token, expected in QWEN3_SPECIAL_TOKEN_IDS.items():
        if int(tokenizer.convert_tokens_to_ids(token)) != expected:
            raise Qwen3TokenizerAssetsError("special-token id does not match the verified mapping")
    if tokenizer.eos_token != QWEN3_EOS_TOKEN or int(tokenizer.eos_token_id) != QWEN3_SPECIAL_TOKEN_IDS["<|im_end|>"]:
        raise Qwen3TokenizerAssetsError("tokenizer EOS is not <|im_end|>")
    if tokenizer.pad_token != QWEN3_PAD_TOKEN or int(tokenizer.pad_token_id) != QWEN3_SPECIAL_TOKEN_IDS["<|endoftext|>"]:
        raise Qwen3TokenizerAssetsError("tokenizer pad is not <|endoftext|>")
    if tokenizer.bos_token is not None:
        raise Qwen3TokenizerAssetsError("tokenizer BOS is set; the verified tokenizer has no BOS token")
    if int(tokenizer.model_max_length) != QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH:
        raise Qwen3TokenizerAssetsError("tokenizer model_max_length changed from the verified value")


def load_qwen3_weights(*_args: object, **_kwargs: object) -> None:
    """Refuse to construct a model. No Hub download is attempted."""
    raise Qwen3WeightsNotAuthorized(
        "Loading Qwen/Qwen3-4B-Instruct-2507 is not authorized. "
        "No weights or tokenizer assets are downloaded by this function."
    )


def observe_installed_libraries() -> dict[str, Any]:
    """Report installed library versions. This does not load a checkpoint."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        return {"available": False, "error": type(exc).__name__, "weights_loaded": False}
    return {
        "available": True,
        "qwen3_module_present": importlib.util.find_spec("transformers.models.qwen3") is not None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "weights_loaded": False,
    }


class Qwen3Extractor:
    """Extractor-shaped refusal. The synthetic campaign must not call this."""

    model_name = QWEN3_MODEL_ID
    model_revision = "unpinned"

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        del text, context
        raise Qwen3WeightsNotAuthorized(
            "Qwen3Extractor.extract is not a campaign execution path. "
            "SYNTHETIC_CONTRACT_TEST remains the only authorized mode."
        )


@dataclass(frozen=True)
class PreparedQwenInput:
    condition: str
    rendered_prompt: str
    rendered_prompt_sha256: str
    serialized_input: str
    serialized_input_sha256: str
    preflight: dict[str, Any]
    serialization_matches_reconstruction: bool | None = None
    serialization_source: str = "reconstructed_single_user_branch"


@dataclass(frozen=True)
class Qwen3AttemptOutcome:
    attempts: tuple[dict[str, Any], ...]
    condition: str
    contract_blockers: tuple[str, ...]
    contract_sha256: str
    eligible_for_export: bool
    final_principal_state: str | None
    generator_calls: int
    harness_incident: str | None
    preflight: dict[str, Any]
    raw_output: str | None
    rendered_prompt: str
    serialized_input: str
    test_only: bool
    validated_triples: tuple[tuple[str, str, str], ...]


class Qwen3Runtime:
    """Serialize, measure, and optionally accept an injected continuation.

    ``complete`` receives the serialized input and the attempt index. It must
    return the raw continuation and may raise ``ExternalInfrastructureError``,
    ``TimeoutError``, or another exception. A ``TimeoutError`` is classified
    as ``timeout`` but does not mean a process limit was enforced.
    """

    def __init__(
        self,
        contract: Qwen3GenerationContract,
        *,
        count_tokens: Callable[[str], int] | None = None,
        counter_id: str | None = None,
        complete: Callable[[str, int], Mapping[str, Any]] | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.contract = contract
        self.count_tokens = count_tokens
        self.counter_id = counter_id
        self.complete = complete
        self.tokenizer = tokenizer
        self.serialized_calls: list[str] = []

    def prepare(self, condition: str, document_text: str) -> PreparedQwenInput:
        rendered = render_controlled_prompt(condition, document_text)
        if self.tokenizer is None:
            serialized = serialize_rendered_prompt(rendered)
            source = "reconstructed_single_user_branch"
            matches = None
        else:
            serialized = official_chat_serialization(self.tokenizer, rendered)
            source = "official_apply_chat_template"
            matches = serialized == serialize_rendered_prompt(rendered)
        preflight = self._preflight(serialized)
        return PreparedQwenInput(
            condition=condition,
            rendered_prompt=rendered,
            rendered_prompt_sha256=sha256_text(rendered),
            serialized_input=serialized,
            serialized_input_sha256=sha256_text(serialized),
            preflight=preflight,
            serialization_matches_reconstruction=matches,
            serialization_source=source,
        )

    def run_attempt(self, condition: str, document_text: str) -> Qwen3AttemptOutcome:
        prepared = self.prepare(condition, document_text)
        document = SyntheticDocument(f"{condition}-document", document_text)
        attempts: list[dict[str, Any]] = []
        validated: tuple[tuple[str, str, str], ...] = ()
        raw_output: str | None = None
        incident: str | None = None
        calls = 0
        if prepared.preflight.get("assurance") == "not_assessed":
            incident = "preflight_not_assessed"
        else:
            for attempt_index in (1, 2):
                if prepared.preflight.get("fits") is not True:
                    attempts.append(self._blocked_attempt(prepared, document, attempt_index))
                    break
                if not self.contract.test_only or self.complete is None:
                    incident = "generator_not_authorized"
                    break
                calls += 1
                self.serialized_calls.append(prepared.serialized_input)
                try:
                    payload = self.complete(prepared.serialized_input, attempt_index)
                except ExternalInfrastructureError as exc:
                    attempts.append(self._failure_attempt(prepared, document, attempt_index, "external_infrastructure_error", exc))
                except TimeoutError as exc:
                    attempts.append(self._failure_attempt(prepared, document, attempt_index, "timeout", exc))
                except Exception as exc:
                    attempts.append(self._failure_attempt(prepared, document, attempt_index, "model_execution_error", exc))
                else:
                    if not isinstance(payload, Mapping) or not isinstance(payload.get("text"), str):
                        incident = "generator_contract_violation"
                        break
                    raw_output = payload["text"]
                    metadata = {
                        "completion_reason_source": "injected_test_double",
                        "do_sample": self.contract.do_sample,
                        "prediction_source": "qwen3_test_double_not_model",
                        "seed": self.contract.seed,
                        "timeout_enforcement": "not_implemented",
                    }
                    if payload.get("output_reached_limit") is True:
                        metadata["output_reached_limit"] = True
                    if payload.get("truncated_input") is True:
                        metadata["truncated_input"] = True
                    common = self._common(prepared, document, attempt_index)
                    attempt, validation = _completed_attempt(raw_output, metadata, common)
                    attempts.append(attempt.to_dict())
                    validated = tuple(
                        (triple.subject, triple.relation, triple.object) for triple in validation.triples
                    )
                if not attempts or attempts[-1].get("principal_state") not in RETRYABLE_STATES or attempt_index == 2:
                    break
        final = attempts[-1]["principal_state"] if attempts else None
        diagnostics = attempts[-1].get("diagnostics") if attempts else None
        truncated = isinstance(diagnostics, dict) and diagnostics.get("truncated_input") is True
        eligible = incident is None and final == "completed" and not truncated
        return Qwen3AttemptOutcome(
            attempts=tuple(attempts),
            condition=condition,
            contract_blockers=self.contract.blockers(),
            contract_sha256=self.contract.identity_sha256(),
            eligible_for_export=eligible,
            final_principal_state=final if isinstance(final, str) else None,
            generator_calls=calls,
            harness_incident=incident,
            preflight=prepared.preflight,
            raw_output=raw_output,
            rendered_prompt=prepared.rendered_prompt,
            serialized_input=prepared.serialized_input,
            test_only=self.contract.test_only,
            validated_triples=validated,
        )

    def _preflight(self, serialized: str) -> dict[str, Any]:
        if self.tokenizer is not None:
            return self._tokenizer_preflight(serialized)
        contract = self.contract
        if not contract.test_only or self.count_tokens is None or contract.effective_context_limit is None:
            missing = ["tokenizer_assets"]
            if contract.effective_context_limit is None:
                missing.append("effective_context_limit")
            if self.count_tokens is None:
                missing.append("token_counter")
            return {
                "assurance": "not_assessed",
                "fits": None,
                "measurement_scope": None,
                "missing": missing,
                "prompt_sha256": sha256_text(serialized),
                "reason": "preflight_not_assessed:" + ",".join(missing),
                "serialized_input_counted": False,
            }
        if contract.reserved_output_tokens is None:
            return {
                "assurance": "not_assessed",
                "fits": None,
                "missing": ["reserved_output_tokens"],
                "prompt_sha256": sha256_text(serialized),
                "reason": "preflight_not_assessed:reserved_output_tokens",
                "serialized_input_counted": False,
            }
        result = assess_preflight(
            PreflightRequest(
                prompt=serialized,
                count_tokens=self.count_tokens,
                effective_context_limit=contract.effective_context_limit,
                reserved_output_tokens=contract.reserved_output_tokens,
                tokenizer_id=contract.tokenizer_id,
                tokenizer_revision=contract.pinned_tokenizer_revision,
                serialization_id=contract.serialization_id,
                special_token_allowance=contract.special_token_allowance,
                counter_id=self.counter_id,
                measurement_scope="test_only",
            )
        )
        result["serialized_input_counted"] = True
        result["real_tokenizer_measured"] = False
        return result

    def _tokenizer_preflight(self, serialized: str) -> dict[str, Any]:
        """Count the official serialized input. This is not GPU or model readiness."""
        contract = self.contract
        if contract.effective_context_limit is None or contract.reserved_output_tokens is None:
            missing = []
            if contract.effective_context_limit is None:
                missing.append("effective_context_limit")
            if contract.reserved_output_tokens is None:
                missing.append("reserved_output_tokens")
            return {
                "approved_effective_context_limit": None,
                "assurance": "not_assessed",
                "fits": None,
                "gpu_ready": False,
                "measurement_scope": "tokenizer_only",
                "missing": missing,
                "model_ready": False,
                "prompt_sha256": sha256_text(serialized),
                "reason": "preflight_not_assessed:" + ",".join(missing),
                "serialized_input_counted": False,
            }
        result = assess_preflight(
            PreflightRequest(
                prompt=serialized,
                count_tokens=lambda text: len(official_input_token_ids(self.tokenizer, text)),
                effective_context_limit=contract.effective_context_limit,
                reserved_output_tokens=contract.reserved_output_tokens,
                tokenizer_id=contract.tokenizer_id,
                tokenizer_revision=QWEN3_OBSERVED_DEFAULT_REVISION,
                serialization_id=contract.serialization_id,
                special_token_allowance=0,
                counter_id=QWEN3_OFFICIAL_COUNTER_ID,
                measurement_scope="tokenizer_only",
            )
        )
        result["native_context_length"] = QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH
        result["real_tokenizer_measured"] = True
        result["serialized_input_counted"] = True
        result["tokenizer_declared_model_max_length"] = QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH
        result["tokenizer_revision_role"] = "technical_validation_not_scientific_pin"
        return result

    def _common(self, prepared: PreparedQwenInput, document: SyntheticDocument, attempt_index: int) -> dict[str, Any]:
        return {
            "condition": prepared.condition,
            "document_id": document.document_id,
            "example_id": document.resolved_example_id,
            "attempt_index": attempt_index,
            "prompt_sha256": prepared.rendered_prompt_sha256,
            "slot_sha256": None,
            "shell_sha256": None,
            "input_sha256": sha256_text(document.text),
            "preflight": prepared.preflight,
            "seed": self.contract.seed,
            "reserved_output_tokens": self.contract.reserved_output_tokens,
            "effective_context_limit": self.contract.effective_context_limit,
        }

    def _blocked_attempt(self, prepared: PreparedQwenInput, document: SyntheticDocument, attempt_index: int) -> dict[str, Any]:
        from re_te_system.controlled_experiment.contracts import AttemptRecord

        return AttemptRecord(principal_state="context_blocked", **self._common(prepared, document, attempt_index)).to_dict()

    def _failure_attempt(
        self,
        prepared: PreparedQwenInput,
        document: SyntheticDocument,
        attempt_index: int,
        state: str,
        exc: BaseException,
    ) -> dict[str, Any]:
        from re_te_system.controlled_experiment.contracts import AttemptRecord

        record = AttemptRecord(
            principal_state=state,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            **self._common(prepared, document, attempt_index),
        ).to_dict()
        if state == "timeout":
            record["timeout_enforcement"] = "not_implemented"
        return record


def _revision_blocker(value: str | None, label: str) -> str | None:
    if value is None:
        return f"{label}_unresolved"
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        return f"{label}_not_immutable"
    return None


@dataclass(frozen=True)
class Qwen3ExecutionGrant:
    """Explicit grant. A test grant cannot load weights, and this phase cannot either."""

    granted: bool
    test_only: bool
    allows_weight_load: bool = False
    purpose: str = ""


def execution_blockers(contract: Qwen3GenerationContract) -> tuple[str, ...]:
    """Gates for an injected or subprocess run. Weight authorization is not one of them."""
    blocked: list[str | None] = [
        None if contract.test_only else "test_only_required_for_unexecuted_runtime",
        _revision_blocker(contract.pinned_checkpoint_revision, "pinned_checkpoint_revision"),
        _revision_blocker(contract.pinned_tokenizer_revision, "pinned_tokenizer_revision"),
        (
            None
            if contract.pinned_checkpoint_revision == QWEN3_OBSERVED_DEFAULT_REVISION
            else "checkpoint_revision_not_the_validated_candidate"
        ),
        (
            None
            if contract.pinned_tokenizer_revision == QWEN3_OBSERVED_DEFAULT_REVISION
            else "tokenizer_revision_not_the_validated_candidate"
        ),
        None if tokenizer_assets_verified() else "tokenizer_assets_not_local",
        None if contract.do_sample is False else "greedy_decoding_not_selected",
        None if contract.eos_policy_id == QWEN3_EOS_POLICY_ID else "eos_policy_unapproved",
        None if contract.pad_token_id == QWEN3_CONTROLLED_PAD_TOKEN_ID else "pad_policy_unapproved",
        None if contract.max_new_tokens is not None else "max_new_tokens_unapproved",
        None if contract.seed is not None else "seed_unapproved",
        None if contract.effective_context_limit is not None else "effective_context_limit_unapproved",
        None if contract.reserved_output_tokens is not None else "reserved_output_tokens_unapproved",
        None if contract.device_policy is not None else "device_policy_unapproved",
        None if contract.dtype_policy is not None else "dtype_policy_unapproved",
        None if contract.timeout_seconds is not None else "process_timeout_budget_unapproved",
        (
            "reserved_output_smaller_than_max_new_tokens"
            if (
                contract.reserved_output_tokens is not None
                and contract.max_new_tokens is not None
                and contract.reserved_output_tokens < contract.max_new_tokens
            )
            else None
        ),
    ]
    if contract.effective_context_limit == QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH:
        blocked.append("effective_context_limit_copies_native_context")
    if contract.effective_context_limit == QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH:
        blocked.append("effective_context_limit_copies_tokenizer_model_max_length")
    return tuple(item for item in blocked if item is not None)


def generation_request(
    contract: Qwen3GenerationContract,
    input_ids: Sequence[int],
    *,
    condition: str,
    document_id: str,
    attempt_index: int,
) -> dict[str, Any]:
    """Build the model call. Sampling values from generation_config.json are absent."""
    return {
        "attempt_index": attempt_index,
        "condition": condition,
        "device_policy": contract.device_policy,
        "document_id": document_id,
        "dtype_policy": contract.dtype_policy,
        "eos_policy_id": contract.eos_policy_id,
        "input_ids": [int(item) for item in input_ids],
        "model_generate_kwargs": {
            "do_sample": False,
            "eos_token_id": list(QWEN3_CONTROLLED_EOS_TOKEN_IDS),
            "max_new_tokens": contract.max_new_tokens,
            "pad_token_id": QWEN3_CONTROLLED_PAD_TOKEN_ID,
        },
        "seed": contract.seed,
        "test_only": True,
        "timeout_seconds": contract.timeout_seconds,
    }


def transport_continuation(continuation_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Drop only a trailing run of chat-end or end-of-text ids. Embedded ids stay."""
    kept = [int(item) for item in continuation_ids]
    removed: list[int] = []
    while kept and kept[-1] in QWEN3_TRANSPORT_TOKEN_IDS:
        removed.append(kept.pop())
    removed.reverse()
    return kept, removed


def controlled_continuation(
    tokenizer: Any,
    input_ids: Sequence[int],
    sequence_ids: Sequence[int],
) -> dict[str, Any]:
    """Split a generated sequence and apply the shared C0/C1 transport rule.

    ``parser_input`` is the only string passed to the TSV parser. The raw
    decode and the original continuation ids are retained unchanged.
    """
    prefix = [int(item) for item in input_ids]
    sequence = [int(item) for item in sequence_ids]
    if sequence[: len(prefix)] != prefix:
        raise ValueError("sequence_ids does not start with the serialized input ids")
    continuation = sequence[len(prefix) :]
    kept, removed = transport_continuation(continuation)
    raw = tokenizer.decode(continuation, skip_special_tokens=False)
    parser_input = tokenizer.decode(kept, skip_special_tokens=False)
    if not isinstance(raw, str) or not isinstance(parser_input, str):
        raise Qwen3TokenizerAssetsError("continuation decode must be text")
    return {
        "continuation_ids": continuation,
        "decoded_raw_output": raw,
        "input_ids": prefix,
        "input_ids_sha256": sha256_payload(prefix),
        "parser_input": parser_input,
        "removed_terminal_ids": removed,
        "sequence_ids_sha256": sha256_payload(sequence),
        "skip_special_tokens": False,
        "transport_policy": QWEN3_TRANSPORT_POLICY_ID,
    }


QWEN3_WEIGHT_INDEX_NAME = "model.safetensors.index.json"
QWEN3_SMOKE_INPUT_SCOPE = "invented_smoke_only"
QWEN3_PROPOSED_SMOKE_ID = "qwen3-proposed-smoke-not-scientific"
# Hub LFS identities observed for the candidate revision. The index has no approved SHA-256.
QWEN3_WEIGHT_SHARDS = (
    {
        "byte_size": 3957900840,
        "filename": "model-00001-of-00003.safetensors",
        "sha256": "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
    },
    {
        "byte_size": 3987450520,
        "filename": "model-00002-of-00003.safetensors",
        "sha256": "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
    },
    {
        "byte_size": 99630640,
        "filename": "model-00003-of-00003.safetensors",
        "sha256": "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d",
    },
)


def official_weight_manifest() -> dict[str, Any]:
    """Approved shard identities. The index SHA-256 is intentionally absent."""
    config_sha, config_size = QWEN3_TOKENIZER_ASSETS["config.json"]
    generation_sha, generation_size = QWEN3_TOKENIZER_ASSETS["generation_config.json"]
    return {
        "config": {"byte_size": config_size, "filename": "config.json", "sha256": config_sha},
        "generation_config": {
            "byte_size": generation_size,
            "filename": "generation_config.json",
            "sha256": generation_sha,
        },
        "index_byte_size": 32819,
        "index_filename": QWEN3_WEIGHT_INDEX_NAME,
        "index_sha256": None,
        "revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "shards": [dict(item) for item in QWEN3_WEIGHT_SHARDS],
    }


def proposed_smoke_contract() -> Qwen3GenerationContract:
    """Candidate smoke settings. They are not a scientific campaign freeze.

    Device, dtype, and both timeout budgets stay unset because this environment
    does not identify the execution machine.
    """
    return Qwen3GenerationContract(
        do_sample=False,
        effective_context_limit=4096,
        eos_policy_id=QWEN3_EOS_POLICY_ID,
        max_new_tokens=256,
        pad_token_id=QWEN3_CONTROLLED_PAD_TOKEN_ID,
        pinned_checkpoint_revision=QWEN3_OBSERVED_DEFAULT_REVISION,
        pinned_tokenizer_revision=QWEN3_OBSERVED_DEFAULT_REVISION,
        reserved_output_tokens=256,
        seed=42,
        test_only=False,
    )


@dataclass(frozen=True)
class Qwen3SmokeAuthorization:
    """Real-smoke grant. It does not open the phase lock and it cannot be a test grant."""

    granted: bool
    revision: str
    input_scope: str
    conditions: tuple[str, ...] = ("C0", "C1")
    configuration_id: str = QWEN3_PROPOSED_SMOKE_ID
    device_policy: str | None = None
    dtype_policy: str | None = None
    load_timeout_seconds: float | None = None
    generation_timeout_seconds: float | None = None


def inspect_qwen3_checkpoint(
    directory: Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash a local checkpoint. This does not download or open a model.

    ``revision.txt`` is not read. The index SHA-256 is observed and is not
    compared with an invented expected digest.
    """
    selected = official_weight_manifest() if manifest is None else dict(manifest)
    root = Path(directory)
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    shards = selected.get("shards")
    if not isinstance(shards, list) or not shards:
        errors.append("manifest has no shards")
        shards = []
    expected_names = []
    for shard in shards:
        name = shard.get("filename")
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            errors.append("manifest shard filename is not a safetensors name")
            continue
        expected_names.append(name)
        files.append(_hash_required_file(root / name, shard, errors))
    for label in ("config", "generation_config"):
        spec = selected.get(label)
        if isinstance(spec, dict):
            files.append(_hash_required_file(root / str(spec.get("filename")), spec, errors))
    present = {path.name for path in root.glob("*.safetensors")} if root.is_dir() else set()
    unexpected = sorted(present - set(expected_names))
    if unexpected:
        errors.append("unexpected weight file: " + ",".join(unexpected))
    index_name = str(selected.get("index_filename") or QWEN3_WEIGHT_INDEX_NAME)
    index_path = root / index_name
    index_sha = None
    total_size = None
    if not index_path.is_file():
        errors.append("missing required checkpoint file: " + index_name)
    else:
        index_bytes = index_path.read_bytes()
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        expected_index_size = selected.get("index_byte_size")
        if expected_index_size is not None and len(index_bytes) != expected_index_size:
            errors.append("weight index size does not match the approved manifest")
        _check_weight_index(index_bytes, expected_names, shards, errors)
        try:
            payload = json.loads(index_bytes.decode("utf-8"))
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            if isinstance(metadata, dict) and "total_size" in metadata:
                total_size = metadata.get("total_size")
        except (UnicodeError, json.JSONDecodeError):
            total_size = None
    if selected.get("index_sha256") is not None:
        errors.append("index sha256 must not be predeclared; hash the local bytes")
    return {
        "directory": str(root),
        "errors": errors,
        "files": files,
        "index_sha256": index_sha,
        "index_sha256_approved": False,
        "index_total_size": total_size,
        "revision": selected.get("revision"),
        "revision_file_used": False,
        "verified": not errors,
    }


def verify_qwen3_checkpoint(
    directory: Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the inspection or raise. A failed check does not load weights."""
    report = inspect_qwen3_checkpoint(directory, manifest)
    if not report["verified"]:
        raise Qwen3ConfigurationError("checkpoint integrity failed: " + "; ".join(report["errors"]))
    return report


def load_qwen3_model(
    contract: Qwen3GenerationContract,
    checkpoint_dir: Path,
    grant: Qwen3ExecutionGrant | None = None,
) -> Any:
    """Legacy weight entry. Integrity runs first. The constructor stays locked."""
    if (
        grant is None
        or grant.granted is not True
        or grant.test_only is not False
        or grant.allows_weight_load is not True
    ):
        raise Qwen3WeightsNotAuthorized("Loading Qwen3 weights is not authorized.")
    if _revision_blocker(contract.pinned_checkpoint_revision, "pinned_checkpoint_revision") is not None:
        raise Qwen3ConfigurationError("checkpoint revision must be an immutable revision")
    if contract.pinned_checkpoint_revision != QWEN3_OBSERVED_DEFAULT_REVISION:
        raise Qwen3ConfigurationError("checkpoint revision is not the validated candidate")
    verify_qwen3_checkpoint(Path(checkpoint_dir))
    return _construct_locked_model(Path(checkpoint_dir), device_policy=None, dtype_policy=None)


def construct_qwen3_causal_lm(
    checkpoint_dir: Path,
    authorization: Qwen3SmokeAuthorization,
    contract: Qwen3GenerationContract,
    verification: Mapping[str, Any],
) -> Any:
    """Load the local causal LM. The phase lock makes this unreachable in this phase."""
    if isinstance(authorization, Qwen3ExecutionGrant) or not isinstance(authorization, Qwen3SmokeAuthorization):
        raise Qwen3WeightsNotAuthorized("a test grant cannot construct Qwen3")
    if authorization.granted is not True or authorization.input_scope != QWEN3_SMOKE_INPUT_SCOPE:
        raise Qwen3WeightsNotAuthorized("smoke authorization does not permit weight loading")
    if verification.get("verified") is not True:
        raise Qwen3ConfigurationError("refusing to construct a model without a verified checkpoint")
    if authorization.device_policy is None or authorization.dtype_policy is None:
        raise Qwen3ConfigurationError("device and dtype are unresolved")
    return _construct_locked_model(
        Path(checkpoint_dir),
        device_policy=authorization.device_policy,
        dtype_policy=authorization.dtype_policy,
    )


def _construct_locked_model(
    checkpoint_dir: Path,
    *,
    device_policy: str | None,
    dtype_policy: str | None,
) -> Any:
    if QWEN3_WEIGHT_EXECUTION_PHASE_LOCK:
        raise Qwen3WeightsNotAuthorized(
            "AutoModelForCausalLM construction is implemented and locked. "
            "EXP.4.3.1 does not execute it."
        )
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir), local_files_only=True)
    # Device and dtype are applied only after the lock above is opened.
    if dtype_policy not in (None, "pending_environment"):
        model = model.to(dtype=_torch_dtype(dtype_policy))
    if device_policy not in (None, "pending_environment"):
        model = model.to(device_policy)
    return model


def _torch_dtype(name: str) -> Any:
    import torch

    value = getattr(torch, name, None)
    if value is None:
        raise Qwen3ConfigurationError("dtype is not a torch dtype name")
    return value


def _hash_required_file(path: Path, spec: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    name = str(spec.get("filename"))
    expected_size = spec.get("byte_size")
    expected_sha = spec.get("sha256")
    if not path.is_file():
        errors.append("missing required checkpoint file: " + name)
        return {"filename": name, "present": False, "sha256_matches": False}
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    observed = digest.hexdigest()
    matches = observed == expected_sha and size == expected_size
    if not matches:
        errors.append(f"sha256 or size mismatch: {name}")
    return {
        "byte_size": size,
        "expected_byte_size": expected_size,
        "filename": name,
        "present": True,
        "sha256": observed,
        "sha256_matches": matches,
    }


def _check_weight_index(
    raw: bytes,
    expected_names: list[str],
    shards: list[Mapping[str, Any]],
    errors: list[str],
) -> None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        errors.append("weight index is not JSON")
        return
    if not isinstance(payload, dict):
        errors.append("weight index is not an object")
        return
    mapping = payload.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        errors.append("weight index has no tensor map")
        return
    referenced: set[str] = set()
    for tensor_name, shard_name in mapping.items():
        if not isinstance(tensor_name, str) or tensor_name == "":
            errors.append("weight index contains an empty tensor name")
            continue
        if not isinstance(shard_name, str):
            errors.append("weight index maps a tensor to a non-string shard")
            continue
        referenced.add(shard_name)
        if shard_name not in expected_names:
            errors.append("weight index references an unexpected shard: " + shard_name)
    missing = [name for name in expected_names if name not in referenced]
    if missing:
        errors.append("weight index does not reference: " + ",".join(missing))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata:
        expected_total = sum(int(item["byte_size"]) for item in shards if "byte_size" in item)
        if metadata.get("total_size") != expected_total:
            errors.append("weight index total_size does not match the shard bytes")


class Qwen3SmokeWorker:
    """One loaded model for many inputs. The real constructor is locked.

    An injected loader is a test double. It does not verify real weight bytes
    beyond the checkpoint inspection this object already performed.
    """

    def __init__(
        self,
        authorization: Qwen3SmokeAuthorization,
        contract: Qwen3GenerationContract,
        checkpoint_dir: Path,
        *,
        manifest: Mapping[str, Any] | None = None,
        loader: Callable[[Mapping[str, Any]], Any] | None = None,
        generator: Callable[[Any, Sequence[int], Mapping[str, Any]], Sequence[int]] | None = None,
    ) -> None:
        if isinstance(authorization, Qwen3ExecutionGrant) or not isinstance(authorization, Qwen3SmokeAuthorization):
            raise Qwen3WeightsNotAuthorized("the smoke worker rejects a test-only grant")
        if _revision_blocker(authorization.revision, "pinned_checkpoint_revision") is not None:
            raise Qwen3ConfigurationError("checkpoint revision must be an immutable revision")
        if authorization.granted is not True or authorization.input_scope != QWEN3_SMOKE_INPUT_SCOPE:
            raise Qwen3ExecutionNotAuthorized("smoke input scope is not authorized")
        if authorization.revision != contract.pinned_checkpoint_revision:
            raise Qwen3ConfigurationError("authorization revision does not match the contract")
        if contract.reserved_output_tokens is None or contract.max_new_tokens is None:
            raise Qwen3ConfigurationError("smoke budgets are unresolved")
        if contract.reserved_output_tokens < contract.max_new_tokens:
            raise Qwen3ConfigurationError("reserved output is smaller than max_new_tokens")
        self.authorization = authorization
        self.contract = contract
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifest = manifest
        self.loader = loader
        self.generator = generator
        self.model: Any = None
        self.load_count = 0
        self.verification: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self.model is not None:
            return {"load_count": self.load_count, "weights_executed": False}
        self.verification = verify_qwen3_checkpoint(self.checkpoint_dir, self.manifest)
        if self.loader is None:
            self.model = construct_qwen3_causal_lm(
                self.checkpoint_dir,
                self.authorization,
                self.contract,
                self.verification,
            )
        else:
            self.model = self.loader(self.verification)
        self.load_count += 1
        return {"load_count": self.load_count, "weights_executed": False}

    def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Generate from the supplied ids. The prompt is not tokenized again."""
        self.load()
        if "tokenizer" in request or "text" in request or "prompt" in request:
            raise Qwen3ConfigurationError("the worker does not accept text or a tokenizer")
        input_ids = [int(item) for item in request["input_ids"]]
        kwargs = dict(request["model_generate_kwargs"])
        if kwargs.get("do_sample") is not False:
            raise Qwen3ConfigurationError("the worker refuses sampling")
        if any(name in kwargs for name in ("temperature", "top_p", "top_k")):
            raise Qwen3ConfigurationError("greedy generation must not inherit sampling parameters")
        if kwargs.get("max_new_tokens") != self.contract.max_new_tokens:
            raise Qwen3ConfigurationError("generation budget does not match the contract")
        if kwargs.get("eos_token_id") != list(QWEN3_CONTROLLED_EOS_TOKEN_IDS):
            raise Qwen3ConfigurationError("EOS policy is not the controlled policy")
        if kwargs.get("pad_token_id") != QWEN3_CONTROLLED_PAD_TOKEN_ID:
            raise Qwen3ConfigurationError("PAD policy is not the verified pad id")
        if self.generator is None:
            raise Qwen3WeightsNotAuthorized("model generation is locked")
        sequence = [int(item) for item in self.generator(self.model, input_ids, kwargs)]
        if sequence[: len(input_ids)] != input_ids:
            raise Qwen3ConfigurationError("worker output does not start with the preflight ids")
        new_tokens = sequence[len(input_ids) :]
        if len(new_tokens) > int(self.contract.max_new_tokens or 0):
            raise Qwen3ConfigurationError("worker returned more new tokens than max_new_tokens")
        termination = "eos" if new_tokens and new_tokens[-1] == QWEN3_CONTROLLED_EOS_TOKEN_IDS[0] else "length"
        return {
            "device_applied": False,
            "dtype_applied": False,
            "load_count": self.load_count,
            "sequence_ids": sequence,
            "termination": termination,
            "weights_executed": False,
        }


class PersistentJsonWorker:
    """Line-delimited JSON session. One process can answer several requests.

    This class does not load a model. Callers supply the worker command.
    """

    def __init__(self, argv: Sequence[str]) -> None:
        self.process = subprocess.Popen(
            [str(item) for item in argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        self.pids: list[int] = []

    def request(self, payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise Qwen3WorkerFailed("worker already exited")
        if self.process.stdin is None or self.process.stdout is None:
            raise Qwen3WorkerFailed("worker pipes are closed")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        try:
            line = _read_worker_line(self.process, timeout_seconds)
        except Qwen3WorkerTimeout as exc:
            self.pids.append(exc.pid)
            raise
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Qwen3WorkerFailed("worker did not return JSON") from exc
        if not isinstance(value, dict):
            raise Qwen3WorkerFailed("worker result is not an object")
        return value

    def close(self) -> None:
        if self.process.poll() is None:
            _terminate_process_group(self.process, 0.5)


def _read_worker_line(process: subprocess.Popen[str], timeout_seconds: float) -> str:
    import select
    import time

    if process.stdout is None:
        raise Qwen3WorkerFailed("worker stdout is closed")
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process, 0.5)
            raise Qwen3WorkerTimeout(timeout_seconds, process.pid)
        ready, _, _ = select.select([process.stdout], [], [], remaining)
        if not ready:
            _terminate_process_group(process, 0.5)
            raise Qwen3WorkerTimeout(timeout_seconds, process.pid)
        line = process.stdout.readline()
        if line == "":
            raise Qwen3WorkerFailed(f"worker exited {process.poll()}")
        return line


def invoke_json_worker(
    argv: Sequence[str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    grace_seconds: float = 0.5,
) -> dict[str, Any]:
    """Run a worker in its own process group and stop that group on timeout.

    This does not load a model. The caller supplies the worker command.
    """
    if timeout_seconds <= 0:
        raise Qwen3ConfigurationError("timeout_seconds must be positive")
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(json.dumps(payload), timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process, grace_seconds)
        raise Qwen3WorkerTimeout(timeout_seconds, process.pid) from exc
    if process.returncode != 0:
        raise Qwen3WorkerFailed(f"worker exited {process.returncode}: {stderr}")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Qwen3WorkerFailed("worker did not return JSON") from exc
    if not isinstance(value, dict):
        raise Qwen3WorkerFailed("worker result is not an object")
    return value


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    import os

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


class ScriptedGenerationBackend:
    """In-process stand-in. It returns caller-supplied ids and does not load weights."""

    def __init__(self, scripts: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> None:
        self.scripts = dict(scripts)
        self.calls: list[dict[str, Any]] = []

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(dict(request))
        script = self.scripts[(request["condition"], request["document_id"], request["attempt_index"])]
        kind = script["kind"]
        if kind == "error":
            raise RuntimeError(str(script.get("message", "model failed")))
        if kind == "infrastructure":
            from re_te_system.controlled_experiment.contracts import ExternalInfrastructureError

            raise ExternalInfrastructureError(str(script.get("message", "external infrastructure failed")))
        continuation = [int(item) for item in script["continuation_ids"]]
        return {
            "sequence_ids": [int(item) for item in request["input_ids"]] + continuation,
            "termination": script.get("termination", "eos"),
        }


class SubprocessGenerationBackend:
    """Process-boundary stand-in. The worker command is supplied by the caller."""

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = [str(item) for item in argv]
        self.calls: list[dict[str, Any]] = []
        self.pids: list[int] = []

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(dict(request))
        try:
            return invoke_json_worker(self.argv, request, float(request["timeout_seconds"]))
        except Qwen3WorkerTimeout as exc:
            self.pids.append(exc.pid)
            raise


def execute_controlled_document(
    contract: Qwen3GenerationContract,
    document: Any,
    condition: str,
    *,
    tokenizer: Any,
    backend: Any,
    grant: Qwen3ExecutionGrant | None,
) -> dict[str, Any]:
    """Run one document through the controlled chain without loading weights."""
    _require_test_grant(grant)
    blocked = execution_blockers(contract)
    if blocked:
        raise Qwen3ConfigurationError("real runtime is not configured: " + ",".join(blocked))
    if backend is None:
        raise Qwen3ExecutionNotAuthorized("no generation backend was supplied")
    from re_te_system.controlled_experiment.campaign import (
        _completed_attempt,
        _final_state,
        _prediction_record,
        _shell,
    )
    from re_te_system.controlled_experiment.contracts import (
        RETRYABLE_STATES,
        AttemptRecord,
        ExternalInfrastructureError,
    )

    shell = _shell()
    shell_sha = sha256_text(shell.shell_template())
    slot_sha = sha256_text(shell.conditioning_for(condition))
    prepared = Qwen3Runtime(contract, tokenizer=tokenizer).prepare(condition, document.text)
    input_ids: list[int] = []
    if prepared.preflight.get("assurance") != "not_assessed":
        input_ids = official_input_token_ids(tokenizer, prepared.serialized_input)
        if prepared.preflight.get("prompt_tokens") != len(input_ids):
            raise Qwen3TokenizerAssetsError("preflight token count is not the serialized input id length")
    attempts: list[dict[str, Any]] = []
    validation = None
    backend_calls = 0
    if prepared.preflight.get("assurance") == "not_assessed":
        record = AttemptRecord(
            condition=condition,
            document_id=document.document_id,
            example_id=document.resolved_example_id,
            attempt_index=None,
            principal_state=None,
            harness_incident="preflight_not_assessed",
            harness_message=str(prepared.preflight.get("reason")),
            prompt_sha256=prepared.rendered_prompt_sha256,
            slot_sha256=slot_sha,
            shell_sha256=shell_sha,
            input_sha256=sha256_text(document.text),
            preflight=prepared.preflight,
            seed=contract.seed,
        ).to_dict()
        attempts.append(record)
        state = {
            "campaign_scientific_id": None,
            "condition": condition,
            "document_id": document.document_id,
            "example_id": document.resolved_example_id,
            "harness_incident": "preflight_not_assessed",
            "harness_message": str(prepared.preflight.get("reason")),
        }
        return _execution_result(attempts, state, None, False, prepared, input_ids, backend_calls)
    for attempt_index in (1, 2):
        common = _attempt_common(prepared, document, condition, attempt_index, shell_sha, slot_sha, contract)
        if prepared.preflight.get("fits") is not True:
            attempts.append(AttemptRecord(principal_state="context_blocked", **common).to_dict())
            break
        request = generation_request(
            contract,
            input_ids,
            condition=condition,
            document_id=document.document_id,
            attempt_index=attempt_index,
        )
        if request["input_ids"] != input_ids:
            raise Qwen3TokenizerAssetsError("generation request changed the preflight token ids")
        backend_calls += 1
        try:
            payload = backend.generate(request)
        except ExternalInfrastructureError as exc:
            attempts.append(_exception_attempt("external_infrastructure_error", exc, common))
        except TimeoutError as exc:
            record = _exception_attempt("timeout", exc, common)
            record["timeout_enforcement"] = "subprocess_process_group"
            record["worker_reaped"] = True
            attempts.append(record)
        except Exception as exc:
            attempts.append(_exception_attempt("model_execution_error", exc, common))
        else:
            try:
                trace = controlled_continuation(tokenizer, input_ids, payload["sequence_ids"])
            except (KeyError, TypeError, ValueError) as exc:
                attempts.append(_exception_attempt("model_execution_error", exc, common))
            else:
                metadata = {
                    "completion_reason_source": "injected_or_subprocess_double",
                    "do_sample": False,
                    "prediction_source": "injected_backend_not_model",
                    "seed": contract.seed,
                    "timeout_enforcement": "subprocess_process_group",
                }
                if payload.get("termination") == "length":
                    metadata["output_reached_limit"] = True
                attempt, validation = _completed_attempt(trace["parser_input"], metadata, common)
                record = attempt.to_dict()
                diagnostics = dict(record.get("diagnostics") or {})
                diagnostics.update(trace)
                record["diagnostics"] = diagnostics
                attempts.append(record)
        if attempts[-1].get("principal_state") not in RETRYABLE_STATES or attempt_index == 2:
            break
    final = attempts[-1]
    earlier = [
        {"attempt_index": item.get("attempt_index"), "principal_state": item.get("principal_state")}
        for item in attempts[:-1]
    ]
    prediction = None
    keep = False
    parsed_triples = None
    invalid_lines = None
    if final.get("principal_state") == "completed":
        invalid_lines = sum(1 for issue in final.get("parse_issues") or [] if issue.get("code") == "UNPARSEABLE_CHUNK")
        parsed_triples = 0 if validation is None else len(validation.triples)
        prediction = _prediction_record("pending", condition, document, validation)
        keep = (final.get("diagnostics") or {}).get("truncated_input") is not True
    state = _final_state(
        "pending",
        condition,
        document,
        final.get("principal_state"),
        earlier,
        len(attempts),
        final.get("diagnostics"),
        parsed_triples,
        invalid_lines,
    )
    return _execution_result(attempts, state, prediction, keep, prepared, input_ids, backend_calls)


def run_qwen3_contract_campaign(
    contract: Qwen3GenerationContract,
    documents: Sequence[Any],
    output_dir: Path,
    *,
    tokenizer: Any,
    backend: Any,
    grant: Qwen3ExecutionGrant | None,
    conditions: tuple[str, ...] = ("C0", "C1"),
) -> dict[str, Any]:
    """Write bridge-compatible artifacts for an injected run. This is not a scientific result."""
    _require_test_grant(grant)
    blocked = execution_blockers(contract)
    if blocked:
        raise Qwen3ConfigurationError("real runtime is not configured: " + ",".join(blocked))
    if any(condition not in ("C0", "C1") for condition in conditions):
        raise Qwen3ConfigurationError("this runtime executes only C0 and C1")
    from re_te_system.controlled_experiment.campaign import (
        _export_condition,
        _observe_readiness,
        _protocol_identity,
        _protocol_text,
        _shell,
    )
    from re_te_system.controlled_experiment.contracts import (
        CAMPAIGN_CONTRACT_VERSION,
        PREVIOUS_CAMPAIGN_CONTRACT_VERSION,
        campaign_scientific_id,
    )
    from re_te_system.controlled_experiment.dependencies import (
        SYSTEM_REPO,
        ensure_package,
        git_revision,
        repository_root_from_package,
    )
    from re_te_system.controlled_experiment.fingerprint import (
        audit_inventory,
        collect_implementation_parts,
        implementation_fingerprint,
    )
    from re_te_system.controlled_experiment.reporting import (
        coverage_report,
        decide_campaign_status,
        write_campaign_files,
    )
    from re_te_system.evaluation_bridge.contracts import FROZEN_POLICY

    ensure_package("re_te_benchmark", "re_te_benchmark")
    ensure_package("re_te_evaluation", "re_te_evaluation")
    shell = _shell()
    benchmark_root = repository_root_from_package("re_te_benchmark")
    evaluation_root = repository_root_from_package("re_te_evaluation")
    shell_text = shell.shell_template()
    shell_sha = sha256_text(shell_text)
    slots = {condition: shell.conditioning_for(condition) for condition in conditions}
    protocol = _protocol_identity(benchmark_root)
    revisions = {
        "re_te_benchmark": git_revision(benchmark_root),
        "re_te_evaluation": git_revision(evaluation_root),
        "re_te_system": git_revision(SYSTEM_REPO),
    }
    parts = collect_implementation_parts(
        system_root=SYSTEM_REPO,
        benchmark_root=benchmark_root,
        evaluation_root=evaluation_root,
        shell_text=shell_text,
        slots=slots,
        protocol_text=_protocol_text(benchmark_root),
    )
    fingerprint = implementation_fingerprint(parts, revisions)
    identity = {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "conditions": list(conditions),
        "context_budget": {
            "approved_effective_context_limit": None,
            "effective_context_limit": contract.effective_context_limit,
            "native_context_length": QWEN3_DECLARED_NATIVE_CONTEXT_LENGTH,
            "reserved_output_tokens": contract.reserved_output_tokens,
            "special_token_allowance": 0,
            "tokenizer_declared_model_max_length": QWEN3_TOKENIZER_DECLARED_MODEL_MAX_LENGTH,
        },
        "decoding": {
            "bitwise_determinism": False,
            "do_sample": False,
            "eos_policy_id": contract.eos_policy_id,
            "eos_token_id": list(QWEN3_CONTROLLED_EOS_TOKEN_IDS),
            "max_new_tokens": contract.max_new_tokens,
            "pad_token_id": contract.pad_token_id,
            "seed": contract.seed,
            "temperature": None,
            "top_k": None,
            "top_p": None,
        },
        "documents": [
            {
                "document_id": document.document_id,
                "example_id": document.resolved_example_id,
                "text_sha256": sha256_text(document.text),
            }
            for document in documents
        ],
        "evaluation_policy": {
            **FROZEN_POLICY,
            "result_role": QWEN3_RESULT_ROLE,
        },
        "extractor": {
            "name": QWEN3_MODEL_ID,
            "prediction_source": "injected_backend_not_model",
            "revision": contract.pinned_checkpoint_revision,
        },
        "input_truncation_policy": "preflight_block_no_silent_truncation",
        "mode": QWEN3_REAL_MODE,
        "prediction_contract_version": "1.0",
        "protocol": protocol,
        "serialization_id": contract.serialization_id,
        "shell": {"id": shell.SHELL_ID, "sha256": shell_sha},
        "slot_sha256": {condition: sha256_text(slots[condition]) for condition in conditions},
        "tokenizer": {
            "chat_template_sha256": QWEN3_OBSERVED_CHAT_TEMPLATE_SHA256,
            "id": QWEN3_MODEL_ID,
            "revision": contract.pinned_tokenizer_revision,
            "revision_role": "technical_validation_not_scientific_pin",
        },
    }
    scientific_id = campaign_scientific_id(identity)
    manifest = {
        "audit": {"relevant_tests": audit_inventory(SYSTEM_REPO)},
        "campaign_scientific_id": scientific_id,
        "contract_compatibility": {
            "previous": PREVIOUS_CAMPAIGN_CONTRACT_VERSION,
            "scoring": "conditions/<condition>/evaluation_predictions.jsonl",
            "version": CAMPAIGN_CONTRACT_VERSION,
        },
        "identity": identity,
        "implementation_fingerprint": fingerprint,
        "observed_readiness": _observe_readiness(benchmark_root),
        "operational": {
            "created_at": None,
            "export_sha256": {},
            "freeze_status": "candidate_not_scientifically_approved",
            "real_execution_authorized": False,
            "repositories": {
                name: {
                    "commit_identifies_tree": record["commit_identifies_tree"],
                    "dirty": record["dirty"],
                    "head": record["head"],
                }
                for name, record in revisions.items()
            },
            "runtime_status": QWEN3_RUNTIME_STATUS,
            "timeout_enforcement": "subprocess_process_group",
            "weight_execution_phase_lock": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    publishable: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for condition in conditions:
        for document in documents:
            outcome = execute_controlled_document(
                contract,
                document,
                condition,
                tokenizer=tokenizer,
                backend=backend,
                grant=grant,
            )
            for attempt in outcome["attempts"]:
                attempt["campaign_scientific_id"] = scientific_id
            state = dict(outcome["state"])
            state["campaign_scientific_id"] = scientific_id
            attempts.extend(outcome["attempts"])
            states.append(state)
            prediction = outcome["prediction"]
            if prediction is not None:
                prediction["run_id"] = scientific_id
                if outcome["keep"]:
                    publishable.append(prediction)
                else:
                    withheld.append(prediction)
    export_sha: dict[str, str] = {}
    export_counts: dict[str, int] = {}
    for condition in conditions:
        _rows, digest = _export_condition(
            output_dir,
            condition,
            [row for row in publishable if row["condition"] == condition],
            [row for row in withheld if row["condition"] == condition],
        )
        export_sha[condition] = digest
        export_counts[condition] = len(_rows)
    status = decide_campaign_status(states, contract_abort=False)
    manifest["operational"]["export_sha256"] = dict(export_sha)
    coverage = coverage_report(tuple(documents), conditions, states, attempts, export_counts)
    result = {
        "attempts": attempts,
        "campaign_status": status,
        "coverage": coverage,
        "document_states": states,
        "implementation_fingerprint": fingerprint,
        "manifest": manifest,
        "prediction_records": publishable,
        "scientific_id": scientific_id,
        "withheld_prediction_records": withheld,
    }
    write_campaign_files(output_dir, result)
    summary = output_dir / "synthetic_evaluation_summary.json"
    if summary.exists():
        summary.unlink()
    return result


def _require_test_grant(grant: Qwen3ExecutionGrant | None) -> None:
    if grant is None or grant.granted is not True or grant.test_only is not True or grant.allows_weight_load:
        raise Qwen3ExecutionNotAuthorized(
            "Qwen3 generation is blocked. A test-only grant is required, and it cannot load weights."
        )


def _execution_result(
    attempts: list[dict[str, Any]],
    state: dict[str, Any],
    prediction: dict[str, Any] | None,
    keep: bool,
    prepared: PreparedQwenInput,
    input_ids: list[int],
    backend_calls: int,
) -> dict[str, Any]:
    return {
        "attempts": attempts,
        "backend_calls": backend_calls,
        "input_ids": input_ids,
        "keep": keep,
        "prediction": prediction,
        "prepared": prepared,
        "state": state,
    }


def _attempt_common(
    prepared: PreparedQwenInput,
    document: Any,
    condition: str,
    attempt_index: int,
    shell_sha: str,
    slot_sha: str,
    contract: Qwen3GenerationContract,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "document_id": document.document_id,
        "example_id": document.resolved_example_id,
        "attempt_index": attempt_index,
        "prompt_sha256": prepared.rendered_prompt_sha256,
        "slot_sha256": slot_sha,
        "shell_sha256": shell_sha,
        "input_sha256": sha256_text(document.text),
        "preflight": prepared.preflight,
        "seed": contract.seed,
        "reserved_output_tokens": contract.reserved_output_tokens,
        "effective_context_limit": contract.effective_context_limit,
    }


def _exception_attempt(state: str, exc: BaseException, common: dict[str, Any]) -> dict[str, Any]:
    from re_te_system.controlled_experiment.contracts import AttemptRecord

    return AttemptRecord(
        principal_state=state,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        **common,
    ).to_dict()
