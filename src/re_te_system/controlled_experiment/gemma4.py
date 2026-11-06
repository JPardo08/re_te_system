"""Gemma 4 12B instruction-tuned runtime for the controlled experiment.

G00-G05 do not run a Gold campaign. The official revision is an identity pin
only. Acquisition, GPU load, and Gold inference stay behind their own grants.
Weights are hashed and copied in bounded chunks. Nothing downloads unless a
caller passes an explicit downloader after the acquisition grant.

The shared pieces are the shell renderer, context preflight, and TSV parser.
Qwen3 token ids, phase locks, and weight-index checks are not reused.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from re_te_system.controlled_experiment.campaign import render_controlled_prompt
from re_te_system.controlled_experiment.contracts import sha256_payload, sha256_text
from re_te_system.controlled_experiment.preflight import PreflightRequest, assess_preflight
from re_te_system.parsing.tsv import generation_issue, parse_tsv


MODEL_ID = "google/gemma-4-12B-it"
BASE_MODEL_ID = "google/gemma-4-12B"
CANDIDATE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
APPROVED_REVISIONS: frozenset[str] = frozenset({CANDIDATE_REVISION})
COPY_CHUNK_BYTES = 8 * 1024 * 1024
OFFICIAL_REVISION_FILES = (
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "generation_config.json",
    "model.safetensors",
)
ARCHITECTURE = "Gemma4UnifiedForConditionalGeneration"
FAMILY = "Gemma4-unified"
DECLARED_NATIVE_CONTEXT_LENGTH = 262144
DECLARED_PARAMETER_COUNT = 11_950_000_000
PROPOSED_DTYPE = "bfloat16"
ESTIMATED_BF16_WEIGHT_BYTES = DECLARED_PARAMETER_COUNT * 2
INCOMPATIBLE_INFERENCE_TRANSFORMERS = "4.57.6"
MINIMUM_TRANSFORMERS = (5, 10, 1)
SERIALIZATION_ID = "gemma4-12b-it-text-user-thinking-off-v1"
CHANNEL_POLICY_ID = "gemma4-12b-thought-channel-v1"
CHANNEL_OPEN = "<|channel>thought\n"
CHANNEL_CLOSE = "<channel|>"
PREFLIGHT_MAX_NEW_TOKENS = 32
INVENTED_PREFLIGHT_TEXT = (
    "A runtime preflight records one invented sentence that is not a statute."
)
G00_ENV = "GEMMA_G00_READINESS_AUTHORIZED"
G01_ENV = "GEMMA_G01_ACQUIRE_AUTHORIZED"
G02_ENV = "GEMMA_G02_PREFLIGHT_AUTHORIZED"
RESULT_ROLE = "gemma4_technical_preflight_not_scientific_f1"
GOLD_PATH_MARKERS = (
    "teresia_hohfeld",
    "canonical/documents.jsonl",
    "canonical/annotations.jsonl",
)


class GemmaRevisionNotApproved(RuntimeError):
    """The revision is not the single approved instruction-tuned pin."""


class GemmaModelIdentityError(RuntimeError):
    """The requested id is not the instruction-tuned 12B checkpoint."""


class GemmaAcquireNotAuthorized(RuntimeError):
    """G01 refused to download."""


class GemmaPreflightNotAuthorized(RuntimeError):
    """G02 refused to load weights."""


class GemmaGoldAccessRefused(RuntimeError):
    """A path or mode tried to read the Hohfeld cohort."""


class GemmaSilentTruncationRefused(RuntimeError):
    """Tokenization or generation asked to drop part of the prompt or answer."""


class GemmaContextBudgetError(RuntimeError):
    """The rendered prompt and reserved output do not fit the approved budget."""


class GemmaChannelError(RuntimeError):
    """Official channel markers were absent, repeated, or unclosed."""


class GemmaThinkingNotAuthorized(RuntimeError):
    """Controlled TSV serialization does not request a reasoning trace."""


class GemmaDuplicatePublication(RuntimeError):
    """An accepted checkpoint directory already exists."""


class GemmaWorkerTimeout(TimeoutError):
    """A bounded child process exceeded its budget and was stopped."""


class GemmaAllocationError(RuntimeError):
    """The SLURM allocation is missing or is not an explicit L40S."""


class GemmaWeightFormatError(RuntimeError):
    """The snapshot is not a Safetensors instruction-tuned checkpoint."""


@dataclass(frozen=True)
class GemmaDecodingContract:
    """Decoding choices. ``None`` means the choice is not approved.

    The preflight cap is an operational bound for one invented sentence.
    It is not a C0/C1 decoding freeze.
    """

    max_new_tokens: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    role: str = "unapproved"

    def __post_init__(self) -> None:
        if self.do_sample is True:
            raise ValueError("sampling is not authorized")
        if self.do_sample is False and self.temperature is not None:
            raise ValueError("greedy decoding does not accept temperature")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    def blockers(self) -> tuple[str, ...]:
        blocked = [
            "max_new_tokens_unapproved" if self.max_new_tokens is None else None,
            "decoding_unapproved" if self.do_sample is None else None,
        ]
        if self.role != "operational_preflight_not_scientific":
            blocked.append("decoding_role_not_operational_preflight")
            if self.eos_token_id is None:
                blocked.append("eos_token_id_unverified")
            if self.pad_token_id is None:
                blocked.append("pad_token_id_unverified")
        return tuple(item for item in blocked if item is not None)


def preflight_decoding_contract() -> GemmaDecodingContract:
    return GemmaDecodingContract(
        max_new_tokens=PREFLIGHT_MAX_NEW_TOKENS,
        do_sample=False,
        role="operational_preflight_not_scientific",
    )


def scientific_decoding_contract() -> GemmaDecodingContract:
    return GemmaDecodingContract()


def revision_is_approved(revision: str | None) -> bool:
    return isinstance(revision, str) and revision in APPROVED_REVISIONS


def assert_revision_approved(revision: str | None) -> None:
    if revision_is_approved(revision):
        return
    raise GemmaRevisionNotApproved(
        f"revision is not the approved instruction-tuned pin {CANDIDATE_REVISION}: {revision!r}"
    )


def assert_instruction_tuned(model_id: str) -> None:
    if model_id == BASE_MODEL_ID or not model_id.endswith("-it"):
        raise GemmaModelIdentityError(
            f"{model_id!r} is not {MODEL_ID}; the base checkpoint is refused"
        )
    if model_id != MODEL_ID:
        raise GemmaModelIdentityError(f"model id must be {MODEL_ID}, not {model_id!r}")


def refuse_gold_reference(value: str | Path | None) -> None:
    if value is None:
        return
    text = str(value).replace("\\", "/")
    if any(marker in text for marker in GOLD_PATH_MARKERS):
        raise GemmaGoldAccessRefused("G00-G02 cannot read Hohfeld Gold or its documents")


def grant_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def serialize_text_turn(rendered_prompt: str, *, enable_thinking: bool = False) -> str:
    """Published 12B text-turn pattern. It is not yet checked against the candidate template."""
    if not isinstance(rendered_prompt, str) or rendered_prompt == "":
        raise ValueError("rendered prompt must be a non-empty string")
    if enable_thinking:
        raise GemmaThinkingNotAuthorized(
            "controlled serialization keeps thinking off; the answer must be TSV"
        )
    return (
        "<|turn>user\n"
        + rendered_prompt
        + "<turn|>\n<|turn>model\n"
        + CHANNEL_OPEN
        + CHANNEL_CLOSE
    )


def render_serialized_turn(condition: str, text: str) -> str:
    refuse_gold_reference(text)
    return serialize_text_turn(render_controlled_prompt(condition, text), enable_thinking=False)


def separate_official_channels(text: str) -> dict[str, Any]:
    """Split one thought channel from the answer. Malformed output is not parsed."""
    if not isinstance(text, str):
        raise TypeError("model output must be a string")
    opens = text.count(CHANNEL_OPEN)
    closes = text.count(CHANNEL_CLOSE)
    other_channel = "<|channel>" in text.replace(CHANNEL_OPEN, "")
    if opens == 0 and closes == 0 and not other_channel:
        return {
            "answer": text,
            "channel_policy_id": CHANNEL_POLICY_ID,
            "status": "answer_only",
            "thought": None,
        }
    if opens != 1 or closes != 1 or other_channel:
        return _malformed_channel("channel markers are missing, repeated, or of another kind")
    start = text.find(CHANNEL_OPEN)
    end = text.find(CHANNEL_CLOSE)
    if end < start:
        return _malformed_channel("thought channel closes before it opens")
    answer = text[end + len(CHANNEL_CLOSE) :]
    if CHANNEL_OPEN in answer or CHANNEL_CLOSE in answer or "<|channel>" in answer:
        return _malformed_channel("answer still contains channel markers")
    return {
        "answer": answer,
        "channel_policy_id": CHANNEL_POLICY_ID,
        "status": "separated",
        "thought": text[start + len(CHANNEL_OPEN) : end],
    }


def transport_and_parse(
    text: str,
    *,
    segment_id: str | None = None,
    generation_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Parse only a separated answer. Invalid channels and invalid TSV yield no repaired triple."""
    separated = separate_official_channels(text)
    truncation = generation_issue(generation_metadata)
    if separated["status"] != "separated" and separated["status"] != "answer_only":
        return {
            "channel": separated,
            "issues": [{"code": "CHANNEL_NOT_SEPARATED", "message": separated["message"]}],
            "triples": [],
            "truncation": None if truncation is None else truncation.code,
        }
    if truncation is not None:
        return {
            "channel": separated,
            "issues": [{"code": truncation.code, "message": truncation.message}],
            "triples": [],
            "truncation": truncation.code,
        }
    parsed = parse_tsv(separated["answer"], segment_id=segment_id)
    return {
        "channel": {key: value for key, value in separated.items() if key != "thought"},
        "issues": [
            {"code": issue.code, "message": issue.message, "chunk_index": issue.chunk_index}
            for issue in parsed.issues
        ],
        "triples": [
            {"subject": item.subject, "relation": item.relation, "object": item.object}
            for item in parsed.triples
        ],
        "truncation": None,
    }


def count_tokens_untruncated(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
    if not isinstance(encoded, Mapping):
        raise GemmaSilentTruncationRefused("tokenizer did not return a mapping")
    if encoded.get("overflowing_tokens") or encoded.get("truncation") is True:
        raise GemmaSilentTruncationRefused("tokenizer reported truncated tokens")
    ids = encoded.get("input_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise GemmaSilentTruncationRefused("tokenizer input_ids are missing")
    return len(ids)


def assess_prompt_budget(
    prompt: str,
    *,
    count_tokens: Callable[[str], int] | None,
    effective_context_limit: int | None,
    reserved_output_tokens: int,
    tokenizer_id: str | None,
    tokenizer_revision: str | None,
    counter_id: str,
) -> dict[str, Any]:
    result = assess_preflight(
        PreflightRequest(
            prompt=prompt,
            count_tokens=count_tokens,
            effective_context_limit=effective_context_limit,
            reserved_output_tokens=reserved_output_tokens,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            serialization_id=SERIALIZATION_ID,
            counter_id=counter_id,
            measurement_scope="tokenizer_only",
        )
    )
    if result.get("fits") is not True:
        raise GemmaContextBudgetError(str(result.get("reason")))
    return result


def readiness_report() -> dict[str, Any]:
    """G00 facts. This does not reserve a GPU and does not approve the candidate."""
    installed = _installed_transformers()
    venv = os.environ.get("GEMMA_VENV")
    venv_python = Path(venv, "bin", "python") if venv else None
    isolated = _probe_isolated_runtime(venv_python)
    report = {
        "approved_revisions": sorted(APPROVED_REVISIONS),
        "base_model_id_refused": BASE_MODEL_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "candidate_revision_approved": revision_is_approved(CANDIDATE_REVISION),
        "channel_policy_id": CHANNEL_POLICY_ID,
        "channel_separator_verified": False,
        "chat_template_verified_against_candidate": False,
        "declared_native_context_length": DECLARED_NATIVE_CONTEXT_LENGTH,
        "declared_parameter_count": DECLARED_PARAMETER_COUNT,
        "dtype_policy": {
            "proposed": PROPOSED_DTYPE,
            "quantization": "refused",
            "status": "proposal_not_measurement",
        },
        "effective_context_limit_approved": None,
        "eos_token_id": None,
        "family": FAMILY,
        "gold_inference_authorized": False,
        "incompatible_inference_transformers": INCOMPATIBLE_INFERENCE_TRANSFORMERS,
        "installed_transformers": installed,
        "isolated_runtime": isolated,
        "minimum_transformers": ".".join(str(part) for part in MINIMUM_TRANSFORMERS),
        "model_id": MODEL_ID,
        "pad_token_id": None,
        "authorizes_later_stages": {"G01": False, "G02": False, "G03": False, "G04": False},
        "ready_for_gpu_reservation": False,
        "required_architecture": ARCHITECTURE,
        "result_role": RESULT_ROLE,
        "scientific_ready": False,
        "serialization_id": SERIALIZATION_ID,
        "stage": "G00",
        "vram_bytes_measured": None,
        "weight_estimate": {
            "bf16_weight_bytes": ESTIMATED_BF16_WEIGHT_BYTES,
            "parameter_count": DECLARED_PARAMETER_COUNT,
            "status": "estimate_not_measurement",
        },
    }
    report["runtime_blockers"] = _runtime_blockers(report)
    report["scientific_blockers"] = _scientific_blockers(report)
    report["blockers"] = list(report["runtime_blockers"])
    report["ready_for_gpu_reservation"] = report["runtime_blockers"] == []
    report["scientific_ready"] = report["scientific_blockers"] == []
    report["authorizes_later_stages"] = {"G01": False, "G02": False, "G03": False, "G04": False}
    return report


def acquisition_plan(model_id: str, revision: str) -> dict[str, Any]:
    assert_instruction_tuned(model_id)
    permitted = False
    refusal = None
    try:
        assert_revision_approved(revision)
        permitted = grant_enabled(G01_ENV)
        if not permitted:
            refusal = "acquire_grant_missing"
    except GemmaRevisionNotApproved as exc:
        refusal = str(exc)
    return {
        "acquire_permitted": permitted and refusal is None,
        "gold_inference_authorized": False,
        "local_files_only_during_inference": True,
        "model_id": model_id,
        "overwrite_accepted_checkpoint": False,
        "refusal": refusal,
        "revision": revision,
        "revision_approved": revision_is_approved(revision),
        "stage": "G01",
        "storage": acquisition_storage_plan(),
    }


def acquire_checkpoint(
    model_id: str,
    revision: str,
    destination: Path,
    downloader: Callable[[str, str], Path] | None = None,
) -> dict[str, Any]:
    plan = acquisition_plan(model_id, revision)
    if plan["acquire_permitted"] is not True:
        raise GemmaAcquireNotAuthorized(plan["refusal"] or "acquisition is closed")
    if downloader is None:
        raise GemmaAcquireNotAuthorized("the download callable is not bound")
    source = Path(downloader(model_id, revision))
    published = publish_snapshot(source, destination, revision)
    published["stage"] = "G01"
    return published


def acquisition_storage_plan() -> dict[str, Any]:
    """Disk plan for one G01 run. Figures are estimates, not measured downloads."""
    weight = ESTIMATED_BF16_WEIGHT_BYTES
    return {
        "cache_policy": "huggingface_hub blob cache may resume an incomplete download",
        "hardlinks": "not used",
        "peak_extra_bytes_estimate": weight * 2,
        "published_copy": "regular files, symlink targets are dereferenced",
        "resume_publish": "an incomplete staging directory is left in place and is not deleted",
        "same_filesystem_rename": True,
        "symlinks_in_published_tree": False,
        "weight_bytes_estimate": weight,
    }


def sha256_file(path: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> tuple[str, int]:
    """Hash one file without reading it into a single buffer."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def stream_copy(source: Path, destination: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> None:
    """Copy bytes in bounded blocks. The destination is a new regular file."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while True:
            block = incoming.read(chunk_bytes)
            if not block:
                break
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def verify_snapshot_directory(path: Path) -> dict[str, Any]:
    root = Path(path)
    files: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if relative.is_absolute() or ".." in relative.parts:
            raise GemmaWeightFormatError(f"snapshot path escapes its root: {relative}")
        digest, size = sha256_file(item)
        files.append(
            {
                "byte_size": size,
                "relative_path": relative.as_posix(),
                "sha256": digest,
            }
        )
    errors: list[str] = []
    config_path = root / "config.json"
    if not config_path.is_file():
        errors.append("config.json missing")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("config.json is not valid JSON")
            config = {}
        architectures = config.get("architectures") or []
        if ARCHITECTURE not in architectures:
            errors.append(f"architectures do not include {ARCHITECTURE}")
        recorded_name = str(config.get("_name_or_path") or "")
        if recorded_name == BASE_MODEL_ID or recorded_name.endswith("/" + BASE_MODEL_ID):
            errors.append("config names the base checkpoint")
    if not (root / "tokenizer.json").is_file() and not (root / "tokenizer_config.json").is_file():
        errors.append("tokenizer files missing")
    has_safetensors = (root / "model.safetensors").is_file() or (
        root / "model.safetensors.index.json"
    ).is_file()
    if not has_safetensors:
        errors.append("safetensors weights missing")
    if any(item.suffix == ".bin" for item in root.glob("*.bin")):
        errors.append("pytorch_model.bin is not an authorized load format")
    template_present = (root / "chat_template.jinja").is_file()
    tokenizer_config = root / "tokenizer_config.json"
    if not template_present and tokenizer_config.is_file():
        try:
            payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        template_present = bool(payload.get("chat_template"))
    if not template_present:
        errors.append("chat template missing")
    return {
        "errors": errors,
        "files": files,
        "verified": not errors,
    }


def official_layout_errors(root: Path) -> list[str]:
    """Files observed on the approved Hub revision. Absence blocks publication of that pin."""
    return [name for name in OFFICIAL_REVISION_FILES if not (root / name).is_file()]


def publish_snapshot(source: Path, destination: Path, revision: str) -> dict[str, Any]:
    assert_revision_approved(revision)
    origin = Path(source)
    verification = verify_snapshot_directory(origin)
    if verification["verified"] is not True:
        raise GemmaWeightFormatError("; ".join(verification["errors"]))
    if revision == CANDIDATE_REVISION:
        missing = official_layout_errors(origin)
        if missing:
            raise GemmaWeightFormatError("approved revision is missing " + ", ".join(missing))
    target = Path(destination)
    parent = target.parent
    marker = parent / "accepted_revision.txt"
    staging = parent / f".{target.name}.partial"
    if marker.exists() or target.exists():
        raise GemmaDuplicatePublication(
            f"refusing to overwrite an accepted Gemma checkpoint at {target}"
        )
    if staging.exists():
        raise GemmaDuplicatePublication(
            f"incomplete publication remains at {staging} and is not deleted automatically"
        )
    parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    for record in verification["files"]:
        relative = Path(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise GemmaWeightFormatError(f"refusing to copy {relative}")
        src = origin / relative
        dest = staging / relative
        if staging.resolve() not in dest.resolve().parents:
            raise GemmaWeightFormatError(f"copy target escapes staging: {dest}")
        stream_copy(src, dest)
    published = verify_snapshot_directory(staging)
    if not _same_file_records(verification["files"], published["files"]):
        raise GemmaWeightFormatError("published bytes do not match the source snapshot")
    _fsync_directory(staging)
    os.rename(staging, target)
    _fsync_directory(parent)
    _write_acceptance_marker(marker, revision)
    _fsync_directory(parent)
    return {
        "accepted_revision_file": str(marker),
        "destination": str(target),
        "duplicate_publication": False,
        "revision": revision,
        "storage": acquisition_storage_plan(),
        "verification": verification,
    }


def _same_file_records(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    def key(record: Mapping[str, Any]) -> tuple[str, str, int]:
        return (str(record["relative_path"]), str(record["sha256"]), int(record["byte_size"]))

    return sorted(key(item) for item in left) == sorted(key(item) for item in right)


def _fsync_directory(directory: Path) -> None:
    """Durability of a directory entry on the destination filesystem."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_acceptance_marker(marker: Path, revision: str) -> None:
    temporary = marker.with_name(marker.name + ".partial")
    if marker.exists() or temporary.exists():
        raise GemmaDuplicatePublication(f"refusing to replace acceptance marker {marker}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(revision + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)


def preflight_plan() -> dict[str, Any]:
    decoding = preflight_decoding_contract()
    serialized = serialize_text_turn(INVENTED_PREFLIGHT_TEXT, enable_thinking=False)
    return {
        "decoding": {
            "blockers": list(decoding.blockers()),
            "do_sample": decoding.do_sample,
            "eos_token_id": decoding.eos_token_id,
            "max_new_tokens": decoding.max_new_tokens,
            "eos_token_id_is_scientific_pin": False,
            "pad_token_id": decoding.pad_token_id,
            "pad_token_id_is_scientific_pin": False,
            "role": decoding.role,
        },
        "fixture": {
            "kind": "invented_sentence",
            "sha256": sha256_text(INVENTED_PREFLIGHT_TEXT),
            "uses_gold": False,
        },
        "gold_inference_authorized": False,
        "local_files_only": True,
        "quantization": "refused",
        "revision_approved": revision_is_approved(CANDIDATE_REVISION),
        "serialized_sha256": sha256_text(serialized),
        "stage": "G02",
        "weights_loaded": False,
    }


def execute_bounded_preflight(
    *,
    revision: str,
    allocation_text: str,
    loader: Callable[..., Any],
    snapshot_dir: Path,
) -> dict[str, Any]:
    if not grant_enabled(G02_ENV):
        raise GemmaPreflightNotAuthorized("G02 grant is missing")
    assert_revision_approved(revision)
    _assert_l40s_allocation(allocation_text)
    verification = verify_snapshot_directory(snapshot_dir)
    if verification["verified"] is not True:
        raise GemmaWeightFormatError("; ".join(verification["errors"]))
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise GemmaPreflightNotAuthorized("GPU preflight requires HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE")
    loaded = loader(
        model_id=MODEL_ID,
        revision=revision,
        local_files_only=True,
        dtype=PROPOSED_DTYPE,
        quantization=None,
    )
    report = preflight_plan()
    report["weights_loaded"] = True
    report["revision_approved"] = True
    report["load"] = {
        "dtype": PROPOSED_DTYPE,
        "local_files_only": True,
        "observed": loaded,
        "quantization": None,
    }
    return report


def run_bounded_command(argv: Sequence[str], timeout_seconds: float) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate(process)
        raise GemmaWorkerTimeout(
            f"child exceeded {timeout_seconds} seconds: {' '.join(argv)}"
        ) from exc
    if process.returncode != 0:
        raise RuntimeError(stderr.strip() or f"child exited {process.returncode}")
    return {"returncode": process.returncode, "stderr": stderr, "stdout": stdout}


def classify_runtime_failure(exc: BaseException) -> dict[str, str]:
    text = str(exc).lower()
    if "out of memory" in text or "cuda error" in text:
        kind = "external_infrastructure_error"
    elif isinstance(exc, GemmaWorkerTimeout):
        kind = "timeout"
    else:
        kind = "model_execution_error"
    return {"kind": kind, "message": str(exc), "parsed_as_triple": "false"}


def technical_manifest(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "gold_inference_authorized": False,
        "model_id": MODEL_ID,
        "report": dict(report),
        "result_role": RESULT_ROLE,
        "runtime": {
            "incompatible_inference_transformers": INCOMPATIBLE_INFERENCE_TRANSFORMERS,
            "minimum_transformers": ".".join(str(part) for part in MINIMUM_TRANSFORMERS),
            "proposed_dtype": PROPOSED_DTYPE,
            "python": sys.version.split()[0],
        },
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload


def _malformed_channel(message: str) -> dict[str, Any]:
    return {
        "answer": None,
        "channel_policy_id": CHANNEL_POLICY_ID,
        "message": message,
        "status": "malformed",
        "thought": None,
    }


def _installed_transformers() -> dict[str, Any]:
    try:
        import importlib.metadata as importlib_metadata

        version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        return {"compatible": False, "status": "not_installed_in_this_interpreter"}
    except Exception as exc:
        return {"compatible": False, "status": f"not_inspected: {exc}"}
    return {
        "compatible": _version_at_least(version, MINIMUM_TRANSFORMERS),
        "status": "observed",
        "version": version,
    }


def _probe_isolated_runtime(python_path: Path | None) -> dict[str, Any]:
    if python_path is None or not python_path.is_file():
        return {"status": "venv_absent", "class_importable": False}
    code = (
        "import importlib.metadata as metadata, importlib\n"
        "version = metadata.version('transformers')\n"
        "importlib.import_module('transformers').Gemma4UnifiedForConditionalGeneration\n"
        "print(version)\n"
    )
    completed = subprocess.run(
        [str(python_path), "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    version = completed.stdout.strip()
    return {
        "class_importable": completed.returncode == 0,
        "returncode": completed.returncode,
        "status": "probed",
        "transformers": version or None,
        "compatible": completed.returncode == 0 and _version_at_least(version, MINIMUM_TRANSFORMERS),
    }


def _runtime_blockers(report: Mapping[str, Any]) -> list[str]:
    """CPU compatibility of GEMMA_VENV. Scientific gaps are not in this list."""
    isolated = report["isolated_runtime"]
    if isolated.get("status") == "venv_absent":
        return ["gemma_venv_absent"]
    if isolated.get("compatible") is True:
        return []
    if isolated.get("status") == "probed" and isolated.get("class_importable") is not True:
        return ["gemma4_unified_class_not_importable"]
    if isolated.get("status") == "probed":
        return ["isolated_transformers_incompatible"]
    return ["gemma4_unified_runtime_not_available"]


def _scientific_blockers(report: Mapping[str, Any]) -> list[str]:
    """Facts G00 cannot establish. They do not decide ready_for_gpu_reservation."""
    blockers: list[str] = []
    if report.get("candidate_revision_approved") is not True:
        blockers.append("candidate_revision_not_approved")
    if report.get("chat_template_verified_against_candidate") is not True:
        blockers.append("chat_template_not_verified_against_candidate")
    if report.get("channel_separator_verified") is not True:
        blockers.append("channel_separator_unverified")
    if report.get("eos_token_id") is None or report.get("pad_token_id") is None:
        blockers.append("eos_pad_ids_unverified")
    if report.get("vram_bytes_measured") is None:
        blockers.append("vram_unverified")
    return blockers


def _version_at_least(version: str, minimum: tuple[int, int, int]) -> bool:
    numbers: list[int] = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3]) >= minimum


def _assert_l40s_allocation(allocation_text: str) -> None:
    lowered = allocation_text.lower()
    if "a100" in lowered:
        raise GemmaAllocationError("A100 is refused")
    if "l40s" not in lowered:
        raise GemmaAllocationError("G02 requires an explicit gpu:l40s allocation")
    if "gpu:1" in lowered and "l40s" not in lowered:
        raise GemmaAllocationError("generic gpu:1 allocation is refused")


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


# G03-G05. Gold execution stays closed. These stages do not approve a revision
# and do not read Hohfeld annotations.

G03_ENV = "GEMMA_G03_SMOKE_AUTHORIZED"
G04_ENV = "GEMMA_G04_RUN_AUTHORIZED"
G04_PHASE_LOCK = True
PAIRED_CONDITIONS = ("C0", "C1")
EXPECTED_DOCUMENT_COUNT = 36
DATASET_ID = "teresia_hohfeld"
RETRYABLE_STATES = frozenset({"timeout", "external_infrastructure_error"})
MAX_ATTEMPTS_PER_DOCUMENT = 2
ESTIMAND_NAMES = {
    "conservative": "conservative_operational_f1",
    "conditional": "f1_conditional_on_completed_generation",
    "paired": "paired_secondary_f1",
}
INVENTED_SMOKE_DOCUMENTS = (
    {
        "document_id": "invented-gemma-01",
        "text": "Ava notifies Blake about one invented duty.",
    },
    {
        "document_id": "invented-gemma-02",
        "text": "Casey records one invented privilege for Drew.",
    },
)
ANNOTATION_KEYS = frozenset({"annotations", "eligibility", "annotation_id", "e1", "e2"})


class GemmaGoldNotAuthorized(RuntimeError):
    """G04 refused to start. The phase lock is one of the reasons, not the only check."""


def paired_condition_identity(text: str) -> dict[str, Any]:
    """Show that C0 and C1 share the shell and differ only by the frozen slot."""
    refuse_gold_reference(text)
    from re_te_system.controlled_experiment.dependencies import ensure_package

    ensure_package("re_te_benchmark", "re_te_benchmark")
    from re_te_benchmark.conditioning.c2 import load_c1_labels
    from re_te_benchmark.conditioning.shell import SHELL_ID, conditioning_for

    if SHELL_ID != "common-instruction-shell-v2":
        raise GemmaModelIdentityError(f"shell is {SHELL_ID}")
    labels = list(load_c1_labels())
    if labels != ["Duty", "Right", "Privilege", "NoRight"]:
        raise GemmaModelIdentityError("C1 inventory is not the frozen Hohfeld list")
    c0_slot = conditioning_for("C0")
    c1_slot = conditioning_for("C1")
    if c0_slot != "":
        raise GemmaModelIdentityError("C0 slot is not empty")
    if c1_slot not in render_controlled_prompt("C1", text):
        raise GemmaModelIdentityError("C1 prompt does not contain its slot")
    c0_prompt = render_controlled_prompt("C0", text)
    c1_prompt = render_controlled_prompt("C1", text)
    if c1_prompt.replace(c1_slot, "", 1) != c0_prompt:
        raise GemmaModelIdentityError("C0 and C1 differ by more than the condition slot")
    decoding = preflight_decoding_contract()
    return {
        "c0_prompt_sha256": sha256_text(c0_prompt),
        "c0_serialized_sha256": sha256_text(serialize_text_turn(c0_prompt)),
        "c0_slot_sha256": sha256_text(c0_slot),
        "c1_prompt_sha256": sha256_text(c1_prompt),
        "c1_serialized_sha256": sha256_text(serialize_text_turn(c1_prompt)),
        "c1_slot_sha256": sha256_text(c1_slot),
        "decoding_sha256": sha256_payload(
            {
                "do_sample": decoding.do_sample,
                "max_new_tokens": decoding.max_new_tokens,
                "role": decoding.role,
            }
        ),
        "labels": labels,
        "same_decoding_for_both_conditions": True,
        "shell_id": SHELL_ID,
    }


def same_retry_policy(condition: str, state: str, attempt_index: int) -> bool:
    """One retry for infrastructure or timeout. The condition does not change it."""
    if condition not in PAIRED_CONDITIONS:
        raise GemmaModelIdentityError(f"retry policy has no condition {condition}")
    return attempt_index == 1 and state in RETRYABLE_STATES


def accept_document_records(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Order and identity only. Annotation-shaped records are refused."""
    if not records:
        raise GemmaGoldAccessRefused("a cohort needs documents")
    identities: list[str] = []
    for record in records:
        if ANNOTATION_KEYS.intersection(record):
            raise GemmaGoldAccessRefused("annotations cannot be generation input")
        document_id = record.get("document_id")
        text = record.get("text")
        if not isinstance(document_id, str) or not document_id:
            raise GemmaGoldAccessRefused("document id is missing")
        if not isinstance(text, str) or not text.strip():
            raise GemmaGoldAccessRefused("document text is missing")
        refuse_gold_reference(text)
        identities.append(document_id)
    if len(identities) != len(set(identities)):
        raise GemmaGoldAccessRefused("document ids are duplicated")
    return identities


def assert_paired_cohort(c0_ids: Sequence[str], c1_ids: Sequence[str]) -> None:
    if list(c0_ids) != list(c1_ids):
        raise GemmaGoldAccessRefused("C0 and C1 do not name the same documents in the same order")


def classify_smoke_output(
    raw: str | None,
    *,
    state: str = "completed",
    truncated: bool = False,
) -> dict[str, Any]:
    if state == "timeout":
        return _failure_result("timeout", "the generation exceeded its budget")
    if state == "external_infrastructure_error":
        return _failure_result("external_infrastructure_error", "the runtime reported an infrastructure error")
    if raw is None or raw == "":
        return _failure_result("empty_model_output", "the model returned no text")
    parsed = transport_and_parse(
        raw,
        generation_metadata={"output_reached_limit": True} if truncated else None,
    )
    valid = parsed["channel"]["status"] in {"answer_only", "separated"} and not parsed["issues"]
    return {
        "failure": None if valid else {
            "kind": parsed["issues"][0]["code"] if parsed["issues"] else parsed["channel"]["status"],
            "message": parsed["issues"][0]["message"] if parsed["issues"] else parsed["channel"].get("message"),
        },
        "parsed": parsed,
        "principal_state": "completed" if state == "completed" and not truncated else state,
        "raw": raw,
        "valid_prediction": valid and not truncated and state == "completed",
    }


def smoke_requests() -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for document in INVENTED_SMOKE_DOCUMENTS:
        identity = paired_condition_identity(document["text"])
        for condition in PAIRED_CONDITIONS:
            rendered = render_controlled_prompt(condition, document["text"])
            serialized = serialize_text_turn(rendered)
            requests.append(
                {
                    "condition": condition,
                    "document_id": document["document_id"],
                    "rendered_prompt_sha256": sha256_text(rendered),
                    "serialized_sha256": sha256_text(serialized),
                    "text_sha256": sha256_text(document["text"]),
                    "uses_gold": False,
                }
            )
        if identity["c0_prompt_sha256"] == identity["c1_prompt_sha256"]:
            raise GemmaModelIdentityError("C0 and C1 prompts are identical")
    return requests


def smoke_technical_pass(attempts: Sequence[Mapping[str, Any]], load_count: int) -> bool:
    """Technical smoke only. Parse errors and P0 scores do not make it scientific."""
    expected = {
        (item["document_id"], condition)
        for item in INVENTED_SMOKE_DOCUMENTS
        for condition in PAIRED_CONDITIONS
    }
    if load_count != 1 or len(attempts) != len(expected):
        return False
    seen: set[tuple[str, str]] = set()
    for attempt in attempts:
        key = (str(attempt.get("document_id")), str(attempt.get("condition")))
        if key not in expected or key in seen:
            return False
        seen.add(key)
        if attempt.get("exception") or attempt.get("within_budget") is not True:
            return False
        if attempt.get("load_count") not in (None, 1):
            return False
    return seen == expected


def write_smoke_artifacts(
    directory: Path,
    attempts: Sequence[Mapping[str, Any]],
    *,
    load_count: int,
) -> dict[str, Any]:
    target = Path(directory)
    if (target / "manifest.json").exists():
        raise GemmaDuplicatePublication(f"refusing to overwrite {target / 'manifest.json'}")
    target.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for attempt in attempts:
        classified = attempt.get("classified")
        if not isinstance(classified, Mapping):
            classified = classify_smoke_output(
                attempt.get("raw") if isinstance(attempt.get("raw"), str) else None,
                state=str(attempt.get("state") or "completed"),
                truncated=attempt.get("truncated") is True,
            )
        row = {
            "condition": attempt["condition"],
            "document_id": attempt["document_id"],
            "parsed_triples": classified["parsed"]["triples"] if classified.get("parsed") else [],
            "parse_issues": classified["parsed"]["issues"] if classified.get("parsed") else [],
            "raw": classified.get("raw"),
            "rendered_prompt_sha256": attempt.get("rendered_prompt_sha256"),
            "serialized_sha256": attempt.get("serialized_sha256"),
            "valid_prediction": classified["valid_prediction"],
        }
        predictions.append(row)
        if classified["failure"] is not None or classified["valid_prediction"] is not True:
            failures.append(
                {
                    "condition": attempt["condition"],
                    "document_id": attempt["document_id"],
                    "kind": (classified["failure"] or {}).get("kind", classified["principal_state"]),
                    "message": (classified["failure"] or {}).get("message", ""),
                }
            )
    passed = smoke_technical_pass(attempts, load_count)
    statistics = {
        "g04_authorized": False,
        "inputs_failed": len({(item["document_id"], item["condition"]) for item in failures}),
        "inputs_succeeded": len(predictions) - len(failures),
        "load_count": load_count,
        "parse_failures": sum(1 for item in failures if item["kind"] not in RETRYABLE_STATES),
        "scientific_result": False,
        "technical_pass": passed,
    }
    manifest = technical_manifest(
        {
            "conditions": list(PAIRED_CONDITIONS),
            "decoding_sha256": paired_condition_identity(INVENTED_SMOKE_DOCUMENTS[0]["text"])[
                "decoding_sha256"
            ],
            "g04_authorized": False,
            "model_id": MODEL_ID,
            "result_role": "gemma4_invented_smoke_not_scientific_f1",
            "revision_approved": False,
            "scientific_result": False,
            "shell_id": "common-instruction-shell-v2",
            "stage": "G03",
            "technical_pass": passed,
        }
    )
    _write_json(target / "manifest.json", manifest)
    _write_jsonl(target / "predictions.jsonl", predictions)
    _write_jsonl(target / "failures.jsonl", failures)
    _write_json(target / "statistics.json", statistics)
    for condition in PAIRED_CONDITIONS:
        condition_dir = target / "conditions" / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        selected = [row for row in predictions if row["condition"] == condition]
        _write_jsonl(condition_dir / "predictions.jsonl", selected)
        _write_jsonl(
            condition_dir / "evaluation_predictions.jsonl",
            [
                {
                    "document_id": row["document_id"],
                    "id": f"g03:{condition}:{row['document_id']}:{index:06d}",
                    "object": triple["object"],
                    "relation": triple["relation"],
                    "subject": triple["subject"],
                }
                for row in selected
                if row["valid_prediction"] is True
                for index, triple in enumerate(row["parsed_triples"])
            ],
        )
        _write_jsonl(condition_dir / "withheld_predictions.jsonl", [])
    return {"manifest": manifest, "statistics": statistics}


def persist_g03(directory: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Write the smoke contract. A loader result cannot authorize G04."""
    if result.get("g04_authorized") is True or result.get("scientific_result") is True:
        raise GemmaGoldNotAuthorized("G03 cannot authorize G04 or a scientific result")
    if result.get("load_count") != 1:
        raise GemmaPreflightNotAuthorized("G03 requires one weight load for C0 and C1")
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise GemmaPreflightNotAuthorized("G03 loader did not return attempts")
    written = write_smoke_artifacts(directory, attempts, load_count=1)
    if written["statistics"]["g04_authorized"] is not False:
        raise GemmaGoldNotAuthorized("smoke statistics authorized G04")
    return written


def g04_blockers(evidence: Mapping[str, Any]) -> list[str]:
    """Every missing gate is reported. An environment variable does not replace them."""
    blockers: list[str] = []
    if evidence.get("revision_approved") is not True:
        blockers.append("checkpoint_identity_not_approved")
    if evidence.get("snapshot_verified") is not True:
        blockers.append("snapshot_integrity_missing")
    if evidence.get("runtime_compatible") is not True:
        blockers.append("runtime_not_compatible")
    g02 = evidence.get("g02_report") if isinstance(evidence.get("g02_report"), Mapping) else {}
    if g02.get("status") != "PREFLIGHT_PASS":
        blockers.append("g02_preflight_not_accredited")
    g03 = evidence.get("g03_report") if isinstance(evidence.get("g03_report"), Mapping) else {}
    if g03.get("technical_pass") is not True:
        blockers.append("g03_smoke_not_accepted")
    if g03.get("scientific_result") is True:
        blockers.append("g03_must_not_be_a_scientific_result")
    if g03.get("g04_authorized") is True:
        blockers.append("g03_cannot_authorize_g04")
    decoding = evidence.get("frozen_decoding") if isinstance(evidence.get("frozen_decoding"), Mapping) else {}
    if not decoding.get("c0") or decoding.get("c0") != decoding.get("c1"):
        blockers.append("generation_policy_not_frozen")
    human = evidence.get("human_authorization") if isinstance(evidence.get("human_authorization"), Mapping) else {}
    if (
        human.get("decision") != "authorize_g04"
        or not isinstance(human.get("actor"), str)
        or not human.get("actor")
        or not revision_is_approved(human.get("revision") if isinstance(human.get("revision"), str) else None)
    ):
        blockers.append("human_authorization_missing")
    if not grant_enabled(G04_ENV):
        blockers.append("execution_grant_missing")
    if evidence.get("output_safe") is not True:
        blockers.append("output_directory_unsafe")
    if G04_PHASE_LOCK is not False:
        blockers.append("g04_phase_lock_closed")
    return blockers


def launch_g04(evidence: Mapping[str, Any], loader: Callable[..., Any] | None = None) -> dict[str, Any]:
    blockers = g04_blockers(evidence)
    if loader is not None:
        blockers.append("loader_was_offered_to_a_closed_gate")
    if blockers:
        raise GemmaGoldNotAuthorized("; ".join(blockers))
    raise GemmaGoldNotAuthorized("g04_phase_lock_closed")


def scientific_plan() -> dict[str, Any]:
    """Identifiers for a future 36-document run. This does not open the corpus."""
    sample = paired_condition_identity(INVENTED_SMOKE_DOCUMENTS[0]["text"])
    return {
        "dataset_id": DATASET_ID,
        "document_count_expected": EXPECTED_DOCUMENT_COUNT,
        "documents_read": False,
        "estimands": dict(ESTIMAND_NAMES),
        "execution_permitted": False,
        "gold_annotations_used_in_prompts": False,
        "gold_inference_authorized": False,
        "same_checkpoint_and_decoding_for_c0_and_c1": True,
        "shell_identity": {
            "c0_slot_sha256": sample["c0_slot_sha256"],
            "c1_slot_sha256": sample["c1_slot_sha256"],
            "decoding_sha256": sample["decoding_sha256"],
            "shell_id": sample["shell_id"],
        },
        "stage": "G04",
        "training": False,
    }


def score_levels(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """P0 predicate and triplet scores. The numbers are not a scientific promotion."""
    from re_te_system.controlled_experiment.dependencies import ensure_package

    ensure_package("re_te_evaluation", "re_te_evaluation")
    from re_te_evaluation import (
        ContextMode,
        EligibilityPolicy,
        EvaluationConfig,
        MetricFamily,
        evaluate_predicates,
        evaluate_triplets,
    )

    shared = {
        "context_mode": ContextMode.DOCUMENT,
        "eligibility_policy": EligibilityPolicy.STRICT_INLINE,
    }
    relation = evaluate_predicates(
        gold_records,
        prediction_records,
        EvaluationConfig(MetricFamily.PREDICATE, **shared),
    )
    triplet = evaluate_triplets(
        gold_records,
        prediction_records,
        EvaluationConfig(MetricFamily.TRIPLET, **shared),
    )
    return {
        "comparability": "not_scientific_in_this_checkout",
        "experimental_f1": False,
        "relation": _level_view(relation.summary),
        "triplet": _level_view(triplet.summary),
    }


def evaluation_table(
    model: str,
    condition: str,
    levels: Mapping[str, Any],
    *,
    estimand: str,
    completeness: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    if estimand not in ESTIMAND_NAMES:
        raise GemmaModelIdentityError(f"unknown estimand {estimand}")
    relation = levels["relation"]
    triplet = levels["triplet"]
    return {
        "comparability": levels["comparability"],
        "completeness": completeness,
        "condition": condition,
        "coverage": coverage,
        "estimand": ESTIMAND_NAMES[estimand],
        "experimental_f1": False,
        "model": model,
        "relation_f1": relation["f1"],
        "relation_fn": relation["fn"],
        "relation_fp": relation["fp"],
        "relation_precision": relation["precision"],
        "relation_recall": relation["recall"],
        "relation_tp": relation["tp"],
        "triplet_f1": triplet["f1"],
        "triplet_fn": triplet["fn"],
        "triplet_fp": triplet["fp"],
        "triplet_precision": triplet["precision"],
        "triplet_recall": triplet["recall"],
        "triplet_tp": triplet["tp"],
    }


def contrast_c1_minus_c0(c0: Mapping[str, Any], c1: Mapping[str, Any]) -> dict[str, Any]:
    """Paired contrast stays withheld. One row, or two rows of the same condition, is not a pair."""
    if c0.get("condition") != "C0" or c1.get("condition") != "C1":
        return {
            "published": False,
            "reason": "a paired contrast requires one C0 row and one C1 row",
        }
    comparable = (
        c0.get("model") == c1.get("model")
        and c0.get("estimand") == c1.get("estimand")
        and c0.get("completeness") == "operationally_complete_publishable"
        and c1.get("completeness") == "operationally_complete_publishable"
        and c0.get("comparability") == "scientific_comparable"
        and c1.get("comparability") == "scientific_comparable"
        and c0.get("experimental_f1") is True
        and c1.get("experimental_f1") is True
    )
    if not comparable:
        return {
            "published": False,
            "reason": "paired contrast is withheld until both conditions are scientific and complete",
        }
    return {"published": False, "reason": "this checkout cannot publish a scientific contrast"}


def _failure_result(kind: str, message: str) -> dict[str, Any]:
    return {
        "failure": {"kind": kind, "message": message},
        "parsed": {"channel": {"status": "not_parsed"}, "issues": [], "triples": []},
        "principal_state": kind,
        "raw": None,
        "valid_prediction": False,
    }


def _level_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    micro = summary["micro"]
    counts = summary["counts"]
    return {
        "f1": micro["f1"],
        "fn": counts["fn"],
        "fp": counts["fp"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "tp": counts["tp"],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
