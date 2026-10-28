"""Synthetic GoLLIE cache compatibility. No GPU and no real weights."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import transformers
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from re_te_system.contracts import InputRecord
from re_te_system.extractors.gollie import (
    ACCREDITED_TRANSFORMERS,
    CACHE_ADAPTER_ID,
    declare_legacy_cache_only,
)
from re_te_system.extractors.mock import MockExtractor
from re_te_system.runner.run import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
MODELING = (
    ROOT.parent
    / "_legacy"
    / "GoLLIE"
    / "src"
    / "model"
    / "patch_models"
    / "modeling_flash_llama.py"
)
sys.path.insert(0, str(SCRIPTS))

import j17_job_paths  # noqa: E402
from flash_attn_pin import EXPECTED_TRANSFORMERS  # noqa: E402


DATASET = {"hash": "abc", "id": "test", "version": "1"}
SOURCE = {"documents_sha256": "abc", "manifest_sha256": "def"}
GENERATION = {
    "do_sample": False,
    "max_new_tokens": None,
    "num_beams": 1,
    "seed": None,
}
PROMPT = [4, 5, 6, 8]
NEW_TOKENS = 3


class ProbeConfig(PretrainedConfig):
    model_type = "gollie-cache-probe"

    def __init__(self, vocab_size: int = 64, **kwargs: object) -> None:
        super().__init__(pad_token_id=0, bos_token_id=1, eos_token_id=2, **kwargs)
        self.vocab_size = vocab_size
        self.num_hidden_layers = 1
        self.hidden_size = 4
        self.is_encoder_decoder = False
        self.use_cache = True


class LegacyCacheProbe(PreTrainedModel, GenerationMixin):
    """Minimal stand-in for the pinned GoLLIE cache contract."""

    config_class = ProbeConfig

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.marker = nn.Parameter(torch.zeros(1))
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        is_padded_inputs: bool | None = None,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise AssertionError("the probe requires input_ids")
        if past_key_values is not None:
            if past_key_values[0][0].shape[2] != 2:
                raise AssertionError("legacy kv axis 2 is the k/v stack")
            past_kv, past_len = past_key_values[0]
            history = past_kv[0, :past_len, 0, 0, 0].to(dtype=torch.long)
            seen = torch.cat([history, input_ids[0]], dim=0)
        else:
            past_len = 0
            seen = input_ids[0]
        self.calls.append(
            {
                "attention_width": None if attention_mask is None else int(attention_mask.shape[-1]),
                "input_ids": input_ids.detach().cpu().clone(),
                "past_key_values": past_key_values,
                "seen": seen.detach().cpu().clone(),
            }
        )
        total = int(seen.sum().item()) % self.config.vocab_size
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -1.0e4,
            dtype=torch.float32,
        )
        logits[:, -1, total] = 0.0
        new_len = past_len + int(input_ids.shape[-1])
        kv = torch.zeros(input_ids.shape[0], new_len, 2, 1, 1, dtype=torch.float32)
        kv[:, :, 0, 0, 0] = seen.to(dtype=kv.dtype)
        kv[:, :, 1, 0, 0] = seen.to(dtype=kv.dtype)
        past = ((kv, new_len),) if use_cache is not False else None
        return CausalLMOutputWithPast(logits=logits, past_key_values=past)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids", None)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs: dict[str, object] = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "is_padded_inputs": (attention_mask is not None)
                and (not attention_mask.all().item()),
            }
        )
        return model_inputs


def _oracle(prompt: list[int], steps: int, vocab: int) -> list[int]:
    seen = list(prompt)
    produced = []
    for _ in range(steps):
        nxt = sum(seen) % vocab
        produced.append(nxt)
        seen.append(nxt)
    return produced


def _generate(model: LegacyCacheProbe, prompt: list[int]) -> torch.Tensor:
    tensor = torch.tensor([prompt], dtype=torch.long)
    attention = torch.ones_like(tensor)
    generated = model.generate(
        input_ids=tensor,
        attention_mask=attention,
        do_sample=False,
        max_new_tokens=NEW_TOKENS,
        min_new_tokens=0,
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=True,
    )
    return generated.sequences


def test_accredited_transformers_pin_matches_the_isolated_runtime() -> None:
    assert ACCREDITED_TRANSFORMERS == EXPECTED_TRANSFORMERS == "4.57.6"
    payload = MODELING.read_bytes()
    assert len(payload) == 46523
    assert hashlib.sha1(b"blob %d\0" % len(payload) + payload).hexdigest() == (
        "1330bc1ca5c6769f0880555350cfd33ea5ea4fbd"
    )
    text = payload.decode("utf-8")
    assert "past_key_values_length = past_key_values[0][0].shape[2]" in text
    assert "input_ids = input_ids[:, -1:]" in text


def test_unadapted_probe_reproduces_the_empty_dynamic_cache_failure() -> None:
    model = LegacyCacheProbe(ProbeConfig())
    assert model._supports_default_dynamic_cache() is True
    with pytest.raises(AttributeError, match="shape"):
        _generate(model, PROMPT)


def test_legacy_cache_keeps_the_prompt_and_the_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transformers, "__version__", "4.58.0")
    with pytest.raises(RuntimeError, match="4.57.6"):
        declare_legacy_cache_only(LegacyCacheProbe(ProbeConfig()))
    monkeypatch.setattr(transformers, "__version__", ACCREDITED_TRANSFORMERS)

    refused = LegacyCacheProbe(ProbeConfig())
    refused.generation_config.cache_implementation = "static"
    with pytest.raises(RuntimeError, match="cache_implementation"):
        declare_legacy_cache_only(refused)
    refused.generation_config.cache_implementation = None
    refused.generation_config.use_cache = False
    with pytest.raises(RuntimeError, match="use_cache"):
        declare_legacy_cache_only(refused)

    original = LegacyCacheProbe(ProbeConfig())
    model = LegacyCacheProbe(ProbeConfig())
    record = declare_legacy_cache_only(model)
    assert record["adapter"] == CACHE_ADAPTER_ID
    assert record["dynamic_cache"] is False
    assert record["use_cache"] is True
    assert model._supports_default_dynamic_cache() is False
    assert model._is_stateful is False
    assert original._supports_default_dynamic_cache() is True
    assert type(original)._supports_default_dynamic_cache() is True
    assert "GoLLIELegacyCacheModel" not in GenerationMixin.__dict__

    adapted_type = type(model)
    assert declare_legacy_cache_only(model) == record
    assert type(model) is adapted_type
    sequences = _generate(model, PROMPT)
    calls = model.calls
    assert len(calls) == NEW_TOKENS
    assert calls[0]["past_key_values"] is None
    assert calls[0]["input_ids"].tolist() == [PROMPT]
    assert calls[0]["seen"].tolist() == PROMPT
    assert calls[0]["attention_width"] == len(PROMPT)
    expected = _oracle(PROMPT, NEW_TOKENS, model.config.vocab_size)
    assert sequences[0, : len(PROMPT)].tolist() == PROMPT
    assert sequences[0, len(PROMPT) :].tolist() == expected
    for step, call in enumerate(calls[1:], start=1):
        past = call["past_key_values"]
        assert isinstance(past, tuple)
        past_kv, past_len = past[0]
        assert past_kv.shape[2] == 2
        assert past_len == len(PROMPT) + step - 1
        assert call["input_ids"].shape[-1] == 1
        assert call["input_ids"][0, 0].item() == expected[step - 1]
        assert call["seen"].tolist() == PROMPT + expected[:step]
        assert call["attention_width"] == len(PROMPT) + step
    with pytest.raises(AttributeError, match="shape"):
        _generate(original, PROMPT)


def test_cache_record_does_not_change_the_scientific_run_id(tmp_path: Path) -> None:
    class Recorded(MockExtractor):
        cache_compatibility = {
            "adapter": CACHE_ADAPTER_ID,
            "dynamic_cache": False,
            "mechanism": "_supports_default_dynamic_cache",
            "transformers": ACCREDITED_TRANSFORMERS,
            "use_cache": True,
        }

    plain = run_pipeline(
        inputs=[InputRecord("gollie-1", "gollie-1", "Ana visited Mary.")],
        extractor=MockExtractor(),
        output_root=tmp_path / "plain",
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )
    recorded = run_pipeline(
        inputs=[InputRecord("gollie-1", "gollie-1", "Ana visited Mary.")],
        extractor=Recorded(),
        output_root=tmp_path / "recorded",
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )
    assert plain.name == recorded.name
    plain_manifest = json.loads((plain / "manifest.json").read_text(encoding="utf-8"))
    recorded_manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    assert plain_manifest["model"] == recorded_manifest["model"]
    assert plain_manifest["generation"] == recorded_manifest["generation"]
    assert "cache_compatibility" not in plain_manifest["operational_metadata"]
    assert recorded_manifest["operational_metadata"]["cache_compatibility"]["adapter"] == (
        CACHE_ADAPTER_ID
    )
    assert recorded_manifest["run_id"] == plain.name


def test_j17_jobs_keep_separate_outputs_and_historical_files(tmp_path: Path) -> None:
    historical = tmp_path / "gollie" / "native-smoke" / "cli-stdout.txt"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical stdout\n", encoding="utf-8")
    first = j17_job_paths.j17_job_directory(tmp_path, "1351430")
    second = j17_job_paths.j17_job_directory(tmp_path, "1351431")
    assert first != second
    assert first.parent == second.parent
    assert historical.parent not in (first, second)
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "cli-stdout.txt").write_text("job 1351430\n", encoding="utf-8")
    (second / "cli-stdout.txt").write_text("job 1351431\n", encoding="utf-8")
    assert historical.read_text(encoding="utf-8") == "historical stdout\n"
    assert (first / "cli-stdout.txt").read_text(encoding="utf-8") != (
        second / "cli-stdout.txt"
    ).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        j17_job_paths.j17_job_directory(tmp_path, "../1351430")
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        j17_job_paths.j17_job_directory(tmp_path, "")
    smoke = (SCRIPTS / "j17_gollie_native_smoke.sh").read_text(encoding="utf-8")
    assert "j17_job_directory" in smoke
    assert 'OUTPUT_ROOT="${P3_RUNS_ROOT}/gollie/native-smoke"' not in smoke
    assert "SLURM_JOB_ID" in smoke
    assert "terminal_run_json.py" in smoke
    assert "pass_baseline_run.py" in smoke
    assert "use_cache=False" not in (ROOT / "src" / "re_te_system" / "extractors" / "gollie.py").read_text(
        encoding="utf-8"
    )
