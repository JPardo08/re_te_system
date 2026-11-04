"""Versioned adaptations between the pinned UIE trainer and Transformers 4.57.6.

The upstream tree is not edited. Its hashes remain comparable with the J12
fingerprint. Nothing here disables the Transformers weight-load safety check.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import os
import pickle
from pathlib import Path

from cesvima_common import fail


def require_trainer_runtime() -> None:
    """Fail before training if Transformers 4.57.6 cannot construct its trainer.

    TrainingArguments.__post_init__ imports accelerate when torch is present
    and raises ImportError below accelerate 0.26.0. The bootstrap installs
    that floor only while creating uie-train. An existing venv is left unchanged.
    """

    try:
        version = importlib_metadata.version("accelerate")
    except importlib_metadata.PackageNotFoundError:
        fail(
            "J13 trainer runtime is missing accelerate>=0.26.0; "
            "an existing uie-train venv is not modified by the bootstrap"
        )
    numbers = tuple(int(part) for part in version.split(".")[:2] if part.isdigit())
    if numbers < (0, 26):
        fail(f"J13 trainer runtime requires accelerate>=0.26.0, found {version}")
    for name in ("datasets", "pyarrow"):
        try:
            importlib.import_module(name)
        except ImportError:
            fail(f"J13 trainer runtime is missing {name}")


def adapt_training_argv(argv: list[str]) -> list[str]:
    """Translate the frozen flag spelling and align save_steps with eval_steps.

    --evaluation_strategy is not a field in Transformers 4.57.6. The value
    stays "steps". save_steps is not part of the frozen scientific argv. The
    library default of 500 is not a multiple of the computed eval_steps, so
    load_best_model_at_end refuses to construct the trainer. Aligning the two
    keeps the eval cadence, the metric, and the batch sizes unchanged.
    """

    translated: list[str] = []
    eval_steps: str | None = None
    for item in argv:
        if item.startswith("--evaluation_strategy="):
            translated.append("--eval_strategy=" + item.split("=", 1)[1])
        elif item == "--evaluation_strategy":
            fail("J13 refuses a split evaluation_strategy flag")
        else:
            translated.append(item)
        if item.startswith("--eval_steps="):
            eval_steps = item.split("=", 1)[1]
        if item.startswith("--save_steps=") or item == "--save_steps":
            fail("J13 refuses an explicit save_steps in the frozen argv")
    if any(item.startswith("--evaluation_strategy") for item in translated):
        fail("J13 failed to translate evaluation_strategy")
    if eval_steps is None or int(eval_steps) <= 0:
        fail("J13 training argv has no positive eval_steps")
    translated.append(f"--save_steps={eval_steps}")
    strategy = [item for item in translated if item.startswith("--eval_strategy=")]
    if strategy != ["--eval_strategy=steps"]:
        fail("J13 only translates evaluation_strategy=steps")
    return translated


def restore_removed_trainer_attributes(trainer: object) -> None:
    """Restore attributes Transformers 4.57.6 no longer sets on the trainer.

    _max_length and _num_beams stay None so the upstream prediction step still
    falls back to the model config, which the trainer sets from max_target_length.
    use_amp follows fp16/bf16. The frozen argv enables neither.
    """

    args = getattr(trainer, "args", None)
    if not hasattr(trainer, "_max_length"):
        trainer._max_length = getattr(args, "generation_max_length", None)
    if not hasattr(trainer, "_num_beams"):
        trainer._num_beams = getattr(args, "generation_num_beams", None)
    if not hasattr(trainer, "use_amp"):
        trainer.use_amp = bool(
            getattr(args, "fp16", False) or getattr(args, "bf16", False)
        )


def compatible_training_step(self, model, inputs, num_items_in_batch=None):
    from transformers import Seq2SeqTrainer

    try:
        return Seq2SeqTrainer.training_step(self, model, inputs, num_items_in_batch)
    except RuntimeError as exc:
        if "out of memory" in str(exc):
            self.oom_batch = getattr(self, "oom_batch", 0) + 1
        raise


def compatible_maybe_log_save_evaluate(
    self,
    tr_loss,
    grad_norm,
    model,
    trial,
    epoch,
    ignore_keys_for_eval,
    start_time,
    learning_rate=None,
):
    """Accept the Transformers 4.57.6 call and keep the upstream save rule.

    The extra arguments are required by the current training loop. The loss
    log, start_eval_step, and save_better_checkpoint comparison follow the
    pinned trainer. The checkpoint write uses the current two-argument method.
    """

    del grad_norm, ignore_keys_for_eval, start_time
    if self.control.should_log:
        logs: dict[str, float] = {}
        tr_loss_scalar = tr_loss.item()
        tr_loss -= tr_loss
        logs["loss"] = round(
            tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged),
            4,
        )
        if learning_rate is not None:
            logs["learning_rate"] = learning_rate
        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.log(logs)

    if self.args.start_eval_step > 0 and self.state.global_step < self.args.start_eval_step:
        return

    previous_best_metric = self.state.best_metric
    metrics = None
    if self.control.should_evaluate:
        metrics = self.evaluate()
        self._report_to_hp_search(trial, epoch, metrics)

    if self.args.save_better_checkpoint and self.args.metric_for_best_model is not None:
        if metrics is not None and previous_best_metric is not None:
            if metrics[self.args.metric_for_best_model] <= previous_best_metric:
                return

    if self.control.should_save:
        from transformers import Seq2SeqTrainer

        Seq2SeqTrainer._save_checkpoint(self, model, trial)
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def _local_directory(pretrained: object) -> None:
    path = Path(str(pretrained))
    if not path.is_dir() or path.is_symlink():
        fail("J13 requires a local model directory")
    if any(path.glob("*.bin")):
        fail("J13 refuses to train from pytorch_model.bin")


def force_local_model(original):
    def wrapped(pretrained, *args, **kwargs):
        _local_directory(pretrained)
        kwargs.pop("mirror", None)
        kwargs["use_safetensors"] = True
        kwargs["local_files_only"] = True
        return original(pretrained, *args, **kwargs)

    return wrapped


def force_local_companion(original):
    def wrapped(pretrained, *args, **kwargs):
        _local_directory(pretrained)
        kwargs.pop("mirror", None)
        kwargs["local_files_only"] = True
        return original(pretrained, *args, **kwargs)

    return wrapped


def load_uie_json_dataset(data_files: dict, uie_root: Path):
    """Read the shot JSONL with the upstream pyarrow schema.

    datasets 5 no longer loads a dataset script, so load_dataset("uie_json.py")
    cannot run. The schema file and the pyarrow options are the ones uie_json.py
    already uses. The pickle is only that schema, and only at this pinned path.
    """

    try:
        import pyarrow as pa
        import pyarrow.json as paj
        import datasets
    except ImportError as exc:
        fail(f"J13 dataset loader is missing a dependency: {exc}")
    schema_path = (uie_root / "etc" / "record.dataload.schema").resolve()
    expected_dir = (uie_root.resolve() / "etc")
    if schema_path.parent != expected_dir or schema_path.is_symlink() or not schema_path.is_file():
        fail("J13 dataset schema path is not the pinned upstream file")
    with schema_path.open("rb") as handle:
        table_schema = pickle.load(handle)
    read_options = paj.ReadOptions(use_threads=True)
    parse_options = paj.ParseOptions(explicit_schema=table_schema)
    splits = {}
    for split_name, files in data_files.items():
        if isinstance(files, str):
            files = [files]
        tables = [
            paj.read_json(file_path, read_options=read_options, parse_options=parse_options)
            for file_path in files
        ]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        splits[split_name] = datasets.Dataset(table)
    return datasets.DatasetDict(splits)


def patched_load_dataset(original, uie_root: Path):
    def wrapped(path, *args, **kwargs):
        if os.path.basename(str(path)) == "uie_json.py":
            data_files = kwargs.get("data_files")
            if not isinstance(data_files, dict) or not data_files:
                fail("J13 dataset loader requires local data_files")
            return load_uie_json_dataset(data_files, uie_root)
        fail("J13 dataset loader refuses a dataset other than the local shot files")

    del original
    return wrapped


def install_shims(uie_root: Path) -> None:
    import datasets
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    from uie.seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainer

    datasets.load_dataset = patched_load_dataset(datasets.load_dataset, uie_root)
    AutoModelForSeq2SeqLM.from_pretrained = force_local_model(
        AutoModelForSeq2SeqLM.from_pretrained
    )
    AutoConfig.from_pretrained = force_local_companion(AutoConfig.from_pretrained)
    AutoTokenizer.from_pretrained = force_local_companion(AutoTokenizer.from_pretrained)

    original_init = ConstraintSeq2SeqTrainer.__init__

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        restore_removed_trainer_attributes(self)

    ConstraintSeq2SeqTrainer.__init__ = _init
    ConstraintSeq2SeqTrainer.training_step = compatible_training_step
    ConstraintSeq2SeqTrainer._maybe_log_save_evaluate = compatible_maybe_log_save_evaluate
