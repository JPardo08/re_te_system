"""Frozen UIE CoNLL04 10-shot training command. This is not a trainer."""

from __future__ import annotations

from pathlib import Path


EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = "1e-4"
SCHEDULER = "constant"
WARMUP_RATIO = "0.06"
SPOT_NOISE = "0.1"
ASOC_NOISE = "0.1"
META_NEGATIVE = "-1"
META_POSITIVE_RATE = "1"
MAX_SOURCE_LENGTH = "256"
MAX_TARGET_LENGTH = "192"
MAX_PREFIX_LENGTH = "-1"
LABEL_SMOOTHING = "0"
TRAINER_SEED = "42"
SPLIT_SEED_ARGUMENT = "1"
BEST_METRIC = "eval_overall-F1"
DECODING_FORMAT = "spotasoc"
TASK_NAME = "meta"
BASE_REVISION = "966f8b1fc4c74e94ab552081605913ad5133cc41"
UIE_UPSTREAM_REVISION = "c88dd08b8ad0016ec597ebc6f8fff49480367bb2"
SINCERE_REVISION = "dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31"


def require_10shot(train_dir: Path) -> Path:
    resolved = train_dir.resolve()
    if resolved.parts[-2:] != ("seed1", "10shot"):
        raise SystemExit(
            f"FAIL: training directory must be seed1/10shot, got {resolved}"
        )
    return resolved


def training_argv(
    *,
    uie_root: Path,
    train_dir: Path,
    model_path: Path,
    output_dir: Path,
    eval_steps: int,
) -> list[str]:
    data = require_10shot(train_dir)
    if eval_steps <= 0:
        raise SystemExit("FAIL: eval_steps must be positive")
    script = uie_root / "run_uie_finetune.py"
    return [
        str(script),
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--use_fast_tokenizer=True",
        "--ddp_find_unused_parameters=False",
        "--predict_with_generate",
        "--evaluation_strategy=steps",
        "--save_strategy=steps",
        "--load_best_model_at_end",
        f"--metric_for_best_model={BEST_METRIC}",
        "--save_total_limit=1",
        f"--max_source_length={MAX_SOURCE_LENGTH}",
        f"--max_prefix_length={MAX_PREFIX_LENGTH}",
        f"--max_target_length={MAX_TARGET_LENGTH}",
        f"--num_train_epochs={EPOCHS}",
        f"--task={TASK_NAME}",
        f"--train_file={data / 'train.json'}",
        f"--validation_file={data / 'val.json'}",
        f"--test_file={data / 'test.json'}",
        f"--record_schema={data / 'record.schema'}",
        f"--per_device_train_batch_size={BATCH_SIZE}",
        f"--per_device_eval_batch_size={BATCH_SIZE * 4}",
        f"--output_dir={output_dir}",
        f"--logging_dir={output_dir}_log",
        f"--model_name_or_path={model_path}",
        f"--learning_rate={LEARNING_RATE}",
        "--source_prefix=meta: ",
        f"--lr_scheduler_type={SCHEDULER}",
        f"--label_smoothing_factor={LABEL_SMOOTHING}",
        f"--eval_steps={eval_steps}",
        f"--decoding_format={DECODING_FORMAT}",
        f"--warmup_ratio={WARMUP_RATIO}",
        "--preprocessing_num_workers=4",
        "--dataloader_num_workers=0",
        f"--meta_negative={META_NEGATIVE}",
        f"--meta_positive_rate={META_POSITIVE_RATE}",
        "--skip_memory_metrics",
        "--no_remove_unused_columns",
        "--ordered_prompt=False",
        "--save_better_checkpoint=True",
        f"--spot_noise={SPOT_NOISE}",
        f"--asoc_noise={ASOC_NOISE}",
        f"--seed={TRAINER_SEED}",
        "--disable_tqdm=True",
    ]
