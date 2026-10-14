"""Static checks for the CESVIMA train-first job layer."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs" / "cesvima"
SCRIPTS = ROOT / "scripts" / "cesvima"
SUBMITTER = SCRIPTS / "submit_train_first.sh"

MREBEL_REVISION = "50e0587ac7ac87d28b9abd069d72333528a5aa09"
UIE_UPSTREAM = "c88dd08b8ad0016ec597ebc6f8fff49480367bb2"
SINCERE = "dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31"
UIE_BASE = "966f8b1fc4c74e94ab552081605913ad5133cc41"
GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"
GENIE_TOKENIZER = "81eb2ccca714fbb62a65283b65c9c153469ee409"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sbatch_directives(path: Path) -> list[str]:
    return [line for line in _read(path).splitlines() if line.startswith("#SBATCH")]


def _all_directives() -> list[str]:
    return [line for path in JOBS.glob("*.sbatch") for line in _sbatch_directives(path)]


def test_critical_pins_are_present() -> None:
    assert MREBEL_REVISION in _read(SCRIPTS / "j01_mrebel_cache.py")
    assert MREBEL_REVISION in _read(JOBS / "j02_mrebel_smoke3.sbatch")
    assert UIE_UPSTREAM in _read(SCRIPTS / "j08_uie_fetch_upstreams.sh")
    assert SINCERE in _read(SCRIPTS / "j08_uie_fetch_upstreams.sh")
    assert UIE_BASE in _read(SCRIPTS / "j11_uie_cache_base.py")
    assert GOLLIE_REVISION in _read(SCRIPTS / "j16_gollie_download.py")
    assert GOLLIE_REVISION in _read(SCRIPTS / "j17_gollie_native_smoke.sh")
    assert "4879373206" in _read(SCRIPTS / "fetch_genie_asset.sh")
    assert "c214da56b6e5d5bd259e0cbe826d92f7" in _read(SCRIPTS / "fetch_genie_asset.sh")
    assert "94255261" in _read(SCRIPTS / "fetch_genie_asset.sh")
    assert "45db4dc2578cdeda891d8ed6ddb3f89f" in _read(SCRIPTS / "fetch_genie_asset.sh")
    assert GENIE_TOKENIZER in _read(SCRIPTS / "j19_genie_tokenizer.py")
    config = _read(SCRIPTS / "uie_train_config.py")
    for needle in (
        'EPOCHS = 200',
        'BATCH_SIZE = 16',
        'LEARNING_RATE = "1e-4"',
        'SCHEDULER = "constant"',
        'WARMUP_RATIO = "0.06"',
        'SPOT_NOISE = "0.1"',
        'ASOC_NOISE = "0.1"',
        'META_NEGATIVE = "-1"',
        'SPLIT_SEED_ARGUMENT = "1"',
        'TRAINER_SEED = "42"',
        'BEST_METRIC = "eval_overall-F1"',
        "--ordered_prompt=False",
    ):
        assert needle in config


def test_no_a100_or_untyped_gpu_directive() -> None:
    directives = _all_directives()
    assert directives
    blob = "\n".join(directives).lower()
    assert "a100" not in blob
    for line in directives:
        if "--gres=" in line:
            assert "gpu:v100:1" in line or "gpu:l40s:1" in line
            assert "gpu:1" not in line.replace("gpu:v100:1", "").replace("gpu:l40s:1", "")


def test_dag_dependencies_are_encoded() -> None:
    submitter = _read(SUBMITTER)
    assert "submit_job J01 jobs/cesvima/j01_mrebel_cache_verify.sbatch J00" in submitter
    assert "submit_job J02 jobs/cesvima/j02_mrebel_smoke3.sbatch J01" in submitter
    assert "submit_job J03 jobs/cesvima/j03_mrebel_full.sbatch J02" in submitter
    assert "submit_job J04 jobs/cesvima/j04_rebel_resolve_revision.sbatch J00" in submitter
    assert "submit_job J05 jobs/cesvima/j05_rebel_download.sbatch J04" in submitter
    assert "j06_rebel_smoke3.sbatch" not in submitter
    assert "j07_rebel_full.sbatch" not in submitter
    assert "submit_job J07 jobs/cesvima/j07_rebel_hohfeld_cross_lingual.sbatch J05" in submitter
    assert "submit_job J09 jobs/cesvima/j09_uie_rebuild_data.sbatch J00 J08" in submitter
    assert "submit_job J12 jobs/cesvima/j12_uie_preflight.sbatch J09 J11" in submitter
    assert "submit_job J13 jobs/cesvima/j13_uie_train.sbatch J12" in submitter
    assert "submit_job J14 jobs/cesvima/j14_uie_positive_smoke.sbatch J13 J10" in submitter
    assert "submit_job J16 jobs/cesvima/j16_gollie_download.sbatch J15" in submitter
    assert "submit_job J17 jobs/cesvima/j17_gollie_native_smoke.sbatch J16" in submitter
    assert "submit_job J19 jobs/cesvima/j19_genie_download_tokenizer.sbatch J00" in submitter
    assert "--dependency=afterok:" in submitter


def test_pythia_has_no_job() -> None:
    names = [path.name.lower() for path in list(JOBS.iterdir()) + list(SCRIPTS.iterdir())]
    assert not any("pythia" in name for name in names)
    document = _read(ROOT / "docs" / "CESVIMA_TRAIN_FIRST.md")
    assert "TARGET / BLOCKED_LICENSE" in document
    assert "descart" not in document.lower()
    assert "rejected" not in document.lower()


def test_genie_has_no_inference_job() -> None:
    genie_jobs = [
        _read(path)
        for path in JOBS.glob("*.sbatch")
        if "genie" in path.name
    ]
    assert len(genie_jobs) == 3
    blob = "\n".join(genie_jobs)
    assert "re_te_system.cli" not in blob
    assert "GenIEExtractor" not in blob
    assert "BLOCKED_LOADER" in _read(SUBMITTER)
    document = _read(ROOT / "docs" / "CESVIMA_TRAIN_FIRST.md")
    assert "TARGET / BLOCKED_LOADER" in document
    assert "STANDBY" in document


def test_mrebel_is_pinned_and_local_only() -> None:
    runner = _read(SCRIPTS / "run_seq2seq_baseline.sh")
    assert "--local-files-only" in runner
    assert "--model-revision" in runner
    assert "--legacy-relation-filter" not in runner
    assert "es_XX" in runner
    assert "tp_XX" in runner
    assert "Babelscape/mrebel-large" in runner
    native = _read(JOBS / "j06_rebel_native_smoke.sbatch") + _read(SCRIPTS / "j06_rebel_native_smoke.py")
    assert "BLOCKED_FIXTURE" in native
    assert "articulo_" not in native
    assert "CANONICAL_DOCUMENTS" not in native
    assert "select_smoke3.py" not in native
    cross = _read(JOBS / "j07_rebel_hohfeld_cross_lingual.sbatch")
    assert "cross-lingual" in cross
    assert "es_XX" not in cross
    assert "tp_XX" not in cross
    assert "re_te_system_gpu_312" in _read(SCRIPTS / "env.sh")
    assert "re_te_system_gpu_312" in _read(SCRIPTS / "j00_verify_infer_env.py")
    assert "0.2.2" in _read(SCRIPTS / "j00_verify_infer_env.py")


def test_uie_j13_requires_j12_fingerprint() -> None:
    source = _read(SCRIPTS / "j13_uie_train.py")
    assert "environment_fingerprint.json" in source
    assert "preflight_report.json" in source
    assert "fingerprint_digest" in source
    assert "differs from J12" in source
    assert "run_uie_finetune_shot.bash" not in source
    assert "sel2record" not in source
    assert "require_10shot" in source


def test_central_paths() -> None:
    env = _read(SCRIPTS / "env.sh")
    assert "PROJECT_ROOT=/home/y064/PROJECT/y064644/paper3" in env
    assert "SCRATCH_ROOT=/home/y064/SCRATCH/y064644/paper3" in env
    assert 'SYSTEM_ROOT="${PROJECT_ROOT}/re_te_system"' in env
    assert 'BENCHMARK_ROOT="${PROJECT_ROOT}/re_te_benchmark"' in env
    assert 'EVALUATION_ROOT="${PROJECT_ROOT}/re_te_evaluation"' in env
    for name in ("HF_HOME", "HF_HUB_CACHE", "PIP_CACHE_DIR", "TMPDIR", "PYTHONUNBUFFERED"):
        assert name in env
    for path in JOBS.glob("*.sbatch"):
        text = _read(path)
        assert "source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh" in text
        assert "--account" not in text
        assert "--mail-user" not in text


def test_require_sha_and_training_scope() -> None:
    sys.path.insert(0, str(SCRIPTS))
    from cesvima_common import require_sha
    from uie_train_config import require_10shot, training_argv

    assert require_sha(MREBEL_REVISION) == MREBEL_REVISION
    try:
        require_sha("main")
    except SystemExit:
        pass
    else:
        raise AssertionError("non-SHA revision was accepted")
    try:
        require_10shot(Path("/tmp/seed1/5shot"))
    except SystemExit:
        pass
    else:
        raise AssertionError("5shot directory was accepted")
    argv = training_argv(
        uie_root=Path("/upstream/UIE"),
        train_dir=Path("/scratch/converted/seed1/10shot"),
        model_path=Path("/scratch/uie-base"),
        output_dir=Path("/scratch/runs/uie"),
        eval_steps=80,
    )
    joined = " ".join(argv)
    assert "run_uie_finetune.py" in joined
    assert "run_exp_shot.bash" not in joined
    assert "--fp16" not in joined
    assert "seed1/10shot/train.json" in joined


def test_dry_run_does_not_submit() -> None:
    fake_bin = Path(tempfile.mkdtemp(prefix="cesvima-fake-sbatch-"))
    marker = fake_bin / "sbatch-was-executed"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\nprintf '%s\\n' executed > \"$(dirname \"$0\")/sbatch-was-executed\"\nexit 99\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    completed = subprocess.run(
        ["bash", str(SUBMITTER), "--dry-run", "--all"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert "DRY-RUN complete; sbatch was not executed" in completed.stdout
    assert "--dependency=afterok:DRY_J00" in completed.stdout
    assert "--dependency=afterok:DRY_J12" in completed.stdout
    assert "BLOCKED_LICENSE" in completed.stdout
    assert "BLOCKED_LOADER" in completed.stdout
    assert "BLOCKED_FIXTURE" in completed.stdout
    assert "j06_rebel_native_smoke.sbatch" not in completed.stdout
    assert "j07_rebel_hohfeld_cross_lingual.sbatch" not in completed.stdout
    assert "/home/y064/PROJECT/y064644/paper3/re_te_system/jobs/cesvima/j13_uie_train.sbatch" in completed.stdout

    empty = subprocess.run(
        ["bash", str(SUBMITTER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert empty.returncode == 2
    assert not marker.exists()

    only_mrebel = subprocess.run(
        ["bash", str(SUBMITTER), "--dry-run", "--mrebel"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert only_mrebel.returncode == 0, only_mrebel.stderr
    assert "j15_gollie_preflight.sbatch" not in only_mrebel.stdout
    assert "j18_genie_download_ckpt.sbatch" not in only_mrebel.stdout
    assert not marker.exists()

    cross = subprocess.run(
        ["bash", str(SUBMITTER), "--dry-run", "--rebel-cross-lingual"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cross.returncode == 0, cross.stderr
    assert "j07_rebel_hohfeld_cross_lingual.sbatch" in cross.stdout
    assert "--dependency=afterok:DRY_J05" in cross.stdout
    assert "j06_rebel_native_smoke.sbatch" not in cross.stdout
    assert not marker.exists()
