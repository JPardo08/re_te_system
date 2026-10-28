"""Per-job output directories for the GoLLIE J17 smoke."""

from __future__ import annotations

import re
from pathlib import Path


_JOB_ID = re.compile(r"[1-9][0-9]*")


def j17_job_directory(runs_root: Path, job_id: str) -> Path:
    """Return the private output root for one SLURM job.

    The scientific ``run_id`` is still created inside this directory. Two jobs
    therefore keep the same run identity without sharing files. Historical
    outputs that live directly under ``native-smoke`` are left in place.
    """
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("J17 requires a numeric SLURM_JOB_ID")
    root = Path(runs_root).resolve()
    jobs = (root / "gollie" / "native-smoke" / "jobs").resolve()
    destination = (jobs / job_id).resolve()
    if destination.parent != jobs:
        raise ValueError("J17 job directory escapes its job root")
    return destination
