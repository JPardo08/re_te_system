"""J16: refuse GoLLIE-7B download until an identified preflight exists.

A legacy report, a report from another JOBID, or a BINARY_MISMATCH does not
pass. Even a complete report does not download while the authorization flag
stays false.
"""

from __future__ import annotations

import os
from pathlib import Path

from cesvima_common import fail, read_json
from flash_attn_pin import GOLLIE_J16_DOWNLOAD_AUTHORIZED
from gollie_environment import accepted_pointer, j16_rejection_reason, job_report_dir
from hf_snapshot import cache_or_download


GOLLIE_REPO = "HiTZ/GoLLIE-7B"
GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"


def load_accepted_report(runs_root: Path) -> dict:
    pointer = accepted_pointer(runs_root)
    if not pointer.is_file():
        fail("J15 preflight is unidentified; refusing to download")
    job_id = pointer.read_text(encoding="utf-8").strip()
    report_path = job_report_dir(runs_root, job_id) / "preflight_report.json"
    if not report_path.is_file():
        fail("J15 preflight report is missing")
    report = read_json(report_path)
    reason = j16_rejection_reason(report, job_id)
    if reason is not None:
        fail(reason)
    return report


def main() -> int:
    runs = Path(os.environ["P3_RUNS_ROOT"])
    load_accepted_report(runs)
    if GOLLIE_J16_DOWNLOAD_AUTHORIZED is not True:
        fail("J16 weight download is not authorized")
    cache_or_download(
        GOLLIE_REPO,
        GOLLIE_REVISION,
        runs / "gollie" / "snapshot.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
