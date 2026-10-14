"""J16: download HiTZ/GoLLIE-7B only after the L40S preflight passed."""

from __future__ import annotations

import os
from pathlib import Path

from cesvima_common import fail, read_json
from hf_snapshot import cache_or_download


GOLLIE_REPO = "HiTZ/GoLLIE-7B"
GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"


def main() -> int:
    report_path = Path(os.environ["P3_RUNS_ROOT"]) / "gollie" / "preflight_report.json"
    if not report_path.is_file():
        fail("J15 preflight report is missing")
    report = read_json(report_path)
    if report.get("status") != "PASS":
        fail(f"J15 status is {report.get('status')}")
    if report.get("weights_downloaded") is not False:
        fail("J15 report does not record weights_downloaded=false")
    cache_or_download(
        GOLLIE_REPO,
        GOLLIE_REVISION,
        Path(os.environ["P3_RUNS_ROOT"]) / "gollie" / "snapshot.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
