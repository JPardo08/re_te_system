"""Child entry point for one authorized J13 training process.

J13 sets UIE_J13_LAUNCHED_BY_GATE only in this process, after the evidence
gates. Running this file by itself does not train.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from cesvima_common import fail
from uie_j13_gate import RUN_ENV, launched_by_gate, run_authorized
from uie_trainer_compat import adapt_training_argv, install_shims, require_trainer_runtime


def main() -> int:
    if not launched_by_gate() or not run_authorized():
        fail("J13 trainer launcher is not authorized")
    if len(sys.argv) < 2 or Path(sys.argv[1]).name != "run_uie_finetune.py":
        fail("J13 trainer launcher refuses a script other than run_uie_finetune.py")
    require_trainer_runtime()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    adapted = adapt_training_argv(sys.argv[1:])
    install_shims(Path.cwd())
    sys.argv = adapted
    runpy.run_path(adapted[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
