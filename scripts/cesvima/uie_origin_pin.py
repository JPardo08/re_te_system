"""Origin pin for luyaojie/uie-base-en pytorch_model.bin.

The SHA-256 and size were measured on CESVIMA. A match proves byte identity
of the preserved pickle. It does not authorize deserializing that pickle.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ORIGIN_BIN_FILENAME = "pytorch_model.bin"
ORIGIN_BIN_SIZE = 990285657
ORIGIN_BIN_SHA256 = "04baf49c28f4be89424f3a61dd30d674c4ab165ff16c1b007525829a8c9441f7"
ORIGIN_REVISION = "966f8b1fc4c74e94ab552081605913ad5133cc41"

# Measured identity is not a load grant. This stays false until a later,
# explicit decision. No caller in this module can flip it.
DESERIALIZATION_AUTHORIZED = False
CONVERSION_ENABLED = False
UIE_J12_DERIVED_LOAD_AUTHORIZED = False
UIE_J13_TRAINING_AUTHORIZED = False

FROZEN_10SHOT_SHA256 = {
    "train.json": "5fecc7091bf7152d04a4eb43641933600f24f3af11995bc3e8f8d268fd43da7b",
    "val.json": "a9be8c5c3716456fb3caffa13f069feea6272a07a4fadce47265db07b30debb8",
    "test.json": "91d6b666268bacf73ea9352f7111c2ef84a212374148bb357933555707811feb",
    "record.schema": "efcac0112405e307aec133d4a289651ed48f93477b004b3fb2e29d1b34913e3d",
}


def deserialization_authorized() -> bool:
    """Always false. The origin hash is not a permission to call torch.load."""
    return False


def origin_identity_matches(size: int, digest: str) -> bool:
    return size == ORIGIN_BIN_SIZE and digest == ORIGIN_BIN_SHA256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_origin_bin(path: Path) -> dict[str, object]:
    """Hash the preserved bin. Do not open it as a pickle."""
    if not path.is_file():
        return {
            "deserialization_authorized": False,
            "filename": ORIGIN_BIN_FILENAME,
            "matches_pin": False,
            "observed": False,
            "sha256": None,
            "size": None,
        }
    size = path.stat().st_size
    digest = _sha256(path)
    return {
        "deserialization_authorized": False,
        "filename": path.name,
        "matches_pin": origin_identity_matches(size, digest),
        "observed": True,
        "sha256": digest,
        "size": size,
    }
