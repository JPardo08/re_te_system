"""Offline bridge from controlled-experiment artifacts to official P0.

The reproducible entry point is ``python -m re_te_system.evaluation_bridge``.
"""

from re_te_system.evaluation_bridge.bridge import evaluate_campaign
from re_te_system.evaluation_bridge.contracts import BRIDGE_CONTRACT_VERSION, BridgeRequest

__all__ = [
    "BRIDGE_CONTRACT_VERSION",
    "BridgeRequest",
    "evaluate_campaign",
]
