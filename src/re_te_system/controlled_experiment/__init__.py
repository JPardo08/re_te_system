"""Controlled-experiment harness for Paper 3.

This package is a campaign layer. It is not a second baseline runner and it
does not execute a model. The scientific rules come from the benchmark
protocol. Parsing, normalization, structural validation, and P0 export are
reused from the existing system contracts.
"""

from re_te_system.controlled_experiment.campaign import (
    CampaignConfig,
    SyntheticDocument,
    run_controlled_campaign,
)
from re_te_system.controlled_experiment.contracts import (
    CAMPAIGN_CONTRACT_VERSION,
    SYNTHETIC_CONTRACT_TEST,
)
from re_te_system.controlled_experiment.preflight import assess_preflight
from re_te_system.controlled_experiment.synthetic import (
    ScriptedTurn,
    ScriptedTsvExtractor,
    SyntheticTokenCounter,
)

__all__ = [
    "CAMPAIGN_CONTRACT_VERSION",
    "SYNTHETIC_CONTRACT_TEST",
    "CampaignConfig",
    "ScriptedTsvExtractor",
    "ScriptedTurn",
    "SyntheticDocument",
    "SyntheticTokenCounter",
    "assess_preflight",
    "run_controlled_campaign",
]
