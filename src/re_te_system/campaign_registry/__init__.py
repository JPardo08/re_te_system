"""Declarative campaign registry for Paper 3.

This package is not an orchestrator. Two existing authorities stay in place
and are not invoked from here:

- ``scripts/cesvima/submit_train_first.sh`` is the versioned SoTA job DAG.
  The registry copies its dependency edges into data and checks them. It does
  not submit, schedule, or import that script.
- ``controlled_experiment`` (contract ``exp3b1-1``) records principal states
  of a document-condition attempt. Those states are not plan states. This
  registry does not call the harness and does not change it.

Plan states (``PLANNED``, ``PREPARED``, ``BLOCKED``, ``AWAITING_RESOURCES``,
``AWAITING_AUTHORIZATION``, ``RUNNING``, ``VALIDATED``, ``FAILED``) describe
preparation. They are not ``not_attempted``, ``context_blocked``,
``external_infrastructure_error``, ``timeout``, ``model_execution_error``, or
``completed``.

The command line only reads local JSON and writes a deterministic report.
"""

from re_te_system.campaign_registry.contracts import REGISTRY_CONTRACT_VERSION
from re_te_system.campaign_registry.loader import load_catalog
from re_te_system.campaign_registry.validator import validate_catalog, validate_documents

__all__ = [
    "REGISTRY_CONTRACT_VERSION",
    "load_catalog",
    "validate_catalog",
    "validate_documents",
]
