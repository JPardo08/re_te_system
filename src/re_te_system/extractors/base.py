"""Thin extractor interface shared by present and future models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from re_te_system.contracts import ExtractionContext, RawExtractionResult


@runtime_checkable
class Extractor(Protocol):
    model_name: str
    model_revision: str

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        """Generate an exact raw model output for one input segment."""
        ...
