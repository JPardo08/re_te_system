# Extractor interface

An extractor implements:

```python
class Extractor(Protocol):
    model_name: str
    model_revision: str
    def extract(
        self, text: str, context: ExtractionContext
    ) -> RawExtractionResult: ...
```

`ExtractionContext` carries only input/document/segment identifiers, condition,
and optional segment boundaries. It contains no Gold, evaluator, Hohfeld,
schema, or future conditioning fields.

`RawExtractionResult` carries the exact decoded `model_output` plus available
generation metadata. Parsing, normalization, validation, persistence, and
evaluation export are downstream concerns. A future extractor can implement
the protocol without depending on mREBEL or changing the prediction stages.

The runner may use an optional `count_tokens(text)` capability for
tokenizer-aware windows; it is not part of the minimum extractor protocol.
