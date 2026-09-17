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
evaluation export are downstream concerns. The shared runner receives the
matching parser callable (`parse_mrebel` or `parse_rebel`); this is the only
format-specific downstream dispatch. Both external baselines otherwise traverse
the identical pipeline.

REBEL receives Spanish source text unchanged and no language control tokens.
mREBEL uses only its unavoidable source/target runtime language configuration.
A future controlled extractor can implement the protocol without depending on
either baseline or changing the prediction stages.

The runner may use an optional `count_tokens(text)` capability for
tokenizer-aware windows; it is not part of the minimum extractor protocol.
