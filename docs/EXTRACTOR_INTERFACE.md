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

For Pythia, the inherited context `condition` remains a compatibility value,
not a controlled-experiment assignment. The manifest marks
`controlled_experiment_condition: not_applicable`.

`RawExtractionResult` carries the exact decoded `model_output` plus available
generation metadata. Parsing, normalization, validation, persistence, and
evaluation export are downstream concerns. The shared runner receives the
matching format-specific parser callable. Current dispatches are
`parse_mrebel`, `parse_rebel`, and `parse_turtle` for Pythia/SPACE-KBP. This is
the only format-specific downstream dispatch; all external baselines otherwise
traverse the identical pipeline.

REBEL receives Spanish source text unchanged and no language control tokens.
mREBEL uses only its unavoidable source/target runtime language configuration.
Pythia receives a versioned clean-room prompt and emits an exact decoded Turtle
continuation. RDF parsing, optional prefix context, normalization, and
validation remain downstream and never overwrite RAW.
A future controlled extractor can implement the protocol without depending on
any baseline or changing the prediction stages.

The runner may use an optional `count_tokens(text)` capability for
tokenizer-aware windows; it is not part of the minimum extractor protocol.
