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

For Pythia, UIE, GoLLIE, and GenIE, the inherited context `condition` remains a
compatibility value, not a controlled-experiment assignment. The manifest
marks `controlled_experiment_condition: not_applicable`.

`RawExtractionResult` carries the exact decoded `model_output` plus available
generation metadata. Parsing, normalization, validation, persistence, and
evaluation export are downstream concerns. The shared runner receives the
matching format-specific parser callable. Current dispatches are
`parse_mrebel`, `parse_rebel`, `parse_turtle` for Pythia/SPACE-KBP,
`parse_sel` for UIE, `parse_gollie` for GoLLIE, and `parse_genie` for GenIE.
UIE additionally injects
typed-structure alignment and binary projection callables because SEL is not
natively a triple stream. GoLLIE injects typed-record alignment and relation
projection because constructor lists are not a triple stream and must not be
executed. All external baselines otherwise traverse the identical stage
pipeline.

REBEL receives Spanish source text unchanged and no language control tokens.
mREBEL uses only its unavoidable source/target runtime language configuration.
Pythia receives a versioned clean-room prompt and emits an exact decoded Turtle
continuation. RDF parsing, optional prefix context, normalization, and
validation remain downstream and never overwrite RAW.
UIE receives its own baseline-specific structural schema/SSI. The extractor
emits exact decoded SEL; typed Spot-Association parsing, exact alignment, and
binary relation projection remain downstream. These UIE schema mechanics are
not the controlled C0-C3 conditioning layer.
GoLLIE receives its own baseline-specific guideline schema. The extractor
emits an exact decoded constructor-list continuation; safe AST parsing, exact
alignment, and binary relation projection remain downstream. GoLLIE schema
mechanics are not Controlled C2 and are not the controlled C0-C3
conditioning layer.
GenIE receives a named constraint profile and optional inventories. The
extractor emits exact decoded `<sub>/<rel>/<obj>/<et>` markup; ordered
parsing, optional ID mapping, and beam metadata remain downstream. GenIE
constraint mechanics are not C3 and are not the controlled C0-C3
conditioning layer.
A future controlled extractor can implement the protocol without depending on
any baseline or changing the prediction stages.

The runner may use an optional `count_tokens(text)` capability for
tokenizer-aware windows; it is not part of the minimum extractor protocol.
