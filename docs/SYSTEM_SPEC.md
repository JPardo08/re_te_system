# System specification P0

## Scope

P0 contains one production extractor (`mREBEL`) and one deterministic test
extractor (`mock`). It contains no C1/C2/C3 conditioning, Hohfeld rescue logic,
REBEL baseline, controlled generator, training, API, UI, semantic validator, or
evaluation matching.

The scientific input unit is a document/article. The loader projects each
benchmark JSONL row to exactly `example_id`, `document_id`, and `text`; nested
annotations are inaccessible to extraction. Configured windows are internal
segments and all triples aggregate back into one document prediction.

## Stages

1. RAW stores every exact decoded segment output and available generation
   metadata.
2. PARSED deterministically interprets common mREBEL linearizations and reports
   malformed chunks. It preserves order, surface strings, and duplicates.
3. NORMALIZED applies Unicode NFC, canonical whitespace, and canonical nulls
   only.
4. VALIDATED annotates structural field, parser, and duplicate violations. It
   never repairs or deletes triples.

Entity offsets are case-insensitive post-hoc string alignment against segment
text. They are marked `posthoc_string_alignment`; null alignment never removes
a prediction.

## Windowing

`NONE`, `CHARACTER`, and tokenizer-aware `TOKEN` strategies are explicit.
Maximum units, overlap, and mechanism are manifest fields. Real mREBEL defaults
to `TOKEN` with its configured input limit; mock defaults to `NONE`. The legacy
1,200-character trigger is not a default.

## Failure and ordering policy

Model/configuration load failure aborts startup. Per-segment extraction failure
is recorded in `failures.jsonl`; other documents continue. Inputs use natural
document-id ordering, windows use ascending offsets, and triples preserve model
order. Content files use canonical sorted-key JSON with no variable timestamp.
