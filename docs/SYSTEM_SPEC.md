# System specification P0.5

## Scope

P0.5 contains two external baseline adapters (`mREBEL` and `REBEL`) and
deterministic fixtures for both output formats. It contains no C1/C2/C3
conditioning, translation, Hohfeld rescue logic, controlled generator,
training, API, UI, semantic validator, or evaluation matching.

mREBEL is the multilingual Spanish-compatible baseline. REBEL is an
English-centric/monolingual predecessor evaluated directly on Spanish as an
intentional out-of-primary-model-scope baseline. Both inherit their native
Wikidata-style schema and receive no target Hohfeld schema knowledge. Their
comparison is not the future controlled C0–C3 causal experiment.

The scientific input unit is a document/article. The loader projects each
benchmark JSONL row to exactly `example_id`, `document_id`, and `text`; nested
annotations are inaccessible to extraction. Configured windows are internal
segments and all triples aggregate back into one document prediction.

## Stages

1. RAW stores every exact decoded segment output and available generation
   metadata.
2. PARSED dispatches to a dedicated mREBEL or REBEL control-token parser and
   reports malformed chunks. Both preserve order, surface strings, and
   duplicates.
3. NORMALIZED applies Unicode NFC, canonical whitespace, and canonical nulls
   only.
4. VALIDATED annotates structural field, parser, and duplicate violations. It
   never repairs or deletes triples.

Entity offsets are case-insensitive post-hoc string alignment against segment
text. They are marked `posthoc_string_alignment`; null alignment never removes
a prediction.

## Windowing

`NONE`, `CHARACTER`, and tokenizer-aware `TOKEN` strategies are explicit.
Maximum units, overlap, and mechanism are manifest fields. Both real adapters
default to `TOKEN` with their configured input limit; mocks default to `NONE`.
The legacy 1,200-character mREBEL trigger is not a default.

## Failure and ordering policy

Model/configuration load failure aborts startup. Per-segment extraction failure
is recorded in `failures.jsonl`; other documents continue. Inputs use natural
document-id ordering, windows use ascending offsets, and triples preserve model
order. Content files use canonical sorted-key JSON with no variable timestamp.
