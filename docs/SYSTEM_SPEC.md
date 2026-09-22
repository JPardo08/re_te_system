# System specification P0.5 (current external-baseline set)

## Scope

The current system preserves the P0.5 contracts and contains three external
baseline adapters: `mREBEL`, `REBEL`, and Pythia/SPACE-KBP. Deterministic
fixtures cover all three output families. The P0.5 naming is retained rather
than retroactively renaming a frozen release. The system contains no C1/C2/C3
conditioning, translation, Hohfeld rescue logic, controlled generator,
training, API, UI, semantic validator, or evaluation matching.

mREBEL is the multilingual Spanish-compatible baseline. REBEL is an
English-centric/monolingual predecessor evaluated directly on Spanish as an
intentional out-of-primary-model-scope baseline. Both are `END_TO_END_TE`,
inherit their native Wikidata-style schema, and receive no target Hohfeld
schema knowledge.

Pythia is a `KBP_DOMAIN_BASELINE` fine-tuned for the fixed SPACE-KBP
space-mission ontology. It is not generic TE, does not accept arbitrary
Hohfeld ontology input, and is not a C0/C1/C2/C3 condition. Comparisons among
any external baselines are outside the future controlled C0–C3 causal
experiment.

The scientific input unit is a document/article. The loader projects each
benchmark JSONL row to exactly `example_id`, `document_id`, and `text`; nested
annotations are inaccessible to extraction. Configured windows are internal
segments and all triples aggregate back into one document prediction.

## Stages

1. RAW stores every exact decoded segment output and available generation
   metadata.
2. PARSED dispatches to the matching format-specific parser: mREBEL
   control tokens, REBEL control tokens, or conservative RDF Turtle. Parsers
   preserve deterministic occurrence order and duplicates and report malformed
   output. Turtle parsing never repairs RAW.
3. NORMALIZED applies Unicode NFC, canonical whitespace, and canonical nulls
   only. RDF literal lexical values are preserved because internal whitespace
   may be semantically significant.
4. VALIDATED annotates structural field, parser, and duplicate violations. It
   never repairs or deletes triples.

Entity offsets are case-insensitive post-hoc string alignment against segment
text. They are marked `posthoc_string_alignment`; null alignment never removes
a prediction.

## Windowing

`NONE`, `CHARACTER`, and tokenizer-aware `TOKEN` strategies are explicit.
Maximum units, overlap, and mechanism are manifest fields. All three real
adapters default to `TOKEN` with their configured effective input limit; mocks
default to `NONE`. Pythia derives its limit from model configuration minus the
generation budget and records truncation explicitly. The legacy 1,200-character
mREBEL trigger and legacy Pythia 1,024-token cutoff are not defaults.

## Failure and ordering policy

Model/configuration load failure aborts startup. Per-segment extraction failure
is recorded in `failures.jsonl`; other documents continue. Inputs use natural
document-id ordering, windows use ascending offsets, and triples preserve model
order. Content files use canonical sorted-key JSON with no variable timestamp.
