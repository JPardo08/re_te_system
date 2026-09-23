# Legacy provenance

Legacy repositories were audited as read-only sources. No UI, service, path,
or destructive code was copied.

## TeresIA mREBEL

Source repository revision:
`f75ad75567c9f10d7efa99cc58e0a20ae82de3ad`.
The audited `src/rebel/v3/core/model.py` was last changed at
`94c79b6a8164104b6db7de93a4d607f3240a92df` and has SHA-256
`5b9eba5dccdef73c7ddbb7a7efe3b562f34bc24934fd8af1b44895d29c994740`.

- `src/rebel/v3/core/model.py:MRebelExtractor` — **ADAPTED**. Retained lazy
  sequence-to-sequence inference concepts; removed pandas, global configuration,
  hardcoded dependencies, and presentation concerns.
- `src/rebel/v3/core/model.py:parse_hybrid` and its parser helpers —
  **REIMPLEMENTED**. P0 accepts typed-tag, subject/object-tag, and pipe
  linearizations, emits parse issues, and deliberately preserves duplicates.
- `src/rebel/v3/core/model.py:_token_windows` and
  `_token_windows_in_sentence` — **ADAPTED**. Replaced implicit branches with
  explicit `NONE`/`CHARACTER`/`TOKEN`, maximum, overlap, and mechanism config.
- `src/rebel/v3/core/model.py:split_sentences_with_offsets` and the
  `len(text) > 1200` trigger — **REFERENCE_ONLY**. The magic threshold and
  sentence-first behavior are not P0 defaults.
- `src/rebel/v3/core/model.py:_first_non_overlapping_span` — **ADAPTED**.
  Alignment remains post-hoc, is labelled as such, and failure retains triples.
- `src/rebel/v3/core/model.py:REL_ALLOWED` — **ADAPTED** only as the explicit
  derived `LEGACY_REL_ALLOWED_DERIVED_VIEW`; primary C0 defaults to no filter.
- `src/rebel/v3/core/model.py:norm_rel`, `map_rel_es_if_needed`, and
  `REL_ES2EN` — **NOT_MIGRATED**. Primary relations retain generated surface
  strings; no aliases or Hohfeld mapping are applied.
- `src/rebel/v3/core/model.py` tokenizer construction with `src_lang="es_XX"`,
  `tgt_lang="tp_XX"` and decoder token `tp_XX` — **ADAPTED** and validated in
  configuration tests.
- `src/rebel/v3/core/io.py:save_run_metadata` — **REFERENCE_ONLY**. Replaced by
  a content-addressed, timestamp-independent manifest contract.
- `src/rebel/v3/streamlit_app/**` and `src/rebel/v3/service/**` —
  **NOT_MIGRATED**.

## TFG Hugo mREBEL

Source repository revision:
`05a563234b283d48a1b19740d45261d7541d93f0`.
The audited `scripts/mrebel_extractor.py` was last changed at
`188501a9644ed417dbecd0896c198e84640f7af7`.

- `scripts/mrebel_extractor.py:extract_triplets_typed` — **REFERENCE_ONLY** for
  token-state parsing comparison. The TeresIA hybrid formats informed P0's
  broader deterministic parser.
- `scripts/mrebel_extractor.py:run_corpus` — **REFERENCE_ONLY**. Its default
  duplicate removal, CSV-only final output, fixed truncation, and row skipping
  are incompatible with the P0 contract.
- `scripts/mrebel_extractor.py` `src_lang="en_XX"` CLI default —
  **NOT_MIGRATED**. Spanish P0 uses `es_XX`; target extraction token remains
  `tp_XX`.
- Beam-search defaults and seed setup — **ADAPTED** as explicit serialized
  configuration. P0 does not claim a seed affects deterministic non-sampling
  generation.

## TFG Hugo REBEL baseline

Audited at the same repository revision
`05a563234b283d48a1b19740d45261d7541d93f0`.

- `scripts/rebel_extractor.py:extract_triplets` — **REIMPLEMENTED** as a
  dedicated deterministic parser for
  `<triplet> subject <subj> object <obj> relation`. The new parser preserves
  duplicates and reports malformed, truncated, and empty output.
- `scripts/rebel_extractor.py:run_corpus` — **REFERENCE_ONLY**. Its pandas/CSV
  loop, silent input truncation, default deduplication, and final-only output
  were not migrated.
- `MODEL_NAME`, `INPUT_MAX_LENGTH`, `GEN_MAX_LENGTH`, and `NUM_BEAMS` —
  **ADAPTED** as explicit defaults: `Babelscape/rebel-large`, 256 input tokens,
  generation `max_length=512`, and three deterministic beams.
- Model-config decoder start token — **ADAPTED**. No mREBEL source or target
  language token is injected.
- Random seed setup — **ADAPTED AS METADATA**. Non-sampling beam generation is
  deterministic; P0.5 does not overstate the seed's effect.
- Automatic CUDA selection — **ADAPTED** behind explicit `device` and `dtype`
  manifest configuration.
- Legacy duplicate removal — **NOT MIGRATED**.

## TFG Hugo Pythia / SPACE-KBP

Audited at source repository revision
`05a563234b283d48a1b19740d45261d7541d93f0`.

- `scripts/pythia_extractor.py` — **REFERENCE_ONLY / CLEAN-ROOM SOURCE**. It
  established historical input/output expectations and anti-patterns; no code,
  prompt, or result was copied verbatim.
- The CausalLM `Extractor` shape — **REIMPLEMENTED CLEAN-ROOM** using the
  shared protocol, a canonical pinned checkpoint identity, local-only loading
  by default, explicit dtype/device, and exact decoded RAW preservation.
- Prompt selection — **REIMPLEMENTED CLEAN-ROOM** as the versioned neutral
  `basic-v1` profile. The hardcoded Aqua example and legacy few-shot prompt
  were not copied or migrated.
- Model length handling — **REIMPLEMENTED CLEAN-ROOM** from
  `max_position_embeddings`, generation budget, and prompt-aware tokenizer
  counts. The fixed 1,024-token cutoff and silent truncation were not migrated.
- Turtle parsing with RDFLib — **REIMPLEMENTED CLEAN-ROOM** as a generic,
  conservative parser with optional external prefix context. Parsed RDF term
  metadata and duplicate occurrences are preserved; RAW is never rewritten.
- Prefix injection into model output, prefix stripping, bare-subject repair,
  invented closing syntax, truncation-to-valid-RDF, and the SpaceKBP-specific
  regex fallback — **NOT MIGRATED**.
- Prediction-derived relation aliases and the model-assisted/manual alias
  workflow — **NOT MIGRATED**.
- Default duplicate removal — **NOT MIGRATED**. Duplicates remain predictions
  and receive descriptive `DUPLICATE_TRIPLE` violations.
- Mutable `anonkbp` model default — **NOT MIGRATED**. P0 pins the canonical
  `expertailab` checkpoint and immutable revision; remote download remains
  unauthorized while the fine-tuned-weight license is undeclared.
- CSV-only flattened output and missing RAW Turtle — **NOT MIGRATED**. Pythia
  uses the versioned PredictionRecord stages, failures artifact, manifest, and
  the existing non-destructive evaluation export.
- Legacy result CSVs, aliases, prompts, and test scores — **REFERENCE_ONLY**;
  none were copied into runtime fixtures or outputs.

REBEL, mREBEL, and Pythia/SPACE-KBP are documented only as external baselines.
REBEL/mREBEL are `END_TO_END_TE`; Pythia is a `KBP_DOMAIN_BASELINE`. None
supplies the controlled model needed for the future C0–C3
target-schema-conditioning experiment.

## Universal-IE UIE

Source repository `universal-ie/UIE` was audited read-only at revision
`c88dd08b8ad0016ec597ebc6f8fff49480367bb2`. A separate legacy reproduction is
documented in `_legacy/UIE/LEGACY_REPRODUCTION.md`.

- `inference.py:HuggingfacePredictor` — **REFERENCE_ONLY** for checkpoint
  loading, SSI+text tokenization, generation, and length metadata. The
  hardcoded `.cuda()` call and monolithic evaluation loop were not migrated.
- `RecordSchema`, `schema_to_ssi`, and `PrefixGenerator` —
  **REIMPLEMENTED CLEAN-ROOM** as the baseline-specific `UIESchema` value
  object, deterministic schema-order SSI, stable schema hash, and explicit
  manifest metadata.
- `SpotAsocPredictParser` — **REFERENCE_ONLY**. A conservative SEL parser was
  independently implemented; upstream bracket completion, first-tree
  truncation, malformed-to-empty fallback, unknown-label/span dropping, and
  `<unk>` repair were not migrated.
- `SEL2Record` and `proprocessing_graph_record` — **ADAPT_CONCEPT**. P0 retains
  typed Spot-Association structures and only projects explicit binary
  associations to canonical triples.
- `EntityRecord`, `RelationRecord`, and `EventRecord` offset mapping —
  **REFERENCE_ONLY / REIMPLEMENTED CLEAN-ROOM** as a separate exact alignment
  stage with `exact`, `ambiguous`, and `not_found` statuses. Fuzzy/closest
  repair and silent relation dropping were not migrated.
- `MapConfig.de_duplicate` — **NOT MIGRATED**. Structure and projected triple
  occurrences are retained; duplicate triples receive descriptive violations.
- `DynamicSSIGenerator` sampling, rejection labels, target noise, and data
  collators — **NOT MIGRATED** because they are training behavior, not the P0
  inference contract.
- `SpotAsocConstraintDecoder` and `prefix_allowed_tokens_fn` —
  **REFERENCE_ONLY / FUTURE OPTIONAL PROFILE**. P0 records constraints as
  disabled and does not migrate the state machine.
- The legacy third `record.schema` line — **ADAPTED AS METADATA ONLY** because
  the audited inference/parser/constraint paths load but do not enforce it.
- No legacy source, parser, dataset example, Gold schema, prediction, or result
  was copied verbatim into `re_te_system`.

UIE is an external `UNIVERSAL_IE` baseline with structural-schema knowledge.
It is not a controlled C0-C3 condition, and interface acceptance of custom
labels is not evidence of reliable unseen-schema zero-shot extraction.

## GoLLIE clean-room integration (2026-09-23)

The clean-room GoLLIE adapter does not import `_legacy/GoLLIE` runtime code.
Official RE class wording used by the weight-free fixture is taken from the
audited official notebook example. Historical audit conclusions in this file
and in `EXTERNAL_SYSTEMS_AUDIT.md` are not rewritten.

- Official Python-class prompt shape — **ADAPT_CONCEPT**. P0 serializes
  explicit schema objects with Black (`line_length=119`) instead of
  `inspect.getsource`.
- Official `eval()` output interpretation — **NOT MIGRATED**. P0 uses
  parse-only `ast` and never executes model output.
- First-case-insensitive mention alignment — **NOT MIGRATED**. P0 uses a
  separate exact/ambiguous/not_found stage.
- Upstream mention-count or scorer deduplication — **NOT MIGRATED**.
- CPU/MPS/Ollama/GGUF/vLLM or alternative attention backends —
  **NOT MIGRATED**. Real loading remains CUDA plus FlashAttention.

GoLLIE is an external `UNIVERSAL_IE` baseline reported as
`GUIDELINE_FOLLOWING_UIE_BASELINE`. It is not Controlled C2 and is not a
controlled C0-C3 condition. See `GOLLIE_BASELINE.md`.

## GenIE clean-room integration (2026-09-23)

The clean-room GenIE adapter does not import `_legacy/GenIE` runtime code.
Official Carson RAW strings used by the weight-free fixture are taken from
the audited official notebook and labeled
`OFFICIAL_NOTEBOOK_RECORDED_OUTPUT`. Historical audit conclusions are not
rewritten.

- `GeniePL` / `GenieHF` — **ADAPT_CONCEPT**. P0 uses the shared Extractor
  protocol and does not load Lightning.
- Official `textual_triplets` set conversion — **NOT MIGRATED**.
- Last-write-wins surface-form ID collisions — **NOT MIGRATED**.
- Official pickle tries in unit tests — **NOT MIGRATED**. Future runtime
  should prefer official `*_original_strings.jsonl` plus the pinned tokenizer.
- Relation-only constrained decoding — **NOT MIGRATED**; it would be
  `NON_NATIVE_ABLATION`.
- Automatic 4.9 GB checkpoint download — **NOT MIGRATED**.

GenIE is an external `KBP_CLOSED_IE` baseline reported as
`CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE`. Hard inventory constraints are not
formal ontology reasoning. See `GENIE_BASELINE.md`.
