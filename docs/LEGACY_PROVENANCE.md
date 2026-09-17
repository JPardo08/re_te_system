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

REBEL and mREBEL are documented only as external baselines. Neither supplies
the controlled model needed for the future C0–C3 target-schema-conditioning
experiment.
