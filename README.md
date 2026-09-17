# re_te_system

Reproducible Paper 3 extraction system. P0 implements the file-based pipeline
`TEXT → RAW → PARSED → NORMALIZED → VALIDATED → prediction artifact → run
manifest` without importing benchmark or evaluator Python packages.

## External baselines and scientific C0

mREBEL and REBEL are external historical baselines with pretrained,
Wikidata-style native relation schemas. Both run with target condition `C0`:
the model receives document text but no Gold labels, Hohfeld types, signatures,
definitions, examples, ontology, or relation aliases. Neither baseline maps
native relations to Hohfeld.

mREBEL is multilingual and receives the required `es_XX`/`tp_XX` runtime
formatting. REBEL is English-centric/monolingual; Spanish text is passed
directly, without translation or language tokens, as an intentional
out-of-primary-model-scope baseline. They are not linguistically equivalent.
The historical mREBEL `REL_ALLOWED` list remains disabled by default.

Future C0–C3 comparisons will use one controlled model to study target-schema
conditioning. Differences between these external baselines are not evidence of
a C0→C3 conditioning effect.

## Install and test

Core and mock execution have no runtime dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

Enable either real baseline separately:

```bash
python -m pip install -e ".[mrebel]"
python -m pip install -e ".[rebel]"
```

Use `--extractor rebel-mock` for the dependency-free REBEL contract fixture, or
`--extractor rebel --local-files-only` for a cached real checkpoint. The latter
defaults to `Babelscape/rebel-large` and never adds mREBEL language tokens.

## Mock document run

```bash
python scripts/run_extraction.py run \
  --extractor mock \
  --documents ../re_te_benchmark/datasets/teresia_hohfeld/canonical/documents.jsonl \
  --benchmark-manifest ../re_te_benchmark/datasets/teresia_hohfeld/manifests/manifest.json \
  --output-root runs
```

Each run contains `manifest.json`, `predictions.jsonl`, `failures.jsonl`, and
`statistics.json`. Prediction records retain document input, exact per-window
raw output, parsed triples, representationally normalized triples, structural
validation, offsets identified as post-hoc alignment, and segment provenance.

Export the evaluator view without changing the primary artifact:

```bash
python scripts/run_extraction.py export-evaluation \
  --predictions runs/RUN_ID/predictions.jsonl \
  --output runs/RUN_ID/evaluation_predictions.jsonl
```

Pass that JSONL by path to `re_te_evaluation` in document mode. See `docs/` for
the frozen contracts, interface, manifest identity, and legacy provenance.
