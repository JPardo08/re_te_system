# re_te_system

Reproducible Paper 3 extraction system. P0 implements the file-based pipeline
`TEXT → RAW → PARSED → NORMALIZED → VALIDATED → prediction artifact → run
manifest` without importing benchmark or evaluator Python packages.

## Scientific C0

Primary `C0` is open mREBEL extraction. The model receives only document text,
stable identifiers, and unavoidable mREBEL language/runtime formatting. It
does not receive Gold labels, Hohfeld types, signatures, definitions, examples,
ontologies, or relation aliases. mREBEL relations are never mapped to Hohfeld.
The historical `REL_ALLOWED` list is disabled by default and is available only
as an explicitly requested derived view.

## Install and test

Core and mock execution have no runtime dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

Enable real inference separately:

```bash
python -m pip install -e ".[mrebel]"
```

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
