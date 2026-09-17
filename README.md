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

## Development environment

The canonical development environment is Python 3.12 in the globally stored
Conda named environment `re_te_system_312`. The package continues to support
Python 3.11 or newer.

```bash
conda create -n re_te_system_312 python=3.12 -y
conda activate re_te_system_312
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mrebel]"
pytest
```

Hugging Face model files can be stored in a repository-local, unversioned
cache:

```bash
export HF_HOME="$PWD/.cache/huggingface"
```

Core and mock execution have no runtime dependencies. For development without
ML dependencies, use `python -m pip install -e ".[dev]"`. The semantically
distinct `mrebel` and `rebel` extras currently declare the same ML dependency
set and can be combined with `dev` independently:

```bash
python -m pip install -e ".[dev,mrebel]"
python -m pip install -e ".[dev,rebel]"
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
