# re_te_system

Reproducible Paper 3 extraction system. P0 implements the file-based pipeline
`TEXT → RAW → PARSED → NORMALIZED → VALIDATED → prediction artifact → run
manifest` without importing benchmark or evaluator Python packages.

## External baselines and controlled C0 separation

mREBEL and REBEL are external historical TE baselines with pretrained,
Wikidata-style native relation schemas. Pythia/SPACE-KBP is a separate
`KBP_DOMAIN_BASELINE` tied to its fixed space-mission ontology. UIE is a
`UNIVERSAL_IE` baseline receiving a dynamic runtime structural schema. GoLLIE
is a separate `GUIDELINE_FOLLOWING_UIE_BASELINE` receiving structural schema
plus definitions/guidelines; it is not Controlled C2. GenIE is a separate
`CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE` / `KBP_CLOSED_IE` comparator with
fixed native schema plus optional KB constraints; it is not C3. None is a
controlled C0/C1/C2/C3 condition, and none maps native output to Hohfeld.

mREBEL is multilingual and receives the required `es_XX`/`tp_XX` runtime
formatting. REBEL is English-centric/monolingual; Spanish text is passed
directly, without translation or language tokens, as an intentional
out-of-primary-model-scope baseline. They are not linguistically equivalent.
The historical mREBEL `REL_ALLOWED` list remains disabled by default.

Future C0–C3 comparisons will use one controlled model to study target-schema
conditioning. Differences between these external baselines are not evidence of
a C0→C3 conditioning effect. Pythia manifests add
`controlled_experiment_condition: not_applicable`; the legacy-compatible
`condition: C0` field must not be interpreted causally. UIE uses the same
explicit `not_applicable` marker and does not establish arbitrary-schema
zero-shot reliability. GoLLIE uses the same `not_applicable` marker and is
not part of same-model causal contrast. GenIE uses the same marker; hard
inventory constraints are not formal ontology reasoning.

## Development environment

The canonical development environment is Python 3.12 in the globally stored
Conda named environment `re_te_system_312`. The package continues to support
Python 3.11 or newer.

```bash
conda create -n re_te_system_312 python=3.12 -y
conda activate re_te_system_312
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mrebel,rebel,pythia,uie,gollie,genie]"
pytest
```

Hugging Face model files can be stored in a repository-local, unversioned
cache:

```bash
export HF_HOME="$PWD/.cache/huggingface"
```

Core and REBEL/mREBEL/UIE mock execution have no runtime dependencies. GoLLIE
mock prompt serialization requires the pinned Black dependency included in
`dev` and `gollie`. Pythia Turtle parsing requires its optional RDFLib
dependency. For development without ML dependencies, use
`python -m pip install -e ".[dev]"`. The semantically distinct `mrebel`,
`rebel`, `pythia`, `uie`, `gollie`, and `genie` extras can be combined with `dev`
independently:

```bash
python -m pip install -e ".[dev,mrebel]"
python -m pip install -e ".[dev,rebel]"
python -m pip install -e ".[dev,pythia]"
python -m pip install -e ".[dev,uie]"
python -m pip install -e ".[dev,gollie]"
python -m pip install -e ".[dev,genie]"
```

Use `--extractor rebel-mock` for the dependency-free REBEL contract fixture, or
`--extractor rebel --local-files-only` for a cached real checkpoint. The latter
defaults to `Babelscape/rebel-large` and never adds mREBEL language tokens.

Use `--extractor pythia-mock` for the weight-free Pythia/Turtle contract
fixture after installing the `pythia` extra. The real adapter is pinned to the
canonical immutable checkpoint and uses local files only. Its fine-tuned weight
license is unresolved, so real model download and smoke are not authorized.
See `docs/PYTHIA_BASELINE.md`.

Use `--extractor uie-mock --uie-schema-file SCHEMA.json` for the weight-free
UIE structural contract. The pinned real adapter is local-only by default.
The schema interface accepts custom labels, but reliable unseen-schema
zero-shot extraction is not established. See `docs/UIE_BASELINE.md`.

Use `--extractor gollie-mock --gollie-schema-file SCHEMA.json` for the
weight-free GoLLIE contract. Prompt serialization requires the pinned Black
dependency included in `dev` and `gollie`. FlashAttention is not a core or
`dev` dependency. Real GoLLIE inference still requires a separate NVIDIA CUDA
GPU environment with official FlashAttention and has no CPU/MPS fallback in
P0. See `docs/GOLLIE_BASELINE.md`.

Use `--extractor genie-mock` for the weight-free GenIE contract. The optional
`genie` extra is for a future direct BART runtime only; Lightning and Hydra
are not installed. P0 does not download `genie_r.ckpt`. See
`docs/GENIE_BASELINE.md`.

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
