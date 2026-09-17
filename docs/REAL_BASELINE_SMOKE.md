# Real baseline smoke

This document records the first completed real mREBEL end-to-end technical
smoke. It is not a reported full experiment.

## Environment

- Python: 3.12.14
- Conda environment: `re_te_system_312`
- torch: 2.14.0
- transformers: 4.57.6
- sentencepiece: 0.2.2
- device: CPU
- MPS: unavailable

## Model

- checkpoint: `Babelscape/mrebel-large`
- resolved Hugging Face revision:
  `50e0587ac7ac87d28b9abd069d72333528a5aa09`
- local-only loading: PASS

## Scientific configuration

- run role: baseline
- target condition: C0
- target schema knowledge: none
- input language: Spanish
- `REL_ALLOWED`: OFF
- aliases: none
- Hohfeld mapping: none
- generation: deterministic

## Documents

- `articulo_61`: 292 characters, 1 Gold annotation
- `articulo_8`: 3769 characters, 5 Gold annotations
- `articulo_12`: 13200 characters, 7 Gold annotations

## Extraction

- documents: 3/3 successful
- windows: 15
- raw outputs: 15
- parsed triples: 15
- validated triples: 15
- parse warnings: 0
- duplicates: 1, preserved
- structural violations: 1 soft duplicate

## Observed relation vocabulary

- `subclass of`: 7
- `part of`: 4
- `diplomatic relation`: 1
- `instance of`: 1
- `main subject`: 1
- `opposite of`: 1

## Evaluation

Predicate awareness:

- TP: 0
- FP: 15
- FN: 13
- precision: 0
- recall: 0
- F1: 0

Complete triplet with `canonical_argument` eligibility:

- TP: 0
- FP: 15
- FN: 12
- excluded Gold: 1
- precision: 0
- recall: 0
- F1: 0

## Interpretation

- The real end-to-end pipeline was validated.
- The expected native-schema mismatch was observed.
- There was no evidence of Hohfeld leakage.
- This was a technical smoke, not a reported full experiment.
- The REBEL real smoke was intentionally not performed.

The unversioned runtime artifacts remain at:
`runs/REAL_MREBEL_SMOKE/output/run-0eab94bf75499cb892b0/`.
