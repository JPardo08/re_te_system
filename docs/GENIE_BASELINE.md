# GenIE external baseline

Integration date: 2026-09-23
Role: `CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE`
Task class: `KBP_CLOSED_IE`
`GENIE_ADAPTER_READY = YES`
`REAL_GENIE_NATIVE_SMOKE = BLOCKED_RESOURCE`
`checkpoint_loading_status = not_real_smoked`
`modernized_loading_status = unverified`
`GENIE_SCIENTIFIC_BASELINE_READY = PARTIAL`

## Scientific role and restrictions

GenIE is an external closed-schema constrained information-extraction
comparator. Supervised training internalizes a fixed Wikidata-like native
schema. At inference it may additionally apply hard entity and relation
inventory constraints.

Hard schema membership constraints are not semantic ontology understanding.

GenIE is not:

- a C0, C1, C2, or C3 condition;
- formal ontology reasoning;
- UIE / SSI / SEL;
- Hohfeld-native;
- a same-model controlled condition;
- evidence that a later Hohfeld schema has been defined or run.

```text
run_role = baseline
model_family = genie
task_class = KBP_CLOSED_IE
scientific_role = CLOSED_SCHEMA_CONSTRAINED_IE_BASELINE
target_schema_knowledge = fixed_native_schema+kb_constraints
controlled_experiment_condition = not_applicable
```

Knowledge-source taxonomy:

```text
FIXED_NATIVE_SCHEMA
KB_CONSTRAINTS
```

## Provenance

| Field | Value |
|---|---|
| Upstream repository | `epfl-dlab/GenIE` |
| Frozen upstream commit | `c53d52f598c2940e00f753c97b3baf4ba144fc03` |
| Local read-only tree | `_legacy/GenIE` |
| Paper | Josifoski et al., NAACL 2022 |
| Selected checkpoint | `genie_r.ckpt` |
| Source | Zenodo `6139236` |
| Size | 4,879,373,206 bytes |
| MD5 | `c214da56b6e5d5bd259e0cbe826d92f7` |
| Training | REBEL |
| Initialization | random |
| Native default constraints | large schema |
| Architecture | BART-large / GenieHF |
| Code license | MIT |
| Checkpoint license | CC-BY-4.0 |
| Loading form | `LIGHTNING_CHECKPOINT_NATIVE` |

P0 does not download the checkpoint. No weights are committed.
`martinjosifoski/genie-rw` is a tokenizer-and-config repository and does not
contain model weights.

## Runtime boundary

Original upstream environment:

```text
Python 3.8
PyTorch 1.8
Lightning 1.4.9
Transformers 4.10
Hydra 1.1
```

The future clean-room inference target is direct BART / Transformers loading
where scientifically equivalent. P0 does **not** claim that the Lightning
checkpoint has been modernized. `EXTRACTED_HF_STATE_DICT` is unnamed as a
verified path.

`pytorch-lightning` and Hydra are not core dependencies.

## Tokenizer

| Field | Value |
|---|---|
| Identity | `martinjosifoski/genie-rw` |
| Revision | `81eb2ccca714fbb62a65283b65c9c153469ee409` |
| Source of revision | Hugging Face model SHA of the tokenizer-only repo |
| Added special tokens | none |

Control tags are ordinary leading-space BPE:

```text
 <sub>  -> <s> Ġ< sub > </s>
 <rel>  -> <s> Ġ< rel > </s>
 <obj>  -> <s> Ġ< obj > </s>
 <et>   -> <s> Ġ< et > </s>
```

## Output grammar

RAW is the exact decoded string. Official grammar:

```text
 <sub> SUBJECT <rel> RELATION <obj> OBJECT <et>
```

Multiple occurrences concatenate. Upstream `textual_triplets` sets are never
used as RAW.

## Parser

`parsing/genie.py` is ordered and conservative:

- preserves occurrence order;
- preserves duplicates;
- requires the explicit tag cycle;
- does not silently drop incomplete suffixes;
- does not convert to a set;
- does not canonicalize text semantically.

Typed intermediate fields: `subject_text`, `relation_text`, `object_text`,
`occurrence_index`, optional inventory status, optional later IDs.

Violations, after the shared `PARSE_` prefix:

```text
PARSE_MALFORMED_GENIE
PARSE_INCOMPLETE_TRIPLET
PARSE_UNEXPECTED_CONTROL_TOKEN
PARSE_EMPTY_STRUCTURE
PARSE_MODEL_FAILURE
```

Duplicate textual occurrences may also receive a descriptive
`PARSE_DUPLICATE_OCCURRENCE` / `DUPLICATE_TRIPLE` violation while remaining
in the prediction.

## Constraint profiles

Canonical native profiles:

```text
unconstrained
small
large
custom_full
```

`small` and `large` are official inventory labels, not millions of strings in
Git. Constrained decoding, when actually applied, requires **both**:

- an entity inventory;
- a relation inventory.

A custom relation trie alone is not native GenIE constrained inference.

`RELATION_ONLY` is not a canonical profile. A future relation-only experiment
would be `NON_NATIVE_ABLATION`.

Gold-derived entity inventories are forbidden because they leak test entities.

## Tries

Official downloaded artifacts include `*_original_strings.jsonl` for small and
large entity/relation tries.

```text
OFFICIAL_TRIE_DESERIALIZATION = RECONSTRUCT_FROM_STRINGS_PREFERRED
```

Future real runtime should rebuild tries from those official string inventories
plus the pinned tokenizer when feasible. Tests never unpickle official tries.
The clean-room `TokenPrefixTrie` uses leading space, drops leading BOS, and
sorts the string iterable.

## Beams

GenIE natively uses beam search. All returned candidates are retained with
`beam_rank`, RAW, and log probability. Evaluator export uses
`top_score_beam0`. There is no post-hoc best-by-Gold selection.

## ID mapping

Optional and separate from parsing. Outcomes are `resolved`, `ambiguous`, or
`unresolved`. Textual triples are kept if mapping fails. Last-write-wins
collision semantics are not ported.

## KG-readiness

Conceptual dimensions only; the generic evaluator is unchanged:

- structurally complete triple;
- relation inventory membership;
- subject inventory membership;
- object inventory membership;
- optional ID resolvability.

## Language

English is the documented native scope (`language_status: supported` for
`input_language: en`). Spanish and any future Spanish Hohfeld run are
`out_of_documented_training_scope`. A BART tokenizer does not imply Spanish
support.

## Official notebook fixtures

All three Carson outputs are `OFFICIAL_NOTEBOOK_RECORDED_OUTPUT`, not local
generations.

Input:

```text
Prior to KTRK, Carson was an anchor for KSAZ in Phoenix, Arizona.
```

| Profile | Best RAW |
|---|---|
| unconstrained | ` <sub> KSAZ-TV <rel> headquarters location <obj> Phoenix, Arizona <et>` |
| small | Phoenix/Arizona capital pair |
| large | same as unconstrained |

`KSAZ-TV` is absent from the small entity inventory. Constrained small
decoding cannot emit it.

## Real smoke

`scripts/run_genie_native_smoke.py` prepares unconstrained / small / large
commands. It does not download `genie_r.ckpt` unless a future explicit user
flag is added and a verified loader exists. P0 execution is not required.

When executed through `genie-mock`, the common artifacts are written:

```text
manifest.json
predictions.jsonl
failures.jsonl
statistics.json
```

CPU inference is technically supported for BART-large. The current local
blocker is the 4.9 GB Lightning checkpoint plus the large entity trie on a
16 GB machine, together with unverified modernized loading.
