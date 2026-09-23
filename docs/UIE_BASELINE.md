# UIE external baseline

Integration date: 2026-09-22
Role: `STRUCTURAL_SCHEMA_GUIDED_UIE_BASELINE`
Task class: `UNIVERSAL_IE`
`UIE_ADAPTER_READY = YES`
`REAL_UIE_NATIVE_POSITIVE_SMOKE = FAIL`
`UIE_SCIENTIFIC_BASELINE_READY = PARTIAL`

## Scientific role and restrictions

UIE is an external universal-information-extraction comparator whose interface
accepts spot and association labels at inference through a Structural Schema
Instructor (SSI).

UIE is not:

- a C0, C1, C2, or C3 condition;
- a Hohfeld-native model;
- a Spanish-native model;
- generic end-to-end TE equivalent to mREBEL;
- evidence that arbitrary unseen schemas are reliably handled zero-shot.

The interface/generalization distinction is explicit:

```text
INTERFACE_ACCEPTS_CUSTOM_SCHEMA = YES
ZERO_SHOT_UNSEEN_SCHEMA_SCIENTIFICALLY_SUPPORTED = UNKNOWN
```

The original work is oriented toward pretraining followed by downstream
fine-tuning and low-resource experiments. Runtime schema acceptance alone does
not establish reliable arbitrary-label semantics.

## Provenance

| Field | Value |
|---|---|
| Upstream repository | `universal-ie/UIE` |
| Upstream commit | `c88dd08b8ad0016ec597ebc6f8fff49480367bb2` |
| Paper | *Unified Structure Generation for Universal Information Extraction* |
| Paper | https://aclanthology.org/2022.acl-long.395/ |
| Code license | CC BY-NC-SA 4.0 / non-commercial per README |
| Checkpoint | `luyaojie/uie-base-en` |
| Checkpoint revision | `966f8b1fc4c74e94ab552081605913ad5133cc41` |
| Checkpoint license metadata | CC BY-NC-SA 4.0 |

The checkpoint is author-published and directly linked by the official
repository. Its model card contains only license metadata; there is no
substantive model card or standalone weight license file. The checkpoint uses
legacy PyTorch pickle serialization rather than safetensors.

## Legacy reproduction

The frozen upstream was reproduced separately under `_legacy/UIE`:

- original documented runtime: Python 3.8, PyTorch 1.8.0, Transformers 4.6.1,
  CUDA 10.2/11.1;
- working Apple M2 runtime: Python 3.8.20, PyTorch 1.9.0,
  Transformers 4.6.1, CPU;
- checkpoint load and generation completed;
- exact raw SEL, parsed record, schema, SSI, and runtime metadata were saved;
- native smoke output was an empty but valid SEL tree.

That empty-tree run validates the legacy contract only. It is not extraction
quality evidence. See `_legacy/UIE/LEGACY_REPRODUCTION.md`.

## Checkpoint architecture

The pinned model is `T5ForConditionalGeneration`, initialized from
T5-v1.1-base and UIE-pretrained:

```text
encoder layers: 12
decoder layers: 12
d_model: 768
d_ff: 2048
attention heads: 12
feed-forward: gated-gelu
vocabulary: 32,102
parameter count: 247,537,920
tokenizer maximum: 512
stored dtype: FP32
```

Tokenizer assets include T5 SentencePiece, 100 `<extra_id_*>` sentinel tokens,
and UIE `<spot>`/`<asoc>` markers.

## Clean-room modules

| Module | Responsibility |
|---|---|
| `extractors/uie.py` | Pinned local-only model loading, deterministic generation, exact RAW SEL |
| `conditioning/uie_schema.py` | Baseline-specific schema value object, deterministic SSI and stable hash |
| `parsing/sel.py` | Conservative SEL parse, exact alignment, explicit relation projection |

No generic C0-C3 conditioning framework is introduced.

## Extractor policy

`UIEExtractor` implements the common `Extractor` protocol and records:

- requested and resolved checkpoint revision;
- tokenizer identity;
- device and dtype;
- source/target limits;
- generation settings;
- schema identity/hash;
- SSI version;
- exact input/output token counts;
- input truncation;
- output-limit status;
- whether special tokens were retained;
- constrained-decoding status.

Default loading is local-only. The clean-room implementation uses the current
Python 3.12-compatible PyTorch/Transformers stack and does not depend on the
legacy environment.

Generation defaults preserve the standalone upstream inference profile:

```text
max source tokens: 256
max target tokens: 192
do_sample: false
num_beams: 1
constraint_decoding: false
```

The decoded continuation uses `skip_special_tokens=False` and disables
tokenizer whitespace cleanup. `<pad>`, `</s>`, and all UIE structural tokens
remain in RAW.

## Structural schema and SSI

`UIESchema` independently preserves:

- ordered `spot_labels`;
- ordered `association_labels`;
- optional `spot_to_association` metadata;
- schema ID and version.

The third legacy `record.schema` line is preserved as metadata but is not
enforced during model inference or SEL parsing because the audited legacy code
loads but does not use it in those paths.

Schema order is preserved exactly. No sorting, randomization, examples,
definitions, aliases, Gold labels, or noise are added.

SSI version:

```text
uie-ssi-v1
```

Shape:

```text
<spot> {spot 1}<spot> {spot 2}<asoc> {association 1}<extra_id_2> {text}
```

Manifest metadata contains the ordered labels, optional mapping, ordering
policy, schema version/hash, SSI version, and exact serialized SSI.

## Structural vocabulary

| Token | Role |
|---|---|
| `<extra_id_0>` | Open outer structure, spot, or association |
| `<extra_id_1>` | Close structure |
| `<extra_id_2>` | Separate SSI from input text |
| `<extra_id_5>` | Separate a node label from its span |
| `<extra_id_6>` | Null/rejection span |
| `<spot>` | Introduce a spot label in SSI |
| `<asoc>` | Introduce an association label in SSI |

## Typed Spot-Association representation

SEL is not flattened immediately. `ParseResult` may retain:

```text
ParsedSpot
  label
  span
  segment_id
  exact alignment fields/status
  associations[]

ParsedAssociation
  label
  span
  exact alignment fields/status
```

Prediction artifacts for UIE add:

```text
parsed_structures
aligned_structures
```

`parsed_structures` contains the conservative structural parse before offset
alignment. `aligned_structures` contains a separate exact alignment result.
Existing baseline records do not add these fields.

## Conservative SEL parser

The P0 parser:

- preserves RAW exactly;
- accepts known `<pad>`/`</s>` wrappers without removing them from RAW;
- parses one complete, balanced SEL outer tree;
- retains unknown labels and emits issues;
- retains `<unk>` spans and emits issues;
- preserves occurrence order and duplicates;
- never repairs or deduplicates.

It does not:

- auto-close missing brackets;
- truncate after the first complete tree;
- replace malformed output with silent empty success;
- drop unknown labels;
- repair `<unk>`;
- drop unknown spans;
- fuzzy-map offsets;
- apply aliases;
- infer types from Gold;
- enforce spot-to-association metadata.

Relevant validation codes use the existing `PARSE_` prefix:

```text
PARSE_MALFORMED_SEL
PARSE_TRUNCATED_SEL
PARSE_UNKNOWN_LABEL
PARSE_UNKNOWN_SPAN
PARSE_UNEXPECTED_TOKEN
PARSE_EMPTY_STRUCTURE
PARSE_MODEL_FAILURE
```

Malformed current nodes are not fabricated. Structures parsed completely
before a later truncation/error remain visible alongside a hard parse issue.

## Upstream mutations not migrated

The audited upstream path:

- removes `<pad>` and `</s>`;
- trims whitespace;
- converts structural tokens to temporary brackets;
- truncates output after the first complete tree;
- adds missing closing brackets;
- replaces unparseable SEL with an empty tree;
- drops incomplete nodes;
- drops labels outside schema;
- repairs `<unk>` against source text;
- drops spans not found in source text;
- infers some target types;
- performs post-hoc offset remapping;
- can deduplicate offset records.

These behaviors are documented as reference semantics only. They are not
treated as model RAW or silently copied.

## Binary relation projection

Only an explicit Spot-Association pair projects to a canonical triple:

```text
spot.span
+ association.label
+ association.span
->
subject / relation / object
```

The source spot label is preserved as `subject_type`. No target type is
invented. Entity-only and event-only structures do not produce triples.
Association occurrences project independently, preserving duplicates.

The existing evaluation export can project resulting binary relations through
its normal subject/relation/object contract.

## Offset alignment

Alignment is separate from SEL parsing and is exact and case-sensitive:

- one occurrence -> exact document-relative offsets;
- multiple occurrences -> `ambiguous`, no forced first offset;
- no occurrence -> `not_found`, no offset;
- `<extra_id_6>` rejection span -> `null`, no offset or triple projection;
- `<unk>` -> retained and `not_found`.

Predictions are retained in every case. Ambiguous/absent spans generate
descriptive `PARSE_UNKNOWN_SPAN` violations. No fuzzy alignment or semantic
repair is performed.

## Duplicate policy

SEL structures and projected triples retain all occurrences. Existing
structural validation emits `DUPLICATE_TRIPLE` for repeated projected
subject/relation/object triples without deleting them.

## Constrained decoding

Default:

```text
constraint_decoding = false
```

The legacy reproduction confirms an optional `prefix_allowed_tokens_fn`
state-machine profile that constrains:

- SEL syntax;
- spot/association label membership;
- token-level source span generation.

It does not guarantee semantic correctness or type-to-association legality.
P0 records the disabled state and does not migrate the constraint subsystem.
A future optional profile may add it without changing unconstrained defaults.

## Manifest profile

```json
{
  "run_role": "baseline",
  "model_family": "uie",
  "task_class": "UNIVERSAL_IE",
  "target_schema_knowledge": "structural_schema",
  "controlled_experiment_condition": "not_applicable",
  "input_language": "es",
  "language_status": "out_of_documented_training_scope",
  "native_schema": {
    "inherited_from_model": false,
    "scope": "dynamic_runtime_structural_schema"
  }
}
```

The compatibility `condition: C0` field remains in the shared contract but
does not place UIE in the controlled C0-C3 experiment.

Model metadata additionally records:

```text
interface_accepts_custom_schema = true
zero_shot_unseen_schema_supported = unknown
parser_policy = conservative_sel_v1
constraint_decoding = false
```

## Language status

The released checkpoint is English. Spanish tokenizer acceptance is not
documented Spanish support. Runs on Spanish inputs use:

```text
language_status = out_of_documented_training_scope
```

No Spanish performance claim is made.

## Hohfeld interface fixture

Tests verify that the schema layer can structurally represent association
labels:

```text
Right
Duty
Privilege
NoRight
```

The fixture uses only generic synthetic spot labels and no Gold examples,
definitions, aliases, entity semantics, or Hohfeld text.

```text
HOHFELD_ZERO_SHOT_INTERFACE_POSSIBLE = YES
HOHFELD_ZERO_SHOT_SCIENTIFICALLY_SUPPORTED = UNKNOWN
HOHFELD_FINETUNING_REQUIRED = LIKELY
```

No Hohfeld inference or fine-tuning is performed in P0.

## Runtime and smoke status

The clean-room code path is covered by weight-free fixtures and one real
canonical-runtime CPU smoke. The pinned checkpoint was downloaded into the
repository-local ignored Hugging Face cache.

The real smoke used the fixed documented MULTAN relation example, native
schema, deterministic generation, and no constraints. Loading, generation,
RAW persistence, parsing, and artifact/manifest creation succeeded, but the
model emitted:

```text
<pad><extra_id_0><extra_id_1></s>
```

No Spot-Association structure or binary relation was produced. The protocol
was not modified after the result. Evidence indicates `uie-base-en` is a
pretrained base checkpoint intended for downstream task fine-tuning, while the
legacy standalone helper defaults to a task-specific ABSA model directory.

CPU execution was practical; GPU is neither the blocker nor required for P0
code readiness. See `UIE_REAL_NATIVE_SMOKE.md`.

## Comparison restrictions

- Report UIE under `UNIVERSAL_IE`, not `END_TO_END_TE`.
- Do not rank UIE and mREBEL without task/input/schema qualifications.
- Do not interpret runtime schema labels as C1/C2/C3 treatment.
- Do not claim unseen-schema zero-shot support.
- Do not claim Spanish support.
- Do not score entity/event-only structures as fabricated triples.
- Do not treat parser repair/filtering as model output.

`UIE_BASELINE_P0_READY = YES`
