# GoLLIE external baseline

Integration date: 2026-09-23
Role: `GUIDELINE_FOLLOWING_UIE_BASELINE`
Task class: `UNIVERSAL_IE`
`GOLLIE_ADAPTER_READY = YES`
`REAL_GOLLIE_NATIVE_SMOKE = BLOCKED_GPU`
`GOLLIE_SCIENTIFIC_BASELINE_READY = PARTIAL`

## Scientific role and restrictions

GoLLIE is an external guideline-following universal-information-extraction
comparator. It receives a Python-like runtime schema whose classes carry
structural signatures and natural-language definitions/guidelines.

GoLLIE is not:

- a C0, C1, C2, or C3 condition;
- Controlled C2;
- an `END_TO_END_TE` model;
- a KBP model;
- a Hohfeld-native model;
- a Spanish-native model;
- evidence that a later Hohfeld schema has been defined or run.

The scientific role is:

```text
GUIDELINE_FOLLOWING_UIE_BASELINE
run_role = baseline
model_family = gollie
task_class = UNIVERSAL_IE
controlled_experiment_condition = not_applicable
target_schema_knowledge = structural_schema+definitions_guidelines
```

Target knowledge is represented as two separable components:

```text
STRUCTURAL_SCHEMA
DEFINITIONS_GUIDELINES
```

P0 does not flatten those components into one opaque prompt-only string. The
exact Black-formatted prompt is also stored after serialization.

## GoLLIE is not Controlled C2

GoLLIE is not Controlled C2.

GoLLIE may receive knowledge that overlaps the C2 inventory:

- labels / class names;
- definitions;
- types;
- argument roles;
- signatures;
- descriptive guidelines.

The overlap does not make GoLLIE a C2 condition:

- it is a different model;
- it is specifically trained for guideline following;
- its prompt representation differs from the future controlled generator;
- it may carry examples or exceptions as guideline metadata;
- its training regime differs;
- it is not part of same-model causal contrast.

Do not interpret a GoLLIE versus C0/C2 difference as a C0→C3 conditioning
effect.

## Provenance

| Field | Value |
|---|---|
| Upstream repository | `hitz-zentroa/GoLLIE` |
| Frozen upstream commit | `164c611743fdc1befe71bbdf03e08c5eb4e35957` |
| Local read-only tree | `_legacy/GoLLIE` |
| Paper | *GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction* |
| Checkpoint | `HiTZ/GoLLIE-7B` |
| Checkpoint revision | `d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9` |
| Architecture | merged full-model Code Llama 7B decoder |
| Base model | `codellama/CodeLlama-7b-hf` |
| Custom modeling | required (`trust_remote_code=True`) |
| Code / adapter license | Apache-2.0 |
| Weight terms | Llama 2 |
| Documented language | English only |
| FlashAttention / CUDA | required for the real extractor |

The selected checkpoint is a merged full model, not a LoRA-only adapter. Remote
download is disabled by default. Unit tests never download weights.

## Legacy reproduction

The frozen upstream was audited separately under `_legacy/GoLLIE`. That audit
is not rewritten here. The clean-room adapter does not import runtime code
from `_legacy/GoLLIE`. Official RE wording used by the native fixture is taken
from the audited official notebook example, not invented.

## Clean-room modules

| Module | Responsibility |
|---|---|
| `extractors/gollie.py` | Pinned local-only CUDA/FlashAttention loading, deterministic generation, exact RAW continuation |
| `conditioning/gollie_schema.py` | Typed Entity/Relation/Event/Template schema, separate structural and guideline hashes, Black prompt serializer |
| `parsing/gollie.py` | Safe AST parse-only interpreter, exact alignment, binary relation projection |

Shared contracts, runner, validation, and evaluation export are reused. No
generic C0-C3 conditioning framework is introduced.

## Schema and guideline model

A GoLLIE class preserves:

- class name;
- kind (`entity`, `relation`, `event`, `template`);
- inheritance/base type;
- natural-language definition/guideline;
- ordered arguments;
- argument names and types;
- field comments/guidelines where present;
- optional examples as a separate field.

Relation extraction is the minimum supported projection in P0. Labels alone
are not a sufficient schema representation.

Manifest metadata keeps:

```text
structural_schema
definitions_guidelines
formal_ontology = false
dynamic_runtime_schema = true
```

Schema hash covers structure only. Guideline hash covers definitions, field
guidelines, and examples. Changing a docstring does not change the schema
hash.

## Prompt serializer and Black policy

Prompt generation is deterministic and is produced from explicit schema
objects. `inspect.getsource` is not part of the scientific contract.

Policy A is used: Black is an explicit serialization contract.

```text
prompt_serializer_version = gollie-prompt-v1
black_policy = explicit_black_serialization_contract
black_target_version = black>=24.8,<25
black_line_length = 119
```

Black versions are recorded in metadata. The exact serialized prompt is stored
after formatting. The prompt is not silently reformatted after hashing.

Expected conceptual shape:

```text
Python-like class definitions
+ docstrings
+ field definitions/comments
+ input text
+ result prefix
```

## Extractor policy

`GoLLIEExtractor` implements the common `Extractor` protocol and records:

- model name and immutable revision;
- tokenizer identity;
- device and dtype;
- generation parameters;
- prompt serializer version;
- schema hash and guideline hash;
- exact serialized prompt;
- input/output token counts;
- truncation;
- FlashAttention status;
- quantization status (`none`);
- exact RAW decoded continuation.

Default loading is local-only. If CUDA or FlashAttention is unavailable, real
extractor initialization fails clearly. Mocks and fixtures remain
network/GPU-free.

P0 does not implement CPU, MPS, Ollama, GGUF, vLLM, or alternative attention
backends. Those would be new runtime variants.

```text
REAL_GOLLIE_NATIVE_SMOKE = BLOCKED_GPU
```

## Safe parser

Generated output is never executed. The parser uses Python `ast` in parse-only
`eval` mode and accepts only:

- `List`
- `Call`
- `Name`
- constant strings/numbers/bools/null
- keyword arguments

It rejects `Attribute`, `Subscript`, `Lambda`, comprehensions, `BinOp`,
function definitions, imports, star args, `**kwargs`, and nested executable
constructs.

`eval()` and `exec()` are not used.

Typed intermediates are `ParsedGoLLIERecord` values (`entity`, `relation`,
`event`, `template`, or `unknown`). Unknown classes remain visible. Parser
issues use the shared short codes and become `PARSE_*` violations:

```text
INVALID_GOLLIE_SYNTAX
UNSAFE_AST_NODE
UNKNOWN_CLASS
MISSING_REQUIRED_ARGUMENT
UNKNOWN_ARGUMENT
ARGUMENT_TYPE_MISMATCH
EMPTY_STRUCTURE
MODEL_FAILURE
```

Model output is never deleted because of violations.

## Relation projection

Only schema-declared Relation classes with string `arg1`/`arg2` project to
canonical triples:

```text
arg1 value + relation class name + arg2 value
```

Entities, events, templates, malformed calls, and unknown class signatures do
not fabricate triples. Relation argument metadata remains on the typed
records.

## Offset alignment

Alignment is a separate exact, case-sensitive stage. First-case-insensitive
match is not hidden truth.

- one exact match -> `exact`
- multiple exact matches -> `ambiguous`
- absent -> `not_found`

A later case-insensitive profile can be added if justified. Predictions are
retained in every case.

The official Ana/Mary sentence mentions each name twice, so the native fixture
is expected to be `ambiguous`. That is the correct P0 behavior.

## Duplicate policy

Typed records and projected triples retain all occurrences and order. No
upstream mention-count or scorer deduplication is applied. Shared structural
validation emits descriptive `DUPLICATE_TRIPLE` violations without deletion.

## Native fixtures

The frozen weight-free native fixture uses the official RE notebook text and
the audited `PhysicalRelation` / `PersonalSocialRelation` wording, including
the documented "describe" spelling. Synthetic RAW

```text
[
    PersonalSocialRelation(arg1="Ana", arg2="Mary")
]
```

validates parser and contracts only. It is not evidence that
`HiTZ/GoLLIE-7B` generated that output.

Additional fixtures:

- hallucinated `UnknownRelation` -> structure retained, `PARSE_UNKNOWN_CLASS`;
- `__import__("os").system("echo unsafe")` -> `PARSE_UNSAFE_AST_NODE`, nothing
  executed.

## Manifest profile

```json
{
  "run_role": "baseline",
  "model_family": "gollie",
  "task_class": "UNIVERSAL_IE",
  "target_schema_knowledge": "structural_schema+definitions_guidelines",
  "controlled_experiment_condition": "not_applicable",
  "input_language": "es",
  "language_status": "out_of_documented_training_scope",
  "native_schema": {
    "inherited_from_model": false,
    "scope": "dynamic_runtime_guideline_schema",
    "dynamic_runtime_schema": true,
    "definitions_guidelines": true,
    "formal_ontology": false
  }
}
```

English native-schema runs use `language_status: supported`. The default
Hohfeld-oriented input language remains `es` and is marked
`out_of_documented_training_scope`. No Spanish performance claim is made.

The compatibility `condition: C0` field remains in the shared contract but
does not place GoLLIE in the controlled C0-C3 experiment.

## Hohfeld status

No Hohfeld guideline definitions are created in P0.
No guidelines are translated.
No Spanish Hohfeld run is executed.
No class mapping or fine-tuning is performed.
No C2 implementation is modified.

The adapter is schema-generic.

## Runtime and smoke status

P0 code readiness does not require real inference. A future command is
prepared by `scripts/run_gollie_native_smoke.py` for:

```text
HiTZ/GoLLIE-7B
+ official native RE example
NVIDIA CUDA GPU
official/custom FlashAttention path
```

The script emits the command only. It does not download or run weights. When
that infrastructure exists, the real extractor must write `manifest.json`,
`predictions.jsonl`, `failures.jsonl`, and `statistics.json` like the other
adapters.

## Comparison restrictions

- Report GoLLIE under `UNIVERSAL_IE` as `GUIDELINE_FOLLOWING_UIE_BASELINE`.
- Do not report GoLLIE as Controlled C2.
- Do not rank GoLLIE against UIE or mREBEL without task/schema qualifications.
- Do not treat guideline overlap as same-model causal contrast.
- Do not claim Spanish support.
- Do not score entity/event/template records as fabricated triples.
- Do not treat the official synthetic RAW fixture as a real checkpoint result.

`GOLLIE_BASELINE_P0_READY = YES`
