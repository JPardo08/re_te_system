# Pythia / SPACE-KBP external baseline

Audit and integration date: 2026-09-22
Role: `KBP_DOMAIN_BASELINE`
Checkpoint provenance gate: `PARTIAL`
`REAL_MODEL_SMOKE_AUTHORIZED = NO`

## Scientific role

This adapter represents the Pythia model family fine-tuned to populate the
SPACE-KBP space-mission ontology. It is an external, ontology-specialist
knowledge-base-population baseline.

It is not:

- a generic triplet-extraction model;
- a model that accepts arbitrary unseen ontologies;
- a Hohfeld extractor;
- a C0, C1, C2, or C3 controlled condition;
- evidence about the causal effect of ontology conditioning.

The external baseline preserves the shared system path:

```text
TEXT
-> RAW
-> PARSED
-> NORMALIZED
-> VALIDATED
-> predictions.jsonl
-> manifest.json
-> evaluation export
```

The current prediction and extraction contracts remain version 1.0. Optional
RDF metadata fields were added to `ParsedTriple` because the required Turtle
representation cannot otherwise preserve URI/literal/datatype/language
information. Absent RDF metadata is omitted from serialized REBEL/mREBEL
triples, so their existing artifact shape and parser contracts remain
compatible.

## Task classification and manifest identity

Pythia runs use:

```text
run_role = baseline
model_family = pythia_spacekbp
task_class = KBP_DOMAIN_BASELINE
target_schema_knowledge = none
controlled_experiment_condition = not_applicable
native_schema.id = space_kbp_space_ontology
native_schema.scope = fixed_domain_ontology
```

For backward compatibility, the shared `condition` and
`target_condition.id` fields remain `C0`. The explicit
`controlled_experiment_condition = not_applicable` field is authoritative for
scientific analysis. Pythia runs must be excluded from C0-C3 causal contrasts.

## Lineage

The primary lineage is:

```text
EleutherAI/pythia-1b-deduped
-> supervised fine-tuning on SPACE-KBP
-> expertailab/pythia-1B-deduped-turtle-only-response-best
-> RDF Turtle generation for the fixed space-mission ontology
```

The legacy identifier
`anonkbp/pythia-1B-deduped-turtle-only-response-best` currently resolves
through the Hugging Face API to
`expertailab/pythia-1B-deduped-turtle-only-response-best`. Both API paths
report the same repository object and revision.

The canonical study is *Autoregressive Language Models for Knowledge Base
Population: A case study in the space mission domain*:

- Paper: https://arxiv.org/abs/2503.18502
- Project/code: https://github.com/expertailab/LLM4KBP
- Audited project revision: `0db8089966041a998ac2f3ab96b400a15e91dba9`
- Dataset: https://huggingface.co/datasets/expertailab/SPACE-KBP
- Base Pythia project: https://github.com/EleutherAI/pythia

The official project repository states that its `SFTPythiaModels` notebooks
train Pythia models on the KBP task and evaluate their generated Turtle. The
checkpoint card identifies the SPACE-KBP dataset and Pythia-1B base. The card
does not cite the paper or describe training. The pinned official training
notebook uses the same checkpoint naming family and saves the complete
model/tokenizer, which establishes reasonable technical attribution while the
minimal model card remains a provenance limitation.

## Checkpoint provenance gate

### Result: `PARTIAL`

The hard-stop identity requirements are established:

| Requirement | Result | Evidence |
|---|---|---|
| Canonical checkpoint | PASS | `expertailab/pythia-1B-deduped-turtle-only-response-best` |
| Current owner | PASS | Hugging Face author `expertailab` |
| Immutable revision | PASS | `8362933633d3584b2e572dd2c31f5190d91f8d87` |
| Full model vs adapter | PASS | One full F32 `model.safetensors`; no PEFT adapter files |
| Tokenizer | PASS | Included `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`; class `GPTNeoXTokenizer` |
| Architecture/base | PASS | `GPTNeoXForCausalLM`; card base tag `EleutherAI/pythia-1b-deduped` |
| Weight composition | PASS | Single safetensors file with 1,011,675,136 F32 parameters |
| Fine-tuned weights license | **UNRESOLVED** | No license tag/text in the checkpoint card or repository |
| Base-model license | PASS | Apache-2.0 |
| SPACE-KBP dataset license | PASS | CC BY-NC 4.0 |
| System attribution | PARTIAL | Owner, name, base, dataset, and project align; checkpoint card lacks paper/training detail |

Because identity, immutable revision, and tokenizer/base relationship are
established, clean-room adapter and fixture work is allowed. Because the
fine-tuned weights license is not stated, real loading/smoke is not authorized
and remote downloads are disabled by default.

### Immutable checkpoint composition

At revision `8362933633d3584b2e572dd2c31f5190d91f8d87`:

| File | Size / role |
|---|---|
| `model.safetensors` | 4,046,723,592 bytes; LFS SHA-256 `a06370578f877115e41d5cc462d10aee8a9e7c20c860ee4d25a59f6d5b933ad7` |
| `config.json` | GPT-NeoX/Pythia model configuration |
| `generation_config.json` | BOS/EOS configuration |
| `tokenizer.json` | 3,564,583-byte tokenizer artifact |
| `tokenizer_config.json` | `GPTNeoXTokenizer`, added `[PAD]`, tokenizer metadata |
| `special_tokens_map.json` | BOS/EOS/UNK and `[PAD]` mapping |
| `README.md` | Base-model and dataset metadata only |

The repository contains no `.bin` weight, adapter config, or adapter weight.
The Hugging Face safetensors metadata reports 1,011,675,136 F32 parameters.
This is a complete fine-tuned checkpoint, not a LoRA/PEFT adapter.

### Base model and tokenizer

Base model:

- ID: `EleutherAI/pythia-1b-deduped`
- Observed base revision: `7199d8fc61a6d565cd1f3c62bf11525b563e13b2`
- Architecture: `GPTNeoXForCausalLM`
- License: Apache-2.0
- Language documented by the base card: English

Fine-tuned config:

- 16 hidden layers;
- hidden size 2,048;
- 8 attention heads;
- vocabulary size 50,278;
- `max_position_embeddings = 2048`;
- F32 checkpoint.

The task checkpoint includes its tokenizer, so the adapter uses the tokenizer
from the same pinned task repository. Its `tokenizer_config.json` declares
`GPTNeoXTokenizer`, end-of-text BOS/EOS/UNK, and `[PAD]`. The tokenizer's
serialized `model_max_length` is an unbounded sentinel and is not trusted for
windowing. The official training notebook derives it from the base
`step143000` revision, adds `[PAD]`, resizes embeddings, and saves it beside the
fine-tuned model. The base card states that `step143000` is identical to
`main`. The adapter derives the real model limit from
`config.max_position_embeddings`.

## License status

Licenses are separate and must not be conflated:

| Artifact | License status |
|---|---|
| EleutherAI Pythia base code/model | Apache-2.0 |
| SPACE-KBP / LLM4KBP code | Apache-2.0 |
| SPACE-KBP dataset | CC BY-NC 4.0 |
| Fine-tuned `expertailab` checkpoint | **Unknown / undeclared** |
| This clean-room adapter | Covered by the `re_te_system` repository terms |

Apache-2.0 on the base model or training code does not automatically establish
the license of the fine-tuned weights. The dataset's non-commercial terms also
do not substitute for an explicit checkpoint license.

Primary machine-readable evidence:

- Checkpoint API:
  https://huggingface.co/api/models/expertailab/pythia-1B-deduped-turtle-only-response-best
- Pinned tree:
  https://huggingface.co/api/models/expertailab/pythia-1B-deduped-turtle-only-response-best/tree/8362933633d3584b2e572dd2c31f5190d91f8d87?recursive=true&expand=true
- Dataset API:
  https://huggingface.co/api/datasets/expertailab/SPACE-KBP
- Base model API:
  https://huggingface.co/api/models/EleutherAI/pythia-1b-deduped

## Adapter and prompt profile

`PythiaSpaceKBPExtractor` implements the common `Extractor` protocol and
retains:

- pinned `model_name` and `model_revision`;
- exact decoded continuation in `RawExtractionResult.model_output`;
- input/output token counts;
- model and effective context limits;
- input truncation and output-limit status;
- prompt profile/version;
- quantization mode;
- segment/window provenance through the shared runner.

P0 implements only `basic-v1`. It neutrally requests RDF Turtle for the stated
text and contains no example, ontology vocabulary, Gold/Silver data, or
Hohfeld label. It is a clean-room prompt and is not the legacy Aqua prompt.

No `published` profile is implemented. The checkpoint card contains no
official inference template, and the public repository's training prompts are
experiment assets rather than a sufficiently documented, licensed checkpoint
inference contract.

## Input, length, and truncation policy

The adapter formats one source segment with the selected prompt profile. The
actual model context is read from `max_position_embeddings` (2,048 for the
pinned checkpoint).

```text
available_input = model_max_length - max_new_tokens
effective_input_limit = min(optional_user_limit, available_input)
```

With the P0 default `max_new_tokens = 512`, the effective maximum prompt/input
length is 1,536 tokens. Token windowing uses the adapter's prompt-aware token
counter. The legacy 1,024-token cutoff is not reused.

Every result records:

- `model_max_length`;
- `effective_input_limit`;
- `max_new_tokens`;
- token counts before and after truncation;
- `truncated_input`;
- `output_reached_limit`.

Reaching the output limit emits a descriptive
`PARSE_TRUNCATED_GENERATION` validation violation. Input truncation remains
explicit in RAW generation metadata. Nothing is silently truncated.

## Output and RAW preservation

The generated continuation is decoded once with special tokens omitted and
tokenizer whitespace cleanup disabled. The returned decoded string is not
stripped, repaired, completed, deduplicated, or semantically normalized before
being stored in RAW.

The shared prediction artifact retains:

- exact top-level decoded RAW;
- exact decoded RAW for every segment;
- generation metadata;
- parsed triples;
- normalized triples;
- validated triples and violations.

Parser context can supply external prefix mappings, but those declarations are
separate parser input context. They never replace or overwrite the stored RAW.
There is no built-in SPACE namespace.

## Conservative Turtle parser

`parse_turtle` uses RDFLib and:

- parses Turtle without regex fallback;
- preserves parser emission order;
- records duplicate triple occurrences before RDF graph set semantics collapse
  them;
- represents URI terms with full URI strings;
- represents blank nodes with deterministic parser-local identifiers;
- preserves literal lexical values, datatype, and language;
- accepts optional externally supplied prefixes;
- reports syntax/prefix/empty-graph/non-URI-predicate issues.

The internal triple representation preserves:

- `subject`, `relation` (RDF predicate), and `object`;
- `subject_is_uri`;
- `predicate_is_uri`;
- `object_is_uri`;
- `object_is_literal`;
- `datatype`;
- `language`.

The evaluation export intentionally projects only `subject`, `relation`, and
`object`, matching the existing evaluator contract.

If RDFLib cannot parse the output:

- parsed triples are empty;
- a hard `INVALID_TURTLE` or `PREFIX_RESOLUTION_FAILURE` parse issue is emitted;
- validation records the corresponding `PARSE_*` violation;
- RAW remains unchanged.

The parser never invents closing syntax, repairs subjects, truncates to a
parseable prefix, adds relations, applies aliases, maps Hohfeld labels, or
silently removes duplicates.

## Validation policy

Validation is descriptive. Relevant codes are:

- `PARSE_INVALID_TURTLE`;
- `PARSE_EMPTY_GRAPH`;
- `PARSE_NON_URI_PREDICATE`;
- `PARSE_PREFIX_RESOLUTION_FAILURE`;
- `DUPLICATE_TRIPLE`;
- `PARSE_TRUNCATED_GENERATION`;
- `PARSE_MODEL_FAILURE`.

The `PARSE_` prefix is the existing shared validation convention for parser or
generation issues. The triple or prediction that caused a violation is not
deleted. Model exceptions are also retained in `failures.jsonl`.

## Quantization and dependencies

P0 supports standard loading only:

```text
quantization = none
```

bitsandbytes is not required or installed by the `pythia` extra. Any future
quantized path must be one explicit option, included in the manifest and tested
separately.

Install the optional environment with:

```bash
conda run -n re_te_system_312 python -m pip install -e ".[dev,pythia]"
```

The optional extra contains current Python 3.12-compatible versions of
RDFLib, PyTorch, and Transformers. It does not pin the old legacy stack.

## Differences from the legacy implementation

| Legacy behavior | P0 clean-room behavior |
|---|---|
| Mutable `anonkbp` default | Canonical ID and immutable revision pinned |
| Aqua few-shot example | Neutral versioned `basic-v1` prompt only |
| Hardcoded SPACE prefixes | No built-in namespaces; optional external parser context |
| Prefix stripping and subject repair | No RAW mutation or repair |
| Truncation closure and regex rescue | Parse failure with explicit violation |
| RDF graph iteration | Parser emission order with duplicate occurrences retained |
| CSV-only flattened output | Versioned PredictionRecord with exact RAW and stages |
| Default deduplication | Duplicates retained and reported |
| Hidden 1,024-token cutoff | Model-config-derived, prompt-aware limit with metadata |
| Optional undeclared bitsandbytes | Standard load only; quantization `none` |
| Unpinned model/tokenizer | Pinned task model, revision, and included tokenizer |
| Prediction-derived aliases | No aliases |
| No controlled-condition marker | `controlled_experiment_condition = not_applicable` |

No legacy code, prompt, regex, aliases, or result files were copied.

## Known limitations and comparison restrictions

- Fine-tuned weights have no declared license.
- The checkpoint card has minimal provenance and no inference template.
- Only English SPACE-KBP training is documented.
- The model is tied to a fixed space-mission ontology.
- The basic clean-room prompt is not claimed to reproduce the paper protocol.
- No arbitrary ontology or Hohfeld-schema following is supported.
- No entity canonicalization or KB identifier grounding is added.
- Turtle syntax validity does not imply ontology or factual validity.
- External prefix context must be supplied and recorded when required.
- Document-level generation can still be expensive and output-limited.
- The compatibility `condition = C0` field must not be used to include Pythia
  in the controlled C0-C3 analysis.
- Pythia scores, if produced later, belong to a separate
  `KBP_DOMAIN_BASELINE` stratum.

## Test and smoke status

Fixture/unit coverage verifies:

- common protocol compatibility;
- pinned/local-only configuration;
- deterministic mock output;
- exact RAW preservation;
- valid and invalid Turtle behavior;
- prefix context without RAW mutation;
- duplicates;
- URI/literal/datatype/language fields;
- deterministic emission ordering;
- Pythia manifest role and controlled-condition exclusion;
- absence of Hohfeld knowledge and legacy aliases;
- truncation and model-failure violations;
- evaluation-export compatibility;
- deterministic artifacts.

No network is required by tests. No weight file was downloaded.

```text
CHECKPOINT_PROVENANCE_GATE = PARTIAL
REAL_MODEL_SMOKE_AUTHORIZED = NO
REAL_SPACEKBP_SMOKE = NOT_RUN
OUT_OF_NATIVE_DOMAIN_TECHNICAL_SMOKE = NOT_RUN
```

The remaining authorization blocker is an explicit license for the fine-tuned
checkpoint, plus preferably a fuller model card tying the immutable artifact
to the published training protocol.
