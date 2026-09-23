# Run manifest 1.0

The manifest records:

- content-derived `run_id`;
- dataset id, version, and documents SHA-256;
- benchmark documents and manifest identities;
- `run_role`, `model_family`, model/checkpoint, requested/resolved revisions,
  tokenizer, device, and dtype;
- backward-compatible `condition: C0` and `conditioning: none`;
- structured `target_condition`, `target_schema_knowledge`, and
  `native_schema`;
- optional `task_class` and `controlled_experiment_condition` when a baseline
  must be kept outside the controlled-condition namespace;
- `input_language` and explicit `language_status`;
- every output-affecting generation parameter;
- window strategy, maximum units, overlap, mechanism, and relation-filter
  policy;
- package version and source commit state;
- prediction and manifest contract versions.

`run_id` is the first 20 hexadecimal characters of SHA-256 over canonical JSON
of stable scientific fields. It is not timestamp-derived. A dirty or unborn
repository is recorded as `UNCOMMITTED`; callers may supply a pinned source
identity for reproducible release runs.

Operational `created_at` is null in P0 so repeat executions with identical
scientific configuration produce byte-identical artifacts. Mutable model names
without an exposed Hugging Face commit are recorded as `UNKNOWN_UNPINNED`.

For deterministic beam search (`do_sample=false`), the seed is recorded for
configuration completeness but is not claimed as a source of determinism.

## External baseline profiles

All external profiles use `run_role: baseline`. They are divided into three
non-equivalent task families; target-schema knowledge is profile-specific.

### Native Wikidata-like TE

REBEL and mREBEL use:

```json
{
  "run_role": "baseline",
  "native_schema": {
    "id": "wikidata_like",
    "inherited_from_model": true
  },
  "target_schema_knowledge": "none"
}
```

mREBEL records Spanish as supported and its `es_XX`/`tp_XX` model formatting.
REBEL records `input_language: es`,
`language_status: out_of_primary_model_scope`, and no language tokens. This
status is deliberately conservative rather than an invented quantitative
language-support claim.

### Fixed-domain ontology KBP

Pythia/SPACE-KBP uses:

```json
{
  "run_role": "baseline",
  "model_family": "pythia_spacekbp",
  "task_class": "KBP_DOMAIN_BASELINE",
  "target_schema_knowledge": "none",
  "controlled_experiment_condition": "not_applicable",
  "native_schema": {
    "id": "space_kbp_space_ontology",
    "inherited_from_model": true,
    "scope": "fixed_domain_ontology"
  }
}
```

The Pythia model metadata also records the pinned model/tokenizer revision,
base model, clean-room prompt profile, model/effective input limits,
quantization mode, prefix context, and whether remote download is authorized.

### Dynamic structural-schema UIE

UIE uses:

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

UIE model metadata records the pinned checkpoint/tokenizer, ordered spot and
association labels, optional spot-to-association metadata, schema/SSI
versions, schema hash, exact serialized SSI, generation/truncation settings,
constraint status, and conservative parser policy. It also records
`interface_accepts_custom_schema: true` separately from
`zero_shot_unseen_schema_supported: unknown`.

### Dynamic guideline-schema GoLLIE

GoLLIE uses:

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

GoLLIE model metadata records the pinned merged full-model checkpoint,
Code Llama base, Llama 2 weight terms, custom-modeling requirement,
FlashAttention/CUDA requirement, prompt serializer version, schema hash,
guideline hash, exact serialized prompt, Black policy/version, quantization
`none`, and `scientific_role: GUIDELINE_FOLLOWING_UIE_BASELINE`. English
native-schema runs use `language_status: supported`. GoLLIE is not Controlled
C2.

### Closed-schema constrained GenIE

GenIE uses:

```json
{
  "run_role": "baseline",
  "model_family": "genie",
  "task_class": "KBP_CLOSED_IE",
  "target_schema_knowledge": "fixed_native_schema+kb_constraints",
  "controlled_experiment_condition": "not_applicable",
  "input_language": "en",
  "language_status": "supported",
  "native_schema": {
    "inherited_from_model": true,
    "scope": "fixed_native_schema+kb_constraints",
    "constraint_profile": "unconstrained",
    "entity_inventory_required": false,
    "relation_inventory_required": false,
    "formal_ontology": false
  }
}
```

Constrained profiles set `entity_inventory_required` and
`relation_inventory_required` to true. Spanish input is
`out_of_documented_training_scope`. GenIE is not C3.

The backward-compatible `condition: C0`, `conditioning: none`, and
`target_condition.id: C0` fields remain in the shared contract. For Pythia,
UIE, GoLLIE, and GenIE, they do **not** imply membership in the causal C0-C3
experiment. `controlled_experiment_condition: not_applicable` and each
`task_class` are the authoritative classification fields.
