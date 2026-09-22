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

All external profiles use `run_role: baseline` and
`target_schema_knowledge: none`. They are divided into two non-equivalent task
families.

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

The backward-compatible `condition: C0`, `conditioning: none`, and
`target_condition.id: C0` fields remain in the shared contract. For Pythia,
they do **not** imply membership in the causal C0-C3 experiment.
`controlled_experiment_condition: not_applicable` and
`task_class: KBP_DOMAIN_BASELINE` are the authoritative classification fields.
