# Run manifest 1.0

The manifest records:

- content-derived `run_id`;
- dataset id, version, and documents SHA-256;
- benchmark documents and manifest identities;
- model/checkpoint, requested and resolved revisions, tokenizer, Spanish source
  language, `tp_XX` target token, device, and dtype;
- `condition: C0` and `conditioning: none`;
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
