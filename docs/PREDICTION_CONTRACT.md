# Prediction contract 1.0

`predictions.jsonl` has one record per source document:

```json
{
  "run_id": "run-...",
  "example_id": "articulo_8",
  "document_id": "articulo_8",
  "segment_id": null,
  "condition": "C0",
  "input": {"text": "..."},
  "raw": {"model_output": "...", "segments": []},
  "parsed": [],
  "normalized": [],
  "validated": {"triples": [], "violations": [], "status": "ok"},
  "prediction_contract_version": "1.0"
}
```

For windowed documents, `raw.segments[]` stores segment id, exact text,
document-relative boundaries, exact decoded output, and generation metadata.
The top-level raw output is a deterministic concatenation of those exact
outputs; it is never reconstructed from triples.

Each triple contains `subject`, `relation`, `object`, optional pretrained entity
types, optional spans, `span_source`, and optional internal `segment_id`.
RDF-producing adapters may additionally preserve `subject_is_uri`,
`predicate_is_uri`, `object_is_uri`, `object_is_literal`, `datatype`, and
`language`; absent RDF metadata fields are omitted.
Parsed, normalized, and validated arrays are separate and duplicates retain
their occurrence order.

Violations have `code`, `severity` (`soft` or `hard`), message, and optional
triple index. Status is `ok`, `soft_fail`, or `hard_fail`.

The deterministic evaluator export is a non-destructive view with
`id`, `prediction_id`, `document_id`, optional `segment_id`, `subject`,
`relation`, and `object`.
