# UIE real native CPU smoke P0

Execution date: 2026-09-22
Iteration: `UIE_REAL_NATIVE_SMOKE_P0`
Result: `FAIL` under the required non-empty Spot-Association criterion.

The adapter/runtime contract succeeded, but the pinned base checkpoint emitted
an empty SEL structure. No prompt, schema, order, decoding parameter, or
constraint was changed after observing the result.

## Fixed protocol

Checkpoint:

```text
luyaojie/uie-base-en
revision: 966f8b1fc4c74e94ab552081605913ad5133cc41
```

Runtime:

```text
Python 3.12.14
PyTorch 2.14.0
Transformers 4.57.6
device: cpu
CUDA: false
MPS: false
wall-clock command: 7.86 seconds
```

The outer `/usr/bin/time -l` wrapper returned status 1 after extraction because
the sandbox denied its final `sysctl kern.clockrate` query. The inner
`re-te-system` command completed, printed the run ID, and wrote all expected
artifacts with zero recorded model failures.

Exact text:

```text
MULTAN , Pakistan , April 27 ( AFP )
```

Exact schema:

```json
{
  "schema_id": "uie-native-relation-multan",
  "schema_version": "uie-structural-schema-v1",
  "spot_labels": [
    "geographical social political",
    "organization"
  ],
  "association_labels": [
    "part whole"
  ],
  "spot_to_association": {
    "geographical social political": ["part whole"],
    "organization": []
  }
}
```

Exact SSI:

```text
<spot> geographical social political<spot> organization<asoc> part whole<extra_id_2>
```

The serialized SSI ends with one ASCII space after `<extra_id_2>`; the manifest
preserves that exact string.

Generation:

```text
constraint_decoding: false
do_sample: false
num_beams: 1
max_length: 192
seed metadata: 42
```

## Artifacts

Run:

```text
runs/UIE_REAL_NATIVE_SMOKE/output/run-e0621c9643771e651eef/
```

Files:

```text
manifest.json
predictions.jsonl
statistics.json
failures.jsonl
```

Input and schema:

```text
runs/UIE_REAL_NATIVE_SMOKE/input_documents.jsonl
runs/UIE_REAL_NATIVE_SMOKE/schema.json
```

Resolved revision in the manifest:

```text
966f8b1fc4c74e94ab552081605913ad5133cc41
```

## Exact result

Exact RAW SEL:

```text
<pad><extra_id_0><extra_id_1></s>
```

Parsed Spot-Association structures:

```json
[]
```

Projected binary triples:

```json
[]
```

Validation:

```json
{
  "status": "soft_fail",
  "violations": [
    {
      "code": "PARSE_EMPTY_STRUCTURE",
      "message": "SEL tree contains no spots",
      "severity": "soft",
      "triple_index": null
    }
  ]
}
```

Generation metadata:

```json
{
  "constraint_decoding": false,
  "decoded_with_special_tokens": true,
  "deterministic": true,
  "effective_input_limit": 256,
  "input_token_count": 26,
  "max_target_tokens": 192,
  "model_max_length": 512,
  "number_of_sequences": 1,
  "output_reached_limit": false,
  "output_token_count": 4,
  "truncated_input": false,
  "untruncated_input_token_count": 26
}
```

Expected optional observation:

```text
MULTAN --part whole--> Pakistan
```

Observed: **NO**.

## Pass/fail assessment

Successful:

- checkpoint loaded;
- pinned revision resolved;
- CPU generation completed;
- exact RAW persisted;
- conservative SEL parser executed;
- prediction and manifest artifacts were produced;
- no model/segment failure occurred.

Failed required criterion:

- no non-empty Spot-Association structure was generated.

Therefore:

```text
REAL_UIE_NATIVE_POSITIVE_SMOKE = FAIL
```

## Empty-output audit

### Is SSI incomplete?

No missing component was identified. The clean-room SSI matches the standalone
upstream `schema_to_ssi` contract:

- sorted/native schema order is already
  `geographical social political`, `organization`;
- association label is `part whole`;
- `<spot>`, `<asoc>`, and `<extra_id_2>` are present;
- input token count is 26, matching the legacy reproduction;
- no input truncation occurred.

### Does the release equal a task-fine-tuned relation checkpoint?

No evidence supports that interpretation. `luyaojie/uie-base-en` is the
released UIE **pretrained base checkpoint** initialized from T5-v1.1-base.
The official README presents it under “Pretrained Models” and then documents a
separate downstream fine-tuning step.

The standalone legacy `inference.py` does not default to this base checkpoint;
its default model path is a task-specific ABSA model directory:

```text
./models/uie_n10_21_50w_absa_14lap
```

This strongly explains why the base checkpoint can accept the schema/input
interface yet emit no extraction for the documented relation example.

### Does the original protocol require another path?

For downstream relation extraction, the documented workflow is:

```text
uie-base-en
-> task-specific fine-tuning
-> inference/evaluation on that task
```

The trainer prediction path can optionally enable constrained decoding, but
constraints were disabled by the fixed smoke protocol. Constraints would
restrict legal tokens/structure; they do not supply learned relation semantics
and are not a justified post-hoc remedy for this result.

### Is hardware the blocker?

No. CPU loading and generation completed quickly and without truncation or
memory failure. A GPU is not expected to turn the same deterministic weights
and input into a semantically different result.

## Scientific conclusion

The clean-room adapter is technically functional, but the audited base
checkpoint is not yet validated as a positive out-of-the-box relation
extractor. No iterative prompt/schema search was performed.

```text
UIE_ADAPTER_READY = YES
UIE_SCIENTIFIC_BASELINE_READY = PARTIAL
```
