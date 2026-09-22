# UIE CoNLL04 10-shot seed-1 reproduction P0

Preparation date: 2026-09-22
Iteration: `UIE_CONLL04_10SHOT_REPRODUCTION_P0`
Status: `DETERMINISTIC_DATA_FROZEN`

No training, Hohfeld work, prompt search, manual data correction, or model
weight commit was performed.

## 1. Frozen upstream

UIE:

```text
repository: universal-ie/UIE
revision: c88dd08b8ad0016ec597ebc6f8fff49480367bb2
```

Base checkpoint:

```text
model: luyaojie/uie-base-en
revision: 966f8b1fc4c74e94ab552081605913ad5133cc41
license metadata: CC BY-NC-SA 4.0
```

Sincere:

```text
repository: btaille/sincere
revision: dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31
dataset introduced at: c8a5972845ca3ae969934a245a3c4f737b8e2cb6
license: Apache-2.0
```

All identities and converter hashes are frozen in:

```text
artifacts/uie_conll04_10shot_seed1_manifest.json
```

## Workspace layout and ownership

Canonical repository ownership:

```text
Desarrollo/_legacy/UIE
  UPSTREAM_READ_ONLY
  universal-ie/UIE@c88dd08b8ad0016ec597ebc6f8fff49480367bb2

Desarrollo/_legacy/sincere
  UPSTREAM_DATA_SOURCE_READ_ONLY
  btaille/sincere@dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31

Desarrollo/re_te_system
  PAPER3_IMPLEMENTATION
  clean-room runtime plus repository-level reproduction tooling
```

No nested Sincere Git repository remains under UIE.

Previous clone used for the first conversion:

```text
Desarrollo/_legacy/UIE/dataset_processing/thirdparty/sincere
```

Its origin and HEAD were verified before canonicalization, and both source
copies produced the required SHA-256 before the nested clone was removed.

UIE consumes the canonical source through a relative symlink:

```text
_legacy/UIE/dataset_processing/data/sincere/conll04.json
  -> ../../../../sincere/data/conll04.json
```

Generated/ignored UIE locations:

```text
_legacy/UIE/dataset_processing/data/relation/conll04/
_legacy/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04/
_legacy/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04_shot/
_legacy/UIE/hf_models/
_legacy/UIE/output/
_legacy/UIE/runs/
```

Paper 3 commits only identities, hashes, manifests, validators, and
documentation. Future orchestration belongs under:

```text
re_te_system/scripts/reproductions/uie_conll04/
```

This directory may orchestrate frozen upstream training but is not imported by
the `re_te_system` runtime package.

### Git publication boundary

Verified ignore coverage:

| Material | Location | Ignore rule |
|---|---|---|
| UIE staged raw CoNLL04 | `_legacy/UIE/dataset_processing/data/` | `dataset_processing/.gitignore: /data` |
| Converted full data | `_legacy/UIE/dataset_processing/converted_data/` | `dataset_processing/.gitignore: /converted_data` |
| Shot splits | under the same `converted_data/` tree | same rule |
| UIE base/trained weights | `_legacy/UIE/hf_models/`, `output/`, `models/` | root UIE `.gitignore` |
| Local UIE runs | `_legacy/UIE/runs/` | root UIE `.gitignore` |
| Main-system HF caches | `re_te_system/.cache/` | `re_te_system/.gitignore` |
| Main-system run outputs | `re_te_system/runs/` | `re_te_system/.gitignore` |

Future CESVIMA logs/checkpoints must be written to external scratch or the
ignored upstream `output/` tree, not to tracked `re_te_system/artifacts/`.

Allowed committed Paper 3 material:

- source identities and revisions;
- checksums and manifests;
- validation reports;
- reproduction scripts;
- documentation.

`CONLL04_PUBLIC_REDISTRIBUTION = UNRESOLVED`

## 2. CoNLL04 source provenance

The exact UIE route delegates CoNLL04 to Sincere. Sincere publishes
`data/conll04.json`, described as:

- formatted as in Eberts and Ulges/SpERT;
- using the Gupta et al. 2016 train/dev/test split;
- originating from the CoNLL04 relation-extraction resource.

No Hugging Face substitute was used.

Source:

```text
https://github.com/btaille/sincere
data/conll04.json
size: 750,139 bytes
git blob: 87918d38a5ac704de9a6df2b4cce9f4a4f3bc495
SHA-256: 3abe11eb5750e2c37e99a428f5e6c7e3b15f77d66053bca85e21e0772d601119
```

Sincere does not provide a command that generates this JSON; it ships the file
pre-generated and unchanged since its initial commit. The file is byte-for-byte
the object formed from the contemporaneous SpERT train/dev/test JSON splits.
The manifest records those three source hashes.

Further lineage:

```text
Roth and Yih CoNLL04 resource
-> Gupta 2016 split
-> Eberts/Ulges SpERT JSON conversion
-> Sincere combined JSON
-> UIE inclusive-span intermediate
-> UIE Spot-Asoc conversion
```

Sincere's code license is Apache-2.0. The original CoNLL04 data's
redistribution/use terms are not established by the Sincere code license and
must be confirmed before publishing or copying converted data outside the
private research workspace. The corpus embeds TREC/TIPSTER newswire, including
material normally distributed under LDC agreements; public availability alone
is not an explicit redistribution license.

## 3. Exact preprocessing order

### 3.1 Pinned source placement

The Sincere repository is frozen at the canonical sibling location
`_legacy/sincere`. Its `data/conll04.json` is linked into:

```text
_legacy/UIE/dataset_processing/data/sincere/conll04.json
```

### 3.2 Sincere-to-UIE intermediate

Upstream converter:

```text
_legacy/UIE/dataset_processing/scripts/sincere_processing.py
SHA-256: f1f6166e402b2dc3b738b1d8972b13f484a90a7f024887537feb83eec28c49a2
```

The script's entrypoint loops over ACE05 and CoNLL04 and fails when licensed
ACE05 data are absent. To avoid fabricating ACE05, the exact upstream
`processing_sentence` function was invoked only for the pinned CoNLL04 object.
No conversion logic or data value was changed.

Intermediate output:

```text
dataset_processing/data/relation/conll04/train.jsonlines  922 rows
dataset_processing/data/relation/conll04/dev.jsonlines    231 rows
dataset_processing/data/relation/conll04/test.jsonlines   288 rows
```

Transformation:

- copy `tokens`;
- copy relation list to `span_pair_list`;
- copy entity list to `span_list`;
- decrement each entity's exclusive `end` to UIE's inclusive end.

### 3.3 UIE Spot-Asoc conversion

Files:

```text
uie_convert.py
SHA-256: 767a016942abe47c7ec6bb5fc669d869d9496ebdf4f512fea66cb15da36ea151

data_config/relation/conll04.yaml
SHA-256: 14c9a45958c58d7b904747da58c178abdfa080bdd8b951ce8d7927306c963057
```

Command:

```bash
cd _legacy/UIE/dataset_processing
conda run -n uie_legacy_38 \
  python uie_convert.py \
  -format spotasoc \
  -config data_config/relation/conll04.yaml \
  -output relation
```

Final full-data directory:

```text
dataset_processing/converted_data/text2spotasoc/relation/conll04/
```

Full converted hashes:

```text
train.json  d59e10133d2272e6daf4837ffe7236aa1f4f708af44142f4c48536e211ffdcc6
val.json    a9be8c5c3716456fb3caffa13f069feea6272a07a4fadce47265db07b30debb8
test.json   91d6b666268bacf73ea9352f7111c2ef84a212374148bb357933555707811feb
```

## 4. Official 10-shot seed-1 definition

Upstream sampler:

```text
dataset_processing/scripts/sample_data_shot.py
SHA-256: 4559990f27202d344235ec598a8b4eaca14528466ec6924ea73904ddea56c473
```

Exact official definition for relation tasks:

1. Load full `train.json`.
2. Call `random.seed(seed)` and shuffle the examples.
3. Index training examples by each label in their `asoc` list.
4. For each relation label independently, sample 10 sentence indices, or all
   if fewer than 10 exist.
5. Append selected examples in relation-label iteration order.
6. Do not deduplicate examples selected for multiple labels.
7. Copy validation, test, and schema files unchanged.

The CLI does not declare an integer type for `-seed`; therefore the exact
official invocation seeds Python `random` with string `"1"`, not integer `1`.
This detail is frozen in the manifest.

Command:

```bash
cd _legacy/UIE/dataset_processing
PYTHONPATH=./ conda run -n uie_legacy_38 \
  python scripts/sample_data_shot.py \
  -src converted_data/text2spotasoc/relation/conll04 \
  -tgt converted_data/text2spotasoc/relation/conll04_shot/seed1 \
  -task relation \
  -seed 1
```

Selected split:

```text
converted_data/text2spotasoc/relation/conll04_shot/seed1/10shot/
```

Counts:

```text
train: 50
validation: 231
test: 288
```

Training examples contain two repeated text occurrences because the same
sentence can be independently selected for multiple relation labels. This is
the upstream sampling policy and was not repaired.

Train split SHA-256:

```text
5fecc7091bf7152d04a4eb43641933600f24f3af11995bc3e8f8d268fd43da7b
```

## 5. Schema

Entity labels:

```text
other
organization
location
people
```

Relation labels:

```text
organization in
work for
located in
live in
kill
```

Spot-to-association map:

```text
location     -> located in
organization -> organization in
other        -> []
people       -> work for, kill, live in
```

Schema hashes:

```text
record.schema   efcac0112405e307aec133d4a289651ed48f93477b004b3fb2e29d1b34913e3d
entity.schema   3aca8841a0525d36de14b9884753a76259d3adad5cf0c84e72988ce83ccff967
relation.schema 64eb41cc308717e5851984ad2ee124b4c68085bb9dfc3b72b9348958b1b34d62
event.schema    3470a9d51834a60a8b709b30d2e205a879d9d28d1a0515068eafe838ffcaeda8
```

## 6. Mechanical validation

Validator:

```text
scripts/reproductions/uie_conll04/validate_reproduction.py
```

Report:

```text
artifacts/uie_conll04_10shot_seed1_validation.json
SHA-256: efc6a7c106c8660177c16c4c61aec20a4dc678f0f2d61b89753074b54e8f21f8
combined validation hash:
8e813869fc6c0546b321e1dc8625f19ac17bced8e644b36ea117389f22542c91
```

Passed:

- every JSON line parses;
- all required fields exist;
- every target SEL parses with no upstream bracket repair or malformed
  fallback;
- all four schemas parse;
- no duplicate IDs exist because converted records contain no IDs;
- validation/test/schema files in the shot folder match the full conversion;
- source/data/converter hashes are frozen.

Recorded warnings, without correction:

```text
duplicate texts within full train: 24
duplicate texts within full validation: 3
duplicate texts within full test: 1
duplicate texts within shot train: 2

full train/validation text overlap: 12
full train/test text overlap: 18
full validation/test text overlap: 3
shot train/validation text overlap: 0
shot train/test text overlap: 1
```

These overlaps originate in the pinned upstream data/split. Removing or
reassigning them would violate the no-manual-correction rule. They must be
reported in any scientific result.

## 7. Canonical training configuration

Strict upstream fidelity is:

```text
base: pinned uie-base-en
epochs: 200
batch size: 16
learning rate: 1e-4
scheduler: constant
warmup ratio: 0.06
spot noise: 0.1
association noise: 0.1
negative schema: all (-1)
random training prompt: enabled
evaluation schema: full ordered
max source length: 256
max target length: 192
constraints: disabled
split sampling seed argument: string `"1"`
training seed: 42
best metric: eval_overall-F1
```

`sample_data_shot.py` receives CLI seed `"1"`. The shot wrapper does not
propagate that split index to the trainer; `function_code.bash` defines
`seed=42`, and `run_uie_finetune_shot.bash` passes `--seed=42`.

Any future trainer-seed-1 run is a separate ablation, not the canonical
reproduction.

## 8. Deterministic data freeze

Exploratory conversions produced different byte hashes because:

- `Text2SpotAsoc.annonote_graph` builds spot/asoc labels with Python sets;
- `uie_convert.py` serializes `list(spot_labels)` and `list(asoc_labels)`;
- schema generation also converts `set` values to lists;
- upstream does not freeze `PYTHONHASHSEED`.

Those exploratory artifacts are noncanonical. Their source texts, counts,
labels, and relations were semantically unchanged, but their list/schema order
depended on the process hash seed. No historical hash was selected or searched
for post hoc.

The canonical policy was fixed prospectively:

```text
PYTHONHASHSEED=0
python_hash_seed_policy=reproducibility_hardening
upstream_hash_seed_controlled=false
```

This value is not a training seed and not the data split seed. Upstream source
code remains unchanged; no sorting or normalization was added after generation.

The entire intermediate, converted, schema, and shot tree was deleted and
regenerated twice from the canonical Sincere source. Both inventories and both
validation reports are byte-identical:

```text
pass1/pass2 inventory file SHA-256:
e7e9b56b16e95afacf2686fdc4c132320355cf1798013a380de19be2fc0a845b

generated tree combined SHA-256:
5bc9929f808ac9acee82a3e71acf21302d3e55e43f774680d2ed5c232a88df75

pass1/pass2 validation file SHA-256:
efc6a7c106c8660177c16c4c61aec20a4dc678f0f2d61b89753074b54e8f21f8

validation report combined hash:
8e813869fc6c0546b321e1dc8625f19ac17bced8e644b36ea117389f22542c91
```

Canonical intermediate hashes:

```text
train.jsonlines e4ee4a3e67a54795f8d64ce247ad8f0c9553ade088239a3b6c2e75aff2be28e5
dev.jsonlines   74b016b284df794bddb8b5fc43ffcbc3abd8a9a8f24d4dd7fea8a3f64ac3aa95
test.jsonlines  d208a4c770f6c4c4bcdcbc778e8edd0b642649ed66ff7ab72410ae8cb77899bc
```

Canonical full converted hashes:

```text
train.json d59e10133d2272e6daf4837ffe7236aa1f4f708af44142f4c48536e211ffdcc6
val.json   a9be8c5c3716456fb3caffa13f069feea6272a07a4fadce47265db07b30debb8
test.json  91d6b666268bacf73ea9352f7111c2ef84a212374148bb357933555707811feb
```

Canonical schema hashes:

```text
record.schema   efcac0112405e307aec133d4a289651ed48f93477b004b3fb2e29d1b34913e3d
entity.schema   3aca8841a0525d36de14b9884753a76259d3adad5cf0c84e72988ce83ccff967
relation.schema 64eb41cc308717e5851984ad2ee124b4c68085bb9dfc3b72b9348958b1b34d62
event.schema    3470a9d51834a60a8b709b30d2e205a879d9d28d1a0515068eafe838ffcaeda8
```

Canonical 10-shot hashes:

```text
train.json 5fecc7091bf7152d04a4eb43641933600f24f3af11995bc3e8f8d268fd43da7b
val.json   a9be8c5c3716456fb3caffa13f069feea6272a07a4fadce47265db07b30debb8
test.json  91d6b666268bacf73ea9352f7111c2ef84a212374148bb357933555707811feb
```

Repeatability status: `PASS`.

## 9. Training environment and resource plan

Original environment:

```text
Python 3.8
PyTorch 1.8.0
Transformers 4.6.1
CUDA 10.2/11.1
```

A modern CESVIMA environment may be acceptable only after a GPU preflight
confirms:

- imports;
- visible CUDA device/model;
- tokenizer load;
- pinned base checkpoint load;
- one forward/backward batch;
- unchanged SSI/noise/collator/generation behavior.

No modern version set is frozen yet because that validation must occur on the
target GPU node.

Qualitative requirements:

```text
GPU: one CUDA GPU, effectively required for practical reproduction
CPU: operationally impractical for 200-epoch seq2seq training
CPU cores: 8 planned
RAM: 32 GB planned
job wall time: to determine with CESVIMA preflight
```

Storage estimate:

```text
base checkpoint: ~1 GB
final full checkpoint: ~1 GB
temporary trainer/optimizer/checkpoint state: several GB
recommended workspace allowance: at least 10 GB, excluding Conda cache
```

No cluster partition, account, private path, GPU model, or wall-time request is
invented before CESVIMA environment discovery.

## 10. Checkpoint policy

Expected logical identity:

```text
uie-base-en-conll04-10shot-seed1
```

The original `output_dir` must remain intact internally. After training, record:

- directory path;
- every artifact name and size;
- SHA-256 of all weight files;
- config hash;
- tokenizer hashes;
- training arguments;
- trainer state;
- best `eval_overall-F1`;
- selected epoch/step;
- environment and SLURM metadata.

The expected output is a complete Transformers checkpoint, not an adapter.
Weights remain ignored and must not be committed.

## 11. Clean-room boundary and later positive smoke

Training implementation:

```text
frozen upstream universal-ie/UIE
```

Paper 3 inference contract:

```text
re_te_system clean-room UIE adapter/parser
```

Upstream parser repairs, fuzzy alignment, and deduplication do not migrate into
`re_te_system`.

After the configuration discrepancy is resolved and training succeeds:

1. select a positive CoNLL04 **test** example not present in shot training;
2. load the frozen best checkpoint through the clean-room adapter;
3. supply only text and native schema/SSI, never target SEL;
4. retain exact RAW, structures, projected triples, violations, and manifest;
5. require a non-empty valid Spot-Association structure;
6. report expected relation recovery separately.

No Hohfeld work occurs before that gate passes.

## 12. Current readiness

```text
converted dataset ready: YES
10-shot seed-1 split ready: YES
mechanical validation: PASS WITH UPSTREAM LEAKAGE WARNINGS
training config exact match: YES (split seed argument "1", trainer seed 42)
deterministic repeatability: PASS (PYTHONHASHSEED=0, two full regenerations)
CESVIMA wrapper: NOT CREATED — outside this iteration
GPU environment: NOT PREFLIGHTED
training manifest: CREATED, status DETERMINISTIC_DATA_FROZEN
weights committed: NO
ready to submit job: NO
```

`UIE_CONLL04_10SHOT_REPRODUCTION_P0_READY = YES`

`UIE_CONLL04_DETERMINISTIC_FREEZE_P0_READY = YES`
