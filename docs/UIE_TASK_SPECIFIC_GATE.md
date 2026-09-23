# UIE task-specific checkpoint gate

Audit date: 2026-09-22
Iteration: `UIE_TASK_SPECIFIC_CHECKPOINT_GATE`
Decision: `B. OFFICIAL_RE_CHECKPOINT_NOT_AVAILABLE_BUT_REPRODUCIBLE_FINE_TUNING`

This audit changes no runtime, downloads no model, and performs no training.

## 1. Critical scientific distinction

```text
uie-base-en
= pretrained/base UIE model

task-fine-tuned UIE
= actual downstream extraction baseline
```

The UIE paper describes two distinct stages:

1. UIE pretraining learns general text-to-structure, structure-generation, and
   text-representation abilities.
2. On-demand supervised fine-tuning adapts that base to a downstream IE task
   and dataset.

Therefore, the empty native smoke with `uie-base-en`:

- does not invalidate UIE;
- does not invalidate the clean-room adapter;
- does not prove general zero-shot incapability;
- does show that the released base checkpoint is not a drop-in
  task-fine-tuned relation extractor for the tested schema/example.

## 2. Official checkpoint inventory

Official author-attributable English releases are limited to:

| Checkpoint | Role | Revision | Size | License |
|---|---|---|---:|---|
| `luyaojie/uie-base-en` | General English UIE pretrained base | `966f8b1fc4c74e94ab552081605913ad5133cc41` | 990,285,657-byte FP32 weight; ~992.5 MB repository | CC BY-NC-SA 4.0 |
| `luyaojie/uie-large-en` | General English UIE pretrained large | `458a7066e8d6217b25945fe71f04f375910e8487` | 3,132,653,069-byte FP32 weight; ~3.135 GB repository | CC BY-NC-SA 4.0 |

Both are complete `T5ForConditionalGeneration` checkpoints with tokenizer and
config, not downstream adapters.

No official or author-attributable task-specific English UIE checkpoint was
found for:

- ACE05-Rel;
- CoNLL04;
- NYT;
- SciERC.

Evidence:

- the official README/project page publishes only `uie-en-base`,
  `uie-en-large`, and a Chinese small base;
- the author's Hugging Face account contains only the two English base models;
- CAS Cloud and Google Drive links mirror those base models;
- the GitHub repository has no releases or tags;
- no supplementary task weights are linked by the ACL paper/artifact page;
- GitHub issue 67, “Task Specific Models,” explicitly asks for the Table 2
  fine-tuned weights and remains open with no comments/author response:
  https://github.com/universal-ie/UIE/issues/67

The names in `scripts/inference_all.bash`:

```text
rel_ace05-rel_66.22
rel_conll04_large_74.97
rel_nyt_93.53
rel_scierc_large_37.05
rel_nyt_base_92.46
```

are expected local directories under `uie_models/`. They demonstrate that the
authors used task-specific output artifacts, but are not download links,
immutable revisions, or released checkpoints.

Third-party reuploads and later non-UIE systems were excluded from canonical
checkpoint consideration.

## 3. Official training-to-inference flow

The frozen repository implements:

```text
raw downstream dataset
-> dataset-specific preprocessing
-> UIE Spot-Asoc conversion
-> data/text2spotasoc/relation/{dataset}
-> base/pretrained checkpoint
-> run_uie_finetune.py
-> selected output_dir checkpoint
-> inference.py --model output_dir
-> raw SEL
-> sel2record
-> offset mapping
-> relation evaluation
```

Exact stages:

1. Prepare raw data according to `dataset_processing/README.md`.
2. Convert dataset-specific input into the unified format with
   `dataset_processing/uie_convert.py`.
3. Link `dataset_processing/converted_data` as repository-root `data`.
4. Place a base/pretrained model under `hf_models/`.
5. Run `run_uie_finetune.bash` directly or through an experiment-grid script.
6. Select the best checkpoint by `eval_overall-F1`.
7. `trainer.save_model()` writes a full Transformers model and tokenizer to
   `output_dir`.
8. Run `scripts/sel2record.py` and `scripts/eval_extraction.py`.
9. Pass that task-specific `output_dir` to `inference.py --model`.

The standalone `inference.py` defaults to a task-specific ABSA model path:

```text
./models/uie_n10_21_50w_absa_14lap
```

It does not default to bare `uie-base-en`.

## 4. Downstream checkpoint production

`run_uie_finetune.py` loads `model_name_or_path` with
`AutoModelForSeq2SeqLM`, trains with teacher forcing, then calls:

```text
trainer.save_model()
trainer.state.save_to_json(...)
```

The wrapper configures:

```text
metric_for_best_model = eval_overall-F1
load_best_model_at_end = true
save_total_limit = 1
```

Thus the output root is a complete, directly loadable Transformers checkpoint,
not a LoRA/adapter. It is architecture-compatible with the clean-room
`UIEExtractor`. Before use in `re_te_system`, the resulting directory must be
frozen with a content hash or published immutable revision.

Expected final base-model artifact: approximately 1 GB plus tokenizer/config
and run metadata. Training checkpoints with optimizer state can require several
GB transiently; the legacy wrappers delete optimizer files after runs.

## 5. Relation extraction datasets and conversion

| Dataset | ID/path | Conversion config | Upstream raw/preprocessing |
|---|---|---|---|
| ACE05-Rel | `relation/ace05-rel` | `dataset_processing/data_config/relation/ace05-rel.yaml` | Sincere -> `data/relation/ace05/*.jsonlines` |
| CoNLL04 | `relation/conll04` | `dataset_processing/data_config/relation/conll04.yaml` | Sincere -> `data/relation/conll04/*.jsonlines` |
| NYT-multi | `relation/NYT` | `dataset_processing/data_config/relation/NYT-multi.yaml` | JointER `train/dev/test.json` via direct wget |
| SciERC | `relation/scierc` | `dataset_processing/data_config/relation/scierc.yaml` | DyGIE++ -> `scierc_processing.py` |

Local repository state:

- all YAML conversion configs are present;
- all conversion/training/evaluation code is present;
- raw RE data are absent;
- converted RE data are absent;
- task-specific output checkpoints are absent.

ACE05-Rel and CoNLL04 require Sincere preprocessing. SciERC requires DyGIE++.
NYT has the simplest download path, but it is not the recommended UIE route
for the reasons in section 8.

## 6. Full supervised RE configurations

Common full-data settings:

```text
task: meta
decoding: spotasoc
source prefix: meta task with dynamic SSI collator
max target length: 192
meta_negative: -1 (all available negative schema labels)
meta_positive_rate: 1
random prompt in training: yes
full ordered schema in evaluation: yes
constraint decoding: disabled
LR scheduler: linear
checkpoint selection: best eval_overall-F1
offset map: closest, de_duplicate=true
runs/seeds: 3
```

| Dataset | Paper rerun model | Config | Epochs | GPUs | Batch/GPU | Learning rate | Noise | Max source | Record evaluation |
|---|---|---|---:|---:|---:|---|---|---:|---|
| ACE05-Rel | `uie-large-en` | `large_ace05rel_conf.ini` | 50 | 4 | 8 | `1e-4`, `3e-4` grid | 0.2 | 384 | normal |
| CoNLL04 | `uie-large-en` | `large_conll04_conf.ini` | 50 | 4 | 8 | `3e-4` | 0.2/0.1 grid | 384 | normal |
| SciERC | `uie-large-en` | `large_scierc_conf.ini` | 50 | 4 | 8 | `3e-4` | 0.2/0.1/0 grid | 384 | normal |
| NYT | `t5-v1_1-large` | `large_model_conf_nyt.ini` | 50 | 4 | 8 | `5e-5` | 0.2 | 384 | set |
| NYT | `t5-v1_1-base` | `base_model_conf_nyt.ini` | 50 | 4 | 16 | `3e-4` | 0.1 | 384 | set |

Warmup ratio is 0.06 and label smoothing is 0 for all listed configs.

Important: the paper does not report a UIE result on NYT because NYT overlaps
the UIE pretraining data. Table 2 reports the non-UIE SEL/T5 baseline score and
marks UIE as not run. The repository's NYT scripts are reproducibility
experiments with T5-v1.1, not a clean task-specific `uie-base-en` recommendation.

## 7. SSI, rejection, noise, and decoding

Training:

- `DataCollatorForMetaSeq2Seq` builds dynamic SSI;
- `--random_prompt` makes training label order random;
- evaluation uses the complete schema in deterministic order;
- `meta_negative=-1` includes all available schema negatives;
- `meta_positive_rate=1` keeps all positive labels;
- spot/asoc noise injects null/rejection target structures;
- `max_prefix_length=-1` means no SSI truncation;
- constrained decoding is off unless explicitly requested.

Generation/evaluation:

- Spot-Asoc SEL;
- greedy/model-default beam count unless otherwise supplied;
- SEL-level best-model metric is `eval_overall-F1`;
- post-training relation metrics are computed after `sel2record`;
- strict relation scoring includes relation label, both argument spans, and
  both argument types;
- boundary scoring omits argument types;
- NYT uses set matching; the other audited RE tasks use normal matching.

## 8. Recommended minimum official fine-tuning route

Recommended dataset:

```text
CoNLL04, official 10-shot seed-1 split
```

Why:

- it is a native English end-to-end RE task used by UIE;
- it has a compact binary-relation schema;
- the repository explicitly documents `uie-base-en` low-resource CoNLL04
  experiments;
- it avoids downloading the 3.1 GB `uie-large-en`;
- it avoids NYT's overlap with UIE pretraining;
- it is substantially smaller than full-data/grid reproduction;
- one 10-shot seed is one unmodified member of the official 10-seed,
  1/5/10-shot protocol.

Official preparation:

1. Obtain/process CoNLL04 through Sincere as documented.
2. Run `sincere_processing.py`.
3. Run `uie_convert.py` to create
   `converted_data/text2spotasoc/relation/conll04`.
4. Run official `sample_data_shot.py` for relation task and seed 1.
5. Select the generated `seed1/10shot` directory.

Official 10-shot configuration:

```text
base checkpoint: uie-base-en
task: meta
decoding: spotasoc
GPU count: 1
epochs: 200
batch size: 16
learning rate: 1e-4
LR scheduler: constant
warmup ratio: 0.06
label smoothing: 0
meta_negative: -1
meta_positive_rate: 1
spot noise: 0.1
association noise: 0.1
random prompt: true
max source length: 256
max target length: 192
constraint decoding: false
offset map: closest
best metric: eval_overall-F1
```

The full `run_exp_shot.bash` performs 10 seeds and all 1/5/10-shot variants.
For `UIE_REAL_NATIVE_POSITIVE_SMOKE_P1`, train only the official
`seed1/10shot` member using the exact command assembled by
`run_uie_finetune_shot.bash`; do not report it as the paper's averaged
low-resource result.

Hardware:

```text
CPU: technically possible but operationally impractical
GPU: effectively required for faithful reproduction
original script: one CUDA GPU for base low-resource runs
```

Expected final artifact:

```text
full UIE-base checkpoint, approximately 1 GB
several GB transient disk while optimizer/checkpoint state exists
```

No training is performed in this gate.

## 9. Why other routes were not selected

### Full CoNLL04 / ACE05-Rel / SciERC

Paper-faithful, but uses `uie-large-en`, four GPUs, 50 epochs, three runs, and
hyperparameter/noise grids. This is not the minimum path.

### NYT

Simplest raw-data acquisition, but scientifically unsuitable as the first UIE
task-specific gate:

- the paper excludes UIE due to overlap with pretraining data;
- official configs use T5-v1.1-base/large rather than UIE-base/large;
- a positive output would not validate the intended UIE-pretrained downstream
  path cleanly.

### ACE05-Rel

Its schema matches the MULTAN/`part whole` example, but data access and Sincere
preprocessing are heavier, and the paper rerun uses UIE-large with a four-GPU
grid. It is a later paper-reproduction candidate, not the minimum P1 gate.

## 10. Released checkpoint gate

| Candidate task | Official checkpoint | Immutable revision | License | Downloadable | Directly loadable |
|---|---|---|---|---|---|
| ACE05-Rel | Not found | N/A | N/A | No | N/A |
| CoNLL04 | Not found | N/A | N/A | No | N/A |
| NYT | Not found | N/A | N/A | No | N/A |
| SciERC | Not found | N/A | N/A | No | N/A |

`OFFICIAL_RE_CHECKPOINT_AVAILABLE` is false.

`OFFICIAL_RE_CHECKPOINT_NOT_AVAILABLE_BUT_REPRODUCIBLE_FINE_TUNING` is true.

## 11. Decision and next action

Decision:

```text
B. OFFICIAL_RE_CHECKPOINT_NOT_AVAILABLE_BUT_REPRODUCIBLE_FINE_TUNING
```

Next action:

1. Stage CoNLL04 through the documented Sincere/UIE converters.
2. Freeze the converted-data hashes and license/provenance.
3. Generate the official relation 10-shot seed-1 split.
4. On one CUDA GPU, fine-tune pinned `uie-base-en` with the exact official
   low-resource configuration above.
5. Freeze the resulting full checkpoint with file hashes, environment, and
   training manifest.
6. Run a native positive CoNLL04 smoke through the clean-room adapter.
7. Do not use Hohfeld until the native task-specific path succeeds.

`UIE_TASK_SPECIFIC_GATE_READY = YES`
