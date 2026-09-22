# External systems integration audit P0

Audit date: 2026-09-22
Iteration: `EXTERNAL_SYSTEMS_INTEGRATION_AUDIT_P0`
Scope: external comparators for Paper 3; no extractor implementation, model
download, full experiment, benchmark/evaluation change, or legacy change.

## 1. Decision summary

Paper 3 needs three scientifically distinct external-comparator lines:

1. **RE/TE:** systems that extract relation triples from raw text. The best
   currently usable Spanish comparator is **mREBEL**; REBEL remains its
   English-centric predecessor.
2. **UIE:** systems that receive a task or schema specification and emit
   heterogeneous structures. **GoLLIE** is the strongest candidate for testing
   definitions and annotation guidelines; **UIE** is the more feasible
   structured-generation implementation target.
3. **KBP / closed IE:** systems that populate a fixed ontology or a closed KB.
   **GenIE** is the canonical closed-IE comparator. **Pythia/SPACE-KBP** is a
   domain-specialist KBP baseline, not a generic state-of-the-art TE model.

The recommended roles are:

- `IMPLEMENTED_BASELINE`: REBEL, mREBEL.
- `PRIORITY_A_IMPLEMENT`: Pythia/SPACE-KBP, UIE, GoLLIE, GenIE.
- `PRIORITY_B_IMPLEMENT_IF_FEASIBLE`: KnowPrompt, SynthIE, InstructUIE.
- `RELATED_WORK_ONLY`: DeepStruct, PTR, GenSIE 2026.
- `REJECT`: none as literature; several are rejected as *directly comparable*
  end-to-end systems.

The best comparator by line is:

- RE/TE: **mREBEL**, because it is end-to-end, implemented, and explicitly
  supports Spanish.
- UIE: **GoLLIE**, because unseen schemas, definitions, and annotation
  guidelines are first-class inference inputs.
- KBP: **GenIE**, because it is the canonical grounded closed-IE system with
  hard inventory constraints. Pythia remains the preferred domain-specialist
  KBP baseline and first adapter to implement.

These choices do not imply task equivalence. UIE and KBP results must be
reported in separate strata from end-to-end TE.

## 2. Scientific positioning and controlled-experiment boundary

External systems answer comparative and contextual questions:

- How does a native pretrained extractor behave without Hohfeld adaptation?
- What is gained by supplying labels, structure, definitions, or guidelines?
- What is gained and lost by closing the output over a KB or ontology?

They do **not** estimate the C0-to-C3 conditioning effect. The controlled
experiment remains:

```text
same controlled generative model
    C0: no target-schema knowledge
    C1: labels / minimal schema
    C2: definitions / richer schema
    C3: complete controlled conditioning
```

REBEL, mREBEL, Pythia, UIE, GoLLIE, GenIE, KnowPrompt, SynthIE, and
InstructUIE are not C0, C1, C2, or C3 conditions. The current system manifest
retains `condition: C0` for implemented historical baselines as a
backward-compatible field, but `run_role: baseline`, `model_family`,
`target_schema_knowledge`, and the Paper 3 analysis must keep those runs
outside the controlled C0-C3 contrast. This naming ambiguity is not a runtime
blocker and does not justify changing the frozen P0 contracts here.

## 3. Current `re_te_system` baseline

### 3.1 Implemented adapters

| Adapter | Checkpoint | Formulation | Spanish status | Native schema | Real smoke |
|---|---|---|---|---|---|
| mREBEL | `Babelscape/mrebel-large` | raw text to typed triples | Explicitly supported (`es_XX`) | 400 Wikidata-derived relations and entity types | Completed |
| REBEL | `Babelscape/rebel-large` | raw text to triples | Out of primary model scope | 220 frequent Wikidata-derived relations | Pending; mock contract runs exist |

Both adapters are external historical baselines. They receive no Hohfeld
labels, signatures, definitions, examples, ontology, or aliases.

### 3.2 Shared contract

- `Extractor` is a runtime-checkable protocol with `model_name`,
  `model_revision`, and
  `extract(text, ExtractionContext) -> RawExtractionResult`.
- `RawExtractionResult` contains the exact decoded `model_output` and available
  generation metadata.
- There is no Python `PredictionRecord` dataclass. The versioned
  `PredictionRecord` is the JSON object written to `predictions.jsonl`.
- The shared runner applies:

```text
TEXT -> RAW -> PARSED -> NORMALIZED -> VALIDATED
```

  - RAW preserves exact per-segment decoded output and metadata.
  - PARSED uses the injected mREBEL or REBEL state-machine parser and preserves
    order and duplicates.
  - NORMALIZED applies NFC, whitespace normalization, and canonical nulls only.
  - VALIDATED annotates structural/parser/duplicate violations without
    repairing or deleting triples.
- Document windows are internal; records aggregate back to one prediction per
  source document.
- Entity offsets are post-hoc string alignments and are marked as such.
- A content-derived run manifest records dataset identity, checkpoint and
  revision, generation settings, windowing, language status, native schema,
  target-schema knowledge, software state, and contract versions.
- `export_evaluation` creates a deterministic, non-destructive view with
  `id`, `prediction_id`, `document_id`, `segment_id`, `subject`, `relation`,
  and `object`.

The architecture should remain unchanged. A new comparator needs an
`Extractor`, a format-specific parser, manifest metadata, and tests; the
runner, stage semantics, canonical prediction artifact, and evaluation export
are reusable. The pending real REBEL smoke and unpinned default Hugging Face
revisions are operational follow-ups, not architecture blockers.

## 4. Critical task taxonomy

### 4.1 Non-equivalent task classes

| Class | Contract | What is evaluated |
|---|---|---|
| `END_TO_END_TE` | text -> subject/relation/object | Entity discovery, argument boundaries, relation recognition, and tuple assembly |
| `RELATION_CLASSIFICATION` | text + supplied entity pair -> relation | Relation choice conditional on an oracle or upstream entity pair |
| `UNIVERSAL_IE` | text + task/schema specification -> structures | Schema following plus extraction under heterogeneous structures |
| `CLOSED_IE / KBP` | text + KB/schema constraints -> grounded triples or ontology instances | Extraction, grounding, inventory adherence, and often constrained decoding |

Results across these classes must not be placed in one undifferentiated ranking.
In particular, a relation classifier with gold entities has an easier input
contract than end-to-end TE, while a closed-IE system solves additional
grounding constraints but cannot emit entities outside its inventory.

### 4.2 Task-comparability matrix

| System | Task class | Entities | Output grounding | Direct end-to-end Hohfeld comparability |
|---|---|---|---|---|
| REBEL | `END_TO_END_TE` | Extracted | Surface strings; native relations | Partial: same input/output shape, wrong native schema and language |
| mREBEL | `END_TO_END_TE` | Extracted | Surface strings plus entity types | Best direct external shape; native relation mismatch remains |
| KnowPrompt | `RELATION_CLASSIFICATION` | **Supplied pair and spans** | One fixed label | **No**; only an oracle-pair stratum |
| UIE | `UNIVERSAL_IE` | Extracted | Source spans in SEL records | Conditional after schema design and likely Hohfeld fine-tuning |
| GoLLIE | `UNIVERSAL_IE` | Extracted in general templates | Python-like typed instances | Conditional; strongest schema-guided analogue, English only |
| InstructUIE | `UNIVERSAL_IE` | Extracted in main RE task; supplied in auxiliary RC tasks | Flat generated strings | Conditional and weaker; sentence-level and labels-only |
| GenIE | `CLOSED_IE / KBP` | Extracted but forced into a predefined entity inventory | KB entity and relation identifiers through labels | No direct equivalence; separate closed-IE stratum |
| Pythia/SPACE-KBP | `KBP_DOMAIN_BASELINE` | New ontology individuals generated | Fixed space ontology; no canonical KB IDs | No direct equivalence; domain-specialist KBP stratum |
| SynthIE | `CLOSED_IE / KBP` | Inventory-constrained | Fixed Wikidata entities/relations | No direct equivalence; separate closed-IE stratum |
| DeepStruct | Mixed structure prediction | Task dependent | Generated triples plus manual schema alignment | Related-work evidence only |
| PTR | `RELATION_CLASSIFICATION` | **Supplied marked pair** | One fixed class | No |
| GenSIE 2026 | `UNIVERSAL_IE` benchmark, not one model | Schema-dependent | JSON Schema-conformant values, not KB IDs | Benchmark/design analogue, not an external model |

KnowPrompt's canonical formulation predicts the relation between two given
entities. Its official processors read subject/object spans and insert entity
markers around them. It must never be described as end-to-end TE.

## 5. System matrix

`Unknown` means that the cited primary paper, repository, or model card did not
document the capability. No undocumented capability is assumed.

| System | Architecture / scale | Released checkpoints and access | License evidence | Native language(s) / Spanish | Formulation and output |
|---|---|---|---|---|---|
| REBEL | BART-large encoder-decoder; model card 0.4B | Public `Babelscape/rebel-large` | CC BY-NC-SA 4.0 in official project/model metadata | English; Spanish unsupported | Raw text -> marker-linearized surface triples |
| mREBEL | mBART-50 large; model card 0.6B; M2M100 base also released | Public `mrebel-large`, `mrebel-large-32`, `mrebel-base` | CC BY-NC-SA 4.0 metadata; card prose has a BY-SA/BY-NC-SA inconsistency | 18 languages including Spanish | Raw text + language token -> typed triples |
| KnowPrompt | RoBERTa-large masked-LM prompt tuning, approximately 355M | Public SemEval Lightning checkpoint; other trained tasks not clearly released | MIT code; checkpoint card says Apache-2.0 | English experiments; no Spanish evidence | Given entity pair -> one fixed relation label |
| UIE | T5-v1.1 base/large encoder-decoder | Public `uie-base-en`, `uie-large-en`; Chinese small link | CC BY-NC-SA 4.0 / non-commercial in official project | English and separate Chinese; no Spanish evidence | SSI + text -> nested SEL records |
| GoLLIE | Code Llama 2 decoder-only, 7B/13B/34B | Public merged models and LoRA adapters | Apache-2.0 code/adapters; Llama 2 terms for merged/base weights | Model card: English; no Spanish evidence | Python class schema/guidelines + text -> class instances |
| InstructUIE | FLAN-T5-XXL, approximately 11B | Public `ZWK/InstructUIE`, about 90 GB float32 shards | MIT code; model card only says generic `openrail` | English experiments; no Spanish evidence | Instruction + label options + sentence -> flat IE strings |
| GenIE | BART-large-shaped encoder-decoder; exact count not paper-reported | Five public approximately 4.9 GB Lightning checkpoints plus tries/data on Zenodo | MIT code; separate checkpoint/data license unclear in inspected record | English | Text -> KB-grounded triples with structural/entity/relation tries |
| Pythia/SPACE-KBP | GPT-NeoX/Pythia decoder-only; study spans small models through 12B | Legacy IDs redirect from `anonkbp` to public `expertailab` pages for at least 1B/1.4B/2.8B; cards incomplete | Apache-2.0 upstream code/base; SPACE-KBP dataset CC BY-NC; fine-tuned weight license not declared | English | Instruction + mission text -> space-ontology Turtle |
| SynthIE | FLAN-T5 base 220M / large 770M | Five public Lightning checkpoints on `martinjosifoski/SynthIE` | MIT code/model repository | English | Text -> fixed-inventory cIE triples, FE or subject-collapsed |
| DeepStruct | GLM 10B in released execution path | Non-standard public 10B state files | Apache-2.0 | English experiments | Task prefix + text -> semicolon-delimited triples |
| PTR | RoBERTa-large, approximately 355M | Code/data scripts; no official trained PTR checkpoint verified | MIT | English | Marked entity pair + rule-composed masks -> fixed label |
| GenSIE 2026 | No single architecture/checkpoint; shared task and starter kit | Starter kit, proposal, dev data, hosted-model protocol | MIT starter kit; complete dataset-instance license unresolved | **Spanish-native task** | Instruction + arbitrary JSON Schema + Spanish text -> nested JSON |

### 5.1 Capability matrix

| System | Input/schema mechanism | Arbitrary unseen labels at inference | Definitions/guidelines | Constrained decoding | Zero/few-shot status | Hohfeld adaptation |
|---|---|---|---|---|---|---|
| REBEL | Raw text; fixed learned relation schema | No evidence | No | No formal constraint | Standalone native-schema extraction; no unseen-schema zero-shot | Supervised Hohfeld fine-tuning required |
| mREBEL | Raw text + source/target language tokens; fixed learned schema | No evidence | No | No formal constraint | Standalone native-schema extraction; no unseen-schema zero-shot | Spanish Hohfeld fine-tuning required |
| KnowPrompt | Marked subject/object + fixed `rel2id` inventory | No | Label names seed virtual words; no definitions/guidelines | Classification restricted to learned labels, not generative decoding | 8/16/32 examples per class and full supervision; not zero-shot | Pairs, labels, and training examples required |
| UIE | Structural Schema Instructor: spot and association names | Interface accepts schema names, but reliable arbitrary unseen Hohfeld labels are not established | No label definitions or guidelines | Optional grammar/prefix constraint | Supervised and 1/5/10-shot **fine-tuning**; no GoLLIE-like dynamic-schema claim | High need for Hohfeld fine-tuning |
| GoLLIE | Python classes, typed fields, docstrings, comments, examples | **Yes, documented for schemas defined on the fly** | **Definitions and annotation guidelines supported** | No; normal greedy generation | Strong documented zero-shot unseen-schema capability; in-context few-shot not principal method | English zero-shot pilot feasible; Spanish/domain tuning advisable |
| InstructUIE | Generic task instruction + candidate label options | Held-out labels evaluated, but unrestricted custom schema is not established | Generic task instruction only; no label-specific guidelines | No; options are semantic, not token constraints | Documented held-out-dataset/label zero-shot; no principal in-context few-shot | Fine-tuning strongly advisable, especially Spanish |
| GenIE | Fixed entity and relation tries | No meaningful semantic support; trie additions only make strings legal | No | **Yes: structural state machine + entity/relation tries** | Supervised; “fewer-shot” means sample efficiency, not schema zero-shot | New inventory, representation, multilingual model, and supervised training required |
| Pythia/SPACE-KBP | Fixed ontology internalized by SFT; ontology optionally present in prompt | No | Paper training prompt includes ontology description/example; legacy inference prompt includes one example | No; Turtle repaired/validated afterward | Supervised specialist; Llama 3 is the zero-shot baseline in the paper | New Spanish Hohfeld specialist training required |
| SynthIE | Fixed KB inventory learned from synthetic supervised data | No | No runtime definitions/guidelines | **Yes: GenIE-style constraints** | Extractor is supervised; zero/few-shot belongs to upstream text generation | Direct checkpoint reuse unsuitable; synthetic-data method transferable |
| DeepStruct | Task prefixes and manual schema alignment | Not established | No | No hard grammar/schema constraint | Task-transfer zero-shot and 1/5-shot reported under aligned schemas | Manual alignment and likely supervised adaptation |
| PTR | Manual logical sub-prompts and verbalizers over a fixed label set | No | Manual rules/verbalizers, not runtime guidelines | Fixed mask candidates/rule mapping | 8/16/32 per-class training; no zero-shot | Supplied pairs, manual rules, and training required |
| GenSIE 2026 | Natural instruction + arbitrary inference-time JSON Schema | **Yes; core benchmark target** | Field descriptions and instructions supported | Allowed; official baseline uses JSON-schema response constraints | Zero-shot schema; fine-tuning prohibited by task rules | Strong design analogue, but not a released comparator checkpoint |

### 5.2 Document-level and runtime matrix

| System | Document-level evidence | Dependencies / loading | Approximate hardware feasibility | Parser complexity | Target-schema mismatch |
|---|---|---|---|---|---|
| REBEL | Competitive supervised DocRED adaptation; standalone long legal documents unproven | Transformers/PyTorch; current local adapter works | Feasible on ordinary research GPU/CPU with slower CPU inference | Low-medium marker state machine | High |
| mREBEL | Wikipedia-abstract contexts; no dedicated full-document result | Transformers/PyTorch/SentencePiece; current local adapter works | Feasible on ordinary research GPU; real local smoke completed | Medium typed-token state machine | High |
| KnowPrompt | DialogRE processor exists; typical max length 256 | Old PyTorch/Transformers/Lightning stack | Moderate | Low output, high upstream pair-enumeration burden | Medium-high |
| UIE | Predominantly sentence-level; defaults source 256/target 192 | Old Python 3.8, PyTorch 1.8, Transformers 4.6 stack | Base/large practical; compatibility work likely | Medium-high SEL tree + offset mapping | Medium-high |
| GoLLIE | 16k configured context and some multi-sentence tasks; no full legal-document pipeline | CUDA, Flash Attention 2, PEFT, bitsandbytes, `trust_remote_code` | 7B in 4-bit feasible on capable CUDA GPU; CPU unsupported | Medium; official `eval()` must be replaced by safe AST parsing | Low-medium representationally; legal reasoning remains out of scope |
| InstructUIE | Input is documented as a sentence; source 512, target 50 | Old CUDA/PyTorch/DeepSpeed stack; huge float32 checkpoint | Poor on ordinary local hardware without conversion/distribution | Medium delimiter parser | High |
| GenIE | Input truncated to 256; no document coreference | Old Lightning/Hydra stack plus tries/mappings | Inference feasible but ten-beam constraints and 4.9 GB checkpoints are heavy; training used 4xV100 | Low tuple parser; high inventory infrastructure | High |
| Pythia/SPACE-KBP | Mission-description level, but 2,048-token model limit and upstream summarization/truncation | Notebook/Transformers stack; legacy adapter adds RDFLib and optional bitsandbytes | 1B/1.4B feasible; 2.8B quantized feasible; 12B expensive | Medium-high Turtle validation without heuristic mutation | Very high for Hohfeld |
| SynthIE | 256-token English inputs; no document coreference | PyTorch 1.13, Transformers 4.23, Lightning/Hydra | Base/large inference feasible; training expensive | Low-medium plus catalog assets | High |
| DeepStruct | No legal-document/DocRED evidence verified | Legacy GLM/Docker/CUDA | High: released path needs 10B and at least 32 GB GPU; preprocessing can exceed 600 GB RAM | Medium | High |
| PTR | Sentence-level RC datasets | Old PyTorch/Transformers/OpenNRE-style stack | Moderate | Low output; high prompt-rule authoring | High out of box |
| GenSIE 2026 | Text fragments, not full documents | Python/FastAPI/OpenAI-compatible hosted endpoint | Starter kit light; organizer-hosted model cost is external | Low-medium JSON + schema validation | Low for dynamic structures; no KB grounding |

## 6. Schema-knowledge taxonomy

The table classifies target knowledge available to the system. Parentheses
distinguish knowledge encoded during training from knowledge passed at
inference.

| System | NONE | FIXED_NATIVE_SCHEMA | LABELS | LABELS_PLUS_TYPES | NATURAL_LANGUAGE_DEFINITIONS | ANNOTATION_GUIDELINES | STRUCTURAL_SCHEMA | KB_ENTITY_RELATION_INVENTORY | HARD_CONSTRAINED_DECODING | FORMAL_ONTOLOGY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REBEL |  | X |  |  |  |  |  |  |  |  |
| mREBEL |  | X |  | X (trained) |  |  |  |  |  |  |
| KnowPrompt |  | X |  | X (trained/fixed) |  |  |  |  |  |  |
| UIE |  |  | X |  |  |  | X |  | X (optional) |  |
| GoLLIE |  |  |  | X | X | X | X |  |  |  |
| InstructUIE |  |  | X |  |  |  |  |  |  |  |
| GenIE |  | X | X |  |  |  | X | X | X |  |
| Pythia/SPACE-KBP |  | X |  |  | X (training/optional prompt) |  | X |  |  | X |
| SynthIE |  | X | X |  |  |  | X | X | X |  |
| DeepStruct |  | X | X |  |  |  | X (manual alignment) |  |  |  |
| PTR |  | X |  | X |  |  | X (manual rules) |  |  |  |
| GenSIE 2026 |  |  |  | X | X | X | X |  | X (model/API dependent) |  |

Important positioning consequences:

- REBEL/mREBEL receive no target Hohfeld knowledge even though their weights
  embody a fixed native schema.
- GoLLIE is the only audited released model that directly accepts detailed
  annotation guidelines for a schema defined at inference.
- UIE's structural schema is not equivalent to definitions or guidelines.
- GenIE/SynthIE constraints guarantee legal inventory membership and syntax,
  not semantic correctness.
- A formal ontology used in training by Pythia is not evidence of arbitrary
  ontology following at inference.

## 7. Legacy Pythia / SPACE-KBP audit

### 7.1 Classification and lineage

Pythia is classified as `KBP_DOMAIN_BASELINE`. The relevant work fine-tunes
EleutherAI Pythia decoder-only models to populate a fixed space-mission
ontology and serialize ontology instances as RDF Turtle. It is not the
unrelated 2026 PYTHIA KGQA project and is not a generic SOTA TE model.

Lineage:

```text
expertailab/anonkbp SPACE-KBP Parquet
    title + summary/abstract + Turtle gold
-> corpus.csv + gold_raw.csv
-> ontology-specialist Pythia generation
-> Turtle parse
-> flat triples
-> normalization and strict/flexible evaluation
```

The canonical study reports the 2.8B model as best under its property-oriented
evaluation. The local legacy experiment reports 1.4B as best under the local
triple evaluator. These are different protocols and must not be presented as a
contradiction or merged into one ranking.

### 7.2 Checkpoints and access

- The only exact task checkpoint literal in legacy code is
  `anonkbp/pythia-1B-deduped-turtle-only-response-best`.
- Result folders and tokenizer fallbacks show runs for 410M, 1B, 1.4B, 2.8B,
  and 12B, but the exact non-1B task checkpoint strings are not persisted in
  run manifests and therefore remain reconstructed rather than provenance
  facts.
- The old `anonkbp` Hugging Face URLs currently redirect to `expertailab`
  pages. Public pages were observed for 1B, 1.4B, and 2.8B.
- Cards are empty or incomplete; the 1.4B page even reports inconsistent size
  metadata, and no fine-tuned-weight license is declared.
- Base tokenizers fall back to corresponding
  `EleutherAI/pythia-*-deduped` checkpoints after a legacy tokenizer JSON
  `untagged enum` failure.

Checkpoint accessibility is therefore technically promising but scientifically
requires an explicit pre-implementation gate: resolve canonical IDs, immutable
revisions, file composition (full model versus PEFT adapter), tokenizer, and
license.

### 7.3 Input, output, and parser

- Input CSV: `doc`, optional `sent_id` (default `-1`), and `text`.
- Default input cap: 1,024 tokens; output cap: 512 new tokens.
- Default decoding is deterministic greedy generation.
- The few-shot prompt supplies one space-mission example; `--basic_prompt`
  supplies only a generic Turtle request.
- Generated target shape is Turtle with `ex:` instances and `spaceont:`
  classes/properties.
- The parser injects prefixes, strips generated prefix declarations, repairs
  bare subjects, attempts truncation closure, parses with RDFLib, and falls
  back to a SpaceKBP-specific regex.
- The exact raw Turtle is not persisted; only flattened CSV triples survive.
- Predictions assign `head_type` and `tail_type` as `unk`.

Quantization support is limited to a legacy `--load_in_8bit` path using
bitsandbytes. The dependency is not declared. No 4-bit/GPTQ path exists.

### 7.4 Historical local results

All complete Pythia results are document-level SpaceKBP few-shot runs:

| Local result | Strict triple F1 | Flexible triple F1 |
|---|---:|---:|
| Pythia-410M | 0.060 | 0.064 |
| Pythia-1B | 0.093 | 0.094 |
| Pythia-1.4B | **0.149** | **0.157** |
| Pythia-2.8B | 0.066 | 0.066 |
| Pythia-12B | 0.066 | 0.066 |

The 1.4B seed-100/seed-999 runs have identical aggregate metrics under greedy
decoding. Raw CSV ordering can vary because RDFLib graph iteration is not an
ordered prediction contract. Pythia was not run on the chunked SpaceKBP corpus;
some DDI2013, DocRED, and earlier SpaceKBP outputs are explicitly incomplete.

### 7.5 Known limitations and bugs

- Silent document truncation at 1,024 input tokens.
- Turtle truncation and heuristic repair can change the model output.
- Raw generations are not retained.
- Hardcoded ontology/example prompt.
- Space-specific fallback parser.
- Default deduplication destroys occurrence information.
- Result order depends on RDF graph iteration.
- Alias dictionaries were derived post hoc from predictions with
  model-assisted/manual review, creating circularity risk.
- Chunk-gold heuristic leaves 903 of 6,051 triples unaligned in the prepared
  chunked dataset.
- Stored document corpus can omit `sent_id`, relying on a fallback.
- No Pythia adapter unit tests.
- No checkpoint revision, environment lock, or run manifest.
- No entity canonicalization, as also acknowledged by the canonical paper.

### 7.6 Clean-room boundary

Concepts that may be reimplemented:

- A CausalLM `Extractor` preserving exact decoded output and generation
  metadata.
- Explicit basic/few-shot prompt profiles recorded in the manifest.
- Standards-based RDF/Turtle parsing as one parser implementation.
- Deterministic document/chunk aggregation and `context_mode` semantics.
- Optional, declared quantized loading.
- Syntax and ontology validation emitted as violations rather than repairs.

Do not copy verbatim:

- The Aqua few-shot prompt.
- Hardcoded namespace block or space-ontology vocabulary.
- Emergency regex parser and silent repair heuristics.
- Prediction-derived alias JSON or Gemini-assisted alias workflow.
- CSV-only output, default deduplication, silent truncation, or row skipping.
- Mutable model defaults without a resolved revision.

A clean-room adapter should preserve RAW Turtle, parse without mutating RAW,
surface parser/ontology violations, retain duplicates/order, aggregate to the
canonical document `PredictionRecord`, and leave metric matching to the
existing evaluation package.

## 8. Integration feasibility for Priority A/B

Scale: `LOW`, `MEDIUM`, `HIGH`. “Scientific comparability” means difficulty,
so `HIGH` denotes a large comparability problem.

| Candidate | Environment setup | Checkpoint acquisition | Model loading | Hohfeld input adaptation | Parser | `PredictionRecord` adaptation | Execution cost | Scientific comparability |
|---|---|---|---|---|---|---|---|---|
| Pythia/SPACE-KBP | MEDIUM | **HIGH** (identity/license/revision gate) | MEDIUM | HIGH | MEDIUM | LOW | MEDIUM | HIGH |
| UIE | HIGH (legacy pins) | MEDIUM | MEDIUM | HIGH | HIGH | MEDIUM | MEDIUM | MEDIUM |
| GoLLIE-7B | HIGH | MEDIUM | HIGH | MEDIUM | MEDIUM | MEDIUM | HIGH | MEDIUM |
| GenIE | HIGH | MEDIUM | HIGH | HIGH | MEDIUM | MEDIUM | HIGH | HIGH |
| KnowPrompt | HIGH | MEDIUM | MEDIUM | HIGH | LOW | HIGH | MEDIUM | **HIGH** |
| SynthIE | HIGH | MEDIUM | MEDIUM | HIGH | MEDIUM | MEDIUM | MEDIUM | HIGH |
| InstructUIE | HIGH | HIGH (large float32 artifact/license ambiguity) | HIGH | MEDIUM | MEDIUM | MEDIUM | HIGH | MEDIUM-HIGH |

Interpretation:

- Pythia has the lowest downstream contract cost because the legacy adapter
  already establishes the model and Turtle concepts. Its checkpoint and
  scientific-schema risks are the gates.
- UIE is smaller and easier to execute than GoLLIE, but needs more Hohfeld
  supervision and a more complex SEL parser.
- GoLLIE has the highest UIE scientific value but a CUDA/Flash Attention/Llama
  licensing burden. Its official `eval()` parser must not be copied; use a
  strict AST allowlist.
- GenIE's parser is simple, but dynamic legal entities, relation/entity tries,
  Spanish, and short context create high adaptation cost.
- KnowPrompt's prediction label is easy to parse, but converting its oracle
  pair results into document-level end-to-end predictions is scientifically
  invalid unless the pair source is explicitly represented.
- SynthIE should follow GenIE because it reuses the same closed-IE assets and
  constraints.
- InstructUIE's 11B float32 release has a poor value-to-runtime ratio.

## 9. Implementation priority and order

### 9.1 Priority A

1. **Pythia/SPACE-KBP** — first adapter because the legacy path offers maximum
   conceptual reuse and it uniquely represents ontology-specialist KBP. Before
   implementation, pass the checkpoint identity/license/revision gate. Report
   it only as `KBP_DOMAIN_BASELINE`.
2. **UIE** — first new UIE implementation because its base checkpoint is
   substantially easier to execute than GoLLIE and it adds structural schema
   plus optional constrained decoding. Treat Hohfeld zero-shot as unproven.
3. **GoLLIE-7B** — best scientific UIE comparator and the only audited model
   with detailed inference-time guidelines; implement after a viable CUDA and
   safe-parser path is confirmed.
4. **GenIE** — canonical cIE/KBP comparator; implement after defining whether
   Paper 3 can supply a defensible document-local entity inventory and a
   reified Hohfeld representation.

The tentative order `Pythia -> UIE -> GoLLIE -> GenIE` is retained. GoLLIE is
the *best scientific UIE comparator*, while UIE comes first operationally
because it provides a smaller checkpoint and an earlier feasibility signal.

### 9.2 Priority B

5. **KnowPrompt** — implement only as an explicitly labeled
   `RELATION_CLASSIFICATION` oracle-pair/few-shot stratum. Never merge its
   scores into raw-text end-to-end TE.
6. **SynthIE** — implement only after GenIE assets work. Its most promising
   contribution may be reverse synthetic-data generation rather than direct
   checkpoint transfer.
7. **InstructUIE** — implement only if sufficient hardware and an exact weight
   license are resolved. It tests labels/options versus GoLLIE guidelines, but
   its 11B sentence-level release is not a priority execution target.

### 9.3 Related work only

- **DeepStruct:** valuable precedent for structure pretraining and manual
  schema alignment, but the released 10B GLM path is operationally excessive
  and has no Spanish/document-level Hohfeld evidence.
- **PTR:** valuable precedent for rule-informed prompt construction, but it is
  sentence-level relation classification with supplied entity pairs and no
  released trained checkpoint.
- **GenSIE 2026:** current, highly relevant Spanish dynamic-schema benchmark
  and pipeline design analogue. It is not a canonical mature extraction model
  and therefore cannot be integrated as one external comparator. Reassess
  after official systems/results and a proceedings paper exist.

### 9.4 Rejected uses

No candidate is rejected from the literature review. The following uses are
rejected:

- KnowPrompt/PTR as end-to-end TE.
- GenIE/SynthIE as open-world arbitrary-schema extraction.
- Pythia/SPACE-KBP as generic SOTA TE.
- GenSIE as a released comparator model.
- Any external model as a C0-C3 condition.
- Any claim of Spanish support based only on multilingual institutions,
  multilingual base families, or untested tokenizer coverage.

## 10. Candidate-specific Paper 3 roles

| Candidate | Role | Paper 3 question |
|---|---|---|
| REBEL | `IMPLEMENTED_BASELINE` | How does an English-centric native Wikidata extractor behave on Spanish Hohfeld text? |
| mREBEL | `IMPLEMENTED_BASELINE` | What is the strongest existing Spanish native-schema end-to-end baseline? |
| Pythia/SPACE-KBP | `PRIORITY_A_IMPLEMENT` | What does a small fixed-ontology KBP specialist contribute, and where does domain transfer fail? |
| UIE | `PRIORITY_A_IMPLEMENT` | Are label/structural-schema prompts plus constrained syntax sufficient after limited Hohfeld adaptation? |
| GoLLIE | `PRIORITY_A_IMPLEMENT` | Do definitions and annotation guidelines enable unseen Hohfeld extraction? |
| GenIE | `PRIORITY_A_IMPLEMENT` | What are the benefits and costs of hard entity/relation inventory grounding? |
| KnowPrompt | `PRIORITY_B_IMPLEMENT_IF_FEASIBLE` | Given oracle entity pairs, how well can few-shot prompt tuning classify Hohfeld relations? |
| SynthIE | `PRIORITY_B_IMPLEMENT_IF_FEASIBLE` | Can synthetic supervision improve rare fixed-schema extraction after GenIE infrastructure exists? |
| InstructUIE | `PRIORITY_B_IMPLEMENT_IF_FEASIBLE` | Do task instructions and label options suffice without detailed guidelines? |
| DeepStruct | `RELATED_WORK_ONLY` | What prior evidence exists for structure pretraining and schema alignment? |
| PTR | `RELATED_WORK_ONLY` | What prior evidence exists for manually rule-informed relation prompts? |
| GenSIE 2026 | `RELATED_WORK_ONLY` (`RELATED_WORK_CURRENT`) | How should Spanish unseen-schema structured extraction be evaluated? |

## 11. Explicit scientific exclusions

- No extractor, parser, model loader, schema adapter, or dependency extra is
  implemented in this iteration.
- No model weights are downloaded.
- No complete inference/evaluation experiment is run.
- No benchmark or evaluator file is changed.
- No legacy file is changed.
- No external result is treated as a C0-C3 causal observation.
- No zero-shot capability is inferred merely because a prompt accepts text.
- No Spanish capability is inferred from a multilingual base unless the
  release explicitly documents Spanish.
- No syntactic constraint is claimed to guarantee Hohfeld semantic validity.
- No automatic derivation of Hohfeld correlatives is attributed to these IE
  models. Extraction of expressed positions and deterministic doctrinal
  post-processing are separate operations.
- No model trained on a fixed ontology is described as accepting arbitrary
  unseen ontologies.
- No result with supplied/gold entity pairs is compared directly with
  end-to-end entity-and-relation extraction.

## 12. Primary references, repositories, and checkpoints

### RE/TE

- REBEL paper: https://aclanthology.org/2021.findings-emnlp.204/
- REBEL/mREBEL repository: https://github.com/Babelscape/rebel
- REBEL checkpoint: https://huggingface.co/Babelscape/rebel-large
- mREBEL paper: https://aclanthology.org/2023.acl-long.237/
- mREBEL checkpoints:
  https://huggingface.co/Babelscape/mrebel-large,
  https://huggingface.co/Babelscape/mrebel-large-32,
  https://huggingface.co/Babelscape/mrebel-base
- KnowPrompt paper: https://doi.org/10.1145/3485447.3511998
- KnowPrompt repository: https://github.com/zjunlp/KnowPrompt
- KnowPrompt checkpoint: https://huggingface.co/zjunlp/KnowPrompt
- DeepStruct paper: https://aclanthology.org/2022.findings-acl.67/
- DeepStruct repository: https://github.com/wang-research-lab/deepstruct
- PTR paper: https://doi.org/10.1016/j.aiopen.2022.11.003
- PTR repository: https://github.com/thunlp/PTR

### UIE

- UIE paper: https://aclanthology.org/2022.acl-long.395/
- UIE repository: https://github.com/universal-ie/UIE
- UIE checkpoints:
  https://huggingface.co/luyaojie/uie-base-en,
  https://huggingface.co/luyaojie/uie-large-en
- GoLLIE paper: https://openreview.net/forum?id=Y3wpuxd7u9
- GoLLIE repository: https://github.com/hitz-zentroa/GoLLIE
- GoLLIE collection:
  https://huggingface.co/collections/HiTZ/gollie-651bf19ee315e8a224aacc4f
- InstructUIE paper: https://arxiv.org/abs/2304.08085
- InstructUIE repository: https://github.com/BeyonderXX/InstructUIE
- InstructUIE checkpoint: https://huggingface.co/ZWK/InstructUIE

### KBP / closed IE

- GenIE paper: https://aclanthology.org/2022.naacl-main.342/
- GenIE repository: https://github.com/epfl-dlab/GenIE
- GenIE artifacts: https://zenodo.org/records/6139236
- SPACE-KBP/Pythia study: https://arxiv.org/abs/2503.18502
- SPACE-KBP project: https://github.com/expertailab/LLM4KBP
- SPACE-KBP dataset:
  https://huggingface.co/datasets/expertailab/SPACE-KBP
- Pythia base project: https://github.com/EleutherAI/pythia
- Legacy-observed task checkpoint redirect:
  https://huggingface.co/anonkbp/pythia-1B-deduped-turtle-only-response-best
- SynthIE paper: https://aclanthology.org/2023.emnlp-main.96/
- SynthIE repository: https://github.com/epfl-dlab/SynthIE
- SynthIE models: https://huggingface.co/martinjosifoski/SynthIE
- GenSIE proposal: https://uhgia.org/gensie/gensie.pdf
- GenSIE site: https://uhgia.org/gensie/
- GenSIE starter kit: https://github.com/gia-uh/gensie

## 13. Unresolved questions and pre-implementation gates

### Cross-cutting

1. What exact Hohfeld record is evaluated: a binary triple, a reified legal
   position, or a richer event-like structure with holder, counterparty,
   action/object, condition, polarity, authority, time, and provenance?
2. Are only textually explicit positions evaluated, or also doctrinally
   inferred correlatives/opposites? The latter needs a separate controlled
   reasoning layer.
3. What entity source is permitted for relation-classification and closed-IE
   strata: gold spans, detected spans, document-local inventory, or global KB?
4. What Spanish adaptation is methodologically allowed: direct inference,
   translation, supervised fine-tuning, or continued pretraining?
5. What hardware is actually available for GoLLIE and InstructUIE?
6. How will chunking, cross-chunk coreference, duplicate retention, and
   document-level aggregation be standardized across external systems?

### Candidate gates

- **Pythia:** resolve canonical fine-tuned checkpoint IDs, immutable revisions,
  tokenizer, adapter/full-weight format, and license before implementation.
- **UIE:** verify repository/checkpoint license text, modern-environment
  compatibility, and whether a pure inference path exists independently of
  downstream fine-tuning.
- **GoLLIE:** verify CUDA/Flash Attention environment, Llama 2 weight terms,
  quantized quality, and safe AST parsing.
- **GenIE:** decide document-local versus global entity inventory and whether
  reified legal acts can satisfy closed-IE grounding.
- **KnowPrompt:** define and report the entity-pair oracle/upstream source;
  otherwise do not implement.
- **SynthIE:** decide whether the target is direct checkpoint comparison or a
  separately governed synthetic-data methodology.
- **InstructUIE:** identify the exact OpenRAIL variant and a feasible converted
  checkpoint before spending integration effort.
- **GenSIE:** reassess after official 2026 shared-task systems/results and
  publication status are stable.

## 14. P0 readiness

This audit supports adapter implementation planning but authorizes no runtime
change. The architecture is reusable, task non-equivalence is explicit,
schema-knowledge channels are separated, and every Priority A/B candidate has
an evidence-based feasibility classification and gate.

`EXTERNAL_SYSTEMS_AUDIT_P0_READY = YES`

## Post-audit status

This section records later implementation state without rewriting the audit
snapshot or its original decisions:

- Pythia/SPACE-KBP clean-room adapter and conservative Turtle parser:
  implemented.
- Canonical checkpoint and immutable revision: pinned.
- Weight-free clean-room test suite: passing.
- Real-model smoke: blocked and unauthorized because the fine-tuned checkpoint
  has no explicit weight license.
- UIE clean-room adapter, deterministic SSI builder, conservative SEL parser,
  typed Spot-Association representation, exact alignment, and binary relation
  projection: implemented.
- UIE checkpoint/revision and CC BY-NC-SA metadata: pinned and documented.
- UIE custom-schema interface: implemented; unseen-schema zero-shot scientific
  support remains `UNKNOWN`.
- UIE main-environment positive smoke: pending suitable GPU/native validation;
  the legacy CPU empty-tree run is contract evidence only.

See `PYTHIA_BASELINE.md` and `UIE_BASELINE.md` for current technical policy.
