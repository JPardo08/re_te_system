# TeresIA systems ecosystem audit

Audit date: 2026-09-22
Iteration: `TERESIA_SYSTEMS_ECOSYSTEM_AUDIT`
Scope: systems, models, tools, and assets related to TeresIA that may inform
Paper 3 or the thesis. This is a documentation-only audit.

## 1. Method and evidence boundary

The audit distinguishes:

1. **Internal assets:** datasets, annotations, ontologies, terminology
   resources, and historical project artifacts used by the thesis.
2. **External project systems:** executable tools or released models developed
   by TeresIA partners or explicitly delivered in TeresIA, but not integrated
   into `re_te_system`.
3. **Partner-adjacent systems:** relevant work from the same institutions whose
   official sources do not establish that it is a TeresIA output.
4. **General SOTA:** systems such as REBEL, mREBEL, UIE, GoLLIE, and GenIE that
   are not TeresIA project developments.

Primary evidence includes the official TeresIA and BSC websites, the official
2024 TeresIA agreement published in the Spanish Official Gazette, canonical
papers, PyPI metadata, official Hugging Face model cards/APIs, official
repositories, and local read-only legacy provenance.

No Notion connector or local Notion export was available in this session.
Consequently, the “Notion de Tesis” could not be independently inspected.
Claims that depend only on Notion are not made; this access limitation remains
an unresolved evidence gap rather than being silently replaced by inference.

Public GitHub API access through the configured CLI returned an authentication
denial. Repository facts below are therefore limited to official indexed
repository pages, package metadata, model cards, papers, and local clones. No
model or package was downloaded.

## 2. Executive conclusions

- **KeyCARE is the only clearly released BSC system explicitly developed as
  part of TeresIA for term extraction, term categorization, and semantic
  relation classification.**
- KeyCARE relation classification is pairwise terminology classification:
  supplied term pair -> `BROAD`, `EXACT`, `NARROW`, or `NO_RELATION`. It is not
  contextual relation extraction and not Hohfeld TE.
- The full KeyCARE workflow can be assembled as
  text -> candidate terms -> biomedical categories -> selected term-pair
  relations, but pair construction is caller-controlled and the released
  categorization/relation checkpoints are biomedical.
- **TermitUp** is an explicit OEG-UPM predecessor named in TeresIA task T3.3
  for semantic relation extraction and terminology linking. It is not itself a
  new TeresIA runtime component.
- **TermiGraph** and **TermonIA** are direct TeresIA services for RDF conversion
  and human validation respectively. They are highly relevant to the thesis
  lifecycle but are not Paper 3 extraction comparators.
- **MEL** is a Spanish legal encoder developed in INESData by IIC/UPM authors.
  No official source establishes it as a TeresIA component. It is valuable
  legal domain pretraining, not a ready extractor.
- No newly discovered TeresIA system should be Priority A for Hohfeld TE.
  KeyCARE is `PRIORITY_B_IMPLEMENT_IF_FEASIBLE`; several project and
  partner-adjacent systems are `RELATED_WORK_HIGH_RELEVANCE`.

## 3. Internal assets versus systems

### 3.1 Internal assets

These resources are thesis/project data or representation assets, not external
systems:

| Asset | Classification | Paper 3 use |
|---|---|---|
| Hohfeld Gold | Internal independent TE asset | Primary Hohfeld evaluation target |
| mREBEL human-validated Silver | Internal model-dependent asset | Weak supervision / acceptance analysis, not independent mREBEL gold |
| TeresIA legal and biomedical terminology corpora | Internal terminology assets | Term-extraction evaluation/training; not Hohfeld TE |
| Labour-law SKOS/JSON-LD terminology and dictionaries | Internal/external-source terminology knowledge | Retrieval or terminology conditioning, not Hohfeld relation gold |
| OEG annotation configs, Hohfeld projections, and conversion notebooks | Internal provenance/tooling | Reproducibility and schema evidence |

The canonical Hohfeld layers must not be conflated: 95 raw canonical
annotations, 65 sentence-aligned projections, and 41 span-complete projected
annotations answer different evaluation questions.

### 3.2 External project systems

| System | Partner | Direct TeresIA evidence | Function |
|---|---|---|---|
| KeyCARE | BSC/NLP4BIA | Explicitly developed as part of TeresIA; named in official BSC and TeresIA outputs | Term extraction, term categorization, pairwise hierarchical relation classification |
| TermiGraph | UPM/OEG / TeresIA | TeresIA deliverable 4.2 and official project repository/service | Convert supported terminology formats to RDF |
| TermonIA | TeresIA/AETER/partners | Official TeresIA validation service | Human expert validation and linguistic sanctioning |
| TeresIA portal/metasearch infrastructure | Consortium | Core project platform | Search, ingest, manage, validate, and expose terminologies |
| TeresIA-hosted AttentionRank/MDERank artifacts | Project/OEG | Repositories exist under project/partner organizations; exact fork/provenance maturity is incompletely documented | Automatic keyword/term extraction |

### 3.3 General SOTA

REBEL, mREBEL, UIE, GoLLIE, GenIE, SynthIE, KnowPrompt, InstructUIE, and
Pythia/SPACE-KBP remain general external comparators. A TeresIA wrapper,
dataset, or experiment does not make the underlying third-party model a
TeresIA-developed model.

## 4. KeyCARE detailed audit

### 4.1 Identity and explicit TeresIA relation

Name: **KeyCARE — Keyword Extraction, term Categorization, and semantic
Relation**.

Institution:

- Natural Language Processing for Biomedical Information Analysis (NLP4BIA).
- Barcelona Supercomputing Center / Centro Nacional de Supercomputación.
- Principal developer: Sergi Marsol Torrent.
- Thesis supervisors: Luis Gascó and Martin Krallinger.

Official artifacts:

- Repository: https://github.com/nlp4bia-bsc/KeyCARE
- PyPI: https://pypi.org/project/keycare/
- Hugging Face organization: https://huggingface.co/BSC-NLP4BIA
- Detailed technical publication:
  https://hdl.handle.net/2445/213240
- Full thesis:
  https://diposit.ub.edu/server/api/core/bitstreams/ade340fd-f881-4b72-9b96-172f80042323/content
- Official BSC TeresIA announcement:
  https://www.bsc.es/news/bsc-news/bsc-uses-the-latest-advances-ai-develop-spanish-terminology-learning-systems-the-teresia-project

The UB thesis states that KeyCARE was developed as part of TeresIA. The BSC
announcement independently states that BSC's TeresIA participation includes
developing KeyCARE for key-term extraction, classification, and relation
extraction between terms. TeresIA deliverables 3.5/3.6 describe the same three
functional areas.

The detailed canonical publication is a University of Barcelona bachelor's
thesis dated 2024-06-05. A four-page KeyCARE contribution appears in the
CASEIB 2024 proceedings, pages 137–140. No standalone peer-reviewed technical
paper with a complete model/evaluation description was found; the repository
and model cards still say that publication/evaluation is forthcoming.

### 4.2 Package, version, compatibility, and licenses

| Property | Evidence |
|---|---|
| Package | `keycare` |
| Current PyPI version | `0.1.0` |
| Release date | 2025-07-07 |
| Distribution | Wheel and source distribution are both listed by PyPI |
| Declared Python | `>=3.9` |
| Development environment | Python 3.10.12 |
| Verified modern matrix | Not published; Python 3.12 compatibility is unverified |

PyPI 0.1.0 tightly pins a 2023-era stack including:

- PyTorch 2.0.1;
- Transformers 4.33.0;
- spaCy 3.6.1;
- SetFit 0.7.0;
- Sentence Transformers 2.2.2;
- Datasets 2.14.4;
- YAKE, RAKE-NLTK, PyTextRank, KeyBERT, NLTK, pandas, and scikit-learn.

Runtime can require Hugging Face checkpoint retrieval, NLTK resources, and
`es_core_news_sm`. CPU inference is possible in source; GPU is not documented
as mandatory for inference.

License evidence is inconsistent:

- Repository `LICENSE`, repository metadata, and PyPI license metadata: **MIT**.
- README license section: **Apache-2.0**.
- Released BSC model cards: **Apache-2.0**.
- UMLS and SNOMED CT training sources have independent access/redistribution
  terms that are not replaced by the model license.

The governing code-license artifact should be treated as MIT, while the README
inconsistency must be resolved before code reuse. Each model remains
Apache-2.0 according to its own card.

### 4.3 Architecture and modules

KeyCARE exposes two public orchestration classes:

```text
TermExtractor
    A. candidate-term extraction
    B. candidate-term categorization

RelExtractor
    C. supplied term-pair relation classification
```

Internal module families:

- `extractors`: RAKE, YAKE, TextRank, KeyBERT.
- `categorizers`: clustering, SetFit, Transformers classification.
- `relators`: SetFit and Transformers pair classifiers.
- records: `Keyword` and `Relation`.

The end-to-end terminology flow is conceptual rather than one atomic API:
application code must call `TermExtractor`, select/build pairs, and then call
`RelExtractor`.

## 5. KeyCARE A — term extraction

Input:

```text
TermExtractor.__call__(text: str)
```

Default and available methods:

- default: TextRank;
- RAKE;
- YAKE;
- KeyBERT;
- multiple extractors can be selected.

Important defaults include Spanish, keyphrases up to three tokens, postprocess
enabled, and a result count derived from text length when `top_n` is absent.

Output:

- `__call__` mutates the extractor and returns `None`;
- results are available in `termextractor.keywords`;
- each `Keyword` carries text, character span `[start, end)`, extraction method,
  score, and later categorization fields.

Scientific boundary:

- this is unsupervised/statistical/neural keyphrase extraction, not supervised
  NER;
- it does not guarantee exhaustive term spans;
- A is the only released KeyCARE component with plausible immediate legal
  transfer, but that transfer must be evaluated against TeresIA legal term gold.

## 6. KeyCARE B — term categorization

Input:

- `Keyword` objects from A;
- only `Keyword.text` reaches the classifier;
- sentence/document context is not supplied.

Methods:

- SetFit by default;
- Transformers `AutoModelForSequenceClassification`;
- clustering.

Released checkpoints:

| Mode | Checkpoint | Immutable observed revision | License |
|---|---|---|---|
| SetFit | `BSC-NLP4BIA/biomedical-term-classifier-setfit` | `f5c4b143bbff3df572cf6ab74cdc4f4813863770` | Apache-2.0 |
| Transformer | `BSC-NLP4BIA/biomedical-term-classifier` | `df709f287bf36c42a28a259843bf1424cbd70b4e` | Apache-2.0 |

Both are Spanish biomedical classifiers based on the BSC Spanish biomedical
SapBERT/RoBERTa line. Training sources include MedProcNER, DisTEMIST,
SympTEMIST, CANTEMIST, and PharmaCoNER.

The fixed 21-label inventory is:

```text
ACTIVIDAD, COMUNIDAD, DEPARTAMENTO, ENFERMEDAD, FAC_GEN, FAC_NOM,
FARMACO, GEO_GEN, GEO_NOM, GPE_GEN, GPE_NOM, HUMAN, IDIOMA,
MORFOLOGIA_NEOPLASIA, NO_CATEGORY, PROCEDIMIENTO, PROFESION, SINTOMA,
SITUACION_LABORAL, SPECIES, TRANSPORTE
```

Output mutates each `Keyword` with a label list and categorization method.
These biomedical categories are not a legal terminology schema and not
Hohfeld types.

## 7. KeyCARE C — semantic relation classification

### 7.1 Exact task contract

`RelExtractor` requires supplied terms:

```text
relextractor(source, target)
```

Each side can be one string/`Keyword` or a list. Lists are paired positionally
unless `all_combinations=True`, in which case the Cartesian product is
classified.

The model receives:

- two term strings;
- their order;
- no sentence;
- no document;
- no trigger/evidence span;
- no actor/entity type;
- no relation context.

`__call__` returns `None`; results are stored in
`relextractor.relations`. Each `Relation` contains source, target, a relation
label list, and the method.

### 7.2 Fixed relation inventory

```text
BROAD
EXACT
NARROW
NO_RELATION
```

- `EXACT`: terminological equivalence/synonymy.
- `BROAD`: source is broader than target.
- `NARROW`: source is narrower than target.
- `NO_RELATION`: no supported equivalence/hierarchy relation.

These are lexical/terminological hierarchy relations, not arbitrary semantic
relations and not Hohfeld jural relations.

### 7.3 Checkpoints, architecture, and training

| Mode | Checkpoint | Observed revision | Architecture | License |
|---|---|---|---|---|
| Transformer | `BSC-NLP4BIA/biomedical-semantic-relation-classifier` | `9521ea298af03d1d922672ccc79361f1babdb468` | `RobertaForSequenceClassification`, pair input | Apache-2.0 |
| SetFit | `BSC-NLP4BIA/biomedical-semantic-relation-classifier-setfit` | `840d2447d8f79852bcdafef0b513e2d35b730fd6` | RoBERTa SentenceTransformer + SetFit head | Apache-2.0 |
| Base encoder | `BSC-NLP4BIA/SapBERT-from-roberta-base-biomedical-clinical-es` | `379ed9999e76a012f6b0403354a9c8b6b3024cb8` | Spanish biomedical RoBERTa/SapBERT | Apache-2.0 |

The base encoder derives from
`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` and Spanish UMLS 2023AA.
The relation classifiers are trained from SNOMED CT hierarchy mapped to UMLS
medical terms.

The Transformer path tokenizes source and target as a sequence pair. The
SetFit path concatenates their texts in order. The API's `language` parameter
does not establish multilingual relation support; released cards declare
Spanish.

The thesis reports strong results on automatically generated SNOMED/UMLS-like
pairs but much lower performance on a separate manually annotated set. It also
reports confusion between semantically similar `NO_RELATION` pairs and
hierarchical relations. Model-card evaluation remains “to be published.”

`model_path` can replace weights, but the four labels and parts of tokenizer
handling are fixed in the released implementation. Supplying a Hohfeld
checkpoint is not a documented plug-in capability.

## 8. KeyCARE comparability

### 8.1 `ORACLE_PAIR_RELATION_CLASSIFIER`

This mode matches the released relation API:

```text
provided/gold term pair -> BROAD | EXACT | NARROW | NO_RELATION
```

It is valid for a separate terminology-alignment experiment when the target
task is equivalence/hypernymy/hyponymy. It is not end-to-end because pair
discovery is external.

For legal text:

- Spanish encoder support: yes;
- legal-domain training: no;
- legal relation labels: no;
- sentence context: no;
- Hohfeld labels: no.

Using the released model on legal pairs would be a domain-transfer diagnostic,
not evidence of legal semantic-relation competence.

### 8.2 `PIPELINE_TERMINOLOGY_IE`

Conceptual contract:

```text
text
-> candidate keyphrases
-> biomedical category labels
-> caller-selected term pairs
-> terminology hierarchy/equivalence labels
```

This pipeline is directly relevant to terminology creation/enrichment. It is
not Hohfeld TE because it extracts terms rather than actor–jural
relation–counterparty structures, and because C receives no legal context.

### 8.3 Compatibility verdicts

| Target | Verdict |
|---|---|
| Spanish | Supported by released checkpoints; extraction method details still require empirical validation |
| Biomedical text | Native intended domain |
| Spanish legal term extraction | Plausible only for A; unvalidated |
| Legal term categories | Unsupported by released B inventory |
| Hohfeld benchmark | Incompatible off the shelf |
| TeresIA legal terminology corpus | Suitable evaluation/retraining source in principle; no evidence released checkpoints used it |
| Low-resource experiments | Strong methodological fit through unsupervised A and SetFit; no legal/Hohfeld result |

KeyCARE task classification:

```text
task_class = TERMINOLOGY_RELATION_PIPELINE
```

Its C component can additionally be tagged
`ORACLE_PAIR_RELATION_CLASSIFIER`; its assembled A→B→C mode can be tagged
`PIPELINE_TERMINOLOGY_IE`. Neither tag means Hohfeld TE.

## 9. Other direct TeresIA systems

### 9.1 TermiGraph

Direct TeresIA service from deliverable 4.2:

- repository: https://github.com/proyectoTeresIA/TermiGraph_esp
- service: https://termigraph.teresia.linkeddata.es/
- task: convert supported terminological resources to RDF;
- input families: plain monolingual glossaries, TBX-Basic, TERMCAT XML, IATE
  JSON, and UNTerm-like Excel;
- process: source-specific preprocessing -> Mapeathor-generated RML mappings ->
  RMLMapper RDF transformation;
- output: RDF under the TeresIA representation model;
- code license: not established by the inspected official repository.

TermiGraph does not extract relations from free text. It is
`RELATED_WORK_HIGH_RELEVANCE` for downstream representation and KG population.

### 9.2 TermonIA

Official TeresIA web application:

- page: https://proyectoteresia.org/termonia
- task: expert conceptual validation and linguistic sanctioning of candidate
  terminology;
- input/output: candidate terminology and review workflow -> expert validation
  decisions;
- model: human-in-the-loop system, not an IE model;
- license/repository: not established from inspected official pages.

TermonIA is `RELATED_WORK_HIGH_RELEVANCE` for human validation methodology,
not an extraction comparator.

### 9.3 Portal and metasearch infrastructure

The portal provides unified terminology search, semantic search, management,
and conversational access. It integrates terminology/ontology storage and
service infrastructure. It is `RELATED_WORK_ONLY` for Paper 3 because it
consumes and exposes resources rather than establishing an extraction model.

### 9.4 Project-hosted term extraction implementations

Official/project repositories include AttentionRank and MDERank-related
artifacts. OEG's `mderanklib` is a documented Apache-2.0 library adapted for
Spanish and English automatic keyword extraction. `spanish-termex` is an
Apache-2.0 benchmark/research implementation.

These are `RELATED_WORK_HIGH_RELEVANCE` for the terminology-extraction track,
but not Hohfeld relation extraction. Exact provenance between project-hosted
copies and upstream algorithms must be pinned before any implementation.

## 10. Explicit partner predecessors and adjacent systems

### 10.1 TermitUp

The official TeresIA agreement's task T3.3 says prior work such as TermitUp
will be reused for relation extraction between terms.

- paper: https://doi.org/10.3233/SW-222885
- repository: https://github.com/Pret-a-LLOD/termitup
- service: https://termitup.oeg.fi.upm.es/
- license: Apache-2.0;
- input: domain corpus;
- pipeline: statistical term extraction, post-processing, enrichment,
  disambiguation/linking, term-relation validation, and RDF publication;
- output: enriched linked terminologies in SKOS/OntoLex JSON-LD and a SPARQL
  endpoint;
- domains: domain-independent design with reported legal-domain applications.

TermitUp is an explicit architectural predecessor, not a Hohfeld TE model.
Role: `RELATED_WORK_HIGH_RELEVANCE`.

### 10.2 MEL

MEL means **Modelo de Español Legal / Legal Spanish Language Model**.

- model: https://huggingface.co/IIC/MEL
- paper: https://arxiv.org/abs/2501.16011
- developers: IIC with OEG-UPM coauthors;
- project: official IIC source attributes it to **INESData**, not TeresIA;
- architecture: XLM-RoBERTa-large encoder, 559,890,432 parameters;
- training: continued masked-language-model pretraining on 5.52 million
  Spanish legal text chunks, about 92.7 GB;
- sources: official gazettes, parliamentary material, judgments, statutes, and
  other legal texts;
- input/output: tokenized Spanish legal text -> contextual representations;
- downstream extraction/classification: requires task-specific heads and
  fine-tuning;
- observed revision: `4584d5998b31adaeb9faebf2d960354971a513b5`;
- license: MEL non-commercial research license, not an open permissive license.

MEL is not a ready term extractor or relation classifier and was officially
evaluated primarily on legal text classification. Role:
`RELATED_WORK_HIGH_RELEVANCE` as a legal-domain encoder and possible future
controlled backbone, not a TeresIA external system.

### 10.3 Term-RAG and legal ATE studies

The OEG-UPM Term-RAG study uses controlled terminology for query expansion in
Spanish legal retrieval and acknowledges both INESData and TeresIA funding:
https://aclanthology.org/2025.ldk-1.16/

The TeresIA legal terminology extraction study creates/evaluates a Spanish
labour-law term gold standard with morphosyntactic patterns, YAKE, LLaMA3-8B,
and Mistral-7B:
https://aclanthology.org/2025.ranlp-1.98/

These are thesis-relevant experiments/resources, not reusable Hohfeld TE
systems. Role: `RELATED_WORK_HIGH_RELEVANCE`.

### 10.4 Narrow mapping/linking precedents

Partner systems such as TEMUNormalizer, Sem4Tags, and CIDER-CL provide
biomedical normalization, contextual DBpedia linking, and ontology alignment.
They are relevant only if the thesis adds a dedicated terminology mapping or
entity-linking stratum. They are not Paper 3 extraction comparators and remain
`RELATED_WORK_ONLY` unless that scope is explicitly activated.

## 11. Knowledge-source taxonomy

The taxonomy records knowledge that materially conditions a system. A blank
cell means unsupported, not absent by assumption.

| System | DOMAIN_PRETRAINING | TERMINOLOGY_KNOWLEDGE | FIXED_NATIVE_SCHEMA | FIXED_ONTOLOGY_SPECIALIZATION | STRUCTURAL_SCHEMA | DEFINITIONS_GUIDELINES | KB_CONSTRAINTS | FORMAL_ONTOLOGY | INFERENCE_TIME_TARGET_SCHEMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MEL | X (Spanish legal corpus) |  |  |  |  |  |  |  |  |
| KeyCARE | X (Spanish biomedical) | X (UMLS/SNOMED and NER corpora) | X (21 categories / 4 relations) | X (training-source specialization) |  |  |  |  |  |
| mREBEL |  |  | X (Wikidata-derived relations/types) |  |  |  |  |  |  |
| Pythia/SPACE-KBP | X (task SFT, not continued pretraining) |  | X | X | X (Turtle learned) |  |  | X (training ontology) |  |
| UIE |  |  |  |  | X |  |  |  | X (labels/structure) |
| GoLLIE |  |  |  |  | X | X |  |  | X |
| GenIE |  |  | X | X (fixed Wikidata inventory) | X |  | X |  |  |
| Controlled C0 | Held constant across conditions, model-dependent |  |  |  |  |  |  |  |  |
| Controlled C1 | Same as C0 |  |  |  |  |  |  |  | X (labels) |
| Controlled C2 | Same as C0 | X where target terminology is supplied |  |  | X (types/signatures) | X |  |  | X |
| Controlled C3 | Same as C0 | X where encoded |  | X (target ontology) | X | X | X where enforced | X | X |

Clarifications:

- Pythia's domain adaptation is supervised ontology-task fine-tuning, not
  generic legal/space domain pretraining; the table flags that distinction.
- KeyCARE learns from SNOMED/UMLS structures, but no formal ontology is
  supplied at inference.
- MEL supplies domain representations only; it does not supply a task schema.
- C0-C3 must use the same controlled model. Domain pretraining is held
  constant and is not itself the manipulated knowledge source.

## 12. Paper 3 and thesis roles

| System/resource | Role | Rationale |
|---|---|---|
| mREBEL | `IMPLEMENTED_BASELINE` | Existing Spanish end-to-end external baseline; wrong native schema remains explicit |
| KeyCARE | `PRIORITY_B_IMPLEMENT_IF_FEASIBLE` | Official TeresIA system; useful term extraction/domain-transfer/oracle-pair terminology experiment, not Hohfeld TE |
| TermitUp | `RELATED_WORK_HIGH_RELEVANCE` | Explicit T3.3 predecessor for terminology relation/linking pipeline |
| TermiGraph | `RELATED_WORK_HIGH_RELEVANCE` | Direct TeresIA RDF conversion and representation layer |
| TermonIA | `RELATED_WORK_HIGH_RELEVANCE` | Direct TeresIA human validation layer |
| MEL | `RELATED_WORK_HIGH_RELEVANCE` | Strong Spanish legal domain encoder, but separate INESData output and no extraction head |
| OEG mderanklib / Spanish-TermEx | `RELATED_WORK_HIGH_RELEVANCE` | Reproducible Spanish terminology extraction baselines |
| Legal ATE gold/LLM study | `RELATED_WORK_HIGH_RELEVANCE` | Internal terminology benchmark/methodology, not TE comparator |
| Term-RAG | `RELATED_WORK_HIGH_RELEVANCE` | Demonstrates terminology knowledge in Spanish legal retrieval, outside extraction |
| TeresIA portal | `RELATED_WORK_ONLY` | Infrastructure consuming/exposing terminologies |
| TEMUNormalizer / Sem4Tags / CIDER-CL | `RELATED_WORK_ONLY` | Narrow mapping/linking precedents outside current task |

No newly audited TeresIA system receives `PRIORITY_A_IMPLEMENT`. KeyCARE is
Priority B because:

1. mode A can be tested against legal terminology gold without claiming
   Hohfeld extraction;
2. mode C has a clean oracle-pair contract useful for terminology hierarchy;
3. B/C require new legal labels/training and C lacks context;
4. a Hohfeld adaptation would be a new model/task, not straightforward use of
   released KeyCARE.

## 13. Explicit exclusions and rejected claims

- KeyCARE pair classification is not end-to-end relation extraction.
- KeyCARE A→B→C is terminology IE, not Hohfeld TE.
- Biomedical Spanish support is not evidence of legal-domain support.
- `BROAD/EXACT/NARROW/NO_RELATION` must not be mapped ad hoc to
  `Duty/Right/Privilege/NoRight`.
- MEL is not claimed as a TeresIA component.
- TermiGraph conversion is not free-text KG population.
- TermonIA expert validation is not a model baseline.
- Portal, metasearch, and conversational services are not extraction systems.
- TeresIA corpora, Hohfeld Gold, Silver, SKOS resources, and representation
  ontologies are assets, not external systems.
- General SOTA models remain separate from TeresIA project systems.

## 14. Unresolved questions

1. What additional systems or decisions are recorded only in the unavailable
   thesis Notion?
2. Which license governs the exact KeyCARE 0.1.0 code if the MIT license file
   and Apache README statement conflict?
3. Are legal-domain KeyCARE checkpoints planned or available privately?
4. Is the complete TeresIA annotated terminology corpus publicly versioned and
   licensed for model training?
5. What exact 21-class schema/version produced each KeyCARE categorizer?
6. Can KeyCARE 0.1.0 install unchanged on the canonical Python 3.12 thesis
   environment despite tightly pinned dependencies?
7. Are project-hosted AttentionRank/MDERank repositories forks, frozen
   experiment copies, or maintained TeresIA deliverables?
8. What license governs TermiGraph and TermonIA source/artifacts?
9. Which TermitUp components were actually reused in TeresIA T3.3/T4 rather
   than cited as prior work?

## 15. Primary references

### TeresIA

- Official project: https://proyectoteresia.org/proyecto
- Results/deliverables: https://proyectoteresia.org/resultados-proyecto-teresia
- 2024 official agreement:
  https://boe.es/diario_boe/txt.php?id=BOE-A-2024-16648
- 2026 amendment:
  https://www.boe.es/diario_boe/txt.php?id=BOE-A-2026-17600
- Overview paper: https://ceur-ws.org/Vol-3703/paper2.pdf

### KeyCARE/BSC

- Repository: https://github.com/nlp4bia-bsc/KeyCARE
- PyPI: https://pypi.org/project/keycare/
- UB thesis: https://hdl.handle.net/2445/213240
- BSC announcement:
  https://www.bsc.es/news/bsc-news/bsc-uses-the-latest-advances-ai-develop-spanish-terminology-learning-systems-the-teresia-project
- Transformer relation model:
  https://huggingface.co/BSC-NLP4BIA/biomedical-semantic-relation-classifier
- SetFit relation model:
  https://huggingface.co/BSC-NLP4BIA/biomedical-semantic-relation-classifier-setfit
- Transformer term model:
  https://huggingface.co/BSC-NLP4BIA/biomedical-term-classifier
- SetFit term model:
  https://huggingface.co/BSC-NLP4BIA/biomedical-term-classifier-setfit
- Spanish SapBERT:
  https://huggingface.co/BSC-NLP4BIA/SapBERT-from-roberta-base-biomedical-clinical-es

### OEG/UPM and adjacent

- TermitUp paper: https://doi.org/10.3233/SW-222885
- TermitUp repository: https://github.com/Pret-a-LLOD/termitup
- TermiGraph repository:
  https://github.com/proyectoTeresIA/TermiGraph_esp
- TermonIA: https://proyectoteresia.org/termonia
- Spanish mderanklib: https://github.com/oeg-upm/mderanklib
- Spanish-TermEx: https://github.com/oeg-upm/spanish-termex
- MEL: https://huggingface.co/IIC/MEL
- MEL paper: https://arxiv.org/abs/2501.16011
- Legal ATE study: https://aclanthology.org/2025.ranlp-1.98/
- Term-RAG study: https://aclanthology.org/2025.ldk-1.16/

`TERESIA_SYSTEMS_AUDIT_READY = YES`
