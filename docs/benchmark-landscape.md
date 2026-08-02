# Public Benchmark and Baseline Landscape

> Audited through 2026-08-02.  This document distinguishes paper claims,
> released artifacts, and ReprForge inferences.  A leaderboard number is not
> treated as reproducible unless the corresponding data and evaluation path are
> public.

## Why benchmark choice is part of the research question

“Long-document benchmark” currently refers to at least four different tasks:

1. a long-context VLM reads a PDF that is already selected;
2. a retriever locates evidence pages inside one known PDF;
3. a retriever searches a corpus containing many PDFs;
4. a system builds and maintains the physical index used by repeated queries.

These tasks reward different systems.  The first mainly measures context
reasoning.  The second measures within-document localization but can hide the
cost of identifying the document.  The third measures a real retrieval index.
The fourth is ReprForge's systems problem, but almost no public benchmark
provides natural document-arrival, update, or query timestamps.

ReprForge therefore needs a *suite*, not one headline dataset:

- a full-corpus benchmark for retrieval quality;
- a corpus where text and vision have demonstrably complementary failures;
- a scale or hard-negative stress test;
- an end-to-end answer-use benchmark;
- an explicit synthetic workload contract for lifecycle claims until a public
  temporal trace is found.

## Benchmark map

The “index fit” column answers whether the benchmark can directly test a
physical representation/index policy.  High does not mean the benchmark is
perfect; it means that queries, corpus items, and relevance labels are exposed.

| Benchmark | Problem actually defined | Public scale | Official metric | Artifact status | Index fit | Important blind spot |
|---|---|---:|---|---|---|---|
| [ViDoRe v3](https://arxiv.org/abs/2601.08620) | Full-corpus page retrieval over professional, visually rich documents; multi-page and cross-document queries | about 26K pages, 3,099 human-verified queries, 10 domains, queries in 6 languages | nDCG@10; grounding and answer tasks also available | [datasets](https://huggingface.co/vidore) and [pipeline evaluator](https://github.com/illuin-tech/vidore-benchmark) released | **High** | Frozen corpus and no natural query chronology; official indexing time excludes some environment/model-loading costs unless a submission reports them |
| [MMDocIR](https://aclanthology.org/2025.emnlp-main.1576/) | Page and layout retrieval *inside a known long document* | 313 documents, mean 65.1 pages, 1,658 expert queries; 173,843 bootstrapped training queries | Recall and nDCG at page/layout cutoffs | paper data and evaluation used by this repository | Medium | Within-document candidate pools are not full-corpus retrieval; irrelevant layouts in other documents are never tested |
| [IRPAPERS](https://arxiv.org/abs/2602.17687) | Needle-in-a-haystack page retrieval and QA over a semantically dense scientific corpus, with image and text forms for every page | 166 papers, 3,230 pages, 180 queries | Recall@1/5/20; answer alignment | [dataset](https://github.com/weaviate/IRPAPERS) and [experiment code](https://github.com/weaviate/query-agent-benchmarking) released | **High** | Small query set, questions derived from only 19 papers, and OCR uses a paid GPT-4.1 transcription path |
| [MIRACL-VISION](https://arxiv.org/abs/2505.11651) | Full-corpus multilingual visual retrieval over rendered Wikipedia articles; hard corpus is formed by removing easy negatives | 338,734 images and 7,898 human-origin questions in 18 languages; 95 GB | nDCG@10 / BEIR-style retrieval | [commercially usable dataset and example evaluator](https://huggingface.co/datasets/nvidia/miracl-vision) released | **High** | Images are rendered Wikipedia content rather than native enterprise PDFs; hard-negative filtering changes real corpus prevalence |
| [Invoice Haystack](https://arxiv.org/abs/2606.25343) | Find one semantically distinct invoice among highly visually homogeneous templates | 1,500 invoices, 200 validated QA pairs; controlled 500/1,000/1,500-document pools | Recall@1/3/5 and answer accuracy | paper links a project/data release; redistribution and exact runner still require local verification | **High when obtainable** | Single-page, single-domain, synthetic/anonymized fields; no long-document reasoning |
| [MultiDocR](https://arxiv.org/abs/2605.30027) | Multi-domain document retrieval with paraphrase robustness and non-binary relevance | 313 documents, 2,581 questions and mean 65 pages in its detailed statistics; the overview table instead says 2,441 and 12.2, an unresolved paper inconsistency | nDCG@10 and reranking metrics | paper published; no official code/data link was found in the paper on the audit date | **High in principle** | Labels are produced by a ColQwen/Jina/Gemini candidate-and-judge pipeline, so relevance outside the pooled top pages may be incomplete |
| [VisR-Bench](https://arxiv.org/abs/2508.07493) | Question-driven page retrieval inside multilingual, visually rich long documents; stratified by table/figure/text | 1,286 documents, about 18 pages/document, 16 languages; the paper inconsistently says over 35K QA in the abstract and 53K in the body | retrieval accuracy/Recall by evidence and language | [data and test code](https://github.com/puar-playground/VisR-Bench) released | Medium–high | Synthetic QA dominates; it primarily isolates within-document evidence pages rather than index construction |
| [ViDoSeek / ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG) | Large-corpus visual retrieval, iterative retrieval/reasoning, and answer generation | released dataset with query, unique answer, file and reference pages | retrieval and LLM-judged answer quality | official code, data, retrievers, and evaluator released | High | Main method entangles hybrid retrieval with an actor–critic multi-agent reader; cost accounting is not the benchmark's central contract |
| [M3DocVQA / M3DocRAG](https://github.com/bloomberg/m3docrag) | Open-domain DocVQA across many PDFs: index pages, retrieve top-k, then answer with a VLM | 3K+ PDFs and 40K+ pages | page retrieval plus answer accuracy/ANLS-style task metrics | official code and dataset construction instructions released | **High for end-to-end transfer** | Full ColPali is the default representation; the benchmark does not itself measure lifecycle or update behavior |
| [MMDocRAG](https://arxiv.org/abs/2505.16470) | Select multimodal evidence quotes and generate answers containing text and images | 222 documents, mean 67 pages, 4,055 expert QA; 2,107 cross-page, 1,590 multi-image, 2,503 cross-modal questions | quote selection plus answer correctness/completeness | [official code and data](https://github.com/MMDocRAG/MMDocRAG) released | Medium for indexing; **high for downstream value** | Each query receives 15/20 preselected quote candidates, so it does not by itself test corpus-wide first-stage retrieval |
| [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) | Given one long PDF, answer text/table/chart/image and cross-page questions | 135 documents, 1,091 questions, mean 47.5 pages and 21,214 tokens; 22.5% unanswerable | answer accuracy/F1 via official evaluator | documents, questions, metadata, code and leaderboard released | Medium | It mainly tests the reader after document selection, not a large shared index |
| [M-LongDoc](https://arxiv.org/abs/2411.06176) | Open-ended QA over super-long multimodal documents, with retrieval-aware model tuning | 851 samples; documents can contain hundreds of pages | answer correctness via automated evaluation | project reports public code, data, and models | Medium | Small number of evaluation samples; training/tuning result is the focus rather than physical retrieval cost |
| [MMLongBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9c8712a9e6d34d60edb8c4c980d4a0f2-Abstract-Datasets_and_Benchmarks_Track.html) | Long-context VLM robustness across Visual RAG, NIAH, many-shot ICL, summarization and DocVQA at controlled token lengths | 13,331 examples at 8K–128K equivalent tokens; 46 models evaluated | task-specific answer scores across length buckets | [official v1.1 evaluator](https://github.com/EdinburghNLP/MMLongBench) released | Low for the index; high for reader stress | The model receives a constructed long context.  Better scores do not prove cheaper or better corpus indexing |
| [EnterpriseRAG-Bench](https://arxiv.org/abs/2605.05253) | Enterprise retrieval/answering under semantic density, near duplicates, conflicts, missing answers and multi-document completeness | about 500K synthetic documents from 9 source types; 500 questions in 10 categories | answer correctness, fact completeness, Recall@10, invalid extra documents | [dataset, generator and evaluator](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) released | High for large text indexes; diagnostic for multimodal systems | Synthetic and text-centric.  Near-duplicate versions exist in one static snapshot; there is no arrival/update event stream |

### Benchmarks that should not be conflated

- **ViDoRe v3 and MIRACL-VISION** test corpus-wide ranking.  They are the
  cleanest primary retrieval benchmarks in this list.
- **MMDocIR and VisR-Bench** give fine-grained long-document localization but
  can hide global document selection.
- **MMDocRAG and MMLongBench-Doc** tell us whether retrieved evidence helps a
  reader; they are not sufficient evidence for index efficiency.
- **MMLongBench** is a long-context model benchmark.  It is useful only after
  ReprForge has produced the evidence context.
- **EnterpriseRAG-Bench** is valuable for 500K-scale distractors, conflicts,
  and near duplicates, but it cannot validate visual capacity allocation
  without adding a public multimodal subset.

## What the newest benchmarks add

### 1. Scale without pretending every negative is useful

[MIRACL-VISION](https://huggingface.co/datasets/nvidia/miracl-vision) is the
largest immediately usable visual retrieval corpus in the audit: 338,734 page
images.  It deliberately removes easy negatives to remain computationally
manageable.  This makes it a strong stress test for resident bytes, index
build work, and candidate-generation recall, but its reported corpus size must
not be described as an untouched production distribution.

[V-SPLADE](https://arxiv.org/abs/2605.30917), although a method rather than a
general benchmark, contributes a separate 18.7M-page scale experiment.  The
paper reports R@5/R@100 and latency, with a full sparse index reaching
0.228/0.396 and a two-stage sparse path reaching 0.183/0.278.  The same-scale
dense HNSW baseline reports 0.071/0.191.  The official repository still says
code will be released, so these are paper claims, not yet a runnable baseline.

### 2. A controlled reason to mix representations

[IRPAPERS](https://github.com/weaviate/IRPAPERS) is especially relevant to
ReprForge because it publishes both image and OCR text for every page and
shows complementary failures.  In the paper's open-source experiment, hybrid
text search reaches 46% Recall@1, image retrieval 43%, and multimodal fusion
49%; 22 questions succeed only in the text path at rank one and 18 only in the
image path.  Its current public leaderboard also includes BM25, dense text,
late interaction, MUVERA, and hybrid systems.  This directly tests whether a
heterogeneous representation policy preserves modality-specific evidence.

The paper also exposes the actual representation tax: its GPT-4.1
transcription averages 25 seconds and about $0.017 per page, while raw page
image conversion is deterministic and much cheaper but image embeddings are
far larger.  That is closer to ReprForge's build-versus-quality question than
a leaderboard that publishes only nDCG.

### 3. Hard negatives produced by representation collapse

[Invoice Haystack](https://arxiv.org/abs/2606.25343) keeps the document domain
and templates intentionally similar.  It reports mean within-corpus visual
cosine similarity 0.73, versus 0.38 for DocHaystack and 0.31 for
InfoHaystack.  At 1,500 invoices, BM25 reaches 38.5% Recall@1, V-RAG 40.0%,
and the proposed text+visual VL-RAG 50.0%.  This benchmark asks a sharper
question than “are charts visual?”: when visual embeddings collapse because
templates repeat, does the system preserve enough lexical identity to avoid
paying for an unhelpful visual tier?

### 4. Relevance labels that challenge lexical shortcuts

[MultiDocR](https://arxiv.org/abs/2605.30027) extends MMDocIR with seven query
types, low-overlap paraphrases (reported Jaccard 0.15), multiple relevant
pages, and five-level relevance.  It is a very good conceptual test for
candidate-relative fusion: if ReprForge's gain disappears after query
rephrasing, the current BM25 locator is a lexical shortcut rather than a
general representation compiler.  The dataset must remain “pending” until a
public artifact is located and its pooled relevance labels are inspected.

### 5. Realistic enterprise density, but not yet visual lifecycle

[EnterpriseRAG-Bench](https://arxiv.org/abs/2605.05253) provides approximately
500K linked internal documents with deliberate near duplicates, conflicting
facts, and missing-answer questions.  It reports local top-10 embedding
similarity 0.83, matching a private Onyx comparison corpus and exceeding the
0.69 open-web reference.  It is a strong future test for corpus scale and
version-aware evidence selection.  However, all versions are present in one
static snapshot.  Replaying filesystem order as time would manufacture a
lifecycle claim the benchmark does not support.

## Baseline and SOTA map

There is no single SOTA because the systems solve different layers.  The
minimum comparison should contain one representative from every row below.

| Baseline family | What it isolates | Public representative | What must be charged |
|---|---|---|---|
| Sparse text | Exact lexical evidence at very low serving cost | BM25; EnterpriseRAG's OpenSearch baseline | parsing/OCR if it is not supplied; index bytes and build time |
| Dense or late-interaction text | Semantic text locator | BGE-M3, Arctic 2.0, mxbai edge ColBERT | document encoding, ANN/late-interaction index and query encoding |
| Full visual single-vector | Uniform visual page representation | DSE/VisRAG-style visual encoder; Nemotron Embed VL | all-page image encoding and index construction |
| Full visual late interaction | Strong uniform high-capacity representation | ColPali/ColQwen; current Nemotron ColEmbed V2 or Argus | all patch-token vectors, MaxSim search and GPU memory/storage |
| Fixed compact visual | Same representation for every page at a lower token budget | [MURE](https://arxiv.org/abs/2603.13349), token pooling, MUVERA | compression/FDE construction, retained exact vectors for reranking, recall loss |
| Transient cascade | Cheap first stage; expensive visual work only for current candidates | [LightSTAR](https://arxiv.org/abs/2606.23539); CLIP/SigLIP + ColQwen cascade | first-stage corpus index plus repeated candidate refinement; no persistence credit |
| Static multimodal hybrid | Text and visual indexes coexist for every page | IRPAPERS hybrid; VL-RAG; ViDoRAG hybrid | both complete indexes, normalization/fusion and optional VLM verification |
| Progressive persistent index | Build visual state from workload and reuse it | ReprForge no-cache, resident, bounded-cache variants | cold-query wait, construction, resident growth, evictions and final quality |
| Oracle | Separates representation headroom from policy failure | full-score per-query or future-aware optimum | label as undeployable and report its extra information |

### Current method snapshots, not timeless rankings

- The [ViDoRe v3 paper](https://arxiv.org/abs/2601.08620) reports that visual
  retrievers outperform their textual bases, but a textual reranker adds 13.2
  nDCG@10 points and produces the strongest pipeline in that study.  This
  means a visual-only SOTA comparison is insufficient.
- [Nemotron ColEmbed V2](https://arxiv.org/abs/2602.03992) reported 63.42
  average nDCG@10 on ViDoRe v3 at its February 2026 snapshot.  The later
  [Argus-Retriever](https://arxiv.org/abs/2606.04300) paper reports 62.50 on
  the eight public v3 tasks for its 9B model and 64.80 when a 27B retrieval
  agent rewrites/decomposes queries.  The agent uses two H100s plus a separate
  H100 retriever, so it is a quality ceiling, not a 4xA100 efficiency
  baseline.
- The official [Qwen3-VL embedding/reranker
  repository](https://github.com/QwenLM/Qwen3-VL-Embedding) reports 52.9 for
  its 2B embedding and 60.8/66.7 for its 2B/8B rerankers on ViDoRe v3.  A
  reranker and an embedding model are not interchangeable index baselines.
- [LightSTAR](https://arxiv.org/abs/2606.23539) reports 89.1 average nDCG@5
  on the original eight-dataset ViDoRe suite and 123.9 seconds on 5,000 pages,
  versus 88.8/466.6 seconds for ColQwen2.5.  At 7,000 pages it reports 2.3x
  lower latency than ColPali.  Its [official repository](https://github.com/bokufa/LightSTAR)
  still contains only a README saying code and weights are forthcoming.
- [MURE](https://arxiv.org/abs/2603.13349) reports that a 512-token/page
  representation surpasses full-resource ColPali on ViDoRe v1/v2 while using
  50% of its visual token budget.  No public implementation was found in this
  audit.  It is a required conceptual baseline but not yet a runnable one.
- [IRPAPERS' current official
  leaderboard](https://github.com/weaviate/IRPAPERS) reports 61% Recall@1 for
  a text query-agent search, 59% for Mixedbread image retrieval, 58% for a
  closed-model multimodal hybrid, and 49% for the paper's open-source hybrid.
  These numbers are directly comparable only inside IRPAPERS.
- [V-SPLADE](https://arxiv.org/abs/2605.30917) is the strongest new warning
  against treating BM25 as the only cheap locator: it trains an image-derived
  sparse index and removes neural query encoding at serving time.  Its code is
  forthcoming, so a paper-level comparison can begin with its stated contract
  but cannot yet claim faithful reproduction.

## What public benchmarks do not currently prove

Across the visual-document benchmarks audited here, none supplies all of:

- natural document arrival and replacement events;
- natural ordered queries with timestamps and repeated users/topics;
- per-event ground truth for the index version visible at that time;
- end-to-end build, update, search, and answer costs.

ViDoRe's query list, MMDocIR's document order, and EnterpriseRAG's directory
order are dataset serialization, not workload traces.  Therefore:

1. public benchmarks can establish static quality, scale, build cost, and
   cold-stream time-to-quality;
2. shuffled or clustered replay can test sensitivity, but must be called a
   synthetic workload;
3. claims about workload drift, optimal admission, or incremental maintenance
   require either a separate public temporal trace or a newly documented
   benchmark contribution.

This is a benchmark gap, not automatic novelty for ReprForge.

## Implications for ReprForge

The benchmark audit changes the immediate evaluation target in four ways.

1. **IRPAPERS is the best next transfer.**  It is small enough for one A100,
   exposes image and text representations, includes strong open baselines,
   and directly demonstrates modality-complementary failures.
2. **Invoice Haystack is the cleanest mechanism stress test.**  It can reveal
   whether the compiler correctly avoids visual over-allocation when template
   similarity makes visual state weak.
3. **MIRACL-VISION is the scale test, not the first correctness test.**  Start
   with English plus two non-English splits and only then run all 95 GB.
4. **MMDocRAG or M3DocVQA is needed before an end-to-end paper claim.**
   Retrieval nDCG alone cannot show that the saved representation preserves
   the evidence an answer model actually uses.

The current ViDoRe HR and Finance results remain valid proof that progressive
construction can beat one full-visual operating point.  They are not a SOTA
claim: the model is an older ColPali checkpoint, the resident set still grows
to 63–81% of the corpus, and the benchmark lacks natural temporal reuse.
