# How Legal GraphRAG Hit All Four Hackathon Bonuses

_A hands-on comparison of LLM-only, Basic RAG, and TigerGraph GraphRAG on 478 legal opinions._

---

## The Problem

When you ask a large language model a question about a corpus of legal opinions, you have three broad options:

1. Ask the LLM directly and rely on pretraining.
2. Retrieve chunks with a vector-style Basic RAG pipeline.
3. Query a graph that has already structured cases, chunks, citations, entities, and community summaries.

The hackathon challenge was to compare those approaches on the same legal corpus, while measuring accuracy, BERTScore, latency, cost, and token use.

---

## The Setup

**Corpus:** `harvard-lil/cold-cases` via HuggingFace: 478 legal opinions, 2,200,374 tokens, chunked into 7,008 chunks at 384 tokens with 64-token overlap.

**Question set:** 20 questions across three types:

- Local factual (7)
- Global synthesis (7)
- Multi-hop (6)

**Base:** Pipeline 3 is built on top of the official TigerGraph GraphRAG repository:

- Upstream: `https://github.com/tigergraph/graphrag`
- Commit: `f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9`
- Vendored path: `vendor/tigergraph-graphrag`

**Judge and scoring:** Gemini 2.5 Flash Lite for LLM-as-a-Judge, with final BERTScore standardized on `distilbert-base-uncased`.

---

## The Three Pipelines

### Pipeline 1: LLM-Only

No retrieval. The model answers directly from parametric memory.

Final average: **143.30 tokens/query**, **1,326.32 ms/query**.

### Pipeline 2: Basic RAG

Hashing-vector retrieval over all chunks, using top-8 chunks as flat context.

Final average: **3,746.85 tokens/query**, **1,318.00 ms/query**.

### Pipeline 3: TigerGraph GraphRAG + LegalGraphRAG Layer

The graph schema includes `LegalCase`, `Chunk`, `Citation`, `Entity`, and `CommunityReport` vertices with `HAS_CHUNK`, `NEXT_CHUNK`, `CITES`, `MENTIONS`, and `RELATED_TO` edges.

At query time, an EA-GraphRAG-style router sends global questions to community report retrieval, and local/multi-hop questions to PathRAG-light entity paths. Candidate paths are scored with:

```text
relevance x edge_weight x hop_penalty
```

The final context is pruned under a token budget before Gemini synthesis.

Final average: **2,375.10 tokens/query**, **1,312.45 ms/query**.

---

## Results

| Pipeline | Judge | BERT raw | BERT rescaled | Avg tokens | Avg latency |
|---|---:|---:|---:|---:|---:|
| LLM-only | 35% | 0.8166 | 0.4507 | 143.30 | 1,326.32 ms |
| Basic RAG | 55% | 0.8337 | 0.5018 | 3,746.85 | 1,318.00 ms |
| GraphRAG | **100%** | **0.9003** | **0.7013** | **2,375.10** | **1,312.45 ms** |

GraphRAG reduced token use by **36.61%** compared with Basic RAG and slightly beat Basic RAG latency in the current full run.

---

## Bonus Status

| Metric | Target | Result | Status |
|---|---:|---:|---|
| LLM-as-a-Judge pass rate | >=90% | 100% | Met |
| BERTScore raw | >=0.88 | 0.9003 | Met |
| BERTScore rescaled | >=0.55 | 0.7013 | Met |
| Token reduction vs Basic RAG | >=30% | 36.61% | Met |

---

## Engineering Lessons

### 1. Routing mattered more than simply adding graph context

Early GraphRAG versions over-retrieved chunks and hurt synthesis. The big lift came from routing simple, global, and multi-hop questions into different retrieval paths.

### 2. Community reports fixed global synthesis

Global questions like "what themes appear across the corpus?" need summary vertices, not just local graph paths. Adding `CommunityReport` retrieval moved global synthesis from weak performance to full coverage.

### 3. PathRAG-light controlled token cost

Scoring candidate paths by relevance, edge strength, and hop penalty kept context focused while preserving relational evidence.

### 4. Official TigerGraph GraphRAG metadata is carried at runtime

Every final GraphRAG output row includes `official_graphrag_base` metadata with the upstream repo, commit, and customization layer, so the benchmark can verify it is built on the official base.

---

## Code

The repository includes the React/Node demo, Python benchmark scripts, official TigerGraph GraphRAG vendor integration, evaluation harness, reports, and the 20-question dev set.
