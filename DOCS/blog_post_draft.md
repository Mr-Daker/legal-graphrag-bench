# Building LegalGraphRAG on TigerGraph: A Technical Write-Up

## Summary

This project benchmarks three legal question-answering pipelines on a 2.2M-token corpus of court opinions:

1. LLM-only baseline
2. Basic RAG over flat chunks
3. GraphRAG built on the official TigerGraph GraphRAG repository

The final GraphRAG system, LegalGraphRAG v9, met all four hackathon bonus thresholds:

| Metric | Target | GraphRAG v9 |
|---|---:|---:|
| LLM-as-a-Judge pass rate | >=90% | 100% |
| BERTScore raw | >=0.88 | 0.9003 |
| BERTScore rescaled | >=0.55 | 0.7013 |
| Token reduction vs Basic RAG | >=30% | 36.61% |

The key lesson was simple: graph retrieval only helps when it is routed and pruned. A graph can over-retrieve just as easily as a vector index can. The final system improved quality by routing each query type to the right retrieval mode and keeping the final context compact.

## Official TigerGraph GraphRAG Base

The implementation is built on top of the official TigerGraph GraphRAG repository:

```text
upstream: https://github.com/tigergraph/graphrag
commit  : f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9
vendored: vendor/tigergraph-graphrag
```

The official repo provides the GraphRAG service foundation, graph/vector retrieval architecture, GSQL retrieval queries, community summarization reference flow, and UI/service layout. This project layers a hackathon-specific legal corpus adapter, legal graph schema, EA-GraphRAG-style router, CommunityReport retrieval, PathRAG-light pruning, Gemini evaluation harness, and React/Node dashboard on top of that base.

Every final GraphRAG output row includes an `official_graphrag_base` metadata object with the upstream repo and commit. The final comparison report verifies that `20/20` GraphRAG benchmark rows carry this metadata.

## Corpus and Benchmark

The corpus comes from `harvard-lil/cold-cases` on HuggingFace:

- 478 legal opinions
- 2,200,374 total tokens
- 7,008 chunks
- 384-token chunk size
- 64-token overlap

The evaluation set contains 20 questions:

- 7 local factual questions
- 7 global synthesis questions
- 6 multi-hop questions

Examples:

- Local factual: "What court decided State v. Howerton?"
- Global synthesis: "What constitutional rights are most frequently raised in criminal appeal cases?"
- Multi-hop: "How do federal circuit courts approach constitutional questions differently from state appellate courts?"

## Pipeline 1: LLM-Only Baseline

The LLM-only baseline sends the question directly to Gemini 2.5 Flash Lite with no retrieval.

This pipeline is cheap and fast, but it has no corpus grounding. It can answer broad legal questions from general knowledge, but it struggles with exact citations, courts, and corpus-specific patterns.

Final performance:

```text
Judge pass rate : 35%
BERT raw        : 0.8166
BERT rescaled   : 0.4507
Avg tokens      : 143.30
Avg latency     : 1,326.32 ms
```

## Pipeline 2: Basic RAG

The Basic RAG pipeline indexes the 7,008 chunks and retrieves the top 8 chunks for each query. The retrieved text is passed to Gemini as flat context.

This works well for local factual questions because the answer often appears directly in a chunk. It is weaker for global synthesis and multi-hop questions because flat chunks do not represent relationships between cases, citations, entities, and legal concepts.

Final performance:

```text
Judge pass rate : 55%
BERT raw        : 0.8337
BERT rescaled   : 0.5018
Avg tokens      : 3,746.85
Avg latency     : 1,318.00 ms
```

## Pipeline 3: LegalGraphRAG

LegalGraphRAG extends the TigerGraph GraphRAG base with a legal schema and query-specific retrieval routes.

### Graph Schema

The final graph uses these vertex types:

- `LegalCase`
- `Chunk`
- `Citation`
- `Entity`
- `CommunityReport`

And these edge types:

- `HAS_CHUNK`
- `NEXT_CHUNK`
- `CITES`
- `MENTIONS`
- `RELATED_TO`

The first version of the graph used only cases, chunks, and citations. That helped with local navigation, but not enough for semantic legal reasoning. The later addition of `Entity`, `MENTIONS`, `RELATED_TO`, and `CommunityReport` made the graph useful for retrieval rather than just storage.

### Query Routing

The final system uses EA-GraphRAG-style routing:

```text
Query
  -> classify as local, global, multi-hop, or fallback
  -> choose retrieval route
  -> assemble compact context
  -> Gemini synthesis
```

Global synthesis questions trigger community report retrieval. These are questions with signals such as:

```text
common, across, corpus, themes, typically, patterns, types, most frequent
```

Local and multi-hop questions use entity/path retrieval. This keeps factual and relational questions on a more precise path-based route instead of sending every query through a large global summary.

### CommunityReport Retrieval

Global synthesis was the weakest part of earlier runs. Path traversal alone performed well on local facts, but it did not summarize the whole corpus.

The fix was to generate and retrieve `CommunityReport` vertices. These reports provide compact topic-level context for corpus-wide questions. Once global questions were routed to community summaries, GraphRAG improved sharply on synthesis questions.

### PathRAG-Light Retrieval

For local and multi-hop questions, the system retrieves entity-centered paths and scores them using:

```text
score(path) = relevance x edge_weight x hop_penalty
```

Where:

- `relevance` measures query/path semantic alignment
- `edge_weight` rewards stronger graph relationships
- `hop_penalty` prefers shorter, cleaner reasoning paths

The highest-scoring paths are selected greedily under a fixed token budget. This prevents GraphRAG from turning into a large chunk dump.

Final performance:

```text
Judge pass rate : 100%
BERT raw        : 0.9003
BERT rescaled   : 0.7013
Avg tokens      : 2,375.10
Avg latency     : 1,312.45 ms
```

## Final Comparison

| Pipeline | Judge | BERT raw | BERT rescaled | Avg tokens | Avg latency |
|---|---:|---:|---:|---:|---:|
| LLM-only | 35% | 0.8166 | 0.4507 | 143.30 | 1,326.32 ms |
| Basic RAG | 55% | 0.8337 | 0.5018 | 3,746.85 | 1,318.00 ms |
| GraphRAG | 100% | 0.9003 | 0.7013 | 2,375.10 | 1,312.45 ms |

GraphRAG reduced average token usage by:

```text
(3,746.85 - 2,375.10) / 3,746.85 = 36.61%
```

It also slightly beat Basic RAG latency in the final local-runtime benchmark:

```text
Basic RAG latency : 1,318.00 ms
GraphRAG latency  : 1,312.45 ms
Reduction         : 0.42%
```

## Per-Question-Type Results

| Question type | Count | LLM-only | Basic RAG | GraphRAG |
|---|---:|---:|---:|---:|
| Local factual | 7 | 0/7 | 7/7 | 7/7 |
| Global synthesis | 7 | 4/7 | 2/7 | 7/7 |
| Multi-hop | 6 | 3/6 | 2/6 | 6/6 |

The most important shift was global synthesis. Earlier GraphRAG versions were good at finding nearby chunks but weak at corpus-level synthesis. CommunityReport routing fixed that failure mode.

## Iteration History

| Run | Judge | BERT raw | Token reduction | Key change |
|---|---:|---:|---:|---|
| v1 | 35% | 0.8391 | n/a | Baseline |
| v4 | 45% | 0.6927 | 37.18% | TigerGraph Savanna API path |
| v6 | 55% | 0.8285 | 36.92% | PathRAG-light entity paths |
| v7 | 70% | 0.8373 | 36.97% | CommunityReport routing |
| v8 | 90% | 0.8750 | 36.77% | Optimized synthesis |
| v9 | 100% | 0.9003 | 36.61% | All bonuses met |

The project did not improve by adding more context. It improved by removing the wrong context.

## Implementation Notes

### Reproducibility

The benchmark can be run with:

```powershell
.\scripts\run_all.ps1 -SkipDataset -SkipIndex
```

The final runner uses:

```text
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=
TG_FORCE_LOCAL=1
PYTHONIOENCODING=utf-8
```

`TG_FORCE_LOCAL=1` uses the exported TigerGraph CSV graph for reproducible local benchmark execution. The project still includes the TigerGraph Savanna path and schema/loading files, but local mode avoids API and workspace variance during demos.

### Report Files

Final results are stored in:

```text
data/reports/results.txt
data/reports/pipeline_comparison_report.md
data/reports/accuracy_report_final_distilbert.md
```

The React/Node dashboard reads the final benchmark numbers and also supports live single-question comparison through:

```text
frontend/
backend/
```

## What Worked

Three engineering choices made the biggest difference:

1. **Routing by query type**
   Simple factual, global synthesis, and multi-hop questions need different retrieval strategies.

2. **Community reports for global questions**
   Corpus-wide questions need corpus-wide summaries, not only local graph paths.

3. **Path pruning under a token budget**
   GraphRAG must stay compact. The graph is useful because it can choose better evidence, not because it can send more evidence.

## Conclusion

LegalGraphRAG outperformed both baselines on judge accuracy, semantic similarity, token usage, and final-run latency. The final system shows that GraphRAG is strongest when the graph is not treated as a larger retrieval bucket, but as a routing and structure layer.

The final result is not just "RAG plus a graph." It is a retrieval system that decides when to use local chunks, when to use entity paths, and when to use community-level summaries.

That is what moved the system from 45% judge accuracy in early GraphRAG runs to 100% in v9 while still reducing token use by 36.61%.
