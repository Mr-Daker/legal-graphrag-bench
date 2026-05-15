# Social Media Post Drafts

## Twitter / X

Benchmarked LLM-only vs Basic RAG vs TigerGraph GraphRAG on 478 legal opinions.

GraphRAG final run:
- 100% judge pass rate
- BERTScore raw 0.9003
- BERTScore rescaled 0.7013
- 36.61% fewer tokens vs Basic RAG

Built on the official TigerGraph GraphRAG repo:
github.com/tigergraph/graphrag

#GraphRAG #TigerGraph #RAG #LegalTech #LLM

---

## LinkedIn

**Legal GraphRAG hit all four hackathon bonuses.**

I benchmarked three approaches over 478 legal opinions from `harvard-lil/cold-cases`:

- LLM-only baseline
- Basic RAG over 7,008 chunks
- TigerGraph GraphRAG with a legal customization layer

Final results on 20 benchmark questions:

| Pipeline | Judge | BERT raw | BERT rescaled | Avg tokens | Avg latency |
|---|---:|---:|---:|---:|---:|
| LLM-only | 35% | 0.8166 | 0.4507 | 143 | 1,326 ms |
| Basic RAG | 55% | 0.8337 | 0.5018 | 3,747 | 1,318 ms |
| GraphRAG | 100% | 0.9003 | 0.7013 | 2,375 | 1,312 ms |

GraphRAG reduced token use by 36.61% vs Basic RAG and met every bonus threshold: judge accuracy, raw BERTScore, rescaled BERTScore, and token reduction.

The final system is built on the official TigerGraph GraphRAG repository at commit `f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9`, with a LegalGraphRAG layer for:

- EA-GraphRAG-style routing
- CommunityReport retrieval for global synthesis
- PathRAG-light entity paths
- relevance x edge_weight x hop_penalty scoring

The biggest lesson: graph retrieval only helps when it is routed and pruned. Once global questions used community summaries and multi-hop questions used entity paths, quality and token efficiency moved together.

#GraphRAG #TigerGraph #LegalTech #RAG #LLM #MachineLearning

---

## Dev.to / Hashnode Teaser

**Title:** Legal GraphRAG hit 100% judge accuracy and cut tokens by 36.61%

**Subtitle:** I built a benchmark over 478 court opinions comparing LLM-only, Basic RAG, and TigerGraph GraphRAG. The final v9 system is built on the official TigerGraph GraphRAG repo, adds EA-style routing and PathRAG-light pruning, and meets all four hackathon bonuses.

**Tags:** graphrag, rag, tigergraph, llm, legaltech, python, react
