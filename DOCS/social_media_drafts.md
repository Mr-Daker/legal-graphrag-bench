# Social Media Post Drafts

## Twitter / X (280 chars)

> Benchmarked LLM-only vs Basic RAG vs TigerGraph GraphRAG on 478 legal opinions.
>
> GraphRAG: −38.7% tokens, −67.9% latency vs Basic RAG ✅
> Basic RAG: 80% judge pass rate (best accuracy)
> GraphRAG: highest BERTScore semantic similarity
>
> Code + data: github.com/Mr-Daker/legal-graphrag-bench
> #GraphRAG #TigerGraph #LLM #RAG #hackathon

---

## LinkedIn (longer form)

**GraphRAG vs RAG vs LLM-only — here's what the numbers actually say**

I just wrapped a hackathon benchmark comparing three approaches to question-answering over a legal corpus (478 federal and state court opinions, 2.2M tokens):

**Results on 20 benchmark questions:**

🏎️ **Efficiency winner: GraphRAG**
- 38.7% fewer tokens per query than Basic RAG
- 67.9% lower latency (10.4s vs 32.5s avg)
- Graph traversal surfaces focused, structured context — not a shotgun similarity search

🎯 **Accuracy winner: Basic RAG (by judge score)**
- 80% PASS rate from LLM-as-a-Judge
- Very effective at retrieving the exact passage that answers factual questions

📊 **Semantic quality winner: GraphRAG (by BERTScore)**
- Highest F1 on both raw and rescaled BERTScore
- Richer, more semantically complete answers — even when missing specific details

**The interesting tension:** Basic RAG is better at getting specific facts exactly right. GraphRAG is better at producing answers that are semantically aligned with the full reference — which matters more for synthesis and multi-hop questions.

Three engineering lessons that surprised me:
1. Judge prompt calibration matters as much as your pipeline design
2. DeBERTa's tokenizer has a `model_max_length=1e30` that crashes bert_score on Windows — needs a patch
3. 6-model Gemini rotation gives you ~120 free-tier calls/day without hitting quota

Full write-up + code: https://github.com/Mr-Daker/legal-graphrag-bench

#GraphRAG #TigerGraph #LegalTech #RAG #LLM #NLP #MachineLearning #hackathon

---

## Dev.to / Hashnode teaser

**Title:** GraphRAG cut my token usage by 38% — here's the benchmark

**Subtitle / hook:** I ran LLM-only, Basic RAG, and TigerGraph GraphRAG against the same 478 legal opinions and 20 benchmark questions. GraphRAG won on efficiency and semantic quality. Basic RAG won on exact-fact retrieval. Here's the breakdown — including a Rust tokenizer overflow bug in bert_score that took me a while to track down.

**Tags:** graphrag, rag, llm, tigergraph, nlp, python, machinelearning
