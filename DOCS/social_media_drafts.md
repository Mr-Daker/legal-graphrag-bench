# Social Media Post Drafts

## Twitter / X (280 chars)

> Benchmarked LLM-only vs Basic RAG vs TigerGraph GraphRAG on 478 legal opinions.
>
> GraphRAG: 70% judge pass rate 🏆 (up from 45% in v4!)
> Basic RAG: highest BERTScore semantic similarity (0.6868)
> GraphRAG: −27% tokens vs Basic RAG
> TigerGraph graph traversal: <100 ms (latency dominated by Gemini free tier)
>
> Code + data: github.com/Mr-Daker/legal-graphrag-bench
> #GraphRAG #TigerGraph #LLM #RAG #hackathon

---

## LinkedIn (longer form)

**GraphRAG vs RAG vs LLM-only — here's what the numbers actually say**

I just wrapped a hackathon benchmark comparing three approaches to question-answering over a legal corpus (478 federal and state court opinions, 2.2M tokens):

**Results on 20 benchmark questions:**

📊 **Efficiency winner: GraphRAG**

- 27.0% fewer tokens per query than Basic RAG
- Token savings from structured graph traversal vs flat similarity search
- TigerGraph graph API adds <100 ms (latency bottleneck is Gemini free tier)

🎯 **Accuracy winner: Basic RAG (by judge score)**

- 55% PASS rate from LLM-as-a-Judge
- Very effective at retrieving the exact passage that answers factual questions

📊 **Semantic quality winner: GraphRAG (by BERTScore)**

- Highest F1 on both raw (0.6927) and rescaled (0.3665) BERTScore
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

**Title:** GraphRAG hit 70% judge accuracy — here's how query routing did it

**Subtitle / hook:** I ran LLM-only, Basic RAG, and TigerGraph GraphRAG against the same 478 legal opinions and 20 benchmark questions. GraphRAG v5 won on judge accuracy (70%, up from 45%) by routing global/multi-hop questions to fetch more cases. Basic RAG won on BERTScore. Here's the breakdown — including a Rust tokenizer overflow bug in bert_score that took me a while to track down.

**Tags:** graphrag, rag, llm, tigergraph, nlp, python, machinelearning
