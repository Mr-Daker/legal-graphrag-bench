# Social Media Post Drafts

## Twitter / X (280 chars)

> Benchmarked LLM-only vs Basic RAG vs TigerGraph GraphRAG on 478 legal opinions.
>
> GraphRAG: **95% judge pass rate** 🏆 (bonus target met!)
> GraphRAG: highest BERTScore (0.7506 raw, 0.4858 rescaled)
> GraphRAG: −36% tokens vs Basic RAG (bonus target met!)
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

- **35.86%** fewer tokens per query than Basic RAG — hackathon bonus target (≥30%) met!
- Token savings from structured graph traversal vs flat similarity search
- TigerGraph graph API adds <100 ms (latency bottleneck is Gemini free tier)

🎯 **Accuracy winner: GraphRAG** (everywhere)

- **95%** PASS rate from LLM-as-a-Judge — hackathon bonus target (≥90%) met!
- Highest BERTScore raw (0.7506) and rescaled (0.4858)
- Tighter context window (max 2,200 tokens) improved both quality AND efficiency simultaneously

**The interesting tension:** Basic RAG is better at getting specific facts exactly right. GraphRAG is better at producing answers that are semantically aligned with the full reference — which matters more for synthesis and multi-hop questions.

Three engineering lessons that surprised me:

1. Judge prompt calibration matters as much as your pipeline design
2. DeBERTa's tokenizer has a `model_max_length=1e30` that crashes bert_score on Windows — needs a patch
3. 6-model Gemini rotation gives you ~120 free-tier calls/day without hitting quota

Full write-up + code: https://github.com/Mr-Daker/legal-graphrag-bench

#GraphRAG #TigerGraph #LegalTech #RAG #LLM #NLP #MachineLearning #hackathon

---

## Dev.to / Hashnode teaser

**Title:** GraphRAG hit 95% judge accuracy — here’s how tighter retrieval did it

**Subtitle / hook:** I ran LLM-only, Basic RAG, and TigerGraph GraphRAG against the same 478 legal opinions and 20 benchmark questions. GraphRAG v6 won on judge accuracy (95%, up from 70%) AND on BERTScore AND on token reduction by tightening the context window to 2,200 tokens. Both hackathon bonus targets (judge ≥90%, token reduction ≥30%) are now met. Here’s the breakdown — including a Rust tokenizer overflow bug in bert_score that took me a while to track down.

**Tags:** graphrag, rag, llm, tigergraph, nlp, python, machinelearning
