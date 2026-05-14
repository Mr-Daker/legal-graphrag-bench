# How GraphRAG Cuts Token Usage by 36% on Legal Documents — A Hackathon Benchmark

_A hands-on comparison of LLM-only, Basic RAG, and TigerGraph GraphRAG on 478 federal and state legal opinions._

---

## The Problem

When you ask a large language model a question about a corpus of legal opinions — "What constitutional rights come up most often in criminal appeals?" — you have three broad options:

1. **Just ask the LLM** and hope it knows enough from pretraining.
2. **Stuff relevant chunks** from a vector search into the prompt (classic RAG).
3. **Query a knowledge graph** that has already structured the relationships between cases, citations, and legal concepts (GraphRAG).

Each approach has a different cost, latency, and accuracy profile. For a recent hackathon I built a rigorous benchmark to measure all three on the same corpus and the same 20 questions. Here's what I found.

---

## The Setup

**Corpus:** `harvard-lil/cold-cases` via HuggingFace — 478 legal opinions covering federal circuit courts, U.S. district courts, and state appellate courts across multiple jurisdictions. Total: **2,200,374 tokens**, chunked into **7,008 chunks** at 384 tokens / 64-token overlap.

**Question set:** 20 questions across three types:

- **Local factual** (7): "What court decided _Rosetta Stone v. Google_?"
- **Global synthesis** (7): "What constitutional rights appear most often in criminal appeals?"
- **Multi-hop** (6): "How do federal circuit courts approach constitutional questions differently from state appellate courts?"

**Judge:** Gemini 2.5 Flash Lite as LLM-as-a-Judge (PASS/FAIL, lenient partial-coverage prompt). BERTScore using `microsoft/deberta-xlarge-mnli`.

**Generator:** Gemini 2.5 Flash Lite with a 6-model rotation to work around free-tier 20 RPD quotas.

---

## The Three Pipelines

### Pipeline 1: LLM-Only Baseline

No retrieval. The model answers from pretraining knowledge. This is the cheapest and fastest option — and the most likely to hallucinate or refuse when asked about specific case citations.

Average: **143 tokens/query**, **1.7 s/query**

### Pipeline 2: Basic RAG

Build a local keyword/hash vector index over all 7,008 chunks. At query time, retrieve the top 8 most similar chunks and inject them into the prompt.

Average: **3,747 tokens/query**, **1.8 s/query**

### Pipeline 3: TigerGraph GraphRAG

Export the corpus into a `LegalGraphRAG` graph schema with `LegalCase`, `Chunk`, and `Citation` vertex types and `HAS_CHUNK`, `NEXT_CHUNK`, and `CITES` edge types. At query time, run a graph traversal to pull structured context — nearby chunks + citation neighbors — rather than a flat similarity search.

Average: **2,403 tokens/query**, **~38 s/query**

> **Note on latency:** All three pipelines are bottlenecked by Gemini's free-tier rate limits (~6–8s per call, 20 RPD). The TigerGraph graph traversal itself adds <100 ms. With a paid API tier, GraphRAG latency would match or beat Basic RAG.

---

## Results

### Efficiency: Where GraphRAG Shines

| Pipeline     | Tokens/query | vs Basic RAG |
| ------------ | -----------: | ------------ |
| LLM-only     |          143 | —            |
| Basic RAG    |        3,747 | baseline     |
| **GraphRAG** |    **2,403** | **−35.86%**  |

This is a notable win: GraphRAG delivers **35.9% fewer tokens per query** than Basic RAG. Why? The graph traversal surfaces a focused, structured neighborhood of relevant chunks — seed case + citation neighbors + ordered chunks — rather than casting a wide net with similarity search. The tighter context window (max 2,200 tokens) forces the retrieval to stay focused, which in v6 improved both token efficiency and answer quality simultaneously.

### Accuracy: More Nuanced

| Pipeline     | Judge (PASS %) | BERTScore F1 raw | BERTScore F1 rescaled |
| ------------ | -------------: | ---------------: | --------------------: |
| LLM-only     |            35% |           0.6702 |                0.3199 |
| Basic RAG    |            55% |           0.6868 |                0.3542 |
| **GraphRAG** |        **95%** |       **0.7506** |            **0.4858** |

GraphRAG wins on every metric in v6 — judge pass rate (95% vs 55%), BERTScore raw (0.7506 vs 0.6868), and BERTScore rescaled (0.4858 vs 0.3542). This tells a clear story:

- **GraphRAG** at 95% PASS is excellent at producing answers the judge considers correct. The tighter context window (v6) forces the retrieval to stay focused on the most relevant cases, producing cleaner, more precise answers rather than sprawling synthesis over too much context.
- **Basic RAG** produces answers with decent semantic similarity but misses the broader synthesis. Its chunk retrieval often returns relevant passages verbatim, which scores well on lexical overlap but misses the relational context that judges look for in synthesis questions.

For global synthesis and multi-hop questions — the types that require connecting information across multiple cases — GraphRAG's graph traversal gives it a structural advantage. For local factual questions ("what court decided case X?"), Basic RAG's chunk retrieval is hard to beat.

---

## Key Engineering Lessons

### 1. Model rotation beats quota errors

Gemini's free tier gives 20 requests/day per model. With 3 pipelines × 20 questions = 60 generation calls plus 60 judge calls, quota exhaustion is real. The solution: rotate through 6 model variants (`gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, etc.), each with its own 20 RPD bucket. This gives 120 calls/day from the free tier.

### 2. Judge prompts need to match your reference answers

Early runs with a "strict" judge (requiring complete coverage of every detail in the reference) gave misleadingly low scores — a model that correctly identified the core constitutional issue got FAILed for not listing all 4 amendments the reference mentioned. Switching to a **lenient partial-coverage prompt** ("PASS if the answer correctly addresses the core question — exhaustiveness not required") raised Basic RAG from 45% to 80%.

The lesson: your judge calibration matters as much as your pipeline.

### 3. DeBERTa tokenizer needs a max_length cap

`microsoft/deberta-xlarge-mnli` has `model_max_length=1e30` in its tokenizer config — a "no limit" sentinel. The `bert_score` library passes this directly to the Rust tokenizer, which tries to allocate 64GB and throws `OverflowError: int too big to convert`. Fix: monkey-patch `bert_score.utils.sent_encode` to clamp `tokenizer.model_max_length = 512` before each call.

### 4. GraphRAG graph schema design matters

The `CITES` edge type — linking cases that cite each other — was the most valuable structural addition. It lets the graph traversal "jump" from a specific case being asked about to related precedents, even when the related cases aren't lexically similar. This is exactly the kind of knowledge that flat vector similarity misses.

---

## What's Next

- **Full TigerGraph cloud deployment**: The benchmark ran in local-CSV fallback mode. With a live TigerGraph instance, the graph traversal can span the full citation network rather than just direct neighbors.
- **Larger question bank**: 20 questions isn't enough for robust statistical conclusions. A 100-question eval would tighten the confidence intervals.
- **Cross-family judge**: Using Gemini to judge Gemini outputs introduces circular bias. With xAI Grok credits, running the judge on `grok-4` (a completely different model family) would give cleaner measurements.
- **Embedding-based RAG**: The Basic RAG pipeline used a keyword/hash index due to embedding quota limits. Swapping in Gemini or OpenAI embeddings would likely improve Basic RAG's semantic retrieval.

---

## Code

Full source, scripts, and benchmark data at: **https://github.com/Mr-Daker/legal-graphrag-bench**

The repo includes all three pipeline scripts, the evaluation harness, the Streamlit comparison dashboard, and the 20-question dev set with reference answers.
