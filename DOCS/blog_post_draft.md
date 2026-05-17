# LegalGraphRAG: Building a Graph-Based Legal RAG System on TigerGraph

## What This Project Is About

This project was built for the TigerGraph GraphRAG hackathon. The goal was to answer legal questions over a large corpus of court opinions and compare three different approaches:

1. Ask the LLM directly.
2. Use Basic RAG with retrieved text chunks.
3. Use GraphRAG with TigerGraph.

The final system is called **LegalGraphRAG**. It is built on top of the official TigerGraph GraphRAG repository and adds a legal graph schema, query routing, community summaries, path-based retrieval, and a benchmark dashboard.

The final result met all four hackathon bonus targets:

LegalGraphRAG passed the LLM-as-a-Judge evaluation with **100% accuracy**, beating the required **90%** target. For the final benchmark, I used the same judge prompt across all three pipelines with Gemini 2.5 Flash Lite, plus BERTScore for semantic similarity. LegalGraphRAG achieved a **BERTScore raw score of 0.9003**, above the required **0.88**, and a **BERTScore rescaled score of 0.7013**, above the required **0.55**. It also reduced token usage by **36.61% compared with Basic RAG**, clearing the required **30%** token-reduction target.

In simple terms: GraphRAG gave better answers than the other two pipelines while using far fewer tokens than Basic RAG.

## Why Legal Documents Are Hard for RAG

Legal documents are not just long. They are connected.

A court opinion may mention:

- the case being decided
- earlier cases cited as precedent
- statutes
- constitutional rights
- judges
- procedural history
- facts from the trial court
- appellate standards of review

Basic RAG can retrieve text chunks that look similar to the question. That works when the answer is directly written in one chunk. But many legal questions need more than one chunk. Some questions ask for patterns across the whole corpus. Some require following citation relationships or understanding how one legal issue connects to another.

That is where a graph becomes useful. A graph can store the relationships between cases, chunks, citations, entities, and topics. Instead of only asking, "Which text chunk looks similar?", the system can ask, "Which connected legal evidence should I use?"

## Official TigerGraph GraphRAG Foundation

The hackathon asked us to build on TigerGraph GraphRAG. This project uses the official TigerGraph GraphRAG repository as the base:

```text
Official repo : https://github.com/tigergraph/graphrag
Commit        : f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9
Vendored at   : vendor/tigergraph-graphrag
```

The official repo provides the GraphRAG foundation: service layout, graph/vector retrieval architecture, GSQL retrieval queries, and community summarization reference code.

LegalGraphRAG adds the project-specific layer on top:

- legal corpus preparation
- legal graph schema
- TigerGraph CSV export
- query routing
- CommunityReport retrieval
- PathRAG-light path scoring
- Gemini answer generation
- benchmark and evaluation scripts
- React/Node dashboard

Every final GraphRAG result includes `official_graphrag_base` metadata. This records the upstream TigerGraph repo and commit used by the project. The final report confirms that all 20 GraphRAG benchmark rows include this metadata.

## Dataset

The corpus comes from `harvard-lil/cold-cases` on HuggingFace.

Final corpus statistics:

The dataset contains **478 legal opinions** and **2,200,374 total tokens**. After preprocessing, it was split into **7,008 chunks**. Each chunk is **384 tokens**, with a **64-token overlap** between neighboring chunks.

The benchmark uses 20 questions:

The benchmark includes **7 local factual questions**, such as "What court decided State v. Howerton?" It also includes **7 global synthesis questions**, such as "What constitutional rights are most frequently raised in criminal appeal cases?" Finally, it includes **6 multi-hop questions**, such as "How do federal circuit courts approach constitutional questions differently from state appellate courts?"

These three types were important because they test different abilities:

- local factual questions test exact retrieval
- global synthesis questions test corpus-level understanding
- multi-hop questions test connected reasoning

## The Three Pipelines

### Pipeline 1: LLM-Only

The first pipeline sends the question directly to Gemini 2.5 Flash Lite.

There is no retrieval and no graph. The model answers from its own learned knowledge.

This is cheap and fast, but it is not grounded in the legal corpus. It can answer some general legal questions, but it often misses exact case names, courts, citations, and corpus-specific details.

Final result:

```text
Judge pass rate : 35%
BERT raw        : 0.8166
BERT rescaled   : 0.4507
Average tokens  : 143.30
Average latency : 1,326.32 ms
```

### Pipeline 2: Basic RAG

The second pipeline is a standard RAG baseline.

It does this:

1. Split all opinions into chunks.
2. Build a local retrieval index.
3. For each question, retrieve the top 8 chunks.
4. Put those chunks into the prompt.
5. Ask Gemini to answer using that context.

This pipeline is strong for local factual questions. If the answer appears directly in the retrieved chunks, Basic RAG works well.

But it struggles with questions that require broader synthesis. For example, a question like "What themes appear across the corpus?" cannot be answered reliably from only a few flat chunks.

Final result:

```text
Judge pass rate : 55%
BERT raw        : 0.8337
BERT rescaled   : 0.5018
Average tokens  : 3,746.85
Average latency : 1,318.00 ms
```

### Pipeline 3: LegalGraphRAG

The third pipeline is the main system.

Instead of treating the corpus as only a list of chunks, LegalGraphRAG stores the corpus as a graph. The graph connects cases, chunks, citations, legal entities, and topic summaries.

The final graph has these vertex types:

The `LegalCase` vertex represents a court opinion or case. The `Chunk` vertex represents a text chunk from a case. The `Citation` vertex stores legal citations found in the corpus. The `Entity` vertex stores extracted people, organizations, legal concepts, locations, statutes, and similar items. The `CommunityReport` vertex stores a summary of an important topic or graph community.

And these edge types:

The `HAS_CHUNK` edge connects a case to its chunks. The `NEXT_CHUNK` edge preserves chunk order inside a case. The `CITES` edge connects cases and citations. The `MENTIONS` edge connects chunks to entities they mention. The `RELATED_TO` edge connects related entities.

This graph structure lets the retrieval system use legal relationships instead of only text similarity.

Final result:

```text
Judge pass rate : 100%
BERT raw        : 0.9003
BERT rescaled   : 0.7013
Average tokens  : 2,375.10
Average latency : 1,312.45 ms
```

## How LegalGraphRAG Works

The most important part of the system is that it does not use the same retrieval strategy for every question.

Different questions need different retrieval behavior.

### Step 1: Route the Question

The query router decides what kind of question was asked.

Example routing rules:

If the query contains signals like **"what court"**, **"what citation"**, or a specific case name, the system treats it as a local factual question. If it contains words like **"across"**, **"common"**, **"themes"**, **"patterns"**, or **"most frequent"**, the system treats it as a global synthesis question. If it asks about relationships, influence, chains, or comparisons, the system treats it as a multi-hop question.

This is inspired by EA-GraphRAG-style routing. The reason is practical: forcing every question through the same GraphRAG path made earlier versions worse. Simple questions need direct evidence. Global questions need summaries. Multi-hop questions need graph paths.

### Step 2: Retrieve the Right Type of Context

For **local factual questions**, LegalGraphRAG retrieves the relevant case and nearby chunks.

For **global synthesis questions**, it retrieves `CommunityReport` summaries. These are short summaries of important themes and communities in the graph.

For **multi-hop questions**, it retrieves entity paths and citation-linked context.

This makes the context more focused.

### Step 3: Score Graph Paths

For entity/path retrieval, the system uses a lightweight PathRAG-style scoring formula:

```text
path_score = relevance x edge_weight x hop_penalty
```

This means:

- prefer paths that match the question
- prefer stronger graph relationships
- prefer shorter paths over noisy long paths

After scoring, the system selects the best paths under a token budget. This is important because GraphRAG can become expensive if it sends too much context to the LLM.

### Step 4: Generate the Final Answer

The selected graph context is sent to Gemini. The prompt asks for a concise answer grounded in the retrieved evidence.

This final answer is then evaluated using:

- LLM-as-a-Judge using the same grading prompt across all pipelines
- BERTScore raw
- BERTScore rescaled
- token count
- latency
- cost per query

## Final Benchmark Results

Here is the final comparison across all three pipelines:

The LLM-only baseline reached a **35% judge pass rate**, with **0.8166 BERTScore raw**, **0.4507 BERTScore rescaled**, **143.30 average tokens**, and **1,326.32 ms average latency**.

Basic RAG reached a **55% judge pass rate**, with **0.8337 BERTScore raw**, **0.5018 BERTScore rescaled**, **3,746.85 average tokens**, and **1,318.00 ms average latency**.

LegalGraphRAG reached a **100% judge pass rate**, with **0.9003 BERTScore raw**, **0.7013 BERTScore rescaled**, **2,375.10 average tokens**, and **1,312.45 ms average latency**.

Token reduction compared with Basic RAG:

```text
Basic RAG average tokens : 3,746.85
GraphRAG average tokens  : 2,375.10
Reduction                : 36.61%
```

This matters because token usage affects cost and speed. LegalGraphRAG gave better answers while sending much less text to the model.

The dashboard also reports cost per query for each pipeline using Gemini token pricing.

## Results by Question Type

The per-question-type breakdown shows why GraphRAG helped.

For **local factual questions**, LLM-only passed **0 out of 7**, Basic RAG passed **7 out of 7**, and LegalGraphRAG also passed **7 out of 7**.

For **global synthesis questions**, LLM-only passed **4 out of 7**, Basic RAG passed **2 out of 7**, and LegalGraphRAG passed **7 out of 7**.

For **multi-hop questions**, LLM-only passed **3 out of 6**, Basic RAG passed **2 out of 6**, and LegalGraphRAG passed **6 out of 6**.

Basic RAG was already strong on local factual questions. The big difference came from global synthesis and multi-hop questions.

The graph helped because those questions require relationships:

- which cases cite which ideas
- which legal concepts appear together
- which themes appear across the corpus
- how different courts reason about similar issues

## How the System Improved Over Time

The final result did not happen in one attempt. The project went through multiple versions.

In **v1**, the system started with the initial baseline. GraphRAG judge accuracy was **35%**, BERTScore raw was **0.8391**, and token reduction was not measured yet.

In **v4**, the system moved to the TigerGraph Savanna API path. GraphRAG judge accuracy rose to **45%**, BERTScore raw was **0.6927**, and token reduction was **37.18%**.

In **v6**, the system added PathRAG-light entity paths. GraphRAG judge accuracy rose to **55%**, BERTScore raw reached **0.8285**, and token reduction was **36.92%**.

In **v7**, the system added CommunityReport routing. GraphRAG judge accuracy rose to **70%**, BERTScore raw reached **0.8373**, and token reduction was **36.97%**.

In **v8**, the system improved prompts and answer synthesis. GraphRAG judge accuracy crossed the bonus line at **90%**, BERTScore raw reached **0.8750**, and token reduction was **36.77%**.

In **v9**, all bonuses were met. GraphRAG judge accuracy reached **100%**, BERTScore raw reached **0.9003**, and token reduction remained strong at **36.61%**.

The biggest lesson from these runs was that "more graph context" is not automatically better.

Earlier GraphRAG versions sometimes retrieved too much or retrieved the wrong kind of context. The final system improved by becoming more selective:

- local questions use local evidence
- global questions use community summaries
- multi-hop questions use graph paths
- all context is pruned before generation

## What Was Hard

### 1. GraphRAG Can Over-Retrieve

At first, the graph retrieved many chunks around a case. That reduced token count compared with Basic RAG, but the answers were not always better.

The fix was to score and prune paths instead of dumping graph neighborhoods into the prompt.

### 2. Global Questions Need Summaries

Questions about "common themes" or "patterns across the corpus" cannot be answered well from only a few chunks.

Community reports fixed this. They gave the model a higher-level view of the corpus.

### 3. Evaluation Had to Be Consistent

The final benchmark uses the same LLM-as-a-Judge prompt and the same BERTScore model across all three pipelines. This made the comparison fair.

Final scoring uses:

```text
Judge model     : Gemini 2.5 Flash Lite
BERTScore model : distilbert-base-uncased
```

## Reproducing the Benchmark

The final benchmark can be run with:

```powershell
.\scripts\run_all.ps1 -SkipDataset -SkipIndex
```

The final run uses:

```text
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=
TG_FORCE_LOCAL=1
PYTHONIOENCODING=utf-8
```

`TG_FORCE_LOCAL=1` means the benchmark uses the exported TigerGraph CSV graph locally. This keeps the demo reproducible and avoids cloud workspace variance.

Important output files:

```text
data/reports/results.txt
data/reports/pipeline_comparison_report.md
data/reports/accuracy_report_final_distilbert.md
```

The project also includes a React/Node dashboard:

```text
frontend/
backend/
```

The dashboard shows:

- live query results for all three pipelines
- token comparison
- cost per query using Gemini token pricing
- benchmark charts
- bonus thresholds
- architecture and version evolution timeline

## Final Takeaway

The final system worked because the graph was not used as just another storage layer.

It was used as a decision layer.

LegalGraphRAG decides:

- when to use direct case/chunk evidence
- when to use community summaries
- when to follow entity and citation paths
- how much context should be sent to the LLM

That is why the final system reached 100% judge accuracy, 0.9003 BERTScore raw, 0.7013 BERTScore rescaled, and 36.61% token reduction.

For this legal corpus, GraphRAG was not simply "RAG with a graph added." It was a better retrieval strategy: more structured, more selective, and more aligned with the way legal information is connected.
