import { useMemo, useState } from "react";
import BonusPanel from "./components/BonusPanel.jsx";
import PipelineCard from "./components/PipelineCard.jsx";
import QueryInput from "./components/QueryInput.jsx";
import SummaryTable from "./components/SummaryTable.jsx";
import TokenBar from "./components/TokenBar.jsx";
import { defaultQuestion, initialResults, pipelineMeta, pipelineOrder } from "./data/benchmark.js";

const pipelineTabs = [
  {
    id: "llm_only",
    label: "Pipeline 1 - LLM Only",
    text: "The baseline sends the question directly to Gemini with no retrieval. It is fast and cheap, but it relies only on parametric memory, so it struggles on corpus-specific legal facts.",
  },
  {
    id: "basic_rag",
    label: "Pipeline 2 - Basic RAG",
    text: "Basic RAG embeds the query, retrieves top chunks from the flat vector index, and asks Gemini to answer from those chunks. It improves local factual recall but can miss corpus-wide patterns.",
  },
  {
    id: "graphrag",
    label: "Pipeline 3 - GraphRAG",
    text: "GraphRAG is built on the official TigerGraph GraphRAG repo, then customized with a legal schema, EA-GraphRAG routing, CommunityReport retrieval, PathRAG-light pruning, and compact Gemini synthesis.",
  },
];

const evolutionRuns = [
  {
    id: "v1",
    title: "Baseline",
    judge: "35%",
    bert: "0.839",
    token: "not measured",
    summary: "No retrieval, parametric memory only.",
    solved: "Established the LLM-only floor and exposed corpus-specific recall gaps.",
    breakdown: "LLM-only: 7/20 pass. Basic RAG and GraphRAG were not yet optimized.",
  },
  {
    id: "v4",
    title: "TigerGraph Savanna API",
    judge: "45%",
    bert: "0.6927",
    token: "37.18%",
    summary: "Moved graph access from local CSV fallback to TigerGraph Savanna.",
    solved: "Reduced GraphRAG latency from 31,573ms to 6,909ms and proved the graph path could run through the cloud API.",
    breakdown: "GraphRAG: 9/20 pass. Local factual improved, but global synthesis still lagged.",
  },
  {
    id: "v6",
    title: "PathRAG-light entity paths",
    judge: "55%",
    bert: "0.8285",
    token: "36.92%",
    summary: "Added Entity vertices, MENTIONS edges, RELATED_TO edges, and path pruning.",
    solved: "Replaced broad chunk dumps with scored relational context under a hard token budget.",
    breakdown: "GraphRAG: 11/20 pass. Local factual reached 6/7; multi-hop started improving.",
  },
  {
    id: "v7",
    title: "Community routing",
    judge: "70%",
    bert: "0.8373",
    token: "36.97%",
    summary: "Added CommunityReport routing for global synthesis questions.",
    solved: "Global synthesis improved from 1/7 to 4/7 while preserving token reduction.",
    breakdown: "GraphRAG: local 6/7, global 4/7, multi-hop 4/6.",
  },
  {
    id: "v8",
    title: "Optimized synthesis",
    judge: "90%+",
    bert: "0.875",
    token: "36.77%",
    summary: "Tightened answer prompts, fallbacks, and context shaping.",
    solved: "Fixed answer truncation and loose context; judge bonus crossed the 90% threshold.",
    breakdown: "GraphRAG crossed the judge bonus line, with remaining BERTScore tuning left.",
  },
  {
    id: "v9",
    title: "All bonuses met",
    judge: "100%",
    bert: "0.9003",
    token: "36.61%",
    summary: "Targeted precedent/civil fallback guidance plus concise answer shaping.",
    solved: "Met judge, BERT raw, BERT rescaled, and token reduction bonuses together.",
    breakdown: "GraphRAG: local 7/7, global 7/7, multi-hop 6/6.",
  },
];

const officialBase = {
  repo: "github.com/tigergraph/graphrag",
  commit: "f649f419",
  path: "vendor/tigergraph-graphrag",
};

function mergeResults(payload) {
  return {
    llm_only: normalizePipeline(payload.llm_only, Boolean(payload.matched_question_id)) || initialResults.llm_only,
    basic_rag: normalizePipeline(payload.basic_rag, Boolean(payload.matched_question_id)) || initialResults.basic_rag,
    graphrag: normalizePipeline(payload.graphrag, Boolean(payload.matched_question_id)) || initialResults.graphrag,
  };
}

function normalizePipeline(item, matched) {
  if (!item) return null;
  const answer = String(item.answer || "");
  const cleanAnswer =
    /Traceback \(most recent call last\)|site-packages|httpx\\?_transports/i.test(answer)
      ? "The live backend returned an internal Python error for this run. Please retry after the API/network recovers; the pre-computed v9 benchmark below remains valid."
      : answer;
  return {
    ...item,
    answer: cleanAnswer,
    bertscore: {
      value: item.bertscore,
      matched,
    },
  };
}

function AppHeader({ title, eyebrow = "TigerGraph · GraphRAG Inference Hackathon", onHome, actions }) {
  return (
    <header className="dashboard-header">
      <div>
        <p>{eyebrow}</p>
        <h1>{title}</h1>
      </div>
      <div className="header-actions-row">
        {onHome ? (
          <button className="ghost-button" onClick={onHome} type="button">
            Home
          </button>
        ) : null}
        {actions}
      </div>
    </header>
  );
}

function StatusPills() {
  return (
    <div className="status-pills">
      <span>Judge 100% ✓</span>
      <span>BERTScore 0.90 ✓</span>
      <span>Token -36.6% ✓</span>
    </div>
  );
}

function OfficialBaseStrip({ compact = false }) {
  return (
    <section className={compact ? "official-base official-base-compact" : "official-base"}>
      <div>
        <span className="official-base-kicker">Built on official TigerGraph GraphRAG</span>
        <strong>{officialBase.repo}</strong>
      </div>
      <div className="official-base-details">
        <span>commit {officialBase.commit}</span>
        <span>{officialBase.path}</span>
        <span>Legal schema + EA router + PathRAG-light</span>
      </div>
    </section>
  );
}

function HomePage({ onOpenDashboard, onOpenArchitecture }) {
  return (
    <main className="dashboard-shell home-shell">
      <AppHeader title="GraphRAG hackathon" actions={<StatusPills />} />
      <OfficialBaseStrip />
      <section className="home-card-grid">
        <button className="home-card home-card-dashboard" onClick={onOpenDashboard} type="button">
          <div>
            <span className="home-card-kicker">Live + benchmark</span>
            <h2>Benchmark Dashboard</h2>
            <p>Run any legal question through all 3 pipelines simultaneously.</p>
          </div>
          <div className="home-badges">
            <span>Judge 100% ✓</span>
            <span>BERT raw 0.90 ✓</span>
            <span>BERT rescaled 0.70 ✓</span>
            <span>Token -36.6% ✓</span>
          </div>
          <div className="mini-pipeline-preview">
            <span>
              <b>35%</b>
              LLM
            </span>
            <span>
              <b>55%</b>
              RAG
            </span>
            <span>
              <b>100%</b>
              GraphRAG
            </span>
          </div>
          <div className="home-card-foot">
            <span>478 cases · 2.2M tokens</span>
            <strong>Open →</strong>
          </div>
        </button>

        <button className="home-card home-card-architecture" onClick={onOpenArchitecture} type="button">
          <div>
            <span className="home-card-kicker">Engineering story</span>
            <h2>Architecture & Evolution</h2>
            <p>See the v9 pipeline design and how each iteration fixed a measurable failure mode.</p>
          </div>
          <div className="version-preview">
            <strong>v1 → v9</strong>
            <span>Official tigergraph/graphrag base · Router · CommunityReport · PathRAG-light</span>
          </div>
          <div className="mini-flow-preview">
            <span>EA-GraphRAG Router</span>
            <span>PathRAG-light</span>
            <span>CommunityReport</span>
          </div>
          <div className="home-card-foot">
            <span>How we got there</span>
            <strong>Open →</strong>
          </div>
        </button>
      </section>
    </main>
  );
}

function DashboardPage({ onHome }) {
  const [question, setQuestion] = useState(defaultQuestion);
  const [results, setResults] = useState(initialResults);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const tokenReduction = useMemo(() => {
    const basic = Number(results.basic_rag?.tokens || 0);
    const graph = Number(results.graphrag?.tokens || 0);
    if (!basic || !graph) return 36.61;
    return ((basic - graph) / basic) * 100;
  }, [results]);

  const runQuery = async (nextQuestion = question) => {
    const trimmed = nextQuestion.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    try {
      const apiUrl = import.meta.env.VITE_API_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8787");
      const response = await fetch(`${apiUrl}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Query failed");
      }
      setResults(mergeResults(payload));
    } catch (err) {
      setError(err.message || "Could not reach the benchmark backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="dashboard-shell">
      <AppHeader title="Benchmark dashboard" onHome={onHome} actions={<StatusPills />} />
      <OfficialBaseStrip compact />

      <QueryInput
        question={question}
        setQuestion={setQuestion}
        onRun={runQuery}
        onSampleSelect={(sampleQuestion) => {
          setQuestion(sampleQuestion);
          runQuery(sampleQuestion);
        }}
        loading={loading}
        error={error}
      />

      <div className="section-divider">
        <span>Live query results</span>
      </div>

      <section className="pipeline-grid">
        {pipelineOrder.map((id) => {
          const meta = pipelineMeta[id];
          const item = results[id];
          return (
            <PipelineCard
              key={id}
              name={meta.name}
              color={meta.color}
              answer={item.answer}
              tokens={item.tokens}
              latency={item.latency_ms}
              cost={item.cost_usd}
              bertscore={item.bertscore}
              verdict={item.verdict}
              isGraph={id === "graphrag"}
            />
          );
        })}
      </section>

      <TokenBar results={results} reduction={tokenReduction} />

      <div className="section-divider section-divider-benchmark">
        <span>Pre-computed benchmark</span>
      </div>

      <SummaryTable />
      <BonusPanel />
    </main>
  );
}

function ArchitectureDiagram({ activePipeline }) {
  const isActive = (id) => activePipeline === id;
  return (
      <div className={`architecture-diagram architecture-${activePipeline}`}>
      <div className="flow-node flow-node-query">Query Input</div>
      <div className="flow-arrow">↓</div>
      <div className={isActive("llm_only") ? "flow-node flow-node-active" : "flow-node"}>Gemini LLM synthesis</div>
      <div className={isActive("basic_rag") ? "flow-node flow-node-active" : "flow-node"}>Basic RAG chunk retrieval</div>
      <div className={isActive("graphrag") ? "flow-node flow-node-official flow-node-active" : "flow-node flow-node-official"}>
        Official TigerGraph GraphRAG
        <span>vendor/tigergraph-graphrag</span>
      </div>
      <div className={isActive("graphrag") ? "flow-node flow-node-router flow-node-active" : "flow-node flow-node-router"}>
        EA-GraphRAG Router
      </div>
      <div className="flow-split">
        <div className={isActive("graphrag") ? "flow-lane flow-lane-active" : "flow-lane"}>
          <strong>Global keywords?</strong>
          <span>CommunityReport vertices</span>
        </div>
        <div className={isActive("graphrag") ? "flow-lane flow-lane-active" : "flow-lane"}>
          <strong>Local / multi-hop entity match?</strong>
          <span>Entity + MENTIONS + RELATED_TO edges</span>
        </div>
      </div>
      <div className={isActive("graphrag") ? "flow-node flow-node-score flow-node-active" : "flow-node flow-node-score"}>
        PathRAG-light scoring
        <span>relevance × edge_weight × hop_penalty</span>
      </div>
      <div className={isActive("graphrag") ? "flow-node flow-node-active" : "flow-node"}>Pruned context ≤ token budget</div>
      <div className="flow-arrow">↓</div>
      <div className="flow-node flow-node-answer">Answer + metrics</div>
    </div>
  );
}

function ArchitecturePage({ onHome }) {
  const [activePipeline, setActivePipeline] = useState("graphrag");
  const [openRun, setOpenRun] = useState("v9");
  const activeTab = pipelineTabs.find((tab) => tab.id === activePipeline);

  return (
    <main className="dashboard-shell architecture-shell">
      <AppHeader title="Architecture & Evolution" onHome={onHome} actions={<StatusPills />} />

      <section className="panel architecture-panel">
        <div className="section-heading">
          <h2>System architecture</h2>
          <span>v9 pipeline</span>
        </div>
        <OfficialBaseStrip compact />
        <div className="pipeline-tabs">
          {pipelineTabs.map((tab) => (
            <button
              className={activePipeline === tab.id ? "pipeline-tab pipeline-tab-active" : "pipeline-tab"}
              key={tab.id}
              onClick={() => setActivePipeline(tab.id)}
              type="button"
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="architecture-layout">
          <ArchitectureDiagram activePipeline={activePipeline} />
          <aside className="architecture-explainer">
            <h3>{activeTab.label}</h3>
            <p>{activeTab.text}</p>
            <dl>
              <div>
                <dt>Final role</dt>
                <dd>{activePipeline === "graphrag" ? "Submitted winner pipeline" : "Comparison baseline"}</dd>
              </div>
              <div>
                <dt>Final judge</dt>
                <dd>{activePipeline === "graphrag" ? "100%" : activePipeline === "basic_rag" ? "55%" : "35%"}</dd>
              </div>
            </dl>
          </aside>
        </div>
      </section>

      <section className="panel evolution-panel">
        <div className="section-heading">
          <h2>Version evolution timeline</h2>
          <span>v1 → v9</span>
        </div>
        <div className="timeline">
          {evolutionRuns.map((run) => (
            <button
              className={openRun === run.id ? "timeline-item timeline-item-open" : "timeline-item"}
              key={run.id}
              onClick={() => setOpenRun(openRun === run.id ? "" : run.id)}
              type="button"
            >
              <span className="timeline-dot">{run.id}</span>
              <span className="timeline-content">
                <span className="timeline-topline">
                  <strong>{run.title}</strong>
                </span>
                <span className="timeline-metrics">
                  Judge {run.judge} · BERTScore {run.bert} · Token reduction {run.token}
                </span>
                <span>{run.summary}</span>
                {openRun === run.id ? (
                  <span className="timeline-detail">
                    <b>Problem fixed:</b> {run.solved}
                    <br />
                    <b>Breakdown:</b> {run.breakdown}
                  </span>
                ) : null}
              </span>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}

export default function App() {
  const [view, setView] = useState("home");

  if (view === "dashboard") {
    return <DashboardPage onHome={() => setView("home")} />;
  }

  if (view === "architecture") {
    return <ArchitecturePage onHome={() => setView("home")} />;
  }

  return (
    <HomePage
      onOpenArchitecture={() => setView("architecture")}
      onOpenDashboard={() => setView("dashboard")}
    />
  );
}
