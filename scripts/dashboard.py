"""
dashboard.py — Streamlit comparison dashboard for the GraphRAG Inference Hackathon.

Run:
    streamlit run scripts/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure scripts/ is importable even when Streamlit changes cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# Change to repo root so relative data paths work regardless of where
# Streamlit was launched from.
os.chdir(_ROOT)

import streamlit as st

from gemini_client import DEFAULT_MODEL, generate_text
from basic_rag_gemini import (
    DEFAULT_CHUNKS,
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    SYSTEM_PROMPT as RAG_SYSTEM_PROMPT,
    answer_query as rag_answer_query,
    build_index,
    load_index,
)
from graphrag_tigergraph import (
    DEFAULT_TIGERGRAPH_DIR,
    DEFAULT_QUERY_NAME,
    DEFAULT_TOP_CASES,
    DEFAULT_CHUNKS_PER_CASE,
    DEFAULT_MAX_CONTEXT_TOKENS,
    answer_query as graphrag_answer,
)
from llm_only_baseline import SYSTEM_PROMPT as LLM_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphRAG Inference Benchmark",
    page_icon="🐯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Basic RAG index…")
def _load_rag_index():
    """Load (or build) the hashing index and return (embeddings, chunks, manifest)."""
    index_dir = DEFAULT_INDEX_DIR
    try:
        return load_index(index_dir)
    except FileNotFoundError:
        # Build index on first launch
        manifest = build_index(
            chunks_path=DEFAULT_CHUNKS,
            index_dir=index_dir,
            embedding_provider="hashing",
            embedding_model="hashing",
            batch_size=256,
            limit=None,
        )
        return load_index(index_dir)


@st.cache_data(show_spinner=False)
def _load_questions() -> list[dict]:
    path = _ROOT / "data" / "eval" / "questions_dev.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_llm_only(query: str, model: str) -> dict:
    result = generate_text(
        prompt=f"Question: {query}",
        system_instruction=LLM_SYSTEM_PROMPT,
        model=model,
        temperature=0.0,
    )
    return {
        "answer": result.answer,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "retrieved_chunks": 0,
    }


def _run_basic_rag(query: str, model: str, top_k: int) -> dict:
    # Ensure index is loaded (warm up cache)
    _load_rag_index()
    result = rag_answer_query(
        query=query,
        question_id="dashboard",
        index_dir=DEFAULT_INDEX_DIR,
        top_k=top_k,
        generation_model=model,
    )
    return {
        "answer": result["answer"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "latency_ms": result["latency_ms"],
        "retrieved_chunks": len(result.get("retrieved_context", [])),
    }


def _run_graphrag(query: str, model: str) -> dict:
    result = graphrag_answer(
        query=query,
        question_id="dashboard",
        tigergraph_dir=DEFAULT_TIGERGRAPH_DIR,
        query_name=DEFAULT_QUERY_NAME,
        top_cases=DEFAULT_TOP_CASES,
        chunks_per_case=DEFAULT_CHUNKS_PER_CASE,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        generation_model=model,
        allow_local_fallback=True,
    )
    return {
        "answer": result.get("answer", ""),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "latency_ms": result.get("latency_ms", 0),
        "retrieved_chunks": len(result.get("retrieved_context", [])),
        "context_source": list(result.get("context_sources", [])),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🐯 GraphRAG Inference Benchmark Dashboard")
st.caption("TigerGraph GraphRAG Hackathon — Three pipelines, one query, side-by-side metrics.")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox(
        "LLM Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
        index=0,
    )
    top_k = st.slider("Basic RAG top-k chunks", min_value=2, max_value=16, value=DEFAULT_TOP_K, step=1)
    st.divider()
    st.markdown("**Pipelines**")
    run_llm = st.checkbox("Pipeline 1 — LLM Only", value=True)
    run_rag = st.checkbox("Pipeline 2 — Basic RAG", value=True)
    run_graph = st.checkbox("Pipeline 3 — GraphRAG", value=True)
    st.divider()
    st.markdown("**Sample Questions**")
    questions = _load_questions()
    q_labels = ["(custom)"] + [f"{q['id']}: {q['question'][:60]}…" for q in questions]
    selected_q = st.selectbox("Load a dev question", q_labels, index=0)

# ── Query input ──
default_query = ""
if selected_q != "(custom)" and questions:
    idx = q_labels.index(selected_q) - 1
    default_query = questions[idx]["question"]

query = st.text_area(
    "Enter your question:",
    value=default_query,
    height=80,
    placeholder="e.g. What court decided State v. Howerton?",
)

run_button = st.button("▶ Run All Pipelines", type="primary", disabled=not query.strip())

# ── Results ──
if run_button and query.strip():
    pipelines_to_run = []
    if run_llm:
        pipelines_to_run.append(("LLM Only", "llm_only"))
    if run_rag:
        pipelines_to_run.append(("Basic RAG", "basic_rag"))
    if run_graph:
        pipelines_to_run.append(("GraphRAG", "graphrag"))

    if not pipelines_to_run:
        st.warning("Enable at least one pipeline in the sidebar.")
        st.stop()

    cols = st.columns(len(pipelines_to_run))
    results: dict[str, dict] = {}

    for col, (label, key) in zip(cols, pipelines_to_run):
        with col:
            with st.spinner(f"Running {label}…"):
                try:
                    t0 = time.perf_counter()
                    if key == "llm_only":
                        res = _run_llm_only(query, model)
                    elif key == "basic_rag":
                        res = _run_basic_rag(query, model, top_k)
                    else:
                        res = _run_graphrag(query, model)
                    res["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    results[key] = res
                    err = None
                except Exception as exc:
                    err = str(exc)

            st.subheader(label)
            if err:
                st.error(f"Error: {err}")
            else:
                st.markdown(res["answer"])
                st.metric("Total tokens", f"{res['total_tokens']:,}")
                c1, c2 = st.columns(2)
                c1.metric("Prompt tokens", f"{res['prompt_tokens']:,}")
                c2.metric("Completion tokens", f"{res['completion_tokens']:,}")
                st.metric("Latency", f"{res.get('wall_ms', res.get('latency_ms', 0)):.0f} ms")
                if res.get("retrieved_chunks"):
                    st.caption(f"Retrieved {res['retrieved_chunks']} chunks")
                if res.get("context_source"):
                    st.caption(f"Source: {', '.join(res['context_source'])}")

    # ── Token reduction summary ──
    if "basic_rag" in results and "graphrag" in results:
        base_tok = results["basic_rag"]["total_tokens"]
        graph_tok = results["graphrag"]["total_tokens"]
        if base_tok > 0:
            reduction = (base_tok - graph_tok) / base_tok * 100
            colour = "normal" if reduction > 0 else "inverse"
            st.divider()
            cols2 = st.columns(3)
            cols2[0].metric(
                "GraphRAG token reduction vs Basic RAG",
                f"{reduction:+.1f}%",
                delta_color=colour,
            )
            cols2[1].metric("Basic RAG tokens", f"{base_tok:,}")
            cols2[2].metric("GraphRAG tokens", f"{graph_tok:,}")

# ── Saved results viewer ──
st.divider()
with st.expander("📊 View Saved Benchmark Results", expanded=False):
    report_path = _ROOT / "data" / "reports" / "pipeline_comparison_report.json"
    acc_path = _ROOT / "data" / "reports" / "accuracy_report.json"

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.subheader("Pipeline Comparison (saved run)")
        summaries = report.get("summaries", {})
        rows = []
        for name, s in summaries.items():
            rows.append({
                "Pipeline": name,
                "Questions": s.get("questions"),
                "Avg Tokens": s.get("avg_total_tokens"),
                "Avg Prompt Tokens": s.get("avg_prompt_tokens"),
                "Avg Latency ms": s.get("avg_latency_ms"),
                "Heuristic Pass Rate": s.get("heuristic_pass_rate"),
            })
        st.dataframe(rows, use_container_width=True)
        comparisons = report.get("comparisons", {})
        if comparisons:
            for k, v in comparisons.items():
                st.metric(k.replace("_", " "), f"{v}%")
    else:
        st.info("Run `compare_pipelines.py` to generate the comparison report.")

    if acc_path.exists():
        acc = json.loads(acc_path.read_text(encoding="utf-8"))
        st.subheader("Accuracy Evaluation (saved run)")
        acc_rows = []
        for name, comp in acc.get("comparison", {}).items():
            acc_rows.append({
                "Pipeline": name,
                "Judge Pass Rate": comp.get("judge_pass_rate"),
                "Bonus Judge (≥90%)": "✅" if comp.get("bonus_judge_met") else "❌",
                "BERTScore F1 Raw": comp.get("bertscore_f1_raw"),
                "BERTScore F1 Rescaled": comp.get("bertscore_f1_rescaled"),
                "Bonus Rescaled (≥0.55)": "✅" if comp.get("bonus_bertscore_rescaled_met") else "❌",
            })
        if acc_rows:
            st.dataframe(acc_rows, use_container_width=True)
    else:
        st.info("Run `evaluate_accuracy.py` to generate the accuracy report.")

# ── Footer ──
st.divider()
st.caption(
    "Dataset: harvard-lil/cold-cases (478 opinions, 2.2M tokens) · "
    "Corpus: 7,008 chunks @ 384 tokens · "
    "Graph: TigerGraph LegalGraphRAG (478 cases, 530 citations)"
)
