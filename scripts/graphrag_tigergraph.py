from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import tiktoken
from dotenv import load_dotenv

from gemini_client import DEFAULT_MODEL, compute_cost, generate_text


DEFAULT_QUESTIONS = Path("data/eval/questions_dev.json")
DEFAULT_OUTPUT = Path("data/results/graphrag_results.jsonl")
DEFAULT_TIGERGRAPH_DIR = Path("data/tigergraph")
DEFAULT_QUERY_NAME = "graphrag_case_context"
DEFAULT_TOP_CASES = 2
DEFAULT_CHUNKS_PER_CASE = 5
DEFAULT_MAX_CONTEXT_TOKENS = 2200

SYSTEM_PROMPT = """Answer the question using only the graph context.
If the graph context is insufficient, say that the answer is not available in the graph context.
Be concise, but include relevant case names, courts, citations, statutes, and reasoning when present."""

STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "case",
    "cases",
    "cited",
    "court",
    "decided",
    "does",
    "from",
    "have",
    "into",
    "legal",
    "opinion",
    "opinions",
    "that",
    "the",
    "their",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class GraphContext:
    case: dict[str, Any]
    chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    source: str


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}", text.lower())


def query_terms(query: str) -> list[str]:
    return [term for term in tokenize(query) if term not in STOPWORDS and len(term) > 2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="|", escapechar="\\", quoting=csv.QUOTE_NONE))


def load_local_graph_tables(tigergraph_dir: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    cases = read_csv(tigergraph_dir / "legal_cases.csv")
    chunks_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(tigergraph_dir / "chunks.csv"):
        row["chunk_index"] = int(row.get("chunk_index") or 0)
        row["token_count"] = int(row.get("token_count") or 0)
        chunks_by_case.setdefault(row["source_id"], []).append(row)

    for chunks in chunks_by_case.values():
        chunks.sort(key=lambda item: item["chunk_index"])
    return cases, chunks_by_case


def score_case(query: str, terms: list[str], case: dict[str, str], chunks: list[dict[str, Any]]) -> float:
    if not terms:
        return 0.0

    case_text = " ".join(
        str(case.get(name, ""))
        for name in ("case_id", "case_name", "citations", "court", "decision_date")
    ).lower()
    chunk_text = " ".join(str(chunk.get("text", ""))[:1200] for chunk in chunks[:3]).lower()
    haystack = f"{case_text} {chunk_text}"

    score = sum(1.0 for term in terms if term in haystack)
    phrase = query.lower().strip()
    if phrase and phrase in haystack:
        score += 5.0

    quoted_names = re.findall(r"\b[A-Z][A-Za-z.']+\s+v\.\s+[A-Z][A-Za-z.']+", query)
    for name in quoted_names:
        if name.lower() in case_text:
            score += 8.0

    return score / max(1, len(terms))


def rank_cases(query: str, tigergraph_dir: Path, top_cases: int) -> list[dict[str, str]]:
    cases, chunks_by_case = load_local_graph_tables(tigergraph_dir)
    terms = query_terms(query)
    scored = [
        (score_case(query, terms, case, chunks_by_case.get(case["case_id"], [])), case)
        for case in cases
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [case for score, case in scored[:top_cases] if score > 0] or [cases[0]]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TigerGraphClient:
    def __init__(self, host: str, graph_name: str, token: str | None = None, secret: str | None = None):
        self.host = host.rstrip("/")
        self.graph_name = graph_name
        self.token = token
        self.secret = secret

    @classmethod
    def from_env(cls) -> "TigerGraphClient | None":
        load_dotenv()
        host = os.getenv("TG_HOST")
        graph_name = os.getenv("TG_GRAPH_NAME", "LegalGraphRAG")
        if not host:
            return None
        token = os.getenv("TG_TOKEN") or os.getenv("TG_AUTH_TOKEN") or os.getenv("TG_BEARER_TOKEN")
        secret = os.getenv("TG_SECRET")
        return cls(host=host, graph_name=graph_name, token=token, secret=secret)

    def ensure_token(self) -> str | None:
        if self.token or not self.secret:
            return self.token

        # TigerGraph Savanna (Cloud) uses POST /gsql/v1/tokens
        try:
            payload = self._request_json_post("/gsql/v1/tokens", {"secret": self.secret}, auth=False)
            token = find_token(payload)
            if token:
                self.token = token
                return token
        except RuntimeError:
            pass

        # Legacy on-prem endpoints
        params = parse.urlencode({"secret": self.secret})
        for path in (f"/requesttoken?{params}", f"/restpp/requesttoken?{params}"):
            try:
                payload = self._request_json("GET", path, auth=False)
                token = find_token(payload)
                if token:
                    self.token = token
                    return token
            except RuntimeError:
                continue
        return None

    def run_query(self, query_name: str, params: dict[str, Any]) -> Any:
        self.ensure_token()
        query = parse.urlencode(params)
        candidates = (
            f"/query/{self.graph_name}/{query_name}?{query}",
            f"/restpp/query/{self.graph_name}/{query_name}?{query}",
        )
        errors = []
        for path in candidates:
            try:
                return self._request_json("GET", path, auth=True)
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError("; ".join(errors))

    def _request_json(self, method: str, path: str, auth: bool) -> Any:
        headers = {"Accept": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = request.Request(f"{self.host}{path}", method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc

    def _request_json_post(self, path: str, body: dict[str, Any], auth: bool) -> Any:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8")
        req = request.Request(f"{self.host}{path}", data=data, method="POST", headers=headers)
        try:
            with request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {path} failed: HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"POST {path} failed: {exc.reason}") from exc


def find_token(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("token", "jwt", "authToken"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            token = find_token(value)
            if token:
                return token
    elif isinstance(payload, list):
        for item in payload:
            token = find_token(item)
            if token:
                return token
    return None


def result_items(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if not isinstance(payload, list):
        return []
    for item in payload:
        if isinstance(item, dict) and key in item and isinstance(item[key], list):
            return item[key]
    return []


def unwrap_vertex(vertex: dict[str, Any]) -> dict[str, Any]:
    attrs = vertex.get("attributes")
    if isinstance(attrs, dict):
        return attrs
    return vertex


def local_context(case: dict[str, str], tigergraph_dir: Path, chunks_per_case: int) -> GraphContext:
    _, chunks_by_case = load_local_graph_tables(tigergraph_dir)
    citations_by_case: dict[str, list[str]] = {}
    for row in read_csv(tigergraph_dir / "cites.csv"):
        citations_by_case.setdefault(row["case_id"], []).append(row["citation_id"])
    citation_text_by_id = {
        row["citation_id"]: row["citation_text"]
        for row in read_csv(tigergraph_dir / "citations.csv")
    }
    citation_ids = citations_by_case.get(case["case_id"], [])
    citations = [
        {"citation_id": cid, "citation_text": citation_text_by_id.get(cid, cid)}
        for cid in citation_ids
    ]
    return GraphContext(
        case=case,
        chunks=chunks_by_case.get(case["case_id"], [])[:chunks_per_case],
        citations=citations,
        source="local_csv_fallback",
    )


def graph_context(
    case: dict[str, str],
    client: TigerGraphClient | None,
    tigergraph_dir: Path,
    query_name: str,
    chunks_per_case: int,
    allow_local_fallback: bool,
) -> GraphContext:
    if client is None:
        if allow_local_fallback:
            return local_context(case, tigergraph_dir, chunks_per_case)
        raise RuntimeError("TG_HOST is missing. Add TigerGraph connection settings to .env.")

    try:
        payload = client.run_query(
            query_name=query_name,
            params={"case_id": case["case_id"], "chunk_limit": chunks_per_case},
        )
        seed_items = result_items(payload, "seed")
        chunk_items = result_items(payload, "chunks")
        cite_items = result_items(payload, "cites")
        graph_case = unwrap_vertex(seed_items[0]) if seed_items else case
        chunks = [unwrap_vertex(item) for item in chunk_items]
        citations = [unwrap_vertex(item) for item in cite_items]
        chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))
        return GraphContext(case=graph_case, chunks=chunks, citations=citations, source="tigergraph")
    except RuntimeError:
        if allow_local_fallback:
            return local_context(case, tigergraph_dir, chunks_per_case)
        raise


def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def build_context_block(contexts: list[GraphContext], max_context_tokens: int) -> tuple[str, list[dict[str, Any]]]:
    blocks = []
    used = 0
    retrieved = []

    for context in contexts:
        case = context.case
        citation_text = "; ".join(
            str(citation.get("citation_text") or citation.get("citation_id") or "")
            for citation in context.citations
        )
        header = (
            f"Case: {case.get('case_name') or case.get('case_id')}\n"
            f"Case ID: {case.get('case_id')}\n"
            f"Citations: {case.get('citations') or citation_text}\n"
        )
        for chunk in context.chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            block = (
                f"{header}"
                f"Chunk: {chunk.get('chunk_index')} | token_count={chunk.get('token_count')}\n"
                f"{text}"
            )
            block_tokens = count_tokens(block)
            if used + block_tokens > max_context_tokens and blocks:
                continue
            blocks.append(block)
            used += block_tokens
            retrieved.append(
                {
                    "source": context.source,
                    "case_id": case.get("case_id"),
                    "case_name": case.get("case_name"),
                    "citations": case.get("citations") or citation_text,
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "token_count": chunk.get("token_count"),
                    "text": text,
                }
            )
            if used >= max_context_tokens:
                break

    return "\n\n---\n\n".join(blocks), retrieved


def answer_query(
    query: str,
    question_id: str,
    tigergraph_dir: Path,
    query_name: str,
    top_cases: int,
    chunks_per_case: int,
    max_context_tokens: int,
    generation_model: str,
    allow_local_fallback: bool,
    context_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    cases = rank_cases(query=query, tigergraph_dir=tigergraph_dir, top_cases=top_cases)
    client = TigerGraphClient.from_env()
    contexts = [
        graph_context(
            case=case,
            client=client,
            tigergraph_dir=tigergraph_dir,
            query_name=query_name,
            chunks_per_case=chunks_per_case,
            allow_local_fallback=allow_local_fallback,
        )
        for case in cases
    ]
    context_text, retrieved = build_context_block(contexts, max_context_tokens=max_context_tokens)
    if context_only:
        return {
            "pipeline": "graphrag",
            "question_id": question_id,
            "question": query,
            "graph_query": query_name,
            "top_cases": top_cases,
            "chunks_per_case": chunks_per_case,
            "max_context_tokens": max_context_tokens,
            "prompt_tokens": count_tokens(context_text),
            "completion_tokens": 0,
            "total_tokens": count_tokens(context_text),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "context_sources": sorted({item.get("source", "unknown") for item in retrieved}),
            "retrieved_context": retrieved,
            "context_preview": context_text[:2000],
        }
    prompt = f"Graph context:\n{context_text}\n\nQuestion: {query}"
    result = generate_text(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        model=generation_model,
        temperature=0.0,
    )
    end_to_end_ms = (time.perf_counter() - started) * 1000

    return {
        "pipeline": "graphrag",
        "question_id": question_id,
        "question": query,
        "answer": result.answer,
        "generation_model": result.model,
        "graph_query": query_name,
        "top_cases": top_cases,
        "chunks_per_case": chunks_per_case,
        "max_context_tokens": max_context_tokens,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": round(end_to_end_ms, 2),
        "generation_latency_ms": round(result.latency_ms, 2),
        "cost_usd": compute_cost(result.model, result.prompt_tokens, result.completion_tokens),
        "context_sources": sorted({item.get("source", "unknown") for item in retrieved}),
        "retrieved_context": retrieved,
    }


def run_questions(
    questions_path: Path,
    output_path: Path,
    tigergraph_dir: Path,
    query_name: str,
    top_cases: int,
    chunks_per_case: int,
    max_context_tokens: int,
    generation_model: str,
    allow_local_fallback: bool,
    context_only: bool,
) -> list[dict[str, Any]]:
    questions = load_questions(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(questions, start=1):
            question_id = str(item.get("id") or f"q{index:03d}")
            record = answer_query(
                query=item["question"],
                question_id=question_id,
                tigergraph_dir=tigergraph_dir,
                query_name=query_name,
                top_cases=top_cases,
                chunks_per_case=chunks_per_case,
                max_context_tokens=max_context_tokens,
                generation_model=generation_model,
                allow_local_fallback=allow_local_fallback,
                context_only=context_only,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            print(f"{question_id}: {record['total_tokens']} tokens, {record['latency_ms']} ms")
            time.sleep(5)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pipeline 3: TigerGraph GraphRAG baseline.")
    parser.add_argument("--query", help="Run one ad hoc question.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tigergraph-dir", type=Path, default=DEFAULT_TIGERGRAPH_DIR)
    parser.add_argument("--graph-query", default=DEFAULT_QUERY_NAME)
    parser.add_argument("--top-cases", type=int, default=DEFAULT_TOP_CASES)
    parser.add_argument("--chunks-per-case", type=int, default=DEFAULT_CHUNKS_PER_CASE)
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Retrieve and print graph context without calling Gemini.",
    )
    parser.add_argument(
        "--no-local-fallback",
        action="store_true",
        help="Fail instead of using local CSV graph tables when TigerGraph REST is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allow_local_fallback = not args.no_local_fallback and env_bool("TG_ALLOW_LOCAL_FALLBACK", True)
    if args.query:
        record = answer_query(
            query=args.query,
            question_id="adhoc",
            tigergraph_dir=args.tigergraph_dir,
            query_name=args.graph_query,
            top_cases=args.top_cases,
            chunks_per_case=args.chunks_per_case,
            max_context_tokens=args.max_context_tokens,
            generation_model=args.model,
            allow_local_fallback=allow_local_fallback,
            context_only=args.context_only,
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    run_questions(
        questions_path=args.questions,
        output_path=args.output,
        tigergraph_dir=args.tigergraph_dir,
        query_name=args.graph_query,
        top_cases=args.top_cases,
        chunks_per_case=args.chunks_per_case,
        max_context_tokens=args.max_context_tokens,
        generation_model=args.model,
        allow_local_fallback=allow_local_fallback,
        context_only=args.context_only,
    )


if __name__ == "__main__":
    main()
