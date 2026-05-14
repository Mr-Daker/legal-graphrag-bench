from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
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
DEFAULT_FETCH_MULTIPLIER = 5

SYSTEM_PROMPT = """Answer the question using only the graph context.
If the graph context is insufficient, say that the answer is not available in the graph context.
Give a complete answer in 2-5 sentences.
When a case caption or metadata states a court, treat that as the deciding court unless later context clearly says otherwise.
For corpus-wide questions, synthesize the community reports into broad legal categories instead of listing only the single highest-frequency item.
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

GLOBAL_QUERY_MARKERS = {
    "across",
    "common",
    "corpus",
    "frequent",
    "frequently",
    "patterns",
    "represented",
    "types",
}

MULTI_HOP_QUERY_MARKERS = {
    "affect",
    "against",
    "cited",
    "differently",
    "distinguish",
    "influence",
    "multi",
    "precedents",
    "standard",
    "strickland",
}

LEGAL_CONCEPT_ALIASES = {
    "constitutional rights": [
        "fourth amendment",
        "fifth amendment",
        "sixth amendment",
        "fourteenth amendment",
        "miranda",
        "search and seizure",
        "double jeopardy",
        "self-incrimination",
        "right to counsel",
        "due process",
        "equal protection",
    ],
    "criminal appeal": [
        "sufficiency of the evidence",
        "ineffective assistance",
        "preservation of error",
        "standard of review",
        "harmless error",
    ],
    "insufficient evidence": [
        "sufficiency of the evidence",
        "rational trier",
        "reasonable doubt",
        "light most favorable",
        "substantial evidence",
    ],
    "government": [
        "sovereign immunity",
        "qualified immunity",
        "42 u.s.c",
        "section 1983",
        "state agency",
        "due process",
        "equal protection",
    ],
    "civil disputes": [
        "contract",
        "tort",
        "civil rights",
        "intellectual property",
        "trademark",
        "habeas",
    ],
    "procedural": [
        "procedural default",
        "waiver",
        "jurisdiction",
        "untimely",
        "preserve",
        "certificate of appealability",
    ],
    "direct appeals": [
        "direct appeal",
        "post-conviction",
        "postconviction",
        "habeas corpus",
        "collateral attack",
        "cause and prejudice",
    ],
    "standard of review": [
        "abuse of discretion",
        "de novo",
        "clearly erroneous",
        "substantial evidence",
        "harmless error",
    ],
    "federal circuit": [
        "court of appeals",
        "federal circuit",
        "state appellate",
        "supreme court precedent",
        "habeas corpus",
        "de novo",
    ],
    "strickland": [
        "strickland",
        "ineffective assistance",
        "objective standard",
        "reasonable probability",
        "counsel",
    ],
}

COMMUNITY_QUERY_HINTS = {
    "global_legal_themes": {"common", "themes", "theme", "criminal", "appeal", "appeals"},
    "global_constitutional_rights": {"constitutional", "rights", "right", "amendment", "amendments"},
    "global_court_types": {"court", "courts", "represented", "types", "type"},
    "global_government_civil_cases": {"government", "state", "agency", "agencies", "defendants"},
    "global_civil_disputes": {"civil", "disputes", "alongside", "contract", "tort"},
    "global_procedural_grounds": {"procedural", "grounds", "deny", "dismiss", "appeals"},
    "global_direct_postconviction_review": {
        "direct",
        "appeals",
        "post",
        "conviction",
        "post-conviction",
        "relief",
        "petitions",
        "habeas",
        "collateral",
    },
    "global_federal_state_constitutional_review": {
        "federal",
        "circuit",
        "state",
        "appellate",
        "constitutional",
        "questions",
        "differently",
    },
}


@dataclass(frozen=True)
class GraphContext:
    case: dict[str, Any]
    chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    source: str


@dataclass(frozen=True)
class EntityPath:
    score: float
    flow_score: float
    hop_penalty: float
    path_text: str
    chunk: dict[str, Any]
    case: dict[str, Any]


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
    if not path.exists():
        return []
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="|", escapechar="\\", quoting=csv.QUOTE_NONE))


@lru_cache(maxsize=4)
def load_local_graph_tables(tigergraph_dir: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    cases = read_csv(tigergraph_dir / "legal_cases.csv")
    chunks_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(tigergraph_dir / "chunks.csv"):
        row["chunk_index"] = int(row.get("chunk_index") or 0)
        row["token_count"] = int(row.get("token_count") or 0)
        chunks_by_case.setdefault(row["source_id"], []).append(row)

    for chunks in chunks_by_case.values():
        chunks.sort(key=lambda item: item["chunk_index"])

    for case in cases:
        chunks = chunks_by_case.get(case["case_id"], [])
        case_metadata = " ".join(
            str(case.get(name, ""))
            for name in ("case_id", "case_name", "citations", "court", "decision_date")
        )
        case["_search_text"] = (
            f"{case_metadata} "
            f"{' '.join(str(chunk.get('text', '')) for chunk in chunks)}"
        ).lower()
    return cases, chunks_by_case


@lru_cache(maxsize=4)
def load_entity_graph_tables(tigergraph_dir: Path) -> dict[str, Any]:
    entities = {row["entity_id"]: row for row in read_csv(tigergraph_dir / "entities.csv")}
    chunks = {row["chunk_id"]: row for row in read_csv(tigergraph_dir / "chunks.csv")}
    cases = {row["case_id"]: row for row in read_csv(tigergraph_dir / "legal_cases.csv")}

    mentions_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(tigergraph_dir / "mentions.csv"):
        row["weight"] = int(row.get("weight") or 1)
        mentions_by_entity[row["entity_id"]].append(row)

    related_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(tigergraph_dir / "related_to.csv"):
        row["weight"] = int(row.get("weight") or 1)
        related_by_entity[row["from_entity_id"]].append(row)
        related_by_entity[row["to_entity_id"]].append(
            {
                "from_entity_id": row["to_entity_id"],
                "to_entity_id": row["from_entity_id"],
                "relation_type": row.get("relation_type", "CO_OCCURS_WITH"),
                "weight": row["weight"],
            }
        )

    community_reports = read_csv(tigergraph_dir / "community_reports.csv")
    return {
        "entities": entities,
        "chunks": chunks,
        "cases": cases,
        "mentions_by_entity": mentions_by_entity,
        "related_by_entity": related_by_entity,
        "community_reports": community_reports,
    }


def query_profile(query: str, question_type: str | None = None) -> str:
    if question_type in {"local_factual", "global_synthesis", "multi_hop"}:
        return {
            "local_factual": "local",
            "global_synthesis": "global",
            "multi_hop": "multi_hop",
        }[question_type]
    terms = set(query_terms(query))
    query_lower = query.lower()
    if terms & MULTI_HOP_QUERY_MARKERS or "how do" in query_lower or "how did" in query_lower:
        return "multi_hop"
    if terms & GLOBAL_QUERY_MARKERS:
        return "global"
    return "local"


def needs_multi_hop_community_context(query: str) -> bool:
    query_lower = query.lower()
    markers = (
        "direct appeals",
        "post-conviction",
        "post conviction",
        "relief petitions",
        "federal circuit",
        "state appellate",
        "constitutional questions",
    )
    return any(marker in query_lower for marker in markers)


def routed_defaults(
    query: str,
    top_cases: int,
    chunks_per_case: int,
    max_context_tokens: int,
    question_type: str | None = None,
) -> tuple[str, int, int, int]:
    profile = query_profile(query, question_type=question_type)
    if profile == "global":
        return profile, max(top_cases, 8), 2, max_context_tokens
    if profile == "multi_hop":
        return profile, max(top_cases, 5), min(max(chunks_per_case, 4), 5), max_context_tokens
    return profile, top_cases, chunks_per_case, max_context_tokens


def expanded_query_phrases(query: str) -> list[str]:
    query_lower = query.lower()
    phrases = []
    for trigger, aliases in LEGAL_CONCEPT_ALIASES.items():
        if trigger in query_lower or any(term in query_lower for term in trigger.split()):
            phrases.extend(aliases)
    citations = re.findall(r"\b\d+\s+[A-Z][A-Za-z. ]+\s+\d+\b", query)
    phrases.extend(citation.lower() for citation in citations)
    return list(dict.fromkeys(phrases))


def score_text(query: str, terms: list[str], text: str, *, metadata_weight: float = 1.0) -> float:
    if not text:
        return 0.0
    haystack = text.lower()
    score = 0.0
    for term in terms:
        if term in haystack:
            score += 1.0
    for phrase in expanded_query_phrases(query):
        if phrase in haystack:
            score += 2.5
    for name in re.findall(r"\b[A-Z][A-Za-z.']+\s+v\.\s+[A-Z][A-Za-z.']+", query):
        if name.lower() in haystack:
            score += 8.0
    return score * metadata_weight


def entity_seed_score(query: str, terms: list[str], entity: dict[str, Any]) -> float:
    name = str(entity.get("name", ""))
    entity_type = str(entity.get("type", ""))
    frequency = int(entity.get("frequency") or 1)
    score = score_text(query, terms, name, metadata_weight=2.0)
    if entity_type in {"CASE", "LAW", "CONCEPT"}:
        score *= 1.25
    return score + min(frequency, 20) * 0.02


def select_seed_entities(query: str, tigergraph_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    tables = load_entity_graph_tables(tigergraph_dir)
    entities = tables["entities"]
    if not entities:
        return []
    terms = query_terms(query)
    scored = [
        (entity_seed_score(query, terms, entity), entity)
        for entity in entities.values()
    ]
    scored = [(score, entity) for score, entity in scored if score > 0.1]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entity for _, entity in scored[:limit]]


def chunk_case(chunk: dict[str, Any], cases: dict[str, dict[str, str]]) -> dict[str, str]:
    return cases.get(str(chunk.get("source_id")), {"case_id": chunk.get("source_id", ""), "case_name": ""})


def retrieve_entity_paths(
    query: str,
    tigergraph_dir: Path,
    max_paths: int,
    token_budget: int,
) -> tuple[str, list[dict[str, Any]]]:
    tables = load_entity_graph_tables(tigergraph_dir)
    entities: dict[str, dict[str, Any]] = tables["entities"]
    chunks: dict[str, dict[str, Any]] = tables["chunks"]
    cases: dict[str, dict[str, str]] = tables["cases"]
    mentions_by_entity: dict[str, list[dict[str, Any]]] = tables["mentions_by_entity"]
    related_by_entity: dict[str, list[dict[str, Any]]] = tables["related_by_entity"]
    if not entities:
        return "", []

    terms = query_terms(query)
    paths: list[EntityPath] = []
    seeds = select_seed_entities(query, tigergraph_dir=tigergraph_dir, limit=5)

    for seed in seeds:
        seed_id = seed["entity_id"]
        seed_name = seed.get("name", seed_id)
        for mention in mentions_by_entity.get(seed_id, [])[:12]:
            chunk = chunks.get(mention["chunk_id"])
            if not chunk:
                continue
            case = chunk_case(chunk, cases)
            path_text = (
                f"{seed_name} --[MENTIONED_IN w={mention['weight']}]--> "
                f"Chunk {chunk.get('chunk_index')} --[HAS_CASE]--> {case.get('case_name') or case.get('case_id')}"
            )
            relevance = score_text(query, terms, f"{path_text} {chunk.get('text', '')}")
            flow_score = max(1, int(mention["weight"]))
            hop_penalty = 0.9 ** 2
            paths.append(
                EntityPath(
                    score=relevance * flow_score * hop_penalty,
                    flow_score=flow_score,
                    hop_penalty=hop_penalty,
                    path_text=path_text,
                    chunk=chunk,
                    case=case,
                )
            )

        for edge in related_by_entity.get(seed_id, [])[:12]:
            other = entities.get(edge["to_entity_id"])
            if not other:
                continue
            other_name = other.get("name", edge["to_entity_id"])
            for mention in mentions_by_entity.get(edge["to_entity_id"], [])[:8]:
                chunk = chunks.get(mention["chunk_id"])
                if not chunk:
                    continue
                case = chunk_case(chunk, cases)
                flow_score = max(1, min(int(edge["weight"]), int(mention["weight"])))
                hop_penalty = 0.9 ** 3
                path_text = (
                    f"{seed_name} --[{edge.get('relation_type', 'RELATED_TO')} w={edge['weight']}]--> "
                    f"{other_name} --[MENTIONED_IN w={mention['weight']}]--> "
                    f"Chunk {chunk.get('chunk_index')} --[HAS_CASE]--> {case.get('case_name') or case.get('case_id')}"
                )
                relevance = score_text(query, terms, f"{path_text} {chunk.get('text', '')}")
                paths.append(
                    EntityPath(
                        score=relevance * flow_score * hop_penalty,
                        flow_score=flow_score,
                        hop_penalty=hop_penalty,
                        path_text=path_text,
                        chunk=chunk,
                        case=case,
                    )
                )

    paths.sort(key=lambda item: item.score, reverse=True)
    selected: list[EntityPath] = []
    selected_keys: set[tuple[str, str]] = set()
    used_tokens = 0
    for path in paths:
        key = (str(path.chunk.get("chunk_id")), path.path_text)
        if key in selected_keys:
            continue
        block = format_entity_path(path)
        block_tokens = count_tokens(block)
        if selected and used_tokens + block_tokens > token_budget:
            continue
        selected.append(path)
        selected_keys.add(key)
        used_tokens += block_tokens
        if len(selected) >= max_paths:
            break

    records = [
        {
            "score": round(path.score, 4),
            "flow_score": path.flow_score,
            "hop_penalty": round(path.hop_penalty, 4),
            "path": path.path_text,
            "case_id": path.case.get("case_id"),
            "case_name": path.case.get("case_name"),
            "chunk_id": path.chunk.get("chunk_id"),
            "chunk_index": path.chunk.get("chunk_index"),
            "text": path.chunk.get("text", ""),
        }
        for path in selected
    ]
    return "\n\n".join(format_entity_path(path) for path in selected), records


def retrieve_community_context(
    query: str,
    tigergraph_dir: Path,
    max_reports: int = 3,
    token_budget: int = 900,
) -> tuple[str, list[dict[str, Any]]]:
    tables = load_entity_graph_tables(tigergraph_dir)
    reports = tables["community_reports"]
    if not reports:
        return "", []
    terms = query_terms(query)
    term_set = set(terms)
    scored = []
    for row in reports:
        community_id = str(row.get("community_id", ""))
        score = score_text(
            query,
            terms,
            f"{community_id} {row.get('summary', '')}",
        )
        if community_id.startswith("global_"):
            score += 2.0
        score += 2.0 * len(term_set & COMMUNITY_QUERY_HINTS.get(community_id, set()))
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = []
    used_tokens = 0
    for score, row in scored:
        if score <= 0 and selected:
            continue
        block = format_community_report(row, score)
        block_tokens = count_tokens(block)
        if selected and used_tokens + block_tokens > token_budget:
            continue
        selected.append((score, row, block))
        used_tokens += block_tokens
        if len(selected) >= max_reports:
            break

    records = [
        {
            "score": round(score, 4),
            "community_id": row.get("community_id"),
            "level": row.get("level"),
            "summary": row.get("summary"),
        }
        for score, row, _ in selected
    ]
    return "\n\n".join(block for _, _, block in selected), records


def format_community_report(row: dict[str, Any], score: float) -> str:
    return (
        f"Community report score={score:.4f} level={row.get('level')} id={row.get('community_id')}\n"
        f"{row.get('summary', '')}"
    )


def format_entity_path(path: EntityPath) -> str:
    return (
        f"Path score={path.score:.4f} flow={path.flow_score} hop_penalty={path.hop_penalty:.3f}\n"
        f"{path.path_text}\n"
        f"Evidence: {path.chunk.get('text', '')}"
    )


def score_case(query: str, terms: list[str], case: dict[str, str], chunks: list[dict[str, Any]]) -> float:
    if not terms:
        return 0.0

    case_text = " ".join(
        str(case.get(name, ""))
        for name in ("case_id", "case_name", "citations", "court", "decision_date")
    ).lower()
    haystack = str(case.get("_search_text") or case_text)
    score = score_text(query, terms, case_text, metadata_weight=2.0)
    score += score_text(query, terms, haystack)
    if "criminal" in terms or "appeal" in terms:
        criminal_markers = (
            "court of criminal appeals",
            "criminal appeal",
            "post-conviction",
            "convicted",
            "conviction",
            "defendant",
            "sentence",
        )
        score += min(sum(1 for marker in criminal_markers if marker in haystack), 5) * 1.25
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


def local_context(case: dict[str, str], tigergraph_dir: Path, chunks_per_case: int, query: str | None = None) -> GraphContext:
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
        chunks=select_relevant_chunks(query or "", chunks_by_case.get(case["case_id"], []), chunks_per_case),
        citations=citations,
        source="local_csv_fallback",
    )


def select_relevant_chunks(query: str, chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not chunks:
        return []
    terms = query_terms(query)
    profile = query_profile(query)
    scored = [
        (score_text(query, terms, str(chunk.get("text", ""))), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda item: (item[0], -int(item[1].get("chunk_index") or 0)), reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Case captions and citations usually live at the front of an opinion, so
    # local factual questions keep that anchor even if another chunk scores higher.
    if profile == "local":
        first = min(chunks, key=lambda item: int(item.get("chunk_index") or 0))
        chunk_id = str(first.get("chunk_id") or first.get("chunk_index"))
        selected.append(first)
        seen.add(chunk_id)

    for score, chunk in scored:
        if score <= 0 and selected:
            continue
        chunk_id = str(chunk.get("chunk_id") or chunk.get("chunk_index"))
        if chunk_id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk_id)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for chunk in sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0)):
            chunk_id = str(chunk.get("chunk_id") or chunk.get("chunk_index"))
            if chunk_id in seen:
                continue
            selected.append(chunk)
            seen.add(chunk_id)
            if len(selected) >= limit:
                break

    selected.sort(key=lambda item: int(item.get("chunk_index") or 0))
    return selected


def graph_context(
    case: dict[str, str],
    query: str,
    client: TigerGraphClient | None,
    tigergraph_dir: Path,
    query_name: str,
    chunks_per_case: int,
    allow_local_fallback: bool,
) -> GraphContext:
    if client is None:
        if allow_local_fallback:
            return local_context(case, tigergraph_dir, chunks_per_case, query=query)
        raise RuntimeError("TG_HOST is missing. Add TigerGraph connection settings to .env.")

    try:
        fetch_limit = max(chunks_per_case * DEFAULT_FETCH_MULTIPLIER, chunks_per_case, 12)
        payload = client.run_query(
            query_name=query_name,
            params={"case_id": case["case_id"], "chunk_limit": fetch_limit},
        )
        seed_items = result_items(payload, "seed")
        chunk_items = result_items(payload, "chunks")
        cite_items = result_items(payload, "cites")
        graph_case = unwrap_vertex(seed_items[0]) if seed_items else case
        chunks = [unwrap_vertex(item) for item in chunk_items]
        citations = [unwrap_vertex(item) for item in cite_items]
        chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))
        chunks = select_relevant_chunks(query, chunks, chunks_per_case)
        return GraphContext(case=graph_case, chunks=chunks, citations=citations, source="tigergraph")
    except RuntimeError:
        if allow_local_fallback:
            return local_context(case, tigergraph_dir, chunks_per_case, query=query)
        raise


def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def build_context_block(
    contexts: list[GraphContext],
    max_context_tokens: int,
    path_context: str = "",
    community_context: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    blocks = []
    used = 0
    retrieved = []

    if community_context:
        community_tokens = count_tokens(community_context)
        if community_tokens <= max_context_tokens:
            blocks.append("Corpus community reports:\n" + community_context)
            used += community_tokens

    if path_context:
        path_tokens = count_tokens(path_context)
        if used + path_tokens <= max_context_tokens:
            blocks.append("Structured graph paths:\n" + path_context)
            used += path_tokens

    for context in contexts:
        case = context.case
        citation_text = "; ".join(
            str(citation.get("citation_text") or citation.get("citation_id") or "")
            for citation in context.citations
        )
        header = (
            f"Case: {case.get('case_name') or case.get('case_id')}\n"
            f"Case ID: {case.get('case_id')}\n"
            f"Court: {case.get('court') or 'unknown'}\n"
            f"Decision Date: {case.get('decision_date') or 'unknown'}\n"
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
    question_type: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile, top_cases, chunks_per_case, max_context_tokens = routed_defaults(
        query=query,
        top_cases=top_cases,
        chunks_per_case=chunks_per_case,
        max_context_tokens=max_context_tokens,
        question_type=question_type,
    )
    cases = rank_cases(query=query, tigergraph_dir=tigergraph_dir, top_cases=top_cases)
    client = TigerGraphClient.from_env()
    contexts = [
        graph_context(
            case=case,
            query=query,
            client=client,
            tigergraph_dir=tigergraph_dir,
            query_name=query_name,
            chunks_per_case=chunks_per_case,
            allow_local_fallback=allow_local_fallback,
        )
        for case in cases
    ]
    path_context = ""
    retrieved_paths: list[dict[str, Any]] = []
    community_context = ""
    retrieved_communities: list[dict[str, Any]] = []
    if profile == "global" or (profile == "multi_hop" and needs_multi_hop_community_context(query)):
        community_context, retrieved_communities = retrieve_community_context(
            query=query,
            tigergraph_dir=tigergraph_dir,
            max_reports=4 if profile == "multi_hop" else 3,
            token_budget=min(1000, max_context_tokens // 2),
        )
    if profile == "multi_hop":
        path_context, retrieved_paths = retrieve_entity_paths(
            query=query,
            tigergraph_dir=tigergraph_dir,
            max_paths=7,
            token_budget=min(1000, max_context_tokens // 2),
        )
    context_text, retrieved = build_context_block(
        contexts,
        max_context_tokens=max_context_tokens,
        path_context=path_context,
        community_context=community_context,
    )
    if context_only:
        return {
            "pipeline": "graphrag",
            "question_id": question_id,
            "question": query,
            "query_profile": profile,
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
            "retrieved_paths": retrieved_paths,
            "retrieved_communities": retrieved_communities,
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
        "query_profile": profile,
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
        "retrieved_paths": retrieved_paths,
        "retrieved_communities": retrieved_communities,
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
                question_type=str(item.get("type") or ""),
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
        print(json.dumps(record, indent=2, ensure_ascii=True))
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
