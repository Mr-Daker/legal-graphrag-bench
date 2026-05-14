from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/tigergraph")
DEFAULT_REPORT = Path("data/reports/tigergraph_export_report.json")
DEFAULT_DELIMITER = "|"
MAX_ENTITIES_PER_CHUNK = 16


LEGAL_CONCEPTS = {
    "abuse of discretion": "CONCEPT",
    "administrative procedure": "CONCEPT",
    "certificate of appealability": "CONCEPT",
    "collateral attack": "CONCEPT",
    "constitutional": "CONCEPT",
    "contract": "CONCEPT",
    "de novo": "CONCEPT",
    "double jeopardy": "CONCEPT",
    "due process": "CONCEPT",
    "equal protection": "CONCEPT",
    "evidentiary ruling": "CONCEPT",
    "fifth amendment": "LAW",
    "fourteenth amendment": "LAW",
    "fourth amendment": "LAW",
    "harmless error": "CONCEPT",
    "habeas corpus": "CONCEPT",
    "ineffective assistance": "CONCEPT",
    "intellectual property": "CONCEPT",
    "jurisdiction": "CONCEPT",
    "miranda": "LAW",
    "post-conviction": "CONCEPT",
    "procedural default": "CONCEPT",
    "qualified immunity": "CONCEPT",
    "reasonable doubt": "CONCEPT",
    "right to counsel": "LAW",
    "search and seizure": "LAW",
    "self-incrimination": "LAW",
    "sixth amendment": "LAW",
    "sovereign immunity": "CONCEPT",
    "standard of review": "CONCEPT",
    "strickland": "LAW",
    "sufficiency of the evidence": "CONCEPT",
    "waiver": "CONCEPT",
}

TOPIC_COMMUNITIES = {
    "global_legal_themes": {
        "level": 2,
        "title": "Common legal themes across the corpus",
        "terms": [
            "standard of review",
            "sufficiency of the evidence",
            "ineffective assistance",
            "preservation of error",
            "harmless error",
            "procedural default",
        ],
    },
    "global_constitutional_rights": {
        "level": 2,
        "title": "Constitutional rights raised in criminal and appellate cases",
        "terms": [
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
        "guidance": (
            "For synthesis, group the rights as Fourth Amendment search-and-seizure protections; "
            "Fifth Amendment double-jeopardy and self-incrimination protections; Sixth Amendment "
            "rights to counsel, jury, confrontation, and fair-trial safeguards; and Fourteenth "
            "Amendment due-process and equal-protection guarantees."
        ),
    },
    "global_court_types": {
        "level": 2,
        "title": "Court types represented in the corpus",
        "terms": [
            "court of appeals",
            "district court",
            "supreme court",
            "supreme judicial court",
            "court of criminal appeals",
            "court of appeal",
        ],
        "guidance": (
            "For synthesis, group these as United States Courts of Appeals, United States District "
            "Courts, state supreme courts, and state intermediate appellate courts, including "
            "specialized state criminal appellate courts where present."
        ),
    },
    "global_government_civil_cases": {
        "level": 2,
        "title": "Civil cases involving government or state agencies",
        "terms": [
            "sovereign immunity",
            "qualified immunity",
            "42 u.s.c",
            "section 1983",
            "state agency",
            "due process",
            "equal protection",
            "administrative procedure",
        ],
    },
    "global_civil_disputes": {
        "level": 2,
        "title": "Civil dispute types appearing alongside criminal cases",
        "terms": [
            "contract",
            "tort",
            "civil rights",
            "intellectual property",
            "trademark",
            "habeas corpus",
        ],
        "guidance": (
            "For synthesis, name the broad civil categories even when individual labels are sparse: "
            "contract disputes, tort claims, civil-rights claims, intellectual-property/trademark "
            "matters, habeas or prisoner civil proceedings, and government/agency disputes."
        ),
    },
    "global_procedural_grounds": {
        "level": 2,
        "title": "Procedural grounds for denial or dismissal",
        "terms": [
            "procedural default",
            "waiver",
            "jurisdiction",
            "untimely",
            "preserve",
            "certificate of appealability",
        ],
        "guidance": (
            "For synthesis, explain that courts deny or dismiss on waiver or failure to preserve "
            "error, lack of jurisdiction, untimeliness, procedural default, and habeas certificate "
            "or exhaustion defects."
        ),
    },
    "global_direct_postconviction_review": {
        "level": 2,
        "title": "Direct appeals versus post-conviction or habeas review",
        "terms": [
            "direct appeal",
            "post-conviction",
            "habeas corpus",
            "collateral attack",
            "procedural default",
            "waiver",
        ],
        "guidance": (
            "Direct appeals challenge trial error soon after judgment. Post-conviction or habeas "
            "petitions are collateral attacks that usually raise constitutional claims after the "
            "direct appeal, and courts apply stricter bars such as exhaustion, waiver, procedural "
            "default, and cause-and-prejudice requirements."
        ),
    },
    "global_federal_state_constitutional_review": {
        "level": 2,
        "title": "Federal circuit versus state appellate constitutional review",
        "terms": [
            "court of appeals",
            "federal circuit",
            "state appellate",
            "supreme court",
            "habeas corpus",
            "de novo",
        ],
        "guidance": (
            "Federal circuit courts apply federal constitutional doctrine and Supreme Court "
            "precedent, often reviewing pure constitutional questions de novo. State appellate "
            "courts apply both federal minimums and state constitutional law, and federal courts "
            "may later review state-prisoner constitutional claims through habeas corpus."
        ),
    },
}

CASE_NAME_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9.'&-]+(?:\s+[A-Z][A-Za-z0-9.'&-]+){0,5}\s+v\.\s+"
    r"[A-Z][A-Za-z0-9.'&-]+(?:\s+[A-Z][A-Za-z0-9.'&-]+){0,6}"
)
CITATION_RE = re.compile(
    r"\b\d+\s+(?:F\.?\s?(?:Supp\.?\s?\d*d?|2d|3d)?|U\.S\.|S\.?\s?Ct\.|"
    r"P\.?\s?\d*d?|N\.?E\.?\s?\d*d?|S\.?W\.?\s?\d*d?|N\.?W\.?\s?\d*d?|"
    r"Mass\.|Idaho|Ohio|OK\s+CR)\s+\d+\b"
)
STATUTE_RE = re.compile(
    r"\b(?:\d+\s+U\.S\.C\.?\s*(?:§|section)?\s*[\w.()-]+|"
    r"\d+\s+O\.S\.[^,;.]{0,40}|"
    r"\d+\s+C\.F\.R\.?\s*[\w.()-]+)\b",
    re.IGNORECASE,
)
COURT_RE = re.compile(
    r"\b(?:United States\s+)?(?:Supreme Court|Court of Appeals|District Court|"
    r"Court of Criminal Appeals|Court of Appeal|Supreme Judicial Court)"
    r"(?:\s+(?:of|for|at)\s+[A-Z][A-Za-z .'-]+)?",
)
JUDGE_RE = re.compile(r"\b(?:Judge|Justice|J\.)\s+[A-Z][A-Za-z.'-]+")
DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b"
)


def safe_text(value: Any, delimiter: str = DEFAULT_DELIMITER) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.replace(delimiter, " ")


def citation_id(citation_text: str) -> str:
    digest = hashlib.sha1(citation_text.encode("utf-8")).hexdigest()
    return f"citation_{digest[:16]}"


def entity_id(name: str, entity_type: str) -> str:
    digest = hashlib.sha1(f"{entity_type}:{normalise_entity_name(name)}".encode("utf-8")).hexdigest()
    return f"entity_{digest[:16]}"


def normalise_entity_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip(" .,;:()[]{}\"'"))


def add_entity(target: list[tuple[str, str]], name: str, entity_type: str) -> None:
    clean = normalise_entity_name(name)
    if len(clean) < 3:
        return
    if clean in {"court", "state", "united states", "opinion"}:
        return
    target.append((clean, entity_type))


def extract_entities(text: str, metadata: dict[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    case_name = metadata.get("case_name")
    if case_name:
        add_entity(found, str(case_name), "CASE")

    citations = metadata.get("citations") or []
    if not isinstance(citations, list):
        citations = [citations]
    for citation in citations:
        add_entity(found, str(citation), "CITATION")

    for regex, entity_type in (
        (CASE_NAME_RE, "CASE"),
        (CITATION_RE, "CITATION"),
        (STATUTE_RE, "LAW"),
        (COURT_RE, "ORG"),
        (JUDGE_RE, "PERSON"),
    ):
        for match in regex.finditer(text):
            add_entity(found, match.group(0), entity_type)

    lower = text.lower()
    for phrase, entity_type in LEGAL_CONCEPTS.items():
        if phrase in lower:
            add_entity(found, phrase, entity_type)

    counts = Counter(found)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))
    return [entity for entity, _ in ranked[:MAX_ENTITIES_PER_CHUNK]]


def extract_court_and_date(text: str) -> tuple[str, str]:
    court = ""
    decision_date = ""
    court_match = COURT_RE.search(text)
    if court_match:
        court = safe_text(court_match.group(0))
        court = re.split(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            court,
            maxsplit=1,
        )[0].strip(" .")
    date_match = DATE_RE.search(text)
    if date_match:
        decision_date = safe_text(date_match.group(0))
    return court, decision_date


def iter_chunks(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def export_csv(chunks_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict[str, str]] = {}
    chunks: list[dict[str, Any]] = []
    citations: dict[str, dict[str, str]] = {}
    has_chunk_edges: list[dict[str, str]] = []
    cites_edges: set[tuple[str, str]] = set()
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entities: dict[str, dict[str, Any]] = {}
    mention_counts: Counter[tuple[str, str]] = Counter()
    related_counts: Counter[tuple[str, str, str]] = Counter()
    community_members: dict[str, set[str]] = defaultdict(set)

    for chunk in iter_chunks(chunks_path):
        metadata = chunk.get("metadata") or {}
        source_id = safe_text(chunk.get("source_id"))
        case_name = safe_text(metadata.get("case_name")) or source_id
        citation_values = metadata.get("citations") or []
        if not isinstance(citation_values, list):
            citation_values = [citation_values]
        citation_values = [safe_text(value) for value in citation_values if safe_text(value)]

        cases.setdefault(
            source_id,
            {
                "case_id": source_id,
                "case_name": case_name,
                "citations": "; ".join(citation_values),
                "court": "",
                "decision_date": "",
            },
        )
        if int(chunk.get("chunk_index") or 0) == 0 and (
            not cases[source_id]["court"] or not cases[source_id]["decision_date"]
        ):
            court, decision_date = extract_court_and_date(str(chunk.get("text", "")))
            if court and not cases[source_id]["court"]:
                cases[source_id]["court"] = court
            if decision_date and not cases[source_id]["decision_date"]:
                cases[source_id]["decision_date"] = decision_date

        chunk_id = safe_text(chunk.get("id"))
        chunk_row = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "chunk_index": int(chunk.get("chunk_index") or 0),
            "token_count": int(chunk.get("token_count") or 0),
            "text": safe_text(chunk.get("text")),
        }
        chunks.append(chunk_row)
        chunks_by_source[source_id].append(chunk_row)
        has_chunk_edges.append({"case_id": source_id, "chunk_id": chunk_id})

        for citation_text in citation_values:
            cid = citation_id(citation_text)
            citations.setdefault(cid, {"citation_id": cid, "citation_text": citation_text})
            cites_edges.add((source_id, cid))

        chunk_entities = extract_entities(str(chunk.get("text", "")), metadata)
        for name, entity_type in chunk_entities:
            eid = entity_id(name, entity_type)
            entities.setdefault(
                eid,
                {
                    "entity_id": eid,
                    "name": name,
                    "type": entity_type,
                    "frequency": 0,
                },
            )
            entities[eid]["frequency"] += 1
            mention_counts[(chunk_id, eid)] += 1
            community_members[entity_type].add(eid)

        unique_entity_ids = sorted({entity_id(name, entity_type) for name, entity_type in chunk_entities})
        for left, right in combinations(unique_entity_ids[:MAX_ENTITIES_PER_CHUNK], 2):
            if left == right:
                continue
            src, tgt = sorted((left, right))
            related_counts[(src, tgt, "CO_OCCURS_WITH")] += 1

    next_chunk_edges = []
    for source_chunks in chunks_by_source.values():
        source_chunks.sort(key=lambda item: item["chunk_index"])
        for current, following in zip(source_chunks, source_chunks[1:]):
            next_chunk_edges.append(
                {
                    "from_chunk_id": current["chunk_id"],
                    "to_chunk_id": following["chunk_id"],
                }
            )

    files = {
        "legal_cases": output_dir / "legal_cases.csv",
        "chunks": output_dir / "chunks.csv",
        "citations": output_dir / "citations.csv",
        "has_chunk": output_dir / "has_chunk.csv",
        "next_chunk": output_dir / "next_chunk.csv",
        "cites": output_dir / "cites.csv",
        "entities": output_dir / "entities.csv",
        "mentions": output_dir / "mentions.csv",
        "related_to": output_dir / "related_to.csv",
        "community_reports": output_dir / "community_reports.csv",
    }

    write_csv(files["legal_cases"], cases.values(), ["case_id", "case_name", "citations", "court", "decision_date"])
    write_csv(files["chunks"], chunks, ["chunk_id", "source_id", "chunk_index", "token_count", "text"])
    write_csv(files["citations"], citations.values(), ["citation_id", "citation_text"])
    write_csv(files["has_chunk"], has_chunk_edges, ["case_id", "chunk_id"])
    write_csv(files["next_chunk"], next_chunk_edges, ["from_chunk_id", "to_chunk_id"])
    write_csv(
        files["cites"],
        [{"case_id": case_id, "citation_id": cid} for case_id, cid in sorted(cites_edges)],
        ["case_id", "citation_id"],
    )
    write_csv(
        files["entities"],
        sorted(entities.values(), key=lambda item: item["entity_id"]),
        ["entity_id", "name", "type", "frequency"],
    )
    write_csv(
        files["mentions"],
        [
            {"chunk_id": chunk_id, "entity_id": eid, "weight": weight}
            for (chunk_id, eid), weight in sorted(mention_counts.items())
        ],
        ["chunk_id", "entity_id", "weight"],
    )
    write_csv(
        files["related_to"],
        [
            {
                "from_entity_id": src,
                "to_entity_id": tgt,
                "relation_type": relation_type,
                "weight": weight,
            }
            for (src, tgt, relation_type), weight in sorted(related_counts.items())
        ],
        ["from_entity_id", "to_entity_id", "relation_type", "weight"],
    )
    community_rows = []
    for entity_type, member_ids in sorted(community_members.items()):
        names = [
            entities[eid]["name"]
            for eid in sorted(member_ids, key=lambda item: (-entities[item]["frequency"], entities[item]["name"]))[:20]
        ]
        top_member_ids = [
            eid
            for eid in sorted(member_ids, key=lambda item: (-entities[item]["frequency"], entities[item]["name"]))[:200]
        ]
        community_rows.append(
            {
                "community_id": f"community_{entity_type.lower()}",
                "level": 1,
                "summary": safe_text(f"{entity_type} community: " + "; ".join(names)),
                "entity_ids": ";".join(top_member_ids),
            }
        )
    entity_by_name = {row["name"]: row for row in entities.values()}
    for community_id, spec in TOPIC_COMMUNITIES.items():
        found = []
        member_ids = []
        for term in spec["terms"]:
            row = entity_by_name.get(term)
            if row:
                found.append(f"{term} (freq={row['frequency']})")
                member_ids.append(row["entity_id"])
            else:
                found.append(f"{term} (freq=0)")
        summary = (
            f"{spec['title']}. Salient extracted concepts: "
            f"{'; '.join(found)}. {spec.get('guidance', '')} "
            "Use this report for corpus-wide/global synthesis questions."
        )
        community_rows.append(
            {
                "community_id": community_id,
                "level": spec["level"],
                "summary": safe_text(summary),
                "entity_ids": ";".join(member_ids),
            }
        )
    write_csv(
        files["community_reports"],
        community_rows,
        ["community_id", "level", "summary", "entity_ids"],
    )

    report = {
        "input_path": str(chunks_path),
        "output_dir": str(output_dir),
        "cases": len(cases),
        "chunks": len(chunks),
        "citations": len(citations),
        "has_chunk_edges": len(has_chunk_edges),
        "next_chunk_edges": len(next_chunk_edges),
        "cites_edges": len(cites_edges),
        "entities": len(entities),
        "mentions_edges": len(mention_counts),
        "related_to_edges": len(related_counts),
        "community_reports": len(community_rows),
        "files": {key: str(value) for key, value in files.items()},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_csv(path: Path, rows, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=DEFAULT_DELIMITER,
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chunked legal corpus into TigerGraph CSV files.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_csv(chunks_path=args.chunks, output_dir=args.output_dir, report_path=args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
