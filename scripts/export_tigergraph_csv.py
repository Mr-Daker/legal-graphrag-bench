from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/tigergraph")
DEFAULT_REPORT = Path("data/reports/tigergraph_export_report.json")
DEFAULT_DELIMITER = "|"


def safe_text(value: Any, delimiter: str = DEFAULT_DELIMITER) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.replace(delimiter, " ")


def citation_id(citation_text: str) -> str:
    digest = hashlib.sha1(citation_text.encode("utf-8")).hexdigest()
    return f"citation_{digest[:16]}"


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

    report = {
        "input_path": str(chunks_path),
        "output_dir": str(output_dir),
        "cases": len(cases),
        "chunks": len(chunks),
        "citations": len(citations),
        "has_chunk_edges": len(has_chunk_edges),
        "next_chunk_edges": len(next_chunk_edges),
        "cites_edges": len(cites_edges),
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
