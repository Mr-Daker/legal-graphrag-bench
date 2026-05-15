"""Resume GraphRAG generation for questions missing from the JSONL output."""

from __future__ import annotations

import json
import time
import argparse

from graphrag_tigergraph import (
    DEFAULT_CHUNKS_PER_CASE,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_QUERY_NAME,
    DEFAULT_QUESTIONS,
    DEFAULT_TIGERGRAPH_DIR,
    DEFAULT_TOP_CASES,
    answer_query,
    env_bool,
    load_questions,
)


def existing_question_ids() -> set[str]:
    if not DEFAULT_OUTPUT.exists():
        return set()
    ids: set[str] = set()
    for line in DEFAULT_OUTPUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(str(json.loads(line)["question_id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume or replace GraphRAG rows in the JSONL output.")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=[],
        help="Question IDs to regenerate. If omitted, only missing rows are generated.",
    )
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds to sleep between LLM calls.")
    args = parser.parse_args()
    ids_to_replace = set(args.ids)

    existing = existing_question_ids()
    items = load_questions(DEFAULT_QUESTIONS)
    allow_local_fallback = env_bool("TG_ALLOW_LOCAL_FALLBACK", True)

    if ids_to_replace and DEFAULT_OUTPUT.exists():
        kept_rows = []
        for line in DEFAULT_OUTPUT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("question_id")) not in ids_to_replace:
                kept_rows.append(row)
        with DEFAULT_OUTPUT.open("w", encoding="utf-8") as handle:
            for row in kept_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        existing = {str(row["question_id"]) for row in kept_rows}

    with DEFAULT_OUTPUT.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(items, start=1):
            question_id = str(item.get("id") or f"q{index:03d}")
            if ids_to_replace and question_id not in ids_to_replace:
                continue
            if question_id in existing:
                continue
            record = answer_query(
                query=item["question"],
                question_id=question_id,
                tigergraph_dir=DEFAULT_TIGERGRAPH_DIR,
                query_name=DEFAULT_QUERY_NAME,
                top_cases=DEFAULT_TOP_CASES,
                chunks_per_case=DEFAULT_CHUNKS_PER_CASE,
                max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
                generation_model=DEFAULT_MODEL,
                allow_local_fallback=allow_local_fallback,
                context_only=False,
                question_type=str(item.get("type") or ""),
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{question_id}: {record['total_tokens']} tokens, {record['latency_ms']} ms")
            if args.delay:
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
