from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from gemini_client import DEFAULT_MODEL, compute_cost, generate_text


DEFAULT_QUESTIONS = Path("data/eval/questions_dev.json")
DEFAULT_OUTPUT = Path("data/results/llm_only_results.jsonl")

SYSTEM_PROMPT = """You answer questions directly and concisely.
If you are not sure, say that the information is not available from your general knowledge.
Do not invent citations."""


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def run_question(question: str, question_id: str, model: str) -> dict[str, Any]:
    result = generate_text(
        prompt=f"Question: {question}",
        system_instruction=SYSTEM_PROMPT,
        model=model,
        temperature=0.0,
    )
    return {
        "pipeline": "llm_only",
        "question_id": question_id,
        "question": question,
        "answer": result.answer,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": round(result.latency_ms, 2),
        "cost_usd": compute_cost(result.model, result.prompt_tokens, result.completion_tokens),
        "retrieved_context": [],
    }


def run_questions(questions: list[dict[str, Any]], output_path: Path, model: str) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(questions, start=1):
            question = item["question"]
            question_id = str(item.get("id") or f"q{index:03d}")
            record = run_question(question=question, question_id=question_id, model=model)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
            print(f"{question_id}: {record['total_tokens']} tokens, {record['latency_ms']} ms")
            time.sleep(5)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pipeline 1: Gemini LLM-only baseline.")
    parser.add_argument("--query", help="Run a single ad hoc question.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query:
        record = run_question(question=args.query, question_id="adhoc", model=args.model)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    questions = load_questions(args.questions)
    run_questions(questions=questions, output_path=args.output, model=args.model)


if __name__ == "__main__":
    main()

