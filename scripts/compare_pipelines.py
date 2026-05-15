from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_QUESTIONS = Path("data/eval/questions_dev.json")
DEFAULT_RESULTS = {
    "llm_only": Path("data/results/llm_only_results.jsonl"),
    "basic_rag": Path("data/results/basic_rag_results.jsonl"),
    "graphrag": Path("data/results/graphrag_results.jsonl"),
}
DEFAULT_JSON_OUTPUT = Path("data/reports/pipeline_comparison_report.json")
DEFAULT_MD_OUTPUT = Path("data/reports/pipeline_comparison_report.md")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "was",
    "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in data}


def norm_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
        if token not in STOPWORDS
    }


def lexical_f1(reference: str, answer: str) -> float:
    ref = norm_words(reference)
    cand = norm_words(answer)
    if not ref or not cand:
        return 0.0
    overlap = len(ref & cand)
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def heuristic_pass(reference: str, answer: str) -> bool:
    if not answer.strip():
        return False
    if "not available" in answer.lower() or "insufficient" in answer.lower():
        return False
    return lexical_f1(reference, answer) >= 0.22


def pct_delta(base: float, value: float) -> float | None:
    if base == 0:
        return None
    return (base - value) / base * 100


def summarize_pipeline(rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "questions": 0,
            "avg_total_tokens": None,
            "avg_prompt_tokens": None,
            "avg_completion_tokens": None,
            "avg_latency_ms": None,
            "heuristic_pass_rate": None,
        }

    scored = []
    for row in rows:
        qid = str(row.get("question_id"))
        reference = str(questions.get(qid, {}).get("reference_answer", ""))
        answer = str(row.get("answer", ""))
        scored.append(
            {
                "question_id": qid,
                "lexical_f1": round(lexical_f1(reference, answer), 4),
                "heuristic_pass": heuristic_pass(reference, answer),
            }
        )

    return {
        "questions": len(rows),
        "avg_total_tokens": round(mean(float(row.get("total_tokens") or 0) for row in rows), 2),
        "avg_prompt_tokens": round(mean(float(row.get("prompt_tokens") or 0) for row in rows), 2),
        "avg_completion_tokens": round(mean(float(row.get("completion_tokens") or 0) for row in rows), 2),
        "avg_latency_ms": round(mean(float(row.get("latency_ms") or 0) for row in rows), 2),
        "heuristic_pass_rate": round(sum(1 for item in scored if item["heuristic_pass"]) / len(scored), 4),
        "scored_questions": scored,
    }


def build_report(questions_path: Path, results: dict[str, Path]) -> dict[str, Any]:
    questions = load_questions(questions_path)
    rows_by_pipeline = {name: load_jsonl(path) for name, path in results.items()}
    summaries = {
        name: summarize_pipeline(rows, questions)
        for name, rows in rows_by_pipeline.items()
    }

    comparisons = {}
    basic = summaries.get("basic_rag", {})
    graph = summaries.get("graphrag", {})
    if basic.get("avg_total_tokens") is not None and graph.get("avg_total_tokens") is not None:
        comparisons["graphrag_vs_basic_rag_token_reduction_pct"] = round(
            pct_delta(float(basic["avg_total_tokens"]), float(graph["avg_total_tokens"])) or 0.0,
            2,
        )
    if basic.get("avg_latency_ms") is not None and graph.get("avg_latency_ms") is not None:
        comparisons["graphrag_vs_basic_rag_latency_reduction_pct"] = round(
            pct_delta(float(basic["avg_latency_ms"]), float(graph["avg_latency_ms"])) or 0.0,
            2,
        )

    graph_rows = rows_by_pipeline.get("graphrag", [])
    official_rows = [
        row.get("official_graphrag_base")
        for row in graph_rows
        if isinstance(row.get("official_graphrag_base"), dict)
    ]
    official_base = official_rows[0] if official_rows else {}
    official_metadata = {
        "built_on": official_base.get("upstream"),
        "commit": official_base.get("commit"),
        "available": official_base.get("available"),
        "customization": official_base.get("customization"),
        "rows_with_metadata": len(official_rows),
        "total_graphrag_rows": len(graph_rows),
        "all_rows_include_metadata": bool(graph_rows) and len(official_rows) == len(graph_rows),
    }

    per_question = defaultdict(dict)
    for pipeline, rows in rows_by_pipeline.items():
        for row in rows:
            qid = str(row.get("question_id"))
            reference = str(questions.get(qid, {}).get("reference_answer", ""))
            answer = str(row.get("answer", ""))
            per_question[qid][pipeline] = {
                "total_tokens": row.get("total_tokens"),
                "latency_ms": row.get("latency_ms"),
                "lexical_f1": round(lexical_f1(reference, answer), 4),
                "heuristic_pass": heuristic_pass(reference, answer),
                "answer_preview": answer[:180].replace("\n", " "),
            }

    return {
        "questions_path": str(questions_path),
        "result_files": {name: str(path) for name, path in results.items()},
        "official_graphrag_base": official_metadata,
        "summaries": summaries,
        "comparisons": comparisons,
        "per_question": dict(sorted(per_question.items())),
        "notes": [
            "Heuristic pass and lexical F1 are lightweight local checks, not replacements for the required LLM-as-a-Judge and BERTScore evaluation.",
            "Costs are intentionally omitted here; add provider pricing at report time to avoid stale pricing assumptions.",
        ],
    }


def markdown_table(report: dict[str, Any]) -> str:
    lines = ["# Pipeline Comparison Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Pipeline | Questions | Avg tokens | Avg prompt | Avg completion | Avg latency ms | Heuristic pass rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, summary in report["summaries"].items():
        lines.append(
            "| {name} | {questions} | {tokens} | {prompt} | {completion} | {latency} | {pass_rate} |".format(
                name=name,
                questions=summary["questions"],
                tokens=summary["avg_total_tokens"],
                prompt=summary["avg_prompt_tokens"],
                completion=summary["avg_completion_tokens"],
                latency=summary["avg_latency_ms"],
                pass_rate=summary["heuristic_pass_rate"],
            )
        )

    official = report.get("official_graphrag_base", {})
    if official.get("built_on"):
        lines.append("")
        lines.append("## Official TigerGraph GraphRAG Base")
        lines.append("")
        lines.append(f"- Built on: `{official.get('built_on')}`")
        lines.append(f"- Commit: `{official.get('commit')}`")
        lines.append(
            f"- Metadata coverage: `{official.get('rows_with_metadata')}/{official.get('total_graphrag_rows')}` GraphRAG rows"
        )
        lines.append(f"- All rows include metadata: `{official.get('all_rows_include_metadata')}`")
        lines.append(f"- Customization: {official.get('customization')}")

    lines.append("")
    lines.append("## GraphRAG vs Basic RAG")
    lines.append("")
    for key, value in report["comparisons"].items():
        lines.append(f"- `{key}`: `{value}%`")

    lines.append("")
    lines.append("## Per Question")
    lines.append("")
    for qid, pipelines in report["per_question"].items():
        lines.append(f"### {qid}")
        lines.append("")
        lines.append("| Pipeline | Tokens | Latency ms | Lexical F1 | Pass | Answer preview |")
        lines.append("|---|---:|---:|---:|---|---|")
        for pipeline, item in pipelines.items():
            preview = str(item["answer_preview"]).replace("|", "\\|")
            lines.append(
                f"| {pipeline} | {item['total_tokens']} | {item['latency_ms']} | "
                f"{item['lexical_f1']} | {item['heuristic_pass']} | {preview} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def load_accuracy_report(path: Path) -> dict[str, Any]:
    """Load an accuracy report generated by evaluate_accuracy.py, if present."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def merge_accuracy(report: dict[str, Any], accuracy: dict[str, Any]) -> None:
    """Merge judge + BERTScore results from accuracy report into the comparison report."""
    if not accuracy:
        return
    for name, comp in accuracy.get("comparison", {}).items():
        if name in report.get("summaries", {}):
            report["summaries"][name]["judge_pass_rate"] = comp.get("judge_pass_rate")
            report["summaries"][name]["bonus_judge_met"] = comp.get("bonus_judge_met")
            report["summaries"][name]["bertscore_f1_raw"] = comp.get("bertscore_f1_raw")
            report["summaries"][name]["bertscore_f1_rescaled"] = comp.get("bertscore_f1_rescaled")
            report["summaries"][name]["bonus_bertscore_raw_met"] = comp.get("bonus_bertscore_raw_met")
            report["summaries"][name]["bonus_bertscore_rescaled_met"] = comp.get("bonus_bertscore_rescaled_met")


def markdown_accuracy_section(report: dict[str, Any]) -> str:
    """Return a markdown section with judge + BERTScore summary if the data is present."""
    has_accuracy = any(
        s.get("judge_pass_rate") is not None
        for s in report.get("summaries", {}).values()
    )
    if not has_accuracy:
        return ""

    def _fmt(v: Any) -> str:
        return str(v) if v is not None else "N/A"

    def _bonus(v: Any) -> str:
        if v is True:
            return "YES"
        if v is False:
            return "NO"
        return "N/A"

    lines = ["## Accuracy Evaluation (LLM-as-a-Judge + BERTScore)", ""]
    lines.append("| Pipeline | Judge pass rate | Bonus judge (>=90%) | BERTScore F1 raw | Bonus raw (>=0.88) | BERTScore F1 rescaled | Bonus rescaled (>=0.55) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in report.get("summaries", {}).items():
        lines.append(
            f"| {name}"
            f" | {_fmt(s.get('judge_pass_rate'))}"
            f" | {_bonus(s.get('bonus_judge_met'))}"
            f" | {_fmt(s.get('bertscore_f1_raw'))}"
            f" | {_bonus(s.get('bonus_bertscore_raw_met'))}"
            f" | {_fmt(s.get('bertscore_f1_rescaled'))}"
            f" | {_bonus(s.get('bonus_bertscore_rescaled_met'))}"
            " |"
        )
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM-only, Basic RAG, and GraphRAG result files.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument(
        "--accuracy-report",
        type=Path,
        default=Path("data/reports/accuracy_report.json"),
        help="Path to accuracy_report.json produced by evaluate_accuracy.py (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.questions, DEFAULT_RESULTS)

    # Merge accuracy data if it exists
    accuracy = load_accuracy_report(args.accuracy_report)
    merge_accuracy(report, accuracy)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Build markdown — inject accuracy table after summary
    md = markdown_table(report)
    acc_section = markdown_accuracy_section(report)
    if acc_section:
        # Insert after the summary table, before "## GraphRAG vs Basic RAG"
        md = md.replace("## GraphRAG vs Basic RAG", acc_section + "## GraphRAG vs Basic RAG", 1)
    args.md_output.write_text(md, encoding="utf-8")

    print(json.dumps(report["summaries"], indent=2))
    print(json.dumps(report["comparisons"], indent=2))
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
