"""
evaluate_accuracy.py — LLM-as-a-Judge + BERTScore evaluation for all three pipelines.

Usage:
    # Run all pipelines through judge + BERTScore:
    python scripts/evaluate_accuracy.py

    # Judge only (skip BERTScore, which requires a large model download):
    python scripts/evaluate_accuracy.py --skip-bertscore

    # Evaluate a single pipeline:
    python scripts/evaluate_accuracy.py --pipeline graphrag
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from gemini_client import DEFAULT_MODEL, DEFAULT_XAI_MODEL, generate_text


DEFAULT_QUESTIONS = Path("data/eval/questions_dev.json")
DEFAULT_RESULTS = {
    "llm_only": Path("data/results/llm_only_results.jsonl"),
    "basic_rag": Path("data/results/basic_rag_results.jsonl"),
    "graphrag": Path("data/results/graphrag_results.jsonl"),
}
DEFAULT_OUTPUT = Path("data/reports/accuracy_report.json")
DEFAULT_MD_OUTPUT = Path("data/reports/accuracy_report.md")

# ──────────────────────────────────────────────────────────────────────────────
# Judge prompt — matches the hackathon spec (PASS / FAIL single-word verdict)
# ──────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a fair evaluator of question-answering systems. "
    "Decide whether a candidate answer is substantially correct. "
    "PASS if the answer correctly addresses the core of the question and covers "
    "the key concepts — it does NOT need to be exhaustive or match every detail "
    "in the reference. "
    "FAIL only if the answer is factually wrong, completely off-topic, or says "
    "it cannot answer when a retrieval-based system should be able to. "
    "Reply with exactly one word: PASS or FAIL."
)

JUDGE_PROMPT_TEMPLATE = """\
Question:         {question}
Reference Answer: {reference}
Candidate Answer: {candidate}

Verdict (PASS or FAIL):"""

DEFAULT_XAI_JUDGE_MODEL = DEFAULT_XAI_MODEL  # e.g. grok-4.20-non-reasoning


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in data}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# LLM-as-a-Judge
# ──────────────────────────────────────────────────────────────────────────────

def _parse_verdict(raw: str) -> str:
    upper = raw.strip().upper()
    if "PASS" in upper:
        return "PASS"
    if "FAIL" in upper:
        return "FAIL"
    return "FAIL"


def judge_single_gemini(question: str, reference: str, candidate: str, model: str) -> dict[str, Any]:
    if not candidate.strip():
        return {"verdict": "FAIL", "raw": "(empty answer)"}
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=reference, candidate=candidate,
    )
    result = generate_text(
        prompt=prompt, system_instruction=JUDGE_SYSTEM, model=model, temperature=0.0,
    )
    raw = result.answer or ""
    return {"verdict": _parse_verdict(raw), "raw": raw.strip()}


def judge_single_xai(question: str, reference: str, candidate: str, model: str) -> dict[str, Any]:
    """Call xAI Grok as judge — different model family, eliminates circular bias."""
    if not candidate.strip():
        return {"verdict": "FAIL", "raw": "(empty answer)"}
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=reference, candidate=candidate,
    )
    try:
        result = generate_text(
            prompt=prompt, system_instruction=JUDGE_SYSTEM, model=model,
            temperature=0.0, provider="xai",
        )
        raw = result.answer or ""
    except Exception as exc:
        return {"verdict": "FAIL", "raw": f"xAI error: {exc}"}
    return {"verdict": _parse_verdict(raw), "raw": raw.strip()}


def judge_single(
    question: str,
    reference: str,
    candidate: str,
    model: str,
    provider: str = "gemini",
) -> dict[str, Any]:
    if provider == "xai":
        return judge_single_xai(question, reference, candidate, model)
    return judge_single_gemini(question, reference, candidate, model)


def run_judge(
    rows: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    model: str,
    delay_sec: float = 0.5,
    provider: str = "gemini",
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        qid = str(row.get("question_id", ""))
        q = questions.get(qid, {})
        reference = str(q.get("reference_answer", ""))
        candidate = str(row.get("answer", ""))
        verdict_info = judge_single(
            question=str(q.get("question", "")),
            reference=reference,
            candidate=candidate,
            model=model,
            provider=provider,
        )
        results.append({
            "question_id": qid,
            "question_type": q.get("type", "unknown"),
            "verdict": verdict_info["verdict"],
            "judge_raw": verdict_info["raw"],
        })
        print(f"  {qid}: {verdict_info['verdict']}")
        if delay_sec:
            time.sleep(delay_sec)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# BERTScore
# ──────────────────────────────────────────────────────────────────────────────

def run_bertscore(
    rows: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    model_type: str = "distilbert-base-uncased",
    rescale: bool = True,
) -> dict[str, Any]:
    """
    Compute BERTScore F1 for a list of pipeline result rows.

    model_type options (in order of quality vs. speed):
        'distilbert-base-uncased'          fast, lightweight
        'roberta-large'                     better quality
        'microsoft/deberta-xlarge-mnli'     best (hackathon spec), slow
    """
    try:
        from bert_score import score as _bert_score  # type: ignore
    except ImportError:
        return {
            "error": "bert-score not installed. Run: pip install bert-score",
            "f1_rescaled": None,
            "f1_raw": None,
            "per_question": [],
        }

    candidates: list[str] = []
    references: list[str] = []
    question_ids: list[str] = []

    for row in rows:
        qid = str(row.get("question_id", ""))
        q = questions.get(qid, {})
        candidates.append(str(row.get("answer", "")))
        references.append(str(q.get("reference_answer", "")))
        question_ids.append(qid)

    if not candidates:
        return {"f1_rescaled": None, "f1_raw": None, "per_question": []}

    # DeBERTa's tokenizer_config has model_max_length=1e30 which overflows Rust.
    # score.py imports get_tokenizer by value at load time so patching
    # bert_score.utils.get_tokenizer has no effect — instead we patch
    # bert_score.utils.sent_encode which IS resolved by module-name lookup.
    if "deberta" in model_type.lower():
        try:
            import bert_score.utils as _bsu
            _orig_sent_encode = _bsu.sent_encode

            def _capped_sent_encode(tokenizer, a):
                if getattr(tokenizer, "model_max_length", 0) > 512:
                    tokenizer.model_max_length = 512
                return _orig_sent_encode(tokenizer, a)

            _bsu.sent_encode = _capped_sent_encode
        except Exception:
            pass

    _, _, F1_raw = _bert_score(
        candidates, references, lang="en",
        model_type=model_type,
        rescale_with_baseline=False,
        verbose=False,
    )

    if rescale:
        _, _, F1_rescaled = _bert_score(
            candidates, references, lang="en",
            model_type=model_type,
            rescale_with_baseline=True,
            verbose=False,
        )
        f1_rescaled_mean = round(float(F1_rescaled.mean()), 4)
        per_rescaled = [round(float(v), 4) for v in F1_rescaled.tolist()]
    else:
        f1_rescaled_mean = None
        per_rescaled = [None] * len(candidates)

    f1_raw_mean = round(float(F1_raw.mean()), 4)
    per_raw = [round(float(v), 4) for v in F1_raw.tolist()]

    per_question = [
        {
            "question_id": qid,
            "f1_raw": r,
            "f1_rescaled": rs,
        }
        for qid, r, rs in zip(question_ids, per_raw, per_rescaled)
    ]

    return {
        "model_type": model_type,
        "f1_raw": f1_raw_mean,
        "f1_rescaled": f1_rescaled_mean,
        "bonus_bertscore_rescaled_met": (
            f1_rescaled_mean is not None and f1_rescaled_mean >= 0.55
        ),
        "bonus_bertscore_raw_met": f1_raw_mean >= 0.88,
        "per_question": per_question,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline(
    name: str,
    rows: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    model: str,
    skip_bertscore: bool,
    bertscore_model: str,
    delay_sec: float,
    judge_provider: str = "gemini",
) -> dict[str, Any]:
    if not rows:
        return {"pipeline": name, "error": "No results found"}

    print(f"\n[Judge:{judge_provider}] {name} ({len(rows)} answers)...")
    judge_results = run_judge(rows, questions, model=model, delay_sec=delay_sec, provider=judge_provider)
    pass_count = sum(1 for r in judge_results if r["verdict"] == "PASS")
    pass_rate = round(pass_count / len(judge_results), 4) if judge_results else 0.0

    bertscore_results: dict[str, Any] = {}
    if not skip_bertscore:
        print(f"[BERTScore] {name}...")
        bertscore_results = run_bertscore(
            rows, questions,
            model_type=bertscore_model,
            rescale=True,
        )

    return {
        "pipeline": name,
        "questions_evaluated": len(rows),
        "judge": {
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "bonus_judge_met": pass_rate >= 0.90,
            "per_question": judge_results,
        },
        "bertscore": bertscore_results if not skip_bertscore else {"skipped": True},
    }


def build_report(
    pipelines: list[str],
    questions_path: Path,
    results_paths: dict[str, Path],
    model: str,
    skip_bertscore: bool,
    bertscore_model: str,
    delay_sec: float,
    judge_provider: str = "gemini",
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    report: dict[str, Any] = {
        "metadata": {
            "judge_provider": judge_provider,
            "judge_model": model,
            "bertscore_model": bertscore_model,
            "bertscore_skipped": skip_bertscore,
        },
        "pipelines": {},
    }

    for name in pipelines:
        path = results_paths.get(name)
        if path is None:
            print(f"Skipping {name}: no results path configured.")
            continue
        rows = load_jsonl(path)
        if not rows:
            print(f"Skipping {name}: result file missing or empty ({path}).")
            continue
        report["pipelines"][name] = evaluate_pipeline(
            name=name,
            rows=rows,
            questions=questions,
            model=model,
            skip_bertscore=skip_bertscore,
            bertscore_model=bertscore_model,
            delay_sec=delay_sec,
            judge_provider=judge_provider,
        )

    # Cross-pipeline comparison
    report["comparison"] = {}
    for name, data in report["pipelines"].items():
        report["comparison"][name] = {
            "judge_pass_rate": data.get("judge", {}).get("pass_rate"),
            "bonus_judge_met": data.get("judge", {}).get("bonus_judge_met"),
            "bertscore_f1_raw": data.get("bertscore", {}).get("f1_raw"),
            "bertscore_f1_rescaled": data.get("bertscore", {}).get("f1_rescaled"),
            "bonus_bertscore_raw_met": data.get("bertscore", {}).get("bonus_bertscore_raw_met"),
            "bonus_bertscore_rescaled_met": data.get("bertscore", {}).get("bonus_bertscore_rescaled_met"),
        }

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def _check(value: bool | None) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "N/A"


def build_markdown(report: dict[str, Any]) -> str:
    lines = ["# Accuracy Evaluation Report", ""]
    metadata = report.get("metadata", {})
    lines.append("## Summary")
    lines.append("")
    lines.append("| Pipeline | Judge pass rate | Bonus judge (>=90%) | BERTScore F1 raw | Bonus raw (>=0.88) | BERTScore F1 rescaled | Bonus rescaled (>=0.55) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, comp in report.get("comparison", {}).items():
        lines.append(
            f"| {name}"
            f" | {comp.get('judge_pass_rate', 'N/A')}"
            f" | {_check(comp.get('bonus_judge_met'))}"
            f" | {comp.get('bertscore_f1_raw', 'N/A')}"
            f" | {_check(comp.get('bonus_bertscore_raw_met'))}"
            f" | {comp.get('bertscore_f1_rescaled', 'N/A')}"
            f" | {_check(comp.get('bonus_bertscore_rescaled_met'))}"
            " |"
        )

    lines.append("")
    lines.append("## Per-Pipeline Judge Details")
    for name, data in report.get("pipelines", {}).items():
        judge = data.get("judge", {})
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Pass rate: **{judge.get('pass_rate', 'N/A')}**  "
                     f"({judge.get('pass_count', 0)}/{data.get('questions_evaluated', 0)})  "
                     f"Bonus threshold ≥ 90%: **{_check(judge.get('bonus_judge_met'))}**")
        lines.append("")
        lines.append("| Question ID | Type | Verdict |")
        lines.append("|---|---|---|")
        for row in judge.get("per_question", []):
            lines.append(f"| {row['question_id']} | {row['question_type']} | {row['verdict']} |")

    lines.append("")
    lines.append("---")
    judge_provider = metadata.get("judge_provider", "gemini")
    judge_model = metadata.get("judge_model", DEFAULT_MODEL)
    bertscore_model = metadata.get("bertscore_model", "distilbert-base-uncased")
    if judge_provider == "xai":
        judge_label = f"xAI LLM-as-a-Judge ({judge_model})"
    else:
        judge_label = f"Gemini LLM-as-a-Judge ({judge_model})"
    lines.append(
        f"*Judge: {judge_label} with the PASS/FAIL grading prompt. "
        f"BERTScore model: {bertscore_model}.*"
    )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipeline accuracy: LLM-as-a-Judge + BERTScore.")
    parser.add_argument(
        "--pipeline",
        choices=list(DEFAULT_RESULTS.keys()) + ["all"],
        default="all",
        help="Which pipeline(s) to evaluate (default: all).",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for the judge.")
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore (saves time; requires bert-score package).",
    )
    parser.add_argument(
        "--bertscore-model",
        default="distilbert-base-uncased",
        help="Transformers model for BERTScore (default: distilbert-base-uncased).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between judge calls (avoids rate limits).",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["gemini", "xai"],
        default="gemini",
        help="Judge backend: 'gemini' (default final benchmark path) or 'xai'.",
    )
    parser.add_argument(
        "--xai-judge-model",
        default=DEFAULT_XAI_JUDGE_MODEL,
        help="xAI model to use when --judge-provider=xai.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipelines = list(DEFAULT_RESULTS.keys()) if args.pipeline == "all" else [args.pipeline]

    if args.judge_provider == "xai":
        judge_model = args.xai_judge_model
    else:
        judge_model = args.model

    report = build_report(
        pipelines=pipelines,
        questions_path=args.questions,
        results_paths=DEFAULT_RESULTS,
        model=judge_model,
        skip_bertscore=args.skip_bertscore,
        bertscore_model=args.bertscore_model,
        delay_sec=args.delay,
        judge_provider=args.judge_provider,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report written to {args.output}")

    md = build_markdown(report)
    args.md_output.write_text(md, encoding="utf-8")
    print(f"Markdown report written to {args.md_output}")

    # Print summary
    print("\n=== ACCURACY SUMMARY ===")
    for name, comp in report.get("comparison", {}).items():
        print(f"  {name}: judge={comp.get('judge_pass_rate')} | bertscore_raw={comp.get('bertscore_f1_raw')} | bertscore_rescaled={comp.get('bertscore_f1_rescaled')}")


if __name__ == "__main__":
    main()
