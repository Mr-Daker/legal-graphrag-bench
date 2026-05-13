from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm

try:
    import tiktoken
except ImportError:  # pragma: no cover - requirements install tiktoken.
    tiktoken = None


DATASET_ID = "harvard-lil/cold-cases"
DEFAULT_OUTPUT = Path("data/raw/cold_cases/opinions.jsonl")
DEFAULT_REPORT = Path("data/reports/corpus_token_report.json")
DEFAULT_TARGET_TOKENS = 2_200_000


def get_encoder():
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder: Any) -> int:
    if encoder is None:
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))
    return len(encoder.encode(text))


def clean_text(value: str) -> str:
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "html.parser").get_text(" ")
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def iter_string_fields(obj: Any, prefix: str = ""):
    if isinstance(obj, str):
        yield prefix.rstrip("."), obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_string_fields(value, f"{prefix}{key}.")
    elif isinstance(obj, list):
        for index, value in enumerate(obj[:20]):
            yield from iter_string_fields(value, f"{prefix}{index}.")


def choose_text_field(row: dict[str, Any]) -> tuple[str, str]:
    candidates = []
    for field, value in iter_string_fields(row):
        cleaned = clean_text(value)
        if len(cleaned) < 500:
            continue
        lowered = field.lower()
        score = len(cleaned)
        if any(term in lowered for term in ("text", "opinion", "casebody", "html", "plain")):
            score += 50_000
        candidates.append((score, field, cleaned))

    if not candidates:
        return "", ""

    _, field, text = max(candidates, key=lambda item: item[0])
    return field, text


def compact_metadata(row: dict[str, Any], selected_field: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"selected_text_field": selected_field}
    for key in (
        "id",
        "name",
        "case_name",
        "court",
        "court_id",
        "date_filed",
        "decision_date",
        "citation",
        "citations",
        "docket_number",
    ):
        if key in row and isinstance(row[key], (str, int, float, bool, type(None), list, dict)):
            metadata[key] = row[key]
    return metadata


def download_sample(
    output_path: Path,
    report_path: Path,
    target_tokens: int,
    max_records: int | None,
    split: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder()
    dataset = load_dataset(DATASET_ID, split=split, streaming=True)

    total_tokens = 0
    written_records = 0
    scanned_records = 0
    skipped_records = 0

    with output_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=target_tokens, unit="tok", desc="Collecting legal text")
        for row in dataset:
            scanned_records += 1
            field, text = choose_text_field(row)
            if not text:
                skipped_records += 1
                continue

            token_count = count_tokens(text, encoder)
            record = {
                "id": f"cold_cases_{scanned_records:08d}",
                "source_dataset": DATASET_ID,
                "record_index": scanned_records,
                "token_count": token_count,
                "text": text,
                "metadata": compact_metadata(row, field),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_tokens += token_count
            written_records += 1
            progress.update(token_count)

            if total_tokens >= target_tokens:
                break
            if max_records is not None and written_records >= max_records:
                break
        progress.close()

    report = {
        "dataset_id": DATASET_ID,
        "split": split,
        "output_path": str(output_path),
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "minimum_required_tokens": 2_000_000,
        "meets_round_1_requirement": total_tokens >= 2_000_000,
        "written_records": written_records,
        "scanned_records": scanned_records,
        "skipped_records": skipped_records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a local sample from harvard-lil/cold-cases.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = download_sample(
        output_path=args.output,
        report_path=args.report,
        target_tokens=args.target_tokens,
        max_records=args.max_records,
        split=args.split,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

