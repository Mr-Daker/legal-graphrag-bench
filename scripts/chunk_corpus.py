from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover - requirements install tiktoken.
    tiktoken = None


DEFAULT_INPUT = Path("data/raw/cold_cases/opinions.jsonl")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")
DEFAULT_REPORT = Path("data/reports/chunk_report.json")
DEFAULT_CHUNK_SIZE = 384
DEFAULT_OVERLAP = 64


def get_encoder():
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def encode_text(text: str, encoder: Any) -> list[Any]:
    if encoder is None:
        return text.split()
    return encoder.encode(text)


def decode_tokens(tokens: list[Any], encoder: Any) -> str:
    if encoder is None:
        return " ".join(tokens)
    return encoder.decode(tokens)


def stable_chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}:{chunk_index}:{text[:200]}".encode("utf-8")).hexdigest()
    return digest[:16]


def iter_chunks(tokens: list[Any], chunk_size: int, overlap: int):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        yield start, end, chunk_tokens
        if end >= len(tokens):
            break


def chunk_corpus(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder()
    documents = 0
    chunks = 0
    input_tokens = 0
    chunked_tokens = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            document = json.loads(line)
            text = document.get("text", "")
            if not text:
                continue

            documents += 1
            tokens = encode_text(text, encoder)
            input_tokens += len(tokens)
            source_id = str(document.get("id") or f"doc_{documents:08d}")

            for chunk_index, start, end, chunk_tokens in (
                (index, start, end, chunk_tokens)
                for index, (start, end, chunk_tokens) in enumerate(
                    iter_chunks(tokens, chunk_size=chunk_size, overlap=overlap)
                )
            ):
                chunk_text = decode_tokens(chunk_tokens, encoder).strip()
                if not chunk_text:
                    continue

                record = {
                    "id": stable_chunk_id(source_id, chunk_index, chunk_text),
                    "source_id": source_id,
                    "source_dataset": document.get("source_dataset"),
                    "chunk_index": chunk_index,
                    "token_start": start,
                    "token_end": end,
                    "token_count": len(chunk_tokens),
                    "text": chunk_text,
                    "metadata": document.get("metadata", {}),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks += 1
                chunked_tokens += len(chunk_tokens)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "documents": documents,
        "chunks": chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "input_tokens": input_tokens,
        "chunked_tokens_including_overlap": chunked_tokens,
        "minimum_required_tokens": 2_000_000,
        "meets_round_1_requirement": input_tokens >= 2_000_000,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk local corpus JSONL for RAG pipelines.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = chunk_corpus(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

