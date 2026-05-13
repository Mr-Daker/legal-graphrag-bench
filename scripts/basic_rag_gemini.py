from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from google.genai import types
from tqdm import tqdm

from gemini_client import DEFAULT_MODEL, compute_cost, generate_text, load_client


DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX_DIR = Path("data/index/basic_rag")
DEFAULT_OUTPUT = Path("data/results/basic_rag_results.jsonl")
DEFAULT_QUESTIONS = Path("data/eval/questions_dev.json")
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_PROVIDER = "hashing"
HASHING_DIMENSIONS = 4096
DEFAULT_TOP_K = 8
DEFAULT_BATCH_SIZE = 32

SYSTEM_PROMPT = """Answer the question using only the provided context.
If the context is insufficient, say that the answer is not available in the retrieved context.
Be concise, but include the relevant case names, courts, statutes, or legal issues when present."""

STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "case",
    "court",
    "decided",
    "does",
    "from",
    "have",
    "how",
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


def load_chunks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def batched(items: list[Any], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def tokenize_for_hashing(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}", text.lower())


def query_terms(query: str) -> list[str]:
    return [term for term in tokenize_for_hashing(query) if term not in STOPWORDS and len(term) > 2]


def hashed_embedding(text: str, dimensions: int = HASHING_DIMENSIONS) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in tokenize_for_hashing(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % dimensions
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def embed_texts_local(texts: list[str]) -> np.ndarray:
    return np.vstack([hashed_embedding(text) for text in texts])


def embed_texts(texts: list[str], model: str, task_type: str) -> np.ndarray:
    client = load_client()
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    vectors = [embedding.values for embedding in response.embeddings]
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return array / norms


def embed_texts_with_retry(texts: list[str], model: str, task_type: str, max_attempts: int = 3) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embed_texts(texts=texts, model=model, task_type=task_type)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("Embedding request failed") from last_error


def build_index(
    chunks_path: Path,
    index_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    chunks = load_chunks(chunks_path, limit=limit)
    if not chunks:
        raise RuntimeError(f"No chunks loaded from {chunks_path}")

    vectors = []
    for batch in tqdm(list(batched(chunks, batch_size)), desc="Embedding chunks"):
        texts = [item["text"] for item in batch]
        if embedding_provider == "gemini":
            vectors.append(
                embed_texts_with_retry(
                    texts=texts,
                    model=embedding_model,
                    task_type="RETRIEVAL_DOCUMENT",
                )
            )
        elif embedding_provider == "hashing":
            vectors.append(embed_texts_local(texts))
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

    embeddings = np.vstack(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )

    manifest = {
        "chunks_path": str(chunks_path),
        "index_dir": str(index_dir),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "chunks": len(chunks),
        "dimensions": int(embeddings.shape[1]),
        "limited": limit is not None,
        "limit": limit,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_index(index_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    embeddings_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"
    if not embeddings_path.exists() or not chunks_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Basic RAG index is missing in {index_dir}. Run with --build first.")

    embeddings = np.load(embeddings_path)
    chunks = load_chunks(chunks_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return embeddings, chunks, manifest


def retrieve(query: str, embeddings: np.ndarray, chunks: list[dict[str, Any]], embedding_model: str, top_k: int):
    query_vector = embed_query(query=query, embedding_model=embedding_model)
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        retrieved.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk_id": chunk["id"],
                "source_id": chunk.get("source_id"),
                "chunk_index": chunk.get("chunk_index"),
                "token_count": chunk.get("token_count"),
                "metadata": chunk.get("metadata", {}),
                "text": chunk["text"],
            }
        )
    return retrieved


def embed_query(query: str, embedding_model: str, embedding_provider: str = "gemini") -> np.ndarray:
    if embedding_provider == "gemini":
        return embed_texts_with_retry(
            texts=[query],
            model=embedding_model,
            task_type="RETRIEVAL_QUERY",
        )[0]
    if embedding_provider == "hashing":
        return hashed_embedding(query)
    raise ValueError(f"Unsupported embedding provider: {embedding_provider}")


def lexical_scores(query: str, chunks: list[dict[str, Any]]) -> np.ndarray:
    terms = query_terms(query)
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata") or {}
        haystack = " ".join(
            str(value or "")
            for value in (
                chunk.get("text", ""),
                metadata.get("case_name", ""),
                " ".join(metadata.get("citations", [])) if isinstance(metadata.get("citations"), list) else metadata.get("citations", ""),
            )
        ).lower()
        matches = sum(1 for term in terms if term in haystack)
        if matches:
            scores[index] = matches / len(terms)
    return scores


def build_prompt(query: str, retrieved: list[dict[str, Any]]) -> str:
    context_blocks = []
    for item in retrieved:
        metadata = item.get("metadata") or {}
        case_name = metadata.get("case_name", "unknown case")
        citations = metadata.get("citations", [])
        citation_text = ", ".join(citations) if isinstance(citations, list) else str(citations)
        context_blocks.append(
            f"[Chunk {item['rank']} | score={item['score']:.4f} | case={case_name} | citations={citation_text}]\n"
            f"{item['text']}"
        )
    context = "\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def answer_query(
    query: str,
    question_id: str,
    index_dir: Path,
    top_k: int,
    generation_model: str,
) -> dict[str, Any]:
    embeddings, chunks, manifest = load_index(index_dir)
    embedding_model = manifest["embedding_model"]
    embedding_provider = manifest.get("embedding_provider", "gemini")

    started = time.perf_counter()
    query_vector = embed_query(
        query=query,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
    )
    dense_scores = embeddings @ query_vector
    scores = dense_scores + (0.35 * lexical_scores(query, chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        retrieved.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk_id": chunk["id"],
                "source_id": chunk.get("source_id"),
                "chunk_index": chunk.get("chunk_index"),
                "token_count": chunk.get("token_count"),
                "metadata": chunk.get("metadata", {}),
                "text": chunk["text"],
            }
        )
    prompt = build_prompt(query=query, retrieved=retrieved)
    result = generate_text(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        model=generation_model,
        temperature=0.0,
    )
    end_to_end_ms = (time.perf_counter() - started) * 1000

    return {
        "pipeline": "basic_rag",
        "question_id": question_id,
        "question": query,
        "answer": result.answer,
        "generation_model": result.model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": round(end_to_end_ms, 2),
        "generation_latency_ms": round(result.latency_ms, 2),
        "cost_usd": compute_cost(result.model, result.prompt_tokens, result.completion_tokens),
        "retrieved_context": retrieved,
    }


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def run_questions(
    questions_path: Path,
    output_path: Path,
    index_dir: Path,
    top_k: int,
    generation_model: str,
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
                index_dir=index_dir,
                top_k=top_k,
                generation_model=generation_model,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            print(f"{question_id}: {record['total_tokens']} tokens, {record['latency_ms']} ms")
            time.sleep(5)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run Pipeline 2: Gemini Basic RAG.")
    parser.add_argument("--build", action="store_true", help="Build the embedding index.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "gemini"],
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Use local hashing vectors now, or Gemini embeddings when quota is available.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N chunks.")
    parser.add_argument("--query", help="Run one ad hoc query.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        manifest = build_index(
            chunks_path=args.chunks,
            index_dir=args.index_dir,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(json.dumps(manifest, indent=2))

    if args.query:
        record = answer_query(
            query=args.query,
            question_id="adhoc",
            index_dir=args.index_dir,
            top_k=args.top_k,
            generation_model=args.model,
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    if not args.build:
        run_questions(
            questions_path=args.questions,
            output_path=args.output,
            index_dir=args.index_dir,
            top_k=args.top_k,
            generation_model=args.model,
        )


if __name__ == "__main__":
    main()
