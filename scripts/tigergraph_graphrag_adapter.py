from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


OFFICIAL_GRAPHRAG_REPO = "https://github.com/tigergraph/graphrag"
OFFICIAL_GRAPHRAG_COMMIT = "f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9"
OFFICIAL_GRAPHRAG_VENDOR_DIR = Path("vendor/tigergraph-graphrag")

REQUIRED_OFFICIAL_FILES = (
    "README.md",
    "docker-compose.yml",
    "configs/server_config.json",
    "common",
    "ecc/app/graphrag/graph_rag.py",
    "ecc/app/graphrag/community_summarizer.py",
    "graphrag",
)


@dataclass(frozen=True)
class OfficialGraphRAGStatus:
    upstream: str
    commit: str
    root: str
    available: bool
    missing: list[str]
    customization: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def official_graphrag_root() -> Path:
    return repo_root() / OFFICIAL_GRAPHRAG_VENDOR_DIR


def register_official_graphrag_paths() -> Path:
    """Expose the vendored official repo to local adapters without importing heavy services."""
    root = official_graphrag_root()
    for path in (root, root / "common", root / "ecc" / "app", root / "graphrag"):
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)
    return root


def official_graphrag_status() -> OfficialGraphRAGStatus:
    root = register_official_graphrag_paths()
    missing = [item for item in REQUIRED_OFFICIAL_FILES if not (root / item).exists()]
    return OfficialGraphRAGStatus(
        upstream=OFFICIAL_GRAPHRAG_REPO,
        commit=OFFICIAL_GRAPHRAG_COMMIT,
        root=str(root),
        available=not missing,
        missing=missing,
        customization=(
            "LegalGraphRAG customizes TigerGraph GraphRAG with a legal schema, "
            "EA-GraphRAG routing, CommunityReport retrieval, and PathRAG-light pruning."
        ),
    )


def official_graphrag_metadata() -> dict[str, object]:
    return asdict(official_graphrag_status())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the official TigerGraph GraphRAG integration.")
    parser.add_argument("--check", action="store_true", help="Print JSON integration status.")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(official_graphrag_metadata(), indent=2))


if __name__ == "__main__":
    main()
