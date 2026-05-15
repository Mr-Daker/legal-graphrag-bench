# TigerGraph GraphRAG Integration

The hackathon brief requires the public submission to be built on top of
`https://github.com/tigergraph/graphrag`. This repo includes that official codebase
as a vendored foundation at:

```text
vendor/tigergraph-graphrag
```

Vendored upstream:

```text
repo   : https://github.com/tigergraph/graphrag
commit : f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9
```

## What Comes From The Official Repo

- TigerGraph GraphRAG service layout
- TigerGraph-first graph/vector retrieval architecture
- document-to-graph GraphRAG reference modules
- community summarization reference flow
- deployment/config examples for GraphRAG services

## LegalGraphRAG Customizations

The root project layers the legal benchmark implementation on top:

- `scripts/export_tigergraph_csv.py` exports the legal graph schema
- `scripts/graphrag_tigergraph.py` runs Pipeline 3 retrieval and generation
- `scripts/tigergraph_graphrag_adapter.py` validates and records the official base
- `backend/server.js` exposes live query execution
- `frontend/` renders the benchmark and architecture dashboard

Schema additions:

- vertices: `LegalCase`, `Chunk`, `Citation`, `Entity`, `CommunityReport`
- edges: `HAS_CHUNK`, `NEXT_CHUNK`, `CITES`, `MENTIONS`, `RELATED_TO`

Retrieval additions:

- EA-GraphRAG-style query routing
- CommunityReport path for global synthesis questions
- entity/path traversal for local and multi-hop questions
- PathRAG-light scoring: relevance x edge weight x hop penalty
- pruned context under a fixed token budget

## Verification

Run:

```powershell
.\.venv-win\Scripts\python.exe scripts\tigergraph_graphrag_adapter.py --check
```

Expected result:

```json
{
  "upstream": "https://github.com/tigergraph/graphrag",
  "commit": "f649f4197f3dc18bf5bd7dd2fb0a0e477a5a70b9",
  "available": true,
  "missing": []
}
```

Every GraphRAG result emitted by `scripts/graphrag_tigergraph.py` also includes
`official_graphrag_base` metadata so the runtime output carries the upstream
provenance.
