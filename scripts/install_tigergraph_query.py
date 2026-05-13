from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from urllib import error, parse, request

from dotenv import load_dotenv

from graphrag_tigergraph import TigerGraphClient


DEFAULT_QUERY_FILE = Path("tigergraph_graphrag_context.gsql")
DEFAULT_QUERY_NAME = "graphrag_case_context"


def extract_create_query(gsql_text: str, query_name: str) -> str:
    pattern = re.compile(
        rf"(CREATE\s+OR\s+REPLACE\s+DISTRIBUTED\s+QUERY\s+{re.escape(query_name)}\s*\(.*?\n\}})",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(gsql_text)
    if not match:
        raise ValueError(f"Could not find CREATE OR REPLACE query block for {query_name}")
    return match.group(1).strip()


def request_json(client: TigerGraphClient, method: str, path: str, body: str | None = None) -> dict:
    client.ensure_token()
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = body.encode("utf-8")
        headers["Content-Type"] = "text/plain"
    if client.token:
        headers["Authorization"] = f"Bearer {client.token}"
    username = os.getenv("TG_USERNAME")
    password = os.getenv("TG_PASSWORD")
    if username and password:
        raw = f"{username}:{password}".encode("utf-8")
        headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"
    req = request.Request(f"{client.host}{path}", data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def create_query(client: TigerGraphClient, graph_name: str, query_code: str) -> dict:
    path = f"/gsql/v1/queries?{parse.urlencode({'graph': graph_name})}"
    return request_json(client, "POST", path, body=query_code)


def install_query(client: TigerGraphClient, graph_name: str, query_name: str) -> dict:
    params = parse.urlencode({"graph": graph_name, "queries": query_name, "flag": "-force"})
    return request_json(client, "GET", f"/gsql/v1/queries/install?{params}")


def poll_install(client: TigerGraphClient, request_id: str, interval_seconds: float, timeout_seconds: float) -> dict:
    deadline = time.time() + timeout_seconds
    last_payload = {}
    while time.time() < deadline:
        payload = request_json(client, "GET", f"/gsql/v1/queries/install/{parse.quote(request_id)}")
        last_payload = payload
        text = json.dumps(payload).lower()
        if "success" in text or "installed" in text or "finished" in text:
            return payload
        if "fail" in text or "error" in text and payload.get("error") is True:
            return payload
        time.sleep(interval_seconds)
    return last_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and install the TigerGraph GraphRAG context query.")
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--query-name", default=DEFAULT_QUERY_NAME)
    parser.add_argument("--poll", action="store_true", help="Poll install status until it completes or times out.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    client = TigerGraphClient.from_env()
    if client is None:
        raise SystemExit("TG_HOST is missing in .env")

    query_code = extract_create_query(
        gsql_text=args.query_file.read_text(encoding="utf-8"),
        query_name=args.query_name,
    )
    create_payload = create_query(client, client.graph_name, query_code)
    print(json.dumps({"create": create_payload}, indent=2))

    install_payload = install_query(client, client.graph_name, args.query_name)
    print(json.dumps({"install": install_payload}, indent=2))

    request_id = str(install_payload.get("requestId") or install_payload.get("request_id") or "")
    if args.poll and request_id:
        status_payload = poll_install(client, request_id, args.interval, args.timeout)
        print(json.dumps({"install_status": status_payload}, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main()
