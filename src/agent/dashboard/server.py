"""Local, read-only trace dashboard.

    uv run python -m agent.dashboard [--trace-dir out/traces] [--port 8765]

Binds to 127.0.0.1 only: traces contain repository source and full LLM prompts, so they
are never served beyond this machine. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from agent.tracing import NodeRecord, RunEnd, RunStart, read_trace, validate_trace

HOST = "127.0.0.1"
RUN_ID_RE = re.compile(r"^[\w.-]+$")
SCRIPTED_MODELS = frozenset({"fake-llm"})
_STATIC = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


# --------------------------------------------------------------------------- data


def _records(path: Path) -> tuple[list[RunStart | NodeRecord | RunEnd], str | None]:
    try:
        return read_trace(path), None
    except (ValidationError, ValueError) as exc:
        return [], f"unparseable trace: {str(exc).splitlines()[0][:200]}"


def summarize_run(path: Path) -> dict[str, Any]:
    """One row of the runs table. Never raises on a malformed trace."""
    records, parse_error = _records(path)
    problems = [parse_error] if parse_error else validate_trace(path)
    start = next((r for r in records if isinstance(r, RunStart)), None)
    end = next((r for r in records if isinstance(r, RunEnd)), None)
    cfg: dict[str, Any] = start.config if start else {}
    critic = cfg.get("critic") or {}
    budget = cfg.get("budget") or {}
    model = cfg.get("model")
    out: dict[str, Any] = {
        "run_id": path.stem,
        "ts": start.ts if start else None,
        "instance_id": start.instance_id if start else None,
        "variant": cfg.get("variant"),
        "model": model,
        "scripted": model in SCRIPTED_MODELS,
        "effort": cfg.get("effort"),
        "K": cfg.get("K"),
        "D": cfg.get("D"),
        "N": cfg.get("N"),
        "seed": cfg.get("seed"),
        "critic_enabled": critic.get("enabled"),
        "critic_uses_sandbox": critic.get("uses_sandbox"),
        "budget_per_run_usd": budget.get("per_run_usd"),
        "config_hash": start.config_hash if start else None,
        "git_sha": start.git_sha if start else None,
        "git_dirty": start.git_dirty if start else None,
        "status": end.status if end else "incomplete",
        "attempts": end.attempts if end else None,
        "reflection_depth": end.reflection_depth if end else None,
        "wall_time_s": end.wall_time_s if end else None,
        "error": end.error if end else None,
        "trace_valid": not problems,
        "problems": problems,
    }
    totals = end.totals.model_dump() if end else {}
    for key in ("cost_usd", "tokens_in", "tokens_out", "llm_calls", "sandbox_runs", "nodes"):
        out[key] = totals.get(key)
    return out


def list_runs(trace_dir: Path) -> list[dict[str, Any]]:
    runs = [summarize_run(p) for p in trace_dir.glob("*.jsonl") if RUN_ID_RE.match(p.stem)]
    return sorted(runs, key=lambda r: r["ts"] or "", reverse=True)


def load_run(trace_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = trace_dir / f"{run_id}.jsonl"
    if not path.is_file():
        return None
    records, _ = _records(path)
    end = next((r for r in records if isinstance(r, RunEnd)), None)
    final_state = None
    if end is not None and end.final_state is not None:
        state_path = _safe_path(trace_dir, end.final_state)
        if state_path is not None:
            final_state = json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "summary": summarize_run(path),
        "records": [r.model_dump(mode="json") for r in records],
        "final_state": final_state,
    }


def _safe_path(trace_dir: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``trace_dir``; None if it escapes or is not a JSON file."""
    if not rel or rel.startswith(("/", "\\")) or "\x00" in rel:
        return None
    root = trace_dir.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or target.suffix != ".json" or not target.is_file():
        return None
    return target


# --------------------------------------------------------------------------- HTTP


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AblationDashboard/1"

    def __init__(self, *args: Any, trace_dir: Path, **kwargs: Any) -> None:
        self.trace_dir = trace_dir
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # quiet; errors still surface as HTTP status codes

    def do_GET(self) -> None:
        url = urlparse(self.path)
        route = url.path
        try:
            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route.startswith("/static/"):
                self._static(route.removeprefix("/static/"))
            elif route == "/api/runs":
                self._json(list_runs(self.trace_dir))
            elif route.startswith("/api/runs/"):
                self._run(route.removeprefix("/api/runs/"))
            elif route == "/api/artifact":
                self._artifact(parse_qs(url.query).get("path", [""])[0])
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:  # one bad file must not take the server down
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _run(self, run_id: str) -> None:
        if not RUN_ID_RE.match(run_id):
            self._error(HTTPStatus.BAD_REQUEST, "invalid run id")
            return
        run = load_run(self.trace_dir, run_id)
        if run is None:
            self._error(HTTPStatus.NOT_FOUND, "run not found")
        else:
            self._json(run)

    def _artifact(self, rel: str) -> None:
        target = _safe_path(self.trace_dir, rel)
        if target is None:
            self._error(HTTPStatus.NOT_FOUND, "artifact not found")
        else:
            self._send(HTTPStatus.OK, "application/json", target.read_bytes())

    def _static(self, name: str) -> None:
        ctype = _STATIC.get(name)  # allowlist: no path ever reaches the filesystem
        if ctype is None:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        body = files("agent.dashboard").joinpath("static", name).read_bytes()
        self._send(HTTPStatus.OK, ctype, body)

    def _json(self, payload: Any) -> None:
        self._send(HTTPStatus.OK, "application/json", json.dumps(payload).encode())

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send(status, "application/json", json.dumps({"error": message}).encode())

    def _send(self, status: HTTPStatus, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()
        self.wfile.write(body)


def make_server(trace_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    handler = partial(DashboardHandler, trace_dir=trace_dir.resolve())
    return ThreadingHTTPServer((HOST, port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local, read-only trace dashboard.")
    ap.add_argument("--trace-dir", type=Path, default=Path("out/traces"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if not args.trace_dir.is_dir():
        ap.error(f"trace dir does not exist: {args.trace_dir}")
    server = make_server(args.trace_dir, args.port)
    host, port = server.server_address[:2]
    print(f"Dashboard: http://{host!s}:{port}  (traces: {args.trace_dir.resolve()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
