"""Dashboard server: a real HTTP server over real traces from a scripted Docker run."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from agent.config import ExperimentConfig
from agent.dashboard.server import HOST, list_runs, make_server
from agent.instance import Instance
from agent.run import RunOutcome
from tests.conftest import CONFIGS
from tests.fake_llm import FakeLLM
from tests.scenarios import C_CORRECT, C_WRONG, arch, loc, run, score, select

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("dash_traces")


@pytest.fixture(scope="module")
def outcome(trace_dir: Path, fixture_image: str) -> RunOutcome:
    data = ExperimentConfig.from_yaml(CONFIGS / "a2.yaml").model_dump()
    data.update({"model": "fake-llm", "max_tokens": 1000, "K": 2, "trace_dir": trace_dir})
    config = ExperimentConfig.model_validate(data)
    assert config.sandbox.image == fixture_image
    instance = Instance.from_json(Path(__file__).parent / "fixtures/instances/calc_001.json")
    fake = FakeLLM({
        "localize": [loc()],
        "architect": [arch(C_WRONG, C_CORRECT)],
        "critic_select": [select(1, score(0), score(1, correctness=5))],
    })  # fmt: skip
    out = run(config, instance, fake)
    assert out.trace_problems == []
    # A corrupted trace next to the good one must be listed, not crash the server.
    (trace_dir / "corrupt-run.jsonl").write_text('{"event": "run_start", "seq": 0}\nnot json\n')
    return out


@pytest.fixture(scope="module")
def base_url(trace_dir: Path, outcome: RunOutcome) -> Iterator[str]:
    server = make_server(trace_dir, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host!s}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type", ""), err.read()


def _json(url: str) -> Any:
    status, ctype, body = _get(url)
    assert status == 200, body
    assert ctype == "application/json"
    return json.loads(body)


def test_binds_to_loopback_only(trace_dir: Path) -> None:
    server = make_server(trace_dir, port=0)
    try:
        assert server.server_address[0] == HOST == "127.0.0.1"
    finally:
        server.server_close()


def test_runs_list(base_url: str, outcome: RunOutcome) -> None:
    runs = _json(f"{base_url}/api/runs")
    by_id = {r["run_id"]: r for r in runs}
    r = by_id[outcome.run_id]
    assert r["variant"] == "A2"
    assert r["status"] == "sandbox_verified"
    assert r["trace_valid"] is True
    assert r["scripted"] is True
    assert r["critic_enabled"] is True
    assert r["K"] == 2
    assert r["llm_calls"] == 3
    assert r["cost_usd"] == pytest.approx(outcome.final_state["cost_so_far"])
    bad = by_id["corrupt-run"]
    assert bad["trace_valid"] is False
    assert bad["status"] == "incomplete"
    assert bad["problems"]


def test_run_detail_includes_final_state(base_url: str, outcome: RunOutcome) -> None:
    d = _json(f"{base_url}/api/runs/{outcome.run_id}")
    nodes = [r["node"] for r in d["records"] if r["event"] == "node"]
    assert "critic" in nodes
    assert d["records"][0]["event"] == "run_start"
    assert d["records"][-1]["event"] == "run_end"
    fs = d["final_state"]
    assert fs["selected_index"] == 1
    assert "+    if not values:" in fs["candidates"][1]["diff"]
    assert [n["candidate_index"] for n in fs["critic_notes"]] == [0, 1]


def test_artifact_served(base_url: str, outcome: RunOutcome) -> None:
    d = _json(f"{base_url}/api/runs/{outcome.run_id}")
    path = next(c["artifact"] for r in d["records"] if r["event"] == "node" for c in r["llm_calls"])
    art = _json(f"{base_url}/api/artifact?path={urllib.parse.quote(path)}")
    assert art["purpose"] == "localize"
    assert art["messages"][0]["role"] == "user"


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/api/runs/does-not-exist", 404),
        ("/api/runs/a%20b", 400),
        ("/api/runs/..%2F..%2Fetc", 400),
        ("/api/artifact?path=../../pyproject.toml", 404),
        ("/api/artifact?path=%2Fetc%2Fpasswd", 404),
        ("/api/artifact?path=", 404),
        ("/static/../server.py", 404),
        ("/static/server.py", 404),
        ("/nope", 404),
    ],
)
def test_rejected_requests(base_url: str, path: str, status: int) -> None:
    code, ctype, body = _get(base_url + path)
    assert code == status
    assert ctype == "application/json"
    assert "error" in json.loads(body)


def test_artifact_must_be_json_inside_trace_dir(base_url: str, outcome: RunOutcome) -> None:
    # The .jsonl trace itself is inside the dir but is not an artifact.
    code, _, _ = _get(f"{base_url}/api/artifact?path={outcome.run_id}.jsonl")
    assert code == 404


@pytest.mark.parametrize(
    ("path", "ctype"),
    [
        ("/", "text/html; charset=utf-8"),
        ("/static/app.js", "text/javascript; charset=utf-8"),
        ("/static/styles.css", "text/css; charset=utf-8"),
    ],
)
def test_static_files_and_headers(base_url: str, path: str, ctype: str) -> None:
    with urllib.request.urlopen(base_url + path, timeout=10) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == ctype
        assert "script-src 'self'" in resp.headers["Content-Security-Policy"]
        assert resp.headers["X-Content-Type-Options"] == "nosniff"


def test_list_runs_on_empty_dir(tmp_path: Path) -> None:
    assert list_runs(tmp_path) == []


def test_frontend_never_uses_innerhtml_with_data() -> None:
    js = files("agent.dashboard").joinpath("static", "app.js").read_text()
    for sink in (".innerHTML", ".outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in js, f"HTML/code sink {sink!r} must not be used with trace data"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_js_parses() -> None:
    path = Path(str(files("agent.dashboard").joinpath("static", "app.js")))
    res = subprocess.run(
        ["node", "--check", str(path)], check=False, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
