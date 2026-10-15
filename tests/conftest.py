from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import docker
import pytest

from agent.config import REPO_ROOT, ExperimentConfig
from agent.instance import Instance
from agent.sandbox.docker_sandbox import DockerSandbox, build_image

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_IMAGE = "ablation-fixture:py3.12"
CONFIGS = REPO_ROOT / "configs"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        help="run tests that call the real Anthropic API (costs money)",
    )
    parser.addoption(
        "--run-swebench",
        action="store_true",
        help="run tests that pull real SWE-bench images (GBs of disk; see docs/EVAL_SETUP.md)",
    )
    parser.addoption(
        "--run-github",
        action="store_true",
        help="run READ-ONLY tests against the real GitHub API (needs GITHUB_TOKEN)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--run-live")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    run_swebench = config.getoption("--run-swebench")
    for item in items:
        if "github" in item.keywords and not (
            config.getoption("--run-github") and os.environ.get("GITHUB_TOKEN")
        ):
            item.add_marker(pytest.mark.skip(reason="GitHub API; --run-github + GITHUB_TOKEN"))
        if "swebench" in item.keywords and not run_swebench:
            item.add_marker(pytest.mark.skip(reason="real SWE-bench images; pass --run-swebench"))
        if "live" in item.keywords:
            if not run_live:
                item.add_marker(pytest.mark.skip(reason="live API test; pass --run-live"))
            elif not has_key:
                item.add_marker(pytest.mark.skip(reason="ANTHROPIC_API_KEY not set"))


@pytest.fixture(scope="session")
def docker_client() -> Any:
    # Docker is a hard requirement: fail loudly rather than skip.
    client = docker.from_env()
    client.ping()
    return client


@pytest.fixture(scope="session")
def fixture_image(docker_client: Any) -> str:
    build_image(FIXTURES / "docker", FIXTURE_IMAGE, client=docker_client)
    return FIXTURE_IMAGE


@pytest.fixture
def sandbox(fixture_image: str, docker_client: Any, instance: Instance) -> Iterator[DockerSandbox]:
    sb = DockerSandbox(fixture_image, client=docker_client, run_label="pytest")
    sb.create(instance.repo, instance.base_commit)
    try:
        yield sb
    finally:
        sb.destroy()


@pytest.fixture
def instance() -> Instance:
    return Instance.from_json(FIXTURES / "instances" / "calc_001.json")


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., ExperimentConfig]:
    """Load a shipped config and apply overrides; traces go to a per-test tmp dir."""

    def _make(variant: str = "A0", **overrides: Any) -> ExperimentConfig:
        base = ExperimentConfig.from_yaml(CONFIGS / f"{variant.lower()}.yaml")
        data = base.model_dump()
        data["trace_dir"] = tmp_path / "traces"
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return ExperimentConfig.model_validate(data)

    return _make


@pytest.fixture
def cfg(make_config: Callable[..., ExperimentConfig]) -> Callable[..., ExperimentConfig]:
    """Like make_config, but priced as the FakeLLM with a small max_tokens."""

    def _make(variant: str = "A0", **kw: Any) -> ExperimentConfig:
        kw.setdefault("model", "fake-llm")
        kw.setdefault("max_tokens", 1000)
        return make_config(variant, **kw)

    return _make
