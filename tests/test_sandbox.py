"""Real Docker sandbox tests. Nothing here is mocked."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from docker.errors import NotFound

from agent.instance import Instance
from agent.sandbox import DockerSandbox, E2BSandbox, SandboxError
from agent.sandbox.base import safe_relpath
from agent.sandbox.docker_sandbox import build_image
from tests.conftest import FIXTURES

pytestmark = pytest.mark.docker


def test_create_resolves_commit_and_copies_repo(sandbox: DockerSandbox) -> None:
    assert sandbox.base_commit is not None
    assert len(sandbox.base_commit) == 40
    files = sandbox.run("git ls-files", 30).stdout.split()
    assert "calc/stats.py" in files
    assert "tests/test_stats.py" in files
    assert not any("__pycache__" in f for f in files)
    assert sandbox.read_diff() == ""


def test_fixture_bug_reproduces(sandbox: DockerSandbox, instance: Instance) -> None:
    res = sandbox.run(instance.test_command, 120)
    assert res.exit_code == 1
    assert not res.timed_out
    assert "ZeroDivisionError" in res.stdout
    assert "1 failed, 2 passed" in res.stdout


def test_write_read_diff_and_reset(sandbox: DockerSandbox) -> None:
    original = sandbox.read_file("calc/stats.py")
    sandbox.write("calc/stats.py", original.replace("Small statistics", "Tiny statistics"))
    sandbox.write("calc/new_mod/extra.py", "X = 1\n")
    diff = sandbox.read_diff()
    assert '-"""Small statistics helpers."""' in diff
    assert '+"""Tiny statistics helpers."""' in diff
    assert "calc/new_mod/extra.py" in diff  # untracked files are included
    assert sandbox.read_base_file("calc/stats.py") == original  # base view unaffected
    sandbox.reset_to_base()
    assert sandbox.read_diff() == ""
    assert sandbox.read_file("calc/stats.py") == original


def test_exit_codes_and_streams(sandbox: DockerSandbox) -> None:
    res = sandbox.run("echo out; echo err >&2; exit 3", 30)
    assert (res.exit_code, res.stdout, res.stderr, res.ok) == (3, "out\n", "err\n", False)
    assert sandbox.run("true", 30).ok


def test_timeout_kills_command(sandbox: DockerSandbox) -> None:
    res = sandbox.run("sleep 30", 2)
    assert res.timed_out
    assert not res.ok
    assert res.duration_s < 10


def test_network_is_disabled(sandbox: DockerSandbox) -> None:
    res = sandbox.run(
        "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\"",
        30,
    )
    assert res.exit_code != 0
    assert "OSError" in res.stderr or "unreachable" in res.stderr.lower()


@pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.txt", "a/../../b", "", "."])
def test_path_escape_rejected(sandbox: DockerSandbox, bad: str) -> None:
    with pytest.raises(SandboxError):
        sandbox.write(bad, "x")


def test_safe_relpath_normalizes() -> None:
    assert safe_relpath("calc/./stats.py") == "calc/stats.py"
    assert safe_relpath("a/b/../c.py") == "a/c.py"


def test_observer_sees_every_exec(
    fixture_image: str, docker_client: Any, instance: Instance
) -> None:
    seen: list[tuple[str, int | None, bool]] = []
    with DockerSandbox(
        fixture_image, client=docker_client,
        observer=lambda cmd, code, _ms, to: seen.append((cmd, code, to)),
    ) as sb:  # fmt: skip
        sb.create(instance.repo, instance.base_commit)
        n_setup = len(seen)
        sb.run("exit 7", 10)
    assert n_setup > 0  # setup commands are observed too
    assert seen[-1] == ("exit 7", 7, False)


def test_destroy_idempotent_and_removes_container(
    fixture_image: str, docker_client: Any, instance: Instance
) -> None:
    sb = DockerSandbox(fixture_image, client=docker_client)
    sb.create(instance.repo, instance.base_commit)
    cid = sb._require().id
    sb.destroy()
    sb.destroy()
    with pytest.raises(NotFound):
        docker_client.containers.get(cid)
    with pytest.raises(SandboxError, match="not created"):
        sb.run("true", 5)


def test_context_manager_cleans_up_on_exception(
    fixture_image: str, docker_client: Any, instance: Instance
) -> None:
    cid = None
    try:
        with DockerSandbox(fixture_image, client=docker_client) as sb:
            sb.create(instance.repo, instance.base_commit)
            cid = sb._require().id
            raise RuntimeError("boom")
    except RuntimeError as exc:
        assert str(exc) == "boom"  # noqa: PT017
    assert cid is not None
    with pytest.raises(NotFound):
        docker_client.containers.get(cid)


def test_failed_create_does_not_leak(
    fixture_image: str, docker_client: Any, instance: Instance
) -> None:
    label = "pytest-badcommit"
    sb = DockerSandbox(fixture_image, client=docker_client, run_label=label)
    with pytest.raises(SandboxError):
        sb.create(instance.repo, "deadbeef" * 5)
    leaked = docker_client.containers.list(
        all=True, filters={"label": f"ablation.sandbox.run={label}"}
    )
    assert leaked == []


def test_prebuilt_mode_requires_repo_in_image(fixture_image: str, docker_client: Any) -> None:
    # A non-local repo means "already inside the image"; the plain fixture image has none.
    with pytest.raises(SandboxError, match="no git repo"):
        DockerSandbox(fixture_image, client=docker_client).create("owner/name", "HEAD")


@pytest.fixture(scope="module")
def prebuilt_image(fixture_image: str, docker_client: Any) -> str:
    tag = "ablation-fixture-prebuilt:py3.12"
    build_image(FIXTURES, tag, client=docker_client, dockerfile="docker/prebuilt.Dockerfile")
    return tag


@pytest.fixture
def prebuilt(prebuilt_image: str, docker_client: Any) -> Iterator[DockerSandbox]:
    sb = DockerSandbox(
        prebuilt_image, client=docker_client, workdir="/testbed",
        env_setup="export FIXTURE_ENV=activated",
        setup_script="touch /tmp/setup_ran",
    )  # fmt: skip
    sb.create("owner/calc", "HEAD")
    try:
        yield sb
    finally:
        sb.destroy()


def test_prebuilt_uses_repo_in_image(prebuilt: DockerSandbox) -> None:
    assert prebuilt.run("pwd", 10).stdout.strip() == "/testbed"
    assert "calc/stats.py" in prebuilt.run("git ls-files", 10).stdout
    assert prebuilt.base_commit is not None
    assert len(prebuilt.base_commit) == 40


def test_env_setup_and_setup_script(prebuilt: DockerSandbox) -> None:
    assert prebuilt.run("echo $FIXTURE_ENV", 10).stdout.strip() == "activated"
    assert prebuilt.run("test -f /tmp/setup_ran", 10).ok


def test_environment_leftovers_preserved_and_hidden(prebuilt: DockerSandbox) -> None:
    # Untracked files shipped with the image never appear in candidate diffs...
    assert prebuilt.read_diff() == ""
    prebuilt.write("calc/stats.py", "X = 1\n")
    diff = prebuilt.read_diff()
    assert "calc/stats.py" in diff
    assert "shipped_untracked.txt" not in diff
    # ...and resets keep gitignored build artifacts the environment depends on.
    prebuilt.reset_to_base()
    assert prebuilt.read_diff() == ""
    assert prebuilt.run("test -f calc/_speedups.so", 10).ok
    assert prebuilt.run("test -f shipped_untracked.txt", 10).ok


def test_e2b_stub_is_interface_complete() -> None:
    sb = E2BSandbox(template="t")
    for call in (
        lambda: sb.create("r", "c"), lambda: sb.write("a", "b"), lambda: sb.run("ls", 1),
        sb.read_diff,
    ):  # fmt: skip
        with pytest.raises(NotImplementedError):
            call()
    sb.destroy()  # idempotent no-op
