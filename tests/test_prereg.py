"""Preregistration gate, tested against real temporary git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.config import REPO_ROOT
from eval.prereg import (
    PLACEHOLDER,
    PREREG_RELPATH,
    PreregistrationError,
    check_preregistered,
    require_preregistered,
)
from eval.run_experiment import main as run_experiment_main

FILLED = "# Preregistration\n\nH1: A1 > A0. alpha = 0.05\n"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, check=True, capture_output=True,
    )  # fmt: skip


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("x\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _write(repo: Path, text: str) -> Path:
    path = repo / PREREG_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _commit(repo: Path) -> None:
    _git(repo, "add", str(PREREG_RELPATH))
    _git(repo, "commit", "-q", "-m", "prereg")


def test_missing_file_refused(repo: Path) -> None:
    st = check_preregistered(repo)
    assert not st.ok
    assert "does not exist" in st.problems[0]


def test_untracked_file_refused(repo: Path) -> None:
    _write(repo, FILLED)
    st = check_preregistered(repo)
    assert any("not committed" in p for p in st.problems)


def test_modified_file_refused(repo: Path) -> None:
    _write(repo, FILLED)
    _commit(repo)
    _write(repo, FILLED + "changed after registration\n")
    assert any("uncommitted changes" in p for p in check_preregistered(repo).problems)


def test_template_placeholders_refused(repo: Path) -> None:
    _write(repo, f"alpha = {PLACEHOLDER}\nK = {PLACEHOLDER}\n")
    _commit(repo)
    st = check_preregistered(repo)
    assert any("2 unfilled" in p for p in st.problems)


def test_dirty_tree_refused(repo: Path) -> None:
    _write(repo, FILLED)
    _commit(repo)
    (repo / "scratch.py").write_text("tuning after registration\n")
    assert any("working tree is not clean" in p for p in check_preregistered(repo).problems)


def test_no_commits_refused(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, FILLED)
    assert any("no commits" in p for p in check_preregistered(tmp_path).problems)


def test_committed_and_filled_passes(repo: Path) -> None:
    _write(repo, FILLED)
    _commit(repo)
    st = require_preregistered(repo)
    assert st.ok
    assert st.commit is not None
    assert len(st.commit) == 40
    assert st.sha256 is not None


def test_require_raises_with_reasons(repo: Path) -> None:
    with pytest.raises(PreregistrationError, match="refusing to run the frozen subset"):
        require_preregistered(repo)


def test_shipped_template_is_not_runnable() -> None:
    text = (REPO_ROOT / PREREG_RELPATH).read_text()
    assert PLACEHOLDER in text  # it is a template until the author fills it in


def test_cli_refuses_frozen_run_in_this_repo(capsys: pytest.CaptureFixture[str]) -> None:
    # Nothing is registered or committed here yet, so the frozen subset must be refused
    # before any config, data or Docker is touched.
    code = run_experiment_main(
        ["--config", "configs/eval/a0.yaml", "--set", "frozen", "--out", "/nonexistent"]
    )
    assert code == 3
    assert "refusing to run the frozen subset" in capsys.readouterr().err
