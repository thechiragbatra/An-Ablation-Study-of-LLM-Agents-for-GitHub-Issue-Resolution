"""READ-ONLY smoke test against the real GitHub API (opt-in: --run-github, GITHUB_TOKEN).
It must never write: the client's write counter is asserted to stay at zero."""

from __future__ import annotations

import pytest

from pilot.config import PilotConfig
from pilot.gates import gate_ai_policy, gate_allowlist
from pilot.github import RestGitHub

pytestmark = pytest.mark.github


def test_read_only_calls() -> None:
    gh = RestGitHub.from_env(require_token=True)
    assert gh.user_login()
    repo = gh.get_repo("octocat/Hello-World")
    assert repo["default_branch"]
    assert isinstance(gh.list_issues("octocat/Hello-World", "bug"), list)
    cfg = PilotConfig()
    assert not gate_allowlist(gh, cfg, "octocat/Hello-World").ok  # not ours, not allowlisted
    gate_ai_policy(gh, cfg, "octocat/Hello-World")
    assert gh.write_calls == 0
