"""Build realistic pending packages for pilot tests without running the agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.publishing import (
    DISCLOSURE_MARKER,
    REVIEWER_PLACEHOLDER,
    TITLE_PREFIX,
    PendingPackage,
    write_package,
)
from pilot.config import PilotConfig

DIFF = """diff --git a/calc/stats.py b/calc/stats.py
index 1111111..2222222 100644
--- a/calc/stats.py
+++ b/calc/stats.py
@@ -1,2 +1,4 @@
 def mean(values):
+    if not values:
+        return 0.0
     return sum(values) / len(values)
diff --git a/run.sh b/run.sh
index 3333333..4444444 100755
--- a/run.sh
+++ b/run.sh
@@ -1 +1 @@
-echo old
+echo new
"""


def package(**over: Any) -> PendingPackage:
    data: dict[str, Any] = {
        "package_id": "upstream__lib-issue7-abcd1234",
        "created": "2026-10-15T12:00:00+00:00",
        "kind": "issue",
        "repo_slug": "upstream/lib",
        "issue_number": 7,
        "issue_url": "https://github.com/upstream/lib/issues/7",
        "default_branch": "main",
        "base_sha": "b" * 40,
        "head_branch": "ai-patch/issue-7-abcd1234",
        "title": TITLE_PREFIX + "mean() crashes on empty list",
        "body": f"{DISCLOSURE_MARKER}\nAI-generated.\nReviewer: {REVIEWER_PLACEHOLDER}\n",
        "diff": DIFF,
        "files": {
            "calc/stats.py": "def mean(values):\n    if not values:\n        return 0.0\n"
                             "    return sum(values) / len(values)\n",
            "run.sh": "echo new\n",
        },
        "deleted": [],
        "run_id": "20261015T120000Z-abcd1234",
        "trace_path": "out/traces/x.jsonl",
        "variant": "A3",
        "model": "claude-opus-5-5",
        "cost_usd": 0.42,
    }  # fmt: skip
    data.update(over)
    return PendingPackage.model_validate(data)


def write(tmp: Path, **over: Any) -> Path:
    return write_package(package(**over), tmp / "pending")


def pilot_config(tmp: Path, **over: Any) -> PilotConfig:
    data: dict[str, Any] = {
        "pending_dir": tmp / "pending",
        "log_path": tmp / "pilot" / "publish_log.jsonl",
        "clone_dir": tmp / "repos",
    }
    data.update(over)
    return PilotConfig.model_validate(data)
