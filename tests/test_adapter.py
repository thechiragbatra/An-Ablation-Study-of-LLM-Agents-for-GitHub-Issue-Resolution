"""SWE-bench record -> agent Instance: regression command derivation and leakage."""

from __future__ import annotations

from typing import Any

import pytest

from eval.data import load_set
from eval.swebench_adapter import (
    AdapterError,
    patch_files,
    regression_command,
    split_eval_script,
    to_instance,
)

SCRIPT = """#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
export LANG=en_US.UTF-8
python -m pip install -e .
git checkout abc tests/a/tests.py
git apply -v - <<'EOF_1234'
diff --git a/tests/a/tests.py b/tests/a/tests.py
+def test_SECRET_new_behaviour(): ...
EOF_1234
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 a.tests b.test_new
: '>>>>> End Test Output'
git checkout abc tests/a/tests.py
"""

TEST_PATCH = """diff --git a/tests/a/tests.py b/tests/a/tests.py
--- a/tests/a/tests.py
+++ b/tests/a/tests.py
@@ -1 +1,2 @@
+def test_SECRET_new_behaviour(): ...
diff --git a/tests/b/test_new.py b/tests/b/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/b/test_new.py
@@ -0,0 +1 @@
+x = 1
diff --git a/tests/b/__init__.py b/tests/b/__init__.py
new file mode 100644
"""


def _record(**kw: Any) -> dict[str, Any]:
    rec = {
        "instance_id": "django__django-1", "repo": "django/django", "base_commit": "abc",
        "problem_statement": "Issue text", "image": "swebench/sweb.eval.x86_64.x:latest",
        "eval_script": SCRIPT, "test_patch": TEST_PATCH, "patch": "GOLD-PATCH",
        "FAIL_TO_PASS": '["test_SECRET_new_behaviour"]', "PASS_TO_PASS": "[]",
        "hints_text": "HINT",
    }  # fmt: skip
    return {**rec, **kw}


def test_split_eval_script() -> None:
    parts = split_eval_script(SCRIPT)
    assert parts.env_setup == [
        "source /opt/miniconda3/bin/activate", "conda activate testbed", "cd /testbed",
        "export LANG=en_US.UTF-8",
    ]  # fmt: skip
    assert parts.setup == ["sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen"]
    assert parts.test_lines == ["./tests/runtests.py --verbosity 2 a.tests b.test_new"]
    with pytest.raises(AdapterError, match="no test command"):
        split_eval_script("echo hi\n")


def test_patch_files_detects_empty_new_files() -> None:
    touched, created = patch_files(TEST_PATCH)
    assert touched == {"tests/a/tests.py", "tests/b/test_new.py", "tests/b/__init__.py"}
    assert created == {"tests/b/test_new.py", "tests/b/__init__.py"}


def test_regression_command_drops_new_test_files() -> None:
    cmd, fallback = regression_command(_record())
    assert cmd == "./tests/runtests.py --verbosity 2 a.tests"
    assert not fallback


def test_fallback_when_only_new_tests() -> None:
    script = SCRIPT.replace("a.tests b.test_new", "b.test_new")
    cmd, fallback = regression_command(_record(eval_script=script))
    assert fallback
    assert cmd == "python -c 'import django'"


def test_instance_never_contains_gold_data() -> None:
    inst = to_instance(_record())
    dumped = inst.model_dump_json()
    for secret in ("GOLD-PATCH", "SECRET", "HINT", "pip install", "EOF_1234"):
        assert secret not in dumped, secret
    assert inst.issue_text == "Issue text"
    assert inst.env_setup is not None
    assert "conda activate testbed" in inst.env_setup
    assert inst.setup_script is not None
    assert "locale-gen" in inst.setup_script


def test_every_frozen_and_dev_record_converts() -> None:
    fallbacks = 0
    for rec in load_set("frozen").records + load_set("dev").records:
        inst = to_instance(rec)
        assert inst.test_command
        assert "pip install" not in inst.test_command
        fallbacks += regression_command(rec)[1]
    assert fallbacks <= 3  # nearly every instance has a real regression command
