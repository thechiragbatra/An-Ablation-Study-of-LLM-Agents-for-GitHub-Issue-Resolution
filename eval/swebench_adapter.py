"""SWE-bench record -> agent-visible ``Instance``.

swebench 5.x is data-driven: every dataset row carries its environment ``image`` and the
official ``eval_script``. The agent gets:

- **env_setup**: the script's conda activation, ``cd`` and ``export`` lines.
- **setup_script**: one-time system setup lines (django's locale generation).
- **test_command**: the official test command, run at the base commit WITHOUT the gold
  test patch, i.e. the *existing* tests in the touched test files (regression signal).
  Targets that the gold test patch creates are dropped (they do not exist at base); if no
  target remains, a per-repo import smoke test is used instead.

Dropped on purpose: ``pip install`` lines (the image already has the environment and the
sandbox has no network), git bookkeeping, and the test-patch heredoc. Never exposed: the
gold patch, test patch contents, FAIL_TO_PASS / PASS_TO_PASS, and hints (standard
SWE-bench setting without hints).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import docker.errors

from agent.instance import Instance
from agent.publishing import diff_files

WORKDIR = "/testbed"
_START = ">>>>> Start Test Output"
_END = ">>>>> End Test Output"
_HEREDOC = re.compile(r"<<\s*'(EOF_\w+)'")
_ENV_LINE = re.compile(
    r"^(source /opt/miniconda3/bin/activate|conda activate \S+|export \w+=.*|cd /testbed)$"
)
_SETUP_LINE = re.compile(r"locale-gen")

# Import smoke test used when no pre-existing test target remains.
FALLBACK_IMPORT = {
    "astropy/astropy": "astropy",
    "django/django": "django",
    "matplotlib/matplotlib": "matplotlib",
    "mwaskom/seaborn": "seaborn",
    "pallets/flask": "flask",
    "psf/requests": "requests",
    "pydata/xarray": "xarray",
    "pylint-dev/pylint": "pylint",
    "pytest-dev/pytest": "pytest",
    "scikit-learn/scikit-learn": "sklearn",
    "sphinx-doc/sphinx": "sphinx",
    "sympy/sympy": "sympy",
}

# Fields that must never reach the agent (checked by tests).
GOLD_FIELDS = ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text")


class AdapterError(Exception):
    pass


@dataclass(frozen=True)
class EvalScriptParts:
    env_setup: list[str]
    setup: list[str]
    test_lines: list[str]


def split_eval_script(script: str) -> EvalScriptParts:
    env: list[str] = []
    setup: list[str] = []
    tests: list[str] = []
    mode = "pre"
    heredoc_end: str | None = None
    for line in script.splitlines():
        if heredoc_end is not None:
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        if m := _HEREDOC.search(line):
            heredoc_end = m.group(1)
            continue
        if _START in line:
            mode = "test"
            continue
        if _END in line:
            mode = "post"
            continue
        stripped = line.strip()
        if mode == "test" and stripped:
            tests.append(stripped)
        elif mode == "pre" and _ENV_LINE.match(stripped) and stripped not in env:
            env.append(stripped)
        elif mode == "pre" and _SETUP_LINE.search(stripped):
            setup.append(stripped)
    if not tests:
        raise AdapterError("eval_script has no test command between the output markers")
    return EvalScriptParts(env, setup, tests)


def patch_files(patch: str) -> tuple[set[str], set[str]]:
    """(all files touched, files created) by a unified git diff."""
    touched, created, _ = diff_files(patch)
    return touched, created


def _candidate_paths(token: str) -> set[str]:
    """Repo paths a test-command token may refer to (file paths, or django modules)."""
    paths = {token}
    if "/" not in token and "." in token and not token.startswith("-") and "=" not in token:
        mod = token.replace(".", "/")
        paths |= {f"tests/{mod}.py", f"tests/{mod}/__init__.py"}
    return paths


def regression_command(record: Mapping[str, Any]) -> tuple[str, bool]:
    """(command, used_fallback). Existing tests only; new gold test files are dropped."""
    parts = split_eval_script(record["eval_script"])
    touched, new_files = patch_files(record["test_patch"])
    commands: list[str] = []
    kept_targets = 0
    for line in parts.test_lines:
        kept: list[str] = []
        for token in shlex.split(line):
            refs = _candidate_paths(token)
            if refs & new_files:
                continue  # does not exist at the base commit
            if refs & touched:
                kept_targets += 1
            kept.append(token)
        commands.append(shlex.join(kept))
    if kept_targets == 0:
        module = FALLBACK_IMPORT.get(record["repo"])
        if module is None:
            raise AdapterError(f"no fallback import for repo {record['repo']}")
        return f"python -c 'import {module}'", True
    return " && ".join(commands), False


def to_instance(record: Mapping[str, Any]) -> Instance:
    parts = split_eval_script(record["eval_script"])
    test_command, _ = regression_command(record)
    env = parts.env_setup or [f"cd {WORKDIR}"]
    return Instance(
        instance_id=record["instance_id"],
        repo=record["repo"],
        base_commit=record["base_commit"],
        issue_text=record["problem_statement"],
        test_command=test_command,
        image=record["image"],
        workdir=WORKDIR,
        env_setup="\n".join(env),
        setup_script="\n".join(parts.setup) or None,
    )


def ensure_image(image: str, client: Any, platform: str = "linux/amd64") -> None:
    """Pull the official (x86-64) environment image if it is not present locally.

    On arm64 hosts it runs under emulation, which is slower but keeps every instance
    and avoids the selection bias of excluding images that fail to build natively."""
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        repo, _, tag = image.rpartition(":")
        client.images.pull(repo, tag=tag or "latest", platform=platform)
