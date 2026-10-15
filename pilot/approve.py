"""Human approval of ONE pending package. Shows exactly what would be published and asks
the reviewer to type the first 8 characters of the package hash. Writes APPROVED.json,
which is bound to the package contents: any later edit invalidates it.

    uv run python -m pilot.approve out/pending/<package_id> --reviewer "Your Name"

This command never contacts GitHub.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from agent.publishing import load_package, package_invariants
from pilot.gates import APPROVAL_FILE


class ApprovalError(Exception):
    pass


def approve(
    pkg_dir: Path,
    reviewer: str,
    confirm: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> Path:
    if not reviewer.strip():
        raise ApprovalError("a reviewer name is required")
    pkg = load_package(pkg_dir)
    problems = package_invariants(pkg)
    if problems:
        raise ApprovalError("package violates invariants: " + "; ".join(problems))
    sha = pkg.content_sha256()
    show(f"Repository : {pkg.repo_slug}  (base {pkg.default_branch} @ {pkg.base_sha[:12]})")
    show(f"Branch     : {pkg.head_branch}  (draft PR)")
    show(f"Title      : {pkg.title}\n")
    show(pkg.final_body(reviewer))
    show("----- diff -----")
    show(pkg.diff)
    show(f"Package sha256: {sha}")
    answer = confirm("Type the first 8 characters of the sha256 to approve: ").strip()
    if answer != sha[:8]:
        raise ApprovalError("confirmation did not match; nothing was approved")
    path = pkg_dir / APPROVAL_FILE
    path.write_text(json.dumps({
        "package_sha256": sha,
        "approved_by": reviewer.strip(),
        "approved_at": datetime.now(UTC).isoformat(),
    }, indent=2))  # fmt: skip
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("package_dir", type=Path)
    ap.add_argument("--reviewer", required=True)
    args = ap.parse_args(argv)
    try:
        path = approve(args.package_dir, args.reviewer)
    except ApprovalError as exc:
        print(f"not approved: {exc}", file=sys.stderr)
        return 2
    print(f"approved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
