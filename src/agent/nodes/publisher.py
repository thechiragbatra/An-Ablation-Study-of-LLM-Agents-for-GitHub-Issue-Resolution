"""Publisher: writes a LOCAL pending PR package for a sandbox-verified run. No network.

Real publishing is a separate, human-gated step (pilot/publish.py --publish plus an
approval file bound to the package contents). This node cannot reach GitHub.
"""

from __future__ import annotations

from agent.deps import Deps
from agent.publishing import PackageError, diff_files, make_package, write_package
from agent.state import AgentState, StateUpdate


def publisher(state: AgentState, deps: Deps) -> StateUpdate:
    if state["status"] != "sandbox_verified":
        raise PackageError("only sandbox-verified patches can be packaged")
    if deps.instance is None:
        raise PackageError("publisher requires the run's instance")
    idx = state["selected_index"]
    if idx is None:
        raise PackageError("no selected candidate")
    diff = state["candidates"][idx]["diff"]
    if not diff or "Binary files" in diff:
        raise PackageError("only non-empty text diffs can be published")

    sb = deps.sandbox
    sb.reset_to_base()
    applied = sb.apply_diff(diff)
    if not applied.ok:
        raise PackageError(f"selected diff no longer applies: {applied.stderr[:200]}")
    touched, _, deleted = diff_files(diff)
    files = {p: sb.read_file(p) for p in sorted(touched - deleted)}
    sb.reset_to_base()

    pkg = make_package(
        state, deps.config, deps.instance, files=files, deleted=sorted(deleted),
        run_id=deps.tracer.run_id, trace_path=str(deps.tracer.path),
    )  # fmt: skip
    write_package(pkg, deps.config.publisher.pending_dir)
    return {}
