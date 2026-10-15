"""E2B-backed sandbox: interface-complete stub. Not used by any Phase 1 config."""

from __future__ import annotations

from agent.sandbox.base import ExecObserver, ExecResult, Sandbox

_MSG = "E2BSandbox is an interface stub; use sandbox.backend=docker"


class E2BSandbox(Sandbox):
    def __init__(self, template: str, observer: ExecObserver | None = None) -> None:
        super().__init__(observer)
        self.template = template

    def create(self, repo: str, base_commit: str) -> str:
        raise NotImplementedError(_MSG)

    def write(self, path: str, content: str) -> None:
        raise NotImplementedError(_MSG)

    def run(self, cmd: str, timeout_s: int) -> ExecResult:
        raise NotImplementedError(_MSG)

    def read_diff(self) -> str:
        raise NotImplementedError(_MSG)

    def destroy(self) -> None:
        # Nothing was created, so release is a no-op (keeps destroy idempotent).
        return None
