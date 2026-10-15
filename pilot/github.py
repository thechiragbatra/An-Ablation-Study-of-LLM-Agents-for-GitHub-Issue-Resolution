"""Minimal GitHub REST client (stdlib only) for the live pilot.

- Token from ``GITHUB_TOKEN`` only; it is sent in a header and never logged or stored.
- Every write is spaced by ``min_write_interval_s``; when the primary rate limit runs low
  or GitHub signals a secondary limit (403/429), the client ABORTS instead of retrying.
- Pull requests can only be created as drafts: ``create_draft_pull`` has no draft flag.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from agent.tracing import redact

API = "https://api.github.com"
_RATE_FLOOR = 50  # stop when fewer primary-limit requests remain


class GitHubError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {redact(message)[:300]}")
        self.status = status


class GitHubRateLimited(GitHubError):
    pass


class GitHubClient(Protocol):
    # reads
    def user_login(self) -> str | None: ...
    def get_repo(self, slug: str) -> dict[str, Any]: ...
    def get_issue(self, slug: str, number: int) -> dict[str, Any]: ...
    def list_issues(self, slug: str, label: str) -> list[dict[str, Any]]: ...
    def get_file(self, slug: str, path: str) -> str | None: ...
    def list_open_pulls(self, slug: str) -> list[dict[str, Any]]: ...
    def commit_exists(self, slug: str, sha: str) -> bool: ...
    def get_commit_tree(self, slug: str, sha: str) -> str: ...
    def get_tree_modes(self, slug: str, tree_sha: str) -> dict[str, str]: ...
    def get_pull(self, slug: str, number: int) -> dict[str, Any]: ...
    def list_reviews(self, slug: str, number: int) -> list[dict[str, Any]]: ...
    def list_review_comments(self, slug: str, number: int) -> list[dict[str, Any]]: ...
    def list_issue_comments(self, slug: str, number: int) -> list[dict[str, Any]]: ...

    # writes
    def create_fork(self, slug: str) -> dict[str, Any]: ...
    def sync_fork(self, slug: str, branch: str) -> None: ...
    def create_blob(self, slug: str, content: str) -> str: ...
    def create_tree(self, slug: str, base_tree: str, entries: list[dict[str, Any]]) -> str: ...
    def create_commit(self, slug: str, message: str, tree: str, parent: str) -> str: ...
    def create_ref(self, slug: str, ref: str, sha: str) -> None: ...
    def create_draft_pull(
        self, slug: str, *, title: str, body: str, head: str, base: str
    ) -> dict[str, Any]: ...


class RestGitHub:
    def __init__(
        self,
        token: str | None,
        *,
        api: str = API,
        min_write_interval_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._api = api.rstrip("/")
        self._min_write = min_write_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last_write: float | None = None
        self.write_calls = 0

    @classmethod
    def from_env(cls, require_token: bool = False) -> RestGitHub:
        token = os.environ.get("GITHUB_TOKEN") or None
        if require_token and token is None:
            raise GitHubError(401, "GITHUB_TOKEN is not set")
        return cls(token)

    @property
    def authenticated(self) -> bool:
        return self._token is not None

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if method != "GET":
            if self._token is None:
                raise GitHubError(401, "writes require GITHUB_TOKEN")
            if self._last_write is not None:
                wait = self._min_write - (self._clock() - self._last_write)
                if wait > 0:
                    self._sleep(wait)
            self._last_write = self._clock()
            self.write_calls += 1
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ablation-study-pilot",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._api + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                self._check_remaining(resp.headers.get("X-RateLimit-Remaining"))
                raw = resp.read()
        except urllib.error.HTTPError as err:
            text = err.read().decode("utf-8", "replace")
            limited = err.code == 429 or (
                err.code == 403
                and ("rate limit" in text.lower() or err.headers.get("Retry-After") is not None)
            )
            cls = GitHubRateLimited if limited else GitHubError
            raise cls(err.code, text) from None
        return json.loads(raw) if raw else None

    def _check_remaining(self, remaining: str | None) -> None:
        if remaining is not None and remaining.isdigit() and int(remaining) < _RATE_FLOOR:
            raise GitHubRateLimited(403, f"only {remaining} API requests left; stopping")

    def _get_or_none(self, path: str) -> Any:
        try:
            return self._request("GET", path)
        except GitHubError as err:
            if err.status == 404:
                return None
            raise

    # -- reads -------------------------------------------------------------

    def user_login(self) -> str | None:
        if self._token is None:
            return None
        return str(self._request("GET", "/user")["login"])

    def get_repo(self, slug: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/repos/{slug}"))

    def get_issue(self, slug: str, number: int) -> dict[str, Any]:
        return dict(self._request("GET", f"/repos/{slug}/issues/{number}"))

    def list_issues(self, slug: str, label: str) -> list[dict[str, Any]]:
        q = urllib.parse.urlencode({"state": "open", "labels": label, "per_page": 50})
        return list(self._request("GET", f"/repos/{slug}/issues?{q}"))

    def get_file(self, slug: str, path: str) -> str | None:
        data = self._get_or_none(f"/repos/{slug}/contents/{urllib.parse.quote(path)}")
        if not isinstance(data, dict) or data.get("encoding") != "base64":
            return None
        return base64.b64decode(data["content"]).decode("utf-8", "replace")

    def list_open_pulls(self, slug: str) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/repos/{slug}/pulls?state=open&per_page=100"))

    def commit_exists(self, slug: str, sha: str) -> bool:
        return self._get_or_none(f"/repos/{slug}/git/commits/{sha}") is not None

    def get_commit_tree(self, slug: str, sha: str) -> str:
        return str(self._request("GET", f"/repos/{slug}/git/commits/{sha}")["tree"]["sha"])

    def get_tree_modes(self, slug: str, tree_sha: str) -> dict[str, str]:
        tree = self._request("GET", f"/repos/{slug}/git/trees/{tree_sha}?recursive=1")
        return {e["path"]: e["mode"] for e in tree["tree"] if e["type"] == "blob"}

    def get_pull(self, slug: str, number: int) -> dict[str, Any]:
        return dict(self._request("GET", f"/repos/{slug}/pulls/{number}"))

    def list_reviews(self, slug: str, number: int) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/repos/{slug}/pulls/{number}/reviews"))

    def list_review_comments(self, slug: str, number: int) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/repos/{slug}/pulls/{number}/comments"))

    def list_issue_comments(self, slug: str, number: int) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/repos/{slug}/issues/{number}/comments"))

    # -- writes ------------------------------------------------------------

    def create_fork(self, slug: str) -> dict[str, Any]:
        return dict(self._request("POST", f"/repos/{slug}/forks", {}))

    def sync_fork(self, slug: str, branch: str) -> None:
        self._request("POST", f"/repos/{slug}/merge-upstream", {"branch": branch})

    def create_blob(self, slug: str, content: str) -> str:
        body = {"content": content, "encoding": "utf-8"}
        return str(self._request("POST", f"/repos/{slug}/git/blobs", body)["sha"])

    def create_tree(self, slug: str, base_tree: str, entries: list[dict[str, Any]]) -> str:
        body = {"base_tree": base_tree, "tree": entries}
        return str(self._request("POST", f"/repos/{slug}/git/trees", body)["sha"])

    def create_commit(self, slug: str, message: str, tree: str, parent: str) -> str:
        body = {"message": message, "tree": tree, "parents": [parent]}
        return str(self._request("POST", f"/repos/{slug}/git/commits", body)["sha"])

    def create_ref(self, slug: str, ref: str, sha: str) -> None:
        self._request("POST", f"/repos/{slug}/git/refs", {"ref": ref, "sha": sha})

    def create_draft_pull(
        self, slug: str, *, title: str, body: str, head: str, base: str
    ) -> dict[str, Any]:
        payload = {
            "title": title, "body": body, "head": head, "base": base,
            "draft": True,  # hard-coded: the pilot never opens a ready-for-review PR
            "maintainer_can_modify": True,
        }  # fmt: skip
        return dict(self._request("POST", f"/repos/{slug}/pulls", payload))
