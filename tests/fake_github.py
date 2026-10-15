"""In-memory GitHub for pilot tests. Records every write so tests can assert exactly what
would have been sent; tests never contact the real GitHub."""

from __future__ import annotations

from typing import Any

from pilot.github import GitHubError


class FakeGitHub:
    def __init__(self, login: str | None = "me") -> None:
        self.login = login
        self.repos: dict[str, dict[str, Any]] = {}
        self.issues: dict[tuple[str, int], dict[str, Any]] = {}
        self.files: dict[tuple[str, str], str] = {}
        self.pulls: dict[str, list[dict[str, Any]]] = {}
        self.commits: set[tuple[str, str]] = set()
        self.modes: dict[str, str] = {}
        self.refs: set[tuple[str, str]] = set()
        self.fork_has_base = True
        self.writes: list[tuple[str, Any]] = []
        self.drafts: list[dict[str, Any]] = []
        self._n = 0

    # -- setup helpers -------------------------------------------------------

    def add_repo(
        self,
        slug: str,
        *,
        owner: str | None = None,
        fork: bool = False,
        archived: bool = False,
        default_branch: str = "main",
        base_sha: str = "b" * 40,
    ) -> None:
        self.repos[slug] = {
            "full_name": slug,
            "owner": {"login": owner or slug.split("/", maxsplit=1)[0]},
            "fork": fork,
            "archived": archived,
            "default_branch": default_branch,
        }
        self.commits.add((slug, base_sha))
        self.pulls.setdefault(slug, [])

    def add_issue(
        self,
        slug: str,
        number: int,
        *,
        state: str = "open",
        locked: bool = False,
        labels: tuple[str, ...] = ("good first issue",),
        **extra: Any,
    ) -> None:
        self.issues[(slug, number)] = {
            "number": number, "state": state, "locked": locked, "title": f"Issue {number}",
            "body": "It is broken.", "html_url": f"https://github.com/{slug}/issues/{number}",
            "labels": [{"name": n} for n in labels], "assignees": [], **extra,
        }  # fmt: skip

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n}"

    # -- reads -----------------------------------------------------------------

    def user_login(self) -> str | None:
        return self.login

    def get_repo(self, slug: str) -> dict[str, Any]:
        if slug not in self.repos:
            raise GitHubError(404, "Not Found")
        return dict(self.repos[slug])

    def get_issue(self, slug: str, number: int) -> dict[str, Any]:
        if (slug, number) not in self.issues:
            raise GitHubError(404, "Not Found")
        return dict(self.issues[(slug, number)])

    def list_issues(self, slug: str, label: str) -> list[dict[str, Any]]:
        return [
            i for (s, _), i in self.issues.items()
            if s == slug and i["state"] == "open" and label in [x["name"] for x in i["labels"]]
        ]  # fmt: skip

    def get_file(self, slug: str, path: str) -> str | None:
        return self.files.get((slug, path))

    def list_open_pulls(self, slug: str) -> list[dict[str, Any]]:
        return list(self.pulls.get(slug, []))

    def commit_exists(self, slug: str, sha: str) -> bool:
        return (slug, sha) in self.commits

    def get_commit_tree(self, slug: str, sha: str) -> str:
        return f"tree-of-{sha[:7]}"

    def get_tree_modes(self, slug: str, tree_sha: str) -> dict[str, str]:
        return dict(self.modes)

    def get_pull(self, slug: str, number: int) -> dict[str, Any]:
        return next(p for p in self.pulls[slug] if p["number"] == number)

    def list_reviews(self, slug: str, number: int) -> list[dict[str, Any]]:
        return []

    def list_review_comments(self, slug: str, number: int) -> list[dict[str, Any]]:
        return []

    def list_issue_comments(self, slug: str, number: int) -> list[dict[str, Any]]:
        return []

    # -- writes ----------------------------------------------------------------

    def create_fork(self, slug: str) -> dict[str, Any]:
        self.writes.append(("create_fork", slug))
        assert self.login is not None
        fork = f"{self.login}/{slug.split('/')[1]}"
        if fork not in self.repos:
            upstream = self.repos[slug]
            self.add_repo(fork, owner=self.login, fork=True,
                          default_branch=upstream["default_branch"], base_sha="0" * 40)  # fmt: skip
            if self.fork_has_base:
                self.commits |= {(fork, sha) for s, sha in self.commits if s == slug}
        return dict(self.repos[fork])

    def sync_fork(self, slug: str, branch: str) -> None:
        self.writes.append(("sync_fork", (slug, branch)))

    def create_blob(self, slug: str, content: str) -> str:
        self.writes.append(("create_blob", (slug, content)))
        return self._id("blob")

    def create_tree(self, slug: str, base_tree: str, entries: list[dict[str, Any]]) -> str:
        self.writes.append(("create_tree", (slug, base_tree, entries)))
        return self._id("tree")

    def create_commit(self, slug: str, message: str, tree: str, parent: str) -> str:
        self.writes.append(("create_commit", (slug, message, tree, parent)))
        return self._id("commit")

    def create_ref(self, slug: str, ref: str, sha: str) -> None:
        self.writes.append(("create_ref", (slug, ref, sha)))
        if (slug, ref) in self.refs:
            raise GitHubError(422, "Reference already exists")
        self.refs.add((slug, ref))

    def create_draft_pull(
        self, slug: str, *, title: str, body: str, head: str, base: str
    ) -> dict[str, Any]:
        payload = {"title": title, "body": body, "head": head, "base": base, "draft": True}
        self.writes.append(("create_draft_pull", payload))
        pr = {
            "number": 100 + len(self.drafts),
            "html_url": f"https://github.com/{slug}/pull/1",
            "draft": True,
            "state": "open",
            "head": {"ref": head.rsplit(":", maxsplit=1)[-1]},
        }
        self.drafts.append(payload)
        self.pulls.setdefault(slug, []).append(pr)
        return pr

    @property
    def write_names(self) -> list[str]:
        return [w[0] for w in self.writes]
