# Controlled live pilot

The pilot takes a real open GitHub issue and turns it into a **draft** pull request. A
human approves every PR. None of this is part of the benchmark: `eval/` refuses configs
with the publisher enabled, and pilot results are reported in their own table.

## Safety model

| Layer | Guarantee |
|---|---|
| Agent graph | The `publisher` node only writes a **local** package under `out/pending/`. Nothing in `src/agent` can reach GitHub. |
| What gets packaged | Only `sandbox_verified` runs: the patch applied and the repo's existing tests passed in an offline sandbox. |
| Package invariants | The title starts with `[AI-Generated Patch]`. The body opens with an AI disclosure. The diff matches the stored file contents. The head branch is `ai-patch/…` and is never a default branch. |
| Human approval | `pilot.approve` shows the exact title, body and diff. You type the first 8 characters of the package hash. The resulting `APPROVED.json` is bound to the package contents, so any later edit invalidates it. |
| Publish step | Dry run **by default**. `--publish` re-runs every gate and needs `GITHUB_TOKEN`. It opens exactly one PR per invocation, always as a **draft**: the client has no non-draft code path. |
| Repo allowlist | Your own repos and forks (owned by the token's user) are allowed implicitly. Any other repo must be added explicitly in `configs/pilot/pilot.yaml`. Archived repos are refused. |
| AI-contribution policy | `CONTRIBUTING*`, `.github/CONTRIBUTING.md`, `docs/CONTRIBUTING.md`, `AI_POLICY.md` and `CODE_OF_CONDUCT.md` are scanned. An explicit prohibition always blocks. **Any** mention of AI also blocks until a human reads it and sets `ai_policy_reviewed: true`. |
| Volume | `max_prs_per_run` (1) and `max_prs_per_day` (3, per UTC day, from the publish log), with at least `min_seconds_between_prs` (120 s) between PRs. |
| Rate limiting | Writes are spaced at least 2 s apart. The client stops when fewer than 50 API calls remain. Secondary rate limits (403/429) **abort**; nothing is retried. |
| Issue state | The issue must be open, unlocked and not a PR, with no existing open `ai-patch/issue-<n>-…` PR. |
| Branches | Commits go to a **new** `ai-patch/…` branch on your fork (or your own repo), never to a default branch. Creating a ref that already exists fails the publish. The PR's base is the upstream default branch, as it must be for a maintainer to merge it. |
| Secrets | `GITHUB_TOKEN` and `ANTHROPIC_API_KEY` come from the environment only. The token is sent in a header and never logged; traces and errors are redacted. |
| Untrusted input | Issue text goes to the LLM. Mitigations: the sandbox has no network, no secrets enter it, and a human approves every PR. |

## Workflow

```bash
export ANTHROPIC_API_KEY=...   # never commit
export GITHUB_TOKEN=...        # fine-grained token: contents + pull requests (write) on your
                               # fork only, issues (read); never commit

# 1. Find candidate issues (open, unassigned, labelled good first issue / help wanted).
uv run python -m pilot.issues --repo yourname/yourrepo

# 2. Attempt one issue. Gates are checked first; the repo is shallow-cloned on the host.
#    A sandbox-verified patch becomes a LOCAL pending package.
uv run python -m pilot.run_issue --repo yourname/yourrepo --issue 12

# 3. Review and approve that one package.
uv run python -m pilot.approve out/pending/<package_id> --reviewer "Your Name"

# 4. Dry run (default): shows gate results and the exact API calls; writes nothing.
uv run python -m pilot.publish out/pending/<package_id>

# 5. Publish as a draft PR.
uv run python -m pilot.publish out/pending/<package_id> --publish

# 6. Later: record outcomes (merged / closed / open-active / open-waiting / ignored).
uv run python -m pilot.outcomes          # -> out/pilot/pr_outcomes.csv and .md
```

## Adding a third-party repository

1. Read its `CONTRIBUTING`, code of conduct and any AI policy yourself.
2. Build a Docker image with its dependencies installed. The sandbox has no network,
   so tests must run offline. For example:
   ```dockerfile
   FROM python:3.12-slim
   RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
   COPY requirements*.txt /tmp/
   RUN pip install --no-cache-dir -r /tmp/requirements.txt pytest
   WORKDIR /workspace
   ```
3. Add the repo to `configs/pilot/pilot.yaml`:
   ```yaml
   allowlist:
     - repo: owner/name
       image: pilot-owner-name:latest
       test_command: python -m pytest -q
       ai_policy_reviewed: true   # only if you read their policy and it permits this
   ```
4. Prefer issues where maintainers have signalled that outside help is welcome. Comment
   on the issue before opening a PR.

## Code-health mode (optional, off by default)

`pilot.scan` runs `ruff --select F,B,E9` in the sandbox. That covers likely logical bugs
only: undefined names, unused or redefined symbols, bugbear checks and syntax errors. It
turns the top finding group into a task.

- **Enabling it:** set `code_health_scan: true` **and** pass `--enable-code-health-scan`.
- **Packaging:** packages are marked "no linked issue; found by static analysis".
- **Not ground truth:** these tasks are never part of the benchmark.

## Etiquette

- **Volume:** keep it low (the caps enforce this) and respond to every review comment.
- **Closing:** close your own PR promptly if a maintainer asks or the issue is resolved
  another way.
- **Hard stop:** never re-submit to a repository that declined an AI-generated PR. Add it
  to your notes, and remove it from the allowlist if it was explicitly listed.
