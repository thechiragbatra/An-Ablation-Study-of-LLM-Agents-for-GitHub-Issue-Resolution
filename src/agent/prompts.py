"""Prompt text. Shared by every variant so prompts are never a confound between A0-A3.

Tune these only against the dev set (Phase 3), never against benchmark instances.
"""

LOCALIZER_SYSTEM = """\
You are localizing a GitHub issue to source files. Given the issue and the repository's \
file list, choose the files that most likely need to change to resolve the issue. \
Prefer source files over tests unless the issue is about tests. Only choose paths that \
appear in the file list."""

ARCHITECT_SYSTEM = """\
You are a software engineer fixing a GitHub issue. You are given the issue, the current \
contents of the relevant files, and possibly feedback from earlier attempts that failed.

Produce a fix as exact search/replace edits:
- `search` must be copied verbatim from the file shown (including indentation) and must \
occur exactly once in that file. Include enough surrounding lines to make it unique.
- `replace` is the full replacement text for that span.
- To create a new file, use an empty `search` and put the whole file in `replace`.
- Keep the change minimal and focused on the issue. Do not modify tests unless the issue \
requires it.
Explain your reasoning briefly in `rationale`."""


# Appended to the architect's user turn only when K > 1, so A0/A1 prompts are unchanged.
ARCHITECT_DIVERSITY = """\
The candidates must take meaningfully different approaches: e.g. a different location \
for the fix, a different mechanism, or a different scope. Do not return cosmetic \
variants of the same change."""

RUBRIC_DIMENSIONS = ("correctness_likelihood", "minimality", "test_impact", "risk")

CRITIC_SELECT_SYSTEM = """\
You are reviewing candidate patches for a GitHub issue. Score every candidate on each \
rubric dimension with an integer from 1 to 5, then select the single candidate most \
likely to resolve the issue without breaking anything.

Rubric (anchors):
- correctness_likelihood: 1 = does not address the issue; 3 = plausible but incomplete \
or unverified; 5 = clearly resolves the issue as described, including edge cases.
- minimality: 1 = large or unrelated changes; 3 = some unnecessary changes; 5 = the \
smallest change that fully resolves the issue.
- test_impact: 1 = likely breaks existing tests or behaviour; 3 = unclear; 5 = keeps \
existing behaviour and makes the relevant tests pass.
- risk: 1 = very low risk; 3 = moderate risk (touches shared code paths); 5 = high risk \
(API changes, broad side effects). Note: for risk, HIGHER means WORSE.

Candidates whose edits could not be applied cannot be selected. If test results from a \
sandbox run are provided, weigh them heavily but remember the tests may be incomplete. \
Give a one-sentence justification per candidate and explain your selection."""

CRITIC_REFLECT_SYSTEM = """\
You are reviewing a patch for a GitHub issue that failed validation. Using the issue, \
the relevant files, the failed patch and the validation output, diagnose why it failed \
and give concrete guidance for the next revision. Be specific about what to change and \
what to keep. Do not write the patch yourself."""


def format_files(files: dict[str, str]) -> str:
    return "\n\n".join(f"<file path={path!r}>\n{body}\n</file>" for path, body in files.items())
