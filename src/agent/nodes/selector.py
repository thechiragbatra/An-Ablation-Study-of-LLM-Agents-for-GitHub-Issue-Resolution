"""Selection without a critic. No LLM.

K=1: the only candidate. K>1: a seeded-random choice among applicable, non-duplicate
candidates, i.e. a "K samples, no judge" baseline for the critic on/off ablation.
"""

from __future__ import annotations

import random

from agent.deps import Deps
from agent.state import (
    AgentState,
    Candidate,
    SelectedBy,
    StateUpdate,
    is_applicable,
    round_candidates,
)


def selection_pool(state: AgentState) -> tuple[list[Candidate], list[Candidate]]:
    """(all candidates of the current round, the applicable non-duplicate ones)."""
    rnd = round_candidates(state["candidates"], state["attempt_count"])
    if not rnd:
        raise ValueError("selection reached with no candidates in the current round")
    pool = [c for c in rnd if is_applicable(c) and c["duplicate_of"] is None]
    return rnd, pool


def mark_selected(cand: Candidate, by: SelectedBy) -> StateUpdate:
    return {
        "selected_index": cand["index"],
        "candidates": [Candidate(**{**cand, "selected_by": by})],
    }


def forced_choice(rnd: list[Candidate], pool: list[Candidate]) -> Candidate | None:
    """The candidate to take when there is nothing to choose between, else None."""
    if len(pool) == 1:
        return pool[0]
    if not pool:
        return rnd[0]  # nothing applies: the validator records the failure
    return None


def selector(state: AgentState, deps: Deps) -> StateUpdate:
    rnd, pool = selection_pool(state)
    forced = forced_choice(rnd, pool)
    if forced is not None:
        return mark_selected(forced, "only")
    rng = random.Random(f"{deps.config.seed}:{state['instance_id']}:{state['attempt_count']}")
    return mark_selected(rng.choice(pool), "random")
