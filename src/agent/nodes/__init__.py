"""Graph nodes. Each is ``fn(state, deps) -> StateUpdate`` and never mutates ``state``."""

from collections.abc import Callable

from agent.deps import Deps
from agent.nodes.architect import architect
from agent.nodes.critic import critic, critic_reflect
from agent.nodes.localizer import localizer
from agent.nodes.patcher import patcher
from agent.nodes.publisher import publisher
from agent.nodes.selector import selector
from agent.nodes.validator import validator
from agent.state import AgentState, StateUpdate

NodeFn = Callable[[AgentState, Deps], StateUpdate]

__all__ = [
    "NodeFn",
    "architect",
    "critic",
    "critic_reflect",
    "localizer",
    "patcher",
    "publisher",
    "selector",
    "validator",
]
