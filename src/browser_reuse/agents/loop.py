"""Bounded, scenario-neutral observe/decide/act trajectory loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from browser_reuse.core import Observation, TaskSpec
from browser_reuse.interfaces import Adapter
from browser_reuse.llm import ChatModel, Message

from .trajectory import RecordedAction


@dataclass(frozen=True)
class AgentRun:
    claimed_success: bool
    actions: tuple[RecordedAction, ...]
    decisions: int
    error: str | None = None
    last_response: str | None = None


def run_agent_loop(
    task: TaskSpec,
    adapter: Adapter,
    model: ChatModel,
    *,
    make_messages: Callable[[TaskSpec, Observation], Sequence[Message]],
    parse_step: Callable[[str, Observation], Mapping[str, object] | None],
    is_terminal: Callable[[Observation], bool],
    max_decisions: int,
    allow_done_claim: bool = False,
) -> AgentRun:
    if max_decisions < 1:
        raise ValueError("max_decisions must be at least 1")
    actions: list[RecordedAction] = []
    decisions = 0
    response: str | None = None
    try:
        observation = adapter.observe()
        while decisions < max_decisions:
            if is_terminal(observation):
                return AgentRun(
                    True,
                    tuple(actions),
                    decisions,
                    last_response=response,
                )
            response = model.complete(make_messages(task, observation))
            decisions += 1
            step = parse_step(response, observation)
            if step is None:
                return AgentRun(
                    allow_done_claim,
                    tuple(actions),
                    decisions,
                    None if allow_done_claim else "premature done",
                    response,
                )
            adapter.execute(step)
            after = adapter.observe()
            actions.append(
                RecordedAction(
                    action=dict(step),
                    before=observation,
                    after=after,
                )
            )
            observation = after
        return AgentRun(
            False,
            tuple(actions),
            decisions,
            "model decision limit reached",
            response,
        )
    except Exception as exc:
        return AgentRun(
            False,
            tuple(actions),
            decisions,
            f"{type(exc).__name__}: {exc}",
            response,
        )
