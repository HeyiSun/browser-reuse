"""A minimal provider-neutral chat model boundary."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel(Protocol):
    def complete(self, messages: Sequence[Message]) -> str:
        ...


class LiteLLMChatModel:
    """Thin adapter around LiteLLM's provider-neutral completion call."""

    def __init__(self, model: str, **options: object) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self._model = model
        self._options = options

    def complete(self, messages: Sequence[Message]) -> str:
        response = _completion(
            model=self._model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            **self._options,
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("model response contains no choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise ValueError("model response contains no text content")
        return content


def _completion(**kwargs: object) -> object:
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            'LiteLLM is not installed; install the "browser-reuse[llm]" extra'
        ) from exc
    return completion(**kwargs)
