"""A minimal provider-neutral chat model boundary."""

import os
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


class _LiteLLMChatModel:
    """Thin adapter around LiteLLM's provider-neutral completion call."""

    def __init__(self, model: str) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self._model = model

    def complete(self, messages: Sequence[Message]) -> str:
        response = _completion(
            model=self._model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("model response contains no choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise ValueError("model response contains no text content")
        return content


def load_chat_model(spec: str | None = None) -> ChatModel:
    """Resolve one model spec without exposing provider adapters to callers.

    Models use ``provider:model`` syntax and are delegated to LiteLLM.
    When omitted, the spec is read from ``BROWSER_REUSE_MODEL``.
    """

    configured = spec if spec is not None else os.environ.get("BROWSER_REUSE_MODEL", "")
    model_spec = configured.strip()
    if not model_spec:
        raise RuntimeError(
            "No model configured; set BROWSER_REUSE_MODEL to a 'provider:model' value"
        )

    provider, separator, model = model_spec.partition(":")
    provider, model = provider.strip(), model.strip()
    if not separator or not provider or not model:
        raise ValueError("model must use 'provider:model' syntax")
    return _LiteLLMChatModel(f"{provider}/{model}")


def _completion(**kwargs: object) -> object:
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            'LiteLLM is not installed; install the "browser-reuse[llm]" extra'
        ) from exc
    return completion(**kwargs)
