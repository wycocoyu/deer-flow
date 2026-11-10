"""OpenAI-compatible adapters that preserve structured response failures."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from deerflow.models.response_errors import EmptyModelResponseError


class OpenAIEmptyChoicesMixin:
    """Raise a typed error before LangChain indexes an empty choices list."""

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if isinstance(choices, (list, tuple)) and not choices:
            raise EmptyModelResponseError("Model returned an HTTP-success response with an empty choices list")
        return super()._create_chat_result(response, generation_info=generation_info)


def _as_content_parts(content: Any) -> list:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _mergeable_user_pair(previous: dict, message: dict) -> bool:
    if previous.get("role") != "user" or message.get("role") != "user":
        return False
    return {k: v for k, v in previous.items() if k != "content"} == {k: v for k, v in message.items() if k != "content"}


def merge_consecutive_user_payloads(messages: list[dict]) -> list[dict]:
    """Merge adjacent Chat Completions ``user`` entries into one, keeping order.

    Context middlewares place hidden user-role blocks (skill activation, durable
    context, memory, client and language context) next to the real user turn.
    OpenAI accepts that, but backends whose chat template enforces strict
    user/assistant alternation reject it with HTTP 400 (#6751). Entries are
    merged only when every non-content field matches, so a ``name`` boundary
    is never erased.
    """
    merged: list[dict] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if previous is None or not _mergeable_user_pair(previous, message):
            merged.append(message)
            continue
        if isinstance(previous.get("content"), str) and isinstance(message.get("content"), str):
            content: Any = f"{previous['content']}\n\n{message['content']}"
        else:
            content = [*_as_content_parts(previous.get("content")), *_as_content_parts(message.get("content"))]
        merged[-1] = {**previous, "content": content}
    return merged


class GuardedChatOpenAI(OpenAIEmptyChoicesMixin, ChatOpenAI):
    """ChatOpenAI with typed handling for valid HTTP responses with no choices."""

    merge_consecutive_user_messages: bool = Field(
        default=False,
        description="Merge adjacent role=user Chat Completions messages for backends that require strict role alternation.",
    )

    def _get_request_payload(self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Responses API payloads carry ``input`` instead of ``messages``; leave them alone.
        if self.merge_consecutive_user_messages and isinstance(payload.get("messages"), list):
            payload["messages"] = merge_consecutive_user_payloads(payload["messages"])
        return payload
