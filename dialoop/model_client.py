from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from .protocol import ToolSpec


DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:32b"


class ModelClientError(RuntimeError):
    """Base error for model client failures."""


class ModelHTTPError(ModelClientError):
    """Raised when an OpenAI-compatible endpoint returns a non-2xx response."""


class ModelResponseError(ModelClientError):
    """Raised when the endpoint response shape is not usable."""


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    timeout: float = 60.0


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConnectionStatus:
    ok: bool
    message: str
    model: str
    base_url: str


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def chat_completions_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def chat(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ToolSpec]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in messages],
        }
        if tools:
            body["tools"] = [tool.to_openai_tool() for tool in tools]
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        payload = self._post_json(self.chat_completions_url, body)
        return self._parse_chat_result(payload)

    def check_connection(self) -> ModelConnectionStatus:
        try:
            result = self.chat(
                messages=[
                    ChatMessage(role="system", content="You are a connection checker."),
                    ChatMessage(role="user", content="Reply with OK."),
                ],
                temperature=0,
                max_tokens=8,
            )
        except ModelClientError as error:
            return ModelConnectionStatus(
                ok=False,
                message=str(error),
                model=self.config.model,
                base_url=self.config.base_url,
            )

        content = result.content.strip()
        return ModelConnectionStatus(
            ok=True,
            message=content or "connection succeeded",
            model=self.config.model,
            base_url=self.config.base_url,
        )

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ModelHTTPError(f"HTTP {error.code} from model endpoint: {detail}") from error
        except urllib.error.URLError as error:
            raise ModelClientError(f"model endpoint connection failed: {error.reason}") from error
        except TimeoutError as error:
            raise ModelClientError("model endpoint request timed out") from error

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ModelResponseError(f"model endpoint returned invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ModelResponseError("model endpoint returned a non-object JSON response")
        return payload

    def _parse_chat_result(self, payload: dict[str, Any]) -> ChatResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseError("model response missing choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ModelResponseError("model response choice must be an object")

        message = choice.get("message")
        if not isinstance(message, dict):
            raise ModelResponseError("model response choice missing message")

        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ModelResponseError("model response message content must be a string")

        return ChatResult(
            content=content,
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
            raw=payload,
        )

    def _parse_tool_calls(self, value: Any) -> list[ToolCall]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ModelResponseError("model response tool_calls must be a list")

        calls: list[ToolCall] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ModelResponseError("tool call must be an object")

            function = item.get("function")
            if not isinstance(function, dict):
                raise ModelResponseError("tool call missing function object")

            name = function.get("name")
            raw_arguments = function.get("arguments") or "{}"
            if not isinstance(name, str) or not name:
                raise ModelResponseError("tool call function missing name")
            if not isinstance(raw_arguments, str):
                raise ModelResponseError("tool call function arguments must be a JSON string")

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise ModelResponseError(f"tool call arguments are invalid JSON: {error}") from error
            if not isinstance(arguments, dict):
                raise ModelResponseError("tool call arguments must decode to an object")

            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"tool-call-{index}"),
                    name=name,
                    arguments=arguments,
                )
            )
        return calls
