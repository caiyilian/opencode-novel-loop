from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .local_tools import DialoopLocalTools, ToolValidationError
from .model_client import ChatMessage, ChatResult, ToolCall
from .protocol import JsonAction, ProtocolError, local_tool_specs, parse_json_action


class AgentLoopError(RuntimeError):
    """Raised when a batch cannot be completed by the agent loop."""


@dataclass(frozen=True)
class AgentLoopConfig:
    protocol: str = "auto"
    max_tool_steps: int = 20
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.protocol not in {"auto", "tools", "json"}:
            raise AgentLoopError(f"unsupported protocol: {self.protocol}")
        if self.max_tool_steps <= 0:
            raise AgentLoopError("max_tool_steps must be greater than 0")


@dataclass(frozen=True)
class ToolExecution:
    name: str
    result: dict[str, Any]

    @property
    def accepted_submit(self) -> bool:
        return self.name == "submit_labels" and self.result.get("accepted") is True


@dataclass(frozen=True)
class AgentBatchResult:
    submitted: bool
    done: bool
    tool_steps: int
    progress: dict[str, Any]
    message: str
    tool_history: list[ToolExecution] = field(default_factory=list)


def system_prompt(protocol: str) -> str:
    json_instruction = ""
    if protocol in {"auto", "json"}:
        json_instruction = (
            "\nIf native tool calls are unavailable, reply with exactly one JSON object in this form: "
            '{"action":"read_novel","args":{"start_line":1,"end_line":20}}. '
            "Use submit_labels when you are ready."
        )

    return (
        "You label the speaker of quoted dialogue in a novel. "
        "Use tools to read context, search the novel, and submit labels. "
        "Do not edit files directly. "
        "Submit exactly one speaker name for each dialogue in the active batch, in order. "
        "If context is insufficient, read more lines before submitting. "
        "Generic speakers such as Unknown, Crowd, Merchant, or Clerk are acceptable when the text is ambiguous."
        + json_instruction
    )


def batch_prompt(batch_result: dict[str, Any]) -> str:
    progress = batch_result["progress"]
    lines = [
        "Label the active dialogue batch.",
        f"Progress: {progress['labeled']}/{progress['total']} labeled, {progress['remaining']} remaining.",
        "",
        "Dialogues:",
    ]
    for offset, dialogue in enumerate(batch_result["dialogues"], start=1):
        lines.append(f"{offset}. index={dialogue['index']} line={dialogue['line_number']} text={dialogue['text']}")
    lines.extend(
        [
            "",
            "Use read_novel or search_novel if needed.",
            "Finish this batch by calling submit_labels with the speaker names in the same order.",
        ]
    )
    return "\n".join(lines)


class AgentRunner:
    def __init__(
        self,
        model_client: Any,
        tools: DialoopLocalTools,
        config: Optional[AgentLoopConfig] = None,
    ):
        self.model_client = model_client
        self.tools = tools
        self.config = config or AgentLoopConfig()

    def run_one_batch(self) -> AgentBatchResult:
        initial_batch = self.tools.get_next_dialogue()
        if initial_batch["done"]:
            return AgentBatchResult(
                submitted=False,
                done=True,
                tool_steps=0,
                progress=initial_batch["progress"],
                message="all dialogues are already labeled",
            )

        messages = [
            ChatMessage(role="system", content=system_prompt(self.config.protocol)),
            ChatMessage(role="user", content=batch_prompt(initial_batch)),
        ]
        history: list[ToolExecution] = []

        for step in range(1, self.config.max_tool_steps + 1):
            response = self._chat(messages)
            executions = self._handle_response(messages, response)
            history.extend(executions)

            for execution in executions:
                if execution.accepted_submit:
                    progress = execution.result["progress"]
                    return AgentBatchResult(
                        submitted=True,
                        done=progress["remaining"] == 0,
                        tool_steps=step,
                        progress=progress,
                        message="submitted labels for one batch",
                        tool_history=history,
                    )

        progress = self.tools.get_progress()
        raise AgentLoopError(
            f"model did not submit labels within {self.config.max_tool_steps} tool step(s); "
            f"progress remains {progress['labeled']}/{progress['total']}"
        )

    def _chat(self, messages: list[ChatMessage]) -> ChatResult:
        tools = None if self.config.protocol == "json" else local_tool_specs()
        return self.model_client.chat(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
        )

    def _handle_response(self, messages: list[ChatMessage], response: ChatResult) -> list[ToolExecution]:
        if response.tool_calls:
            if self.config.protocol == "json":
                messages.append(
                    ChatMessage(
                        role="user",
                        content="Native tool calls were returned while protocol=json. Reply with a JSON action object instead.",
                    )
                )
                return []

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                )
            )
            return [self._execute_native_tool_call(messages, call) for call in response.tool_calls]

        messages.append(ChatMessage(role="assistant", content=response.content))

        if self.config.protocol in {"auto", "json"}:
            action = self._parse_json_action_or_remind(messages, response.content)
            if action is not None:
                execution = self._execute_json_action(action)
                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"Tool result for {execution.name}: {format_tool_result(execution.result)}",
                    )
                )
                return [execution]

        messages.append(
            ChatMessage(
                role="user",
                content="You must use a tool or JSON action and finish by calling submit_labels for the active batch.",
            )
        )
        return []

    def _parse_json_action_or_remind(self, messages: list[ChatMessage], content: str) -> Optional[JsonAction]:
        try:
            return parse_json_action(content)
        except ProtocolError as error:
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Your response was not a valid JSON action: {error}. Reply with exactly one action object.",
                )
            )
            return None

    def _execute_native_tool_call(self, messages: list[ChatMessage], call: ToolCall) -> ToolExecution:
        execution = self._execute_tool(call.name, call.arguments)
        messages.append(
            ChatMessage(
                role="tool",
                content=format_tool_result(execution.result),
                tool_call_id=call.id,
            )
        )
        return execution

    def _execute_json_action(self, action: JsonAction) -> ToolExecution:
        return self._execute_tool(action.action, action.args)

    def _execute_tool(self, name: str, args: dict[str, Any]) -> ToolExecution:
        try:
            if name == "get_next_dialogue":
                result = self.tools.get_next_dialogue(batch_size=optional_int(args, "batch_size"))
            elif name == "read_novel":
                result = self.tools.read_novel(
                    start_line=required_int(args, "start_line"),
                    end_line=required_int(args, "end_line"),
                )
            elif name == "search_novel":
                result = self.tools.search_novel(
                    keyword=required_str(args, "keyword"),
                    limit=optional_int(args, "limit"),
                )
            elif name == "submit_labels":
                result = self.tools.submit_labels(speakers=required_str_list(args, "speakers"))
            else:
                raise AgentLoopError(f"unknown tool call: {name}")
        except KeyError as error:
            result = {"accepted": False, "error": f"missing required argument: {error.args[0]}"}
        except ToolValidationError as error:
            result = {"accepted": False, "error": str(error)}

        return ToolExecution(name=name, result=result)


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def required_int(args: dict[str, Any], name: str) -> int:
    value = args[name]
    if type(value) is not int:
        raise ToolValidationError(f"{name} must be an integer")
    return value


def optional_int(args: dict[str, Any], name: str) -> Optional[int]:
    if name not in args or args[name] is None:
        return None
    return required_int(args, name)


def required_str(args: dict[str, Any], name: str) -> str:
    value = args[name]
    if not isinstance(value, str):
        raise ToolValidationError(f"{name} must be a string")
    return value


def required_str_list(args: dict[str, Any], name: str) -> list[str]:
    value = args[name]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ToolValidationError(f"{name} must be a list of strings")
    return value
