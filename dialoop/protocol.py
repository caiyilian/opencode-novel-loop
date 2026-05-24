from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    """Raised when model protocol data cannot be parsed or validated."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class JsonAction:
    action: str
    args: dict[str, Any]


def local_tool_specs(submit_label_count: int | None = None) -> list[ToolSpec]:
    speaker_items_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": submit_label_count or 1,
    }
    if submit_label_count is not None:
        speaker_items_schema["maxItems"] = submit_label_count

    return [
        ToolSpec(
            name="get_next_dialogue",
            description="Get the next unlabeled dialogue batch and current labeling progress.",
            parameters={
                "type": "object",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional batch size override for this request.",
                    }
                },
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="read_novel",
            description="Read the source novel by 1-based inclusive line range. The result includes line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["start_line", "end_line"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="search_novel",
            description="Search the source novel for an exact keyword and return matching line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="submit_labels",
            description=(
                "Submit one speaker name for each dialogue in the active batch, in order. "
                'Arguments must use the exact shape {"speakers":["speaker1", "..."]}. '
                "When possible, include evidence_lines, reason, rejected_candidates, and confidence. "
                "Do not include labels for previous_dialogues, following_dialogues, or raw context lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "speakers": speaker_items_schema,
                    "evidence_lines": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "description": "1-based novel line numbers that support the submitted speaker label(s).",
                    },
                    "evidence_lines_by_dialogue": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                        "description": "Optional per-dialogue evidence line numbers, aligned with speakers.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short reason for the submitted speaker label(s).",
                    },
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional per-dialogue reasons, aligned with speakers.",
                    },
                    "rejected_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Speaker candidates considered but rejected.",
                    },
                    "rejected_candidates_by_dialogue": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "description": "Optional per-dialogue rejected candidates, aligned with speakers.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence in the submitted speaker label(s).",
                    },
                    "confidences": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["high", "medium", "low"]},
                        "description": "Optional per-dialogue confidence values, aligned with speakers.",
                    },
                },
                "required": ["speakers"],
                "additionalProperties": False,
            },
        ),
    ]


def openai_tools_schema() -> list[dict[str, Any]]:
    return [spec.to_openai_tool() for spec in local_tool_specs()]


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ProtocolError("model output does not contain a JSON object")
    return stripped[start : end + 1]


def parse_json_action(text: str) -> JsonAction:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON action: {error}") from error

    if not isinstance(payload, dict):
        raise ProtocolError("JSON action must be an object")

    action = payload.get("action")
    args = payload.get("args", {})
    if not isinstance(action, str) or not action:
        raise ProtocolError("JSON action must include a non-empty string `action`")
    if not isinstance(args, dict):
        raise ProtocolError("JSON action `args` must be an object")

    known = {spec.name for spec in local_tool_specs()}
    if action not in known:
        raise ProtocolError(f"unknown JSON action: {action}")

    return JsonAction(action=action, args=args)
