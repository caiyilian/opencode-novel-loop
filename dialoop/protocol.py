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


def local_tool_specs(
    submit_label_count: int | None = None,
    include_identity_tools: bool = True,
) -> list[ToolSpec]:
    speaker_items_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": submit_label_count or 1,
    }
    if submit_label_count is not None:
        speaker_items_schema["maxItems"] = submit_label_count

    specs = [
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
    ]

    if include_identity_tools:
        specs.extend(
            [
                ToolSpec(
                    name="locate_identity",
                    description=(
                        "Required before submit_labels when a trackable concrete person is only labeled by a temporary "
                        "identity such as a girl, young person, or old person. It scans later novel lines for bounded "
                        "identity-introduction ranges and returns candidates only; it never changes labels. Do not use "
                        "for first-person pronouns or verbal quirks such as 我/咱, and do not use for characters inside a "
                        "story, play, rumor, or quoted example narrated by the real speaker."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "minLength": 1},
                            "dialogue_index": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Optional dialogue index to anchor the search; defaults to the active batch.",
                            },
                            "search_after_line": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Optional 1-based line to start searching after.",
                            },
                            "lookahead_lines": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Optional bounded lookahead line count.",
                            },
                            "max_candidates": {"type": "integer", "minimum": 1},
                        },
                        "required": ["speaker"],
                        "additionalProperties": False,
                    },
                ),
                ToolSpec(
                    name="resolve_identity",
                    description=(
                        "Use immediately after locate_identity returns candidate ranges. Read a bounded candidate range and "
                        "decide whether it contains a stable name for the same temporary speaker. Returns resolved, "
                        "not_same_person, or not_enough_evidence style metadata; it never changes labels. A place, "
                        "organization, or story-internal character is not a resolved speaker identity."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "minLength": 1},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "dialogue_index": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Optional dialogue index for current-dialogue context.",
                            },
                        },
                        "required": ["speaker", "start_line", "end_line"],
                        "additionalProperties": False,
                    },
                ),
                ToolSpec(
                    name="record_character",
                    description=(
                        "Add or update a lightweight character library entry after a stable speaker name or alias is "
                        "supported by evidence. Use this before submit_labels when identity lookup resolves a new display "
                        "name. This auxiliary memory tool does not overwrite submitted labels."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "display_name": {"type": "string", "minLength": 1},
                            "aliases": {"type": "array", "items": {"type": "string"}},
                            "summary": {"type": "string"},
                            "evidence_lines": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 1},
                            },
                            "last_seen_dialogue_index": {"type": "integer", "minimum": 0},
                            "last_seen_line_number": {"type": "integer", "minimum": 1},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["display_name"],
                        "additionalProperties": False,
                    },
                ),
                ToolSpec(
                    name="normalize_speaker",
                    description=(
                        "Required before submit_labels when the character library is non-empty and the candidate speaker may "
                        "be an alias, short form, or temporary description. Returns a suggestion only; the model must still "
                        "submit the final speaker explicitly."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "minLength": 1},
                        },
                        "required": ["speaker"],
                        "additionalProperties": False,
                    },
                ),
                ToolSpec(
                    name="arbitrate_identity",
                    description=(
                        "Required when Labeler, Verifier, Identity Resolver, and Name Normalizer conclusions conflict. "
                        "Compares the evidence-backed conclusions and returns a recommendation only; it never writes labels."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "labeler_speaker": {"type": "string", "minLength": 1},
                            "verifier_verdict": {"type": "string", "enum": ["pass", "fail", "uncertain", "error"]},
                            "resolver_verdict": {
                                "type": "string",
                                "enum": ["resolved", "not_same_person", "not_enough_evidence"],
                            },
                            "resolver_speaker": {"type": "string"},
                            "normalizer_speaker": {"type": "string"},
                        },
                        "required": ["labeler_speaker"],
                        "additionalProperties": False,
                    },
                ),
            ]
        )

    submit_description = (
        "Submit one speaker name for each dialogue in the active batch, in order. "
        'Arguments must use the exact shape {"speakers":["speaker1", "..."]}. '
        "When possible, include evidence_lines, reason, rejected_candidates, and confidence. "
    )
    if include_identity_tools:
        submit_description += (
            "Do not submit a trackable temporary identity speaker before using locate_identity/resolve_identity. "
            "Do not submit an alias-like speaker before checking normalize_speaker when known characters exist. "
        )
    submit_description += "Do not include labels for previous_dialogues, following_dialogues, or raw context lines."

    specs.append(
        ToolSpec(
            name="submit_labels",
            description=submit_description,
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
    )
    return specs


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
