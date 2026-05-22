from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path

from dialoop.agent_loop import (
    AgentLoopConfig,
    AgentLoopError,
    AgentRunner,
    ToolExecution,
    batch_prompt,
    format_prompt_messages,
    format_recent_tools,
    system_prompt,
)
from dialoop.local_tools import DialogueIndex, DialoopLocalTools, LabelStore
from dialoop.model_client import ChatMessage, ChatResult, ToolCall


SAMPLE_TEXT = "\n".join(
    [
        "Lawrence said: \u300cHello.\u300d",
        "Holo answered: \u300cHi.\u300d",
        "Narration only.",
    ]
)


class FakeModelClient:
    def __init__(self, responses: list[ChatResult]):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("model was called more times than expected")
        return self.responses.pop(0)


class AgentLoopTest(unittest.TestCase):
    def test_native_tool_loop_reads_context_and_submits_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 2},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=3)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertFalse(result.done)
            self.assertEqual(result.tool_steps, 2)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertIsNotNone(client.calls[0]["tools"])
            submit_spec = next(spec for spec in client.calls[0]["tools"] if spec.name == "submit_labels")
            self.assertEqual(submit_spec.parameters["properties"]["speakers"]["maxItems"], 1)
            sent_messages = [message.to_dict() for message in client.calls[1]["messages"]]
            self.assertIn("简体中文", sent_messages[0]["content"])
            self.assertIn("第一步请调用 read_novel", sent_messages[1]["content"])
            self.assertIn("tool_calls", sent_messages[2])
            self.assertEqual(sent_messages[3]["role"], "tool")
            self.assertEqual(sent_messages[3]["tool_call_id"], "call-read")

    def test_json_action_loop_submits_labels_without_native_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(content='{"action":"read_novel","args":{"start_line":1,"end_line":1}}'),
                    ChatResult(content='{"action":"submit_labels","args":{"speakers":["Lawrence"]}}'),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="json", max_tool_steps=3)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertIsNone(client.calls[0]["tools"])

    def test_prompt_output_prints_initial_messages_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 1},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                ]
            )
            output = StringIO()

            AgentRunner(client, tools, AgentLoopConfig(protocol="tools"), prompt_output=output).run_one_batch()

            printed = output.getvalue()
            self.assertIn("Dialoop prompt:", printed)
            self.assertIn("--- system ---", printed)
            self.assertIn("--- user ---", printed)
            self.assertIn("read_novel", printed)
            self.assertNotIn("Tool result", printed)

    def test_format_prompt_messages_labels_roles(self) -> None:
        rendered = format_prompt_messages(
            [
                ChatMessage(role="system", content="system text"),
                ChatMessage(role="user", content="user text"),
            ]
        )

        self.assertIn("--- system ---\nsystem text", rendered)
        self.assertIn("--- user ---\nuser text", rendered)

    def test_prompt_rules_cover_identity_lookahead_and_non_character_sounds(self) -> None:
        prompt = system_prompt("auto")

        self.assertIn("有限范围内揭示其姓名或稳定称呼", prompt)
        self.assertIn("不要为了无名群体", prompt)
        self.assertIn("非人物发声", prompt)
        self.assertIn("短句、追问、省略号", prompt)

    def test_batch_prompt_uses_configured_context_window(self) -> None:
        prompt = batch_prompt(
            {
                "progress": {"labeled": 0, "total": 1, "remaining": 1},
                "dialogues": [{"index": 0, "line_number": 100, "text": "你好。"}],
            },
            context_window_lines=40,
        )

        self.assertIn("read_novel(start_line=60, end_line=140)", prompt)
        self.assertIn("必须且只能提交 1 个 speaker", prompt)

    def test_batch_prompt_includes_neighbor_dialogues(self) -> None:
        prompt = batch_prompt(
            {
                "progress": {"labeled": 2, "total": 4, "remaining": 2},
                "dialogues": [{"index": 2, "line_number": 10, "text": "为什么？"}],
                "previous_dialogues": [
                    {"index": 0, "line_number": 8, "text": "你好。", "speaker": "甲"},
                    {"index": 1, "line_number": 9, "text": "你好呀。", "speaker": "乙"},
                ],
                "following_dialogues": [{"index": 3, "line_number": 11, "text": "因为如此。"}],
            },
            context_window_lines=5,
        )

        self.assertIn("最近已标注对话", prompt)
        self.assertIn("speaker=甲", prompt)
        self.assertIn("后续未标注对话", prompt)
        self.assertIn("index=3", prompt)

    def test_config_rejects_invalid_context_window(self) -> None:
        with self.assertRaises(AgentLoopError):
            AgentLoopConfig(context_window_lines=0)

    def test_submit_before_context_is_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="early-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 1},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="good-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=4)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertEqual(result.tool_history[0].result["accepted"], False)
            self.assertIn("call read_novel or search_novel", result.tool_history[0].result["error"])
            self.assertIn("automatic_context", result.tool_history[0].result)

    def test_repeated_submit_before_context_can_recover_without_looping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="early-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="retry-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=2)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(result.tool_steps, 2)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertEqual(result.tool_history[0].result["accepted"], False)
            self.assertIn("automatic_context", result.tool_history[0].result)
            self.assertEqual(result.tool_history[1].result["accepted"], True)

    def test_submit_validation_error_is_returned_to_model_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=2)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 2},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="bad-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="good-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence", "Holo"]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=3)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence", "Holo"])
            self.assertEqual(result.tool_history[1].result["accepted"], False)
            self.assertIn("speaker count mismatch", result.tool_history[1].result["error"])
            self.assertEqual(result.tool_history[1].result["expected_count"], 2)

    def test_extra_submit_labels_are_recovered_by_using_active_batch_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 2},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="extra-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence", "Holo", "Narrator"]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=3)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(result.tool_steps, 2)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertIn("with recovery", result.message)
            self.assertEqual(result.tool_history[1].result["accepted"], True)
            self.assertEqual(result.tool_history[1].result["ignored_speakers"], ["Holo", "Narrator"])

    def test_bad_tool_argument_type_is_returned_to_model_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 1},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="bad-submit",
                                name="submit_labels",
                                arguments={"speakers": "Lawrence"},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="good-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=3)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertIn("speakers must be a list of strings", result.tool_history[1].result["error"])

    def test_done_batch_does_not_call_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            labels.write_text("Lawrence\nHolo\n", encoding="utf-8")
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient([])

            result = AgentRunner(client, tools).run_one_batch()

            self.assertFalse(result.submitted)
            self.assertTrue(result.done)
            self.assertEqual(client.calls, [])

    def test_raises_when_model_never_submits_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient([ChatResult(content="I think it is Lawrence.")])

            with self.assertRaises(AgentLoopError) as raised:
                AgentRunner(client, tools, AgentLoopConfig(protocol="auto", max_tool_steps=1)).run_one_batch()

            self.assertIn("active batch: index=0", str(raised.exception))
            self.assertIn("recent tools: none", str(raised.exception))
            self.assertEqual(LabelStore(labels).labels(), [])

    def test_warns_before_final_tool_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 1},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"]},
                            )
                        ],
                    ),
                ]
            )

            AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=2)).run_one_batch()

            sent_messages = [message.to_dict() for message in client.calls[1]["messages"]]
            self.assertTrue(
                any("Only one model response remains" in message["content"] for message in sent_messages)
            )

    def test_format_recent_tools_includes_error_reason(self) -> None:
        rendered = format_recent_tools(
            [
                ToolExecution("submit_labels", {"accepted": False, "error": "call read_novel first"}),
                ToolExecution("submit_labels", {"accepted": True}),
            ]
        )

        self.assertIn("submit_labels(error=call read_novel first)", rendered)
        self.assertIn("submit_labels(accepted=true)", rendered)


if __name__ == "__main__":
    unittest.main()
