from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dialoop.agent_loop import AgentLoopConfig, AgentLoopError, AgentRunner
from dialoop.local_tools import DialogueIndex, DialoopLocalTools, LabelStore
from dialoop.model_client import ChatResult, ToolCall


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
            sent_messages = [message.to_dict() for message in client.calls[1]["messages"]]
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
            self.assertEqual(result.tool_history[0].result["accepted"], False)
            self.assertIn("speaker count mismatch", result.tool_history[0].result["error"])

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
            self.assertIn("speakers must be a list of strings", result.tool_history[0].result["error"])

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

            with self.assertRaises(AgentLoopError):
                AgentRunner(client, tools, AgentLoopConfig(protocol="auto", max_tool_steps=1)).run_one_batch()

            self.assertEqual(LabelStore(labels).labels(), [])


if __name__ == "__main__":
    unittest.main()
