from __future__ import annotations

import json
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
from dialoop.annotations import AnnotationStore
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

    def test_agent_writes_annotation_for_successful_submit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / ".dialoop" / "annotations.jsonl"
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
                                arguments={
                                    "speakers": ["Lawrence"],
                                    "evidence_lines": [1],
                                    "reason": "The narration says Lawrence said it.",
                                    "rejected_candidates": ["Holo"],
                                    "confidence": "high",
                                },
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=3),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            annotation = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(result.annotations_written, 1)
            self.assertEqual(annotation["speaker"], "Lawrence")
            self.assertEqual(annotation["evidence_lines"], [1])
            self.assertEqual(annotation["reason"], "The narration says Lawrence said it.")
            self.assertEqual(annotation["rejected_candidates"], ["Holo"])
            self.assertEqual(annotation["confidence"], "high")
            self.assertEqual(annotation["risk"]["level"], "low")
            self.assertIsNone(annotation["verifier"])
            self.assertEqual(annotation["coordinator_trace"][0]["agent"], "labeler")
            self.assertEqual(annotation["coordinator_trace"][-1]["agent"], "verifier")
            self.assertEqual(annotation["coordinator_trace"][-1]["action"], "skipped")
            self.assertEqual(annotation["tool_summary"]["read_novel"][0]["requested_start_line"], 1)

    def test_risk_mode_verifier_records_review_for_high_risk_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
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
                                arguments={
                                    "speakers": ["Lawrence"],
                                    "evidence_lines": [1],
                                    "reason": "Short line requires turn context.",
                                    "confidence": "low",
                                },
                            )
                        ],
                    ),
                    ChatResult(
                        content=(
                            '{"verdict":"pass","reason":"Evidence is enough.",'
                            '"counter_evidence_lines":[],"confidence":"high"}'
                        )
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=3, verifier_mode="risk"),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            annotation = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(annotation["risk"]["level"], "high")
            self.assertEqual(annotation["verifier"]["verdict"], "pass")
            self.assertIn("low_confidence", annotation["verifier"]["risk_signal_codes"])
            self.assertIn(
                ("verifier", "called"),
                [(event["agent"], event["action"]) for event in annotation["coordinator_trace"]],
            )
            self.assertEqual(annotation["coordinator_trace"][-1]["result"]["verdict"], "accept")

    def test_verifier_uncertain_records_arbiter_review_in_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
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
                                arguments={
                                    "speakers": ["Lawrence"],
                                    "evidence_lines": [1],
                                    "reason": "Short line requires turn context.",
                                    "confidence": "low",
                                },
                            )
                        ],
                    ),
                    ChatResult(
                        content=(
                            '{"verdict":"uncertain","reason":"Evidence is weak.",'
                            '"counter_evidence_lines":[],"confidence":"low"}'
                        )
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=3, verifier_mode="risk"),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            annotation = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertEqual(annotation["verifier"]["verdict"], "uncertain")
            self.assertEqual(annotation["verifier"]["confidence"], "low")
            self.assertEqual(annotation["arbiter"]["decision"], "needs_more_evidence")
            self.assertIn("Evidence is weak", annotation["arbiter"]["reason"])
            self.assertIn(
                ("arbiter", "uncertain"),
                [(event["agent"], event["action"]) for event in annotation["coordinator_trace"]],
            )

    def test_verifier_failure_blocks_write_and_requests_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
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
                                id="bad-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Holo"], "confidence": "low"},
                            )
                        ],
                    ),
                    ChatResult(
                        content=(
                            '{"verdict":"fail","reason":"The evidence does not support Holo.",'
                            '"counter_evidence_lines":[1]}'
                        )
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="good-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Lawrence"], "confidence": "low"},
                            )
                        ],
                    ),
                    ChatResult(
                        content=(
                            '{"verdict":"pass","reason":"Retry is supported.",'
                            '"counter_evidence_lines":[],"confidence":"high"}'
                        )
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=4, verifier_mode="risk"),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            annotation = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

            self.assertTrue(result.submitted)
            self.assertEqual(result.tool_steps, 3)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertEqual(result.tool_history[1].result["accepted"], False)
            self.assertIn("verifier rejected", result.tool_history[1].result["error"])
            self.assertEqual(result.tool_history[1].result["arbiter"]["decision"], "reject_labeler")
            self.assertIn("Verifier rejected", result.tool_history[1].result["arbiter"]["reason"])
            self.assertEqual(annotation["speaker"], "Lawrence")
            self.assertEqual(annotation["verifier"]["verdict"], "pass")
            self.assertEqual(annotation["recovery"]["blocked_reviews"][0]["arbiter"]["decision"], "reject_labeler")
            self.assertIn(
                "Verifier rejected",
                annotation["recovery"]["blocked_reviews"][0]["arbiter"]["reason"],
            )

    def test_repeated_fragile_verifier_block_unblocks_after_one_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
            text = "Holo answered: \u300c\u55ef\uff1f\u300d"
            tools = DialoopLocalTools(DialogueIndex.from_text(text), LabelStore(labels), batch_size=1)
            fragile_pass = ChatResult(
                content=(
                    '{"verdict":"pass","reason":"Turn order seems plausible.",'
                    '"counter_evidence_lines":[],"confidence":"high"}'
                )
            )
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
                                id="first-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Holo"]},
                            )
                        ],
                    ),
                    fragile_pass,
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="repeat-submit",
                                name="submit_labels",
                                arguments={"speakers": ["Holo"]},
                            )
                        ],
                    ),
                    fragile_pass,
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=4, verifier_mode="risk"),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            annotation = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

            self.assertTrue(result.submitted)
            self.assertEqual(result.tool_steps, 3)
            self.assertEqual(LabelStore(labels).labels(), ["Holo"])
            self.assertEqual(result.tool_history[1].result["accepted"], False)
            self.assertEqual(result.tool_history[2].result["accepted"], True)
            self.assertEqual(
                annotation["recovery"]["blocked_reviews"][0]["arbiter"]["block_reason_code"],
                "fragile_high_risk_pass",
            )
            self.assertFalse(annotation["arbiter"]["blocks_submission"])
            self.assertTrue(annotation["arbiter"]["unblocked_after_repeated_review"])
            self.assertIn(
                ("arbiter", "unblocked"),
                [(event["agent"], event["action"]) for event in annotation["coordinator_trace"]],
            )

    def test_agent_writes_multi_dialogue_annotations_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
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
                                id="call-submit",
                                name="submit_labels",
                                arguments={
                                    "speakers": ["Lawrence", "Holo"],
                                    "evidence_lines_by_dialogue": [[1], [2]],
                                    "reasons": ["Lawrence is named.", "Holo answered."],
                                    "rejected_candidates_by_dialogue": [["Holo"], ["Lawrence"]],
                                    "confidences": ["high", "medium"],
                                },
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=3),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(result.annotations_written, 2)
            self.assertEqual([row["speaker"] for row in rows], ["Lawrence", "Holo"])
            self.assertEqual([row["line_number"] for row in rows], [1, 2])
            self.assertEqual(rows[1]["evidence_lines"], [2])
            self.assertEqual(rows[1]["confidence"], "medium")

    def test_identity_tools_are_available_in_loop_and_annotation_summary(self) -> None:
        text = "\n".join(
            [
                "\u5c11\u5973\u8bf4\uff1a\u300cHelp.\u300d",
                "\u5979\u540e\u6765\u8bf4\uff1a\u300c\u6211\u53eb\u963f\u6d1b\u3002\u300d",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
            tools = DialoopLocalTools(DialogueIndex.from_text(text), LabelStore(labels), batch_size=1)
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="locate",
                                name="locate_identity",
                                arguments={"speaker": "\u5c11\u5973"},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="resolve",
                                name="resolve_identity",
                                arguments={"speaker": "\u5c11\u5973", "start_line": 1, "end_line": 2},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="record",
                                name="record_character",
                                arguments={
                                    "display_name": "\u963f\u6d1b",
                                    "aliases": ["\u5c11\u5973"],
                                    "evidence_lines": [2],
                                    "confidence": "high",
                                },
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="normalize",
                                name="normalize_speaker",
                                arguments={"speaker": "\u5c11\u5973"},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="arbitrate",
                                name="arbitrate_identity",
                                arguments={
                                    "labeler_speaker": "\u5c11\u5973",
                                    "resolver_verdict": "resolved",
                                    "resolver_speaker": "\u963f\u6d1b",
                                    "normalizer_speaker": "\u963f\u6d1b",
                                },
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="submit",
                                name="submit_labels",
                                arguments={"speakers": ["\u963f\u6d1b"], "evidence_lines": [2]},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=6),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            row = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])
            labels_text = labels.read_text(encoding="utf-8")

        self.assertTrue(result.submitted)
        self.assertEqual(labels_text, "\u963f\u6d1b\n")
        self.assertEqual(row["speaker"], "\u963f\u6d1b")
        self.assertEqual(row["tool_summary"]["locate_identity"][0]["candidate_count"], 1)
        self.assertEqual(row["tool_summary"]["resolve_identity"][0]["recommended_speaker"], "\u963f\u6d1b")
        self.assertEqual(row["tool_summary"]["record_character"][0]["display_name"], "\u963f\u6d1b")
        self.assertEqual(row["tool_summary"]["normalize_speaker"][0]["suggested_display_name"], "\u963f\u6d1b")
        self.assertEqual(row["tool_summary"]["arbitrate_identity"][0]["decision"], "use_resolved_identity")
        self.assertIn(
            "identity_resolver",
            [event["agent"] for event in row["coordinator_trace"]],
        )
        self.assertIn("normalizer", [event["agent"] for event in row["coordinator_trace"]])
        self.assertIn("arbiter", [event["agent"] for event in row["coordinator_trace"]])

    def test_coordinator_identity_agent_resolves_temporary_speaker_without_labeler_tool_call(self) -> None:
        text = "\n".join(
            [
                "\u5c11\u5973\u8bf4\uff1a\u300cHelp.\u300d",
                "\u5979\u540e\u6765\u4f4e\u58f0\u8bf4\uff1a\u300c\u6211\u53eb\u963f\u6d1b\u3002\u300d",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            annotations = Path(directory) / "annotations.jsonl"
            tools = DialoopLocalTools(
                DialogueIndex.from_text(text),
                LabelStore(labels),
                batch_size=1,
                identity_lookahead_lines=5,
            )
            client = FakeModelClient(
                [
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="read",
                                name="read_novel",
                                arguments={"start_line": 1, "end_line": 2},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="submit-temp",
                                name="submit_labels",
                                arguments={
                                    "speakers": ["\u5c11\u5973"],
                                    "evidence_lines": [1],
                                    "reason": "Only a temporary identity appears at the dialogue line.",
                                    "confidence": "medium",
                                },
                            )
                        ],
                    ),
                    ChatResult(
                        content=(
                            '{"candidates":[{"start_line":2,"end_line":2,"matched_line":2,'
                            '"suggested_names":["\u963f\u6d1b"],"reason":"line 2 gives a stable name"}],'
                            '"reason":"bounded later context has a name marker"}'
                        )
                    ),
                    ChatResult(
                        content=(
                            '{"verdict":"resolved","same_person":true,"recommended_speaker":"\u963f\u6d1b",'
                            '"evidence_lines":[2],"reason":"line 2 names the same temporary speaker",'
                            '"confidence":"high"}'
                        )
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="submit-resolved",
                                name="submit_labels",
                                arguments={
                                    "speakers": ["\u963f\u6d1b"],
                                    "evidence_lines": [2],
                                    "reason": "Identity Resolver found the stable name in bounded later context.",
                                    "rejected_candidates": ["\u5c11\u5973"],
                                    "confidence": "high",
                                },
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=4, verifier_mode="off"),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            stored_labels = LabelStore(labels).labels()
            row = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

        self.assertTrue(result.submitted)
        self.assertEqual(result.tool_steps, 3)
        self.assertEqual(stored_labels, ["\u963f\u6d1b"])
        self.assertEqual(result.tool_history[1].result["accepted"], False)
        self.assertIn("identity resolver", result.tool_history[1].result["error"])
        self.assertEqual(row["speaker"], "\u963f\u6d1b")
        self.assertEqual(row["recovery"]["blocked_reviews"][0]["identity"]["recommended_speaker"], "\u963f\u6d1b")
        self.assertEqual(row["recovery"]["blocked_reviews"][0]["identity"]["evidence_lines"], [2])
        self.assertTrue(row["recovery"]["blocked_reviews"][0]["identity"]["same_person"])
        self.assertEqual(
            row["recovery"]["blocked_reviews"][0]["identity"]["candidate_ranges"][0]["matched_line"],
            2,
        )
        self.assertEqual(
            row["recovery"]["blocked_reviews"][0]["arbiter"]["block_reason_code"],
            "identity_resolved_conflict",
        )
        self.assertIsNone(row["identity"])
        self.assertEqual(len(client.calls), 5)
        self.assertIn("Identity Locator Agent", client.calls[2]["messages"][0].content)
        self.assertIn("Identity Resolver Agent", client.calls[3]["messages"][0].content)

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
        self.assertIn("必须先调用 locate_identity", prompt)
        self.assertIn("record_character", prompt)
        self.assertIn("必须调用 arbitrate_identity", prompt)
        self.assertIn("不要把“我”“咱”“汝”“你”“您”等代词", prompt)
        self.assertIn("讲故事、转述戏曲", prompt)
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
        self.assertIn("身份工具检查", prompt)
        self.assertIn("必须先调用 locate_identity", prompt)
        self.assertIn("角色库检查", prompt)
        self.assertIn("故事、戏曲、传闻", prompt)

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

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=4, identity_mode="off"),
            ).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertEqual(result.tool_history[0].result["accepted"], False)
            self.assertIn("call read_novel or search_novel", result.tool_history[0].result["error"])
            self.assertIn("automatic_context", result.tool_history[0].result)

    def test_temporary_identity_submit_requires_identity_tool_before_write(self) -> None:
        text = "\n".join(
            [
                "\u5c11\u5973\u8bf4\uff1a\u300c\u6551\u547d\u3002\u300d",
                "\u5979\u540e\u6765\u8dd1\u8fdb\u68ee\u6797\u3002",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(text), LabelStore(labels), batch_size=1)
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
                                id="early-temp-submit",
                                name="submit_labels",
                                arguments={
                                    "speakers": ["\u5c11\u5973"],
                                    "evidence_lines": [1],
                                    "reason": "Only a temporary identity is known.",
                                    "confidence": "medium",
                                },
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="locate",
                                name="locate_identity",
                                arguments={"speaker": "\u5c11\u5973"},
                            )
                        ],
                    ),
                    ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="submit-after-lookup",
                                name="submit_labels",
                                arguments={
                                    "speakers": ["\u5c11\u5973"],
                                    "evidence_lines": [1],
                                    "reason": "Bounded identity lookup found no stable name, so keep the temporary identity.",
                                    "confidence": "medium",
                                },
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=4, identity_mode="off"),
            ).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["\u5c11\u5973"])
            self.assertEqual(result.tool_history[1].result["accepted"], False)
            self.assertIn("temporary identity speaker", result.tool_history[1].result["error"])
            self.assertIn("locate_identity", result.tool_history[1].result["instruction"])
            self.assertEqual(result.tool_history[2].name, "locate_identity")
            self.assertEqual(result.tool_history[3].result["accepted"], True)

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
            annotations = Path(directory) / "annotations.jsonl"
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

            result = AgentRunner(
                client,
                tools,
                AgentLoopConfig(protocol="tools", max_tool_steps=3),
                annotation_store=AnnotationStore(annotations),
            ).run_one_batch()
            annotation = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])

            self.assertTrue(result.submitted)
            self.assertEqual(result.tool_steps, 2)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])
            self.assertIn("with recovery", result.message)
            self.assertEqual(result.tool_history[1].result["accepted"], True)
            self.assertEqual(result.tool_history[1].result["ignored_speakers"], ["Holo", "Narrator"])
            self.assertEqual(annotation["speaker"], "Lawrence")
            self.assertEqual(annotation["recovery"]["ignored_speakers"], ["Holo", "Narrator"])
            self.assertEqual(annotation["tool_summary"]["submit_labels"][0]["ignored_speakers"], ["Holo", "Narrator"])

    def test_missing_submit_speakers_argument_gets_retry_instruction_and_can_recover(self) -> None:
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
                                id="missing-speakers",
                                name="submit_labels",
                                arguments={},
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
            self.assertEqual(result.tool_history[1].result["accepted"], False)
            self.assertIn("missing required argument: speakers", result.tool_history[1].result["error"])
            self.assertEqual(result.tool_history[1].result["expected_arguments"], {"speakers": ["<speaker>"]})
            sent_messages = [message.to_dict() for message in client.calls[2]["messages"]]
            self.assertTrue(any("Retry submit_labels now" in message["content"] for message in sent_messages))

    def test_single_dialogue_submit_accepts_speaker_alias_string(self) -> None:
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
                                id="alias-submit",
                                name="submit_labels",
                                arguments={"speaker": "Lawrence"},
                            )
                        ],
                    ),
                ]
            )

            result = AgentRunner(client, tools, AgentLoopConfig(protocol="tools", max_tool_steps=2)).run_one_batch()

            self.assertTrue(result.submitted)
            self.assertEqual(LabelStore(labels).labels(), ["Lawrence"])

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
                                arguments={"speakers": 123},
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
