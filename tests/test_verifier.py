from __future__ import annotations

import unittest

from dialoop.annotations import AnnotationRecord
from dialoop.model_client import ChatResult
from dialoop.risk import assess_annotation_risk
from dialoop.verifier import VerifierAgent, review_from_payload, verifier_messages


class FakeVerifierClient:
    def __init__(self, responses: ChatResult | list[ChatResult]):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("verifier was called more times than expected")
        return self.responses.pop(0)


def sample_record() -> AnnotationRecord:
    return AnnotationRecord(
        index=0,
        line_number=1,
        text="你呢？",
        speaker="甲",
        evidence_lines=[1],
        reason="The previous turn names the speaker.",
        rejected_candidates=["乙"],
        confidence="low",
        tool_summary={},
    )


class VerifierTest(unittest.TestCase):
    def test_verifier_messages_request_json_without_tools(self) -> None:
        record = sample_record()
        messages = verifier_messages(record, assess_annotation_risk(record))

        self.assertEqual(messages[0].role, "system")
        self.assertIn("Verifier Agent", messages[0].content)
        self.assertIn('"verdict"', messages[1].content)
        self.assertIn('"risk"', messages[1].content)

    def test_review_from_payload_keeps_counter_evidence_lines(self) -> None:
        record = sample_record()
        risk = assess_annotation_risk(record)

        review = review_from_payload(
            {
                "verdict": "fail",
                "reason": "Line 3 contradicts the submitted speaker.",
                "counter_evidence_lines": [3, 3, "bad", -1],
            },
            risk,
        )

        self.assertTrue(review.blocks_submission)
        self.assertEqual(review.counter_evidence_lines, [3])
        self.assertIn("low_confidence", review.risk_signal_codes)

    def test_agent_returns_error_review_for_invalid_json(self) -> None:
        record = sample_record()
        risk = assess_annotation_risk(record)
        client = FakeVerifierClient(ChatResult(content="not json"))

        review = VerifierAgent(client).verify(record, risk)

        self.assertEqual(review.verdict, "error")
        self.assertFalse(review.blocks_submission)
        self.assertEqual(len(client.calls), 2)
        self.assertIsNone(client.calls[0]["tools"])
        self.assertIn("not usable JSON", client.calls[1]["messages"][1].content)

    def test_agent_retries_invalid_json_with_compact_repair_prompt(self) -> None:
        record = sample_record()
        risk = assess_annotation_risk(record)
        client = FakeVerifierClient(
            [
                ChatResult(content='```json\n{"verdict":"pass", "reason": "cut off"'),
                ChatResult(content='{"verdict":"pass","reason":"Retry fixed it.","counter_evidence_lines":[]}'),
            ]
        )

        review = VerifierAgent(client).verify(record, risk)

        self.assertEqual(review.verdict, "pass")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["max_tokens"], 1200)


if __name__ == "__main__":
    unittest.main()
