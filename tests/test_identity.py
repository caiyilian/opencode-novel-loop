from __future__ import annotations

import unittest

from dialoop.annotations import AnnotationRecord
from dialoop.identity import IdentityPipelineAgent
from dialoop.local_tools import DialogueIndex
from dialoop.model_client import ChatResult


class FakeIdentityClient:
    def __init__(self, responses: list[ChatResult]):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("identity model was called more times than expected")
        return self.responses.pop(0)


def sample_record(**overrides) -> AnnotationRecord:
    data = {
        "index": 0,
        "line_number": 1,
        "text": "Help.",
        "speaker": "\u5c11\u5973",
        "evidence_lines": [1],
        "reason": "Temporary identity at the dialogue line.",
        "rejected_candidates": [],
        "confidence": "medium",
        "tool_summary": {},
    }
    data.update(overrides)
    return AnnotationRecord(**data)


class IdentityPipelineAgentTest(unittest.TestCase):
    def test_model_locator_and_resolver_return_identity_review(self) -> None:
        dialogue_index = DialogueIndex.from_text(
            "\u5c11\u5973\u8bf4\uff1a\u300cHelp.\u300d\n"
            "\u5979\u4f4e\u58f0\u8bf4\uff1a\u300c\u6211\u53eb\u963f\u6d1b\u3002\u300d"
        )
        client = FakeIdentityClient(
            [
                ChatResult(
                    content=(
                        '{"candidates":[{"start_line":2,"end_line":2,"matched_line":2,'
                        '"suggested_names":["\u963f\u6d1b"],"reason":"line 2 gives a name"}],'
                        '"reason":"found later identity marker"}'
                    )
                ),
                ChatResult(
                    content=(
                        '{"verdict":"resolved","same_person":true,"recommended_speaker":"\u963f\u6d1b",'
                        '"evidence_lines":[2],"reason":"line 2 is the same girl naming herself",'
                        '"confidence":"high"}'
                    )
                ),
            ]
        )

        review = IdentityPipelineAgent(dialogue_index, model_client=client, lookahead_lines=5).review(sample_record())

        self.assertEqual(review["verdict"], "resolved")
        self.assertEqual(review["recommended_speaker"], "\u963f\u6d1b")
        self.assertEqual(review["evidence_lines"], [2])
        self.assertTrue(review["same_person"])
        self.assertTrue(review["model_agent"])
        self.assertEqual(review["candidate_ranges"][0]["matched_line"], 2)
        self.assertEqual(len(client.calls), 2)
        self.assertIsNone(client.calls[0]["tools"])
        self.assertIn("Identity Locator Agent", client.calls[0]["messages"][0].content)
        self.assertIn("Identity Resolver Agent", client.calls[1]["messages"][0].content)

    def test_model_identity_does_not_call_model_for_false_positive_speaker(self) -> None:
        dialogue_index = DialogueIndex.from_text("\u300cHelp.\u300d")
        client = FakeIdentityClient([])

        review = IdentityPipelineAgent(dialogue_index, model_client=client).review(
            sample_record(speaker="\u620f\u66f2\u6545\u4e8b\u91cc\u7684\u7537\u5b69")
        )

        self.assertFalse(review["triggered"])
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
