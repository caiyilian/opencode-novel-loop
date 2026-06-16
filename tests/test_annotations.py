from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dialoop.annotations import AnnotationRecord, AnnotationStore, build_annotation_records


class AnnotationRecordTest(unittest.TestCase):
    def test_serializes_to_json_line(self) -> None:
        record = AnnotationRecord(
            index=1,
            line_number=12,
            text="Hello.",
            speaker="Lawrence",
            evidence_lines=[10, 12],
            reason="Narration says Lawrence answered.",
            rejected_candidates=["Holo"],
            confidence="high",
            tool_summary={"read_novel": [{"start_line": 10, "end_line": 12}]},
            recovery=None,
        )

        loaded = json.loads(record.to_json_line())

        self.assertEqual(loaded["index"], 1)
        self.assertEqual(loaded["speaker"], "Lawrence")
        self.assertEqual(loaded["evidence_lines"], [10, 12])
        self.assertIsNone(loaded["recovery"])
        self.assertIsNone(loaded["arbiter"])
        self.assertIsNone(loaded["coordinator_trace"])

    def test_serializes_coordinator_trace(self) -> None:
        record = AnnotationRecord(
            index=1,
            line_number=12,
            text="Hello.",
            speaker="Lawrence",
            evidence_lines=[12],
            reason="Narration says Lawrence answered.",
            rejected_candidates=[],
            confidence="high",
            tool_summary={},
            coordinator_trace=[
                {
                    "step": 1,
                    "agent": "labeler",
                    "action": "accepted",
                    "reason": "Labeler submitted a speaker candidate.",
                }
            ],
        )

        loaded = json.loads(record.to_json_line())

        self.assertEqual(loaded["coordinator_trace"][0]["agent"], "labeler")
        self.assertEqual(loaded["coordinator_trace"][0]["action"], "accepted")

    def test_serializes_arbiter_review(self) -> None:
        record = AnnotationRecord(
            index=1,
            line_number=12,
            text="Hello.",
            speaker="Lawrence",
            evidence_lines=[12],
            reason="Narration says Lawrence answered.",
            rejected_candidates=[],
            confidence="high",
            tool_summary={},
            arbiter={
                "decision": "reject_labeler",
                "reason": "Verifier found counter-evidence.",
                "blocks_submission": True,
            },
        )

        loaded = json.loads(record.to_json_line())

        self.assertEqual(loaded["arbiter"]["decision"], "reject_labeler")
        self.assertTrue(loaded["arbiter"]["blocks_submission"])

    def test_serializes_identity_review(self) -> None:
        record = AnnotationRecord(
            index=1,
            line_number=12,
            text="Hello.",
            speaker="\u5c11\u5973",
            evidence_lines=[12],
            reason="Temporary speaker.",
            rejected_candidates=[],
            confidence="medium",
            tool_summary={},
            identity={
                "verdict": "resolved",
                "recommended_speaker": "\u963f\u6d1b",
                "evidence_lines": [20],
            },
        )

        loaded = json.loads(record.to_json_line())

        self.assertEqual(loaded["identity"]["verdict"], "resolved")
        self.assertEqual(loaded["identity"]["recommended_speaker"], "\u963f\u6d1b")

    def test_store_appends_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".dialoop" / "annotations.jsonl"
            store = AnnotationStore(path)
            record = AnnotationRecord(
                index=0,
                line_number=1,
                text="Hi.",
                speaker="Holo",
                evidence_lines=[1],
                reason="Direct speech.",
                rejected_candidates=[],
                confidence="medium",
                tool_summary={},
            )

            self.assertEqual(store.append([record]), 1)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["speaker"], "Holo")

    def test_build_records_supports_per_dialogue_fields_and_fallbacks(self) -> None:
        records = build_annotation_records(
            dialogues=[
                {"index": 0, "line_number": 4, "text": "One."},
                {"index": 1, "line_number": 5, "text": "Two."},
            ],
            speakers=["A", "B"],
            submit_args={
                "speakers": ["A", "B"],
                "evidence_lines_by_dialogue": [[3, 4], [5]],
                "reasons": ["A said it.", "B answered."],
                "rejected_candidates_by_dialogue": [["B"], ["A"]],
                "confidences": ["high", "medium"],
            },
            tool_summary={"read_novel": []},
        )

        self.assertEqual([record.speaker for record in records], ["A", "B"])
        self.assertEqual(records[0].evidence_lines, [3, 4])
        self.assertEqual(records[1].reason, "B answered.")
        self.assertEqual(records[1].rejected_candidates, ["A"])
        self.assertEqual(records[1].confidence, "medium")

    def test_build_records_falls_back_when_model_omits_evidence(self) -> None:
        records = build_annotation_records(
            dialogues=[{"index": 0, "line_number": 9, "text": "Maybe."}],
            speakers=["未知"],
            submit_args={"speakers": ["未知"]},
            tool_summary={"read_novel": [{"start_line": 1, "end_line": 10}]},
        )

        self.assertEqual(records[0].evidence_lines, [9])
        self.assertEqual(records[0].confidence, "low")
        self.assertIn("fallback annotation", records[0].reason)


if __name__ == "__main__":
    unittest.main()
