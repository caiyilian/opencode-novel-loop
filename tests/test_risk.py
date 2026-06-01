from __future__ import annotations

import unittest

from dialoop.annotations import AnnotationRecord, DEFAULT_REASON
from dialoop.risk import assess_annotation_risk


def record(
    text: str,
    confidence: str = "high",
    reason: str = "Direct narration supports the speaker.",
    rejected_candidates: list[str] | None = None,
) -> AnnotationRecord:
    return AnnotationRecord(
        index=0,
        line_number=1,
        text=text,
        speaker="甲",
        evidence_lines=[1],
        reason=reason,
        rejected_candidates=["乙"] if rejected_candidates is None else rejected_candidates,
        confidence=confidence,
        tool_summary={},
    )


class RiskAssessmentTest(unittest.TestCase):
    def test_low_confidence_and_missing_metadata_are_high_risk(self) -> None:
        assessment = assess_annotation_risk(
            record(
                text="也许。",
                confidence="low",
                reason=DEFAULT_REASON,
                rejected_candidates=[],
            )
        )

        self.assertEqual(assessment.level, "high")
        self.assertTrue(assessment.needs_verifier)
        self.assertIn("low_confidence", [signal.code for signal in assessment.signals])
        self.assertIn("fallback_annotation_metadata", [signal.code for signal in assessment.signals])

    def test_second_person_short_dialogue_is_high_risk(self) -> None:
        assessment = assess_annotation_risk(record(text="你呢？", confidence="high"))

        self.assertEqual(assessment.level, "high")
        self.assertIn("second_person_address", [signal.code for signal in assessment.signals])
        self.assertIn("short_question", [signal.code for signal in assessment.signals])

    def test_archaic_second_person_marker_is_high_risk(self) -> None:
        assessment = assess_annotation_risk(record(text="汝曾去过吗？", confidence="high"))

        self.assertEqual(assessment.level, "high")
        self.assertIn("second_person_address", [signal.code for signal in assessment.signals])

    def test_repeated_call_is_high_risk(self) -> None:
        assessment = assess_annotation_risk(record(text="甲！甲！甲！", confidence="high"))

        self.assertEqual(assessment.level, "high")
        self.assertIn("repeated_call", [signal.code for signal in assessment.signals])

    def test_specific_well_evidenced_dialogue_is_low_risk(self) -> None:
        assessment = assess_annotation_risk(
            record(text="This sentence is directly attributed by the surrounding narration.")
        )

        self.assertEqual(assessment.level, "low")
        self.assertFalse(assessment.needs_verifier)
        self.assertEqual(assessment.signals, [])


if __name__ == "__main__":
    unittest.main()
