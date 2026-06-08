from __future__ import annotations

import unittest

from dialoop.annotations import AnnotationRecord
from dialoop.coordinator import AgentResult, Coordinator
from dialoop.risk import RiskAssessment, RiskSignal
from dialoop.verifier import VerifierReview


class FakeVerifierAgent:
    def __init__(self, review: VerifierReview):
        self.review = review
        self.calls: list[tuple[AnnotationRecord, RiskAssessment]] = []

    def verify(self, record: AnnotationRecord, risk: RiskAssessment) -> VerifierReview:
        self.calls.append((record, risk))
        return self.review


def sample_record(**overrides) -> AnnotationRecord:
    data = {
        "index": 0,
        "line_number": 10,
        "text": "Short.",
        "speaker": "A",
        "evidence_lines": [10],
        "reason": "Nearby narration names A.",
        "rejected_candidates": ["B"],
        "confidence": "high",
        "tool_summary": {},
    }
    data.update(overrides)
    return AnnotationRecord(**data)


def risk(level: str, code: str | None = None) -> RiskAssessment:
    signals = []
    if code is not None:
        signals.append(RiskSignal(code=code, message=f"{code} message", level=level))
    return RiskAssessment(level=level, signals=signals)


class CoordinatorTest(unittest.TestCase):
    def test_agent_result_schema_matches_stage_six_contract(self) -> None:
        result = AgentResult(
            agent="verifier",
            verdict="reject",
            recommended_speaker="A",
            evidence_lines=[10, 10, "bad"],
            counter_evidence_lines=[12],
            reason="Counter-evidence contradicts the submitted speaker.",
            confidence="high",
        )

        self.assertEqual(
            set(result.to_dict()),
            {
                "agent",
                "verdict",
                "recommended_speaker",
                "evidence_lines",
                "counter_evidence_lines",
                "reason",
                "confidence",
            },
        )
        self.assertEqual(result.to_dict()["evidence_lines"], [10])
        self.assertEqual(result.to_dict()["counter_evidence_lines"], [12])

        with self.assertRaises(ValueError):
            AgentResult(agent="coordinator", verdict="accept", reason="bad")
        with self.assertRaises(ValueError):
            AgentResult(agent="verifier", verdict="pass", reason="bad")

    def test_risk_mode_calls_verifier_for_high_risk_annotation(self) -> None:
        verifier = FakeVerifierAgent(
            VerifierReview(
                enabled=True,
                verdict="pass",
                reason="Evidence is enough.",
                counter_evidence_lines=[],
                risk_signal_codes=["low_confidence"],
                confidence="high",
            )
        )

        decision = Coordinator(verifier, verifier_mode="risk").review(
            sample_record(confidence="low"),
            risk("high", "low_confidence"),
        )
        trace = decision.trace_dicts()

        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(decision.verifier["verdict"], "pass")
        self.assertFalse(decision.blocks_submission)
        self.assertIn(("verifier", "called"), [(event["agent"], event["action"]) for event in trace])
        self.assertIn(("verifier", "accepted"), [(event["agent"], event["action"]) for event in trace])
        self.assertEqual(trace[-1]["result"]["verdict"], "accept")

    def test_low_confidence_high_risk_verifier_pass_goes_to_arbiter_and_blocks(self) -> None:
        verifier = FakeVerifierAgent(
            VerifierReview(
                enabled=True,
                verdict="pass",
                reason="Weak pass.",
                counter_evidence_lines=[],
                risk_signal_codes=["low_confidence"],
                confidence="low",
            )
        )

        decision = Coordinator(verifier, verifier_mode="risk").review(
            sample_record(confidence="low"),
            risk("high", "low_confidence"),
        )
        trace = decision.trace_dicts()

        self.assertTrue(decision.blocks_submission)
        self.assertEqual(decision.verifier["verdict"], "pass")
        self.assertEqual(decision.verifier["confidence"], "low")
        self.assertEqual(decision.arbiter["decision"], "needs_more_evidence")
        self.assertTrue(decision.arbiter["blocks_submission"])
        self.assertIn(("arbiter", "called"), [(event["agent"], event["action"]) for event in trace])
        self.assertIn(("arbiter", "uncertain"), [(event["agent"], event["action"]) for event in trace])

    def test_risk_mode_skips_verifier_for_low_risk_annotation(self) -> None:
        verifier = FakeVerifierAgent(
            VerifierReview(
                enabled=True,
                verdict="fail",
                reason="Should not be called.",
                counter_evidence_lines=[],
                risk_signal_codes=[],
            )
        )

        decision = Coordinator(verifier, verifier_mode="risk").review(sample_record(), risk("low"))
        trace = decision.trace_dicts()

        self.assertEqual(verifier.calls, [])
        self.assertIsNone(decision.verifier)
        self.assertEqual(trace[-1]["agent"], "verifier")
        self.assertEqual(trace[-1]["action"], "skipped")
        self.assertEqual(trace[-1]["metadata"]["risk_level"], "low")

    def test_verifier_failure_records_reject_trace_and_blocks(self) -> None:
        verifier = FakeVerifierAgent(
            VerifierReview(
                enabled=True,
                verdict="fail",
                reason="Counter-evidence points to B.",
                counter_evidence_lines=[12],
                risk_signal_codes=["fallback_annotation_metadata"],
                confidence="high",
            )
        )

        decision = Coordinator(verifier, verifier_mode="all").review(
            sample_record(),
            risk("low"),
        )
        trace = decision.trace_dicts()

        self.assertTrue(decision.blocks_submission)
        self.assertEqual(decision.verifier["verdict"], "fail")
        self.assertEqual(decision.arbiter["decision"], "reject_labeler")
        self.assertTrue(decision.arbiter["blocks_submission"])
        self.assertIn(("verifier", "rejected"), [(event["agent"], event["action"]) for event in trace])
        self.assertIn(("arbiter", "called"), [(event["agent"], event["action"]) for event in trace])
        self.assertIn(("arbiter", "rejected"), [(event["agent"], event["action"]) for event in trace])
        arbiter_events = [event for event in trace if event["agent"] == "arbiter" and event["action"] == "rejected"]
        self.assertEqual(arbiter_events[0]["result"]["verdict"], "reject")
        self.assertEqual(arbiter_events[0]["result"]["counter_evidence_lines"], [12])

    def test_verifier_uncertain_records_arbiter_reason_without_blocking(self) -> None:
        verifier = FakeVerifierAgent(
            VerifierReview(
                enabled=True,
                verdict="uncertain",
                reason="Evidence is too weak.",
                counter_evidence_lines=[],
                risk_signal_codes=["short_dialogue"],
                confidence="low",
            )
        )

        decision = Coordinator(verifier, verifier_mode="all").review(sample_record(), risk("low"))
        trace = decision.trace_dicts()

        self.assertFalse(decision.blocks_submission)
        self.assertEqual(decision.verifier["verdict"], "uncertain")
        self.assertEqual(decision.arbiter["decision"], "needs_more_evidence")
        self.assertIn("could not fully confirm", decision.arbiter["reason"])
        self.assertIn(("arbiter", "uncertain"), [(event["agent"], event["action"]) for event in trace])

    def test_records_existing_identity_tool_result_in_trace(self) -> None:
        record = sample_record(
            tool_summary={
                "resolve_identity": [
                    {
                        "verdict": "resolved",
                        "recommended_speaker": "Stable Name",
                        "reason": "Later evidence names the speaker.",
                    }
                ]
            }
        )

        decision = Coordinator(verifier_mode="off").review(record, risk("low"))
        trace = decision.trace_dicts()

        resolver_events = [event for event in trace if event["agent"] == "identity_resolver"]
        self.assertEqual(len(resolver_events), 1)
        self.assertEqual(resolver_events[0]["action"], "observed")
        self.assertEqual(resolver_events[0]["result"]["verdict"], "resolved")
        self.assertEqual(resolver_events[0]["result"]["recommended_speaker"], "Stable Name")


if __name__ == "__main__":
    unittest.main()
