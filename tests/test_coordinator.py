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


class FakeIdentityAgent:
    def __init__(self, review: dict):
        self.payload = review
        self.calls: list[AnnotationRecord] = []

    def review_identity(self, record: AnnotationRecord) -> dict:
        self.calls.append(record)
        return self.payload

    def review(self, record: AnnotationRecord) -> dict:
        return self.review_identity(record)


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

    def test_short_dialogue_without_rejected_candidates_blocks_high_risk_verifier_pass(self) -> None:
        verifier = FakeVerifierAgent(
            VerifierReview(
                enabled=True,
                verdict="pass",
                reason="Turn order seems plausible.",
                counter_evidence_lines=[],
                risk_signal_codes=["no_rejected_candidates_for_short_dialogue"],
                confidence="high",
            )
        )

        decision = Coordinator(verifier, verifier_mode="risk").review(
            sample_record(rejected_candidates=[]),
            RiskAssessment(
                level="high",
                signals=[
                    RiskSignal(
                        code="no_rejected_candidates_for_short_dialogue",
                        message="Short dialogue lacks rejected candidates.",
                        level="medium",
                    )
                ],
            ),
        )
        trace = decision.trace_dicts()

        self.assertTrue(decision.blocks_submission)
        self.assertEqual(decision.verifier["verdict"], "pass")
        self.assertEqual(decision.arbiter["decision"], "needs_more_evidence")
        self.assertTrue(decision.arbiter["blocks_submission"])
        self.assertIn("short dialogue has no rejected speaker candidates", decision.arbiter["reason"])
        self.assertIn(("arbiter", "called"), [(event["agent"], event["action"]) for event in trace])

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

    def test_coordinator_calls_identity_agent_for_temporary_speaker_and_blocks_conflict(self) -> None:
        identity = FakeIdentityAgent(
            {
                "enabled": True,
                "triggered": True,
                "speaker": "\u5c11\u5973",
                "verdict": "resolved",
                "recommended_speaker": "\u963f\u6d1b",
                "evidence_lines": [3],
                "reason": "bounded identity evidence",
                "confidence": "high",
                "candidate_ranges": [{"start_line": 2, "end_line": 4, "matched_line": 3}],
                "locator_attempts": [],
                "resolver": {"verdict": "resolved", "recommended_speaker": "\u963f\u6d1b"},
            }
        )

        decision = Coordinator(identity_agent=identity, verifier_mode="off").review(
            sample_record(speaker="\u5c11\u5973"),
            risk("low"),
        )
        trace = decision.trace_dicts()

        self.assertEqual(len(identity.calls), 1)
        self.assertEqual(decision.identity["recommended_speaker"], "\u963f\u6d1b")
        self.assertTrue(decision.blocks_submission)
        self.assertEqual(decision.arbiter["decision"], "use_resolved_identity")
        self.assertEqual(decision.arbiter["recommended_speaker"], "\u963f\u6d1b")
        self.assertEqual(decision.arbiter["block_reason_code"], "identity_resolved_conflict")
        self.assertIn(("identity_locator", "called"), [(event["agent"], event["action"]) for event in trace])
        self.assertIn(("identity_resolver", "resolved"), [(event["agent"], event["action"]) for event in trace])

    def test_coordinator_does_not_trigger_identity_for_known_false_positive_terms(self) -> None:
        for speaker in ("\u7537\u5b50", "\u54b1", "\u7537\u5b69", "\u620f\u66f2\u6545\u4e8b\u91cc\u7684\u7537\u5b69"):
            with self.subTest(speaker=speaker):
                identity = FakeIdentityAgent({"verdict": "resolved", "recommended_speaker": "\u9519\u8bef"})

                decision = Coordinator(identity_agent=identity, verifier_mode="off").review(
                    sample_record(speaker=speaker),
                    risk("low"),
                )

                self.assertEqual(identity.calls, [])
                self.assertIsNone(decision.identity)
                self.assertFalse(decision.blocks_submission)


if __name__ == "__main__":
    unittest.main()
