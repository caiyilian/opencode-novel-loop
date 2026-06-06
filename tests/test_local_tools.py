from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dialoop.identity import extract_stable_names
from dialoop.local_tools import DialogueIndex, DialoopLocalTools, LabelStore, ToolValidationError


SAMPLE_TEXT = "\n".join(
    [
        "第一行没有对话。",
        "罗伦斯说：「你好。」赫萝回答：「你好呀。」",
        "中间叙述。",
        "商人问：「这是最后一件了吧？」",
        "很远的叙述。",
        "又有人说：「下一句。」",
    ]
)


class DialogueIndexTest(unittest.TestCase):
    def test_extracts_dialogues_with_indexes_and_line_numbers(self) -> None:
        index = DialogueIndex.from_text(SAMPLE_TEXT)

        self.assertEqual(index.total, 4)
        self.assertEqual(index.dialogues[0].to_dict(), {"index": 0, "line_number": 2, "text": "你好。"})
        self.assertEqual(index.dialogues[1].to_dict(), {"index": 1, "line_number": 2, "text": "你好呀。"})
        self.assertEqual(index.dialogues[3].line_number, 6)

    def test_next_batch_respects_progress_batch_size_and_line_gap(self) -> None:
        index = DialogueIndex.from_text(SAMPLE_TEXT)

        batch = index.next_batch(labeled_count=0, batch_size=3)
        self.assertEqual([dialogue.text for dialogue in batch], ["你好。", "你好呀。", "这是最后一件了吧？"])

        contiguous = index.next_batch(labeled_count=0, batch_size=3, max_line_gap=0)
        self.assertEqual([dialogue.text for dialogue in contiguous], ["你好。", "你好呀。"])

    def test_read_lines_clips_and_marks_truncation(self) -> None:
        index = DialogueIndex.from_text(SAMPLE_TEXT)

        result = index.read_lines(2, 6, line_limit=2)

        self.assertTrue(result["truncated"])
        self.assertEqual(result["start_line"], 2)
        self.assertEqual(result["end_line"], 3)
        self.assertIn("2: 罗伦斯说", result["text"])
        self.assertIn("3: 中间叙述。", result["text"])

    def test_read_lines_outside_file_returns_empty_text(self) -> None:
        index = DialogueIndex.from_text(SAMPLE_TEXT)

        result = index.read_lines(99, 120)

        self.assertEqual(result["text"], "")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["start_line"], 7)
        self.assertEqual(result["end_line"], 6)

    def test_search_returns_limited_matches(self) -> None:
        index = DialogueIndex.from_text(SAMPLE_TEXT)

        result = index.search("说", limit=1)

        self.assertEqual(result["total_matches"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["matches"]), 1)


class LabelStoreTest(unittest.TestCase):
    def test_counts_non_empty_lines_and_appends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            labels.write_text("罗伦斯\n\n赫萝\n", encoding="utf-8")
            store = LabelStore(labels)

            self.assertEqual(store.count(), 2)
            self.assertEqual(store.append(["商人"]), 1)
            self.assertEqual(store.labels(), ["罗伦斯", "赫萝", "商人"])

    def test_rejects_empty_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LabelStore(Path(directory) / "labels.txt")

            with self.assertRaises(ToolValidationError):
                store.append([""])


class DialoopLocalToolsTest(unittest.TestCase):
    def test_tool_flow_requires_active_batch_and_valid_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=2)

            with self.assertRaises(ToolValidationError):
                tools.submit_labels(["罗伦斯"])

            next_dialogue = tools.get_next_dialogue()
            self.assertFalse(next_dialogue["done"])
            self.assertEqual(len(next_dialogue["dialogues"]), 2)

            with self.assertRaises(ToolValidationError):
                tools.submit_labels(["罗伦斯"])

            result = tools.submit_labels(["罗伦斯", "赫萝"])
            self.assertTrue(result["accepted"])
            self.assertEqual(result["written"], 2)
            self.assertEqual(result["progress"]["labeled"], 2)
            self.assertEqual(LabelStore(labels).labels(), ["罗伦斯", "赫萝"])

    def test_submit_labels_recovers_from_extra_speakers_by_keeping_active_batch_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=1)
            tools.get_next_dialogue()

            result = tools.submit_labels(["罗伦斯", "赫萝", "村民"])

            self.assertTrue(result["accepted"])
            self.assertEqual(result["written"], 1)
            self.assertEqual(result["expected_count"], 1)
            self.assertEqual(result["received_count"], 3)
            self.assertEqual(result["ignored_speakers"], ["赫萝", "村民"])
            self.assertIn("ignored 2 extra", result["warning"])
            self.assertEqual(LabelStore(labels).labels(), ["罗伦斯"])

    def test_tools_read_search_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            labels.write_text("甲\n乙\n丙\n丁\n", encoding="utf-8")
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels))

            self.assertTrue(tools.get_next_dialogue()["done"])
            self.assertIn("2: 罗伦斯说", tools.read_novel(2, 2)["text"])
            self.assertEqual(tools.search_novel("赫萝")["total_matches"], 1)

    def test_get_next_dialogue_includes_neighbor_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            labels.write_text("罗伦斯\n赫萝\n", encoding="utf-8")
            tools = DialoopLocalTools(
                DialogueIndex.from_text(SAMPLE_TEXT),
                LabelStore(labels),
                batch_size=1,
                previous_context_dialogues=2,
                following_context_dialogues=1,
            )

            next_dialogue = tools.get_next_dialogue()

            self.assertEqual(next_dialogue["dialogues"][0]["text"], "这是最后一件了吧？")
            self.assertEqual(
                [(item["text"], item["speaker"]) for item in next_dialogue["previous_dialogues"]],
                [("你好。", "罗伦斯"), ("你好呀。", "赫萝")],
            )
            self.assertEqual([item["text"] for item in next_dialogue["following_dialogues"]], ["下一句。"])

    def test_read_active_context_uses_active_batch_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels), batch_size=2)
            tools.get_next_dialogue()

            context = tools.read_active_context(context_window_lines=1)

            self.assertEqual(context["start_line"], 1)
            self.assertEqual(context["end_line"], 3)
            self.assertIn("2: 罗伦斯说", context["text"])

    def test_rejects_negative_neighbor_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"

            with self.assertRaises(ToolValidationError):
                DialoopLocalTools(
                    DialogueIndex.from_text(SAMPLE_TEXT),
                    LabelStore(labels),
                    previous_context_dialogues=-1,
                )

    def test_get_next_dialogue_respects_max_line_gap(self) -> None:
        text = "\n".join(
            [
                "A: \u300cOne.\u300d",
                "narration",
                "B: \u300cTwo.\u300d",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(
                DialogueIndex.from_text(text),
                LabelStore(labels),
                batch_size=2,
                max_line_gap=0,
            )

            next_dialogue = tools.get_next_dialogue()

            self.assertEqual(len(next_dialogue["dialogues"]), 1)

    def test_identity_lookup_resolves_bounded_later_name(self) -> None:
        text = "\n".join(
            [
                "\u5c11\u5973\u8bf4\uff1a\u300cHelp.\u300d",
                "\u53d9\u8ff0\u3002",
                "\u5979\u540e\u6765\u8bf4\uff1a\u300c\u6211\u53eb\u963f\u6d1b\u3002\u300d",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(
                DialogueIndex.from_text(text),
                LabelStore(labels),
                identity_lookahead_lines=5,
            )
            tools.get_next_dialogue()

            located = tools.locate_identity("\u5c11\u5973")
            candidate = located["candidates"][0]
            resolved = tools.resolve_identity(
                "\u5c11\u5973",
                start_line=candidate["start_line"],
                end_line=candidate["end_line"],
            )

        self.assertEqual(located["search_start_line"], 2)
        self.assertEqual(candidate["matched_line"], 3)
        self.assertEqual(candidate["suggested_names"], ["\u963f\u6d1b"])
        self.assertEqual(resolved["verdict"], "resolved")
        self.assertEqual(resolved["recommended_speaker"], "\u963f\u6d1b")
        self.assertEqual(resolved["evidence_lines"], [3])

    def test_identity_lookup_ignores_repeated_temporary_term_without_name_marker(self) -> None:
        text = "\n".join(
            [
                "\u5c11\u5973\u8bf4\uff1a\u300cHelp.\u300d",
                "\u5c11\u5973\u7f13\u7f13\u5f20\u5f00\u773c\u775b\u3002",
                "\u5c11\u5973\u7ad9\u4e86\u8d77\u6765\u3002",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(
                DialogueIndex.from_text(text),
                LabelStore(labels),
                identity_lookahead_lines=5,
            )
            tools.get_next_dialogue()

            located = tools.locate_identity("\u5c11\u5973")

        self.assertEqual(located["candidates"], [])

    def test_identity_resolver_rejects_unrelated_later_self_introduction(self) -> None:
        text = "\n".join(
            [
                "\u7537\u5b50\u8bf4\uff1a\u300cInteresting.\u300d",
                "\u4e00\u6bb5\u53d9\u8ff0\u3002",
                "\u53c8\u4e00\u6bb5\u53d9\u8ff0\u3002",
                "\u65b0\u7684\u4eba\u9760\u8fd1\u8fc7\u6765\u3002",
                "\u300c\u5c0f\u7684\u540d\u53eb\u6770\u5ec9\u3002\u300d",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(
                DialogueIndex.from_text(text),
                LabelStore(labels),
                identity_lookahead_lines=8,
            )
            tools.get_next_dialogue()

            located = tools.locate_identity("\u7537\u5b50")
            resolved = tools.resolve_identity(
                "\u7537\u5b50",
                start_line=3,
                end_line=5,
            )

        self.assertEqual(located["candidates"], [])
        self.assertEqual(resolved["verdict"], "not_same_person")
        self.assertIsNone(resolved["recommended_speaker"])

    def test_extract_stable_names_filters_places_and_role_prefixes(self) -> None:
        self.assertEqual(extract_stable_names("\u6211\u4f4f\u5728\u4e00\u4e2a\u53eb\u505a\u4f69\u8fde\u4f50\u7684\u57ce\u9547\u3002"), [])
        self.assertEqual(extract_stable_names("\u554a\uff01\u5c0f\u7684\u540d\u53eb\u6770\u5ec9\u3002"), ["\u6770\u5ec9"])
        self.assertEqual(extract_stable_names("\u6211\u662f\u65c5\u884c\u5546\u4eba\u7f57\u4f26\u65af\u3002"), ["\u7f57\u4f26\u65af"])
        self.assertEqual(extract_stable_names("\u6211\u662f\u5546\u4eba\u3002"), [])

    def test_character_library_records_and_suggests_normalized_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels))

            recorded = tools.record_character(
                display_name="\u7f57\u4f26\u65af",
                aliases=["\u65c5\u884c\u5546\u4eba"],
                summary="merchant",
                evidence_lines=[10, 10, 11],
                last_seen_dialogue_index=3,
                last_seen_line_number=11,
                confidence="high",
            )
            normalized = tools.normalize_speaker("\u65c5\u884c\u5546\u4eba")
            next_dialogue = tools.get_next_dialogue()

        self.assertEqual(recorded["record"]["display_name"], "\u7f57\u4f26\u65af")
        self.assertEqual(recorded["record"]["evidence_lines"], [10, 11])
        self.assertTrue(normalized["matched"])
        self.assertEqual(normalized["suggested_display_name"], "\u7f57\u4f26\u65af")
        self.assertEqual(next_dialogue["known_characters"][0]["display_name"], "\u7f57\u4f26\u65af")

    def test_identity_lookup_requires_active_batch_or_dialogue_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels))

            with self.assertRaises(ToolValidationError):
                tools.locate_identity("\u5c11\u5973")

    def test_identity_lookup_round_limit_prevents_unbounded_search(self) -> None:
        text = "\n".join(
            [
                "\u5c11\u5973\u8bf4\uff1a\u300cHelp.\u300d",
                "\u5979\u8bf4\uff1a\u300c\u6211\u53eb\u963f\u6d1b\u3002\u300d",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(
                DialogueIndex.from_text(text),
                LabelStore(labels),
                identity_lookahead_rounds=1,
            )
            tools.get_next_dialogue()

            first = tools.locate_identity("\u5c11\u5973")
            second = tools.locate_identity("\u5c11\u5973")

        self.assertFalse(first["round_limit_reached"])
        self.assertEqual(first["round"], 1)
        self.assertTrue(second["round_limit_reached"])
        self.assertEqual(second["round_limit"], 1)
        self.assertEqual(second["candidates"], [])

    def test_identity_arbiter_prefers_resolved_evidence_backed_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.txt"
            tools = DialoopLocalTools(DialogueIndex.from_text(SAMPLE_TEXT), LabelStore(labels))

            decision = tools.arbitrate_identity(
                labeler_speaker="\u5c11\u5973",
                verifier_verdict="pass",
                resolver_verdict="resolved",
                resolver_speaker="\u963f\u6d1b",
                normalizer_speaker="\u5c11\u5973",
            )

        self.assertEqual(decision["decision"], "use_resolved_identity")
        self.assertEqual(decision["recommended_speaker"], "\u963f\u6d1b")


if __name__ == "__main__":
    unittest.main()
