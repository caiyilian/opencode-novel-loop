from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
