from __future__ import annotations

import unittest

from dialoop.protocol import ProtocolError, local_tool_specs, openai_tools_schema, parse_json_action


class ProtocolTest(unittest.TestCase):
    def test_local_tool_specs_convert_to_openai_tools(self) -> None:
        specs = local_tool_specs()
        names = {spec.name for spec in specs}

        self.assertEqual(
            names,
            {
                "get_next_dialogue",
                "read_novel",
                "search_novel",
                "locate_identity",
                "resolve_identity",
                "record_character",
                "normalize_speaker",
                "arbitrate_identity",
                "submit_labels",
            },
        )
        schema = openai_tools_schema()
        self.assertEqual(schema[0]["type"], "function")
        self.assertIn("parameters", schema[0]["function"])
        locate = next(spec for spec in specs if spec.name == "locate_identity")
        normalize = next(spec for spec in specs if spec.name == "normalize_speaker")
        submit = next(spec for spec in specs if spec.name == "submit_labels")
        self.assertIn("Required before submit_labels", locate.description)
        self.assertIn("Required before submit_labels", normalize.description)
        self.assertIn("locate_identity/resolve_identity", submit.description)
        self.assertIn("first-person pronouns", locate.description)
        self.assertIn("story, play", locate.description)

    def test_submit_labels_schema_can_pin_active_batch_count(self) -> None:
        specs = local_tool_specs(submit_label_count=2)
        submit = next(spec for spec in specs if spec.name == "submit_labels")
        speakers = submit.parameters["properties"]["speakers"]

        self.assertEqual(speakers["minItems"], 2)
        self.assertEqual(speakers["maxItems"], 2)

    def test_parse_json_action_from_plain_json(self) -> None:
        action = parse_json_action('{"action":"read_novel","args":{"start_line":1,"end_line":3}}')

        self.assertEqual(action.action, "read_novel")
        self.assertEqual(action.args, {"start_line": 1, "end_line": 3})

    def test_parse_json_action_accepts_identity_tools(self) -> None:
        action = parse_json_action('{"action":"locate_identity","args":{"speaker":"girl"}}')

        self.assertEqual(action.action, "locate_identity")
        self.assertEqual(action.args, {"speaker": "girl"})

    def test_parse_json_action_from_fenced_json(self) -> None:
        action = parse_json_action(
            """```json
{"action":"submit_labels","args":{"speakers":["罗伦斯"]}}
```"""
        )

        self.assertEqual(action.action, "submit_labels")
        self.assertEqual(action.args, {"speakers": ["罗伦斯"]})

    def test_parse_json_action_rejects_unknown_action(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_json_action('{"action":"unknown","args":{}}')


if __name__ == "__main__":
    unittest.main()
