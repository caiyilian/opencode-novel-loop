from __future__ import annotations

import unittest

from dialoop.protocol import ProtocolError, local_tool_specs, openai_tools_schema, parse_json_action


class ProtocolTest(unittest.TestCase):
    def test_local_tool_specs_convert_to_openai_tools(self) -> None:
        specs = local_tool_specs()
        names = {spec.name for spec in specs}

        self.assertEqual(
            names,
            {"get_next_dialogue", "read_novel", "search_novel", "submit_labels"},
        )
        schema = openai_tools_schema()
        self.assertEqual(schema[0]["type"], "function")
        self.assertIn("parameters", schema[0]["function"])

    def test_parse_json_action_from_plain_json(self) -> None:
        action = parse_json_action('{"action":"read_novel","args":{"start_line":1,"end_line":3}}')

        self.assertEqual(action.action, "read_novel")
        self.assertEqual(action.args, {"start_line": 1, "end_line": 3})

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
