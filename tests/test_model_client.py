from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from unittest.mock import patch

from dialoop.model_client import (
    ChatMessage,
    ModelConfig,
    ModelHTTPError,
    ModelResponseError,
    ModelTimeoutError,
    OpenAICompatibleClient,
)
from dialoop.protocol import local_tool_specs


class StubHandler(BaseHTTPRequestHandler):
    response_status: ClassVar[int] = 200
    response_body: ClassVar[dict[str, Any]] = {}
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw) if raw else {}
        self.__class__.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "payload": payload,
            }
        )

        body = json.dumps(self.__class__.response_body).encode("utf-8")
        self.send_response(self.__class__.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


class StubServer:
    def __init__(self, response_body: dict[str, Any], response_status: int = 200):
        StubHandler.response_body = response_body
        StubHandler.response_status = response_status
        StubHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "StubServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return StubHandler.requests


def chat_response(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


class FakeURLResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def __enter__(self) -> "FakeURLResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ModelClientTest(unittest.TestCase):
    def test_default_timeout_is_long_enough_for_local_models(self) -> None:
        self.assertEqual(ModelConfig().timeout, 60.0)

    def test_chat_posts_openai_compatible_payload_and_parses_content(self) -> None:
        with StubServer(chat_response({"role": "assistant", "content": "OK"})) as server:
            client = OpenAICompatibleClient(ModelConfig(base_url=server.base_url, api_key="secret", model="test-model"))

            result = client.chat([ChatMessage(role="user", content="hello")])

            self.assertEqual(result.content, "OK")
            self.assertEqual(server.requests[0]["path"], "/v1/chat/completions")
            self.assertEqual(server.requests[0]["payload"]["model"], "test-model")
            self.assertEqual(server.requests[0]["headers"]["Authorization"], "Bearer secret")

    def test_chat_sends_tool_schema_and_parses_tool_calls(self) -> None:
        response = chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_novel",
                            "arguments": '{"start_line": 1, "end_line": 3}',
                        },
                    }
                ],
            }
        )
        with StubServer(response) as server:
            client = OpenAICompatibleClient(ModelConfig(base_url=server.base_url, model="test-model"))

            result = client.chat([ChatMessage(role="user", content="hello")], tools=local_tool_specs())

            self.assertEqual(server.requests[0]["payload"]["tools"][0]["type"], "function")
            self.assertEqual(result.tool_calls[0].name, "read_novel")
            self.assertEqual(result.tool_calls[0].arguments, {"start_line": 1, "end_line": 3})

    def test_check_connection_reports_success(self) -> None:
        with StubServer(chat_response({"role": "assistant", "content": "OK"})) as server:
            client = OpenAICompatibleClient(ModelConfig(base_url=server.base_url, model="test-model"))

            status = client.check_connection()

            self.assertTrue(status.ok)
            self.assertEqual(status.message, "OK")

    def test_http_error_is_reported(self) -> None:
        with StubServer({"error": {"message": "bad key"}}, response_status=401) as server:
            client = OpenAICompatibleClient(ModelConfig(base_url=server.base_url, model="test-model"))

            with self.assertRaises(ModelHTTPError):
                client.chat([ChatMessage(role="user", content="hello")])

    def test_missing_choices_is_rejected(self) -> None:
        with StubServer({"choices": []}) as server:
            client = OpenAICompatibleClient(ModelConfig(base_url=server.base_url, model="test-model"))

            with self.assertRaises(ModelResponseError):
                client.chat([ChatMessage(role="user", content="hello")])

    def test_chat_retries_timed_out_request(self) -> None:
        response = FakeURLResponse(chat_response({"role": "assistant", "content": "OK"}))
        with patch("dialoop.model_client.urllib.request.urlopen", side_effect=[TimeoutError(), response]) as urlopen:
            client = OpenAICompatibleClient(
                ModelConfig(
                    base_url="http://127.0.0.1:9999/v1",
                    model="test-model",
                    timeout=0.01,
                    retries=1,
                    retry_delay=0,
                )
            )

            result = client.chat([ChatMessage(role="user", content="hello")])

        self.assertEqual(result.content, "OK")
        self.assertEqual(urlopen.call_count, 2)

    def test_chat_raises_after_timeout_retries_are_exhausted(self) -> None:
        with patch("dialoop.model_client.urllib.request.urlopen", side_effect=TimeoutError()) as urlopen:
            client = OpenAICompatibleClient(
                ModelConfig(
                    base_url="http://127.0.0.1:9999/v1",
                    model="test-model",
                    timeout=0.01,
                    retries=1,
                    retry_delay=0,
                )
            )

            with self.assertRaises(ModelTimeoutError) as raised:
                client.chat([ChatMessage(role="user", content="hello")])

        self.assertIn("after 2 attempt", str(raised.exception))
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
