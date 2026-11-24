"""Exercise native tool calling through the pinned Pi binary and a loopback model."""
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from pi_backend import PiUnavailable, TOOLS, pi_dialog, provider, runtime


def response_events(calls):
    response = {"id": "resp_local", "object": "response", "model": "local-model", "status": "in_progress", "output": []}
    yield {"type": "response.created", "response": response}
    for i, call in enumerate(calls):
        item = {"type": "function_call", "id": "fc_" + call["id"], "call_id": call["id"],
                "name": call["function"]["name"], "arguments": "", "status": "in_progress"}
        yield {"type": "response.output_item.added", "output_index": i, "item": item}
        args = call["function"]["arguments"]
        yield {"type": "response.function_call_arguments.delta", "output_index": i, "item_id": item["id"], "delta": args}
        item.update(arguments=args, status="completed")
        yield {"type": "response.output_item.done", "output_index": i, "item": item}
        response["output"].append(item)
    response["status"] = "completed"
    yield {"type": "response.completed", "response": response}


@contextmanager
def model_server(replies):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            if len(requests) > len(replies):
                self.send_error(400, "unexpected extra turn")
                return
            calls = replies[len(requests) - 1]
            delta = {"role": "assistant"}
            if calls:
                delta["tool_calls"] = [
                    {"index": i, "id": f"call_{len(requests)}_{i}", "type": "function",
                     "function": {"name": "fp_" + name, "arguments": json.dumps(args)}}
                    for i, (name, args) in enumerate(calls)]
            else:
                delta["content"] = "The device looks fine."
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if self.path.endswith("/responses"):
                events = response_events(delta.get("tool_calls", []))
            else:
                chunks = [{"delta": delta, "finish_reason": None},
                          {"delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]
                events = [{"id": "local", "object": "chat.completion.chunk", "model": "local-model",
                           "choices": [{"index": 0, **chunk}]} for chunk in chunks]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")


    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield {"base_url": f"http://127.0.0.1:{server.server_port}/v1",
                   "model": "local-model", "api_key": "test-secret"}, requests
        finally:
            server.shutdown()
            thread.join()


class PiConfigTests(unittest.TestCase):
    def test_endpoint_and_env_reference(self):
        for protocol, suffix, api in (("chat", "/chat/completions", "openai-completions"),
                                      ("responses", "/responses", "openai-responses")):
            cfg = dict(protocol=protocol, model="test", api_key="secret", base_url="http://localhost/v1" + suffix)
            parsed = provider(cfg)["providers"]["factorport"]
            self.assertEqual((parsed["baseUrl"], parsed["api"]), ("http://localhost/v1", api))
            self.assertEqual(parsed["apiKey"], "$FACTORPORT_PI_KEY")
            self.assertNotIn("secret", json.dumps(parsed))

    def test_explicit_missing_binary(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"FACTORPORT_PI_BIN": str(Path(tmp)/"missing")}):
            with self.assertRaises(PiUnavailable):
                runtime()


class PiRpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.binary, _ = runtime()
        except PiUnavailable as exc:
            raise unittest.SkipTest(str(exc))

    def test_repair_and_role_scopes(self):
        for role in ("inspect", "port", "simt", "tile"):
            with self.subTest(role=role):
                replies = [[("finish", {"summary": "too early"}), ("probe", {"topic": "system"})],
                           [("probe", {"topic": "gpu"})], [("finish", {"summary": "observed"})]]
                calls = []
                def dispatch(turn, actions):
                    calls.append((turn, actions[0]["tool"]))
                    return [{"error": "probe before finishing"}] if turn == 1 else [{"result": "observed"}]
                with model_server(replies) as (cfg, requests):
                    pi_dialog(self.binary, cfg, "port" if role == "inspect" else role, "Test.", {"mode": role},
                              4, 15, dispatch, lambda: len(calls) == 3, lambda text: text)
                self.assertEqual(calls, [(1, "finish"), (2, "probe"), (3, "finish")])
                for path, auth, req in requests:
                    self.assertEqual((path, auth, req["model"]), ("/v1/chat/completions", "Bearer test-secret", "local-model"))
                    self.assertEqual({t["function"]["name"] for t in req["tools"]}, {"fp_"+name for name in TOOLS[role]})
                self.assertIn("probe before finishing", json.dumps(requests[1][2]["messages"]))
                history = requests[2][2]["messages"]
                prompt = next(m["content"][0]["text"] for m in history if m["role"] == "user")
                self.assertEqual(json.loads(prompt)["model_turn_budget"], 4)
                self.assertEqual(json.loads(history[-1]["content"]),
                                 {"result": "observed", "model_turns_remaining_after_this_turn": 2})

    def test_provider_error_stops_without_dispatch(self):
        with model_server([]) as (cfg, requests):
            with self.assertRaisesRegex(RuntimeError, "Pi model request failed"):
                pi_dialog(self.binary, cfg, "port", "Test.", {"mode": "inspect"}, 2, 15,
                          lambda *_: self.fail("tool ran after API error"), lambda: False, lambda text: text)
            self.assertEqual(len(requests), 1)

    def test_responses_function_result(self):
        replies = [[("probe", {"topic": "gpu"})], [("finish", {"summary": "done"})]]
        with model_server(replies) as (cfg, requests):
            cfg.update(protocol="responses", base_url=cfg["base_url"]+"/responses")
            calls = []
            def dispatch(step, actions):
                calls.extend(a["tool"] for a in actions)
                return [{"result": "observed-device"}]
            pi_dialog(self.binary, cfg, "port", "Test.", {"mode": "inspect"}, 3, 15, dispatch,
                      lambda: "finish" in calls, lambda text: text)
            self.assertEqual(calls, ["probe", "finish"])
            self.assertEqual([r[0] for r in requests], ["/v1/responses"] * 2)
            self.assertIn("observed-device", json.dumps(requests[1][2]["input"]))

    def test_text_cannot_finish_and_turn_budget_stops_work(self):
        with model_server([[], [("probe", {"topic": "gpu"})]]) as (cfg, requests):
            calls = []
            def dispatch(turn, actions):
                calls.append(turn)
                return [{"result": "ok"}]
            pi_dialog(self.binary, cfg, "port", "Test.", {"mode": "inspect"}, 2, 15, dispatch,
                      lambda: False, lambda text: text)
            self.assertEqual(calls, [2])
            self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
