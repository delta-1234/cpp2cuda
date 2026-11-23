"""Local protocol tests; --gpu also exercises the full agent loop on a CUDA node."""
import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from api_agent import ROOT, snapshot
from model_api import complete, configure, read_stream, settings
from unittest.mock import patch

DUMMY = "fixture-token-for-loopback-only"


@contextlib.contextmanager
def provider(reply):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.headers.get("Authorization") != "Bearer "+DUMMY or payload["model"] != "fixture":
                self.send_error(401)
                return
            if not self.path.startswith("/v1/"):
                self.send_error(404)
                return
            if self.path.endswith("/responses"):
                text = reply(payload["instructions"],payload["input"])
                data = {"output":[{"type":"message","content":[{"type":"output_text","text":text}]}]}
            else:
                assert self.path == "/v1/chat/completions"
                messages = payload["messages"]
                text = reply(messages[0]["content"],messages[1:])
                data = {"choices":[{"message":{"role":"assistant","content":text}}]}
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def scripted_agent(system,messages):
    if "Reply with OK" in system:
        return "OK"
    initial = json.loads(messages[0]["content"])
    assert "driver" in initial["hardware"] and "stream/scan.cu" in initial["editable"]
    turn = sum(m["role"]=="assistant" for m in messages)
    marker = "#error OFFLINE_AGENT_REPAIR_TEST\n"
    line = "__device__ Changes add("
    actions = [
        {"action":"read","path":"../.ssh/id_rsa"},
        {"action":"read","path":"stream/scan.cu"},
        {"action":"edit","path":"stream/scan.cu","old":line,"new":marker+line},
        {"action":"verify"},
        {"action":"edit","path":"stream/scan.cu","old":marker,"new":"// Verified model API test change.\n"},
        {"action":"verify"},
        {"action":"benchmark","orders":[2600],"repeat":1},
        {"action":"finish","summary":"模拟端点已收到真实设备的构建诊断、验证与计时。"},
    ]
    if turn == 1: assert "inventory" in messages[-1]["content"]
    if turn == 4: assert "OFFLINE_AGENT_REPAIR_TEST" in messages[-1]["content"] and "exited" in messages[-1]["content"]
    if turn == 6: assert json.loads(messages[-1]["content"])["cases"]["sample_ticks.csv"]==912
    if turn == 7: assert "warm/2600/simt" in json.loads(messages[-1]["content"])
    return json.dumps(actions[turn])


def config_test():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/".env"
        path.write_text("export OPENAI_MODEL='first-model'\nOPENAI_BASE_URL=http://localhost:9000/v1\n"
                        "OPENAI_API_KEY='fixture # value'\nUNRELATED=preserved\n")
        env = {"FACTOR_API_CONFIG":str(path)}
        with patch.dict(os.environ,env,clear=True):
            assert settings()["api_key"] == "fixture # value"
            with patch.dict(os.environ,{"OPENAI_MODEL":"override"}):
                assert settings()["model"] == "override"
            with patch.dict(os.environ,{"OPENAI_BASE_URL":"http://localhost:9001/v1"}):
                assert "api_key" not in settings()
            with patch("builtins.input",side_effect=["","next-model",""]), patch("getpass.getpass",return_value=""):
                configure()
            assert settings()["model"] == "next-model" and settings()["api_key"] == "fixture # value"
            assert "UNRELATED=preserved" in path.read_text() and path.stat().st_mode & 0o777 == 0o600


def stream_test():
    chat = b'data: {"choices":[{"delta":{"content":"C++"}}]}\n\ndata: [DONE]\n'
    assert read_stream(io.BytesIO(chat), True) == "C++"
    responses = (b'data: {"type":"response.output_text.delta","delta":"CUDA"}\n'
                 b'data: {"type":"response.completed"}\n')
    assert read_stream(io.BytesIO(responses), False) == "CUDA"
    try:
        read_stream(io.BytesIO(chat.split(b"data: [DONE]")[0]), True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("truncated stream accepted")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",action="store_true")
    args = p.parse_args()
    config_test()
    stream_test()
    with provider(scripted_agent) as url:
        cfg = dict(base_url=url,api_key=DUMMY,model="fixture")
        for protocol in ("chat","responses"):
            cfg["protocol"] = protocol
            assert complete(cfg,"Reply with OK only.",[{"role":"user","content":"ping"}])=="OK"
        print("Both HTTP API protocols passed.",flush=True)
        if not args.gpu:
            return
        temp = Path(tempfile.mkdtemp(prefix="api-loop-test-",dir=ROOT/"runs"))
        repo = temp/"repo"
        snapshot(ROOT,repo)
        for name in ("api_agent.py","agent_jail.sh","model_api.py"):
            shutil.copyfile(ROOT/"workflow"/name,repo/"workflow"/name)
        config = temp/"fixture.json"
        config.write_text(json.dumps(cfg))
        config.chmod(0o600)
        env = {k:v for k,v in os.environ.items() if k not in
               ("OPENAI_API_KEY","OPENAI_BASE_URL","OPENAI_MODEL","FACTOR_API_PROTOCOL")}
        env["FACTOR_API_CONFIG"] = str(config)
        env.setdefault("WITH_TRITON","0")
        cmd = ["bash","workflow/cuda.sh","--agent","exercise a repair on this disposable test copy"]
        subprocess.run(cmd,cwd=repo,env=env,check=True)
        report_path, = list((repo/"runs").glob("api-agent-*/report.json"))
        r = json.loads(report_path.read_text())
        assert r["status"]=="verified" and r["applied"]
        assert DUMMY not in report_path.read_text()
        assert r["actions"][0]["error"] and r["actions"][3]["error"]
        assert len(r["results"]["median_ms"])==(10 if env["WITH_TRITON"]=="1" else 8)
        assert "// Verified model API test change." in (repo/"stream/scan.cu").read_text()
        assert "// Verified model API test change." not in (ROOT/"stream/scan.cu").read_text()
        print("Full isolated API/GPU loop verified:", report_path)


if __name__=="__main__":
    main()
