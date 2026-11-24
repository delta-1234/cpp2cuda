"""Run a real Pi RPC session with only the tools exposed by the CUDA controller."""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from model_api import endpoint

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.6.0"
COMMON = "probe catalog search page read experiment diagnostics log finish".split()
INSPECT = COMMON + ["select_toolchain", "blocked"]
EDIT = "write edit remove build test_case".split()
TOOLS = {
    "inspect": INSPECT,
    "port": INSPECT + EDIT + "case baseline verify benchmark restore".split(),
    "simt": COMMON + EDIT + ["trial", "reset_best"],
    "tile": COMMON + EDIT + ["trial", "reset_best"],
}


class PiUnavailable(RuntimeError):
    pass


def runtime():
    prepared = ROOT / "runs" / "deps" / f"pi-{VERSION}" / "pi" / "pi"
    binary = os.environ.get("FACTORPORT_PI_BIN") or (prepared if prepared.is_file() else shutil.which("pi"))
    if not binary:
        raise PiUnavailable("Pi is missing. Prepare the offline bundle or set FACTORPORT_PI_BIN.")
    path = Path(binary).expanduser().resolve()
    try:
        with tempfile.TemporaryDirectory() as home:
            env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": home, "PI_OFFLINE": "1"}
            check = subprocess.run([path, "--version"], cwd=home, env=env, capture_output=True,
                                   text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PiUnavailable(f"Cannot start Pi at {path}: {exc}") from exc
    version = check.stdout.strip()
    if not version:
        raise PiUnavailable(f"Pi at {path} returned no version")
    return path, version


def provider(cfg):
    chat = cfg.get("protocol", "chat") == "chat"
    suffix = "/chat/completions" if chat else "/responses"
    return {"providers": {"factorport": {
        "baseUrl": endpoint(cfg).removesuffix(suffix),
        "api": "openai-completions" if chat else "openai-responses",
        "apiKey": "$FACTORPORT_PI_KEY",
        "models": [{"id": cfg["model"]}],
    }}}


async def session(binary, work, cfg, names, system, initial, turns, timeout, dispatch, done, scrub):
    agent_dir = work / "agent"
    agent_dir.mkdir(mode=0o700)
    (agent_dir / "models.json").write_text(json.dumps(provider(cfg)))
    (agent_dir / "settings.json").write_text(json.dumps({
        "retry": {"enabled": False}, "compaction": {"enabled": False}, "httpIdleTimeoutMs": timeout * 1000,
    }))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
                                     "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy") if k in os.environ}
    env.update(HOME=str(work), PI_CODING_AGENT_DIR=str(agent_dir), PI_OFFLINE="1",
               FACTORPORT_PI_KEY=cfg.get("api_key") or "unused", FACTORPORT_PI_TOOLS=",".join(names),
               FACTORPORT_PI_TURNS=str(turns))
    cli = [binary, "--mode", "rpc", "--offline", "--no-session", "--no-builtin-tools", "--tools", "fp_*",
           "--no-extensions", "-e", ROOT/"workflow/pi_tools.ts", "--no-mcp", "--no-skills",
           "--no-prompt-templates", "--no-themes", "--no-context-files", "--provider", "factorport",
           "--model", cfg["model"], "--thinking", "off", "--system-prompt", system]
    with tempfile.TemporaryFile() as errors:
        proc = await asyncio.create_subprocess_exec(*cli, cwd=work, env=env, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=errors, limit=32*1024*1024)

        async def send(message):
            proc.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
            await proc.stdin.drain()

        clock = asyncio.get_running_loop().time
        deadline = clock() + min(timeout, 30)
        expected = sorted("fp_" + n for n in names)
        ready = False
        step = count = 0
        failed_action = False
        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), max(0, deadline - clock()))
                if not line:
                    errors.seek(0, 2)
                    errors.seek(max(0, errors.tell()-2000))
                    raise RuntimeError("Pi exited before completion: " + scrub(errors.read().decode(errors="replace")))
                e = json.loads(line)
                kind = e.get("type")
                if kind == "extension_ui_request" and e.get("method") == "notify":
                    msg = e.get("message", "")
                    if msg.startswith("factorport.ready ") and not ready:
                        if sorted(json.loads(msg.split(" ", 1)[1])) != expected:
                            raise RuntimeError("Pi active tools differ from the role allowlist")
                        await send({"id": "identity", "type": "get_state"})
                elif kind == "response":
                    if not e.get("success"):
                        raise RuntimeError("Pi RPC: " + scrub(str(e.get("error", "request failed"))))
                    if e.get("id") == "identity":
                        model = e["data"].get("model") or {}
                        if (model.get("provider"), model.get("id")) != ("factorport", cfg["model"]):
                            raise RuntimeError("Pi did not select the configured provider and model")
                        ready = True
                        task = {**initial, "model_turn_budget": turns}
                        await send({"type": "prompt", "message": json.dumps(task, ensure_ascii=False)})
                        deadline = clock() + timeout
                elif kind == "turn_start":
                    if not ready:
                        raise RuntimeError("Unexpected Pi turn before startup checks")
                    step += 1
                    if step > turns:
                        return
                    count, failed_action = 0, False
                    deadline = clock() + timeout
                elif kind == "extension_ui_request" and e.get("method") in ("input", "select", "confirm", "editor"):
                    if not ready or not step or (e["method"], e.get("title")) != ("input", "factorport.call"):
                        raise RuntimeError("Unexpected Pi extension request")
                    call = json.loads(e["placeholder"])
                    name, params = call["name"], call["params"]
                    if name not in names or not isinstance(params, dict) or "tool" in params:
                        raise RuntimeError("Unrecognized Pi tool request")
                    count += 1
                    if count > 6 or failed_action:
                        result = {"error": "Start a new turn and reassess the preceding tool results."}
                    else:
                        result = dispatch(step, [{**params, "tool": name}])[0]
                        failed_action = "error" in result
                        if done():
                            return
                    feedback = {**result, "model_turns_remaining_after_this_turn": turns - step}
                    value = scrub(json.dumps(feedback, ensure_ascii=False))
                    await send({"type": "extension_ui_response", "id": e["id"], "value": value})
                    deadline = clock() + timeout
                elif kind == "message_end":
                    msg = e.get("message", {})
                    if msg.get("role") == "assistant" and msg.get("stopReason") in ("error", "aborted"):
                        raise RuntimeError("Pi model request failed: " + scrub(str(msg.get("errorMessage", msg["stopReason"])))[:2000])
                elif kind == "turn_end" and step >= turns:
                    return
                elif kind == "agent_settled" and ready:
                    await send({"type": "prompt", "message": "Continue with a native fp_ tool call based on the evidence."})
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Pi timed out waiting for {'model or tools' if ready else 'the tool extension handshake'}") from exc
        finally:
            proc.stdin.close()
            if proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), 3)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()


def pi_dialog(binary, cfg, role, system, initial, turns, timeout, dispatch, done, scrub):
    scope = initial["mode"] if role == "port" else role
    system += "\nUse the native fp_ tools for this role. Perform up to six calls per turn. Use fp_finish when the evidence is complete."
    with tempfile.TemporaryDirectory(prefix="factorport-pi-") as tmp:
        asyncio.run(session(binary, Path(tmp), cfg, TOOLS[scope], system, initial, turns, timeout, dispatch, done, scrub))
