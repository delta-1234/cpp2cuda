"""Configuration and HTTPS client for the CUDA workflow's model endpoint."""
import getpass
import json
import os
import re
import shlex
import tempfile
from pathlib import Path
from urllib import error, parse, request


def config_path(for_write=False):
    if os.getenv("FACTOR_API_CONFIG"):
        return Path(os.environ["FACTOR_API_CONFIG"]).expanduser()
    project = Path(__file__).resolve().parents[1]/".env"
    if for_write:
        return project
    for path in (project, Path.home()/".config/order-fill-rate-cuda/api.json", Path.home()/".env"):
        if path.is_file():
            return path
    return project


FIELDS = {"OPENAI_BASE_URL":"base_url", "OPENAI_MODEL":"model", "OPENAI_API_KEY":"api_key",
          "FACTOR_API_PROTOCOL":"protocol"}


def read_dotenv(path):
    cfg = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)$", line)
        if not match or match[1] not in FIELDS:
            continue
        value = match[2].strip()
        if value.startswith(('"', "'")):
            try:
                parts = shlex.split(value, comments=True)
                if len(parts)!=1:
                    raise ValueError()
                value = parts[0]
            except ValueError:
                raise ValueError(f"invalid quoted value for {match[1]} at line {n}") from None
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        cfg[FIELDS[match[1]]] = value
    return cfg


def settings():
    path = config_path()
    if not path.is_file() and os.getenv("FACTOR_API_CONFIG"):
        raise ValueError(f"configuration file not found: {path}")
    cfg = {}
    if path.is_file():
        cfg = json.loads(path.read_text()) if path.suffix.lower()==".json" else read_dotenv(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected a JSON object in {path}")
    if any(not isinstance(cfg.get(k,""),str) for k in ("base_url","model","api_key","protocol")):
        raise ValueError("configuration values must be strings")
    if "OPENAI_BASE_URL" in os.environ and os.environ["OPENAI_BASE_URL"].rstrip("/") != cfg.get("base_url","").rstrip("/"):
        cfg.pop("api_key", None)
    for env, field in FIELDS.items():
        if env in os.environ:
            cfg[field] = os.environ[env]
    return cfg


def endpoint(cfg):
    base = cfg.get("base_url", "").strip().rstrip("/")
    url = parse.urlsplit(base)
    if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("set OPENAI_BASE_URL or run: bash workflow/cuda.sh --configure-api")
    if not cfg.get("model", "").strip():
        raise ValueError("set model in the configuration or OPENAI_MODEL")
    protocol = cfg.get("protocol", "chat")
    suffix = {"chat": "/chat/completions", "responses": "/responses"}.get(protocol)
    if not suffix:
        raise ValueError("protocol must be chat or responses")
    if base.endswith(("/chat/completions", "/responses")) and not base.endswith(suffix):
        raise ValueError("endpoint path and protocol do not match")
    return base if base.endswith(suffix) else base + suffix


def configure():
    path = config_path(for_write=True)
    is_json = path.suffix.lower() == ".json"
    old = {}
    if path.is_file():
        old = json.loads(path.read_text()) if is_json else read_dotenv(path)
    cfg = {}
    for field, label, env, fallback in (
            ("base_url", "API Base URL（含版本路径，如 /v1）", "OPENAI_BASE_URL", ""),
            ("model", "模型 ID", "OPENAI_MODEL", ""),
            ("protocol", "协议 chat 或 responses", "FACTOR_API_PROTOCOL", "chat")):
        previous = old.get(field) or os.getenv(env) or fallback
        cfg[field] = input(f"{label} [{previous}] ").strip() or previous
    endpoint(cfg)
    previous_key = old.get("api_key", "")
    # Never copy an environment key into a file implicitly or reuse a key for a different host.
    if cfg["base_url"] != old.get("base_url"):
        previous_key = ""
    prompt = "API Key（回车保留已保存的密钥）" if previous_key else "API Key（可留空，由环境变量提供或使用免密服务）"
    cfg["api_key"] = getpass.getpass(prompt + " ") or previous_key
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        try:
            if is_json:
                json.dump({**old, **cfg}, tmp, ensure_ascii=False, indent=2)
                tmp.write("\n")
            else:
                # Keep comments and application-specific variables outside this workflow.
                for line in path.read_text().splitlines() if path.exists() else ():
                    m = re.match(r"^\s*(?:export\s+)?(\w+)\s*=", line)
                    if not m or m[1] not in FIELDS:
                        tmp.write(line + "\n")
                for env, field in FIELDS.items():
                    tmp.write(f"{env}={shlex.quote(cfg[field])}\n")
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    print(f"已保存 {path}（0600）")


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def complete(cfg, system, messages):
    url = endpoint(cfg)
    chat = cfg.get("protocol", "chat") == "chat"
    body = {"model": cfg["model"], "stream": False}
    if chat:
        body["messages"] = [{"role": "system", "content": system}, *messages]
    else:
        body.update(instructions=system, input=messages, store=False)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = cfg.get("api_key")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with request.build_opener(NoRedirect()).open(req, timeout=180) as resp:
            data = json.load(resp)
    except error.HTTPError as exc:
        # Gateways sometimes echo request credentials in their error messages.
        raise RuntimeError(f"model API returned HTTP {exc.code}; check endpoint, credentials, model and quota") from None
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"model API connection failed ({type(exc).__name__})") from None
    try:
        if chat:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part["text"] for part in content if part.get("type") == "text")
        else:
            content = "".join(p["text"] for item in data["output"] if item.get("type") == "message"
                              for p in item.get("content", []) if p.get("type") == "output_text")
        if isinstance(content, str) and content.strip():
            return content.strip()
    except (KeyError, IndexError, TypeError):
        pass
    raise ValueError("model API returned no text in the configured protocol")
