"""TTS Studio 的 MCP 服务：让 Claude Code / Cursor / Codex 等 Agent 直接调用本机配音。

Agent 用 stdio 方式启动它（配置见 README「接入 Agent（MCP）」）。它本身不加载模型，
只是转发给 TTS Studio 的开放接口（/v1/...）：
- TTS Studio 没开着 → 自动在后台启动（不弹窗口），之后一直开着，下次调用不用再等加载
- 接口 Key 直接读本机 data/config/settings.json，不用手填

也可以连别的电脑上的 TTS Studio：设环境变量 TTS_STUDIO_URL=http://IP:端口 和 TTS_STUDIO_KEY=接口Key。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from starlette.requests import Request  # 心跳接口的参数注解要能在模块里找到（本文件开了延迟注解）
try:  # mcp 2.x 把 FastMCP 改名成了 MCPServer，两个版本都兼容
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP
try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    from mcp.server.fastmcp.exceptions import ToolError

from splitter import split_sentences

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "data" / "config" / "settings.json"
OUT_DIR = ROOT / "data" / "outputs" / "mcp"
LOG_DIR = ROOT / "data" / "logs"
BASE = os.environ.get("TTS_STUDIO_URL", f"http://127.0.0.1:{os.environ.get('TTS_STUDIO_PORT', 7872)}").rstrip("/")
REMOTE = "TTS_STUDIO_URL" in os.environ
PUBLIC_BASE = ""  # 软件内置模式下由 mount() 填（开了局域网就是局域网地址）
FORMATS = ("wav", "mp3", "flac", "opus", "aac")

mcp = FastMCP(
    "tts-studio",
    instructions=(
        "本机中文配音（TTS Studio）。先用 list_voices 看有哪些音色，再用 speak 生成单段音频，"
        "整篇口播稿用 dub_script（自动拆句、逐句生成、按停顿合并）。"
        "生成的音频存成文件，返回的是文件路径；本地模型第一次调用要加载几十秒，属于正常。"
    ),
)


# ---------- 连接 TTS Studio ----------

def _key() -> str:
    if os.environ.get("TTS_STUDIO_KEY"):
        return os.environ["TTS_STUDIO_KEY"]
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))["server"]["api_key"]
    except Exception:
        raise RuntimeError("没读到接口 Key：先打开一次 TTS Studio（会自动生成），或设置环境变量 TTS_STUDIO_KEY")


def _port_open() -> bool:
    u = urlparse(BASE)
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=0.5):
            return True
    except OSError:
        return False


def _start_detached_windows(py: str, env: dict) -> bool:
    """用 WMI 启动服务：这样它不在 Agent 的进程树里，Agent 关掉 MCP 时不会被连带杀掉，
    模型加载一次就一直在，下次调用不用再等。失败返回 False（退回普通启动）。"""
    import base64
    sets = "&& ".join(f"set {k}={env[k]}" for k in
                      ("TTS_STUDIO_PORT", "TTS_STUDIO_NO_BROWSER", "PYTHONIOENCODING", "PYTHONUNBUFFERED"))
    cmdline = (f'cmd.exe /c "{sets}&& "{py}" app.py > "{LOG_DIR / "app.log"}" '
               f'2> "{LOG_DIR / "app.err.log"}""')
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0};"
          f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{"
          f"CommandLine={_ps_quote(cmdline)}; CurrentDirectory={_ps_quote(str(ROOT))}; ProcessStartupInformation=$si}};"
          "exit [int]$r.ReturnValue")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                            base64.b64encode(ps.encode("utf-16-le")).decode()],
                           capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0
    except Exception:
        return False


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _ensure_running():
    """本机的 TTS Studio 没开就在后台启动它，最多等 3 分钟。"""
    if _port_open():
        return
    if REMOTE:
        raise RuntimeError(f"连不上 {BASE}：确认那台电脑上的 TTS Studio 开着，并打开了局域网访问")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    py = Path(sys.executable).with_name("python.exe")
    env = dict(os.environ, TTS_STUDIO_PORT=str(urlparse(BASE).port), TTS_STUDIO_NO_BROWSER="1",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    py = str(py if py.exists() else sys.executable)
    if not (os.name == "nt" and _start_detached_windows(py, env)):
        subprocess.Popen([py, "app.py"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                         stdout=open(LOG_DIR / "app.log", "w", encoding="utf-8"),
                         stderr=open(LOG_DIR / "app.err.log", "w", encoding="utf-8"),
                         start_new_session=True)
    for _ in range(360):
        time.sleep(0.5)
        if _port_open():
            return
    raise RuntimeError("TTS Studio 3 分钟内没启动起来，报错看 data/logs/app.err.log")


def _call(method: str, path: str, **kw) -> requests.Response:
    _ensure_running()
    r = requests.request(method, BASE + path, headers={"Authorization": f"Bearer {_key()}"},
                         timeout=kw.pop("timeout", 30), **kw)
    if r.status_code >= 400:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:300]
        raise RuntimeError(f"TTS Studio 返回错误（{r.status_code}）：{msg}")
    return r


def _out_path(output_path: str | None, fmt: str, stem: str) -> Path:
    if output_path:
        p = Path(output_path).expanduser()
        if p.suffix == "" or p.is_dir():  # 给的是文件夹
            p = p / f"{stem}.{fmt}"
    else:
        p = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{stem}.{fmt}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _stem(text: str) -> str:
    keep = "".join(c for c in text if c.isalnum())[:12]
    return keep or "tts"


def _speech(text, voice, fmt, speed, instructions, seed) -> bytes:
    body = {"input": text, "voice": voice, "response_format": fmt, "speed": speed}
    if instructions:
        body["instructions"] = instructions
    if seed is not None:
        body["seed"] = seed
    return _call("POST", "/v1/audio/speech", json=body, timeout=900).content


def _file_url(p: Path) -> dict:
    """存在 data/outputs/mcp/ 下的文件给一个下载地址（别的电脑上的 Agent 拿不到本机路径，用这个下载）。"""
    try:
        rel = p.resolve().relative_to(OUT_DIR.resolve()).as_posix()
    except ValueError:
        return {}
    from urllib.parse import quote
    return {"url": f"{PUBLIC_BASE or BASE}/mcp-files/{quote(rel)}"}


def _wav_seconds(data: bytes) -> float:
    import io
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


# ---------- 工具 ----------

def _friendly(fn):
    """出错时把中文原因原样交给 Agent（不包成 ToolError 的话，mcp 2.x 只会显示 Error executing tool）。"""
    import functools

    @functools.wraps(fn)
    def wrap(*a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError:
            raise
        except Exception as ex:
            raise ToolError(str(ex)) from ex
    return wrap


@mcp.tool()
@_friendly
def list_voices() -> dict:
    """列出能用的音色。

    返回：
    - voices：音色库里存好的音色（voice 参数直接填 name），带所属模型和是否可用（ready）
    - builtin：模型自带的预设音色，写成「模型:音色ID」也能直接用，比如 qwen3:vivian
    """
    return _call("GET", "/v1/voices").json()


@mcp.tool()
@_friendly
def list_models() -> list[dict]:
    """列出所有配音模型（本地模型 + 配音 API），以及是否已下载 / 已配置好（ready）、没好的原因（note）。"""
    return [{k: m.get(k) for k in ("id", "name", "kind", "ready", "note")}
            for m in _call("GET", "/v1/models").json()["data"]]


@mcp.tool()
@_friendly
def speak(text: str, voice: str, output_path: str | None = None, format: str = "wav",
          speed: float = 1.0, instructions: str = "", seed: int | None = None) -> dict:
    """把一段文字生成成配音，存成音频文件，返回文件路径。适合一句到一小段话；整篇稿子请用 dub_script。

    参数：
    - text：要读的文字
    - voice：音色名（list_voices 里的 name），或「模型:音色ID」，比如 qwen3:vivian
    - output_path：存到哪里（文件路径或文件夹）；不填就存到 TTS Studio 的 data/outputs/mcp/
    - format：wav（默认）/ mp3 / flac / opus / aac
    - speed：语速，0.5～2.0
    - instructions：语气提示词，比如「平稳、像在跟朋友聊天」（只有部分模型支持）
    - seed：随机种子；同样的文字换个 seed 会换一种读法
    """
    if format not in FORMATS:
        raise ValueError(f"format 只能是 {', '.join(FORMATS)}")
    data = _speech(text, voice, format, speed, instructions, seed)
    p = _out_path(output_path, format, _stem(text))
    p.write_bytes(data)
    out = {"path": str(p), "bytes": len(data), **_file_url(p)}
    if format == "wav":
        out["seconds"] = round(_wav_seconds(data), 2)
    return out


@mcp.tool()
@_friendly
def dub_script(script: str, voice: str, output_path: str | None = None, pause_ms: int = 300,
               speed: float = 1.0, instructions: str = "", seed: int | None = None,
               keep_sentences: bool = False) -> dict:
    """给整篇口播稿配音：自动拆句 → 一句一句生成 → 按停顿合并成一条 wav，返回文件路径和每句的时间点。

    参数：
    - script：整篇稿子（按句号/问号/换行拆句，太长的句子在逗号处再拆）
    - voice：音色名或「模型:音色ID」，同 speak
    - output_path：合并后的 wav 存到哪里；不填就存到 data/outputs/mcp/
    - pause_ms：句子之间的停顿（毫秒）
    - speed / instructions / seed：同 speak
    - keep_sentences：true 时把每句单独的 wav 也存下来（放在同名文件夹里），方便后面单独重做

    返回的 sentences 里有每句的文字和在成品里的起止秒数，可以直接拿去对字幕或动画。
    """
    import numpy as np
    import soundfile as sf
    import io

    sents = [s.strip() for s in split_sentences(script) if s.strip()]
    if not sents:
        raise ValueError("稿子是空的")
    final = _out_path(output_path, "wav", _stem(sents[0]))
    sent_dir = final.with_suffix("")
    if keep_sentences:
        sent_dir.mkdir(parents=True, exist_ok=True)

    sr, parts, timeline, t = None, [], [], 0.0
    for i, s in enumerate(sents, 1):
        audio, r = sf.read(io.BytesIO(_speech(s, voice, "wav", speed, instructions, seed)), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = sr or r
        if keep_sentences:
            sf.write(sent_dir / f"{i:03d}.wav", audio, sr)
        if parts:
            gap = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
            parts.append(gap)
            t += len(gap) / sr
        parts.append(audio)
        dur = len(audio) / sr
        timeline.append({"index": i, "text": s, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur
    sf.write(final, np.concatenate(parts), sr, subtype="PCM_16")
    out = {"path": str(final), **_file_url(final), "seconds": round(t, 2), "count": len(sents), "sentences": timeline}
    if keep_sentences:
        out["sentence_dir"] = str(sent_dir)
    return out


# ---------- 连接状态（软件里「MCP 连接」页显示） ----------
# 两种接法都会在这里登记：
#   http  —— 豆包 / 扣子 / Cherry Studio / Cursor 等填网址连进来的，由 mount() 里的中间件登记
#   stdio —— Claude Code / Claude Desktop 等启动 mcp_server.py 的，由它每 15 秒发心跳登记
CONNS: dict[str, dict] = {}
ALIVE_SECONDS = 45  # 这么久没动静就算断开

_KNOWN_HOSTS = {  # 进程名 → 显示名（stdio 方式用来认出是谁启动的）
    "claude": "Claude", "cursor": "Cursor", "trae": "Trae", "doubao": "豆包", "code": "VS Code",
    "windsurf": "Windsurf", "cherry": "Cherry Studio", "codex": "Codex", "cline": "Cline", "kiro": "Kiro",
    "qoder": "Qoder", "workbuddy": "WorkBuddy", "codebuddy": "CodeBuddy", "lingma": "通义灵码",
    "chatgpt": "ChatGPT", "zed": "Zed", "goose": "Goose",
}


def touch(cid: str, name, transport: str):
    now = time.time()
    c = CONNS.setdefault(cid, {"first": now, "calls": 0, "streams": 0})
    c.update(name=name or c.get("name") or "未知客户端", transport=transport, last=now, closed=False)
    return c


def close(cid: str):
    if cid in CONNS:
        CONNS[cid]["closed"] = True


def status() -> list[dict]:
    now, out = time.time(), []
    for cid, c in sorted(CONNS.items(), key=lambda kv: -kv[1]["last"]):
        alive = not c.get("closed") and (c.get("streams", 0) > 0 or now - c["last"] < ALIVE_SECONDS)
        if not alive and now - c["last"] > 3600:  # 断开一小时就不显示了
            continue
        out.append({**c, "id": cid, "alive": alive})
    return out


def _host_name() -> str:
    """stdio 模式：顺着父进程往上找，认出是哪个软件启动了我。"""
    try:
        import psutil
        p = psutil.Process().parent()
        for _ in range(6):
            if p is None:
                break
            n = p.name().lower().removesuffix(".exe")
            for k, v in _KNOWN_HOSTS.items():
                if k in n:
                    return v
            p = p.parent()
    except Exception:
        pass
    return "本机 Agent"


def _heartbeat_loop(cid: str, name: str):
    """stdio 模式：软件开着就每 15 秒报一次「我还连着」；软件没开不会为了心跳去启动它。"""
    while True:
        if _port_open():
            try:
                requests.post(BASE + "/v1/mcp/heartbeat", json={"id": cid, "name": name}, timeout=3,
                              headers={"Authorization": f"Bearer {_key()}"})
            except Exception:
                pass
        time.sleep(15)


# ---------- 挂到 TTS Studio 自己的服务上（HTTP 方式） ----------

class _Guard:
    """/mcp 的鉴权 + 记录是谁连进来的（从 initialize 请求里读 clientInfo）。"""

    def __init__(self, app, check_key):
        self.inner, self.check_key = app, check_key

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not (path == "/mcp" or path.startswith("/mcp/")):
            return await self.inner(scope, receive, send)
        import json as _json
        from starlette.requests import Request as _Req
        from starlette.responses import JSONResponse
        req = _Req(scope)
        if not self.check_key(req):
            resp = JSONResponse({"error": "Key 不对：请求头加 Authorization: Bearer <Key>，或网址后面加 ?key=<Key>"},
                                status_code=401)
            return await resp(scope, receive, send)
        sid = req.headers.get("mcp-session-id")
        method = scope["method"]
        body, more = b"", True  # 先把请求体读出来看一眼（initialize 里有客户端名字），再原样交给 MCP
        if method == "POST":
            while more:
                m = await receive()
                body += m.get("body", b"")
                more = m.get("more_body", False)
        name = None
        if body:
            try:
                msg = _json.loads(body)
                for one in (msg if isinstance(msg, list) else [msg]):
                    if one.get("method") == "initialize":
                        ci = (one.get("params") or {}).get("clientInfo") or {}
                        name = ci.get("title") or ci.get("name")
            except Exception:
                pass
        replay = {"done": method != "POST"}

        async def recv():
            if not replay["done"]:
                replay["done"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        got = {"sid": sid}

        async def snd(m):
            if m["type"] == "http.response.start" and not got["sid"]:
                for k, v in m.get("headers", []):
                    if k.decode().lower() == "mcp-session-id":
                        got["sid"] = v.decode()
            await send(m)

        if method == "DELETE" and sid:
            close("http-" + sid)
            return await self.inner(scope, recv, snd)
        if method == "GET" and sid:  # 客户端挂着的长连接：开着就算已连接
            c = touch("http-" + sid, None, "http")
            c["streams"] += 1
            try:
                return await self.inner(scope, recv, snd)
            finally:
                c["streams"] = max(0, c["streams"] - 1)
                c["last"] = time.time()
        await self.inner(scope, recv, snd)
        if got["sid"]:
            c = touch("http-" + got["sid"], name, "http")
            if b'"tools/call"' in body:
                c["calls"] += 1


def mount(app, port: int, lan: bool, check_key):
    """在软件的 FastAPI 上加：
      /mcp              —— Streamable HTTP 方式的 MCP（豆包、扣子、Cherry Studio、Cursor 等填这个网址）
      /mcp-files/...    —— 生成的音频下载地址
      /v1/mcp/heartbeat —— stdio 方式的心跳
    鉴权：请求头 Authorization: Bearer <Key>，或者网址后面加 ?key=<Key>（有些软件只能填网址）。
    check_key(request) -> bool。
    """
    global BASE, PUBLIC_BASE
    import contextlib

    from fastapi.responses import FileResponse, JSONResponse

    BASE = f"http://127.0.0.1:{port}"
    PUBLIC_BASE = BASE
    if lan:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
                sk.connect(("10.255.255.255", 1))
                PUBLIC_BASE = f"http://{sk.getsockname()[0]}:{port}"
        except OSError:
            pass

    sub = mcp.streamable_http_app(streamable_http_path="/mcp", host="0.0.0.0" if lan else "127.0.0.1")
    for r in sub.routes:  # 路由直接并进主程序（要在网页挂到 / 之前调用）
        app.router.routes.append(r)

    old_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(a):
        async with mcp.session_manager.run():
            async with old_lifespan(a) as st:
                yield st
    app.router.lifespan_context = lifespan

    @app.post("/v1/mcp/heartbeat")
    async def _hb(request: Request):
        if not check_key(request):
            return JSONResponse({"error": "key"}, status_code=401)
        b = await request.json()
        touch("stdio-" + str(b.get("id"))[:40], str(b.get("name") or "本机 Agent")[:40], "stdio")
        return {"ok": True}

    @app.get("/mcp-files/{rel:path}")
    def _file(rel: str):
        f = (OUT_DIR / rel).resolve()
        if OUT_DIR.resolve() not in f.parents or not f.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(f)

    app.add_middleware(_Guard, check_key=check_key)


if __name__ == "__main__":
    import threading
    import uuid
    threading.Thread(target=_heartbeat_loop, args=(uuid.uuid4().hex[:12], _host_name()), daemon=True).start()
    mcp.run()
