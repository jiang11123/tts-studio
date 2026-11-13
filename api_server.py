"""开放接口：让别的程序 / Agent 直接调用本机的配音模型。

格式和 OpenAI 的语音接口一样（POST /v1/audio/speech），所以支持「OpenAI 兼容 TTS」的工具改个地址和 Key 就能用。
  GET  /v1/voices          音色库里的音色（voice 填名字）
  GET  /v1/models          有哪些模型、能不能用
  POST /v1/audio/speech    {"input": "要读的字", "voice": "音色名", "response_format": "wav", "speed": 1.0,
                            "instructions": "语气提示词（可选）", "seed": 42}
voice 也可以写「模型:音色ID」，不用先建音色：qwen3:vivian、minimax:female-shaonv、volc:BV700_streaming、openai:alloy。
鉴权：请求头 Authorization: Bearer <设置页里的接口 Key>。
"""

from __future__ import annotations

import secrets
import subprocess

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import engines as E
from speed import change_speed

FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "pcm": "audio/L16", "opus": "audio/ogg",
           "aac": "audio/aac"}


def server_conf() -> dict:
    """接口设置；第一次用时自动生成一个 Key。"""
    s = E.settings_load()
    srv = s.setdefault("server", {})
    changed = False
    if "api_enabled" not in srv:
        srv["api_enabled"], changed = True, True
    if not srv.get("api_key"):
        srv["api_key"], changed = "sk-tts-" + secrets.token_urlsafe(24), True
    srv.setdefault("lan", False)
    if changed:
        E.settings_save(s)
    return srv


def new_key() -> str:
    s = E.settings_load()
    s.setdefault("server", {})["api_key"] = "sk-tts-" + secrets.token_urlsafe(24)
    E.settings_save(s)
    return s["server"]["api_key"]


def _err(status, msg, code="invalid_request_error"):
    return JSONResponse({"error": {"message": msg, "type": code}}, status_code=status)


def _encode(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "pcm":  # 16 位单声道小端，和 OpenAI 一样
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    if fmt in ("wav", "flac"):
        import io
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format=fmt.upper(), subtype="PCM_16")
        return buf.getvalue()
    exe = E.ffmpeg_exe()
    if not exe:
        raise ValueError(f"{fmt} 格式需要 ffmpeg：pip install imageio-ffmpeg，或改用 wav")
    args = {"mp3": ["-f", "mp3", "-b:a", "128k"], "opus": ["-f", "ogg", "-c:a", "libopus"],
            "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"]}[fmt]
    p = subprocess.run([exe, "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", *args, "pipe:1"],
                       input=np.asarray(audio, np.float32).tobytes(), capture_output=True)
    if p.returncode:
        raise ValueError("转码失败：" + p.stderr.decode("utf-8", "ignore")[:200])
    return p.stdout


def register(app: FastAPI, list_voices, read_voice):
    """把 /v1 接口挂到 app 上。list_voices() -> 名字列表；read_voice(名字) -> 音色 dict。"""

    def check(request: Request):
        srv = server_conf()
        if not srv.get("api_enabled"):
            return _err(403, "开放接口没有开启：在「开放接口」页里打开", "permission_error")
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not secrets.compare_digest(key.encode(), srv["api_key"].encode()):
            return _err(401, "Key 不对：请求头加 Authorization: Bearer <设置页里的接口 Key>", "authentication_error")
        return None

    def resolve(name: str):
        name = (name or "").strip()
        if name in list_voices():
            return read_voice(name)
        if ":" in name:
            eid, spk = name.split(":", 1)
            if eid in E.ENGINES and "speaker" in E.ENGINES[eid].voice_kinds:
                return {"engine": eid, "kind": "speaker", "speaker": spk}
        return None

    @app.get("/v1/voices")
    def voices(request: Request):
        if (e := check(request)) is not None:
            return e
        out = []
        for n in list_voices():
            v = read_voice(n)
            eng = E.engine_of(v)
            out.append({"name": n, "engine": eng.id, "kind": v.get("kind") or ("clone" if v.get("ref_audio") else "design"),
                        "ready": eng.ready()[0]})
        builtin = {eid: e.speakers() for eid, e in E.ENGINES.items() if "speaker" in e.voice_kinds}
        return {"voices": out, "builtin": {k: [f"{k}:{s}" for s in v] for k, v in builtin.items()}}

    @app.get("/v1/models")
    def models(request: Request):
        if (e := check(request)) is not None:
            return e
        data = []
        for eid, eng in E.ENGINES.items():
            ok, why = eng.ready()
            data.append({"id": eid, "object": "model", "owned_by": "tts-studio", "name": eng.name,
                         "kind": eng.kind, "ready": ok, "note": why})
        return {"object": "list", "data": data}

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        if (e := check(request)) is not None:
            return e
        try:
            body = await request.json()
        except Exception:
            return _err(400, "请求体要是 JSON")
        text = (body.get("input") or "").strip()
        if not text:
            return _err(400, "input 是空的")
        voice = resolve(body.get("voice", ""))
        if voice is None:
            return _err(400, f"找不到音色「{body.get('voice', '')}」。可用的音色：GET /v1/voices；"
                             "也可以写 模型:音色ID，比如 qwen3:vivian")
        fmt = (body.get("response_format") or "wav").lower()
        if fmt not in FORMATS:
            return _err(400, f"不支持的格式 {fmt}，可选：{', '.join(FORMATS)}")
        tone = body.get("instructions") or ""
        seed = int(body.get("seed", voice.get("seed", 42)))
        cfg = float(body.get("cfg", 2.0))
        speed = float(body.get("speed", 1.0))

        def work():
            audio = E.synth(text, voice, tone, cfg, seed)
            if abs(speed - 1.0) > 1e-3:
                audio = change_speed(audio, E.OUT_SR, max(0.5, min(2.0, speed)))
            return _encode(audio, E.OUT_SR, fmt)

        from starlette.concurrency import run_in_threadpool
        try:
            data = await run_in_threadpool(work)  # 生成很慢，放线程里跑，别卡住网页
        except Exception as ex:  # gr.Error 的 message 就是给人看的中文说明
            return _err(400 if "还没" in str(ex) or "缺" in str(ex) else 500, str(getattr(ex, "message", ex)),
                        "server_error")
        return Response(data, media_type=FORMATS[fmt])
