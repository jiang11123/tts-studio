"""自动更新：查 GitHub Releases 上有没有新版本，有的话下载新安装包、关掉软件、静默装好、重新打开。

- 只有「安装版」（目录里有 TTS Studio.exe 和 runtime\\）能一键更新；从源码运行的只提示去下载页。
- 静默安装会覆盖程序文件，data\\（音色、项目、设置、模型）和已装好的运行环境都不动；
  新版 requirements.txt 有变化时，下次打开 boot.py 会自动补装差的依赖。
- 发版本用 packaging/release.py。
"""
import base64
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "jiang11123/tts-studio"
PAGE = f"https://github.com/{REPO}/releases"
API = f"https://api.github.com/repos/{REPO}/releases?per_page=10"
UPD_DIR = ROOT / "data" / "updates"
CHECK_EVERY = 6 * 3600

_state = {"checked": 0.0, "latest": None, "error": "", "busy": False, "progress": "", "lock": threading.Lock()}


def current() -> str:
    try:
        return (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"


def _vtuple(v: str):
    out = []
    for part in v.lstrip("vV").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def installed() -> bool:
    """安装版才能自己更新（源码运行的没有 TTS Studio.exe / runtime）。"""
    return (ROOT / "TTS Studio.exe").is_file() and (ROOT / "runtime" / "python").is_dir()


def check(force: bool = False) -> dict | None:
    """返回比当前新的版本 {version, notes, url, size, page}，没有就 None。6 小时最多查一次。"""
    with _state["lock"]:
        if not force and time.time() - _state["checked"] < CHECK_EVERY:
            return _state["latest"]
        _state["checked"] = time.time()
        try:
            req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": "tts-studio-updater"})
            rels = json.load(urllib.request.urlopen(req, timeout=10))
            best = None
            for r in rels:
                if r.get("draft"):
                    continue
                asset = next((a for a in r.get("assets", []) if a["name"].lower().endswith(".exe")), None)
                v = (r.get("tag_name") or "").lstrip("vV")
                if asset and (best is None or _vtuple(v) > _vtuple(best["version"])):
                    best = {"version": v, "notes": r.get("body") or "", "url": asset["browser_download_url"],
                            "size": asset.get("size", 0), "page": r.get("html_url") or PAGE}
            _state["latest"] = best if best and _vtuple(best["version"]) > _vtuple(current()) else None
            _state["error"] = ""
        except Exception as e:  # 没网 / GitHub 连不上：不打扰用户，下次再查
            _state["error"] = str(e)
        return _state["latest"]


def progress() -> str:
    return _state["progress"]


def _download(url: str, dest: Path, size: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tts-studio-updater"})
    with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or size or 0)
        got = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                _state["progress"] = f"正在下载新版本… {got * 100 // total}%（{got / 1048576:.0f}/{total / 1048576:.0f} MB）"
    if size and tmp.stat().st_size != size:
        raise RuntimeError("下载不完整，再点一次更新试试")
    tmp.replace(dest)


def _run_detached(cmdline: str):
    """用 WMI 启动：不在软件的进程树里，软件被关掉时它不会被连带结束。"""
    def q(s):
        return "'" + s.replace("'", "''") + "'"
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0};"
          f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{"
          f"CommandLine={q(cmdline)}; CurrentDirectory={q(str(UPD_DIR))}; ProcessStartupInformation=$si}};"
          "exit [int]$r.ReturnValue")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                        base64.b64encode(ps.encode("utf-16-le")).decode()],
                       capture_output=True, timeout=60, creationflags=0x08000000)
    if r.returncode != 0:
        raise RuntimeError("启动更新程序失败")


def apply(info: dict):
    """下载 → 起一个独立的更新脚本（等软件关掉 → 静默安装 → 重新打开）→ 返回后由调用方提示用户。"""
    if not installed():
        raise RuntimeError("从源码运行的版本不能一键更新：去下载页下载新安装包，或者 git pull")
    if _state["busy"]:
        raise RuntimeError("已经在更新了")
    _state["busy"] = True
    try:
        exe = UPD_DIR / f"TTS-Studio-Setup-{info['version']}.exe"
        if not (exe.is_file() and (not info.get("size") or exe.stat().st_size == info["size"])):
            _download(info["url"], exe, info.get("size", 0))
        _state["progress"] = "下载好了，软件马上关闭并自动安装新版本，装完会自己重新打开（约半分钟）…"
        root = str(ROOT)
        # 结束本目录下所有进程（窗口、服务）→ 静默装到原目录 → 重新打开
        ps = (f"Start-Sleep 3; "
              f"Get-Process | Where-Object {{ $_.Path -and $_.Path.StartsWith({_q(root + os.sep)}, 'OrdinalIgnoreCase') }} "
              f"| Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep 2; "
              f"$p = Start-Process -FilePath {_q(str(exe))} -ArgumentList '/S', {_q('/D=' + root)}, '/NOSHORTCUTS' "
              f"-Wait -PassThru; "
              f"Add-Content -Path {_q(str(UPD_DIR / 'update.log'))} -Value (\"$(Get-Date) 安装 {info['version']} 退出码 \" + $p.ExitCode); "
              f"Start-Process -FilePath {_q(str(ROOT / 'TTS Studio.exe'))} -WorkingDirectory {_q(root)}")
        _run_detached("powershell -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand "
                      + base64.b64encode(ps.encode("utf-16-le")).decode())
    except Exception:
        _state["busy"] = False
        raise


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"
