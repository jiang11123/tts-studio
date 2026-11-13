"""TTS Studio 桌面版：用系统自带的 WebView2 开一个独立软件窗口，不走浏览器。

- 双击启动：后台起服务（无黑窗口），窗口先显示“启动中”，服务好了自动进界面
- 关闭窗口：由本窗口启动的服务一起关掉
- 如果服务已经在跑（比如之前用 start.bat 开的），直接连上去，关窗口时不动它
- 重复双击：不会开第二个窗口，而是把已有窗口提到最前
"""
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("TTS_STUDIO_PORT", 7872))
URL = f"http://127.0.0.1:{PORT}"
TITLE = "TTS Studio"
ICON = ROOT / "launcher" / "tts-studio.ico"
LOG_DIR = ROOT / "data" / "logs"

LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f6f5f2;color:#333}
.b{display:flex;gap:6px;justify-content:center;height:44px;align-items:center;margin-bottom:22px}
.b i{width:6px;background:#5b6cff;border-radius:3px;animation:w 1s ease-in-out infinite}
.b i:nth-child(2){animation-delay:.1s}.b i:nth-child(3){animation-delay:.2s}
.b i:nth-child(4){animation-delay:.3s}.b i:nth-child(5){animation-delay:.4s}
@keyframes w{0%,100%{height:8px}50%{height:40px}}
h1{font-size:20px;font-weight:600;margin:0 0 8px;text-align:center}
p{margin:0;color:#888;font-size:14px;text-align:center}
</style></head><body><div><div class="b"><i></i><i></i><i></i><i></i><i></i></div>
<h1>TTS Studio 正在启动</h1><p>大概十几秒，好了会自动进入</p></div></body></html>"""

ERROR_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f6f5f2;color:#333;text-align:center}
h1{font-size:20px;color:#c0392b}p{color:#666;font-size:14px}code{background:#eee;padding:2px 6px;border-radius:4px}
</style></head><body><div><h1>启动失败</h1>
<p>服务 3 分钟内没有起来，报错在 <code>tts-studio\\data\\logs\\app.err.log</code></p></div></body></html>"""


def port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return True
    except OSError:
        return False


def find_python() -> str:
    """服务进程用 python.exe（不是 pythonw），日志才写得进文件。"""
    exe = Path(sys.executable)
    cand = exe.with_name("python.exe")
    return str(cand if cand.exists() else exe)


def start_server() -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TTS_STUDIO_PORT=str(PORT), TTS_STUDIO_NO_BROWSER="1",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [find_python(), "app.py"], cwd=ROOT, env=env,
        stdout=open(LOG_DIR / "app.log", "w", encoding="utf-8"),
        stderr=open(LOG_DIR / "app.err.log", "w", encoding="utf-8"),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def stop_server(proc: subprocess.Popen | None):
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def single_instance() -> bool:
    """已经有窗口开着：把它提到前面，然后本次退出。"""
    ctypes.windll.kernel32.CreateMutexW(None, False, f"TTSStudioDesktopMutex_{PORT}")  # 同一个端口只开一个窗口
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        hwnd = ctypes.windll.user32.FindWindowW(None, TITLE)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        return False
    return True


def main():
    if not single_instance():
        return
    try:  # 任务栏单独成组，不和其他 Python 程序挤在一起
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TTSStudio.Desktop")
    except Exception:
        pass

    owned = None if port_open() else start_server()
    window = webview.create_window(TITLE, html=LOADING_HTML, width=1400, height=900,
                                   min_size=(960, 640), background_color="#f6f5f2")

    def boot():
        for _ in range(360):
            if port_open():
                window.load_url(URL)
                return
            if owned and owned.poll() is not None:  # 服务进程直接退出了
                break
            time.sleep(0.5)
        window.load_html(ERROR_HTML)

    window.events.closed += lambda: stop_server(owned)
    webview.settings["ALLOW_DOWNLOADS"] = True
    try:
        webview.start(boot, icon=str(ICON), private_mode=False, storage_path=str(ROOT / "data" / "webview"))
    finally:
        stop_server(owned)


if __name__ == "__main__":
    main()
