"""安装版的入口：TTS Studio.exe → runtime\\python\\pythonw.exe boot.py。

第一次打开时运行环境还没装（安装包里只带了 Python 和 uv，PyTorch 等依赖约 3 GB 不进安装包），
弹一个进度窗口，用 uv 从国内镜像装好；之后每次打开直接进软件。
requirements.txt 变了（升级到新版）会自动补装差的那部分。
只用标准库 + tkinter，因为这时候别的包都还没有。
"""
import hashlib
import os
import queue
import runpy
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RT = ROOT / "runtime"
PY = RT / "python" / "python.exe"
UV = RT / "uv.exe"
MARK = RT / "deps.ok"
LOG = ROOT / "data" / "logs" / "setup.log"
TORCH = "2.9.1"
TUNA = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI = "https://pypi.org/simple"
NO_WINDOW = 0x08000000


def has_nvidia() -> bool:
    sysroot = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    return bool(shutil.which("nvidia-smi")) or (sysroot / "System32" / "nvidia-smi.exe").is_file()


def want_key(gpu: bool) -> str:
    req = (ROOT / "requirements.txt").read_bytes()
    return hashlib.sha1(req + f"|torch={TORCH}|gpu={gpu}".encode()).hexdigest()


def is_ready(gpu: bool) -> bool:
    try:
        return MARK.read_text(encoding="utf-8").strip() == want_key(gpu)
    except OSError:
        return False


def plan(gpu: bool):
    """[(说明, [几种来源的参数，前一个失败就换下一个])]"""
    base = [str(UV), "pip", "install", "--python", str(PY)]
    if gpu:
        tv = f"{TORCH}+cu128"
        torch_srcs = [
            [f"torch=={tv}", f"torchaudio=={tv}", "--index-url", TUNA,
             "--find-links", "https://mirrors.aliyun.com/pytorch-wheels/cu128/"],
            [f"torch=={tv}", f"torchaudio=={tv}", "--index-url", "https://download.pytorch.org/whl/cu128",
             "--extra-index-url", TUNA, "--index-strategy", "unsafe-best-match"],
        ]
        torch_note = "第 1/2 步：下载 PyTorch 显卡版（约 2.5 GB，最久的一步）"
    else:
        torch_srcs = [[f"torch=={TORCH}", f"torchaudio=={TORCH}", "--index-url", src] for src in (TUNA, PYPI)]
        torch_note = "第 1/2 步：下载 PyTorch（没检测到 N 卡，装 CPU 版，约 300 MB）"
    req = str(ROOT / "requirements.txt")
    req_srcs = [["-r", req, "--index-url", src] for src in (TUNA, PYPI)]
    return [(torch_note, [base + s for s in torch_srcs]),
            ("第 2/2 步：安装其他依赖（Gradio、Transformers 等）", [base + s for s in req_srcs])]


def uv_env():
    env = dict(os.environ)
    own_cache = "TTS_STUDIO_UV_CACHE" not in env
    env["UV_CACHE_DIR"] = env.get("TTS_STUDIO_UV_CACHE") or str(RT / "uv-cache")
    env.update(UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never", UV_NO_CONFIG="1", UV_HTTP_TIMEOUT="300",
               UV_BREAK_SYSTEM_PACKAGES="1", PYTHONIOENCODING="utf-8", NO_COLOR="1")
    return env, own_cache


def launch_app():
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    runpy.run_path(str(ROOT / "desktop.py"), run_name="__main__")


# ───────────────────────── 首次安装窗口 ─────────────────────────
def setup_window(gpu: bool) -> bool:
    import tkinter as tk
    from tkinter import ttk

    LOG.parent.mkdir(parents=True, exist_ok=True)
    msgs: "queue.Queue[tuple[str, str]]" = queue.Queue()
    state = {"proc": None, "ok": False, "cancel": False, "t0": time.time()}

    def worker():
        env, own_cache = uv_env()
        with open(LOG, "a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 开始安装（显卡：{gpu}） =====\n")
            for note, variants in plan(gpu):
                msgs.put(("step", note))
                done = False
                for k, cmd in enumerate(variants, 1):
                    if state["cancel"]:
                        return
                    if k > 1:
                        msgs.put(("line", f"换一个下载源重试（{k}/{len(variants)}）…"))
                    log.write("$ " + " ".join(cmd) + "\n")
                    log.flush()
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                         creationflags=NO_WINDOW, text=True, encoding="utf-8", errors="replace")
                    state["proc"] = p
                    for line in p.stdout:
                        log.write(line)
                        line = line.strip()
                        if line:
                            msgs.put(("line", line))
                    p.wait()
                    log.flush()
                    if p.returncode == 0:
                        done = True
                        break
                if not done:
                    if not state["cancel"]:
                        msgs.put(("fail", note.split("：", 1)[-1]))
                    return
            MARK.write_text(want_key(gpu), encoding="utf-8")
            if own_cache:  # 装好就删掉下载缓存，省 3 GB 硬盘
                shutil.rmtree(RT / "uv-cache", ignore_errors=True)
            log.write("安装完成\n")
        msgs.put(("done", ""))

    root = tk.Tk()
    root.title("TTS Studio · 首次启动")
    try:
        root.iconbitmap(str(ROOT / "launcher" / "tts-studio.ico"))
    except Exception:
        pass
    W, H = 560, 280
    root.geometry(f"{W}x{H}+{(root.winfo_screenwidth() - W) // 2}+{(root.winfo_screenheight() - H) // 3}")
    root.resizable(False, False)
    root.configure(bg="#ffffff")
    font = ("Microsoft YaHei UI", 10)
    tk.Label(root, text="正在准备运行环境", font=("Microsoft YaHei UI", 15, "bold"), bg="#fff", fg="#1d1d27"
             ).pack(anchor="w", padx=28, pady=(24, 4))
    tk.Label(root, text="只有第一次打开需要，装好以后双击就能直接用。需要联网，大约 10～20 分钟。",
             font=font, bg="#fff", fg="#6b6b7a").pack(anchor="w", padx=28)
    step = tk.Label(root, text="", font=("Microsoft YaHei UI", 10, "bold"), bg="#fff", fg="#7c5cff")
    step.pack(anchor="w", padx=28, pady=(18, 6))
    bar = ttk.Progressbar(root, mode="indeterminate", length=W - 56)
    bar.pack(padx=28)
    bar.start(12)
    line = tk.Label(root, text="", font=("Microsoft YaHei UI", 9), bg="#fff", fg="#8a8a95", anchor="w", width=70)
    line.pack(anchor="w", padx=28, pady=(8, 0))
    btns = tk.Frame(root, bg="#fff")
    btns.pack(side="bottom", fill="x", padx=28, pady=18)
    clock = tk.Label(btns, text="", font=font, bg="#fff", fg="#8a8a95")
    clock.pack(side="left")

    def cancel():
        state["cancel"] = True
        p = state["proc"]
        if p and p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], creationflags=NO_WINDOW,
                           capture_output=True)
        root.destroy()

    def retry():
        for b in extra:
            b.destroy()
        extra.clear()
        bar.configure(mode="indeterminate")
        bar.start(12)
        line.configure(fg="#8a8a95")
        state["t0"] = time.time()
        threading.Thread(target=worker, daemon=True).start()

    extra = []
    tk.Button(btns, text="取消", font=font, width=8, command=cancel).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel)

    def pump():
        try:
            while True:
                kind, text = msgs.get_nowait()
                if kind == "step":
                    step.configure(text=text)
                elif kind == "line":
                    line.configure(text=text[:90])
                elif kind == "done":
                    state["ok"] = True
                    root.destroy()
                    return
                elif kind == "fail":
                    bar.stop()
                    step.configure(text=f"安装失败：{text}")
                    line.configure(text="多半是网络问题。检查网络后点「重试」；还不行就把日志发给作者。", fg="#d4380d")
                    b1 = tk.Button(btns, text="重试", font=font, width=8, command=retry)
                    b2 = tk.Button(btns, text="打开日志", font=font, width=10, command=lambda: os.startfile(LOG))
                    b1.pack(side="right", padx=(8, 0))
                    b2.pack(side="right", padx=(8, 0))
                    extra.extend([b1, b2])
        except queue.Empty:
            pass
        m, s = divmod(int(time.time() - state["t0"]), 60)
        clock.configure(text=f"已用时 {m}:{s:02d}")
        root.after(300, pump)

    threading.Thread(target=worker, daemon=True).start()
    root.after(300, pump)
    root.mainloop()
    return state["ok"]


def main():
    gpu = has_nvidia()
    if not is_ready(gpu) and not setup_window(gpu):
        return
    launch_app()


if __name__ == "__main__":
    main()
