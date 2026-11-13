"""打安装包：python packaging/build.py  →  dist/TTS-Studio-Setup-<版本>.exe

安装包里有：程序代码 + 一份独立的 Python 3.11（uv 管理的 python-build-standalone，可整个搬走）+ uv.exe + 启动器。
PyTorch 等依赖不进安装包，第一次打开时 boot.py 联网装。
需要：本机装过 uv、uv 装过 Python 3.11（uv python install 3.11）；用 Windows 自带的 .NET Framework csc.exe 编译 exe。
"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

VERSION = (Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="utf-8").strip()  # 版本号只在根目录 VERSION 里改
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = ROOT / "dist"
STAGE = DIST / "stage"
CSC = Path(os.environ.get("SystemRoot", r"C:\Windows")) / r"Microsoft.NET\Framework64\v4.0.30319\csc.exe"
ICON = ROOT / "launcher" / "tts-studio.ico"

APP_FILES = ["app.py", "engines.py", "updater.py", "api_server.py", "mcp_server.py", "speed.py", "splitter.py", "desktop.py",
             "requirements.txt", "README.md", "LICENSE", "NOTICE"]
APP_DIRS = ["third_party"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def find_python311() -> Path:
    out = subprocess.run(["uv", "python", "find", "3.11", "--managed-python"], capture_output=True, text=True)
    exe = Path(out.stdout.strip())
    if out.returncode or not exe.is_file():
        sys.exit("没找到 uv 管理的 Python 3.11：先运行  uv python install 3.11")
    return exe.parent


def csc(src: Path, out: Path, extra=()):
    cmd = [str(CSC), "/nologo", "/codepage:65001", "/optimize+", "/target:winexe", "/platform:anycpu",
           f"/win32icon:{ICON}", f"/out:{out}", "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll",
           "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll", *extra, str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="mbcs", errors="replace")
    if r.returncode:
        sys.exit(f"编译 {src.name} 失败：\n{r.stdout}\n{r.stderr}")


def main():
    if not CSC.is_file():
        sys.exit(f"没找到 {CSC}（Windows 自带的 .NET Framework 4 编译器）")
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)

    print("1/4 复制程序文件")
    for f in APP_FILES:
        shutil.copy2(ROOT / f, STAGE / f)
    for d in APP_DIRS:
        shutil.copytree(ROOT / d, STAGE / d, ignore=SKIP)
    (STAGE / "launcher").mkdir()
    shutil.copy2(ICON, STAGE / "launcher" / ICON.name)
    shutil.copy2(HERE / "boot.py", STAGE / "boot.py")
    (STAGE / "VERSION").write_text(VERSION, encoding="utf-8")

    print("2/4 复制独立 Python 和 uv")
    pydir = find_python311()
    shutil.copytree(pydir, STAGE / "runtime" / "python", ignore=SKIP)
    (STAGE / "runtime" / "python" / "Lib" / "EXTERNALLY-MANAGED").unlink(missing_ok=True)  # 这是软件私有的 Python，允许往里装包
    shutil.copy2(shutil.which("uv"), STAGE / "runtime" / "uv.exe")

    print("3/4 编译启动器")
    csc(HERE / "launcher.cs", STAGE / "TTS Studio.exe")

    print("4/4 打包安装程序")
    payload = DIST / "payload.zip"
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE).as_posix())
    out = DIST / f"TTS-Studio-Setup-{VERSION}.exe"
    csc(HERE / "setup.cs", out, [f"/resource:{payload},payload.zip"])
    payload.unlink()
    print(f"完成：{out}  ({out.stat().st_size / 1024 ** 2:.0f} MB)")


if __name__ == "__main__":
    main()
