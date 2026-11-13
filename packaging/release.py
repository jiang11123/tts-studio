"""一键发新版本：改版本号 → 打安装包 → 提交推送到 GitHub → 建 Release 并上传安装包。
装了 TTS Studio 的人，打开软件后左下角会提示「发现新版本」，点一下就自动更新。

用法：
    python packaging/release.py 0.1.1 "这次改了什么（一行一条，用 \\n 分隔也行）"
    python packaging/release.py --push-only "提交说明"     只把代码推到 GitHub，不发版本（用户收不到更新）

GitHub 登录用的是 git 自己存的凭据（第一次 git push 时浏览器授权过的那个）。
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "jiang11123/tts-studio"
def _gcm() -> str:
    """直接指定 Git 自带的凭据管理器：用默认设置时，推送完会弹「选择凭据助手」小窗卡住。"""
    import shutil
    exe = shutil.which("git-credential-manager")
    if not exe:
        r = subprocess.run(["git", "--exec-path"], capture_output=True, text=True)
        for cand in Path(r.stdout.strip()).parents:
            for sub in ("bin", "mingw64/bin", "ucrt64/bin"):
                if (cand / sub / "git-credential-manager.exe").is_file():
                    exe = str(cand / sub / "git-credential-manager.exe")
                    break
            if exe:
                break
    return f'!"{Path(exe).as_posix()}"' if exe else "manager"


GCM = _gcm()
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, check=True, inp=None):
    cmd = ["git", "-c", "credential.helper=", "-c", f"credential.helper={GCM}", *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", input=inp,
                       env={**__import__("os").environ, "GCM_INTERACTIVE": "auto"})
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} 失败：\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def token() -> str:
    out = git("credential", "fill", inp="protocol=https\nhost=github.com\n\n")
    tok = next((ln[9:] for ln in out.splitlines() if ln.startswith("password=")), "")
    if not tok:
        sys.exit("拿不到 GitHub 登录凭据：先手动 git push 一次，在弹出的浏览器里授权")
    return tok


def api(method, url, tok, data=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "Content-Type": ctype,
        "User-Agent": "tts-studio-release"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def commit_and_push(msg: str):
    git("add", "-A")
    if git("status", "--porcelain"):
        git("commit", "-q", "-m", msg + TRAILER)
    git("push", "-q", "origin", "HEAD")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--push-only":
        commit_and_push(sys.argv[2] if len(sys.argv) > 2 else "更新")
        print(f"已推送到 https://github.com/{REPO}")
        return
    if len(sys.argv) < 3 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        sys.exit(__doc__)
    ver, notes = sys.argv[1], sys.argv[2].replace("\\n", "\n")
    old = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if tuple(map(int, ver.split("."))) <= tuple(map(int, old.split("."))):
        sys.exit(f"新版本号 {ver} 要比现在的 {old} 大")
    (ROOT / "VERSION").write_text(ver + "\n", encoding="utf-8")

    print(f"1/4 打安装包 v{ver}")
    r = subprocess.run([sys.executable, str(ROOT / "packaging" / "build.py")], cwd=ROOT)
    if r.returncode:
        (ROOT / "VERSION").write_text(old + "\n", encoding="utf-8")
        sys.exit("打包失败，版本号已改回")
    exe = ROOT / "dist" / f"TTS-Studio-Setup-{ver}.exe"

    print("2/4 提交并推送代码")
    commit_and_push(f"发布 v{ver}\n\n{notes}")
    git("tag", f"v{ver}")
    git("push", "-q", "origin", f"v{ver}")

    print("3/4 建 Release")
    tok = token()
    body = (f"## 更新内容\n\n{notes}\n\n## 怎么更新\n\n已经装了的：打开 TTS Studio，左下角会提示「发现新版本」，点「更新」自动装好并重启，"
            f"音色、项目和模型都会保留。\n\n第一次装：下载下面的 **{exe.name}**，双击安装（Windows 10/11 64 位）。")
    rel = api("POST", f"https://api.github.com/repos/{REPO}/releases", tok,
              json.dumps({"tag_name": f"v{ver}", "name": f"TTS Studio {ver}", "body": body}).encode())

    print(f"4/4 上传安装包（{exe.stat().st_size / 1048576:.0f} MB，可能要几分钟）")
    api("POST", f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets?name={exe.name}", tok,
        exe.read_bytes(), "application/octet-stream")
    print(f"完成：{rel['html_url']}")


if __name__ == "__main__":
    main()
