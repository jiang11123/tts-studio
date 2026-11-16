"""TTS Studio 的模型层：本地模型（Qwen3-TTS、Breeze TTS 2）+ 配音 API（MiniMax、火山/豆包、OpenAI 兼容）。

每个音色带着模型走：音色库里的 voice.json 有 "engine" 字段（没有 = breeze）。
所有模型都实现 synth(text, voice, tone, cfg, seed) -> (float32 波形, 采样率)。
本地模型共用一块显卡：同一时刻只加载一个，换模型时先卸载上一个。
本地模型不预装，在「设置」里点了才从魔搭（modelscope.cn）下载。
用户数据（设置、音色、项目、模型）都在 data/ 下，不进仓库。
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONF_DIR = DATA / "config"
CONF_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_JSON = CONF_DIR / "settings.json"
OUT_SR = 24000  # 所有句子统一存成 24k，合并时不会因为模型不同而变调

DEFAULT_SETTINGS = {
    "model_dir": str(DATA / "models"),
    "preload": "",
    "minimax": {"api_key": "", "base_url": "https://api.minimaxi.com", "model": "speech-02-hd"},
    "volc": {"appid": "", "token": "", "cluster": "volcano_tts"},
    "openai": {"base_url": "", "api_key": "", "model": "tts-1"},
}


def settings_load() -> dict:
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        saved = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        for k, v in saved.items():
            if isinstance(v, dict) and isinstance(s.get(k), dict):
                s[k].update(v)
            else:
                s[k] = v
    except Exception:
        pass
    return s


def settings_save(s: dict):
    SETTINGS_JSON.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def to_out_sr(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == OUT_SR:
        return audio
    import soxr
    return soxr.resample(audio, sr, OUT_SR).astype(np.float32)


def ffmpeg_exe() -> str | None:
    """系统里装了 ffmpeg 就用系统的，没有就用 imageio-ffmpeg 自带的。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """API 返回的 wav/mp3 字节 → 波形。soundfile 读不了就交给 ffmpeg。"""
    try:
        a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return a.mean(axis=1), sr
    except Exception:
        exe = ffmpeg_exe()
        if not exe:
            raise gr.Error("接口返回的音频需要 ffmpeg 才能解码：pip install imageio-ffmpeg")
        p = subprocess.run([exe, "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(OUT_SR), "pipe:1"],
                           input=data, capture_output=True)
        if p.returncode or not p.stdout:
            raise gr.Error("接口返回的音频解不开：" + p.stderr.decode("utf-8", "ignore")[:200])
        return np.frombuffer(p.stdout, dtype=np.float32), OUT_SR


# ───────────────────────── 魔搭模型目录 + 下载 ─────────────────────────
CATALOG = [
    {"id": "qwen3-1.7b-base", "engine": "qwen3", "type": "base", "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
     "name": "Qwen3-TTS 1.7B 克隆版", "size": "4.5 GB", "desc": "上传一段参考音频克隆声音（效果更好）"},
    {"id": "qwen3-0.6b-base", "engine": "qwen3", "type": "base", "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
     "name": "Qwen3-TTS 0.6B 克隆版", "size": "2.5 GB", "desc": "克隆声音，小一号，更快更省显存"},
    {"id": "qwen3-1.7b-custom", "engine": "qwen3", "type": "custom_voice", "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
     "name": "Qwen3-TTS 1.7B 预设音色", "size": "4.5 GB", "desc": "9 个自带音色，可以用语气提示词控制"},
    {"id": "qwen3-0.6b-custom", "engine": "qwen3", "type": "custom_voice", "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
     "name": "Qwen3-TTS 0.6B 预设音色", "size": "2.5 GB", "desc": "9 个自带音色，小一号"},
    {"id": "qwen3-1.7b-design", "engine": "qwen3", "type": "voice_design", "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
     "name": "Qwen3-TTS 1.7B 描述造音色", "size": "4.5 GB", "desc": "用一句话描述想要的声音"},
    {"id": "breeze-tts-2", "engine": "breeze", "type": "breeze", "repo": "BreezeBlue/Breeze-TTS-2",
     "name": "Breeze TTS 2", "size": "7.2 GB",
     "desc": "克隆 + 语气提示词，中文自然；显存约 8 GB；许可证仅限非商业用途（生成的音频也算）"},
]
CAT = {c["id"]: c for c in CATALOG}
MS = "https://modelscope.cn"

_dl = {}  # 模型 id → {"state": downloading/done/error/cancel, "done": 字节, "total": 字节, "msg": ...}
_dl_lock = threading.Lock()


def model_path(mid: str) -> Path:
    return Path(settings_load()["model_dir"]) / CAT[mid]["repo"].split("/")[-1]


def model_installed(mid: str) -> bool:
    """config.json 在、并且没有「下载中」标记 = 装好了。已有模型的话把文件夹放（或链接）到 data/models/ 下同名即可。"""
    p = model_path(mid)
    return (p / "config.json").is_file() and not (p / ".downloading").exists()


def _dl_worker(mid: str):
    st = _dl[mid]
    repo = CAT[mid]["repo"]
    dst = model_path(mid)
    try:
        r = requests.get(f"{MS}/api/v1/models/{repo}/repo/files", params={"Revision": "master", "Recursive": "true"},
                         timeout=30)
        r.raise_for_status()
        files = [f for f in r.json()["Data"]["Files"] if f.get("Type") != "tree"]
        st["total"] = sum(f.get("Size", 0) for f in files)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / ".downloading").write_text(repo, encoding="utf-8")
        for f in files:
            out = dst / f["Path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            size = f.get("Size", 0)
            if out.is_file() and out.stat().st_size == size:  # 断点续传：已下完的文件跳过
                st["done"] += size
                continue
            st["msg"] = f["Path"]
            part = out.with_name(out.name + ".part")
            have = part.stat().st_size if part.is_file() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(f"{MS}/models/{repo}/resolve/master/{f['Path']}", headers=headers, stream=True,
                              timeout=60) as resp:
                resp.raise_for_status()
                if have and resp.status_code != 206:
                    have = 0
                st["done"] += have
                with open(part, "ab" if have else "wb") as fp:
                    for chunk in resp.iter_content(1 << 20):
                        if st["state"] == "cancel":
                            return
                        fp.write(chunk)
                        st["done"] += len(chunk)
            part.replace(out)
        (dst / ".downloading").unlink(missing_ok=True)
        st["state"] = "done"
    except Exception as e:
        if st["state"] != "cancel":
            st["state"] = "error"
            st["msg"] = str(e)[:200]


def download_start(mid: str) -> str:
    with _dl_lock:
        if _dl.get(mid, {}).get("state") == "downloading":
            return "已经在下载了"
        _dl[mid] = {"state": "downloading", "done": 0, "total": 0, "msg": "连接魔搭…"}
    threading.Thread(target=_dl_worker, args=(mid,), daemon=True).start()
    return f"开始下载「{CAT[mid]['name']}」，可以关掉这页，下载在后台继续（断了再点下载会接着下）。"


def download_cancel(mid: str) -> str:
    if _dl.get(mid, {}).get("state") == "downloading":
        _dl[mid]["state"] = "cancel"
        return "已暂停，下次点下载会接着下。"
    return "没有在下载。"


def _is_link(p: Path) -> bool:
    """符号链接或 Windows 目录联接（junction）：真实位置和自己不一样。"""
    try:
        return p.is_symlink() or os.path.realpath(p).lower() != os.path.abspath(p).lower()
    except OSError:
        return False


def model_delete(mid: str) -> str:
    if _dl.get(mid, {}).get("state") == "downloading":
        raise gr.Error("正在下载，先点「暂停」再删除")
    unload_all()
    p = model_path(mid)
    if _is_link(p):  # 用户自己链接进来的模型：只拆链接，绝不删真实文件
        target = os.path.realpath(p)
        try:
            p.unlink()
        except OSError:
            os.rmdir(p)  # junction 用 rmdir 拆，不会动里面的文件
        _dl.pop(mid, None)
        return f"已移除「{CAT[mid]['name']}」的链接，原来的模型文件没动（{target}）。"
    if p.is_dir():
        shutil.rmtree(p)
    _dl.pop(mid, None)
    return f"已删除「{CAT[mid]['name']}」的文件（{p}）。"


def model_status(mid: str) -> str:
    st = _dl.get(mid)
    if st and st["state"] == "downloading":
        if not st["total"]:
            return "下载中…"
        return f"下载中 {st['done'] / st['total'] * 100:.0f}%（{st['done'] / 1e9:.2f}/{st['total'] / 1e9:.2f} GB）"
    if model_installed(mid):
        return "已下载"
    if st and st["state"] == "error":
        return "下载出错：" + st["msg"]
    if model_path(mid).is_dir():
        return "没下完（点下载接着下）"
    return "未下载"


def any_downloading() -> bool:
    return any(s["state"] == "downloading" for s in _dl.values())


# ───────────────────────── 模型基类 ─────────────────────────
class Engine:
    id = ""
    name = ""
    kind = "api"  # local / api
    voice_kinds: tuple = ()  # clone / design / speaker
    tone_ok = True  # 能不能用语气提示词

    def ready(self) -> tuple[bool, str]:
        return True, ""

    def speakers(self) -> list[str]:
        return []

    def synth(self, text, voice: dict, tone: str, cfg: float, seed: int) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def unload(self):
        pass


_gpu_lock = threading.Lock()  # 本地模型同一时刻只跑一个任务
_loaded: dict = {"engine": None}


def _use_gpu(engine: "Engine"):
    """要用某个本地模型前调用：别的本地模型占着显卡就先卸掉。"""
    cur = _loaded["engine"]
    if cur is not None and cur is not engine:
        cur.unload()
    _loaded["engine"] = engine


def unload_all():
    with _gpu_lock:
        if _loaded["engine"] is not None:
            _loaded["engine"].unload()
            _loaded["engine"] = None


def _free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _mem_gb() -> tuple[float, float]:
    """(可用物理内存 GB, 物理内存总量 GB)。

    注意：物理内存不是加载大模型的真正门槛。真正会拦住加载的是"提交内存"
    （物理 + 页面文件，见 _commit_gb）。加载 6.5 GB 权重时，如果提交额度不够，
    Windows 抛的是 `OSError: 页面文件太小，无法完成操作。(os error 1455)`；
    更糟的情况是它不抛异常，而是在访问 mmap 页时直接段错误（0xc0000005），
    把整个网页服务带走。
    """
    try:
        return _memstatus()[0], _memstatus()[1]
    except Exception:
        return (999.0, 999.0)


def _commit_gb() -> tuple[float, float]:
    """(可用提交内存 GB, 提交上限 GB)。加载模型真正要看的就是这个数。"""
    try:
        return _memstatus()[2], _memstatus()[3]
    except Exception:
        return (999.0, 999.0)


def _memstatus() -> tuple[float, float, float, float]:
    """(物理可用, 物理总量, 提交可用, 提交上限)，单位 GB。"""
    import ctypes

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MS()
    st.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    GIB = 1024 ** 3
    # ullTotalPageFile / ullAvailPageFile 是"提交上限/剩余提交额度"，不是页面文件本身的大小
    return (st.ullAvailPhys / GIB, st.ullTotalPhys / GIB,
            st.ullAvailPageFile / GIB, st.ullTotalPageFile / GIB)


def _vram_gb() -> tuple[float, float] | None:
    """(显存空闲 GB, 显存总量 GB)；没有 CUDA 就返回 None。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        GIB = 1024 ** 3
        return free / GIB, total / GIB
    except Exception:
        return None


def _require_headroom(need_commit: float, need_vram: float, what: str) -> None:
    """加载前检查提交内存和显存，不够就报错，而不是让进程崩掉。"""
    avail_c, cap_c = _commit_gb()
    avail_p, cap_p = _mem_gb()
    if avail_c < need_commit:
        raise RuntimeError(
            f"可用提交内存不够，装不下 {what}：现在只剩 {avail_c:.1f} GB（上限 {cap_c:.1f} GB，"
            f"物理可用 {avail_p:.1f}/{cap_p:.1f} GB），加载大约需要 {need_commit:.1f} GB。\n"
            "这跟物理内存不一定相关 —— 是 Windows 的「提交内存/页面文件」额度被占满了。\n"
            "两个办法：1) 关掉 Chrome 多余标签页、Claude、QQ、WPS 这些程序，腾出提交额度，马上就能再试；\n"
            "2) 永久解决：系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存，把页面文件调到固定 32768 MB，重启一次。"
        )
    # 显存只拦明显装不下的情况（空闲不到所需的六成）。差一点点时照旧先加载 ——
    # 旧版不检查，空闲 6.6 GB 也能正常装下 Breeze；真装不下由 load 里的 OOM 兜底报错。
    v = _vram_gb()
    if v is not None and v[0] < need_vram * 0.6:
        raise RuntimeError(
            f"显存不够，装不下 {what}：现在空闲 {v[0]:.1f} GB（共 {v[1]:.1f} GB），大约需要 {need_vram:.1f} GB。\n"
            "把浏览器/剪辑软件关掉腾出显存，或者在「设置 → 显卡」里先点一下释放。"
        )


# ───────────────────────── Breeze TTS 2（本地） ─────────────────────────
def _breeze_path():
    """Breeze 官方推理代码（Apache 2.0）放在 third_party/breeze，用到时才加进 import 路径。"""
    p = str(ROOT / "third_party" / "breeze")
    if p not in sys.path:
        sys.path.insert(0, p)


_ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64",
              "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def _stream_state_dict(ckpt_dir: Path, device: str, dtype) -> dict:
    """不用 mmap，逐个张量读 safetensors 直接送显卡：额外提交内存只占当前这一个张量。"""
    import torch
    sd = {}
    for f in sorted(Path(ckpt_dir).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
            base = 8 + n
            items = sorted(((k, v) for k, v in hdr.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            for name, info in items:
                s, e = info["data_offsets"]
                dt = getattr(torch, _ST_DTYPES[info["dtype"]])
                if e == s:
                    t = torch.empty(info["shape"], dtype=dt)
                else:
                    fh.seek(base + s)
                    buf = bytearray(e - s)
                    fh.readinto(buf)
                    t = torch.frombuffer(buf, dtype=dt).reshape(info["shape"])
                if t.is_floating_point():
                    t = t.to(dtype)
                sd[name] = t.to(device)
                del t
    return sd


def _load_breeze_streaming(ckpt_dir: Path, device: str):
    """等同 breeze_infer.runtime.load_runtime（显卡版），只是主模型权重不走 mmap。"""
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from models.breeze import BreezeForConditionalGeneration
    from qwen_tts import Qwen3TTSTokenizer
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(ckpt_dir, fix_mistral_regex=False)
    cfg = BreezeForConditionalGeneration.config_class.from_pretrained(ckpt_dir)
    gen_cfg = GenerationConfig.from_pretrained(ckpt_dir)
    sd = _stream_state_dict(ckpt_dir, device, torch.bfloat16)
    # state_dict 已经在显卡上，transformers 直接拿来当参数，不会再复制一份
    model = BreezeForConditionalGeneration.from_pretrained(
        None, config=cfg, state_dict=sd, generation_config=gen_cfg, dtype=torch.bfloat16,
        attn_implementation="eager", device_map=device)
    del sd
    model.eval()
    atok_dir = ckpt_dir / "audio_tokenizer"
    if not atok_dir.is_dir():
        raise FileNotFoundError(f"Breeze 模型包里缺 audio_tokenizer：{atok_dir}")
    atok = Qwen3TTSTokenizer.from_pretrained(str(atok_dir), device_map=device)
    return tok, model, atok


def _make_cpu_embedding_cls():
    import torch

    class _CpuEmbedding(torch.nn.Module):
        """词表放内存：查表在 CPU 上做，结果同步拷回原来的显卡（不用 non_blocking、不用 hook）。"""

        def __init__(self, emb):
            super().__init__()
            self.dev = emb.weight.device
            self.num_embeddings, self.embedding_dim = emb.num_embeddings, emb.embedding_dim
            self.padding_idx = emb.padding_idx
            self.weight = torch.nn.Parameter(emb.weight.detach().to("cpu").clone(), requires_grad=False)

        def forward(self, input_ids):
            out = torch.nn.functional.embedding(input_ids.detach().to("cpu"), self.weight, self.padding_idx)
            return out.to(self.dev)

    return _CpuEmbedding


def _CpuEmbedding(emb):
    global _CPU_EMB_CLS
    try:
        cls = _CPU_EMB_CLS
    except NameError:
        cls = _CPU_EMB_CLS = _make_cpu_embedding_cls()
    return cls(emb)


def _embed_to_cpu(model) -> None:
    """把 1 GB 的文字词表 embed_text_tokens 放回内存，省出 1 GB 显存。

    11-14 实测（4060 8G）：模型 7 GB + 桌面/资源管理器占的 0.85 GB 把显存塞满，
    多出的 0.9 GB 被驱动挪到「共享显存」（走 PCIe 读内存），一句话要生成好几分钟。
    这张表只在每句开头查一次（不在逐帧的 CUDA Graph 里），放 CPU 上查完再搬回显卡，几乎不费时间。
    文本编码器自己的词表在它的 CUDA Graph 里用，不能动。"""
    import torch
    emb = getattr(model, "embed_text_tokens", None)
    if not isinstance(emb, torch.nn.Embedding) or emb.weight.device.type != "cuda":
        return
    model.embed_text_tokens = _CpuEmbedding(emb)
    _free_gpu()


class BreezeEngine(Engine):
    id, name, kind = "breeze", "Breeze TTS 2（本地）", "local"
    voice_kinds = ("clone", "design")

    def __init__(self):
        self.m = None  # (tokenizer, model, audio_tokenizer, runtimes, sr)

    def ready(self):
        if model_installed("breeze-tts-2"):
            return True, ""
        return False, "还没下载 Breeze TTS 2：去「设置 → 本地模型」点下载"

    def load(self):
        if self.m:
            return
        _breeze_path()
        # 先量资源再动手：提交内存不够时，transformers 读分片会抛 1455
        # （"页面文件太小"），或者更糟 —— 在访问 mmap 页时段错误，把整个网页带走。
        _d = Path(model_path("breeze-tts-2"))
        _size = sum(f.stat().st_size for f in _d.glob("*.safetensors")) / 1024 ** 3
        _atok = _d / "audio_tokenizer" / "model.safetensors"  # glob 不含子目录，单独算
        if _atok.is_file():
            _size += _atok.stat().st_size / 1024 ** 3
        # 11-13 实测（4060 8G）：官方 load_runtime 用 safetensors 的 mmap（写时复制映射），
        # Windows 会把整个 5 GB 分片一次性记进提交额度，「系统管理大小」的页面文件来不及长大，
        # 进程直接段错误 / 被杀（网页显示「服务器丢失」）。有显卡时改走 _load_breeze_streaming：
        # 逐个张量读文件直接送显卡，提交额度一点点涨，页面文件跟得上。显存分配本身也占提交额度，
        # 所以加载完进程大约 10 GB，这部分省不掉。纯 CPU 仍走官方加载、按全量检查。
        try:
            import torch
            _gpu = torch.cuda.is_available()
        except Exception:
            _gpu = False
        _require_headroom(2.5 if _gpu else _size + 1.0, _size * 0.95, "Breeze TTS 2")
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        print("loading Breeze ...", flush=True)
        try:
            dev = resolve_device()
            if dev.startswith("cuda"):
                tok, model, atok = _load_breeze_streaming(Path(model_path("breeze-tts-2")), dev)
            else:
                tok, model, atok = load_runtime(model_path("breeze-tts-2"), device=dev,
                                                attn_implementation="eager")
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            _free_gpu()
            v = _vram_gb()
            raise RuntimeError(
                "显存不够，Breeze TTS 2 没装下"
                + (f"（加载前空闲 {v[0]:.1f} GB，共 {v[1]:.1f} GB）" if v else "")
                + "。把剪映/浏览器等占显卡的软件关掉再试。") from e
        update_generation_config_for_breeze(model)
        _embed_to_cpu(model)

        def new_rt():
            return FastBreezeStreamingRuntime(
                model, atok, FastStreamingConfig(max_new_tokens=1500, max_seq_len=2048, repetition_penalty=1.1),
                tokenizer=tok)

        rts = {False: new_rt()}
        self.m = {"tok": tok, "model": model, "atok": atok, "rts": rts, "new_rt": new_rt, "sr": rts[False].sample_rate}
        print("Breeze ready", flush=True)

    def unload(self):
        if self.m:
            self.m = None
            _free_gpu()
            print("Breeze unloaded", flush=True)

    def synth(self, text, voice, tone, cfg, seed):
        _breeze_path()
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name
        ref_audio, ref_text = voice.get("ref_audio"), voice.get("ref_text", "")
        instruction = (voice.get("instruction") or "").strip()
        tone = (tone or "").strip()
        cfg = float(cfg)
        if tone:
            instruction = (tone + instruction).strip()
            if cfg <= 1.0:
                cfg = 2.0  # 提示词要 cfg>1 才生效；实测 cfg≥2.5 音色容易跑偏
        request = {"id": "ui-request", "text": text, "speaker": "S0"}
        if ref_audio:
            if not (ref_text or "").strip():
                raise gr.Error("用参考音频克隆时，必须填写它的准确文字稿")
            request["ref_audio_path"] = str(ref_audio)
            request["ref_text"] = ref_text.strip()
            if instruction:
                request["instruction"] = instruction  # 声音方向模式
            else:
                cfg = 1.0  # 纯克隆不支持 cfg>1
        elif instruction:
            request["instruction"] = instruction
        else:
            raise gr.Error("Breeze 音色需要参考音频（克隆）或一句音色描述")
        _use_gpu(self)
        self.load()
        # 11-13 实测：模型装好后生成一句，进程提交内存还要再涨约 3 GB（10 → 12.9 GB）。
        # 额度不够时 Windows 直接杀进程（网页「Connection lost」），所以生成前先看一眼，宁可报错别崩。
        _avail, _cap = _commit_gb()
        if _avail < 3.5:
            raise RuntimeError(
                f"虚拟内存快用完了（提交额度只剩 {_avail:.1f} GB / 上限 {_cap:.1f} GB），这句生成到一半会被 Windows 强制结束。\n"
                "先关掉 Chrome 多余标签页、微信、QQ、WPS 再点；一劳永逸：运行软件文件夹里的「设置虚拟内存.bat」并重启电脑。")
        m = self.m
        rid = f"ui-{uuid.uuid4().hex}"
        request["id"] = rid
        set_all_seeds(int(seed))
        inputs = prepare_inputs(m["tok"], m["atok"], m["model"], [request],
                                get_template(select_template_name(request)),
                                guidance_scale=cfg, guidance_scale_ref=None, guidance_scale_ins=None)
        use_cfg = cfg != 1.0
        if use_cfg not in m["rts"]:
            m["rts"][use_cfg] = m["new_rt"]()
        chunks = []
        gen = m["rts"][use_cfg].iter_audio_chunks(inputs, request_id=rid, seed=int(seed))
        try:
            for c in gen:
                chunks.append(np.asarray(c.audio, dtype=np.float32))
        finally:
            gen.close()
        _free_gpu()
        return (np.concatenate(chunks) if chunks else np.zeros(1, np.float32)), m["sr"]


# ───────────────────────── Qwen3-TTS（本地，魔搭下载） ─────────────────────────
QWEN_SPEAKERS = ["vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"]
QWEN_SPEAKER_TIPS = {"vivian": "女声", "serena": "女声", "uncle_fu": "大叔", "dylan": "北京话男声", "eric": "四川话男声",
                     "ryan": "英文男声", "aiden": "英文男声", "ono_anna": "日语女声", "sohee": "韩语女声"}
QWEN_TYPE = {"clone": "base", "speaker": "custom_voice", "design": "voice_design"}


class QwenEngine(Engine):
    id, name, kind = "qwen3", "Qwen3-TTS（本地）", "local"
    voice_kinds = ("clone", "speaker", "design")

    def __init__(self):
        self.m = None
        self.mid = None
        self._prompts = {}  # 参考音频 → 克隆提示（同一个音色只算一次）

    def pick_model(self, vkind: str, want: str | None = None) -> str | None:
        if want and want in CAT and model_installed(want):
            return want
        t = QWEN_TYPE[vkind]
        for c in CATALOG:  # 目录里 1.7B 排在前面，优先用大的
            if c["engine"] == "qwen3" and c["type"] == t and model_installed(c["id"]):
                return c["id"]
        return None

    def ready(self):
        if any(model_installed(c["id"]) for c in CATALOG if c["engine"] == "qwen3"):
            return True, ""
        return False, "还没下载 Qwen3-TTS 模型：去「设置 → 本地模型」点下载"

    def speakers(self):
        return QWEN_SPEAKERS

    def load(self, mid):
        if self.m and self.mid == mid:
            return
        self.unload()
        import torch
        from qwen_tts import Qwen3TTSModel
        print(f"loading {mid} ...", flush=True)
        kw = {"device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
              "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        try:
            self.m = Qwen3TTSModel.from_pretrained(str(model_path(mid)), attn_implementation="sdpa", **kw)
        except Exception:
            self.m = Qwen3TTSModel.from_pretrained(str(model_path(mid)), **kw)
        self.mid = mid
        print(f"{mid} ready", flush=True)

    def unload(self):
        if self.m:
            self.m = None
            self.mid = None
            self._prompts.clear()
            _free_gpu()
            print("Qwen3 unloaded", flush=True)

    def synth(self, text, voice, tone, cfg, seed):
        import torch
        lang = qwen_lang(voice.get("language"))
        vkind = voice.get("kind") or ("clone" if voice.get("ref_audio") else "speaker")
        mid = self.pick_model(vkind, voice.get("model"))
        if not mid:
            need = {"clone": "克隆版", "speaker": "预设音色", "design": "描述造音色"}[vkind]
            raise gr.Error(f"这个音色要用 Qwen3-TTS「{need}」模型，还没下载：去「设置 → 本地模型」下载")
        _use_gpu(self)
        self.load(mid)
        torch.manual_seed(int(seed))
        tone = (tone or "").strip()
        if vkind == "clone":
            ref, rt = voice.get("ref_audio"), (voice.get("ref_text") or "").strip()
            if not ref:
                raise gr.Error("克隆音色缺参考音频")
            key = (ref, rt)
            if key not in self._prompts:
                self._prompts[key] = self.m.create_voice_clone_prompt(ref_audio=ref, ref_text=rt or None,
                                                                      x_vector_only_mode=not rt)
            wavs, sr = self.m.generate_voice_clone(text=text, language=lang, voice_clone_prompt=self._prompts[key])
        elif vkind == "speaker":
            wavs, sr = self.m.generate_custom_voice(text=text, speaker=voice.get("speaker") or "vivian",
                                                    language=lang, instruct=tone or None)
        else:
            ins = ((voice.get("instruction") or "") + tone).strip()
            if not ins:
                raise gr.Error("描述造音色要写一句音色描述")
            wavs, sr = self.m.generate_voice_design(text=text, instruct=ins, language=lang)
        _free_gpu()
        return np.asarray(wavs[0], dtype=np.float32), sr


# ───────────────────────── 语言 ─────────────────────────
# 文本朗读页的语言下拉框：显示名 → 内部代号。Qwen3 和 MiniMax 会按这个读；Breeze、豆包、OpenAI 兼容接口自己识别
LANGS = [("自动检测", "auto"), ("中文", "Chinese"), ("粤语", "Cantonese"), ("英语", "English"), ("日语", "Japanese"),
         ("韩语", "Korean"), ("法语", "French"), ("德语", "German"), ("西班牙语", "Spanish"),
         ("葡萄牙语", "Portuguese"), ("俄语", "Russian"), ("意大利语", "Italian")]
LANG_ENGINES = ("qwen3", "minimax")  # 会按选的语言读的模型
MINIMAX_LANG = {"auto": "auto", "Chinese": "Chinese", "Cantonese": "Chinese,Yue", "English": "English",
                "Japanese": "Japanese", "Korean": "Korean", "French": "French", "German": "German",
                "Spanish": "Spanish", "Portuguese": "Portuguese", "Russian": "Russian", "Italian": "Italian"}


def qwen_lang(lang):
    """Qwen3-TTS 的 language 参数。没指定 = 中文（和以前一样）；粤语 Qwen 不单独支持，按中文读。"""
    if not lang:
        return "Chinese"
    return {"auto": "Auto", "Cantonese": "Chinese"}.get(lang, lang)


# ───────────────────────── MiniMax 语音 API ─────────────────────────
MINIMAX_SPEAKERS = ["male-qn-qingse", "male-qn-jingying", "male-qn-badao", "male-qn-daxuesheng", "female-shaonv",
                    "female-yujie", "female-chengshu", "female-tianmei", "presenter_male", "presenter_female",
                    "audiobook_male_1", "audiobook_female_1"]


def _api_err(name, r):
    raise gr.Error(f"{name} 接口报错（HTTP {r.status_code}）：{r.text[:300]}")


class MiniMaxEngine(Engine):
    id, name = "minimax", "MiniMax 语音（API）"
    voice_kinds = ("speaker",)
    tone_ok = False

    def ready(self):
        s = settings_load()["minimax"]
        return (bool(s["api_key"]), "" if s["api_key"] else "还没填 MiniMax 的 API Key：去「设置 → 接入第三方配音 API」")

    def speakers(self):
        return MINIMAX_SPEAKERS

    def synth(self, text, voice, tone, cfg, seed):
        s = settings_load()["minimax"]
        ok, msg = self.ready()
        if not ok:
            raise gr.Error(msg)
        body = {"model": s["model"] or "speech-02-hd", "text": text, "stream": False,
                "voice_setting": {"voice_id": voice.get("speaker") or "male-qn-qingse", "speed": 1.0, "vol": 1.0,
                                  "pitch": 0},
                "audio_setting": {"sample_rate": 32000, "format": "wav", "channel": 1}}
        if voice.get("language"):
            body["language_boost"] = MINIMAX_LANG.get(voice["language"], "auto")
        r = requests.post(s["base_url"].rstrip("/") + "/v1/t2a_v2", json=body, timeout=120,
                          headers={"Authorization": "Bearer " + s["api_key"].strip()})
        if r.status_code != 200:
            _api_err("MiniMax", r)
        d = r.json()
        br = d.get("base_resp") or {}
        if br.get("status_code", 0) != 0 or not (d.get("data") or {}).get("audio"):
            raise gr.Error(f"MiniMax 返回错误：{br.get('status_msg') or str(d)[:300]}")
        return decode_audio(bytes.fromhex(d["data"]["audio"]))


# ───────────────────────── 火山引擎 / 豆包语音 API ─────────────────────────
VOLC_SPEAKERS = ["BV700_streaming", "BV001_streaming", "BV002_streaming", "BV406_streaming", "BV407_streaming"]


class VolcEngine(Engine):
    id, name = "volc", "豆包 / 火山引擎（API）"
    voice_kinds = ("speaker",)
    tone_ok = False

    def ready(self):
        s = settings_load()["volc"]
        ok = bool(s["appid"] and s["token"])
        return ok, "" if ok else "还没填火山引擎的 AppID 和 Access Token：去「设置 → 接入第三方配音 API」"

    def speakers(self):
        return VOLC_SPEAKERS

    def synth(self, text, voice, tone, cfg, seed):
        s = settings_load()["volc"]
        ok, msg = self.ready()
        if not ok:
            raise gr.Error(msg)
        body = {"app": {"appid": s["appid"].strip(), "token": s["token"].strip(), "cluster": s["cluster"] or "volcano_tts"},
                "user": {"uid": "tts-studio"},
                "audio": {"voice_type": voice.get("speaker") or "BV700_streaming", "encoding": "wav", "speed_ratio": 1.0},
                "request": {"reqid": uuid.uuid4().hex, "text": text, "operation": "query"}}
        r = requests.post("https://openspeech.bytedance.com/api/v1/tts", json=body, timeout=120,
                          headers={"Authorization": "Bearer;" + s["token"].strip()})
        if r.status_code != 200:
            _api_err("火山引擎", r)
        d = r.json()
        if d.get("code") != 3000 or not d.get("data"):
            raise gr.Error(f"火山引擎返回错误：{d.get('message') or str(d)[:300]}")
        return decode_audio(base64.b64decode(d["data"]))


# ───────────────────────── OpenAI 兼容接口 ─────────────────────────
OPENAI_SPEAKERS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class OpenAIEngine(Engine):
    id, name = "openai", "OpenAI 兼容接口（API）"
    voice_kinds = ("speaker",)

    def ready(self):
        s = settings_load()["openai"]
        ok = bool(s["base_url"] and s["api_key"])
        return ok, "" if ok else "还没填 OpenAI 兼容接口的地址和 Key：去「设置 → 接入第三方配音 API」"

    def speakers(self):
        return OPENAI_SPEAKERS

    def synth(self, text, voice, tone, cfg, seed):
        s = settings_load()["openai"]
        ok, msg = self.ready()
        if not ok:
            raise gr.Error(msg)
        body = {"model": s["model"] or "tts-1", "input": text, "voice": voice.get("speaker") or "alloy",
                "response_format": "wav"}
        if (tone or "").strip():
            body["instructions"] = tone.strip()  # 支持的模型（如 gpt-4o-mini-tts）会照语气读，不支持的一般会忽略
        r = requests.post(s["base_url"].rstrip("/") + "/audio/speech", json=body, timeout=120,
                          headers={"Authorization": "Bearer " + s["api_key"].strip()})
        if r.status_code != 200:
            _api_err("OpenAI 兼容接口", r)
        return decode_audio(r.content)


ENGINES: dict[str, Engine] = {e.id: e for e in (QwenEngine(), BreezeEngine(), MiniMaxEngine(), VolcEngine(), OpenAIEngine())}
VOICE_KIND_NAMES = {"clone": "克隆（参考音频）", "speaker": "预设音色 / 音色 ID", "design": "文字描述造音色"}


def engine_of(voice: dict) -> Engine:
    return ENGINES.get(voice.get("engine") or "breeze", ENGINES["breeze"])


def synth(text: str, voice: dict, tone: str = "", cfg: float = 2.0, seed: int = 42) -> np.ndarray:
    """按音色所属的模型读一句，返回 24k 波形。"""
    text = (text or "").strip()
    if not text:
        raise gr.Error("文字是空的")
    eng = engine_of(voice)
    ok, msg = eng.ready()
    if not ok:
        raise gr.Error(msg)
    if not eng.tone_ok:
        tone = ""
    if eng.kind == "local":
        with _gpu_lock:
            audio, sr = eng.synth(text, voice, tone, cfg, seed)
    else:
        audio, sr = eng.synth(text, voice, tone, cfg, seed)
    return to_out_sr(audio, sr)


def preload(voice: dict | None = None):
    """设置里开了「启动时预加载」就在后台先把本地模型加载好：开软件时内存最空，第一次生成也不用等。
    默认不开，免得只用 API 的人白占显卡。
    "auto"（现在的开关）= 跟上次项目的音色走：Breeze 音色预载 Breeze，Qwen3 音色预载对应的 Qwen3 模型，
    API 音色什么都不载；没有上次项目时按 Breeze。旧设置里的 "breeze" 当成 auto。"""
    if settings_load().get("preload") not in ("auto", "breeze"):
        return
    eng = engine_of(voice) if voice else ENGINES["breeze"]
    if eng.kind != "local" or not eng.ready()[0]:
        return
    mid = None
    if eng.id == "qwen3":
        vkind = voice.get("kind") or ("clone" if voice.get("ref_audio") else "speaker")
        mid = eng.pick_model(vkind, voice.get("model"))
        if not mid:
            return

    def run():
        with _gpu_lock:
            if _loaded["engine"] is None:
                _use_gpu(eng)
                try:
                    eng.load(mid) if mid else eng.load()
                except Exception as e:
                    print(f"预加载 {eng.name} 失败：", e, flush=True)
                    eng.unload()
                    _loaded["engine"] = None
    threading.Thread(target=run, daemon=True).start()


def gpu_state() -> str:
    e = _loaded["engine"]
    if e is None or not getattr(e, "m", None):
        return "显卡空闲（第一次生成时自动加载模型）"
    extra = f" · {CAT[e.mid]['name']}" if getattr(e, "mid", None) else ""
    return f"显卡上：{e.name}{extra}"
