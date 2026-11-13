"""TTS Studio：多个配音模型（本地 Qwen3-TTS / Breeze，MiniMax / 火山 / OpenAI 兼容 API）共用一套界面。
音色设计 + 整稿分句生成（可逐句重做、逐句换音色），模型在「设置」里下载和配置。"""

from __future__ import annotations

import hashlib
import html
import json
import re
import os
import shutil
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

import api_server
import mcp_server
import updater
import engines as E
from speed import change_pitch, change_speed, change_volume
from splitter import group_sentences, split_sentences

ROOT = Path(__file__).resolve().parent
DATA = E.DATA  # 用户自己的东西都在 data/ 下（不进仓库）
OUT_DIR = DATA / "outputs"
VOICE_DIR = DATA / "voices"
PROJ_DIR = DATA / "projects"
for d in (OUT_DIR, VOICE_DIR, PROJ_DIR):
    d.mkdir(parents=True, exist_ok=True)

SR = E.OUT_SR
E.preload()

# ───────────────────────── 语气库 ─────────────────────────
# 语气和音色分开：音色 = 谁在说（参考音频），语气 = 怎么说（一句提示词）。
# 任何音色都能配任何语气；提示词可以自己改，改好起个名字存进语气库。
# 纯克隆会把参考音频里的语气一起学过来；选了语气就改用「声音方向」模式，用提示词盖过去。
TONE_JSON = VOICE_DIR / "语气.json"
TONE_NONE = "原样（跟参考音频走）"
DEFAULT_TONES = {
    "平稳": "用平稳、自然的语气朗读，语调平直，不带情绪起伏，不夸张、不笑、不感叹，像平静地讲解。",
    "激昂": "语气激昂、充满热情和力量，语速稍快，重音明显，情绪饱满，像在给大家打气。",
    "亲切聊天": "语气亲切自然，像和朋友面对面聊天，轻松随和，带一点笑意。",
    "认真讲解": "语气认真清晰，语速适中，像老师在耐心讲解知识点，重点处稍微加重。",
    "悬念神秘": "声音稍微压低，语速放慢，带一点神秘感，像在讲一个悬念，吊人胃口。",
}


def tones_load() -> dict:
    try:
        return json.loads(TONE_JSON.read_text(encoding="utf-8"))
    except Exception:
        TONE_JSON.write_text(json.dumps(DEFAULT_TONES, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_TONES)


TONE_CUSTOM = "自定义"


def tone_choices():
    return [TONE_NONE, *tones_load(), TONE_CUSTOM]


def tone_match(text):
    """看提示词实际是哪个语气：空 = 原样；和语气库里某条一样 = 那一条；否则 = 自定义。"""
    text = (text or "").strip()
    if not text:
        return TONE_NONE
    for name, t in tones_load().items():
        if t.strip() == text:
            return name
    return TONE_CUSTOM


def tone_pick(name):
    """选语气 → 把它的提示词填进输入框（可以接着改）。"""
    if name == TONE_CUSTOM:
        return gr.skip()  # 点「自定义」不动提示词，接着在下面改
    return "" if not name or name == TONE_NONE else tones_load().get(name, "")


def tone_save(name, text):
    name = _safe(name)
    text = (text or "").strip()
    if not name or name in (TONE_NONE, TONE_CUSTOM):
        raise gr.Error("给这个语气起个名字（不能叫「原样」或「自定义」）")
    if not text:
        raise gr.Error("语气提示词是空的")
    t = tones_load()
    t[name] = text
    TONE_JSON.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
    return gr.update(choices=tone_choices(), value=name), f"已存进语气库：「{name}」。所有音色都能直接选它。"


def tone_delete(name):
    t = tones_load()
    if not name or name not in t:
        return gr.skip(), gr.skip(), "先点一个要删的语气再删（「原样」「自定义」不能删）。"
    t.pop(name)
    TONE_JSON.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
    return gr.update(choices=tone_choices(), value=TONE_NONE), "", f"已删除语气「{name}」。"


# ───────────────────────── 音色库 ─────────────────────────
# 每个音色一个文件夹：voices/库/<名字>/ref.wav + voice.json。整稿页从下拉框选，不再默认沿用上一次。
LIB_DIR = VOICE_DIR / "库"
LIB_DIR.mkdir(exist_ok=True)


def _safe(name: str) -> str:
    return "".join(ch for ch in (name or "").strip() if ch not in '\\/:*?"<>|')


def lib_list():
    return sorted(p.parent.name for p in LIB_DIR.glob("*/voice.json"))


def lib_read(name: str) -> dict:
    v = json.loads((LIB_DIR / name / "voice.json").read_text(encoding="utf-8"))
    if not v.get("ref_audio") or not Path(v["ref_audio"]).is_file():
        local = sorted((LIB_DIR / name).glob("ref.*")) if v.get("ref_audio") else []
        v["ref_audio"] = str(local[0]) if local else None
    return v


ENGINE_SHORT = {"breeze": "Breeze", "qwen3": "Qwen3", "minimax": "MiniMax", "volc": "豆包", "openai": "OpenAI"}


def lib_choices():
    """音色下拉框：显示「名字 · 模型」，值还是名字。"""
    out = []
    for n in lib_list():
        try:
            eng = lib_read(n).get("engine") or "breeze"
        except Exception:
            eng = "breeze"
        out.append((f"{n} · {ENGINE_SHORT.get(eng, eng)}", n))
    return out


def lib_write(name, ref_audio, ref_text, instruction, seed, engine="breeze", kind=None, speaker="") -> str:
    name = _safe(name)
    if not name:
        raise gr.Error("给音色起个名字")
    engine = engine or "breeze"
    kind = kind or ("clone" if ref_audio else "design")
    if kind == "speaker":
        if not (speaker or "").strip():
            raise gr.Error("填一个音色 ID（下拉框里选，或者粘贴接口后台里的音色 ID）")
        ref_audio, ref_text, instruction = None, "", ""
    elif kind == "clone":
        if not ref_audio:
            raise gr.Error("克隆音色要上传一段参考音频（5～15 秒最好）")
        if engine == "breeze" and not (ref_text or "").strip():
            raise gr.Error("参考音频必须配上它的准确文字稿")
    elif not (instruction or "").strip():
        raise gr.Error("写一句音色描述，比如：二十多岁的男生，声音清爽温和")
    d = LIB_DIR / name
    d.mkdir(exist_ok=True)
    if (d / "voice.json").is_file():  # 同名覆盖前留一份旧的
        bak = d / f"备份_{time.strftime('%m%d_%H%M%S')}"
        bak.mkdir()
        for f in d.iterdir():
            if f.is_file():
                shutil.copyfile(f, bak / f.name)
    saved = ""
    if ref_audio:
        src = Path(ref_audio)
        dst = d / f"ref{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        saved = str(dst)
    (d / "voice.json").write_text(
        json.dumps(
            {"engine": engine, "kind": kind, "speaker": (speaker or "").strip(),
             "ref_audio": saved, "ref_text": (ref_text or "").strip(), "instruction": (instruction or "").strip(),
             "seed": int(seed)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return name


def lib_save(name, ref_audio, ref_text, instruction, seed, engine, kind, speaker):
    kind = {v: k for k, v in E.VOICE_KIND_NAMES.items()}.get(kind, kind)
    name = lib_write(name, ref_audio, ref_text, instruction, seed, engine, kind, speaker)
    return (gr.update(choices=lib_choices(), value=name),
            f"已存进音色库：「{name}」（{ENGINE_SHORT.get(engine, engine)}）。文本朗读的音色下拉框里直接选。")


def new_voice_form(engine):
    """新建音色：选了模型，只显示这个模型支持的音色类型。"""
    eng = E.ENGINES[engine]
    kinds = [E.VOICE_KIND_NAMES[k] for k in eng.voice_kinds]
    ok, msg = eng.ready()
    return (gr.update(choices=kinds, value=kinds[0]),
            gr.update(choices=eng.speakers(), value=(eng.speakers() or [None])[0]),
            ("" if ok else f"⚠️ {msg}") + _engine_tip(engine))


def _engine_tip(engine):
    return {
        "breeze": "",
        "qwen3": "\n\nQwen3：克隆音色不吃语气提示词；预设音色可以用语气；预设里 dylan 是北京话、eric 是四川话。",
        "minimax": "\n\nMiniMax：下拉框是自带音色；在 MiniMax 后台克隆好的音色，把它的 voice_id 粘贴进来就行。",
        "volc": "\n\n火山：下拉框是几个常用音色；控制台里音色列表的「voice_type」粘贴进来就行。",
        "openai": "\n\nOpenAI 兼容：填接口支持的 voice 名字。",
    }.get(engine, "")


def new_voice_kind(kind):
    k = {v: kk for kk, v in E.VOICE_KIND_NAMES.items()}.get(kind, kind)
    return gr.update(visible=k == "clone"), gr.update(visible=k == "clone"), gr.update(visible=k == "design"), \
        gr.update(visible=k == "speaker")


def lib_pick(name):
    """选一个音色，把它的设置填进下面的参数里。"""
    if not name or name not in lib_list():  # 没选 = 清空，避免下拉框是空的、实际却还在用上一个音色
        return None, "", "", 42, "还没选音色：在上面的「音色」下拉框里选一个。"
    v = lib_read(name)
    return (v.get("ref_audio") or None, v.get("ref_text", ""), v.get("instruction", ""),
            v.get("seed", 42), f"已选用音色「{name}」。")


def lib_delete(name):
    if not name or not (LIB_DIR / name).is_dir():
        raise gr.Error("先在下拉框里选要删的音色")
    trash = VOICE_DIR / "已删除"
    trash.mkdir(exist_ok=True)
    shutil.move(str(LIB_DIR / name), str(trash / f"{name}_{time.strftime('%m%d_%H%M%S')}"))
    return gr.update(choices=lib_choices(), value=None), f"已删除「{name}」（移到了 voices/已删除，想找回可以去那里拿）。"


def voice_preview(name):
    """长稿页：选完音色直接显示它的原声，点一下就能听。"""
    if not name or name not in lib_list():
        return gr.update(value=None, visible=False)
    ref = lib_read(name).get("ref_audio") or None  # 预设/API 音色没有原声：给 None，给空字符串播放器会报错
    return gr.update(value=ref, visible=bool(ref), label="原声试听")


def refresh_libs():
    return lib_list()


# ───────────────────────── 音色设计（文字描述 → 3 个候选 → 选一个存进音色库） ─────────────────────────
DESIGN_DIR = OUT_DIR / "design"
DESIGN_DIR.mkdir(parents=True, exist_ok=True)
VD_MAX = 300  # 试听文本最多多少字
# 预设人设：名字、头像 emoji、头像底色（色相）、音色描述、配套试听文本
VD_PRESETS = [
    ("年迈智者", "🧙", 32, "一位年迈的智者，声音略带沙哑但充满智慧，语速平缓，讲述着人生的哲理。",
     "世间万物皆有其规律，年轻人，切勿急躁。静下心来，你会听到花开的声音，也会明白时间的重量。"),
    ("激情解说", "🎙️", 262, "一位充满激情的体育解说员，声音高亢有力，语速很快，情绪饱满，感染力极强。",
     "球进了！这脚射门简直不可思议，整个球场都沸腾了，这就是足球的魅力！"),
    ("悬疑故事家", "🕯️", 290, "一位擅长讲悬疑故事的男声，声音低沉，语速放慢，带着神秘感，像在讲一个吊人胃口的秘密。",
     "那天夜里，走廊尽头的门忽然自己开了。可屋里明明一个人都没有，地上却多了一串湿漉漉的脚印。"),
    ("摇篮曲女声", "🌙", 215, "一位温柔的年轻妈妈，声音轻柔甜美，语速很慢，像在床边哄孩子入睡。",
     "宝贝闭上眼睛，月亮已经爬上窗台，星星也在悄悄说晚安，我们一起做个甜甜的梦吧。"),
    ("严厉女教师", "📏", 345, "一位严厉的中年女教师，声音清亮干脆，语气严肃，不容置疑。",
     "上课铃已经响了，还有人在说话吗？把作业本拿出来放在桌上，我现在一个一个检查。"),
    ("知识讲解", "💡", 150, "二十多岁的男生，声音清爽温和，像在耐心地讲解知识，自然不做作，没有播音腔。",
     "今天我们来讲一个很实用的方法，你不用会写代码，只要按顺序做三步，就能把这件事交代清楚。"),
]
VD_EXTRA = [  # 只给「随机」用的，不占预设按钮
    ("元气少女", "一个元气满满的少女，声音清脆明亮，语速轻快，说话总带着笑意。",
     "早上好呀！今天天气超级好，我们一起去公园野餐吧，我带了好多好吃的哦！"),
    ("新闻主播", "一位专业的新闻男主播，声音浑厚标准，吐字清晰，语速平稳，庄重大方。",
     "各位观众晚上好，欢迎收看今天的新闻。下面请看详细报道。"),
    ("东北大哥", "一位豪爽的东北大哥，嗓门洪亮，带点东北口音，说话直爽幽默。",
     "哎呀妈呀，这事儿你早说啊！走走走，哥请你吃锅包肉去，管够！"),
    ("温柔学姐", "一位温柔知性的学姐，声音柔和清晰，语气亲切，像在给学弟学妹出主意。",
     "别着急，这道题其实不难，我们先把条件一个个列出来，你会发现答案就在里面。"),
    ("古风侠客", "一位潇洒的古风侠客，声音清朗有力，语气洒脱，带着江湖气。",
     "江湖路远，后会有期。若有一日你再遇难处，只需吹响这支竹笛，我必踏月而来。"),
]
VD_ALL = [(n, pr, tx) for n, _, _, pr, tx in VD_PRESETS] + VD_EXTRA
VD_TEXTS = [tx for _, _, tx in VD_ALL]


def vd_engine_choices():
    """能用文字描述造音色的模型。"""
    out = [("Qwen3-TTS 描述造音色", "qwen3"), ("Breeze TTS 2", "breeze")]
    return out


def vd_engine_default():
    if E.ENGINES["qwen3"].pick_model("design"):
        return "qwen3"
    if E.ENGINES["breeze"].ready()[0]:
        return "breeze"
    return "qwen3"


def _vd_chip_updates(on):
    return [gr.update(elem_classes=["vd-chip", "on"] if i == on else ["vd-chip"]) for i in range(len(VD_PRESETS))]


def vd_pick(i):
    _, _, _, prompt, text = VD_PRESETS[i]
    return [prompt, text, i, *_vd_chip_updates(i)]


def vd_shuffle(cur_prompt):
    import random
    pool = [x for x in VD_ALL if x[1] != (cur_prompt or "").strip()]
    name, prompt, text = random.choice(pool)
    on = next((i for i, x in enumerate(VD_PRESETS) if x[0] == name), None)
    return [prompt, text, on, *_vd_chip_updates(on)]


def vd_prompt_edit(prompt, on):
    """手动改了提示词，就不再高亮预设按钮。"""
    if on is not None and (prompt or "").strip() != VD_PRESETS[on][3]:
        return [None, *_vd_chip_updates(None)]
    return [gr.skip()] * (1 + len(VD_PRESETS))


def vd_autotext(prompt, cur):
    import random
    prompt = (prompt or "").strip()
    for _, pr, tx in VD_ALL:
        if pr == prompt and tx != (cur or "").strip():
            return tx
    return random.choice([t for t in VD_TEXTS if t != (cur or "").strip()])


def vd_count(text):
    return f"<span class='vd-count'>{len(text or '')} / {VD_MAX} 字符</span>"


def vd_card(k, cand=None, busy=False, sel=None):
    """一个候选音色卡片里的「播放 + 名字」部分；选中圈是旁边的按钮。"""
    name = f"音色 {k + 1}"
    if busy:
        return f"<div class='vd-c'><span class='vd-play wait'></span><span class='vd-cname'>{name}</span>" \
               f"<span class='vd-cnote'>生成中…</span></div>"
    if not cand:
        return f"<div class='vd-c'><span class='vd-play off'></span><span class='vd-cname'>{name}</span></div>"
    src = f"/_vd/{Path(cand['audio']).name}"
    js = ("const a=this.parentElement.querySelector('audio');"
          "document.querySelectorAll('.vd-c audio').forEach(x=>{if(x!==a){x.pause();x.currentTime=0;}});"
          "a.paused?a.play():a.pause();")
    return (f"<div class='vd-c'><button class='vd-play' onclick=\"{js}\" title='播放'></button>"
            f"<span class='vd-cname'>{name}</span><span class='vd-cnote'>{cand['dur']:.1f} 秒</span>"
            f"<audio src='{src}' preload='auto' onplay=\"this.parentElement.classList.add('playing')\" "
            f"onpause=\"this.parentElement.classList.remove('playing')\" "
            f"onended=\"this.parentElement.classList.remove('playing')\"></audio></div>")


def _vd_radios(sel, cands):
    return [gr.update(elem_classes=["vd-radio", "on"] if k == sel else ["vd-radio"], interactive=bool(cands[k]))
            for k in range(3)]


def vd_generate(prompt, text, engine):
    import random
    prompt, text = (prompt or "").strip(), (text or "").strip()
    empty = [None, None, None]

    def fail(msg, cands=empty, sel=None):
        # 不抛 gr.Error：那样每张卡片上都会贴一个「错误」。弹个提示，状态行写清楚就行
        gr.Warning(msg)
        return [*[vd_card(j, cands[j]) for j in range(3)], cands, sel, *_vd_radios(sel, cands),
                f"<span class='vd-err'>{html.escape(msg)}</span>", gr.update(value="生成")]

    if not prompt:
        yield fail("先写一句音色描述，或点上面的预设")
        return
    if not text:
        yield fail("试听文本是空的：写一句，或点「自动生成」")
        return
    eng = E.ENGINES[engine]
    ok, why = eng.ready()
    if engine == "qwen3" and not eng.pick_model("design"):
        ok, why = False, "还没下载「Qwen3-TTS 1.7B 描述造音色」：去「设置 → 本地模型」下载，或换成 Breeze"
    if not ok:
        yield fail(why)
        return
    base = random.randint(1, 10_000_000)
    cands = [None, None, None]
    voice = {"engine": engine, "kind": "design", "instruction": prompt}
    t0 = time.time()
    for k in range(3):
        cards = [vd_card(j, cands[j], busy=(j == k)) for j in range(3)]
        yield [*cards, cands, None, *_vd_radios(None, cands), f"正在生成第 {k + 1}/3 个音色…", gr.skip()]
        try:
            audio = E.synth(text, voice, "", 4.0, base + k)  # Breeze 描述模式用强度 4（和以前的单句试音一样）
        except Exception as e:
            msg = getattr(e, "message", None) or str(e) or type(e).__name__
            got = [c for c in cands if c]
            yield fail(f"第 {k + 1} 个生成失败：{msg}", cands, 0 if got else None)
            return
        out = DESIGN_DIR / f"{uuid.uuid4().hex}.wav"
        sf.write(out, audio, SR, subtype="PCM_16")
        cands[k] = {"audio": str(out), "seed": base + k, "dur": len(audio) / SR}
    sel = 0
    yield [*[vd_card(j, cands[j]) for j in range(3)], cands, sel, *_vd_radios(sel, cands),
           f"3 个音色都好了，用时 {time.time() - t0:.0f} 秒。点播放听听，选一个点「确认选择」。",
           gr.update(value="重新生成")]


def vd_select(k, cands):
    if not cands or not cands[k]:
        return [gr.skip()] * 4
    return [k, *_vd_radios(k, cands)]


def vd_confirm(name, cands, sel, prompt, text, engine):
    if sel is None or not cands or not cands[sel]:
        raise gr.Error("先生成，再在下面的结果里选一个音色")
    c = cands[sel]
    name = _safe(name) or f"设计音色-{time.strftime('%m%d-%H%M%S')}"
    # 存成「克隆」音色：拿选中的这条当参考音频，之后每句都是同一个声音（直接存描述的话每句音色会飘）
    name = lib_write(name, c["audio"], text, "", c["seed"], engine, "clone")
    try:
        meta = LIB_DIR / name / "voice.json"
        v = json.loads(meta.read_text(encoding="utf-8"))
        v["design_prompt"] = (prompt or "").strip()  # 记下当初的描述，方便以后再造
        meta.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    msg = f"已存进音色库：「{name}」。去「文本朗读」的音色下拉框里选它就能读整篇稿子。"
    if engine == "qwen3" and not E.ENGINES["qwen3"].pick_model("clone"):
        msg += "\n\n⚠️ 用 Qwen3 读这个音色还需要下载「Qwen3-TTS 克隆版」（设置 → 本地模型）。"
    if c["dur"] < 5:
        msg += "\n\n提示：这条试听不到 5 秒，试听文本写长一点（10 秒左右），克隆出来会更稳。"
    return gr.update(choices=lib_choices(), value=name), msg, lib_list(), ""


# ───────────────────────── 声音克隆（照着念一段 → 录音 → 试听 → 存进音色库） ─────────────────────────
# 场景：名字、图标（线条 SVG）、朗读文本（念 10 秒左右，克隆最稳）
VC_SCENES = [
    ("有声读物", "<path d='M3 5.5C5.5 4 8.5 4 12 6c3.5-2 6.5-2 9-.5V19c-2.5-1.5-5.5-1.5-9 .5-3.5-2-6.5-2-9-.5z'/><path d='M12 6v13.5'/>",
     "夜深了，小镇的路灯一盏接一盏亮起来。她推开那扇旧木门，屋里还是小时候的样子，墙上的挂钟慢悠悠地走着，好像一直在等她回家。"),
    ("影视配音", "<path d='M4 8h16v11H4z'/><path d='M4 8l3-4M10 8l3-4M16 8l3-4M4 12h16'/>",
     "我知道你不相信我，可是现在我们已经没有退路了。听我的，天亮之前必须离开这里，不然谁也走不了。"),
    ("Vlog独白", "<rect x='3' y='6' width='13' height='12' rx='2'/><path d='M16 10l5-3v10l-5-3'/>",
     "今天是来这座城市的第三天，早上六点就起床去看日出。说实话有点冷，但当太阳从海面跳出来的那一刻，我觉得一切都值了。"),
    ("教育培训", "<path d='M2 9l10-5 10 5-10 5z'/><path d='M6 11v5c3 2 9 2 12 0v-5'/>",
     "同学们，我们先来看这道题。题目给了两个条件，第一个告诉我们速度，第二个告诉我们时间，那路程怎么求呢？对，速度乘以时间。"),
    ("电台播客", "<circle cx='12' cy='11' r='3'/><path d='M7.5 15.5a6.5 6.5 0 1 1 9 0M12 14v7'/>",
     "欢迎收听今晚的节目，我是你的老朋友。窗外下着小雨，泡一杯热茶，我们慢慢聊聊最近让你开心的那些小事。"),
    ("智能客服", "<path d='M4 5h16v11H9l-5 4z'/><path d='M8.5 10.5h.01M12 10.5h.01M15.5 10.5h.01'/>",
     "您好，很高兴为您服务。您的订单已经发货了，预计明天下午送达。如果还有其他问题，随时联系我就好。"),
]
VC_EXTRA = [
    "今天坐地铁的时候，广播说前面有一站临时停运，要绕路走。大家都低头查路线，我也跟着换了一趟车，虽然多花了十几分钟，但到了以后一切正常。",
    "周末我在家试着做了一次红烧肉，先把肉焯水，再加冰糖慢慢炒出糖色。炖了一个小时，满屋子都是香味，连邻居都过来问是什么菜。",
    "我一直觉得，学东西最快的办法不是看很多教程，而是先动手做一个最小的版本，做坏了也没关系，改着改着就会了。",
]
VC_ALL = [t for _, _, t in VC_SCENES] + VC_EXTRA
VC_DEFAULT_TRY = "你好，很高兴为你提供配音服务。挑一个你喜欢的声音，我们一起开启声音创作的奇妙旅程吧。"
VC_CONSENT = "我确认上传或录制的声音是我本人的，或已取得声音主人的明确授权，只把克隆出来的声音用在合法用途上。"


def vc_engine_choices():
    return [("Qwen3-TTS 克隆", "qwen3"), ("Breeze TTS 2", "breeze")]


def vc_engine_default():
    if E.ENGINES["qwen3"].pick_model("clone"):
        return "qwen3"
    if E.ENGINES["breeze"].ready()[0]:
        return "breeze"
    return "qwen3"


def _vc_chip_updates(on):
    return [gr.update(elem_classes=["vc-chip", "on"] if i == on else ["vc-chip"]) for i in range(len(VC_SCENES) + 1)]


def vc_scene(i, cur):
    """i = 0 是「随机」，1… 是各个场景。"""
    import random
    if i == 0:
        text = random.choice([t for t in VC_ALL if t != (cur or "").strip()])
    else:
        text = VC_SCENES[i - 1][2]
    return [text, *_vc_chip_updates(i)]


def vc_card(cand=None, busy=False):
    if busy:
        return ("<div class='vd-c'><span class='vd-play wait'></span><span class='vd-cname'>克隆音色</span>"
                "<span class='vd-cnote'>生成中…</span></div>")
    if not cand:
        return "<div class='vc-empty'>生成的音色结果将展示在此处</div>"
    return vd_card(0, cand).replace("音色 1", "克隆音色", 1)


def _vc_dur(path):
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return None


def vc_generate(ref, ref_text, text, lang, engine, agree):
    import random
    ref_text, text = (ref_text or "").strip(), (text or "").strip()

    def fail(msg):
        gr.Warning(msg)
        return [vc_card(), None, f"<span class='vd-err'>{html.escape(msg)}</span>", gr.update(value="生成"),
                gr.Row(visible=False)]

    if not agree:
        yield fail("先勾选下面的授权确认")
        return
    if not ref:
        yield fail("先点麦克风录一段（照着上面的文字念），或者上传一段 5～15 秒的音频")
        return
    dur = _vc_dur(ref)
    if dur is not None and dur < 3:
        yield fail(f"录音只有 {dur:.1f} 秒，太短了：完整念完上面那段文字（10 秒左右）")
        return
    if engine == "breeze" and not ref_text:
        yield fail("Breeze 克隆要有录音对应的文字：上面的朗读文本要和你念的内容一致")
        return
    if not text:
        yield fail("试听文本是空的，写一句用来试听的话")
        return
    eng = E.ENGINES[engine]
    ok, why = eng.ready()
    if engine == "qwen3" and not eng.pick_model("clone"):
        ok, why = False, "还没下载「Qwen3-TTS 克隆版」：去「设置 → 本地模型」下载，或换成 Breeze"
    if not ok:
        yield fail(why)
        return
    yield [vc_card(busy=True), None, "正在克隆你的声音，读一遍试听文本…", gr.skip(), gr.Row(visible=False)]
    seed = random.randint(1, 10_000_000)
    voice = {"engine": engine, "kind": "clone", "ref_audio": ref, "ref_text": ref_text}
    if lang:
        voice["language"] = lang
    t0 = time.time()
    try:
        audio = E.synth(text, voice, "", 2.0, seed)
    except Exception as e:
        yield fail("生成失败：" + (getattr(e, "message", None) or str(e) or type(e).__name__))
        return
    out = DESIGN_DIR / f"{uuid.uuid4().hex}.wav"
    sf.write(out, audio, SR, subtype="PCM_16")
    cand = {"audio": str(out), "seed": seed, "dur": len(audio) / SR}
    msg = f"好了，用时 {time.time() - t0:.0f} 秒。点播放听听像不像，满意就起个名字存进音色库。"
    if dur and dur > 20:
        msg += " 提示：录音超过 20 秒，截一段 10 秒左右的效果通常更稳。"
    yield [vc_card(cand), cand, msg, gr.update(value="重新生成"), gr.Row(visible=True)]


def vc_save(name, cand, ref, ref_text, engine):
    if not cand or not ref:
        raise gr.Error("先生成，听过满意再存")
    name = _safe(name) or f"克隆音色-{time.strftime('%m%d-%H%M%S')}"
    name = lib_write(name, ref, ref_text, "", cand["seed"], engine, "clone")
    msg = f"已存进音色库：「{name}」。去「文本朗读」的音色下拉框里选它就能读整篇稿子。"
    return gr.update(choices=lib_choices(), value=name), msg, lib_list(), ""



# ───────────────────────── 自动更新（侧栏底部） ─────────────────────────
def upd_view(info=None, msg=""):
    """侧栏底部：平时只显示版本号；有新版本时变成一张小卡片 + 更新按钮。"""
    cur = updater.current()
    if not info:
        return (f"<div class='upd'><span class='upd-ver'>v{cur}</span>{html.escape(msg)}</div>",
                gr.update(visible=False))
    first = next((ln.strip("# ").strip() for ln in info["notes"].splitlines() if ln.strip()), "")
    card = (f"<div class='upd new'><b>发现新版本 v{html.escape(info['version'])}</b>"
            f"<span>当前 v{cur}{'：' + html.escape(first[:40]) if first else ''}</span>"
            f"<a href='{html.escape(info['page'])}' target='_blank'>看更新内容</a>"
            + (f"<em>{html.escape(msg)}</em>" if msg else "") + "</div>")
    if not updater.installed():
        return card + "<div class='upd-tip'>源码运行：git pull 或去下载页装新版</div>", gr.update(visible=False)
    return card, gr.update(visible=True, value=f"更新到 v{info['version']}", interactive=True)


def upd_check():
    return upd_view(updater.check())


def upd_go():
    import threading
    info = updater.check()
    if not info:
        yield upd_view(None, " · 已是最新")
        return
    err = []
    t = threading.Thread(target=_upd_apply, args=(info, err), daemon=True)
    t.start()
    while t.is_alive():
        yield upd_view(info, updater.progress())[0], gr.update(interactive=False, value="更新中…")
        time.sleep(0.5)
    if err:
        yield upd_view(info, "更新失败：" + err[0])
    else:
        yield upd_view(info, updater.progress())[0], gr.update(interactive=False, value="正在重启…")


def _upd_apply(info, err):
    try:
        updater.apply(info)
        return True
    except Exception as e:
        err.append(str(e) or type(e).__name__)
        return False


# ───────────────────────── 项目（整稿分句） ─────────────────────────
def proj_path(name: str) -> Path:
    safe = "".join(ch for ch in (name or "").strip() if ch not in '\\/:*?"<>|') or "未命名"
    p = PROJ_DIR / safe
    p.mkdir(exist_ok=True)
    return p


LAST_PROJ = PROJ_DIR / "_last.txt"


def save_project(name, script, items, settings=None):
    """settings：这个项目用的音色/语气 {voice, tone, tone_text, cfg}；不传就沿用项目里原来记的。"""
    f = proj_path(name) / "project.json"
    LAST_PROJ.write_text(f.parent.name, encoding="utf-8")  # 记住最近用的项目，打开页面时自动载入
    old = {}
    if f.is_file():
        try:  # 稿子文字变了（比如拆句覆盖了旧稿）时，先把旧的存一份备份，防止丢失
            old = json.loads(f.read_text(encoding="utf-8"))
            if old.get("script") != script and old.get("items"):
                shutil.copyfile(f, f.with_name(f"project_备份_{time.strftime('%m%d_%H%M%S')}.json"))
        except Exception:
            old = {}
    data = {"script": script, "items": items}
    for k in ("voice", "tone", "tone_text", "cfg"):
        v = settings.get(k) if settings is not None else old.get(k)
        if v not in (None, ""):
            data[k] = v
    f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def projects_list():
    return sorted(p.parent.name for p in PROJ_DIR.glob("*/project.json"))


def do_split(script, name, base_seed, per, items):
    sents = group_sentences(split_sentences(script or ""), per)
    if not sents:
        raise gr.Error("稿子是空的")
    old = {it["text"]: it for it in items}  # 文字没变的句子保留已生成的音频
    new = []
    for s in sents:
        o = old.get(s)
        if o:
            new.append(o)
        else:
            new.append({"text": s, "seed": int(base_seed), "audio": None, "gen_text": None})
    save_project(name, script, new)
    keep = sum(1 for it in new if it["audio"])
    return new, f"拆成 {len(new)} 句（其中 {keep} 句沿用已生成的音频）。"


VOICE_FOLLOW = "跟整篇"


def _item_voice(it):
    """这句单独选的音色名；没选 / 音色已被删 = None（跟右边整篇的音色）。"""
    v = it.get("voice")
    return v if v and v in lib_list() else None


def _needs_gen(it):
    """要（重新）生成：没音频、字改了、或单独换了音色。"""
    return bool(it["text"]) and (not it["audio"] or it["gen_text"] != it["text"]
                                 or _item_voice(it) != it.get("gen_voice"))


PAUSE_RE = re.compile(r"<#(\d+(?:\.\d+)?)#>")  # 文字里的 <#0.5#> = 在这里停 0.5 秒


def _synth_with_pauses(text, voice, tone, cfg, seed):
    """按 <#秒数#> 把一句切开分别读，中间插静音；没有停顿标记就是普通的一次合成。"""
    bits = PAUSE_RE.split(text)
    if len(bits) == 1:
        return E.synth(text, voice, tone, cfg, seed)
    parts = []
    for k, b in enumerate(bits):
        if k % 2:
            parts.append(np.zeros(int(min(10.0, float(b)) * SR), dtype=np.float32))
        elif b.strip():
            parts.append(np.asarray(E.synth(b.strip(), voice, tone, cfg, seed), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def _gen_item(it, name, idx, gvoice, cfg, tone, lang=""):
    v = _item_voice(it)  # 这句单独选了音色就用它（可以是别的模型），否则用整篇的；语气和强度跟整篇
    use = v or gvoice
    if not use or use not in lib_list():
        raise gr.Error(f"第 {idx + 1} 句没有音色：先在右边「音色」里选一个")
    voice = dict(lib_read(use))
    if lang:
        voice["language"] = lang
    audio = _synth_with_pauses(it["text"], voice, tone, cfg, it["seed"])
    d = proj_path(name)
    out = d / f"s{idx + 1:03d}_{time.strftime('%H%M%S')}.wav"
    sf.write(out, audio, SR, subtype="PCM_16")
    it["audio"] = str(out)
    it["gen_text"] = it["text"]
    it["gen_voice"] = v


def set_item_voice(i, v, items, name, script):
    """某一句单独换音色：只记下来并标成「音色改了」，点那行「重新生成」或底部「生成全部」才真正读。"""
    if i >= len(items):
        return gr.skip(), gr.skip()
    v = None if not v or v == VOICE_FOLLOW else v
    if v == items[i].get("voice"):
        return gr.skip(), gr.skip()
    items[i]["voice"] = v
    save_project(name, script, items)
    who = f"音色「{v}」" if v else "跟整篇的音色"
    return items, f"第 {i + 1} 句改成{who}：点那一行的「重新生成」，或点底部「生成全部」（只会补这类改过的句子）。"


def gen_missing(items, name, script, voice, tone_name, cfg, tone, seed, lang="", progress=gr.Progress()):
    if not items:
        raise gr.Error("先点「拆句」")
    if voice not in lib_list() and any(_needs_gen(it) and not _item_voice(it) for it in items):
        raise gr.Error("先在「音色」下拉框里选一个音色")
    # 记住这个项目用的音色和语气，下次载入自动选上
    save_project(name, script, items,
                 {"voice": voice, "tone": tone_name, "tone_text": (tone or "").strip(), "cfg": float(cfg)})
    todo = [i for i, it in enumerate(items) if _needs_gen(it)]  # 刚插入还没填字的空句跳过
    if not todo:
        yield items, "所有句子都已生成，没有需要补的。", gr.skip()
        return
    t0 = time.time()
    for n, i in enumerate(todo):
        progress(n / len(todo), desc=f"生成第 {i + 1} 句（{n + 1}/{len(todo)}）")
        _gen_item(items[i], name, i, voice, cfg, tone, lang)
        save_project(name, script, items)
        last = n == len(todo) - 1
        # 每句生成完立刻推给「实时试听」自动播放，句子列表也每句刷新，能看到哪句已经 ✅
        yield (
            items,
            f"已生成 {n + 1}/{len(todo)}：第 {i + 1} 句「{items[i]['text']}」，已用 {time.time() - t0:.0f} 秒"
            + ("。全部完成。" if last else "，正在做下一句…"),
            items[i]["audio"],
        )


def regen_one(i, text, items, name, script, voice, cfg, tone, _seed, lang, bump):
    it = items[i]
    it["text"] = (text or "").strip()
    # 文字和种子都不变时读出来几乎一样（实测波形相关 0.999），所以没改字也要换种子；只换了音色就不用换种子
    if bump or (it["text"] == it.get("gen_text") and _item_voice(it) == it.get("gen_voice")):
        it["seed"] = int(it["seed"]) + 1
    t0 = time.time()
    _gen_item(it, name, i, voice, cfg, tone, lang)
    save_project(name, script, items)
    return items, f"第 {i + 1} 句已重新生成（种子 {it['seed']}），用时 {time.time() - t0:.0f} 秒。"


def edit_text(i, text, items, name, script):
    if i < len(items):
        items[i]["text"] = (text or "").strip()  # 就地修改，不触发整表重绘
        save_project(name, script, items)


def insert_one(pos, items, name, script, base_seed):
    """在第 pos 个位置插一句空白句，其他句子和已生成的音频原样保留。"""
    items.insert(pos, {"text": "", "seed": int(base_seed), "audio": None, "gen_text": None})
    save_project(name, script, items)
    return items, f"已在第 {pos + 1} 句的位置插入一句空白句：填好文字后点那一行的「重新生成」，或点底部的「生成全部」。"


def clear_audio(items, name, script):
    """一键清空已生成的音频：文件挪进「旧音频_时间」文件夹（不删），所有句子回到 ⬜ 未生成。"""
    if not items:
        raise gr.Error("还没有句子")
    d = proj_path(name)
    bak = d / f"旧音频_{time.strftime('%m%d_%H%M%S')}"
    moved = 0
    for it in items:
        p = Path(it["audio"]) if it.get("audio") else None
        if p and p.is_file():
            bak.mkdir(exist_ok=True)
            shutil.move(str(p), str(bak / p.name))
            moved += 1
        it["audio"] = None
        it["gen_text"] = None
    if (d / "project.json").is_file() and moved:
        shutil.copyfile(d / "project.json", bak / "project_清空前.json")
    save_project(name, script, items)
    where = f"旧文件挪到了 {bak}" if moved else "没有需要挪的旧文件"
    return items, f"已清空 {moved} 句的音频（{where}）。现在点底部的「生成全部」，每生成好一句，那一行就会从「未生成」变成「已生成」。"


def delete_one(i, items, name, script):
    items.pop(i)
    save_project(name, script, items)
    return items, f"已删除第 {i + 1} 句。"


def merge_all(items, gap, speed, name, pitch=0, vol=1.0):
    parts, skipped = [], []
    gap_audio = np.zeros(int(float(gap) * SR), dtype=np.float32)
    for n, it in enumerate(items, 1):
        if not it["audio"]:
            if it["text"]:  # 没填字的空白句直接跳过，不算漏生成
                skipped.append(f"第{n}句未生成")
            continue
        if it["gen_text"] != it["text"]:
            skipped.append(f"第{n}句文字已改、音频是旧的")
        a, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
        a = E.to_out_sr(a.mean(axis=1), sr)  # 不同模型采样率不同，统一后再拼
        if parts:
            parts.append(gap_audio)
        parts.append(change_speed(a, SR, speed))  # 逐句变速，停顿仍是设定的秒数；原句音频不动
    if not parts:
        raise gr.Error("还没有生成任何音频")
    full = change_volume(change_pitch(np.concatenate(parts), SR, pitch), vol)
    out = proj_path(name) / f"完整_{float(speed):.2f}倍速_{time.strftime('%H%M%S')}.wav"
    sf.write(out, full, SR, subtype="PCM_16")
    msg = (f"已合并 {sum(1 for it in items if it['audio'])} 句，"
           f"{float(speed):.2f} 倍速，共 {len(full) / SR:.0f} 秒：{out}")
    if skipped:
        msg += "\n\n⚠️ " + "；".join(skipped)
    return str(out), msg


def load_project(pick):
    if not pick:
        raise gr.Error("先在下拉框里选一个项目")
    d = json.loads((PROJ_DIR / pick / "project.json").read_text(encoding="utf-8"))
    for it in d["items"]:
        if it.get("audio") and not Path(it["audio"]).is_file():
            it["audio"] = None
            it["gen_text"] = None
    v = d.get("voice")
    if v and v in lib_list():
        tip = f"，音色「{v}」"
    else:
        v = None
        tip = "。这个项目还没选音色，先在「音色」下拉框里选一个"
    tone_text = d.get("tone_text", "")
    tone_name = tone_match(tone_text)
    if v:
        tip += f"，语气「{tone_name if tone_text else TONE_NONE}」"
    return (pick, d["script"], d["items"], v, tone_name, tone_text, d.get("cfg", 2.0),
            f"已载入项目「{pick}」，共 {len(d['items'])} 句{tip}。")


def auto_load():
    """打开/刷新页面（包括服务重启后）自动载入最近的项目，不用每次手动点「载入」。"""
    try:
        last = LAST_PROJ.read_text(encoding="utf-8").strip()
    except OSError:
        last = ""
    if not last or not (PROJ_DIR / last / "project.json").is_file():
        return (gr.skip(),) * 8
    *rest, msg = load_project(last)
    return (*rest, msg.replace("已载入", "已自动载入上次的"))


# ───────────────────────── 界面 ─────────────────────────
# 长稿页按操作顺序排：左边稿子/句子，右边固定的音色语气设置，底部固定的生成+合并操作条。
CSS = """
:root, .dark {
  --nw-accent: #7c5cff; --nw-accent-2: #a28bff; --nw-soft: #f4f1ff;
  --nw-line: #ececf2; --nw-text: #1d1d27; --nw-sub: #8a8a99; --nw-bg: #ffffff;
  --nw-side-w: 220px; --nw-bar-h: 150px;
}
body, gradio-app, .gradio-container { background: var(--nw-bg) !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; overflow: visible !important; overflow-x: clip !important;
  font-family: "PingFang SC", "Microsoft YaHei", "HarmonyOS Sans SC", system-ui, sans-serif !important; }
footer { display: none !important; }
.gradio-container > .main, .gradio-container .main.app { max-width: none !important; margin: 0 !important; padding: 0 !important; }

/* 左侧导航：固定在左边，滚动时不会断 */
#nw-shell { gap: 0 !important; flex-wrap: nowrap !important; }
#nw-side { background: #fbfbfd; border-right: 1px solid var(--nw-line); padding: 22px 12px !important;
  position: fixed !important; left: 0; top: 0; bottom: 0; width: var(--nw-side-w) !important;
  min-width: var(--nw-side-w) !important; gap: 2px !important; z-index: 20; overflow-y: auto; }
#nw-side .nw-brand { display: flex; align-items: center; gap: 10px; padding: 2px 8px 26px; }
#nw-side .nw-logo { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--nw-accent), var(--nw-accent-2)); color: #fff; font-weight: 800; font-size: 18px; }
#nw-side .nw-name { font-weight: 800; font-size: 17px; color: var(--nw-text); }
#nw-side .nw-name small { display: block; font-weight: 500; font-size: 12px; color: var(--nw-sub); }
#nw-side button.nav { justify-content: flex-start !important; background: transparent !important; border: none !important;
  box-shadow: none !important; color: var(--nw-text) !important; font-size: 16px !important; font-weight: 500 !important;
  padding: 12px 16px !important; border-radius: 14px !important; gap: 14px !important; }
#nw-side button.nav::before { content: ""; width: 22px; height: 22px; flex: none; background: currentColor;
  -webkit-mask: var(--ico) center / contain no-repeat; mask: var(--ico) center / contain no-repeat; }
#nw-side button.nav:hover { background: #f6f6f9 !important; }
#nw-side button.nav.on { background: #f3f4f7 !important; font-weight: 700 !important; }
#nw-side .nav-group { font-size: 13px; color: var(--nw-sub); padding: 18px 12px 6px; }
#nw-side .nav-gap { flex: 1; min-height: 18px; }
.upd { font-size: 12px; color: var(--nw-sub); padding: 6px 10px 0; display: flex; flex-direction: column; gap: 2px; }
.upd-ver { color: #b0b0ba; }
.upd.new { background: #f3f0ff; border-radius: 12px; padding: 10px 12px; margin: 8px 6px 0; color: #5a5a68; }
.upd.new b { color: var(--nw-accent); font-size: 13px; }
.upd.new a { color: var(--nw-accent); text-decoration: underline; width: fit-content; }
.upd.new em { font-style: normal; color: #5a5a68; margin-top: 4px; }
.upd-tip { font-size: 11px; color: var(--nw-sub); padding: 4px 16px 0; }
button.upd-btn { margin: 6px 6px 0 !important; background: var(--nw-accent) !important; color: #fff !important; border: none !important;
  border-radius: 999px !important; height: 34px; font-size: 13px !important; font-weight: 700 !important; box-shadow: none !important; }
#nw-side .nw-foot { margin-top: auto; font-size: 12px; color: var(--nw-sub); line-height: 1.7; padding: 0 10px; }

/* 主区 */
#nw-main { margin-left: var(--nw-side-w) !important; padding: 22px 36px 40px !important; min-width: 0; }
#nw-pages > .tab-wrapper, #nw-pages > div[role=tablist], #nw-pages .tab-container { display: none !important; }
#nw-pages { border: none !important; padding: 0 !important; }
/* 切页时整页外面会留一圈紫色“生成中”边框，去掉 */
#nw-pages > .wrap.generating { border-color: transparent !important; }
#nw-pages .tabitem { border: none !important; padding: 0 !important; background: transparent !important; }
.topbar { align-items: center !important; gap: 14px !important; margin-bottom: 14px; flex-wrap: nowrap !important; }
.nw-title { font-size: 24px; font-weight: 900; color: var(--nw-text); white-space: nowrap; }
.nw-hint { color: var(--nw-sub); font-size: 13px; }
.nw-label { font-weight: 700; font-size: 14px; color: var(--nw-text); margin: 6px 0 -2px; }

.card { border: 1px solid var(--nw-line) !important; border-radius: 18px !important; background: #fff !important;
  padding: 16px 18px !important; box-shadow: 0 1px 2px rgba(20, 20, 40, .03) !important; margin-bottom: 14px; }
.card .form, .card .block { border: none !important; box-shadow: none !important; background: transparent !important; }
.nw-sec { font-weight: 800; font-size: 17px; color: var(--nw-text); margin: 4px 0 6px; }
.nw-divider { border-top: 1px solid var(--nw-line); margin: 6px 0; }
.tone-line textarea { background: #fafafe !important; font-size: 13px !important; }

.bare textarea { border: none !important; box-shadow: none !important; background: transparent !important;
  font-size: 15px !important; line-height: 1.75 !important; padding: 4px 2px !important; }
.bare textarea::placeholder { color: #b3b3c0; }

/* 药丸选项（语气 / 预设音色） */
.chips .wrap { gap: 8px !important; }
.chips label { border: 1px solid var(--nw-line) !important; border-radius: 10px !important; background: #fff !important;
  padding: 6px 12px !important; box-shadow: none !important; font-weight: 600; color: #3a3a48; font-size: 13px !important; }
.chips label:hover { border-color: #d6cffc !important; }
.chips label.selected, .chips label:has(input:checked) { background: var(--nw-soft) !important; border-color: var(--nw-accent) !important;
  color: var(--nw-accent) !important; }
.chips input[type=radio] { display: none !important; }

.toolbar, .topbar { justify-content: flex-start !important; }
/* gr.HTML 不认 scale=0，小标题会把后面的控件挤到最右边，这里手动收紧 */
.topbar > .block:has(.nw-title), .toolbar > .block:has(.nw-label), .toolbar > .block:has(.nw-sec) {
  flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.toolbar { align-items: center !important; gap: 10px !important; flex-wrap: wrap !important; }
.pill, .pill .wrap, .pill input { border-radius: 12px !important; }
.pill { border: 1px solid var(--nw-line) !important; background: #fff !important; }
button.warn, button.ghost, button.go, button.mini { white-space: nowrap !important; }
button.go { background: linear-gradient(90deg, var(--nw-accent), var(--nw-accent-2)) !important; color: #fff !important;
  border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 10px 20px !important;
  box-shadow: 0 6px 16px rgba(124, 92, 255, .25) !important; }
button.go:hover { filter: brightness(1.05); }
button.ghost { background: #fff !important; border: 1px solid var(--nw-accent) !important; border-radius: 12px !important;
  color: var(--nw-accent) !important; font-weight: 700 !important; box-shadow: none !important; }
button.warn { background: #fff !important; border: 1px solid #ffd2d2 !important; border-radius: 12px !important;
  color: #e5484d !important; font-weight: 600 !important; box-shadow: none !important; }
button.mini { background: transparent !important; border: 1px solid var(--nw-line) !important; border-radius: 8px !important;
  color: #6a6a78 !important; box-shadow: none !important; font-weight: 500 !important; padding: 4px 8px !important; }
button.mini:hover { color: var(--nw-accent) !important; border-color: #d6cffc !important; }
button.mini.del:hover { color: #e5484d !important; border-color: #ffd2d2 !important; }

/* 长稿页：左右两栏 + 底部操作条 */
#nw-work { align-items: flex-start !important; gap: 20px !important; padding-bottom: calc(var(--nw-bar-h) + 20px); }
#nw-right { position: sticky !important; top: 16px; max-height: calc(100vh - var(--nw-bar-h) - 124px);  /* 页面没滚时右栏从标题栏下面开始，留够位置别被底部操作条盖住 */ overflow-y: auto;
  scrollbar-width: thin; }
#nw-right { flex-wrap: nowrap !important; }  /* 不许换列：放不下时「更多」会被挤到右边另起一列 */
#nw-right > * { flex-shrink: 0 !important; }
#nw-right .card { margin-bottom: 12px; }
#nw-bar { position: fixed !important; left: var(--nw-side-w); right: 0; bottom: 0; z-index: 60; width: auto !important;
  background: #fff !important; border-top: 1px solid var(--nw-line);
  box-shadow: 0 -6px 20px rgba(20, 20, 40, .05); padding: 8px 36px 10px !important; gap: 4px !important;
  max-height: 45vh; overflow-y: auto; }
#nw-bar .bar-row { gap: 16px !important; align-items: center !important; flex-wrap: nowrap !important; }
#nw-bar .form { background: transparent !important; border: none !important; box-shadow: none !important; }
#nw-bar .block { border: none !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; }
#nw-bar .bar-sep { width: 1px; height: 70px; background: var(--nw-line); }
#nw-bar .bar-status { font-size: 13px; color: #5a5a68; min-height: 1.4em; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
#nw-bar .bar-status p { margin: 0 !important; overflow: hidden; text-overflow: ellipsis; }
#nw-bar .bar-sliders { gap: 2px !important; }
#nw-bar .bar-sliders span[data-testid="block-info"], #nw-bar label span[data-testid="block-label"],
#nw-bar label[data-testid="block-label"] { background: transparent !important; color: #5a5a68 !important;
  font-size: 12px !important; font-weight: 600 !important; padding: 0 !important; border: none !important; white-space: nowrap; }
/* 操作条里的播放器去掉波形，只留播放键和进度，条子矮一半 */
#nw-bar .waveform-container { display: none !important; }
#nw-bar .component-wrapper, #nw-bar .audio-container { padding: 0 !important; min-height: 0 !important; }
#nw-bar .empty { min-height: 44px !important; height: 44px !important; }
#nw-bar .bar-sliders input[type=number] { width: 52px !important; font-size: 12px !important; }

/* 句子列表 */
.list-hint { color: var(--nw-sub); font-size: 13px; margin-bottom: 4px; }
.sent-row { border: 1px solid var(--nw-line); border-radius: 14px; padding: 8px 10px !important; margin-bottom: 8px;
  background: #fff; align-items: center !important; }
.sent-row:hover { border-color: #ddd6ff; }
.sent-no { font-weight: 800; color: var(--nw-text); font-size: 15px; }
.sent-st { display: inline-block; white-space: nowrap; font-size: 12px; border-radius: 6px; padding: 1px 6px; margin-top: 4px; font-weight: 600; }
.st-todo { background: #f2f2f5; color: #8a8a99; } .st-ok { background: #e8f7ee; color: #1f9a55; }
.st-old { background: #fff4e0; color: #c27a00; }
.mrow { border-bottom: 1px solid var(--nw-line); padding: 8px 0 !important; align-items: center !important; }
.mname { font-weight: 700; color: var(--nw-text); }
.mname small { display: block; font-weight: 400; color: var(--nw-sub); font-size: 12px; margin-top: 2px; }
.mstate { font-size: 12px; font-weight: 600; border-radius: 6px; padding: 2px 8px; white-space: nowrap; }
.mstate.ok { background: #e8f8ee; color: #1f8a4c; } .mstate.run { background: var(--nw-soft); color: var(--nw-accent); }
.mstate.err { background: #fff0f0; color: #d33; } .mstate.todo { background: #f3f3f6; color: var(--nw-sub); }
.set-info:empty { display: none; }
.sent-ops { gap: 6px !important; }
.sent-voice input { font-size: 12px !important; }
.sent-voice.own, .sent-voice.own .wrap { border-color: var(--nw-accent-2) !important; background: var(--nw-soft) !important; }
.sent-ops .row { gap: 6px !important; flex-wrap: nowrap !important; }

/* 音色卡片 */
.voice-grid { flex-wrap: wrap !important; gap: 16px !important; border: none !important; }
.voice-card { border: 1px solid var(--nw-line) !important; border-radius: 18px !important; background: #fff !important;
  padding: 14px !important; flex: 0 1 250px !important; min-width: 230px !important; }
.voice-card:hover { box-shadow: 0 8px 24px rgba(124, 92, 255, .12) !important; border-color: #ddd6ff !important; }
.voice-card .row { flex-wrap: nowrap !important; gap: 8px !important; }
.vc-art { height: 110px; border-radius: 14px; background: var(--nw-soft); display: grid; place-items: center; margin-bottom: 10px;
  position: relative; overflow: hidden; }
.vc-art i { position: absolute; border-radius: 50%; opacity: .9; }
.vc-name { font-weight: 800; font-size: 16px; color: var(--nw-text); }
.vc-tag { display: inline-block; font-size: 12px; color: var(--nw-accent); background: var(--nw-soft); border-radius: 8px;
  padding: 1px 8px; margin-left: 6px; font-weight: 600; vertical-align: middle; }
.vc-desc { color: var(--nw-sub); font-size: 13px; margin-top: 4px; min-height: 36px; }

/* 窗口窄时：音色语气挪到句子上面，操作条仍固定在底部（放不下就折成两行） */
@media (max-width: 1100px) {
  :root { --nw-bar-h: 210px; }
  #nw-bar { padding: 8px 20px 10px !important; }
  #nw-bar .bar-row { flex-wrap: wrap !important; row-gap: 6px !important; }
  #nw-bar .bar-row > * { flex: 1 1 220px !important; min-width: 0 !important; }
  #nw-bar .bar-row > button { flex: 0 0 auto !important; }
  #nw-bar .bar-row > .block:has(.bar-sep) { display: none !important; }
  #nw-main { padding: 18px 20px 30px !important; }
  #nw-work { flex-wrap: wrap !important; }
  #nw-work > div { min-width: 100% !important; }
  #nw-right { position: static !important; order: -1; max-height: none; overflow: visible; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="violet", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.Font(f) for f in ("PingFang SC", "Microsoft YaHei", "system-ui", "sans-serif")],
)

PAGES = [("lib", "音色库"), ("home", "文本朗读"), ("trial", "音色设计"), ("clone", "声音克隆"), ("api", "开放接口"), ("mcp", "MCP 连接"), ("set", "设置")]
NAV_GROUP = {"home": "AI 生成", "trial": "AI 工具", "api": "开发者"}  # 这个页面上面加一行分组小标题
NAV_ICON = {  # 线条图标（24×24，stroke），用 CSS mask 着色，跟文字同色
    "lib": "<path d='M4 20V5M8 20V5M13 20V4l7 2v14'/>",
    "home": "<rect x='3' y='3' width='14' height='14' rx='3'/><path d='M7 7.5h6M10 7.5V14'/>"
            "<path d='M20 13v8M17 16.5v3M23 15.5v4' />",
    "trial": "<path d='M3 10v4M6.5 7v10M10 9v6M13.5 11v2'/><path d='M18.5 3l1.2 2.8L22.5 7l-2.8 1.2L18.5 11l-1.2-2.8L14.5 7"
             "l2.8-1.2z'/>",
    "clone": "<path d='M8 3H5a2 2 0 0 0-2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 3h3a2 2 0 0 1 2 2v3'/>"
             "<rect x='9' y='7' width='5' height='8' rx='2.5'/><path d='M7 13a4.5 4.5 0 0 0 9 0M11.5 17.5V20M17 17v4M15 19h4'/>",
    "api": "<path d='M8 7l-5 5 5 5M16 7l5 5-5 5M13.5 4.5l-3 15'/>",
    "mcp": "<path d='M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4'/>",
    "set": "<circle cx='12' cy='12' r='3'/><path d='M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1"
           "M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1'/>",
}


def _nav_css():
    from urllib.parse import quote
    out = []
    for pid, svg in NAV_ICON.items():
        full = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
                f"stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>{svg}</svg>")
        out.append(f'#nav-{pid} {{ --ico: url("data:image/svg+xml,{quote(full)}"); }}')
    return "\n".join(out)


CSS += _nav_css()

def _vd_css():
    """音色设计页的样式（照 MiniMax 音色设计）：预设头像、图标都在这里生成。"""
    from urllib.parse import quote

    def ico(svg):
        full = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
                f"stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'>{svg}</svg>")
        return f'url("data:image/svg+xml,{quote(full)}")'

    out = [f"#vd-chip-{i} {{ --emo: '{emo}'; --hue: {hue}; }}" for i, (_, emo, hue, _p, _t) in enumerate(VD_PRESETS)]
    shuf = ico("<path d='M3 7h3.5c4 0 6 10 10 10H21M3 17h3.5c1.6 0 2.8-1.6 3.9-3.5M13.6 9.5C14.7 7.6 15.9 7 17.5 7H21M18 4l3 3-3 3M18 14l3 3-3 3'/>")
    clr = ico("<path d='M7 20h13M5.5 14.5l7.8-7.8a2 2 0 0 1 2.8 0l2.2 2.2a2 2 0 0 1 0 2.8L12 18H8.5z'/><path d='M9.5 10.5l5 5'/>")
    spark = ico("<path d='M10 3l1.8 5.2L17 10l-5.2 1.8L10 17l-1.8-5.2L3 10l5.2-1.8z' fill='black'/><path d='M18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z' fill='black'/>")
    out.append("#vd-shuffle { --ico: %s; }" % shuf)
    out.append("#vd-clear { --ico: %s; }" % clr)
    out.append("#vd-auto, .vd-primary { --spark: %s; }" % spark)
    return "\n".join(out)


CSS += """
/* ───── 音色设计页 ───── */
#vd-page { max-width: 1180px; margin: 0 auto !important; padding-top: 2px !important; }
.vd-head h1 { font-size: 24px; font-weight: 900; color: var(--nw-text); margin: 0 0 6px; }
.vd-head p { font-size: 13px; color: var(--nw-sub); margin: 0 0 6px; }
.vd-label { font-size: 15px; font-weight: 700; color: var(--nw-text); margin: 16px 0 4px; }
.vd-box { border: 1px solid #e8e8ee !important; border-radius: 18px !important; background: #fff !important;
  padding: 0 !important; gap: 0 !important; overflow: hidden; box-shadow: none !important; }
.vd-text, .vd-text > label, .vd-text .wrap { background: transparent !important; border: none !important; box-shadow: none !important; }
.vd-text textarea { border: none !important; box-shadow: none !important; background: transparent !important;
  font-size: 15px !important; line-height: 1.75 !important; padding: 14px 18px !important; color: var(--nw-text) !important; }
.vd-chips { border-top: 1px solid #ececf2; padding: 10px 16px !important; gap: 8px !important; flex-wrap: wrap !important;
  align-items: center !important; }
.vd-chips button { flex: none !important; }
button.vd-chip { background: #fff !important; border: 1px solid #ececf2 !important; border-radius: 999px !important;
  box-shadow: none !important; color: var(--nw-text) !important; font-size: 14px !important; font-weight: 500 !important;
  padding: 4px 14px 4px 4px !important; gap: 8px !important; min-height: 38px; }
button.vd-chip::before { content: var(--emo); width: 30px; height: 30px; border-radius: 50%; display: grid;
  place-items: center; font-size: 15px; flex: none;
  background: radial-gradient(circle at 30% 30%, hsl(var(--hue) 70% 78%), hsl(var(--hue) 55% 38%)); }
button.vd-chip:hover { background: #f8f8fb !important; }
button.vd-chip.on { border: 1.5px solid var(--nw-text) !important; }
button.vd-shuf, button.vd-icon { background: #fff !important; border: 1px solid #ececf2 !important; border-radius: 50% !important;
  box-shadow: none !important; width: 38px; height: 38px; min-width: 38px !important; padding: 0 !important; }
button.vd-icon { border: none !important; width: 28px; height: 28px; min-width: 28px !important; }
button.vd-shuf::before, button.vd-icon::before { content: ""; width: 20px; height: 20px; background: #555;
  -webkit-mask: var(--ico) center / contain no-repeat; mask: var(--ico) center / contain no-repeat; }
button.vd-icon::before { width: 18px; height: 18px; background: #888; }
.vd-row { align-items: flex-end !important; margin-top: 12px; }
.vd-row .vd-label { margin: 0; }
button.vd-auto { background: #fff !important; border: 1px solid #ececf2 !important; border-radius: 999px !important;
  box-shadow: none !important; color: var(--nw-text) !important; font-weight: 600 !important; font-size: 14px !important;
  height: 36px; padding: 0 16px !important; gap: 6px !important; flex-direction: row !important; white-space: nowrap; }
button.vd-auto::before, button.vd-primary:not(#vd-ok)::before { content: ""; width: 15px; height: 15px; background: currentColor;
  -webkit-mask: var(--spark) center / contain no-repeat; mask: var(--spark) center / contain no-repeat; }
.vd-foot { padding: 0 16px 10px !important; align-items: center !important; gap: 6px !important; }
.vd-count { font-size: 13px; color: var(--nw-sub); white-space: nowrap; }
.vd-actions { justify-content: flex-end !important; align-items: center !important; gap: 12px !important; margin-top: 14px; }
button.vd-primary { background: var(--nw-accent) !important; color: #fff !important; border: none !important;
  border-radius: 999px !important; height: 42px; font-size: 15px !important; font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(124, 92, 255, .24) !important; gap: 8px !important; }
button.vd-primary:hover { filter: brightness(1.06); }
button.vd-primary[disabled] { opacity: .6; }
.vd-engine, .vd-name { min-width: 200px; }
.vd-name input { height: 40px; border-radius: 999px !important; padding: 0 20px !important; }
.vd-status:empty { display: none; }
.vd-status { text-align: right; color: var(--nw-sub); }
.vd-err { color: #d4380d; }
.vd-results { gap: 14px !important; flex-wrap: wrap !important; margin-top: 6px; }
.vd-card { background: #f5f6f8 !important; border-radius: 14px !important; padding: 12px 16px !important;
  align-items: center !important; flex: 1 1 260px !important; min-width: 240px; gap: 8px !important; flex-wrap: nowrap !important;
  border: 1.5px solid transparent; transition: background .15s; }
.vd-card:has(button.vd-radio.on) { background: #eceef3 !important; }
.vd-c { display: flex; align-items: center; gap: 12px; }
.vd-play { width: 38px; height: 38px; border-radius: 50%; background: #1d1d27; border: none; cursor: pointer; flex: none;
  display: inline-grid; place-items: center; padding: 0; }
.vd-play::before { content: ""; width: 0; height: 0; margin-left: 4px; border-left: 12px solid #fff;
  border-top: 7px solid transparent; border-bottom: 7px solid transparent; }
.vd-c.playing .vd-play::before { width: 14px; height: 16px; margin: 0; border: none;
  background: linear-gradient(90deg, #fff 0 35%, transparent 35% 65%, #fff 65%); }
.vd-play.off, .vd-play.wait { background: #c9cad3; cursor: default; }
.vd-play.wait::before { border: 3px solid #fff; border-top-color: transparent; border-radius: 50%; width: 16px; height: 16px;
  margin: 0; animation: vdspin .8s linear infinite; }
@keyframes vdspin { to { transform: rotate(360deg); } }
.vd-cname { font-size: 16px; font-weight: 600; color: var(--nw-text); }
.vd-cnote { font-size: 13px; color: var(--nw-sub); }
.vd-c audio { display: none; }
button.vd-radio { width: 22px !important; height: 22px; min-width: 22px !important; border-radius: 50% !important; padding: 0 !important;
  background: transparent !important; border: 2px solid #8a8a95 !important; box-shadow: none !important; flex: none !important; }
button.vd-radio.on { background: #1d1d27 !important; border-color: #1d1d27 !important; }
button.vd-radio.on::before { content: ""; width: 10px; height: 5px; border-left: 2px solid #fff; border-bottom: 2px solid #fff;
  transform: translateY(-1px) rotate(-45deg); }
button.vd-radio[disabled] { opacity: .45; }
#vd-page .html-container { padding: 0 !important; }
"""
CSS += _vd_css()


def _vc_css():
    from urllib.parse import quote

    def ico(svg):
        full = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
                f"stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>{svg}</svg>")
        return f'url("data:image/svg+xml,{quote(full)}")'

    shuf = "<path d='M3 7h3.5c4 0 6 10 10 10H21M3 17h3.5c1.6 0 2.8-1.6 3.9-3.5M13.6 9.5C14.7 7.6 15.9 7 17.5 7H21M18 4l3 3-3 3M18 14l3 3-3 3'/>"
    out = ["#vc-chip-0 { --ico: %s; }" % ico(shuf)]
    out += ["#vc-chip-%d { --ico: %s; }" % (i + 1, ico(svg)) for i, (_, svg, _t) in enumerate(VC_SCENES)]
    out.append("#vc-clear { --ico: %s; }" % ico("<path d='M7 20h13M5.5 14.5l7.8-7.8a2 2 0 0 1 2.8 0l2.2 2.2a2 2 0 0 1 0 2.8L12 18H8.5z'/><path d='M9.5 10.5l5 5'/>"))
    return "\n".join(out)


CSS += """
/* ───── 声音克隆页（版式跟音色设计一样） ───── */
#vc-page { max-width: 1180px; margin: 0 auto !important; padding-top: 2px !important; }
#vc-page .html-container { padding: 0 !important; }
.vc-chips { gap: 8px !important; flex-wrap: wrap !important; margin: 2px 0 10px; }
.vc-chips button { flex: none !important; }
button.vc-chip { background: #fff !important; border: 1px solid #ececf2 !important; border-radius: 999px !important;
  box-shadow: none !important; color: var(--nw-text) !important; font-size: 14px !important; font-weight: 500 !important;
  padding: 0 14px !important; gap: 6px !important; height: 36px; flex-direction: row !important; white-space: nowrap; }
button.vc-chip::before { content: ""; width: 16px; height: 16px; background: #555; flex: none;
  -webkit-mask: var(--ico) center / contain no-repeat; mask: var(--ico) center / contain no-repeat; }
button.vc-chip:hover { background: #f8f8fb !important; }
button.vc-chip.on { border: 1.5px solid var(--nw-accent) !important; color: var(--nw-accent) !important; }
button.vc-chip.on::before { background: var(--nw-accent); }
.vc-read textarea { text-align: center; font-size: 16px !important; font-weight: 600 !important; line-height: 1.85 !important;
  padding: 18px 22px 4px !important; min-height: 0 !important; height: auto !important; field-sizing: content; max-height: 220px; }
.vc-rec { width: min(520px, 92%) !important; margin: 16px auto 10px !important; border: 1px solid #ececf2 !important;
  border-radius: 22px !important; box-shadow: 0 4px 14px rgba(0,0,0,.06) !important; background: #fff !important; }
.vc-rec .label-wrap, .vc-rec label { display: none !important; }
.vc-tip { text-align: center; font-size: 12px; color: var(--nw-sub); margin: 0 0 16px; padding: 0 16px; }
.vc-agree input[type=checkbox] { border: 1.5px solid #9a9aa6 !important; }
#vc-page .vd-box:has(.vc-tip) { padding-bottom: 16px !important; }
.vc-langrow { border-bottom: 1px solid #ececf2; padding: 4px 8px !important; }
.vc-lang, .vc-lang .wrap, .vc-lang input { background: transparent !important; border: none !important; box-shadow: none !important; }
.vc-lang input { font-size: 15px !important; font-weight: 600 !important; }
.vc-agree { margin-top: 12px; background: transparent !important; border: none !important; box-shadow: none !important; }
.vc-agree label span { font-size: 13px !important; color: var(--nw-sub) !important; line-height: 1.7; }
.vc-empty { color: #b0b0ba; font-size: 18px; font-weight: 700; padding: 6px 0 4px; }
.vc-result { margin-top: 18px; }
.vc-result .vd-card { max-width: 420px; }
"""
CSS += _vc_css()



# ───────────────────────── 设置页 ─────────────────────────
def _st_html(mid):
    st = E.model_status(mid)
    cls = "ok" if st == "已下载" else ("run" if st.startswith("下载中") else ("err" if "出错" in st else "todo"))
    return f"<span class='mstate {cls}'>{html.escape(st)}</span>"


def settings_tick():
    return [E.gpu_state()] + [_st_html(c["id"]) for c in E.CATALOG]


def save_paths(model_dir, preload):
    if E.any_downloading():
        raise gr.Error("有模型正在下载，下完（或暂停）再改文件夹")
    s = E.settings_load()
    s["model_dir"] = (model_dir or "").strip() or E.DEFAULT_SETTINGS["model_dir"]
    s["preload"] = "breeze" if preload else ""
    E.settings_save(s)
    return f"已保存。模型下载到：{s['model_dir']}"


def save_api(mm_key, mm_base, mm_model, vc_appid, vc_token, vc_cluster, oa_base, oa_key, oa_model):
    s = E.settings_load()
    s["minimax"].update(api_key=mm_key.strip(), base_url=mm_base.strip(), model=mm_model.strip())
    s["volc"].update(appid=vc_appid.strip(), token=vc_token.strip(), cluster=vc_cluster.strip())
    s["openai"].update(base_url=oa_base.strip(), api_key=oa_key.strip(), model=oa_model.strip())
    E.settings_save(s)
    return "已保存（存在本机 config/settings.json，不会上传）。"


def api_test(engine_id, speaker):
    eng = E.ENGINES[engine_id]
    t0 = time.time()
    audio = E.synth("你好，这是一段测试配音，听听效果怎么样。",
                    {"engine": engine_id, "kind": "speaker", "speaker": speaker or eng.speakers()[0]})
    out = OUT_DIR / f"test_{engine_id}_{time.strftime('%H%M%S')}.wav"
    sf.write(out, audio, SR, subtype="PCM_16")
    return str(out), f"{eng.name} 通了：{len(audio) / SR:.1f} 秒，用时 {time.time() - t0:.1f} 秒。"


PORT = int(os.environ.get("TTS_STUDIO_PORT", 7861))


def _lan_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))  # 不会真的发包，只是让系统选一块网卡
            return sk.getsockname()[0]
    except OSError:
        return "本机IP"


def api_base(lan=None):
    lan = api_server.server_conf()["lan"] if lan is None else lan
    return f"http://{_lan_ip() if lan else '127.0.0.1'}:{PORT}/v1"


def api_examples(lan=None):
    base = api_base(lan)
    voice = (lib_list() or ["你的音色名"])[0]
    return f"""# 1) 看有哪些音色
curl {base}/voices -H "Authorization: Bearer 你的接口Key"

# 2) 读一句话，存成 out.wav（voice 填音色库里的名字，或 qwen3:vivian 这种「模型:音色ID」）
curl {base}/audio/speech -H "Authorization: Bearer 你的接口Key" -H "Content-Type: application/json" \\
  -d '{{"input": "你好，这是一段测试配音。", "voice": "{voice}", "response_format": "wav"}}' -o out.wav

# 3) Python（OpenAI 官方 SDK，pip install openai）
from openai import OpenAI
client = OpenAI(base_url="{base}", api_key="你的接口Key")
client.audio.speech.create(model="tts-studio", voice="{voice}", input="你好",
                           response_format="wav").write_to_file("out.wav")

# 可选参数：speed（0.5~2）、instructions（语气提示词）、seed、response_format（wav/mp3/flac/aac/opus/pcm）
"""


def save_server(enabled, lan):
    s = E.settings_load()
    srv = s.setdefault("server", {})
    restart = bool(lan) != bool(srv.get("lan"))
    srv["api_enabled"], srv["lan"] = bool(enabled), bool(lan)
    E.settings_save(s)
    msg = "已保存。" + ("开放接口已开启。" if enabled else "开放接口已关闭，外部调用会被拒绝。")
    if restart:
        msg += "**局域网访问要重启软件后才生效**" + ("；重启后打开网页要登录：用户名随便填，密码是接口 Key。" if lan else "。")
    return msg, api_base(lan), api_examples(lan)


def api_voice_list():
    """开放接口页：列出 voice 能填什么，以及现在能不能用。"""
    rows = ["| voice 填这个 | 模型 | 现在能用吗 |", "|---|---|---|"]
    for n in lib_list():
        eng = E.engine_of(lib_read(n))
        ok, why = eng.ready()
        rows.append(f"| `{n}` | {eng.name} | {'✅' if ok else '⚠️ ' + why} |")
    for eid, eng in E.ENGINES.items():
        if "speaker" in eng.voice_kinds:
            ok, why = eng.ready()
            ids = "、".join(f"`{eid}:{x}`" for x in eng.speakers()[:4]) + " …"
            rows.append(f"| {ids} | {eng.name} 自带音色 | {'✅' if ok else '⚠️ ' + why} |")
    return "\n".join(rows)


def regen_key():
    return api_server.new_key(), "已换成新 Key，旧 Key 立刻失效；用到它的程序记得改。"


def gpu_release():
    E.unload_all()
    return E.gpu_state()


def _art(name: str) -> str:
    """音色卡片上的抽象图案：按名字算颜色和位置，每个音色长得不一样。"""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    dots = []
    for k in range(7):
        size = 18 + (h >> (k * 5)) % 46
        x = 8 + (h >> (k * 7 + 3)) % 80
        y = 10 + (h >> (k * 6 + 11)) % 70
        hue = 250 + (h >> (k * 4 + 17)) % 50
        dots.append(f"<i style='width:{size}px;height:{size}px;left:{x}%;top:{y}%;"
                    f"background:hsl({hue} 80% {72 + k * 2}%)'></i>")
    return f"<div class='vc-art'>{''.join(dots)}</div>"


# ═════ MCP 连接页 ═════
MCP_CSS = """
.mcp-badge { display:flex; align-items:center; gap:12px; font-size:18px; font-weight:800; color:var(--nw-text); }
.mcp-dot { width:12px; height:12px; border-radius:50%; background:#c7cbd6; flex:none; }
.mcp-dot.on { background:#22c55e; box-shadow:0 0 0 5px rgba(34,197,94,.18); }
.mcp-sub { color:var(--nw-sub); font-size:13px; font-weight:400; margin-top:4px; }
.mcp-list { width:100%; border-collapse:collapse; margin-top:14px; font-size:14px; }
.mcp-list th { text-align:left; color:var(--nw-sub); font-weight:500; padding:6px 8px; border-bottom:1px solid var(--nw-line); }
.mcp-list td { padding:9px 8px; border-bottom:1px solid var(--nw-line); }
.mcp-tag { display:inline-block; padding:2px 10px; border-radius:99px; font-size:12px; font-weight:600; }
.mcp-tag.on { background:#dcfce7; color:#15803d; } .mcp-tag.off { background:#f1f5f9; color:#64748b; }
.mcp-tools { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }
.mcp-tools div { border:1px solid var(--nw-line); border-radius:14px; padding:10px 12px; font-size:13px; color:var(--nw-sub); }
.mcp-field textarea, .mcp-field input { height: 40px !important; min-height: 0 !important; max-height: 40px !important;
  overflow: hidden !important; resize: none !important; white-space: nowrap; }
.mcp-tools b { display:block; color:var(--nw-text); font-family:ui-monospace,Consolas,monospace; margin-bottom:2px; }
"""
CSS += MCP_CSS


def _ago(t):
    d = max(0, int(time.time() - t))
    return "刚刚" if d < 10 else f"{d} 秒前" if d < 60 else f"{d // 60} 分钟前" if d < 3600 else time.strftime("%H:%M", time.localtime(t))


def mcp_status_html():
    rows = mcp_server.status()
    live = [r for r in rows if r["alive"]]
    if not api_server.server_conf().get("api_enabled"):
        head = ("<div class='mcp-badge'><span class='mcp-dot'></span>MCP 没开</div>"
                "<div class='mcp-sub'>MCP 和开放接口共用一个开关：去「开放接口」页打开「开启接口」。</div>")
    elif live:
        names = "、".join(dict.fromkeys(r["name"] for r in live))
        head = (f"<div class='mcp-badge'><span class='mcp-dot on'></span>已连接 · {len(live)} 个</div>"
                f"<div class='mcp-sub'>{html.escape(names)} 正在连着，可以直接让它配音。</div>")
    else:
        head = ("<div class='mcp-badge'><span class='mcp-dot'></span>等待连接</div>"
                "<div class='mcp-sub'>MCP 已就绪。按下面的方法在 Agent 里加上它，连上后这里会显示「已连接」。</div>")
    if not rows:
        return f"<div>{head}</div>"
    tr = []
    for r in rows:
        how = "本机启动（stdio）" if r["transport"] == "stdio" else "网址连接（HTTP）"
        tag = "<span class='mcp-tag on'>已连接</span>" if r["alive"] else "<span class='mcp-tag off'>已断开</span>"
        calls = r.get("calls") or 0
        tr.append(f"<tr><td><b>{html.escape(r['name'])}</b></td><td>{how}</td><td>{tag}</td>"
                  f"<td>{_ago(r['first'])}</td><td>{'保持连接中' if r['alive'] and r.get('streams') else _ago(r['last'])}</td><td>{calls if r['transport'] == 'http' else '—'}</td></tr>")
    return (f"<div>{head}<table class='mcp-list'><tr><th>客户端</th><th>连接方式</th><th>状态</th><th>连上</th>"
            f"<th>最近活动</th><th>调用次数</th></tr>{''.join(tr)}</table></div>")


def _mcp_py():
    """给 Agent 用的 python：用虚拟环境里的（别用基础解释器，那里没装依赖）。"""
    import sys
    cand = Path(sys.prefix) / "Scripts" / "python.exe"
    return str(cand if cand.exists() else Path(sys.executable))


def mcp_configs(lan=None):
    """各家软件的接入配置。返回 dict：名字 -> (说明, 语言, 内容)。"""
    srv = api_server.server_conf()
    lan = srv["lan"] if lan is None else lan
    base = f"http://{_lan_ip() if lan else '127.0.0.1'}:{PORT}"
    key = srv["api_key"]
    py, script = _mcp_py(), str(Path(__file__).resolve().parent / "mcp_server.py")
    stdio = {"command": py, "args": [script]}
    http = {"type": "http", "url": f"{base}/mcp", "headers": {"Authorization": f"Bearer {key}"}}
    j = lambda d: json.dumps({"mcpServers": {"tts-studio": d}}, ensure_ascii=False, indent=2)
    return {
        "豆包 / WorkBuddy（填表）": (
            "在「新建自定义连接器 / 添加 MCP」里照下面一格一格填（每格右边可以复制）：传输类型选 HTTP，"
            "自定义 Headers 点「添加」填一行。保存后回到这里，上面会显示「已连接」。"
            "这些软件要装在这台电脑上才连得到（网页版、云端智能体连不到本机）。",
            "form",
            {"name": "tts-studio", "transport": "HTTP", "url": f"{base}/mcp",
             "h_name": "Authorization", "h_value": f"Bearer {key}", "url_key": f"{base}/mcp?key={key}"}),
        "Claude Code": ("在终端里运行这一行（--scope user 表示所有项目都能用）。加完开一个新会话就能用。", "shell",
                        f'claude mcp add tts-studio --scope user -- "{py}" "{script}"'),
        "Claude Desktop": ("Claude Desktop → 设置 → 开发者 → 编辑配置，把下面这段合并进 claude_desktop_config.json，"
                           "然后完全退出 Claude 再打开。", "json", j(stdio)),
        "Cursor / Trae / VS Code": ("Cursor：设置 → MCP → Add new MCP server；Trae：AI 对话框 → MCP → 手动添加；"
                                    "VS Code：.vscode/mcp.json（把最外层的 mcpServers 改成 servers）。粘贴下面这段。",
                                    "json", j(http)),
        "扣子 / Cherry Studio / 其他（填网址）": (
            "支持「Streamable HTTP」MCP 的软件（Cherry Studio、豆包/扣子里能加自定义 MCP 的地方、各种 Agent 框架），"
            "填下面的网址就行。能填请求头的，用第一行网址 + 请求头；只能填网址的，用第二行（Key 已经拼在网址里）。"
            "注意：云端的智能体（比如扣子网页版）连不到你这台电脑，要先在「开放接口」页打开局域网访问，"
            "再用内网穿透工具把这个地址映射到公网。",
            "shell",
            f"网址：  {base}/mcp\n请求头：Authorization: Bearer {key}\n\n只能填网址时：\n{base}/mcp?key={key}"),
        "Codex": ("把下面这段加到 ~/.codex/config.toml。", "shell",
                  f'[mcp_servers.tts-studio]\ncommand = {json.dumps(py, ensure_ascii=False)}\n'
                  f'args = [{json.dumps(script, ensure_ascii=False)}]'),
    }


MCP_TOOLS_HTML = "<div class='mcp-tools'>" + "".join(
    f"<div><b>{n}</b>{d}</div>" for n, d in [
        ("list_voices", "查音色库里有哪些音色"),
        ("list_models", "查哪些模型能用、不能用的原因"),
        ("speak", "一段文字 → 音频文件"),
        ("dub_script", "整篇稿子 → 拆句、逐句生成、合成一条，带每句时间点"),
    ]) + "</div>"


def mcp_key_ok(request) -> bool:
    """MCP / 心跳的鉴权：和开放接口同一个开关、同一个 Key；支持请求头或 ?key=。"""
    import secrets
    srv = api_server.server_conf()
    if not srv.get("api_enabled"):
        return False
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else \
        (request.headers.get("x-api-key") or request.query_params.get("key", ""))
    return bool(key) and secrets.compare_digest(key.encode(), srv["api_key"].encode())


# ═════ 文本朗读页（照 MiniMax「文字转语音」） ═════
TTS_MAX = 5000
LANG_CHOICES = [(n, v) for n, v in E.LANGS]
ALL_MODELS = "全部模型"


def tts_count(text):
    n = len(text or "")
    cls = " over" if n > TTS_MAX else ""
    return f"<span class='tts-count{cls}'>{n:,} / {TTS_MAX:,} 字符</span>"


def tts_model_choices():
    return [ALL_MODELS] + [(e.name, eid) for eid, e in E.ENGINES.items()]


def tts_voice_choices(model):
    """右边音色下拉：按顶上选的模型筛。"""
    out = []
    for label, n in lib_choices():
        try:
            eng = lib_read(n).get("engine") or "breeze"
        except Exception:
            eng = "breeze"
        if model in (None, "", ALL_MODELS) or eng == model:
            out.append((label, n))
    return out


def tts_model_pick(model, cur):
    ch = tts_voice_choices(model)
    names = [v for _l, v in ch]
    return gr.update(choices=ch, value=cur if cur in names else (names[0] if names else None))


def tts_voice_card(name):
    if not name or name not in lib_list():
        return ("<div class='tts-vc'><div class='tts-vc-art empty'></div><div><div class='tts-vc-name'>还没选音色</div>"
                "<div class='tts-vc-sub'>在下面选一个；音色库里没有就先去「音色库 → 新建音色」</div></div></div>")
    v = lib_read(name)
    eng = E.ENGINES.get(v.get("engine") or "breeze")
    kind = {"clone": "克隆音色", "speaker": "预设音色", "design": "描述音色"}.get(
        v.get("kind") or ("clone" if v.get("ref_audio") else "design"), "")
    sub = " · ".join(x for x in (eng.name if eng else "", kind) if x)
    return (f"<div class='tts-vc'>{_art(name).replace('vc-art', 'tts-vc-art')}<div><div class='tts-vc-name'>"
            f"{html.escape(name)}</div><div class='tts-vc-sub'>{html.escape(sub)}</div></div></div>")


def tts_lang_tip(voice, lang):
    if not lang or lang == "auto" or not voice or voice not in lib_list():
        return ""
    eid = lib_read(voice).get("engine") or "breeze"
    if eid in E.LANG_ENGINES:
        return ""
    return "<div class='tts-tip'>这个音色的模型会自己识别语言，选的语言对它不起作用（Qwen3 和 MiniMax 才会按选的读）。</div>"


def tts_history(name):
    """生成历史：这个项目合并出来的完整音频，新的在前。"""
    try:
        files = sorted(proj_path(name).glob("完整_*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:30]
    except Exception:
        files = []
    if not files:
        return "<div class='tts-empty'>还没有生成记录。点「生成音频」后，每次的成品都会出现在这里。</div>"
    rows = []
    from urllib.parse import quote
    for f in files:
        try:
            dur = sf.info(str(f)).duration
        except Exception:
            dur = 0
        when = time.strftime("%m-%d %H:%M", time.localtime(f.stat().st_mtime))
        src = "/gradio_api/file=" + quote(str(f).replace("\\", "/"))
        rows.append(f"<div class='tts-h'><div class='tts-h-top'><b>{html.escape(f.stem.replace('完整_', ''))}</b>"
                    f"<span>{when} · {dur:.0f} 秒</span></div><audio controls preload='none' src='{src}'></audio></div>")
    return "".join(rows)


def tts_upload(f):
    if not f:
        return gr.skip()
    raw = Path(f).read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise gr.Error("读不出这个文件的文字，换成 UTF-8 编码的 .txt 再试")


def tts_generate(script, name, seed, per, items, longmode, voice, tone_name, cfg, tone, lang,
                 gap, speed, pitch, vol, progress=gr.Progress()):
    """「生成音频」：拆句（长文模式下已有句子就不重拆）→ 只生成没生成/改过的句子 → 合并成一条。"""
    if not (script or "").strip() and not items:
        raise gr.Error("先在上面输入文字")
    if len(script or "") > TTS_MAX:
        raise gr.Error(f"文字超过 {TTS_MAX:,} 字了，分成几个项目来做")
    if not longmode or not items:
        items, msg = do_split(script, name, seed, per, items)
        yield items, msg, gr.skip(), gr.skip(), gr.skip()
    last_audio = None
    for items, msg, au in gen_missing(items, name, script, voice, tone_name, cfg, tone, seed, lang, progress):
        last_audio = au if au is not gr.skip() else last_audio
        yield items, msg, au, gr.skip(), gr.skip()
    if longmode:  # 拆句模式：只出逐句的音频，满意了再点「合并成完整音频」
        done = sum(1 for it in items if it["audio"])
        yield items, f"已生成 {done}/{len(items)} 句。逐句听一遍，不满意的改字或点「重新生成」，都好了点「合并成完整音频」。", \
            gr.skip(), gr.skip(), gr.skip()
        return
    out, mmsg = merge_all(items, gap, speed, name, pitch, vol)
    yield items, mmsg, gr.skip(), out, tts_history(name)


def tts_remerge(items, gap, speed, name, pitch, vol):
    out, msg = merge_all(items, gap, speed, name, pitch, vol)
    return out, msg, tts_history(name)


def ins_js(token):
    """工具栏按钮：把标记插到编辑框光标处（跟 MiniMax 一样），再通知 Gradio 文字变了。"""
    tok = json.dumps(token, ensure_ascii=False)
    return ("() => { const ta = document.querySelector('#tts-script textarea'); if (!ta) return;"
            " const a = ta.selectionStart ?? ta.value.length, b = ta.selectionEnd ?? a;"
            f" ta.value = ta.value.slice(0, a) + {tok} + ta.value.slice(b);"
            f" ta.selectionStart = ta.selectionEnd = a + {len(token)};"
            " ta.dispatchEvent(new Event('input', {bubbles: true})); ta.focus(); }")


TTS_CSS = """
/* ───── 文本朗读页（照 MiniMax 文字转语音） ───── */
#tts-top { align-items: center !important; gap: 14px !important; margin-bottom: 14px; flex-wrap: nowrap !important; }
#tts-top .nw-title { font-size: 26px; }
.tts-model { border: 1px solid var(--nw-line); border-radius: 14px; padding: 2px 6px 2px 18px !important; gap: 4px !important;
  align-items: center !important; flex-wrap: nowrap !important; background: #fff; min-width: 380px; }
.tts-model .tts-model-l { font-weight: 700; font-size: 14px; color: var(--nw-text); white-space: nowrap; padding-right: 12px;
  border-right: 1px solid var(--nw-line); }
.tts-model .block, .tts-model .form, .tts-model .wrap { border: none !important; box-shadow: none !important; background: transparent !important; }
#tts-grid { gap: 18px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }
#tts-left { min-width: 0 !important; gap: 14px !important; }
.tts-editor { border: 1px solid var(--nw-line) !important; border-radius: 18px !important; background: #fff !important;
  padding: 0 !important; gap: 0 !important; overflow: hidden; box-shadow: none !important; }
#tts-script, #tts-script .wrap, #tts-script label { border: none !important; box-shadow: none !important; background: transparent !important; }
#tts-script textarea { border: none !important; box-shadow: none !important; background: transparent !important; resize: none !important;
  font-size: 15px !important; line-height: 1.85 !important; padding: 20px 22px !important; color: var(--nw-text) !important;
  min-height: calc(100vh - 330px) !important; }
#tts-script textarea::placeholder { color: #b6b6c3; }
#tts-left:has(.tts-long:not(.hidden):not([hidden])) #tts-script textarea { min-height: 240px !important; max-height: 360px !important; overflow-y: auto !important; }  /* 长文模式：编辑框变矮，句子列表露出来 */
.tts-tools { padding: 8px 18px !important; gap: 8px !important; align-items: center !important; flex-wrap: wrap !important; }
.tts-tools > * { flex: none !important; }
button.tts-tag { background: #fff !important; border: 1px solid var(--nw-line) !important; border-radius: 8px !important;
  box-shadow: none !important; color: #4a4a58 !important; font-size: 13px !important; font-weight: 500 !important;
  height: 30px; padding: 0 10px !important; min-width: 0 !important; }
button.tts-tag:hover { border-color: #d6cffc !important; color: var(--nw-accent) !important; }
button.tts-tag.emo { background: var(--nw-accent) !important; border-color: var(--nw-accent) !important; color: #fff !important; }
.tts-sep { width: 1px; height: 18px; background: var(--nw-line); }
.tts-tools .tts-fill { flex: 1 1 auto !important; }
.tts-switch { min-width: 0 !important; }
.tts-switch label { gap: 8px !important; flex-direction: row-reverse !important; font-size: 13px !important; color: #4a4a58 !important;
  background: transparent !important; border: none !important; padding: 0 !important; white-space: nowrap; }
.tts-switch input[type=checkbox] { appearance: none; -webkit-appearance: none; width: 30px !important; height: 18px !important;
  border-radius: 99px !important; background: #d9d9e3 !important; position: relative; cursor: pointer; border: none !important;
  transition: background .15s; margin: 0 !important; }
.tts-switch input[type=checkbox]::after { content: ""; position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
  border-radius: 50%; background: #fff; transition: left .15s; box-shadow: 0 1px 2px rgba(0,0,0,.2); }
.tts-switch input[type=checkbox]:checked { background: var(--nw-accent) !important; }
.tts-switch input[type=checkbox]:checked::after { left: 14px; }
.tts-count { font-size: 13px; color: var(--nw-sub); white-space: nowrap; }
.tts-count.over { color: #e5484d; }
button.tts-icon { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
  width: 28px; height: 28px; min-width: 28px !important; color: #888 !important; font-size: 15px !important; }
.tts-bottom { border-top: 1px solid var(--nw-line); padding: 10px 18px 12px !important; gap: 10px !important;
  align-items: center !important; flex-wrap: nowrap !important; }
.tts-bottom > * { flex: none !important; }
.tts-lang { min-width: 210px !important; }
.tts-lang .wrap { border-radius: 10px !important; background: #f6f6f9 !important; border: none !important; }
.tts-up { border: 1px solid var(--nw-line) !important; border-radius: 10px !important; background: #fff !important;
  box-shadow: none !important; height: 38px; min-width: 38px !important; width: 38px; padding: 0 !important; font-size: 16px !important; color: #333 !important; }
button.tts-go { background: var(--nw-accent) !important; color: #fff !important; border: none !important; border-radius: 999px !important;
  height: 42px; font-size: 15px !important; font-weight: 700 !important; box-shadow: 0 4px 12px rgba(124, 92, 255, .24) !important;
  padding: 0 34px !important; white-space: nowrap; }
button.tts-go:hover { filter: brightness(1.06); }
button.tts-split { background: #fff !important; color: var(--nw-accent) !important; border: 1.5px solid var(--nw-accent) !important;
  border-radius: 999px !important; height: 42px; font-size: 15px !important; font-weight: 700 !important; box-shadow: none !important;
  padding: 0 26px !important; white-space: nowrap; }
button.tts-split:hover { background: #f5f2ff !important; }
.tts-status { font-size: 13px; color: #5a5a68; }
.tts-status:empty { display: none; }
.tts-status p { margin: 0 !important; }
.tts-result { border: 1px solid var(--nw-line) !important; border-radius: 16px !important; background: #fff !important;
  padding: 10px 16px !important; box-shadow: none !important; }
.tts-result .block { border: none !important; box-shadow: none !important; }
.tts-long { border: 1px solid var(--nw-line) !important; border-radius: 16px !important; background: #fff !important;
  padding: 12px 16px !important; box-shadow: none !important; gap: 8px !important; }
#tts-right { width: 360px !important; min-width: 360px !important; max-width: 360px; flex: none !important;
  border: 1px solid var(--nw-line) !important; border-radius: 18px !important; background: #fff !important;
  padding: 14px 18px 18px !important; gap: 10px !important; position: sticky !important; top: 16px;
  max-height: calc(100vh - 40px); overflow-y: auto; flex-wrap: nowrap !important; }
#tts-right > * { flex-shrink: 0 !important; }
.tts-tabs { border-bottom: 1px solid var(--nw-line); margin-bottom: 4px; }
.tts-tabs .wrap { gap: 22px !important; }
.tts-tabs label { background: transparent !important; border: none !important; box-shadow: none !important; padding: 6px 0 10px !important;
  font-size: 16px !important; font-weight: 600 !important; color: #a3a3b2 !important; border-radius: 0 !important; }
.tts-tabs label.selected, .tts-tabs label:has(input:checked) { color: var(--nw-text) !important;
  border-bottom: 2px solid var(--nw-text) !important; }
.tts-tabs input[type=radio] { display: none !important; }
.tts-sec { display: flex; justify-content: space-between; align-items: center; font-weight: 700; font-size: 15px;
  color: var(--nw-text); margin: 6px 0 2px; }
button.tts-reset { background: transparent !important; border: none !important; box-shadow: none !important;
  color: #6b6b7a !important; font-size: 13px !important; padding: 0 !important; min-width: 0 !important; height: 24px; }
.tts-vcard { border: 1px solid var(--nw-line) !important; border-radius: 14px !important; background: #f8f8fb !important;
  padding: 12px !important; gap: 10px !important; box-shadow: none !important; }
.tts-vc { display: flex; align-items: center; gap: 14px; }
.tts-vc-art { width: 58px; height: 58px; border-radius: 12px; flex: none; position: relative; overflow: hidden;
  background: linear-gradient(135deg, #efe9ff, #dcd3ff); }
.tts-vc-art.empty { background: #ececf2; }
.tts-vc-art i { position: absolute; border-radius: 50%; filter: blur(5px); opacity: .9; transform: translate(-50%, -50%); }
.tts-vc-name { font-weight: 700; font-size: 16px; color: var(--nw-text); }
.tts-vc-sub { font-size: 12px; color: var(--nw-sub); margin-top: 3px; }
.tts-vcard .wrap { background: #fff !important; border-radius: 10px !important; }
.tts-tip { font-size: 12px; color: #b26a00; background: #fff7e6; border-radius: 8px; padding: 6px 10px; }
.tts-tip:empty, .tts-tipbox:has(.tts-tip:empty) { display: none; }
.tts-acc { border: none !important; border-top: 1px solid var(--nw-line) !important; border-bottom: 1px solid var(--nw-line) !important;
  border-radius: 0 !important; box-shadow: none !important; padding: 4px 0 !important; background: transparent !important; }
.tts-acc > button { font-weight: 700 !important; font-size: 15px !important; color: var(--nw-text) !important; }
.tts-sl { margin-top: 4px; }
.tts-sl input[type=range] { accent-color: #1d1d27; }
.tts-sl span[data-testid="block-info"], .tts-sl label span { font-weight: 600 !important; font-size: 14px !important; color: var(--nw-text) !important; }
.tts-sl input[type=number] { width: 56px !important; border-radius: 8px !important; background: #f6f6f9 !important; border: none !important; text-align: center; }
.tts-h { border: 1px solid var(--nw-line); border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }
.tts-h-top { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; margin-bottom: 6px; }
.tts-h-top b { color: var(--nw-text); font-weight: 600; }
.tts-h-top span { color: var(--nw-sub); white-space: nowrap; }
.tts-h audio { width: 100%; height: 34px; }
.tts-empty { color: var(--nw-sub); font-size: 13px; padding: 20px 0; text-align: center; }
.tts-tools > .block, .tts-bottom > .block { width: auto !important; flex: none !important; min-width: 0 !important;
  padding: 0 !important; border: none !important; background: transparent !important; box-shadow: none !important; }
.tts-tools > .block.tts-fill, .tts-bottom > .block.tts-fill { flex: 1 1 auto !important; }
.tts-tools, .tts-bottom { flex-wrap: nowrap !important; }
.tts-acc { overflow: visible !important; }
button.tts-reset { white-space: nowrap !important; width: auto !important; }
.tts-sl { border: none !important; background: transparent !important; box-shadow: none !important; padding: 2px 0 !important; }
.tts-prev .waveform-container, .tts-prev .timestamps { display: none !important; }
.tts-prev { border: none !important; box-shadow: none !important; padding: 0 !important; }
.tts-prev .component-wrapper { padding: 0 !important; min-height: 0 !important; }
#tts-right .toolbar { align-items: center !important; flex-wrap: nowrap !important; }
#tts-right { overflow-x: hidden !important; }
#tts-right .toolbar > .block { width: auto !important; flex: 1 1 auto !important; min-width: 0 !important; }
#tts-right .toolbar > button.tts-reset { flex: none !important; }
#tts-right > .column, #tts-right > div { max-width: 100%; }
#tts-right .form { background: transparent !important; border: none !important; box-shadow: none !important;
  gap: 10px !important; padding: 0 !important; }
#tts-right .tts-sl, #tts-right .tts-sl .wrap, #tts-right .tts-sl .head, #tts-right .tts-sl label,
#tts-right .tts-sl span[data-testid="block-info"] { background: transparent !important; box-shadow: none !important;
  border: none !important; padding-left: 0 !important; }
#tts-right .tts-sl span[data-testid="block-info"] { padding: 0 !important; }
#tts-right .tts-sl .min_value, #tts-right .tts-sl .max_value { font-size: 12px !important; color: #a3a3b2 !important; }
.tts-prev label[data-testid="block-label"], .tts-prev .block-label { background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important; color: var(--nw-sub) !important; font-size: 13px !important; position: static !important; }
.tts-prev .icon-button-wrapper { display: none !important; }
@media (max-width: 1100px) {
  #tts-grid { flex-wrap: wrap !important; }
  #tts-right { width: 100% !important; max-width: none; min-width: 0 !important; position: static !important; }
  .tts-model { min-width: 0; }
}
"""
CSS += TTS_CSS


def go_page(pid):
    return [gr.Tabs(selected=pid)] + [gr.update(elem_classes=["nav", "on"] if p == pid else ["nav"]) for p, _ in PAGES]


def proj_switch(name):
    """项目框：选已有项目 = 载入；输入一个新名字 = 空白新项目（旧项目都在硬盘上，随时能选回来）。"""
    name = _safe(name)
    if name in projects_list():
        return load_project(name)
    return (name, "", [], gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            f"新项目「{name}」：在左边粘贴稿子，点「拆句」后才会存下来。")


def script_open(items):
    """有句子时收起整篇稿子，页面上只留一份稿子（句子列表）。"""
    return gr.Accordion(open=not items, label="整篇稿子" + (f"（已拆成 {len(items)} 句，点开可重新拆句）" if items else ""))


with gr.Blocks(title="TTS Studio") as demo:
    with gr.Row(elem_id="nw-shell"):
        # ── 左侧导航 ──
        with gr.Column(elem_id="nw-side", scale=0, min_width=220):
            gr.HTML("<div class='nw-brand'><div class='nw-logo'>声</div>"
                    "<div class='nw-name'>TTS Studio<small>多模型配音 · 本地</small></div></div>")
            nav_btns = []
            for pid, label in PAGES:
                if pid == "set":
                    gr.HTML("<div class='nav-gap'></div>")
                elif pid in NAV_GROUP:
                    gr.HTML(f"<div class='nav-group'>{NAV_GROUP[pid]}</div>")
                nav_btns.append(gr.Button(label, elem_id=f"nav-{pid}", elem_classes=["nav", "on"] if pid == "home" else ["nav"]))
            gr.HTML("<div class='nw-foot'>Breeze 可加 [笑] [叹气] [咳嗽]<br>显存不够时先关掉其他占显卡的软件<br>"
                    "克隆别人的声音前请先取得本人同意</div>")
            upd_html = gr.HTML(upd_view()[0])
            upd_btn = gr.Button("更新", visible=False, elem_classes="upd-btn")

        # ── 主区 ──
        with gr.Column(elem_id="nw-main", scale=1):
            with gr.Tabs(elem_id="nw-pages", selected="home") as pages:

                # ═════ 文本朗读 ═════
                with gr.Tab("文本朗读", id="home"):
                    with gr.Row(elem_id="tts-top"):
                        gr.HTML("<div class='nw-title'>文本朗读</div>", scale=0, min_width=110)
                        proj_name = gr.Dropdown(projects_list(), value="稿子1", allow_custom_value=True, show_label=False,
                                                container=False, scale=0, min_width=200, elem_classes="pill",
                                                info=None)
                        gr.HTML("<span class='nw-hint'>项目：选已有的自动载入，输入新名字就是新项目</span>", scale=1)
                        with gr.Row(elem_classes="tts-model", scale=0):
                            gr.HTML("<span class='tts-model-l'>模型</span>", scale=0, min_width=60)
                            model_dd = gr.Dropdown(tts_model_choices(), value=ALL_MODELS, show_label=False,
                                                   container=False, scale=1, min_width=230)

                    with gr.Row(elem_id="tts-grid"):
                        # ── 左：编辑框 + 结果 + 长文模式的句子列表 ──
                        with gr.Column(elem_id="tts-left", scale=1):
                            with gr.Column(elem_classes="tts-editor"):
                                script = gr.Textbox(show_label=False, lines=14, max_lines=400, elem_id="tts-script",
                                                    container=False,
                                                    placeholder="在此处开始输入文字，生成你的个性化音频。\n"
                                                                "💡 点下面的「停顿」「语气词」，会在光标处插入标记："
                                                                "<#0.5#> 表示停 0.5 秒，[笑] 这类语气词 Breeze 会读出来。")
                                with gr.Row(elem_classes="tts-tools"):
                                    emo_btn = gr.Button("＋ 情绪", elem_classes=["tts-tag", "emo"], scale=0, min_width=0)
                                    gr.HTML("<div class='tts-sep'></div>", scale=0, min_width=1)
                                    pause_btn = gr.Button("<#> 停顿", elem_classes="tts-tag", scale=0, min_width=0)
                                    laugh_btn = gr.Button("( ) 语气词", elem_classes="tts-tag", scale=0, min_width=0)
                                    gr.HTML("", elem_classes="tts-fill", scale=1, min_width=10)
                                    longmode = gr.Checkbox(False, label="长文模式", container=False, scale=0, min_width=0,
                                                           elem_classes="tts-switch")
                                    gr.HTML("<div class='tts-sep'></div>", scale=0, min_width=1)
                                    count_html = gr.HTML(tts_count(""), scale=0, min_width=0)
                                    erase_btn = gr.Button("⌫", elem_classes="tts-icon", scale=0, min_width=0)
                                with gr.Row(elem_classes="tts-bottom"):
                                    lang_dd = gr.Dropdown(LANG_CHOICES, value="auto", show_label=False, container=False,
                                                          scale=0, min_width=210, elem_classes="tts-lang")
                                    up_btn = gr.UploadButton("⇪", file_types=[".txt", ".md"], elem_classes="tts-up",
                                                             scale=0, min_width=38)
                                    gr.HTML("", elem_classes="tts-fill", scale=1, min_width=10)
                                    split_tag = gr.Button("拆句", elem_classes="tts-split", scale=0, min_width=96)
                                    gen_btn = gr.Button("✦ 生成音频", elem_classes="tts-go", scale=0, min_width=170)
                            lang_tip = gr.HTML("", elem_classes="tts-tipbox")
                            status = gr.Markdown(elem_classes="tts-status")
                            with gr.Column(elem_classes="tts-result") as result_box:
                                full_audio = gr.Audio(label="生成结果", type="filepath", interactive=False)

                            # 长文模式：句子列表（逐句改字 / 重新生成 / 换音色 / 插句删句）
                            with gr.Column(visible=False, elem_classes="tts-long") as long_box:
                                with gr.Row(elem_classes="toolbar"):
                                    per_group = gr.Slider(1, 5, value=1, step=1, scale=1, label="每段几句")
                                    split_btn = gr.Button("重新拆句", elem_classes="ghost", scale=0, min_width=100)
                                    merge_btn = gr.Button("合并成完整音频", elem_classes="ghost", scale=0, min_width=150)
                                gr.HTML("<span class='nw-hint'>重新拆句会按上面的文字重排：文字没变的句子保留音频，"
                                        "在列表里插入的句子会消失。逐句都满意了点「合并成完整音频」，上面才会出现完整的一条。</span>")
                                live = gr.Audio(label="实时试听（每生成一句自动播放）", autoplay=True, interactive=False,
                                                type="filepath")
                                sents = gr.State([])

                                @gr.render(inputs=[sents])
                                def show_rows(items):
                                    if not items:
                                        gr.HTML("<div class='list-hint'>在上面输入文字，点「生成音频」或「重新拆句」，这里会一句一行列出来。</div>")
                                        return
                                    with gr.Row(elem_classes="toolbar"):
                                        gr.HTML(f"<div class='nw-sec'>句子列表 · {len(items)} 句</div>", scale=0, min_width=140)
                                        gr.HTML("<div class='list-hint'>点进句子直接改字，再点「重新生成」；"
                                                "没改字点它就换一种读法。</div>", scale=1)
                                        b_top = gr.Button("＋ 最前面插一句", size="sm", key="instop", elem_classes="mini",
                                                          scale=0, min_width=120)
                                    b_top.click(
                                        lambda items, name, scr, sd: insert_one(0, items, name, scr, sd),
                                        [sents, proj_name, script, b_seed], [sents, status],
                                    )
                                    for i, it in enumerate(items):
                                        if not it["audio"]:
                                            st = "<span class='sent-st st-todo'>未生成</span>"
                                        elif it["gen_text"] != it["text"]:
                                            st = "<span class='sent-st st-old'>字改了</span>"
                                        elif _item_voice(it) != it.get("gen_voice"):
                                            st = "<span class='sent-st st-old'>音色改了</span>"
                                        else:
                                            st = "<span class='sent-st st-ok'>已生成</span>"
                                        with gr.Row(equal_height=True, key=f"row{i}", elem_classes="sent-row"):
                                            gr.HTML(f"<div class='sent-no'>{i + 1}</div>{st}", min_width=62, scale=0, key=f"no{i}")
                                            tb = gr.Textbox(
                                                value=it["text"], lines=2, show_label=False, scale=5, key=f"tb{i}",
                                                interactive=True,  # @gr.render 里不显式声明会被当成只读
                                                preserved_by_key=None,  # 以 sents 为准，否则删句/重新拆句后旧文字会留在错的行
                                            )
                                            gr.Audio(
                                                value=it["audio"], interactive=False, show_label=False, scale=3, key=f"au{i}",
                                                preserved_by_key=None,
                                            )
                                            with gr.Column(scale=0, min_width=150, key=f"ops{i}", elem_classes="sent-ops"):
                                                iv = _item_voice(it)
                                                vd = gr.Dropdown(
                                                    [(VOICE_FOLLOW, VOICE_FOLLOW), *lib_choices()], value=iv or VOICE_FOLLOW, show_label=False,
                                                    container=False, key=f"vo{i}", interactive=True, preserved_by_key=None,
                                                    elem_classes=["pill", "sent-voice"] + (["own"] if iv else []),
                                                )
                                                b_re = gr.Button("重新生成", size="sm", key=f"re{i}", elem_classes="ghost")
                                                with gr.Row(key=f"opr{i}"):
                                                    b_ins = gr.Button("＋ 插句", size="sm", key=f"ins{i}", elem_classes="mini",
                                                                      min_width=40)
                                                    b_del = gr.Button("删除", size="sm", key=f"del{i}", elem_classes=["mini", "del"],
                                                                      min_width=40)
                                        vd.input(
                                            lambda v, items, name, scr, i=i: set_item_voice(i, v, items, name, scr),
                                            [vd, sents, proj_name, script], [sents, status], show_progress="hidden",
                                        )
                                        tb.blur(
                                            lambda t, items, name, scr, i=i: edit_text(i, t, items, name, scr),
                                            [tb, sents, proj_name, script], None, show_progress="hidden",
                                        )
                                        # 没改字时 regen_one 会自动换种子 = 换一种读法
                                        b_re.click(
                                            lambda t, items, name, scr, *v, i=i: regen_one(i, t, items, name, scr, *v, False),
                                            [tb, sents, proj_name, script, *VOICE, b_seed, lang_dd], [sents, status],
                                        )
                                        b_ins.click(
                                            lambda items, name, scr, sd, i=i: insert_one(i + 1, items, name, scr, sd),
                                            [sents, proj_name, script, b_seed], [sents, status],
                                        )
                                        b_del.click(
                                            lambda items, name, scr, i=i: delete_one(i, items, name, scr),
                                            [sents, proj_name, script], [sents, status],
                                        )


                        # ── 右：调试台 / 生成历史 ──
                        with gr.Column(elem_id="tts-right", scale=0, min_width=360):
                            side_tab = gr.Radio(["调试台", "生成历史"], value="调试台", show_label=False, container=False,
                                                elem_classes="tts-tabs")
                            with gr.Column(visible=True) as side_debug:
                                with gr.Row(elem_classes="toolbar"):
                                    gr.HTML("<div class='tts-sec'>音色</div>", scale=1)
                                    reset_btn = gr.Button("↺ 参数重置", elem_classes="tts-reset", scale=0, min_width=0)
                                with gr.Column(elem_classes="tts-vcard"):
                                    voice_card = gr.HTML(tts_voice_card(None))
                                    voice_dd = gr.Dropdown(tts_voice_choices(ALL_MODELS), value=None, show_label=False,
                                                           container=False, elem_classes="pill")
                                voice_prev = gr.Audio(label="原声试听", type="filepath", interactive=False, visible=False,
                                                      elem_classes="tts-prev", buttons=[])
                                with gr.Accordion("音色效果调节（语气）", open=False, elem_classes="tts-acc") as tone_acc:
                                    tone_dd = gr.Radio(tone_choices(), value=TONE_NONE, show_label=False, container=False,
                                                       elem_classes="chips")
                                    tone_text = gr.Textbox(show_label=False, container=False, lines=2, max_lines=4,
                                                           placeholder="语气提示词，留空 = 跟参考音频走；改了字就变成「自定义」",
                                                           elem_classes="tone-line")
                                    b_cfg = gr.Slider(1, 4, value=2, step=0.5, label="语气强度",
                                                      info="越大越贴提示词，音色也越容易跑，建议 1.5～2")
                                    with gr.Row(elem_classes="toolbar"):
                                        tone_name = gr.Textbox(show_label=False, container=False, scale=2,
                                                               placeholder="新语气名字")
                                        tone_save_btn = gr.Button("存进语气", elem_classes="ghost", size="sm", scale=0,
                                                                  min_width=90)
                                    tone_del = gr.Button("删除选中的语气", elem_classes="warn", size="sm")
                                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速", elem_classes="tts-sl")
                                pitch = gr.Slider(-12, 12, value=0, step=1, label="声调", elem_classes="tts-sl")
                                vol = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="音量", elem_classes="tts-sl")
                                gap = gr.Slider(0, 1.5, value=0.3, step=0.05, label="句间停顿（秒）", elem_classes="tts-sl")
                                with gr.Accordion("更多：种子 · 清空", open=False, elem_classes="tts-acc"):
                                    b_seed = gr.Number(value=42, precision=0, label="种子（整篇共用）")
                                    clear_btn = gr.Button("清空已生成音频", elem_classes="warn", size="sm")
                                    with gr.Column(visible=False) as clear_confirm:  # 页面内二次确认，不依赖浏览器弹窗
                                        gr.Markdown("确定清空吗？旧文件不删，挪到项目里的「旧音频_时间」文件夹。")
                                        with gr.Row():
                                            clear_yes = gr.Button("确认清空", variant="stop", size="sm")
                                            clear_no = gr.Button("取消", size="sm")
                            with gr.Column(visible=False) as side_hist:
                                hist_html = gr.HTML(tts_history("稿子1"))

                # ═════ 音色设计（照 MiniMax 的样子） ═════
                with gr.Tab("音色设计", id="trial", elem_id="vd-page"):
                    gr.HTML("<div class='vd-head'><h1>音色设计</h1><p>使用文字描述设计属于你的全新音色</p></div>")
                    gr.HTML("<div class='vd-label'>提示词</div>")
                    with gr.Column(elem_classes="vd-box"):
                        vd_prompt = gr.Textbox(VD_PRESETS[0][3], lines=4, max_lines=8, show_label=False, container=False,
                                               elem_classes="vd-text", placeholder="描述你想要的声音：年龄、性别、音色、语速、情绪……")
                        with gr.Row(elem_classes="vd-chips"):
                            vd_shuf = gr.Button("", elem_id="vd-shuffle", elem_classes="vd-shuf", scale=0, min_width=46)
                            vd_chips = [gr.Button(n, elem_id=f"vd-chip-{i}", elem_classes=["vd-chip", "on"] if i == 0 else ["vd-chip"],
                                                  scale=0, min_width=60) for i, (n, *_r) in enumerate(VD_PRESETS)]
                    with gr.Row(elem_classes="vd-row"):
                        gr.HTML("<div class='vd-label'>试听文本</div>")
                        vd_auto = gr.Button("自动生成", elem_id="vd-auto", elem_classes="vd-auto", scale=0, min_width=130)
                    with gr.Column(elem_classes="vd-box"):
                        vd_text = gr.Textbox(VD_PRESETS[0][4], lines=3, max_lines=6, max_length=VD_MAX, show_label=False,
                                             container=False, elem_classes="vd-text", placeholder="用来试听的一段话")
                        with gr.Row(elem_classes="vd-foot"):
                            vd_cnt = gr.HTML(vd_count(VD_PRESETS[0][4]), scale=1, min_width=120)
                            vd_clear = gr.Button("", elem_id="vd-clear", elem_classes="vd-icon", scale=0, min_width=30)
                    with gr.Row(elem_classes="vd-actions"):
                        gr.HTML("", scale=1)
                        vd_engine = gr.Dropdown(vd_engine_choices(), value=vd_engine_default(), show_label=False,
                                                container=False, scale=0, min_width=230, elem_classes=["pill", "vd-engine"])
                        vd_go = gr.Button("生成", elem_classes="vd-primary", scale=0, min_width=160)
                    vd_status = gr.Markdown(elem_classes="vd-status")
                    gr.HTML("<div class='vd-label'>生成音色结果</div>")
                    vd_cards, vd_radios = [], []
                    with gr.Row(elem_classes="vd-results"):
                        for k in range(3):
                            with gr.Row(elem_classes="vd-card", equal_height=True):
                                vd_cards.append(gr.HTML(vd_card(k), scale=1))
                                vd_radios.append(gr.Button("", elem_classes="vd-radio", scale=0, min_width=32,
                                                           interactive=False))
                    with gr.Row(elem_classes="vd-actions"):
                        gr.HTML("", scale=1)
                        vd_name = gr.Textbox(show_label=False, container=False, scale=0, min_width=280, elem_classes="vd-name",
                                             placeholder="给选中的音色起个名字（可不填）")
                        vd_ok = gr.Button("确认选择", elem_id="vd-ok", elem_classes="vd-primary", scale=0, min_width=160)
                    vd_cands = gr.State([None, None, None])
                    vd_sel = gr.State(None)
                    vd_on = gr.State(0)

                # ═════ 声音克隆（照 MiniMax 的样子） ═════
                with gr.Tab("声音克隆", id="clone", elem_id="vc-page"):
                    gr.HTML("<div class='vd-head'><h1>声音克隆</h1><p>朗读一段文字，即可克隆你的专属声音</p></div>")
                    gr.HTML("<div class='vd-label'>选择场景并朗读</div>")
                    with gr.Row(elem_classes="vc-chips"):
                        vc_chips = [gr.Button(n, elem_id=f"vc-chip-{i}", elem_classes=["vc-chip", "on"] if i == 0 else ["vc-chip"],
                                              scale=0, min_width=60)
                                    for i, n in enumerate(["随机"] + [x[0] for x in VC_SCENES])]
                    with gr.Column(elem_classes="vd-box"):
                        vc_read = gr.Textbox(VC_EXTRA[0], lines=1, max_lines=5, show_label=False, container=False,
                                             elem_classes=["vd-text", "vc-read"],
                                             placeholder="照着念的文字；上传自己的音频时，改成音频里说的原话")
                        vc_ref = gr.Audio(sources=["microphone", "upload"], type="filepath", show_label=False,
                                          elem_classes="vc-rec", waveform_options=gr.WaveformOptions(show_recording_waveform=True))
                        gr.HTML("<div class='vc-tip'>点麦克风开始录音，照着上面的文字念一遍（10 秒左右）；也可以上传一段 5～15 秒的干净人声</div>")
                    gr.HTML("<div class='vd-label'>试听文本</div>")
                    with gr.Column(elem_classes="vd-box"):
                        with gr.Row(elem_classes="vc-langrow"):
                            vc_lang = gr.Dropdown(LANG_CHOICES, value="Chinese", show_label=False, container=False,
                                                  elem_classes="vc-lang")
                        vc_text = gr.Textbox(VC_DEFAULT_TRY, lines=3, max_lines=6, max_length=VD_MAX, show_label=False,
                                             container=False, elem_classes="vd-text", placeholder="用克隆出来的声音读这段话")
                        with gr.Row(elem_classes="vd-foot"):
                            vc_cnt = gr.HTML(vd_count(VC_DEFAULT_TRY), scale=1, min_width=120)
                            vc_clear = gr.Button("", elem_id="vc-clear", elem_classes="vd-icon", scale=0, min_width=30)
                    vc_agree = gr.Checkbox(False, label=VC_CONSENT, elem_classes="vc-agree", container=False)
                    with gr.Row(elem_classes="vd-actions"):
                        gr.HTML("", scale=1)
                        vc_engine = gr.Dropdown(vc_engine_choices(), value=vc_engine_default(), show_label=False,
                                                container=False, scale=0, min_width=230, elem_classes=["pill", "vd-engine"])
                        vc_go = gr.Button("生成", elem_classes="vd-primary", scale=0, min_width=160)
                    vc_status = gr.Markdown(elem_classes="vd-status")
                    with gr.Column(elem_classes="vc-result"):
                        with gr.Row(elem_classes="vd-results"):
                            with gr.Row(elem_classes="vd-card", equal_height=True):
                                vc_res = gr.HTML(vc_card(), scale=1)
                    with gr.Row(elem_classes="vd-actions", visible=False) as vc_save_row:
                        gr.HTML("", scale=1)
                        vc_name = gr.Textbox(show_label=False, container=False, scale=0, min_width=280, elem_classes="vd-name",
                                             placeholder="给这个声音起个名字（可不填）")
                        vc_ok = gr.Button("保存到音色库", elem_id="vc-ok", elem_classes="vd-primary", scale=0, min_width=170)
                    vc_cand = gr.State(None)

                # ═════ 音色库 ═════
                with gr.Tab("音色库", id="lib"):
                    with gr.Row(elem_classes="topbar"):
                        gr.HTML("<div class='nw-title'>音色库</div>", scale=0, min_width=90)
                        gr.HTML("<span class='nw-hint'>点「用这个音色」直接回文本朗读；删除的音色挪到 voices/已删除，能找回。</span>",
                                scale=1)
                    lib_state = gr.State(lib_list())

                    @gr.render(inputs=[lib_state])
                    def voice_cards(names):
                        if not names:
                            gr.Markdown("*音色库还是空的：去「音色设计」试出满意的声音，再存进来。*")
                            return
                        with gr.Row(elem_classes="voice-grid"):
                            for name in names:
                                v = lib_read(name)
                                eng = v.get("engine") or "breeze"
                                vk = v.get("kind") or ("clone" if v.get("ref_audio") else "design")
                                kind = ENGINE_SHORT.get(eng, eng) + " · " + {"clone": "克隆", "design": "描述",
                                                                             "speaker": "预设"}.get(vk, vk)
                                with gr.Column(elem_classes="voice-card", key=f"vc-{name}", min_width=230):
                                    gr.HTML(_art(name) + f"<div class='vc-name'>{html.escape(name)}"
                                            f"<span class='vc-tag'>{kind}</span></div>"
                                            f"<div class='vc-desc'>{html.escape((v.get('ref_text') or v.get('instruction') or v.get('speaker') or '')[:40])}</div>")
                                    if v.get("ref_audio"):
                                        try:  # 单个音色的音频出问题时只少一个播放器，不让整页挂掉
                                            gr.Audio(value=v["ref_audio"], interactive=False, show_label=False,
                                                     key=f"va-{name}", preserved_by_key=None)
                                        except Exception as e:
                                            gr.Markdown(f"*参考音频读不了：{html.escape(str(e))[:60]}*", key=f"ve-{name}")
                                    with gr.Row(key=f"vr-{name}"):
                                        b_use = gr.Button("用这个音色", elem_classes="go", size="sm", key=f"vu-{name}",
                                                          min_width=100)
                                        b_vdel = gr.Button("删除", elem_classes="warn", size="sm", key=f"vd-{name}",
                                                           scale=0, min_width=64)
                                b_use.click(lambda n=name: (n, *go_page("home")), None, [voice_dd, pages, *nav_btns], show_progress="hidden")
                                b_vdel.click(lambda n=name: lib_delete(n), None, [voice_dd, status]).then(
                                    refresh_libs, None, [lib_state]).then(lambda x: list(x), [sents], [sents])

                    with gr.Accordion("新建音色 / 查看当前音色参数", open=False, elem_classes="card"):
                        gr.HTML("<div class='nw-label'>用哪个模型</div>")
                        nv_engine = gr.Radio([(e.name, e.id) for e in E.ENGINES.values()], value="qwen3",
                                             show_label=False, container=False, elem_classes="chips")
                        nv_kind = gr.Radio([E.VOICE_KIND_NAMES[k] for k in E.ENGINES["qwen3"].voice_kinds],
                                           value=E.VOICE_KIND_NAMES["clone"], label="音色类型")
                        nv_tip = gr.Markdown()
                        b_ref = gr.Audio(value=None, type="filepath", label="参考音频（克隆，5～15 秒最好）")
                        b_ref_text = gr.Textbox(label="参考音频的准确文字稿（Qwen3 可留空，但填了更像）")
                        b_ins = gr.Textbox(label="音色描述", lines=2, visible=False,
                                           placeholder="比如：二十多岁的男生，声音清爽温和，语速适中")
                        nv_speaker = gr.Dropdown([], label="音色 ID（下拉选，或直接粘贴）", allow_custom_value=True,
                                                 visible=False)
                        with gr.Row(elem_classes="toolbar"):
                            new_name = gr.Textbox(show_label=False, container=False, placeholder="新音色名字，比如：讲解男声-2号", scale=3)
                            save_btn = gr.Button("存进音色库", elem_classes="go", scale=0, min_width=130)
                        lib_info = gr.Markdown()

                # ═════ 设置 ═════
                with gr.Tab("设置", id="set"):
                    with gr.Row(elem_classes="topbar"):
                        gr.HTML("<div class='nw-title'>设置</div>", scale=0, min_width=70)
                        gr.HTML("<span class='nw-hint'>这里管「用哪些模型」：本地模型点了才从魔搭下载；第三方配音的 Key 只存在这台电脑上。"
                                "想让别的程序调用这里的配音，去左边「开放接口」。</span>", scale=1)
                    _s0 = E.settings_load()
                    set_info = gr.Markdown(elem_classes="set-info")
                    with gr.Column(elem_classes="card"):
                        with gr.Row(elem_classes="toolbar"):
                            gpu_md = gr.Markdown(E.gpu_state())
                            gpu_btn = gr.Button("释放显卡", elem_classes="ghost", size="sm", scale=0, min_width=100)
                        gr.HTML("<div class='nw-hint'>本地模型共用一块显卡，同一时间只放一个；用到别的本地模型时会自动换，"
                                "第一次换要加载几十秒。要用别的吃显卡的软件前可以先点「释放显卡」。</div>")

                    gr.HTML("<div class='nw-sec'>本地模型（魔搭下载）</div>")
                    with gr.Column(elem_classes="card"):
                        st_htmls = []
                        for c in E.CATALOG:
                            with gr.Row(elem_classes="mrow", equal_height=True):
                                gr.HTML(f"<div class='mname'>{html.escape(c['name'])}<small>{html.escape(c['desc'])}"
                                        f" · {c['size']} · {c['repo']}</small></div>", scale=3)
                                st_htmls.append(gr.HTML(_st_html(c["id"]), scale=1, min_width=150))
                                b_dl = gr.Button("下载", elem_classes="go", size="sm", scale=0, min_width=70)
                                b_pause = gr.Button("暂停", elem_classes="mini", size="sm", scale=0, min_width=60)
                                b_mdel = gr.Button("删除", elem_classes=["mini", "del"], size="sm", scale=0, min_width=60)
                            b_dl.click(lambda m=c["id"]: E.download_start(m), None, set_info, concurrency_limit=None)
                            b_pause.click(lambda m=c["id"]: E.download_cancel(m), None, set_info, concurrency_limit=None)
                            b_mdel.click(lambda m=c["id"]: E.model_delete(m), None, set_info)
                        gr.HTML("<div class='nw-hint'>已经有模型文件的话，把文件夹放到（或链接到）下面这个文件夹里，"
                                "名字和魔搭上一样（比如 Breeze-TTS-2），就算装好了，不用重新下。</div>")
                        with gr.Row(elem_classes="toolbar"):
                            set_model_dir = gr.Textbox(_s0["model_dir"], label="模型下载到哪个文件夹", scale=3)
                            set_preload = gr.Checkbox(_s0.get("preload") == "breeze", label="启动时预加载 Breeze", scale=1)
                            b_paths = gr.Button("保存", elem_classes="ghost", size="sm", scale=0, min_width=90)

                    gr.HTML("<div class='nw-sec'>接入第三方配音 API</div>")
                    with gr.Column(elem_classes="card"):
                        gr.HTML("<div class='nw-label'>MiniMax 语音</div>")
                        with gr.Row():
                            mm_key = gr.Textbox(_s0["minimax"]["api_key"], label="API Key", type="password", scale=3)
                            mm_base = gr.Textbox(_s0["minimax"]["base_url"], label="接口地址（国内 api.minimaxi.com）", scale=2)
                            mm_model = gr.Textbox(_s0["minimax"]["model"], label="模型", scale=1)
                        with gr.Row(elem_classes="toolbar"):
                            mm_spk = gr.Dropdown(E.MINIMAX_SPEAKERS, value=E.MINIMAX_SPEAKERS[0], allow_custom_value=True,
                                                 label="试听用的音色", scale=2)
                            mm_test = gr.Button("保存并试听", elem_classes="ghost", size="sm", scale=0, min_width=110)
                        gr.HTML("<div class='nw-divider'></div><div class='nw-label'>豆包 / 火山引擎语音合成</div>")
                        with gr.Row():
                            vc_appid = gr.Textbox(_s0["volc"]["appid"], label="AppID", scale=1)
                            vc_token = gr.Textbox(_s0["volc"]["token"], label="Access Token", type="password", scale=2)
                            vc_cluster = gr.Textbox(_s0["volc"]["cluster"], label="Cluster（复刻音色用 volcano_icl）", scale=1)
                        with gr.Row(elem_classes="toolbar"):
                            vc_spk = gr.Dropdown(E.VOLC_SPEAKERS, value=E.VOLC_SPEAKERS[0], allow_custom_value=True,
                                                 label="试听用的音色（voice_type）", scale=2)
                            vc_test = gr.Button("保存并试听", elem_classes="ghost", size="sm", scale=0, min_width=110)
                        gr.HTML("<div class='nw-divider'></div><div class='nw-label'>OpenAI 兼容接口</div>")
                        with gr.Row():
                            oa_base = gr.Textbox(_s0["openai"]["base_url"], label="接口地址（到 /v1 为止）", scale=2,
                                                 placeholder="https://api.openai.com/v1")
                            oa_key = gr.Textbox(_s0["openai"]["api_key"], label="API Key", type="password", scale=2)
                            oa_model = gr.Textbox(_s0["openai"]["model"], label="模型", scale=1)
                        with gr.Row(elem_classes="toolbar"):
                            oa_spk = gr.Dropdown(E.OPENAI_SPEAKERS, value=E.OPENAI_SPEAKERS[0], allow_custom_value=True,
                                                 label="试听用的音色（voice）", scale=2)
                            oa_test = gr.Button("保存并试听", elem_classes="ghost", size="sm", scale=0, min_width=110)
                        b_api = gr.Button("保存第三方 API 设置", elem_classes="go", size="sm")
                        api_audio = gr.Audio(label="试听", type="filepath", autoplay=True, interactive=False)

                    set_tick = gr.Timer(2.0)

                # ═════ 开放接口（别的程序 / Agent 调用这里的配音） ═════
                with gr.Tab("开放接口", id="api"):
                    with gr.Row(elem_classes="topbar"):
                        gr.HTML("<div class='nw-title'>开放接口</div>", scale=0, min_width=100)
                        gr.HTML("<span class='nw-hint'>让别的软件、脚本、Agent 调用这里的模型和音色库。"
                                "格式和 OpenAI 的语音接口一样。</span>", scale=1)
                    api_info = gr.Markdown(elem_classes="set-info")
                    with gr.Column(elem_classes="card"):
                        _srv = api_server.server_conf()
                        gr.HTML("<div class='nw-hint'>支持「OpenAI 兼容 TTS」的工具，把地址和 Key 换成下面这两个就能用。"
                                "接口文档：<a href='/v1/docs' target='_blank'>/v1/docs</a></div>")
                        with gr.Row(elem_classes="toolbar"):
                            srv_on = gr.Checkbox(_srv["api_enabled"], label="开启接口", scale=1)
                            srv_lan = gr.Checkbox(_srv["lan"], label="允许局域网其他电脑访问（重启生效，网页也会要求登录）",
                                                  scale=2)
                            b_srv = gr.Button("保存", elem_classes="ghost", size="sm", scale=0, min_width=80)
                        with gr.Row():
                            srv_url = gr.Textbox(api_base(), label="接口地址（base_url）", interactive=False,
                                                 buttons=["copy"], scale=2)
                            srv_key = gr.Textbox(_srv["api_key"], label="接口 Key", interactive=False, type="password",
                                                 buttons=["copy"], scale=2)
                            b_key = gr.Button("换一个 Key", elem_classes="mini", size="sm", scale=0, min_width=90)
                        srv_code = gr.Code(api_examples(), language="shell", label="调用示例", interactive=False,
                                           buttons=["copy"])
                    gr.HTML("<div class='nw-sec'>能调用的音色</div>")
                    with gr.Column(elem_classes="card"):
                        srv_voices = gr.Markdown(api_voice_list())
                        b_vrefresh = gr.Button("刷新", elem_classes="mini", size="sm")

                # ═════ MCP 连接（让 Claude、豆包等 Agent 直接调用） ═════
                with gr.Tab("MCP 连接", id="mcp"):
                    with gr.Row(elem_classes="topbar"):
                        gr.HTML("<div class='nw-title'>MCP 连接</div>", scale=0, min_width=110)
                        gr.HTML("<span class='nw-hint'>把这里的配音接到 Claude、Cursor、Trae、豆包等 Agent 上，"
                                "在对话里直接说「用某某音色把这段稿子配出来」。</span>", scale=1)
                    with gr.Column(elem_classes="card"):
                        mcp_stat = gr.HTML(mcp_status_html())
                    gr.HTML("<div class='nw-sec'>Agent 能用的工具</div>")
                    with gr.Column(elem_classes="card"):
                        gr.HTML(MCP_TOOLS_HTML)
                    gr.HTML("<div class='nw-sec'>怎么接入</div>")
                    with gr.Column(elem_classes="card"):
                        _cfgs = mcp_configs()
                        _first = next(iter(_cfgs))
                        mcp_pick = gr.Radio(list(_cfgs), value=_first, show_label=False, elem_classes="mcp-pick")
                        mcp_tip = gr.HTML(f"<div class='nw-hint'>{html.escape(_cfgs[_first][0])}</div>")
                        _f = _cfgs[_first][2]
                        with gr.Column(visible=True) as mcp_form:  # 和豆包 / WorkBuddy 的表单一一对应
                            with gr.Row():
                                mf_name = gr.Textbox(_f["name"], label="服务器名称", interactive=False, lines=1, max_lines=1, elem_classes="mcp-field", buttons=["copy"])
                                mf_type = gr.Textbox(_f["transport"], label="传输类型（下拉选 HTTP）", interactive=False, lines=1, max_lines=1, elem_classes="mcp-field")
                            mf_url = gr.Textbox(_f["url"], label="服务器 URL", interactive=False, lines=1, max_lines=1, elem_classes="mcp-field", buttons=["copy"])
                            with gr.Row():
                                mf_hn = gr.Textbox(_f["h_name"], label="Header 名称", interactive=False, lines=1, max_lines=1, elem_classes="mcp-field",
                                                   buttons=["copy"])
                                mf_hv = gr.Textbox(_f["h_value"], label="Header 值（Bearer 加空格加 Key）",
                                                   interactive=False, lines=1, max_lines=1, elem_classes="mcp-field", type="password", buttons=["copy"])
                            mf_urlk = gr.Textbox(_f["url_key"], label="没有 Headers 可填的软件：URL 直接填这个（Key 已拼在里面）",
                                                 interactive=False, lines=1, max_lines=1, elem_classes="mcp-field", type="password", buttons=["copy"])
                        mcp_code = gr.Code("", language=None, interactive=False, buttons=["copy"],
                                           show_label=False, visible=False)

                        def _mcp_pick(name):
                            tip, lang, code = mcp_configs()[name]
                            tip = f"<div class='nw-hint'>{html.escape(tip)}</div>"
                            if lang == "form":
                                return (tip, gr.Column(visible=True), code["name"], code["url"], code["h_value"],
                                        code["url_key"], gr.Code(visible=False))
                            return (tip, gr.Column(visible=False), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                                    gr.Code(code, visible=True))
                        mcp_pick.change(_mcp_pick, [mcp_pick],
                                        [mcp_tip, mcp_form, mf_name, mf_url, mf_hv, mf_urlk, mcp_code],
                                        show_progress="hidden")
                    mcp_tick = gr.Timer(3.0)
                    mcp_tick.tick(mcp_status_html, None, [mcp_stat], show_progress="hidden")

    VOICE = [voice_dd, b_cfg, tone_text]
    script_acc = None  # 旧版「整篇稿子」折叠框已去掉

    for pid_btn, (pid, _) in zip(nav_btns, PAGES):
        pid_btn.click(lambda p=pid: go_page(p), None, [pages, *nav_btns], show_progress="hidden")

    # 先在服务端标成展开再收起：用户手动点开过的话，直接发 open=False 服务端会认为没变化、不生效
    # 工具条上的「拆句」：拆好后打开长文模式，下面直接出句子列表
    split_tag.click(do_split, [script, proj_name, b_seed, per_group, sents], [sents, status]).then(
        lambda: (True, gr.Column(visible=True), gr.Column(visible=False)), None, [longmode, long_box, result_box],
        show_progress="hidden").then(
        lambda n: gr.update(choices=projects_list(), value=n), [proj_name], [proj_name])
    split_btn.click(do_split, [script, proj_name, b_seed, per_group, sents], [sents, status]).then(
        lambda n: gr.update(choices=projects_list(), value=n), [proj_name], [proj_name])
    gen_btn.click(tts_generate, [script, proj_name, b_seed, per_group, sents, longmode, voice_dd, tone_dd, b_cfg,
                                 tone_text, lang_dd, gap, speed, pitch, vol],
                  [sents, status, live, full_audio, hist_html]).then(
        lambda n: gr.update(choices=projects_list(), value=n), [proj_name], [proj_name])
    merge_btn.click(tts_remerge, [sents, gap, speed, proj_name, pitch, vol], [full_audio, status, hist_html]).then(
        lambda: gr.Column(visible=True), None, [result_box], show_progress="hidden")
    # MiniMax 式编辑框：字数、清空、上传、插入停顿 / 语气词、长文模式、语言、模型筛音色
    script.change(tts_count, [script], [count_html], show_progress="hidden", queue=False)
    erase_btn.click(lambda: "", None, [script], show_progress="hidden")
    up_btn.upload(tts_upload, [up_btn], [script])
    pause_btn.click(None, None, None, js=ins_js("<#0.5#>"))
    laugh_btn.click(None, None, None, js=ins_js("[笑]"))
    emo_btn.click(lambda: gr.Accordion(open=True), None, [tone_acc])
    longmode.change(lambda on: (gr.Column(visible=bool(on)), gr.Column(visible=not on)), [longmode],
                    [long_box, result_box], show_progress="hidden")
    model_dd.change(tts_model_pick, [model_dd, voice_dd], [voice_dd], show_progress="hidden")
    voice_dd.change(tts_voice_card, [voice_dd], [voice_card], show_progress="hidden")
    lang_dd.change(tts_lang_tip, [voice_dd, lang_dd], [lang_tip], show_progress="hidden")
    voice_dd.change(tts_lang_tip, [voice_dd, lang_dd], [lang_tip], show_progress="hidden")
    reset_btn.click(lambda: (1.0, 0, 1.0, 0.3), None, [speed, pitch, vol, gap], show_progress="hidden")
    side_tab.change(lambda t, n: (gr.Column(visible=t == "调试台"), gr.Column(visible=t == "生成历史"),
                                  tts_history(n) if t == "生成历史" else gr.skip()),
                    [side_tab, proj_name], [side_debug, side_hist, hist_html], show_progress="hidden")
    clear_btn.click(lambda: gr.update(visible=True), None, clear_confirm)
    clear_no.click(lambda: (gr.update(visible=False), "已取消，没有清空。"), None, [clear_confirm, status])
    clear_yes.click(lambda: gr.update(visible=False), None, clear_confirm).then(
        clear_audio, [sents, proj_name, script], [sents, status],
    )
    voice_dd.change(lib_pick, [voice_dd], [b_ref, b_ref_text, b_ins, b_seed, status]).then(
        voice_preview, [voice_dd], [voice_prev])
    save_btn.click(lib_save, [new_name, b_ref, b_ref_text, b_ins, b_seed, nv_engine, nv_kind, nv_speaker],
                   [voice_dd, lib_info]).then(refresh_libs, None, [lib_state]).then(lambda x: list(x), [sents], [sents])
    nv_engine.change(new_voice_form, [nv_engine], [nv_kind, nv_speaker, nv_tip])
    nv_kind.change(new_voice_kind, [nv_kind], [b_ref, b_ref_text, b_ins, nv_speaker])
    # 用 .input（只响应手动选择）：载入项目时程序改语气，不要拿语气库的原文盖掉项目里改过的提示词
    tone_dd.input(tone_pick, [tone_dd], [tone_text])
    tone_text.change(tone_match, [tone_text], [tone_dd], show_progress="hidden")
    tone_save_btn.click(tone_save, [tone_name, tone_text], [tone_dd, status])
    tone_del.click(tone_delete, [tone_dd], [tone_dd, tone_text, status])
    PROJ_OUT = [proj_name, script, sents, voice_dd, tone_dd, tone_text, b_cfg, status]
    proj_name.input(proj_switch, [proj_name], PROJ_OUT).then(tts_history, [proj_name], [hist_html])
    demo.load(auto_load, None, PROJ_OUT).then(tts_history, [proj_name], [hist_html])
    # 查更新要联网，别占着队列（默认一次只跑一个事件）
    demo.load(upd_check, None, [upd_html, upd_btn], show_progress="hidden", concurrency_limit=None)
    upd_btn.click(upd_go, None, [upd_html, upd_btn], show_progress="hidden", concurrency_limit=None)

    API_IN = [mm_key, mm_base, mm_model, vc_appid, vc_token, vc_cluster, oa_base, oa_key, oa_model]
    b_api.click(save_api, API_IN, set_info, concurrency_limit=None)
    for _eid, _btn, _spk in (("minimax", mm_test, mm_spk), ("volc", vc_test, vc_spk), ("openai", oa_test, oa_spk)):
        _btn.click(save_api, API_IN, set_info, concurrency_limit=None).then(
            lambda sp, e=_eid: api_test(e, sp), [_spk], [api_audio, set_info], concurrency_limit=None)
    b_paths.click(save_paths, [set_model_dir, set_preload], set_info, concurrency_limit=None)
    gpu_btn.click(gpu_release, None, gpu_md)
    b_srv.click(save_server, [srv_on, srv_lan], [api_info, srv_url, srv_code], concurrency_limit=None)
    b_key.click(regen_key, None, [srv_key, api_info], concurrency_limit=None)
    b_vrefresh.click(api_voice_list, None, srv_voices, concurrency_limit=None)
    set_tick.tick(settings_tick, None, [gpu_md, *st_htmls], show_progress="hidden", queue=False)

    for _i, _c in enumerate(vd_chips):
        _c.click(lambda i=_i: vd_pick(i), None, [vd_prompt, vd_text, vd_on, *vd_chips], show_progress="hidden")
    vd_shuf.click(vd_shuffle, [vd_prompt], [vd_prompt, vd_text, vd_on, *vd_chips], show_progress="hidden")
    vd_prompt.input(vd_prompt_edit, [vd_prompt, vd_on], [vd_on, *vd_chips], show_progress="hidden")
    vd_auto.click(vd_autotext, [vd_prompt, vd_text], [vd_text], show_progress="hidden")
    vd_text.change(vd_count, [vd_text], [vd_cnt], show_progress="hidden")
    vd_clear.click(lambda: "", None, [vd_text], show_progress="hidden")
    vd_go.click(vd_generate, [vd_prompt, vd_text, vd_engine],
                [*vd_cards, vd_cands, vd_sel, *vd_radios, vd_status, vd_go],
                show_progress="hidden")  # 卡片自己有「生成中…」转圈，不要 Gradio 默认那条紫色进度条盖在上面
    for _k, _r in enumerate(vd_radios):
        _r.click(lambda c, k=_k: vd_select(k, c), [vd_cands], [vd_sel, *vd_radios], show_progress="hidden")
    vd_ok.click(vd_confirm, [vd_name, vd_cands, vd_sel, vd_prompt, vd_text, vd_engine],
                [voice_dd, vd_status, lib_state, vd_name]).then(lambda x: list(x), [sents], [sents])

    for _i, _c in enumerate(vc_chips):
        _c.click(lambda cur, i=_i: vc_scene(i, cur), [vc_read], [vc_read, *vc_chips], show_progress="hidden")
    vc_text.change(vd_count, [vc_text], [vc_cnt], show_progress="hidden")
    vc_clear.click(lambda: "", None, [vc_text], show_progress="hidden")
    vc_go.click(vc_generate, [vc_ref, vc_read, vc_text, vc_lang, vc_engine, vc_agree],
                [vc_res, vc_cand, vc_status, vc_go, vc_save_row], show_progress="hidden")
    vc_ok.click(vc_save, [vc_name, vc_cand, vc_ref, vc_read, vc_engine],
                [voice_dd, vc_status, lib_state, vc_name]).then(lambda x: list(x), [sents], [sents])

if __name__ == "__main__":
    import secrets
    import threading
    import webbrowser

    import uvicorn
    from fastapi import FastAPI

    _srv = api_server.server_conf()
    host = "0.0.0.0" if _srv["lan"] else "127.0.0.1"
    fapp = FastAPI(title="TTS Studio API", docs_url="/v1/docs", openapi_url="/v1/openapi.json")
    api_server.register(fapp, lib_list, lib_read)  # /v1/... 先注册，网页挂在 / 下
    mcp_server.mount(fapp, PORT, _srv["lan"], mcp_key_ok)  # /mcp：Claude、豆包等 Agent 连这里

    from fastapi.responses import FileResponse, PlainTextResponse

    @fapp.get("/_vd/{fn}")
    def _vd_audio(fn: str):  # 音色设计的候选试听
        f = DESIGN_DIR / fn
        if not re.fullmatch(r"[0-9a-f]{32}\.wav", fn) or not f.is_file():
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(f, media_type="audio/wav")
    demo.queue(max_size=8, default_concurrency_limit=1)
    # 开了局域网访问：网页也要登录（用户名随便填，密码 = 接口 Key），免得同一个网里的人随便用
    auth = (lambda u, pw: secrets.compare_digest(pw.encode(), api_server.server_conf()["api_key"].encode())) \
        if _srv["lan"] else None
    fapp = gr.mount_gradio_app(fapp, demo, path="/", allowed_paths=[str(DATA)], theme=THEME, css=CSS, auth=auth,
                               auth_message="用户名随便填，密码是「开放接口」页里的接口 Key")
    print(f"网页：http://127.0.0.1:{PORT}    接口：{api_base()}    接口文档：http://127.0.0.1:{PORT}/v1/docs",
          flush=True)
    if not os.environ.get("TTS_STUDIO_NO_BROWSER"):
        threading.Timer(1.5, webbrowser.open, [f"http://127.0.0.1:{PORT}"]).start()
    uvicorn.run(fapp, host=host, port=PORT, log_level="warning")
