"""把长口播稿拆成一句一句（只依赖标准库，方便单独测试）。"""

from __future__ import annotations

import re

_END = "。！？!?；…"
_CLOSERS = "”」』）)"
# 一句 = 若干非句末字符 + 连续的句末标点 + 紧跟的收尾引号/括号
_SENT_RE = re.compile(rf"[^{_END}]*[{_END}]+[{_CLOSERS}]*|[^{_END}]+$")
_COMMA_SPLIT = re.compile(r"(?<=[，,、：:])")


def _visible_len(s: str) -> int:
    return len(re.sub(rf"[\s{_END}，,、：:{_CLOSERS}]", "", s))


def _split_long(piece: str, max_len: int) -> list[str]:
    """超长句子在逗号处再拆，尽量凑到接近 max_len。"""
    if _visible_len(piece) <= max_len:
        return [piece]
    chunks = [c for c in _COMMA_SPLIT.split(piece) if c]
    out: list[str] = []
    cur = ""
    for c in chunks:
        if cur and _visible_len(cur + c) > max_len:
            out.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


def group_sentences(sents: list[str], n: int) -> list[str]:
    """每 n 句合成一段（跨换行也合并，稿子常常一句一行）。"""
    n = max(1, int(n))
    return ["".join(sents[i:i + n]) for i in range(0, len(sents), n)]


def split_sentences(text: str, max_len: int = 60, min_len: int = 3) -> list[str]:
    """按句末标点和换行拆句。

    - 每个换行视为一个段落边界，段落之间不会合并；
    - 太短（少于 min_len 个字）的碎句并到同段落相邻的句子里；
    - 太长（超过 max_len 个字）的句子在逗号处再拆。
    """
    result: list[str] = []
    for para in text.replace("\r", "").split("\n"):
        para = para.strip()
        if not para:
            continue
        pieces: list[str] = []
        for p in _SENT_RE.findall(para):
            p = p.strip()
            if p:
                pieces.extend(_split_long(p, max_len))
        merged: list[str] = []
        for p in pieces:
            if merged and _visible_len(p) < min_len:
                merged[-1] += p
            elif merged and _visible_len(merged[-1]) < min_len:
                merged[-1] += p
            else:
                merged.append(p)
        result.extend(merged)
    return result
