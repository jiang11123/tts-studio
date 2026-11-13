"""语速调整：用 ffmpeg atempo 精确变速，音高不变。"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np


def _ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise RuntimeError("没有找到 ffmpeg：pip install imageio-ffmpeg")
    return ffmpeg


def change_pitch(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """升降调（半音），时长不变：先按比例改采样率变调，再用 atempo 把时长拉回来。"""
    semitones = float(semitones)
    if abs(semitones) < 1e-3:
        return audio
    f = 2 ** (semitones / 12)
    fmt = ["-f", "f32le", "-ar", str(sr), "-ac", "1"]
    flt = f"asetrate={sr * f:.2f},aresample={sr},atempo={1 / f:.5f}"
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", *fmt, "-i", "pipe:0", "-filter:a", flt, *fmt, "pipe:1"]
    done = subprocess.run(cmd, input=np.asarray(audio, dtype=np.float32).tobytes(), capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace"))
    return np.frombuffer(done.stdout, dtype=np.float32).copy()


def change_volume(audio: np.ndarray, vol: float) -> np.ndarray:
    """音量倍数，1 = 原样；放大后超出的部分削掉，免得爆音。"""
    vol = float(vol)
    if abs(vol - 1.0) < 1e-3:
        return audio
    return np.clip(np.asarray(audio, dtype=np.float32) * vol, -1.0, 1.0)


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """speed=1.2 表示快 20%，时长变为原来的 1/1.2。输入输出都是单声道 float32。"""
    speed = float(speed)
    if abs(speed - 1.0) < 1e-3:
        return audio
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:  # 没装系统 ffmpeg 就用 imageio-ffmpeg 自带的
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise RuntimeError("没有找到 ffmpeg，无法调整语速：pip install imageio-ffmpeg")
    fmt = ["-f", "f32le", "-ar", str(sr), "-ac", "1"]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", *fmt, "-i", "pipe:0",
           "-filter:a", f"atempo={speed:.4f}", *fmt, "pipe:1"]
    done = subprocess.run(cmd, input=np.asarray(audio, dtype=np.float32).tobytes(), capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace"))
    return np.frombuffer(done.stdout, dtype=np.float32).copy()
