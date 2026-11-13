#!/usr/bin/env bash
cd "$(dirname "$0")"
# 端口：7861 默认被 Breeze-TTS 的 webui 占着，这里固定走 7872
: "${TTS_STUDIO_PORT:=7872}"
export TTS_STUDIO_PORT

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x ../breeze-tts/.venv/Scripts/python.exe ]; then
  PY=../breeze-tts/.venv/Scripts/python.exe
else
  echo "没找到可用的 Python 环境，先跑 ./install.sh 或确认 ../breeze-tts/.venv 存在"; exit 1
fi

PYTHONIOENCODING=utf-8 exec "$PY" app.py
