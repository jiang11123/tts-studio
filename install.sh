#!/usr/bin/env bash
# 用法：./install.sh        （默认源）
#       ./install.sh cn     （国内镜像）
set -e
cd "$(dirname "$0")"
[ "$1" = "cn" ] && export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "没找到 python3，请先安装 Python 3.11"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -U pip
if [ "$(uname)" = "Darwin" ]; then
  # Mac 没有 CUDA：能跑 API，本地模型会很慢
  .venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1
else
  .venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
fi
.venv/bin/python -m pip install -r requirements.txt
echo "安装完成。运行 ./start.sh 启动；模型在网页的「设置」里点下载。"
