@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem 端口：7861 默认被 Breeze-TTS 的 webui 占着，所以这里固定走 7872
if not defined TTS_STUDIO_PORT set TTS_STUDIO_PORT=7872

rem 依次找可用的 Python：本目录 .venv -> 上级 breeze-tts/.venv（已装好 torch+gradio）
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "..\breeze-tts\.venv\Scripts\python.exe" set "PY=..\breeze-tts\.venv\Scripts\python.exe"

if not defined PY (
  echo 没找到可用的 Python 环境。
  echo   - 想独立安装：双击 install.bat
  echo   - 或者确保 ..\breeze-tts\.venv 存在
  pause
  exit /b 1
)

echo 使用 %PY% 启动，端口 %TTS_STUDIO_PORT%
"%PY%" app.py
pause
