@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem 用法：install.bat        （默认源）
rem       install.bat cn     （国内镜像，装得快）

if "%1"=="cn" set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

where python >nul 2>nul || (echo 没找到 Python，请先安装 Python 3.11（安装时勾选 Add to PATH） & pause & exit /b 1)

if not exist .venv (
  echo [1/3] 创建虚拟环境 .venv
  python -m venv .venv || (pause & exit /b 1)
)
echo [2/3] 安装 PyTorch（CUDA 12.8，约 3 GB）
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 || (pause & exit /b 1)
echo [3/3] 安装其他依赖
.venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
echo.
echo 安装完成。双击 start.bat 启动；模型在网页的「设置」里点下载。
pause
