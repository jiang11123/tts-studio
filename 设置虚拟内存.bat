@echo off
chcp 65001 >nul
title Pagefile Setup

rem Elevated copy comes back with --elevated and jumps straight to :run,
rem so a self-elevation loop is impossible (no repeated UAC prompts).
if /i "%~1"=="--elevated" goto run

rem fltmc is a reliable admin check (does not depend on the Server service).
fltmc >nul 2>&1
if %errorlevel% equ 0 goto run

echo.
echo Admin rights required. A UAC prompt will pop up - click YES.
echo.
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '--elevated' -Verb RunAs"
if errorlevel 1 (
  echo Elevation cancelled. Nothing was changed.
  pause
)
exit /b

:run
echo ================= START =================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0set-pagefile.ps1"
echo ================= DONE (exit code %errorlevel%) =================
echo.
echo Press any key to close this window...
pause >nul
