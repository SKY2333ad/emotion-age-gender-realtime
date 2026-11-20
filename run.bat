@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
cd /d "%~dp0"
title Nhom 1 - Nhan dien cam xuc

rem Goi thang: run.bat booth ^| public ^| app ^| stop. Khong co tham so thi hien menu.
set "MODE=%~1"
if not "%MODE%"=="" goto run

echo  1. Web Photo Booth
echo  2. Web Photo Booth qua Internet (can ngrok)
echo  3. App nhan dien realtime (Streamlit)
echo  4. Dung Photo Booth
echo.
choice /c 1234 /n /m "Chon [1-4]: "
if errorlevel 4 (set "MODE=stop") else if errorlevel 3 (set "MODE=app") else if errorlevel 2 (set "MODE=public") else set "MODE=booth"

:run
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run.ps1" -Mode %MODE%
if errorlevel 1 pause
