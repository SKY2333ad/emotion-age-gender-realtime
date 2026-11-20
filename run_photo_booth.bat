@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
cd /d "%~dp0"
title AI Smart Photo Booth
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\portable_photo_booth.ps1"
if errorlevel 1 pause
