@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=[IO.Path]::GetFullPath('%~dp0'); $runtime=Join-Path $root '.runtime'; foreach($name in @('server','ngrok')){$file=Join-Path $runtime ($name+'.pid'); if(Test-Path -LiteralPath $file){$processId=[int](Get-Content -LiteralPath $file -Raw); $process=Get-Process -Id $processId -ErrorAction SilentlyContinue; if($process){Stop-Process -Id $processId -Force; Write-Host ('[PhotoBooth] Da dung '+$name)}; Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue}}"
pause
