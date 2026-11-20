param(
    [ValidateSet("booth", "public", "app", "stop")]
    [string]$Mode = "booth",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Continue"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements.txt"
$runtimeDir = Join-Path $projectRoot ".runtime"
$serverScript = Join-Path $projectRoot "sites\ai-smart-photo-booth\server.py"
$localUrl = "http://127.0.0.1:4173"
$healthUrl = "$localUrl/api/health"

function Write-Step([string]$message) {
    Write-Host "[Nhom1] $message" -ForegroundColor Cyan
}

function Test-PythonVersion([string]$file, [string[]]$prefixArgs) {
    try {
        $code = 'import sys; raise SystemExit(0 if (3,12) <= sys.version_info[:2] <= (3,13) else 1)'
        & $file @prefixArgs -c $code 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Find-BootstrapPython {
    $candidates = @(
        @{ File = "py"; Args = @("-3.13") },
        @{ File = "py"; Args = @("-V:Astral/CPython3.13.2") },
        @{ File = "py"; Args = @("-3.12") },
        @{ File = "python"; Args = @() },
        @{ File = (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"); Args = @() },
        @{ File = (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"); Args = @() }
    )
    foreach ($candidate in $candidates) {
        if (Test-PythonVersion $candidate.File $candidate.Args) { return $candidate }
    }
    return $null
}

function Install-PythonIfPossible {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { return $false }
    Write-Step "Chua co Python 3.12/3.13. Dang cai Python 3.13 bang winget..."
    & winget install --id Python.Python.3.13 --exact --silent --accept-package-agreements --accept-source-agreements
    return $LASTEXITCODE -eq 0
}

function Test-Dependencies {
    if (-not (Test-Path -LiteralPath $venvPython)) { return $false }
    & $venvPython -c "import cv2, keras, mediapipe, numpy, onnxruntime, PIL, tensorflow, streamlit, pandas, matplotlib" 2>$null
    return $LASTEXITCODE -eq 0
}

# Lan dau: tao venv, cai thu vien va kiem tra he thong. Cac lan sau bo qua.
function Initialize-Environment {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $bootstrap = Find-BootstrapPython
        if (-not $bootstrap -and (Install-PythonIfPossible)) { $bootstrap = Find-BootstrapPython }
        if (-not $bootstrap) {
            throw "Can Python 3.12 hoac 3.13 (64-bit). Python 3.14+ chua dung duoc voi TensorFlow."
        }
        Write-Step "Dang tao moi truong venv..."
        & $bootstrap.File @($bootstrap.Args) -m venv (Join-Path $projectRoot "venv")
        if ($LASTEXITCODE -ne 0) { throw "Khong tao duoc moi truong ao." }
    }
    if (Test-Dependencies) { return }
    Write-Step "Lan dau chay: dang cai thu vien (can Internet, khoang 10-20 phut)..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Khong nang cap duoc pip." }
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Cai thu vien that bai." }
    if (-not (Test-Dependencies)) { throw "Thu vien da cai nhung kiem tra import khong dat." }
    & $venvPython (Join-Path $projectRoot "scripts\check_sync.py")
    if ($LASTEXITCODE -ne 0) { throw "Kiem tra he thong (check_sync.py) khong dat." }
}

function Wait-ForHealth([int]$seconds = 180) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
            if ($health.ok) { return $true }
        } catch {}
        if (($i % 10) -eq 0) { Write-Step "Dang nap model AI... ($i/$seconds giay)" }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-LocalServer {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.ok) {
            Write-Step "Server da chay san tai $localUrl"
            return
        }
    } catch {}

    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $stdout = Join-Path $runtimeDir "server.log"
    $stderr = Join-Path $runtimeDir "server-error.log"
    Write-Step "Dang khoi dong server..."
    $process = Start-Process -FilePath $venvPython -ArgumentList @($serverScript) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $runtimeDir "server.pid") -Value $process.Id -Encoding ascii
    if (-not (Wait-ForHealth)) {
        Write-Host "Khong khoi dong duoc server. Xem file .runtime\server-error.log" -ForegroundColor Red
        if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Tail 30 }
        throw "Server khong san sang."
    }
    Write-Step "Server da san sang."
}

function Find-Ngrok {
    $bundled = Join-Path $projectRoot "tools\ngrok.exe"
    if (Test-Path -LiteralPath $bundled) { return $bundled }
    $command = Get-Command ngrok -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    return $null
}

function Start-PublicTunnel {
    $ngrok = Find-Ngrok
    if (-not $ngrok) {
        Write-Host "Khong co ngrok. Website local van chay binh thuong: $localUrl" -ForegroundColor Yellow
        return $null
    }
    & $ngrok config check *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ngrok chua co authtoken. Website local van chay binh thuong." -ForegroundColor Yellow
        return $null
    }
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $stdout = Join-Path $runtimeDir "ngrok.log"
    $stderr = Join-Path $runtimeDir "ngrok-error.log"
    $process = Start-Process -FilePath $ngrok -ArgumentList @("http", "4173") -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $runtimeDir "ngrok.pid") -Value $process.Id -Encoding ascii
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
            $url = @($tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
            if ($url) { Write-Step "Public URL: $url"; return $url }
        } catch {}
        Start-Sleep -Seconds 1
    }
    Write-Host "Ngrok chua tao duoc tunnel; tiep tuc dung local." -ForegroundColor Yellow
    return $null
}

function Stop-Background {
    foreach ($name in @("server", "ngrok")) {
        $file = Join-Path $runtimeDir "$name.pid"
        if (-not (Test-Path -LiteralPath $file)) { continue }
        $process = Get-Process -Id ([int](Get-Content -LiteralPath $file -Raw)) -ErrorAction SilentlyContinue
        if ($process) { Stop-Process -Id $process.Id -Force; Write-Step "Da dung $name" }
        Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue
    }
}

try {
    if ($Mode -eq "stop") { Stop-Background; exit 0 }
    Write-Step "Chuan bi moi truong chay..."
    Initialize-Environment
    if ($Mode -eq "app") {
        Write-Step "Mo app Streamlit (dong cua so nay de dung)..."
        & $venvPython -m streamlit run (Join-Path $projectRoot "app.py")
        exit $LASTEXITCODE
    }
    Start-LocalServer
    $openUrl = if ($Mode -eq "public") { Start-PublicTunnel } else { $null }
    if (-not $openUrl) { $openUrl = $localUrl }
    if (-not $NoBrowser -and $env:PHOTO_BOOTH_NO_BROWSER -ne "1") { Start-Process $openUrl }
    Write-Host ""
    Write-Host "Photo Booth dang chay: $openUrl" -ForegroundColor Green
    Write-Host "Co the dong cua so nay. Chon 'Dung Photo Booth' trong run.bat de tat server." -ForegroundColor Gray
    exit 0
} catch {
    Write-Host ""
    Write-Host "LOI: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
