# ==============================================================
# stock_ai 端到端流水线（Windows PowerShell）
# 串联 6 条命令，使用 fast_dev.yaml 小样本快速验证
# 用法：powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1
# ==============================================================
$ErrorActionPreference = "Stop"

$PY = "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    $PY = (Get-Command python -ErrorAction SilentlyContinue).Source
}

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$CFG   = "configs/fast_dev.yaml"
$SYMS  = @("600519")
$START = "20200101"
$END   = "20241231"

function Step($n, $title) {
    Write-Host "`n=============================================================" -ForegroundColor Cyan
    Write-Host " [$n] $title" -ForegroundColor Cyan
    Write-Host "=============================================================" -ForegroundColor Cyan
}

# ---- 1) data fetch ----
Step "1/6" "data fetch"
& $PY -m stock_ai data fetch --config $CFG --symbols $SYMS --start $START --end $END --source akshare

# ---- 2) data refresh ----
Step "2/6" "data refresh (incremental)"
& $PY -m stock_ai data refresh --config $CFG --symbols $SYMS

# ---- 3) sample build ----
Step "3/6" "sample build"
& $PY -m stock_ai sample build --config $CFG --symbols $SYMS --window 10

# ---- 4) train ----
Step "4/6" "train"
& $PY -m stock_ai train --config $CFG --algo ppo --timesteps 5000

# ---- 5) backtest ----
Step "5/6" "backtest"
$MODEL = Get-ChildItem models\ppo_*.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $MODEL) { Write-Host "[error] 未找到模型文件" -ForegroundColor Red; exit 1 }
& $PY -m stock_ai backtest --config $CFG --model $MODEL.FullName

# ---- 6) predict ----
Step "6/6" "predict"
& $PY -m stock_ai predict --config $CFG --model $MODEL.FullName `
    --symbols $SYMS --start "20240101" --end "20241231"

Write-Host "`n[完成] 端到端流水线全部通过" -ForegroundColor Green
Write-Host "产物: data/  models/  logs/  reports/" -ForegroundColor Green
