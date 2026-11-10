# ==============================================================
# stock_ai 一键环境搭建（Windows PowerShell）
# 用法：powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
# ==============================================================
$ErrorActionPreference = "Stop"

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " stock_ai 环境搭建" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

# ---- 定位 Python（优先使用托管版绝对路径）----
$PY = "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    Write-Host "[warn] 未找到托管 venv，尝试使用 PATH 中的 python" -ForegroundColor Yellow
    $PY = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PY) {
        Write-Host "[error] 未找到 Python 解释器，请手动安装 Python 3.11+" -ForegroundColor Red
        exit 1
    }
}
Write-Host "[info] 使用解释器: $PY" -ForegroundColor Green

# ---- 项目根目录 ----
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
Write-Host "[info] 项目根目录: $ROOT" -ForegroundColor Green

# ---- 1) 升级 pip ----
Write-Host "`n[1/5] 升级 pip ..." -ForegroundColor Cyan
& $PY -m pip install --upgrade pip

# ---- 2) 先装轻量核心（不涉及 torch，保证基础链路可用）----
Write-Host "`n[2/5] 安装核心依赖 ..." -ForegroundColor Cyan
& $PY -m pip install "numpy>=2.1,<2.6" "pandas>=2.2,<3.0" "scipy>=1.14" `
    "pyarrow>=17,<22" "click>=8.1" "pyyaml>=6.0" "matplotlib>=3.9" `
    "requests>=2.32" "tqdm>=4.66" "gymnasium==1.1.1"

# ---- 3) torch CPU 版（固定索引，避免拉到 CUDA 版）----
Write-Host "`n[3/5] 安装 torch (CPU) ..." -ForegroundColor Cyan
try {
    & $PY -m pip install "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cpu
} catch {
    Write-Host "[warn] torch 安装失败，将走 NumpyPPO 兜底（不影响主链路）" -ForegroundColor Yellow
}

# ---- 4) SB3（可选）----
Write-Host "`n[4/5] 安装 stable-baselines3 (可选) ..." -ForegroundColor Cyan
try {
    & $PY -m pip install "stable-baselines3>=2.6,<2.9"
} catch {
    Write-Host "[warn] SB3 安装失败，将走 NumpyPPO 兜底（不影响主链路）" -ForegroundColor Yellow
}

# ---- 5) 可编辑安装本项目 ----
Write-Host "`n[5/5] 可编辑安装 stock_ai ..." -ForegroundColor Cyan
& $PY -m pip install -e .

# ---- 自检 ----
Write-Host "`n[doctor] 运行环境自检 ..." -ForegroundColor Cyan
& $PY -m stock_ai doctor

Write-Host "`n[完成] 环境搭建结束" -ForegroundColor Green
Write-Host "下一步: powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1" -ForegroundColor Green
Write-Host "`n[提醒] 本项目全部依赖装在 venv 中，新开终端后请先激活再使用:" -ForegroundColor Yellow
Write-Host '  & "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\Activate.ps1"' -ForegroundColor Yellow
Write-Host "或直接用 venv 解释器绝对路径:" -ForegroundColor Yellow
Write-Host '  & "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -m stock_ai <cmd>' -ForegroundColor Yellow
Write-Host "命令速查: python -m stock_ai data fetch --symbols 600519 --start 20120101" -ForegroundColor Yellow
