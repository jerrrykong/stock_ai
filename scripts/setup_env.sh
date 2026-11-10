#!/usr/bin/env bash
# ==============================================================
# stock_ai 一键环境搭建（Linux / macOS Bash）
# 用法：bash scripts/setup_env.sh
#
# 与 scripts/setup_env.ps1 功能对等：
#   定位解释器 -> 建 venv -> 升级 pip -> 装核心依赖 -> 装 torch(CPU)
#   -> 装 SB3(可选) -> pip install -e . -> doctor 自检
#
# 设计要点：
#   * venv 建在项目内 $ROOT/.venv（用户级，无需 sudo）
#   * pip 走镜像源（可用环境变量 PIP_INDEX_URL 覆盖）
#   * torch / SB3 安装失败【不致命】，自动回退纯 NumPy PPO 兜底
# ==============================================================
set -euo pipefail

# ---- 强制 UTF-8，避免中文输出乱码（远程 locale 可能是 C）----
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-C.UTF-8}"
# 限制 BLAS / OpenMP 线程数，规避低内存机器上 torch 线程爆炸
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

# ---- 定位脚本/项目根目录 ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

# ---- 可配置项（环境变量优先）----
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PIP_TIMEOUT="${PIP_TIMEOUT:-60}"
PIP_RETRIES="${PIP_RETRIES:-3}"

# ---- 打印辅助 ----
c_cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }
c_green()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
c_red()    { printf '\033[31m%s\033[0m\n' "$*"; }

c_cyan "============================================================="
c_cyan " stock_ai 环境搭建 (Bash)"
c_cyan "============================================================="

# ---- 定位基础 Python 解释器（禁止硬编码 Windows 路径）----
find_base_python() {
    local cand
    for cand in python3 python3.11 python3.12 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            printf '%s' "$cand"
            return 0
        fi
    done
    return 1
}

BASE_PY="$(find_base_python || true)"
if [ -z "$BASE_PY" ]; then
    c_red "[error] 未找到 Python 解释器（需要 python3.11+），请先安装"
    exit 1
fi
c_green "[info] 基础解释器: $(command -v "$BASE_PY") ($("$BASE_PY" --version 2>&1))"
c_green "[info] 项目根目录: $ROOT"

# ---- 走镜像源安装的 pip 封装 ----
pip_mirror() {
    "$PY" -m pip install -i "$PIP_INDEX_URL" \
        --timeout "$PIP_TIMEOUT" --retries "$PIP_RETRIES" "$@"
}

# ---- 1) 建 venv ----
c_cyan ""
c_cyan "[1/6] 创建 / 复用虚拟环境: $VENV_DIR"
if [ -x "$VENV_DIR/bin/python" ]; then
    c_green "[info] venv 已存在，复用"
else
    "$BASE_PY" -m venv "$VENV_DIR"
    c_green "[info] venv 已创建"
fi

PY="$VENV_DIR/bin/python"
if [ ! -x "$PY" ]; then
    c_red "[error] venv 解释器不存在: $PY"
    exit 1
fi
c_green "[info] 使用解释器: $PY ($("$PY" --version 2>&1))"

# ---- 2) 升级 pip / setuptools / wheel ----
c_cyan ""
c_cyan "[2/6] 升级 pip / setuptools / wheel ..."
pip_mirror --upgrade pip setuptools wheel

# ---- 3) 先装轻量核心（不涉及 torch，保证基础链路可用）----
c_cyan ""
c_cyan "[3/6] 安装核心依赖（轻量，保证 NumPy 后端主链路可用）..."
pip_mirror \
    "numpy>=2.1,<2.6" "pandas>=2.2,<3.0" "scipy>=1.14" \
    "pyarrow>=17,<22" "click>=8.1" "pyyaml>=6.0" "matplotlib>=3.9" \
    "requests>=2.32" "tqdm>=4.66" "gymnasium==1.1.1"

# ---- 4) torch CPU 版（固定索引，避免拉到 CUDA 版；失败不致命）----
c_cyan ""
c_cyan "[4/6] 安装 torch (CPU) —— 失败不致命，将回退 NumPy PPO ..."
if "$PY" -m pip install "torch>=2.6,<2.8" --index-url "$TORCH_INDEX_URL" \
        --timeout "$PIP_TIMEOUT" --retries "$PIP_RETRIES"; then
    c_green "[info] torch 安装成功"
else
    c_yellow "[warn] torch 安装失败，将走 NumPyPPO 兜底（不影响主链路）"
fi

# ---- 5) SB3（可选；失败不致命）----
c_cyan ""
c_cyan "[5/6] 安装 stable-baselines3 (可选) —— 失败不致命 ..."
if pip_mirror "stable-baselines3>=2.6,<2.9"; then
    c_green "[info] SB3 安装成功"
else
    c_yellow "[warn] SB3 安装失败，将走 NumPyPPO 兜底（不影响主链路）"
fi

# ---- 6) 可编辑安装本项目 ----
c_cyan ""
c_cyan "[6/6] 以可编辑模式安装 stock_ai ..."
pip_mirror -e .

# ---- 自检 ----
c_cyan ""
c_cyan "[doctor] 运行环境自检 ..."
"$PY" -m stock_ai doctor

c_green ""
c_green "[完成] 环境搭建结束"
c_green "下一步: bash scripts/run_pipeline.sh"
c_yellow ""
c_yellow "[提醒] 本环境位于项目内 venv，可直接用:"
c_yellow "  $PY -m stock_ai <cmd>"
c_yellow "或先激活: source $VENV_DIR/bin/activate"
