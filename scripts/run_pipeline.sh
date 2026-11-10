#!/usr/bin/env bash
# ==============================================================
# stock_ai 端到端流水线（Linux / macOS Bash）
# 串联 6 条命令，使用 configs/fast_dev.yaml 小样本快速验证
# 用法：bash scripts/run_pipeline.sh
#
# 与 scripts/run_pipeline.ps1 功能对等，并修正：
#   * 用 `ls -t models/<algo>_*.zip | head -1` 健壮地找最新模型
#   * 训练步数可用环境变量 TIMESTEPS 覆盖（低内存机器可调小）
# ==============================================================
set -euo pipefail

# ---- UTF-8 + 线程限制 ----
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-C.UTF-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

# ---- 项目根目录 ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

# ---- 定位解释器：$ROOT/.venv/bin/python -> python3 -> 报错 ----
resolve_python() {
    if [ -x "$ROOT/.venv/bin/python" ]; then
        printf '%s' "$ROOT/.venv/bin/python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

PY="${STOCKAI_PYTHON:-$(resolve_python || true)}"
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    printf '\033[31m[error] 未找到 Python 解释器，请先运行: bash scripts/setup_env.sh\033[0m\n' >&2
    exit 1
fi

# ---- 可配置参数（环境变量优先，便于在低内存机器上调整）----
CFG="${CFG:-configs/fast_dev.yaml}"
SYMBOLS_CSV="${SYMBOLS:-600519}"
START="${START:-20200101}"
END="${END:-20241231}"
SOURCE="${SOURCE:-akshare}"
WINDOW="${WINDOW:-10}"
ALGO="${ALGO:-ppo}"
TIMESTEPS="${TIMESTEPS:-5000}"
BACKEND="${BACKEND:-}"
PREDICT_START="${PREDICT_START:-20240101}"
PREDICT_END="${PREDICT_END:-20241231}"

# ---- 打印步骤标题 ----
step() {
    printf '\n\033[36m=============================================================\033[0m\n'
    printf '\033[36m [%s] %s\033[0m\n' "$1" "$2"
    printf '\033[36m=============================================================\033[0m\n'
}

# ---- 把逗号分隔的股票代码展开为多个 --symbols ----
IFS=',' read -ra _SYMS <<< "$SYMBOLS_CSV"
SYM_ARGS=()
for _s in "${_SYMS[@]}"; do
    _s="$(printf '%s' "$_s" | tr -d '[:space:]')"
    if [ -n "$_s" ]; then
        SYM_ARGS+=("--symbols" "$_s")
    fi
done

printf '\033[32m[info] 解释器 : %s\033[0m\n' "$PY"
printf '\033[32m[info] 配置   : %s\033[0m\n' "$CFG"
printf '\033[32m[info] 股票   : %s\033[0m\n' "$SYMBOLS_CSV"
printf '\033[32m[info] 数据源 : %s\033[0m\n' "$SOURCE"

# ---- 1) data fetch ----
step "1/6" "data fetch"
"$PY" -m stock_ai data fetch --config "$CFG" "${SYM_ARGS[@]}" \
    --start "$START" --end "$END" --source "$SOURCE"

# ---- 2) data refresh ----
step "2/6" "data refresh (incremental)"
"$PY" -m stock_ai data refresh --config "$CFG" "${SYM_ARGS[@]}"

# ---- 3) sample build ----
step "3/6" "sample build"
"$PY" -m stock_ai sample build --config "$CFG" "${SYM_ARGS[@]}" --window "$WINDOW"

# ---- 4) train ----
step "4/6" "train"
TRAIN_ARGS=(train --config "$CFG" --algo "$ALGO" --timesteps "$TIMESTEPS")
if [ -n "$BACKEND" ]; then
    TRAIN_ARGS+=(--backend "$BACKEND")
fi
"$PY" -m stock_ai "${TRAIN_ARGS[@]}"

# ---- 5) backtest（健壮地定位最新模型）----
step "5/6" "backtest"
MODEL=""
if compgen -G "models/${ALGO}_*.zip" >/dev/null 2>&1; then
    MODEL="$(ls -1t models/"${ALGO}"_*.zip | head -n1)"
elif compgen -G "models/*.zip" >/dev/null 2>&1; then
    MODEL="$(ls -1t models/*.zip | head -n1)"
fi
if [ -z "$MODEL" ]; then
    printf '\033[31m[error] 未找到模型文件 (models/%s_*.zip)\033[0m\n' "$ALGO" >&2
    exit 1
fi
printf '\033[32m[info] 使用模型: %s\033[0m\n' "$MODEL"
"$PY" -m stock_ai backtest --config "$CFG" --model "$MODEL"

# ---- 6) predict ----
step "6/6" "predict"
"$PY" -m stock_ai predict --config "$CFG" --model "$MODEL" "${SYM_ARGS[@]}" \
    --start "$PREDICT_START" --end "$PREDICT_END"

printf '\n\033[32m[完成] 端到端流水线全部通过\033[0m\n'
printf '\033[32m产物: data/  models/  logs/  reports/\033[0m\n'
