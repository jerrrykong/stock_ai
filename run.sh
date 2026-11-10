#!/usr/bin/env bash
# ==============================================================
# run.sh - stock_ai 统一入口脚本（Linux / macOS Bash）
#
# 用法:
#   bash run.sh                         # 默认显示项目全景摘要 (status)
#   bash run.sh status                  # 同上
#   bash run.sh help                    # 显示帮助
#   bash run.sh <命令> [参数]            # 调用对应功能
#   bash run.sh <命令> --help           # 显示该命令的完整 CLI 帮助
#   bash run.sh cli "<参数串>"           # 透传任意参数给 python -m stock_ai
#
# 参数风格（GNU）:
#   --symbols --start --end --source --adjust --force
#   --algo --timesteps --backend --reward
#   --model --out --config --window --samples --help
#
# 与 run.ps1 功能对等：status / help / 六大命令快捷入口 / 参数透传。
# ==============================================================
set -euo pipefail

# ---- 强制 UTF-8 + 线程限制（远程 locale 可能是 C）----
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-C.UTF-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

# ---- 项目根目录 ----
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT"

# ---- 定位解释器：$ROOT/.venv/bin/python -> python3 -> python -> 报错 ----
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
    printf '\033[31m[错误] 未找到 Python 解释器, 请先运行: bash scripts/setup_env.sh\033[0m\n' >&2
    exit 1
fi

# ---- 校验解释器已安装本项目（避免回退到系统 python3 时报零散的 ModuleNotFound）----
if ! "$PY" -c "import stock_ai" >/dev/null 2>&1; then
    printf '\033[31m[错误] 当前解释器未安装 stock_ai（或依赖不完整）:\033[0m %s\n' "$PY" >&2
    printf '  请先运行:            bash scripts/setup_env.sh\n' >&2
    printf '  或手动安装:          %s -m pip install -e .\n' "$PY" >&2
    printf '  也可临时指定解释器:  STOCKAI_PYTHON=/path/to/python bash run.sh ...\n' >&2
    exit 1
fi

# ==============================================================
# 帮助
# ==============================================================
show_help() {
    printf '\n'
    printf '\033[36mstock_ai 统一入口  (解释器: %s)\033[0m\n' "$PY"
    printf '\033[2m==============================================================\033[0m\n'
    printf '数据      bash run.sh data-fetch   [--symbols 600519,000001] [--start 20120101] [--end 20241231] [--source akshare] [--adjust qfq|hfq|none] [--force]  # --adjust 默认 none 不复权\n'
    printf '          bash run.sh data-refresh [--symbols 600519] [--adjust qfq|hfq|none]   # 增量刷新(只拉缺失日期), --adjust 默认 none 不复权\n'
    printf '          bash run.sh data-info                                # 缓存概览\n'
    printf '样本      bash run.sh sample-build [--config configs/default.yaml] [--window 20]\n'
    printf '训练      bash run.sh train        [--algo ppo|sac|dqn] [--timesteps 200000] [--backend auto|sb3|numpy]\n'
    printf '回测      bash run.sh backtest     [--model models/ppo_xxx.zip]\n'
    printf '预测      bash run.sh predict      [--model models/ppo_xxx.zip] [--start 20250101] [--end 20250601]\n'
    printf '自检      bash run.sh doctor\n'
    printf '摘要      bash run.sh status                                   # 数据/样本/模型/报告全景 (默认)\n'
    printf '大规模    bash run.sh scale-fetch    [--source sina] [--throttle 0.5] [--force]  # 拉取沪深300 最近20年 不复权日线\n'
    printf '          bash run.sh scale-units    [--seed 42]                # 生成 units.json + split.json (可复现)\n'
    printf '          bash run.sh scale-train    [--timesteps 20000] [--algo ppo] [--backend auto]  # 长期训练(可中断/续跑)\n'
    printf '          bash run.sh scale-detect                               # 检测编排(选模型 + 回测)\n'
    printf '          bash run.sh scale-status                               # 打印训练进度(总数/完成/失败/剩余/ETA)\n'
    printf '          bash run.sh scale-ui       [--port 8765]               # 启动本机 UI (127.0.0.1)\n'
    printf '\033[2m==============================================================\033[0m\n'
    printf '查看某命令完整参数:  bash run.sh <命令> --help      (如 bash run.sh train --help)\n'
    printf '透传任意 CLI 参数:   bash run.sh cli "data fetch --symbols 600519 --force"\n'
    printf '\n'
}

# ==============================================================
# 通用执行封装：打印命令 -> 执行 -> 透传退出码
# ==============================================================
run_cli() {
    local args=("$@")
    if [ "$HELPFLAG" -eq 1 ]; then
        args+=(--help)
    fi
    printf '\033[90m>> python -m stock_ai %s\033[0m\n' "${args[*]}"
    local rc=0
    "$PY" -m stock_ai "${args[@]}" || rc=$?
    exit "$rc"
}

# ==============================================================
# status: 全景摘要（纯标准库解析 JSON，不依赖 jq）
# ==============================================================
show_status() {
    local ver
    ver="$("$PY" -m stock_ai --version 2>/dev/null || true)"
    printf '\n'
    printf '\033[36m================ stock_ai 项目状态 ================\033[0m\n'
    printf '项目根目录 : %s\n' "$ROOT"
    printf '解释器     : %s\n' "$PY"
    printf '包版本     : %s\n' "$ver"

    ROOT="$ROOT" "$PY" - <<'PYEOF'
import json
import os
import time
from pathlib import Path

root = Path(os.environ["ROOT"])


def fmt_pct(v):
    """百分比（对应 PowerShell {0:P2}）。"""
    if v is None:
        return "-"
    try:
        return f"{float(v):.2%}"
    except Exception:
        return "-"


def fmt_num(v):
    """三位小数（对应 {0:F3}）。"""
    if v is None or v == "-":
        return "-"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "-"


def fmt_int(v):
    """千分位整数（对应 {0:N0}）。"""
    if v is None or v == "-":
        return "-"
    try:
        return f"{int(float(v)):,}"
    except Exception:
        return "-"


def kb(n):
    """字节 -> KB 千分位整数。"""
    return f"{n / 1024:,.0f} KB"


def mtime(p):
    """文件修改时间字符串。"""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
    except Exception:
        return "-"


def load_json(p):
    """安全读取 JSON，失败返回 None。"""
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---- [1] 原始行情数据 ----
print()
print("\033[33m[1] 原始日线数据  (data/raw)\033[0m")
raw = root / "data" / "raw"
meta_p = raw / "meta.json"
if meta_p.exists():
    meta = load_json(meta_p) or {}
    syms = meta.get("symbols") or {}
    print(f"  共 {len(syms)} 只股票:")
    for name, s in syms.items():
        s = s or {}
        print("    {0}: {1} 行  {2} ~ {3}  来源={4}  复权={5}  缺口={6} 天  更新={7}".format(
            name, s.get("rows", "-"), s.get("start", "-"), s.get("end", "-"),
            s.get("source", "-"), s.get("adjust", "-"), s.get("missing_days", "0"),
            s.get("updated_at", "-")))
elif raw.is_dir():
    pq = sorted(raw.glob("*.parquet"))
    print(f"  无 meta.json; 发现 {len(pq)} 个 parquet 文件 (建议运行 data-info 查看)")
    for f in pq:
        print(f"    {f.name}  ({kb(f.stat().st_size)})")
else:
    print("  (空) 尚未拉取数据, 运行: bash run.sh data-fetch --symbols 600519")
print()

# ---- [2] 训练/验证/测试样本 ----
print("\033[33m[2] 样本数据  (data/samples)\033[0m")
sp = root / "data" / "samples"
sp_meta = sp / "meta.json"
if sp_meta.exists():
    m = load_json(sp_meta) or {}
    tr = m.get("train_range") or {}
    te = m.get("test_range") or {}
    rows = m.get("rows") or {}
    feats = m.get("feature_names") or []
    syms = m.get("symbols") or []
    print("  股票: {0}   窗口: {1} 步   特征: {2} 维   embargo: {3} 交易日   seed: {4}".format(
        ",".join(str(x) for x in syms), m.get("window", "-"), len(feats),
        m.get("embargo_days", "0"), m.get("seed", "-")))
    print("  划分: 训练 {0} 步 ({1}~{2}) | 验证 {3} 步 | 测试 {4} 步 ({5}~{6})".format(
        rows.get("train", "-"), tr.get("start", "-"), tr.get("end", "-"),
        rows.get("val", "-"), rows.get("test", "-"),
        te.get("start", "-"), te.get("end", "-")))
    dh = str(m.get("data_hash", ""))
    print(f"  data_hash: {dh[:8]}...   生成时间: {m.get('created_at', '-')}")
    for n in ("train", "val", "test"):
        f = sp / f"{n}.npz"
        if f.exists():
            print(f"    {n}.npz  ({kb(f.stat().st_size)})")
        else:
            print(f"    {n}.npz  (缺失!)")
else:
    print("  (空) 尚未构建样本, 运行: bash run.sh sample-build")
print()

# ---- [3] 训练产物模型 ----
print("\033[33m[3] 训练模型  (models)\033[0m")
mdl = root / "models"
zips = sorted(mdl.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)[:5] if mdl.is_dir() else []
if zips:
    for f in zips:
        side = f.with_suffix(".json")
        algo = backend = steps = "-"
        created = mtime(f)
        if side.exists():
            sc = load_json(side) or {}
            algo = sc.get("algo", "-")
            backend = sc.get("backend", "-")
            steps = (sc.get("train_cfg") or {}).get("timesteps", "-")
            created = sc.get("created_at", created)
        print("    {0}  [{1}/{2}]  {3} 步  {4}  {5}".format(
            f.name, algo, backend, steps, kb(f.stat().st_size), created))
else:
    print("  (空) 尚未训练, 运行: bash run.sh train --algo ppo")
print()

# ---- [4] 回测报告 ----
print("\033[33m[4] 回测报告  (reports/backtest_*.json, 最近 5 份)\033[0m")
rep = root / "reports"
bts = sorted(rep.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5] if rep.is_dir() else []
if bts:
    for f in bts:
        r = load_json(f)
        if not r:
            print(f"    {f.name}  (解析失败)")
            continue
        mt = r.get("meta") or {}
        me = r.get("metrics") or {}
        tr = mt.get("test_range") or {}
        print(f"    {f.name}")
        print("      模型: {0} [{1}/{2}]  测试区间: {3}~{4}".format(
            mt.get("model_path", "-"), mt.get("algo", "-"), mt.get("backend", "-"),
            tr.get("start", "-"), tr.get("end", "-")))
        print("      年化 {0}  累计 {1}  最大回撤 {2}  夏普 {3}  胜率 {4}  交易 {5} 笔  基准累计 {6}".format(
            fmt_pct(me.get("annual_return")), fmt_pct(me.get("total_return")),
            fmt_pct(me.get("max_drawdown")), fmt_num(me.get("sharpe")),
            fmt_pct(me.get("win_rate")), fmt_int(me.get("num_trades")),
            fmt_pct(me.get("benchmark_return"))))
else:
    print("  (空) 尚未回测, 运行: bash run.sh backtest")
print()

# ---- [5] 预测输出 ----
print("\033[33m[5] 预测输出  (reports/predict_*.csv, 最近 5 份)\033[0m")
pds = sorted(rep.glob("predict_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:5] if rep.is_dir() else []
if pds:
    for f in pds:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                n = sum(1 for _ in fh)
        except Exception:
            n = 0
        print("    {0}  {1} 条预测  {2}  {3}".format(
            f.name, max(0, n - 1), kb(f.stat().st_size), mtime(f)))
else:
    print("  (空) 尚未预测, 运行: bash run.sh predict")
print()

# ---- [6] 最近日志 ----
print("\033[33m[6] 最近日志  (logs)\033[0m")
lg = root / "logs"
files = [p for p in lg.rglob("*") if p.is_file()] if lg.is_dir() else []
files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
if files:
    for f in files[:3]:
        try:
            rel = f.relative_to(root)
        except Exception:
            rel = f
        print(f"    {rel}  {mtime(f)}")
else:
    print("  (空)")
print()
PYEOF
}

# ==============================================================
# 命令解析
# ==============================================================
CMD="${1:-status}"
if [ "$#" -gt 0 ]; then
    shift
fi
CMD_KEY="$(printf '%s' "$CMD" | tr '[:upper:]' '[:lower:]' | sed 's#^[-/]*##')"

# ---- cli 透传模式：run.sh cli "<完整参数串>" ----
if [ "$CMD_KEY" = "cli" ]; then
    if [ "$#" -lt 1 ]; then
        printf '\033[31m[错误] cli 透传需要参数\033[0m\n' >&2
        printf '  示例: bash run.sh cli "data fetch --symbols 600519 --force"\n' >&2
        exit 1
    fi
    printf '\033[90m>> python -m stock_ai %s\033[0m\n' "$*"
    rc=0
    # shellcheck disable=SC2086  # 有意做词分割：把 "data info" 拆成两个参数
    "$PY" -m stock_ai $* || rc=$?
    exit "$rc"
fi

# ---- help ----
if [ "$CMD_KEY" = "help" ] || [ "$CMD_KEY" = "h" ]; then
    if [ "$#" -gt 0 ]; then
        printf '\033[90m>> python -m stock_ai %s --help\033[0m\n' "$*"
        rc=0
        "$PY" -m stock_ai "$@" --help || rc=$?
        exit "$rc"
    fi
    show_help
    exit 0
fi

# ---- 解析短/长参数 ----
SYMBOLS=""; START=""; END=""; ALGO=""; TIMESTEPS=""; BACKEND=""
MODEL=""; SOURCE=""; ADJUST=""; OUT=""; CONFIG=""; REWARD=""; SAMPLES=""; WINDOW=""
PORT=""; SEED=""
FORCE=0; HELPFLAG=0
REST=()

set_var() {
    case "$1" in
        symbols)   SYMBOLS="$2" ;;
        start)     START="$2" ;;
        end)       END="$2" ;;
        algo)      ALGO="$2" ;;
        timesteps) TIMESTEPS="$2" ;;
        backend)   BACKEND="$2" ;;
        model)     MODEL="$2" ;;
        source)    SOURCE="$2" ;;
        adjust)    ADJUST="$2" ;;
        out)       OUT="$2" ;;
        config)    CONFIG="$2" ;;
        reward)    REWARD="$2" ;;
        samples)   SAMPLES="$2" ;;
        window)    WINDOW="$2" ;;
        port)      PORT="$2" ;;
        seed)      SEED="$2" ;;
        *)         : ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --*=*)
            _k="${1%%=*}"
            _k="${_k#--}"
            _k="$(printf '%s' "$_k" | tr '[:upper:]' '[:lower:]')"
            set_var "$_k" "${1#*=}"
            shift
            ;;
        --symbols|--start|--end|--algo|--timesteps|--backend|--model|--source|--adjust|--out|--config|--reward|--samples|--window|--port|--seed)
            if [ "$#" -lt 2 ]; then
                printf '\033[31m[错误] 选项 %s 缺少参数\033[0m\n' "$1" >&2
                exit 2
            fi
            _k="${1#--}"
            set_var "$_k" "$2"
            shift 2
            ;;
        --force|-f)
            FORCE=1
            shift
            ;;
        --help|-h)
            HELPFLAG=1
            shift
            ;;
        --)
            shift
            REST+=("$@")
            break
            ;;
        *)
            REST+=("$1")
            shift
            ;;
    esac
done

# ---- 组装公共参数 ----
common=()
if [ -n "$CONFIG" ]; then
    common+=(--config "$CONFIG")
fi

# ---- 股票代码 -> 多个 --symbols（支持逗号分隔）----
sym_args=()
if [ -n "$SYMBOLS" ]; then
    IFS=',' read -ra _SYMS <<< "$SYMBOLS"
    for _s in "${_SYMS[@]}"; do
        _s="$(printf '%s' "$_s" | tr -d '[:space:]')"
        if [ -n "$_s" ]; then
            sym_args+=(--symbols "$_s")
        fi
    done
fi

# ==============================================================
# 命令分发
# ==============================================================
args=()
case "$CMD_KEY" in
    data-fetch)
        args=(data fetch)
        args+=("${common[@]}")
        args+=("${sym_args[@]}")
        if [ -n "$START" ]; then args+=(--start "$START"); fi
        if [ -n "$END" ]; then args+=(--end "$END"); fi
        if [ -n "$SOURCE" ]; then args+=(--source "$SOURCE"); fi
        if [ -n "$ADJUST" ]; then args+=(--adjust "$ADJUST"); fi
        if [ "$FORCE" -eq 1 ]; then args+=(--force); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    data-refresh)
        args=(data refresh)
        args+=("${common[@]}")
        args+=("${sym_args[@]}")
        if [ -n "$ADJUST" ]; then args+=(--adjust "$ADJUST"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    data-info)
        args=(data info)
        args+=("${common[@]}")
        args+=("${sym_args[@]}")
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    sample-build)
        args=(sample build)
        args+=("${common[@]}")
        args+=("${sym_args[@]}")
        if [ -n "$WINDOW" ]; then args+=(--window "$WINDOW"); fi
        if [ -n "$OUT" ]; then args+=(--out "$OUT"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    train)
        args=(train)
        args+=("${common[@]}")
        if [ -n "$ALGO" ]; then args+=(--algo "$ALGO"); fi
        if [ -n "$TIMESTEPS" ]; then args+=(--timesteps "$TIMESTEPS"); fi
        if [ -n "$BACKEND" ]; then args+=(--backend "$BACKEND"); fi
        if [ -n "$REWARD" ]; then args+=(--reward "$REWARD"); fi
        if [ -n "$SAMPLES" ]; then args+=(--samples "$SAMPLES"); fi
        if [ -n "$OUT" ]; then args+=(--out "$OUT"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    backtest)
        args=(backtest)
        args+=("${common[@]}")
        if [ -n "$MODEL" ]; then args+=(--model "$MODEL"); fi
        if [ -n "$SAMPLES" ]; then args+=(--samples "$SAMPLES"); fi
        if [ -n "$OUT" ]; then args+=(--out "$OUT"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    predict)
        args=(predict)
        args+=("${common[@]}")
        if [ -n "$MODEL" ]; then args+=(--model "$MODEL"); fi
        args+=("${sym_args[@]}")
        if [ -n "$START" ]; then args+=(--start "$START"); fi
        if [ -n "$END" ]; then args+=(--end "$END"); fi
        if [ -n "$OUT" ]; then args+=(--out "$OUT"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    doctor)
        args=(doctor)
        args+=("${common[@]}")
        run_cli "${args[@]}"
        ;;
    scale-fetch)
        args=(scale fetch)
        args+=("${common[@]}")
        if [ -n "$SOURCE" ]; then args+=(--source "$SOURCE"); fi
        if [ -n "$SEED" ]; then args+=(--seed "$SEED"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    scale-units)
        args=(scale units)
        args+=("${common[@]}")
        if [ -n "$SEED" ]; then args+=(--seed "$SEED"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    scale-train)
        args=(scale train)
        args+=("${common[@]}")
        if [ -n "$SEED" ]; then args+=(--seed "$SEED"); fi
        if [ -n "$TIMESTEPS" ]; then args+=(--timesteps "$TIMESTEPS"); fi
        if [ -n "$ALGO" ]; then args+=(--algo "$ALGO"); fi
        if [ -n "$BACKEND" ]; then args+=(--backend "$BACKEND"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    scale-detect)
        args=(scale detect)
        args+=("${common[@]}")
        if [ -n "$SEED" ]; then args+=(--seed "$SEED"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    scale-status)
        args=(scale status)
        args+=("${common[@]}")
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    scale-ui)
        args=(scale ui)
        args+=("${common[@]}")
        if [ -n "$PORT" ]; then args+=(--port "$PORT"); fi
        args+=("${REST[@]}")
        run_cli "${args[@]}"
        ;;
    status)
        show_status
        ;;
    *)
        printf '\033[31m[错误] 未知命令: %s\033[0m\n' "$CMD" >&2
        show_help
        exit 1
        ;;
esac
