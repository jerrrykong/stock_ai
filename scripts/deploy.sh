#!/usr/bin/env bash
# ==============================================================
# stock_ai 部署脚本：把本地项目同步到远程 Linux 主机
#
# 安全约定：远程连接信息一律通过命令行参数或环境变量传入，
# 本脚本不写死任何主机 IP、SSH 用户等敏感信息。
#
# 用法：
#   bash scripts/deploy.sh --host <REMOTE_HOST> --user <REMOTE_USER> \
#        [--dir <REMOTE_DIR>] [--port <SSH_PORT>]
#
# 行为：
#   * 打包本地项目（排除 .git/.venv 及运行时产物 data/models/reports/logs）
#   * scp 上传 -> 远程解包（就地覆盖代码，保留远程运行时产物）
#   * 需要 SSH 密钥认证（BatchMode，不交互输密码）
#   * StrictHostKeyChecking=accept-new：首次连接自动记录新主机密钥，
#     主机密钥变化时仍会拒绝（比 StrictHostKeyChecking=no 更安全）
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
用法: bash scripts/deploy.sh --host <REMOTE_HOST> --user <REMOTE_USER> [--dir <REMOTE_DIR>] [--port <SSH_PORT>]

  --host  远程主机（IP 或域名，必填；或环境变量 REMOTE_HOST）
  --user  远程 SSH 用户（必填；或环境变量 REMOTE_USER）
  --dir   远程项目目录（默认 '~/stock_ai'，~ 由远程 shell 展开；或环境变量 REMOTE_DIR；
          含 ~ 时建议加引号传入，路径不要含空格）
  --port  SSH 端口（默认 22；或环境变量 SSH_PORT）

说明：本脚本不含任何主机 IP / 账号默认值；优先级为 命令行参数 > 环境变量。
USAGE
}

# ---- 参数解析（命令行参数 > 环境变量；无敏感默认值）----
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_DIR="${REMOTE_DIR:-~/stock_ai}"
SSH_PORT="${SSH_PORT:-22}"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) REMOTE_HOST="${2:-}"; shift 2 ;;
    --user) REMOTE_USER="${2:-}"; shift 2 ;;
    --dir)  REMOTE_DIR="${2:-}";  shift 2 ;;
    --port) SSH_PORT="${2:-}";    shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf '未知参数: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_USER" ]; then
  printf '错误：必须提供远程主机与 SSH 用户（--host/--user，或环境变量 REMOTE_HOST/REMOTE_USER）。\n本脚本不含任何主机/账号默认值。\n\n' >&2
  usage >&2
  exit 2
fi

if ! [[ "$SSH_PORT" =~ ^[0-9]+$ ]]; then
  printf '错误：SSH 端口必须是数字，收到: %s\n' "$SSH_PORT" >&2
  exit 2
fi

# 注意：ssh 的端口选项是 -p，scp 的是 -P，二者不可混用
COMMON_OPTS=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15)
SSH_OPTS=(-p "$SSH_PORT" "${COMMON_OPTS[@]}")
SCP_OPTS=(-P "$SSH_PORT" "${COMMON_OPTS[@]}")
DEST="${REMOTE_USER}@${REMOTE_HOST}"

# ---- 打包（排除版本库与运行时产物）----
TARBALL="$(mktemp -t stock_ai_deploy_XXXXXX.tar.gz 2>/dev/null || mktemp)"
trap 'rm -f "$TARBALL"' EXIT

printf '\033[36m[deploy] 打包本地项目...\033[0m\n'
# 注意：--exclude 必须用 './name' 锚定到顶层，否则会误伤任意同名子目录
# （例如 'data' 会连 src/stock_ai/data 一起排除掉）。
tar czf "$TARBALL" \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./data' \
    --exclude='./models' \
    --exclude='./reports' \
    --exclude='./logs' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='./.pytest_cache' \
    --exclude='./.mypy_cache' \
    --exclude='./.ruff_cache' \
    -C "$ROOT" .

# ---- 创建远程目录 ----
printf '\033[36m[deploy] 连接 %s ...\033[0m\n' "$DEST"
ssh "${SSH_OPTS[@]}" "$DEST" "mkdir -p $REMOTE_DIR"

# ---- 上传 ----
# 说明：先 scp 到远程家目录（相对路径，避免远程路径里的 ~ 在 sftp 模式下不展开），
#       再在远程 shell 里 mv 到目标目录（远程 shell 会正确展开 ~）。
printf '\033[36m[deploy] 上传 %s ...\033[0m\n' "$TARBALL"
scp "${SCP_OPTS[@]}" "$TARBALL" "$DEST:_stock_ai_deploy.tar.gz"

# ---- 远程解包 ----
printf '\033[36m[deploy] 远程解包到 %s ...\033[0m\n' "$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$DEST" \
    "mv _stock_ai_deploy.tar.gz $REMOTE_DIR/ && cd $REMOTE_DIR && tar xzf _stock_ai_deploy.tar.gz && rm -f _stock_ai_deploy.tar.gz && chmod +x run.sh scripts/*.sh 2>/dev/null || true"

printf '\033[32m[deploy] 完成。远程目录: %s\033[0m\n' "$REMOTE_DIR"
printf '\033[32m[deploy] 下一步（远程执行）:\033[0m\n'
printf '  ssh %s "cd %s && bash scripts/setup_env.sh"\n' "$DEST" "$REMOTE_DIR"
