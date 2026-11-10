# ==============================================================
# run.ps1 - stock_ai 统一入口脚本
#
# 用法:
#   .\run.ps1                     # 默认显示项目数据/模型/报告全景摘要(status)
#   .\run.ps1 status              # 同上
#   .\run.ps1 help                # 显示帮助
#   .\run.ps1 <命令> [参数]       # 调用对应功能
#   .\run.ps1 <命令> -Help        # 显示该命令的完整 CLI 帮助
#   .\run.ps1 cli -CliArgs "..."  # 透传任意参数组合给 python -m stock_ai
#
# 若提示禁止运行脚本, 先执行:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ==============================================================

param(
    [Parameter(Position = 0)]
    [string]$Cmd = "status",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @(),

    # ---- 常用快捷参数 (映射到各子命令的 CLI 选项) ----
    [string]$Symbols = "",      # 股票代码, 逗号分隔: 600519,000001
    [string]$Start = "",        # 起始日期 YYYYMMDD
    [string]$End = "",          # 结束日期 YYYYMMDD
    [string]$Source = "",       # akshare | tencent | eastmoney | sina | synthetic
    [string]$Adjust = "",       # qfq | hfq | none (复权方式, 默认 none 不复权)
    [switch]$Force,             # 忽略缓存强制重拉
    [double]$Throttle = 0,      # 每只之间的间隔秒 (规避行情源窗口风控)
    [switch]$AllowSynthetic,    # 允许真实源全失败时写 synthetic 兜底 (默认禁止)
    [string]$Algo = "",         # ppo | sac | dqn
    [int]$Timesteps = 0,        # 训练步数
    [string]$Backend = "",      # auto | sb3 | numpy
    [string]$Model = "",        # 模型文件路径 (backtest/predict 用)
    [string]$Out = "",          # 输出目录覆盖
    [string]$Config = "",       # 配置文件路径
    [int]$Port = 0,             # scale-ui 监听端口
    [int]$Seed = 0,             # 随机种子覆盖 (0 = 不覆盖)
    [string]$CliArgs = "",      # cli 透传: 完整参数串, 如 "data fetch --symbols 600519 --force"
    [switch]$Help,              # 显示该命令的 CLI help
    [switch]$H
)

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location $ROOT

# ---- 强制 UTF-8: 保证 python 子进程的中文输出经管道/重定向时不乱码 ----
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ---- 定位 Python (优先受管 venv) ----
$VENV_PY = "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if (Test-Path $VENV_PY) { $PY = $VENV_PY }
else {
    $PY = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PY) { Write-Host "[错误] 未找到 Python, 请先运行 scripts\setup_env.ps1" -ForegroundColor Red; exit 1 }
    Write-Host "[提示] 未找到受管 venv, 使用系统 python: $PY" -ForegroundColor Yellow
}

# ---------------- 帮助 ----------------
function Show-Help {
    Write-Host ""
    Write-Host "stock_ai 统一入口  (解释器: $PY)" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor DarkCyan
    Write-Host "数据      .\run.ps1 data-fetch    [-Symbols 600519,000001] [-Start 20120101] [-End 20241231] [-Source akshare] [-Adjust qfq|hfq|none] [-Force]   # -Adjust 默认 none 不复权"
    Write-Host "          .\run.ps1 data-refresh  [-Symbols 600519] [-Adjust qfq|hfq|none]   # 增量刷新(只拉缺失日期), -Adjust 默认 none 不复权"
    Write-Host "          .\run.ps1 data-info                                # 缓存概览"
    Write-Host "样本      .\run.ps1 sample-build  [-Config configs\default.yaml]"
    Write-Host "训练      .\run.ps1 train         [-Algo ppo|sac|dqn] [-Timesteps 200000] [-Backend auto|sb3|numpy]"
    Write-Host "回测      .\run.ps1 backtest      [-Model models\ppo_xxx.zip]"
    Write-Host "预测      .\run.ps1 predict       [-Model models\ppo_xxx.zip] [-Start 20250101] [-End 20250601]"
    Write-Host "自检      .\run.ps1 doctor"
    Write-Host "摘要      .\run.ps1 status                                   # 数据/样本/模型/报告全景 (默认)"
    Write-Host "大规模    .\run.ps1 scale-fetch    [-Source sina] [-Throttle 0.5] [-Force]  # 拉取沪深300 最近20年 不复权日线"
    Write-Host "          .\run.ps1 scale-units    [-Seed 42]                  # 生成 units.json + split.json (可复现)"
    Write-Host "          .\run.ps1 scale-train    [-Timesteps 20000] [-Algo ppo] [-Backend auto]  # 长期训练(可中断/续跑)"
    Write-Host "          .\run.ps1 scale-detect                               # 检测编排(选模型 + 回测)"
    Write-Host "          .\run.ps1 scale-status                               # 打印训练进度(总数/完成/失败/剩余/ETA)"
    Write-Host "          .\run.ps1 scale-ui       [-Port 8765]                # 启动本机 UI (127.0.0.1)"
    Write-Host "==============================================================" -ForegroundColor DarkCyan
    Write-Host "查看某命令完整参数:  .\run.ps1 <命令> -Help      (如 .\run.ps1 train -Help)"
    Write-Host "透传任意 CLI 参数:   .\run.ps1 cli -CliArgs `"data fetch --symbols 600519 --force`""
    Write-Host ""
}

# ---------------- 通用调用 ----------------
function Invoke-Cli {
    param([string[]]$CliArgs, [switch]$PassHelp)
    $all = @($CliArgs)
    if ($Help -or $H -or $PassHelp) { $all += "--help" }
    Write-Host ">> python -m stock_ai $($all -join ' ')" -ForegroundColor DarkGray
    & $PY -m stock_ai @all
    exit $LASTEXITCODE
}

# 把快捷参数映射为通用 CLI 片段
function Get-CommonArgs {
    $a = @()
    if ($Config) { $a += @("--config", $Config) }
    return $a
}

# ---------------- status: 全景摘要 ----------------
function Format-Pct { param($v) if ($null -eq $v) { return "-" }; return ("{0:P2}" -f [double]$v) }
function Format-Num { param($v, [string]$fmt = "F3") if ($null -eq $v -or $v -eq "-") { return "-" }; return ("{0:$fmt}" -f [double]$v) }
function Format-Int { param($v) if ($null -eq $v -or $v -eq "-") { return "-" }; return ("{0:N0}" -f [long][double]$v) }

function Get-JsonProp { param($obj, [string]$name, $default = $null)
    if ($null -eq $obj) { return $default }
    $p = $obj.PSObject.Properties[$name]
    if ($null -eq $p) { return $default }
    return $p.Value
}

function Show-Status {
    $ver = (& $PY -m stock_ai --version 2>$null)
    Write-Host ""
    Write-Host "================ stock_ai 项目状态 ================" -ForegroundColor Cyan
    Write-Host "项目根目录 : $ROOT"
    Write-Host "解释器     : $PY"
    Write-Host "包版本     : $ver"
    Write-Host ""

    # ---- 1) 原始行情数据 ----
    Write-Host "[1] 原始日线数据  (data\raw)" -ForegroundColor Yellow
    $rawDir = Join-Path $ROOT "data\raw"
    $metaPath = Join-Path $rawDir "meta.json"
    if (Test-Path $metaPath) {
        $meta = Get-Content $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $syms = Get-JsonProp $meta "symbols"
        if ($null -ne $syms) {
            $props = $syms.PSObject.Properties
            Write-Host ("  共 {0} 只股票:" -f ($props | Measure-Object).Count)
            foreach ($p in $props) {
                $s = $p.Value
                Write-Host ("    {0}: {1} 行  {2} ~ {3}  来源={4}  复权={5}  缺口={6} 天  更新={7}" -f `
                    $p.Name, (Get-JsonProp $s "rows" "-"), (Get-JsonProp $s "start" "-"), `
                    (Get-JsonProp $s "end" "-"), (Get-JsonProp $s "source" "-"), `
                    (Get-JsonProp $s "adjust" "-"),
                    (Get-JsonProp $s "missing_days" "0"), (Get-JsonProp $s "updated_at" "-"))
            }
        }
    } elseif (Test-Path $rawDir) {
        $pq = Get-ChildItem $rawDir -Filter *.parquet -ErrorAction SilentlyContinue
        Write-Host ("  无 meta.json; 发现 {0} 个 parquet 文件 (建议运行 data-info 查看)" -f ($pq | Measure-Object).Count)
        foreach ($f in $pq) { Write-Host ("    {0}  ({1:N0} KB)" -f $f.Name, ($f.Length / 1KB)) }
    } else {
        Write-Host "  (空) 尚未拉取数据, 运行: .\run.ps1 data-fetch -Symbols 600519" -ForegroundColor DarkGray
    }
    Write-Host ""

    # ---- 2) 训练/验证/测试样本 ----
    Write-Host "[2] 样本数据  (data\samples)" -ForegroundColor Yellow
    $spDir = Join-Path $ROOT "data\samples"
    $spMeta = Join-Path $spDir "meta.json"
    if (Test-Path $spMeta) {
        $m = Get-Content $spMeta -Raw -Encoding UTF8 | ConvertFrom-Json
        $tr = Get-JsonProp $m "train_range";  $te = Get-JsonProp $m "test_range"
        $rows = Get-JsonProp $m "rows"
        $feats = @(Get-JsonProp $m "feature_names" @())
        $syms = @(Get-JsonProp $m "symbols" @())
        Write-Host ("  股票: {0}   窗口: {1} 步   特征: {2} 维   embargo: {3} 交易日   seed: {4}" -f `
            ($syms -join ","), (Get-JsonProp $m "window" "-"), $feats.Count, `
            (Get-JsonProp $m "embargo_days" "0"), (Get-JsonProp $m "seed" "-"))
        Write-Host ("  划分: 训练 {0} 步 ({1}~{2}) | 验证 {3} 步 | 测试 {4} 步 ({5}~{6})" -f `
            (Get-JsonProp $rows "train" "-"), (Get-JsonProp $tr "start" "-"), (Get-JsonProp $tr "end" "-"), `
            (Get-JsonProp $rows "val" "-"), (Get-JsonProp $rows "test" "-"), `
            (Get-JsonProp $te "start" "-"), (Get-JsonProp $te "end" "-"))
        $dh = [string](Get-JsonProp $m "data_hash" "")
        Write-Host ("  data_hash: {0}...   生成时间: {1}" -f $dh.Substring(0, [Math]::Min(8, $dh.Length)), (Get-JsonProp $m "created_at" "-"))
        foreach ($n in @("train", "val", "test")) {
            $f = Join-Path $spDir "$n.npz"
            if (Test-Path $f) { Write-Host ("    {0}.npz  ({1:N0} KB)" -f $n, ((Get-Item $f).Length / 1KB)) }
            else { Write-Host "    $n.npz  (缺失!)" -ForegroundColor Red }
        }
    } else {
        Write-Host "  (空) 尚未构建样本, 运行: .\run.ps1 sample-build" -ForegroundColor DarkGray
    }
    Write-Host ""

    # ---- 3) 训练产物模型 ----
    Write-Host "[3] 训练模型  (models)" -ForegroundColor Yellow
    $models = Get-ChildItem (Join-Path $ROOT "models") -Filter *.zip -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 5
    if ($models) {
        foreach ($f in $models) {
            $side = [IO.Path]::ChangeExtension($f.FullName, ".json")
            $algo = "-"; $backend = "-"; $steps = "-"; $created = $f.LastWriteTime
            if (Test-Path $side) {
                $sc = Get-Content $side -Raw -Encoding UTF8 | ConvertFrom-Json
                $algo = Get-JsonProp $sc "algo" "-"
                $backend = Get-JsonProp $sc "backend" "-"
                $tc = Get-JsonProp $sc "train_cfg"
                $steps = Get-JsonProp $tc "timesteps" "-"
                $created = Get-JsonProp $sc "created_at" $created
            }
            Write-Host ("    {0}  [{1}/{2}]  {3} 步  {4:N0} KB  {5}" -f `
                $f.Name, $algo, $backend, $steps, ($f.Length / 1KB), $created)
        }
    } else {
        Write-Host "  (空) 尚未训练, 运行: .\run.ps1 train -Algo ppo" -ForegroundColor DarkGray
    }
    Write-Host ""

    # ---- 4) 回测报告 ----
    Write-Host "[4] 回测报告  (reports\backtest_*.json, 最近 5 份)" -ForegroundColor Yellow
    $bts = Get-ChildItem (Join-Path $ROOT "reports") -Filter backtest_*.json -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 5
    if ($bts) {
        foreach ($f in $bts) {
            try {
                $r = Get-Content $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
                $mt = Get-JsonProp $r "meta"; $me = Get-JsonProp $r "metrics"
                Write-Host ("    {0}" -f $f.Name)
                Write-Host ("      模型: {0} [{1}/{2}]  测试区间: {3}~{4}" -f `
                    (Get-JsonProp $mt "model_path" "-"), (Get-JsonProp $mt "algo" "-"), `
                    (Get-JsonProp $mt "backend" "-"), `
                    (Get-JsonProp (Get-JsonProp $mt "test_range") "start" "-"), `
                    (Get-JsonProp (Get-JsonProp $mt "test_range") "end" "-"))
                Write-Host ("      年化 {0}  累计 {1}  最大回撤 {2}  夏普 {3}  胜率 {4}  交易 {5} 笔  基准累计 {6}" -f `
                    (Format-Pct (Get-JsonProp $me "annual_return")), `
                    (Format-Pct (Get-JsonProp $me "total_return")), `
                    (Format-Pct (Get-JsonProp $me "max_drawdown")), `
                    (Format-Num (Get-JsonProp $me "sharpe")), `
                    (Format-Pct (Get-JsonProp $me "win_rate")), `
                    (Format-Int (Get-JsonProp $me "num_trades")), `
                    (Format-Pct (Get-JsonProp $me "benchmark_return")))
            } catch { Write-Host ("    {0}  (解析失败: {1})" -f $f.Name, $_.Exception.Message) -ForegroundColor Red }
        }
    } else {
        Write-Host "  (空) 尚未回测, 运行: .\run.ps1 backtest" -ForegroundColor DarkGray
    }
    Write-Host ""

    # ---- 5) 预测输出 ----
    Write-Host "[5] 预测输出  (reports\predict_*.csv, 最近 5 份)" -ForegroundColor Yellow
    $pds = Get-ChildItem (Join-Path $ROOT "reports") -Filter predict_*.csv -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 5
    if ($pds) {
        foreach ($f in $pds) {
            $lines = (Get-Content $f.FullName | Measure-Object -Line).Lines
            Write-Host ("    {0}  {1} 条预测  {2:N0} KB  {3}" -f $f.Name, ([Math]::Max(0, $lines - 1)), ($f.Length / 1KB), $f.LastWriteTime)
        }
    } else {
        Write-Host "  (空) 尚未预测, 运行: .\run.ps1 predict" -ForegroundColor DarkGray
    }
    Write-Host ""

    # ---- 6) 最近日志 ----
    Write-Host "[6] 最近日志  (logs)" -ForegroundColor Yellow
    $logs = Get-ChildItem (Join-Path $ROOT "logs") -File -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 3
    if ($logs) { foreach ($f in $logs) { Write-Host ("    {0}  {1}" -f $f.FullName.Substring($ROOT.Length + 1), $f.LastWriteTime) } }
    else { Write-Host "  (空)" -ForegroundColor DarkGray }
    Write-Host ""
    exit 0
}

# ---------------- 命令分发 ----------------
$cmdKey = $Cmd.ToLower().TrimStart("-", "/")

# cli 透传模式: run.ps1 cli -CliArgs "<完整参数串>"
# (PowerShell 5.1 会把 --xxx token 尝试绑定到本脚本参数, 故透传须用单字符串)
if ($cmdKey -eq "cli") {
    if (-not $CliArgs) {
        Write-Host "[错误] cli 透传需要 -CliArgs 参数" -ForegroundColor Red
        Write-Host '  示例: .\run.ps1 cli -CliArgs "data fetch --symbols 600519 --force"' -ForegroundColor Yellow
        exit 1
    }
    Write-Host ">> python -m stock_ai $CliArgs" -ForegroundColor DarkGray
    Invoke-Expression "& '$PY' -m stock_ai $CliArgs"
    exit $LASTEXITCODE
}

# help / 无参数
if ($cmdKey -in @("", "help", "h")) {
    if ($Rest.Count -gt 0) {
        # run.ps1 help train -> 显示 train 的 CLI help
        Write-Host ">> python -m stock_ai $($Rest -join ' ') --help" -ForegroundColor DarkGray
        & $PY -m stock_ai @Rest --help
        exit $LASTEXITCODE
    }
    Show-Help
    exit 0
}

# 透传模式: run.ps1 -- <任意参数>
if ($cmdKey -eq "" -or $Cmd -eq "--") {
    & $PY -m stock_ai @Rest
    exit $LASTEXITCODE
}

$common = Get-CommonArgs

switch ($cmdKey) {
    "data-fetch" {
        $a = @("data", "fetch") + $common
        if ($Symbols) { foreach ($s in ($Symbols -split ",")) { $a += @("--symbols", $s.Trim()) } }
        if ($Start)   { $a += @("--start", $Start) }
        if ($End)     { $a += @("--end", $End) }
        if ($Source)  { $a += @("--source", $Source) }
        if ($Adjust)  { $a += @("--adjust", $Adjust) }
        if ($Force)   { $a += "--force" }
        Invoke-Cli $a
    }
    "data-refresh" {
        $a = @("data", "refresh") + $common
        if ($Symbols) { foreach ($s in ($Symbols -split ",")) { $a += @("--symbols", $s.Trim()) } }
        if ($Adjust)  { $a += @("--adjust", $Adjust) }
        Invoke-Cli $a
    }
    "data-info" {
        $a = @("data", "info") + $common
        if ($Symbols) { foreach ($s in ($Symbols -split ",")) { $a += @("--symbols", $s.Trim()) } }
        Invoke-Cli $a
    }
    "sample-build" {
        $a = @("sample", "build") + $common + @($Rest)
        Invoke-Cli $a
    }
    "train" {
        $a = @("train") + $common
        if ($Algo)      { $a += @("--algo", $Algo) }
        if ($Timesteps) { $a += @("--timesteps", $Timesteps) }
        if ($Backend)   { $a += @("--backend", $Backend) }
        if ($Out)       { $a += @("--out", $Out) }
        Invoke-Cli $a
    }
    "backtest" {
        $a = @("backtest") + $common
        if ($Model) { $a += @("--model", $Model) }
        if ($Out)   { $a += @("--out", $Out) }
        Invoke-Cli $a
    }
    "predict" {
        $a = @("predict") + $common
        if ($Model) { $a += @("--model", $Model) }
        if ($Start) { $a += @("--start", $Start) }
        if ($End)   { $a += @("--end", $End) }
        if ($Out)   { $a += @("--out", $Out) }
        Invoke-Cli $a
    }
    "doctor"   { Invoke-Cli (@("doctor") + $common) }
    "scale-fetch" {
        $a = @("scale", "fetch") + $common
        if ($Source) { $a += @("--source", $Source) }
        if ($Seed)   { $a += @("--seed", $Seed) }
        if ($Throttle) { $a += @("--throttle", $Throttle) }
        if ($AllowSynthetic) { $a += @("--allow-synthetic") }
        if ($Rest.Count -gt 0) { $a += $Rest }
        Invoke-Cli $a
    }
    "scale-units" {
        $a = @("scale", "units") + $common
        if ($Seed) { $a += @("--seed", $Seed) }
        Invoke-Cli $a
    }
    "scale-train" {
        $a = @("scale", "train") + $common
        if ($Seed)      { $a += @("--seed", $Seed) }
        if ($Timesteps) { $a += @("--timesteps", $Timesteps) }
        if ($Algo)      { $a += @("--algo", $Algo) }
        if ($Backend)   { $a += @("--backend", $Backend) }
        Invoke-Cli $a
    }
    "scale-detect" {
        $a = @("scale", "detect") + $common
        if ($Seed) { $a += @("--seed", $Seed) }
        Invoke-Cli $a
    }
    "scale-status" {
        $a = @("scale", "status") + $common
        Invoke-Cli $a
    }
    "scale-ui" {
        $a = @("scale", "ui") + $common
        if ($Port) { $a += @("--port", $Port) }
        Invoke-Cli $a
    }
    "status"   { Show-Status }
    default {
        Write-Host "[错误] 未知命令: $Cmd" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
