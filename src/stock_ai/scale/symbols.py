"""符号名单载入、年份区间解析、配置指纹。

- :func:`load_symbols`：读取 ``scale.symbols_file``（``symbols: [...]``）。
- :func:`resolve_year_range`：Q1/主理人裁决 U1 —— **动态取系统时钟**，
  默认返回最近 20 个**完整**自然年 ``(today.year - 20, today.year - 1)``；
  当前（2026）→ ``(2006, 2025)``；``scale.years`` 显式配置可覆盖。
- :func:`config_hash`：``sha256:`` 前缀的确定性配置指纹（AC3 复现审计）。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from stock_ai.scale.errors import ScaleError

__all__ = ["load_symbols", "resolve_year_range", "config_hash", "DEFAULT_YEARS_BACK"]

#: 默认回看的完整自然年数量（「最近 20 年」）。
DEFAULT_YEARS_BACK: int = 20

#: A 股 6 位代码（沪深主/科创/创业板）。
_CODE_RE = re.compile(r"^\d{6}$")


def load_symbols(symbols_file: Path) -> list[str]:
    """从 YAML 名单文件载入股票代码列表（去重、保持稳定顺序）。

    Args:
        symbols_file: 名单文件路径（``symbols: [ "600519", ... ]``）。

    Returns:
        6 位数字代码列表（已去重、已排序）。

    Raises:
        ScaleError: 文件不存在 / YAML 解析失败 / ``symbols`` 字段缺失或非法。
    """
    import yaml

    p = Path(symbols_file)
    if not p.exists():
        raise ScaleError(
            f"股票名单文件不存在: {p}", context={"path": str(p)}
        )
    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001 - 依赖 yaml 具体异常
        raise ScaleError(
            f"股票名单解析失败: {p} ({exc})", context={"path": str(p)}
        ) from exc

    if not isinstance(raw, dict) or "symbols" not in raw:
        raise ScaleError(
            f"股票名单格式非法（缺少 symbols 字段）: {p}", context={"path": str(p)}
        )

    items = raw.get("symbols") or []
    if not isinstance(items, list):
        raise ScaleError(f"symbols 必须是列表: {p}", context={"path": str(p)})

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        code = str(item).strip()
        if not code:
            continue
        if not _CODE_RE.match(code):
            raise ScaleError(
                f"非法股票代码 {item!r}（应为 6 位数字，如 600519/000001/300750/688981）: {p}",
                context={"symbol": item, "path": str(p)},
            )
        if code in seen:
            continue
        seen.add(code)
        out.append(code)

    if not out:
        raise ScaleError(f"股票名单为空: {p}", context={"path": str(p)})

    out.sort()
    return out


def resolve_year_range(
    years: list[int] | None,
    today: date | None = None,
) -> tuple[int, int]:
    """解析年份闭区间 ``(start_year, end_year)``。

    规则（主理人裁决 U1，**覆盖架构文档 Part C 的错误表述**）：

    - ``years`` 显式给出（``[start, end]`` 或年份列表）→ 取 ``(min, max)``；
    - 否则 **动态取系统时钟**：``(today.year - 20, today.year - 1)``，
      即**最近 20 个完整自然年**（不含当前未完结年份）；
    - 当前系统年（2026）→ ``(2006, 2025)``。

    Args:
        years: 显式年份配置；``None`` 表示动态推导。
        today: 基准日期（默认 ``date.today()``，测试可注入）。

    Returns:
        ``(start_year, end_year)`` 含端点闭区间。

    Raises:
        ScaleError: ``years`` 非空但无法解析为至少 2 个整数。
    """
    if years:
        try:
            nums = [int(y) for y in years]
        except (TypeError, ValueError) as exc:
            raise ScaleError(
                f"scale.years 非法: {years!r}（应为形如 [2006, 2025] 的年份列表）",
                context={"years": years},
            ) from exc
        if len(nums) < 2:
            raise ScaleError(
                f"scale.years 至少需要 2 个年份（起、止）: {years!r}",
                context={"years": years},
            )
        start, end = min(nums), max(nums)
        if start > end:
            raise ScaleError(
                f"scale.years 起始年 {start} 晚于结束年 {end}", context={"years": years}
            )
        return (int(start), int(end))

    base = today or date.today()
    return (int(base.year) - DEFAULT_YEARS_BACK, int(base.year) - 1)


def config_hash(payload: dict[str, Any]) -> str:
    """对配置载荷计算确定性 sha256 指纹（``sha256:`` 前缀）。

    载荷按键排序后以紧凑 JSON 序列化，保证同输入 → 同指纹（AC3）。

    Args:
        payload: 待指纹化的配置字典（``scale`` 段 + 名单来源 + 年份 + 比例 + 种子）。

    Returns:
        形如 ``sha256:<hex>`` 的字符串。
    """
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
