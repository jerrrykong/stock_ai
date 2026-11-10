"""检测阶段的模型选取策略（Q6 / PRD §5.4）。

| 策略 | 规则 | 无匹配时 |
|---|---|---|
| ``nearest_past``（默认） | 同代码训练组中 ``train_year <= detect_year`` 且**最大**者 | 回退 ``max_year``；再无 → ``""`` |
| ``nearest`` | 同代码训练组中 ``|train_year - detect_year|`` 最小；并列取较小年份 | 无 → ``""`` |
| ``max_year`` | 同代码训练组中年份最大者 | 无 → ``""`` |
| ``none`` | 不自动选，全部跳过 | 全部 ``""`` |

时间因果优先：默认 ``nearest_past`` 避免用「未来年份」模型评估「过去年份」检测集造成前视。

``policy`` 不在上表白名单内时 :meth:`ModelSelector.select` 直接抛
:class:`~stock_ai.scale.errors.ScaleError`（**不静默回退**到 ``nearest_past``）。
"""

from __future__ import annotations

from stock_ai.scale.errors import ScaleError
from stock_ai.scale.units import Unit

__all__ = ["ModelSelector", "SUPPORTED_POLICIES"]

#: 支持的策略白名单。
SUPPORTED_POLICIES: frozenset[str] = frozenset(
    {"nearest_past", "nearest", "max_year", "none"}
)


class ModelSelector:
    """检测模型选取器（无状态）。"""

    @staticmethod
    def select(
        policy: str,
        symbol: str,
        detect_year: int,
        train_units: list[str],
        available: set[str],
    ) -> str:
        """按策略选出用于检测的模型 ``unit_id``。

        Args:
            policy: 策略名（见 :data:`SUPPORTED_POLICIES`）。
            symbol: 检测单元代码。
            detect_year: 检测单元年份。
            train_units: 划分计划中的训练单元 ``unit_id`` 列表。
            available: 实际存在模型产物的 ``unit_id`` 集合。

        Returns:
            选中的模型 ``unit_id``；无匹配返回 ``""``。

        Raises:
            ScaleError: ``policy`` 不在 :data:`SUPPORTED_POLICIES` 白名单内
                （**明确报错**，绝不静默回退到 ``nearest_past``）。
        """
        pol = str(policy or "nearest_past").lower()
        if pol not in SUPPORTED_POLICIES:
            raise ScaleError(
                f"未知的检测模型策略: {policy!r}；"
                f"支持的策略为 {sorted(SUPPORTED_POLICIES)}",
                context={"policy": str(policy), "supported": sorted(SUPPORTED_POLICIES)},
            )
        if pol == "none":
            return ""

        # 仅保留：同代码 + 训练组 + 模型可用 的候选
        candidates: list[tuple[int, str]] = []
        for uid in train_units:
            if uid not in available:
                continue
            try:
                u = Unit.parse(uid)
            except ValueError:
                continue
            if u.symbol == symbol:
                candidates.append((u.year, uid))

        if not candidates:
            return ""

        if pol == "max_year":
            return max(candidates, key=lambda t: (t[0], t[1]))[1]

        if pol == "nearest":
            # |Δ| 最小；并列取较小年份
            return min(
                candidates, key=lambda t: (abs(t[0] - int(detect_year)), t[0])
            )[1]

        # nearest_past（默认）
        past = [c for c in candidates if c[0] <= int(detect_year)]
        if past:
            return max(past, key=lambda t: (t[0], t[1]))[1]
        # 回退 max_year（该代码训练组最新年份）
        return max(candidates, key=lambda t: (t[0], t[1]))[1]
