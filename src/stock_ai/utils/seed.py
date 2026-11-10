"""全局种子对齐（K4）。

所有随机性只从 ``AppConfig.random_seed`` 读取；入口处必须调用
:func:`set_global_seed`。禁止模块内自行 ``np.random.seed()``。
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_global_seed"]


def set_global_seed(seed: int) -> None:
    """对齐 Python / NumPy（以及可选 torch）随机种子。

    Args:
        seed: 全局随机种子。
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch 为可选依赖：延迟 import，任何失败都不影响主链路
    try:  # pragma: no cover - 取决于运行环境是否装 torch
        import importlib
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            torch = importlib.import_module("torch")
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover - CPU 环境
                torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass
