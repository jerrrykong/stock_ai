"""配置对象（Configuration Object）。

按架构文档 §4.1 定义 ``AppConfig`` 及其子配置 dataclass。
所有配置仅在 ``cli/`` 层通过 CLI 参数覆盖；模块内部只接受 config 对象。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from stock_ai.utils.errors import ConfigError

__all__ = [
    "DataConfig",
    "SampleConfig",
    "EnvConfig",
    "TrainConfig",
    "PathConfig",
    "ScaleTrainConfig",
    "ScaleDetectConfig",
    "ScaleConfig",
    "AppConfig",
    "SUPPORTED_ADJUSTS",
    "normalize_adjust",
]

#: 支持的复权方式白名单（A：``--adjust``）。``none`` 表示不复权。
SUPPORTED_ADJUSTS: frozenset[str] = frozenset({"qfq", "hfq", "none"})


def normalize_adjust(value: object) -> str:
    """归一化复权方式并校验白名单（A：``--adjust``）。

    - ``None`` / ``""`` / 纯空白 —— 历史遗留的"不复权"写法 —— 归一化为 ``"none"``；
    - 其余值小写化后必须在 :data:`SUPPORTED_ADJUSTS` 内。

    Args:
        value: 用户/配置给出的复权方式。

    Returns:
        规范化后的 ``"qfq"`` / ``"hfq"`` / ``"none"``。

    Raises:
        ConfigError: 非法复权方式。
    """
    raw = str(value or "").strip().lower()
    if raw == "":
        return "none"
    if raw not in SUPPORTED_ADJUSTS:
        raise ConfigError(
            f"不支持的复权方式: {value!r}（可选: {sorted(SUPPORTED_ADJUSTS)}）",
            context={"adjust": value, "valid": sorted(SUPPORTED_ADJUSTS)},
        )
    return raw


@dataclass
class DataConfig:
    """数据层配置。"""

    symbols: list[str] = field(default_factory=list)
    start: str = "20180101"          # YYYYMMDD
    end: str = "20241231"
    source: str = "akshare"          # akshare|tencent|eastmoney|sina|synthetic
    cache_dir: Path = Path("data/raw")
    adjust: str = "none"             # qfq|hfq|none（前复权/后复权/不复权，默认不复权）
    retry_times: int = 3
    retry_backoff: float = 1.5       # 指数退避基数(秒)
    timeout: float = 15.0

    def __post_init__(self) -> None:
        """构造时归一化并校验 ``adjust``（非法值立即抛 ConfigError）。"""
        self.adjust = normalize_adjust(self.adjust)


@dataclass
class SampleConfig:
    """样本层配置。"""

    window: int = 20                 # 历史窗口 N 步
    features: list[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ma5",
            "ma10",
            "ma20",
            "momentum",
            "volatility",
            "volume_ratio",
        ]
    )
    train_start: str = "20180101"
    train_end: str = "20211231"
    test_start: str = "20220101"
    test_end: str = "20241231"
    embargo_days: int = 5            # 训练/测试边界隔离
    normalize: bool = True           # 统计量仅 train 窗口 fit
    val_ratio: float = 0.15          # 从训练窗口尾部切出的验证集比例


@dataclass
class EnvConfig:
    """A 股交易环境配置。"""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.00025  # 佣金 0.025% 双边
    stamp_tax_rate: float = 0.0005    # 印花税 0.05% 仅卖出
    slippage_rate: float = 0.001      # 滑点 0.1%
    min_commission: float = 5.0       # 最低佣金 5 元
    reward_type: str = "drawdown_penalty"  # gross|drawdown_penalty|sharpe|excess
    lambda_drawdown: float = 0.1
    lambda_turnover: float = 0.001
    lambda_leverage: float = 0.01
    max_position: float = 1.0         # 最大仓位比例(v1 单标的=1.0)
    lot_size: int = 100               # A 股 100 股整手


@dataclass
class TrainConfig:
    """训练层配置。"""

    algo: str = "ppo"                 # ppo|sac|dqn
    timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.0
    train_split: float = 1.0          # npz 内已切分，此处仅微调
    val_split: float = 0.0
    torch_threads: int = 1            # Windows 下防止线程爆炸
    backend: str = "auto"             # auto|sb3|numpy
    out_dir: Path = Path("models")


@dataclass
class PathConfig:
    """运行时目录配置。"""

    root: Path = Path(".")
    raw: Path = Path("data/raw")
    samples: Path = Path("data/samples")
    models: Path = Path("models")
    logs: Path = Path("logs")
    reports: Path = Path("reports")


@dataclass
class ScaleTrainConfig:
    """单元级训练配置（大规模训练编排，Q4）。

    Attributes:
        timesteps: 单个单元的目标训练步数。
        algo: 训练算法（``ppo|sac|dqn``）。
        backend: 后端偏好（``auto|sb3|numpy``）。
    """

    timesteps: int = 20_000
    algo: str = "ppo"
    backend: str = "auto"


@dataclass
class ScaleDetectConfig:
    """单元级检测配置（Q5/Q6）。

    Attributes:
        model_policy: 模型选取策略（``nearest_past|nearest|max_year|none``）。
        mode: 检测模式（首版仅 ``backtest``）。
    """

    model_policy: str = "nearest_past"
    mode: str = "backtest"


@dataclass
class ScaleConfig:
    """大规模训练 / 检测编排配置（``configs/*.yaml`` 新增 ``scale:`` 段）。

    该配置段整体为**增量叠加**：不影响既有 6 条 CLI 的任何行为。

    Attributes:
        years: 显式年份区间 ``[start, end]``；``None`` 时动态取「最近 20 个完整自然年」。
        symbols_file: 沪深300 名单文件（由 ``resolve_paths`` 解析为绝对路径）。
        split_ratio: 训练/检测划分比例，默认 ``{train: 0.8, detect: 0.2}``。
        workers: 并发度（P2 预留，首版严格串行 = 1）。
        poll_sec: 控制/状态轮询间隔（秒）。
        train: 单元级训练配置。
        detect: 单元级检测配置。
    """

    years: list[int] | None = None
    symbols_file: Path = Path("configs/symbols_hs300_full.yaml")
    split_ratio: dict = field(
        default_factory=lambda: {"train": 0.8, "detect": 0.2}
    )
    workers: int = 1
    poll_sec: float = 2.0
    train: ScaleTrainConfig = field(default_factory=ScaleTrainConfig)
    detect: ScaleDetectConfig = field(default_factory=ScaleDetectConfig)


@dataclass
class AppConfig:
    """全链路根配置对象。"""

    data: DataConfig = field(default_factory=DataConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    random_seed: int = 42
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """从 YAML 文件载入配置。

        Args:
            path: YAML 配置文件路径。

        Returns:
            解析后的 :class:`AppConfig`。

        Raises:
            ConfigError: 文件不存在或 YAML 解析失败。
        """
        import yaml

        p = Path(path)
        if not p.exists():
            raise ConfigError(f"配置文件不存在: {p}", context={"path": str(p)})
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:  # pragma: no cover - 依赖 yaml 具体异常
            raise ConfigError(
                f"配置文件解析失败: {p} ({exc})", context={"path": str(p)}
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件顶层必须是 mapping: {p}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppConfig":
        """从普通 dict 构造 :class:`AppConfig`（未知键被忽略）。"""
        if not isinstance(d, dict):
            raise ConfigError("AppConfig.from_dict 需要 dict 输入")

        plain: dict[str, Any] = dict(d)

        data = DataConfig(**_pick(plain.get("data", {}), DataConfig))
        sample = SampleConfig(**_pick(plain.get("sample", {}), SampleConfig))
        env = EnvConfig(**_pick(plain.get("env", {}), EnvConfig))
        train = TrainConfig(**_pick(plain.get("train", {}), TrainConfig))
        paths = PathConfig(**_pick(plain.get("paths", {}), PathConfig))
        scale = _build_scale(plain.get("scale", {}))

        # Path 类型字段由字符串转换
        data.cache_dir = Path(data.cache_dir)
        train.out_dir = Path(train.out_dir)
        for f in fields(PathConfig):
            setattr(paths, f.name, Path(getattr(paths, f.name)))

        cfg = cls(
            data=data,
            sample=sample,
            env=env,
            train=train,
            paths=paths,
            scale=scale,
            random_seed=int(plain.get("random_seed", 42)),
            log_level=str(plain.get("log_level", "INFO")),
        )
        return cfg

    # ------------------------------------------------------------------ #
    # 覆盖合并
    # ------------------------------------------------------------------ #
    def merge_cli_overrides(self, overrides: dict[str, Any]) -> "AppConfig":
        """应用 CLI 覆盖（CLI > YAML > 代码默认值）。

        Args:
            overrides: 形如 ``{"data.symbols": [...], "train.algo": "ppo"}`` 的扁平字典，
                值为 ``None`` 的键将被忽略。

        Returns:
            ``self``（原地修改后返回，便于链式调用）。
        """
        for dotted, value in overrides.items():
            if value is None or dotted is None:
                continue
            if "." not in dotted:
                # 顶层字段
                if hasattr(self, dotted):
                    setattr(self, dotted, value)
                continue
            section, _, key = dotted.partition(".")
            target = getattr(self, section, None)
            if target is None or not hasattr(target, key):
                continue
            current = getattr(target, key)
            if isinstance(current, Path):
                setattr(target, key, Path(value))
            else:
                setattr(target, key, value)
        return self

    # ------------------------------------------------------------------ #
    # 导出
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON/YAML 序列化的普通 dict（``Path`` -> ``str``）。"""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        return _convert(asdict(self))

    def ensure_dirs(self) -> None:
        """确保所有运行时目录存在。"""
        for p in (
            self.paths.raw,
            self.paths.samples,
            self.paths.models,
            self.paths.logs,
            self.paths.reports,
            self.data.cache_dir,
            self.train.out_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)


def _pick(raw: Any, cls: type) -> dict[str, Any]:
    """从 raw dict 中挑出目标 dataclass 已声明的字段，忽略未知键。"""
    if not isinstance(raw, dict):
        return {}
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}


def _build_scale(raw: Any) -> "ScaleConfig":
    """从 ``scale:`` YAML 段构造 :class:`ScaleConfig`（含嵌套 train/detect）。

    未知键忽略；``symbols_file`` 转为 :class:`Path`；嵌套段独立解析后回填，
    避免把 dict 直接塞进 dataclass 字段造成类型不一致。
    """
    section: dict[str, Any] = raw if isinstance(raw, dict) else {}
    train = ScaleTrainConfig(**_pick(section.get("train", {}), ScaleTrainConfig))
    detect = ScaleDetectConfig(**_pick(section.get("detect", {}), ScaleDetectConfig))
    top = _pick(section, ScaleConfig)
    top.pop("train", None)
    top.pop("detect", None)
    sc = ScaleConfig(**top, train=train, detect=detect)
    sc.symbols_file = Path(sc.symbols_file)
    return sc
