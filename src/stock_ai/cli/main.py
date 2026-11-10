"""CLI 主入口：click Group，注册 data/sample/train/backtest/predict/doctor。

双入口：
    - ``python -m stock_ai <cmd>``
    - ``stockai <cmd>``（entry_point）
各子模块另有 ``python -m stock_ai.<module>`` 独立入口（见各 ``*_cli.py``）。
"""

from __future__ import annotations

import sys

import click

from stock_ai import __version__
from stock_ai.utils.errors import StockAIError
from stock_ai.utils.io import ensure_utf8_console

__all__ = ["main", "cli"]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="stock_ai")
def cli() -> None:
    """stock_ai —— A 股强化学习交易工程流水线。

    6 条命令互相独立，仅通过文件契约通信。
    """
    ensure_utf8_console()


# 延迟注册子命令（避免顶层 import 过重 + 循环依赖）
def _register_commands() -> None:
    """把各子命令挂载到主 group。"""
    from stock_ai.cli.backtest_cmd import backtest
    from stock_ai.cli.data_cmd import data
    from stock_ai.cli.doctor_cmd import doctor
    from stock_ai.cli.predict_cmd import predict
    from stock_ai.cli.sample_cmd import sample
    from stock_ai.cli.scale_cmd import scale
    from stock_ai.cli.train_cmd import train

    cli.add_command(data)
    cli.add_command(sample)
    cli.add_command(train)
    cli.add_command(backtest)
    cli.add_command(predict)
    cli.add_command(doctor)
    cli.add_command(scale)


_register_commands()


def main() -> None:
    """CLI 入口：捕获 :class:`StockAIError` 打印友好信息并 ``exit(1)``（K6）。"""
    ensure_utf8_console()
    try:
        cli(standalone_mode=False)
    except StockAIError as exc:
        click.echo(f"[错误] {exc}", err=True)
        sys.exit(1)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.exceptions.Abort:  # pragma: no cover
        click.echo("已取消", err=True)
        sys.exit(130)
    except SystemExit:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"[未预期错误] {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
