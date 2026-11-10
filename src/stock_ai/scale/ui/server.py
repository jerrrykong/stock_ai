"""ScaleUI 服务：``ThreadingHTTPServer`` 路由 + 子进程管理。

路由表（架构 §1.3）::

    GET  /                        -> 单页 index.html
    GET  /api/status?phase=..     -> 读 state/{phase}.json + split.json 摘要
    GET  /api/units?phase&status&q&page   -> 单元明细（服务端过滤/分页）
    GET  /api/data                -> 读 data/raw/meta.json + parquet 统计
    GET  /api/logs?phase&tail=200 -> 读 logs/scale/{phase}_*.log 尾部
    GET  /api/plot?unit_id&kind   -> 回读 curve.png / 资金曲线
    GET  /api/control/status?phase-> 进程态 + 是否在跑
    POST /api/control {phase,action} -> start|pause|resume|stop

进程模型（UI-8 关键）：
    - ``procs: dict[phase, Popen]``；``start`` 仅 ``Popen`` 一次，UI 不复活子进程；
    - ``pause/resume/stop`` 只写 ``control.json``（由编排器在单元边界消费）；
    - 不注册任何服务/计划任务/守护进程。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from stock_ai.scale.control import ControlStore
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.replay import ReplayError, ReplayService
from stock_ai.scale.state import StateStore
from stock_ai.utils.io import read_json

__all__ = ["ScaleUI", "ScaleRequestHandler"]

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_FILES = {
    "/static/replay.js": ("replay.js", "application/javascript; charset=utf-8"),
    "/static/vendor/echarts.min.js": ("vendor/echarts.min.js", "application/javascript; charset=utf-8"),
}

#: 允许的控制动作 → 是否写 control.json（start 直接起子进程，不写 control）。
_CONTROL_ACTIONS: frozenset[str] = frozenset({"start", "pause", "resume", "stop"})


class ScaleUI:
    """本机 UI 服务（单进程，仅监听 ``127.0.0.1``）。

    Args:
        cfg: 全局配置。
        paths: 路径集合。
        logger: 日志器（可空）。
        host: 监听地址（默认 ``127.0.0.1``）。
        port: 监听端口（默认 8765）。
    """

    def __init__(
        self,
        cfg: Any,
        paths: ScalePaths,
        logger: Any = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        """初始化。"""
        self.cfg = cfg
        self.paths: ScalePaths = paths
        self.logger = logger
        self.host: str = str(host)
        self.port: int = int(port)
        #: 阶段 → 子进程；**仅此一处**持有，绝不自启。
        self.procs: dict[str, subprocess.Popen] = {}
        self._lock: threading.Lock = threading.Lock()
        self.replay = ReplayService(cfg, paths)

    # ------------------------------------------------------------------ #
    # 子进程管理
    # ------------------------------------------------------------------ #
    def running(self, phase: str) -> bool:
        """该阶段子进程是否仍在运行。"""
        p = self.procs.get(phase)
        return p is not None and p.poll() is None

    def start(self, phase: str) -> dict:
        """启动一次 ``scale <phase>`` 子进程（幂等：已在跑则不重复启动）。

        Returns:
            ``{"ok": bool, "pid": int, "already": bool, "msg": str}``。
        """
        with self._lock:
            if self.running(phase):
                return {"ok": True, "already": True, "pid": self.procs[phase].pid,
                        "msg": "进程已在运行"}
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            cmd = [sys.executable, "-m", "stock_ai", "scale", phase]
            kwargs: dict[str, Any] = {
                "cwd": str(self.paths.root),
                "env": env,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name == "nt":  # pragma: no cover - Windows 专属，CI 可能非 Windows
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            try:
                proc = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "pid": 0, "already": False, "msg": f"启动失败: {exc}"}
            self.procs[phase] = proc
            if self.logger is not None:
                self.logger.info("已启动子进程: %s (pid=%d)", " ".join(cmd), proc.pid)
            return {"ok": True, "already": False, "pid": proc.pid, "msg": "已启动"}

    def unified_control(self, payload: dict) -> dict:
        from stock_ai.scale.unified import RUN_ID, write_control

        action = payload.get("action")
        if action not in _CONTROL_ACTIONS:
            return {"ok": False, "msg": "非法 unified 控制动作"}
        if action != "start":
            write_control(self.paths.root, action)
            return {"ok": True, "msg": f"已请求 {action}；在训练 step/准备或评估边界生效"}
        with self._lock:
            if self.running("unified"):
                return {"ok": True, "already": True, "msg": "unified 已在运行"}
            steps = payload.get("timesteps", self.cfg.train.timesteps)
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100000000:
                return {"ok": False, "msg": "timesteps 必须为 1~100000000 整数"}
            run_id = payload.get("run_id")
            if run_id is not None and (not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)):
                return {"ok": False, "msg": "非法 run_id"}
            if payload.get("resume") and not run_id:
                return {"ok": False, "msg": "续训必须选择 run_id"}
            execution = payload.get("execution", "legacy")
            if not isinstance(execution, str) or execution not in ("legacy", "strict", "strict-ca"):
                return {"ok": False, "msg": "execution 必须为 legacy / strict / strict-ca"}
            if execution == "strict-ca" and not payload.get("resume"):
                ca_dir = self.paths.root / "data" / "corporate_actions"
                if not ca_dir.is_dir() or not any(ca_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")):
                    return {"ok": False, "msg": "缺少已核验公司行为账本；拒绝启动，不能回退旧撮合"}
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["STOCKAI_UNIFIED_CONFIG"] = json.dumps(self.cfg.to_dict())
            cmd = [sys.executable, "-m", "stock_ai", "scale", "unified-train", "--timesteps", str(steps)]
            if not payload.get("resume") or "execution" in payload:
                cmd += ["--execution", execution]
            if execution in ("strict", "strict-ca"):
                cmd += ["--feature-dir", str(self.paths.root / "data" / "raw_hfq")]
            if execution == "strict-ca":
                cmd += ["--corporate-actions-dir", str(self.paths.root / "data" / "corporate_actions")]
            if run_id:
                cmd += ["--run-id", run_id]
            if payload.get("resume"):
                cmd += ["--resume"]
            log_path = self.paths.logs_dir / "unified.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("ab") as log:
                    self.procs["unified"] = subprocess.Popen(cmd, cwd=str(self.paths.root), env=env,
                                                              stdout=log, stderr=log)
            except OSError as exc:
                return {"ok": False, "msg": f"启动失败: {exc}"}
            return {"ok": True, "pid": self.procs["unified"].pid, "msg": "已启动 unified；日志 logs/scale/unified.log"}

    def control(self, phase: str, action: str) -> dict:
        """写控制信号（``pause|resume|stop``）。"""
        if action not in _CONTROL_ACTIONS:
            return {"ok": False, "msg": f"非法动作: {action}"}
        if action == "start":
            return self.start(phase)
        ControlStore(self.paths).write(phase, action)
        return {"ok": True, "msg": f"已写入 {action}"}

    # ------------------------------------------------------------------ #
    # 服务生命周期
    # ------------------------------------------------------------------ #
    def make_server(self) -> ThreadingHTTPServer:
        """构造并返回绑定到 ``(host, port)`` 的 HTTP 服务器。"""
        handler = type("_BoundHandler", (ScaleRequestHandler,), {"ui": self})
        server = ThreadingHTTPServer((self.host, self.port), handler)
        server.daemon_threads = True
        return server

    def serve(self, open_browser: bool = True) -> None:
        """阻塞式启动服务，直到 ``Ctrl+C``。"""
        server = self.make_server()
        url = f"http://{self.host}:{self.port}/"
        print(f"[scale ui] 服务已启动: {url}（仅本机访问，Ctrl+C 退出）")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - 无头环境忽略
                pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[scale ui] 收到中断，正在关闭 ...")
        finally:
            server.shutdown()
            server.server_close()


class ScaleRequestHandler(BaseHTTPRequestHandler):
    """HTTP 路由处理器（GET 读取 / POST 控制）。"""

    #: 由 :meth:`ScaleUI.make_server` 注入。
    ui: ScaleUI
    server_version = "StockAIScaleUI/1.0"

    # ------------------------------------------------------------------ #
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """把访问日志降级为 debug，避免污染控制台。"""
        ui = getattr(self, "ui", None)
        if ui is not None and getattr(ui, "logger", None) is not None:
            ui.logger.debug("UI %s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        """处理 GET 请求。"""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._serve_index()
            elif path in _STATIC_FILES:
                self._serve_static(path)
            elif path == "/api/replay/catalog":
                self._json(self.ui.replay.catalog())
            elif path == "/api/unified/status":
                from stock_ai.scale.unified import read_status
                state = read_status(self.ui.paths.root)
                active_lock = self.ui.paths.state_dir / "unified.lock"
                self._json({"ok": True, **state, "running": self.ui.running("unified") or active_lock.exists(),
                            "runs": self.ui.replay.unified_runs()})
            elif path == "/api/status":
                self._json(self._api_status(query))
            elif path == "/api/units":
                self._json(self._api_units(query))
            elif path == "/api/data":
                self._json(self._api_data())
            elif path == "/api/logs":
                self._json(self._api_logs(query))
            elif path == "/api/plot":
                self._serve_plot(query)
            elif path == "/api/control/status":
                self._json(self._api_control_status(query))
            else:
                self._json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001 - UI 不应因单请求异常崩溃
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        """处理 POST 请求（仅 ``/api/control``）。"""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length < 0 or length > 16384:
                raise ValueError("请求体大小必须为 0~16384 字节")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("请求体必须为 JSON 对象")
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": f"请求体解析失败: {exc}"}, status=400)
            return

        try:
            if path == "/api/replay":
                self._json(self.ui.replay.run(payload))
            elif path == "/api/unified/control":
                result = self.ui.unified_control(payload)
                self._json(result, status=200 if result.get("ok") else 400)
            elif path == "/api/control":
                phase = str(payload.get("phase", "train"))
                action = str(payload.get("action", ""))
                result = self.ui.control(phase, action)
                self._json(result, status=200 if result.get("ok") else 400)
            else:
                self._json({"ok": False, "error": "not found"}, status=404)
        except ReplayError as exc:
            self._json({"ok": False, "error": str(exc)}, status=exc.status)
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    # ------------------------------------------------------------------ #
    # 各 API 实现
    # ------------------------------------------------------------------ #
    def _api_status(self, query: dict) -> dict:
        """进度总览：计数 / ETA / 当前单元 + split 摘要。"""
        phase = _q1(query, "phase", "train")
        store = StateStore(self.ui.paths)
        state = store.load(phase)
        counts = state.recompute_counts()
        processed = counts.get("done", 0) + counts.get("failed", 0)
        remaining = max(0, int(state.total) - processed - counts.get("skipped", 0))
        return {
            "ok": True,
            "phase": phase,
            "process_state": state.process_state,
            "running": self.ui.running(phase),
            "total": int(state.total),
            "counts": counts,
            "remaining": remaining,
            "current_unit": state.current_unit,
            "elapsed_sec": float(state.elapsed_sec),
            "avg_unit_sec": float(state.avg_unit_sec),
            "eta_sec": float(state.eta_sec),
            "updated_at": state.updated_at,
            "split": self._split_summary(),
        }

    def _split_summary(self) -> dict:
        """``split.json`` 摘要（缺失返回空）。"""
        p = self.ui.paths.split_json
        if not p.exists():
            return {}
        try:
            d = read_json(p)
        except Exception:  # noqa: BLE001
            return {}
        return {
            "random_seed": d.get("random_seed"),
            "ratio": d.get("ratio"),
            "units_total": d.get("units_total"),
            "units_available": d.get("units_available"),
            "train": len(d.get("train_units", [])),
            "detect": len(d.get("detect_units", [])),
            "year_range": d.get("year_range"),
            "config_hash": d.get("config_hash"),
        }

    def _api_units(self, query: dict) -> dict:
        """单元明细（服务端过滤：阶段/状态/关键字 + 分页）。"""
        phase = _q1(query, "phase", "train")
        status = _q1(query, "status", "")
        q = _q1(query, "q", "").strip()
        page = _to_int(_q1(query, "page", "1"), 1)
        page_size = _to_int(_q1(query, "page_size", "100"), 100)
        page = max(1, page)
        page_size = max(1, min(1000, page_size))

        store = StateStore(self.ui.paths)
        state = store.load(phase)
        items: list[dict] = []
        for uid, us in state.units.items():
            if status and us.status != status:
                continue
            if q and q not in uid:
                continue
            rec = {"unit_id": uid}
            rec.update(us.to_dict())
            items.append(rec)
        items.sort(key=lambda r: r["unit_id"])

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "ok": True,
            "phase": phase,
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items[start:end],
        }

    def _api_data(self) -> dict:
        """数据观察：缓存概览（代码数 / 区间 / 行数 / 缺口）。"""
        raw_dir = Path(self.ui.cfg.paths.raw)
        meta: dict = {}
        meta_path = raw_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = read_json(meta_path)
            except Exception:  # noqa: BLE001
                meta = {}
        symbols_meta = (meta.get("symbols", {}) or {}) if isinstance(meta, dict) else {}

        parquet_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []
        rows: list[dict] = []
        d_min, d_max = "", ""
        for p in parquet_files:
            sym = p.stem
            entry = symbols_meta.get(sym, {}) or {}
            s = str(entry.get("start", "") or "")
            e = str(entry.get("end", "") or "")
            rows.append({
                "symbol": sym,
                "rows": int(entry.get("rows", 0) or 0),
                "start": s,
                "end": e,
                "missing_days": int(entry.get("missing_days", 0) or 0),
                "source": str(entry.get("source", "") or ""),
                "adjust": str(entry.get("adjust", "") or ""),
            })
            if s and (not d_min or s < d_min):
                d_min = s
            if e and (not d_max or e > d_max):
                d_max = e

        return {
            "ok": True,
            "cache_dir": str(raw_dir),
            "symbols_count": len(parquet_files),
            "date_range": {"start": d_min, "end": d_max},
            "items": rows,
        }

    def _api_logs(self, query: dict) -> dict:
        """结构化日志尾部（``logs/scale/<phase>_*.log``）。"""
        phase = _q1(query, "phase", "train")
        tail = max(1, min(2000, _to_int(_q1(query, "tail", "200"), 200)))
        logs_dir = self.ui.paths.logs_dir
        candidates = sorted(logs_dir.glob(f"{phase}_*.log")) if logs_dir.exists() else []
        if not candidates:
            return {"ok": True, "phase": phase, "file": "", "lines": []}
        latest = candidates[-1]
        try:
            text = latest.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "lines": []}
        lines = text.splitlines()[-tail:]
        return {"ok": True, "phase": phase, "file": latest.name, "lines": lines}

    def _api_control_status(self, query: dict) -> dict:
        """控制/进程态。"""
        phase = _q1(query, "phase", "train")
        cmd = ControlStore(self.ui.paths).poll(phase)
        return {"ok": True, "phase": phase, "running": self.ui.running(phase), "cmd": cmd}

    # ------------------------------------------------------------------ #
    # 文件/字节响应
    # ------------------------------------------------------------------ #
    def _serve_index(self) -> None:
        """返回单页 ``index.html``。"""
        index = _STATIC_DIR / "index.html"
        if not index.exists():
            self._json({"ok": False, "error": "index.html 缺失"}, status=500)
            return
        body = index.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        """只暴露精确白名单资源，不拼接任意用户路径。"""
        relative, content_type = _STATIC_FILES[path]
        target = (_STATIC_DIR / relative).resolve()
        if not _is_within(target, _STATIC_DIR.resolve()) or not target.is_file():
            self._json({"ok": False, "error": "静态资源不存在"}, status=404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_plot(self, query: dict) -> None:
        """回读曲线 PNG（curve / equity）。"""
        uid = _q1(query, "unit_id", "")
        kind = _q1(query, "kind", "curve")
        if not uid:
            self._json({"ok": False, "error": "缺少 unit_id"}, status=400)
            return
        if kind == "equity":
            target = self.ui.paths.report_png(uid)
        else:
            target = self.ui.paths.curve(uid)
        # 防目录穿越：必须位于受控目录内
        allowed_dirs = [self.ui.paths.models_dir.resolve(), self.ui.paths.reports_dir.resolve()]
        try:
            resolved = target.resolve()
        except Exception:  # noqa: BLE001
            resolved = target
        if not any(_is_within(resolved, d) for d in allowed_dirs) or not target.exists():
            self._json({"ok": False, "error": "图片不存在"}, status=404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        """返回 JSON 响应。"""
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _q1(query: dict, key: str, default: str = "") -> str:
    """取查询参数首值。"""
    vals = query.get(key)
    if not vals:
        return default
    return str(vals[0])


def _to_int(value: Any, default: int) -> int:
    """安全转 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_within(path: Path, directory: Path) -> bool:
    """``path`` 是否位于 ``directory`` 之内（防穿越）。"""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False
