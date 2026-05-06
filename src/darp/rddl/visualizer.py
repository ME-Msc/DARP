"""Standalone HTML visualizer for the built-in RDDL AST."""

# TODO(phase-4.1): Show exact token spans for typed expression nodes when the
# compiler moves to a factored RDDL AST.

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import re
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from darp.online import FiniteHorizonOnlinePlanner, initial_belief_from_observation, update_belief
from darp.core.problem import PlanningProblem
from darp.rddl.ast import RDDLASTNode
from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.compiler import RDDLCompiler
from darp.rddl.lexicon import RDDL_KEYWORDS
from darp.rddl.loader import RDDLLoader, available_frontends
from darp.sim.local import LocalSimulator

MIN_NODE_WIDTH = 154
MIN_NODE_HEIGHT = 60
CHAR_WIDTH = 7.4
PAD_X = 14
BADGE_BASELINE_Y = 19
LABEL_BASELINE_Y = 39
LINE_HEIGHT = 15
BOTTOM_PAD = 14
H_GAP = 42
V_GAP = 92
MARGIN_X = 36
MARGIN_Y = 32
MAX_COLLAPSED_CHILD_LINES = 5
MAX_COLLAPSED_LINE_LENGTH = 70

TOKEN_PATTERN = re.compile(
    r"\s+|==|!=|<=|>=|\d+(?:\.\d+)?|@[A-Za-z0-9_-]+|\?[A-Za-z0-9_-]+|"
    r"[A-Za-z_][A-Za-z0-9_'\-]*|[{}()[\]:=,;./^|&!<>+]|."
)


@dataclass(frozen=True)
class _TokenSpan:
    """Store one highlighted text span. / 保存一个高亮文本片段。"""

    text: str
    css_class: str


@dataclass(frozen=True)
class _VisualLine:
    """Store highlighted spans for one display line. / 保存一行显示文本的高亮片段。"""

    tokens: tuple[_TokenSpan, ...]

    @property
    def text(self) -> str:
        """Return this visual line as plain text. / 将当前可视行返回为纯文本。"""
        return "".join(token.text for token in self.tokens)


@dataclass(frozen=True)
class _VisualNode:
    """Store display metadata for one AST node. / 保存单个 AST 节点的显示元数据。"""

    node: RDDLASTNode
    node_id: str
    lines: tuple[_VisualLine, ...]
    collapsed_lines: tuple[_VisualLine, ...]
    width: float
    height: float
    collapsed_width: float
    collapsed_height: float


@dataclass(frozen=True)
class _RuntimeMarkers:
    """Store known semantic markers from non-fluent atoms. / 保存从 non-fluent atom 中识别出的语义标记。"""

    starts: tuple[str, ...]
    risks: tuple[str, ...]
    goals: tuple[str, ...]


def render_html(
    ast: RDDLASTNode,
    title: str = "RDDL AST",
    runtime: dict[str, object] | None = None,
) -> str:
    """Render an AST as a self-contained interactive HTML page. / 将 AST 渲染为独立交互式 HTML 页面。"""

    data_json = _json_for_html(_build_visual_data(ast, runtime=runtime))
    escaped_title = html.escape(title)
    summary = html.escape(ast.summary())
    runtime_panel = _runtime_panel_html(runtime)
    workspace_class = "workspace has-runtime" if runtime_panel else "workspace"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>{_styles()}</style>
</head>
<body>
  <header>
    <div class="header-main">
      <div>
        <h1>{escaped_title}</h1>
        <div class="summary">{summary}</div>
      </div>
      <div class="panel-toggles" role="toolbar" aria-label="Panel visibility">
        <button id="toggle-source-panel" type="button" aria-pressed="true">Source</button>
        <button id="toggle-ast-panel" type="button" aria-pressed="true">AST</button>
        <button id="toggle-runtime-panel" type="button" aria-pressed="true">Runtime</button>
      </div>
    </div>
  </header>
  <main class="{workspace_class}">
    <aside class="text-panel" aria-label="RDDL source text">
      <div class="panel-heading">RDDL Source</div>
      <div id="text-outline" class="text-outline"></div>
    </aside>
    <div id="split-resizer" class="split-resizer" role="separator" aria-label="Resize source and AST panes"></div>
    <section class="graph-panel" aria-label="AST graph">
      <div class="panel-heading">AST</div>
      <div class="ast-tools">
        <div class="toolbar" role="toolbar" aria-label="Source and AST controls">
          <div class="toolbar-group">
            <span class="group-label">Tree</span>
            <button id="expand-all" type="button">Expand all</button>
            <button id="collapse-all" type="button">Collapse all</button>
          </div>
          <div class="toolbar-group">
            <span class="group-label">Depth</span>
            <input id="depth-input" class="compact-input" type="number" min="0" step="1" value="2" aria-label="Expansion depth">
            <button id="expand-depth" type="button">Open depth</button>
          </div>
          <div class="toolbar-group toolbar-search">
            <span class="group-label">Search</span>
            <input id="search-input" type="search" placeholder="Kind or label" aria-label="Search AST nodes">
            <label class="search-toggle" title="Match case">
              <input id="match-case" type="checkbox">
              <span>Aa</span>
            </label>
            <label class="search-toggle" title="Match whole word">
              <input id="whole-word" type="checkbox">
              <span>Word</span>
            </label>
            <label class="search-toggle" title="Use regular expression">
              <input id="use-regex" type="checkbox">
              <span>Regex</span>
            </label>
            <button id="previous-match" type="button">Previous</button>
            <button id="next-match" type="button">Next</button>
            <button id="reveal-matches" type="button">Reveal paths</button>
            <button id="clear-search" type="button">Clear</button>
          </div>
          <div class="toolbar-group">
            <span class="group-label">Zoom</span>
            <button id="zoom-out" type="button">-</button>
            <button id="zoom-reset" type="button">100%</button>
            <button id="zoom-in" type="button">+</button>
          </div>
        </div>
        <div class="status-row">
          <span id="visible-status"></span>
          <span id="match-status"></span>
          <span id="selected-status"></span>
        </div>
      </div>
      <div class="canvas" id="canvas">
        <svg id="ast-svg" role="img" aria-label="{escaped_title}"></svg>
      </div>
    </section>
    {runtime_panel}
  </main>
  <script id="ast-data" type="application/json">{data_json}</script>
  <script>{_script()}</script>
</body>
</html>
"""


def _runtime_panel_html(runtime: dict[str, object] | None) -> str:
    """Return the runtime panel markup only for DARP internal simulation. / 仅为 DARP 内部仿真返回运行面板标记。"""
    if not runtime or runtime.get("mode") != "internal":
        return ""
    return """
    <div id="runtime-resizer" class="split-resizer runtime-resizer" role="separator" aria-label="Resize AST and runtime panes"></div>
    <aside class="runtime-panel" aria-label="Execution state machine">
      <div class="panel-heading">Execution State Machine</div>
      <div id="runtime-root" class="runtime-root"></div>
    </aside>
    """


def _build_visual_data(
    ast: RDDLASTNode, runtime: dict[str, object] | None = None
) -> dict[str, object]:
    """Convert AST nodes into JSON-ready visual metadata. / 将 AST 节点转换为可写入 JSON 的可视化元数据。"""
    nodes: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    max_depth = 0

    def visit(
        node: RDDLASTNode, depth: int, parent_id: str | None, source_file_id: str | None
    ) -> str:
        """Append one visual node and visit its children. / 添加一个可视节点并递归访问子节点。"""
        nonlocal max_depth
        node_id = f"node{len(nodes)}"
        current_source_file_id = node_id if node.kind == "file" else source_file_id
        visual = _make_visual_node(node, node_id)
        payload: dict[str, object] = {
            "id": node_id,
            "kind": node.kind,
            "label": node.label,
            "cssClass": f"node-{_safe_class(node.kind)}",
            "searchText": _search_text(node, visual, depth),
            "width": round(visual.width, 1),
            "height": round(visual.height, 1),
            "depth": depth,
            "parent": parent_id,
            "sourceFile": current_source_file_id,
            "line": node.line,
            "endLine": node.end_line if node.end_line is not None else node.line,
            "children": [],
            "lines": [
                [{"text": token.text, "className": token.css_class} for token in line.tokens]
                for line in visual.lines
            ],
            "collapsedLines": [
                [{"text": token.text, "className": token.css_class} for token in line.tokens]
                for line in visual.collapsed_lines
            ],
            "collapsedWidth": round(visual.collapsed_width, 1),
            "collapsedHeight": round(visual.collapsed_height, 1),
        }
        nodes.append(payload)
        if node.kind == "file":
            files.append(_file_payload(node_id, node.label))
        max_depth = max(max_depth, depth)
        children = payload["children"]
        assert isinstance(children, list)
        for child in node.children:
            children.append(visit(child, depth + 1, node_id, current_source_file_id))
        return node_id

    root_id = visit(ast, 0, None, None)
    return {
        "rootId": root_id,
        "nodes": nodes,
        "files": files,
        "totalNodes": len(nodes),
        "maxDepth": max_depth,
        "layout": {
            "hGap": H_GAP,
            "vGap": V_GAP,
            "marginX": MARGIN_X,
            "marginY": MARGIN_Y,
            "padX": PAD_X,
            "badgeBaselineY": BADGE_BASELINE_Y,
            "labelBaselineY": LABEL_BASELINE_Y,
            "lineHeight": LINE_HEIGHT,
        },
        "runtime": runtime or {"enabled": False},
    }


def _file_payload(file_id: str, path: str) -> dict[str, object]:
    """Build source text payload for one file node. / 为单个 file 节点构建源码文本载荷。"""
    file_path = Path(path)
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        source = ""
    return {
        "id": file_id,
        "name": file_path.name or path,
        "path": path,
        "lines": [
            [{"text": token.text, "className": token.css_class} for token in _tokenize(line)]
            for line in source.splitlines()
        ],
    }


def _search_text(node: RDDLASTNode, visual: _VisualNode, depth: int) -> str:
    """Build searchable text for one AST node. / 为单个 AST 节点构建可搜索文本。"""
    rendered = " ".join(line.text for line in (*visual.lines, *visual.collapsed_lines))
    return f"{node.kind} {node.label} {rendered} depth:{depth}"


def _json_for_html(payload: dict[str, object]) -> str:
    """Encode JSON safely for inline HTML. / 将 JSON 安全编码到内联 HTML 中。"""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


class _RuntimeController:
    """Own the local simulator state behind the browser API. / 持有浏览器 API 背后的本地 simulator 状态。"""

    def __init__(self, problem: PlanningProblem, markers: _RuntimeMarkers, seed: int = 0) -> None:
        """Create a controller and reset the simulator once. / 创建控制器并先重置一次 simulator。"""
        self.problem = problem
        self.markers = markers
        self.simulator = LocalSimulator(problem, seed=seed)
        self.planner = FiniteHorizonOnlinePlanner(problem)
        self.lock = threading.Lock()
        self.trace: list[dict[str, object]] = []
        self.last_action: str | None = None
        self.previous_state: object | None = None
        self.observation: object | None = None
        self.belief = dict(problem.initial_belief)
        self.reward = 0.0
        self.done = False
        self.reset()

    def reset(self) -> dict[str, object]:
        """Reset the simulation and return a browser snapshot. / 重置仿真并返回浏览器快照。"""
        with self.lock:
            self.observation = self.simulator.reset()
            self.belief = initial_belief_from_observation(self.problem, self.observation)
            self.reward = 0.0
            self.done = False
            self.last_action = None
            self.previous_state = None
            self.trace = [
                {
                    "step": 0,
                    "action": "reset",
                    "state": self.simulator.state,
                    "observation": self.observation,
                    "belief": self.belief,
                    "reward": 0.0,
                    "done": False,
                }
            ]
            snapshot = self._snapshot_locked()
            print(
                "Runtime reset: "
                f"state={snapshot['state']}, observation={snapshot['observation']}, "
                f"planned_action={snapshot['planned_action']}",
                flush=True,
            )
            return snapshot

    def step(self) -> dict[str, object]:
        """Select a DARP action, apply it, and return a snapshot. / 选择 DARP 动作、执行并返回快照。"""
        with self.lock:
            if self.done:
                return self._snapshot_locked()
            decision = self._plan_locked()
            action = decision.action
            self.previous_state = self.simulator.state
            previous_belief = dict(self.belief)
            self.observation, self.reward, self.done, info = self.simulator.step(action)
            self.belief = update_belief(self.problem, self.belief, action, self.observation)
            self.last_action = action
            self.trace.append(
                {
                    "step": self.simulator.steps,
                    "action": action,
                    "previous_state": self.previous_state,
                    "state": info["state"],
                    "observation": self.observation,
                    "belief": previous_belief,
                    "next_belief": self.belief,
                    "decision": decision.to_dict(),
                    "reward": self.reward,
                    "done": self.done,
                }
            )
            snapshot = self._snapshot_locked()
            print(
                "Runtime step: "
                f"action={action}, from={self.previous_state}, to={snapshot['state']}, "
                f"observation={snapshot['observation']}, reward={snapshot['reward']}, "
                f"done={snapshot['done']}",
                flush=True,
            )
            return snapshot

    def snapshot(self) -> dict[str, object]:
        """Return the current browser snapshot. / 返回当前浏览器快照。"""
        with self.lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        """Build a snapshot while the controller lock is held. / 在持锁状态下构建快照。"""
        state = self.simulator.state
        decision = None if self.done else self._plan_locked()
        return {
            "state": state,
            "observation": self.observation,
            "belief": self.belief,
            "reward": self.reward,
            "step": self.simulator.steps,
            "done": self.done,
            "last_action": self.last_action,
            "previous_state": self.previous_state,
            "initial_state": self.trace[0]["state"] if self.trace else None,
            "initial_observation": self.trace[0]["observation"] if self.trace else None,
            "planner": self.planner.name,
            "planned_action": decision.action if decision is not None else None,
            "planned_decision": decision.to_dict() if decision is not None else None,
            "trace": self.trace,
        }

    def _plan_locked(self):
        """Plan from the current online belief while the lock is held. / 在持锁状态下根据当前在线 belief 规划。"""
        remaining_depth = max(1, self.problem.max_depth - self.simulator.steps)
        return self.planner.choose_action(self.belief, remaining_depth=remaining_depth)


def _runtime_payload(
    problem: PlanningProblem,
    markers: _RuntimeMarkers,
    snapshot: dict[str, object],
    *,
    endpoint: str,
) -> dict[str, object]:
    """Build JSON data for the runtime panel. / 构建运行面板所需的 JSON 数据。"""
    return {
        "enabled": True,
        "mode": "internal",
        "simulator": "darp",
        "endpoint": endpoint,
        "snapshot": snapshot,
        "problem": {
            "name": problem.name,
            "horizon": problem.max_depth,
            "states": [str(state) for state in problem.states],
            "actions": list(problem.actions),
            "markers": {
                "starts": list(markers.starts),
                "risks": list(markers.risks),
                "goals": list(markers.goals),
            },
            "layout": _runtime_layout(problem),
            "transitions": [
                {
                    "from": str(source),
                    "action": action,
                    "to": str(target),
                    "prob": probability,
                    "reward": problem.reward(source, action),
                }
                for (source, action, target), probability in problem.transitions.items()
                if probability > 1e-12
            ],
        },
    }


def _external_simulator_payload(simulator: str, loaded: object) -> dict[str, object]:
    """Build runtime metadata for non-DARP simulators without drawing a state machine. / 为非 DARP simulator 构建不绘制状态机的运行元数据。"""
    metadata = getattr(loaded, "metadata", {})
    return {
        "enabled": True,
        "mode": "external",
        "simulator": simulator,
        "metadata": metadata,
    }


def _runtime_layout(problem: PlanningProblem) -> dict[str, object]:
    """Infer a small display layout for runtime states. / 推断运行状态的小型显示布局。"""
    grid_cells: dict[str, dict[str, int]] = {}
    for state in problem.states:
        match = re.fullmatch(r"c(\d+)(\d+)", str(state))
        if match is None:
            return {"kind": "state-machine"}
        row = int(match.group(1))
        col = int(match.group(2))
        grid_cells[str(state)] = {"row": row, "col": col}
    return {
        "kind": "grid",
        "rows": max(cell["row"] for cell in grid_cells.values()),
        "cols": max(cell["col"] for cell in grid_cells.values()),
        "cells": grid_cells,
    }


def _runtime_markers(ast: RDDLASTNode) -> _RuntimeMarkers:
    """Extract start/risk/goal markers from known non-fluent atoms. / 从已知 non-fluent atom 中提取 start/risk/goal 标记。"""
    starts: set[str] = set()
    risks: set[str] = set()
    goals: set[str] = set()
    for node in _walk_ast(ast):
        if node.kind != "statement":
            continue
        match = re.fullmatch(r"(is-start|is-risk|is-goal)\(@?([A-Za-z0-9_-]+)\)", node.label)
        if match is None:
            continue
        marker, state = match.groups()
        if marker == "is-start":
            starts.add(state)
        elif marker == "is-risk":
            risks.add(state)
        elif marker == "is-goal":
            goals.add(state)
    return _RuntimeMarkers(tuple(sorted(starts)), tuple(sorted(risks)), tuple(sorted(goals)))


def _walk_ast(node: RDDLASTNode) -> list[RDDLASTNode]:
    """Return a flat preorder list of AST nodes. / 返回 AST 节点的前序扁平列表。"""
    result = [node]
    for child in node.children:
        result.extend(_walk_ast(child))
    return result


def _serve_runtime_visualizer(
    *,
    ast: RDDLASTNode,
    problem: PlanningProblem,
    markers: _RuntimeMarkers,
    host: str,
    port: int,
    seed: int = 0,
    open_browser: bool = True,
) -> int:
    """Serve the interactive runtime visualizer over local HTTP. / 通过本地 HTTP 提供交互式运行可视化。"""
    controller = _RuntimeController(problem, markers, seed=seed)

    class Handler(BaseHTTPRequestHandler):
        """Handle one visualizer HTTP request. / 处理一个 visualizer HTTP 请求。"""

        def log_message(self, format: str, *args: Any) -> None:
            """Silence default HTTP logs. / 静默默认 HTTP 日志。"""
            return

        def do_GET(self) -> None:
            """Serve the HTML page or current runtime state. / 提供 HTML 页面或当前运行状态。"""
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                self._send_json(controller.snapshot())
                return
            if parsed.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            payload = _runtime_payload(problem, markers, controller.snapshot(), endpoint="/api")
            page = render_html(ast, title="RDDL AST + Runtime", runtime=payload)
            self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            """Apply runtime commands from the browser. / 执行浏览器发来的运行命令。"""
            parsed = urlparse(self.path)
            if parsed.path == "/api/reset":
                self._send_json(controller.reset())
                return
            if parsed.path == "/api/step":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self.rfile.read(content_length)
                self._send_json(controller.step())
                return
            self.send_error(404)

        def _send_json(self, payload: dict[str, object]) -> None:
            """Send a JSON response. / 发送 JSON 响应。"""
            self._send_bytes(json.dumps(payload, default=str).encode("utf-8"), "application/json")

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            """Send bytes with a content type. / 按指定 content type 发送字节。"""
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = server.server_address
    display_host = "127.0.0.1" if actual_host in {"", "0.0.0.0"} else actual_host
    url = f"http://{display_host}:{actual_port}/"
    thread = threading.Thread(target=server.serve_forever, name="darp-visualizer-http", daemon=True)
    thread.start()
    print(f"Interactive RDDL visualizer running at {url}", flush=True)
    print("Press Ctrl+C to stop the visualizer server.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        while thread.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
    return 0


def _serve_html_visualizer(
    *,
    ast: RDDLASTNode,
    title: str,
    runtime: dict[str, object] | None,
    host: str,
    port: int,
    open_browser: bool = True,
) -> int:
    """Serve a live HTML visualizer without writing a static file. / 提供不写静态文件的实时 HTML 可视化服务。"""

    class Handler(BaseHTTPRequestHandler):
        """Handle one non-interactive visualizer HTTP request. / 处理一个非交互式 visualizer HTTP 请求。"""

        def log_message(self, format: str, *args: Any) -> None:
            """Silence default HTTP logs. / 静默默认 HTTP 日志。"""
            return

        def do_GET(self) -> None:
            """Serve the current HTML page. / 提供当前 HTML 页面。"""
            parsed = urlparse(self.path)
            if parsed.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            page = render_html(ast, title=title, runtime=runtime)
            payload = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = server.server_address
    display_host = "127.0.0.1" if actual_host in {"", "0.0.0.0"} else actual_host
    url = f"http://{display_host}:{actual_port}/"
    thread = threading.Thread(target=server.serve_forever, name="darp-visualizer-http", daemon=True)
    thread.start()
    print(f"Interactive RDDL visualizer running at {url}", flush=True)
    print("Press Ctrl+C to stop the visualizer server.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        while thread.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
    return 0


def _styles() -> str:
    """Return CSS for the interactive visualizer. / 返回交互式可视化页面的 CSS。"""
    return """
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #172534;
      --muted: #617488;
      --line: #8ca4ba;
      --node-border: #4e6d88;
      --keyword: #7c3aed;
      --symbol: #047857;
      --number: #b45309;
      --literal: #2563eb;
      --parameter: #be185d;
      --operator: #475569;
      --match: #f59e0b;
      --active-match: #ef4444;
      --selected: #0284c7;
      --rddl: #e8f0ff;
      --file: #eef4f8;
      --domain: #e9f8ef;
      --instance: #fff3dd;
      --block: #edf3ff;
      --assignment: #f9edff;
      --statement: #fbfcfd;
    }
    * {
      box-sizing: border-box;
    }
    [hidden] {
      display: none !important;
    }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 14px 28px 12px;
      border-bottom: 1px solid #d7e1ea;
      background: rgba(255, 255, 255, 0.96);
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(8px);
    }
    .header-main {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 {
      font-size: 22px;
      margin: 0 0 6px;
      letter-spacing: 0;
    }
    .summary {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 0;
    }
    .panel-toggles {
      display: inline-flex;
      flex: 0 0 auto;
      align-items: center;
      gap: 6px;
      padding: 5px;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #f8fbfd;
    }
    .panel-toggles button {
      min-width: 72px;
      height: 30px;
      padding: 0 9px;
      font-size: 12px;
      font-weight: 700;
    }
    .panel-toggles button[aria-pressed="true"] {
      border-color: #0284c7;
      background: #e8f4ff;
      color: #075985;
    }
    .panel-toggles button[disabled] {
      cursor: not-allowed;
      opacity: 0.45;
    }
    .ast-tools {
      flex: 0 0 auto;
      padding: 10px 10px 8px;
      border-bottom: 1px solid #d7e1ea;
      background: #ffffff;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .toolbar-group {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 44px;
      padding: 6px;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #f8fbfd;
    }
    .toolbar-search {
      flex: 0 1 auto;
      min-width: 0;
    }
    .group-label {
      align-self: stretch;
      display: inline-flex;
      align-items: center;
      padding: 0 8px 0 4px;
      border-right: 1px solid #d7e1ea;
      color: #52677c;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .toolbar input {
      height: 32px;
      border: 1px solid #bfd0de;
      border-radius: 6px;
      padding: 0 9px;
      color: var(--ink);
      background: #ffffff;
    }
    .compact-input {
      width: 72px;
    }
    #search-input {
      flex: 0 0 210px;
      width: 210px;
      min-width: 160px;
      max-width: 28vw;
    }
    #search-input.search-error {
      border-color: #ef4444;
      box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.16);
    }
    .search-toggle {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      height: 32px;
      padding: 0 8px;
      border: 1px solid #bfd0de;
      border-radius: 6px;
      background: #ffffff;
      color: #24384c;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }
    .search-toggle input {
      width: 13px;
      height: 13px;
      margin: 0;
      accent-color: #0284c7;
    }
    .search-toggle:has(input:checked) {
      border-color: #0284c7;
      background: #e8f4ff;
      color: #075985;
      font-weight: 700;
    }
    button {
      height: 32px;
      border: 1px solid #bfd0de;
      border-radius: 6px;
      padding: 0 10px;
      background: #ffffff;
      color: #24384c;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }
    button:hover {
      border-color: #7fa0ba;
      background: #f8fbfd;
    }
    .status-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding-top: 2px;
    }
    .status-row span {
      min-height: 24px;
      padding: 4px 8px;
      border: 1px solid #d7e1ea;
      border-radius: 6px;
      background: #ffffff;
      color: #475569;
      font-size: 12px;
    }
    main.workspace {
      display: grid;
      grid-template-columns:
        minmax(280px, var(--source-width, 28vw))
        10px
        minmax(360px, 1fr);
      gap: 8px;
      padding: 18px;
      align-items: stretch;
    }
    main.workspace.has-runtime {
      grid-template-columns:
        minmax(280px, var(--source-width, 28vw))
        10px
        minmax(360px, 1fr)
        10px
        minmax(300px, var(--runtime-width, 28vw));
    }
    .split-resizer {
      position: relative;
      min-height: calc(100vh - 132px);
      max-height: calc(100vh - 132px);
      border-radius: 8px;
      cursor: col-resize;
    }
    .split-resizer::before {
      content: "";
      position: absolute;
      top: 8px;
      bottom: 8px;
      left: 4px;
      width: 2px;
      border-radius: 2px;
      background: #bfd0de;
    }
    .split-resizer:hover::before,
    .split-resizer.resizer-active::before {
      background: #0284c7;
      box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.14);
    }
    .text-panel,
    .graph-panel,
    .runtime-panel {
      min-height: calc(100vh - 132px);
      max-height: calc(100vh - 132px);
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 14px 28px rgba(42, 62, 82, 0.10);
      overflow: hidden;
    }
    .text-panel {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .graph-panel,
    .runtime-panel {
      display: flex;
      flex-direction: column;
    }
    .panel-heading {
      flex: 0 0 auto;
      padding: 11px 14px;
      border-bottom: 1px solid #d7e1ea;
      background: #f8fbfd;
      color: #274158;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .text-outline {
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      min-height: 0;
    }
    .source-block {
      display: flex;
      flex: 1 1 0;
      flex-direction: column;
      min-height: 118px;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }
    .source-header {
      display: grid;
      grid-template-columns: 1fr;
      gap: 2px;
      padding: 9px 11px;
      border-bottom: 1px solid #d7e1ea;
      background: #f8fbfd;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .source-name {
      color: #172534;
      font-size: 12px;
      font-weight: 800;
    }
    .source-path {
      color: #617488;
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .source-body {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding: 6px 0;
    }
    .source-resizer {
      position: relative;
      flex: 0 0 12px;
      cursor: row-resize;
    }
    .source-resizer::before {
      content: "";
      position: absolute;
      top: 5px;
      left: 12px;
      right: 12px;
      height: 2px;
      border-radius: 2px;
      background: #bfd0de;
    }
    .source-resizer:hover::before,
    .source-resizer.resizer-active::before {
      background: #0284c7;
      box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.14);
    }
    .source-line {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      align-items: start;
      width: 100%;
      height: auto;
      min-height: 24px;
      margin: 0;
      padding: 0;
      border: 1px solid transparent;
      border-radius: 0;
      background: transparent;
      color: #172534;
      text-align: left;
      font: inherit;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }
    .source-line:hover {
      border-color: #bfd0de;
      background: #f8fbfd;
    }
    .source-line.source-match {
      border-color: rgba(245, 158, 11, 0.72);
      background: #fff7ed;
    }
    .source-line.source-active-match {
      border-color: rgba(239, 68, 68, 0.86);
      background: #fef2f2;
      box-shadow: inset 4px 0 0 rgba(239, 68, 68, 0.86);
    }
    .source-line.source-selected {
      border-color: rgba(2, 132, 199, 0.9);
      background: #e8f4ff;
      box-shadow: inset 4px 0 0 rgba(2, 132, 199, 0.9);
      opacity: 1;
    }
    .source-gutter {
      position: relative;
      min-height: 24px;
      padding: 3px 9px 3px 0;
      border-right: 1px solid #d7e1ea;
      color: #7b8da0;
      text-align: right;
      user-select: none;
    }
    .source-gutter::after {
      content: "";
      position: absolute;
      top: 0;
      bottom: -1px;
      right: -1px;
      width: 1px;
      background: #d7e1ea;
    }
    .source-code {
      min-height: 24px;
      padding: 3px 10px;
      white-space: pre;
      overflow-x: auto;
    }
    .graph-panel {
      overflow: hidden;
      min-width: 0;
    }
    .canvas {
      flex: 1 1 auto;
      width: 100%;
      min-width: 0;
      padding: 12px;
      overflow: auto;
      cursor: grab;
      touch-action: none;
    }
    .canvas svg {
      cursor: inherit;
      touch-action: none;
    }
    .canvas.ast-panning {
      cursor: grabbing;
      user-select: none;
    }
    .runtime-panel {
      min-width: 0;
      overflow: hidden;
    }
    .runtime-root {
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
      padding: 12px;
      gap: 10px;
    }
    .runtime-message {
      padding: 12px;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #f8fbfd;
      color: #52677c;
      font-size: 13px;
      line-height: 1.45;
    }
    .runtime-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) repeat(6, auto);
      gap: 8px;
      align-items: center;
    }
    .runtime-planned-action {
      min-width: 0;
      height: 32px;
      display: flex;
      align-items: center;
      gap: 6px;
      border: 1px solid #bfd0de;
      border-radius: 6px;
      padding: 0 9px;
      background: #ffffff;
      color: #172534;
      font: inherit;
      font-size: 13px;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
    .runtime-planned-action strong {
      color: #52677c;
      font-size: 11px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .runtime-status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .runtime-info {
      display: grid;
      gap: 4px;
      padding: 9px 10px;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #ffffff;
      color: #274158;
      font-size: 12px;
      line-height: 1.35;
    }
    .runtime-chip {
      padding: 8px 10px;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #f8fbfd;
      color: #274158;
      font-size: 12px;
      min-width: 0;
    }
    .runtime-chip strong {
      display: block;
      margin-bottom: 2px;
      color: #617488;
      font-size: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .runtime-stage {
      flex: 1 1 auto;
      min-height: 320px;
      position: relative;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #ffffff;
      overflow: auto;
    }
    .runtime-svg {
      display: block;
      min-width: 0;
      min-height: 0;
      background: #ffffff;
      touch-action: none;
    }
    .runtime-node-group {
      cursor: grab;
      touch-action: none;
    }
    .runtime-edge-drag-area {
      fill: none;
      stroke: transparent;
      stroke-width: 18;
      cursor: grab;
      pointer-events: stroke;
      touch-action: none;
    }
    .runtime-edge-label-group {
      cursor: grab;
      touch-action: none;
    }
    .runtime-node-group:active,
    .runtime-edge-label-group:active,
    .runtime-edge-drag-area:active {
      cursor: grabbing;
    }
    .runtime-cell {
      fill: #f8fbfd;
      stroke: #bfd0de;
      stroke-width: 1.2;
    }
    .runtime-cell.current {
      fill: #e8f4ff;
      stroke: #0284c7;
      stroke-width: 2.4;
    }
    .runtime-cell.start {
      fill: #e9f8ef;
    }
    .runtime-cell.risk {
      fill: #fef2f2;
      stroke: #ef4444;
    }
    .runtime-cell.goal {
      fill: #fff7ed;
      stroke: #f59e0b;
    }
    .runtime-cell-label {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 8px;
      font-weight: 700;
      fill: #172534;
      pointer-events: none;
    }
    .runtime-marker-label {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      font-size: 6px;
      font-weight: 800;
      fill: #475569;
      pointer-events: none;
      text-transform: uppercase;
    }
    .runtime-token {
      fill: #0284c7;
      stroke: #ffffff;
      stroke-width: 1.5;
      filter: drop-shadow(0 4px 8px rgba(2, 132, 199, 0.30));
    }
    .runtime-edge {
      fill: none;
      stroke: #8ca4ba;
      stroke-width: 1.3;
      opacity: 0.62;
    }
    .runtime-edge.active {
      stroke: #0284c7;
      stroke-width: 2.6;
      opacity: 1;
    }
    .runtime-edge-label-link {
      fill: none;
      stroke: #b6c7d6;
      stroke-width: 1;
      stroke-dasharray: 4 4;
      opacity: 0.8;
      pointer-events: none;
    }
    .runtime-edge-label-link.active {
      stroke: #0284c7;
      stroke-width: 1.5;
      opacity: 1;
    }
    .runtime-edge-label-bg {
      fill: #ffffff;
      stroke: #d7e1ea;
      stroke-width: 1;
      opacity: 0.94;
    }
    .runtime-edge-label {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 5.2px;
      font-weight: 700;
      fill: #274158;
      pointer-events: none;
    }
    .runtime-node {
      fill: #f8fbfd;
      stroke: #8ca4ba;
      stroke-width: 1.5;
    }
    .runtime-node.current {
      fill: #e8f4ff;
      stroke: #0284c7;
      stroke-width: 3;
    }
    .runtime-trace {
      max-height: 120px;
      overflow: auto;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      background: #f8fbfd;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.45;
      color: #274158;
    }
    svg {
      display: block;
      background: #ffffff;
    }
    .edge {
      fill: none;
      stroke: var(--line);
      stroke-width: 2;
      stroke-linecap: round;
    }
    .node {
      cursor: pointer;
    }
    .node-collapsed .node-card {
      stroke-dasharray: 5 4;
    }
    .node-card {
      fill: var(--panel);
      stroke: var(--node-border);
      stroke-width: 1.2;
      filter: drop-shadow(0 5px 8px rgba(35, 54, 74, 0.16));
      transition: stroke 120ms ease, stroke-width 120ms ease, filter 120ms ease;
    }
    .node-halo {
      fill: transparent;
      pointer-events: none;
    }
    .node-match .node-halo {
      stroke: rgba(245, 158, 11, 0.48);
      stroke-width: 8;
      filter: drop-shadow(0 0 10px rgba(245, 158, 11, 0.45));
    }
    .node-selected .node-halo {
      stroke: rgba(2, 132, 199, 0.38);
      stroke-width: 8;
      filter: drop-shadow(0 0 10px rgba(2, 132, 199, 0.42));
    }
    .node-focus-pulse .node-halo {
      stroke: rgba(14, 165, 233, 0.72);
      stroke-width: 11;
      filter: drop-shadow(0 0 16px rgba(14, 165, 233, 0.62));
      animation: focusPulse 1.2s ease-out;
    }
    .node-active-match .node-halo {
      stroke: rgba(239, 68, 68, 0.58);
      stroke-width: 10;
      filter: drop-shadow(0 0 14px rgba(239, 68, 68, 0.60));
    }
    .node-match .node-card {
      stroke: var(--match);
      stroke-width: 3.4;
      filter: drop-shadow(0 6px 11px rgba(245, 158, 11, 0.25));
    }
    .node-selected .node-card {
      stroke: var(--selected);
      stroke-width: 3.2;
      filter: drop-shadow(0 6px 12px rgba(2, 132, 199, 0.30));
    }
    .node-active-match .node-card {
      stroke: var(--active-match);
      stroke-width: 4;
      filter: drop-shadow(0 8px 14px rgba(239, 68, 68, 0.35));
    }
    .match-bar {
      fill: var(--match);
      opacity: 0.96;
      pointer-events: none;
    }
    .active-match-bar {
      fill: var(--active-match);
    }
    .badge {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
      fill: #274158;
    }
    .label {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      fill: #172534;
      pointer-events: none;
    }
    .collapsed-summary {
      fill: #52677c;
      font-size: 11px;
    }
    .toggle-button {
      cursor: pointer;
    }
    .toggle-box {
      fill: #ffffff;
      stroke: #8ba5bd;
      stroke-width: 1.1;
    }
    .toggle-icon {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 14px;
      font-weight: 700;
      fill: #274158;
      pointer-events: none;
    }
    .tok-keyword { fill: var(--keyword); color: var(--keyword); font-weight: 700; }
    .tok-symbol { fill: var(--symbol); color: var(--symbol); }
    .tok-number { fill: var(--number); color: var(--number); }
    .tok-literal { fill: var(--literal); color: var(--literal); font-weight: 700; }
    .tok-parameter { fill: var(--parameter); color: var(--parameter); }
    .tok-operator { fill: var(--operator); color: var(--operator); font-weight: 700; }
    .tok-punct { fill: #64748b; color: #64748b; }
    .tok-path { fill: #334155; color: #334155; }
    .tok-space { fill: currentColor; color: inherit; }
    .node-rddl .node-card { fill: var(--rddl); }
    .node-file .node-card { fill: var(--file); }
    .node-domain .node-card { fill: var(--domain); }
    .node-instance .node-card { fill: var(--instance); }
    .node-block .node-card { fill: var(--block); }
    .node-assignment .node-card { fill: var(--assignment); }
    .node-statement .node-card { fill: var(--statement); }
    @keyframes focusPulse {
      0% { opacity: 0.25; }
      35% { opacity: 1; }
      100% { opacity: 0.55; }
    }
    @media (max-width: 900px) {
      main.workspace {
        grid-template-columns: 1fr;
      }
      .split-resizer {
        display: none;
      }
      .text-panel,
      .graph-panel,
      .runtime-panel {
        max-height: none;
      }
      .text-panel {
        min-height: 260px;
        max-height: 360px;
      }
      .runtime-panel {
        min-height: 360px;
      }
    }
    body.is-resizing {
      cursor: col-resize;
      user-select: none;
    }
    body.is-resizing-vertical {
      cursor: row-resize;
      user-select: none;
    }
    body.is-dragging-runtime {
      cursor: grabbing;
      user-select: none;
    }
    body.is-panning-ast {
      cursor: grabbing;
      user-select: none;
    }
    """


def _script() -> str:
    """Return JavaScript for layout, folding, search, and zoom. / 返回布局、折叠、搜索和缩放所需的 JavaScript。"""
    return r"""
    (() => {
      const SVG_NS = "http://www.w3.org/2000/svg";
      const data = JSON.parse(document.getElementById("ast-data").textContent);
      const workspace = document.querySelector("main.workspace");
      const sourcePanel = document.querySelector(".text-panel");
      const graphPanel = document.querySelector(".graph-panel");
      const runtimePanel = document.querySelector(".runtime-panel");
      const svg = document.getElementById("ast-svg");
      const canvas = document.getElementById("canvas");
      const outline = document.getElementById("text-outline");
      const splitResizer = document.getElementById("split-resizer");
      const runtimeResizer = document.getElementById("runtime-resizer");
      const runtimeRoot = document.getElementById("runtime-root");
      const toggleSourcePanel = document.getElementById("toggle-source-panel");
      const toggleAstPanel = document.getElementById("toggle-ast-panel");
      const toggleRuntimePanel = document.getElementById("toggle-runtime-panel");
      const depthInput = document.getElementById("depth-input");
      const searchInput = document.getElementById("search-input");
      const matchCaseInput = document.getElementById("match-case");
      const wholeWordInput = document.getElementById("whole-word");
      const regexInput = document.getElementById("use-regex");
      const visibleStatus = document.getElementById("visible-status");
      const matchStatus = document.getElementById("match-status");
      const selectedStatus = document.getElementById("selected-status");
      const nodes = new Map(data.nodes.map((node) => [node.id, node]));
      const parentById = new Map(data.nodes.map((node) => [node.id, node.parent]));
      const expanded = new Set();
      let matchIds = [];
      let activeMatchIndex = -1;
      let selectedId = null;
      let focusPulseId = null;
      let searchError = "";
      let zoom = 1;
      let latestAstLayout = null;
      let astPan = { x: 0, y: 0 };
      let suppressNextAstClick = false;
      const panelVisible = {
        source: true,
        ast: true,
        runtime: Boolean(runtimePanel),
      };
      let runtimeState = data.runtime && data.runtime.snapshot ? data.runtime.snapshot : null;
      let runtimeGraphZoom = 1;
      let runtimeResizeObserver = null;
      const runtimeDraggedPositions = new Map();
      const runtimeDraggedEdges = new Map();
      const runtimeDraggedLabels = new Map();

      depthInput.max = String(data.maxDepth);
      depthInput.value = String(Math.min(2, data.maxDepth));

      function makeSvg(name, attrs = {}) {
        const element = document.createElementNS(SVG_NS, name);
        for (const [key, value] of Object.entries(attrs)) {
          element.setAttribute(key, String(value));
        }
        return element;
      }

      function makeSpan(className, text) {
        const span = document.createElement("span");
        span.className = className;
        span.textContent = text;
        return span;
      }

      function appendHighlightedLine(container, line) {
        for (const token of line) {
          container.appendChild(makeSpan(token.className, token.text));
        }
      }

      function renderTextOutline() {
        outline.replaceChildren();
        data.files.forEach((file, fileIndex) => {
          const block = document.createElement("section");
          block.className = "source-block";
          block.id = `source-block-${file.id}`;
          block.dataset.fileId = file.id;

          const header = document.createElement("div");
          header.className = "source-header";
          header.appendChild(makeSpan("source-name", file.name));
          header.appendChild(makeSpan("source-path", file.path));
          block.appendChild(header);

          const body = document.createElement("div");
          body.className = "source-body";
          file.lines.forEach((line, index) => {
            const lineNumber = index + 1;
            const row = document.createElement("button");
            row.type = "button";
            row.id = `source-line-${file.id}-${lineNumber}`;
            row.className = "source-line";
            row.dataset.fileId = file.id;
            row.dataset.line = String(lineNumber);

            const gutter = document.createElement("span");
            gutter.className = "source-gutter";
            gutter.textContent = String(lineNumber).padStart(2, " ");
            row.appendChild(gutter);

            const code = document.createElement("span");
            code.className = "source-code";
            appendHighlightedLine(code, line);
            row.appendChild(code);

            row.addEventListener("click", () => {
              const node = bestNodeForLine(file.id, lineNumber);
              if (node) {
                selectNode(node.id, { revealGraph: true, scrollGraph: true, scrollText: false, pulse: true });
              }
            });
            body.appendChild(row);
          });
          block.appendChild(body);
          outline.appendChild(block);
          if (fileIndex < data.files.length - 1) {
            const resizer = document.createElement("div");
            resizer.className = "source-resizer";
            resizer.setAttribute("role", "separator");
            resizer.setAttribute("aria-label", "Resize source file panes");
            outline.appendChild(resizer);
          }
        });
        setupSourceResizers();
      }

      function updateTextOutline(visibleIds) {
        const activeMatchId = activeMatchIndex >= 0 ? matchIds[activeMatchIndex] : null;
        const selected = selectedId ? nodes.get(selectedId) : null;
        const selectedFile = selected ? selected.sourceFile : null;
        const selectedStart = selected && selected.line ? selected.line : null;
        const selectedEnd = selected && selected.endLine ? selected.endLine : selectedStart;
        const matchLines = new Set();
        for (const id of matchIds) {
          const node = nodes.get(id);
          if (!node || !node.sourceFile || !node.line) {
            continue;
          }
          for (let line = node.line; line <= (node.endLine || node.line); line += 1) {
            matchLines.add(`${node.sourceFile}:${line}`);
          }
        }
        const active = activeMatchId ? nodes.get(activeMatchId) : null;
        for (const file of data.files) {
          file.lines.forEach((_line, index) => {
            const lineNumber = index + 1;
            const row = document.getElementById(`source-line-${file.id}-${lineNumber}`);
            if (!row) {
              return;
            }
            const isSelected = selectedFile === file.id
              && selectedStart !== null
              && lineNumber >= selectedStart
              && lineNumber <= selectedEnd;
            const isActive = active
              && active.sourceFile === file.id
              && lineNumber >= active.line
              && lineNumber <= (active.endLine || active.line);
            row.classList.toggle("source-selected", isSelected);
            row.classList.toggle("source-match", matchLines.has(`${file.id}:${lineNumber}`));
            row.classList.toggle("source-active-match", Boolean(isActive));
          });
        }
      }

      function bestNodeForLine(fileId, lineNumber) {
        const candidates = data.nodes.filter((node) => {
          if (node.sourceFile !== fileId || !node.line) {
            return false;
          }
          const endLine = node.endLine || node.line;
          return lineNumber >= node.line && lineNumber <= endLine;
        });
        if (candidates.length === 0) {
          return null;
        }
        candidates.sort((left, right) => {
          const leftSpan = (left.endLine || left.line) - left.line;
          const rightSpan = (right.endLine || right.line) - right.line;
          return leftSpan - rightSpan || right.depth - left.depth;
        });
        return candidates[0];
      }

      function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
      }

      function setupSourceResizers() {
        for (const resizer of outline.querySelectorAll(".source-resizer")) {
          resizer.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            const previous = resizer.previousElementSibling;
            const next = resizer.nextElementSibling;
            if (!previous || !next) {
              return;
            }
            const startY = event.clientY;
            const previousStart = previous.getBoundingClientRect().height;
            const nextStart = next.getBoundingClientRect().height;
            const total = previousStart + nextStart;
            const minHeight = 112;
            resizer.classList.add("resizer-active");
            document.body.classList.add("is-resizing-vertical");

            const onMove = (moveEvent) => {
              const delta = moveEvent.clientY - startY;
              const previousHeight = clamp(previousStart + delta, minHeight, total - minHeight);
              const nextHeight = total - previousHeight;
              previous.style.flex = `0 0 ${previousHeight}px`;
              next.style.flex = `0 0 ${nextHeight}px`;
            };
            const onUp = () => {
              resizer.classList.remove("resizer-active");
              document.body.classList.remove("is-resizing-vertical");
              window.removeEventListener("pointermove", onMove);
              window.removeEventListener("pointerup", onUp);
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp, { once: true });
          });
        }
      }

      function setupPanelToggles() {
        toggleSourcePanel.addEventListener("click", () => togglePanel("source"));
        toggleAstPanel.addEventListener("click", () => togglePanel("ast"));
        if (runtimePanel) {
          toggleRuntimePanel.addEventListener("click", () => togglePanel("runtime"));
        } else {
          toggleRuntimePanel.disabled = true;
          toggleRuntimePanel.setAttribute("aria-pressed", "false");
          toggleRuntimePanel.title = "Runtime panel is only available with DARP internal simulator";
        }
        updatePanelVisibility();
      }

      function togglePanel(name) {
        if (!panelVisible[name]) {
          panelVisible[name] = true;
        } else if (visiblePanelCount() > 1) {
          panelVisible[name] = false;
        }
        updatePanelVisibility();
      }

      function visiblePanelCount() {
        return Number(panelVisible.source) + Number(panelVisible.ast) + Number(panelVisible.runtime);
      }

      function updatePanelVisibility() {
        sourcePanel.hidden = !panelVisible.source;
        graphPanel.hidden = !panelVisible.ast;
        splitResizer.hidden = !(panelVisible.source && panelVisible.ast);
        if (runtimePanel) {
          runtimePanel.hidden = !panelVisible.runtime;
        }
        if (runtimeResizer) {
          runtimeResizer.hidden = !(panelVisible.ast && panelVisible.runtime);
        }
        toggleSourcePanel.setAttribute("aria-pressed", String(panelVisible.source));
        toggleAstPanel.setAttribute("aria-pressed", String(panelVisible.ast));
        if (runtimePanel) {
          toggleRuntimePanel.setAttribute("aria-pressed", String(panelVisible.runtime));
        }
        const onlyOnePanelVisible = visiblePanelCount() <= 1;
        toggleSourcePanel.disabled = panelVisible.source && onlyOnePanelVisible;
        toggleAstPanel.disabled = panelVisible.ast && onlyOnePanelVisible;
        if (runtimePanel) {
          toggleRuntimePanel.disabled = panelVisible.runtime && onlyOnePanelVisible;
        }
        const columns = [];
        if (panelVisible.source) {
          columns.push("minmax(280px, var(--source-width, 28vw))");
        }
        if (panelVisible.source && panelVisible.ast) {
          columns.push("10px");
        }
        if (panelVisible.ast) {
          columns.push("minmax(360px, 1fr)");
        }
        if (panelVisible.ast && panelVisible.runtime) {
          columns.push("10px");
        }
        if (panelVisible.runtime) {
          columns.push("minmax(300px, var(--runtime-width, 28vw))");
        }
        workspace.style.gridTemplateColumns = columns.join(" ") || "minmax(360px, 1fr)";
        workspace.classList.toggle("source-hidden", !panelVisible.source);
        workspace.classList.toggle("ast-hidden", !panelVisible.ast);
        workspace.classList.toggle("runtime-hidden", !panelVisible.runtime);
        if (panelVisible.ast) {
          window.requestAnimationFrame(render);
        }
        if (panelVisible.runtime) {
          window.requestAnimationFrame(refreshRuntime);
        }
      }

      function setupSplitResizer() {
        splitResizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          const startX = event.clientX;
          const sourcePanel = document.querySelector(".text-panel");
          const sourceStart = sourcePanel.getBoundingClientRect().width;
          const workspaceWidth = workspace.getBoundingClientRect().width;
          const minSource = 280;
          const minGraph = 320;
          const runtimePanel = document.querySelector(".runtime-panel");
          const runtimeWidth = runtimePanel ? runtimePanel.getBoundingClientRect().width : 0;
          splitResizer.classList.add("resizer-active");
          document.body.classList.add("is-resizing");

          const onMove = (moveEvent) => {
            const delta = moveEvent.clientX - startX;
            const reservedWidth = runtimePanel ? runtimeWidth + 28 : 14;
            const width = clamp(sourceStart + delta, minSource, workspaceWidth - minGraph - reservedWidth);
            workspace.style.setProperty("--source-width", `${width}px`);
          };
          const onUp = () => {
            splitResizer.classList.remove("resizer-active");
            document.body.classList.remove("is-resizing");
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
          };
          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp, { once: true });
        });
        if (!runtimeResizer) {
          return;
        }
        runtimeResizer.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          const startX = event.clientX;
          const runtimePanel = document.querySelector(".runtime-panel");
          const runtimeStart = runtimePanel.getBoundingClientRect().width;
          const workspaceWidth = workspace.getBoundingClientRect().width;
          const sourceWidth = document.querySelector(".text-panel").getBoundingClientRect().width;
          const minRuntime = 280;
          const minGraph = 320;
          runtimeResizer.classList.add("resizer-active");
          document.body.classList.add("is-resizing");

          const onMove = (moveEvent) => {
            const delta = moveEvent.clientX - startX;
            const width = clamp(runtimeStart - delta, minRuntime, workspaceWidth - sourceWidth - minGraph - 28);
            workspace.style.setProperty("--runtime-width", `${width}px`);
          };
          const onUp = () => {
            runtimeResizer.classList.remove("resizer-active");
            document.body.classList.remove("is-resizing");
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
          };
          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp, { once: true });
        });
      }

      function setupAstCanvasPan() {
        canvas.addEventListener("pointerdown", (event) => {
          if (event.button !== 0 || closestFromEvent(event, ".toggle-button")) {
            return;
          }
          const startX = event.clientX;
          const startY = event.clientY;
          const startPan = { x: astPan.x, y: astPan.y };
          let didDrag = false;
          canvas.classList.add("ast-panning");
          document.body.classList.add("is-panning-ast");

          const onMove = (moveEvent) => {
            const deltaX = moveEvent.clientX - startX;
            const deltaY = moveEvent.clientY - startY;
            if (Math.hypot(deltaX, deltaY) > 3) {
              didDrag = true;
              moveEvent.preventDefault();
            }
            astPan = {
              x: startPan.x - deltaX / zoom,
              y: startPan.y - deltaY / zoom,
            };
            applyAstViewBox();
          };
          const onUp = () => {
            canvas.classList.remove("ast-panning");
            document.body.classList.remove("is-panning-ast");
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            if (didDrag) {
              suppressNextAstClick = true;
              window.setTimeout(() => {
                suppressNextAstClick = false;
              }, 120);
            }
          };
          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp, { once: true });
        });
      }

      function closestFromEvent(event, selector) {
        return event.target && event.target.closest ? event.target.closest(selector) : null;
      }

      function clampAstPan(layout = latestAstLayout) {
        if (!layout) {
          return;
        }
        const viewport = astViewport(layout);
        const marginX = Math.max(120, viewport.viewWidth * 0.18);
        const marginY = Math.max(120, viewport.viewHeight * 0.18);
        astPan = {
          x: clamp(
            astPan.x,
            Math.min(-marginX, layout.width - viewport.viewWidth - marginX),
            Math.max(marginX, layout.width - viewport.viewWidth + marginX),
          ),
          y: clamp(
            astPan.y,
            Math.min(-marginY, layout.height - viewport.viewHeight - marginY),
            Math.max(marginY, layout.height - viewport.viewHeight + marginY),
          ),
        };
      }

      function applyAstViewBox(layout = latestAstLayout) {
        if (!layout) {
          return null;
        }
        const viewport = astViewport(layout);
        clampAstPan(layout);
        svg.setAttribute(
          "viewBox",
          `${astPan.x.toFixed(1)} ${astPan.y.toFixed(1)} ${viewport.viewWidth.toFixed(1)} ${viewport.viewHeight.toFixed(1)}`,
        );
        return viewport;
      }

      function astViewport(layout) {
        const bounds = canvas.getBoundingClientRect();
        const visibleWidth = Math.max(320, Math.floor(bounds.width - 24));
        const visibleHeight = Math.max(260, Math.floor(bounds.height - 24));
        const displayWidth = Math.max(visibleWidth, Math.ceil(layout.width * zoom));
        const displayHeight = Math.max(visibleHeight, Math.ceil(layout.height * zoom));
        return {
          displayWidth,
          displayHeight,
          viewWidth: displayWidth / zoom,
          viewHeight: displayHeight / zoom,
        };
      }

      function escapeRegExp(value) {
        return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      }

      function buildSearchMatcher() {
        const query = searchInput.value.trim();
        if (!query) {
          return { matcher: null, error: "" };
        }
        const source = regexInput.checked ? query : escapeRegExp(query);
        const wordChars = "A-Za-z0-9_?'-";
        const pattern = wholeWordInput.checked
          ? `(^|[^${wordChars}])(?:${source})(?=$|[^${wordChars}])`
          : source;
        try {
          const regex = new RegExp(pattern, matchCaseInput.checked ? "u" : "iu");
          return { matcher: (text) => regex.test(text), error: "" };
        } catch (error) {
          return { matcher: null, error: `Invalid regex: ${error.message}` };
        }
      }

      function expandAll() {
        expanded.clear();
        for (const node of data.nodes) {
          if (node.children.length > 0) {
            expanded.add(node.id);
          }
        }
      }

      function collapseAll() {
        expanded.clear();
      }

      function expandToDepth(depth) {
        expanded.clear();
        for (const node of data.nodes) {
          if (node.children.length > 0 && node.depth < depth) {
            expanded.add(node.id);
          }
        }
      }

      function revealNode(id) {
        let cursor = parentById.get(id);
        while (cursor) {
          expanded.add(cursor);
          cursor = parentById.get(cursor);
        }
      }

      function revealMatches() {
        for (const id of matchIds) {
          revealNode(id);
        }
      }

      function nodeMetrics(node) {
        const isCollapsed = node.children.length > 0 && !expanded.has(node.id);
        const collapsedLines = node.collapsedLines || [];
        const lines = isCollapsed && collapsedLines.length > 0
          ? node.lines.concat(collapsedLines)
          : node.lines;
        return {
          isCollapsed,
          width: isCollapsed ? (node.collapsedWidth || node.width) : node.width,
          height: isCollapsed ? (node.collapsedHeight || node.height) : node.height,
          lines,
        };
      }

      function measureVisibleTree(id, depth, levelHeights) {
        const node = nodes.get(id);
        const metrics = nodeMetrics(node);
        levelHeights.set(depth, Math.max(levelHeights.get(depth) || 0, metrics.height));
        const childBranches = expanded.has(id)
          ? node.children.map((childId) => measureVisibleTree(childId, depth + 1, levelHeights))
          : [];
        const childWidth = childBranches.length
          ? childBranches.reduce((sum, child) => sum + child.subtreeWidth, 0)
            + data.layout.hGap * (childBranches.length - 1)
          : 0;
        return {
          id,
          depth,
          children: childBranches,
          width: metrics.width,
          height: metrics.height,
          subtreeWidth: Math.max(metrics.width, childWidth),
        };
      }

      function placeVisibleTree(branch, left, levelY, placements, edges) {
        const centerX = left + branch.subtreeWidth / 2;
        placements.push({
          id: branch.id,
          x: centerX - branch.width / 2,
          y: levelY.get(branch.depth),
          width: branch.width,
          height: branch.height,
        });
        if (branch.children.length === 0) {
          return;
        }
        const childrenWidth = branch.children.reduce((sum, child) => sum + child.subtreeWidth, 0)
          + data.layout.hGap * (branch.children.length - 1);
        let childLeft = left + (branch.subtreeWidth - childrenWidth) / 2;
        for (const child of branch.children) {
          edges.push([branch.id, child.id]);
          placeVisibleTree(child, childLeft, levelY, placements, edges);
          childLeft += child.subtreeWidth + data.layout.hGap;
        }
      }

      function layoutVisible() {
        const levelHeights = new Map();
        const tree = measureVisibleTree(data.rootId, 0, levelHeights);
        const levelY = new Map();
        let currentY = data.layout.marginY;
        const maxVisibleDepth = Math.max(...levelHeights.keys());
        for (let depth = 0; depth <= maxVisibleDepth; depth += 1) {
          levelY.set(depth, currentY);
          currentY += (levelHeights.get(depth) || 0) + data.layout.vGap;
        }
        const placements = [];
        const edges = [];
        placeVisibleTree(tree, data.layout.marginX, levelY, placements, edges);
        return {
          placements,
          edges,
          width: tree.subtreeWidth + data.layout.marginX * 2,
          height: currentY - data.layout.vGap + data.layout.marginY,
        };
      }

      function render() {
        const layout = layoutVisible();
        latestAstLayout = layout;
        const placementById = new Map(layout.placements.map((placement) => [placement.id, placement]));
        const visibleIds = new Set(layout.placements.map((placement) => placement.id));
        svg.replaceChildren();
        const viewport = applyAstViewBox(layout);
        svg.setAttribute("width", String(viewport.displayWidth));
        svg.setAttribute("height", String(viewport.displayHeight));
        svg.style.width = `${viewport.displayWidth}px`;
        svg.style.height = `${viewport.displayHeight}px`;

        const edgeLayer = makeSvg("g", { class: "edges" });
        for (const [parentId, childId] of layout.edges) {
          const parentPlace = placementById.get(parentId);
          const childPlace = placementById.get(childId);
          const x1 = parentPlace.x + parentPlace.width / 2;
          const y1 = parentPlace.y + parentPlace.height;
          const x2 = childPlace.x + childPlace.width / 2;
          const y2 = childPlace.y;
          const midY = (y1 + y2) / 2;
          edgeLayer.appendChild(makeSvg("path", {
            class: "edge",
            d: `M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${x1.toFixed(1)} ${midY.toFixed(1)}, ${x2.toFixed(1)} ${midY.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`,
          }));
        }
        svg.appendChild(edgeLayer);

        const nodeLayer = makeSvg("g", { class: "nodes" });
        for (const placement of layout.placements.sort((a, b) => a.y - b.y || a.x - b.x)) {
          nodeLayer.appendChild(renderNode(nodes.get(placement.id), placement));
        }
        svg.appendChild(nodeLayer);
        updateStatus(visibleIds);
        updateTextOutline(visibleIds);
      }

      function renderNode(node, placement) {
        const classes = ["node", node.cssClass];
        const metrics = nodeMetrics(node);
        const isMatch = matchIds.includes(node.id);
        const isActiveMatch = activeMatchIndex >= 0 && matchIds[activeMatchIndex] === node.id;
        const isSelected = selectedId === node.id;
        const isFocusPulse = focusPulseId === node.id;
        if (metrics.isCollapsed) {
          classes.push("node-collapsed");
        }
        if (isMatch) {
          classes.push("node-match");
        }
        if (isActiveMatch) {
          classes.push("node-active-match");
        }
        if (isSelected) {
          classes.push("node-selected");
        }
        if (isFocusPulse) {
          classes.push("node-focus-pulse");
        }
        const group = makeSvg("g", {
          id: `ast-node-${node.id}`,
          class: classes.join(" "),
          transform: `translate(${placement.x.toFixed(1)} ${placement.y.toFixed(1)})`,
        });
        if (isMatch || isActiveMatch || isSelected || isFocusPulse) {
          group.appendChild(makeSvg("rect", {
            class: "node-halo",
            x: -7,
            y: -7,
            width: metrics.width + 14,
            height: metrics.height + 14,
            rx: 13,
          }));
        }
        group.appendChild(makeSvg("rect", {
          class: "node-card",
          x: 0,
          y: 0,
          width: metrics.width,
          height: metrics.height,
          rx: 8,
        }));
        if (isMatch) {
          group.appendChild(makeSvg("rect", {
            class: isActiveMatch ? "match-bar active-match-bar" : "match-bar",
            x: 10,
            y: 5,
            width: Math.max(24, metrics.width - 20),
            height: 5,
            rx: 3,
          }));
        }

        const title = makeSvg("title");
        title.textContent = `${node.kind}: ${node.label}`;
        group.appendChild(title);

        const badge = makeSvg("text", {
          class: "badge",
          x: data.layout.padX,
          y: data.layout.badgeBaselineY,
          "text-anchor": "start",
        });
        badge.textContent = node.kind;
        group.appendChild(badge);

        metrics.lines.forEach((line, lineIndex) => {
          const text = makeSvg("text", {
            class: lineIndex >= node.lines.length ? "label collapsed-summary" : "label",
            x: data.layout.padX,
            y: data.layout.labelBaselineY + lineIndex * data.layout.lineHeight,
            "xml:space": "preserve",
          });
          for (const token of line) {
            const span = makeSvg("tspan", { class: token.className });
            span.textContent = token.text;
            text.appendChild(span);
          }
          group.appendChild(text);
        });

        if (node.children.length > 0) {
          const toggle = makeSvg("g", {
            class: "toggle-button",
            transform: `translate(${metrics.width - 28} 8)`,
            role: "button",
            tabindex: "0",
          });
          toggle.appendChild(makeSvg("rect", {
            class: "toggle-box",
            x: 0,
            y: 0,
            width: 20,
            height: 20,
            rx: 5,
          }));
          const icon = makeSvg("text", {
            class: "toggle-icon",
            x: 10,
            y: 15,
            "text-anchor": "middle",
          });
          icon.textContent = expanded.has(node.id) ? "-" : "+";
          toggle.appendChild(icon);
          toggle.addEventListener("click", (event) => {
            event.stopPropagation();
            toggleExpanded(node.id);
          });
          toggle.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              toggleExpanded(node.id);
            }
          });
          group.appendChild(toggle);
        }

        group.addEventListener("click", () => {
          if (suppressNextAstClick) {
            return;
          }
          selectNode(node.id, { revealGraph: false, scrollGraph: false, scrollText: true, pulse: false });
        });
        group.addEventListener("dblclick", () => {
          if (suppressNextAstClick) {
            return;
          }
          if (node.children.length > 0) {
            toggleExpanded(node.id);
          }
        });
        return group;
      }

      function toggleExpanded(id) {
        if (expanded.has(id)) {
          expanded.delete(id);
        } else {
          expanded.add(id);
        }
        selectedId = id;
        focusPulseId = id;
        render();
        scrollNodeIntoView(id);
        scrollTextIntoView(id);
        clearFocusPulse(id);
      }

      function selectNode(id, options = {}) {
        selectedId = id;
        if (options.revealGraph) {
          revealNode(id);
        }
        if (options.pulse) {
          focusPulseId = id;
        }
        render();
        if (options.scrollGraph !== false) {
          scrollNodeIntoView(id);
        }
        if (options.scrollText !== false) {
          scrollTextIntoView(id);
        }
        if (options.pulse) {
          clearFocusPulse(id);
        }
      }

      function clearFocusPulse(id) {
        window.setTimeout(() => {
          if (focusPulseId === id) {
            focusPulseId = null;
            render();
          }
        }, 1200);
      }

      function updateStatus(visibleIds) {
        visibleStatus.textContent = `Visible ${visibleIds.size}/${data.totalNodes}`;
        searchInput.classList.toggle("search-error", Boolean(searchError));
        if (searchError) {
          matchStatus.textContent = searchError;
          if (selectedId) {
            const selected = nodes.get(selectedId);
            selectedStatus.textContent = `Selected ${selected.kind}: ${selected.label}`;
          } else {
            selectedStatus.textContent = "Selected none";
          }
          return;
        }
        const matchTotal = matchIds.length;
        matchStatus.textContent = activeMatchIndex >= 0
          ? `Matches ${activeMatchIndex + 1}/${matchTotal}`
          : `Matches ${matchTotal}`;
        if (selectedId) {
          const selected = nodes.get(selectedId);
          selectedStatus.textContent = `Selected ${selected.kind}: ${selected.label}`;
        } else {
          selectedStatus.textContent = "Selected none";
        }
      }

      function applySearch() {
        const result = buildSearchMatcher();
        searchError = result.error;
        matchIds = result.matcher
          ? data.nodes.filter((node) => result.matcher(node.searchText)).map((node) => node.id)
          : [];
        activeMatchIndex = matchIds.length > 0 ? 0 : -1;
        if (activeMatchIndex >= 0) {
          revealNode(matchIds[activeMatchIndex]);
          selectedId = matchIds[activeMatchIndex];
        }
        render();
        scrollActiveMatch();
      }

      function moveMatch(delta) {
        if (matchIds.length === 0) {
          return;
        }
        activeMatchIndex = (activeMatchIndex + delta + matchIds.length) % matchIds.length;
        const id = matchIds[activeMatchIndex];
        revealNode(id);
        selectedId = id;
        render();
        scrollActiveMatch();
      }

      function scrollActiveMatch() {
        if (activeMatchIndex < 0) {
          return;
        }
        const id = matchIds[activeMatchIndex];
        scrollNodeIntoView(id);
        scrollTextIntoView(id);
      }

      function scrollNodeIntoView(id) {
        window.requestAnimationFrame(() => {
          const target = document.getElementById(`ast-node-${id}`);
          if (target && target.scrollIntoView) {
            target.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
          }
        });
      }

      function scrollTextIntoView(id) {
        window.requestAnimationFrame(() => {
          const node = nodes.get(id);
          const target = node && node.sourceFile && node.line
            ? document.getElementById(`source-line-${node.sourceFile}-${node.line}`)
            : null;
          if (target && target.scrollIntoView) {
            target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
          }
        });
      }

      function setZoom(nextZoom) {
        zoom = Math.min(1.8, Math.max(0.35, nextZoom));
        document.getElementById("zoom-reset").textContent = `${Math.round(zoom * 100)}%`;
        render();
      }

      function runtimeEnabled() {
        return data.runtime && data.runtime.enabled && data.runtime.problem;
      }

      function renderRuntime() {
        if (!runtimeRoot) {
          return;
        }
        if (runtimeResizeObserver) {
          runtimeResizeObserver.disconnect();
          runtimeResizeObserver = null;
        }
        runtimeRoot.replaceChildren();
        if (!runtimeEnabled()) {
          const message = document.createElement("div");
          message.className = "runtime-message";
          message.textContent = "Runtime simulator is disabled. Start DARP with --domain and --instance, or pass --simulator darp, to advance DARP-selected actions.";
          runtimeRoot.appendChild(message);
          return;
        }

        const controls = document.createElement("div");
        controls.className = "runtime-controls";
        const plannedAction = document.createElement("div");
        plannedAction.className = "runtime-planned-action";
        const plannedLabel = document.createElement("strong");
        plannedLabel.textContent = "DARP action";
        plannedAction.appendChild(plannedLabel);
        plannedAction.appendChild(document.createTextNode(runtimeState && runtimeState.planned_action ? runtimeState.planned_action : "none"));
        controls.appendChild(plannedAction);

        const stepButton = document.createElement("button");
        stepButton.type = "button";
        stepButton.textContent = "Next step";
        stepButton.disabled = runtimeState && runtimeState.done;
        stepButton.addEventListener("click", () => stepRuntime());
        controls.appendChild(stepButton);

        const resetButton = document.createElement("button");
        resetButton.type = "button";
        resetButton.textContent = "Reset";
        resetButton.addEventListener("click", resetRuntime);
        controls.appendChild(resetButton);

        const runtimeZoomOut = document.createElement("button");
        runtimeZoomOut.type = "button";
        runtimeZoomOut.textContent = "-";
        runtimeZoomOut.title = "Zoom out state machine";
        runtimeZoomOut.addEventListener("click", () => setRuntimeZoom(runtimeGraphZoom - 0.1));
        controls.appendChild(runtimeZoomOut);

        const runtimeZoomReset = document.createElement("button");
        runtimeZoomReset.type = "button";
        runtimeZoomReset.id = "runtime-zoom-reset";
        runtimeZoomReset.textContent = `${Math.round(runtimeGraphZoom * 100)}%`;
        runtimeZoomReset.title = "Reset state machine zoom";
        runtimeZoomReset.addEventListener("click", () => setRuntimeZoom(1));
        controls.appendChild(runtimeZoomReset);

        const runtimeZoomIn = document.createElement("button");
        runtimeZoomIn.type = "button";
        runtimeZoomIn.textContent = "+";
        runtimeZoomIn.title = "Zoom in state machine";
        runtimeZoomIn.addEventListener("click", () => setRuntimeZoom(runtimeGraphZoom + 0.1));
        controls.appendChild(runtimeZoomIn);

        const runtimeAutoLayout = document.createElement("button");
        runtimeAutoLayout.type = "button";
        runtimeAutoLayout.textContent = "Auto";
        runtimeAutoLayout.title = "Reset dragged node, edge, and label positions";
        runtimeAutoLayout.addEventListener("click", () => {
          runtimeDraggedPositions.clear();
          runtimeDraggedEdges.clear();
          runtimeDraggedLabels.clear();
          setRuntimeZoom(1);
        });
        controls.appendChild(runtimeAutoLayout);
        runtimeRoot.appendChild(controls);

        const info = document.createElement("div");
        info.className = "runtime-info";
        info.appendChild(runtimeInfoLine("Initialization", runtimeState ? `${runtimeState.initial_state} / obs ${runtimeState.initial_observation}` : "-"));
        info.appendChild(runtimeInfoLine("Planner", runtimeState && runtimeState.planner ? runtimeState.planner : "none"));
        info.appendChild(runtimeInfoLine("Last action", runtimeState && runtimeState.last_action ? runtimeState.last_action : "none"));
        info.appendChild(runtimeInfoLine("DARP planned action", runtimeState && runtimeState.planned_action ? runtimeState.planned_action : "none"));
        info.appendChild(runtimeInfoLine("Belief peak", runtimeState && runtimeState.belief ? runtimeBeliefSummary(runtimeState.belief) : "-"));
        info.appendChild(runtimeInfoLine("Available actions", data.runtime.problem.actions.join(", ")));
        runtimeRoot.appendChild(info);

        const status = document.createElement("div");
        status.className = "runtime-status";
        status.appendChild(runtimeChip("State", runtimeState ? runtimeState.state : "not reset"));
        status.appendChild(runtimeChip("Observation", runtimeState ? runtimeState.observation : "-"));
        status.appendChild(runtimeChip("Reward", runtimeState ? String(runtimeState.reward) : "0"));
        status.appendChild(runtimeChip("Step", runtimeState ? `${runtimeState.step}/${data.runtime.problem.horizon}` : "0"));
        runtimeRoot.appendChild(status);

        const stage = document.createElement("div");
        stage.className = "runtime-stage";
        runtimeRoot.appendChild(stage);
        renderRuntimeGraph(stage);
        if (window.ResizeObserver) {
          runtimeResizeObserver = new ResizeObserver(() => renderRuntimeGraph(stage));
          runtimeResizeObserver.observe(stage);
        }

        const trace = document.createElement("div");
        trace.className = "runtime-trace";
        const rows = runtimeState && runtimeState.trace ? runtimeState.trace.slice(-8) : [];
        trace.textContent = rows.length
          ? rows.map((row) => `t=${row.step} ${row.action} -> ${row.state} r=${row.reward}`).join("\n")
          : "No executed steps yet.";
        runtimeRoot.appendChild(trace);
      }

      function runtimeChip(label, value) {
        const chip = document.createElement("div");
        chip.className = "runtime-chip";
        const strong = document.createElement("strong");
        strong.textContent = label;
        chip.appendChild(strong);
        chip.appendChild(document.createTextNode(value === null || value === undefined ? "-" : String(value)));
        return chip;
      }

      function runtimeInfoLine(label, value) {
        const row = document.createElement("div");
        const strong = document.createElement("strong");
        strong.textContent = `${label}: `;
        row.appendChild(strong);
        row.appendChild(document.createTextNode(value === null || value === undefined ? "-" : String(value)));
        return row;
      }

      function runtimeBeliefSummary(belief) {
        const entries = Object.entries(belief || {})
          .filter((entry) => Number(entry[1]) > 1e-6)
          .sort((left, right) => Number(right[1]) - Number(left[1]));
        if (!entries.length) {
          return "empty";
        }
        const [state, probability] = entries[0];
        return `${state}=${formatProbability(probability)}`;
      }

      function setRuntimeZoom(nextZoom) {
        runtimeGraphZoom = clamp(nextZoom, 0.45, 2.5);
        const zoomLabel = document.getElementById("runtime-zoom-reset");
        if (zoomLabel) {
          zoomLabel.textContent = `${Math.round(runtimeGraphZoom * 100)}%`;
        }
        const stage = document.querySelector(".runtime-stage");
        if (stage) {
          renderRuntimeGraph(stage);
        }
      }

      function renderRuntimeGraph(stage) {
        if (!runtimeEnabled() || !stage) {
          return;
        }
        const bounds = stage.getBoundingClientRect();
        const width = Math.max(320, Math.floor(bounds.width - 2));
        const height = Math.max(260, Math.floor(bounds.height - 2));
        stage.replaceChildren(renderStateMachineSvg(data.runtime.problem, width, height, stage));
      }

      function renderStateMachineSvg(problem, width, height, stage) {
        const nodeRadius = clamp(Math.min(width, height) / 36, 10, 18);
        const labelFontSize = clamp(nodeRadius * 0.64, 8.5, 12.5);
        const layout = runtimePositions(problem, nodeRadius, width, height);
        const positions = layout.positions;
        const displayWidth = Math.max(1, Math.round(layout.width * runtimeGraphZoom));
        const displayHeight = Math.max(1, Math.round(layout.height * runtimeGraphZoom));
        const svgElement = makeSvg("svg", {
          class: "runtime-svg",
          viewBox: `0 0 ${layout.width} ${layout.height}`,
          width: displayWidth,
          height: displayHeight,
        });
        svgElement.style.width = `${displayWidth}px`;
        svgElement.style.height = `${displayHeight}px`;
        svgElement.appendChild(runtimeArrowDefs());
        const markers = problem.markers || { starts: [], risks: [], goals: [] };
        const groups = groupedTransitions(problem);
        const laneSpacing = Math.max(nodeRadius * 4.4, Math.min(layout.width, layout.height) * 0.075, 58);

        groups.forEach((group) => {
          const from = positions.get(group.from);
          const to = positions.get(group.to);
          if (!from || !to) {
            return;
          }
          const isActive = group.items.some((transition) => isActiveTransition(transition));
          if (group.from === group.to) {
            const laneOffset = Number(group.laneOffset || 0);
            const sideFromCenter = Math.sign(from.x - layout.width / 2);
            const side = sideFromCenter || (laneOffset < 0 ? -1 : 1);
            const verticalFromCenter = Math.sign(from.y - layout.height / 2);
            const verticalSide = verticalFromCenter || -1;
            const laneSize = Math.abs(laneOffset) + 1;
            const loopSpread = Math.max(laneSpacing * (0.8 + laneSize * 0.38), nodeRadius * 2.8);
            const loopLift = Math.max(laneSpacing * (0.7 + laneSize * 0.32), nodeRadius * 2.4);
            const defaultControl = { x: from.x + side * loopSpread, y: from.y + verticalSide * loopLift };
            const control = runtimePointWithDrag(runtimeDraggedEdges, group.id, defaultControl.x, defaultControl.y, layout.width, layout.height);
            const pathData = `M ${from.x.toFixed(1)} ${(from.y + verticalSide * nodeRadius).toFixed(1)} C ${control.x.toFixed(1)} ${(control.y + verticalSide * nodeRadius).toFixed(1)}, ${(control.x + side * loopSpread).toFixed(1)} ${control.y.toFixed(1)}, ${(from.x + side * nodeRadius).toFixed(1)} ${(from.y + verticalSide * nodeRadius * 0.25).toFixed(1)}`;
            const edgePath = makeSvg("path", {
              class: isActive ? "runtime-edge active" : "runtime-edge",
              d: pathData,
              "marker-end": isActive ? "url(#runtime-arrow-active)" : "url(#runtime-arrow)",
            });
            svgElement.appendChild(edgePath);
            const labelDefaultX = clamp(control.x + side * nodeRadius, 36, layout.width - 36);
            const labelDefaultY = clamp(control.y + verticalSide * nodeRadius * 0.65, 20, layout.height - 20);
            const labelPosition = runtimePointWithDrag(runtimeDraggedLabels, group.id, labelDefaultX, labelDefaultY, layout.width, layout.height);
            const dragPath = makeSvg("path", {
              class: "runtime-edge-drag-area",
              d: pathData,
            });
            dragPath.addEventListener("pointerdown", (event) => {
              startRuntimeLinkedElementDrag(
                event,
                svgElement,
                stage,
                runtimeDraggedEdges,
                group.id,
                control,
                runtimeDraggedLabels,
                labelPosition,
              );
            });
            svgElement.appendChild(dragPath);
            appendRuntimeEdgeLabelLink(svgElement, control, labelPosition, isActive);
            appendRuntimeEdgeLabel(
              svgElement,
              labelPosition.x,
              labelPosition.y,
              transitionGroupLabel(group),
              labelFontSize,
              (event) => startRuntimeLinkedElementDrag(
                event,
                svgElement,
                stage,
                runtimeDraggedLabels,
                group.id,
                labelPosition,
                runtimeDraggedEdges,
                control,
              ),
            );
            return;
          }
          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const length = Math.max(1, Math.hypot(dx, dy));
          const ux = dx / length;
          const uy = dy / length;
          const perpendicularX = -uy;
          const perpendicularY = ux;
          const directionSign = String(group.from) <= String(group.to) ? 1 : -1;
          const offset = Number(group.laneOffset || 0) * laneSpacing * directionSign;
          const startX = from.x + ux * nodeRadius;
          const startY = from.y + uy * nodeRadius;
          const endX = to.x - ux * (nodeRadius + 8);
          const endY = to.y - uy * (nodeRadius + 8);
          const defaultControlX = (startX + endX) / 2 + perpendicularX * offset;
          const defaultControlY = (startY + endY) / 2 + perpendicularY * offset;
          const control = runtimePointWithDrag(runtimeDraggedEdges, group.id, defaultControlX, defaultControlY, layout.width, layout.height);
          const pathData = `M ${startX.toFixed(1)} ${startY.toFixed(1)} Q ${control.x.toFixed(1)} ${control.y.toFixed(1)} ${endX.toFixed(1)} ${endY.toFixed(1)}`;
          const edgePath = makeSvg("path", {
            class: isActive ? "runtime-edge active" : "runtime-edge",
            d: pathData,
            "marker-end": isActive ? "url(#runtime-arrow-active)" : "url(#runtime-arrow)",
          });
          svgElement.appendChild(edgePath);
          const labelDefaultX = clamp(control.x + perpendicularX * nodeRadius * 0.65, 34, layout.width - 34);
          const labelDefaultY = clamp(control.y + perpendicularY * nodeRadius * 0.65, 18, layout.height - 18);
          const labelPosition = runtimePointWithDrag(runtimeDraggedLabels, group.id, labelDefaultX, labelDefaultY, layout.width, layout.height);
          const dragPath = makeSvg("path", {
            class: "runtime-edge-drag-area",
            d: pathData,
          });
          dragPath.addEventListener("pointerdown", (event) => {
            startRuntimeLinkedElementDrag(
              event,
              svgElement,
              stage,
              runtimeDraggedEdges,
              group.id,
              control,
              runtimeDraggedLabels,
              labelPosition,
            );
          });
          svgElement.appendChild(dragPath);
          appendRuntimeEdgeLabelLink(svgElement, control, labelPosition, isActive);
          appendRuntimeEdgeLabel(
            svgElement,
            labelPosition.x,
            labelPosition.y,
            transitionGroupLabel(group),
            labelFontSize,
            (event) => startRuntimeLinkedElementDrag(
              event,
              svgElement,
              stage,
              runtimeDraggedLabels,
              group.id,
              labelPosition,
              runtimeDraggedEdges,
              control,
            ),
          );
        });

        for (const state of problem.states) {
          const position = positions.get(state);
          if (!position) {
            continue;
          }
          const classes = ["runtime-cell"];
          if (runtimeState && runtimeState.state === state) {
            classes.push("current");
          }
          if (markers.starts.includes(state)) {
            classes.push("start");
          }
          if (markers.risks.includes(state)) {
            classes.push("risk");
          }
          if (markers.goals.includes(state)) {
            classes.push("goal");
          }
          const group = makeSvg("g", {
            class: "runtime-node-group",
            transform: `translate(${position.x.toFixed(1)} ${position.y.toFixed(1)})`,
          });
          group.appendChild(makeSvg("circle", {
            class: classes.join(" "),
            cx: 0,
            cy: 0,
            r: nodeRadius,
          }));
          const label = makeSvg("text", {
            class: "runtime-cell-label",
            x: 0,
            y: nodeRadius * 0.28,
            style: `font-size: ${clamp(nodeRadius * 0.66, 8, 15)}px;`,
            "text-anchor": "middle",
          });
          label.textContent = state;
          group.appendChild(label);
          const markerLabel = markerText(markers, state);
          if (markerLabel) {
            const marker = makeSvg("text", {
              class: "runtime-marker-label",
              x: 0,
              y: nodeRadius + Math.max(11, nodeRadius * 0.75),
              style: `font-size: ${clamp(nodeRadius * 0.38, 6, 10)}px;`,
              "text-anchor": "middle",
            });
            marker.textContent = markerLabel;
            group.appendChild(marker);
          }
          if (runtimeState && runtimeState.state === state) {
            group.appendChild(makeSvg("circle", {
              class: "runtime-token",
              cx: nodeRadius * 0.58,
              cy: -nodeRadius * 0.58,
              r: Math.max(4, nodeRadius * 0.28),
            }));
          }
          group.addEventListener("pointerdown", (event) => {
            startRuntimeNodeDrag(event, svgElement, stage, state, position, nodeRadius);
          });
          svgElement.appendChild(group);
        }
        return svgElement;
      }

      function runtimeArrowDefs() {
        const defs = makeSvg("defs");
        const marker = makeSvg("marker", {
          id: "runtime-arrow",
          markerWidth: 8,
          markerHeight: 8,
          refX: 7,
          refY: 4,
          orient: "auto",
          markerUnits: "strokeWidth",
        });
        marker.appendChild(makeSvg("path", {
          d: "M 0 0 L 8 4 L 0 8 z",
          fill: "#8ca4ba",
        }));
        const activeMarker = makeSvg("marker", {
          id: "runtime-arrow-active",
          markerWidth: 8,
          markerHeight: 8,
          refX: 7,
          refY: 4,
          orient: "auto",
          markerUnits: "strokeWidth",
        });
        activeMarker.appendChild(makeSvg("path", {
          d: "M 0 0 L 8 4 L 0 8 z",
          fill: "#0284c7",
        }));
        defs.appendChild(marker);
        defs.appendChild(activeMarker);
        return defs;
      }

      function runtimePositions(problem, nodeRadius, width, height) {
        const positions = new Map();
        if (problem.layout && problem.layout.kind === "grid") {
          const rows = problem.layout.rows;
          const cols = problem.layout.cols;
          const insetX = Math.max(80, nodeRadius * 6);
          const insetY = Math.max(70, nodeRadius * 5.5);
          const maxGridWidth = Math.max(1, width - insetX * 2);
          const maxGridHeight = Math.max(1, height - insetY * 2);
          const targetGap = clamp(Math.min(width, height) / 4.1, 95, 190);
          const gapX = cols <= 1 ? 0 : Math.min(maxGridWidth / (cols - 1), targetGap);
          const gapY = rows <= 1 ? 0 : Math.min(maxGridHeight / (rows - 1), targetGap * 0.95);
          const originX = width / 2 - (gapX * (cols - 1)) / 2;
          const originY = height / 2 - (gapY * (rows - 1)) / 2;
          for (const state of problem.states) {
            const cellInfo = problem.layout.cells[state];
            if (!cellInfo) {
              continue;
            }
            const x = cols <= 1
              ? width / 2
              : originX + (cellInfo.col - 1) * gapX;
            const y = rows <= 1
              ? height / 2
              : originY + (cellInfo.row - 1) * gapY;
            positions.set(state, runtimePositionWithDrag(state, x, y, width, height, nodeRadius));
          }
          return { positions, width, height };
        }
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.max(1, Math.min(width, height) / 2 - Math.max(66, nodeRadius * 4));
        problem.states.forEach((state, index) => {
          const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(1, problem.states.length);
          positions.set(
            state,
            runtimePositionWithDrag(
              state,
              centerX + radius * Math.cos(angle),
              centerY + radius * Math.sin(angle),
              width,
              height,
              nodeRadius,
            ),
          );
        });
        return { positions, width, height };
      }

      function runtimePositionWithDrag(state, x, y, width, height, nodeRadius) {
        const dragged = runtimeDraggedPositions.get(state);
        if (!dragged) {
          return { x, y };
        }
        return {
          x: clamp(dragged.x * width, nodeRadius + 10, width - nodeRadius - 10),
          y: clamp(dragged.y * height, nodeRadius + 10, height - nodeRadius - 10),
        };
      }

      function runtimePointWithDrag(store, key, x, y, width, height) {
        const dragged = store.get(key);
        if (!dragged) {
          return { x, y };
        }
        return {
          x: clamp(dragged.x * width, 8, width - 8),
          y: clamp(dragged.y * height, 8, height - 8),
        };
      }

      function groupedTransitions(problem) {
        const groups = [];
        for (const transition of problem.transitions) {
          if (transition.prob <= 0) {
            continue;
          }
          groups.push({
            id: `${transition.from}\u0000${transition.action}\u0000${transition.to}\u0000${groups.length}`,
            from: transition.from,
            to: transition.to,
            action: transition.action,
            items: [transition],
            laneOffset: 0,
          });
        }
        const buckets = new Map();
        for (const group of groups) {
          const key = group.from === group.to
            ? `self\u0000${group.from}`
            : [String(group.from), String(group.to)].sort().join("\u0000");
          if (!buckets.has(key)) {
            buckets.set(key, []);
          }
          buckets.get(key).push(group);
        }
        for (const bucket of buckets.values()) {
          bucket.sort((left, right) => {
            if (left.from !== right.from) {
              return String(left.from).localeCompare(String(right.from));
            }
            if (left.to !== right.to) {
              return String(left.to).localeCompare(String(right.to));
            }
            return String(left.action).localeCompare(String(right.action));
          });
          const center = (bucket.length - 1) / 2;
          bucket.forEach((group, index) => {
            group.laneOffset = index - center;
          });
        }
        return groups;
      }

      function transitionGroupLabel(group) {
        return group.items
          .map((transition) => `${shortAction(transition.action)}:${formatProbability(transition.prob)}`);
      }

      function isActiveTransition(transition) {
        return runtimeState
          && runtimeState.last_action === transition.action
          && runtimeState.previous_state === transition.from
          && runtimeState.state === transition.to;
      }

      function appendRuntimeEdgeLabelLink(svgElement, anchor, labelPosition, isActive) {
        const dx = labelPosition.x - anchor.x;
        const dy = labelPosition.y - anchor.y;
        if (Math.hypot(dx, dy) < 4) {
          return;
        }
        const pathData = `M ${anchor.x.toFixed(1)} ${anchor.y.toFixed(1)} L ${labelPosition.x.toFixed(1)} ${labelPosition.y.toFixed(1)}`;
        svgElement.appendChild(makeSvg("path", {
          class: isActive ? "runtime-edge-label-link active" : "runtime-edge-label-link",
          d: pathData,
        }));
      }

      function appendRuntimeEdgeLabel(svgElement, x, y, labelLines, fontSize = 6, onDragStart = null) {
        const lines = Array.isArray(labelLines) ? labelLines : [String(labelLines)];
        const maxLength = Math.max(...lines.map((line) => line.length));
        const lineHeight = fontSize + 3;
        const width = Math.max(44, maxLength * fontSize * 0.66 + 14);
        const height = Math.max(fontSize + 12, lines.length * lineHeight + 8);
        const group = makeSvg("g", { class: "runtime-edge-label-group" });
        if (onDragStart) {
          group.addEventListener("pointerdown", onDragStart);
        }
        group.appendChild(makeSvg("rect", {
          class: "runtime-edge-label-bg",
          x: x - width / 2,
          y: y - height / 2,
          width,
          height,
          rx: 3,
        }));
        const label = makeSvg("text", {
          class: "runtime-edge-label",
          x,
          y: y - ((lines.length - 1) * lineHeight) / 2 + fontSize * 0.34,
          style: `font-size: ${fontSize}px;`,
          "text-anchor": "middle",
        });
        lines.forEach((line, index) => {
          const tspan = makeSvg("tspan", {
            x,
            dy: index === 0 ? 0 : lineHeight,
          });
          tspan.textContent = line;
          label.appendChild(tspan);
        });
        group.appendChild(label);
        svgElement.appendChild(group);
      }

      function startRuntimeNodeDrag(event, svgElement, stage, state, position, nodeRadius) {
        event.preventDefault();
        event.stopPropagation();
        const startPoint = runtimeSvgPoint(svgElement, event);
        const offsetX = position.x - startPoint.x;
        const offsetY = position.y - startPoint.y;
        document.body.classList.add("is-dragging-runtime");

        const onMove = (moveEvent) => {
          const activeSvg = stage.querySelector("svg.runtime-svg") || svgElement;
          const point = runtimeSvgPoint(activeSvg, moveEvent);
          const width = Number(activeSvg.viewBox.baseVal.width) || activeSvg.getBoundingClientRect().width;
          const height = Number(activeSvg.viewBox.baseVal.height) || activeSvg.getBoundingClientRect().height;
          const nextX = clamp(point.x + offsetX, nodeRadius + 10, width - nodeRadius - 10);
          const nextY = clamp(point.y + offsetY, nodeRadius + 10, height - nodeRadius - 10);
          runtimeDraggedPositions.set(state, {
            x: nextX / width,
            y: nextY / height,
          });
          renderRuntimeGraph(stage);
        };
        const onUp = () => {
          document.body.classList.remove("is-dragging-runtime");
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
      }

      function startRuntimeLinkedElementDrag(
        event,
        svgElement,
        stage,
        store,
        key,
        position,
        linkedStore = null,
        linkedPosition = null,
      ) {
        event.preventDefault();
        event.stopPropagation();
        const startPoint = runtimeSvgPoint(svgElement, event);
        const offsetX = position.x - startPoint.x;
        const offsetY = position.y - startPoint.y;
        const startPosition = { x: position.x, y: position.y };
        const startLinkedPosition = linkedPosition ? { x: linkedPosition.x, y: linkedPosition.y } : null;
        document.body.classList.add("is-dragging-runtime");

        const onMove = (moveEvent) => {
          const activeSvg = stage.querySelector("svg.runtime-svg") || svgElement;
          const point = runtimeSvgPoint(activeSvg, moveEvent);
          const width = Number(activeSvg.viewBox.baseVal.width) || activeSvg.getBoundingClientRect().width;
          const height = Number(activeSvg.viewBox.baseVal.height) || activeSvg.getBoundingClientRect().height;
          const nextX = clamp(point.x + offsetX, 8, width - 8);
          const nextY = clamp(point.y + offsetY, 8, height - 8);
          store.set(key, {
            x: nextX / width,
            y: nextY / height,
          });
          if (linkedStore && startLinkedPosition) {
            linkedStore.set(key, {
              x: clamp(startLinkedPosition.x + nextX - startPosition.x, 8, width - 8) / width,
              y: clamp(startLinkedPosition.y + nextY - startPosition.y, 8, height - 8) / height,
            });
          }
          renderRuntimeGraph(stage);
        };
        const onUp = () => {
          document.body.classList.remove("is-dragging-runtime");
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
      }

      function runtimeSvgPoint(svgElement, event) {
        const matrix = svgElement.getScreenCTM();
        if (!matrix) {
          return { x: event.offsetX, y: event.offsetY };
        }
        const point = svgElement.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(matrix.inverse());
      }

      function shortAction(action) {
        return String(action).replace(/^move[-_]?/, "");
      }

      function formatProbability(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          return String(value);
        }
        return numeric.toFixed(3).replace(/\.?0+$/, "");
      }

      function markerText(markers, state) {
        if (markers.goals.includes(state)) {
          return "goal";
        }
        if (markers.risks.includes(state)) {
          return "risk";
        }
        if (markers.starts.includes(state)) {
          return "start";
        }
        return "";
      }

      async function refreshRuntime() {
        if (!runtimeEnabled()) {
          renderRuntime();
          return;
        }
        try {
          const response = await fetch(`${data.runtime.endpoint}/state`);
          runtimeState = await response.json();
        } catch (error) {
          runtimeState = {
            state: "connection error",
            observation: error.message,
            reward: 0,
            step: 0,
            done: true,
            trace: [],
          };
        }
        renderRuntime();
      }

      async function stepRuntime() {
        const response = await fetch(`${data.runtime.endpoint}/step`, {
          method: "POST",
        });
        runtimeState = await response.json();
        renderRuntime();
      }

      async function resetRuntime() {
        const response = await fetch(`${data.runtime.endpoint}/reset`, { method: "POST" });
        runtimeState = await response.json();
        renderRuntime();
      }

      document.getElementById("expand-all").addEventListener("click", () => {
        expandAll();
        render();
      });
      document.getElementById("collapse-all").addEventListener("click", () => {
        collapseAll();
        render();
      });
      document.getElementById("expand-depth").addEventListener("click", () => {
        const parsedDepth = Number.parseInt(depthInput.value || "0", 10);
        const depth = Number.isFinite(parsedDepth)
          ? Math.max(0, Math.min(data.maxDepth, parsedDepth))
          : 0;
        depthInput.value = String(depth);
        expandToDepth(depth);
        render();
      });
      document.getElementById("previous-match").addEventListener("click", () => moveMatch(-1));
      document.getElementById("next-match").addEventListener("click", () => moveMatch(1));
      document.getElementById("reveal-matches").addEventListener("click", () => {
        revealMatches();
        render();
      });
      document.getElementById("clear-search").addEventListener("click", () => {
        searchInput.value = "";
        matchIds = [];
        activeMatchIndex = -1;
        searchError = "";
        render();
      });
      document.getElementById("zoom-out").addEventListener("click", () => setZoom(zoom - 0.1));
      document.getElementById("zoom-reset").addEventListener("click", () => setZoom(1));
      document.getElementById("zoom-in").addEventListener("click", () => setZoom(zoom + 0.1));
      searchInput.addEventListener("input", applySearch);
      for (const option of [matchCaseInput, wholeWordInput, regexInput]) {
        option.addEventListener("change", applySearch);
      }
      searchInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          moveMatch(event.shiftKey ? -1 : 1);
        }
      });
      window.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
          event.preventDefault();
          searchInput.focus();
          searchInput.select();
        }
        if (event.key === "Escape" && document.activeElement === searchInput) {
          searchInput.value = "";
          applySearch();
        }
      });
      window.addEventListener("resize", () => {
        if (panelVisible.ast && latestAstLayout) {
          render();
        }
      });

      renderTextOutline();
      setupSplitResizer();
      setupAstCanvasPan();

      if (data.totalNodes <= 60) {
        expandAll();
      } else {
        expandToDepth(2);
      }
      render();
      refreshRuntime();
      setupPanelToggles();
    })();
    """


def _make_visual_node(node: RDDLASTNode, node_id: str) -> _VisualNode:
    """Create display lines and dimensions for one AST node. / 为单个 AST 节点创建显示行和尺寸。"""
    visual_lines = _label_lines(node)
    collapsed_lines = _collapsed_summary_lines(node)
    max_text_length = max([len(node.kind)] + [len(line.text) for line in visual_lines])
    collapsed_text_length = max(
        [len(node.kind)] + [len(line.text) for line in (*visual_lines, *collapsed_lines)]
    )
    width = max(MIN_NODE_WIDTH, PAD_X * 2 + max_text_length * CHAR_WIDTH)
    height = max(MIN_NODE_HEIGHT, LABEL_BASELINE_Y + (len(visual_lines) - 1) * LINE_HEIGHT + BOTTOM_PAD)
    collapsed_width = max(MIN_NODE_WIDTH, PAD_X * 2 + collapsed_text_length * CHAR_WIDTH)
    collapsed_height = max(
        MIN_NODE_HEIGHT,
        LABEL_BASELINE_Y + (len(visual_lines) + len(collapsed_lines) - 1) * LINE_HEIGHT + BOTTOM_PAD,
    )
    return _VisualNode(
        node=node,
        node_id=node_id,
        lines=visual_lines,
        collapsed_lines=collapsed_lines,
        width=width,
        height=height,
        collapsed_width=collapsed_width,
        collapsed_height=collapsed_height,
    )


def _collapsed_summary_lines(node: RDDLASTNode) -> tuple[_VisualLine, ...]:
    """Summarize direct children inside a folded parent. / 在折叠父节点内摘要直接子节点。"""
    if not node.children:
        return ()
    child_lines = [_child_summary_line(child) for child in node.children[:MAX_COLLAPSED_CHILD_LINES]]
    remaining = len(node.children) - len(child_lines)
    if remaining > 0:
        child_lines.append(f"... {remaining} more")
    return tuple(_visual_line(line) for line in ("contains:", *child_lines))


def _child_summary_line(child: RDDLASTNode) -> str:
    """Build one compact child phrase for folded nodes. / 为折叠节点构建一个简洁子节点短语。"""
    label = _compact_summary_text(child.label)
    if child.kind in {"assignment", "statement"}:
        text = label or child.kind
    elif label:
        text = f"{child.kind}: {label}"
    else:
        text = f"{child.kind}: {len(child.children)} children"
    return f"- {_compact_summary_text(text)}"


def _compact_summary_text(value: str) -> str:
    """Collapse whitespace and truncate long folded summaries. / 压缩空白并截断过长的折叠摘要。"""
    compacted = " ".join(value.strip().split())
    if len(compacted) <= MAX_COLLAPSED_LINE_LENGTH:
        return compacted
    return compacted[: MAX_COLLAPSED_LINE_LENGTH - 3].rstrip() + "..."


def _label_lines(node: RDDLASTNode) -> tuple[_VisualLine, ...]:
    """Split a node label into stable semantic visual lines. / 将节点标签切成稳定的语义显示行。"""
    label = node.label.strip()
    if node.kind == "file":
        return tuple(_visual_line(part, prefer_path=True) for part in _split_path_label(label))
    if node.kind == "assignment":
        return tuple(_visual_line(line) for line in _assignment_lines(label))
    if node.kind == "statement":
        return tuple(_visual_line(line) for line in _statement_lines(label))
    return (_visual_line(label),)


def _split_path_label(label: str) -> tuple[str, ...]:
    """Split a file path into directory and file lines. / 将文件路径拆成目录行和文件名行。"""
    path = Path(label)
    if path.parent == Path("."):
        return (label,)
    parent = path.parent.as_posix().rstrip("/") + "/"
    return (parent, path.name)


def _assignment_lines(label: str) -> tuple[str, ...]:
    """Split assignment labels around the equals sign. / 围绕等号拆分赋值标签。"""
    key, separator, value = label.partition("=")
    if not separator:
        return (label,)
    key = key.strip()
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return (f"{key} =", _spaced_braces(value))
    return (f"{key} = {value}",)


def _statement_lines(label: str) -> tuple[str, ...]:
    """Split statement labels around type or value separators. / 围绕类型或取值分隔符拆分语句标签。"""
    if " : " in label:
        left, _, right = label.partition(" : ")
        right = right.strip()
        if right.startswith("{") and right.endswith("}"):
            return (f"{left.strip()} :", _spaced_braces(right))
        return (f"{left.strip()} : {right}",)
    if " = " in label:
        left, _, right = label.partition(" = ")
        return (f"{left.strip()} =", right.strip())
    return (label,)


def _spaced_braces(value: str) -> str:
    """Normalize brace contents for readable display. / 规范化花括号内容以便阅读。"""
    inner = value.strip()[1:-1].strip()
    if not inner:
        return "{ }"
    normalized = ", ".join(part.strip() for part in inner.split(","))
    return f"{{ {normalized} }}"


def _visual_line(value: str, prefer_path: bool = False) -> _VisualLine:
    """Tokenize one display line for syntax highlighting. / 将一行显示文本切成用于语法高亮的 token。"""
    if prefer_path:
        return _VisualLine((_TokenSpan(value, "tok-path"),))
    return _VisualLine(tuple(_tokenize(value)))


def _tokenize(value: str) -> list[_TokenSpan]:
    """Tokenize display text into classified spans. / 将显示文本切分为带分类的片段。"""
    return [_TokenSpan(token, _classify_token(token)) for token in TOKEN_PATTERN.findall(value)]


def _classify_token(token: str) -> str:
    """Map one token to a syntax-highlighting CSS class. / 将单个 token 映射为语法高亮 CSS 类。"""
    if token.isspace():
        return "tok-space"
    if token in {"=", ":"}:
        return "tok-operator"
    if token in {"{", "}", "(", ")", "[", "]", ",", ";", ".", "/"}:
        return "tok-punct"
    if token.replace(".", "", 1).isdigit():
        return "tok-number"
    lower = token.lower()
    if lower in {"true", "false"}:
        return "tok-literal"
    if lower in RDDL_KEYWORDS:
        return "tok-keyword"
    if token.startswith("?"):
        return "tok-parameter"
    return "tok-symbol"


def _safe_class(value: str) -> str:
    """Convert an arbitrary value into a safe CSS class suffix. / 将任意值转换为安全的 CSS 类后缀。"""
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "node"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for HTML visualization. / 构建用于 HTML 可视化的命令行 parser。"""
    parser = argparse.ArgumentParser(description="Serve RDDL AST as an interactive HTML visualizer.")
    parser.add_argument("domain", help="RDDL domain file to visualize")
    parser.add_argument("instance", help="RDDL instance file to visualize")
    parser.add_argument(
        "--simulator",
        choices=("darp", "rddlgym", "pyrddlgym"),
        help="runtime simulator; use darp for the internal state-machine panel",
    )
    parser.add_argument(
        "--with-simulator",
        nargs="?",
        const="darp",
        choices=("darp", "rddlgym", "pyrddlgym"),
        dest="simulator",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-open", action="store_true", help="serve without opening a browser")
    parser.add_argument(
        "--frontend",
        default="darp",
        choices=available_frontends(),
        help="frontend used for compilation in simulator mode",
    )
    parser.add_argument("--host", default="127.0.0.1", help="host for simulator mode")
    parser.add_argument("--port", type=int, default=0, help="port for simulator mode; 0 picks a free port")
    parser.add_argument("--seed", type=int, default=0, help="local simulator random seed")
    return parser


def serve_visualizer(
    *,
    domain: str | Path,
    instance: str | Path,
    simulator: str | None,
    frontend: str,
    host: str,
    port: int,
    seed: int = 0,
    open_browser: bool = True,
) -> int:
    """Serve the RDDL visualizer from the top-level DARP CLI. / 从 DARP 顶层 CLI 启动 RDDL 可视化服务。"""
    ast = BasicRDDLParser().parse_files(domain, instance)
    if simulator == "darp":
        loaded = RDDLLoader(frontend).load(domain, instance)
        problem = RDDLCompiler().compile(loaded)
        markers = _runtime_markers(ast)
        return _serve_runtime_visualizer(
            ast=ast,
            problem=problem,
            markers=markers,
            host=host,
            port=port,
            seed=seed,
            open_browser=open_browser,
        )
    if simulator in {"rddlgym", "pyrddlgym"}:
        loaded = RDDLLoader("pyrddlgym").load(domain, instance)
        runtime = _external_simulator_payload("rddlgym", loaded)
        print("pyRDDLGym simulator loaded; DARP state-machine panel is hidden.", flush=True)
        return _serve_html_visualizer(
            ast=ast,
            title="RDDL AST + pyRDDLGym",
            runtime=runtime,
            host=host,
            port=port,
            open_browser=open_browser,
        )
    return _serve_html_visualizer(
        ast=ast,
        title="RDDL AST",
        runtime=None,
        host=host,
        port=port,
        open_browser=open_browser,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the standalone visualizer command. / 运行独立可视化命令。"""
    args = build_parser().parse_args(argv)
    return serve_visualizer(
        domain=args.domain,
        instance=args.instance,
        simulator=args.simulator,
        frontend=args.frontend,
        host=args.host,
        port=args.port,
        seed=args.seed,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    raise SystemExit(main())
