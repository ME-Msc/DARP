"""Standalone HTML visualizer for the built-in RDDL AST."""

# TODO(parser): Add source spans and expression-level nodes once the parser
# supports full RDDL semantics.

from __future__ import annotations

import argparse
import html
import json
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from darp.rddl.ast import RDDLASTNode
from darp.rddl.basic_parser import BasicRDDLParser

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

RDDL_KEYWORDS = {
    "action-fluent",
    "bool",
    "cpfs",
    "default",
    "derived-fluent",
    "discount",
    "domain",
    "false",
    "horizon",
    "init-state",
    "instance",
    "interm-fluent",
    "max-nondef-actions",
    "non-fluent",
    "object",
    "objects",
    "observ-fluent",
    "pvariables",
    "requirements",
    "reward",
    "reward-deterministic",
    "state-fluent",
    "true",
    "types",
}
TOKEN_PATTERN = re.compile(r"\s+|\d+(?:\.\d+)?|[A-Za-z_?][A-Za-z0-9_?'-]*|[{}()[\]:=,;./]|.")


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
    width: float
    height: float


def render_html(ast: RDDLASTNode, title: str = "RDDL AST") -> str:
    """Render an AST as a self-contained interactive HTML page. / 将 AST 渲染为独立交互式 HTML 页面。"""

    data_json = _json_for_html(_build_visual_data(ast))
    escaped_title = html.escape(title)
    summary = html.escape(ast.summary())
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
    <h1>{escaped_title}</h1>
    <div class="summary">{summary}</div>
    <div class="toolbar" role="toolbar" aria-label="AST controls">
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
  </header>
  <main>
    <div class="canvas" id="canvas">
      <svg id="ast-svg" role="img" aria-label="{escaped_title}"></svg>
    </div>
  </main>
  <script id="ast-data" type="application/json">{data_json}</script>
  <script>{_script()}</script>
</body>
</html>
"""


def write_html(path: str | Path, ast: RDDLASTNode, title: str = "RDDL AST") -> Path:
    """Write the rendered AST HTML to disk. / 将渲染后的 AST HTML 写入磁盘。"""
    output = Path(path)
    output.write_text(render_html(ast, title=title), encoding="utf-8")
    return output


def _build_visual_data(ast: RDDLASTNode) -> dict[str, object]:
    """Convert AST nodes into JSON-ready visual metadata. / 将 AST 节点转换为可写入 JSON 的可视化元数据。"""
    nodes: list[dict[str, object]] = []
    max_depth = 0

    def visit(node: RDDLASTNode, depth: int, parent_id: str | None) -> str:
        """Append one visual node and visit its children. / 添加一个可视节点并递归访问子节点。"""
        nonlocal max_depth
        node_id = f"node{len(nodes)}"
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
            "children": [],
            "lines": [
                [{"text": token.text, "className": token.css_class} for token in line.tokens]
                for line in visual.lines
            ],
        }
        nodes.append(payload)
        max_depth = max(max_depth, depth)
        children = payload["children"]
        assert isinstance(children, list)
        for child in node.children:
            children.append(visit(child, depth + 1, node_id))
        return node_id

    root_id = visit(ast, 0, None)
    return {
        "rootId": root_id,
        "nodes": nodes,
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
    }


def _search_text(node: RDDLASTNode, visual: _VisualNode, depth: int) -> str:
    """Build searchable text for one AST node. / 为单个 AST 节点构建可搜索文本。"""
    rendered = " ".join(line.text for line in visual.lines)
    return f"{node.kind} {node.label} {rendered} depth:{depth}"


def _json_for_html(payload: dict[str, object]) -> str:
    """Encode JSON safely for inline HTML. / 将 JSON 安全编码到内联 HTML 中。"""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


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
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 18px 28px 14px;
      border-bottom: 1px solid #d7e1ea;
      background: rgba(255, 255, 255, 0.96);
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(8px);
    }
    h1 {
      font-size: 22px;
      margin: 0 0 6px;
      letter-spacing: 0;
    }
    .summary {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
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
    main {
      padding: 24px;
      overflow: auto;
    }
    .canvas {
      width: max-content;
      min-width: 100%;
      padding: 12px;
    }
    svg {
      display: block;
      background: #ffffff;
      border: 1px solid #d7e1ea;
      border-radius: 8px;
      box-shadow: 0 14px 28px rgba(42, 62, 82, 0.10);
    }
    .edge {
      fill: none;
      stroke: var(--line);
      stroke-width: 2;
      stroke-linecap: round;
    }
    .node {
      cursor: default;
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
    .tok-keyword { fill: var(--keyword); font-weight: 700; }
    .tok-symbol { fill: var(--symbol); }
    .tok-number { fill: var(--number); }
    .tok-literal { fill: var(--literal); font-weight: 700; }
    .tok-parameter { fill: var(--parameter); }
    .tok-operator { fill: var(--operator); font-weight: 700; }
    .tok-punct { fill: #64748b; }
    .tok-path { fill: #334155; }
    .tok-space { fill: currentColor; }
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
    """


def _script() -> str:
    """Return JavaScript for layout, folding, search, and zoom. / 返回布局、折叠、搜索和缩放所需的 JavaScript。"""
    return r"""
    (() => {
      const SVG_NS = "http://www.w3.org/2000/svg";
      const data = JSON.parse(document.getElementById("ast-data").textContent);
      const svg = document.getElementById("ast-svg");
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

      depthInput.max = String(data.maxDepth);
      depthInput.value = String(Math.min(2, data.maxDepth));

      function makeSvg(name, attrs = {}) {
        const element = document.createElementNS(SVG_NS, name);
        for (const [key, value] of Object.entries(attrs)) {
          element.setAttribute(key, String(value));
        }
        return element;
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

      function measureVisibleTree(id, depth, levelHeights) {
        const node = nodes.get(id);
        levelHeights.set(depth, Math.max(levelHeights.get(depth) || 0, node.height));
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
          subtreeWidth: Math.max(node.width, childWidth),
        };
      }

      function placeVisibleTree(branch, left, levelY, placements, edges) {
        const node = nodes.get(branch.id);
        const centerX = left + branch.subtreeWidth / 2;
        placements.push({
          id: branch.id,
          x: centerX - node.width / 2,
          y: levelY.get(branch.depth),
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
        const placementById = new Map(layout.placements.map((placement) => [placement.id, placement]));
        const visibleIds = new Set(layout.placements.map((placement) => placement.id));
        svg.replaceChildren();
        svg.setAttribute("viewBox", `0 0 ${Math.ceil(layout.width)} ${Math.ceil(layout.height)}`);
        svg.setAttribute("width", String(Math.ceil(layout.width * zoom)));
        svg.setAttribute("height", String(Math.ceil(layout.height * zoom)));
        svg.style.width = `${Math.ceil(layout.width * zoom)}px`;
        svg.style.height = `${Math.ceil(layout.height * zoom)}px`;

        const edgeLayer = makeSvg("g", { class: "edges" });
        for (const [parentId, childId] of layout.edges) {
          const parent = nodes.get(parentId);
          const child = nodes.get(childId);
          const parentPlace = placementById.get(parentId);
          const childPlace = placementById.get(childId);
          const x1 = parentPlace.x + parent.width / 2;
          const y1 = parentPlace.y + parent.height;
          const x2 = childPlace.x + child.width / 2;
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
      }

      function renderNode(node, placement) {
        const classes = ["node", node.cssClass];
        const isMatch = matchIds.includes(node.id);
        const isActiveMatch = activeMatchIndex >= 0 && matchIds[activeMatchIndex] === node.id;
        const isSelected = selectedId === node.id;
        const isFocusPulse = focusPulseId === node.id;
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
            width: node.width + 14,
            height: node.height + 14,
            rx: 13,
          }));
        }
        group.appendChild(makeSvg("rect", {
          class: "node-card",
          x: 0,
          y: 0,
          width: node.width,
          height: node.height,
          rx: 8,
        }));
        if (isMatch) {
          group.appendChild(makeSvg("rect", {
            class: isActiveMatch ? "match-bar active-match-bar" : "match-bar",
            x: 10,
            y: 5,
            width: Math.max(24, node.width - 20),
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

        node.lines.forEach((line, lineIndex) => {
          const text = makeSvg("text", {
            class: "label",
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
            transform: `translate(${node.width - 28} 8)`,
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
          selectedId = node.id;
          render();
        });
        group.addEventListener("dblclick", () => {
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
      }

      function scrollNodeIntoView(id) {
        window.requestAnimationFrame(() => {
          const target = document.getElementById(`ast-node-${id}`);
          if (target && target.scrollIntoView) {
            target.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
          }
        });
      }

      function setZoom(nextZoom) {
        zoom = Math.min(1.8, Math.max(0.35, nextZoom));
        document.getElementById("zoom-reset").textContent = `${Math.round(zoom * 100)}%`;
        render();
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

      if (data.totalNodes <= 60) {
        expandAll();
      } else {
        expandToDepth(2);
      }
      render();
    })();
    """


def _make_visual_node(node: RDDLASTNode, node_id: str) -> _VisualNode:
    """Create display lines and dimensions for one AST node. / 为单个 AST 节点创建显示行和尺寸。"""
    visual_lines = _label_lines(node)
    max_text_length = max([len(node.kind)] + [len(line.text) for line in visual_lines])
    width = max(MIN_NODE_WIDTH, PAD_X * 2 + max_text_length * CHAR_WIDTH)
    height = max(MIN_NODE_HEIGHT, LABEL_BASELINE_Y + (len(visual_lines) - 1) * LINE_HEIGHT + BOTTOM_PAD)
    return _VisualNode(
        node=node,
        node_id=node_id,
        lines=visual_lines,
        width=width,
        height=height,
    )


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
    parser = argparse.ArgumentParser(description="Render RDDL AST as a standalone HTML tree.")
    parser.add_argument("paths", nargs="+", help="RDDL domain/instance files to visualize")
    parser.add_argument("--output", default="rddl_ast.html", help="HTML output path")
    parser.add_argument("--open", action="store_true", help="open the generated HTML in a browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone visualizer command. / 运行独立可视化命令。"""
    args = build_parser().parse_args(argv)
    ast = BasicRDDLParser().parse_files(*args.paths)
    output = write_html(args.output, ast, title="RDDL AST")
    print(f"RDDL AST visualizer written to {output}")
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
