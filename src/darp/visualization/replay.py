"""Generate side-by-side HTML replays for solver comparison results.

The visualizer reads the long-format `runs.csv` produced by the experiment
runner. It uses pyRDDLGym through DARP's loader to extract grid/navigation
structure when possible, then embeds all data into a standalone HTML file.
"""

# TODO(phase-9.4): Add richer domain-specific renderers beyond grid/navigation-style layouts.

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from darp.visualization.graph import graph_from_rows, graph_with_replay_states
from darp.visualization.traces import (
    enrich_rows_from_darp_traces,
    reachable_bellman_replay_rows,
    run_payload,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment visualizer CLI parser."""
    parser = argparse.ArgumentParser(description="Generate a side-by-side solver replay HTML file.")
    parser.add_argument("runs_csv", help="long-format runs.csv from an experiment directory")
    parser.add_argument("--output", help="HTML output path; defaults to replay.html beside runs.csv")
    parser.add_argument("--title", help="optional page title")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone visualizer generator."""
    args = build_parser().parse_args(argv)
    runs_csv = Path(args.runs_csv).resolve()
    output = Path(args.output).resolve() if args.output else runs_csv.with_name("replay.html")
    build_replay_html(runs_csv, output, title=args.title)
    print(f"Wrote replay visualizer: {output}")
    return 0


def build_replay_html(runs_csv: str | Path, output: str | Path, *, title: str | None = None) -> Path:
    """Build a standalone replay HTML file from one experiment `runs.csv`."""
    runs_path = Path(runs_csv).resolve()
    rows = _read_rows(runs_path)
    if not rows:
        raise ValueError(f"No experiment rows found in {runs_path}.")
    rows = enrich_rows_from_darp_traces(rows, runs_path.parent)
    graph = graph_from_rows(rows)
    replay_rows = reachable_bellman_replay_rows(rows)
    runs = [run_payload(row, graph) for row in replay_rows]
    graph = graph_with_replay_states(graph, runs)
    payload = {
        "title": title or rows[0].get("scenario") or runs_path.parent.name,
        "graph": graph,
        "runs": runs,
    }
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(payload), encoding="utf-8")
    return output_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read long-format experiment rows."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _html_document(payload: dict[str, Any]) -> str:
    """Return a standalone HTML replay document."""
    data = json.dumps(payload, ensure_ascii=False)
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} Replay</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #c7d4e3;
      --ink: #15283c;
      --muted: #607086;
      --blue: #087cc1;
      --green: #1c9b62;
      --red: #d64c4c;
      --amber: #c2821c;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .toolbar, .panel-toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button, select {{
      border: 1px solid #abc0d5;
      border-radius: 6px;
      padding: 6px 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }}
    button:hover {{
      border-color: var(--blue);
      color: var(--blue);
      cursor: pointer;
    }}
    main {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      padding: 14px;
      height: calc(100vh - 70px);
      box-sizing: border-box;
    }}
    section {{
      min-width: 0;
      display: grid;
      grid-template-rows: auto auto minmax(420px, 1fr) auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .metric {{
      border: 1px solid #d6e1ec;
      border-radius: 6px;
      padding: 6px 8px;
      background: #f8fbff;
      min-width: 0;
    }}
    .metric b {{
      display: block;
      font-size: 10px;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .metric span {{
      word-break: break-word;
    }}
    svg {{
      width: 100%;
      height: 100%;
      min-height: 420px;
      background: linear-gradient(#ffffff, #f9fbfd);
    }}
    .edge {{
      stroke: #9db2c8;
      stroke-width: 2;
      fill: none;
      opacity: .85;
    }}
    .edge.active {{
      stroke: var(--amber);
      stroke-width: 4;
      opacity: 1;
    }}
    .edge.lost-edge {{
      stroke-dasharray: 7 6;
      stroke: var(--red);
    }}
    .node circle {{
      fill: #f7fbff;
      stroke: #b4c6d8;
      stroke-width: 2;
    }}
    .node.goal circle {{
      fill: #e8f8ef;
      stroke: var(--green);
    }}
    .node.start circle {{
      fill: #e1f3ff;
      stroke: var(--blue);
      stroke-width: 3;
    }}
    .node.risk circle {{
      fill: #fff2f2;
      stroke: var(--red);
    }}
    .node.lost circle {{
      fill: #ffe9e9;
      stroke: var(--red);
      stroke-width: 3;
    }}
    .node.obstacle circle {{
      stroke: var(--red);
      stroke-width: 3;
    }}
    .node.active circle {{
      fill: #e1f3ff;
      stroke: var(--blue);
      stroke-width: 4;
    }}
    .node text {{
      text-anchor: middle;
      dominant-baseline: middle;
      font-weight: 700;
      font-size: 12px;
    }}
    .edge-label {{
      fill: var(--muted);
      font-size: 10px;
      text-anchor: middle;
      paint-order: stroke;
      stroke: white;
      stroke-width: 3px;
    }}
    .node-badge {{
      font-size: 8px;
      font-weight: 800;
      letter-spacing: .04em;
      text-anchor: middle;
      fill: var(--muted);
    }}
    .obstacle-marker {{
      fill: var(--red);
      stroke: white;
      stroke-width: 2;
    }}
    .obstacle-label {{
      fill: white;
      font-size: 8px;
      font-weight: 900;
      text-anchor: middle;
      dominant-baseline: middle;
    }}
    .trace {{
      border-top: 1px solid var(--line);
      padding: 10px 12px;
      display: grid;
      gap: 6px;
      max-height: 190px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      background: #fbfdff;
    }}
    .step-line.active {{
      color: var(--blue);
      font-weight: 700;
    }}
    .warning {{
      margin: 12px;
      color: #8a5b00;
    }}
    @media (max-width: 1000px) {{
      main {{ grid-template-columns: 1fr; }}
      main {{ height: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{title}</h1>
      <div id="graph-note"></div>
    </div>
    <div class="toolbar">
      <button id="both-reset">Reset both</button>
      <button id="both-prev">Previous both</button>
      <button id="both-next">Next both</button>
      <button id="both-play">Play both</button>
    </div>
  </header>
  <main>
    <section id="left-panel">
      <div class="panel-head">
        <strong>DARP</strong>
        <select id="left-select"></select>
        <div class="panel-toolbar">
          <button data-side="left" data-action="reset">Reset</button>
          <button data-side="left" data-action="prev">Previous</button>
          <button data-side="left" data-action="next">Next</button>
          <button data-side="left" data-action="play">Play</button>
        </div>
      </div>
      <div class="meta" id="left-meta"></div>
      <svg id="left-svg" viewBox="0 0 800 520" role="img"></svg>
      <div class="trace" id="left-trace"></div>
    </section>
    <section id="right-panel">
      <div class="panel-head">
        <strong>Baseline</strong>
        <select id="right-select"></select>
        <div class="panel-toolbar">
          <button data-side="right" data-action="reset">Reset</button>
          <button data-side="right" data-action="prev">Previous</button>
          <button data-side="right" data-action="next">Next</button>
          <button data-side="right" data-action="play">Play</button>
        </div>
      </div>
      <div class="meta" id="right-meta"></div>
      <svg id="right-svg" viewBox="0 0 800 520" role="img"></svg>
      <div class="trace" id="right-trace"></div>
    </section>
  </main>
  <script>
    const DATA = {data};
    const state = {{
      left: {{index: 0, run: null, timer: null}},
      right: {{index: 0, run: null, timer: null}},
    }};

    const bySystem = {{
      left: DATA.runs.filter(r => r.system === 'DARP'),
      right: DATA.runs.filter(r => r.system !== 'DARP'),
    }};

    function setup() {{
      document.getElementById('graph-note').textContent = DATA.graph.note ? `Graph note: ${{DATA.graph.note}}` : `${{DATA.graph.kind}}`;
      setupSelect('left');
      setupSelect('right');
      document.getElementById('both-reset').onclick = () => {{ reset('left'); reset('right'); }};
      document.getElementById('both-prev').onclick = () => {{ previous('left'); previous('right'); }};
      document.getElementById('both-next').onclick = () => {{ next('left'); next('right'); }};
      document.getElementById('both-play').onclick = toggleBoth;
      document.querySelectorAll('button[data-side]').forEach(button => {{
        button.onclick = () => {{
          const side = button.dataset.side;
          const action = button.dataset.action;
          if (action === 'reset') reset(side);
          if (action === 'prev') previous(side);
          if (action === 'next') next(side);
          if (action === 'play') togglePlay(side);
        }};
      }});
      render('left');
      render('right');
      window.addEventListener('resize', () => {{
        render('left');
        render('right');
      }});
    }}

    function setupSelect(side) {{
      const select = document.getElementById(`${{side}}-select`);
      const runs = bySystem[side];
      select.innerHTML = '';
      if (!runs.length) {{
        const option = document.createElement('option');
        option.textContent = side === 'left' ? 'No DARP run' : 'No baseline run';
        select.appendChild(option);
        state[side].run = null;
        return;
      }}
      runs.forEach((run, index) => {{
        const option = document.createElement('option');
        option.value = index;
        option.textContent = run.variant || run.system;
        select.appendChild(option);
      }});
      const preferred = side === 'left'
        ? Math.max(0, runs.findIndex(run => run.heuristic === 'reachable-bellman' || (run.variant || '').includes('reachable-bellman')))
        : 0;
      select.value = String(preferred);
      state[side].run = runs[preferred];
      select.onchange = () => {{
        stop(side);
        state[side].run = runs[Number(select.value)];
        state[side].index = 0;
        render(side);
      }};
    }}

    function reset(side) {{ state[side].index = 0; render(side); }}
    function previous(side) {{ state[side].index = Math.max(0, state[side].index - 1); render(side); }}
    function next(side) {{
      const run = state[side].run;
      const frames = framesOf(run);
      const maxIndex = Math.max((frames.length || 1) - 1, 0);
      state[side].index = Math.min(maxIndex, state[side].index + 1);
      render(side);
    }}
    function toggleBoth() {{
      const anyPlaying = state.left.timer || state.right.timer;
      if (anyPlaying) {{ stop('left'); stop('right'); return; }}
      togglePlay('left');
      togglePlay('right');
    }}
    function togglePlay(side) {{
      if (state[side].timer) {{ stop(side); return; }}
      state[side].timer = setInterval(() => next(side), 900);
    }}
    function stop(side) {{
      if (state[side].timer) clearInterval(state[side].timer);
      state[side].timer = null;
    }}

    function framesOf(run) {{
      if (!run) return [];
      if (Array.isArray(run.frames) && run.frames.length) return run.frames;
      const states = Array.isArray(run.states) ? run.states : [];
      const actions = Array.isArray(run.actions) ? run.actions : [];
      const obstacles = Array.isArray(run.obstacles) ? run.obstacles : [];
      const total = Math.max(states.length, obstacles.length, actions.length + 1);
      const frames = [];
      for (let i = 0; i < total; i++) {{
        frames.push({{
          step: i,
          agent: states[i] || states[states.length - 1] || '',
          obstacles: obstacles[i] || [],
          action: actions[i] || '',
          next_agent: states[i + 1] || '',
          reward: '',
          raw_state: '',
        }});
      }}
      return frames;
    }}

    function frameAt(run, index) {{
      const frames = framesOf(run);
      if (!frames.length) return {{}};
      return frames[Math.max(0, Math.min(index, frames.length - 1))] || {{}};
    }}

    function render(side) {{
      const run = state[side].run;
      renderMeta(side, run);
      renderSvg(side, run);
      renderTrace(side, run);
    }}

    function renderMeta(side, run) {{
      const target = document.getElementById(`${{side}}-meta`);
      if (!run) {{
        target.innerHTML = '<div class="metric"><b>Status</b><span>No run data</span></div>';
        return;
      }}
      const t = state[side].index;
      const frames = framesOf(run);
      const frame = frameAt(run, t);
      const nextFrame = frames[t + 1] || {{}};
      const action = frame.action || run.actions[t] || '(none)';
      const current = frame.agent || '(unknown)';
      const nextState = frame.next_agent || nextFrame.agent || '';
      const obstacles = frame.obstacles || [];
      target.innerHTML = [
        metric('Step', `${{t}}/${{Math.max(run.actions.length, frames.length - 1)}}`),
        metric('State', current),
        metric('Action', action),
        metric('Next', nextState || '(unknown)'),
        metric('Obstacles', obstacles.length ? obstacles.join(', ') : '(none)'),
        metric('Reward', frame.reward || run.reward || ''),
        metric('Decision ms', run.decision_ms || ''),
        metric('Runtime s', run.runtime_s || ''),
        metric('Planner', [run.planner, run.heuristic].filter(Boolean).join(' / ')),
      ].join('');
    }}

    function metric(label, value) {{
      return `<div class="metric"><b>${{escapeHtml(label)}}</b><span>${{escapeHtml(String(value))}}</span></div>`;
    }}

    function renderSvg(side, run) {{
      const svg = document.getElementById(`${{side}}-svg`);
      const nodes = DATA.graph.nodes || [];
      if (!nodes.length) {{
        svg.innerHTML = `<text x="24" y="36" class="warning">${{escapeHtml(DATA.graph.note || 'No graph available')}}</text>`;
        return;
      }}
      const layoutResult = layout(nodes, svg);
      const positions = layoutResult.positions;
      svg.setAttribute('viewBox', `${{layoutResult.minX}} ${{layoutResult.minY}} ${{layoutResult.width}} ${{layoutResult.height}}`);
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      const frame = frameAt(run, state[side].index);
      const previousFrame = frameAt(run, Math.max(0, state[side].index - 1));
      const current = frame.agent || '';
      const previousState = previousFrame.agent || '';
      const activeAction = previousFrame.action || run?.actions?.[Math.max(0, state[side].index - 1)] || '';
      const obstacleSet = new Set(frame.obstacles || []);
      const parts = [];
      const markerId = `${{side}}-arrow`;
      parts.push(`<defs><marker id="${{markerId}}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#9db2c8"/></marker></defs>`);
      for (const edge of DATA.graph.edges || []) {{
        const source = positions[edge.source];
        const target = positions[edge.target];
        if (!source || !target) continue;
        const active = previousState === edge.source && current === edge.target && (!activeAction || activeAction === edge.action);
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        parts.push(`<line class="edge ${{active ? 'active' : ''}}" x1="${{source.x}}" y1="${{source.y}}" x2="${{target.x}}" y2="${{target.y}}" marker-end="url(#${{markerId}})"/>`);
        if (active || DATA.graph.nodes.length <= 9) {{
          parts.push(`<text class="edge-label" x="${{midX}}" y="${{midY - 5}}">${{escapeHtml(edge.label || edge.action || '')}}</text>`);
        }}
      }}
      if (current === 'lost' && previousState && positions[previousState] && positions[current]) {{
        const source = positions[previousState];
        const target = positions[current];
        parts.push(`<line class="edge active lost-edge" x1="${{source.x}}" y1="${{source.y}}" x2="${{target.x}}" y2="${{target.y}}" marker-end="url(#${{markerId}})"/>`);
        parts.push(`<text class="edge-label" x="${{(source.x + target.x) / 2}}" y="${{(source.y + target.y) / 2 - 8}}">lost</text>`);
      }}
      for (const node of nodes) {{
        const point = positions[node.id];
        const classes = ['node'];
        if (node.start) classes.push('start');
        if (node.goal) classes.push('goal');
        if (node.risk) classes.push('risk');
        if (node.lost) classes.push('lost');
        if (obstacleSet.has(node.id)) classes.push('obstacle');
        if (node.id === current) classes.push('active');
        const sub = node.probability !== null && node.probability !== undefined ? `p=${{Number(node.probability).toFixed(2)}}` : '';
        const badge = node.lost ? 'LOST' : node.goal ? 'GOAL' : node.start ? 'START' : '';
        parts.push(`<g class="${{classes.join(' ')}}" transform="translate(${{point.x}},${{point.y}})">`);
        parts.push('<circle r="25"></circle>');
        parts.push(`<text y="${{sub ? -5 : 0}}">${{escapeHtml(node.label)}}</text>`);
        if (sub) parts.push(`<text y="14" style="font-size:10px;font-weight:600;fill:#607086">${{sub}}</text>`);
        if (badge) parts.push(`<text class="node-badge" y="39">${{badge}}</text>`);
        if (obstacleSet.has(node.id)) {{
          parts.push('<rect class="obstacle-marker" x="12" y="-31" width="18" height="18" rx="4"></rect>');
          parts.push('<text class="obstacle-label" x="21" y="-22">O</text>');
        }}
        parts.push('</g>');
      }}
      if (current && !positions[current]) {{
        parts.push(`<text x="24" y="500" class="warning">Current state not on graph: ${{escapeHtml(current)}}</text>`);
      }}
      svg.innerHTML = parts.join('');
    }}

    function layout(nodes, svg) {{
      const minX = Math.min(...nodes.map(n => n.x));
      const maxX = Math.max(...nodes.map(n => n.x));
      const minY = Math.min(...nodes.map(n => n.y));
      const maxY = Math.max(...nodes.map(n => n.y));
      const gridWidth = Math.max(1, maxX - minX);
      const gridHeight = Math.max(1, maxY - minY);
      const rect = svg.getBoundingClientRect();
      const availableWidth = Math.max(520, rect.width || 520);
      const availableHeight = Math.max(420, rect.height || 420);
      const margin = 80;
      const fitStep = Math.min(
        (availableWidth - margin * 2) / gridWidth,
        (availableHeight - margin * 2) / gridHeight
      );
      const step = Math.max(90, Math.min(190, fitStep));
      const viewWidth = gridWidth * step + margin * 2;
      const viewHeight = gridHeight * step + margin * 2;
      const positions = {{}};
      for (const node of nodes) {{
        positions[node.id] = {{
          x: margin + (node.x - minX) * step,
          y: margin + (node.y - minY) * step,
        }};
      }}
      return {{positions, minX: 0, minY: 0, width: viewWidth, height: viewHeight}};
    }}

    function renderTrace(side, run) {{
      const target = document.getElementById(`${{side}}-trace`);
      if (!run) {{ target.textContent = 'No run data'; return; }}
      const rows = [];
      const frames = framesOf(run);
      const total = Math.max(run.actions.length, frames.length - 1);
      for (let i = 0; i < total; i++) {{
        const active = i === state[side].index || i === state[side].index - 1;
        const frame = frameAt(run, i);
        const nextFrame = frames[i + 1] || {{}};
        const obstacles = frame.obstacles || [];
        const obstacleText = obstacles.length ? ` obs=[${{obstacles.join(',')}}]` : '';
        const action = frame.action || run.actions[i] || 'noop';
        const nextState = frame.next_agent || nextFrame.agent || '?';
        rows.push(`<div class="step-line ${{active ? 'active' : ''}}">t=${{i}} ${{escapeHtml(frame.agent || '?')}} --${{escapeHtml(action)}}--> ${{escapeHtml(nextState)}}${{escapeHtml(obstacleText)}}</div>`);
      }}
      if (!rows.length) rows.push('<div class="step-line">No action/state sequence available.</div>');
      target.innerHTML = rows.join('');
    }}

    function escapeHtml(value) {{
      return value.replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}

    setup();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
