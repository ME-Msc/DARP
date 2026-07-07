"""Trace normalization helpers for solver replay visualizations."""

# TODO(visualization): Add RAO*/DARP domain-specific frame decoders as
# benchmark adapters mature.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from darp.visualization.graph import (
    graph_start_state,
    grounding_args,
    graph_next_state,
)
from darp.visualization.schema import ReplayFrame, frame_from_mapping


def enrich_rows_from_darp_traces(rows: list[dict[str, str]], experiment_dir: Path) -> list[dict[str, str]]:
    """Fill missing DARP robot/obstacle traces from adjacent JSON traces when available."""
    enriched = []
    for row in rows:
        row = dict(row)
        if row.get("system") == "DARP":
            frames = _darp_trace_frames(experiment_dir, row)
            if frames:
                row["frames"] = json.dumps([frame.to_dict() for frame in frames])
                row["states"] = "->".join(frame.agent for frame in frames if frame.agent)
                row["obstacles"] = json.dumps([list(frame.obstacles) for frame in frames])
        enriched.append(row)
    return enriched


def reachable_bellman_replay_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep reachable-bellman rows when present, otherwise preserve all rows."""
    replay_rows = []
    has_reachable = any(
        row.get("heuristic") == "reachable-bellman" or "reachable-bellman" in row.get("variant", "")
        for row in rows
    )
    if not has_reachable:
        return rows
    for row in rows:
        if row.get("system") != "DARP":
            replay_rows.append(row)
            continue
        heuristic = row.get("heuristic", "")
        variant = row.get("variant", "")
        if heuristic == "reachable-bellman" or "reachable-bellman" in variant:
            replay_rows.append(row)
    return replay_rows


def run_payload(row: dict[str, str], graph: dict[str, Any]) -> dict[str, Any]:
    """Convert one CSV row into a browser-friendly replay record."""
    actions = split_sequence(row.get("actions", ""), separator=",")
    frames = _frame_sequence(row.get("frames", ""))
    if frames:
        if not actions:
            actions = [frame.action for frame in frames if frame.action]
        states = [frame.agent for frame in frames]
        obstacles = [list(frame.obstacles) for frame in frames]
        raw_states = [frame.raw_state for frame in frames]
    else:
        raw_states = split_sequence(row.get("states", ""), separator="->")
        obstacles = _obstacle_sequence(row.get("obstacles", ""))
        states = _complete_state_sequence(
            [_normalize_state_label(state) for state in raw_states],
            actions,
            graph,
        )
        frames = _frames_from_sequences(states, obstacles, actions, raw_states=raw_states)
    return {
        "system": row.get("system", ""),
        "variant": row.get("variant", ""),
        "planner": row.get("planner", ""),
        "heuristic": row.get("heuristic", ""),
        "reward": row.get("total_reward", ""),
        "turns": row.get("turns", ""),
        "runtime_s": row.get("runtime_s", ""),
        "decision_ms": row.get("decision_ms", ""),
        "actions": actions,
        "states": states,
        "obstacles": obstacles,
        "raw_states": raw_states,
        "frames": [frame.to_dict() for frame in frames],
    }


def split_sequence(value: str, *, separator: str) -> list[str]:
    """Split a CSV sequence cell while dropping empty pieces."""
    if not value:
        return []
    return [piece.strip() for piece in value.split(separator) if piece.strip()]


def _darp_trace_frames(experiment_dir: Path, row: dict[str, str]) -> list[ReplayFrame]:
    """Read DARP replay frames from seed_N/darp_<heuristic>_trace.json."""
    seed = row.get("seed", "")
    heuristic = row.get("heuristic") or row.get("variant", "").removeprefix("DARP-")
    if not seed or not heuristic:
        return []
    key = re.sub(r"[^A-Za-z0-9_]+", "_", heuristic).strip("_")
    trace_path = experiment_dir / f"seed_{seed}" / f"darp_{key}_trace.json"
    if not trace_path.exists():
        return []
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    steps = payload.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return []
    frames: list[ReplayFrame] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        state = step.get("state")
        next_state = step.get("next_state")
        frames.append(
            ReplayFrame(
                step=index,
                agent=_state_label_from_mapping(state),
                obstacles=tuple(_obstacle_labels_from_mapping(state)),
                action=str(step.get("action", "") or ""),
                reward=_scalar_text(step.get("reward")),
                next_agent=_state_label_from_mapping(next_state),
                raw_state=_active_state_label(state),
            )
        )
    last_next_state = steps[-1].get("next_state") if isinstance(steps[-1], dict) else None
    if last_next_state is not None:
        frames.append(
            ReplayFrame(
                step=len(frames),
                agent=_state_label_from_mapping(last_next_state),
                obstacles=tuple(_obstacle_labels_from_mapping(last_next_state)),
                raw_state=_active_state_label(last_next_state),
            )
        )
    return frames


def _darp_trace_sequences(experiment_dir: Path, row: dict[str, str]) -> tuple[list[str], list[list[str]]]:
    """Read DARP robot and obstacle traces from seed_N/darp_<heuristic>_trace.json."""
    frames = _darp_trace_frames(experiment_dir, row)
    return [frame.agent for frame in frames if frame.agent], [list(frame.obstacles) for frame in frames]


def _frame_sequence(value: str) -> list[ReplayFrame]:
    """Parse a JSON-encoded replay-frame sequence."""
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    frames: list[ReplayFrame] = []
    for item in data:
        if isinstance(item, dict):
            frames.append(frame_from_mapping(item))
    return frames


def _frames_from_sequences(
    states: list[str],
    obstacles: list[list[str]],
    actions: list[str],
    *,
    raw_states: list[str] | None = None,
    include_action_tail: bool = True,
) -> list[ReplayFrame]:
    """Build replay frames from legacy separated state/action/obstacle arrays."""
    action_frame_count = len(actions) + 1 if include_action_tail else len(actions)
    frame_count = max(len(states), len(obstacles), action_frame_count)
    frames: list[ReplayFrame] = []
    for index in range(frame_count):
        agent = states[index] if index < len(states) else (states[-1] if states else "")
        frame_obstacles = tuple(obstacles[index]) if index < len(obstacles) else ()
        action = actions[index] if index < len(actions) else ""
        next_agent = states[index + 1] if index + 1 < len(states) else ""
        raw_state = raw_states[index] if raw_states and index < len(raw_states) else ""
        frames.append(
            ReplayFrame(
                step=index,
                agent=agent,
                obstacles=frame_obstacles,
                action=action,
                next_agent=next_agent,
                raw_state=raw_state,
            )
        )
    return frames


def _normalize_state_label(label: str) -> str:
    """Normalize solver state labels to graph node ids when possible."""
    if not label or label in {"(none)", "noop"}:
        return ""
    if label == "lost":
        return "lost"
    if "+" in label:
        label = label.split("+", 1)[0]
    if label.startswith("at___"):
        return label.removeprefix("at___")
    if label.startswith("robot-at___"):
        args = grounding_args(label)
        if len(args) == 2:
            return f"{args[0]},{args[1]}"
    return label


def _complete_state_sequence(
    states: list[str],
    actions: list[str],
    graph: dict[str, Any],
) -> list[str]:
    """Infer missing replay states from the graph and action sequence."""
    if not actions:
        return states
    clean_states = [state for state in states if state]
    if not clean_states:
        start = graph_start_state(graph)
        clean_states = [start] if start else []
    if not clean_states:
        return states
    completed = [clean_states[0]]
    for index, action in enumerate(actions):
        if index + 1 < len(clean_states):
            completed.append(clean_states[index + 1])
        else:
            completed.append(graph_next_state(graph, completed[-1], action))
    return completed


def _state_label_from_mapping(state: Any) -> str:
    """Return the robot/agent state label from a DARP trace state mapping."""
    if not isinstance(state, dict):
        return ""
    robot = [str(name) for name, value in sorted(state.items()) if value is True and str(name).startswith("robot-at___")]
    if robot:
        return _xy_grounding_label(robot[0])
    if any(str(name).startswith("robot-at___") for name in state):
        return "lost"
    at_fluents = [str(name) for name, value in sorted(state.items()) if value is True and str(name).startswith("at___")]
    if at_fluents:
        return _short_state_name(at_fluents[0])
    return ""


def _obstacle_labels_from_mapping(state: Any) -> list[str]:
    """Return obstacle positions from a DARP trace state mapping."""
    if not isinstance(state, dict):
        return []
    return [
        _xy_grounding_label(str(name))
        for name, value in sorted(state.items())
        if value is True and str(name).startswith("obstacle-at___")
    ]


def _active_state_label(state: Any) -> str:
    """Return all true state-fluent labels from one pyRDDLGym state mapping."""
    if not isinstance(state, dict):
        return ""
    active = [str(name) for name, value in sorted(state.items()) if value is True]
    return "+".join(active)


def _obstacle_sequence(value: str) -> list[list[str]]:
    """Parse a JSON-encoded per-step obstacle sequence."""
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    result: list[list[str]] = []
    for step in data:
        if isinstance(step, list):
            result.append([str(item) for item in step])
        else:
            result.append([])
    return result


def _xy_grounding_label(name: str) -> str:
    """Convert `robot-at___x__y`/`obstacle-at___x__y` to `x,y`."""
    args = grounding_args(name)
    if len(args) == 2:
        return f"{args[0]},{args[1]}"
    return name


def _short_state_name(name: str) -> str:
    """Shorten grounded labels in DARP JSON traces."""
    if name.startswith("at___"):
        return name.removeprefix("at___")
    return name


def _scalar_text(value: Any) -> str:
    """Render scalar trace values compactly for browser metadata."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
