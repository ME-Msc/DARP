"""Graph extraction helpers for DARP replay visualizations."""

# TODO(visualization): Add non-grid graph layouts for benchmarks without location/xpos/ypos objects.

from __future__ import annotations

import re
from typing import Any

from darp.adapter.loader import RDDLLoader


def graph_from_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Extract a grid/navigation graph from the first row's RDDL files."""
    domain = rows[0].get("domain")
    instance = rows[0].get("instance")
    if not domain or not instance:
        return empty_graph("missing RDDL paths in runs.csv")
    try:
        problem = RDDLLoader().load(domain, instance)
    except Exception as exc:  # pragma: no cover - visualizer should degrade gracefully.
        return empty_graph(f"pyRDDLGym load failed: {exc}")
    model = problem.model
    type_to_objects = getattr(model, "type_to_objects", {})
    if "location" in type_to_objects:
        return _location_graph(model)
    if "xpos" in type_to_objects and "ypos" in type_to_objects:
        return _xy_graph(model)
    return empty_graph("no location or xpos/ypos object structure found")


def graph_with_replay_states(graph: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Add special replay-only nodes such as `lost` when traces need them."""
    graph = dict(graph)
    nodes = [dict(node) for node in graph.get("nodes", [])]
    node_ids = {str(node.get("id", "")) for node in nodes}
    replay_states = {
        state
        for run in runs
        for state in _run_replay_states(run)
        if state and state not in node_ids
    }
    if "lost" in replay_states:
        max_x = max((float(node.get("x", 0)) for node in nodes), default=0.0)
        max_y = max((float(node.get("y", 0)) for node in nodes), default=0.0)
        nodes.append(
            {
                "id": "lost",
                "label": "LOST",
                "x": max_x + 1.0,
                "y": max_y,
                "goal": False,
                "risk": True,
                "start": False,
                "lost": True,
                "probability": None,
            }
        )
    graph["nodes"] = nodes
    return graph


def _run_replay_states(run: dict[str, Any]) -> list[str]:
    """Return replay states from the canonical frame sequence, with legacy fallback."""
    frames = run.get("frames", [])
    if isinstance(frames, list) and frames:
        return [str(frame.get("agent", "")) for frame in frames if isinstance(frame, dict)]
    return [str(state) for state in run.get("states", [])]


def sort_object_labels(values: Any) -> list[str]:
    """Sort object labels by embedded numbers when possible."""
    return sorted((str(value) for value in values), key=object_sort_key)


def object_sort_key(value: str) -> tuple[str, int, str]:
    """Return a stable key such as x6 < x9 < x14 < x21."""
    match = re.fullmatch(r"([A-Za-z_]+)(-?\d+)", value)
    if not match:
        return (value, 0, value)
    return (match.group(1), int(match.group(2)), value)


def empty_graph(note: str) -> dict[str, Any]:
    """Return an empty graph payload with an explanatory note."""
    return {"kind": "unknown", "nodes": [], "edges": [], "note": note}


def grounding_args(label: str) -> list[str]:
    """Split a pyRDDLGym grounded label into argument names."""
    if "___" not in label:
        return []
    args = label.split("___", 1)[1].removesuffix("'")
    return args.split("__") if args else []


def graph_start_state(graph: dict[str, Any]) -> str:
    """Return the graph start node id when one is available."""
    for node in graph.get("nodes", []):
        if node.get("start"):
            return str(node.get("id", ""))
    nodes = graph.get("nodes", [])
    return str(nodes[0].get("id", "")) if nodes else ""


def graph_next_state(graph: dict[str, Any], state: str, action: str) -> str:
    """Infer a deterministic nominal next state from graph edges."""
    if not state or not action or action == "noop":
        return state
    for edge in graph.get("edges", []):
        if edge.get("source") == state and edge.get("action") == action:
            return str(edge.get("target", state))
    return state


def _location_graph(model: Any) -> dict[str, Any]:
    """Build a graph for object-valued `location` domains such as tiny_grid."""
    locations = [str(value) for value in model.type_to_objects.get("location", [])]
    coords = _location_coordinates(locations)
    nodes = [
        {
            "id": location,
            "label": location,
            "x": coords[location][0],
            "y": coords[location][1],
            "goal": _bool_nonfluent(model, "is-goal", (location,)),
            "risk": _bool_nonfluent(model, "is-risk", (location,)),
            "start": _bool_nonfluent(model, "is-start", (location,)) or _bool_state(model, "at", (location,)),
            "probability": None,
        }
        for location in locations
    ]
    edges = []
    for direction in ("north", "south", "west", "east"):
        for source, target in _true_groundings(model, f"next-{direction}", arity=2):
            edges.append({"source": source, "target": target, "action": f"move-{direction}", "label": direction})
    return {"kind": "location-grid", "nodes": nodes, "edges": edges, "note": ""}


def _xy_graph(model: Any) -> dict[str, Any]:
    """Build a graph for `xpos`/`ypos` navigation domains."""
    xpos = sort_object_labels(str(value) for value in model.type_to_objects.get("xpos", []))
    ypos = sort_object_labels(str(value) for value in model.type_to_objects.get("ypos", []))
    nodes = []
    for col, x_value in enumerate(xpos):
        for row, y_value in enumerate(reversed(ypos)):
            node_id = f"{x_value},{y_value}"
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "x": col,
                    "y": row,
                    "goal": _bool_nonfluent(model, "GOAL", (x_value, y_value)),
                    "risk": False,
                    "start": _bool_state(model, "robot-at", (x_value, y_value)),
                    "probability": _number_nonfluent(model, "P", (x_value, y_value)),
                }
            )
    edges = []
    for source_x, target_x in _true_groundings(model, "EAST", arity=2):
        for y_value in ypos:
            edges.append({"source": f"{source_x},{y_value}", "target": f"{target_x},{y_value}", "action": "move-east", "label": "east"})
    for source_x, target_x in _true_groundings(model, "WEST", arity=2):
        for y_value in ypos:
            edges.append({"source": f"{source_x},{y_value}", "target": f"{target_x},{y_value}", "action": "move-west", "label": "west"})
    for source_y, target_y in _true_groundings(model, "NORTH", arity=2):
        for x_value in xpos:
            edges.append({"source": f"{x_value},{source_y}", "target": f"{x_value},{target_y}", "action": "move-north", "label": "north"})
    for source_y, target_y in _true_groundings(model, "SOUTH", arity=2):
        for x_value in xpos:
            edges.append({"source": f"{x_value},{source_y}", "target": f"{x_value},{target_y}", "action": "move-south", "label": "south"})
    return {"kind": "xy-grid", "nodes": nodes, "edges": edges, "note": ""}


def _location_coordinates(locations: list[str]) -> dict[str, tuple[int, int]]:
    """Infer compact grid coordinates from labels like c11; otherwise use row-major order."""
    coords: dict[str, tuple[int, int]] = {}
    parsed = [(location, re.fullmatch(r"[A-Za-z_]*?(\d)(\d)", location)) for location in locations]
    if all(match for _, match in parsed):
        for location, match in parsed:
            assert match is not None
            coords[location] = (int(match.group(2)) - 1, int(match.group(1)) - 1)
        return coords
    width = max(1, int(len(locations) ** 0.5))
    for index, location in enumerate(locations):
        coords[location] = (index % width, index // width)
    return coords


def _bool_nonfluent(model: Any, name: str, args: tuple[str, ...]) -> bool:
    """Return one grounded Boolean non-fluent value."""
    return bool(_grounded_value(model, name, args, default=False))


def _number_nonfluent(model: Any, name: str, args: tuple[str, ...]) -> float | None:
    """Return one grounded numeric non-fluent value."""
    value = _grounded_value(model, name, args, default=None)
    return None if value is None else float(value)


def _bool_state(model: Any, name: str, args: tuple[str, ...]) -> bool:
    """Return one grounded Boolean state-fluent value."""
    return bool(_grounded_value(model, name, args, default=False, table_name="state_fluents"))


def _grounded_value(
    model: Any,
    name: str,
    args: tuple[str, ...],
    *,
    default: Any,
    table_name: str = "non_fluents",
) -> Any:
    """Look up a grounded lifted-model value by variable name and arguments."""
    groundings = getattr(model, "variable_groundings", {}).get(name, [])
    values = getattr(model, table_name, {}).get(name, [])
    target = _grounding_name(name, args)
    for grounding, value in zip(groundings, values, strict=False):
        if grounding == target:
            return value
    return default


def _true_groundings(model: Any, name: str, *, arity: int) -> list[tuple[str, ...]]:
    """Return true-valued non-fluent groundings for one predicate."""
    groundings = getattr(model, "variable_groundings", {}).get(name, [])
    values = getattr(model, "non_fluents", {}).get(name, [])
    result = []
    for grounding, value in zip(groundings, values, strict=False):
        if value is True:
            args = tuple(grounding_args(grounding))
            if len(args) == arity:
                result.append(args)
    return result


def _grounding_name(name: str, args: tuple[str, ...]) -> str:
    """Return pyRDDLGym's grounded variable label for a lifted name and args."""
    return name if not args else f"{name}___{'__'.join(args)}"
