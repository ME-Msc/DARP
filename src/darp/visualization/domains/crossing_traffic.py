"""Crossing-traffic replay decoding helpers."""

# TODO(visualization): Replace PROST bit-vector assumptions with parser metadata if PROST exposes it.

from __future__ import annotations

import re
from typing import Any

from darp.visualization.graph import object_sort_key, sort_object_labels


def prost_sequences(
    text: str,
    actions: list[str],
    graph: dict[str, Any],
) -> tuple[list[str], list[list[str]]]:
    """Decode PROST crossing-traffic state vectors into robot and obstacle traces."""
    if graph.get("kind") != "xy-grid":
        return [], []
    xpos, ypos = _xy_orders_from_graph(graph)
    if len(xpos) < 2 or len(ypos) < 3:
        return [], []
    middle_y = ypos[1:-1]
    deterministic_obstacle_order = [(x_value, y_value) for x_value in xpos[:-1] for y_value in middle_y]
    robot_order = [(x_value, y_value) for x_value in xpos for y_value in ypos]
    stochastic_obstacle_order = [(xpos[-1], y_value) for y_value in middle_y]
    expected_width = len(deterministic_obstacle_order) + len(robot_order) + len(stochastic_obstacle_order)
    states: list[str] = []
    obstacles: list[list[str]] = []
    for before_bar, after_bar in re.findall(r"^Current state:\s*([01](?:\s+[01])*)\s*\|\s*([01](?:\s+[01])*)?\s*$", text, re.M):
        bits = [int(value) for value in before_bar.split()]
        bits.extend(int(value) for value in after_bar.split())
        if len(bits) != expected_width:
            continue
        deterministic_end = len(deterministic_obstacle_order)
        robot_end = deterministic_end + len(robot_order)
        deterministic_obstacle_bits = bits[:deterministic_end]
        robot_bits = bits[deterministic_end:robot_end]
        stochastic_obstacle_bits = bits[robot_end:]
        states.append(_state_from_ordered_bits(robot_order, robot_bits))
        obstacle_labels = _positions_from_ordered_bits(deterministic_obstacle_order, deterministic_obstacle_bits)
        obstacle_labels.extend(_positions_from_ordered_bits(stochastic_obstacle_order, stochastic_obstacle_bits))
        obstacles.append(sorted(obstacle_labels, key=_xy_label_sort_key))
    states = [state for state in states if state]
    return states, obstacles[: len(states)]


def _xy_orders_from_graph(graph: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return sorted xpos/ypos labels from an xy-grid graph."""
    x_values: set[str] = set()
    y_values: set[str] = set()
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", ""))
        if "," not in node_id:
            continue
        x_value, y_value = node_id.split(",", 1)
        x_values.add(x_value)
        y_values.add(y_value)
    return sort_object_labels(x_values), sort_object_labels(y_values)


def _state_from_ordered_bits(order: list[tuple[str, str]], bits: list[int]) -> str:
    """Return the unique active position from ordered PROST robot bits."""
    active = _positions_from_ordered_bits(order, bits)
    if not active:
        return "lost"
    return active[0] if len(active) == 1 else "+".join(active)


def _positions_from_ordered_bits(order: list[tuple[str, str]], bits: list[int]) -> list[str]:
    """Return position labels whose matching bit is active."""
    return [
        f"{x_value},{y_value}"
        for (x_value, y_value), bit in zip(order, bits, strict=False)
        if bit == 1
    ]


def _xy_label_sort_key(label: str) -> tuple[tuple[str, int, str], tuple[str, int, str]]:
    """Sort labels such as x2,y3 using the same object-label order."""
    if "," not in label:
        return (object_sort_key(label), ("", 0, ""))
    x_value, y_value = label.split(",", 1)
    return (object_sort_key(x_value), object_sort_key(y_value))
