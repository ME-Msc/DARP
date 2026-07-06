"""Run the tiny_grid_fixed_1 DARP-vs-PROST experiment.

The generic runner lives in `darp_prost_compare.py`; this file only defines the
tiny-grid RDDL paths and the PROST state-vector parser.
"""

# TODO(phase-9.2): Add more scenario scripts that reuse `darp_prost_compare.py`.

from __future__ import annotations

import re
from pathlib import Path

from darp_prost_compare import DARPProstExperimentSpec, run_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_GRID_STATE_ORDER = ("c11", "c12", "c13", "c21", "c22", "c23", "c31", "c32", "c33")
TINY_GRID_TRANSITIONS = {
    "move-north": {
        "c11": "c11",
        "c12": "c12",
        "c13": "c13",
        "c21": "c11",
        "c22": "c12",
        "c23": "c13",
        "c31": "c21",
        "c32": "c22",
        "c33": "c23",
    },
    "move-south": {
        "c11": "c21",
        "c12": "c22",
        "c13": "c23",
        "c21": "c31",
        "c22": "c32",
        "c23": "c33",
        "c31": "c31",
        "c32": "c32",
        "c33": "c33",
    },
    "move-west": {
        "c11": "c11",
        "c12": "c11",
        "c13": "c12",
        "c21": "c21",
        "c22": "c21",
        "c23": "c22",
        "c31": "c31",
        "c32": "c31",
        "c33": "c32",
    },
    "move-east": {
        "c11": "c12",
        "c12": "c13",
        "c13": "c13",
        "c21": "c22",
        "c22": "c23",
        "c23": "c23",
        "c31": "c32",
        "c32": "c33",
        "c33": "c33",
    },
    "noop": {location: location for location in TINY_GRID_STATE_ORDER},
}


def tiny_grid_prost_state_trace(text: str, actions: tuple[str, ...]) -> tuple[str, ...]:
    """Parse PROST tiny-grid state vectors and append the inferred final state."""
    states = tuple(
        _prost_state_label(bits)
        for bits in re.findall(r"^Current state:\s*([01](?:\s+[01])*)\s*\|", text, re.M)
    )
    states = tuple(state for state in states if state)
    if not states or not actions:
        return states
    final_state = _tiny_grid_next_state(states[-1], actions[-1])
    return states + (final_state,) if final_state else states


def _prost_state_label(bits: str) -> str:
    """Map PROST's tiny-grid Boolean state vector back to a location label."""
    values = tuple(int(value) for value in bits.split())
    active = [
        location
        for location, value in zip(TINY_GRID_STATE_ORDER, values, strict=False)
        if value == 1
    ]
    return "+".join(active)


def _tiny_grid_next_state(state: str, action: str) -> str:
    """Infer the next tiny-grid state from a parsed PROST action."""
    if "+" in state:
        return ""
    return TINY_GRID_TRANSITIONS.get(action, TINY_GRID_TRANSITIONS["noop"]).get(state, state)


SPEC = DARPProstExperimentSpec(
    name="tiny_grid_fixed_1",
    description="Compare DARP HILP reachable-bellman and PROST on tiny_grid_fixed_1.",
    domain_path=REPO_ROOT / "examples" / "rddl" / "tiny_grid_domain.rddl",
    instance_path=REPO_ROOT / "examples" / "rddl" / "tiny_grid_instance.rddl",
    duration_path=REPO_ROOT / "examples" / "durations" / "tiny_grid.yaml",
    prost_instance_name="tiny_grid_inst",
    default_output_dir=REPO_ROOT / "experiments" / "tiny_grid_fixed_1",
    prost_state_trace_parser=tiny_grid_prost_state_trace,
    prost_state_note="PROST final state is inferred from the last printed state and action",
)


def main(argv: list[str] | None = None) -> int:
    """Run the configured tiny-grid comparison experiment."""
    return run_experiment(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
