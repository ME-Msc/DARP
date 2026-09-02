"""Run the reduced Power Supply model with the pinned RAOStar class."""

from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any, Iterator

import numpy as np

RAOSTAR_URL = "https://github.com/ME-Msc/rao-star.git"
RAOSTAR_COMMIT = "f51bfdc1ff8fcb2504dcb38c3b36d719506501fb"
REQUIRED_FILES = (
    "raostar.py",
    "models/models.py",
    "rao/__init__.py",
)

ACTIONS = (
    "noop",
    "open_cb1",
    "close_cb1",
    "open_sd1",
    "close_sd1",
    "open_sd2",
    "close_sd2",
    "open_cb2",
    "close_cb2",
)


@dataclass(frozen=True, slots=True)
class RAOStarRunner:
    """Thin delegate to a verified RAO* compatibility-fork checkout."""

    repository: Path
    planner_class: type[Any]
    model_class: type[Any]

    @classmethod
    def create(
        cls,
        *,
        repository: Path | None,
        cache_root: Path,
    ) -> "RAOStarRunner":
        checkout = _resolve(repository, cache_root)
        # NumPy 2 removed this alias; keep compatibility in the adapter process
        # without touching the pinned external source.
        if not hasattr(np, "infty"):
            np.infty = np.inf
        with _import_path(checkout), redirect_stdout(io.StringIO()):
            planner_module = importlib.import_module("rao.raostar")
            model_module = importlib.import_module("rao.models.models")
        _require_origin(planner_module, checkout / "raostar.py")
        _require_origin(model_module, checkout / "models" / "models.py")
        return cls(
            checkout,
            planner_module.RAOStar,
            _power_supply_model_class(model_module.CCHyperGraphModel),
        )

    def make_problem(self, sensor_count: int) -> tuple[Any, dict[Any, Any]]:
        """Return the reduced three-line model and uniform fault belief."""

        model = self.model_class(sensor_count=sensor_count)
        return model, model.initial_belief()

    def run(
        self,
        model: Any,
        belief: dict[Any, Any],
        *,
        delta: float,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Run RAOStar.search() to natural completion."""

        planner = self.planner_class(
            model,
            node_name="id",
            cc=delta,
            cc_type="overall",
            terminal_prob=1.0,
            randomization=0.0,
            propagate_risk=True,
            halt_on_violation=False,
            verbose=0,
        )
        with redirect_stdout(io.StringIO()):
            policy, _, performance = planner.search(
                belief,
                time_limit=inf if timeout_s is None else timeout_s,
            )
        complete = bool(policy) and not planner._opennodes
        if not complete:
            raise RuntimeError("RAO* did not naturally complete its search.")
        risk = float(performance["exec_risk_for_optimal_value"])
        if risk > delta + 1e-9:
            raise RuntimeError(f"RAO* risk {risk:.17g} exceeds delta={delta}.")
        return {
            "cost": float(performance["optimal_value"]),
            "risk": risk,
            "time_s": float(performance["total_elapsed_time"]),
            "expanded_nodes": int(performance["expanded_nodes"]),
            "evaluated_states": int(performance["evaluated_particles"]),
            "iterations": len(performance["root_value_series"]),
            "complete": True,
        }


def _power_supply_model_class(base: type[Any]) -> type[Any]:
    class PowerSupplyModel(base):
        """Exact RAO* Model API view of the checked-in reduced PSR RDDL."""

        def __init__(self, sensor_count: int) -> None:
            super().__init__()
            if sensor_count not in (1, 2):
                raise ValueError("sensor_count must be 1 or 2")
            self.sensor_count = sensor_count
            self.horizon = 4
            self.last_action_depth = 3
            self.line_penalty = 5.0
            self.is_maximization = False
            self.immutable_actions = True

        @staticmethod
        def action_label(action: Any) -> str:
            label = str(action)
            if label not in ACTIONS:
                raise ValueError(f"Unknown Power Supply action: {label}")
            return label

        def initial_belief(self) -> dict[Any, Any]:
            belief: dict[Any, Any] = {}
            for fault in (1, 2):
                state = {
                    "fault": fault,
                    "cb1_closed": False,
                    "sd1_closed": True,
                    "sd2_closed": False,
                    "cb2_closed": True,
                    "unsafe": False,
                    "done": False,
                    "depth": 0,
                }
                belief[self.hash_state(state)] = [state, 0.5]
            return belief

        def actions(self, state: dict[str, Any]) -> list[str]:
            return [] if self.is_terminal(state) else list(ACTIONS)

        def state_transitions(
            self,
            state: dict[str, Any],
            action: Any,
        ) -> list[list[Any]]:
            return [[_next_state(state, self.action_label(action)), 1.0]]

        def observations(self, state: dict[str, Any]) -> list[list[Any]]:
            cb1 = int(bool(state["cb1_closed"]))
            cb2 = int(bool(state["cb2_closed"]))
            observation = cb1 if self.sensor_count == 1 else cb1 + 2 * cb2
            return [[observation, 1.0]]

        @staticmethod
        def obs_repr(observation: Any) -> str:
            return str(observation)

        def value(self, state: dict[str, Any], action: Any) -> float:
            label = self.action_label(action)
            if bool(state["done"]):
                return 0.0
            if label == "noop":
                return self.line_penalty * _unpowered_healthy_lines(state)
            cost = 1.0
            if int(state["depth"]) == self.last_action_depth:
                cost += self.line_penalty * _unpowered_healthy_lines(
                    _next_state(state, label)
                )
            return cost

        def terminal_value(self, state: dict[str, Any]) -> float:
            # RAO* may mark a belief terminal when its remaining risk bound
            # admits no action.  That implicit stop must pay the same breakdown
            # cost as the explicit noop/finish action.  Natural RDDL terminal
            # states already paid it in value(), so they add nothing here.
            if self.is_terminal(state):
                return 0.0
            return self.line_penalty * _unpowered_healthy_lines(state)

        @staticmethod
        def heuristic(state: dict[str, Any]) -> float:
            del state
            return 0.0

        @staticmethod
        def state_risk(state: dict[str, Any]) -> float:
            return 1.0 if bool(state["unsafe"]) else 0.0

        def execution_risk_heuristic(self, state: dict[str, Any]) -> float:
            return self.state_risk(state)

        def is_terminal(self, state: dict[str, Any]) -> bool:
            return bool(state["done"]) or int(state["depth"]) > self.last_action_depth

    PowerSupplyModel.__name__ = "PowerSupplyModel"
    return PowerSupplyModel


def _next_state(state: dict[str, Any], action: str) -> dict[str, Any]:
    def requested(name: str) -> bool:
        return action != f"open_{name}" and (
            action == f"close_{name}" or bool(state[f"{name}_closed"])
        )

    sd1 = requested("sd1")
    sd2 = requested("sd2")
    cb1_requested = requested("cb1")
    cb2_requested = requested("cb2")
    fault = int(state["fault"])
    cb1_fault = fault == 1 or (sd1 and fault == 2) or (sd1 and sd2 and fault == 3)
    cb2_fault = fault == 3 or (sd2 and fault == 2) or (sd2 and sd1 and fault == 1)
    return {
        "fault": fault,
        "cb1_closed": cb1_requested and not cb1_fault,
        "sd1_closed": sd1,
        "sd2_closed": sd2,
        "cb2_closed": cb2_requested and not cb2_fault,
        "unsafe": bool(state["unsafe"])
        or (cb1_requested and cb1_fault)
        or (cb2_requested and cb2_fault),
        "done": bool(state["done"]) or action == "noop",
        "depth": int(state["depth"])
        if bool(state["done"])
        else int(state["depth"]) + 1,
    }


def _unpowered_healthy_lines(state: dict[str, Any]) -> int:
    fault = int(state["fault"])
    cb1 = bool(state["cb1_closed"])
    sd1 = bool(state["sd1_closed"])
    sd2 = bool(state["sd2_closed"])
    cb2 = bool(state["cb2_closed"])
    line1 = fault != 1 and not (cb1 or (cb2 and sd2 and sd1))
    line2 = fault != 2 and not ((cb1 and sd1) or (cb2 and sd2))
    line3 = fault != 3 and not (cb2 or (cb1 and sd1 and sd2))
    return int(line1) + int(line2) + int(line3)


def _resolve(explicit: Path | None, cache_root: Path) -> Path:
    if explicit is not None:
        return _verify(explicit)
    destination = (
        cache_root.expanduser().resolve()
        / f"ME-Msc-rao-star-{RAOSTAR_COMMIT[:12]}"
    )
    if destination.exists():
        return _verify(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rao-star-download-", dir=destination.parent
    ) as temporary:
        candidate = Path(temporary) / "checkout"
        _command(
            "git", "clone", "--quiet", "--no-checkout", RAOSTAR_URL, str(candidate)
        )
        _git(candidate, "checkout", "--quiet", "--detach", RAOSTAR_COMMIT)
        _verify(candidate)
        candidate.rename(destination)
    return _verify(destination)


def _verify(path: Path) -> Path:
    repository = path.expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (repository / name).is_file()]
    if missing:
        raise RuntimeError(f"{repository} is missing: {', '.join(missing)}")
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise RuntimeError(f"Source path is not a repository root: {repository}")
    commit = _git(repository, "rev-parse", "HEAD")
    if commit != RAOSTAR_COMMIT:
        raise RuntimeError(f"Unexpected RAO* commit {commit}; expected {RAOSTAR_COMMIT}")
    if status := _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(f"External RAO* repository is not clean:\n{status}")
    return repository


@contextmanager
def _import_path(repository: Path) -> Iterator[None]:
    sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(repository))


def _require_origin(module: Any, expected: Path) -> None:
    origin = Path(getattr(module, "__file__", "")).resolve()
    if origin != expected.resolve():
        raise RuntimeError(f"Loaded {module.__name__} from {origin}, expected {expected}")


def _git(repository: Path, *arguments: str) -> str:
    return _command("git", "-C", str(repository), *arguments)


def _command(*command: str) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result.stdout.strip()
