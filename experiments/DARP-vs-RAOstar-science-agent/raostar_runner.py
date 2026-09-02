"""Run the pinned compatibility fork's Science model with its RAOStar class."""

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
    "models/fake_planner.py",
    "models/rock_sample.py",
    "rao/__init__.py",
    "rmpyl/rmpyl.py",
)

SITES = {
    "minerals": {"coords": (0.0, -10.0), "value": 10.0},
    "funny_rock": {"coords": (-10.0, -10.0), "value": 5.0},
    "geiser": {"coords": (0.0, 10.0), "value": 4.0},
    "alien_lair": {"coords": (13.0, 13.0), "value": 100.0},
    "relay": {"coords": (0.0, 0.0), "value": 0.0},
}
PRIORS = {
    "minerals": 0.5,
    "funny_rock": 0.5,
    "geiser": 0.5,
    "alien_lair": 0.2,
    "relay": 0.0,
}


@dataclass(frozen=True, slots=True)
class RAOStarRunner:
    """Thin delegate to one verified compatibility-fork checkout."""

    repository: Path
    model_class: type[Any]
    planner_class: type[Any]
    temporal_constraint_class: type[Any]

    @classmethod
    def create(
        cls,
        *,
        repository: Path | None,
        cache_root: Path,
    ) -> "RAOStarRunner":
        checkout = _resolve(repository, cache_root)
        # NumPy 2 removed this alias, while the pinned 2016 RAO* source uses it.
        # The compatibility stays in this adapter process and never edits source.
        if not hasattr(np, "infty"):
            np.infty = np.inf
        with _import_path(checkout), redirect_stdout(io.StringIO()):
            model_module = importlib.import_module("rao.models.fake_planner")
            planner_module = importlib.import_module("rao.raostar")
            rmpyl_module = importlib.import_module("rmpyl.rmpyl")
        _require_origin(model_module, checkout / "models" / "fake_planner.py")
        _require_origin(planner_module, checkout / "raostar.py")
        _require_origin(rmpyl_module, checkout / "rmpyl" / "rmpyl.py")
        return cls(
            checkout,
            model_module.tFakePlannerRockSampleModel,
            planner_module.RAOStar,
            rmpyl_module.TemporalConstraint,
        )

    def make_problem(self) -> tuple[Any, Any]:
        """Build the existing source-test model and its 16-particle belief."""

        sites = {
            name: {"coords": tuple(data["coords"]), "value": data["value"]}
            for name, data in SITES.items()
        }
        model = self.model_class(
            sites=sites,
            perform_scheduling=False,
            duration_type="uncontrollable_probabilistic",
            path_risks=[0.001, 0.01],
            prob_discovery=0.9,
            verbose=0,
        )
        # This is copied from tests/psulu/test_tfake_planner.py.  Scheduling is
        # disabled, so the temporal constraint is retained as inert provenance.
        time_window = self.temporal_constraint_class(
            start=model.global_start_event,
            end=model.global_end_event,
            ctype="controllable",
            lb=0.0,
            ub=1000.0,
        )
        belief = model.get_initial_belief(
            prior=dict(PRIORS),
            initial_pos=(-12.5, 13.5),
            init_tcs=[time_window],
            goal_site="relay",
        )
        return model, belief

    def run(
        self,
        model: Any,
        belief: Any,
        *,
        delta: float,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Run the original RAOStar.search() and return its native counters."""

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
            "objective": float(performance["optimal_value"]),
            "risk": risk,
            "time_s": float(performance["total_elapsed_time"]),
            "expanded_nodes": int(performance["expanded_nodes"]),
            # This is the original paper code's cumulative evaluated-particles
            # counter; it is not a distinct-state count like DARP's metric.
            "evaluated_states": int(performance["evaluated_particles"]),
            "iterations": len(performance["root_value_series"]),
            "complete": True,
        }


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
