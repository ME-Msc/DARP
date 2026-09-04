"""Fetch the pinned external sources and run their existing RAO* adapter."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from math import inf
from pathlib import Path
from types import ModuleType
from typing import Any

CONSTRAINED_POMDP_URL = "https://github.com/ME-Msc/Constrained-POMDP.git"
CONSTRAINED_POMDP_COMMIT = "d84d099493b973a63d879255d2221c1930d649aa"
RAOSTAR_URL = "https://github.com/ME-Msc/RAOStar.git"
RAOSTAR_COMMIT = "543f782d80ceb9555130e911c1fcf7074153d267"


@dataclass(frozen=True, slots=True)
class _Source:
    name: str
    url: str
    commit: str
    files: tuple[str, ...]


CONSTRAINED_POMDP = _Source(
    "ME-Msc-Constrained-POMDP",
    CONSTRAINED_POMDP_URL,
    CONSTRAINED_POMDP_COMMIT,
    ("grid_experiment.py", "raostar_adapter.py", "instance.py"),
)
RAOSTAR = _Source(
    "ME-Msc-RAOStar",
    RAOSTAR_URL,
    RAOSTAR_COMMIT,
    ("raostar.py", "belief.py", "raostarhypergraph.py"),
)


@dataclass(frozen=True, slots=True)
class RAOStarRunner:
    """Thin delegate to verified Constrained-POMDP and RAOStar checkouts."""

    constrained_pomdp_path: Path
    raostar_path: Path
    grid_experiment: ModuleType
    adapter: ModuleType

    @classmethod
    def create(
        cls,
        *,
        constrained_pomdp_repo: Path | None,
        raostar_repo: Path | None,
        cache_root: Path,
    ) -> RAOStarRunner:
        constrained = _resolve(
            CONSTRAINED_POMDP, constrained_pomdp_repo, cache_root
        )
        raostar = _resolve(RAOSTAR, raostar_repo, cache_root)
        with _import_path(constrained):
            _module_from(constrained, "instance")
            adapter = _module_from(constrained, "raostar_adapter")
            experiment = _module_from(constrained, "grid_experiment")
        if getattr(adapter, "EXPECTED_RAOSTAR_COMMIT", None) != RAOSTAR_COMMIT:
            raise RuntimeError("The external adapter expects a different RAOStar commit.")
        if not callable(getattr(experiment, "make_grid_instance", None)):
            raise TypeError("External grid_experiment lacks make_grid_instance().")
        if not callable(getattr(adapter, "run_raostar", None)):
            raise TypeError("External raostar_adapter lacks run_raostar().")
        return cls(constrained, raostar, experiment, adapter)

    def make_grid(self, size: int, horizon: int, delta: float) -> Any:
        scenario = self.grid_experiment.Scenario(size, horizon, delta)
        return self.grid_experiment.make_grid_instance(scenario)

    def run(self, grid: Any, *, timeout_s: float | None) -> dict[str, Any]:
        metrics = self.adapter.run_raostar(
            grid,
            self.raostar_path,
            allow_unverified=False,
            time_limit=inf if timeout_s is None else timeout_s,
        )
        if not bool(metrics.complete):
            raise RuntimeError("RAO* did not finish its search.")
        risk = float(metrics.risk)
        if risk > float(grid.delta) + 1e-6:
            raise RuntimeError(f"RAO* risk {risk:.17g} exceeds delta={grid.delta}.")
        return {
            "objective": float(metrics.objective),
            "risk": risk,
            "time_s": float(metrics.time_s),
            "n": int(metrics.n),
            "iterations": int(metrics.iterations),
            "complete": True,
        }


def _resolve(source: _Source, explicit: Path | None, cache_root: Path) -> Path:
    if explicit is not None:
        return _verify(explicit, source)
    destination = cache_root.expanduser().resolve() / (
        f"{source.name}-{source.commit[:12]}"
    )
    if destination.exists():
        return _verify(destination, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{source.name}-download-", dir=destination.parent
    ) as temporary:
        candidate = Path(temporary) / "checkout"
        _command("git", "clone", "--quiet", "--no-checkout", source.url, str(candidate))
        _git(candidate, "checkout", "--quiet", "--detach", source.commit)
        _verify(candidate, source)
        try:
            candidate.rename(destination)
        except OSError:
            if not destination.exists():
                raise
    return _verify(destination, source)


def _verify(path: Path, source: _Source) -> Path:
    repository = path.expanduser().resolve()
    missing = [name for name in source.files if not (repository / name).is_file()]
    if missing:
        raise RuntimeError(f"{repository} is missing: {', '.join(missing)}")
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise RuntimeError(f"Source path is not a repository root: {repository}")
    commit = _git(repository, "rev-parse", "HEAD")
    if commit != source.commit:
        raise RuntimeError(f"Unexpected commit {commit}; expected {source.commit}")
    if status := _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise RuntimeError(f"External repository is not clean: {repository}\n{status}")
    return repository


@contextmanager
def _import_path(repository: Path) -> Iterator[None]:
    sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(repository))


def _module_from(repository: Path, name: str) -> ModuleType:
    expected = (repository / f"{name}.py").resolve()
    module = sys.modules.get(name)
    if module is None:
        module = importlib.import_module(name)
    origin = Path(getattr(module, "__file__", "")).resolve()
    if origin != expected:
        raise RuntimeError(f"Loaded {name!r} from {origin}, expected {expected}")
    return module


def _git(repository: Path, *arguments: str) -> str:
    return _command("git", "-C", str(repository), *arguments)


def _command(*command: str) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(detail.strip()) from exc
    return result.stdout.strip()


__all__ = [
    "CONSTRAINED_POMDP_COMMIT",
    "RAOSTAR_COMMIT",
    "RAOStarRunner",
]
