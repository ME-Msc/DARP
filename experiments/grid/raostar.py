"""Run pinned external sources through their RAO* adapter.

下载并核验固定版本的外部源码，通过其已有适配器运行 RAO*。
"""

from __future__ import annotations

import importlib
import logging
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

logger = logging.getLogger(__name__)


def run_instance(runner, domain: Path, instance: Path, timeout: float) -> dict:
    """Only compare the supported fixed-duration grid. / 仅执行已匹配的固定时长 Grid。"""
    from darp.adapter.loader import load_rddl
    canonical = Path(__file__).resolve().parents[2] / "benchmarks/grid/fixed-duration/domain.rddl"
    if domain.read_text() != canonical.read_text():
        raise ValueError("RAOstar adapter requires the canonical grid domain, including its risk predicate.")
    problem = load_rddl(domain, instance)
    try:
        model = problem.env.model
        values = model.non_fluents
        size = int(values["max_row"]) + 1
        expected_state = {"grid_row": size - 1, "grid_col": 0, "row_mod5": (size - 1) % 5, "col_mod5": 0}
        if (size not in (5, 100) or int(values["max_col"]) + 1 != size
                or (values["goal_row"], values["goal_col"]) != (0, size - 1)
                or values["transition_accuracy"] != .85 or values["observation_accuracy"] != .85
                or values["normal_duration"] != 1 or values["muddy_duration"] != 1
                or values["duration_noise_variance"] != 0
                or {key: model.state_fluents[key] for key in expected_state} != expected_state
                or model.max_allowed_actions != 1 or model.discount != 1):
            raise ValueError("RAOstar supports only the canonical 5/100 grid with duration=1 and paper defaults.")
        grid = runner.make_grid(size, int(model.horizon), float(problem.native_ast.instance.risk_budget))
        metrics = runner.run(grid, timeout_s=timeout)
        return {"status": "ok", **{k: v for k, v in metrics.items() if k in ("objective", "risk", "time_s", "complete")}}
    finally:
        problem.env.close()

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
    """Delegate to verified repositories. 仅代理到已核验的两个外部仓库。"""

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
            logger.debug(
                "Baseline checkout rename failed; fallback will verify existing cache: "
                "candidate=%s destination=%s",
                candidate,
                destination,
                exc_info=True,
            )
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
