"""pyRDDLGym-to-DARP planning-model adapter boundary."""

# TODO(phase-4.1): Add finite-discrete enumerability checks for the optional
# pyRDDLGym model/env to PlanningProblem path.
# TODO(phase-6.2): Thread DurationModel sidecars through both runtime and
# explicit PlanningProblem adapters.
# TODO(phase-7.1): Feed stable planner interfaces into AND-OR tree, full ILP,
# and HILP implementations.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from darp.core.duration import DurationModel
from darp.core.problem import PlanningProblem
from darp.rddl.artifacts import RDDLArtifacts
from darp.rddl.loader import RDDLLoader


class RDDLCompileError(NotImplementedError):
    """Raised when pyRDDLGym artifacts cannot yet become a PlanningProblem. / pyRDDLGym 产物暂不能转成 PlanningProblem 时抛出。"""


@dataclass(frozen=True)
class PyRDDLGymPlanningAdapter:
    """Reserve the pyRDDLGym model/env extraction boundary. / 预留 pyRDDLGym model/env 抽取边界。"""

    duration_model: DurationModel | None = None

    def compile(self, loaded: RDDLArtifacts) -> PlanningProblem:
        """Convert pyRDDLGym artifacts into a DARP PlanningProblem. / 将 pyRDDLGym 产物转换为 DARP PlanningProblem。"""
        if loaded.model is None or loaded.env is None:
            raise RDDLCompileError("DARP expects pyRDDLGym RDDLArtifacts with model and env.")
        raise RDDLCompileError(
            "pyRDDLGym planning extraction is planned but not implemented yet. "
            "Next steps: wrap reset/step/model as a generative runtime, then add "
            "optional finite-discrete PlanningProblem enumeration and DurationModel hooks."
        )


def summarize_pyrddlgym_artifacts(loaded: RDDLArtifacts) -> dict[str, Any]:
    """Return a compact summary of pyRDDLGym parser artifacts. / 返回 pyRDDLGym parser 产物摘要。"""
    model = loaded.model
    return {
        "source": loaded.metadata.get("source", "pyRDDLGym"),
        "domain": loaded.domain,
        "instance": loaded.instance,
        "artifacts": loaded.artifact_summary(),
        "metadata": dict(loaded.metadata),
        "model": {
            "domain_name": getattr(model, "domain_name", None),
            "instance_name": getattr(model, "instance_name", None),
            "horizon": getattr(model, "horizon", None),
            "discount": getattr(model, "discount", None),
            "state_fluents": _keys(getattr(model, "state_fluents", None)),
            "action_fluents": _keys(getattr(model, "action_fluents", None)),
            "observ_fluents": _keys(getattr(model, "observ_fluents", None)),
            "non_fluents": _keys(getattr(model, "non_fluents", None)),
            "types": _keys(getattr(model, "type_to_objects", None)),
        },
        "planning_problem": None,
        "planned_extraction": [
            "pyRDDLGym generative runtime interface",
            "state/action/type/object metadata",
            "action construction and constraint propagation",
            "MDP/POMDP observation and belief boundary",
            "optional finite-discrete PlanningProblem enumeration",
            "DurationModel sidecar hook",
            "AND-OR tree / full ILP / HILP planner inputs",
        ],
    }


def _keys(value: object) -> list[str]:
    """Return sorted mapping keys for summaries. / 返回摘要中使用的排序键列表。"""
    if isinstance(value, dict):
        return sorted(str(key) for key in value)
    return []


def build_parser() -> argparse.ArgumentParser:
    """Build the pyRDDLGym inspection CLI parser. / 构建 pyRDDLGym 检查命令 parser。"""
    parser = argparse.ArgumentParser(description="Load standard RDDL with pyRDDLGym.")
    parser.add_argument("domain", help="RDDL domain file")
    parser.add_argument("instance", help="RDDL instance file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pyRDDLGym artifact inspection command. / 运行 pyRDDLGym 产物检查命令。"""
    args = build_parser().parse_args(argv)
    loaded = RDDLLoader().load(Path(args.domain), Path(args.instance))
    print(json.dumps(summarize_pyrddlgym_artifacts(loaded), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
