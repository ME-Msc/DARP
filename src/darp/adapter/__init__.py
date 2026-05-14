"""Adapter for loading, grounding, and simulating standard RDDL through pyRDDLGym."""

# pyRDDLGym is the only supported standard-RDDL parser/runtime path.

from typing import Any

__all__ = [
    "ActionDict",
    "GroundedCPF",
    "GroundedRDDLView",
    "GroundedVariable",
    "ParticleBelief",
    "PyRDDLGymProblem",
    "PyRDDLGymRuntime",
    "RDDLLoadError",
    "RDDLLoader",
    "UnsupportedRDDLFeature",
    "UnsupportedRDDLFeatureError",
]


def __getattr__(name: str) -> Any:
    """Lazily expose adapter classes without importing command modules. / 惰性暴露 adapter 类，避免预先导入命令模块。"""
    if name in {
        "GroundedCPF",
        "GroundedRDDLView",
        "GroundedVariable",
        "UnsupportedRDDLFeature",
        "UnsupportedRDDLFeatureError",
    }:
        from darp.adapter.grounded import (
            GroundedCPF,
            GroundedRDDLView,
            GroundedVariable,
            UnsupportedRDDLFeature,
            UnsupportedRDDLFeatureError,
        )

        return {
            "GroundedCPF": GroundedCPF,
            "GroundedRDDLView": GroundedRDDLView,
            "GroundedVariable": GroundedVariable,
            "UnsupportedRDDLFeature": UnsupportedRDDLFeature,
            "UnsupportedRDDLFeatureError": UnsupportedRDDLFeatureError,
        }[name]
    if name in {"PyRDDLGymProblem", "RDDLLoadError"}:
        from darp.adapter.problem import PyRDDLGymProblem, RDDLLoadError

        return {"PyRDDLGymProblem": PyRDDLGymProblem, "RDDLLoadError": RDDLLoadError}[name]
    if name == "RDDLLoader":
        from darp.adapter.loader import RDDLLoader

        return RDDLLoader
    if name in {"ActionDict", "ParticleBelief", "PyRDDLGymRuntime"}:
        from darp.adapter.runtime import ActionDict, ParticleBelief, PyRDDLGymRuntime

        return {
            "ActionDict": ActionDict,
            "ParticleBelief": ParticleBelief,
            "PyRDDLGymRuntime": PyRDDLGymRuntime,
        }[name]
    raise AttributeError(name)
