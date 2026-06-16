"""pyRDDLGym problem bundle for standard RDDL inputs."""

# TODO(phase-9.1): Add richer benchmark summaries for arbitrary constrained-cost
# fluents beyond sidecar-selected finite bool costs.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from darp.adapter.grounded import GroundedRDDLView

if TYPE_CHECKING:
    from pyRDDLGym.core.compiler.model import RDDLGroundedModel


class RDDLLoadError(RuntimeError):
    """Raised when pyRDDLGym cannot load RDDL. / pyRDDLGym 无法加载 RDDL 时抛出。"""


@dataclass(frozen=True)
class PyRDDLGymProblem:
    """Carry pyRDDLGym env/model/AST for one RDDL problem. / 承载一个 RDDL 问题的 pyRDDLGym env/model/AST。"""

    domain: str
    instance: str
    native_ast: Any | None = None
    model: Any | None = None
    env: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def component_summary(self) -> dict[str, str | None]:
        """Summarize pyRDDLGym problem components by type. / 按类型汇总 pyRDDLGym problem 组件。"""
        return {
            "native_ast": type(self.native_ast).__name__ if self.native_ast is not None else None,
            "model": type(self.model).__name__ if self.model is not None else None,
            "env": type(self.env).__name__ if self.env is not None else None,
        }

    def build_grounded_model(self) -> "RDDLGroundedModel":
        """Return an enum-safe pyRDDLGym grounded model. / 返回 enum 安全的 pyRDDLGym grounded model。"""
        if self.native_ast is None:
            raise RDDLLoadError("PyRDDLGymProblem has no native AST to ground.")
        try:
            return _build_enum_aware_grounded_model(self.native_ast)
        except ImportError as exc:
            raise RDDLLoadError("pyRDDLGym grounder is required to ground RDDL.") from exc

    def build_grounded_view(self) -> GroundedRDDLView:
        """Return DARP's stable view over the pyRDDLGym grounded model. / 返回 pyRDDLGym grounded model 的 DARP 稳定视图。"""
        return GroundedRDDLView(self.build_grounded_model())

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for CLI inspection. / 返回适合 CLI 检查的 JSON 友好摘要。"""
        model = self.model
        return {
            "source": self.metadata.get("source", "pyRDDLGym"),
            "domain": self.domain,
            "instance": self.instance,
            "components": self.component_summary(),
            "metadata": self.metadata,
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
            "planner_interfaces": [
                "pyRDDLGym grounded model view",
                "AND-OR history tree over action/observation histories",
                "Phase 7 full-tree/HILP search over the grounded model and duration sidecars",
                "Phase 8 Gurobi full-ILP/p-ILP solver",
            ],
        }


def rddl_path(path: str | Path) -> Path:
    """Normalize one RDDL file path without requiring absolutes. / 规范化 RDDL 文件路径但不强制绝对路径。"""
    return Path(path).expanduser()


def rddl_load_error(domain: Path, instance: Path, exc: Exception) -> RDDLLoadError:
    """Wrap pyRDDLGym load failures with file context. / 用文件上下文包装 pyRDDLGym 加载失败。"""
    return RDDLLoadError(f"pyRDDLGym failed to load domain={domain} instance={instance}: {exc}")


def _build_enum_aware_grounded_model(native_ast: Any) -> Any:
    """Ground pyRDDLGym ASTs while normalizing enum literals. / 归一化 enum literal 后 ground pyRDDLGym AST。"""
    from pyRDDLGym.core.compiler.model import RDDLPlanningModel
    from pyRDDLGym.core.debug.exception import raise_warning
    from pyRDDLGym.core.grounder import RDDLGrounder

    class EnumAwareRDDLGrounder(RDDLGrounder):
        """Normalize enum literals in pyRDDLGym init blocks. / 归一化 pyRDDLGym 初始化块里的 enum literal。"""

        def _ground_init_state(self) -> None:
            """Ground init-state entries after stripping enum markers. / 去掉 enum 标记后 ground init-state。"""
            if hasattr(self.AST.instance, "init_state"):
                for init_vals in self.AST.instance.init_state:
                    (key, subs), val = init_vals
                    if subs:
                        key = self._append_variation_to_name(key, RDDLPlanningModel.strip_literals(subs))
                    if key in self.states:
                        self.states[key] = _strip_literal_value(val, RDDLPlanningModel)
                    else:
                        raise_warning(
                            f"Init-state block initializes undefined state-fluent <{key}>.",
                            "red",
                        )

        def _ground_init_non_fluents(self) -> None:
            """Ground non-fluent entries after stripping enum markers. / 去掉 enum 标记后 ground non-fluent。"""
            if hasattr(self.AST.non_fluents, "init_non_fluent"):
                for init_vals in self.AST.non_fluents.init_non_fluent:
                    (key, variations_list), val = init_vals
                    if variations_list is not None:
                        key = self._generate_grounded_names(
                            key,
                            [RDDLPlanningModel.strip_literals(variations_list)],
                            return_grounding_param_dict=False,
                        )[0]
                    if key in self.nonfluents:
                        self.nonfluents[key] = _strip_literal_value(val, RDDLPlanningModel)
                    else:
                        raise_warning(
                            f"Non-fluents block initializes undefined non-fluent <{key}>.",
                            "red",
                        )

    return EnumAwareRDDLGrounder(native_ast).ground()


def _strip_literal_value(value: Any, planning_model: Any) -> Any:
    """Strip one enum marker from string values. / 去掉字符串值中的一个 enum 标记。"""
    if isinstance(value, str):
        return planning_model.strip_literal(value)
    return value


def _keys(value: object) -> list[str]:
    """Return sorted mapping keys for summaries. / 返回摘要中使用的排序键列表。"""
    if isinstance(value, dict):
        return sorted(str(key) for key in value)
    return []
