"""pyRDDLGym problem bundle for standard RDDL inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from darp.adapter.grounded import GroundedRDDLView

if TYPE_CHECKING:
    from pyRDDLGym.core.compiler.model import RDDLGroundedModel


class RDDLLoadError(RuntimeError):
    """Raised when pyRDDLGym cannot load RDDL. / pyRDDLGym 无法加载 RDDL 时抛出。"""


@dataclass(frozen=True)
class PyRDDLGymProblem:
    """Carry the pyRDDLGym environment and AST needed by DARP."""

    native_ast: Any
    env: Any
    _grounded_model_cache: Any | None = field(default=None, init=False, repr=False, compare=False)

    def build_grounded_model(self) -> "RDDLGroundedModel":
        """Return and cache one enum-safe pyRDDLGym grounded model.

        pyRDDLGym's environment keeps a lifted model, so the initial syntax
        grounding is still required. Repeated planner/view construction must
        not pay that full cost again.

        / pyRDDLGym env 保存 lifted model；首次仍需 grounding，但同一问题后续
        构建 view/planner 会复用缓存，不再重复完整 grounding。
        """
        if self._grounded_model_cache is not None:
            return self._grounded_model_cache
        try:
            grounded = _build_enum_aware_grounded_model(self.native_ast)
        except ImportError as exc:
            raise RDDLLoadError("pyRDDLGym grounder is required to ground RDDL.") from exc
        object.__setattr__(self, "_grounded_model_cache", grounded)
        return grounded

    def build_grounded_view(self) -> GroundedRDDLView:
        """Return DARP's stable view over the pyRDDLGym grounded model. / 返回 pyRDDLGym grounded model 的 DARP 稳定视图。"""
        return GroundedRDDLView(self.build_grounded_model())


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
