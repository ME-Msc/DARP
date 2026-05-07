"""Small semantic checks for RDDL requirements."""

# TODO(phase-4.6): Implement `cpf-deterministic` as the next isolated
# requirement feature after this reward-deterministic baseline.

from __future__ import annotations

from dataclasses import dataclass

from darp.rddl.ast import RDDLASTNode
from darp.rddl.expressions import RDDLExpressionError, expression_uses_distribution, parse_expression


class RDDLCompileError(ValueError):
    """Raised when ParsedRDDL cannot be compiled. / 在 ParsedRDDL 无法编译时抛出。"""


@dataclass(frozen=True)
class PVariable:
    """Store one parsed pvariable declaration. / 保存一个解析出的 pvariable 声明。"""

    name: str
    roles: frozenset[str]
    parameters: tuple[str, ...]
    default: bool | float | str = False


@dataclass(frozen=True)
class RDDLDocument:
    """Group top-level RDDL blocks needed by compiler checks. / 组合 compiler 校验需要的顶层 RDDL 块。"""

    domain: RDDLASTNode
    instance: RDDLASTNode
    non_fluents: RDDLASTNode | None


@dataclass(frozen=True)
class RequirementInfo:
    """Describe one RDDL requirement and DARP rollout status. / 描述一个 RDDL requirement 和 DARP 实现状态。"""

    name: str
    status: str
    en: str
    zh: str


REQUIREMENT_PLAN: dict[str, RequirementInfo] = {
    "reward-deterministic": RequirementInfo(
        name="reward-deterministic",
        status="supported",
        en="Reward must be deterministic; DARP rejects stochastic distribution calls in reward.",
        zh="reward 必须是确定性的；DARP 会拒绝 reward 中的随机分布调用。",
    ),
    "cpf-deterministic": RequirementInfo(
        name="cpf-deterministic",
        status="planned-next",
        en="Transition CPFs must be deterministic; this will be implemented as the next small step.",
        zh="转移 CPF 必须是确定性的；这是下一步准备单独实现的功能。",
    ),
    "partially-observed": RequirementInfo(
        name="partially-observed",
        status="planned",
        en="Observation fluents define a POMDP observation model; planned as a separate step.",
        zh="观测 fluent 定义 POMDP 观测模型；后续会作为独立步骤实现。",
    ),
    "concurrent": RequirementInfo(
        name="concurrent",
        status="planned",
        en="Multiple non-default actions can happen in one step; planned as a separate step.",
        zh="一步中可以有多个非默认动作；后续会作为独立步骤实现。",
    ),
    "constrained-state": RequirementInfo(
        name="constrained-state",
        status="planned",
        en="State invariants and action preconditions constrain valid behavior; planned later.",
        zh="状态不变量和动作前置条件约束合法行为；后续实现。",
    ),
    "integer-valued": RequirementInfo(
        name="integer-valued",
        status="planned",
        en="Integer-valued pvariables require a richer value/state representation.",
        zh="整数值 pvariable 需要更完整的值和状态表示。",
    ),
    "continuous": RequirementInfo(
        name="continuous",
        status="planned",
        en="Real-valued pvariables require continuous or numeric state/value support.",
        zh="实数值 pvariable 需要连续或数值状态和值支持。",
    ),
    "multivalued": RequirementInfo(
        name="multivalued",
        status="planned",
        en="Enum-valued pvariables require multivalued grounding beyond boolean fluents.",
        zh="枚举值 pvariable 需要超出布尔 fluent 的多值 grounding。",
    ),
    "intermediate-nodes": RequirementInfo(
        name="intermediate-nodes",
        status="planned",
        en="Intermediate and derived fluents require layered CPF evaluation.",
        zh="中间和派生 fluent 需要分层 CPF 求值。",
    ),
}
"""Requirement rollout plan with bilingual notes. / 带中英文说明的 requirement 逐步实现计划。"""

SUPPORTED_REQUIREMENTS = frozenset({"reward-deterministic"})
"""Requirements supported in the current baseline. / 当前基线真正支持的 requirements。"""


def requirements_from_domain(domain: RDDLASTNode) -> frozenset[str]:
    """Parse the domain requirements section. / 解析 domain 的 requirements 区块。"""
    requirements: list[str] = []
    assignment = _assignment(domain, "requirements")
    if assignment is not None:
        requirements.extend(_set_items(assignment))
    block = _child_block(domain, "requirements")
    if block is not None:
        for statement in block.children:
            requirements.extend(_set_items("{" + statement.label + "}"))
    return frozenset(item.strip() for item in requirements if item.strip())


def validate_rddl_semantics(document: RDDLDocument, requirements: frozenset[str]) -> None:
    """Validate only the current requirement baseline. / 只校验当前基线支持的 requirement。"""
    _validate_known_requirements(requirements)
    _validate_supported_requirements(requirements)
    if "reward-deterministic" in requirements:
        _validate_reward_deterministic(document.domain)


def _validate_known_requirements(requirements: frozenset[str]) -> None:
    """Reject misspelled or unknown requirement names. / 拒绝拼错或未知的 requirement 名称。"""
    unknown = sorted(requirements - frozenset(REQUIREMENT_PLAN))
    if unknown:
        raise RDDLCompileError(f"Unknown RDDL requirement(s): {unknown!r}.")


def _validate_supported_requirements(requirements: frozenset[str]) -> None:
    """Reject requirements that are planned but not implemented yet. / 拒绝已规划但尚未实现的 requirements。"""
    unsupported = sorted(requirements - SUPPORTED_REQUIREMENTS)
    if unsupported:
        raise RDDLCompileError(
            "Only 'reward-deterministic' is supported in this baseline; "
            f"unsupported requirement(s): {unsupported!r}."
        )


def _validate_reward_deterministic(domain: RDDLASTNode) -> None:
    """Reject stochastic distribution calls in reward. / 拒绝 reward 中的随机分布调用。"""
    reward_text = _assignment(domain, "reward")
    if reward_text is None:
        raise RDDLCompileError("No reward assignment was found in the RDDL domain.")
    try:
        reward = parse_expression(reward_text)
    except RDDLExpressionError as exc:
        raise RDDLCompileError(f"Unsupported RDDL reward expression: {exc}") from exc
    if expression_uses_distribution(reward):
        raise RDDLCompileError(
            "reward-deterministic forbids stochastic distribution calls in reward."
        )


def _assignment(node: RDDLASTNode, name: str) -> str | None:
    """Return the raw value of one assignment child. / 返回某个 assignment 子节点的原始值。"""
    prefix = f"{name} ="
    for child in node.children:
        if child.kind == "assignment" and child.label.startswith(prefix):
            return child.label.split("=", 1)[1].strip()
    return None


def _child_block(node: RDDLASTNode, label: str) -> RDDLASTNode | None:
    """Return a direct block child by label. / 按 label 返回一个直接 block 子节点。"""
    for child in node.children:
        if child.kind == "block" and child.label == label:
            return child
    return None


def _set_items(text: str) -> list[str]:
    """Parse a simple RDDL set-like value. / 解析简单的 RDDL 集合式值。"""
    cleaned = text.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    return [item.strip() for item in cleaned.split(",") if item.strip()]
