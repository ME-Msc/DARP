"""Compile parsed RDDL documents into DARP PlanningProblem objects."""

# TODO(phase-2.4): Replace structural identity dynamics with grounded CPF
# expression evaluation for general standard RDDL benchmarks.
# TODO(phase-2.4): Keep this compiler dependent on ParsedRDDL only, not on a
# concrete pyRDDLGym or pyrddl parser implementation.

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
from typing import Any

from darp.core.duration import FixedDurationModel
from darp.core.problem import PlanningProblem
from darp.core.types import Action, ObservationKey, RewardKey, State, TransitionKey
from darp.rddl.ast import RDDLASTNode
from darp.rddl.frontend import ParsedRDDL
from darp.rddl.loader import RDDLLoader, available_frontends


class RDDLCompileError(ValueError):
    """Raised when ParsedRDDL cannot be compiled. / 在 ParsedRDDL 无法编译时抛出。"""


@dataclass(frozen=True)
class _PVariable:
    """Store one parsed pvariable declaration. / 保存一个解析出的 pvariable 声明。"""

    name: str
    roles: frozenset[str]
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class _RDDLDocument:
    """Group top-level RDDL blocks needed by the compiler. / 组合 compiler 需要的顶层 RDDL 块。"""

    domain: RDDLASTNode
    instance: RDDLASTNode
    non_fluents: RDDLASTNode | None


class RDDLCompiler:
    """Compile canonical DARP AST into a PlanningProblem. / 将 DARP 标准 AST 编译为 PlanningProblem。"""

    def compile(self, loaded: ParsedRDDL) -> PlanningProblem:
        """Compile a ParsedRDDL object into a planning model. / 将 ParsedRDDL 编译成规划模型。"""
        if not isinstance(loaded.ast, RDDLASTNode):
            raise RDDLCompileError(
                "ParsedRDDL.ast must be a DARP RDDLASTNode before compilation."
            )

        document = _RDDLDocumentBuilder().build(loaded.ast)
        domain_name = document.domain.label
        instance_name = document.instance.label
        instance_domain = _assignment(document.instance, "domain")
        if instance_domain and instance_domain != domain_name:
            raise RDDLCompileError(
                f"Instance {instance_name!r} references domain {instance_domain!r}, "
                f"but parsed domain is {domain_name!r}."
            )

        objects = _object_table(document)
        pvariables = _pvariables(document.domain)
        state_variables = [pvar for pvar in pvariables if "state-fluent" in pvar.roles]
        action_variables = [pvar for pvar in pvariables if "action-fluent" in pvar.roles]
        if not state_variables:
            raise RDDLCompileError("No state-fluent pvariables were found in the RDDL domain.")
        if not action_variables:
            raise RDDLCompileError("No action-fluent pvariables were found in the RDDL domain.")

        primary_type = _primary_state_type(state_variables, objects)
        states = tuple(objects[primary_type])
        actions = tuple(pvar.name for pvar in action_variables)
        if not states:
            raise RDDLCompileError(f"No objects were found for state type {primary_type!r}.")

        initial_states = _initial_states(document.instance, state_variables[0].name)
        initial_belief = _initial_belief(states, initial_states)
        horizon = _float_assignment(document.instance, "horizon", default=1.0)
        discount = _float_assignment(document.instance, "discount", default=1.0)
        reward_constant = _float_assignment(document.domain, "reward", default=0.0)

        transitions = _identity_transitions(states, actions)
        observations = states
        observation_model = _identity_observations(observations, actions)
        rewards = _constant_rewards(states, actions, reward_constant)
        max_depth = max(1, int(ceil(horizon)))

        return PlanningProblem(
            states=states,
            actions=actions,
            observations=observations,
            transitions=transitions,
            observation_model=observation_model,
            rewards=rewards,
            initial_belief=initial_belief,
            horizon=horizon,
            discount=discount,
            duration_model=FixedDurationModel({action: 1.0 for action in actions}),
            max_depth=max_depth,
            name=instance_name,
            metadata={
                "source": "rddl-compiler",
                "frontend": loaded.frontend,
                "domain": domain_name,
                "instance": instance_name,
                "state_fluent": state_variables[0].name,
                "state_type": primary_type,
                "compiler_mode": "structural-identity",
            },
        )


class _RDDLDocumentBuilder:
    """Build a compact document view from a generic AST. / 从通用 AST 构建紧凑文档视图。"""

    def build(self, ast: RDDLASTNode) -> _RDDLDocument:
        """Extract domain, instance, and non-fluents blocks. / 提取 domain、instance 和 non-fluents 块。"""
        domain = _single_top_level(ast, "domain")
        instance = _single_top_level(ast, "instance")
        non_fluents = _optional_top_level(ast, "non-fluents")
        return _RDDLDocument(domain=domain, instance=instance, non_fluents=non_fluents)


def _single_top_level(ast: RDDLASTNode, kind: str) -> RDDLASTNode:
    """Return exactly one top-level block of a kind. / 返回指定类型的唯一顶层块。"""
    matches = [node for node in _top_level_blocks(ast) if node.kind == kind]
    if len(matches) != 1:
        raise RDDLCompileError(f"Expected exactly one top-level {kind!r} block, found {len(matches)}.")
    return matches[0]


def _optional_top_level(ast: RDDLASTNode, kind: str) -> RDDLASTNode | None:
    """Return zero or one top-level block of a kind. / 返回指定类型的零个或一个顶层块。"""
    matches = [node for node in _top_level_blocks(ast) if node.kind == kind]
    if len(matches) > 1:
        raise RDDLCompileError(f"Expected at most one top-level {kind!r} block, found {len(matches)}.")
    return matches[0] if matches else None


def _top_level_blocks(ast: RDDLASTNode) -> list[RDDLASTNode]:
    """Return all top-level RDDL blocks under file nodes. / 返回 file 节点下所有顶层 RDDL 块。"""
    if ast.kind != "rddl":
        raise RDDLCompileError(f"Expected RDDL root node, found {ast.kind!r}.")
    return [block for file_node in ast.children for block in file_node.children]


def _assignment(node: RDDLASTNode, name: str) -> str | None:
    """Return the raw value of one assignment child. / 返回某个 assignment 子节点的原始值。"""
    prefix = f"{name} ="
    for child in node.children:
        if child.kind == "assignment" and child.label.startswith(prefix):
            return child.label.split("=", 1)[1].strip()
    return None


def _float_assignment(node: RDDLASTNode, name: str, *, default: float) -> float:
    """Return one assignment as a float with a fallback. / 以 float 读取一个 assignment 并支持默认值。"""
    value = _assignment(node, name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RDDLCompileError(f"Assignment {name!r} must be numeric, found {value!r}.") from exc


def _object_table(document: _RDDLDocument) -> dict[str, tuple[str, ...]]:
    """Parse object declarations from non-fluents or instance blocks. / 从 non-fluents 或 instance 块解析对象声明。"""
    container = document.non_fluents or document.instance
    block = _child_block(container, "objects")
    if block is None:
        raise RDDLCompileError("No objects block was found in the RDDL input.")

    objects: dict[str, tuple[str, ...]] = {}
    for statement in block.children:
        type_name, _, raw_values = statement.label.partition(":")
        if not raw_values:
            continue
        objects[type_name.strip()] = tuple(_set_items(raw_values))
    return objects


def _pvariables(domain: RDDLASTNode) -> list[_PVariable]:
    """Parse pvariable declarations from a domain block. / 从 domain 块解析 pvariable 声明。"""
    block = _child_block(domain, "pvariables")
    if block is None:
        raise RDDLCompileError("No pvariables block was found in the RDDL domain.")

    result = []
    for statement in block.children:
        name_part, _, spec_part = statement.label.partition(":")
        name, parameters = _signature(name_part.strip())
        roles = frozenset(_set_items(spec_part))
        result.append(_PVariable(name=name, roles=roles, parameters=parameters))
    return result


def _child_block(node: RDDLASTNode, label: str) -> RDDLASTNode | None:
    """Return a direct block child by label. / 按 label 返回一个直接 block 子节点。"""
    for child in node.children:
        if child.kind == "block" and child.label == label:
            return child
    return None


def _signature(text: str) -> tuple[str, tuple[str, ...]]:
    """Parse a pvariable or fluent signature. / 解析 pvariable 或 fluent 的签名。"""
    if "(" not in text:
        return text.strip(), ()
    name, _, rest = text.partition("(")
    params, _, _tail = rest.partition(")")
    return name.strip(), tuple(item.strip() for item in params.split(",") if item.strip())


def _set_items(text: str) -> list[str]:
    """Parse a simple RDDL set-like value. / 解析简单的 RDDL 集合式值。"""
    cleaned = text.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _primary_state_type(
    state_variables: list[_PVariable], objects: dict[str, tuple[str, ...]]
) -> str:
    """Choose the object type used as compact states. / 选择用作紧凑 state 的对象类型。"""
    for pvar in state_variables:
        for parameter in pvar.parameters:
            if parameter in objects:
                return parameter
    return next(iter(objects))


def _initial_states(instance: RDDLASTNode, state_fluent: str) -> tuple[State, ...]:
    """Extract true initial states for one state fluent. / 提取某个 state fluent 为真的初始 state。"""
    block = _child_block(instance, "init-state")
    if block is None:
        return ()
    result: list[State] = []
    for statement in block.children:
        name, parameters = _signature(statement.label)
        if name == state_fluent and parameters:
            result.append(parameters[0])
    return tuple(result)


def _initial_belief(states: tuple[State, ...], initial_states: tuple[State, ...]) -> dict[State, float]:
    """Build a normalized initial belief over compact states. / 构建紧凑 state 上的归一化初始 belief。"""
    if not initial_states:
        initial_states = (states[0],)
    unknown = [state for state in initial_states if state not in states]
    if unknown:
        raise RDDLCompileError(f"Initial state objects are not declared: {unknown!r}.")
    mass = 1.0 / len(initial_states)
    return {state: (mass if state in initial_states else 0.0) for state in states}


def _identity_transitions(
    states: tuple[State, ...], actions: tuple[Action, ...]
) -> dict[TransitionKey, float]:
    """Create identity dynamics for structural compilation. / 为结构化编译创建 identity 动态。"""
    return {
        (source, action, target): 1.0 if source == target else 0.0
        for source in states
        for action in actions
        for target in states
    }


def _identity_observations(
    observations: tuple[State, ...], actions: tuple[Action, ...]
) -> dict[ObservationKey, float]:
    """Create a fully observable identity observation model. / 创建完全可观测的 identity 观测模型。"""
    return {
        (observation, state, action): 1.0 if observation == state else 0.0
        for observation in observations
        for state in observations
        for action in actions
    }


def _constant_rewards(
    states: tuple[State, ...], actions: tuple[Action, ...], value: float
) -> dict[RewardKey, float]:
    """Create a constant reward table. / 创建常数 reward 表。"""
    return {(state, action): value for state in states for action in actions}


def build_parser() -> argparse.ArgumentParser:
    """Build the compiler inspection CLI parser. / 构建 compiler 检查命令的 CLI parser。"""
    parser = argparse.ArgumentParser(description="Compile RDDL into a DARP PlanningProblem.")
    parser.add_argument("domain", help="RDDL domain file")
    parser.add_argument("instance", help="RDDL instance file")
    parser.add_argument(
        "--frontend",
        default="darp",
        choices=available_frontends(),
        help="parser frontend to use before compilation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the compiler inspection command. / 运行 compiler 检查命令。"""
    args = build_parser().parse_args(argv)
    loaded = RDDLLoader(args.frontend).load(Path(args.domain), Path(args.instance))
    problem = RDDLCompiler().compile(loaded)
    print(json.dumps(problem.to_summary_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
