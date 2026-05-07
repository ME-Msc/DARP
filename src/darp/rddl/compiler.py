"""Compile parsed RDDL documents into DARP PlanningProblem objects."""

# TODO(phase-6.2): Reuse factored CPF dependency information during AND-OR
# expansion instead of grounding every next-state valuation eagerly.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from itertools import product
import json
from math import ceil
from pathlib import Path
from typing import Any, Mapping

from darp.core.duration import FixedDurationModel
from darp.core.problem import PlanningProblem
from darp.core.types import Action, GroundAtom, Observation, ObservationKey, RewardKey, State, TransitionKey
from darp.rddl.ast import RDDLASTNode
from darp.rddl.expressions import (
    EvaluationContext,
    ExpressionValue,
    RDDLExpressionError,
    parse_expression,
)
from darp.rddl.frontend import ParsedRDDL
from darp.rddl.loader import RDDLLoader, available_frontends
from darp.rddl.semantics import PVariable as _PVariable
from darp.rddl.semantics import RDDLCompileError
from darp.rddl.semantics import RDDLDocument as _RDDLDocument
from darp.rddl.semantics import requirements_from_domain
from darp.rddl.semantics import validate_rddl_semantics


@dataclass(frozen=True)
class _GroundedTables:
    """Store grounded transition and reward tables. / 保存 grounding 后的 transition 和 reward 表。"""

    transitions: dict[TransitionKey, float]
    rewards: dict[RewardKey, float]
    compiler_mode: str


@dataclass(frozen=True)
class _FactoredGrounding:
    """Store all explicit tables for a factored finite model. / 保存 factored 有限模型的显式表。"""

    states: tuple[State, ...]
    observations: tuple[Observation, ...]
    transitions: dict[TransitionKey, float]
    observation_model: dict[ObservationKey, float]
    reset_observation_model: dict[tuple[Observation, State], float]
    rewards: dict[RewardKey, float]
    initial_belief: dict[State, float]


@dataclass(frozen=True)
class _CPF:
    """Store one CPF target signature and expression. / 保存一个 CPF 目标签名和表达式。"""

    name: str
    parameters: tuple[str, ...]
    expression: str


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
        requirements = requirements_from_domain(document.domain)
        validate_rddl_semantics(document, requirements)
        state_variables = [pvar for pvar in pvariables if "state-fluent" in pvar.roles]
        action_variables = [pvar for pvar in pvariables if "action-fluent" in pvar.roles]
        if not state_variables:
            raise RDDLCompileError("No state-fluent pvariables were found in the RDDL domain.")
        if not action_variables:
            raise RDDLCompileError("No action-fluent pvariables were found in the RDDL domain.")

        horizon = _float_assignment(document.instance, "horizon", default=1.0)
        discount = _float_assignment(document.instance, "discount", default=1.0)
        max_nondef_actions = _int_assignment(document.instance, "max-nondef-actions", default=1)
        actions, action_fluents = _ground_actions(action_variables, objects, max_nondef_actions)
        action_pvariables = frozenset(pvar.name for pvar in action_variables)
        if not _use_factored_grounding(state_variables):
            primary_type = _primary_state_type(state_variables, objects)
            states = tuple(objects[primary_type])
            if not states:
                raise RDDLCompileError(f"No objects were found for state type {primary_type!r}.")
            initial_states = _initial_states(document.instance, state_variables[0].name)
            initial_belief = _initial_belief(states, initial_states)
            grounded = _ground_compact_tables(
                document,
                states,
                actions,
                state_variables[0].name,
                objects,
                action_pvariables,
                action_fluents,
            )
            observations: tuple[Observation, ...] = states
            observation_model = _identity_observations(observations, actions)
            reset_observation_model = {}
            compiler_metadata = {
                "state_fluent": state_variables[0].name,
                "state_type": primary_type,
                "compiler_mode": grounded.compiler_mode,
            }
        else:
            grounded_model = _ground_factored_model(
                document=document,
                state_variables=state_variables,
                action_variables=action_variables,
                objects=objects,
                actions=actions,
                action_fluents=action_fluents,
                action_pvariables=action_pvariables,
            )
            states = grounded_model.states
            observations = grounded_model.observations
            initial_belief = grounded_model.initial_belief
            grounded = _GroundedTables(
                transitions=grounded_model.transitions,
                rewards=grounded_model.rewards,
                compiler_mode="factored-grounded-rddl-expressions",
            )
            observation_model = grounded_model.observation_model
            reset_observation_model = grounded_model.reset_observation_model
            compiler_metadata = {
                "state_fluents": [pvar.name for pvar in state_variables],
                "observation_fluents": [pvar.name for pvar in _observ_variables(pvariables)],
                "compiler_mode": grounded.compiler_mode,
                "state_encoding": "active-ground-atom-tuple",
            }
        max_depth = max(1, int(ceil(horizon)))

        return PlanningProblem(
            states=states,
            actions=actions,
            observations=observations,
            transitions=grounded.transitions,
            observation_model=observation_model,
            rewards=grounded.rewards,
            initial_belief=initial_belief,
            horizon=horizon,
            discount=discount,
            duration_model=FixedDurationModel({action: 1.0 for action in actions}),
            reset_observation_model=reset_observation_model,
            action_fluents=action_fluents,
            max_nondef_actions=max_nondef_actions,
            max_depth=max_depth,
            name=instance_name,
            metadata={
                "source": "rddl-compiler",
                "frontend": loaded.frontend,
                "domain": domain_name,
                "instance": instance_name,
                "requirements": sorted(requirements),
                "max_nondef_actions": max_nondef_actions,
                **compiler_metadata,
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


def _int_assignment(node: RDDLASTNode, name: str, *, default: int) -> int:
    """Return one assignment as an int with a fallback. / 以 int 读取一个 assignment 并支持默认值。"""
    value = _assignment(node, name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise RDDLCompileError(f"Assignment {name!r} must be integer-like, found {value!r}.") from exc


def _object_table(document: _RDDLDocument) -> dict[str, tuple[str, ...]]:
    """Parse object declarations from non-fluents or instance blocks. / 从 non-fluents 或 instance 块解析对象声明。"""
    container = document.non_fluents or document.instance
    block = _child_block(container, "objects")
    if block is None:
        domain_objects = _enum_object_table(document.domain)
        if domain_objects:
            return domain_objects
        return {}

    objects: dict[str, tuple[str, ...]] = {}
    for statement in block.children:
        type_name, _, raw_values = statement.label.partition(":")
        if not raw_values:
            continue
        objects[type_name.strip()] = tuple(_object_items(raw_values))
    return objects


def _enum_object_table(domain: RDDLASTNode) -> dict[str, tuple[str, ...]]:
    """Parse enum type declarations as compact objects. / 将 enum 类型声明解析为紧凑对象。"""
    block = _child_block(domain, "types")
    if block is None:
        return {}
    objects: dict[str, tuple[str, ...]] = {}
    for statement in block.children:
        type_name, _, raw_values = statement.label.partition(":")
        if not raw_values or raw_values.strip() == "object":
            continue
        objects[type_name.strip()] = tuple(_object_items(raw_values))
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
        result.append(
            _PVariable(
                name=name,
                roles=roles,
                parameters=parameters,
                default=_pvariable_default(roles),
            )
        )
    return result


def _pvariable_default(roles: frozenset[str]) -> ExpressionValue:
    """Extract the default value from a pvariable role set. / 从 pvariable role 集合中提取默认值。"""
    for role in roles:
        key, separator, value = role.partition("=")
        if separator and key.strip() == "default":
            return _literal_value(value.strip())
    return False


def _literal_value(text: str) -> ExpressionValue:
    """Parse a simple bool/number/object literal. / 解析简单 bool、数字或对象 literal。"""
    cleaned = text.strip()
    if cleaned.lower() == "true":
        return True
    if cleaned.lower() == "false":
        return False
    try:
        return float(cleaned)
    except ValueError:
        return _object_name(cleaned)


def _use_factored_grounding(state_variables: list[_PVariable]) -> bool:
    """Return whether the compiler should enumerate factored valuations. / 判断是否应枚举 factored valuation。"""
    return len(state_variables) != 1 or len(state_variables[0].parameters) != 1


def _observ_variables(pvariables: list[_PVariable]) -> list[_PVariable]:
    """Return observation-fluent pvariables. / 返回 observation-fluent pvariable。"""
    return [pvar for pvar in pvariables if "observ-fluent" in pvar.roles]


def _ground_actions(
    action_variables: list[_PVariable],
    objects: dict[str, tuple[str, ...]],
    max_nondef_actions: int,
) -> tuple[tuple[Action, ...], dict[Action, frozenset[GroundAtom]]]:
    """Ground action pvariables into constrained action sets. / 将 action pvariable grounding 成受约束的 action 集。"""
    atoms: list[GroundAtom] = []
    for pvar in action_variables:
        if not pvar.parameters:
            atoms.append((pvar.name, ()))
            continue
        domains = []
        for parameter in pvar.parameters:
            if parameter not in objects:
                raise RDDLCompileError(
                    f"Action parameter type {parameter!r} for {pvar.name!r} has no objects."
                )
            domains.append(objects[parameter])
        for values in product(*domains):
            atoms.append((pvar.name, tuple(values)))

    if max_nondef_actions < 0:
        raise RDDLCompileError("max-nondef-actions must be non-negative.")
    if max_nondef_actions == 0:
        return ("noop",), {"noop": frozenset()}

    action_fluents: dict[Action, frozenset[GroundAtom]] = {}
    for size in range(1, min(max_nondef_actions, len(atoms)) + 1):
        for active_atoms in combinations(atoms, size):
            action = _format_action_set(active_atoms)
            action_fluents[action] = frozenset(active_atoms)
    return tuple(action_fluents), action_fluents


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


def _object_items(text: str) -> list[str]:
    """Parse object names while hiding enum literal markers. / 解析对象名并隐藏 enum literal 标记。"""
    return [_object_name(item) for item in _set_items(text)]


def _object_name(text: str) -> str:
    """Normalize one object literal for compact state names. / 将单个对象 literal 规范化为紧凑 state 名。"""
    return text.strip().lstrip("@")


def _format_ground_atom(name: str, parameters: tuple[str, ...]) -> str:
    """Format one grounded pvariable as a compact string. / 将一个 grounded pvariable 格式化为紧凑字符串。"""
    if not parameters:
        return name
    return f"{name}({','.join(parameters)})"


def _format_action_set(atoms: tuple[GroundAtom, ...]) -> Action:
    """Format active action fluents as one planner action. / 将活动 action fluent 格式化为一个 planner action。"""
    return " + ".join(_format_ground_atom(name, parameters) for name, parameters in atoms)


def _format_atom_valuation(atoms: tuple[GroundAtom, ...]) -> str:
    """Format a boolean valuation as active ground atoms. / 将布尔 valuation 格式化为 active ground atom 集。"""
    if not atoms:
        return "{}"
    return "{" + ",".join(_format_ground_atom(name, parameters) for name, parameters in atoms) + "}"


def _split_top_level_commas(text: str) -> list[str]:
    """Split comma text while preserving function-call arguments. / 按顶层逗号切分并保留函数参数。"""
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _ground_atom_signature(text: str) -> tuple[str, tuple[str, ...]]:
    """Parse a compact grounded atom string. / 解析紧凑 grounded atom 字符串。"""
    name, parameters = _signature(str(text))
    return name, tuple(_object_name(parameter) for parameter in parameters)


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
            result.append(_object_name(parameters[0]))
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


def _ground_compact_tables(
    document: _RDDLDocument,
    states: tuple[State, ...],
    actions: tuple[Action, ...],
    state_fluent: str,
    objects: dict[str, tuple[str, ...]],
    action_pvariables: frozenset[str],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
) -> _GroundedTables:
    """Ground supported compact RDDL semantics into tables. / 将已支持的紧凑 RDDL 语义 grounding 成表。"""
    cpf = _state_cpf(document.domain, state_fluent)
    reward_text = _assignment(document.domain, "reward")
    if cpf is None:
        reward_constant = _constant_value(reward_text, default=0.0)
        return _GroundedTables(
            transitions=_identity_transitions(states, actions),
            rewards=_constant_rewards(states, actions, reward_constant),
            compiler_mode="structural-identity",
        )

    try:
        transition_expr = parse_expression(cpf.expression)
        reward_expr = parse_expression(reward_text or "0")
    except RDDLExpressionError as exc:
        raise RDDLCompileError(f"Unsupported RDDL expression: {exc}") from exc

    fluent_values = _non_fluent_values(document, objects)
    non_fluents = frozenset(
        key for key, value in fluent_values.items() if isinstance(value, bool) and value
    )
    transitions = _ground_state_transitions(
        states=states,
        actions=actions,
        state_fluent=state_fluent,
        cpf_parameter=cpf.parameter,
        expression=transition_expr,
        non_fluents=non_fluents,
        fluent_values=fluent_values,
        objects=objects,
        action_pvariables=action_pvariables,
        action_fluents_by_action=action_fluents,
    )
    rewards = _ground_rewards(
        states=states,
        actions=actions,
        state_fluent=state_fluent,
        expression=reward_expr,
        non_fluents=non_fluents,
        fluent_values=fluent_values,
        objects=objects,
        action_pvariables=action_pvariables,
        action_fluents_by_action=action_fluents,
    )
    return _GroundedTables(
        transitions=transitions,
        rewards=rewards,
        compiler_mode="grounded-rddl-expressions",
    )


def _ground_factored_model(
    *,
    document: _RDDLDocument,
    state_variables: list[_PVariable],
    action_variables: list[_PVariable],
    objects: dict[str, tuple[str, ...]],
    actions: tuple[Action, ...],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
    action_pvariables: frozenset[str],
) -> _FactoredGrounding:
    """Ground small boolean factored RDDL into explicit finite tables. / 将小型布尔 factored RDDL grounding 成显式有限表。"""
    state_atoms = _ground_pvariable_atoms(state_variables, objects)
    observ_variables = _observ_variables(_pvariables(document.domain))
    observation_atoms = _ground_pvariable_atoms(observ_variables, objects)
    state_pvariables = frozenset(pvar.name for pvar in state_variables)
    observation_pvariables = frozenset(pvar.name for pvar in observ_variables)
    fluent_values = _non_fluent_values(document, objects)
    non_fluents = frozenset(
        key for key, value in fluent_values.items() if isinstance(value, bool) and value
    )
    state_cpfs = _cpfs_by_name(document.domain, state_pvariables)
    missing_state_cpfs = [name for name in state_pvariables if name not in state_cpfs]
    if missing_state_cpfs:
        raise RDDLCompileError(f"Missing CPF(s) for state fluent(s): {missing_state_cpfs!r}.")
    observation_cpfs = _cpfs_by_name(document.domain, observation_pvariables)
    missing_observation_cpfs = [name for name in observation_pvariables if name not in observation_cpfs]
    if missing_observation_cpfs:
        raise RDDLCompileError(
            f"Missing CPF(s) for observ-fluent(s): {missing_observation_cpfs!r}."
        )

    states = _enumerate_atom_valuations(state_atoms)
    observations: tuple[Observation, ...]
    observations = _enumerate_atom_valuations(observation_atoms) if observation_atoms else states
    initial_state = _initial_factored_state(document.instance, state_atoms, state_variables)
    initial_belief = {state: (1.0 if state == initial_state else 0.0) for state in states}

    transition_exprs = {
        name: parse_expression(cpf.expression) for name, cpf in state_cpfs.items()
    }
    observation_exprs = {
        name: parse_expression(cpf.expression) for name, cpf in observation_cpfs.items()
    }
    reward_expr = parse_expression(_assignment(document.domain, "reward") or "0")

    transitions = _ground_factored_transitions(
        states=states,
        state_atoms=state_atoms,
        actions=actions,
        action_fluents=action_fluents,
        state_cpfs=state_cpfs,
        transition_exprs=transition_exprs,
        state_pvariables=state_pvariables,
        action_pvariables=action_pvariables,
        non_fluents=non_fluents,
        fluent_values=fluent_values,
        objects=objects,
    )
    rewards = _ground_factored_rewards(
        states=states,
        actions=actions,
        action_fluents=action_fluents,
        reward_expr=reward_expr,
        state_pvariables=state_pvariables,
        action_pvariables=action_pvariables,
        non_fluents=non_fluents,
        fluent_values=fluent_values,
        objects=objects,
    )
    if observation_atoms:
        observation_model = _ground_factored_observations(
            states=states,
            observations=observations,
            observation_atoms=observation_atoms,
            actions=actions,
            action_fluents=action_fluents,
            observation_cpfs=observation_cpfs,
            observation_exprs=observation_exprs,
            state_pvariables=state_pvariables,
            observation_pvariables=observation_pvariables,
            action_pvariables=action_pvariables,
            non_fluents=non_fluents,
            fluent_values=fluent_values,
            objects=objects,
        )
        reset_observation_model = _ground_reset_observations(
            states=states,
            observations=observations,
            observation_atoms=observation_atoms,
            observation_cpfs=observation_cpfs,
            observation_exprs=observation_exprs,
            state_pvariables=state_pvariables,
            observation_pvariables=observation_pvariables,
            action_pvariables=action_pvariables,
            non_fluents=non_fluents,
            fluent_values=fluent_values,
            objects=objects,
        )
    else:
        observation_model = _identity_observations(observations, actions)
        reset_observation_model = {
            (observation, state): 1.0 if observation == state else 0.0
            for observation in observations
            for state in states
        }
    return _FactoredGrounding(
        states=states,
        observations=observations,
        transitions=transitions,
        observation_model=observation_model,
        reset_observation_model=reset_observation_model,
        rewards=rewards,
        initial_belief=initial_belief,
    )


@dataclass(frozen=True)
class _StateCPF:
    """Store the compact state CPF expression and its object parameter. / 保存紧凑 state CPF 表达式及其对象参数。"""

    parameter: str
    expression: str


def _state_cpf(domain: RDDLASTNode, state_fluent: str) -> _StateCPF | None:
    """Return the CPF for one state fluent if present. / 返回某个 state fluent 的 CPF。"""
    block = _child_block(domain, "cpfs")
    if block is None:
        return None
    for statement in block.children:
        left, separator, expression = statement.label.partition("=")
        if not separator:
            continue
        name, parameters = _signature(left)
        if name.rstrip("'") == state_fluent:
            if len(parameters) != 1:
                raise RDDLCompileError(f"State CPF {name!r} must have one compact object parameter.")
            return _StateCPF(parameter=parameters[0], expression=expression.strip())
    return None


def _cpfs_by_name(domain: RDDLASTNode, names: frozenset[str]) -> dict[str, _CPF]:
    """Return CPF definitions keyed by fluent name. / 按 fluent 名称返回 CPF 定义。"""
    block = _child_block(domain, "cpfs")
    if block is None:
        return {}
    result: dict[str, _CPF] = {}
    for statement in block.children:
        left, separator, expression = statement.label.partition("=")
        if not separator:
            continue
        name, parameters = _signature(left.strip())
        base_name = name.rstrip("'")
        if base_name in names:
            result[base_name] = _CPF(
                name=base_name,
                parameters=parameters,
                expression=expression.strip(),
            )
    return result


def _ground_pvariable_atoms(
    pvariables: list[_PVariable], objects: dict[str, tuple[str, ...]]
) -> tuple[GroundAtom, ...]:
    """Ground pvariables into canonical atom signatures. / 将 pvariable grounding 成标准 atom 签名。"""
    atoms: list[GroundAtom] = []
    for pvar in pvariables:
        if not pvar.parameters:
            atoms.append((pvar.name, ()))
            continue
        domains: list[tuple[str, ...]] = []
        for parameter in pvar.parameters:
            if parameter not in objects:
                raise RDDLCompileError(
                    f"Parameter type {parameter!r} for {pvar.name!r} has no objects."
                )
            domains.append(objects[parameter])
        for values in product(*domains):
            atoms.append((pvar.name, tuple(values)))
    return tuple(atoms)


def _enumerate_atom_valuations(atoms: tuple[GroundAtom, ...]) -> tuple[str, ...]:
    """Enumerate active-atom tuple states for boolean fluents. / 枚举布尔 fluent 的 active-atom tuple 状态。"""
    states: list[str] = []
    for flags in product((False, True), repeat=len(atoms)):
        active = (
            (name, parameters)
            for (name, parameters), is_active in zip(atoms, flags, strict=True)
            if is_active
        )
        states.append(_format_atom_valuation(tuple(active)))
    return tuple(states)


def _initial_factored_state(
    instance: RDDLASTNode,
    state_atoms: tuple[GroundAtom, ...],
    state_variables: list[_PVariable],
) -> State:
    """Build the deterministic factored initial state from init-state. / 从 init-state 构建确定性 factored 初始状态。"""
    atom_set = set(state_atoms)
    values = {
        atom: bool(_default_for_atom(atom, state_variables))
        for atom in state_atoms
    }
    block = _child_block(instance, "init-state")
    if block is not None:
        for statement in block.children:
            left, separator, raw_value = statement.label.partition("=")
            name, parameters = _signature(left.strip())
            atom = (name, tuple(_object_name(parameter) for parameter in parameters))
            if atom not in atom_set:
                raise RDDLCompileError(f"Initial state atom {left.strip()!r} is not a state fluent.")
            values[atom] = bool(_literal_value(raw_value.strip())) if separator else True
    return _format_atom_valuation(
        tuple((name, parameters) for name, parameters in state_atoms if values[(name, parameters)])
    )


def _default_for_atom(atom: GroundAtom, pvariables: list[_PVariable]) -> ExpressionValue:
    """Return the declared default for one grounded atom. / 返回一个 grounded atom 的声明默认值。"""
    name, _parameters = atom
    for pvar in pvariables:
        if pvar.name == name:
            return pvar.default
    return False


def _atoms_from_state(state: State) -> frozenset[GroundAtom]:
    """Parse a factored state tuple into active atoms. / 将 factored state tuple 解析为 active atom。"""
    if isinstance(state, tuple):
        return frozenset(_ground_atom_signature(atom) for atom in state)
    text = str(state).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return frozenset()
    inner = text[1:-1].strip()
    if not inner:
        return frozenset()
    return frozenset(
        _ground_atom_signature(atom.strip()) for atom in _split_top_level_commas(inner) if atom.strip()
    )


def _context_for_state_action(
    *,
    state: State,
    action: Action,
    actions: tuple[Action, ...],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
    state_pvariables: frozenset[str],
    action_pvariables: frozenset[str],
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
    variables: dict[str, str] | None = None,
) -> EvaluationContext:
    """Build an expression context for factored grounding. / 为 factored grounding 构建表达式上下文。"""
    return EvaluationContext(
        state_fluent="",
        current_state=state,
        action=action,
        actions=actions,
        non_fluents=non_fluents,
        variables=variables or {},
        fluent_values=fluent_values,
        objects=objects,
        state_atoms=_atoms_from_state(state),
        state_pvariables=state_pvariables,
        action_fluents=action_fluents.get(action, frozenset()),
        action_pvariables=action_pvariables,
    )


def _ground_factored_transitions(
    *,
    states: tuple[State, ...],
    state_atoms: tuple[GroundAtom, ...],
    actions: tuple[Action, ...],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
    state_cpfs: Mapping[str, _CPF],
    transition_exprs: Mapping[str, Any],
    state_pvariables: frozenset[str],
    action_pvariables: frozenset[str],
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
) -> dict[TransitionKey, float]:
    """Ground independent boolean state CPFs into transition probabilities. / 将独立布尔 state CPF grounding 为转移概率。"""
    transitions: dict[TransitionKey, float] = {}
    for source in states:
        for action in actions:
            true_probabilities = {
                atom: _factored_atom_true_probability(
                    atom,
                    state_cpfs,
                    transition_exprs,
                    _context_for_state_action(
                        state=source,
                        action=action,
                        actions=actions,
                        action_fluents=action_fluents,
                        state_pvariables=state_pvariables,
                        action_pvariables=action_pvariables,
                        non_fluents=non_fluents,
                        fluent_values=fluent_values,
                        objects=objects,
                        variables=_variables_for_atom(atom, state_cpfs[atom[0]]),
                    ),
                )
                for atom in state_atoms
            }
            for target in states:
                target_atoms = _atoms_from_state(target)
                probability = 1.0
                for atom, true_probability in true_probabilities.items():
                    probability *= true_probability if atom in target_atoms else 1.0 - true_probability
                transitions[(source, action, target)] = probability
    return transitions


def _ground_factored_rewards(
    *,
    states: tuple[State, ...],
    actions: tuple[Action, ...],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
    reward_expr: Any,
    state_pvariables: frozenset[str],
    action_pvariables: frozenset[str],
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
) -> dict[RewardKey, float]:
    """Ground reward over factored states and action sets. / 在 factored state 和 action 集上 grounding reward。"""
    rewards: dict[RewardKey, float] = {}
    for state in states:
        for action in actions:
            context = _context_for_state_action(
                state=state,
                action=action,
                actions=actions,
                action_fluents=action_fluents,
                state_pvariables=state_pvariables,
                action_pvariables=action_pvariables,
                non_fluents=non_fluents,
                fluent_values=fluent_values,
                objects=objects,
            )
            rewards[(state, action)] = float(reward_expr.evaluate(context))
    return rewards


def _ground_factored_observations(
    *,
    states: tuple[State, ...],
    observations: tuple[Observation, ...],
    observation_atoms: tuple[GroundAtom, ...],
    actions: tuple[Action, ...],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
    observation_cpfs: Mapping[str, _CPF],
    observation_exprs: Mapping[str, Any],
    state_pvariables: frozenset[str],
    observation_pvariables: frozenset[str],
    action_pvariables: frozenset[str],
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
) -> dict[ObservationKey, float]:
    """Ground observation CPFs into P(observation | state, action). / 将观测 CPF grounding 成 P(observation | state, action)。"""
    model: dict[ObservationKey, float] = {}
    for state in states:
        for action in actions:
            true_probabilities = _observation_true_probabilities(
                state=state,
                action=action,
                actions=actions,
                action_fluents=action_fluents,
                observation_atoms=observation_atoms,
                observation_cpfs=observation_cpfs,
                observation_exprs=observation_exprs,
                state_pvariables=state_pvariables,
                observation_pvariables=observation_pvariables,
                action_pvariables=action_pvariables,
                non_fluents=non_fluents,
                fluent_values=fluent_values,
                objects=objects,
            )
            for observation in observations:
                model[(observation, state, action)] = _factored_observation_probability(
                    observation, true_probabilities
                )
    return model


def _ground_reset_observations(
    *,
    states: tuple[State, ...],
    observations: tuple[Observation, ...],
    observation_atoms: tuple[GroundAtom, ...],
    observation_cpfs: Mapping[str, _CPF],
    observation_exprs: Mapping[str, Any],
    state_pvariables: frozenset[str],
    observation_pvariables: frozenset[str],
    action_pvariables: frozenset[str],
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
) -> dict[tuple[Observation, State], float]:
    """Ground reset observation probabilities without active actions. / 在无活动 action 下 grounding reset observation 概率。"""
    model: dict[tuple[Observation, State], float] = {}
    reset_action = "__reset__"
    for state in states:
        true_probabilities = _observation_true_probabilities(
            state=state,
            action=reset_action,
            actions=(reset_action,),
            action_fluents={reset_action: frozenset()},
            observation_atoms=observation_atoms,
            observation_cpfs=observation_cpfs,
            observation_exprs=observation_exprs,
            state_pvariables=state_pvariables,
            observation_pvariables=observation_pvariables,
            action_pvariables=action_pvariables,
            non_fluents=non_fluents,
            fluent_values=fluent_values,
            objects=objects,
        )
        for observation in observations:
            model[(observation, state)] = _factored_observation_probability(
                observation, true_probabilities
            )
    return model


def _observation_true_probabilities(
    *,
    state: State,
    action: Action,
    actions: tuple[Action, ...],
    action_fluents: Mapping[Action, frozenset[GroundAtom]],
    observation_atoms: tuple[GroundAtom, ...],
    observation_cpfs: Mapping[str, _CPF],
    observation_exprs: Mapping[str, Any],
    state_pvariables: frozenset[str],
    observation_pvariables: frozenset[str],
    action_pvariables: frozenset[str],
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
) -> dict[GroundAtom, float]:
    """Evaluate every observation atom's true probability. / 求值每个 observation atom 为真的概率。"""
    return {
        atom: _factored_atom_true_probability(
            atom,
            observation_cpfs,
            observation_exprs,
            _context_for_state_action(
                state=state,
                action=action,
                actions=actions,
                action_fluents=action_fluents,
                state_pvariables=state_pvariables | observation_pvariables,
                action_pvariables=action_pvariables,
                non_fluents=non_fluents,
                fluent_values=fluent_values,
                objects=objects,
                variables=_variables_for_atom(atom, observation_cpfs[atom[0]]),
            ),
        )
        for atom in observation_atoms
    }


def _factored_atom_true_probability(
    atom: GroundAtom,
    cpfs: Mapping[str, _CPF],
    expressions: Mapping[str, Any],
    context: EvaluationContext,
) -> float:
    """Evaluate a boolean CPF as a true probability. / 将布尔 CPF 求值为 true 概率。"""
    name, _parameters = atom
    if name not in cpfs:
        raise RDDLCompileError(f"Missing CPF for fluent {name!r}.")
    value = expressions[name].evaluate(context)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        probability = float(value)
        if probability < -1e-12 or probability > 1.0 + 1e-12:
            raise RDDLCompileError(
                f"CPF for atom {_format_ground_atom(*atom)!r} returned probability {probability}."
            )
        return min(1.0, max(0.0, probability))
    return 1.0 if _object_name(value).lower() == "true" else 0.0


def _variables_for_atom(atom: GroundAtom, cpf: _CPF) -> dict[str, str]:
    """Bind CPF parameters to one grounded atom. / 将 CPF 参数绑定到一个 grounded atom。"""
    _name, parameters = atom
    if len(parameters) != len(cpf.parameters):
        raise RDDLCompileError(
            f"CPF {cpf.name!r} expects {len(cpf.parameters)} parameter(s), "
            f"but atom has {len(parameters)}."
        )
    return {variable: value for variable, value in zip(cpf.parameters, parameters, strict=True)}


def _factored_observation_probability(
    observation: Observation, true_probabilities: Mapping[GroundAtom, float]
) -> float:
    """Return probability for one active-observation tuple. / 返回某个 active-observation tuple 的概率。"""
    active_atoms = _atoms_from_state(observation)
    probability = 1.0
    for atom, true_probability in true_probabilities.items():
        probability *= true_probability if atom in active_atoms else 1.0 - true_probability
    return probability


def _non_fluent_values(
    document: _RDDLDocument, objects: dict[str, tuple[str, ...]]
) -> dict[tuple[str, tuple[str, ...]], ExpressionValue]:
    """Parse ground non-fluent values from the non-fluents block. / 从 non-fluents 块解析 ground non-fluent 值。"""
    if document.non_fluents is None:
        return {}
    block = _child_block(document.non_fluents, "non-fluents")
    if block is None:
        return {}
    values: dict[tuple[str, tuple[str, ...]], ExpressionValue] = {}
    for statement in block.children:
        left, separator, raw_value = statement.label.partition("=")
        name, parameters = _signature(left.strip())
        key = (name, tuple(_object_name(parameter) for parameter in parameters))
        if separator:
            values[key] = _evaluate_static_expression(raw_value.strip(), values, objects)
        else:
            values[key] = True
    return values


def _evaluate_static_expression(
    text: str,
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
) -> ExpressionValue:
    """Evaluate a non-state expression used in non-fluent declarations. / 求值 non-fluent 声明中的非状态表达式。"""
    try:
        expression = parse_expression(text)
        return expression.evaluate(
            EvaluationContext(
                state_fluent="",
                current_state="",
                action="",
                actions=(),
                non_fluents=frozenset(
                    key for key, value in fluent_values.items() if isinstance(value, bool) and value
                ),
                variables={},
                fluent_values=fluent_values,
                objects=objects,
            )
        )
    except RDDLExpressionError as exc:
        raise RDDLCompileError(f"Unsupported non-fluent expression {text!r}: {exc}") from exc


def _ground_state_transitions(
    *,
    states: tuple[State, ...],
    actions: tuple[Action, ...],
    state_fluent: str,
    cpf_parameter: str,
    expression: Any,
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
    action_pvariables: frozenset[str],
    action_fluents_by_action: Mapping[Action, frozenset[GroundAtom]],
) -> dict[TransitionKey, float]:
    """Evaluate one compact state CPF into transition probabilities. / 将一个紧凑 state CPF 求值为转移概率。"""
    transitions = {(source, action, target): 0.0 for source in states for action in actions for target in states}
    for source in states:
        for action in actions:
            probabilities: list[tuple[State, float]] = []
            for target in states:
                context = EvaluationContext(
                    state_fluent=state_fluent,
                    current_state=source,
                    action=action,
                    actions=actions,
                    non_fluents=non_fluents,
                    variables={cpf_parameter: str(target)},
                    fluent_values=fluent_values,
                    objects=objects,
                    state_atoms=frozenset({(state_fluent, (str(source),))}),
                    state_pvariables=frozenset({state_fluent}),
                    action_fluents=action_fluents_by_action.get(action, frozenset()),
                    action_pvariables=action_pvariables,
                )
                probabilities.append((target, _transition_probability(expression.evaluate(context), target)))
            total_probability = sum(probability for _, probability in probabilities)
            if abs(total_probability - 1.0) > 1e-8:
                raise RDDLCompileError(
                    f"State CPF for source={source!r}, action={action!r} produced "
                    f"transition probability mass {total_probability:.6g}; expected 1.0."
                )
            for target, probability in probabilities:
                transitions[(source, action, target)] = probability
    return transitions


def _transition_probability(value: ExpressionValue, target: State) -> float:
    """Convert a CPF value for one target into probability mass. / 将某个 target 的 CPF 值转换为概率质量。"""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        probability = float(value)
        if probability < -1e-12:
            raise RDDLCompileError(f"Transition probability cannot be negative: {probability}.")
        return max(0.0, probability)
    return 1.0 if str(target) == _object_name(value) else 0.0


def _ground_rewards(
    *,
    states: tuple[State, ...],
    actions: tuple[Action, ...],
    state_fluent: str,
    expression: Any,
    non_fluents: frozenset[tuple[str, tuple[str, ...]]],
    fluent_values: dict[tuple[str, tuple[str, ...]], ExpressionValue],
    objects: dict[str, tuple[str, ...]],
    action_pvariables: frozenset[str],
    action_fluents_by_action: Mapping[Action, frozenset[GroundAtom]],
) -> dict[RewardKey, float]:
    """Evaluate the reward expression for every compact state-action pair. / 对每个紧凑 state-action 对求值 reward。"""
    rewards: dict[RewardKey, float] = {}
    for state in states:
        for action in actions:
            context = EvaluationContext(
                state_fluent=state_fluent,
                current_state=state,
                action=action,
                actions=actions,
                non_fluents=non_fluents,
                variables={},
                fluent_values=fluent_values,
                objects=objects,
                state_atoms=frozenset({(state_fluent, (str(state),))}),
                state_pvariables=frozenset({state_fluent}),
                action_fluents=action_fluents_by_action.get(action, frozenset()),
                action_pvariables=action_pvariables,
            )
            rewards[(state, action)] = float(expression.evaluate(context))
    return rewards


def _constant_value(text: str | None, *, default: float) -> float:
    """Parse one optional numeric constant. / 解析一个可选数字常量。"""
    if text is None:
        return default
    try:
        return float(text)
    except ValueError:
        return default


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
