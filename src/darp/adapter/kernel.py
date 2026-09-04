"""Sparse floating-point kernel over pyRDDLGym grounded expressions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import isfinite, prod
from typing import Any

StateKey = tuple[tuple[str, Hashable], ...]
StateSelector = tuple[tuple[str, bool | int], ...]
ObservationKey = tuple[tuple[str, Hashable], ...]
Distribution = dict[Hashable, float]
ActionKey = tuple[tuple[str, Hashable], ...]


@dataclass(frozen=True, slots=True)
class SparseTransitionRow:
    """Store one cached sparse row of $$T_a$$."""

    next_state_ids: tuple[int, ...]
    probabilities: tuple[float, ...]


@dataclass(slots=True)
class _LazyStateIndex:
    """Assign compact ids only to states reached by search. / 只为搜索触达的状态分配紧凑编号。"""

    key_to_id: dict[StateKey, int] = field(default_factory=dict)
    id_to_key: list[StateKey] = field(default_factory=list)

    def register(self, state: StateKey) -> int:
        """Return an existing id or append one discovered state. / 返回已有编号或登记新发现状态。"""
        state_id = self.key_to_id.get(state)
        if state_id is not None:
            return state_id
        state_id = len(self.id_to_key)
        self.key_to_id[state] = state_id
        self.id_to_key.append(state)
        return state_id

    def key(self, state_id: int) -> StateKey:
        """Return the state key for an integer id. / 返回整数编号对应的状态键。"""
        return self.id_to_key[state_id]


class KernelError(ValueError):
    """Raised when finite-kernel evaluation is unsupported. / 有限内核求值不支持时抛出。"""


@dataclass(frozen=True, slots=True)
class RiskConstraintSpec:
    r"""Describe the paper CC-POMDP tuple :math:`\langle R,\Delta\rangle`.

    Each partial selector is a conjunction of grounded fluent equalities; the
    selectors form their union.  A state is risky when it matches any selector.
    / 每个部分状态 selector 内部为 AND，多个 selector 之间为 OR，共同定义风险集
    :math:`R`。
    """

    budget: float | None = None
    risky_states: tuple[StateSelector, ...] = ()


@dataclass(frozen=True)
class ConstraintMassOutcome:
    """One observation branch of an unnormalized constraint flow."""

    observation: ObservationKey
    label: str
    state_mass: Mapping[StateKey, float]


@dataclass(frozen=True)
class ConstraintMassExpansion:
    """Sparse float mass propagation for one constraint action."""

    coefficient: float
    post_action_mass: Mapping[StateKey, float]
    observations: tuple[ConstraintMassOutcome, ...]


@dataclass(frozen=True)
class RDDLKernel:
    """Lazily compile reached grounded states into sparse float kernels. / 将触达的 grounded 状态按需编译为稀疏浮点内核。"""

    grounded_model: Any
    risk: RiskConstraintSpec = field(default_factory=RiskConstraintSpec)
    _state_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _action_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _observation_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _intermediate_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _non_fluents_cache: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _cpfs_cache: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _terminations_cache: tuple[Any, ...] = field(default=(), init=False, repr=False, compare=False)
    _state_index: _LazyStateIndex = field(default_factory=_LazyStateIndex, init=False, repr=False, compare=False)
    _action_ids: dict[ActionKey, int] = field(default_factory=dict, init=False, repr=False, compare=False)
    _transition_rows: dict[tuple[int, int], SparseTransitionRow] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _reward_cache: dict[tuple[int, int], float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _state_failure_cache: dict[int, float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _transition_risk_rows: dict[tuple[int, int], float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _observation_cache: dict[tuple[int, int], Mapping[ObservationKey, float]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    def __post_init__(self) -> None:
        """Freeze grounded metadata reused by every numeric evaluation. / 固定每次数值求值都会复用的 grounded 元数据。"""
        object.__setattr__(
            self,
            "_state_names_cache",
            tuple(sorted(_mapping_keys(getattr(self.grounded_model, "state_fluents", None)))),
        )
        object.__setattr__(
            self,
            "_action_names_cache",
            tuple(sorted(_mapping_keys(getattr(self.grounded_model, "action_fluents", None)))),
        )
        object.__setattr__(
            self,
            "_observation_names_cache",
            tuple(sorted(_mapping_keys(getattr(self.grounded_model, "observ_fluents", None)))),
        )
        levels = getattr(self.grounded_model, "cpf_to_level", {}) or {}
        object.__setattr__(
            self,
            "_intermediate_names_cache",
            tuple(
                sorted(
                    _mapping_keys(getattr(self.grounded_model, "interm_fluents", None)),
                    key=lambda name: (levels.get(name, 1), name),
                )
            ),
        )
        non_fluents = getattr(self.grounded_model, "non_fluents", None)
        cpfs = getattr(self.grounded_model, "cpfs", None)
        object.__setattr__(self, "_non_fluents_cache", non_fluents if isinstance(non_fluents, Mapping) else {})
        object.__setattr__(self, "_cpfs_cache", cpfs if isinstance(cpfs, Mapping) else {})
        object.__setattr__(
            self,
            "_terminations_cache",
            tuple(getattr(self.grounded_model, "terminations", ()) or ()),
        )

    @classmethod
    def from_grounded_model(
        cls,
        grounded_model: Any,
        *,
        risk: RiskConstraintSpec | None = None,
    ) -> RDDLKernel:
        """Build a sparse kernel from a pyRDDLGym grounded model. / 从 pyRDDLGym grounded model 构建稀疏内核。"""
        kernel = cls(grounded_model=grounded_model, risk=risk or RiskConstraintSpec())
        kernel._validate_supported()
        return kernel

    @property
    def state_names(self) -> tuple[str, ...]:
        """Return deterministic grounded state fluent names. / 返回确定性的 grounded state fluent 名称。"""
        return self._state_names_cache

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return deterministic grounded action fluent names. / 返回确定性的 grounded action fluent 名称。"""
        return self._action_names_cache

    @property
    def observation_names(self) -> tuple[str, ...]:
        """Return deterministic grounded observation fluent names. / 返回确定性的 grounded observation fluent 名称。"""
        return self._observation_names_cache

    @property
    def non_fluents(self) -> Mapping[str, Any]:
        """Return grounded non-fluent values. / 返回 grounded non-fluent 值。"""
        return self._non_fluents_cache

    @property
    def cpfs(self) -> Mapping[str, Any]:
        """Return grounded CPF expressions. / 返回 grounded CPF 表达式。"""
        return self._cpfs_cache

    def initial_belief_from_state(self, state: Mapping[str, Any]) -> Mapping[StateKey, float]:
        """Return a singleton belief from a pyRDDLGym state dict. / 从 pyRDDLGym state dict 返回单点 belief。"""
        state_key = self.state_key(state)
        self._state_index.register(state_key)
        return {state_key: 1.0}

    def initial_constraint_mass(
        self,
        belief: Mapping[StateKey, float],
    ) -> Mapping[StateKey, float]:
        """Return a normalized sparse mass for the initial belief."""
        mass = _normalized_mass(belief)
        for state in mass:
            self._state_index.register(state)
        return mass

    def initial_safe_mass(
        self,
        belief: Mapping[StateKey, float],
    ) -> Mapping[StateKey, float]:
        """Return unnormalized root mass that has survived initial failure."""
        ordinary = self.initial_constraint_mass(belief)
        safe: dict[StateKey, float] = {}
        for state, probability in ordinary.items():
            surviving = probability * (1.0 - self.state_failure(state))
            if surviving > 0:
                safe[state] = surviving
        return safe

    @staticmethod
    def constraint_mass_belief(
        mass: Mapping[StateKey, float],
    ) -> Mapping[StateKey, float]:
        """Normalize an unnormalized mass into a conditional belief."""
        return _mass_belief(mass)

    def initial_belief_from_model(self) -> Mapping[StateKey, float]:
        """Return the declared deterministic RDDL initial belief.

        This reads the grounded model declaration instead of the simulator's
        current hidden state. Standard RDDL initial assignments are
        deterministic; a non-degenerate ``b0`` needs an explicit belief
        adapter rather than access to simulator state.

        / 从模型声明构造确定性初始 belief，不读取模拟器隐藏真状态；非退化
        ``b0`` 应由显式 belief adapter 提供。
        """
        declared = getattr(self.grounded_model, "state_fluents", None)
        if not isinstance(declared, Mapping):
            raise KernelError("Grounded model does not expose a declared initial state.")
        return self.initial_belief_from_state(declared)

    def belief_is_terminal(self, belief: Mapping[StateKey, float]) -> bool:
        """Match RDDL termination semantics for every positive-support state."""
        if not self._terminations_cache:
            return False
        states = tuple(state for state, probability in belief.items() if probability > 0)
        return bool(states) and all(self._state_is_terminal(state) for state in states)

    def _state_is_terminal(self, state: StateKey) -> bool:
        context = self._context(self.state_from_key(state), {})
        for expression in self._terminations_cache:
            distribution = self.expression_distribution(expression, context)
            if len(distribution) != 1:
                raise KernelError("RDDL termination expressions must be deterministic.")
            if bool(next(iter(distribution))):
                return True
        return False

    def state_key(self, state: Mapping[str, Any]) -> StateKey:
        """Convert a state mapping to a stable key. / 将 state mapping 转成稳定 key。"""
        return tuple((name, _plain_value(state.get(name, False))) for name in self.state_names)

    def state_from_key(self, key: StateKey) -> dict[str, Any]:
        """Convert a state key back to a mapping. / 将 state key 转回 mapping。"""
        return dict(key)

    def state_label(self, key: StateKey) -> str:
        """Return the existing compact state label for a key. / 返回已有的紧凑 state label。"""
        state = self.state_from_key(key)
        active = [str(name) for name, value in state.items() if value is True]
        if active:
            return ",".join(active)
        if not state:
            return "(empty)"
        return repr(state)

    def expand_ordinary_mass(
        self,
        state_mass: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> ConstraintMassExpansion:
        """Propagate ordinary history mass through sparse transition rows."""
        action_id = self._action_id(action)
        post_action_mass = self._transition_mass(state_mass, action_id, action)
        return ConstraintMassExpansion(
            coefficient=0.0,
            post_action_mass=post_action_mass,
            observations=self._constraint_mass_observations(post_action_mass, action),
        )

    def expand_safe_constraint_mass(
        self,
        safe_mass: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> ConstraintMassExpansion:
        """Propagate Lemma 3.3 unnormalized safe-prefix mass."""
        action_id = self._action_id(action)
        post_action_mass: dict[StateKey, float] = {}
        for source, target, transition_mass in self._transition_branches(
            safe_mass, action_id, action
        ):
            failure = self.transition_failure(source, target, action)
            if failure <= 0.0:
                surviving = transition_mass
            elif failure >= 1.0:
                continue
            else:
                surviving = transition_mass * (1.0 - failure)
            if surviving > 0:
                post_action_mass[target] = post_action_mass.get(target, 0.0) + surviving
        return ConstraintMassExpansion(
            coefficient=self.safe_constraint_coefficient_for_mass(safe_mass, action),
            post_action_mass=post_action_mass,
            observations=self._constraint_mass_observations(post_action_mass, action),
        )

    def safe_constraint_coefficient_for_mass(
        self,
        safe_mass: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> float:
        """Return Lemma 3.3's first-entry risk without materializing children.

        An unselected HILP frontier needs :math:`r_q` in the global risk row,
        but does not yet need surviving post-action mass. Cached state-action
        risk rows avoid rescanning the same transition branches at every
        frontier history.
        """
        action_id = self._action_id(action)
        return _probability(
            sum(
                mass
                * self._transition_risk_row(
                    self._state_index.register(state),
                    action_id,
                    action,
                )
                for state, mass in safe_mass.items()
            )
        )

    def _transition_mass(
        self,
        state_mass: Mapping[StateKey, float],
        action_id: int,
        action: Mapping[str, Any],
    ) -> Mapping[StateKey, float]:
        """Apply cached transition rows to sparse state mass."""
        result: dict[StateKey, float] = {}
        for _, target, transition_mass in self._transition_branches(
            state_mass, action_id, action
        ):
            result[target] = result.get(target, 0.0) + transition_mass
        return result

    def _transition_branches(
        self,
        state_mass: Mapping[StateKey, float],
        action_id: int,
        action: Mapping[str, Any],
    ) -> Iterable[tuple[StateKey, StateKey, float]]:
        """Yield positive transition mass while reusing cached rows."""
        for source, source_mass in state_mass.items():
            if source_mass <= 0.0:
                continue
            row = self._transition_row(
                self._state_index.register(source),
                action_id,
                action,
            )
            if not row.probabilities:
                continue
            for target_id, probability in zip(
                row.next_state_ids,
                row.probabilities,
            ):
                if probability <= 0:
                    continue
                target = self._state_index.key(int(target_id))
                yield source, target, source_mass * probability

    def _constraint_mass_observations(
        self,
        post_action_mass: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> tuple[ConstraintMassOutcome, ...]:
        """Split unnormalized state mass by cached observation rows."""
        if not self.observation_names:
            return tuple(
                ConstraintMassOutcome(
                    observation=(("__state__", state),),
                    label=self.state_label(state),
                    state_mass={state: mass},
                )
                for state, mass in sorted(post_action_mass.items(), key=lambda item: repr(item[0]))
                if mass > 0
            )

        buckets: dict[ObservationKey, dict[StateKey, float]] = {}
        for state, mass in post_action_mass.items():
            distribution = self._observation_distribution_for_state(state, action)
            for observation, probability in distribution.items():
                if probability <= 0:
                    continue
                bucket = buckets.setdefault(observation, {})
                bucket[state] = bucket.get(state, 0.0) + mass * probability
        return tuple(
            ConstraintMassOutcome(
                observation=observation,
                label=_observation_label(observation),
                state_mass=state_weights,
            )
            for observation, state_weights in sorted(
                buckets.items(),
                key=lambda item: repr(item[0]),
            )
        )

    def belief_state_risk(
        self,
        belief: Mapping[StateKey, float],
    ) -> float:
        """Return initial/root state risk."""
        mass = _normalized_mass(belief)
        return _probability(
            sum(
                probability * self.state_failure(state)
                for state, probability in mass.items()
            )
        )

    def state_failure(self, state: StateKey) -> float:
        """Return the indicator that ``state`` belongs to the risky set."""
        state_id = self._state_index.register(state)
        cached = self._state_failure_cache.get(state_id)
        if cached is not None:
            return cached
        state_mapping = self.state_from_key(state)
        failure = float(
            any(
                _state_matches_selector(state_mapping, selector)
                for selector in self.risk.risky_states
            )
        )
        self._state_failure_cache[state_id] = failure
        return failure

    def transition_failure(
        self,
        source: StateKey,
        target: StateKey,
        action: Mapping[str, Any],
    ) -> float:
        """Return whether the transition enters the paper's risky-state set."""
        del source, action
        return self.state_failure(target)

    def _transition_risk_row(
        self,
        source_id: int,
        action_id: int,
        action: Mapping[str, Any],
    ) -> float:
        """Return cached expected first-entry risk for one state-action row."""
        cache_key = (source_id, action_id)
        cached = self._transition_risk_rows.get(cache_key)
        if cached is not None:
            return cached
        source = self._state_index.key(source_id)
        row = self._transition_row(source_id, action_id, action)
        risk = _probability(
            sum(
                probability
                * self.transition_failure(
                    source,
                    self._state_index.key(int(target_id)),
                    action,
                )
                for target_id, probability in zip(
                    row.next_state_ids,
                    row.probabilities,
                )
            )
        )
        self._transition_risk_rows[cache_key] = risk
        return risk

    def transition_distribution(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> Mapping[StateKey, float]:
        """Return one sparse transition row."""
        source_id = self._state_index.register(self.state_key(state))
        row = self._transition_row(source_id, self._action_id(action), action)
        return {
            self._state_index.key(int(next_id)): probability
            for next_id, probability in zip(row.next_state_ids, row.probabilities)
            if probability > 0
        }

    def _transition_row(
        self,
        source_id: int,
        action_id: int,
        action: Mapping[str, Any],
    ) -> SparseTransitionRow:
        r"""Return cached $$T(s,a,\cdot)$$ and discover only its successors. / 返回缓存的转移行并仅发现其后继状态。"""
        cache_key = (source_id, action_id)
        cached = self._transition_rows.get(cache_key)
        if cached is not None:
            return cached
        state = self.state_from_key(self._state_index.key(source_id))
        partials: dict[StateKey, float] = {}
        for context, context_probability in self._intermediate_contexts(
            self._context(state, action)
        ):
            context_partials: dict[StateKey, float] = {(): context_probability}
            for state_name in self.state_names:
                expr = self._state_cpf_expression(state_name)
                # All next-state CPFs see the same sampled intermediates.
                # / 所有下一状态 CPF 共享同一次采样得到的中间变量。
                value_weights = _normalize_distribution(
                    self.expression_distribution(expr, context)
                )
                updated: dict[StateKey, float] = {}
                for partial_key, partial_prob in context_partials.items():
                    partial_state = dict(partial_key)
                    for value, value_prob in value_weights.items():
                        next_partial = tuple(
                            sorted(
                                {
                                    **partial_state,
                                    state_name: _plain_value(value),
                                }.items()
                            )
                        )
                        updated[next_partial] = (
                            updated.get(next_partial, 0.0)
                            + partial_prob * value_prob
                        )
                context_partials = updated
            for partial_key, probability in context_partials.items():
                partials[partial_key] = partials.get(partial_key, 0.0) + probability
        distribution = _normalize_distribution(partials)
        row = SparseTransitionRow(
            next_state_ids=tuple(
                self._state_index.register(state_key) for state_key in distribution
            ),
            probabilities=tuple(distribution.values()),
        )
        self._transition_rows[cache_key] = row
        return row

    def _intermediate_contexts(
        self,
        context: Mapping[str, Any],
    ) -> tuple[tuple[Mapping[str, Any], float], ...]:
        """Evaluate intermediate CPFs once and share their random values.

        / 对中间 CPF 只求值一次并共享其随机结果。
        """
        branches: list[tuple[Mapping[str, Any], float]] = [(context, 1.0)]
        for name in self._intermediate_names_cache:
            expression = _cpf_expression(self.cpfs[name])
            following: list[tuple[Mapping[str, Any], float]] = []
            for branch, branch_probability in branches:
                values = _normalize_distribution(
                    self.expression_distribution(expression, branch)
                )
                following.extend(
                    (
                        {**branch, name: _plain_value(value)},
                        branch_probability * probability,
                    )
                    for value, probability in values.items()
                )
            branches = following
        return tuple(branches)

    def observation_probability(
        self,
        observation: ObservationKey,
        state: StateKey,
        action: Mapping[str, Any],
    ) -> float:
        """Return one observation likelihood."""
        if observation and observation[0][0] == "__state__":
            return 1.0 if observation[0][1] == state else 0.0
        return float(
            self._observation_distribution_for_state(state, action).get(observation, 0.0)
        )

    def backward_message(
        self,
        current_states: Mapping[StateKey, float],
        next_message: Mapping[StateKey, float],
        action: Mapping[str, Any],
        observation: ObservationKey,
    ) -> Mapping[StateKey, float]:
        """Apply Algorithm 2's backward operator."""
        action_id = self._action_id(action)
        result: dict[StateKey, float] = {}
        for state in current_states:
            row = self._transition_row(
                self._state_index.register(state),
                action_id,
                action,
            )
            probability_of_future = sum(
                (
                    transition_probability
                    * self.observation_probability(
                        observation,
                        self._state_index.key(int(target_id)),
                        action,
                    )
                    * next_message.get(self._state_index.key(int(target_id)), 0.0)
                )
                for target_id, transition_probability in zip(
                    row.next_state_ids,
                    row.probabilities,
                )
            )
            result[state] = probability_of_future
        return result

    def utility_coefficient_for_mass(
        self,
        state_mass: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> float:
        """Return ``sum_s mass(s) U(s,a)``."""
        action_id = self._action_id(action)
        value = sum(
            mass
            * self._reward_for_ids(
                self._state_index.register(state),
                action_id,
                action,
            )
            for state, mass in state_mass.items()
        )
        if not isfinite(value):
            raise KernelError("Utility coefficient must be finite.")
        return value

    def expected_reward(self, context: Mapping[str, Any]) -> float:
        """Return the expectation of finite reward support."""
        reward = getattr(self.grounded_model, "reward", None)
        if reward is None:
            raise KernelError("Grounded model does not expose a reward expression.")
        return _expectation(self.expression_distribution(reward, context))

    def expression_distribution(self, expr: Any, context: Mapping[str, Any]) -> Distribution:
        """Evaluate a grounded expression into a finite distribution. / 将 grounded expression 求值为有限分布。"""
        if not _is_expression(expr):
            return {_plain_value(expr): 1.0}
        etype, op = expr.etype
        args = expr.args
        if etype == "constant":
            return {_plain_value(args): 1.0}
        if etype == "pvar":
            name, params = args
            if params not in (None, []):
                raise KernelError(f"Kernel expected grounded pvar but got {name}{params}.")
            if name not in context:
                raise KernelError(f"Expression references unknown grounded pvar: {name}")
            return {_plain_value(context[name]): 1.0}
        if etype == "arithmetic":
            return _combine_distributions([self.expression_distribution(arg, context) for arg in _as_args(args)], _arith(op))
        if etype == "boolean":
            return _combine_distributions([self.expression_distribution(arg, context) for arg in _as_args(args)], _logic(op))
        if etype == "relational":
            return _combine_distributions([self.expression_distribution(arg, context) for arg in _as_args(args)], _relation(op))
        if etype == "control" and op == "if":
            condition, then_expr, else_expr = args
            result: dict[Hashable, float] = {}
            for truth, truth_prob in self.expression_distribution(condition, context).items():
                branch = then_expr if bool(truth) else else_expr
                for value, value_prob in self.expression_distribution(branch, context).items():
                    result[value] = (
                        result.get(value, 0.0)
                        + truth_prob * value_prob
                    )
            return _normalize_distribution(result)
        if etype == "randomvar":
            return self._random_distribution(op, _as_args(args), context)
        if etype == "aggregation":
            raise KernelError(f"Grounded aggregation is not implemented for operator {op}.")
        raise KernelError(f"Unsupported expression type: {etype}/{op}")

    def _random_distribution(
        self,
        name: str,
        args: Sequence[Any],
        context: Mapping[str, Any],
    ) -> Distribution:
        """Return finite distribution for supported random expressions. / 返回受支持随机表达式的有限分布。"""
        if name in {"KronDelta", "DiracDelta"}:
            _check_arity(args, 1, name)
            return self.expression_distribution(args[0], context)
        if name == "Bernoulli":
            _check_arity(args, 1, name)
            p = _expectation(self.expression_distribution(args[0], context))
            if not isfinite(p) or p < 0 or p > 1:
                raise KernelError(f"Bernoulli probability out of range: {p}")
            return _normalize_distribution({True: p, False: 1.0 - p})
        if name == "Discrete":
            if len(args) % 2 != 0:
                raise KernelError("Discrete expects alternating value/probability arguments.")
            result: dict[Hashable, float] = {}
            for value_expr, prob_expr in zip(args[0::2], args[1::2]):
                value_dist = self.expression_distribution(value_expr, context)
                if len(value_dist) != 1:
                    raise KernelError("Discrete values must be deterministic.")
                value = next(iter(value_dist))
                prob_value = _expectation(self.expression_distribution(prob_expr, context))
                result[value] = result.get(value, 0.0) + prob_value
            return _normalize_distribution(result)
        raise KernelError(f"Random distribution {name} is not finite in current DARP.")

    def _context(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        """Build expression context from non-fluents, state, and action. / 从 non-fluent、state 和 action 构建表达式上下文。"""
        context = dict(self.non_fluents)
        context.update({name: False for name in self.state_names})
        context.update(state)
        context.update({name: False for name in self.action_names})
        context.update(action)
        return context

    def _action_id(self, action: Mapping[str, Any]) -> int:
        """Return a compact id for a concrete grounded action. / 返回具体 grounded action 的紧凑编号。"""
        key: ActionKey = tuple(
            (name, _plain_value(action.get(name, False)))
            for name in self.action_names
        )
        action_id = self._action_ids.get(key)
        if action_id is None:
            action_id = len(self._action_ids)
            self._action_ids[key] = action_id
        return action_id

    def _reward_for_ids(
        self,
        state_id: int,
        action_id: int,
        action: Mapping[str, Any],
    ) -> float:
        """Evaluate and cache one reward matrix entry."""
        cache_key = (state_id, action_id)
        cached = self._reward_cache.get(cache_key)
        if cached is not None:
            return cached
        state = self.state_from_key(self._state_index.key(state_id))
        context = self._context(state, action)
        row = self._transition_row(state_id, action_id, action)
        reward = sum(
            probability
            * self.expected_reward(
                {
                    **context,
                    **{
                        f"{name}'": self.state_from_key(
                            self._state_index.key(int(target_id))
                        ).get(name, False)
                        for name in self.state_names
                    },
                }
            )
            for target_id, probability in zip(row.next_state_ids, row.probabilities)
        )
        if not isfinite(reward):
            raise KernelError("Expected reward must be finite.")
        self._reward_cache[cache_key] = reward
        return reward

    def _observation_distribution_for_state(
        self,
        state: StateKey,
        action: Mapping[str, Any],
    ) -> Mapping[ObservationKey, float]:
        r"""Return cached $$O(o\mid s',a)$$ support. / 返回缓存的观测似然支持集。"""
        state_id = self._state_index.register(state)
        action_id = self._action_id(action)
        cache_key = (state_id, action_id)
        cached = self._observation_cache.get(cache_key)
        if cached is not None:
            return cached
        state_mapping = self.state_from_key(state)
        context = self._context(state_mapping, action)
        context.update({f"{name}'": state_mapping.get(name, False) for name in self.state_names})
        distribution = self._observation_distribution(context)
        self._observation_cache[cache_key] = distribution
        return distribution

    def _state_cpf_expression(self, state_name: str) -> Any:
        """Return the CPF expression for one next-state fluent. / 返回一个 next-state fluent 的 CPF 表达式。"""
        key = f"{state_name}'"
        value = self.cpfs.get(key)
        if value is None:
            raise KernelError(f"Missing next-state CPF for {state_name}.")
        return _cpf_expression(value)

    def _observation_distribution(self, context: Mapping[str, Any]) -> Mapping[ObservationKey, float]:
        """Return observation-value distribution from observation CPFs. / 从 observation CPF 返回 observation-value 分布。"""
        partials: dict[ObservationKey, float] = {(): 1.0}
        for obs_name in self.observation_names:
            if obs_name in self.cpfs:
                expr = self.cpfs[obs_name]
            elif f"{obs_name}'" in self.cpfs:
                expr = self.cpfs[f"{obs_name}'"]
            else:
                raise KernelError(f"Missing observation CPF for {obs_name}.")
            updated: dict[ObservationKey, float] = {}
            for partial_key, partial_prob in partials.items():
                partial_obs = dict(partial_key)
                value_dist = self.expression_distribution(_cpf_expression(expr), {**context, **partial_obs})
                for value, value_prob in _normalize_distribution(value_dist).items():
                    next_partial = tuple(sorted({**partial_obs, obs_name: _plain_value(value)}.items()))
                    updated[next_partial] = (
                        updated.get(next_partial, 0.0)
                        + partial_prob * value_prob
                    )
            partials = updated
        return _normalize_distribution(partials)

    def _validate_supported(self) -> None:
        """Validate finite state ranges and CC-POMDP risky-state selectors."""
        state_ranges = getattr(self.grounded_model, "state_ranges", {}) or {}
        unsupported = [
            name
            for name in self.state_names
            if str(state_ranges.get(name, "bool")) not in {"bool", "int"}
        ]
        if unsupported:
            raise KernelError(
                "Current kernel supports bool and finite-horizon int "
                "state fluents only: "
                + ", ".join(unsupported)
            )
        selector_names = {
            name
            for selector in self.risk.risky_states
            for name, _ in selector
        }
        unknown_names = sorted(selector_names - set(self.state_names))
        if unknown_names:
            raise KernelError(
                "Risky-state selectors reference unknown grounded state fluents: "
                + ", ".join(unknown_names)
            )
        for selector in self.risk.risky_states:
            for name, expected in selector:
                state_range = str(state_ranges.get(name, "bool"))
                expected_type = bool if state_range == "bool" else int
                if type(expected) is not expected_type:
                    raise KernelError(
                        f"Risky-state selector {name!r} must use a "
                        f"{state_range} value, got {expected!r}."
                    )


def _cpf_expression(value: Any) -> Any:
    """Extract the expression from pyRDDLGym CPF entries. / 从 pyRDDLGym CPF entry 中提取表达式。"""
    if isinstance(value, tuple) and len(value) == 2:
        return value[1]
    return value


def _mapping_keys(value: object) -> tuple[str, ...]:
    """Return string keys from a mapping. / 从 mapping 返回字符串键。"""
    return tuple(str(key) for key in value) if isinstance(value, Mapping) else ()


def _is_expression(value: object) -> bool:
    """Return whether a value looks like pyRDDLGym Expression. / 判断值是否像 pyRDDLGym Expression。"""
    return hasattr(value, "etype") and hasattr(value, "args")


def _as_args(value: Any) -> tuple[Any, ...]:
    """Return expression args as a tuple. / 将 expression args 转为 tuple。"""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _check_arity(args: Sequence[Any], expected: int, name: str) -> None:
    """Check expression arity. / 检查表达式参数个数。"""
    if len(args) != expected:
        raise KernelError(f"{name} expects {expected} arguments, got {len(args)}.")


def _plain_value(value: Any) -> Hashable:
    """Convert numpy scalar values to hashable Python values. / 将 numpy scalar 转为可哈希 Python 值。"""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _state_matches_selector(
    state: Mapping[str, Any],
    selector: StateSelector,
) -> bool:
    """Return whether a state satisfies every equality in one selector."""
    for name, expected in selector:
        if name not in state:
            return False
        actual = _plain_value(state[name])
        if type(actual) is not type(expected) or actual != expected:
            return False
    return True


def _normalize_distribution(
    distribution: Mapping[Hashable, float],
) -> dict[Hashable, float]:
    """Validate and normalize one finite non-negative distribution."""
    numeric: dict[Hashable, float] = {}
    for key, raw_value in distribution.items():
        value = float(raw_value)
        if not isfinite(value):
            raise KernelError(
                f"Probability distribution contains non-finite mass: {key!r}"
            )
        if value < 0:
            raise KernelError(
                f"Probability distribution contains negative mass: {key!r}={value!r}"
            )
        if value > 0:
            numeric[key] = value
    total = sum(numeric.values())
    if total <= 0:
        return {}
    if not isfinite(total):
        raise KernelError("Probability distribution has non-finite total mass.")
    return {key: value / total for key, value in numeric.items()}


def _normalized_mass(
    distribution: Mapping[StateKey, float],
) -> dict[StateKey, float]:
    """Validate and normalize a non-empty state mass."""
    normalized = _normalize_distribution(distribution)
    if not normalized:
        raise KernelError("Constraint mass requires positive probability mass.")
    return normalized


def _mass_belief(
    mass: Mapping[StateKey, float],
) -> dict[StateKey, float]:
    """Normalize an unnormalized state mass, allowing an empty safe flow."""
    return _normalize_distribution(mass)


def _non_negative(value: float) -> float:
    """Validate one finite non-negative coefficient."""
    value = float(value)
    if not isfinite(value):
        raise KernelError("Constraint aggregation produced a non-finite value.")
    if value < 0.0:
        raise KernelError(f"Expected a non-negative quantity, got {value!r}.")
    return value


def _probability(value: float) -> float:
    """Validate and clamp a floating-point probability to ``[0, 1]``."""
    value = _non_negative(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _expectation(distribution: Mapping[Hashable, float]) -> float:
    """Return the expectation of finite numeric support."""
    value = sum(
        float(item) * probability
        for item, probability in distribution.items()
    )
    if not isfinite(value):
        raise KernelError("Distribution expectation must be finite.")
    return value


def _combine_distributions(distributions: Sequence[Distribution], fn: Any) -> Distribution:
    """Combine independent finite distributions with one operator. / 用一个算子组合多个有限分布。"""
    if not distributions:
        return {fn(): 1.0}
    result: dict[Hashable, float] = {}
    keys = [tuple(dist.items()) for dist in distributions]
    for combination in product(*keys):
        values = [item[0] for item in combination]
        probability = prod(item[1] for item in combination)
        output = _plain_value(fn(*values))
        result[output] = result.get(output, 0.0) + probability
    return _normalize_distribution(result)


def _arith(op: str) -> Any:
    """Return arithmetic operator. / 返回算术算子。"""
    if op == "+":
        return lambda *values: sum(values)
    if op == "-":
        return lambda *values: -values[0] if len(values) == 1 else values[0] - values[1]
    if op == "*":
        return lambda *values: prod(values)
    if op == "/":
        return lambda lhs, rhs: lhs / rhs
    raise KernelError(f"Unsupported arithmetic operator: {op}")


def _logic(op: str) -> Any:
    """Return boolean operator. / 返回布尔算子。"""
    if op in {"^", "&"}:
        return lambda *values: all(bool(value) for value in values)
    if op == "|":
        return lambda *values: any(bool(value) for value in values)
    if op == "~":
        return lambda value: not bool(value)
    if op == "=>":
        return lambda lhs, rhs: (not bool(lhs)) or bool(rhs)
    if op == "<=>":
        return lambda lhs, rhs: bool(lhs) == bool(rhs)
    raise KernelError(f"Unsupported logical operator: {op}")


def _relation(op: str) -> Any:
    """Return relational operator. / 返回关系算子。"""
    if op == ">=":
        return lambda lhs, rhs: lhs >= rhs
    if op == "<=":
        return lambda lhs, rhs: lhs <= rhs
    if op == "<":
        return lambda lhs, rhs: lhs < rhs
    if op == ">":
        return lambda lhs, rhs: lhs > rhs
    if op == "==":
        return lambda lhs, rhs: lhs == rhs
    if op == "~=":
        return lambda lhs, rhs: lhs != rhs
    raise KernelError(f"Unsupported relational operator: {op}")


def _observation_label(observation: ObservationKey) -> str:
    """Return a compact *injective* label for one observation value.

    Returning only the names of Boolean entries whose value is ``True``
    silently merges histories such as ``{flag=True, colour='red'}`` and
    ``{flag=True, colour='blue'}`` in the AND-OR node arena.  Since a policy
    may select different actions after those observations, that changes the
    policy-tree ILP rather than merely changing diagnostics.

    Keep the familiar one-hot Boolean label used by existing navigation
    models, but encode every other observation from the complete ordered key.
    / 仅对 one-hot Boolean observation 保留紧凑名称；其余情况编码完整 key，
    防止不同 observation history 被错误合并。
    """
    active = [name for name, value in observation if value is True]
    if len(active) == 1 and all(isinstance(value, bool) for _, value in observation):
        return active[0]
    return repr(observation)
