"""Exact finite-kernel view over pyRDDLGym grounded expressions."""

# TODO(phase-9.1): Extend finite expression support beyond Bernoulli/Discrete
# random variables when benchmark domains require additional finite laws.
# TODO(phase-9.2): Add dependency-aware factored CPF evaluation so one
# stochastic CPF does not require materializing unrelated state factors.

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

StateKey = tuple[tuple[str, Hashable], ...]
ObservationKey = tuple[tuple[str, Hashable], ...]
Distribution = dict[Hashable, float]
ActionKey = tuple[tuple[str, Hashable], ...]


@dataclass(frozen=True, slots=True)
class SparseProbabilityVector:
    """Store an exact sparse distribution with integer state ids. / 用整数状态编号保存精确稀疏分布。"""

    state_ids: np.ndarray  # Non-zero support ids. / 非零概率状态编号。
    probabilities: np.ndarray  # Probabilities aligned with state_ids. / 与状态编号对齐的概率。


@dataclass(frozen=True, slots=True)
class SparseTransitionRow:
    """Store one cached sparse row of $$T_a$$. / 保存 $$T_a$$ 的一条缓存稀疏行。"""

    next_state_ids: np.ndarray  # Reachable successor ids. / 可达后继状态编号。
    probabilities: np.ndarray  # $$T(s,a,s')$$ for each successor. / 每个后继的转移概率。


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


class ExactKernelError(ValueError):
    """Raised when exact finite-kernel compilation is unsupported. / exact finite-kernel 编译不支持时抛出。"""


@dataclass(frozen=True)
class RiskConstraintSpec:
    """Describe optional C-POMDP risk/cost rows from sidecar config. / 描述 sidecar 中的 C-POMDP risk/cost 配置。"""

    budget: float | None = None
    state_fluent_costs: Mapping[str, float] = field(default_factory=dict)
    next_state_fluent_costs: Mapping[str, float] = field(default_factory=dict)

    def has_costs(self) -> bool:
        """Return whether any risk/cost selector is configured. / 返回是否配置了任何 risk/cost selector。"""
        return bool(self.state_fluent_costs or self.next_state_fluent_costs)


@dataclass(frozen=True)
class ExactTransitionOutcome:
    """Store one exact transition branch. / 保存一个 exact transition 分支。"""

    state: StateKey
    probability: float


@dataclass(frozen=True)
class ExactObservationOutcome:
    """Store one exact observation branch and posterior belief. / 保存一个 exact observation 分支和 posterior belief。"""

    observation: ObservationKey
    label: str
    probability: float
    belief: Mapping[StateKey, float]


@dataclass(frozen=True)
class ExactActionExpansion:
    """Store exact Algorithm 2 constants for an action history. / 保存 action history 的 exact Algorithm 2 常量。"""

    utility: float
    risk: float
    prior_belief: Mapping[StateKey, float]
    observations: tuple[ExactObservationOutcome, ...]


@dataclass(frozen=True)
class SafeActionExpansion:
    """Store chance-constrained safe-belief action constants. / 保存 chance-constrained safe-belief action 常量。"""

    utility: float
    risk: float
    prior_belief: Mapping[StateKey, float]
    survival_probability: float
    observations: tuple[ExactObservationOutcome, ...]


@dataclass(frozen=True)
class ExactBeliefState:
    """Store an exact online belief for paper-path planners. / 保存论文 planner 使用的 exact online belief。"""

    belief: Mapping[StateKey, float]
    observation: Mapping[str, Any]
    is_pomdp: bool
    source: str
    support: Mapping[str, float]

    @classmethod
    def from_runtime(
        cls,
        exact_kernel: Any,
        runtime: Any,
        observation: Mapping[str, Any],
        *,
        is_pomdp: bool,
        source: str = "exact-initial-state",
    ) -> "ExactBeliefState":
        """Build an exact belief from the current runtime state. / 从当前 runtime state 构建 exact belief。"""
        belief = exact_kernel.initial_belief_from_state(runtime.state)
        return cls.from_belief(
            exact_kernel,
            belief,
            observation,
            is_pomdp=is_pomdp,
            source=source,
        )

    @classmethod
    def from_belief(
        cls,
        exact_kernel: Any,
        belief: Mapping[StateKey, float],
        observation: Mapping[str, Any],
        *,
        is_pomdp: bool,
        source: str,
    ) -> "ExactBeliefState":
        """Build an exact belief record from a normalized state distribution. / 从状态分布构建 exact belief 记录。"""
        normalized = _normalize_distribution(dict(belief))
        if not normalized:
            raise ExactKernelError("Exact belief must contain positive probability mass.")
        return cls(
            belief=normalized,
            observation=dict(observation),
            is_pomdp=is_pomdp,
            source=source,
            support=_belief_support(exact_kernel, normalized),
        )

    def advance(
        self,
        exact_kernel: Any,
        action: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        observed_state: Mapping[str, Any] | None = None,
    ) -> "ExactBeliefState":
        r"""Return the exact Bayes update after one action and observation.

        The paper-path online loop should update the root belief with:

        $$
          b'(s')=\eta\,O(o\mid s',a)\sum_s T(s,a,s')b(s).
        $$

        / 根据 action 和 observation 精确执行 Bayes belief update，供下一轮
        full-ILP/HILP 作为 root belief。
        """
        next_belief = _advance_exact_belief(
            exact_kernel,
            self.belief,
            action,
            observation,
            observed_state=observed_state,
        )
        return ExactBeliefState.from_belief(
            exact_kernel,
            next_belief,
            observation,
            is_pomdp=self.is_pomdp,
            source="exact-bayes",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly exact belief summary. / 返回 JSON 友好的 exact belief 摘要。"""
        return {
            "is_pomdp": self.is_pomdp,
            "is_exact": True,
            "source": self.source,
            "particle_count": None,
            "requested_particles": None,
            "attempts": None,
            "observation": _json_ready(self.observation),
            "support": dict(self.support),
        }


@dataclass(frozen=True)
class ExactRDDLKernel:
    """Lazily compile reached grounded states into exact sparse kernels. / 将触达的 grounded 状态按需编译为精确稀疏内核。"""

    grounded_model: Any
    risk: RiskConstraintSpec = field(default_factory=RiskConstraintSpec)
    _state_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _action_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _observation_names_cache: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _non_fluents_cache: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _cpfs_cache: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _state_index: _LazyStateIndex = field(default_factory=_LazyStateIndex, init=False, repr=False, compare=False)
    _action_ids: dict[ActionKey, int] = field(default_factory=dict, init=False, repr=False, compare=False)
    _transition_rows: dict[tuple[int, int], SparseTransitionRow] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _reward_cache: dict[tuple[int, int], float] = field(default_factory=dict, init=False, repr=False, compare=False)
    _state_cost_cache: dict[int, float] = field(default_factory=dict, init=False, repr=False, compare=False)
    _next_state_cost_cache: dict[int, float] = field(default_factory=dict, init=False, repr=False, compare=False)
    _observation_cache: dict[tuple[int, int], Mapping[ObservationKey, float]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _cache_hits: dict[str, int] = field(
        default_factory=lambda: {"transition": 0, "reward": 0, "observation": 0},
        init=False,
        repr=False,
        compare=False,
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
        non_fluents = getattr(self.grounded_model, "non_fluents", None)
        cpfs = getattr(self.grounded_model, "cpfs", None)
        object.__setattr__(self, "_non_fluents_cache", non_fluents if isinstance(non_fluents, Mapping) else {})
        object.__setattr__(self, "_cpfs_cache", cpfs if isinstance(cpfs, Mapping) else {})

    @classmethod
    def from_grounded_model(
        cls,
        grounded_model: Any,
        *,
        risk: RiskConstraintSpec | None = None,
    ) -> "ExactRDDLKernel":
        """Build an exact kernel from a pyRDDLGym grounded model. / 从 pyRDDLGym grounded model 构建 exact kernel。"""
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

    def cache_info(self) -> Mapping[str, int]:
        """Return lazy-kernel sizes and hit counters. / 返回按需内核规模和缓存命中计数。"""
        return {
            "discovered_states": len(self._state_index.id_to_key),
            "discovered_actions": len(self._action_ids),
            "transition_rows": len(self._transition_rows),
            "reward_entries": len(self._reward_cache),
            "risk_entries": len(self._state_cost_cache) + len(self._next_state_cost_cache),
            "observation_entries": len(self._observation_cache),
            "transition_hits": self._cache_hits["transition"],
            "reward_hits": self._cache_hits["reward"],
            "observation_hits": self._cache_hits["observation"],
        }

    def sparse_belief(self, belief: Mapping[StateKey, float]) -> SparseProbabilityVector:
        r"""Index and normalize an exact belief without enumerating absent states.

        Only states in the non-zero support of $$b_q$$ receive ids. / 只为
        $$b_q$$ 非零支持集中的状态分配编号并构造 NumPy 稀疏向量。
        """
        clean = _normalize_distribution(dict(belief))
        ids = np.fromiter(
            (self._state_index.register(state) for state in clean),
            dtype=np.int64,
            count=len(clean),
        )
        probabilities = np.fromiter(clean.values(), dtype=np.float64, count=len(clean))
        return SparseProbabilityVector(ids, probabilities)

    def belief_mapping(self, belief: SparseProbabilityVector) -> Mapping[StateKey, float]:
        """Convert an indexed sparse belief back to the public mapping API. / 将编号稀疏 belief 转回公开 mapping 接口。"""
        return {
            self._state_index.key(int(state_id)): float(probability)
            for state_id, probability in zip(belief.state_ids, belief.probabilities)
            if abs(float(probability)) > 1e-15
        }

    def update_belief(
        self,
        belief: Mapping[StateKey, float],
        action: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        observed_state: Mapping[str, Any] | None = None,
    ) -> Mapping[StateKey, float]:
        r"""Return the exact one-step posterior belief.

        $$
          b'(s')=\eta\,O(o\mid s',a)\sum_s T(s,a,s')b(s).
        $$

        / 用 grounded transition 和 observation CPF 精确计算一阶 posterior belief。
        """
        expansion = self.expand_action(belief, action)
        observation_key = self.observation_key(observation, observed_state=observed_state)
        for outcome in expansion.observations:
            if outcome.observation == observation_key:
                return outcome.belief
        raise ExactKernelError(
            "Observed outcome is outside exact observation support: "
            f"{observation_key!r}"
        )

    def observation_key(
        self,
        observation: Mapping[str, Any],
        *,
        observed_state: Mapping[str, Any] | None = None,
    ) -> ObservationKey:
        """Convert a runtime observation into an exact observation key. / 将 runtime observation 转成 exact observation key。"""
        if not self.observation_names:
            state = observed_state if observed_state is not None else observation
            return (("__state__", self.state_key(state)),)
        return tuple(
            (name, _plain_value(observation.get(name, False)))
            for name in self.observation_names
        )

    def safe_belief_from_belief(self, belief: Mapping[StateKey, float]) -> Mapping[StateKey, float]:
        r"""Condition a belief on the initial safe event.

        Lemma 3.3 uses $$R=\Delta-r(b_0)$$ and then propagates safe
        beliefs.  This helper forms the corresponding root safe belief:

        $$
           b^*_0(s)=\frac{b_0(s)(1-P_R(s))}
                         {1-r(b_0)}.
        $$

        / 将 root belief 条件化到初始安全事件；初始风险由 ILP risk budget
        单独扣除，后续 tree 中传播的是 safe belief。
        """
        b_0 = self.sparse_belief(belief)
        survival = np.fromiter(
            (1.0 - self._state_failure_for_id(int(state_id)) for state_id in b_0.state_ids),
            dtype=np.float64,
            count=b_0.state_ids.size,
        )
        weights = b_0.probabilities * survival
        positive = weights > 1e-15
        total = float(np.sum(weights[positive]))
        if total <= 0.0:
            return {}
        b_star_0 = SparseProbabilityVector(b_0.state_ids[positive], weights[positive] / total)
        return self.belief_mapping(b_star_0)

    def fluent_belief(self, belief: Mapping[StateKey, float]) -> Mapping[str, float]:
        r"""Return fluent marginals by matrix multiplication.

        $$m=bF$$ where $$F_{s,f}=1$$ iff fluent $$f$$ is true in state $$s$$.
        / 通过状态-fluent 指示矩阵计算边缘概率。
        """
        sparse = self.sparse_belief(belief)
        if sparse.state_ids.size == 0:
            return {name: 0.0 for name in self.state_names}
        truth = np.asarray(
            [
                [bool(dict(self._state_index.key(int(state_id))).get(name, False)) for name in self.state_names]
                for state_id in sparse.state_ids
            ],
            dtype=np.float64,
        )
        marginals = sparse.probabilities @ truth
        return {name: float(marginals[index]) for index, name in enumerate(self.state_names)}

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

    def expand_action(
        self,
        belief: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> ExactActionExpansion:
        r"""Compute exact Algorithm 2 constants for one action.

        $$
           b_{qa}(s')=\sum_s T(s,a,s')b_q(s),\quad
           u_{qa}=\sum_s b_q(s)U(s,a),\quad
           r_{qa}=\sum_s b_q(s)P(s,a).
        $$

        / 精确计算 action 后的 prior belief、utility 与 risk/cost 常量。
        """
        b_q = self.sparse_belief(belief)  # Indexed non-zero support of $$b_q$$. / $$b_q$$ 的编号非零支持集。
        action_id = self._action_id(action)
        rewards = np.fromiter(
            (self._reward_for_ids(int(state_id), action_id, action) for state_id in b_q.state_ids),
            dtype=np.float64,
            count=b_q.state_ids.size,
        )
        current_costs = np.fromiter(
            (self._state_cost_for_id(int(state_id)) for state_id in b_q.state_ids),
            dtype=np.float64,
            count=b_q.state_ids.size,
        )
        utility = float(b_q.probabilities @ rewards)  # $$u_{qa}=b_q^T R_a$$. / belief 与 reward 向量点积。
        current_risk = float(b_q.probabilities @ current_costs)  # $$b_q^T C_a$$. / 当前状态风险点积。
        b_qa = self._propagate_sparse_belief(b_q, action_id, action)
        next_costs = np.fromiter(
            (self._next_state_cost_for_id(int(state_id)) for state_id in b_qa.state_ids),
            dtype=np.float64,
            count=b_qa.state_ids.size,
        )
        next_risk = float(b_qa.probabilities @ next_costs) if b_qa.state_ids.size else 0.0
        prior = self.belief_mapping(b_qa)  # Public Algorithm 2 mapping. / Algorithm 2 对外使用的 belief mapping。
        observations = self.observation_outcomes(prior, action)
        return ExactActionExpansion(
            utility=utility,
            risk=current_risk + next_risk,
            prior_belief=prior,
            observations=observations,
        )

    def expand_safe_action(
        self,
        safe_belief: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> SafeActionExpansion:
        r"""Compute the CC-POMDP safe-belief action update.

        For chance constraints, $$\rho^*(q)$$ is the probability of
        reaching history $$q$$ without failure, and
        $$b^*_q$$ is the belief conditioned on that safe prefix.  For one
        action:

        $$
           p_{\mathrm{safe}}(q,a)=
           \sum_{s,s'}b^*_q(s)T(s,a,s')\Pr(\mathrm{safe}\mid s,a,s')
        $$

        $$
           b^*_{qa}(s')=
           \frac{\sum_s b^*_q(s)T(s,a,s')\Pr(\mathrm{safe}\mid s,a,s')}
                {p_{\mathrm{safe}}(q,a)}.
        $$

        / 计算 chance-constrained safe belief 递推；risk 是本 action 下发生
        failure 的条件概率，prior_belief 是已条件化 survival 的 safe prior。
        """
        b_star_q = self.sparse_belief(safe_belief)
        if b_star_q.state_ids.size == 0:
            return SafeActionExpansion(
                utility=0.0,
                risk=0.0,
                prior_belief={},
                survival_probability=0.0,
                observations=(),
            )
        action_id = self._action_id(action)
        rewards = np.fromiter(
            (self._reward_for_ids(int(state_id), action_id, action) for state_id in b_star_q.state_ids),
            dtype=np.float64,
            count=b_star_q.state_ids.size,
        )
        utility = float(b_star_q.probabilities @ rewards)
        safe_qa, survival_probability = self._propagate_sparse_belief(
            b_star_q,
            action_id,
            action,
            condition_on_survival=True,
        )
        safe_prior = self.belief_mapping(safe_qa)
        return SafeActionExpansion(
            utility=utility,
            risk=max(0.0, min(1.0, 1.0 - survival_probability)),
            prior_belief=safe_prior,
            survival_probability=max(0.0, min(1.0, survival_probability)),
            observations=self.observation_outcomes(safe_prior, action) if safe_prior else (),
        )

    def transition_survival_probability(
        self,
        source: StateKey,
        target: StateKey,
    ) -> float:
        r"""Return $$Pr(\mathrm{safe}\mid s,a,s')$$ for configured risk fluents.

        Safe beliefs are already conditioned on the safe prefix, so this
        transition-level survival checks the target state only.  Both
        `state_fluent_costs` and `next_state_fluent_costs` may mark target
        risky states; values are interpreted as probabilities and clamped to
        ``[0, 1]``.

        / safe belief 已经条件化了安全前缀，所以这里不重复计算 source risk；
        transition survival 只检查 target state 是否安全。
        """
        del source
        return 1.0 - self.transition_failure_probability(target)

    def belief_state_risk_probability(self, belief: Mapping[StateKey, float]) -> float:
        """Return initial/root state risk probability r(b). / 返回初始/root state risk 概率 r(b)。"""
        sparse = self.sparse_belief(belief)
        failure = np.fromiter(
            (self._state_failure_for_id(int(state_id)) for state_id in sparse.state_ids),
            dtype=np.float64,
            count=sparse.state_ids.size,
        )
        return _clamped_probability(float(sparse.probabilities @ failure))

    def state_failure_probability(self, state: StateKey) -> float:
        """Return configured current-state failure probability. / 返回配置的当前 state failure 概率。"""
        return self._state_failure_for_id(self._state_index.register(state))

    def next_state_failure_probability(self, state: StateKey) -> float:
        """Return configured next-state failure probability. / 返回配置的 next-state failure 概率。"""
        return _clamped_probability(self._next_state_cost_for_id(self._state_index.register(state)))

    def transition_failure_probability(self, state: StateKey) -> float:
        """Return target-state failure probability for safe-belief recursion. / 返回 safe-belief 递推中的 target-state failure 概率。"""
        state_failure = self.state_failure_probability(state)
        next_state_failure = self.next_state_failure_probability(state)
        return 1.0 - (1.0 - state_failure) * (1.0 - next_state_failure)

    def transition_distribution(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> Mapping[StateKey, float]:
        """Return a cached sparse next-state row generated on first access. / 首次访问时生成并缓存稀疏后继行。"""
        source = self.state_key(state)
        source_id = self._state_index.register(source)
        action_id = self._action_id(action)
        row = self._transition_row(source_id, action_id, action)
        return {
            self._state_index.key(int(next_id)): float(probability)
            for next_id, probability in zip(row.next_state_ids, row.probabilities)
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
            self._cache_hits["transition"] += 1
            return cached
        state = self.state_from_key(self._state_index.key(source_id))
        context = self._context(state, action)
        partials: dict[StateKey, float] = {(): 1.0}
        for state_name in self.state_names:
            expr = self._state_cpf_expression(state_name)
            # Synchronous CPFs read the same current context, so evaluate each
            # value distribution once per CPF. / 同步 CPF 共享当前上下文，每个 CPF 只求值一次。
            value_dist = self.expression_distribution(expr, context)
            updated: dict[StateKey, float] = {}
            for partial_key, partial_prob in partials.items():
                partial_state = dict(partial_key)
                for value, value_prob in value_dist.items():
                    next_partial = tuple(sorted({**partial_state, state_name: _plain_value(value)}.items()))
                    updated[next_partial] = updated.get(next_partial, 0.0) + partial_prob * value_prob
            partials = updated
        distribution = _normalize_distribution(partials)
        row = SparseTransitionRow(
            next_state_ids=np.fromiter(
                (self._state_index.register(state_key) for state_key in distribution),
                dtype=np.int64,
                count=len(distribution),
            ),
            probabilities=np.fromiter(distribution.values(), dtype=np.float64, count=len(distribution)),
        )
        self._transition_rows[cache_key] = row
        return row

    def _propagate_sparse_belief(
        self,
        belief: SparseProbabilityVector,
        action_id: int,
        action: Mapping[str, Any],
        *,
        condition_on_survival: bool = False,
    ) -> SparseProbabilityVector | tuple[SparseProbabilityVector, float]:
        r"""Compute sparse $$bT_a$$ with vectorized accumulation.

        With safe conditioning, each branch is additionally multiplied by
        $$Pr(safe\mid s,a,s')$$. / 用 NumPy 聚合同一后继状态；safe 模式额外乘生存概率。
        """
        target_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        for source_id, source_probability in zip(belief.state_ids, belief.probabilities):
            row = self._transition_row(int(source_id), action_id, action)
            weights = float(source_probability) * row.probabilities
            if condition_on_survival:
                survival = np.fromiter(
                    (
                        self.transition_survival_probability(
                            self._state_index.key(int(source_id)),
                            self._state_index.key(int(target_id)),
                        )
                        for target_id in row.next_state_ids
                    ),
                    dtype=np.float64,
                    count=row.next_state_ids.size,
                )
                weights = weights * survival
            positive = weights > 1e-15
            if np.any(positive):
                target_chunks.append(row.next_state_ids[positive])
                weight_chunks.append(weights[positive])
        if not target_chunks:
            empty = SparseProbabilityVector(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))
            return (empty, 0.0) if condition_on_survival else empty
        target_ids = np.concatenate(target_chunks)
        weights = np.concatenate(weight_chunks)
        accumulated = np.bincount(target_ids, weights=weights, minlength=len(self._state_index.id_to_key))
        support = np.flatnonzero(accumulated > 1e-15).astype(np.int64, copy=False)
        support_weights = accumulated[support]
        total = float(np.sum(support_weights))
        normalized = support_weights / total if total > 0.0 else support_weights
        result = SparseProbabilityVector(support, normalized)
        return (result, total) if condition_on_survival else result

    def transition_probability(
        self,
        source: StateKey,
        action: Mapping[str, Any],
        target: StateKey,
    ) -> float:
        r"""Return one transition probability $$T(s,a,s')$$.

        / 返回一个转移概率 $$T(s,a,s')$$，供论文 Algorithm 2 的
        backward message 递推使用。
        """
        return float(
            self.transition_distribution(self.state_from_key(source), action).get(target, 0.0)
        )

    def observation_outcomes(
        self,
        prior_belief: Mapping[StateKey, float],
        action: Mapping[str, Any],
    ) -> tuple[ExactObservationOutcome, ...]:
        """Return exact observation outcomes and posterior beliefs. / 返回 exact observation outcomes 和 posterior beliefs。"""
        if not self.observation_names:
            return tuple(
                ExactObservationOutcome(
                    observation=(("__state__", state_key),),
                    label=self.state_label(state_key),
                    probability=probability,
                    belief={state_key: 1.0},
                )
                for state_key, probability in sorted(prior_belief.items(), key=lambda item: repr(item[0]))
                if probability > 0.0
            )
        observation_state_prob: dict[ObservationKey, dict[StateKey, float]] = {}
        for state_key, state_prob in prior_belief.items():
            obs_dist = self._observation_distribution_for_state(state_key, action)
            for obs_key, obs_prob in obs_dist.items():
                weighted = state_prob * obs_prob
                if weighted <= 0.0:
                    continue
                bucket = observation_state_prob.setdefault(obs_key, {})
                bucket[state_key] = bucket.get(state_key, 0.0) + weighted
        outcomes: list[ExactObservationOutcome] = []
        for obs_key, state_weights in sorted(observation_state_prob.items(), key=lambda item: repr(item[0])):
            probability = sum(state_weights.values())
            outcomes.append(
                ExactObservationOutcome(
                    observation=obs_key,
                    label=_observation_label(obs_key),
                    probability=probability,
                    belief=_normalize_distribution(state_weights),
                )
            )
        return tuple(outcomes)

    def observation_probability(
        self,
        observation: ObservationKey,
        state: StateKey,
        action: Mapping[str, Any],
    ) -> float:
        r"""Return one observation likelihood $$O(o,s,a)$$.

        The fallback MDP observation mode treats the true state as the
        observation. Explicit POMDP observations are evaluated from grounded
        observation CPFs.

        / 返回一个观测似然 $$O(o,s,a)$$；MDP fallback 中 observation 就是
        真实 state，POMDP 中则从 grounded observation CPF 计算。
        """
        if observation and observation[0][0] == "__state__":
            observed_state = observation[0][1]
            return 1.0 if observed_state == state else 0.0
        return float(self._observation_distribution_for_state(state, action).get(observation, 0.0))

    def backward_message(
        self,
        current_belief: Mapping[StateKey, float],
        next_message: Mapping[StateKey, float],
        action: Mapping[str, Any],
        observation: ObservationKey,
    ) -> Mapping[StateKey, float]:
        r"""Apply one sparse Algorithm 2 backward operator.

        $$f_i(s)=\sum_{s'}T(s,a,s')O(o\mid s',a)f_{i+1}(s').$$

        Transition rows are cached; NumPy evaluates each row dot product. /
        转移行由缓存复用，并用 NumPy 点积计算每个当前状态的 backward message。
        """
        action_id = self._action_id(action)
        result: dict[StateKey, float] = {}
        for state in current_belief:
            source_id = self._state_index.register(state)
            row = self._transition_row(source_id, action_id, action)
            future = np.fromiter(
                (next_message.get(self._state_index.key(int(target_id)), 0.0) for target_id in row.next_state_ids),
                dtype=np.float64,
                count=row.next_state_ids.size,
            )
            likelihood = np.fromiter(
                (
                    self.observation_probability(
                        observation,
                        self._state_index.key(int(target_id)),
                        action,
                    )
                    for target_id in row.next_state_ids
                ),
                dtype=np.float64,
                count=row.next_state_ids.size,
            )
            result[state] = float(row.probabilities @ (likelihood * future))
        return result

    def expected_state_action_reward(
        self,
        state: StateKey,
        action: Mapping[str, Any],
    ) -> float:
        """Return cached $$U(s,a)$$ for one indexed state/action. / 返回一个编号状态动作的缓存 $$U(s,a)$$。"""
        state_id = self._state_index.register(state)
        action_id = self._action_id(action)
        return self._reward_for_ids(state_id, action_id, action)

    def expected_reward(self, context: Mapping[str, Any]) -> float:
        """Return expected reward expression value under finite random support. / 返回有限随机支持下的期望 reward。"""
        reward = getattr(self.grounded_model, "reward", None)
        if reward is None:
            raise ExactKernelError("Grounded model does not expose a reward expression.")
        return float(_expectation(self.expression_distribution(reward, context)))

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
                raise ExactKernelError(f"Exact kernel expected grounded pvar but got {name}{params}.")
            if name not in context:
                raise ExactKernelError(f"Expression references unknown grounded pvar: {name}")
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
                    result[value] = result.get(value, 0.0) + truth_prob * value_prob
            return _normalize_distribution(result)
        if etype == "randomvar":
            return self._random_distribution(op, _as_args(args), context)
        if etype == "aggregation":
            raise ExactKernelError(f"Grounded exact aggregation is not implemented for operator {op}.")
        raise ExactKernelError(f"Unsupported exact expression type: {etype}/{op}")

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
            p = float(_expectation(self.expression_distribution(args[0], context)))
            if p < -1e-12 or p > 1.0 + 1e-12:
                raise ExactKernelError(f"Bernoulli probability out of range: {p}")
            p = min(1.0, max(0.0, p))
            return _normalize_distribution({True: p, False: 1.0 - p})
        if name == "Discrete":
            if len(args) % 2 != 0:
                raise ExactKernelError("Exact Discrete expects alternating value/probability arguments.")
            result: dict[Hashable, float] = {}
            for value_expr, prob_expr in zip(args[0::2], args[1::2]):
                value_dist = self.expression_distribution(value_expr, context)
                if len(value_dist) != 1:
                    raise ExactKernelError("Exact Discrete values must be deterministic.")
                value = next(iter(value_dist))
                prob_value = float(_expectation(self.expression_distribution(prob_expr, context)))
                result[value] = result.get(value, 0.0) + prob_value
            return _normalize_distribution(result)
        raise ExactKernelError(f"Random distribution {name} is not finite/exact in current DARP.")

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

    def _reward_for_ids(self, state_id: int, action_id: int, action: Mapping[str, Any]) -> float:
        """Evaluate and cache one reward matrix entry. / 求值并缓存 reward 矩阵中的一个元素。"""
        cache_key = (state_id, action_id)
        cached = self._reward_cache.get(cache_key)
        if cached is not None:
            self._cache_hits["reward"] += 1
            return cached
        state = self.state_from_key(self._state_index.key(state_id))
        reward = self.expected_reward(self._context(state, action))
        self._reward_cache[cache_key] = reward
        return reward

    def _state_cost_for_id(self, state_id: int) -> float:
        """Return a cached current-state cost vector entry. / 返回缓存的当前状态 cost 向量元素。"""
        if state_id not in self._state_cost_cache:
            state = self.state_from_key(self._state_index.key(state_id))
            self._state_cost_cache[state_id] = self._state_cost(state, self.risk.state_fluent_costs)
        return self._state_cost_cache[state_id]

    def _next_state_cost_for_id(self, state_id: int) -> float:
        """Return a cached next-state cost vector entry. / 返回缓存的后继状态 cost 向量元素。"""
        if state_id not in self._next_state_cost_cache:
            state = self.state_from_key(self._state_index.key(state_id))
            self._next_state_cost_cache[state_id] = self._state_cost(state, self.risk.next_state_fluent_costs)
        return self._next_state_cost_cache[state_id]

    def _state_failure_for_id(self, state_id: int) -> float:
        """Return a clamped current-state failure probability. / 返回截断后的当前状态失败概率。"""
        return _clamped_probability(self._state_cost_for_id(state_id))

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
            self._cache_hits["observation"] += 1
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
            raise ExactKernelError(f"Missing next-state CPF for {state_name}.")
        return _cpf_expression(value)

    def _observation_distribution(self, context: Mapping[str, Any]) -> Mapping[ObservationKey, float]:
        """Return observation-value distribution from observation CPFs. / 从 observation CPF 返回 observation-value 分布。"""
        partials: dict[ObservationKey, float] = {(): 1.0}
        for obs_name in self.observation_names:
            expr = self.cpfs.get(obs_name) or self.cpfs.get(f"{obs_name}'")
            if expr is None:
                raise ExactKernelError(f"Missing observation CPF for {obs_name}.")
            updated: dict[ObservationKey, float] = {}
            for partial_key, partial_prob in partials.items():
                partial_obs = dict(partial_key)
                value_dist = self.expression_distribution(_cpf_expression(expr), {**context, **partial_obs})
                for value, value_prob in value_dist.items():
                    next_partial = tuple(sorted({**partial_obs, obs_name: _plain_value(value)}.items()))
                    updated[next_partial] = updated.get(next_partial, 0.0) + partial_prob * value_prob
            partials = updated
        return _normalize_distribution(partials)

    def _state_cost(self, state: Mapping[str, Any], costs: Mapping[str, float]) -> float:
        """Return configured state cost for one state. / 返回一个 state 的配置 cost。"""
        return sum(float(cost) for fluent, cost in costs.items() if bool(state.get(fluent, False)))

    def _validate_supported(self) -> None:
        """Validate finite bool/enumerable assumptions. / 验证有限 bool/enumerable 假设。"""
        state_ranges = getattr(self.grounded_model, "state_ranges", {}) or {}
        unsupported = [
            name
            for name in self.state_names
            if str(state_ranges.get(name, "bool")) != "bool"
        ]
        if unsupported:
            raise ExactKernelError(
                "Current exact kernel supports bool state fluents only: "
                + ", ".join(unsupported)
            )


def _cpf_expression(value: Any) -> Any:
    """Extract the expression from pyRDDLGym CPF entries. / 从 pyRDDLGym CPF entry 中提取表达式。"""
    if isinstance(value, tuple) and len(value) == 2:
        return value[1]
    return value


def _advance_exact_belief(
    exact_kernel: Any,
    belief: Mapping[StateKey, float],
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    observed_state: Mapping[str, Any] | None,
) -> Mapping[StateKey, float]:
    """Advance an exact belief using kernel-native Bayes update when available. / 使用 kernel 原生 Bayes update 推进 exact belief。"""
    if hasattr(exact_kernel, "update_belief"):
        return exact_kernel.update_belief(
            belief,
            action,
            observation,
            observed_state=observed_state,
        )
    expansion = exact_kernel.expand_action(belief, action)
    observation_key = _observation_key_for_kernel(
        exact_kernel,
        observation,
        observed_state=observed_state,
    )
    for outcome in expansion.observations:
        if outcome.observation == observation_key:
            return outcome.belief
    raise ExactKernelError(
        "Observed outcome is outside exact observation support: "
        f"{observation_key!r}"
    )


def _observation_key_for_kernel(
    exact_kernel: Any,
    observation: Mapping[str, Any],
    *,
    observed_state: Mapping[str, Any] | None,
) -> ObservationKey:
    """Convert a runtime observation through a generic exact-kernel interface. / 通过通用 exact-kernel 接口转换 observation。"""
    if hasattr(exact_kernel, "observation_key"):
        return exact_kernel.observation_key(observation, observed_state=observed_state)
    observation_names = tuple(getattr(exact_kernel, "observation_names", ()) or ())
    if observation_names:
        return tuple((name, _plain_value(observation.get(name, False))) for name in observation_names)
    state = observed_state if observed_state is not None else observation
    initial = exact_kernel.initial_belief_from_state(state)
    if not initial:
        raise ExactKernelError("Cannot build MDP observation key from an empty state belief.")
    state_key = next(iter(initial))
    return (("__state__", state_key),)


def _belief_support(exact_kernel: Any, belief: Mapping[StateKey, float]) -> dict[str, float]:
    """Return compact labels for an exact belief support. / 返回 exact belief support 的紧凑标签。"""
    support: dict[str, float] = {}
    for state, probability in belief.items():
        label = exact_kernel.state_label(state)
        support[label] = support.get(label, 0.0) + float(probability)
    return dict(sorted(support.items()))


def _json_ready(value: Any) -> Any:
    """Convert exact belief fields to JSON-friendly values. / 将 exact belief 字段转为 JSON 友好值。"""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return value.item()
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
        raise ExactKernelError(f"{name} expects {expected} arguments, got {len(args)}.")


def _plain_value(value: Any) -> Hashable:
    """Convert numpy scalar values to hashable Python values. / 将 numpy scalar 转为可哈希 Python 值。"""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _normalize_distribution(distribution: Mapping[Hashable, float]) -> dict[Hashable, float]:
    """Normalize a finite probability distribution. / 归一化有限概率分布。"""
    cleaned = {key: float(value) for key, value in distribution.items() if abs(float(value)) > 1e-15}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in cleaned.items()}


def _clamped_probability(value: float) -> float:
    """Clamp a numeric selector to a probability. / 将数值 selector 截断为概率。"""
    return max(0.0, min(1.0, float(value)))


def _expectation(distribution: Mapping[Hashable, float]) -> float:
    """Return numeric expectation of a finite distribution. / 返回有限分布的数值期望。"""
    return sum(float(value) * probability for value, probability in distribution.items())


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
    raise ExactKernelError(f"Unsupported arithmetic operator: {op}")


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
    raise ExactKernelError(f"Unsupported logical operator: {op}")


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
    raise ExactKernelError(f"Unsupported relational operator: {op}")


def _observation_label(observation: ObservationKey) -> str:
    """Return a compact observation label. / 返回紧凑 observation label。"""
    active = [name for name, value in observation if value is True]
    if active:
        return ",".join(active)
    return repr(dict(observation))
