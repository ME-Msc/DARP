"""Validate and execute solved DARP policies through pyRDDLGym."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean, median, pstdev
from time import perf_counter
from typing import Any

from pyRDDLGym.core.policy import BaseAgent

from darp.adapter.kernel import ObservationKey
from darp.planning.policy import (
    ConditionalPolicy,
    PolicyNode,
    policy_input_key,
)


class MissingPolicyTransitionError(LookupError):
    """Raised when an observation has no edge in a complete policy."""


@dataclass(frozen=True, slots=True)
class PolicyExecutionResult:
    """One sampled pyRDDLGym execution of a conditional policy."""

    elapsed_s: float
    steps: int
    total_reward: float
    discounted_return: float
    stop_reason: str
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": self.elapsed_s,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "discounted_return": self.discounted_return,
            "stop_reason": self.stop_reason,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class PolicyExecutor(BaseAgent):
    """Execute a DARP policy through pyRDDLGym's ``BaseAgent`` interface."""

    def __init__(self, policy: ConditionalPolicy) -> None:
        if not policy.duration_complete or policy.feasible is not True:
            raise ValueError("Only complete feasible policies can be executed.")
        self.policy = policy
        self._nodes = {node.node_id: node for node in policy.nodes}
        if len(self._nodes) != len(policy.nodes):
            raise ValueError("Policy contains duplicate node ids.")
        if policy.root not in self._nodes:
            raise ValueError("Policy root does not name an action node.")
        self._max_steps = _validate_policy_graph(self._nodes, policy.root)
        self.reset()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyExecutor:
        """Build an executor from serialized policy data."""
        return cls(ConditionalPolicy.from_dict(value))

    @property
    def history(self) -> tuple[ObservationKey, ...]:
        """Return observations received in the current episode."""
        return self._history

    @property
    def at_leaf(self) -> bool:
        """Return whether the current history is a valid policy leaf."""
        return self._node_id is None

    @property
    def max_steps(self) -> int:
        """Return the greatest action depth encoded by the policy."""
        return self._max_steps

    def reset(self) -> None:
        """Start a new episode at the policy root."""
        self._history: tuple[ObservationKey, ...] = ()
        self._node_id: str | None = self.policy.root
        self._awaiting_observation = False

    def sample_action(self, state: Any) -> dict[str, Any] | None:
        """Return the pyRDDLGym action for the current observation or state.

        DARP uses act-then-observe timing, so the input from ``env.reset()`` is
        ignored. Later calls first consume the outcome of the preceding action.
        """

        if self._awaiting_observation:
            if not isinstance(state, Mapping):
                raise TypeError("Policy input must be a grounded fluent mapping.")
            self._observe(state)
        return self._action()

    def _action(self) -> dict[str, Any] | None:
        if self._awaiting_observation:
            raise RuntimeError("The previous action outcome has not been consumed.")
        if self.at_leaf:
            return None
        node = self._nodes[self._node_id]
        self._awaiting_observation = True
        return dict(node.assignment)

    def _observe(self, observation: Mapping[str, Any]) -> None:
        if not self._awaiting_observation:
            raise RuntimeError("Select an action before consuming its outcome.")
        key = policy_input_key(observation, self.policy.input_kind)
        node = self._nodes[self._node_id]
        try:
            next_node = node.transitions[key]
        except KeyError as error:
            raise MissingPolicyTransitionError(
                f"No transition from {node.node_id!r} for observation {key!r}."
            ) from error
        self._history += (key,)
        self._node_id = next_node
        self._awaiting_observation = False

    def run_episode(
        self,
        env: Any,
        *,
        seed: int | None = None,
        verbose: bool = False,
        render: bool = False,
    ) -> PolicyExecutionResult:
        """Run one policy-leaf-aware episode in a scalar pyRDDLGym env."""

        if bool(getattr(env, "vectorized", False)) != self.use_tensor_obs:
            raise ValueError(
                "RDDLEnv vectorized must match PolicyExecutor.use_tensor_obs."
            )

        # pyRDDLGym also treats the RDDL duration threshold as a step cap, but
        # a stochastic-duration policy may require more action transitions.
        original_horizon = int(getattr(env, "horizon", 0) or 0)
        env.horizon = max(original_horizon, self.max_steps)
        try:
            return self._run_episode(env, seed, verbose, render)
        finally:
            env.horizon = original_horizon

    def _run_episode(
        self,
        env: Any,
        seed: int | None,
        verbose: bool,
        render: bool,
    ) -> PolicyExecutionResult:
        self.reset()
        initial_input, _ = env.reset(seed=seed)
        if verbose:
            print(f"initial input = {initial_input!r}")

        steps = 0
        total_reward = 0.0
        discounted_return = 0.0
        discount = 1.0
        gamma = float(env.discount)
        terminated = False
        truncated = False
        stop_reason = "policy_leaf"
        started = perf_counter()

        for step in range(self.max_steps):
            if render:
                env.render()
            action = self._action()
            if action is None:
                break
            observation, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            reward = float(reward)
            total_reward += reward
            discounted_return += reward * discount
            discount *= gamma
            policy_input = (
                observation if self.policy.input_kind == "observation" else env.state
            )
            if not isinstance(policy_input, Mapping):
                raise TypeError(
                    "RDDL simulator did not return a grounded fluent mapping."
                )
            self._observe(policy_input)

            if verbose:
                print(
                    f"step={step} action={action!r} input={policy_input!r} "
                    f"reward={reward!r}"
                )
            if terminated:
                stop_reason = "model_terminal"
                break
            if self.at_leaf:
                break
            if truncated:
                raise RuntimeError(
                    "RDDL simulator truncated before the policy reached a leaf."
                )
        else:
            if not self.at_leaf and not terminated:
                raise RuntimeError("Policy did not reach a leaf within max_steps.")

        return PolicyExecutionResult(
            elapsed_s=perf_counter() - started,
            steps=steps,
            total_reward=total_reward,
            discounted_return=discounted_return,
            stop_reason=stop_reason,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def evaluate(
        self,
        env: Any,
        episodes: int = 1,
        verbose: bool = False,
        render: bool = False,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Evaluate like ``BaseAgent.evaluate``, stopping at policy leaves."""

        if episodes < 1:
            raise ValueError("episodes must be positive")
        returns: list[float] = []
        for episode in range(episodes):
            result = self.run_episode(
                env,
                seed=seed if episode == 0 else None,
                verbose=verbose,
                render=render,
            )
            returns.append(result.discounted_return)
            if verbose:
                print(
                    f"episode {episode + 1} ended with return "
                    f"{result.discounted_return}"
                )
        return {
            "mean": fmean(returns),
            "median": float(median(returns)),
            "min": min(returns),
            "max": max(returns),
            "std": pstdev(returns),
        }


def _validate_policy_graph(nodes: Mapping[str, PolicyNode], root: str) -> int:
    """Validate one finite graph and return its maximum number of actions."""
    if nodes[root].stage != 0:
        raise ValueError("Policy root must be at stage zero.")
    memo: dict[str, int] = {}
    reached: set[str] = set()

    def depth(node_id: str, active: set[str]) -> int:
        if node_id in active:
            raise ValueError("DARP policies must be acyclic.")
        if node_id in memo:
            reached.add(node_id)
            return memo[node_id]
        node = nodes[node_id]
        if not node.transitions:
            raise ValueError(f"Policy node {node_id!r} has no outcomes.")
        reached.add(node_id)
        child_depths: list[int] = []
        for next_node in node.transitions.values():
            if next_node is None:
                child_depths.append(1)
                continue
            child = nodes.get(next_node)
            if child is None:
                raise ValueError(
                    f"Policy transition names unknown node {next_node!r}."
                )
            if child.stage != node.stage + 1:
                raise ValueError("A policy transition must advance exactly one stage.")
            child_depths.append(1 + depth(child.node_id, active | {node_id}))
        memo[node_id] = max(child_depths)
        return memo[node_id]

    maximum = depth(root, set())
    if reached != set(nodes):
        raise ValueError("Policy contains nodes unreachable from its root.")
    return maximum


__all__ = [
    "MissingPolicyTransitionError",
    "PolicyExecutionResult",
    "PolicyExecutor",
]
