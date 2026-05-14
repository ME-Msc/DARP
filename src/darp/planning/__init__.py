"""Planning algorithms and online session orchestration."""

# TODO(phase-9.1): Expose the final HILP/Gurobi planner as the stable planning API.

from darp.planning.expand import ExpandedAction, ExpansionMetrics, expand_frontier_item
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner, HILPSearchStats
from darp.planning.preprocess import FrontierItem, PreprocessedSearchTree, preprocess_search_tree
from darp.planning.rollout import ActionDecision, RolloutPlanner, action_label
from darp.planning.session import OnlineSessionResult, OnlineStep, run_online_session

__all__ = [
    "ActionDecision",
    "ExpandedAction",
    "ExpansionMetrics",
    "FrontierItem",
    "FullILPPlanner",
    "HILPPlanner",
    "HILPSearchStats",
    "OnlineSessionResult",
    "OnlineStep",
    "PreprocessedSearchTree",
    "RolloutPlanner",
    "action_label",
    "expand_frontier_item",
    "preprocess_search_tree",
    "run_online_session",
]
