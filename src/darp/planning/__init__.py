"""Planning algorithms and online session orchestration."""

# TODO(phase-9.1): Expose the final HILP/Gurobi planner as the stable planning API.

from darp.planning.expand import ExpandedAction, ExpansionMetrics, expand_frontier_item
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner, HILPSearchStats
from darp.planning.ilp_tree import (
    FrontierSelectionILP,
    GeneratedPolicyTreeILP,
    build_frontier_selection_ilp,
    build_generated_full_tree_ilp,
)
from darp.planning.preprocess import FrontierItem, PreprocessedSearchTree, preprocess_search_tree
from darp.planning.rollout import ActionDecision, RolloutPlanner, action_label
from darp.planning.session import OnlineSessionResult, OnlineStep, PlannerName, run_online_session

__all__ = [
    "ActionDecision",
    "ExpandedAction",
    "ExpansionMetrics",
    "FrontierSelectionILP",
    "FrontierItem",
    "FullILPPlanner",
    "GeneratedPolicyTreeILP",
    "HILPPlanner",
    "HILPSearchStats",
    "OnlineSessionResult",
    "OnlineStep",
    "PlannerName",
    "PreprocessedSearchTree",
    "RolloutPlanner",
    "action_label",
    "build_frontier_selection_ilp",
    "build_generated_full_tree_ilp",
    "expand_frontier_item",
    "preprocess_search_tree",
    "run_online_session",
]
