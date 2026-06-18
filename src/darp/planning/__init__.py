"""Planning algorithms and online session orchestration."""

# TODO(phase-9.1): Expose the final HILP/Gurobi planner as the stable planning API.

from darp.planning.expand import ExpandedAction, ExpansionMetrics, expand_frontier_item
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner, HILPSearchStats
from darp.planning.ilp_tree import (
    Algorithm1ExpansionRecord,
    PolicyTreeILP,
    build_full_tree_ilp,
    paper_preprocess,
)
from darp.planning.preprocess import FrontierItem, RootFrontier, initialize_root_frontier
from darp.planning.rollout import ActionDecision, RolloutPlanner, action_label
from darp.planning.session import OnlineSessionResult, OnlineStep, PlannerName, run_online_session

__all__ = [
    "ActionDecision",
    "Algorithm1ExpansionRecord",
    "ExpandedAction",
    "ExpansionMetrics",
    "FrontierItem",
    "FullILPPlanner",
    "HILPPlanner",
    "HILPSearchStats",
    "OnlineSessionResult",
    "OnlineStep",
    "PlannerName",
    "PolicyTreeILP",
    "RootFrontier",
    "RolloutPlanner",
    "action_label",
    "build_full_tree_ilp",
    "expand_frontier_item",
    "initialize_root_frontier",
    "paper_preprocess",
    "run_online_session",
]
