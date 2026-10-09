"""Grid Manhattan utility bound. / Grid 的 Manhattan 效用上界。"""

from darp.planning.heuristic import HeuristicInput, UtilityHeuristic


def negative_manhattan(value: HeuristicInput) -> float:
    """Negate remaining distance for reward maximization. / 剩余距离取负以匹配奖励最大化。"""
    return -float(abs(value.state["grid_row"] - value.non_fluents["goal_row"])
                  + abs(value.state["grid_col"] - value.non_fluents["goal_col"]))


MANHATTAN = UtilityHeuristic(name="paper-grid-manhattan", evaluate=negative_manhattan, upper_bound=True)
