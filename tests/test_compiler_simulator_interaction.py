"""Executable demo test for compiler and simulator interaction."""

from __future__ import annotations

from dataclasses import dataclass

from darp.rddl.compiler import RDDLCompiler
from darp.rddl.loader import RDDLLoader
from darp.sim.local import LocalSimulator

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


@dataclass(frozen=True)
class DemoStep:
    """Store one visible simulator step. / 保存一条可见的 simulator 执行步骤。"""

    action: str
    observation: object
    reward: float
    done: bool
    state: object


def run_compiler_simulator_demo() -> tuple[object, list[DemoStep]]:
    """Compile tiny-grid RDDL and run a short local simulation. / 编译 tiny-grid RDDL 并运行一段本地仿真。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)
    simulator = LocalSimulator(problem, seed=7)

    initial_observation = simulator.reset()
    steps: list[DemoStep] = []
    prefix_actions = ("move-east", "move-east", "move-south", "move-south")
    while len(steps) < problem.max_depth:
        action = prefix_actions[len(steps)] if len(steps) < len(prefix_actions) else "move-east"
        observation, reward, done, info = simulator.step(action)
        steps.append(
            DemoStep(
                action=action,
                observation=observation,
                reward=reward,
                done=done,
                state=info["state"],
            )
        )
    return problem, [DemoStep("reset", initial_observation, 0.0, False, initial_observation), *steps]


def print_demo_trace() -> None:
    """Print the compiler summary and simulator trace. / 打印 compiler 摘要和 simulator 轨迹。"""
    problem, trace = run_compiler_simulator_demo()
    print("Compiler summary")
    print(f"  name: {problem.name}")
    print(f"  mode: {problem.metadata['compiler_mode']}")
    print(f"  states: {', '.join(str(state) for state in problem.states)}")
    print(f"  actions: {', '.join(problem.actions)}")
    print(f"  nonzero transitions: {sum(1 for value in problem.transitions.values() if value > 0)}")
    print("Simulator trace")
    for index, step in enumerate(trace):
        if step.action == "reset":
            print(f"  t={index}: reset -> observation={step.observation}, state={step.state}")
        else:
            print(
                f"  t={index}: action={step.action}, observation={step.observation}, "
                f"reward={step.reward}, done={step.done}, state={step.state}"
            )


def test_compiler_simulator_interaction_demo():
    """Check and print the end-to-end compiler/simulator interaction. / 检查并打印 compiler/simulator 端到端交互。"""
    problem, trace = run_compiler_simulator_demo()
    print_demo_trace()

    assert problem.metadata["compiler_mode"] == "grounded-rddl-expressions"
    assert [step.state for step in trace[:5]] == ["c11", "c12", "c13", "c23", "c33"]
    assert [step.reward for step in trace[:5]] == [0.0, -1.0, -1.0, -1.0, 20.0]
    assert len(trace) == problem.max_depth + 1
    assert trace[-1].done is True
    assert trace[-1].state == "c33"


if __name__ == "__main__":
    print_demo_trace()
