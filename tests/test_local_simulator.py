"""Tests for the DARP local simulator."""

from darp.rddl.compiler import RDDLCompiler
from darp.rddl.loader import RDDLLoader
from darp.sim.local import LocalSimulator

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_local_simulator_runs_compiled_tiny_grid():
    """Check that compiled RDDL dynamics change simulator state. / 检查编译后的 RDDL 动态会改变 simulator state。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)
    simulator = LocalSimulator(problem, seed=7)

    assert simulator.reset() == "c11"

    observation, reward, done, info = simulator.step("move-east")
    assert observation == "c12"
    assert reward == -1.0
    assert done is False
    assert info["state"] == "c12"

    observation, reward, done, info = simulator.step("move-east")
    assert observation == "c13"
    assert reward == -1.0
    assert done is False
    assert info["state"] == "c13"

    observation, reward, done, info = simulator.step("move-south")
    assert observation == "c23"
    assert reward == -1.0
    assert done is False
    assert info["state"] == "c23"

    observation, reward, done, info = simulator.step("move-south")
    assert observation == "c33"
    assert reward == 20.0
    assert done is (simulator.steps >= problem.max_depth)
    assert info["state"] == "c33"

    while not done:
        observation, reward, done, info = simulator.step("move-east")

    assert observation == "c33"
    assert reward == 20.0
    assert done is True
    assert info["state"] == "c33"
    assert simulator.steps == problem.max_depth


def test_local_simulator_rejects_unknown_actions():
    """Check that invalid action names fail clearly. / 检查非法 action 名称会清晰失败。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)
    simulator = LocalSimulator(problem)

    try:
        simulator.step("missing-action")
    except ValueError as exc:
        assert "Unknown action" in str(exc)
    else:
        raise AssertionError("Expected an unknown action error.")
