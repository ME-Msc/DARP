"""Run one pinned upstream RAO* process and emit a machine-readable result.

It executes inside the exact environment declared by ``raostar_quad.json`` and
imports the upstream implementation; it does not contain a RAO* implementation.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import sys
import time


RESULT_PREFIX = "DARP_EXTERNAL_RAOSTAR_RESULT="


def _load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _environment_report(manifest):
    expected = manifest["tested_environment"]
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    actual_packages = {}
    problems = []
    for package, expected_version in sorted(expected["packages"].items()):
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
        actual_packages[package] = actual_version
        if actual_version != expected_version:
            problems.append(
                "{}: expected {}, found {}".format(
                    package,
                    expected_version,
                    actual_version if actual_version is not None else "missing",
                )
            )
    if actual_python != expected["python"]:
        problems.insert(
            0,
            "python: expected {}, found {}".format(
                expected["python"], actual_python
            ),
        )
    return {
        "compatible": not problems,
        "policy": expected["compatibility_policy"],
        "expected_python": expected["python"],
        "actual_python": actual_python,
        "expected_packages": expected["packages"],
        "actual_packages": actual_packages,
        "problems": problems,
    }


def _json_number(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _termination_reason(search_complete, elapsed, time_limit, iterations, iter_limit):
    if search_complete:
        return "complete"
    if math.isfinite(iter_limit) and iterations is not None and iterations > iter_limit:
        return "iteration-limit"
    if math.isfinite(time_limit) and elapsed >= max(0.0, time_limit * 0.99):
        return "time-limit"
    return "partial-native-termination"


def _quad_observation_label(observation):
    return json.dumps(observation, separators=(",", ":"), allow_nan=False)


def _quad_policy_outcomes(model, belief, action):
    grouped = {}
    for state, state_probability in belief.items():
        if action not in tuple(model.actions(state)):
            raise ValueError(
                "upstream QuadModel action {!r} unavailable at {!r}".format(
                    action, state
                )
            )
        for successor, transition_probability in model.state_transitions(state, action):
            for observation, observation_probability in model.observations(successor):
                label = _quad_observation_label(observation)
                branch = grouped.setdefault(label, {})
                branch[successor] = branch.get(successor, 0.0) + (
                    float(state_probability)
                    * float(transition_probability)
                    * float(observation_probability)
                )
    outcomes = []
    for label in sorted(grouped):
        branch = grouped[label]
        total = sum(branch.values())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("upstream Quad callbacks produced invalid observation mass")
        outcomes.append((label, {state: value / total for state, value in branch.items()}))
    return outcomes


def _beliefs_close(left, right):
    return set(left) == set(right) and all(
        math.isclose(
            float(left[state]),
            float(right[state]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        for state in left
    )


def _export_quad_policy(model, graph, initial_belief, horizon):
    """Serialize one complete native graph without modifying the model."""

    policy = {}

    def recurse(node, belief, history, depth):
        if not _beliefs_close(node.state.belief, belief):
            raise ValueError(
                "RAO* Quad graph belief differs from callback-generated belief"
            )
        positive_states = [
            state for state, probability in belief.items()
            if float(probability) > 0.0
        ]
        if not positive_states:
            raise ValueError("RAO* Quad policy reached an empty belief")
        if depth == horizon or all(
            bool(model.is_terminal(state)) for state in positive_states
        ):
            return
        if node.best_action is None:
            raise ValueError("complete RAO* Quad graph has a nonterminal dead end")
        action = str(node.best_action.name)
        policy[history] = action
        outcomes = _quad_policy_outcomes(model, belief, action)
        remaining = list(graph.hyperedge_successors(node, node.best_action))
        if len(remaining) != len(outcomes):
            raise ValueError("RAO* Quad hyperedge branch count differs from callbacks")
        for label, posterior in outcomes:
            matches = [
                child
                for child in remaining
                if _beliefs_close(child.state.belief, posterior)
            ]
            if not matches:
                raise ValueError("RAO* Quad child posterior differs from callbacks")
            child = matches[0]
            remaining.remove(child)
            recurse(child, posterior, history + (label,), depth + 1)

    recurse(graph.root, initial_belief, (), 0)
    rules = [
        {"observations": list(history), "action": action}
        for history, action in sorted(
            policy.items(), key=lambda item: (len(item[0]), item[0], item[1])
        )
    ]
    payload = {
        "schema_version": 1,
        "scenario": "upstream-quad",
        "horizon": int(horizon),
        "rules": rules,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _build_upstream_quad(scenario):
    """Instantiate the unmodified model used by upstream ``quad_raos.py``."""

    from quad_model import QuadModel
    config = scenario["canonical_parameters"]
    world_size = tuple(config["world_size"])
    goal_state = tuple(config["goal_state"])
    quad_initial = tuple(config["quad_initial"])
    guest_initial = tuple(config["guest_initial"])
    if (
        world_size != (7, 7)
        or goal_state != (5, 5, 90)
        or quad_initial != (1, 1, 90, 0)
        or guest_initial != (3, 1, 90, 0)
    ):
        raise ValueError("upstream-quad parameters must exactly match quad_raos.py")
    model = QuadModel(world_size, goal_state)
    if getattr(model, "optimization", None) != "minimize":
        raise ValueError("upstream QuadModel optimization must be 'minimize'")
    return model, {(quad_initial, guest_initial): 1.0}


def _run_quad(args, manifest):
    checkout = Path(args.checkout).resolve()
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(checkout / "models"))

    from raostar import RAOStar

    scenario = manifest["scenarios"]["quad"]
    adapter = scenario.get("adapter", {})
    if adapter.get("kind") != "upstream-quad":
        raise ValueError("Quad manifest must use the upstream-quad adapter")
    model, initial_belief = _build_upstream_quad(scenario)

    planner = RAOStar(
        model,
        cc=args.chance_constraint,
        cc_type="o",
        fixed_horizon=args.horizon,
        terminal_prob=1.0,
        debugging=False,
        randomization=0.0,
        halt_on_violation=False,
        random_node_selection=False,
    )

    captured = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        _policy, graph = planner.search(
            initial_belief,
            time_limit=args.time_limit,
            iter_limit=args.iter_limit,
        )
    elapsed = time.perf_counter() - started
    stdout = captured.getvalue()
    match = re.search(r"after\s+(\d+)\s+iterations", stdout)
    iterations = int(match.group(1)) if match else None
    open_nodes = len(planner.opennodes)
    search_complete = open_nodes == 0
    warning_count = stdout.count("Warning: root value improved. Check admissibility")
    best_action = getattr(graph.root, "best_action", None)
    policy_protocol = (
        _export_quad_policy(model, graph, initial_belief, args.horizon)
        if search_complete
        else None
    )

    return {
        "status": "ok",
        "elapsed_s": elapsed,
        "search_complete": search_complete,
        "termination_reason": _termination_reason(
            search_complete,
            elapsed,
            args.time_limit,
            iterations,
            args.iter_limit,
        ),
        "iterations": iterations,
        "root_value": _json_number(graph.root.value),
        "root_execution_risk": _json_number(graph.root.exec_risk),
        "root_action": None if best_action is None else str(best_action.name),
        "policy_protocol": policy_protocol,
        "graph_nodes": len(graph.nodes),
        "admissibility_warning_count": warning_count,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--chance-constraint", type=float, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--time-limit", type=float, required=True)
    parser.add_argument("--iter-limit", type=float, default=math.inf)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    manifest = _load_manifest(args.manifest)
    environment = _environment_report(manifest)
    if not environment["compatible"]:
        result = {
            "status": "environment-incompatible",
            "environment": environment,
            "message": "The upstream source was not executed under an untested numerical environment.",
        }
    else:
        try:
            if tuple(manifest.get("scenarios", {})) != ("quad",):
                raise ValueError("External RAO* worker only supports the Quad manifest")
            result = _run_quad(args, manifest)
        except Exception as error:  # keep the process boundary machine-readable
            result = {
                "status": "upstream-error",
                "environment": environment,
                "error_type": type(error).__name__,
                "message": str(error),
            }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
