# DARP result and policy format

DARP writes UTF-8 JSON rather than Python pickle so the result can be read by executors in any language. The top-level object is versioned:

```json
{
  "format": "darp-result",
  "version": 1,
  "elapsed_s": 0.42,
  "risk_budget": 0.1,
  "decision": {
    "action": {"move_up": true, "move_right": false},
    "label": "move_up",
    "complete": true,
    "value": -5.0,
    "value_kind": "achieved_utility",
    "timing": {},
    "policy": {
      "format": "darp-policy-graph",
      "version": 1,
      "input": "observation",
      "timing": "act-then-observe",
      "root": "n0",
      "nodes": [
        {
          "id": "n0",
          "stage": 0,
          "action_label": "move_up",
          "action": {"move_up": true, "move_right": false},
          "transitions": [
            {"observation": {"obs": 0}, "next": "n1"},
            {"observation": {"obs": 1}, "next": null}
          ]
        },
        {
          "id": "n1",
          "stage": 1,
          "action_label": "move_right",
          "action": {"move_up": false, "move_right": true},
          "transitions": [
            {"observation": {"obs": 0}, "next": null},
            {"observation": {"obs": 1}, "next": null}
          ]
        }
      ],
      "complete": true,
      "feasible": true,
      "solver_status": "optimal",
      "achieved_utility": -5.0,
      "active_constraint_value": 0.08
    }
  }
}
```

## Policy-graph contract

- `root` identifies the first action node.
- `format` and `version` version the standalone policy payload; DARP version 1 emits a finite acyclic tree encoded as a graph.
- Each node contains the complete grounded RDDL action assignment. The executor sends `action` to the environment and uses it when evaluating the duration expression; `action_label` is descriptive.
- `stage` is the number of prior actions and must increase by one along every edge.
- After executing the node action, match the returned grounded fluent map against one `transitions[].observation` object.
- `next` names the next action node. `null` means that the policy has reached a valid terminal leaf.
- `input = "observation"` means transition objects contain RDDL observation fluents. `input = "state"` means they contain state fluents for a fully observable model.
- `timing = "act-then-observe"` means the node action is executed before its outgoing observation edge is selected.
- Fluent and node ordering has no semantic meaning. Names and JSON scalar values must match exactly.
- Only a policy with `complete = true` and `feasible = true` is executable.

Equivalent executor pseudocode is:

```text
node = policy.root
loop:
    execute(policy.nodes[node].action)
    values = receive_observation_or_state()
    edge = exact_match(policy.nodes[node].transitions, values)
    if edge.next is null: stop
    node = edge.next
```

`darp.executor.PolicyExecutor` directly subclasses pyRDDLGym 2.7's `BaseAgent` and implements `reset()`, `sample_action(state)` and a policy-leaf-aware `evaluate(...)`. Python callers can therefore use it through the normal pyRDDLGym agent interface; a manual loop must stop when `sample_action(...)` returns `None`. The caller supplies the pyRDDLGym environment, so source RDDL paths belong in experiment configuration or human-readable notes rather than the executable policy schema. Other languages only need the contract above.

## Load and execute a saved result

Run this example from the repository root. The RDDL environment must match the problem used to generate the saved policy.

```python
from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.solve import DARPResult

domain = "experiments/DARP-vs-RAOstar-grid/rddl/domain.rddl"
instance = "experiments/DARP-table1-grid/rddl/instance_e_h3.rddl"
result = DARPResult.load(
    "experiments/DARP-table1-grid/output/results/smoke-raw/"
    "darp-e-h3-d0p1-full-ilp-trial01.json"
)
agent = PolicyExecutor(result.decision.policy)

env = load_rddl(domain, instance).env
statistics = agent.evaluate(
    env, episodes=1000, seed=0,
    risk_path="experiments/DARP-vs-RAOstar-grid/rddl/risk.json",
)
```

`evaluate(env)` is also valid. Supply `risk_path` to measure first-entry risk frequency. Duration comes from the environment's DARP-extended RDDL AST; no duration JSON or policy metadata is required.

## Evaluation results

The executor checks the policy graph, follows its observation edges and stops at a policy leaf or environment termination. `evaluate` returns pyRDDLGym reward statistics and execution measurements:

| Statistics key | Meaning |
| --- | --- |
| `mean`, `median`, `min`, `max`, `std` | Sampled discounted returns using the original RDDL reward. |
| `risk_rate` | Fraction of episodes that enter the risky set at least once, including the initial state; present when `risk_path` is supplied. |
| `physical_duration_mean` | Mean accumulated action duration along the sampled true-state trajectories. |
| `rollout_time_s`, `episodes` | Total rollout wall-clock time and number of episodes. |

An episode is marked failed once, even if it repeatedly visits risky states. Execution continues according to the policy after failure. Hidden simulator state is used for risk and duration measurements only; a POMDP policy still receives observations. `risk_rate` is a sample frequency, not the solver's probability guarantee.

Physical duration is read from the original RDDL expression at each action's true starting state. Fixed and state-dependent values are accumulated directly; Normal durations use a separate random stream without changing environment transition/observation sampling. Normal variance is retained even when planning uses only its expectation. Physical duration is distinct from the paper's belief-based duration stopping criterion and from execution wall-clock time.

The experiments keep solver objectives in their main tables and episode count, risk frequency, physical duration and execution time in supplementary tables. `statistics["mean"]` remains available for inspecting raw RDDL reward; execution does not apply the planner's optional terminal heuristic or compare results with paper tables.

## Why this representation

The format follows the finite-controller convention used by [pomdp-solve policy graphs](https://www.pomdp.org/code/pg-file-spec.html), where an action node advances along an observation-labelled edge, and the `plan`/`update` interface exposed by [pomdp_py](https://h2r.github.io/pomdp-py/html/examples.external_solvers.html#policygraph-and-alphavectorpolicy). It also mirrors the action-list/induced-strategy views offered by [PRISM](https://www.prismmodelchecker.org/manual/RunningPRISM/Strategies).

The legacy `.pg` format uses integer action and observation positions tied to a separate `.pomdp` file, so it cannot preserve named, factored RDDL assignments by itself. SARSOP `.policy` files contain alpha vectors and represent a different belief-to-action policy class. pyRDDLGym's [RDDL policy block](https://pyrddlgym.readthedocs.io/en/latest/rddl.html#policy-blocks) is useful for compact symbolic or parametric policies, but an expanded finite observation tree is clearer and more portable as data. DARP therefore uses the same policy-graph semantics with self-describing RDDL fluent names in JSON.
