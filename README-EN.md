# DARP

Durative Action RDDL Planner is a Python research prototype for implementing fixed-horizon POMDP / constrained POMDP / durative-action planning algorithms on standard RDDL problems, with a staged path toward the AND-OR tree, full-tree baseline, and HILP search workflow from "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions".

The main branch keeps one implementation path: pyRDDLGym owns standard RDDL parsing, grounding, and simulation; DARP uses its own data structures for the AND-OR history tree, converts the fixed-horizon CC-POMDP search problem into the paper's full ILP / HILP p-ILP form, and targets Gurobi as the only ILP solver. Durative actions are currently defined through YAML/JSON sidecars; future native RDDL syntax should extend the pyRDDLGym parser instead of adding a second parser.

## Feature Status

- `adapter/` loads standard RDDL domain/instance files through pyRDDLGym and returns `PyRDDLGymProblem(env, model, native_ast)`.
- `PyRDDLGymProblem.build_grounded_model()` directly reuses pyRDDLGym's `RDDLGrounder(...).ground()` and returns pyRDDLGym's `RDDLGroundedModel`; `GroundedRDDLView` wraps it, so DARP no longer implements grounding itself.
- `GroundedRDDLView.build_and_or_interface()` now turns grounded actions, observation scope, and root history into an AND-OR search interface.
- `adapter.ExactRDDLKernel` enumerates transition / observation / reward exactly from pyRDDLGym grounded CPFs for finite bool-state models, and propagates the Lemma 3.3 chance-constrained safe belief plus `rho*(q)` risk constants. Online full-ILP/HILP maintains the root belief with exact Bayes belief updates before building the tree.
- `darp --domain --instance` defaults to a fast pyRDDLGym + rollout online trace; `--planner full-ilp` / `--planner hilp` switches to the paper-aligned planner path.
- `model/` keeps DARP-native `DurationModel`, duration sidecars, and AND-OR tree data structures, now wired into Phase 7 `tau(q)` pruning.
- `planning/` provides paper-aligned `preprocess`, `Expand`, full-tree baseline, and HILP-style partial-tree search. The full-ILP no longer applies an extra lookahead-depth cutoff; it expands to the remaining RDDL horizon / duration stopping condition. HILP solves the current `E ∪ F` partial-tree p-ILP with Gurobi and selects the root action from the partial solution with an exact utility/risk frontier heuristic, without falling back to full-ILP.
- `ilp/` provides DARP's small binary ILP schema and the only solver adapter: Gurobi.
- Durative actions are currently defined only through YAML/JSON sidecars. Future native syntax should extend the pyRDDLGym parser by inheritance.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

Gurobi is the only solver target for the paper ILP/HILP path. Install it when running ILP/HILP solving:

```bash
python -m pip install -e ".[gurobi]"
```

## CLI

Show help:

```bash
darp -h
```

Load standard RDDL and execute the default online trace:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

Set rollout/HILP lookahead; `--particles` only affects rollout's approximate POMDP belief:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --lookahead-depth 4 \
  --particles 32
```

Use a duration sidecar and the full-tree ILP planner:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner full-ilp
```

Use the HILP partial-tree p-ILP planner:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner hilp \
  --lookahead-depth 4 \
  --hilp-iterations 4 \
  --frontier-width 1
```

`full-ilp` / `hilp` are the paper-path planners and require working `gurobipy` plus a Gurobi license; they fail directly when Gurobi is unavailable. `rollout` remains the non-Gurobi baseline.

Note: the CC-POMDP planning time budget is not Python wall-clock runtime. DARP uses the RDDL instance `horizon` plus action durations from the sidecar, and `tau(q)` decides whether a history can keep expanding.

## Duration Sidecar

A duration sidecar describes only the action-duration model, not fields already defined by RDDL. The `horizon` comes from the RDDL instance; do not put `horizon` or `version` in the sidecar.

Minimal fixed-duration example:

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
```

`default` is the fallback duration for actions not explicitly listed in `actions`. In the example above, if the RDDL model also has `move-west` and the sidecar omits it, DARP uses `default: 1` as its duration.

Fixed-duration C-POMDP risk constraints can be declared in the same sidecar. The example below treats entering `at___c22` as a risk cost and sets the total risk budget:

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
  move-west: 1
  move-north: 1
risk:
  budget: 0.25
  next_state_fluents:
    at___c22: 1
```

In full-ILP, DARP follows the paper's Lemma 3.3 safe-belief chance constraint: the ILP right-hand side is `R = Delta - r(b0)`, and each action history uses the risk coefficient `rho*(q) * r(b_q)`. `rho*(q)` and the safe belief `b*` are propagated along the AND-OR tree instead of simply accumulating ordinary-belief expected costs.

Stochastic Duration with Percentile Risk Criteria uses `kind: gaussian` and `zeta`. `zeta` is the paper's percentile threshold `\varsigma`; DARP computes action-duration means/variances from Algorithm 2 smoothed-belief marginals and uses them in `tau(q)`:

```yaml
kind: gaussian
zeta: 0.3
default_mean: 1
default_variance: 0.05
state_actions:
  at___c22:
    move-east:
      mean: 2
      variance: 0.25
risk:
  budget: 0.25
  next_state_fluents:
    at___c22: 1
```

Write the online trace as JSON:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_pyrddlgym_trace.json
```

Inspect pyRDDLGym problem components and planner-interface boundaries:

```bash
python -m darp.adapter.loader \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

## Execution Flow

The current main path is deliberately small:

```text
darp CLI
  -> adapter.RDDLLoader.load(domain, instance)
  -> pyRDDLGym.make(...)
  -> PyRDDLGymProblem(env, model, native_ast)
  -> adapter.PyRDDLGymRuntime.reset/step
  -> planning.RolloutPlanner.choose_action (default fast path)
  -> planning.run_online_session manages action, observation, state, belief, and trace
```

The paper-aligned path is:

```text
PyRDDLGymProblem.build_grounded_model()
  -> pyRDDLGym.core.compiler.model.RDDLGroundedModel
  -> adapter.GroundedRDDLView (reusing pyRDDLGym grounding)
  -> model.ANDORNode / History
  -> model.DurationModel (from YAML/JSON sidecar or default unit duration)
  -> planning.preprocess / planning.expand
  -> planning.FullILPPlanner / planning.HILPPlanner
  -> Gurobi full ILP / p-ILP solve
  -> OnlineSession takes one action and receives observation/reward/state from pyRDDLGym env
```

## Architecture

- pyRDDLGym: standard RDDL parser, semantic handling, grounder, Gym-style environment, and simulator.
- DARP `adapter/`: fixed pyRDDLGym adapter for loading, grounded-model views, runtime reset/step, and lightweight belief helpers.
- DARP `model/`: stores DARP-native planning data structures such as duration models, histories, and AND-OR tree nodes; this package does not directly depend on pyRDDLGym.
- DARP `planning/`: stores the rollout baseline, paper search scaffolding, HILP/full-tree planners, online sessions, and traces; this package reaches the simulator through the pyRDDLGym runtime.
- DARP `ilp/`: stores the binary ILP schema and Gurobi adapter; `planning/` encodes exact AND-OR trees/frontiers into full ILP or HILP p-ILP models.

## File Layout

```text
DARP/
├── README.md                         # Chinese primary documentation.
├── README-EN.md                      # English mirror documentation.
├── pyproject.toml                    # Python package metadata and the Gurobi solver extra.
├── requirements.txt                  # Runtime dependencies, including pyRDDLGym.
├── requirements-dev.txt              # Development/test dependencies.
│
├── examples/
│   ├── benchmarks/                   # PROST/IPC RDDL MDP benchmark corpus.
│   │   ├── README.md                 # Benchmark source notes and import list.
│   │   └── <domain-year>/            # Individual benchmark domain directory.
│   ├── durations/                    # DARP duration sidecar examples.
│   │   ├── tiny_grid.yaml            # Tiny-grid fixed-duration + risk sidecar.
│   │   └── tiny_grid_gaussian.yaml   # Tiny-grid Gaussian percentile duration sidecar.
│   └── rddl/                         # Small hand-written RDDL examples.
│       ├── tiny_grid_domain.rddl     # Tiny-grid standard RDDL domain.
│       ├── tiny_grid_instance.rddl   # Tiny-grid instance.
│       ├── factored_door_domain.rddl # Partial-observation toy domain for future adapter regression.
│       └── factored_door_instance.rddl # Factored-door instance.
│
├── src/darp/
│   ├── __init__.py                   # Package version entrypoint.
│   ├── __main__.py                   # Top-level `darp` CLI.
│   ├── adapter/                      # Adapter layer for the current single external system, pyRDDLGym.
│   │   ├── __init__.py               # Adapter package entrypoint.
│   │   ├── problem.py                # PyRDDLGymProblem container, load errors, and pyRDDLGym grounder entrypoint.
│   │   ├── loader.py                 # Loads standard RDDL with pyRDDLGym.
│   │   ├── grounded.py               # GroundedRDDLView wrapper over pyRDDLGym grounded models.
│   │   ├── exact.py                  # Exact finite transition/observation/reward/risk kernel from grounded CPFs.
│   │   └── runtime.py                # pyRDDLGym reset/step/action/belief runtime.
│   ├── model/                        # DARP-native planning data structures.
│   │   ├── __init__.py               # Model package entrypoint.
│   │   ├── and_or_tree.py            # Base AND-OR history tree nodes and history structures.
│   │   ├── duration.py               # DurationModel, HistoryDurationEvaluator, and tau(q) calculations.
│   │   └── duration_sidecar.py       # JSON/YAML duration sidecar loader.
│   ├── ilp/                          # DARP binary ILP schema and Gurobi solver adapter.
│   │   ├── __init__.py               # ILP package entrypoint.
│   │   ├── model.py                  # ILPVariable, ILPLinearConstraint, ILPModelSpec, and solve results.
│   │   └── gurobi.py                 # Gurobi adapter; the only ILP solver for the paper path.
│   └── planning/                     # Planners and online execution orchestration.
│       ├── __init__.py               # Planning package entrypoint.
│       ├── preprocess.py             # Root-frontier initialization helper for Algorithm 1.
│       ├── expand.py                 # Paper Expand operation; computes rho/u/r/tau, backward messages, and smoothed beliefs.
│       ├── ilp_tree.py               # Runs Algorithm 1 preprocessing and encodes full ILP and HILP p-ILP models.
│       ├── full_ilp.py               # Gurobi-only full-tree ILP planner; builds the policy tree via paper Algorithms 1/2 before solving.
│       ├── hilp.py                   # HILP partial-tree search that maintains Algorithm 3 E/F frontier sets and repeatedly solves Gurobi p-ILPs.
│       ├── rollout.py                # Current pyRDDLGym rollout baseline planner.
│       └── session.py                # Online session loop and trace structures.
└── tests/
    ├── test_darp_entrypoint.py       # Top-level CLI and pyRDDLGym online-trace tests.
    ├── test_and_or_tree.py           # DARP AND-OR tree base-structure tests.
    ├── test_duration_sidecar.py      # Duration sidecar and history-duration tests.
    ├── test_exact_kernel.py          # Exact finite kernel, risk sidecar, and Gaussian percentile duration tests.
    ├── test_gurobi_ilp.py            # Phase 8 ILP schema, fake Gurobi, full ILP, and HILP p-ILP tests.
    ├── test_pyrddlgym_runtime.py     # pyRDDLGym runtime and simple online-trace tests.
    └── test_rddl_loader.py           # pyRDDLGym loader, summary, and grounder-reuse tests.
```

## Roadmap

- [x] Phase 1: Project foundation
  - [x] 1.1: README/README-EN, package metadata, and test entrypoint
  - [x] 1.2: Minimal CLI, duration abstractions, and pyRDDLGym runtime tests
  - [x] 1.3: Import PROST/IPC benchmark corpus
- [x] Phase 2: pyRDDLGym-first RDDL input
  - [x] 2.1: Move standard RDDL parser/simulator responsibility to pyRDDLGym
  - [x] 2.2: Remove the DARP-owned parser path and standardize on the pyRDDLGym parser
  - [x] 2.3: Provide `PyRDDLGymProblem` summaries and pyRDDLGym grounder-reuse tests
- [x] Phase 3: pyRDDLGym generative runtime adapter
  - [x] 3.1: Define a DARP planner-facing runtime protocol around pyRDDLGym `reset/step/model`
  - [x] 3.2: Extract type/object/fluent/action metadata while keeping native pyRDDLGym references
  - [x] 3.3: Implement initial bool action candidates, noop/default actions, and action-constraint error propagation
  - [x] 3.4: Define MDP/POMDP observation/state/belief boundaries, with sampling/particle interfaces when states are not enumerable
  - [x] 3.5: Make `darp --domain --instance` execute an online step trace through the pyRDDLGym runtime
- [x] Phase 4: pyRDDLGym grounded-model view
  - [x] 4.1: Wrap pyRDDLGym `RDDLGroundedModel` behind state/action/observation/reward/CPF accessors
  - [x] 4.2: Build the action/observation/history interface required by the AND-OR tree from the grounded model and runtime
  - [x] 4.3: Report unsupported RDDL structures clearly
- [ ] Phase 5: Verifiable execution workflow
  - [x] 5.1: Wire the `full-ilp` / `hilp` paper planners into online sessions and the CLI
  - [x] 5.2: Record planner, duration, decision value, and solve elapsed time in CLI/JSON traces
  - [ ] 5.3: Tune HILP/full-ILP runtime so the paper path can become the default planner
  - [ ] 5.4: Add offline policy JSON plus replay/evaluation workflow
- [x] Phase 6: DurationModel and DARP sidecars
  - [x] 6.1: Design a YAML/JSON duration sidecar schema without `version` / `horizon`
  - [x] 6.2: Wire fixed, expected, and Gaussian durations into history tree and HILP `tau(q)` evaluation
  - [x] 6.3: Keep durations in YAML/JSON sidecars without changing standard RDDL grammar
- [x] Phase 7: Paper search algorithm scaffolding
  - [x] 7.1: Implement AND-OR history tree
  - [x] 7.2: Implement paper `Expand` and preprocessing
  - [x] 7.3: Implement exact policy-tree preprocessing/Expand scaffolding for full-ILP constants
  - [x] 7.4: Implement E/F bookkeeping for HILP-style partial frontier search
- [x] Phase 8: Gurobi ILP solving
  - [x] 8.1: Encode finite exact CC-POMDP trees into full ILP / p-ILP variables, objectives, and constraints
  - [x] 8.2: Solve exact/full-tree ILP with Gurobi as the only solver
  - [x] 8.3: Solve the current HILP `E ∪ F` partial-tree p-ILP with Gurobi and select both the root action and frontier refinements directly from the partial solution
  - [x] 8.4: Record Gurobi status, runtime, MIP gap, objective, and selected variables through `ILPSolveResult`
  - [x] 8.5: Wire duration-sidecar safe-belief chance constraints, Algorithm 2 backward messages / smoothed beliefs, and Gaussian percentile `tau(q)`
  - [x] 8.6: Make online full-ILP/HILP maintain the root belief through `ExactBeliefState` and exact Bayes updates instead of particle belief
- [ ] Phase 9: Benchmarks, experiments, and syntax extension
  - [ ] 9.1: Implement benchmark runner and pyRDDLGym/rddlrepository import checks
  - [ ] 9.2: Implement PROST/rddlsim-style online protocol compatibility
  - [ ] 9.3: Add paper-style experiment scripts
  - [ ] 9.4: Extend action-space enumeration for concurrent action combinations and non-bool actions
  - [ ] 9.5: Implement augmented-state chance-constrained duration and broader random-expression / continuous-distribution support
  - [ ] 9.6: If native durative-action syntax is needed, extend the pyRDDLGym parser by inheritance

## Testing

```bash
python -m pytest
```

Phase 8 unit tests use a fake `gurobipy` module to cover DARP's ILP encoding and adapter boundaries, so the base test suite does not require a local Gurobi install. Real solve experiments still require `pip install -e ".[gurobi]"` and a valid license.

## Current Limitations

- RDDL inputs currently execute online traces through a pyRDDLGym generative runtime; DARP no longer maintains a separate `PlanningProblem` compilation path.
- The default CLI planner remains the fast rollout baseline. `hilp` now uses partial-tree p-ILPs to avoid full-horizon enumeration, but still needs benchmark-scale pruning before becoming the default; `full-ilp` expands to the full remaining RDDL horizon and grows exponentially with action/observation histories.
- The grounded AND-OR interface and pyRDDLGym rollout baseline currently enumerate noop and single bool actions only; action combinations and non-bool actions produce clear unsupported errors and remain future planner/action-space work.
- The rollout baseline still uses a lightweight particle approximation for POMDP belief. Online full-ILP/HILP uses exact Bayes belief updates. Benchmark-quality POMDP evaluation still needs richer initial-belief modeling, reachable-state pruning, and scalable belief representations.
- DARP reuses pyRDDLGym grounding; finite bool grounded models can be enumerated exactly through `ExactRDDLKernel`, while continuous distributions, large state spaces, and complex random expressions remain later benchmark work.
- Native DARP-RDDL syntax is not maintained on main; durative actions currently enter only through YAML/JSON sidecars.
- Phase 8 ILP now supports exact finite transition/observation branches, Algorithm 2 smoothed beliefs, sidecar safe-belief chance constraints, and duration percentile `tau(q)`; augmented-state chance-constrained duration, benchmark-scale pruning, continuous distributions, and complex random expressions remain Phase 9 work.
