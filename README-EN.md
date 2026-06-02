# DARP

Durative Action RDDL Planner is a Python research prototype for implementing fixed-horizon POMDP / constrained POMDP / durative-action planning algorithms on standard RDDL problems, with a staged path toward the AND-OR tree, full-tree baseline, and HILP search workflow from "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions".

The main branch keeps one implementation path: pyRDDLGym owns standard RDDL parsing, grounding, and simulation; DARP uses its own data structures for the AND-OR history tree, converts the fixed-horizon CC-POMDP search problem into the paper's full ILP / HILP p-ILP form, and targets Gurobi as the only ILP solver. Durative actions are currently defined through YAML/JSON sidecars; future native RDDL syntax should extend the pyRDDLGym parser instead of adding a second parser.

## Feature Status

- `adapter/` loads standard RDDL domain/instance files through pyRDDLGym and returns `PyRDDLGymProblem(env, model, native_ast)`.
- `PyRDDLGymProblem.build_grounded_model()` directly reuses pyRDDLGym's `RDDLGrounder(...).ground()` and returns pyRDDLGym's `RDDLGroundedModel`; `GroundedRDDLView` wraps it, so DARP no longer implements grounding itself.
- `GroundedRDDLView.build_and_or_interface()` now turns grounded actions, observation scope, and root history into an AND-OR search interface.
- `darp --domain --instance` defaults to a fast pyRDDLGym + rollout online trace; `--planner full-ilp` / `--planner hilp` switches to the paper-aligned planner path.
- `model/` keeps DARP-native `DurationModel`, duration sidecars, and AND-OR tree data structures, now wired into Phase 7 `tau(q)` pruning.
- `planning/` provides paper-aligned `preprocess`, `Expand`, full-tree baseline, and HILP-style partial frontier search. Phase 8 now connects generated full-tree ILP and HILP frontier p-ILP models to Gurobi.
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

Set lookahead and particle count:

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
  --planner full-ilp \
  --lookahead-depth 4
```

Use the HILP frontier p-ILP planner:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner hilp \
  --lookahead-depth 3 \
  --hilp-iterations 1 \
  --frontier-width 1
```

`full-ilp` / `hilp` call Gurobi when `gurobipy` is installed; without it they use the generated-tree DP fallback for debugging. Use `--require-gurobi` to fail when Gurobi is unavailable.

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
- DARP `ilp/`: stores the binary ILP schema and Gurobi adapter; `planning/` encodes generated AND-OR trees/frontiers into full ILP or HILP p-ILP models.

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
│   │   └── tiny_grid.yaml            # Tiny-grid fixed-duration sidecar.
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
│       ├── preprocess.py             # Paper-search preprocessing; initializes root and frontier.
│       ├── expand.py                 # Paper Expand operation; computes rho/u/r/tau-style metrics.
│       ├── ilp_tree.py               # Encodes generated AND-OR trees/frontiers as full ILP and HILP p-ILP models.
│       ├── full_ilp.py               # Gurobi full-tree ILP planner, with fallback when Gurobi is absent.
│       ├── hilp.py                   # HILP partial frontier search using Gurobi p-ILP frontier selection.
│       ├── rollout.py                # Current pyRDDLGym rollout baseline planner.
│       └── session.py                # Online session loop and trace structures.
└── tests/
    ├── test_darp_entrypoint.py       # Top-level CLI and pyRDDLGym online-trace tests.
    ├── test_and_or_tree.py           # DARP AND-OR tree base-structure tests.
    ├── test_duration_sidecar.py      # Duration sidecar and history-duration tests.
    ├── test_gurobi_ilp.py            # Phase 8 ILP schema, fake Gurobi, full ILP, and HILP p-ILP tests.
    ├── test_phase7_search.py         # Phase 7 preprocess, Expand, full-tree, and HILP tests.
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
  - [x] 5.2: Record planner, duration, decision fallback, and time-budget status in CLI/JSON traces
  - [ ] 5.3: Tune HILP/full-ILP runtime so the paper path can become the default planner
  - [ ] 5.4: Add offline policy JSON plus replay/evaluation workflow
- [x] Phase 6: DurationModel and DARP sidecars
  - [x] 6.1: Design a YAML/JSON duration sidecar schema without `version` / `horizon`
  - [x] 6.2: Wire fixed, expected, and Gaussian durations into history tree and HILP `tau(q)` evaluation
  - [x] 6.3: Keep durations in YAML/JSON sidecars without changing standard RDDL grammar
- [x] Phase 7: Paper search algorithm scaffolding
  - [x] 7.1: Implement AND-OR history tree
  - [x] 7.2: Implement paper `Expand` and preprocessing
  - [x] 7.3: Implement the generated full-tree DP baseline as the no-Gurobi fallback and diagnostic value path
  - [x] 7.4: Implement E/F bookkeeping for HILP-style partial frontier search
- [x] Phase 8: Gurobi ILP solving
  - [x] 8.1: Encode generated CC-POMDP trees into full ILP / p-ILP variables, objectives, and constraints
  - [x] 8.2: Solve generated full-tree ILP with Gurobi while keeping a no-Gurobi fallback
  - [x] 8.3: Solve HILP frontier-selection p-ILP iterations with Gurobi
  - [x] 8.4: Record Gurobi status, runtime, MIP gap, objective, and selected variables through `ILPSolveResult`
- [ ] Phase 9: Benchmarks, experiments, and syntax extension
  - [ ] 9.1: Implement benchmark runner and pyRDDLGym/rddlrepository import checks
  - [ ] 9.2: Implement PROST/rddlsim-style online protocol compatibility
  - [ ] 9.3: Add paper-style experiment scripts
  - [ ] 9.4: If native durative-action syntax is needed, extend the pyRDDLGym parser by inheritance

## Testing

```bash
python -m pytest
```

Phase 8 unit tests use a fake `gurobipy` module to cover DARP's ILP encoding and adapter boundaries, so the base test suite does not require a local Gurobi install. Real solve experiments still require `pip install -e ".[gurobi]"` and a valid license.

## Current Limitations

- RDDL inputs currently execute online traces through a pyRDDLGym generative runtime; DARP no longer maintains a separate `PlanningProblem` compilation path.
- The default CLI planner remains the fast rollout baseline; `full-ilp` / `hilp` are wired into online sessions, but need performance tuning around pyRDDLGym deep copies and generated-tree expansion before becoming the default.
- The grounded AND-OR interface and pyRDDLGym rollout baseline currently enumerate noop and single bool actions only; action combinations and non-bool actions produce clear unsupported errors and remain future planner/action-space work.
- Current POMDP belief uses a lightweight particle approximation for debugging runtime boundaries; benchmark-quality POMDP evaluation needs later likelihood weighting/resampling.
- DARP reuses pyRDDLGym grounding and does not reimplement RDDL grounding or finite-state enumeration.
- Native DARP-RDDL syntax is not maintained on main; durative actions currently enter only through YAML/JSON sidecars.
- Phase 8 ILP currently uses observation branches generated/sampled through pyRDDLGym; full stochastic observation enumeration, risk/cost fluent extraction, and benchmark-scale constrained rows remain Phase 9 work.
