# DARP

Durative Action RDDL Planner is a Python research prototype for implementing fixed-horizon POMDP / constrained POMDP / durative-action planning algorithms on standard RDDL problems, with a staged path toward the AND-OR tree, full ILP, and HILP search workflow from "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions".

The main branch now follows a **pyRDDLGym-first** architecture: pyRDDLGym owns standard RDDL parsing, semantic checks, grounding, and simulation; DARP focuses on turning pyRDDLGym `model/env/grounded_model` objects into planner-facing runtime and search inputs, then wiring `DurationModel`, AND-OR trees, ILP/full ILP/HILP, and online sessions. DARP's own parser, expression parser, AST visualizer, and EBNF have been preserved on the archive branch and are no longer maintained on main.

## Feature Status

- `adapter/` loads standard RDDL domain/instance files through pyRDDLGym and returns `PyRDDLGymProblem(env, model, native_ast)`.
- `PyRDDLGymProblem.build_grounded_model()` directly reuses pyRDDLGym's `RDDLGrounder(...).ground()` and returns pyRDDLGym's `RDDLGroundedModel`; `GroundedRDDLView` wraps it, so DARP no longer implements grounding itself.
- `GroundedRDDLView.build_and_or_interface()` now turns grounded actions, observation scope, and root history into an AND-OR search interface.
- `darp --domain --instance` can now execute an online trace through pyRDDLGym; the current planner is a small rollout baseline used to validate the runtime/session path.
- `model/` keeps DARP-native `DurationModel` and AND-OR tree data structures for later durative-action sidecars and HILP `tau(q)` calculations.
- Future AND-OR tree / ILP / HILP code should consume pyRDDLGym runtime and grounded-model views directly instead of compiling into a separate DARP `PlanningProblem`.
- Native DARP-RDDL syntax extensions are not implemented on main; durative actions should first enter through YAML/JSON sidecars or Python plugins.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

Optional solver dependencies are reserved for future ILP backends:

```bash
python -m pip install -e ".[highs]"     # future HiGHS backend
python -m pip install -e ".[gurobi]"    # future Gurobi backend
```

## CLI

Show help:

```bash
darp -h
```

Load standard RDDL and execute an online trace through pyRDDLGym:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

Set rollout lookahead:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --lookahead-depth 4 \
  --particles 32
```

Write the online trace as JSON:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_pyrddlgym_trace.json
```

Inspect the pyRDDLGym problem components and future search boundary:

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
  -> planning.RolloutPlanner.choose_action
  -> planning.run_online_session manages action, observation, state, belief, and trace
```

The future paper-aligned path is:

```text
PyRDDLGymProblem.build_grounded_model()
  -> pyRDDLGym.core.compiler.model.RDDLGroundedModel
  -> adapter.GroundedRDDLView (reusing pyRDDLGym grounding)
  -> model.ANDORNode / History
  -> ILP/full ILP/HILP planner
  -> OnlineSession takes one action and receives observation/reward/state from pyRDDLGym env
```

## Architecture

- pyRDDLGym: standard RDDL parser, semantic handling, grounder, Gym-style environment, and simulator.
- DARP `adapter/`: isolates the pyRDDLGym dependency and owns loading, grounded-model views, runtime reset/step, and lightweight belief helpers; split this package only if another external system is added later.
- DARP `model/`: stores DARP-native planning data structures such as duration models, histories, and AND-OR tree nodes; this package does not directly depend on pyRDDLGym.
- DARP `planning/`: stores planners, online sessions, traces, and the future planner registry; this package reaches external simulators through adapters/runtimes.
- Future `search/` / `ilp/` or `planning/search/` / `planning/ilp/`: AND-OR tree, full ILP, HILP, and HiGHS/Gurobi backends directly over `GroundedRDDLView`, runtime, and `model/` structures.

## File Layout

```text
DARP/
├── README.md                         # Chinese primary documentation.
├── README-EN.md                      # English mirror documentation.
├── pyproject.toml                    # Python package metadata and optional solver dependencies.
├── requirements.txt                  # Runtime dependencies, including pyRDDLGym.
├── requirements-dev.txt              # Development/test dependencies.
│
├── examples/
│   ├── benchmarks/                   # PROST/IPC RDDL MDP benchmark corpus.
│   │   ├── README.md                 # Benchmark source notes and import list.
│   │   └── <domain-year>/            # Individual benchmark domain directory.
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
│   │   └── duration.py               # DurationModel interface and fixed/expected/Gaussian duration models.
│   └── planning/                     # Planners and online execution orchestration.
│       ├── __init__.py               # Planning package entrypoint.
│       ├── rollout.py                # Current pyRDDLGym rollout baseline planner.
│       └── session.py                # Online session loop and trace structures.
└── tests/
    ├── test_darp_entrypoint.py       # Top-level CLI and pyRDDLGym online-trace tests.
    ├── test_and_or_tree.py           # DARP AND-OR tree base-structure tests.
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
  - [x] 2.2: Remove DARP-owned parser/AST/expression/visualizer maintenance from main and preserve it on the archive branch
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
- [ ] Phase 5: Deferred and folded into Phase 9
  - [ ] Reason: with only the rollout baseline available, a planner registry and offline replay/evaluation workflow would be premature; they become useful once AND-OR, full ILP, and HILP exist.
- [ ] Phase 6: DurationModel and DARP sidecars
  - [ ] 6.1: Design YAML/JSON duration sidecar schema
  - [ ] 6.2: Wire fixed, expected, and Gaussian durations into runtime, history tree, and HILP `tau(q)`
  - [ ] 6.3: Reserve a Python plugin interface without changing standard RDDL grammar
- [ ] Phase 7: Paper search algorithms
  - [ ] 7.1: Implement AND-OR history tree
  - [ ] 7.2: Implement paper `Expand` and preprocessing
  - [ ] 7.3: Implement full ILP baseline
  - [ ] 7.4: Implement HILP partial-ILP search
- [ ] Phase 8: ILP backend
  - [ ] 8.1: Implement ILP model/backend protocol and internal backend
  - [ ] 8.2: Add HiGHS
  - [ ] 8.3: Add Gurobi
- [ ] Phase 9: Benchmarks and PROST/rddlsim compatibility
  - [ ] 9.1: Add a planner registry for rollout, AND-OR, full ILP, and HILP
  - [ ] 9.2: Add unified trace output, time-budget fallback, and trace formatting
  - [ ] 9.3: Add offline policy JSON plus replay/evaluation workflow
  - [ ] 9.4: Implement benchmark runner and pyRDDLGym/rddlrepository import checks
  - [ ] 9.5: Implement rddlsim/PROST-style online protocol adapter
  - [ ] 9.6: Add paper-style experiment scripts
  - [ ] 9.7: Evaluate integration between pyRDDLGym visualizers and DARP planner traces

## Testing

```bash
python -m pytest
```

## Current Limitations

- RDDL inputs currently execute online traces through a pyRDDLGym generative runtime; DARP no longer maintains a separate `PlanningProblem` compilation path.
- The grounded AND-OR interface and pyRDDLGym rollout baseline currently enumerate noop and single bool actions only; action combinations and non-bool actions produce clear unsupported errors and remain future planner/action-space work.
- Current POMDP belief uses a lightweight particle approximation for debugging runtime boundaries; benchmark-quality POMDP evaluation needs later likelihood weighting/resampling.
- DARP reuses pyRDDLGym grounding and does not reimplement RDDL grounding or finite-state enumeration.
- Native DARP-RDDL syntax is not maintained on main; durative actions should first use sidecars/plugins.
- AND-OR tree, full ILP, HILP, and HiGHS/Gurobi backends remain future phases.
