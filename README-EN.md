# DARP

Durative Action RDDL Planner is a Python research prototype for implementing fixed-horizon POMDP / constrained POMDP / durative-action planning algorithms on standard RDDL problems, with a staged path toward the AND-OR tree, full ILP, and HILP search workflow from "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions".

The main branch now follows a **pyRDDLGym-first** architecture: pyRDDLGym owns standard RDDL parsing, baseline semantic handling, and simulation; DARP focuses on adapting pyRDDLGym `model/env` artifacts into planner runtimes, generating research-oriented `PlanningProblem` objects only for small enumerable cases, wiring `DurationModel`, and implementing online/offline planners. DARP's own parser, expression parser, AST visualizer, and EBNF have been preserved on the archive branch and are no longer maintained on main.

## Feature Status

- Load standard RDDL domain/instance files through pyRDDLGym and return `RDDLEnv`, `RDDLLiftedModel`, and pyRDDLGym's native AST.
- `darp --domain --instance` can now execute an online trace through pyRDDLGym; the current planner is a small rollout baseline used to validate the runtime path.
- Provide a `PyRDDLGymPlanningAdapter` boundary; the next implementation step is a pyRDDLGym generative runtime wrapper, with explicit enumeration only for small finite problems.
- Keep DARP's explicit `PlanningProblem`, `DurationModel`, and finite-horizon DP/belief helpers for future enumerable RDDL, ILP/HILP, and algorithm unit tests.
- Current RDDL inputs first run through a pyRDDLGym generative runtime; converting pyRDDLGym models into DARP `PlanningProblem` objects is not implemented yet.
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

Inspect pyRDDLGym artifacts directly:

```bash
python -m darp.rddl.compiler \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

## Architecture

- pyRDDLGym: standard RDDL parser, semantic handling, Gym-style environment, and simulator capability.
- DARP `rddl/`: loads pyRDDLGym artifacts and reserves both the pyRDDLGym runtime boundary and the small-scale `model/env -> PlanningProblem` adapter boundary.
- DARP `core/`: explicit finite-horizon model, typed identifiers, and duration abstractions used by planners.
- DARP `online.py`: DP and belief helpers over explicit `PlanningProblem` objects for future enumerable RDDL and algorithm tests.
- Future planner interface: first support pyRDDLGym generative reset/step, then build explicit `PlanningProblem` objects only for enumerable small discrete cases.
- Future `search/` / `ilp/`: AND-OR tree, full ILP, HILP, and HiGHS/Gurobi backends on top of stable planner interfaces.

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
│   ├── online.py                     # Finite-horizon online replanning over explicit PlanningProblem objects.
│   │
│   ├── core/
│   │   ├── __init__.py               # core package entrypoint.
│   │   ├── duration.py               # DurationModel interface and fixed/expected/Gaussian duration models.
│   │   ├── problem.py                # DARP explicit PlanningProblem model.
│   │   └── types.py                  # State, Action, Observation, and related aliases.
│   │
│   ├── rddl/
│   │   ├── __init__.py               # pyRDDLGym-backed RDDL package entrypoint.
│   │   ├── artifacts.py              # RDDLArtifacts container and RDDLLoadError.
│   │   ├── loader.py                 # Loads standard RDDL with pyRDDLGym and returns artifacts.
│   │   ├── runtime.py                # pyRDDLGym reset/step runtime and rollout online trace.
│   │   └── compiler.py               # Future pyRDDLGym runtime and optional PlanningProblem enumeration boundary.
└── tests/
    ├── test_darp_entrypoint.py       # Top-level CLI and pyRDDLGym online-trace tests.
    ├── test_online.py                # Online replanning and belief-update tests.
    ├── test_pyrddlgym_runtime.py     # pyRDDLGym runtime and simple online-trace tests.
    └── test_rddl_loader.py           # pyRDDLGym loader and future adapter-boundary tests.
```

## Roadmap

- [x] Phase 1: Project foundation
  - [x] 1.1: README/README-EN, package metadata, and test entrypoint
  - [x] 1.2: Explicit `PlanningProblem`, belief helpers, and online DP unit tests
  - [x] 1.3: Import PROST/IPC benchmark corpus
- [x] Phase 2: pyRDDLGym-first RDDL input
  - [x] 2.1: Move standard RDDL parser/simulator responsibility to pyRDDLGym
  - [x] 2.2: Remove DARP-owned parser/AST/expression/visualizer maintenance from main and preserve it on the archive branch
  - [x] 2.3: Provide pyRDDLGym artifact summaries and adapter-boundary tests
- [x] Phase 3: pyRDDLGym generative runtime adapter
  - [x] 3.1: Define a DARP planner-facing runtime protocol around pyRDDLGym `reset/step/model`
  - [x] 3.2: Extract type/object/fluent/action metadata while keeping native pyRDDLGym references
  - [x] 3.3: Implement initial bool action candidates, noop/default actions, and action-constraint error propagation
  - [x] 3.4: Define MDP/POMDP observation/state/belief boundaries, with sampling/particle interfaces when states are not enumerable
  - [x] 3.5: Make `darp --domain --instance` execute an online step trace through the pyRDDLGym runtime
- [ ] Phase 4: Small finite RDDL to explicit `PlanningProblem`
  - [ ] 4.1: Add finite-discrete enumerability checks with clear errors for continuous, too-large, or unsupported structures
  - [ ] 4.2: Extract reward, transition, and observation tables for deterministic and supported finite stochastic structures
  - [ ] 4.3: Convert enumerable RDDL into `PlanningProblem` and regression-test tiny/factored examples
- [ ] Phase 5: Verifiable baseline solver
  - [ ] 5.1: Let baseline planners work over both pyRDDLGym generative runtime and explicit `PlanningProblem`
  - [ ] 5.2: Add planner registry, unified trace output, and time-budget fallback
  - [ ] 5.3: Add offline policy JSON plus replay/evaluation workflow
- [ ] Phase 6: DurationModel and DARP sidecars
  - [ ] 6.1: Design YAML/JSON duration sidecar schema
  - [ ] 6.2: Wire fixed, expected, and Gaussian durations into the runtime adapter and explicit `PlanningProblem`
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
  - [ ] 9.1: Implement benchmark runner and pyRDDLGym/rddlrepository import checks
  - [ ] 9.2: Implement rddlsim/PROST-style online protocol adapter
  - [ ] 9.3: Add paper-style experiment scripts
  - [ ] 9.4: Evaluate integration between pyRDDLGym visualizers and DARP planner traces

## Testing

```bash
python -m pytest
```

## Current Limitations

- RDDL inputs currently execute online traces through a pyRDDLGym generative runtime; they do not yet produce DARP `PlanningProblem` objects.
- The pyRDDLGym rollout baseline currently enumerates noop and single bool actions only; action combinations and non-bool actions are future planner/action-space work.
- Current POMDP belief uses a lightweight particle approximation for debugging runtime boundaries; benchmark-quality POMDP evaluation needs later likelihood weighting/resampling.
- General RDDL cannot be assumed to enumerate into a full table MDP/POMDP; main will first support a generative runtime, with explicit enumeration only for small finite discrete cases.
- Explicit `PlanningProblem` DP helpers are for algorithm unit tests and future small-scale enumeration paths, not replacements for the pyRDDLGym runtime.
- Native DARP-RDDL syntax is not maintained on main; durative actions should first use sidecars/plugins.
- AND-OR tree, full ILP, HILP, and HiGHS/Gurobi backends remain future phases.
