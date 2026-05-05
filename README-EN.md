# DARP

Durative Action RDDL Planner is a Python research prototype for parsing RDDL problems, compiling them into finite-horizon POMDP / C-POMDP / CC-POMDP-style planning models, and incrementally implementing the full ILP and HILP search workflow from "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions".

The current code focuses on the standard RDDL input pipeline, DARP's small internal simulator, an interactive HTML visualizer, and a local PROST-like online solve loop. External simulator protocols, AND-OR trees, full ILP, HILP, HiGHS/Gurobi backends, and durative-action interfaces will be added in later phases.

## Feature Status

- Parse standard RDDL domain/instance files into DARP-owned `RDDLASTNode` ASTs.
- Align the `darp`, `pyrddl`, and `pyrddlgym` parser frontends through `RDDLFrontend`.
- Ground the currently supported RDDL CPF/reward expressions into a minimal `PlanningProblem`.
- Execute small explicit transition/observation/reward tables with DARP's internal simulator.
- Run the default non-visual online solve loop with `darp --domain DOMAIN.rddl --instance INSTANCE.rddl` and print a readable terminal trace.
- Start the live HTML UI with `--visualizer` to inspect source, AST, and an execution state machine where DARP's planner selects actions and the internal simulator advances states.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install -r requirements-dev.txt
```

If the system Python lacks `ensurepip`, use:

```bash
virtualenv --clear .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install -r requirements-dev.txt
```

Optional dependencies:

```bash
python -m pip install -e ".[rddl]"      # pyRDDLGym
python -m pip install -e ".[pyrddl]"    # pyrddl
python -m pip install -e ".[highs]"     # future HiGHS backend
python -m pip install -e ".[gurobi]"    # future Gurobi backend
```

## Command Line

After editable installation, run:

```bash
darp -h
```

You can also avoid the console script and run:

```bash
python -m darp -h
```

There is only one primary command:

```bash
darp [--domain DOMAIN.rddl --instance INSTANCE.rddl] [--visualizer] [options]
```

Default behavior:

| Default | Value | Description |
| --- | --- | --- |
| mode | `online` | Replan an action from the current belief at each step. |
| visualizer | disabled | Print a readable terminal trace by default; only open the web UI when `--visualizer` is provided. |
| simulator | `darp` | Use DARP's internal simulator to advance state, observation, and reward. |
| frontend | `darp` | Use DARP's own parser/compiler path. |
| host | `127.0.0.1` | Listen on localhost by default. |
| port | `0` | Pick a free port automatically. |

Primary command arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--domain PATH` | no | RDDL domain file path; with `--instance`, compiles explicit RDDL, otherwise non-visual mode uses the built-in demo. |
| `--instance PATH` | no | RDDL instance file path; must be provided with `--domain`. |
| `--mode online` | no | Phase 3 currently supports the local online solve loop. Defaults to `online`. |
| `--frontend {darp,pyrddl,pyrddlgym}` | no | RDDL parser/compiler frontend. Defaults to `darp`. |
| `--simulator {darp,rddlgym,pyrddlgym}` | no | Visualizer runtime simulator. Defaults to `darp`; non-visual mode currently supports only DARP's internal simulator. |
| `--seed N` | no | Random seed for DARP's internal simulator. Defaults to `0`. |
| `--host HOST` | no | Visualizer HTTP host. Defaults to `127.0.0.1`. |
| `--port PORT` | no | Visualizer HTTP port. Defaults to `0`, which chooses a free port. |
| `--no-open` | no | Serve without opening a browser. |
| `--visualizer` | no | Start the live HTML visualizer; requires `--domain` and `--instance`. |
| `--time-budget-ms MS` | no | Soft per-decision time budget recorded in the trace. |
| `--output PATH` | no | In non-visual mode, write the full JSON trace to a file; without this option, no JSON is emitted. |
| `-h`, `--help` | no | Show help text. |

`--frontend` and `--simulator` are not duplicates: `--frontend` controls how RDDL text is parsed/compiled into a DARP model, while `--simulator` controls who receives actions and advances state/observation/reward in the visual runtime. Most users can omit both because both default to `darp`.

The online solve length comes from the RDDL `horizon` compiled into `problem.max_depth`; it is not manually truncated from the command line.

`--seed` controls stochastic sampling in DARP's internal simulator. The current tiny grid is deterministic, so the seed does not change the trajectory; once random initial beliefs, random transitions, or random observations are supported, the seed will make debugging, tests, and benchmarks reproducible.

## Examples

Print a non-visual terminal trace from the RDDL tiny grid:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --seed 7
```

Print a non-visual terminal trace from the built-in demo:

```bash
darp --seed 7
```

Write the full JSON trace to a file:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --seed 7 \
  --output tiny_grid_trace.json
```

Start the live visualizer: DARP chooses actions, the internal simulator advances state, and the browser shows RDDL text, AST, and the runtime state machine.

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

Serve the visualizer on a fixed port without opening a browser:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --host 127.0.0.1 \
  --port 8080 \
  --no-open
```

Compile with a selected frontend while still using DARP's internal simulator:

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --frontend darp
```

Load pyRDDLGym in visualizer mode while hiding DARP's internal state machine:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --simulator rddlgym
```

## Architecture

- `rddl/`: RDDL parser frontends, loading, ASTs, expression grounding, and `PlanningProblem` compilation.
- `core/`: Minimal planning-problem, type, and duration structures for the current phase.
- `online.py`: Local PROST-like online solve loop, terminal traces, optional JSON traces, and finite-horizon online planner.
- `sim/`: DARP's internal simulator and future external simulator adapters.
- `search/`: Future AND-OR tree, Expand, full ILP, and HILP search algorithms.
- `ilp/`: Future internal, HiGHS, and Gurobi ILP backends.

`search/` and `ilp/` are separate layers: `search/` decides how to explore policy trees, while `ilp/` solves the ILP/p-ILP subproblems constructed during search. HiGHS and Gurobi are lower-level ILP backends, not alternatives to HILP.

## RDDLFrontend

`RDDLFrontend` is DARP's parser protocol. Each frontend returns `ParsedRDDL`:

- `ast`: DARP's own `RDDLASTNode`, used by the compiler and visualizer.
- `native_ast`: Third-party parser-native ASTs for debugging or future subclassing.
- `env/model`: Executable artifacts from frontends such as pyRDDLGym.
- `metadata`: Frontend, version, and parsing details.

Current frontends:

- `darp`: DARP's own basic parser, currently the default internal path.
- `pyrddl`: Reuses `pyrddl.parser.RDDLParser`.
- `pyrddlgym`: Reuses pyRDDLGym's parser/simulator ecosystem.

## Repository Layout

```text
DARP/
├── README.md                         # Chinese main documentation for development and maintenance.
├── README-EN.md                      # English mirror documentation for international collaborators.
├── pyproject.toml                    # Python package metadata, console script, and optional dependencies.
├── requirements.txt                  # Runtime dependency record; the current core stays lightweight.
├── requirements-dev.txt              # Development and test dependency record.
│
├── examples/                         # Example input directory.
│   └── rddl/                         # RDDL example files.
│       ├── tiny_grid_domain.rddl     # 3x3 tiny-grid domain with CPF/reward dynamics.
│       └── tiny_grid_instance.rddl   # 3x3 tiny-grid instance with objects and horizon settings.
│
├── src/
│   └── darp/                         # Main DARP Python package.
│       ├── __init__.py               # Package version and top-level metadata.
│       ├── __main__.py               # Top-level `darp` entrypoint for terminal traces and optional `--visualizer` UI.
│       ├── online.py                 # Local online solve loop, terminal traces, optional JSON traces, and finite-horizon dynamic-programming planner.
│       │
│       ├── core/                     # Minimal planning model for the current phase.
│       │   ├── __init__.py           # core package entrypoint.
│       │   ├── types.py              # Shared aliases for states, actions, observations, and transitions.
│       │   ├── duration.py           # Action-duration interface and fixed-duration model.
│       │   └── problem.py            # `PlanningProblem` data structure and built-in tiny-grid model.
│       │
│       ├── rddl/                     # RDDL parsing, loading, compilation, and visualization.
│       │   ├── __init__.py           # rddl package entrypoint.
│       │   ├── ast.py                # DARP-owned `RDDLASTNode` AST node.
│       │   ├── basic_parser.py       # Dependency-free basic structural RDDL parser.
│       │   ├── lexicon.py            # RDDL keywords, block names, and lexical symbols.
│       │   ├── expressions.py        # Standard RDDL expression parsing and evaluation for grounding.
│       │   ├── frontend.py           # `RDDLFrontend` protocol and `ParsedRDDL` container.
│       │   ├── extended.py           # DARP-owned frontend with future extension-syntax hooks.
│       │   ├── pyrddl_frontend.py    # `pyrddl` frontend adapter.
│       │   ├── pyrddlgym_frontend.py # `pyRDDLGym` frontend adapter.
│       │   ├── loader.py             # Loads RDDL through a selected frontend name.
│       │   ├── compiler.py           # Compiles `ParsedRDDL` into `PlanningProblem`.
│       │   └── visualizer.py         # Live HTML visualizer and internal-simulator state-machine panel.
│       │
│       └── sim/                      # Simulator adapter layer.
│           ├── __init__.py           # sim package entrypoint.
│           └── local.py              # DARP internal small simulator backed by explicit tables.
│
└── tests/                            # Current phase tests.
    ├── test_basic_rddl_parser.py     # Parser and HTML visualizer tests.
    ├── test_darp_entrypoint.py       # Top-level `darp` CLI argument and `-h` tests.
    ├── test_rddl_frontends.py        # Frontend loader and third-party parser adapter tests.
    ├── test_rddl_compiler.py         # RDDL-to-`PlanningProblem` compiler tests.
    ├── test_rddl_grounding.py        # CPF/reward grounding behavior tests.
    ├── test_local_simulator.py       # DARP internal simulator tests.
    ├── test_online.py                # Local online solve loop and belief-update tests.
    └── test_compiler_simulator_interaction.py # Compiler/simulator integration tests.
```

## Roadmap

DARP should not hard-code a policy for tiny grid. The roadmap below is already ordered by implementation priority: first make the local solve loop reliable, then expand general RDDL modeling, then implement verifiable baselines, then implement the paper search algorithms, then add external simulators, and finally extend durative actions and DARP-RDDL syntax.

- [x] Phase 1: Project foundation
  - [x] 1.1: Project plan, README/README-EN, and file structure notes
  - [x] 1.2: Python packaging, requirements, and `.venv` workflow
  - [x] 1.3: Minimal RDDL examples and pytest entrypoint
- [x] Phase 2: RDDL input pipeline
  - [x] 2.1: Basic RDDL parser and interactive HTML AST visualizer
  - [x] 2.2: Align `darp`, `pyrddl`, and `pyrddlgym` through `RDDLFrontend`
  - [x] 2.3: Structurally compile `ParsedRDDL` into a minimal `PlanningProblem`
  - [x] 2.4: Complete standard RDDL CPF/reward expression grounding and verify state progression with DARP's internal simulator
    - [x] 2.4.1: Ground tiny-grid CPF/reward dynamics
    - [x] 2.4.2: Implement DARP's internal simulator and run the tiny-grid experiment
    - [x] 2.4.3: Complete general standard RDDL CPF/reward expression grounding without adding new syntax
- [ ] Phase 3: PROST-like realtime execution
  - [x] 3.1: Implement a local online solve loop: replan each step, return actions, receive observations
  - [x] 3.2: Unify the top-level `darp` entrypoint: default to non-visual terminal traces, use `--visualizer` for the web UI, and write JSON through `--output`
  - [ ] 3.3: Refine cross-step belief/state carryover and hard time-budget control
- [ ] Phase 4: General RDDL problem modeling
  - [ ] 4.1: Stabilize `PlanningProblem`, typed identifiers, and model validation
  - [ ] 4.2: Support multiple state fluents and factored states, replacing the current one-hot compact-state assumption
  - [ ] 4.3: Support stochastic CPFs, non-identity observations, initial belief distributions, and action constraints
  - [ ] 4.4: Validate compiler and simulator semantics against small pyRDDLGym/rddlsim domains
- [ ] Phase 5: Verifiable baseline solvers
  - [ ] 5.1: Refine the explicit-state finite-horizon DP baseline for offline policies and online replanning
  - [ ] 5.2: Add planner registry, unified trace output, and algorithm-selection parameters
  - [ ] 5.3: Add stochastic/tie-break policies and seed-driven reproducibility tests
- [ ] Phase 6: Paper search algorithms
  - [ ] 6.1: Refine the AND-OR history tree in `and_or_tree.py`
  - [ ] 6.2: Implement paper `Expand` and full-tree preprocessing
  - [ ] 6.3: Implement the full ILP baseline
  - [ ] 6.4: Implement HILP partial-ILP search
- [ ] Phase 7: ILP solving layer
  - [ ] 7.1: Implement the ILP model/backend protocol and internal backend
  - [ ] 7.2: Add optional HiGHS backend
  - [ ] 7.3: Add optional Gurobi backend
- [ ] Phase 8: External simulators and PROST compatibility
  - [ ] 8.1: Design the rddlsim/PROST-style online protocol adapter
  - [ ] 8.2: Implement an external simulator client sharing the local action/observation interface
  - [ ] 8.3: Add external simulator integration tests and benchmark runner
- [ ] Phase 9: Durative actions and DARP-RDDL syntax
  - [ ] 9.1: Design YAML/JSON sidecar schema and compiler/runtime interfaces
  - [ ] 9.2: Wire fixed, expected, and Gaussian duration models
  - [ ] 9.3: Connect paper duration/smoothed-belief constraints to HILP
  - [ ] 9.4: Design and implement native DARP-RDDL syntax extensions
- [ ] Phase 10: Output, interfaces, and experiments
  - [ ] 10.1: Refine offline policy JSON and trace output
  - [ ] 10.2: Add benchmarks and paper-style experiments
  - [ ] 10.3: Clean up public APIs and algorithm registry

## Testing

```bash
python -m pytest
```

## Current Limitations

- The compiler currently targets small, discrete RDDL problems with a compact one-hot state fluent.
- The internal simulator runs explicit transition/observation/reward tables and is not a general high-performance RDDL simulator.
- Multiple state fluents, factored states, stochastic observations, full POMDP belief carryover, DARP-RDDL syntax extensions, native durative-action syntax, HILP, and HiGHS/Gurobi backends remain later phases.
