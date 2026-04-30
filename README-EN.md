# DARP

Durative Action RDDL Planner is a Python research prototype for parsing RDDL problems, compiling them into finite-horizon POMDP / C-POMDP / CC-POMDP-style planning models, and incrementally implementing the full ILP and HILP search workflow from "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions".

The current code focuses on the standard RDDL input pipeline, DARP's small internal simulator, and an interactive HTML visualizer. PROST-like online execution, AND-OR trees, full ILP, HILP, HiGHS/Gurobi backends, and durative-action interfaces will be added in later phases.

## Feature Status

- Parse standard RDDL domain/instance files into DARP-owned `RDDLASTNode` ASTs.
- Align the `darp`, `pyrddl`, and `pyrddlgym` parser frontends through `RDDLFrontend`.
- Ground the currently supported RDDL CPF/reward expressions into a minimal `PlanningProblem`.
- Execute small explicit transition/observation/reward tables with DARP's internal simulator.
- Start a live HTML UI through the top-level `darp --visualizer` command to inspect source, AST, and optionally an execution state machine where DARP selects actions and the internal simulator advances states.

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

The current top-level command shape is:

```bash
darp --visualizer --domain DOMAIN.rddl --instance INSTANCE.rddl [options]
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--visualizer` | yes | Start the live HTML visualizer. |
| `--domain PATH` | yes | RDDL domain file path. |
| `--instance PATH` | yes | RDDL instance file path. |
| `--with-simulator [darp\|rddlgym\|pyrddlgym]` | no | Enable simulator mode; omitting the value defaults to `darp` and shows DARP's internal state machine. Passing `rddlgym`/`pyrddlgym` loads pyRDDLGym and hides DARP's internal state machine. |
| `--frontend {darp,pyrddl,pyrddlgym}` | no | Frontend used when DARP compiles the problem for its internal simulator. Defaults to `darp`. |
| `--host HOST` | no | Visualizer HTTP host. Defaults to `127.0.0.1`. |
| `--port PORT` | no | Visualizer HTTP port. Defaults to `0`, which chooses a free port. |
| `--no-open` | no | Serve without opening a browser. |
| `-h`, `--help` | no | Show help text. |

## Examples

View source and AST only:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

Start DARP's internal simulator; the right HTML panel advances the environment while DARP selects actions:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --with-simulator
```

Compile with a selected frontend, then use DARP's internal simulator:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --frontend darp \
  --with-simulator darp
```

Load pyRDDLGym while hiding DARP's internal state machine:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --with-simulator rddlgym
```

Serve on a fixed port without opening a browser:

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --with-simulator \
  --host 127.0.0.1 \
  --port 8080 \
  --no-open
```

## Architecture

- `rddl/`: RDDL parser frontends, loading, ASTs, expression grounding, and `PlanningProblem` compilation.
- `core/`: Minimal planning-problem, type, and duration structures for the current phase.
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
│       ├── __main__.py               # Top-level `darp` command entrypoint for `--visualizer` and related options.
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
    └── test_compiler_simulator_interaction.py # Compiler/simulator integration tests.
```

## Roadmap

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
  - [ ] 3.1: Implement a local online solve loop: replan each step, return actions, receive observations
  - [ ] 3.2: Add rddlsim/PROST-style external simulator protocol support
  - [ ] 3.3: Add cross-step belief/state carryover and time-budget control
- [ ] Phase 4: Planning core model
  - [ ] 4.1: Stabilize `PlanningProblem`, typed identifiers, and model validation
  - [ ] 4.2: Refine history, belief, constraints, and policy-tree basics
  - [ ] 4.3: Extend multi-constraint, chance-risk, and continuous/large-state interfaces
- [ ] Phase 5: Search algorithms
  - [ ] 5.1: Refine the AND-OR history tree in `and_or_tree.py`
  - [ ] 5.2: Implement paper `Expand` and full-tree preprocessing
  - [ ] 5.3: Implement the full ILP baseline
  - [ ] 5.4: Implement HILP partial-ILP search
- [ ] Phase 6: ILP solving layer
  - [ ] 6.1: Implement the ILP model/backend protocol and internal backend
  - [ ] 6.2: Add optional HiGHS backend
  - [ ] 6.3: Add optional Gurobi backend
- [ ] Phase 7: Durative action sidecar
  - [ ] 7.1: Design YAML/JSON sidecar schema and compiler/runtime interfaces
  - [ ] 7.2: Wire fixed, expected, and Gaussian duration models
  - [ ] 7.3: Connect paper duration/smoothed-belief constraints to HILP
- [ ] Phase 8: DARP-RDDL new syntax
  - [ ] 8.1: Design DARP-RDDL syntax extensions
  - [ ] 8.2: Choose and implement parser inheritance, fork, or owned grammar
  - [ ] 8.3: Migrate sidecar capabilities into optional native syntax
- [ ] Phase 9: Output, interfaces, and experiments
  - [ ] 9.1: Refine offline policy JSON and trace output
  - [ ] 9.2: Add benchmarks and paper-style experiments
  - [ ] 9.3: Clean up public APIs and algorithm registry

## Testing

```bash
python -m pytest
```

## Current Limitations

- The compiler currently targets small, discrete RDDL problems with a compact one-hot state fluent.
- The internal simulator runs explicit transition/observation/reward tables and is not a general high-performance RDDL simulator.
- DARP-RDDL syntax extensions, native durative-action syntax, HILP, and HiGHS/Gurobi backends remain later phases.
