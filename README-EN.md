# DARP

Durative Action RDDL Planner is a Python research prototype for compiling RDDL-style planning problems into the finite-horizon (C)C-POMDP with durative actions used in "Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions", then solving them with full ILP or HILP partial-ILP search.

## Project Goal

DARP v1 focuses on a runnable, readable, and extensible research path:

- Represent POMDP / C-POMDP / CC-POMDP problems, beliefs, histories, durations, and policy trees in Python.
- Use an AND-OR history tree for the paper's observation histories `O~` and action histories `A~`.
- Implement the paper's `Expand`, full ILP baseline, and HILP partial-ILP search.
- Run independently with a built-in small-scale ILP backend, while reserving optional HiGHS and Gurobi support.
- Support offline policy-tree output and leave a clean online replanning interface.

## Paper Alignment

- `core/history.py` corresponds to the paper's action-observation history `q`.
- `search/and_or_tree.py` corresponds to the AND-OR tree in Figures 1 and 2.
- `search/expand.py` corresponds to Algorithm 2 `Expand`, computing `u_q`, `r_q`, `rho(q)`, beliefs, and `tau(q)`.
- `search/preprocess.py` performs complete preprocessing for the full ILP baseline.
- `search/full_ilp.py` builds and solves the complete ILP.
- `search/hilp.py` corresponds to Algorithm 3 `HILP`, incrementally expanding frontiers and repeatedly solving p-ILPs.
- `ilp/` only represents and solves ILP/p-ILP subproblems.

## Architecture

`rddl/`, `search/`, and `ilp/` have different responsibilities:

- `rddl/` is the parsing and compilation entrypoint. It answers "how should standard RDDL or future DARP-RDDL extended syntax become a common intermediate representation?"
- `search/` is the planning/search layer. It answers "how should the policy tree be explored?" HILP, the full-ILP wrapper, and online replanning live here.
- `ilp/` is the mathematical programming layer. It answers "how should this ILP/p-ILP model be solved?" internal, HiGHS, and Gurobi are backends here.

Therefore, HiGHS and Gurobi are not alternatives to HILP. They are lower-level solvers that HILP can call for each p-ILP.

### RDDLFrontend Protocol

`RDDLFrontend` is DARP's unified parser protocol, not a concrete parser. Every parser frontend implements:

- `name`: the frontend name, such as `pyrddlgym`, `pyrddl`, or `darp`.
- `supports_extended_syntax`: whether the frontend supports DARP-specific syntax.
- `parse(domain, instance) -> ParsedRDDL`: parse a domain/instance pair into a common container.

`ParsedRDDL` is the only input shape the compiler should see. Its `ast` is always DARP's own `RDDLASTNode`, third-party parser ASTs live in `native_ast`, pyRDDLGym executable artifacts live in `model/env`, and extra details live in metadata. This lets DARP reuse `pyrddl`, reuse or subclass pyRDDLGym parser internals, or eventually replace both with a DARP-owned parser without forcing changes in `compiler.py`, `core/`, `search/`, or `ilp/`.

The current frontend slots are:

- `pyrddlgym`: default standard-RDDL frontend for pyRDDLGym parser/simulator reuse.
- `pyrddl`: direct `pyrddl.parser.RDDLParser` frontend, useful as a fork or DARP-owned parser starting point.
- `darp`: DARP-owned basic parser frontend. It currently parses RDDL file/block/statement structure and reserves the future DARP-RDDL extension entrypoint.

The basic parser can be verified from the command line:

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

To view the AST graphically, generate a standalone syntax-highlighted HTML visualizer with an English UI, node folding, depth expansion, precise search, and zoom:

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --html-output tiny_grid_ast.html
```

You can also use the standalone visualizer module:

```bash
python -m darp.rddl.visualizer \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_ast.html
```

## Repository Map

```text
DARP/
├── README.md                       # Chinese main documentation for goals, paper alignment, architecture, roadmap, and run commands.
├── README-EN.md                    # English mirror documentation for collaborators.
├── LICENSE                         # Apache-2.0 license text for the project.
├── .gitignore                      # Ignores Python caches, virtual environments, build artifacts, and local config.
├── pyproject.toml                  # Python package metadata, dependencies, and optional backend extras.
├── requirements.txt                # Records runtime core dependencies; the current core only uses the Python standard library.
├── requirements-dev.txt            # Records development and test dependencies such as pytest.
│
├── examples/                       # Minimal RDDL examples used by demos and tests.
│   ├── rddl/                       # RDDL domain and instance files.
│   │   ├── tiny_grid_domain.rddl   # Placeholder tiny-grid RDDL domain for demos.
│   │   └── tiny_grid_instance.rddl # Placeholder tiny-grid RDDL instance for demos.
│
├── src/
│   └── darp/                       # Main DARP Python package.
│       ├── __init__.py             # Defines package version and top-level exports.
│       │
│       ├── rddl/                   # RDDL parser frontends, loading, and compilation code.
│       │   ├── __init__.py         # Marks the RDDL subpackage and keeps parser/compiler TODOs.
│       │   ├── ast.py              # Defines the basic RDDL AST node structure.
│       │   ├── basic_parser.py     # Implements the dependency-free structural RDDL parser and command-line entrypoint.
│       │   ├── visualizer.py       # Renders the basic AST as a standalone syntax-highlighted graphical HTML tree with folding, precise search, and zoom.
│       │   ├── frontend.py         # Defines the RDDLFrontend protocol and ParsedRDDL container.
│       │   ├── pyrddlgym_frontend.py # Reuses pyRDDLGym while returning DARP AST and environment objects.
│       │   ├── pyrddl_frontend.py  # Reuses pyrddl.parser.RDDLParser while keeping DARP AST and native AST.
│       │   ├── extended.py         # Uses the DARP-owned parser and reserves future DARP-RDDL extended syntax.
│       │   ├── compiler.py         # Structurally compiles ParsedRDDL's DARP AST into a minimal PlanningProblem.
│       │   └── loader.py           # Selects a concrete parser frontend by name.
│       │
│       ├── core/                   # Minimal planning problem data structures needed by Phase 2.3.
│       │   ├── __init__.py         # Marks the core subpackage and keeps public API TODOs.
│       │   ├── types.py            # Defines shared aliases for states, actions, observations, and transitions.
│       │   ├── duration.py         # Defines the duration interface and fixed-duration model needed by PlanningProblem.
│       │   └── problem.py          # Defines PlanningProblem and the built-in tiny-grid problem.
│
└── tests/                          # Unit and end-to-end tests.
    ├── test_basic_rddl_parser.py   # Tests the basic RDDL parser and HTML visualizer.
    ├── test_rddl_frontends.py      # Tests RDDLFrontend loader, pyrddl, and pyRDDLGym alignment.
    └── test_rddl_compiler.py       # Tests structural compilation from ParsedRDDL to PlanningProblem.
```

The current `core/` commit only contains the minimal model needed by the Phase 2.3 compiler; the fuller `core/`, `search/`, `ilp/`, `sim/`, `output/`, and CLI modules will be added in their corresponding phases and reflected here in the same commits.

## Development Roadmap

- [x] Phase 1: Project foundation
  - [x] 1.1: Project plan, README/README-EN, and file structure notes
  - [x] 1.2: Python packaging, requirements, and `.venv` workflow
  - [x] 1.3: Minimal RDDL examples and pytest entrypoint
- [ ] Phase 2: RDDL input pipeline
  - [x] 2.1: Basic RDDL parser and interactive HTML AST visualizer
  - [x] 2.2: Align `darp`, `pyrddl`, and `pyrddlgym` through `RDDLFrontend`
  - [x] 2.3: Structurally compile `ParsedRDDL` into a minimal `PlanningProblem`
  - [ ] 2.4: Complete standard RDDL CPF/reward semantic grounding without adding new syntax
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

## Install And Run

Install in development mode:

```bash
virtualenv .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install -r requirements-dev.txt
```

If your machine already has `python3-venv`, `python -m venv .venv` also works. On this machine, `.venv` was created with `virtualenv --clear .venv` because the system Python is missing `ensurepip`.

Install runtime dependencies only:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional external dependencies remain recorded as `pyproject.toml` extras:

```bash
python -m pip install -e ".[dev,rddl,pyrddl]"
python -m pip install -e ".[rddl]"
python -m pip install -e ".[pyrddl]"
python -m pip install -e ".[highs]"
python -m pip install -e ".[gurobi]"
```

Run tests:

```bash
python -m pytest
```

Verify the basic RDDL parser:

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

Inspect all three frontends through the shared `RDDLFrontend` loader:

```bash
python -m darp.rddl.loader \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --frontend darp

python -m darp.rddl.loader \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --frontend pyrddl

python -m darp.rddl.loader \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --frontend pyrddlgym
```

Compile RDDL into a DARP `PlanningProblem` summary:

```bash
python -m darp.rddl.compiler \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --frontend darp
```

Generate a syntax-highlighted graphical AST HTML page with folding, precise search, and zoom:

```bash
python -m darp.rddl.visualizer \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_ast.html
```

## Current Limitations And Next Steps

- The current basic parser only reads RDDL file, block, assignment, and statement structure for AST/HTML visualization; full RDDL expression semantics remain later Phase 2 work.
- `RDDLCompiler` now structurally compiles small discrete `ParsedRDDL` inputs into a minimal `PlanningProblem`; full CPF/reward semantic grounding remains Phase 2.4.
- DARP-RDDL extended syntax is still undefined and intentionally remains a later phase so it does not block the standard-RDDL compilation path.
- Planned `search/`, `ilp/`, `sim/`, `output/`, and CLI modules will be added in grouped future commits.
