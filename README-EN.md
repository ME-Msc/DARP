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
│       │   └── loader.py           # Selects a concrete parser frontend by name.
│
└── tests/                          # Unit and end-to-end tests.
    ├── test_basic_rddl_parser.py   # Tests the basic RDDL parser and HTML visualizer.
    └── test_rddl_frontends.py      # Tests RDDLFrontend loader, pyrddl, and pyRDDLGym alignment.
```

Planned `core/`, `search/`, `ilp/`, `sim/`, `output/`, and CLI modules will be added in their corresponding phases, with this section updated in the same commits.

## Development Roadmap

- [x] Phase 1: Project scaffold, dependency manifests, test setup, examples
- [x] Phase 2.1: Implement a basic RDDL parser with command-line success output and interactive HTML visualization
- [x] Phase 2.2: Align pyrddl/pyRDDLGym frontends through RDDLFrontend
- [ ] Phase 2.3: Compile ParsedRDDL into PlanningProblem
- [ ] Phase 3: Implement core POMDP/(C)C-POMDP model
- [ ] Phase 4: Implement AND-OR tree in `and_or_tree.py`
- [ ] Phase 5: Implement paper `Expand` and preprocessing
- [ ] Phase 6: Implement internal ILP backend
- [ ] Phase 7: Implement full ILP baseline
- [ ] Phase 8: Implement HILP partial-ILP search
- [ ] Phase 9: Output offline policy JSON
- [ ] Phase 10: Implement online replanning mode
- [ ] Phase 11: Add optional HiGHS backend
- [ ] Phase 12: Add optional Gurobi backend
- [ ] Phase 13: Add benchmarks and paper-style experiments

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

Generate a syntax-highlighted graphical AST HTML page with folding, precise search, and zoom:

```bash
python -m darp.rddl.visualizer \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_ast.html
```

## Current Limitations And Next Steps

- The current basic parser only reads RDDL file, block, assignment, and statement structure for AST/HTML visualization; full RDDL expression semantics remain later Phase 2 work.
- `RDDLFrontend` now returns unified `ParsedRDDL` containers and guarantees that `ast` is DARP's `RDDLASTNode`; complete RDDL-to-PlanningProblem compilation remains Phase 2.3.
- DARP-RDDL extended syntax is not defined yet; sidecar configs are still the recommended way to express duration/risk/HILP metadata for now.
- Planned `core/`, `search/`, `ilp/`, `sim/`, `output/`, and CLI modules will be added in grouped future commits.
