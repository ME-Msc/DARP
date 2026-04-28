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

`ParsedRDDL` is the only input shape the compiler should see. It can carry an `ast`, `model`, `env`, and metadata. This lets DARP reuse `pyrddl`, reuse or subclass pyRDDLGym parser internals, or eventually replace both with a DARP-owned parser without forcing changes in `compiler.py`, `core/`, `search/`, or `ilp/`.

The current frontend slots are:

- `pyrddlgym`: default standard-RDDL frontend for pyRDDLGym parser/simulator reuse.
- `pyrddl`: direct `pyrddl.parser.RDDLParser` frontend, useful as a fork or DARP-owned parser starting point.
- `darp`: DARP-owned basic parser frontend. It currently parses RDDL file/block/statement structure and reserves the future DARP-RDDL extension entrypoint.

The basic parser can be verified from the command line and can print the AST as Graphviz DOT:

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --dot
```

To save the DOT file:

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --dot-output tiny_grid_ast.dot
```

## Repository Map

```text
DARP/
├── README.md                       # Chinese main documentation for goals, paper alignment, architecture, roadmap, and run commands.
├── README-EN.md                    # English mirror documentation for collaborators.
├── LICENSE                         # Apache-2.0 license text for the project.
├── .gitignore                      # Ignores Python caches, virtual environments, build artifacts, and local config.
├── .codex                          # Local Codex workspace configuration placeholder.
├── pyproject.toml                  # Python package metadata, dependencies, CLI entrypoint, and optional backend extras.
├── requirements.txt                # Records runtime core dependencies; the current core only uses the Python standard library.
├── requirements-dev.txt            # Records development and test dependencies such as pytest.
│
├── examples/                       # Minimal RDDL examples and duration configs used by demos and tests.
│   ├── rddl/                       # RDDL domain and instance files.
│   │   ├── tiny_grid_domain.rddl   # Placeholder tiny-grid RDDL domain for demos.
│   │   └── tiny_grid_instance.rddl # Placeholder tiny-grid RDDL instance for demos.
│   └── durations/                  # Sidecar configs for durative-action settings.
│       └── tiny_grid.yaml          # Sidecar duration config for tiny-grid actions.
│
├── src/
│   └── darp/                       # Main DARP Python package.
│       ├── __init__.py             # Defines package version and top-level exports.
│       ├── cli.py                  # Command-line entrypoint for solve and evaluate modes.
│       │
│       ├── rddl/                   # RDDL parser frontends, loading, and compilation code.
│       │   ├── __init__.py         # Marks the RDDL subpackage and keeps parser/compiler TODOs.
│       │   ├── ast.py              # Defines basic RDDL AST nodes and Graphviz DOT export.
│       │   ├── basic_parser.py     # Implements the dependency-free structural RDDL parser and command-line entrypoint.
│       │   ├── frontend.py         # Defines the RDDLFrontend protocol and ParsedRDDL container.
│       │   ├── pyrddlgym_frontend.py # Reuses pyRDDLGym to parse standard RDDL and return environment objects.
│       │   ├── pyrddl_frontend.py  # Reuses pyrddl.parser.RDDLParser to produce direct ASTs.
│       │   ├── extended.py         # Uses the DARP-owned parser and reserves future DARP-RDDL extended syntax.
│       │   ├── loader.py           # Selects a parser frontend from --rddl-frontend.
│       │   ├── compiler.py         # Compiles ParsedRDDL into DARP's PlanningProblem.
│       │   └── durations.py        # Reads duration sidecar configs.
│       │
│       ├── core/                   # Solver-independent planning data structures.
│       │   ├── __init__.py         # Marks the core subpackage and reserves stable API exports.
│       │   ├── types.py            # Central state, action, observation, and distribution type aliases.
│       │   ├── problem.py          # Defines the finite-horizon POMDP/(C)C-POMDP problem interface.
│       │   ├── history.py          # Defines observation histories and action histories matching the paper.
│       │   ├── belief.py           # Implements belief updates, safe beliefs, and risk probability.
│       │   ├── duration.py         # Implements fixed, expected/state-dependent, and Gaussian percentile duration models.
│       │   ├── constraints.py      # Defines expected-cost and chance-risk constraints.
│       │   └── policy.py           # Represents solve results, action sequences, policy trees, and JSON export data.
│       │
│       ├── search/                 # Planning algorithm layer for searching policy space.
│       │   ├── __init__.py         # Marks the search subpackage and reserves algorithm registry metadata.
│       │   ├── base.py             # Defines the common Planner interface.
│       │   ├── and_or_tree.py      # Defines the shared AND-OR history tree structure.
│       │   ├── expand.py           # Implements the paper's Expand step and computes ILP constants.
│       │   ├── preprocess.py       # Expands the complete finite tree for the full ILP baseline.
│       │   ├── full_ilp.py         # Builds and solves the complete ILP without HILP frontier pruning.
│       │   ├── hilp.py             # Implements the HILP partial-ILP heuristic search from Algorithm 3.
│       │   ├── heuristics.py       # Provides frontier utility/risk heuristics.
│       │   └── online_replanner.py # Wraps a PROST-style online replanning loop.
│       │
│       ├── ilp/                    # ILP/p-ILP model representation and backend solvers.
│       │   ├── __init__.py         # Marks the ilp subpackage and keeps backend roadmap TODOs.
│       │   ├── model.py            # Defines solver-neutral variables, objectives, and linear constraints.
│       │   ├── backend.py          # Defines the backend protocol shared by internal, HiGHS, and Gurobi.
│       │   ├── internal.py         # Implements the built-in small-scale binary ILP solver for independent runs.
│       │   ├── highs.py            # Wraps the optional HiGHS backend.
│       │   ├── gurobi.py           # Wraps the optional Gurobi backend.
│       │   └── factory.py          # Selects an ILP backend from CLI/config.
│       │
│       ├── sim/                    # Local and external simulation adapters.
│       │   ├── __init__.py         # Marks the sim subpackage and reserves simulator registration.
│       │   ├── local.py            # Local sampling simulator based on PlanningProblem.
│       │   ├── rddlsim_client.py   # Future rddlsim-style external TCP simulator client.
│       │   └── protocol.py         # Defines the simulator interaction protocol.
│       │
│       └── output/                 # Utilities for solve results and trace output.
│           ├── __init__.py         # Marks the output subpackage and reserves benchmark writers.
│           ├── json_policy.py      # Writes policy/action-sequence JSON.
│           └── trace.py            # Records solve/evaluation traces.
│
└── tests/                          # Unit and end-to-end tests.
    ├── test_basic_rddl_parser.py   # Tests the basic RDDL parser AST and DOT output.
    ├── test_history.py             # Tests observation/action history concatenation, parents, and depth.
    ├── test_belief.py              # Tests belief prediction, observation update, and risk probability.
    ├── test_duration.py            # Tests fixed and Gaussian duration tau calculations.
    ├── test_internal_backend.py    # Tests the built-in binary ILP backend on a tiny model.
    ├── test_and_or_tree.py         # Tests AND-OR tree expansion from the root.
    ├── test_preprocess_expand.py   # Tests full-tree preprocessing and Expand integration.
    ├── test_hilp_tiny_grid.py      # Tests HILP and full ILP agreement on tiny grid.
    └── test_cli.py                 # Tests CLI solve output as parseable JSON.
```

## Development Roadmap

- [x] Phase 1: Project scaffold, CLI, test setup, examples
- [x] Phase 2.1: Implement a basic RDDL parser with command-line success output and AST DOT export
- [ ] Phase 2.2: Align pyrddl/pyRDDLGym frontends through RDDLFrontend
- [ ] Phase 2.3: Compile ParsedRDDL into PlanningProblem
- [x] Phase 3: Implement core POMDP/(C)C-POMDP model
- [x] Phase 4: Implement AND-OR tree in `and_or_tree.py`
- [x] Phase 5: Implement paper `Expand` and preprocessing
- [x] Phase 6: Implement internal ILP backend
- [x] Phase 7: Implement full ILP baseline
- [x] Phase 8: Implement HILP partial-ILP search
- [x] Phase 9: Output offline policy JSON
- [x] Phase 10: Implement online replanning mode
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
python -m pip install -e ".[rddl]"
python -m pip install -e ".[pyrddl]"
python -m pip install -e ".[highs]"
python -m pip install -e ".[gurobi]"
```

Run HILP on the built-in tiny grid:

```bash
darp solve --algorithm hilp --solver internal
```

Run the full ILP baseline:

```bash
darp solve --algorithm full-ilp --solver internal
```

Write a JSON policy:

```bash
darp solve --algorithm hilp --solver internal --output policy.json
```

Run tests:

```bash
python -m pytest
```

Verify the basic RDDL parser and print the DOT AST in the terminal:

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --dot
```

## Current Limitations And Next Steps

- The current basic parser only reads RDDL file, block, assignment, and statement structure for AST/DOT visualization; full RDDL expression semantics remain later Phase 2 work.
- The default tiny grid uses a built-in Python problem model; the `RDDLFrontend` parsing layer is reserved, but complete RDDL-to-PlanningProblem compilation remains Phase 2.
- DARP-RDDL extended syntax is not defined yet; sidecar configs are still the recommended way to express duration/risk/HILP metadata for now.
- The internal ILP backend uses exhaustive binary search and is intended for small examples and tests, not performance.
- HiGHS/Gurobi backend files are reserved and include dependency checks; full performance experiments are planned later.
- Gaussian percentile duration has a runnable approximation; the paper's full smoothed-belief details will be refined later.
- The current implementation supports one expected-cost or chance-risk constraint; multi-constraint support is planned for the benchmark stage.
