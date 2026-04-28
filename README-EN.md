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

`search/` and `ilp/` have different responsibilities:

- `search/` is the planning/search layer. It answers "how should the policy tree be explored?" HILP, the full-ILP wrapper, and online replanning live here.
- `ilp/` is the mathematical programming layer. It answers "how should this ILP/p-ILP model be solved?" internal, HiGHS, and Gurobi are backends here.

Therefore, HiGHS and Gurobi are not alternatives to HILP. They are lower-level solvers that HILP can call for each p-ILP.

## Repository Map

<!-- README.md: Chinese main documentation for goals, paper alignment, architecture, roadmap, and run commands. -->
<!-- README-EN.md: English mirror documentation for collaborators. -->
<!-- LICENSE: Apache-2.0 license text for the project. -->
<!-- .gitignore: Ignores Python caches, virtual environments, build artifacts, and local config. -->
<!-- .codex: Local Codex workspace configuration placeholder. -->
<!-- pyproject.toml: Python package metadata, dependencies, CLI entrypoint, and optional backend extras. -->
<!-- requirements.txt: Records runtime core dependencies; the current core only uses the Python standard library. -->
<!-- requirements-dev.txt: Records development and test dependencies such as pytest. -->
<!-- examples/: Minimal RDDL examples and duration configs used by demos and tests. -->
<!-- examples/rddl/: RDDL domain and instance files. -->
<!-- examples/rddl/tiny_grid_domain.rddl: Placeholder tiny-grid RDDL domain for demos. -->
<!-- examples/rddl/tiny_grid_instance.rddl: Placeholder tiny-grid RDDL instance for demos. -->
<!-- examples/durations/: Sidecar configs for durative-action settings. -->
<!-- examples/durations/tiny_grid.yaml: Sidecar duration config for tiny-grid actions. -->
<!-- src/darp/: Main DARP Python package. -->
<!-- src/darp/__init__.py: Defines package version and top-level exports. -->
<!-- src/darp/cli.py: Command-line entrypoint for solve and evaluate modes. -->
<!-- src/darp/rddl/: RDDL loading and compilation code. -->
<!-- src/darp/rddl/__init__.py: Marks the RDDL subpackage and keeps phase TODOs. -->
<!-- src/darp/rddl/loader.py: Loads RDDL files or environments through pyRDDLGym. -->
<!-- src/darp/rddl/compiler.py: Compiles a pyRDDLGym environment into DARP's PlanningProblem. -->
<!-- src/darp/rddl/durations.py: Reads duration sidecar configs. -->
<!-- src/darp/core/: Solver-independent planning data structures. -->
<!-- src/darp/core/__init__.py: Marks the core subpackage and reserves stable API exports. -->
<!-- src/darp/core/types.py: Central state, action, observation, and distribution type aliases. -->
<!-- src/darp/core/problem.py: Defines the finite-horizon POMDP/(C)C-POMDP problem interface. -->
<!-- src/darp/core/history.py: Defines observation histories and action histories matching the paper. -->
<!-- src/darp/core/belief.py: Implements belief updates, safe beliefs, and risk probability. -->
<!-- src/darp/core/duration.py: Implements fixed, expected/state-dependent, and Gaussian percentile duration models. -->
<!-- src/darp/core/constraints.py: Defines expected-cost and chance-risk constraints. -->
<!-- src/darp/core/policy.py: Represents solve results, action sequences, policy trees, and JSON export data. -->
<!-- src/darp/search/: Planning algorithm layer for searching policy space. -->
<!-- src/darp/search/__init__.py: Marks the search subpackage and reserves algorithm registry metadata. -->
<!-- src/darp/search/base.py: Defines the common Planner interface. -->
<!-- src/darp/search/and_or_tree.py: Defines the shared AND-OR history tree structure. -->
<!-- src/darp/search/expand.py: Implements the paper's Expand step and computes ILP constants. -->
<!-- src/darp/search/preprocess.py: Expands the complete finite tree for the full ILP baseline. -->
<!-- src/darp/search/full_ilp.py: Builds and solves the complete ILP without HILP frontier pruning. -->
<!-- src/darp/search/hilp.py: Implements the HILP partial-ILP heuristic search from Algorithm 3. -->
<!-- src/darp/search/heuristics.py: Provides frontier utility/risk heuristics. -->
<!-- src/darp/search/online_replanner.py: Wraps a PROST-style online replanning loop. -->
<!-- src/darp/ilp/: ILP/p-ILP model representation and backend solvers. -->
<!-- src/darp/ilp/__init__.py: Marks the ilp subpackage and keeps backend roadmap TODOs. -->
<!-- src/darp/ilp/model.py: Defines solver-neutral variables, objectives, and linear constraints. -->
<!-- src/darp/ilp/backend.py: Defines the backend protocol shared by internal, HiGHS, and Gurobi. -->
<!-- src/darp/ilp/internal.py: Implements the built-in small-scale binary ILP solver for independent runs. -->
<!-- src/darp/ilp/highs.py: Wraps the optional HiGHS backend. -->
<!-- src/darp/ilp/gurobi.py: Wraps the optional Gurobi backend. -->
<!-- src/darp/ilp/factory.py: Selects an ILP backend from CLI/config. -->
<!-- src/darp/sim/: Local and external simulation adapters. -->
<!-- src/darp/sim/__init__.py: Marks the sim subpackage and reserves simulator registration. -->
<!-- src/darp/sim/local.py: Local sampling simulator based on PlanningProblem. -->
<!-- src/darp/sim/rddlsim_client.py: Future rddlsim-style external TCP simulator client. -->
<!-- src/darp/sim/protocol.py: Defines the simulator interaction protocol. -->
<!-- src/darp/output/: Utilities for solve results and trace output. -->
<!-- src/darp/output/__init__.py: Marks the output subpackage and reserves benchmark writers. -->
<!-- src/darp/output/json_policy.py: Writes policy/action-sequence JSON. -->
<!-- src/darp/output/trace.py: Records solve/evaluation traces. -->
<!-- tests/: Unit and end-to-end tests. -->
<!-- tests/test_history.py: Tests observation/action history concatenation, parents, and depth. -->
<!-- tests/test_belief.py: Tests belief prediction, observation update, and risk probability. -->
<!-- tests/test_duration.py: Tests fixed and Gaussian duration tau calculations. -->
<!-- tests/test_internal_backend.py: Tests the built-in binary ILP backend on a tiny model. -->
<!-- tests/test_and_or_tree.py: Tests AND-OR tree expansion from the root. -->
<!-- tests/test_preprocess_expand.py: Tests full-tree preprocessing and Expand integration. -->
<!-- tests/test_hilp_tiny_grid.py: Tests HILP and full ILP agreement on tiny grid. -->
<!-- tests/test_cli.py: Tests CLI solve output as parseable JSON. -->

## Development Roadmap

- [x] Phase 1: Project scaffold, CLI, test setup, examples
- [ ] Phase 2: Load RDDL through pyRDDLGym
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

## Current Limitations And Next Steps

- The default tiny grid uses a built-in Python problem model; the `pyRDDLGym` loader is reserved, but complete RDDL-to-PlanningProblem compilation remains Phase 2.
- The internal ILP backend uses exhaustive binary search and is intended for small examples and tests, not performance.
- HiGHS/Gurobi backend files are reserved and include dependency checks; full performance experiments are planned later.
- Gaussian percentile duration has a runnable approximation; the paper's full smoothed-belief details will be refined later.
- The current implementation supports one expected-cost or chance-risk constraint; multi-constraint support is planned for the benchmark stage.
