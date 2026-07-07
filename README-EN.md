# DARP

DARP (Durative Action RDDL Planner) is a Python research planner for RDDL. It uses pyRDDLGym for standard RDDL parsing, grounding, and simulation; DARP owns the AND-OR history tree, duration sidecars, chance-constrained risk modeling, full-ILP/HILP encodings, and Gurobi solving path.

The comparison baseline in this repository is RAO*. RAO* is not part of the DARP planner API; it lives under `experiments/baselines/rao_star/` and is invoked by experiment scripts.

## Installation

```bash
bash tools/install_linux_deps.sh
source .venv/bin/activate
```

On a fresh Ubuntu/Debian machine:

```bash
INSTALL_SYSTEM_DEPS=1 bash tools/install_linux_deps.sh
```

Gurobi note: the script installs `gurobipy`, but real full-ILP/HILP runs still require a valid local Gurobi license.

## DARP CLI

Show help:

```bash
darp -h
```

Run the default online trace:

```bash
darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl
```

Run HILP:

```bash
darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl \
  --duration experiments/inputs/durations/tiny_grid.yaml \
  --planner hilp \
  --hilp-heuristic reachable-bellman \
  --heuristic-lookahead-depth 4 \
  --expansion-rounds 4 \
  --frontier-width 1 \
  --output /tmp/darp_tiny_grid_hilp.json
```

Run full-ILP:

```bash
darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl \
  --duration experiments/inputs/durations/tiny_grid.yaml \
  --planner full-ilp
```

HILP frontier heuristics:

- `reachable-bellman`: finite-horizon fully observable Bellman backup over states reachable from the current frontier action.
- `one-step-greedy`: one-step expected reward for the current action; faster but greedier.

## RAO* Comparison Experiments

Experiment entrypoints live under `experiments/scripts/`. Once Science Agent / PSR adapters are implemented, `--domain`, `--instance`, and `--duration` should point to files under `experiments/inputs/rao_star/`.

```bash
python experiments/scripts/run_rao_star_suite.py \
  --name science_agent_small_fixed \
  --domain experiments/inputs/rao_star/science_agent/science_agent_domain.rddl \
  --instance experiments/inputs/rao_star/science_agent/science_agent_small.rddl \
  --duration experiments/inputs/durations/fixed_1.yaml \
  --seeds 0 \
  --planners rao-star,hilp \
  --hilp-heuristic reachable-bellman \
  --heuristic-lookahead-depth 2 \
  --frontier-width 1
```

Outputs are written under:

```text
experiments/outputs/<experiment-name>/
```

Generate a replay page:

```bash
python experiments/scripts/visualize_replay.py \
  experiments/outputs/<experiment-name>/runs.csv \
  --output experiments/outputs/<experiment-name>/replay.html
```

LaTeX table templates live under:

```text
experiments/reports/latex/
```

## Experiment Workspace

```text
experiments/
├── inputs/
│   ├── rddl/            # tiny_grid, factored_door, and other small RDDL sanity checks.
│   ├── durations/       # fixed / Gaussian duration sidecar examples.
│   ├── rao_star/        # RAO* paper Science Agent / PSR reproduction inputs.
│   └── benchmarks/      # RDDL/IPPC benchmark corpus for later extensions.
├── baselines/
│   └── rao_star/        # External RAO* comparison baseline.
├── scripts/             # Experiment runners, baseline wrappers, and replay visualization.
├── outputs/             # Generated JSON/CSV/log/replay artifacts, ignored by default.
└── reports/
    └── latex/           # Versioned LaTeX table templates and preview entrypoints.
```

## Duration Sidecars

A duration sidecar describes action durations and optional risk; it does not duplicate the RDDL instance `horizon`.

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
risk:
  budget: 0.25
  next_state_fluents:
    at___c22: 1
```

## Architecture

```text
darp CLI
  -> adapter.RDDLLoader
  -> pyRDDLGym env/model/native AST
  -> adapter.GroundedRDDLView
  -> adapter.ExactRDDLKernel
  -> model.ANDORNode / History / DurationModel
  -> planning.preprocess / planning.expand
  -> planning.FullILPPlanner or planning.HILPPlanner
  -> ilp.GurobiSolver
  -> planning.OnlineSession
```

Repository layout:

```text
DARP/
├── src/darp/            # DARP core planner, adapter, model, ILP, and visualization code.
├── experiments/         # Experiment inputs, external baselines, scripts, outputs, and reports.
├── docs/                # Paper symbols, benchmark strategy, and development notes.
├── tools/               # Installation and maintenance scripts.
└── tests/               # Unit and lightweight integration tests.
```

## Roadmap

- [x] Use pyRDDLGym as the standard RDDL parser/grounder/simulator.
- [x] Build DARP AND-OR history tree, duration sidecars, and exact finite kernels.
- [x] Implement Gurobi-backed full-tree ILP and HILP partial-tree solving paths.
- [x] Support fixed duration, Gaussian percentile duration, and sidecar risk budgets.
- [x] Move RAO* out of the DARP planner API into an external baseline wrapper.
- [x] Unify experiment inputs, baselines, scripts, outputs, and reports.
- [ ] Reproduce the RAO* paper Science Agent / PSR benchmark adapters.
- [ ] Improve HILP benchmark-scale pruning and large action-space handling.
- [ ] Support concurrent action combinations and non-boolean actions.
- [ ] If needed, extend native durative-action RDDL syntax through the pyRDDLGym parser.

## Tests

```bash
python -m pytest
```

Basic tests do not require a local Gurobi license; real full-ILP/HILP experiments do.

## Current Limitations

- The exact kernel currently targets finite, grounded, boolean fluent/action RDDL problems.
- `experiments/baselines/rao_star/` is currently a small exact deterministic-policy comparator wrapper; formal experiments should move to the RAO* paper's Science Agent / PSR scenarios.
- full-ILP expands the full remaining horizon, so its size grows exponentially with action/observation histories.
- HILP is partial-tree refinement and is not a global optimality certificate unless it expands to the full tree or gets strict bounds/certificates.
