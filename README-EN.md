# DARP

DARP is a research implementation of dual-space search for chance-constrained POMDPs with durative actions. pyRDDLGym parses and grounds RDDL; DARP provides the exact sparse kernel, AND/OR history tree, duration model, full-ILP/HILP encodings, and Gurobi solve path.

The current code is organized around Algorithms 1–3 of the paper and generates reachable numerical state/transition rows only when tree expansion needs them.

## Reproduce locally

From the repository root:

```bash
bash tools/install_linux_deps.sh
tools/run_repro.sh
```

The installer requires Python >=3.12 (the lock was generated and tested with 3.13) and creates `.venv` from `requirements-lock.txt`. The reproduction script captures the environment, runs the full test suite, runs the horizon-3 HILP/RAO*-style representative matrix, and performs a horizon-2 exact-prefix cross-check against full-ILP. Outputs are written to `outputs/darp-review/`.

HILP and full-ILP require a valid local Gurobi license. To install Ubuntu/Debian system packages as well:

```bash
INSTALL_SYSTEM_DEPS=1 bash tools/install_linux_deps.sh
```

## Representative comparison

```bash
.venv/bin/python -m experiments.scripts.run_paper_grid \
  --horizons 3 \
  --risk-budgets 0.1 0.2 0.3 \
  --repetitions 3 \
  --algorithms hilp rao-star-style \
  --expansion-rounds 100 \
  --output outputs/darp-review/paper_grid_results.csv \
  --summary-output outputs/darp-review/paper_grid_summary.csv
```

`experiments/baselines/rao_star.py` is an independent exhaustive finite belief-hypergraph/Pareto implementation of RAO* semantics for this small benchmark. It is not the official RAO* heuristic forward-search implementation. Reported speedups are therefore relative to this explicit comparator, not official RAO*.

The default matrix uses an admissible Manhattan terminal tail and is marked `heuristic/admissible-tail`. `objective_cost=-objective` includes that tail and is not an independently simulated expected arrival cost.

full-ILP remains available as a small exact-prefix oracle. Disable the terminal tail so all three planners optimize the same objective:

```bash
.venv/bin/python -m experiments.scripts.run_paper_grid \
  --horizons 2 \
  --risk-budgets 0.1 \
  --algorithms hilp rao-star-style \
  --include-full-ilp \
  --full-ilp-max-horizon 2 \
  --disable-terminal-tail \
  --output outputs/darp-review/paper_grid_oracle_h2.csv
```

See `experiments/README.md` for field definitions and benchmark assumptions.

## DARP CLI

```bash
.venv/bin/darp -h
```

A supported unit-duration online HILP smoke run:

```bash
.venv/bin/darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl \
  --duration experiments/inputs/durations/fixed_1.yaml \
  --planner hilp \
  --hilp-heuristic reachable-bellman \
  --heuristic-lookahead-depth 2 \
  --expansion-rounds 1 \
  --frontier-width 1 \
  --output /tmp/darp_tiny_grid_hilp.json
```

## Paper-to-code map

```text
RDDL + duration sidecar
  -> adapter.GroundedRDDLView / ExactRDDLKernel
  -> planning.preprocess       # Algorithm 1 root histories
  -> planning.expand           # Algorithm 2 ordinary utility + safe risk flows
  -> planning.ilp_tree         # policy variables and flow/risk coefficients
  -> planning.hilp             # Algorithm 3 incumbent-guided refinement
  -> ilp.GurobiSolver          # deterministic single-thread solve + MIP starts
```

Key implementation properties:

- pyRDDLGym still grounds parameterized fluent/CPF syntax up front; that result is cached once.
- DARP does not pre-enumerate the state-assignment Cartesian product. Reachable states and non-zero transition/reward/observation rows are created lazily and cached.
- Ordinary belief/probability drives unconditional expected utility; safe-conditioned belief/probability drives first-entry execution risk.
- Fixed duration uses an O(1) progress statistic; stochastic duration retains Algorithm 2 backward smoothing.
- HILP refines only frontier variables selected by the current p-ILP incumbent.
- Full-tree construction has explicit node and solve-time limits.

## Current boundaries

- The exact path targets finite grounded Boolean fluent/action models with one active action. Unsupported concurrency, non-Boolean actions, action preconditions, invariants, terminations, and non-unit discount fail closed.
- Lazy work currently applies to reachable numerical state rows, not lifted symbolic grounding. True incremental lifted grounding needs CPF dependency slicing below pyRDDLGym's whole-model grounder.
- Multi-step exact online replanning rejects global chance budgets and non-unit durations because resetting budget/progress each step is unsound. Offline conditional-policy planning remains supported.
- full-ILP is retained for very small exact-prefix checks, not the default performance matrix.

The full Git-history review, paper mapping, and experiment interpretation are in `docs/IMPLEMENTATION_REVIEW.md`.
