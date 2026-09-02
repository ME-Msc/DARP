# Science Agent: DARP-HILP vs RAO*

| Δ | Algorithm | Objective | Risk | Time (s) | Expanded nodes† | Evaluated states‡ | Iterations |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.002 | DARP-HILP | 19.980000 | 0.001999 | 1.121 ± 0.031 | 30.0 | 816.0 | 18.0 |
| 0.002 | RAO* | 19.980799 | 0.001999 | 0.622 ± 0.026 | 57.0 | 544.0 | 17.0 |
| 0.01 | DARP-HILP | 29.454524 | 0.004990 | 2.796 ± 0.069 | 214.0 | 1040.0 | 64.0 |
| 0.01 | RAO* | 29.455194 | 0.004990 | 0.445 ± 0.034 | 53.0 | 409.0 | 11.0 |
| 0.05 | DARP-HILP | 29.454524 | 0.004990 | 2.105 ± 0.025 | 152.0 | 992.0 | 50.0 |
| 0.05 | RAO* | 29.455194 | 0.004990 | 0.434 ± 0.019 | 53.0 | 409.0 | 11.0 |

Trials per configuration: 3.

† DARP expands action histories; RAO* expands belief hypergraph nodes. These are native search-effort counters, not identical units.

‡ DARP reports distinct grounded states lazily compiled during search. RAO* reports the original implementation's cumulative evaluated belief particles; the two counters describe implementation effort but are not identical quantities.

## Scope and provenance

- Semantic scope: `source_non_scheduling`.
- Source model: `tFakePlannerRockSampleModel(perform_scheduling=False)` with the inert 1000-second constraint from its checked-in test.
- DARP uses the equivalent RDDL model, fixed unit duration, and horizon 5; no-revisit actions guarantee relay or crash within that bound.
- RAO* source: [https://github.com/ME-Msc/rao-star.git](https://github.com/ME-Msc/rao-star.git) at `f51bfdc1ff8fcb2504dcb38c3b36d719506501fb`.
- Paper: [RAO*: An Algorithm for Chance-Constrained POMDPs](https://ojs.aaai.org/index.php/AAAI/article/view/10423).
- Domain ancestor: [Benazera et al. (2005)](https://aiweb.cs.washington.edu/ai/planning/papers/mausam-ijcai05.pdf) describes a continuous-resource HAO* rover, not the exact RAO* test parameters.
- This is not a reproduction of the paper's PARIS scheduling/time-window Table 1; `perform_scheduling=False` is required for exact shared semantics with current DARP.
- Small objective differences are within DARP's configured Gurobi MIPGap; the complete reachable T/O/reward/risk/heuristic/terminal model is checked before timing.
