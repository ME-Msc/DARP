# Table 2: Simulation results with heuristics

Arithmetic means over 1 completed trials; time is planner wall-clock seconds.
Objective values are solver-native.

| Problem | h | Δ | DARP-HILP Obj. | DARP-HILP Time (s) | DARP-HILP n | DARP-HILP Iter. | RAO* Obj. | RAO* Time (s) | RAO* n | RAO* Iter. |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5×5 | 3 | 0.1 | 8.93 | 0.07 | 268 | 12 | 9.75 | 0.08 | 337 | 40 |

## Supplementary risk diagnostics

This diagnostic table is not part of the original Table 2.

| Problem | h | Δ | DARP-HILP Risk | RAO* Risk |
|:--|--:|--:|--:|--:|
| 5×5 | 3 | 0.1 | 0.089766 | 0.089268 |

## DARP policy execution

Execution time is one `agent.evaluate(env, episodes=1)` wall-clock measurement; it is not modeled action duration.

| Problem | h | Δ | DARP-HILP execution time (s) |
|:--|--:|--:|--:|
| 5×5 | 3 | 0.1 | 0.003214 |
