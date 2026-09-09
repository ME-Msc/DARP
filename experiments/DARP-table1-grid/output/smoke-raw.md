# DARP Table 1 grid experiment

Values are means over successful trials; `Exp.% = HILP Exp.n / Full-ILP Act.n`.
E/S use source-or-intended mud contact; S uses `Normal(mean, variance=0.1)` and `varsigma=0.3`.
The paper does not publish its E/S artifact, so these are auditable DARP results rather than copied reference output.
`—` means that the configuration has not produced a successful row in the raw CSV.

| h | Δ | Full-ILP Obj F | Full-ILP Obj E | Full-ILP Obj S | Full-ILP Time F | Full-ILP Time E | Full-ILP Time S | Full-ILP n F | Full-ILP n E | Full-ILP n S | Full-ILP Act.n F | Full-ILP Act.n E | Full-ILP Act.n S | DARP-HILP Obj F | DARP-HILP Obj E | DARP-HILP Obj S | DARP-HILP Time F | DARP-HILP Time E | DARP-HILP Time S | DARP-HILP Exp.n F | DARP-HILP Exp.n E | DARP-HILP Exp.n S | DARP-HILP Exp.% F | DARP-HILP Exp.% E | DARP-HILP Exp.% S |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.1 | 8.93 | 8.93 | 9.59 | 1.06 | 0.45 | 6.45 | 785 | 785 | 8010 | 628 | 628 | 6408 | 8.93 | 8.93 | 9.59 | 0.07 | 0.08 | 0.17 | 268 | 268 | 404 | 42.7 | 42.7 | 6.3 |

## Policy execution

Each cell is `sampled discounted return / evaluate seconds` from one `agent.evaluate(env, episodes=1)` call. Return follows the RDDL reward sign and is not the expected planner objective; time is not modeled action duration.

| h | Δ | Full-ILP F | Full-ILP E | Full-ILP S | DARP-HILP F | DARP-HILP E | DARP-HILP S |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.1 | -3.00 / 0.007161 | -3.00 / 0.003914 | -4.00 / 0.004184 | -3.00 / 0.003070 | -3.00 / 0.003037 | -4.00 / 0.004606 |

Failed trials recorded in raw CSV: 0.
