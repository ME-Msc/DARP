# Benchmark Domains

This folder contains the upstream RDDL MDP benchmark domains copied from the
PROST/IPC benchmark collection. In DARP, these files are kept as example data
for parser compatibility checks, future benchmark runners, and later
algorithmic experiments.

Not every benchmark domain is supported by the current DARP compiler yet. The
current compiler targets small discrete RDDL subsets first; broader IPC domain
coverage will be added incrementally through the roadmap.

Please keep benchmark source files under `examples/benchmarks/<domain-year>/`.
Tests should reference these files instead of duplicating benchmark data under
`tests/`.

This folder aims to contain all RDDL MDP domains that were used in any IPC.
Please contact the upstream benchmark maintainers if you find domains that are
missing from the original collection.

Author:   Thomas Keller
Date:     March, 2019
Contact:  tho.keller [@] unibas.ch

===[ Subfolders ]===

Each subfolder corresponds to a domain in the format `domain-year` where
`domain` is the respective name and `year` is the IPC year where it was first
introduced. A few domains were reused in different IPC editions. In such cases,
we keep several folders if each edition used different instances. In the cases
where all instances were identical, we just keep the folder of the first IPC
edition which introduced the domain.

===[ DARP layout ]===

- `examples/benchmarks/README.md`: this benchmark-corpus note.
- `examples/benchmarks/<domain-year>/*_mdp.rddl`: one domain file, or the
  domain-style files provided by the upstream benchmark.
- `examples/benchmarks/<domain-year>/*_inst_mdp__*.rddl`: benchmark instances.

Current imported corpus:

| Domain folder | RDDL files |
| --- | ---: |
| academic-advising-2014 | 11 |
| academic-advising-2018 | 21 |
| chromatic-dice-2018 | 21 |
| cooperative-recon-2018 | 21 |
| crossing-traffic-2011 | 11 |
| earth-observation-2018 | 21 |
| elevators-2011 | 11 |
| game-of-life-2011 | 11 |
| manufacturer-2018 | 21 |
| navigation-2011 | 11 |
| push-your-luck-2018 | 21 |
| recon-2011 | 11 |
| red-finned-blue-eye-2018 | 21 |
| skill-teaching-2011 | 11 |
| sysadmin-2011 | 11 |
| tamarisk-2014 | 11 |
| traffic-2011 | 11 |
| triangle-tireworld-2014 | 11 |
| wildfire-2014 | 11 |
| wildlife-preserve-2018 | 40 |
