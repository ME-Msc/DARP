# Power Supply: DARP-HILP vs RAO*

| Δ | Algorithm | Time (s), 1/2 sensors | Nodes, 1/2 sensors | Evaluated particles†, 1/2 sensors | Cost, 1/2 sensors | Risk, 1/2 sensors |
|---:|:---|---:|---:|---:|---:|---:|
| 0 | DARP-HILP | 3.059/2.923 | 276.0/276.0 | —/— | 5.000/5.000 | 0.000/0.000 |
| 0 | RAO* | 0.052/0.051 | 15.0/15.0 | 30.0/30.0 | 5.000/5.000 | 0.000/0.000 |
| 0.5 | DARP-HILP | 3.782/3.946 | 317.0/326.0 | —/— | 2.500/2.500 | 0.500/0.500 |
| 0.5 | RAO* | 0.041/0.043 | 15.0/16.0 | 25.0/25.0 | 2.500/2.500 | 0.500/0.500 |
| 1 | DARP-HILP | 12.552/13.382 | 592.0/599.0 | —/— | 2.500/2.500 | 0.500/0.500 |
| 1 | RAO* | 0.049/0.055 | 22.0/23.0 | 39.0/39.0 | 2.500/2.500 | 0.500/0.500 |

Trials per configuration: 3; cells show arithmetic means. The slash follows RAO* Table 2 and means 1/2 breaker sensors.

DARP nodes are expanded action histories; RAO* nodes are expanded belief hypergraph nodes. † This is RAO*'s evaluated-belief-particle counter; DARP has no identical counter, so its cell is shown as —.

## Scope and provenance

- Semantic scope: `darp_authored_reduced_psr`; horizon 4; fixed unit duration.
- Network: `Bonet-Thiebaux-three-line`; prior: `uniform exactly-one fault on l1/l2`.
- Domain sources: [Thiébaux & Cordier 2001](https://users.cecs.anu.edu.au/~thiebaux/papers/ecp01.pdf) and the executable GPT formalization in [Bonet & Thiébaux 2003](https://users.cecs.anu.edu.au/~thiebaux/papers/icaps03.pdf); the [official benchmark page](https://users.cecs.anu.edu.au/~thiebaux/benchmarks/pds/) indexes network data/tools.
- Costs follow the GPT PSR formalization: one per switch operation and five per healthy line left unpowered at finish.
- Connecting a generator to a hidden fault sets a nonterminal unsafe state; the chance constraint bounds the probability of ever reaching it.
- RAO* source: [https://github.com/ME-Msc/rao-star.git](https://github.com/ME-Msc/rao-star.git) at `f51bfdc1ff8fcb2504dcb38c3b36d719506501fb`.
- This is a DARP-authored reduced benchmark whose topology, action effects, and finish cost follow the cited PSR work. The prior, horizon, sensor subsets, and chance constraint are explicit experiment choices.
- It is not the original RAO* 2016 semi-rural instance or a numerical reproduction of Table 2.
