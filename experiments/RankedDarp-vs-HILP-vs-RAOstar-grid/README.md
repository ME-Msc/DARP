# Ranked DARP vs HILP vs RAO* grid

比较原 HILP、筛选 E+F 的 prun-HILP（Rank）、只筛 F 的 F-only。HILP 与 E+F 使用已完成首批数据，F-only 为后续补跑；均串行运行。RAO* 引用基础实验的历史 Table 2。

配置：5×5/100×100，h=3/4/5/6，Δ=0.1/0.2/0.3，α=1，λ=0.3/0.5/0.7/0.9；每方法每配置 3 次，完整矩阵为 648 次搜索。每配置各方法的第一次策略保存后回放 1000 条轨迹，共 216 份策略。回放不计入搜索时间。表格集中在 `output/letax/`（按用户指定拼写），结果只输出 LaTeX 和 PDF。算法见[方案说明](../../docs/ALGORITHM_MAPPING.md#rank-subtree)。

## 复现

```bash
.venv/bin/python -m experiments.RankedDarp-vs-HILP-vs-RAOstar-grid.run \
  --trials 3 --timeout 120 --episodes 1000
```

中断后在同一命令添加 `--resume`。只重新汇总已有数据并生成 LaTeX 表（不求解）：

```bash
.venv/bin/python -m experiments.RankedDarp-vs-HILP-vs-RAOstar-grid.run \
  --summary-only
```

默认包含 F-only（禁用 `_prune_expanded`），使用 `--no-ablation` 可省略。F-only 仍保留必要的不可行预筛选；HILP（λ=1）始终包含。`--resume` 跳过已有记录，只补缺少的实验；metadata 保留首批和补跑设置。

## 保留的产物

| 文件 | 用途 |
|:--|:--|
| [output/letax/pruning-comparison.tex](output/letax/pruning-comparison.tex) | 按网格及剪枝方式分页；沿用 Obj./Time/n/Iter. 分组列 |
| [output/letax/pruning-comparison.pdf](output/letax/pruning-comparison.pdf) | 编译后的结果表；下划线表示更快，粗体表示目标值变化 |
| [output/pruning-comparison-raw.csv](output/pruning-comparison-raw.csv) | 逐次搜索、风险及执行结果 |
| [output/pruning-comparison-raw.metadata.json](output/pruning-comparison-raw.metadata.json) | 首批及补跑参数、软件版本和 Gurobi 设置 |
| `output/results/pruning-comparison-raw/` | 各配置 trial 1 的完整策略 JSON |

旧 `subtree-raw` 产物已清理；`retention-raw` 已更名并修正 CSV 中的策略路径。旧结果表和冒烟产物不参与新统计。正式 CSV、metadata 和当前策略 JSON 应保留，便于论文追溯和独立回放。

## 结果边界

`Cost increase (%) = 100 × (C_pruned - C_HILP) / |C_HILP|`。诊断表报告各配置中位成本的最大相对增量；成本越小越好，加粗表示变化而非优势。加速比为各配置 HILP/剪枝中位时间比的中位数。下划线表示观察到的更快，不代表显著性检验。近似筛选允许成本上升，不保证最优性；F-only 补跑与首批计时日期不同，metadata 保留批次信息。

编译表格：

```bash
latexmk -pdf -outdir=experiments/RankedDarp-vs-HILP-vs-RAOstar-grid/output/letax \
  experiments/RankedDarp-vs-HILP-vs-RAOstar-grid/output/letax/pruning-comparison.tex
```
