# Experiments

输入在 `benchmarks/<场景>/<duration类型>/`；独立批次在 `experiments/<场景>/runs/<批次>/`。规模、horizon、预算和剪枝参数作为记录字段，不继续拆目录。

```bash
python experiments/run.py --scene grid --name smoke \
  --instances benchmarks/grid/fixed-duration/grid-5x5-h3-d1-r1-b0.1.rddl \
  --algorithms HILP pruningF pruningEF --alpha 1 --lambdas 0.5 --trials 1
```

批量实验读取配置；命令行参数覆盖同名配置：

```bash
python experiments/run.py --config experiments/grid/configs/table1.json --name table1-new
python experiments/run.py --config experiments/grid/configs/table2.json --name table2-new
python experiments/run.py --config experiments/grid/configs/pruning.json --name pruning-new
```

已有批次不会自动覆盖。`--resume` 只补做尚未记录的 trial；输入内容、参数和运行环境须匹配。失败 trial 保留，不自动重试。RAOstar 首次使用可缓存固定版本源码，也可传 `--constrained-repo`、`--raostar-repo` 指定本地 checkout。

交互选择已有批次、benchmark 或外部 RDDL：

```bash
python experiments/compare.py
```

外部 RDDL 使用独立场景名称，明确指定 domain、instance 和可选 heuristic。RAOstar 仅支持已有适配的固定时长 Grid。补跑前显示求解数量并要求确认。

非交互复用结果：

```bash
python experiments/compare.py --scene grid --name pruning \
  --runs experiments/grid/runs/pruning-HILP \
         experiments/grid/runs/pruning-pruningF \
         experiments/grid/runs/pruning-pruningEF
```

输出为 `experiments/output/grid-HILP-pruningF-pruningEF/pruning.tex`、`.pdf`、`.selection.json`。PDF 需要 `pdflatex`；`--tex-only` 仅生成源文件。相同表名须显式 `--overwrite`，也可另取表名。

每个批次保存 `config.json`、`results.csv` 和可选 `policies/`。策略文件路径相对本批次；selection 记录生成表格的数据来源。历史迁移批次保留全部原始数值列，缺失环境参数不补造，旧计时不用于自动标示加速；迁移清单位于 `grid/migration.json`。

对比按实际输入内容匹配。不同 duration/风险配置不合并；无相同配置、失败或缺失数据显示 `--`。同一算法的不同批次单独列出。加粗表示相对唯一 HILP 基准的 cost 变化；下划线仅用于设置与 trial/seed 配对的更低耗时。RAOstar 的原生 objective 边界与 DARP 不同，不计算它相对 DARP 的 cost 增量。
