# DARP–RAO* Quad 实验协议

## 1. 研究问题与证据边界

当前唯一保留的外部对照是固定 `JarvisIsFriday/RAOStar` revision 上的原生 `QuadModel`：

1. 当 DARP-HILP 与上游 RAO* 调用完全相同的模型回调时，是否得到满足同一 chance budget 的完整策略？
2. 两者的 native cost、first-entry execution risk、搜索完成率、节点/迭代数和 wall-clock time 如何变化？
3. 在很小 horizon 上，DARP-HILP 与 DARP full-ILP 的策略与独立评价是否一致？

该实验是 shared-model callback-parity 与 artifact comparison。Quad 完全可观测且规模小；公开仓库又是第三方 RAO* reimplementation，不是 AAAI 2016 作者 artifact。因此结果只能用其 manifest 标签，不能写成“复现 RAO* 论文 PSR”或“证明一般 CC-POMDP SOTA”。RAO* 原论文见 [AAAI 2016 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/10423)。

## 2. 固定实验对象

上游 revision：

```text
repository  https://github.com/JarvisIsFriday/RAOStar
commit      543f782d80ceb9555130e911c1fcf7074153d267
scenario    quad_raos.py + models/quad_model.py
label       jarvisisfriday-raostar-reimplementation
license     license-not-provided; external checkout only
```

`experiments/manifests/raostar_quad.json` 固定所需文件、Python/NumPy 版本、world、goal、初始状态、horizon 和 chance constraint。runner 在运行前后检查 commit、tracked/untracked 状态、HEAD blob 和本地 runner/bridge/manifest 哈希，并从 `git archive` 临时快照执行上游搜索。

DARP bridge 不复制模型公式。两者共同使用：

- state-dependent `actions`；
- `state_transitions` 与 `observations`；
- `values` 与 `heuristic`；
- `state_risk` 与 `is_terminal`。

DARP 只执行 `utility = -upstream_value` 的符号变换。双方完整策略都转换为相同的 observation-history policy protocol，再由独立 evaluator 直接调用上游回调重算 native cost 与 first-entry risk。

上游目标有一个必须披露的特殊性：`values(state, action)` 每步包含 action cost 和当前 state heuristic，fixed-horizon 或提前 terminal 叶节点还会再调用 heuristic。因此 `native_cost` 不是纯路径成本，不能与采用其他 reward 定义的实验混表。

## 3. 实验矩阵

主矩阵：

```text
horizon       2, 3, 4
risk budget   0, 0.25, 0.5
algorithms    darp-hilp, jarvisisfriday-raostar-reimplementation
repetitions   >= 25 fresh processes per cell
timeout       identical wall-clock limit per planner run
```

结构 oracle 仅增加 `darp-full-ilp`，且默认限制在 horizon 2。full-ILP 的指数树增长使它不适合作为大规模主算法。

上游搜索从 graph-node 对象的 `set` 中选择 open node；对象 identity 顺序不由 `PYTHONHASHSEED` 固定。并列展开可能在 fresh process 间返回不同策略，所以主结果不能只运行一次。每次 repetition 都必须新建 DARP planner，并单独启动 RAO* subprocess。

## 4. 运行命令

```bash
bash tools/install.sh

git clone https://github.com/JarvisIsFriday/RAOStar /path/to/RAOStar
git -C /path/to/RAOStar checkout --detach 543f782d80ceb9555130e911c1fcf7074153d267
mkdir -p experiments/outputs/raostar-quad

.venv/bin/python -m experiments.scripts.run_raostar_quad \
  --checkout /path/to/RAOStar \
  --python .venv/bin/python \
  --accept-no-license \
  --horizons 2 3 4 \
  --risk-budgets 0 0.25 0.5 \
  --repetitions 25 \
  --timeout 300 \
  --include-full-ilp \
  --full-ilp-max-horizon 2 \
  --output experiments/outputs/raostar-quad/runs.jsonl
```

固定 manifest 当前要求 CPython 3.12.3 与 NumPy 2.4.6。若项目环境不同，应创建满足 manifest 的独立解释器并传给 `--python`，不能绕过环境核验。提交论文时同时归档 `runs.jsonl`、`requirements-lock.txt` 和 Git revision，并在论文 artifact 说明中记录 Gurobi 版本/许可证类型、CPU、内存和操作系统信息。

快速验证可用：

```bash
OUTPUT_DIR=/tmp/darp-repro tools/run_repro.sh

RAOSTAR_CHECKOUT=/path/to/RAOStar \
RAOSTAR_PYTHON=.venv/bin/python \
RAOSTAR_ACCEPT_NO_LICENSE=1 \
OUTPUT_DIR=/tmp/darp-repro-with-raostar \
tools/run_repro.sh
```

## 5. 结果过滤与统计

每个配置先报告运行总数，再按以下顺序过滤可独立评价的策略：

1. source/environment/provenance 核验通过；
2. `status == "ok"`；
3. DARP 的 `policy_duration_complete` 为真，或 RAO* 的 `search_complete` 为真并成功导出策略；
4. independent evaluator 成功；
5. `first_entry_risk <= risk_budget`。

`search_complete`、数值 gap 和 admissibility warning 是必须报告的结果维度，而不是把可行完整策略静默删除的统一前置条件；当前浮点系数无法给出数学 complete certificate。失败、超时和 partial policy 不能从分母删除，必须单独报告比例与原因。对通过上述独立评价过滤的运行报告：

- native cost：median、IQR、mean 和 95% bootstrap CI；
- planning time：median、IQR、95% bootstrap CI；
- first-entry risk、节点数、迭代数；
- root action/policy digest 的频率分布；
- 完成率、风险可行率和 admissibility-warning rate。

算法配对比较应按相同 `(horizon, risk budget, repetition)` 组织。时间分布通常偏斜，优先给出 paired bootstrap difference/ratio；若做显著性检验，应报告效应量、多重比较校正和原始样本，不只报告 p-value。

Gurobi `MIPGap=0`/`MIPGapAbs=0` 只能说明浮点模型的数值 gap 闭合。`mathematically_optimal` 必须保持 false，除非未来加入独立的精确最优性证明。DARP 的选中策略仍由 `Fraction` 重算约束与 achieved utility；这证明所选策略在已编码模型中的精确可行性，不证明全局目标最优。
