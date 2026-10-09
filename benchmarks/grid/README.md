# Grid benchmark

文件名为 `grid-<规模>-h<horizon>-d<duration编号>-r<risk编号>-b<预算>.rddl`。同目录的 `domain.rddl` 与 instance 配对使用；实际问题以 RDDL 内容为准。编号不代表时长数值。fixed-duration 的 d2 示例表示固定时长 2，不进入原论文 d1 实验矩阵。

| 目录 | d1 的含义 |
| --- | --- |
| fixed-duration | 固定时长 1；用于 Table 1 F、Table 2 和剪枝对比 |
| expected-duration | 论文 E 的当前代表性均值模型：接触泥地时长 2，其余 1，方差 0 |
| stochastic-duration | 同样的状态相关均值，Normal 方差 0.1，shortfall 概率阈值 0.3 |
| state-dependent-duration | 确定性状态相关时长；与当前 E 输入相同，供独立验证，不作为第四个原论文 Table 1 模型 |

r1 表示危险格模板 `(0,0),(3,0),(3,1),(1,3),(1,4)`，大网格按坐标模 5 平铺；b 是整棵策略首次失败概率预算。起点为左下角，目标为右上角。转移和观测正确率均为 0.85。

`h` 是 duration horizon。alpha、lambda 是求解算法参数，由命令行指定，不属于 RDDL 文件名。`heuristic.py:MANHATTAN` 提供负 Manhattan 距离上界；实验使用终端 heuristic replacement。

现有 E/S 配置与论文原表存在已记录差异，不能仅由目录名称宣称完整复现。详见 [实验协议](../../docs/EXPERIMENT_PROTOCOL.md)。
