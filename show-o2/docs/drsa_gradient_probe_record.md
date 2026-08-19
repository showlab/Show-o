# DRSA U/G 梯度探针记录

来源：`runs/showo2_drsa_train/gradient-probe-20260807T100500Z/results.json`，固定 4 个 SPRIGHT 样本、4 个 VSR 样本和同一 noise/timestep，比较 random、step-3000、step-20000。

| state | SPRIGHT U | SPRIGHT G | raw U/G | VSR U | VSR G |
|---|---:|---:|---:|---:|---:|
| random | 0.028140 | 0.193383 | 0.147642 | 4.125728 | 0 |
| step-3000 | 0.028691 | 0.059718 | 0.455792 | 2.365843 | 0 |
| step-20000 | 0.025970 | 0.065336 | 0.376313 | 2.700144 | 0 |

`raw U/G` 是 SPRIGHT flow 的 U/G raw gradient norm ratio，不是 cosine。U 与 G 是不相交的参数空间，不能直接把两者 flatten 后计算 cosine；cosine 只对共享 relation encoder 或同一路跨任务梯度有意义。

step-3000 的 relation encoder 跨任务 relation 约为 `8.0e-5`，step-20000 约为 `3.42e-4`；对应 VSR 约为 `4.92e-3` 和 `6.64e-3`。VSR 的 G 梯度严格为 0，因为 `forward_und_only` 不经过 generation adapter。

U/G 参数 RMS 近似相同，说明差异主要来自目标路径而不是参数初始化尺度。AdamW 的实际 update ratio 也不等于 raw ratio：step-3000 约 0.54，step-20000 约 0.48。因而后续必须把 U、G、relation encoder 放到独立 parameter groups，至少使用 `1e-5 / 5e-6 / 5e-6` 的分组起点，并持续记录 raw、clipped、AdamW update 三类量。

本记录是 BoxDiff adapter smoke 的学习率参考，不是最终超参结论。BoxDiff loss 会改变梯度结构，正式训练前必须重新 probe。

