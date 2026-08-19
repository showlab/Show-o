# Stage B 500-Step 训练记录

运行目录：`/root/autodl-tmp/runs/showo2_stage_b/stage-b-500-20260807T145350Z`

## 配方

- 初始化：native A selected checkpoint `step-000400.pt`
- 模式：`spatial_ramp`
- steps：500
- `native_steps=0`
- `spatial_ramp_steps=500`
- task pattern：2 个 SPRIGHT T2I : 1 个 VSR MMU
- checkpoint：每 100 step
- U LR：`1e-5`
- G LR：`5e-6`
- relation LR：`5e-6`
- global grad clip：`1.0`

## 运行状态

```text
status: completed
steps: 500/500
elapsed: 170.17 seconds
error: none
SPRIGHT T2I: 334 steps
VSR MMU: 166 steps
```

## 梯度和实际更新

SPRIGHT T2I 的平均 raw gradient：

```text
U = 3.3451e-2
G = 2.4062e-1
G/U = 7.19
```

平均 AdamW 实际参数更新：

```text
U = 3.8533e-3
G = 1.8708e-3
G/U = 0.485
```

因此 G 的 raw gradient 虽然明显更大，但半学习率和 AdamW 自适应归一化后，实际 G update 仍小于 U update，没有覆盖 U 路。

VSR/MMU：

```text
max G gradient = 0
max G update = 0
```

global clip：

```text
clipped steps = 1/500
max pre-clip global norm = 1.5
```

这不是持续性梯度爆炸；当前学习率不需要因为 G raw gradient 较大而立即调整。

## Loss 首尾趋势

SPRIGHT T2I raw loss 的前 50 / 后 50 step 均值：

| 项 | 前 50 | 后 50 |
|---|---:|---:|
| native flow | 0.11384 | 0.11190 |
| guided flow | 0.11306 | 0.11137 |
| retention | 0.000966 | 0.000909 |
| U box | 1.02881 | 1.02920 |
| G box | 1.11883 | 1.11691 |
| U relation | 0.04001 | 0.04107 |
| G relation | 0.04031 | 0.04033 |

native/guided flow 和 retention 有轻微下降，但 U/G box loss 基本持平。因此训练链路和梯度路由已经正常，空间目标是否真正改善不能由 loss 单独判断，必须进入阶段 B gate 评测。

最终 checkpoint：

```text
/root/autodl-tmp/runs/showo2_stage_b/stage-b-500-20260807T145350Z/checkpoints/step-000500.pt
```

下一步应对 step-100/200/300/400/500 做 VSR 256 和 Position paired gate，选择“空间 gain 不以 native retention loss 为代价”的最佳 checkpoint；在 checkpoint 选择完成前不直接进入阶段 C。
