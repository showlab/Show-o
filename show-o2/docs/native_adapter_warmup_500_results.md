# Native-Objective DRSA Warm-up 500-step 结果

## 运行信息

```text
run_id: native-warmup-500-20260807T070000Z
run_dir: /root/autodl-tmp/runs/showo2_native_warmup/native-warmup-500-20260807T070000Z
status: completed
steps: 500/500
tasks: 250 SPRIGHT T2I + 250 VSR MMU
elapsed: 76.78 seconds
```

Show-o2 主干冻结检查为 true。训练参数仅为 DRSA U/G/relation，共 `19,437,504`。使用 native-objective proxy，而非官方 66M/100M full native data。

## 配方

```text
L_t2i = 1.0  * L_flow
L_mmu = 0.01 * L_ntp

lr_u        = 1e-5
lr_g        = 5e-6
lr_relation = 5e-6
warmup      = 50 steps
grad clip   = 1.0
checkpoint  = every 100 steps
```

BoxDiff、spatial-caption alignment 和旧 retention loss 均关闭。

## 500-step 统计

| task/group | raw grad mean | clipped grad mean | actual update mean | actual update median | actual update max |
|---|---:|---:|---:|---:|---:|
| T2I U | 0.02976 | 0.02921 | 0.004109 | 0.003058 | 0.011781 |
| T2I G | 0.22108 | 0.21653 | 0.001776 | 0.001709 | 0.003585 |
| MMU U | 0.02130 | 0.02130 | 0.004356 | 0.003142 | 0.012763 |
| MMU G | 0 | 0 | 0 | 0 | 0 |

T2I G/U raw gradient 均值约 `7.43`，但 AdamW actual update 的 G/U step ratio 平均约 `0.53`。因此不能依据 raw gradient ratio 直接等比例缩小 G 学习率；当前独立学习率使 T2I U、T2I G、MMU U 的实际更新保持在同一数量级。

全局 grad clip 在 250 个 T2I step 中触发 3 次，在 MMU step 中未触发。relation encoder 在 500 步中的 498 步产生非零 actual update，说明零初始化门控已经打开。

## Loss 观察

```text
T2I flow first 25 mean: 0.11455
T2I flow last 25 mean:  0.12910

VSR NTP first 25 mean: 1.40609
VSR NTP last 25 mean:  0.36879
```

VSR NTP 明显下降。T2I flow 没有呈单调下降，但前后窗口来自不同图像和随机 timestep，不能把这个差异直接解释为退化；需要 paired/fixed-noise retention benchmark。

## Checkpoints

```text
step-000100.pt
step-000200.pt
step-000300.pt
step-000400.pt
step-000500.pt
```

每个 checkpoint 约 116.7 MB，运行目录约 557 MB。latest 指向 `step-000500.pt`。

## 当前结论

500-step native-objective warm-up 在数值和实际更新尺度上稳定，U/G/relation 均按结构预期工作。该结果仍不证明生成或 Position 指标提升；下一步必须对 step-100/200/300/400/500 做固定 VSR 与 Position paired retention，再选择进入 BoxDiff 消融的初始化 checkpoint。
