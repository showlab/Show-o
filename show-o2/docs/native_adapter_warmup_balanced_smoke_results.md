# Native-Objective Balanced Warm-up Smoke 结果

## 运行信息

```text
run_id: native-warmup-balanced-smoke-20260807T063000Z
run_dir: /root/autodl-tmp/runs/showo2_native_warmup/native-warmup-balanced-smoke-20260807T063000Z
status: completed
steps: 4/4
elapsed: 12.61 seconds
```

数据为本地 SPRIGHT-COCO `original_caption + image` 和 VSR image-relation yes/no，属于 Show-o2 native-objective proxy，不是官方 66M/100M full native data。Show-o2 主干冻结检查为 `true`，可训练参数仅为 DRSA U/G/relation encoder，共 `19,437,504`。

## 配方

```text
L_t2i = 1.0  * L_flow
L_mmu = 0.01 * L_ntp

lr_u        = 1e-5
lr_g        = 5e-6
lr_relation = 5e-6
```

BoxDiff loss、spatial-caption alignment 和旧 retention 项均关闭。

## 结果

| step | task | raw loss | weighted loss | U grad | G grad | relation grad |
|---:|---|---:|---:|---:|---:|---:|
| 1 | SPRIGHT T2I | 0.08643 | 0.08643 | 0.01538 | 0.09946 | 0 |
| 2 | VSR MMU | 0.85547 | 0.00855 | 0.03908 | 0 | 0 |
| 3 | SPRIGHT T2I | 0.10742 | 0.10742 | 0.03275 | 0.25390 | 1.04e-11 |
| 4 | VSR MMU | 1.75000 | 0.01750 | 0.04106 | 0 | 2.40e-9 |

与未缩放 smoke 相比，MMU U gradient 从 `3.89-4.24` 降为 `0.039-0.041`，已与 T2I U gradient 的 `0.015-0.033` 进入同一数量级。VSR/MMU step 的 G gradient 继续严格为 0，符合结构预期。

T2I 中 G raw gradient 仍约为 U 的 `6.5-7.8x`。当前 G 学习率是 U 的一半，但 AdamW update 不与 raw gradient 严格成比例，因此短 warm-up 前仍需增加实际 update norm 记录，不能仅按 raw ratio 再降学习率。

relation encoder 早期梯度近乎为零，与零初始化 residual output/relative-bias 的逐步开门机制一致。该结果证明 native-objective warm-up 的数据和反传路径可运行，不证明 Position 或生成质量提升。

## 下一判定点

下一步不是直接加入 BoxDiff 长训练，而是用 500-step native warm-up checkpoint 做：

1. 固定 VSR 子集 retention；
2. 固定 Position paired generation retention；
3. U/G/relation AdamW update norm；
4. random-init 与 native-warm-up 初始化的 BoxDiff U-only/G-only/U+G 消融。

