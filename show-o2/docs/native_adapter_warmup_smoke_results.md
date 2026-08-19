# Native-Objective Warm-up Smoke 结果与协议修订

本记录补充 `docs/native_adapter_warmup_design.md`。第一轮未缩放 smoke 发现 task 梯度失衡，因此后续 balanced 协议优先于原设计中的 1:1 未加权表达。

## 数据与冻结状态

运行 `native-warmup-smoke-20260807T060100Z`，使用本地 SPRIGHT-COCO `original_caption + image` 计算 T2I velocity flow，使用 VSR image-relation yes/no 计算 MMU NTP。官方 66M/100M full native data 未落盘；这仍是 native-objective proxy。Show-o2 主干和 VAE 均冻结，训练参数仅为 DRSA U、G 和 relation encoder。

## 第一轮未缩放结果

| step | task | raw loss | U grad | G grad | relation grad |
|---:|---|---:|---:|---:|---:|
| 1 | SPRIGHT T2I | 0.08643 | 0.01537 | 0.09941 | 0 |
| 2 | VSR MMU | 0.85547 | 3.89152 | 0 | 0 |
| 3 | SPRIGHT T2I | 0.10742 | 0.03277 | 0.25401 | 2.24e-11 |
| 4 | VSR MMU | 1.74219 | 4.23848 | 0 | 4.74e-7 |

T2I 中 G/U raw gradient ratio 约为 `6.5-7.8`；MMU 对 U 的梯度比 T2I 对 U 大约 `130-250x`。U/G 使用独立学习率是必要但不充分的，因为同一 U 参数组仍同时承受两个量级不同的任务目标。

checkpoint 中零初始化 output 已实际更新：U output norm `0.02877`，G output norm `0.00791`。VSR step 的 G gradient 严格为 0，符合结构预期。

## Balanced smoke 修订

下一轮使用：

```text
L_t2i_weighted = 1.0  * L_flow
L_mmu_weighted = 0.01 * L_ntp

lr_u        = 1e-5
lr_g        = 5e-6
lr_relation = 5e-6
```

`0.01` 只用于把 MMU U gradient 降到与 T2I U 同一数量级，是 smoke 起点而非最终超参。正式训练前还需要测 AdamW 实际 update norm、跨任务 U gradient cosine 和 retention benchmark。
