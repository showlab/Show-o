# Native-Objective DRSA Adapter Warm-up 实验设计 V2

本文件是当前生效的 canonical 实验协议。V1 的未加权任务表达已被 smoke 梯度数据修订。

## 目标

冻结官方 `showlab/show-o2-1.5B` 的 Show-o2/SigLIP/fusion/Qwen/diffusion head/VAE，只训练新增 DRSA understanding、generation 和 relation encoder。先用 Show-o2 原生的 flow/NTP 目标让 adapter 适应已有 hidden/velocity 分布，再进入 BoxDiff/GLIGEN/MS-Diffusion grounding 消融。

## 数据声明

官方 stage-1/stage-2 配置的数据路径没有随仓库发布，本机也没有 66M/100M 官方 full native data。当前使用：

```text
SPRIGHT-COCO original_caption + image -> T2I velocity flow
VSR image + relation statement       -> MMU yes/no NTP
```

它们是 native-objective proxy，不是官方训练集复现。数据审计：`/root/autodl-tmp/runs/showo2_native_warmup/native_warmup_data_manifest.json`。

## 冻结与参数组

```text
fusion_proj -> DRSA.U -> Qwen -> DRSA.G -> diff_proj -> diffusion head
```

```text
lr_u        = 1e-5
lr_g        = 5e-6
lr_relation = 5e-6
```

U/G/relation 必须是独立 AdamW parameter groups；禁止退回单一学习率。adapter 的 output projection 和 relative-bias 最后一层保持零初始化，使初始 residual 接近零。

## 生效 Loss

```text
SPRIGHT odd step:
    L = 1.0 * MSE(v_pred, velocity_target)
    caption = original_caption

VSR even step:
    L = 0.01 * CE(logits, yes/no target)
```

本阶段关闭 BoxDiff inside/outside/corner/relation loss、SPRIGHT spatial-caption alignment 和旧 retention 项，以隔离 native warm-up。VSR step 不经过 G，故 `grad_G=0` 必须成立。

`lambda_mmu=0.01` 来自未缩放 smoke：VSR U raw gradient 为 `3.89-4.24`，而 T2I U 仅 `0.015-0.033`。balanced smoke 后 VSR U 降为 `0.039-0.041`，已与 T2I U 进入同一数量级。

## Smoke 结论

成功运行：`native-warmup-balanced-smoke-20260807T063000Z`。

| step | task | weighted loss | U grad | G grad |
|---:|---|---:|---:|---:|
| 1 | T2I | 0.08643 | 0.01538 | 0.09946 |
| 2 | MMU | 0.00855 | 0.03908 | 0 |
| 3 | T2I | 0.10742 | 0.03275 | 0.25390 |
| 4 | MMU | 0.01750 | 0.04106 | 0 |

主干冻结检查为 true。checkpoint 中 U output norm `0.02921`、G output norm `0.00790`，两路均实际更新。该 smoke 只证明数据和反传路径成立，不证明 Position/T2I 指标提升。

## 后续实验顺序

1. 运行 500-step native warm-up，保存 step-100/250/500。
2. 每个 checkpoint 测固定 VSR 子集与 Position paired retention。
3. 增加 U/G/relation AdamW actual update norm 与跨任务 U-gradient cosine。
4. 在同一 native warm-up checkpoint 上做 BoxDiff U-only、G-only、U+G。
5. 只有显式 box loss 有稳定增益后，再引入 GLIGEN grounding token 或 MS-Diffusion grounding resampler。

正式 BoxDiff 训练必须使用 COCO instance box 或经人工抽检的 grounding pseudo box；GenEval 规则框只允许用于实现 smoke。
