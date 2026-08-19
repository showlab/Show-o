# Native-Objective Warm-up Retention 结果

## 协议

评测对象：`native-warmup-500-20260807T070000Z` 的 step-100/200/300/400/500。

VSR 使用 zero-shot test 的固定前 256 条。Position 使用 `position_unseen72_v1.jsonl` 的前 8 个 prompt，每 prompt 4 个样本、固定 seed `12026`、12 个 Euler 采样步。baseline 与所有 checkpoint 的 32 个 initial latent pair 均满足 `all_complete=true`、`all_match=true`。

Position 是小规模 retention development set，不是 50-step/full-72 最终结论。

## VSR

| variant | correct | accuracy | delta vs baseline |
|---|---:|---:|---:|
| baseline | 145/256 | 56.64% | 0 |
| native-100 | 146/256 | 57.03% | +1 |
| native-200 | 147/256 | 57.42% | +2 |
| native-300 | 147/256 | 57.42% | +2 |
| native-400 | 149/256 | 58.20% | +4 |
| native-500 | 148/256 | 57.81% | +3 |

VSR 最佳为 step-400。step-500 相比 step-400 回落 1 个正确样本，说明继续 warm-up 未带来单调收益。

## Position Paired Retention

| variant | correct | gain | loss | net | exact paired retention |
|---|---:|---:|---:|---:|---:|
| baseline | 5/32 | 0 | 0 | 0 | reference |
| native-100 | 5/32 | 1 | 1 | 0 | no |
| native-200 | 4/32 | 0 | 1 | -1 | no |
| native-300 | 5/32 | 1 | 1 | 0 | no |
| native-400 | 5/32 | 0 | 0 | 0 | yes, 32/32 unchanged |
| native-500 | 5/32 | 1 | 1 | 0 | no |

step-100/300/500 的总正确数不变，但都在同一个 above prompt 上发生 1 gain + 1 loss；它们不是逐样本严格 retention。step-200 丢失 1 个 below 样本。step-400 是唯一在 32 个 paired 样本上与 baseline 完全一致的 checkpoint。

该子集的 baseline 仅为 15.63%，关系构成为 above 16、below 4、left 4、right 8；left/right 全部为错。因此它适合检查明显退化，不足以证明位置生成已经改善。

## 选择

后续 BoxDiff U-only、G-only、U+G 消融统一从以下 checkpoint 初始化：

```text
/root/autodl-tmp/runs/showo2_native_warmup/native-warmup-500-20260807T070000Z/checkpoints/step-000400.pt
```

选择理由：VSR 256 子集最佳，同时 Position 32 paired 样本零 gain、零 loss、逐样本完全保持。step-500 不作为默认初始化点。

## 限制与下一步

1. native warm-up 使用 SPRIGHT/VSR native-objective proxy，不是官方 full native data。
2. Position 使用 12 步而不是官方 50 步，且只有 8 prompts。
3. step-400 只证明当前 retention gate 最优，不证明 BoxDiff explicit box loss 会提升。
4. BoxDiff 消融必须使用相同 step-400 初始化、相同真实/pseudo box 数据和相同 paired evaluation protocol。

