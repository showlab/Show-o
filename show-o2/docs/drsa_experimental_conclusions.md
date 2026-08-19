# DRSA 实验结论与后续消融

## 范围

本记录总结 Show-o2 的 Dual-Route Spatial Alignment Adapter（DRSA）实验。U 路位于 `fusion_proj` 后、Qwen 主干前；G 路位于 Qwen 最终 hidden states 后、`diff_proj` 前。Show-o2、SigLIP、fusion、Qwen、diffusion head 和 VAE 权重冻结，训练参数只有 adapter。

## 训练目标审计

当前 SPRIGHT step 使用：

```text
L_spright = L_flow + 0.1 * L_alignment + 0.2 * L_retention
```

`L_flow` 是最终 diffusion velocity 与解析 velocity target 的 masked MSE。它监督整张 latent 的流场，不直接监督对象框、主体/客体绑定或 left/right、above/below 的方向。

`L_alignment` 只比较关系短语文本 code 与完整 spatial caption 文本 code 的 cosine；没有图像 token、box、mask、U/G hidden state 或 attention map。

`L_retention` 是 original caption 条件下再次计算同一 velocity target 的 MSE，不是 adapter 与 frozen baseline 的蒸馏，也不是 `v_spatial` 与 `v_original` 的显式一致性。它还复用了从 spatial caption 提取的 relation code。

VSR step 使用 understanding-only next-token cross-entropy。G 路不在计算图中，因此 G 不更新；U 和 shared relation encoder 更新。

## 主要实验数据

### Mixed DRSA 全量 benchmark

此前 mixed 训练的 20k checkpoint 与同 seed baseline：

| 指标 | Baseline | DRSA step-20000 | 变化 |
|---|---:|---:|---:|
| Position full | 60/288 = 20.83% | 50/288 = 17.36% | -10 correct |
| VSR full | 722/1222 = 59.08% | 871/1222 = 71.28% | +149 correct |

Position paired latent 288/288 完整匹配，transition 为 gain 11、loss 21、net -10。左右关系由于 baseline 已接近 floor，不能用绝对准确率判断是否下降；above 为 26->21，below 为 30->25。

### VSR-only 6000 消融

VSR-only 从随机 adapter 开始，G 显式冻结，每 1000 step 保存 checkpoint。验证使用同一 256 条 VSR 子集和 72 个 Position paired 样本，因此只用于曲线比较，不能把子集绝对值与 full benchmark 混用。

| Checkpoint | VSR 256 | Position 72 |
|---|---:|---:|
| Baseline | 145/256 = 56.64% | 6/72 = 8.33% |
| VSR-1000 | 158/256 = 61.72% | 7/72 = 9.72% |
| VSR-2000 | 168/256 = 65.63% | 7/72 = 9.72% |
| VSR-3000 | 168/256 = 65.63% | 6/72 = 8.33% |
| VSR-4000 | 173/256 = 67.58% | 7/72 = 9.72% |
| VSR-5000 | 171/256 = 66.80% | 6/72 = 8.33% |
| VSR-6000 | 172/256 = 67.19% | 7/72 = 9.72% |

这个消融显示：VSR-only 能明显改善 U 路理解，而在小规模 Position paired 集上没有出现 mixed 训练那样的明确净崩溃。它不能证明 U 漂移永远无害，但说明 mixed 生成 loss/G 路交互是重要嫌疑。

### 梯度探针

固定 4 个 SPRIGHT 样本、4 个 VSR 样本、同一 noise/timestep，测量 random、step-3000、step-20000：

| 状态 | SPRIGHT `||g_U||` | SPRIGHT `||g_G||` | raw U/G | VSR `||g_U||` | VSR `||g_G||` |
|---|---:|---:|---:|---:|---:|
| Random | 0.0281 | 0.1934 | 0.148 | 4.126 | 0 |
| step-3000 | 0.0287 | 0.0597 | 0.456 | 2.366 | 0 |
| step-20000 | 0.0260 | 0.0653 | 0.376 | 2.700 | 0 |

SPRIGHT 上 G 的 flow raw gradient 大于 U；VSR 上 G 严格为零，但 U 梯度比 SPRIGHT U 大约 80-100 倍。参数 RMS 近似相同，因此不是参数尺度造成的表面差异。

step-3000 和 step-20000 的 optimizer state 中学习率均为 `1e-5`；AdamW update ratio 分别约为 U/G=`0.54` 和 `0.48`。这说明 raw gradient 不能直接当作实际 update。后续 U/G 必须使用独立 parameter groups 和独立学习率，不能继续假设同一学习率合理；关系 encoder 还要单独监测跨任务 cosine。

完整梯度 JSON：`/root/autodl-tmp/runs/showo2_drsa_train/gradient-probe-20260807T100500Z/results.json`。

## 结论

1. VSR 是清晰的 understanding supervision，确实能改善 U 路理解。
2. SPRIGHT 的 `L_flow` 对最终 latent velocity 有意义，但没有显式位置语义；单独把它称作 spatial alignment 不充分。
3. `L_alignment` 是 text-text code alignment，不是 image-text、U-G 或 object-layout alignment。
4. `L_retention` 没有真正的 baseline teacher，且可能削弱空间 caption 的独特作用。
5. Mixed 训练出现“VSR 提升、Position full 下降”，说明联合目标存在明显 objective mismatch；20k 续训时的学习率跳变是额外混杂因素。
6. VSR-only 没有在轻量 paired Position 上重现同等崩溃，支持继续做任务消融。

## 后续消融顺序

```text
A. VSR-only U + relation encoder（已完成）
B. SPRIGHT-only U/G，拆开观察 flow 对两路的影响
C. U/G 分组学习率 + shared relation encoder 单独学习率
D. BoxDiff-style explicit box/attention loss
E. GLIGEN-style phrase+box grounding condition
F. MS-Diffusion-style grounding resampler/multi-subject condition
```

BoxDiff 路线先使用 Position manifest 的规则框做 controlled smoke；它不是 SPRIGHT 的真实框标注。正式训练前需要真实 box/mask 或 Grounding DINO/GLIP pseudo-box，并且必须用未参与选择的 spatial benchmark 复核。
