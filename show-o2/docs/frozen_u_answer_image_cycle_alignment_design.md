# 冻结 U 教师的答案图监督生成回路设计

更新时间：2026-08-11 UTC

状态：已被 prompt-only canonical 设计取代，仅保留为历史设计稿；尚未实现或启动训练。

当前 canonical 文件：`docs/frozen_u_prompt_conditioned_generation_cycle_design.md`。

本文定义 Show-o2 空间生成下一阶段的 canonical 设计：以 `prompt + 示例图` 作为条件生成预测图，以配对答案图同时提供 Flow、Mask2Former 和冻结 U 路监督。预测图必须进入实际图像域，再回传到冻结 U 路；U 参数不更新，但 U 对预测图输入的梯度继续回到 G 路。

本文不把 U/G 原始 hidden state 直接设为相等。U 偏高层图文理解，G 偏时间条件化的细节生成，两者表示空间可以功能等价但数值不一致。核心改为：答案图和预测图都经过同一个冻结 U 路，在统一 U 空间中比较行为、对象表征和图文 grounding。

## 1. 研究问题与假设

当前证据显示：

```text
U route：VSR/空间关系理解能力较强
G route：对象存在、绑定和方向生成不稳定
Mask2Former：可提高外部 detector 中的对象证据
Flow/retention：可约束 velocity，但不能保证生成图被 U 正确理解
```

主假设为：

> 冻结已经具备空间理解能力的 U 路，把答案图作为 U teacher，把 G 生成的预测图作为 U student 输入，通过答案行为蒸馏和 U-space perceptual alignment，将 U 的对象空间理解能力迁移到 G，同时保留 Mask 对象监督和 Flow 生成动力学。

需要验证的子假设：

1. Mask loss 负责对象存在和像素区域，U-cycle 负责模型内部可理解语义，两者互补。
2. Flow 和 B1-G retention 可降低 frozen-U critic 被对抗性利用或生成轨迹偏离。
3. 答案图 teacher 比直接对齐 U/G 原始 hidden 更稳定，因为 teacher/student 均落在同一个 U 表示空间。
4. 单步 predicted-clean 可能仍有 sampler gap，需要用短程可微 rollout 逐步验证。

## 2. 数据合同

正式样本定义为：

```text
D_i = {
  prompt:       p_i,
  example_image:I_i^ex,
  answer_image: I_i^ans,
  answer:       y_i,
  objects:      [A_i, B_i],
  relation:     r_i,
  inverse:      inverse(r_i),
  metadata:     source / split / ids
}
```

字段语义：

| 字段 | 用途 |
|---|---|
| `prompt` | 定义基于示例图要完成的空间生成任务 |
| `example_image` | 输入条件或任务上下文，必须同时提供给 teacher/student |
| `answer_image` | 真实目标图；Flow、Mask teacher 和 U teacher 的共同监督源 |
| `answer` | 冻结 U 应输出的结构化答案或关系标签 |
| `objects/relation` | Mask class conditioning、正反问题和方向评测 |

理想正式协议是 `I_ex != I_ans` 的配对 example-to-answer 数据。当前 C6 COCO/SPRIGHT manifest 只有 `prompt + real image`，没有独立示例图，只能定义为：

```text
proxy mode:
  I_ex  = absent
  I_ans = current real image
```

不能静默设置 `I_ex = I_ans` 并宣称完成了 example-to-answer 推理，因为这会产生答案图泄漏。若正式数据缺少独立示例图，报告必须标为 text-to-answer-image proxy。

## 3. 条件序列合同

G 路条件建议为：

```text
[prompt text]
[example image context tokens]
[target image noisy latent tokens x_t]
```

U 路 teacher/student 使用相同上下文，只有候选答案图不同：

```text
Teacher U:
  [prompt] [example image] [answer image] [question prefix]

Student U:
  [prompt] [example image] [predicted image] [question prefix]
```

U hidden 和 logits 必须在 answer token 输入前截取，避免 teacher-forcing answer leakage。U validity 由 next-token logits 判断，而不是先把正确 `Yes/No` token 放入上下文再提取 hidden。

实现前必须验证 Show-o2 的 `modality_positions`、attention mask 和 sequence length 能稳定支持 example/candidate 双图。如果不支持，先实现和测试多图上下文接口，不能把两张图拼成像素大图代替。

## 4. 冻结和可训练边界

冻结：

```text
Show-o2 backbone
U route / understanding adapter
原 G route / generation adapter
relation encoder
diff_proj 和原 diffusion head
Wan VAE
Mask2Former
```

训练：

```text
drsa_adapter.generation_spatial
必要时的 G-side 小型 conditioning module
```

第一版不引入独立大型 projector，避免 loss 只训练一个可丢弃的 readout。若后续加入 projector，必须有 projector-only control，并记录 projector/G-spatial 各自 update norm。

冻结审计：

1. 训练前保存所有 frozen tensor SHA256 或逐 tensor digest。
2. 每个 checkpoint 检查 frozen parameter `.grad is None`。
3. 训练后逐 tensor 比较 U、原 G、relation encoder、backbone、VAE 和 detector。
4. 任一 frozen tensor 变化则 checkpoint 无效。

## 5. 主数据流

```text
                             answer image I_ans
                           /                    \
                    frozen VAE E             frozen Mask2Former
                         |                           |
                   clean latent x_ans         teacher mask/logits
                         |                           |
        noise eps -------+                           |
             |                                       |
             v                                       |
  x_t=(1-t)eps+t*x_ans                                |
             |                                       |
 prompt + example image                              |
             |                                       |
             v                                       |
 frozen backbone + G + trainable generation_spatial |
             |                                       |
        velocity v_G                                 |
             |                                       |
  x_hat_ans=x_t+(1-t)v_G                             |
             |                                       |
      frozen VAE decode                              |
             |                                       |
       predicted image I_pred -----------------> frozen Mask2Former
             |                                       |
      frozen VAE encode                              |
             |                                       |
       x_U_pred                                      |
             |                                       |
 prompt + example image + x_U_pred                   |
             |                                       |
         frozen U route                              |
             |                                       |
    answer logits / U hidden                         |
             \---------------- losses ---------------/
                              |
                              v
                    generation_spatial only
```

## 6. 图像回路与梯度边界

答案 latent：

\[
x_{ans}=E_{VAE}(I_{ans})
\]

线性 flow：

\[
x_t=(1-t)\epsilon+t x_{ans}
\]

\[
u^*=x_{ans}-\epsilon
\]

G 预测 clean latent：

\[
\hat x_{ans}=x_t+(1-t)v_G(x_t,t,p,I_{ex})
\]

实际图像回路：

\[
I_{pred}=D_{VAE}(\hat x_{ans})
\]

\[
x_U^{pred}=E_{VAE}^{mean}(I_{pred})
\]

`E_VAE^mean` 使用确定性 posterior mean，不允许在 U-cycle 中重新采样噪声。Teacher answer image 也必须走相同 deterministic U image preprocessing，避免 teacher/student 预处理不一致。

梯度规则：

```text
Teacher U(answer image): torch.no_grad() + stop-gradient
Student U(predicted image): U params frozen，但 forward 不使用 torch.no_grad()
VAE decode/encode: 参数冻结，但保留对输入的 autograd
```

因此：

```text
L_U
-> frozen U input gradient
-> frozen VAE encoder
-> predicted image
-> frozen VAE decoder
-> predicted clean latent
-> G velocity
-> generation_spatial
```

## 7. Teacher Validity Gate

Gate 只基于冻结 teacher 和答案图，训练前离线预计算；不能依赖不断变化的预测图。

### 7.1 U teacher gate

对答案图构造：

```text
q1: A r B?              target Yes
q2: A inverse(r) B?     target No
q3: B inverse(r) A?     target Yes
```

严格 gate：

\[
g_U=1[q1,q2,q3\ correct\ \land\ confidence\ margin>\delta_U]
\]

可记录连续置信权重，但主实验先使用二值 gate，避免低置信 teacher 主导。

### 7.2 Detector gate

\[
g_D=1[A,B\ both\ detected\ \land\ score_A,score_B>\tau_D]
\]

关系 teacher gate：

\[
g_R=g_D\cdot 1[r_{teacher}>\tau_R]
\]

### 7.3 Joint gate 用法

| Loss | Gate |
|---|---|
| Mask BCE/Dice/Presence | `g_D` |
| Relation distillation | `g_R` |
| Absolute relation pass | `g_D`，目标来自 manifest |
| U answer CE/KD | `g_U` |
| U feature/grounding alignment | `g_U * g_D` |
| Flow/G retention | 无 gate |

必须报告 gate coverage 的方向、类别和数据源分布。所有 U-cycle 主消融使用同一 gated subset，并增加 `same gated data, U loss=0` 控制组，以区分数据过滤收益和 U teacher 收益。

## 8. Mask 与对象空间监督

Teacher：答案图冻结 Mask2Former 输出。Student：预测图冻结 Mask2Former 输出。

保留 class-conditioned soft query aggregation，解决 Mask2Former query 无序问题。

Mask objective：

\[
L_{mask}=0.01L_{BCE}+0.07L_{Dice}+0.16L_{presence}
+0.50L_{rel-distill}+1.00L_{rel-pass}
\]

第一版继续令 `L_object-detect` 权重为 0，避免同时改变过多变量；后续单独消融。

与旧 `original_mask_logits` 不同，teacher-dependent loss 必须启用 validity gate。答案图的 COCO box 只用于数据/关系审计；像素监督仍来自 frozen detector soft masks。

## 9. Flow 与 G 路保留

Flow：

\[
L_{flow}=MSE(v_G,u^*)
\]

保留当前 B1-G teacher retention：同一 `original_caption/example context/x_t/t`，关闭 `generation_spatial` 得到 frozen G teacher：

\[
L_{G-ret}=MSE(v_{G+spatial},stopgrad(v_{B1-G}))
\]

Flow 负责答案 latent 轨迹，G retention 防止新增 U-cycle 只优化 U judge 而破坏原生成分布。二者不能被 U loss 替代。

## 10. 冻结 U 的答案行为监督

Teacher logits：

\[
\ell_U^{ans}=U(p,I_{ex},I_{ans})
\]

Student logits：

\[
\ell_U^{pred}=U(p,I_{ex},I_{pred})
\]

Ground-truth answer loss：

\[
L_{U-task}=g_U\cdot CE(\ell_U^{pred},y)
\]

Teacher distribution distillation：

\[
L_{U-KD}=g_U\cdot T^2 KL\left(
stopgrad\left[softmax(\ell_U^{ans}/T)\right]
\parallel softmax(\ell_U^{pred}/T)
\right)
\]

正、反、交换问题均计算，防止模型只生成模糊布局或 U 路恒答 `Yes`。

这两个 loss 的区别：

```text
U-task：答案图的结构化 ground truth
U-KD：答案图在冻结 U 中的完整置信分布和暗知识
```

## 11. U-space Perceptual 与 Grounding 监督

不直接比较 G hidden 与 U hidden。答案图和预测图均经过同一个 frozen U，再在 U 空间比较。

选定 U 空间能力最强的 Transformer layer `l*`，提取 image-token states：

\[
H_U^{ans}=U_{l*}(p,I_{ex},I_{ans})
\]

\[
H_U^{pred}=U_{l*}(p,I_{ex},I_{pred})
\]

使用答案图 teacher mask 下采样到 U token grid，进行对象池化：

\[
z_k^{ans}=MaskPool(H_U^{ans},M_k^{ans})
\]

\[
z_k^{pred}=MaskPool(H_U^{pred},M_k^{ans})
\]

对象 perceptual loss：

\[
L_{U-feat}=g_Ug_D\frac12\sum_{k\in\{A,B\}}[1-cos(z_k^{pred},stopgrad(z_k^{ans}))]
\]

图文 grounding map 由同一 U layer 的对象词 token 与 image tokens 构造：

\[
G_k(i)=softmax_i(cos(t_k,H_U(i))/\tau)
\]

\[
L_{U-ground}=g_Ug_D\sum_k KL(stopgrad(G_k^{ans})\parallel G_k^{pred})
\]

因为两边都在 U 空间、同一 `27x27` 坐标网格，不存在 U/G embedding 坐标系不一致的问题。

## 12. 第一版总目标

Base：

\[
L_{base}=L_{mask}+\lambda_fL_{flow}+\lambda_rL_{G-ret}
\]

One-step U-cycle：

\[
L_{cycle}=\lambda_tL_{U-task}+\lambda_kL_{U-KD}
+\lambda_hL_{U-feat}+\lambda_gL_{U-ground}
\]

总目标：

\[
L_{total}=L_{base}+w_{cycle}(s)L_{cycle}
\]

`w_cycle(s)` 从 0 线性 ramp。新增权重不预先拍定，先在固定 calibration batch 上测 `generation_spatial` 的 component gradient norm，使每个 U 分量初始梯度处于 Mask/Flow 主梯度的约 `10%-30%`，再冻结权重进入正式消融。

每步记录各 loss 对 `generation_spatial` 的 gradient norm 和 gradient cosine。只有观察到稳定冲突后才引入 PCGrad，第一版不提前增加优化器复杂度。

## 13. 单步到多步的递进路线

### Stage 0：Teacher audit/cache

```text
冻结 U/Mask2Former
运行答案图 q1/q2/q3
计算 g_U/g_D/g_R 和置信度
检查方向/类别覆盖
缓存 answer logits、选定 U object features、grounding maps
```

### Stage 1：Base reproduction

```text
Mask + Flow + G retention
使用 gated subset，但 U loss=0
同 seed/data order 复现基座
```

### Stage 2：One-step answer behavior cycle

```text
Stage 1 + U-task + U-KD
x_hat_ans 单步预测 -> decode -> deterministic re-encode -> frozen U
```

### Stage 3：One-step U perceptual cycle

```text
Stage 2 + U-feat + U-ground
只在 joint-valid 样本计算
```

### Stage 4：Differentiable short rollout

从纯噪声使用相同 Euler/CFG 协议做 `K=4` 或 `K=8` 步可微 rollout，最终图进入 U-cycle。为控制显存：

```text
每 4 个训练 step 运行一次 rollout branch
gradient checkpoint sampler/model blocks
先只反传最后 1-2 个 Euler step（truncated BPTT）
其余 step stop-gradient
```

### Stage 5：Sampler binding

只有 Stage 4 明确优于 one-step，才扩大可反传 sampler step 数。第一版不做 50-step 全反传；50-step 仅用于正式评测。

## 14. 主消融矩阵

| ID | Box | Mask | Flow | G-ret | Gate-filter data | U-task | U-KD | U-feat/ground | Short rollout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 B1 |  |  |  |  |  |  |  |  |  |
| A1 Box negative | yes |  | yes | optional |  |  |  |  |  |
| A2 Mask-only |  | yes |  |  |  |  |  |  |  |
| A3 Mask+Flow |  | yes | yes |  |  |  |  |  |  |
| A4 Current base |  | yes | yes | yes |  |  |  |  |  |
| A5 Gate data control |  | yes | yes | yes | yes |  |  |  |  |
| A6 Answer CE |  | yes | yes | yes | yes | yes |  |  |  |
| A7 Answer KD |  | yes | yes | yes | yes |  | yes |  |  |
| A8 Behavior cycle |  | yes | yes | yes | yes | yes | yes |  |  |
| A9 Full one-step |  | yes | yes | yes | yes | yes | yes | yes |  |
| A10 Short rollout |  | yes | yes | yes | yes | yes | yes | yes | yes |

A5 是必要控制：A6-A10 的任何提升必须先排除 gate 数据筛选本身的影响。

## 15. 机制消融

| 消融 | 研究问题 |
|---|---|
| 无 U gate（短程诊断） | 错误 teacher 是否导致负迁移 |
| positive-only | inverse/swap 是否防止关系塌缩 |
| CE-only vs KD-only | ground truth 和 teacher dark knowledge 的贡献 |
| 无 U feature | 行为监督是否足够 |
| 无 grounding | 对象词-区域迁移是否必要 |
| latent-direct U vs decode-reencode U | 实际图像回路是否优于 latent proxy |
| 无 example image | 增益来自示例条件还是答案图 teacher |
| one-step vs 4-step vs 8-step | sampler gap 的变化 |
| U layer probe | 哪层空间理解最适合作 teacher |
| object-detect loss 0 vs enabled | Mask presence 额外项是否必要 |

无 gate 和 U-unfrozen 不进入正式长训练；只允许有限 smoke 诊断。主方法始终冻结 U。

## 16. 公平训练协议

所有主消融固定：

```text
相同初始化 checkpoint
相同 gated manifest 和样本顺序
相同 noise/timestep seed
相同 optimizer step 数和 LR schedule
相同可训练参数（除明确模块消融）
相同 checkpoint：500/1000/2500/5000
相同 GenEval/Self-cycle 采样 seed
```

不能再用 500-step control 对比 5000-step candidate 做 loss 因果结论。

## 17. 训练诊断

必须记录：

```text
g_U/g_D/g_R coverage 和按方向分解
U(real answer) q1/q2/q3 accuracy/confidence
U(predicted image) q1/q2/q3 accuracy/confidence
Mask/Flow/G-ret/U-task/U-KD/U-feat/U-ground raw/weighted loss
每项对 generation_spatial 的 gradient norm/cosine
VAE decode-reencode latent drift
one-step/rollout sampler step 和显存/耗时
generation_spatial update norm
所有 frozen parameter audit
```

训练 proxy 下降不能作为 checkpoint 选择依据。

## 18. 评测与预注册 Gate

正式评测：

```text
GenEval Position-72：correct、both objects、per-direction、paired gain/loss/net
Self-cycle 四 seed：overall、2D、3D、Counting、Multi-object、paired 95% CI
Frozen U VSR：必须不变
内部 cycle：答案图/预测图 q1/q2/q3、KD、object feature/grounding
独立质量：ImageReward、CLIP counterfactual margin 或独立 VLM judge
```

建议预注册通过条件：

1. Frozen U/backbone/VAE/detector tensor 完全不变。
2. GenEval paired gain > loss，both-object 不低于 A5。
3. Self-cycle `2D+3D` 至少 3/4 seed 提升。
4. Self-cycle overall 不低于预设 non-inferiority margin。
5. 独立 judge 不退化，防止模型只优化 frozen U 或 Mask2Former。
6. U-cycle 提升必须在 A5 同数据控制之上成立。

## 19. 关键风险与防护

### U critic exploitation

G 可能生成 U 易判断但人类不可接受的纹理。保留 Flow、Mask、G-retention，并使用独立 judge；不能只看 U loss。

### Teacher/evaluator leakage

U 既训练监督又用于内部 cycle 指标，Mask2Former 既训练又用于 GenEval detector。最终结论必须依赖独立 VLM/ImageReward/外部空间集。

### 一图多解

答案图是一个有效目标，不是 prompt 的唯一正确像素结果。避免强像素 L1；以 Flow、Mask、U 行为和对象 perceptual 为主。

### Gate selection bias

严格 gate 可能过滤难例或某些方向。必须使用 A5 控制、方向重采样和覆盖报告。

### Decode-reencode gap

Wan VAE decode/encode 会引入误差。记录 `E(D(x_ans))-x_ans`，并用 latent-direct 作为机制消融，但正式主方法坚持实际图像回路。

### 多步显存

短 rollout 使用周期执行、checkpoint 和 truncated BPTT。不能为了省显存静默 detach 最后图像，导致 U loss 无法回传。

### 示例图/答案图泄漏

正式 paired 数据必须审计 `I_ex` 和 `I_ans` 是否重复、近重复或来自同一视频相邻帧。若重复，单独报告并划分去重 split。

## 20. 实现前置决策

进入编码前必须完成：

1. 明确正式 example-to-answer 数据来源与许可，或明确第一阶段仅是无示例图 proxy。
2. 固定 prompt、example image、candidate image 和 question prefix 的 token/attention 模板。
3. 在冻结 U 上完成 layer probe，并预注册 `l*`。
4. 完成 Teacher Validity Gate cache 和方向覆盖审计。
5. 确认双图 U/G sequence 在 Show-o2 中可运行且无答案泄漏。
6. 完成 decode-reencode 的梯度 smoke 和 frozen-parameter audit。
7. 固定 A5-A10 的相同 manifest、sample order、noise/timestep schedule。

在这些条件完成前，不启动正式训练。

## 21. 研究表述

主方法可命名为：

```text
Frozen-Understanding Answer-Supervised Generation Cycle
```

或：

```text
Understanding-Guided Mask-Flow Alignment for Answer-Image Generation
```

核心贡献表述：

> 将具有更强空间理解能力的冻结 U 路作为答案图 teacher，通过实际图像域的可微生成回路，把答案行为、对象表示和图文 grounding 监督反传到 G 路；同时用 class-conditioned Mask2Former loss 保证对象证据，用 Flow 和 B1-G retention 保持生成动力学。

