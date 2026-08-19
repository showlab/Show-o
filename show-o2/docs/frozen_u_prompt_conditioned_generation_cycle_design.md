# Prompt-Conditioned Frozen-U Generation Cycle 设计

更新时间：2026-08-11 UTC

状态：canonical 设计；Stage 2 gated 四方向 frozen-U 回路已实现并通过梯度与 mixed-loss smoke，正式长训尚未启动。详见 `docs/frozen_u_prompt_conditioned_generation_cycle_smoke_20260811.md`。

## 1. 修正后的任务边界

本设计不要求输入中包含示例图，也不把任务定义为 image-to-image generation。

正式训练合同是：

```text
生成条件：prompt only
训练目标：target / answer image
模型输出：predicted image
内部监督：prompt 语义与 frozen-U 对 predicted image 的图文理解一致
```

答案图不是 G 的输入条件，只在训练时提供：

```text
Flow target
Mask2Former teacher
Frozen-U Teacher Validity Gate
可选的 answer-image U distillation/perceptual target
```

核心研究假设：

> Prompt 明确给出了对象和空间关系；冻结 U 路对预测图执行图文理解，并将其正反关系答案、对象 grounding 和空间语义梯度回传给 G，使 U 的空间理解能力约束 G 的生成结果。

## 2. 重要架构事实

纯文本 prompt 不会经过 `drsa_adapter.understanding`，因为 U adapter 只作用于 image tokens。文本单独输入时经过的是冻结的 tokenizer、embedding 和共享 Qwen text path。

因此本文区分：

```text
T_U(prompt)：冻结的 prompt/text semantic branch
U(prompt, image)：真正的冻结图文理解 route
```

不能把 `T_U(prompt)` 称为已经执行了图像空间理解，也不能直接对 `T_U(prompt)` 和 multimodal image hidden 做 raw cosine 后宣称完成 U/G 对齐。

第一版主监督必须是功能性的：

```text
prompt 派生目标 relation / positive-inverse-swap questions
predicted image 进入 frozen U
frozen U 必须给出与 prompt 一致的答案
```

Prompt/text hidden alignment 只作为辅助 semantic target，并使用冻结 relation decoder 或 prototype canonical space，不能比较任意 raw embedding。

## 3. 数据合同

正式样本：

```text
D_i = {
  prompt:       p_i,
  answer_image: I_i^ans,
  answer:       y_i,
  objects:      [A_i, B_i],
  relation:     r_i,
  inverse:      inverse(r_i),
  metadata:     source / split / ids
}
```

字段用途：

| 字段 | 用途 |
|---|---|
| `prompt` | G 的唯一语义条件；构造 U 正反空间问题和 prompt semantic target |
| `answer_image` | Flow target、Mask teacher、U validity teacher；绝不作为 G condition |
| `answer` | 冻结 U 对预测图应输出的 ground-truth answer |
| `objects/relation` | class-conditioned masks、关系问题、方向评测 |

当前 C6 COCO/SPRIGHT 的 `prompt + real image` 可直接满足这一合同：

```text
p_i       = normalized spatial prompt
I_i^ans   = paired real COCO image
y_i       = relation-derived structured answer
```

不再需要独立 example-image 数据，也不允许把答案图 token 混入 G prompt context。

## 4. 三条冻结 U 分支

### 4.1 Prompt semantic branch

```text
prompt p
-> frozen tokenizer/embedding/shared Qwen text path
-> object phrase / relation text representations
-> frozen relation semantic decoder or relation prototypes
-> prompt target distribution pi_prompt
```

该分支不包含图像，不激活 understanding adapter。它只定义“prompt 要求什么”。

### 4.2 Answer-image validity branch

```text
prompt + answer image
-> frozen U
-> positive / inverse / swap relation logits
-> g_U validity and confidence
```

该分支使用 `torch.no_grad()`，只决定 U 是否有资格成为当前样本的 teacher。可选地缓存 answer-image U logits/features，供机制消融使用。

### 4.3 Predicted-image student branch

```text
prompt + predicted image
-> frozen U
-> answer logits / image hidden / text-image grounding
-> U-cycle losses
-> gradient 回到 predicted image 和 G
```

U 参数全部冻结，但该 forward 不能使用 `torch.no_grad()`。

## 5. 总数据流

```text
                                      answer image I_ans
                                  /          |           \
                           frozen VAE     Mask2Former    frozen U
                               |              |             |
                         clean latent x_ans  mask teacher  g_U validity
                               |
noise eps ---------------------+
        |
        v
x_t=(1-t)eps+t*x_ans
        |
prompt only
        |
frozen backbone + G + trainable generation_spatial
        |
velocity v_G
        |
x_hat_ans=x_t+(1-t)v_G
        |
frozen VAE decode
        |
predicted image I_pred ----------------------> frozen Mask2Former
        |
frozen deterministic VAE encode
        |
prompt + predicted-image latent
        |
frozen U, input gradient enabled
        |
answer logits / U hidden / grounding maps
        |
Mask + Flow + G-retention + Prompt/U-cycle losses
        |
generation_spatial only
```

## 6. Flow 与实际图像回路

答案 latent：

\[
x_{ans}=E_{VAE}^{mean}(I_{ans})
\]

线性 flow：

\[
x_t=(1-t)\epsilon+t x_{ans},\qquad u^*=x_{ans}-\epsilon
\]

G 预测：

\[
v_G=G(x_t,t,p)
\]

\[
\hat x_{ans}=x_t+(1-t)v_G
\]

实际图像域回路：

\[
I_{pred}=D_{VAE}(\hat x_{ans})
\]

\[
x_U^{pred}=E_{VAE}^{mean}(I_{pred})
\]

U student 使用 `x_U^pred`，不是直接使用 `x_hat_ans`。这样训练路径与实际“生图后再理解”一致。`latent-direct U(x_hat_ans)` 只作为机制消融。

Wan VAE 参数冻结，但 decode 和 predicted-image encode 均保留对输入的 autograd。

## 7. 冻结边界

冻结：

```text
Show-o2 backbone
U route / understanding adapter
原 G route / generation adapter
relation encoder
diff_proj / official diffusion head
Wan VAE
Mask2Former
prompt semantic decoder / relation prototypes
```

训练：

```text
drsa_adapter.generation_spatial
必要时的小型 G-side residual conditioning module
```

训练前后必须逐 tensor 审计所有 frozen state；任何 U/backbone/VAE/detector tensor 变化都使 checkpoint 无效。

## 8. Teacher Validity Gate

Gate 只基于答案图和冻结 teacher，训练前离线计算，不依赖预测图。

对答案图构造：

```text
q1: A r B?              -> Yes
q2: A inverse(r) B?     -> No
q3: B inverse(r) A?     -> Yes
```

最初预注册的 q1/q2/q3 yes/no gate 在实现后审计为 0/16 strict valid，主要表现为三问输出常量 yes，不能作为主训练 critic。当前 Stage 2 改用固定语义坐标的四方向 multiple-choice readout：

```text
Where is A relative to B?
A: left, B: right, C: above, D: below
```

答案图 gate 为：

\[
g_U=1[argmax(\ell_U^{ans})=r_i\land margin(\ell_U^{ans},r_i)>\delta_U]
\]

完整独立 validation 1024 张上，DRSA-U readout accuracy 为 `42.58%`，positive-margin coverage 为 `41.21%`；yes/no triples 只保留为机制负结果。

Detector gate：

\[
g_D=1[A,B\ both\ detected\land score_A,score_B>\tau_D]
\]

Relation detector gate：

\[
g_R=g_D\cdot1[r_{teacher}>\tau_R]
\]

Frozen-U grounding readout gate：

\[
g_{Ground}=g_Ug_D\cdot1[IoU(G_U^{ans},M^{M2F})>\tau_G]
\]

Loss gate：

| Loss | Gate |
|---|---|
| Mask BCE/Dice/Presence | `g_D` |
| Relation distillation | `g_R` |
| Absolute relation pass | `g_D` |
| Prompt/U answer losses | `g_U` |
| U grounding loss | `g_Ground` |
| Flow/G-retention | 无 gate |

必须记录 gate coverage 的方向、类别和来源分布。所有 U-cycle 消融使用相同 gated subset，并设置 `same gated data, U loss=0` 控制组。

## 9. Mask 对齐

Teacher：答案图冻结 Mask2Former。Student：预测图冻结 Mask2Former。

保留 class-conditioned soft query aggregation：

\[
L_{mask}=0.01L_{BCE}+0.07L_{Dice}+0.16L_{presence}
+0.50L_{rel-distill}+1.00L_{rel-pass}
\]

第一版继续令 `L_object-detect=0`，后续单独消融。与旧 `original_mask_logits` 不同，teacher-dependent loss 必须启用 validity gate。

## 10. Flow 与 G 保留

\[
L_{flow}=MSE(v_G,u^*)
\]

同一 `original_caption/x_t/t` 下，关闭 `generation_spatial` 得到 frozen B1-G teacher：

\[
L_{G-ret}=MSE(v_{G+spatial},stopgrad(v_{B1-G}))
\]

Flow 和 G-retention 防止 G 通过生成 U 容易判断但偏离真实生成轨迹的图案来降低 U-cycle loss。

## 11. Prompt 派生的功能性 U 监督

Prompt 解析得到目标关系 `r`。预测图进入冻结 U，并使用固定 `left/right/above/below` 语义坐标的四方向 candidate logits：

\[
\ell_U^{pred}=U(p,I_{pred},q_{4way})
\]

主任务 loss：

\[
L_{U-task}=g_U\cdot CE(\ell_U^{pred},r)
\]

`g_U` 只由答案图 frozen-U 四方向 readout 的正确性和 target margin 决定。该 loss 不依赖 answer-image feature matching，也不把答案图输入 G。原 positive/inverse/swap yes/no CE 因 teacher 对反事实问题输出近常量答案，不进入正式主目标。

## 12. Prompt Semantic Distribution 对齐

从 prompt-only frozen text branch 提取对象词和关系表示，经预训练后冻结的 relation semantic decoder 得到：

\[
\pi_{prompt}=softmax(C_{rel}^{frozen}(T_U(p))/T)
\]

预测图分支使用同一冻结 relation decoder 对 U multimodal relation representation 解码：

\[
\pi_{pred}=softmax(C_{rel}^{frozen}(H_U(p,I_{pred}))/T)
\]

Prompt semantic KD：

\[
L_{U-prompt}=g_U\cdot T^2KL(stopgrad(\pi_{prompt})\parallel\pi_{pred})
\]

不能直接比较 raw prompt hidden 与 image hidden。`C_rel^frozen` 必须预先在冻结 U/text features 上验证并冻结，且解码器输出有固定 `left/right/above/below` 语义坐标。

Prompt branch 明确包含目标关系，可能成为简单 target encoder，因此 `L_U-prompt` 只作为辅助；主证据仍来自预测图 U answer behavior。

## 13. Frozen-U Grounding Readout

Raw U hidden 是 `[729,1536]` 表示，Mask2Former teacher 是二维概率 mask；两者不在同一个表示空间，禁止直接做 MSE、cosine、BCE 或 KL：

```text
错误：H_U <-> Mask2Former mask
正确：frozen readout(H_U) -> U grounding map <-> U grounding map
```

先在独立 calibration split 上运行答案图：

\[
H_U^{ans}=U_{l*}(p,I_{ans})
\]

\[
M_k^{M2F}=Mask2Former(I_{ans},class=k)
\]

训练一个低容量 U grounding readout `R_U`，输入对象词 token 与 U image tokens，输出同一 `27x27` 网格上的 logits：

\[
G_{U,k}^{ans}=softmax(R_U(t_k^{ans},H_{U,img}^{ans})/\tau)
\]

`R_U` 的 calibration loss 使用下采样 Mask2Former mask 的 BCE/Dice；U 始终冻结。`R_U` 必须在独立 validation split 上达到预注册的 mask IoU、centroid 和方向阈值，随后完全冻结。

Grounding readout validity：

\[
g_{Ground}=g_Ug_D\cdot1[IoU(G_{U,k}^{ans},M_k^{M2F})>\tau_G\ \forall k]
\]

G 训练时，答案图和预测图都经过同一个 frozen U 和同一个 frozen `R_U`：

\[
G_{U,k}^{ans}=R_U(U(p,I_{ans}),k)
\]

\[
G_{U,k}^{pred}=R_U(U(p,I_{pred}),k)
\]

主 grounding loss 在同一个 U-readout 空间中计算：

\[
L_{U-ground}=g_{Ground}\sum_kKL(stopgrad(G_{U,k}^{ans})\parallel G_{U,k}^{pred})
\]

Mask2Former 在主 loss 中只负责校准 `R_U` 和决定 `g_Ground`，不与 raw U hidden 直接比较。梯度通过 frozen `R_U`、frozen U、VAE encode/decode 回到 G。

可选的跨模型空间 target 仅作为消融：

\[
L_{cross-ground}=g_{Ground}\sum_kKL(normalize(M_k^{M2F})\parallel G_{U,k}^{pred})
\]

只有 `L_cross-ground` 明确标为 Mask-to-U-readout auxiliary；它不能被描述成 hidden alignment，也不进入第一版主目标。

## 14. 可选的 Answer-Image U Distillation

答案图仍然可作为额外 teacher，但不属于 G 输入：

\[
\ell_U^{ans}=U(p,I_{ans},q)
\]

\[
L_{U-answer-KD}=g_U\cdot T^2KL(
stopgrad[softmax(\ell_U^{ans}/T)]
\parallel softmax(\ell_U^{pred}/T))
\]

同一 U 空间中的对象 perceptual：

\[
L_{U-feat}=g_Ug_D\frac12\sum_{k\in\{A,B\}}[
1-cos(z_k^{pred},stopgrad(z_k^{ans}))]
\]

这两项用于回答：ground-truth relation 行为监督是否已经足够，还是需要答案图在 U 中的更细暗知识。

## 15. 第一版总目标

Base：

\[
L_{base}=L_{mask}+\lambda_fL_{flow}+\lambda_rL_{G-ret}
\]

Prompt-conditioned U-cycle：

\[
L_{cycle}=\lambda_tL_{U-task}+\lambda_pL_{U-prompt}
+\lambda_gL_{U-ground}
\]

第一版：

\[
L_{total}=L_{base}+w_{cycle}(s)L_{cycle}
\]

Answer-image U-KD/feature loss 暂不进入第一版主目标，只在主回路有效后追加消融。

`w_cycle(s)` 从 0 线性 ramp。新增权重先按 `generation_spatial` component gradient norm 校准，使单个 U loss 初始梯度约为 Mask/Flow 主梯度的 `10%-30%`，固定后再进入正式对比。

## 16. 单步到多步路线

### Stage 0：Prompt/U teacher audit

```text
构造 prompt relation target 和 q1/q2/q3
在答案图上计算 g_U/g_D/g_R
训练并冻结 relation semantic decoder
在独立 split 上训练、验证并冻结 U grounding readout
计算 g_Ground，剔除 readout 无法从 U hidden 可靠还原空间区域的样本
检查 prompt-only target 与答案图 U behavior 一致性
缓存 gate、answer masks、可选 answer U logits/features
```

### Stage 1：Base reproduction

```text
Mask + Flow + G-retention
使用相同 gated subset，U-cycle weight=0
```

### Stage 2：One-step prompt behavior cycle

```text
Stage 1 + L_U-task
predicted clean -> decode -> deterministic encode -> frozen U
```

### Stage 3：Prompt semantic/grounding cycle

```text
Stage 2 + L_U-prompt + L_U-ground
```

### Stage 4：Answer-image U distillation ablation

```text
Stage 3 + L_U-answer-KD / L_U-feat / L_cross-ground
```

### Stage 5：Differentiable short rollout

从纯噪声使用 `K=4/8` Euler rollout，最终图进入 frozen U。先每 4 个 optimizer step 执行一次，只反传最后 1-2 个 sampler steps，并使用 gradient checkpointing。

50-step sampler 第一阶段仅用于评测，不做完整反传。

## 17. 主消融矩阵

| ID | Box | Mask | Flow | G-ret | Gate data | U-task | Prompt KD | Grounding | Answer-U KD/feat | Rollout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 B1 |  |  |  |  |  |  |  |  |  |  |
| A1 Box negative | yes |  | yes | optional |  |  |  |  |  |  |
| A2 Mask-only |  | yes |  |  |  |  |  |  |  |  |
| A3 Mask+Flow |  | yes | yes |  |  |  |  |  |  |  |
| A4 Current base |  | yes | yes | yes |  |  |  |  |  |  |
| A5 Gate data control |  | yes | yes | yes | yes |  |  |  |  |  |
| A6 Prompt behavior |  | yes | yes | yes | yes | yes |  |  |  |  |
| A7 Prompt semantic |  | yes | yes | yes | yes | yes | yes |  |  |  |
| A8 Prompt grounding |  | yes | yes | yes | yes | yes | yes | yes |  |  |
| A9 Answer-U distill |  | yes | yes | yes | yes | yes | yes | yes | yes |  |
| A10 Short rollout |  | yes | yes | yes | yes | yes | yes | yes | selected | yes |

A5 是必要数据控制。A6 是最小新方法；A8 是第一版完整 prompt-conditioned U-cycle；A9 检验答案图 U 暗知识是否额外有用。

## 18. 机制消融

| 消融 | 研究问题 |
|---|---|
| 无 U gate（短 smoke） | 错误 U teacher 是否导致负迁移 |
| positive-only | inverse/swap 是否防止关系答案塌缩 |
| U-task only | 功能性关系监督是否足够 |
| Prompt KD only | text target 是否只是复述 prompt |
| 无 grounding | 对象词-图像区域监督是否必要 |
| Answer-U KD/feat on/off | 答案图暗知识是否提供额外信息 |
| latent-direct vs decode-reencode | 实际生图回路是否必要 |
| frozen readout vs trainable readout | 可训练 readout 是否吸收了 G 本应学习的监督 |
| U-to-U grounding vs direct Mask-to-U | 同空间蒸馏是否优于跨模型空间 target |
| one-step vs 4-step vs 8-step | sampler gap 的影响 |
| object-detect 0 vs enabled | presence loss 的额外价值 |
| relation decoder frozen vs trainable | 可训练 decoder 是否吸收全部监督 |

主方法始终冻结 U。无 gate 和 U-unfrozen 不进入正式长训练。

## 19. 公平协议与训练诊断

所有主组固定：

```text
相同初始化 checkpoint
相同 gated manifest / sample order
相同 noise / timestep seed
相同 optimizer steps / LR schedule
相同 checkpoint：500/1000/2500/5000
相同 GenEval / Self-cycle generation seeds
```

记录：

```text
g_U/g_D/g_R coverage 和方向/类别分解
U(answer image) q1/q2/q3 accuracy/confidence
U(predicted image) q1/q2/q3 accuracy/confidence
prompt target vs predicted U distribution KL
prompt-object grounding vs answer masks
所有 raw/weighted loss、gradient norm/cosine
decode-reencode latent drift
sampler steps、显存、耗时
generation_spatial update norm
frozen tensor audit
```

## 20. 评测与预注册 Gate

正式评测：

```text
GenEval Position-72：correct / both objects / per-direction / paired gain-loss-net
Self-cycle 四 seed：overall / 2D / 3D / Counting / Multi-object / paired 95% CI
Frozen U VSR：必须不变
内部 cycle：predicted-image q1/q2/q3、prompt KL、grounding
独立 judge：ImageReward、counterfactual CLIP/VLM、外部空间集
```

候选通过条件：

1. U/backbone/VAE/detector frozen tensor 完全不变。
2. GenEval paired gain > loss，both-object 不低于 A5。
3. Self-cycle `2D+3D` 至少 3/4 seed 提升。
4. Self-cycle overall 不低于预设 non-inferiority margin。
5. 独立 judge 不退化，防止 G 只优化 frozen U 或 Mask2Former。
6. A6-A10 的收益必须高于 A5 gated-data control。

## 21. 风险与防护

### Prompt shortcut

Prompt 明确包含 relation，prompt semantic decoder 很容易输出正确标签。主证据必须来自 `U(prompt,predicted image)` 的正反行为和 grounding，不能只看 Prompt KD。

### U critic exploitation

G 可能生成 U 易判断但人类不可接受的纹理。保留 Flow、Mask、G-retention，并使用独立 judge。

### Teacher/evaluator leakage

U 和 Mask2Former 都进入训练。最终结论不能只用 U-cycle 或同一 detector；必须报告独立 VLM/ImageReward/外部空间结果。

### Answer target leakage

答案图只允许进入 Flow target、Mask teacher、U validity/可选 distillation。任何 answer-image token、embedding 或 cache key 泄漏到 G condition 都使实验无效。

### 一图多解

答案图只是 prompt 的一个有效目标。避免强 pixel L1；主监督使用 Flow、Mask、U behavior 和 grounding。

### Gate bias

Gate 可能过滤难例或某些方向。必须有 A5 控制、方向重采样和 coverage 报告。

### Decode-reencode 与多步开销

记录 VAE round-trip drift；短 rollout 使用周期执行、checkpoint 和 truncated BPTT，不能静默 detach 最终图像。

## 22. 实现前置条件

1. 固定 prompt relation parser、对象 spans 和 q1/q2/q3 模板。
2. 验证 prompt-only branch 不被误称为 understanding adapter output。
3. 在答案图上完成四方向 U Teacher Validity cache、margin 和按方向覆盖审计。
4. 训练、验证并冻结 relation semantic decoder，检查其无图像时只作为 target encoder。
5. 在独立 calibration/validation split 上训练并冻结 U grounding readout，预注册 IoU/centroid gate。
6. 完成 predicted image decode-reencode 到 frozen U/readout 的梯度 smoke。
7. 完成 U/backbone/VAE/Mask2Former/readout frozen audit。
8. 固定 A5-A10 的 manifest、sample order 和 timestep/noise schedule。

在这些条件完成前，不启动正式训练。

## 23. 研究表述

方法名建议：

```text
Prompt-Conditioned Frozen-Understanding Generation Cycle
```

核心表述：

> 以 prompt 作为 G 的唯一生成条件，将预测图回传到冻结 U 路，通过 prompt 派生的正反空间问答、冻结语义分布和对象 grounding 对 G 反传；答案图仅作为 Flow、Mask teacher 和 teacher-validity/可选 distillation 目标，从而在不引入 image-to-image 条件的情况下把 U 的空间理解能力用于约束 G。
