# 双轴 Mask-logits 实验计划罗盘

更新时间：2026-08-15

## 1. 目标

本实验用于回答两个核心问题：

1. 将空间关系从单一四方向标签改为水平轴与垂直轴解耦的正交多标签后，是否能提高单轴和双轴空间生成正确率。
2. 在双轴 Mask-logits 上加入 Flow 轨迹约束后，是否能提高训练稳定性、延缓后期退化，并保持图文一致性与生成后再理解能力。

本阶段只训练生成侧参数，不训练理解头，不加入 TIFA，不加入 frozen-U self-distillation，也不加入 Teacher Replay。

本阶段结束后先检查四组五 checkpoint 的结果，再决定是否启动新双轴矩阵中的 Teacher Replay 两组。

## 2. 固定研究边界

### 2.1 当前训练对象

只训练以下两种参数匹配路线：

| 路线 | 位置 | 目标参数规模 |
|---|---|---:|
| G-adapter | Show-o2 生成侧 generation adaptation | 约 9.53M |
| LoRA | 与 G-adapter 参数量匹配的生成侧 LoRA | 与 G-adapter 差异不超过 12 参数 |

以下模块冻结：

- Show-o2/Qwen 主干参数
- 原始 diffusion head 主参数
- Wan VAE
- Mask2Former
- U/MMU 理解路径

### 2.2 本阶段不包含

- Teacher Replay / Teacher Retention
- frozen-U relation choice
- TIFA 问答对齐
- DRSA understanding adapter
- COCO FID/KID/CMMD
- 多 seed 完整 Self-cycle

这些内容只在首轮 A/B 结果通过后继续。

## 3. 数据来源

## 3.1 旧单轴历史数据

路径：

```text
/root/autodl-tmp/data/drsa/bprime_coco_geneval_v2/train.jsonl
```

协议：

```text
bprime-geneval-aligned-manifest-v2
```

来源：

- SPRIGHT 本地 COCO-origin 图像
- COCO 2017 instance boxes
- 中心裁剪后的归一化 box
- GenEval relative-position 几何公式

统计：

| 项目 | 数值 |
|---|---:|
| 训练行数 | 17,100 |
| 唯一图像 | 7,712 |
| left | 4,275 |
| right | 4,275 |
| above | 4,275 |
| below | 4,275 |
| 每条关系数 | 1 |
| source score 最低训练门槛 | 0.6 |

SHA256：

```text
19db05947add7166825f1740e7f02ef4fc0076f35e128bde6955c60b110b1774
```

旧数据使用主导轴规则，同一物体 pair 只保留一个方向。明显对角关系会被过滤或只标为单轴。

## 3.2 新双轴数据

路径：

```text
/root/autodl-tmp/data/drsa/mask_logits_coco_multiaxis_v1/train.jsonl
```

协议：

```text
mask-logits-orthogonal-multilabel-v1
```

原始资源：

| 资源 | 数量或位置 |
|---|---|
| SPRIGHT COCO-origin 图像 | 40,504 |
| COCO 类别 | 80 |
| COCO 图像与 instance box | train2017 + val2017 |
| 本地 SPRIGHT shards | `/root/autodl-tmp/data/drsa/spright_coco/data` |
| COCO annotation zip | `/root/autodl-tmp/datasets/COCO2017/downloads/modelscope/annotations_trainval2017.zip` |

构造规则：

- 每个物体 pair 最多包含一个水平关系。
- 每个物体 pair 最多包含一个垂直关系。
- `left/right` 轴内互斥。
- `above/below` 轴内互斥。
- 水平和垂直关系允许同时存在。
- 每个 source image 最多选两个不同 object pair。
- train/validation 按 image id 哈希切分，图像零重叠。
- source box 的每个目标关系必须满足 GenEval score 大于 0.5。
- 训练 relation-pass 目标仍为 0.6。

训练统计：

| 项目 | 数值 |
|---|---:|
| 训练行数 | 17,100 |
| 唯一图像 | 6,386 |
| 单轴行数 | 13,702 |
| 双轴行数 | 3,398 |
| 双轴比例 | 19.87% |
| left targets | 5,125 |
| right targets | 5,125 |
| above targets | 5,124 |
| below targets | 5,124 |

训练 manifest SHA256：

```text
cc05cd82ecdf941ed4fd0cebea4f9f6e031df36f5881c58341065a4d3fcee413
```

验证统计：

| 项目 | 数值 |
|---|---:|
| 验证行数 | 976 |
| 唯一图像 | 396 |
| 双轴行数 | 416 |
| 双轴比例 | 42.62% |
| 每个方向 target | 348 |

验证 manifest SHA256：

```text
d5765a776206c8848ca6b5f7f15ac10bc1ee6c4e4ab7149fe630b198ddd22101
```

双轴 prompt 示例：

```text
a photo of train to the left of and above suitcase
a photo of fork to the right of and below cup
```

## 4. Mask-logits Loss 设计

## 4.1 教师分支

真实图像进入冻结 Mask2Former：

```text
real COCO image
-> frozen Mask2Former
-> class-conditioned soft masks
-> presence and relation scores
-> stop-gradient teacher target
```

记为：

```math
T = stopgrad(D_frozen(I_real))
```

## 4.2 学生分支

空间 prompt 经当前生成模型得到预测图：

```text
spatial prompt
-> frozen Show-o2 backbone
-> trainable G-adapter or LoRA
-> velocity prediction
-> predicted clean latent
-> frozen VAE decode
-> predicted image
-> frozen Mask2Former
-> student masks and relation scores
```

记为：

```math
S_theta = D_frozen(I_hat_theta)
```

梯度路径：

```text
Mask/relation loss
-> Mask2Former input image
-> VAE decode
-> predicted clean latent
-> velocity
-> G-adapter or LoRA
```

Mask2Former、VAE 和 Show-o2 主参数均不更新。

## 4.3 双轴关系表示

每条样本的目标集合为：

```math
R_i subset {left, right, above, below}
```

约束：

```text
|R_i| in {1, 2}
left and right cannot coexist
above and below cannot coexist
one horizontal and one vertical target may coexist
```

例如：

```json
{"relations": ["left", "above"]}
```

## 4.4 Relation score

从两个 soft mask 的 soft box 计算四个独立分数：

```math
s_left, s_right, s_above, s_below
```

四个分数不经过全局 softmax。左上布局可以同时满足：

```text
s_left > threshold
s_above > threshold
```

## 4.5 样本内聚合

每个目标方向分别计算：

```math
L_distill_r = SmoothL1(s_student_r, s_teacher_r)
```

```math
L_pass_r = ReLU(0.6 - s_student_r)^2
```

单个样本的 relation loss：

```math
L_relation_i =
1 / |R_i| *
sum over r in R_i of
(lambda_distill * L_distill_r + lambda_pass * L_pass_r)
```

样本内取平均，因此双轴样本不会天然获得单轴样本两倍的总权重，但两个正交方向都会反传梯度。

## 4.6 固定 Mask-logits 权重

| Loss | 权重 |
|---|---:|
| mask BCE | 0.01 |
| mask Dice | 0.07 |
| object presence | 0.16 |
| relation distill | 0.50 |
| relation pass | 1.00 |
| object detect | 0.00 |

## 4.7 Flow 条件

纯 Mask-logits：

```math
L_total = L_mask
```

Mask-logits + Flow：

```math
L_total = L_mask + 0.25 * L_flow
```

其中：

```math
L_flow = MSE(v_pred, v_target)
```

本轮固定：

```text
rollout_steps = 1
Flow weight = 0 or 0.25
Teacher Replay weight = 0
U-task weight = 0
```

## 5. A/B Test 矩阵

本轮核心因子：

- 因子 A：关系协议，旧单轴 vs 新双轴
- 因子 B：Flow，关闭 vs 开启
- 分层变量：G-adapter vs LoRA
- 时间变量：训练 checkpoint

## 5.1 四个训练组

| ID | 数据与关系协议 | Flow | G-adapter | LoRA |
|---|---|---:|---|---|
| S0-M | 旧单轴 Mask-logits | 0 | 已完成 | 已完成 |
| S0-MF | 旧单轴 Mask-logits | 0.25 | 已完成 | 已完成 |
| S1-M | 新双轴 Mask-logits | 0 | 当前训练 | 当前队列第二项 |
| S1-MF | 新双轴 Mask-logits | 0.25 | 待接队列 | 待接队列 |

旧单轴纯 Mask 根目录：

```text
/root/autodl-tmp/runs/showo2_generation_route_pair/generation-route-pair-masklogits-full17100-ckpt12-20260814T113818Z
```

旧单轴 Mask+Flow 根目录：

```text
/root/autodl-tmp/runs/showo2_generation_factorial/mask-flow-teacher-replay-factorial-full17100-ckpt12-20260814T113818Z/mask_flow
```

新双轴纯 Mask 根目录：

```text
/root/autodl-tmp/runs/showo2_generation_route_pair/mask-logits-multiaxis-full17100-20260815T072000Z
```

新双轴 Mask+Flow 根目录在训练启动时固定并写入本表。

## 5.2 固定 checkpoint

所有训练仍保存 12 个 checkpoint：

```text
100, 200, 500, 1000, 2000, 4000,
5000, 6000, 8000, 10000, 13000, 17100
```

本轮只 benchmark：

```text
100, 500, 1000, 5000, 17100
```

模型点总数：

```text
4 groups * 2 routes * 5 checkpoints = 40 model points
```

## 5.3 主要对比

| 对比 | 回答的问题 |
|---|---|
| S0-M vs S1-M | 双轴数据与双轴 loss 本身是否有效 |
| S0-MF vs S1-MF | 双轴设计在加入 Flow 后是否仍有收益 |
| S0-M vs S0-MF | Flow 对旧单轴设计的作用 |
| S1-M vs S1-MF | Flow 对新双轴设计的作用 |
| G-adapter vs LoRA | 参数位置与参数化方式的影响 |
| step-100 -> 17100 | 早期收益、最佳点和后期退化 |

G-adapter 与 LoRA 首轮分别报告，不先做平均。

旧单轴与新双轴的比较是“历史协议 vs 新协议”比较，不应写成只改变了一个 loss 项，因为数据采样和 prompt 也同时改变。

## 6. Benchmark 罗盘

## 6.1 指标分层

| 层级 | Benchmark | 角色 |
|---|---|---|
| P0 | DualAxis-96 | 直接验证双轴联合生成 |
| P0 | GenEval Position-72 | 验证标准单轴空间生成 |
| P1 | VQAScore Position-72 paired | 独立模型检查图文和方向语义 |
| P1 | Self-cycle compact96 | 检查生成后再理解保持 |
| P2 | VSR dev/test | 检查 generation adapter bridge 到 U 路后的迁移或破坏 |

不能只依赖 GenEval，因为 Mask-logits 训练与 GenEval 使用同一官方 Mask2Former，会存在 evaluator-family bias。VQAScore 和 Self-cycle 用于提供不同模型路径的交叉证据。

## 6.2 DualAxis-96

状态：需要新增固定 manifest 和 evaluator，原始验证数据已经就绪。

来源：

```text
/root/autodl-tmp/data/drsa/mask_logits_coco_multiaxis_v1/validation.jsonl
```

筛选：

- 只取双轴样本。
- 固定 96 条 prompt。
- 四个象限各 24 条：
  - left + above
  - left + below
  - right + above
  - right + below
- 固定 seed：20260810。
- 首轮每 prompt 生成 1 张图。
- 不使用 GPT-4o 或人工评分。

每 checkpoint 生成：

```text
96 images
```

主要指标：

| 指标 | 定义 |
|---|---|
| object pair pass | 两个目标物体均被检测 |
| horizontal pass | 请求的 left/right 成立 |
| vertical pass | 请求的 above/below 成立 |
| joint pass | object、horizontal、vertical 全部成立 |
| horizontal score | 目标水平分数均值 |
| vertical score | 目标垂直分数均值 |
| joint soft score | 两轴 soft score 的几何均值或最小值 |
| quadrant accuracy | 四个象限分别统计 |

主指标：

```text
DualAxis joint pass
```

注意：该指标仍使用冻结 Mask2Former，与训练 teacher 同族，因此必须结合 VQAScore 和其他指标解释。

## 6.3 GenEval Position-72

固定协议：

```text
72 prompts
4 images per prompt
288 images per checkpoint
```

主要指标：

- image accuracy
- prompt pass
- object detection pass
- position pass
- left/right/above/below 分方向准确率

作用：

- 检查新双轴训练是否保持标准单轴生成。
- 检查 Flow 是否改善后期稳定性。
- 与旧历史结果保持可比。

## 6.4 Self-cycle compact96

数据：

```text
/root/autodl-tmp/data/showo2_self_cycle/unicycle/self_built.jsonl
```

完整数据：

```text
689 records
1087 questions
```

首轮固定子集：

```text
protocol = self-cycle-compact96-stratified-v1
seed = 20260810
96 records
166 questions
14 type strata
num_inference_steps = 50
```

该子集约占：

```text
13.9% of records
15.3% of questions
```

固定要求：

- 每 checkpoint 只跑一个 seed。
- 生成 prompt 不提供给 MMU 问答路径。
- 结构化 QA 时关闭 generation adaptation。
- 不在首轮跑四 seed 或完整 689 records。

主要指标：

- question accuracy
- mean token F1
- prompt hard score
- prompt soft score
- 按 type 分组结果

作用：

- 检查生成图是否仍能被冻结理解路径正确读取。
- 检查空间优化是否破坏颜色、计数、物体、文字等非空间属性。
- 不是当前训练的直接优化目标。

## 6.5 VQAScore Position-72 Paired

本地实现：

```text
evaluation/c7_generation_benchmark/score_vqa.py
```

评分模型：

```text
Qwen2.5-VL-7B-Instruct
/root/autodl-tmp/models/c7/Qwen2.5-VL-7B-Instruct
```

协议：

```text
c7-vqascore-qwen2.5-vl-7b-position72-paired-v1
```

问题形式：

```text
Does this image show "<prompt>"?
```

每张图同时评分：

1. 原始空间 prompt。
2. 将 left/right 或 above/below 翻转后的 counterfactual prompt。

每 checkpoint 复用 GenEval 已生成的 288 张图，不额外运行 Show-o2 生图。

主要指标：

| 指标 | 解释 |
|---|---|
| mean VQAScore | 原 prompt 的 Yes 概率 |
| counterfactual VQAScore | 反向 prompt 的 Yes 概率 |
| relation margin | 原 prompt score - 反向 prompt score |
| positive margin rate | margin 大于 0 的图像比例 |
| by relation | 四个方向分别统计 |
| prompt-cluster bootstrap CI | 按 prompt 聚类的 95% CI |

主指标优先级：

```text
relation margin
positive margin rate
mean VQAScore
```

不能只看 mean VQAScore，因为它可能主要反映目标物体出现，而不是方向正确。

任务组织：

- 每个 `route + step` 建一个四组 paired job。
- 5 steps * 2 routes = 10 VQAScore jobs。
- 每个 job 评分 4 variants * 288 = 1,152 images。
- 总计 11,520 image scores。
- 每张图两个问题，总计 23,040 VQA queries。
- 使用 partial JSONL resume，避免中断后重算。

VQAScore 是冻结开源模型评分，不是 GPT-4o，但仍属于 learned metric，只作为交叉证据，不单独决定实验结论。

## 6.6 VSR

本地资源：

```text
/root/autodl-tmp/data/drsa/vsr/vsr_images.zip
/root/autodl-tmp/data/drsa/vsr/annotations/zeroshot/dev.jsonl
/root/autodl-tmp/data/drsa/vsr/annotations/zeroshot/test.jsonl
```

数据规模：

| Split | Rows |
|---|---:|
| dev | 340 |
| test | 1,222 |

本地 generation-adaptation evaluator：

```text
evaluation/c6_mask2former_alignment/eval_g_adapter_vsr.py
```

评测方式：

- `bridge=off`：generation adapter 不进入理解路径。
- `bridge=on`：将 checkpoint 的 generation adaptation 接到 U forward。
- 支持从当前 G-adapter 或 LoRA checkpoint 恢复。
- 输出 yes/no next-token logits。

本阶段正确定位：

```text
VSR is a retention/transfer sentinel, not proof of trained understanding.
```

原因：

- 当前训练没有 VSR loss。
- 当前训练没有 TIFA loss。
- 当前训练没有 U-task loss。
- VSR bridge-on 只测试生成 adapter 接入 U 路后是否产生迁移或副作用。

首轮协议：

- baseline bridge-off 只跑一次。
- 40 个 model points 跑 VSR dev bridge-on。
- 不为每个 checkpoint 重复跑 bridge-off，因为理论上应与 baseline 相同。
- 首轮筛选后，候选 checkpoint 再跑 VSR test 1,222 条。

主要指标：

- overall accuracy
- balanced accuracy by yes/no label
- exact core-4 subset accuracy：
  - above
  - below
  - left of
  - right of
- by relation accuracy
- prediction yes/no count
- checkpoint delta vs baseline bridge-off

VSR 只能作为退化否决项和迁移观察项，不能作为当前生成训练的主优化收益。

## 7. 首轮任务规模

40 个 model points 的首轮计划：

| 项目 | 每模型点 | 模型点 | 总量 |
|---|---:|---:|---:|
| GenEval | 288 images | 40 | 11,520 images |
| DualAxis-96 | 96 images | 40 | 3,840 images |
| Self-cycle compact96 | 96 images + 166 QA | 40 | 3,840 images + 6,640 QA |
| VQAScore | 复用 GenEval images | 40 | 11,520 scored images |
| VSR dev bridge-on | 340 predictions | 40 | 13,600 predictions |

需要 Show-o2 实际生成的新图总量：

```text
11,520 + 3,840 + 3,840 = 19,200 images
```

VQAScore 不增加 Show-o2 生图，只增加 Qwen2.5-VL scoring 时间。

已有生成结果只有在以下条件全部一致时才允许复用：

- checkpoint SHA/path 一致
- protocol version 一致
- prompt manifest hash 一致
- seed 一致
- inference steps 一致
- sample index 一致

否则重新生成。

## 8. 统计与报告方式

## 8.1 不只报告最佳值

每个 group/route 都报告五个 checkpoint 的完整曲线：

```text
100 -> 500 -> 1000 -> 5000 -> 17100
```

需要区分：

- 早期快速收益
- 中期最佳点
- 后期稳定或退化
- G-adapter 与 LoRA 的差异

## 8.2 配对比较

GenEval 和 VQAScore 使用相同 prompt/sample key 做 paired comparison。

优先报告：

- paired mean delta
- prompt-cluster bootstrap 95% CI
- wins/losses/ties
- per-relation delta

Self-cycle compact96 使用完全相同 record indices。

VSR 使用完全相同 dev/test row order。

## 8.3 不先合成单一总分

首轮使用 Pareto 选择，不把所有指标硬加权成一个总分。

候选 checkpoint 至少应满足：

1. DualAxis joint pass 相对旧同 route/step 提升。
2. GenEval prompt pass 没有明显退化。
3. VQAScore relation margin 不下降。
4. Self-cycle compact96 没有系统性下降。
5. VSR dev 不出现显著理解侧破坏。

## 9. 初步决策门槛

以下门槛用于首轮筛选，不作为最终论文显著性结论。

## 9.1 双轴设计继续条件

新双轴相对旧单轴，在相同 route/step 下：

- DualAxis joint pass 提升至少 3 个百分点，或在多个相邻 checkpoint 持续为正。
- GenEval prompt pass 下降不超过 2 个百分点。
- VQAScore relation margin 均值不下降。
- Self-cycle accuracy 或 mean token F1 下降不超过 3 个百分点。
- VSR dev 仅作为 veto，若下降超过 3 个百分点且多个 checkpoint 重复出现，则检查 bridge 设计。

## 9.2 Flow 继续条件

S1-MF 相对 S1-M：

- DualAxis joint pass 提升至少 2 个百分点，或
- step-5000/17100 的退化明显减轻，且
- GenEval、VQAScore、Self-cycle 不出现明显反向变化。

如果 Flow 只改善 late checkpoint，但 early checkpoint 变差，保留两条曲线并选择 checkpoint，不直接否定 Flow。

## 9.3 强证据条件

可在论文或简历中写为明确收益的结果应至少满足：

- 两条 route 中方向一致，或
- 同一 route 的多个 checkpoint 方向一致，且
- VQAScore paired bootstrap CI 支持非负或正向变化，且
- 非同族指标没有明显反证。

## 10. 执行顺序

### Phase 0：当前双轴纯 Mask

当前队列：

```text
session=showo2_mask_logits_multiaxis
run=/root/autodl-tmp/runs/showo2_generation_route_pair/mask-logits-multiaxis-full17100-20260815T072000Z
```

顺序：

```text
S1-M G-adapter
-> S1-M LoRA
```

### Phase 1：双轴 Mask + Flow

当前队列完成后启动：

```text
S1-MF G-adapter
-> S1-MF LoRA
```

固定：

```text
manifest = mask_logits_coco_multiaxis_v1/train.jsonl
training_seed = 20260808
max_steps = 17100
Flow weight = 0.25
Teacher Replay = 0
U-task = 0
rollout_steps = 1
save steps = existing 12 checkpoints
```

### Phase 2：40-point preflight

在任何 benchmark 生图前验证：

- 四组两个 route 均完成。
- 五个目标 checkpoint 均存在。
- checkpoint 内 step、route、参数量正确。
- G-adapter 与 LoRA 参数量匹配。
- 旧组使用旧 manifest。
- 新组使用新 manifest。
- 每组 Flow 权重正确。
- U-task 与 Teacher Replay 均为 0。
- 不允许混用旧/新 schema。

### Phase 3：首轮 benchmark

推荐顺序：

1. GenEval Position-72
2. DualAxis-96
3. Self-cycle compact96
4. VQAScore paired scoring
5. VSR dev bridge-on
6. 聚合四组五 checkpoint 曲线

GenEval 与 Self-cycle 可以沿用 checkpoint queue 思路，但新的比较器必须允许旧组和新组拥有不同训练 manifest，不能继续使用“所有 variant 必须共享同一 manifest”的旧 factorial validator 假设。

### Phase 4：人工检查结果

需要先生成：

- 每 route 五 checkpoint 曲线
- 旧 vs 新 paired delta
- Flow on/off delta
- 每方向和每象限结果
- VQAScore margin
- Self-cycle type breakdown
- VSR overall/core-4/label balance

检查后再决定是否继续。

### Phase 5：剩余新双轴矩阵

只有 Phase 4 通过后，再启动：

| 新双轴组 | Flow | Teacher Replay |
|---|---:|---:|
| S1-MT | 0 | 0.20 |
| S1-MFT | 0.25 | 0.20 |

旧单轴 Teacher-Replay 的中断任务保留为历史记录，不优先恢复，也不填入新双轴矩阵。

## 11. 需要新增或调整的代码

本罗盘写完时，资源状态如下。

已存在：

- GenEval Position-72 generation/scoring
- Self-cycle compact96 selector
- Self-cycle prompt-hidden QA protocol
- VQAScore Position-72 paired scorer
- VSR generation-adaptation bridge evaluator
- G-adapter/LoRA checkpoint loader
- 百分比与 ETA monitor 基础组件

需要新增：

1. `DualAxis-96` 固定 prompt manifest。
2. `DualAxis-96` generation queue。
3. 双轴 Mask2Former joint evaluator。
4. 允许旧/新训练 manifest 并存的四组 checkpoint validator。
5. 10 个 route-step 四组 VQAScore job 的队列包装。
6. VSR dev 40-point queue 与 baseline-once 逻辑。
7. 统一 aggregate report。
8. 统一实时百分比和 ETA monitor。

## 12. 风险与解释约束

### 12.1 Same-detector bias

训练与 GenEval/DualAxis evaluator 使用同一官方 Mask2Former。

因此：

- GenEval/DualAxis 提升是必要证据。
- 但不能作为唯一证据。
- 必须结合 VQAScore、Self-cycle 和图像抽样检查。

### 12.2 数据协议同时变化

旧 vs 新不只是 loss 改变，还包括：

- 对角样本保留
- prompt 写法变化
- object pair 采样变化
- unique image 数变化
- source score 门槛变化

报告时使用“single-axis historical protocol vs orthogonal-multilabel protocol”，不要声称是严格单变量 loss ablation。

### 12.3 VQAScore 不是几何检测器

VQAScore 可能更关注物体和整体语义。

因此主看：

- relation margin
- counterfactual separation
- positive margin rate

不要只看原 prompt 的平均 Yes 概率。

### 12.4 VSR 不是当前训练目标

VSR bridge-on 的变化只表示 generation adapter 被接入 U forward 后的迁移或干扰。

不得写成：

```text
Mask-logits directly trained understanding ability
```

可以写成：

```text
The generation-side adaptation preserved or transferred spatial signals when
bridged into the frozen understanding path.
```

前提是结果确实支持。

### 12.5 Self-cycle compact96 只用于筛选

compact96 单 seed 适合首轮 A/B，不足以作为最终稳定性结论。

最终候选需要再跑：

- 完整 689 records，或
- 至少四 seed compact/full protocol

该步骤放在 Phase 4 之后。

## 13. 最终交付物

首轮应至少落盘：

```text
experiment_registry.json
training_preflight.json
dualaxis96_manifest.jsonl
dualaxis96_selection_report.json
compact96_stratified_v1.json
benchmark_queue_status.json
geneval_summary.json
dualaxis96_summary.json
self_cycle_summary.json
vqascore_summary.json
vsr_dev_summary.json
comparison_matrix.json
decision_report.md
```

每份 summary 必须记录：

- checkpoint absolute path
- checkpoint step
- route
- adaptation parameter count
- training manifest path and SHA256
- benchmark manifest path and SHA256
- seed
- inference steps
- scorer model/checkpoint
- protocol version
- completion marker
- elapsed time
- peak GPU memory

## 14. 当前结论边界

当前只能确认：

- 双轴数据和 loss 已实现。
- 双轴 CUDA smoke 已证明两个正交目标同时产生生成侧梯度。
- 新双轴纯 Mask 正在正式训练。
- 旧单轴纯 Mask 与旧单轴 Mask+Flow 已完成，可作为历史协议对照。

在 40-point benchmark 完成前，不预先判断：

- 双轴一定优于单轴。
- Flow 一定有益。
- G-adapter 一定优于 LoRA。
- 当前训练提升了理解能力。
- late checkpoint 一定优于 early checkpoint。

