# Native-Objective DRSA Adapter Warm-up 实验设计

## 1. 实验问题

当前 DRSA adapter 的 U/G、relation encoder 是新增模块，Show-o2 原始主干保持冻结。第一轮 BoxDiff smoke 已证明 U/G 框损失可以反传，但使用的是 GenEval Position prompt 加规则 pseudo box，不能说明 adapter 已适应 Show-o2 原始 hidden/flow 分布。

本实验先用 Show-o2 原生训练目标重新 warm-up adapter，再决定是否加入显式目标框 loss。目的不是重新训练 Show-o2，而是让随机初始化的 adapter 学会在已有 fusion、Qwen 和 diffusion head 上做近似零破坏的残差调节。

## 2. 数据边界

官方训练配置 `configs/showo2_1.5b_stage_1_a.yaml` 和 `stage_2_a.yaml` 要求的训练数据路径仍是 `path/to/your.jsonl` 或 `path/to/your/image/dir`。本机没有官方 66M/100M 级预训练数据，因此不能把本实验称为官方 native full-data retraining。

本地可复现数据是：

| 数据 | 本地路径 | 目标 | 角色 |
|---|---|---|---|
| SPRIGHT-COCO | `/root/autodl-tmp/data/drsa/spright_coco/data/*.tar` | image + original caption 的 velocity flow | T2I native-objective proxy |
| VSR | `/root/autodl-tmp/data/drsa/vsr/annotations/zeroshot/train.jsonl` + `vsr_images.zip` | image + relation statement 的 yes/no NTP | MMU native-objective proxy |

SPRIGHT 的 `original_caption` 用于 warm-up，避免第一阶段把 spatial caption 误当作位置监督；SPRIGHT 的 spatial caption 和 GenEval pseudo box 留给后续显式 grounding 消融。VSR 只提供理解目标，G 路在该任务上应为零梯度，这是预期行为。

## 3. 模型与冻结策略

模型从本地缓存 `showlab/show-o2-1.5B` 加载，新增 `DualRouteSpatialAlignmentAdapter`：

```text
fusion_proj -> DRSA.understanding -> Qwen backbone
Qwen hidden -> DRSA.generation -> diff_proj -> diffusion head
```

冻结：Show-o2、SigLIP、fusion、Qwen、image embedders、diffusion head、diff projection、VAE。训练：

```text
DRSA.understanding
DRSA.generation
DRSA.relation_encoder
```

output projection 和 relative-bias 最后一层的零初始化保留，使 adapter 初始行为接近原始 Show-o2。

## 4. Native-objective loss

SPRIGHT/T2I step 只使用原生 flow 目标：

```text
L_native_t2i = MSE(v_pred, velocity_target)
```

caption 使用 `original_caption`，不加入 BoxDiff inside/outside/corner/relation loss，不加入 text-text relation alignment。

VSR/MMU step 使用原生 next-token prediction：

```text
L_native_mmu = CE(logits, yes/no labels)
```

总目标按交替 task 执行，不在一个 batch 中混合两种目标：

```text
step odd:  L = L_native_t2i
step even: L = L_native_mmu
```

这样可以单独记录两个目标对 U/G/relation 的梯度。VSR step 的 G gradient 应为 0；如果非零，说明 forward 路径或冻结策略有问题。

## 5. 优化器与梯度控制

不能对 U/G 使用同一个学习率。初始参数组采用既有梯度探针支持的保守值：

```text
U adapter:        1e-5
G adapter:        5e-6
relation encoder: 5e-6
```

使用 AdamW、weight decay `0.01`、global grad clip `1.0`。每一步记录：

```text
loss_native_t2i / loss_native_mmu
raw grad U/G/relation
clipped grad U/G/relation
parameter-group learning rates
```

SMOKE 只使用 `workers=0`，避免 2GB/低 CPU 环境下 DataLoader worker 导致额外内存占用。正式训练前需重新 probe，因为 native flow 的梯度尺度不等于 BoxDiff loss 的梯度尺度。

## 6. 运行分层

### Smoke

4 steps，2 条 SPRIGHT + 2 条 VSR，随机初始化 adapter。验证：

1. 本地 tar/zip 数据可读取；
2. `L_flow` 和 `L_ntp` 都能反传；
3. T2I step 的 G gradient 非零；
4. VSR step 的 G gradient 为零；
5. U/G/relation checkpoint 实际更新且没有 NaN/OOM。

### 短 warm-up

建议 500-2000 steps，仍然只用 native objectives，保存每 500 step 的 adapter checkpoint。在固定 VSR 子集、Position paired 子集和 T2I baseline 上做 retention 检查。

### Grounding 消融

只有 warm-up 不破坏原能力后，才比较：

```text
A: random adapter + BoxDiff loss
B: native warm-up adapter + BoxDiff loss
C: native warm-up + U-only box loss
D: native warm-up + G-only box loss
E: native warm-up + U/G box loss
```

Box loss 的正式版本需要 COCO instance box 或高置信度 Grounding DINO/GLIP pseudo box；GenEval 规则框只用于实现 smoke。

## 7. 判定标准

本实验不以训练 loss 单独判定成功。至少满足：

1. native T2I retention 不低于随机 adapter baseline 的预设容差；
2. VSR 理解不能因 warm-up 明显下降；
3. T2I step 的 G gradient 非零、VSR step 的 G gradient 为零；
4. U/G 的实际 AdamW update 与分组学习率记录一致；
5. 后续 BoxDiff 训练在 Position paired 和未见关系集上分别报告 gain/loss，而不是只看平均分。

## 8. 现阶段结论

该路线可以利用 Show-o2 原始模型已经学到的视觉、文本和 flow 能力，但“冻结主干”不等于“无需原生数据”：adapter 仍需要在原始 hidden distribution 上 warm-up。由于官方 full native data 未在本机落盘，本次 smoke 只验证本地 SPRIGHT/VSR 的 native-objective proxy，不宣称复现官方预训练。

