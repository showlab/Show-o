# DRSA Adapter A/B/C 分阶段训练设计

本文是 DRSA（Dual-Route Spatial Alignment Adapter）A/B/C 训练协议的 canonical 设计文件。它统一说明阶段目的、数据、任务调度、Loss、参数更新范围、gate 和当前执行状态；不再另建同类设计文档。

## 1. 适用范围与模型边界

DRSA 插入 Show-o2 的两个位置：

```text
fusion_proj -> DRSA.understanding (U) -> Qwen backbone
Qwen hidden -> DRSA.generation (G) -> diff_proj -> diffusion head
```

三个阶段始终冻结：

```text
Show-o2 主干、SigLIP、fusion_proj、Qwen、diff_proj、diffusion head、Wan VAE
```

只训练：

```text
DRSA.understanding
DRSA.generation
DRSA.relation_encoder
```

优化器为三个独立 AdamW parameter groups：

| 参数组 | 学习率 | 作用 |
|---|---:|---|
| U | `1e-5` | 调整视觉理解 token 进入 Qwen 前的空间表示 |
| G | `5e-6` | 调整 Qwen hidden 进入 `diff_proj` 前的生成表示 |
| relation | `5e-6` | 为 U/G 提供共享的关系条件编码 |

公共设置为 `weight_decay=0.01`、global grad clip `1.0`。G 的 raw gradient 在阶段 B 中约为 U 的 `7.19x`，但 AdamW 实际 update 约为 U 的 `0.485x`，因此维持更小的 G 学习率，不能只按 raw gradient 比例改学习率。

## 2. 三阶段总览

| 阶段 | 当前状态 | 初始化 | 核心目的 | 训练方式 | 主要数据 | 主要 gate |
|---|---|---|---|---|---|---|
| A `native_anchor` | 已完成 | 近零残差的随机 DRSA | 让新 adapter 适应 Show-o2 已有 hidden/velocity 分布，先保住原生理解和生成 | SPRIGHT T2I 与 VSR MMU 交替；不启用空间框损失 | SPRIGHT `image + original_caption`；VSR train | VSR 256 不下降；Position paired 不退化；选中 step-400 |
| B `spatial_ramp` | 已完成并扩大 gate | A step-400 | 在 native 锚点仍生效时，逐步加入 U/G 空间监督，避免突然覆盖原生成分布 | 本地 500 steps 内把空间项从 0 线性升到目标权重；T2I:MMU=`2:1` | SPRIGHT 同图的 `original_caption + spatial_caption`；VSR train | VSR、Position-72、同 seed gain/loss、目标物体保留；当前 B step-400 为 provisional Pareto checkpoint |
| C1 `g_geometry` | 未启动 | B step-400，冻结 U 和官方 diffusion head | 无 flow 竞争地学习 G 几何 | 只训练新增 G spatial residual + relation；只反传 G-side box/relation/presence | 清洗、四方向均衡的 SPRIGHT/COCO grounding 样本 | 固定 G-map 验证、Position、object presence；flow 只读监控 |
| C2 `head_binding` | 未启动 | 通过 gate 的 C1，冻结 U、B400 G 和 C1 residual | 把 G 几何传递到 diffusion head | 只训练 diffusion layers 4-7 的零初始化 residual/LoRA；只反传 head-side 空间 Loss | 与 C1 相同的干净均衡 grounding 数据 | Position、object presence、native velocity drift；任一退化即早停 |

概念上的连续调度是：

```text
A: step 1 .. N_A
B: step N_A+1 .. N_A+N_B
C1: B selected checkpoint -> G geometry alignment
C2: C1 selected checkpoint -> diffusion-head binding
```

本机为了每阶段独立 gate，采用 checkpoint 续接：

```text
A 独立训练 500 steps，gate 选择 A step-400
B 从 A step-400 初始化，独立训练 500-step ramp
C1 将从 B step-400 初始化；C2 从 C1 gate 选中的 checkpoint 初始化，均未启动
```

当前 `StagedAdapterLossConfig` 的 `joint_stabilize` 是旧版“固定 flow + 空间 Loss”定义，不再作为 C 的执行协议。C1/C2 需要独立 composer，确保 flow 不对空间参数反传；两者均尚未实现或执行。

## 3. 数据来源与使用边界

| 数据 | 上游来源 | 本地路径 | 训练字段 | A 用法 | B/C1/C2 用法 |
|---|---|---|---|---|---|
| SPRIGHT-COCO | `SPRIGHT-T2I/spright_coco` | `/root/autodl-tmp/data/drsa/spright_coco/data/*.tar` | `image`、`original_caption`、`spatial_caption` | `original_caption` 提供 native T2I velocity proxy | B 构造 native/guided forward；C1/C2 只用清洗、均衡后的实体关系与 grounding 标签 |
| VSR zero-shot train | `albertvillanova/visual-spatial-reasoning` | `/root/autodl-tmp/data/drsa/vsr/annotations/zeroshot/train.jsonl`、`/root/autodl-tmp/data/drsa/vsr/vsr_images.zip` | `image`、`caption`、`relation`、`subj`、`obj`、`label` | yes/no MMU next-token supervision | 继续提供 U 路理解锚点；MMU step 不经过 G |

本机共有 11 个 SPRIGHT tar shard，约 9.19 GiB；VSR zip 约 1.0 GiB、包含 6940 张图。数据审计文件为：

```text
/root/autodl-tmp/runs/showo2_native_warmup/native_warmup_data_manifest.json
```

这些数据是 native-objective proxy，不是 Show-o2 官方 66M/100M stage-1/stage-2 全量训练数据。官方配置中的完整数据没有随仓库发布，因此 A/B/C 都不能表述为“复现官方全量预训练”。

SugarCrepe、What'sUp、VISOR、GenEval 和 Position holdout 均为评测数据，不进入 adapter 训练。

### 3.1 阶段 B 的布局标签来源

SPRIGHT 的自由文本 `spatial_caption` 先筛选可解析的 `left/right/above/below` 样本，再规范化为：

```text
a photo of <object-0> <relation> <object-1>
```

布局框不是 COCO instance annotation，而是由关系生成的规则 pseudo box：

```text
margin = 0.08
gap    = 0.10
left/right  -> 画面左右两个半平面
above/below -> 画面上下两个半平面
```

因此 B 的 BoxDiff loss 监督“关系对应的粗区域”，不监督真实物体框。这一限制必须保留：B 的 box loss 不下降或 G 路不增益，既可能是 adapter 路径问题，也可能是规则 pseudo box 与真实图像布局不匹配。

## 4. Loss 定义

对一个 SPRIGHT 图像，`c_native` 表示原始 caption，`c_guided` 表示规范化空间 caption。同一 clean latent 使用同一 transport 采样得到 noisy latent `x_t`、时间 `t` 和目标 velocity `v_target`。

| Loss | 定义 | 监督位置 | 目的 |
|---|---|---|---|
| `L_native_flow` | `MSE(v_adapter(x_t,t,c_native), v_target)` | U、G、relation 可反传 | 锚定原生 velocity 动力学，避免只学空间而忘记生成 |
| `L_mmu` | `CE(logits, yes/no target)` | U、relation；G 必须零梯度 | 保持/提升图像关系理解 |
| `L_guided_flow` | `MSE(v_adapter(x_t,t,c_guided), v_target)` | U、G、relation | 让空间 caption 仍预测同一真实图像的 velocity |
| `L_retention` | `MSE(v_adapter(c_native), stopgrad(v_base(c_native)))` | U、G、relation | 约束 adapter native forward 接近关闭 DRSA 的冻结 teacher |
| `L_box_u` | Qwen image-query/text-key attention 上的 inside/outside/corner BoxDiff energy | 主要 U，亦可经共享图影响 relation | 把实体文本注意力质量集中到规则目标区域 |
| `L_box_g` | G 输出的 27x27 hidden 与冻结 phrase embedding 相似度图上的 inside/outside/corner energy | G、relation | 在没有 UNet cross-attention 的 Show-o2 中提供 G-side 可微空间 proxy |
| `L_relation_u` | `ReLU(0.20 - signed_delta_u)^2` | U、relation | 约束 U attention centroid 满足左右/上下方向 |
| `L_relation_g` | `ReLU(0.20 - signed_delta_g)^2` | G、relation | 约束 G hidden-map centroid 满足方向 |
| `L_relation_code` | 预留的跨路 relation-code alignment | 预留 | 当前权重为 0，未参与 A/B 训练，不能宣称已经做了显式 code 对齐 |

其中 U/G 的 box 内部均采用：

```text
L_box = L_inside + 0.5 * L_outside + 0.25 * L_corner
```

U 路读取 Qwen 自注意力中的 image-query/text-key 切片；G 路读取 generation adapter 输出并用 phrase-to-image hidden map 构造概率图。G 路是为 Show-o2 自注意力架构设计的 proxy，不等同于 BoxDiff 原论文对 UNet cross-attention 的直接优化。

总 composer 接口为：

```text
L_total =
    w_native_flow  * L_native_flow
  + w_mmu          * L_mmu
  + w_guided_flow  * L_guided_flow
  + w_retention    * L_retention
  + w_box_u        * L_box_u
  + w_box_g        * L_box_g
  + w_relation_u   * L_relation_u
  + w_relation_g   * L_relation_g
  + w_relation_code * L_relation_code
```

所有项是加法；`L_relation_code` 不是负号。某个 optimizer step 没有计算的分项按 0 处理，但不会 detach 已存在分项的 autograd graph。

## 5. 每类训练 step 实际激活哪些 Loss

总公式不是每一步同时计算全部 Loss，而是由当前 task 和阶段共同决定：

| 阶段/task | 实际计算的 Loss | 预期更新 | 必须为零的路径 |
|---|---|---|---|
| A / SPRIGHT T2I | `L_native_flow` | U、G；relation 由 relation code 图决定 | 空间项全部为零 |
| A / VSR MMU | `L_mmu` | U、relation | G gradient/update 必须为 0 |
| B / SPRIGHT T2I | `L_native_flow + L_guided_flow + L_retention + L_box_u + L_box_g + L_relation_u + L_relation_g`，再乘 ramp 权重 | U、G、relation | `L_mmu=0`、`L_relation_code=0` |
| B / VSR MMU | `L_mmu` | U、relation | G gradient/update 必须为 0；所有 T2I loss 为 0 |
| C1 / G geometry | `L_box_g + L_relation_g + L_presence_g` | 新增 G spatial residual、relation | U、B400 G、diffusion head 冻结；所有 flow loss 不反传 |
| C2 / head binding | `L_box_head + L_relation_head + L_presence_head + L_geometry_consistency` | diffusion layers 4-7 residual/LoRA | U、B400 G、C1 residual 和官方 head 权重冻结；所有 flow loss 不反传 |

teacher retention 不是 caption embedding cosine，也不是像素重建 loss；它比较同一 noisy latent、时间步和 native caption 下的两条 velocity。

## 6. 阶段权重

| 阶段 | `native_flow` | `mmu` | `guided_flow` | `retention` | `box_u` | `box_g` | `relation_u` | `relation_g` | `relation_code` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A `native_anchor` | 1.0 | 0.01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B `spatial_ramp` 起点 | 1.0 | 0.01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B `spatial_ramp` 终点 | 1.0 | 0.01 | 0.25 | 0.20 | 0.05 | 0.01 | 0.02 | 0.005 | 0 |
| 旧 C `joint_stabilize`（弃用） | 1.0 | 0.01 | 0.25 | 0.20 | 0.05 | 0.01 | 0.02 | 0.005 | 0 |

阶段 B 在本地 step `s=1..500` 上使用：

```text
progress = s / 500
w_spatial(s) = progress * w_spatial_target
```

`native_flow=1.0` 和 `mmu=0.01` 不参与 B 的 ramp。新版 C1/C2 不使用这张固定权重表：flow 只作为无梯度监控指标，不进入空间参数的 optimizer step。

## 7. 各阶段具体执行协议

### 7.1 A：Native Anchor

目的：先把随机新增的 adapter 校准到冻结 Show-o2 的输入/输出分布，而不是一开始用规则框强迫空间变化。

实际运行：

```text
500 steps
SPRIGHT T2I : VSR MMU = 1 : 1
250 T2I + 250 MMU
linear LR warm-up = 50 steps
每 100 steps 保存 checkpoint
```

训练后 gate 比较 step-100/200/300/400/500；step-400 的 VSR 为 `149/256`，Position paired 为 `5/32`、gain/loss=`0/0`，因此作为 B 初始化。

### 7.2 B：Spatial Ramp

目的：在 native 锚点仍开启时，验证 U/G 空间 loss 是否能带来可测量的布局增益，同时避免突然引入大权重空间梯度。

实际运行：

```text
初始化 A step-400
本地 500 steps spatial ramp
task pattern = SPRIGHT, SPRIGHT, VSR
334 T2I + 166 MMU
334 个可解析空间样本，跳过 2561 个不满足四方向解析约束的记录
linear LR warm-up = 20 steps
每 100 steps 保存 checkpoint
```

扩大 gate 的结论是 U 路 VSR 改善，但 G 路 Position 没有显著增益；B step-400 是当前 joint gate 的 provisional checkpoint，B step-500 出现更多目标物体缺失。阶段 B 不能描述为已经解决空间生成。

### 7.3 C1：G Geometry Alignment

从 B400 的稳定基线出发，冻结 U、B400 原 G 和官方 diffusion head；增加零初始化、可关闭的 G spatial residual，只学习几何与实体存在：

```text
L_C1 = lambda_box * L_box_g
     + lambda_rel * L_relation_g
     + lambda_presence * L_presence_g
```

C1 不计算或不反传 flow Loss。flow 仅在固定验证 batch 上记录为只读 drift 指标。训练数据必须四方向均衡、去代词、限制实体短语长度，并优先使用真实 COCO instance box。

### 7.4 C2：Diffusion-Head Binding

冻结通过 gate 的 C1 G residual，单独学习 G 几何如何穿过 diffusion head。在 diffusion layers 4-7 增加零初始化 residual/LoRA，官方约 8.8 亿参数 head 保持冻结：

```text
L_C2 = lambda_box * L_box_head
     + lambda_rel * L_relation_head
     + lambda_presence * L_presence_head
     + lambda_consistency * L_geometry_consistency(head_map, stopgrad(g_map))
```

C2 同样不让 flow 对 head adapter 反传。native velocity drift、VSR 和生成质量只作为 checkpoint gate。这样 flow 不能旋转空间梯度，且新增 residual 可随时关闭回到 B400/C1。

## 8. Gate 与成功判定

| 检查 | A | B | C1/C2 |
|---|---|---|---|
| 主干冻结 | 必须 | 必须 | 必须 |
| VSR/MMU | 不低于 baseline | 不低于 A selected | U 冻结；必须不低于 B400 |
| Position paired | 不退化即可 | 必须报告 gain/loss/net 和按方向分解 | C1/C2 分别 gate，并在外部空间集复验 |
| 初始 latent | 同 seed、全部匹配 | 同 seed、全部匹配 | 同 seed、全部匹配 |
| G on MMU | gradient/update 严格为 0 | 严格为 0 | 严格为 0 |
| native generation | flow 与 paired proxy | teacher retention + 目标物体保留 | flow 仅只读监控；依赖 residual 可关闭性、object presence 和生成 gate |
| 选择方式 | gate 选 checkpoint | Pareto 选择，不按训练 loss 单独选择 | C1 先选几何 checkpoint，再固定进入 C2；任一退化则早停 |

训练 loss 下降不是充分条件。特别是 `L_native_flow`/`L_guided_flow` 下降只表示 velocity 拟合改善，不等价于实体存在、方向正确或文本意图忠实。

## 9. 代码与实验记录

核心实现：

```text
evaluation/c5_boxdiff_adapter/staged_adapter_loss.py
evaluation/c5_boxdiff_adapter/staged_spatial_adapter_smoke.py
evaluation/c5_boxdiff_adapter/native_objective_warmup_smoke.py
evaluation/c5_boxdiff_adapter/test_staged_adapter_loss.py
```

阶段记录：

```text
docs/native_adapter_warmup_experiment_v2.md
docs/staged_adapter_stage_b_500_results.md
/root/autodl-tmp/runs/showo2_native_warmup/native-warmup-500-20260807T070000Z
/root/autodl-tmp/runs/showo2_stage_b/stage-b-500-20260807T145350Z
/root/autodl-tmp/runs/showo2_stage_b_position72/stage-b-position72-20260807T173156Z/summary.json
```

当前状态总结：A 已完成，B 已完成并扩大 gate；旧版 fixed-flow C 已弃用，C1/C2 尚未实现或启动。后续修改 C 配方时应更新本文，而不是创建新的重复 A/B/C 设计文件。

## 10. B-prime：Flow/Position 解耦因果实验（已完成）

B-prime 用于直接验证“旧 B 的 flow 梯度是否覆盖 position 梯度”。它不是通过 gate 的 C1，而是进入正式 C1/C2 前的因果实验。

### 10.1 数据修正

旧 B 的 334 条空间样本来自自由文本 parser，方向分布为 `above/below/left/right = 257/54/17/6`，实体短语长且含代词。B-prime 不再解析 `spatial_caption`，而是把 SPRIGHT 的 `COCO_val2014_*` numeric image ID 映射到 COCO 2017 train/val instance annotation：

```text
40,504 SPRIGHT images scanned
COCO bbox 按 Show-o2 短边缩放 + 中心裁剪规则同步变换
只保留裁剪后可见比例 >= 0.70、面积 >= 0.01 的实例
每个类别在图中必须唯一，两个类别必须不同
方向主轴 gap >= 0.12，主轴/副轴 >= 1.25
pair IoU <= 0.20，小框被交集覆盖比例 <= 0.50
同一 image ID 不能跨 train/validation
```

最终数据：

| split | rows | 每方向 | unique images | image overlap |
|---|---:|---:|---:|---:|
| train | 8000 | 2000 | 5625 | 0 |
| validation | 1024 | 256 | 694 | 0 |

审计文件：`/root/autodl-tmp/data/drsa/bprime_coco_boxes_v1/audit.json`。

### 10.2 两段训练与冻结

```text
A step-400
  -> B1 native anchor 200 steps
       L_native_flow + 0.01 L_mmu
       train U/G/relation with 1e-5/5e-6/5e-6
       optimizer 从 A checkpoint 重新创建
  -> B2 position-only 300 steps
       enable zero-init generation_spatial
       freeze Show-o2 + U + old G + relation + diffusion head
       only train generation_spatial at 1e-5
       optimizer 再次重新创建，不继承 B1 AdamW state
```

B2 的 backward 明确禁止 `native_flow/guided_flow/velocity_retention/mmu`。实际 Loss 为：

```text
L_B2 = w_box * L_box_g
     + w_relation * L_relation_g
     + w_presence * L_presence_g
     + w_separation * L_separation
     + w_norm * L_residual_norm
```

`L_box_g` 使用真实 COCO box 内概率质量和 centroid 回归；`L_relation_g` 回归真实 bbox center 的 signed delta 并保留方向 margin；`L_presence_g` 约束空间分支后的 phrase log-sum-exp 不低于冻结 B1 hidden；`L_separation` 是两个 object map 的 Bhattacharyya overlap；`L_residual_norm` 限制分支偏移。

首个固定 batch 的 raw gradient norm 为 `box=0.02228`、`relation=0.00857`、`separation=0.00307`，据此得到权重 `0.38448/0.5/0.69663`。presence 和 residual 在零初始化点的梯度严格为 0，只能使用默认 `0.25/0.01`；这是本次设计的已知限制，不能称为五项都完成了 gradient normalization。`lr_spatial=1e-5` 的 smoke relative update 为 `1.43e-4`，与 B1 U 路 `1.47e-4` 同量级。

### 10.3 正式结果

训练使用 `/root/autodl-tmp/runs/showo2_bprime/bprime-split-500-20260808T035200`，评测使用 `/root/autodl-tmp/runs/showo2_bprime_benchmark/bprime-vs-oldb400-20260808T040000`。Position-72 为 72 prompts x 4 same-seed samples，所有初始 latent 配对一致。

| variant | VSR-256 | Position-72 | 两目标均可检测 | 对 baseline gain/loss/net |
|---|---:|---:|---:|---:|
| baseline | 145/256 | 58/288 | 185/288 | - |
| B1-200 native anchor | 151/256 | 58/288 | 179/288 | 7/7/0 |
| 旧 B400 joint flow+position | 152/256 | 60/288 | 179/288 | 7/5/+2 |
| B2-200 position-only | 151/256 | 53/288 | 178/288 | 8/13/-5 |

B1→B2 直接同 prompt/seed 配对为 `gain=7, loss=12, net=-5`。B2 按方向为 `above=22/76`、`below=25/84`、`left=6/48`、`right=0/80`；旧 B400 为 `28/76、28/84、4/48、0/80`。B2 的 left 增加 2，但上下合计减少 9，right 仍完全没有解决。

B2 的 proxy 确实下降：first-50→last-50 为 `box 2.94796→2.74523`、`relation 0.14675→0.10847`；但 separation `0.99772→0.99757` 几乎不变，step-300 只读 velocity drift 增至 `0.00567 MSE`。这说明优化器在改变 G hidden，却没有形成两个可分离、能被 diffusion head 使用的 object-specific spatial map。

配对差异未达到统计显著：B2 vs baseline exact McNemar `p≈0.383`，B2 vs B1 `p≈0.359`，B2 vs old B400 `p≈0.143`。因此不能宣称 B2 必然有害；但三项点估计均无增益、presence 不升、right 为 0，按预注册 gate 必须拒绝 B2 checkpoint，不能进入后续训练。

### 10.4 设计诊断与后续约束

这次实验排除了旧 parser、伪框、方向失衡和 flow 同步 backward 四个混杂项，仍然失败，说明问题不只是 flow 覆盖。当前 `generation_spatial` 只接收一个全局 pooled relation code，并在 image tokens 内做 self-attention；两个 object phrase 没有作为独立 grounding tokens 输入分支。外部 Loss 又把 post-Qwen hidden 与 input token embedding 做 cosine，这个代理空间没有被证明与 diffusion velocity 的对象控制对齐。proxy 下降而实际图像下降，是该代理不充分的直接证据。

下一轮不能只调 B2 权重或增加步数。必须同时满足：

1. 引入 object-specific grounding tokens：`phrase feature + bbox/geometry embedding + object id`，通过 gated cross-attention/GLIGEN-like block 进入 G 路。
2. 在 grounding attention 上定义可解释的 object-to-region Loss，避免用未校准的 input-embedding/G-hidden cosine 充当唯一代理。
3. presence 使用正 margin 或在若干步后重新 probe/归一化，避免零初始化时梯度为 0 导致权重失真。
4. 只有 object-specific G map 通过 validation gate 后，才进入 C2，在 diffusion layers 4-7 增加零初始化 residual/LoRA 做 head binding；不能直接解冻整个 diffusion head。
5. flow 继续只读监控，不重新混入 position backward。

当前状态更新：B-prime 已实现并完成，但 gate 失败；它不能替代正式 C1。C1 需要按上述 object-specific grounding 重新设计，C2 仍未启动。
