# Show-o2 本地项目上下文

更新时间：2026-08-09 UTC

本文是当前本地 Show-o2 空间/组合生成项目的统一入口。它描述当前代码、数据、运行产物和已经验证的实验结论，不替代上游 `README.md`，也不把早期计划当作已完成事实。

## 1. 信息优先级

遇到文档与结果不一致时，按下面的顺序取值：

1. `runs/**/summary.json`、`results.json` 和完成态 `status.json`。
2. 与对应 run 同期写入的结果文档。
3. `docs/` 中的实验设计和阶段总结。
4. `../showo2_project_plan.md`。该文件是最初的 training-free 两周计划，已经明显早于 C4-C7 实验。

不同协议的绝对值不能混比。尤其要区分：

- 官方 GenEval Full：553 prompts、每 prompt 4 张，共 2212 张。
- Position-72：72 prompts、每 prompt 4 个同 seed 样本，共 288 张。
- Position quick/paired8：开发或 retention 子集，只用于筛选和排错。
- VSR-256 与 VSR full 1222：前者是固定开发子集，后者才是全量结果。

## 2. 仓库与运行环境

- Git 根目录：`/root/autodl-tmp/Show-o`。
- 主要代码目录：`/root/autodl-tmp/Show-o/show-o2`。
- 分支：`main`，指向 `45a5a2d`，与 `origin/main` 一致。
- 仓库是 shallow clone，当前只有一条可见提交，不能从本地历史重建上游演进。
- 工作树很脏：4 个 tracked 文件有修改，大量 C0-C7、DRSA、guidance、文档和脚本未跟踪。不要执行 reset/checkout/clean，也不要假设未跟踪文件可重新生成。
- 核心 tracked 修改：`models/modeling_showo2_qwen2_5.py`、`datasets/__init__.py`、`evaluation/compute_dpg_bench.py`、`utils.py`。
- GPU：NVIDIA RTX 6000D，85651 MiB；当前 GPU 空闲。
- 数据盘：450 GiB，总使用约 363 GiB，剩余约 88 GiB。
- 当前没有 tmux 会话，也没有 Show-o2 训练或评测进程运行。
- 关键环境：`showo2-eval`、`showo2-score-reward`、`showo2-score-vqa`、`showo2-score-fid`。

## 3. 上游 Show-o2 架构

Show-o2 是统一理解和生成模型。核心路径为：

```text
image/video
  -> Wan2.1 3D Causal VAE latent
  -> dual patch embedding
       semantic path: SigLIP vision encoder (last layer removed)
       generation path: direct latent patch embedding
  -> concat + fusion_proj
  -> Qwen2.5 shared backbone
       text logits -> autoregressive next-token prediction
       final hidden states -> diff_proj -> 10-layer diffusion head
  -> velocity field in VAE latent space
  -> Linear transport / flow-matching ODE
  -> Wan VAE decode
```

训练的两个原生目标是：

```text
L = ntp_coeff * L_next_token + flow_coeff * L_velocity_mse
```

默认 1.5B Stage-1 配置使用 BF16、AdamW、linear transport、velocity prediction、log-normal timestep sampling、time shift；标准推理使用 Euler、CFG 5.0、50 步。项目内的快速 Position 实验经常改为 12 步，因此不能和官方 50 步结果直接比较。

主要入口：

- `inference_t2i.py`：上游 T2I 推理。
- `inference_mmu.py`、`inference_mmu_vid.py`：图像/视频理解。
- `train_stage_one.py`、`train_stage_two.py`：官方两阶段联合训练路径。
- `transport/`：flow path、训练采样和 ODE/SDE 积分。
- `models/modeling_showo2_qwen2_5.py`：统一模型和本地 DRSA 接入点。

## 4. 本地空间对齐改造

### 4.1 DRSA 双路适配器

`models/spatial_alignment_adapter.py` 实现 `DualRouteSpatialAlignmentAdapter`：

- U route：插在 `fusion_proj` 之后、Qwen 之前，作用于理解侧 image tokens。
- G route：插在 Qwen final hidden states 之后、`diff_proj` 之前，作用于生成侧 image-token spans。
- shared relation encoder：把 relation text features 池化到 256 维 relation code。
- U/G 各自使用 `RelationConditionedSpatialAttention`。
- 几何只进入 Q/K 和 attention logits，不直接加到 value 或 residual hidden states。
- 位置机制由 axial 2D RoPE 和 relation-conditioned continuous relative bias 组成。
- relative bias 使用 12 个方向/距离特征，包括 signed `dx/dy`。
- output projection 和 relative-bias 最后一层零初始化，初始 checkpoint 应严格保持基线行为。

基础 DRSA 仅训练 U/G/relation，共 19,437,504 个参数。B-prime 还可按需创建 `generation_spatial`，仅训练该分支时有 9,487,584 个参数。

### 4.2 Training-free guidance

`guidance/` 和 `inference_t2i_guided.py` 实现独立的 inference-time latent guidance：

- 规则解析 left/right/above/below，并映射到 Qwen token spans。
- 从 unified self-attention 中截取 image-query 到 text-key 子块，形成 pseudo cross-attention。
- 支持 centroid、inside/cross、entropy/presence、BoxDiff inner/outer/corner 和 relation energy。
- 在固定步 Euler flow 采样的早期 step 对当前 latent 求梯度；不更新模型权重。
- 支持 step-size 限制、timestep decay、backtracking 和完整 diagnostics。

这不是原生 UNet cross-attention，不能把 BoxDiff 在 Show-o2 上的 attention proxy 当成等价机制。

### 4.3 B-prime 和 C6

B-prime 将 flow anchor 与 position-only 分开：

- B1：native flow + 小权重 MMU，训练 U/G/relation。
- B2：冻结 Show-o2、U、旧 G、relation 和 diffusion head，只训练新 `generation_spatial`。
- B2 使用真实 COCO box 构建 position、presence、separation 和 residual proxy loss。

C6 把官方 GenEval Mask2Former 放进训练计算图：

- 对 frozen teacher 和 student one-step prediction 计算 class-conditioned soft masks。
- loss 包含 mask BCE/Dice、presence、relation distillation/pass 和 object detection。
- 训练仍只更新 `generation_spatial`。
- 该路线与最终 GenEval 使用同一 detector，天然存在 evaluator leakage，必须用独立 judge 复核。

## 5. 数据合同

### SPRIGHT-COCO

- 约 40K COCO images，包含 `spatial_caption` 和 `original_caption`。
- 11 个 WebDataset tar shards，约 9.19 GiB，`SprightCocoIterableDataset` 流式读取，不解压。
- shard 会按 DDP rank/data worker 切分，并支持确定性 shuffle buffer。

### VSR

- 使用 Visual Spatial Reasoning zero-shot split。
- 图片在约 1 GiB zip 中按需读取；记录保留 caption、label、relation、subject、object。
- VSR 只直接监督 U route；`forward_und_only` 不经过 G route，因此 VSR step 上 G 梯度严格为 0。

### B-prime COCO boxes

- 从 40,504 个 SPRIGHT images 映射 COCO instance annotations，并同步 Show-o2 resize/center-crop。
- train 8000 rows，每方向 2000；validation 1024 rows，每方向 256；image split 无重叠。
- 数据审计：`/root/autodl-tmp/data/drsa/bprime_coco_boxes_v1/audit.json`。
- `BprimeCocoBoxDataset` 保持图片在原 tar 内，按 manifest 定位读取。

SugarCrepe、What'sUp、VISOR、GenEval 和 T2I-CompBench 应保持 evaluation-only，不得混入 adapter 训练。

## 6. C0-C7 实验地图

| 阶段 | 问题 | 当前证据 | 判定 |
|---|---|---|---|
| C0 | 复现官方理解/生成基线 | GenEval `0.70930`，Position `165/400=41.25%`；DPG `85.0719`；理解任务全队列完成 | 基线有效 |
| C1 | MMU 与 T2I 是否梯度冲突 | 96 paired samples；Qwen all cosine `0.0027`、conflict `45.83%`、MMU/flow norm ratio `176.27` | 存在近正交和强尺度不平衡，不是单一负 cosine 问题 |
| C2 | 文本条件在哪一层衰减 | 6636 records；Qwen final condition gain `0.00714`，diffusion head final `0.07280`，velocity `0.02570` | 条件信号存在且被放大；正确/错误样本的 paired gain 无显著差异 |
| C3 | training-free attention guidance 能否修复位置 | Qwen layer 21 的方向可读性最好但仅 35%；Position-72 BoxDiff gain/loss `10/10`，net 0 | 未获得净提升 |
| C4 | DRSA mixed training 能否同时提升 U/G | VSR full `59.08% -> 71.28%`，Position `20.83% -> 17.36%` | U 提升、G 退化，原目标存在 mismatch |
| C5 | staged/B-prime 能否隔离 flow 与 position | 早期 B2 Position `53/288`，低于 baseline/B1；后续 full-epoch selected checkpoint `0/288` | gate 失败，不能继续扩步或只调权重 |
| C6 | 在线 Mask2Former loss 能否把 proxy 传到图像 | baseline `58/288`；corrected C6 step 500/1000/2500/5000 为 `41/38/46/39` | 全部被 baseline 严格支配 |
| C7 | 独立指标是否揭示被 detector 漏掉的收益 | corrected-2500 CLIPScore `0.72529` vs baseline `0.70355`，但 relation margin 仍接近 0 且为负；ImageReward `0.51969` vs baseline `0.75569` | 只有全局 CLIP 上升，空间忠实度和偏好未通过 gate |

### C0 参考结果

官方 GenEval Full：

```text
single_object  99.06%
two_object     83.33%
counting       57.81%
colors         85.37%
position       41.25%
color_attr     58.75%
overall        0.70930
```

代表性理解基线：MME perception `1441.43`、GQA `59.93%`、AI2D `68.36%`、MMMU val `36.22%`、MMStar average `43.95%`、SEEDBench image `71.65%`。这些任务来自 2026-08-02 的 C0 full run。

### C3 结论边界

- Qwen layer 21 比 diffusion-head attention 更能读出方向，但 35% 仍不足以作为稳定控制信号。
- 最佳 C3 v3 在 32 张开发图上与 baseline 完全相同。
- BoxDiff unseen72 虽产生 10 gain 和 10 loss，但净值为 0；不能描述为改善。
- 当前证据指向 Qwen representation 到可控 diffusion velocity 的 transfer/readout gap。

### C4/C5 结论边界

- 旧 mixed loss 中 `L_flow` 监督整张 velocity，不直接监督 object box、binding 或 signed relation。
- 旧 `L_alignment` 实际是 relation text code 与 full caption text code 的 text-text cosine。
- 旧 `L_retention` 不是 frozen baseline distillation。
- VSR-only 能稳定提升 U route，说明理解监督本身有效。
- B-prime 排除了旧 parser、伪框、方向失衡和 flow 同步 backward，proxy 仍下降而 Position 退化。
- 根因不再只是 loss 权重：当前 G spatial branch 只有一个 pooled global relation code，没有两个独立 object grounding tokens。

### C6/C7 结论边界

- C6 relation-pass 在大量 invalid-teacher 样本上成为近常数主导项；最后一步示例中 teacher relation score 和 student relation score 都为 0。
- C6 可以提高全局 CLIPScore，但 CLIP relation counterfactual margin 仍接近 0，说明提升主要来自对象/场景整体匹配，而不是正确方向。
- ImageReward 对所有主要 adapter checkpoint 均低于 baseline；训练越久通常下降越多。
- 因此当前没有任何 adapter checkpoint 同时通过 Position、relation margin 和 human-preference gate。

## 7. 当前可信结论

1. Show-o2 的空间弱项是真实且可复现的，但不同 Position 协议的绝对值不同。
2. 文本条件没有简单地在 Qwen 或 diffusion head 中消失；C2 显示信号沿生成头被放大。
3. 失败更接近“对象级绑定和空间控制没有进入可用 velocity basis”，而不是“模型完全看不到 relation text”。
4. 单一 global relation code + image-token self-attention 不足以稳定控制两个对象。
5. proxy loss 下降不等于最终图像空间指标提升；B-prime 和 C6 都已经直接证明这一点。
6. 使用同一个 Mask2Former 训练和评价会放大 evaluator leakage，且在本项目中仍未得到最终指标收益。
7. 当前 baseline 仍是正式候选；A-400、B1-200、B2、C6 和 full-epoch extension 都不能宣称生成改进。

## 8. C7 当前完成度

已完成：

- Position-72 上的 OpenCLIP ViT-L/14 CLIPScore、反事实 relation margin。
- ImageReward-v1.0 全量重评分和 prompt-cluster bootstrap。
- ImageReward 权重、BERT tokenizer 和完整 Qwen2.5-VL-7B-Instruct 权重已在本机。

尚未完成：

- `score_vqa.py` 和 VQAScore/counterfactual QA 运行。
- 独立 VISOR/SR2D 或 T2I-CompBench++ spatial 评测。
- COCO-5K FID/KID。
- 标准化 speed/VRAM benchmark。
- C7 aggregate/gate 脚本。

因此 `docs/generation_benchmark_c7_design.md` 第 9 节中的“VLM/ImageReward 未缓存”已经过时；设计原则仍有效，但资源状态应以本节和实际目录为准。

## 9. 关键路径

代码：

- `models/modeling_showo2_qwen2_5.py`：DRSA 接入、attention probe API、T2I velocity。
- `models/spatial_alignment_adapter.py`：U/G/generation_spatial 几何注意力。
- `datasets/drsa_dataset.py`：SPRIGHT、VSR、B-prime 数据层。
- `guidance/`：training-free layout energy 和 guided Euler sampler。
- `evaluation/c0` 到 `evaluation/c7_generation_benchmark`：实验与评测阶段。

设计与结论：

- `docs/drsa_experimental_conclusions.md`
- `docs/staged_adapter_loss_design.md`
- `docs/generation_benchmark_c7_design.md`
- `guidance/README.md`

关键正式产物：

- C0：`/root/autodl-tmp/runs/showo2_c0/c0-full-20260802T125630Z`
- C1：`/root/autodl-tmp/runs/showo2_c1/c1-conflict-20260803T084024Z`
- C2：`/root/autodl-tmp/runs/showo2_c2/c2-full-coverage-20260803T105404Z`
- C3 probe：`/root/autodl-tmp/runs/showo2_layout_probe/full-20260804T054100Z`
- DRSA 20k：`/root/autodl-tmp/runs/showo2_drsa_benchmark/drsa-benchmark-20k-20260807T005441Z`
- B-prime：`/root/autodl-tmp/runs/showo2_bprime_benchmark/bprime-vs-oldb400-20260808T040000`
- C6 rollback：`/root/autodl-tmp/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z`
- C7 corrected scores：`/root/autodl-tmp/runs/showo2_c7/c7-corrected-scores-20260809T093027Z`

## 10. 后续工作的硬约束

下一轮训练不应继续采用“global relation code + image self-attention + 更多步数”的同构方案。新的生成分支至少需要：

1. object-specific grounding tokens：`phrase feature + bbox/geometry embedding + object id`。
2. object-to-region gated cross-attention 或 GLIGEN/MS-Diffusion 风格的 grounding block。
3. 在 grounding attention 上定义可解释的 region/binding loss。
4. 先用小 validation gate 证明 object map 可分离，再考虑 diffusion layers 4-7 的零初始化 residual/LoRA。
5. flow 只读监控，除非新实验明确重新注册 joint objective。
6. 最终空间结论必须由独立 scorer 支持，不能只看训练使用过的 Mask2Former。

近期最有信息量的低风险动作是先完成已有 2304 张 C7 图片的 VQAScore 与正确/交换关系反事实 QA。若 VQAScore 也没有正信号，应结束当前 adapter 家族，转向 object-specific grounding，而不是继续对 C6 checkpoint 扩步。

## 11. 工作区保护

- `Wan2.1_VAE.pth`、C7 judge weights、`runs/` 和 `data/` 都是大资产，不应进入普通源码 commit。
- `.orig` 文件记录了本地补丁前状态；在确认不再需要前不要删除。
- `__pycache__`、`wget-log*` 是生成物，但当前仍属于用户工作区；未经明确授权不要清理。
- 在整理提交前，应先把 C0-C7 源码、实验文档、权重/缓存和 run 产物分成不同的提交或忽略策略。
