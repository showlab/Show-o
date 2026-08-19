# Show-o2 C7 多维生成评测设计

本文定义 Show-o2/DRSA 后续生成能力评测的 canonical 协议。目标是把当前单一、检测器依赖的 Position-72 gate，扩展为覆盖图文一致性、人类偏好、细粒度忠实度、分布质量、空间关系与推理效率的多维评测，同时保持严格的同 prompt、同 seed 配对。

## 1. 背景与问题

最近的 adapter 实验大多没有在当前 Position-72 上得到可信增益，反而频繁出现对象检出与空间正确率下降。当前代表性结果为：

| variant | VSR full | Position-72 | 两对象均检出 |
|---|---:|---:|---:|
| baseline | `722/1222` | 当前对照见各 run | 当前对照见各 run |
| B1 step-200 | `741/1222` | `55/288` | `178/288` |
| mask-logits step-500 | `741/1222` | `47/288` | `166/288` |

mask-logits step-500 的 U 路与 B1 共有的 28 个 adapter 参数逐位完全一致，因此 VSR 预测也完全保持；G 路 Position 则被 B1 严格支配。该结果说明需要保留现有 gate 作为回归证据，但不能继续把它当作唯一生成质量标准。

当前 Position-72 的核心限制：

1. 只测试两个 COCO 对象与 `left/right/above/below` 四类粗关系。
2. 每张图只有二值 `correct`，不评价真实感、审美、细节或连续语义质量。
3. 对象类别受 COCO 闭集限制，检测阈值、框误差和类别混淆会直接变成生成错误。
4. 在线训练 Loss 与评测都使用同一套官方 GenEval Mask2Former，存在 evaluator leakage。
5. CLIP 虽被 GenEval 加载，但在纯 Position 任务中不参与评分；它只服务颜色分类。
6. 当前 run 时间包含模型加载、生成、写盘和评分，不能作为标准化速度指标。

C7 的目的不是寻找一个恰好把失败 checkpoint 判成成功的指标，而是用彼此独立的 scorer 区分：

```text
检测器误判
整体语义变化
人类偏好变化
细粒度 prompt 忠实度变化
真实感/多样性变化
精确空间关系变化
推理效率变化
```

## 2. 评测原则

1. 不合成未经验证的单一总分，报告多维向量。
2. 所有可配对指标固定 prompt、seed、采样步、CFG、分辨率和 dtype。
3. 训练使用过的 Mask2Former 不再是唯一或最终 scorer。
4. 同一指标固定模型版本、权重 SHA、预处理和软件环境。
5. checkpoint 选择使用 full protocol，不使用 Position Quick-32。
6. 先重评分已有图，再生成新数据；只有通过低成本 gate 的候选才进入 COCO-5K。
7. 对四个 seed 的结果按 prompt 聚类统计，不能把 288 张图错误视为 288 个完全独立样本。

首轮固定候选：

```text
baseline
B1 step-200
mask-logits step-500
```

需要扩展时再加入：

```text
early mixed DRSA step-500
old B400
object-detect step-500
```

## 3. 四个核心指标

### 3.1 CLIPScore

对 prompt `T` 与图像 `I`，计算归一化 CLIP embedding 的余弦相似度：

```text
s_clip = cosine(E_text(T), E_image(I))
CLIPScore = 2.5 * max(s_clip, 0)
```

它是逐图、无参考图、越高越好的全局图文一致性指标。C7 同时报告 raw cosine 与标准缩放分数，避免实现差异。

必须输出：

```text
mean / median
paired delta vs B1
paired win/tie/loss
prompt-cluster bootstrap 95% CI
relation counterfactual margin
```

反事实 margin 定义为：

```text
M_clip = CLIPScore(I, original relation prompt)
       - CLIPScore(I, swapped relation prompt)
```

`left<->right`、`above<->below` 成对交换。如果原 prompt 分数高但 margin 约为 0，只能说明对象/场景整体匹配，不能说明位置忠实。

CLIPScore 不作为空间 hard gate，因为它对数量、否定、属性绑定和左右关系不够敏感。

### 3.2 VQAScore

VQAScore 用视觉语言模型计算图像对文本命题回答 `Yes` 的条件概率：

```text
Q(T) = "Does this image show: <T>? Answer yes or no."
VQAScore(I,T) = P("Yes" | I, Q(T))
```

多 token 答案使用 token 概率的几何平均。C7 分两层运行：

```text
official score：对完整 prompt 计算一个 VQAScore
diagnostic QA：分别询问 object A、object B、正确关系和交换关系
```

空间诊断输出：

```text
P(A present)
P(B present)
P(correct relation)
P(swapped relation)
M_vqa = P(correct relation) - P(swapped relation)
```

`M_vqa` 是最重要的独立空间连续指标。只看 `P(Yes)` 会受 yes-bias 影响，因此禁止省略反事实问题。

主 scorer 在实现时必须预注册并冻结版本。建议：

```text
首轮：t2v-metrics 当前稳定版 + Qwen2.5-VL-7B 或同等级开源 judge
最终候选复核：t2v-metrics v3.0 的 CLIP-FlanT5 系列或另一非 Qwen judge
```

同一张表中不得混用不同 judge 的绝对分数。Qwen judge 与 Show-o2 的 Qwen2.5 语言家族存在相关性，因此最终结论需要第二架构复核。

### 3.3 ImageReward

ImageReward-v1.0 对 `(prompt, image)` 输出人类偏好 reward：

```text
r = R(T, I)
```

它综合画面自然度、构图、主体清晰度、明显瑕疵和整体 prompt 匹配。分数可正可负，越高越好；绝对值只在固定版本内有意义。

必须输出：

```text
mean / median reward
paired delta vs B1
paired win/tie/loss
prompt-cluster bootstrap 95% CI
按四方向与 object-pair bucket 的分数
```

ImageReward 主要回答“图是否更像人类愿意选择的结果”，不能回答精确位置是否正确。它只能作为质量/偏好 gate，不能覆盖 VQAScore、VISOR 或 GenEval Position。

未来可增加 VisionReward 多维评分，但不得用新版结果替换 ImageReward-v1.0 历史表而不重跑全部基线。

### 3.4 FID 与 KID

FID 比较真实图与生成图的 Inception-v3 特征高斯分布：

```text
FID = ||mu_real - mu_gen||^2
    + Tr(Sigma_real + Sigma_gen - 2 * (Sigma_real Sigma_gen)^(1/2))
```

FID 越低越好，反映分布级真实感、多样性和模式坍缩；它不评价单张 prompt 是否被正确执行。KID 使用特征上的无偏 MMD，有限样本下应与 FID 同时报告。

禁止在当前每 variant 288 张 Position 图上报告正式 FID，原因是：

1. 样本过少，协方差估计不稳定。
2. Position prompt 分布没有匹配的真实参考图分布。
3. FID 不支持同 prompt 的逐图配对判断。

正式协议采用 MLPerf Text-to-Image 风格：

```text
COCO-2014 validation 固定 5000 captions
每 caption 生成 1 张
真实 COCO 图作为 reference
生成与真实图使用同一 resize/crop 规范
报告 FID、KID、CLIPScore
```

FID 实现必须固定为 MLPerf accuracy 脚本或 CleanFID 中的一种；确定后不得跨实现比较数值。

## 4. 标准化速度与资源指标

速度不能从当前队列 wall time 推断。标准协议固定：

```text
GPU：NVIDIA RTX 6000D
分辨率：432x432
dtype：BF16
CFG：5.0
采样步：12
prompt：固定 128 条，覆盖短/长、单/双/多对象
warmup：10 次，不计入统计
```

分别测试 `batch=1` 与 `batch=4`，使用 `torch.cuda.Event` 和 `torch.cuda.synchronize()`，报告：

```text
cold start latency
warm end-to-end latency p50/p95
text/Qwen conditioning latency
12-step denoising latency
Wan VAE decode latency
images/second
seconds/image
peak allocated VRAM
peak reserved VRAM
```

图片写盘与 scorer 时间单独报告，不计入模型生成 latency。

## 5. 分阶段执行协议

### C7.0：现有 864 张零生成重评分

输入：

```text
/root/autodl-tmp/runs/showo2_mask2former_rollback/
  masklogits-step500-full-20260809T090000Z/full_position
```

图像构成：

```text
baseline          288
B1 step-200       288
mask-logits 0500  288
```

执行：

```text
CLIPScore
ImageReward-v1.0
VQAScore full-prompt
VQAScore object/relation counterfactual QA
```

目的：不重新生成，先判断当前 GenEval Position 是否漏掉全局语义、偏好或 VLM 判断上的收益。

### C7.1：独立空间 benchmark

运行官方：

```text
VISOR / SR2D
T2I-CompBench++ 2D spatial
```

每个 benchmark 使用官方 prompt、官方 scorer 和固定 seed。至少报告：

```text
object presence
unconditional spatial accuracy
conditional spatial accuracy given both objects
relation counterfactual margin
left/right/above/below 分项
```

这一步用于消除训练/评测共用 Mask2Former 的泄漏。若只有 GenEval 上升而 VISOR、CompBench 和 VQAScore 不升，不允许宣称空间生成改善。

### C7.2：综合 prompt 忠实度

运行：

```text
GenEval Full 553 prompts x 4 seeds
GenAI-Bench / VQAScore
DPG-Bench dense prompts（最终候选）
```

GenEval Full 分别报告：

```text
single object
two object
counting
colors
position
color attribution
overall macro average
```

DPG/GenAI 分别报告实体、属性、关系和复杂组合，不只报告一个总体均值。

### C7.3：COCO-5K 质量、速度与最终 gate

只有通过 C7.0/C7.1 的候选进入：

```text
COCO-5K FID/KID/CLIPScore
标准化 speed benchmark
峰值显存
```

该阶段预计每 variant 生成约 5000 张，不能对所有失败 checkpoint 重复执行。

## 6. 统计协议

逐图连续指标使用同 prompt/seed 配对，但置信区间的重采样单位是 prompt：

```text
一个 prompt 及其全部 seed 作为一个 cluster
bootstrap clusters 10000 次
报告 paired mean delta 与 95% CI
```

二值指标报告：

```text
gain / loss / net
exact McNemar p
Wilson interval
```

连续指标报告：

```text
mean / median
paired delta
win/tie/loss
cluster bootstrap 95% CI
effect size
```

FID/KID 报告固定实现的点估计，并通过 prompt bootstrap 或多 seed 重复给出方差。禁止只根据第三位小数排序模型。

同一轮涉及多个 scorer 时，保留原始 p 值和 Holm 校正结果；模型准入主要依据预注册 gate 与效应方向，不根据一次 `p<0.05` 临时改变指标。

## 7. Gate 设计

C7 不使用加权总分。候选进入下一阶段需同时满足：

```text
U retention：VSR 不低于初始化 checkpoint 的预设容差
Faithfulness：VQAScore 或 relation margin 出现正向、CI 可解释的增益
Preference：ImageReward 不显著下降
Global alignment：CLIPScore 不显著下降
Spatial：VISOR/CompBench 至少一个独立指标提升，另一个不显著退化
Distribution：FID/KID 不发生超出 baseline variance 的退化
Efficiency：延迟、吞吐、VRAM 在预设工程容差内
```

具体数值容差在 baseline/B1 重复运行估计方差后冻结，不能预先凭主观指定，也不能看到候选结果后修改。

一个指标上升但另一个下降时，报告 Pareto trade-off，不使用“综合分更高”掩盖能力崩溃。例如：

```text
ImageReward 上升 + VISOR 下降
= 图更好看，但位置更不忠实

VQAScore 上升 + FID 下降
= prompt 更忠实，但分布质量/多样性退化
```

## 8. 实现与环境隔离

建议新增：

```text
evaluation/c7_generation_benchmark/
  protocol.py
  score_clip.py
  score_vqa.py
  score_image_reward.py
  score_fid_kid.py
  benchmark_speed.py
  aggregate.py
  monitor.py
  run_c7_queue.sh
```

输出目录统一包含：

```text
protocol.json
environment.json
variant_map.json
per_image_scores.jsonl
paired_statistics.json
summary.json
status.json
logs/
```

每张图的 key 固定为：

```text
(benchmark, prompt_id, seed, variant)
```

评分环境隔离：

| 环境 | 用途 | 原因 |
|---|---|---|
| `geneval-score` | OpenCLIP ViT-L/14 | 已安装并有本地权重 |
| `showo2-score-vqa` | t2v-metrics + VLM judge | 可能需要 transformers 5，不能污染 Show-o2 4.47 |
| `showo2-score-reward` | ImageReward-v1.0 | 旧依赖与主环境隔离 |
| `showo2-score-fid` | MLPerf/CleanFID/KID | 固定 Inception 与预处理版本 |

禁止升级或覆盖现有 `showo2-eval`、`geneval-score` 环境中的核心 torch/transformers。

## 9. 当前本机可行性与预算

截至本设计落盘时：

```text
GPU：RTX 6000D 86GB
RAM：110GB
数据盘剩余：约 58GB
现有 C7.0 图片：864 张，约 200MB
OpenCLIP：2.20.0，已安装
CLIP ViT-L/14：890MB，已缓存
VQAScore/t2v-metrics：未安装
VLM judge 权重：未缓存；Qwen3-VL 目录只有 ref，不是完整权重
ImageReward：未安装、未缓存
FID/CleanFID：未安装
COCO-2014 val：未下载
```

预计资源：

| 阶段 | 新下载 | 新生成 | 预计 GPU 时间 |
|---|---:|---:|---:|
| CLIPScore 864 | `0` | `0` | `1-3 min` |
| ImageReward 864 | `1-2GB` 量级 | `0` | `5-15 min` |
| VQAScore 864 | `15-18GB` 量级 judge | `0` | `15-40 min` |
| COCO-5K | COCO `6-7GB` + scorer | `5000/variant` | 当前速度约 `1.7h/variant` |

按当前 PNG 大小，三个 variant 的 15000 张图约 `3.5GB`。磁盘可以完成一套 judge + COCO-5K，但必须保留至少 `10-15GB` 安全余量，不同时缓存多个大型 VLM judge。

## 10. 失败模式与审计

1. CLIPScore 可能对交换位置不敏感，所以必须报告 counterfactual margin。
2. VQAScore 可能受 yes-bias 和语言先验影响，所以必须成对询问正确/错误关系。
3. ImageReward 可能偏好好看的错误图，所以不能作为忠实度 gate。
4. FID 可能被 resize/crop 和样本规模支配，所以实现与参考集必须冻结。
5. 多个 reward model 可能共享预训练偏差，所以最终候选需独立 judge 复核。
6. 不允许在训练中直接优化最终 C7 scorer 后仍把同一 scorer 称为独立测试。
7. 不允许只展示最有利指标或最有利 seed；所有预注册指标和失败项都必须落盘。

## 11. 推荐执行顺序

```text
1. C7.0 CLIPScore smoke + full
2. C7.0 ImageReward smoke + full
3. 建立隔离 VQAScore 环境，做 8 图 smoke
4. C7.0 VQAScore 864 + counterfactual QA
5. 汇总 baseline/B1/mask-logits 的多维 paired table
6. 只有观察到独立正信号时，运行 VISOR 与 T2I-CompBench Spatial
7. 只有独立空间/忠实度 gate 通过时，运行 COCO-5K FID/KID 与 speed
```

第一阶段的结论边界：如果 CLIPScore/ImageReward 上升而 VQAScore/空间指标下降，只能说明画面整体偏好提升，不能称为布局对齐成功。如果四类独立指标都不升，应接受 adapter 没有产生有效生成增益，而不是继续更换 scorer。

## 12. 官方参考

- GenEval: https://github.com/djghosh13/geneval
- VISOR/SR2D: https://arxiv.org/abs/2212.10015
- T2I-CompBench++: https://github.com/Karine-Huang/T2I-CompBench
- TIFA: https://github.com/Yushi-Hu/tifa
- VQAScore/GenAI-Bench: https://github.com/linzhiqiu/t2v_metrics
- DPG-Bench/ELLA: https://github.com/TencentQQGYLab/ELLA
- HPS v2: https://github.com/tgxs002/HPSv2
- ImageReward: https://github.com/THUDM/ImageReward
- PickScore: https://github.com/yuvalkirstain/PickScore
- MLPerf Text-to-Image: https://github.com/mlcommons/inference/tree/master/text_to_image
