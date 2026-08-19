# Show-o2 优化项目 · 空间/组合可控生成
### 训练-free 注意力引导，攻 GenEval Position 弱项（2 周 · 单卡 80G）

---

## 一、背景与靶点（为什么做这个）

Show-o2 官方论文自曝的生成弱项里，**GenEval 的 Position（空间位置）子项明显偏科**：约 0.46–0.52，而强竞品在 0.66–0.79。也就是说，遇到"A 在 B 的左边 / 上方"这类**带空间关系**的 prompt，模型经常把物体位置摆错。这是统一多模态模型的一个公认痛点（组合/空间绑定难）。

**为什么这是个好靶点**：
- **有明确、公开的量化指标**（GenEval Position 子分），做完能直接给出"提升了多少"；
- **单卡可落地**：主打 training-free 的推理期方法，不需要重训（重训 1.5B 要 64×H100×1.5 天，单卡不可能）；
- **创新缺口清晰**（见下）。

---

## 二、优化点 & 创新性定位

**核心思路**：在 flow-matching 采样的**早期若干步**，从模型内部取出"图像 token → 文本 token"的注意力图，用一个**布局能量函数**把每个物体词的注意力**推进它该在的区域（bounding box）**，通过反向传播微调当前隐向量——从而在**不训练**的前提下纠正物体的空间位置。方法原型来自 UNet 扩散上的 **Layout-Guidance / 交叉注意力反向引导（backward guidance）**。

**创新性（和已有工作的差异，可辩护）**：

| 已有工作 | 它们的设定 | 本项目的差异 |
|---|---|---|
| Layout-Guidance / BoxDiff (2023-24) | UNet + DDPM，交叉注意力块 | **迁移到 Show-o2：LLM 骨干 Omni-Attention + flow-matching**（不是 U-Net、不是 DDPM） |
| COMPASS (2026) | Ming-Lite-Uni(8B MoE)，**微调** + **参考图** | **training-free** + **纯文本 prompt**，不动权重 |
| ConsistCompose | 布局控制需训练 | 主打免训练 |

**一句话创新点**：*据我们所知，这是首个在"统一 flow-matching 多模态模型（Show-o2）"上实现 training-free 空间/组合控制的方案——把交叉注意力反向引导从 U-Net/DDPM 适配到 LLM 骨干 Omni-Attention + ODE 采样器。*

三处**非平凡的适配**（这也是工作量与技术含量所在）：
1. **注意力来源变了**：Show-o2 的"交叉注意力"是 image-query→text-key，藏在 decoder LLM 的 Omni-Attention 里，不是 U-Net 的独立 cross-attn 块——**哪些层/头携带空间-物体信号，需要先探测**（本项目的分析贡献）。
2. **采样器变了**：DDPM 的 `z_t ← z_t − σ_t²·η·∇E` 要改写到 **flow-matching 的 ODE 步**里——在每个欧拉步之前对隐向量 x 施加引导梯度。
3. **连续隐空间**：引导作用在 VAE 连续隐向量上，不是离散 token。

---

## 三、技术方案（可复现的具体做法）

**① 布局能量函数**（沿用 Layout-Guidance 的形式）
对要约束的物体词 token `i` 和目标框 `B`：
```
E(A, B, i) = ( 1 − Σ_{u∈B} A_{u,i} / Σ_u A_{u,i} )²
```
含义：让 token `i` 的注意力质量尽量落进框 `B` 内。多个物体、多层注意力求和。

**② 引导施加在 flow-matching 采样步里**
在早期去噪步（前 ~10/50 步），对当前隐向量 x 做 K 次（~5）内迭代：
```
x ← x − λ_t · η · ∇_x Σ_layers E(A, B, i)     # 反向引导：把 x 往"注意力进框"的方向推
然后再走正常的欧拉步 x ← x + v·dt
```
- 只在**早期步**引导（布局在去噪早期就定型）；
- `η`（引导强度）~20–50，`K`~5，引导层取"探测"筛出的中间若干层；
- **噪声选择技巧**：初始噪声采样多个，取引导几步后能量最低的那个起步（论文证明能显著提升）。

**③ 框从哪来（prompt→box）**
- GenEval Position 是**模板化 prompt**（"a photo of a A {left of/above} a B"），可**规则解析**出物体词 + 左右/上下两个框；
- 通用 prompt 用一个小 LLM（可复用 Qwen）解析出"物体+相对位置→粗框"。

---

## 四、可行性分析（单卡 80G）

| 环节 | 显存/算力 | 结论 |
|---|---|---|
| Show-o2-1.5B 推理 + GenEval 评测 | 远小于 80G | ✅ 轻松 |
| 反向引导（对 x 求梯度，非权重） | 一次 fwd+bwd 的激活；配 grad-checkpoint | ✅ 1.5B 可行；7B 用 gradient checkpointing 亦可尝试 |
| 每图开销 | 早期 10 步 ×5 次引导 ≈ +50 次 fwd/bwd | 慢但仅评测用，可接受 |
| （拉伸目标）LoRA 微调 1.5B | LoRA 参数量小 | ✅ 单卡可训 |

**结论**：主线（training-free）**几乎零训练风险**；LoRA 作为增强对照，单卡也放得下。**默认在 1.5B 上做，7B 仅做推理对照评测。**

---

## 五、两周逐日计划

### Week 1 — 打通基线 + 探测 + 核心方法

| 天 | 任务 | 产出 |
|---|---|---|
| D1–2 | 配环境、跑通 Show-o2-1.5B 推理；跑 **GenEval 基线**，拿到 Position 及各子项基线分 | 基线数字表（复现弱项） |
| D3–4 | **注意力探测**：hook 骨干 Omni-Attention（及扩散头）取 image→text 注意力图，可视化哪些层/头携带空间-物体信号 | 注意力热力图 + 选层结论 |
| D5–7 | 实现**反向引导**适配 flow-matching 采样器；能量函数、对 x 求梯度、欧拉步前施加；在几条 Position prompt 上跑通 | 可运行的 guidance 模块 + 定性 before/after 图 |

### Week 2 — 全量方法 + 消融 + 评测 + 收尾

| 天 | 任务 | 产出 |
|---|---|---|
| D8–9 | prompt→box 解析（GenEval 模板规则 + 小 LLM 通用解析）；调 η/步数/选层；加噪声选择技巧 | 完整可控生成 pipeline |
| D10–11 | **全量 GenEval 评测**（Position + 两物体/计数/属性绑定各子项）+ 一个空间基准(如 T2I-CompBench spatial)；**消融**（选层 / η / 引导步数 / 有无噪声选择） | 主结果表 + 消融表 |
| D12 | （拉伸）在布局增强数据上做**轻量 LoRA** 作对照，看训练-free vs 轻训练差距 | 对照结果（有则加分，无则不影响主线） |
| D13–14 | 整理代码、画图、挑 demo 图、写 README/报告 | 代码仓 + 简短技术报告 |

---

## 六、评测方案

- **主指标**：GenEval **Position 子项**（核心），并报告 overall 及 Two-object/Counting/Color-attribution 各子项（确认不牺牲其他能力）；
- **辅指标**：T2I-CompBench 的 spatial 项 或 VISOR，交叉验证空间正确率；
- **质量护栏**：FID / CLIPScore 不明显退化（确认引导没伤画质）；
- **基线**：原版 Show-o2-1.5B（同 seed、同步数、同 CFG）；
- **消融**：选层、η、引导步数、噪声选择开关、（拉伸）LoRA 对照。

**预期结果口径**（保守）：Position 子项提升 **+0.10~0.20**、overall 不降、画质护栏不破——参照 Layout-Guidance 类方法在 SD 上 +5~10% 的空间正确率量级。

---

## 七、风险与降级方案

| 风险 | 降级方案 |
|---|---|
| 反向引导穿 LLM 骨干太慢/不稳 | 改用**前向引导**（直接改注意力图，λ≈0.8）或**只用噪声选择**，仍 training-free 见效 |
| 探测发现骨干注意力空间信号弱 | 改从**扩散头的注意力**取图，或对图像 token 施加**区域软掩码** |
| GenEval 环境依赖繁杂 | 官方给了 GenEval/DPG 的独立 env 脚本，按其 README 装即可 |
| training-free 提升有限 | 上**轻量 LoRA**（布局增强数据）兜底，保证有量化提升 |

**关键点**：主线不依赖训练，D7 就应有"能跑通、有定性效果"的里程碑，后续都是加分——**2 周内出结果的概率很高**。

---

## 八、预期产出（Deliverables）

1. **代码**：`guidance/`（注意力引导模块）+ `eval/`（GenEval 跑分脚本）+ 注意力可视化工具；
2. **技术报告**（3–5 页）：基线、方法、主结果表、消融、before/after 对比图；
3. **一句话成绩**：*"在不训练的前提下，将 Show-o2-1.5B 的 GenEval Position 从 X 提升到 Y（+Z），overall 与画质不退化。"*

---
---

# 配套简历写法（中文）

> 口径：**复现 + 分析 + 方法改进 + 消融实验**，训练-free 为主线（真实、可辩护）。把"迁移到 Show-o2 新架构"和"量化提升"讲清楚，比堆工作量更有力。

## 标准版（推荐，4 条）

**统一多模态模型 Show-o2 的空间可控生成优化（个人研究项目）** &nbsp; 2026.xx
**技术栈**：PyTorch · Flow Matching · 交叉注意力引导 · GenEval · Qwen2.5 · SigLIP/VAE

- **问题定位**：复现 Show-o2 并跑通 GenEval 评测，定位其 **Position（空间位置）子项显著偏弱**的问题，锁定为优化靶点。
- **方法创新**：将扩散模型的**交叉注意力反向引导**（Layout-Guidance）**首次适配到统一 flow-matching 模型**——从 LLM 骨干 Omni-Attention 中探测并提取 image→text 注意力，用布局能量函数在 ODE 采样早期步反向微调隐向量，实现 **training-free 的空间/组合控制**（区别于需微调+参考图的 COMPASS 等方案）。
- **工程落地**：单卡 80G 上完成注意力探测、能量引导、prompt→box 解析与噪声选择技巧，构建完整可控生成 pipeline。
- **量化结果**：在**不训练**的前提下将 GenEval Position 子项提升 **+X**，overall 与 FID/CLIPScore 护栏不退化；并完成选层/引导强度/步数的消融分析。

## 精简版（2 条）

- 复现 Show-o2 并定位其 GenEval **空间位置**弱项；将交叉注意力反向引导**首次迁移到统一 flow-matching 模型**，从 Omni-Attention 提取注意力、在 ODE 早期步反向引导隐向量，实现 **training-free 空间可控生成**。
- 单卡 80G 落地完整 pipeline 与消融，**零训练**下 GenEval Position **+X**、整体与画质不退化。

## 面试讲解口径（背熟）

- **一句话**："我发现 Show-o2 在 GenEval 空间位置子项偏弱，就把 SD 上成熟的交叉注意力布局引导迁移到它的 flow-matching 架构上——难点是它的'交叉注意力'藏在 LLM 骨干的 Omni-Attention 里、采样是连续 ODE 不是离散 DDPM，我先探测哪些层携带空间信号，再把反向引导改写进欧拉步之前，training-free 就把 Position 提升了 X。"
- **可能追问 & 答**：
  - *为什么 training-free？* → 单卡重训 1.5B 不现实（官方 64×H100×1.5 天），且免训练方案更能验证"是架构里已有空间信息、只是没被用好"。
  - *和 COMPASS 区别？* → COMPASS 要微调 + 参考图；我是纯文本 prompt + 零训练，靶向 GenEval Position。
  - *引导为什么放早期步？* → 布局在去噪早期定型，早期引导性价比最高（消融验证）。
  - *怎么保证不伤画质？* → 只在早期若干步、限定层、加 FID/CLIPScore 护栏监控。

## ⚠️ 诚信提醒
简历里的 **+X** 用你实际跑出的数字；若最终主要跑通的是 training-free 主线、LoRA 未做完，就据实只写主线。"首次适配到统一 flow-matching 模型"这个说法在写简历时**再快速检索一次**确认无人抢先（研究进展快），稳妥可改为"将……适配到 Show-o2"。

---

## 参考文献（技术锚点）
- Show-o2: Improved Native Unified Multimodal Models (arXiv 2506.15564)
- Training-Free Layout Control with Cross-Attention Guidance (arXiv 2304.03373)
- BoxDiff / Check-Locate-Rectify (arXiv 2311.15773) / Marginal Attention Constraints (arXiv 2411.10495)
- COMPASS (arXiv 2606.28696)、ConsistCompose (arXiv 2511.18333)（相关工作，用于差异化定位）
