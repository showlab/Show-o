# BoxDiff-style DRSA Adapter 设计

## 目标与边界

这条路线把 BoxDiff 的“attention 对目标框做约束”改造成 Show-o2 的 adapter 旁路训练目标。Show-o2 原始主干、SigLIP、fusion、Qwen、diffusion head 和 VAE 均冻结，只训练 `DualRouteSpatialAlignmentAdapter`。因此它不是改写 Qwen 自注意力，也不是把 BoxDiff 的 UNet cross-attention 直接移植进来。

第一版只使用 `position_unseen72_v1.jsonl` 中由 prompt 规则生成的 pseudo box 做 controlled smoke。该 manifest 的框由 `left/right/above/below` 和固定 margin/gap 推导，不是 SPRIGHT 的人工真实框，不能据此宣称模型已经学会真实目标检测或 grounding。

## 旁路位置

U 路：`fusion_proj -> DRSA.understanding -> Qwen`。Qwen 自注意力的 image-query/text-key 子矩阵被视为 pseudo cross-attention，U-side 框损失通过该注意力反传到 understanding adapter 和 relation encoder。

G 路：`Qwen hidden -> DRSA.generation -> diff_proj -> diffusion head`。只对 Qwen attention 求框损失时，G 路梯度严格为零，因为 G adapter 位于 Qwen attention 之后。为避免这个结构性盲点，smoke 在 generation adapter 输出上构造冻结 phrase embedding 与 image-token hidden 的 cosine map，并对该 map 求 G-side box loss。这个 map 是可微的 proxy，不等同于官方 cross-attention。

## Loss

对每个对象 phrase 和目标框 `B`，令 `P_B(i,j)` 为 image-token 到 phrase 的空间概率图，`M_B` 为框 mask，`M_out=1-M_B`。

```text
L_inside = (1 - sum(P_B * M_B))^2
L_outside = sum(P_B * M_out)
L_corner = |proj_x(P_B)-proj_x(M_B)|_1
         + |proj_y(P_B)-proj_y(M_B)|_1
```

两个对象的关系使用加 margin 的质心约束。对于 left/right 使用 x 方向，对于 above/below 使用 y 方向：

```text
L_relation = ReLU(margin - signed_delta)^2
```

U-side 的 `L_box_u` 复用 `guidance/energy.py::LayoutEnergy` 的 inside/outside/corner 实现；G-side 的 `L_box_g` 对 phrase-to-image hidden-state map 使用同一组定义。smoke 总目标为：

```text
L_box = lambda_u * (L_box_u + L_relation_u)
      + lambda_g * (L_box_g + L_relation_g)
```

正式训练时再加回原始 `L_flow` 作为小权重 anchor，并另做 no-box、U-only、G-only、box-only、box+flow 消融；不能把 pseudo-box loss 直接替换 velocity loss。

## 学习率与梯度记录

梯度探针显示 SPRIGHT flow 上 G raw gradient 大于 U，而 VSR 的 U 梯度约大两个数量级、G 严格为零。因此 smoke 使用独立参数组作为保守起点：

```text
lr_u = 1e-5
lr_g = 5e-6
lr_relation = 5e-6
```

这不是最终最优值；每一步都记录 `L_box_u/L_box_g/L_relation`、U/G/relation raw gradient norm、AdamW update norm 和各自学习率，再用 BoxDiff loss 的真实梯度重新校准。严禁用一个学习率覆盖三组参数。

## 风险与下一步

1. Qwen self-attention 不是 BoxDiff 的 cross-attention，text-key 概率会和图像 key 竞争；需要记录 text gate 并做 text-conditional normalization 消融。
2. phrase-to-image map 只是 G-side proxy，必须和 diffusion-head attention、生成图像的 box/关系评测交叉验证。
3. pseudo box 只能验证实现和梯度路径。正式训练需要真实 box/mask，或经过人工抽检的 Grounding DINO/GLIP pseudo grounding 数据。
4. 任何正式 checkpoint 都必须在未参与 pseudo-box 选择的 Position、T2I-CompBench spatial 和 GenEval/DPG 上复核，且保留 VSR baseline/最佳 VSR-only checkpoint。

