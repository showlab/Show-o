# MS-Diffusion-style Grounding 设计

## 参考机制

MS-Diffusion 使用 grounding resampler 和 multi-subject cross-attention，把每个 subject 的文本、参考图像和 box 保持为独立条件，再在多主体场景中进行隔离注入。基础扩散模型冻结，训练重点是 resampler/grounding 投影；MS-Bench 提供预设布局和多主体组合。

## Show-o2 适配

Show-o2 可采用轻量版本：每个对象建立独立 slot：

```text
subject_i = {phrase_i, box_i, optional_image_i}
z_i = GroundingResampler(subject_i)
```

`z_i` 不再和所有关系短语平均池化成单一 256 维 code。U 路根据 image-token 坐标对每个 slot 做 gated spatial attention；G 路在 diffusion-head 前对每个 slot 单独做 token-to-image binding。slot 间可增加轻量 exclusivity/overlap penalty，防止两个对象抢同一块空间。

## Loss 与训练策略

基础目标仍是 flow/NTP retention，位置辅助项包括 per-slot inside/outside/corner、质心关系和 slot exclusivity。建议先冻结 base model，只训练 resampler、U/G 投影和独立 gate；U/G 学习率由梯度探针校准，不能共享一个值。若引入参考图像，应先只做 text+box 版本，避免视觉 reference、布局和 flow 三个因素同时变化。

## 评测与风险

需要多主体、同类主体和交换位置的 paired benchmark，报告每个 subject 的 binding、box adherence、关系准确率、VSR 以及 GenEval/DPG。SPRIGHT 的 spatial caption 不能替代真实 per-subject box 标注；它只能提供 flow 图像和关系文本，不能证明 slot 已绑定到正确物体。MS-Diffusion 路线应在 BoxDiff smoke 和 GLIGEN ablation 之后实施。

