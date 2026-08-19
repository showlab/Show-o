# GLIGEN-style Grounding 设计

## 参考机制

GLIGEN 将每个短语和其 bounding box 编码成 grounding token，经 gated cross-attention 注入冻结的扩散 backbone。可训练的是 grounding tokenizer、线性映射和 gate；基础扩散模型不更新。它的核心不是一个事后采样 energy，而是把“对象是什么”和“对象在哪里”保留为独立条件。

## Show-o2 适配

Show-o2 没有 UNet cross-attention，因此不能照搬 GLIGEN 的层和张量形状。建议新增一个轻量 `GroundingConditioner`：

```text
phrase tokens + normalized box + relation embedding
        -> MLP / small resampler -> grounding token
```

U 路把 grounding token 作为 fusion 后 image token 的 gated residual 条件；G 路把同一个对象级 token 作为 diffusion-head 前 image hidden 的 gated residual 条件。两路使用不同投影和不同 gate，relation encoder 可以共享但必须单独设参数组。

```text
h_u' = h_u + sigmoid(g_u) * W_u(g_object)
h_g' = h_g + sigmoid(g_g) * W_g(g_object)
```

gate 零初始化，以保证加载 adapter 时 baseline parity；训练时只打开 grounding 模块，Qwen 与 diffusion head 继续冻结。

## 监督与消融

Grounding token 的直接监督是 box-conditioned attention / phrase-to-image similarity，而不是把 box 坐标拼到 caption 尾部。建议消融：无 gate、共享 gate、U-only、G-only、U/G 独立 gate、真实 box 与 pseudo box。重点指标包括对象 presence、box IoU/centroid、关系方向、VSR 和文本到图像整体质量。

## 与 BoxDiff 路线的关系

BoxDiff 是训练时的 attention/energy 约束，GLIGEN 是可训练的 grounding condition。前者适合先验证位置梯度能否到达 U/G adapter，后者适合下一步解决“文本短语与空间条件被压成一个 relation code”的表示瓶颈。两者不能在第一轮同时引入，否则无法判断增益来自框损失还是新条件模块。

