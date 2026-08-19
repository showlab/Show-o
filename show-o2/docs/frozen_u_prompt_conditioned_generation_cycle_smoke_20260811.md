# Prompt-Conditioned Frozen-U Cycle 实现与 Smoke 记录

更新时间：2026-08-11 UTC

## 实现范围

本轮只实现 Stage 2 最小回路，不加入 Prompt KD、grounding readout、answer-image feature matching 或多步 rollout：

```text
L_mask + L_flow + L_G-ret
                  + predicted clean latent
                    -> frozen VAE decode
                    -> predicted image
                    -> frozen VAE deterministic re-encode
                    -> frozen U/MMU four-way relation-choice logits
                    -> gated L_U-task
                    -> generation_spatial only
```

答案图不进入 G condition，只用于 Flow/Mask teacher 和 frozen-U validity gate。默认 `weight_u_task=0`，因此旧 C6 路径不会执行 U forward；专用 formal 启动脚本默认使用 strict gate，并要求显式提供校准后的 U 权重。

实现文件：

- `evaluation/c6_mask2former_alignment/frozen_u_cycle.py`
- `evaluation/c6_mask2former_alignment/train_online_alignment.py`
- `evaluation/c6_mask2former_alignment/run_prompt_u_cycle.sh`
- `evaluation/c6_mask2former_alignment/run_prompt_u_cycle_smoke.sh`
- `evaluation/c6_mask2former_alignment/test_frozen_u_cycle.py`

## 验证结果

CPU 测试：

```text
19 tests passed
```

1-step CUDA gradient smoke：

```text
run: prompt-u-cycle-gradient-smoke-20260811T145500Z
peak VRAM: 22.17 GiB
student deterministic U latent requires_grad: true
grad_norm(L_U-task -> generation_spatial): 3.136358
grad_norm(L_flow -> generation_spatial): 0.126814
decode-reencode latent MSE: 0.007619
frozen model/U/VAE/Mask2Former parameter grads: none
```

该 smoke 使用 `gate=none` 只验证 autograd 链。原始 U 权重 1.0 的梯度约为主要加权 Mask 分量的 10%-30%，可作为后续 calibration 起点，不能据此直接预注册正式权重。

## Yes/No Teacher Validity 负结果

同一固定 shuffle order 上运行两个 16-step strict-gate smoke：

```text
frozen DRSA understanding route: strict valid 0/16
frozen base Show-o2 U route:     strict valid 0/16
```

DRSA-U 的 16 个答案图中，15 个样本对 q1/q2/q3 输出 `[yes, yes, yes]`，1 个输出 `[no, no, no]`。Base-U 也主要输出常量 yes。也就是说，当前 U logits 对 positive/inverse/swap 问题缺少必要的反事实敏感性；失败不是单独由 DRSA understanding adapter 引起。

因此 yes/no triples 不能启动正式 5k：

- strict gate 下 `L_U-task` 全部为 0，没有新增监督；
- 关闭 gate 会把常量 yes teacher/critic shortcut 直接反传到 G；
- positive-only 会隐藏同一问题，不能作为空间关系对齐证据。

## 四方向 Readout 与最终 Smoke

固定候选语义坐标为 `left/right/above/below`，答案图只决定 target correctness/margin gate。审计结果：

```text
train 固定 64：
  base-U accuracy                  26/64 = 40.63%
  DRSA-U accuracy                 28/64 = 43.75%

独立 validation 1024：
  base-U accuracy                434/1024 = 42.38%
  DRSA-U accuracy               436/1024 = 42.58%
  DRSA-U positive-margin gate   422/1024 = 41.21%
  above/below/left/right accuracy          37.9%/48.4%/53.5%/30.5%
```

四方向 1-step gradient smoke：

```text
run: prompt-u-choice-gradient-smoke-20260811T072000Z
grad_norm(raw L_U-task -> generation_spatial): 18.581295
grad_norm(raw L_flow -> generation_spatial):    0.126809
peak VRAM: 17.51 GiB
```

由主要加权 Mask 基座梯度校准，strict mixed smoke 使用 `weight_u_task=0.2` 和 4-step ramp：

```text
run: prompt-u-choice-strict-smoke-20260811T072500Z
completed: 16/16
strict-valid: 7/16 = 43.75%
peak VRAM: 17.52 GiB
frozen model/U/VAE/Mask2Former parameter grads: none
```

正式长训仍需固定 `weight_u_task=0.2`、strict gate、相同 sample/noise order，并报告按方向 gate coverage；不得回退到 `gate=none` 或 positive-only。四方向 coverage 不均衡，checkpoint 结论必须按方向拆分并由独立 scorer 验证。
