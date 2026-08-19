# BoxDiff Adapter Smoke 记录

运行目录：`/root/autodl-tmp/runs/showo2_boxdiff_adapter/boxdiff-adapter-smoke-20260807T042200Z`。协议为 `boxdiff-adapter-pseudo-box-smoke-v1`，使用 4 条规则 pseudo-box prompt、4 个 optimizer step，基础 Show-o2 完全冻结。

| step | total loss | U box | G box | U grad | G grad | relation grad |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.21684 | 1.01021 | 1.12267 | 0.00618 | 0.02325 | 0 |
| 2 | 2.19708 | 1.00019 | 1.11914 | 0.01012 | 0.02016 | 0 |
| 3 | 2.19327 | 1.00020 | 1.11826 | 0.00750 | 0.02464 | 6.96e-10 |
| 4 | 2.19721 | 1.00574 | 1.11905 | 0.00928 | 0.02284 | 1.21e-8 |

4 步平均 raw gradient 约为 U `8.27e-3`、G `2.27e-2`，G/U 约 `2.75`。当前独立学习率为 U `1e-5`、G `5e-6`、relation `5e-6`。

step-1 到 step-4 的参数 delta norm 为 U `0.02494`、G `0.01276`、relation encoder `5.08e-8`。U/G 两侧反传链路和参数更新均已接通，但 relation encoder 几乎未更新。原因是 residual output 和 relative-bias 最后一层零初始化：早期梯度先更新输出投影，relation code 的梯度要在门控打开后才出现。

这个 smoke 只证明实现链路成立，不证明生成质量提升。下一轮应比较 random-init 与 VSR-4000 init，并增加 box-only、U-only、G-only、U+G 四组；在加入 `L_flow` 或 teacher retention 前不启动长训练。

## Native step-400 三路消融 smoke

运行目录：`/root/autodl-tmp/runs/showo2_boxdiff_adapter/boxdiff-ablation-smoke-20260807T083632Z`。三组均从 native warm-up `step-000400.pt` 初始化，使用相同的 4 条 prompt、seed `20260807`、4 个 optimizer step；学习率保持 U `1e-5`、G `5e-6`、relation `5e-6`。

| route | step-4 U grad | step-4 G grad | step-4 relation grad | step-4 U update | step-4 G update | step-4 relation update |
|---|---:|---:|---:|---:|---:|---:|
| U-only | 7.5073e-3 | 0 | 6.5402e-7 | 8.1217e-3 | 0 | 7.6623e-6 |
| G-only | 0 | 2.2123e-2 | 2.2650e-6 | 0 | 4.1633e-3 | 3.7282e-5 |
| U+G | 9.0774e-3 | 2.2212e-2 | 1.6060e-6 | 8.1277e-3 | 4.1641e-3 | 3.1642e-5 |

这里的 update 是 `optimizer.step()` 前后参数张量的 L2 delta norm，不是用 raw gradient 乘学习率估计的代理值。

实现检查结论：

- U-only 的 G 梯度和实际更新严格为 0。
- G-only 的 U 梯度和实际更新严格为 0。
- U+G 的两路梯度和实际更新都非零。
- relation encoder 在三组中均有非零梯度和更新。

因此路由冻结、独立 parameter group、独立学习率和 step-400 共同初始化均已验证。这个结果仍只是 4-step 反传链路 smoke；BoxDiff 的 G loss 当前是 phrase-to-image hidden-map proxy，尚不能据此声称位置生成质量提升。正式训练前还需加入 native `L_flow`/teacher retention，并用 paired Position 与 VSR 协议评价效果和遗忘。
