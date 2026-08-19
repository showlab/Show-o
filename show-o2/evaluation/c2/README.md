# Show-o2 C2 condition-localization benchmark

C2 tests where text conditioning is attenuated along Show-o2's text-to-image path. It keeps the target-image latent,
noise, and timestep fixed while batching four prompt variants: matched, duplicated same-prompt, within-category
counterfactual, and null prompt.

The probe captures image-token activations at Fusion, Qwen early/middle/final states, `diff_proj`, diffusion-head
layers 1/5/10, final velocity tokens, unpatchified velocity, and a one-step Euler prediction. Fusion is expected to be
prompt-independent because it runs before image tokens attend to prompt tokens; it serves as an internal negative
control. The duplicated same-prompt branch measures the numerical floor.

Counterfactual edits are minimal for counting, color, spatial relation, and color attribution. All results include
overall, per-category, per-timestep, generation-outcome, and token-length-matched aggregates. The default full profile
uses all 553 GenEval prompts and all 2,212 C0 images, including 1,553 correct and 659 incorrect generations. Three
timesteps produce 6,636 forward records.

Confidence intervals use `prompt_id` cluster bootstrap: four images and three timesteps from one prompt are not counted
as independent samples. An additional within-prompt analysis uses prompts that produced both correct and incorrect C0
images, computes correct-minus-incorrect condition gain inside each prompt, and then bootstraps those paired prompt
differences. Set `PER_TAG`, `IMAGES_PER_PROMPT`, and `OUTCOMES` only for smaller debugging runs; zero means no truncation
for the first two variables.

Run in tmux:

```bash
cd /root/autodl-tmp/Show-o/show-o2
./evaluation/c2/start_probe_tmux.sh
```

Monitor every five seconds:

```bash
./evaluation/c2/monitor_probe.sh "" 5
```
