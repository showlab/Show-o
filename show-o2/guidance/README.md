# Show-o2 Layout-Guided Inference

This package adds an independent, training-free layout-guidance path for Show-o2 T2I.
The original `inference_t2i.py` and `t2i_generate()` behavior remain unchanged.

## Components

- `layout.py`: parses left/right/above/below prompts and aligns object spans to Qwen tokens.
- `energy.py`: computes legacy centroid energy and BoxDiff Inner/Outer/Corner energy over
  image-query to text-key attention.
- `sampler.py`: applies latent gradient descent and fixed-step flow Euler sampling.
- `../inference_t2i_guided.py`: loads Show-o2, runs guidance, decodes the image, and saves diagnostics.

No model weights are updated. The guided path freezes Show-o2 and differentiates only with
respect to the current VAE latent.

## BoxDiff Adaptation

BoxDiff was designed for UNet cross-attention. Show-o2 has a unified self-attention sequence,
so this implementation extracts the image-query to prompt-text-key block and re-normalizes it
over prompt text keys. BOS and BOI are excluded. This is a pseudo cross-attention map, not an
architecturally identical UNet cross-attention map.

For entity `i`, target box mask `M_i`, attention map `A_i`, foreground fraction `P`, and corner
radius `L`, the implemented loss is:

```text
k_in  = max(1, round(P * |M_i|))
k_out = max(1, round(P * |1 - M_i|))

L_inner  = relu(1 - mean(topk(A_i * M_i, k_in)))
L_outer  = relu(mean(topk(A_i * (1 - M_i), k_out)))
L_corner = mean(|max_y(A_i) - max_y(M_i)| * C_x)
         + mean(|max_x(A_i) - max_x(M_i)| * C_y)

E = w_inner * sum_i L_inner
  + w_outer * sum_i L_outer
  + w_corner * sum_i L_corner
  + w_relation * mean_layout L_relation
```

`C_x` and `C_y` select `L` grid cells around each target-box boundary. The default BoxDiff
values match the official repository: `P=0.2`, `L=1`, and text-attention scale `100`.

## Layer Probe Result

The probe in `evaluation/c3_layout_guidance/layout_layer_probe.py` ran on the 20-prompt
Position holdout at timesteps 0.1, 0.3, and 0.5. Each record uses the same latent for a prompt
and its inverse relation. A layer only passes when both directions are correct.

```text
Qwen layer 21:       pair direction 35.00%, inverse flip 35.00%
Qwen layer 27:       pair direction 20.00%, inverse flip 20.00%
Diffusion layer 7:   pair direction 13.33%, inverse flip 18.33%
Diffusion layer 8:   pair direction 13.33%, inverse flip 15.00%
```

The result localizes the best readable relation geometry to Qwen layer 21 (zero-based CLI
index `20`). Diffusion-head attention has larger spatial deltas but substantially worse
direction fidelity, especially for left/right. This is evidence for a Qwen-to-diffusion
spatial transfer gap, not proof that a single layer exclusively owns layout.

UND and GEN cannot parse the prompt independently because both occur before Fusion and Qwen.
Their vector-Jacobian products measure how downstream layout energy is transported back to the
latent. Across representative Qwen and diffusion objectives, UND/GEN latent-gradient norm ratios
range from 0.56 to 1.27 and their mean cosine is close to zero or mildly positive. The probe does
not support the claim that the deeper UND branch simply overwhelms GEN by gradient magnitude.

For the selected Qwen layer 21 objective at `t=0.3`, UND and GEN latent-gradient RMS values are
`2.29e-6` and `2.18e-6` (ratio `1.05`, cosine `-0.002`). For the strongest direction-readable
Diffusion layer 7, they are `4.14e-4` and `6.68e-4` (ratio `0.62`, cosine `0.177`). SigLIP layers
1 and 2 consistently rank highest within UND by gradient-times-activation, so they are important
spatial transport points, but they still cannot own prompt-relation parsing before Qwen.

Probe outputs:

```text
/root/autodl-tmp/runs/showo2_layout_probe/full-20260804T054100Z/summary.json
```

## Parse Smoke

```bash
cd /root/autodl-tmp/Show-o/show-o2

/root/autodl-tmp/venvs/showo2-eval/bin/python inference_t2i_guided.py \
  --prompt "a red cube to the left of a blue ball" \
  --parse-only \
  --output-dir /tmp/showo2-layout-parse
```

## Minimal Real Smoke

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SHOWO2_MODEL=showlab/show-o2-1.5B
export SHOWO2_VAE_PATH=/root/autodl-tmp/Show-o/show-o2/Wan2.1_VAE.pth

/root/autodl-tmp/venvs/showo2-eval/bin/python inference_t2i_guided.py \
  --prompt "a red cube to the left of a blue ball" \
  --num-inference-steps 3 \
  --guidance-steps 1 \
  --inner-steps 1 \
  --layers 14 \
  --energy-step-size 1 \
  --output-dir /root/autodl-tmp/runs/showo2_guided/smoke
```

## BoxDiff Real Smoke

Use the probed Qwen layer 21 (`--layers 20`) and disable the legacy inside/cross terms:

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python inference_t2i_guided.py \
  --prompt "a red cube left of a blue ball" \
  --attention-source qwen \
  --layers 20 \
  --inside-weight 0 \
  --cross-weight 0 \
  --relation-weight 1 \
  --boxdiff-inner-weight 1 \
  --boxdiff-outer-weight 1 \
  --boxdiff-corner-weight 1 \
  --boxdiff-p 0.2 \
  --boxdiff-corner-radius 1 \
  --num-inference-steps 50 \
  --guidance-steps 10 \
  --inner-steps 5 \
  --energy-step-size 30 \
  --output-dir /root/autodl-tmp/runs/showo2_guided/boxdiff-qwen21
```

The step size and number of guided steps still require same-seed detector-scored A/B tuning.
The layer probe selects a representation; it does not validate a final generation hyperparameter.

## Full 432x432 Run

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python inference_t2i_guided.py \
  --prompt "a red cube to the left of a blue ball" \
  --num-inference-steps 50 \
  --guidance-steps 10 \
  --inner-steps 5 \
  --layers 7,14,21 \
  --energy-step-size 30 \
  --output-dir /root/autodl-tmp/runs/showo2_guided/cube-left-ball
```

For BoxDiff guidance, prefer the probed single layer `20` over the old provisional layer set.

## Same-Seed Baseline

Set `--guidance-steps 0` and keep every other argument, including `--seed`, identical:

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python inference_t2i_guided.py \
  --prompt "a red cube to the left of a blue ball" \
  --seed 2026 \
  --guidance-steps 0 \
  --output-dir /root/autodl-tmp/runs/showo2_guided/baseline-seed2026
```

## Explicit Layout JSON

For prompts outside the built-in relation templates:

```json
{
  "relation": "explicit",
  "objects": [
    {"phrase": "red cube", "box": [0.08, 0.08, 0.45, 0.92]},
    {"phrase": "blue ball", "box": [0.55, 0.08, 0.92, 0.92]}
  ]
}
```

Pass the file with `--layout-json /path/to/layout.json`.

## Outputs

Each run writes:

- `guided.png`: decoded 432x432 image.
- `layout.json`: parsed object phrases, boxes, character spans, and Qwen token indices.
- `diagnostics.json`: per-step energy, inside ratio, gradient RMS, update RMS, velocity RMS,
  BoxDiff Inner/Outer/Corner components, and cosine between the negative energy gradient and
  model velocity.
- `run.json`: resolved run configuration and output paths.

## Tests

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python -m unittest \
  evaluation.c3_layout_guidance.test_guidance -v
```
