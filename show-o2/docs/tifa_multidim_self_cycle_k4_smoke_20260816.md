# TIFA K=4 Self-cycle Smoke and Start-point Audit (2026-08-16)

## Scope

This record validates only the differentiable K=4 training path and audits the
four proposed S1 continuation starts. No optimizer, tmux session, formal
training queue, or continuation checkpoint was created.

## Smoke protocol

- Dataset row: `coco_421933` from the pinned TIFA COCO multidimensional train split.
- U input: predicted image + question + choices; generation caption excluded.
- Questions: one spatial question plus activity, object, and attribute questions.
- Gate: frozen-U real-image correctness per question, margin > 0, at least two valid questions.
- Rollout: 4 velocity evaluations, 3 recurrent Euler transitions, 50-point shifted schedule.
- Shared times: `[0.12820512, 0.13913043, 0.15044247, 0.16216215]`.
- Pass condition: spatial gate valid, at least two valid dimensions, finite positive
  loss, predicted-image U input differentiable, and every adaptation tensor has
  a nonzero gradient.

## Results

| Start | Status | Student correct | Cycle loss | Gradient tensors | Peak allocated |
|---|---:|---:|---:|---:|---:|
| S1-M G-adapter step-5000 | pass | 3/4 | 0.577065 | 12/12 | 31.31 GiB |
| S1-MF G-adapter step-6000 | pass | 4/4 | 0.244007 | 12/12 | 31.31 GiB |
| S1-M LoRA step-6000 | pass | 4/4 | 0.255991 | 224/224 | 33.13 GiB |
| S1-MF LoRA step-6000 | pass | 4/4 | 0.259832 | 224/224 | 33.13 GiB |

All four real-image teacher gates were valid for every run, including the
spatial question. Gradient norms are intentionally not compared across
parameterizations.

## Spatial-objective checkpoint audit

The Show-o2 spatial weakness is represented here by GenEval Position-72.
Checkpoint selection therefore prioritizes image accuracy and prompt pass,
then relation balance. Self-cycle, VQAScore margin, and VSR are retention
signals rather than the primary selection objective.

| Branch | Route | Current | Current GenEval image/prompt | Spatial-preferred | Preferred image/prompt |
|---|---|---:|---:|---:|---:|
| S1-M | G-adapter | 5000 | 19.44 / 8.33 | 5000 | 19.44 / 8.33 |
| S1-MF | G-adapter | 6000 | 20.14 / 9.72 | 6000 | 20.14 / 9.72 |
| S1-M | LoRA | 6000 | 24.31 / 12.50 | 8000 | 25.00 / 12.50 |
| S1-MF | LoRA | 6000 | 23.96 / 12.50 | 8000 | 25.00 / 13.89 |

Baseline is 20.14 image accuracy and 8.33 prompt pass. The G-adapter starts are
the best spatial points within their branches, although S1-M step-5000 remains
slightly below baseline image accuracy. Both LoRA step-8000 checkpoints are
strictly better spatial starts than step-6000. The S1-M LoRA step-8000 also
improves compact self-cycle over step-6000 (53.01 vs 51.81 accuracy). S1-MF
LoRA step-8000 trades lower VQAScore margin and VSR for better GenEval, which is
the correct trade under a spatial-generation-first objective.

## Formal-training constraints

- Retain all four branches only if the experiment must estimate interactions
  between M/MF and G-adapter/LoRA. Use an exact no-cycle control from every
  selected start so the self-cycle delta is causal.
- If compute is constrained and the objective is best spatial generation,
  prioritize the two LoRA step-8000 starts.
- Do not select and claim the final model on GenEval alone. Report an independent
  spatial benchmark such as VISOR or a held-out GQA/spatial split.
- TIFA's natural pool is not spatial-dominant. Formal training should stratify
  spatial-bearing prompts and balance horizontal/vertical relations while
  preserving multidimensional questions and M/MF replay.
