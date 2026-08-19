# C9 SR2D/VISOR External Evaluation

This module evaluates whether GenEval/Mask2Former-supervised adapters transfer
to an independent prompt distribution and detector. It pins Microsoft VISOR at
commit `b9e368b73422aad655cc7121648beaca5c94563b` and uses the official
`google/owlvit-base-patch32` query template, score threshold `0.1`, any-box-pair
centroid geometry, and OA/VISOR metrics.

The official SR2D file contains 31,680 rows: 25,280 two-object directional
prompts, 6,320 `and` controls, and 80 single-object prompts. The directional
split contains 3,160 unordered object pairs, both subject/object orders, and
four relations. The official full four-sample protocol requires 101,120 images
per model. This module therefore also provides a pinned pair-complete 200-prompt
evaluation while retaining full-protocol support.

## Local resources

```text
/root/autodl-tmp/visor/
  README.md
  LICENSE
  create_spatial_phrases.py
  text_spatial_rel_phrases.json
  text2image_realtime_evaluate.ipynb
  visor.py

/root/autodl-tmp/models/visor/owlvit-base-patch32/
  model.safetensors
  config/tokenizer/preprocessor files

/root/autodl-tmp/data/visor/
  sr2d_pair1_smoke.jsonl
  sr2d_balanced200_seed20260811.jsonl
  sr2d_full_relations.jsonl
  corresponding *.audit.json files
```

The 200-prompt manifest selects 25 complete object-pair groups. Every group
contains both object orders and all four directions, so direction and swap
comparisons remain paired. No VISOR prompt is used for training or checkpoint
selection.

## Run one variant

```bash
RUN_DIR=/root/autodl-tmp/runs/showo2_visor/baseline \
VARIANT_NAME=baseline \
evaluation/c9_visor/run_variant.sh
```

For an adapter checkpoint:

```bash
RUN_DIR=/root/autodl-tmp/runs/showo2_visor/mask-flow-500 \
VARIANT_NAME=mask_flow_500 \
DRSA_CHECKPOINT=/absolute/path/to/step-000500.pt \
evaluation/c9_visor/run_variant.sh
```

Generation is resumable and writes official-style `<unique_id>_<sample>.png`
files. Scoring is separately resumable and writes `image_results.jsonl`,
`official_detection_results.json`, and `summary.json`.

## Paired comparison

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python \
  evaluation/c9_visor/compare_variants.py \
  --control /root/autodl-tmp/runs/showo2_visor/baseline/owlvit/image_results.jsonl \
  --candidate /root/autodl-tmp/runs/showo2_visor/mask-flow-500/owlvit/image_results.jsonl \
  --output /root/autodl-tmp/runs/showo2_visor/baseline-vs-mask-flow-500.json
```

The comparison reports paired joint/object gains and losses, per-direction
joint deltas, and prompt-cluster bootstrap intervals for OA, VISOR conditional,
and VISOR unconditional deltas.

## Interpretation boundary

OWL-ViT is independent of the GenEval Mask2Former used for training, but VISOR
still uses detector geometry and its official any-pair rule can accept one of
several duplicate detections. Report OA, conditional relation accuracy,
unconditional success, per-direction results, and paired gain/loss together.
Use Self-cycle, DPG-Bench, and an independent VLM or human audit as additional
retention checks.
