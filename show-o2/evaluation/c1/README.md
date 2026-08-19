# Show-o2 C1 gradient-conflict probe

C1 reuses the official Show-o2 Stage-2 training path to measure task-gradient
interference. It does not infer conflict from unrelated benchmark scores.

Each scored GenEval image/prompt pair is formatted twice with the official
`ImageTextDataset` sequence functions: image-conditioned captioning provides
the MMU next-token-prediction loss, while text-to-image training provides the
flow-matching loss. A third text-only next-token loss isolates language versus
visual-task gradients in the shared Qwen backbone.

The probe reports gradient dot products, cosine similarity, conflict rate,
gradient-norm dominance, and cancellation ratio for Fusion, the two visual
branches, every Qwen attention/MLP block, and early/middle/late Qwen scopes.
A strict conflict is a negative dot product. The Fusion result is meaningful
for MMU versus T2I; text-only inputs do not traverse Fusion.

Run one paired sample:

```bash
PER_TAG=1 LIMIT=1 RUN_ID=c1-smoke ./evaluation/c1/run_probe.sh
```

Start the default stratified run (8 samples per GenEval category, 48 total):

```bash
./evaluation/c1/start_probe_tmux.sh
```

Monitor the latest run every five seconds:

```bash
./evaluation/c1/monitor_probe.sh "" 5
```

Outputs are appended to `samples.jsonl` so the same run directory can resume
after interruption. `summary.json` preserves all strata and bootstrap
confidence intervals; `summary.md` contains the core tables.
