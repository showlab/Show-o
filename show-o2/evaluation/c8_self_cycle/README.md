# C8 Show-o2 Self-Cycle Benchmark

This benchmark measures whether one Show-o2 checkpoint can recover structured
facts from images produced by its own T2I path. The MMU request receives only
the saved PNG and the question; the original generation prompt is withheld.

The scorer does not use an API or an LLM judge:

- multiple-choice and yes/no answers use candidate token log likelihood;
- short open answers use deterministic greedy decoding;
- results use local normalization, exact match, and token F1;
- images are reloaded from PNG and encoded with the deterministic VAE posterior mean.

## Resources

The downloader pins UniCycle commit
`aa1cf81f791e9053e3623eabff0cbe989c614fbf` and validates Git blob IDs,
row counts, and JSONL structure. The upstream repository does not declare a
license in its repository metadata at this commit, so the downloaded data is
marked for research evaluation only.

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python \
  evaluation/c8_self_cycle/download_resources.py
```

## Tests And Smoke

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python -m unittest \
  evaluation.c8_self_cycle.test_scoring

RUN_DIR=/root/autodl-tmp/runs/showo2_c8/c8-self-cycle-smoke \
LIMIT_PROMPTS=1 NUM_INFERENCE_STEPS=4 GENERATION_BATCH_SIZE=1 \
  evaluation/c8_self_cycle/run_baseline.sh
```

## Baseline

```bash
evaluation/c8_self_cycle/start_baseline_tmux.sh
```

Monitor the latest baseline:

```bash
/root/autodl-tmp/venvs/showo2-eval/bin/python \
  evaluation/c8_self_cycle/monitor_self_cycle.py --watch
```

`overall` counts generated images and answered questions as benchmark work
units. `generation` and `structured` are also reported separately so phase
weighting is explicit.

## Strict Caption Cycle

The structured-QA run is an auxiliary QA-cycle diagnostic. The strict cycle is
evaluated separately as `text -> image -> reconstructed caption`. It reuses the
saved baseline PNG files, withholds the original prompt from Show-o2, and uses
a pinned local `all-mpnet-base-v2` snapshot plus lexical metrics for scoring.

```bash
evaluation/c8_self_cycle/start_caption_cycle_baseline_tmux.sh

/root/autodl-tmp/venvs/showo2-eval/bin/python \
  evaluation/c8_self_cycle/monitor_caption_cycle.py --watch
```
