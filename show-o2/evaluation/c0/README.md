# Show-o2 C0 benchmark

This directory provides one reproducible entry point for Show-o2 understanding
and text-to-image alignment evaluation.

## Required assets

- Show-o2 source: `SHOWO2_ROOT`
- lmms-eval 0.7.2 source: `LMMS_EVAL_ROOT`
- Show-o2 checkpoint or Hugging Face id: `SHOWO2_MODEL`
- Wan 2.1 VAE checkpoint: `SHOWO2_VAE_PATH`
- GenEval source: `GENEVAL_ROOT`
- A CUDA GPU for every real smoke or benchmark run

Build the isolated environment once:

```bash
./lmms_eval_plugin/setup_eval_env.sh
```

The setup defaults to PyTorch 2.7.1 and torchvision 0.22.1 from the official
CUDA 12.8 wheel index. This is required for RTX 6000D (`sm_120`) and remains
compatible with the pinned Show-o2 Python stack. Override `TORCH_INDEX_URL`,
`TORCH_SPEC`, and `TORCHVISION_SPEC` only when targeting a different CUDA
runtime. Full preflight executes a BF16 CUDA matrix multiplication so an
unsupported GPU architecture cannot pass based only on `torch.cuda.is_available()`.

A host without a GPU can validate installation and request wiring:

```bash
./evaluation/c0/c0_benchmark.sh smoke dry
```

After attaching a GPU and placing `Wan2.1_VAE.pth`, run one real local-image
inference before downloading any benchmark dataset:

```bash
SHOWO2_VAE_PATH=/path/to/Wan2.1_VAE.pth \
  ./evaluation/c0/c0_benchmark.sh smoke real
```

## Benchmark commands

```bash
# Understanding: MME, GQA, SEEDBench, MMBench-dev, MMMU-val, MMStar, AI2D
./evaluation/c0/c0_benchmark.sh understanding

# Generate and score GenEval + DPG-Bench
./evaluation/c0/c0_benchmark.sh generation

# Full C0 followed by a tar.gz archive
./evaluation/c0/c0_benchmark.sh all
```

Useful overrides include `TASKS`, `LIMIT`, `NUM_PROCESSES`, `OUTPUT_ROOT`,
`WEIGHT_TYPE`, `MAX_NEW_TOKENS`, and `CUDA_VISIBLE_DEVICES`. Generation scoring
also requires the official evaluator dependencies: set `GENEVAL_PYTHON` and
`GENEVAL_MODEL_PATH` for GenEval; install ModelScope/mPLUG for DPG-Bench.

To generate samples without those two scoring environments:

```bash
./evaluation/c0/c0_benchmark.sh generation-samples
```

Archive an existing completed run:

```bash
RUN_DIR=/path/to/c0-run ./evaluation/c0/c0_benchmark.sh archive
```
