# DRSA-Adapter Data and Geometry Protocol

This directory prepares the first training stage for the Dual-Route Spatial
Alignment Adapter (DRSA-Adapter). It does not modify the Show-o2 checkpoint or
the existing C0/C3 evaluation code.

## Data contract

The initial training recipe uses only released datasets:

- `SPRIGHT-T2I/spright_coco`: about 40K COCO images with both
  `spatial_caption` and `original_caption`. The 11 WebDataset tar shards occupy
  about 9.19 GiB and are consumed without extraction.
- `albertvillanova/visual-spatial-reasoning`: the VSR zero-shot split with
  image, caption, binary label, relation, subject, and object fields. The image
  archive occupies about 1.0 GiB.

SugarCrepe, What'sUp, VISOR, and GenEval remain evaluation-only. They are not
downloaded by this script and must not be mixed into adapter training.

Run the full resumable download in tmux:

```bash
cd /root/autodl-tmp/Show-o/show-o2
./evaluation/c4_drsa_adapter/start_drsa_data_tmux.sh
```

The launcher deliberately uses `https://hf-mirror.com` even if the parent
shell has a stale `HF_ENDPOINT`. Set `DRSA_HF_ENDPOINT` to override it.
Set `WAIT_FOR_STATUS` to the first downloader's `status.json` when a full run
must start after a small download. The full run proceeds after either
`completed` or `failed`, so a failed smoke or failed first download never
prevents the resumable full downloader from filling missing files. `WAIT_FOR`
is available only for workflows that genuinely require a successful marker.

Monitor exact bytes, including partial files:

```bash
./evaluation/c4_drsa_adapter/monitor_drsa_data.sh
```

Attach to the persistent shell:

```bash
tmux attach -t showo2_drsa_data
```

For a small network and loader smoke, use `SCOPE=smoke`. This downloads the
smallest SPRIGHT shard plus the VSR image archive and annotations:

```bash
SCOPE=smoke SESSION_NAME=showo2_drsa_data_smoke \
  ./evaluation/c4_drsa_adapter/start_drsa_data_tmux.sh
```

## Position mechanism decision

DRSA does not add a fixed `(x, y)` vector to every image hidden state. Direct
addition entangles geometry with semantic values, introduces an absolute-grid
bias, and duplicates the position information already present in Show-o2.

The first adapter version instead uses two complementary mechanisms inside the
adapter attention only:

1. Axial 2D RoPE rotates image-token Q/K channels separately with height and
   width coordinates. Values and residual hidden states remain coordinate-free.
2. A zero-initialized relation-conditioned continuous relative position bias
   modulates attention logits as a function of normalized `(delta_x, delta_y)`.

This choice follows the useful properties demonstrated by Qwen2-VL M-RoPE,
RoPE-ViT, FiT, and Swin Transformer V2:

- relative rather than absolute geometry;
- arbitrary grid sizes and aspect ratios;
- no learned table tied to 27x27 tokens;
- no coordinate contamination of token values;
- minimal extra compute and strict checkpoint parity when the adapter gate is
  disabled.

The relation bias should use antisymmetric directional features so reversing
`delta_x` or `delta_y` can reverse left/right or above/below evidence. The base
Qwen and diffusion-head RoPE implementations remain unchanged. This isolates
all new geometry inside DRSA and avoids invalidating pretrained weights.

## Training stages

The first implementation should train in this order:

1. Warm up the understanding adapter and shared relation bottleneck on VSR
   zero-shot train.
2. Train the generation adapter on SPRIGHT spatial captions while using the
   original captions for velocity retention.
3. Align image-side and prompt-side relation codes on paired SPRIGHT examples.
4. Evaluate on held-out VSR, SugarCrepe, What'sUp, VISOR, and GenEval before
   adding preference optimization or Qwen LoRA.

No spatially focused, reliable winner/loser T2I dataset was found at a useful
scale. The first DRSA stage therefore uses flow supervision and cross-route
alignment, not DPO.

`datasets/drsa_dataset.py` provides the native PyTorch data layer. SPRIGHT is
an `IterableDataset` that partitions tar shards across DDP ranks and data
workers, supports deterministic shard/buffer shuffle, and returns both caption
variants. VSR reads images lazily from the released zip and preserves the
structured relation, subject, object, and label fields.
