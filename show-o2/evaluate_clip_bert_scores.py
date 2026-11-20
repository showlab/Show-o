#!/usr/bin/env python3
"""
Evaluate Show-o2 × Qwen3-VL generated outputs with:

1) CLIPScore:
      image <-> generated Observation
   CLIPScore = 2.5 * max(
       cosine(image_embedding, observation_embedding),
       0
   )

2) BERTScore:
      generated Observation <-> PaliGemma pseudo-reference caption

The complete generated response is preserved in the output,
but CLIPScore and BERTScore evaluate only the Observation component.


Supported prediction formats
----------------------------
A) Flat CSV / JSONL / JSON rows, e.g.
{
  "image": "/path/to/frame.jpg",
  "stage": "stage2",
  "alpha": 0.6,
  "prompt_id": "prompt_2",
  "response": "..."
}

B) The nested ablation JSON format used in the experiments, e.g.
{
  "image": "/path/to/frame.jpg",
  "stage": "stage2",
  "prompt_file": "prompts/prompt2.txt",
  "results": [
    {"alpha": 0.0, "response": "..."},
    {"alpha": 0.1, "response": "..."}
  ]
}

PaliGemma pseudo-reference formats
----------------------------------
CSV / JSONL / JSON rows containing an image key and a caption, e.g.
{"image": "/path/to/frame.jpg", "caption": "Two people walk near a car."}

The loader auto-detects common field names.

Example
-------
# CLIPScore only (no pseudo-reference captions required)
python evaluate_clip_bertscore.py \
    --metrics clip \
    --predictions ./evaluation_outputs \
    --image-root /data/UCF_frames \
    --output-dir ./metric_results

# BERTScore only (does not load CLIP/LongCLIP/SPECS or images)
python evaluate_clip_bertscore.py \
    --metrics bert \
    --predictions ./evaluation_outputs \
    --paligemma ./captions_verified_paligemma.jsonl \
    --output-dir ./metric_results

# CLIPScore + BERTScore
python evaluate_clip_bertscore.py \
    --metrics clip bert \
    --predictions ./evaluation_outputs \
    --paligemma ./captions_vlm_paligemma.jsonl \
    --image-root /data/UCF_frames \
    --output-dir ./metric_results

Outputs
-------
metric_results/
    per_sample_metrics.csv
    summary_by_stage_alpha_prompt.csv
    run_metadata.json

Notes
-----
* Qwen-generated pseudo-captions are NOT used for BERTScore.
* BERTScore summaries also report Observation length, PaliGemma-caption length,
  and their word-count ratio for each stage × alpha × prompt configuration.
* PaliGemma captions are treated as pseudo-references, not ground truth.
* CLIPScore is reference-free and can be run without --paligemma.
* Use --metrics clip for CLIP-only evaluation.
* Standard CLIP text encoders have a short context window (typically 77 tokens).
  The script records whether each full generated response was truncated by CLIP.
  The original response is still kept unchanged in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


IMAGE_KEYS = (
    "image", "image_path", "img_path", "path", "file", "filename",
    "image_file", "image_name",
)
TEXT_KEYS = (
    "response", "output", "generated_text", "prediction", "answer",
    "caption", "text",
)
REF_TEXT_KEYS = (
    "caption", "generated_caption", "paligemma_caption", "text",
    "response", "output",
)
PROMPT_ID_KEYS = ("prompt_id", "prompt_name", "prompt_file", "prompt")
STAGE_KEYS = ("stage", "training_stage", "ft_stage")
ALPHA_KEYS = (
    "alpha",
    "fusion_alpha",
    "alpha_qwen",
)


def first_value(d: dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    return default


def clean_text(x) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def extract_observation_text(text: str) -> str:
    """
    Extract all Observation sections from a structured response.

    Supports outputs such as:

        Observation: ...
        Significance: ...
        Risk: ...
        Confidence: ...

    and repeated findings containing multiple Observation sections.
    """

    if not text:
        return ""

    # Remove common Markdown bold markers.
    s = str(text).replace("**", "").replace("__", "")

    pattern = re.compile(
        r"""
        ^\s*
        (?:\#{1,6}\s*)?
        (?:[-*]\s*)?
        (?:\d+[\.\)]\s*)?
        Observations?\s*:\s*

        (.*?)

        (?=
            ^\s*
            (?:\#{1,6}\s*)?
            (?:[-*]\s*)?
            (?:\d+[\.\)]\s*)?
            (?:
            Significance
            |
            Possible\s+investigative/security\s+significance
            |
            Risk(?:\s+level)?
            |
            Confidence(?:\s+level)?
            |
            Observations?
            )
            \s*:
            |
            \Z
        )
        """,
        re.IGNORECASE
        | re.MULTILINE
        | re.DOTALL
        | re.VERBOSE,
    )

    observations = [
        clean_text(match.group(1))
        for match in pattern.finditer(s)
    ]

    observations = [
        x for x in observations if x
    ]

    return " ".join(observations)

def normalize_path_string(x: str) -> str:
    return os.path.normpath(str(x)).replace("\\", "/").lower()


def canonical_basename(x: str) -> str:
    return os.path.basename(normalize_path_string(x))


def prompt_id_from_value(value, source_file: Path) -> Tuple[str, str]:
    """
    Returns (prompt_id, prompt_text).
    If value looks like a filename, use its basename as the ID.
    If it is full prompt text, create a stable hash-based ID and preserve text.
    """
    if value is None or str(value).strip() == "":
        return source_file.stem, ""

    s = str(value).strip()
    if "\n" not in s and len(s) < 220 and (
        s.endswith(".txt") or "/" in s or "\\" in s
    ):
        return Path(s).name, ""

    # If it is a short symbolic name, keep it.
    if "\n" not in s and len(s) <= 80:
        return s, ""

    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"prompt_{digest}", s


def infer_stage(obj: dict, source_file: Path) -> str:
    stage = first_value(obj, STAGE_KEYS)
    if stage is not None:
        return str(stage)

    candidates = [
        str(source_file),
        str(obj.get("checkpoint", "")),
        str(obj.get("adapter_checkpoint", "")),
    ]
    blob = " ".join(candidates).lower()
    if (
            "pre_stage1" in blob
            or "pre-stage1" in blob
            or "pre_stage-1" in blob
    ):
        return "pre_stage1"

    if "stage2" in blob or "stage_2" in blob or "stage-2" in blob:
        return "stage2"

    if "stage1" in blob or "stage_1" in blob or "stage-1" in blob:
        return "stage1"
    return "unknown"


def expand_prediction_object(obj: dict, source_file: Path) -> Iterator[dict]:
    """
    Expand either a flat prediction object or a nested object with `results`.
    """
    if not isinstance(obj, dict):
        return

    top_image = first_value(obj, IMAGE_KEYS)
    top_stage = infer_stage(obj, source_file)
    top_prompt_value = first_value(obj, PROMPT_ID_KEYS)
    top_prompt_id, top_prompt_text = prompt_id_from_value(
        top_prompt_value, source_file
    )

    # Our ablation JSON stores:
    #   prompt_file -> prompt identifier/path
    #   prompt      -> full prompt text
    explicit_top_prompt_text = first_value(obj, ("prompt_text",))

    if (
            explicit_top_prompt_text is None
            and first_value(obj, ("prompt_file",)) is not None
            and first_value(obj, ("prompt",)) is not None
    ):
        explicit_top_prompt_text = first_value(obj, ("prompt",))

    if explicit_top_prompt_text is not None:
        top_prompt_text = clean_text(explicit_top_prompt_text)

    results = obj.get("results")

    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict):
                continue

            image = first_value(r, IMAGE_KEYS, top_image)
            response = first_value(r, TEXT_KEYS)
            alpha = first_value(r, ALPHA_KEYS)
            stage = str(first_value(r, STAGE_KEYS, top_stage))

            prompt_val = first_value(r, PROMPT_ID_KEYS, top_prompt_value)
            prompt_id, prompt_text = prompt_id_from_value(
                prompt_val, source_file
            )

            nested_prompt_text = first_value(r, ("prompt_text",))

            if nested_prompt_text is not None:
                prompt_text = clean_text(nested_prompt_text)
            elif not prompt_text:
                prompt_text = top_prompt_text

            if image is None or response is None:
                continue

            yield {
                "image": str(image),
                "stage": stage,
                "alpha": alpha,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,

                # Keep complete answer.
                "generated_text": clean_text(response),

                # Text actually used by CLIPScore/BERTScore.
                "observation_text": extract_observation_text(response),

                "source_file": str(source_file),
            }
        return

    # Flat row
    image = first_value(obj, IMAGE_KEYS)
    response = first_value(obj, TEXT_KEYS)
    if image is None or response is None:
        return

    alpha = first_value(obj, ALPHA_KEYS)
    prompt_val = first_value(obj, PROMPT_ID_KEYS)
    prompt_id, prompt_text = prompt_id_from_value(prompt_val, source_file)

    explicit_prompt_text = first_value(obj, ("prompt_text",))

    if (
            explicit_prompt_text is None
            and first_value(obj, ("prompt_file",)) is not None
            and first_value(obj, ("prompt",)) is not None
    ):
        explicit_prompt_text = first_value(obj, ("prompt",))

    if explicit_prompt_text is not None:
        prompt_text = clean_text(explicit_prompt_text)

    yield {
        "image": str(image),
        "stage": infer_stage(obj, source_file),
        "alpha": alpha,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "generated_text": clean_text(response),
        "observation_text": extract_observation_text(response),
        "source_file": str(source_file),
    }


def iter_json_file(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield from expand_prediction_object(item, path)
    elif isinstance(data, dict):
        yield from expand_prediction_object(data, path)


def iter_jsonl_file(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] JSONL parse error: {path}:{line_no}: {e}",
                    file=sys.stderr,
                )
                continue
            if isinstance(obj, dict):
                yield from expand_prediction_object(obj, path)


def iter_csv_file(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield from expand_prediction_object(dict(row), path)


def prediction_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    files = []
    for ext in ("*.json", "*.jsonl", "*.csv"):
        files.extend(path.rglob(ext))
    return sorted(set(files))


def iter_prediction_records(path: Path) -> Iterator[dict]:
    files = prediction_files(path)
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL/CSV prediction files found in {path}")

    for p in files:
        suffix = p.suffix.lower()
        try:
            if suffix == ".json":
                yield from iter_json_file(p)
            elif suffix == ".jsonl":
                yield from iter_jsonl_file(p)
            elif suffix == ".csv":
                yield from iter_csv_file(p)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}", file=sys.stderr)


def split_reference_captions(value) -> List[str]:
    """
    UCF annotation format:
        caption 1 | caption 2 | caption 3 | ...

    Also accepts a list for future compatibility.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split("|")

    return [
        clean_text(x)
        for x in parts
        if clean_text(x)
    ]


def iter_reference_rows(path: Path) -> Iterator[dict]:
    """
    Load reference captions.

    Supports both:

    1) Old one-caption-per-row format:
       image, caption

    2) UCF format:
       image_name, category, ref_captions, color_type

       where ref_captions contains:
           caption 1 | caption 2 | caption 3 | ...
    """

    files = prediction_files(path)

    if not files:
        raise FileNotFoundError(
            f"No JSON/JSONL/CSV reference files found in {path}"
        )

    for p in files:
        try:

            if p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                rows = (
                    data
                    if isinstance(data, list)
                    else [data]
                )

            elif p.suffix.lower() == ".jsonl":
                rows = []

                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(
                                json.loads(line)
                            )

            elif p.suffix.lower() == ".csv":
                with p.open(
                    "r",
                    encoding="utf-8-sig",
                    newline="",
                ) as f:
                    rows = list(
                        csv.DictReader(f)
                    )

            else:
                continue

            for row in rows:

                if not isinstance(row, dict):
                    continue

                image = first_value(
                    row,
                    IMAGE_KEYS,
                )

                if image is None:
                    continue

                # ---------------------------------------------
                # New UCF format
                # ---------------------------------------------
                if row.get("ref_captions") not in (
                    None,
                    "",
                ):
                    captions = split_reference_captions(
                        row["ref_captions"]
                    )

                # ---------------------------------------------
                # Old single-reference format
                # ---------------------------------------------
                else:
                    caption = first_value(
                        row,
                        REF_TEXT_KEYS,
                    )

                    captions = (
                        [clean_text(caption)]
                        if caption is not None
                        else []
                    )

                n_refs = len(captions)

                for ref_index, caption in enumerate(
                    captions,
                    start=1,
                ):
                    yield {
                        "image": str(image),
                        "caption": caption,

                        # Useful for checking the expansion.
                        "reference_index": ref_index,
                        "reference_count": n_refs,

                        # Preserve UCF metadata.
                        "category": clean_text(
                            row.get("category", "")
                        ),
                        "color_type": clean_text(
                            row.get("color_type", "")
                        ),
                    }

        except Exception as e:
            print(
                f"[WARN] Skipping reference file {p}: {e}",
                file=sys.stderr,
            )


class ReferenceIndex:
    """
    One image -> LIST of reference captions.
    """

    def __init__(
        self,
        rows: Iterable[dict],
    ):
        self.exact: Dict[str, List[dict]] = defaultdict(list)

        # basename -> set of actual image keys.
        #
        # Important:
        # multiple captions for ONE image must NOT make the
        # basename ambiguous.
        basename_keys: Dict[str, set] = defaultdict(set)

        for row in rows:

            image = row["image"]

            key = normalize_path_string(
                image
            )

            self.exact[key].append(
                row
            )

            basename_keys[
                canonical_basename(image)
            ].add(key)

        # A basename is safe for fallback if it corresponds
        # to exactly one image, regardless of how many captions
        # that image has.
        self.basename_to_key = {
            basename: next(iter(keys))
            for basename, keys in basename_keys.items()
            if len(keys) == 1
        }

        self.ambiguous_basenames = {
            basename
            for basename, keys in basename_keys.items()
            if len(keys) > 1
        }

    def get_all(
        self,
        image: str,
    ) -> List[dict]:

        key = normalize_path_string(
            image
        )

        exact = self.exact.get(
            key
        )

        if exact is not None:
            return exact

        basename = canonical_basename(
            image
        )

        fallback_key = self.basename_to_key.get(
            basename
        )

        if fallback_key is None:
            return []

        return self.exact.get(
            fallback_key,
            [],
        )


class NullReferenceIndex:
    """Reference lookup used when BERTScore is disabled."""

    ambiguous_basenames = set()

    def get_all(
        self,
        image: str,
    ) -> List[dict]:
        return []





class ImageResolver:
    def __init__(self, image_root: Optional[Path]):
        self.image_root = image_root
        self.basename_index: Dict[str, Optional[Path]] = {}

        if image_root is not None:
            print(f"Indexing images under: {image_root}")
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            for p in tqdm(image_root.rglob("*"), desc="Image index"):
                if not p.is_file() or p.suffix.lower() not in image_exts:
                    continue
                key = p.name.lower()
                if key not in self.basename_index:
                    self.basename_index[key] = p
                else:
                    # Duplicate basename -> don't guess.
                    self.basename_index[key] = None

    def resolve(self, image_string: str) -> Optional[Path]:
        p = Path(image_string)
        if p.exists():
            return p

        if self.image_root is not None:
            candidate = self.image_root / image_string
            if candidate.exists():
                return candidate

            candidate = self.basename_index.get(p.name.lower())
            if candidate is not None and candidate.exists():
                return candidate

        return None


def batched(iterator: Iterable[dict], batch_size: int) -> Iterator[List[dict]]:
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def device_from_arg(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


class StandardCLIPScorer:
    """
    Standard CLIPScore:
        2.5 * max(cosine(E_image, E_text), 0)

    Image features are cached on CPU because the same UCF frame is evaluated
    under many stages / alphas / prompts.
    """
    def __init__(
        self,
        model_name: str,
        device: str,
        image_batch_size: int = 32,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model_name = model_name
        self.image_batch_size = image_batch_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.cache: Dict[str, torch.Tensor] = {}

        text_cfg = getattr(self.model.config, "text_config", None)
        self.max_text_tokens = int(
            getattr(text_cfg, "max_position_embeddings", 77)
        )

    @torch.inference_mode()
    def _encode_missing_images(
        self,
        image_pairs: List[Tuple[str, Path]],
    ) -> None:
        # Deduplicate within batch and against cache.
        unique = {}
        for key, path in image_pairs:
            if key not in self.cache:
                unique[key] = path

        items = list(unique.items())
        for start in range(0, len(items), self.image_batch_size):
            chunk = items[start:start + self.image_batch_size]

            pil_images = []
            valid_keys = []
            for key, path in chunk:
                try:
                    with Image.open(path) as im:
                        pil_images.append(im.convert("RGB").copy())
                    valid_keys.append(key)
                except (OSError, UnidentifiedImageError) as e:
                    print(f"[WARN] Could not read image {path}: {e}", file=sys.stderr)

            if not pil_images:
                continue

            inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.device)

            feats = self.model.get_image_features(pixel_values=pixel_values)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats.float(), dim=-1).cpu()

            for key, feat in zip(valid_keys, feats):
                self.cache[key] = feat

    @torch.inference_mode()
    def score_batch(
        self,
        records: List[dict],
        resolver: ImageResolver,
    ) -> List[dict]:
        image_pairs = []
        resolved = []

        for r in records:
            image_path = resolver.resolve(r["image"])
            resolved.append(image_path)
            if image_path is not None:
                key = str(image_path.resolve())
                image_pairs.append((key, image_path))

        self._encode_missing_images(image_pairs)

        texts = [r["observation_text"] for r in records]

        # Record token lengths before truncation.
        tokenized_full = self.processor.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
        )
        token_counts = [len(ids) for ids in tokenized_full["input_ids"]]

        text_inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        text_inputs = {
            k: v.to(self.device)
            for k, v in text_inputs.items()
            if k in ("input_ids", "attention_mask")
        }

        text_feats = self.model.get_text_features(**text_inputs)
        if hasattr(text_feats, "pooler_output"):
            text_feats = text_feats.pooler_output
        text_feats = torch.nn.functional.normalize(text_feats.float(), dim=-1)

        outputs = []
        for idx, (r, path, token_count) in enumerate(
            zip(records, resolved, token_counts)
        ):
            out = dict(r)
            out["resolved_image_path"] = str(path) if path else ""
            out["clip_token_count"] = token_count
            out["clip_was_truncated"] = token_count > self.max_text_tokens
            out["output_words"] = len(r["generated_text"].split())
            out["observation_words"] = len(r["observation_text"].split())

            if path is None:
                out["clip_score"] = math.nan
                out["error"] = "image_not_found"
                outputs.append(out)
                continue

            key = str(path.resolve())
            image_feat = self.cache.get(key)
            if image_feat is None:
                out["clip_score"] = math.nan
                out["error"] = "image_encoding_failed"
                outputs.append(out)
                continue

            image_feat = image_feat.to(self.device)
            cosine = torch.sum(image_feat * text_feats[idx]).item()

            out["clip_score"] = (
                    2.5 * max(cosine, 0.0)
            )

            out["error"] = ""
            outputs.append(out)

        return outputs

    def close(self):
        del self.model
        del self.processor
        self.cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class LongContextCLIPScorer:
    """
    Long-CLIP-based image/Observation score.

    Score:
        2.5 * max(cosine(E_image, E_observation), 0)
    """

    def __init__(
            self,
            checkpoint: str,
            repo_root: str,
            device: str,
            image_batch_size: int = 32,
            score_mode: str = "longclip",
    ):

        self.device = device
        self.checkpoint = checkpoint
        self.image_batch_size = image_batch_size
        self.score_mode = score_mode

        if score_mode not in (
                "longclip",
                "specs",
        ):
            raise ValueError(
                f"Unknown score mode: {score_mode}"
            )

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from model import longclip
        from model.simple_tokenizer import SimpleTokenizer

        self.longclip = longclip
        self.tokenizer = SimpleTokenizer()

        self.model, self.preprocess = (
            longclip.load(
                checkpoint,
                device=device,
            )
        )

        self.model.eval()

        self.max_text_tokens = int(
            self.model.positional_embedding.shape[0]
        )

        print(
            f"{score_mode} context length: "
            f"{self.max_text_tokens}"
        )

        self.cache = {}

    @torch.inference_mode()
    def _encode_missing_images(
        self,
        image_pairs: List[Tuple[str, Path]],
    ) -> None:

        unique = {}

        for key, path in image_pairs:
            if key not in self.cache:
                unique[key] = path

        items = list(unique.items())

        for start in range(
            0,
            len(items),
            self.image_batch_size,
        ):

            chunk = items[
                start:start + self.image_batch_size
            ]

            image_tensors = []
            valid_keys = []

            for key, path in chunk:

                try:
                    with Image.open(path) as im:
                        image = im.convert("RGB")
                        tensor = self.preprocess(image)

                    image_tensors.append(tensor)
                    valid_keys.append(key)

                except (
                    OSError,
                    UnidentifiedImageError,
                ) as e:

                    print(
                        f"[WARN] Could not read image "
                        f"{path}: {e}",
                        file=sys.stderr,
                    )

            if not image_tensors:
                continue

            pixel_values = torch.stack(
                image_tensors
            ).to(self.device)

            feats = self.model.encode_image(
                pixel_values
            )

            feats = (
                torch.nn.functional.normalize(
                    feats.float(),
                    dim=-1,
                )
                .cpu()
            )

            for key, feat in zip(
                valid_keys,
                feats,
            ):
                self.cache[key] = feat

    @torch.inference_mode()
    def score_batch(
        self,
        records: List[dict],
        resolver: ImageResolver,
    ) -> List[dict]:

        image_pairs = []
        resolved = []

        for r in records:

            image_path = resolver.resolve(
                r["image"]
            )

            resolved.append(image_path)

            if image_path is not None:
                key = str(
                    image_path.resolve()
                )
                image_pairs.append(
                    (key, image_path)
                )

        self._encode_missing_images(
            image_pairs
        )

        # Observation only, exactly like standard CLIP.
        texts = [
            r["observation_text"]
            for r in records
        ]

        # Count actual Long-CLIP tokens before truncation.
        token_counts = [
            len(
                self.tokenizer.encode(text)
            ) + 2
            for text in texts
        ]

        text_tokens = self.longclip.tokenize(
            texts,
            context_length=self.max_text_tokens,
            truncate=True,
        ).to(self.device)

        text_feats = self.model.encode_text(
            text_tokens
        )

        text_feats = (
            torch.nn.functional.normalize(
                text_feats.float(),
                dim=-1,
            )
        )

        outputs = []

        for idx, (
            r,
            path,
            token_count,
        ) in enumerate(
            zip(
                records,
                resolved,
                token_counts,
            )
        ):

            out = dict(r)

            out["resolved_image_path"] = (
                str(path) if path else ""
            )

            out["clip_token_count"] = (
                token_count
            )

            out["clip_was_truncated"] = (
                token_count
                > self.max_text_tokens
            )

            out["output_words"] = len(
                r["generated_text"].split()
            )

            out["observation_words"] = len(
                r["observation_text"].split()
            )

            if path is None:
                out["clip_score"] = math.nan
                out["error"] = "image_not_found"
                outputs.append(out)
                continue

            key = str(
                path.resolve()
            )

            image_feat = self.cache.get(key)

            if image_feat is None:
                out["clip_score"] = math.nan
                out["error"] = (
                    "image_encoding_failed"
                )
                outputs.append(out)
                continue

            image_feat = image_feat.to(
                self.device
            )

            image_feat = torch.nn.functional.normalize(
                image_feat.float(),
                dim=-1,
            )

            text_feats = torch.nn.functional.normalize(
                text_feats.float(),
                dim=-1,
            )

            cosine = torch.sum(
                image_feat
                * text_feats[idx]
            ).item()

            if self.score_mode == "specs":
                out["clip_score"] = max(
                    (cosine + 1.0) / 2.0,
                    0.0,
                )
            else:
                out["clip_score"] = (
                        2.5 * max(cosine, 0.0)
                )

            out["error"] = ""

            outputs.append(out)

        return outputs

    def close(self):

        del self.model

        self.cache.clear()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class JinaCLIPScorer:
    """
    Jina-CLIP-v2 image <-> Observation cosine similarity.

    Unlike CLIPScore/SPECS, we report raw cosine similarity.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        image_batch_size: int = 16,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.model_name = model_name
        self.image_batch_size = image_batch_size

        print(f"Loading Jina-CLIP-v2: {model_name}")

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Official advertised maximum.
        self.max_text_tokens = 8192

        self.cache = {}

        print(
            f"Jina-CLIP-v2 context length: "
            f"{self.max_text_tokens}"
        )

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x

        return torch.as_tensor(x)

    @torch.inference_mode()
    def _encode_missing_images(
        self,
        image_pairs: List[Tuple[str, Path]],
    ) -> None:

        unique = {}

        for key, path in image_pairs:
            if key not in self.cache:
                unique[key] = path

        items = list(unique.items())

        for start in range(
            0,
            len(items),
            self.image_batch_size,
        ):
            chunk = items[
                start:start + self.image_batch_size
            ]

            keys = [
                key
                for key, _ in chunk
            ]

            paths = [
                str(path)
                for _, path in chunk
            ]

            try:
                feats = self.model.encode_image(
                    paths,
                    truncate_dim=None,
                    batch_size=8
                )

                feats = self._to_tensor(feats)

                feats = torch.nn.functional.normalize(
                    feats.float(),
                    dim=-1,
                ).cpu()

                for key, feat in zip(keys, feats):
                    self.cache[key] = feat

            except Exception as exc:
                print(
                    f"[WARN] Jina image encoding failed: "
                    f"{exc}",
                    file=sys.stderr,
                )

    @torch.inference_mode()
    def score_batch(
        self,
        records: List[dict],
        resolver: ImageResolver,
    ) -> List[dict]:

        resolved = []
        image_pairs = []

        for r in records:

            image_path = resolver.resolve(
                r["image"]
            )

            resolved.append(image_path)

            if image_path is not None:

                key = str(
                    image_path.resolve()
                )

                image_pairs.append(
                    (key, image_path)
                )

        self._encode_missing_images(
            image_pairs
        )

        # Same component used by the other metrics.
        texts = [
            r["observation_text"]
            for r in records
        ]

        # Real Jina tokenizer counts, before truncation.
        token_counts = [
            len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in texts
        ]

        text_feats = self.model.encode_text(
            texts,
            truncate_dim=None,
            max_length=self.max_text_tokens,
            truncation=True,
            batch_size=8,
        )

        text_feats = self._to_tensor(
            text_feats
        )

        text_feats = torch.nn.functional.normalize(
            text_feats.float(),
            dim=-1,
        )

        outputs = []

        for idx, (
            r,
            path,
            token_count,
        ) in enumerate(
            zip(
                records,
                resolved,
                token_counts,
            )
        ):

            out = dict(r)

            out["resolved_image_path"] = (
                str(path)
                if path
                else ""
            )

            out["clip_token_count"] = (
                token_count
            )

            out["clip_was_truncated"] = (
                token_count
                > self.max_text_tokens
            )

            out["output_words"] = len(
                r["generated_text"].split()
            )

            out["observation_words"] = len(
                r["observation_text"].split()
            )

            if path is None:
                out["clip_score"] = math.nan
                out["error"] = "image_not_found"
                outputs.append(out)
                continue

            key = str(
                path.resolve()
            )

            image_feat = self.cache.get(
                key
            )

            if image_feat is None:
                out["clip_score"] = math.nan
                out["error"] = (
                    "image_encoding_failed"
                )
                outputs.append(out)
                continue

            image_feat = image_feat.to(
                text_feats.device
            )

            cosine = torch.sum(
                image_feat
                * text_feats[idx]
            ).item()

            # IMPORTANT:
            # raw Jina image-text cosine.
            out["clip_score"] = cosine

            out["error"] = ""

            outputs.append(out)

        return outputs

    def close(self):

        del self.model
        self.cache.clear()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def write_clip_pass(
    args,
    ref_index: ReferenceIndex,
    resolver: ImageResolver,
    temp_jsonl: Path,
) -> dict:

    if args.clip_backbone == "clip":

        scorer = StandardCLIPScorer(
            model_name=args.clip_model,
            device=args.device,
            image_batch_size=
            args.clip_image_batch_size,
        )

    elif args.clip_backbone == "longclip":

        scorer = LongContextCLIPScorer(
            checkpoint=args.longclip_checkpoint,
            repo_root=args.longclip_root,
            device=args.device,
            image_batch_size=
            args.clip_image_batch_size,
            score_mode="longclip",
        )

    elif args.clip_backbone == "specs":

        scorer = LongContextCLIPScorer(
            checkpoint=args.specs_checkpoint,
            repo_root=args.specs_root,
            device=args.device,
            image_batch_size=
            args.clip_image_batch_size,
            score_mode="specs",
        )

    elif args.clip_backbone == "jina":

        scorer = JinaCLIPScorer(
            model_name=args.jina_model,
            device=args.device,
            image_batch_size=args.clip_image_batch_size,
        )

    else:
        raise ValueError(
            f"Unknown backbone: "
            f"{args.clip_backbone}"
        )

    counters = defaultdict(int)
    total_files = len(prediction_files(args.predictions))
    print(f"Prediction files found: {total_files}")

    records = iter_prediction_records(args.predictions)

    with temp_jsonl.open("w", encoding="utf-8") as out_f:
        for batch in tqdm(
            batched(records, args.clip_text_batch_size),
            desc="CLIPScore",
            unit="batch",
        ):
            valid_batch = []

            for r in batch:

                counters["prediction_rows"] += 1

                if not r["generated_text"]:
                    counters["empty_predictions"] += 1
                    continue

                if not r.get("observation_text"):
                    counters["missing_observation"] += 1
                    continue

                refs = ref_index.get_all(
                    r["image"]
                )

                if not refs:

                    counters[
                        "missing_paligemma_reference"
                    ] += 1

                    if args.require_reference:
                        continue

                    refs = [
                        {
                            "caption": "",
                            "reference_index": 1,
                            "reference_count": 0,
                            "category": "",
                            "color_type": "",
                        }
                    ]

                # Do NOT duplicate the image/Observation here.
                #
                # CLIP does not depend on the reference caption, so score
                # the image/Observation only once and expand afterwards.
                row = dict(r)

                row["_reference_rows"] = refs

                valid_batch.append(
                    row
                )

            if not valid_batch:
                continue

            scored = scorer.score_batch(valid_batch, resolver)
            for base_row in scored:

                refs = base_row.pop(
                    "_reference_rows",
                    [],
                )

                if not refs:
                    refs = [
                        {
                            "caption": "",
                            "reference_index": 1,
                            "reference_count": 0,
                            "category": "",
                            "color_type": "",
                        }
                    ]

                # --------------------------------------------------------
                # Expand:
                #
                # image + caption 1
                # image + caption 2
                # image + caption 3
                # ...
                # --------------------------------------------------------
                for ref in refs:

                    row = dict(
                        base_row
                    )

                    row[
                        "paligemma_reference"
                    ] = ref["caption"]

                    row[
                        "reference_index"
                    ] = ref.get(
                        "reference_index"
                    )

                    row[
                        "reference_count"
                    ] = ref.get(
                        "reference_count"
                    )

                    row[
                        "reference_category"
                    ] = ref.get(
                        "category",
                        "",
                    )

                    row[
                        "reference_color_type"
                    ] = ref.get(
                        "color_type",
                        "",
                    )

                    if row.get(
                            "clip_was_truncated"
                    ):
                        counters[
                            "clip_truncated"
                        ] += 1

                    if row.get("error"):
                        counters[
                            row["error"]
                        ] += 1

                    out_f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    counters[
                        "clip_rows_written"
                    ] += 1

    scorer.close()
    return dict(counters)



def write_bert_input_pass(
    args,
    ref_index: ReferenceIndex,
    temp_jsonl: Path,
) -> dict:
    """
    Prepare normalized prediction rows for BERTScore without loading any
    CLIP/LongCLIP/SPECS model and without resolving/opening image files.

    This pass:
      - parses prediction files,
      - extracts the Observation component,
      - matches each prediction to its PaliGemma pseudo-reference,
      - writes the normalized rows to temp_jsonl.
    """

    counters = defaultdict(int)
    total_files = len(prediction_files(args.predictions))
    print(f"Prediction files found: {total_files}")

    records = iter_prediction_records(args.predictions)

    with temp_jsonl.open("w", encoding="utf-8") as out_f:

        for batch in tqdm(
            batched(records, args.bert_batch_size),
            desc="Preparing BERTScore inputs",
            unit="batch",
        ):

            for r in batch:
                counters["prediction_rows"] += 1

                if not r["generated_text"]:
                    counters["empty_predictions"] += 1
                    continue

                if not r.get("observation_text"):
                    counters["missing_observation"] += 1
                    continue

                refs = ref_index.get_all(
                    r["image"]
                )

                if not refs:

                    counters[
                        "missing_paligemma_reference"
                    ] += 1

                    if args.require_reference:
                        continue

                    refs = [
                        {
                            "caption": "",
                            "reference_index": 1,
                            "reference_count": 0,
                            "category": "",
                            "color_type": "",
                        }
                    ]

                for ref in refs:
                    row = dict(r)

                    row[
                        "paligemma_reference"
                    ] = ref["caption"]

                    row[
                        "reference_index"
                    ] = ref.get(
                        "reference_index"
                    )

                    row[
                        "reference_count"
                    ] = ref.get(
                        "reference_count"
                    )

                    row[
                        "reference_category"
                    ] = ref.get(
                        "category",
                        "",
                    )

                    row[
                        "reference_color_type"
                    ] = ref.get(
                        "color_type",
                        "",
                    )

                    # No CLIP in BERT-only mode.
                    row[
                        "resolved_image_path"
                    ] = ""

                    row[
                        "clip_score"
                    ] = math.nan

                    row[
                        "clip_token_count"
                    ] = math.nan

                    row[
                        "clip_was_truncated"
                    ] = False

                    row[
                        "output_words"
                    ] = len(
                        row[
                            "generated_text"
                        ].split()
                    )

                    row[
                        "observation_words"
                    ] = len(
                        row[
                            "observation_text"
                        ].split()
                    )

                    row[
                        "error"
                    ] = ""

                    out_f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )



                    counters["bert_input_rows_written"] += 1

    return dict(counters)


def read_jsonl_batches(path: Path, batch_size: int) -> Iterator[List[dict]]:
    batch = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


class RunningStats:
    def __init__(self):
        self.n = 0
        self.sums = defaultdict(float)
        self.sumsq = defaultdict(float)
        self.counts = defaultdict(int)
        self.truncated = 0

    def update(self, row: dict):
        self.n += 1

        for key in (
            "clip_score",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "output_words",
            "observation_words",
            "paligemma_words",
            "observation_paligemma_length_ratio",
        ):
            value = row.get(key)

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isnan(value):
                continue

            self.sums[key] += value
            self.sumsq[key] += value * value
            self.counts[key] += 1

        self.truncated += int(
            bool(
                row.get(
                    "clip_was_truncated",
                    False,
                )
            )
        )

    def result(self):
        out = {"n": self.n}

        for key in (
            "clip_score",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "output_words",
            "observation_words",
            "paligemma_words",
            "observation_paligemma_length_ratio",
        ):
            count = self.counts[key]

            if count == 0:
                mean = std = math.nan
            else:
                mean = self.sums[key] / count
                variance = max(
                    self.sumsq[key] / count
                    - mean * mean,
                    0.0,
                )
                std = math.sqrt(variance)

            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std

        out["clip_truncated_n"] = self.truncated
        out["clip_truncated_rate"] = (
            self.truncated / self.n
            if self.n
            else math.nan
        )

        return out


def normalise_alpha(x):
    if x is None or x == "":
        return "NA"
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)

def write_clip_dimension_summary(
    per_sample_csv: Path,
    output_csv: Path,
):
    """
    Create an additional CLIPScore summary with:

    1) mean/std per stage, averaged over all alphas, prompts and images
    2) mean/std per alpha, averaged over all stages, prompts and images
    3) mean/std per prompt, averaged over all stages, alphas and images
    """

    df = pd.read_csv(per_sample_csv)

    # Each image/Observation may appear once per reference caption.
    # CLIP is independent of the reference caption, so keep only
    # one copy of each actual CLIP evaluation.
    df = df.drop_duplicates(
        subset=[
            "image",
            "stage",
            "alpha",
            "prompt_id",
            "observation_text",
        ]
    )

    if "clip_score" not in df.columns:
        raise ValueError(
            "clip_score column not found in per-sample results."
        )

    # Make sure CLIPScore is numeric.
    df["clip_score"] = pd.to_numeric(
        df["clip_score"],
        errors="coerce",
    )

    # Only rows with a valid CLIPScore participate.
    df = df.dropna(
        subset=["clip_score"]
    )

    summary_parts = []

    # --------------------------------------------------------
    # 1. PER STAGE
    # --------------------------------------------------------

    stage_summary = (
        df.groupby(
            "stage",
            dropna=False,
        )["clip_score"]
        .agg(
            n="count",
            clip_score_mean="mean",
            clip_score_std=lambda x: x.std(ddof=0),
        )
        .reset_index()
        .rename(
            columns={
                "stage": "group_value"
            }
        )
    )

    stage_summary.insert(
        0,
        "group_type",
        "stage",
    )

    summary_parts.append(
        stage_summary
    )

    # --------------------------------------------------------
    # 2. PER ALPHA
    # --------------------------------------------------------

    alpha_summary = (
        df.groupby(
            "alpha",
            dropna=False,
        )["clip_score"]
        .agg(
            n="count",
            clip_score_mean="mean",
            clip_score_std=lambda x: x.std(ddof=0),
        )
        .reset_index()
        .rename(
            columns={
                "alpha": "group_value"
            }
        )
    )

    # Sort alpha numerically.
    alpha_summary["_alpha_numeric"] = pd.to_numeric(
        alpha_summary["group_value"],
        errors="coerce",
    )

    alpha_summary = (
        alpha_summary
        .sort_values(
            "_alpha_numeric",
            na_position="last",
        )
        .drop(
            columns=["_alpha_numeric"]
        )
    )

    alpha_summary.insert(
        0,
        "group_type",
        "alpha",
    )

    summary_parts.append(
        alpha_summary
    )

    # --------------------------------------------------------
    # 3. PER PROMPT
    # --------------------------------------------------------

    prompt_summary = (
        df.groupby(
            "prompt_id",
            dropna=False,
        )["clip_score"]
        .agg(
            n="count",
            clip_score_mean="mean",
            clip_score_std=lambda x: x.std(ddof=0),
        )
        .reset_index()
        .rename(
            columns={
                "prompt_id": "group_value"
            }
        )
    )

    prompt_summary.insert(
        0,
        "group_type",
        "prompt",
    )

    summary_parts.append(
        prompt_summary
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    summary_df = pd.concat(
        summary_parts,
        ignore_index=True,
    )

    summary_df.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"CLIP dimension summary saved to: "
        f"{output_csv}"
    )


def bertscore_pass(
    args,
    temp_jsonl: Path,
    final_csv: Path,
    summary_csv: Path,
) -> dict:
    from bert_score import BERTScorer

    print(f"Loading BERTScore model: {args.bert_model}")
    bert = BERTScorer(
        model_type=args.bert_model,
        lang="en",
        rescale_with_baseline=args.bert_rescale,
        device=args.device,
    )

    groups: Dict[Tuple[str, str, str], RunningStats] = defaultdict(RunningStats)
    counters = defaultdict(int)

    fieldnames = [
        "image",
        "resolved_image_path",
        "stage",
        "alpha",
        "prompt_id",
        "prompt_text",
        "generated_text",
        "observation_text",

        "reference_index",
        "reference_count",
        "reference_category",
        "reference_color_type",
        "paligemma_reference",

        "clip_score",
        "clip_token_count",
        "clip_was_truncated",
        "output_words",
        "observation_words",
        "paligemma_words",
        "observation_paligemma_length_ratio",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "source_file",
        "error",
    ]

    with final_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for batch in tqdm(
            read_jsonl_batches(temp_jsonl, args.bert_batch_size),
            desc="BERTScore",
            unit="batch",
        ):
            # BERTScore needs references. If --require-reference is false,
            # rows without references keep NaN BERTScore but retain CLIPScore.
            indices = [
                i for i, r in enumerate(batch)
                if (
                        clean_text(
                            r.get(
                                "paligemma_reference",
                                "",
                            )
                        )
                        and
                        clean_text(
                            r.get(
                                "observation_text",
                                "",
                            )
                        )
                )
            ]

            P_vals = [math.nan] * len(batch)
            R_vals = [math.nan] * len(batch)
            F_vals = [math.nan] * len(batch)

            if indices:
                cands = [batch[i]["observation_text"] for i in indices]
                refs = [batch[i]["paligemma_reference"] for i in indices]

                P, R, F1 = bert.score(
                    cands,
                    refs,
                    batch_size=args.bert_internal_batch_size,
                )

                for j, i in enumerate(indices):
                    P_vals[i] = float(P[j].cpu())
                    R_vals[i] = float(R[j].cpu())
                    F_vals[i] = float(F1[j].cpu())

            for i, row in enumerate(batch):
                row["bertscore_precision"] = P_vals[i]
                row["bertscore_recall"] = R_vals[i]
                row["bertscore_f1"] = F_vals[i]

                # Length statistics for the exact two texts compared by BERTScore.
                observation_words = len(
                    clean_text(
                        row.get("observation_text", "")
                    ).split()
                )
                paligemma_words = len(
                    clean_text(
                        row.get("paligemma_reference", "")
                    ).split()
                )

                row["observation_words"] = observation_words
                row["paligemma_words"] = paligemma_words
                row["observation_paligemma_length_ratio"] = (
                    observation_words / paligemma_words
                    if paligemma_words > 0
                    else math.nan
                )

                writer.writerow(row)
                counters["final_rows"] += 1

                key = (
                    str(row.get("stage", "unknown")),
                    normalise_alpha(row.get("alpha")),
                    str(row.get("prompt_id", "unknown")),
                )
                groups[key].update(row)

    summary_rows = []
    for (stage, alpha, prompt_id), stats in groups.items():
        row = {
            "stage": stage,
            "alpha": alpha,
            "prompt_id": prompt_id,
        }
        row.update(stats.result())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        # Numeric helper for sensible alpha ordering.
        summary_df["_alpha_numeric"] = pd.to_numeric(
            summary_df["alpha"], errors="coerce"
        )
        summary_df = summary_df.sort_values(
            ["stage", "prompt_id", "_alpha_numeric", "alpha"],
            na_position="last",
        ).drop(columns=["_alpha_numeric"])
    summary_df.to_csv(summary_csv, index=False)

    del bert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(counters)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--predictions",
        type=Path,
        #required=True,
        default='/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/results/ablation_results/crime_deepstack',
        help="Prediction file or directory containing JSON/JSONL/CSV outputs.",
    )
    p.add_argument(
        "--paligemma",
        type=Path,
        default='/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/one_frame_per_video_split/captions_florence_detailed.jsonl',
        help=(
            "PaliGemma pseudo-reference file/directory. Required only when "
            "BERTScore is requested."
        ),
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=("clip", "bert"),
        default=["clip", "bert"],
        help=(
            "Metrics to run. Examples: '--metrics clip' for image-text score only, "
            "'--metrics bert' for BERTScore only, or '--metrics clip bert' for both."
        ),
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default='/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/one_frame_per_video_split/images',
        help="Optional root directory used to resolve image paths/basenames.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        #required=True,
        default='/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/results/crime_deepstack_florence_less',
    )
    p.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )

    p.add_argument(
        "--clip-backbone",
        choices=("clip", "longclip", "specs", "jina"),
        default="longclip",
        help=(
            "Image-text backbone: "
            "'clip' = standard OpenAI CLIP, "
            "'longclip' = Long-CLIP."
            "'specs' = SPECS."
        ),
    )

    p.add_argument(
        "--jina-model",
        type=str,
        default="jinaai/jina-clip-v2",
        help="Hugging Face Jina-CLIP-v2 model.",
    )

    p.add_argument(
        "--specs-root",
        type=str,
        default=(
            "/home/chatziko/PycharmProjects/"
            "PythonProject/SPECS"
        ),
        help="Root directory of the SPECS repository.",
    )

    p.add_argument(
        "--specs-checkpoint",
        type=str,
        default=(
            "/home/chatziko/PycharmProjects/"
            "PythonProject/SPECS/"
            "checkpoints/spec.pt"
        ),
        help="Path to the official SPECS checkpoint.",
    )


    # CLIPScore
    p.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model.",
    )

    p.add_argument(
        "--longclip-root",
        type=str,
        default=(
            "/home/chatziko/PycharmProjects/"
            "PythonProject/Long-CLIP"
        ),
        help="Root directory of cloned Long-CLIP repository.",
    )

    p.add_argument(
        "--longclip-checkpoint",
        type=str,
        default=(
            "/home/chatziko/PycharmProjects/"
            "PythonProject/Long-CLIP/"
            "checkpoints/longclip-L.pt"
        ),
        help="Path to Long-CLIP checkpoint.",
    )

    p.add_argument(
        "--clip-text-batch-size",
        type=int,
        default=1024,
        help="Number of generated responses encoded per CLIP text batch.",
    )
    p.add_argument(
        "--clip-image-batch-size",
        type=int,
        default=512,
        help="Number of previously unseen images encoded per CLIP image batch.",
    )

    # BERTScore
    p.add_argument(
        "--bert-model",
        default="roberta-large",
        help=(
            "BERTScore backbone. For a heavier alternative, try "
            "'microsoft/deberta-xlarge-mnli'."
        ),
    )
    p.add_argument(
        "--bert-batch-size",
        type=int,
        default=1024,
        help="Rows read from the intermediate file at once.",
    )
    p.add_argument(
        "--bert-internal-batch-size",
        type=int,
        default=256,
        help="Batch size passed to BERTScorer.score().",
    )
    p.add_argument(
        "--bert-rescale",
        action="store_true",
        default=False,
        help="Use BERTScore baseline rescaling.",
    )

    p.add_argument(
        "--allow-missing-reference",
        action="store_true",
        default=False,
        help=(
            "Keep rows that have no PaliGemma reference. They receive CLIPScore "
            "but NaN BERTScore. By default such rows are skipped."
        ),
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        default=True,
        help="Keep the intermediate CLIP-pass JSONL.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.device = device_from_arg(args.device)

    metrics = set(args.metrics)
    run_clip = "clip" in metrics
    run_bert = "bert" in metrics

    if run_bert and args.paligemma is None:
        raise SystemExit(
            "ERROR: --paligemma is required when BERTScore is requested. "
            "For image-text-score-only evaluation use: --metrics clip"
        )

    # BERTScore requires references unless missing references are explicitly allowed.
    args.require_reference = (
        run_bert
        and
        (not args.allow_missing_reference)
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_csv = (
        args.output_dir
        / "per_sample_metrics.csv"
    )

    summary_csv = (
        args.output_dir
        / "summary_by_stage_alpha_prompt.csv"
    )

    metadata_json = (
        args.output_dir
        / "run_metadata.json"
    )

    dimension_summary_csv = (
        args.output_dir
        / "summary_clip_by_dimension.csv"
    )

    temp_jsonl = (
        args.output_dir
        / "_metric_input.tmp.jsonl"
    )

    print("=" * 72)
    print("Show-o2 × Qwen3-VL evaluation")
    print(f"Device:       {args.device}")
    print(f"Metrics:      {', '.join(args.metrics)}")
    print(f"Predictions:  {args.predictions}")

    print(
        f"PaliGemma:    "
        f"{args.paligemma if args.paligemma else 'not used'}"
    )

    if run_clip:
        print(f"Image root:   {args.image_root}")

        print(
            f"CLIP backbone: "
            f"{args.clip_backbone}"
        )

        if args.clip_backbone == "clip":
            print(
                f"CLIP model: {args.clip_model}"
            )

        elif args.clip_backbone == "longclip":
            print(
                f"Long-CLIP checkpoint: "
                f"{args.longclip_checkpoint}"
            )

        elif args.clip_backbone == "specs":
            print(
                f"SPECS checkpoint: "
                f"{args.specs_checkpoint}"
            )

        elif args.clip_backbone == "jina":
            print(
                f"Jina-CLIP-v2 model: "
                f"{args.jina_model}"
            )
    else:
        print(
            "Image root:   not used "
            "(BERTScore-only mode)"
        )

    if run_bert:
        print(
            f"BERT model:   {args.bert_model}"
        )

        print(
            "BERT baseline rescaling: "
            f"{args.bert_rescale}"
        )

    print("=" * 72)

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    if run_bert:

        print(
            "\nLoading PaliGemma pseudo-references..."
        )

        ref_rows = list(
            iter_reference_rows(
                args.paligemma
            )
        )

        ref_index = ReferenceIndex(
            ref_rows
        )

        print(
            f"PaliGemma references: "
            f"{len(ref_rows):,}"
        )

        if ref_index.ambiguous_basenames:

            print(
                f"[WARN] "
                f"{len(ref_index.ambiguous_basenames):,} "
                f"duplicate basenames cannot use "
                f"basename fallback."
            )

    else:
        ref_index = NullReferenceIndex()

    # --------------------------------------------------------
    # Prepare normalized rows.
    #
    # If CLIP is requested, the normal CLIP pass also prepares
    # the rows needed by BERTScore.
    #
    # If BERTScore alone is requested, perform a lightweight
    # preparation pass that never loads CLIP and never opens
    # image files.
    # --------------------------------------------------------

    if run_clip:

        resolver = ImageResolver(
            args.image_root
        )

        if run_bert:
            print(
                "\nPhase 1/2: "
                "image-text scoring"
            )
        else:
            print(
                "\nImage-text score evaluation"
            )

        clip_counts = write_clip_pass(
            args=args,
            ref_index=ref_index,
            resolver=resolver,
            temp_jsonl=temp_jsonl,
        )

        bert_prep_counts = {}

    else:

        print(
            "\nPreparing BERTScore inputs "
            "(CLIP/SPECS not loaded)..."
        )

        bert_prep_counts = (
            write_bert_input_pass(
                args=args,
                ref_index=ref_index,
                temp_jsonl=temp_jsonl,
            )
        )

        clip_counts = {}

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    if run_bert:

        if run_clip:
            print(
                "\nPhase 2/2: "
                "BERTScore against PaliGemma"
            )
        else:
            print(
                "\nBERTScore against PaliGemma"
            )

        bert_counts = bertscore_pass(
            args=args,
            temp_jsonl=temp_jsonl,
            final_csv=final_csv,
            summary_csv=summary_csv,
        )

    else:

        # CLIP-only: turn the temporary JSONL into the same
        # user-facing CSV and create an aggregated summary
        # without BERTScore columns.

        groups: Dict[
            Tuple[str, str, str],
            RunningStats,
        ] = defaultdict(
            RunningStats
        )

        rows = []

        with temp_jsonl.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                row = json.loads(
                    line
                )

                row[
                    "bertscore_precision"
                ] = math.nan

                row[
                    "bertscore_recall"
                ] = math.nan

                row[
                    "bertscore_f1"
                ] = math.nan

                rows.append(
                    row
                )

                key = (
                    str(
                        row.get(
                            "stage",
                            "unknown",
                        )
                    ),
                    normalise_alpha(
                        row.get(
                            "alpha"
                        )
                    ),
                    str(
                        row.get(
                            "prompt_id",
                            "unknown",
                        )
                    ),
                )

                groups[key].update(
                    row
                )

        fieldnames = [
            "image",
            "resolved_image_path",
            "stage",
            "alpha",
            "prompt_id",
            "prompt_text",
            "generated_text",
            "observation_text",
            "clip_score",
            "clip_token_count",
            "clip_was_truncated",
            "output_words",
            "observation_words",
            "source_file",
            "error",
        ]

        with final_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    row
                )

        summary_rows = []

        for (
            stage,
            alpha,
            prompt_id,
        ), stats in groups.items():

            s = stats.result()

            summary_rows.append(
                {
                    "stage":
                        stage,

                    "alpha":
                        alpha,

                    "prompt_id":
                        prompt_id,

                    "n":
                        s["n"],

                    "clip_score_mean":
                        s["clip_score_mean"],

                    "clip_score_std":
                        s["clip_score_std"],

                    "output_words_mean":
                        s["output_words_mean"],

                    "output_words_std":
                        s["output_words_std"],

                    "clip_truncated_n":
                        s["clip_truncated_n"],

                    "clip_truncated_rate":
                        s["clip_truncated_rate"],
                }
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

        if not summary_df.empty:

            summary_df[
                "_alpha_numeric"
            ] = pd.to_numeric(
                summary_df["alpha"],
                errors="coerce",
            )

            summary_df = (
                summary_df
                .sort_values(
                    [
                        "stage",
                        "prompt_id",
                        "_alpha_numeric",
                        "alpha",
                    ],
                    na_position="last",
                )
                .drop(
                    columns=[
                        "_alpha_numeric"
                    ]
                )
            )

        summary_df.to_csv(
            summary_csv,
            index=False,
        )

        bert_counts = {}

    # Only create a CLIP dimension summary when a CLIP-family
    # image-text score was actually computed.
    if run_clip:

        write_clip_dimension_summary(
            per_sample_csv=
                final_csv,

            output_csv=
                dimension_summary_csv,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if run_clip:

        if args.clip_backbone == "clip":

            score_definition = (
                "CLIPScore: "
                "2.5 * max(cosine(image, observation), 0)"
            )
            clip_model_used = args.clip_model

        elif args.clip_backbone == "longclip":

            score_definition = (
                "LongCLIP-based CLIPScore: "
                "2.5 * max(cosine(image, observation), 0)"
            )
            clip_model_used = args.longclip_checkpoint

        elif args.clip_backbone == "specs":

            score_definition = (
                "SPECS: "
                "max((cosine(image, observation) + 1) / 2, 0)"
            )
            clip_model_used = args.specs_checkpoint

        elif args.clip_backbone == "jina":

            score_definition = (
                "Jina-CLIP-v2: "
                "cosine(image, observation)"
            )
            clip_model_used = args.jina_model

        else:

            score_definition = (
                "SPECS: "
                "max(("
                "cosine(image, observation) + 1"
                ") / 2, 0)"
            )

            clip_model_used = (
                args.specs_checkpoint
            )

    else:
        score_definition = None
        clip_model_used = None

    metadata = {
        "predictions":
            str(
                args.predictions
            ),

        "paligemma":
            (
                str(args.paligemma)
                if args.paligemma
                else None
            ),

        "image_root":
            (
                str(args.image_root)
                if (
                    run_clip
                    and
                    args.image_root
                )
                else None
            ),

        "device":
            args.device,

        "metrics":
            list(
                args.metrics
            ),

        "clip_model":
            clip_model_used,

        "clip_backbone":
            (
                args.clip_backbone
                if run_clip
                else None
            ),

        "clipscore_definition":
            score_definition,

        "bert_model":
            (
                args.bert_model
                if run_bert
                else None
            ),

        "bert_rescale_with_baseline":
            (
                args.bert_rescale
                if run_bert
                else None
            ),

        "bertscore_reference":
            (
                "Generated Observation vs "
                "PaliGemma pseudo-reference caption"
                if run_bert
                else None
            ),

        "full_generated_response_scored":
            False,

        "scored_text_component":
            "Observation",

        "clip_counts":
            clip_counts,

        "bert_input_counts":
            bert_prep_counts,

        "bert_counts":
            bert_counts,
    }

    metadata_json.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not args.keep_temp:

        try:
            temp_jsonl.unlink()

        except FileNotFoundError:
            pass

    print("\nDone.")
    print(
        f"Per-sample metrics: "
        f"{final_csv}"
    )
    print(
        f"Grouped summary:    "
        f"{summary_csv}"
    )
    print(
        f"Run metadata:       "
        f"{metadata_json}"
    )

    if run_clip:

        truncated = clip_counts.get(
            "clip_truncated",
            0,
        )

        total = clip_counts.get(
            "clip_rows_written",
            0,
        )

        if total:

            print(
                f"\nCLIP text truncation: "
                f"{truncated:,}/{total:,} "
                f"("
                f"{100.0 * truncated / total:.2f}%"
                f")."
            )

            if truncated:

                print(
                    "Important: this refers to the extracted "
                    "Observation component. Some Observations "
                    "exceeded the image-text model's context "
                    "window and were truncated for encoding. "
                    "The complete generated response is still "
                    "retained in the CSV."
                )


if __name__ == "__main__":
    main()