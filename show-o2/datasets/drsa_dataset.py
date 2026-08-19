"""Native PyTorch datasets for DRSA without WebDataset extraction."""

from __future__ import annotations

import collections
import io
import json
import random
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info


ImageTransform = Callable[[Image.Image], Any]


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _member_key(name: str) -> tuple[str, str] | None:
    basename = Path(name).name
    if basename.endswith(".metadata.json"):
        return basename[: -len(".metadata.json")], "metadata"
    if basename.endswith(".json"):
        return basename[: -len(".json")], "caption"
    if basename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return basename.rsplit(".", 1)[0], "image"
    return None


def _decode_image(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def iter_spright_shard(path: Path) -> Iterator[dict[str, Any]]:
    pending: dict[str, dict[str, bytes]] = collections.defaultdict(dict)
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parsed = _member_key(member.name)
            if parsed is None:
                continue
            stem, kind = parsed
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            pending[stem][kind] = extracted.read()
            if not {"image", "caption", "metadata"}.issubset(pending[stem]):
                continue
            raw = pending.pop(stem)
            caption = json.loads(raw["caption"])
            metadata = json.loads(raw["metadata"])
            spatial_caption = caption.get("spatial_caption")
            original_caption = caption.get("original_caption")
            if not isinstance(spatial_caption, str) or not spatial_caption.strip():
                raise ValueError(f"SPRIGHT sample {stem} has no spatial_caption")
            if not isinstance(original_caption, str) or not original_caption.strip():
                raise ValueError(f"SPRIGHT sample {stem} has no original_caption")
            yield {
                "sample_id": stem,
                "image": _decode_image(raw["image"]),
                "spatial_caption": spatial_caption,
                "original_caption": original_caption,
                "image_origin": metadata.get("image_origin"),
                "image_id": metadata.get("image_id"),
                "shard": path.name,
            }


class SprightCocoIterableDataset(IterableDataset):
    """Stream SPRIGHT triples and partition shards across DDP/data workers."""

    def __init__(
        self,
        shard_root: str | Path,
        *,
        image_transform: ImageTransform | None = None,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        self.shard_root = Path(shard_root)
        self.shards = sorted(self.shard_root.glob("*.tar"), key=_natural_key)
        if not self.shards:
            raise FileNotFoundError(f"No SPRIGHT tar files found in {self.shard_root}")
        if shuffle_buffer < 0:
            raise ValueError("shuffle_buffer must be non-negative")
        self.image_transform = image_transform
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _worker_partition(self) -> tuple[list[Path], random.Random]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers_per_rank = worker.num_workers if worker else 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        global_worker = rank * workers_per_rank + worker_id
        global_workers = world_size * workers_per_rank
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        shards = list(self.shards)
        if self.shuffle_shards:
            rng.shuffle(shards)
        return shards[global_worker::global_workers], random.Random(
            self.seed + 1_000_003 * self.epoch + 9_973 * global_worker
        )

    def _shuffle(self, records: Iterator[dict[str, Any]], rng: random.Random) -> Iterator[dict[str, Any]]:
        if self.shuffle_buffer <= 1:
            yield from records
            return
        buffer: list[dict[str, Any]] = []
        for record in records:
            buffer.append(record)
            if len(buffer) >= self.shuffle_buffer:
                index = rng.randrange(len(buffer))
                yield buffer.pop(index)
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        shards, rng = self._worker_partition()

        def records() -> Iterator[dict[str, Any]]:
            for shard in shards:
                yield from iter_spright_shard(shard)

        for record in self._shuffle(records(), rng):
            if self.image_transform is not None:
                record["image"] = self.image_transform(record["image"])
            yield record


class VsrRelationDataset(Dataset):
    """Read VSR annotations and images directly from the released zip."""

    required_fields = {"image", "caption", "label", "relation", "subj", "obj"}

    def __init__(
        self,
        annotation_path: str | Path,
        image_zip_path: str | Path,
        *,
        image_transform: ImageTransform | None = None,
        require_role_text_match: bool = False,
    ):
        self.annotation_path = Path(annotation_path)
        self.image_zip_path = Path(image_zip_path)
        if not self.annotation_path.is_file():
            raise FileNotFoundError(self.annotation_path)
        if not self.image_zip_path.is_file():
            raise FileNotFoundError(self.image_zip_path)
        self.rows: list[dict[str, Any]] = []
        with self.annotation_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                missing = self.required_fields - row.keys()
                if missing:
                    raise ValueError(
                        f"VSR line {line_number} is missing fields: {sorted(missing)}"
                    )
                caption = str(row["caption"]).lower()
                role_match = (
                    str(row["subj"]).lower() in caption
                    and str(row["obj"]).lower() in caption
                )
                if require_role_text_match and not role_match:
                    continue
                row["role_text_match"] = role_match
                self.rows.append(row)
        with zipfile.ZipFile(self.image_zip_path) as archive:
            self.image_members = {
                Path(name).name: name
                for name in archive.namelist()
                if not name.endswith("/")
            }
        missing_images = {
            Path(str(row["image"])).name
            for row in self.rows
            if Path(str(row["image"])).name not in self.image_members
        }
        if missing_images:
            preview = sorted(missing_images)[:5]
            raise ValueError(f"VSR zip is missing {len(missing_images)} images: {preview}")
        self.image_transform = image_transform
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.image_zip_path)
        return self._archive

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        image_name = Path(str(row["image"])).name
        image = _decode_image(self._zip().read(self.image_members[image_name]))
        if self.image_transform is not None:
            image = self.image_transform(image)
        row["image"] = image
        row["label"] = int(row["label"])
        return row

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_archive"] = None
        return state

    def __del__(self) -> None:
        archive = getattr(self, "_archive", None)
        if archive is not None:
            archive.close()


class BprimeCocoBoxDataset(Dataset):
    """Read balanced COCO-box rows while keeping SPRIGHT images inside tar files."""

    required_fields = {
        "sample_id",
        "shard",
        "tar_stem",
        "image_id",
        "prompt",
        "objects",
    }
    valid_relations = {"left", "right", "above", "below"}
    relation_axes = {
        "left": "horizontal",
        "right": "horizontal",
        "above": "vertical",
        "below": "vertical",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        shard_root: str | Path,
        *,
        image_transform: ImageTransform | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.shard_root = Path(shard_root)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.rows: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                missing = self.required_fields - row.keys()
                if missing:
                    raise ValueError(
                        f"B-prime line {line_number} is missing fields: {sorted(missing)}"
                    )
                raw_relations = row.get("relations")
                if raw_relations is None:
                    raw_relations = [row.get("relation")]
                if not isinstance(raw_relations, list) or not 1 <= len(raw_relations) <= 2:
                    raise ValueError(
                        f"B-prime line {line_number} must have one or two relations"
                    )
                relations = tuple(str(value) for value in raw_relations)
                if len(set(relations)) != len(relations) or any(
                    relation not in self.valid_relations for relation in relations
                ):
                    raise ValueError(
                        f"B-prime line {line_number} has invalid relations: {relations}"
                    )
                axes = [self.relation_axes[relation] for relation in relations]
                if len(set(axes)) != len(axes):
                    raise ValueError(
                        f"B-prime line {line_number} has conflicting same-axis relations: {relations}"
                    )
                if len(row["objects"]) != 2:
                    raise ValueError(f"B-prime line {line_number} must have two objects")
                row["relations"] = list(relations)
                row.setdefault("relation", relations[0])
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"B-prime manifest is empty: {self.manifest_path}")
        missing_shards = {
            str(row["shard"])
            for row in self.rows
            if not (self.shard_root / str(row["shard"])).is_file()
        }
        if missing_shards:
            raise FileNotFoundError(f"Missing B-prime shards: {sorted(missing_shards)}")
        self.image_transform = image_transform
        self._archives: dict[str, tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _archive(self, shard: str) -> tarfile.TarFile:
        if shard not in self._archives:
            self._archives[shard] = tarfile.open(self.shard_root / shard, mode="r:")
        return self._archives[shard]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        archive = self._archive(str(row["shard"]))
        stem = str(row["tar_stem"])
        member = archive.getmember(f"{stem}.jpg")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not read {row['shard']}:{stem}.jpg")
        image = _decode_image(extracted.read())
        if self.image_transform is not None:
            image = self.image_transform(image)
        row["image"] = image
        return row

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_archives"] = {}
        return state

    def __del__(self) -> None:
        for archive in getattr(self, "_archives", {}).values():
            archive.close()
