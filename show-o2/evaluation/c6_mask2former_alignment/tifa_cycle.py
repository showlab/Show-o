"""Multi-dimensional TIFA next-token supervision for differentiable self-cycle."""

from __future__ import annotations

import io
import json
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


OPTION_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class TifaChoiceQuestion:
    name: str
    question: str
    choices: tuple[str, ...]
    target: int
    dimension: str

    @property
    def statement(self) -> str:
        options = " ".join(
            f"{OPTION_LABELS[index]}. {choice}"
            for index, choice in enumerate(self.choices)
        )
        labels = ", ".join(OPTION_LABELS[: len(self.choices)])
        return (
            f"Question about this image: {self.question} {options} "
            f"Answer with exactly one option letter: {labels}. Answer:"
        )


@dataclass(frozen=True)
class TifaQuestionBatch:
    questions: tuple[TifaChoiceQuestion, ...]
    text_tokens: torch.Tensor
    modality_positions: torch.Tensor
    answer_logit_indices: torch.Tensor
    targets: torch.Tensor
    candidate_mask: torch.Tensor


class TifaCocoCycleDataset(Dataset):
    """Read grouped TIFA prompts with optional COCO reference images.

    COCO rows carry a real reference image for the strict frozen-U gate. The
    DrawBench/PaintSkill/PartiPrompt rows are intentionally image-free and use
    the official answer as their supervision target after pure prompt-only
    generation.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        image_transform=None,
        shard_root: str | Path | None = None,
        allow_unpaired: bool = False,
    ):
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.rows = []
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                required = {"sample_id", "caption", "qa_pairs", "dimensions"}
                missing = required - row.keys()
                if missing:
                    raise ValueError(
                        f"TIFA cycle row {line_number} is missing {sorted(missing)}"
                    )
                has_image_locator = "image_path" in row or {
                    "shard", "tar_stem"
                }.issubset(row)
                if "image_path" in row:
                    image_path = Path(row["image_path"])
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                elif not has_image_locator and not allow_unpaired:
                    raise ValueError(
                        f"TIFA cycle row {line_number} has no image locator"
                    )
                if len(row["dimensions"]) < 1:
                    raise ValueError(
                        f"TIFA cycle row {line_number} is not multidimensional"
                    )
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"TIFA cycle manifest is empty: {self.manifest}")
        self.image_transform = image_transform
        self.shard_root = Path(shard_root) if shard_root is not None else None
        self.allow_unpaired = allow_unpaired
        if any(
            "image_path" not in row and not {"shard", "tar_stem"}.issubset(row)
            for row in self.rows
        ) and not allow_unpaired:
            raise ValueError("unpaired TIFA rows require allow_unpaired=True")
        if any(
            "image_path" not in row and {"shard", "tar_stem"}.issubset(row)
            for row in self.rows
        ):
            if self.shard_root is None:
                raise ValueError("tar-backed TIFA rows require shard_root")
            missing_shards = {
                str(row["shard"])
                for row in self.rows
                if "image_path" not in row
                and {"shard", "tar_stem"}.issubset(row)
                and not (self.shard_root / str(row["shard"])).is_file()
            }
            if missing_shards:
                raise FileNotFoundError(
                    f"missing TIFA shards: {sorted(missing_shards)}"
                )
        self._archives: dict[str, tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        if "image_path" in row:
            image_source: Any = row["image_path"]
        elif {"shard", "tar_stem"}.issubset(row):
            shard = str(row["shard"])
            if shard not in self._archives:
                assert self.shard_root is not None
                self._archives[shard] = tarfile.open(
                    self.shard_root / shard, mode="r:"
                )
            archive = self._archives[shard]
            member = archive.getmember(f"{row['tar_stem']}.jpg")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(
                    f"could not read {shard}:{row['tar_stem']}.jpg"
                )
            image_source = io.BytesIO(extracted.read())
        else:
            image_source = None
        if image_source is not None:
            with Image.open(image_source) as image:
                image.load()
                image = image.convert("RGB")
            if self.image_transform is not None:
                image = self.image_transform(image)
            row["image"] = image
            row["reference_image_available"] = True
        else:
            row["reference_image_available"] = False
        return row

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_archives"] = {}
        return state

    def __del__(self) -> None:
        for archive in getattr(self, "_archives", {}).values():
            archive.close()


def option_token_ids(tokenizer: Any) -> tuple[int, int, int, int]:
    values = []
    for label in OPTION_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise RuntimeError(f"TIFA option is not a single token: {label}={token_ids}")
        values.append(int(token_ids[0]))
    if len(set(values)) != len(values):
        raise RuntimeError(f"TIFA option token ids are not unique: {values}")
    return tuple(values)  # type: ignore[return-value]


def select_multidimensional_questions(
    qa_pairs: Sequence[Mapping[str, Any]],
    *,
    max_questions: int,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    """Choose distinct dimensions first, then fill remaining question slots."""
    if max_questions < 2:
        raise ValueError("multidimensional sampling requires max_questions >= 2")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for qa in qa_pairs:
        dimension = str(qa.get("element_type", "unknown")).strip().casefold()
        groups[dimension].append(qa)
    if len(groups) < 2:
        candidates = list(qa_pairs)
        random.Random(seed).shuffle(candidates)
        return tuple(candidates[:max_questions])
    rng = random.Random(seed)
    dimensions = sorted(groups)
    rng.shuffle(dimensions)
    selected: list[Mapping[str, Any]] = []
    remaining: list[Mapping[str, Any]] = []
    for dimension in dimensions:
        candidates = list(groups[dimension])
        rng.shuffle(candidates)
        selected.append(candidates.pop())
        remaining.extend(candidates)
        if len(selected) == max_questions:
            return tuple(selected)
    rng.shuffle(remaining)
    selected.extend(remaining[: max_questions - len(selected)])
    return tuple(selected)


def build_tifa_question_batch(
    qa_pairs: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    token_ids: Mapping[str, int],
    *,
    max_text_tokens: int,
    sequence_length: int,
    device: torch.device,
) -> TifaQuestionBatch:
    """Build answer-free image+question prefixes without the generation caption."""
    if not qa_pairs:
        raise ValueError("TIFA supervision requires at least one question")
    questions: list[TifaChoiceQuestion] = []
    rows: list[torch.Tensor] = []
    answer_indices: list[int] = []
    positions: list[torch.Tensor] = []
    masks: list[list[bool]] = []
    image_token_count = 730
    for index, qa in enumerate(qa_pairs):
        question_text = str(qa.get("question", "")).strip()
        raw_choices = qa.get("choices")
        if not question_text or not isinstance(raw_choices, Sequence) or isinstance(
            raw_choices, (str, bytes)
        ):
            raise ValueError(f"TIFA question {index} is malformed")
        choices = tuple(str(value).strip() for value in raw_choices)
        if not 2 <= len(choices) <= len(OPTION_LABELS) or any(
            not value for value in choices
        ):
            raise ValueError(f"TIFA question {index} must have 2-4 choices")
        target = int(qa.get("target_index", -1))
        if not 0 <= target < len(choices):
            raise ValueError(f"TIFA question {index} has invalid target {target}")
        question = TifaChoiceQuestion(
            name=f"tifa_choice_{index}",
            question=question_text,
            choices=choices,
            target=target,
            dimension=str(qa.get("element_type", "unknown")).strip().casefold(),
        )
        prompt_ids = tokenizer(
            question.statement, add_special_tokens=False
        )["input_ids"]
        if not prompt_ids:
            raise ValueError(f"TIFA question {index} has no prompt tokens")
        if len(prompt_ids) > max_text_tokens:
            raise ValueError(
                f"TIFA question {index} needs {len(prompt_ids)} text tokens; "
                f"max_text_tokens={max_text_tokens} would truncate the answer prefix"
            )
        tokens = (
            [int(token_ids["bos_id"]), int(token_ids["boi_id"])]
            + [int(token_ids["img_pad_id"])] * image_token_count
            + [int(token_ids["eoi_id"])]
            + [int(value) for value in prompt_ids]
        )
        if len(tokens) > sequence_length:
            raise ValueError(
                f"TIFA question {index} requires {len(tokens)} tokens but "
                f"sequence_length={sequence_length}"
            )
        answer_indices.append(len(tokens) - 1)
        tokens.extend(
            [int(tokenizer.pad_token_id)] * (sequence_length - len(tokens))
        )
        rows.append(torch.tensor(tokens, dtype=torch.long, device=device))
        positions.append(
            torch.tensor([[2, image_token_count]], dtype=torch.long, device=device)
        )
        masks.append(
            [candidate < len(choices) for candidate in range(len(OPTION_LABELS))]
        )
        questions.append(question)
    return TifaQuestionBatch(
        questions=tuple(questions),
        text_tokens=torch.stack(rows),
        modality_positions=torch.stack(positions),
        answer_logit_indices=torch.tensor(
            answer_indices, dtype=torch.long, device=device
        ),
        targets=torch.tensor(
            [question.target for question in questions],
            dtype=torch.long,
            device=device,
        ),
        candidate_mask=torch.tensor(masks, dtype=torch.bool, device=device),
    )


def tifa_next_token_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    teacher_logits: torch.Tensor | None = None,
    gate_policy: str = "none",
    gate_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute masked option-letter CE with optional real-image teacher gating."""
    if student_logits.ndim != 2 or student_logits.shape[-1] != len(OPTION_LABELS):
        raise ValueError("student_logits must have shape [Q,4]")
    if targets.shape != student_logits.shape[:-1]:
        raise ValueError("one target is required per TIFA question")
    if candidate_mask.shape != student_logits.shape:
        raise ValueError("candidate_mask must match student_logits")
    if gate_policy not in ("none", "strict"):
        raise ValueError("gate_policy must be none or strict")
    if gate_margin < 0.0:
        raise ValueError("gate_margin must be non-negative")
    targets = targets.long()
    mask = candidate_mask.to(device=student_logits.device, dtype=torch.bool)
    indices = torch.arange(targets.numel(), device=targets.device)
    if int(targets.min()) < 0 or int(targets.max()) >= student_logits.shape[-1]:
        raise ValueError("TIFA target is outside the option logits")
    if not bool(mask[indices, targets].all()):
        raise ValueError("every TIFA target must be enabled by candidate_mask")
    if not bool((mask.sum(dim=-1) >= 2).all()):
        raise ValueError("every TIFA question must enable at least two choices")
    masked_student = student_logits.float().masked_fill(~mask, float("-inf"))
    per_question = F.cross_entropy(masked_student, targets, reduction="none")
    raw_loss = per_question.mean()
    gate = torch.ones((), dtype=torch.bool, device=student_logits.device)
    teacher_details: dict[str, Any] = {}
    if gate_policy == "strict":
        if teacher_logits is None or teacher_logits.shape != student_logits.shape:
            raise ValueError("strict TIFA gate requires matching real-image teacher logits")
        masked_teacher = teacher_logits.detach().float().masked_fill(
            ~mask, float("-inf")
        )
        target_logits = masked_teacher[indices, targets]
        competitors = masked_teacher.masked_fill(
            F.one_hot(targets, num_classes=masked_teacher.shape[-1]).bool(),
            float("-inf"),
        ).amax(dim=-1)
        predictions = masked_teacher.argmax(dim=-1)
        margins = target_logits - competitors
        correct = predictions.eq(targets)
        gate = correct.all() & (margins > gate_margin).all()
        teacher_details = {
            "teacher_predictions": predictions.detach().cpu().tolist(),
            "teacher_correct": correct.detach().cpu().tolist(),
            "teacher_target_margins": margins.detach().cpu().tolist(),
        }
    loss = raw_loss * gate.to(raw_loss.dtype)
    predictions = masked_student.argmax(dim=-1)
    details = {
        "gate_policy": gate_policy,
        "gate_valid": bool(gate.detach().cpu()),
        "gate_margin_threshold": float(gate_margin),
        "targets": targets.detach().cpu().tolist(),
        "candidate_mask": mask.detach().cpu().tolist(),
        "student_predictions": predictions.detach().cpu().tolist(),
        "student_correct": predictions.eq(targets).detach().cpu().tolist(),
        "per_question_ce": per_question.detach().cpu().tolist(),
        "ungated_mean_ce": float(raw_loss.detach().cpu()),
        **teacher_details,
    }
    return loss, details
