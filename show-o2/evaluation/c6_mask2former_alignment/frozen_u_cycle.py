"""Differentiable frozen-U relation supervision for predicted images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


RELATION_TEXT = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}

INVERSE_RELATION = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}

RELATION_OPTIONS = ("left", "right", "above", "below")
OPTION_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class RelationQuestion:
    name: str
    subject: str
    relation: str
    object: str
    target: int

    @property
    def statement(self) -> str:
        relation_text = RELATION_TEXT[self.relation]
        return (
            f"Statement: In this image, {self.subject} is {relation_text} "
            f"{self.object}. Answer yes or no:"
        )


@dataclass(frozen=True)
class RelationChoiceQuestion:
    name: str
    subject: str
    object: str
    relation: str
    target: int

    @property
    def statement(self) -> str:
        return (
            f"Where is {self.subject} relative to {self.object} in the image? "
            "A. left. B. right. C. above. D. below. "
            "Answer with exactly one option letter: A, B, C, or D. Answer:"
        )


@dataclass(frozen=True)
class UQuestionBatch:
    questions: tuple[RelationQuestion | RelationChoiceQuestion, ...]
    text_tokens: torch.Tensor
    modality_positions: torch.Tensor
    answer_logit_indices: torch.Tensor
    targets: torch.Tensor


def relation_questions(
    phrases: Sequence[str], relation: str
) -> tuple[RelationQuestion, ...]:
    """Build positive, inverse-negative, and swapped-inverse checks."""
    if len(phrases) != 2:
        raise ValueError("frozen-U relation supervision requires exactly two objects")
    if relation not in INVERSE_RELATION:
        raise ValueError(f"unsupported spatial relation: {relation!r}")
    first, second = (str(value).strip() for value in phrases)
    if not first or not second:
        raise ValueError("object phrases must be non-empty")
    inverse = INVERSE_RELATION[relation]
    return (
        RelationQuestion("positive", first, relation, second, 0),
        RelationQuestion("inverse_negative", first, inverse, second, 1),
        RelationQuestion("swapped_inverse", second, inverse, first, 0),
    )


def yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Return the single-token ids used by the existing VSR protocol."""
    yes = tokenizer(" yes", add_special_tokens=False)["input_ids"]
    no = tokenizer(" no", add_special_tokens=False)["input_ids"]
    if len(yes) != 1 or len(no) != 1:
        raise RuntimeError(f"U-cycle labels are not single tokens: yes={yes}, no={no}")
    if yes[0] == no[0]:
        raise RuntimeError("yes and no resolved to the same token")
    return int(yes[0]), int(no[0])


def relation_option_token_ids(tokenizer: Any) -> tuple[int, int, int, int]:
    values = []
    for label in OPTION_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise RuntimeError(
                f"U relation option is not a single token: {label}={token_ids}"
            )
        values.append(int(token_ids[0]))
    if len(set(values)) != len(values):
        raise RuntimeError(f"U relation options do not have unique tokens: {values}")
    return tuple(values)  # type: ignore[return-value]


def build_question_batch(
    phrases: Sequence[str],
    relation: str,
    tokenizer: Any,
    token_ids: Mapping[str, int],
    *,
    max_text_tokens: int,
    sequence_length: int,
    device: torch.device,
) -> UQuestionBatch:
    """Build answer-free MMU prefixes; logits at the prefix end predict yes/no."""
    if max_text_tokens < 1:
        raise ValueError("max_text_tokens must be positive")
    questions = relation_questions(phrases, relation)
    rows: list[torch.Tensor] = []
    answer_indices: list[int] = []
    positions: list[torch.Tensor] = []
    image_token_count = 730
    for question in questions:
        prompt_ids = tokenizer(
            question.statement,
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_tokens,
        )["input_ids"]
        if not prompt_ids:
            raise ValueError(f"question has no prompt tokens: {question.statement!r}")
        tokens = (
            [int(token_ids["bos_id"]), int(token_ids["boi_id"])]
            + [int(token_ids["img_pad_id"])] * image_token_count
            + [int(token_ids["eoi_id"])]
            + [int(value) for value in prompt_ids]
        )
        if len(tokens) > sequence_length:
            raise ValueError(
                f"U question requires {len(tokens)} tokens but sequence_length="
                f"{sequence_length}: {question.statement!r}"
            )
        answer_indices.append(len(tokens) - 1)
        tokens.extend([int(tokenizer.pad_token_id)] * (sequence_length - len(tokens)))
        rows.append(torch.tensor(tokens, dtype=torch.long, device=device))
        positions.append(
            torch.tensor([[2, image_token_count]], dtype=torch.long, device=device)
        )
    return UQuestionBatch(
        questions=questions,
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
    )


def build_relation_choice_batch(
    phrases: Sequence[str],
    relation: str,
    tokenizer: Any,
    token_ids: Mapping[str, int],
    *,
    max_text_tokens: int,
    sequence_length: int,
    device: torch.device,
) -> UQuestionBatch:
    if len(phrases) != 2:
        raise ValueError("relation choice requires exactly two objects")
    if relation not in RELATION_OPTIONS:
        raise ValueError(f"unsupported spatial relation: {relation!r}")
    first, second = (str(value).strip() for value in phrases)
    if not first or not second:
        raise ValueError("object phrases must be non-empty")
    question = RelationChoiceQuestion(
        name="relation_multiple_choice",
        subject=first,
        object=second,
        relation=relation,
        target=RELATION_OPTIONS.index(relation),
    )
    prompt_ids = tokenizer(
        question.statement,
        add_special_tokens=False,
        truncation=True,
        max_length=max_text_tokens,
    )["input_ids"]
    if not prompt_ids:
        raise ValueError("relation choice question has no prompt tokens")
    image_token_count = 730
    tokens = (
        [int(token_ids["bos_id"]), int(token_ids["boi_id"])]
        + [int(token_ids["img_pad_id"])] * image_token_count
        + [int(token_ids["eoi_id"])]
        + [int(value) for value in prompt_ids]
    )
    if len(tokens) > sequence_length:
        raise ValueError(
            f"relation choice requires {len(tokens)} tokens but sequence_length="
            f"{sequence_length}"
        )
    answer_index = len(tokens) - 1
    tokens.extend([int(tokenizer.pad_token_id)] * (sequence_length - len(tokens)))
    return UQuestionBatch(
        questions=(question,),
        text_tokens=torch.tensor([tokens], dtype=torch.long, device=device),
        modality_positions=torch.tensor(
            [[[2, image_token_count]]], dtype=torch.long, device=device
        ),
        answer_logit_indices=torch.tensor(
            [answer_index], dtype=torch.long, device=device
        ),
        targets=torch.tensor([question.target], dtype=torch.long, device=device),
    )


def deterministic_image_latents(
    vae: Any, images: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Encode decoded RGB images through the deterministic posterior mean."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected RGB images [B,3,H,W], got {tuple(images.shape)}")
    latents = vae.sample(
        images.to(dtype=dtype).unsqueeze(2),
        deterministic=True,
    )
    return latents.squeeze(2).to(dtype)


def frozen_u_candidate_logits(
    model: Any,
    image_latents: torch.Tensor,
    batch: UQuestionBatch,
    candidate_token_ids: Sequence[int],
    *,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask_builder: Callable[..., torch.Tensor],
    relation_codes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run frozen U while retaining autograd with respect to image_latents."""
    question_count = len(batch.questions)
    if len(candidate_token_ids) < 2:
        raise ValueError("frozen-U scoring requires at least two candidate tokens")
    if image_latents.ndim != 4 or image_latents.shape[0] != 1:
        raise ValueError(
            "frozen-U cycle currently expects one image latent per optimizer step"
        )
    if relation_codes is not None and relation_codes.shape[0] != question_count:
        raise ValueError(
            f"expected {question_count} relation codes, got {tuple(relation_codes.shape)}"
        )
    repeated_latents = image_latents.expand(question_count, -1, -1, -1).contiguous()
    attention_mask = attention_mask_builder(
        question_count,
        sequence_length,
        batch.modality_positions,
        device,
    ).to(dtype)
    logits = model.forward_und_only(
        text_tokens=batch.text_tokens,
        image_latents=repeated_latents,
        t=torch.ones((question_count,), device=device, dtype=dtype),
        attention_mask=attention_mask,
        modality_positions=batch.modality_positions,
        output_hidden_states=True,
        max_seq_len=sequence_length,
        device=device,
        drsa_relation_codes=relation_codes,
    )
    row_indices = torch.arange(question_count, device=device)
    next_token_logits = logits[row_indices, batch.answer_logit_indices].float()
    candidate_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=device)
    return next_token_logits.index_select(-1, candidate_ids)


def prompt_u_task_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gate_policy: str = "strict",
    gate_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply teacher validity gating to predicted-image classification."""
    if (
        student_logits.shape != teacher_logits.shape
        or student_logits.ndim != 2
        or student_logits.shape[-1] < 2
    ):
        raise ValueError(
            "student and teacher logits must have matching [Q,C>=2] shapes"
        )
    if targets.shape != student_logits.shape[:-1]:
        raise ValueError("one binary target is required per U question")
    if gate_policy not in ("strict", "none"):
        raise ValueError("gate_policy must be strict or none")
    if gate_margin < 0.0:
        raise ValueError("gate_margin must be non-negative")

    teacher_logits = teacher_logits.detach().float()
    student_logits = student_logits.float()
    targets = targets.long()
    if int(targets.min()) < 0 or int(targets.max()) >= student_logits.shape[-1]:
        raise ValueError("classification target is outside the candidate logits")
    indices = torch.arange(targets.numel(), device=targets.device)
    teacher_target_logits = teacher_logits[indices, targets]
    competitor_logits = teacher_logits.masked_fill(
        F.one_hot(targets, num_classes=teacher_logits.shape[-1]).bool(),
        float("-inf"),
    ).amax(dim=-1)
    teacher_margins = teacher_target_logits - competitor_logits
    teacher_predictions = teacher_logits.argmax(dim=-1)
    teacher_correct = teacher_predictions.eq(targets)
    strict_valid = teacher_correct.all() & (teacher_margins > gate_margin).all()
    gate = strict_valid if gate_policy == "strict" else strict_valid.new_ones(())

    per_question = F.cross_entropy(student_logits, targets, reduction="none")
    raw_loss = per_question.mean()
    loss = raw_loss * gate.to(raw_loss.dtype)
    student_probabilities = student_logits.softmax(dim=-1)
    teacher_probabilities = teacher_logits.softmax(dim=-1)
    details = {
        "gate_policy": gate_policy,
        "gate_margin_threshold": float(gate_margin),
        "gate_valid": bool(gate.detach().cpu()),
        "teacher_strict_valid": bool(strict_valid.detach().cpu()),
        "teacher_targets": targets.detach().cpu().tolist(),
        "teacher_predictions": teacher_predictions.detach().cpu().tolist(),
        "teacher_correct": teacher_correct.detach().cpu().tolist(),
        "teacher_target_margins": teacher_margins.detach().cpu().tolist(),
        "teacher_target_probabilities": teacher_probabilities[
            indices, targets
        ].detach().cpu().tolist(),
        "student_predictions": student_logits.argmax(dim=-1).detach().cpu().tolist(),
        "student_correct": student_logits.argmax(dim=-1).eq(targets).detach().cpu().tolist(),
        "student_target_probabilities": student_probabilities[
            indices, targets
        ].detach().cpu().tolist(),
        "per_question_ce": per_question.detach().cpu().tolist(),
        "ungated_mean_ce": float(raw_loss.detach().cpu()),
    }
    if student_logits.shape[-1] == 2:
        details["teacher_yes_probabilities"] = (
            teacher_probabilities[:, 0].detach().cpu().tolist()
        )
        details["student_yes_probabilities"] = (
            student_probabilities[:, 0].detach().cpu().tolist()
        )
    return loss, details
