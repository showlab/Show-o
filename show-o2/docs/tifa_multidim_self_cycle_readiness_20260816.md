# TIFA Multi-dimensional Self-cycle Readiness (2026-08-16)

## Decision

The pinned TIFA v1.0 COCO subset is ready for implementation of the formal
Show-o2 continuation trainer. The data, prompt protocol, per-question frozen-U
gate, and gradient path into both generation adaptation routes have been
validated. The formal mixed replay/K=4 training loop and queue are not launched
or claimed complete by this audit.

## Pinned source and splits

- Repository: `https://github.com/Yushi-Hu/tifa`
- Commit: `8e37050f44ab4ee681e4251f3c8a9d4cc86e901e`
- Source bytes: `9,630,428`
- Git blob SHA1: `6a563f9c157e5b2a24b9f4936edd3934a8a70add`
- Questions: `25,829`
- Grouped prompts: `4,073`
- Local COCO-reference prompts: `2,000`
- COCO prompts with at least two semantic dimensions: `1,975`
- Multidimensional split: train `1,761`, validation `112`, test `102`
- Split unit: SHA256 of `sample_id + caption`; no prompt crosses splits.

The official source has 15 malformed multiple-choice records. The deterministic
cleaning report retains the official answer, removes duplicate choices, and
replaces the final distractor when the answer is missing from a four-choice
question. Every change is recorded in `cleaning_report.json`.

## Prompt protocol

For each generated image, select at most four questions with distinct semantic
dimensions first. The frozen understanding path receives only:

```text
predicted image + question + answer choices
```

The generation caption is not included. Eighteen questions that reproduce the
full caption are filtered at sampling time. Every answer is represented by a
single option-letter token (`A`, `B`, `C`, or `D`). Two- and three-choice
questions use a candidate mask, so invalid option logits do not affect the loss.

Choice order is deterministically reshuffled from `base_seed`, `sample_id`, and
`cycle_step`. G-adapter and LoRA therefore receive identical questions and
permutations. The final target frequencies are balanced:

- Two choices: `48.74% / 51.26%`
- Four choices: `24.72% / 25.26% / 24.59% / 25.44%`

The sampled question pool covers 12 semantic dimensions. Spatial questions are
only `7.24%`; object, activity, human/animal, location, attribute, counting,
color, food, material, shape, and other questions supply the remaining signal.
The longest Qwen-tokenized prompt has 65 tokens, so `max_text_tokens=96` is safe.

## Frozen-U gate

Use a real-image strict gate per question, not an all-or-nothing prompt gate.
A question contributes student next-token CE only when frozen U answers the
same question correctly on the paired real COCO image and exceeds the configured
target margin. One invalid question does not discard valid dimensions from the
same prompt.

Held-out validation (`112` prompts, `446` sampled questions, margin `0`) gives:

- Valid questions: `381/446 = 85.43%`
- Prompts with any valid question: `112/112 = 100%`
- Prompts with at least two valid questions: `109/112 = 97.32%`
- Prompts with at least two valid dimensions: `108/112 = 96.43%`
- Prompts with every sampled question valid: `60/112 = 53.57%`

The old whole-prompt strict gate would unnecessarily discard about 46% of
prompts. Per-question gating preserves the strict teacher criterion without
starving multi-dimensional training.

## Gradient probes

Both probes use the same COCO/TIFA sample and activity, color, and human/animal
questions. Generation adaptation is disabled during frozen-U scoring.

- S1-M G-adapter step-5000: 12/12 parameter tensors receive gradients;
  total norm `20.1760`.
- S1-M LoRA step-6000: 224/224 parameter tensors receive gradients;
  total norm `0.7891`.

The norms are not comparable across parameterizations. The relevant result is
that every trainable tensor in both routes receives a nonzero gradient through
`next-token logits -> predicted-image latent -> generation route`, while the U
path and base model remain frozen.

## Remaining formal-training work

1. Wire `coco_multidim_train.jsonl`, position-balanced sampling, and
   `per_question_tifa_cycle_loss` into the continuation trainer.
2. Mix the same generation replay/control objective into treatment and
   without-cycle controls; the only causal difference must be the TIFA cycle
   loss.
3. Run a K=4 memory and gradient smoke. The completed end-to-end probes validate
   a single differentiable predicted-clean step, not the final K=4 schedule.
4. Continue the four selected S1 starts with identical samples and seeds.
5. Do not report official TIFA v1.0 as an unseen benchmark after training on its
   prompts. Use the held-out split and GQA or another external benchmark.
