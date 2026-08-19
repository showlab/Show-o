"""C0 SEED-Bench views split by modality and checkpoint."""

from __future__ import annotations

from lmms_eval.tasks.seedbench.utils import seed_aggregation_result, seed_doc_to_text


def _filter(dataset, data_type: str):
    filtered = dataset.filter(lambda doc: doc["data_type"] == data_type)
    if len(filtered) == 0:
        raise ValueError(f"SEED-Bench has no {data_type!r} documents after modality filtering")
    return filtered


def seed_process_image_docs(dataset):
    return _filter(dataset, "image")


def seed_process_video_docs(dataset):
    return _filter(dataset, "video")


def seed_doc_to_visual(doc):
    visuals = doc["image"]
    if not isinstance(visuals, (list, tuple)):
        visuals = [visuals]
    return [image.convert("RGB") if hasattr(image, "convert") else image for image in visuals]


def _prediction(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    return {"pred": pred, "answer": doc["answer"], "question_id": doc["question_id"]}


def seed_process_image_result(doc, result):
    if doc["data_type"] != "image":
        raise ValueError(f"Unexpected SEED data_type in image task: {doc['data_type']!r}")
    return {"seed_image": _prediction(doc, result)}


def seed_process_video_result(doc, result):
    if doc["data_type"] != "video":
        raise ValueError(f"Unexpected SEED data_type in video task: {doc['data_type']!r}")
    return {"seed_video": _prediction(doc, result)}


__all__ = [
    "seed_aggregation_result",
    "seed_doc_to_text",
    "seed_doc_to_visual",
    "seed_process_image_docs",
    "seed_process_image_result",
    "seed_process_video_docs",
    "seed_process_video_result",
]
