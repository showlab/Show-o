import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from evaluation.c4_drsa_adapter.smoke_drsa_data import validate_spright, validate_vsr
from datasets.drsa_dataset import SprightCocoIterableDataset, VsrRelationDataset


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class DrsaDataSmokeTest(unittest.TestCase):
    def test_spright_tar_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "spright_coco" / "data" / "10.tar"
            shard.parent.mkdir(parents=True)
            with tarfile.open(shard, "w") as archive:
                _add_bytes(archive, "7.jpg", _jpeg_bytes())
                _add_bytes(
                    archive,
                    "7.json",
                    json.dumps(
                        {
                            "spatial_caption": "A cup is left of a book.",
                            "original_caption": "A cup and a book.",
                        }
                    ).encode(),
                )
                _add_bytes(
                    archive,
                    "7.metadata.json",
                    json.dumps({"image_origin": "COCO", "image_id": "7.jpg"}).encode(),
                )
            result = validate_spright(root, 1)
            self.assertEqual(result["samples_checked"], 1)
            self.assertEqual(result["relation_counts"]["left of"], 1)
            record = next(iter(SprightCocoIterableDataset(shard.parent, shuffle_buffer=0)))
            self.assertEqual(record["sample_id"], "7")
            self.assertEqual(record["image"].mode, "RGB")

    def test_vsr_contract_and_zip_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "vsr" / "annotations" / "zeroshot" / "train.jsonl"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(
                json.dumps(
                    {
                        "image": "sample.jpg",
                        "caption": "The cup is right of the book.",
                        "label": 1,
                        "relation": "right of",
                        "subj": "cup",
                        "obj": "book",
                    }
                )
                + "\n"
            )
            image_zip = root / "vsr" / "vsr_images.zip"
            with zipfile.ZipFile(image_zip, "w") as archive:
                archive.writestr("images/sample.jpg", _jpeg_bytes())
            result = validate_vsr(root, 1, True)
            self.assertEqual(result["rows_checked"], 1)
            self.assertEqual(result["missing_images"], 0)
            self.assertEqual(result["role_text_mismatches"], 0)
            dataset = VsrRelationDataset(annotation, image_zip)
            record = dataset[0]
            self.assertEqual(record["relation"], "right of")
            self.assertEqual(record["image"].size, (32, 24))


if __name__ == "__main__":
    unittest.main()
