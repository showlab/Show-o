from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image


class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.shift = 9

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        img_id = self.coco.imgs[idx]["id"]
        img = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Получаем аннотации для этого изображения
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Берем первую аннотацию (caption)
        if len(anns) > 0:
            caption = anns[0]["caption"]
        else:
            caption = "No caption available"

        if self.transform:
            img = self.transform(img)

        return {"caption": caption, "image": img}


if __name__ == "__main__":
    dataset = COCODataset(
        root="/home/jovyan/vasiliev/notebooks/Show-o/train2017",
        annFile="/home/jovyan/vasiliev/notebooks/Show-o/annotations/captions_train2017.json",
    )
    print(len(dataset))
    for i in range(100):
        print(dataset[i])
