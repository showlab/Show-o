import json
import os
import torch
from torchvision import transforms
import PIL.Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.llava import conversation as conversation_lib
from llava.data.utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
                raise NotImplementedError
    return source


def image_transform(image, resolution=256):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)(image)
    # get crop coordinates
    # c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
    # image = transforms.functional.crop(image, c_top, c_left, resolution, resolution)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    # added by xavier on June 26, morning
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


class VQADatasetEpoch(torch.utils.data.Dataset):
    def __init__(
        self,
        image_size=256,
    ):
        self.image_size = image_size
        data_file_path = "/mnt/bn/vgfm2/test_dit/blip_laion_cc_sbu_558k.json"
        self.vqa_image_root = os.path.join("/mnt/bn/vgfm2/test_dit/pretraining_data")

        with open(data_file_path, 'r') as f:
            vqa_data = json.load(f)
        self.vqa_data = []
        for item in vqa_data:
            if 'image' in item.keys():
                self.vqa_data.append(item)

        print("LLaVA Instruction Tuning dataset loaded.")

    def __len__(self):
        return len(self.vqa_data)

    def __getitem__(self, idx):
        try:
            item = self.vqa_data[idx]
            image_path = os.path.join(self.vqa_image_root, item["image"])

            image_clip = PIL.Image.open(image_path).convert("RGB")
            image_clip = image_transform(image_clip, resolution=self.image_size)

            conv = conversation_lib.default_conversation.copy()
            source = item["conversations"]
            source = preprocess_multimodal(
                source,
                mm_use_im_start_end=False
            )

            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conversations = []
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
            questions = conversations

            return (
                image_path,
                image_clip,
                image_clip,
                conversations,
                questions,
                torch.Tensor([-1]),
                "understanding",
                False
            )

        except Exception as e:
            print(e)
            return self.__getitem__(idx + 1)

