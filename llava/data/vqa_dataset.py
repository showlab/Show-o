import json
import os
import random

import cv2
import torch
from transformers import CLIPImageProcessor
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import sys
# sys.path.append("..")
# DEFAULT_IMAGE_TOKEN = "<image>"

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

from torchvision import transforms
import PIL.Image
# def image_transform(image, resolution=256):
#     image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)(image)
#     # get crop coordinates
#     c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
#     image = transforms.functional.crop(image, c_top, c_left, resolution, resolution)
#     image = transforms.ToTensor()(image)
#     return image

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

class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        batch_size=4,
        image_size=256,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # data_file_path =  "/mnt/bn/vgfm2/cc3m_vlp/chat.json"
        # self.vqa_image_root = os.path.join("/mnt/bn/vgfm2/cc3m_vlp/images")

        data_file_path = "/mnt/bn/vgfm2/test_dit/llava_v1_5_mix665k.json"
        self.vqa_image_root = os.path.join("/mnt/bn/vgfm2/test_dit/tuning_data")
        #
        # data_file_path = "/mnt/bn/vgfm2/test_dit/blip_laion_cc_sbu_558k.json"
        # self.vqa_image_root = os.path.join("/mnt/bn/vgfm2/test_dit/pretraining_data")

        with open(data_file_path, 'r') as f:
            vqa_data = json.load(f)
        self.vqa_data = []
        for item in vqa_data:
            if 'image' in item.keys():
                self.vqa_data.append(item)

        self.batch_size = batch_size
        self.image_size = image_size
        # import pdb; pdb.set_trace()
        with open("/mnt/bn/vgfm2/test_dit/LlmDiffuser_phi1.5/LlmDiffuser/questions.json") as f:
            self.caption_prompt = json.load(f)

        print("LLaVA Instruction Tuning dataset loaded.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_clip = PIL.Image.open(image_path).convert("RGB")
        image_clip = image_transform(image_clip, resolution=self.image_size)

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=False
        )
        # import ipdb
        # ipdb.set_trace()
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
        # print(conv.get_prompt())
        # conversations.append(conv.get_prompt())
        # remove the system prompt
        # my_systemt_prompt = random.sample(self.caption_prompt, 1)[0]
        conversations.append(conv.get_prompt().split("user's questions. ")[-1])
        # print(my_systemt_prompt)
        # print(conv.get_prompt().split("ASSISTANT: ")[-1])
        # import time
        # time.sleep(10)
        # conversations.append(my_systemt_prompt + ' ' + conv.get_prompt().split("ASSISTANT: ")[-1])


        questions = conversations

        return (
            image_path,
            image_clip,
            image_clip,
            conversations,
            questions,
            torch.Tensor([-1]),
            "understanding"
        )


if __name__ == '__main__':
    dataset = VQADataset(
        "openai/clip-vit-large-patch14-336",
        samples_per_epoch=500,
    )
    print("Length: ", len(dataset))
    item = dataset.__getitem__(0)
    import pdb
    pdb.set_trace()

