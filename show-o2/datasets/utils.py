import copy
import math
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode


def image_transform(image, resolution=256, normalize=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                    y0_centercrop=False):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    if y0_centercrop:
        width, height = image.size
        left = (width - resolution) / 2
        top = 0
        right = (width + resolution) / 2
        bottom = resolution
        image = image.crop((left, top, right, bottom))
    else:
        image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=mean, std=std, inplace=True)(image)
    return image


def to_tensor_and_normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=mean, std=std, inplace=True)(image)
    return image


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '').replace("This image showcases ", '').replace("This suggests ", '').replace(
        "In the photo, we see ", '').replace("This is ", '').replace("This image is ", '').replace(
        "In the photo, we have ", '').replace("The photo features ", '').replace("The photo depicts ", '').replace(
        "The photo appears to be ", '')

    return caption


# At this time, we do not model the text in image-text pairs for t2i
def format_sequence_gen_qwen2_5(text_tokens, system_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, system_token_len):
    if system_token_len == 0:
        modality_positions = torch.tensor([[len(text_tokens) + 1 + 1, num_image_tokens]])
        # text_labels = [bos_id] + [-100] * len(text_tokens) + [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        # text_labels = [bos_id] + text_tokens + [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        text_labels = [-100] + [-100] * len(text_tokens) + [-100] + [-100] * num_image_tokens + [-100] + [-100]
        text_tokens = [bos_id] + text_tokens + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]
    else:
        # TODO TO BE VERIFIED
        modality_positions = torch.tensor([[1 + system_token_len + len(text_tokens) + 1 + 1, num_image_tokens]])
        text_labels = [bos_id] + [-100] * len(system_tokens[0] + system_tokens[1] + text_tokens) + [eos_id] + \
                      [-100] * len(system_tokens[2]) + \
                      [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        text_tokens = [bos_id] + system_tokens[0] + system_tokens[1] + text_tokens + [eos_id] + system_tokens[2] + \
                      [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask

def format_sequence_und(text_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                        num_image_tokens, max_seq_len):
    modality_positions = torch.tensor([[1 + 1, num_image_tokens]])

    text_labels = [bos_id] + [boi_id] + [-100] * num_image_tokens + [eoi_id] + \
                  text_tokens + [eos_id]

    text_tokens = [bos_id] + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + \
                  text_tokens + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def format_interleaved_sequence(image_list, text_token_list, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, max_num_images, system_tokens=None, system_token_len=0):
    """
    # generation
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    # eg. 0        1-9           10          11-15        16         17
    # understanding
    # [bos_id, im_start, image_tokens, im_end, text_tokens, eos_id, pad_id]
    # eg. 0        1            2-6           7           8-16       17
    """

    text_tokens = []
    text_labels = []
    modality_positions = []

    cur_len = 1 + system_token_len # bos token
    for txt_token, image in zip(text_token_list, image_list):
        if txt_token is not None:
            text_tokens.extend(txt_token)
            text_labels.extend(copy.deepcopy(txt_token))
            cur_len += len(txt_token)

        if image is not None:
            text_tokens.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
            text_labels.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
            # +1 for one <|img_start|> token
            modality_positions.append((cur_len + 1, num_image_tokens))
            cur_len = cur_len + 1 + num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

    if system_token_len == 0:
        text_labels = [bos_id] + text_labels + [eos_id]
        text_tokens = [bos_id] + text_tokens + [eos_id]
    else:
        # TODO TO BE VERIFIED
        text_labels = [bos_id] + [-100] * system_token_len + text_labels + [eos_id]
        text_tokens = [bos_id] + system_tokens[0] + system_tokens[1] + system_tokens[2] + text_tokens + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    if len(modality_positions) < max_num_images:
        modality_positions += [(0, 0) for _ in range(max_num_images - len(modality_positions))]

    modality_positions = torch.tensor(modality_positions)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def resize_crop(image, image_height, image_width):
    aspect_ratio = image_width / image_height
    if isinstance(image, torch.Tensor) and image.ndim == 4:
        frame_height, frame_width = image[0].size(1), image[0].size(2)
        original_size_as_tuple = torch.tensor([frame_height, frame_width])
        image_aspect_ratio = frame_width / frame_height
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (frame_width / frame_height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (frame_height / frame_width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                           interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                         image_width)
    else:
        frame_height, frame_width = image.size(1), image.size(2)
        image_aspect_ratio = frame_width / frame_height
        original_size_as_tuple = torch.tensor([frame_height, frame_width])
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (frame_width / frame_height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (frame_height / frame_width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                           interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                         image_width)
    crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
    return image, original_size_as_tuple, crop_coords_top_left


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image
