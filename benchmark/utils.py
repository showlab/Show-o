import os
import sys

import torch
import torch.nn.functional as F
from inference_t2i import get_model, get_vq_model_class
from torchvision import transforms
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import numpy as np
import clip
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance
from models import Showo, MAGVITv2, get_mask_chedule
from models import Showo
from training.utils import get_config
from PIL import Image, ImageDraw, ImageFont


def create_comparison_image(
    generated_image, target_image, caption, step_idx, img_idx, output_dir
):
    target_image = target_image.resize(generated_image.size, Image.Resampling.LANCZOS)
    img_width, img_height = generated_image.size
    canvas_width = img_width * 2 + 20  # 20px между изображениями
    canvas_height = img_height + 100  # 100px для текста
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(generated_image, (0, 50))
    canvas.paste(target_image, (img_width + 20, 50))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), "Generated", fill="black", font=font)
    draw.text((img_width + 30, 10), "Target", fill="black", font=font)
    caption_lines = []
    words = caption.split()
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width > canvas_width - 20:
            if current_line:
                caption_lines.append(current_line)
                current_line = word
            else:
                caption_lines.append(word)
        else:
            current_line = test_line

    if current_line:
        caption_lines.append(current_line)

    y_offset = img_height + 60
    for line in caption_lines:
        draw.text((10, y_offset), line, fill="black", font=font)
        y_offset += 20

    filename = f"comparison_step_{step_idx}_img_{img_idx}.png"
    filepath = os.path.join(output_dir, filename)
    canvas.save(filepath)
    return filepath
