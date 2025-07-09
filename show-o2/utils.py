from typing import Any, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from collections import OrderedDict
import random

##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              misc
##################################################

def _count_params(module):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in module.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def _freeze_params(model, frozen_params=None):
    if frozen_params is not None:
        for n, p in model.named_parameters():
            for name in frozen_params:
                if name in n:
                    p.requires_grad = False


path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": 'qwen2_5',
    "Qwen/Qwen2.5-1.5B-Instruct": 'qwen2_5',
    "meta-llama/Llama-3.2-1B-Instruct": 'llama3'
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denorm(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images

def denorm_vid(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    # B, C, T, H, W --> B, T, C, H, W
    images = images.permute(0, 2, 1, 3, 4).cpu().numpy().astype(np.uint8)
    return images


def get_hyper_params(config, text_tokenizer, showo_token_ids, is_video=False, is_hq=False):
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    max_seq_len = config.dataset.preprocessing.max_seq_length
    num_video_tokens = config.dataset.preprocessing.num_video_tokens
    if is_video:
        max_text_len = max_seq_len - num_video_tokens - 4
        latent_width = config.dataset.preprocessing.video_latent_width
        latent_height = config.dataset.preprocessing.video_latent_height
        num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
        num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
    else:
        if is_hq:
            latent_width = config.dataset.preprocessing.hq_latent_width
            latent_height = config.dataset.preprocessing.hq_latent_height
            num_t2i_image_tokens = config.dataset.preprocessing.num_hq_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            max_seq_len = config.dataset.preprocessing.max_hq_seq_length
            max_text_len = max_seq_len - num_t2i_image_tokens - 4
        else:
            num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            latent_width = config.dataset.preprocessing.latent_width
            latent_height = config.dataset.preprocessing.latent_height
            max_text_len = max_seq_len - num_t2i_image_tokens - 4

    image_latent_dim = config.model.showo.image_latent_dim
    patch_size = config.model.showo.patch_size

    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    boi_id = showo_token_ids['boi_id']
    eoi_id = showo_token_ids['eoi_id']
    bov_id = showo_token_ids['bov_id']
    eov_id = showo_token_ids['eov_id']
    img_pad_id = showo_token_ids['img_pad_id']
    vid_pad_id = showo_token_ids['vid_pad_id']

    guidance_scale = config.transport.guidance_scale

    return num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, \
           latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, \
           vid_pad_id, guidance_scale


# these save and recover functions are based on our internal packages
# please modified them when necessary
def save_dataloader_state(rank, loader, ckpt_path="./"):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    saved_state = deepcopy(loader.__getstate__())
    torch.save(saved_state, ckpt_path)

def recover_dataloader_state(rank, loader, ckpt_path='./'):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            loader_state_dict = torch.load(f)
            loader.__setstate__(loader_state_dict)
        print(f"rank {rank} loader state dict loaded successfully!")


def save_images_as_grid(pil_images, fn, path, grid_size=(2, 2)):

    os.makedirs(path, exist_ok=True)

    rows, cols = grid_size

    num_images = len(pil_images)
    if num_images > rows * cols:
        raise ValueError(f"Number of images ({num_images}) exceeds grid capacity ({rows * cols}).")

    img_width, img_height = pil_images[0].size

    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))  # 白色背景

    for idx, image in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x_offset = col * img_width
        y_offset = row * img_height
        grid_image.paste(image, (x_offset, y_offset))

    grid_image.save(os.path.join(path, f"{fn}.png"))

    return grid_image


def load_state_dict(model_path):
    if model_path.endswith(".bin"):
        state_dict = torch.load(model_path)
    else:
        checkpoint_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.bin')]
        )

        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            state_dict.update(checkpoint)

    return state_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)