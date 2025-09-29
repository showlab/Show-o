# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import wandb
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize

# set_seed(10)

logger = get_logger(__name__, log_level="INFO")

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name,
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.float32

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path,
                                                         add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type)
    model.eval()

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)

    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    if not (config.mmu_image_path.endswith('.jpg') or config.mmu_image_path.endswith('.png')):
        file_list = [os.path.join(config.mmu_image_path, fn) for fn in os.listdir(config.mmu_image_path)]
    else:
        file_list = [config.mmu_image_path]

    config.question = config.question.split(' *** ')

    sys_prompt_ids = text_tokenizer("system\nYou are a helpful assistant.<|im_end|>",
                                    add_special_tokens=False)['input_ids']
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

    for step, image_path in enumerate(tqdm(file_list)):
        image_ori = Image.open(image_path).convert("RGB")
        # not center cropping
        # image = resize_and_pad_image(image, target_resolution=(config.dataset.preprocessing.resolution,
        #                                                        config.dataset.preprocessing.resolution))
        # image = to_tensor_and_normalize(image)
        # center crop
        image = image_transform(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
        image = image.unsqueeze(0)

        image_latents = vae_model.sample(image.unsqueeze(2)).squeeze(2).to(weight_type)

        image_embeds_und = model.image_embedder_und(image_latents)
        image_embeds_gen = model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
        image_embeds_und = model.und_trans(image_embeds_und)['last_hidden_state']
        image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        batch_size = 1
        responses = ['' for j in range(len(file_list))]
        images = [image]
        for j, question in enumerate(config.question):
            input_ids = text_tokenizer(question, add_special_tokens=False).input_ids
            text_tokens_a = torch.tensor([showo_token_ids['bos_id']] + sys_prompt_ids + role_a).to(device)[None, :]
            text_tokens_b = torch.tensor([showo_token_ids['boi_id'], showo_token_ids['eoi_id']] + input_ids + role_b).to(device)[None, :]
            text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
            text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

            if config.model.showo.add_time_embeds:
                time_embeds = model.time_embed(torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype)
                if hasattr(model, 'time_embed_proj'):
                    time_embeds = model.time_embed_proj(time_embeds)
                input_embeds = torch.cat([
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    time_embeds,
                    image_embeds,
                    text_embeds_b[:, 1:]
                ], dim=1).to(weight_type)
                modality_positions = torch.tensor([text_tokens_a.shape[1] + 2, num_mmu_image_tokens])[None, None, :].to(device)
            else:
                input_embeds = torch.cat([
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    image_embeds,
                    text_embeds_b[:, 1:]
                ], dim=1).to(weight_type)
                modality_positions = torch.tensor([text_tokens_a.shape[1] + 1, num_mmu_image_tokens])[None, None, :].to(device)

            attention_mask = omni_attn_mask_naive(
                B=input_embeds.size(0),
                LEN=input_embeds.size(1),
                modalities=modality_positions,
                device=device, inverted=True
            ).to(input_embeds.dtype)

            output_tokens = model.mmu_generate(input_embeds=input_embeds,
                                                     attention_mask=attention_mask,
                                                     top_k=top_k,
                                                     max_new_tokens=300,
                                                     eos_token=text_tokenizer.eos_token_id)

            output_tokens = torch.stack(output_tokens).squeeze()[None]

        text = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        responses[j] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'

        images = torch.cat(images, dim=0)
        images = denorm(images)
        pil_images = [Image.fromarray(image) for image in images]

        wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
        wandb.log({"Multimodal understanding responses": wandb_images}, step=step)


