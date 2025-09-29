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
import torch.nn.functional as F
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, \
    set_seed, load_video
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

    # wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
    #
    # wandb.init(
    #     project="demo",
    #     name=config.experiment.name,
    #     config=wandb_config,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16

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

    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    if not any(config.mmu_video_path.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS):
        file_list = [
            os.path.join(config.mmu_video_path, fn)
            for fn in os.listdir(config.mmu_video_path)
            if any(fn.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)
        ]
    else:
        file_list = [config.mmu_video_path]

    config.question = config.question.split(' *** ')

    sys_prompt_ids = text_tokenizer("system\nYou are a helpful assistant.<|im_end|>",
                                    add_special_tokens=False)['input_ids']
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

    with torch.no_grad():
        for step, video_path in enumerate(tqdm(file_list)):
            video_frames, frame_time, video_time = load_video(video_path, config.num_video_frames_mmu,
                                                               fps=1, force_sample=True)
            for i in range(len(video_frames)):
                video_frames[i] = resize_and_pad_image(video_frames[i],
                                                       target_resolution=(config.dataset.preprocessing.resolution,
                                                                          config.dataset.preprocessing.resolution))
                video_frames[i] = to_tensor_and_normalize(video_frames[i], mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            video = torch.stack(video_frames, dim=0).to(device, non_blocking=True)

            image_latents = vae_model.sample(video.unsqueeze(2)).squeeze(2).to(weight_type)
            b, c, h, w = image_latents.shape
            p = config.model.showo.patch_size
            h_, w_ = h // p, w // p

            image_embeds_und = model.image_embedder_und(image_latents)
            image_embeds_gen = model.image_embedder_gen(image_latents)
            image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
            image_embeds_und = model.und_trans(image_embeds_und)['last_hidden_state']
            image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

            temp_image_embeds = image_embeds.reshape(config.num_video_frames_mmu, h_, w_,
                                                     image_embeds.size(-1)).permute(0, 3, 1, 2).contiguous()
            # downsample the frame size to get compressed 14x14=196 tokens
            temp_image_embeds = F.interpolate(temp_image_embeds, size=(14, 14), mode="bilinear")
            image_embeds = temp_image_embeds.permute(0, 2, 3, 1).reshape(-1, image_embeds.size(-1)).unsqueeze(0)

            batch_size = 1
            responses = ['' for j in range(len(file_list))]

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
            print(responses[j])

            with open('./vid_und_output.txt', 'a+') as file:
                file.write(responses[j])


