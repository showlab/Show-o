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

import copy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import wandb
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_mixed_modal_gen_input
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from transport import Sampler, create_transport

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

    # VQ model for processing image into discrete tokens
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
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(
            device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type)
    model.eval()

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    num_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)

    num_mixed_modal_tokens = config.dataset.preprocessing.num_mixed_modal_tokens
    latent_height = config.dataset.preprocessing.mixed_modal_latent_height
    latent_width = config.dataset.preprocessing.mixed_modal_latent_width

    # load users passed arguments
    batch_size = config.batch_size
    guidance_scale = config.guidance_scale
    config.transport.num_inference_steps = config.num_inference_steps
    # config.transport.sampling_method = config.sampling_method
    if config.get("validation_prompts_file", None) is not None:
        validation_prompts_file = config.validation_prompts_file
    # load from users passed arguments

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=config.dataset.preprocessing.num_mixed_modal_tokens,
    )  # default: velocity;

    sampler = Sampler(transport)

    config.batch_size = 1
    results = []
    for step in tqdm(range(len(validation_prompts))):
        text_history = validation_prompts[step]
        text_history_null = ""

        image_latents_history = []
        modality_positions_history = None
        modality_positions_null_history = None
        responses_history = []

        for i in range(4):
            if not text_history.startswith("<|im_start|>"):
                text_history = "<|im_start|>" + text_history

            input_ids = text_tokenizer(text_history, add_special_tokens=False).input_ids

            if "<|image_pad|>" in text_history:
                attention_mask = omni_attn_mask_naive(
                                                      config.batch_size, len(input_ids),
                                                      modality_positions_history,
                                                      device).to(torch.bfloat16)

                response = model.mm_generate(
                    input_ids=input_ids,
                    image_latents=torch.cat(image_latents_history, dim=0).to(torch.bfloat16),
                    attention_mask=attention_mask,
                    t=torch.zeros((len(image_latents_history))).to(device),
                    modality_positions=modality_positions_history,
                    tokenizer=text_tokenizer,
                    max_new_tokens=200,
                    boi_token=boi_id,
                    device=device
                )

            else:
                response = model.lm_generate(
                    input_ids=input_ids,
                    tokenizer=text_tokenizer,
                    max_new_tokens=200,
                    boi_token=boi_id,
                    device=device
                )

            responses_history.append(response)
            text_history += response

            if text_history.endswith("<|vision_start|>"):
                text_history = text_history[len("<|im_start|>"):]
                text_history = text_history[:-len("<|vision_start|>")]
                text_history_null = text_history_null[len("<|im_start|>"):]
                text_history_null = text_history_null[:-len("<|vision_start|>")]

                batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
                    prepare_mixed_modal_gen_input(
                        [text_history], [text_history_null], text_tokenizer, num_mixed_modal_tokens, bos_id,
                        boi_id, eoi_id, pad_id, img_pad_id, device
                    )
                text_history = text_tokenizer.batch_decode(batch_text_tokens)[0].replace("[PAD]", "")
                text_history_null = text_tokenizer.batch_decode(batch_text_tokens_null)[0].replace("[PAD]", "")

                z = torch.randn((config.batch_size, image_latent_dim,
                                 latent_height * patch_size,
                                 latent_width * patch_size)).to(torch.bfloat16).to(device)

                if len(image_latents_history) != 0:
                    z_null = z.clone().repeat(len(image_latents_history) + 1, 1, 1, 1)
                    z = torch.cat(
                        [
                            *image_latents_history,
                            z
                        ], dim=0
                    )
                    batch_modality_positions = torch.cat(
                        [
                            modality_positions_history,
                            batch_modality_positions
                        ], dim=1
                    )

                    batch_modality_positions_null = torch.zeros_like(batch_modality_positions).long()
                    batch_modality_positions_null[:] = torch.tensor([2, num_image_tokens]).to(device)
                else:
                    z_null = z.clone()
                modality_positions_history = copy.deepcopy(batch_modality_positions)
                modality_positions_null_history = copy.deepcopy(modality_positions_null_history)

                if guidance_scale > 0:
                    z = torch.cat([z, z_null], dim=0)
                    text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
                    modality_positions = torch.cat([batch_modality_positions,
                                                    batch_modality_positions_null], dim=0)
                    block_mask = omni_attn_mask_naive(
                                                        text_tokens.shape[0], text_tokens.shape[1],
                                                        modality_positions,
                                                        device).to(torch.bfloat16)
                else:
                    text_tokens = batch_text_tokens
                    modality_positions = batch_modality_positions
                    block_mask = omni_attn_mask_naive(
                                                        text_tokens.shape[0], text_tokens.shape[1],
                                                        modality_positions,
                                                        device).to(torch.bfloat16)

                model_kwargs = dict(
                    text_tokens=text_tokens,
                    attention_mask=block_mask,
                    modality_positions=modality_positions,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    guidance_scale=guidance_scale if i == 0 else 5.0,
                    only_denoise_last_image=True,
                )

                sample_fn = sampler.sample_ode(
                    sampling_method=config.transport.sampling_method,
                    num_steps=config.transport.num_inference_steps,
                    atol=config.transport.atol,
                    rtol=config.transport.rtol,
                    reverse=config.transport.reverse,
                    time_shifting_factor=config.transport.time_shifting_factor
                )
                samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
                if guidance_scale > 0:
                    samples = torch.chunk(samples, 2)[0]
                image_latents_history.append(samples[-1:])

        samples = torch.cat(image_latents_history)
        if config.model.vae_model.type == 'wan21':
            samples = samples.unsqueeze(2)
            images = vae_model.batch_decode(samples)
            images = images.squeeze(2)
        else:
            raise NotImplementedError

        # Convert to PIL images
        images = denorm(images)
        pil_images = [Image.fromarray(image) for image in images]

        # Log images
        wandb_images = [wandb.Image(image, caption=responses_history[i]) for i, image in enumerate(pil_images)]
        wandb.log({"Generated images": wandb_images}, step=step)
