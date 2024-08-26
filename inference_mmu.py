import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGViTv2
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit
from training.utils import get_config, flatten_omega_conf
from transformers import AutoTokenizer
from torchvision import transforms
from models.clip_encoder import CLIPVisionTower
from transformers import CLIPImageProcessor

import training.conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28

def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGViTv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

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
        name=config.experiment.name + '_mmu',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("**soi**", "**eoi**", "**sov**", "**eov**", "**t2i**", "**mmu**", "**t2v**", "**v2v**", "**lvg**"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)().to(device)
    vq_model.load_state_dict(torch.load(config.model.vq_model.pretrained))
    vq_model.requires_grad_(False)
    vq_model.eval()

    vision_tower_name = "openai/clip-vit-large-patch14-336"
    vision_tower =  CLIPVisionTower(vision_tower_name).to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

    model = Showo(**config.model.showo).to(device)
    state_dict = torch.load(config.pretrained_model_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    del state_dict

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    file_list = os.listdir(config.mmu_image_root)
    responses = ['' for i in range(len(file_list))]
    images = []
    config.question = config.question.split(' *** ')
    for i, file_name in enumerate(tqdm(file_list)):
        image_path = os.path.join(config.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        images.append(image)

        pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1

        for question in config.question:
            if config.model.showo.w_clip_vit:
                conv = conversation_lib.default_conversation.copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input = []
                question_input.append(prompt_question.strip())

                input_ids_system = [uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids
                                        for _ in range(batch_size)]
                input_ids_system = torch.stack(input_ids_system, dim=0)
                assert input_ids_system.shape[-1] == 28
                input_ids_system = input_ids_system.to(device)
                input_ids_system = input_ids_system[0]

                input_ids = [uni_prompting.text_tokenizer(prompt, return_tensors="pt", padding="longest").input_ids
                                for prompt in question_input]

                input_ids = torch.stack(input_ids)
                input_ids = torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True, padding_value=uni_prompting.text_tokenizer.pad_token_id
                )
                input_ids = torch.tensor(input_ids).to(device).squeeze(0)
                # import pdb; pdb.set_trace()
                input_ids_llava = torch.cat([
                        (torch.ones(input_ids.shape[0], 1) *uni_prompting.sptids_dict['**mmu**']).to(device),
                        input_ids_system,
                        (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['**soi**']).to(device),
                        # place your img embedding here
                        (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['**eoi**']).to(device),
                        input_ids,
                ], dim=1).long()

                images_embeddings = vision_tower(pixel_values[None])
                images_embeddings = model.mm_projector(images_embeddings)

                text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

                # Full input seq
                part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
                part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)

                attention_mask_llava = create_attention_mask_for_mmu_vit(input_embeddings,
                                                                        system_prompt_len=SYSTEM_PROMPT_LEN)

                cont_toks_list = model.mmu_generate(input_embeddings=input_embeddings,
                                                    attention_mask=attention_mask_llava[0].unsqueeze(0),
                                                    max_new_tokens=100,
                                                    top_k=top_k,
                                                    eot_token=uni_prompting.sptids_dict['**eot**']
                                                    )
            else:
                input_ids = uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])[
                    'input_ids']
                input_ids = torch.tensor(input_ids).to(device)

                input_ids = torch.cat([
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['**mmu**']).to(device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['**soi**']).to(device),
                    image_tokens,
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['**eoi**']).to(device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['**sot**']).to(device),
                    input_ids
                ], dim=1).long()

                attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                               eoi_id=int(uni_prompting.sptids_dict['**eoi**']))

                cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask,
                                            max_new_tokens=100, top_k=top_k,
                                            eot_token=uni_prompting.sptids_dict['**eot**'])

            cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

            text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
            print(text)
            responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'

    images = torch.cat(images, dim=0)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
    wandb.log({"multimodal understanding": wandb_images}, step=0)
