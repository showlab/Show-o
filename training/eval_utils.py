import numpy as np
import torch
from PIL import Image
from accelerate.logging import get_logger
from PIL import ImageDraw, ImageFont
from utils import log_images_to_mlflow

from training.prompting_utils import (
    create_attention_mask_predict_next,
    create_attention_mask_for_mmu,
)
from training.activations_recorder import LayerOutputRecorder

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def visualize_predictions(
    model,
    vq_model,
    uni_prompting,
    config,
    global_step,
    input_ids,
    image_tokens_ori,
    ori_images,
    texts,
    logits,
    mlflow_client=None,
    mlflow_run_id=None,
):
    logger.info("Visualizing predictions...")
    model.eval()

    # Debug prints
    logger.info(f"🔍 DEBUG - input_ids shape: {input_ids.shape}")
    logger.info(f"🔍 DEBUG - logits shape: {logits.shape}")
    logger.info(f"🔍 DEBUG - image_tokens_ori shape: {image_tokens_ori.shape}")
    logger.info(f"🔍 DEBUG - ori_images shape: {ori_images.shape}")
    logger.info(f"🔍 DEBUG - batch_size_t2i: {config.training.batch_size_t2i}")
    logger.info(f"🔍 DEBUG - num_vq_tokens: {config.model.showo.num_vq_tokens}")

    recons_images = vq_model.decode_code(
        image_tokens_ori - len(uni_prompting.text_tokenizer)
    )
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    logger.info(f"🔍 DEBUG - recons_images shape: {recons_images.shape}")

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    logger.info(f"🔍 DEBUG - images shape: {images.shape}")

    # Используем фактическое количество VQ токенов из image_tokens_ori, а не из конфига
    actual_num_vq_tokens = image_tokens_ori.shape[1]
    logger.info(f"🔍 DEBUG - actual_num_vq_tokens: {actual_num_vq_tokens}")

    predictions = logits[
        : config.training.batch_size_t2i,
        -(actual_num_vq_tokens + 1) : -1 :,
        config.model.showo.llm_vocab_size
        + config.model.showo.num_new_special_tokens : -1,
    ]
    logger.info(f"🔍 DEBUG - predictions shape before argmax: {predictions.shape}")
    predictions = predictions.argmax(axis=-1)
    logger.info(f"🔍 DEBUG - predictions shape after argmax: {predictions.shape}")

    mask_token_id = (
        config.model.showo.vocab_size - 1 - len(uni_prompting.text_tokenizer)
    )
    logger.info(f"🔍 DEBUG - mask_token_id: {mask_token_id}")

    input_ids = input_ids[
        : config.training.batch_size_t2i, -(actual_num_vq_tokens + 1) : -1 :
    ] - len(uni_prompting.text_tokenizer)
    logger.info(f"🔍 DEBUG - input_ids (VQ part) shape: {input_ids.shape}")
    logger.info(f"🔍 DEBUG - input_ids min/max: {input_ids.min()}/{input_ids.max()}")

    mask_ratio = list(
        (
            torch.where(input_ids == mask_token_id, 1, 0).sum(dim=-1)
            / actual_num_vq_tokens
        )
        .cpu()
        .numpy()
    )
    logger.info(f"🔍 DEBUG - mask_ratio: {mask_ratio}")

    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
    logger.info(
        f"🔍 DEBUG - predicted_images shape before decode: {predicted_images.shape}"
    )
    logger.info(
        f"🔍 DEBUG - predicted_images min/max: {predicted_images.min()}/{predicted_images.max()}"
    )

    predicted_images = vq_model.decode_code(predicted_images)
    logger.info(
        f"🔍 DEBUG - predicted_images shape after decode: {predicted_images.shape}"
    )

    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = (
        predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    )
    logger.info(f"🔍 DEBUG - predicted_images final shape: {predicted_images.shape}")

    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    logger.info(f"🔍 DEBUG - concatenated shape: {predicted_images.shape}")

    pil_images = [Image.fromarray(image) for image in predicted_images]

    filenames = [
        f"prediction_{i}_step_{global_step}.png" for i in range(len(pil_images))
    ]
    log_images_to_mlflow(
        pil_images, filenames, "visualizations", mlflow_client, mlflow_run_id
    )

    for i, (ratio, text) in enumerate(zip(mask_ratio, texts)):
        logger.info(
            f"📊 Prediction {i} (mask ratio: {ratio:.2f}, text: {text[:50] if len(text) > 50 else text}...) logged to MLflow"
        )

    model.train()


@torch.no_grad()
def generate_images(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    mask_schedule,
    mlflow_client=None,
    mlflow_run_id=None,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    if hasattr(model, "module"):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
        mask_token_id = model.module.mask_token_id
        vocab_size = model.module.vocab_size
        codebook_size = model.module.config.codebook_size
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        mask_dtype = unwrapped_model.showo.model.embed_tokens.weight.dtype
        mask_token_id = unwrapped_model.mask_token_id
        vocab_size = unwrapped_model.vocab_size
        codebook_size = unwrapped_model.config.codebook_size
    
    logger.info(f"🎨 Generating images at step {global_step}")
    logger.info(f"   mask_token_id: {mask_token_id}")
    logger.info(f"   vocab_size: {vocab_size}")
    logger.info(f"   codebook_size: {codebook_size}")
    logger.info(f"   config.model.showo.vocab_size: {config.model.showo.vocab_size}")
    
    image_tokens = (
        torch.ones(
            (len(validation_prompts), config.model.showo.num_vq_tokens),
            dtype=torch.long,
            device=accelerator.device,
        )
        * mask_token_id
    )
    input_ids, _ = uni_prompting((validation_prompts, image_tokens), "t2i_gen")
    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(
            ([""] * len(validation_prompts), image_tokens), "t2i_gen"
        )
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        ).to(mask_dtype)
    else:
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        ).to(mask_dtype)
        uncond_input_ids = None

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast(
        "cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"
    ):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

        
    logger.info(f"   Generated tokens - min: {gen_token_ids.min().item()}, max: {gen_token_ids.max().item()}, mean: {gen_token_ids.float().mean().item():.2f}")
    logger.info(f"   Clamping to [0, {codebook_size - 1}]")
    
    gen_token_ids = torch.clamp(
        gen_token_ids,
        max=codebook_size - 1,
        min=0,
    )
    logger.info(f"   After clamp - min: {gen_token_ids.min().item()}, max: {gen_token_ids.max().item()}")
    
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    captioned_images = []
    for img, prompt in zip(pil_images, validation_prompts):
        img_width, img_height = img.size
        text_height = 40
        new_img = Image.new(
            "RGB", (img_width, img_height + text_height), (255, 255, 255)
        )
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except:
            font = ImageFont.load_default()
        prompt_text = prompt[:100] + "..." if len(prompt) > 100 else prompt
        draw.text((5, img_height + 5), prompt_text, fill=(0, 0, 0), font=font)
        captioned_images.append(new_img)

    filenames = [
        f"generated_{i}_step_{global_step}.png" for i in range(len(captioned_images))
    ]
    log_images_to_mlflow(
        captioned_images, filenames, "generated_images", mlflow_client, mlflow_run_id
    )

    logger.info(
        f"🎨 Сгенерированные изображения отправлены в MLflow: generated_images/"
    )


@torch.no_grad()
def evaluate_mmu(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    batch_mmu,
    mlflow_client=None,
    mlflow_run_id=None,
):
    logger.info("Generating captions for MMU evaluation...")
    model.eval()

    num_samples = min(4, batch_mmu["images"].shape[0])
    pixel_values_mmu = batch_mmu["images"][:num_samples].to(accelerator.device)
    labels_mmu = batch_mmu["labels"][:num_samples]

    ground_truth_texts = []
    for label_seq in labels_mmu:
        valid_tokens = label_seq[label_seq != -100]
        gt_text = uni_prompting.text_tokenizer.decode(
            valid_tokens, skip_special_tokens=True
        )
        ground_truth_texts.append(gt_text)

    image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
    image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    if hasattr(model, "module"):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = accelerator.unwrap_model(
            model
        ).showo.model.embed_tokens.weight.dtype

    generated_texts = []
    with torch.autocast(
        "cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"
    ):
        for i in range(num_samples):
            input_ids_single = torch.cat(
                [
                    (torch.ones(1, 1) * uni_prompting.sptids_dict["<|mmu|>"]).to(
                        accelerator.device
                    ),
                    (torch.ones(1, 1) * uni_prompting.sptids_dict["<|soi|>"]).to(
                        accelerator.device
                    ),
                    image_tokens_mmu[i : i + 1],
                    (torch.ones(1, 1) * uni_prompting.sptids_dict["<|eoi|>"]).to(
                        accelerator.device
                    ),
                ],
                dim=1,
            ).long()

            attention_mask_single = create_attention_mask_for_mmu(
                input_ids_single, eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"])
            ).to(mask_dtype)

            generated_ids = accelerator.unwrap_model(model).mmu_generate(
                idx=input_ids_single,
                attention_mask=attention_mask_single,
                max_new_tokens=config.training.get("mmu_max_new_tokens", 128),
                temperature=config.training.get("mmu_temperature", 0.7),
                top_k=config.training.get("mmu_top_k", None),
                eot_token=int(uni_prompting.sptids_dict["<|endoftext|>"])
                if "<|endoftext|>" in uni_prompting.sptids_dict
                else None,
            )

            generated_text = uni_prompting.text_tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            generated_texts.append(generated_text)

    images = torch.clamp((pixel_values_mmu + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    captioned_images = []
    for img, gt_text, gen_text in zip(images, ground_truth_texts, generated_texts):
        pil_img = Image.fromarray(img)
        img_width, img_height = pil_img.size
        text_height = 80
        new_img = Image.new(
            "RGB", (img_width, img_height + text_height), (255, 255, 255)
        )
        new_img.paste(pil_img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
        except:
            font = ImageFont.load_default()

        gt_text_str = gt_text[:80] + "..." if len(gt_text) > 80 else gt_text
        gen_text_str = gen_text[:80] + "..." if len(gen_text) > 80 else gen_text

        draw.text(
            (5, img_height + 5), f"GT: {gt_text_str}", fill=(0, 100, 0), font=font
        )
        draw.text(
            (5, img_height + 25), f"Gen: {gen_text_str}", fill=(0, 0, 200), font=font
        )
        captioned_images.append(new_img)

    filenames = [
        f"mmu_eval_{i}_step_{global_step}.png" for i in range(len(captioned_images))
    ]
    log_images_to_mlflow(
        captioned_images, filenames, "mmu_evaluations", mlflow_client, mlflow_run_id
    )

    logger.info(f"MMU evaluation отправлен в MLflow: mmu_evaluations/")

    for i, (gt, gen) in enumerate(zip(ground_truth_texts, generated_texts)):
        logger.info(f"  Example {i}: GT='{gt[:50]}...' | Gen='{gen[:50]}...'")

    model.train()


def log_grad_norm(
    model, accelerator, global_step, mlflow_client=None, mlflow_run_id=None
):
    import torch
    from collections import defaultdict
    import re

    layer_grad_norms = defaultdict(list)
    layer_param_norms = defaultdict(dict)
    moe_grad_norms = []
    base_grad_norms = []
    embedding_grad_norms = []
    other_grad_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()

            is_moe = any(x in name for x in ["mlp.experts", "mlp.gate", "mlp.alpha"])
            is_embedding = "embed" in name.lower()

            # Извлекаем информацию о слое и типе параметра
            match = re.search(r"layers\.(\d+)\.", name)
            layer_idx = int(match.group(1)) if match else None

            param_type = None
            if "self_attn.q_proj" in name:
                param_type = "q_proj"
            elif "self_attn.k_proj" in name:
                param_type = "k_proj"
            elif "self_attn.v_proj" in name:
                param_type = "v_proj"
            elif "self_attn.dense" in name:
                param_type = "attn_dense"
            elif "mlp.experts" in name:
                expert_match = re.search(r"experts\.(\d+)\.", name)
                if expert_match:
                    expert_id = expert_match.group(1)
                    param_type = f"expert_{expert_id}"
            elif "mlp.gate" in name:
                param_type = "moe_gate"
            elif "mlp.alpha" in name:
                param_type = "moe_alpha"
            elif "input_layernorm" in name:
                param_type = "input_layernorm"
            elif "post_attention_layernorm" in name:
                param_type = "post_attn_layernorm"

            if is_moe:
                moe_grad_norms.append(grad_norm)
                if layer_idx is not None:
                    layer_grad_norms[f"layer_{layer_idx}_moe"].append(grad_norm)
                    if param_type:
                        layer_param_norms[layer_idx][f"moe_{param_type}"] = grad_norm
            elif is_embedding:
                embedding_grad_norms.append(grad_norm)
            elif "showo.model.layers" in name:
                base_grad_norms.append(grad_norm)
                if layer_idx is not None:
                    layer_grad_norms[f"layer_{layer_idx}_base"].append(grad_norm)
                    if param_type:
                        layer_param_norms[layer_idx][param_type] = grad_norm
            else:
                other_grad_norms.append(grad_norm)

    metrics = {}

    if moe_grad_norms:
        metrics["grad_norm/moe_mean"] = sum(moe_grad_norms) / len(moe_grad_norms)
        metrics["grad_norm/moe_max"] = max(moe_grad_norms)
        metrics["grad_norm/moe_min"] = min(moe_grad_norms)

    if base_grad_norms:
        metrics["grad_norm/base_mean"] = sum(base_grad_norms) / len(base_grad_norms)
        metrics["grad_norm/base_max"] = max(base_grad_norms)
        metrics["grad_norm/base_min"] = min(base_grad_norms)

    if embedding_grad_norms:
        metrics["grad_norm/embedding_mean"] = sum(embedding_grad_norms) / len(
            embedding_grad_norms
        )

    if other_grad_norms:
        metrics["grad_norm/other_mean"] = sum(other_grad_norms) / len(other_grad_norms)

    for layer_name, norms in layer_grad_norms.items():
        if norms:
            metrics[f"grad_norm/{layer_name}_mean"] = sum(norms) / len(norms)

    for layer_idx, param_dict in layer_param_norms.items():
        for param_name, grad_value in param_dict.items():
            metrics[f"grad_norm/layer_{layer_idx}/{param_name}"] = grad_value

    if mlflow_client is not None and mlflow_run_id is not None:
        for metric_name, metric_value in metrics.items():
            mlflow_client.log_metric(
                mlflow_run_id, metric_name, metric_value, step=global_step
            )

    if hasattr(accelerator, "trackers") and len(accelerator.trackers) > 0:
        accelerator.log(metrics, step=global_step)

    logger.info(f"📊 Gradient norms at step {global_step}:")
    if moe_grad_norms:
        logger.info(
            f"   MoE: mean={metrics.get('grad_norm/moe_mean', 0):.6f}, max={metrics.get('grad_norm/moe_max', 0):.6f}"
        )
    if base_grad_norms:
        logger.info(
            f"   Base: mean={metrics.get('grad_norm/base_mean', 0):.6f}, max={metrics.get('grad_norm/base_max', 0):.6f}"
        )

    sample_layers = sorted(layer_param_norms.keys())[:3]
    if sample_layers:
        logger.info(f"   Sample layers {sample_layers}:")
        for layer_idx in sample_layers:
            params_info = ", ".join(
                [
                    f"{k}={v:.6f}"
                    for k, v in sorted(layer_param_norms[layer_idx].items())[:4]
                ]
            )
            logger.info(f"      Layer {layer_idx}: {params_info}")


def log_training_metrics(
    avg_loss_t2i,
    avg_loss_lm,
    avg_loss_mmu,
    balance_loss,
    balance_coeff,
    avg_masking_rate,
    lr_scheduler,
    batch_time_m,
    data_time_m,
    samples_per_second_per_gpu,
    global_step,
    mlflow_client=None,
    mlflow_run_id=None,
    logger=None,
):
    all_lrs = lr_scheduler.get_last_lr()

    logs = {
        "step_loss_t2i": avg_loss_t2i.item(),
        "step_loss_mmu": avg_loss_mmu.item(),
        "step_loss_lm": avg_loss_lm.item(),
        "step_loss_balance": balance_loss.item(),
        "balance_coeff": balance_coeff,
        "avg_masking_rate": avg_masking_rate.item(),
        "samples/sec/gpu": samples_per_second_per_gpu,
        "data_time": data_time_m.val,
        "batch_time": batch_time_m.val,
    }

    for idx, lr_val in enumerate(all_lrs):
        logs[f"lr_group_{idx}"] = lr_val

    if len(all_lrs) >= 4:
        logs["lr_moe"] = all_lrs[0]
        logs["lr_base"] = all_lrs[2]
    else:
        logs["lr"] = all_lrs[0]

    if mlflow_client is not None and mlflow_run_id is not None:
        for metric_name, metric_value in logs.items():
            mlflow_client.log_metric(
                mlflow_run_id, metric_name, metric_value, step=global_step
            )

    if logger is not None:
        if len(all_lrs) >= 4:
            lr_info = f"LR_moe: {all_lrs[0]:0.6f} LR_base: {all_lrs[2]:0.6f}"
        else:
            lr_info = f"LR: {all_lrs[0]:0.6f}"

        logger.info(
            f"Step: {global_step} "
            f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
            f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
            f"Loss_lm: {avg_loss_lm.item():0.4f} "
            f"Loss_balance: {balance_loss.item():0.6f} "
            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
            f"Batch (t): {batch_time_m.val:0.4f} "
            f"{lr_info}"
        )


@torch.no_grad()
def collect_and_log_moe_activations(
    model,
    accelerator,
    input_ids,
    attention_mask,
    labels,
    config,
    batch_size_t2i,
    batch_size_lm,
    batch_size_mmu,
    global_step,
    mlflow_client=None,
    mlflow_run_id=None,
):
    recorder = LayerOutputRecorder(device=accelerator.device)
    unwrapped_model = accelerator.unwrap_model(model)
    moe_modules = []
    for layer_idx, layer in enumerate(unwrapped_model.showo.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe_modules.append((f"layer_{layer_idx}", layer.mlp))

    if moe_modules:
        recorder.register_hooks(moe_modules)
        _ = model(
            input_ids=input_ids,
            input_embeddings=None,
            attention_mask=attention_mask,
            labels=labels,
            label_smoothing=config.training.label_smoothing,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            max_seq_length=config.dataset.preprocessing.max_seq_length,
        )
        recorder.remove_hooks()
        recorder.clear()
        del recorder

        logger.info(f"📊 Статистики активаций MoE собраны")
