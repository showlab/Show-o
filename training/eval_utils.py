import os
import shutil
import tempfile
import logging

import numpy as np
import torch
from PIL import Image
from accelerate.logging import get_logger
from PIL import ImageDraw, ImageFont

from training.prompting_utils import create_attention_mask_predict_next
from moe_utils import LayerOutputRecorder, log_stats_to_mlflow
from mlflow.tracking import MlflowClient

logger = get_logger(__name__, log_level="INFO")


def log_images_to_mlflow(pil_images, filenames, artifact_path, mlflow_client=None, mlflow_run_id=None):
    temp_dir = tempfile.mkdtemp()
    
    for image, filename in zip(pil_images, filenames):
        image_path = os.path.join(temp_dir, filename)
        image.save(image_path)

        if mlflow_client is not None and mlflow_run_id is not None:
            client = mlflow_client
            run_id = mlflow_run_id
        else:
            import mlflow
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            if run_id is None:
                continue
            client = MlflowClient()
        
        client.log_artifact(run_id, image_path, artifact_path)
    
    shutil.rmtree(temp_dir)


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

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
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

    predictions = logits[:config.training.batch_size_t2i, -(actual_num_vq_tokens + 1):-1:,
                  config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens:-1]
    logger.info(f"🔍 DEBUG - predictions shape before argmax: {predictions.shape}")
    predictions = predictions.argmax(axis=-1)
    logger.info(f"🔍 DEBUG - predictions shape after argmax: {predictions.shape}")

    mask_token_id = config.model.showo.vocab_size - 1 - len(uni_prompting.text_tokenizer)
    logger.info(f"🔍 DEBUG - mask_token_id: {mask_token_id}")
    
    input_ids = input_ids[:config.training.batch_size_t2i, -(actual_num_vq_tokens + 1):-1:] - len(
        uni_prompting.text_tokenizer)
    logger.info(f"🔍 DEBUG - input_ids (VQ part) shape: {input_ids.shape}")
    logger.info(f"🔍 DEBUG - input_ids min/max: {input_ids.min()}/{input_ids.max()}")
    
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / actual_num_vq_tokens).cpu().numpy())
    logger.info(f"🔍 DEBUG - mask_ratio: {mask_ratio}")
    
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
    logger.info(f"🔍 DEBUG - predicted_images shape before decode: {predicted_images.shape}")
    logger.info(f"🔍 DEBUG - predicted_images min/max: {predicted_images.min()}/{predicted_images.max()}")

    predicted_images = vq_model.decode_code(predicted_images)
    logger.info(f"🔍 DEBUG - predicted_images shape after decode: {predicted_images.shape}")
    
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    logger.info(f"🔍 DEBUG - predicted_images final shape: {predicted_images.shape}")
    
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    logger.info(f"🔍 DEBUG - concatenated shape: {predicted_images.shape}")
    
    pil_images = [Image.fromarray(image) for image in predicted_images]

    filenames = [f"prediction_{i}_step_{global_step}.png" for i in range(len(pil_images))]
    log_images_to_mlflow(pil_images, filenames, "visualizations", mlflow_client, mlflow_run_id)
    
    for i, (ratio, text) in enumerate(zip(mask_ratio, texts)):
        logger.info(f"📊 Prediction {i} (mask ratio: {ratio:.2f}, text: {text[:50] if len(text) > 50 else text}...) logged to MLflow")

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

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = accelerator.unwrap_model(model).showo.model.embed_tokens.weight.dtype

    mask_token_id = config.model.showo.vocab_size - 1
    image_tokens = torch.ones((len(validation_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, _ = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
    else:
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
        uncond_input_ids = None

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
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
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
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
        new_img = Image.new('RGB', (img_width, img_height + text_height), (255, 255, 255))
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        prompt_text = prompt[:100] + "..." if len(prompt) > 100 else prompt
        draw.text((5, img_height + 5), prompt_text, fill=(0, 0, 0), font=font)
        captioned_images.append(new_img)

    filenames = [f"generated_{i}_step_{global_step}.png" for i in range(len(captioned_images))]
    log_images_to_mlflow(captioned_images, filenames, "generated_images", mlflow_client, mlflow_run_id)
    
    logger.info(f"🎨 Сгенерированные изображения отправлены в MLflow: generated_images/")


def log_grad_norm(model, accelerator, global_step):
    if hasattr(accelerator, 'trackers') and len(accelerator.trackers) > 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads = param.grad.detach().data
                grad_norm = (grads.norm(p=2) / grads.numel()).item()
                accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


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
    """Логирует метрики обучения в MLflow и консоль."""
    logs = {
        "step_loss_t2i": avg_loss_t2i.item(),
        "step_loss_mmu": avg_loss_mmu.item(),
        "step_loss_lm": avg_loss_lm.item(),
        "step_loss_balance": balance_loss.item(),
        "balance_coeff": balance_coeff,
        "lr": lr_scheduler.get_last_lr()[0],
        "avg_masking_rate": avg_masking_rate.item(),
        "samples/sec/gpu": samples_per_second_per_gpu,
        "data_time": data_time_m.val,
        "batch_time": batch_time_m.val,
    }
    
    if mlflow_client is not None and mlflow_run_id is not None:
        for metric_name, metric_value in logs.items():
            mlflow_client.log_metric(mlflow_run_id, metric_name, metric_value, step=global_step)
    
    if logger is not None:
        logger.info(
            f"Step: {global_step} "
            f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
            f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
            f"Loss_lm: {avg_loss_lm.item():0.4f} "
            f"Loss_balance: {balance_loss.item():0.6f} "
            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
            f"Batch (t): {batch_time_m.val:0.4f} "
            f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
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
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
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
        if mlflow_client is not None and mlflow_run_id is not None:
            log_stats_to_mlflow(recorder, mlflow_client, mlflow_run_id, global_step, "moe_activations")
        recorder.remove_hooks()
        recorder.clear()
        del recorder
        
        logger.info(f"📊 Статистики активаций MoE собраны и отправлены в MLflow")

