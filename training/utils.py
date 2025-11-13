import math
import random
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union
from torch.optim import AdamW
import tempfile
import os
import shutil

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
        return [
            (f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)
        ]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [
            (f"{key}.{idx}", v1)
            for idx, v1 in flatten_omega_conf(value, resolve=resolve)
        ]

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


def log_images_to_mlflow(
    pil_images, filenames, artifact_path, mlflow_client=None, mlflow_run_id=None
):
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


##################################################
#              training utils
##################################################
def soft_target_cross_entropy(logits, targets, soft_targets):
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(
    image_tokens, mask_id, config, mask_schedule, is_train=True
):
    batch_size, seq_len = image_tokens.shape

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get(
        "mask_contiguous_region_prob", None
    )

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(
            batch_size, seq_len, device=image_tokens.device
        ).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len**0.5)
        mask = torch.zeros(
            (batch_size, resolution, resolution), device=image_tokens.device
        )

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution),
                min(resolution, num_token_masked_),
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(
                num_token_masked_ / num_token_masked_height
            )
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
                batch_idx,
                start_idx_height : start_idx_height + num_token_masked_height,
                start_idx_width : start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens,
            low=0,
            high=config.model.codebook_size,
            device=image_tokens.device,
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
        config.training.get("predict_all_tokens", False)
        or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob


##################################################
#              misc
##################################################
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


from torchvision import transforms


def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.BICUBIC
    )(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )(image)
    return image


def get_optimizer(optimizer_config, named_parameters, logger, moe_config=None):
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_params = optimizer_config.params
    optimizer_type = optimizer_config.name
    
    # Приоритет: moe.learning_rate > optimizer.params.moe_learning_rate > optimizer.params.learning_rate
    if moe_config is not None and "learning_rate" in moe_config and moe_config["learning_rate"] is not None:
        moe_lr = float(moe_config["learning_rate"])
    else:
        moe_lr = optimizer_params.get("moe_learning_rate", optimizer_params.learning_rate)
    
    base_lr = optimizer_params.learning_rate

    moe_params_decay = []
    moe_params_no_decay = []
    base_params_decay = []
    base_params_no_decay = []

    for n, p in named_parameters:
        if not p.requires_grad:
            continue

        is_moe = any(x in n for x in ["mlp.experts", "mlp.gate", "mlp.alpha"])
        has_no_decay = any(nd in n for nd in no_decay)

        if is_moe:
            if has_no_decay:
                moe_params_no_decay.append(p)
            else:
                moe_params_decay.append(p)
        else:
            if has_no_decay:
                base_params_no_decay.append(p)
            else:
                base_params_decay.append(p)

    optimizer_grouped_parameters = [
        {
            "params": moe_params_decay,
            "weight_decay": optimizer_params.weight_decay,
            "lr": moe_lr,
        },
        {
            "params": moe_params_no_decay,
            "weight_decay": 0.0,
            "lr": moe_lr,
        },
        {
            "params": base_params_decay,
            "weight_decay": optimizer_params.weight_decay,
            "lr": base_lr,
        },
        {
            "params": base_params_no_decay,
            "weight_decay": 0.0,
            "lr": base_lr,
        },
    ]

    logger.info(f"Optimizer groups:")
    logger.info(f"  MoE params (decay): {len(moe_params_decay)} tensors, lr={moe_lr}")
    logger.info(
        f"  MoE params (no decay): {len(moe_params_no_decay)} tensors, lr={moe_lr}"
    )
    logger.info(
        f"  Base params (decay): {len(base_params_decay)} tensors, lr={base_lr}"
    )
    logger.info(
        f"  Base params (no decay): {len(base_params_no_decay)} tensors, lr={base_lr}"
    )

    if optimizer_type != "adamw":
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(optimizer_params.beta1, optimizer_params.beta2),
        eps=optimizer_params.epsilon,
    )

    return optimizer
