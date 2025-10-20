import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.utilities import CombinedLoader
from accelerate.logging import get_logger

from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from models.llava.llava_data_vq_unified import get_instruct_data_loader

logger = get_logger(__name__, log_level="INFO")

SYSTEM_PROMPT_LEN = 28


class DummyLMDataset:
    def __len__(self):
        return 1000000
    
    def __getitem__(self, idx):
        return {'input_ids': [0]}  # dummy
    
    def collate_fn(self, batch):
        return {'input_ids': ['dummy text'] * len(batch)}


def create_dataloaders(config, accelerator, tokenizer, create_imagetext_dataloader=None):
    """
    Создает все необходимые датасеты и датасеты лоадеры для обучения.
    
    Args:
        config: конфигурация эксперимента
        accelerator: объект Accelerator
        tokenizer: токенизатор
        create_imagetext_dataloader: функция для создания parquet dataloaders (опционально)
    
    Returns:
        tuple: (combined_dataloader, num_update_steps_per_epoch, num_train_epochs)
    """
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
        config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # ============================================================
    # Data for generation (T2I)
    # ============================================================
    if config.dataset.gen_type == "t2i":
        dataset = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_t2i,
            per_gpu_batch_size=config.training.batch_size_t2i,
            global_batch_size=total_batch_size_t2i_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
        )
        train_dataloader_t2i = dataset.train_dataloader
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    elif config.dataset.gen_type == "t2i_parquet":
        if create_imagetext_dataloader is None:
            raise ValueError("create_imagetext_dataloader function is required for t2i_parquet mode")
        
        num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        train_dataloader_t2i = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            batch_size=config.training.batch_size_t2i,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size
        )

    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )

        logger.info(f'Process index: {accelerator.process_index}, num_processes: {accelerator.num_processes}, '
                   f'Dataset length: {len(dataset_imagenet)}')

        if accelerator.num_processes > 1:
            sampler = DistributedSampler(
                dataset_imagenet,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dataloader_t2i = DataLoader(
            dataset_imagenet, 
            batch_size=config.training.batch_size_t2i,
            sampler=sampler, 
            collate_fn=dataset_imagenet.collate_fn,
            shuffle=shuffle, 
            num_workers=dataset_config.num_workers
        )
        num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset gen_type: {config.dataset.gen_type}")

    # ============================================================
    # Data for image captioning / multimodal understanding (MMU)
    # ============================================================
    total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes
    
    if config.dataset.und_type == "captioning":
        dataset_mmu = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_mmu,
            per_gpu_batch_size=config.training.batch_size_mmu,
            global_batch_size=total_batch_size_mmu_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            is_captioning=True,
            add_caption_prompt=dataset_config.add_caption_prompt,
        )
        train_dataloader_mmu = dataset_mmu.train_dataloader

    elif config.dataset.und_type == "captioning_parquet":
        if create_imagetext_dataloader is None:
            raise ValueError("create_imagetext_dataloader function is required for captioning_parquet mode")
        
        train_dataloader_mmu = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            batch_size=config.training.batch_size_mmu,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            is_captioning=True
        )

    elif config.dataset.und_type == "llava_pretrain":
        train_dataloader_mmu = get_instruct_data_loader(
            tokenizer,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length if config.dataset.add_system_prompt else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
            phase="pretrain"
        )

    elif config.dataset.und_type == "llava_tuning":
        train_dataloader_mmu = get_instruct_data_loader(
            tokenizer,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length if config.dataset.add_system_prompt else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
            phase="tuning"
        )

    else:
        raise NotImplementedError(f"Unsupported dataset und_type: {config.dataset.und_type}")

    # ============================================================
    # Dummy LM dataloader
    # ============================================================
    train_dataloader_lm = torch.utils.data.DataLoader(
        DummyLMDataset(), 
        batch_size=config.training.batch_size_lm,
        sampler=None, 
        collate_fn=DummyLMDataset().collate_fn
    )

    # ============================================================
    # Combined dataloader
    # ============================================================
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    logger.info(f"Dataloaders created. Num update steps per epoch: {num_update_steps_per_epoch}, "
               f"Num train epochs: {num_train_epochs}")

    return combined_dataloader, num_update_steps_per_epoch, num_train_epochs

