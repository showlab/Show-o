import math
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.utilities import CombinedLoader
from accelerate.logging import get_logger

from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from models.llava.llava_data_vq_unified import get_instruct_data_loader
from models.llava.domain_datasets import get_domain_data_loader

logger = get_logger(__name__, log_level="INFO")

SYSTEM_PROMPT_LEN = 28


class DummyLMDataset:
    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        return {"input_ids": [0]}  # dummy

    def collate_fn(self, batch):
        return {"input_ids": ["dummy text"] * len(batch)}


def create_dataloaders(
    config, accelerator, tokenizer, create_imagetext_dataloader=None
):
    total_batch_size_t2i_without_accum = (
        config.training.batch_size_t2i * accelerator.num_processes
    )
    total_batch_size_t2i = (
        config.training.batch_size_t2i
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
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
            train_dataloader_t2i.num_batches
            / config.training.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(
            config.training.max_train_steps / num_update_steps_per_epoch
        )

    elif config.dataset.gen_type == "t2i_parquet":
        if create_imagetext_dataloader is None:
            raise ValueError(
                "create_imagetext_dataloader function is required for t2i_parquet mode"
            )

        num_update_steps_per_epoch = math.ceil(
            config.experiment.max_train_examples_t2i / total_batch_size_t2i
        )
        num_train_epochs = math.ceil(
            config.training.max_train_steps / num_update_steps_per_epoch
        )

        train_dataloader_t2i = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            batch_size=config.training.batch_size_t2i,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        )

    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )

        logger.info(
            f"Process index: {accelerator.process_index}, num_processes: {accelerator.num_processes}, "
            f"Dataset length: {len(dataset_imagenet)}"
        )

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
            num_workers=dataset_config.num_workers,
        )
        num_update_steps_per_epoch = math.ceil(
            len(dataset_imagenet) / total_batch_size_t2i
        )
        num_train_epochs = math.ceil(
            config.training.max_train_steps / num_update_steps_per_epoch
        )

    else:
        raise ValueError(f"Unsupported dataset gen_type: {config.dataset.gen_type}")

    # ============================================================
    # Data for image captioning / multimodal understanding (MMU)
    # ============================================================
    total_batch_size_mmu_without_accum = (
        config.training.batch_size_mmu * accelerator.num_processes
    )

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
            raise ValueError(
                "create_imagetext_dataloader function is required for captioning_parquet mode"
            )

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
            is_captioning=True,
        )

    elif config.dataset.und_type == "llava_pretrain":
        train_dataloader_mmu = get_instruct_data_loader(
            tokenizer,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length
            if config.dataset.add_system_prompt
            else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
            phase="pretrain",
        )

    elif config.dataset.und_type == "llava_tuning":
        train_dataloader_mmu = get_instruct_data_loader(
            tokenizer,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length
            if config.dataset.add_system_prompt
            else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
            phase="tuning",
        )

    else:
        raise NotImplementedError(
            f"Unsupported dataset und_type: {config.dataset.und_type}"
        )

    # ============================================================
    # Domain-specific datasets (VQAv2, TextVQA, DocVQA, Kvasir-VQA, TextVQA Experiments, VQAv2 Experiments)
    # ============================================================
    train_dataloader_vqav2 = None
    train_dataloader_textvqa = None
    train_dataloader_textvqa_experiments = None
    train_dataloader_vqav2_experiments = None
    train_dataloader_docvqa = None
    train_dataloader_kvasir = None
    
    # VQAv2
    if hasattr(dataset_config, 'vqav2_data_file_path') and dataset_config.vqav2_data_file_path:
        if os.path.exists(dataset_config.vqav2_data_file_path):
            train_dataloader_vqav2 = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_vqav2 if hasattr(config.training, 'batch_size_vqav2') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="vqav2",
                data_file_path=dataset_config.vqav2_data_file_path,
                image_root=dataset_config.vqav2_image_root,
            )
        else:
            logger.warning(f"VQAv2 dataset file not found: {dataset_config.vqav2_data_file_path}. Skipping VQAv2 dataloader.")
    
    # TextVQA
    if hasattr(dataset_config, 'textvqa_data_file_path') and dataset_config.textvqa_data_file_path:
        if os.path.exists(dataset_config.textvqa_data_file_path):
            train_dataloader_textvqa = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_textvqa if hasattr(config.training, 'batch_size_textvqa') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="textvqa",
                data_file_path=dataset_config.textvqa_data_file_path,
                image_root=dataset_config.textvqa_image_root,
            )
        else:
            logger.warning(f"TextVQA dataset file not found: {dataset_config.textvqa_data_file_path}. Skipping TextVQA dataloader.")
    
    # TextVQA Experiments (датасет для экспериментов MoE - 1000 семплов)
    if hasattr(dataset_config, 'textvqa_experiments_data_file_path') and dataset_config.textvqa_experiments_data_file_path:
        if os.path.exists(dataset_config.textvqa_experiments_data_file_path):
            train_dataloader_textvqa_experiments = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_textvqa_experiments if hasattr(config.training, 'batch_size_textvqa_experiments') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="textvqa",
                data_file_path=dataset_config.textvqa_experiments_data_file_path,
                image_root=dataset_config.textvqa_experiments_image_root,
            )
        else:
            logger.warning(f"TextVQA Experiments dataset file not found: {dataset_config.textvqa_experiments_data_file_path}. Skipping TextVQA Experiments dataloader.")
    
    # VQAv2 Experiments (датасет для экспериментов MoE - 1000 семплов)
    if hasattr(dataset_config, 'vqav2_experiments_data_file_path') and dataset_config.vqav2_experiments_data_file_path:
        if os.path.exists(dataset_config.vqav2_experiments_data_file_path):
            train_dataloader_vqav2_experiments = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_vqav2 if hasattr(config.training, 'batch_size_vqav2') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="vqav2",
                data_file_path=dataset_config.vqav2_experiments_data_file_path,
                image_root=dataset_config.vqav2_experiments_image_root,
            )
        else:
            logger.warning(f"VQAv2 Experiments dataset file not found: {dataset_config.vqav2_experiments_data_file_path}. Skipping VQAv2 Experiments dataloader.")
    
    # DocVQA
    if hasattr(dataset_config, 'docvqa_data_file_path') and dataset_config.docvqa_data_file_path:
        if os.path.exists(dataset_config.docvqa_data_file_path):
            train_dataloader_docvqa = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_docvqa if hasattr(config.training, 'batch_size_docvqa') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="docvqa",
                data_file_path=dataset_config.docvqa_data_file_path,
                image_root=dataset_config.docvqa_image_root,
            )
        else:
            logger.warning(f"DocVQA dataset file not found: {dataset_config.docvqa_data_file_path}. Skipping DocVQA dataloader.")
    
    # Kvasir-VQA
    if hasattr(dataset_config, 'kvasir_data_file_path') and dataset_config.kvasir_data_file_path:
        if os.path.exists(dataset_config.kvasir_data_file_path):
            train_dataloader_kvasir = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_kvasir if hasattr(config.training, 'batch_size_kvasir') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="kvasir",
                data_file_path=dataset_config.kvasir_data_file_path,
                image_root=dataset_config.kvasir_image_root,
            )
        else:
            logger.warning(f"Kvasir-VQA dataset file not found: {dataset_config.kvasir_data_file_path}. Skipping Kvasir-VQA dataloader.")
    
    # CLEVR
    train_dataloader_clevr = None
    if hasattr(dataset_config, 'clevr_data_file_path') and dataset_config.clevr_data_file_path:
        if os.path.exists(dataset_config.clevr_data_file_path):
            train_dataloader_clevr = get_domain_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_clevr if hasattr(config.training, 'batch_size_clevr') else config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length
                if config.dataset.add_system_prompt
                else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                dataset_type="clevr",
                data_file_path=dataset_config.clevr_data_file_path,
                image_root=dataset_config.clevr_image_root,
            )
        else:
            logger.warning(f"CLEVR dataset file not found: {dataset_config.clevr_data_file_path}. Skipping CLEVR dataloader.")

    # ============================================================
    # Dummy LM dataloader
    # ============================================================
    train_dataloader_lm = torch.utils.data.DataLoader(
        DummyLMDataset(),
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=DummyLMDataset().collate_fn,
    )

    # ============================================================
    # Combined dataloader
    # ============================================================
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }
    
    # Добавляем доменные датасеты в iterables, если они доступны и включены через флаги
    # Каждый датасет управляется отдельным флагом в конфиге
    use_vqav2 = config.dataset.get("use_vqav2", False)
    use_textvqa = config.dataset.get("use_textvqa", False)
    use_textvqa_experiments = config.dataset.get("use_textvqa_experiments", False)
    use_vqav2_experiments = config.dataset.get("use_vqav2_experiments", False)
    use_clevr = config.dataset.get("use_clevr", False)
    use_docvqa = config.dataset.get("use_docvqa", False)
    use_kvasir = config.dataset.get("use_kvasir", False)
    
    if use_vqav2 and train_dataloader_vqav2 is not None:
        iterables["vqav2_flow"] = train_dataloader_vqav2
        logger.info("VQAv2 dataloader added to combined loader")
    
    if use_textvqa and train_dataloader_textvqa is not None:
        iterables["textvqa_flow"] = train_dataloader_textvqa
        logger.info("TextVQA dataloader added to combined loader")
    
    if use_textvqa_experiments and train_dataloader_textvqa_experiments is not None:
        iterables["textvqa_experiments_flow"] = train_dataloader_textvqa_experiments
        logger.info("TextVQA Experiments dataloader added to combined loader")
    
    if use_vqav2_experiments and train_dataloader_vqav2_experiments is not None:
        iterables["vqav2_experiments_flow"] = train_dataloader_vqav2_experiments
        logger.info("VQAv2 Experiments dataloader added to combined loader")
    
    if use_clevr and train_dataloader_clevr is not None:
        iterables["clevr_flow"] = train_dataloader_clevr
        logger.info("CLEVR dataloader added to combined loader")
    
    if use_docvqa and train_dataloader_docvqa is not None:
        iterables["docvqa_flow"] = train_dataloader_docvqa
        logger.info("DocVQA dataloader added to combined loader")
    
    if use_kvasir and train_dataloader_kvasir is not None:
        iterables["kvasir_flow"] = train_dataloader_kvasir
        logger.info("Kvasir-VQA dataloader added to combined loader")

    combined_dataloader = CombinedLoader(
        iterables, mode=config.dataset.combined_loader_mode
    )

    logger.info(
        f"Dataloaders created. Num update steps per epoch: {num_update_steps_per_epoch}, "
        f"Num train epochs: {num_train_epochs}"
    )

    return combined_dataloader, num_update_steps_per_epoch, num_train_epochs
