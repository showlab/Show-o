import copy
import json
import os
from functools import partial

import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from training.utils import image_transform
from models.llava.llava import conversation as conversation_lib
from models.llava.llava_data_vq_unified import preprocess_multimodal, preprocess_v0, collate_fn, IGNORE_INDEX

DEFAULT_IMAGE_TOKEN = "<image>"
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."


class DocVQADataset(Dataset):
    def __init__(self, tokenizer, data_file_path, image_root):
        super(DocVQADataset, self).__init__()
        self.tokenizer = tokenizer
        self.image_root = image_root
        
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            if 'image' in item.keys():
                # Преобразуем формат DocVQA в формат LLaVA
                # Убеждаемся, что value всегда строка
                question_str = str(item.get('question', ''))
                if isinstance(item.get('answers'), list) and len(item.get('answers', [])) > 0:
                    answer_str = str(item.get('answers')[0])
                else:
                    answer_str = str(item.get('answer', ''))
                
                docvqa_item = {
                    'image': item['image'],
                    'conversations': [
                        {
                            'from': 'human',
                            'value': question_str
                        },
                        {
                            'from': 'gpt',
                            'value': answer_str
                        }
                    ]
                }
                self.list_data_dict.append(docvqa_item)
        
        print(f"DocVQA dataset loaded: {len(self.list_data_dict)} samples")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_root
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = image_transform(image)
        except:
            print("Read image error. Use dummy data.")
            crop_size = 256
            image = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        data_dict = preprocess_v0(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


class KvasirVQADataset(Dataset):
    """Dataset для Kvasir-VQA в формате, совместимом с LLaVA"""
    
    def __init__(self, tokenizer, data_file_path, image_root):
        super(KvasirVQADataset, self).__init__()
        self.tokenizer = tokenizer
        self.image_root = image_root
        
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            if 'image' in item.keys():
                # Преобразуем формат Kvasir-VQA в формат LLaVA
                kvasir_item = {
                    'image': item['image'],
                    'conversations': [
                        {
                            'from': 'human',
                            'value': item.get('question', '')
                        },
                        {
                            'from': 'gpt',
                            'value': item.get('answers', [item.get('answer', '')])[0] if isinstance(item.get('answers'), list) else item.get('answer', '')
                        }
                    ]
                }
                self.list_data_dict.append(kvasir_item)
        
        print(f"Kvasir-VQA dataset loaded: {len(self.list_data_dict)} samples")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_root
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = image_transform(image)
        except:
            print("Read image error. Use dummy data.")
            crop_size = 256
            image = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        data_dict = preprocess_v0(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


def get_domain_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        dataset_type,  # "vqav2", "textvqa", "docvqa", or "kvasir"
        data_file_path,
        image_root,
):
    """Создает dataloader для доменного датасета"""
    # VQAv2, TextVQA и CLEVR используют тот же формат, что и DocVQA
    if dataset_type in ["vqav2", "textvqa", "docvqa", "clevr"]:
        train_dataset = DocVQADataset(tokenizer, data_file_path, image_root)
    elif dataset_type == "kvasir":
        train_dataset = KvasirVQADataset(tokenizer, data_file_path, image_root)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Supported: vqav2, textvqa, docvqa, kvasir, clevr")
    
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader

