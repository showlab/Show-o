import numpy as np
import torch

from llava.llava import conversation as conversation_lib
from llava.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX)
from llava.llava.mm_utils import tokenizer_image_token

from llava.data.vqa_dataset import VQADataset

import tokenizers
from packaging import version
import transformers
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def pad_or_truncate(sequence, max_length, pad_value):
    if sequence.shape[1] < max_length:
        # 填充
        padding = torch.full((max_length - sequence.shape[1],), pad_value, dtype=sequence.dtype)[None,:].repeat(sequence.shape[0],1)
        sequence = torch.cat([sequence, padding],dim=1)
    else:
        # 截断
        sequence = sequence[:,:max_length]
    return sequence

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, max_length = 600
):
    image_path_list = []
    images_vae_list = []
    images_clip_list = []
    conversation_list = []
    questions_list = []
    class_condition_list = []
    training_task_list = []
    inferences = []
    for (
        image_path,
        images_vae,
        images_clip,
        conversations,
        questions,
        class_condition,
        training_task,
        inference
    ) in batch[0]:
        image_path_list.append(image_path)
        images_vae_list.append(images_vae)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        questions_list.append(questions)
        class_condition_list.append(class_condition)
        training_task_list.append(training_task)
        inferences.append(inference)

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()
    has_image = True

    conv = conversation_lib.default_conversation.copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    sep = conv.sep + conv.roles[1] + ": "
    # print(conv_type)

    for conversation, target in zip(conversation_list, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(
        #     conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        cur_len = 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # +1 for <|endoftext|>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len)


    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
    
    # index = torch.nonzero(input_ids == -200, as_tuple=False)[0]
    # print(input_ids)
    # print(index)
    # index = index.item()
    # input_ids = torch.cat((input_ids[:index], input_ids[index + 1:]))
    # targets = torch.cat((targets[:index], targets[index + 1:]))
    # new_element = torch.tensor([tokenizer.bos_token_id])
    # input_ids = torch.cat((new_element, input_ids))
    # new_element = torch.tensor([IGNORE_INDEX])
    # targets = torch.cat((IGNORE_INDEX, targets))
    # index = torch.nonzero(input_dict['input_ids'] == -200, as_tuple=False)[0][-1]
    # index = index.item()
    # input_dict['input_ids'] = torch.cat((input_dict['input_ids'][:,:index], input_dict['input_ids'][:,index + 1:]),dim=1)
    # input_dict['labels'] = torch.cat((input_dict['labels'][:,:index], input_dict['labels'][:,index + 1:]),dim=1)
    # new_element = torch.tensor([tokenizer.bos_token_id]).repeat(input_dict['labels'].shape[0])
    # input_dict['input_ids'] = torch.cat((new_element[:,None], input_dict['input_ids']),dim=1)
    # new_element = torch.tensor([-100]).repeat(input_dict['labels'].shape[0])
    # input_dict['labels'] = torch.cat((new_element[:,None], input_dict['labels']),dim=1)



    input_dict = {
        "image_paths": image_path_list,
        "images_vae": torch.stack(images_vae_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "questions_list": questions_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "training_task_list": training_task_list,
        "class_conditions": torch.cat(class_condition_list, dim=0),
    }
    index = torch.nonzero(input_dict['input_ids'] == -200, as_tuple=False)[0][-1]
    index = index.item()
    input_dict['input_ids'] = torch.cat((input_dict['input_ids'][:,:index], input_dict['input_ids'][:,index + 1:]),dim=1)
    input_dict['labels'] = torch.cat((input_dict['labels'][:,:index], input_dict['labels'][:,index + 1:]),dim=1)
    new_element = torch.tensor([tokenizer.bos_token_id]).repeat(input_dict['labels'].shape[0])
    input_dict['input_ids'] = torch.cat((new_element[:,None], input_dict['input_ids']),dim=1)
    new_element = torch.tensor([-100]).repeat(input_dict['labels'].shape[0])
    input_dict['labels'] = torch.cat((new_element[:,None], input_dict['labels']),dim=1)
    input_dict['labels'] = pad_or_truncate(input_dict['labels'],max_length,-100)
    input_dict['input_ids'] = pad_or_truncate(input_dict['input_ids'],max_length,tokenizer.pad_token_id)

    # indexes = torch.nonzero(input_dict['labels'] == 50256, as_tuple=False)[:,-1]
    # for i in range(indexes.size(0)):
    #     if len(input_dict['input_ids'][i])>indexes[i]+1:
    #         input_dict['input_ids'][i, indexes[i]+1:] = 50295

    return input_dict


# class HybridDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         vision_tower,
#         samples_per_epoch=500 * 8 * 2 * 10,
#         precision: str = "fp32",
#         image_size: int = 224,
#         dataset="vqa,imagenet",
#         sample_rate=[1, 1],
#         batch_size=4,
#     ):
#         self.dataset = dataset
#         self.samples_per_epoch = samples_per_epoch
#         self.image_size = image_size
#         self.precision = precision
#         self.batch_size = batch_size
#
#         self.datasets = dataset.split(",")
#         assert len(self.datasets) == len(sample_rate), (len(self.datasets), len(sample_rate), sample_rate)
#         sample_rate_expand = []
#
#         self.all_datasets = []
#         for idx, dataset in enumerate(self.datasets):
#             if dataset == "vqa":
#                 self.all_datasets.append(
#                     VQADataset(
#                         vision_tower,
#                         samples_per_epoch,
#                         batch_size=batch_size//8,
#                     )
#                 )
#                 sample_rate_expand.append(sample_rate[idx])
#             elif dataset == "ImageNet":
#                 self.all_datasets.append(
#                     ImageNetDataset(
#                         "/mnt/bn/vgfm2/test_dit/imagenet/ILSVRC/Data/CLS-LOC/train",
#                         image_size=image_size,
#                         samples_per_epoch=samples_per_epoch,
#                         batch_size=batch_size,
#                     )
#                 )
#                 sample_rate_expand.append(sample_rate[idx])
#
#         import ipdb
#         ipdb.set_trace()
#         assert len(self.all_datasets) == len(sample_rate_expand)
#         sample_rate = np.array(sample_rate_expand)
#         for idx in range(len(sample_rate)):
#             print("Dataset: {}, sample rate: {}".format(self.all_datasets[idx], sample_rate[idx]))
#         self.sample_rate = sample_rate / sample_rate.sum()
#
#     def __len__(self):
#         return self.samples_per_epoch
#
#     def __getitem__(self, idx):
#         ind = np.random.choice(list(range(len(self.all_datasets))), p=self.sample_rate)
#         data = self.all_datasets[ind]
#         inference = False
#         batch_data = []
#         batch_size = self.batch_size
#         # import pdb;pdb.set_trace()
#         if len(self.all_datasets)>1 and ind==0:
#             batch_size//=8
#         for _ in range(batch_size):
#             batch_data.append([*data[0], inference])
#
#         return batch_data
class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        dataset="vqa,imagenet",
        sample_rate=[1, 1],
        batch_size=4,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.precision = precision
        self.batch_size = batch_size

        self.datasets = dataset.split(",")
        assert len(self.datasets) == len(sample_rate), (len(self.datasets), len(sample_rate), sample_rate)
        sample_rate_expand = []

        self.all_datasets = []
        for idx, dataset in enumerate(self.datasets):
            if dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        vision_tower,
                        samples_per_epoch,
                        batch_size=batch_size//8,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "ImageNet":
                self.all_datasets.append(
                    ImageNetDataset(
                        "/mnt/bn/vgfm2/test_dit/imagenet/ILSVRC/Data/CLS-LOC/train",
                        image_size=image_size,
                        samples_per_epoch=samples_per_epoch,
                        batch_size=batch_size,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])

        # import ipdb
        # ipdb.set_trace()
        assert len(self.all_datasets) == len(sample_rate_expand)
        sample_rate = np.array(sample_rate_expand)
        for idx in range(len(sample_rate)):
            print("Dataset: {}, sample rate: {}".format(self.all_datasets[idx], sample_rate[idx]))
        self.sample_rate = sample_rate / sample_rate.sum()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.all_datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        batch_data = []
        batch_size = self.batch_size
        # import pdb;pdb.set_trace()
        if len(self.all_datasets)>1 and ind==0:
            batch_size//=8
        for _ in range(batch_size):
            batch_data.append([*data[0], inference])

        return batch_data

