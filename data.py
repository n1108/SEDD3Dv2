import re
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader, DistributedSampler
from dataset.carla_dataset import CarlaDataset
from dataset.kitti360_dataset import Kitti360Dataset


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def get_dataloaders(config, current_stage_config, distributed=True, prev_scene_path=None, eval_batch_size=None):
    stage_batch_size = current_stage_config.batch_size # 这是该阶段的总批大小 (跨GPU，未累积)
    stage_eval_batch_size = eval_batch_size if eval_batch_size else \
                            current_stage_config.get('eval_batch_size', stage_batch_size) # 若无eval_batch_size则用训练的

    if (stage_batch_size * config.ngpus) % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Stage '{current_stage_config.name}' Train Batch Size {stage_batch_size * config.ngpus} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if (stage_eval_batch_size * config.ngpus) % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Stage '{current_stage_config.name}' Eval Batch Size {stage_eval_batch_size * config.ngpus} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    dataset_name = current_stage_config.get('dataset_type', config.data.dataset) # 允许阶段覆盖数据集类型
    DatasetClass = None
    if dataset_name == 'carla':
        DatasetClass = CarlaDataset
    elif dataset_name == 'kitti360':
        DatasetClass = Kitti360Dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    # 使用 current_stage_config 中的参数初始化 Dataset
    train_set = DatasetClass(
        directory=current_stage_config.train_data_path, 
        quantized_directory=current_stage_config.quantized_train_data_path,
        data_argumentation=config.data.data_argumentation, # 全局数据增强开关
        mode='train',
        prev_stage=current_stage_config.prev_stage,
        next_stage=current_stage_config.next_stage,
        prev_data_size=current_stage_config.prev_data_size,
        next_data_size=current_stage_config.image_size, # image_size 是目标尺寸
        # prev_scene_path 和 infer_data_source 可以来自全局 config.data
        prev_scene_path=config.data.prev_scene_path, 
        infer_data_source=config.data.infer_data_source,
        args=config.data, # 传递全局 config.data 给 dataset (如果它需要其他参数)
    )
    valid_set = DatasetClass(
        directory=current_stage_config.valid_data_path, 
        quantized_directory=current_stage_config.quantized_valid_data_path,
        data_argumentation=False, # 验证集通常不使用数据增强
        mode='inference',
        prev_stage=current_stage_config.prev_stage,
        next_stage=current_stage_config.next_stage,
        prev_data_size=current_stage_config.prev_data_size,
        next_data_size=current_stage_config.image_size,
        # prev_scene_path 可以被特定调用的 prev_scene_path 参数覆盖 (主要用于采样脚本)
        prev_scene_path=prev_scene_path if prev_scene_path else config.data.prev_scene_path,
        infer_data_source='generation' if prev_scene_path else config.data.infer_data_source,
        args=config.data,
    )

        # TODO:
    # train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    # valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=current_stage_config.batch_size // config.training.accum,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=train_set.collate_fn,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=stage_eval_batch_size // config.training.accum, # (通常 eval 时 accum=1)
        sampler=test_sampler,
        num_workers=4,
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
        shuffle=False,
    ))
    return train_loader, valid_loader

