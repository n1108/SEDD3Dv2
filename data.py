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


def get_dataloaders(config, current_stage_config, distributed=True, prev_scene_path=None, eval_batch_size_override=None):
    # current_stage_config.batch_size 是该阶段在单个GPU上的微批次大小
    # config.training.accum 是全局累积数
    
    # stage_batch_size_per_gpu 是此阶段在一个GPU上进行一次前向传播的样本数
    stage_batch_size_per_gpu = current_stage_config.batch_size 
    
    # stage_eval_batch_size_per_gpu 是评估时此阶段在一个GPU上的样本数
    # 如果提供了 eval_batch_size_override (例如从 run_sample.py 调用时)，则使用它
    # 否则，尝试从 current_stage_config 获取 eval_batch_size，如果不存在，则使用训练时的 stage_batch_size_per_gpu
    stage_eval_batch_size_per_gpu = eval_batch_size_override if eval_batch_size_override is not None else \
                                   current_stage_config.get('eval_batch_size', stage_batch_size_per_gpu)

    # 这里的检查需要基于总的有效批处理大小来考虑
    # 总批大小 (跨所有GPU, 未累积) = stage_batch_size_per_gpu * config.ngpus
    # 有效批大小 (累积后) = stage_batch_size_per_gpu * config.ngpus * config.training.accum (这是不对的, accum 是微批次的数量)
    # 实际上，我们关心的是每个GPU的负载是否均匀。
    # DataLoader 将为每个GPU提供 stage_batch_size_per_gpu 的数据。
    # 梯度累积的逻辑在 step_fn 中处理。

    dataset_name = current_stage_config.get('dataset_type', config.data.dataset) 
    DatasetClass = None
    if dataset_name == 'carla':
        DatasetClass = CarlaDataset
    elif dataset_name == 'kitti360':
        DatasetClass = Kitti360Dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    train_set = DatasetClass(
        directory=current_stage_config.train_data_path, 
        quantized_directory=current_stage_config.quantized_train_data_path,
        data_argumentation=config.data.data_argumentation, 
        mode='train',
        prev_stage=current_stage_config.prev_stage,
        next_stage=current_stage_config.next_stage,
        prev_data_size=current_stage_config.prev_data_size,
        next_data_size=current_stage_config.image_size, # image_size 是目标尺寸
        prev_scene_path=config.data.get('prev_scene_path', None), # 全局的 prev_scene_path
        infer_data_source=config.data.infer_data_source,
        args=config.data, 
    )
    valid_set = DatasetClass(
        directory=current_stage_config.valid_data_path, 
        quantized_directory=current_stage_config.quantized_valid_data_path,
        data_argumentation=False, 
        mode='inference', # 通常验证集使用 'inference' 或 'eval' 模式
        prev_stage=current_stage_config.prev_stage,
        next_stage=current_stage_config.next_stage,
        prev_data_size=current_stage_config.prev_data_size,
        next_data_size=current_stage_config.image_size,
        # prev_scene_path 可以被特定调用的 prev_scene_path 参数覆盖 (主要用于采样脚本 run_sample.py)
        prev_scene_path=prev_scene_path if prev_scene_path else config.data.get('prev_scene_path', None),
        infer_data_source='generation' if prev_scene_path else config.data.infer_data_source,
        args=config.data,
    )

    if distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True) # 训练时通常需要 shuffle
        test_sampler = DistributedSampler(valid_set, shuffle=False) # 验证/测试时不需要 shuffle
    else:
        train_sampler = None
        test_sampler = None
    
    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=stage_batch_size_per_gpu, # 每个GPU的微批次大小
        sampler=train_sampler,
        num_workers=4, # 可根据系统调整
        collate_fn=train_set.collate_fn,
        pin_memory=True,
        shuffle=(train_sampler is None), # 如果没有 sampler (单GPU)，则由 DataLoader shuffle
        persistent_workers=True,
    ), sampler=train_sampler) # 将 sampler 传递给 cycle_loader 用于 set_epoch

    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=stage_eval_batch_size_per_gpu, # 评估时每个GPU的批大小
        sampler=test_sampler,
        num_workers=4, # 可根据系统调整
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
        shuffle=False, # 验证集不 shuffle
    ), sampler=test_sampler) # 将 sampler 传递给 cycle_loader 用于 set_epoch
    
    return train_loader, valid_loader

