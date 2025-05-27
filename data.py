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


def get_dataloaders(config, distributed=True, prev_scene_path=None, eval_batch_size=None):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    if config.data.dataset == 'carla':
        train_set = CarlaDataset(directory=config.data.train_data_path, 
                            quantized_directory=config.data.quantized_train_data_path,
                            data_argumentation=config.data.data_argumentation,
                            mode='train',
                            prev_stage=config.data.prev_stage,
                            next_stage=config.data.next_stage,
                            prev_data_size=config.data.prev_data_size,
                            next_data_size=config.data.next_data_size,
                            prev_scene_path=config.data.prev_scene_path,
                            infer_data_source=config.data.infer_data_source,
                            args=config.data,
                            )
        valid_set = CarlaDataset(directory=config.data.valid_data_path, 
                            quantized_directory=config.data.quantized_valid_data_path,
                            data_argumentation=False,
                            mode='inference',
                            prev_stage=config.data.prev_stage,
                            next_stage=config.data.next_stage,
                            prev_data_size=config.data.prev_data_size,
                            next_data_size=config.data.next_data_size,
                            prev_scene_path=prev_scene_path if prev_scene_path else config.data.prev_scene_path,
                            infer_data_source='generation' if prev_scene_path else config.data.infer_data_source,
                            args=config.data,
                            )
    elif config.data.dataset == 'kitti360':
        train_set = Kitti360Dataset(directory=config.data.train_data_path, 
                            quantized_directory=config.data.quantized_train_data_path,
                            data_argumentation=config.data.data_argumentation,
                            mode='train',
                            prev_stage=config.data.prev_stage,
                            next_stage=config.data.next_stage,
                            prev_data_size=config.data.prev_data_size,
                            next_data_size=config.data.next_data_size,
                            prev_scene_path=config.data.prev_scene_path,
                            infer_data_source=config.data.infer_data_source,
                            args=config.data,
                            )
        valid_set = Kitti360Dataset(directory=config.data.valid_data_path, 
                            quantized_directory=config.data.quantized_valid_data_path,
                            data_argumentation=False,
                            mode='inference',
                            prev_stage=config.data.prev_stage,
                            next_stage=config.data.next_stage,
                            prev_data_size=config.data.prev_data_size,
                            next_data_size=config.data.next_data_size,
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
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=train_set.collate_fn,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=eval_batch_size if eval_batch_size else config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
        shuffle=False,
    ))
    return train_loader, valid_loader

