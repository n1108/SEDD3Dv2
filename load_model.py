import os
import torch
from model import SEDD, SEDDCond, SEDDCondBlock
from model.cnn import DenoiseCNN
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device, ckpt_dir=None):
    cfg = utils.load_hydra_config_from_run(root_dir)
    if hasattr(cfg.data, 'crop_size'):
        del cfg.data.crop_size
        cfg.image_size = cfg.data.next_data_size
    if cfg.data.next_data_size[0] == 512:
        cfg.image_size = [cfg.image_size[0]//2, cfg.image_size[1]//2, cfg.image_size[2]]
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    # build score model
    if cfg.model.type == 'cnn':
        if cfg.data.prev_stage != 'none':
            score_model = DenoiseCNN(cfg, cond=True).to(device)
        else:
            score_model = DenoiseCNN(cfg, cond=False).to(device)
    elif cfg.data.prev_stage != 'none' and hasattr(cfg, 'block_dit') and cfg.block_dit == True:
        score_model = SEDDCondBlock(cfg).to(device)
    elif cfg.data.prev_stage != 'none':
        score_model = SEDDCond(cfg).to(device)
    else:
        score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = ckpt_dir if ckpt_dir else os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    print(ckpt_dir)
    loaded_state = torch.load(ckpt_dir, map_location=device)

    score_model.load_state_dict(loaded_state['model'], strict=False)
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise, cfg


def load_model(root_dir, device, ckpt_dir=None):
    try:
        return load_model_local(root_dir, device, ckpt_dir)
    except:
        return load_model_hf(root_dir, device)
