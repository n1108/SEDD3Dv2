import os
import torch
# from model import SEDD, SEDDCond, SEDDCondBlock
from model import SEDDCond
# from model.cnn import DenoiseCNN
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = SEDDCond.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device, ckpt_dir=None):
    cfg = utils.load_hydra_config_from_run(root_dir) # cfg is the multi-res config from the run

    graph = graph_lib.get_graph(cfg, device) # cfg.tokens should be global
    noise = noise_lib.get_noise(cfg).to(device) # noise params should be global
    
    # Instantiate the model, assuming SEDDCond for multi-resolution trained models
    score_model = SEDDCond(cfg).to(device) # Model uses the loaded cfg for its init
    
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = ckpt_dir if ckpt_dir else os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    print(ckpt_dir)
    loaded_state = torch.load(ckpt_dir, map_location=device)

    score_model.load_state_dict(loaded_state['model'], strict=False)
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise, cfg # Return the original loaded cfg


def load_model(root_dir, device, ckpt_dir=None):
    try:
        return load_model_local(root_dir, device, ckpt_dir)
    except:
        return load_model_hf(root_dir, device)
