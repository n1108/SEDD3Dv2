"""Training and evaluation"""

from tqdm import tqdm
import hydra
import os
import numpy as np
from run_train import cleanup, setup
import utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

import os

import numpy as np
import torch
import argparse

from load_model import load_model
# from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
import data
import utils


def _run(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model, graph, noise, cfg = load_model(args.model_path, device, args.ckpt_dir)
    if args.prev_data_sr:
        cfg.data.prev_data_sr = args.prev_data_sr
    sample_dir = os.path.join(args.model_path, "samples")
    this_sample_dir = args.this_sample_dir if args.this_sample_dir else os.path.join(sample_dir, "test")
    utils.makedirs(os.path.join(this_sample_dir, 'generated'))
    if cfg.data.prev_stage != 'none':
        utils.makedirs(os.path.join(this_sample_dir, 'cond'))
        utils.makedirs(os.path.join(this_sample_dir, 'gt'))
        _, eval_ds = data.get_dataloaders(cfg, distributed=False, prev_scene_path=args.prev_scene_path, eval_batch_size=args.eval_batch_size)
        eval_iter = iter(eval_ds)
    # else:
    #     _, eval_ds = data.get_dataloaders(cfg, distributed=False, eval_batch_size=args.eval_batch_size)

    batch_size = args.eval_batch_size if args.eval_batch_size else cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum)
    sampling_shape = (batch_size, cfg.image_size[0] * cfg.image_size[1] * cfg.image_size[2])
    sampling_eps = 1e-5
    samples_per_gpu = args.samples // world_size
    cur_id = 0 
    if cfg.data.prev_stage != 'none':
        while cur_id < samples_per_gpu // sampling_shape[0] * rank:
            eval_cond, eval_batch = next(eval_iter)
            cur_id += sampling_shape[0]
    for k in tqdm(range(0, args.samples//world_size, sampling_shape[0]), disable=rank!=0):
        if cfg.data.prev_stage != 'none':
            eval_cond, eval_batch = next(eval_iter)
            cond = eval_cond.to(device)
            cond = cond[:sampling_shape[0]]
        else:
            cond=None
        generated = True
        for j in range(sampling_shape[0]):
            if not os.path.exists(os.path.join(this_sample_dir, 'generated', f"sample_{cur_id+j}.txt")):
                generated=False
                break
        if generated:
            cur_id += sampling_shape[0]
            continue
        if cfg.data.next_data_size[0] == 512:
            _, l, w, h = cond.shape
            sample = []
            for i in range(2):
                l_start = (l * i) // 2
                l_end   = (l * (i + 1)) // 2
                sample_i = []
                for j in range(2):
                    w_start = (w * j) // 2
                    w_end   = (w * (j + 1)) // 2
                    cond_ij = cond[:,l_start:l_end, w_start:w_end, :]
                    sampling_fn = sampling.get_pc_sampler(graph=graph,
                                    noise=noise,
                                    batch_dims=sampling_shape,
                                    predictor=cfg.sampling.predictor,
                                    steps=args.steps if args.steps else cfg.sampling.steps,
                                    denoise=cfg.sampling.noise_removal,
                                    eps=sampling_eps,
                                    device=device,
                                    cond=cond_ij,
                                    )
                        
                    sample_ij = sampling_fn(model)
                    sample_ij = sample_ij.reshape(sample_ij.shape[0], cfg.image_size[0], cfg.image_size[1], cfg.image_size[2]).cpu()
                    sample_i.append(sample_ij)
                sample_i = torch.cat(sample_i, dim=2)
                sample.append(sample_i)
            sample = torch.cat(sample, dim=1)
        else:
            sampling_fn = sampling.get_pc_sampler(graph=graph,
                            noise=noise,
                            batch_dims=sampling_shape,
                            predictor=cfg.sampling.predictor,
                            steps=args.steps if args.steps else cfg.sampling.steps,
                            denoise=cfg.sampling.noise_removal,
                            eps=sampling_eps,
                            device=device,
                            cond=cond,
                            )
            
            sample = sampling_fn(model)
            sample = sample.reshape(sample.shape[0], cfg.image_size[0], cfg.image_size[1], cfg.image_size[2]).cpu()

        for j in range(sample.shape[0]):
            generated_index = []
            cond_index = []
            gt_index = []
            for i in range(1, cfg.tokens):
                index = torch.nonzero(sample[j] == i ,as_tuple=False)
                generated_index.append(F.pad(index,(1,0),'constant',value = i))
                if cfg.data.prev_stage != 'none':
                    index = torch.nonzero(cond[j] == i ,as_tuple=False)
                    cond_index.append(F.pad(index,(1,0),'constant',value = i))
                    index = torch.nonzero(eval_batch[j] == i ,as_tuple=False)
                    gt_index.append(F.pad(index,(1,0),'constant',value = i))

            generated_indexes = torch.cat(generated_index, dim = 0).numpy()
            file_name = os.path.join(this_sample_dir, 'generated', f"sample_{cur_id+j}.txt")
            np.savetxt(file_name, generated_indexes)
            if cfg.data.prev_stage != 'none':
                cond_indexes = torch.cat(cond_index, dim = 0).cpu().numpy()
                file_name = os.path.join(this_sample_dir, 'cond', f"sample_{cur_id+j}.txt")
                np.savetxt(file_name, cond_indexes)
                gt_indexes = torch.cat(gt_index, dim = 0).cpu().numpy()
                file_name = os.path.join(this_sample_dir, 'gt', f"sample_{cur_id+j}.txt")
                np.savetxt(file_name, gt_indexes)
        cur_id += sampling_shape[0]


def run_multiprocess(rank, world_size, args, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, args)
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--prev_scene_path", default=None, type=str)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--this_sample_dir", type=str, default=None)
    parser.add_argument("--prev_data_sr", type=int, nargs=3, default=None)
    args = parser.parse_args()
    ngpus = args.ngpus

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))

    if args.ngpus > 1:
        mp.set_start_method("forkserver")
        mp.spawn(run_multiprocess, args=(ngpus, args, port), nprocs=ngpus, join=True)
    else:
        _run(0, 1, args)

            
if __name__=="__main__":
    main()