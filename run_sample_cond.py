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


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--prev_scene_path", default=None, type=str)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise, cfg = load_model(args.model_path, device)
    sample_dir = os.path.join(args.model_path, "samples")
    this_sample_dir = os.path.join(sample_dir, "test")
    utils.makedirs(os.path.join(this_sample_dir, 'generated_cond'))
    utils.makedirs(os.path.join(this_sample_dir, 'cond_cond'))
    utils.makedirs(os.path.join(this_sample_dir, 'gt_cond'))
    _, eval_ds = data.get_dataloaders(cfg, distributed=False)
    eval_iter = iter(eval_ds)

    sampling_shape = (min(cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum), 4), cfg.image_size[0] * cfg.image_size[1] * cfg.image_size[2])
    sampling_eps = 1e-5
    sample_num = 0
    while sample_num < args.samples:
        eval_cond, eval_batch = next(eval_iter)
        cond = eval_cond.to(device)
        cond = cond[:sampling_shape[0]]
        # sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, cond)
        def proj_fun(x):
            idx = cond.reshape(cond.shape[0], -1) == 10
            x[idx] = 10
            return x
        
        sampling_fn = sampling.get_pc_sampler(graph=graph,
                        noise=noise,
                        batch_dims=sampling_shape,
                        predictor=cfg.sampling.predictor,
                        steps=args.steps if args.steps else cfg.sampling.steps,
                        denoise=cfg.sampling.noise_removal,
                        eps=sampling_eps,
                        device=device,
                        proj_fun=proj_fun,
                        cond=cond,
                        )
        
        sample = sampling_fn(model)
        sample = sample.reshape(sample.shape[0], cfg.image_size[0], cfg.image_size[1], cfg.image_size[2])

        for j in range(sample.shape[0]):
            generated_index = []
            cond_index = []
            gt_index = []
            for i in range(1, cfg.tokens):
                index = torch.nonzero(sample[j] == i ,as_tuple=False)
                generated_index.append(F.pad(index,(1,0),'constant',value = i))
                index = torch.nonzero(cond[j] == i ,as_tuple=False)
                cond_index.append(F.pad(index,(1,0),'constant',value = i))
                index = torch.nonzero(eval_batch[j] == i ,as_tuple=False)
                gt_index.append(F.pad(index,(1,0),'constant',value = i))

            generated_indexes = torch.cat(generated_index, dim = 0).cpu().numpy()
            file_name = os.path.join(this_sample_dir, 'generated_cond', f"sample_{sample_num+j}.txt")
            np.savetxt(file_name, generated_indexes)
            cond_indexes = torch.cat(cond_index, dim = 0).cpu().numpy()
            file_name = os.path.join(this_sample_dir, 'cond_cond', f"sample_{sample_num+j}.txt")
            np.savetxt(file_name, cond_indexes)
            gt_indexes = torch.cat(gt_index, dim = 0).cpu().numpy()
            file_name = os.path.join(this_sample_dir, 'gt_cond', f"sample_{sample_num+j}.txt")
            np.savetxt(file_name, gt_indexes)
        sample_num += sample.shape[0]
        print(sample_num, 'samples generated')
            
if __name__=="__main__":
    main()