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
from pathlib import Path

from load_model import load_model
import torch.nn.functional as F
import sampling
import data
import utils

def mprint(msg): # Simple print for non-rank 0, or adjust for actual logging needs
    # if rank == 0: # Assuming rank is available
    print(msg)

def _run(rank, world_size, args): # args are from argparse
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # load_model returns the multi-stage config (cfg)
    # model is unwrapped here
    model, graph, noise, cfg_loaded = load_model(args.model_path, device, args.ckpt_dir)
    
    # --- Determine sampling configuration based on args and loaded_cfg ---
    # Default to the last stage in the config if not specified by args
    # Add --sampling_stage_idx to argparse in main()
    sampling_stage_idx = getattr(args, 'sampling_stage_idx', -1) # Default to -1 (last stage)
    
    if not hasattr(cfg_loaded.data, 'stages') or not cfg_loaded.data.stages:
        # Fallback for older single-stage configs if necessary
        if rank == 0: mprint("Warning: cfg_loaded.data.stages not found. Using global image_size from loaded config.")
        # Create a dummy stage_conf from global cfg_loaded for consistency
        current_sampling_stage_cfg = OmegaConf.create({
            "name": "global_fallback",
            "image_size": cfg_loaded.image_size, # Must exist in old config
            "prev_data_size": cfg_loaded.data.prev_data_size if hasattr(cfg_loaded.data, 'prev_data_size') else None,
            "prev_stage_token": cfg_loaded.data.prev_stage if hasattr(cfg_loaded.data, 'prev_stage') else 'none',
            "batch_size": cfg_loaded.eval.batch_size, # Use eval batch size
            # Fill other fields if dataloader needs them (e.g. paths, though likely not for unconditional)
            "train_data_path": None, "quantized_train_data_path": None, 
            "valid_data_path": None, "quantized_valid_data_path": None,
        })
        # cfg_for_dataloader is the main config for things like ngpus, accum
        cfg_for_dataloader = cfg_loaded # The main loaded config
    else:
        if sampling_stage_idx >= len(cfg_loaded.data.stages) or sampling_stage_idx < -len(cfg_loaded.data.stages):
            mprint(f"Warning: sampling_stage_idx {sampling_stage_idx} is out of bounds for {len(cfg_loaded.data.stages)} stages. Defaulting to last stage.")
            sampling_stage_idx = -1
        current_sampling_stage_cfg = cfg_loaded.data.stages[sampling_stage_idx]
        cfg_for_dataloader = cfg_loaded # The main loaded config containing ngpus etc.

    current_image_size = list(current_sampling_stage_cfg.image_size) # Ensure list [H,W,U]

    if rank == 0:
        mprint(f"--- Sampling Configuration ---")
        mprint(f"Using stage: {current_sampling_stage_cfg.name} (index {sampling_stage_idx})")
        mprint(f"Sampling image_size: {current_image_size}")
        # Batch size for this sampling run (per GPU)
        # args.eval_batch_size is total batch size for this sampling run, to be split by world_size
        # If args.eval_batch_size is None, use stage's configured batch_size (which is per-GPU for training)
        gpus_for_sampling = world_size if world_size > 0 else 1 # Avoid division by zero if world_size is 0 from bad call
        
        #单GPU的采样批大小
        if args.eval_batch_size is not None:
            sampling_batch_size_per_gpu = args.eval_batch_size // gpus_for_sampling
        else:
            # Use stage's configured batch_size (assuming it's a per-GPU training/eval bs)
            sampling_batch_size_per_gpu = current_sampling_stage_cfg.get('eval_batch_size', current_sampling_stage_cfg.batch_size)
        mprint(f"Sampling batch_size per GPU: {sampling_batch_size_per_gpu}")
        mprint(f"Total samples to generate: {args.samples}")


    # prev_data_sr for upscaling loaded .txt condition files (if any)
    # This might need careful integration if prev_data_size also comes from stage config
    if args.prev_data_sr:
        # This typically modifies how a dataset (like CarlaDataset) loads .txt completions
        # We need to pass this to the dataset if it's used.
        # For now, store it in a way the dataset can access if needed, e.g., by modifying cfg_for_dataloader.data
        if not hasattr(cfg_for_dataloader.data, 'prev_data_sr_runtime'): # Avoid overwriting if already there
            OmegaConf.set_struct(cfg_for_dataloader.data, False) # Allow adding new keys
            cfg_for_dataloader.data.prev_data_sr_runtime = args.prev_data_sr
            OmegaConf.set_struct(cfg_for_dataloader.data, True)


    sample_dir_base = os.path.join(args.model_path, "samples_multires") # Changed base dir name
    this_sample_dir = args.this_sample_dir if args.this_sample_dir else os.path.join(sample_dir_base, current_sampling_stage_cfg.name)
    utils.makedirs(os.path.join(this_sample_dir, 'generated'))

    # Determine if model is conditional based on the *sampling stage's* prev_stage_token
    model_is_conditional_for_sampling = current_sampling_stage_cfg.prev_stage_token != 'none'
    
    eval_iter = None
    if model_is_conditional_for_sampling:
        utils.makedirs(os.path.join(this_sample_dir, 'cond'))
        utils.makedirs(os.path.join(this_sample_dir, 'gt'))
        # Get a dataloader for condition data, specific to this sampling stage and rank
        # For distributed sampling, each rank needs its segment of the dataset.
        # data.get_dataloaders already handles DistributedSampler if world_size > 1.
        # Pass world_size to get_dataloaders's distributed flag.
        _, eval_ds_loader = data.get_dataloaders(
            config=cfg_for_dataloader, # Main config for ngpus, accum (accum usually 1 for eval)
            current_stage_config=current_sampling_stage_cfg, 
            distributed=(world_size > 1), 
            prev_scene_path=args.prev_scene_path, # For inference from generated low-res
             # Pass sampling_batch_size_per_gpu as the eval_batch_size for the loader
            eval_batch_size=sampling_batch_size_per_gpu 
        )
        eval_iter = iter(eval_ds_loader)
    
    sampling_shape = (sampling_batch_size_per_gpu, 
                      current_image_size[0] * current_image_size[1] * current_image_size[2])
    sampling_eps = 1e-5 # Or from args/config

    # samples_per_gpu_this_run = args.samples // world_size (if args.samples is total)
    # If args.samples is per GPU, then samples_per_gpu_this_run = args.samples
    # Assuming args.samples is TOTAL samples to generate across all GPUs.
    if world_size == 0 and rank == 0 : gpus_for_calc = 1 #单GPU运行的特殊情况
    else: gpus_for_calc = world_size

    if args.samples % gpus_for_calc != 0 and rank == 0:
        mprint(f"Warning: Total samples {args.samples} not evenly divisible by {gpus_for_calc} GPUs.")
    
    num_samples_this_gpu = args.samples // gpus_for_calc
    # Handle uneven division for the first few ranks
    if rank < args.samples % gpus_for_calc:
        num_samples_this_gpu += 1
    
    # Global start index for samples this GPU will generate
    global_start_idx_for_gpu = (args.samples // gpus_for_calc) * rank + min(rank, args.samples % gpus_for_calc)

    num_batches_this_gpu = (num_samples_this_gpu + sampling_batch_size_per_gpu - 1) // sampling_batch_size_per_gpu

    mprint(f"GPU {rank}: Will generate {num_samples_this_gpu} samples, starting from global ID {global_start_idx_for_gpu}, in {num_batches_this_gpu} batches.")

    for k_batch_idx in tqdm(range(num_batches_this_gpu), disable=(rank!=0), desc=f"GPU {rank} Sampling"):
        # Determine the global sample IDs for the current batch on this GPU
        batch_start_global_id = global_start_idx_for_gpu + k_batch_idx * sampling_batch_size_per_gpu
        # Number of samples to actually generate in this batch (can be less than sampling_batch_size_per_gpu for the last batch)
        num_in_this_batch_actual = min(sampling_batch_size_per_gpu, num_samples_this_gpu - k_batch_idx * sampling_batch_size_per_gpu)

        if num_in_this_batch_actual <= 0: continue # Should not happen if loop range is correct

        current_batch_cond_data = None
        current_batch_gt_data = None # For saving ground truth
        if model_is_conditional_for_sampling:
            try:
                eval_cond_gpu_batch, eval_gt_gpu_batch = next(eval_iter)
                # Ensure the loaded batch is sliced to num_in_this_batch_actual if sampler produced more
                current_batch_cond_data = eval_cond_gpu_batch[:num_in_this_batch_actual].to(device)
                current_batch_gt_data = eval_gt_gpu_batch[:num_in_this_batch_actual] # Keep GT on CPU
            except StopIteration:
                mprint(f"GPU {rank}: Ran out of conditional data at batch {k_batch_idx}. Stopping sampling for this GPU.")
                break # Stop if no more conditions

        # Check if all samples needed for this batch are already generated
        all_generated_in_batch = True
        for j_sample_offset in range(num_in_this_batch_actual):
            global_sample_id_to_check = batch_start_global_id + j_sample_offset
            if not os.path.exists(os.path.join(this_sample_dir, 'generated', f"sample_{global_sample_id_to_check}.txt")):
                all_generated_in_batch = False
                break
        if all_generated_in_batch:
            if rank == 0: mprint(f"GPU {rank} Batch for global IDs {batch_start_global_id}-{batch_start_global_id+num_in_this_batch_actual-1} already generated. Skipping.")
            continue
        
        # Adjust sampling_shape for the actual number of samples in this batch
        current_batch_sampling_shape = (num_in_this_batch_actual, sampling_shape[1])

        # --- Perform sampling (split logic or standard) ---
        # The 512 split logic: This should be a model capability or a separate sampling strategy.
        # For now, assume standard sampling. If split is needed, it has to be very carefully
        # integrated with current_image_size and current_batch_sampling_shape.
        # The original split logic seemed to halve dimensions.
        
        # Standard sampling:
        _pc_sampler_fn = sampling.get_pc_sampler(
            graph=graph, noise=noise, # noise should be unwrapped if DDP'd in main script
            batch_dims=current_batch_sampling_shape, 
            predictor=cfg_loaded.sampling.predictor, # Use global sampling config
            steps=args.steps if args.steps else cfg_loaded.sampling.steps,
            denoise=cfg_loaded.sampling.noise_removal,
            eps=sampling_eps,
            device=device,
            cond=current_batch_cond_data,
        )
        
        generated_output_batch = _pc_sampler_fn(model) # model is already unwrapped
        generated_output_batch = generated_output_batch.reshape(
            num_in_this_batch_actual, # Use actual number in batch
            current_image_size[0], 
            current_image_size[1], 
            current_image_size[2]
        ).cpu()

        # Save samples
        for j_sample_offset in range(generated_output_batch.shape[0]): # Iterate over actual samples in generated_output_batch
            global_sample_id_to_save = batch_start_global_id + j_sample_offset
            
            # Skip if this specific sample was already found (double check for multi-file race conditions if any)
            # This check is mainly for resuming. If a batch was partially done.
            if os.path.exists(os.path.join(this_sample_dir, 'generated', f"sample_{global_sample_id_to_save}.txt")):
                # if rank == 0: mprint(f"Sample {global_sample_id_to_save} found. Skipping save.")
                continue

            sample_to_save = generated_output_batch[j_sample_offset]
            
            generated_tokens_list = []
            for token_val in range(1, cfg_loaded.tokens): # Use global cfg_loaded.tokens
                token_coords = torch.nonzero(sample_to_save == token_val, as_tuple=False)
                if token_coords.numel() > 0:
                    generated_tokens_list.append(F.pad(token_coords, (1,0), 'constant', value=token_val))
            
            if generated_tokens_list:
                all_gen_tokens = torch.cat(generated_tokens_list, dim=0).numpy()
                file_name_gen = os.path.join(this_sample_dir, 'generated', f"sample_{global_sample_id_to_save}.txt")
                np.savetxt(file_name_gen, all_gen_tokens, fmt='%i')
            else: # Create empty file if no tokens found
                Path(os.path.join(this_sample_dir, 'generated', f"sample_{global_sample_id_to_save}.txt")).touch()


            if model_is_conditional_for_sampling and current_batch_cond_data is not None and current_batch_gt_data is not None:
                # Save condition
                cond_to_save = current_batch_cond_data[j_sample_offset].cpu() # Condition was on device
                cond_tokens_list = []
                for token_val in range(1, cfg_loaded.tokens):
                    token_coords = torch.nonzero(cond_to_save == token_val, as_tuple=False)
                    if token_coords.numel() > 0:
                        cond_tokens_list.append(F.pad(token_coords, (1,0), 'constant', value=token_val))
                if cond_tokens_list:
                    all_cond_tokens = torch.cat(cond_tokens_list, dim=0).numpy()
                    file_name_cond = os.path.join(this_sample_dir, 'cond', f"sample_{global_sample_id_to_save}.txt")
                    np.savetxt(file_name_cond, all_cond_tokens, fmt='%i')
                else: Path(os.path.join(this_sample_dir, 'cond', f"sample_{global_sample_id_to_save}.txt")).touch()

                # Save ground truth
                gt_to_save = current_batch_gt_data[j_sample_offset] # GT was on CPU
                gt_tokens_list = []
                for token_val in range(1, cfg_loaded.tokens):
                    token_coords = torch.nonzero(gt_to_save == token_val, as_tuple=False)
                    if token_coords.numel() > 0:
                        gt_tokens_list.append(F.pad(token_coords, (1,0), 'constant', value=token_val))
                if gt_tokens_list:
                    all_gt_tokens = torch.cat(gt_tokens_list, dim=0).numpy()
                    file_name_gt = os.path.join(this_sample_dir, 'gt', f"sample_{global_sample_id_to_save}.txt")
                    np.savetxt(file_name_gt, all_gt_tokens, fmt='%i')
                else: Path(os.path.join(this_sample_dir, 'gt', f"sample_{global_sample_id_to_save}.txt")).touch()

    if rank == 0: mprint(f"GPU {rank} finished its sampling quota.")


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