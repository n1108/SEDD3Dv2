import datetime
import os
import os.path
import gc
from itertools import chain
import signal

import numpy as np
# from model.cnn import DenoiseCNN
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
# from model import SEDD, SEDDCond, SEDDCondBlock, SEDDBlock
from model import SEDDCond
from model.ema import ExponentialMovingAverage


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()

def handle_sigterm(signum, frame):
    print("Received SIGTERM, shutting down gracefully...")

def _run(rank, world_size, cfg):
    # signal.signal(signal.SIGTERM, handle_sigterm)
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    if cfg.data.prev_stage != 'none':
        score_model = SEDDCond(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # --- PER-STAGE DATA LOADERS ---
    train_iters = []
    eval_iters = []
    if not hasattr(cfg.data, 'stages') or not cfg.data.stages:
        raise ValueError("Multi-resolution training requires 'cfg.data.stages' to be defined in the config.")
    num_stages = len(cfg.data.stages)
    mprint(f"Initializing {num_stages} training stages as defined in cfg.data.stages.")

    for stage_idx, stage_conf in enumerate(cfg.data.stages):
        mprint(f"Loading data for stage {stage_idx}: {stage_conf.name}, image_size: {stage_conf.image_size}, batch_size (per-GPU): {stage_conf.batch_size}")
        # get_dataloaders expects global config (for ngpus, accum) and stage_conf (for paths, sizes)
        train_ds_stage, eval_ds_stage = data.get_dataloaders(
            config=cfg, 
            current_stage_config=stage_conf, 
            distributed=(world_size > 1)
        )
        train_iters.append(iter(train_ds_stage))
        eval_iters.append(iter(eval_ds_stage))
        mprint(f"Stage {stage_idx} '{stage_conf.name}' dataloaders initialized.")

    # --- STEP FUNCTIONS ---
    # optimize_fn is global (uses global cfg.optim)
    optimize_fn = losses.optimization_manager(cfg) 
    # Step functions are generic; they will receive current_image_size at call time.
    # Note: noise_module.module is passed to get_step_fn.
    train_step_fn = losses.get_step_fn(noise.module, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise.module, graph, False, optimize_fn, cfg.training.accum) # accum usually 1 for eval

    num_train_steps = cfg.training.n_total_iters # Use n_total_iters from global config
    mprint(f"Starting training loop from optimizer_step {initial_step} up to {num_train_steps}.")

    # --- MAIN TRAINING LOOP ---
    # state['step'] is the global optimizer step counter.
    # This loop continues as long as the number of optimizer steps is less than total.
    current_data_fetch_step = initial_step * cfg.training.accum # Approx. starting data fetch step
    
    while state['step'] < num_train_steps + 1:
        
        # Determine current stage. Cycle through stages per data fetch or per optimizer step.
        # To ensure balanced exposure per optimizer step if accum > 1,
        # it's common to tie stage cycling to the *optimizer step*.
        current_stage_idx = state['step'] % num_stages 
        current_stage_cfg = cfg.data.stages[current_stage_idx]
        current_image_size = list(current_stage_cfg.image_size) # Ensure it's a list [H,W,U]
        
        # Fetch data for the current stage for one accumulation step
        try:
            cond_data, batch_data = next(train_iters[current_stage_idx])
        except StopIteration: # Should not happen with cycle_loader
            mprint(f"Re-initializing train_iter for stage {current_stage_idx} as it was exhausted.")
            # Re-create only the specific loader that exhausted
            new_train_loader_for_stage, _ = data.get_dataloaders(cfg, current_stage_cfg, distributed=(world_size > 1))
            train_iters[current_stage_idx] = iter(new_train_loader_for_stage)
            cond_data, batch_data = next(train_iters[current_stage_idx])

        # batch_data is [B_gpu, H, W, U], cond_data is [B_gpu, Hc, Wc, Uc]
        # Reshape batch_data to [B_gpu, L_voxels] for loss_fn and graph ops
        batch_data_flat = batch_data.to(device).reshape(batch_data.shape[0], -1) 
        cond_data = cond_data.to(device) # Model expects [B, Hc, Wc, Uc]

        pre_call_optimizer_step = state['step'] # Optimizer step before calling train_step_fn

        # Pass current_image_size to train_step_fn
        # loss_val is the loss for this accumulation step (or the accumulated loss if optimizer step happened)
        loss_val = train_step_fn(state, batch_data_flat, current_image_size, cond_data)

        # Check if an optimizer step was made by train_step_fn
        if state['step'] > pre_call_optimizer_step:
            current_optimizer_step = state['step'] # This is the new optimizer step count
            
            # Logging (log accumulated loss after an optimizer step)
            if current_optimizer_step % cfg.training.log_freq == 0:
                if world_size > 1: 
                    dist.all_reduce(loss_val, op=dist.ReduceOp.AVG) # Average loss across GPUs
                mprint(f"optimizer_step: {current_optimizer_step}, stage_idx: {current_stage_idx} ({current_stage_cfg.name}), "
                       f"training_loss: {loss_val.item():.5e}")
            
            # Snapshot for preemption
            if current_optimizer_step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            # Evaluation
            if current_optimizer_step % cfg.training.eval_freq == 0:
                # Evaluate on a specific stage (e.g., current, or cycle, or all)
                # For simplicity, evaluate on the current training stage
                eval_stage_idx_for_log = current_stage_idx 
                eval_stage_cfg_for_log = current_stage_cfg
                eval_image_size_for_log = current_image_size
                
                mprint(f"Evaluating on stage: {eval_stage_cfg_for_log.name} at optimizer_step {current_optimizer_step}")
                try:
                    eval_cond, eval_batch = next(eval_iters[eval_stage_idx_for_log])
                except StopIteration:
                    mprint(f"Re-initializing eval_iter for stage {eval_stage_idx_for_log}.")
                    _, new_eval_loader_for_stage = data.get_dataloaders(cfg, eval_stage_cfg_for_log, distributed=(world_size > 1))
                    eval_iters[eval_stage_idx_for_log] = iter(new_eval_loader_for_stage)
                    eval_cond, eval_batch = next(eval_iters[eval_stage_idx_for_log])

                eval_batch_flat = eval_batch.to(device).reshape(eval_batch.shape[0], -1)
                eval_cond = eval_cond.to(device)
                
                eval_loss_val = eval_step_fn(state, eval_batch_flat, eval_image_size_for_log, eval_cond)
                if world_size > 1: 
                    dist.all_reduce(eval_loss_val, op=dist.ReduceOp.AVG)
                mprint(f"optimizer_step: {current_optimizer_step}, stage: {eval_stage_cfg_for_log.name}, "
                       f"evaluation_loss: {eval_loss_val.item():.5e}")

            # Full checkpoint saving and optional sampling
            if current_optimizer_step > 0 and \
               (current_optimizer_step % cfg.training.snapshot_freq == 0 or current_optimizer_step == num_train_steps):
                if rank == 0:
                    save_path = os.path.join(checkpoint_dir, f'checkpoint_{current_optimizer_step}.pth')
                    utils.save_checkpoint(save_path, state)
                    mprint(f"Saved checkpoint to {save_path}")

                if cfg.training.snapshot_sampling and rank == 0: # Sampling only on rank 0
                    # Choose a stage for sampling, e.g., the last one (often largest resolution)
                    # Or allow configuration of which stage to sample from.
                    sampling_stage_to_use_idx = cfg.training.get("sampling_stage_idx", -1) # Default to last stage
                    sampling_stage_cfg = cfg.data.stages[sampling_stage_to_use_idx]
                    sampling_image_size = list(sampling_stage_cfg.image_size)
                    
                    # Determine batch size for sampling on this single GPU
                    # Use stage's eval_batch_size or training_batch_size, not divided by accum
                    sampling_loader_bs = sampling_stage_cfg.get('eval_batch_size', sampling_stage_cfg.batch_size)
                    # Max with 1 to ensure at least one sample if bs is small
                    sampling_gpu_batch_size = max(sampling_loader_bs // (cfg.ngpus if world_size > 1 else 1), 1)


                    mprint(f"Generating snapshot samples at optimizer_step: {current_optimizer_step} "
                           f"using stage '{sampling_stage_cfg.name}' (resolution {sampling_image_size}), "
                           f"sampling batch size (GPU 0): {sampling_gpu_batch_size}")
                    
                    # sampling_shape is (num_samples_per_gpu, num_voxels)
                    sampling_shape = (sampling_gpu_batch_size, 
                                      sampling_image_size[0] * sampling_image_size[1] * sampling_image_size[2])
                    
                    # Get condition data for sampling
                    s_cond_data = None
                    # Model's prev_stage flag (e.g. from cfg.model or global cfg.data.prev_stage)
                    # This should ideally align with the sampling_stage_cfg.prev_stage_token
                    model_is_conditional = sampling_stage_cfg.prev_stage_token != 'none'

                    if model_is_conditional:
                        # Get a non-distributed loader for sampling conditions on rank 0
                        # Need to ensure this loader also uses the correct sampler for set_epoch if cycle_loader used internally
                        _, s_eval_loader = data.get_dataloaders(cfg, sampling_stage_cfg, distributed=False)
                        s_eval_iter = iter(s_eval_loader)
                        try:
                            s_eval_cond_batch, _ = next(s_eval_iter)
                            s_cond_data = s_eval_cond_batch.to(device)
                            s_cond_data = s_cond_data[:sampling_gpu_batch_size] # Ensure correct batch size
                        except StopIteration:
                            mprint("Warning: Could not get condition data for snapshot sampling.")
                            s_cond_data = None # Proceed with unconditional sampling if cond not found
                        del s_eval_loader, s_eval_iter # Clean up

                    # Get the sampler function
                    # Pass noise_module.module to get_pc_sampler
                    _pc_sampler_fn = sampling.get_pc_sampler(
                        graph=graph, 
                        noise=noise.module, 
                        batch_dims=sampling_shape,
                        predictor=cfg.sampling.predictor, # Global sampling predictor
                        steps=cfg.sampling.steps,         # Global sampling steps
                        denoise=cfg.sampling.noise_removal,
                        eps=sampling_eps,
                        device=device,
                        cond=s_cond_data,
                        current_image_size=sampling_image_size
                    )
                    
                    current_iter_sample_dir = os.path.join(sample_dir, f"iter_{current_optimizer_step}")
                    utils.makedirs(current_iter_sample_dir)

                    # Use EMA parameters for sampling
                    state['ema'].store(score_model.module.parameters())
                    state['ema'].copy_to(score_model.module.parameters())
                    
                    generated_sample = _pc_sampler_fn(score_model.module) # Pass unwrapped model
                    generated_sample = generated_sample.reshape(
                        generated_sample.shape[0], 
                        sampling_image_size[0], 
                        sampling_image_size[1], 
                        sampling_image_size[2]
                    )
                    state['ema'].restore(score_model.module.parameters())

                    # Save samples (e.g., as .txt files)
                    for j_idx in range(generated_sample.shape[0]):
                        sample_to_save = generated_sample[j_idx].cpu() # Move to CPU
                        # Format and save (example from original code)
                        formatted_indices_list = []
                        for token_val in range(1, cfg.tokens): # Assuming cfg.tokens is global vocab size (excluding special tokens)
                            token_coords = torch.nonzero(sample_to_save == token_val, as_tuple=False)
                            if token_coords.numel() > 0:
                                formatted_indices_list.append(F.pad(token_coords, (1,0), 'constant', value=token_val))
                        
                        if formatted_indices_list:
                            all_formatted_indices = torch.cat(formatted_indices_list, dim=0).numpy()
                            save_file_name = os.path.join(current_iter_sample_dir, f"sample_{rank}_{j_idx}.txt") # rank is 0 here
                            np.savetxt(save_file_name, all_formatted_indices, fmt='%i')
                        else:
                            mprint(f"Warning: No tokens found for sample {rank}_{j_idx} at step {current_optimizer_step}. Skipping save.")
                
                if world_size > 1:
                    dist.barrier() # Ensure all processes sync after potential saving/sampling on rank 0
        
        current_data_fetch_step +=1 # Increment for each data batch fetched

    mprint("Training loop finished.")
