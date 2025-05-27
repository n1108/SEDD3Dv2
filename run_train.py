import datetime
import os
import os.path
import gc
from itertools import chain
import signal

import numpy as np
from model.cnn import DenoiseCNN
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
from model import SEDD, SEDDCond, SEDDCondBlock, SEDDBlock
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
    if cfg.model.type == 'cnn':
        if cfg.data.prev_stage != 'none':
            score_model = DenoiseCNN(cfg, cond=True).to(device)
        else:
            score_model = DenoiseCNN(cfg, cond=False).to(device)
    elif hasattr(cfg, 'block_dit') and cfg.block_dit == True:
        if cfg.data.prev_stage != 'none':
            score_model = SEDDCondBlock(cfg).to(device)
        else:
            score_model = SEDDBlock(cfg).to(device)
    elif cfg.data.prev_stage != 'none':
        score_model = SEDDCond(cfg).to(device)
    else:
        score_model = SEDD(cfg).to(device)
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

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)


    if cfg.training.snapshot_sampling:
        sampling_shape = (max(cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum), 1), cfg.image_size[0] * cfg.image_size[1] * cfg.image_size[2])
        if cfg.data.prev_stage != 'none':
            eval_cond, _ = next(eval_iter)
            cond = eval_cond.to(device)
            cond = cond[:sampling_shape[0]]
        else:
            cond=None
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device, cond)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']

        cond, batch = next(train_iter)
        batch, cond = batch.to(device).reshape(batch.shape[0], -1), cond.to(device)
        loss = train_step_fn(state, batch, cond)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                eval_cond, eval_batch = next(eval_iter)
                eval_batch, eval_cond = eval_batch.to(device).reshape(eval_batch.shape[0], -1), eval_cond.to(device)
                eval_loss = eval_step_fn(state, eval_batch, eval_cond)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples TODO:
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating voxels at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    sample = sample.reshape(sample.shape[0], cfg.image_size[0], cfg.image_size[1], cfg.image_size[2])
                    ema.restore(score_model.parameters())


                    for j in range(sample.shape[0]):
                        generated_index = []

                        for i in range(1, cfg.tokens):
                            index = torch.nonzero(sample[j] == i ,as_tuple=False)
                            generated_index.append(F.pad(index,(1,0),'constant',value = i))

                        generated_indexes = torch.cat(generated_index, dim = 0).cpu().numpy()
                        file_name = os.path.join(this_sample_dir, f"sample_{rank}_{j}.txt")
                        np.savetxt(file_name, generated_indexes)

                    dist.barrier()
