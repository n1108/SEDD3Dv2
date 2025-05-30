import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, current_image_size, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        if cond is not None:
            sr = round((batch.shape[1] // cond.shape[1] // cond.shape[2] // cond.shape[3]) ** (1/3))
            cond_expanded = cond.repeat_interleave(sr, dim=1) \
                         .repeat_interleave(sr, dim=2) \
                         .repeat_interleave(sr, dim=3).reshape(cond.shape[0], -1)
        else:
            cond_expanded = None
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None], cond_expanded)

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, cond, sigma, current_image_size=current_image_size)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch, cond_expanded)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, rank, mprint_fn, log_opt_step_freq): # 新增参数
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter_count = 0 # 重命名以避免与外部作用域变量冲突
    current_total_loss = 0 # 重命名以避免与外部作用域变量冲突
    
    # 用于存储当前日志周期内每个累积批次的损失 (仅 rank 0)
    batch_losses_for_current_log_period = []

    def step_fn(state, batch, current_image_size, cond=None):
        nonlocal accum_iter_count
        nonlocal current_total_loss
        nonlocal batch_losses_for_current_log_period

        model = state['model']
        loss_to_report_for_main_log = torch.tensor(0.0, device=batch.device) # 初始化

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            
            # 计算当前这个小批次的原始损失 (不除以 accum)
            current_batch_raw_loss = loss_fn(model, batch, current_image_size, cond=cond).mean()
            # 用于反向传播的损失 (除以 accum)
            loss_for_backward = current_batch_raw_loss / accum
            
            scaler.scale(loss_for_backward).backward()

            accum_iter_count += 1
            current_total_loss += loss_for_backward.detach() # 累加的是已经除以 accum 的损失

            # 如果是 rank 0，收集当前原始批次损失用于后续可能的打印
            if rank == 0:
                batch_losses_for_current_log_period.append(current_batch_raw_loss.item())

            if accum_iter_count == accum: # 优化器步骤将要发生
                accum_iter_count = 0

                state['step'] += 1 # 优化器步骤计数器增加
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss_to_report_for_main_log = current_total_loss # 这是累积后的平均损失
                current_total_loss = 0 # 重置累积损失

                # 如果是 rank 0 并且是主日志记录的优化器步骤
                if rank == 0 and state['step'] % log_opt_step_freq == 0:
                    mprint_fn(f"optimizer_step: {state['step']}")
                    for i, ind_loss in enumerate(batch_losses_for_current_log_period):
                        mprint_fn(f"    accum_batch: {i+1}/{accum}, raw_loss: {ind_loss:.5e}")
                    batch_losses_for_current_log_period.clear() # 打印后清空
                elif rank == 0: # 如果不是日志步骤，也清空列表，避免累积过多
                    batch_losses_for_current_log_period.clear()
            else:
                loss_to_report_for_main_log = current_total_loss 
        else: # 评估模式
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                # 评估模式下，accum 通常为1，所以 "individual" 和 "accumulated" 是一样的
                eval_loss = loss_fn(model, batch, current_image_size, cond=cond).mean()
                loss_to_report_for_main_log = eval_loss
                ema.restore(model.parameters())
                
                # 评估模式下不需要打印单个累积批次损失的逻辑
                if rank == 0:
                    batch_losses_for_current_log_period.clear()


        return loss_to_report_for_main_log

    return step_fn