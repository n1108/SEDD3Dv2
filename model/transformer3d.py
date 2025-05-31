import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn import flash_attn_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

class ResolutionEmbedder(nn.Module):
    """
    Embeds 3D resolution (H_patch, W_patch, U_patch) into vector representations.
    """
    def __init__(self, cond_dim, spatial_freq_emb_size=128, max_period_spatial=1000):
        super().__init__()
        self.cond_dim = cond_dim
        self.spatial_freq_emb_size = spatial_freq_emb_size # Sinusoidal embedding size for EACH spatial dimension
        self.max_period_spatial = max_period_spatial

        # MLP to process concatenated sinusoidal embeddings of H_patch, W_patch, U_patch
        # Input to MLP will be 3 * spatial_freq_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.spatial_freq_emb_size, cond_dim * 2), # Intermediate layer
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim)
        )
        self.mlp[0].weight.data.normal_(0, 0.02) # Initialize MLP like DiT
        self.mlp[0].bias.data.zero_()
        self.mlp[2].weight.data.normal_(0, 0.02)
        self.mlp[2].bias.data.zero_()


    def _sinusoidal_embedding(self, value_tensor, dim):
        """
        Create sinusoidal embeddings for a batch of scalar values.
        :param value_tensor: a 1-D Tensor of N values.
        :param dim: the dimension of the output.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period_spatial) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=value_tensor.device)
        
        args = value_tensor.float().unsqueeze(-1) * freqs.unsqueeze(0)
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: 
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, hwu_tensor_batch):
        """
        :param hwu_tensor_batch: A tensor of shape (B, 3) representing (H_patch, W_patch, U_patch) for each item in batch.
        """
        if hwu_tensor_batch.ndim == 1: 
            hwu_tensor_batch = hwu_tensor_batch.unsqueeze(0)
        
        B = hwu_tensor_batch.shape[0]

        h_p = hwu_tensor_batch[:, 0] 
        w_p = hwu_tensor_batch[:, 1] 
        u_p = hwu_tensor_batch[:, 2] 

        h_emb = self._sinusoidal_embedding(h_p, self.spatial_freq_emb_size) 
        w_emb = self._sinusoidal_embedding(w_p, self.spatial_freq_emb_size) 
        u_emb = self._sinusoidal_embedding(u_p, self.spatial_freq_emb_size) 

        concatenated_emb = torch.cat([h_emb, w_emb, u_emb], dim=-1) 
        res_emb = self.mlp(concatenated_emb) 
        return res_emb


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary._apply_rotary_pos_emb_torchscript(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x

class DDiTSparseBlock(DDiTBlock):

    def __init__(self, dim, n_heads, cond_dim, block_size, mlp_ratio=4, dropout=0.1):
        super().__init__(dim, n_heads, cond_dim, mlp_ratio, dropout)
        self.block_size = block_size
        self.key_value_empty = nn.Parameter(torch.zeros(1, self.block_size**3, 2*dim))
        nn.init.xavier_uniform_(self.key_value_empty.data)

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, 
                x, 
                c, 
                rotary_cos_sin_query, 
                rotary_cos_sin_key, 
                nonzero_blocks, 
                nonzero_blocks_neighbor, 
                hwu,
                seqlens=None):
        h, w, u = hwu
        block_size = self.block_size
        in_x = h // block_size
        in_y = w // block_size
        in_z = u // block_size
        batch_size, seq_len = x.shape[0], x.shape[1]

        def transform(x):
            x = rearrange(x, 'b (n m l) d -> b n m l d', n=h, m=w, l=u)
            x = rearrange(x, 'b (x l) (y w) (z u) d -> b (x y z) (l w u) d', l=block_size, w=block_size, u=block_size)
            return x

        def inverse_transform(x):
            x = rearrange(x, 'b (x y z) (l w u) d -> b (x l y w z u) d', l=block_size, w=block_size, u=block_size, x=in_x, y=in_y, z=in_z)
            return x

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        x = transform(x)        # B*block_len*(bs*bs*bs)*d
        x_selected = x[nonzero_blocks]    # nonzero_blocks_num*(bs*bs*bs)*d, bs:block_size
        qkv = self.attn_qkv(x_selected)
        (q, k, v) = qkv.chunk(3, dim=-1)  # nonzero_blocks_num*(bs*bs*bs)*d
        
        k_empty, v_empty = self.key_value_empty.to(k.dtype).chunk(2, dim=-1)
        k_full = k_empty.repeat_interleave(x.shape[1], dim=0).unsqueeze(0).repeat_interleave(x.shape[0], dim=0)  # B*block_len*(bs*bs*bs)*d
        v_full = v_empty.repeat_interleave(x.shape[1], dim=0).unsqueeze(0).repeat_interleave(x.shape[0], dim=0)  # B*block_len*(bs*bs*bs)*d
        k_full[nonzero_blocks] = k
        v_full[nonzero_blocks] = v
        k = rearrange(k_full, 'b s p d -> (b s) p d')[nonzero_blocks_neighbor]  # (nonzero_blocks_num*num_neighbor)*(bs*bs*bs)*d
        v = rearrange(v_full, 'b s p d -> (b s) p d')[nonzero_blocks_neighbor]  # (nonzero_blocks_num*num_neighbor)*(bs*bs*bs)*d
        q = rearrange(q, 'b p (h d) -> b p h d', h=self.n_heads)  # nonzero_blocks_num*(bs*bs*bs)*h*hd
        k = rearrange(k, 'b p (h d) -> b p h d', h=self.n_heads)  # (nonzero_blocks_num*num_neighbor)*(bs*bs*bs)*h*hd
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin_query
            q = rotary._apply_rotary_pos_emb_torchscript(
                q, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
            cos, sin = rotary_cos_sin_key
            k = rotary._apply_rotary_pos_emb_torchscript(
                k, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        k = rearrange(k, '(b n) p ... -> b (n p) ...', b = q.shape[0])  # nonzero_blocks_num*(num_neighbor*bs*bs*bs)*h*hd
        v = rearrange(v, '(b n) p (h d) -> b (n p) h d', b = q.shape[0], h=self.n_heads)
        output = flash_attn_func(q, k, v)   # nonzero_blocks_num*(bs*bs*bs)*h*hd
        
        output = rearrange(output, 'b s h d -> b s (h d)')

        x = torch.zeros(x.shape, dtype=output.dtype, device=output.device)   # B*block_len*(bs*bs*bs)*d
        x[nonzero_blocks] = output 
        x = inverse_transform(x)
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, patch_size):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c, hwu):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return unpatchify(x, c=self.out_channels, p=self.patch_size, hwu=hwu)


def unpatchify(x, c, p, hwu=None):
    """
    x: (N, T, patch_size**3 * C)
    imgs: (N, H, W, C)
    """
    if hwu is None:
        h = w = u= int(x.shape[1] ** 0.5)
    else:
        h, w, u = hwu
    assert h * w * u == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, u, p, p, p, c))
    x = torch.einsum('nhwupqrc->nhpwqurc', x)
    imgs = x.reshape(shape=(x.shape[0], h * p, w * p, u * p, c))
    
    return imgs


class VoxelPatchEmbeddingMixin(nn.Module):
    def __init__(self, dim, vocab_dim, hidden_size, patch_size, bias=True):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.embedding = nn.Embedding(vocab_dim, dim)
        self.proj = nn.Conv3d(dim, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        nn.init.kaiming_uniform_(self.embedding.weight.data, a=math.sqrt(5))
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        voxels = x
        emb = self.embedding(voxels).permute(0, 4, 1, 2, 3) 
        emb = self.proj(emb)
        return emb
    

class SEDDCond(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)  # TODO: class 0 for absorb

        self.vocab_embed = VoxelPatchEmbeddingMixin(config.model.hidden_size, vocab_size, config.model.hidden_size, config.model.patch_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)

        spatial_freq_emb_size = config.model.get('spatial_freq_emb_size', 128) 
        max_period_spatial = config.model.get('max_period_spatial', 1000)    
        self.res_map = ResolutionEmbedder(config.model.cond_dim, 
                                          spatial_freq_emb_size=spatial_freq_emb_size,
                                          max_period_spatial=max_period_spatial)

        self.rotary_emb = rotary.RotaryPositionEmbedding3D(
            hidden_size_head=config.model.hidden_size // config.model.n_heads,
        )



        self.blocks = nn.ModuleList([
            DDiTSparseBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, 
                            block_size=config.model.block_size,
                            dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim, config.model.patch_size)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, indices, cond, sigma, current_image_size):
        # print(f"[DEBUG] SEDDCond.forward: current_image_size received: {current_image_size}") # 添加调试打印
        b = indices.shape[0]
        h, w, u = current_image_size[0], current_image_size[1], current_image_size[2]
        # print(f"[DEBUG] SEDDCond.forward: h={h}, w={w}, u={u}") # 添加调试打印
        hwu=[h//self.config.model.patch_size, w//self.config.model.patch_size, u//self.config.model.patch_size]
        num_patches = hwu[0] * hwu[1] * hwu[2]
        position_ids = torch.zeros(num_patches, 3, device=indices.device)
        position_ids[:, 0] = torch.arange(num_patches) // (current_image_size[2] // self.config.model.patch_size) \
                            // (current_image_size[1] // self.config.model.patch_size) \
                            % (current_image_size[0] // self.config.model.patch_size)
        position_ids[:, 1] = torch.arange(num_patches) // (current_image_size[2] // self.config.model.patch_size) \
                            % (current_image_size[1] // self.config.model.patch_size)
        position_ids[:, 2] = torch.arange(num_patches) % (current_image_size[2] // self.config.model.patch_size)
        position_ids = torch.repeat_interleave(position_ids.unsqueeze(0), indices.shape[0], dim=0).long()
        position_ids[position_ids==-1] = 0
        if hasattr(self.config.data, 'crop_size'):
            position_ids[:,:,0] += torch.randint(0, current_image_size[0] // self.config.model.patch_size, (indices.shape[0], 1), device=indices.device)
            position_ids[:,:,1] += torch.randint(0, current_image_size[1] // self.config.model.patch_size, (indices.shape[0], 1), device=indices.device)
            position_ids[:,:,2] += torch.randint(0, current_image_size[2] // self.config.model.patch_size, (indices.shape[0], 1), device=indices.device)

        def transform(x):
            x = rearrange(x, 'b (n m l) d -> b n m l d', n=h//self.config.model.patch_size, m=w//self.config.model.patch_size, l=u//self.config.model.patch_size)
            x = rearrange(x, 'b (x l) (y w) (z u) d -> b (x y z) (l w u) d', l=self.config.model.block_size, w=self.config.model.block_size, u=self.config.model.block_size)
            return x
        
        position_ids = transform(position_ids)   # B*block_len*(bs*bs*bs)*3

        indices_tmp = indices.reshape(indices.shape[0], h, w, u)
        kernel_voxel_size = self.config.model.block_size * self.config.model.patch_size
        unfolded = indices_tmp.unfold(1, kernel_voxel_size, kernel_voxel_size).unfold(2, kernel_voxel_size, kernel_voxel_size).unfold(3, kernel_voxel_size, kernel_voxel_size)
        unfolded = unfolded.contiguous().view(indices_tmp.shape[0], -1, kernel_voxel_size, kernel_voxel_size, kernel_voxel_size)
        block_sums = unfolded.sum(dim=[2, 3, 4])
        nonzero_blocks = (block_sums != 0)
        neighbor_indices = get_neighbor_indices(b, 
                                                h//self.config.model.patch_size//self.config.model.block_size, 
                                                w//self.config.model.patch_size//self.config.model.block_size, 
                                                u//self.config.model.patch_size//self.config.model.block_size, 
                                                indices.device,
                                                self.config.model.neighbor_direction)
        nonzero_blocks_neighbor = neighbor_indices[nonzero_blocks].reshape(-1)     # nonzero_blocks_num*num_neighbor
        rope_position_ids_query = position_ids[nonzero_blocks]    # nonzero_blocks_num*(bs*bs*bs)*3
        rope_position_ids_key = position_ids.reshape(-1, position_ids.shape[2], position_ids.shape[3])[nonzero_blocks_neighbor]    # (nonzero_blocks_num*num_neighbor)*(bs*bs*bs)*3
        if not self.training:
            # indices_tmp = indices.reshape(indices.shape[0], h, w, u)
            # x1 = self.vocab_embed(indices_tmp[:,:indices_tmp.shape[1]//2,:indices_tmp.shape[2]//2,:], 
            #                       cond[:,:cond.shape[1]//2,:cond.shape[2]//2,:])
            # x2 = self.vocab_embed(indices_tmp[:,:indices_tmp.shape[1]//2,indices_tmp.shape[2]//2:,:], 
            #                       cond[:,:cond.shape[1]//2,cond.shape[2]//2:,:])
            # x3 = self.vocab_embed(indices_tmp[:,indices_tmp.shape[1]//2:,:indices_tmp.shape[2]//2,:], 
            #                       cond[:,cond.shape[1]//2:,:cond.shape[2]//2,:])
            # x4 = self.vocab_embed(indices_tmp[:,indices_tmp.shape[1]//2:,indices_tmp.shape[2]//2:,:], 
            #                       cond[:,cond.shape[1]//2:,cond.shape[2]//2:,:])
            # x = torch.cat((torch.cat((x1, x2), dim=3), torch.cat((x3, x4), dim=3)), dim=2)

            # indices_tmp = indices.reshape(indices.shape[0], h, w, u)
            # _, h_cond, w_cond, u_cond = cond.shape

            # 准备一个列表存放“每一行”拼好的结果
            rows = []
            for i in range(4):  # 在height方向切4份
                # 每一行再存放若干块
                row_blocks = []

                h_start = (h * i) // 4
                h_end   = (h * (i + 1)) // 4
                # h_cond_start = (h_cond * i) // 4
                # h_cond_end   = (h_cond * (i + 1)) // 4

                for j in range(4):  # 在width方向切4份
                    w_start = (w * j) // 4
                    w_end   = (w * (j + 1)) // 4
                    # w_cond_start = (w_cond * j) // 4
                    # w_cond_end   = (w_cond * (j + 1)) // 4

                    # 取出对应子块 (在 h, w 维度切片)
                    block_indices = indices_tmp[:, h_start:h_end, w_start:w_end, :]

                    # 送进 vocab_embed
                    x_ij = self.vocab_embed(block_indices)
                    row_blocks.append(x_ij)

                # 把本行的 4 块在 dim=3 上拼接(沿 width 方向)
                row_cat = torch.cat(row_blocks, dim=3)
                rows.append(row_cat)

            # 最后把 4 行在 dim=2 上拼接(沿 height 方向)，得到完整特征
            x = torch.cat(rows, dim=2)

        else:
            x = self.vocab_embed(indices.reshape(indices.shape[0], h, w, u))
        x = x.flatten(2).transpose(1, 2) 

        sigma_emb = self.sigma_map(sigma) # (B, cond_dim)
        hwu_tensor_batch = torch.tensor(hwu, device=indices.device, dtype=torch.float32).repeat(b, 1)
        res_emb = self.res_map(hwu_tensor_batch) # (B, cond_dim)
        
        # c = F.silu(self.sigma_map(sigma))
        c = F.silu(sigma_emb + res_emb) # 将时间和分辨率嵌入相加，然后通过 SiLU

        # rotary_cos_sin_query = self.rotary_emb(x, rope_position_ids_query, True)
        # rotary_cos_sin_key = self.rotary_emb(x, rope_position_ids_key, True)
        # rotary_cos_sin = self.rotary_emb(x)
        current_patched_size_tensor = torch.tensor(hwu, device=x.device, dtype=torch.long) # 确保是 long tensor
        rotary_cos_sin_query = self.rotary_emb(x, rope_position_ids_query, current_patched_size_tensor, True)
        rotary_cos_sin_key = self.rotary_emb(x, rope_position_ids_key, current_patched_size_tensor, True)

        target_resolution_for_checkpointing = [256, 256, 16]
        enable_checkpointing = (list(current_image_size) == target_resolution_for_checkpointing)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if enable_checkpointing:
                # print(f"DEBUG: Using checkpoint at resolution {current_image_size}") # 可选的调试信息
                for i in range(len(self.blocks)):
                    x = checkpoint.checkpoint(
                    self.blocks[i],                  # 要 checkpoint 的模块
                    x,                               # 第一个参数
                    c,                               # 第二个参数 (c)
                    rotary_cos_sin_query,            # ...
                    rotary_cos_sin_key,
                    nonzero_blocks,
                    nonzero_blocks_neighbor,
                    hwu,
                    None,                            # seqlens
                    use_reentrant=True              # 对于较新 PyTorch 版本，推荐 False 以获得更好性能和兼容性
                                                    # 如果使用旧版本或遇到问题，可以尝试 True
                )
            else:
                # print(f"DEBUG: Using direct call at resolution {current_image_size}") # 可选的调试信息
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x, 
                                   rotary_cos_sin_query=rotary_cos_sin_query, 
                                   rotary_cos_sin_key=rotary_cos_sin_key, 
                                   nonzero_blocks=nonzero_blocks, 
                                   nonzero_blocks_neighbor=nonzero_blocks_neighbor, 
                                   hwu=hwu,
                                   c=c, 
                                   seqlens=None)
                
            x = self.output_layer(x, c, hwu)


        x = x.reshape(x.shape[0], -1, x.shape[-1])
        if self.scale_by_sigma and self.absorb:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        return x
    

def get_neighbor_indices(batch_size, H, W, L, device, neighbor_direction='cross'):
    indices = torch.stack(torch.meshgrid(
        torch.arange(batch_size, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        torch.arange(L, device=device),
        indexing='ij',  # 'ij' indexing to match (batch, H, W, L) format
    ), dim=-1)  # Shape: (batch_size, H, W, L, 4)
    if neighbor_direction == 'ltf':
        # Define the relative shifts for the 8 neighbors
        shifts = torch.tensor([
            [0, 0, 0, 0],  # Self
            [0, 0, -1, 0],  # Left
            [0, -1, -1, 0],  # Left-up
            [0, -1, 0, 0],  # Up
            [0, 0, 0, -1],  # Front
            [0, 0, -1, -1],  # Front-left
            [0, -1, -1, -1],  # Front-left-up
            [0, -1, 0, -1],  # Front-up
        ], device=device)
    elif neighbor_direction == 'cross':
        shifts = torch.tensor([
            [0, 0, 0, 0],  # Self
            [0, 0, -1, 0],  # Left
            [0, -1, 0, 0],  # Up
            [0, 0, 0, -1],  # Front
            [0, 0, 1, 0],  # Right
            [0, 1, 0, 0],  # Down
            [0, 0, 0, 1],  # Back
        ], device=device)
    else:
        raise NotImplementedError
    num_neighbor = shifts.shape[0]
    # Broadcast indices and shifts to get neighbor indices
    neighbors = indices.unsqueeze(-2) + shifts.view(1, 1, 1, 1, -1, 4)  # Shape: (batch_size, H, W, L, num_neighbor, 4)
    # Handle boundary conditions (clipping)
    neighbors[..., 1] = torch.clamp(neighbors[..., 1], 0, H - 1)  # H dimension
    neighbors[..., 2] = torch.clamp(neighbors[..., 2], 0, W - 1)  # W dimension
    neighbors[..., 3] = torch.clamp(neighbors[..., 3], 0, L - 1)  # L dimension
    neighbors = neighbors[..., 0]*H*W*L + neighbors[..., 1]*W*L + neighbors[..., 2]*L + neighbors[..., 3]
    return neighbors.reshape(batch_size, -1, num_neighbor)      # B, block_len, num_neighbor