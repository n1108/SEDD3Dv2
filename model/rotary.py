import torch
from torch import nn

from rotary_embedding_torch import RotaryEmbedding


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached

class RotaryPositionEmbedding3D(nn.Module):
    def __init__(
            self,
            image_size,
            hidden_size_head,
            cache=False,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            params_dtype=torch.float32,
    ):
        super().__init__()
        self.pos_emb = RotaryEmbedding(
            custom_freqs=custom_freqs,
            theta=theta,
            dim = hidden_size_head // 3,  
            freqs_for = freqs_for,
            max_freq = max_freq,
            num_freqs=num_freqs,
        )
        freqs = self.pos_emb.get_axial_freqs(image_size[0], image_size[1], image_size[2]).to(params_dtype)
        self.register_buffer("freqs", freqs, persistent=False)
        self.cache = cache
        self.seq_len_cached = None


    def forward(self, t, rope_position_ids, seperate_qkv=False):
        x_coords = rope_position_ids[:, :, 0]
        y_coords = rope_position_ids[:, :, 1]
        z_coords = rope_position_ids[:, :, 2]
        # breakpoint()
        freqs = self.freqs[x_coords, y_coords, z_coords]   # batch, seq_len
        if seperate_qkv:
            # dims are: batch, seq_len, head, dim
            self.cos_cached = freqs.cos()[:, :, None, :]
            self.sin_cached = freqs.sin()[:, :, None, :]
            return self.cos_cached, self.sin_cached
        # dims are: batch, seq_len, qkv, head, dim
        self.cos_cached = freqs.cos()[:, :, None, None, :].repeat(1,1,3,1,1)
        self.sin_cached = freqs.sin()[:, :, None, None, :].repeat(1,1,3,1,1)
        # This makes the transformation on v an identity.
        self.cos_cached[:,:,2,:,:].fill_(1.)
        self.sin_cached[:,:,2,:,:].fill_(0.)
            
        return self.cos_cached, self.sin_cached
 

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=-1
    )


# @torch.jit.script
def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
    print('cos', cos.shape, 'sin', sin.shape, 'qkv', qkv.shape)
    try:
        import flash_attn.layers.rotary
        cos = cos[0,:,0,0,:cos.shape[-1]//2]
        sin = sin[0,:,0,0,:sin.shape[-1]//2]
        print('cos_half', cos.shape, 'sin_half', sin.shape)
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(
            qkv, cos, sin
        )
    except:
        print('_apply_rotary_pos_emb_torchscript')
        return _apply_rotary_pos_emb_torchscript(qkv, cos, sin)