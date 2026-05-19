"""
Qwen3.5 Model implementation
Ported from model.ipynb with nanochat optimizations (COMPUTE_DTYPE, FlashAttention3, Linear casting).
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoqwen35.common import get_dist_info, print0, COMPUTE_DTYPE
from nanoqwen35.optim import MuonAdamW, DistMuonAdamW
from nanoqwen35.flash_attention import flash_attn

@dataclass
class Qwen3_5ModelConfig:
    vocab_size: int = 248320
    context_length: int = 4096 # can be large but we cap for memory usually
    emb_dim: int = 1024
    n_heads: int = 8
    n_layers: int = 24
    hidden_dim: int = 3584
    head_dim: int = 256
    qk_norm: bool = True
    n_kv_groups: int = 2
    rope_base: float = 10000000.0
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    layer_types: list = None
    linear_num_value_heads: int = 16
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    hidden_act: str = "silu"

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.fc2 = Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.fc3 = Linear(cfg.hidden_dim, cfg.emb_dim, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(x.dtype)

class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

def compute_rope_params(head_dim, theta_base=10_000.0, context_length=4096, partial_rotary_factor=1.0, dtype=torch.float32):
    assert head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, rotary_dim, 2, dtype=dtype)[: rotary_dim // 2] / rotary_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope_bhsd(x, cos, sin):
    # x shape: (B, num_heads, seq_len, head_dim)
    _, _, seq_len, head_dim = x.shape
    rot_dim = cos.shape[-1]
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x_rot * cos) + (rotated * sin)
    x_out = torch.cat([x_rotated, x_pass], dim=-1)
    return x_out.to(x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.d_in = cfg.emb_dim
        self.num_heads = cfg.n_heads
        self.num_kv_groups = cfg.n_kv_groups
        self.group_size = self.num_heads // self.num_kv_groups
        self.head_dim = cfg.head_dim if cfg.head_dim else self.d_in // self.num_heads
        self.d_out = self.num_heads * self.head_dim
        self.layer_idx = layer_idx

        # In Qwen3.5 full attention, W_query outputs d_out * 2 because of the context gate
        self.W_query = Linear(self.d_in, self.d_out * 2, bias=False)
        self.W_key = Linear(self.d_in, self.num_kv_groups * self.head_dim, bias=False)
        self.W_value = Linear(self.d_in, self.num_kv_groups * self.head_dim, bias=False)
        self.out_proj = Linear(self.d_out, self.d_in, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(b, num_tokens, self.num_heads, 2 * self.head_dim)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # RoPE
        queries = apply_rope_bhsd(queries, cos, sin)
        keys_new = apply_rope_bhsd(keys_new, cos, sin)

        # Transpose for FA3: (b, num_tokens, num_heads, head_dim)
        queries = queries.transpose(1, 2)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)

        if cache is None:
            # Training: casual attention with FA3
            context = flash_attn.flash_attn_func(queries, keys_new, values_new, causal=True)
            next_cache = None
        else:
            # Inference: with KV cache
            k_cache, v_cache = cache.get_layer_cache(self.layer_idx)
            context = flash_attn.flash_attn_with_kvcache(
                queries, k_cache, v_cache,
                k=keys_new, v=values_new,
                cache_seqlens=cache.cache_seqlens,
                causal=True
            )
            # Advance cache position after the last block processes
            if self.layer_idx == cache.n_layers - 1:
                cache.advance(num_tokens)
            next_cache = cache

        context = context.contiguous().reshape(b, num_tokens, self.d_out)
        context = context * torch.sigmoid(gate)
        out = self.out_proj(context)
        return out, next_cache

def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len :]) 
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out

def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - (sequence_length % chunk_size)) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype) if initial_state is None else initial_state.to(value.dtype)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=value.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (last_recurrent_state * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value) if initial_state is None else initial_state.to(value)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + (k_t.unsqueeze(-1) * delta.unsqueeze(-2))
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

class Qwen3_5GatedDeltaNet(nn.Module):
    # Note: Unoptimized PyTorch Implementation!
    # In the future, you can replace the delta rule functions with custom CUDA kernels (like FlashLinearAttention)
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.emb_dim
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_update = torch_causal_conv1d_update
        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

        self.in_proj_qkv = Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, mask=None, cos=None, sin=None, start_pos=0, cache=None):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (cache is not None and getattr(cache, 'has_previous_state', False) and seq_len == 1)

        conv_state, recurrent_state = None, None
        if cache is not None:
            conv_state = cache.linear_conv_states[self.layer_idx]
            recurrent_state = cache.linear_recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
        else:
            if cache is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache.linear_conv_states[self.layer_idx] = conv_state
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=None, output_final_state=cache is not None, use_qk_l2norm_in_kernel=True
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state, output_final_state=cache is not None, use_qk_l2norm_in_kernel=True
            )

        if cache is not None:
            cache.linear_recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out), cache

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_type, layer_idx):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(cfg, layer_idx)
        elif layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(cfg, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=cfg.rms_norm_eps)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.token_mixer(x, mask=mask, cos=cos, sin=sin, start_pos=start_pos, cache=cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, next_cache

class Qwen3_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.emb_dim),
            "h": nn.ModuleList([TransformerBlock(config, config.layer_types[i], i) for i in range(config.n_layers)]),
        })
        self.final_norm = RMSNorm(config.emb_dim, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.emb_dim, config.vocab_size, bias=False)

        cos, sin = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_base,
            context_length=config.context_length,
            partial_rotary_factor=config.partial_rotary_factor,
            dtype=COMPUTE_DTYPE,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        return 6 * nparams
        
    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        lm_head = self.lm_head.weight.numel() if hasattr(self, 'lm_head') else wte
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.final_norm.weight.numel()
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': sum(p.numel() for p in self.parameters()),
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5, use_muon=True):
        # Based on nanochat optimization setup
        model_dim = self.config.emb_dim
        ddp, rank, local_rank, world_size = get_dist_info()

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # All other params (transformer blocks and final norm)
        other_params = list(self.transformer.h.parameters()) + list(self.final_norm.parameters())

        matrix_params = [p for p in other_params if p.ndim >= 2]
        scalar_params = [p for p in other_params if p.ndim < 2]

        dmodel_lr_scale = (model_dim / 1024) ** -0.5

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=scalar_params, lr=scalar_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if use_muon:
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                ))
        else:
            param_groups.append(dict(
                kind='adamw', params=matrix_params, lr=matrix_lr,
                betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @torch.no_grad()
    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, Linear)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif hasattr(m, 'dt_bias'):
                # DeltaNet specific
                if hasattr(m.dt_bias, 'data'):
                    torch.nn.init.zeros_(m.dt_bias)
                if hasattr(m, 'A_log'):
                    torch.nn.init.normal_(m.A_log, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else getattr(kv_cache, 'pos', 0)
        cos_sin = (self.cos[T0:T0+T], self.sin[T0:T0+T])
        
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        
        for i, block in enumerate(self.transformer.h):
            x, _ = block(x, mask=None, cos=cos_sin[0], sin=cos_sin[1], start_pos=T0, cache=kv_cache)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
