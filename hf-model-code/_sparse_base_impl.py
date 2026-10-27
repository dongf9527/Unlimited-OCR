# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek model and compatible with both DeepSeekV2 and DeepSeekV3"""
import math
import os as _os
import time as _time
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from einops import repeat
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as _llama_apply_rotary_pos_emb,
    repeat_kv as _llama_repeat_kv,
    # LlamaFlashAttention2
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

# TIME_DIAG=1 -> coarse wall-clock attribution per region. Forward and the
# gradient-checkpoint recompute accumulate into the same counters; the backward of each
# region is NOT attributed (it lands outside these blocks), so read the numbers as
# "share of the forward+recompute cost". Synchronises around every region -> diagnostic
# runs only. Printed (and reset) from the CE call site every TIME_DIAG_INTERVAL forwards.
_TD_ON = _os.environ.get("TIME_DIAG", "0") == "1"
_TD_MS = {}
_TD_CNT = {}

_WARMUP_MAIN_LOCAL_WINDOW = _os.environ.get("WARMUP_MAIN_LOCAL_WINDOW", "1") == "1"
_PAGE_SCOPE_PREFILL = _os.environ.get("PAGE_SCOPE_PREFILL", "0") == "1"

_CODE_STAMP = "warmup-window-page-switches-v4"
if _os.environ.get("CODE_STAMP_LOG", "1") == "1":
    _cs_switches = (f"local_window:{int(_WARMUP_MAIN_LOCAL_WINDOW)},"
                    f"page_prefill:{int(_PAGE_SCOPE_PREFILL)}")
    try:
        import hashlib as _hashlib
        with open(__file__, "rb") as _f:
            _cs_sha = _hashlib.sha256(_f.read()).hexdigest()[:12]
        _cs_mtime = _time.strftime("%Y-%m-%d %H:%M:%S",
                                   _time.localtime(_os.path.getmtime(__file__)))
        print(f"[CodeStamp] {_CODE_STAMP} sha256={_cs_sha} mtime={_cs_mtime} "
              f"switches={_cs_switches} file={__file__}", flush=True)
    except Exception as _cs_e:  # noqa: BLE001
        print(f"[CodeStamp] {_CODE_STAMP} switches={_cs_switches} "
              f"(stat failed: {_cs_e!r})", flush=True)


class _td:
    __slots__ = ("name", "t0")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if _TD_ON:
            torch.cuda.synchronize()
            self.t0 = _time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _TD_ON:
            torch.cuda.synchronize()
            dt = (_time.perf_counter() - self.t0) * 1e3
            _TD_MS[self.name] = _TD_MS.get(self.name, 0.0) + dt
            _TD_CNT[self.name] = _TD_CNT.get(self.name, 0) + 1
        return False


def _td_now():
    """Manual variant of ``_td`` for regions that cannot be wrapped in a with-block."""
    if not _TD_ON:
        return None
    torch.cuda.synchronize()
    return _time.perf_counter()


def _td_add(name, t0):
    if not _TD_ON or t0 is None:
        return
    torch.cuda.synchronize()
    _TD_MS[name] = _TD_MS.get(name, 0.0) + (_time.perf_counter() - t0) * 1e3
    _TD_CNT[name] = _TD_CNT.get(name, 0) + 1


def _td_report(tag):
    if not _TD_ON or not _TD_MS:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        _TD_MS.clear()
        _TD_CNT.clear()
        return
    _TD_MS["_n"] = _TD_MS.get("_n", 0.0)
    _TD_CNT["_n"] = _TD_CNT.get("_n", 0) + 1
    if _TD_CNT["_n"] % max(1, int(_os.environ.get("TIME_DIAG_INTERVAL", "4"))) != 0:
        return
    items = sorted(((k, v) for k, v in _TD_MS.items() if k != "_n"), key=lambda kv: -kv[1])
    tot = sum(v for k, v in _TD_MS.items() if k != "_n")
    body = " ".join(f"{k}={v:.0f}ms/{_TD_CNT[k]}" for k, v in items)
    print(f"[TimeDiag {tag} fwd_batches={_TD_CNT['_n']}] measured_total={tot:.0f}ms {body}",
          flush=True)
    _TD_MS.clear()
    _TD_CNT.clear()


# segment-aware helpers enabling sequence packing on the sparse
# path (ported from Megatron-Bridge). The `from .packing_patch import ...` line
# is the form transformers' dynamic-module loader detects as a relative import
# (its regex only matches `from .X import`, NOT `from . import X`), so it forces
# packing_patch.py to be copied into the HF modules cache. try/except keeps a
# direct (non-package) import working too.
try:
    from .packing_patch import is_packed as _pk_copy_hint  # noqa: F401
    from . import packing_patch as _pk
except Exception:  # pragma: no cover
    import packing_patch as _pk
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from .configuration_deepseek_v2 import DeepseekV2Config

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# optional xformers backend for sparse main-branch attention.
try:
    import xformers.ops as _xops
    from xformers.ops.fmha.attn_bias import LowerTriangularMaskWithTensorBias as _XLTBias
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False

# optional flex_attention path (PT >= 2.5). is_torch_flex_attn_available
# is used at forward time to pick this path when block_indices is present.
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    _compiled_flex_attention = torch.compile(flex_attention)
    def is_torch_flex_attn_available():
        return True
except Exception:
    def is_torch_flex_attn_available():
        return False
    _compiled_flex_attention = None

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV2Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekV2RMSNorm)




class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)


    # print()

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)


    return q_embed, k_embed


class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV2MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV2MLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        # config.kv_lora_rank + config.qk_rope_head_dim,
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            # self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
            #     self.qk_rope_head_dim,
            #     max_position_embeddings=self.max_position_embeddings,
            #     scaling_factor=scaling_factor,
            #     base=self.rope_theta,
            # )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)


        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        if (str((getattr(self.config, "sparse_attention_config", None) or {}).get(
                "sparse_main_rope_mode", "1d")).lower() == "page2d"
                and not getattr(DeepseekV2Attention, "_page_rope_warned", False)):
            DeepseekV2Attention._page_rope_warned = True
            print("[PageRope main][WARN] DeepseekV2Attention uses 1D RoPE, but "
                  "sparse_main_rope_mode=page2d: this branch has no 2-axis coordinates, "
                  "so its positions will not match the main branch.")
        cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)

        kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (torch.matmul(q_pe, k_pe.mT) +
                        torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)) * self.softmax_scale
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)

        attn_output = torch.matmul(attn_output, out_absorb.mT)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV2
class DeepseekV2FlashAttention2(DeepseekV2Attention):
    """
    DeepseekV2 flash attention module. This module inherits from `DeepseekV2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekV2FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        if (str((getattr(self.config, "sparse_attention_config", None) or {}).get(
                "sparse_main_rope_mode", "1d")).lower() == "page2d"
                and not getattr(DeepseekV2FlashAttention2, "_page_rope_warned", False)):
            DeepseekV2FlashAttention2._page_rope_warned = True
            print("[PageRope main][WARN] DeepseekV2FlashAttention2 uses 1D RoPE, but "
                  "sparse_main_rope_mode=page2d: this branch has no 2-axis coordinates, "
                  "so its positions will not match the main branch.")
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        # TODO: support compressed_kv for kv_cache (instead of key_states, value_states) in flash_attention version
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class SlidingWindowLlamaAttention(LlamaAttention):
    """LlamaAttention with sliding window KV cache using a ring buffer during decode."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # New transformers moved rotary_emb to model level; create our own
        if not hasattr(self, 'rotary_emb'):
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # Save sliding_window separately so we can disable it in config to prevent
        # DynamicCache from truncating prefill tokens
        self._sliding_window = getattr(config, 'sliding_window', None)

        # DSA lightning indexer (block-sparse attention). Attached per config.
        sparse_config = getattr(config, 'sparse_attention_config', {}) or {}
        self.use_sparse_attention = sparse_config.get('use_sparse_attention', False)
        _freq = sparse_config.get('sparse_attention_freq', [])
        self.indexer = None
        if self.use_sparse_attention and len(_freq) > layer_idx and _freq[layer_idx] == 1:
            self.indexer = DeepseekV2Indexer(config, layer_idx)

        # sparse_inconsistency_freq controls whether THIS
        # sparse layer runs its own indexer (Anchor/source, ==1) or reuses the
        # block_indices produced by the nearest preceding Anchor layer (Reuse, ==0).
        # The indexer module is STILL instantiated for every sparse layer so that
        # state_dict keys stay identical across warmup (all-1s) and sparse stages
        # (mixed 0/1) -> checkpoint resume / load_state_dict is fully compatible.
        _incon = (sparse_config.get('sparse_inconsistency_freq', None)
                  or getattr(config, 'sparse_inconsistency_freq', [])
                  or [])
        if self.indexer is not None and len(_incon) > layer_idx and _incon[layer_idx] == 0:
            self._is_indexer_source = False
            self.anchor_layer_idx = next(
                (a for a in range(layer_idx - 1, -1, -1)
                 if len(_incon) > a and _incon[a] == 1
                 and len(_freq) > a and _freq[a] == 1),
                None)
            if self.anchor_layer_idx is None:
                raise ValueError(
                    f"[SparseReuseConfigError] Layer {layer_idx} is configured as a Reuse layer "
                    f"(sparse_inconsistency_freq[{layer_idx}] == 0), but no preceding Anchor layer "
                    f"(freq == 1) was found in sparse_inconsistency_freq. Invalid configuration!"
                )
        else:
            self._is_indexer_source = True
            self.anchor_layer_idx = layer_idx

        self.main_rope_mode = str(sparse_config.get("sparse_main_rope_mode", "1d")).lower()
        self.rotary_emb_2d_main = None
        self._main_rope_pos_2d = None
        if self.main_rope_mode == "page2d":
            _pp = int(sparse_config.get("sparse_rope_page_pairs", 8))
            _pt = float(sparse_config.get("sparse_rope_page_theta", 256.0))
            _tt = float(sparse_config.get("sparse_rope_tok_theta",
                                          getattr(config, "rope_theta", 10000.0)))
            _tm = str(sparse_config.get("sparse_main_rope_tok_freq_mode", "truncate"))
            _pm = int(sparse_config.get("sparse_rope_page_mod", 0))
            try:
                self.rotary_emb_2d_main = DeepseekV2IndexerRotary2D(
                    self.head_dim, _tt, _pt, _pp, tok_freq_mode=_tm, page_mod=_pm)
            except ValueError as _e:
                print(f"[PageRope main layer={layer_idx}] WARNING: {_e}; falling back to 1d",
                      flush=True)
                self.main_rope_mode = "1d"
            if layer_idx == 0 and self.rotary_emb_2d_main is not None:
                _r = self.rotary_emb_2d_main
                print(f"[PageRope main] mode=page2d head_dim={self.head_dim} "
                      f"page_pairs={_pp} page_theta={_pt} tok_theta={_tt} tok_freq_mode={_tm} "
                      f"page_mod={_pm} "
                      f"capacity page={math.pi / float(_r.inv_page[-1]):.0f}pages "
                      f"tok={math.pi / float(_r.inv_tok[-1]):.0f}tokens", flush=True)

    def _main_rope(self, x, position_ids, past_kv=None, q_len=None):
        """Single entry point for the main-branch rope. All three call sites
        (_attn_forward / sparse teacher q,k / ring decode) go through it, so prefill and
        decode share one coordinate system (the KV cache stores already-rotated K)."""
        if self.main_rope_mode != "page2d" or self.rotary_emb_2d_main is None:
            return self.rotary_emb(x, position_ids)
        if q_len is None:
            q_len = x.shape[2] if x.dim() == 4 else x.shape[1]
        coords = self._main_rope_coords(q_len, past_kv)
        if coords is None:
            return self.rotary_emb(x, position_ids)
        return self.rotary_emb_2d_main(x, coords[0], coords[1])

    def _main_rope_coords(self, q_len, past_kv):
        """prefill: full-sequence coordinates passed down by the model layer (or computed
        here as a fallback); decode: incremental state machine. Returning None means this
        call falls back to 1D."""
        if self.indexer is None:
            return None
        if q_len > 1:
            coords = self._main_rope_pos_2d
            if coords is None:
                src = getattr(self, '_sparse_source_input_ids', None)
                coords = self.indexer._build_page_rope_pos(src, q_len)
                self._main_rope_pos_2d = coords
            if coords is None or coords[0].shape[-1] != q_len:
                return None
            if past_kv is not None:
                meta = getattr(self.indexer, '_page_rope_meta', None)
                if meta is not None:
                    st = getattr(past_kv, '_page_rope_state', None)
                    if st is None:
                        st = {}
                        past_kv._page_rope_state = st
                    st[self.layer_idx] = {
                        'page': coords[1][:, -1].clone(),
                        'tok': coords[0][:, -1].clone(),
                        'pf_cnt': meta['pf_cnt'],
                        'n_pg': meta['n_pg'],
                    }
            return coords
        st = getattr(past_kv, '_page_rope_state', None) if past_kv is not None else None
        st = st.get(self.layer_idx) if st else None
        ids = getattr(self, '_sparse_source_input_ids', None)
        if st is None or ids is None or ids.shape[-1] < 1:
            return self.indexer._page_rope_bail('decode without page-rope state')
        new = ids[:, -1].to(st['page'].device)
        is_pg = new == self.indexer.page_token_id
        st['page'] = torch.where(is_pg, (st['page'] + 1).clamp(max=st['n_pg'] - 1), st['page'])
        pg_start = st['pf_cnt'].gather(1, st['page'][:, None].clamp(min=0)).squeeze(1)
        st['tok'] = torch.where(is_pg, pg_start, st['tok'] + 1)
        return st['tok'][:, None], st['page'][:, None]

    def forward(self, *args, **kwargs):
        import math

        # Compatibility: new DynamicCache uses .layers[i].keys/.values instead of .key_cache[i]/.value_cache[i]
        def _get_kcache(cache, layer_idx):
            if hasattr(cache, 'key_cache'):
                return cache.key_cache[layer_idx]
            return cache.layers[layer_idx].keys

        def _get_vcache(cache, layer_idx):
            if hasattr(cache, 'value_cache'):
                return cache.value_cache[layer_idx]
            return cache.layers[layer_idx].values

        # Extract args
        def _get(name, idx, default=None):
            if name in kwargs:
                return kwargs[name]
            if len(args) > idx:
                return args[idx]
            return default

        hidden_states = _get('hidden_states', 0)
        attention_mask = _get('attention_mask', 1)
        position_ids = _get('position_ids', 2)
        past_kv = _get('past_key_value', 3)
        if past_kv is None:
            past_kv = kwargs.get('past_key_values', None)
        output_attentions = _get('output_attentions', 4, False)

        # Dimensions from config (new transformers removed self.num_heads)
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim
        num_kv_groups = self.num_key_value_groups

        bsz, q_len, _ = hidden_states.size()
        W = getattr(self.config, '_ring_window', None)  # Read from config (set before generate)

        # --- Helper: standard QKV attention ---
        def _attn_forward(use_cache_update=True, page_ids=None):
            query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            cos, sin = self._main_rope(value_states, position_ids, past_kv, q_len)
            query_states, key_states = _llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_kv is not None and use_cache_update:
                key_states, value_states = past_kv.update(key_states, value_states, self.layer_idx)

            k = _llama_repeat_kv(key_states, num_kv_groups)
            v = _llama_repeat_kv(value_states, num_kv_groups)

            # SDPA: avoids materializing the [B, H, S, S] fp32 softmax tensor
            # (~25 GiB at S~26k) that OOMed during gradient-checkpoint recompute.
            # Attention probs are never consumed externally (the KL teacher probs
            # are computed separately in the prefill branch), so this is equivalent.
            attn_mask = None
            _is_causal = False
            if attention_mask is not None:
                attn_mask = attention_mask[:, :, :, :k.shape[-2]]
            elif _pk.is_packed(position_ids) and query_states.shape[2] > 1:
                # no dense mask is passed under padding_free, but a
                # packed row concatenates several samples -> plain causal would
                # leak across sample boundaries. Build a segment-block-diagonal
                # causal keep-mask (True == attend) from the reset position_ids.
                attn_mask = _pk.dense_segment_keep(position_ids, query_states.device)
            elif query_states.shape[2] > 1:
                # no dense mask passed (no padding) -> let SDPA do the
                # causal masking itself instead of materialising an [B,1,S,S] mask.
                _is_causal = True

            # warmup + image_bucket_constraint -> the dense teacher
            # must not attend the image BUCKETS of other pages (text blocks stay visible),
            # i.e. exactly the block-level domain the sparse stage selects over.
            # Turn whatever mask we have into a bool keep-mask and AND the page
            # constraint in place (chunked over queries, no extra [B,1,S,S] tensor).
            if page_ids is not None and query_states.shape[2] > 1:
              with _td("pagemask"):
                _ipi, _qpi = page_ids
                _klen = k.shape[-2]
                _qpos_flat = torch.arange(
                    query_states.shape[2], device=query_states.device).unsqueeze(0).expand(bsz, -1)
                if attn_mask is None:
                    _kpos = torch.arange(_klen, device=query_states.device)
                    _qpos = position_ids if position_ids is not None else _qpos_flat
                    keep = _kpos[None, None, None, :] <= _qpos[:, None, :, None]
                elif attn_mask.dtype == torch.bool:
                    keep = attn_mask
                else:
                    keep = attn_mask[..., :_klen] > -1.0
                self.indexer.apply_page_image_attn_mask(keep, _ipi, _qpi, _qpos_flat)
                attn_mask = keep
                _is_causal = False

            if (_WARMUP_MAIN_LOCAL_WINDOW and self.indexer is not None
                    and self.indexer.training_stage == 'warmup'
                    and query_states.shape[2] > 1):
                with _td("localwindow"):
                    _klen = k.shape[-2]
                    _qpos_flat = torch.arange(
                        query_states.shape[2], device=query_states.device).unsqueeze(0).expand(bsz, -1)
                    _qpos = position_ids if position_ids is not None else _qpos_flat
                    if attn_mask is None:
                        _kpos = torch.arange(_klen, device=query_states.device)
                        keep_w = _kpos[None, None, None, :] <= _qpos[:, None, :, None]
                    elif attn_mask.dtype == torch.bool:
                        keep_w = attn_mask
                    else:
                        keep_w = attn_mask[..., :_klen] > -1.0
                    _kpos_w = torch.arange(_klen, device=query_states.device)
                    local_ok = _kpos_w[None, None, None, :] > (
                        _qpos[:, None, :, None] - self.indexer.local_window)
                    _img_key = None
                    _src_w = getattr(self, '_sparse_source_input_ids', None)
                    if (_src_w is not None and _src_w.ndim == 2
                            and self.indexer.image_token_id is not None):
                        _ids_w = _src_w[..., :_klen].to(device=query_states.device)
                        if _ids_w.shape[-1] < _klen:
                            _ids_w = F.pad(_ids_w, (0, _klen - _ids_w.shape[-1]), value=-1)
                        _itm_w = _ids_w == self.indexer.image_token_id       # [B, Sk]
                        _bs_w = self.indexer.block_size
                        _nblk_w = -(-_klen // _bs_w)
                        _pad_w = _nblk_w * _bs_w - _klen
                        if _pad_w > 0:
                            _itm_w = F.pad(_itm_w, (0, _pad_w), value=False)
                        _img_key = _itm_w.view(_itm_w.shape[0], _nblk_w, _bs_w).any(dim=-1)
                        _img_key = _img_key.repeat_interleave(
                            _bs_w, dim=-1)[..., :_klen][:, None, None, :]
                    keep_local = local_ok if _img_key is None else (local_ok | _img_key)
                    attn_mask = keep_w & keep_local
                    _is_causal = False
                    # layer-2 only, every kl_log_interval calls (mirrors
                    # DeepseekV2Indexer._diag_due). `pre` is the mask before the window
                    # (page-image bucket + causal/packed text), `post` is what SDPA
                    # actually sees.
                    if self.indexer._diag_due("_mainwin_log_step"):
                        with torch.no_grad():
                            _pre = keep_w.sum(dim=-1).float()      # [B, H|1, Sq]
                            _post = attn_mask.sum(dim=-1).float()
                            _loc = (keep_w & local_ok).sum(dim=-1).float()
                            _out_img = (attn_mask & ~local_ok).sum(dim=-1).float()
                            print(
                                f"[MainWindow layer={self.layer_idx}] "
                                f"local_window={self.indexer.local_window} "
                                f"k_len={_klen} img_key={_img_key is not None} "
                                f"pre/q: mean={_pre.mean().item():.1f} "
                                f"max={_pre.max().item():.0f} | "
                                f"post/q: mean={_post.mean().item():.1f} "
                                f"min={_post.min().item():.0f} "
                                f"max={_post.max().item():.0f} | "
                                f"in_window/q: max={_loc.max().item():.0f} | "
                                f"img_outside/q: mean={_out_img.mean().item():.1f} "
                                f"max={_out_img.max().item():.0f}",
                                flush=True)
            with _td("attn_sdpa"):
                attn_output = nn.functional.scaled_dot_product_attention(
                    query_states, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=_is_causal,
                )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_kv

        # Prefill or no sliding window
        # True prefill: W disabled, no cache, or first forward (prefill_length not yet recorded)
        _is_true_prefill = (W is None or past_kv is None or
                            (q_len > 1 and (not hasattr(past_kv, '_prefill_length') or
                                            self.layer_idx not in past_kv._prefill_length)))
        if _is_true_prefill:
            kl_loss = None
            if self.indexer is not None and self.training:
                idx = self.indexer
                # main-branch q/k/v with RoPE (mirror _attn_forward, needed as KL teacher)
                q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                k_ = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                v_ = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                cos, sin = self._main_rope(v_, position_ids, past_kv, q_len)
                q, k_ = _llama_apply_rotary_pos_emb(q, k_, cos, sin)
                pos_ids = position_ids
                if pos_ids is None:
                    pos_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
                scaling = 1.0 / math.sqrt(head_dim)
                itm = ipi = qpi = None
                src_ids = getattr(self, '_sparse_source_input_ids', None)
                with _td("pageids"):
                    if idx.image_bucket_constraint:
                        itm, ipi, qpi = idx._build_page_aware_ids(
                            src_ids, idx.image_token_id, idx.page_token_id, k_.shape[2])
                    else:
                        # image_bucket_constraint=False: restrict remote top-k to ALL
                        # image blocks (any page); only the flat image mask is needed.
                        itm = idx._build_flat_image_token_mask(
                            src_ids, idx.image_token_id, k_.shape[2])
                idx._log_image_mask(itm, src_ids, k_.shape[2])
                # build (tok_pos, page_pos) ONCE per layer-forward and hand it
                # to the indexer; _project_index_qk falls back to 1D whenever it is None.
                # packing rows are excluded: the <PAGE> cumsum would run across the sample
                # boundaries of a packed row and mis-number every page after the first sample.
                if idx.rope_mode == "page2d":
                    if _pk.is_packed(pos_ids):
                        idx._rope_pos_2d = idx._page_rope_bail("packed row")
                    else:
                        idx._rope_pos_2d = idx._build_page_rope_pos(src_ids, k_.shape[2])
                if idx.training_stage == 'sparse':
                    # Anchor (source) layers compute &
                    # cache block_indices; Reuse layers read from the cache keyed by
                    # their anchor_layer_idx. The cache is reset per-forward in
                    # DeepseekV2Model.forward and survives gradient-checkpointing
                    # backward recompute (Anchor layers recompute & re-store their
                    # own key; Reuse layers read the matching anchor key).
                    # NOTE: must NOT be stored on `self.config` -- PretrainedConfig
                    # gets JSON-serialized on every checkpoint save (config.save_pretrained
                    # -> to_json_string), and a cached Tensor there breaks
                    # `TypeError: Object of type Tensor is not JSON serializable`.
                    _bi_cache = getattr(self, '_sparse_block_indices_cache', None)
                    if _bi_cache is None:
                        _bi_cache = {}
                        self._sparse_block_indices_cache = _bi_cache
                    if self._is_indexer_source:
                        block_indices, kl_loss = idx.select_and_kl_loss(
                            hidden_states, q, k_, (cos, sin), pos_ids, scaling,
                            image_token_mask=itm, image_token_page_ids=ipi, query_token_page_ids=qpi)
                        _bi_cache[self.layer_idx] = block_indices
                    else:
                        block_indices = _bi_cache.get(self.anchor_layer_idx)
                        kl_loss = None
                        if block_indices is None:
                            raise RuntimeError(
                                f"[SparseReuseError] Layer {self.layer_idx} (Reuse layer) failed to retrieve "
                                f"cached block_indices from Anchor layer {self.anchor_layer_idx}. "
                                f"Fallback to dense attention is NOT allowed!"
                            )
                    idx._log_token_coverage(block_indices, pos_ids, k_.shape[2])

                    # fused Triton gather-attention for the sparse
                    # main branch. Attends per query over selected NON-LOCAL top-K blocks
                    # UNION the token-level local window [pos-local_window+1, pos] (the
                    # local union is done inside the kernel), avoiding the dense
                    # [B, H, Sq, Sk] block mask. Env-gated (GATHER_ATTN_TRITON, default on);
                    # returns None -> block-mask fallback on unsupported shapes.
                    import os as _os_g
                    _gather_out = None
                    # the eager 4D mask is a *pure causal* mask whenever there is no
                    # padding/packing (a padded key column would be masked for EVERY query row,
                    # incl. the last). The gather kernel is causal by construction, so a
                    # padding-free mask is redundant -> pass None to enable the fused path
                    # instead of falling back to the dense [B, H, Sq, Sk] softmax (OOM at long
                    # ctx). If any key is masked for the last query row, real padding exists ->
                    # keep the mask and let it fall back (correctness over memory).
                    _gather_mask = attention_mask
                    if _gather_mask is not None and bool((_gather_mask[:, :, -1, :] > -1.0).all()):
                        _gather_mask = None
                    if (_os_g.environ.get("GATHER_ATTN_TRITON", "1") == "1"
                            and q.is_cuda and block_indices is not None
                            and block_indices.shape[-1] > 0):
                        try:
                            from .sparse_gather_triton import triton_gather_attention
                            _gather_out = triton_gather_attention(
                                q, k_, v_, block_indices, pos_ids, _gather_mask,
                                idx.block_size, 1.0 / math.sqrt(head_dim),
                                local_window=idx.local_window,
                            )
                        except Exception as _e:
                            _gather_out = None
                            if self.layer_idx == 2 and not getattr(self, "_gather_triton_warned", False):
                                self._gather_triton_warned = True
                                print(f"[SparseAttn] triton gather path failed "
                                      f"({type(_e).__name__}: {_e}); falling back to block-mask attention",
                                      flush=True)
                    if _gather_out is not None:
                        if self.layer_idx == 2 and not getattr(self, "_gather_logged", False):
                            self._gather_logged = True
                            print(f"[SparseAttn layer={self.layer_idx}] using fused triton gather kernel "
                                  f"(k_len={k_.shape[2]})", flush=True)
                        attn_output = _gather_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                        attn_output = self.o_proj(attn_output)
                        if past_kv is not None:
                            k_, v_ = past_kv.update(k_, v_, self.layer_idx)
                        result = (attn_output, None, past_kv)
                    else:
                        # flex_attention path — avoids materialising [B,H,Sq,Sk] dense mask.
                        # Activated via sparse_attention_config.use_flex_attention=true.
                        _use_flex = (
                            is_torch_flex_attn_available()
                            and _compiled_flex_attention is not None
                            and isinstance(getattr(self.config, "sparse_attention_config", None), dict)
                            and self.config.sparse_attention_config.get("use_flex_attention", False)
                            and block_indices is not None
                            and block_indices.shape[-1] > 0
                            # flex causal_mask_mod is single-segment
                            # (q_idx>=kv_idx global); it would leak across samples.
                            # Fall through to the segment-aware build_block_mask path.
                            and not _pk.is_packed(pos_ids)
                        )
                        if _use_flex:
                            flex_bm = idx.build_flex_block_mask(block_indices, k_.shape[2], pos_ids)
                            kk_flex = _llama_repeat_kv(k_, num_kv_groups)
                            vv_flex = _llama_repeat_kv(v_, num_kv_groups)
                            attn_out = _compiled_flex_attention(
                                q, kk_flex, vv_flex, block_mask=flex_bm,
                                scale=1.0 / math.sqrt(head_dim),
                            )
                            attn_output = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                            attn_output = self.o_proj(attn_output)
                            if past_kv is not None:
                                k_, v_ = past_kv.update(k_, v_, self.layer_idx)
                            result = (attn_output, None, past_kv)
                        else:
                            # --- dense sparse fallback (eager SDPA-style with block mask) ---
                            kk = _llama_repeat_kv(k_, num_kv_groups)
                            vv = _llama_repeat_kv(v_, num_kv_groups)
                            attn_weights = torch.matmul(q, kk.transpose(2, 3)) / math.sqrt(head_dim)
                            cmask = attention_mask[:, :, :, :kk.shape[-2]] if attention_mask is not None else None
                            if cmask is not None:
                                attn_weights = attn_weights + cmask
                            keep = idx.build_block_mask(block_indices, cmask, k_.shape[2],
                                                        attn_weights.dtype, attn_weights.device, pos_ids)

                            # xformers backend (activated via use_xformers=true)
                            _use_xformers = (
                                _XFORMERS_AVAILABLE
                                and isinstance(getattr(self.config, "sparse_attention_config", None), dict)
                                and self.config.sparse_attention_config.get("use_xformers", False)
                            )
                            if _use_xformers:
                                xq = q.transpose(1, 2)  # [B, Sq, H, D]
                                xk = kk.transpose(1, 2)
                                xv = vv.transpose(1, 2)
                                xbias = torch.zeros(keep.shape, dtype=xq.dtype, device=xq.device)
                                xbias.masked_fill_(~keep, -1e9)
                                attn_out = _xops.memory_efficient_attention(
                                    xq, xk, xv, attn_bias=xbias, scale=1.0 / math.sqrt(head_dim))
                                attn_output = attn_out.reshape(bsz, q_len, -1)
                            else:
                                attn_weights = attn_weights.masked_fill(~keep, float('-inf'))
                                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                                attn_output = torch.matmul(attn_weights, vv)
                                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                            attn_output = self.o_proj(attn_output)
                            if past_kv is not None:
                                k_, v_ = past_kv.update(k_, v_, self.layer_idx)
                            result = (attn_output, None, past_kv)
                else:  # warmup: dense main branch + KL alignment only
                    # Reuse layers skip KL loss (no
                    # indexer compute); only Anchor (source) layers align.
                    if self._is_indexer_source:
                        with _td("kl"):
                            kl_loss = idx.compute_kl_loss(
                                hidden_states, q, k_, None, (cos, sin), pos_ids, scaling,
                                restrict_to_selected=False,
                                image_token_mask=itm, image_token_page_ids=ipi,
                                query_token_page_ids=qpi)
                    # with image_bucket_constraint the dense teacher
                    # masks out the image buckets that hold no image token of the query's page.
                    # _build_page_aware_ids。
                    _page_ids = (ipi, qpi) if (idx.image_bucket_constraint
                                               and ipi is not None and qpi is not None) else None
                    result = _attn_forward(page_ids=_page_ids)
            else:
                result = _attn_forward()
            if W is not None and past_kv is not None and q_len > 1:
                # Only record prefill_length the FIRST time (don't overwrite on subsequent q_len>1 calls)
                if not hasattr(past_kv, '_prefill_length'):
                    past_kv._prefill_length = {}
                if self.layer_idx not in past_kv._prefill_length:
                    past_kv._prefill_length[self.layer_idx] = _get_kcache(past_kv, self.layer_idx).shape[-2]
            return result + (kl_loss,)

        # Decode path: first decode step -> record prefill_length (only once!)
        if not hasattr(past_kv, '_prefill_length') or self.layer_idx not in past_kv._prefill_length:
            if not hasattr(past_kv, '_prefill_length'):
                past_kv._prefill_length = {}
            past_kv._prefill_length[self.layer_idx] = _get_kcache(past_kv, self.layer_idx).shape[-2]

        prefill_len = past_kv._prefill_length[self.layer_idx]
        cur_len = _get_kcache(past_kv, self.layer_idx).shape[-2]

        # Warmup: cat-append until ring region is full
        if cur_len < prefill_len + W:
            result = _attn_forward()
            new_len = _get_kcache(past_kv, self.layer_idx).shape[-2]
            if new_len >= prefill_len + W:
                if not hasattr(past_kv, '_ring_pos'):
                    past_kv._ring_pos = {}
                past_kv._ring_pos[self.layer_idx] = 0
            return result + (None,)

        # Steady state: ring in-place overwrite
        if not hasattr(past_kv, '_ring_pos') or self.layer_idx not in past_kv._ring_pos:
            past_kv._ring_pos = getattr(past_kv, '_ring_pos', {}) or {}
            past_kv._ring_pos[self.layer_idx] = 0

        # Ring decode: overwrite ring slots, then attention over full cache
        ring_pos = past_kv._ring_pos[self.layer_idx]
        kcache = _get_kcache(past_kv, self.layer_idx)
        vcache = _get_vcache(past_kv, self.layer_idx)

        # Compute new K, V and apply RoPE, then overwrite ring slots
        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = self._main_rope(value_states, position_ids, past_kv, q_len)
        query_states, key_states = _llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Overwrite ring slots in-place
        for t in range(q_len):
            slot = prefill_len + ring_pos
            kcache[:, :, slot:slot + 1, :] = key_states[:, :, t:t + 1, :]
            vcache[:, :, slot:slot + 1, :] = value_states[:, :, t:t + 1, :]
            ring_pos = (ring_pos + 1) % W
        past_kv._ring_pos[self.layer_idx] = ring_pos

        # Attention over full cache (no causal mask needed for decode q_len=1)
        k = _llama_repeat_kv(kcache, num_kv_groups)
        v = _llama_repeat_kv(vcache, num_kv_groups)
        attn_weights = torch.matmul(query_states, k.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_kv, None


class _GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: normalize in fp32 and scale by (1 + weight). weight init = 0."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class DeepseekV2IndexerRotaryEmbedding(nn.Module):
    """RoPE derived at the indexer's own head_dim instead of slicing the main head's cos/sin.

    Slicing (`cos[..., :d]`) is wrong twice over: the main embedding is `cat(freqs, freqs)`, so
    `rotate_half` — which pairs dim j with j + d/2 — would combine two *different* frequencies
    instead of the same angle (the score stops being a function of the relative distance), and the
    slice keeps only the fastest d/2 frequencies (max wavelength ~2*pi*theta**((d/2-1)/(D/2))),
    leaving no long-range component. Re-deriving inv_freq at `dim` sub-samples the whole spectrum,
    so the wavelength range still reaches ~theta.
    """

    def __init__(self, dim, base, config=None, rope_scaling=None):
        super().__init__()
        self.attention_scaling = 1.0
        _rt = None
        if rope_scaling:
            _rt = rope_scaling.get("rope_type", rope_scaling.get("type"))
        self.rope_type = _rt or "default"
        if self.rope_type == "default":
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        else:
            from types import SimpleNamespace
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            _shim = SimpleNamespace(
                rope_theta=base,
                head_dim=dim,
                hidden_size=dim,
                num_attention_heads=1,
                rope_scaling=dict(rope_scaling),
                max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
            )
            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](
                _shim, torch.device("cpu")
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # force float32
            freqs = (inv_freq.float() @ positions.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV2IndexerRotary2D(nn.Module):
    """2-axis indexer RoPE over (tok_pos, page_pos) -- an mrope with
    ``mrope_section = [n_tok_pairs, page_pairs]`` in FREQUENCY-PAIR space.

    Why two axes: the response text of page k sits thousands of tokens after page k's
    image tokens, and that gap GROWS with the page count, so a single 1D axis has to
    resolve the text->image relation at the far end of its spectrum. Splitting into
    (intra-page offset, page id) makes the token-axis delta a few hundred and independent
    of the document length, and moves "which page" onto a dedicated axis.

    Why the split must happen per frequency PAIR: the embedding is ``cat(ang, ang)`` and
    ``rotate_half`` pairs dim j with j + dim/2, so slicing raw dims would mix the two axes
    inside one rotation. Pairs [0, n_tok) drive the token axis, pairs [n_tok, half) the
    page axis, and both halves of the concat inherit the same assignment.

    Why the page axis gets its own theta instead of reusing the token spectrum's tail:
    with base=1e4 the slowest 8 pairs have omega ~1e-4, i.e. a page distance of 1 rotates
    the phase by 3e-4 rad -- no page discrimination at all. base_page=256 puts the 8 page
    pairs at wavelengths 6.3/12.6/25/50/101/201/402/804 pages (x2 per rung), giving a
    monotone (unambiguous) range of ~402 pages.

    The token axis re-derives inv_freq at its own reduced dim (56 pairs instead of 64) so
    it still SUB-SAMPLES the whole spectrum -- both ends kept, spacing 1.15 -> 1.17.

    ``page_mod > 0`` folds the page-axis coordinate into ``page_pos % page_mod``.
    page_mod=2 with page_pairs=1 is the configuration used here: the page axis keeps 2
    dimensions (one frequency pair) and page indices become a 0/1/0/1/... parity marker,
    so the axis only expresses "same page / adjacent (different) page" instead of linearly
    encoding the page number. The folding happens here only (at the angle computation);
    upstream ``_build_page_rope_pos``, the decode state machine and the pf_cnt gather still
    see the original increasing page numbers, so token-axis bucketing and the incremental
    advance are unaffected.
    """

    def __init__(self, dim, base_tok, base_page, page_pairs, tok_freq_mode="resample",
                 page_mod=0):
        super().__init__()
        half = dim // 2
        page_pairs = int(page_pairs)
        if not (0 < page_pairs < half):
            raise ValueError(
                f"[IndexerRotary2D] page_pairs={page_pairs} must be in (0, {half}); "
                f"dim={dim}")
        n_tok = half - page_pairs
        _ar = torch.arange(0, n_tok, dtype=torch.int64).float()
        if str(tok_freq_mode).lower() == "truncate":
            inv_tok = 1.0 / (base_tok ** (_ar * 2 / dim))
        else:
            inv_tok = 1.0 / (base_tok ** (_ar / n_tok))
        inv_page = 1.0 / (base_page ** (torch.arange(0, page_pairs, dtype=torch.int64).float()
                                       / page_pairs))
        self.register_buffer("inv_tok", inv_tok, persistent=False)
        self.register_buffer("inv_page", inv_page, persistent=False)
        self.dim = dim
        self.n_tok = n_tok
        self.page_pairs = page_pairs
        self.base_tok = base_tok
        self.base_page = base_page
        self.tok_freq_mode = str(tok_freq_mode).lower()
        self.page_mod = int(page_mod or 0)

    @torch.no_grad()
    def forward(self, x, tok_pos, page_pos):
        """tok_pos / page_pos: [B, S] int -> cos/sin [B, S, dim] (same layout as 1D)."""
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # force float32
            t = tok_pos.to(device=x.device).float()[:, :, None]        # [B, S, 1]
            p = page_pos.to(device=x.device)
            if self.page_mod > 0:
                p = torch.remainder(p.to(torch.int64), self.page_mod)
            p = p.float()[:, :, None]
            a_tok = t * self.inv_tok.to(x.device).float()[None, None, :]    # [B, S, n_tok]
            a_page = p * self.inv_page.to(x.device).float()[None, None, :]  # [B, S, page_pairs]
            ang = torch.cat((a_tok, a_page), dim=-1)                   # [B, S, half]
            emb = torch.cat((ang, ang), dim=-1)                        # [B, S, dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Module-level KL / LM-head helpers ported from Qwen3 sparse.
# Pure functions + autograd.Function wrappers so we can recompute in backward
# without retaining the [B,H,C,Sk] fp32 softmax tensors across chunks.
# ---------------------------------------------------------------------------


def _kl_chunk_core(q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk, block_size,
                   scaling, group, kv_group, local_window=None, same_seg=None):
    """KL for one query chunk as a pure function so backward can recompute it
    (see _KLChunkFn). Teacher branch under no_grad; in-place masked_fill_ and a
    broadcast mask (single index head) avoid the extra [B,H,C,Sk] fp32/bool
    copies that OOM the 100-page warmup backward.

    q_chunk      [B, H_idx, C, D]    (student; grads -> W_idx)
    idx_k        [B, H_idx, Sk, D]   (student; grads -> W_idx)
    q_main_chunk [B, H, C, Dh]       (teacher, detached upstream)
    k_main       [B, H_kv, Sk, Dh]   (teacher, detached upstream)
    keep_block   [B, H_idx, C, nblk] bool  (block-level KL candidate mask)
    pos_chunk    [B, C] int          (query position ids for this chunk)
    block_size   int

    the token-level keep mask [B,H_idx,C,Sk] is rebuilt here
    from the block-level candidate + causal + TOKEN-level local exclusion
    instead of being passed in / saved for backward. Retaining keep_block
    ([.., nblk]) rather than keep ([.., Sk]) shrinks the saved tensor
    ~block_size x; the rebuild is O(C*Sk) bool ops (no matmul), negligible vs
    the recompute.

    (9) union semantics: the main branch attends the token-level local window
    [pos - local_window + 1, pos] via the ``sparse_keep = block_keep | local_keep``
    union at build_block_mask time, so the KL teacher/student compete only
    OUTSIDE the local window. Enforced here by masking those token positions
    out of the rebuilt ``keep``. ``local_window`` defaults to ``block_size``
    (= sparse_local_block 1) when not supplied.
    Returns kl [B, H_idx, C] (0 for rows with an empty keep set).
    """
    neg = -1e9
    batch, n_idx_heads, cq, _ = q_chunk.shape
    k_len = k_main.shape[2]
    if local_window is None:
        local_window = block_size

    # --- rebuild token-level keep from block-level candidate + causal + token-level local ---
    k_positions = torch.arange(k_len, device=q_chunk.device)
    token_future = k_positions[None, None, None, :] > pos_chunk[:, None, :, None]  # [B,1,C,Sk]
    # (9) token-level local window [pos - local_window + 1, pos]
    local_start = (pos_chunk - (local_window - 1)).clamp(min=0)  # [B, C]
    local_token = (
        (k_positions[None, None, None, :] >= local_start[:, None, :, None])
        & (k_positions[None, None, None, :] <= pos_chunk[:, None, :, None])
    )  # [B,1,C,Sk]
    keep = keep_block.repeat_interleave(block_size, dim=-1)[..., :k_len]  # [B,H_idx,C,Sk]
    keep = keep & ~token_future & ~local_token
    # drop cross-sample keys. Callers pass GLOBAL query positions in
    # pos_chunk (so token_future/local_token above are global-correct) plus a
    # same-segment mask here; None for the non-packed path (no-op).
    if same_seg is not None:
        keep = keep & same_seg

    # --- student distribution P_idx (differentiable); bf16 matmul, fp32 softmax ---
    s_idx = torch.matmul(q_chunk, idx_k.transpose(-1, -2)).float()  # [B, H_idx, C, Sk]
    s_idx.masked_fill_(~keep, neg)  # in-place, avoid a second fp32 copy
    log_p_idx = F.log_softmax(s_idx, dim=-1)
    del s_idx

    # --- teacher distribution P, averaged within each GQA group, detached ---
    # loop over main heads instead of one [B, H, C, Sk] fp32 matmul.
    # Per-head transient is [B, C, Sk] fp32 while total FLOPs are unchanged.
    # Group-mean is accumulated head by head (mathematically identical to
    # softmax-then-mean over the full tensor).
    with torch.no_grad():
        n_main_heads = q_main_chunk.shape[1]  # H = n_idx_heads * group
        kv_rep = n_main_heads // k_main.shape[1]
        p = torch.zeros((batch, n_idx_heads, cq, k_len),
                        dtype=torch.float32, device=q_chunk.device)
        for h in range(n_main_heads):
            k_h = k_main[:, h // kv_rep]  # [B, Sk, Dh] bf16 view, no repeat_kv copy
            s_h = torch.matmul(q_main_chunk[:, h], k_h.transpose(-1, -2)).float()  # [B, C, Sk]
            s_h.mul_(scaling)
            s_h.masked_fill_(~keep[:, h // group], neg)
            p[:, h // group] += F.softmax(s_h, dim=-1)
            del s_h
        p.div_(group)

    # --- KL(P || P_idx) summed over keys; zero-out non-kept to avoid 0*inf=nan ---
    # in-place mul_/masked_fill_ keep one fp32 [B,H_idx,C,Sk] buffer
    # instead of the extra where/zeros_like copies.
    log_p = torch.log(p.clamp_min(1e-12))
    term = (log_p - log_p_idx).mul_(p)
    term.masked_fill_(~keep, 0)
    return term.sum(dim=-1)  # [B, H_idx, C]


class _KLChunkFn(torch.autograd.Function):
    """recompute-in-backward for one KL chunk. A plain
    autograd.Function instead of torch.utils.checkpoint because swift's
    monkeypatch forces use_reentrant=True under deepspeed and reentrant
    checkpoint cannot nest inside the decoder layer's own checkpoint. Only
    per-chunk INPUTS (small slices + shared idx_k/k_main refs + the block-level
    keep_block mask and pos_chunk) are retained across chunks; the token-level
    [*, Sk] keep and the fp32 score/softmax tensors are rebuilt/live one chunk
    at a time. keep_block ([.., nblk]) is ~block_size x smaller than the
    token-level keep it replaces."""

    @staticmethod
    def forward(ctx, q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk, block_size,
                scaling, group, kv_group, local_window=None, same_seg=None):
        ctx.save_for_backward(q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk)
        ctx.scaling, ctx.group, ctx.kv_group = scaling, group, kv_group
        ctx.block_size = block_size
        ctx.local_window = local_window
        ctx.same_seg = same_seg  # bool [B,1,C,Sk] or None
        return _kl_chunk_core(q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk,
                              block_size, scaling, group, kv_group, local_window, ctx.same_seg)

    @staticmethod
    def backward(ctx, grad_out):
        q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk = ctx.saved_tensors
        q = q_chunk.detach().requires_grad_(True)
        kk = idx_k.detach().requires_grad_(True)
        with torch.enable_grad():
            kl = _kl_chunk_core(q, kk, q_main_chunk, k_main, keep_block, pos_chunk,
                                ctx.block_size, ctx.scaling, ctx.group, ctx.kv_group,
                                ctx.local_window, ctx.same_seg)
        gq, gk = torch.autograd.grad(kl, (q, kk), grad_out)
        return gq, gk, None, None, None, None, None, None, None, None, None, None


def _fused_chunked_lm_ce(
    hidden_states,
    lm_head,
    labels,
    vocab_size,
    num_items_in_batch=None,
    ignore_index=-100,
    chunk_size=1024,
    return_token_acc=False,
):
    """fused chunked linear cross-entropy.

    Computes ``lm_head(hidden)`` + cross-entropy in sequence chunks so the full
    ``[B, seq, vocab]`` logits tensor (and its fp32 upcast) is never materialised.
    Each chunk's logits are recomputed in backward via non-reentrant checkpoint,
    keeping peak activation at O(chunk_size * vocab).

    Numerically matches transformers ``ForCausalLMLoss``: right-pad + shift, then
    ``sum / num_items_in_batch`` when provided, else ``mean`` over non-ignored tokens.
    """
    import torch.utils.checkpoint as _ckpt_mod

    B, L, H = hidden_states.shape
    # Same shift as transformers ForCausalLMLoss: pad labels on the right with
    # ignore_index then drop the first position -> logits[i] predicts labels[i+1].
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    hs = hidden_states.reshape(B * L, H)
    tgt = shift_labels.reshape(B * L)

    N = hs.shape[0]
    if chunk_size <= 0:
        chunk_size = N
    # swift monkeypatches torch.utils.checkpoint.checkpoint to force use_reentrant=True
    # under deepspeed; grab the original so we can request non-reentrant recompute.
    _checkpoint = getattr(_ckpt_mod, '_old_checkpoint', _ckpt_mod.checkpoint)

    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    # optional token-accuracy tracking. Computed per-chunk from the same logits
    # (argmax under no_grad) so it costs no extra lm_head forward and never materialises
    # the full [B, seq, vocab] logits. Kept detached; only `loss` carries grad.
    if return_token_acc:
        total_correct = hidden_states.new_zeros((), dtype=torch.float32)
        total_valid = hidden_states.new_zeros((), dtype=torch.float32)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        h_chunk = hs[s:e]
        t_chunk = tgt[s:e]

        def _compute(h, _t=t_chunk):
            logit = lm_head(h).float()
            ce = F.cross_entropy(logit, _t, ignore_index=ignore_index, reduction='sum')
            if return_token_acc:
                with torch.no_grad():
                    _mask = _t != ignore_index
                    _correct = ((logit.argmax(dim=-1) == _t) & _mask).sum().float()
                    _valid = _mask.sum().float()
                return ce, _correct, _valid
            return ce

        if torch.is_grad_enabled() and h_chunk.requires_grad:
            out_chunk = _checkpoint(_compute, h_chunk, use_reentrant=False)
        else:
            out_chunk = _compute(h_chunk)
        if return_token_acc:
            loss_chunk, correct_chunk, valid_chunk = out_chunk
            total_correct = total_correct + correct_chunk
            total_valid = total_valid + valid_chunk
        else:
            loss_chunk = out_chunk
        total_loss = total_loss + loss_chunk

    if num_items_in_batch is not None:
        loss = total_loss / num_items_in_batch
    else:
        n_valid = (tgt != ignore_index).sum().clamp_min(1)
        loss = total_loss / n_valid
    if return_token_acc:
        # return the raw correct/valid token counts (not the pre-averaged ratio)
        # so the trainer can token-weight the accuracy exactly like swift's native
        # compute_acc path (sum(correct)/sum(valid)) instead of averaging per-step.
        return loss, total_correct, total_valid
    return loss


class DeepseekV2Indexer(nn.Module):
    r"""DeepSeek/MiniMax-M3 style Lightning Indexer (DSA block-sparse attention) for Unlimited-OCR.

    Ported from the InternVL(Qwen3) sparse implementation. Page-aware image bucket constraint
    adapted to the Unlimited-OCR token layout: query pages come from cumsum of a single <PAGE>
    token id. Image pages come from the SAME <PAGE> markers when the prompt carries them
    ("<PAGE><image><PAGE><image>..." data, marker mode); otherwise they fall back to splitting
    the contiguous <image> run into fixed `per_page_image_tokens` blocks (legacy data, where the
    prompt is a bare <image> run and contiguous-segment detection is unusable).
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        sparse_config = getattr(config, "sparse_attention_config", {}) or {}
        self.head_dim = sparse_config.get("sparse_index_dim", 128)
        self.num_heads = sparse_config.get("sparse_num_index_heads", 1)
        self.block_size = sparse_config.get("sparse_block_size", 128)
        self.topk_blocks = sparse_config.get("sparse_topk_blocks", 4)
        self.local_blocks = sparse_config.get("sparse_local_block", 1)
        # token-level local window length, driven by sparse_local_block.
        # `_remote_topk` spends topk_blocks - local_blocks on remote blocks, so the
        # local side must actually cover local_blocks * block_size tokens for the
        # total per-query budget to equal topk_blocks * block_size.
        self.local_blocks = max(1, int(self.local_blocks))
        self.local_window = self.local_blocks * self.block_size

        self.use_kl_loss = sparse_config.get("use_kl_loss", False)
        self.kl_loss_weight = sparse_config.get("kl_loss_weight", 1.0)
        self.kl_chunk_size = sparse_config.get("kl_chunk_size", 1024)
        self.kl_log_interval = sparse_config.get("kl_log_interval", 1)
        self._kl_log_step = 0

        self.training_stage = sparse_config.get("training_stage", "warmup")
        if self.training_stage not in ("warmup", "sparse"):
            print(f"[IndexerStage] WARNING: invalid training_stage={self.training_stage!r}, "
                  f"falling back to 'warmup'", flush=True)
            self.training_stage = "warmup"

        self.image_bucket_constraint = sparse_config.get("image_bucket_constraint", False)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.q_norm = _GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # indexer-native RoPE (dim = index head_dim, same theta as the main branch) — see
        # DeepseekV2IndexerRotaryEmbedding for why slicing the main branch cos/sin is incorrect.
        self.rope_theta = sparse_config.get("sparse_rope_theta", getattr(config, "rope_theta", 10000.0))
        # "1d" == original single-axis RoPE (bit-identical baseline);
        # "page2d" == 2-axis (intra_page_offset, page_id) RoPE, see DeepseekV2IndexerRotary2D.
        self.rope_mode = str(sparse_config.get("sparse_rope_mode", "1d")).lower()
        self.rope_page_pairs = int(sparse_config.get("sparse_rope_page_pairs", 8))
        self.rope_page_theta = float(sparse_config.get("sparse_rope_page_theta", 256.0))
        self.rope_tok_theta = float(sparse_config.get("sparse_rope_tok_theta", self.rope_theta))
        self.rope_page_span = int(sparse_config.get("sparse_rope_page_span", 0))
        self.rope_tok_freq_mode = str(sparse_config.get("sparse_rope_tok_freq_mode", "resample"))
        self.rope_page_mod = int(sparse_config.get("sparse_rope_page_mod", 0))
        self.rotary_emb_1d = DeepseekV2IndexerRotaryEmbedding(
            self.head_dim, self.rope_theta, config=config,
            rope_scaling=sparse_config.get("sparse_rope_scaling",
                                           getattr(config, "rope_scaling", None)))
        self.rotary_emb_2d = None
        if self.rope_mode == "page2d":
            try:
                self.rotary_emb_2d = DeepseekV2IndexerRotary2D(
                    self.head_dim, self.rope_tok_theta, self.rope_page_theta,
                    self.rope_page_pairs, tok_freq_mode=self.rope_tok_freq_mode,
                    page_mod=self.rope_page_mod)
            except ValueError as _e:
                print(f"[PageRope layer={layer_idx}] WARNING: {_e}; falling back to 1d", flush=True)
                self.rope_mode = "1d"
        self.rotary_emb = self.rotary_emb_2d if self.rope_mode == "page2d" else self.rotary_emb_1d
        if layer_idx == 0:
            print(f"[PageRope] mode={self.rope_mode} head_dim={self.head_dim} "
                  f"tok_theta={self.rope_tok_theta} fallback_1d_theta={self.rope_theta} "
                  f"page_pairs={self.rope_page_pairs} page_theta={self.rope_page_theta} "
                  f"page_mod={self.rope_page_mod} "
                  f"tok_freq_mode={self.rope_tok_freq_mode} page_span={self.rope_page_span}",
                  flush=True)
            if self.rotary_emb_2d is not None:
                _r = self.rotary_emb_2d
                _mono = lambda inv: math.pi / float(inv[-1])
                print(f"[PageRope] pairs page/{_r.page_pairs} tok/{_r.n_tok} "
                      f"capacity page={_mono(_r.inv_page):.0f}pages "
                      f"tok={_mono(_r.inv_tok):.0f}tokens", flush=True)
        # (tok_pos, page_pos) for the CURRENT forward, filled by the attention
        # module before the indexer runs; None => fall back to 1D. Not a buffer/parameter.
        self._rope_pos_2d = None
        # 1/sqrt(d) temperature on the index scores. q_norm/k_norm pin |q|=|k|=sqrt(d), so
        # unscaled logits grow with d and the effective temperature changes whenever sparse_index_dim
        # changes; this keeps it dimension-independent and on the same scale as the teacher logits.
        # folded into idx_q in _project_index_qk (equivalent to scaling every q@k, but also
        # covers the fused Triton block-score / KL-recompute paths without touching the kernels).
        self.softmax_scale = self.head_dim ** -0.5

    # token ids / per-page length read lazily from config so runtime add_tokens is picked up.
    @property
    def _sp(self):
        return getattr(self.config, "sparse_attention_config", {}) or {}

    @property
    def image_token_id(self):
        return self._sp.get("image_token_id", 128815)

    @property
    def page_token_id(self):
        return self._sp.get("page_token_id", None)

    @property
    def per_page_image_tokens(self):
        return self._sp.get("per_page_image_tokens", 273)

    def _build_image_block_allow_mask(self, image_token_mask, num_key_blocks):
        B, Sk = image_token_mask.shape
        img_mask_raw = image_token_mask.to(device=self.q_proj.weight.device, dtype=torch.bool)
        pad = num_key_blocks * self.block_size - Sk
        if pad > 0:
            img_mask = F.pad(img_mask_raw, (0, pad), value=False)
        elif pad < 0:
            img_mask = img_mask_raw[..., :num_key_blocks * self.block_size]
        else:
            img_mask = img_mask_raw
        return img_mask.view(B, num_key_blocks, self.block_size).any(dim=-1)

    def _build_page_aware_ids(self, source_input_ids, img_context_token_id, page_token_id, key_length):
        if source_input_ids is None or img_context_token_id is None or page_token_id is None:
            return None, None, None
        if source_input_ids.ndim != 2:
            return None, None, None

        device = self.q_proj.weight.device
        token_ids = source_input_ids[..., :key_length].to(device=device)
        if token_ids.shape[-1] < key_length:
            token_ids = F.pad(token_ids, (0, key_length - token_ids.shape[-1]), value=-1)

        image_token_mask = token_ids == img_context_token_id
        page_starts = token_ids == page_token_id
        cum_pages = torch.cumsum(page_starts.long(), dim=-1)          # 1-based marker count

        # if the PROMPT itself carries <PAGE> before every <image>
        # ("<PAGE><image><PAGE><image>..." data), image pages are read off the SAME markers
        # as the text pages -> no per_page_image_tokens assumption (which only holds for
        # Base mode; Gundam/Gundam-M pages have more visual tokens). Detected per row as
        # "markers seen up to the last image token > 0".
        seq = token_ids.shape[-1]
        ar = torch.arange(seq, device=token_ids.device)
        # last image-token index per row (-1 when the row has no image at all)
        last_img = (image_token_mask.long() * (ar + 1)).max(dim=-1).values - 1        # [B]
        prompt_pages = cum_pages.gather(1, last_img.clamp(min=0)[:, None])            # [B, 1]
        prompt_pages = prompt_pages * (last_img >= 0)[:, None].long()
        marker_mode = prompt_pages > 0                                # [B, 1] bool

        per_page = max(1, int(self.per_page_image_tokens))
        img_running = torch.cumsum(image_token_mask.long(), dim=-1) - 1
        image_token_page_ids = torch.where(marker_mode, cum_pages - 1, img_running // per_page)
        image_token_page_ids = image_token_page_ids.masked_fill(~image_token_mask, -1)

        # The response markers continue the prompt's count, so in marker mode subtract the
        # prompt markers to restart the text page numbering at 0 (image page k <-> text page k).
        raw_page = cum_pages - 1                                      # prompt-side numbering
        resp_page = raw_page - torch.where(
            marker_mode, prompt_pages, torch.zeros_like(prompt_pages))  # response numbering
        # the PREFILL region (everything up to and including the
        # last image token) is page-isolated too, not just the response. There the raw marker
        # count IS the page id, so both the image tokens and the prompt text/markers of page k
        # get page id k -- a page-k prefill token may attend text freely but only the image
        # buckets of page k, never page k-1's image tokens. Before this, image tokens were
        # force-set to -1 and prompt text fell below 0, i.e. every prefill row was "page
        # unknown" and left unconstrained by all three consumers (_query_page_ids -> main attn
        # mask, _build_page_aware_image_block_mask -> KL / sparse candidate set), so pages
        # leaked into each other during prefill. Prompt text AFTER the last image (the
        # instruction) stays -1 on purpose: it is a global query, not a per-page one.
        in_prefill = ar[None, :] <= last_img[:, None]                 # [B, S]
        query_token_page_ids = torch.where(in_prefill, raw_page, resp_page)
        if not _PAGE_SCOPE_PREFILL:
            query_token_page_ids = query_token_page_ids.masked_fill(in_prefill, -1)
        query_token_page_ids = query_token_page_ids.masked_fill(query_token_page_ids < 0, -1)

        if not getattr(self, "_page_aware_diag_logged", False):
            self._page_aware_diag_logged = True
            n_page = int(page_starts.sum().item())
            n_img = int(image_token_mask.sum().item())
            n_pages_q = int(query_token_page_ids.max().item() + 1) if (query_token_page_ids >= 0).any() else 0
            n_img_seg = int(image_token_page_ids.max().item() + 1) if (image_token_page_ids >= 0).any() else 0
            enabled = not (image_token_mask.sum() == 0 or (query_token_page_ids >= 0).sum() == 0)
            _mode = "marker" if bool(marker_mode.any()) else "per_page"
            print(f"[PageAwareKL layer={self.layer_idx}] image_page_mode={_mode} "
                  f"page_token_id={page_token_id} prompt_page_markers={int(prompt_pages.max().item())} "
                  f"per_page_image_tokens={per_page} page_markers={n_page} image_tokens={n_img} "
                  f"query_pages={n_pages_q} image_pages={n_img_seg} page_aware_enabled={enabled}", flush=True)
            # per-page token ranges of row 0. A page-id OFFSET (text page k
            # pointing at the image of page k-1/k+1) shows up here directly: for every k the
            # img[...] range must END BEFORE the txt[...] range STARTS (causality), and the
            # img token count must look like one page of visual tokens (~1135 in Gundam).
            # ok=False on any page means the constraint masks the wrong image.
            # pre[...] is the PREFILL query-page range of page k (image tokens + that page's
            # prompt markers/text, page-scope-prefill); txt[...] is read off a RESPONSE-ONLY
            # view of query_token_page_ids, since prefill rows now carry a page id too and
            # would otherwise swallow the txt range and break the causality check.
            ipi0 = image_token_page_ids[0]
            qpi0 = query_token_page_ids[0].masked_fill(in_prefill[0], -1)
            pre0 = query_token_page_ids[0].masked_fill(~in_prefill[0], -1)
            n_show = min(6, max(int(ipi0.max().item()) + 1, int(qpi0.max().item()) + 1))
            parts = []
            for _k in range(max(0, n_show)):
                im = (ipi0 == _k).nonzero().flatten()
                tm = (qpi0 == _k).nonzero().flatten()
                pr = (pre0 == _k).nonzero().flatten()
                if im.numel() and tm.numel():
                    ok = bool(im[-1].item() < tm[0].item())
                else:
                    ok = False
                im_s = (f"{im[0].item()}-{im[-1].item()}#{im.numel()}" if im.numel() else "none")
                tm_s = (f"{tm[0].item()}-{tm[-1].item()}#{tm.numel()}" if tm.numel() else "none")
                pr_s = (f"{pr[0].item()}-{pr[-1].item()}#{pr.numel()}" if pr.numel() else "none")
                parts.append(f"p{_k}:img[{im_s}] pre[{pr_s}] txt[{tm_s}] ok={ok}")
            mk = page_starts[0].nonzero().flatten()[:12].tolist()
            print(f"[PageAwareKL layer={self.layer_idx} row0] markers@{mk} "
                  + " ".join(parts), flush=True)

        if image_token_mask.sum() == 0 or (query_token_page_ids >= 0).sum() == 0:
            return None, None, None
        return image_token_mask, image_token_page_ids, query_token_page_ids

    def _page_rope_bail(self, reason):
        """one-shot warning + fall back to the 1D indexer RoPE."""
        if not getattr(self, "_page_rope_bail_logged", False):
            self._page_rope_bail_logged = True
            print(f"[PageRope layer={self.layer_idx}] fallback to 1d: {reason}", flush=True)
        return None

    def _build_page_rope_pos(self, source_input_ids, key_length):
        """build (tok_pos, page_pos) for the 2-axis indexer RoPE.

        page_pos: prefill rows take the raw <PAGE> marker count (1..N); response rows take
            ``count - N`` so image page k and text page k share the id. Everything with no
            page (BOS/system prefix, and the instruction that follows the last image) gets
            page 0 -- a RESERVED bucket, real pages start at 1, so Δpage(prompt, any page)
            >= 1 and the instruction can never look like "same page" (Δpage == 0).
        tok_pos: running count inside the token's own page bucket. A running count (not
            ``index - segment_start``) is required because one page owns SEVERAL disjoint
            regions -- page k has its prompt image run plus its response text run, page 0
            has the BOS prefix plus the instruction -- and per-segment restarts would hand
            two different tokens the same (page, tok) coordinate. As a side effect the
            response text of page k automatically starts at that page's prefill length, so
            VARIABLE per-page image token counts need no fixed-page-size assumption.

        Returns (tok_pos, page_pos) as [B, key_length] long tensors, or None to fall back.
        """
        if source_input_ids is None or getattr(source_input_ids, "ndim", 0) != 2:
            return self._page_rope_bail("no source_input_ids")
        if self.image_token_id is None or self.page_token_id is None:
            return self._page_rope_bail("image_token_id / page_token_id unavailable")

        device = self.q_proj.weight.device
        ids = source_input_ids[..., :key_length].to(device=device)
        if ids.shape[-1] < key_length:
            ids = F.pad(ids, (0, key_length - ids.shape[-1]), value=-1)
        B, S = ids.shape
        ar = torch.arange(S, device=device)
        img = ids == self.image_token_id
        mk = ids == self.page_token_id
        if not bool(img.any()) or not bool(mk.any()):
            return self._page_rope_bail("row has no image token / no <PAGE> marker")

        cum = torch.cumsum(mk.long(), dim=-1)                          # 1-based marker count
        last_img = (img.long() * (ar + 1)).max(dim=-1).values - 1       # [B]
        n_pages = cum.gather(1, last_img.clamp(min=0)[:, None])         # markers up to last image
        n_pages = n_pages * (last_img >= 0)[:, None].long()             # [B, 1]
        if not bool((n_pages > 0).all()):
            # non-marker data: _build_page_aware_ids would bucket images by
            # per_page_image_tokens (273), which is 4-7x wrong for Gundam pages -> bail.
            return self._page_rope_bail("non-marker mode (prompt carries no <PAGE>)")

        in_prefill = ar[None, :] <= last_img[:, None]                   # [B, S]
        page = torch.where(in_prefill, cum, cum - n_pages).clamp(min=0)  # 0 = prompt/no-page
        return self._page_rope_offsets(page, in_prefill, img, B, S)

    def _page_rope_offsets(self, page, in_prefill, img, B, S):
        """running-count offsets inside each page bucket (+ optional span
        normalisation), then the one-shot [PageRope] diagnostic. Returns (tok_pos, page_pos)."""
        n_pg = int(page.max().item()) + 1
        onehot = F.one_hot(page, num_classes=n_pg).to(torch.int32)      # [B, S, n_pg]
        run = onehot.cumsum(dim=1)                                     # inclusive running count
        tok = run.gather(2, page[:, :, None].to(torch.int64)).squeeze(-1).to(torch.int64) - 1

        pf_cnt = (onehot * in_prefill[:, :, None].to(torch.int32)).sum(dim=1)   # [B, n_pg]
        pf_k = pf_cnt.gather(1, page.reshape(B, S).to(torch.int64)).to(torch.int64)
        self._page_rope_meta = {'pf_cnt': pf_cnt.to(torch.int64), 'n_pg': n_pg}
        if self.rope_page_span > 0:
            # ablation: map each page's prefill span onto a fixed length S_span so the
            # token-axis delta no longer depends on how many visual tokens the page has.
            s_span = float(self.rope_page_span)
            pre = torch.round(tok.float() * s_span / pf_k.clamp(min=1).float())
            post = s_span + (tok - pf_k).float()
            scaled = torch.where(in_prefill, pre, post).to(torch.int64)
            tok = torch.where(page > 0, scaled, tok)                   # page 0 stays raw

        if not getattr(self, "_page_rope_diag_logged", False):
            self._page_rope_diag_logged = True
            p0, t0, pre0, img0 = page[0], tok[0], in_prefill[0], img[0]
            parts = []
            for _k in range(1, min(n_pg, 5)):
                im = ((p0 == _k) & img0).nonzero().flatten()
                tx = ((p0 == _k) & ~pre0).nonzero().flatten()
                if im.numel() and tx.numel():
                    d = int(t0[tx[0]].item() - t0[im[-1]].item())
                    parts.append(
                        f"p{_k}:img[{int(t0[im[0]])}-{int(t0[im[-1]])}#{im.numel()}] "
                        f"txt[{int(t0[tx[0]])}-{int(t0[tx[-1]])}#{tx.numel()}] "
                        f"dtok={d} ok={d > 0}")
                else:
                    parts.append(f"p{_k}:img#{im.numel()} txt#{tx.numel()} ok=False")
            pr = (p0 == 0).nonzero().flatten()
            pr_s = (f"{int(t0[pr[0]])}-{int(t0[pr[-1]])}#{pr.numel()}" if pr.numel() else "none")
            uniq = int(torch.unique(p0 * (int(t0.max().item()) + 1) + t0).numel())
            mono = int(self.rotary_emb_2d.base_page ** (
                (self.rotary_emb_2d.page_pairs - 1) / self.rotary_emb_2d.page_pairs) * math.pi)
            _pm = int(getattr(self.rotary_emb_2d, "page_mod", 0) or 0)
            if _pm > 0:
                _fold = [int(_k % _pm) for _k in range(0, min(n_pg, 9))]
                mono_s = f"page_mod={_pm} folded_pages={_fold}"
            else:
                mono_s = f"page_mono_range={mono} extrapolating={n_pg - 1 > mono}"
            print(f"[PageRope layer={self.layer_idx} row0] n_pages={n_pg - 1} "
                  f"{mono_s} "
                  f"prompt(page0)[{pr_s}] uniq_coords={uniq}/{S} " + " ".join(parts), flush=True)
        return tok, page

    def _build_flat_image_token_mask(self, source_input_ids, img_context_token_id, key_length):
        """image_bucket_constraint=False path: token-level image mask over ALL pages.

        Only needs the image token id (no page_token_id / per-page bucketing), so the
        remote top-k candidate set becomes "all image blocks (any page)" rather than
        "all blocks". Returns [B, key_length] bool, or None if unavailable/no image.
        """
        if source_input_ids is None or img_context_token_id is None:
            return None
        if source_input_ids.ndim != 2:
            return None
        device = self.q_proj.weight.device
        token_ids = source_input_ids[..., :key_length].to(device=device)
        if token_ids.shape[-1] < key_length:
            token_ids = F.pad(token_ids, (0, key_length - token_ids.shape[-1]), value=-1)
        mask = token_ids == img_context_token_id
        if mask.sum() == 0:
            return None
        return mask

    def _build_page_aware_image_block_mask(self, image_token_page_ids, query_token_page_ids,
                                           pos_chunk, num_key_blocks, cq):
        batch = pos_chunk.shape[0]
        k_len = image_token_page_ids.shape[-1]
        pos_clamped = pos_chunk.clamp(0, query_token_page_ids.shape[-1] - 1)
        query_pages = query_token_page_ids.gather(1, pos_clamped)

        pad = num_key_blocks * self.block_size - k_len
        if pad > 0:
            img_page_padded = F.pad(image_token_page_ids.float(), (0, pad), value=-1).long()
        elif pad < 0:
            img_page_padded = image_token_page_ids[..., :num_key_blocks * self.block_size]
        else:
            img_page_padded = image_token_page_ids
        img_page_blocks = img_page_padded.view(batch, num_key_blocks, self.block_size)

        qp = query_pages[:, :, None, None]
        kp = img_page_blocks[:, None, :, :]
        same_page = (qp >= 0) & (qp == kp)
        block_has_same_page = same_page.any(dim=-1)
        return block_has_same_page.unsqueeze(1)

    def _query_page_ids(self, image_token_page_ids, query_token_page_ids, q_positions):
        """page id of each query token, indexed by its GLOBAL
        (flat) token position. Text queries take the id of their `<PAGE>` marker.

        An IMAGE token as query stays -1 ("page unknown") on purpose: image queries are
        left UNCONSTRAINED, i.e. they may attend every earlier position regardless of
        page. This matches the KL / selection side, where `query_token_page_ids` is -1 at
        image tokens and `_build_kl_candidate_mask`'s per-query fallback opens the domain
        to all blocks -- so all three paths agree for image queries."""
        seq_len = query_token_page_ids.shape[-1]
        qp_idx = q_positions.clamp(0, seq_len - 1)
        return query_token_page_ids.gather(1, qp_idx)

    @torch.no_grad()
    def apply_page_image_attn_mask(self, keep, image_token_page_ids, query_token_page_ids,
                                   q_positions, chunk=2048):
        """AND "no other-page image BUCKET" into the WARMUP dense
        main-branch bool keep mask ``[B, 1, Sq, Sk]`` (True == attend), in place.

        BUCKET (block) granularity, i.e. the same domain the sparse stage selects over:
        a key block is dropped only if it is an image block that holds NO image token of
        the query's own page. Consequences of the block granularity: text blocks stay
        visible, and the (at most two) boundary blocks of a page are kept whole, so up to
        ``block_size - 1`` neighbouring-page image tokens survive. That is deliberate —
        the sparse main branch gathers whole blocks too, so teacher and student see the
        same buckets. Queries with no page (IMAGE tokens, and text before the first
        `<PAGE>` marker) are left untouched: they may attend every earlier position.

        Done in query chunks so no second ``[B, 1, Sq, Sk]`` tensor is materialised
        (~1 GiB at 32k).
        """
        bs = self.block_size
        k_len = keep.shape[-1]
        nblk = -(-k_len // bs)
        kp = image_token_page_ids[..., :k_len]
        pad = nblk * bs - k_len
        if pad > 0:
            kp = F.pad(kp, (0, pad), value=-1)
        kp_blk = kp.view(kp.shape[0], nblk, bs)                     # [B, nblk, bs]
        img_blk = (kp_blk >= 0).any(dim=-1)                         # [B, nblk]
        qp = self._query_page_ids(image_token_page_ids, query_token_page_ids, q_positions)
        q_len = keep.shape[-2]
        for c0 in range(0, q_len, chunk):
            c1 = min(c0 + chunk, q_len)
            qp_c = qp[:, c0:c1]                                     # [B, C]
            # block holds an image token of this query's page
            same_blk = (kp_blk[:, None, :, :] == qp_c[:, :, None, None]).any(dim=-1)
            # per-query fallback: a query that sees no own-page image bucket at all is
            # left unconstrained (mirrors _build_kl_candidate_mask's fallback).
            no_page = ~same_blk.any(dim=-1, keepdim=True)
            allow_blk = ((~img_blk[:, None, :]) | same_blk
                         | (qp_c < 0)[:, :, None] | no_page)
            keep[:, :, c0:c1] &= allow_blk[:, None].repeat_interleave(bs, dim=-1)[..., :k_len]
        return keep

    def _build_kl_candidate_mask(self, pos_chunk, num_key_blocks, image_token_mask, batch, cq,
                                 image_token_page_ids=None, query_token_page_ids=None):
        """TOKEN-level union semantics — the local window
        is excluded at the token level in the callers (via _build_local_token_mask),
        so this candidate mask must NOT drop partial-overlap blocks based on locality.

        Remote top-k candidates are ALWAYS restricted to image blocks; the flag only
        controls the page granularity:
        - image_bucket_constraint=True  + page-aware ids: same-page (current page) image blocks.
        - image_bucket_constraint=True  + only flat image_token_mask: all image blocks (fallback).
        - image_bucket_constraint=False + image_token_mask: all image blocks (any page).
        - no image_token_mask available (text-only / missing src ids): all blocks.
        A per-query fallback to all blocks applies whenever a query sees no image candidate.
        """
        device = pos_chunk.device
        if self.image_bucket_constraint and image_token_page_ids is not None \
                and query_token_page_ids is not None:
            # true: restrict to same-page image blocks
            candidate = self._build_page_aware_image_block_mask(
                image_token_page_ids, query_token_page_ids, pos_chunk, num_key_blocks, cq)
        elif image_token_mask is not None:
            # false (any-page image), or true without page ids: all image blocks
            img_blk = self._build_image_block_allow_mask(image_token_mask, num_key_blocks)
            candidate = img_blk[:, None, None, :].expand(batch, 1, cq, num_key_blocks).contiguous()
        else:
            # no image info available: all blocks are candidates
            candidate = torch.ones(batch, 1, cq, num_key_blocks, dtype=torch.bool, device=device)
        # per-query fallback: if no candidate visible, fall back to all blocks
        has_candidate = candidate.any(dim=-1, keepdim=True)
        fallback = torch.ones_like(candidate)
        candidate = torch.where(has_candidate, candidate, fallback)
        return candidate

    def _build_local_token_mask(self, position_ids, k_positions):
        """return [B, 1, Sq, Sk] for the inclusive local window
        [q - local_window + 1, q], with local_window = sparse_local_block * block_size."""
        local_start = (position_ids - (self.local_window - 1)).clamp(min=0)
        key_pos = k_positions[None, None, None, :]
        return (key_pos >= local_start[:, None, :, None]) & (
            key_pos <= position_ids[:, None, :, None]
        )

    def _remote_topk(self, num_key_blocks):
        """remote budget = topk_blocks - local_blocks.
        The local coverage is added via the sparse_keep = block_keep | local_keep
        union in build_block_mask, not by boosting local blocks in top-k."""
        return min(max(self.topk_blocks - self.local_blocks, 0), num_key_blocks)

    def _diag_due(self, attr):
        """layer-2 only, every kl_log_interval calls. SPARSE_DIAG_LOG=0 disables."""
        import os as _os_d
        if _os_d.environ.get("SPARSE_DIAG_LOG", "1") != "1" or self.layer_idx != 2:
            return False
        step = getattr(self, attr, 0) + 1
        setattr(self, attr, step)
        return step % max(1, self.kl_log_interval) == 0

    @torch.no_grad()
    def _log_image_mask(self, image_token_mask, source_input_ids, key_length):
        """is the image token mask actually populated? Prints why it is
        empty (missing src ids / no image token hit) and how many blocks it allows."""
        if not self._diag_due("_itm_log_step"):
            return
        if image_token_mask is None:
            if source_input_ids is None:
                why = "source_input_ids=None (not injected by the model wrapper)"
            elif self.image_token_id is None:
                why = "image_token_id=None in sparse_attention_config"
            else:
                hit = int((source_input_ids == self.image_token_id).sum().item())
                why = (f"builder returned None: src_shape={tuple(source_input_ids.shape)} "
                       f"image_token_id={self.image_token_id} hits={hit}")
            print(f"[ImageMask layer={self.layer_idx}] itm=None -> remote candidates = ALL "
                  f"blocks. reason: {why}", flush=True)
            return
        num_key_blocks = -(-key_length // self.block_size)
        n_img = int(image_token_mask.sum().item())
        n_tok = image_token_mask.numel()
        img_blk = self._build_image_block_allow_mask(image_token_mask, num_key_blocks)
        n_blk = img_blk.float().sum(dim=-1).mean().item()
        print(f"[ImageMask layer={self.layer_idx}] itm ok shape={tuple(image_token_mask.shape)} "
              f"image_token_id={self.image_token_id} "
              f"image_tokens={n_img}/{n_tok} ({100.0 * n_img / max(1, n_tok):.1f}%) "
              f"image_blocks={n_blk:.1f}/{num_key_blocks} "
              f"({100.0 * n_blk / max(1, num_key_blocks):.1f}%) "
              f"empty={n_img == 0}", flush=True)

    @torch.no_grad()
    def _log_token_coverage(self, block_indices, position_ids, key_length):
        """how many keys each query actually attends = local window
        [pos-local_window+1, pos] UNION the selected remote blocks (clipped to the
        causal / non-local range, matching the gather kernel). Sampled over queries."""
        if not self._diag_due("_cov_log_step"):
            return
        if block_indices is None or block_indices.shape[-1] == 0:
            print(f"[TokenCoverage layer={self.layer_idx}] no remote blocks selected "
                  f"(topk=0); local window only: {self.local_window} tokens", flush=True)
            return
        bi = block_indices[:, 0]                       # [B, Sq, topk]
        stride = max(1, bi.shape[1] // 64)
        cols = torch.arange(0, bi.shape[1], stride, device=bi.device)
        bi_s = bi[:, cols]                             # [B, C, topk]
        pos_s = position_ids[:, cols].to(bi.device)    # [B, C]

        ls = (pos_s - (self.local_window - 1)).clamp(min=0)
        local_cnt = (pos_s - ls + 1).float()           # [B, C]

        valid = bi_s >= 0
        start = bi_s.clamp(min=0) * self.block_size
        end = (start + self.block_size).clamp(max=key_length)
        # remote keys are clipped to < local_start (kernel: `key < ls`)
        remote_cnt = (torch.minimum(end, ls.unsqueeze(-1)) - start).clamp(min=0)
        remote_cnt = remote_cnt.masked_fill(~valid, 0).sum(-1).float()

        n_valid = valid.sum(-1).float()
        srt, _ = bi_s.masked_fill(~valid, -1).sort(dim=-1)
        dup = ((srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)).sum(-1).float()

        budget = self.topk_blocks * self.block_size
        total = local_cnt + remote_cnt
        print(f"[TokenCoverage layer={self.layer_idx}] k_len={key_length} "
              f"local_blocks={self.local_blocks} local_window={self.local_window} "
              f"remote_topk={bi.shape[-1]} "
              f"valid_blocks/q={n_valid.mean().item():.2f} "
              f"dup_blocks/q={dup.mean().item():.2f} "
              f"local_tok/q={local_cnt.mean().item():.1f} "
              f"remote_tok/q={remote_cnt.mean().item():.1f} "
              f"total_tok/q={total.mean().item():.1f} (budget={budget}) "
              f"min={total.min().item():.0f} max={total.max().item():.0f}", flush=True)


    @torch.no_grad()
    def _log_indexer_recall(self, idx_q, idx_k, q_main, k_main, position_ids, scaling,
                            num_key_blocks, image_token_mask, image_token_page_ids,
                            query_token_page_ids, stage, kl_avg):
        """paper-style indexer alignment metrics (DSA report Fig.3).

        I* = Top-k blocks induced by the MAIN branch scores, I_hat = the Index Branch
        Top-k selection, both over the SAME domain the selector uses (candidate blocks,
        causal, token-level local window excluded):

            block recall = |I* ^ I_hat| / |I*|
            score recall = sum_{b in I* ^ I_hat} P_b / sum_{b in I*} P_b

        P_b is the main-branch attention probability summed over the tokens of block b,
        averaged within each GQA group onto the index heads. Sampled over at most
        RECALL_LOG_SAMPLE_Q query rows under no_grad -> pure diagnostic, no grad/loss
        impact.

        Every metric is reported twice, split by row population, because the page
        constraint makes them incomparable otherwise:
          page[...]  query_token_page_ids >= 0 -- answer text of a known page, candidates
                     are that page's image blocks (~48 blocks at 1513 image tokens/page).
          free[...]  fallback rows (image tokens, prompt text before the first answer
                     <PAGE>): _build_kl_candidate_mask opens their candidates to ALL
                     causal blocks, so cand/q is in the hundreds.
        Chance level for student_kept / recall / score is ~ topk / cand_q of the SAME group,
        so always read a number against the cand/q printed next to it. Every metric is
        averaged over CONTESTED rows only (cand > topk; otherwise top-k trivially keeps
        everything).
        RECALL_LOG=0 disables; RECALL_LOG_ALL_RANKS=1 prints on every rank.
        """
        import os as _os_r
        if _os_r.environ.get("RECALL_LOG", "1") != "1":
            return
        # rank0 only by default -- 8 ranks x 12 layers = ~100 lines per log
        # event drowns the log, and every rank sees a different sample anyway.
        if (_os_r.environ.get("RECALL_LOG_ALL_RANKS", "0") != "1"
                and dist.is_available() and dist.is_initialized() and dist.get_rank() != 0):
            return
        topk = self._remote_topk(num_key_blocks)
        if topk == 0:
            return

        batch, n_idx_heads, q_len, _ = idx_q.shape
        k_len = idx_k.shape[2]
        device = idx_q.device
        n_sample = max(1, int(_os_r.environ.get("RECALL_LOG_SAMPLE_Q", "64")))
        sel = torch.arange(0, q_len, max(1, q_len // n_sample), device=device)[:n_sample]
        cq = sel.numel()
        k_positions = torch.arange(k_len, device=device)

        if _pk.is_packed(position_ids):
            pm = _pk.packed_masks(position_ids, k_positions, sel, self.local_window)
            pos_chunk, token_future = pm['gpos'], pm['token_future']
            local_keep, same_seg = pm['local_token'], pm['same_seg']
        else:
            pos_chunk = position_ids[:, sel]
            token_future = k_positions[None, None, None, :] > pos_chunk[:, None, :, None]
            local_keep = self._build_local_token_mask(pos_chunk, k_positions)
            same_seg = None

        candidate = self._build_kl_candidate_mask(
            pos_chunk, num_key_blocks, image_token_mask, batch, cq,
            image_token_page_ids=image_token_page_ids,
            query_token_page_ids=query_token_page_ids)          # [B, 1, C, nblk]
        keep = candidate.repeat_interleave(self.block_size, dim=-1)[..., :k_len]
        keep = keep & ~token_future & ~local_keep
        if same_seg is not None:
            keep = keep & same_seg
        keep = keep[:, 0]                                       # [B, C, Sk]
        if not keep.any():
            return

        # split every metric into the two row populations the page
        # constraint creates -- `page` rows (query_token_page_ids >= 0, i.e. answer text
        # of a known page; candidates == that page's image blocks) and `free` rows (image
        # tokens and the prompt text before the first answer <PAGE>; _build_kl_candidate_mask
        # opens their candidate set to ALL causal blocks). Mixing them makes cand_blocks/q
        # and student_kept uninterpretable, since the two populations differ by ~5x in
        # candidate count.
        if query_token_page_ids is not None:
            _qidx = sel.clamp(0, query_token_page_ids.shape[-1] - 1)[None].expand(batch, -1)
            page_row = query_token_page_ids.gather(1, _qidx) >= 0    # [B, C]
        else:
            page_row = torch.zeros((batch, cq), dtype=torch.bool, device=device)
        pad = num_key_blocks * self.block_size - k_len

        def _blk(x, reduce):                                    # [B,C,Sk] -> [B,C,nblk]
            if pad:
                x = F.pad(x, (0, pad), value=0.0 if reduce == "sum" else float("-inf"))
            x = x.view(batch, cq, num_key_blocks, self.block_size)
            return x.sum(dim=-1) if reduce == "sum" else x.amax(dim=-1)

        valid_blk = _blk(keep.float(), "sum") > 0               # [B, C, nblk]
        n_valid_blk = valid_blk.sum(dim=-1)

        # --- teacher: P_b from the dense main branch, GQA-group averaged ---
        num_heads = q_main.shape[1]
        group = num_heads // n_idx_heads
        kv_group = num_heads // k_main.shape[1]
        q_main_sel = q_main[:, :, sel]
        p_b = torch.zeros((batch, n_idx_heads, cq, num_key_blocks),
                          dtype=torch.float32, device=device)
        for h in range(num_heads):
            s_h = torch.matmul(q_main_sel[:, h],
                               k_main[:, h // kv_group].transpose(-1, -2)).float()
            s_h.mul_(scaling).masked_fill_(~keep, -1e9)
            p_b[:, h // group] += _blk(F.softmax(s_h, dim=-1), "sum")
            del s_h
        p_b.div_(group)
        # every metric is accumulated TWICE, once per row population
        # (`page` = page-constrained answer text, `free` = fallback rows), because the two
        # differ by ~5x in candidate count and mixing them hides both signals:
        #   top1_hit   = teacher's best block == indexer's best block.
        # rows == contested。
        # Chance level for all of them is roughly topk / cand_blocks -- always compare
        # against the per-population cand/q printed on the same line.
        z = p_b.new_zeros
        acc = {k: z(()) for k in ("mass_p", "tea_p", "t1_p", "n_p", "br_p", "sr_p", "nc_p",
                                  "mass_f", "tea_f", "t1_f", "n_f", "br_f", "sr_f", "nc_f")}
        m_page = page_row.float()
        m_free = (~page_row).float()
        for h in range(n_idx_heads):
            # index-branch block score = block-max of the indexer logits (same as forward())
            s_idx = torch.matmul(idx_q[:, h, sel].float(),
                                 idx_k[:, 0].float().transpose(-1, -2))
            s_blk = _blk(s_idx.masked_fill(~keep, float("-inf")), "amax")
            del s_idx
            p_h = p_b[:, h]
            p_masked = p_h.masked_fill(~valid_blk, -1.0)
            s_masked = s_blk.masked_fill(~valid_blk, float("-inf"))
            star = p_masked.topk(topk, dim=-1).indices
            hat = s_masked.topk(topk, dim=-1).indices
            # & valid_blk: topk over fewer than k valid entries also returns invalid slots
            star_m = torch.zeros_like(valid_blk).scatter_(-1, star, True) & valid_blk
            hat_m = torch.zeros_like(valid_blk).scatter_(-1, hat, True) & valid_blk

            mass_all = (p_h * valid_blk).sum(dim=-1)
            mass_star = (p_h * star_m).sum(dim=-1)
            row_ok = ((n_valid_blk > topk) & (mass_star > 0)).float()
            kept_frac = (p_h * hat_m).sum(dim=-1) / mass_all.clamp_min(1e-9)
            teach_frac = mass_star / mass_all.clamp_min(1e-9)
            hit = (p_masked.argmax(dim=-1) == s_masked.argmax(dim=-1)).float()
            for sfx, m in (("p", m_page), ("f", m_free)):
                w = row_ok * m
                acc["mass_" + sfx] = acc["mass_" + sfx] + (kept_frac * w).sum()
                acc["tea_" + sfx] = acc["tea_" + sfx] + (teach_frac * w).sum()
                acc["t1_" + sfx] = acc["t1_" + sfx] + (hit * w).sum()
                acc["n_" + sfx] = acc["n_" + sfx] + w.sum()

            inter_m = star_m & hat_m
            br = inter_m.sum(dim=-1).float() / topk
            sr = (p_h * inter_m).sum(dim=-1) / mass_star.clamp_min(1e-9)
            for sfx, m in (("p", m_page), ("f", m_free)):
                w = row_ok * m
                acc["br_" + sfx] = acc["br_" + sfx] + (br * w).sum()
                acc["sr_" + sfx] = acc["sr_" + sfx] + (sr * w).sum()
                acc["nc_" + sfx] = acc["nc_" + sfx] + w.sum()

        n_page = int(page_row.sum().item())
        n_free = batch * cq - n_page
        cand_f = n_valid_blk.float()

        def _grp(sfx, n_rows_pop, row_mask):
            n = acc["n_" + sfx].item()
            nc = acc["nc_" + sfx].item()
            if n == 0:
                return f"{'page' if sfx == 'p' else 'free'}[rows=0]"
            cq_mean = (cand_f * row_mask).sum().item() / max(1, n_rows_pop)
            rc = (f"recall@{topk}={acc['br_' + sfx].item() / nc:.4f} "
                  f"score@{topk}={acc['sr_' + sfx].item() / nc:.4f}"
                  if nc > 0 else f"recall@{topk}=n/a score@{topk}=n/a")
            return (f"{'page' if sfx == 'p' else 'free'}["
                    f"cand/q={cq_mean:.1f} "
                    f"student_kept={acc['mass_' + sfx].item() / n:.4f} "
                    f"teacher_kept={acc['tea_' + sfx].item() / n:.4f} "
                    f"top1={acc['t1_' + sfx].item() / n:.4f} {rc} "
                    f"rows={int(n)} contested={int(nc)}]")

        if (acc["n_p"] + acc["n_f"]).item() == 0:
            return
        if query_token_page_ids is not None:
            _pages_per_sample = (query_token_page_ids.amax(dim=-1) + 1).clamp(min=0).float()
            _n_pages_info = (f"n_pages[mean={_pages_per_sample.mean().item():.1f} "
                            f"max={int(_pages_per_sample.max().item())} "
                            f"min={int(_pages_per_sample.min().item())}] ")
        else:
            _n_pages_info = "n_pages[n/a] "
        print(
            f"[IndexerRecall layer={self.layer_idx} step={self._kl_log_step}] stage={stage} "
            f"kl_avg={kl_avg:.4f} topk={topk} nblk={num_key_blocks} "
            f"block_size={self.block_size} {_n_pages_info}"
            f"{_grp('p', n_page, m_page)} {_grp('f', n_free, m_free)} "
            f"all[cand/q={n_valid_blk.float().mean().item():.1f} "
            f"rows={int((acc['n_p'] + acc['n_f']).item())}/{batch * n_idx_heads * cq}]",
            flush=True,
        )
    def _project_index_qk(self, hidden_states, position_ids, past_key_values):
        batch, q_len, _ = hidden_states.shape
        idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
        idx_q = self.q_norm(idx_q).transpose(1, 2)
        idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
        idx_k = self.k_norm(idx_k).transpose(1, 2)
        # cos/sin built at the index head_dim, not sliced from the main branch
        _p2d = self._rope_pos_2d if self.rope_mode == "page2d" else None
        if _p2d is not None and q_len > 1 and _p2d[0].shape[-1] >= q_len:
            _off = _p2d[0].shape[-1] - q_len
            cos, sin = self.rotary_emb_2d(idx_q, _p2d[0][:, _off:], _p2d[1][:, _off:])
        else:
            if _p2d is not None and q_len > 1:
                self._page_rope_bail(f"coord len {_p2d[0].shape[-1]} < q_len {q_len}")
            cos, sin = self.rotary_emb_1d(idx_q, position_ids)
        idx_q, idx_k = _llama_apply_rotary_pos_emb(idx_q, idx_k, cos, sin)
        # fold the 1/sqrt(d) score temperature into q (commutes with RoPE, which is a
        # rotation) so every downstream q@k -- eager, fused Triton, KL recompute -- is scaled.
        idx_q = idx_q * self.softmax_scale

        if past_key_values is not None:
            if not hasattr(past_key_values, "_idx_keys"):
                past_key_values._idx_keys = {}
            if self.layer_idx not in past_key_values._idx_keys:
                past_key_values._idx_keys[self.layer_idx] = idx_k
            else:
                past_key_values._idx_keys[self.layer_idx] = torch.cat(
                    [past_key_values._idx_keys[self.layer_idx], idx_k], dim=2)
            idx_k = past_key_values._idx_keys[self.layer_idx]
        return idx_q, idx_k

    def forward(self, hidden_states, position_embeddings, past_key_values, position_ids,
                image_token_mask=None):
        """produce remote top-K block indices per query token.
        Local window is handled at TOKEN level in build_block_mask (union), not here."""
        batch, q_len, _ = hidden_states.shape
        idx_q, idx_k = self._project_index_qk(hidden_states, position_ids, past_key_values)

        k_len = idx_k.shape[2]
        num_key_blocks = -(-k_len // self.block_size)
        idx_k_f = idx_k.float()
        k_positions = torch.arange(k_len, device=idx_q.device)

        chunk = self.kl_chunk_size if self.kl_chunk_size and self.kl_chunk_size > 0 else q_len
        block_scores_chunks = []
        for c0 in range(0, q_len, chunk):
            c1 = min(c0 + chunk, q_len)
            q_chunk = idx_q[:, :, c0:c1].float()
            scores = torch.matmul(q_chunk, idx_k_f.transpose(-1, -2))
            pos_chunk = position_ids[:, c0:c1]
            token_future = k_positions[None, None, None, :] > pos_chunk[:, None, :, None]
            # TOKEN-level local window exclusion — remote blocks
            # only score keys strictly below the local window [q - local_window + 1, q].
            local_token = self._build_local_token_mask(pos_chunk, k_positions)
            scores = scores.masked_fill(token_future | local_token, float("-inf"))
            pad = num_key_blocks * self.block_size - k_len
            if pad:
                scores = F.pad(scores, (0, pad), value=float("-inf"))
            scores = scores.view(batch, self.num_heads, c1 - c0, num_key_blocks, self.block_size)
            block_scores_chunks.append(scores.amax(dim=-1))
        block_scores = torch.cat(block_scores_chunks, dim=2)

        if self.image_bucket_constraint and image_token_mask is not None:
            img_blk = self._build_image_block_allow_mask(image_token_mask, num_key_blocks)
            block_scores = block_scores.masked_fill(~img_blk[:, None, None, :], float("-inf"))

        topk = self._remote_topk(num_key_blocks)
        if topk == 0:
            return torch.empty(
                batch, self.num_heads, q_len, 0, dtype=torch.long, device=block_scores.device)
        topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
        return topk_indices.masked_fill(topk_scores == float("-inf"), -1)

    def compute_kl_loss(self, hidden_states, query_states, key_states, block_indices,
                        position_embeddings, position_ids, scaling, restrict_to_selected=True,
                        image_token_mask=None, image_token_page_ids=None, query_token_page_ids=None):
        """KL alignment between indexer (student) and
        main-branch softmax (teacher). Uses _KLChunkFn recompute-in-backward + optional
        KL_QUERY_STRIDE subsampling to keep memory / compute bounded on long sequences.

          - restrict_to_selected=True  (sparse stage): keep_block = selected non-local
                                                       blocks ∩ candidate
          - restrict_to_selected=False (warmup stage): keep_block = candidate (all KL-eligible)
        The TOKEN-level local window is excluded inside ``_kl_chunk_core`` — the main
        branch attends it via build_block_mask's block_keep|local_keep union.
        """
        import os as _os
        batch, q_len, _ = hidden_states.shape
        idx_q, idx_k = self._project_index_qk(hidden_states.detach(), position_ids, None)

        q_main = query_states.detach()
        num_heads = q_main.shape[1]
        # keep k_main at KV-head granularity (no repeat_kv copy)
        k_main = key_states.detach()
        n_kv_heads = k_main.shape[1]
        n_idx_heads = self.num_heads
        group = num_heads // n_idx_heads
        kv_group = num_heads // n_kv_heads

        k_len = idx_k.shape[2]
        num_key_blocks = -(-k_len // self.block_size)
        k_positions = torch.arange(k_len, device=idx_q.device)

        chunk = self.kl_chunk_size if self.kl_chunk_size and self.kl_chunk_size > 0 else q_len
        total_kl = idx_q.new_zeros((), dtype=torch.float32)
        total_cnt = idx_q.new_zeros((), dtype=torch.float32)
        # KL_QUERY_STRIDE subsamples query rows; KL_CHUNK_RECOMPUTE
        # runs each chunk under _KLChunkFn (recompute-in-backward) so [B,H,C,Sk] fp32
        # softmax tensors are not retained across chunks.
        _stride = max(1, int(_os.environ.get("KL_QUERY_STRIDE", "1")))
        _recompute = _os.environ.get("KL_CHUNK_RECOMPUTE", "1") == "1"
        q_sel_positions = torch.arange(0, q_len, _stride, device=idx_q.device)
        # with image_bucket_constraint the KL scope is the set of
        # BUCKETS (blocks) holding the query's own-page image tokens — block level, so the
        # fused Triton block-score kernel stays usable in the sparse stage.
        _page_scope = (self.image_bucket_constraint and image_token_page_ids is not None
                       and query_token_page_ids is not None)

        for c0 in range(0, q_sel_positions.numel(), chunk):
            sel = q_sel_positions[c0:c0 + chunk]
            cq = sel.numel()
            pos_chunk = position_ids[:, sel]  # [B, C]

            # under packing, `sel` are the GLOBAL flat query indices
            # and position_ids are per-sample resets. Evaluate causal/local in
            # global key space and carry a same-segment mask. _same_seg is None
            # (no-op) for the non-packed path.
            _packed = _pk.is_packed(position_ids)
            if _packed:
                pm = _pk.packed_masks(position_ids, k_positions, sel, self.local_window)
                pos_chunk = pm['gpos']
                token_future = pm['token_future']
                local_keep = pm['local_token']
                _same_seg = pm['same_seg']
            else:
                token_future = k_positions[None, None, None, :] > pos_chunk[:, None, :, None]
                local_keep = self._build_local_token_mask(pos_chunk, k_positions)
                _same_seg = None

            candidate = self._build_kl_candidate_mask(
                pos_chunk, num_key_blocks, image_token_mask, batch, cq,
                image_token_page_ids=image_token_page_ids,
                query_token_page_ids=query_token_page_ids)  # [B, 1, C, nblk]

            if restrict_to_selected:
                bi = block_indices[:, :, sel]  # [B, H_idx, C, topk]
                safe = bi.masked_fill(bi < 0, num_key_blocks)
                keep_block = bi.new_zeros(
                    (batch, n_idx_heads, cq, num_key_blocks + 1), dtype=torch.bool)
                keep_block.scatter_(-1, safe, True)
                keep_block = keep_block[..., :num_key_blocks]
                keep_block = keep_block & candidate  # image constraint (no-op when off)
            else:
                keep_block = candidate.expand(batch, n_idx_heads, cq, num_key_blocks).contiguous()

            q_chunk = idx_q[:, :, sel]
            q_main_chunk = q_main[:, :, sel]

            # _KLChunkFn/_kl_chunk_core rebuild the [B,H_idx,C,Sk]
            # token-level keep (with token-level local + causal) internally; only
            # keep_block ([.., nblk]) is saved for backward.
            if _recompute:
                kl = _KLChunkFn.apply(q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk,
                                      self.block_size, scaling, group, kv_group, self.local_window,
                                      _same_seg)
            else:
                kl = _kl_chunk_core(q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk,
                                    self.block_size, scaling, group, kv_group, self.local_window,
                                    _same_seg)

            keep_tok = keep_block.repeat_interleave(self.block_size, dim=-1)[..., :k_len]
            keep_tok = keep_tok & ~local_keep & ~token_future
            if _same_seg is not None:
                keep_tok = keep_tok & _same_seg
            valid = keep_tok.any(dim=-1).float()
            del keep_tok
            total_kl = total_kl + (kl * valid).sum()
            total_cnt = total_cnt + valid.sum()

        self._kl_log_step = getattr(self, "_kl_log_step", 0) + 1
        _log_due = self._kl_log_step % self.kl_log_interval == 0
        if _log_due:
            _keep_avg = (total_kl.detach() / (total_cnt.detach() + 1e-6)).item() if total_cnt.item() > 0 else 0.0
            _valid_q = int(total_cnt.item())
            _total_q = batch * n_idx_heads * q_sel_positions.numel()
            _stage = "sparse" if restrict_to_selected else "warmup"
            if self.layer_idx == 2:
                print(
                    f"[KLMask layer={self.layer_idx} step={self._kl_log_step}] stage={_stage} "
                    f"image_bucket_constraint={self.image_bucket_constraint} "
                    f"page_image_scope={_page_scope} "
                    f"kl_stride={_stride} recompute={_recompute} "
                    f"kl_avg={_keep_avg:.4f} "
                    f"valid_queries={_valid_q}/{_total_q}",
                    flush=True,
                )
            # paper-style block/score recall -> how well the indexer is
            # already aligned with the main branch (indexer training progress).
            # Printed for EVERY sparse layer (kl_avg is that layer's own KL).
            self._log_indexer_recall(
                idx_q, idx_k, q_main, k_main, position_ids, scaling, num_key_blocks,
                image_token_mask, image_token_page_ids, query_token_page_ids,
                _stage, _keep_avg)

        return total_kl / (total_cnt + 1e-6)

    def select_and_kl_loss(self, hidden_states, query_states, key_states, position_embeddings,
                           position_ids, scaling, image_token_mask=None,
                           image_token_page_ids=None, query_token_page_ids=None):
        """fused sparse-stage fast path.

        Combines block selection AND KL loss on a single index projection, with:
          * ``INDEXER_SCORE_TRITON=1`` — fused Triton block-max score (no dense
            [B,H,C,Sk] fp32 tensor). Requires n_idx_heads == 1.
          * ``KL_QUERY_STRIDE=N`` — subsample query rows in KL (distillation-only).
          * Gather-based KL on selected blocks only (S_sel = topk * block_size ≪ Sk).
          * ``_KLChunkFn`` recompute-in-backward fallback for multi-index-head.
        Returns (block_indices, kl_loss). Block indices are REMOTE-only (token-level
        local window is unioned in build_block_mask); grads flow only to W_idx.
        """
        import os as _os
        batch, q_len, _ = hidden_states.shape
        idx_q, idx_k = self._project_index_qk(hidden_states.detach(), position_ids, None)

        q_main = query_states.detach()  # [B, H, Sq, Dh]
        num_heads = q_main.shape[1]
        # keep k_main at KV-head granularity (no repeat_kv copy)
        k_main = key_states.detach()  # [B, H_kv, Sk, Dh]
        n_kv_heads = k_main.shape[1]
        n_idx_heads = self.num_heads
        group = num_heads // n_idx_heads
        kv_group = num_heads // n_kv_heads

        k_len = idx_k.shape[2]
        num_key_blocks = -(-k_len // self.block_size)
        pad = num_key_blocks * self.block_size - k_len
        k_positions = torch.arange(k_len, device=idx_q.device)
        topk = self._remote_topk(num_key_blocks)
        neg = -1e9

        chunk = self.kl_chunk_size if self.kl_chunk_size and self.kl_chunk_size > 0 else q_len
        block_indices = idx_q.new_empty((batch, n_idx_heads, q_len, topk), dtype=torch.long)
        total_kl = idx_q.new_zeros((), dtype=torch.float32)
        total_cnt = idx_q.new_zeros((), dtype=torch.float32)
        _total_fallback = 0
        # same-page image BUCKET domain (block level -> the fused
        # Triton block-score kernel below stays enabled).
        _page_scope = (self.image_bucket_constraint and image_token_page_ids is not None
                       and query_token_page_ids is not None)

        _use_score_triton = _os.environ.get("INDEXER_SCORE_TRITON", "1") == "1"

        for c0 in range(0, q_len, chunk):
            c1 = min(c0 + chunk, q_len)
            cq = c1 - c0
            pos_chunk = position_ids[:, c0:c1]
            token_future = k_positions[None, None, None, :] > pos_chunk[:, None, :, None]

            # global (flat) query positions arange(c0,c1); build
            # global causal + same-segment masks. _same_seg is None (no-op) when
            # not packed. The fused Triton kernels stay enabled: they take the
            # per-query segment start (segh / seg) and mask cross-sample keys.
            _packed = _pk.is_packed(position_ids)
            if _packed:
                if c0 == 0:
                    # warn once if the packed samples are not block-aligned
                    # (block grid would differ from single-sample inference).
                    _pk.check_block_alignment(position_ids, self.block_size)
                _sel = torch.arange(c0, c1, device=position_ids.device)
                pm = _pk.packed_masks(position_ids, k_positions, _sel, self.local_window)
                pos_chunk = pm['gpos']
                token_future = pm['token_future']
                _same_seg = pm['same_seg']
                _seg_chunk = pos_chunk - position_ids[:, c0:c1]  # per-query global segment start
            else:
                _same_seg = None
                _seg_chunk = None

            q_chunk = idx_q[:, :, c0:c1]  # [B, H_idx, C, D] bf16

            # --- block-score computation (non-differentiable) ---
            # fused Triton "score + block-amax" produces only
            # [B, H_idx, C, nblk] on-chip (no dense [.., Sk] tensor); matches the
            # eager `token_future | local_token` masking. Dense fallback below.
            # the page constraint is applied at BUCKET level on the
            # kernel's OUTPUT (block_scores masked by `candidate` below), so the fused
            # Triton block-score path stays enabled.
            block_scores = None
            if _use_score_triton and idx_q.is_cuda:
                try:
                    from .sparse_gather_triton import triton_block_max_score
                    # fold index heads into the batch dim -> single 3D kernel call,
                    # supports sparse_num_index_heads >= 1. idx_k is shared across
                    # query index heads ([B, 1, Sk, D]) -> expand before folding.
                    _Bg = batch * n_idx_heads
                    _D = q_chunk.shape[-1]
                    qh = q_chunk.reshape(_Bg, cq, _D).contiguous()
                    kh = idx_k.expand(batch, n_idx_heads, k_len, _D).reshape(_Bg, k_len, _D).contiguous()
                    posh = pos_chunk.unsqueeze(1).expand(batch, n_idx_heads, cq).reshape(_Bg, cq)
                    # per-query segment start (global pos here is pos_chunk;
                    # zeros when not packed -> kernel reduces to the original masking).
                    if _seg_chunk is None:
                        segh = torch.zeros_like(posh)
                    else:
                        segh = _seg_chunk.unsqueeze(1).expand(batch, n_idx_heads, cq).reshape(_Bg, cq)
                    block_scores = triton_block_max_score(
                        qh, kh, posh, segh, self.block_size, num_key_blocks,
                        local_window=self.local_window,
                    ).reshape(batch, n_idx_heads, cq, num_key_blocks)  # [B, HI, C, nblk] fp32
                except Exception as _e:
                    block_scores = None
                    if not getattr(self, "_score_triton_warned", False):
                        self._score_triton_warned = True
                        print(f"[SparseAttn] triton block-score path failed "
                              f"({type(_e).__name__}: {_e}); falling back to dense matmul",
                              flush=True)

            with torch.no_grad():
                if block_scores is None:
                    local_token = self._build_local_token_mask(pos_chunk, k_positions)
                    # pos_chunk holds GLOBAL positions when packed, so
                    # token_future/local_token are global-correct; also drop
                    # cross-segment keys so purely cross-sample blocks score -inf.
                    _excl = token_future | local_token
                    if _same_seg is not None:
                        _excl = _excl | ~_same_seg
                    s_full_sel = torch.matmul(q_chunk, idx_k.transpose(-1, -2))  # bf16
                    sc = s_full_sel.float().masked_fill(_excl, float("-inf"))
                    if pad:
                        sc = F.pad(sc, (0, pad), value=float("-inf"))
                    sc = sc.view(batch, n_idx_heads, cq, num_key_blocks, self.block_size)
                    block_scores = sc.amax(dim=-1)

                candidate = self._build_kl_candidate_mask(
                    pos_chunk, num_key_blocks, image_token_mask, batch, cq,
                    image_token_page_ids=image_token_page_ids,
                    query_token_page_ids=query_token_page_ids)

                if c0 == 0 and self.image_bucket_constraint and not getattr(self, "_img_blk_kl_logged", False):
                    self._img_blk_kl_logged = True
                    if image_token_mask is not None:
                        img_blk_log = self._build_image_block_allow_mask(image_token_mask, num_key_blocks)
                        avg_img_blk = img_blk_log.float().sum(dim=-1).mean().item()
                        print(
                            f"[ImageConstraint layer={self.layer_idx}] select_and_kl_loss() "
                            f"num_key_blocks={num_key_blocks} "
                            f"avg_image_blocks={avg_img_blk:.1f} "
                            f"image_block_ratio={avg_img_blk / num_key_blocks * 100:.1f}% "
                            f"(KL/selection candidates: image blocks; token-level local excluded)",
                            flush=True,
                        )

                block_scores_c = block_scores.masked_fill(~candidate.expand_as(block_scores), float("-inf"))
                # image-bucket safety fallback (applies to both true/false image
                # constraint): if a query has < topk finite candidate scores after
                # causal/local masking, revert it to the unconstrained scores.
                if topk > 0:
                    valid_c = (block_scores_c > float("-inf")).sum(dim=-1)
                    if (valid_c < topk).any():
                        need_fallback = (valid_c < topk).unsqueeze(-1).expand_as(block_scores_c)
                        block_scores_c = torch.where(need_fallback, block_scores, block_scores_c)

                if topk == 0:
                    bi = torch.empty(
                        batch, n_idx_heads, cq, 0,
                        dtype=torch.long, device=block_scores.device)
                else:
                    topk_scores, topk_indices = block_scores_c.topk(topk, dim=-1)
                    bi = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
                block_indices[:, :, c0:c1] = bi

                if image_token_mask is not None and topk > 0:
                    _total_fallback += int((bi < 0).all(dim=-1).sum().item())

            if topk == 0:
                # no candidate blocks: KL contributes nothing for this chunk
                continue

            # --- KL loss over the selected blocks ---
            if n_idx_heads == 1:
                # softmax over the topk*block_size selected keys
                # per query (~512..2048) instead of the full Sk. Assumes n_idx_heads==1
                # (Unlimited config). Union semantics via per-token non-local mask.
                s_sel_len = topk * self.block_size
                bs_ar = torch.arange(self.block_size, device=idx_q.device)
                safe_bi = bi.clamp_min(0)  # [B, 1, C, topk]
                tok_raw = (safe_bi.unsqueeze(-1) * self.block_size + bs_ar).view(
                    batch, n_idx_heads, cq, s_sel_len
                )  # [B, 1, C, S_sel]
                local_start = (pos_chunk - (self.local_window - 1)).clamp(min=0)
                keep_sel = (
                    (bi >= 0).unsqueeze(-1).expand(-1, -1, -1, -1, self.block_size).reshape(
                        batch, n_idx_heads, cq, s_sel_len)
                    & (tok_raw < k_len)
                    & (tok_raw <= pos_chunk[:, None, :, None])
                    & (tok_raw < local_start[:, None, :, None])
                )  # [B, 1, C, S_sel]
                # pos_chunk is global here; also require the selected
                # key and its query share a segment (trims boundary-straddling blocks).
                if _same_seg is not None:
                    _seg_q1 = _pk.segment_ids(position_ids)[:, c0:c1]         # [B, C]
                    _seg_tok1 = _pk.segment_of(position_ids, tok_raw)         # [B,1,C,S_sel]
                    keep_sel = keep_sel & (_seg_tok1 == _seg_q1[:, None, :, None])
                tok = tok_raw.clamp(max=k_len - 1)

                # --- student P_idx (differentiable) via gather + small matmul ---
                idx_k_flat = idx_k[:, 0].reshape(batch * k_len, self.head_dim)
                flat_tok = (
                    tok[:, 0].reshape(batch, -1)
                    + torch.arange(batch, device=tok.device)[:, None] * k_len
                ).reshape(-1)
                k_sel_idx = idx_k_flat.index_select(0, flat_tok).view(
                    batch, cq, s_sel_len, self.head_dim)
                s_sel = torch.matmul(k_sel_idx, q_chunk[:, 0].unsqueeze(-1)).squeeze(-1)
                s_sel = s_sel.view(batch, n_idx_heads, cq, s_sel_len).float()
                s_sel = s_sel.masked_fill(~keep_sel, neg)
                log_p_idx = F.log_softmax(s_sel, dim=-1)

                # --- teacher P: gather selected keys, per-KV-head scores, group-average ---
                with torch.no_grad():
                    head_dim_m = k_main.shape[-1]
                    q_main_chunk = q_main[:, :, c0:c1]
                    q_r = q_main_chunk.view(batch, n_kv_heads, kv_group, cq, head_dim_m)
                    k_bt = k_main.permute(0, 2, 1, 3).reshape(
                        batch * k_len, n_kv_heads * head_dim_m)
                    flat_idx = (
                        torch.arange(batch, device=tok.device)[:, None, None] * k_len + tok[:, 0]
                    ).reshape(-1)
                    k_sel = k_bt.index_select(0, flat_idx).view(
                        batch, cq, s_sel_len, n_kv_heads, head_dim_m
                    ).permute(0, 3, 1, 2, 4)  # [B, H_kv, C, S_sel, Dh]
                    s_t = torch.einsum("bkgcd,bkcsd->bkgcs", q_r, k_sel).float() * scaling
                    s_t = s_t.masked_fill(~keep_sel[:, :, None], neg)
                    p = F.softmax(s_t, dim=-1)
                    p = p.reshape(batch, num_heads, cq, s_sel_len).view(
                        batch, n_idx_heads, group, cq, s_sel_len
                    ).mean(dim=2)  # [B, 1, C, S_sel]
                    del k_bt, k_sel, s_t

                log_p = torch.log(p.clamp_min(1e-12))
                term = p * (log_p - log_p_idx)
                term = torch.where(keep_sel, term, torch.zeros_like(term))
                kl = term.sum(dim=-1)  # [B, 1, C]
                valid = keep_sel.any(dim=-1).float()
            else:
                # --- dense fallback for multi-index-head configs (mem-opt via _KLChunkFn) ---
                safe = bi.masked_fill(bi < 0, num_key_blocks)
                keep_block = bi.new_zeros(
                    (batch, n_idx_heads, cq, num_key_blocks + 1), dtype=torch.bool)
                keep_block.scatter_(-1, safe, True)
                keep_block = keep_block[..., :num_key_blocks]
                q_main_chunk = q_main[:, :, c0:c1]
                kl = _KLChunkFn.apply(q_chunk, idx_k, q_main_chunk, k_main, keep_block, pos_chunk,
                                      self.block_size, scaling, group, kv_group, self.local_window,
                                      _same_seg)
                keep_tok = keep_block.repeat_interleave(self.block_size, dim=-1)[..., :k_len]
                local_keep = self._build_local_token_mask(pos_chunk, k_positions)
                keep_tok = keep_tok & ~local_keep & ~token_future
                if _same_seg is not None:
                    keep_tok = keep_tok & _same_seg
                valid = keep_tok.any(dim=-1).float()
                del keep_tok

            total_kl = total_kl + (kl * valid).sum()
            total_cnt = total_cnt + valid.sum()

        self._kl_log_step = getattr(self, "_kl_log_step", 0) + 1
        if self.layer_idx == 2 and self._kl_log_step % self.kl_log_interval == 0:
            _keep_avg = (total_kl.detach() / (total_cnt.detach() + 1e-6)).item() if total_cnt.item() > 0 else 0.0
            _valid_q = int(total_cnt.item())
            _total_q = batch * n_idx_heads * q_len
            _fallback_info = f" fallback_queries={_total_fallback}" if self.image_bucket_constraint else ""
            print(
                f"[KLMask layer={self.layer_idx} step={self._kl_log_step}] stage=sparse "
                f"image_bucket_constraint={self.image_bucket_constraint} "
                f"page_image_scope={_page_scope} "
                f"keep_avg_keys={_keep_avg:.1f} "
                f"valid_queries={_valid_q}/{_total_q}"
                f"{_fallback_info}",
                flush=True,
            )

        return block_indices, total_kl / (total_cnt + 1e-6)

    def build_block_mask(self, block_indices, attention_mask, key_length, dtype, device, position_ids):
        """sparse_keep = block_keep | local_keep union.

        block_keep = selected remote block IDs expanded to token level.
        local_keep = TOKEN-level local window [q - local_window + 1, q].
        Matches Qwen3 sparse-attention main-branch semantics."""
        batch, n_idx_heads, q_len, _ = block_indices.shape
        num_key_blocks = -(-key_length // self.block_size)
        safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)
        block_bitmap = block_indices.new_zeros((batch, n_idx_heads, q_len, num_key_blocks + 1), dtype=torch.bool)
        block_bitmap.scatter_(-1, safe, True)
        block_bitmap = block_bitmap[..., :num_key_blocks]
        block_keep = block_bitmap.repeat_interleave(self.block_size, dim=-1)[..., :key_length]
        group = self.config.num_attention_heads // n_idx_heads
        block_keep = block_keep.repeat_interleave(group, dim=1)

        k_positions = torch.arange(key_length, device=device)
        if _pk.is_packed(position_ids):
            # evaluate the local window / causal in GLOBAL key space
            # and AND with same-segment so no query attends across a sample
            # boundary (block ids are global, so a boundary-straddling selected
            # block is trimmed to its in-segment keys here).
            q_len_p = position_ids.shape[1]
            m = _pk.packed_masks(position_ids, k_positions,
                                 torch.arange(q_len_p, device=device), self.local_window)
            sparse_keep = (block_keep | m['local_token']) & m['same_seg']
            if attention_mask is not None:
                keep = sparse_keep & (attention_mask[..., :key_length] > -1000.0)
            else:
                keep = sparse_keep & ~m['token_future']
            return keep
        local_keep = self._build_local_token_mask(position_ids, k_positions)
        sparse_keep = block_keep | local_keep

        if attention_mask is not None:
            keep = sparse_keep & (attention_mask[..., :key_length] > -1000.0)
        else:
            token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
            keep = sparse_keep & ~token_future
        return keep

    def build_flex_block_mask(self, block_indices, key_length, position_ids):
        """build a flex_attention BlockMask from sparse block_indices.

        Aggregates per-token block selections into per-q-block bitmaps (a kv_block is
        selected for q_block qb if ANY token in qb selected that kv_block), then wraps
        in a BlockMask with a causal mask_mod. Avoids materialising a dense [B,H,Sq,Sk]
        float bias — key when Sq/Sk exceed 10s of K.
        """
        from torch.nn.attention.flex_attention import BlockMask
        batch, n_idx_heads, q_len, topk = block_indices.shape
        group = self.config.num_attention_heads // n_idx_heads
        num_q_blocks = -(-q_len // self.block_size)
        num_kv_blocks = -(-key_length // self.block_size)

        block_indices_h = block_indices.repeat_interleave(group, dim=1)  # [B, H_main, Sq, topk]
        H_main = block_indices_h.shape[1]

        token_ids = torch.arange(q_len, device=block_indices.device)
        q_block_ids = token_ids // self.block_size  # [Sq]

        BH = batch * H_main
        bi_flat = block_indices_h.reshape(BH, q_len, topk)

        dense = torch.zeros(BH, num_q_blocks, num_kv_blocks, dtype=torch.bool,
                            device=block_indices.device)
        valid = bi_flat >= 0
        qb = q_block_ids.unsqueeze(0).unsqueeze(-1).expand(BH, q_len, topk)
        kv = bi_flat.clamp(min=0)

        qb_flat = qb[valid]
        kv_flat = kv[valid]
        bh_flat = torch.arange(BH, device=block_indices.device).unsqueeze(1).unsqueeze(2).expand(
            BH, q_len, topk)[valid]

        dense[bh_flat, qb_flat, kv_flat] = True
        dense = dense.reshape(batch, H_main, num_q_blocks, num_kv_blocks)

        kv_num_blocks = dense.sum(-1).to(torch.int32)
        kv_indices = torch.argsort(dense.int(), dim=-1, descending=True, stable=True).to(torch.int32)

        def causal_mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            BLOCK_SIZE=self.block_size,
            mask_mod=causal_mask_mod,
            seq_lengths=(q_len, key_length),
        )

    def _maybe_log(self, total_kl, total_cnt, batch, n_idx_heads, q_len, stage):
        self._kl_log_step = getattr(self, "_kl_log_step", 0) + 1
        if self.layer_idx == 2 and self._kl_log_step % self.kl_log_interval == 0:
            _keep_avg = (total_kl.detach() / (total_cnt.detach() + 1e-6)).item() if total_cnt.item() > 0 else 0.0
            print(f"[KLMask layer={self.layer_idx} step={self._kl_log_step}] stage={stage} "
                  f"image_bucket_constraint={self.image_bucket_constraint} "
                  f"kl_avg={_keep_avg:.4f} valid_queries={int(total_cnt.item())}/{batch * n_idx_heads * q_len}",
                  flush=True)


ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,

    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,

    "mha_eager": SlidingWindowLlamaAttention,
    # "mha_flash_attention_2": LlamaFlashAttention2
}


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size


        if config.use_mla:
            attn_implementation = "mla_" + config._attn_implementation
        else:
            attn_implementation = "mha_" + config._attn_implementation

        self.self_attn = ATTENTION_CLASSES[attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states, self_attn_weights, present_key_value = attn_outputs[0], attn_outputs[1], attn_outputs[2]
        # sparse indexer KL loss is the optional 4th element (None when no indexer)
        kl_loss = attn_outputs[3] if len(attn_outputs) > 3 else None
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # KL loss always the LAST element so the model can read layer_outputs[-1]
        outputs += (kl_loss,)

        return outputs


DeepseekV2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeepseekV2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, _GemmaRMSNorm):
            # indexer q_norm/k_norm are new params (not in checkpoint); must zero-init
            # (Gemma-style scale = 1 + weight), else uninitialized garbage explodes idx q/k.
            module.weight.data.zero_()


DeepseekV2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # print(config._attn_implementation)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Skip 4D causal mask for decode (q_len=1 with KV cache doesn't need it)
        if seq_length == 1 and past_key_values_length > 0:
            attention_mask = None
        elif self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif attention_mask is None or (attention_mask.dim() == 2 and bool(attention_mask.all())):
            # no padding -> skip the O(seq^2) dense 4D causal mask
            # (it OOMs at long ctx and defeats block-sparse). Downstream treats None as
            # pure causal: the triton gather kernel is causal by construction, _attn_forward
            # uses SDPA is_causal, and build_block_mask derives causality from position_ids.
            attention_mask = None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        indexer_kl_loss = None
        indexer_kl_count = 0

        # per-forward cache of Anchor-layer block_indices,
        # SHARED across all layers (Reuse layers must see what Anchor layers wrote).
        # Reset here so each forward step starts clean (no cross-step leak / OOM).
        # Populated by Anchor (source) layers, read by Reuse layers, both during the
        # forward pass and the gradient-checkpointing backward recompute.
        # NOTE: the shared dict is attached to each self_attn module (NOT to
        # self.config) -- PretrainedConfig is JSON-serialized on every checkpoint
        # save (config.save_pretrained -> to_json_string), and a cached Tensor
        # there would raise `TypeError: Object of type Tensor is not JSON serializable`.
        _shared_bi_cache = {}
        for decoder_layer in self.layers:
            decoder_layer.self_attn._sparse_block_indices_cache = _shared_bi_cache

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # sparse indexer KL loss is always the last element of the layer output
            layer_kl = layer_outputs[-1]
            if layer_kl is not None:
                indexer_kl_loss = layer_kl if indexer_kl_loss is None else indexer_kl_loss + layer_kl
                indexer_kl_count += 1

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if indexer_kl_loss is not None and indexer_kl_count > 0:
            indexer_kl_loss = indexer_kl_loss / indexer_kl_count

        # unbind temporary block_indices cache to prevent Tensor reference
        # leak into optimizer step and allow immediate CUDA CachingAllocator deallocation.
        for decoder_layer in self.layers:
            decoder_layer.self_attn._sparse_block_indices_cache = None

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache  # Always return DynamicCache to preserve custom attributes
        # one-shot diagnostic before returning
        if not getattr(self, '_kl_out_debug_logged', False):
            self._kl_out_debug_logged = True
            print(f"[debug DeepseekV2Model] indexer_kl_loss_type={type(indexer_kl_loss).__name__} "
                  f"count={indexer_kl_count} return_dict={return_dict} "
                  f"training={self.training}", flush=True)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        out.indexer_kl_loss = indexer_kl_loss  # dynamic attr, read by ForCausalLM
        # verify attr set right after assignment
        if not getattr(self, '_kl_out_verify_logged', False):
            self._kl_out_verify_logged = True
            print(f"[debug DeepseekV2Model post-set] hasattr={hasattr(out, 'indexer_kl_loss')} "
                  f"getattr={getattr(out, 'indexer_kl_loss', 'MISSING')} "
                  f"id(out)={id(out)}", flush=True)
        return out


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

        hidden_states = outputs[0]

        # fused chunked linear cross-entropy. Avoids materialising the full
        # [B, seq, vocab] logits (and its fp32 upcast) that OOMs on long multi-image sequences.
        # Gated by FUSED_CHUNK_CE (default on when labels present); falls back to the classic
        # path so generation / return-logits callers stay unaffected.
        import os as _os_ce
        _use_fused_ce = (
            labels is not None
            and _os_ce.environ.get('FUSED_CHUNK_CE', '1') == '1'
        )

        logits = None
        loss = None
        if _use_fused_ce:
            loss = _fused_chunked_lm_ce(
                hidden_states,
                self.lm_head,
                labels,
                self.config.vocab_size,
                num_items_in_batch=None,
                chunk_size=int(_os_ce.environ.get('CE_CHUNK_SIZE', '1024')),
            )
        else:
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        # add indexer KL loss (populated by DeepseekV2Model.forward when sparse is on)
        _indexer_kl = getattr(outputs, 'indexer_kl_loss', None)
        if loss is not None and _indexer_kl is not None:
            _kl_weight = 1.0
            _sp_cfg = getattr(self.config, 'sparse_attention_config', {}) or {}
            if isinstance(_sp_cfg, dict):
                _kl_weight = float(_sp_cfg.get('kl_loss_weight', 1.0))
            loss = loss + _kl_weight * _indexer_kl

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = getattr(past_key_values, 'get_max_length', lambda: None)()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if self.generation_config.cache_implementation == "static":
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The DeepseekV2 Model transformer with a sequence classification head on top (linear layer).

    [`DeepseekV2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2ForSequenceClassification(DeepseekV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                        torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
