"""DASH overlay on top of the page-isolated (page2d) base modeling.

Thin monkey-patch: `_sparse_base_impl.py` carries the full base implementation; this file
only swaps the **prefill** path of SlidingWindowLlamaAttention for
dash_attention.DashAttention (entmax soft prior + log(prior) bias, gradients reaching the
router straight from the LM loss, no KL teacher) when
`sparse_attention_config.training_stage == "dash"`. Decode, the non-dash stages and all
other classes keep using the base implementation.

Three details the dash path relies on -- page isolation degrades silently if any of them
is dropped:

1. The main-branch rope goes through `self._main_rope(...)` rather than
   `self.rotary_emb(...)`, otherwise sparse_main_rope_mode=page2d is bypassed and the main
   branch falls back to 1D.
2. `self.indexer` is **not** set to None: `_main_rope_coords` returns None without an
   indexer (main branch falls back to 1D), and every page/rope helper
   (_build_page_aware_ids / _build_page_rope_pos / rotary_emb_2d /
   _build_kl_candidate_mask) lives on the indexer.
3. Router scoring uses the indexer's own 2-axis rope (sparse_rope_*: one page pair at
   theta=256 with the page index taken mod 2, plus 63 token pairs at theta=1e4, resampled)
   together with a page-aware PER-QUERY candidate domain [B,S,nblk].
"""

import copy
import math
import os

import torch
import torch.nn.functional as F

if os.environ.get("DASH_DEBUG_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)
    print("[DASH] torch.autograd anomaly detection ENABLED", flush=True)

# Re-export the ENTIRE base namespace (including underscore helpers) so that
# modeling_unlimitedocr.py's `from .modeling_deepseekv2 import <name>` keeps working.
from . import _sparse_base_impl as _base  # noqa: E402

globals().update({k: v for k, v in vars(_base).items() if not k.startswith("__")})

from ._sparse_base_impl import (  # noqa: E402
    DeepseekV2PreTrainedModel as _BasePreTrained,
    SlidingWindowLlamaAttention as _BaseAttn,
    _llama_apply_rotary_pos_emb,
    _llama_repeat_kv,
)
from .dash_attention import DashAttention  # noqa: E402
from .dash_attention import entmax15 as _entmax15  # noqa: E402
from .dash_attention import _SCORE_NEG, _ACT_CLAMP, _MASK_NEG  # noqa: E402

_orig_init = _BaseAttn.__init__
_orig_forward = _BaseAttn.forward
_DASH_LOGGED = False
_IDX_ROPE_WARNED = False
# id(sparse_attention_config) -> deep copy taken BEFORE DashAttention.__init__ rewrites it
# (_persist_router_cfg writes the live process state -- k_start=64, env DASH_STAGE0 default
# "indexer" -- over dash_topk_blocks / dash_stage0 / ...). Inference reads the snapshot.
_CKPT_SP = {}
# Inference switch, toggled by UnlimitedOCRForCausalLM.infer (see configure_dash_inference).
# When on, DASH layers route exactly like vLLM's `unlimited_page2d_dash` plugin
# (dash_router.DashRouterMath.route_and_select: top-M entmax window + hard top-K read set)
# in prefill AND decode; decode no longer falls back to dense full-cache attention.
_INFER = {"on": False}
_ENTMAX_NEG = -1.0e4


def _patched_init(self, config, layer_idx):
    sp = getattr(config, "sparse_attention_config", None)
    is_dash = isinstance(sp, dict) and sp.get("training_stage") == "dash"
    if is_dash and id(sp) not in _CKPT_SP:
        _CKPT_SP[id(sp)] = copy.deepcopy(sp)
    if is_dash:
        _saved_stage = sp.get("training_stage")
        sp["training_stage"] = "warmup"
    _orig_init(self, config, layer_idx)
    if is_dash:
        sp["training_stage"] = _saved_stage

    self.dash_router = None
    if is_dash and self.indexer is not None:
        self.dash_router = DashAttention(config)
        global _DASH_LOGGED
        if not _DASH_LOGGED:
            _DASH_LOGGED = True
            print(f"[DASH] differentiable DASH attention active "
                  f"(entmax router, block_size={self.dash_router.block_size}, "
                  f"local_window={self.dash_router.local_window}, "
                  f"index_dim={self.dash_router.index_dim}, "
                  f"main_rope={self.main_rope_mode}, "
                  f"idx_rope={self.indexer.rope_mode}, "
                  f"page_bucket={self.indexer.image_bucket_constraint}); "
                  f"first dash layer_idx={layer_idx}", flush=True)


def _dash_page_ids(self, idx, src_ids, key_length):
    """Page-aware ids, built with the same calls as the base prefill path."""
    if idx.image_bucket_constraint:
        itm, ipi, qpi = idx._build_page_aware_ids(
            src_ids, idx.image_token_id, idx.page_token_id, key_length)
    else:
        itm = idx._build_flat_image_token_mask(src_ids, idx.image_token_id, key_length)
        ipi = qpi = None
    idx._log_image_mask(itm, src_ids, key_length)
    return itm, ipi, qpi


def _dash_block_cand(self, idx, router, pos_ids, itm, ipi, qpi, nblk, bsz, q_len):
    """PER-QUERY candidate domain [B,S,nblk], reusing the indexer's
    _build_kl_candidate_mask (same-page image bucket + per-query fallback). Built in
    q_chunk slices so the [B,S,nblk,block_size] intermediate is never materialised at
    once (32k x 1024 x 32 bool is roughly 1 GiB)."""
    if itm is None and (ipi is None or qpi is None):
        return None
    step = max(1, int(router.q_chunk))
    out = []
    for c0 in range(0, q_len, step):
        c1 = min(c0 + step, q_len)
        cm = idx._build_kl_candidate_mask(
            pos_ids[:, c0:c1], nblk, itm, bsz, c1 - c0,
            image_token_page_ids=ipi, query_token_page_ids=qpi)   # [B,1,C,nblk]
        out.append(cm[:, 0])
    return torch.cat(out, dim=1)                                  # [B,S,nblk]


def _dash_indexer_rope(self, idx, router, x, pos_ids, coords, main_pe):
    """cos/sin used for router scoring: the indexer's 2-axis rope (sparse_rope_*) when
    the coordinates are available, otherwise the indexer's 1D rope. With
    index_dim < head_dim the two axes cannot be sliced (cat(ang, ang) + rotate_half would
    mix the phases of the two axes), so 1D is used instead."""
    global _IDX_ROPE_WARNED
    if coords is not None and getattr(idx, "rotary_emb_2d", None) is not None:
        if router.index_dim == idx.head_dim:
            return idx.rotary_emb_2d(x, coords[0], coords[1])
        if not _IDX_ROPE_WARNED:
            _IDX_ROPE_WARNED = True
            print(f"[PageRope][WARN] sparse_index_dim={router.index_dim} != "
                  f"head_dim={idx.head_dim}: the 2-axis cos/sin cannot be sliced per "
                  f"frequency pair, router scoring falls back to 1D RoPE "
                  f"(the main branch is unaffected)", flush=True)
    if getattr(idx, "rotary_emb_1d", None) is not None:
        return idx.rotary_emb_1d(x, pos_ids)
    return main_pe


def _dash_attn_forward(self, hidden_states, attention_mask, position_ids, past_kv):
    idx = self.indexer
    router = self.dash_router
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    num_kv_groups = self.num_key_value_groups
    bsz, q_len, _ = hidden_states.size()

    pos_ids = position_ids
    if _base._pk.is_packed(pos_ids):
        raise RuntimeError(
            "[DASH] packing rows are not supported by the dash path "
            "(fused e2e kernel requires attn_mask=None, and page rope would run the "
            "<PAGE> cumsum across sample boundaries). Disable packing.")

    q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    src_ids = getattr(self, '_sparse_source_input_ids', None)
    itm, ipi, qpi = _dash_page_ids(self, idx, src_ids, q_len)

    coords = None
    if idx.rope_mode == "page2d":
        coords = self._main_rope_pos_2d
        if coords is not None and coords[0].shape[-1] != q_len:
            coords = None
        if coords is None:
            coords = idx._build_page_rope_pos(src_ids, q_len)
        idx._rope_pos_2d = coords
    if self.main_rope_mode == "page2d" and coords is not None:
        self._main_rope_pos_2d = coords
        cos, sin = self._main_rope(value_states, pos_ids, past_kv, q_len)
    else:
        self._main_rope_pos_2d = None
        cos, sin = self.rotary_emb(value_states, pos_ids)
    q, key_states = _llama_apply_rotary_pos_emb(q, key_states, cos, sin)

    if past_kv is not None:
        key_states, value_states = past_kv.update(key_states, value_states, self.layer_idx)
    k = _llama_repeat_kv(key_states, num_kv_groups)
    v = _llama_repeat_kv(value_states, num_kv_groups)
    idx_pe = _dash_indexer_rope(self, idx, router, value_states, pos_ids, coords, (cos, sin))

    nblk = -(-k.shape[2] // router.block_size)
    block_cand = _dash_block_cand(self, idx, router, pos_ids, itm, ipi, qpi,
                                  nblk, bsz, q_len)

    router_mask = attention_mask
    if router_mask is not None and bool((router_mask[:, :, -1, :] > -1.0).all()):
        router_mask = None

    if _INFER["on"]:
        attn_output = _infer_prefill_attend(self, router, hidden_states, q, k, v, idx_pe,
                                            pos_ids, block_cand, router_mask, past_kv)
    else:
        attn_output = router(hidden_states, q, k, v,
                             position_embeddings=idx_pe,
                             position_ids=pos_ids,
                             attention_mask=router_mask,
                             image_block_mask=None,
                             block_cand=block_cand,
                             page_ids=(ipi, qpi))
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_kv, None


def _patched_forward(self, *args, **kwargs):
    if self.dash_router is not None:
        def _get(name, idx_, default=None):
            if name in kwargs:
                return kwargs[name]
            if len(args) > idx_:
                return args[idx_]
            return default

        hidden_states = _get('hidden_states', 0)
        attention_mask = _get('attention_mask', 1)
        position_ids = _get('position_ids', 2)
        past_kv = _get('past_key_value', 3)
        if past_kv is None:
            past_kv = kwargs.get('past_key_values', None)

        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            if position_ids is None:
                position_ids = torch.arange(
                    q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            return _dash_attn_forward(self, hidden_states, attention_mask,
                                      position_ids, past_kv)
        if (_INFER["on"] and past_kv is not None
                and self.layer_idx in getattr(past_kv, "_dash_idx_k", {})):
            return _dash_decode_forward(self, hidden_states, position_ids, past_kv)

    return _orig_forward(self, *args, **kwargs)


_BaseAttn.__init__ = _patched_init
_BaseAttn.forward = _patched_forward

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_orig_init_weights = _BasePreTrained._init_weights


def _patched_init_weights(self, module):
    _orig_init_weights(self, module)
    if isinstance(module, DashAttention):
        module.head_cls.data.zero_()


_BasePreTrained._init_weights = _patched_init_weights


# ===========================================================================
# Inference path -- parity with vLLM's `unlimited_page2d_dash` plugin
# (plugin_page2d_dash/unlimited_page2d_dash: dash_router.py + dash_attention_layer.py).
# Only active between configure_dash_inference(model, True) / (model, False), i.e. inside
# UnlimitedOCRForCausalLM.infer; training never enters these functions.
#
# vs. the training forward (DashAttention.forward) the inference router differs in:
#   * routing params come from the checkpoint snapshot (_CKPT_SP): dash_topk_blocks (hard
#     top-K read budget), dash_stage0, per-layer dash_scale_per_layer / dash_floor_per_layer
#     -- the live config dict has been overwritten by _persist_router_cfg at construction;
#   * entmax runs on the top-M (DASH_INFER_TOPM, deploy default 16) scores, then the read
#     set is the first K candidate slots (DashRouterMath.route_and_select, budget=topk);
#   * decode (q_len == 1) is ALSO block-sparse (vLLM DASH_DECODE_DENSE=0): head_cls block
#     summaries over the cached router keys, page-aware candidate mask, bias-aware attend
#     over the selected blocks + the token-level local window.
# ===========================================================================


def _sched_slot(sp, layer_idx):
    freq = sp.get("sparse_attention_freq")
    if not isinstance(freq, (list, tuple)) or layer_idx >= len(freq):
        return int(layer_idx)
    return sum(1 for f in freq[:layer_idx] if f)


def _per_layer_val(sp, key, layer_idx):
    arr = sp.get(key)
    if not isinstance(arr, (list, tuple)) or not arr:
        return None
    slot = _sched_slot(sp, int(layer_idx))
    return float(arr[slot]) if 0 <= slot < len(arr) else None


def configure_dash_inference(decoder, on, topm=16, page_scope_prefill=True, state=None):
    """Switch the vLLM-parity DASH inference path on/off for ``decoder`` (DeepseekV2Model).

    on=True returns a state dict; pass it back with on=False to restore.
    """
    if not on:
        if state is not None:
            _base._PAGE_SCOPE_PREFILL = state["page_scope_prefill"]
            for attn, stage0 in state["stage0"]:
                attn.dash_router.stage0 = stage0
        _INFER["on"] = False
        return None
    state = {"page_scope_prefill": _base._PAGE_SCOPE_PREFILL, "stage0": []}
    cfg = decoder.config
    live = getattr(cfg, "sparse_attention_config", None) or {}
    sp = _CKPT_SP.get(id(live), live)
    stage0 = str(os.environ.get("DASH_STAGE0") or sp.get("dash_stage0") or "indexer").lower()
    budget = sp.get("dash_topk_blocks") or sp.get("dash_image_fallback_remote") or 12
    for layer in decoder.layers:
        attn = getattr(layer, "self_attn", None)
        router = getattr(attn, "dash_router", None)
        if router is None:
            continue
        li = attn.layer_idx
        scale = _per_layer_val(sp, "dash_scale_per_layer", li)
        floor = _per_layer_val(sp, "dash_floor_per_layer", li)
        if scale is None:
            scale = float(sp.get("dash_scaling_factor", router.scaling_factor))
        if floor is None:
            floor = float(sp.get("dash_prior_floor", router._prior_floor))
        router._infer_params = dict(
            idx_scale=float(scale) / math.sqrt(router.index_dim), floor=float(floor),
            sigma=float(sp.get("dash_sigma", router.sigma)), K=int(budget), topm=int(topm))
        state["stage0"].append((attn, router.stage0))
        router.stage0 = stage0
    # vLLM page_state.py isolates PREFILL queries too (image query -> its own page, prompt
    # marker -> raw page, trailing instruction -> unconstrained) == training PAGE_SCOPE_PREFILL=1.
    _base._PAGE_SCOPE_PREFILL = bool(page_scope_prefill)
    _INFER["on"] = True
    return state


@torch.no_grad()
def _infer_route_select(block_scores, block_cand, p):
    """vLLM DashRouterMath.route_and_select (budget_topk=True) with training's candidacy.

    block_scores [B,Lq,nblk] fp32 (_SCORE_NEG = not a remote block); block_cand
    [B,Lq,nblk] bool page-aware candidate domain or None.
    Returns (topi [B,Lq,K] int64, bias [B,Lq,K] fp32, valid [B,Lq,K] bool).
    """
    remote = block_scores > (_SCORE_NEG * 0.5)
    cand = remote if block_cand is None else (remote & block_cand)
    cand = torch.where(~cand.any(dim=-1, keepdim=True), remote, cand)
    x = torch.where(cand, block_scores * p["idx_scale"], block_scores.new_full((), _ENTMAX_NEG))
    x = torch.nan_to_num(x, nan=_ENTMAX_NEG, posinf=_ENTMAX_NEG, neginf=_ENTMAX_NEG)
    nblk = x.shape[-1]
    M = max(1, min(int(p["topm"]), nblk)) if int(p["topm"]) > 0 else nblk
    xv, xi = x.topk(M, dim=-1)                      # descending
    prior = _entmax15(xv, dim=-1)                   # tau* over the top-M window
    is_cand = cand.gather(-1, xi)
    K = max(1, min(int(p["K"]), M))
    xi, prior, valid = xi[..., :K], prior[..., :K], is_cand[..., :K]
    logp = torch.log(prior.clamp_min(p["floor"]))
    cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1)
    mu = torch.where(valid, logp, torch.zeros_like(logp)).sum(dim=-1, keepdim=True) / cnt
    bias = torch.where(valid, (logp - mu) * (1.0 / p["sigma"]), torch.zeros_like(logp))
    return xi, bias.float(), valid


def _router_qk(router, hidden_states, pe):
    """Router index q/k exactly as DashAttention._block_scores_headcls builds them."""
    B, S, _ = hidden_states.shape
    d = router.index_dim
    hs = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
    idx_q = router.q_norm(router.q_proj(hs).view(B, S, 1, d)).transpose(1, 2)
    idx_k = router.k_norm(router.k_proj(hs).view(B, S, 1, d)).transpose(1, 2)
    if pe is not None:
        cos, sin = pe
        idx_q, idx_k = _llama_apply_rotary_pos_emb(idx_q, idx_k, cos[..., :d], sin[..., :d])
    idx_q = torch.nan_to_num(idx_q[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
    idx_k = torch.nan_to_num(idx_k[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
    return idx_q, idx_k


@torch.no_grad()
def _block_scores_infer(router, idx_q, idx_k, q_pos, nblk):
    """[B,Lq,nblk] fp32 remote block scores for queries ``idx_q`` at positions ``q_pos``
    against ALL keys ``idx_k`` [B,S,d] (prefill: S == Lq; decode: the cached keys).
    Same math as DashAttention._block_scores_headcls / _block_scores."""
    B, S, d = idx_k.shape
    bs = router.block_size
    inv_sqrt_d = 1.0 / math.sqrt(d)
    Sp = nblk * bs
    blk_ids = torch.arange(nblk, device=idx_k.device)
    out = []
    if router.stage0 == "head_cls":
        if Sp > S:
            kk = F.pad(idx_k, (0, 0, 0, Sp - S))
            real = F.pad(idx_k.new_ones(B, S, dtype=torch.bool), (0, Sp - S), value=False)
        else:
            kk = idx_k[:, :Sp]
            real = idx_k.new_ones(B, Sp, dtype=torch.bool)
        kk = kk.view(B, nblk, bs, d)
        vmask = real.view(B, nblk, bs)
        hc = router.head_cls.to(kk.dtype).view(1, 1, 1, d)
        aw = ((kk * hc).sum(-1) * inv_sqrt_d).clamp(-_ACT_CLAMP, _ACT_CLAMP)
        neg = torch.finfo(aw.dtype).min
        row_max = torch.where(vmask, aw, aw.new_full((), neg)).amax(dim=-1, keepdim=True)
        row_max = torch.where(vmask.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max))
        e = torch.exp(aw - row_max) * vmask.to(aw.dtype)
        w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        k_bar = torch.nan_to_num((w.unsqueeze(-1) * kk).sum(dim=2), nan=0.0, posinf=0.0, neginf=0.0)
        chunk_valid = vmask.any(dim=-1)                                    # [B,nblk]
        for c0 in range(0, idx_q.shape[1], router.q_chunk):
            c1 = min(c0 + router.q_chunk, idx_q.shape[1])
            z = torch.matmul(idx_q[:, c0:c1], k_bar.transpose(-1, -2)) * inv_sqrt_d
            z = z.clamp(-_ACT_CLAMP, _ACT_CLAMP)
            qblk = q_pos[:, c0:c1] // bs
            keep = (blk_ids[None, None, :] <= (qblk[:, :, None] - router.local_blocks)) \
                & chunk_valid[:, None, :]
            out.append(z.masked_fill(~keep, _SCORE_NEG))
        return torch.cat(out, dim=1)
    # stage0 == "indexer": block-max of token-level scores, remote-only (causal minus local)
    kpos = torch.arange(S, device=idx_k.device)
    for c0 in range(0, idx_q.shape[1], router.q_chunk):
        c1 = min(c0 + router.q_chunk, idx_q.shape[1])
        sc = torch.matmul(idx_q[:, c0:c1], idx_k.transpose(-1, -2)).clamp(-_ACT_CLAMP, _ACT_CLAMP)
        pos = q_pos[:, c0:c1]
        future = kpos[None, None, :] > pos[:, :, None]
        local = (kpos[None, None, :] > (pos[:, :, None] - router.local_window)) & ~future
        sc = sc.masked_fill(future | local, _SCORE_NEG)
        if Sp > S:
            sc = F.pad(sc, (0, Sp - S), value=_SCORE_NEG)
        out.append(sc.view(B, c1 - c0, nblk, bs).amax(dim=-1))
    return torch.cat(out, dim=1)


def _infer_prefill_attend(self, router, hidden_states, q, k, v, idx_pe, pos_ids,
                          block_cand, attn_mask, past_kv):
    """Prefill (q_len > 1) under _INFER: vLLM-parity routing, training attend kernels."""
    B, H, S, D = q.shape
    nblk = (S + router.block_size - 1) // router.block_size
    p = router._infer_params
    with torch.autocast(device_type=q.device.type, enabled=False):
        idx_q, idx_k = _router_qk(router, hidden_states, idx_pe)
        if past_kv is not None:
            if not hasattr(past_kv, "_dash_idx_k"):
                past_kv._dash_idx_k = {}
            past_kv._dash_idx_k[self.layer_idx] = idx_k
        sc = _block_scores_infer(router, idx_q, idx_k, pos_ids, nblk)
        topi, bias, valid = _infer_route_select(sc, block_cand, p)
    if router.use_triton and q.is_cuda and attn_mask is None:
        out = router._forward_triton(q, k, v, topi, bias, valid, S, nblk, pos_ids)
        if out is not None:
            return out
    return router._forward_dense(q, k, v, topi, bias, valid, S, nblk, attn_mask)


@torch.no_grad()
def _decode_attend(q, k_all, v_all, topi, bias, valid, q_pos, block_size, local_window, scale):
    """Bias-aware block-sparse attention for decode queries (fp32 math).

    Selected remote blocks add their ``bias``; the token-level local window
    (q_pos - local_window, q_pos] is attended without bias and wins on overlap --
    same read set / weights as vLLM dash_sparse_attn decode and training _forward_dense.
    """
    B, H, Lq, D = q.shape
    S = k_all.shape[2]
    nblk = (S + block_size - 1) // block_size
    kpos = torch.arange(S, device=q.device)
    src = torch.where(valid, bias, bias.new_full((), _MASK_NEG))
    blk = torch.full((B, Lq, nblk), _MASK_NEG, device=q.device, dtype=torch.float32)
    blk = blk.scatter(-1, topi.long(), src.float())
    base = blk.index_select(-1, kpos // block_size)                       # [B,Lq,S]
    qp = q_pos[:, :, None]
    causal = kpos[None, None, :] <= qp
    local = causal & (kpos[None, None, :] > (qp - local_window))
    keep = causal & (local | (base > _MASK_NEG * 0.5))
    m = torch.where(local, torch.zeros_like(base), base).masked_fill(~keep, float("-inf"))
    kk = k_all.float()
    vv = v_all.float()
    if kk.shape[1] != H:
        kk = _llama_repeat_kv(kk, H // kk.shape[1])
        vv = _llama_repeat_kv(vv, H // vv.shape[1])
    s = torch.matmul(q.float(), kk.transpose(-1, -2)) * scale + m[:, None]
    return torch.matmul(torch.softmax(s, dim=-1), vv)                     # [B,H,Lq,D]


def _dash_decode_forward(self, hidden_states, position_ids, past_kv):
    """Decode step (q_len == 1) of a DASH layer under _INFER (vLLM DASH_DECODE_DENSE=0)."""
    idx = self.indexer
    router = self.dash_router
    nh = self.config.num_attention_heads
    nkv = self.config.num_key_value_heads
    hd = self.head_dim
    bsz, q_len, _ = hidden_states.size()
    if position_ids is None:
        past = past_kv.get_seq_length(self.layer_idx)
        position_ids = torch.arange(past, past + q_len, device=hidden_states.device)[None].expand(bsz, -1)

    q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)

    # (tok_pos, page_pos) from the incremental decode state machine (advances it once).
    coords = None
    if self.main_rope_mode == "page2d" and self.rotary_emb_2d_main is not None:
        coords = self._main_rope_coords(q_len, past_kv)
    if coords is not None:
        cos, sin = self.rotary_emb_2d_main(v, coords[0], coords[1])
    else:
        cos, sin = self.rotary_emb(v, position_ids)
    q, k = _llama_apply_rotary_pos_emb(q, k, cos, sin)
    k_all, v_all = past_kv.update(k, v, self.layer_idx)
    idx_pe = _dash_indexer_rope(self, idx, router, v, position_ids,
                                coords if idx.rope_mode == "page2d" else None, (cos, sin))

    p = router._infer_params
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        idx_q, idx_k_new = _router_qk(router, hidden_states, idx_pe)
        idx_k = torch.cat([past_kv._dash_idx_k[self.layer_idx], idx_k_new], dim=1)
        past_kv._dash_idx_k[self.layer_idx] = idx_k
        S = k_all.shape[2]
        nblk = (S + router.block_size - 1) // router.block_size
        sc = _block_scores_infer(router, idx_q, idx_k, position_ids, nblk)   # [B,Lq,nblk]
        allow = getattr(self, "_dash_dec_allow", None)                      # [B,nblk] or None
        cand = None
        if allow is not None and router.image_only:
            if allow.shape[-1] < nblk:
                allow = F.pad(allow, (0, nblk - allow.shape[-1]), value=False)
            cand = allow[:, None, :nblk].expand(bsz, q_len, nblk)
        topi, bias, valid = _infer_route_select(sc, cand, p)
        out = _decode_attend(q, k_all, v_all, topi, bias, valid, position_ids,
                             router.block_size, router.local_window, router._scale)
    out = out.to(hidden_states.dtype).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    return self.o_proj(out), None, past_kv, None
