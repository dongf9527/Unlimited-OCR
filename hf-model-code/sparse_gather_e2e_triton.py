"""fused Triton gather-attention WITH a differentiable per-block bias.

This is the high-performance backend for HiLS-style end-to-end selector training.
It is a copy of ``sparse_gather_triton.py`` (the production sparse-stage kernel, whose
union/causal/local-window semantics are already battle-tested) with ONE addition:

  * a per-(token, kv-block) additive bias ``blk_bias[B, T, M]`` (M = num kv blocks) is
    added to the logits of the tokens attended through the SELECTED (remote) blocks --
    exactly HiLS's inter-chunk gate. The forward adds it; the backward returns
    ``d_blk_bias`` so the LM-loss gradient flows back through the bias into whatever
    produced it (the indexer q/k projections). Local-window keys get zero bias, matching
    the eager reference in ``DeepseekV2Indexer.e2e_sparse_attention``.

The differentiable log-sum-exp block mass itself is still computed in PyTorch (validated
by test_e2e_math.py); this kernel only replaces the heavy gather + softmax + matmul and
their backward, and threads the bias gradient through.

Semantics (must match the eager e2e reference):
  logit(query t, key j in selected block b) = scale * q_t . k_j + blk_bias[t, b]
                                               for j causal and j < local_start(t)
  logit(query t, key j in local window)     = scale * q_t . k_j     (bias 0)
Single index head (block indices shared across kv heads), single contiguous causal
sequence, q_len == k_len (training prefill). The wrapper returns None on unsupported
shapes so the caller falls back to the eager torch path.
"""

import torch
import triton
import triton.language as tl

# Force transformers' dynamic-module loader to copy this file (see the twin sentinel
# in sparse_gather_triton.py). modeling_unlimitedocr.py imports this name at top level.
SPARSE_E2E_SENTINEL = 1


@triton.jit(do_not_specialize=['T'])
def _e2e_attn_fwd_kernel(
    q, k, v, o, lse, scale, block_indices, blk_bias,
    T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, S: tl.constexpr, M: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LW: tl.constexpr, NLB: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * S
    blk_bias += (bos + i_t) * M

    o_g = tl.arange(0, GP)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_g = o_g < G
    m_k = o_k < K
    m_v = (i_v * BV + o_v) < V

    p_q = q + (bos + i_t) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
    b_q = tl.load(p_q, mask=m_g[:, None] & m_k[None, :], other=0.)
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([GP, BV], dtype=tl.float32)
    b_m = tl.full([GP], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([GP], dtype=tl.float32)
    ls = i_t - LW + 1

    # --- remote: selected non-local blocks (keys < local_start), WITH per-block bias ---
    for i in range(S):
        bid = tl.load(block_indices + i).to(tl.int32)
        i_s = bid * BS
        if i_s <= i_t and i_s >= 0:
            b_bias = tl.load(blk_bias + bid).to(tl.float32)
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_s = tl.where(((i_t >= key) & (key < ls))[None, :], b_s + b_bias, float('-inf'))
            # safe online softmax: a fully-masked tile keeps b_m at -inf without NaN
            # (b_m_safe==0 there, so alpha=exp(-inf)=0 and b_p=exp(-inf)=0).
            b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
            b_m_safe = tl.where(b_m_new == float('-inf'), 0.0, b_m_new)
            b_r = tl.exp(b_m - b_m_safe)
            b_p = tl.exp(b_s - b_m_safe[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
            b_m = b_m_new

    # --- local window blocks (keys >= local_start), NO bias ---
    q_blk = i_t // BS
    for lb in range(NLB):
        i_s = (q_blk - lb) * BS
        if i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_s = tl.where(((i_t >= key) & (key >= ls))[None, :], b_s, float('-inf'))
            b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
            b_m_safe = tl.where(b_m_new == float('-inf'), 0.0, b_m_new)
            b_r = tl.exp(b_m - b_m_safe)
            b_p = tl.exp(b_s - b_m_safe[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
            b_m = b_m_new

    b_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_safe[:, None]
    b_lse = b_m + tl.log(b_safe)

    p_o = o + (bos + i_t) * HQ * V + (i_h * G + o_g)[:, None] * V + (i_v * BV + o_v)[None, :]
    tl.store(p_o, b_o.to(q.dtype.element_ty), mask=m_g[:, None] & m_v[None, :])
    if i_v == 0:
        p_lse = lse + (bos + i_t) * HQ + i_h * G + o_g
        tl.store(p_lse, b_lse, mask=m_g)


@triton.jit(do_not_specialize=['T'])
def _e2e_attn_bwd_dq_kernel(
    q, k, v, lse, delta, do, dq, scale, block_indices, blk_bias, d_bias,
    T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, S: tl.constexpr, M: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LW: tl.constexpr, NLB: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * S
    blk_bias += (bos + i_t) * M
    d_bias += (bos + i_t) * M

    o_g = tl.arange(0, GP)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_g = o_g < G
    m_k = o_k < K
    m_v = (i_v * BV + o_v) < V

    p_q = q + (bos + i_t) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
    b_q = tl.load(p_q, mask=m_g[:, None] & m_k[None, :], other=0.)
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = do + (bos + i_t) * HQ * V + (i_h * G + o_g)[:, None] * V + (i_v * BV + o_v)[None, :]
    b_do = tl.load(p_do, mask=m_g[:, None] & m_v[None, :], other=0.)
    p_lse = lse + (bos + i_t) * HQ + i_h * G + o_g
    b_lse = tl.load(p_lse, mask=m_g, other=float('inf'))
    p_delta = delta + (bos + i_t) * HQ + i_h * G + o_g
    b_delta = tl.load(p_delta, mask=m_g, other=0.)

    b_dq = tl.zeros([GP, BK], dtype=tl.float32)
    ls = i_t - LW + 1

    # --- remote: selected non-local blocks (bias added; accumulate d_bias) ---
    for i in range(S):
        bid = tl.load(block_indices + i).to(tl.int32)
        i_s = bid * BS
        if i_s <= i_t and i_s >= 0:
            b_bias = tl.load(blk_bias + bid).to(tl.float32)
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_p = tl.exp((b_s + b_bias) - b_lse[:, None])
            b_p = tl.where(((i_t >= key) & (key < ls))[None, :], b_p, 0.)
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
            # d_bias[t, bid] += sum over this kv-head-group's query rows & keys of ds.
            # Atomics accumulate the remaining kv heads (other programs); padded head
            # rows have lse=+inf -> b_p=0, non-kept keys -> b_p=0, so they add nothing.
            if i_v == 0:
                tl.atomic_add(d_bias + bid, tl.sum(b_ds))

    # --- local window blocks (no bias, no d_bias) ---
    q_blk = i_t // BS
    for lb in range(NLB):
        i_s = (q_blk - lb) * BS
        if i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where(((i_t >= key) & (key >= ls))[None, :], b_p, 0.)
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    p_dq = dq + (bos + i_t) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
    tl.store(p_dq, b_dq.to(q.dtype.element_ty), mask=m_g[:, None] & m_k[None, :])


@triton.jit(do_not_specialize=['T'])
def _e2e_attn_bwd_dkv_kernel(
    q, k, v, lse, delta, do, dk, dv, block_mask, blk_bias, scale,
    T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, M: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LW: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    o_g = tl.arange(0, GP)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_g = o_g < G
    m_k = o_k < K
    m_v = (i_v * BV + o_v) < V

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    key = i_s * BS + tl.arange(0, BS)   # [BS] key positions of this kv block

    for i in range(i_s * BS, T):
        b_flag = tl.load(block_mask + (bos + i) * M + i_s)
        _proc = (b_flag != 0) | (i < (i_s + 1) * BS + LW)
        if _proc:
            p_q = q + (bos + i) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
            b_q = tl.load(p_q, mask=m_g[:, None] & m_k[None, :], other=0.)
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = do + (bos + i) * HQ * V + (i_h * G + o_g)[:, None] * V + (i_v * BV + o_v)[None, :]
            b_do = tl.load(p_do, mask=m_g[:, None] & m_v[None, :], other=0.)
            p_lse = lse + (bos + i) * HQ + i_h * G + o_g
            b_lse = tl.load(p_lse, mask=m_g, other=float('inf'))
            p_delta = delta + (bos + i) * HQ + i_h * G + o_g
            b_delta = tl.load(p_delta, mask=m_g, other=0.)

            ls = i - LW + 1
            is_local = key > (i - LW)               # [BS]
            is_remote = key < ls                     # [BS]
            # bias applies only to REMOTE keys of a SELECTED block (dense cell is 0 for
            # non-selected blocks) -> matches the fwd remote-loop bias, local keys get 0.
            b_bias = tl.load(blk_bias + (bos + i) * M + i_s).to(tl.float32)
            bias_key = tl.where(is_remote, b_bias, 0.0)   # [BS]

            b_s = tl.dot(b_k, tl.trans(b_q))         # [BS, GP]
            b_p = tl.exp((b_s + bias_key[:, None]) - b_lse[None, :])
            causal = i >= key
            keep = causal & (is_local | (b_flag != 0))
            b_p = tl.where(keep[:, None], b_p, 0.)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(q.dtype.element_ty), boundary_check=(0, 1))


def _meta(q, k, v, sel, block_size):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[3]
    S = sel.shape[-1]
    G = HQ // H
    GP = max(16, triton.next_power_of_2(G))
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(V))
    if BV < V:
        raise NotImplementedError(f"head_dim V={V} > 128 not supported")
    if BK > 256:
        raise NotImplementedError(f"head_dim K={K} > 256 not supported")
    return B, T, HQ, H, K, V, S, G, GP, block_size, BK, BV


def _local_meta(block_size, local_window):
    lw = block_size if local_window is None else int(local_window)
    lw = max(1, lw)
    nlb = (lw + block_size - 1) // block_size + 1
    return lw, nlb


class _E2EGatherAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sel, blk_bias, block_size, scale, local_window):
        B, T, HQ, H, K, V, S, G, GP, BS, BK, BV = _meta(q, k, v, sel, block_size)
        LW, NLB = _local_meta(BS, local_window)
        M = triton.cdiv(T, BS)
        assert blk_bias.shape[-1] == M, f"blk_bias last dim {blk_bias.shape[-1]} != M {M}"
        blk_bias = blk_bias.contiguous().float()

        o = torch.empty(B, T, HQ, V, dtype=q.dtype, device=q.device)
        lse = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
        _e2e_attn_fwd_kernel[(T, 1, B * H)](
            q, k, v, o, lse, scale, sel, blk_bias,
            T=T, H=H, HQ=HQ, G=G, GP=GP, K=K, V=V, S=S, M=M, BS=BS, BK=BK, BV=BV,
            LW=LW, NLB=NLB, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, sel, blk_bias, o, lse)
        ctx.block_size = block_size
        ctx.scale = scale
        ctx.local_window = local_window
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sel, blk_bias, o, lse = ctx.saved_tensors
        block_size, scale, local_window = ctx.block_size, ctx.scale, ctx.local_window
        B, T, HQ, H, K, V, S, G, GP, BS, BK, BV = _meta(q, k, v, sel, block_size)
        LW, NLB = _local_meta(BS, local_window)
        M = triton.cdiv(T, BS)
        do = do.contiguous()
        delta = (o.float() * do.float()).sum(-1)  # [B, T, HQ]

        dq = torch.empty_like(q)
        d_bias = torch.zeros(B, T, M, dtype=torch.float32, device=q.device)
        _e2e_attn_bwd_dq_kernel[(T, 1, B * H)](
            q, k, v, lse, delta, do, dq, scale, sel, blk_bias, d_bias,
            T=T, H=H, HQ=HQ, G=G, GP=GP, K=K, V=V, S=S, M=M, BS=BS, BK=BK, BV=BV,
            LW=LW, NLB=NLB, num_warps=4,
        )

        block_mask = torch.zeros(B, T, M + 1, dtype=torch.bool, device=q.device)
        block_mask.scatter_(-1, sel.long().clamp(min=0).masked_fill(sel < 0, M), True)
        block_mask = block_mask[..., :M].contiguous()

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _e2e_attn_bwd_dkv_kernel[(1, M, B * H)](
            q, k, v, lse, delta, do, dk, dv, block_mask, blk_bias, scale,
            T=T, H=H, HQ=HQ, G=G, GP=GP, K=K, V=V, M=M, BS=BS, BK=BK, BV=BV,
            LW=LW, num_warps=4,
        )
        return dq, dk, dv, None, d_bias, None, None, None


def triton_gather_attention_e2e(
    query_states,   # [B, HQ, T, D]
    key_states,     # [B, Hkv, Sk, D]
    value_states,   # [B, Hkv, Sk, D]
    block_indices,  # [B, 1, T, S] int, -1 = padding
    blk_bias_slots,  # [B, T, S] float, differentiable per-selected-block bias (aligned with block_indices)
    position_ids,   # [B, T]
    attention_mask,
    block_size,
    scale,
    local_window=None,
):
    """Fused e2e gather-attention entry. Returns [B, HQ, T, D], or None on unsupported
    shape (caller must fall back to the eager torch path). Gradients flow to q/k/v AND
    to blk_bias_slots (-> the indexer projections)."""
    if attention_mask is not None:
        return None
    if block_indices.shape[1] != 1:
        return None
    B, HQ, T, D = query_states.shape
    if key_states.shape[2] != T:
        return None
    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[-1] != T:
        return None
    ar = torch.arange(T, device=position_ids.device)
    if not bool((position_ids == ar).all()):
        return None

    sel = block_indices[:, 0].contiguous()      # [B, T, S]
    bias_slots = blk_bias_slots                  # [B, T, S] (grad)
    M = triton.cdiv(T, block_size)

    # dedup repeated block ids within a row (would double-count softmax mass).
    valid = sel >= 0
    sel_sorted, order = sel.sort(dim=-1)
    dup_sorted = torch.zeros_like(valid)
    dup_sorted[..., 1:] = sel_sorted[..., 1:] == sel_sorted[..., :-1]
    dup = torch.zeros_like(valid)
    dup.scatter_(-1, order, dup_sorted)
    sel = sel.masked_fill(dup & valid, -1)

    # scatter per-slot bias into a dense [B, T, M] tensor keyed by block id (differentiable
    # w.r.t. bias_slots). -1 slots (pad/dedup) route to an overflow column then dropped.
    scatter_idx = sel.long().clamp(min=0).masked_fill(sel < 0, M)   # [B, T, S]
    blk_bias_dense = bias_slots.new_zeros(B, T, M + 1).scatter(-1, scatter_idx, bias_slots)
    blk_bias_dense = blk_bias_dense[..., :M].contiguous()

    sel = sel.to(torch.int32).contiguous()
    q = query_states.transpose(1, 2).contiguous()  # [B, T, HQ, D]
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()
    o = _E2EGatherAttnFn.apply(q, k, v, sel, blk_bias_dense, block_size, scale, local_window)
    return o.transpose(1, 2)  # [B, HQ, T, D]
