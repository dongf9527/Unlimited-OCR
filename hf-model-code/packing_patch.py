"""Sequence-packing support for the sparse-attention path.

Ported from Megatron-Bridge (``deepseek_ocr2/modeling_deepseek_ocr_mcore.py``):
the two-stage warmup/sparse pipeline there runs with packing, and does so by
making every causal / local-window / block mask *segment aware* instead of
assuming a single contiguous sequence per row.

Why this is needed
-------------------
With packing, ms-swift concatenates several samples into one row and emits
per-sample-RESET ``position_ids``  (``[0,1,..,L1-1, 0,1,..,L2-1, ...]``; see
``swift/template/base.py::packing_row``). The sparse code compares the query
``position_ids`` against ``k_positions = arange(k_len)`` (the GLOBAL flat key
index) to build causal / local masks. With reset positions that comparison is
wrong (a sample-2 query at local pos 0 would be "allowed" to see only key 0,
which belongs to sample 1), so packing silently corrupts attention.

Fix (mirrors Megatron-Bridge)
-----------------------------
* Derive a per-token SEGMENT id from the ``position_ids`` resets.
* Evaluate causal / local windows in GLOBAL key space using the flat query
  index (for a packed row with batch==1 the flat index *is* the global
  position, and ``k_positions`` is already global).
* AND every keep-mask with a ``same_segment`` constraint so no query attends
  across a sample boundary.
* Bypass the fused Triton kernels when packed (they assume ``position_ids ==
  arange`` and a single segment); the eager PyTorch paths below are made
  segment aware instead.

batch==1 is assumed for packed rows (``padding_free``/packing collapses the
batch dimension into the sequence).

Block-grid alignment (sparse stage)
-----------------------------------
Key blocks are cut on the FLAT (global) token index, so a sample that does not
start on a ``block_size`` boundary gets a block grid shifted by
``start % block_size`` relative to single-sample inference. That is safe (the
segment masks above still prevent any cross-sample attention) but it is a
train/inference discrepancy in *how the keys are grouped*, plus the first block
of each sample straddles the previous one and gets trimmed. Run packing with
``PACKING_ALIGN_SIZE=<block_size>`` (ms-swift ``Template._align_packing_row``)
to make every sample start on a block boundary; ``check_block_alignment`` warns
once when it is not the case.
"""
import torch


def is_packed(position_ids):
    """A packed row restarts ``position_ids`` at 0 at each sample boundary, so a
    0 appears at some column >= 1. A plain (non-packed) sequence only has its
    single 0 at column 0. Robust for training (no left padding under
    padding_free)."""
    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[1] < 2:
        return False
    return bool((position_ids[:, 1:] == 0).any())


def segment_ids(position_ids):
    """[B, S] long: 0-based segment index of every token, from position_ids
    resets. ``(position_ids == 0)`` marks each segment start; cumsum-1 numbers
    them. For a non-packed row this is all zeros (single segment)."""
    return (position_ids == 0).to(torch.long).cumsum(dim=1) - 1


def packed_masks(position_ids, k_positions, q_index, local_window):
    """All masks are built in GLOBAL key space and are [B, 1, C, Sk] bool.

    Args:
        position_ids: [B, S] reset positions (training prefill: S == k_len).
        k_positions:  [Sk]   global key indices == ``arange(k_len)``.
        q_index:      [C]    global (flat) indices of the query rows in this
                             chunk (e.g. ``arange(c0, c1)`` or the strided
                             ``sel`` used by the KL loss). These ARE the global
                             query positions for a packed batch==1 row.
        local_window: int    inclusive local window length in tokens.

    Returns dict with:
        token_future [B,1,C,Sk]  key is strictly after the query (to mask out)
        local_token  [B,1,C,Sk]  key in [gpos-local_window+1, gpos]
        same_seg     [B,1,C,Sk]  key and query share a segment
        gpos         [B,C]       global query positions
    """
    device = position_ids.device
    B = position_ids.shape[0]
    Sk = k_positions.shape[0]
    seg = segment_ids(position_ids)                       # [B, S]
    seg_k = seg[:, :Sk]                                   # [B, Sk]
    q_index = q_index.to(device=device, dtype=torch.long)
    seg_q = seg.index_select(1, q_index)                 # [B, C]
    gpos = q_index.view(1, -1).expand(B, -1)             # [B, C] global q pos

    kp = k_positions.view(1, 1, 1, -1)                    # [1,1,1,Sk]
    gpos_e = gpos[:, None, :, None]                       # [B,1,C,1]
    token_future = kp > gpos_e
    ls = (gpos - (local_window - 1)).clamp(min=0)[:, None, :, None]
    local_token = (kp >= ls) & (kp <= gpos_e)
    same_seg = seg_k[:, None, None, :] == seg_q[:, None, :, None]
    return {
        'token_future': token_future,
        'local_token': local_token,
        'same_seg': same_seg,
        'gpos': gpos,
    }


def segment_of(position_ids, token_indices):
    """Gather the segment id for arbitrary GLOBAL token indices.

    Args:
        position_ids: [B, S]
        token_indices: [B, ...] long, global key indices (clamped by caller).
    Returns tensor shaped like ``token_indices`` with the segment id of each.
    """
    seg = segment_ids(position_ids)                       # [B, S]
    flat = token_indices.reshape(token_indices.shape[0], -1)
    out = seg.gather(1, flat.clamp(min=0, max=seg.shape[1] - 1))
    return out.reshape(token_indices.shape)


def segment_starts(position_ids):
    """[N] long: GLOBAL start index of every segment in a packed row (batch 0).
    ``start = flat_index - position_ids`` at the reset points."""
    return (position_ids[0] == 0).nonzero(as_tuple=True)[0]


_ALIGN_WARNED = False


def check_block_alignment(position_ids, block_size):
    """one-shot warning when a packed row's samples do NOT start on a
    ``block_size`` boundary.

    The sparse stage cuts its key blocks on the FLAT (global) token index, so a sample
    starting at ``s`` gets a block grid shifted by ``s % block_size`` relative to
    single-sample inference (which starts at 0), and its first block straddles the previous
    sample (the segment mask then trims it, wasting part of the remote top-k budget).
    Correctness is unaffected (no cross-sample leakage), but the train/inference block
    partitioning differs. Set ``PACKING_ALIGN_SIZE=<block_size>`` so ms-swift's
    ``packing_row`` pads every sample up to that multiple and the two grids coincide.
    """
    global _ALIGN_WARNED
    if _ALIGN_WARNED or position_ids is None or block_size <= 1:
        return
    starts = segment_starts(position_ids)
    n_bad = int((starts % block_size != 0).sum())
    if n_bad:
        _ALIGN_WARNED = True
        print(f'[PackingAlign] {n_bad}/{starts.numel()} packed samples do not start on a '
              f'{block_size}-token block boundary -> block grid is shifted w.r.t. '
              f'single-sample inference. Set PACKING_ALIGN_SIZE={block_size} to align.',
              flush=True)


def dense_segment_keep(position_ids, device):
    """Boolean [B, 1, S, S] keep-mask for the DENSE (warmup) main branch:
    True where a query may attend (same segment AND causal). Passed straight to
    ``scaled_dot_product_attention(attn_mask=...)`` (bool: True == attend).
    Mirrors the ``same_seg & causal`` SDPA mask of Megatron-Bridge's dense path.

    NOTE: this materialises a [B, 1, S, S] bool tensor (~1 GiB at S=32k, B=1).
    A chunked-query SDPA (as in MB's ``modeling_deepseek_ocr_mcore`` cp>1 loop)
    would avoid it and is a follow-up memory optimisation."""
    B, S = position_ids.shape
    seg = segment_ids(position_ids)                       # [B, S]
    pos = torch.arange(S, device=device)                  # global positions
    causal = pos[None, :] <= pos[:, None]                 # [S, S] q >= k
    same_seg = seg[:, :, None] == seg[:, None, :]         # [B, S, S]
    return (same_seg & causal[None])[:, None]             # [B, 1, S, S] bool
