"""Pure-PyTorch, fully-differentiable DASH attention for Unlimited-OCR.

VARIANT B: entmax routing + HARD top-k budget + PER-LAYER entmax temperature
(see the banner below the imports for the full B-vs-base rationale).

DASH = Differentiable and Adaptive Sparse Hierarchical attention
(https://arxiv.org/abs/2605.18753). Unlike the Lightning-Indexer + top-k DSA
path of the base implementation (whose block selection is DISCRETE and therefore
needs a separate warmup + KL-distillation stage to train the indexer), DASH
routes with **1.5-entmax**, which yields a sparse *and differentiable*
distribution over key blocks. Gradient reaches the router directly from the LM
loss, so no KL teacher / no two-stage curriculum is required.

This module deliberately uses **plain PyTorch ops only** (no Triton). The only
custom autograd is the tiny 1.5-entmax Jacobian (a standard, well-known
closed form over the block axis) -- everything else is ordinary autograd.
This is the "correctness-first" realization: it is O(S^2) in the attention
stage (computed in query chunks to bound peak memory), so it is meant for
training / convergence validation, not for maximum throughput. A fused sparse
kernel is a later optimization once the recipe is validated.

Three stages (mirroring the paper and ``dash-attention-main``):
  stage1  summarize_blocks : one learned key summary per KV block, via a
                             softmax attention with a learnable ``head_cls``.
  stage2  route            : query->summary scores -> 1.5-entmax over blocks
                             (adaptive, differentiable sparse "prior").
  stage3  prior_biased_attn: token-level softmax attention restricted to the
                             routed blocks UNION the local window, with
                             ``log(prior)`` injected as an additive bias so the
                             hierarchy stays differentiable end-to-end.

Optional image-block restriction: when an image-block mask is supplied, the routing
candidate set is restricted to image blocks (any page), matching the
``image_bucket_constraint=False`` semantics of the base implementation.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_DASH_DEBUG = os.environ.get("DASH_DEBUG_ANOMALY") == "1"

# ===========================================================================
# VARIANT B -- entmax + hard top-k + per-layer temperature
# (overlay on patch_config/sparse_dash_code)
#
# Differences vs the base DASH overlay:
#   * The attended set is a HARD top-k, k annealed 64 -> 12 (`_KSchedule`), so
#     every query in every layer reads exactly k remote blocks. The base overlay
#     attended the entmax support, whose size varied per layer (13-27 at a mean
#     of 16) and per step.
#   * entmax is KEPT: it still produces the differentiable prior whose
#     mean-centered log is the routing bias, so this stays faithful to the paper's
#     adaptive weighting -- only the cardinality is now fixed.
#   * Each layer runs its OWN closed loop on its OWN entmax temperature, with the
#     current k as setpoint (`_PerLayerScheduler`). Once the count is fixed the
#     temperature's job changes: it has to keep each layer's entmax support ~= k,
#     otherwise the kept blocks either sit at the prior floor (support << k, no
#     gradient) or get an almost flat prior (support >> k, no ranking signal).
#     The base overlay's single global scale could only pin the across-layer mean.
#   * Per-layer one-shot calibration replaces the leader-only calibration, so
#     every layer starts at the initial k instead of inheriting layer 0's
#     temperature.
# Variant A (patch_config/dash_softmax_topk) is the same budget without entmax.
# ===========================================================================
# DASH_LOG_SPARSITY=1 -> print entmax routing sparsity (how many blocks/tokens
# the router keeps vs drops) during training, throttled to every
# DASH_LOG_SPARSITY_EVERY calls OF EACH LAYER (so every layer shows up in the log;
# a single global counter aliased against the layer count and only ever printed
# the same few layers).
_LOG_SPARSITY = os.environ.get("DASH_LOG_SPARSITY", "0") == "1"
_SPARSITY_EVERY = max(1, int(os.environ.get("DASH_LOG_SPARSITY_EVERY", "50")))
_sparsity_calls = {}


def _log_train_sparsity(prior, cand, valid, block_size, sid=0):
    """Print this layer's entmax support vs the hard k budget it is aimed at.

    ``attn`` is what the layer actually reads (== min(k, #candidates), guaranteed
    by the top-k); ``act`` is the entmax support size, the quantity this layer's
    own closed loop drives onto k. act << attn means most kept blocks sit at the
    prior floor and carry no gradient; act >> attn means the prior is flat over
    the kept set. Both are per-layer, hence the L<sid> tag.
    """
    n = _sparsity_calls.get(sid, 0) + 1
    _sparsity_calls[sid] = n
    # Always print each layer's FIRST call: with EVERY=N the first line of a layer
    # would otherwise take N micro-batches, so a run that dies early (or a short
    # smoke test) showed no per-layer sparsity at all.
    if n != 1 and n % _SPARSITY_EVERY != 0:
        return
    with torch.no_grad():
        nblk = prior.shape[-1]
        c = cand.sum(dim=-1).float()                   # causal candidate blocks/query
        a = ((prior > 0) & cand).sum(dim=-1).float()   # entmax support/query
        t = valid.sum(dim=-1).float()                  # blocks actually attended
        cm, am, tm = c.mean().item(), a.mean().item(), t.mean().item()
        drop = (1.0 - t.sum() / c.clamp(min=1).sum()).item() * 100.0
        print(f"[DASH sparsity L{sid}] call#{n}: attn {tm:.1f}/{cm:.1f} cand "
              f"blocks/query (k={_KS.k}, drop {drop:.1f}%); entmax act={am:.1f}; "
              f"total_blocks={nblk}; attn ~{tm * block_size:.0f} remote tok/query "
              f"(block_size={block_size}){_SCHED.state_str(sid)}", flush=True)


# DASH_LOG_NEFF_EVERY>0 -> print, per layer, how many of the k gathered blocks
# the ATTENTION actually uses. `sparsity` only reports how many blocks the
# ROUTER selected; it cannot see that the log(prior) bias spans ~9 (floor 1e-4,
# sigma 1) while qk/sqrt(d) spans ~3-6, so the bias is free to veto a block
# top-k already paid to read. n_eff = exp(H(w)) with w = each block's share of
# the remote attention mass: n_eff ~= 2 at k=12 means 10 of the 12 gathered
# blocks are dead weight -- a failure the temperature loop cannot observe.
# remote_mass = the remote blocks' share of (remote + local window) mass; n_eff
# means nothing when the row spends all its mass on the free local context.
# Cost: one no_grad [R, K*block_size] qk per print. ROWS query rows are sampled
# and processed CHUNK rows at a time, because these temporaries live alongside
# the training activations (S=32k is already tight).
_LOG_NEFF_EVERY = max(0, int(os.environ.get("DASH_LOG_NEFF_EVERY", "0")))
_LOG_NEFF_ROWS = max(1, int(os.environ.get("DASH_LOG_NEFF_ROWS", "64")))
_LOG_NEFF_CHUNK = max(1, int(os.environ.get("DASH_LOG_NEFF_CHUNK", "8")))
_neff_calls = {}


def _log_neff(q, k, topi, blk_bias, valid, position_ids, block_size,
              local_window, scale, sid=0):
    """Effective block utilisation of the k blocks this layer gathered.

    Same throttling convention as _log_train_sparsity: per-layer counter, the
    first call of every layer always prints. Batch element 0 only.
    """
    n = _neff_calls.get(sid, 0) + 1
    _neff_calls[sid] = n
    if n != 1 and n % _LOG_NEFF_EVERY != 0:
        return
    with torch.no_grad():
        _, H, S, D = q.shape
        K = topi.shape[-1]
        b = 0
        rows = torch.nonzero(valid[b].any(dim=-1)).flatten()
        if rows.numel() == 0:
            print(f"[DASH n_eff L{sid}] call#{n}: no remote selection "
                  f"(k={_KS.k})", flush=True)
            return
        if rows.numel() > _LOG_NEFF_ROWS:
            rows = rows[::max(1, rows.numel() // _LOG_NEFF_ROWS)][:_LOG_NEFF_ROWS]
        ar = torch.arange(block_size, device=q.device)
        aw = torch.arange(local_window, device=q.device)
        neff = top1 = rmass = 0.0
        for c0 in range(0, rows.numel(), _LOG_NEFF_CHUNK):
            r = rows[c0:c0 + _LOG_NEFF_CHUNK]
            R = r.numel()
            qpos = position_ids[b, r].long()                          # [R]
            qi = q[b][:, r, :]                                        # [H,R,D]
            # remote: the exact blocks the kernel/dense path reads, same bias.
            pos = (topi[b][r].clamp_min(0)[:, :, None] * block_size + ar
                   ).reshape(R, -1)                                   # [R,K*bs]
            keep = ((pos <= qpos[:, None]) & (pos < S)
                    & valid[b][r][:, :, None].expand(R, K, block_size
                                                     ).reshape(R, -1))
            kk = k[b][:, pos.clamp(0, S - 1).reshape(-1), :].view(
                H, R, K * block_size, D)
            lg = torch.einsum("hrd,hrnd->hrn", qi, kk).float() * scale
            lg = lg + blk_bias[b][r].float()[:, :, None].expand(
                R, K, block_size).reshape(R, -1)
            lg = lg.masked_fill(~keep, float("-inf"))
            # local window: free context the remote blocks compete against.
            lpos = qpos[:, None] - aw                                 # [R,W]
            lkk = k[b][:, lpos.clamp_min(0).reshape(-1), :].view(
                H, R, local_window, D)
            llg = torch.einsum("hrd,hrnd->hrn", qi, lkk).float() * scale
            llg = llg.masked_fill(~(lpos >= 0), float("-inf"))
            # shared max: remote and local must be softmaxed in one denominator.
            m = torch.maximum(lg.amax(-1), llg.amax(-1)).clamp_min(-1e30)
            er = torch.nan_to_num(torch.exp(lg - m[..., None]))
            el = torch.nan_to_num(torch.exp(llg - m[..., None]))
            zr = er.view(H, R, K, block_size).sum(-1)                 # [H,R,K]
            wr = zr / zr.sum(-1, keepdim=True).clamp_min(1e-20)
            ent = -(wr * torch.log(wr.clamp_min(1e-20))).sum(-1)      # [H,R]
            neff += torch.exp(ent).sum().item()
            top1 += wr.amax(-1).sum().item()
            tot = (er.sum(-1) + el.sum(-1)).clamp_min(1e-20)
            rmass += (er.sum(-1) / tot).sum().item()
        cnt = float(H * rows.numel())
        print(f"[DASH n_eff L{sid}] call#{n}: n_eff={neff / cnt:.2f} of K={K} "
              f"(k={_KS.k}); top1_blk={top1 / cnt:.3f}; "
              f"remote_mass={rmass / cnt:.3f}; rows={rows.numel()}", flush=True)


# DASH_LOG_RECALL_EVERY>0 -> per layer, the paper-style ORACLE recall of the
# router's top-k selection (same definition as the DSA report's Fig.3 and the
# `_log_indexer_recall` of the base implementation):
#
#   I*    = oracle top-k blocks, ranked by the DENSE attention mass P_b that the
#           main branch (q@k) puts on block b.
#   I_hat = the blocks the router actually selected (`topi`, top-k over the prior).
#   block recall@k = |I* ^ I_hat| / k
#   score recall@k = sum_{b in I* ^ I_hat} P_b / sum_{b in I*} P_b
#
# Both are measured on the SAME domain the selector uses (candidate blocks only:
# causal, outside the local window, image-constrained if dash_image_only), so
# P_b is a softmax normalized over the candidate tokens and mass_all == 1.
# student_kept / teacher_kept are the same numbers without the intersection:
# what fraction of the candidate mass the router's k blocks capture vs. what the
# best possible k blocks capture -- their gap is the pure ranking loss, which is
# easier to read than either absolute value.
# Only CONTESTED rows count (n_cand > k); when cand <= k the top-k trivially
# takes everything and recall is 1.0 by construction, which would inflate the
# mean. Chance level is ~k/cand_q, so always read recall against the printed
# cand/q.
# Cost: one no_grad [H, chunk, S] qk per print (the full dense row, unlike n_eff
# which only re-reads the k gathered blocks), so keep EVERY large and ROWS small.
_LOG_RECALL_EVERY = max(0, int(os.environ.get("DASH_LOG_RECALL_EVERY", "0")))
_LOG_RECALL_ROWS = max(1, int(os.environ.get("DASH_LOG_RECALL_ROWS", "32")))
_LOG_RECALL_CHUNK = max(1, int(os.environ.get("DASH_LOG_RECALL_CHUNK", "4")))
_recall_calls = {}


def _log_oracle_recall(q, k, topi, valid, cand, position_ids, block_size,
                       scale, sid=0):
    """Oracle (dense-attention) recall of this layer's routed block set.

    Same throttling convention as the other diagnostics (per-layer counter, each
    layer's first call always prints). Batch element 0 only, rank 0 only unless
    DASH_LOG_RECALL_ALL_RANKS=1.
    """
    n = _recall_calls.get(sid, 0) + 1
    _recall_calls[sid] = n
    if n != 1 and n % _LOG_RECALL_EVERY != 0:
        return
    if os.environ.get("DASH_LOG_RECALL_ALL_RANKS", "0") != "1":
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    with torch.no_grad():
        _, H, S, D = q.shape
        K = topi.shape[-1]
        b = 0
        nblk = cand.shape[-1]
        pad = nblk * block_size - S
        # A row is only meaningful if the router had a real choice to make.
        ncand = cand[b].sum(dim=-1)                                   # [S]
        rows = torch.nonzero((ncand > K) & valid[b].any(dim=-1)).flatten()
        if rows.numel() == 0:
            print(f"[DASH recall L{sid}] call#{n}: no contested row "
                  f"(k={_KS.k}, K={K})", flush=True)
            return
        if rows.numel() > _LOG_RECALL_ROWS:
            rows = rows[::max(1, rows.numel() // _LOG_RECALL_ROWS)][:_LOG_RECALL_ROWS]
        kpos = torch.arange(S, device=q.device)
        br = sr = kept = teach = t1 = 0.0
        cnt = 0
        for c0 in range(0, rows.numel(), _LOG_RECALL_CHUNK):
            r = rows[c0:c0 + _LOG_RECALL_CHUNK]
            R = r.numel()
            qpos = position_ids[b, r].long()                           # [R]
            cnd = cand[b][r]                                           # [R,nblk]
            # candidate blocks -> candidate tokens, then re-impose causality
            # (the last candidate block can extend past the query for pad rows).
            keep = cnd.repeat_interleave(block_size, dim=-1)[:, :S] \
                & (kpos[None, :] <= qpos[:, None])                     # [R,S]
            lg = torch.einsum("hrd,hsd->hrs", q[b][:, r, :], k[b]).float() * scale
            lg = lg.masked_fill(~keep[None], float("-inf"))
            p = torch.softmax(lg, dim=-1)
            p = torch.nan_to_num(p)                                    # empty rows
            if pad:
                p = F.pad(p, (0, pad), value=0.0)
            # router selection is head-shared, so the oracle is the head-mean
            # block distribution (one teacher per query row).
            p_blk = p.view(H, R, nblk, block_size).sum(-1).mean(0)     # [R,nblk]
            p_msk = p_blk.masked_fill(~cnd, -1.0)
            star = p_msk.topk(K, dim=-1).indices
            star_m = torch.zeros_like(cnd).scatter_(-1, star, True) & cnd
            # invalid (padding) selection slots are routed to a scratch column so
            # they cannot be mistaken for a hit on block 0.
            hidx = torch.where(valid[b][r], topi[b][r].clamp_min(0),
                               torch.full_like(topi[b][r], nblk))
            hat_m = torch.zeros((R, nblk + 1), dtype=torch.bool, device=q.device
                                ).scatter_(-1, hidx, True)[:, :nblk] & cnd
            inter = star_m & hat_m
            m_all = (p_blk * cnd).sum(-1).clamp_min(1e-9)
            m_star = (p_blk * star_m).sum(-1)
            ok = m_star > 0
            br += (inter.sum(-1).float() / K)[ok].sum().item()
            sr += ((p_blk * inter).sum(-1) / m_star.clamp_min(1e-9))[ok].sum().item()
            kept += ((p_blk * hat_m).sum(-1) / m_all)[ok].sum().item()
            teach += (m_star / m_all)[ok].sum().item()
            t1 += (p_msk.argmax(-1) == topi[b][r][:, 0])[ok].float().sum().item()
            cnt += int(ok.sum().item())
        if cnt == 0:
            return
        print(f"[DASH recall L{sid}] call#{n}: k={_KS.k} K={K} "
              f"block_recall@k={br / cnt:.4f} score_recall@k={sr / cnt:.4f}; "
              f"student_kept={kept / cnt:.4f} oracle_kept={teach / cnt:.4f} "
              f"top1={t1 / cnt:.4f}; cand/q={ncand[rows].float().mean().item():.1f} "
              f"contested_rows={cnt}/{rows.numel()} nblk={nblk}", flush=True)


# ---------------------------------------------------------------------------
# Sparsity scheduler -- opt-in via DASH_SPARSITY_SCHED=1 (default OFF, in which
# case every value below is ignored and behaviour is bit-identical to before).
#
# WHY the entmax scale is the sparsity knob: `_route` feeds
#   block_scores * dash_scaling_factor / sqrt(index_dim)
# to 1.5-entmax, and entmax's support size is governed by the ABSOLUTE scale of
# its input (scale -> 0 degenerates to uniform/dense, scale up shrinks the
# support). The shipped 1.0/sqrt(128) ~= 0.088 flattens the score gaps, so the
# prior stays near-dense. Ramping the scale up = annealing a temperature down.
#
# WHY prior_floor must follow: the router's only gradient channel is
#   bias = (log(clamp(p, prior_floor)) - mu) / sigma
# If the active p_j fall below `prior_floor`, every logp clamps to the same
# log(floor), the mean-centered bias collapses to 0 and the router stops
# learning. So the floor is annealed DOWN geometrically as the prior sharpens.
#
# Step unit = one training micro-batch forward. Gradient-checkpoint recompute
# passes are detected (they run inside a backward graph task) and NOT counted,
# and only the first-registered layer advances the clock, so every DASH layer
# shares one schedule.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_LOG_PAGE_EVERY = max(0, int(os.environ.get("DASH_LOG_PAGE_EVERY", "0")))
_LOG_PAGE_ROWS = max(1, int(os.environ.get("DASH_LOG_PAGE_ROWS", "64")))


@torch.no_grad()
def _log_page_purity(topi, valid, cand, remote, img_page_ids, query_pages,
                     block_size, nblk, sid=0, n_call=0):
    if img_page_ids is None or query_pages is None:
        return
    S = topi.shape[1]
    step = max(1, S // _LOG_PAGE_ROWS)
    rows = torch.arange(0, S, step, device=topi.device)[:_LOG_PAGE_ROWS]
    ip = img_page_ids[:1]
    pad = nblk * block_size - ip.shape[-1]
    if pad > 0:
        ip = F.pad(ip, (0, pad), value=-1)
    ip_blk = ip[..., :nblk * block_size].view(nblk, block_size)
    ti, vi = topi[0, rows], valid[0, rows]                    # [R,K]
    qp = query_pages[0, rows][:, None, None]                  # [R,1,1]
    bp = ip_blk[ti]                                           # [R,K,block_size]
    is_img = bp >= 0
    n = int(vi.sum().clamp(min=1))
    nonimg = int(((is_img.sum(-1) == 0) & vi).sum())
    other = (is_img & (bp != qp)).sum(-1)
    same = ((bp == qp) & (qp >= 0)).sum(-1)
    cross = int(((other > 0) & vi).sum())
    pure = int(((other > 0) & (same == 0) & vi).sum())
    empty = float((~(cand[0, rows] & remote[0, rows]).any(-1)).float().mean() * 100)
    no_page = float((query_pages[0, rows] < 0).float().mean() * 100)
    print(f"[DASH page L{sid}] call#{n_call}: rows={len(rows)} sel={n} "
          f"non_image={100 * nonimg / n:.2f}% cross_block={100 * cross / n:.2f}% "
          f"pure_cross={100 * pure / n:.2f}% cand_empty={empty:.1f}% "
          f"no_page={no_page:.1f}%", flush=True)


_SCHED_ON = os.environ.get("DASH_SPARSITY_SCHED", "0") == "1"


def _envf(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


def _envi(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)


# ---------------------------------------------------------------------------
# Hard top-k budget with a dense-to-sparse k ANNEAL.
#
# The routed-block count is a HARD, deterministic budget: every query attends
# min(k, #candidates) remote blocks in EVERY layer, so "same number of active
# blocks in every layer at every step" holds by construction. entmax still runs
# (it produces the differentiable weights on the kept blocks), and each layer's
# entmax temperature is closed-looped onto this same k -- see
# `_PerLayerScheduler` for why that matters once the count is fixed.
#
# k anneals geometrically K_START -> K_END over K_ANNEAL_STEPS optimizer steps.
# WHY anneal at all when the budget is exact: the base checkpoint was pretrained
# dense, and 12 blocks is ~2.3% of the KV at 16k. Truncating that hard at step 0
# with lr=1e-4 full-finetune and no warmup perturbs the representation before the
# router has learned to rank blocks; starting wide gives the router a usable
# gradient signal while the model is still close to its pretrained state.
#
# k MUST be constant for a whole micro-batch: with gradient checkpointing the
# layers are recomputed in backward, and a k that moved mid-forward changes the
# gathered width -> `CheckpointError: recomputed values have different metadata`.
# So it is refreshed ONLY at a micro-batch edge, by the leader layer.
# ---------------------------------------------------------------------------
class _KSchedule:
    # gc scans are not free, so only probe for the TrainerState a few times.
    _PROBE_AT = (1, 4, 16, 64, 256)

    def __init__(self):
        self.k_start = _envi("DASH_K_START", 64)
        self.k_end = _envi("DASH_K_END", 12)
        self.steps = max(1, _envi("DASH_K_ANNEAL_STEPS", 1000))
        self.log_every = max(1, _envi("DASH_K_LOG_EVERY", 50))
        self._k = self.k_start
        self._micro = 0
        self._step = 0
        self._state = None
        self._leader = None
        self._n_reg = 0
        self._logged = False

    def register(self):
        """Returns an id; the first caller becomes the clock leader."""
        sid = self._n_reg
        self._n_reg += 1
        if self._leader is None:
            self._leader = sid
        return sid

    def _probe_state(self):
        """Locate the live HF TrainerState (gives the optimizer-step clock)."""
        try:
            import gc
            for o in gc.get_objects():
                if (type(o).__name__ == "TrainerState"
                        and int(getattr(o, "max_steps", 0) or 0) > 0):
                    return o
        except Exception:
            pass
        return None

    def k_for(self, step):
        """Geometric interpolation K_START -> K_END, clamped to the endpoints."""
        p = min(1.0, max(0.0, step / float(self.steps)))
        k = self.k_start * (self.k_end / float(self.k_start)) ** p
        lo, hi = min(self.k_start, self.k_end), max(self.k_start, self.k_end)
        return max(lo, min(hi, int(round(k))))

    def advance(self, sid, training):
        """Refresh k once per micro-batch; skip gradient-checkpoint recompute."""
        if not training or sid != self._leader:
            return
        if torch._C._current_graph_task_id() != -1:      # inside backward
            return
        self._micro += 1
        if self._state is None and self._micro in self._PROBE_AT:
            self._state = self._probe_state()
        self._step = (int(self._state.global_step) if self._state is not None
                      else self._micro)
        self._k = self.k_for(self._step)
        if not self._logged:
            self._logged = True
            print(f"[DASH k-sched] hard top-k: {self.k_start} -> {self.k_end} "
                  f"geometric over {self.steps} steps "
                  f"(clock={'TrainerState' if self._state is not None else 'micro-batch'})",
                  flush=True)
        elif self._micro % self.log_every == 0:
            print(f"[DASH k-sched] step={self._step} k={self._k}", flush=True)

    @property
    def k(self):
        return self._k


_KS = _KSchedule()


class _PerLayerScheduler:
    """Per-layer entmax temperature, closed-looped onto the CURRENT k budget.

    Variant B keeps entmax (adaptive, differentiable sparse prior) but the
    attended set is a hard top-k, so entmax no longer decides HOW MANY blocks are
    read -- it decides how the kept blocks are WEIGHTED, and therefore how much
    gradient each one carries. Two failure modes follow, both per-layer:

      * support << k -> most of the k kept blocks have prior 0, their bias clamps
        to log(prior_floor) and they contribute no gradient.
      * support >> k -> the prior is flat over the kept blocks, so the bias
        carries almost no ranking information.

    So each layer runs its OWN loop driving its entmax support size onto the
    current k (``_KS.k``, which itself anneals 64->12). The base overlay shared
    ONE global scale and could only pin the across-layer MEAN: the per-layer
    counts spread 13-27 at a mean of 16, i.e. some layers saturated while others
    stayed flat. ``prior_floor`` follows each layer's own scale geometrically.

    The scale MUST be constant for a whole micro-batch: with gradient
    checkpointing the layers are recomputed in backward, and a scale that moved
    mid-forward changes the entmax support -> a different bias -> the recompute
    disagrees with the saved forward. So every layer refreshes its scale at most
    once per micro-batch, from the PREVIOUS micro-batch's measurement.
    """

    # gc scans are not free, so only probe for the TrainerState a few times.
    _PROBE_AT = (1, 4, 16, 64, 256)

    def __init__(self):
        self.on = os.environ.get("DASH_SPARSITY_SCHED", "0") == "1"
        self.scale_max = _envf("DASH_SCHED_SCALE_MAX", 128.0)
        self.scale_min = _envf("DASH_SCHED_SCALE_MIN", 0.05)
        self.log_every = max(1, _envi("DASH_SCHED_LOG_EVERY", 50))
        self.gain = _envf("DASH_SCHED_GAIN", 0.1)              # log-domain P gain
        self.ema = _envf("DASH_SCHED_EMA", 0.9)                # measurement smoothing
        self.max_step_ratio = _envf("DASH_SCHED_MAX_STEP", 1.05)  # per-step cap
        # one-shot PER-LAYER calibration: bisect this layer's scale on its first
        # real batch so it starts exactly at the initial k, instead of inheriting
        # a temperature tuned for another layer's score spread.
        self.calibrate_on = _envi("DASH_SCHED_CALIB", 1) == 1
        self.calib_rows = max(16, _envi("DASH_SCHED_CALIB_ROWS", 512))
        self.floor_start = None      # resolved from config on first register()
        self.floor_end = _envf("DASH_SCHED_FLOOR_END", 1.0e-4)
        self.scale_init = None
        self.scale_ref = None        # floor interpolates over [init, ref]
        self._scale = {}             # sid -> entmax scale
        self._floor = {}             # sid -> prior floor
        self._measured = {}          # sid -> EMA of entmax-active blocks/query
        self._pending = {}           # sid -> newest raw measurement
        self._micro = {}             # sid -> own micro-batch counter
        self._calibrated = set()
        self._step = 0
        self._state = None
        self._leader = None
        self._n_reg = 0
        self._logged = False
    # -- setup ---------------------------------------------------------------
    def register(self, cfg_scale, cfg_floor):
        """Returns an id; the first caller becomes the clock leader."""
        if self.scale_init is None:
            self.scale_init = _envf("DASH_SCHED_SCALE_START", cfg_scale)
            self.scale_ref = _envf("DASH_SCHED_SCALE_END", 8.0)
            self.floor_start = _envf("DASH_SCHED_FLOOR_START", cfg_floor)
        sid = self._n_reg
        self._n_reg += 1
        if self._leader is None:
            self._leader = sid
        self._scale[sid] = self.scale_init
        self._floor[sid] = self.floor_start
        self._micro[sid] = 0
        return sid

    # -- clock ---------------------------------------------------------------
    def _probe_state(self):
        """Locate the live HF TrainerState (gives the optimizer-step clock)."""
        try:
            import gc
            for o in gc.get_objects():
                if (type(o).__name__ == "TrainerState"
                        and int(getattr(o, "max_steps", 0) or 0) > 0):
                    return o
        except Exception:
            pass
        return None

    def advance(self, sid, training):
        """Refresh THIS layer's scale; skip gradient-checkpoint recompute."""
        if not self.on or not training:
            return
        if torch._C._current_graph_task_id() != -1:      # inside backward
            return
        n = self._micro.get(sid, 0) + 1
        self._micro[sid] = n
        if sid == self._leader:
            if self._state is None and n in self._PROBE_AT:
                self._state = self._probe_state()
            self._step = (int(self._state.global_step) if self._state is not None
                          else n)
        # fold the previous micro-batch's measurement in, then P-control
        m = self._pending.pop(sid, None)
        if m is not None:
            prev = self._measured.get(sid)
            self._measured[sid] = m if prev is None else (
                self.ema * prev + (1.0 - self.ema) * m)
        self._scale[sid] = self._closed_loop_scale(sid)
        self._floor[sid] = self._floor_for(self._scale[sid])
        if not self._logged:
            self._logged = True
            print(f"[DASH sched] per-layer closed loop ON: setpoint = current k "
                  f"(anneal {_KS.k_start}->{_KS.k_end}), scale init "
                  f"{self.scale_init:g} in [{self.scale_min:g}, {self.scale_max:g}], "
                  f"floor {self.floor_start:g}->{self.floor_end:g}, gain={self.gain:g} "
                  f"ema={self.ema:g} max_step={self.max_step_ratio:g}", flush=True)
        elif sid == self._leader and n % self.log_every == 0:
            print(f"[DASH sched] step={self._step} k={_KS.k}" + self._table(),
                  flush=True)

    def _table(self):
        """`| L0 scale=.. act=.. | L1 ...` -- every layer's temperature + support."""
        out = []
        for s in sorted(self._scale):
            act = self._measured.get(s)
            out.append(f" | L{s} scale={self._scale[s]:.2f} "
                       f"act={'-' if act is None else format(act, '.1f')}")
        return "".join(out)
    # -- closed loop ---------------------------------------------------------
    def _floor_for(self, scale):
        """Tie the floor to how far the scale has travelled (log domain).

        WHY the floor must follow: the router's gradient channel is
        ``bias = (log(clamp(p, floor)) - mu)/sigma``. If the active p_j fall below
        `floor`, every logp clamps to the same value, the mean-centered bias
        collapses to 0 and the router stops learning. A sharper prior has smaller
        active p, so the floor is annealed DOWN with the scale.
        """
        a, b = self.scale_init, self.scale_ref
        if not a or b <= a:
            p = 0.0
        else:
            p = min(1.0, max(0.0, math.log(max(scale, 1e-8) / a) / math.log(b / a)))
        fa, fb = self.floor_start, self.floor_end
        return fa * (fb / max(fa, 1e-12)) ** p          # geometric, floor goes DOWN

    def _closed_loop_scale(self, sid):
        """Multiplicative P-control in the log domain, per layer: support above
        the k budget -> raise the scale (sharper entmax), below -> back off."""
        tgt = float(_KS.k)
        m = self._measured.get(sid)
        cur = self._scale[sid]
        if tgt <= 0 or m is None:
            return cur
        ratio = math.exp(self.gain * math.log(max(m, 1e-6) / tgt))
        ratio = min(self.max_step_ratio, max(1.0 / self.max_step_ratio, ratio))
        return min(self.scale_max, max(self.scale_min, cur * ratio))

    def observe(self, sid, prior, cand, training):
        """Record this layer's entmax support size for its own loop.

        MEASUREMENT ONLY -- the scale it feeds is applied by ``advance()`` at the
        next micro-batch boundary (mutating it mid-forward breaks the
        gradient-checkpoint recompute).
        """
        if not self.on or not training:
            return
        if torch._C._current_graph_task_id() != -1:
            return
        with torch.no_grad():
            a = ((prior > 0) & cand).sum(dim=-1).float().mean()
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(a, op=dist.ReduceOp.AVG)   # keep ranks in sync
            except Exception:
                pass
            m = float(a.item())
        if math.isfinite(m):
            self._pending[sid] = m

    def needs_calib(self, sid, training):
        return (self.on and self.calibrate_on and sid not in self._calibrated
                and training and torch._C._current_graph_task_id() == -1)

    @torch.no_grad()
    def calibrate(self, sid, block_scores, cand, neg, sqrt_d):
        """Bisect THIS layer's entmax scale on its first batch so its support
        starts at the current k budget.

        The active-block count is monotonically non-increasing in the scale, so a
        geometric bisection converges in a few iterations. Runs on a strided
        subset of the query rows (the full [B,S,nblk] would sort tens of millions
        of entries per probe).
        """
        self._calibrated.add(sid)        # never retry, even if this bails out
        target = float(_KS.k)
        stride = max(1, block_scores.shape[1] // self.calib_rows)
        bs = block_scores[:, ::stride].float()
        cd = cand[:, ::stride]

        def active(s):
            x = torch.where(cd, bs * (s / sqrt_d), bs.new_full((), neg))
            return float(((entmax15(x, dim=-1) > 0) & cd).sum(dim=-1).float().mean())

        lo, hi = self.scale_min, self.scale_max
        if active(lo) <= target:
            s, why = lo, f"already at {active(lo):.1f} blk <= k"
        elif active(hi) > target:
            s, why = hi, f"SCALE_MAX only reaches {active(hi):.1f} blk"
        else:
            for _ in range(20):
                mid = math.sqrt(lo * hi)
                if active(mid) > target:
                    lo = mid
                else:
                    hi = mid
            s, why = hi, f"bisected, {active(hi):.1f} blk"
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                t = torch.tensor([s], device=block_scores.device, dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)   # one scale per layer, all ranks
                s = float(t.item())
        except Exception:
            pass
        self._scale[sid] = s
        self._floor[sid] = self._floor_for(s)
        print(f"[DASH sched] L{sid} calibrated scale={s:.3f} for k={target:g} "
              f"({why})", flush=True)

    # -- readout -------------------------------------------------------------
    def scale(self, sid, fallback):
        return self._scale.get(sid, fallback) if self.on else fallback

    def floor(self, sid, fallback):
        return self._floor.get(sid, fallback) if self.on else fallback

    def state_str(self, sid):
        if not self.on or sid not in self._scale:
            return ""
        return (f" [sched step={self._step} scale={self._scale[sid]:.3f} "
                f"floor={self._floor[sid]:.2e}]")


_SCHED = _PerLayerScheduler()


def _dbg_finite(name, t):
    """Forward-side finiteness probe (only under DASH_DEBUG_ANOMALY=1)."""
    if _DASH_DEBUG and torch.is_tensor(t) and not torch.isfinite(t).all():
        print(f"[DASH-DBG fwd] non-finite in {name}: shape={tuple(t.shape)} "
              f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())}",
              flush=True)


def _dbg_grad(name, t):
    """Register a backward hook that reports the first non-finite gradient."""
    if _DASH_DEBUG and torch.is_tensor(t) and t.requires_grad:
        def _hook(g, _n=name):
            if g is not None and not torch.isfinite(g).all():
                print(f"[DASH-DBG bwd] non-finite GRAD at {_n}: shape={tuple(g.shape)} "
                      f"nan={int(torch.isnan(g).sum())} inf={int(torch.isinf(g).sum())}",
                      flush=True)
            return None
        t.register_hook(_hook)

# log-floor used for zeroed entmax priors; `base > _LOG_TINY + 1` <=> prior > 0.
_LOG_TINY = math.log(1e-20)
# finite "-inf" for SDPA additive masks (real -inf makes SDPA's fused backward
# produce NaN gradients via 0*inf). exp(_MASK_NEG - rowmax) underflows to 0.
_MASK_NEG = -1.0e9
# The router NEVER uses real -inf: masked_fill(-inf) + amax makes fully-masked
# rows -inf, and that -inf reaches the amax/matmul backward, which manufactures
# inf/NaN grads (0*inf) in q_norm/k_norm -> the MulBackward0 NaN anomaly.
# A finite sentinel keeps every backward well defined.
_SCORE_NEG = -1.0e9
# fp32 overflow guard: a *finite but huge* grad (e.g. 1e37) still overflows the
# grad_weight reduction inside RMSNorm's mul backward (sum -> +-inf -> NaN), so
# grads must be BOUNDED, not merely nan_to_num'd.
_GRAD_CLAMP = 1.0e4
_ACT_CLAMP = 1.0e4


def _sanitize_grad(g):
    """nan_to_num + bound: keeps every downstream reduction inside fp32 range."""
    if g is None:
        return None
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        -_GRAD_CLAMP, _GRAD_CLAMP)


# ---------------------------------------------------------------------------
# 1.5-entmax (Peters, Niculae & Martins, 2019). Exact forward + backward.
# The ONLY custom autograd in this file. Operates over `dim` (the block axis).
# Masked-out entries must be passed as a finite, very-negative value (not -inf)
# so the sorting-based threshold stays numerically well-defined; they receive
# probability exactly 0.
# ---------------------------------------------------------------------------
def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _entmax_threshold_and_support(X, dim=-1):
    Xsrt, _ = torch.sort(X, descending=True, dim=dim)
    rho = _make_ix_like(X, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)
    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    # Guard: a degenerate/NaN row can give support_size == 0, so support_size - 1
    # would be -1 and crash `gather` with a device-side index-out-of-bounds
    # assert. Clamp to >=1 (falls back to the top element's threshold).
    support_size = support_size.clamp(min=1)
    tau_star = tau.gather(dim, support_size - 1)
    return tau_star, support_size


class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, dim):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val          # numerical stability, shift-invariant
        X = X / 2                # exponent handling for alpha=1.5
        tau_star, _ = _entmax_threshold_and_support(X, dim)
        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()          # = clamp(X - tau, 0)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim).clamp_min(1e-12)
        q = q.unsqueeze(ctx.dim)
        dX = dX - q * gppr
        return dX, None


def entmax15(X, dim=-1):
    return _Entmax15.apply(X, dim)


# RoPE for the index projections (available at runtime in the assembled model
# dir; absent in a bare unit-test import -> rope is skipped).
try:
    from ._sparse_base_impl import _llama_apply_rotary_pos_emb as _rope_fn
except Exception:
    _rope_fn = None

# The router's norms are NOT in the checkpoint, so they are "missing keys" and get
# whatever `_init_weights` gives them. That hook only matches nn.Linear /
# nn.Embedding / _GemmaRMSNorm -- an unmatched module type is left as
# UNINITIALIZED memory, which is what produced `WEIGHT non-finite: nan=1` in the
# very first forward. Subclassing _GemmaRMSNorm makes the isinstance check hit,
# so the weight is zero-init'd (Gemma scale = 1 + weight => scale 1 at init).
try:
    from ._sparse_base_impl import _GemmaRMSNorm as _RMSNormBase
except Exception:
    class _RMSNormBase(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.zeros(dim))


class _RMSNorm(_RMSNormBase):
    """Gemma-style RMSNorm (scale = 1 + weight, weight init 0) with fp32 +
    finiteness guards for the router's index projections."""

    def forward(self, x):
        x32 = x.float()
        # Sanitize BEFORE the variance reduction: a non-finite input makes
        # mean(x^2)=inf -> rsqrt(inf)=0 -> inf*0=NaN, and that NaN is baked into
        # the tensors saved for backward, so a post-hoc nan_to_num on the output
        # cannot undo it (this is exactly the MulBackward0 NaN anomaly mode caught).
        # Zeroing non-finite positions here keeps BOTH forward and backward finite.
        x32 = torch.nan_to_num(x32, nan=0.0, posinf=0.0, neginf=0.0)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        x32 = x32.clamp(-_ACT_CLAMP, _ACT_CLAMP)
        out = (x32 * (1.0 + self.weight.float())).clamp(
            -_ACT_CLAMP, _ACT_CLAMP).to(x.dtype)
        # The NaN anomaly fires INSIDE this mul's backward (grad wrt `weight`):
        # incoming grads that are finite-but-huge overflow the sum over [B,S] to
        # +-inf and cancel to NaN. Downstream hooks (idx_q/idx_k, post-RoPE)
        # cannot prevent that, so bound the grad AT this output.
        if out.requires_grad:
            out.register_hook(_sanitize_grad)
        if _DASH_DEBUG:
            if not torch.isfinite(self.weight).all():
                print(f"[DASH-DBG rmsnorm] WEIGHT non-finite: "
                      f"nan={int(torch.isnan(self.weight).sum())} "
                      f"inf={int(torch.isinf(self.weight).sum())}", flush=True)
            _dbg_finite("rmsnorm_out", out)
            _dbg_grad("rmsnorm_out(grad_in)", out)
        return out


# ---------------------------------------------------------------------------
# DASH router / attention module
# ---------------------------------------------------------------------------
class DashAttention(nn.Module):
    """Differentiable adaptive sparse hierarchical attention.

    Block scoring uses a SEPARATE low-dim index projection of ``hidden_states``
    (the proven-stable Lightning-Indexer path that ``e2e.sh`` uses), NOT a
    softmax summary over the raw attention keys. The raw-key summary averaged
    over EVERY key position, so any non-finite key from the base model (dead /
    special-token positions the sparse attention never reads -- which is why
    sparse.sh is unaffected) turned a whole block's summary into NaN and poisoned
    the router matmul's backward. The index projection avoids that entirely.

    On top of the (non-local, causal) block-max scores we apply 1.5-entmax to get
    a sparse, differentiable ``prior``; its top-k gives the block selection and
    ``log(prior)`` is fed as the differentiable per-block bias to the fused e2e
    kernel -> router trains end-to-end from the LM loss, no KL distillation.
    """

    def __init__(self, config):
        super().__init__()
        sp = getattr(config, "sparse_attention_config", {}) or {}
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # block granularity reuses the base sparse knobs so configs stay aligned
        self.block_size = int(sp.get("sparse_block_size", 128))
        self.local_blocks = max(1, int(sp.get("sparse_local_block", 1)))
        self.local_window = self.local_blocks * self.block_size

        # DASH-specific knobs
        self.scaling_factor = float(sp.get("dash_scaling_factor", 2.0))  # entmax sharpness
        self.q_chunk = int(sp.get("dash_query_chunk", 2048))             # memory knob
        self.image_only = bool(sp.get("dash_image_only", True))
        self._image_token_id = int(sp.get("image_token_id", 128815))
        # log-prior bias = log(prior)/sigma; prior_floor bounds the 1/p gradient.
        self.sigma = float(sp.get("dash_sigma", 1.0))
        self._prior_floor = float(sp.get("dash_prior_floor", 1e-2))

        self.topk_blocks = int(sp.get("sparse_topk_blocks", 16))
        self.remote_blocks = max(1, self.topk_blocks - self.local_blocks)
        self.use_triton = bool(sp.get("dash_use_triton", True))

        # single-head index projections (like DeepseekV2Indexer, minus KL).
        self.index_dim = int(sp.get("sparse_index_dim", self.head_dim))
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_proj = nn.Linear(config.hidden_size, self.index_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.index_dim, bias=False)
        self.q_norm = _RMSNorm(self.index_dim, eps=eps)
        self.k_norm = _RMSNorm(self.index_dim, eps=eps)

        # Stage-0 summarization mode (env DASH_STAGE0):
        #   "indexer" (default) -> block-max of token-level idx_q.idx_k (current).
        #   "head_cls"          -> paper Stage-0: a learned per-router summary
        #                          query `head_cls` (zero-init => starts as mean
        #                          pooling) attends the index-keys within each
        #                          chunk to build one summary key k_bar_c, then
        #                          the block score is idx_q . k_bar_c.
        self.stage0 = os.environ.get("DASH_STAGE0", "indexer").lower()
        # Learned summary query (zero-init: uniform local attention == mean pool
        # at the start, then gradually learns an expressive chunk summary). Kept
        # unconditionally so a checkpoint can switch modes without a shape change.
        self.head_cls = nn.Parameter(torch.zeros(self.index_dim))

        self._scale = 1.0 / math.sqrt(self.head_dim)          # main attention
        self._idx_scale = self.scaling_factor / math.sqrt(self.index_dim)  # entmax input
        self._neg = -1.0e4   # finite mask fill for entmax candidate masking

        # Per-layer entmax temperature loop + the shared k schedule. Both are
        # registered here so this layer gets its own scale/floor slot and its own
        # ordinal (the L<sid> tag in every log line).
        self._sched_id = _SCHED.register(self.scaling_factor, self._prior_floor)
        _KS.register()

        # Live handle on config.sparse_attention_config: the ROUTER STATE that
        # inference must reproduce (entmax scale, prior floor, stage-0 mode) is
        # partly runtime (the scheduler anneals scale/floor) and partly env
        # (DASH_STAGE0). save_pretrained serializes this dict, so mirroring the
        # current values into it is what makes the exported checkpoint
        # self-describing -- otherwise inference re-reads the ORIGINAL
        # dash_scaling_factor / dash_prior_floor and routes at a completely
        # different temperature than the trained one.
        # NOTE: the scale/floor are PER LAYER now, so the scalar keys carry the
        # leader layer's values (backwards compatible) and the full profile goes
        # to dash_scale_per_layer / dash_floor_per_layer. dash_topk_blocks is the
        # k this checkpoint was trained at -- mid-anneal exports are NOT at K_END.
        self._sp = sp if isinstance(sp, dict) else None
        self._persist_router_cfg()

    # -- export the effective router state into the saved config -------------
    def _persist_router_cfg(self):
        if self._sp is None or self._sched_id != _SCHED._leader:
            return
        self._sp["dash_stage0"] = self.stage0
        self._sp["dash_router"] = "entmax_topk_perlayer"
        self._sp["dash_sigma"] = float(self.sigma)
        self._sp["dash_topk_blocks"] = int(_KS.k)
        self._sp["dash_scaling_factor"] = float(
            self._cur_idx_scale() * math.sqrt(self.index_dim))
        self._sp["dash_prior_floor"] = float(self._cur_prior_floor())
        self._sp["dash_scale_per_layer"] = [
            float(_SCHED.scale(s, self.scaling_factor)) for s in sorted(_SCHED._scale)]
        self._sp["dash_floor_per_layer"] = [
            float(_SCHED.floor(s, self._prior_floor)) for s in sorted(_SCHED._floor)]

    # -- scheduled knobs (identity when the scheduler is off) ---------------
    def _cur_idx_scale(self):
        return (_SCHED.scale(self._sched_id, self.scaling_factor)
                / math.sqrt(self.index_dim))

    def _cur_prior_floor(self):
        return _SCHED.floor(self._sched_id, self._prior_floor)

    @torch.no_grad()
    def build_image_block_mask(self, source_input_ids, key_length, nblk, device):
        """Return [B, nblk] bool: block contains >=1 image token. None if n/a."""
        if source_input_ids is None or source_input_ids.ndim != 2:
            return None
        ids = source_input_ids[..., :key_length].to(device)
        if ids.shape[-1] < key_length:
            ids = F.pad(ids, (0, key_length - ids.shape[-1]), value=-1)
        img = ids == self._image_token_id                            # [B, key_length]
        if img.sum() == 0:
            return None
        Sp = nblk * self.block_size
        if Sp > img.shape[-1]:
            img = F.pad(img, (0, Sp - img.shape[-1]), value=False)
        else:
            img = img[..., :Sp]
        return img.view(img.shape[0], nblk, self.block_size).any(dim=-1)  # [B,nblk]

    # -- block scoring (indexer-style, non-local causal block-max) --------
    def _block_scores(self, hidden_states, position_embeddings, position_ids, nblk):
        """[B,S,nblk] fp32 remote block scores; ``_SCORE_NEG`` where a block has no
        non-local causal key. Computed from a low-dim index projection (finite by
        construction), chunked over queries to bound memory."""
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        # Router reads hidden_states directly; if an upstream layer emitted a
        # non-finite activation, the projections + matmul below manufacture NaN
        # block_scores (and NaN grads in q_norm/k_norm). Sanitize the router's own
        # view so it never propagates upstream NaN into the routing path.
        hs = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        idx_q = self.q_norm(self.q_proj(hs).view(B, S, 1, self.index_dim)).transpose(1, 2)
        idx_k = self.k_norm(self.k_proj(hs).view(B, S, 1, self.index_dim)).transpose(1, 2)
        if _rope_fn is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            idx_q, idx_k = _rope_fn(idx_q, idx_k, cos[..., :self.index_dim], sin[..., :self.index_dim])
        # nan_to_num defaults map +/-inf to +/-3.4e38, which then OVERFLOWS the
        # score matmul back into inf -> NaN. Map non-finite to 0 instead.
        idx_q = torch.nan_to_num(idx_q[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)  # [B,S,d]
        idx_k = torch.nan_to_num(idx_k[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)  # [B,S,d]
        # The amax/matmul backward below sits BETWEEN block_scores and the q/k_norm
        # RMSNorms and can manufacture NaN from the -inf entries masked_fill'd into
        # `sc` even when the incoming block_scores grad is finite -- which is why a
        # block_scores-level hook does not stop the k_norm MulBackward0 NaN. Report
        # (debug) then sanitize the grad right at the norm outputs (tight guard).
        _dbg_grad("idx_q(grad_in)", idx_q); _dbg_grad("idx_k(grad_in)", idx_k)
        if idx_q.requires_grad:
            idx_q.register_hook(_sanitize_grad)
        if idx_k.requires_grad:
            idx_k.register_hook(_sanitize_grad)
        kpos = torch.arange(S, device=device)
        Sp = nblk * self.block_size
        out = []
        for c0 in range(0, S, self.q_chunk):
            c1 = min(c0 + self.q_chunk, S)
            sc = torch.matmul(idx_q[:, c0:c1], idx_k.transpose(-1, -2))   # [B,Lq,S]
            sc = sc.clamp(-_ACT_CLAMP, _ACT_CLAMP)
            pos = position_ids[:, c0:c1]                                  # [B,Lq]
            future = kpos[None, None, :] > pos[:, :, None]
            local = (kpos[None, None, :] > (pos[:, :, None] - self.local_window)) & ~future
            sc = sc.masked_fill(future | local, _SCORE_NEG)               # remote-only
            if Sp > S:
                sc = F.pad(sc, (0, Sp - S), value=_SCORE_NEG)
            sc = sc.view(B, c1 - c0, nblk, self.block_size)
            out.append(sc.amax(dim=-1))                                  # [B,Lq,nblk]
        return torch.cat(out, dim=1)                                     # [B,S,nblk]

    # -- Stage-0 (head_cls): learned-summary block scoring (paper) --------
    def _block_scores_headcls(self, hidden_states, position_embeddings, position_ids, nblk):
        """[B,S,nblk] fp32 remote block scores via a learned chunk summary.

        Mirrors the paper's Stage-0/Stage-1: a per-router summary query
        ``head_cls`` (zero-init -> starts as mean pooling) attends the index-keys
        within each chunk to form one summary key ``k_bar_c``; the block score is
        ``idx_q_i . k_bar_c``. Same single-head index space (q_proj/k_proj/norms)
        and the same remote-only (causal minus local-window) masking as the
        ``indexer`` path, so it drops straight into ``_route``. NOTE: the summary
        is over the INDEX-key space (not the main keys) to stay single-head and
        self-contained -- a deliberate adaptation of the paper to this codebase.
        """
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        hs = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        idx_q = self.q_norm(self.q_proj(hs).view(B, S, 1, self.index_dim)).transpose(1, 2)
        idx_k = self.k_norm(self.k_proj(hs).view(B, S, 1, self.index_dim)).transpose(1, 2)
        if _rope_fn is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            idx_q, idx_k = _rope_fn(idx_q, idx_k, cos[..., :self.index_dim], sin[..., :self.index_dim])
        idx_q = torch.nan_to_num(idx_q[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)  # [B,S,d]
        idx_k = torch.nan_to_num(idx_k[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)  # [B,S,d]
        if idx_q.requires_grad:
            idx_q.register_hook(_sanitize_grad)
        if idx_k.requires_grad:
            idx_k.register_hook(_sanitize_grad)
        d = self.index_dim
        inv_sqrt_d = 1.0 / math.sqrt(d)
        Sp = nblk * self.block_size

        # Pad the index-keys up to whole chunks and track which slots are real.
        if Sp > S:
            idx_k = F.pad(idx_k, (0, 0, 0, Sp - S))                       # [B,Sp,d]
            real = F.pad(idx_k.new_ones(B, S, dtype=torch.bool), (0, Sp - S), value=False)
        else:
            idx_k = idx_k[:, :Sp]
            real = idx_k.new_ones(B, Sp, dtype=torch.bool)
        kk = idx_k.view(B, nblk, self.block_size, d)                      # [B,nblk,bs,d]
        vmask = real.view(B, nblk, self.block_size)                       # [B,nblk,bs]

        # Stage 0: head_cls local attention within each chunk -> summary key.
        # NaN-safe MASKED softmax: never feed -inf to softmax (a fully-padding
        # chunk would give all -inf -> NaN in BOTH forward AND SoftmaxBackward0,
        # which nan_to_num on the output cannot repair). Instead subtract the row
        # max (finite), exp, zero out padding multiplicatively, and normalize by a
        # clamped denominator -> forward and backward stay finite. hc=0 (zero-init)
        # -> uniform weights over valid tokens == mean pooling.
        hc = self.head_cls.to(kk.dtype).view(1, 1, 1, d)
        aw = (kk * hc).sum(-1) * inv_sqrt_d                               # [B,nblk,bs]
        aw = aw.clamp(-_ACT_CLAMP, _ACT_CLAMP)
        # NaN-safe masked softmax. Take the row-max over VALID tokens only: if we
        # max over padding too (padding aw==0), a chunk whose valid scores are all
        # negative would be dominated by padding, its valid exp() underflow, denom
        # -> 0, and the 1/denom in the division backward explodes (-> BmmBackward0
        # NaN downstream). Valid-only max keeps >=1 entry at e==1, so denom>=1.
        _neg = torch.finfo(aw.dtype).min
        row_max = torch.where(vmask, aw, aw.new_full((), _neg)).amax(dim=-1, keepdim=True)
        row_max = torch.where(vmask.any(dim=-1, keepdim=True), row_max,
                              torch.zeros_like(row_max))                  # all-pad chunk -> 0
        aw = aw - row_max
        e = torch.exp(aw) * vmask.to(aw.dtype)                            # padding -> 0
        w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)               # [B,nblk,bs]
        k_bar = (w.unsqueeze(-1) * kk).sum(dim=2)                         # [B,nblk,d]
        # k_bar is a saved input of the Stage-1 matmul's backward -- force it
        # finite so BmmBackward0 can never emit NaN from a bad saved activation.
        k_bar = torch.nan_to_num(k_bar, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_valid = vmask.any(dim=-1)                                   # [B,nblk]

        # Stage 1: z[i,c] = idx_q_i . k_bar_c, remote-only (causal minus local).
        blk_ids = torch.arange(nblk, device=device)
        out = []
        for c0 in range(0, S, self.q_chunk):
            c1 = min(c0 + self.q_chunk, S)
            z = torch.matmul(idx_q[:, c0:c1], k_bar.transpose(-1, -2))    # [B,Lq,nblk]
            # Sanitize the grad AT the matmul output: this is exactly the
            # grad_output BmmBackward0 consumes, so a NaN/Inf arriving from the
            # entmax/route backward (which the block_scores hook can miss under
            # gradient checkpointing) is cleaned before the matmul backward runs.
            if z.requires_grad:
                z.register_hook(_sanitize_grad)
            z = z * inv_sqrt_d
            z = z.clamp(-_ACT_CLAMP, _ACT_CLAMP)
            qblk = (position_ids[:, c0:c1] // self.block_size)            # [B,Lq]
            # chunk c is a remote candidate iff it lies fully before the local
            # window: c <= qblk - local_blocks (matches the indexer path's
            # token-level `pos + local_blocks*block_size <= query_pos`).
            remote_ok = blk_ids[None, None, :] <= (qblk[:, :, None] - self.local_blocks)
            keep = remote_ok & chunk_valid[:, None, :]
            z = z.masked_fill(~keep, _SCORE_NEG)
            out.append(z)
        return torch.cat(out, dim=1)                                     # [B,S,nblk]

    def _route(self, block_scores, image_block_mask, block_cand=None):
        """block_scores [B,S,nblk] -> (prior [B,S,nblk], cand [B,S,nblk] bool).

        ``block_cand`` [B,S,nblk] is the PER-QUERY candidate domain the caller builds
        from the indexer's page-aware ids (``_build_kl_candidate_mask`` -> same-page image
        bucket, with its own per-query fallback). It takes precedence over
        ``image_block_mask`` [B,nblk], which is page-agnostic (image blocks of any page,
        identical for every query) and would drop page isolation entirely.
        """
        # remote candidacy is now flagged by the finite sentinel, not by -inf
        remote = block_scores > (_SCORE_NEG * 0.5)
        cand = remote
        if block_cand is not None:
            cand = remote & block_cand
        elif self.image_only and image_block_mask is not None:
            cand = remote & image_block_mask[:, None, :]
        # per-query fallback: if image left no candidate but remote blocks exist,
        # drop the image constraint for that query.
        no_cand = ~cand.any(dim=-1, keepdim=True)
        cand = torch.where(no_cand, remote, cand)
        # one-shot per-layer calibration: land THIS layer's entmax support on the
        # current k budget, measured on real block scores (first training batch).
        if _SCHED.needs_calib(self._sched_id, self.training):
            _SCHED.calibrate(self._sched_id, block_scores.detach(), cand, self._neg,
                             math.sqrt(self.index_dim))
        x = torch.where(cand, block_scores * self._cur_idx_scale(),
                        block_scores.new_full((), self._neg))
        x = torch.nan_to_num(x, nan=self._neg, posinf=self._neg, neginf=self._neg)
        prior = entmax15(x, dim=-1)                                      # [B,S,nblk]
        return prior, cand

    def _select_and_bias(self, prior, cand, nblk):
        """Hard top-k selection + the entmax log-prior bias on the kept blocks.

        Returns (topi [B,S,K] int64, bias [B,S,K], valid [B,S,K] bool).

        Selection is top-k over the MASKED PRIOR; entmax is monotone in its input
        within a row, so that is the same ranking as top-k over the block scores
        and the k highest-scoring candidates are always the ones read -- including
        blocks entmax pushed to exactly 0 when its support is narrower than k.
        Those carry bias log(prior_floor) - mu: attended, but no gradient, which is
        precisely what the per-layer temperature loop exists to prevent.

        bias = (log(clamp(p, floor)) - mu)/sigma, mu = per-query MEAN over the kept
        log-priors (paper's log_g). Mean-centering is what lets an above-average
        remote block outweigh the zero-bias local window, instead of every remote
        block being uniformly down-weighted by log(p) <= 0.
        """
        K = max(1, min(int(_KS.k), nblk))
        sel_prior = prior.masked_fill(~cand, -1.0)   # non-candidates rank last
        topv, topi = sel_prior.topk(K, dim=-1)                           # [B,S,K]
        valid = topv >= 0.0                          # a real candidate slot
        logp = torch.log(topv.clamp_min(self._cur_prior_floor()))
        cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1)
        mu = torch.where(valid, logp, torch.zeros_like(logp)).sum(
            dim=-1, keepdim=True) / cnt
        bias = (logp - mu) * (1.0 / self.sigma)
        bias = torch.where(valid, bias, torch.zeros_like(bias)).clamp(
            -_ACT_CLAMP, _ACT_CLAMP)
        return topi, bias, valid

    # -- Triton block-sparse forward (32k) --------------------------------
    def _forward_triton(self, q, k, v, topi, blk_bias, valid, S, nblk, position_ids):
        try:
            from .sparse_gather_e2e_triton import triton_gather_attention_e2e
        except Exception:
            return None
        # Fixed budget: the gathered width is exactly k for every query, every
        # layer, every step (padding slots -> index -1, skipped by the kernel).
        # A constant width also means the gradient-checkpoint recompute can never
        # disagree with the saved forward on metadata.
        block_indices = topi.masked_fill(~valid, -1).to(torch.int32).unsqueeze(1)
        out = triton_gather_attention_e2e(
            q, k, v, block_indices, blk_bias, position_ids, None,
            self.block_size, self._scale, self.local_window,
        )
        _dbg_grad("triton_out", out)
        if out is not None and not getattr(self, "_triton_logged", False):
            self._triton_logged = True
            print(f"[DASH] Triton block-sparse path active (S={S}, "
                  f"block_size={self.block_size}, remote_topk={topi.shape[-1]}, "
                  f"local_window={self.local_window})", flush=True)
        return out

    # -- dense fallback (CPU / unsupported shape / validation reference) --
    def _forward_dense(self, q, k, v, topi, blk_bias, valid, S, nblk, attention_mask):
        B, H, _, D = q.shape
        device = q.device
        kidx = torch.arange(S, device=device)
        key_block = kidx // self.block_size
        # Scatter the per-selection bias back to a full [B,S,nblk] map so the dense
        # reference attends EXACTLY the same k blocks as the kernel; unselected
        # entries carry _MASK_NEG and are dropped below. `scatter` (functional)
        # keeps the autograd path into blk_bias.
        src = torch.where(valid, blk_bias, blk_bias.new_full((), _MASK_NEG))
        logp = torch.full((B, blk_bias.shape[1], nblk), _MASK_NEG,
                          device=device, dtype=blk_bias.dtype).scatter(-1, topi, src)
        out_chunks = []
        for qs in range(0, S, self.q_chunk):
            qe = min(qs + self.q_chunk, S)
            qidx = torch.arange(qs, qe, device=device)
            q_c = q[:, :, qs:qe, :]
            base = logp[:, qs:qe].index_select(-1, key_block)[:, None]   # [B,1,Lq,S]
            causal = (kidx[None, :] <= qidx[:, None])[None, None]
            local = ((kidx[None, :] <= qidx[:, None]) &
                     (kidx[None, :] > (qidx[:, None] - self.local_window)))[None, None]
            sel = base > (_MASK_NEG * 0.5)
            keep = causal & (local | sel)
            mask = torch.where(local, torch.zeros_like(base), base)
            mask = mask.masked_fill(~keep, _MASK_NEG)
            if attention_mask is not None:
                mask = mask + attention_mask[:, :, qs:qe, :S].to(mask.dtype)
            mask = mask.clamp_min(_MASK_NEG).to(q.dtype)
            out_chunks.append(torch.nn.functional.scaled_dot_product_attention(
                q_c, k, v, attn_mask=mask, is_causal=False, scale=self._scale))
        return torch.cat(out_chunks, dim=2)

    # -- forward -----------------------------------------------------------
    def forward(self, hidden_states, q, k, v, position_embeddings=None,
                position_ids=None, attention_mask=None, image_block_mask=None,
                block_cand=None, page_ids=None):
        """hidden_states [B,S,hidden]; q,k,v [B,H,S,D] (RoPE'd, GQA-expanded).

        Returns [B,H,S,D] (before o_proj). image_block_mask: [B,nblk] bool or None.
        block_cand: [B,S,nblk] bool per-query candidate domain (page isolation) or None.
        page_ids: (image_token_page_ids, query_page_ids) -- used only by the
            ``DASH_LOG_PAGE_EVERY`` diagnostics, never in the computation.
        """
        B, H, S, D = q.shape
        nblk = (S + self.block_size - 1) // self.block_size
        device = q.device
        if position_ids is None:
            position_ids = torch.arange(S, device=device)[None].expand(B, -1)

        # Refresh THIS layer's entmax temperature and the shared k (once per
        # micro-batch each; both stay frozen for the rest of the forward).
        _KS.advance(self._sched_id, self.training)
        _SCHED.advance(self._sched_id, self.training)
        # Mirror the current scale/floor/k into the config so any checkpoint saved
        # after this step carries the router state inference has to reproduce.
        self._persist_router_cfg()

        if self.stage0 == "head_cls":
            block_scores = self._block_scores_headcls(hidden_states, position_embeddings, position_ids, nblk)
        else:
            block_scores = self._block_scores(hidden_states, position_embeddings, position_ids, nblk)
        if not getattr(self, "_stage0_logged", False):
            self._stage0_logged = True
            print(f"[DASH] Stage-0 summarization = {self.stage0!r}", flush=True)
        # Router gradient choke point. `prior`'s own hook (below) only cleans the
        # grad ENTERING prior; the entmax / where / log backward BETWEEN prior and
        # block_scores can still manufacture NaN/Inf on degenerate rows, which then
        # flows into q_norm/k_norm (the MulBackward0 anomaly that persists even
        # after the forward was made finite). Sanitizing here -- upstream of the
        # RMSNorm backward -- guarantees only finite grads reach q_proj/k_proj.
        if block_scores.requires_grad:
            block_scores.register_hook(_sanitize_grad)
        prior, cand = self._route(block_scores, image_block_mask, block_cand)  # [B,S,nblk]
        topi, blk_bias, valid = self._select_and_bias(prior, cand, nblk)
        _dbg_finite("block_scores", block_scores); _dbg_finite("prior", prior)
        _dbg_grad("prior", prior)
        # `prior` is the SINGLE gradient channel from the LM loss back to the router
        # (loss -> kernel d_bias / dense mask -> blk_bias/log(prior) -> prior ->
        # block_scores -> q_proj/k_proj). The kernel's d_bias backward on our
        # log(prior) bias can emit NaN/Inf for degenerate selections (finite loss,
        # NaN grad_norm). Sanitizing the gradient AT this choke point keeps the
        # router params finite while still training them from finite gradients.
        if prior.requires_grad:
            prior.register_hook(_sanitize_grad)

        if _LOG_SPARSITY:
            _log_train_sparsity(prior, cand, valid, self.block_size, self._sched_id)
        if (_LOG_PAGE_EVERY and page_ids is not None and block_cand is not None
                and self.training and torch._C._current_graph_task_id() == -1):
            self._page_log_n = getattr(self, "_page_log_n", 0) + 1
            if self._page_log_n % _LOG_PAGE_EVERY == 0 or self._page_log_n == 1:
                _log_page_purity(topi, valid, block_cand,
                                 block_scores.detach() > (_SCORE_NEG * 0.5),
                                 page_ids[0], page_ids[1], self.block_size, nblk,
                                 self._sched_id, self._page_log_n)
        # How much of the paid-for k blocks the attention really uses. Skipped in
        # the gradient-checkpoint recompute (same guard as the scheduler), else a
        # layer's counter advances twice per micro-batch.
        if _LOG_NEFF_EVERY and self.training and \
                torch._C._current_graph_task_id() == -1:
            _log_neff(q, k, topi, blk_bias, valid, position_ids, self.block_size,
                      self.local_window, self._scale, self._sched_id)
        # Oracle recall of the routed set (block/score recall vs. the dense
        # attention mass). Same recompute guard: the [H,rows,S] qk is expensive
        # enough that running it twice per micro-batch is not free.
        if _LOG_RECALL_EVERY and self.training and \
                torch._C._current_graph_task_id() == -1:
            _log_oracle_recall(q, k, topi, valid, cand, position_ids,
                               self.block_size, self._scale, self._sched_id)
        # this layer's own closed loop reads back its entmax support size
        _SCHED.observe(self._sched_id, prior, cand, self.training)

        if self.use_triton and q.is_cuda and attention_mask is None:
            out = self._forward_triton(q, k, v, topi, blk_bias, valid, S, nblk,
                                       position_ids)
            if out is not None:
                return out

        return self._forward_dense(q, k, v, topi, blk_bias, valid, S, nblk,
                                   attention_mask)
