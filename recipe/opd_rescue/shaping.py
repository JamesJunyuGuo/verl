"""Verified per-method math for the PivotOPD-appendix baseline family.

Every function here was checked against the method's OWN code or paper on
2026-10-10 (file:line / equation pointers in docs/baselines_impl.md). The
family splits into four structural groups, which pilot.baselines.REGISTRY
and scripts/stage4/distill.py organize around:

  adv_shaping  GRPO loss untouched; the ADVANTAGE is modified per token or
               per turn from delta_t = lpT_t - lpS_t (teacher = usually the
               policy itself under a privileged context, scoring the SAME
               sampled tokens, no gradient):
               RLSD, SOD, StepOPSD, AgentOPSD, OPID, Skill-SD.
  add_loss     GRPO untouched; a small additive likelihood term: SDAR.
  distill      a true full-vocab KL distillation loss: OPSD, TurnOPD, TCOD.
  selection    no new loss at all; a state filter feeding stock reset-GRPO:
               PivotRL.

delta below always means lpT - lpS on the sampled token (teacher minus
student log-prob). Pure torch + python -- never imports vllm.
"""

from __future__ import annotations

import math

import torch


# =============================================================================
# Full-vocab KL (the `distill` group: OPSD / TurnOPD / TCOD)
# =============================================================================


def selected_pred_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rows of the distribution that PREDICTS each masked token: logits at
    position p-1 predict token p (same shift as grpo.selected_token_logprobs).
    logits [B, T, V], mask [B, T] bool -> [N, V] in row-major mask order.
    mask must never mark position 0 of a row (true for build_*_example)."""
    b, t, v = logits.shape
    pos = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if pos.shape[0] == 0:
        return logits.new_zeros((0, v))
    return logits.reshape(b * t, v).index_select(0, pos - 1)


def kl_token_terms(student_rows: torch.Tensor, teacher_rows: torch.Tensor,
                   direction: str = "reverse", chunk: int = 256) -> torch.Tensor:
    """Per-token full-vocab KL between aligned prediction rows [N, V].
    direction="reverse": KL(pi || T) (mode-seeking, the OPD default);
    "forward": KL(T || pi). teacher_rows must already be detached. Chunked
    fp32 softmax so [N, V] float32 never materializes twice."""
    assert student_rows.shape == teacher_rows.shape, \
        f"row mismatch: {tuple(student_rows.shape)} vs {tuple(teacher_rows.shape)}"
    outs = []
    for s in range(0, student_rows.shape[0], chunk):
        e = min(s + chunk, student_rows.shape[0])
        s_lp = torch.log_softmax(student_rows[s:e].float(), dim=-1)
        t_lp = torch.log_softmax(teacher_rows[s:e].float(), dim=-1)
        if direction == "reverse":
            outs.append((s_lp.exp() * (s_lp - t_lp)).sum(-1))
        elif direction == "forward":
            outs.append((t_lp.exp() * (t_lp - s_lp)).sum(-1))
        else:
            raise ValueError(f"unknown KL direction: {direction}")
    if not outs:
        return student_rows.new_zeros(0)
    return torch.cat(outs)


def turnopd_blend(step_frac: float) -> float:
    """TurnOPD (arXiv 2607.05804) progressive loss budgeting: blend c in
    [0, 1] between trajectory-level aggregation (c=0: plain token-mean over
    the trajectory) and turn-balanced aggregation (c=1: mean of per-turn
    means). Linear ramp over the budget fraction. UNVERIFIED DETAIL: the
    ramp shape (secondary sources say linear; the paper's schedule parameter
    was not retrievable); the rollout-depth probe controller is NOT
    implemented. Clamped to [0, 1]."""
    return min(1.0, max(0.0, float(step_frac)))


def turnopd_token_weights(turn_lens: list[int], blend: float) -> list[float]:
    """Per-token aggregation weights realizing the trajectory<->turn blend
    for ONE trajectory whose suffix turns have `turn_lens` output tokens:
    trajectory-level gives every token 1/total; turn-balanced gives each
    TURN 1/n_turns split over its tokens. Returned flat, turn by turn;
    weights sum to 1 (one trajectory = one unit of loss either way)."""
    total = sum(turn_lens)
    n = len(turn_lens)
    if total == 0 or n == 0:
        return []
    out = []
    for L in turn_lens:
        w_traj = 1.0 / total
        w_turn = (1.0 / n) / L if L else 0.0
        out.extend([(1.0 - blend) * w_traj + blend * w_turn] * L)
    return out


def tcod_depth_cap(step_frac: float, max_turns: int, k: float = 1.0,
                   min_turns: int = 1) -> int:
    """TCOD-F2B (arXiv 2604.24005): temporal curriculum over the trajectory
    depth the student is distilled on -- start with the earliest turns and
    linearly extend to the full horizon; k is the paper's linear pacing rate
    (depth grows ~ k * budget fraction). UNVERIFIED DETAIL: the exact pacing
    formula (the paper's figure says "linear pacing" with rate k); verify
    against github.com/kokolerk/TCOD before reporting. The B2F variant is
    not implemented."""
    frac = min(1.0, max(0.0, k * float(step_frac)))
    return max(min_turns, min(max_turns, math.ceil(frac * max_turns)))


# =============================================================================
# adv_shaping group
# =============================================================================


def lambda_schedule(step: int, target: float, warmup_steps: int = 0,
                    decay_steps: int = 0) -> float:
    """RLSD's lambda schedule (iie-ycx/RLSD opsd_trainer.py:1952-1961),
    shared by the StepOPSD mixing strength (their lam_mix also decays to ~0
    by ~step 50): linear warmup to `target` over warmup_steps, then linear
    decay to 0 over decay_steps. decay_steps=0 means constant after warmup.
    Note both methods that anneal privileged pressure to zero echo this
    repo's own L2 lesson."""
    if warmup_steps > 0 and step < warmup_steps:
        return target * (step / warmup_steps)
    if decay_steps > 0 and step >= warmup_steps:
        return target * max(1.0 - (step - warmup_steps) / decay_steps, 0.0)
    return target


def rlsd_token_multipliers(adv: float, delta: list[float], lam: float,
                           clip_range: float = 0.2) -> list[float]:
    """RLSD (iie-ycx/RLSD dp_opsd_actor.py:221-239): per-token advantage
    multiplier (1-lam) + lam * clip(exp(sign(A) * delta_t), 1-c, 1+c).
    sign(A) is constant over the sequence (GRPO adv is per-sequence); A = 0
    keeps the multiplier at 1. Multiplicative and per token -- NOT a softmax
    over the group. Their teacher = a frozen copy of the policy re-synced
    every ~20 steps, hint = the reference solution (use_gt_as_hint);
    defaults lam 0.5 decayed to 0 by step 60, clip_range 0.2."""
    s = (adv > 0) - (adv < 0)
    lo, hi = 1.0 - clip_range, 1.0 + clip_range
    return [(1.0 - lam) + lam * min(hi, max(lo, math.exp(s * d))) for d in delta]


def sod_turn_weights(d_per_turn: list[float], eps: float = 1e-6,
                     delta_cap: float = 0.2) -> list[float]:
    """SOD (YoungZ365/SOD ray_trainer.py:363-440): d_k = the turn's mean
    |lpS - lpT| over its own tokens; w_1 = 1 and, because the ratio
    telescopes, w_k = min((d_1 + eps) / (d_k + eps), 1 + delta_cap) for
    k >= 2 -- every turn is compared to TURN 1, capped above at
    1 + delta_cap, unbounded below (a turn that drifted far from the
    teacher gets a weight near 0). Detached floats; no normalization
    across turns. Their teacher is an EXTERNAL trained model."""
    if not d_per_turn:
        return []
    w = [1.0]
    cum = 1.0
    for u in range(len(d_per_turn) - 1):
        cum *= (d_per_turn[u] + eps) / (d_per_turn[u + 1] + eps)
        w.append(min(cum, 1.0 + delta_cap))
    return w


def sod_token_advantages(adv: float, delta: list[float], w_turn: float,
                         opd_coef: float = 1.0) -> list[float]:
    """SOD (ray_trainer.py:878-929): per-token total advantage
    A_t = A_GRPO + opd_coef * w_turn * delta_t -- the OPD signal enters
    ADDITIVELY per token; the PPO-clip loss then consumes A_t unchanged."""
    return [adv + opd_coef * w_turn * d for d in delta]


def opid_token_advantages(adv: float, delta: list[float],
                          weight: float = 1e-3) -> list[float]:
    """OPID (jinyangwu/OPID core_gigpo.py:290-292, 471-480): per-token
    A_t = A_GRPO + weight * delta_t, delta_t = lp(y_t | x + skill) -
    lp(y_t | x) under the SAME pre-update policy, detached. The skill text
    is written fresh each batch by an external analyzer LLM from the
    batch's own rollouts (episode-level workflow/avoidance skill; up to a
    few critical-step skills that take priority on their steps). Shipped
    weight: 0.001 for both skill kinds."""
    return [adv + weight * d for d in delta]


def skillsd_token_advantages(adv: float, delta: list[float],
                             coef: float = 1.0) -> list[float]:
    """Skill-SD (arXiv 2604.10674): the per-token reverse-KL signal to the
    frozen skill-conditioned self-teacher acts as a per-token advantage in
    the policy-gradient objective, paired with GRPO. Implemented as
    A_t = A_GRPO + coef * delta_t (sampled-token estimate; same shape as
    OPID with skills summarized from finished trajectories instead of a
    per-batch analyzer). PAPER-LEVEL: verify coef and the pairing against
    the paper's equations before reporting."""
    return [adv + coef * d for d in delta]


def stepopsd_multiplier(adv: float, delta_step: float, lam: float,
                        alpha_clip: float = 0.2) -> float:
    """StepOPSD (arXiv 2605.27140 eq. 3-4): sign-preserving per-step
    advantage multiplier (1-lam) + lam * clip(2*sigmoid(sign(A)*Delta),
    1-alpha, 1+alpha); Delta = the step span's aggregated delta, teacher
    conditioned on the HINDSIGHT context (a successful peer of the same
    rollout group; a binary outcome note when the group has no winner).
    lam decays over training (lambda_schedule). w = 1 exactly at Delta = 0.
    A step segment = one assistant turn here (their action-centered span)."""
    s = (adv > 0) - (adv < 0)
    w = 2.0 / (1.0 + math.exp(-s * delta_step))
    w = min(1.0 + alpha_clip, max(1.0 - alpha_clip, w))
    return (1.0 - lam) + lam * w


def agentopsd_turn_multipliers(adv_sign: float, delta_turn_sums: list[float],
                               v0: float, lam: float = 0.5, band: float = 0.2,
                               gamma: float = 1.0, eps: float = 1e-6) -> list[float]:
    """AgentOPSD (ZethWang/AgentOPSD opsd_utils.py:305-529, the shipped
    belief_mult/belief_gate/mult_signed path, turn granularity = their
    ALFWorld config):

      logit V_k = logit(clamp(v0)) + discounted-cumsum_gamma(sum_t delta_t of turn k)
      dV_k      = V_k - V_{k-1}            (V_0 = v0; V sigmoid-clamped)
      z_k       = (dV_k - mean) / (std + eps)   within the episode
      m_k       = clip(1 + band * sign(A_seq) * z_k, 1-band, 1+band)
      mult_k    = (1 - lam) + lam * m_k

    v0 = the GRPO group success rate clamped to [1e-4, 1-1e-4]; gamma = 1
    (default) is a plain cumsum. Shipped lam 0.5, band 0.2 (multiplier in
    [0.9, 1.1])."""
    v0 = min(1.0 - 1e-4, max(1e-4, v0))
    logit = math.log(v0 / (1.0 - v0))
    vs = []
    acc = 0.0
    for d in delta_turn_sums:
        acc = gamma * acc + d
        x = logit + acc
        vs.append(min(1.0 - 1e-6, max(1e-6, 1.0 / (1.0 + math.exp(-x)))))
    prev = [v0] + vs[:-1]
    dv = [a - b for a, b in zip(vs, prev)]
    n = len(dv)
    mean = sum(dv) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in dv) / n)
    s = (adv_sign > 0) - (adv_sign < 0)
    out = []
    for x in dv:
        z = (x - mean) / (std + eps)
        m = min(1.0 + band, max(1.0 - band, 1.0 + band * s * z))
        out.append((1.0 - lam) + lam * m)
    return out


# =============================================================================
# add_loss group (SDAR)
# =============================================================================


def sdar_gate_values(delta: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    """SDAR (ZJU-REAL/SDAR sdar_utils.py:14-67): per-token confidence gate
    g_t = sigmoid(beta * delta_t), detached by the caller. Soft and
    threshold-free: 0.5 at agreement, -> 1 where the privileged teacher is
    more confident. The SDAR objective ADDS coef * token_mean(g_t *
    (lpT_t - lpS_t)) to the PPO-clip loss -- advantages untouched; only lpS
    carries gradient, so the term is a gate-weighted likelihood boost on
    the student's own sampled tokens. Shipped coef 0.01, beta 5."""
    return torch.sigmoid(beta * delta)


# =============================================================================
# selection group (PivotRL)
# =============================================================================


def pivotrl_select(rows: list[dict], v_lo: float = 0.05, v_hi: float = 0.95,
                   prefer_low_mean: bool = True) -> list[dict]:
    """PivotRL (arXiv 2603.21383): keep intermediate states whose LOCAL
    rollouts have MIXED outcomes -- the Stage-1 measured V_r (success rate
    of K student continuations from h_r) strictly inside (v_lo, v_hi);
    zero-variance states (all-fail, all-win) carry no group-normalized
    gradient and are dropped, which is the paper's filter. Sorted by V_r
    ascending when prefer_low_mean (hard-but-learnable first). Training is
    then STOCK reset-GRPO from the kept states (short local rollouts);
    rows without a 'V_r' field are skipped. PAPER-LEVEL thresholds: their
    exact K / cutoffs were not retrievable -- ours reuse Stage-1's K=16."""
    kept = [r for r in rows if r.get("V_r") is not None and v_lo < r["V_r"] < v_hi]
    if prefer_low_mean:
        kept.sort(key=lambda r: r["V_r"])
    return kept
