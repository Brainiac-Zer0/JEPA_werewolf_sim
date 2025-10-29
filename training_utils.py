# training_utils.py ── utility helpers for JEPA training, determinism & checkpoint I/O
from __future__ import annotations

import os
import random
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Iterable, DefaultDict
from collections import defaultdict

import numpy as np
import torch, yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# AMP (config-toggled)
from torch.cuda import amp as torch_amp

from encoders import ActionEncoder, PlannerHead, WorldModelMLP
# Phase-aware + factorized planner path
from encoders import PhaseActionEncoder  # learnable, persisted
try:
    from encoders import FactorizedPlanner, TalkHead, VoteHead, KillHead  # optional convenience
except Exception:
    # Fallback shim if FactorizedPlanner isn't exported; assumes Talk/Vote/Kill are available
    class FactorizedPlanner(nn.Module):
        def __init__(self, latent_dim: int, num_agents: int, num_talk_cats: int):
            super().__init__()
            self.talk = TalkHead(latent_dim, num_talk_cats)
            self.vote = VoteHead(latent_dim, num_agents)
            self.kill = KillHead(latent_dim, num_agents)

        def forward(
            self,
            z: torch.Tensor,
            *,
            talk_mask: Optional[torch.Tensor] = None,
            vote_mask: Optional[torch.Tensor] = None,
            kill_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            return {
                "talk": self.talk(z, talk_mask),
                "vote": self.vote(z, vote_mask),
                "kill": self.kill(z, kill_mask),
            }

# ── Optional mouthpiece controllers (if present)
try:
    # SpeakerBandit: produces logits over templates given latent/role/history features
    # LogitBiasHead: produces token-level biases or category biases; API varies by repo
    from speaker_llm import SpeakerBandit, LogitBiasHead  # type: ignore
except Exception:
    SpeakerBandit = None  # type: ignore
    LogitBiasHead = None  # type: ignore

# Optional language coupling helpers (only used if present)
# Prefer SocialInfluence from social.py; fall back to encoders for BC
try:
    from social import SocialInfluence  # type: ignore
except Exception:
    try:
        from encoders import SocialInfluence  # type: ignore
    except Exception:
        SocialInfluence = None  # type: ignore

try:
    from encoders import MessageEncoder  # type: ignore
except Exception:
    MessageEncoder = None  # type: ignore

CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# Tunables
LAMBDA_BC: float = float(CFG.get("LAMBDA_BC", 0.5))            # used for vote/kill heads (and legacy BC)
LAMBDA_TALK: float = float(CFG.get("LAMBDA_TALK", LAMBDA_BC))  # CE weight for talk head (defaults to LAMBDA_BC)
MAX_NORM: float = float(CFG.get("MAX_NORM", 1.0))

# Optional dims (used by helpers below; keep legacy defaults)
PHASE_COUNT: int = int(CFG.get("PHASE_COUNT", 3))              # DISCUSS/VOTE/NIGHT
NUM_TALK_CATS: int = int(CFG.get("NUM_TALK_CATS", 5))
NUM_AGENTS_CFG: int = int(CFG.get("NUM_AGENTS", 6))
ACTION_EMBED_DIM: int = int(CFG.get("ACTION_DIM", CFG.get("ACTION_EMBED_DIM", 8)))  # prefer ACTION_DIM if present
LATENT_DIM: int = int(CFG.get("LATENT_DIM", 32))

# New Phase-4/5 switches
USE_AMP: bool = bool(CFG.get("USE_AMP", False))
SCHEDULER: str = str(CFG.get("LR_SCHEDULER", "none")).lower()  # 'none'|'cosine'|'onecycle'
SAVE_OPTIM: bool = bool(CFG.get("SAVE_OPTIM", False))
DEBUG_MASKS: bool = bool(CFG.get("DEBUG_MASKS", False))
STRATIFY_PHASE_BATCHES: bool = bool(CFG.get("STRATIFY_PHASE_BATCHES", True))

# ── Phase-5: mouthpiece & rewards config (defaults are conservative)
RW = CFG.get("MOUTHPIECE_REWARD_WEIGHTS", {}) or {}
R_COHE   = float(RW.get("coherence", 0.40))
R_ROLE   = float(RW.get("role_alignment", 0.30))
R_SAFE   = float(RW.get("social_safety", 0.15))
R_TRUTH  = float(RW.get("truthfulness_villager", 0.15))
R_DECEP  = float(RW.get("deception_wolf", 0.30))  # applied to (1 - truthfulness)
R_ALIGN  = float(RW.get("talk_vote_alignment", 0.10))  # extra bonus

# ── Phase-5: reward/rubric alignment guard (default False to match rubric)
REWARD_USE_ROLE_ALIGNMENT: bool = bool(CFG.get("REWARD_USE_ROLE_ALIGNMENT", False))

# ── Lambda-weighted speaker reward config (and optional night convergence bonus)
SPK = CFG.get("speaker", {}) if isinstance(CFG.get("speaker", {}), dict) else {}
LJ = float(SPK.get("lambda_j", 1.0))
LC = float(SPK.get("lambda_c", 0.25))
LO = float(SPK.get("lambda_o", 0.2))

MB = CFG.get("MOUTHPIECE_TRAINING", {}) or {}
SB_LR       = float(MB.get("speaker_bandit_lr", 5e-4))
SB_EPOCHS   = int(MB.get("speaker_bandit_epochs", 1))
SB_ENTROPY  = float(MB.get("speaker_bandit_entropy_coef", 0.01))
SB_BASE_EMA = float(MB.get("speaker_bandit_baseline_ema", 0.9))
# NEW (Phase-5): small KL to TalkHead prior and arg-aux weight
SB_KL_TO_INTENT = float(MB.get("speaker_bandit_kl_to_intent", 0.01))
SB_ARG_AUX_WEIGHT = float(MB.get("speaker_bandit_arg_aux_weight", 0.25))

BH_LR      = float(MB.get("bias_head_lr", 5e-4))
BH_EPOCHS  = int(MB.get("bias_head_epochs", 1))
BH_ENT_REG = float(MB.get("bias_head_entropy_reg", 0.005))
BH_KL_REG  = float(MB.get("bias_head_kl_reg", 0.0))
# NEW (Phase-5): align bias categories to TalkHead prior (KL)
BH_KL_TO_INTENT = float(MB.get("bias_head_kl_to_intent", 0.01))

LANG_COUP = CFG.get("LANGUAGE_COUPLING", {}) or {}
LC_ENABLED  = bool(LANG_COUP.get("enabled", False))
LC_WEIGHT   = float(LANG_COUP.get("loss_weight", 0.05))

# =============================================================================
# Determinism & run metadata
# =============================================================================

def set_global_determinism(seed: int = 1337) -> None:
    """Make training runs reproducible (as much as PyTorch allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

def save_run_config(run_id: str, cfg: Dict[str, Any]) -> str:
    """Snapshot the effective config for auditability."""
    run_dir = os.path.join(LOGS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, "config.snapshot.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return out

# =============================================================================
# Training epoch logger (CSV)
# =============================================================================

class TrainingEpochLogger:
    """Append epoch-level metrics to logs/metrics_train.csv."""
    def __init__(self, csv_path: str = os.path.join(LOGS_DIR, "metrics_train.csv")):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "run_id","role","epoch","L_mse","L_bc",
                        "grad_norm","lr","n_batches"
                    ],
                )
                w.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id","role","epoch","L_mse","L_bc",
                    "grad_norm","lr","n_batches"
                ],
            )
            w.writerow(row)

# =============================================================================
# Metrics helpers (acc & illegal mass)
# =============================================================================

@torch.no_grad()
def top1_acc(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute top-1 accuracy with an optional boolean mask applied to logits.
    """
    if mask is not None:
        # Ensure broadcast safety
        if logits.dim() == 2 and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand_as(logits)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
    else:
        masked_logits = logits
    pred = masked_logits.argmax(dim=-1)
    correct = (pred == targets).float().mean().item()
    return float(correct)

@torch.no_grad()
def illegal_mass(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    """
    Softmax probability mass on illegal indices (diagnostic; lower is better).
    """
    if mask is None:
        return 0.0
    if logits.dim() == 2 and mask.dim() == 1:
        mask = mask.unsqueeze(0).expand_as(logits)
    probs = torch.softmax(logits, dim=-1)
    return float(probs.masked_fill(mask, 0.0).sum().item())

# =============================================================================
# Rollout normalization (legacy & phase-aware)
# =============================================================================

@dataclass
class RolloutSample:
    z_t: torch.Tensor
    a_idx: Optional[torch.Tensor]
    z_next: torch.Tensor
    role: str
    # Phase-aware (optional)
    phase: Optional[int] = None              # 0..PHASE_COUNT-1
    payload_idx: Optional[int] = None        # talk_cat or target idx
    choice_type: Optional[str] = None        # "TALK_INTENT"|"VOTE_TARGET"|"KILL_TARGET"
    # Optional per-row meta for masks (alive flags, wolf team, self idx, etc.)
    aux: Optional[Dict[str, Any]] = None

def normalize_rollouts(raw: List[Tuple]) -> List[RolloutSample]:
    """
    Accepts:
      - legacy: (z_t, a_idx, z_next, role)
      - phase-aware: (z_t, phase_code, action_payload, z_{t+1}, role[, choice_type[, aux_meta_dict]])
    """
    out: List[RolloutSample] = []
    for tup in raw:
        if len(tup) == 4:
            z_t, a_idx, z_next, role = tup
            out.append(RolloutSample(z_t=z_t, a_idx=a_idx, z_next=z_next, role=role))
        else:
            # phase-aware path
            z_t, phase, payload, z_next, role = tup[:5]
            ct = tup[5] if len(tup) >= 6 else None
            aux = tup[6] if len(tup) >= 7 and isinstance(tup[6], dict) else None
            out.append(RolloutSample(
                z_t=z_t, a_idx=None, z_next=z_next, role=role,
                phase=int(phase), payload_idx=int(payload), choice_type=ct, aux=aux
            ))
    return out

# =============================================================================
# Optimizer & scheduler factory
# =============================================================================

def make_optimizer_and_scheduler(
    params: Iterable[nn.Parameter],
    learning_rate: float,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    opt = optim.Adam(list(params), lr=learning_rate)
    sched: Optional[optim.lr_scheduler._LRScheduler] = None
    total_steps = max(1, steps_per_epoch * epochs)
    if SCHEDULER == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    elif SCHEDULER == "onecycle":
        sched = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1
        )
    return opt, sched

# =============================================================================
# Phase-stratified batching (optional)
# =============================================================================

def _roundrobin_pop(queues: Dict[int, List[RolloutSample]], batch_size: int) -> List[RolloutSample]:
    out: List[RolloutSample] = []
    keys = sorted(queues.keys())
    k = 0
    while len(out) < batch_size and any(queues.values()):
        key = keys[k % len(keys)]
        if queues[key]:
            out.append(queues[key].pop())
        k += 1
    return out

def make_batches(
    rows: List[RolloutSample], batch_size: int
) -> List[List[RolloutSample]]:
    if not STRATIFY_PHASE_BATCHES:
        return [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
    # Group by phase (default bucket for None)
    buckets: DefaultDict[int, List[RolloutSample]] = defaultdict(list)
    for r in rows:
        buckets[int(0 if r.phase is None else r.phase)].append(r)
    # Shuffle within buckets for randomness
    for b in buckets.values():
        random.shuffle(b)
    # Build batches round-robin
    batches: List[List[RolloutSample]] = []
    remaining = sum(len(v) for v in buckets.values())
    while remaining > 0:
        batch = _roundrobin_pop(buckets, batch_size)
        if not batch:
            break
        batches.append(batch)
        remaining = sum(len(v) for v in buckets.values())
    return batches

# =============================================================================
# Legacy trainer (baseline) + optional CSV logging
# =============================================================================

def train_jepa(
    rollout_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    world_model: WorldModelMLP,
    action_encoder: ActionEncoder,
    planner: PlannerHead,
    role_name: str = "agent",
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    run_id: str = "run",
    epoch_logger: Optional[TrainingEpochLogger] = None,
) -> None:
    if not rollout_data:
        print(f"[{role_name}] No rollout data; skipping JEPA update.")
        return

    world_model.to(DEVICE)
    action_encoder.to(DEVICE)
    planner.to(DEVICE)

    # Simple chunk count for scheduler; recomputed each epoch
    steps_per_epoch = max(1, math.ceil(len(rollout_data)/batch_size))
    optimizer, scheduler = make_optimizer_and_scheduler(
        list(world_model.parameters()) + list(action_encoder.parameters()) + list(planner.parameters()),
        learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
    scaler = torch_amp.GradScaler(enabled=USE_AMP)

    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train(), action_encoder.train(), planner.train()

    for ep in range(1, epochs + 1):
        random.shuffle(rollout_data)
        batches = [rollout_data[i:i+batch_size] for i in range(0, len(rollout_data), batch_size)]

        epoch_mse = 0.0
        epoch_bc  = 0.0
        total_gn  = 0.0

        for batch in batches:
            z_t, a_idx, z_next = zip(*[(r[0], r[1], r[2]) for r in batch])
            z_t_tensor     = torch.stack(z_t).to(DEVICE)                     # [B, latent]
            a_idx_tensor   = torch.stack(a_idx).long().squeeze().to(DEVICE)  # [B]
            z_next_tensor  = torch.stack(z_next).to(DEVICE)                   # [B, latent]

            with torch_amp.autocast(enabled=USE_AMP):
                a_embed = action_encoder(a_idx_tensor)                        # [B, a_dim]
                z_pred  = world_model(z_t_tensor, a_embed)                    # [B, latent]
                logits  = planner(z_t_tensor)                                 # [B, num_agents]
                L_mse = mse_loss(z_pred, z_next_tensor)
                L_bc  = ce_loss(logits, a_idx_tensor)

                # Optional language coupling (legacy has no texts; stays zero unless you extend rollouts)
                L_lc = torch.tensor(0.0, device=DEVICE)

                loss  = L_mse + LAMBDA_BC * L_bc + L_lc

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # NaN guard + grad clipping
            params = list(world_model.parameters()) + list(action_encoder.parameters()) + list(planner.parameters())
            for p in params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            epoch_mse += L_mse.item()
            epoch_bc  += L_bc.item()
            total_gn  += float(gnorm)

        denom = max(1, len(batches))
        mean_mse = epoch_mse/denom
        mean_bc  = epoch_bc/denom
        mean_gn  = total_gn/denom if denom else 0.0
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(f"[{role_name}  Epoch {ep}/{epochs}]  MSE: {mean_mse:.4f}  BC: {mean_bc:.4f}  |grad|: {mean_gn:.3f}")

        if epoch_logger is not None:
            epoch_logger.log({
                "run_id": run_id,
                "role": role_name,
                "epoch": ep,
                "L_mse": round(mean_mse, 6),
                "L_bc": round(mean_bc, 6),
                "grad_norm": round(mean_gn, 6),
                "lr": cur_lr,
                "n_batches": len(batches),
            })

    # save
    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa.pt")
    state = {
        "world_model": world_model.state_dict(),
        "action_encoder": action_encoder.state_dict(),
        "planner": planner.state_dict(),
    }
    if SAVE_OPTIM:
        state["optim"] = optimizer.state_dict()
        if scheduler is not None:
            state["sched"] = scheduler.state_dict()
    torch.save(state, save_path)
    print(f"[SAVE] {role_name} models saved → {save_path}")

# =============================================================================
# Phase-aware trainer (bridge → learnable PAE)
# =============================================================================

def train_jepa_phaseaware(
    rollout_data_phaseaware: List[Tuple],  # (z_t, phase_code, action_payload, z_next, role[, choice_type])
    world_model: WorldModelMLP,
    planner: PlannerHead,                  # legacy head; optional BC on vote/kill
    *,
    role_name: str = "agent",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    run_id: str = "run",
    epoch_logger: Optional[TrainingEpochLogger] = None,
    phase_action_encoder: Optional[PhaseActionEncoder] = None,
) -> None:
    rows = normalize_rollouts(rollout_data_phaseaware)
    if not rows:
        print(f"[{role_name}] No rollout data; skipping JEPA (phase-aware) update.")
        return

    pae = phase_action_encoder or PhaseActionEncoder(
        action_dim=ACTION_EMBED_DIM,
        num_agents=NUM_AGENTS_CFG,
        num_talk=NUM_TALK_CATS,
    )

    world_model.to(DEVICE)
    planner.to(DEVICE)
    pae.to(DEVICE)

    batches_dummy = max(1, math.ceil(len(rows)/batch_size))
    optimizer, scheduler = make_optimizer_and_scheduler(
        list(world_model.parameters()) + list(planner.parameters()) + list(pae.parameters()),
        learning_rate,
        steps_per_epoch=batches_dummy,
        epochs=epochs,
    )
    scaler = torch_amp.GradScaler(enabled=USE_AMP)

    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train(), planner.train(), pae.train()

    # ---- Language coupling singletons (lazy) ---------------------------------
    _LC_MSG = None  # type: Optional[Any]
    _LC_SOC = None  # type: Optional[nn.Module]

    def _ensure_lang_coupling_modules() -> Tuple[Optional[Any], Optional[nn.Module]]:
        nonlocal _LC_MSG, _LC_SOC
        if not LC_ENABLED or MessageEncoder is None or SocialInfluence is None:
            return None, None
        if _LC_MSG is None:
            msg_model = (CFG.get("model", {})
                           .get("encoders", {})
                           .get("message_model", "sentence-transformers/all-MiniLM-L6-v2"))
            _LC_MSG = MessageEncoder(model_name=msg_model)
        if _LC_SOC is None:
            soc_cfg = (CFG.get("sim", {}).get("social", {}) or {})
            hidden = int(soc_cfg.get("hidden", 64))
            scale  = float(soc_cfg.get("scale", 0.2))
            _LC_SOC = SocialInfluence(text_dim=_LC_MSG.output_dim, latent_dim=LATENT_DIM, hidden=hidden, scale=scale)
            _LC_SOC.to(DEVICE)
        return _LC_MSG, _LC_SOC

    def _collect_texts_from_batch(batch: List[RolloutSample]) -> list:
        texts: list = []
        for r in batch:
            aux = r.aux or {}
            if isinstance(aux, dict):
                if "recent_texts" in aux and isinstance(aux["recent_texts"], list):
                    texts.extend([t for t in aux["recent_texts"] if isinstance(t, str)])
                elif "neighbor_texts" in aux and isinstance(aux["neighbor_texts"], list):
                    texts.extend([t for t in aux["neighbor_texts"] if isinstance(t, str)])
        return texts

    for ep in range(1, epochs + 1):
        random.shuffle(rows)
        batches = make_batches(rows, batch_size)

        epoch_mse = 0.0
        epoch_bc  = 0.0
        total_gn  = 0.0

        for batch in batches:
            z_t_tensor    = torch.stack([r.z_t for r in batch]).to(DEVICE)
            z_next_tensor = torch.stack([r.z_next for r in batch]).to(DEVICE)

            phase = torch.tensor([r.phase if r.phase is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            payload = torch.tensor([r.payload_idx if r.payload_idx is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            choice_types = [r.choice_type for r in batch]

            # Build learned phase-aware action embedding
            is_talk_mask = torch.tensor([ct == "TALK_INTENT" for ct in choice_types], device=DEVICE, dtype=torch.bool)
            a_embed = torch.zeros(z_t_tensor.size(0), ACTION_EMBED_DIM, device=DEVICE)
            if is_talk_mask.any():
                a_embed[is_talk_mask] = pae(
                    phase_code=phase[is_talk_mask],
                    payload_idx=payload[is_talk_mask],
                    is_talk=True,
                )
            if (~is_talk_mask).any():
                a_embed[~is_talk_mask] = pae(
                    phase_code=phase[~is_talk_mask],
                    payload_idx=payload[~is_talk_mask],
                    is_talk=False,
                )

            with torch_amp.autocast(enabled=USE_AMP):
                z_pred = world_model(z_t_tensor, a_embed)
                L_mse = mse_loss(z_pred, z_next_tensor)

                # Optional planner BC against vote/kill payloads (skip talk)
                mask_idx = [i for i, ct in enumerate(choice_types) if ct in (None, "VOTE_TARGET", "KILL_TARGET")]
                if mask_idx:
                    idx_tensor = torch.tensor(mask_idx, device=DEVICE, dtype=torch.long)
                    logits = planner(z_t_tensor[idx_tensor])
                    targets = payload[idx_tensor].clamp(0, NUM_AGENTS_CFG-1)
                    L_bc = ce_loss(logits, targets)
                else:
                    L_bc = torch.tensor(0.0, device=DEVICE)

                # ---- Optional language coupling --------------------------------
                L_lc = torch.tensor(0.0, device=DEVICE)
                if LC_ENABLED:
                    msg_enc, soc = _ensure_lang_coupling_modules()
                    if msg_enc is not None and soc is not None:
                        texts = _collect_texts_from_batch(batch)
                        if texts:
                            L_lc = language_coupling_loss(
                                z_next=z_next_tensor,
                                texts=texts,
                                msg_encoder=msg_enc, social=soc, weight=LC_WEIGHT
                            )

                # ---- Optional Stage-A social JEPA regularizer -----------------
                L_soc = torch.tensor(0.0, device=DEVICE)
                try:
                    lambda_reg = float(CFG.get("social", {}).get("lambda_reg", 0.0))
                except Exception:
                    lambda_reg = 0.0
                if lambda_reg > 0.0:
                    vals = []
                    for r in batch:
                        if r.aux and ("delta_social_norm2" in r.aux):
                            try:
                                vals.append(float(r.aux["delta_social_norm2"]))
                            except Exception:
                                pass
                    if vals:
                        L_soc = torch.tensor(sum(vals)/len(vals), device=DEVICE)

                loss = L_mse + LAMBDA_BC * L_bc + L_lc + (lambda_reg * L_soc)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            params = list(world_model.parameters()) + list(planner.parameters()) + list(pae.parameters())
            for p in params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            epoch_mse += L_mse.item()
            epoch_bc  += float(L_bc.item())
            total_gn  += float(gnorm)

        denom = max(1, len(batches))
        mean_mse = epoch_mse/denom
        mean_bc  = epoch_bc/denom
        mean_gn  = total_gn/denom if denom else 0.0
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(f"[{role_name} (phase) Epoch {ep}/{epochs}]  MSE: {mean_mse:.4f}  BC: {mean_bc:.4f}  |grad|: {mean_gn:.3f}")

        if epoch_logger is not None:
            epoch_logger.log({
                "run_id": run_id,
                "role": f"{role_name}-phase",
                "epoch": ep,
                "L_mse": round(mean_mse, 6),
                "L_bc": round(mean_bc, 6),
                "grad_norm": round(mean_gn, 6),
                "lr": cur_lr,
                "n_batches": len(batches),
            })

    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa_phase.pt")
    state = {
        "world_model": world_model.state_dict(),
        "planner": planner.state_dict(),
        "phase_action_encoder": pae.state_dict(),
    }
    if SAVE_OPTIM:
        state["optim"] = optimizer.state_dict()
        if scheduler is not None:
            state["sched"] = scheduler.state_dict()
    torch.save(state, save_path)
    print(f"[SAVE] {role_name} (phase-aware) models saved → {save_path}")

# =============================================================================
# Factorized/Masked heads trainer (TalkHead, VoteHead, KillHead)
# =============================================================================

def _mask_all_true(shape: Tuple[int, int]) -> torch.Tensor:
    return torch.ones(*shape, dtype=torch.bool, device=DEVICE)

def build_talk_mask(batch_size: int, num_cats: int) -> torch.Tensor:
    """Default: allow all talk categories."""
    return _mask_all_true((batch_size, num_cats))

def build_vote_mask_from_aux(batch: List[RolloutSample], num_agents: int) -> torch.Tensor:
    """
    Build per-row vote mask using optional aux:
      aux.alive: list[bool] length=num_agents
      aux.self_idx: int
    If aux missing, allow all.
    """
    B = len(batch)
    mask = _mask_all_true((B, num_agents))
    for i, r in enumerate(batch):
        aux = r.aux or {}
        alive = aux.get("alive", None)
        self_idx = aux.get("self_idx", None)
        if isinstance(alive, list) and len(alive) == num_agents:
            alive_t = torch.tensor(alive, dtype=torch.bool, device=DEVICE)
            mask[i] &= alive_t
        if isinstance(self_idx, int) and 0 <= self_idx < num_agents:
            mask[i, self_idx] = False
    return mask

def build_kill_mask_from_aux(batch: List[RolloutSample], num_agents: int) -> torch.Tensor:
    """
    Build per-row kill mask using optional aux:
      aux.alive: list[bool]
      aux.self_idx: int
      aux.wolves: list[bool]  (team membership)
    Behavior:
      - wolf actor: can kill alive non-wolves (and not self, though self is filtered by non-wolf rule anyway)
      - non-wolf actor: mask all False (no-ops); if aux missing, we conservatively allow all.
    """
    B = len(batch)
    mask = _mask_all_true((B, num_agents))  # default permissive if aux missing
    for i, r in enumerate(batch):
        aux = r.aux or {}
        alive = aux.get("alive", None)
        wolves = aux.get("wolves", None)
        self_idx = aux.get("self_idx", None)

        # If we don't have meta, leave permissive mask (training won't explode).
        if not (isinstance(alive, list) and len(alive) == num_agents):
            continue

        alive_t = torch.tensor(alive, dtype=torch.bool, device=DEVICE)

        if isinstance(wolves, list) and len(wolves) == num_agents and isinstance(self_idx, int):
            wolves_t = torch.tensor(wolves, dtype=torch.bool, device=DEVICE)
            is_wolf_actor = bool(wolves[self_idx])
            if is_wolf_actor:
                # wolves can only kill alive non-wolves
                mask[i] &= alive_t & (~wolves_t)
            else:
                # non-wolves do not issue kill; mask all false
                mask[i, :] = False
        else:
            # Missing team info; at least don't kill dead
            mask[i] &= alive_t
    return mask

def train_jepa_factorized(
    rollout_data_phaseaware: List[Tuple],
    world_model: WorldModelMLP,
    phase_action_encoder: Optional[PhaseActionEncoder],
    planner_factorized: FactorizedPlanner,
    *,
    role_name: str = "agent",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    run_id: str = "run",
    epoch_logger: Optional[TrainingEpochLogger] = None,
) -> None:
    """
    Full phase-aware + factorized-head trainer.
    Expects per-row choice_type ∈ {"TALK_INTENT","VOTE_TARGET","KILL_TARGET"} (or None).
    Optionally consumes per-row aux dicts to build action masks.
    """
    rows = normalize_rollouts(rollout_data_phaseaware)
    if not rows:
        print(f"[{role_name}] No rollout data; skipping JEPA (factorized) update.")
        return

    pae = phase_action_encoder or PhaseActionEncoder(
        action_dim=ACTION_EMBED_DIM,
        num_agents=NUM_AGENTS_CFG,
        num_talk=NUM_TALK_CATS,
    )

    world_model.to(DEVICE)
    planner_factorized.to(DEVICE)
    pae.to(DEVICE)

    batches_dummy = max(1, math.ceil(len(rows)/batch_size))
    optimizer, scheduler = make_optimizer_and_scheduler(
        list(world_model.parameters()) + list(planner_factorized.parameters()) + list(pae.parameters()),
        learning_rate,
        steps_per_epoch=batches_dummy,
        epochs=epochs,
    )
    scaler = torch_amp.GradScaler(enabled=USE_AMP)

    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train()
    planner_factorized.train()
    pae.train()

    # ---- Language coupling singletons (lazy) ---------------------------------
    _LC_MSG = None  # type: Optional[Any]
    _LC_SOC = None  # type: Optional[nn.Module]

    def _ensure_lang_coupling_modules() -> Tuple[Optional[Any], Optional[nn.Module]]:
        nonlocal _LC_MSG, _LC_SOC
        if not LC_ENABLED or MessageEncoder is None or SocialInfluence is None:
            return None, None
        if _LC_MSG is None:
            msg_model = (CFG.get("model", {})
                           .get("encoders", {})
                           .get("message_model", "sentence-transformers/all-MiniLM-L6-v2"))
            _LC_MSG = MessageEncoder(model_name=msg_model)
        if _LC_SOC is None:
            soc_cfg = (CFG.get("sim", {}).get("social", {}) or {})
            hidden = int(soc_cfg.get("hidden", 64))
            scale  = float(soc_cfg.get("scale", 0.2))
            _LC_SOC = SocialInfluence(text_dim=_LC_MSG.output_dim, latent_dim=LATENT_DIM, hidden=hidden, scale=scale)
            _LC_SOC.to(DEVICE)
        return _LC_MSG, _LC_SOC

    def _collect_texts_from_batch(batch: List[RolloutSample]) -> list:
        texts: list = []
        for r in batch:
            aux = r.aux or {}
            if isinstance(aux, dict):
                if "recent_texts" in aux and isinstance(aux["recent_texts"], list):
                    texts.extend([t for t in aux["recent_texts"] if isinstance(t, str)])
                elif "neighbor_texts" in aux and isinstance(aux["neighbor_texts"], list):
                    texts.extend([t for t in aux["neighbor_texts"] if isinstance(t, str)])
        return texts

    for ep in range(1, epochs + 1):
        random.shuffle(rows)
        batches = make_batches(rows, batch_size)

        epoch_mse = 0.0
        epoch_talk = 0.0
        epoch_vote = 0.0
        epoch_kill = 0.0
        total_gn  = 0.0

        for batch in batches:
            B = len(batch)
            z_t_tensor    = torch.stack([r.z_t for r in batch]).to(DEVICE)
            z_next_tensor = torch.stack([r.z_next for r in batch]).to(DEVICE)
            phase = torch.tensor([r.phase if r.phase is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            payload = torch.tensor([r.payload_idx if r.payload_idx is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            choice_types = [r.choice_type for r in batch]

            # === Action embeddings (learned)
            is_talk_mask = torch.tensor([ct == "TALK_INTENT" for ct in choice_types], device=DEVICE, dtype=torch.bool)
            a_embed = torch.zeros(B, ACTION_EMBED_DIM, device=DEVICE)
            if is_talk_mask.any():
                a_embed[is_talk_mask] = pae(phase_code=phase[is_talk_mask], payload_idx=payload[is_talk_mask], is_talk=True)
            if (~is_talk_mask).any():
                a_embed[~is_talk_mask] = pae(phase_code=phase[~is_talk_mask], payload_idx=payload[~is_talk_mask], is_talk=False)

            # === Build masks for the factorized heads
            talk_mask = build_talk_mask(B, NUM_TALK_CATS)
            vote_mask = build_vote_mask_from_aux(batch, NUM_AGENTS_CFG)
            kill_mask = build_kill_mask_from_aux(batch, NUM_AGENTS_CFG)

            with torch_amp.autocast(enabled=USE_AMP):
                # === World model (JEPA)
                z_pred = world_model(z_t_tensor, a_embed)
                L_mse = mse_loss(z_pred, z_next_tensor)

                # === Factorized planner forward
                logits = planner_factorized(
                    z_t_tensor,
                    talk_mask=talk_mask,
                    vote_mask=vote_mask,
                    kill_mask=kill_mask
                )

                # === Per-head CE losses (only where that head applies)
                is_vote_mask = torch.tensor([ct == "VOTE_TARGET" for ct in choice_types], device=DEVICE, dtype=torch.bool)
                is_kill_mask = torch.tensor([ct == "KILL_TARGET" for ct in choice_types], device=DEVICE, dtype=torch.bool)

                if DEBUG_MASKS:
                    # Assert targets are legal where supervised
                    if is_vote_mask.any():
                        illegal = ~vote_mask[is_vote_mask, payload[is_vote_mask]]
                        assert not illegal.any().item(), "Vote target illegal under vote_mask"
                    if is_kill_mask.any():
                        illegal = ~kill_mask[is_kill_mask, payload[is_kill_mask]]
                        assert not illegal.any().item(), "Kill target illegal under kill_mask"

                L_talk = torch.tensor(0.0, device=DEVICE)
                L_vote = torch.tensor(0.0, device=DEVICE)
                L_kill = torch.tensor(0.0, device=DEVICE)

                if is_talk_mask.any():
                    L_talk = ce_loss(logits["talk"][is_talk_mask], payload[is_talk_mask])
                if is_vote_mask.any():
                    L_vote = ce_loss(logits["vote"][is_vote_mask], payload[is_vote_mask])
                if is_kill_mask.any():
                    L_kill = ce_loss(logits["kill"][is_kill_mask], payload[is_kill_mask])

                # ---- Optional language coupling --------------------------------
                L_lc = torch.tensor(0.0, device=DEVICE)
                if LC_ENABLED:
                    msg_enc, soc = _ensure_lang_coupling_modules()
                    if msg_enc is not None and soc is not None:
                        texts = _collect_texts_from_batch(batch)
                        if texts:
                            L_lc = language_coupling_loss(
                                z_next=z_next_tensor,
                                texts=texts,
                                msg_encoder=msg_enc, social=soc, weight=LC_WEIGHT
                            )

                # ---- Optional Stage-A social JEPA regularizer -----------------
                L_soc = torch.tensor(0.0, device=DEVICE)
                try:
                    lambda_reg = float(CFG.get("social", {}).get("lambda_reg", 0.0))
                except Exception:
                    lambda_reg = 0.0
                if lambda_reg > 0.0:
                    vals = []
                    for r in batch:
                        if r.aux and ("delta_social_norm2" in r.aux):
                            try:
                                vals.append(float(r.aux["delta_social_norm2"]))
                            except Exception:
                                pass
                    if vals:
                        L_soc = torch.tensor(sum(vals)/len(vals), device=DEVICE)

                loss = L_mse + (LAMBDA_TALK * L_talk) + (LAMBDA_BC * (L_vote + L_kill)) + L_lc + (lambda_reg * L_soc)

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            params = list(world_model.parameters()) + list(planner_factorized.parameters()) + list(pae.parameters())
            for p in params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            epoch_mse  += float(L_mse.item())
            epoch_talk += float(L_talk.item())
            epoch_vote += float(L_vote.item())
            epoch_kill += float(L_kill.item())
            total_gn   += float(gnorm)

        denom = max(1, len(batches))
        mean_mse  = epoch_mse/denom
        mean_talk = epoch_talk/denom
        mean_vote = epoch_vote/denom
        mean_kill = epoch_kill/denom
        mean_gn   = total_gn/denom if denom else 0.0
        cur_lr    = optimizer.param_groups[0]["lr"]

        print(
            f"[{role_name} (factorized) Epoch {ep}/{epochs}]  "
            f"MSE: {mean_mse:.4f}  TALK: {mean_talk:.4f}  VOTE: {mean_vote:.4f}  KILL: {mean_kill:.4f}  "
            f"|grad|: {mean_gn:.3f}"
        )

        if epoch_logger is not None:
            # Keep CSV compact: push (vote+kill+talk) under L_bc for compatibility
            epoch_logger.log({
                "run_id": run_id,
                "role": f"{role_name}-factorized",
                "epoch": ep,
                "L_mse": round(mean_mse, 6),
                "L_bc": round(mean_talk + mean_vote + mean_kill, 6),
                "grad_norm": round(mean_gn, 6),
                "lr": cur_lr,
                "n_batches": len(batches),
            })

    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa_factorized.pt")
    state = {
        "world_model": world_model.state_dict(),
        "phase_action_encoder": pae.state_dict(),
        "factorized_planner": planner_factorized.state_dict(),
    }
    if SAVE_OPTIM:
        state["optim"] = optimizer.state_dict()
        if scheduler is not None:
            state["sched"] = scheduler.state_dict()
    torch.save(state, save_path)
    print(f"[SAVE] {role_name} (factorized) models saved → {save_path}")

# =============================================================================
# Evaluation helpers (no optimizer)
# =============================================================================

@torch.no_grad()
def evaluate_jepa(
    rollouts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    world_model: WorldModelMLP,
    action_encoder: ActionEncoder,
    planner: PlannerHead,
    batch_size: int = 32,
) -> Dict[str, float]:
    world_model.eval(); action_encoder.eval(); planner.eval()
    if not rollouts:
        return {"mse": 0.0, "bc_ce": 0.0, "bc_acc": 0.0}
    batches = [rollouts[i:i+batch_size] for i in range(0, len(rollouts), batch_size)]
    mse_loss = nn.MSELoss(reduction="sum")
    ce_loss  = nn.CrossEntropyLoss(reduction="sum")
    tot_mse = tot_ce = tot_acc = n = 0.0
    for batch in batches:
        z_t, a_idx, z_next = zip(*[(r[0], r[1], r[2]) for r in batch])
        z_t_tensor     = torch.stack(z_t).to(DEVICE)
        a_idx_tensor   = torch.stack(a_idx).long().squeeze().to(DEVICE)
        z_next_tensor  = torch.stack(z_next).to(DEVICE)
        a_embed = action_encoder(a_idx_tensor)
        z_pred  = world_model(z_t_tensor, a_embed)
        logits  = planner(z_t_tensor)
        tot_mse += float(mse_loss(z_pred, z_next_tensor).item())
        tot_ce  += float(ce_loss(logits, a_idx_tensor).item())
        tot_acc += float((logits.argmax(-1) == a_idx_tensor).float().sum().item())
        n += z_t_tensor.size(0)
    return {
        "mse": tot_mse / n,
        "bc_ce": tot_ce / n,
        "bc_acc": tot_acc / n,
    }

@torch.no_grad()
def evaluate_jepa_phase(
    rollout_data_phaseaware: List[Tuple],
    world_model: WorldModelMLP,
    phase_action_encoder: PhaseActionEncoder,
    planner: PlannerHead,
    batch_size: int = 32,
) -> Dict[str, float]:
    rows = normalize_rollouts(rollout_data_phaseaware)
    if not rows:
        return {"mse": 0.0, "bc_ce": 0.0, "bc_acc": 0.0}
    world_model.eval(); planner.eval(); phase_action_encoder.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    ce_loss  = nn.CrossEntropyLoss(reduction="sum")
    tot_mse = tot_ce = tot_acc = n = 0.0
    batches = make_batches(rows, batch_size)
    for batch in batches:
        z_t_tensor    = torch.stack([r.z_t for r in batch]).to(DEVICE)
        z_next_tensor = torch.stack([r.z_next for r in batch]).to(DEVICE)
        phase   = torch.tensor([r.phase or 0 for r in batch], device=DEVICE)
        payload = torch.tensor([r.payload_idx or 0 for r in batch], device=DEVICE)
        choice_types = [r.choice_type for r in batch]
        is_talk_mask = torch.tensor([ct == "TALK_INTENT" for ct in choice_types], device=DEVICE, dtype=torch.bool)
        a_embed = torch.zeros(z_t_tensor.size(0), ACTION_EMBED_DIM, device=DEVICE)
        if is_talk_mask.any():
            a_embed[is_talk_mask] = phase_action_encoder(phase[is_talk_mask], payload[is_talk_mask], is_talk=True)
        if (~is_talk_mask).any():
            a_embed[~is_talk_mask] = phase_action_encoder(phase[~is_talk_mask], payload[~is_talk_mask], is_talk=False)
        z_pred = world_model(z_t_tensor, a_embed)
        tot_mse += float(mse_loss(z_pred, z_next_tensor).item())

        # optional BC on vote/kill
        mask_idx = [i for i, ct in enumerate(choice_types) if ct in (None, "VOTE_TARGET", "KILL_TARGET")]
        if mask_idx:
            idx_tensor = torch.tensor(mask_idx, device=DEVICE, dtype=torch.long)
            logits = planner(z_t_tensor[idx_tensor])
            targets = payload[idx_tensor].clamp(0, NUM_AGENTS_CFG-1)
            tot_ce += float(ce_loss(logits, targets).item())
            tot_acc += float((logits.argmax(-1) == targets).float().sum().item())
            n += float(idx_tensor.numel())
        # count n for mse separately
        # If no BC rows, n for CE/acc remains as previous.
    m = sum(len(b) for b in batches)
    return {
        "mse": tot_mse / max(1.0, float(m)),
        "bc_ce": tot_ce / max(1.0, n),
        "bc_acc": tot_acc / max(1.0, n),
    }

@torch.no_grad()
def evaluate_jepa_factorized(
    rollout_data_phaseaware: List[Tuple],
    world_model: WorldModelMLP,
    phase_action_encoder: PhaseActionEncoder,
    planner_factorized: FactorizedPlanner,
    batch_size: int = 32,
) -> Dict[str, float]:
    rows = normalize_rollouts(rollout_data_phaseaware)
    if not rows:
        return {"mse": 0.0, "talk_ce": 0.0, "vote_ce": 0.0, "kill_ce": 0.0,
                "talk_acc": 0.0, "vote_acc": 0.0, "kill_acc": 0.0,
                "vote_illegal": 0.0, "kill_illegal": 0.0}
    world_model.eval(); planner_factorized.eval(); phase_action_encoder.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    ce_loss  = nn.CrossEntropyLoss(reduction="sum")

    tot = {
        "mse": 0.0, "talk_ce": 0.0, "vote_ce": 0.0, "kill_ce": 0.0,
        "talk_acc_n": 0.0, "vote_acc_n": 0.0, "kill_acc_n": 0.0,
        "talk_acc_c": 0.0, "vote_acc_c": 0.0, "kill_acc_c": 0.0,
        "vote_illegal": 0.0, "kill_illegal": 0.0, "n": 0.0
    }

    batches = make_batches(rows, batch_size)
    for batch in batches:
        B = len(batch)
        z_t_tensor    = torch.stack([r.z_t for r in batch]).to(DEVICE)
        z_next_tensor = torch.stack([r.z_next for r in batch]).to(DEVICE)
        phase   = torch.tensor([r.phase or 0 for r in batch], device=DEVICE)
        payload = torch.tensor([r.payload_idx or 0 for r in batch], device=DEVICE)
        choice_types = [r.choice_type for r in batch]

        # action embed
        is_talk_mask = torch.tensor([ct == "TALK_INTENT" for ct in choice_types], device=DEVICE)
        a_embed = torch.zeros(B, ACTION_EMBED_DIM, device=DEVICE)
        if is_talk_mask.any():
            a_embed[is_talk_mask] = phase_action_encoder(phase[is_talk_mask], payload[is_talk_mask], is_talk=True)
        if (~is_talk_mask).any():
            a_embed[~is_talk_mask] = phase_action_encoder(phase[~is_talk_mask], payload[~is_talk_mask], is_talk=False)

        z_pred = world_model(z_t_tensor, a_embed)
        tot["mse"] += float(mse_loss(z_pred, z_next_tensor).item())
        tot["n"] += float(B)

        # masks + logits
        talk_mask = build_talk_mask(B, NUM_TALK_CATS)
        vote_mask = build_vote_mask_from_aux(batch, NUM_AGENTS_CFG)
        kill_mask = build_kill_mask_from_aux(batch, NUM_AGENTS_CFG)
        logits = planner_factorized(
            z_t_tensor, talk_mask=talk_mask, vote_mask=vote_mask, kill_mask=kill_mask
        )

        is_vote_mask = torch.tensor([ct == "VOTE_TARGET" for ct in choice_types], device=DEVICE)
        is_kill_mask = torch.tensor([ct == "KILL_TARGET" for ct in choice_types], device=DEVICE)

        if is_talk_mask.any():
            idx = is_talk_mask
            tot["talk_ce"] += float(ce_loss(logits["talk"][idx], payload[idx]).item())
            tot["talk_acc_c"] += float((logits["talk"][idx].argmax(-1) == payload[idx]).float().sum().item())
            tot["talk_acc_n"] += float(idx.sum().item())

        if is_vote_mask.any():
            idx = is_vote_mask
            tot["vote_ce"] += float(ce_loss(logits["vote"][idx], payload[idx]).item())
            tot["vote_acc_c"] += float((logits["vote"][idx].argmax(-1) == payload[idx]).float().sum().item())
            tot["vote_acc_n"] += float(idx.sum().item())
            tot["vote_illegal"] += illegal_mass(logits["vote"][idx], vote_mask[idx])

        if is_kill_mask.any():
            idx = is_kill_mask
            tot["kill_ce"] += float(ce_loss(logits["kill"][idx], payload[idx]).item())
            tot["kill_acc_c"] += float((logits["kill"][idx].argmax(-1) == payload[idx]).float().sum().item())
            tot["kill_acc_n"] += float(idx.sum().item())
            tot["kill_illegal"] += illegal_mass(logits["kill"][idx], kill_mask[idx])

    return {
        "mse": tot["mse"] / max(1.0, tot["n"]),
        "talk_ce": tot["talk_ce"] / max(1.0, tot["talk_acc_n"]),
        "vote_ce": tot["vote_ce"] / max(1.0, tot["vote_acc_n"]),
        "kill_ce": tot["kill_ce"] / max(1.0, tot["kill_acc_n"]),
        "talk_acc": tot["talk_acc_c"] / max(1.0, tot["talk_acc_n"]),
        "vote_acc": tot["vote_acc_c"] / max(1.0, tot["vote_acc_n"]),
        "kill_acc": tot["kill_acc_c"] / max(1.0, tot["kill_acc_n"]),
        "vote_illegal": tot["vote_illegal"] / max(1.0, tot["vote_acc_n"]),
        "kill_illegal": tot["kill_illegal"] / max(1.0, tot["kill_acc_n"]),
    }

# =============================================================================
# Phase-5: Utterance dataset collation + mouthpiece trainers
# =============================================================================

@dataclass
class UtteranceSample:
    # inputs
    z_t: torch.Tensor
    role: str
    round_num: int
    template_id: Optional[int]
    talk_cat: Optional[int]
    hist_feats: Optional[torch.Tensor]   # any small embedding/feature vec (or None)
    # NEW (Phase-5): fused TalkHead prior (intent probs) and argument id
    p_intent: Optional[List[float]] = None   # e.g., fused intent distribution over categories
    arg_id: Optional[int] = None             # optional argument/target chosen by the bandit
    # scored targets
    judge_score: float = 0.0
    coherence: float = 0.0
    truthfulness: float = 0.0
    role_alignment: float = 0.0
    social_safety: float = 0.0
    align_tv: Optional[float] = None         # talk→vote alignment [0..1]
    # optional lexical diversity penalty captured from runtime (if provided)
    rep_penalty: Optional[float] = None

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _role_is_wolf(role: str) -> bool:
    return (role or "").strip().lower() in ("werewolf", "wolf")

def collect_utterance_dataset(agents: List[Any]) -> List[UtteranceSample]:
    """
    Pull per-utterance training rows from agents' msg_buffer (populated in sim.py Phase-5).
    Each msg_buffer entry is expected to contain:
      - 'z' (latent at speak time) or we fallback to agent.encode_current_belief(...)
      - 'template_id' (int) if the speaker used a template-scaffold
      - 'judge_score' and 'judge_subscores' with keys: coherence, truthfulness, role_alignment, social_safety
      - 'alignment_vote' (float) for talk→vote alignment
      - 'talk_category' or agent.talk_category_last for intent supervision
      - optional 'p_intent' (List[float]) fused TalkHead prior for α-fusion logging/training
      - optional 'arg_id' (int) chosen by two-stage bandit (argument/target)
      - optional 'hist_feats' tensor-like
      - optional 'repetition_penalty' float in [0,1]
    """
    out: List[UtteranceSample] = []
    for ag in agents or []:
        buf = getattr(ag, "msg_buffer", None)
        if not buf:
            continue
        role = getattr(ag, "role", "") or ""
        for row in buf:
            z = row.get("z", None)
            if z is None:
                # best-effort fallback (rare; we prefer the exact z_t captured when talking)
                try:
                    z = ag.encode_current_belief(int(row.get("round", 0)), agents).detach()
                except Exception:
                    continue
            if not torch.is_tensor(z):
                try:
                    z = torch.tensor(z).float()
                except Exception:
                    continue
            j = row.get("judge_subscores", {}) or {}
            # Capture p_intent (list of floats) if present and sane
            p_intent = row.get("p_intent", None)
            if isinstance(p_intent, (list, tuple)) and len(p_intent) > 0:
                try:
                    p_sum = float(sum(float(x) for x in p_intent))
                    if p_sum > 0:
                        p_intent = [float(x) / p_sum for x in p_intent]
                    else:
                        p_intent = None
                except Exception:
                    p_intent = None
            else:
                p_intent = None
            # argument id (for two-stage bandit)
            arg_id = row.get("arg_id", None)
            try:
                arg_id = int(arg_id) if arg_id is not None else None
            except Exception:
                arg_id = None

            out.append(UtteranceSample(
                z_t = z.detach(),
                role = role,
                round_num = int(row.get("round", 0)),
                template_id = int(row["template_id"]) if "template_id" in row else None,
                talk_cat = int(row["talk_category"]) if "talk_category" in row else int(getattr(ag, "talk_category_last", -1)) if getattr(ag, "talk_category_last", -1) != -1 else None,
                hist_feats = row.get("hist_feats", None),
                p_intent = p_intent,
                arg_id = arg_id,
                judge_score = _safe_float(row.get("judge_score", 0.0)),
                coherence = _safe_float(j.get("coherence", 0.0)),
                truthfulness = _safe_float(j.get("truthfulness", 0.0)),
                role_alignment = _safe_float(j.get("role_alignment", 0.0)),
                social_safety = _safe_float(j.get("social_safety", 0.0)),
                align_tv = _safe_float(row.get("alignment_vote", 0.0)) if "alignment_vote" in row else None,
                rep_penalty = _safe_float(row.get("repetition_penalty", 0.0)) if "repetition_penalty" in row else None,
            ))
    return out

def _compute_reward(sample: UtteranceSample) -> float:
    """
    Role-aware scalar reward combining judge subscores and alignment.
    NOTE: role_alignment is included only if REWARD_USE_ROLE_ALIGNMENT=True
    to keep consistency with rubric (truthfulness, coherence, social_safety).
    """
    base = (R_COHE * sample.coherence) + (R_SAFE * sample.social_safety)

    if REWARD_USE_ROLE_ALIGNMENT:
        base += (R_ROLE * sample.role_alignment)

    # Truthfulness handling differs by role (wolves rewarded for deception)
    if _role_is_wolf(sample.role):
        base += (R_DECEP * max(0.0, 1.0 - sample.truthfulness))
    else:
        base += (R_TRUTH * sample.truthfulness)

    if sample.align_tv is not None:
        base += (R_ALIGN * sample.align_tv)

    # clamp to sane range
    return float(max(-1.0, min(2.0, base)))

# --- Phase-5: Lambda-weighted reward assembly (+ optional night convergence) ---
def assemble_total_reward(sample: Dict[str, Any]) -> float:
    """
    Compute λ-weighted total reward for a rollout/utterance sample:
      R_total = LJ * r_judge + LC * r_consistency + LO * r_outcome
    Optionally adds a tiny bonus for werewolves who converge during night chat.
    """
    r_judge = float(sample.get("r_judge", 0.0))
    r_consistency = float(sample.get("r_consistency", 0.0))
    r_outcome = float(sample.get("r_outcome", 0.0))

    R_total = (LJ * r_judge) + (LC * r_consistency) + (LO * r_outcome)

    # Optional night convergence reward (wolves only)
    if sample.get("phase") == "NIGHT_DISCUSS" and sample.get("role") == "Werewolf":
        r_night_consensus = float(sample.get("night_consensus", 0.0))  # 0..1 external calc if logged
        R_total += 0.05 * r_night_consensus

    sample["reward_total"] = float(R_total)
    return float(R_total)

# ─────────────────────────────────────────────────────────────────────────────
# SpeakerBandit trainer (policy gradient with EMA baseline + entropy bonus)
# ─────────────────────────────────────────────────────────────────────────────

class _BanditLogger:
    def __init__(self, csv_path: str = os.path.join(LOGS_DIR, "metrics_speaker.csv")):
        self.path = csv_path
        if not os.path.exists(self.path,):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch","loss","reward_mean","reward_std","entropy"])
                w.writeheader()
    def log(self, epoch: int, loss: float, r_mean: float, r_std: float, ent: float):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","loss","reward_mean","reward_std","entropy"])
            w.writerow({"epoch": epoch, "loss": round(loss,6), "reward_mean": round(r_mean,6), "reward_std": round(r_std,6), "entropy": round(ent,6)})

def _kl_categorical(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    KL(p||q) for categorical distributions with small epsilon for numerical stability.
    p,q are probability vectors [B,C] or [C].
    """
    if p.dim() == 1:
        p = p.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    eps = 1e-8
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=-1).mean()

def train_speaker_bandit(
    dataset: List[UtteranceSample],
    speaker: Optional[nn.Module],
    *,
    epochs: int = SB_EPOCHS,
    lr: float = SB_LR,
    entropy_coef: float = SB_ENTROPY,
    baseline_ema: float = SB_BASE_EMA,
) -> None:
    if speaker is None or SpeakerBandit is None:
        print("[MOUTHPIECE] SpeakerBandit not available; skipping bandit training.")
        return
    # Filter rows that have a template_id (required for supervised action id)
    rows = [s for s in dataset if s.template_id is not None]
    if not rows:
        print("[MOUTHPIECE] No template_id in dataset; skipping bandit training.")
        return

    speaker.to(DEVICE)
    speaker.train()
    opt = optim.Adam(speaker.parameters(), lr=lr)

    # moving baseline for variance reduction
    baseline = 0.0

    logger = _BanditLogger()

    for ep in range(1, max(1, epochs) + 1):
        random.shuffle(rows)
        total_loss = 0.0
        ent_acc = 0.0
        rewards_all: List[float] = []

        for s in rows:
            # Prepare inputs: latent + optional hist feats + role bit
            z = s.z_t.to(DEVICE)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            role_bit = torch.tensor([[1.0 if _role_is_wolf(s.role) else 0.0]], device=DEVICE)
            if s.hist_feats is None:
                hist = torch.zeros((1, 8), device=DEVICE)  # small default
            else:
                h = s.hist_feats
                if not torch.is_tensor(h):
                    h = torch.tensor(h).float()
                hist = h.to(DEVICE).view(1, -1)

            # Forward: support two-stage dict outputs with backward compatibility
            out = speaker(z, role_bit=role_bit, hist_feats=hist)
            if isinstance(out, dict):
                cat_logits = out.get("cat_logits", None)
                arg_logits = out.get("arg_logits", None)
                if cat_logits is None:  # degenerate safety
                    cat_logits = out[list(out.keys())[0]]
            else:
                cat_logits, arg_logits = out, None

            cat_probs = torch.softmax(cat_logits, dim=-1)
            cat_logprobs = torch.log_softmax(cat_logits, dim=-1)

            # pick the actually-used template_id as the "action"
            t_id = int(s.template_id)
            if t_id < 0 or t_id >= cat_probs.size(-1):
                # Skip if speaker head doesn't match dataset template space
                continue

            lp_cat = cat_logprobs[0, t_id]
            ent_cat = -(cat_probs * cat_logprobs).sum()

            # reward
            R = _compute_reward(s)
            rewards_all.append(R)
            adv = R - baseline

            # Base REINFORCE term on categories
            loss = -(adv * lp_cat) - (entropy_coef * ent_cat)

            # Optional auxiliary REINFORCE on argument/target choice if present
            if (arg_logits is not None) and (s.arg_id is not None):
                a_logits = arg_logits
                a_logprobs = torch.log_softmax(a_logits, dim=-1)
                a_id = int(s.arg_id)
                if 0 <= a_id < a_logits.size(-1):
                    lp_arg = a_logprobs[0, a_id]
                    loss = loss - (SB_ARG_AUX_WEIGHT * adv * lp_arg)  # small aux

            # Small KL(π_cat || p_intent) regularizer when p_intent present
            if s.p_intent is not None and len(s.p_intent) == cat_probs.size(-1):
                q = torch.tensor(s.p_intent, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                kl = _kl_categorical(cat_probs.detach(), q)  # use current probs; don't backprop through q
                loss = loss + SB_KL_TO_INTENT * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(speaker.parameters(), MAX_NORM)
            opt.step()

            # EMA baseline update
            baseline = baseline_ema * baseline + (1.0 - baseline_ema) * R

            total_loss += float(loss.item())
            ent_acc += float(ent_cat.item())

        n = max(1, len(rewards_all))
        logger.log(ep, total_loss / n, float(np.mean(rewards_all)) if rewards_all else 0.0, float(np.std(rewards_all)) if rewards_all else 0.0, ent_acc / n)
        print(f"[SpeakerBandit] epoch={ep} loss={total_loss/n:.4f} Rμ={np.mean(rewards_all) if rewards_all else 0.0:.3f} H={ent_acc/n:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# LogitBiasHead trainer (intent CE + reward-weighted soft targets)
# ─────────────────────────────────────────────────────────────────────────────

class _BiasLogger:
    def __init__(self, csv_path: str = os.path.join(LOGS_DIR, "metrics_bias_head.csv")):
        self.path = csv_path
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch","ce","reward_ce","entropy","kl","kl_intent"])
                w.writeheader()
    def log(self, epoch: int, ce: float, rce: float, ent: float, kl: float, kl_intent: float):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","ce","reward_ce","entropy","kl","kl_intent"])
            w.writerow({"epoch": epoch, "ce": round(ce,6), "reward_ce": round(rce,6), "entropy": round(ent,6), "kl": round(kl,6), "kl_intent": round(kl_intent,6)})

def _entropy_categorical(logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    return -(p * lp).sum(dim=-1).mean()

def _kl_to_uniform(logits: torch.Tensor) -> torch.Tensor:
    # KL(p || U) = sum p log(p / U); U=1/K => -H(p) + log K
    k = logits.size(-1)
    H = _entropy_categorical(logits)
    return (math.log(k) - H).clamp(min=0.0)

def train_bias_head_on_intents(
    dataset: List[UtteranceSample],
    bias_head: Optional[nn.Module],
    *,
    epochs: int = BH_EPOCHS,
    lr: float = BH_LR,
    ent_reg: float = BH_ENT_REG,
    kl_reg: float = BH_KL_REG,
    num_categories: int = NUM_TALK_CATS,
) -> None:
    """
    We interpret LogitBiasHead as a module that can output *category* logits given latent features.
    If your implementation exposes a different API, adapt this call-site accordingly.
    Fallback: skip with a clear message.
    """
    if bias_head is None or LogitBiasHead is None:
        print("[MOUTHPIECE] LogitBiasHead not available; skipping bias training.")
        return

    # Build supervision: only rows with a talk category label
    rows = [s for s in dataset if s.talk_cat is not None and s.talk_cat >= 0 and s.talk_cat < num_categories]
    if not rows:
        print("[MOUTHPIECE] No talk category labels; skipping bias training.")
        return

    bias_head.to(DEVICE)
    bias_head.train()
    opt = optim.Adam(bias_head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    logger = _BiasLogger()

    for ep in range(1, max(1, epochs) + 1):
        random.shuffle(rows)
        ce_sum = rce_sum = ent_sum = kl_sum = kl_intent_sum = 0.0
        n = 0

        for s in rows:
            z = s.z_t.to(DEVICE)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            role_bit = torch.tensor([[1.0 if _role_is_wolf(s.role) else 0.0]], device=DEVICE)
            # Try to call a generic `category_logits` API; fallback to `forward` if available
            if hasattr(bias_head, "category_logits"):
                logits = bias_head.category_logits(z, role_bit=role_bit)  # [1, C]
            else:
                logits = bias_head(z)  # best effort

            target = torch.tensor([int(s.talk_cat)], device=DEVICE, dtype=torch.long)

            # Plain CE to observed intent
            L_ce = ce(logits, target)

            # Reward-weighted soft target (encourage high-judge-score categories)
            with torch.no_grad():
                R = max(0.0, _compute_reward(s))
                # start near-uniform, then bump observed category proportional to reward
                soft = torch.full_like(logits, 1.0 / logits.size(-1))
                soft[0, target.item()] = min(1.0, 0.5 + 0.5 * R)
                soft = (soft / soft.sum(dim=-1, keepdim=True)).detach()

            logp = torch.log_softmax(logits, dim=-1)
            L_rce = -(soft * logp).sum(dim=-1).mean()

            # regularizers
            H = _entropy_categorical(logits)
            KL_u = _kl_to_uniform(logits)

            # KL(p_bias || p_intent) when p_intent provided
            KL_intent = torch.tensor(0.0, device=DEVICE)
            if s.p_intent is not None and len(s.p_intent) == logits.size(-1):
                p_bias = torch.softmax(logits, dim=-1)
                q = torch.tensor(s.p_intent, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                KL_intent = _kl_categorical(p_bias, q)

            loss = L_ce + L_rce + (-ent_reg * H) + (kl_reg * KL_u) + (BH_KL_TO_INTENT * KL_intent)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bias_head.parameters(), MAX_NORM)
            opt.step()

            ce_sum  += float(L_ce.item())
            rce_sum += float(L_rce.item())
            ent_sum += float(H.item())
            kl_sum  += float(KL_u.item())
            kl_intent_sum += float(KL_intent.item())
            n += 1

        n = max(1, n)
        logger.log(ep, ce_sum/n, rce_sum/n, ent_sum/n, kl_sum/n, kl_intent_sum/n)
        print(f"[BiasHead] epoch={ep} CE={ce_sum/n:.4f} RCE={rce_sum/n:.4f} H={ent_sum/n:.3f} KL_U={kl_sum/n:.3f} KL_I={kl_intent_sum/n:.3f}")

# =============================================================================
# Mouthpiece checkpoint I/O
# =============================================================================

def save_mouthpiece(role: str, *, speaker: Optional[nn.Module] = None, bias_head: Optional[nn.Module] = None) -> str:
    out = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_mouthpiece.pt")
    state: Dict[str, Any] = {}
    if speaker is not None:
        try:
            state["speaker_bandit"] = speaker.state_dict()
        except Exception:
            print("[SAVE] SpeakerBandit state_dict failed; skipping.")
    if bias_head is not None:
        try:
            state["bias_head"] = bias_head.state_dict()
        except Exception:
            print("[SAVE] BiasHead state_dict failed; skipping.")
    torch.save(state, out)
    print(f"[SAVE] Mouthpiece saved → {out}")
    return out

def load_mouthpiece(role: str, *, speaker: Optional[nn.Module] = None, bias_head: Optional[nn.Module] = None) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_mouthpiece.pt")
    if not os.path.exists(path):
        print(f"[INIT] No mouthpiece checkpoint for {role}.")
        return speaker, bias_head
    state = torch.load(path, map_location="cpu")
    if speaker is not None and "speaker_bandit" in state:
        try:
            speaker.load_state_dict(state["speaker_bandit"])
            print(f"[LOAD] SpeakerBandit ← {path}")
        except Exception as e:
            print(f"[LOAD] SpeakerBandit load failed: {e}")
    if bias_head is not None and "bias_head" in state:
        try:
            bias_head.load_state_dict(state["bias_head"])
            print(f"[LOAD] BiasHead ← {path}")
        except Exception as e:
            print(f"[LOAD] BiasHead load failed: {e}")
    return speaker, bias_head

# =============================================================================
# (Optional) Language coupling helper (not wired by default to JEPA loops)
# =============================================================================

def language_coupling_loss(
    z_next: torch.Tensor,
    texts: List[str],
    *,
    msg_encoder: Optional[Any] = None,
    social: Optional[Any] = None,
    weight: float = LC_WEIGHT,
) -> torch.Tensor:
    """
    Small auxiliary that pulls z_next toward a SocialInfluence projection of the
    mean text embedding (encourages language↔state consistency).
    Not used by default; can be called from custom training scripts.
    """
    if not texts or msg_encoder is None or social is None or weight <= 0.0:
        return torch.tensor(0.0, device=z_next.device if torch.is_tensor(z_next) else DEVICE)
    try:
        with torch.no_grad():
            t_emb = msg_encoder(texts).mean(dim=0)  # [D_text]
        delta = social(t_emb.to(DEVICE))
        if z_next.dim() == 1:
            z_next = z_next.unsqueeze(0)
        target = delta.unsqueeze(0).expand_as(z_next)
        return weight * F.mse_loss(z_next, target)
    except Exception:
        return torch.tensor(0.0, device=DEVICE)

# =============================================================================
# I/O for checkpoints (uses CFG dims if present)
# =============================================================================

def load_role_models(role: str) -> Tuple[WorldModelMLP, ActionEncoder, PlannerHead]:
    """
    Legacy loader: returns (WM, ActionEncoder, PlannerHead) and loads {role}_jepa.pt if present.
    """
    wm = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_EMBED_DIM)
    ae = ActionEncoder(num_actions=NUM_AGENTS_CFG, action_dim=ACTION_EMBED_DIM)
    planner = PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS_CFG)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        wm.load_state_dict(state["world_model"])
        ae.load_state_dict(state["action_encoder"])
        planner.load_state_dict(state["planner"])
    else:
        print(f"[INIT] No checkpoint for {role}. Starting fresh.")

    return wm, ae, planner

def load_role_models_phase(role: str) -> Tuple[WorldModelMLP, PhaseActionEncoder, PlannerHead]:
    """
    Phase-aware loader: returns (WM, PhaseActionEncoder, PlannerHead) and loads {role}_jepa_phase.pt if present.
    """
    wm = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_EMBED_DIM)
    pae = PhaseActionEncoder(action_dim=ACTION_EMBED_DIM, num_agents=NUM_AGENTS_CFG, num_talk=NUM_TALK_CATS)
    planner = PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS_CFG)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa_phase.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} phase checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        wm.load_state_dict(state["world_model"])
        if "phase_action_encoder" in state:
            pae.load_state_dict(state["phase_action_encoder"])
        else:
            print(f"[WARN] Phase checkpoint missing 'phase_action_encoder'; starting it fresh.")
        if "planner" in state:
            planner.load_state_dict(state["planner"])
    else:
        print(f"[INIT] No phase checkpoint for {role}. Starting fresh.")

    return wm, pae, planner

def load_role_models_factorized(role: str) -> Tuple[WorldModelMLP, PhaseActionEncoder, FactorizedPlanner]:
    """
    Factorized loader: returns (WM, PhaseActionEncoder, FactorizedPlanner) and loads {role}_jepa_factorized.pt if present.
    """
    wm = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_EMBED_DIM)
    pae = PhaseActionEncoder(action_dim=ACTION_EMBED_DIM, num_agents=NUM_AGENTS_CFG, num_talk=NUM_TALK_CATS)
    fplanner = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS_CFG, num_talk_cats=NUM_TALK_CATS)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa_factorized.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} factorized checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if "world_model" in state:
            wm.load_state_dict(state["world_model"])
        else:
            print("[WARN] factorized ckpt missing 'world_model'; starting WM fresh.")
        if "phase_action_encoder" in state:
            pae.load_state_dict(state["phase_action_encoder"])
        else:
            print("[WARN] factorized ckpt missing 'phase_action_encoder'; starting PAE fresh.")
        if "factorized_planner" in state:
            fplanner.load_state_dict(state["factorized_planner"])
        else:
            print("[WARN] factorized ckpt missing 'factorized_planner'; starting planner fresh.")
    else:
        print(f"[INIT] No factorized checkpoint for {role}. Starting fresh.")

    return wm, pae, fplanner

# =============================================================================
# Sim runner shim (unchanged)
# =============================================================================

def run_sim_and_collect_rollouts(visual: bool = False):
    """Normalize sim output to (rollouts, meta) so train.py can use meta['agents'].""" 
    from sim import simulate_game
    ret = simulate_game(visual=visual)
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret
    return ret, {}
