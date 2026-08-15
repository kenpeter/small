"""
Pretraining with MegaTrain + Kimi K2 MuonClip
CPU offloaded training of 1.03B model with orthogonal updates.
Loads .bin shards directly (already tokenized with SmolLM2 tokenizer).
"""
import os, time, logging, argparse, math, shutil, random, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Cosine LR schedule with linear warmup
# ============================================================================

def get_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================================
# Kimi K2 MuonClip Optimizer — Full Implementation
# Based on arXiv 2502.16982 + github.com/AkulDatta/muonclip
# ============================================================================

def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration for matrix orthogonalization."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    # Normalize by Frobenius norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class KimiMuonClip(torch.optim.Optimizer):
    """
    Kimi K2 MuonClip optimizer + Kimi K3 Per-Head Muon for attention projections.
    - Muon (Newton-Schulz + momentum) for 2D hidden weights
    - Per-Head Muon for Q/K/V projections (Kimi K3 §2.5)
    - AdamW for 1D scalars (norms, biases)
    - AdamW for embeddings + lm_head
    - Consistent RMS scaling across all layers
    - Momentum warmup: 0.90 -> 0.95 over first 300 steps
    - QK-Clip proxy: spectral norm cap on attention projections
    """
    def __init__(self, param_groups, tau: float = 150.0, ns_steps: int = 7, use_gpu_ns: bool = True):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.01)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("per_head", False)
                group.setdefault("head_dim", 128)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        defaults = dict(tau=tau, ns_steps=ns_steps, use_gpu_ns=use_gpu_ns)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None, global_step: int = 0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Momentum warmup: 0.90 -> 0.95 over first 300 steps
        frac = min(global_step / 300.0, 1.0)
        warmed_momentum = (1 - frac) * 0.90 + frac * 0.95

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            if group["use_muon"]:
                # Use warmed momentum
                beta = warmed_momentum if group.get("warmup", True) else group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    # Momentum: Mt = μ * Mt-1 + (1-μ) * Gt  (EMA, not SGD-style)
                    buf.mul_(beta).add_(p.grad, alpha=1-beta)

                    # Newton-Schulz orthogonalization — GPU if available
                    use_gpu = torch.cuda.is_available() and self.defaults.get("use_gpu_ns", True)
                    if use_gpu:
                        buf_gpu = buf.cuda(non_blocking=False)
                        if group.get("per_head", False) and p.ndim == 2:
                            # Per-Head Muon (Kimi K3 §2.5): partition along head dimension
                            head_dim = group.get("head_dim", 128)
                            n_heads = p.shape[0] // head_dim
                            blocks = buf_gpu.chunk(n_heads, dim=0)
                            updates = [newton_schulz(b, steps=self.defaults["ns_steps"]) for b in blocks]
                            update = torch.cat(updates, dim=0).cpu()
                            del blocks, updates
                        elif p.ndim > 2:
                            orig_shape = buf_gpu.shape
                            buf_2d = buf_gpu.view(buf_gpu.shape[0], -1)
                            update_gpu = newton_schulz(buf_2d, steps=self.defaults["ns_steps"])
                            update = update_gpu.view(orig_shape).cpu()
                        else:
                            update = newton_schulz(buf_gpu, steps=self.defaults["ns_steps"]).cpu()
                        del buf_gpu
                    else:
                        if group.get("per_head", False) and p.ndim == 2:
                            # Per-Head Muon on CPU
                            head_dim = group.get("head_dim", 128)
                            n_heads = p.shape[0] // head_dim
                            blocks = buf.chunk(n_heads, dim=0)
                            updates = [newton_schulz(b, steps=self.defaults["ns_steps"]) for b in blocks]
                            update = torch.cat(updates, dim=0)
                        elif p.ndim > 2:
                            orig_shape = buf.shape
                            buf_2d = buf.view(buf.shape[0], -1)
                            update = newton_schulz(buf_2d, steps=self.defaults["ns_steps"])
                            update = update.view(orig_shape)
                        else:
                            update = newton_schulz(buf, steps=self.defaults["ns_steps"])

                    # Paper: Muon scaling = 0.2 * max(n, m)  (arXiv:2502.16982 §3)
                    n, m = p.shape[0], p.shape[1] if p.ndim > 1 else 1
                    rms_factor = max(n, m) * 0.2
                    update *= rms_factor

                    # Weight decay + update
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

            else:
                # AdamW for 1D params / embed / head
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        # QK-Clip proxy: cap spectral norm of attention-like projections — GPU
        tau = self.defaults["tau"]
        for group in self.param_groups:
            if group.get("use_muon", False):
                for p in group["params"]:
                    if p.ndim >= 2 and p.shape[0] <= p.shape[1]:
                        with torch.no_grad():
                            p_gpu = p.data.cuda(non_blocking=False)
                            if p_gpu.dtype != torch.float32:
                                p_gpu = p_gpu.float()
                            spec_norm = torch.linalg.matrix_norm(p_gpu, ord=2)
                            if spec_norm.item() > tau:
                                p_gpu.mul_(tau / spec_norm)
                                p.data.copy_(p_gpu)
                            del p_gpu

        return loss


SEQ_LEN = 2048

# ============================================================================
# Stratified multi-domain shard loader with 13-gram dedup
# ============================================================================

SHARD_DIRS = {
    # Tiered per domain: easy → medium → hard
    "math_easy":     Path("/home/kenpeter/work/data/_shards_math_easy"),     # 25.6B tokens
    "math_medium":   Path("/home/kenpeter/work/data/_shards_math_medium"),   # 144M tokens
    "synth_easy":    Path("/home/kenpeter/work/data/_shards_synth_easy"),    # 72M tokens
    "synth_medium":  Path("/home/kenpeter/work/data/_shards_synth_medium"),  # 757M tokens
    "synth_hard":    Path("/home/kenpeter/work/data/_shards_synth_hard"),    # 59M tokens
    "code_easy":     Path("/home/kenpeter/work/data/_shards_code_easy"),     # 75M tokens
    "code_medium":   Path("/home/kenpeter/work/data/_shards_code_medium"),   # 132M tokens
    "code_hard":     Path("/home/kenpeter/work/data/_shards_code_hard"),     # 5.9M tokens
    "web_easy":      Path("/home/kenpeter/work/data/_shards_web_easy"),       # fineweb-edu
    "web_medium":    Path("/home/kenpeter/work/data/_shards_web_medium"),    # 5.3G
    "web_hard":      Path("/home/kenpeter/work/data/_shards_web_hard"),      # empty
    "math_hard":     Path("/home/kenpeter/work/data/_shards_math_hard"),     # 5.1G
    "reformat_easy": Path("/home/kenpeter/work/data/_shards_reformat_easy"),  # was 8.5M tokens
    "gold_hard":     Path("/home/kenpeter/work/data/_shards_gold"),           # Qwen gold set (3 methods + 5 variants)
}

# x-small style flat farm: uniform random interleave of ALL tiered shards
# (1098-bin symlink farm over the SHARD_DIRS above, ~42B tokens)
FARM_DIR = Path("/home/kenpeter/work/data/_shards_final")

# ============================================================
# G1-G4 Curriculum (SAW-style data organization)
# G1 Boundary Sharpening: easy-heavy start → hard-heavy end (smooth)
# G2 Cyclic Scheduling: periodic easy-data review wave (anti-forgetting)
# G3 Curriculum Continuity: smooth tier blend — ratios shift gradually,
#    sampled every CURRICULUM_UPDATE_INTERVAL steps (no cliff switches)
# G4 Local Diversity: JIT windowed shuffle in _build_stratified_order
# ============================================================
CURRICULUM_UPDATE_INTERVAL = 2000  # steps between ratio rebuilds (rebuild ~35s)

# Tier → domain split fractions (within-tier proportions, sum=1 per tier)
# reformat_easy included so the 5-domain mix (math/web/code/synth/reformat) is
# preserved under G1-G4 curriculum (it's absent from all splits → excluded otherwise)
EASY_SPLIT = {"math_easy": 0.35, "web_easy": 0.35, "synth_easy": 0.125, "code_easy": 0.10, "reformat_easy": 0.075}
MED_SPLIT = {"math_medium": 0.25, "web_medium": 0.25, "synth_medium": 0.333, "code_medium": 0.167}
HARD_SPLIT = {"math_hard": 0.50, "web_hard": 0.15, "synth_hard": 0.20, "code_hard": 0.10, "gold_hard": 0.05}  # web_hard live: QuRatedPajama 594M tok; gold_hard ×5 via curriculum_boost.json

# Web boost (Aug 15): web lags every other domain (ref losses 2.96-3.26 vs
# 1.4-2.3 for the rest — measured from the step-59,100 checkpoint). Domains
# get their within-tier share multiplied by WEB_BOOST, then each tier is
# renormalized so G1-G4 tier totals stay intact (same principle as DoReMi).
#
# HOT-RELOAD (agent-in-the-loop DoReMi, Aug 15): if the file
# curriculum_boost.json exists next to this module, its per-domain multipliers
# OVERRIDE WEB_BOOST (missing domains keep WEB_BOOST). The file is re-read at
# every curriculum re-glide (2000 steps) — the agent edits the JSON and the
# next re-glide applies it. No restart. Format:
#   {"web_easy": 1.8, "web_medium": 1.8, "web_hard": 1.8}
WEB_BOOST = 1.5
WEB_BOOST_FILE = Path(__file__).parent / "curriculum_boost.json"

def _smooth_tier_weights(t):
    """G1+G3: 2-fold reversal (paper STR/SAW-2, arXiv:2605.30334).

    Fold 1 (t < 0.5): easy→hard — easy 0.30→0.175, hard 0.25→0.475.
    Fold 2 (t ≥ 0.5): MIRRORED hard→easy — the curve runs back, so the
    run starts AND ends easy-heavy with a hard peak mid-training. Order
    bias from fold 1 cancels out in fold 2.
    Continuous at the fold point (t=0.5 both sides give identical weights) → G3.
    """
    tt = t if t < 0.5 else 1.0 - t
    w_easy = 0.05 + 0.25 * (1 - tt)
    w_hard = min(0.70, 0.25 + 0.45 * tt)
    w_med = max(0.0, 1.0 - w_easy - w_hard)
    return w_easy, w_med, w_hard

def get_curriculum_ratios(step, total_steps):
    t = min(1.0, max(0.0, step / total_steps))
    w_easy, w_med, w_hard = _smooth_tier_weights(t)
    # G2: cyclic review wave — easy data gets a periodic boost
    # (full cosine cycle every 1/8 of training; amplitude 0.12)
    cycle = max(1, total_steps // 8)
    review = 0.12 * (0.5 - 0.5 * math.cos(2 * math.pi * step / cycle))
    w_easy = min(0.5, w_easy + review)
    # renormalize after G2 boost
    s = w_easy + w_med + w_hard
    w_easy, w_med, w_hard = w_easy / s, w_med / s, w_hard / s

    ratios = {}
    for dom, frac in EASY_SPLIT.items():
        ratios[dom] = round(w_easy * frac, 4)
    for dom, frac in MED_SPLIT.items():
        ratios[dom] = round(w_med * frac, 4)
    for dom, frac in HARD_SPLIT.items():
        ratios[dom] = round(w_hard * frac, 4)

    # Web boost: multiply web_* shares by WEB_BOOST, then renormalize per
    # tier so the G1-G4 tier totals (easy/medium/hard) stay exactly intact.
    # HOT-RELOAD: per-domain multipliers from curriculum_boost.json override
    # WEB_BOOST when present (re-read every call → applied at next re-glide).
    boost = {}
    try:
        if WEB_BOOST_FILE.exists():
            with open(WEB_BOOST_FILE) as f:
                boost = json.load(f)
    except Exception:
        boost = {}
    # capture pre-boost tier totals — renormalization must restore these
    tier_totals = {}
    for tier in ("_easy", "_medium", "_hard"):
        tier_totals[tier] = sum(v for k, v in ratios.items() if k.endswith(tier))
    for dom in ratios:
        # default: WEB_BOOST for web_* (the historical behavior), 1.0 for the
        # rest; curriculum_boost.json per-domain multipliers override either.
        dom_boost = WEB_BOOST if dom.startswith("web") else 1.0
        dom_boost = boost.get(dom, dom_boost)
        ratios[dom] = round(ratios[dom] * dom_boost, 4)
    if WEB_BOOST != 1.0 or boost:
        for tier in ("_easy", "_medium", "_hard"):
            doms = [d for d in ratios if d.endswith(tier)]
            s = sum(ratios[d] for d in doms)
            if s <= 0:
                continue
            for d in doms:
                ratios[d] = round(ratios[d] * tier_totals[tier] / s, 4)
    return {k: v for k, v in ratios.items() if v > 0}


def _load_shard_list(shards_dir: Path, seq_len: int):
    shard_paths = sorted(shards_dir.glob("*.bin"))
    shard_paths = [p for p in shard_paths if p.stat().st_size > 0]
    entries = []
    total = 0
    for p in shard_paths:
        n_tokens = p.stat().st_size // 2
        n_seqs = n_tokens // seq_len
        if n_seqs == 0:
            continue
        entries.append((p, n_seqs, total))
        total += n_seqs
    return entries, total


def _hash_13gram(tokens: np.ndarray) -> int:
    """Cheap 13-gram hash for dedup."""
    if len(tokens) < 13:
        return 0
    # Use first 13 tokens + last 13 tokens as proxy fingerprint
    front = tokens[:13].tobytes()
    back = tokens[-13:].tobytes()
    return hash((front, back))


class StratifiedShardDataset(Dataset):
    """
    Loads shards from multiple domains, applies stratified sampling ratios,
    and exact 13-gram deduplication across the whole corpus.
    """
    _causal_mask_4d = None

    def __init__(self, shard_dirs: dict, seq_len: int = 2048,
                 ratios: dict = None, dedup: bool = True):
        self.seq_len = seq_len
        self.ratios = ratios or RATIOS
        self.domains = []
        self.domain_entries = {}
        self.domain_totals = {}
        grand_total = 0

        for domain, dpath in shard_dirs.items():
            if not dpath.exists():
                logger.warning(f"Shard dir missing for '{domain}': {dpath} — skipping")
                continue
            entries, total = _load_shard_list(dpath, seq_len)
            if total == 0:
                continue
            self.domains.append(domain)
            self.domain_entries[domain] = entries
            self.domain_totals[domain] = total
            grand_total += total
            logger.info(f"Domain '{domain}': {len(entries)} shards, {total:,} seqs")

        if not self.domains:
            raise FileNotFoundError("No valid shard directories found")

        # Build flat index with domain tags for stratified sampling
        self.index = []          # (global_idx, domain, local_idx)
        self.domain_offsets = {}  # domain -> start in flat index
        cursor = 0
        for domain in self.domains:
            self.domain_offsets[domain] = cursor
            n = self.domain_totals[domain]
            self.index.extend([(cursor + i, domain, i) for i in range(n)])
            cursor += n
        self.raw_len = len(self.index)

        # Optional: exact 13-gram dedup (CPU, one-time scan)
        self.dedup = dedup
        self.valid_mask = None
        if dedup:
            self.valid_mask = self._compute_dedup_mask()
            kept = self.valid_mask.sum()
            logger.info(f"Dedup: {self.raw_len:,} raw → {kept:,} unique  (dropped {self.raw_len - kept:,})")
        else:
            self.valid_mask = torch.ones(self.raw_len, dtype=torch.bool)

        # Precompute stratified per-batch ordering
        self._build_stratified_order()

    def _compute_dedup_mask(self):
        seen = set()
        mask = torch.zeros(self.raw_len, dtype=torch.bool)
        # Scan every sequence once — slow but one-time
        for global_idx in range(self.raw_len):
            tokens = self._fetch_tokens(global_idx)
            h = _hash_13gram(tokens.numpy())
            if h not in seen:
                seen.add(h)
                mask[global_idx] = True
        return mask

    def _fetch_tokens(self, global_idx: int) -> torch.Tensor:
        _, domain, local_idx = self.index[global_idx]
        for shard_path, n_seqs, start_idx in self.domain_entries[domain]:
            if local_idx < start_idx + n_seqs:
                local = local_idx - start_idx
                offset = local * self.seq_len
                mm = np.memmap(str(shard_path), dtype=np.uint16, mode='r',
                               offset=offset * 2, shape=(self.seq_len,))
                tokens = torch.from_numpy(mm.copy().astype(np.int64))
                del mm
                return tokens
        raise IndexError(f"Bad index {global_idx}")

    def _build_stratified_order(self):
        # Create an epoch ordering that respects ratios
        valid_indices = torch.where(self.valid_mask)[0].tolist()
        # Only include domains that are in the current phase ratios
        active_domains = [d for d in self.domains if d in self.ratios]
        if not active_domains:
            active_domains = self.domains[:]
        # Bucket by domain
        buckets = {d: [] for d in active_domains}
        for idx in valid_indices:
            _, domain, _ = self.index[idx]
            if domain in buckets:
                buckets[domain].append(idx)
        # Shuffle each bucket
        for d in active_domains:
            random.shuffle(buckets[d])

        # Interleave according to ratios
        self.epoch_order = []
        ptrs = {d: 0 for d in active_domains}
        total_valid = len(valid_indices)
        # Emit proportionally to ratios: integer multiples of the smallest
        # active ratio so proportions are exact. Physical batch size is
        # handled by the DataLoader; this only shapes the interleave mix.
        # (Fixed: was max(1, int(2*ratio)) → floored every ratio to 1 and
        #  silently disabled ratio enforcement.)
        min_ratio = min(self.ratios[d] for d in active_domains)
        if min_ratio <= 0:
            # A domain ratio rounded to 0.0000 (e.g. after an extreme DoReMi
            # reweight) would make n_emit = ratio/0 → ZeroDivisionError.
            # Drop zero-weight domains instead of crashing (Aug 15 crash).
            active_domains = [d for d in active_domains if self.ratios[d] > 0]
            if not active_domains:
                raise RuntimeError("All domain ratios are zero — cannot build epoch order")
            min_ratio = min(self.ratios[d] for d in active_domains)
        # We just build a flat list; DataLoader batching will grab sequentially
        # To enforce ratios per step, we emit in repeating pattern
        while sum(ptrs[d] < len(buckets[d]) for d in active_domains) > 0:
            for domain in active_domains:
                # emit ~ratio proportion (exact integer multiples of min ratio)
                n_emit = max(1, round(self.ratios[domain] / min_ratio))
                for _ in range(n_emit):
                    if ptrs[domain] < len(buckets[domain]):
                        self.epoch_order.append(buckets[domain][ptrs[domain]])
                        ptrs[domain] += 1
            # safety break
            if len(self.epoch_order) > total_valid * 2:
                break
        # Trim to exact count
        self.epoch_order = self.epoch_order[:total_valid]
        # G4 (JIT - Jittering Ordering): shuffle within local windows to restore
        # gradient diversity while preserving the global tier trend (paper w=5000)
        jit_window = 5000
        for i in range(0, len(self.epoch_order), jit_window):
            chunk = self.epoch_order[i:i + jit_window]
            random.shuffle(chunk)
            self.epoch_order[i:i + jit_window] = chunk
        logger.info(f"Stratified epoch: {len(self.epoch_order):,} samples (JIT w={jit_window})")

    def __len__(self):
        return len(self.epoch_order)

    def __getitem__(self, idx):
        global_idx = self.epoch_order[idx]
        # Dict form carries the per-sample domain for DoReMi-lite per-domain
        # loss tracking (collate_pretrain branches on dict vs tensor).
        return {"input_ids": self._fetch_tokens(global_idx),
                "domain": self.index[global_idx][1]}


# Backwards compat alias
BinShardDataset = StratifiedShardDataset


class FlatFarmDataset(Dataset):
    """
    x-small style flat-farm loader: consumes the `_shards_final` symlink farm
    (uniform random interleave of all tiered shards) in sequential order with
    the same per-shard mechanics as x-small's BinShardDataset:
      - md5-hash stable split → tail n_val shards = val
      - per-shard hash offset start (random-ish, avoids always starting at 0)
      - paired reversal: even shards forward, odd shards reversed
    """
    _causal_mask_4d = None

    def __init__(self, farm_dir: Path, seq_len: int = 2048,
                 val_frac: float = 0.01, is_val: bool = False):
        import hashlib
        self.seq_len = seq_len
        shards = sorted(farm_dir.glob("*.bin"))
        if not shards:
            raise FileNotFoundError(f"No .bin shards found in farm {farm_dir}")
        sorted_shards = sorted(shards, key=lambda p: hashlib.md5(str(p).encode()).hexdigest())
        n_val = max(1, int(len(sorted_shards) * val_frac))
        self.shards = sorted_shards[-n_val:] if is_val else sorted_shards[:-n_val]

        # Build flat sequence index with x-small ordering semantics
        self.index = []  # (shard_idx, start)
        for si, sp in enumerate(self.shards):
            n_tokens = sp.stat().st_size // 2
            n = n_tokens - seq_len
            if n <= 0:
                continue
            offset = (hash(str(sp)) % n) if n > 0 else 0
            starts = list(range(offset, n, seq_len))
            if si % 2 == 1:
                starts.reverse()
            self.index.extend((si, s) for s in starts)
        logger.info(
            f"FlatFarm: {len(self.shards)} shards, {len(self.index):,} seqs "
            f"(val={is_val})"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, start = self.index[idx]
        sp = self.shards[si]
        mm = np.memmap(str(sp), dtype=np.uint16, mode="r")
        tokens = torch.from_numpy(mm[start:start + self.seq_len].astype(np.int64))
        del mm
        return tokens


def collate_pretrain(batch):
    # StratifiedShardDataset yields {"input_ids", "domain"} dicts (DoReMi-lite);
    # FlatFarmDataset yields raw tensors — both paths stay supported.
    if isinstance(batch[0], dict):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        domains = [b["domain"] for b in batch]
    else:
        input_ids = torch.stack(batch)
        domains = None
    B, T = input_ids.shape
    if FlatFarmDataset._causal_mask_4d is None or FlatFarmDataset._causal_mask_4d.shape[-1] != T:
        FlatFarmDataset._causal_mask_4d = torch.tril(torch.ones((1, 1, T, T), dtype=torch.bool))
    labels = input_ids.clone()
    out = {"input_ids": input_ids, "attention_mask": FlatFarmDataset._causal_mask_4d.expand(B, -1, -1, -1).contiguous(), "labels": labels}
    if domains is not None:
        out["domain"] = domains
    return out


def validate_cpu_params(model, logger):
    bad = 0
    bad_info = []
    params = model.get_parameters()
    for i, p in enumerate(params):
        if p is not None and not torch.isfinite(p).all():
            bad += 1
            mask = ~torch.isfinite(p)
            bad_info.append(f"  param[{i}] shape={tuple(p.shape)} dtype={p.dtype} nonfinite={mask.sum().item()}/{p.numel()} min={p.min().item():.3e} max={p.max().item():.3e}")
    if bad:
        logger.error(f"CRITICAL: {bad} CPU master parameters are non-finite after sync. Training would corrupt checkpoints.")
        for info in bad_info:
            logger.error(info)
        raise RuntimeError(f"NaN/Inf detected in {bad} CPU master params after optimizer step. Aborting to preserve clean state.")


def save_checkpoint_robust(state, output_dir, is_best, logger):
    model_sd = state.get("model_state_dict", {})
    bad_keys = []
    for k, v in model_sd.items():
        if not torch.isfinite(v).all():
            n_bad = (~torch.isfinite(v)).sum().item()
            n_total = v.numel()
            bad_keys.append(f"{k}: {n_bad}/{n_total} non-finite")
    if bad_keys:
        logger.warning(f"Checkpoint SAVE ABORTED — non-finite tensors detected ({len(bad_keys)}):")
        for msg in bad_keys[:5]:
            logger.warning(f"  {msg}")
        if len(bad_keys) > 5:
            logger.warning(f"  ... and {len(bad_keys) - 5} more")
        return False

    latest_path = os.path.join(output_dir, "megatrain_latest.pt")
    tmp_path = latest_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)
    logger.info(f"  Saved checkpoint to {latest_path}")

    if is_best:
        best_path = os.path.join(output_dir, "megatrain_best.pt")
        bak_path = best_path + ".bak"
        if os.path.exists(best_path):
            shutil.copy2(best_path, bak_path)
        torch.save(state, tmp_path)
        os.replace(tmp_path, best_path)
        logger.info(f"  Best loss {state['best_loss']:.4f} — saved to {best_path}")

    return True


def should_save_checkpoint(step_num, save_interval, last_save_time, now, save_every_minutes=0):
    """Decide whether a checkpoint save is due (step-based OR time-based).

    step_num is 1-indexed (the step that just completed). Returns True when:
      - step-based: step_num is a positive multiple of save_interval, or
      - time-based: save_every_minutes > 0 and now - last_save_time >= save_every_minutes*60.
    Time-based cadence keeps the worst-case loss window small (e.g. 20 min)
    regardless of step speed — a crash/pause/restart loses at most the time
    since the last save.
    """
    if save_interval and step_num > 0 and step_num % save_interval == 0:
        return True
    if save_every_minutes and save_every_minutes > 0 and (now - last_save_time) >= save_every_minutes * 60:
        return True
    return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--max-seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=12)
    parser.add_argument("--num-grad-slabs", type=int, default=12)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=120)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="/home/kenpeter/work/checkpoints")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Model dtype")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warmup steps")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KIMI K2 MUONCLIP — Fresh training from scratch")
    logger.info("=" * 60)
    logger.info(f"Model: Custom 1032M (dim=1536, L=32, h=12, kv=4, ffn=4608)")
    logger.info(f"Data: {FARM_DIR} (x-small style flat farm)")
    logger.info(f"Params: batch={args.batch_size}, seq_len={args.max_seq_len}, steps={args.num_steps}, dtype={args.dtype}")
    logger.info(f"LR={args.lr}, warmup={args.warmup_steps}, min_lr={args.min_lr}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

    logger.info("Creating model from custom config (random init)...")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    hf_config = LlamaConfig(
        vocab_size=49152,
        hidden_size=1536,
        intermediate_size=4608,
        num_hidden_layers=32,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        torch_dtype=args.dtype,
        head_dim=128,
        architectures=["LlamaForCausalLM"],
    )
    hf_model = AutoModelForCausalLM.from_config(
        hf_config,
        dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    n_params = sum(p.numel() for p in hf_model.parameters())
    logger.info(f"Model loaded: {n_params:,} parameters ({n_params/1e9:.2f}B)")

    # MegaTrain config
    config = CPUMasterConfig(
        model_name="custom-1B",
        dataset_path="/tmp/dummy",
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
        device=args.device,
        dtype=torch_dtype,
        log_interval=1,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model = CPUMasterModel(hf_model, config)
    del hf_model

    # === STANDARD ADAMW FOR ALL PARAMS ===
    params = model.get_parameters()
    vocab_embed_numel = 49152 * 1536  # embed_tokens + lm_head

    adamw_2d = [p for p in params if p.ndim >= 2 and p.numel() != vocab_embed_numel]
    embed_head_params = [p for p in params if p.ndim >= 2 and p.numel() == vocab_embed_numel]
    scalar_params = [p for p in params if p.ndim < 2]

    logger.info(f"AdamW | 2D: {len(adamw_2d)} params, Embed/Head: {len(embed_head_params)}, Scalars: {len(scalar_params)}")

    param_groups = [
        dict(params=adamw_2d, lr=args.lr, betas=(0.9, 0.95),
             eps=1e-8, weight_decay=config.weight_decay),
        dict(params=embed_head_params, lr=args.lr, betas=(0.9, 0.95),
             eps=1e-8, weight_decay=config.weight_decay),
        dict(params=scalar_params, lr=args.lr, betas=(0.9, 0.95),
             eps=1e-8, weight_decay=0.0),
    ]
    optimizer = torch.optim.AdamW(param_groups)
    logger.info("Optimizer: torch.optim.AdamW (all params)")

    # Dataset
    logger.info("Loading dataset...")
    dataset = None
    dataloader = None
    data_iter = None

    def rebuild_dataset(current_step):
        nonlocal dataset, dataloader, data_iter
        logger.info(f"Loading flat farm (x-small order): {FARM_DIR}")
        dataset = FlatFarmDataset(FARM_DIR, seq_len=args.max_seq_len, val_frac=0.01, is_val=False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_pretrain,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        data_iter = iter(dataloader)

    rebuild_dataset(0)

    # Training loop
    logger.info("=" * 60)
    best_loss = float("inf")
    global_step = 0
    start_step = 0

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        # CPUMasterModel doesn't have load_state_dict; manually map HF keys to layers
        sd = checkpoint["model_state_dict"]
        for key, tensor in sd.items():
            if key == "model.embed_tokens.weight":
                model.embedding.weight.data.copy_(tensor)
            elif key == "model.norm.weight":
                model.norm.weight.data.copy_(tensor)
            elif key == "lm_head.weight":
                model.lm_head.weight.data.copy_(tensor)
            elif key.startswith("model.layers."):
                parts = key.split(".")  # model.layers.N.layer_name...
                layer_idx = int(parts[2])
                sub_key = ".".join(parts[3:])
                target = model.cpu_layers[layer_idx]
                for name, param in target.named_parameters():
                    if name == sub_key:
                        param.data.copy_(tensor)
                        break
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Override LR with CLI value on resume (checkpoint restores old base_lr)
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = args.lr
            group.pop("base_lr", None)
        logger.info(f"  LR overridden: adam={args.lr}")
        global_step = checkpoint.get("step", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        start_step = global_step
        logger.info(f"  Resumed at step {global_step}, best_loss={best_loss:.4f}")
    else:
        logger.info("Starting pretraining from scratch...")

    logger.info("=" * 60)

    for step in range(start_step, config.num_steps):
        # Static flat-farm order (x-small style) — built once, no ratio rebuilds

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        t0 = time.perf_counter()

        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )

        if (step + 1) % config.gradient_accumulation_steps == 0:
            global_step += 1

            # Apply cosine LR schedule to each param group
            for group in optimizer.param_groups:
                base_lr = group["base_lr"] if "base_lr" in group else group["lr"]
                if "base_lr" not in group:
                    group["base_lr"] = base_lr  # store original once
                group["lr"] = get_lr(
                    step + 1,  # outer sample step, not global_step (optimizer step)
                    args.warmup_steps,
                    args.num_steps,
                    base_lr,
                    args.min_lr,
                )

            # Gradient clipping for all params
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], config.max_grad_norm)

            optimizer.step()

            model._sync_params_to_gpu()
            validate_cpu_params(model, logger)
            model.zero_grad()
            optimizer.zero_grad()

        step_time = time.perf_counter() - t0
        tps = config.batch_size * config.max_seq_len / step_time

        if (step + 1) % args.log_interval == 0:
            gpu_mem = torch.cuda.max_memory_allocated(args.device) / 1024**3
            current_adam_lr = next((g["lr"] for g in optimizer.param_groups), args.lr)
            logger.info(
                f"Step {step+1}/{config.num_steps} | "
                f"Loss {loss_val:.4f} | "
                f"LR {current_adam_lr:.2e} | "
                f"{step_time:.2f}s/step | "
                f"{tps:.0f} tok/s | "
                f"GPU {gpu_mem:.2f}GB"
            )

        # Save checkpoints
        if (step + 1) % args.save_interval == 0 or step == config.num_steps - 1:
            is_best = loss_val < best_loss
            if is_best:
                best_loss = loss_val

            full_sd = {}
            embed_sd = model.embedding.state_dict()
            for k, v in embed_sd.items():
                full_sd[f"model.embed_tokens.{k}"] = v
            for i, layer in enumerate(model.cpu_layers):
                for k, v in layer.state_dict().items():
                    full_sd[f"model.layers.{i}.{k}"] = v
            norm_sd = model.norm.state_dict()
            for k, v in norm_sd.items():
                full_sd[f"model.norm.{k}"] = v
            head_sd = model.lm_head.state_dict()
            for k, v in head_sd.items():
                full_sd[f"lm_head.{k}"] = v

            state = {
                "step": step + 1,
                "loss": loss_val,
                "best_loss": best_loss,
                "model_state_dict": full_sd,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint_robust(state, args.output_dir, is_best, logger)

    model.cleanup()
    logger.info("Pretraining complete!")


if __name__ == "__main__":
    main()
