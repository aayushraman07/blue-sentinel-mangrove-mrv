"""
================================================================================
 BLUE SENTINEL v9  "MANGROVE ECO-MONITOR — SPATIO-TEMPORAL MAMBA EDITION"
 Complete from-scratch implementation of the v9 engineering blueprint.

 REQUIRES: Python >= 3.10, PyTorch >= 2.1 (Linux or WSL2 recommended).
           Optional: mamba-ssm  (fast CUDA selective-scan kernels; a correct
                     pure-PyTorch chunked scan is used as automatic fallback),
                     rasterio    (GeoTIFF export; NPZ fallback otherwise).

--------------------------------------------------------------------------------
 BLUEPRINT -> IMPLEMENTATION MAP
--------------------------------------------------------------------------------
 §1  Tensor ingestion ......... scene loaders yield a (T, 19, H, W) cube;
                                channels 0-16 are network inputs, ch 17 (GEDI)
                                and ch 18 (seg label) are supervision targets.
 §2  Multi-modal stems ........ OpticalStem (ch 0-10), RadarStem (ch 11-14),
                                AuxStem (ch 15-16); concat + 1x1 fusion to D.
 §3  SSM core ................. discretized zero-order-hold selective scan
                                (dA = exp(dt*A), dB = dt*B), 4-directional
                                spatial cross-scan + fwd/bwd temporal sweeps
                                inside every VSS block (SS3D).
 §4  Hierarchical U-Net ....... 256/D -> 128/2D -> 64/4D -> 32/8D bottleneck,
                                patch-merging between stages, symmetric decoder
                                with skip concatenation back to 256 x 256.
 §5  Task heads ............... (1) seg: 1x1 conv + Symmetric Cross-Entropy
                                with ignore_index=255; (2) hydrology: softplus
                                (mu, sigma^2) + Gaussian NLL; (3) canopy:
                                Sparse Masked L1 over GEDI orbital tracks.
 §6  Curriculum ............... Phase 1 (ep 1-25)  : seg only, regression
                                                   heads frozen.
                                Phase 2 (ep 26-60): all heads, lambda_reg=0.1.
                                Phase 3 (ep 61+)  : joint training, cosine
                                                   annealing 1e-4 -> 1e-6.

 SCENE SCHEMA (folder of dated step NPZs, or a single cube NPZ):
   step NPZ keys:  s2 (H,W,10) surface reflectance          [required]
                   scl (H,W) scene classification           [optional]
                   s1 (H,W,2) VV,VH  (dB or linear)         [required]
                   lband (H,W,2) HH,HV (dB or linear)       [optional]
                   precip (H,W) or scalar, mm               [optional]
                   inundation (H,W) mm (tide-model prior)   [optional]
                   gedi (H,W) meters, NaN off-track         [optional]
                   depth (H,W) mm, observed water depth     [optional]
                   label (H,W) class ids, 255 = ignore      [last step]
                   date (scalar, decimal year)              [optional]
   cube NPZ keys:  cube (T,19,H,W) exactly per blueprint §1, plus optional
                   dates (T,). Labels/GEDI are read from the LAST timestep.

   Hydrology target: 'depth' if present, else the inundation prior channel
   itself (self-distillation / reconstruction mode). Canopy target: 'gedi'.

 MODES:  python blue_sentinel_v9.py smoke | train | eval | export
================================================================================
"""

import os
import re
import gc
import json
import glob
import math
import time
import shutil
import hashlib
import argparse
import platform
import warnings
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "9.0.0"
MIN_TORCH = (2, 1)

# ------------------------------------------------------------------------------
# Optional fast selective-scan kernel (mamba-ssm). A pure-PyTorch fallback is
# always available; the fallback is exact but much slower at stage-1 resolution
# (L = H*W), so gradient checkpointing is applied per scan chunk by default.
# ------------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except Exception:
    selective_scan_fn = None
    HAS_MAMBA = False


# ==============================================================================
# 1. ENVIRONMENT
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def environment_check():
    ok = True
    if _parse_version(torch.__version__) < MIN_TORCH:
        print(f"[FATAL] PyTorch {torch.__version__} < "
              f"{'.'.join(map(str, MIN_TORCH))}")
        ok = False
    if platform.system() == "Windows" and not torch.cuda.is_available() \
            and shutil.which("nvidia-smi"):
        print("=" * 78)
        print("[HARDWARE BLOCKER] Windows + NVIDIA driver present but no CUDA.")
        print("Use WSL2: wsl --install -d Ubuntu, then install torch+cu121.")
        print("=" * 78)
        ok = False
    return ok


ENV_OK = environment_check()
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v9/"
os.makedirs(WORK_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Blue Sentinel v{VERSION} | torch {torch.__version__} | "
      f"device={DEVICE} | bf16={AMP_BF16} | mamba_ssm={HAS_MAMBA}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
if not HAS_MAMBA:
    print("  [ssm] mamba-ssm not installed -> pure-PyTorch chunked scan "
          "(correct but slow at 256x256; pip install mamba-ssm for CUDA "
          "kernels).")


# ==============================================================================
# 2. CONFIG
# ==============================================================================
# Channel map of the 19-channel cube (blueprint §1)
CH_S2 = slice(0, 10)      # Sentinel-2 SR, 10 bands
CH_SCL = 10               # scene classification layer
CH_S1 = slice(11, 13)     # Sentinel-1 VV, VH (dB)
CH_LBAND = slice(13, 15)  # ALOS PALSAR-2 HH, HV (dB)
CH_PRECIP = 15            # CHIRPS 30-day precipitation (mm)
CH_INUND = 16             # tide-model inundation depth (mm)
CH_GEDI = 17              # GEDI canopy height (target only)
CH_LABEL = 18             # seg ground truth (target only)
N_INPUT_CH = 17           # channels 0..16 enter the network


@dataclass
class Config:
    # ---- data ----
    train_dir: str = "/kaggle/input/mangrove-v9/train/"
    val_dir: str = "/kaggle/input/mangrove-v9/val/"
    test_dir: str = "/kaggle/input/mangrove-v9/test/"
    patch_size: int = 256
    train_stride: int = 128
    max_train_patches: int = 8000
    max_val_patches: int = 1200
    min_valid_frac: float = 0.30
    max_cloud_frac: float = 0.85
    ignore_label: int = 255
    invalid_scl: tuple = (0, 1, 2, 3, 7, 8, 9, 10, 11)
    rare_classes: tuple = (2, 3)          # perm water / dense mangrove
    rare_boost: float = 4.0
    num_classes: int = 4                  # bare, tidal, perm water, dense m.
    temporal_steps: int = 5               # blueprint §1: T = 5
    # ---- model ----
    dim: int = 96                         # blueprint §4: D = 96
    depths: tuple = (2, 2, 2, 2)          # 2 VSS blocks per stage
    d_state: int = 16                     # SSM state dimension N
    expand: int = 2                       # VSS inner expansion
    dt_rank: str = "auto"                 # ceil(d_inner / 16)
    drop_path: float = 0.10
    ssm_chunk: int = 512                  # fallback scan chunk length
    grad_ckpt: bool = True                # checkpoint scan chunks (memory)
    # ---- training / curriculum (blueprint §6) ----
    epochs: int = 100
    phase1_end: int = 25                  # seg-only, regression heads frozen
    phase2_end: int = 60                  # regression lambda ramp-in
    phase2_lambda: float = 0.1
    lr: float = 1e-4
    final_lr: float = 1e-6
    weight_decay: float = 5e-2
    batch_size: int = 4
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    num_workers: int = 0
    seed: int = 42
    deterministic: bool = False
    # ---- losses ----
    sce_alpha: float = 1.0                # CE weight
    sce_beta: float = 0.1                 # RCE weight
    depth_scale: float = 1000.0           # mm -> m
    height_scale: float = 30.0            # canopy meters -> ~unit
    w_hydro: float = 1.0
    w_canopy: float = 1.0
    # ---- inference / export ----
    infer_batch: int = 4
    infer_tta: bool = True
    infer_overlap: float = 0.5
    pixel_size_m: float = 10.0
    shuffle_scenes: bool = True

    def check(self):
        assert self.patch_size % 32 == 0, "patch_size must be divisible by 32"
        assert self.temporal_steps >= 1
        assert self.epochs >= self.phase2_end >= self.phase1_end >= 1
        return True

    def save(self, path):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        for k in ("invalid_scl", "rare_classes", "depths"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)


CFG = Config()

CLASS_NAMES = ["Bare Land", "Tidal Water", "Permanent Water", "Dense Mangrove"]
CLASS_COLORS = ["#8B4513", "#00BFFF", "#191970", "#006400"]


# ==============================================================================
# 2b. DATASET VERSIONING
# ==============================================================================
def dataset_fingerprint(scene_descs):
    h = hashlib.sha256()
    n = 0
    for kind, path in sorted(scene_descs, key=lambda d: d[1]):
        files = sorted(glob.glob(os.path.join(path, "*.npz"))) \
            if kind == "folder" else [path]
        for f in files:
            h.update(os.path.basename(f).encode())
            try:
                h.update(str(os.path.getsize(f)).encode())
                n += 1
            except OSError:
                h.update(b"missing")
    return h.hexdigest()[:16], n


# ==============================================================================
# 3. SCENE LOADING  (folder of step NPZs  OR  single cube NPZ)
# ==============================================================================
class NPZSchemaError(ValueError):
    pass


def _to_db(x, lo, hi):
    x = x.astype(np.float32)
    if np.nanmin(x) >= 0:                      # linear power -> dB
        x = 10.0 * np.log10(np.clip(x, 1e-6, None))
    return np.clip(x, lo, hi)


def _process_s2(s2):
    s2 = s2.astype(np.float32)
    mx = np.nanmax(s2) if s2.size else 0.0
    if mx > 20000:
        s2 *= 1.0 / 65535.0
    elif mx > 100:
        s2 *= 1e-4
    return np.clip(s2, 0.0, 1.5)


def _sanitize(arr, name, path, max_bad=0.30):
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / arr.size
        if frac > max_bad:
            raise NPZSchemaError(f"{path}: '{name}' {frac:.0%} non-finite")
        arr = np.where(bad, 0.0, arr)
    return arr


def _date_from_name(path):
    m = re.search(r"((?:19|20)\d{2})[-_.]?(\d{2})?", os.path.basename(path))
    if not m:
        return 0.0
    y = int(m.group(1))
    mo = m.group(2)
    if mo and 1 <= int(mo) <= 12:
        return y + (int(mo) - 1) / 12.0
    return float(y)


def _as_hw(x, H, W, fill=0.0):
    """Accept (H,W), scalar, or missing -> (H,W) float32."""
    if x is None:
        return np.full((H, W), fill, np.float32)
    x = np.asarray(x, np.float32)
    if x.ndim == 0:
        return np.full((H, W), float(x), np.float32)
    if x.shape != (H, W):
        raise NPZSchemaError(f"spatial mismatch {x.shape} vs {(H, W)}")
    return x


def load_step_npz(path, cfg):
    """One timestep -> dict(x17, label, gedi, depth, date)."""
    with np.load(path) as d:
        files = set(d.files)
        if "s2" not in files or "s1" not in files:
            raise NPZSchemaError(f"{path}: missing required 's2'/'s1'")
        s2 = _sanitize(_process_s2(d["s2"]), "s2", path)
        if s2.ndim != 3 or s2.shape[-1] != 10:
            raise NPZSchemaError(f"{path}: bad s2 shape {s2.shape}")
        H, W = s2.shape[:2]
        s1 = _sanitize(_to_db(d["s1"], -35.0, 5.0), "s1", path)
        if s1.shape != (H, W, 2):
            raise NPZSchemaError(f"{path}: bad s1 shape {s1.shape}")
        scl = _as_hw(d["scl"] if "scl" in files else None, H, W, fill=4.0)
        lband = (_sanitize(_to_db(d["lband"], -40.0, 5.0), "lband", path)
                 if "lband" in files else np.zeros((H, W, 2), np.float32))
        precip = _sanitize(_as_hw(d.get("precip"), H, W), "precip", path)
        precip = np.clip(precip, 0.0, 2000.0)
        inund = _sanitize(_as_hw(d.get("inundation"), H, W), "inundation", path)
        inund = np.clip(inund, 0.0, 20000.0)
        # targets (NaN preserved; NOT sanitized)
        gedi = (_as_hw(d["gedi"], H, W, fill=np.nan) if "gedi" in files
                else np.full((H, W), np.nan, np.float32))
        depth = (_as_hw(d["depth"], H, W, fill=np.nan) if "depth" in files
                 else None)
        label = (d["label"].astype(np.int64) if "label" in files
                 else np.full((H, W), cfg.ignore_label, np.int64))
        date = float(d["date"]) if "date" in files else _date_from_name(path)

    x17 = np.concatenate([
        s2,                                   # 0-9
        scl[..., None].astype(np.float32),    # 10
        s1,                                   # 11-12
        lband,                                # 13-14
        precip[..., None],                    # 15
        inund[..., None],                     # 16
    ], axis=-1).astype(np.float32)            # (H, W, 17)
    x17 = np.transpose(x17, (2, 0, 1))        # -> (17, H, W): channels-first,
                                               # matching _load_cube_npz and
                                               # what the model + patch_generator
                                               # actually assume everywhere else.
    assert x17.shape[0] == N_INPUT_CH
    return {"x": x17, "label": label, "gedi": gedi, "depth": depth,
            "date": np.float32(date)}


def _assemble_scene(steps, cfg):
    """steps: list of step dicts -> scene dict with (T,17,H,W) inputs and
    last-step targets. Sub-samples evenly or repeats the last step to hit T."""
    T = cfg.temporal_steps
    steps = sorted(steps, key=lambda s: s["date"])
    if len(steps) > T:
        idx = np.unique(np.linspace(0, len(steps) - 1, T).round().astype(int))
        steps = [steps[int(i)] for i in idx]
    while len(steps) < T:
        steps.append(steps[-1])
    last = steps[-1]
    depth = last["depth"]
    if depth is None:
        depth = last["x"][CH_INUND].copy()     # pseudo-label fallback
                                                 # (channels-first: index axis 0,
                                                 # not [..., CH_INUND] which was
                                                 # left over from the old
                                                 # channels-last x17 layout)
    return {
        "x": np.stack([s["x"] for s in steps], 0),          # (T,17,H,W)
        "label": last["label"],
        "gedi": last["gedi"].astype(np.float32),
        "depth": depth.astype(np.float32),
        "dates": np.array([s["date"] for s in steps], np.float32),
    }


def _load_cube_npz(path, cfg):
    with np.load(path) as d:
        files = set(d.files)
        if "cube" not in files:
            raise NPZSchemaError(f"{path}: missing 'cube'")
        cube = d["cube"].astype(np.float32)
        if cube.ndim != 4 or cube.shape[1] != 19:
            raise NPZSchemaError(f"{path}: cube must be (T,19,H,W), "
                                 f"got {cube.shape}")
        dates = (d["dates"].astype(np.float32) if "dates" in files
                 else np.arange(cube.shape[0], dtype=np.float32))
    x = cube[:, :N_INPUT_CH]
    for c in range(N_INPUT_CH):
        x[:, c] = _sanitize(x[:, c], f"cube[:,{c}]", path)
    scene = {
        "x": x,
        "label": np.where(np.isfinite(cube[-1, CH_LABEL]),
                          cube[-1, CH_LABEL], cfg.ignore_label).astype(np.int64),
        "gedi": cube[-1, CH_GEDI].astype(np.float32),
        "depth": cube[-1, CH_INUND].astype(np.float32).copy(),
        "dates": dates,
    }
    T = cfg.temporal_steps
    while scene["x"].shape[0] < T:
        scene["x"] = np.concatenate([scene["x"], scene["x"][-1:]], 0)
    if scene["x"].shape[0] > T:
        idx = np.unique(np.linspace(0, scene["x"].shape[0] - 1, T)
                        .round().astype(int))
        scene["x"] = scene["x"][idx]
    return scene


def discover_scenes(root):
    if not root or not os.path.isdir(root):
        return []
    scenes = [("folder", p) for p in sorted(glob.glob(os.path.join(root, "*")))
              if os.path.isdir(p) and glob.glob(os.path.join(p, "*.npz"))]
    scenes += [("npz", p) for p in sorted(glob.glob(os.path.join(root, "*.npz")))]
    return scenes


def load_scene(desc, cfg):
    kind, path = desc
    if kind == "npz":
        return _load_cube_npz(path, cfg)
    steps, n_fail = [], 0
    for p in sorted(glob.glob(os.path.join(path, "*.npz"))):
        try:
            steps.append(load_step_npz(p, cfg))
        except Exception as e:
            n_fail += 1
            print(f"  [data] bad step {os.path.basename(p)}: {str(e)[:100]}")
    if not steps:
        raise NPZSchemaError(f"{path}: no valid steps ({n_fail} failed)")
    return _assemble_scene(steps, cfg)


# ==============================================================================
# 4. PATCH GENERATOR (class-balanced, numpy geo-aug, per-scene fault isolation)
# ==============================================================================
def _patch_weight(label, cfg):
    valid = label != cfg.ignore_label
    if valid.sum() == 0:
        return 0.0
    counts = np.bincount(label[valid].ravel(),
                         minlength=cfg.num_classes).astype(np.float32)
    frac = counts / max(counts.sum(), 1)
    return 1.0 + cfg.rare_boost * float(frac[list(cfg.rare_classes)].sum())


def _np_geometric_aug(p, rng):
    p = dict(p)
    lr = rng.random() < 0.5
    ud = rng.random() < 0.5
    k = int(rng.integers(0, 4))

    def t(v):
        ax = (-2, -1)                                  # last two dims = H, W
        if lr:
            v = np.flip(v, axis=ax[1])
        if ud:
            v = np.flip(v, axis=ax[0])
        if k:
            v = np.rot90(v, k, axes=ax)
        return np.ascontiguousarray(v)

    for key in ("x", "label", "gedi", "depth"):
        p[key] = t(p[key])
    return p


def _np_photometric_aug(p, rng):
    p = dict(p)
    x = p["x"].copy()
    s2 = x[:, CH_S2]
    delta = rng.uniform(-0.10, 0.10)
    contra = rng.uniform(0.85, 1.15)
    mean = s2.mean(axis=(1, 2), keepdims=True)
    x[:, CH_S2] = np.clip((s2 - mean) * contra + mean + delta, -1.0, 2.0)
    x[:, CH_S1] += rng.normal(0, 0.3, x[:, CH_S1].shape).astype(np.float32)
    x[:, CH_LBAND] += rng.normal(0, 0.3, x[:, CH_LBAND].shape).astype(np.float32)
    p["x"] = x
    return p


def patch_generator(scene_descs, cfg, training, max_patches, seed=None):
    ps = cfg.patch_size
    stride = cfg.train_stride if training else ps
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    descs = list(scene_descs)
    if training and cfg.shuffle_scenes:
        rng.shuffle(descs)
    invalid = np.array(list(cfg.invalid_scl))
    quota = max(1, max_patches // max(1, len(descs))) if training else max_patches
    emitted = failed = 0
    for desc in descs:
        if emitted >= max_patches:
            break
        try:
            scene = load_scene(desc, cfg)
        except Exception as e:
            failed += 1
            print(f"  [data] SKIP {desc[1]}: {str(e)[:140]}")
            if failed >= 25:
                raise RuntimeError("Too many scene failures; check data root.")
            continue
        label = scene["label"]
        h, w = label.shape
        if h < ps or w < ps:
            continue
        coords = []
        for i in range(0, h - ps + 1, stride):
            for j in range(0, w - ps + 1, stride):
                lp = label[i:i + ps, j:j + ps]
                if float(np.mean(lp != cfg.ignore_label)) < cfg.min_valid_frac:
                    continue
                sc = scene["x"][-1, CH_SCL, i:i + ps, j:j + ps]
                if float(np.mean(np.isin(sc, invalid))) > cfg.max_cloud_frac:
                    continue
                wgt = _patch_weight(lp, cfg) if training else 1.0
                if wgt > 0:
                    coords.append((i, j, wgt))
        if training and len(coords) > quota:
            wts = np.array([c[2] for c in coords], np.float64)
            wts /= wts.sum()
            sel = rng.choice(len(coords), size=quota, replace=False, p=wts)
            coords = [coords[int(k)] for k in sel]
        for (i, j, _w) in coords:
            if emitted >= max_patches:
                break
            sl = np.s_[i:i + ps, j:j + ps]
            patch = {
                "x": np.ascontiguousarray(scene["x"][:, :, sl[0], sl[1]]),
                "label": np.ascontiguousarray(label[sl]),
                "gedi": np.ascontiguousarray(scene["gedi"][sl]),
                "depth": np.ascontiguousarray(scene["depth"][sl]),
                "dates": scene["dates"],
            }
            if training:
                patch = _np_geometric_aug(patch, rng)
                patch = _np_photometric_aug(patch, rng)
            yield patch
            emitted += 1
        del scene
        gc.collect()
    if emitted == 0:
        raise RuntimeError(
            "patch_generator emitted 0 patches. Check scene dirs, patch_size, "
            "min_valid_frac / max_cloud_frac, and that labels exist.")


class PatchStream(IterableDataset):
    def __init__(self, scene_descs, cfg, training, max_patches, epoch_seed=0):
        super().__init__()
        self.scene_descs = scene_descs
        self.cfg = cfg
        self.training = training
        self.max_patches = max_patches
        self.epoch_seed = epoch_seed

    def set_epoch(self, ep):
        self.epoch_seed = ep

    def __iter__(self):
        info = get_worker_info()
        seed = self.cfg.seed + 1000 * self.epoch_seed + \
            (info.id if info else 0)
        for idx, patch in enumerate(patch_generator(
                self.scene_descs, self.cfg, self.training,
                self.max_patches, seed=seed)):
            if info is not None and idx % info.num_workers != info.id:
                continue
            yield {"x": torch.from_numpy(patch["x"]),
                   "label": torch.from_numpy(patch["label"]),
                   "gedi": torch.from_numpy(patch["gedi"]),
                   "depth": torch.from_numpy(patch["depth"])}


def make_loader(scene_descs, cfg, training, max_patches):
    ds = PatchStream(scene_descs, cfg, training, max_patches)
    return DataLoader(ds, batch_size=cfg.batch_size,
                      num_workers=cfg.num_workers,
                      pin_memory=(DEVICE.type == "cuda"),
                      drop_last=training), ds


# ==============================================================================
# 5. NORMALIZATION STATS (channels 0-16; SCL channel forced to /11 scaling)
# ==============================================================================
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v9.npz")


def compute_stats(scene_descs, cfg, max_patches=300):
    print("Computing per-channel normalization stats...")
    n = 0
    s_sum = np.zeros(N_INPUT_CH)
    s_sq = np.zeros(N_INPUT_CH)
    for patch in patch_generator(scene_descs, cfg, training=False,
                                 max_patches=max_patches):
        x = patch["x"].reshape(-1, N_INPUT_CH)          # (T*H*W, 17)
        s_sum += x.sum(0)
        s_sq += (x ** 2).sum(0)
        n += x.shape[0]
    if n == 0:
        raise RuntimeError("compute_stats: no patches collected.")
    mean = s_sum / n
    var = np.clip(s_sq / n - mean ** 2, 1e-6, None)
    mean[CH_SCL] = 0.0                                  # scl / 11 scaling
    var[CH_SCL] = 121.0
    np.savez(STATS_PATH, mean=mean.astype(np.float32),
             var=var.astype(np.float32))
    print(f"  mean: {np.round(mean, 3).tolist()}")
    return mean.astype(np.float32), var.astype(np.float32)


def load_or_compute_stats(scene_descs, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        print("Loaded cached norm stats.")
        return z["mean"], z["var"]
    return compute_stats(scene_descs, cfg)


# ==============================================================================
# 6. SELECTIVE SCAN  (blueprint §3-A: zero-order-hold discretization)
# ==============================================================================
def _scan_chunk(u_c, dt_c, h0, A, B_c, C_c):
    """Sequential selective scan over one chunk.
    u_c, dt_c: (B, l, D);  h0: (B, D, N);  A: (D, N);
    B_c, C_c: (B, l, N).  Returns (y_c (B,l,D), h_last (B,D,N))."""
    h = h0
    ys = []
    for t in range(u_c.shape[1]):
        dA = torch.exp(dt_c[:, t].unsqueeze(-1) * A)              # (B,D,N)
        dB = dt_c[:, t].unsqueeze(-1) * B_c[:, t].unsqueeze(1)    # (B,D,N)
        h = dA * h + dB * u_c[:, t].unsqueeze(-1)
        ys.append(torch.einsum("bdn,bn->bd", h, C_c[:, t]))
    return torch.stack(ys, dim=1), h


def selective_scan_torch(u, dt, A, B, C, D, chunk=512, use_ckpt=True):
    """Reference pure-PyTorch selective scan (fp32).
    u, dt: (B, L, D); A: (D, N); B, C: (B, L, N); D: (D,)."""
    Bs, L, Dd = u.shape
    N = A.shape[-1]
    u32, dt32 = u.float(), dt.float()
    B32, C32 = B.float(), C.float()
    h = u32.new_zeros(Bs, Dd, N)
    outs = []
    ckpt = use_ckpt and torch.is_grad_enabled()
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        args = (u32[:, s:e], dt32[:, s:e], h, A, B32[:, s:e], C32[:, s:e])
        if ckpt:
            y_c, h = torch.utils.checkpoint.checkpoint(
                _scan_chunk, *args, use_reentrant=False)
        else:
            y_c, h = _scan_chunk(*args)
        outs.append(y_c)
    y = torch.cat(outs, dim=1)
    y = y + u32 * D.float()
    return y.to(u.dtype)


class SSMCore(nn.Module):
    """Parameter holder + dispatch for one set of directional scans.
    dt_rank / A / D / projections are per-direction, as in VMamba."""

    def __init__(self, d_inner, n_dirs, d_state=16, dt_rank="auto",
                 dt_min=1e-3, dt_max=0.1, chunk=512, grad_ckpt=True):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_dirs = n_dirs
        self.chunk = chunk
        self.grad_ckpt = grad_ckpt
        self.dt_rank = int(math.ceil(d_inner / 16)) \
            if dt_rank == "auto" else int(dt_rank)

        self.x_projs = nn.ModuleList([
            nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(n_dirs)])
        self.dt_projs = nn.ModuleList([
            self._init_dt_proj(dt_min, dt_max) for _ in range(n_dirs)])

        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
            .repeat(n_dirs, d_inner, 1)                          # (dirs,D,N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(n_dirs, d_inner))

    def _init_dt_proj(self, dt_min, dt_max):
        proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        k = 1.0 / max(self.dt_rank, 1)
        nn.init.uniform_(proj.weight, -k, k)
        dt = torch.exp(torch.rand(self.d_inner)
                       * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp_min(1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))               # softplus^-1
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        return proj

    def scan(self, seq, d):
        """seq: (B, L, d_inner) for direction index d -> (B, L, d_inner)."""
        dtbc = self.x_projs[d](seq)
        dt, Bm, Cm = torch.split(
            dtbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log[d].float())
        Dskip = self.D[d].float()

        if HAS_MAMBA and seq.is_cuda:
            delta = self.dt_projs[d](dt)                          # (B,L,D)
            y = selective_scan_fn(
                seq.transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                A,
                Bm.transpose(1, 2).contiguous(),
                Cm.transpose(1, 2).contiguous(),
                Dskip, z=None,
                delta_bias=self.dt_projs[d].bias.float(),
                delta_softplus=True)
            return y.transpose(1, 2)

        dtf = F.softplus(self.dt_projs[d](dt.float()))
        return selective_scan_torch(seq, dtf, A, Bm, Cm, Dskip,
                                    chunk=self.chunk,
                                    use_ckpt=self.grad_ckpt)


# ==============================================================================
# 7. SS3D LAYER + VSS BLOCK  (blueprint §3-B: 4 spatial + 2 temporal sweeps)
# ==============================================================================
class DropPath(nn.Module):
    def __init__(self, rate=0.0):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x):
        if self.rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device, dtype=torch.float32)
                < keep).to(x.dtype)
        return x * mask / keep


class SpatioTemporalSSM(nn.Module):
    """LN -> in-proj -> depthwise 3x3 conv -> SiLU -> 4-directional spatial
    cross-scan (+ forward/backward temporal sweeps) -> gated out-proj.
    Operates on (B, T, C, H, W); temporal sweeps are skipped when
    temporal=False (decoder, T=1)."""

    def __init__(self, dim, temporal=True, expand=2, d_state=16,
                 dt_rank="auto", chunk=512, grad_ckpt=True):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expand)
        self.temporal = temporal
        n_dirs = 6 if temporal else 4
        self.ln = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner)
        self.dw = nn.Conv2d(self.d_inner, self.d_inner, 3, 1, 1,
                            groups=self.d_inner)
        self.core = SSMCore(self.d_inner, n_dirs, d_state=d_state,
                            dt_rank=dt_rank, chunk=chunk,
                            grad_ckpt=grad_ckpt)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        di = self.d_inner
        x = x.permute(0, 1, 3, 4, 2)                    # (B,T,H,W,C)
        x = self.ln(x)
        xi, z = self.in_proj(x).chunk(2, dim=-1)        # each (B,T,H,W,di)
        xi = xi.permute(0, 1, 4, 2, 3).reshape(B * T, di, H, W)
        xi = F.silu(self.dw(xi))
        xi = xi.view(B, T, di, H, W)

        # --- 4-directional spatial cross-scan (per timestep) ---
        s1 = xi.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, di)   # TL->BR
        s2 = s1.flip(1)                                            # BR->TL
        s3 = xi.permute(0, 1, 4, 3, 2).reshape(B * T, H * W, di)   # TR->BL
        s4 = s3.flip(1)                                            # BL->TR
        y1 = self.core.scan(s1, 0).view(B, T, H, W, di) \
            .permute(0, 1, 4, 2, 3)
        y2 = self.core.scan(s2, 1).flip(1).view(B, T, H, W, di) \
            .permute(0, 1, 4, 2, 3)
        y3 = self.core.scan(s3, 2).view(B, T, W, H, di) \
            .permute(0, 1, 4, 3, 2)
        y4 = self.core.scan(s4, 3).flip(1).view(B, T, W, H, di) \
            .permute(0, 1, 4, 3, 2)
        y = y1 + y2 + y3 + y4

        # --- forward/backward temporal sweeps (per pixel location) ---
        if self.temporal and T > 1:
            s5 = xi.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, di)
            s6 = s5.flip(1)
            y5 = self.core.scan(s5, 4).view(B, H, W, T, di) \
                .permute(0, 3, 4, 1, 2)
            y6 = self.core.scan(s6, 5).flip(1).view(B, H, W, T, di) \
                .permute(0, 3, 4, 1, 2)
            y = y + y5 + y6

        y = y.permute(0, 1, 3, 4, 2)                    # (B,T,H,W,di)
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)                            # (B,T,H,W,C)
        return y.permute(0, 1, 4, 2, 3)                 # (B,T,C,H,W)


class VSSBlock(nn.Module):
    def __init__(self, dim, temporal=True, drop_path=0.0, **ssm_kw):
        super().__init__()
        self.ssm = SpatioTemporalSSM(dim, temporal=temporal, **ssm_kw)
        self.dp = DropPath(drop_path)
        self.use_ckpt = ssm_kw.get("grad_ckpt", True)

    def _forward(self, x):
        return x + self.dp(self.ssm(x))

    def forward(self, x):
        if self.use_ckpt and self.training and torch.is_grad_enabled() \
                and not (HAS_MAMBA and x.is_cuda):
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False)
        return self._forward(x)


# ==============================================================================
# 8. STEMS / PATCH MERGING / DECODER  (blueprint §2, §4)
# ==============================================================================
def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class Stem(nn.Sequential):
    """3x3 conv -> BatchNorm -> GELU, projecting a modality group to D."""

    def __init__(self, cin, dim):
        super().__init__(
            nn.Conv2d(cin, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.GELU())


class PatchMerging2D(nn.Module):
    """Swin/VMamba-style 2x2 patch merging: 4C -> LayerNorm -> Linear(2C)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.norm = nn.LayerNorm(4 * cin)
        self.reduction = nn.Linear(4 * cin, cout, bias=False)

    def forward(self, x):                               # (B,T,C,H,W)
        x0 = x[:, :, :, 0::2, 0::2]
        x1 = x[:, :, :, 1::2, 0::2]
        x2 = x[:, :, :, 0::2, 1::2]
        x3 = x[:, :, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=2)          # (B,T,4C,H/2,W/2)
        x = x.permute(0, 1, 3, 4, 2)
        x = self.reduction(self.norm(x))
        return x.permute(0, 1, 4, 2, 3)


class DecoderStage(nn.Module):
    """Transpose-conv upsample -> concat skip -> fuse -> VSS block -> refine."""

    def __init__(self, cin, cskip, cout, cfg):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(cout + cskip, cout, 1), _gn(cout), nn.GELU())
        self.vss = VSSBlock(cout, temporal=False, drop_path=0.0,
                            expand=cfg.expand, d_state=cfg.d_state,
                            dt_rank=cfg.dt_rank, chunk=cfg.ssm_chunk,
                            grad_ckpt=cfg.grad_ckpt)
        self.refine = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, 1), _gn(cout), nn.GELU())

    def forward(self, x, skip):                         # 2D: (B,C,H,W)
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.vss(x.unsqueeze(1)).squeeze(1)         # T=1: spatial scans only
        return self.refine(x)


# ==============================================================================
# 9. BLUE SENTINEL v9 MODEL
# ==============================================================================
class BlueSentinelV9(nn.Module):
    def __init__(self, cfg, norm_mean, norm_var):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim
        dims = [D, 2 * D, 4 * D, 8 * D]                 # 96,192,384,768
        self.register_buffer("in_mean",
                             torch.tensor(norm_mean).view(1, 1, -1, 1, 1))
        self.register_buffer("in_std",
                             torch.tensor(np.sqrt(norm_var)).view(1, 1, -1, 1, 1))

        # ---- blueprint §2: three isolated stems ----
        self.opt_stem = Stem(11, D)                     # ch 0-10  (S2 + SCL)
        self.radar_stem = Stem(4, D)                    # ch 11-14 (C- & L-band)
        self.aux_stem = Stem(2, D)                      # ch 15-16 (precip,inund)
        self.fuse = nn.Conv2d(3 * D, D, 1)

        # ---- blueprint §4: hierarchical encoder ----
        ssm_kw = dict(expand=cfg.expand, d_state=cfg.d_state,
                      dt_rank=cfg.dt_rank, chunk=cfg.ssm_chunk,
                      grad_ckpt=cfg.grad_ckpt)
        dpr = np.linspace(0.0, cfg.drop_path, sum(cfg.depths)).tolist()
        self.stages = nn.ModuleList()
        k = 0
        for i, dep in enumerate(cfg.depths):
            blocks = []
            for _ in range(dep):
                blocks.append(VSSBlock(dims[i], temporal=True,
                                       drop_path=dpr[k], **ssm_kw))
                k += 1
            self.stages.append(nn.ModuleList(blocks))
        self.merges = nn.ModuleList([
            PatchMerging2D(dims[i], dims[i + 1]) for i in range(3)])

        # ---- symmetric decoder (2D, last-timestep features) ----
        self.dec2 = DecoderStage(dims[3], dims[2], dims[2], cfg)   # 32->64
        self.dec1 = DecoderStage(dims[2], dims[1], dims[1], cfg)   # 64->128
        self.dec0 = DecoderStage(dims[1], dims[0], dims[0], cfg)   # 128->256

        # ---- blueprint §5: task heads ----
        trunk = min(64, D)
        self.head_trunk = nn.Sequential(
            nn.Conv2d(D, trunk, 3, 1, 1), _gn(trunk), nn.GELU())
        self.seg_head = nn.Conv2d(trunk, cfg.num_classes, 1)
        self.hydro_head = nn.Conv2d(trunk, 2, 1)        # (mu_raw, var_raw)
        self.canopy_head = nn.Conv2d(trunk, 1, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, T, 17, H, W) -> dict of head outputs (float32 heads)."""
        assert x.dim() == 5 and x.shape[2] == N_INPUT_CH, \
            f"expected (B,T,{N_INPUT_CH},H,W), got {tuple(x.shape)}"
        B, T, _, H, W = x.shape
        x = (x - self.in_mean) / self.in_std

        f = x.reshape(B * T, N_INPUT_CH, H, W)
        o = self.opt_stem(f[:, 0:11])
        r = self.radar_stem(f[:, 11:15])
        a = self.aux_stem(f[:, 15:17])
        z = self.fuse(torch.cat([o, r, a], dim=1))      # (B*T, D, H, W)
        z = z.view(B, T, self.cfg.dim, H, W)

        skips = []
        for i, stage in enumerate(self.stages):
            for blk in stage:
                z = blk(z)
            skips.append(z)
            if i < len(self.merges):
                z = self.merges[i](z)

        # reference-frame decoder: last timestep has absorbed full temporal
        # context via the per-block temporal sweeps.
        d = self.dec2(z[:, -1], skips[2][:, -1])
        d = self.dec1(d, skips[1][:, -1])
        d = self.dec0(d, skips[0][:, -1])

        h = self.head_trunk(d)
        seg = self.seg_head(h)
        hydro = self.hydro_head(h).float()
        mu = F.softplus(hydro[:, 0:1])                  # non-negative depth
        var = F.softplus(hydro[:, 1:2]) + 1e-3          # strictly positive
        canopy = self.canopy_head(h).float()
        return {"seg": seg.float(), "hydro_mu": mu, "hydro_var": var,
                "canopy": canopy}

    def set_phase(self, phase):
        """Blueprint §6 phase 1: freeze hydrology + canopy heads."""
        trainable = phase > 1
        for head in (self.hydro_head, self.canopy_head):
            for p in head.parameters():
                p.requires_grad_(trainable)


# ==============================================================================
# 10. LOSSES  (blueprint §5)
# ==============================================================================
class SymmetricCELoss(nn.Module):
    """SCE = alpha * CE + beta * RCE, masked at ignore_index.
    RCE clamps log q at -4 (Wang et al., ICCV 2019)."""

    def __init__(self, num_classes, ignore_index=255, alpha=1.0, beta=0.1):
        super().__init__()
        self.C = num_classes
        self.ignore = ignore_index
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, target):
        valid = target != self.ignore
        n = valid.sum()
        if n == 0:
            return logits.sum() * 0.0
        logp = F.log_softmax(logits.float(), dim=1)
        t = target.clamp(0, self.C - 1)
        ce = -logp.gather(1, t.unsqueeze(1)).squeeze(1)
        p = logp.exp()
        q = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        logq = q.clamp_min(1e-4).log().clamp_min(-4.0)
        rce = -(p * logq).sum(dim=1)
        loss = self.alpha * ce + self.beta * rce
        return (loss * valid).sum() / (n + 1e-6)


def gaussian_nll_masked(mu, var, target, valid, eps=1e-6):
    """0.5*log(sigma^2) + (y-mu)^2 / (2 sigma^2), reduced over valid px."""
    nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
    m = valid.float()
    return (nll * m).sum() / (m.sum() + eps)


def sparse_masked_l1(pred, target, valid, eps=1e-6):
    m = valid.float()
    return ((pred - target).abs() * m).sum() / (m.sum() + eps)


# ==============================================================================
# 11. METRICS
# ==============================================================================
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.C = num_classes
        self.cm = np.zeros((num_classes, num_classes), np.int64)

    @torch.no_grad()
    def update(self, logits, target, ignore=255):
        p = logits.argmax(1).flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        v = t != ignore
        idx = t[v] * self.C + p[v]
        self.cm += np.bincount(idx, minlength=self.C * self.C) \
            .reshape(self.C, self.C)

    def iou(self):
        d = np.diag(self.cm).astype(np.float64)
        u = self.cm.sum(0) + self.cm.sum(1) - d
        return np.where(u > 0, d / np.maximum(u, 1), np.nan)

    def miou(self):
        return float(np.nanmean(self.iou()))


class MaskedMAE:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, pred, target, valid):
        if valid.sum() > 0:
            self.total += float((pred[valid] - target[valid]).abs().sum())
            self.count += int(valid.sum())

    def value(self):
        return self.total / max(self.count, 1)


# ==============================================================================
# 12. EMA
# ==============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    @torch.no_grad()
    def store(self, model):
        self._backup = {k: v.detach().clone()
                        for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in sd.items():
            v.copy_(self.shadow[k].to(v.dtype))

    @torch.no_grad()
    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


# ==============================================================================
# 13. CURRICULUM + LR SCHEDULE  (blueprint §6)
# ==============================================================================
def phase_for_epoch(epoch1, cfg):
    if epoch1 <= cfg.phase1_end:
        return 1
    if epoch1 <= cfg.phase2_end:
        return 2
    return 3


def phase_loss_weights(cfg, phase):
    lam = {1: 0.0, 2: cfg.phase2_lambda, 3: 1.0}[phase]
    return {"seg": 1.0, "hydro": cfg.w_hydro * lam,
            "canopy": cfg.w_canopy * lam}


def lr_lambda_factory(cfg):
    ratio = cfg.final_lr / cfg.lr

    def lr_lambda(ep0):                                 # 0-based epoch
        ep1 = ep0 + 1
        if ep1 <= cfg.phase2_end:
            return 1.0                                  # constant 1e-4
        prog = (ep1 - cfg.phase2_end) / max(1, cfg.epochs - cfg.phase2_end)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return ratio + (1.0 - ratio) * cos              # cosine -> 1e-6
    return lr_lambda


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "A_log" in n or n.endswith(".D"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def amp_context(cfg):
    if cfg.amp and DEVICE.type == "cuda":
        dtype = torch.bfloat16 if AMP_BF16 else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


# ==============================================================================
# 14. TRAIN / EVAL LOOPS
# ==============================================================================
def _move(batch):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}


def _targets(batch, cfg):
    label = batch["label"].long()
    depth = batch["depth"].float() / cfg.depth_scale
    v_depth = torch.isfinite(depth)
    depth = torch.nan_to_num(depth, nan=0.0).unsqueeze(1)
    gedi = batch["gedi"].float() / cfg.height_scale
    v_gedi = torch.isfinite(gedi)
    gedi = torch.nan_to_num(gedi, nan=0.0).unsqueeze(1)
    return label, depth, v_depth, gedi, v_gedi


def run_epoch(model, loader, cfg, phase, optimizer=None, scaler=None,
              ema=None, epoch1=1):
    training = optimizer is not None
    model.train(training)
    w = phase_loss_weights(cfg, phase)
    sce = SymmetricCELoss(cfg.num_classes, cfg.ignore_label,
                          cfg.sce_alpha, cfg.sce_beta)
    agg = {"loss": 0.0, "seg": 0.0, "hydro": 0.0, "canopy": 0.0, "steps": 0}
    cm = ConfusionMatrix(cfg.num_classes)
    mae_h, mae_c = MaskedMAE(), MaskedMAE()

    for batch in loader:
        batch = _move(batch)
        label, depth, v_depth, gedi, v_gedi = _targets(batch, cfg)
        ctx = amp_context(cfg) if training else \
            (torch.autocast("cuda", dtype=torch.bfloat16)
             if cfg.amp and DEVICE.type == "cuda" else nullcontext())
        with torch.set_grad_enabled(training), ctx:
            out = model(batch["x"].float())
            l_seg = sce(out["seg"], label)
            l_hyd = gaussian_nll_masked(out["hydro_mu"], out["hydro_var"],
                                        depth, v_depth)
            l_can = sparse_masked_l1(out["canopy"], gedi, v_gedi)
            total = w["seg"] * l_seg + w["hydro"] * l_hyd + w["canopy"] * l_can

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
                optimizer.step()
            if ema is not None:
                ema.update(model)

        agg["loss"] += float(total)
        agg["seg"] += float(l_seg)
        agg["hydro"] += float(l_hyd)
        agg["canopy"] += float(l_can)
        agg["steps"] += 1
        with torch.no_grad():
            cm.update(out["seg"], label, cfg.ignore_label)
            mae_h.update(out["hydro_mu"].float(), depth, v_depth)
            mae_c.update(out["canopy"].float(), gedi, v_gedi)

    n = max(agg["steps"], 1)
    return {
        "loss": agg["loss"] / n, "seg_loss": agg["seg"] / n,
        "hydro_loss": agg["hydro"] / n, "canopy_loss": agg["canopy"] / n,
        "miou": cm.miou(), "hydro_mae": mae_h.value(),
        "canopy_mae": mae_c.value(), "cm": cm.cm,
    }


def train(cfg=CFG):
    cfg.check()
    print(f"\n=== BLUE SENTINEL v{VERSION} - TRAINING ===\n")
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    train_scenes = discover_scenes(cfg.train_dir)
    val_scenes = discover_scenes(cfg.val_dir)
    assert train_scenes and val_scenes, "Set cfg.train_dir / cfg.val_dir"
    fp, nf = dataset_fingerprint(train_scenes)
    print(f"train scenes: {len(train_scenes)} | val scenes: {len(val_scenes)}"
          f" | T={cfg.temporal_steps} | fingerprint: {fp} ({nf} files)")

    mean, var = load_or_compute_stats(train_scenes, cfg)
    train_loader, train_stream = make_loader(train_scenes, cfg, True,
                                             cfg.max_train_patches)
    val_loader, _ = make_loader(val_scenes, cfg, False, cfg.max_val_patches)

    model = BlueSentinelV9(cfg, mean, var).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.1f}M | D={cfg.dim} depths={cfg.depths}"
          f" d_state={cfg.d_state}")
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg))
    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.amp and DEVICE.type == "cuda" and not AMP_BF16))
    ema = ModelEMA(model, cfg.ema_decay)

    csv_path = os.path.join(WORK_DIR, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,phase,lr,loss,seg_loss,hydro_loss,canopy_loss,"
                "miou,val_loss,val_miou,val_hydro_mae,val_canopy_mae\n")

    best_miou = -1.0
    ckpt_last = os.path.join(WORK_DIR, "last_v9.pt")
    ckpt_best = os.path.join(WORK_DIR, "best_v9.pt")

    for epoch1 in range(1, cfg.epochs + 1):
        phase = phase_for_epoch(epoch1, cfg)
        model.set_phase(phase)
        train_stream.set_epoch(epoch1)
        t0 = time.time()
        tr = run_epoch(model, train_loader, cfg, phase, optimizer=optimizer,
                       scaler=scaler, ema=ema, epoch1=epoch1)
        scheduler.step()

        ema.store(model)
        ema.copy_to(model)
        va = run_epoch(model, val_loader, cfg, phase=3, optimizer=None)
        ema.restore(model)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"ep {epoch1:3d}/{cfg.epochs} [P{phase}] lr={lr_now:.2e} "
              f"loss={tr['loss']:.4f} seg={tr['seg_loss']:.4f} "
              f"hyd={tr['hydro_loss']:.4f} can={tr['canopy_loss']:.4f} "
              f"miou={tr['miou']:.3f} | val loss={va['loss']:.4f} "
              f"miou={va['miou']:.3f} hydMAE={va['hydro_mae']:.4f} "
              f"canMAE={va['canopy_mae']:.3f} ({time.time() - t0:.0f}s)")
        with open(csv_path, "a") as f:
            f.write(f"{epoch1},{phase},{lr_now:.3e},{tr['loss']:.5f},"
                    f"{tr['seg_loss']:.5f},{tr['hydro_loss']:.5f},"
                    f"{tr['canopy_loss']:.5f},{tr['miou']:.4f},"
                    f"{va['loss']:.5f},{va['miou']:.4f},{va['hydro_mae']:.5f},"
                    f"{va['canopy_mae']:.5f}\n")

        payload = {"model": model.state_dict(),
                   "ema": ema.shadow,
                   "norm_mean": mean, "norm_var": var,
                   "cfg": asdict(cfg), "epoch": epoch1,
                   "val_miou": va["miou"], "version": VERSION,
                   "dataset_fingerprint": fp}
        torch.save(payload, ckpt_last)
        if va["miou"] >= best_miou:
            best_miou = va["miou"]
            torch.save(payload, ckpt_best)
            print(f"  [ckpt] new best (val mIoU={best_miou:.3f}) -> {ckpt_best}")

    cfg.save(os.path.join(WORK_DIR, "config_v9.json"))
    with open(os.path.join(WORK_DIR, "dataset_version.json"), "w") as f:
        json.dump({"fingerprint": fp, "n_files": nf,
                   "train_dir": cfg.train_dir,
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())}, f, indent=2)
    print(f"\nTraining complete. best val mIoU={best_miou:.3f}")
    print(f"checkpoints: {ckpt_best} / {ckpt_last}")
    return ckpt_best


# ==============================================================================
# 15. CHECKPOINT LOADING / EVAL MODE
# ==============================================================================
def load_checkpoint(path, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_d = ckpt["cfg"]
    for k in ("invalid_scl", "rare_classes", "depths"):
        if k in cfg_d:
            cfg_d[k] = tuple(cfg_d[k])
    cfg = Config(**cfg_d)
    model = BlueSentinelV9(cfg, ckpt["norm_mean"], ckpt["norm_var"]).to(device)
    if "ema" in ckpt:
        sd = model.state_dict()
        for k, v in sd.items():
            v.copy_(ckpt["ema"][k].to(v.dtype))
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {path} (epoch={ckpt.get('epoch')}, "
          f"val mIoU={ckpt.get('val_miou', 0):.3f})")
    return model, cfg


def evaluate(model, cfg, scene_descs, split="test"):
    loader, _ = make_loader(scene_descs, cfg, False, cfg.max_val_patches)
    res = run_epoch(model, loader, cfg, phase=3, optimizer=None)
    iou = ConfusionMatrix(cfg.num_classes)
    iou.cm = res["cm"]
    per = iou.iou()
    print(f"\n-- {split} results --")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:16s} IoU={per[i]:.3f}  px={res['cm'][i].sum():,}")
    print(f"  mIoU={res['miou']:.3f} | hydro MAE={res['hydro_mae']:.4f} m "
          f"| canopy MAE={res['canopy_mae'] * cfg.height_scale:.2f} m")
    np.savetxt(os.path.join(WORK_DIR, f"confusion_{split}.csv"),
               res["cm"], fmt="%d", delimiter=",")
    return res


# ==============================================================================
# 16. TILED INFERENCE (Hann blending + flip TTA) + EXPORT
# ==============================================================================
def _tile_starts(dim, ps, overlap):
    stride = max(1, int(ps * (1 - overlap)))
    if dim <= ps:
        return [0]
    starts = list(range(0, dim - ps + 1, stride))
    if starts[-1] != dim - ps:
        starts.append(dim - ps)
    return starts


def _hann_kernel(ps, floor=0.05):
    h = np.maximum(np.hanning(ps).astype(np.float32), floor)
    return np.outer(h, h)


@torch.no_grad()
def predict_scene(model, scene, cfg, batch_size=None, tta=None, overlap=None):
    """scene: dict from load_scene. Returns numpy product maps."""
    batch_size = batch_size or cfg.infer_batch
    tta = cfg.infer_tta if tta is None else tta
    overlap = cfg.infer_overlap if overlap is None else overlap
    ps, C = cfg.patch_size, cfg.num_classes
    x_np = scene["x"]                                     # (T,17,H,W)
    T, _, h, w = x_np.shape
    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        x_np = np.pad(x_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    H, W = x_np.shape[-2:]
    kernel = _hann_kernel(ps)
    acc = {"seg": np.zeros((H, W, C), np.float32),
           "mu": np.zeros((H, W), np.float32),
           "var": np.zeros((H, W), np.float32),
           "can": np.zeros((H, W), np.float32)}
    weight = np.zeros((H, W), np.float32)
    coords = [(i, j) for i in _tile_starts(H, ps, overlap)
              for j in _tile_starts(W, ps, overlap)]
    variants = [(0, False, False)] if not tta else \
        [(0, False, False), (0, True, False), (0, False, True), (1, False, False)]
    print(f"Inference: {h}x{w} -> padded {H}x{W} | tiles={len(coords)} | "
          f"TTA={tta} | batch={batch_size}")
    model.eval()
    for k90, fh, fv in variants:
        for s in range(0, len(coords), batch_size):
            chunk = coords[s:s + batch_size]
            tiles = np.stack([x_np[:, :, i:i + ps, j:j + ps]
                              for i, j in chunk])          # (b,T,17,ps,ps)
            if fh:
                tiles = tiles[..., ::-1]
            if fv:
                tiles = tiles[..., ::-2 + 1]  # placeholder; replaced below
            # (re)compute transforms explicitly to keep it readable
            tiles = np.stack([x_np[:, :, i:i + ps, j:j + ps]
                              for i, j in chunk])
            if fh:
                tiles = tiles[..., ::-1]
            if fv:
                tiles = tiles[..., ::-1, :]
            if k90:
                tiles = np.rot90(tiles, 1, axes=(-2, -1))
            xb = torch.from_numpy(np.ascontiguousarray(tiles)) \
                .to(DEVICE).float()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(cfg.amp and DEVICE.type == "cuda")):
                out = model(xb)
            seg = F.softmax(out["seg"].float(), dim=1) \
                .permute(0, 2, 3, 1).cpu().numpy()         # (b,ps,ps,C)
            mu = out["hydro_mu"][:, 0].float().cpu().numpy()
            var = out["hydro_var"][:, 0].float().cpu().numpy()
            can = out["canopy"][:, 0].float().cpu().numpy()
            if k90:
                seg = np.rot90(seg, -1, axes=(1, 2))
                mu = np.rot90(mu, -1, axes=(1, 2))
                var = np.rot90(var, -1, axes=(1, 2))
                can = np.rot90(can, -1, axes=(1, 2))
            if fv:
                seg, mu, var, can = seg[:, ::-1], mu[:, ::-1], \
                    var[:, ::-1], can[:, ::-1]
            if fh:
                seg, mu, var, can = seg[:, :, ::-1], mu[:, :, ::-1], \
                    var[:, :, ::-1], can[:, :, ::-1]
            for n, (i, j) in enumerate(chunk):
                sl = np.s_[i:i + ps, j:j + ps]
                acc["seg"][sl] += seg[n] * kernel[..., None]
                acc["mu"][sl] += mu[n] * kernel
                acc["var"][sl] += var[n] * kernel
                acc["can"][sl] += can[n] * kernel
            weight[:0]  # no-op
            for i, j in chunk:
                weight[i:i + ps, j:j + ps] += kernel
    w_safe = np.maximum(weight, 1e-6)
    seg = acc["seg"] / w_safe[..., None]
    seg = seg / np.maximum(seg.sum(-1, keepdims=True), 1e-6)
    res = {
        "seg_probs": seg[:h, :w].astype(np.float32),
        "classes": seg.argmax(-1).astype(np.uint8)[:h, :w],
        "hydro_depth_m": (acc["mu"] / w_safe)[:h, :w].astype(np.float32),
        "hydro_var": (acc["var"] / w_safe)[:h, :w].astype(np.float32),
        "canopy_height_m": ((acc["can"] / w_safe) * cfg.height_scale)
        [:h, :w].astype(np.float32),
    }
    res["hydro_std_m"] = np.sqrt(np.maximum(res["hydro_var"], 0.0))
    return res


def export_geotiff(reference_tif, array, out_path, dtype="float32",
                   nodata=None):
    try:
        import rasterio
    except ImportError:
        print(f"  [export] rasterio missing; NPZ fallback for "
              f"{os.path.basename(out_path)}")
        np.savez(out_path.replace(".tif", ".npz"), array=array)
        return None
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    if array.ndim == 2:
        array = array[..., None]
    bands = array.shape[-1]
    profile.update(count=bands, dtype=dtype, driver="GTiff", compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256,
                   predictor=2 if dtype == "float32" else 1)
    if nodata is not None:
        profile.update(nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(bands):
            dst.write(array[..., b].astype(dtype), b + 1)
    print(f"  saved {out_path}")
    return out_path


def export_products(model, cfg, scene_path, ref_tif, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    desc = ("folder", scene_path) if os.path.isdir(scene_path) \
        else ("npz", scene_path)
    scene = load_scene(desc, cfg)
    res = predict_scene(model, scene, cfg)
    plan = [("classes.tif", "classes", "uint8", 255),
            ("seg_probs.tif", "seg_probs", "float32", None),
            ("hydro_depth_m.tif", "hydro_depth_m", "float32", -999.0),
            ("hydro_std_m.tif", "hydro_std_m", "float32", -999.0),
            ("canopy_height_m.tif", "canopy_height_m", "float32", -999.0)]
    for fname, key, dtype, nodata in plan:
        export_geotiff(ref_tif, res[key], os.path.join(out_dir, fname),
                       dtype=dtype, nodata=nodata)
    pa_ha = (cfg.pixel_size_m ** 2) / 1e4
    rows = [("metric", "value", "unit")]
    for i, n in enumerate(CLASS_NAMES):
        rows.append((f"area_{n.replace(' ', '_')}",
                     f"{float((res['classes'] == i).sum()) * pa_ha:.2f}", "ha"))
    m_valid = np.isfinite(res["canopy_height_m"])
    rows.append(("canopy_height_mean",
                 f"{float(res['canopy_height_m'][m_valid].mean()):.2f}", "m"))
    rows.append(("hydro_depth_mean",
                 f"{float(res['hydro_depth_m'].mean()):.3f}", "m"))
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    fp, nf = dataset_fingerprint([desc])
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_path, "version": VERSION,
                   "dates": scene["dates"].tolist(),
                   "dataset_fingerprint": fp, "dataset_files": nf}, f, indent=2)
    print(f"All products exported to: {out_dir}")
    return res


# ==============================================================================
# 17. SMOKE TEST (loader -> build -> 2 phases -> ckpt -> tiled infer -> scans)
# ==============================================================================
def _write_dummy_scene(root, name, cfg, T, H=96, W=96, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for t in range(T):
        y = 2020 + t
        keys = {
            "s2": rng.uniform(0, 0.5, (H, W, 10)).astype(np.float32),
            "scl": rng.choice([4, 5, 6, 8, 9], (H, W)).astype(np.float32),
            "s1": rng.uniform(-30, 0, (H, W, 2)).astype(np.float32),
            "lband": rng.uniform(-35, -5, (H, W, 2)).astype(np.float32),
            "precip": rng.uniform(0, 300, (H, W)).astype(np.float32),
            "inundation": rng.uniform(0, 3000, (H, W)).astype(np.float32),
            "date": np.float32(y + 0.5),
        }
        if t == T - 1:
            gedi = np.full((H, W), np.nan, np.float32)
            track = rng.uniform(0, 25, (H, 4)).astype(np.float32)
            gedi[:, 10:14] = track
            keys["gedi"] = gedi
            keys["label"] = rng.choice([0, 1, 2, 3, 255],
                                       (H, W)).astype(np.int64)
        np.savez(os.path.join(d, f"step_{y}.npz"), **keys)
    return d


def smoke_test():
    print(f"\n=== SMOKE TEST v{VERSION} ===")
    cfg = Config(patch_size=64, temporal_steps=3, dim=24, depths=(1, 1, 1, 1),
                 d_state=8, batch_size=2, epochs=2, phase1_end=1,
                 phase2_end=1, max_train_patches=8, max_val_patches=4,
                 min_valid_frac=0.1, max_cloud_frac=0.95, ssm_chunk=64,
                 infer_tta=True, amp=False)
    cfg.check()
    root = os.path.join(WORK_DIR, "smoke_data")
    shutil.rmtree(root, ignore_errors=True)
    for split, n in (("train", 2), ("val", 1)):
        for s in range(n):
            _write_dummy_scene(root, f"{split}/scene_{s}", cfg,
                               cfg.temporal_steps, seed=s)
    train_scenes = discover_scenes(os.path.join(root, "train"))
    val_scenes = discover_scenes(os.path.join(root, "val"))
    assert len(train_scenes) == 2 and len(val_scenes) == 1
    print(f"  scenes: train={len(train_scenes)} val={len(val_scenes)}")

    mean = np.zeros(N_INPUT_CH, np.float32)
    var = np.ones(N_INPUT_CH, np.float32)
    var[CH_SCL] = 121.0
    model = BlueSentinelV9(cfg, mean, var).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  model params: {n_params:.2f}M")

    # fallback selective-scan numerical check vs naive recurrence
    u = torch.randn(2, 33, 8)
    dt = F.softplus(torch.randn(2, 33, 8))
    A = -torch.exp(torch.randn(8, 4))
    Bm = torch.randn(2, 33, 4)
    Cm = torch.randn(2, 33, 4)
    Dv = torch.randn(8)
    y_ref = selective_scan_torch(u, dt, A, Bm, Cm, Dv, chunk=16,
                                 use_ckpt=False)
    h = torch.zeros(2, 8, 4)
    ys = []
    for t in range(33):
        h = torch.exp(dt[:, t].unsqueeze(-1) * A) * h + \
            (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * \
            u[:, t].unsqueeze(-1)
        ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
    y_naive = torch.stack(ys, 1) + u * Dv
    err = float((y_ref - y_naive).abs().max())
    assert err < 1e-4, f"selective scan mismatch: {err}"
    print(f"  selective-scan fallback vs naive: max err {err:.2e} OK")

    train_loader, stream = make_loader(train_scenes, cfg, True, 8)
    val_loader, _ = make_loader(val_scenes, cfg, False, 4)
    opt = make_optimizer(model, cfg)
    ema = ModelEMA(model, 0.9)
    for ep in (1, 2):
        phase = phase_for_epoch(ep, cfg)
        model.set_phase(phase)
        stream.set_epoch(ep)
        tr = run_epoch(model, train_loader, cfg, phase, optimizer=opt, ema=ema)
        assert np.isfinite(tr["loss"]), "non-finite loss"
        frozen_ok = all(not p.requires_grad
                        for p in model.hydro_head.parameters()) == (phase == 1)
        assert frozen_ok, "curriculum freeze/unfreeze broken"
        print(f"  phase {phase}: loss={tr['loss']:.4f} miou={tr['miou']:.3f} "
              f"hyd={tr['hydro_loss']:.4f} can={tr['canopy_loss']:.4f} "
              f"| hydro head frozen={phase == 1}")
    va = run_epoch(model, val_loader, cfg, phase=3)
    assert np.isfinite(va["loss"])
    print(f"  val: loss={va['loss']:.4f} miou={va['miou']:.3f}")

    ckpt = os.path.join(WORK_DIR, "smoke_v9.pt")
    torch.save({"model": model.state_dict(), "ema": ema.shadow,
                "norm_mean": mean, "norm_var": var, "cfg": asdict(cfg),
                "epoch": 2, "val_miou": va["miou"], "version": VERSION}, ckpt)
    m2, cfg2 = load_checkpoint(ckpt)
    print("  checkpoint save/load OK")

    scene = load_scene(val_scenes[0], cfg2)
    res = predict_scene(m2, scene, cfg2, batch_size=2, tta=True)
    assert res["classes"].shape == scene["label"].shape
    assert np.allclose(res["seg_probs"].sum(-1), 1.0, atol=1e-3)
    assert np.isfinite(res["hydro_depth_m"]).all()
    print(f"  tiled inference OK: {sorted(res.keys())}")
    print(f"  canopy range [{res['canopy_height_m'].min():.1f}, "
          f"{res['canopy_height_m'].max():.1f}] m")
    shutil.rmtree(root, ignore_errors=True)
    print("=== SMOKE TEST PASSED ===")
    return True


# ==============================================================================
# 18. CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=f"Blue Sentinel v{VERSION}")
    ap.add_argument("mode", choices=["smoke", "train", "eval", "export"],
                    nargs="?", default="smoke")
    ap.add_argument("scene", nargs="?", help="scene folder/cube NPZ (export)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (export)")
    ap.add_argument("out", nargs="?", help="output dir (export)")
    ap.add_argument("--config", help="path to config_v9.json")
    ap.add_argument("--ckpt", default=os.path.join(WORK_DIR, "best_v9.pt"))
    ap.add_argument("--train-dir")
    ap.add_argument("--val-dir")
    ap.add_argument("--test-dir")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    cfg = Config.load(args.config) if args.config else Config()
    if args.train_dir:
        cfg.train_dir = args.train_dir
    if args.val_dir:
        cfg.val_dir = args.val_dir
    if args.test_dir:
        cfg.test_dir = args.test_dir
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.epochs = args.epochs
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.no_tta:
        cfg.infer_tta = False

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        model, mcfg = load_checkpoint(args.ckpt)
        scenes = discover_scenes(cfg.test_dir)
        assert scenes, "eval: set --test-dir"
        evaluate(model, mcfg, scenes, split="test")
    elif args.mode == "export":
        if not (args.scene and args.ref and args.out):
            ap.error("export requires <scene> <ref.tif> <out_dir>")
        model, mcfg = load_checkpoint(args.ckpt)
        if args.no_tta:
            mcfg.infer_tta = False
        export_products(model, mcfg, args.scene, args.ref, args.out)


if __name__ == "__main__":
    main()