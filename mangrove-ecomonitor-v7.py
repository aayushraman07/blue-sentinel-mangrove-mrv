"""
================================================================================
 MANGROVE ECO-MONITOR v7.0  "BLUE SENTINEL ULTRA"
 Unified production system: merges v5 stability with v6 multi-temporal design.

 REQUIRES: Python >= 3.10, TensorFlow >= 2.16 (Keras 3), Linux or WSL2.
           Optional: rasterio (COG/GeoJSON export), scikit-learn, matplotlib.

--------------------------------------------------------------------------------
 WHAT V7 IS
--------------------------------------------------------------------------------
 * v5 core (Swin-V2 siamese encoders, per-level fusion, FiLM change decoder,
   Kendall task weighting, EMA, calibration, Hann/TTA tiling, MRV export)
 * v6 features DONE CORRECTLY: T-step temporal transformer over bottleneck
   sequences (per-location trajectory embeddings), global scene-context branch,
   species / change_year / recovery / damage heads (pluggable, auto-pruned).
 * ONE unified scene abstraction: a scene is either a FOLDER of dated NPZs
   (multi-temporal) or a single legacy pair-NPZ (v5 schema). Both become a
   stack of T steps; T=2 reproduces v5 behaviour exactly.

--------------------------------------------------------------------------------
 NEW BUGS FIXED IN V7 (present in v5 and/or v6, found by v7 audit)
--------------------------------------------------------------------------------
 [F11] v5 change decoder ended at ps/2 (change_widths had 5 entries -> 4
       upsamples from the /32 bottleneck) while targets are (ps,ps): guaranteed
       shape crash with the change head ON. v7 upsamples in a while-loop until
       full resolution, independent of the width list length.
 [F12] v5 derived boundary/deep targets with tf.nn.max_pool2d / avg_pool2d on
       RANK-3 tensors before batching -> ValueError. v7 expands dims explicitly.
 [F13] v6 ChangeF1 / MeanIoU had their state-variable creation indented OUT of
       __init__ (class-body no-op) -> metrics crashed on first update.
 [F14] v6 created Conv2D / CrossAttentionFusion layers INSIDE call() -> new,
       untracked, untrained weights on every forward pass. v7 pre-builds every
       layer once and shares it across all T time steps (true siamese).
 [F15] v6 decoder upsampled 16x straight from the bottleneck with NO skip
       connections. v7 keeps temporal aggregation at the bottleneck only and
       decodes the reference step with the full v5 skip pathway.
 [F16] v6 make_dataset_multi output_signature had a spurious leading batch dim
       and mismatched target specs -> from_generator died on first element.
 [F17] v5 pretrain contained a dead tf.function referencing an undefined
       helper plus a step.__globals__ monkey-patch. Removed; single clean path.
 [F18] numpy flip/rot90 views carry negative strides which tf.convert_to_tensor
       rejects inside from_generator -> v7 forces ascontiguousarray.

--------------------------------------------------------------------------------
 SCENE SCHEMA v7 (superset of v5/v6)
--------------------------------------------------------------------------------
 A "scene" is EITHER:
   (a) a folder containing >=1 dated step NPZs, each with:
         s1 (H,W,2|4) dB | s2 (H,W,10) | optional: s1_std, scl, aux, date
         (decimal year; else parsed from filename), and per-step labels
         (label/carbon/eco/change/risk/species/debris/recovery/change_year).
       Supervision labels are read from the LAST step; interval labels
       (change, recovery, debris, change_year) should live in the last file.
   (b) a legacy pair NPZ (v5 keys: s1,s2,...,s1_t2,s2_t2,...,date_t1,date_t2).
 Stacks are date-sorted, uniformly sub-sampled or edge-padded to T steps.

 HEAD DATA SOURCES (honest wiring; validator auto-disables missing ones)
   seg      : WorldCover+GSW+GMW labels            key "label"       [ON ]
   change   : Hansen lossyear / GMW alerts / pairs key "change"      [ON ]
   quality  : derived from SCL (free)              -                 [ON ]
   boundary : derived from seg (free)              -                 [ON ]
   species  : VHR/field inventories                key "species"     [auto]
   change_year : Hansen-style loss year (decimal)  key "change_year" [auto]
   recovery : regrowth maps / dense time series    key "recovery"    [auto]
   damage   : cyclone debris (SAR texture labels)  key "debris"      [auto]
   carbon   : GEDI / ESA CCI Biomass               key "carbon"      [OFF]
   eco      : Landsat/MODIS LST                    key "eco"         [OFF]
   risk     : surge model / heuristic              key "risk"        [OFF]

 MODES: python mangrove_v7.py smoke | pretrain | train | eval | export
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
import base64
import argparse
import platform
import subprocess
import warnings
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "7.0.0"
MIN_TF = (2, 16)


# ==============================================================================
# 1. ENVIRONMENT, GPU, PLATFORM GUARDS
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


def environment_check():
    ok = True
    if _parse_version(tf.__version__) < MIN_TF:
        print(f"[FATAL] TensorFlow {tf.__version__} < {'.'.join(map(str, MIN_TF))}. "
              f"v7 requires Keras 3 APIs. pip install -U tensorflow")
        ok = False
    gpus = tf.config.list_physical_devices("GPU")
    if platform.system() == "Windows" and not gpus and shutil.which("nvidia-smi"):
        print("=" * 78)
        print("[HARDWARE BLOCKER] Windows + NVIDIA driver present, but TF sees no GPU.")
        print("TF dropped native-Windows GPU support after v2.10. Fix with WSL2:")
        print("  1) wsl --install -d Ubuntu   (admin PowerShell)")
        print("  2) Install the Windows NVIDIA driver only (it serves WSL2 too)")
        print("  3) In Ubuntu: pip install tensorflow[and-cuda]")
        print("=" * 78)
        ok = False
    return ok, gpus


ENV_OK, GPUS = environment_check()
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v7/"
os.makedirs(WORK_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(42)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
for g in GPUS:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

STRATEGY = tf.distribute.MirroredStrategy() if GPUS else tf.distribute.get_strategy()
N_REPLICAS = STRATEGY.num_replicas_in_sync
GPU_NAMES = []
for g in GPUS:
    try:
        GPU_NAMES.append(tf.config.experimental.get_device_details(g).get("device_name", ""))
    except Exception:
        GPU_NAMES.append("")
print(f"TF {tf.__version__} | GPUs: {len(GPUS)} {GPU_NAMES} | replicas: {N_REPLICAS}")


def _gpu_vram_gb():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=10).decode().strip().splitlines()
        return float(out[0]) / 1024.0 if out else None
    except Exception:
        return None


# ==============================================================================
# 2. CONFIG
# ==============================================================================
@dataclass
class Config:
    # ---- data ----
    train_dir: str = "/kaggle/input/mangrove-v7/train/"
    val_dir: str = "/kaggle/input/mangrove-v7/val/"
    test_dir: str = "/kaggle/input/mangrove-v7/test/"
    patch_size: int = 256                 # -> 64/32/16/8, window 8 divides all
    train_stride: int = 128
    max_train_patches: int = 8000
    max_val_patches: int = 1200
    min_valid_frac: float = 0.35
    max_cloud_frac: float = 0.85
    ignore_label: int = 255
    num_classes: int = 5
    invalid_scl: tuple = (0, 1, 2, 3, 7, 8, 9, 10, 11)
    cloud_dilate: int = 5
    aux_names: tuple = ("elev", "slope", "ndvi_trend", "flood_freq",
                        "lst_trend", "s1_trend")
    aux_scale: tuple = (50.0, 20.0, 0.1, 1.0, 50.0, 1.0)
    rare_classes: tuple = (2, 3)
    rare_boost: float = 4.0
    # ---- temporal ----
    temporal_steps: int = 4               # T=2 reproduces v5 pair behaviour
    max_time_gap_years: float = 20.0
    temporal_heads: int = 8
    temporal_blocks: int = 2
    temporal_dropout: float = 0.1
    # ---- model ----
    embed_dims: tuple = (48, 96, 192, 384)
    depths: tuple = (2, 2, 2, 2)
    window_size: int = 8
    drop_path: float = 0.15
    mlp_ratio: float = 4.0
    fusion_heads: int = 8
    cross_attn_levels: tuple = (2, 3)
    dec_widths: tuple = (192, 96, 48)
    refine_widths: tuple = (32, 24)
    change_widths: tuple = (128, 96, 64, 48, 32)   # length-agnostic (F11)
    decoder_swin_levels: int = 2
    use_se: bool = True
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.3
    time_embed_dim: int = 64
    num_species: int = 3
    species_names: tuple = ("Avicennia", "Rhizophora", "Other")
    active_heads: tuple = ("seg", "change", "quality", "boundary",
                           "species", "change_year", "recovery", "damage")
    # ---- training ----
    hardware: str = "auto"
    batch_per_replica: int = 4
    grad_accum: int = 1
    epochs: int = 120
    lr: float = 2e-4
    lr_warmup_epochs: int = 3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    task_weighting: str = "learned"
    fixed_loss_weights: dict = field(default_factory=lambda: {
        "seg": 1.0, "change": 1.2, "quality": 0.2, "boundary": 0.3,
        "species": 0.8, "change_year": 0.5, "recovery": 0.5, "damage": 0.5,
        "carbon": 0.6, "eco": 0.25, "risk": 0.25})
    carbon_scale: float = 100.0
    eco_scale: tuple = (50.0, 1.0)
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    deterministic: bool = False
    auto_batch: bool = False
    oom_retries: int = 2
    max_scene_failures: int = 25          # failure budget before aborting
    pretrain_weights: str = ""
    # ---- augmentation ----
    augment: bool = True
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_noise: float = 0.02
    modality_dropout_p: float = 0.08
    # ---- pretraining ----
    pretrain_epochs: int = 40
    pretrain_lr: float = 3e-4
    pretrain_mask_ratio: float = 0.5
    pretrain_block: int = 32
    # ---- inference ----
    infer_batch: int = 8
    infer_tta: bool = True
    infer_overlap: float = 0.5
    mc_samples: int = 0
    seg_temperature: float = 1.0
    change_threshold: float = 0.5
    change_conformal_q: float = 0.0       # >0 overrides threshold (conformal)
    # ---- MRV ----
    pixel_size_m: float = 10.0
    carbon_to_co2e: float = 44.0 / 12.0
    c_fraction: float = 0.5
    shuffle_buffer: int = 256

    @property
    def aux_channels(self):
        return len(self.aux_names)

    def save(self, path):
        d = asdict(self)
        for k in ("invalid_scl", "aux_names", "aux_scale", "rare_classes",
                  "embed_dims", "depths", "cross_attn_levels", "dec_widths",
                  "refine_widths", "change_widths", "active_heads", "eco_scale",
                  "species_names"):
            if k in d:
                d[k] = list(d[k])
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        for k in ("invalid_scl", "aux_names", "aux_scale", "rare_classes",
                  "embed_dims", "depths", "cross_attn_levels", "dec_widths",
                  "refine_widths", "change_widths", "active_heads", "eco_scale",
                  "species_names"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)


CFG = Config()

HW_PRESETS = {
    "h100": (32, 1, 32), "a100": (24, 1, 32), "rtx6000ada": (16, 1, 16),
    "a6000": (16, 1, 16), "l40": (12, 1, 16), "4090": (8, 1, 8),
    "l4": (6, 1, 8), "t4x2": (4, 2, 8), "cpu": (2, 1, 2),
}


def resolve_hardware(cfg):
    name = " ".join(GPU_NAMES)
    hw = cfg.hardware
    if hw == "auto":
        n = name.lower()
        if "h100" in n:
            hw = "h100"
        elif "a100" in n:
            hw = "a100"
        elif "6000 ada" in n or "rtx 6000" in n:
            hw = "rtx6000ada"
        elif "a6000" in n:
            hw = "a6000"
        elif "l40" in n:
            hw = "l40"
        elif "4090" in n:
            hw = "4090"
        elif "l4" in n and "l40" not in n:
            hw = "l4"
        elif "t4" in n:
            hw = "t4x2"
        else:
            vram = _gpu_vram_gb()
            if vram is None:
                hw = "cpu" if not GPUS else "t4x2"
            elif vram >= 70:
                hw = "a100"
            elif vram >= 40:
                hw = "rtx6000ada"
            elif vram >= 20:
                hw = "4090"
            else:
                hw = "t4x2"
    b, ga, ib = HW_PRESETS.get(hw, HW_PRESETS["t4x2"])
    cfg.batch_per_replica, cfg.grad_accum, cfg.infer_batch = b, ga, ib
    print(f"Hardware: {hw} ({name.strip() or 'cpu'}) | batch/replica={b} "
          f"| grad_accum={ga} | infer_batch={ib}")
    return hw


# ==============================================================================
# 3. UNIFIED SCENE LOADING (folder-of-steps OR legacy pair NPZ -> T-step stack)
# ==============================================================================
class NPZSchemaError(ValueError):
    pass


def _process_s1(s1):
    s1 = s1.astype(np.float32)
    if np.nanmin(s1) >= 0:
        s1 = 10.0 * np.log10(np.clip(s1, 1e-6, None))
    return np.clip(s1, -35.0, 5.0)


def _process_s2(s2):
    s2 = s2.astype(np.float32)
    mx = np.nanmax(s2) if s2.size else 0.0
    if mx > 20000:
        s2 *= 1.0 / 65535.0
    elif mx > 100:
        s2 *= 1e-4
    return np.clip(s2, 0.0, 1.5)


def _sanitize(arr, name, path):
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / arr.size
        if frac > 0.30:
            raise NPZSchemaError(f"{path}: '{name}' is {frac:.0%} non-finite")
        print(f"  [sanitize] {os.path.basename(path)}:{name} "
              f"{n_bad} non-finite px -> 0")
        arr = np.where(bad, 0.0, arr)
    return arr


def _date_from_name(path):
    """Parse a decimal year from filenames like 2021-06.npz / scene_2021.npz."""
    base = os.path.basename(path)
    m = re.search(r"((?:19|20)\d{2})[-_.]?(\d{2})?", base)
    if not m:
        return 0.0
    y = int(m.group(1))
    mo = m.group(2)
    if mo and 1 <= int(mo) <= 12:
        return y + (int(mo) - 1) / 12.0
    return float(y)


def _fix_aux(aux, H, W, A):
    if aux.shape[-1] != A:
        fixed = np.zeros((H, W, A), np.float32)
        n = min(A, aux.shape[-1])
        fixed[..., :n] = aux[..., :n]
        aux = fixed
    return aux


def _label_maps_from(d, files, H, W):
    """Read every optional label key with sentinels (single npz handle)."""
    g = lambda k, shp, val, dt: (d[k].astype(dt) if k in files
                                 else np.full(shp, val, dt))
    return {
        "label": g("label", (H, W), 255, np.int32),
        "carbon": g("carbon", (H, W), -999.0, np.float32),
        "eco": g("eco", (H, W, 2), -999.0, np.float32),
        "change": g("change", (H, W), 255, np.int32),
        "risk": g("risk", (H, W), -999.0, np.float32),
        "species": g("species", (H, W), 255, np.int32),
        "debris": g("debris", (H, W), 255, np.int32),
        "recovery": g("recovery", (H, W), 255, np.int32),
        "change_year": g("change_year", (H, W), -999.0, np.float32),
    }


def load_step_npz(path, cfg):
    """One time step file. Returns (s1,s2,scl,aux,date, label_maps)."""
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
    with np.load(path) as d:
        files = set(d.files)
        if "s1" not in files or "s2" not in files:
            raise NPZSchemaError(f"{path}: missing required 's1'/'s2'")
        s1 = _sanitize(_process_s1(d["s1"]), "s1", path)
        s2 = _sanitize(_process_s2(d["s2"]), "s2", path)
        if s1.ndim != 3 or s2.ndim != 3 or s2.shape[-1] != 10:
            raise NPZSchemaError(f"{path}: bad shapes s1{s1.shape} s2{s2.shape}")
        H, W = s2.shape[:2]
        if s1.shape[:2] != (H, W):
            raise NPZSchemaError(f"{path}: s1/s2 spatial mismatch")
        std = (d["s1_std"].astype(np.float32) if "s1_std" in files
               else np.zeros((H, W, 2), np.float32))
        s1 = np.concatenate([s1, _sanitize(std, "s1_std", path)], -1)
        scl = (d["scl"].astype(np.float32)[..., None] if "scl" in files
               else np.full((H, W, 1), 4.0, np.float32))
        aux = (d["aux"].astype(np.float32) if "aux" in files
               else np.zeros((H, W, A), np.float32))
        aux = _sanitize(_fix_aux(aux, H, W, A), "aux", path) / scale
        date = float(d["date"]) if "date" in files else _date_from_name(path)
        labels = _label_maps_from(d, files, H, W)
    return s1, s2, scl, aux, np.float32(date), labels


def _load_pair_npz(path, cfg):
    """Legacy v5 pair schema -> two steps + root labels."""
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
    with np.load(path) as d:
        files = set(d.files)
        if "s1" not in files or "s2" not in files:
            raise NPZSchemaError(f"{path}: missing 's1'/'s2'")

        def step(s1k, s2k, sclk, auxk, datek, default_date):
            s1 = _sanitize(_process_s1(d[s1k]), s1k, path)
            s2 = _sanitize(_process_s2(d[s2k]), s2k, path)
            H, W = s2.shape[:2]
            std = (d[s1k + "_std"].astype(np.float32) if s1k + "_std" in files
                   else np.zeros((H, W, 2), np.float32))
            s1 = np.concatenate([s1, _sanitize(std, s1k + "_std", path)], -1)
            scl = (d[sclk].astype(np.float32)[..., None] if sclk in files
                   else np.full((H, W, 1), 4.0, np.float32))
            aux = (d[auxk].astype(np.float32) if auxk in files
                   else np.zeros((H, W, A), np.float32))
            aux = _sanitize(_fix_aux(aux, H, W, A), auxk, path) / scale
            date = float(d[datek]) if datek in files else default_date
            return s1, s2, scl, aux, np.float32(date)

        t1 = step("s1", "s2", "scl", "aux", "date_t1", 0.0)
        has_t2 = "s1_t2" in files and "s2_t2" in files
        t2 = (step("s1_t2", "s2_t2", "scl_t2", "aux_t2", "date_t2", 1.0)
              if has_t2 else t1)
        H, W = t1[1].shape[:2]
        labels = _label_maps_from(d, files, H, W)
        if not has_t2:
            labels["change"] = np.full((H, W), 255, np.int32)
            labels["recovery"] = np.full((H, W), 255, np.int32)
    return [t1, t2], labels


def _assemble_scene(steps, labels, cfg):
    """Date-sort, uniformly sub-sample or edge-pad to exactly T steps."""
    T = cfg.temporal_steps
    steps = sorted(steps, key=lambda s: s[4])
    if len(steps) > T:
        idx = np.unique(np.linspace(0, len(steps) - 1, T).round().astype(int))
        steps = [steps[int(i)] for i in idx]
    while len(steps) < T:
        steps.append(steps[-1])
    dates = np.array([s[4] for s in steps], np.float32)
    if np.allclose(dates, dates[0]):                    # no real dates: synthesize
        dates = dates[0] + np.arange(T, dtype=np.float32)
    scene = {
        "s1_seq": np.stack([s[0] for s in steps], 0).astype(np.float32),
        "s2_seq": np.stack([s[1] for s in steps], 0).astype(np.float32),
        "scl_seq": np.stack([s[2] for s in steps], 0).astype(np.float32),
        "aux_seq": np.stack([s[3] for s in steps], 0).astype(np.float32),
        "dates": dates.astype(np.float32),
    }
    scene.update(labels)
    return scene


def discover_scenes(root):
    """A scene is a sub-folder with NPZs, or a top-level NPZ (legacy pair)."""
    if not root or not os.path.isdir(root):
        return []
    scenes = [("folder", p) for p in sorted(glob.glob(os.path.join(root, "*")))
              if os.path.isdir(p) and glob.glob(os.path.join(p, "*.npz"))]
    scenes += [("npz", p) for p in sorted(glob.glob(os.path.join(root, "*.npz")))]
    return scenes


def load_scene(desc, cfg):
    kind, path = desc
    if kind == "folder":
        steps, last_labels, n_fail = [], None, 0
        for p in sorted(glob.glob(os.path.join(path, "*.npz"))):
            try:
                s1, s2, scl, aux, date, labels = load_step_npz(p, cfg)
                steps.append((s1, s2, scl, aux, date))
                last_labels = labels
            except Exception as e:
                n_fail += 1
                print(f"  [data] bad step {os.path.basename(p)}: {str(e)[:100]}")
        if not steps:
            raise NPZSchemaError(f"{path}: no valid step files ({n_fail} failed)")
        return _assemble_scene(steps, last_labels, cfg)
    steps, labels = _load_pair_npz(path, cfg)
    return _assemble_scene(steps, labels, cfg)


def scene_from_arrays(s1, s2, scl=None, aux=None, t2=None, dates=None, cfg=CFG):
    """Ad-hoc inference helper: arrays -> unified scene dict (aux pre-scaled)."""
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
    H, W = s2.shape[:2]
    if s1.shape[-1] == 2:
        s1 = np.concatenate([s1, np.zeros_like(s1)], -1)
    scl = np.full((H, W, 1), 4.0, np.float32) if scl is None \
        else scl[..., None].astype(np.float32)
    aux = np.zeros((H, W, A), np.float32) if aux is None \
        else aux.astype(np.float32) / scale
    d0, d1 = (0.0, 1.0) if dates is None else (float(dates[0]), float(dates[-1]))
    steps = [(s1.astype(np.float32), s2.astype(np.float32), scl, aux,
              np.float32(d0))]
    if t2 is not None:
        s1b = t2["s1"]
        if s1b.shape[-1] == 2:
            s1b = np.concatenate([s1b, np.zeros_like(s1b)], -1)
        s2b = t2["s2"].astype(np.float32)
        sclb = (t2["scl"][..., None].astype(np.float32)
                if t2.get("scl") is not None else scl)
        auxb = (t2["aux"].astype(np.float32) / scale
                if t2.get("aux") is not None else aux)
        steps.append((s1b.astype(np.float32), s2b, sclb, auxb, np.float32(d1)))
    labels = _label_maps_from({}, set(), H, W)
    return _assemble_scene(steps, labels, cfg)


HEAD_REQUIRED_KEYS = {
    "seg": ["label"], "change": ["change"], "carbon": ["carbon"],
    "eco": ["eco"], "risk": ["risk"], "species": ["species"],
    "change_year": ["change_year"], "recovery": ["recovery"],
    "damage": ["debris"], "quality": [], "boundary": [],
}


def validate_heads_vs_data(scene_descs, cfg, sample=5):
    """Pre-flight: disable enabled heads whose label key is absent everywhere."""
    present = set()
    for desc in scene_descs[:sample]:
        try:
            if desc[0] == "folder":
                for p in sorted(glob.glob(os.path.join(desc[1], "*.npz"))):
                    with np.load(p) as d:
                        present |= set(d.files)
            else:
                with np.load(desc[1]) as d:
                    present |= set(d.files)
        except Exception:
            continue
    active = []
    for h in cfg.active_heads:
        need = HEAD_REQUIRED_KEYS.get(h, [])
        if need and not any(k in present for k in need):
            print(f"  [heads] '{h}' enabled but no '{need[0]}' key in sampled "
                  f"scenes -> head DISABLED (wire labels, then re-enable).")
        else:
            active.append(h)
    if not active:
        raise NPZSchemaError("No task head has data; check your scene schema.")
    if tuple(active) != tuple(cfg.active_heads):
        cfg.active_heads = tuple(active)
        print(f"  [heads] active set resolved to: {list(active)}")
    return cfg


# ==============================================================================
# 4. PATCH GENERATOR (class-balanced, per-scene fault isolation, numpy geo-aug)
# ==============================================================================
def _patch_weight(lab, cfg):
    valid = lab != cfg.ignore_label
    if valid.sum() == 0:
        return 0.0
    counts = np.bincount(lab[valid].ravel(), minlength=cfg.num_classes) \
        .astype(np.float32)
    frac = counts / max(counts.sum(), 1)
    return 1.0 + cfg.rare_boost * float(frac[list(cfg.rare_classes)].sum())


def _slice_spatial(v, sl):
    return v[(slice(None),) + sl] if v.ndim == 4 else v[sl]


def _np_geometric_aug(patch, rng):
    """Consistent flips/rot90 across every spatial array (F18: contiguous)."""
    p = dict(patch)
    lr = rng.random() < 0.5
    ud = rng.random() < 0.5
    k = int(rng.integers(0, 4))

    def t(v):
        ax = (-3, -2) if v.ndim >= 3 else (-2, -1)
        if lr:
            v = np.flip(v, axis=ax[1])
        if ud:
            v = np.flip(v, axis=ax[0])
        if k:
            v = np.rot90(v, k, axes=ax)
        return np.ascontiguousarray(v)

    for key, v in p.items():
        if key != "dates" and isinstance(v, np.ndarray) and v.ndim >= 2:
            p[key] = t(v)
    return p


def patch_generator(scene_descs, cfg, training, max_patches, seed=42):
    """Yields patch dicts. Per-scene try/except with a failure budget (no
    stalls: one corrupt scene can never wedge the pipeline)."""
    ps = cfg.patch_size
    stride = cfg.train_stride if training else ps
    rng = np.random.default_rng(seed)
    descs = list(scene_descs)
    if training:
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
            if failed >= cfg.max_scene_failures:
                raise RuntimeError(f"Too many scene failures ({failed}); "
                                   f"fix your data root.") from e
            continue
        lab = scene["label"]
        h, w = lab.shape
        if h < ps or w < ps:
            continue
        coords = []
        for i in range(0, h - ps + 1, stride):
            for j in range(0, w - ps + 1, stride):
                lp = lab[i:i + ps, j:j + ps]
                if float(np.mean(lp != cfg.ignore_label)) < cfg.min_valid_frac:
                    continue
                sc = scene["scl_seq"][-1, i:i + ps, j:j + ps]
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
            patch = {k: (_slice_spatial(v, sl) if isinstance(v, np.ndarray)
                         and v.ndim >= 2 else v) for k, v in scene.items()}
            if training and cfg.augment:
                patch = _np_geometric_aug(patch, rng)
            yield patch
            emitted += 1
        del scene
        gc.collect()
    if emitted == 0:
        raise RuntimeError(
            "patch_generator emitted 0 patches. Check: scene dirs exist, "
            "patch_size <= scene size, min_valid_frac / max_cloud_frac not "
            "too strict, labels present.")


# ==============================================================================
# 5. FEATURE ENGINEERING (shared across time steps)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FeatureEngineer(layers.Layer):
    """S1: [vv,vh,vh-vv,rvi,vv_std,vh_std] (6)
       S2: [10 raw, ndvi,ndwi,mndwi,ndmi,ndre,bsi,evi, cloud_mask, aux(A)]."""

    S1_OUT = 6
    S2_IDX_COUNT = 7
    S2_MASK_IDX = 17
    S2_BASE = 18

    def __init__(self, invalid_scl=(0, 1, 2, 3, 7, 8, 9, 10, 11), dilate=5,
                 aux_channels=6, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.invalid_scl = [int(c) for c in invalid_scl]
        self.dilate = int(dilate)
        self.aux_channels = int(aux_channels)
        self.eps = float(eps)

    def _ndx(self, a, b):
        return (a - b) / (a + b + self.eps)

    def call(self, inputs):
        s1, s2, scl, aux = inputs
        s1 = tf.cast(s1, tf.float32)
        s2 = tf.cast(s2, tf.float32)
        aux = tf.cast(aux, tf.float32)
        vv, vh, vv_std, vh_std = tf.split(s1, 4, axis=-1)
        lin_vv = tf.pow(10.0, vv / 10.0)
        lin_vh = tf.pow(10.0, vh / 10.0)
        rvi = 4.0 * lin_vh / (lin_vv + lin_vh + self.eps)
        s1_feat = tf.concat([vv, vh, vh - vv, rvi, vv_std, vh_std], axis=-1)
        b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12 = tf.split(s2, 10, axis=-1)
        ndvi = self._ndx(b8, b4)
        ndwi = self._ndx(b3, b8)
        mndwi = self._ndx(b3, b11)
        ndmi = self._ndx(b8, b11)
        ndre = self._ndx(b8, b5)
        bsi = self._ndx(b11 + b4, b8 + b2)
        evi = 2.5 * self._ndx(b8, b4 + 6.0 * b2 - 7.5 * b3 + self.eps)
        s2_feat = tf.concat([s2, ndvi, ndwi, mndwi, ndmi, ndre, bsi, evi], -1)
        scl_i = tf.cast(scl, tf.int32)
        mask = tf.zeros(tf.shape(scl_i), dtype=tf.bool)
        for c in self.invalid_scl:
            mask = tf.logical_or(mask, tf.equal(scl_i, c))
        mask_f = tf.cast(mask, tf.float32)
        if self.dilate > 1:
            mask_f = tf.nn.max_pool2d(mask_f, ksize=self.dilate, strides=1,
                                      padding="SAME")
            mask_f = tf.cast(mask_f > 0.0, tf.float32)
        s2_feat = s2_feat * (1.0 - mask_f)
        s2_feat = tf.concat([s2_feat, mask_f, aux], axis=-1)
        return s1_feat, s2_feat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"invalid_scl": self.invalid_scl, "dilate": self.dilate,
                    "aux_channels": self.aux_channels, "eps": self.eps})
        return cfg


C1 = FeatureEngineer.S1_OUT
C2 = lambda cfg: FeatureEngineer.S2_BASE + cfg.aux_channels


# ==============================================================================
# 6. NORMALIZATION STATS + CLASS ALPHA
# ==============================================================================
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v7.npz")


def compute_stats(scene_descs, cfg, max_patches=400):
    print("Computing normalization stats + class frequencies...")
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, cfg.aux_channels)
    C = cfg.num_classes
    c1, c2 = C1, C2(cfg)
    n = 0
    s1_sum = np.zeros(c1); s1_sq = np.zeros(c1)
    s2_sum = np.zeros(c2); s2_sq = np.zeros(c2)
    counts = np.zeros(C, np.int64)
    for patch in patch_generator(scene_descs, cfg, training=False,
                                 max_patches=max_patches):
        s1 = patch["s1_seq"][-1]; s2 = patch["s2_seq"][-1]
        scl = patch["scl_seq"][-1]; aux = patch["aux_seq"][-1]
        lab = patch["label"]
        f1, f2 = fe([s1[None], s2[None], scl[None], aux[None]])
        a = f1[0].numpy().reshape(-1, c1)
        b = f2[0].numpy().reshape(-1, c2)
        s1_sum += a.sum(0); s1_sq += (a ** 2).sum(0)
        s2_sum += b.sum(0); s2_sq += (b ** 2).sum(0)
        n += a.shape[0]
        v = lab[lab != cfg.ignore_label]
        counts += np.bincount(v, minlength=C)
    if n == 0:
        raise RuntimeError("compute_stats: no patches collected; check data dirs.")
    s1_mean = s1_sum / n
    s1_var = np.clip(s1_sq / n - s1_mean ** 2, 1e-6, None)
    s2_mean = s2_sum / n
    s2_var = np.clip(s2_sq / n - s2_mean ** 2, 1e-6, None)
    mask_idx = c2 - cfg.aux_channels - 1
    assert mask_idx == FeatureEngineer.S2_MASK_IDX
    s2_mean[mask_idx] = 0.0
    s2_var[mask_idx] = 1.0
    freq = counts / max(counts.sum(), 1)
    alpha = 1.0 / (freq + 1e-6)
    alpha = np.clip(alpha / max(alpha.mean(), 1e-9), 0.25, 4.0)
    np.savez(STATS_PATH, s1_mean=s1_mean, s1_var=s1_var, s2_mean=s2_mean,
             s2_var=s2_var, alpha=alpha, counts=counts,
             seg_temperature=np.float32(1.0))
    print(f"  class counts: {counts.tolist()}  alpha: {np.round(alpha, 3).tolist()}")
    return s1_mean, s1_var, s2_mean, s2_var, alpha


def load_or_compute_stats(scene_descs, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        print("Loaded cached norm stats.")
        if "seg_temperature" in z.files:
            cfg.seg_temperature = float(z["seg_temperature"])
        return z["s1_mean"], z["s1_var"], z["s2_mean"], z["s2_var"], z["alpha"]
    return compute_stats(scene_descs, cfg)


# ==============================================================================
# 7. tf.data PIPELINE (dict-based; photometric/noise/modality-dropout in TF,
#    geometric aug already done in numpy -> no channel-misalignment bug class)
# ==============================================================================
def output_signature(cfg):
    T, ps, A = cfg.temporal_steps, cfg.patch_size, cfg.aux_channels
    F = tf.float32
    return {
        "s1_seq": tf.TensorSpec((T, ps, ps, 4), F),
        "s2_seq": tf.TensorSpec((T, ps, ps, 10), F),
        "scl_seq": tf.TensorSpec((T, ps, ps, 1), F),
        "aux_seq": tf.TensorSpec((T, ps, ps, A), F),
        "dates": tf.TensorSpec((T,), F),
        "label": tf.TensorSpec((ps, ps), tf.int32),
        "carbon": tf.TensorSpec((ps, ps), F),
        "eco": tf.TensorSpec((ps, ps, 2), F),
        "change": tf.TensorSpec((ps, ps), tf.int32),
        "risk": tf.TensorSpec((ps, ps), F),
        "species": tf.TensorSpec((ps, ps), tf.int32),
        "debris": tf.TensorSpec((ps, ps), tf.int32),
        "recovery": tf.TensorSpec((ps, ps), tf.int32),
        "change_year": tf.TensorSpec((ps, ps), F),
    }


def make_dataset(gen_fn, cfg, training, batch_size):
    C, S = cfg.num_classes, cfg.num_species
    max_gap = float(cfg.max_time_gap_years)
    carbon_scale = float(cfg.carbon_scale)
    eco_scale = tf.constant(list(cfg.eco_scale), tf.float32)
    invalid = list(cfg.invalid_scl)
    deep = bool(cfg.deep_supervision)
    active = set(cfg.active_heads)

    def tf_aug(d):
        """Rank-agnostic photometric + noise + modality dropout (inputs only)."""
        delta = tf.random.uniform((), -cfg.aug_brightness, cfg.aug_brightness)
        contra = tf.random.uniform((), 1 - cfg.aug_contrast, 1 + cfg.aug_contrast)
        s2 = d["s2_seq"]
        mean = tf.reduce_mean(s2, axis=[1, 2, 3], keepdims=True)
        s2 = tf.clip_by_value((s2 - mean) * contra + mean + delta, -1.0, 2.0)
        s1 = d["s1_seq"] + tf.random.normal(tf.shape(d["s1_seq"]),
                                            stddev=cfg.aug_noise)
        s2 = s2 + tf.random.normal(tf.shape(s2), stddev=cfg.aug_noise)
        p = cfg.modality_dropout_p
        if p > 0:
            drop_opt = tf.random.uniform(()) < p
            drop_sar = tf.logical_and(tf.random.uniform(()) < p,
                                      tf.logical_not(drop_opt))
            s2 = tf.where(drop_opt, tf.zeros_like(s2), s2)
            s1 = tf.where(drop_sar, tf.zeros_like(s1), s1)
        d = dict(d)
        d["s1_seq"], d["s2_seq"] = s1, s2
        return d

    def prepare(d):
        inputs = {"s1_seq": d["s1_seq"], "s2_seq": d["s2_seq"],
                  "scl_seq": d["scl_seq"], "aux_seq": d["aux_seq"],
                  "dates": d["dates"]}
        targets = {}
        lab = d["label"]
        seg = tf.one_hot(tf.where(lab == cfg.ignore_label, -1, lab), C,
                         dtype=tf.float32)
        targets["seg_head"] = seg
        if deep:  # F12: explicit batch rank for pooling
            targets["seg_deep_head"] = tf.nn.avg_pool2d(seg[None], 4, 4,
                                                        "VALID")[0]
        # quality: usable pixel from last-step SCL (free supervision)
        scl_last = tf.cast(d["scl_seq"][-1], tf.int32)
        bad = tf.zeros(tf.shape(scl_last), tf.bool)
        for c in invalid:
            bad = tf.logical_or(bad, tf.equal(scl_last, c))
        targets["quality_head"] = 1.0 - tf.cast(bad, tf.float32)
        # boundary: morphological gradient of seg one-hot (F12 fix)
        valid_seg = tf.cast(tf.reduce_sum(seg, -1, keepdims=True) > 0,
                            tf.float32)
        mx = tf.nn.max_pool2d(seg[None], 3, 1, "SAME")[0]
        mn = -tf.nn.max_pool2d(-seg[None], 3, 1, "SAME")[0]
        edge = tf.reduce_max(tf.clip_by_value(mx - mn, 0.0, 1.0), -1,
                             keepdims=True)
        targets["boundary_head"] = tf.where(valid_seg > 0, edge, -1.0)
        if "species" in active:
            sp = d["species"]
            targets["species_head"] = tf.one_hot(
                tf.where(sp == 255, -1, sp), S, dtype=tf.float32)

        def _bin(key):
            v = tf.cast(d[key], tf.float32)
            return tf.where(v == 255.0, -1.0,
                            tf.clip_by_value(v, 0.0, 1.0))[..., None]

        if "change" in active:
            targets["change_head"] = _bin("change")
        if "recovery" in active:
            targets["recovery_head"] = _bin("recovery")
        if "damage" in active:
            targets["damage_head"] = _bin("debris")
        if "change_year" in active:
            cy = tf.cast(d["change_year"], tf.float32)
            rel = (cy - d["dates"][0]) / max_gap
            targets["change_year_head"] = tf.where(
                cy > 0.0, tf.clip_by_value(rel, 0.0, 1.0), -999.0)[..., None]
        if "carbon" in active:
            carb = tf.cast(d["carbon"], tf.float32)[..., None]
            targets["carbon_head"] = tf.where(carb > -900.0,
                                              carb / carbon_scale, -999.0)
        if "eco" in active:
            eco = tf.cast(d["eco"], tf.float32)
            targets["eco_head"] = tf.where(eco > -900.0, eco / eco_scale,
                                           -999.0)
        if "risk" in active:
            r = tf.cast(d["risk"], tf.float32)[..., None]
            targets["risk_head"] = tf.where(r > -900.0, r, -999.0)
        return inputs, targets

    ds = tf.data.Dataset.from_generator(gen_fn,
                                        output_signature=output_signature(cfg))
    if training:
        ds = ds.shuffle(cfg.shuffle_buffer, seed=42,
                        reshuffle_each_iteration=True)
        if cfg.augment:
            ds = ds.map(tf_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(cfg.deterministic)
    ds = ds.with_options(opts)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 8. SWIN-V2 COMPONENTS (buffers are non-trainable weights; Keras-3 safe)
# ==============================================================================
def window_partition(x, ws):
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = x.shape[-1]
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, ws * ws, C])


def window_reverse(windows, ws, H, W):
    C = windows.shape[-1]
    B = tf.shape(windows)[0] // ((H // ws) * (W // ws))
    x = tf.reshape(windows, [B, H // ws, W // ws, ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [B, H, W, C])


@tf.keras.utils.register_keras_serializable(package="mangrove")
class DropPath(layers.Layer):
    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, x, training=None):
        if self.rate == 0.0 or training is not True:
            return x
        keep = 1.0 - self.rate
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        mask = tf.cast(tf.random.uniform(shape) < keep, x.dtype)
        return x * mask / keep

    def get_config(self):
        cfg = super().get_config()
        cfg["rate"] = self.rate
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class WindowAttentionV2(layers.Layer):
    """Scaled-cosine attention + clamped learnable logit scale + log-spaced
    Continuous Position Bias MLP. All buffers stored as non-trainable weights."""

    def __init__(self, dim, window_size, num_heads, attn_drop=0.0,
                 proj_drop=0.0, cpb_hidden=256, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.attn_drop_rate = float(attn_drop)
        self.proj_drop_rate = float(proj_drop)
        self.cpb_hidden = int(cpb_hidden)

    def build(self, input_shape):
        ws = self.window_size
        n = ws * ws
        self.qkv = layers.Dense(self.dim * 3, use_bias=True, name="qkv")
        self.proj = layers.Dense(self.dim, name="proj")
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        self.logit_scale = self.add_weight(
            name="logit_scale", shape=(self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(math.log(10.0)),
            trainable=True)
        self.cpb1 = layers.Dense(self.cpb_hidden, activation="relu", name="cpb1")
        self.cpb2 = layers.Dense(self.num_heads, use_bias=False, name="cpb2")
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws),
                                      indexing="ij"))
        cf = coords.reshape(2, -1).astype(np.float32)
        rel = cf[:, :, None] - cf[:, None, :]
        rel = rel.transpose(1, 2, 0) / max(ws - 1, 1) * 8.0
        rel = np.sign(rel) * np.log2(np.abs(rel) + 1.0) / np.log2(8.0)
        self.rel_coords = self.add_weight(
            name="rel_coords", shape=(n, n, 2), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(rel.astype(np.float32)),
            trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        Bn = tf.shape(x)[0]
        N = self.window_size * self.window_size
        qkv = tf.reshape(self.qkv(x), [Bn, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        qn = tf.math.l2_normalize(tf.cast(q, tf.float32), axis=-1)
        kn = tf.math.l2_normalize(tf.cast(k, tf.float32), axis=-1)
        scale = tf.exp(tf.minimum(self.logit_scale, math.log(100.0)))
        attn = tf.matmul(qn, kn, transpose_b=True) * tf.cast(scale, tf.float32)
        bias = 16.0 * tf.sigmoid(self.cpb2(self.cpb1(self.rel_coords)))
        attn += tf.transpose(bias, [2, 0, 1])[None]
        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) + mask[None]
            attn = tf.reshape(attn, [Bn, self.num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(tf.cast(attn, v.dtype), training=training)
        out = tf.matmul(attn, v)
        out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [Bn, N, self.dim])
        return self.proj_drop(self.proj(out), training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "window_size": self.window_size,
                    "num_heads": self.num_heads,
                    "attn_drop": self.attn_drop_rate,
                    "proj_drop": self.proj_drop_rate,
                    "cpb_hidden": self.cpb_hidden})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class SwinV2Block(layers.Layer):
    """Res-post-norm: x = norm1(x + dp(attn(x))); x = norm2(x + dp(mlp(x)))."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)
        self.drop = float(drop)
        self.attn_drop = float(attn_drop)
        self.drop_path_rate = float(drop_path)

    def build(self, input_shape):
        _, H, W, C = input_shape
        assert C == self.dim, f"SwinV2Block dim {self.dim} != input C {C}"
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"Feature map {H}x{W} not divisible by window {self.window_size}"
        self.H, self.W = int(H), int(W)
        self.attn = WindowAttentionV2(self.dim, self.window_size, self.num_heads,
                                      attn_drop=self.attn_drop,
                                      proj_drop=self.drop, name="wmsa")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.dp1 = DropPath(self.drop_path_rate)
        self.dp2 = DropPath(self.drop_path_rate)
        hidden = int(self.dim * self.mlp_ratio)
        self.mlp = models.Sequential([
            layers.Dense(hidden, activation="gelu"), layers.Dropout(self.drop),
            layers.Dense(self.dim), layers.Dropout(self.drop)], name="mlp")
        if self.shift_size > 0:
            mask_np = self._build_mask_np()
            self.attn_mask = self.add_weight(
                name="attn_mask", shape=mask_np.shape, dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(mask_np),
                trainable=False)
        else:
            self.attn_mask = None
        super().build(input_shape)

    def _build_mask_np(self):
        ws, ss, H, W = self.window_size, self.shift_size, self.H, self.W
        img_mask = np.zeros((H, W), np.float32)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[hs, ws_] = cnt
                cnt += 1
        m = img_mask.reshape(H // ws, ws, W // ws, ws) \
                    .transpose(0, 2, 1, 3).reshape(-1, ws * ws)
        attn_mask = m[:, :, None] - m[:, None, :]
        attn_mask = np.where(attn_mask != 0.0, -100.0, 0.0).astype(np.float32)
        return attn_mask[:, None, :, :]

    def call(self, x, training=None):
        h = x
        if self.shift_size > 0:
            h = tf.roll(h, shift=[-self.shift_size, -self.shift_size],
                        axis=[1, 2])
        w = window_partition(h, self.window_size)
        w = self.attn(w, mask=self.attn_mask, training=training)
        h = window_reverse(w, self.window_size, self.H, self.W)
        if self.shift_size > 0:
            h = tf.roll(h, shift=[self.shift_size, self.shift_size],
                        axis=[1, 2])
        x = self.norm1(x + self.dp1(h, training=training))
        x = self.norm2(x + self.dp2(self.mlp(x, training=training),
                                    training=training))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads,
                    "window_size": self.window_size,
                    "shift_size": self.shift_size,
                    "mlp_ratio": self.mlp_ratio, "drop": self.drop,
                    "attn_drop": self.attn_drop,
                    "drop_path": self.drop_path_rate})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class PatchMerging(layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = int(out_dim)

    def build(self, input_shape):
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.reduction = layers.Dense(self.out_dim, use_bias=False,
                                      name="reduction")
        super().build(input_shape)

    def call(self, x):
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(tf.concat([x0, x1, x2, x3], axis=-1)))

    def get_config(self):
        cfg = super().get_config()
        cfg["out_dim"] = self.out_dim
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class AbsDiff(layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)


# ==============================================================================
# 9. FUSION + TEMPORAL CONDITIONING
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class GatedFusion(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)

    def build(self, input_shape):
        self.mix = layers.Conv2D(self.dim, 1, name="mix")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="ln")
        self.gate = self.add_weight(name="gate", shape=(1, 1, 1, self.dim),
                                    initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        f_opt, f_sar = inputs
        m = self.mix(tf.concat([f_opt, f_sar], axis=-1))
        return self.norm(f_opt + tf.cast(tf.tanh(self.gate), f_opt.dtype) * m)

    def get_config(self):
        cfg = super().get_config()
        cfg["dim"] = self.dim
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class CrossAttentionFusion(layers.Layer):
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)

    def build(self, input_shape):
        H, W = input_shape[0][1], input_shape[0][2]
        self.H, self.W = int(H), int(W)
        self.pos = self.add_weight(
            name="pos_embed", shape=(1, self.H * self.W, self.dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.mha_s2o = layers.MultiHeadAttention(
            self.num_heads, self.dim // self.num_heads, name="sar_queries_optical")
        self.mha_o2s = layers.MultiHeadAttention(
            self.num_heads, self.dim // self.num_heads, name="optical_queries_sar")
        self.ln_s = layers.LayerNormalization(epsilon=1e-5, name="ln_sar")
        self.ln_o = layers.LayerNormalization(epsilon=1e-5, name="ln_opt")
        self.proj = layers.Conv2D(self.dim, 1, name="fuse_proj")
        self.ln_out = layers.LayerNormalization(epsilon=1e-5, name="ln_out")
        super().build(input_shape)

    def call(self, inputs, training=None):
        f_sar, f_opt = inputs
        B = tf.shape(f_sar)[0]
        N = self.H * self.W
        t_sar = tf.reshape(f_sar, [B, N, self.dim]) + tf.cast(self.pos,
                                                              f_sar.dtype)
        t_opt = tf.reshape(f_opt, [B, N, self.dim]) + tf.cast(self.pos,
                                                              f_opt.dtype)
        s = t_sar + self.mha_s2o(query=t_sar, value=t_opt, training=training)
        o = t_opt + self.mha_o2s(query=t_opt, value=t_sar, training=training)
        s = self.ln_s(tf.reshape(s, [B, self.H, self.W, self.dim]))
        o = self.ln_o(tf.reshape(o, [B, self.H, self.W, self.dim]))
        return self.ln_out(self.proj(tf.concat([s, o], axis=-1)) + f_opt)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class TimeEmbedding(layers.Layer):
    """Rank-agnostic sinusoidal time embedding: (...,) -> (..., dim)."""

    def __init__(self, dim, freqs=16, max_years=20.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.freqs = int(freqs)
        self.max_years = float(max_years)

    def build(self, input_shape):
        f = np.arange(self.freqs, dtype=np.float32) / self.freqs
        self.freq_table = self.add_weight(
            name="freq_table", shape=(self.freqs,), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1.0 / (10000.0 ** f)),
            trainable=False)
        self.d1 = layers.Dense(self.dim, activation="gelu", name="t1")
        self.d2 = layers.Dense(self.dim, name="t2")
        super().build(input_shape)

    def call(self, dt):
        dt = tf.cast(dt, tf.float32)[..., None] / self.max_years
        ang = dt * self.freq_table * 2.0 * math.pi
        emb = tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)
        return self.d2(self.d1(emb))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "freqs": self.freqs,
                    "max_years": self.max_years})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class FiLM(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)

    def build(self, input_shape):
        self.dense = layers.Dense(2 * self.channels, name="film")
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        gb = self.dense(cond)
        g, b = tf.split(gb, 2, axis=-1)
        g = tf.reshape(g, [-1, 1, 1, self.channels])
        b = tf.reshape(b, [-1, 1, 1, self.channels])
        return x * (1.0 + tf.cast(g, x.dtype)) + tf.cast(b, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg["channels"] = self.channels
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class TemporalTrajectoryEncoder(layers.Layer):
    """Per-location temporal transformer over the bottleneck sequence.
    (B,T,H,W,D) tokens + (B,T) dates -> trajectory map (B,H,W,D) + pooled (B,D).
    Attention runs over TIME per spatial location: cost is O(N*T^2), tiny."""

    def __init__(self, dim, steps, num_heads=8, num_blocks=2, mlp_ratio=2.0,
                 dropout=0.1, max_years=20.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.steps = int(steps)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.max_years = float(max_years)

    def build(self, input_shape):
        bshape = input_shape[0]
        self.H, self.W = int(bshape[2]), int(bshape[3])
        self.N = self.H * self.W
        self.pos = self.add_weight(
            name="temporal_pos", shape=(1, self.steps, self.dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.date_emb = TimeEmbedding(self.dim, max_years=self.max_years,
                                      name="date_emb")
        self.blocks = []
        for i in range(self.num_blocks):
            self.blocks.append({
                "ln1": layers.LayerNormalization(epsilon=1e-5, name=f"tb{i}_ln1"),
                "mha": layers.MultiHeadAttention(
                    self.num_heads, max(8, self.dim // self.num_heads),
                    dropout=self.dropout, name=f"tb{i}_mha"),
                "ln2": layers.LayerNormalization(epsilon=1e-5, name=f"tb{i}_ln2"),
                "mlp": models.Sequential([
                    layers.Dense(int(self.dim * self.mlp_ratio),
                                 activation="gelu"),
                    layers.Dropout(self.dropout),
                    layers.Dense(self.dim)], name=f"tb{i}_mlp"),
                "drop": layers.Dropout(self.dropout),
            })
        self.out_norm = layers.LayerNormalization(epsilon=1e-5, name="traj_norm")
        super().build(input_shape)

    def call(self, inputs, training=None):
        tokens, dates = inputs
        B = tf.shape(tokens)[0]
        x = tf.reshape(tokens, [B, self.steps, self.N, self.dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B * self.N, self.steps, self.dim])
        x = x + tf.cast(self.pos, x.dtype)
        de = self.date_emb(dates)                       # (B,T,dim)
        de = tf.repeat(de, self.N, axis=0)              # matches B*N ordering
        x = x + tf.cast(de, x.dtype)
        for blk in self.blocks:
            h = blk["ln1"](x)
            x = x + blk["drop"](blk["mha"](query=h, value=h, key=h,
                                           training=training),
                                training=training)
            h = blk["ln2"](x)
            x = x + blk["drop"](blk["mlp"](h, training=training),
                                training=training)
        x = self.out_norm(x)
        last = x[:, -1, :]
        traj = tf.reshape(last, [B, self.H, self.W, self.dim])
        pooled = tf.reduce_mean(traj, axis=[1, 2])
        return traj, pooled

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "steps": self.steps,
                    "num_heads": self.num_heads, "num_blocks": self.num_blocks,
                    "mlp_ratio": self.mlp_ratio, "dropout": self.dropout,
                    "max_years": self.max_years})
        return cfg


# ==============================================================================
# 10. ENCODERS / DECODER BLOCKS
# ==============================================================================
def _heads_for(dim):
    return max(2, min(12, dim // 16))


def make_swin_encoder(prefix, dims, depths, cfg, dp_rates):
    """All layers built ONCE; the returned closure is reused for every time
    step (true weight sharing; fixes v6 F14)."""
    ws = cfg.window_size
    stem = models.Sequential([
        layers.Conv2D(dims[0] // 2, 3, strides=2, padding="same"),
        layers.BatchNormalization(), layers.Activation("gelu"),
        layers.Conv2D(dims[0], 3, strides=2, padding="same"),
        layers.BatchNormalization(), layers.Activation("gelu"),
        layers.Conv2D(dims[0], 3, padding="same"),
    ], name=f"{prefix}_stem")
    stem_norm = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_stem_norm")
    stages, mergers = [], []
    k = 0
    for i, (d, dep) in enumerate(zip(dims, depths)):
        blocks = []
        for j in range(dep):
            shift = 0 if j % 2 == 0 else ws // 2
            blocks.append(SwinV2Block(d, _heads_for(d), ws, shift,
                                      mlp_ratio=cfg.mlp_ratio,
                                      drop_path=dp_rates[k],
                                      name=f"{prefix}_stage{i}_blk{j}"))
            k += 1
        stages.append(blocks)
        if i < len(dims) - 1:
            mergers.append(PatchMerging(dims[i + 1], name=f"{prefix}_merge{i}"))
    assert k == len(dp_rates)

    def apply(x, training=None):
        x = stem_norm(stem(x))
        skips = []
        for i, blocks in enumerate(stages):
            for blk in blocks:
                x = blk(x, training=training)
            skips.append(x)
            if i < len(mergers):
                x = mergers[i](x)
        return x, skips
    return apply


def se_block(x, ratio=8, name=None):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    s = layers.Dense(max(c // ratio, 8), activation="relu", name=f"{name}_fc1")(s)
    s = layers.Dense(c, activation="sigmoid", name=f"{name}_fc2")(s)
    s = layers.Reshape((1, 1, c), name=f"{name}_rs")(s)
    return layers.multiply([x, s], name=f"{name}_scale")


def conv_block(x, filters, dropout=0.1, use_se=True, name=None):
    shortcut = layers.Conv2D(filters, 1, padding="same", name=f"{name}_proj")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("gelu", name=f"{name}_act1")(x)
    x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.add([x, shortcut], name=f"{name}_add")
    x = layers.Activation("gelu", name=f"{name}_out")(x)
    return se_block(x, name=f"{name}_se") if use_se else x


# ==============================================================================
# 11. FULL V7 MODEL
# ==============================================================================
def build_model(cfg, s1_stats, s2_stats):
    ops = tf.keras.ops
    C, S, ps, A, T = (cfg.num_classes, cfg.num_species, cfg.patch_size,
                      cfg.aux_channels, cfg.temporal_steps)
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)
    ws = cfg.window_size
    active = set(cfg.active_heads)
    L = len(dims)

    s1_seq = layers.Input((T, ps, ps, 4), name="s1_seq")
    s2_seq = layers.Input((T, ps, ps, 10), name="s2_seq")
    scl_seq = layers.Input((T, ps, ps, 1), name="scl_seq")
    aux_seq = layers.Input((T, ps, ps, A), name="aux_seq")
    dates_in = layers.Input((T,), name="dates")

    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    n1 = layers.Normalization(mean=list(s1_stats[0]),
                              variance=list(s1_stats[1]), name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]),
                              variance=list(s2_stats[1]), name="s2_norm")

    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])

    # per-level SAR projection + fusion, built ONCE and shared across time
    sar_proj = [layers.Conv2D(d, 1, name=f"sar_proj{i}")
                for i, d in enumerate(dims)]
    fusions = [(CrossAttentionFusion(d, cfg.fusion_heads, name=f"fusion{i}")
                if i in tuple(cfg.cross_attn_levels)
                else GatedFusion(d, name=f"fusion{i}"))
               for i, d in enumerate(dims)]

    step_bottlenecks = []
    first_fused = first_sproj = ref_fused = ref_sproj = ref_sk1 = None
    for t in range(T):
        f1, f2 = fe([s1_seq[:, t], s2_seq[:, t], scl_seq[:, t], aux_seq[:, t]])
        b1, sk1 = enc_s1(n1(f1))
        b2, sk2 = enc_s2(n2(f2))
        sproj = [sar_proj[i](sk1[i]) for i in range(L)]
        fused = [fusions[i]([sk2[i], sproj[i]]) for i in range(L)]
        step_bottlenecks.append(fused[-1])
        if t == 0:
            first_fused, first_sproj = fused, sproj
        if t == T - 1:
            ref_fused, ref_sproj, ref_sk1 = fused, sproj, sk1

    # temporal trajectory at the bottleneck (cheap: N=(ps/32)^2 locations)
    bn_seq = ops.stack(step_bottlenecks, axis=1)        # (B,T,h,w,D)
    traj, pooled = TemporalTrajectoryEncoder(
        dims[-1], T, num_heads=cfg.temporal_heads,
        num_blocks=cfg.temporal_blocks, dropout=cfg.temporal_dropout,
        max_years=cfg.max_time_gap_years,
        name="temporal_encoder")([bn_seq, dates_in])

    # global scene context + trajectory injection into the reference bottleneck
    gap = layers.GlobalAveragePooling2D(name="ctx_gap")(ref_fused[-1])
    ctx = layers.Dense(dims[-1], activation="gelu", name="ctx_fc1")(gap)
    ctx = layers.Dense(dims[-1], name="ctx_fc2")(ctx + pooled)
    bn = layers.Conv2D(dims[-1], 1, name="traj_proj")(traj)
    bn = layers.add([ref_fused[-1], bn], name="bn_add")
    bn = layers.LayerNormalization(epsilon=1e-5, name="bn_norm")(bn)
    bn = FiLM(dims[-1], name="bn_film")([bn, ctx])

    # ---- main decoder on the reference (last) step: full skip pathway (F15)
    x = bn
    deep_out = None
    for i, wd in enumerate(cfg.dec_widths):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"dec_up{i}")(x)
        li = L - 2 - i
        if li >= 0:
            x = layers.concatenate([x, ref_fused[li], ref_sk1[li]],
                                   name=f"dec_cat{i}")
        x = conv_block(x, wd, use_se=cfg.use_se, name=f"dec_c{i}")
        if i < cfg.decoder_swin_levels:
            shift = 0 if i % 2 == 0 else ws // 2
            x = SwinV2Block(wd, _heads_for(wd), ws, shift,
                            name=f"dec_swin{i}")(x)
        if cfg.deep_supervision and i == len(cfg.dec_widths) - 1:
            deep_out = layers.Conv2D(C, 1, padding="same", dtype="float32",
                                     name="seg_deep_head")(x)
    for j, wd in enumerate(cfg.refine_widths):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"refine_up{j}")(x)
        x = conv_block(x, wd, use_se=cfg.use_se, name=f"refine_c{j}")

    # ---- change decoder: first-vs-last diffs + FiLM(dt) + trajectory (F11)
    dt = dates_in[:, -1:] - dates_in[:, :1]
    temb = TimeEmbedding(cfg.time_embed_dim, name="time_embed")(dt)
    diffs = [layers.concatenate(
        [AbsDiff(name=f"diff_opt{i}")([ref_fused[i], first_fused[i]]),
         AbsDiff(name=f"diff_sar{i}")([ref_sproj[i], first_sproj[i]])],
        name=f"diff_cat{i}") for i in range(L)]
    cw = list(cfg.change_widths)
    d = conv_block(diffs[-1], cw[0], use_se=cfg.use_se, name="change_bottleneck")
    d = FiLM(cw[0], name="film_bottleneck")([d, temb])
    d = FiLM(cw[0], name="film_traj")([d, ctx])
    size = ps // 32
    i = 0
    while size < ps:                                    # F11: always reach ps
        wd = cw[min(i + 1, len(cw) - 1)]
        d = layers.UpSampling2D(interpolation="bilinear",
                                name=f"change_up{i}")(d)
        size *= 2
        li = L - 2 - i
        if li >= 0:
            d = layers.concatenate([d, diffs[li]], name=f"change_cat{i}")
        d = conv_block(d, wd, use_se=cfg.use_se, name=f"change_c{i}")
        d = FiLM(wd, name=f"film_{i}")([d, temb])
        i += 1

    # ---- heads (pluggable) ----
    outputs = {}
    if "seg" in active:
        outputs["seg_head"] = layers.Activation(
            "softmax", dtype="float32", name="seg_head")(
            layers.Conv2D(C, 1, padding="same", name="seg_logits")(x))
    if cfg.deep_supervision and deep_out is not None:
        outputs["seg_deep_head"] = deep_out
    if "species" in active:
        outputs["species_head"] = layers.Activation(
            "softmax", dtype="float32", name="species_head")(
            layers.Conv2D(S, 1, padding="same", name="species_logits")(x))
    if "carbon" in active:
        outputs["carbon_head"] = layers.Conv2D(
            2, 1, padding="same", dtype="float32", name="carbon_head")(x)
    if "eco" in active:
        outputs["eco_head"] = layers.Conv2D(
            2, 1, padding="same", dtype="float32", name="eco_head")(x)
    if "risk" in active:
        outputs["risk_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="risk_head")(
            layers.Conv2D(1, 1, padding="same", name="risk_logits")(x))
    if "quality" in active:
        outputs["quality_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="quality_head")(
            layers.Conv2D(1, 1, padding="same", name="quality_logits")(x))
    if "boundary" in active:
        outputs["boundary_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="boundary_head")(
            layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x))
    if "change" in active:
        outputs["change_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="change_head")(
            layers.Conv2D(1, 1, padding="same", name="change_logits")(d))
    if "change_year" in active:
        outputs["change_year_head"] = layers.Conv2D(
            1, 1, padding="same", dtype="float32", name="change_year_head")(d)
    if "recovery" in active:
        outputs["recovery_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="recovery_head")(
            layers.Conv2D(1, 1, padding="same", name="recovery_logits")(d))
    if "damage" in active:
        outputs["damage_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="damage_head")(
            layers.Conv2D(1, 1, padding="same", name="damage_logits")(d))

    return models.Model(inputs=[s1_seq, s2_seq, scl_seq, aux_seq, dates_in],
                        outputs=outputs, name=f"MangroveEcoMonitor_v{VERSION}")


# ==============================================================================
# 12b. METRICS (completion)  --  ChangeF1 + CarbonMAE
# NOTE: this replaces the truncated ChangeF1 stub at the end of part 1.
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class ChangeF1(metrics.Metric):
    """Masked F1 for binary heads (change / recovery / damage). -1 = ignore."""
    def __init__(self, name="change_f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        p_bin = tf.cast(p > self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y * p_bin * mask))
        self.fp.assign_add(tf.reduce_sum((1.0 - y) * p_bin * mask))
        self.fn.assign_add(tf.reduce_sum(y * (1.0 - p_bin) * mask))

    def result(self):
        prec = self.tp / (self.tp + self.fp + 1e-6)
        rec = self.tp / (self.tp + self.fn + 1e-6)
        return 2.0 * prec * rec / (prec + rec + 1e-6)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["threshold"] = self.threshold
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonMAE(metrics.Metric):
    def __init__(self, name="carbon_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros",
                                     dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros",
                                     dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.cast(y_true, tf.float32)
        mean = tf.cast(y_pred[..., 0:1], tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        self.total.assign_add(
            tf.reduce_sum(tf.abs(tf.where(mask > 0, y, 0.0) - mean) * mask))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + 1e-6)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# ==============================================================================
# 13. TRAINER: custom train_step, learnable Kendall task weighting,
#     single (native) gradient path, explicit metric names (F21 fixed)
# ==============================================================================
HEAD_LOSS_MAP = {
    "seg": "focal_dice", "species": "focal_dice",
    "change": "binary_focal", "recovery": "binary_focal", "damage": "binary_focal",
    "quality": "bce", "boundary": "bce",
    "change_year": "masked_mae", "carbon": "carbon_nll",
    "eco": "huber", "risk": "huber",
}


class EcoMonitorTrainer(tf.keras.Model):
    """Wraps the functional net. Owns per-task losses, learnable log-variance
    task weighting, and explicit metrics. Gradient accumulation is the NATIVE
    optimizer path only (no second custom accumulator)."""

    def __init__(self, cfg, net, alpha, **kwargs):
        super().__init__(**kwargs)
        self.cfg_obj = cfg
        self.net = net
        self.active = list(cfg.active_heads)
        self.loss_tracker = metrics.Mean(name="loss")

        self.loss_fns = {}
        for h in self.active:
            kind = HEAD_LOSS_MAP[h]
            if kind == "focal_dice":
                self.loss_fns[h] = FocalDiceLoss(
                    alpha=alpha if h == "seg" else None,   # species alpha unknown
                    gamma=cfg.focal_gamma,
                    label_smoothing=cfg.label_smoothing)
            elif kind == "binary_focal":
                self.loss_fns[h] = MaskedBinaryFocal()
            elif kind == "bce":
                self.loss_fns[h] = MaskedBCE()
            elif kind == "masked_mae":
                self.loss_fns[h] = MaskedMAELoss()
            elif kind == "carbon_nll":
                self.loss_fns[h] = CarbonNLLLoss()
            else:
                self.loss_fns[h] = MaskedHuber()
        if cfg.deep_supervision and "seg" in self.active:
            self.loss_fns["seg_deep"] = FocalDiceLoss(
                alpha=alpha, gamma=cfg.focal_gamma,
                label_smoothing=cfg.label_smoothing)

        self.use_learned_w = cfg.task_weighting == "learned"
        self.task_log_vars = {}
        for name in self.loss_fns.keys():
            self.task_log_vars[name] = self.add_weight(
                name=f"task_logvar_{name}", shape=(), dtype=tf.float32,
                initializer="zeros", trainable=self.use_learned_w)

        self.per_head_loss = {n: metrics.Mean(name=f"{n}_loss")
                              for n in self.loss_fns}
        self.metric_objs = {}
        if "seg" in self.active:
            self.metric_objs["seg"] = MeanIoU(cfg.num_classes, name="seg_miou")
        if "species" in self.active:
            self.metric_objs["species"] = MeanIoU(cfg.num_species,
                                                  name="species_miou")
        for h in ("change", "recovery", "damage"):
            if h in self.active:
                self.metric_objs[h] = ChangeF1(name=f"{h}_f1")
        if "carbon" in self.active:
            self.metric_objs["carbon"] = CarbonMAE()
        for h in ("quality", "boundary"):
            if h in self.active:
                self.metric_objs[h] = metrics.Mean(name=f"{h}_acc")

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    @property
    def metrics(self):
        return ([self.loss_tracker] + list(self.per_head_loss.values()) +
                list(self.metric_objs.values()))

    def _per_task_losses(self, y, p):
        out = {}
        for name, fn in self.loss_fns.items():
            key = "seg_deep_head" if name == "seg_deep" else f"{name}_head"
            tgt = y.get(key)
            prd = p.get(key) if isinstance(p, dict) else None
            if tgt is not None and prd is not None:
                out[name] = fn(tgt, prd)
        return out

    def _total(self, per_task):
        total = 0.0
        for name, l in per_task.items():
            if self.use_learned_w:
                lv = self.task_log_vars[name]
                w_extra = (self.cfg_obj.deep_supervision_weight
                           if name == "seg_deep" else 1.0)
                total = total + w_extra * (l * tf.exp(-lv) + lv)
            else:
                if name == "seg_deep":
                    w = self.cfg_obj.deep_supervision_weight
                else:
                    w = self.cfg_obj.fixed_loss_weights.get(name, 1.0)
                total = total + w * l
        return total

    def _update_metrics(self, per_task, y, p, total):
        self.loss_tracker.update_state(total)
        for name, l in per_task.items():
            self.per_head_loss[name].update_state(l)
        for h, m in self.metric_objs.items():
            key = f"{h}_head"
            if key not in y or key not in p:
                continue
            if h in ("quality", "boundary"):
                valid = tf.cast(y[key] >= 0, tf.float32)
                correct = tf.cast(tf.equal(
                    tf.cast(p[key] > 0.5, tf.float32),
                    tf.clip_by_value(y[key], 0.0, 1.0)), tf.float32)
                m.update_state(tf.reduce_sum(correct * valid) /
                               (tf.reduce_sum(valid) + 1e-6))
            else:
                m.update_state(y[key], p[key])

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            p = self.net(x, training=True)
            per_task = self._per_task_losses(y, p)
            total = self._total(per_task) + sum(self.net.losses)
        scale_fn = getattr(self.optimizer, "scale_loss", None)
        scaled = scale_fn(total) if callable(scale_fn) else total
        grads = tape.gradient(scaled, self.trainable_variables)
        grads = [g if g is not None else tf.zeros_like(v)
                 for g, v in zip(grads, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._update_metrics(per_task, y, p, total)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        p = self.net(x, training=False)
        per_task = self._per_task_losses(y, p)
        total = self._total(per_task)
        self._update_metrics(per_task, y, p, total)
        return {m.name: m.result() for m in self.metrics}


def make_optimizer(cfg, lr):
    """Single accumulation mechanism (native Keras path or hard error) +
    explicit mixed-precision loss scaling (F22)."""
    kwargs = dict(learning_rate=lr, weight_decay=cfg.weight_decay,
                  clipnorm=cfg.grad_clip)
    if cfg.grad_accum > 1:
        try:
            opt = optimizers.AdamW(
                gradient_accumulation_steps=int(cfg.grad_accum), **kwargs)
        except TypeError as e:
            raise RuntimeError(
                "cfg.grad_accum > 1 requires Keras>=3.4 native "
                "gradient_accumulation_steps. Upgrade TF/Keras or set "
                "grad_accum=1 (v7 never runs a second custom accumulator)."
            ) from e
    else:
        opt = optimizers.AdamW(**kwargs)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        try:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        except Exception:
            pass            # trainer's scale_loss getattr guard covers this
    return opt


# ==============================================================================
# 14. EMA CALLBACK (Polyak averaging; val + export run on EMA weights)
# ==============================================================================
class EMACallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.999, update_every=1):
        super().__init__()
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.shadow = None
        self._stash = None
        self._step = 0

    def on_train_begin(self, logs=None):
        self.shadow = [tf.Variable(w, trainable=False)
                       for w in self.model.trainable_weights]

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % self.update_every:
            return
        d = self.decay
        for s, w in zip(self.shadow, self.model.trainable_weights):
            s.assign(d * s + (1.0 - d) * tf.cast(w, s.dtype))

    def swap_in(self):
        self._stash = [tf.identity(w) for w in self.model.trainable_weights]
        for w, s in zip(self.model.trainable_weights, self.shadow):
            w.assign(tf.cast(s, w.dtype))

    def swap_out(self):
        for w, old in zip(self.model.trainable_weights, self._stash):
            w.assign(old)
        self._stash = None

    def on_test_begin(self, logs=None):
        if self.shadow is not None:
            self.swap_in()

    def on_test_end(self, logs=None):
        if self._stash is not None:
            self.swap_out()


# ==============================================================================
# 15. LR SCHEDULE / PLOTS / EVAL / CALIBRATION (temperature + conformal)
# ==============================================================================
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_steps, alpha=0.01):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.alpha = float(alpha)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.initial_lr * step / warm
        decay_steps = tf.cast(max(self.total_steps - self.warmup_steps, 1),
                              tf.float32)
        decay_step = tf.maximum(step - warm, 0.0)
        cosine = 0.5 * (1.0 + tf.cos(math.pi * tf.minimum(decay_step /
                                                          decay_steps, 1.0)))
        decayed_lr = self.initial_lr * ((1.0 - self.alpha) * cosine + self.alpha)
        return tf.cond(step < warm, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self):
        return {"initial_lr": self.initial_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "alpha": self.alpha}


CLASS_NAMES = ["Bare Earth/Agri", "Tidal Water", "Perm Flood",
               "Deg. Mangrove", "Dense Mangrove"]
CLASS_COLORS = ["#8B4513", "#00BFFF", "#191970", "#90EE90", "#006400"]


def _preds_to_dict(model, preds):
    if isinstance(preds, dict):
        return preds
    return dict(zip(model.output_names, preds))


def evaluate_per_class(net, val_ds, cfg):
    if "seg_head" not in net.output_names:
        print("  [eval] seg head not present; skipping per-class eval.")
        return None, None
    C = cfg.num_classes
    cm = np.zeros((C, C), np.int64)
    for inputs, targets in val_ds:
        out = _preds_to_dict(net, net(inputs, training=False))
        p = out["seg_head"].numpy().argmax(-1)
        y = targets["seg_head"].numpy()
        t = y.argmax(-1)
        v = y.sum(-1) > 0
        try:
            from sklearn.metrics import confusion_matrix as sk_cm
            cm += sk_cm(t[v].ravel(), p[v].ravel(), labels=list(range(C)))
        except ImportError:
            for a, b in zip(t[v].ravel(), p[v].ravel()):
                cm[a, b] += 1
    diag = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - diag
    iou = np.where(union > 0, diag / np.maximum(union, 1), np.nan)
    print("\n-- Per-class segmentation results --")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:18s} IoU={iou[i]:.3f}  px={cm[i].sum():,}")
    print(f"  mIoU={np.nanmean(iou):.3f}")
    np.savetxt(os.path.join(WORK_DIR, "confusion_matrix.csv"), cm,
               fmt="%d", delimiter=",")
    out_metrics = {n: {"iou": float(iou[i]), "pixels": int(cm[i].sum())}
                   for i, n in enumerate(CLASS_NAMES)}
    out_metrics["mIoU"] = float(np.nanmean(iou))
    with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
        json.dump(out_metrics, f, indent=2)
    return cm, iou


def calibrate_temperature(net, val_ds, cfg, max_batches=10):
    """Post-hoc temperature scaling of seg probabilities on the val set."""
    if "seg_head" not in net.output_names:
        return 1.0
    probs, labels = [], []
    for i, (inputs, targets) in enumerate(val_ds):
        if i >= max_batches:
            break
        out = _preds_to_dict(net, net(inputs, training=False))
        p = out["seg_head"].numpy().reshape(-1, cfg.num_classes)
        y = targets["seg_head"].numpy().reshape(-1, cfg.num_classes)
        v = y.sum(-1) > 0
        if v.any():
            probs.append(p[v])
            labels.append(y[v].argmax(-1))
    if not probs:
        return 1.0
    p = np.clip(np.concatenate(probs), 1e-6, 1.0)
    y = np.concatenate(labels)
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 4.0, 36):
        q = np.exp(np.log(p) / t)
        q /= q.sum(-1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-9, 1.0)).mean()
        if nll < best_nll:
            best_nll, best_t = nll, t
    print(f"  [calibration] seg temperature = {best_t:.2f} (NLL {best_nll:.4f})")
    cfg.seg_temperature = float(best_t)
    if os.path.exists(STATS_PATH):
        z = dict(np.load(STATS_PATH))
        z["seg_temperature"] = np.float32(best_t)
        np.savez(STATS_PATH, **z)
    return best_t


def calibrate_change_conformal(net, val_ds, cfg, alpha=0.10, max_batches=15):
    """Split-conformal threshold for change alerts (finite-sample marginal
    coverage on the change class). Non-conformity score s = 1 - p(change) on
    truly-changed pixels; alert when p > 1 - Q_{1-alpha}(s)."""
    if "change_head" not in net.output_names:
        return None
    scores = []
    for i, (inputs, targets) in enumerate(val_ds):
        if i >= max_batches:
            break
        out = _preds_to_dict(net, net(inputs, training=False))
        p = out["change_head"].numpy()[..., 0]
        y = targets["change_head"].numpy()[..., 0]
        pos = y == 1.0
        if pos.any():
            scores.append((1.0 - p)[pos])
    if not scores:
        print("  [conformal] no positive change pixels sampled; "
              "keeping fixed threshold.")
        return None
    s = np.concatenate(scores)
    n = len(s)
    if n < 50:
        print(f"  [conformal] only {n} positive pixels; too few, skipping.")
        return None
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    q = float(np.quantile(s, level))
    thr = float(1.0 - q)
    cfg.change_conformal_q = thr
    print(f"  [conformal] change alert threshold = {thr:.3f} "
          f"(alpha={alpha}, n={n:,} positive px)")
    return thr


def plot_training_history(history):
    if history is None or not hasattr(history, "history"):
        return
    keys = [k for k in history.history if not k.startswith("val_")]
    n = min(len(keys), 9)
    if n == 0:
        return
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flat
    for ax, k in zip(axes, keys[:n]):
        ax.plot(history.history[k], label="train")
        if f"val_{k}" in history.history:
            ax.plot(history.history[f"val_{k}"], label="val")
        ax.set_title(k)
        ax.legend()
        ax.grid(alpha=.3)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "training_history.png"), dpi=150)
    plt.close()


def _stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def visualize_predictions(net, val_ds, cfg, n_samples=4):
    cmap = mcolors.ListedColormap(CLASS_COLORS)
    extra = [h for h in ("change_head", "quality_head", "boundary_head",
                         "species_head", "damage_head", "recovery_head",
                         "carbon_head", "change_year_head")
             if h in net.output_names]
    ncols = 3 + len(extra)
    fig, axes = plt.subplots(n_samples, ncols,
                             figsize=(4 * ncols, 4 * n_samples))
    axes = np.atleast_2d(axes)
    titles = ["S2 RGB (last step)", "GT Seg", "Pred Seg"] + \
             [h.replace("_head", "") for h in extra]
    for a, t in zip(axes[0], titles):
        a.set_title(t, fontsize=10)
    count = 0
    for inputs, targets in val_ds.take(2):
        preds = _preds_to_dict(net, net(inputs, training=False))
        preds = {k: v.numpy() for k, v in preds.items()}
        bs = len(targets["seg_head"])
        for i in range(min(bs, n_samples - count)):
            s2 = inputs["s2_seq"][i, -1].numpy()
            axes[count, 0].imshow(_stretch(s2[:, :, [2, 1, 0]]))
            y = targets["seg_head"][i].numpy().argmax(-1)
            axes[count, 1].imshow(y, cmap=cmap, vmin=0, vmax=4)
            if "seg_head" in preds:
                axes[count, 2].imshow(preds["seg_head"][i].argmax(-1),
                                      cmap=cmap, vmin=0, vmax=4)
            col = 3
            for h in extra:
                arr = preds[h][i]
                if h == "carbon_head":
                    axes[count, col].imshow(arr[..., 0] * cfg.carbon_scale,
                                            cmap="YlGn")
                elif h == "species_head":
                    axes[count, col].imshow(arr.argmax(-1), cmap="tab10",
                                            vmin=0, vmax=cfg.num_species - 1)
                elif h == "change_year_head":
                    axes[count, col].imshow(arr[..., 0], cmap="magma",
                                            vmin=0, vmax=1)
                else:
                    axes[count, col].imshow(arr[..., 0], cmap="Reds",
                                            vmin=0, vmax=1)
                col += 1
            for a in axes[count]:
                a.axis("off")
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "predictions_preview.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


# ==============================================================================
# 16. SELF-SUPERVISED PRETRAINING (temporal SimMIM; F17 + F20 fixed)
# ==============================================================================
def build_pretrain_model(cfg, s1_stats, s2_stats):
    """Masked-block reconstruction from the fused temporal stack. Encoder and
    fusion layer NAMES match the main model -> by_name weight transfer."""
    ps, A, T = cfg.patch_size, cfg.aux_channels, cfg.temporal_steps
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)
    L = len(dims)

    s1_seq = layers.Input((T, ps, ps, 4), name="s1_seq")
    s2_seq = layers.Input((T, ps, ps, 10), name="s2_seq")
    scl_seq = layers.Input((T, ps, ps, 1), name="scl_seq")
    aux_seq = layers.Input((T, ps, ps, A), name="aux_seq")

    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    n1 = layers.Normalization(mean=list(s1_stats[0]),
                              variance=list(s1_stats[1]), name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]),
                              variance=list(s2_stats[1]), name="s2_norm")
    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])
    sar_proj = [layers.Conv2D(d, 1, name=f"sar_proj{i}")
                for i, d in enumerate(dims)]
    fusions = [(CrossAttentionFusion(d, cfg.fusion_heads, name=f"fusion{i}")
                if i in tuple(cfg.cross_attn_levels)
                else GatedFusion(d, name=f"fusion{i}"))
               for i, d in enumerate(dims)]

    bns = []
    for t in range(T):
        f1, f2 = fe([s1_seq[:, t], s2_seq[:, t], scl_seq[:, t], aux_seq[:, t]])
        b1, sk1 = enc_s1(n1(f1))
        b2, sk2 = enc_s2(n2(f2))
        sproj = [sar_proj[i](sk1[i]) for i in range(L)]
        fused = [fusions[i]([sk2[i], sproj[i]]) for i in range(L)]
        bns.append(fused[-1])
    z = tf.keras.ops.mean(tf.keras.ops.stack(bns, axis=1), axis=1)

    x = z
    size = ps // 32
    widths = [192, 96, 64]
    wi = 0
    while size < ps // 4:              # F20: recon res == target res (ps/4)
        x = layers.UpSampling2D(interpolation="bilinear", name=f"pt_up{wi}")(x)
        x = conv_block(x, widths[min(wi, len(widths) - 1)], use_se=False,
                       name=f"pt_c{wi}")
        size *= 2
        wi += 1
    rec = layers.Conv2D(C1 + 10, 1, padding="same", dtype="float32",
                        name="recon_head")(x)
    return models.Model([s1_seq, s2_seq, scl_seq, aux_seq], rec,
                        name="MangrovePretrainer_v7")


def _masked_block_seq(x, ratio, block):
    """Zero random square blocks independently per (batch, time step).
    x: (B,T,ps,ps,C). Returns (x_masked, keep) with keep 1 = visible."""
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    ps = tf.shape(x)[2]
    g = tf.maximum(ps // block, 1)
    cells = tf.cast(tf.random.uniform((B, T, g, g, 1)) < ratio, tf.float32)
    mask = tf.image.resize(tf.reshape(cells, [B * T, g, g, 1]), [ps, ps],
                           method="nearest")
    mask = tf.reshape(mask, [B, T, ps, ps, 1])          # 1 = masked
    keep = 1.0 - mask
    return x * keep, keep


def pretrain(cfg=CFG):
    """Temporal SimMIM: reconstruct the LAST step's normalized S1 features +
    raw S2 bands at 1/4 resolution, from the heavily masked full stack."""
    print("\n=== v7 SELF-SUPERVISED PRETRAINING (temporal SimMIM) ===")
    scenes = discover_scenes(cfg.train_dir)
    assert scenes, "pretrain: set cfg.train_dir (scene folders or legacy NPZs)"
    s1m, s1v, s2m, s2v, _ = load_or_compute_stats(scenes, cfg)
    batch = cfg.batch_per_replica * N_REPLICAS
    ds = (tf.data.Dataset.from_generator(
            lambda: patch_generator(scenes, cfg, False, cfg.max_train_patches),
            output_signature=output_signature(cfg))
          .shuffle(cfg.shuffle_buffer, seed=42, reshuffle_each_iteration=True)
          .batch(batch, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))
    s1_mean = tf.constant(s1m, tf.float32)
    s1_std = tf.sqrt(tf.constant(s1v, tf.float32))

    with STRATEGY.scope():
        pt = build_pretrain_model(cfg, (s1m, s1v), (s2m, s2v))
        opt = optimizers.AdamW(learning_rate=cfg.pretrain_lr,
                               weight_decay=cfg.weight_decay)

        @tf.function
        def train_one(batch):
            s1 = batch["s1_seq"]
            s2 = batch["s2_seq"]
            scl = batch["scl_seq"]
            aux = batch["aux_seq"]
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block_seq(s1, cfg.pretrain_mask_ratio,
                                                cfg.pretrain_block)
                s2_m, keep2 = _masked_block_seq(s2, cfg.pretrain_mask_ratio,
                                                cfg.pretrain_block)
                rec = pt({"s1_seq": s1_m, "s2_seq": s2_m,
                          "scl_seq": scl, "aux_seq": aux}, training=True)
                vv, vh, vvs, vhs = tf.split(s1[:, -1], 4, axis=-1)
                lin_vv = tf.pow(10.0, vv / 10.0)
                lin_vh = tf.pow(10.0, vh / 10.0)
                rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
                t_s1 = tf.concat([vv, vh, vh - vv, rvi, vvs, vhs], -1)
                t_s1 = (t_s1 - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2[:, -1]], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1[:, -1], 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2[:, -1], 4, 4, "VALID")
                m = tf.concat([tf.repeat(tf.cast(m1 > 0.5, tf.float32), C1, -1),
                               tf.repeat(tf.cast(m2 > 0.5, tf.float32), 10, -1)],
                              -1)
                loss = tf.reduce_sum(tf.abs(rec - tgt_q) * m) / \
                    (tf.reduce_sum(m) + 1e-6)
            grads = tape.gradient(loss, pt.trainable_variables)
            opt.apply_gradients([(g, v) for g, v in
                                 zip(grads, pt.trainable_variables)
                                 if g is not None])
            return loss

        for ep in range(cfg.pretrain_epochs):
            losses = []
            t0 = time.time()
            for batch in ds:
                losses.append(float(train_one(batch)))
            print(f"  pretrain {ep + 1}/{cfg.pretrain_epochs} "
                  f"loss={np.mean(losses):.4f} ({time.time() - t0:.0f}s)")

        out = os.path.join(WORK_DIR, "pretrain_encoder_v7.weights.h5")
        pt.save_weights(out)
        print(f"Pretrained encoder weights saved: {out}")
        print(f"Fine-tune with: --pretrain {out}")
        return out


# ==============================================================================
# 17. BATCH PREFLIGHT + OOM-RESILIENT TRAINING
# ==============================================================================
def preflight_batch(trainer, ds):
    try:
        x, y = next(iter(ds))
        trainer.train_on_batch(x, y)
        return True
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        print(f"  [preflight] OOM at current batch: {str(e)[:160]}")
        return False


def train(cfg=CFG):
    print(f"\n=== MANGROVE ECO-MONITOR v{VERSION} - TRAINING ===\n")
    train_scenes = discover_scenes(cfg.train_dir)
    val_scenes = discover_scenes(cfg.val_dir)
    assert train_scenes and val_scenes, \
        "Set cfg.train_dir / cfg.val_dir (scene folders or legacy NPZs)"
    validate_heads_vs_data(train_scenes, cfg)
    print(f"train scenes: {len(train_scenes)} | val scenes: {len(val_scenes)} "
          f"| T={cfg.temporal_steps}")

    s1m, s1v, s2m, s2v, alpha = load_or_compute_stats(train_scenes, cfg)
    if cfg.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    oom_left = cfg.oom_retries
    history = None
    while True:
        batch = cfg.batch_per_replica * N_REPLICAS
        ds_train = make_dataset(
            lambda: patch_generator(train_scenes, cfg, True,
                                    cfg.max_train_patches), cfg, True, batch)
        ds_val = make_dataset(
            lambda: patch_generator(val_scenes, cfg, False,
                                    cfg.max_val_patches), cfg, False, batch)

        with STRATEGY.scope():
            net = build_model(cfg, (s1m, s1v), (s2m, s2v))
            if cfg.pretrain_weights and os.path.exists(cfg.pretrain_weights):
                try:
                    net.load_weights(cfg.pretrain_weights, by_name=True,
                                     skip_mismatch=True)
                    print(f"  [pretrain] loaded encoder weights (by_name, "
                          f"skip_mismatch) from {cfg.pretrain_weights}")
                except Exception as e:
                    print(f"  [pretrain] weight load failed ({e}); "
                          f"training from scratch.")
            trainer = EcoMonitorTrainer(cfg, net, alpha)
            est_steps = max(1, cfg.max_train_patches // batch)
            lr_sched = WarmUpCosineDecay(
                cfg.lr, cfg.epochs * est_steps,
                cfg.lr_warmup_epochs * est_steps)
            trainer.compile(optimizer=make_optimizer(cfg, lr_sched))

        if cfg.auto_batch:
            ok = preflight_batch(trainer, ds_train)
            while not ok and cfg.batch_per_replica > 1:
                cfg.batch_per_replica //= 2
                tf.keras.backend.clear_session()
                gc.collect()
                batch = cfg.batch_per_replica * N_REPLICAS
                ds_train = make_dataset(
                    lambda: patch_generator(train_scenes, cfg, True,
                                            cfg.max_train_patches),
                    cfg, True, batch)
                with STRATEGY.scope():
                    net = build_model(cfg, (s1m, s1v), (s2m, s2v))
                    trainer = EcoMonitorTrainer(cfg, net, alpha)
                    trainer.compile(optimizer=make_optimizer(cfg, lr_sched))
                ok = preflight_batch(trainer, ds_train)
            print(f"  [auto_batch] settled on batch/replica="
                  f"{cfg.batch_per_replica}")

        # F21 fix: metric names are explicit; monitor matches the log key.
        monitor = "val_seg_miou" if "seg" in cfg.active_heads else "val_loss"
        mode = "max" if "miou" in monitor else "min"
        ema = EMACallback(cfg.ema_decay)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=15,
                                             restore_best_weights=True,
                                             verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(WORK_DIR, "best_v7.weights.h5"), monitor,
                mode=mode, save_best_only=True, save_weights_only=True,
                verbose=1),
            tf.keras.callbacks.BackupAndRestore(
                os.path.join(WORK_DIR, "backup")),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(os.path.join(WORK_DIR,
                                                      "training_log.csv")),
            tf.keras.callbacks.TensorBoard(os.path.join(WORK_DIR, "tb")),
            ema,
        ]
        try:
            history = trainer.fit(ds_train, validation_data=ds_val,
                                  epochs=cfg.epochs, callbacks=cbs)
            break
        except tf.errors.ResourceExhaustedError as e:
            if oom_left <= 0 or cfg.batch_per_replica <= 1:
                raise
            oom_left -= 1
            cfg.batch_per_replica = max(1, cfg.batch_per_replica // 2)
            print(f"  [OOM-RETRY] halving batch to "
                  f"{cfg.batch_per_replica}/replica and restarting with fresh "
                  f"optimizer state ({oom_left} retries left). "
                  f"Detail: {str(e)[:120]}")
            tf.keras.backend.clear_session()
            gc.collect()

    print("\nSwapping in EMA weights for final eval + export...")
    ema.swap_in()
    calibrate_temperature(net, ds_val, cfg)
    calibrate_change_conformal(net, ds_val, cfg)
    model_path = os.path.join(WORK_DIR, "mangrove_ecomonitor_v7.keras")
    net.save(model_path)
    cfg.save(os.path.join(WORK_DIR, "config_v7.json"))
    plot_training_history(history)
    if "seg" in cfg.active_heads:
        evaluate_per_class(net, ds_val, cfg)
    visualize_predictions(net, ds_val, cfg)
    print(f"\nModel saved: {model_path}")
    print(f"Config saved: {os.path.join(WORK_DIR, 'config_v7.json')}")
    return net


# ==============================================================================
# 18. TILED INFERENCE (Hann blending, per-pass TTA weights, MC-dropout,
#     SAR-only fallback; F19 fixed: TTA touches spatial arrays ONLY)
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
    """Centre-peaked 2-D Hann window (the v4 inverted-kernel bug class)."""
    h = np.hanning(ps).astype(np.float32)
    h = np.maximum(h, floor)
    return np.outer(h, h)


def predict_scene(net, s1, s2=None, scl=None, aux=None, t2=None, cfg=CFG,
                  batch_size=None, tta=None, overlap=None, sar_only=False,
                  dates=None):
    """Production inference on a unified scene.

    Call either with a scene dict (load_scene / scene_from_arrays output) as
    `s1`, or v5-style with arrays: predict_scene(net, s1, s2, scl, aux, t2).
    """
    batch_size = batch_size or cfg.infer_batch
    tta = cfg.infer_tta if tta is None else tta
    overlap = cfg.infer_overlap if overlap is None else overlap
    ps, C, A = cfg.patch_size, cfg.num_classes, cfg.aux_channels
    S = cfg.num_species
    temp = float(getattr(cfg, "seg_temperature", 1.0))

    if isinstance(s1, dict) and "s1_seq" in s1:
        scene = dict(s1)
    else:
        scene = scene_from_arrays(s1, s2, scl, aux, t2, dates=dates, cfg=cfg)

    h, w = scene["s2_seq"].shape[1:3]
    if sar_only:   # optical branch zeroed across ALL steps (modality-dropout
        scene["s2_seq"] = np.zeros_like(scene["s2_seq"])  # trained for this)

    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        pad = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        for k in ("s1_seq", "s2_seq", "scl_seq", "aux_seq"):
            scene[k] = np.pad(scene[k], pad)
    H, W = scene["s2_seq"].shape[1:3]

    out_names = list(net.output_names)
    shapes = {"seg_head": (H, W, C), "species_head": (H, W, S),
              "carbon_head": (H, W, 2), "eco_head": (H, W, 2),
              "change_head": (H, W, 1), "change_year_head": (H, W, 1),
              "recovery_head": (H, W, 1), "damage_head": (H, W, 1),
              "risk_head": (H, W, 1), "quality_head": (H, W, 1),
              "boundary_head": (H, W, 1)}
    acc = {n: np.zeros(shapes[n], np.float32)
           for n in out_names if n in shapes}
    if not acc:
        raise RuntimeError("predict_scene: model exposes no known heads.")
    weight = np.zeros((H, W), np.float32)
    kernel = _hann_kernel(ps)
    mc = max(0, int(cfg.mc_samples))
    mc_sq = ({n: np.zeros_like(acc[n]) for n in acc} if mc > 1 else None)

    # F19: transforms defined for 5-D (N,T,ps,ps,C) inputs / 4-D predictions,
    # applied ONLY to keys ending in "_seq"; dates are never transformed.
    variants = [
        (lambda x: x, lambda p: p),
        (lambda x: x[:, :, :, ::-1], lambda p: p[:, :, ::-1]),
        (lambda x: x[:, :, ::-1], lambda p: p[:, ::-1]),
        (lambda x: np.rot90(x, 1, (2, 3)), lambda p: np.rot90(p, -1, (1, 2))),
    ] if tta else [(lambda x: x, lambda p: p)]

    coords = [(i, j) for i in _tile_starts(H, ps, overlap)
              for j in _tile_starts(W, ps, overlap)]
    print(f"Inference: {h}x{w} -> padded {H}x{W} | tiles={len(coords)} | "
          f"batch={batch_size} | TTA={tta} | MC={mc} | sar_only={sar_only} | "
          f"dates={scene['dates'].tolist()}")

    for fwd, inv in variants:
        for k in range(0, len(coords), batch_size):
            chunk = coords[k:k + batch_size]

            def stack(arr):
                return np.stack([arr[:, i:i + ps, j:j + ps] for i, j in chunk])

            batch = {"s1_seq": stack(scene["s1_seq"]),
                     "s2_seq": stack(scene["s2_seq"]),
                     "scl_seq": stack(scene["scl_seq"]),
                     "aux_seq": stack(scene["aux_seq"]),
                     "dates": np.repeat(scene["dates"][None], len(chunk),
                                        axis=0)}
            fb = {k2: (np.ascontiguousarray(fwd(v))
                       if k2.endswith("_seq") else v)
                  for k2, v in batch.items()}
            reps = mc if mc > 1 else 1
            for rep in range(reps):
                if mc > 1:
                    preds = _preds_to_dict(net, net(fb, training=(rep > 0)))
                    preds = {k2: v.numpy() for k2, v in preds.items()}
                else:
                    preds = _preds_to_dict(net, net.predict(
                        fb, verbose=0, batch_size=len(chunk)))
                for name in acc:
                    pr = inv(preds[name])
                    pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
                    for n_i, (i, j) in enumerate(chunk):
                        sl = np.s_[i:i + ps, j:j + ps]
                        acc[name][sl] += pr[n_i] * kernel[..., None]
                        if mc_sq is not None:
                            mc_sq[name][sl] += (pr[n_i] ** 2) * kernel[..., None]
            for i, j in chunk:          # weight added PER PASS (v5 F9 class)
                weight[i:i + ps, j:j + ps] += kernel

    weight = np.maximum(weight[..., None], 1e-6)
    out = {n: acc[n] / weight for n in acc}

    res = {}
    d0 = float(scene["dates"][0])
    if "seg_head" in out:
        p = out["seg_head"]
        if temp != 1.0:
            lp = np.log(np.clip(p, 1e-6, 1.0)) / temp
            lp -= lp.max(-1, keepdims=True)
            p = np.exp(lp)
            p /= p.sum(-1, keepdims=True)
        res["classes"] = p.argmax(-1).astype(np.uint8)[:h, :w]
        res["seg_probs"] = p.astype(np.float32)[:h, :w]
        res["flood_risk"] = (p[..., 1] + p[..., 2]).astype(np.float32)[:h, :w]
        res["mangrove_extent"] = ((res["classes"] == 3) |
                                  (res["classes"] == 4)).astype(np.uint8)
    if "species_head" in out:
        sp = out["species_head"]
        sp = sp / np.maximum(sp.sum(-1, keepdims=True), 1e-6)
        res["species_probs"] = sp.astype(np.float32)[:h, :w]
        res["species_classes"] = sp.argmax(-1).astype(np.uint8)[:h, :w]
    if "change_head" in out:
        cp = out["change_head"][..., 0].astype(np.float32)
        res["change_prob"] = cp[:h, :w]
        thr = (cfg.change_conformal_q if cfg.change_conformal_q > 0
               else cfg.change_threshold)
        res["change_alert"] = (cp > thr).astype(np.uint8)[:h, :w]
    if "change_year_head" in out:
        rel = np.clip(out["change_year_head"][..., 0], 0.0, 1.0)
        res["change_year"] = (d0 + rel * cfg.max_time_gap_years) \
            .astype(np.float32)[:h, :w]
    if "recovery_head" in out:
        res["recovery_prob"] = out["recovery_head"][..., 0] \
            .astype(np.float32)[:h, :w]
    if "damage_head" in out:
        res["damage_prob"] = out["damage_head"][..., 0] \
            .astype(np.float32)[:h, :w]
    if "carbon_head" in out:
        cm = out["carbon_head"][..., 0] * cfg.carbon_scale
        cs = np.sqrt(np.exp(np.clip(out["carbon_head"][..., 1], -6, 8))) \
            * cfg.carbon_scale
        res["carbon_mean"] = cm.astype(np.float32)[:h, :w]
        res["carbon_std"] = cs.astype(np.float32)[:h, :w]
        res["carbon_ci_lower"] = np.maximum(cm - 1.96 * cs, 0) \
            .astype(np.float32)[:h, :w]
        res["carbon_ci_upper"] = (cm + 1.96 * cs).astype(np.float32)[:h, :w]
    if "eco_head" in out:
        res["eco_temp"] = (out["eco_head"][..., 0] * cfg.eco_scale[0]) \
            .astype(np.float32)[:h, :w]
        res["eco_moisture"] = out["eco_head"][..., 1].astype(np.float32)[:h, :w]
    if "risk_head" in out:
        res["risk_index"] = (out["risk_head"][..., 0] * 100.0) \
            .astype(np.float32)[:h, :w]
    if "quality_head" in out:
        res["quality_prob"] = out["quality_head"][..., 0] \
            .astype(np.float32)[:h, :w]
    if mc_sq is not None and "seg_head" in out:
        var = np.maximum(mc_sq["seg_head"] / weight - out["seg_head"] ** 2,
                         0.0)
        res["seg_epistemic_var"] = var.mean(-1).astype(np.float32)[:h, :w]
    return res


# ==============================================================================
# 19. EXPORT: COG GeoTIFFs + STAC item + GeoJSON alerts + MRV summary
#     + self-contained HTML dashboard
# ==============================================================================
def export_geotiff(reference_tif, array, out_path, dtype="float32",
                   nodata=None, band_names=None):
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
            if band_names:
                dst.set_band_description(b + 1, band_names[b])
    print(f"  saved {out_path}")
    return out_path


def _pixel_area_ha(reference_tif, cfg):
    try:
        import rasterio
        with rasterio.open(reference_tif) as src:
            a, _, _, _, e, _ = src.transform[:6]
            if src.crs and src.crs.is_geographic:
                print("  [mrv] geographic CRS detected; using nominal "
                      f"{cfg.pixel_size_m} m pixel size.")
                return (cfg.pixel_size_m ** 2) / 1e4
            return abs(a * e) / 1e4
    except Exception:
        return (cfg.pixel_size_m ** 2) / 1e4


def export_change_geojson(products, reference_tif, out_path, cfg, period=""):
    try:
        import rasterio
        from rasterio.features import shapes
    except ImportError:
        print("  [export] rasterio missing; GeoJSON alerts skipped.")
        return None
    alert = products.get("change_alert")
    if alert is None or alert.sum() == 0:
        print("  [export] no change alerts above threshold.")
        return None
    pa_ha = _pixel_area_ha(reference_tif, cfg)
    with rasterio.open(reference_tif) as src:
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
    feats = []
    for geom, val in shapes(alert, mask=alert > 0, transform=transform):
        feats.append({"type": "Feature",
                      "properties": {"alert": "mangrove_loss",
                                     "period": period,
                                     "confidence": cfg.change_conformal_q
                                     if cfg.change_conformal_q > 0
                                     else cfg.change_threshold},
                      "geometry": geom})
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs or "EPSG:4326"}},
          "features": feats}
    with open(out_path, "w") as f:
        json.dump(fc, f)
    print(f"  saved {out_path} ({len(feats)} alert polygons)")
    return out_path


def write_stac_item(out_dir, products_meta, reference_tif, scene_id, cfg):
    item = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": scene_id,
        "collection": "mangrove-ecomonitor-v7",
        "properties": {
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processing:software": f"mangrove-ecomonitor v{VERSION}",
            "mangrove:classes": CLASS_NAMES,
            "mangrove:species": list(cfg.species_names),
            "mangrove:active_heads": list(cfg.active_heads),
            "mangrove:temporal_steps": cfg.temporal_steps,
        },
        "assets": {},
        "links": [],
    }
    try:
        import rasterio
        with rasterio.open(reference_tif) as src:
            b = src.bounds
            item["bbox"] = [b.left, b.bottom, b.right, b.top]
            item["geometry"] = {"type": "Polygon", "coordinates": [[
                [b.left, b.bottom], [b.right, b.bottom], [b.right, b.top],
                [b.left, b.top], [b.left, b.bottom]]]}
    except Exception:
        pass
    for fname, title in products_meta:
        item["assets"][os.path.splitext(fname)[0]] = {
            "href": f"./{fname}", "type": "image/tiff; application=geotiff",
            "title": title, "roles": ["data"]}
    with open(os.path.join(out_dir, "stac_item.json"), "w") as f:
        json.dump(item, f, indent=2)
    print("  saved stac_item.json")


def mrv_summary(products, reference_tif, out_dir, cfg):
    """VM0033-aligned reporting primitives: class + species areas (ha), carbon
    stock with 95% CI in tCO2e, change/recovery/damage areas."""
    pa_ha = _pixel_area_ha(reference_tif, cfg)
    rows = [("metric", "value", "unit")]
    if "classes" in products:
        for i, n in enumerate(CLASS_NAMES):
            area = float((products["classes"] == i).sum()) * pa_ha
            rows.append((f"area_class_{i}_{n.replace(' ', '_').replace('/', '_')}",
                         f"{area:.2f}", "ha"))
    if "species_classes" in products and "mangrove_extent" in products:
        m = products["mangrove_extent"] > 0
        for i, n in enumerate(cfg.species_names):
            area = float(((products["species_classes"] == i) & m).sum()) * pa_ha
            rows.append((f"area_species_{i}_{n.replace(' ', '_')}",
                         f"{area:.2f}", "ha"))
    if "carbon_mean" in products and "mangrove_extent" in products:
        m = products["mangrove_extent"] > 0
        agb_t = float(products["carbon_mean"][m].sum()) * pa_ha
        var_t = float((products["carbon_std"][m] ** 2).sum()) * (pa_ha ** 2)
        std_t = math.sqrt(max(var_t, 0.0))
        co2e = agb_t * cfg.c_fraction * cfg.carbon_to_co2e
        co2e_std = std_t * cfg.c_fraction * cfg.carbon_to_co2e
        rows += [
            ("agb_stock", f"{agb_t:.1f}", "t"),
            ("agb_stock_std", f"{std_t:.1f}", "t"),
            ("co2e_stock", f"{co2e:.1f}", "tCO2e"),
            ("co2e_ci95_low", f"{max(co2e - 1.96 * co2e_std, 0):.1f}", "tCO2e"),
            ("co2e_ci95_high", f"{co2e + 1.96 * co2e_std:.1f}", "tCO2e"),
        ]
    if "change_alert" in products:
        rows.append(("change_alert_area",
                     f"{float(products['change_alert'].sum()) * pa_ha:.2f}",
                     "ha"))
    if "change_year" in products and "change_alert" in products:
        m = products["change_alert"] > 0
        if m.any():
            rows.append(("change_year_median",
                         f"{float(np.median(products['change_year'][m])):.2f}",
                         "year"))
    if "recovery_prob" in products:
        rows.append(("recovery_area_p50",
                     f"{float((products['recovery_prob'] > 0.5).sum()) * pa_ha:.2f}",
                     "ha"))
    if "damage_prob" in products:
        rows.append(("damage_area_p50",
                     f"{float((products['damage_prob'] > 0.5).sum()) * pa_ha:.2f}",
                     "ha"))
    path = os.path.join(out_dir, "mrv_summary.csv")
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  saved {path}")
    return rows


def write_dashboard_html(out_dir, products, rows, cfg, scene_id):
    """Self-contained MRV dashboard: preview panels embedded as base64 PNG."""
    panels = []
    if "classes" in products:
        panels.append(("classes", "Land cover", "classes"))
    if "change_prob" in products:
        panels.append(("change_prob", "Change probability", "Reds"))
    if "carbon_mean" in products:
        panels.append(("carbon_mean", "AGB (t/ha)", "YlGn"))
    if "quality_prob" in products:
        panels.append(("quality_prob", "Pixel usability", "viridis"))
    b64 = ""
    if panels:
        try:
            from io import BytesIO
            cmap_cls = mcolors.ListedColormap(CLASS_COLORS)
            fig, axes = plt.subplots(1, len(panels),
                                     figsize=(5 * len(panels), 5))
            axes = np.atleast_1d(axes)
            for ax, (key, title, cm) in zip(axes, panels):
                if cm == "classes":
                    ax.imshow(products[key], cmap=cmap_cls, vmin=0, vmax=4)
                elif cm in ("Reds", "viridis"):
                    ax.imshow(products[key], cmap=cm, vmin=0, vmax=1)
                else:
                    ax.imshow(products[key], cmap=cm)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"  [dashboard] preview render failed: {e}")
    rows_html = "".join(
        f"<tr><td>{r[0]}</td><td style='text-align:right'>{r[1]}</td>"
        f"<td>{r[2]}</td></tr>" for r in rows[1:])
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>MRV dashboard - {scene_id}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#0b1e2d;"
        "color:#e8f1f5}"
        "h1{color:#39c2d7;font-size:22px}"
        "table{border-collapse:collapse;margin-top:18px}"
        "td,th{border:1px solid #2a4a5f;padding:6px 12px;font-size:14px}"
        "img{max-width:100%;border:1px solid #2a4a5f;border-radius:8px;"
        "margin-top:14px}"
        ".meta{color:#8fb8c9;font-size:13px}"
        "</style></head><body>"
        f"<h1>Mangrove Eco-Monitor v{VERSION} &mdash; MRV Dashboard</h1>"
        f"<div class='meta'>scene: {scene_id} | heads: "
        f"{', '.join(cfg.active_heads)} | generated "
        f"{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} UTC</div>"
        + (f"<img alt='product preview' src='data:image/png;base64,{b64}'>"
           if b64 else "")
        + "<table><tr><th>metric</th><th>value</th><th>unit</th></tr>"
        + rows_html + "</table></body></html>")
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  saved {path}")


def export_all_products(net, scene_path, ref_tif_path, out_dir, cfg=CFG,
                        sar_only=False):
    os.makedirs(out_dir, exist_ok=True)
    desc = ("folder", scene_path) if os.path.isdir(scene_path) \
        else ("npz", scene_path)
    scene = load_scene(desc, cfg)
    products = predict_scene(net, scene, cfg=cfg, sar_only=sar_only)

    meta = []
    plan = [
        ("classes.tif", "classes", "uint8", 255, "Land-cover class (GMW-aligned)"),
        ("seg_probs.tif", "seg_probs", "float32", None, "Per-class probability"),
        ("species_classes.tif", "species_classes", "uint8", 255, "Species class"),
        ("carbon_mean.tif", "carbon_mean", "float32", -999, "AGB t/ha"),
        ("carbon_std.tif", "carbon_std", "float32", -999, "AGB uncertainty t/ha"),
        ("change_prob.tif", "change_prob", "float32", -999, "Change probability"),
        ("change_alert.tif", "change_alert", "uint8", 255, "Change alert mask"),
        ("change_year.tif", "change_year", "float32", -999,
         "Estimated year of change (decimal)"),
        ("recovery_prob.tif", "recovery_prob", "float32", -999,
         "Recovery probability"),
        ("damage_prob.tif", "damage_prob", "float32", -999,
         "Debris/damage probability"),
        ("risk_index.tif", "risk_index", "float32", -999, "Risk index 0-100"),
        ("eco_temperature.tif", "eco_temp", "float32", -999, "LST C"),
        ("eco_moisture.tif", "eco_moisture", "float32", -999, "Moisture 0-1"),
        ("quality.tif", "quality_prob", "float32", -999, "Pixel usability prob"),
        ("mangrove_extent.tif", "mangrove_extent", "uint8", 255, "Mangrove mask"),
        ("seg_epistemic.tif", "seg_epistemic_var", "float32", -999,
         "MC-dropout epistemic variance"),
    ]
    for fname, key, dtype, nodata, title in plan:
        if key in products:
            p = export_geotiff(ref_tif_path, products[key],
                               os.path.join(out_dir, fname), dtype=dtype,
                               nodata=nodata)
            if p:
                meta.append((fname, title))

    period = f"{float(scene['dates'][0]):.2f}-{float(scene['dates'][-1]):.2f}"
    export_change_geojson(products, ref_tif_path,
                          os.path.join(out_dir, "change_alerts.geojson"),
                          cfg, period=period)
    scene_id = os.path.splitext(os.path.basename(
        scene_path.rstrip("/")))[0] or "scene"
    write_stac_item(out_dir, meta, ref_tif_path, scene_id, cfg)
    rows = mrv_summary(products, ref_tif_path, out_dir, cfg)
    write_dashboard_html(out_dir, products, rows, cfg, scene_id)
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_path, "reference": ref_tif_path,
                   "version": VERSION, "active_heads": list(cfg.active_heads),
                   "sar_only": sar_only, "dates": scene["dates"].tolist(),
                   "seg_temperature": cfg.seg_temperature,
                   "change_conformal_q": cfg.change_conformal_q}, f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products


# ==============================================================================
# 20. SMOKE TEST (exercises every default head + TTA + SAR-only + reload)
# ==============================================================================
def _smoke_generator(cfg, n=16):
    T, ps, A = cfg.temporal_steps, cfg.patch_size, cfg.aux_channels
    rng = np.random.default_rng(42)
    for _ in range(n):
        s1_seq = np.concatenate(
            [rng.uniform(-30, 0, (T, ps, ps, 2)),
             rng.uniform(0, 3, (T, ps, ps, 2))], -1).astype(np.float32)
        s2_seq = rng.uniform(0, 0.5, (T, ps, ps, 10)).astype(np.float32)
        scl_seq = rng.choice([4, 5, 6, 8, 9], (T, ps, ps, 1)).astype(np.float32)
        aux_seq = rng.uniform(-2, 2, (T, ps, ps, A)).astype(np.float32)
        dates = (2020.0 + np.arange(T, dtype=np.float32) * 0.5)
        label = rng.choice([0, 1, 2, 3, 4, 255], (ps, ps)).astype(np.int32)
        species = rng.choice([0, 1, 2, 255], (ps, ps)).astype(np.int32)
        change = rng.choice([0, 0, 0, 1, 255], (ps, ps)).astype(np.int32)
        recovery = rng.choice([0, 0, 1, 255], (ps, ps)).astype(np.int32)
        debris = rng.choice([0, 0, 1, 255], (ps, ps)).astype(np.int32)
        change_year = np.where(change == 1, 2021.5, -999.0).astype(np.float32)
        carbon = rng.uniform(0, 250, (ps, ps)).astype(np.float32)
        eco = np.stack([rng.uniform(-5, 45, (ps, ps)),
                        rng.uniform(0, 1, (ps, ps))], -1).astype(np.float32)
        risk = rng.uniform(0, 1, (ps, ps)).astype(np.float32)
        yield {"s1_seq": s1_seq, "s2_seq": s2_seq, "scl_seq": scl_seq,
               "aux_seq": aux_seq, "dates": dates, "label": label,
               "carbon": carbon, "eco": eco, "change": change, "risk": risk,
               "species": species, "debris": debris, "recovery": recovery,
               "change_year": change_year}


def smoke_test(cfg=CFG):
    print(f"\n=== SMOKE TEST v{VERSION} (dummy data) ===")
    cfg.batch_per_replica = min(cfg.batch_per_replica, 2)
    batch = cfg.batch_per_replica * N_REPLICAS
    ds_tr = make_dataset(lambda: _smoke_generator(cfg, 16), cfg, True, batch)
    ds_va = make_dataset(lambda: _smoke_generator(cfg, 8), cfg, False, batch)

    c1, c2 = C1, C2(cfg)
    s1_stats = (np.zeros(c1), np.ones(c1))
    s2_stats = (np.zeros(c2), np.ones(c2))

    with STRATEGY.scope():
        net = build_model(cfg, s1_stats, s2_stats)
        trainer = EcoMonitorTrainer(cfg, net, np.ones(cfg.num_classes))
        trainer.compile(optimizer=make_optimizer(cfg, 1e-3))
    print(f"  model params: {net.count_params() / 1e6:.1f}M | "
          f"outputs: {net.output_names}")
    trainer.fit(ds_tr, validation_data=ds_va, epochs=2)

    model_path = os.path.join(WORK_DIR, "smoke_v7.keras")
    net.save(model_path)
    m2 = tf.keras.models.load_model(model_path)
    print("  reload OK - all custom objects deserialized.")

    H, W, A = 300, 380, cfg.aux_channels
    rng = np.random.default_rng(0)
    s1 = rng.uniform(-30, 0, (H, W, 2)).astype(np.float32)
    s2 = rng.uniform(0, 0.5, (H, W, 10)).astype(np.float32)
    aux = rng.uniform(-2, 2, (H, W, A)).astype(np.float32)
    t2 = {"s1": s1 + 0.5, "s2": np.clip(s2 + 0.02, 0, 1),
          "scl": np.full((H, W), 4, np.float32), "aux": aux + 0.1}
    products = predict_scene(m2, s1, s2, aux=aux, t2=t2, cfg=cfg,
                             batch_size=2, tta=True, dates=(2022.0, 2024.0))
    products_sar = predict_scene(m2, s1, s2, aux=aux, t2=t2, cfg=cfg,
                                 batch_size=2, tta=False, sar_only=True,
                                 dates=(2022.0, 2024.0))
    print(f"  inference OK. products: {sorted(products.keys())}")
    if "change_prob" in products:
        pr = products["change_prob"]
        print(f"  change_prob range: [{pr.min():.3f}, {pr.max():.3f}] "
              f"(finite={np.isfinite(pr).all()})")
    if "seg_probs" in products:
        sp = products["seg_probs"]
        print(f"  seg prob sums ~= 1: "
              f"{np.allclose(sp.sum(-1), 1.0, atol=1e-3)}")
    print(f"  sar_only inference OK: {sorted(products_sar.keys())[:3]}...")
    print("=== SMOKE TEST PASSED ===")
    return True


# ==============================================================================
# 21. CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=f"Mangrove Eco-Monitor v{VERSION}")
    ap.add_argument("mode", choices=["smoke", "pretrain", "train", "eval",
                                     "export"], nargs="?", default="smoke")
    ap.add_argument("scene", nargs="?",
                    help="scene folder or legacy NPZ (export)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (export)")
    ap.add_argument("out", nargs="?", help="output dir (export)")
    ap.add_argument("--config", help="path to config_v7.json")
    ap.add_argument("--sar-only", action="store_true",
                    help="optical-free inference (cloudy scenes)")
    ap.add_argument("--pretrain", dest="pretrain_w", default="",
                    help="encoder weights to fine-tune from")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.pretrain_w:
        cfg.pretrain_weights = args.pretrain_w
    resolve_hardware(cfg)

    if args.mode == "smoke":
        smoke_test(cfg)
    elif args.mode == "pretrain":
        pretrain(cfg)
    elif args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        net = tf.keras.models.load_model(
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v7.keras"))
        scenes = discover_scenes(cfg.test_dir)
        assert scenes, "eval: set cfg.test_dir"
        ds = make_dataset(lambda: patch_generator(scenes, cfg, False,
                                                  cfg.max_val_patches),
                          cfg, False, cfg.batch_per_replica * N_REPLICAS)
        evaluate_per_class(net, ds, cfg)
    elif args.mode == "export":
        if not (args.scene and args.ref and args.out):
            ap.error("export requires <scene> <ref.tif> <out_dir>")
        net = tf.keras.models.load_model(
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v7.keras"))
        export_all_products(net, args.scene, args.ref, args.out, cfg,
                            sar_only=args.sar_only)


if __name__ == "__main__":
    main()