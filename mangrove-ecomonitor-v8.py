"""
================================================================================
 MANGROVE ECO-MONITOR v8.0  "BLUE SENTINEL APEX"
 Unified production system: v7 multi-temporal core + full loss/metric layer,
 explainability, active learning, edge export, temporal smoothing, dataset
 versioning, and hardened fault tolerance.

 REQUIRES: Python >= 3.10, TensorFlow >= 2.16 (Keras 3), Linux or WSL2.
           Optional: rasterio (COG/GeoJSON export), scikit-learn, matplotlib,
                     tf2onnx (ONNX export).

--------------------------------------------------------------------------------
 WHAT V8 FIXES / ADDS (audit of v7)
--------------------------------------------------------------------------------
 [V8-1]  v7 referenced SIX loss classes that were never defined
         (FocalDiceLoss, MaskedBinaryFocal, MaskedBCE, MaskedMAELoss,
         CarbonNLLLoss, MaskedHuber) -> guaranteed NameError when the trainer
         was constructed. All are implemented here, Keras-3 serializable.
 [V8-2]  MeanIoU metric class was missing -> NameError. Implemented as a
         masked, streaming confusion-matrix metric (ignore = all-zero one-hot
         rows; classes with zero support are excluded from the mean).
 [V8-3]  TimeEmbedding was fed a (B,1) tensor producing a rank-3 (B,1,dim)
         conditioning vector. FiLM happened to survive the reshape, but the
         contract was ambiguous. v8 flattens dt to (B,) -> clean (B,dim).
 [V8-4]  Deep-supervision target pooling was hard-coded to 4x with a comment
         saying "assumes ps divisible by 4". v8 derives the stride from
         len(dec_widths): deep_stride = max(1, 32 // 2**len(dec_widths)),
         with a config-time divisibility assert.
 [V8-5]  Static Python unroll over T symbolic steps explodes the graph for
         large T. v8 adds cfg.temporal_unroll_warn (default 12): a loud
         warning with remediation guidance.
 [V8-6]  Conformal threshold now ships with a calibration certificate
         (n positives, alpha, timestamp, exchangeability caveat) saved to
         calibration_certificate.json, and warns when n < 1000.
 [V8-7]  Post-training calibration / eval are fault-isolated: a failure
         there can no longer destroy a finished 120-epoch training run.
 [V8-8]  NEW: export-tflite mode (TFLite with SELECT_TF_OPS fallback and
         optional int8 dynamic quantization; ONNX attempted if tf2onnx is
         installed). Failures print actionable guidance, never traceback-spam.
 [V8-9]  NEW: Grad-CAM explainability (--gradcam on export): class-
         discriminative heatmaps from the bottleneck features, rendered to
         gradcam.png for regulatory / stakeholder audits.
 [V8-10] NEW: Active-learning review queue. Tiles with low quality_prob,
         change_prob near the decision boundary, or high MC epistemic
         variance are exported (inputs + predictions + reason) to
         review_queue/ for human relabeling.
 [V8-11] NEW: post-temporal HMM smoothing (mode "smooth"): vectorized
         Viterbi decoding over a time series of per-date seg-probability
         stacks; forbids implausible per-pixel class flicker
         (bare -> dense mangrove in one step) via a stay-biased transition.
 [V8-12] NEW: dataset versioning. SHA-256 fingerprint of the scene list
         (names + sizes) is computed before training and logged to config,
         run metadata and the training log for full reproducibility.

 SCENE SCHEMA: unchanged from v7 (folder of dated step NPZs OR legacy pair
 NPZ; labels read from last step; validator auto-disables headless tasks).

 MODES: python mangrove_v8.py smoke | pretrain | train | eval | export |
        export-tflite | smooth
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
import hashlib
import argparse
import platform
import subprocess
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "8.0.0"
MIN_TF = (2, 16)

# FIX: Prevent TF CPU ArithmeticOptimizer crash on mixed_float16 temp vars
tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})


# ==============================================================================
# 1. ENVIRONMENT, GPU, PLATFORM GUARDS
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


def environment_check():
    ok = True
    if _parse_version(tf.__version__) < MIN_TF:
        print(f"[FATAL] TensorFlow {tf.__version__} < {'.'.join(map(str, MIN_TF))}. "
              f"v8 requires Keras 3 APIs. pip install -U tensorflow")
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
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v8/"
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
    train_dir: str = "/kaggle/input/mangrove-v8/train/"
    val_dir: str = "/kaggle/input/mangrove-v8/val/"
    test_dir: str = "/kaggle/input/mangrove-v8/test/"
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
    temporal_steps: int = 2               # T=2 reproduces pair behaviour
    temporal_unroll_warn: int = 12        # V8-5: static-unroll graph guard
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
    change_widths: tuple = (128, 96, 64, 48, 32)   # length-agnostic
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
    max_scene_failures: int = 25
    pretrain_weights: str = ""
    seed: int = 42                        # V8: shuffle/aug seed is configurable
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
    conformal_alpha: float = 0.10         # V8-6
    # ---- explainability / active learning (V8-9 / V8-10) ----
    gradcam_class: int = 4                # default: dense mangrove
    gradcam_layer: str = "bn_norm"
    gradcam_tiles: int = 4
    review_queue_max: int = 64
    review_tile: int = 256
    review_low_quality: float = 0.5
    review_change_margin: float = 0.10
    # ---- temporal smoothing (V8-11) ----
    smooth_stay: float = 0.85             # HMM self-transition probability
    # ---- edge export (V8-8) ----
    tflite_quantize: bool = False
    try_onnx: bool = True
    # ---- MRV ----
    pixel_size_m: float = 10.0
    carbon_to_co2e: float = 44.0 / 12.0
    c_fraction: float = 0.5
    shuffle_buffer: int = 256

    @property
    def aux_channels(self):
        return len(self.aux_names)

    @property
    def deep_stride(self):
        """V8-4: deep-supervision target downsampling, derived from decoder
        depth (bottleneck is ps/32; len(dec_widths) bilinear upsamples)."""
        return max(1, 32 // (2 ** len(self.dec_widths)))

    def check(self):
        assert self.patch_size % 32 == 0, "patch_size must be divisible by 32"
        assert self.patch_size % self.deep_stride == 0, \
            f"patch_size {self.patch_size} not divisible by deep_stride " \
            f"{self.deep_stride} (adjust dec_widths)"
        assert self.temporal_steps >= 1
        return True

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
# 2b. DATASET VERSIONING (V8-12)
# ==============================================================================
def dataset_fingerprint(scene_descs):
    """SHA-256 over sorted scene file names + sizes. Cheap, deterministic,
    and enough to detect any training-set change across runs."""
    h = hashlib.sha256()
    n_files = 0
    for kind, path in sorted(scene_descs, key=lambda d: d[1]):
        files = (sorted(glob.glob(os.path.join(path, "*.npz")))
                 if kind == "folder" else [path])
        for f in files:
            h.update(os.path.basename(f).encode())
            try:
                h.update(str(os.path.getsize(f)).encode())
                n_files += 1
            except OSError:
                h.update(b"missing")
    return h.hexdigest()[:16], n_files


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
    T = cfg.temporal_steps
    steps = sorted(steps, key=lambda s: s[4])
    if len(steps) > T:
        idx = np.unique(np.linspace(0, len(steps) - 1, T).round().astype(int))
        steps = [steps[int(i)] for i in idx]
    while len(steps) < T:
        steps.append(steps[-1])
    dates = np.array([s[4] for s in steps], np.float32)
    if np.allclose(dates, dates[0]):
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
        return np.ascontiguousarray(v)      # negative strides kill tf.convert

    for key, v in p.items():
        if key != "dates" and isinstance(v, np.ndarray) and v.ndim >= 2:
            p[key] = t(v)
    return p


def patch_generator(scene_descs, cfg, training, max_patches, seed=None):
    ps = cfg.patch_size
    stride = cfg.train_stride if training else ps
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
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
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v8.npz")


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
# 7. tf.data PIPELINE
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
    deep_k = int(cfg.deep_stride)                    # V8-4: derived, not fixed 4
    active = set(cfg.active_heads)

    def tf_aug(d):
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
        if deep:  # V8-4: pool factor derived from decoder depth
            targets["seg_deep_head"] = tf.nn.avg_pool2d(
                seg[None], deep_k, deep_k, "VALID")[0]
        scl_last = tf.cast(d["scl_seq"][-1], tf.int32)
        bad = tf.zeros(tf.shape(scl_last), tf.bool)
        for c in invalid:
            bad = tf.logical_or(bad, tf.equal(scl_last, c))
        targets["quality_head"] = 1.0 - tf.cast(bad, tf.float32)
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
        ds = ds.shuffle(cfg.shuffle_buffer, seed=cfg.seed,
                        reshuffle_each_iteration=True)
        if cfg.augment:
            ds = ds.map(tf_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(cfg.deterministic)
    ds = ds.with_options(opts)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 8. SWIN-V2 COMPONENTS
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
        self.cpb1 = layers.Dense(self.cpb_hidden, activation="relu",
                                 name="cpb1", dtype="float32")
        self.cpb2 = layers.Dense(self.num_heads, use_bias=False,
                                 name="cpb2", dtype="float32")
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
        attn += tf.cast(tf.transpose(bias, [2, 0, 1])[None], tf.float32)
        if mask is not None:
            nW = tf.shape(mask)[0]
            mask_f32 = tf.cast(mask, tf.float32)
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) + mask_f32[None]
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
    """Per-location temporal transformer over the bottleneck sequence."""

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
        de = tf.repeat(de, self.N, axis=0)
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
    
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FlattenDt(layers.Layer):
    def call(self, x):
        return tf.reshape(x, [-1])

# ==============================================================================
# 10. ENCODERS / DECODER BLOCKS
# ==============================================================================
def _heads_for(dim):
    return max(2, min(12, dim // 16))


def make_swin_encoder(prefix, dims, depths, cfg, dp_rates):
    """All layers built ONCE; closure reused for every time step (siamese)."""
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
# 11. FULL V8 MODEL
# ==============================================================================
def build_model(cfg, s1_stats, s2_stats):
    ops = tf.keras.ops
    C, S, ps, A, T = (cfg.num_classes, cfg.num_species, cfg.patch_size,
                      cfg.aux_channels, cfg.temporal_steps)
    if T > cfg.temporal_unroll_warn:                    # V8-5
        print(f"  [WARN] temporal_steps={T} > {cfg.temporal_unroll_warn}: the "
              f"siamese encoders are statically unrolled T times, so the "
              f"graph grows linearly with T (slow compile/save/load). "
              f"Consider T<=8, or sub-sample scenes to fewer steps.")
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

    bn_seq = ops.stack(step_bottlenecks, axis=1)        # (B,T,h,w,D)
    traj, pooled = TemporalTrajectoryEncoder(
        dims[-1], T, num_heads=cfg.temporal_heads,
        num_blocks=cfg.temporal_blocks, dropout=cfg.temporal_dropout,
        max_years=cfg.max_time_gap_years,
        name="temporal_encoder")([bn_seq, dates_in])

    gap = layers.GlobalAveragePooling2D(name="ctx_gap")(ref_fused[-1])
    ctx = layers.Dense(dims[-1], activation="gelu", name="ctx_fc1")(gap)
    ctx = layers.Dense(dims[-1], name="ctx_fc2")(ctx + pooled)
    bn = layers.Conv2D(dims[-1], 1, name="traj_proj")(traj)
    bn = layers.add([ref_fused[-1], bn], name="bn_add")
    bn = layers.LayerNormalization(epsilon=1e-5, name="bn_norm")(bn)
    bn = FiLM(dims[-1], name="bn_film")([bn, ctx])

    # ---- main decoder on the reference (last) step: full skip pathway ----
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

    # ---- change decoder: first-vs-last diffs + FiLM(dt) + trajectory ----
    dt = FlattenDt(name="dt_flatten")(dates_in[:, -1:] - dates_in[:, :1])
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
    while size < ps:                                    # always reach ps
        wd = cw[min(i + 1, len(cw) - 1)]
        d = layers.UpSampling2D(interpolation="bilinear",
                                name=f"change_up{i}")(d)
        size *= 2
        li = L - 2 - i
        if li >= 0:                                     # high-res diff skips
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
# 12. LOSSES (V8-1: all six were referenced but undefined in v7)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FocalDiceLoss(tf.keras.losses.Loss):
    """Multi-class focal CE + soft Dice on one-hot targets.
    Ignored pixels are all-zero rows (one_hot of -1)."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05,
                 eps=1e-6, name="focal_dice", **kw):
        super().__init__(name=name, reduction="none", **kw)
        self.gamma = float(gamma)
        self.ls = float(label_smoothing)
        self.eps = float(eps)
        self.alpha = None if alpha is None else [float(a) for a in alpha]

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        valid = tf.reduce_sum(y, -1, keepdims=True) > 0.0
        m = tf.cast(valid, tf.float32)
        n = tf.reduce_sum(m) + 1e-6
        y = tf.where(valid, y, tf.zeros_like(y))
        if self.ls > 0:
            Cc = tf.cast(tf.shape(y)[-1], tf.float32)
            y = (y * (1.0 - self.ls) + self.ls / Cc) * m
        p = tf.clip_by_value(p, 1e-6, 1.0)
        ce = -tf.reduce_sum(y * tf.math.log(p), -1, keepdims=True)
        pt = tf.exp(-ce)
        focal = tf.pow(1.0 - pt, self.gamma) * ce
        if self.alpha is not None:
            a = tf.constant(self.alpha, tf.float32)
            focal = focal * tf.reduce_sum(y * a, -1, keepdims=True)
        focal = tf.reduce_sum(focal * m) / n
        inter = tf.reduce_sum(y * p * m, axis=[1, 2])
        denom = tf.reduce_sum((y + p) * m, axis=[1, 2])
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return focal + tf.reduce_mean(dice)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma,
                    "label_smoothing": self.ls, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedBinaryFocal(tf.keras.losses.Loss):
    """Binary focal loss; targets in {0,1}, -1 = ignore."""

    def __init__(self, gamma=2.0, pos_weight=2.0, name="masked_binary_focal",
                 **kw):
        super().__init__(name=name, reduction="none", **kw)
        self.gamma = float(gamma)
        self.pos_weight = float(pos_weight)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        m = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
        ce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        pt = tf.where(y > 0.5, p, 1.0 - p)
        fl = tf.pow(1.0 - pt, self.gamma) * ce
        w = tf.where(y > 0.5, self.pos_weight, 1.0)
        return tf.reduce_sum(fl * w * m) / (tf.reduce_sum(m) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "pos_weight": self.pos_weight})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedBCE(tf.keras.losses.Loss):
    """Plain masked BCE (quality / boundary heads); -1 = ignore."""

    def __init__(self, name="masked_bce", **kw):
        super().__init__(name=name, reduction="none", **kw)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        m = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
        bce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        return tf.reduce_sum(bce * m) / (tf.reduce_sum(m) + 1e-6)


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedMAELoss(tf.keras.losses.Loss):
    """Masked MAE (change_year head); sentinel < -900 = ignore."""

    def __init__(self, sentinel=-900.0, name="masked_mae", **kw):
        super().__init__(name=name, reduction="none", **kw)
        self.sentinel = float(sentinel)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        m = tf.cast(y > self.sentinel, tf.float32)
        p = tf.clip_by_value(p, 0.0, 1.0)      # head is linear; clamp to range
        return tf.reduce_sum(tf.abs(y - p) * m) / (tf.reduce_sum(m) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg["sentinel"] = self.sentinel
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonNLLLoss(tf.keras.losses.Loss):
    """Heteroscedastic Gaussian NLL. pred[...,0]=mean, pred[...,1]=log_var.
    Sentinel < -900 = ignore. Targets are pre-scaled by cfg.carbon_scale."""

    def __init__(self, sentinel=-900.0, name="carbon_nll", **kw):
        super().__init__(name=name, reduction="none", **kw)
        self.sentinel = float(sentinel)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        m = tf.cast(y > self.sentinel, tf.float32)
        mean = tf.cast(y_pred[..., 0:1], tf.float32)
        log_var = tf.clip_by_value(tf.cast(y_pred[..., 1:2], tf.float32),
                                   -6.0, 8.0)
        nll = 0.5 * (tf.exp(-log_var) * tf.square(y - mean) + log_var)
        return tf.reduce_sum(nll * m) / (tf.reduce_sum(m) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg["sentinel"] = self.sentinel
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedHuber(tf.keras.losses.Loss):
    """Masked Huber (eco / risk heads); sentinel < -900 = ignore."""

    def __init__(self, delta=1.0, sentinel=-900.0, name="masked_huber", **kw):
        super().__init__(name=name, reduction="none", **kw)
        self.delta = float(delta)
        self.sentinel = float(sentinel)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        m = tf.cast(y > self.sentinel, tf.float32)
        err = tf.abs(y - p)
        hub = tf.where(err <= self.delta, 0.5 * tf.square(err),
                       self.delta * (err - 0.5 * self.delta))
        return tf.reduce_sum(hub * m) / (tf.reduce_sum(m) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"delta": self.delta, "sentinel": self.sentinel})
        return cfg


# ==============================================================================
# 12b. METRICS (V8-2: MeanIoU added; ChangeF1 / CarbonMAE kept)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class MeanIoU(metrics.Metric):
    """Streaming masked mean-IoU for one-hot targets. Ignored pixels are
    all-zero one-hot rows; classes with zero union are excluded."""

    def __init__(self, num_classes, name="miou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(name="cm", shape=(self.num_classes,
                                                    self.num_classes),
                                  initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        C = self.num_classes
        y = tf.argmax(tf.cast(y_true, tf.float32), axis=-1)
        p = tf.argmax(tf.cast(y_pred, tf.float32), axis=-1)
        valid = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=-1) > 0.0
        y = tf.reshape(y[valid], [-1])
        p = tf.reshape(p[valid], [-1])
        idx = tf.cast(y, tf.int32) * C + tf.cast(p, tf.int32)
        cm = tf.cast(tf.math.bincount(idx, minlength=C * C, maxlength=C * C),
                     tf.float32)
        self.cm.assign_add(tf.reshape(cm, [C, C]))

    def result(self):
        d = tf.linalg.diag_part(self.cm)
        u = tf.reduce_sum(self.cm, 0) + tf.reduce_sum(self.cm, 1) - d
        valid = u > 0.0
        iou = tf.where(valid, d / tf.maximum(u, 1.0), tf.zeros_like(d))
        n = tf.reduce_sum(tf.cast(valid, tf.float32))
        return tf.reduce_sum(iou) / tf.maximum(n, 1.0)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

    def get_config(self):
        cfg = super().get_config()
        cfg["num_classes"] = self.num_classes
        return cfg


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
# 13. TRAINER
# ==============================================================================
HEAD_LOSS_MAP = {
    "seg": "focal_dice", "species": "focal_dice",
    "change": "binary_focal", "recovery": "binary_focal", "damage": "binary_focal",
    "quality": "bce", "boundary": "bce",
    "change_year": "masked_mae", "carbon": "carbon_nll",
    "eco": "huber", "risk": "huber",
}


class EcoMonitorTrainer(tf.keras.Model):
    """Wraps the functional net. Owns per-task losses, learnable Kendall
    log-variance task weighting, and explicit metrics."""

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
                    alpha=alpha if h == "seg" else None,
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
                "grad_accum=1.") from e
    else:
        opt = optimizers.AdamW(**kwargs)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        try:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        except Exception:
            pass
    return opt


# ==============================================================================
# 14. EMA CALLBACK
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
# 15. LR SCHEDULE / PLOTS / EVAL / CALIBRATION
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


def calibrate_change_conformal(net, val_ds, cfg, max_batches=15):
    """Split-conformal change threshold + calibration certificate (V8-6).
    NOTE: marginal coverage holds only if calibration and deployment data
    are exchangeable (same region, season, sensor stack, date range)."""
    if "change_head" not in net.output_names:
        return None
    alpha = float(cfg.conformal_alpha)
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
    if n < 1000:
        print(f"  [conformal] WARNING: n={n} positives is small; the "
              f"finite-sample guarantee is loose. Add more val scenes.")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    q = float(np.quantile(s, level))
    thr = float(1.0 - q)
    cfg.change_conformal_q = thr
    cert = {
        "type": "split-conformal change alert calibration",
        "alpha": alpha, "threshold": thr, "n_positive_pixels": n,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caveat": ("Marginal coverage (1-alpha) on truly-changed pixels "
                   "holds only under exchangeability of calibration and "
                   "deployment data. Recalibrate when region, season, "
                   "sensor stack or date range changes."),
    }
    with open(os.path.join(WORK_DIR, "calibration_certificate.json"), "w") as f:
        json.dump(cert, f, indent=2)
    print(f"  [conformal] change alert threshold = {thr:.3f} "
          f"(alpha={alpha}, n={n:,} positive px); certificate saved.")
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
# 16. SELF-SUPERVISED PRETRAINING (temporal SimMIM)
# ==============================================================================
def build_pretrain_model(cfg, s1_stats, s2_stats):
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
    while size < ps // 4:
        x = layers.UpSampling2D(interpolation="bilinear", name=f"pt_up{wi}")(x)
        x = conv_block(x, widths[min(wi, len(widths) - 1)], use_se=False,
                       name=f"pt_c{wi}")
        size *= 2
        wi += 1
    rec = layers.Conv2D(C1 + 10, 1, padding="same", dtype="float32",
                        name="recon_head")(x)
    return models.Model([s1_seq, s2_seq, scl_seq, aux_seq], rec,
                        name="MangrovePretrainer_v8")


def _masked_block_seq(x, ratio, block):
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    ps = tf.shape(x)[2]
    g = tf.maximum(ps // block, 1)
    cells = tf.cast(tf.random.uniform((B, T, g, g, 1)) < ratio, tf.float32)
    mask = tf.image.resize(tf.reshape(cells, [B * T, g, g, 1]), [ps, ps],
                           method="nearest")
    mask = tf.reshape(mask, [B, T, ps, ps, 1])
    keep = 1.0 - mask
    return x * keep, keep


def pretrain(cfg=CFG):
    print("\n=== v8 SELF-SUPERVISED PRETRAINING (temporal SimMIM) ===")
    scenes = discover_scenes(cfg.train_dir)
    assert scenes, "pretrain: set cfg.train_dir (scene folders or legacy NPZs)"
    fp, nf = dataset_fingerprint(scenes)
    print(f"  dataset fingerprint: {fp} ({nf} files)")
    s1m, s1v, s2m, s2v, _ = load_or_compute_stats(scenes, cfg)
    batch = cfg.batch_per_replica * N_REPLICAS
    ds = (tf.data.Dataset.from_generator(
            lambda: patch_generator(scenes, cfg, False, cfg.max_train_patches),
            output_signature=output_signature(cfg))
          .shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)
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

        out = os.path.join(WORK_DIR, "pretrain_encoder_v8.weights.h5")
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
    cfg.check()
    print(f"\n=== MANGROVE ECO-MONITOR v{VERSION} - TRAINING ===\n")
    train_scenes = discover_scenes(cfg.train_dir)
    val_scenes = discover_scenes(cfg.val_dir)
    assert train_scenes and val_scenes, \
        "Set cfg.train_dir / cfg.val_dir (scene folders or legacy NPZs)"
    validate_heads_vs_data(train_scenes, cfg)
    fp, nf = dataset_fingerprint(train_scenes)               # V8-12
    print(f"train scenes: {len(train_scenes)} | val scenes: {len(val_scenes)} "
          f"| T={cfg.temporal_steps}")
    print(f"  dataset fingerprint: {fp} ({nf} files)")

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

        monitor = "val_seg_miou" if "seg" in cfg.active_heads else "val_loss"
        mode = "max" if "miou" in monitor else "min"
        ema = EMACallback(cfg.ema_decay)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=15,
                                             restore_best_weights=True,
                                             verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(WORK_DIR, "best_v8.weights.h5"), monitor,
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
    # V8-7: post-training steps are fault-isolated from the training result
    try:
        calibrate_temperature(net, ds_val, cfg)
        calibrate_change_conformal(net, ds_val, cfg)
    except Exception as e:
        print(f"  [calibration] FAILED (non-fatal, model still saved): {e}")
    model_path = os.path.join(WORK_DIR, "mangrove_ecomonitor_v8.keras")
    net.save(model_path)
    cfg.save(os.path.join(WORK_DIR, "config_v8.json"))
    with open(os.path.join(WORK_DIR, "dataset_version.json"), "w") as f:
        json.dump({"fingerprint": fp, "n_files": nf,
                   "train_dir": cfg.train_dir,
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())}, f, indent=2)
    plot_training_history(history)
    try:
        if "seg" in cfg.active_heads:
            evaluate_per_class(net, ds_val, cfg)
        visualize_predictions(net, ds_val, cfg)
    except Exception as e:
        print(f"  [eval] FAILED (non-fatal): {e}")
    print(f"\nModel saved: {model_path}")
    print(f"Config saved: {os.path.join(WORK_DIR, 'config_v8.json')}")
    return net


# ==============================================================================
# 18. TILED INFERENCE (Hann blending, per-pass TTA weights, MC-dropout,
#     SAR-only fallback)
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
    h = np.hanning(ps).astype(np.float32)
    h = np.maximum(h, floor)
    return np.outer(h, h)


def predict_scene(net, s1, s2=None, scl=None, aux=None, t2=None, cfg=CFG,
                  batch_size=None, tta=None, overlap=None, sar_only=False,
                  dates=None, return_tiles=False):
    """Production inference on a unified scene.

    Call either with a scene dict (load_scene / scene_from_arrays output) as
    `s1`, or pair-style with arrays: predict_scene(net, s1, s2, scl, aux, t2).
    return_tiles=True additionally returns (padded_scene, coords) for the
    explainability / review-queue stages.
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
    if sar_only:
        scene["s2_seq"] = np.zeros_like(scene["s2_seq"])

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
                try:
                    if mc > 1:
                        preds = _preds_to_dict(net, net(fb, training=(rep > 0)))
                        preds = {k2: v.numpy() for k2, v in preds.items()}
                    else:
                        preds = _preds_to_dict(net, net.predict(
                            fb, verbose=0, batch_size=len(chunk)))
                except tf.errors.ResourceExhaustedError:
                    print("  [infer] OOM on a tile batch; retrying singly.")
                    preds = None
                    for single in range(len(chunk)):
                        one = {k2: v[single:single + 1]
                               for k2, v in fb.items()}
                        p1 = _preds_to_dict(net, net.predict(
                            one, verbose=0, batch_size=1))
                        if preds is None:
                            preds = {k2: np.concatenate(
                                [p1[k2]] + [np.zeros_like(p1[k2])] *
                                (len(chunk) - 1)) for k2 in p1}
                        else:
                            for k2 in preds:
                                preds[k2][single:single + 1] = p1[k2]
                for name in acc:
                    pr = inv(preds[name])
                    pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
                    for n_i, (i, j) in enumerate(chunk):
                        sl = np.s_[i:i + ps, j:j + ps]
                        acc[name][sl] += pr[n_i] * kernel[..., None]
                        if mc_sq is not None:
                            mc_sq[name][sl] += (pr[n_i] ** 2) * kernel[..., None]
            for i, j in chunk:
                weight[i:i + ps, j:j + ps] += kernel

    weight = np.maximum(weight[..., None], 1e-6)
    out = {n: acc[n] / weight for n in acc}
    
    # FIX: Renormalize seg_head to guarantee valid probability distribution 
    # even if NaN-to-zero corruption deflated the blended sum.
    if "seg_head" in out:
        s = out["seg_head"].sum(-1, keepdims=True)
        out["seg_head"] = out["seg_head"] / np.maximum(s, 1e-6)

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
        # FIX: Add visibility for NaN corruption in change head accumulator
        cp = out["change_head"][..., 0].astype(np.float32)
        n_nan = int(np.isnan(acc["change_head"]).sum())
        if n_nan > 0:
            print(f"  [predict_scene] WARNING: {n_nan} NaN values in change_head "
                  f"accumulator before blending -- results may be corrupted.")
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
    if return_tiles:
        return res, scene, coords
    return res


# ==============================================================================
# 18b. EXPLAINABILITY: GRAD-CAM (V8-9)
# ==============================================================================
def grad_cam_tiles(net, scene, coords, cfg, class_idx=None, layer_name=None,
                   n_tiles=None, out_size=256):
    """Class-discriminative localization from the FiLM-conditioned bottleneck.
    Returns list of (i, j, cam01, s2_rgb) tuples for the highest-activation
    tiles. Used for regulatory audit: 'why was this pixel called mangrove?'"""
    class_idx = cfg.gradcam_class if class_idx is None else int(class_idx)
    layer_name = layer_name or cfg.gradcam_layer
    n_tiles = n_tiles or cfg.gradcam_tiles
    ps = cfg.patch_size
    try:
        target = net.get_layer(layer_name).output
        logits = net.get_layer("seg_logits").output
    except ValueError as e:
        print(f"  [gradcam] layer lookup failed ({e}); skipping.")
        return []
    gm = tf.keras.Model(net.inputs, [logits, target])

    picks = coords[:max(1, n_tiles)]
    results = []
    for (i, j) in picks:
        batch = {
            "s1_seq": scene["s1_seq"][:, i:i + ps, j:j + ps][None],
            "s2_seq": scene["s2_seq"][:, i:i + ps, j:j + ps][None],
            "scl_seq": scene["scl_seq"][:, i:i + ps, j:j + ps][None],
            "aux_seq": scene["aux_seq"][:, i:i + ps, j:j + ps][None],
            "dates": scene["dates"][None],
        }
        with tf.GradientTape() as tape:
            lg, feats = gm(batch, training=False)
            cls = min(class_idx, lg.shape[-1] - 1)
            score = tf.reduce_sum(lg[..., cls])
        grads = tape.gradient(score, feats)
        if grads is None:
            continue
        wts = tf.reduce_mean(grads, axis=[1, 2], keepdims=True)
        cam = tf.reduce_sum(tf.nn.relu(tf.cast(wts, tf.float32) * tf.cast(feats, tf.float32)), -1)[0]
        cam = cam - tf.reduce_min(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-6)
        cam_up = tf.image.resize(cam[..., None], [out_size, out_size],
                                 method="bilinear")[..., 0].numpy()
        rgb = _stretch(scene["s2_seq"][-1, i:i + ps, j:j + ps, [2, 1, 0]])
        results.append((i, j, cam_up, rgb))
    print(f"  [gradcam] computed {len(results)} heatmaps "
          f"(class={class_idx}, layer='{layer_name}')")
    return results


def save_gradcam_figure(cams, out_path, cfg):
    if not cams:
        return None
    n = len(cams)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    axes = np.atleast_2d(axes)
    cname = CLASS_NAMES[min(cfg.gradcam_class, len(CLASS_NAMES) - 1)]
    for r, (i, j, cam, rgb) in enumerate(cams):
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"tile ({i},{j}) S2 RGB")
        axes[r, 1].imshow(cam, cmap="inferno", vmin=0, vmax=1)
        axes[r, 1].set_title(f"Grad-CAM: '{cname}'")
        axes[r, 2].imshow(rgb)
        axes[r, 2].imshow(cam, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        axes[r, 2].set_title("overlay")
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")
    return out_path


# ==============================================================================
# 18c. ACTIVE-LEARNING REVIEW QUEUE (V8-10)
# ==============================================================================
def build_review_queue(products, scene, cfg, out_dir):
    """Rank scene tiles by labeling value: low pixel usability, change prob
    near the decision boundary, or high epistemic variance. Export inputs +
    predictions + reason so a human can relabel and feed back to training."""
    rdir = os.path.join(out_dir, "review_queue")
    os.makedirs(rdir, exist_ok=True)
    h, w = products.get("classes", products.get("change_prob")).shape[:2] \
        if ("classes" in products or "change_prob" in products) else (0, 0)
    if h == 0:
        print("  [review] no usable products; queue empty.")
        return []
    t = int(cfg.review_tile)
    thr = (cfg.change_conformal_q if cfg.change_conformal_q > 0
           else cfg.change_threshold)
    cand = []
    for i in range(0, h - t + 1, t):
        for j in range(0, w - t + 1, t):
            score, reasons = 0.0, []
            if "quality_prob" in products:
                q = float(np.mean(products["quality_prob"][i:i + t, j:j + t]))
                if q < cfg.review_low_quality:
                    score += (cfg.review_low_quality - q) * 2.0
                    reasons.append(f"low_quality({q:.2f})")
            if "change_prob" in products:
                cp = products["change_prob"][i:i + t, j:j + t]
                border = float(np.mean(np.abs(cp - thr) <
                                       cfg.review_change_margin))
                if border > 0.05:
                    score += border
                    reasons.append(f"borderline_change({border:.2f})")
            if "seg_epistemic_var" in products:
                ev = float(np.mean(
                    products["seg_epistemic_var"][i:i + t, j:j + t]))
                if ev > 0.02:
                    score += ev * 10.0
                    reasons.append(f"epistemic({ev:.3f})")
            if score > 0:
                cand.append((score, i, j, reasons))
    cand.sort(key=lambda c: -c[0])
    cand = cand[:int(cfg.review_queue_max)]
    manifest = []
    pad = ((0, 0), (0, max(0, t - h)), (0, max(0, t - w)), (0, 0))
    for idx, (score, i, j, reasons) in enumerate(cand):
        rec = {
            "score": round(score, 4), "tile": [i, j], "reasons": reasons,
            "dates": scene["dates"].tolist(),
        }
        payload = {}
        for k in ("s1_seq", "s2_seq", "scl_seq", "aux_seq"):
            arr = scene[k]
            hh, ww = arr.shape[1:3]
            if i + t <= hh and j + t <= ww:
                payload[k] = arr[:, i:i + t, j:j + t]
        if not payload:
            continue
        payload["dates"] = scene["dates"]
        for key in ("classes", "change_prob", "quality_prob"):
            if key in products:
                v = products[key]
                payload[f"pred_{key}"] = v[i:i + t, j:j + t]
        fname = f"review_{idx:04d}.npz"
        np.savez_compressed(os.path.join(rdir, fname), **payload)
        rec["file"] = fname
        manifest.append(rec)
    with open(os.path.join(rdir, "manifest.json"), "w") as f:
        json.dump({"version": VERSION,
                   "criteria": {"low_quality_below": cfg.review_low_quality,
                                "change_margin": cfg.review_change_margin,
                                "change_threshold": thr},
                   "tiles": manifest}, f, indent=2)
    print(f"  [review] queued {len(manifest)} tiles -> {rdir}")
    return manifest


# ==============================================================================
# 18d. POST-TEMPORAL HMM SMOOTHING (V8-11)
# ==============================================================================
def _viterbi_chunk(lp, logA):
    """lp: (T,N,C) log-probs; logA: (C,C) log transition. Returns (T,N)."""
    T, N, C = lp.shape
    delta = lp[0]
    backs = np.empty((T - 1, N, C), np.uint8)
    for t in range(1, T):
        cand = delta[:, :, None] + logA[None, :, :]       # (N, Cprev, Ccur)
        backs[t - 1] = cand.argmax(1).astype(np.uint8)
        delta = lp[t] + cand.max(1)
    z = delta.argmax(-1)
    out = np.empty((T, N), np.uint8)
    out[T - 1] = z
    for t in range(T - 2, -1, -1):
        z = backs[t][np.arange(N), z]
        out[t] = z
    return out


def smooth_class_series(prob_series, stay=0.85, chunk=200_000):
    """Viterbi smoothing of a per-date stack of class-probability maps.
    prob_series: list of (H,W,C) float arrays, time-ordered. Returns a list
    of (H,W) uint8 class maps with implausible flicker suppressed."""
    if len(prob_series) < 2:
        return [p.argmax(-1).astype(np.uint8) for p in prob_series]
    P = np.stack([np.clip(p, 1e-6, 1.0) for p in prob_series]).astype(np.float32)
    T, H, W, C = P.shape
    logP = np.log(P).reshape(T, H * W, C)
    logA = np.full((C, C), math.log(max(1e-6, (1.0 - stay) / max(C - 1, 1))),
                   np.float32)
    np.fill_diagonal(logA, math.log(stay))
    out = np.empty((T, H * W), np.uint8)
    for s in range(0, H * W, chunk):
        e = min(s + chunk, H * W)
        out[:, s:e] = _viterbi_chunk(logP[:, s:e], logA)
    print(f"  [smooth] Viterbi-smoothed {T} steps over {H * W:,} px "
          f"(stay={stay})")
    return [out[t].reshape(H, W) for t in range(T)]


def smooth_mode(prob_dir, cfg, out_path=None):
    """CLI entry: read per-date seg_probs from NPZs in prob_dir (files must
    contain a 'seg_probs' array; date parsed from filename), smooth, save."""
    files = sorted(glob.glob(os.path.join(prob_dir, "*.npz")),
                   key=lambda f: _date_from_name(f))
    probs = []
    for f in files:
        with np.load(f) as d:
            if "seg_probs" in d.files:
                probs.append(d["seg_probs"].astype(np.float32))
    if len(probs) < 2:
        raise RuntimeError(f"smooth: need >=2 NPZs with 'seg_probs' in "
                           f"{prob_dir}, found {len(probs)}")
    smoothed = smooth_class_series(probs, stay=cfg.smooth_stay)
    out_path = out_path or os.path.join(prob_dir, "smoothed_classes.npz")
    np.savez_compressed(out_path,
                        **{f"classes_t{t}": c for t, c in enumerate(smoothed)})
    print(f"  saved {out_path}")
    return smoothed


# ==============================================================================
# 19. EXPORT: COG GeoTIFFs + STAC item + GeoJSON alerts + MRV summary
#     + dashboard + review queue + gradcam
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
        "collection": "mangrove-ecomonitor-v8",
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
                        sar_only=False, gradcam=False, review=True):
    os.makedirs(out_dir, exist_ok=True)
    desc = ("folder", scene_path) if os.path.isdir(scene_path) \
        else ("npz", scene_path)
    scene = load_scene(desc, cfg)
    products, scene_pad, coords = predict_scene(
        net, scene, cfg=cfg, sar_only=sar_only, return_tiles=True)

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

    # V8-9: explainability
    if gradcam:
        try:
            cams = grad_cam_tiles(net, scene_pad, coords, cfg)
            save_gradcam_figure(cams, os.path.join(out_dir, "gradcam.png"), cfg)
        except Exception as e:
            print(f"  [gradcam] failed (non-fatal): {e}")

    # V8-10: active-learning queue
    if review:
        try:
            build_review_queue(products, scene_pad, cfg, out_dir)
        except Exception as e:
            print(f"  [review] failed (non-fatal): {e}")

    fp, nf = dataset_fingerprint([desc])
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_path, "reference": ref_tif_path,
                   "version": VERSION, "active_heads": list(cfg.active_heads),
                   "sar_only": sar_only, "dates": scene["dates"].tolist(),
                   "seg_temperature": cfg.seg_temperature,
                   "change_conformal_q": cfg.change_conformal_q,
                   "dataset_fingerprint": fp, "dataset_files": nf},
                  f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products


# ==============================================================================
# 19b. EDGE EXPORT: TFLITE (+ optional int8) and ONNX (V8-8)
# ==============================================================================
def export_tflite(cfg=CFG, keras_path=None, out_dir=None):
    """Convert the trained .keras model to TFLite. Swin uses tf.roll /
    l2_normalize etc., so SELECT_TF_OPS is enabled. If quantization is
    requested, a small representative dataset is streamed from train_dir.
    Any failure prints actionable guidance instead of a raw traceback."""
    keras_path = keras_path or os.path.join(
        WORK_DIR, "mangrove_ecomonitor_v8.keras")
    out_dir = out_dir or WORK_DIR
    print(f"\n=== EDGE EXPORT (TFLite{' int8' if cfg.tflite_quantize else ''}"
          f"{' + ONNX' if cfg.try_onnx else ''}) ===")
    try:
        net = tf.keras.models.load_model(keras_path)
    except Exception as e:
        print(f"  [tflite] cannot load {keras_path}: {e}")
        return None

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(net)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        if cfg.tflite_quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            scenes = discover_scenes(cfg.train_dir)

            def rep_ds():
                n = 0
                for patch in patch_generator(scenes, cfg, False, 32):
                    n += 1
                    yield {k: patch[k][None] for k in
                           ("s1_seq", "s2_seq", "scl_seq", "aux_seq", "dates")}
                    if n >= 32:
                        return
            if scenes:
                converter.representative_dataset = rep_ds
        buf = converter.convert()
        name = ("mangrove_v8_int8.tflite" if cfg.tflite_quantize
                else "mangrove_v8.tflite")
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(buf)
        print(f"  saved {path} ({len(buf) / 1e6:.1f} MB)")
        print("  NOTE: SELECT_TF_OPS requires the TF Lite Flex delegate on "
              "device. For pure-builtin edge deployment, replace tf.roll/"
              "l2_normalize with builtin equivalents and retrain.")
    except Exception as e:
        print(f"  [tflite] conversion failed: {str(e)[:300]}")
        print("  Hints: (1) mixed_float16 heads are fine, but some Swin ops "
              "lack TFLite kernels; (2) try cfg.tflite_quantize=False; "
              "(3) fall back to SavedModel export + on-device TF.")
        path = None

    if cfg.try_onnx:
        try:
            import tf2onnx
            spec = tuple(tf.TensorSpec(s.shape, s.dtype, s.name)
                         for s in net.inputs)
            model_proto, _ = tf2onnx.convert.from_keras(
                net, input_signature=spec, opset=15)
            opath = os.path.join(out_dir, "mangrove_v8.onnx")
            with open(opath, "wb") as f:
                f.write(model_proto.SerializeToString())
            print(f"  saved {opath}")
        except ImportError:
            print("  [onnx] tf2onnx not installed; skipping "
                  "(pip install tf2onnx).")
        except Exception as e:
            print(f"  [onnx] conversion failed (non-fatal): {str(e)[:200]}")
    return path


# ==============================================================================
# 20. SMOKE TEST (every default head + TTA + SAR-only + reload + gradcam
#     + smoothing + losses/metrics)
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
    cfg.check()
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
    print(f"  losses wired: {sorted(trainer.loss_fns.keys())} | "
          f"metrics: {[m.name for m in trainer.metric_objs.values()]}")
    trainer.fit(ds_tr, validation_data=ds_va, epochs=2,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()])

    model_path = os.path.join(WORK_DIR, "smoke_v8.keras")
    net.save(model_path)
    m2 = tf.keras.models.load_model(model_path)
    print("  reload OK - all custom objects deserialized "
          "(layers + losses + metrics).")

    H, W, A = 300, 380, cfg.aux_channels
    rng = np.random.default_rng(0)
    s1 = rng.uniform(-30, 0, (H, W, 2)).astype(np.float32)
    s2 = rng.uniform(0, 0.5, (H, W, 10)).astype(np.float32)
    aux = rng.uniform(-2, 2, (H, W, A)).astype(np.float32)
    t2 = {"s1": s1 + 0.5, "s2": np.clip(s2 + 0.02, 0, 1),
          "scl": np.full((H, W), 4, np.float32), "aux": aux + 0.1}
    products, scene_pad, coords = predict_scene(
        m2, s1, s2, aux=aux, t2=t2, cfg=cfg, batch_size=2, tta=True,
        dates=(2022.0, 2024.0), return_tiles=True)
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

    # V8-9 gradcam sanity
    cams = grad_cam_tiles(m2, scene_pad, coords, cfg, n_tiles=2)
    print(f"  gradcam OK: {len(cams)} heatmaps")
    # V8-10 review queue sanity
    rq = build_review_queue(products, scene_pad, cfg, WORK_DIR)
    print(f"  review queue OK: {len(rq)} tiles")
    # V8-11 smoothing sanity
    pseq = [products["seg_probs"],
            np.roll(products["seg_probs"], 1, axis=-1)]
    sm = smooth_class_series(pseq, stay=cfg.smooth_stay)
    print(f"  temporal smoothing OK: {len(sm)} maps, dtype={sm[0].dtype}")
    print("=== SMOKE TEST PASSED ===")
    return True


# ==============================================================================
# 21. CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=f"Mangrove Eco-Monitor v{VERSION}")
    ap.add_argument("mode", choices=["smoke", "pretrain", "train", "eval",
                                     "export", "export-tflite", "smooth"],
                    nargs="?", default="smoke")
    ap.add_argument("scene", nargs="?",
                    help="scene folder or legacy NPZ (export) / prob dir (smooth)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (export)")
    ap.add_argument("out", nargs="?", help="output dir (export/export-tflite)")
    ap.add_argument("--config", help="path to config_v8.json")
    ap.add_argument("--sar-only", action="store_true",
                    help="optical-free inference (cloudy scenes)")
    ap.add_argument("--gradcam", action="store_true",
                    help="write Grad-CAM explainability figure on export")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the active-learning review queue on export")
    ap.add_argument("--int8", action="store_true",
                    help="int8 dynamic quantization for export-tflite")
    ap.add_argument("--pretrain", dest="pretrain_w", default="",
                    help="encoder weights to fine-tune from")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.pretrain_w:
        cfg.pretrain_weights = args.pretrain_w
    if args.int8:
        cfg.tflite_quantize = True
    resolve_hardware(cfg)

    if args.mode == "smoke":
        smoke_test(cfg)
    elif args.mode == "pretrain":
        pretrain(cfg)
    elif args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        net = tf.keras.models.load_model(
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v8.keras"))
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
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v8.keras"))
        export_all_products(net, args.scene, args.ref, args.out, cfg,
                            sar_only=args.sar_only, gradcam=args.gradcam,
                            review=not args.no_review)
    elif args.mode == "export-tflite":
        export_tflite(cfg, out_dir=args.out or WORK_DIR)
    elif args.mode == "smooth":
        if not args.scene:
            ap.error("smooth requires <prob_dir> containing per-date NPZs "
                     "with a 'seg_probs' array")
        smooth_mode(args.scene, cfg)


if __name__ == "__main__":
    main()