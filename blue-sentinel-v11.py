"""
================================================================================
 BLUE SENTINEL APEX v11  "MANGROVE ECO-MONITOR — UNIFIED SPATIO-TEMPORAL SSM"
 Merge of v8 (TF/Keras all-in-one GIS suite) + v9 (PyTorch Mamba core).

 REQUIRES: Python >= 3.10, PyTorch >= 2.1 (Linux/WSL2 recommended).
 Optional:  mamba-ssm (CUDA scan kernels; exact pure-torch fallback otherwise),
            rasterio (GeoTIFF/STAC/GeoJSON export; NPZ fallback otherwise),
            onnx + onnx2tf (edge export), matplotlib (dashboards/gradcam png).

 UNIFIED 25-CHANNEL INPUT CUBE  (T, 25, H, W)
   0-9   Sentinel-2 SR (10)          10    SCL
   11-14 Sentinel-1 vv,vh,vv_std,vh_std (dB)
   15-16 ALOS PALSAR-2 HH,HV (dB)    17    CHIRPS precip (mm)
   18    tide-model inundation (mm)  19-24 aux(6): elev,slope,ndvi_trend,
                                         flood_freq,lst_trend,s1_trend
 TARGETS (kept OUTSIDE the cube — v11 schema fix):
   label(255=ignore) species(255) change/recovery/debris(255) carbon/eco/
   risk/change_year(-999=ignore) gedi(NaN) depth(NaN; falls back to the
   inundation prior as pseudo-label, v9 behaviour).

 SCENE SCHEMAS (both auto-detected):
   A) folder of dated step NPZs (v8/v9 superset keys: s2,scl,s1[,s1_std],
      lband,precip,inundation,aux,gedi,depth,date; labels on the LAST step:
      label,carbon,eco,change,risk,species,debris,recovery,change_year)
   B) single cube NPZ (v9): cube (T,19,H,W) + optional dates — remapped.

 HEADS (14): seg(+deep) species change change_year recovery damage quality
   boundary carbon eco risk hydro(mu,var) canopy

 MODES: python blue_sentinel_v11.py smoke | pretrain | train | eval |
        export | export-edge | smooth

  v11.4 (dataset release):
    - clearer no-scenes assert in train() (points at the gee_dataset_builder
      pack step for raw GEE GeoTIFF exports)
  v11.3 (GPU release):
    - FIX crash on GPU: calibrate_change_conformal called .numpy() on a CUDA
      tensor -> the smoke died at step 6/9 on Kaggle T4s (CPU runs masked it)
    - NEW: mirror multi-GPU (DataParallel = PyTorch MirroredStrategy):
      auto-wraps when >1 CUDA GPU is visible (cfg.mirror_gpus=True), giving
      2x T4 sessions the combined 32 GB VRAM. EMA / checkpoints / gradcam /
      set_phase are unwrap-aware, so checkpoints stay single-GPU-portable
      (no 'module.' prefix) and eval/export mirror automatically. Default
      batch_size scales with GPU count (2x T4 -> 4, i.e. 2/GPU) unless
      --batch-size is given; inference batches at least one tile per GPU.
    - smoke now exercises the mirror path on multi-GPU machines
  v11.2 (Kaggle bugfix release):
    - FIX (CRITICAL) inference: MC-dropout repetitions inflated every
      continuous product by mc_samples (hydro/carbon/canopy/change_year/
      risk/eco/quality/recovery/damage); tile weights now accumulate per
      repetition, so products are the true MC mean and the epistemic
      variance formula is exact. Caught by a new smoke invariant.
    - FIX crash: dirty labels >= num_classes crashed bincount in
      compute_stats/_patch_weight (now truncated, like the confusion matrix)
    - FIX: SimMIM pretrain target now at the recon decoder's true
      resolution (avg-pool 2; was pool 4 + upsample = blurred target)
    - FIX: export falls back to NPZ / skips GeoJSON when the reference
      GeoTIFF is missing or unreadable (was a hard crash)
    - FIX: GradScaler via torch.amp with torch.cuda.amp fallback (torch
      < 2.3); explicit torch.utils.checkpoint import
  v11.1 (Kaggle-tuned bugfix release):
   - FIX crash: depth pseudo-label read channel from wrong axis after the
     (C,H,W) transpose -> hydro loss broadcast error on folder/cube scenes
     without 'depth'
   - FIX: post-training calibration no longer clobbers best_v11.pt
     (best-mIoU epoch) with final-epoch weights -> final_calibrated_v11.pt
   - FIX: photometric contrast used mean over wrong axes
   - FIX: class alpha from compute_stats is now actually passed to the
     focal-dice loss
   - FIX: norm-stats cache keyed by dataset fingerprint (stale-stats trap)
   - FIX: confusion matrix drops out-of-range labels instead of crashing
   - FIX: precip clip raised 2000 -> 6000 mm (monsoon mangrove sites)
   - FIX: cube-schema GEDI zeros treated as missing (NaN), not 0 m canopy
   - FIX: phase-2 keeps task weighting (scales weighted sums)
   - NEW: pretrain/stats work on unlabelled scenes (require_labels bypass)
   - NEW: --resume from last_v11.pt (Kaggle 12h session cap)
   - NEW: scene-pixel RAM guard at inference (<=30 GB target)
   - NEW: ckpt_mamba grad-checkpointing even with CUDA kernels (16 GB T4)
   - NEW: export also writes seg_probs_<date>.npz so `smooth` can consume it
   - defaults tuned for Kaggle: batch_size=2, infer_batch=2
================================================================================
"""

import os, re, gc, json, glob, math, time, shutil, hashlib, argparse
import platform, warnings, base64
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext

# FIX (v11.4.3): v11.4.2 forced PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# — that run died with "CUDA error: an illegal memory access was encountered"
# under DataParallel (a known failure mode of expandable_segments with the
# per-forward broadcast/free churn on some torch builds), while v11.4.1 without
# it OOM'd cleanly. NOT set by default anymore; opt in via your own env var.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint                     # FIX (v11.2): explicit
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "11.4.3"
MIN_TORCH = (2, 1)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except Exception:
    selective_scan_fn = None
    HAS_MAMBA = False

# ------------------------------ channels -------------------------------------
CH_S2 = slice(0, 10); CH_SCL = 10
CH_S1 = slice(11, 15); CH_LBAND = slice(15, 17)
CH_PRECIP = 17; CH_INUND = 18; CH_AUX = slice(19, 25)
N_CH = 25

CLASS_NAMES = ["Bare Earth/Agri", "Tidal Water", "Perm Flood",
               "Deg. Mangrove", "Dense Mangrove"]
CLASS_COLORS = ["#8B4513", "#00BFFF", "#191970", "#90EE90", "#006400"]


# ==============================================================================
# 1. ENVIRONMENT
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def environment_check():
    ok = True
    if _parse_version(torch.__version__) < MIN_TORCH:
        print(f"[FATAL] PyTorch {torch.__version__} < 2.1"); ok = False
    if platform.system() == "Windows" and not torch.cuda.is_available() \
            and shutil.which("nvidia-smi"):
        print("=" * 78)
        print("[HARDWARE BLOCKER] Windows + NVIDIA driver but no CUDA. Use WSL2:")
        print("  wsl --install -d Ubuntu && pip install torch (cu121+)")
        print("=" * 78)
        ok = False
    return ok


ENV_OK = environment_check()
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v11/"
os.makedirs(WORK_DIR, exist_ok=True)
# FIX: Force PyTorch to utilize all available CPU cores
import multiprocessing
torch.set_num_threads(multiprocessing.cpu_count())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

N_GPUS = torch.cuda.device_count() if DEVICE.type == "cuda" else 0


def unwrap(model):
    """Underlying module (strips nn.DataParallel)."""
    return model.module if isinstance(model, nn.DataParallel) else model


def wrap_mirror(model, enabled=True):
    """Mirror strategy (Keras-speak) == nn.DataParallel: replicate the model
    on every CUDA GPU and split each batch across them. No-op on single-GPU /
    CPU boxes, so all call sites are safe everywhere."""
    if enabled and DEVICE.type == "cuda" and N_GPUS > 1 \
            and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
        print(f"  [gpu] mirror mode ON: DataParallel over {N_GPUS}x "
              f"{torch.cuda.get_device_name(0)} "
              f"({N_GPUS * 16} GB combined VRAM; per-GPU batch = "
              f"batch_size / {N_GPUS})")
    return model


print(f"Blue Sentinel v{VERSION} | torch {torch.__version__} | {DEVICE} | "
      f"bf16={AMP_BF16} | mamba_ssm={HAS_MAMBA}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}"
          + (f" x{N_GPUS} (mirror-ready)" if N_GPUS > 1 else ""))
if not HAS_MAMBA:
    print("  [ssm] pure-PyTorch chunked scan fallback (pip install mamba-ssm "
          "for CUDA kernels)")


# ==============================================================================
# 2. CONFIG
# ==============================================================================
@dataclass
class Config:
    # data
    train_dir: str = "/kaggle/input/mangrove-v11/train/"
    val_dir: str = "/kaggle/input/mangrove-v11/val/"
    test_dir: str = "/kaggle/input/mangrove-v11/test/"
    patch_size: int = 256
    train_stride: int = 128
    max_train_patches: int = 8000
    max_val_patches: int = 1200
    min_valid_frac: float = 0.30
    max_cloud_frac: float = 0.85
    ignore_label: int = 255
    invalid_scl: tuple = (0, 1, 2, 3, 7, 8, 9, 10, 11)
    rare_classes: tuple = (2, 3)
    rare_boost: float = 4.0
    num_classes: int = 5
    num_species: int = 3
    species_names: tuple = ("Avicennia", "Rhizophora", "Other")
    temporal_steps: int = 5
    max_time_gap_years: float = 20.0
    aux_names: tuple = ("elev", "slope", "ndvi_trend", "flood_freq",
                        "lst_trend", "s1_trend")
    aux_scale: tuple = (50.0, 20.0, 0.1, 1.0, 50.0, 1.0)
    # model
    dim: int = 96
    depths: tuple = (2, 2, 2, 2)
    d_state: int = 16
    expand: int = 2
    dt_rank: str = "auto"
    drop_path: float = 0.10
    ssm_chunk: int = 512
    grad_ckpt: bool = True
    change_widths: tuple = (128, 96, 64)
    time_embed_dim: int = 64
    head_dropout: float = 0.10
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.3
    # curriculum
    epochs: int = 100
    phase1_end: int = 25
    phase2_end: int = 60
    phase2_lambda: float = 0.1
    lr: float = 1e-4
    final_lr: float = 1e-6
    weight_decay: float = 5e-2
    batch_size: int = 2                   # Kaggle T4 16 GB safe
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_mamba: bool = True               # ckpt VSS even with CUDA kernels (VRAM)
    mirror_gpus: bool = True              # DataParallel over >1 CUDA GPU (2x T4)
    precip_clip: float = 6000.0           # mm; monsoon sites exceed 2000
    max_scene_pixels: int = 8_000_000     # inference RAM guard (<=30 GB RAM)
    resume: bool = False                  # continue from last_v11.pt
    ema_decay: float = 0.999
    num_workers: int = 0
    seed: int = 42
    deterministic: bool = False
    task_weighting: str = "learned"          # or "fixed"
    fixed_loss_weights: dict = field(default_factory=lambda: {
        "seg": 1.0, "seg_deep": 0.3, "species": 0.8, "change": 1.2,
        "change_year": 0.5, "recovery": 0.5, "damage": 0.5, "quality": 0.2,
        "boundary": 0.3, "carbon": 0.6, "eco": 0.25, "risk": 0.25,
        "hydro": 1.0, "canopy": 1.0})
    # losses
    seg_loss: str = "sce"                    # "sce" | "focal_dice"
    sce_alpha: float = 1.0
    sce_beta: float = 0.1
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    pos_weight: float = 2.0
    depth_scale: float = 1000.0
    height_scale: float = 30.0
    carbon_scale: float = 100.0
    eco_scale: tuple = (50.0, 1.0)
    # inference
    infer_batch: int = 2
    infer_tta: bool = True
    infer_overlap: float = 0.5
    mc_samples: int = 0
    seg_temperature: float = 1.0
    change_threshold: float = 0.5
    change_conformal_q: float = 0.0
    conformal_alpha: float = 0.10
    # gradcam / review / smoothing
    gradcam_class: int = 4
    gradcam_tiles: int = 4
    review_queue_max: int = 64
    review_tile: int = 256
    review_low_quality: float = 0.5
    review_change_margin: float = 0.10
    smooth_stay: float = 0.85
    # pretrain
    pretrain_epochs: int = 40
    pretrain_lr: float = 3e-4
    pretrain_mask_ratio: float = 0.5
    pretrain_block: int = 32
    pretrain_weights: str = ""
    # mrv
    pixel_size_m: float = 10.0
    carbon_to_co2e: float = 44.0 / 12.0
    c_fraction: float = 0.5
    active_heads: tuple = ("seg", "species", "change", "change_year",
                           "recovery", "damage", "quality", "boundary",
                           "carbon", "eco", "risk", "hydro", "canopy")

    @property
    def aux_channels(self):
        return len(self.aux_names)

    def check(self):
        assert self.patch_size % 32 == 0
        assert self.temporal_steps >= 1
        assert self.epochs >= self.phase2_end >= self.phase1_end >= 1
        assert self.seg_loss in ("sce", "focal_dice")
        assert self.task_weighting in ("learned", "fixed")
        assert N_CH == 19 + self.aux_channels, "aux_names must have 6 entries"
        return True

    _TUPLES = ("invalid_scl", "rare_classes", "depths", "change_widths",
               "aux_names", "aux_scale", "species_names", "eco_scale",
               "active_heads")

    def save(self, path):
        d = asdict(self)
        for k in self._TUPLES:
            if k in d: d[k] = list(d[k])
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        for k in Config._TUPLES:
            if k in d: d[k] = tuple(d[k])
        return Config(**d)


CFG = Config()

HEAD_REQUIRED_KEYS = {
    "seg": ["label"], "species": ["species"], "change": ["change"],
    "change_year": ["change_year"], "recovery": ["recovery"],
    "damage": ["debris"], "carbon": ["carbon"], "eco": ["eco"],
    "risk": ["risk"], "canopy": ["gedi"],
    "quality": [], "boundary": [], "hydro": [],   # derived / pseudo-label
}


def dataset_fingerprint(scene_descs):
    h = hashlib.sha256(); n = 0
    for kind, path in sorted(scene_descs, key=lambda d: d[1]):
        files = sorted(glob.glob(os.path.join(path, "*.npz"))) \
            if kind == "folder" else [path]
        for f in files:
            h.update(os.path.basename(f).encode())
            try:
                h.update(str(os.path.getsize(f)).encode()); n += 1
            except OSError:
                h.update(b"missing")
    return h.hexdigest()[:16], n


# ==============================================================================
# 3. SCENE LOADING (folder-of-steps OR v9 cube NPZ -> unified 25-ch cube)
# ==============================================================================
class NPZSchemaError(ValueError):
    pass


def _to_db(x, lo, hi):
    x = x.astype(np.float32)
    if np.nanmin(x) >= 0:
        x = 10.0 * np.log10(np.clip(x, 1e-6, None))
    return np.clip(x, lo, hi)


def _process_s2(s2):
    s2 = s2.astype(np.float32)
    mx = np.nanmax(s2) if s2.size else 0.0
    if mx > 20000: s2 *= 1.0 / 65535.0
    elif mx > 100: s2 *= 1e-4
    return np.clip(s2, 0.0, 1.5)


def _sanitize(arr, name, path, max_bad=0.30):
    bad = ~np.isfinite(arr)
    if bad.any():
        frac = bad.sum() / arr.size
        if frac > max_bad:
            raise NPZSchemaError(f"{path}: '{name}' {frac:.0%} non-finite")
        arr = np.where(bad, 0.0, arr)
    return arr


def _date_from_name(path):
    m = re.search(r"((?:19|20)\d{2})[-_.]?(\d{2})?", os.path.basename(path))
    if not m: return 0.0
    y = int(m.group(1)); mo = m.group(2)
    if mo and 1 <= int(mo) <= 12:
        return y + (int(mo) - 1) / 12.0
    return float(y)


def _as_hw(x, H, W, fill=0.0):
    if x is None:
        return np.full((H, W), fill, np.float32)
    x = np.asarray(x, np.float32)
    if x.ndim == 0:
        return np.full((H, W), float(x), np.float32)
    if x.shape != (H, W):
        raise NPZSchemaError(f"spatial mismatch {x.shape} vs {(H, W)}")
    return x


def _default_targets(H, W, cfg):
    return {
        "label": np.full((H, W), cfg.ignore_label, np.int64),
        "species": np.full((H, W), 255, np.int64),
        "change": np.full((H, W), 255, np.int64),
        "recovery": np.full((H, W), 255, np.int64),
        "debris": np.full((H, W), 255, np.int64),
        "carbon": np.full((H, W), -999.0, np.float32),
        "eco": np.full((H, W, 2), -999.0, np.float32),
        "risk": np.full((H, W), -999.0, np.float32),
        "change_year": np.full((H, W), -999.0, np.float32),
        "gedi": np.full((H, W), np.nan, np.float32),
        "depth": None,
    }


def load_step_npz(path, cfg):
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
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
        s1_std = (_sanitize(d["s1_std"].astype(np.float32), "s1_std", path)
                  if "s1_std" in files else np.zeros((H, W, 2), np.float32))
        scl = _as_hw(d["scl"] if "scl" in files else None, H, W, fill=4.0)
        lband = (_sanitize(_to_db(d["lband"], -40.0, 5.0), "lband", path)
                 if "lband" in files else np.zeros((H, W, 2), np.float32))
        precip = np.clip(_sanitize(_as_hw(d.get("precip"), H, W), "precip",
                                   path), 0.0, cfg.precip_clip)
        inund = np.clip(_sanitize(_as_hw(d.get("inundation"), H, W),
                                  "inundation", path), 0.0, 20000.0)
        aux = (d["aux"].astype(np.float32) if "aux" in files
               else np.zeros((H, W, A), np.float32))
        if aux.shape[-1] != A:
            fixed = np.zeros((H, W, A), np.float32)
            fixed[..., :min(A, aux.shape[-1])] = aux[..., :A]
            aux = fixed
        aux = _sanitize(aux, "aux", path) / scale
        # targets: float sentinels preserved. Int bands ARE sanitized —
        # GEE exports carry NaN nodata there and a blind int cast turns
        # NaN into huge negative garbage that crashes bincount downstream
        # (FIX v11.4.1). Non-finite/negative/255+ -> 255 (ignore).
        t = _default_targets(H, W, cfg)
        for k in ("label", "species", "change", "recovery", "debris"):
            if k in files:
                a = d[k].astype(np.float64)
                bad = ~(np.isfinite(a) & (a >= 0) & (a < 255))
                t[k] = np.where(bad, 255.0, a).astype(np.int64)
        for k in ("carbon", "risk", "change_year"):
            if k in files:
                t[k] = d[k].astype(np.float32)
        if "eco" in files:
            t["eco"] = d["eco"].astype(np.float32)
        if "gedi" in files:
            t["gedi"] = _as_hw(d["gedi"], H, W, fill=np.nan)
        if "depth" in files:
            t["depth"] = _as_hw(d["depth"], H, W, fill=np.nan)
        date = float(d["date"]) if "date" in files else _date_from_name(path)

        x = np.concatenate([s2, scl[..., None].astype(np.float32), s1, s1_std,
                        lband, precip[..., None], inund[..., None], aux],
                       axis=-1).astype(np.float32)
    assert x.shape[-1] == N_CH, f"{path}: {x.shape[-1]} channels != {N_CH}"
    # FIX: Transpose from TensorFlow (H, W, C) to PyTorch (C, H, W)
    x = x.transpose(2, 0, 1)
    return {"x": x, "targets": t, "date": np.float32(date)}


def _assemble_scene(steps, cfg):
    T = cfg.temporal_steps
    steps = sorted(steps, key=lambda s: s["date"])
    if len(steps) > T:
        idx = np.unique(np.linspace(0, len(steps) - 1, T).round().astype(int))
        steps = [steps[int(i)] for i in idx]
    while len(steps) < T:
        steps.append(steps[-1])
    last = steps[-1]
    targets = dict(last["targets"])
    if targets["depth"] is None:                      # v9 pseudo-label fallback
        # FIX (v11.1): x is already (C,H,W) here, so index the CHANNEL axis.
        # The old x[..., CH_INUND] read column 18 of width -> (C,H) array ->
        # hydro loss crashed on a broadcast mismatch for every folder scene
        # without an explicit 'depth' key (the documented default!).
        targets["depth"] = last["x"][CH_INUND].copy()
    return {"x": np.stack([s["x"] for s in steps], 0),
            "targets": targets,
            "dates": np.array([s["date"] for s in steps], np.float32)}


def _load_cube_npz(path, cfg):
    """v9 cube: (T,19,H,W) -> remap into the unified 25-ch layout."""
    with np.load(path) as d:
        if "cube" not in d.files:
            raise NPZSchemaError(f"{path}: missing 'cube'")
        cube = d["cube"].astype(np.float32)
        if cube.ndim != 4 or cube.shape[1] != 19:
            raise NPZSchemaError(f"{path}: cube must be (T,19,H,W), "
                                 f"got {cube.shape}")
        dates = (d["dates"].astype(np.float32) if "dates" in d.files
                 else np.arange(cube.shape[0], dtype=np.float32))
    T0, _, H, W = cube.shape
    x = np.zeros((T0, N_CH, H, W), np.float32)
    x[:, 0:11] = cube[:, 0:11]                        # s2 + scl
    x[:, 11:13] = cube[:, 11:13]                      # s1 vv,vh (std = 0)
    x[:, 15:19] = cube[:, 13:17]                      # lband, precip, inund
    for c in range(N_CH):
        x[:, c] = _sanitize(x[:, c], f"cube[:,{c}]", path)
    t = _default_targets(H, W, cfg)
    t["label"] = np.where(np.isfinite(cube[-1, 18]), cube[-1, 18],
                          cfg.ignore_label).astype(np.int64)
    gedi_c = cube[-1, 17].astype(np.float32)      # FIX: 0 = absent -> NaN
    t["gedi"] = np.where(gedi_c > 0, gedi_c, np.nan).astype(np.float32)
        # FIX: x is already (T, N_CH, H, W), so x[i] is correctly (N_CH, H, W).
    # The previous .transpose(1, 2, 0) was incorrectly moving channels to the end.
    steps = [{"x": x[i], "targets": t,
              "date": np.float32(dates[i] if i < len(dates) else i)}
             for i in range(T0)]
    return _assemble_scene(steps, cfg)

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


def validate_heads_vs_data(scene_descs, cfg, sample=5):
    present = set()
    for desc in scene_descs[:sample]:
        try:
            if desc[0] == "folder":
                for p in sorted(glob.glob(os.path.join(desc[1], "*.npz"))):
                    with np.load(p) as d:
                        present |= set(d.files)
            else:
                present |= {"cube", "label", "gedi"}   # cube carries both
        except Exception:
            continue
    active = []
    for h in cfg.active_heads:
        need = HEAD_REQUIRED_KEYS.get(h, [])
        if need and not any(k in present for k in need):
            print(f"  [heads] '{h}' enabled but no '{need[0]}' in sampled "
                  f"scenes -> DISABLED (wire labels, then re-enable).")
        else:
            active.append(h)
    if "seg" not in active:
        raise NPZSchemaError("seg head has no labels; check scene schema.")
    if tuple(active) != tuple(cfg.active_heads):
        cfg.active_heads = tuple(active)
        print(f"  [heads] active set: {list(active)}")
    return cfg


# ==============================================================================
# 4. PATCH GENERATOR (class-balanced, per-scene fault isolation, geo+photo aug,
#    modality dropout)
# ==============================================================================
TARGET_KEYS_2D = ("label", "species", "change", "recovery", "debris",
                  "carbon", "risk", "change_year", "gedi", "depth")


def _patch_weight(label, cfg):
    # FIX (v11.4.1): negatives (NaN-cast garbage) crash bincount -> drop
    valid = (label != cfg.ignore_label) & (label >= 0)
    if valid.sum() == 0:
        return 0.0
    # FIX (v11.2): dirty labels >= num_classes made bincount longer than
    # num_classes -> broadcast crash. Truncate, as the confusion matrix does.
    counts = np.bincount(label[valid].ravel(),
                         minlength=cfg.num_classes)[:cfg.num_classes] \
        .astype(np.float32)
    frac = counts / max(counts.sum(), 1)
    return 1.0 + cfg.rare_boost * float(frac[list(cfg.rare_classes)].sum())


def _np_geometric_aug(p, rng):
    """Axis-safe: cube (T,C,H,W) & 2D targets flip/rot on (-2,-1);
    eco (H,W,2) on (0,1). Fixes the v9 channel-flip bug."""
    p = dict(p)
    lr = rng.random() < 0.5; ud = rng.random() < 0.5
    k = int(rng.integers(0, 4))

    def t(v, ax):
        if lr: v = np.flip(v, axis=ax[1])
        if ud: v = np.flip(v, axis=ax[0])
        if k: v = np.rot90(v, k, axes=ax)
        return np.ascontiguousarray(v)

    p["x"] = t(p["x"], (-2, -1))
    for key in TARGET_KEYS_2D:
        p[key] = t(p[key], (-2, -1))
    p["eco"] = t(p["eco"], (0, 1))
    return p


def _np_photometric_aug(p, rng, cfg):
    p = dict(p); x = p["x"].copy()
    s2 = x[:, CH_S2]
    delta = rng.uniform(-0.10, 0.10); contra = rng.uniform(0.85, 1.15)
    mean = s2.mean(axis=(2, 3), keepdims=True)  # FIX: per-channel spatial mean
    x[:, CH_S2] = np.clip((s2 - mean) * contra + mean + delta, -1.0, 2.0)
    x[:, CH_S1] += rng.normal(0, 0.3, x[:, CH_S1].shape).astype(np.float32)
    x[:, CH_LBAND] += rng.normal(0, 0.3, x[:, CH_LBAND].shape).astype(np.float32)
    if rng.random() < 0.08:                            # optical dropout
        x[:, 0:11] = 0.0
    elif rng.random() < 0.08:                          # radar dropout
        x[:, 11:17] = 0.0
    p["x"] = x
    return p


def patch_generator(scene_descs, cfg, training, max_patches, seed=None,
                    require_labels=True):
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
            if failed >= 25:
                raise RuntimeError("Too many scene failures; check data root.")
            continue
        label = scene["targets"]["label"]
        h, w = label.shape
        if h < ps or w < ps:
            continue
        coords = []
        for i in range(0, h - ps + 1, stride):
            for j in range(0, w - ps + 1, stride):
                lp = label[i:i + ps, j:j + ps]
                # FIX (v11.1): pretrain/stats on unlabelled scenes must not
                # be starved by the label-coverage filter.
                if require_labels and \
                        float(np.mean(lp != cfg.ignore_label)) < cfg.min_valid_frac:
                    continue
                sc = scene["x"][-1, CH_SCL, i:i + ps, j:j + ps]
                if float(np.mean(np.isin(sc, invalid))) > cfg.max_cloud_frac:
                    continue
                wgt = _patch_weight(lp, cfg) \
                    if (training and require_labels) else 1.0
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
            patch = {"x": np.ascontiguousarray(scene["x"][:, :, sl[0], sl[1]]),
                     "dates": scene["dates"]}
            for key in TARGET_KEYS_2D + ("eco",):
                patch[key] = np.ascontiguousarray(scene["targets"][key][sl]
                    if scene["targets"][key].ndim == 2
                    else scene["targets"][key][sl[0], sl[1], :])
            if training:
                patch = _np_geometric_aug(patch, rng)
                patch = _np_photometric_aug(patch, rng, cfg)
            yield patch
            emitted += 1
        del scene
        gc.collect()
    if emitted == 0:
        raise RuntimeError("patch_generator emitted 0 patches. Check scene "
                           "dirs, patch_size, min_valid_frac/max_cloud_frac, "
                           "labels.")


class PatchStream(IterableDataset):
    def __init__(self, scene_descs, cfg, training, max_patches,
                 require_labels=True):
        super().__init__()
        self.scene_descs, self.cfg = scene_descs, cfg
        self.training, self.max_patches = training, max_patches
        self.require_labels = require_labels
        self.epoch_seed = 0

    def set_epoch(self, ep):
        self.epoch_seed = ep

    def __iter__(self):
        info = get_worker_info()
        seed = self.cfg.seed + 1000 * self.epoch_seed + (info.id if info else 0)
        for idx, patch in enumerate(patch_generator(
                self.scene_descs, self.cfg, self.training,
                self.max_patches, seed=seed,
                require_labels=self.require_labels)):
            if info is not None and idx % info.num_workers != info.id:
                continue
            yield {k: torch.from_numpy(np.asarray(v)) for k, v in patch.items()}


def make_loader(scene_descs, cfg, training, max_patches, require_labels=True):
    ds = PatchStream(scene_descs, cfg, training, max_patches,
                     require_labels=require_labels)
    return DataLoader(ds, batch_size=cfg.batch_size,
                      num_workers=cfg.num_workers,
                      pin_memory=(DEVICE.type == "cuda"),
                      drop_last=training), ds


# ==============================================================================
# 5. NORMALIZATION STATS
# ==============================================================================
def stats_path(fp):
    return os.path.join(WORK_DIR, f"norm_stats_v11_{fp}.npz")


def compute_stats(scene_descs, cfg, max_patches=300, out_path=None):
    assert out_path, "compute_stats: out_path required (use load_or_compute_stats)"
    print("Computing per-channel normalization stats...")
    n = 0
    s_sum = np.zeros(N_CH); s_sq = np.zeros(N_CH)
    counts = np.zeros(cfg.num_classes, np.int64)
    for patch in patch_generator(scene_descs, cfg, False, max_patches,
                                 require_labels=False):
        x = patch["x"].transpose(0, 2, 3, 1).reshape(-1, N_CH)
        s_sum += x.sum(0); s_sq += (x ** 2).sum(0); n += x.shape[0]
        lab = patch["label"]
        v = lab[(lab != cfg.ignore_label) & (lab >= 0)]
        # FIX (v11.2): truncate like the confusion matrix (dirty-label guard)
        # FIX (v11.4.1): negatives (NaN-cast garbage) crash bincount -> drop
        counts += np.bincount(v, minlength=cfg.num_classes)[:cfg.num_classes]
    if n == 0:
        raise RuntimeError("compute_stats: no patches collected.")
    mean = s_sum / n
    var = np.clip(s_sq / n - mean ** 2, 1e-6, None)
    mean[CH_SCL] = 0.0; var[CH_SCL] = 121.0            # scl / 11 scaling
    freq = counts / max(counts.sum(), 1)
    alpha = 1.0 / (freq + 1e-6)
    alpha = np.clip(alpha / max(alpha.mean(), 1e-9), 0.25, 4.0)
    np.savez(out_path, mean=mean.astype(np.float32),
             var=var.astype(np.float32), alpha=alpha.astype(np.float32))
    print(f"  class counts: {counts.tolist()} alpha: {np.round(alpha, 3).tolist()}")
    return mean.astype(np.float32), var.astype(np.float32), alpha


def load_or_compute_stats(scene_descs, cfg):
    # FIX (v11.1): cache keyed by dataset fingerprint — a fixed filename
    # silently served stale stats after the dataset changed.
    fp, _ = dataset_fingerprint(scene_descs)
    path = stats_path(fp)
    if os.path.exists(path):
        z = np.load(path)
        print(f"Loaded cached norm stats ({os.path.basename(path)}).")
        return z["mean"], z["var"], z["alpha"]
    return compute_stats(scene_descs, cfg, out_path=path)


# ==============================================================================
# 6. SELECTIVE SCAN (v9 core — kept; verified against naive recurrence)
# ==============================================================================
def _scan_chunk(u_c, dt_c, h0, A, B_c, C_c):
    h = h0; ys = []
    for t in range(u_c.shape[1]):
        dA = torch.exp(dt_c[:, t].unsqueeze(-1) * A)
        dB = dt_c[:, t].unsqueeze(-1) * B_c[:, t].unsqueeze(1)
        h = dA * h + dB * u_c[:, t].unsqueeze(-1)
        ys.append(torch.einsum("bdn,bn->bd", h, C_c[:, t]))
    return torch.stack(ys, dim=1), h


def selective_scan_torch(u, dt, A, B, C, D, chunk=512, use_ckpt=True):
    Bs, L, Dd = u.shape; N = A.shape[-1]
    u32, dt32, B32, C32 = u.float(), dt.float(), B.float(), C.float()
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
    y = torch.cat(outs, dim=1) + u32 * D.float()
    return y.to(u.dtype)


class SSMCore(nn.Module):
    def __init__(self, d_inner, n_dirs, d_state=16, dt_rank="auto",
                 dt_min=1e-3, dt_max=0.1, chunk=512, grad_ckpt=True):
        super().__init__()
        self.d_inner, self.d_state, self.n_dirs = d_inner, d_state, n_dirs
        self.chunk, self.grad_ckpt = chunk, grad_ckpt
        self.dt_rank = int(math.ceil(d_inner / 16)) \
            if dt_rank == "auto" else int(dt_rank)
        self.x_projs = nn.ModuleList([
            nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(n_dirs)])
        self.dt_projs = nn.ModuleList([
            self._init_dt_proj(dt_min, dt_max) for _ in range(n_dirs)])
        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
            .repeat(n_dirs, d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(n_dirs, d_inner))

    def _init_dt_proj(self, dt_min, dt_max):
        proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        k = 1.0 / max(self.dt_rank, 1)
        nn.init.uniform_(proj.weight, -k, k)
        dt = torch.exp(torch.rand(self.d_inner)
                       * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp_min(1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        return proj

    def scan(self, seq, d):
        dtbc = self.x_projs[d](seq)
        dt, Bm, Cm = torch.split(
            dtbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log[d].float())
        Dskip = self.D[d].float()
        if HAS_MAMBA and seq.is_cuda:
            delta = self.dt_projs[d](dt)
            y = selective_scan_fn(
                seq.transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(), A,
                Bm.transpose(1, 2).contiguous(),
                Cm.transpose(1, 2).contiguous(), Dskip, z=None,
                delta_bias=self.dt_projs[d].bias.float(), delta_softplus=True)
            return y.transpose(1, 2)
        dtf = F.softplus(self.dt_projs[d](dt.float()))
        return selective_scan_torch(seq, dtf, A, Bm, Cm, Dskip,
                                    chunk=self.chunk, use_ckpt=self.grad_ckpt)


# ==============================================================================
# 7. SS3D / VSS BLOCKS (v9) + FiLM TIME CONDITIONING (v8, ported)
# ==============================================================================
class DropPath(nn.Module):
    def __init__(self, rate=0.0):
        super().__init__(); self.rate = float(rate)

    def forward(self, x):
        if self.rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class SpatioTemporalSSM(nn.Module):
    def __init__(self, dim, temporal=True, expand=2, d_state=16,
                 dt_rank="auto", chunk=512, grad_ckpt=True):
        super().__init__()
        self.dim = dim; self.d_inner = int(dim * expand)
        self.temporal = temporal
        n_dirs = 6 if temporal else 4
        self.ln = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner)
        self.dw = nn.Conv2d(self.d_inner, self.d_inner, 3, 1, 1,
                            groups=self.d_inner)
        self.core = SSMCore(self.d_inner, n_dirs, d_state=d_state,
                            dt_rank=dt_rank, chunk=chunk, grad_ckpt=grad_ckpt)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)

    def forward(self, x):                              # (B,T,C,H,W)
        B, T, C, H, W = x.shape
        di = self.d_inner
        x = x.permute(0, 1, 3, 4, 2)
        x = self.ln(x)
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = xi.permute(0, 1, 4, 2, 3).reshape(B * T, di, H, W)
        xi = F.silu(self.dw(xi)).view(B, T, di, H, W)
        s1 = xi.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, di)
        s2 = s1.flip(1)
        s3 = xi.permute(0, 1, 4, 3, 2).reshape(B * T, H * W, di)
        s4 = s3.flip(1)
        y1 = self.core.scan(s1, 0).view(B, T, H, W, di).permute(0, 1, 4, 2, 3)
        y2 = self.core.scan(s2, 1).flip(1).view(B, T, H, W, di) \
            .permute(0, 1, 4, 2, 3)
        y3 = self.core.scan(s3, 2).view(B, T, W, H, di).permute(0, 1, 4, 3, 2)
        y4 = self.core.scan(s4, 3).flip(1).view(B, T, W, H, di) \
            .permute(0, 1, 4, 3, 2)
        y = y1 + y2 + y3 + y4
        if self.temporal and T > 1:
            s5 = xi.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, di)
            s6 = s5.flip(1)
            y5 = self.core.scan(s5, 4).view(B, H, W, T, di).permute(0, 3, 4, 1, 2)
            y6 = self.core.scan(s6, 5).flip(1).view(B, H, W, T, di) \
                .permute(0, 3, 4, 1, 2)
            y = y + y5 + y6
        y = y.permute(0, 1, 3, 4, 2)
        y = self.out_norm(y) * F.silu(z)
        return self.out_proj(y).permute(0, 1, 4, 2, 3)


class VSSBlock(nn.Module):
    def __init__(self, dim, temporal=True, drop_path=0.0, ckpt_mamba=True,
                 **ssm_kw):
        super().__init__()
        self.ssm = SpatioTemporalSSM(dim, temporal=temporal, **ssm_kw)
        self.dp = DropPath(drop_path)
        self.use_ckpt = ssm_kw.get("grad_ckpt", True)
        # v11.1: allow checkpointing even when CUDA mamba kernels are active —
        # a 16 GB T4 cannot hold all VSS activations at patch 256 otherwise.
        self.ckpt_mamba = bool(ckpt_mamba)

    def _forward(self, x):
        return x + self.dp(self.ssm(x))

    def forward(self, x):
        if self.use_ckpt and self.training and torch.is_grad_enabled() \
                and (self.ckpt_mamba or not (HAS_MAMBA and x.is_cuda)):
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False)
        return self._forward(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, freqs=16, max_years=20.0):
        super().__init__()
        f = torch.arange(freqs).float() / freqs
        self.register_buffer("freq_table", 1.0 / (10000.0 ** f))
        self.max_years = float(max_years)
        self.mlp = nn.Sequential(nn.Linear(2 * freqs, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, dt):                             # (B,) years
        ang = dt.float()[:, None] / self.max_years * self.freq_table[None] \
            * 2.0 * math.pi
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class FiLM(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        g, b = self.fc(cond).chunk(2, -1)
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


# ==============================================================================
# 8. STEMS / MERGING / DECODERS
# ==============================================================================
def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class Stem(nn.Sequential):
    def __init__(self, cin, dim):
        super().__init__(nn.Conv2d(cin, dim, 3, 1, 1),
                         nn.BatchNorm2d(dim), nn.GELU())


class PatchMerging2D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm = nn.LayerNorm(4 * cin)
        self.reduction = nn.Linear(4 * cin, cout, bias=False)

    def forward(self, x):                              # (B,T,C,H,W)
        x = torch.cat([x[:, :, :, 0::2, 0::2], x[:, :, :, 1::2, 0::2],
                       x[:, :, :, 0::2, 1::2], x[:, :, :, 1::2, 1::2]], dim=2)
        x = x.permute(0, 1, 3, 4, 2)
        return self.reduction(self.norm(x)).permute(0, 1, 4, 2, 3)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.bn1 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.bn2 = _gn(cout)
        self.proj = nn.Conv2d(cin, cout, 1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.proj(x))


class DecoderStage(nn.Module):
    """Transpose-conv up -> concat skip -> fuse -> VSS(T=1) -> refine."""

    def __init__(self, cin, cskip, cout, cfg):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.fuse = nn.Sequential(nn.Conv2d(cout + cskip, cout, 1),
                                  _gn(cout), nn.GELU())
        self.vss = VSSBlock(cout, temporal=False, drop_path=0.0,
                            expand=cfg.expand, d_state=cfg.d_state,
                            dt_rank=cfg.dt_rank, chunk=cfg.ssm_chunk,
                            grad_ckpt=cfg.grad_ckpt,
                            ckpt_mamba=getattr(cfg, "ckpt_mamba", True))
        self.refine = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1),
                                    _gn(cout), nn.GELU())

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.vss(x.unsqueeze(1)).squeeze(1)
        return self.refine(x)


class ChangeStage(nn.Module):
    """DecoderStage + FiLM time conditioning (v8 change decoder, ported)."""

    def __init__(self, cin, cskip, cout, cond_dim, cfg):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.fuse = ConvBlock(cout + cskip, cout)
        self.film = FiLM(cout, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.film(self.fuse(torch.cat([x, skip], dim=1)), cond)


# ==============================================================================
# 9. BLUE SENTINEL v11 MODEL
# ==============================================================================
class BlueSentinelV11(nn.Module):
    def __init__(self, cfg, norm_mean, norm_var):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim
        dims = [D, 2 * D, 4 * D, 8 * D]
        self.dims = dims
        self.register_buffer("in_mean",
                             torch.tensor(norm_mean).view(1, 1, -1, 1, 1))
        self.register_buffer("in_std",
                             torch.tensor(np.sqrt(norm_var)).view(1, 1, -1, 1, 1))

        self.opt_stem = Stem(11, D)                    # ch 0-10
        self.radar_stem = Stem(6, D)                   # ch 11-16 (S1x4 + Lx2)
        self.aux_stem = Stem(2 + cfg.aux_channels, D)  # ch 17-24
        self.fuse = nn.Conv2d(3 * D, D, 1)

        ssm_kw = dict(expand=cfg.expand, d_state=cfg.d_state,
                      dt_rank=cfg.dt_rank, chunk=cfg.ssm_chunk,
                      grad_ckpt=cfg.grad_ckpt,
                      ckpt_mamba=getattr(cfg, "ckpt_mamba", True))
        dpr = np.linspace(0.0, cfg.drop_path, sum(cfg.depths)).tolist()
        self.stages = nn.ModuleList(); k = 0
        for i, dep in enumerate(cfg.depths):
            blocks = [VSSBlock(dims[i], temporal=True, drop_path=dpr[k + j],
                               **ssm_kw) for j in range(dep)]
            k += dep
            self.stages.append(nn.ModuleList(blocks))
        self.merges = nn.ModuleList(
            [PatchMerging2D(dims[i], dims[i + 1]) for i in range(3)])

        self.dec2 = DecoderStage(dims[3], dims[2], dims[2], cfg)
        self.dec1 = DecoderStage(dims[2], dims[1], dims[1], cfg)
        self.dec0 = DecoderStage(dims[1], dims[0], dims[0], cfg)

        cw = list(cfg.change_widths)
        self.time_embed = TimeEmbedding(cfg.time_embed_dim,
                                        max_years=cfg.max_time_gap_years)
        self.ctx_proj = nn.Sequential(nn.Linear(dims[3], cfg.time_embed_dim),
                                      nn.GELU(),
                                      nn.Linear(cfg.time_embed_dim,
                                                cfg.time_embed_dim))
        self.change_in = ConvBlock(dims[3], cw[0])
        self.change_film0 = FiLM(cw[0], cfg.time_embed_dim)
        self.change_stages = nn.ModuleList([
            ChangeStage(cw[0], dims[2], cw[1], cfg.time_embed_dim, cfg),
            ChangeStage(cw[1], dims[1], cw[2], cfg.time_embed_dim, cfg),
            ChangeStage(cw[2], dims[0], cw[2], cfg.time_embed_dim, cfg)])

        t = min(64, D)
        self.trunk = nn.Sequential(nn.Conv2d(D, t, 3, 1, 1), _gn(t), nn.GELU(),
                                   nn.Dropout2d(cfg.head_dropout))
        self.ctrunk = nn.Sequential(nn.Conv2d(cw[2], t, 3, 1, 1), _gn(t),
                                    nn.GELU(), nn.Dropout2d(cfg.head_dropout))
        C, S = cfg.num_classes, cfg.num_species
        self.seg_head = nn.Conv2d(t, C, 1)
        self.deep_head = nn.Conv2d(dims[1], C, 1)      # @128 (stride 2)
        self.species_head = nn.Conv2d(t, S, 1)
        self.carbon_head = nn.Conv2d(t, 2, 1)          # (mean, logvar)
        self.eco_head = nn.Conv2d(t, 2, 1)
        self.risk_head = nn.Conv2d(t, 1, 1)
        self.quality_head = nn.Conv2d(t, 1, 1)
        self.boundary_head = nn.Conv2d(t, 1, 1)
        self.hydro_head = nn.Conv2d(t, 2, 1)
        self.canopy_head = nn.Conv2d(t, 1, 1)
        self.change_head = nn.Conv2d(t, 1, 1)
        self.change_year_head = nn.Conv2d(t, 1, 1)
        self.recovery_head = nn.Conv2d(t, 1, 1)
        self.damage_head = nn.Conv2d(t, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, dates=None):
        """x: (B,T,25,H,W); dates: (B,T) decimal years (needed for change
        FiLM; defaults to integer spacing)."""
        assert x.dim() == 5 and x.shape[2] == N_CH
        B, T, _, H, W = x.shape
        if dates is None:
            dates = torch.arange(T, device=x.device, dtype=x.dtype) \
                .expand(B, T)
        x = (x - self.in_mean) / self.in_std

        f = x.reshape(B * T, N_CH, H, W)
        z = self.fuse(torch.cat([self.opt_stem(f[:, 0:11]),
                                 self.radar_stem(f[:, 11:17]),
                                 self.aux_stem(f[:, 17:25])], dim=1))
        z = z.view(B, T, self.cfg.dim, H, W)

        skips = []
        for i, stage in enumerate(self.stages):
            for blk in stage:
                z = blk(z)
            skips.append(z)
            if i < len(self.merges):
                z = self.merges[i](z)

        last = [s[:, -1] for s in skips]               # reference frame
        d0 = self.dec2(z[:, -1], last[2])
        d1 = self.dec1(d0, last[1])
        d2 = self.dec0(d1, last[0])
        h = self.trunk(d2)

        # change branch: |last - first| diffs + FiLM(dt, ctx)
        diffs = [(sk[:, -1] - sk[:, 0]).abs() for sk in skips]
        dt = (dates[:, -1] - dates[:, 0]).float()
        ctx = self.ctx_proj(z[:, -1].mean(dim=(2, 3))) + self.time_embed(dt)
        c = self.change_film0(self.change_in(diffs[3]), ctx)
        c = self.change_stages[0](c, diffs[2], ctx)
        c = self.change_stages[1](c, diffs[1], ctx)
        c = self.change_stages[2](c, diffs[0], ctx)
        ch = self.ctrunk(c)

        hydro = self.hydro_head(h).float()
        return {
            "seg": self.seg_head(h).float(),
            "seg_deep": self.deep_head(d1).float(),
            "species": self.species_head(h).float(),
            "carbon": self.carbon_head(h).float(),
            "eco": self.eco_head(h).float(),
            "risk": self.risk_head(h).float(),
            "quality": self.quality_head(h).float(),
            "boundary": self.boundary_head(h).float(),
            "hydro_mu": F.softplus(hydro[:, 0:1]),
            "hydro_var": F.softplus(hydro[:, 1:2]) + 1e-3,
            "canopy": self.canopy_head(h).float(),
            "change": self.change_head(ch).float(),
            "change_year": self.change_year_head(ch).float(),
            "recovery": self.recovery_head(ch).float(),
            "damage": self.damage_head(ch).float(),
        }

    def set_phase(self, phase):
        """Phase 1: only seg-supervision heads train (v8 aux heads quality/
        boundary stay open — their supervision is free)."""
        trainable = phase > 1
        for head in (self.species_head, self.carbon_head, self.eco_head,
                     self.risk_head, self.hydro_head, self.canopy_head,
                     self.change_head, self.change_year_head,
                     self.recovery_head, self.damage_head,
                     self.change_in, self.change_stages, self.ctrunk):
            for p in head.parameters():
                p.requires_grad_(trainable)


# ==============================================================================
# 10. LOSSES  (v8 set + v9 set; unified (pred, target, valid) signature)
# ==============================================================================
def _n(valid):
    return valid.float().sum() + 1e-6


class SCELoss(nn.Module):
    """Symmetric CE = alpha*CE + beta*RCE (RCE clamped at -4)."""

    def __init__(self, num_classes, alpha=1.0, beta=0.1):
        super().__init__()
        self.C, self.alpha, self.beta = num_classes, alpha, beta

    def forward(self, logits, target, valid):
        if valid.sum() == 0:
            return logits.sum() * 0.0
        logp = F.log_softmax(logits.float(), dim=1)
        t = target.clamp(0, self.C - 1)
        ce = -logp.gather(1, t.unsqueeze(1)).squeeze(1)
        p = logp.exp()
        q = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        logq = q.clamp_min(1e-4).log().clamp_min(-4.0)
        rce = -(p * logq).sum(dim=1)
        loss = self.alpha * ce + self.beta * rce
        return (loss * valid).sum() / _n(valid)


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, ls=0.05):
        super().__init__()
        self.C, self.gamma, self.ls = num_classes, gamma, ls
        self.register_buffer("alpha", None if alpha is None
                             else torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target, valid):
        if valid.sum() == 0:
            return logits.sum() * 0.0
        logp = F.log_softmax(logits.float(), dim=1)
        p = logp.exp()
        t = target.clamp(0, self.C - 1)
        y = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        m = valid.unsqueeze(1).float()
        if self.ls > 0:
            y = (y * (1 - self.ls) + self.ls / self.C) * m
        ce = -(y * logp).sum(1)
        pt = (-ce).exp()
        focal = (1 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            focal = focal * (y * self.alpha.view(1, -1, 1, 1)).sum(1)
        focal = (focal * valid).sum() / _n(valid)
        inter = (y * p * m).sum(dim=(2, 3))
        denom = ((y + p) * m).sum(dim=(2, 3))
        dice = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
        return focal + dice.mean()


class DeepSupervisionLoss(nn.Module):
    """Soft CE on 2x avg-pooled one-hot targets (deep head @ stride 2)."""

    def __init__(self, num_classes):
        super().__init__()
        self.C = num_classes

    def forward(self, logits, target, valid):
        t = target.clamp(0, self.C - 1)
        y = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        m = valid.unsqueeze(1).float()
        y = F.avg_pool2d(y * m, 2)
        m2 = F.avg_pool2d(m, 2)
        y = y / m2.clamp_min(1e-6)
        v = (m2.squeeze(1) > 0.5)
        if v.sum() == 0:
            return logits.sum() * 0.0
        logp = F.log_softmax(logits.float(), dim=1)
        return (-(y * logp).sum(1) * v).sum() / _n(v)


class MaskedBinaryFocal(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=2.0):
        super().__init__()
        self.gamma, self.pos_weight = gamma, pos_weight

    def forward(self, logit, target, valid):
        if valid.sum() == 0:
            return logit.sum() * 0.0
        p = torch.sigmoid(logit.float()).clamp(1e-6, 1 - 1e-6)
        y = target.float()
        ce = -(y * p.log() + (1 - y) * (1 - p).log())
        pt = torch.where(y > 0.5, p, 1 - p)
        fl = (1 - pt).pow(self.gamma) * ce
        w = torch.where(y > 0.5, self.pos_weight, 1.0)
        return (fl * w * valid).sum() / _n(valid)


class MaskedBCE(nn.Module):
    def forward(self, logit, target, valid):
        if valid.sum() == 0:
            return logit.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(
            logit.float(), target.float(), reduction="none")
        return (bce * valid).sum() / _n(valid)


class MaskedMAE(nn.Module):
    def forward(self, pred, target, valid):
        if valid.sum() == 0:
            return pred.sum() * 0.0
        return ((pred.float() - target.float()).abs() * valid).sum() / _n(valid)


class MaskedHuber(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__(); self.delta = delta

    def forward(self, pred, target, valid):
        if valid.sum() == 0:
            return pred.sum() * 0.0
        err = (pred.float() - target.float()).abs()
        hub = torch.where(err <= self.delta, 0.5 * err ** 2,
                          self.delta * (err - 0.5 * self.delta))
        return (hub * valid).sum() / _n(valid)


class HeteroGaussianNLL(nn.Module):
    """pred = (mean, log_var) 2-channel; used by carbon head."""

    def forward(self, pred, target, valid):
        if valid.sum() == 0:
            return pred.sum() * 0.0
        mean = pred[:, 0:1].float()
        log_var = pred[:, 1:2].float().clamp(-6.0, 8.0)
        nll = 0.5 * ((-log_var).exp() * (target.float() - mean) ** 2 + log_var)
        return (nll.squeeze(1) * valid).sum() / _n(valid)


class GaussianNLLVar(nn.Module):
    """v9 hydrology loss: pred given as (mu, var)."""

    def forward(self, mu, var, target, valid):
        if valid.sum() == 0:
            return mu.sum() * 0.0
        nll = 0.5 * (var.float().log()
                     + (target.float() - mu.float()) ** 2 / var.float())
        return (nll.squeeze(1) * valid).sum() / _n(valid)


class SparseMaskedL1(nn.Module):
    def forward(self, pred, target, valid):
        if valid.sum() == 0:
            return pred.sum() * 0.0
        return ((pred.float().squeeze(1) - target.float()).abs()
                * valid).sum() / _n(valid)


# ==============================================================================
# 11. METRICS
# ==============================================================================
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.C = num_classes
        self.cm = np.zeros((num_classes, num_classes), np.int64)

    @torch.no_grad()
    def update(self, logits, target, valid):
        p = logits.argmax(1).flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        v = valid.flatten().cpu().numpy().astype(bool)
        v &= (t >= 0) & (t < self.C)   # FIX: drop dirty labels (bincount crash)
        if v.sum() == 0:
            return
        idx = t[v] * self.C + p[v]
        self.cm += np.bincount(idx, minlength=self.C * self.C) \
            .reshape(self.C, self.C)

    def iou(self):
        d = np.diag(self.cm).astype(np.float64)
        u = self.cm.sum(0) + self.cm.sum(1) - d
        return np.where(u > 0, d / np.maximum(u, 1), np.nan)

    def miou(self):
        return float(np.nanmean(self.iou()))


class MaskedF1:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = 0.0

    @torch.no_grad()
    def update(self, logit, target, valid):
        if valid.sum() == 0:
            return
        p = (torch.sigmoid(logit.float()) > self.threshold)
        y = target > 0.5
        v = valid.bool()
        self.tp += float((p & y & v).sum())
        self.fp += float((p & ~y & v).sum())
        self.fn += float((~p & y & v).sum())

    def value(self):
        prec = self.tp / (self.tp + self.fp + 1e-6)
        rec = self.tp / (self.tp + self.fn + 1e-6)
        return 2 * prec * rec / (prec + rec + 1e-6)


class MaskedMAEMetric:
    def __init__(self):
        self.total = 0.0; self.count = 0

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
    """All methods unwrap nn.DataParallel first, so EMA state keys never get
    the 'module.' prefix and checkpoints stay single-GPU-portable."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in unwrap(model).state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in unwrap(model).state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    @torch.no_grad()
    def store(self, model):
        self._backup = {k: v.detach().clone()
                        for k, v in unwrap(model).state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model):
        for k, v in unwrap(model).state_dict().items():
            v.copy_(self.shadow[k].to(v.dtype))

    @torch.no_grad()
    def restore(self, model):
        if self._backup is not None:
            unwrap(model).load_state_dict(self._backup)
            self._backup = None


# ==============================================================================
# 13. TARGETS / CURRICULUM / OPTIMIZER
# ==============================================================================
def build_targets(batch, cfg):
    """Returns dict name -> (target, valid) plus seg raw label."""
    dev = batch["x"].device
    label = batch["label"].long()
    dates = batch["dates"].float()
    scl_last = batch["x"][:, -1, CH_SCL]
    invalid = torch.zeros_like(scl_last, dtype=torch.bool)
    for c in cfg.invalid_scl:
        invalid |= (scl_last == c)
    t = {}
    seg_valid = label != cfg.ignore_label
    t["seg"] = (label, seg_valid)
    t["seg_deep"] = (label, seg_valid)
    # free aux supervision (v8 prepare(), ported)
    t["quality"] = ((~invalid).float(), torch.ones_like(seg_valid))
    C = cfg.num_classes
    y = F.one_hot(label.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    mx = F.max_pool2d(y, 3, 1, 1)
    mn = -F.max_pool2d(-y, 3, 1, 1)
    edge = (mx - mn).clamp(0, 1).max(1).values
    t["boundary"] = (edge, seg_valid)

    def _i64(key):
        v = batch[key].long()
        return v.float().clamp(0, 1), v != 255

    for name, key in (("species", "species"),):
        sp = batch[key].long()
        t[name] = (sp, sp != 255)
    for name, key in (("change", "change"), ("recovery", "recovery"),
                      ("damage", "debris")):
        t[name] = _i64(key)
    cy = batch["change_year"].float()
    rel = ((cy - dates[:, 0:1, None]) / cfg.max_time_gap_years).clamp(0, 1)
    t["change_year"] = (rel, cy > 0.0)
    carb = batch["carbon"].float()
    t["carbon"] = (carb / cfg.carbon_scale, carb > -900.0)
        # FIX: Permute eco from (B, H, W, 2) to PyTorch format (B, 2, H, W)
    eco = batch["eco"].float().permute(0, 3, 1, 2)
    scale = torch.tensor(list(cfg.eco_scale), device=dev).view(1, 2, 1, 1)
    t["eco"] = (eco / scale, eco[:, 0] > -900.0)
    risk = batch["risk"].float()
    t["risk"] = (risk, risk > -900.0)
    depth = batch["depth"].float() / cfg.depth_scale
    t["hydro"] = (torch.nan_to_num(depth, nan=0.0), torch.isfinite(depth))
    gedi = batch["gedi"].float() / cfg.height_scale
    t["canopy"] = (torch.nan_to_num(gedi, nan=0.0), torch.isfinite(gedi))
    return t


def phase_for_epoch(ep1, cfg):
    return 1 if ep1 <= cfg.phase1_end else (2 if ep1 <= cfg.phase2_end else 3)


def lr_lambda_factory(cfg):
    ratio = cfg.final_lr / cfg.lr

    def lr_lambda(ep0):
        ep1 = ep0 + 1
        if ep1 <= cfg.phase2_end:
            return 1.0
        prog = (ep1 - cfg.phase2_end) / max(1, cfg.epochs - cfg.phase2_end)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return ratio + (1.0 - ratio) * cos
    return lr_lambda


def make_optimizer(model, task_logvars, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "A_log" in n or n.endswith(".D"))
         else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if task_logvars is not None:
        groups.append({"params": list(task_logvars.parameters()),
                       "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def amp_context(cfg):
    if cfg.amp and DEVICE.type == "cuda":
        return torch.autocast("cuda",
                              dtype=torch.bfloat16 if AMP_BF16
                              else torch.float16)
    return nullcontext()


# ==============================================================================
# 14. TRAIN / EVAL LOOPS
# ==============================================================================
ALL_TASKS = ("seg", "seg_deep", "species", "change", "change_year",
             "recovery", "damage", "quality", "boundary", "carbon", "eco",
             "risk", "hydro", "canopy")
PHASE1_TASKS = ("seg", "seg_deep", "quality", "boundary")


def _loss_fns(cfg, alpha):
    seg_fn = SCELoss(cfg.num_classes, cfg.sce_alpha, cfg.sce_beta) \
        if cfg.seg_loss == "sce" else \
        FocalDiceLoss(cfg.num_classes, alpha=alpha, gamma=cfg.focal_gamma,
                      ls=cfg.label_smoothing)
    return {
        "seg": seg_fn,
        "seg_deep": DeepSupervisionLoss(cfg.num_classes),
        "species": SCELoss(cfg.num_species, cfg.sce_alpha, cfg.sce_beta),
        "change": MaskedBinaryFocal(cfg.focal_gamma, cfg.pos_weight),
        "recovery": MaskedBinaryFocal(cfg.focal_gamma, cfg.pos_weight),
        "damage": MaskedBinaryFocal(cfg.focal_gamma, cfg.pos_weight),
        "quality": MaskedBCE(), "boundary": MaskedBCE(),
        "change_year": MaskedMAE(), "carbon": HeteroGaussianNLL(),
        "eco": MaskedHuber(), "risk": MaskedHuber(),
        "hydro": GaussianNLLVar(), "canopy": SparseMaskedL1(),
    }


def run_epoch(model, loader, cfg, phase, task_logvars=None, optimizer=None,
              scaler=None, ema=None, alpha=None):
    training = optimizer is not None
    model.train(training)
    active = set(cfg.active_heads) | {"quality", "boundary"}
    if cfg.deep_supervision and "seg" in active:
        active.add("seg_deep")
    tasks = [t for t in ALL_TASKS if t in active]
    if phase == 1:
        tasks = [t for t in tasks if t in PHASE1_TASKS]
    lam = 1.0 if phase == 3 else cfg.phase2_lambda
    fns = _loss_fns(cfg, alpha)   # FIX: class alpha is actually wired in now
    agg = {"steps": 0}
    cm = ConfusionMatrix(cfg.num_classes)
    f1 = MaskedF1()
    mae_c = MaskedMAEMetric()
    mae_h = MaskedMAEMetric()
    mae_can = MaskedMAEMetric()

    for batch in loader:
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        tg = build_targets(batch, cfg)
        with torch.set_grad_enabled(training), amp_context(cfg):
            out = model(batch["x"].float(), batch["dates"].float())
            per_task = {}
            for tname in tasks:
                target, valid = tg[tname]
                if tname == "hydro":
                    per_task[tname] = fns["hydro"](out["hydro_mu"],
                                                   out["hydro_var"],
                                                   target, valid)
                elif tname == "carbon":
                    per_task[tname] = fns["carbon"](out["carbon"],
                                                    target, valid)
                elif tname in ("eco",):
                    per_task[tname] = fns["eco"](out["eco"], target,
                                                 valid.unsqueeze(1))
                elif tname in ("risk", "quality", "boundary", "change",
                               "recovery", "damage", "change_year", "canopy"):
                    per_task[tname] = fns[tname](out[tname][:, 0]
                                                 if tname != "canopy"
                                                 else out[tname],
                                                 target, valid)
                else:
                    per_task[tname] = fns[tname](out[tname], target, valid)
            # FIX (v11.1): keep learned/fixed task weights + deep-supervision
            # weight in ALL phases; phase 2 scales the *weighted* sums.
            weighted = {}
            for tname, l in per_task.items():
                w_extra = cfg.deep_supervision_weight \
                    if tname == "seg_deep" else 1.0
                if cfg.task_weighting == "learned" and task_logvars is not None:
                    lv = task_logvars[tname]
                    weighted[tname] = w_extra * (l * torch.exp(-lv) + lv)
                else:
                    w = cfg.fixed_loss_weights.get(tname, 1.0) * w_extra
                    weighted[tname] = w * l
            if phase == 2:
                total = sum(weighted[n] for n in per_task if n in PHASE1_TASKS) \
                    + lam * sum(weighted[n] for n in per_task
                                if n not in PHASE1_TASKS)
            else:
                total = sum(weighted[n] for n in per_task)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip)
                optimizer.step()
            if ema is not None:
                ema.update(model)

        agg["steps"] += 1
        for n, l in per_task.items():
            agg[n] = agg.get(n, 0.0) + float(l)
        agg["loss"] = agg.get("loss", 0.0) + float(total)
        with torch.no_grad():
            cm.update(out["seg"], tg["seg"][0], tg["seg"][1])
            if "change" in per_task:
                f1.update(out["change"][:, 0], tg["change"][0], tg["change"][1])
            if "carbon" in per_task:
                mae_c.update(out["carbon"][:, 0], tg["carbon"][0],
                             tg["carbon"][1])
            mae_h.update(out["hydro_mu"][:, 0], tg["hydro"][0], tg["hydro"][1])
            mae_can.update(out["canopy"][:, 0], tg["canopy"][0],
                           tg["canopy"][1])

    n = max(agg.pop("steps"), 1)
    res = {k: v / n for k, v in agg.items()}
    res.update({"miou": cm.miou(), "change_f1": f1.value(),
                "carbon_mae": mae_c.value(), "hydro_mae": mae_h.value(),
                "canopy_mae": mae_can.value(), "cm": cm.cm})
    return res


def train(cfg=CFG):
    cfg.check()
    print(f"\n=== BLUE SENTINEL v{VERSION} - TRAINING ===\n")
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    train_scenes = discover_scenes(cfg.train_dir)
    val_scenes = discover_scenes(cfg.val_dir)
    assert train_scenes and val_scenes, (
        f"no v11 scenes found (train={len(train_scenes)} from {cfg.train_dir},"
        f" val={len(val_scenes)} from {cfg.val_dir}). If your data is raw GEE "
        "GeoTIFF exports, pack it first with "
        "'python gee_dataset_builder_v11.py pack --raw-dir ... --out ...'; "
        "else pass --train-dir / --val-dir at scene folders.")
    validate_heads_vs_data(train_scenes, cfg)
    fp, nf = dataset_fingerprint(train_scenes)
    print(f"train={len(train_scenes)} val={len(val_scenes)} "
          f"T={cfg.temporal_steps} fingerprint={fp} ({nf} files)")

    mean, var, alpha = load_or_compute_stats(train_scenes, cfg)
    train_loader, train_stream = make_loader(train_scenes, cfg, True,
                                             cfg.max_train_patches)
    val_loader, _ = make_loader(val_scenes, cfg, False, cfg.max_val_patches)

    model = BlueSentinelV11(cfg, mean, var).to(DEVICE)
    model = wrap_mirror(model, cfg.mirror_gpus)
    core = unwrap(model)          # EMA/ckpt/phase ops always target the core
    if cfg.pretrain_weights and os.path.exists(cfg.pretrain_weights):
        sd = torch.load(cfg.pretrain_weights, map_location=DEVICE,
                        weights_only=True)
        missing, unexpected = core.load_state_dict(sd, strict=False)
        print(f"  [pretrain] loaded {os.path.basename(cfg.pretrain_weights)} "
              f"({len(missing)} missing / {len(unexpected)} unexpected keys)")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    task_logvars = None
    if cfg.task_weighting == "learned":
        task_logvars = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(()))
            for t in ALL_TASKS}).to(DEVICE)
    optimizer = make_optimizer(core, task_logvars, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda_factory(cfg))
    # FIX (v11.2): torch.amp.GradScaler exists only on torch >= 2.3; fall
    # back to the (deprecated but present) torch.cuda.amp.GradScaler.
    gs_cls = getattr(torch.amp, "GradScaler", None) \
        or getattr(torch.cuda.amp, "GradScaler")
    scaler = gs_cls("cuda", enabled=(
        cfg.amp and DEVICE.type == "cuda" and not AMP_BF16))
    ema = ModelEMA(model, cfg.ema_decay)

    # ---- resume support (Kaggle's ~12h session cap) ----
    best = -1.0
    start_ep = 0
    ckpt_best = os.path.join(WORK_DIR, "best_v11.pt")
    ckpt_last = os.path.join(WORK_DIR, "last_v11.pt")
    csv_path = os.path.join(WORK_DIR, "training_log.csv")
    resumed = False
    if getattr(cfg, "resume", False) and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=DEVICE, weights_only=False)
        core.load_state_dict(ck["model"], strict=False)
        ema.shadow = {k: v.detach().clone().float()
                      for k, v in ck["ema"].items()}
        start_ep = int(ck.get("epoch", 0))
        best = float(ck.get("val_miou", -1.0))
        for _ in range(start_ep):
            scheduler.step()                       # keep LR schedule in sync
        resumed = True
        print(f"  [resume] {ckpt_last} @ epoch {start_ep} "
              f"(best mIoU {best:.3f}; task_logvars reset)")
    with open(csv_path, "a" if resumed else "w") as f:
        if not resumed:
            f.write("epoch,phase,lr,loss,miou,val_loss,val_miou,change_f1,"
                    "carbon_mae,hydro_mae,canopy_mae\n")

    _OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    for ep1 in range(start_ep + 1, cfg.epochs + 1):
        phase = phase_for_epoch(ep1, cfg)
        core.set_phase(phase)
        train_stream.set_epoch(ep1)
        t0 = time.time()
        # FIX (v11.4.2): the auto/mirror batch choice is only a heuristic —
        # if the epoch OOMs, halve the batch, rebuild the loaders and retry
        # instead of losing the session. (Batches already stepped in the
        # dying epoch simply run again — deterministic per epoch seed.)
        while True:
            try:
                tr = run_epoch(model, train_loader, cfg, phase, task_logvars,
                               optimizer, scaler, ema, alpha)
                break
            except _OOM:
                gc.collect(); torch.cuda.empty_cache()
                if cfg.batch_size <= 1:
                    raise
                cfg.batch_size = max(1, cfg.batch_size // 2)
                print(f"  [gpu] CUDA OOM -> retrying epoch {ep1} with "
                      f"batch_size={cfg.batch_size}")
                train_loader, train_stream = make_loader(
                    train_scenes, cfg, True, cfg.max_train_patches)
                val_loader, _ = make_loader(val_scenes, cfg, False,
                                            cfg.max_val_patches)
                train_stream.set_epoch(ep1)
        scheduler.step()
        ema.store(model); ema.copy_to(model)
        va = run_epoch(model, val_loader, cfg, phase=3)
        ema.restore(model)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"ep {ep1:3d}/{cfg.epochs} [P{phase}] lr={lr_now:.2e} "
              f"loss={tr['loss']:.4f} miou={tr['miou']:.3f} | "
              f"val loss={va['loss']:.4f} miou={va['miou']:.3f} "
              f"F1={va['change_f1']:.3f} hydMAE={va['hydro_mae']:.4f} "
              f"canMAE={va['canopy_mae']:.3f} ({time.time() - t0:.0f}s)")
        with open(csv_path, "a") as f:
            f.write(f"{ep1},{phase},{lr_now:.3e},{tr['loss']:.5f},"
                    f"{tr['miou']:.4f},{va['loss']:.5f},{va['miou']:.4f},"
                    f"{va['change_f1']:.4f},{va['carbon_mae']:.5f},"
                    f"{va['hydro_mae']:.5f},{va['canopy_mae']:.5f}\n")
        payload = {"model": core.state_dict(), "ema": ema.shadow,
                   "norm_mean": mean, "norm_var": var, "alpha": alpha,
                   "cfg": asdict(cfg), "epoch": ep1, "val_miou": va["miou"],
                   "version": VERSION, "dataset_fingerprint": fp,
                   "seg_temperature": cfg.seg_temperature,
                   "change_conformal_q": cfg.change_conformal_q}
        torch.save(payload, ckpt_last)
        if va["miou"] >= best:
            best = va["miou"]
            torch.save(payload, ckpt_best)
            print(f"  [ckpt] new best val mIoU={best:.3f} -> {ckpt_best}")

    cfg.save(os.path.join(WORK_DIR, "config_v11.json"))
    with open(os.path.join(WORK_DIR, "dataset_version.json"), "w") as f:
        json.dump({"fingerprint": fp, "n_files": nf,
                   "train_dir": cfg.train_dir,
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())}, f, indent=2)
    # post-training calibration is fault-isolated (v8 V8-7 lesson)
    try:
        ema.copy_to(model)
        calibrate_temperature(model, val_loader, cfg)
        calibrate_change_conformal(model, val_loader, cfg)
        # FIX (v11.1): do NOT clobber ckpt_best (the best-mIoU epoch) with
        # final-epoch weights; the calibrated final model gets its own file.
        ckpt_final = os.path.join(WORK_DIR, "final_calibrated_v11.pt")
        torch.save({"model": model.state_dict(), "ema": ema.shadow,
                    "norm_mean": mean, "norm_var": var, "alpha": alpha,
                    "cfg": asdict(cfg), "epoch": cfg.epochs,
                    "val_miou": best, "version": VERSION,
                    "dataset_fingerprint": fp,
                    "seg_temperature": cfg.seg_temperature,
                    "change_conformal_q": cfg.change_conformal_q}, ckpt_final)
        print(f"  [calibration] final calibrated model -> {ckpt_final}")
    except Exception as e:
        print(f"  [calibration] FAILED (non-fatal, ckpt intact): {e}")
    print(f"\nTraining complete. best val mIoU={best:.3f}\n{ckpt_best}")
    return ckpt_best


# ==============================================================================
# 15. CHECKPOINT / EVAL / CALIBRATION
# ==============================================================================
def load_checkpoint(path, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_d = ckpt["cfg"]
    for k in Config._TUPLES:
        if k in cfg_d:
            cfg_d[k] = tuple(cfg_d[k])
    cfg = Config(**cfg_d)
    model = BlueSentinelV11(cfg, ckpt["norm_mean"], ckpt["norm_var"]).to(device)
    src = ckpt.get("ema", ckpt["model"])
    for k, v in model.state_dict().items():
        v.copy_(src[k].to(v.dtype))
    cfg.seg_temperature = float(ckpt.get("seg_temperature", 1.0))
    cfg.change_conformal_q = float(ckpt.get("change_conformal_q", 0.0))
    model.eval()
    model = wrap_mirror(model, cfg.mirror_gpus)   # mirror at eval/export too
    print(f"Loaded {path} (ep={ckpt.get('epoch')}, "
          f"mIoU={ckpt.get('val_miou', 0):.3f})")
    return model, cfg


def evaluate(model, cfg, scene_descs, split="test"):
    loader, _ = make_loader(scene_descs, cfg, False, cfg.max_val_patches)
    res = run_epoch(model, loader, cfg, phase=3)
    cm = ConfusionMatrix(cfg.num_classes); cm.cm = res["cm"]
    per = cm.iou()
    print(f"\n-- {split} results --")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:18s} IoU={per[i]:.3f}  px={res['cm'][i].sum():,}")
    print(f"  mIoU={res['miou']:.3f} | change F1={res['change_f1']:.3f} | "
          f"hydro MAE={res['hydro_mae']:.4f} m | canopy MAE="
          f"{res['canopy_mae'] * cfg.height_scale:.2f} m")
    np.savetxt(os.path.join(WORK_DIR, f"confusion_{split}.csv"),
               res["cm"], fmt="%d", delimiter=",")
    with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
        json.dump({n: {"iou": float(per[i])} for i, n in
                   enumerate(CLASS_NAMES)} | {"mIoU": res["miou"]}, f, indent=2)
    return res


@torch.no_grad()
def calibrate_temperature(model, val_loader, cfg, max_batches=10):
    probs, labels = [], []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with amp_context(cfg):
            out = model(batch["x"].float(), batch["dates"].float())
        p = F.softmax(out["seg"].float(), 1).permute(0, 2, 3, 1) \
            .reshape(-1, cfg.num_classes).cpu().numpy()
        y = batch["label"].reshape(-1).cpu().numpy()
        v = y != cfg.ignore_label
        if v.any():
            probs.append(p[v]); labels.append(y[v])
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
    return best_t


@torch.no_grad()
def calibrate_change_conformal(model, val_loader, cfg, max_batches=15):
    """Split-conformal change threshold + certificate (marginal coverage only
    under exchangeability: same region/season/sensors/date range)."""
    if "change" not in cfg.active_heads:
        return None
    alpha = float(cfg.conformal_alpha)
    scores = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with amp_context(cfg):
            out = model(batch["x"].float(), batch["dates"].float())
        p = torch.sigmoid(out["change"].float())[:, 0].cpu().numpy()
        # FIX (v11.3): tensor lives on the GPU -> .cpu() before .numpy()
        # (this crashed the whole smoke on Kaggle T4s; CPU runs masked it)
        y = batch["change"].cpu().numpy()
        pos = y == 1
        if pos.any():
            scores.append((1.0 - p)[pos])
    if not scores:
        print("  [conformal] no positive change pixels; keeping fixed threshold.")
        return None
    s = np.concatenate(scores); n = len(s)
    if n < 50:
        print(f"  [conformal] only {n} positives; skipping.")
        return None
    if n < 1000:
        print(f"  [conformal] WARNING: n={n} is small; guarantee is loose.")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    thr = float(1.0 - np.quantile(s, level))
    cfg.change_conformal_q = thr
    cert = {"type": "split-conformal change alert calibration",
            "alpha": alpha, "threshold": thr, "n_positive_pixels": n,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "caveat": ("Marginal coverage (1-alpha) holds only under "
                       "exchangeability of calibration and deployment data. "
                       "Recalibrate on region/season/sensor change.")}
    with open(os.path.join(WORK_DIR, "calibration_certificate.json"), "w") as f:
        json.dump(cert, f, indent=2)
    print(f"  [conformal] threshold={thr:.3f} (alpha={alpha}, n={n:,})")
    return thr


# ==============================================================================
# 16. TILED INFERENCE (Hann blend + flip/rot TTA + MC-dropout + modality flags)
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


def _set_mc_dropout(model, on):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train() if on else m.eval()


@torch.no_grad()
def predict_scene(model, scene, cfg, batch_size=None, tta=None, overlap=None,
                  sar_only=False, optical_only=False, return_tiles=False):
    batch_size = batch_size or cfg.infer_batch
    if isinstance(model, nn.DataParallel):
        # mirror mode: at least one tile per GPU per forward
        batch_size = max(batch_size, len(model.device_ids))
    tta = cfg.infer_tta if tta is None else tta
    overlap = cfg.infer_overlap if overlap is None else overlap
    ps, C, S = cfg.patch_size, cfg.num_classes, cfg.num_species
    x_np = scene["x"]
    T, _, h, w = x_np.shape
    msp = getattr(cfg, "max_scene_pixels", 8_000_000)
    if h * w > msp:
        raise ValueError(
            f"scene {h}x{w} = {h * w:,} px exceeds cfg.max_scene_pixels="
            f"{msp:,} (RAM guard: full-cube tiling must stay <=30 GB). "
            f"Build smaller AOIs or raise the limit deliberately.")
    if sar_only:
        x_np = x_np.copy(); x_np[:, 0:11] = 0.0
    if optical_only:
        x_np = x_np.copy(); x_np[:, 11:17] = 0.0
    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        x_np = np.pad(x_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    H, W = x_np.shape[-2:]
    kernel = _hann_kernel(ps)
    acc_shapes = {"seg": (H, W, C), "species": (H, W, S),
                  "carbon": (H, W, 2), "eco": (H, W, 2),
                  "risk": (H, W), "quality": (H, W), "boundary": (H, W),
                  "hydro_mu": (H, W), "hydro_var": (H, W), "canopy": (H, W),
                  "change": (H, W), "change_year": (H, W), "recovery": (H, W),
                  "damage": (H, W)}
    acc = {k: np.zeros(s, np.float32) for k, s in acc_shapes.items()}
    mc = max(0, int(cfg.mc_samples))
    mc_sq = {k: np.zeros_like(v) for k, v in acc.items()} if mc > 1 else None
    nan_count = {k: 0 for k in acc}
    weight = np.zeros((H, W), np.float32)
    coords = [(i, j) for i in _tile_starts(H, ps, overlap)
              for j in _tile_starts(W, ps, overlap)]
    variants = [(0, False, False)] if not tta else \
        [(0, False, False), (0, True, False), (0, False, True),
         (1, False, False)]
    print(f"Inference: {h}x{w} -> {H}x{W} | tiles={len(coords)} TTA={tta} "
          f"MC={mc} sar_only={sar_only} optical_only={optical_only}")
    model.eval()

    def extract(out):
        return {
            "seg": F.softmax(out["seg"].float(), 1).permute(0, 2, 3, 1),
            "species": F.softmax(out["species"].float(), 1).permute(0, 2, 3, 1),
            "carbon": out["carbon"].permute(0, 2, 3, 1),
            "eco": out["eco"].permute(0, 2, 3, 1),
            "risk": torch.sigmoid(out["risk"].float())[:, 0],
            "quality": torch.sigmoid(out["quality"].float())[:, 0],
            "boundary": torch.sigmoid(out["boundary"].float())[:, 0],
            "hydro_mu": out["hydro_mu"][:, 0], "hydro_var": out["hydro_var"][:, 0],
            "canopy": out["canopy"][:, 0],
            "change": torch.sigmoid(out["change"].float())[:, 0],
            "change_year": out["change_year"][:, 0],
            "recovery": torch.sigmoid(out["recovery"].float())[:, 0],
            "damage": torch.sigmoid(out["damage"].float())[:, 0]}

    for k90, fh, fv in variants:
        for s in range(0, len(coords), batch_size):
            chunk = coords[s:s + batch_size]
            tiles = np.stack([x_np[:, :, i:i + ps, j:j + ps] for i, j in chunk])
            if fh: tiles = tiles[..., ::-1]               # width flip
            if fv: tiles = tiles[..., ::-1, :]            # height flip (fixed)
            if k90: tiles = np.rot90(tiles, 1, axes=(-2, -1))
            xb = torch.from_numpy(np.ascontiguousarray(tiles)).to(DEVICE).float()
            db = torch.from_numpy(
                np.repeat(scene["dates"][None], len(chunk), 0)).to(DEVICE)
            reps = mc if mc > 1 else 1
            for rep in range(reps):
                if mc > 1 and rep > 0:
                    _set_mc_dropout(model, True)
                    with amp_context(cfg):
                        out = model(xb, db)
                    model.eval()
                else:
                    with amp_context(cfg):
                        out = model(xb, db)
                preds = {k: v.cpu().numpy() for k, v in extract(out).items()}
                for name in acc:
                    pr = preds[name]
                    nan_count[name] += int(np.isnan(pr).sum())
                    pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
                    if k90:
                        pr = np.rot90(pr, -1, axes=(1, 2))
                    if fv:
                        pr = pr[:, ::-1]
                    if fh:
                        pr = pr[:, :, ::-1]
                    for n_i, (i, j) in enumerate(chunk):
                        sl = np.s_[i:i + ps, j:j + ps]
                        if pr.ndim == 4:
                            acc[name][sl] += pr[n_i] * kernel[..., None]
                            if mc_sq is not None:
                                mc_sq[name][sl] += pr[n_i] ** 2 * kernel[..., None]
                        else:
                            acc[name][sl] += pr[n_i] * kernel
                            if mc_sq is not None:
                                mc_sq[name][sl] += pr[n_i] ** 2 * kernel
                # FIX (v11.2, CRITICAL): accumulate the tile weight once per
                # repetition, mirroring `acc`. The old code added it once per
                # TTA variant, so with mc_samples=R every continuous product
                # (hydro/carbon/canopy/change_year/risk/eco/...) came out R*x
                # too large and the epistemic-variance formula turned negative
                # garbage (clamped to 0). Now out = true MC mean and
                # mc_sq/w - out^2 is the exact TTA+MC variance.
                for i, j in chunk:
                    weight[i:i + ps, j:j + ps] += kernel

    w_safe = np.maximum(weight, 1e-6)
    out = {}
    for name in acc:
        out[name] = acc[name] / (w_safe[..., None] if acc[name].ndim == 3
                                else w_safe)
    for name, cnt in nan_count.items():
        if cnt:
            print(f"  [predict] WARNING: {cnt} NaNs zeroed in '{name}'")

    seg = out["seg"] / np.maximum(out["seg"].sum(-1, keepdims=True), 1e-6)
    temp = float(getattr(cfg, "seg_temperature", 1.0))
    if temp != 1.0:
        lp = np.log(np.clip(seg, 1e-6, 1.0)) / temp
        lp -= lp.max(-1, keepdims=True)
        seg = np.exp(lp); seg /= seg.sum(-1, keepdims=True)
    sp = out["species"] / np.maximum(out["species"].sum(-1, keepdims=True), 1e-6)
    d0 = float(scene["dates"][0])
    thr = cfg.change_conformal_q if cfg.change_conformal_q > 0 \
        else cfg.change_threshold
    cm_ = out["carbon"][..., 0] * cfg.carbon_scale
    cs = np.sqrt(np.exp(np.clip(out["carbon"][..., 1], -6, 8))) \
        * cfg.carbon_scale
    res = {
        "seg_probs": seg[:h, :w].astype(np.float32),
        "classes": seg.argmax(-1).astype(np.uint8)[:h, :w],
        "flood_risk": (seg[..., 1] + seg[..., 2]).astype(np.float32)[:h, :w],
        "species_probs": sp[:h, :w].astype(np.float32),
        "species_classes": sp.argmax(-1).astype(np.uint8)[:h, :w],
        "change_prob": out["change"][:h, :w].astype(np.float32),
        "change_alert": (out["change"] > thr).astype(np.uint8)[:h, :w],
        "change_year": (d0 + np.clip(out["change_year"], 0, 1)
                        * cfg.max_time_gap_years).astype(np.float32)[:h, :w],
        "recovery_prob": out["recovery"][:h, :w].astype(np.float32),
        "damage_prob": out["damage"][:h, :w].astype(np.float32),
        "quality_prob": out["quality"][:h, :w].astype(np.float32),
        "risk_index": (out["risk"] * 100.0).astype(np.float32)[:h, :w],
        "carbon_mean": cm_.astype(np.float32)[:h, :w],
        "carbon_std": cs.astype(np.float32)[:h, :w],
        "carbon_ci_lower": np.maximum(cm_ - 1.96 * cs, 0).astype(np.float32)[:h, :w],
        "carbon_ci_upper": (cm_ + 1.96 * cs).astype(np.float32)[:h, :w],
        "eco_temp": (out["eco"][..., 0] * cfg.eco_scale[0])
        .astype(np.float32)[:h, :w],
        "eco_moisture": out["eco"][..., 1].astype(np.float32)[:h, :w],
        "hydro_depth_m": out["hydro_mu"].astype(np.float32)[:h, :w],
        "hydro_std_m": np.sqrt(np.maximum(out["hydro_var"], 0))
        .astype(np.float32)[:h, :w],
        "canopy_height_m": (out["canopy"] * cfg.height_scale)
        .astype(np.float32)[:h, :w],
    }
    res["mangrove_extent"] = ((res["classes"] == 3) |
                              (res["classes"] == 4)).astype(np.uint8)
    if mc_sq is not None:
        var = np.maximum(mc_sq["seg"] / w_safe[..., None] - out["seg"] ** 2, 0)
        res["seg_epistemic_var"] = var.mean(-1).astype(np.float32)[:h, :w]
    if return_tiles:
        padded = dict(scene); padded["x"] = x_np
        return res, padded, coords
    return res


# ==============================================================================
# 17. GRAD-CAM (v8 port, torch autograd)
# ==============================================================================
def grad_cam_tiles(model, scene_pad, coords, cfg, class_idx=None, n_tiles=None):
    model = unwrap(model)      # one tile at a time on one GPU is what CAM wants
    class_idx = cfg.gradcam_class if class_idx is None else int(class_idx)
    n_tiles = n_tiles or cfg.gradcam_tiles
    ps = cfg.patch_size
    acts = {}
    hook = model.stages[-1][-1].register_forward_hook(
        lambda m, i, o: acts.__setitem__("bn", o))
    results = []
    try:
        for (i, j) in coords[:max(1, n_tiles)]:
            x = torch.from_numpy(
                scene_pad["x"][:, :, i:i + ps, j:j + ps][None]) \
                .to(DEVICE).float()
            db = torch.from_numpy(scene_pad["dates"][None]).to(DEVICE).float()
            with torch.enable_grad():
                out = model(x, db)
                cls = min(class_idx, out["seg"].shape[1] - 1)
                score = out["seg"][0, cls].sum()
                bn = acts["bn"]
                grads = torch.autograd.grad(score, bn, retain_graph=False)[0]
            a = bn[0, -1].float(); g = grads[0, -1].float()
            wts = g.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((wts * a).sum(0))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            cam = F.interpolate(cam[None, None], size=(ps, ps),
                                mode="bilinear", align_corners=False)[0, 0]
            rgb = scene_pad["x"][-1, [2, 1, 0], i:i + ps, j:j + ps] \
                .transpose(1, 2, 0)
            lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
            rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
            results.append((i, j, cam.detach().cpu().numpy(), rgb))
    finally:
        hook.remove()
    print(f"  [gradcam] {len(results)} heatmaps (class={class_idx})")
    return results


def save_gradcam_figure(cams, out_path, cfg):
    if not cams:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cams)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    axes = np.atleast_2d(axes)
    cname = CLASS_NAMES[min(cfg.gradcam_class, len(CLASS_NAMES) - 1)]
    for r, (i, j, cam, rgb) in enumerate(cams):
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"tile ({i},{j}) RGB")
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
# 18. ACTIVE-LEARNING REVIEW QUEUE (v8 port)
# ==============================================================================
def build_review_queue(products, scene_pad, cfg, out_dir):
    rdir = os.path.join(out_dir, "review_queue")
    os.makedirs(rdir, exist_ok=True)
    ref = products.get("classes", products.get("change_prob"))
    if ref is None:
        print("  [review] no usable products; queue empty.")
        return []
    h, w = ref.shape[:2]
    t = int(min(cfg.review_tile, h, w))          # fixed: small-scene clamp
    thr = cfg.change_conformal_q if cfg.change_conformal_q > 0 \
        else cfg.change_threshold
    cand = []
    for i in range(0, h - t + 1, t):
        for j in range(0, w - t + 1, t):
            score, reasons = 0.0, []
            q = float(np.mean(products["quality_prob"][i:i + t, j:j + t]))
            if q < cfg.review_low_quality:
                score += (cfg.review_low_quality - q) * 2.0
                reasons.append(f"low_quality({q:.2f})")
            if "change_prob" in products:
                cp = products["change_prob"][i:i + t, j:j + t]
                border = float(np.mean(np.abs(cp - thr)
                                       < cfg.review_change_margin))
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
    for idx, (score, i, j, reasons) in enumerate(cand):
        payload = {"x": scene_pad["x"][:, :, i:i + t, j:j + t],
                   "dates": scene_pad["dates"]}
        for key in ("classes", "change_prob", "quality_prob"):
            if key in products:
                payload[f"pred_{key}"] = products[key][i:i + t, j:j + t]
        fname = f"review_{idx:04d}.npz"
        np.savez_compressed(os.path.join(rdir, fname), **payload)
        manifest.append({"score": round(score, 4), "tile": [i, j],
                         "reasons": reasons,
                         "dates": scene_pad["dates"].tolist(), "file": fname})
    with open(os.path.join(rdir, "manifest.json"), "w") as f:
        json.dump({"version": VERSION,
                   "criteria": {"low_quality_below": cfg.review_low_quality,
                                "change_margin": cfg.review_change_margin,
                                "change_threshold": thr},
                   "tiles": manifest}, f, indent=2)
    print(f"  [review] queued {len(manifest)} tiles -> {rdir}")
    return manifest


# ==============================================================================
# 19. HMM VITERBI TEMPORAL SMOOTHING (v8, framework-agnostic numpy)
# ==============================================================================
def _viterbi_chunk(lp, logA):
    T, N, C = lp.shape
    delta = lp[0]
    backs = np.empty((T - 1, N, C), np.uint8)
    for t in range(1, T):
        cand = delta[:, :, None] + logA[None, :, :]
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
    print(f"  [smooth] Viterbi-smoothed {T} steps over {H * W:,} px (stay={stay})")
    return [out[t].reshape(H, W) for t in range(T)]


def smooth_mode(prob_dir, cfg, out_path=None):
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
# 20. SELF-SUPERVISED PRETRAINING (SimMIM over the SSM encoder)
# ==============================================================================
class ReconDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d8 = cfg.dim * 8
        self.up1 = nn.ConvTranspose2d(d8, cfg.dim * 2, 2, 2)
        self.c1 = ConvBlock(cfg.dim * 2, cfg.dim * 2)
        self.up2 = nn.ConvTranspose2d(cfg.dim * 2, cfg.dim, 2, 2)
        self.c2 = ConvBlock(cfg.dim, cfg.dim)
        self.head = nn.Conv2d(cfg.dim, N_CH, 1)

    def forward(self, z):                              # (B,8D,32,32)->ps/4
        return self.head(self.c2(self.up2(self.c1(self.up1(z)))))


def _masked_block(x, ratio, block):
    B, T, C, H, W = x.shape
    g = max(H // block, 1)
    cells = (torch.rand(B, T, g, g, 1, device=x.device) < ratio).float()
    mask = F.interpolate(cells.reshape(B * T, 1, g, g), size=(H, W),
                         mode="nearest").reshape(B, T, 1, H, W)
    return x * (1 - mask), mask


def pretrain(cfg=CFG):
    print(f"\n=== v{VERSION} SELF-SUPERVISED PRETRAINING (SimMIM-SSM) ===")
    scenes = discover_scenes(cfg.train_dir)
    assert scenes, "pretrain: set cfg.train_dir"
    fp, nf = dataset_fingerprint(scenes)
    print(f"  fingerprint: {fp} ({nf} files)")
    mean, var, _ = load_or_compute_stats(scenes, cfg)
    # FIX (v11.1): SSL pretraining scenes are typically UNLABELLED — bypass
    # the label-coverage filter or the generator emits 0 patches.
    loader, stream = make_loader(scenes, cfg, False, cfg.max_train_patches,
                                 require_labels=False)
    model = BlueSentinelV11(cfg, mean, var).to(DEVICE)
    enc_params = [p for n, p in model.named_parameters()
                  if n.split(".")[0] in ("opt_stem", "radar_stem", "aux_stem",
                                         "fuse", "stages", "merges")]
    recon = ReconDecoder(cfg).to(DEVICE)
    opt = torch.optim.AdamW(list(enc_params) + list(recon.parameters()),
                            lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.pretrain_epochs):
        model.train(); recon.train()
        stream.set_epoch(ep)
        losses = []; t0 = time.time()
        for batch in loader:
            x = batch["x"].to(DEVICE).float()
            with amp_context(cfg):
                xm, mask = _masked_block(x, cfg.pretrain_mask_ratio,
                                         cfg.pretrain_block)
                B, T = x.shape[:2]
                f = model.opt_stem  # encoder path only:
                z = model.fuse(torch.cat([
                    model.opt_stem(xm.reshape(B * T, N_CH,
                                              *x.shape[-2:])[:, 0:11]),
                    model.radar_stem(xm.reshape(B * T, N_CH,
                                                *x.shape[-2:])[:, 11:17]),
                    model.aux_stem(xm.reshape(B * T, N_CH,
                                              *x.shape[-2:])[:, 17:25])],
                    dim=1)).view(B, T, cfg.dim, *x.shape[-2:])
                for i, stage in enumerate(model.stages):
                    for blk in stage:
                        z = blk(z)
                    if i < len(model.merges):
                        z = model.merges[i](z)
                rec = recon(z.mean(1))
                xn = ((x - model.in_mean) / model.in_std)[:, -1]
                # FIX (v11.2): the recon decoder outputs at H/2 (2 of the 3
                # downsamples are undone). The old target was avg-pooled 4x
                # then bilinearly UPSAMPLED to H/2 — a doubly blurred target
                # (and an over-coarsened mask). Pool 2 instead; the interpolate
                # stays only as an odd-size guard.
                tgt = F.avg_pool2d(xn, 2)
                m = F.avg_pool2d(mask[:, -1].expand(-1, N_CH, -1, -1)
                                 .reshape(B, N_CH, *x.shape[-2:]), 2)
                if tgt.shape[-2:] != rec.shape[-2:]:
                    tgt = F.interpolate(tgt, size=rec.shape[-2:],
                                        mode="bilinear", align_corners=False)
                    m = F.interpolate(m, size=rec.shape[-2:], mode="nearest")
                m = m > 0.5
                loss = ((rec - tgt).abs() * m).sum() / (m.sum() * N_CH + 1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        print(f"  pretrain {ep + 1}/{cfg.pretrain_epochs} "
              f"loss={np.mean(losses):.4f} ({time.time() - t0:.0f}s)")
    sd = {k: v for k, v in model.state_dict().items()
          if k.split(".")[0] in ("opt_stem", "radar_stem", "aux_stem",
                                 "fuse", "stages", "merges", "in_mean",
                                 "in_std")}
    out = os.path.join(WORK_DIR, "pretrain_encoder_v11.pt")
    torch.save(sd, out)
    print(f"Pretrained encoder saved: {out}\nFine-tune: --pretrain {out}")
    return out


# ==============================================================================
# 21. EXPORT: GeoTIFF + STAC + GeoJSON + MRV + dashboard + gradcam + review
# ==============================================================================
def export_geotiff(reference_tif, array, out_path, dtype="float32", nodata=None):
    try:
        import rasterio
    except ImportError:
        print(f"  [export] rasterio missing; NPZ fallback "
              f"({os.path.basename(out_path)})")
        np.savez(out_path.replace(".tif", ".npz"), array=array)
        return None
    # FIX (v11.2): a missing/unreadable reference GeoTIFF must fall back to
    # NPZ (it used to hard-crash with RasterioIOError).
    try:
        with rasterio.open(reference_tif) as src:
            profile = src.profile.copy()
    except Exception as e:
        print(f"  [export] cannot read reference '{reference_tif}' "
              f"({str(e)[:80]}); NPZ fallback ({os.path.basename(out_path)})")
        np.savez(out_path.replace(".tif", ".npz"), array=array)
        return None
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


def _pixel_area_ha(reference_tif, cfg):
    try:
        import rasterio
        with rasterio.open(reference_tif) as src:
            a, _, _, _, e, _ = src.transform[:6]
            if src.crs and src.crs.is_geographic:
                return (cfg.pixel_size_m ** 2) / 1e4
            return abs(a * e) / 1e4
    except Exception:
        return (cfg.pixel_size_m ** 2) / 1e4


def export_change_geojson(products, reference_tif, out_path, cfg, period=""):
    try:
        import rasterio
        from rasterio.features import shapes
    except ImportError:
        print("  [export] rasterio missing; GeoJSON skipped.")
        return None
    alert = products.get("change_alert")
    if alert is None or alert.sum() == 0:
        print("  [export] no change alerts above threshold.")
        return None
    # FIX (v11.2): guard the reference read (was a hard crash)
    try:
        with rasterio.open(reference_tif) as src:
            transform = src.transform
            crs = src.crs.to_string() if src.crs else None
    except Exception as e:
        print(f"  [export] cannot read reference '{reference_tif}' "
              f"({str(e)[:80]}); GeoJSON skipped.")
        return None
    feats = [{"type": "Feature",
              "properties": {"alert": "mangrove_loss", "period": period,
                             "confidence": cfg.change_conformal_q
                             if cfg.change_conformal_q > 0
                             else cfg.change_threshold},
              "geometry": geom}
             for geom, _ in shapes(alert, mask=alert > 0, transform=transform)]
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name",
                           "properties": {"name": crs or "EPSG:4326"}},
                   "features": feats}, f)
    print(f"  saved {out_path} ({len(feats)} polygons)")
    return out_path


def write_stac_item(out_dir, products_meta, reference_tif, scene_id, cfg):
    item = {"stac_version": "1.0.0", "type": "Feature", "id": scene_id,
            "collection": "blue-sentinel-v11",
            "properties": {
                "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "processing:software": f"blue-sentinel v{VERSION}",
                "mangrove:classes": CLASS_NAMES,
                "mangrove:species": list(cfg.species_names),
                "mangrove:active_heads": list(cfg.active_heads),
                "mangrove:temporal_steps": cfg.temporal_steps},
            "assets": {}, "links": []}
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
    for i, n in enumerate(CLASS_NAMES):
        rows.append((f"area_class_{i}_{n.replace(' ', '_').replace('/', '_')}",
                     f"{float((products['classes'] == i).sum()) * pa_ha:.2f}",
                     "ha"))
    m = products["mangrove_extent"] > 0
    for i, n in enumerate(cfg.species_names):
        rows.append((f"area_species_{i}_{n.replace(' ', '_')}",
                     f"{float(((products['species_classes'] == i) & m).sum()) * pa_ha:.2f}", "ha"))
    if m.any():
        agb = float(products["carbon_mean"][m].sum()) * pa_ha
        var = float((products["carbon_std"][m] ** 2).sum()) * (pa_ha ** 2)
        std = math.sqrt(max(var, 0.0))
        co2e = agb * cfg.c_fraction * cfg.carbon_to_co2e
        cstd = std * cfg.c_fraction * cfg.carbon_to_co2e
        rows += [("agb_stock", f"{agb:.1f}", "t"),
                 ("agb_stock_std", f"{std:.1f}", "t"),
                 ("co2e_stock", f"{co2e:.1f}", "tCO2e"),
                 ("co2e_ci95_low", f"{max(co2e - 1.96 * cstd, 0):.1f}", "tCO2e"),
                 ("co2e_ci95_high", f"{co2e + 1.96 * cstd:.1f}", "tCO2e"),
                 ("canopy_height_mean",
                  f"{float(products['canopy_height_m'][m].mean()):.2f}", "m"),
                 ("hydro_depth_mean",
                  f"{float(products['hydro_depth_m'][m].mean()):.3f}", "m")]
    if "change_alert" in products:
        rows.append(("change_alert_area",
                     f"{float(products['change_alert'].sum()) * pa_ha:.2f}",
                     "ha"))
        mch = products["change_alert"] > 0
        if mch.any():
            rows.append(("change_year_median",
                         f"{float(np.median(products['change_year'][mch])):.2f}",
                         "year"))
    rows.append(("recovery_area_p50",
                 f"{float((products['recovery_prob'] > 0.5).sum()) * pa_ha:.2f}",
                 "ha"))
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
    b64 = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from io import BytesIO
        panels = [("classes", "Land cover", "cls"),
                  ("change_prob", "Change prob", "Reds"),
                  ("carbon_mean", "AGB (t/ha)", "YlGn"),
                  ("canopy_height_m", "Canopy (m)", "Greens"),
                  ("hydro_depth_m", "Water depth (m)", "Blues"),
                  ("quality_prob", "Pixel usability", "viridis")]
        cmap_cls = mcolors.ListedColormap(CLASS_COLORS)
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (key, title, cm) in zip(axes, panels):
            if cm == "cls":
                ax.imshow(products[key], cmap=cmap_cls, vmin=0, vmax=4)
            elif cm in ("Reds", "viridis"):
                ax.imshow(products[key], cmap=cm, vmin=0, vmax=1)
            else:
                ax.imshow(products[key], cmap=cm)
            ax.set_title(title); ax.axis("off")
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"  [dashboard] preview render failed: {e}")
    rows_html = "".join(
        f"<tr><td>{r[0]}</td><td style='text-align:right'>{r[1]}</td>"
        f"<td>{r[2]}</td></tr>" for r in rows[1:])
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>MRV dashboard - {scene_id}</title><style>"
            "body{font-family:Arial,sans-serif;margin:24px;background:#0b1e2d;"
            "color:#e8f1f5}h1{color:#39c2d7;font-size:22px}"
            "table{border-collapse:collapse;margin-top:18px}"
            "td,th{border:1px solid #2a4a5f;padding:6px 12px;font-size:14px}"
            "img{max-width:100%;border:1px solid #2a4a5f;border-radius:8px;"
            "margin-top:14px}.meta{color:#8fb8c9;font-size:13px}</style>"
            "</head><body>"
            f"<h1>Blue Sentinel v{VERSION} &mdash; MRV Dashboard</h1>"
            f"<div class='meta'>scene: {scene_id} | heads: "
            f"{', '.join(cfg.active_heads)} | generated "
            f"{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} UTC</div>"
            + (f"<img alt='products' src='data:image/png;base64,{b64}'>"
               if b64 else "")
            + "<table><tr><th>metric</th><th>value</th><th>unit</th></tr>"
            + rows_html + "</table></body></html>")
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  saved {path}")


def export_products(model, cfg, scene_path, ref_tif, out_dir, sar_only=False,
                    optical_only=False, gradcam=False, review=True):
    os.makedirs(out_dir, exist_ok=True)
    desc = ("folder", scene_path) if os.path.isdir(scene_path) \
        else ("npz", scene_path)
    scene = load_scene(desc, cfg)
    products, scene_pad, coords = predict_scene(
        model, scene, cfg, sar_only=sar_only, optical_only=optical_only,
        return_tiles=True)
    plan = [
        ("classes.tif", "classes", "uint8", 255, "Land-cover class"),
        ("seg_probs.tif", "seg_probs", "float32", None, "Per-class probability"),
        ("species_classes.tif", "species_classes", "uint8", 255, "Species"),
        ("carbon_mean.tif", "carbon_mean", "float32", -999, "AGB t/ha"),
        ("carbon_std.tif", "carbon_std", "float32", -999, "AGB std t/ha"),
        ("change_prob.tif", "change_prob", "float32", -999, "Change prob"),
        ("change_alert.tif", "change_alert", "uint8", 255, "Change alert"),
        ("change_year.tif", "change_year", "float32", -999, "Year of change"),
        ("recovery_prob.tif", "recovery_prob", "float32", -999, "Recovery"),
        ("damage_prob.tif", "damage_prob", "float32", -999, "Damage/debris"),
        ("risk_index.tif", "risk_index", "float32", -999, "Risk 0-100"),
        ("eco_temperature.tif", "eco_temp", "float32", -999, "LST C"),
        ("eco_moisture.tif", "eco_moisture", "float32", -999, "Moisture"),
        ("quality.tif", "quality_prob", "float32", -999, "Pixel usability"),
        ("hydro_depth_m.tif", "hydro_depth_m", "float32", -999, "Water depth m"),
        ("hydro_std_m.tif", "hydro_std_m", "float32", -999, "Depth std m"),
        ("canopy_height_m.tif", "canopy_height_m", "float32", -999,
         "GEDI canopy height m"),
        ("mangrove_extent.tif", "mangrove_extent", "uint8", 255, "Mangrove"),
        ("seg_epistemic.tif", "seg_epistemic_var", "float32", -999,
         "MC epistemic var"),
    ]
    meta = []
    for fname, key, dtype, nodata, title in plan:
        if key in products:
            p = export_geotiff(ref_tif, products[key],
                               os.path.join(out_dir, fname), dtype=dtype,
                               nodata=nodata)
            if p:
                meta.append((fname, title))
    # v11.1: also dump seg probabilities as a date-stamped NPZ so that
    # `smooth` mode can consume exports of the same AOI at different dates.
    d_last = float(scene["dates"][-1])
    y0 = int(d_last)
    mo = min(12, max(1, int(round((d_last - y0) * 12)) + 1))
    np.savez_compressed(
        os.path.join(out_dir, f"seg_probs_{y0}-{mo:02d}.npz"),
        seg_probs=products["seg_probs"])
    period = f"{float(scene['dates'][0]):.2f}-{float(scene['dates'][-1]):.2f}"
    export_change_geojson(products, ref_tif,
                          os.path.join(out_dir, "change_alerts.geojson"),
                          cfg, period=period)
    scene_id = os.path.splitext(os.path.basename(scene_path.rstrip("/")))[0] \
        or "scene"
    write_stac_item(out_dir, meta, ref_tif, scene_id, cfg)
    rows = mrv_summary(products, ref_tif, out_dir, cfg)
    write_dashboard_html(out_dir, products, rows, cfg, scene_id)
    if gradcam:
        try:
            cams = grad_cam_tiles(model, scene_pad, coords, cfg)
            save_gradcam_figure(cams, os.path.join(out_dir, "gradcam.png"), cfg)
        except Exception as e:
            print(f"  [gradcam] failed (non-fatal): {e}")
    if review:
        try:
            build_review_queue(products, scene_pad, cfg, out_dir)
        except Exception as e:
            print(f"  [review] failed (non-fatal): {e}")
    fp, nf = dataset_fingerprint([desc])
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_path, "reference": ref_tif,
                   "version": VERSION, "active_heads": list(cfg.active_heads),
                   "sar_only": sar_only, "optical_only": optical_only,
                   "dates": scene["dates"].tolist(),
                   "seg_temperature": cfg.seg_temperature,
                   "change_conformal_q": cfg.change_conformal_q,
                   "dataset_fingerprint": fp, "dataset_files": nf}, f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products


# ==============================================================================
# 22. EDGE EXPORT (ONNX; TFLite via onnx2tf when available)
# ==============================================================================
class _OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, dates):
        out = self.model(x, dates)
        return (out["seg"], out["species"], out["change"], out["change_year"],
                out["recovery"], out["damage"], out["quality"],
                out["boundary"], out["carbon"], out["eco"], out["risk"],
                out["hydro_mu"], out["hydro_var"], out["canopy"])


EDGE_OUTPUT_NAMES = ["seg", "species", "change", "change_year", "recovery",
                     "damage", "quality", "boundary", "carbon", "eco", "risk",
                     "hydro_mu", "hydro_var", "canopy"]


def export_edge(cfg=CFG, ckpt=None, out_dir=None):
    ckpt = ckpt or os.path.join(WORK_DIR, "best_v11.pt")
    out_dir = out_dir or WORK_DIR
    print(f"\n=== EDGE EXPORT (ONNX{'' if HAS_MAMBA else ' — fallback scan'}) ===")
    try:
        model, mcfg = load_checkpoint(ckpt)
    except Exception as e:
        print(f"  [edge] cannot load {ckpt}: {e}")
        return None
    wrapper = _OnnxWrapper(model).to(DEVICE).eval()
    ps, T = mcfg.patch_size, mcfg.temporal_steps
    dx = torch.zeros(1, T, N_CH, ps, ps, device=DEVICE)
    dd = torch.zeros(1, T, device=DEVICE)
    opath = os.path.join(out_dir, "blue_sentinel_v11.onnx")
    try:
        torch.onnx.export(wrapper, (dx, dd), opath, opset_version=17,
                          input_names=["cube", "dates"],
                          output_names=EDGE_OUTPUT_NAMES,
                          dynamic_axes={"cube": {0: "batch"},
                                        "dates": {0: "batch"}})
        print(f"  saved {opath} ({os.path.getsize(opath) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  [onnx] export failed: {str(e)[:300]}")
        print("  Hint: the pure-torch scan uses Python loops; export with a "
              "fixed T, or install mamba-ssm so the CUDA kernel path is used "
              "at training and a traced fallback is avoided.")
        return None
    try:
        import onnx2tf  # noqa: F401
        import subprocess as sp
        r = sp.run(["onnx2tf", "-i", opath, "-o",
                    os.path.join(out_dir, "tflite")],
                   capture_output=True, text=True, timeout=1800)
        print(r.stdout[-500:] if r.returncode == 0
              else f"  [tflite] onnx2tf failed:\n{r.stderr[-500:]}")
    except ImportError:
        print("  [tflite] onnx2tf not installed; pip install onnx2tf to get "
              "TFLite. ONNX is ready for onnxruntime-mobile.")
    return opath


# ==============================================================================
# 23. SMOKE TEST — self-asserting, exercises BOTH schemas + every subsystem
# ==============================================================================
def _write_dummy_scene(root, name, cfg, T, H=96, W=96, seed=0, full=True):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for t in range(T):
        keys = {
            "s2": rng.uniform(0, 0.5, (H, W, 10)).astype(np.float32),
            "scl": rng.choice([4, 5, 6, 8, 9], (H, W)).astype(np.float32),
            "s1": rng.uniform(-30, 0, (H, W, 2)).astype(np.float32),
            "s1_std": rng.uniform(0, 3, (H, W, 2)).astype(np.float32),
            "lband": rng.uniform(-35, -5, (H, W, 2)).astype(np.float32),
            "precip": rng.uniform(0, 300, (H, W)).astype(np.float32),
            "inundation": rng.uniform(0, 3000, (H, W)).astype(np.float32),
            "aux": rng.uniform(-2, 2, (H, W, 6)).astype(np.float32),
            "date": np.float32(2020 + t + 0.5),
        }
        if t == T - 1 and full:
            gedi = np.full((H, W), np.nan, np.float32)
            gedi[:, 10:14] = rng.uniform(0, 25, (H, 4)).astype(np.float32)
            keys.update({
                "gedi": gedi,
                "depth": rng.uniform(0, 2000, (H, W)).astype(np.float32),
                "label": rng.choice([0, 1, 2, 3, 4, 255], (H, W)).astype(np.int64),
                "species": rng.choice([0, 1, 2, 255], (H, W)).astype(np.int64),
                "change": rng.choice([0, 0, 0, 1, 255], (H, W)).astype(np.int64),
                "recovery": rng.choice([0, 0, 1, 255], (H, W)).astype(np.int64),
                "debris": rng.choice([0, 0, 1, 255], (H, W)).astype(np.int64),
                "change_year": rng.choice([-999.0, 2021.5], (H, W))
                .astype(np.float32),
                "carbon": rng.uniform(0, 250, (H, W)).astype(np.float32),
                "eco": np.stack([rng.uniform(-5, 45, (H, W)),
                                 rng.uniform(0, 1, (H, W))], -1)
                .astype(np.float32),
                "risk": rng.uniform(0, 1, (H, W)).astype(np.float32)})
        np.savez(os.path.join(d, f"step_{2020 + t}.npz"), **keys)
    return d


def _write_dummy_cube(root, name, cfg, T, H=96, W=96, seed=7):
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    cube = np.zeros((T, 19, H, W), np.float32)
    cube[:, 0:10] = rng.uniform(0, 0.5, (T, 10, H, W))
    cube[:, 10] = rng.choice([4, 5, 6], (T, H, W))
    cube[:, 11:13] = rng.uniform(-30, 0, (T, 2, H, W))
    cube[:, 13:15] = rng.uniform(-35, -5, (T, 2, H, W))
    cube[:, 15] = rng.uniform(0, 300, (T, H, W))
    cube[:, 16] = rng.uniform(0, 3000, (T, H, W))
    cube[-1, 17, :, 10:14] = rng.uniform(0, 25, (H, 4))     # gedi track
    cube[-1, 18] = rng.choice([0, 1, 2, 3], (H, W))          # label
    p = os.path.join(root, f"{name}.npz")
    np.savez(p, cube=cube, dates=np.arange(2020, 2020 + T, dtype=np.float32))
    return p


def smoke_test():
    print(f"\n=== SMOKE TEST v{VERSION} ===")
    cfg = Config(patch_size=64, temporal_steps=3, dim=24, depths=(1, 1, 1, 1),
                 d_state=8, batch_size=2, epochs=3, phase1_end=1,
                 phase2_end=2, max_train_patches=8, max_val_patches=4,
                 min_valid_frac=0.1, max_cloud_frac=0.95, ssm_chunk=64,
                 infer_tta=True, amp=False, gradcam_tiles=2,
                 review_queue_max=4, review_tile=64, pretrain_epochs=1,
                 pretrain_block=16, mc_samples=2, task_weighting="learned")
    cfg.check()

    # -- scan fallback correctness (vs naive recurrence) --
    u = torch.randn(2, 33, 8)
    dt = F.softplus(torch.randn(2, 33, 8))
    A = -torch.exp(torch.randn(8, 4))
    Bm, Cm, Dv = torch.randn(2, 33, 4), torch.randn(2, 33, 4), torch.randn(8)
    y_ref = selective_scan_torch(u, dt, A, Bm, Cm, Dv, chunk=16, use_ckpt=False)
    h = torch.zeros(2, 8, 4); ys = []
    for t in range(33):
        h = torch.exp(dt[:, t].unsqueeze(-1) * A) * h + \
            (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
        ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
    err = float((y_ref - (torch.stack(ys, 1) + u * Dv)).abs().max())
    assert err < 1e-4, f"selective scan mismatch: {err}"
    print(f"  [1/9] selective-scan fallback OK (max err {err:.2e})")

    # -- both data schemas --
    root = os.path.join(WORK_DIR, "smoke_data")
    shutil.rmtree(root, ignore_errors=True)
    for split, n in (("train", 2), ("val", 1)):
        for s in range(n):
            _write_dummy_scene(root, f"{split}/scene_{s}", cfg,
                               cfg.temporal_steps, seed=s)
    cube_path = _write_dummy_cube(os.path.join(root, "cube"), "scene_cube",
                                  cfg, cfg.temporal_steps)
    train_scenes = discover_scenes(os.path.join(root, "train"))
    val_scenes = discover_scenes(os.path.join(root, "val"))
    assert len(train_scenes) == 2 and len(val_scenes) == 1
    validate_heads_vs_data(train_scenes, cfg)
    sc_cube = load_scene(("npz", cube_path), cfg)
    assert sc_cube["x"].shape[1] == N_CH
    sc_fold = load_scene(train_scenes[0], cfg)
    assert sc_fold["x"].shape[1] == N_CH
    assert np.isfinite(sc_fold["targets"]["gedi"]).sum() > 0
    print(f"  [2/9] loaders OK (folder + cube), active heads: "
          f"{list(cfg.active_heads)}")

    # -- model + all losses/metrics --
    mean = np.zeros(N_CH, np.float32); var = np.ones(N_CH, np.float32)
    var[CH_SCL] = 121.0
    model = BlueSentinelV11(cfg, mean, var).to(DEVICE)
    model = wrap_mirror(model, cfg.mirror_gpus)
    core = unwrap(model)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    x0 = torch.from_numpy(sc_fold["x"][None]).to(DEVICE).float()
    d0 = torch.from_numpy(sc_fold["dates"][None]).to(DEVICE).float()
    with torch.no_grad():
        out0 = model(x0[:, :, :, :64, :64], d0)
    assert out0["seg"].shape[-1] == 64 and out0["change"].shape[-1] == 64
    assert out0["seg_deep"].shape[-1] == 32
    print(f"  [3/9] model OK ({n_params:.2f}M params, 14 heads)")

    # -- 3-phase curriculum training --
    train_loader, stream = make_loader(train_scenes, cfg, True, 8)
    val_loader, _ = make_loader(val_scenes, cfg, False, 4)
    tlv = nn.ParameterDict({t: nn.Parameter(torch.zeros(()))
                            for t in ALL_TASKS}).to(DEVICE)
    opt = make_optimizer(model, tlv, cfg)
    ema = ModelEMA(model, 0.9)
    for ep in (1, 2, 3):
        phase = phase_for_epoch(ep, cfg)
        core.set_phase(phase)
        stream.set_epoch(ep)
        tr = run_epoch(model, train_loader, cfg, phase, tlv, opt, None, ema)
        assert np.isfinite(tr["loss"]), "non-finite loss"
        frozen = not next(core.hydro_head.parameters()).requires_grad
        assert frozen == (phase == 1), "curriculum freeze broken"
        print(f"      phase {phase}: loss={tr['loss']:.4f} miou={tr['miou']:.3f}"
              f" hydro_frozen={frozen}")
    va = run_epoch(model, val_loader, cfg, phase=3)
    assert np.isfinite(va["loss"])
    print(f"  [4/9] curriculum + learned task weighting OK "
          f"(val loss={va['loss']:.4f})")

    # -- checkpoint round trip with embedded calibration --
    ckpt = os.path.join(WORK_DIR, "smoke_v11.pt")
    torch.save({"model": core.state_dict(), "ema": ema.shadow,
                "norm_mean": mean, "norm_var": var,
                "alpha": np.ones(cfg.num_classes, np.float32),
                "cfg": asdict(cfg), "epoch": 3, "val_miou": va["miou"],
                "version": VERSION, "dataset_fingerprint": "smoke",
                "seg_temperature": 1.0, "change_conformal_q": 0.0}, ckpt)
    m2, cfg2 = load_checkpoint(ckpt)
    print("  [5/9] checkpoint save/load OK")

    # -- calibration (temperature + conformal) on dummy val --
    calibrate_temperature(m2, val_loader, cfg2, max_batches=2)
    calibrate_change_conformal(m2, val_loader, cfg2, max_batches=2)
    print(f"  [6/9] calibration OK (T={cfg2.seg_temperature:.2f}, "
          f"q={cfg2.change_conformal_q:.3f})")

        # -- tiled inference: padding, TTA, MC, sar-only --
    # FIX: Use np.pad to safely handle PyTorch (T, C, H, W) format
    pad_h = 150 - sc_fold["x"].shape[2]
    pad_w = 200 - sc_fold["x"].shape[3]
    big_x = np.pad(sc_fold["x"], 
                   ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 
                   mode="constant", constant_values=0)
    big = {"x": big_x, "dates": sc_fold["dates"], "targets": sc_fold["targets"]}
    res, scene_pad, coords = predict_scene(m2, big, cfg2, batch_size=2,
                                           tta=True, return_tiles=True)
    assert res["classes"].shape == (150, 200)
    assert np.allclose(res["seg_probs"].sum(-1), 1.0, atol=1e-3)
    assert np.isfinite(res["hydro_depth_m"]).all()
    assert "seg_epistemic_var" in res
    res_sar = predict_scene(m2, big, cfg2, batch_size=2, tta=False,
                            sar_only=True)
    assert np.isfinite(res_sar["change_prob"]).all()
    print(f"  [7/9] tiled inference OK ({len(res)} products, TTA+MC+sar_only)")

    # -- MC-dropout averaging invariance (v11.2 regression guard): a model
    # that always outputs 1.0 must produce products == 1.0 no matter how
    # many MC repetitions run (the v11.1 bug returned mc_samples x too much).
    class _MockConst(nn.Module):
        def forward(self, x, dates):
            n, ps = x.shape[0], x.shape[-1]
            o = lambda *s: torch.ones(n, *s, device=x.device)
            return {"seg": o(cfg.num_classes, ps, ps),
                    "seg_deep": o(cfg.num_classes, ps // 2, ps // 2),
                    "species": o(cfg.num_species, ps, ps),
                    "carbon": o(2, ps, ps), "eco": o(2, ps, ps),
                    "risk": o(1, ps, ps), "quality": o(1, ps, ps),
                    "boundary": o(1, ps, ps), "hydro_mu": o(1, ps, ps),
                    "hydro_var": o(1, ps, ps) + 0.5, "canopy": o(1, ps, ps),
                    "change": o(1, ps, ps), "change_year": o(1, ps, ps),
                    "recovery": o(1, ps, ps), "damage": o(1, ps, ps)}

    cfg_mc = Config(patch_size=64, temporal_steps=cfg.temporal_steps,
                    mc_samples=2, infer_tta=False, infer_overlap=0.0)
    res_mc = predict_scene(_MockConst(), sc_fold, cfg_mc, batch_size=1)
    assert np.allclose(res_mc["hydro_depth_m"], 1.0, atol=1e-4), \
        "MC averaging broken: products scaled by mc_samples"
    assert res_mc["seg_epistemic_var"].max() < 1e-6, \
        "epistemic variance nonzero for a constant model"
    print("  [7b/9] MC-dropout averaging invariance OK "
          "(constant model -> exact mean, zero epistemic var)")

    # -- gradcam / review / smoothing --
    cams = grad_cam_tiles(m2, scene_pad, coords, cfg2, n_tiles=2)
    assert len(cams) == 2 and np.isfinite(cams[0][2]).all()
    rq = build_review_queue(res, scene_pad, cfg2, WORK_DIR)
    sm = smooth_class_series([res["seg_probs"],
                              np.roll(res["seg_probs"], 1, axis=-1)],
                             stay=cfg2.smooth_stay)
    assert sm[0].shape == (150, 200)
    print(f"  [8/9] gradcam ({len(cams)}), review queue ({len(rq)}), "
          f"HMM smoothing OK")

    # -- pretrain + fine-tune weight load --
        # -- pretrain + fine-tune weight load --
    # FIX: Point pretrain to the dummy data we generated earlier in the smoke test
    cfg2.train_dir = os.path.join(WORK_DIR, "smoke_data", "train")
    pt_path = pretrain(cfg2)
    sd = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = BlueSentinelV11(cfg2, mean, var) \
        .load_state_dict(sd, strict=False)
    assert len(unexpected) == 0
    print(f"  [9/9] SimMIM pretrain + strict=False reload OK "
          f"({len(missing)} head keys left random, expected)")
    shutil.rmtree(root, ignore_errors=True)
    print("=== SMOKE TEST PASSED ===")
    return True


# ==============================================================================
# 24. CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=f"Blue Sentinel v{VERSION}")
    ap.add_argument("mode", choices=["smoke", "pretrain", "train", "eval",
                                     "export", "export-edge", "smooth"],
                    nargs="?", default="smoke")
    ap.add_argument("scene", nargs="?",
                    help="scene folder/cube NPZ (export) / prob dir (smooth)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (export)")
    ap.add_argument("out", nargs="?", help="output dir")
    ap.add_argument("--config")
    ap.add_argument("--ckpt", default=os.path.join(WORK_DIR, "best_v11.pt"))
    ap.add_argument("--pretrain", dest="pretrain_w", default="")
    ap.add_argument("--train-dir"); ap.add_argument("--val-dir")
    ap.add_argument("--test-dir")
    ap.add_argument("--batch-size", type=int); ap.add_argument("--epochs", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--resume", action="store_true",
                    help="resume training from last_v11.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-mirror", action="store_true",
                    help="run on a single GPU even when several are "
                         "visible (isolates DataParallel issues)")
    ap.add_argument("--sar-only", action="store_true")
    ap.add_argument("--optical-only", action="store_true")
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    cfg = Config.load(args.config) if args.config else Config()
    for a, k in (("train_dir", "train_dir"), ("val_dir", "val_dir"),
                 ("test_dir", "test_dir"), ("batch_size", "batch_size"),
                 ("epochs", "epochs"), ("workers", "num_workers")):
        v = getattr(args, a)
        if v is not None:
            setattr(cfg, k, v)
    if args.resume:
        cfg.resume = True
    if args.pretrain_w:
        cfg.pretrain_weights = args.pretrain_w
    if args.no_tta:
        cfg.infer_tta = False
    if args.no_mirror:
        cfg.mirror_gpus = False
        print("[gpu] --no-mirror: running on cuda:0 only")

    # Mirror (DataParallel): with N>1 GPUs, scale the default batch so each
    # GPU still gets a full per-GPU batch (v11.3). --batch-size overrides.
    # FIX (v11.4.2): the blind doubling (2/GPU) OOM'd on 2x T4 16 GB in the
    # selective scan; pick the per-GPU batch by the card's VRAM instead.
    if DEVICE.type == "cuda" and cfg.mirror_gpus and N_GPUS > 1:
        if args.batch_size is None:
            per_gpu = 2 if torch.cuda.get_device_properties(0) \
                .total_memory >= (32 << 30) else 1
            cfg.batch_size = per_gpu * N_GPUS
        print(f"[gpu] mirror: {N_GPUS}x {torch.cuda.get_device_name(0)} -> "
              f"batch_size={cfg.batch_size} "
              f"({cfg.batch_size // N_GPUS}/GPU); raise --batch-size "
              f"if VRAM allows")

    if args.mode == "pretrain":
        pretrain(cfg)
    elif args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        if not os.path.exists(args.ckpt):
            ap.error(f"checkpoint not found: {args.ckpt} (train first)")
        model, mcfg = load_checkpoint(args.ckpt)
        scenes = discover_scenes(cfg.test_dir)
        assert scenes, "eval: set --test-dir"
        evaluate(model, mcfg, scenes, split="test")
    elif args.mode == "export":
        if not (args.scene and args.ref and args.out):
            ap.error("export requires <scene> <ref.tif> <out_dir>")
        if not os.path.exists(args.ckpt):
            ap.error(f"checkpoint not found: {args.ckpt} (train first)")
        model, mcfg = load_checkpoint(args.ckpt)
        export_products(model, mcfg, args.scene, args.ref, args.out,
                        sar_only=args.sar_only,
                        optical_only=args.optical_only,
                        gradcam=args.gradcam, review=not args.no_review)
    elif args.mode == "export-edge":
        export_edge(cfg, ckpt=args.ckpt, out_dir=args.out or WORK_DIR)
    elif args.mode == "smooth":
        if not args.scene:
            ap.error("smooth requires <prob_dir> with per-date 'seg_probs' NPZs")
        smooth_mode(args.scene, cfg)


if __name__ == "__main__":
    main()