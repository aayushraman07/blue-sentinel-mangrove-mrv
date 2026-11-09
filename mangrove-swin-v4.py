"""
================================================================================
 MANGROVE ECO-MONITOR v4.1
 Dual-Branch Swin-Transformer U-Net + Cross-Attention Fusion
 + Siamese (T1/T2) Change Detection + 6 Multi-Task Heads
 + Temporal Backcasting + Production Inference Pipeline

 HARDWARE: Kaggle T4 x2 (30GB) | RTX A6000 (48GB) | A100 (80GB) (auto-detected, mixed_float16)

 ARCHITECTURE
 ------------
   T1: S1(4ch) ──► Swin Encoder SAR ──┐
       S2(10)+SCL+AUX ─► Swin Encoder OPT ─► CROSS-ATTENTION FUSION (SAR<->OPT)
   T2: (same encoders, SHARED weights) ───► |F_t1 - F_t2| ─► Change Decoder

   Main hybrid decoder (CNN + Swin blocks) ─► 4 heads @ full res:
     1. seg_head     (B,H,W,5)  softmax   : 5-class land cover
     2. carbon_head  (B,H,W,2)  linear    : ch0=mean biomass (t/ha / 100),
                                            ch1=log_var -> Gaussian NLL = mean + uncertainty
     3. eco_head     (B,H,W,2)  linear    : [LST / 50, soil moisture 0-1]
     5. risk_head    (B,H,W,1)  sigmoid   : storm-surge vulnerability 0-1 (x100 at export)
   Change decoder ─►
     4. change_head  (B,H,W,1)  sigmoid   : deforestation alert T1->T2

 TEMPORAL FEATURES (10-year GEE metrics as auxiliary channels)
   - ndvi_trend: linear regression slope of annual mean NDVI
   - flood_freq: proportion of S1 acquisitions showing flooding
   - lst_trend:  linear regression slope of LST
   - s1_backscatter_trend: VV/VH stability metric
   - coastal_erosion_rate: shoreline change proxy

 NPZ SCHEMA (per scene, v4.1)
 ----------------------
   REQUIRED
     s1      : (H,W,2) float  [VV,VH] dB (linear auto-converted to dB)
     s2      : (H,W,10)       [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] uint16 / 0-10000 / 0-1 auto-scaled
   OPTIONAL (auto-filled + loss-masked if absent)
     s1_std  : (H,W,2) multi-temporal std           | label  : (H,W) 0..4, 255=ignore
     scl     : (H,W) SCL class                      | carbon : (H,W) t/ha, -999=invalid
     aux     : (H,W,A) [elev_m, slope_deg,          | eco    : (H,W,2) [temp_C, moist 0-1], -999=invalid
                        ndvi_trend/yr, flood_freq,   | change : (H,W) 0/1, 255=ignore
                        lst_trend,                   | risk   : (H,W) 0..1, -999=invalid
                        s1_trend, erosion]  (10-yr)
     s1_t2, s2_t2, scl_t2, aux_t2 : second timestep (if absent -> T2=T1 copy, change ignored)

 MODES (bottom of file): "smoke" | "train" | "infer" | "export"
 NOTE on window size: default 8 because feature maps are 64/32/16/8 (exact division,
 no padding). If you want the "classic" window 7, set patch_size=224 (56/28/14/7).

 TRAINING NOTES FOR 10-YEAR DATA
 -------------------------------
 - Use temporal metric stacking: each scene includes 10-year trend bands in aux
 - For true multi-temporal training, prepare NPZ pairs (T1, T2) spanning years
 - The model learns temporal invariance via shared Siamese weights
 - Change detection head learns from actual year-pairs
 - For backcasting: train on (T_recent, T_old) pairs where T_old is target
================================================================================
"""
import os, gc, glob, json, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# 1. ENVIRONMENT & GPU
# ─────────────────────────────────────────────
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR  = "/kaggle/working/" if ON_KAGGLE else "./output/"
os.makedirs(WORK_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(42)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
print(f"GPUs: {len(gpus)} | {[tf.config.experimental.get_device_details(g).get('device_name', 'unknown') for g in gpus]}")

strategy = tf.distribute.MirroredStrategy()
N_REPLICAS = strategy.num_replicas_in_sync
print(f"Replicas: {N_REPLICAS}")

# ─────────────────────────────────────────────
# 2. CONFIG + HARDWARE PRESETS
# ─────────────────────────────────────────────
CONFIG = {
    # data
    "train_dir":        "/kaggle/input/mangrove-v4-npz/train/",
    "val_dir":          "/kaggle/input/mangrove-v4-npz/val/",
    "test_dir":         "/kaggle/input/mangrove-v4-npz/test/",
    "patch_size":       256,          # 224 if you want window_size=7
    "train_stride":     128,
    "max_train_patches": 8000,        # increased for 10-year temporal diversity
    "max_val_patches":  1200,
    "min_valid_frac":   0.35,         # slightly lower for rare classes
    "max_cloud_frac":   0.85,
    "ignore_label":     255,
    "num_classes":      5,
    "invalid_scl":      [0, 1, 2, 3, 7, 8, 9, 10, 11],
    "cloud_dilate":     5,
    "aux_channels":     6,            # [elev, slope, ndvi_trend, flood_freq, lst_trend, s1_trend]
    "aux_scale":        [50.0, 20.0, 0.1, 1.0, 50.0, 1.0],
    # model
    "embed_dims":       [48, 96, 192, 384],
    "depths":           [2, 2, 2, 2],
    "window_size":      8,
    "drop_path":        0.15,         # increased for regularization on large temporal data
    "fusion_heads":     8,
    "dec_widths":       [192, 96, 48],
    "refine_widths":    [32, 24],
    "change_widths":    [128, 96, 64, 48, 32, 24],
    "hybrid_stem":      True,
    "decoder_swin":     True,
    "decoder_swin_levels": 2,
    "use_se":           True,         # squeeze-excitation in decoder
    "use_deep_supervision": False,     # auxiliary losses at decoder levels
    # training
    "hardware":         "auto",
    "batch_per_replica": 4,
    "grad_accum":       1,
    "epochs":           120,
    "lr":               2e-4,
    "lr_warmup_epochs": 3,
    "weight_decay":     1e-4,
    "focal_gamma":      2.0,
    "label_smoothing":  0.05,         # for better generalization on noisy 10-yr labels
    "loss_weights":     {"seg_head": 1.0, "carbon_head": 0.6, "eco_head": 0.25,
                         "change_head": 1.2, "risk_head": 0.25},
    # target normalisation
    "carbon_scale":     100.0,
    "eco_scale":        [50.0, 1.0],
    # augmentation
    "augment":          True,
    "aug_brightness":   0.15,
    "aug_contrast":     0.15,
    "aug_noise":        0.02,
    # temporal
    "temporal_pairs":   True,          # expect T1/T2 pairs for change detection
    "min_temporal_gap_years": 2,      # minimum years between T1 and T2
    # inference
    "infer_batch":      8,
    "infer_tta":        True,
    "infer_overlap":    0.5,           # 50% overlap for smooth blending
    "shuffle_buffer":   256,
}

HW_PRESETS = {
    "t4x2":  dict(batch_per_replica=4,  grad_accum=8,  infer_batch=8),
    "a6000": dict(batch_per_replica=16, grad_accum=2, infer_batch=16),
    "a100":  dict(batch_per_replica=24, grad_accum=2, infer_batch=32),
    "cpu":   dict(batch_per_replica=2,  grad_accum=1, infer_batch=2),
}

def resolve_hardware(cfg):
    name = ""
    try:
        if gpus:
            name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "")
    except Exception:
        pass
    hw = cfg["hardware"]
    if hw == "auto":
        if any(k in name for k in ("A100", "H100")): hw = "a100"
        elif any(k in name for k in ("A6000", "A100", "4090", "L40", "L4")): hw = "a6000"
        elif "T4" in name: hw = "t4x2"
        else: hw = "cpu" if not gpus else "t4x2"
    preset = HW_PRESETS.get(hw, HW_PRESETS["t4x2"])
    cfg.update(preset)
    print(f"Hardware: {hw} ({name or 'unknown'}) | batch/replica={cfg['batch_per_replica']} "
          f"| grad_accum={cfg['grad_accum']}")
    return hw

HW = resolve_hardware(CONFIG)
GLOBAL_BATCH = CONFIG["batch_per_replica"] * N_REPLICAS
EFFECTIVE_BATCH = GLOBAL_BATCH * CONFIG["grad_accum"]
print(f"Global batch: {GLOBAL_BATCH} | effective: {EFFECTIVE_BATCH}")

MODEL_PATH = os.path.join(WORK_DIR, "mangrove_swin_v4.keras")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v4.npz")

CLASS_NAMES  = ["Bare Earth/Agri", "Tidal Water", "Perm Flood", "Deg. Mangrove", "Dense Mangrove"]
CLASS_COLORS = ["#8B4513", "#00BFFF", "#191970", "#90EE90", "#006400"]

# ─────────────────────────────────────────────
# 3. NPZ LOADING (v4.1 schema, graceful missing-field handling)
# ─────────────────────────────────────────────
def _process_s1(s1):
    s1 = s1.astype(np.float32)
    if s1.min() >= 0:
        s1 = 10.0 * np.log10(np.clip(s1, 1e-6, None))
    return np.clip(s1, -35.0, 5.0)

def _process_s2(s2):
    s2 = s2.astype(np.float32)
    if s2.max() > 20000:   s2 *= 1.0 / 65535.0
    elif s2.max() > 100:   s2 *= 1e-4
    return s2

def load_npz(path, cfg=CONFIG):
    """Returns 13 arrays. Missing -> sentinel-filled."""
    A = cfg["aux_channels"]
    with np.load(path) as d:
        files = d.files
        s1  = _process_s1(d["s1"])
        s2  = _process_s2(d["s2"])
        H, W = s2.shape[:2]
        s1_std = d["s1_std"].astype(np.float32) if "s1_std" in files else np.zeros_like(s1)
        s1  = np.concatenate([s1, s1_std], -1)
        scl = d["scl"].astype(np.float32)[..., None] if "scl" in files \
              else np.full((H, W, 1), 4.0, np.float32)
        aux = d["aux"].astype(np.float32) if "aux" in files else np.zeros((H, W, A), np.float32)
        aux = aux / np.asarray(cfg["aux_scale"], np.float32)

        label  = d["label"].astype(np.int32)   if "label"  in files else np.full((H, W), 255, np.int32)
        carbon = d["carbon"].astype(np.float32) if "carbon" in files else np.full((H, W), -999.0, np.float32)
        eco    = d["eco"].astype(np.float32)   if "eco"    in files else np.full((H, W, 2), -999.0, np.float32)
        change = d["change"].astype(np.int32)  if "change" in files else np.full((H, W), 255, np.int32)
        risk   = d["risk"].astype(np.float32)  if "risk"   in files else np.full((H, W), -999.0, np.float32)

        # Temporal pair handling
        has_t2 = "s1_t2" in files and "s2_t2" in files
        if has_t2:
            s1t2 = _process_s1(d["s1_t2"])
            s1t2_std = d["s1_t2_std"].astype(np.float32) if "s1_t2_std" in files else np.zeros_like(s1t2)
            s1t2 = np.concatenate([s1t2, s1t2_std], -1)
            s2t2 = _process_s2(d["s2_t2"])
            sclt2 = d["scl_t2"].astype(np.float32)[..., None] if "scl_t2" in files \
                    else np.full((H, W, 1), 4.0, np.float32)
            auxt2 = (d["aux_t2"].astype(np.float32) if "aux_t2" in files
                     else np.zeros((H, W, A), np.float32)) / np.asarray(cfg["aux_scale"], np.float32)
        else:
            s1t2, s2t2, sclt2, auxt2 = s1.copy(), s2.copy(), scl.copy(), aux.copy()
            if "change" not in files:
                change = np.full((H, W), 255, np.int32)
    return (s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
            label, carbon, eco, change, risk)

# ─────────────────────────────────────────────
# 4. PATCH GENERATOR (with temporal filtering)
# ─────────────────────────────────────────────
def patch_generator(file_list, cfg, training, max_patches):
    ps     = cfg["patch_size"]
    stride = cfg["train_stride"] if training else ps
    ignore = cfg["ignore_label"]
    invalid = np.array(cfg["invalid_scl"])
    emitted = 0
    for path in file_list:
        data = load_npz(path, cfg)
        label = data[8]
        h, w = label.shape
        if h < ps or w < ps:
            continue
        for i in range(0, h - ps + 1, stride):
            for j in range(0, w - ps + 1, stride):
                if emitted >= max_patches:
                    return
                sl = (slice(i, i + ps), slice(j, j + ps))
                lab = label[sl]
                if float(np.mean(lab != ignore)) < cfg["min_valid_frac"]:
                    continue
                if training:
                    cloud_frac = float(np.mean(np.isin(data[2][sl], invalid)))
                    if cloud_frac > cfg["max_cloud_frac"]:
                        continue
                yield tuple(arr[sl] if arr.ndim == 2 else arr[sl[0], sl[1], :]
                            for arr in data)
                emitted += 1
    gc.collect()

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING LAYER
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FeatureEngineer(layers.Layer):
    def __init__(self, invalid_scl=(0,1,2,3,7,8,9,10,11), dilate=5, aux_channels=6, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.invalid_scl = [int(c) for c in invalid_scl]
        self.dilate = int(dilate)
        self.aux_channels = int(aux_channels)
        self.eps = float(eps)

    def _ndx(self, a, b):
        return (a - b) / (a + b + self.eps)

    def call(self, inputs):
        s1, s2, scl, aux = inputs
        s1 = tf.cast(s1, tf.float32); s2 = tf.cast(s2, tf.float32)
        aux = tf.cast(aux, tf.float32)

        vv, vh, vv_std, vh_std = tf.split(s1, 4, axis=-1)
        lin_vv = tf.pow(10.0, vv / 10.0)
        lin_vh = tf.pow(10.0, vh / 10.0)
        rvi = 4.0 * lin_vh / (lin_vv + lin_vh + self.eps)
        # Additional SAR features
        vh_vv_ratio = vh - vv  # dB difference
        s1_feat = tf.concat([vv, vh, vh_vv_ratio, rvi, vv_std, vh_std], axis=-1)

        b2,b3,b4,b5,b6,b7,b8,b8a,b11,b12 = tf.split(s2, 10, axis=-1)
        ndvi  = self._ndx(b8, b4)
        ndwi  = self._ndx(b3, b8)
        mndwi = self._ndx(b3, b11)
        ndmi  = self._ndx(b8, b11)
        ndre  = self._ndx(b8, b5)
        bsi   = self._ndx(b11 + b4, b8 + b2)
        evi   = 2.5 * self._ndx(b8, b4 + 6.0 * b2 - 7.5 * b3 + self.eps)  # Enhanced Vegetation Index
        s2_feat = tf.concat([s2, ndvi, ndwi, mndwi, ndmi, ndre, bsi, evi], axis=-1)

        scl_i = tf.cast(scl, tf.int32)
        mask = tf.zeros(tf.shape(scl_i), dtype=tf.bool)
        for c in self.invalid_scl:
            mask = tf.logical_or(mask, tf.equal(scl_i, c))
        mask_f = tf.cast(mask, tf.float32)
        if self.dilate > 1:
            mask_f = tf.nn.max_pool2d(mask_f, ksize=self.dilate, strides=1, padding="SAME")
            mask_f = tf.cast(mask_f > 0.0, tf.float32)
        s2_feat = s2_feat * (1.0 - mask_f)
        s2_feat = tf.concat([s2_feat, mask_f, aux], axis=-1)
        return s1_feat, s2_feat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"invalid_scl": self.invalid_scl, "dilate": self.dilate,
                    "aux_channels": self.aux_channels, "eps": self.eps})
        return cfg

# ─────────────────────────────────────────────
# 6. NORMALIZATION STATS + CLASS ALPHA
# ─────────────────────────────────────────────
def compute_stats(file_list, cfg, max_patches=400):
    print("Computing normalization stats + class frequencies...")
    fe = FeatureEngineer(cfg["invalid_scl"], cfg["cloud_dilate"], cfg["aux_channels"])
    C = cfg["num_classes"]
    c1, c2 = 6, 18 + cfg["aux_channels"]  # updated for EVI
    n = 0
    s1_sum = np.zeros(c1); s1_sq = np.zeros(c1)
    s2_sum = np.zeros(c2); s2_sq = np.zeros(c2)
    counts = np.zeros(C, np.int64)
    for data in patch_generator(file_list, cfg, training=False, max_patches=max_patches):
        s1, s2, scl, aux = data[0], data[1], data[2], data[3]
        lab = data[8]
        f1, f2 = fe([s1[None], s2[None], scl[None], aux[None]])
        a = f1[0].numpy().reshape(-1, c1); b = f2[0].numpy().reshape(-1, c2)
        s1_sum += a.sum(0); s1_sq += (a ** 2).sum(0)
        s2_sum += b.sum(0); s2_sq += (b ** 2).sum(0)
        n += a.shape[0]
        v = lab[lab != cfg["ignore_label"]]
        counts += np.bincount(v, minlength=C)
    s1_mean = s1_sum / n; s1_var = np.clip(s1_sq / n - s1_mean ** 2, 1e-6, None)
    s2_mean = s2_sum / n; s2_var = np.clip(s2_sq / n - s2_mean ** 2, 1e-6, None)
    s2_mean[18] = 0.0; s2_var[18] = 1.0  # cloud-mask channel: identity

    freq = counts / max(counts.sum(), 1)
    alpha = 1.0 / (freq + 1e-6)
    alpha = np.clip(alpha / alpha.mean(), 0.25, 4.0)
    np.savez(STATS_PATH, s1_mean=s1_mean, s1_var=s1_var,
             s2_mean=s2_mean, s2_var=s2_var, alpha=alpha, counts=counts)
    print(f"  class counts: {counts.tolist()}  alpha: {np.round(alpha, 3).tolist()}")
    return s1_mean, s1_var, s2_mean, s2_var, alpha

def load_or_compute_stats(file_list, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        print("Loaded cached norm stats.")
        return z["s1_mean"], z["s1_var"], z["s2_mean"], z["s2_var"], z["alpha"]
    return compute_stats(file_list, cfg)

# ─────────────────────────────────────────────
# 7. tf.data PIPELINE (pack -> augment -> unpack)
# ─────────────────────────────────────────────
def output_signature(cfg):
    ps, A = cfg["patch_size"], cfg["aux_channels"]
    F = tf.float32
    return (
        tf.TensorSpec((ps, ps, 4),  F),
        tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F),
        tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps, 4),  F),
        tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F),
        tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps),     tf.int32),
        tf.TensorSpec((ps, ps),     F),
        tf.TensorSpec((ps, ps, 2),  F),
        tf.TensorSpec((ps, ps),     tf.int32),
        tf.TensorSpec((ps, ps),     F),
    )

def make_dataset(gen_fn, cfg, training):
    C, A = cfg["num_classes"], cfg["aux_channels"]
    SIZES = [4, 10, 1, A, 4, 10, 1, A, C, 1, 2, 1, 1]
    carbon_scale = float(cfg["carbon_scale"])
    eco_scale = tf.constant(cfg["eco_scale"], tf.float32)

    def pack(s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2, label, carbon, eco, change, risk):
        seg = tf.one_hot(tf.where(label == cfg["ignore_label"], -1, label), C, dtype=tf.float32)
        change_f = tf.cast(change, tf.float32)
        change_f = tf.where(change_f == 255.0, -1.0, change_f)
        return tf.concat([s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
                          seg, carbon[..., None], eco,
                          change_f[..., None], risk[..., None]], axis=-1)

    def random_aug(x):
        # Flips
        do_lr = tf.random.uniform(()) > 0.5
        x = tf.cond(do_lr, lambda: tf.image.flip_left_right(x), lambda: x)
        do_ud = tf.random.uniform(()) > 0.5
        x = tf.cond(do_ud, lambda: tf.image.flip_up_down(x), lambda: x)
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        x = tf.image.rot90(x, k)
        # Brightness/contrast (only on optical bands: indices 4:14 and 20:30)
        if training and cfg.get("augment", True):
            delta = tf.random.uniform((), -cfg["aug_brightness"], cfg["aug_brightness"])
            contrast = tf.random.uniform((), 1 - cfg["aug_contrast"], 1 + cfg["aug_contrast"])
            # Apply to optical T1 and T2
            x = tf.concat([
                x[..., :4],  # S1 T1 unchanged
                tf.image.adjust_contrast(tf.image.adjust_brightness(x[..., 4:14], delta), contrast),
                x[..., 14:18],  # SCL, AUX
                x[..., 18:22],  # S1 T2 unchanged
                tf.image.adjust_contrast(tf.image.adjust_brightness(x[..., 22:32], delta), contrast),
                x[..., 32:]
            ], axis=-1)
            # Gaussian noise on all
            noise = tf.random.normal(tf.shape(x), stddev=cfg["aug_noise"])
            x = x + noise
        return x

    def unpack_to_xy(x):
        (s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
         seg, carbon, eco, change, risk) = tf.split(x, SIZES, axis=-1)
        carbon_y = tf.where(carbon > -900.0, carbon / carbon_scale, -999.0)
        eco_y    = tf.where(eco > -900.0, eco / eco_scale, -999.0)
        risk_y   = tf.where(risk > -900.0, risk, -999.0)
        inputs = {"s1_input": s1, "s2_input": s2, "scl_input": scl, "aux_input": aux,
                  "s1_t2_input": s1t2, "s2_t2_input": s2t2,
                  "scl_t2_input": sclt2, "aux_t2_input": auxt2}
        targets = {"seg_head": seg, "carbon_head": carbon_y, "eco_head": eco_y,
                   "change_head": change, "risk_head": risk_y}
        return inputs, targets

    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=output_signature(cfg))
    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(cfg["shuffle_buffer"], seed=42, reshuffle_each_iteration=True)
        ds = ds.map(random_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(unpack_to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options(); opts.experimental_deterministic = False
    ds = ds.with_options(opts)
    return ds.batch(GLOBAL_BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# ─────────────────────────────────────────────
# 8. SWIN TRANSFORMER COMPONENTS
# ─────────────────────────────────────────────
def window_partition(x, ws):
    B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
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
        cfg = super().get_config(); cfg["rate"] = self.rate; return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim); self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_bias = bool(qkv_bias)
        self.attn_drop_rate = float(attn_drop); self.proj_drop_rate = float(proj_drop)

    def build(self, input_shape):
        ws = self.window_size
        self.qkv = layers.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.dim, name="proj")
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        self.rpb = self.add_weight(
            name="rel_pos_bias", shape=((2 * ws - 1) ** 2, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), trainable=True)
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij"))
        cf = coords.reshape(2, -1)
        rel = (cf[:, :, None] - cf[:, None, :]).transpose(1, 2, 0)
        rel[..., 0] += ws - 1
        rel[..., 1] += ws - 1
        rel[..., 0] *= 2 * ws - 1
        self.rel_pos_idx = tf.constant(rel.sum(-1), dtype=tf.int32)
        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        Bn = tf.shape(x)[0]
        N = self.window_size * self.window_size
        qkv = tf.reshape(self.qkv(x), [Bn, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q * self.scale, k, transpose_b=True)
        bias = tf.gather(self.rpb, tf.reshape(self.rel_pos_idx, [-1]))
        bias = tf.transpose(tf.reshape(bias, [N, N, self.num_heads]), [2, 0, 1])
        attn = attn + tf.cast(bias[None], attn.dtype)
        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) + tf.cast(mask[None], attn.dtype)
            attn = tf.reshape(attn, [Bn, self.num_heads, N, N])
        attn = tf.nn.softmax(tf.cast(attn, tf.float32), axis=-1)
        attn = self.attn_drop(tf.cast(attn, v.dtype), training=training)
        x = tf.matmul(attn, v)
        x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [Bn, N, self.dim])
        return self.proj_drop(self.proj(x), training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "window_size": self.window_size,
                    "num_heads": self.num_heads, "qkv_bias": self.qkv_bias,
                    "attn_drop": self.attn_drop_rate, "proj_drop": self.proj_drop_rate})
        return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class SwinBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim); self.num_heads = int(num_heads)
        self.window_size = int(window_size); self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio); self.drop = float(drop)
        self.attn_drop = float(attn_drop); self.drop_path_rate = float(drop_path)

    def build(self, input_shape):
        _, H, W, C = input_shape
        assert C == self.dim, f"SwinBlock dim {self.dim} != input C {C}"
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"Feature map {H}x{W} not divisible by window {self.window_size}"
        self.H, self.W = int(H), int(W)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(self.dim, self.window_size, self.num_heads,
                                    attn_drop=self.attn_drop, proj_drop=self.drop, name="wmsa")
        self.dp1 = DropPath(self.drop_path_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        hidden = int(self.dim * self.mlp_ratio)
        self.mlp = models.Sequential([layers.Dense(hidden, activation="gelu"),
                                      layers.Dropout(self.drop),
                                      layers.Dense(self.dim),
                                      layers.Dropout(self.drop)], name="mlp")
        self.dp2 = DropPath(self.drop_path_rate)
        self.attn_mask = self._build_mask() if self.shift_size > 0 else None
        super().build(input_shape)

    def _build_mask(self):
        ws, ss, H, W = self.window_size, self.shift_size, self.H, self.W
        img_mask = np.zeros((H, W), np.float32)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[hs, ws_] = cnt; cnt += 1
        m = img_mask.reshape(H // ws, ws, W // ws, ws).transpose(0, 2, 1, 3).reshape(-1, ws * ws)
        attn_mask = m[:, :, None] - m[:, None, :]
        attn_mask = np.where(attn_mask != 0.0, -100.0, 0.0).astype(np.float32)
        return tf.constant(attn_mask[:, None, :, :])

    def call(self, x, training=None):
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        w = window_partition(x, self.window_size)
        w = self.attn(w, mask=self.attn_mask, training=training)
        x = window_reverse(w, self.window_size, self.H, self.W)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        x = shortcut + self.dp1(x, training=training)
        x = x + self.dp2(self.mlp(self.norm2(x), training=training), training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads,
                    "window_size": self.window_size, "shift_size": self.shift_size,
                    "mlp_ratio": self.mlp_ratio, "drop": self.drop,
                    "attn_drop": self.attn_drop, "drop_path": self.drop_path_rate})
        return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class PatchMerging(layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = int(out_dim)
    def build(self, input_shape):
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.reduction = layers.Dense(self.out_dim, use_bias=False, name="reduction")
        super().build(input_shape)
    def call(self, x):
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(tf.concat([x0, x1, x2, x3], axis=-1)))
    def get_config(self):
        cfg = super().get_config(); cfg["out_dim"] = self.out_dim; return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class AbsDiff(layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

@tf.keras.utils.register_keras_serializable(package="mangrove")
class CrossAttentionFusion(layers.Layer):
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim); self.num_heads = int(num_heads)
    def build(self, input_shape):
        H, W = input_shape[0][1], input_shape[0][2]
        self.H, self.W = int(H), int(W)
        self.pos = self.add_weight(name="pos_embed", shape=(1, self.H * self.W, self.dim),
                                   initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.mha_s2o = layers.MultiHeadAttention(self.num_heads, self.dim // self.num_heads,
                                                 name="sar_queries_optical")
        self.mha_o2s = layers.MultiHeadAttention(self.num_heads, self.dim // self.num_heads,
                                                 name="optical_queries_sar")
        self.ln_s = layers.LayerNormalization(epsilon=1e-5, name="ln_sar")
        self.ln_o = layers.LayerNormalization(epsilon=1e-5, name="ln_opt")
        self.proj = layers.Conv2D(self.dim, 1, name="fuse_proj")
        self.ln_out = layers.LayerNormalization(epsilon=1e-5, name="ln_out")
        super().build(input_shape)
    def call(self, inputs, training=None):
        f_sar, f_opt = inputs
        B = tf.shape(f_sar)[0]; N = self.H * self.W
        t_sar = tf.reshape(f_sar, [B, N, self.dim]) + tf.cast(self.pos, f_sar.dtype)
        t_opt = tf.reshape(f_opt, [B, N, self.dim]) + tf.cast(self.pos, f_opt.dtype)
        s = t_sar + self.mha_s2o(query=t_sar, value=t_opt, training=training)
        o = t_opt + self.mha_o2s(query=t_opt, value=t_sar, training=training)
        s = self.ln_s(tf.reshape(s, [B, self.H, self.W, self.dim]))
        o = self.ln_o(tf.reshape(o, [B, self.H, self.W, self.dim]))
        out = self.proj(tf.concat([s, o], axis=-1)) + f_opt
        return self.ln_out(out)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads})
        return cfg

def _heads_for(dim):
    return max(2, min(12, dim // 16))

def make_swin_encoder(prefix, dims, depths, cfg):
    ws = cfg["window_size"]; dp = cfg["drop_path"]
    if cfg["hybrid_stem"]:
        stem = models.Sequential([
            layers.Conv2D(dims[0] // 2, 3, strides=2, padding="same"),
            layers.BatchNormalization(), layers.Activation("gelu"),
            layers.Conv2D(dims[0], 3, strides=2, padding="same"),
            layers.BatchNormalization(), layers.Activation("gelu"),
            layers.Conv2D(dims[0], 3, padding="same"),
        ], name=f"{prefix}_stem")
    else:
        stem = models.Sequential([layers.Conv2D(dims[0], 4, strides=4, padding="same")],
                                 name=f"{prefix}_stem")
    stem_norm = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_stem_norm")

    stages, mergers = [], []
    for i, (d, dep) in enumerate(zip(dims, depths)):
        blocks = []
        for j in range(dep):
            shift = 0 if j % 2 == 0 else ws // 2
            blocks.append(SwinBlock(d, _heads_for(d), ws, shift, drop_path=dp,
                                    name=f"{prefix}_stage{i}_blk{j}"))
        stages.append(blocks)
        if i < len(dims) - 1:
            mergers.append(PatchMerging(dims[i + 1], name=f"{prefix}_merge{i}"))

    def apply(x):
        x = stem_norm(stem(x))
        skips = []
        for i, blocks in enumerate(stages):
            for blk in blocks:
                x = blk(x)
            skips.append(x)
            if i < len(mergers):
                x = mergers[i](x)
        return x, skips
    return apply

# ─────────────────────────────────────────────
# 9. CNN DECODER BLOCKS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 10. FULL MODEL
# ─────────────────────────────────────────────
def build_model(cfg, s1_stats, s2_stats):
    C  = cfg["num_classes"]; ps = cfg["patch_size"]; A = cfg["aux_channels"]
    dims     = cfg["embed_dims"]
    s1_dims  = [max(16, d // 2) for d in dims]
    depths   = cfg["depths"]; ws = cfg["window_size"]

    # 8 inputs
    s1_in   = layers.Input((ps, ps, 4),  name="s1_input")
    s2_in   = layers.Input((ps, ps, 10), name="s2_input")
    scl_in  = layers.Input((ps, ps, 1),  name="scl_input")
    aux_in  = layers.Input((ps, ps, A),  name="aux_input")
    s1t2_in  = layers.Input((ps, ps, 4),  name="s1_t2_input")
    s2t2_in  = layers.Input((ps, ps, 10), name="s2_t2_input")
    sclt2_in = layers.Input((ps, ps, 1),  name="scl_t2_input")
    auxt2_in = layers.Input((ps, ps, A),  name="aux_t2_input")

    # Feature engineering + normalization
    fe_t1 = FeatureEngineer(cfg["invalid_scl"], cfg["cloud_dilate"], A, name="feature_engineer")
    fe_t2 = FeatureEngineer(cfg["invalid_scl"], cfg["cloud_dilate"], A, name="feature_engineer_t2")
    f1, f2 = fe_t1([s1_in, s2_in, scl_in, aux_in])
    g1, g2 = fe_t2([s1t2_in, s2t2_in, sclt2_in, auxt2_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]), variance=list(s1_stats[1]), name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]), variance=list(s2_stats[1]), name="s2_norm")
    f1 = n1(f1); f2 = n2(f2); g1 = n1(g1); g2 = n2(g2)

    # Siamese encoders
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg)
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg)
    b1, sk1 = enc_s1(f1)
    b2, sk2 = enc_s2(f2)
    c1, _   = enc_s1(g1)
    c2, _   = enc_s2(g2)

    # Cross-attention fusion
    b1p = layers.Conv2D(dims[-1], 1, name="sar_bottleneck_proj")(b1)
    b1p = layers.LayerNormalization(epsilon=1e-5, name="sar_bottleneck_ln")(b1p)
    fused = CrossAttentionFusion(dims[-1], cfg["fusion_heads"],
                                 name="cross_attn_fusion")([b1p, b2])

    # Main hybrid decoder
    x = fused
    for i, f in enumerate(cfg["dec_widths"]):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"dec_up{i}")(x)
        x = layers.concatenate([x, sk1[2 - i], sk2[2 - i]], name=f"dec_cat{i}")
        x = conv_block(x, f, use_se=cfg["use_se"], name=f"dec_c{i}")
        if cfg["decoder_swin"] and i < cfg["decoder_swin_levels"]:
            shift = 0 if i % 2 == 0 else ws // 2
            x = SwinBlock(f, _heads_for(f), ws, shift, name=f"dec_swin{i}")(x)
    for j, f in enumerate(cfg["refine_widths"]):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"refine_up{j}")(x)
        x = conv_block(x, f, use_se=cfg["use_se"], name=f"refine_c{j}")

    # Change decoder
    d = layers.concatenate([AbsDiff(name="diff_sar")([b1, c1]),
                            AbsDiff(name="diff_opt")([b2, c2])], name="change_diff")
    d = conv_block(d, cfg["change_widths"][0], use_se=cfg["use_se"], name="change_bottleneck")
    for i, f in enumerate(cfg["change_widths"][1:]):
        d = layers.UpSampling2D(interpolation="bilinear", name=f"change_up{i}")(d)
        d = conv_block(d, f, use_se=cfg["use_se"], name=f"change_c{i}")

    # 5 task heads
    seg = layers.Conv2D(C, 1, padding="same", name="seg_logits")(x)
    seg = layers.Activation("softmax", dtype="float32", name="seg_head")(seg)

    carbon = layers.Conv2D(2, 1, padding="same", dtype="float32", name="carbon_head")(x)

    eco = layers.Conv2D(2, 1, padding="same", dtype="float32", name="eco_head")(x)

    change = layers.Conv2D(1, 1, padding="same", name="change_logits")(d)
    change = layers.Activation("sigmoid", dtype="float32", name="change_head")(change)

    risk = layers.Conv2D(1, 1, padding="same", name="risk_logits")(x)
    risk = layers.Activation("sigmoid", dtype="float32", name="risk_head")(risk)

    return models.Model(
        inputs=[s1_in, s2_in, scl_in, aux_in, s1t2_in, s2t2_in, sclt2_in, auxt2_in],
        outputs=[seg, carbon, eco, change, risk],
        name="MangroveSwinV4_EcoMonitor")

# ─────────────────────────────────────────────
# 11. LOSSES & METRICS
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=1.0,
                 label_smoothing=0.0, **kwargs):
        super().__init__(name="focal_dice", **kwargs)
        self.alpha = None if alpha is None else [float(a) for a in alpha]
        self.gamma = float(gamma); self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        valid = tf.cast(tf.reduce_sum(y_true, -1, keepdims=True) > 0, tf.float32)
        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / tf.cast(tf.shape(y_true)[-1], tf.float32)
        a = tf.ones_like(y_pred) if self.alpha is None else tf.constant(self.alpha, tf.float32)
        ce = -y_true * tf.math.log(y_pred)
        fl = tf.reduce_sum(a * tf.pow(1.0 - y_pred, self.gamma) * ce, axis=-1, keepdims=True) * valid
        focal = tf.reduce_sum(fl, axis=[1, 2, 3]) / (tf.reduce_sum(valid, axis=[1, 2, 3]) + eps)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
        dice = tf.reduce_mean(1.0 - (2.0 * inter + 1.0) / (denom + 1.0), axis=-1)
        return self.focal_weight * focal + self.dice_weight * dice
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma,
                    "focal_weight": self.focal_weight, "dice_weight": self.dice_weight,
                    "label_smoothing": self.label_smoothing})
        return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonNLLLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="carbon_nll", **kwargs)
    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        mean = tf.cast(y_pred[..., 0:1], tf.float32)
        log_var = tf.clip_by_value(tf.cast(y_pred[..., 1:2], tf.float32), -6.0, 8.0)
        mask = tf.cast(y > -900.0, tf.float32)
        y = tf.maximum(y, 0.0)
        nll = 0.5 * (log_var + tf.square(y - mean) * tf.exp(-log_var))
        return tf.reduce_sum(nll * mask) / (tf.reduce_sum(mask) + 1e-6)

@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedMSE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="masked_mse", **kwargs)
    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32); p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        se = tf.square(tf.where(mask > 0, y, 0.0) - p) * mask
        return tf.reduce_sum(se) / (tf.reduce_sum(mask) + 1e-6)

@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedBinaryFocal(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(name="masked_binary_focal", **kwargs)
        self.alpha = float(alpha); self.gamma = float(gamma)
    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32); p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
        ce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        pt = y * p + (1.0 - y) * (1.0 - p)
        at = y * self.alpha + (1.0 - y) * (1.0 - self.alpha)
        fl = at * tf.pow(1.0 - pt, self.gamma) * ce
        return tf.reduce_sum(fl * mask) / (tf.reduce_sum(mask) + 1e-6)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma})
        return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedMAE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="masked_mae", **kwargs)
    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32); p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        ae = tf.abs(tf.where(mask > 0, y, 0.0) - p) * mask
        return tf.reduce_sum(ae) / (tf.reduce_sum(mask) + 1e-6)

@tf.keras.utils.register_keras_serializable(package="mangrove")
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.total_cm = self.add_weight(name="cm", shape=(self.num_classes, self.num_classes),
                                        initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.reshape(tf.reduce_sum(y_true, -1) > 0, [-1])
        yt = tf.boolean_mask(tf.reshape(tf.argmax(y_true, -1, output_type=tf.int32), [-1]), valid)
        yp = tf.boolean_mask(tf.reshape(tf.argmax(y_pred, -1, output_type=tf.int32), [-1]), valid)
        cm = tf.math.confusion_matrix(yt, yp, num_classes=self.num_classes, dtype=tf.float32)
        return self.total_cm.assign_add(cm)
    def result(self):
        cm = self.total_cm
        diag = tf.linalg.diag_part(cm)
        union = tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - diag
        present = tf.cast(union > 0, tf.float32)
        iou = tf.where(union > 0, diag / (union + 1e-7), tf.zeros_like(union))
        return tf.reduce_sum(iou) / (tf.reduce_sum(present) + 1e-7)
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))
    def get_config(self):
        cfg = super().get_config(); cfg["num_classes"] = self.num_classes; return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonMAE(tf.keras.metrics.Metric):
    def __init__(self, name="carbon_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.cast(y_true, tf.float32)
        mean = tf.cast(y_pred[..., 0:1], tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        self.total.assign_add(tf.reduce_sum(tf.abs(tf.where(mask > 0, y, 0.0) - mean) * mask))
        self.count.assign_add(tf.reduce_sum(mask))
    def result(self):
        return self.total / (self.count + 1e-6)
    def reset_state(self):
        self.total.assign(0.0); self.count.assign(0.0)

@tf.keras.utils.register_keras_serializable(package="mangrove")
class ChangeF1(tf.keras.metrics.Metric):
    """F1 score for change detection (handles class imbalance)."""
    def __init__(self, name="change_f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.cast(y_true, tf.float32); p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        p_bin = tf.cast(p > self.threshold, tf.float32)
        tp = tf.reduce_sum(y * p_bin * mask)
        fp = tf.reduce_sum((1 - y) * p_bin * mask)
        fn = tf.reduce_sum(y * (1 - p_bin) * mask)
        self.tp.assign_add(tp); self.fp.assign_add(fp); self.fn.assign_add(fn)
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        return 2 * precision * recall / (precision + recall + 1e-6)
    def reset_state(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

# ─────────────────────────────────────────────
# 12. CALLBACKS / EVAL / VISUALIZATION
# ─────────────────────────────────────────────
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_steps, alpha=0.01):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * step / tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        decay_step = tf.maximum(step - tf.cast(self.warmup_steps, tf.float32), 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * decay_step / decay_steps))
        decayed = (1.0 - self.alpha) * cosine_decay + self.alpha
        decayed_lr = self.initial_lr * decayed
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

def _preds_to_dict(model, preds):
    if isinstance(preds, dict):
        return preds
    return dict(zip(model.output_names, preds))

def evaluate_per_class(model, val_ds, cfg):
    from sklearn.metrics import confusion_matrix
    C = cfg["num_classes"]; cm = np.zeros((C, C), np.int64)
    for inputs, targets in val_ds:
        p = _preds_to_dict(model, model.predict(inputs, verbose=0))["seg_head"].argmax(-1)
        y = targets["seg_head"].numpy()
        t = y.argmax(-1); v = y.sum(-1) > 0
        cm += confusion_matrix(t[v].ravel(), p[v].ravel(), labels=list(range(C)))
    diag = np.diag(cm); union = cm.sum(0) + cm.sum(1) - diag
    iou = np.where(union > 0, diag / np.maximum(union, 1), np.nan)
    print("\n── Per-class segmentation results ──")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:18s} IoU={iou[i]:.3f}  px={cm[i].sum():,}")
    print(f"  mIoU={np.nanmean(iou):.3f}")
    np.savetxt(os.path.join(WORK_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    # Save metrics JSON
    metrics = {n: {"iou": float(iou[i]), "pixels": int(cm[i].sum())} for i, n in enumerate(CLASS_NAMES)}
    metrics["mIoU"] = float(np.nanmean(iou))
    with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return cm, iou

def _stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)

def plot_training_history(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("loss", "Total Loss"),
        ("seg_head_mean_iou", "Seg mIoU"),
        ("carbon_head_carbon_mae", "Carbon MAE"),
        ("change_head_change_f1", "Change F1"),
        ("risk_head_masked_mae", "Risk MAE"),
        ("eco_head_masked_mse", "Eco MSE")
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        train_key = key
        val_key = f"val_{key}"
        if train_key in history.history:
            ax.plot(history.history[train_key], label="train")
        if val_key in history.history:
            ax.plot(history.history[val_key], label="val")
        ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "training_history.png"), dpi=150); plt.close()

def visualize_predictions(model, val_ds, cfg, n_samples=4):
    cmap = mcolors.ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(n_samples, 7, figsize=(28, 5 * n_samples))
    titles = ["S2 RGB", "SAR VH-VV", "GT Seg", "Pred Seg", "Carbon μ±σ", "Change Alert", "Risk Index"]
    for a, t in zip(axes[0], titles): a.set_title(t, fontsize=11)
    count = 0
    for inputs, targets in val_ds.take(2):
        preds = _preds_to_dict(model, model.predict(inputs, verbose=0))
        for i in range(min(len(targets["seg_head"]), n_samples - count)):
            s2 = inputs["s2_input"][i].numpy(); s1 = inputs["s1_input"][i].numpy()
            y = targets["seg_head"][i].numpy().argmax(-1)
            p = preds["seg_head"][i].argmax(-1)
            carbon_mu = preds["carbon_head"][i, :, :, 0] * cfg["carbon_scale"]
            carbon_sigma = np.sqrt(np.exp(np.clip(preds["carbon_head"][i, :, :, 1], -6, 8))) * cfg["carbon_scale"]
            change_p = preds["change_head"][i, :, :, 0]
            risk = preds["risk_head"][i, :, :, 0] * 100.0

            axes[count, 0].imshow(_stretch(s2[:, :, [2, 1, 0]]))
            sar = np.stack([_stretch(s1[:, :, 2])] * 3, -1)  # VH-VV difference
            axes[count, 1].imshow(sar, cmap="RdBu_r")
            axes[count, 2].imshow(y, cmap=cmap, vmin=0, vmax=4)
            axes[count, 3].imshow(p, cmap=cmap, vmin=0, vmax=4)
            im = axes[count, 4].imshow(carbon_mu, cmap="YlGn", vmin=0, vmax=300)
            # Overlay uncertainty as alpha
            axes[count, 4].imshow(np.ones_like(carbon_mu), cmap="gray", alpha=carbon_sigma/100)
            axes[count, 5].imshow(change_p, cmap="Reds", vmin=0, vmax=1)
            axes[count, 6].imshow(risk, cmap="YlOrRd", vmin=0, vmax=100)
            for a in axes[count]: a.axis("off")
            count += 1
            if count >= n_samples: break
        if count >= n_samples: break
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "predictions_preview.png"), dpi=150,
                bbox_inches="tight"); plt.close()

# ─────────────────────────────────────────────
# 13. TILED INFERENCE (production-ready)
# ─────────────────────────────────────────────
def _tile_starts(dim, ps, overlap):
    stride = int(ps * (1 - overlap))
    if dim <= ps: return [0]
    starts = list(range(0, dim - ps + 1, stride))
    if starts[-1] != dim - ps: starts.append(dim - ps)
    return starts

def predict_scene(model, s1, s2, scl=None, aux=None, t2=None, cfg=CONFIG,
                  batch_size=None, tta=None, overlap=None):
    """
    Production inference with smooth blending, TTA, and full product export.
    """
    batch_size = batch_size or cfg["infer_batch"]
    tta = cfg["infer_tta"] if tta is None else tta
    overlap = overlap if overlap is not None else cfg.get("infer_overlap", 0.5)
    ps, C, A = cfg["patch_size"], cfg["num_classes"], cfg["aux_channels"]
    h, w = s2.shape[:2]

    if s1.shape[-1] == 2:
        s1 = np.concatenate([s1, np.zeros_like(s1)], -1)
    scl = np.full((h, w, 1), 4.0, np.float32) if scl is None else scl[..., None].astype(np.float32)
    aux = np.zeros((h, w, A), np.float32) if aux is None else aux.astype(np.float32)
    if t2 is None:
        s1t2, s2t2, sclt2, auxt2 = s1, s2, scl, aux
    else:
        s1t2 = t2["s1"]
        if s1t2.shape[-1] == 2:
            s1t2 = np.concatenate([s1t2, np.zeros_like(s1t2)], -1)
        s2t2 = t2["s2"]
        sclt2 = t2["scl"][..., None].astype(np.float32) if t2.get("scl") is not None else scl
        auxt2 = t2["aux"].astype(np.float32) if t2.get("aux") is not None else aux

    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        pad = ((0, pad_h), (0, pad_w), (0, 0))
        s1, s2, aux = np.pad(s1, pad), np.pad(s2, pad), np.pad(aux, pad)
        s1t2, s2t2, auxt2 = np.pad(s1t2, pad), np.pad(s2t2, pad), np.pad(auxt2, pad)
        scl = np.pad(scl, pad, constant_values=4)
        sclt2 = np.pad(sclt2, pad, constant_values=4)
    H, W = s2.shape[:2]

    acc = {"seg_head":    np.zeros((H, W, C), np.float32),
           "carbon_head": np.zeros((H, W, 2), np.float32),
           "eco_head":    np.zeros((H, W, 2), np.float32),
           "change_head": np.zeros((H, W, 1), np.float32),
           "risk_head":   np.zeros((H, W, 1), np.float32)}
    weight = np.zeros((H, W), np.float32)

    # Gaussian weight kernel for smooth blending
    y_coords = np.arange(ps); x_coords = np.arange(ps)
    y_coords = np.minimum(y_coords, ps - 1 - y_coords)
    x_coords = np.minimum(x_coords, ps - 1 - x_coords)
    gauss_kernel = np.outer(np.exp(-0.5 * (y_coords / (ps/6))**2),
                            np.exp(-0.5 * (x_coords / (ps/6))**2))

    variants = [(lambda x: x, lambda p: p),
                (lambda x: x[:, :, ::-1], lambda p: p[:, :, ::-1]),
                (lambda x: x[:, ::-1],    lambda p: p[:, ::-1]),
                (lambda x: np.rot90(x, 1, (1, 2)), lambda p: np.rot90(p, -1, (1, 2)))] if tta \
               else [(lambda x: x, lambda p: p)]

    coords = [(i, j) for i in _tile_starts(H, ps, overlap) for j in _tile_starts(W, ps, overlap)]
    total_tiles = len(coords)
    print(f"Inference: {h}x{w} -> padded {H}x{W} | {total_tiles} tiles | batch={batch_size} | TTA={tta}")

    for k in range(0, len(coords), batch_size):
        chunk = coords[k:k + batch_size]
        def stack(arr): return np.stack([arr[i:i + ps, j:j + ps] for i, j in chunk])
        batch = {"s1_input": stack(s1), "s2_input": stack(s2), "scl_input": stack(scl),
                 "aux_input": stack(aux), "s1_t2_input": stack(s1t2), "s2_t2_input": stack(s2t2),
                 "scl_t2_input": stack(sclt2), "aux_t2_input": stack(auxt2)}
        for fwd, inv in variants:
            fb = {k2: fwd(v) for k2, v in batch.items()}
            preds = _preds_to_dict(model, model.predict(fb, verbose=0, batch_size=len(chunk)))
            for name in acc:
                pr = inv(preds[name])
                for n, (i, j) in enumerate(chunk):
                    acc[name][i:i + ps, j:j + ps] += pr[n] * gauss_kernel[..., None if pr[n].ndim == 3 else np.newaxis]
        for i, j in chunk:
            weight[i:i + ps, j:j + ps] += gauss_kernel
        if k % (batch_size * 10) == 0:
            print(f"  processed {min(k + batch_size, total_tiles)}/{total_tiles} tiles")

    weight = np.maximum(weight[..., None], 1e-6)
    seg_probs = acc["seg_head"] / weight
    carbon    = acc["carbon_head"] / weight
    eco       = acc["eco_head"] / weight
    change    = acc["change_head"] / weight
    risk      = acc["risk_head"] / weight

    carbon_mean = carbon[..., 0] * cfg["carbon_scale"]
    carbon_std  = np.sqrt(np.exp(np.clip(carbon[..., 1], -6, 8))) * cfg["carbon_scale"]
    out = {
        "classes":      seg_probs.argmax(-1).astype(np.uint8)[:h, :w],
        "seg_probs":    seg_probs[:h, :w],
        "flood_risk":   (seg_probs[..., 1] + seg_probs[..., 2]).astype(np.float32)[:h, :w],
        "carbon_mean":  carbon_mean.astype(np.float32)[:h, :w],
        "carbon_std":   carbon_std.astype(np.float32)[:h, :w],
        "carbon_ci_lower": np.maximum(carbon_mean - 1.96 * carbon_std, 0).astype(np.float32)[:h, :w],
        "carbon_ci_upper": (carbon_mean + 1.96 * carbon_std).astype(np.float32)[:h, :w],
        "eco_temp":     (eco[..., 0] * cfg["eco_scale"][0]).astype(np.float32)[:h, :w],
        "eco_moisture": eco[..., 1].astype(np.float32)[:h, :w],
        "change_prob":  change[..., 0].astype(np.float32)[:h, :w],
        "risk_index":   (risk[..., 0] * 100.0).astype(np.float32)[:h, :w],
    }
    # Derived products
    out["mangrove_extent"] = ((out["classes"] == 3) | (out["classes"] == 4)).astype(np.uint8)
    out["total_carbon_stock"] = out["carbon_mean"] * out["mangrove_extent"]  # proxy: t/ha * binary mask
    return out

def export_geotiff(reference_tif, array, out_path, dtype="float32", nodata=None, band_names=None):
    """Export array(s) to GeoTIFF with proper georeferencing."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        print("rasterio not installed; saving as NPZ instead")
        np.savez(out_path.replace(".tif", ".npz"), array=array)
        return
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
    if array.ndim == 2:
        array = array[..., None]
    bands = array.shape[-1] if array.ndim == 3 else 1
    profile.update(count=bands, dtype=dtype, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    if nodata is not None: profile.update(nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(bands):
            dst.write(array[..., b].astype(dtype), b + 1)
            if band_names:
                dst.update_tags(bands={b+1: {"NAME": band_names[b]}})
    print(f"Saved GeoTIFF: {out_path}")

def export_all_products(model, scene_npz_path, ref_tif_path, out_dir, cfg=CONFIG):
    """Full production pipeline: load NPZ, run inference, export all products."""
    os.makedirs(out_dir, exist_ok=True)
    data = load_npz(scene_npz_path, cfg)
    s1, s2, scl, aux = data[0], data[1], data[2], data[3]
    t2 = {"s1": data[4], "s2": data[5], "scl": data[6], "aux": data[7]} if cfg["temporal_pairs"] else None

    products = predict_scene(model, s1, s2, scl, aux, t2, cfg)

    # Export each product
    exports = [
        ("classes.tif", products["classes"], "uint8", 255, ["land_cover_class"]),
        ("carbon_mean.tif", products["carbon_mean"], "float32", -999, ["carbon_tons_per_ha"]),
        ("carbon_uncertainty.tif", products["carbon_std"], "float32", -999, ["carbon_std_tons_per_ha"]),
        ("change_alert.tif", products["change_prob"], "float32", -999, ["deforestation_probability"]),
        ("risk_index.tif", products["risk_index"], "float32", -999, ["storm_surge_risk_0_100"]),
        ("eco_temperature.tif", products["eco_temp"], "float32", -999, ["lst_celsius"]),
        ("eco_moisture.tif", products["eco_moisture"], "float32", -999, ["soil_moisture_0_1"]),
        ("mangrove_extent.tif", products["mangrove_extent"], "uint8", 255, ["mangrove_binary"]),
    ]
    for fname, arr, dtype, nodata, bands in exports:
        export_geotiff(ref_tif_path, arr, os.path.join(out_dir, fname),
                      dtype=dtype, nodata=nodata, band_names=bands)

    # Save metadata
    meta = {
        "scene": scene_npz_path,
        "reference": ref_tif_path,
        "products": [e[0] for e in exports],
        "carbon_stats": {
            "mean": float(np.mean(products["carbon_mean"][products["carbon_mean"] > 0])),
            "total_mangrove_pixels": int(np.sum(products["mangrove_extent"])),
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products

# ─────────────────────────────────────────────
# 14. GRADIENT ACCUMULATION TRAINER
# ─────────────────────────────────────────────
class GradientAccumulationModel(tf.keras.Model):
    """Custom Model that handles gradient accumulation inside train_step."""
    def __init__(self, accum_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_steps = max(1, int(accum_steps))
        self._accum_gradients = []

    def train_step(self, data):
        if self.accum_steps == 1:
            return super().train_step(data)

        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = loss / tf.cast(self.accum_steps, loss.dtype)

        gradients = tape.gradient(loss, self.trainable_variables)
        self._accum_gradients.append(gradients)

        if len(self._accum_gradients) >= self.accum_steps:
            # Average accumulated gradients
            averaged = []
            for grads in zip(*self._accum_gradients):
                averaged.append(tf.reduce_mean(tf.stack(grads), axis=0))
            self.optimizer.apply_gradients(zip(averaged, self.trainable_variables))
            self._accum_gradients = []
            # Update metrics
            self.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.metrics}
        else:
            # Return current loss without updating weights
            return {"loss": loss * tf.cast(self.accum_steps, loss.dtype)}

    def reset_accumulation(self):
        self._accum_gradients = []

# ─────────────────────────────────────────────
# 15. TRAINING
# ─────────────────────────────────────────────
def _make_optimizer(cfg, lr):
    kwargs = dict(learning_rate=lr, weight_decay=cfg["weight_decay"], clipnorm=1.0)
    try:
        return optimizers.AdamW(**kwargs,
                                gradient_accumulation_steps=cfg["grad_accum"])
    except TypeError:
        print("WARNING: Keras lacks gradient_accumulation_steps; using custom accumulation.")
        return optimizers.AdamW(**kwargs)

def train(cfg=CONFIG):
    print("\n=== MANGROVE ECO-MONITOR v4.1 — TRAINING ===\n")
    train_files = sorted(glob.glob(os.path.join(cfg["train_dir"], "*.npz")))
    val_files   = sorted(glob.glob(os.path.join(cfg["val_dir"], "*.npz")))
    assert train_files and val_files, "Set train_dir/val_dir in CONFIG"
    print(f"train scenes: {len(train_files)} | val scenes: {len(val_files)} "
          f"| global batch: {GLOBAL_BATCH} | effective: {EFFECTIVE_BATCH}")

    s1m, s1v, s2m, s2v, alpha = load_or_compute_stats(train_files, cfg)

    ds_train = make_dataset(lambda: patch_generator(train_files, cfg, True,
                                                    cfg["max_train_patches"]), cfg, True)
    ds_val   = make_dataset(lambda: patch_generator(val_files, cfg, False,
                                                    cfg["max_val_patches"]), cfg, False)

    with strategy.scope():
        model = build_model(cfg, (s1m, s1v), (s2m, s2v))
        est_steps = max(1, cfg["max_train_patches"] // GLOBAL_BATCH)
        total_steps = cfg["epochs"] * est_steps
        warmup_steps = cfg.get("lr_warmup_epochs", 3) * est_steps
        lr_sched = WarmUpCosineDecay(cfg["lr"], total_steps, warmup_steps, alpha=0.01)
        opt = _make_optimizer(cfg, lr_sched)

        # Wrap with gradient accumulation if needed
        if cfg["grad_accum"] > 1:
            model = GradientAccumulationModel(accum_steps=cfg["grad_accum"], inputs=model.inputs,
                                              outputs=model.outputs, name=model.name)
            model._is_mangrove_v4 = True

        model.compile(
            optimizer=opt,
            loss={"seg_head":    FocalDiceLoss(alpha=alpha, gamma=cfg["focal_gamma"],
                                               label_smoothing=cfg.get("label_smoothing", 0.0)),
                  "carbon_head": CarbonNLLLoss(),
                  "eco_head":    MaskedMSE(),
                  "change_head": MaskedBinaryFocal(),
                  "risk_head":   MaskedMAE()},
            loss_weights=cfg["loss_weights"],
            metrics={"seg_head": [MeanIoU(cfg["num_classes"])],
                     "carbon_head": CarbonMAE(),
                     "change_head": ChangeF1(),
                     "risk_head": MaskedMAE(),
                     "eco_head": MaskedMSE()})

    model.summary(print_fn=lambda s: None if "model" in s.lower() else print(s))

    monitor = "val_seg_head_mean_iou"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor, mode="max", patience=15,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor, mode="max",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.BackupAndRestore(os.path.join(WORK_DIR, "backup")),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(os.path.join(WORK_DIR, "training_log.csv")),
        tf.keras.callbacks.TensorBoard(os.path.join(WORK_DIR, "tb")),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.5,
                                             patience=5, min_lr=1e-6, verbose=1),
        MemoryCleanupCallback(),
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=cfg["epochs"],
                        callbacks=callbacks)

    model.save(MODEL_PATH)
    plot_training_history(history)
    evaluate_per_class(model, ds_val, cfg)
    visualize_predictions(model, ds_val, cfg)
    print(f"\nModel saved: {MODEL_PATH}")
    return model

# ─────────────────────────────────────────────
# 16. SMOKE TEST
# ─────────────────────────────────────────────
def _smoke_generator(cfg, n=24):
    ps, A = cfg["patch_size"], cfg["aux_channels"]
    rng = np.random.default_rng(42)
    for _ in range(n):
        s1  = np.concatenate([rng.uniform(-30, 0, (ps, ps, 2)),
                              rng.uniform(0, 3, (ps, ps, 2))], -1).astype(np.float32)
        s2  = rng.uniform(0, 0.5, (ps, ps, 10)).astype(np.float32)
        scl = rng.choice([4, 5, 6, 8, 9], (ps, ps, 1)).astype(np.float32)
        aux = rng.uniform(-2, 2, (ps, ps, A)).astype(np.float32)
        s1t2 = np.clip(s1 + rng.normal(0, 0.5, s1.shape), -35, 5).astype(np.float32)
        s2t2 = np.clip(s2 + rng.normal(0, 0.02, s2.shape), 0, 1).astype(np.float32)
        label = rng.choice([0, 1, 2, 3, 4, 255], (ps, ps)).astype(np.int32)
        carbon = rng.uniform(0, 250, (ps, ps)).astype(np.float32)
        carbon[::17] = -999.0
        eco = np.stack([rng.uniform(-5, 45, (ps, ps)),
                        rng.uniform(0, 1, (ps, ps))], -1).astype(np.float32)
        change = rng.choice([0, 0, 0, 1, 255], (ps, ps)).astype(np.int32)
        risk = rng.uniform(0, 1, (ps, ps)).astype(np.float32)
        yield (s1, s2, scl, aux, s1t2, s2t2, scl.copy(), aux.copy(),
               label, carbon, eco, change, risk)

def smoke_test(cfg=CONFIG):
    print("\n=== SMOKE TEST v4.1 (dummy data) ===")
    ds_tr = make_dataset(lambda: _smoke_generator(cfg, 24), cfg, True)
    ds_va = make_dataset(lambda: _smoke_generator(cfg, 8), cfg, False)

    c1, c2 = 6, 18 + cfg["aux_channels"]
    s1_stats = (np.zeros(c1), np.ones(c1)); s2_stats = (np.zeros(c2), np.ones(c2))

    with strategy.scope():
        model = build_model(cfg, s1_stats, s2_stats)
        opt = _make_optimizer(cfg, 1e-3)
        model.compile(
            optimizer=opt,
            loss={"seg_head": FocalDiceLoss(), "carbon_head": CarbonNLLLoss(),
                  "eco_head": MaskedMSE(), "change_head": MaskedBinaryFocal(),
                  "risk_head": MaskedMAE()},
            loss_weights=cfg["loss_weights"],
            metrics={"seg_head": MeanIoU(cfg["num_classes"]),
                     "carbon_head": CarbonMAE(),
                     "change_head": ChangeF1(),
                     "risk_head": MaskedMAE(),
                     "eco_head": MaskedMSE()})
    model.summary(print_fn=lambda s: None if "model" in s.lower() else print(s))
    model.fit(ds_tr, validation_data=ds_va, epochs=2)

    model.save(MODEL_PATH)
    m2 = tf.keras.models.load_model(MODEL_PATH)
    print("Reload OK — all custom objects deserialized.")

    # Tiled inference on odd-sized scene
    H, W, A = 300, 380, cfg["aux_channels"]
    rng = np.random.default_rng(0)
    s1 = rng.uniform(-30, 0, (H, W, 2)).astype(np.float32)
    s2 = rng.uniform(0, 0.5, (H, W, 10)).astype(np.float32)
    aux = rng.uniform(-2, 2, (H, W, A)).astype(np.float32)
    t2 = {"s1": s1 + 0.5, "s2": np.clip(s2 + 0.02, 0, 1), "scl": np.full((H, W), 4, np.float32),
          "aux": aux + 0.1}
    products = predict_scene(m2, s1, s2, aux=aux, t2=t2, cfg=cfg, batch_size=4, tta=True)
    print(f"Inference OK. Products: {list(products.keys())}")
    print(f"  classes shape: {products['classes'].shape}")
    print(f"  carbon_mean range: [{products['carbon_mean'].min():.1f}, {products['carbon_mean'].max():.1f}]")
    print(f"  change_prob range: [{products['change_prob'].min():.3f}, {products['change_prob'].max():.3f}]")
    print("\n=== SMOKE TEST PASSED ===")
    return True

# ─────────────────────────────────────────────
# 17. MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke_test()
    elif mode == "train":
        train()
    elif mode == "infer":
        # Example: python script.py infer /path/to/scene.npz /path/to/ref.tif /output/dir/
        if len(sys.argv) < 5:
            print("Usage: python script.py infer <scene.npz> <ref.tif> <out_dir>")
            sys.exit(1)
        scene_path, ref_path, out_dir = sys.argv[2], sys.argv[3], sys.argv[4]
        model = tf.keras.models.load_model(MODEL_PATH)
        export_all_products(model, scene_path, ref_path, out_dir)
    else:
        print(f"Unknown mode: {mode}. Use: smoke | train | infer")