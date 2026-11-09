"""
================================================================================
 MANGROVE FUSION SEGMENTATION v3
 Sentinel-1 SAR + Sentinel-2 optical dual-branch U-Net
 Classes: 0 Bare Earth/Agri | 1 Tidal Water | 2 Permanent Flood | 3 Degraded Mangrove | 4 Dense Mangrove

 NEW IN v3
 ---------
 * Flood/tidal-prone area mapping (probability export + flood-risk raster)
 * Spectral indices computed in-model: NDVI, NDWI, MNDWI, NDMI, NDRE, BSI + SAR ratio/RVI
 * SCL cloud masking done RIGHT: dilated mask, appended as a channel (no ambiguous zeroing)
 * Tidal variance handling: multi-temporal S1 std channels (VV_std, VH_std)
 * NIR/SWIR exploitation: red-edge (B5/B6/B7), SWIR (B11/B12) indices + SE channel attention
 * Sen1Floods11 (Bonafilia et al., 2020) pretraining + weight transfer
 * Kaggle T4x2 optimized: mixed_float16, MirroredStrategy, BackupAndRestore, disk cache
 * Self-contained .keras model: feed RAW VV/VH + 10 S2 bands + SCL -> segmentation

 NPZ SCHEMA (per scene)
 ----------------------
   s1      : (H,W,2) float32  [VV, VH] in dB
   s1_std  : (H,W,2) float32  [VV_std, VH_std] multi-temporal std (OPTIONAL, zeros if absent)
   s2      : (H,W,10)         [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]; uint16 (0-10000) or float (0-1)
   scl     : (H,W) uint8      Sentinel-2 Scene Classification Layer (OPTIONAL -> assumed clear)
   label   : (H,W) uint8      0..4, 255 = ignore/nodata

 MODES (set MODE at bottom): "smoke" | "pretrain" | "train" | "infer"
================================================================================
"""
import os, sys, gc, glob, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# 1. ENVIRONMENT & GPU (Kaggle T4 x2)
# ─────────────────────────────────────────────
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR  = "/kaggle/working/" if ON_KAGGLE else "./output/"
os.makedirs(WORK_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(42)
tf.keras.mixed_precision.set_global_policy("mixed_float16")   # T4 tensor cores

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print(f"GPUs: {len(gpus)}")

strategy = tf.distribute.MirroredStrategy()
N_REPLICAS = strategy.num_replicas_in_sync
print(f"Replicas: {N_REPLICAS}")

# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # data
    "train_dir":        "/kaggle/input/mangrove-npz/train/",
    "val_dir":          "/kaggle/input/mangrove-npz/val/",
    "pretrain_dir":     "/kaggle/input/sen1floods11-npz/",     # output of convert_sen1floods11()
    "patch_size":       256,
    "train_stride":     128,          # 50% overlap -> more samples
    "max_train_patches": 5000,
    "max_val_patches":  1000,
    "min_valid_frac":   0.40,         # drop patches with <40% labelled pixels
    "max_cloud_frac":   0.90,         # drop train patches >90% cloud
    "ignore_label":     255,
    "num_classes":      5,
    # SCL classes to mask: 0 nodata,1 saturated,2 dark/shadow,3 cloud shadow,
    # 7 low-prob cloud,8 med cloud,9 high cloud,10 cirrus,11 snow
    "invalid_scl":      [0, 1, 2, 3, 7, 8, 9, 10, 11],
    "cloud_dilate":     5,            # maxpool k -> dilates mask ~2px (cloud edges)
    # model
    "s2_widths":        [32, 64, 128, 256],
    "s1_widths":        [16, 32, 64, 128],   # SAR branch half-width (fewer channels)
    "bottleneck":       512,
    "dec_widths":       [256, 128, 64, 32],
    # training
    "batch_per_replica": 16,          # T4 16GB: 16x256x256 safe -> global 32 on T4x2
    "epochs":           100,
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "focal_gamma":      2.0,
    "dice_weight":      1.0,
    "focal_weight":     1.0,
    "jit_compile":      False,        # set True to try XLA (can be flaky w/ custom layers)
    "use_disk_cache":   False,        # True -> cache compressed patches on disk (needs ~15GB)
    "pretrained_path":  os.path.join(WORK_DIR, "sen1floods11_pretrained.keras"),
    # inference
    "infer_batch":      8,            # small for local i7 CPU
    "infer_tta":        True,         # test-time augmentation (4x slower, smoother maps)
}
GLOBAL_BATCH = CONFIG["batch_per_replica"] * N_REPLICAS

MODEL_PATH = os.path.join(WORK_DIR, "mangrove_fusion_v3.keras")
STATS_PATH = os.path.join(WORK_DIR, "norm_stats.npz")

CLASS_NAMES  = ["Bare Earth/Agri", "Tidal Water", "Perm Flood", "Deg. Mangrove", "Dense Mangrove"]
CLASS_COLORS = ["saddlebrown", "deepskyblue", "midnightblue", "lightgreen", "darkgreen"]

# ─────────────────────────────────────────────
# 3. NPZ LOADING & PATCH GENERATOR
# ─────────────────────────────────────────────
def load_npz(path):
    """Robust loader: auto-scales S2, auto-dB S1, fills missing SCL / temporal std."""
    with np.load(path) as d:                                   # closes handle properly
        s1 = d["s1"].astype(np.float32)
        s1_std = d["s1_std"].astype(np.float32) if "s1_std" in d.files else np.zeros_like(s1)
        s2 = d["s2"].astype(np.float32)
        scl = d["scl"].astype(np.float32)[..., None] if "scl" in d.files \
              else np.full(s2.shape[:2] + (1,), 4.0, np.float32)   # 4 = vegetation (valid)
        label = d["label"].astype(np.int32)

    if s2.max() > 20000:   s2 *= 1.0 / 65535.0     # uint16 full-range storage
    elif s2.max() > 100:   s2 *= 1e-4              # 0-10000 reflectance storage
    if s1.min() >= 0:      s1 = 10.0 * np.log10(np.clip(s1, 1e-6, None))  # linear -> dB
    s1 = np.clip(s1, -35.0, 5.0)

    s1 = np.concatenate([s1, s1_std], axis=-1)     # (H,W,4): VV,VH,VV_std,VH_std
    return s1, s2, scl, label

def patch_generator(file_list, cfg, training, max_patches):
    ps     = cfg["patch_size"]
    stride = cfg["train_stride"] if training else ps
    ignore = cfg["ignore_label"]
    invalid = np.array(cfg["invalid_scl"])
    emitted = 0
    for path in file_list:
        s1, s2, scl, label = load_npz(path)
        h, w = label.shape
        if h < ps or w < ps:
            continue
        for i in range(0, h - ps + 1, stride):
            for j in range(0, w - ps + 1, stride):
                if emitted >= max_patches:
                    return
                lab = label[i:i+ps, j:j+ps]
                valid_frac = float(np.mean(lab != ignore))
                if valid_frac < cfg["min_valid_frac"]:
                    continue
                if training:
                    cloud_frac = float(np.mean(np.isin(scl[i:i+ps, j:j+ps], invalid)))
                    if cloud_frac > cfg["max_cloud_frac"]:
                        continue
                yield (s1[i:i+ps, j:j+ps], s2[i:i+ps, j:j+ps],
                       scl[i:i+ps, j:j+ps], lab)
                emitted += 1
    gc.collect()

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING LAYER (lives INSIDE the model -> self-contained .keras)
#    S1: VV,VH,VH-VV,RVI,VV_std,VH_std          (6 ch)
#    S2: 10 raw + NDVI,NDWI,MNDWI,NDMI,NDRE,BSI + cloud mask  (17 ch)
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FeatureEngineer(layers.Layer):
    def __init__(self, invalid_scl=(0,1,2,3,7,8,9,10,11), dilate=5, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.invalid_scl = [int(c) for c in invalid_scl]
        self.dilate = int(dilate)
        self.eps = float(eps)

    def _ndx(self, a, b):
        return (a - b) / (a + b + self.eps)

    def call(self, inputs):
        s1, s2, scl = inputs
        s1 = tf.cast(s1, tf.float32); s2 = tf.cast(s2, tf.float32)

        # ---- SAR features (tidal variance = VV_std/VH_std channels) ----
        vv, vh, vv_std, vh_std = tf.split(s1, 4, axis=-1)
        lin_vv = tf.pow(10.0, vv / 10.0)
        lin_vh = tf.pow(10.0, vh / 10.0)
        rvi = 4.0 * lin_vh / (lin_vv + lin_vh + self.eps)          # radar vegetation index
        s1_feat = tf.concat([vv, vh, vh - vv, rvi, vv_std, vh_std], axis=-1)

        # ---- Optical indices (NIR = B8 idx6, SWIR1 = B11 idx8, red-edge = B5 idx3) ----
        b2,b3,b4,b5,b6,b7,b8,b8a,b11,b12 = tf.split(s2, 10, axis=-1)
        ndvi  = self._ndx(b8, b4)        # canopy greenness  -> dense vs degraded
        ndwi  = self._ndx(b3, b8)        # McFeeters open water
        mndwi = self._ndx(b3, b11)       # Xu: water vs built-up/soil
        ndmi  = self._ndx(b8, b11)       # canopy moisture stress -> degradation
        ndre  = self._ndx(b8, b5)        # red-edge: chlorophyll / canopy density
        bsi   = self._ndx(b11 + b4, b8 + b2)   # bare soil
        s2_feat = tf.concat([s2, ndvi, ndwi, mndwi, ndmi, ndre, bsi], axis=-1)  # 16

        # ---- SCL cloud mask: dilated, appended as channel (NOT ambiguous zeroing) ----
        scl_i = tf.cast(scl, tf.int32)
        mask = tf.zeros(tf.shape(scl_i), dtype=tf.bool)
        for c in self.invalid_scl:
            mask = tf.logical_or(mask, tf.equal(scl_i, c))
        mask_f = tf.cast(mask, tf.float32)
        if self.dilate > 1:
            mask_f = tf.nn.max_pool2d(mask_f, ksize=self.dilate, strides=1, padding="SAME")
            mask_f = tf.cast(mask_f > 0.0, tf.float32)
        s2_feat = s2_feat * (1.0 - mask_f)                       # neutralize cloudy px
        s2_feat = tf.concat([s2_feat, mask_f], axis=-1)          # 17 ch; model KNOWS where clouds are
        return s1_feat, s2_feat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"invalid_scl": self.invalid_scl, "dilate": self.dilate, "eps": self.eps})
        return cfg

# ─────────────────────────────────────────────
# 5. NORMALIZATION STATS + CLASS ALPHA (computed through the SAME layer -> no skew)
# ─────────────────────────────────────────────
def compute_stats(file_list, cfg, max_patches=300):
    print("Computing normalization stats + class frequencies...")
    fe = FeatureEngineer(cfg["invalid_scl"], cfg["cloud_dilate"])
    C = cfg["num_classes"]
    n = 0
    s1_sum = np.zeros(6);  s1_sq = np.zeros(6)
    s2_sum = np.zeros(17); s2_sq = np.zeros(17)
    counts = np.zeros(C, np.int64)
    for s1, s2, scl, lab in patch_generator(file_list, cfg, training=False, max_patches=max_patches):
        f1, f2 = fe([s1[None], s2[None], scl[None]])
        a = f1[0].numpy().reshape(-1, 6); b = f2[0].numpy().reshape(-1, 17)
        s1_sum += a.sum(0); s1_sq += (a**2).sum(0)
        s2_sum += b.sum(0); s2_sq += (b**2).sum(0)
        n += a.shape[0]
        v = lab[lab != cfg["ignore_label"]]
        counts += np.bincount(v, minlength=C)
    s1_mean = s1_sum / n; s1_var = np.clip(s1_sq / n - s1_mean**2, 1e-6, None)
    s2_mean = s2_sum / n; s2_var = np.clip(s2_sq / n - s2_mean**2, 1e-6, None)
    s2_mean[-1] = 0.0; s2_var[-1] = 1.0                          # mask channel: identity

    freq = counts / max(counts.sum(), 1)
    alpha = 1.0 / (freq + 1e-6)
    alpha = np.clip(alpha / alpha.mean(), 0.25, 4.0)             # median-ish balanced, clipped
    np.savez(STATS_PATH, s1_mean=s1_mean, s1_var=s1_var,
             s2_mean=s2_mean, s2_var=s2_var, alpha=alpha, counts=counts)
    print(f"  class counts: {counts.tolist()}  alpha: {np.round(alpha,3).tolist()}")
    return s1_mean, s1_var, s2_mean, s2_var, alpha

def load_or_compute_stats(file_list, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        return z["s1_mean"], z["s1_var"], z["s2_mean"], z["s2_var"], z["alpha"]
    return compute_stats(file_list, cfg)

# ─────────────────────────────────────────────
# 6. tf.data PIPELINE
# ─────────────────────────────────────────────
def make_dataset(file_list, cfg, training, max_patches, cache_path=None):
    ps = cfg["patch_size"]
    sig = (
        tf.TensorSpec((ps, ps, 4),  tf.float32),    # s1: VV,VH,VV_std,VH_std
        tf.TensorSpec((ps, ps, 10), tf.float32),    # s2 raw
        tf.TensorSpec((ps, ps, 1),  tf.float32),    # scl
        tf.TensorSpec((ps, ps),     tf.int32),      # label
    )
    ds = tf.data.Dataset.from_generator(
        lambda: patch_generator(file_list, cfg, training, max_patches), output_signature=sig)

    if cache_path:   # compress -> cache -> decompress (saves ~50% disk)
        ds = ds.map(lambda a,b,c,d: (tf.cast(a,tf.float16), tf.cast(b,tf.float16), c, tf.cast(d,tf.int16)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache(filename=cache_path)
        ds = ds.map(lambda a,b,c,d: (tf.cast(a,tf.float32), tf.cast(b,tf.float32), c, tf.cast(d,tf.int32)),
                    num_parallel_calls=tf.data.AUTOTUNE)

    def parse(s1, s2, scl, label):
        y = tf.where(label == cfg["ignore_label"], -1, label)   # -1 -> all-zeros one-hot
        y = tf.one_hot(y, depth=cfg["num_classes"], dtype=tf.float32)
        return {"s1_input": s1, "s2_input": s2, "scl_input": scl}, y

    def augment(inputs, y):
        s1, s2, scl = inputs["s1_input"], inputs["s2_input"], inputs["scl_input"]
        do_lr = tf.random.uniform(()) > 0.5
        s1, s2, scl, y = tf.cond(do_lr, lambda: (tf.image.flip_left_right(s1), tf.image.flip_left_right(s2),
                                                 tf.image.flip_left_right(scl), tf.image.flip_left_right(y)),
                                        lambda: (s1, s2, scl, y))
        do_ud = tf.random.uniform(()) > 0.5
        s1, s2, scl, y = tf.cond(do_ud, lambda: (tf.image.flip_up_down(s1), tf.image.flip_up_down(s2),
                                                 tf.image.flip_up_down(scl), tf.image.flip_up_down(y)),
                                        lambda: (s1, s2, scl, y))
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        s1, s2, scl, y = tf.image.rot90(s1,k), tf.image.rot90(s2,k), tf.image.rot90(scl,k), tf.image.rot90(y,k)
        return {"s1_input": s1, "s2_input": s2, "scl_input": scl}, y

    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1024, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options(); opts.experimental_deterministic = False
    ds = ds.with_options(opts)
    return ds.batch(GLOBAL_BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# ─────────────────────────────────────────────
# 7. LOSS (masked Focal + soft Dice, per-class alpha) & METRIC (masked MeanIoU)
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=1.0, **kwargs):
        super().__init__(name="focal_dice", **kwargs)
        self.alpha = None if alpha is None else [float(a) for a in alpha]
        self.gamma = float(gamma); self.focal_weight = float(focal_weight); self.dice_weight = float(dice_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        valid = tf.cast(tf.reduce_sum(y_true, -1, keepdims=True) > 0, tf.float32)   # ignore 255

        a = tf.ones_like(y_pred) if self.alpha is None else tf.constant(self.alpha, tf.float32)
        ce = -y_true * tf.math.log(y_pred)
        fl = tf.reduce_sum(a * tf.pow(1.0 - y_pred, self.gamma) * ce, axis=-1, keepdims=True) * valid
        focal = tf.reduce_sum(fl, axis=[1,2,3]) / (tf.reduce_sum(valid, axis=[1,2,3]) + eps)

        inter = tf.reduce_sum(y_true * y_pred, axis=[1,2])
        denom = tf.reduce_"""
================================================================================
 MANGROVE ECO-MONITOR v5.0  "BLUE SENTINEL"
 Production-grade dual-modality (Sentinel-1 SAR x Sentinel-2 optical) mangrove
 monitoring system: Swin-V2 Siamese U-Net, per-level gated fusion, time-aware
 change detection, pluggable multi-task heads, uncertainty-aware MRV outputs.

 REQUIRES: Python >= 3.10, TensorFlow >= 2.16 (Keras 3), Linux or WSL2 on
           Windows (native Windows has NO TensorFlow GPU support since TF 2.10).
           Optional: rasterio (COG/GeoJSON export), scikit-learn (eval fallback
           exists), matplotlib (plots).

--------------------------------------------------------------------------------
 WHAT V5 FIXES (all confirmed defects from the v4.1 review + code audit)
--------------------------------------------------------------------------------
 [F1] Keras-3 fatal crash: tf.constant() buffers created in build() go out of
      scope under re-tracing. v5 stores EVERY buffer (relative coords, shift
      masks) as non-trainable weights via add_weight(Constant). The Swin-V2
      CPB design removes the giant RPB table entirely.
 [F2] Normalization off-by-one: cloud-mask identity stats were written to
      channel 18 (elevation) instead of 17 (mask). v5 computes the mask index
      symbolically from the feature layout; aux channels now normalized with
      their real stats.
 [F3] Hardware detection: "NVIDIA RTX 6000 Ada Generation" matched nothing and
      silently fell back to the 16 GB T4 preset. v5 adds explicit RTX-6000-Ada
      matching PLUS a VRAM-probing tier fallback via nvidia-smi.
 [F4] Double gradient accumulation (native AdamW kwarg + custom train_step
      wrapper fighting each other). v5 uses ONE mechanism: the native Keras
      optimizer path, with a version guard and a hard error if unavailable.
 [F5] Four of five heads had no label source. v5 makes heads PLUGGABLE:
      seg/change/quality/boundary ship ON (all have real labels), carbon/eco/
      risk ship OFF with documented data sources; a pre-flight scan validates
      enabled heads against the NPZ keys actually present.
 [F6] Windows GPU: v5 detects Windows + CUDA-visible-but-TF-invisible GPUs and
      prints exact WSL2 remediation steps.

 NEW BUGS FOUND BY V5 AUDIT (were NOT in the review):
 [F7] Augmentation channel misalignment: random_aug sliced x[..., 14:18] /
      18:22 / 22:32 assuming aux=4, but aux=6 -> brightness/contrast was applied
      to SCL+AUX and S1-T2 bands; everything downstream was corrupted. v5 uses
      cumulative symbolic offsets derived from SIZES.
 [F8] Gaussian noise was added to the WHOLE packed tensor including one-hot seg
      labels, carbon/eco/change/risk targets -> destroyed label integrity.
      v5 restricts noise to input bands only.
 [F9] TTA blending bug: predictions were accumulated once PER TTA variant but
      the weight map only ONCE per tile -> all outputs inflated ~4x with TTA on.
      v5 accumulates weights per variant pass.
 [F10] Inverted blending kernel: v4's "gaussian" peaked at tile EDGES and was
      ~0.01 at the centre. v5 uses a proper centre-peaked Hann window.

--------------------------------------------------------------------------------
 V5 MAJOR UPGRADES (grounded in 2025-2026 research)
--------------------------------------------------------------------------------
 U1  Swin-V2 blocks: res-post-norm + scaled-cosine attention with learnable
     per-head logit scale (clamped) + log-spaced Continuous Position Bias MLP.
     More stable at depth, transferable across window sizes.
 U2  Per-level fusion: cheap gated 1x1 fusion at high-res levels, bidirectional
     cross-attention at deep levels (v4 fused ONLY at the bottleneck).
 U3  Time-aware Siamese change decoder: per-level feature diffs as skips +
     FiLM conditioning on real acquisition-date gap (decimal years). Enables
     backcasting to arbitrary dates and GMW-style annual loss products.
 U4  Pluggable task heads with LEARNABLE uncertainty weighting (Kendall et
     al.): loss weights are learned log-variances, not hand-tuned constants.
 U5  Free-supervision heads: quality head (pixel usability from SCL, trained
     WITHOUT extra labels) and boundary head (morphological edge target derived
     from seg labels) -> sharp coastlines + reliability masks at inference.
 U6  Deep supervision (aux seg logits at 1/4 res) with small weight.
 U7  SimMIM-style self-supervised pretraining mode for unlabeled S1+S2 scenes
     (masked-block reconstruction), then fine-tune with by-name weight load.
     This is the standard answer to EO label scarcity (MAE/SimMIM/SatMAE line).
 U8  Modality dropout: randomly zero the SAR or optical branch during training
     so the model stays operational when S2 is cloudy (ESA RAMM shows SAR-only
     alerting is a hard operational requirement in the tropics). predict_scene
     supports sar_only=True.
 U9  Failure tolerance: NaN/Inf NPZ sanitization with counts, schema validation
     with actionable errors, OOM auto-retry with halved batch (resumes from
     last checkpoint), memory preflight probe, TerminateOnNaN, BackupAndRestore.
 U10 EMA of weights (Polyak averaging) - evaluation, checkpointing and export
     all run on EMA weights; separate EMA checkpoint saved.
 U11 Post-hoc temperature calibration of seg probabilities on the val set;
     calibrated probabilities used at inference and stored in norm stats.
 U12 MRV-ready export pack: COG GeoTIFFs + STAC 1.0 item JSON + change-alert
     GeoJSON polygons (area, period, confidence) + summary CSV with class areas
     (ha) and carbon stock with 95% CI in tCO2e (VM0033-aligned reporting
     primitives) + preview PNG.
 U13 Class-balanced patch sampling: rare classes (perm flood, degraded
     mangrove) oversampled via per-patch weighting.
 U14 Hardware: RTX-6000-Ada preset, generic VRAM tiers, optional batch
     auto-tuner that empirically probes the largest safe batch.
 U15 NISAR/L-band readiness: aux channel count is config-driven; drop NISAR
     L-band HV/HH ratio or ESA BIOMASS P-band features into aux and everything
     (stats, stems, aug) adapts. aux_names/scale are configurable.

 NPZ SCHEMA v5 (backward-compatible superset of v4.1)
   required: s1 (H,W,2|4) dB | s2 (H,W,10) reflectance
   optional: s1_std, scl, aux (H,W,A), label (0..4, 255=ignore),
             carbon (t/ha, -999), eco (H,W,2, -999), change (0/1, 255),
             risk (0..1, -999), s1_t2/s2_t2/scl_t2/aux_t2,
   NEW:      date_t1, date_t2 : decimal year scalars (e.g. 2024.42) ->
             drives the time-aware change decoder. If a real T2 exists but
             dates are absent, dt defaults to 1.0.

 HEAD DATA SOURCES (honest wiring)
   seg     : WorldCover + GSW + GMW v4.x derived labels            [ON ]
   change  : Hansen GFC v1.13 lossyear / GMW alerts / T2 pairs     [ON ]
   quality : derived from SCL at train time (no NPZ field needed)  [ON ]
   boundary: derived from seg labels at train time                 [ON ]
   carbon  : ESA CCI Biomass / GEDI / NISAR L-band products        [OFF]
   eco     : Landsat/MODIS LST (soil moisture NOT viable @10m)     [OFF]
   risk    : hydrodynamic surge model or heuristic (weak labels)   [OFF]

 MODES:  python mangrove_v5.py smoke | pretrain | train | eval | infer | export
================================================================================
"""

import os
import gc
import sys
import json
import glob
import math
import time
import shutil
import argparse
import platform
import subprocess
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "5.0.0"
MIN_TF = (2, 16)


# ==============================================================================
# 1. ENVIRONMENT, GPU, PLATFORM GUARDS
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


def environment_check():
    """Fail fast with actionable messages. Addresses review F1/F6 context."""
    ok = True
    if _parse_version(tf.__version__) < MIN_TF:
        print(f"[FATAL] TensorFlow {tf.__version__} < {'.'.join(map(str, MIN_TF))}. "
              f"v5 requires Keras 3 APIs. pip install -U tensorflow")
        ok = False

    gpus = tf.config.list_physical_devices("GPU")
    if platform.system() == "Windows" and not gpus:
        smi = shutil.which("nvidia-smi")
        if smi is not None:
            print("=" * 78)
Web search


            print("[HARDWARE BLOCKER] Windows detected, NVIDIA driver present, but"
                  " TensorFlow sees NO GPU.")
            print("TensorFlow dropped native-Windows GPU support after v2.10. Fix:")
            print("  1) wsl --install -d Ubuntu   (PowerShell, admin)")
            print("  2) Install the Windows NVIDIA driver (it serves WSL2 too; do NOT")
            print("     install a Linux driver inside WSL2)")
            print("  3) Inside Ubuntu: python -m pip install tensorflow[and-cuda]")
            print("  4) Verify: python -c \"import tensorflow as tf; "
                  "print(tf.config.list_physical_devices('GPU'))\"")
            print("  WSL2 gives near-native CUDA performance for this workload.")
            print("=" * 78)
            ok = False
    return ok, gpus


ENV_OK, GPUS = environment_check()
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v5/"
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
    """Best-effort VRAM probe via nvidia-smi (works on Linux/WSL2)."""
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
    train_dir: str = "/kaggle/input/mangrove-v4-npz/train/"
    val_dir: str = "/kaggle/input/mangrove-v4-npz/val/"
    test_dir: str = "/kaggle/input/mangrove-v4-npz/test/"
    patch_size: int = 256                 # -> maps 64/32/16/8, window 8 divides all
    train_stride: int = 128
    max_train_patches: int = 8000
    max_val_patches: int = 1200
    min_valid_frac: float = 0.35
    max_cloud_frac: float = 0.85
    ignore_label: int = 255
    num_classes: int = 5
    invalid_scl: tuple = (0, 1, 2, 3, 7, 8, 9, 10, 11)
    cloud_dilate: int = 5
    # aux channels are config-driven (NISAR-ready: add L-band features here)
    aux_names: tuple = ("elev", "slope", "ndvi_trend", "flood_freq",
                        "lst_trend", "s1_trend")
    aux_scale: tuple = (50.0, 20.0, 0.1, 1.0, 50.0, 1.0)
    # class-balanced sampling
    rare_classes: tuple = (2, 3)          # perm flood, degraded mangrove
    rare_boost: float = 4.0
    # ---- model ----
    embed_dims: tuple = (48, 96, 192, 384)
    depths: tuple = (2, 2, 2, 2)
    window_size: int = 8
    drop_path: float = 0.15               # linear stochastic-depth schedule 0 -> dp
    mlp_ratio: float = 4.0
    fusion_heads: int = 8
    cross_attn_levels: tuple = (2, 3)     # full cross-attn only at deep levels
    dec_widths: tuple = (192, 96, 48)
    refine_widths: tuple = (32, 24)
    change_widths: tuple = (128, 96, 64, 48, 32)
    decoder_swin_levels: int = 2
    use_se: bool = True
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.3
    time_embed_dim: int = 64
    active_heads: tuple = ("seg", "change", "quality", "boundary")
    # heads OFF by default; enable only after wiring real labels (see header)
    # e.g. active_heads = ("seg","change","quality","boundary","carbon")
    # ---- training ----
    hardware: str = "auto"
    batch_per_replica: int = 4
    grad_accum: int = 1                   # 1 = native single-step (recommended)
    epochs: int = 120
    lr: float = 2e-4
    lr_warmup_epochs: int = 3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    task_weighting: str = "learned"       # "learned" (Kendall) or "fixed"
    fixed_loss_weights: dict = field(default_factory=lambda: {
        "seg": 1.0, "change": 1.2, "quality": 0.2, "boundary": 0.3,
        "carbon": 0.6, "eco": 0.25, "risk": 0.25})
    carbon_scale: float = 100.0
    eco_scale: tuple = (50.0, 1.0)
Web search


    ema_decay: float = 0.999
    grad_clip: float = 1.0
    deterministic: bool = False
    auto_batch: bool = False              # empirical preflight probe
    oom_retries: int = 2                  # halve batch and resume on OOM
    pretrain_weights: str = ""            # path to SimMIM encoder weights
    # ---- augmentation ----
    augment: bool = True
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_noise: float = 0.02
    modality_dropout_p: float = 0.08      # per-modality, same choice for T1/T2
    # ---- pretraining ----
    pretrain_epochs: int = 40
    pretrain_lr: float = 3e-4
    pretrain_mask_ratio: float = 0.5
    pretrain_block: int = 32
    # ---- inference ----
    infer_batch: int = 8
    infer_tta: bool = True
    infer_overlap: float = 0.5
    mc_samples: int = 0                   # >1 enables MC-dropout epistemic maps
    seg_temperature: float = 1.0          # filled by calibration
    change_threshold: float = 0.5
    # ---- MRV ----
    pixel_size_m: float = 10.0            # fallback if ref tif unavailable
    carbon_to_co2e: float = 44.0 / 12.0   # tC -> tCO2e
    c_fraction: float = 0.5               # biomass -> carbon
    shuffle_buffer: int = 256

    @property
    def aux_channels(self):
        return len(self.aux_names)

    def save(self, path):
        d = asdict(self)
        d["invalid_scl"] = list(self.invalid_scl)
        d["aux_names"] = list(self.aux_names)
        d["aux_scale"] = list(self.aux_scale)
        d["rare_classes"] = list(self.rare_classes)
        d["embed_dims"] = list(self.embed_dims)
        d["depths"] = list(self.depths)
        d["cross_attn_levels"] = list(self.cross_attn_levels)
        d["dec_widths"] = list(self.dec_widths)
        d["refine_widths"] = list(self.refine_widths)
        d["change_widths"] = list(self.change_widths)
        d["active_heads"] = list(self.active_heads)
        d["eco_scale"] = list(self.eco_scale)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        for k in ("invalid_scl", "aux_names", "aux_scale", "rare_classes",
                  "embed_dims", "depths", "cross_attn_levels", "dec_widths",
                  "refine_widths", "change_widths", "active_heads", "eco_scale"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)


CFG = Config()

HW_PRESETS = {
    # name: (batch_per_replica, grad_accum, infer_batch)
    "h100":        (32, 1, 32),
    "a100":        (24, 1, 32),
    "rtx6000ada":  (16, 1, 16),   # 48 GB Ada workstation card (review F3)
    "a6000":       (16, 1, 16),
    "l40":         (12, 1, 16),
    "4090":        (8, 1, 8),
    "l4":          (6, 1, 8),
    "t4x2":        (4, 2, 8),
    "cpu":         (2, 1, 2),
}


def resolve_hardware(cfg):
    """Explicit name match first, then VRAM tiers, then conservative default."""
    name = " ".join(GPU_NAMES)
    hw = cfg.hardware
    if hw == "auto":
        n = name.lower()
        if "h100" in n:
            hw = "h100"
        elif "a100" in n:
            hw = "a100"
        elif "6000 ada" in n or "rtx 6000" in n:     # RTX 6000 Ada (48 GB)
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
# 3. NPZ LOADING WITH VALIDATION + SANITIZATION (failure tolerance)
# ==============================================================================
class NPZSchemaError(ValueError):
    pass


def _process_s1(s1):
    s1 = s1.astype(np.float32)
    if np.nanmin(s1) >= 0:                                # linear -> dB
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
    """NaN/Inf guard with counts (F9-tolerance). Returns fixed copy."""
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / arr.size
        if frac > 0.30:
            raise NPZSchemaError(f"{path}: '{name}' is {frac:.0%} non-finite")
        print(f"  [sanitize] {os.path.basename(path)}:{name} "
              f"had {n_bad} non-finite px -> 0")
        arr = np.where(bad, 0.0, arr)
    return arr


def load_npz(path, cfg=CFG):
    """Returns 15 arrays (13 v4.1 fields + date_t1 + date_t2).
    Missing -> sentinel-filled; NaN/Inf sanitized with counts."""
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
    with np.load(path) as d:
        files = d.files
        if "s1" not in files or "s2" not in files:
            raise NPZSchemaError(f"{path}: missing required keys 's1'/'s2' "
                                 f"(found: {sorted(files)})")
        s1 = _sanitize(_process_s1(d["s1"]), "s1", path)
        s2 = _sanitize(_process_s2(d["s2"]), "s2", path)
        if s1.ndim != 3 or s2.ndim != 3 or s2.shape[-1] != 10:
            raise NPZSchemaError(f"{path}: bad shapes s1{s1.shape} s2{s2.shape}")
        H, W = s2.shape[:2]
        if s1.shape[:2] != (H, W):
            raise NPZSchemaError(f"{path}: s1/s2 spatial mismatch "
                                 f"{s1.shape[:2]} vs {(H, W)}")

        s1_std = (d["s1_std"].astype(np.float32)
                  if "s1_std" in files else np.zeros((H, W, 2), np.float32))
        s1 = np.concatenate([s1, _sanitize(s1_std, "s1_std", path)], -1)
        scl = (d["scl"].astype(np.float32)[..., None]
               if "scl" in files else np.full((H, W, 1), 4.0, np.float32))
        aux = (d["aux"].astype(np.float32)
               if "aux" in files else np.zeros((H, W, A), np.float32))
        if aux.shape[-1] != A:
            fixed = np.zeros((H, W, A), np.float32)
            n = min(A, aux.shape[-1])
            fixed[..., :n] = aux[..., :n]
            aux = fixed
        aux = _sanitize(aux, "aux", path) / scale

        label = (d["label"].astype(np.int32)
                 if "label" in files else np.full((H, W), 255, np.int32))
        carbon = (d["carbon"].astype(np.float32)
                  if "carbon" in files else np.full((H, W), -999.0, np.float32))
        eco = (d["eco"].astype(np.float32)
               if "eco" in files else np.full((H, W, 2), -999.0, np.float32))
        change = (d["change"].astype(np.int32)
                  if "change" in files else np.full((H, W), 255, np.int32))
        risk = (d["risk"].astype(np.float32)
                if "risk" in files else np.full((H, W), -999.0, np.float32))

        has_t2 = "s1_t2" in files and "s2_t2" in files
        if has_t2:
            s1t2 = _sanitize(_process_s1(d["s1_t2"]), "s1_t2", path)
            s1t2_std = (d["s1_t2_std"].astype(np.float32)
                        if "s1_t2_std" in files else np.zeros_like(s1t2))
            s1t2 = np.concatenate([s1t2, s1t2_std], -1)
            s2t2 = _sanitize(_process_s2(d["s2_t2"]), "s2_t2", path)
            sclt2 = (d["scl_t2"].astype(np.float32)[..., None]
                     if "scl_t2" in files else np.full((H, W, 1), 4.0, np.float32))
            auxt2 = ((d["aux_t2"].astype(np.float32)
                      if "aux_t2" in files else np.zeros((H, W, A), np.float32))
                     / scale)
        else:
            s1t2, s2t2 = s1.copy(), s2.copy()
            sclt2, auxt2 = scl.copy(), aux.copy()
            change = np.full((H, W), 255, np.int32)   # never learn from copy

        date_t1 = float(d["date_t1"]) if "date_t1" in files else 0.0
        date_t2 = float(d["date_t2"]) if "date_t2" in files else (1.0 if has_t2 else 0.0)

    return (s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
            label, carbon, eco, change, risk,
            np.float32(date_t1), np.float32(date_t2))


HEAD_REQUIRED_KEYS = {
    "seg": ["label"], "change": ["change"], "carbon": ["carbon"],
    "eco": ["eco"], "risk": ["risk"],
    "quality": [], "boundary": [],      # derived on-the-fly: always available
}


def validate_heads_vs_data(files, cfg, sample=5):
    """Pre-flight: ensure every enabled head actually has labels in the NPZs."""
    present = set()
    for p in files[:sample]:
        with np.load(p) as d:
            present |= set(d.files)
    active, warnings_out = [], []
    for h in cfg.active_heads:
        need = HEAD_REQUIRED_KEYS.get(h, [])
        if need and not any(k in present for k in need):
            warnings_out.append(
                f"  [heads] '{h}' enabled but no '{need[0]}' key found in sampled "
                f"NPZs -> head DISABLED (wire real labels, then re-enable).")
        else:
            active.append(h)
    if not active:
        raise NPZSchemaError("No task head has data; check your NPZ schema.")
    for w in warnings_out:
        print(w)
    if tuple(active) != tuple(cfg.active_heads):
        cfg.active_heads = tuple(active)
        print(f"  [heads] active set resolved to: {list(active)}")
    return cfg


# ==============================================================================
# 4. PATCH GENERATOR (class-balanced for training)
# ==============================================================================
def _patch_weight(lab, cfg):
    """Higher sampling weight when rare classes are present."""
    C = cfg.num_classes
    valid = lab != cfg.ignore_label
    if valid.sum() == 0:
        return 0.0
    counts = np.bincount(lab[valid].ravel(), minlength=C).astype(np.float32)
    frac = counts / max(counts.sum(), 1)
    w = 1.0 + cfg.rare_boost * float(frac[list(cfg.rare_classes)].sum())
    return w


def patch_generator(file_list, cfg, training, max_patches, seed=42):
    ps = cfg.patch_size
    stride = cfg.train_stride if training else ps
    ignore = cfg.ignore_label
    invalid = np.array(list(cfg.invalid_scl))
    rng = np.random.default_rng(seed)

    if training:
        # pass 1: collect candidate coords + weights (labels only - cheap)
        pool = []   # (path, i, j, weight)
        for path in file_list:
            data = load_npz(path, cfg)
            label, scl = data[8], data[2]
            h, w = label.shape
            if h < ps or w < ps:
                continue
            for i in range(0, h - ps + 1, stride):
                for j in range(0, w - ps + 1, stride):
                    lab = label[i:i + ps, j:j + ps]
                    if float(np.mean(lab != ignore)) < cfg.min_valid_frac:
                        continue
                    if float(np.mean(np.isin(scl[i:i + ps, j:j + ps], invalid))) \
                            > cfg.max_cloud_frac:
                        continue
                    wgt = _patch_weight(lab, cfg)
                    if wgt > 0:
                        pool.append((path, i, j, wgt))
            del data
        gc.collect()
        if not pool:
            return
        wts = np.array([p[3] for p in pool], np.float64)
        wts /= wts.sum()
        idx = rng.choice(len(pool), size=min(max_patches, len(pool)),
                         replace=False, p=wts)
        chosen = {}
        for k in idx:
            chosen.setdefault(pool[int(k)][0], []).append(pool[int(k)])
        # pass 2: load each scene once, emit chosen patches
        emitted = 0
        for path, items in chosen.items():
            data = load_npz(path, cfg)
            for (_, i, j, _w) in items:
                if emitted >= max_patches:
                    return
                sl = np.s_[i:i + ps, j:j + ps]
                yield tuple(arr[sl] if arr.ndim >= 2 else arr for arr in data)
                emitted += 1
            del data
            gc.collect()
    else:
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
                    lab = label[i:i + ps, j:j + ps]
                    if float(np.mean(lab != ignore)) < cfg.min_valid_frac:
                        continue
                    sl = np.s_[i:i + ps, j:j + ps]
                    yield tuple(arr[sl] if arr.ndim >= 2 else arr for arr in data)
                    emitted += 1
            del data
        gc.collect()


# ==============================================================================
# 5. FEATURE ENGINEERING LAYER
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FeatureEngineer(layers.Layer):
    """S1: [vv, vh, vh-vv, rvi, vv_std, vh_std] (6)
       S2: [10 raw bands, ndvi, ndwi, mndwi, ndmi, ndre, bsi, evi,
            cloud_mask, aux(A)] -> 18 + A channels.
       MASK channel index is SYMBOLIC: S2_MASK_IDX = 17."""

    S1_OUT = 6
    S2_IDX_COUNT = 7
    S2_MASK_IDX = 10 + S2_IDX_COUNT          # = 17  (review F2 fix anchor)
    S2_BASE = 10 + S2_IDX_COUNT + 1          # = 18, before aux

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
C2 = lambda cfg: FeatureEngineer.S2_BASE + cfg.aux_channels   # 18 + A


# ==============================================================================
# 6. NORMALIZATION STATS + CLASS ALPHA (F2 fixed: symbolic mask index)
# ==============================================================================
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v5.npz")


def compute_stats(file_list, cfg, max_patches=400):
    print("Computing normalization stats + class frequencies...")
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, cfg.aux_channels)
    C = cfg.num_classes
    c1, c2 = C1, C2(cfg)
    n = 0
    s1_sum = np.zeros(c1); s1_sq = np.zeros(c1)
    s2_sum = np.zeros(c2); s2_sq = np.zeros(c2)
    counts = np.zeros(C, np.int64)
    for data in patch_generator(file_list, cfg, training=False,
                                max_patches=max_patches):
        s1, s2, scl, aux = data[0], data[1], data[2], data[3]
        lab = data[8]
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
    # FIX F2: identity stats for the cloud-mask channel at its TRUE index.
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


def load_or_compute_stats(file_list, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        print("Loaded cached norm stats.")
        cfg.seg_temperature = float(z["seg_temperature"]) \
            if "seg_temperature" in z.files else 1.0
        return z["s1_mean"], z["s1_var"], z["s2_mean"], z["s2_var"], z["alpha"]
    return compute_stats(file_list, cfg)


# ==============================================================================
# 7. tf.data PIPELINE  (pack -> augment -> unpack), symbolic channel offsets
# ==============================================================================
def output_signature(cfg):
    ps, A = cfg.patch_size, cfg.aux_channels
    F = tf.float32
    return (
        tf.TensorSpec((ps, ps, 4),  F), tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F), tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps, 4),  F), tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F), tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps),     tf.int32), tf.TensorSpec((ps, ps), F),
        tf.TensorSpec((ps, ps, 2),  F), tf.TensorSpec((ps, ps), tf.int32),
        tf.TensorSpec((ps, ps),     F),
        tf.TensorSpec((),           F), tf.TensorSpec((),  F),   # date_t1/t2
    )


# symbolic offsets into the packed tensor (FIX F7: no more hardcoded slices)
def _packed_layout(cfg):
    A, C = cfg.aux_channels, cfg.num_classes
    names = ["s1", "s2", "scl", "aux", "s1t2", "s2t2", "sclt2", "auxt2",
             "seg", "carbon", "eco", "change", "risk", "dt"]
    sizes = [4, 10, 1, A, 4, 10, 1, A, C, 1, 2, 1, 1, 1]
    off, out = 0, {}
    for n, s in zip(names, sizes):
        out[n] = (off, off + s)
        off += s
    return out, off


def make_dataset(gen_fn, cfg, training, batch_size):
    C, A = cfg.num_classes, cfg.aux_channels
    LAY, _ = _packed_layout(cfg)
    carbon_scale = float(cfg.carbon_scale)
    eco_scale = tf.constant(list(cfg.eco_scale), tf.float32)
    invalid_scl = tf.constant(list(cfg.invalid_scl), tf.int32)
    deep = bool(cfg.deep_supervision)

    def pack(s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
             label, carbon, eco, change, risk, dt1, dt2):
        seg = tf.one_hot(tf.where(label == cfg.ignore_label, -1, label),
                         C, dtype=tf.float32)
        change_f = tf.cast(change, tf.float32)
        change_f = tf.where(change_f == 255.0, -1.0, change_f)
        dt = tf.reshape(tf.abs(dt2 - dt1), (1,))
        return tf.concat([s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
                          seg, carbon[..., None], eco,
                          change_f[..., None], risk[..., None], dt], axis=-1)

    def random_aug(x):
        # geometric (applied to EVERYTHING incl. targets: spatially consistent)
        do_lr = tf.random.uniform(()) > 0.5
        x = tf.cond(do_lr, lambda: tf.image.flip_left_right(x), lambda: x)
        do_ud = tf.random.uniform(()) > 0.5
        x = tf.cond(do_ud, lambda: tf.image.flip_up_down(x), lambda: x)
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        x = tf.image.rot90(x, k)

        if training and cfg.augment:
            # FIX F7: photometric ops hit OPTICAL bands only, via symbolic offsets
            delta = tf.random.uniform((), -cfg.aug_brightness, cfg.aug_brightness)
            contra = tf.random.uniform((), 1 - cfg.aug_contrast, 1 + cfg.aug_contrast)

            def photo(t, lo, hi):
                pre, mid, post = t[..., :lo], t[..., lo:hi], t[..., hi:]
                mid = tf.image.adjust_contrast(tf.image.adjust_brightness(mid, delta), contra)
                return tf.concat([pre, mid, post], axis=-1)

            lo1, hi1 = LAY["s2"]
            lo2, hi2 = LAY["s2t2"]
            x = photo(x, lo1, hi1) if hi1 < lo2 else x
            # optical T2 sits later in the tensor; adjust with a rebuild
            if hi1 < lo2:
                pre = x[..., :lo2]
                mid = x[..., lo2:hi2]
                post = x[..., hi2:]
                mid = tf.image.adjust_contrast(
                    tf.image.adjust_brightness(mid, delta), contra)
                x = tf.concat([pre, mid, post], axis=-1)

            # FIX F8: noise on INPUT bands only (never on packed targets)
            n_inputs_end = LAY["auxt2"][1]
            noise = tf.random.normal(tf.shape(x[..., :n_inputs_end]),
                                     stddev=cfg.aug_noise)
            x = tf.concat([x[..., :n_inputs_end] + noise, x[..., n_inputs_end:]],
                          axis=-1)

            # modality dropout: same modality choice for T1 and T2
            p = cfg.modality_dropout_p
            if p > 0:
                drop_opt = tf.random.uniform(()) < p
                drop_sar = tf.logical_and(tf.random.uniform(()) < p,
                                          tf.logical_not(drop_opt))
                one = tf.ones((), tf.float32)
                zero = tf.zeros((), tf.float32)
                g_opt = tf.where(drop_opt, zero, one)     # zeroes S2 bands T1+T2
                g_sar = tf.where(drop_sar, zero, one)     # zeroes S1 bands T1+T2
                s1a, s1b = LAY["s1"]; s2a, s2b = LAY["s2"]
                t1a, t1b = LAY["s1t2"]; t2a, t2b = LAY["s2t2"]
                parts = [
                    x[..., :s1a],
                    x[..., s1a:s1b] * g_sar,
                    x[..., s1b:s2a],
                    x[..., s2a:s2b] * g_opt,
                    x[..., s2b:t1a],
                    x[..., t1a:t1b] * g_sar,
                    x[..., t1b:t2a],
                    x[..., t2a:t2b] * g_opt,
                    x[..., t2b:],
                ]
                x = tf.concat(parts, axis=-1)
        return x

    def unpack_to_xy(x):
        g = lambda n: x[..., LAY[n][0]:LAY[n][1]]
        s1, s2, scl, aux = g("s1"), g("s2"), g("scl"), g("aux")
        s1t2, s2t2, sclt2, auxt2 = g("s1t2"), g("s2t2"), g("sclt2"), g("auxt2")
        seg, carbon, eco, change, risk = g("seg"), g("carbon"), g("eco"), \
            g("change"), g("risk")
        dt = g("dt")

        carbon_y = tf.where(carbon > -900.0, carbon / carbon_scale, -999.0)
        eco_y = tf.where(eco > -900.0, eco / eco_scale, -999.0)
        risk_y = tf.where(risk > -900.0, risk, -999.0)

        # --- free-supervision targets derived on the fly ---
        # quality: 1 = usable pixel (SCL not invalid). Always has labels.
        scl_i = tf.cast(scl, tf.int32)
        bad = tf.zeros(tf.shape(scl_i), tf.bool)
        for c in list(cfg.invalid_scl):
            bad = tf.logical_or(bad, tf.equal(scl_i, c))
        quality_y = 1.0 - tf.cast(bad, tf.float32)

        # boundary: morphological gradient of seg one-hot (valid only where seg is)
        valid_seg = tf.cast(tf.reduce_sum(seg, -1, keepdims=True) > 0, tf.float32)
        mx = tf.nn.max_pool2d(seg, 3, 1, "SAME")
        mn = -tf.nn.max_pool2d(-seg, 3, 1, "SAME")
        edge = tf.reduce_max(tf.clip_by_value(mx - mn, 0.0, 1.0), -1,
                             keepdims=True)
        boundary_y = tf.where(valid_seg > 0, edge, -1.0)

        inputs = {"s1_input": s1, "s2_input": s2, "scl_input": scl,
                  "aux_input": aux, "s1_t2_input": s1t2, "s2_t2_input": s2t2,
                  "scl_t2_input": sclt2, "aux_t2_input": auxt2, "dt_input": dt}
        targets = {"seg_head": seg, "carbon_head": carbon_y, "eco_head": eco_y,
                   "change_head": change, "risk_head": risk_y,
                   "quality_head": quality_y, "boundary_head": boundary_y}
        if deep:
            targets["seg_deep_head"] = tf.nn.avg_pool2d(seg, 4, 4, "VALID")
        return inputs, targets

    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=output_signature(cfg))
    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(cfg.shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(random_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(unpack_to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(cfg.deterministic)
    ds = ds.with_options(opts)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 8. SWIN-V2 COMPONENTS  (all buffers are non-trainable WEIGHTS: F1 fixed)
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
    """Swin-V2 attention: scaled-cosine attention (learnable per-head logit
    scale, clamped) + log-spaced Continuous Position Bias via meta-MLP.
    All coordinate buffers stored as non-trainable weights (Keras-3 safe)."""

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
        # learnable logit scale: init log(10), clamped to <= log(100)
        self.logit_scale = self.add_weight(
            name="logit_scale", shape=(self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(math.log(10.0)),
            trainable=True)
        # CPB meta-network on log-spaced relative coordinates
        self.cpb1 = layers.Dense(self.cpb_hidden, activation="relu", name="cpb1")
        self.cpb2 = layers.Dense(self.num_heads, use_bias=False, name="cpb2")

        # ---- buffers as non-trainable WEIGHTS (FIX F1) ----
        # pairwise relative coords for each of the N x N token pairs, log-spaced
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws),
                                      indexing="ij"))            # (2, ws, ws)
        cf = coords.reshape(2, -1).astype(np.float32)            # (2, N)
        rel = cf[:, :, None] - cf[:, None, :]                    # (2, N, N)
        rel = rel.transpose(1, 2, 0)                             # (N, N, 2)
        rel = rel / max(ws - 1, 1) * 8.0                         # normalize to [-8, 8]
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

        # cosine attention, fp32 for stability
        qn = tf.math.l2_normalize(tf.cast(q, tf.float32), axis=-1)
        kn = tf.math.l2_normalize(tf.cast(k, tf.float32), axis=-1)
        scale = tf.exp(tf.minimum(self.logit_scale, math.log(100.0)))
        attn = tf.matmul(qn, kn, transpose_b=True) * tf.cast(scale, tf.float32)

        # continuous position bias, squashed to [-16, 16] like Swin-V2
        bias = self.cpb2(self.cpb1(self.rel_coords))             # (N, N, heads)
        bias = 16.0 * tf.sigmoid(bias)
        attn += tf.transpose(bias, [2, 0, 1])[None]

        if mask is not None:                                     # (nW, 1, N, N)
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
    """Res-post-norm Swin-V2 block:
       x = norm1(x + dp(attn(x)));  x = norm2(x + dp(mlp(x)))."""

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
            layers.Dense(hidden, activation="gelu"),
            layers.Dropout(self.drop),
            layers.Dense(self.dim),
            layers.Dropout(self.drop)], name="mlp")
        # shifted-window mask as non-trainable WEIGHT (FIX F1, same bug class)
        if self.shift_size > 0:
            mask_np = self._build_mask_np()                      # (nW, 1, N, N)
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
            h = tf.roll(h, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        w = window_partition(h, self.window_size)
        w = self.attn(w, mask=self.attn_mask, training=training)
        h = window_reverse(w, self.window_size, self.H, self.W)
        if self.shift_size > 0:
            h = tf.roll(h, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        x = self.norm1(x + self.dp1(h, training=training))       # post-norm
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
    """Cheap fusion for high-res levels: opt + tanh(gate) * W[opt, sar].
    Gate initialized at 0 -> identity at init (stable)."""

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
    """Bidirectional token cross-attention (deep levels only, small N)."""

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


@tf.keras.utils.register_keras_serializable(package="mangrove")
class TimeEmbedding(layers.Layer):
    """Sinusoidal embedding of the absolute time gap (years) + MLP."""

    def __init__(self, dim, freqs=16, max_years=20.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.freqs = int(freqs)
        self.max_years = float(max_years)

    def build(self, input_shape):
        f = np.arange(self.freqs, dtype=np.float32) / self.freqs
        self.freq_table = self.add_weight(
            name="freq_table", shape=(self.freqs,), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(
                1.0 / (10000.0 ** f)), trainable=False)
        self.d1 = layers.Dense(self.dim, activation="gelu", name="t1")
        self.d2 = layers.Dense(self.dim, name="t2")
        super().build(input_shape)

    def call(self, dt):
        dt = tf.cast(tf.reshape(dt, [-1, 1]), tf.float32) / self.max_years
        ang = dt * self.freq_table * 2.0 * math.pi
        emb = tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)
        return self.d2(self.d1(emb))                               # (B, dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "freqs": self.freqs,
                    "max_years": self.max_years})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class FiLM(layers.Layer):
    """Feature-wise modulation of a feature map by a conditioning vector."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)

    def build(self, input_shape):
        self.dense = layers.Dense(2 * self.channels, name="film")
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        gb = self.dense(cond)                                      # (B, 2C)
        g, b = tf.split(gb, 2, axis=-1)
        g = tf.reshape(g, [-1, 1, 1, self.channels])
        b = tf.reshape(b, [-1, 1, 1, self.channels])
        return x * (1.0 + tf.cast(g, x.dtype)) + tf.cast(b, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg["channels"] = self.channels
        return cfg


# ==============================================================================
# 10. ENCODERS / DECODER BLOCKS
# ==============================================================================
def _heads_for(dim):
    return max(2, min(12, dim // 16))


def make_swin_encoder(prefix, dims, depths, cfg, dp_rates):
    """Returns apply(x) -> (bottleneck, skips). dp_rates consumed in order."""
    ws = cfg.window_size
    if True:  # hybrid CNN stem (more stable than a single strided conv)
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
# 11. FULL MODEL
# ==============================================================================
def build_model(cfg, s1_stats, s2_stats):
    C, ps, A = cfg.num_classes, cfg.patch_size, cfg.aux_channels
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)
    ws = cfg.window_size
    active = set(cfg.active_heads)

    s1_in = layers.Input((ps, ps, 4), name="s1_input")
    s2_in = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1), name="scl_input")
    aux_in = layers.Input((ps, ps, A), name="aux_input")
    s1t2_in = layers.Input((ps, ps, 4), name="s1_t2_input")
    s2t2_in = layers.Input((ps, ps, 10), name="s2_t2_input")
    sclt2_in = layers.Input((ps, ps, 1), name="scl_t2_input")
    auxt2_in = layers.Input((ps, ps, A), name="aux_t2_input")
    dt_in = layers.Input((1,), name="dt_input")

    # shared feature engineering (siamese: ONE layer instance for both dates)
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    f1, f2 = fe([s1_in, s2_in, scl_in, aux_in])
    g1, g2 = fe([s1t2_in, s2t2_in, sclt2_in, auxt2_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]), variance=list(s1_stats[1]),
                              name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]), variance=list(s2_stats[1]),
                              name="s2_norm")
    f1 = n1(f1); f2 = n2(f2)
    g1 = n1(g1); g2 = n2(g2)

    # linear stochastic-depth schedule across all encoder blocks
    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])

    b1, sk1 = enc_s1(f1)                       # SAR T1
    b2, sk2 = enc_s2(f2)                       # OPT T1
    c1, _ = enc_s1(g1)                         # SAR T2 (shared weights)
    c2, sk2_t2 = enc_s2(g2)                    # OPT T2

    # per-level fusion (opt stream fused with sar context; siamese-shared)
    fused_t1, fused_t2 = [], []
    sar_sk_t2 = []
    for i, d in enumerate(dims):
        s_t1, s_t2 = sk1[i], None
        o_t1, o_t2 = sk2[i], sk2_t2[i]
        # project sar to opt dims where needed
        if s_t1.shape[-1] != d:
            s_t1 = layers.Conv2D(d, 1, name=f"sar_proj{i}")(s_t1)
        s_t2 = layers.Conv2D(d, 1, name=f"sar_t2_proj{i}")(sk1[i]) \
            if sk1[i].shape[-1] != d else sk1[i]
        if i in tuple(cfg.cross_attn_levels):
            fuse = CrossAttentionFusion(d, cfg.fusion_heads,
                                        name=f"cross_attn_fusion{i}")
        else:
            fuse = GatedFusion(d, name=f"gated_fusion{i}")
        fused_t1.append(fuse([o_t1, s_t1]))
        fused_t2.append(fuse([o_t2, s_t2]))    # same instance: siamese
        sar_sk_t2.append(s_t2)
    fused_b, fused_b_t2 = fused_t1[-1], fused_t2[-1]

    # ---- main decoder (T1) ----
    x = fused_b
    deep_out = None
    for i, wd in enumerate(cfg.dec_widths):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"dec_up{i}")(x)
        li = len(dims) - 2 - i                 # fuse levels 2,1,0 as skips
        if li >= 0:
            x = layers.concatenate([x, fused_t1[li], sk1[li]], name=f"dec_cat{i}")
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

    # ---- time-aware change decoder: per-level diffs + FiLM(dt) ----
    temb = TimeEmbedding(cfg.time_embed_dim, name="time_embed")(dt_in)
    diffs = []
    for i in range(len(dims)):
        diffs.append(layers.concatenate(
            [AbsDiff(name=f"diff_opt{i}")([fused_t1[i], fused_t2[i]]),
             AbsDiff(name=f"diff_sar{i}")([sk1[i], sar_sk_t2[i]])],
            name=f"diff_cat{i}"))
    d = conv_block(diffs[-1], cfg.change_widths[0], use_se=cfg.use_se,
                   name="change_bottleneck")
    d = FiLM(cfg.change_widths[0], name="film_bottleneck")([d, temb])
    for i, wd in enumerate(cfg.change_widths[1:]):
        d = layers.UpSampling2D(interpolation="bilinear", name=f"change_up{i}")(d)
        li = len(dims) - 2 - i
        if li >= 0:
            d = layers.concatenate([d, diffs[li]], name=f"change_cat{i}")
        d = conv_block(d, wd, use_se=cfg.use_se, name=f"change_c{i}")
        d = FiLM(wd, name=f"film_{i}")([d, temb])

    # ---- heads (pluggable) ----
    outputs = {}
    if "seg" in active:
        outputs["seg_head"] = layers.Activation(
            "softmax", dtype="float32", name="seg_head")(
            layers.Conv2D(C, 1, padding="same", name="seg_logits")(x))
    if cfg.deep_supervision and deep_out is not None:
        outputs["seg_deep_head"] = deep_out
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
    if "change" in active:
        outputs["change_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="change_head")(
            layers.Conv2D(1, 1, padding="same", name="change_logits")(d))
    if "quality" in active:
        outputs["quality_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="quality_head")(
            layers.Conv2D(1, 1, padding="same", name="quality_logits")(x))
    if "boundary" in active:
        outputs["boundary_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="boundary_head")(
            layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x))

    return models.Model(
        inputs=[s1_in, s2_in, scl_in, aux_in,
                s1t2_in, s2t2_in, sclt2_in, auxt2_in, dt_in],
        outputs=outputs, name=f"MangroveEcoMonitor_v{VERSION}")


# ==============================================================================
# 12. LOSSES & METRICS
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=1.0,
                 label_smoothing=0.0, **kwargs):
        super().__init__(name="focal_dice", **kwargs)
        self.alpha = None if alpha is None else [float(a) for a in alpha]
        self.gamma = float(gamma)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        valid = tf.cast(tf.reduce_sum(y_true, -1, keepdims=True) > 0, tf.float32)
        if self.label_smoothing > 0:
            C = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / C
        a = tf.ones_like(y_pred) if self.alpha is None \
            else tf.constant(self.alpha, tf.float32)
        ce = -y_true * tf.math.log(y_pred)
        fl = tf.reduce_sum(a * tf.pow(1.0 - y_pred, self.gamma) * ce,
                           axis=-1, keepdims=True) * valid
        focal = tf.reduce_sum(fl, axis=[1, 2, 3]) / \
            (tf.reduce_sum(valid, axis=[1, 2, 3]) + eps)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
        dice = tf.reduce_mean(1.0 - (2.0 * inter + 1.0) / (denom + 1.0), axis=-1)
        return self.focal_weight * focal + self.dice_weight * dice

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma,
                    "focal_weight": self.focal_weight,
                    "dice_weight": self.dice_weight,
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
class MaskedHuber(tf.keras.losses.Loss):
    """Huber on valid pixels (more robust than MSE for noisy eco/risk labels)."""
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(name="masked_huber", **kwargs)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        err = tf.abs(tf.where(mask > 0, y, 0.0) - p)
        hub = tf.where(err <= self.delta, 0.5 * err ** 2,
                       self.delta * (err - 0.5 * self.delta))
        return tf.reduce_sum(hub * mask) / (tf.reduce_sum(mask) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg["delta"] = self.delta
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedBinaryFocal(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(name="masked_binary_focal", **kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
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
class MaskedBCE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="masked_bce", **kwargs)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
        mask = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        bce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-6)


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.total_cm = self.add_weight(
            name="cm", shape=(self.num_classes, self.num_classes),
            initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.reshape(tf.reduce_sum(y_true, -1) > 0, [-1])
        yt = tf.boolean_mask(
            tf.reshape(tf.argmax(y_true, -1, output_type=tf.int32), [-1]), valid)
        yp = tf.boolean_mask(
            tf.reshape(tf.argmax(y_pred, -1, output_type=tf.int32), [-1]), valid)
        cm = tf.math.confusion_matrix(yt, yp, num_classes=self.num_classes,
                                      dtype=tf.float32)
        self.total_cm.assign_add(cm)

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
        cfg = super().get_config()
        cfg["num_classes"] = self.num_classes
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonMAE(tf.keras.metrics.Metric):
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


@tf.keras.utils.register_keras_serializable(package="mangrove")
class ChangeF1(tf.keras.metrics.Metric):
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
        self.fp.assign_add(tf.reduce_sum((1 - y) * p_bin * mask))
        self.fn.assign_add(tf.reduce_sum(y * (1 - p_bin) * mask))

    def result(self):
        prec = self.tp / (self.tp + self.fp + 1e-6)
        rec = self.tp / (self.tp + self.fn + 1e-6)
        return 2 * prec * rec / (prec + rec + 1e-6)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["threshold"] = self.threshold
        return cfg


# ==============================================================================
# 13. TRAINER: custom train_step, LEARNABLE task weighting, single grad path
# ==============================================================================
HEAD_LOSS_MAP = {
    "seg": "focal_dice", "change": "binary_focal", "quality": "bce",
    "boundary": "bce", "carbon": "carbon_nll", "eco": "huber", "risk": "huber",
}


class EcoMonitorTrainer(tf.keras.Model):
    """Wraps the functional net. Owns:
       - per-task losses + learnable log-variance task weighting (Kendall 2018)
       - explicit metrics per active head
       Gradient accumulation: NATIVE optimizer path only (FIX F4). If
       cfg.grad_accum > 1 we require optimizer gradient_accumulation_steps
       support and never run a second, custom accumulator."""

    def __init__(self, cfg, net, alpha, **kwargs):
        super().__init__(**kwargs)
        self.cfg_obj = cfg
        self.net = net
        self.active = list(cfg.active_heads)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.loss_fns = {}
        for h in self.active:
            kind = HEAD_LOSS_MAP[h]
            if kind == "focal_dice":
                self.loss_fns[h] = FocalDiceLoss(
                    alpha=alpha, gamma=cfg.focal_gamma,
                    label_smoothing=cfg.label_smoothing)
            elif kind == "binary_focal":
                self.loss_fns[h] = MaskedBinaryFocal()
            elif kind == "bce":
                self.loss_fns[h] = MaskedBCE()
            elif kind == "carbon_nll":
                self.loss_fns[h] = CarbonNLLLoss()
            else:
                self.loss_fns[h] = MaskedHuber()
        if cfg.deep_supervision:
            self.loss_fns["seg_deep"] = FocalDiceLoss(
                alpha=alpha, gamma=cfg.focal_gamma,
                label_smoothing=cfg.label_smoothing)

        # learnable per-task log-variances (trainable weights of the wrapper)
        self.use_learned_w = cfg.task_weighting == "learned"
        self.task_log_vars = {}
        for name in list(self.loss_fns.keys()):
            self.task_log_vars[name] = self.add_weight(
                name=f"task_logvar_{name}", shape=(), dtype=tf.float32,
                initializer="zeros", trainable=self.use_learned_w)

        # metrics
        self.per_head_loss = {n: tf.keras.metrics.Mean(name=f"{n}_loss")
                              for n in self.loss_fns}
        self.metric_objs = {}
        if "seg" in self.active:
            self.metric_objs["seg"] = MeanIoU(cfg.num_classes)
        if "change" in self.active:
            self.metric_objs["change"] = ChangeF1()
        if "carbon" in self.active:
            self.metric_objs["carbon"] = CarbonMAE()
        for h in ("quality", "boundary"):
            if h in self.active:
                self.metric_objs[h] = tf.keras.metrics.Mean(name=f"{h}_acc")

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    @property
    def metrics(self):
        ms = [self.loss_tracker] + list(self.per_head_loss.values()) + \
             list(self.metric_objs.values())
        return ms

    def _per_task_losses(self, y, p):
        out = {}
        for name, fn in self.loss_fns.items():
            key = name if name != "seg_deep" else "seg_deep_head"
            tgt = y.get(key)
            prd = p.get(key if key in p else None)
            if tgt is None or prd is None:
                continue
            out[name] = fn(tgt, prd)
        return out

    def _total(self, per_task):
        total = 0.0
        for name, l in per_task.items():
            if self.use_learned_w:
                lv = self.task_log_vars[name]
                w_extra = self.cfg_obj.deep_supervision_weight \
                    if name == "seg_deep" else 1.0
                total = total + w_extra * (l * tf.exp(-lv) + lv)
            else:
                w = self.cfg_obj.fixed_loss_weights.get(
                    name.replace("_deep", ""), 1.0)
                if name == "seg_deep":
                    w = self.cfg_obj.deep_supervision_weight
                total = total + w * l
        return total

    def _update_metrics(self, per_task, y, p, total):
        self.loss_tracker.update_state(total)
        for name, l in per_task.items():
            self.per_head_loss[name].update_state(l)
        if "seg" in self.metric_objs and "seg_head" in p:
            self.metric_objs["seg"].update_state(y["seg_head"], p["seg_head"])
        if "change" in self.metric_objs and "change_head" in p:
            self.metric_objs["change"].update_state(y["change_head"],
                                                    p["change_head"])
        if "carbon" in self.metric_objs and "carbon_head" in p:
            self.metric_objs["carbon"].update_state(y["carbon_head"],
                                                    p["carbon_head"])
        for h in ("quality", "boundary"):
            key = f"{h}_head"
            if h in self.metric_objs and key in p:
                valid = tf.cast(y[key] >= 0, tf.float32)
                acc = tf.cast(tf.equal(tf.cast(p[key] > 0.5, tf.float32),
                                       y[key]), tf.float32)
                denom = tf.reduce_sum(valid)
                val = tf.reduce_sum(acc * valid) / (denom + 1e-6)
                self.metric_objs[h].update_state(val)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            p = self.net(x, training=True)
            per_task = self._per_task_losses(y, p)
            total = self._total(per_task) + sum(self.net.losses)
        # mixed-precision safe scaling (Keras-3 LossScaleOptimizer path)
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
    """Single accumulation mechanism: native optimizer support or hard error."""
    kwargs = dict(learning_rate=lr, weight_decay=cfg.weight_decay,
                  clipnorm=cfg.grad_clip)
    if cfg.grad_accum > 1:
        try:
            return optimizers.AdamW(
                gradient_accumulation_steps=int(cfg.grad_accum), **kwargs)
        except TypeError as e:
            raise RuntimeError(
                "cfg.grad_accum > 1 requires Keras>=3.4 native "
                "gradient_accumulation_steps. Either upgrade TF/Keras or set "
                "grad_accum=1 (v5 does NOT run a second custom accumulator "
                "- see FIX F4).") from e
    return optimizers.AdamW(**kwargs)


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
# 15. LR SCHEDULE / PLOTS / EVAL
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
    C = cfg.num_classes
    cm = np.zeros((C, C), np.int64)
    for inputs, targets in val_ds:
        out = net(inputs, training=False)
        p = _preds_to_dict(net, out)["seg_head"].numpy().argmax(-1)
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
    metrics = {n: {"iou": float(iou[i]), "pixels": int(cm[i].sum())}
               for i, n in enumerate(CLASS_NAMES)}
    metrics["mIoU"] = float(np.nanmean(iou))
    with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return cm, iou


def calibrate_temperature(net, val_ds, cfg, max_batches=10):
    """Post-hoc temperature scaling on validation seg probs (grid search)."""
    if "seg" not in cfg.active_heads:
        return 1.0
    probs, labels = [], []
    for i, (inputs, targets) in enumerate(val_ds):
        if i >= max_batches:
            break
        out = _preds_to_dict(net, net(inputs, training=False))
        p = out["seg_head"].numpy().reshape(-1, cfg.num_classes)
        y = targets["seg_head"].numpy().reshape(-1, cfg.num_classes)
        v = y.sum(-1) > 0
        probs.append(p[v]); labels.append(y[v].argmax(-1))
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
    print(f"  [calibration] seg temperature = {best_t:.2f} "
          f"(NLL {best_nll:.4f})")
    cfg.seg_temperature = float(best_t)
    # persist alongside norm stats
    if os.path.exists(STATS_PATH):
        z = dict(np.load(STATS_PATH))
        z["seg_temperature"] = np.float32(best_t)
        np.savez(STATS_PATH, **z)
    return best_t


def plot_training_history(history):
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
    heads = ["seg_head", "change_head", "quality_head", "boundary_head",
             "carbon_head", "risk_head"]
    have = [h for h in heads if h in [o for o in net.output_names]]
    ncols = 3 + len([h for h in have if h != "seg_head"])
    fig, axes = plt.subplots(n_samples, ncols, figsize=(4 * ncols,
                                                        4 * n_samples))
    axes = np.atleast_2d(axes)
    titles = ["S2 RGB", "GT Seg", "Pred Seg"] + \
             [h.replace("_head", "") for h in have if h != "seg_head"]
    for a, t in zip(axes[0], titles):
        a.set_title(t, fontsize=10)
    count = 0
    for inputs, targets in val_ds.take(2):
        preds = _preds_to_dict(net, net(inputs, training=False))
        preds = {k: v.numpy() for k, v in preds.items()}
        bs = len(targets["seg_head"])
        for i in range(min(bs, n_samples - count)):
            s2 = inputs["s2_input"][i].numpy()
            axes[count, 0].imshow(_stretch(s2[:, :, [2, 1, 0]]))
            y = targets["seg_head"][i].numpy().argmax(-1)
            axes[count, 1].imshow(y, cmap=cmap, vmin=0, vmax=4)
            if "seg_head" in preds:
                axes[count, 2].imshow(preds["seg_head"][i].argmax(-1),
                                      cmap=cmap, vmin=0, vmax=4)
            col = 3
            for h in have:
                if h == "seg_head":
                    continue
                arr = preds[h][i]
                if h == "carbon_head":
                    arr = arr[..., 0] * cfg.carbon_scale
                    axes[count, col].imshow(arr, cmap="YlGn")
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
# 16. SELF-SUPERVISED PRETRAINING (SimMIM-style masked reconstruction)
# ==============================================================================
def build_pretrain_model(cfg, s1_stats, s2_stats):
    """Masked-block reconstruction of S1 feats + S2 bands from the fused
    bottleneck. Encoder layer names match the main model -> by-name transfer."""
    ps, A = cfg.patch_size, cfg.aux_channels
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)

    s1_in = layers.Input((ps, ps, 4), name="s1_input")
    s2_in = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1), name="scl_input")
    aux_in = layers.Input((ps, ps, A), name="aux_input")

    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    f1, f2 = fe([s1_in, s2_in, scl_in, aux_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]),
                              variance=list(s1_stats[1]), name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]),
                              variance=list(s2_stats[1]), name="s2_norm")
    f1n, f2n = n1(f1), n2(f2)

    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])
    b1, _ = enc_s1(f1n)
    b2, _ = enc_s2(f2n)

    if b1.shape[-1] != dims[-1]:
        b1 = layers.Conv2D(dims[-1], 1, name="sar_proj3")(b1)
    fuse = CrossAttentionFusion(dims[-1], cfg.fusion_heads,
                                name="cross_attn_fusion3")
    z = fuse([b2, b1])

    x = z
    for i, wd in enumerate([192, 96]):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"pt_up{i}")(x)
        x = conv_block(x, wd, use_se=False, name=f"pt_c{i}")
    rec = layers.Conv2D(C1 + 10, 1, padding="same", dtype="float32",
                        name="recon_head")(x)
    return models.Model([s1_in, s2_in, scl_in, aux_in], rec,
                        name="MangrovePretrainer_v5"), (f1, f2, f1n, f2n)


def _masked_block(x_raw, ratio, block):
    """Zero random square blocks of an image tensor. Returns (x_masked, keep)."""
    ps = tf.shape(x_raw)[1]
    g = tf.maximum(ps // block, 1)
    cells = tf.random.uniform((tf.shape(x_raw)[0], g, g, 1)) < ratio
    mask = tf.image.resize(tf.cast(cells, tf.float32), [ps, ps],
                           method="nearest")                    # 1 = masked
    keep = 1.0 - mask
    return x_raw * keep, keep


def pretrain(cfg=CFG):
    """SimMIM-style: reconstruct normalized S1 feats + raw S2 bands at 1/4 res
    under heavy block masking. No labels needed."""
    print("\n=== v5 SELF-SUPERVISED PRETRAINING (SimMIM-style) ===")
    files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.npz")))
    assert files, "pretrain: set cfg.train_dir to a folder of NPZ scenes"
    s1m, s1v, s2m, s2v, _ = load_or_compute_stats(files, cfg)

    def gen():
        yield from patch_generator(files, cfg, False, cfg.max_train_patches)

    ps = cfg.patch_size
    sig = output_signature(cfg)

    def to_inputs(*vals):
        s1, s2, scl, aux = vals[0], vals[1], vals[2], vals[3]
        return (s1, s2, scl, aux)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig) \
        .map(lambda *v: to_inputs(*v), num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(cfg.shuffle_buffer) \
        .batch(cfg.batch_per_replica * N_REPLICAS, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    s1_mean = tf.constant(s1m, tf.float32)
    s1_std = tf.sqrt(tf.constant(s1v, tf.float32))

    with STRATEGY.scope():
        pt, refs = build_pretrain_model(cfg, (s1m, s1v), (s2m, s2v))
        f1_raw, f2_raw, f1n, f2n = refs
        opt = optimizers.AdamW(learning_rate=cfg.pretrain_lr,
                               weight_decay=cfg.weight_decay)

        @tf.function
        def step(s1, s2, scl, aux):
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block(s1, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                s2_m, keep2 = _masked_block(s2, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                rec = pt({"s1_input": s1_m, "s2_input": s2_m,
                          "scl_input": scl, "aux_input": aux}, training=True)
                # targets at 1/4 res: normalized s1 feats + raw s2 bands
                t_s1 = (feats_from(s1, scl) - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1, 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2, 4, 4, "VALID")
                m = tf.concat([tf.repeat(m1, C1, -1), tf.repeat(m2, 10, -1)], -1)
                m = tf.cast(m > 0.5, tf.float32)               # 1 = masked cell
                loss = tf.reduce_sum(tf.abs(rec - tgt_q) * m) / \
                    (tf.reduce_sum(m) + 1e-6)
            grads = tape.gradient(loss, pt.trainable_variables)
            opt.apply_gradients([(g, v) for g, v in
                                 zip(grads, pt.trainable_variables)
                                 if g is not None])
            return loss

        # small helper: recompute S1 features OUTSIDE the model for targets
        def feats_from(s1, scl):
            vv, vh, vv_std, vh_std = tf.split(s1, 4, axis=-1)
            lin_vv = tf.pow(10.0, vv / 10.0)
            lin_vh = tf.pow(10.0, vh / 10.0)
            rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
            return tf.concat([vv, vh, vh - vv, rvi, vv_std, vh_std], -1)

        # make feats_from visible to the tf.function closure
        step.__globals__["feats_from"] = feats_from if hasattr(step, "__globals__") else None

        step_fn = step
        # NOTE: nested def ordering - rebuild step with helper bound
        @tf.function
        def train_one(s1, s2, scl, aux):
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block(s1, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                s2_m, keep2 = _masked_block(s2, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                rec = pt({"s1_input": s1_m, "s2_input": s2_m,
                          "scl_input": scl, "aux_input": aux}, training=True)
                vv, vh, vvs, vhs = tf.split(s1, 4, axis=-1)
                lin_vv = tf.pow(10.0, vv / 10.0)
                lin_vh = tf.pow(10.0, vh / 10.0)
                rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
                t_s1 = tf.concat([vv, vh, vh - vv, rvi, vvs, vhs], -1)
                t_s1 = (t_s1 - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1, 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2, 4, 4, "VALID")
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
            for s1, s2, scl, aux in ds:
                losses.append(float(train_one(s1, s2, scl, aux)))
            print(f"  pretrain {ep + 1}/{cfg.pretrain_epochs} "
                  f"loss={np.mean(losses):.4f} ({time.time() - t0:.0f}s)")

        out = os.path.join(WORK_DIR, "pretrain_encoder_v5.weights.h5")
        pt.save_weights(out)
        print(f"Pretrained encoder weights saved: {out}")
        print("Fine-tune with: cfg.pretrain_weights = "
              f"'{out}' before train()")
        return out


# ==============================================================================
# 17. BATCH PREFLIGHT + OOM-RESILIENT TRAINING
# ==============================================================================
def preflight_batch(trainer, ds):
    """One train_on_batch to verify memory; returns True/False."""
    try:
        x, y = next(iter(ds))
        trainer.train_on_batch(x, y)
        return True
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        print(f"  [preflight] OOM at current batch: {str(e)[:160]}")
        return False


def train(cfg=CFG):
    print(f"\n=== MANGROVE ECO-MONITOR v{VERSION} - TRAINING ===\n")
    train_files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.npz")))
    val_files = sorted(glob.glob(os.path.join(cfg.val_dir, "*.npz")))
    assert train_files and val_files, "Set cfg.train_dir / cfg.val_dir"
    validate_heads_vs_data(train_files, cfg)
    print(f"train scenes: {len(train_files)} | val scenes: {len(val_files)}")

    s1m, s1v, s2m, s2v, alpha = load_or_compute_stats(train_files, cfg)

    if cfg.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    attempt, oom_left = 0, cfg.oom_retries
    while True:
        batch = cfg.batch_per_replica * N_REPLICAS
        ds_train = make_dataset(
            lambda: patch_generator(train_files, cfg, True,
                                    cfg.max_train_patches), cfg, True, batch)
        ds_val = make_dataset(
            lambda: patch_generator(val_files, cfg, False,
                                    cfg.max_val_patches), cfg, False, batch)

        with STRATEGY.scope():
            net = build_model(cfg, (s1m, s1v), (s2m, s2v))
            if cfg.pretrain_weights and os.path.exists(cfg.pretrain_weights):
                try:
                    net.load_weights(cfg.pretrain_weights, by_name=True,
                                     skip_mismatch=True)
                    print(f"  [pretrain] loaded encoder weights "
                          f"(by_name, skip_mismatch) from {cfg.pretrain_weights}")
                except Exception as e:
                    print(f"  [pretrain] weight load failed ({e}); training "
                          f"from scratch.")
            trainer = EcoMonitorTrainer(cfg, net, alpha)
            est_steps = max(1, cfg.max_train_patches // batch)
            lr_sched = WarmUpCosineDecay(
                cfg.lr, cfg.epochs * est_steps,
                cfg.lr_warmup_epochs * est_steps)
            trainer.compile(optimizer=make_optimizer(cfg, lr_sched))

        if cfg.auto_batch and attempt == 0:
            ok = preflight_batch(trainer, ds_train)
            while not ok and cfg.batch_per_replica > 1:
                cfg.batch_per_replica //= 2
                tf.keras.backend.clear_session()
                gc.collect()
                batch = cfg.batch_per_replica * N_REPLICAS
                ds_train = make_dataset(
                    lambda: patch_generator(train_files, cfg, True,
                                            cfg.max_train_patches),
                    cfg, True, batch)
                with STRATEGY.scope():
                    net = build_model(cfg, (s1m, s1v), (s2m, s2v))
                    trainer = EcoMonitorTrainer(cfg, net, alpha)
                    trainer.compile(optimizer=make_optimizer(cfg, lr_sched))
                ok = preflight_batch(trainer, ds_train)
            print(f"  [auto_batch] settled on batch/replica="
                  f"{cfg.batch_per_replica}")

        monitor = "val_seg_mean_iou" if "seg" in cfg.active_heads else "val_loss"
        mode = "max" if "iou" in monitor else "min"
        ema = EMACallback(cfg.ema_decay)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=15,
                                             restore_best_weights=True,
                                             verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(WORK_DIR, "best_v5.weights.h5"), monitor,
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
        except (tf.errors.ResourceExhaustedError,) as e:
            if oom_left <= 0 or cfg.batch_per_replica <= 1:
                raise
            oom_left -= 1"""
================================================================================
 MANGROVE ECO-MONITOR v5.0  "BLUE SENTINEL"
 Production-grade dual-modality (Sentinel-1 SAR x Sentinel-2 optical) mangrove
 monitoring system: Swin-V2 Siamese U-Net, per-level gated fusion, time-aware
 change detection, pluggable multi-task heads, uncertainty-aware MRV outputs.

 REQUIRES: Python >= 3.10, TensorFlow >= 2.16 (Keras 3), Linux or WSL2 on
           Windows (native Windows has NO TensorFlow GPU support since TF 2.10).
           Optional: rasterio (COG/GeoJSON export), scikit-learn (eval fallback
           exists), matplotlib (plots).

--------------------------------------------------------------------------------
 WHAT V5 FIXES (all confirmed defects from the v4.1 review + code audit)
--------------------------------------------------------------------------------
 [F1] Keras-3 fatal crash: tf.constant() buffers created in build() go out of
      scope under re-tracing. v5 stores EVERY buffer (relative coords, shift
      masks) as non-trainable weights via add_weight(Constant). The Swin-V2
      CPB design removes the giant RPB table entirely.
 [F2] Normalization off-by-one: cloud-mask identity stats were written to
      channel 18 (elevation) instead of 17 (mask). v5 computes the mask index
      symbolically from the feature layout; aux channels now normalized with
      their real stats.
 [F3] Hardware detection: "NVIDIA RTX 6000 Ada Generation" matched nothing and
      silently fell back to the 16 GB T4 preset. v5 adds explicit RTX-6000-Ada
      matching PLUS a VRAM-probing tier fallback via nvidia-smi.
 [F4] Double gradient accumulation (native AdamW kwarg + custom train_step
      wrapper fighting each other). v5 uses ONE mechanism: the native Keras
      optimizer path, with a version guard and a hard error if unavailable.
 [F5] Four of five heads had no label source. v5 makes heads PLUGGABLE:
      seg/change/quality/boundary ship ON (all have real labels), carbon/eco/
      risk ship OFF with documented data sources; a pre-flight scan validates
      enabled heads against the NPZ keys actually present.
 [F6] Windows GPU: v5 detects Windows + CUDA-visible-but-TF-invisible GPUs and
      prints exact WSL2 remediation steps.

 NEW BUGS FOUND BY V5 AUDIT (were NOT in the review):
 [F7] Augmentation channel misalignment: random_aug sliced x[..., 14:18] /
      18:22 / 22:32 assuming aux=4, but aux=6 -> brightness/contrast was applied
      to SCL+AUX and S1-T2 bands; everything downstream was corrupted. v5 uses
      cumulative symbolic offsets derived from SIZES.
 [F8] Gaussian noise was added to the WHOLE packed tensor including one-hot seg
      labels, carbon/eco/change/risk targets -> destroyed label integrity.
      v5 restricts noise to input bands only.
 [F9] TTA blending bug: predictions were accumulated once PER TTA variant but
      the weight map only ONCE per tile -> all outputs inflated ~4x with TTA on.
      v5 accumulates weights per variant pass.
 [F10] Inverted blending kernel: v4's "gaussian" peaked at tile EDGES and was
      ~0.01 at the centre. v5 uses a proper centre-peaked Hann window.

--------------------------------------------------------------------------------
 V5 MAJOR UPGRADES (grounded in 2025-2026 research)
--------------------------------------------------------------------------------
 U1  Swin-V2 blocks: res-post-norm + scaled-cosine attention with learnable
     per-head logit scale (clamped) + log-spaced Continuous Position Bias MLP.
     More stable at depth, transferable across window sizes.
 U2  Per-level fusion: cheap gated 1x1 fusion at high-res levels, bidirectional
     cross-attention at deep levels (v4 fused ONLY at the bottleneck).
 U3  Time-aware Siamese change decoder: per-level feature diffs as skips +
     FiLM conditioning on real acquisition-date gap (decimal years). Enables
     backcasting to arbitrary dates and GMW-style annual loss products.
 U4  Pluggable task heads with LEARNABLE uncertainty weighting (Kendall et
     al.): loss weights are learned log-variances, not hand-tuned constants.
 U5  Free-supervision heads: quality head (pixel usability from SCL, trained
     WITHOUT extra labels) and boundary head (morphological edge target derived
     from seg labels) -> sharp coastlines + reliability masks at inference.
 U6  Deep supervision (aux seg logits at 1/4 res) with small weight.
 U7  SimMIM-style self-supervised pretraining mode for unlabeled S1+S2 scenes
     (masked-block reconstruction), then fine-tune with by-name weight load.
     This is the standard answer to EO label scarcity (MAE/SimMIM/SatMAE line).
 U8  Modality dropout: randomly zero the SAR or optical branch during training
     so the model stays operational when S2 is cloudy (ESA RAMM shows SAR-only
     alerting is a hard operational requirement in the tropics). predict_scene
     supports sar_only=True.
 U9  Failure tolerance: NaN/Inf NPZ sanitization with counts, schema validation
     with actionable errors, OOM auto-retry with halved batch (resumes from
     last checkpoint), memory preflight probe, TerminateOnNaN, BackupAndRestore.
 U10 EMA of weights (Polyak averaging) - evaluation, checkpointing and export
     all run on EMA weights; separate EMA checkpoint saved.
 U11 Post-hoc temperature calibration of seg probabilities on the val set;
     calibrated probabilities used at inference and stored in norm stats.
 U12 MRV-ready export pack: COG GeoTIFFs + STAC 1.0 item JSON + change-alert
     GeoJSON polygons (area, period, confidence) + summary CSV with class areas
     (ha) and carbon stock with 95% CI in tCO2e (VM0033-aligned reporting
     primitives) + preview PNG.
 U13 Class-balanced patch sampling: rare classes (perm flood, degraded
     mangrove) oversampled via per-patch weighting.
 U14 Hardware: RTX-6000-Ada preset, generic VRAM tiers, optional batch
     auto-tuner that empirically probes the largest safe batch.
 U15 NISAR/L-band readiness: aux channel count is config-driven; drop NISAR
     L-band HV/HH ratio or ESA BIOMASS P-band features into aux and everything
     (stats, stems, aug) adapts. aux_names/scale are configurable.

 NPZ SCHEMA v5 (backward-compatible superset of v4.1)
   required: s1 (H,W,2|4) dB | s2 (H,W,10) reflectance
   optional: s1_std, scl, aux (H,W,A), label (0..4, 255=ignore),
             carbon (t/ha, -999), eco (H,W,2, -999), change (0/1, 255),
             risk (0..1, -999), s1_t2/s2_t2/scl_t2/aux_t2,
   NEW:      date_t1, date_t2 : decimal year scalars (e.g. 2024.42) ->
             drives the time-aware change decoder. If a real T2 exists but
             dates are absent, dt defaults to 1.0.

 HEAD DATA SOURCES (honest wiring)
   seg     : WorldCover + GSW + GMW v4.x derived labels            [ON ]
   change  : Hansen GFC v1.13 lossyear / GMW alerts / T2 pairs     [ON ]
   quality : derived from SCL at train time (no NPZ field needed)  [ON ]
   boundary: derived from seg labels at train time                 [ON ]
   carbon  : ESA CCI Biomass / GEDI / NISAR L-band products        [OFF]
   eco     : Landsat/MODIS LST (soil moisture NOT viable @10m)     [OFF]
   risk    : hydrodynamic surge model or heuristic (weak labels)   [OFF]

 MODES:  python mangrove_v5.py smoke | pretrain | train | eval | infer | export
================================================================================
"""

import os
import gc
import sys
import json
import glob
import math
import time
import shutil
import argparse
import platform
import subprocess
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "5.0.0"
MIN_TF = (2, 16)


# ==============================================================================
# 1. ENVIRONMENT, GPU, PLATFORM GUARDS
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


def environment_check():
    """Fail fast with actionable messages. Addresses review F1/F6 context."""
    ok = True
    if _parse_version(tf.__version__) < MIN_TF:
        print(f"[FATAL] TensorFlow {tf.__version__} < {'.'.join(map(str, MIN_TF))}. "
              f"v5 requires Keras 3 APIs. pip install -U tensorflow")
        ok = False

    gpus = tf.config.list_physical_devices("GPU")
    if platform.system() == "Windows" and not gpus:
        smi = shutil.which("nvidia-smi")
        if smi is not None:
            print("=" * 78)
Web search


            print("[HARDWARE BLOCKER] Windows detected, NVIDIA driver present, but"
                  " TensorFlow sees NO GPU.")
            print("TensorFlow dropped native-Windows GPU support after v2.10. Fix:")
            print("  1) wsl --install -d Ubuntu   (PowerShell, admin)")
            print("  2) Install the Windows NVIDIA driver (it serves WSL2 too; do NOT")
            print("     install a Linux driver inside WSL2)")
            print("  3) Inside Ubuntu: python -m pip install tensorflow[and-cuda]")
            print("  4) Verify: python -c \"import tensorflow as tf; "
                  "print(tf.config.list_physical_devices('GPU'))\"")
            print("  WSL2 gives near-native CUDA performance for this workload.")
            print("=" * 78)
            ok = False
    return ok, gpus


ENV_OK, GPUS = environment_check()
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v5/"
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
    """Best-effort VRAM probe via nvidia-smi (works on Linux/WSL2)."""
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
    train_dir: str = "/kaggle/input/mangrove-v4-npz/train/"
    val_dir: str = "/kaggle/input/mangrove-v4-npz/val/"
    test_dir: str = "/kaggle/input/mangrove-v4-npz/test/"
    patch_size: int = 256                 # -> maps 64/32/16/8, window 8 divides all
    train_stride: int = 128
    max_train_patches: int = 8000
    max_val_patches: int = 1200
    min_valid_frac: float = 0.35
    max_cloud_frac: float = 0.85
    ignore_label: int = 255
    num_classes: int = 5
    invalid_scl: tuple = (0, 1, 2, 3, 7, 8, 9, 10, 11)
    cloud_dilate: int = 5
    # aux channels are config-driven (NISAR-ready: add L-band features here)
    aux_names: tuple = ("elev", "slope", "ndvi_trend", "flood_freq",
                        "lst_trend", "s1_trend")
    aux_scale: tuple = (50.0, 20.0, 0.1, 1.0, 50.0, 1.0)
    # class-balanced sampling
    rare_classes: tuple = (2, 3)          # perm flood, degraded mangrove
    rare_boost: float = 4.0
    # ---- model ----
    embed_dims: tuple = (48, 96, 192, 384)
    depths: tuple = (2, 2, 2, 2)
    window_size: int = 8
    drop_path: float = 0.15               # linear stochastic-depth schedule 0 -> dp
    mlp_ratio: float = 4.0
    fusion_heads: int = 8
    cross_attn_levels: tuple = (2, 3)     # full cross-attn only at deep levels
    dec_widths: tuple = (192, 96, 48)
    refine_widths: tuple = (32, 24)
    change_widths: tuple = (128, 96, 64, 48, 32)
    decoder_swin_levels: int = 2
    use_se: bool = True
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.3
    time_embed_dim: int = 64
    active_heads: tuple = ("seg", "change", "quality", "boundary")
    # heads OFF by default; enable only after wiring real labels (see header)
    # e.g. active_heads = ("seg","change","quality","boundary","carbon")
    # ---- training ----
    hardware: str = "auto"
    batch_per_replica: int = 4
    grad_accum: int = 1                   # 1 = native single-step (recommended)
    epochs: int = 120
    lr: float = 2e-4
    lr_warmup_epochs: int = 3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    task_weighting: str = "learned"       # "learned" (Kendall) or "fixed"
    fixed_loss_weights: dict = field(default_factory=lambda: {
        "seg": 1.0, "change": 1.2, "quality": 0.2, "boundary": 0.3,
        "carbon": 0.6, "eco": 0.25, "risk": 0.25})
    carbon_scale: float = 100.0
    eco_scale: tuple = (50.0, 1.0)
Web search


    ema_decay: float = 0.999
    grad_clip: float = 1.0
    deterministic: bool = False
    auto_batch: bool = False              # empirical preflight probe
    oom_retries: int = 2                  # halve batch and resume on OOM
    pretrain_weights: str = ""            # path to SimMIM encoder weights
    # ---- augmentation ----
    augment: bool = True
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_noise: float = 0.02
    modality_dropout_p: float = 0.08      # per-modality, same choice for T1/T2
    # ---- pretraining ----
    pretrain_epochs: int = 40
    pretrain_lr: float = 3e-4
    pretrain_mask_ratio: float = 0.5
    pretrain_block: int = 32
    # ---- inference ----
    infer_batch: int = 8
    infer_tta: bool = True
    infer_overlap: float = 0.5
    mc_samples: int = 0                   # >1 enables MC-dropout epistemic maps
    seg_temperature: float = 1.0          # filled by calibration
    change_threshold: float = 0.5
    # ---- MRV ----
    pixel_size_m: float = 10.0            # fallback if ref tif unavailable
    carbon_to_co2e: float = 44.0 / 12.0   # tC -> tCO2e
    c_fraction: float = 0.5               # biomass -> carbon
    shuffle_buffer: int = 256

    @property
    def aux_channels(self):
        return len(self.aux_names)

    def save(self, path):
        d = asdict(self)
        d["invalid_scl"] = list(self.invalid_scl)
        d["aux_names"] = list(self.aux_names)
        d["aux_scale"] = list(self.aux_scale)
        d["rare_classes"] = list(self.rare_classes)
        d["embed_dims"] = list(self.embed_dims)
        d["depths"] = list(self.depths)
        d["cross_attn_levels"] = list(self.cross_attn_levels)
        d["dec_widths"] = list(self.dec_widths)
        d["refine_widths"] = list(self.refine_widths)
        d["change_widths"] = list(self.change_widths)
        d["active_heads"] = list(self.active_heads)
        d["eco_scale"] = list(self.eco_scale)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        for k in ("invalid_scl", "aux_names", "aux_scale", "rare_classes",
                  "embed_dims", "depths", "cross_attn_levels", "dec_widths",
                  "refine_widths", "change_widths", "active_heads", "eco_scale"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)


CFG = Config()

HW_PRESETS = {
    # name: (batch_per_replica, grad_accum, infer_batch)
    "h100":        (32, 1, 32),
    "a100":        (24, 1, 32),
    "rtx6000ada":  (16, 1, 16),   # 48 GB Ada workstation card (review F3)
    "a6000":       (16, 1, 16),
    "l40":         (12, 1, 16),
    "4090":        (8, 1, 8),
    "l4":          (6, 1, 8),
    "t4x2":        (4, 2, 8),
    "cpu":         (2, 1, 2),
}


def resolve_hardware(cfg):
    """Explicit name match first, then VRAM tiers, then conservative default."""
    name = " ".join(GPU_NAMES)
    hw = cfg.hardware
    if hw == "auto":
        n = name.lower()
        if "h100" in n:
            hw = "h100"
        elif "a100" in n:
            hw = "a100"
        elif "6000 ada" in n or "rtx 6000" in n:     # RTX 6000 Ada (48 GB)
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
# 3. NPZ LOADING WITH VALIDATION + SANITIZATION (failure tolerance)
# ==============================================================================
class NPZSchemaError(ValueError):
    pass


def _process_s1(s1):
    s1 = s1.astype(np.float32)
    if np.nanmin(s1) >= 0:                                # linear -> dB
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
    """NaN/Inf guard with counts (F9-tolerance). Returns fixed copy."""
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / arr.size
        if frac > 0.30:
            raise NPZSchemaError(f"{path}: '{name}' is {frac:.0%} non-finite")
        print(f"  [sanitize] {os.path.basename(path)}:{name} "
              f"had {n_bad} non-finite px -> 0")
        arr = np.where(bad, 0.0, arr)
    return arr


def load_npz(path, cfg=CFG):
    """Returns 15 arrays (13 v4.1 fields + date_t1 + date_t2).
    Missing -> sentinel-filled; NaN/Inf sanitized with counts."""
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
    with np.load(path) as d:
        files = d.files
        if "s1" not in files or "s2" not in files:
            raise NPZSchemaError(f"{path}: missing required keys 's1'/'s2' "
                                 f"(found: {sorted(files)})")
        s1 = _sanitize(_process_s1(d["s1"]), "s1", path)
        s2 = _sanitize(_process_s2(d["s2"]), "s2", path)
        if s1.ndim != 3 or s2.ndim != 3 or s2.shape[-1] != 10:
            raise NPZSchemaError(f"{path}: bad shapes s1{s1.shape} s2{s2.shape}")
        H, W = s2.shape[:2]
        if s1.shape[:2] != (H, W):
            raise NPZSchemaError(f"{path}: s1/s2 spatial mismatch "
                                 f"{s1.shape[:2]} vs {(H, W)}")

        s1_std = (d["s1_std"].astype(np.float32)
                  if "s1_std" in files else np.zeros((H, W, 2), np.float32))
        s1 = np.concatenate([s1, _sanitize(s1_std, "s1_std", path)], -1)
        scl = (d["scl"].astype(np.float32)[..., None]
               if "scl" in files else np.full((H, W, 1), 4.0, np.float32))
        aux = (d["aux"].astype(np.float32)
               if "aux" in files else np.zeros((H, W, A), np.float32))
        if aux.shape[-1] != A:
            fixed = np.zeros((H, W, A), np.float32)
            n = min(A, aux.shape[-1])
            fixed[..., :n] = aux[..., :n]
            aux = fixed
        aux = _sanitize(aux, "aux", path) / scale

        label = (d["label"].astype(np.int32)
                 if "label" in files else np.full((H, W), 255, np.int32))
        carbon = (d["carbon"].astype(np.float32)
                  if "carbon" in files else np.full((H, W), -999.0, np.float32))
        eco = (d["eco"].astype(np.float32)
               if "eco" in files else np.full((H, W, 2), -999.0, np.float32))
        change = (d["change"].astype(np.int32)
                  if "change" in files else np.full((H, W), 255, np.int32))
        risk = (d["risk"].astype(np.float32)
                if "risk" in files else np.full((H, W), -999.0, np.float32))

        has_t2 = "s1_t2" in files and "s2_t2" in files
        if has_t2:
            s1t2 = _sanitize(_process_s1(d["s1_t2"]), "s1_t2", path)
            s1t2_std = (d["s1_t2_std"].astype(np.float32)
                        if "s1_t2_std" in files else np.zeros_like(s1t2))
            s1t2 = np.concatenate([s1t2, s1t2_std], -1)
            s2t2 = _sanitize(_process_s2(d["s2_t2"]), "s2_t2", path)
            sclt2 = (d["scl_t2"].astype(np.float32)[..., None]
                     if "scl_t2" in files else np.full((H, W, 1), 4.0, np.float32))
            auxt2 = ((d["aux_t2"].astype(np.float32)
                      if "aux_t2" in files else np.zeros((H, W, A), np.float32))
                     / scale)
        else:
            s1t2, s2t2 = s1.copy(), s2.copy()
            sclt2, auxt2 = scl.copy(), aux.copy()
            change = np.full((H, W), 255, np.int32)   # never learn from copy

        date_t1 = float(d["date_t1"]) if "date_t1" in files else 0.0
        date_t2 = float(d["date_t2"]) if "date_t2" in files else (1.0 if has_t2 else 0.0)

    return (s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
            label, carbon, eco, change, risk,
            np.float32(date_t1), np.float32(date_t2))


HEAD_REQUIRED_KEYS = {
    "seg": ["label"], "change": ["change"], "carbon": ["carbon"],
    "eco": ["eco"], "risk": ["risk"],
    "quality": [], "boundary": [],      # derived on-the-fly: always available
}


def validate_heads_vs_data(files, cfg, sample=5):
    """Pre-flight: ensure every enabled head actually has labels in the NPZs."""
    present = set()
    for p in files[:sample]:
        with np.load(p) as d:
            present |= set(d.files)
    active, warnings_out = [], []
    for h in cfg.active_heads:
        need = HEAD_REQUIRED_KEYS.get(h, [])
        if need and not any(k in present for k in need):
            warnings_out.append(
                f"  [heads] '{h}' enabled but no '{need[0]}' key found in sampled "
                f"NPZs -> head DISABLED (wire real labels, then re-enable).")
        else:
            active.append(h)
    if not active:
        raise NPZSchemaError("No task head has data; check your NPZ schema.")
    for w in warnings_out:
        print(w)
    if tuple(active) != tuple(cfg.active_heads):
        cfg.active_heads = tuple(active)
        print(f"  [heads] active set resolved to: {list(active)}")
    return cfg


# ==============================================================================
# 4. PATCH GENERATOR (class-balanced for training)
# ==============================================================================
def _patch_weight(lab, cfg):
    """Higher sampling weight when rare classes are present."""
    C = cfg.num_classes
    valid = lab != cfg.ignore_label
    if valid.sum() == 0:
        return 0.0
    counts = np.bincount(lab[valid].ravel(), minlength=C).astype(np.float32)
    frac = counts / max(counts.sum(), 1)
    w = 1.0 + cfg.rare_boost * float(frac[list(cfg.rare_classes)].sum())
    return w


def patch_generator(file_list, cfg, training, max_patches, seed=42):
    ps = cfg.patch_size
    stride = cfg.train_stride if training else ps
    ignore = cfg.ignore_label
    invalid = np.array(list(cfg.invalid_scl))
    rng = np.random.default_rng(seed)

    if training:
        # pass 1: collect candidate coords + weights (labels only - cheap)
        pool = []   # (path, i, j, weight)
        for path in file_list:
            data = load_npz(path, cfg)
            label, scl = data[8], data[2]
            h, w = label.shape
            if h < ps or w < ps:
                continue
            for i in range(0, h - ps + 1, stride):
                for j in range(0, w - ps + 1, stride):
                    lab = label[i:i + ps, j:j + ps]
                    if float(np.mean(lab != ignore)) < cfg.min_valid_frac:
                        continue
                    if float(np.mean(np.isin(scl[i:i + ps, j:j + ps], invalid))) \
                            > cfg.max_cloud_frac:
                        continue
                    wgt = _patch_weight(lab, cfg)
                    if wgt > 0:
                        pool.append((path, i, j, wgt))
            del data
        gc.collect()
        if not pool:
            return
        wts = np.array([p[3] for p in pool], np.float64)
        wts /= wts.sum()
        idx = rng.choice(len(pool), size=min(max_patches, len(pool)),
                         replace=False, p=wts)
        chosen = {}
        for k in idx:
            chosen.setdefault(pool[int(k)][0], []).append(pool[int(k)])
        # pass 2: load each scene once, emit chosen patches
        emitted = 0
        for path, items in chosen.items():
            data = load_npz(path, cfg)
            for (_, i, j, _w) in items:
                if emitted >= max_patches:
                    return
                sl = np.s_[i:i + ps, j:j + ps]
                yield tuple(arr[sl] if arr.ndim >= 2 else arr for arr in data)
                emitted += 1
            del data
            gc.collect()
    else:
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
                    lab = label[i:i + ps, j:j + ps]
                    if float(np.mean(lab != ignore)) < cfg.min_valid_frac:
                        continue
                    sl = np.s_[i:i + ps, j:j + ps]
                    yield tuple(arr[sl] if arr.ndim >= 2 else arr for arr in data)
                    emitted += 1
            del data
        gc.collect()


# ==============================================================================
# 5. FEATURE ENGINEERING LAYER
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FeatureEngineer(layers.Layer):
    """S1: [vv, vh, vh-vv, rvi, vv_std, vh_std] (6)
       S2: [10 raw bands, ndvi, ndwi, mndwi, ndmi, ndre, bsi, evi,
            cloud_mask, aux(A)] -> 18 + A channels.
       MASK channel index is SYMBOLIC: S2_MASK_IDX = 17."""

    S1_OUT = 6
    S2_IDX_COUNT = 7
    S2_MASK_IDX = 10 + S2_IDX_COUNT          # = 17  (review F2 fix anchor)
    S2_BASE = 10 + S2_IDX_COUNT + 1          # = 18, before aux

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
C2 = lambda cfg: FeatureEngineer.S2_BASE + cfg.aux_channels   # 18 + A


# ==============================================================================
# 6. NORMALIZATION STATS + CLASS ALPHA (F2 fixed: symbolic mask index)
# ==============================================================================
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v5.npz")


def compute_stats(file_list, cfg, max_patches=400):
    print("Computing normalization stats + class frequencies...")
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, cfg.aux_channels)
    C = cfg.num_classes
    c1, c2 = C1, C2(cfg)
    n = 0
    s1_sum = np.zeros(c1); s1_sq = np.zeros(c1)
    s2_sum = np.zeros(c2); s2_sq = np.zeros(c2)
    counts = np.zeros(C, np.int64)
    for data in patch_generator(file_list, cfg, training=False,
                                max_patches=max_patches):
        s1, s2, scl, aux = data[0], data[1], data[2], data[3]
        lab = data[8]
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
    # FIX F2: identity stats for the cloud-mask channel at its TRUE index.
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


def load_or_compute_stats(file_list, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        print("Loaded cached norm stats.")
        cfg.seg_temperature = float(z["seg_temperature"]) \
            if "seg_temperature" in z.files else 1.0
        return z["s1_mean"], z["s1_var"], z["s2_mean"], z["s2_var"], z["alpha"]
    return compute_stats(file_list, cfg)


# ==============================================================================
# 7. tf.data PIPELINE  (pack -> augment -> unpack), symbolic channel offsets
# ==============================================================================
def output_signature(cfg):
    ps, A = cfg.patch_size, cfg.aux_channels
    F = tf.float32
    return (
        tf.TensorSpec((ps, ps, 4),  F), tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F), tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps, 4),  F), tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F), tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps),     tf.int32), tf.TensorSpec((ps, ps), F),
        tf.TensorSpec((ps, ps, 2),  F), tf.TensorSpec((ps, ps), tf.int32),
        tf.TensorSpec((ps, ps),     F),
        tf.TensorSpec((),           F), tf.TensorSpec((),  F),   # date_t1/t2
    )


# symbolic offsets into the packed tensor (FIX F7: no more hardcoded slices)
def _packed_layout(cfg):
    A, C = cfg.aux_channels, cfg.num_classes
    names = ["s1", "s2", "scl", "aux", "s1t2", "s2t2", "sclt2", "auxt2",
             "seg", "carbon", "eco", "change", "risk", "dt"]
    sizes = [4, 10, 1, A, 4, 10, 1, A, C, 1, 2, 1, 1, 1]
    off, out = 0, {}
    for n, s in zip(names, sizes):
        out[n] = (off, off + s)
        off += s
    return out, off


def make_dataset(gen_fn, cfg, training, batch_size):
    C, A = cfg.num_classes, cfg.aux_channels
    LAY, _ = _packed_layout(cfg)
    carbon_scale = float(cfg.carbon_scale)
    eco_scale = tf.constant(list(cfg.eco_scale), tf.float32)
    invalid_scl = tf.constant(list(cfg.invalid_scl), tf.int32)
    deep = bool(cfg.deep_supervision)

    def pack(s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
             label, carbon, eco, change, risk, dt1, dt2):
        seg = tf.one_hot(tf.where(label == cfg.ignore_label, -1, label),
                         C, dtype=tf.float32)
        change_f = tf.cast(change, tf.float32)
        change_f = tf.where(change_f == 255.0, -1.0, change_f)
        dt = tf.reshape(tf.abs(dt2 - dt1), (1,))
        return tf.concat([s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
                          seg, carbon[..., None], eco,
                          change_f[..., None], risk[..., None], dt], axis=-1)

    def random_aug(x):
        # geometric (applied to EVERYTHING incl. targets: spatially consistent)
        do_lr = tf.random.uniform(()) > 0.5
        x = tf.cond(do_lr, lambda: tf.image.flip_left_right(x), lambda: x)
        do_ud = tf.random.uniform(()) > 0.5
        x = tf.cond(do_ud, lambda: tf.image.flip_up_down(x), lambda: x)
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        x = tf.image.rot90(x, k)

        if training and cfg.augment:
            # FIX F7: photometric ops hit OPTICAL bands only, via symbolic offsets
            delta = tf.random.uniform((), -cfg.aug_brightness, cfg.aug_brightness)
            contra = tf.random.uniform((), 1 - cfg.aug_contrast, 1 + cfg.aug_contrast)

            def photo(t, lo, hi):
                pre, mid, post = t[..., :lo], t[..., lo:hi], t[..., hi:]
                mid = tf.image.adjust_contrast(tf.image.adjust_brightness(mid, delta), contra)
                return tf.concat([pre, mid, post], axis=-1)

            lo1, hi1 = LAY["s2"]
            lo2, hi2 = LAY["s2t2"]
            x = photo(x, lo1, hi1) if hi1 < lo2 else x
            # optical T2 sits later in the tensor; adjust with a rebuild
            if hi1 < lo2:
                pre = x[..., :lo2]
                mid = x[..., lo2:hi2]
                post = x[..., hi2:]
                mid = tf.image.adjust_contrast(
                    tf.image.adjust_brightness(mid, delta), contra)
                x = tf.concat([pre, mid, post], axis=-1)

            # FIX F8: noise on INPUT bands only (never on packed targets)
            n_inputs_end = LAY["auxt2"][1]
            noise = tf.random.normal(tf.shape(x[..., :n_inputs_end]),
                                     stddev=cfg.aug_noise)
            x = tf.concat([x[..., :n_inputs_end] + noise, x[..., n_inputs_end:]],
                          axis=-1)

            # modality dropout: same modality choice for T1 and T2
            p = cfg.modality_dropout_p
            if p > 0:
                drop_opt = tf.random.uniform(()) < p
                drop_sar = tf.logical_and(tf.random.uniform(()) < p,
                                          tf.logical_not(drop_opt))
                one = tf.ones((), tf.float32)
                zero = tf.zeros((), tf.float32)
                g_opt = tf.where(drop_opt, zero, one)     # zeroes S2 bands T1+T2
                g_sar = tf.where(drop_sar, zero, one)     # zeroes S1 bands T1+T2
                s1a, s1b = LAY["s1"]; s2a, s2b = LAY["s2"]
                t1a, t1b = LAY["s1t2"]; t2a, t2b = LAY["s2t2"]
                parts = [
                    x[..., :s1a],
                    x[..., s1a:s1b] * g_sar,
                    x[..., s1b:s2a],
                    x[..., s2a:s2b] * g_opt,
                    x[..., s2b:t1a],
                    x[..., t1a:t1b] * g_sar,
                    x[..., t1b:t2a],
                    x[..., t2a:t2b] * g_opt,
                    x[..., t2b:],
                ]
                x = tf.concat(parts, axis=-1)
        return x

    def unpack_to_xy(x):
        g = lambda n: x[..., LAY[n][0]:LAY[n][1]]
        s1, s2, scl, aux = g("s1"), g("s2"), g("scl"), g("aux")
        s1t2, s2t2, sclt2, auxt2 = g("s1t2"), g("s2t2"), g("sclt2"), g("auxt2")
        seg, carbon, eco, change, risk = g("seg"), g("carbon"), g("eco"), \
            g("change"), g("risk")
        dt = g("dt")

        carbon_y = tf.where(carbon > -900.0, carbon / carbon_scale, -999.0)
        eco_y = tf.where(eco > -900.0, eco / eco_scale, -999.0)
        risk_y = tf.where(risk > -900.0, risk, -999.0)

        # --- free-supervision targets derived on the fly ---
        # quality: 1 = usable pixel (SCL not invalid). Always has labels.
        scl_i = tf.cast(scl, tf.int32)
        bad = tf.zeros(tf.shape(scl_i), tf.bool)
        for c in list(cfg.invalid_scl):
            bad = tf.logical_or(bad, tf.equal(scl_i, c))
        quality_y = 1.0 - tf.cast(bad, tf.float32)

        # boundary: morphological gradient of seg one-hot (valid only where seg is)
        valid_seg = tf.cast(tf.reduce_sum(seg, -1, keepdims=True) > 0, tf.float32)
        mx = tf.nn.max_pool2d(seg, 3, 1, "SAME")
        mn = -tf.nn.max_pool2d(-seg, 3, 1, "SAME")
        edge = tf.reduce_max(tf.clip_by_value(mx - mn, 0.0, 1.0), -1,
                             keepdims=True)
        boundary_y = tf.where(valid_seg > 0, edge, -1.0)

        inputs = {"s1_input": s1, "s2_input": s2, "scl_input": scl,
                  "aux_input": aux, "s1_t2_input": s1t2, "s2_t2_input": s2t2,
                  "scl_t2_input": sclt2, "aux_t2_input": auxt2, "dt_input": dt}
        targets = {"seg_head": seg, "carbon_head": carbon_y, "eco_head": eco_y,
                   "change_head": change, "risk_head": risk_y,
                   "quality_head": quality_y, "boundary_head": boundary_y}
        if deep:
            targets["seg_deep_head"] = tf.nn.avg_pool2d(seg, 4, 4, "VALID")
        return inputs, targets

    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=output_signature(cfg))
    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(cfg.shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(random_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(unpack_to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(cfg.deterministic)
    ds = ds.with_options(opts)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 8. SWIN-V2 COMPONENTS  (all buffers are non-trainable WEIGHTS: F1 fixed)
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
    """Swin-V2 attention: scaled-cosine attention (learnable per-head logit
    scale, clamped) + log-spaced Continuous Position Bias via meta-MLP.
    All coordinate buffers stored as non-trainable weights (Keras-3 safe)."""

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
        # learnable logit scale: init log(10), clamped to <= log(100)
        self.logit_scale = self.add_weight(
            name="logit_scale", shape=(self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(math.log(10.0)),
            trainable=True)
        # CPB meta-network on log-spaced relative coordinates
        self.cpb1 = layers.Dense(self.cpb_hidden, activation="relu", name="cpb1")
        self.cpb2 = layers.Dense(self.num_heads, use_bias=False, name="cpb2")

        # ---- buffers as non-trainable WEIGHTS (FIX F1) ----
        # pairwise relative coords for each of the N x N token pairs, log-spaced
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws),
                                      indexing="ij"))            # (2, ws, ws)
        cf = coords.reshape(2, -1).astype(np.float32)            # (2, N)
        rel = cf[:, :, None] - cf[:, None, :]                    # (2, N, N)
        rel = rel.transpose(1, 2, 0)                             # (N, N, 2)
        rel = rel / max(ws - 1, 1) * 8.0                         # normalize to [-8, 8]
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

        # cosine attention, fp32 for stability
        qn = tf.math.l2_normalize(tf.cast(q, tf.float32), axis=-1)
        kn = tf.math.l2_normalize(tf.cast(k, tf.float32), axis=-1)
        scale = tf.exp(tf.minimum(self.logit_scale, math.log(100.0)))
        attn = tf.matmul(qn, kn, transpose_b=True) * tf.cast(scale, tf.float32)

        # continuous position bias, squashed to [-16, 16] like Swin-V2
        bias = self.cpb2(self.cpb1(self.rel_coords))             # (N, N, heads)
        bias = 16.0 * tf.sigmoid(bias)
        attn += tf.transpose(bias, [2, 0, 1])[None]

        if mask is not None:                                     # (nW, 1, N, N)
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
    """Res-post-norm Swin-V2 block:
       x = norm1(x + dp(attn(x)));  x = norm2(x + dp(mlp(x)))."""

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
            layers.Dense(hidden, activation="gelu"),
            layers.Dropout(self.drop),
            layers.Dense(self.dim),
            layers.Dropout(self.drop)], name="mlp")
        # shifted-window mask as non-trainable WEIGHT (FIX F1, same bug class)
        if self.shift_size > 0:
            mask_np = self._build_mask_np()                      # (nW, 1, N, N)
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
            h = tf.roll(h, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        w = window_partition(h, self.window_size)
        w = self.attn(w, mask=self.attn_mask, training=training)
        h = window_reverse(w, self.window_size, self.H, self.W)
        if self.shift_size > 0:
            h = tf.roll(h, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        x = self.norm1(x + self.dp1(h, training=training))       # post-norm
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
    """Cheap fusion for high-res levels: opt + tanh(gate) * W[opt, sar].
    Gate initialized at 0 -> identity at init (stable)."""

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
    """Bidirectional token cross-attention (deep levels only, small N)."""

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


@tf.keras.utils.register_keras_serializable(package="mangrove")
class TimeEmbedding(layers.Layer):
    """Sinusoidal embedding of the absolute time gap (years) + MLP."""

    def __init__(self, dim, freqs=16, max_years=20.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.freqs = int(freqs)
        self.max_years = float(max_years)

    def build(self, input_shape):
        f = np.arange(self.freqs, dtype=np.float32) / self.freqs
        self.freq_table = self.add_weight(
            name="freq_table", shape=(self.freqs,), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(
                1.0 / (10000.0 ** f)), trainable=False)
        self.d1 = layers.Dense(self.dim, activation="gelu", name="t1")
        self.d2 = layers.Dense(self.dim, name="t2")
        super().build(input_shape)

    def call(self, dt):
        dt = tf.cast(tf.reshape(dt, [-1, 1]), tf.float32) / self.max_years
        ang = dt * self.freq_table * 2.0 * math.pi
        emb = tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)
        return self.d2(self.d1(emb))                               # (B, dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "freqs": self.freqs,
                    "max_years": self.max_years})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class FiLM(layers.Layer):
    """Feature-wise modulation of a feature map by a conditioning vector."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)

    def build(self, input_shape):
        self.dense = layers.Dense(2 * self.channels, name="film")
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        gb = self.dense(cond)                                      # (B, 2C)
        g, b = tf.split(gb, 2, axis=-1)
        g = tf.reshape(g, [-1, 1, 1, self.channels])
        b = tf.reshape(b, [-1, 1, 1, self.channels])
        return x * (1.0 + tf.cast(g, x.dtype)) + tf.cast(b, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg["channels"] = self.channels
        return cfg


# ==============================================================================
# 10. ENCODERS / DECODER BLOCKS
# ==============================================================================
def _heads_for(dim):
    return max(2, min(12, dim // 16))


def make_swin_encoder(prefix, dims, depths, cfg, dp_rates):
    """Returns apply(x) -> (bottleneck, skips). dp_rates consumed in order."""
    ws = cfg.window_size
    if True:  # hybrid CNN stem (more stable than a single strided conv)
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
# 11. FULL MODEL
# ==============================================================================
def build_model(cfg, s1_stats, s2_stats):
    C, ps, A = cfg.num_classes, cfg.patch_size, cfg.aux_channels
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)
    ws = cfg.window_size
    active = set(cfg.active_heads)

    s1_in = layers.Input((ps, ps, 4), name="s1_input")
    s2_in = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1), name="scl_input")
    aux_in = layers.Input((ps, ps, A), name="aux_input")
    s1t2_in = layers.Input((ps, ps, 4), name="s1_t2_input")
    s2t2_in = layers.Input((ps, ps, 10), name="s2_t2_input")
    sclt2_in = layers.Input((ps, ps, 1), name="scl_t2_input")
    auxt2_in = layers.Input((ps, ps, A), name="aux_t2_input")
    dt_in = layers.Input((1,), name="dt_input")

    # shared feature engineering (siamese: ONE layer instance for both dates)
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    f1, f2 = fe([s1_in, s2_in, scl_in, aux_in])
    g1, g2 = fe([s1t2_in, s2t2_in, sclt2_in, auxt2_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]), variance=list(s1_stats[1]),
                              name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]), variance=list(s2_stats[1]),
                              name="s2_norm")
    f1 = n1(f1); f2 = n2(f2)
    g1 = n1(g1); g2 = n2(g2)

    # linear stochastic-depth schedule across all encoder blocks
    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])

    b1, sk1 = enc_s1(f1)                       # SAR T1
    b2, sk2 = enc_s2(f2)                       # OPT T1
    c1, _ = enc_s1(g1)                         # SAR T2 (shared weights)
    c2, sk2_t2 = enc_s2(g2)                    # OPT T2

    # per-level fusion (opt stream fused with sar context; siamese-shared)
    fused_t1, fused_t2 = [], []
    sar_sk_t2 = []
    for i, d in enumerate(dims):
        s_t1, s_t2 = sk1[i], None
        o_t1, o_t2 = sk2[i], sk2_t2[i]
        # project sar to opt dims where needed
        if s_t1.shape[-1] != d:
            s_t1 = layers.Conv2D(d, 1, name=f"sar_proj{i}")(s_t1)
        s_t2 = layers.Conv2D(d, 1, name=f"sar_t2_proj{i}")(sk1[i]) \
            if sk1[i].shape[-1] != d else sk1[i]
        if i in tuple(cfg.cross_attn_levels):
            fuse = CrossAttentionFusion(d, cfg.fusion_heads,
                                        name=f"cross_attn_fusion{i}")
        else:
            fuse = GatedFusion(d, name=f"gated_fusion{i}")
        fused_t1.append(fuse([o_t1, s_t1]))
        fused_t2.append(fuse([o_t2, s_t2]))    # same instance: siamese
        sar_sk_t2.append(s_t2)
    fused_b, fused_b_t2 = fused_t1[-1], fused_t2[-1]

    # ---- main decoder (T1) ----
    x = fused_b
    deep_out = None
    for i, wd in enumerate(cfg.dec_widths):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"dec_up{i}")(x)
        li = len(dims) - 2 - i                 # fuse levels 2,1,0 as skips
        if li >= 0:
            x = layers.concatenate([x, fused_t1[li], sk1[li]], name=f"dec_cat{i}")
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

    # ---- time-aware change decoder: per-level diffs + FiLM(dt) ----
    temb = TimeEmbedding(cfg.time_embed_dim, name="time_embed")(dt_in)
    diffs = []
    for i in range(len(dims)):
        diffs.append(layers.concatenate(
            [AbsDiff(name=f"diff_opt{i}")([fused_t1[i], fused_t2[i]]),
             AbsDiff(name=f"diff_sar{i}")([sk1[i], sar_sk_t2[i]])],
            name=f"diff_cat{i}"))
    d = conv_block(diffs[-1], cfg.change_widths[0], use_se=cfg.use_se,
                   name="change_bottleneck")
    d = FiLM(cfg.change_widths[0], name="film_bottleneck")([d, temb])
    for i, wd in enumerate(cfg.change_widths[1:]):
        d = layers.UpSampling2D(interpolation="bilinear", name=f"change_up{i}")(d)
        li = len(dims) - 2 - i
        if li >= 0:
            d = layers.concatenate([d, diffs[li]], name=f"change_cat{i}")
        d = conv_block(d, wd, use_se=cfg.use_se, name=f"change_c{i}")
        d = FiLM(wd, name=f"film_{i}")([d, temb])

    # ---- heads (pluggable) ----
    outputs = {}
    if "seg" in active:
        outputs["seg_head"] = layers.Activation(
            "softmax", dtype="float32", name="seg_head")(
            layers.Conv2D(C, 1, padding="same", name="seg_logits")(x))
    if cfg.deep_supervision and deep_out is not None:
        outputs["seg_deep_head"] = deep_out
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
    if "change" in active:
        outputs["change_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="change_head")(
            layers.Conv2D(1, 1, padding="same", name="change_logits")(d))
    if "quality" in active:
        outputs["quality_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="quality_head")(
            layers.Conv2D(1, 1, padding="same", name="quality_logits")(x))
    if "boundary" in active:
        outputs["boundary_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="boundary_head")(
            layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x))

    return models.Model(
        inputs=[s1_in, s2_in, scl_in, aux_in,
                s1t2_in, s2t2_in, sclt2_in, auxt2_in, dt_in],
        outputs=outputs, name=f"MangroveEcoMonitor_v{VERSION}")


# ==============================================================================
# 12. LOSSES & METRICS
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=1.0,
                 label_smoothing=0.0, **kwargs):
        super().__init__(name="focal_dice", **kwargs)
        self.alpha = None if alpha is None else [float(a) for a in alpha]
        self.gamma = float(gamma)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        valid = tf.cast(tf.reduce_sum(y_true, -1, keepdims=True) > 0, tf.float32)
        if self.label_smoothing > 0:
            C = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / C
        a = tf.ones_like(y_pred) if self.alpha is None \
            else tf.constant(self.alpha, tf.float32)
        ce = -y_true * tf.math.log(y_pred)
        fl = tf.reduce_sum(a * tf.pow(1.0 - y_pred, self.gamma) * ce,
                           axis=-1, keepdims=True) * valid
        focal = tf.reduce_sum(fl, axis=[1, 2, 3]) / \
            (tf.reduce_sum(valid, axis=[1, 2, 3]) + eps)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
        dice = tf.reduce_mean(1.0 - (2.0 * inter + 1.0) / (denom + 1.0), axis=-1)
        return self.focal_weight * focal + self.dice_weight * dice

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma,
                    "focal_weight": self.focal_weight,
                    "dice_weight": self.dice_weight,
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
class MaskedHuber(tf.keras.losses.Loss):
    """Huber on valid pixels (more robust than MSE for noisy eco/risk labels)."""
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(name="masked_huber", **kwargs)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        err = tf.abs(tf.where(mask > 0, y, 0.0) - p)
        hub = tf.where(err <= self.delta, 0.5 * err ** 2,
                       self.delta * (err - 0.5 * self.delta))
        return tf.reduce_sum(hub * mask) / (tf.reduce_sum(mask) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg["delta"] = self.delta
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedBinaryFocal(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(name="masked_binary_focal", **kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
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
class MaskedBCE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="masked_bce", **kwargs)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
        mask = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        bce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-6)


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.total_cm = self.add_weight(
            name="cm", shape=(self.num_classes, self.num_classes),
            initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.reshape(tf.reduce_sum(y_true, -1) > 0, [-1])
        yt = tf.boolean_mask(
            tf.reshape(tf.argmax(y_true, -1, output_type=tf.int32), [-1]), valid)
        yp = tf.boolean_mask(
            tf.reshape(tf.argmax(y_pred, -1, output_type=tf.int32), [-1]), valid)
        cm = tf.math.confusion_matrix(yt, yp, num_classes=self.num_classes,
                                      dtype=tf.float32)
        self.total_cm.assign_add(cm)

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
        cfg = super().get_config()
        cfg["num_classes"] = self.num_classes
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonMAE(tf.keras.metrics.Metric):
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


@tf.keras.utils.register_keras_serializable(package="mangrove")
class ChangeF1(tf.keras.metrics.Metric):
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
        self.fp.assign_add(tf.reduce_sum((1 - y) * p_bin * mask))
        self.fn.assign_add(tf.reduce_sum(y * (1 - p_bin) * mask))

    def result(self):
        prec = self.tp / (self.tp + self.fp + 1e-6)
        rec = self.tp / (self.tp + self.fn + 1e-6)
        return 2 * prec * rec / (prec + rec + 1e-6)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["threshold"] = self.threshold
        return cfg


# ==============================================================================
# 13. TRAINER: custom train_step, LEARNABLE task weighting, single grad path
# ==============================================================================
HEAD_LOSS_MAP = {
    "seg": "focal_dice", "change": "binary_focal", "quality": "bce",
    "boundary": "bce", "carbon": "carbon_nll", "eco": "huber", "risk": "huber",
}


class EcoMonitorTrainer(tf.keras.Model):
    """Wraps the functional net. Owns:
       - per-task losses + learnable log-variance task weighting (Kendall 2018)
       - explicit metrics per active head
       Gradient accumulation: NATIVE optimizer path only (FIX F4). If
       cfg.grad_accum > 1 we require optimizer gradient_accumulation_steps
       support and never run a second, custom accumulator."""

    def __init__(self, cfg, net, alpha, **kwargs):
        super().__init__(**kwargs)
        self.cfg_obj = cfg
        self.net = net
        self.active = list(cfg.active_heads)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.loss_fns = {}
        for h in self.active:
            kind = HEAD_LOSS_MAP[h]
            if kind == "focal_dice":
                self.loss_fns[h] = FocalDiceLoss(
                    alpha=alpha, gamma=cfg.focal_gamma,
                    label_smoothing=cfg.label_smoothing)
            elif kind == "binary_focal":
                self.loss_fns[h] = MaskedBinaryFocal()
            elif kind == "bce":
                self.loss_fns[h] = MaskedBCE()
            elif kind == "carbon_nll":
                self.loss_fns[h] = CarbonNLLLoss()
            else:
                self.loss_fns[h] = MaskedHuber()
        if cfg.deep_supervision:
            self.loss_fns["seg_deep"] = FocalDiceLoss(
                alpha=alpha, gamma=cfg.focal_gamma,
                label_smoothing=cfg.label_smoothing)

        # learnable per-task log-variances (trainable weights of the wrapper)
        self.use_learned_w = cfg.task_weighting == "learned"
        self.task_log_vars = {}
        for name in list(self.loss_fns.keys()):
            self.task_log_vars[name] = self.add_weight(
                name=f"task_logvar_{name}", shape=(), dtype=tf.float32,
                initializer="zeros", trainable=self.use_learned_w)

        # metrics
        self.per_head_loss = {n: tf.keras.metrics.Mean(name=f"{n}_loss")
                              for n in self.loss_fns}
        self.metric_objs = {}
        if "seg" in self.active:
            self.metric_objs["seg"] = MeanIoU(cfg.num_classes)
        if "change" in self.active:
            self.metric_objs["change"] = ChangeF1()
        if "carbon" in self.active:
            self.metric_objs["carbon"] = CarbonMAE()
        for h in ("quality", "boundary"):
            if h in self.active:
                self.metric_objs[h] = tf.keras.metrics.Mean(name=f"{h}_acc")

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    @property
    def metrics(self):
        ms = [self.loss_tracker] + list(self.per_head_loss.values()) + \
             list(self.metric_objs.values())
        return ms

    def _per_task_losses(self, y, p):
        out = {}
        for name, fn in self.loss_fns.items():
            key = name if name != "seg_deep" else "seg_deep_head"
            tgt = y.get(key)
            prd = p.get(key if key in p else None)
            if tgt is None or prd is None:
                continue
            out[name] = fn(tgt, prd)
        return out

    def _total(self, per_task):
        total = 0.0
        for name, l in per_task.items():
            if self.use_learned_w:
                lv = self.task_log_vars[name]
                w_extra = self.cfg_obj.deep_supervision_weight \
                    if name == "seg_deep" else 1.0
                total = total + w_extra * (l * tf.exp(-lv) + lv)
            else:
                w = self.cfg_obj.fixed_loss_weights.get(
                    name.replace("_deep", ""), 1.0)
                if name == "seg_deep":
                    w = self.cfg_obj.deep_supervision_weight
                total = total + w * l
        return total

    def _update_metrics(self, per_task, y, p, total):
        self.loss_tracker.update_state(total)
        for name, l in per_task.items():
            self.per_head_loss[name].update_state(l)
        if "seg" in self.metric_objs and "seg_head" in p:
            self.metric_objs["seg"].update_state(y["seg_head"], p["seg_head"])
        if "change" in self.metric_objs and "change_head" in p:
            self.metric_objs["change"].update_state(y["change_head"],
                                                    p["change_head"])
        if "carbon" in self.metric_objs and "carbon_head" in p:
            self.metric_objs["carbon"].update_state(y["carbon_head"],
                                                    p["carbon_head"])
        for h in ("quality", "boundary"):
            key = f"{h}_head"
            if h in self.metric_objs and key in p:
                valid = tf.cast(y[key] >= 0, tf.float32)
                acc = tf.cast(tf.equal(tf.cast(p[key] > 0.5, tf.float32),
                                       y[key]), tf.float32)
                denom = tf.reduce_sum(valid)
                val = tf.reduce_sum(acc * valid) / (denom + 1e-6)
                self.metric_objs[h].update_state(val)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            p = self.net(x, training=True)
            per_task = self._per_task_losses(y, p)
            total = self._total(per_task) + sum(self.net.losses)
        # mixed-precision safe scaling (Keras-3 LossScaleOptimizer path)
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
    """Single accumulation mechanism: native optimizer support or hard error."""
    kwargs = dict(learning_rate=lr, weight_decay=cfg.weight_decay,
                  clipnorm=cfg.grad_clip)
    if cfg.grad_accum > 1:
        try:
            return optimizers.AdamW(
                gradient_accumulation_steps=int(cfg.grad_accum), **kwargs)
        except TypeError as e:
            raise RuntimeError(
                "cfg.grad_accum > 1 requires Keras>=3.4 native "
                "gradient_accumulation_steps. Either upgrade TF/Keras or set "
                "grad_accum=1 (v5 does NOT run a second custom accumulator "
                "- see FIX F4).") from e
    return optimizers.AdamW(**kwargs)


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
        """
================================================================================
 MANGROVE ECO-MONITOR v5.0  "BLUE SENTINEL"
 Production-grade dual-modality (Sentinel-1 SAR x Sentinel-2 optical) mangrove
 monitoring system: Swin-V2 Siamese U-Net, per-level gated fusion, time-aware
 change detection, pluggable multi-task heads, uncertainty-aware MRV outputs.

 REQUIRES: Python >= 3.10, TensorFlow >= 2.16 (Keras 3), Linux or WSL2 on
           Windows (native Windows has NO TensorFlow GPU support since TF 2.10).
           Optional: rasterio (COG/GeoJSON export), scikit-learn (eval fallback
           exists), matplotlib (plots).

--------------------------------------------------------------------------------
 WHAT V5 FIXES (all confirmed defects from the v4.1 review + code audit)
--------------------------------------------------------------------------------
 [F1] Keras-3 fatal crash: tf.constant() buffers created in build() go out of
      scope under re-tracing. v5 stores EVERY buffer (relative coords, shift
      masks) as non-trainable weights via add_weight(Constant). The Swin-V2
      CPB design removes the giant RPB table entirely.
 [F2] Normalization off-by-one: cloud-mask identity stats were written to
      channel 18 (elevation) instead of 17 (mask). v5 computes the mask index
      symbolically from the feature layout; aux channels now normalized with
      their real stats.
 [F3] Hardware detection: "NVIDIA RTX 6000 Ada Generation" matched nothing and
      silently fell back to the 16 GB T4 preset. v5 adds explicit RTX-6000-Ada
      matching PLUS a VRAM-probing tier fallback via nvidia-smi.
 [F4] Double gradient accumulation (native AdamW kwarg + custom train_step
      wrapper fighting each other). v5 uses ONE mechanism: the native Keras
      optimizer path, with a version guard and a hard error if unavailable.
 [F5] Four of five heads had no label source. v5 makes heads PLUGGABLE:
      seg/change/quality/boundary ship ON (all have real labels), carbon/eco/
      risk ship OFF with documented data sources; a pre-flight scan validates
      enabled heads against the NPZ keys actually present.
 [F6] Windows GPU: v5 detects Windows + CUDA-visible-but-TF-invisible GPUs and
      prints exact WSL2 remediation steps.

 NEW BUGS FOUND BY V5 AUDIT (were NOT in the review):
 [F7] Augmentation channel misalignment: random_aug sliced x[..., 14:18] /
      18:22 / 22:32 assuming aux=4, but aux=6 -> brightness/contrast was applied
      to SCL+AUX and S1-T2 bands; everything downstream was corrupted. v5 uses
      cumulative symbolic offsets derived from SIZES.
 [F8] Gaussian noise was added to the WHOLE packed tensor including one-hot seg
      labels, carbon/eco/change/risk targets -> destroyed label integrity.
      v5 restricts noise to input bands only.
 [F9] TTA blending bug: predictions were accumulated once PER TTA variant but
      the weight map only ONCE per tile -> all outputs inflated ~4x with TTA on.
      v5 accumulates weights per variant pass.
 [F10] Inverted blending kernel: v4's "gaussian" peaked at tile EDGES and was
      ~0.01 at the centre. v5 uses a proper centre-peaked Hann window.

--------------------------------------------------------------------------------
 V5 MAJOR UPGRADES (grounded in 2025-2026 research)
--------------------------------------------------------------------------------
 U1  Swin-V2 blocks: res-post-norm + scaled-cosine attention with learnable
     per-head logit scale (clamped) + log-spaced Continuous Position Bias MLP.
     More stable at depth, transferable across window sizes.
 U2  Per-level fusion: cheap gated 1x1 fusion at high-res levels, bidirectional
     cross-attention at deep levels (v4 fused ONLY at the bottleneck).
 U3  Time-aware Siamese change decoder: per-level feature diffs as skips +
     FiLM conditioning on real acquisition-date gap (decimal years). Enables
     backcasting to arbitrary dates and GMW-style annual loss products.
 U4  Pluggable task heads with LEARNABLE uncertainty weighting (Kendall et
     al.): loss weights are learned log-variances, not hand-tuned constants.
 U5  Free-supervision heads: quality head (pixel usability from SCL, trained
     WITHOUT extra labels) and boundary head (morphological edge target derived
     from seg labels) -> sharp coastlines + reliability masks at inference.
 U6  Deep supervision (aux seg logits at 1/4 res) with small weight.
 U7  SimMIM-style self-supervised pretraining mode for unlabeled S1+S2 scenes
     (masked-block reconstruction), then fine-tune with by-name weight load.
     This is the standard answer to EO label scarcity (MAE/SimMIM/SatMAE line).
 U8  Modality dropout: randomly zero the SAR or optical branch during training
     so the model stays operational when S2 is cloudy (ESA RAMM shows SAR-only
     alerting is a hard operational requirement in the tropics). predict_scene
     supports sar_only=True.
 U9  Failure tolerance: NaN/Inf NPZ sanitization with counts, schema validation
     with actionable errors, OOM auto-retry with halved batch (resumes from
     last checkpoint), memory preflight probe, TerminateOnNaN, BackupAndRestore.
 U10 EMA of weights (Polyak averaging) - evaluation, checkpointing and export
     all run on EMA weights; separate EMA checkpoint saved.
 U11 Post-hoc temperature calibration of seg probabilities on the val set;
     calibrated probabilities used at inference and stored in norm stats.
 U12 MRV-ready export pack: COG GeoTIFFs + STAC 1.0 item JSON + change-alert
     GeoJSON polygons (area, period, confidence) + summary CSV with class areas
     (ha) and carbon stock with 95% CI in tCO2e (VM0033-aligned reporting
     primitives) + preview PNG.
 U13 Class-balanced patch sampling: rare classes (perm flood, degraded
     mangrove) oversampled via per-patch weighting.
 U14 Hardware: RTX-6000-Ada preset, generic VRAM tiers, optional batch
     auto-tuner that empirically probes the largest safe batch.
 U15 NISAR/L-band readiness: aux channel count is config-driven; drop NISAR
     L-band HV/HH ratio or ESA BIOMASS P-band features into aux and everything
     (stats, stems, aug) adapts. aux_names/scale are configurable.

 NPZ SCHEMA v5 (backward-compatible superset of v4.1)
   required: s1 (H,W,2|4) dB | s2 (H,W,10) reflectance
   optional: s1_std, scl, aux (H,W,A), label (0..4, 255=ignore),
             carbon (t/ha, -999), eco (H,W,2, -999), change (0/1, 255),
             risk (0..1, -999), s1_t2/s2_t2/scl_t2/aux_t2,
   NEW:      date_t1, date_t2 : decimal year scalars (e.g. 2024.42) ->
             drives the time-aware change decoder. If a real T2 exists but
             dates are absent, dt defaults to 1.0.

 HEAD DATA SOURCES (honest wiring)
   seg     : WorldCover + GSW + GMW v4.x derived labels            [ON ]
   change  : Hansen GFC v1.13 lossyear / GMW alerts / T2 pairs     [ON ]
   quality : derived from SCL at train time (no NPZ field needed)  [ON ]
   boundary: derived from seg labels at train time                 [ON ]
   carbon  : ESA CCI Biomass / GEDI / NISAR L-band products        [OFF]
   eco     : Landsat/MODIS LST (soil moisture NOT viable @10m)     [OFF]
   risk    : hydrodynamic surge model or heuristic (weak labels)   [OFF]

 MODES:  python mangrove_v5.py smoke | pretrain | train | eval | infer | export
================================================================================
"""

import os
import gc
import sys
import json
import glob
import math
import time
import shutil
import argparse
import platform
import subprocess
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "5.0.0"
MIN_TF = (2, 16)


# ==============================================================================
# 1. ENVIRONMENT, GPU, PLATFORM GUARDS
# ==============================================================================
def _parse_version(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


def environment_check():
    """Fail fast with actionable messages. Addresses review F1/F6 context."""
    ok = True
    if _parse_version(tf.__version__) < MIN_TF:
        print(f"[FATAL] TensorFlow {tf.__version__} < {'.'.join(map(str, MIN_TF))}. "
              f"v5 requires Keras 3 APIs. pip install -U tensorflow")
        ok = False

    gpus = tf.config.list_physical_devices("GPU")
    if platform.system() == "Windows" and not gpus:
        smi = shutil.which("nvidia-smi")
        if smi is not None:
            print("=" * 78)
Web search


            print("[HARDWARE BLOCKER] Windows detected, NVIDIA driver present, but"
                  " TensorFlow sees NO GPU.")
            print("TensorFlow dropped native-Windows GPU support after v2.10. Fix:")
            print("  1) wsl --install -d Ubuntu   (PowerShell, admin)")
            print("  2) Install the Windows NVIDIA driver (it serves WSL2 too; do NOT")
            print("     install a Linux driver inside WSL2)")
            print("  3) Inside Ubuntu: python -m pip install tensorflow[and-cuda]")
            print("  4) Verify: python -c \"import tensorflow as tf; "
                  "print(tf.config.list_physical_devices('GPU'))\"")
            print("  WSL2 gives near-native CUDA performance for this workload.")
            print("=" * 78)
            ok = False
    return ok, gpus


ENV_OK, GPUS = environment_check()
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working/" if ON_KAGGLE else "./output_v5/"
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
    """Best-effort VRAM probe via nvidia-smi (works on Linux/WSL2)."""
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
    train_dir: str = "/kaggle/input/mangrove-v4-npz/train/"
    val_dir: str = "/kaggle/input/mangrove-v4-npz/val/"
    test_dir: str = "/kaggle/input/mangrove-v4-npz/test/"
    patch_size: int = 256                 # -> maps 64/32/16/8, window 8 divides all
    train_stride: int = 128
    max_train_patches: int = 8000
    max_val_patches: int = 1200
    min_valid_frac: float = 0.35
    max_cloud_frac: float = 0.85
    ignore_label: int = 255
    num_classes: int = 5
    invalid_scl: tuple = (0, 1, 2, 3, 7, 8, 9, 10, 11)
    cloud_dilate: int = 5
    # aux channels are config-driven (NISAR-ready: add L-band features here)
    aux_names: tuple = ("elev", "slope", "ndvi_trend", "flood_freq",
                        "lst_trend", "s1_trend")
    aux_scale: tuple = (50.0, 20.0, 0.1, 1.0, 50.0, 1.0)
    # class-balanced sampling
    rare_classes: tuple = (2, 3)          # perm flood, degraded mangrove
    rare_boost: float = 4.0
    # ---- model ----
    embed_dims: tuple = (48, 96, 192, 384)
    depths: tuple = (2, 2, 2, 2)
    window_size: int = 8
    drop_path: float = 0.15               # linear stochastic-depth schedule 0 -> dp
    mlp_ratio: float = 4.0
    fusion_heads: int = 8
    cross_attn_levels: tuple = (2, 3)     # full cross-attn only at deep levels
    dec_widths: tuple = (192, 96, 48)
    refine_widths: tuple = (32, 24)
    change_widths: tuple = (128, 96, 64, 48, 32)
    decoder_swin_levels: int = 2
    use_se: bool = True
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.3
    time_embed_dim: int = 64
    active_heads: tuple = ("seg", "change", "quality", "boundary")
    # heads OFF by default; enable only after wiring real labels (see header)
    # e.g. active_heads = ("seg","change","quality","boundary","carbon")
    # ---- training ----
    hardware: str = "auto"
    batch_per_replica: int = 4
    grad_accum: int = 1                   # 1 = native single-step (recommended)
    epochs: int = 120
    lr: float = 2e-4
    lr_warmup_epochs: int = 3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    task_weighting: str = "learned"       # "learned" (Kendall) or "fixed"
    fixed_loss_weights: dict = field(default_factory=lambda: {
        "seg": 1.0, "change": 1.2, "quality": 0.2, "boundary": 0.3,
        "carbon": 0.6, "eco": 0.25, "risk": 0.25})
    carbon_scale: float = 100.0
    eco_scale: tuple = (50.0, 1.0)
Web search


    ema_decay: float = 0.999
    grad_clip: float = 1.0
    deterministic: bool = False
    auto_batch: bool = False              # empirical preflight probe
    oom_retries: int = 2                  # halve batch and resume on OOM
    pretrain_weights: str = ""            # path to SimMIM encoder weights
    # ---- augmentation ----
    augment: bool = True
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_noise: float = 0.02
    modality_dropout_p: float = 0.08      # per-modality, same choice for T1/T2
    # ---- pretraining ----
    pretrain_epochs: int = 40
    pretrain_lr: float = 3e-4
    pretrain_mask_ratio: float = 0.5
    pretrain_block: int = 32
    # ---- inference ----
    infer_batch: int = 8
    infer_tta: bool = True
    infer_overlap: float = 0.5
    mc_samples: int = 0                   # >1 enables MC-dropout epistemic maps
    seg_temperature: float = 1.0          # filled by calibration
    change_threshold: float = 0.5
    # ---- MRV ----
    pixel_size_m: float = 10.0            # fallback if ref tif unavailable
    carbon_to_co2e: float = 44.0 / 12.0   # tC -> tCO2e
    c_fraction: float = 0.5               # biomass -> carbon
    shuffle_buffer: int = 256

    @property
    def aux_channels(self):
        return len(self.aux_names)

    def save(self, path):
        d = asdict(self)
        d["invalid_scl"] = list(self.invalid_scl)
        d["aux_names"] = list(self.aux_names)
        d["aux_scale"] = list(self.aux_scale)
        d["rare_classes"] = list(self.rare_classes)
        d["embed_dims"] = list(self.embed_dims)
        d["depths"] = list(self.depths)
        d["cross_attn_levels"] = list(self.cross_attn_levels)
        d["dec_widths"] = list(self.dec_widths)
        d["refine_widths"] = list(self.refine_widths)
        d["change_widths"] = list(self.change_widths)
        d["active_heads"] = list(self.active_heads)
        d["eco_scale"] = list(self.eco_scale)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        for k in ("invalid_scl", "aux_names", "aux_scale", "rare_classes",
                  "embed_dims", "depths", "cross_attn_levels", "dec_widths",
                  "refine_widths", "change_widths", "active_heads", "eco_scale"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)


CFG = Config()

HW_PRESETS = {
    # name: (batch_per_replica, grad_accum, infer_batch)
    "h100":        (32, 1, 32),
    "a100":        (24, 1, 32),
    "rtx6000ada":  (16, 1, 16),   # 48 GB Ada workstation card (review F3)
    "a6000":       (16, 1, 16),
    "l40":         (12, 1, 16),
    "4090":        (8, 1, 8),
    "l4":          (6, 1, 8),
    "t4x2":        (4, 2, 8),
    "cpu":         (2, 1, 2),
}


def resolve_hardware(cfg):
    """Explicit name match first, then VRAM tiers, then conservative default."""
    name = " ".join(GPU_NAMES)
    hw = cfg.hardware
    if hw == "auto":
        n = name.lower()
        if "h100" in n:
            hw = "h100"
        elif "a100" in n:
            hw = "a100"
        elif "6000 ada" in n or "rtx 6000" in n:     # RTX 6000 Ada (48 GB)
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
# 3. NPZ LOADING WITH VALIDATION + SANITIZATION (failure tolerance)
# ==============================================================================
class NPZSchemaError(ValueError):
    pass


def _process_s1(s1):
    s1 = s1.astype(np.float32)
    if np.nanmin(s1) >= 0:                                # linear -> dB
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
    """NaN/Inf guard with counts (F9-tolerance). Returns fixed copy."""
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / arr.size
        if frac > 0.30:
            raise NPZSchemaError(f"{path}: '{name}' is {frac:.0%} non-finite")
        print(f"  [sanitize] {os.path.basename(path)}:{name} "
              f"had {n_bad} non-finite px -> 0")
        arr = np.where(bad, 0.0, arr)
    return arr


def load_npz(path, cfg=CFG):
    """Returns 15 arrays (13 v4.1 fields + date_t1 + date_t2).
    Missing -> sentinel-filled; NaN/Inf sanitized with counts."""
    A = cfg.aux_channels
    scale = np.asarray(cfg.aux_scale, np.float32)
    with np.load(path) as d:
        files = d.files
        if "s1" not in files or "s2" not in files:
            raise NPZSchemaError(f"{path}: missing required keys 's1'/'s2' "
                                 f"(found: {sorted(files)})")
        s1 = _sanitize(_process_s1(d["s1"]), "s1", path)
        s2 = _sanitize(_process_s2(d["s2"]), "s2", path)
        if s1.ndim != 3 or s2.ndim != 3 or s2.shape[-1] != 10:
            raise NPZSchemaError(f"{path}: bad shapes s1{s1.shape} s2{s2.shape}")
        H, W = s2.shape[:2]
        if s1.shape[:2] != (H, W):
            raise NPZSchemaError(f"{path}: s1/s2 spatial mismatch "
                                 f"{s1.shape[:2]} vs {(H, W)}")

        s1_std = (d["s1_std"].astype(np.float32)
                  if "s1_std" in files else np.zeros((H, W, 2), np.float32))
        s1 = np.concatenate([s1, _sanitize(s1_std, "s1_std", path)], -1)
        scl = (d["scl"].astype(np.float32)[..., None]
               if "scl" in files else np.full((H, W, 1), 4.0, np.float32))
        aux = (d["aux"].astype(np.float32)
               if "aux" in files else np.zeros((H, W, A), np.float32))
        if aux.shape[-1] != A:
            fixed = np.zeros((H, W, A), np.float32)
            n = min(A, aux.shape[-1])
            fixed[..., :n] = aux[..., :n]
            aux = fixed
        aux = _sanitize(aux, "aux", path) / scale

        label = (d["label"].astype(np.int32)
                 if "label" in files else np.full((H, W), 255, np.int32))
        carbon = (d["carbon"].astype(np.float32)
                  if "carbon" in files else np.full((H, W), -999.0, np.float32))
        eco = (d["eco"].astype(np.float32)
               if "eco" in files else np.full((H, W, 2), -999.0, np.float32))
        change = (d["change"].astype(np.int32)
                  if "change" in files else np.full((H, W), 255, np.int32))
        risk = (d["risk"].astype(np.float32)
                if "risk" in files else np.full((H, W), -999.0, np.float32))

        has_t2 = "s1_t2" in files and "s2_t2" in files
        if has_t2:
            s1t2 = _sanitize(_process_s1(d["s1_t2"]), "s1_t2", path)
            s1t2_std = (d["s1_t2_std"].astype(np.float32)
                        if "s1_t2_std" in files else np.zeros_like(s1t2))
            s1t2 = np.concatenate([s1t2, s1t2_std], -1)
            s2t2 = _sanitize(_process_s2(d["s2_t2"]), "s2_t2", path)
            sclt2 = (d["scl_t2"].astype(np.float32)[..., None]
                     if "scl_t2" in files else np.full((H, W, 1), 4.0, np.float32))
            auxt2 = ((d["aux_t2"].astype(np.float32)
                      if "aux_t2" in files else np.zeros((H, W, A), np.float32))
                     / scale)
        else:
            s1t2, s2t2 = s1.copy(), s2.copy()
            sclt2, auxt2 = scl.copy(), aux.copy()
            change = np.full((H, W), 255, np.int32)   # never learn from copy

        date_t1 = float(d["date_t1"]) if "date_t1" in files else 0.0
        date_t2 = float(d["date_t2"]) if "date_t2" in files else (1.0 if has_t2 else 0.0)

    return (s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
            label, carbon, eco, change, risk,
            np.float32(date_t1), np.float32(date_t2))


HEAD_REQUIRED_KEYS = {
    "seg": ["label"], "change": ["change"], "carbon": ["carbon"],
    "eco": ["eco"], "risk": ["risk"],
    "quality": [], "boundary": [],      # derived on-the-fly: always available
}


def validate_heads_vs_data(files, cfg, sample=5):
    """Pre-flight: ensure every enabled head actually has labels in the NPZs."""
    present = set()
    for p in files[:sample]:
        with np.load(p) as d:
            present |= set(d.files)
    active, warnings_out = [], []
    for h in cfg.active_heads:
        need = HEAD_REQUIRED_KEYS.get(h, [])
        if need and not any(k in present for k in need):
            warnings_out.append(
                f"  [heads] '{h}' enabled but no '{need[0]}' key found in sampled "
                f"NPZs -> head DISABLED (wire real labels, then re-enable).")
        else:
            active.append(h)
    if not active:
        raise NPZSchemaError("No task head has data; check your NPZ schema.")
    for w in warnings_out:
        print(w)
    if tuple(active) != tuple(cfg.active_heads):
        cfg.active_heads = tuple(active)
        print(f"  [heads] active set resolved to: {list(active)}")
    return cfg


# ==============================================================================
# 4. PATCH GENERATOR (class-balanced for training)
# ==============================================================================
def _patch_weight(lab, cfg):
    """Higher sampling weight when rare classes are present."""
    C = cfg.num_classes
    valid = lab != cfg.ignore_label
    if valid.sum() == 0:
        return 0.0
    counts = np.bincount(lab[valid].ravel(), minlength=C).astype(np.float32)
    frac = counts / max(counts.sum(), 1)
    w = 1.0 + cfg.rare_boost * float(frac[list(cfg.rare_classes)].sum())
    return w


def patch_generator(file_list, cfg, training, max_patches, seed=42):
    ps = cfg.patch_size
    stride = cfg.train_stride if training else ps
    ignore = cfg.ignore_label
    invalid = np.array(list(cfg.invalid_scl))
    rng = np.random.default_rng(seed)

    if training:
        # pass 1: collect candidate coords + weights (labels only - cheap)
        pool = []   # (path, i, j, weight)
        for path in file_list:
            data = load_npz(path, cfg)
            label, scl = data[8], data[2]
            h, w = label.shape
            if h < ps or w < ps:
                continue
            for i in range(0, h - ps + 1, stride):
                for j in range(0, w - ps + 1, stride):
                    lab = label[i:i + ps, j:j + ps]
                    if float(np.mean(lab != ignore)) < cfg.min_valid_frac:
                        continue
                    if float(np.mean(np.isin(scl[i:i + ps, j:j + ps], invalid))) \
                            > cfg.max_cloud_frac:
                        continue
                    wgt = _patch_weight(lab, cfg)
                    if wgt > 0:
                        pool.append((path, i, j, wgt))
            del data
        gc.collect()
        if not pool:
            return
        wts = np.array([p[3] for p in pool], np.float64)
        wts /= wts.sum()
        idx = rng.choice(len(pool), size=min(max_patches, len(pool)),
                         replace=False, p=wts)
        chosen = {}
        for k in idx:
            chosen.setdefault(pool[int(k)][0], []).append(pool[int(k)])
        # pass 2: load each scene once, emit chosen patches
        emitted = 0
        for path, items in chosen.items():
            data = load_npz(path, cfg)
            for (_, i, j, _w) in items:
                if emitted >= max_patches:
                    return
                sl = np.s_[i:i + ps, j:j + ps]
                yield tuple(arr[sl] if arr.ndim >= 2 else arr for arr in data)
                emitted += 1
            del data
            gc.collect()
    else:
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
                    lab = label[i:i + ps, j:j + ps]
                    if float(np.mean(lab != ignore)) < cfg.min_valid_frac:
                        continue
                    sl = np.s_[i:i + ps, j:j + ps]
                    yield tuple(arr[sl] if arr.ndim >= 2 else arr for arr in data)
                    emitted += 1
            del data
        gc.collect()


# ==============================================================================
# 5. FEATURE ENGINEERING LAYER
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FeatureEngineer(layers.Layer):
    """S1: [vv, vh, vh-vv, rvi, vv_std, vh_std] (6)
       S2: [10 raw bands, ndvi, ndwi, mndwi, ndmi, ndre, bsi, evi,
            cloud_mask, aux(A)] -> 18 + A channels.
       MASK channel index is SYMBOLIC: S2_MASK_IDX = 17."""

    S1_OUT = 6
    S2_IDX_COUNT = 7
    S2_MASK_IDX = 10 + S2_IDX_COUNT          # = 17  (review F2 fix anchor)
    S2_BASE = 10 + S2_IDX_COUNT + 1          # = 18, before aux

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
C2 = lambda cfg: FeatureEngineer.S2_BASE + cfg.aux_channels   # 18 + A


# ==============================================================================
# 6. NORMALIZATION STATS + CLASS ALPHA (F2 fixed: symbolic mask index)
# ==============================================================================
STATS_PATH = os.path.join(WORK_DIR, "norm_stats_v5.npz")


def compute_stats(file_list, cfg, max_patches=400):
    print("Computing normalization stats + class frequencies...")
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, cfg.aux_channels)
    C = cfg.num_classes
    c1, c2 = C1, C2(cfg)
    n = 0
    s1_sum = np.zeros(c1); s1_sq = np.zeros(c1)
    s2_sum = np.zeros(c2); s2_sq = np.zeros(c2)
    counts = np.zeros(C, np.int64)
    for data in patch_generator(file_list, cfg, training=False,
                                max_patches=max_patches):
        s1, s2, scl, aux = data[0], data[1], data[2], data[3]
        lab = data[8]
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
    # FIX F2: identity stats for the cloud-mask channel at its TRUE index.
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


def load_or_compute_stats(file_list, cfg):
    if os.path.exists(STATS_PATH):
        z = np.load(STATS_PATH)
        print("Loaded cached norm stats.")
        cfg.seg_temperature = float(z["seg_temperature"]) \
            if "seg_temperature" in z.files else 1.0
        return z["s1_mean"], z["s1_var"], z["s2_mean"], z["s2_var"], z["alpha"]
    return compute_stats(file_list, cfg)


# ==============================================================================
# 7. tf.data PIPELINE  (pack -> augment -> unpack), symbolic channel offsets
# ==============================================================================
def output_signature(cfg):
    ps, A = cfg.patch_size, cfg.aux_channels
    F = tf.float32
    return (
        tf.TensorSpec((ps, ps, 4),  F), tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F), tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps, 4),  F), tf.TensorSpec((ps, ps, 10), F),
        tf.TensorSpec((ps, ps, 1),  F), tf.TensorSpec((ps, ps, A),  F),
        tf.TensorSpec((ps, ps),     tf.int32), tf.TensorSpec((ps, ps), F),
        tf.TensorSpec((ps, ps, 2),  F), tf.TensorSpec((ps, ps), tf.int32),
        tf.TensorSpec((ps, ps),     F),
        tf.TensorSpec((),           F), tf.TensorSpec((),  F),   # date_t1/t2
    )


# symbolic offsets into the packed tensor (FIX F7: no more hardcoded slices)
def _packed_layout(cfg):
    A, C = cfg.aux_channels, cfg.num_classes
    names = ["s1", "s2", "scl", "aux", "s1t2", "s2t2", "sclt2", "auxt2",
             "seg", "carbon", "eco", "change", "risk", "dt"]
    sizes = [4, 10, 1, A, 4, 10, 1, A, C, 1, 2, 1, 1, 1]
    off, out = 0, {}
    for n, s in zip(names, sizes):
        out[n] = (off, off + s)
        off += s
    return out, off


def make_dataset(gen_fn, cfg, training, batch_size):
    C, A = cfg.num_classes, cfg.aux_channels
    LAY, _ = _packed_layout(cfg)
    carbon_scale = float(cfg.carbon_scale)
    eco_scale = tf.constant(list(cfg.eco_scale), tf.float32)
    invalid_scl = tf.constant(list(cfg.invalid_scl), tf.int32)
    deep = bool(cfg.deep_supervision)

    def pack(s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
             label, carbon, eco, change, risk, dt1, dt2):
        seg = tf.one_hot(tf.where(label == cfg.ignore_label, -1, label),
                         C, dtype=tf.float32)
        change_f = tf.cast(change, tf.float32)
        change_f = tf.where(change_f == 255.0, -1.0, change_f)
        dt = tf.reshape(tf.abs(dt2 - dt1), (1,))
        return tf.concat([s1, s2, scl, aux, s1t2, s2t2, sclt2, auxt2,
                          seg, carbon[..., None], eco,
                          change_f[..., None], risk[..., None], dt], axis=-1)

    def random_aug(x):
        # geometric (applied to EVERYTHING incl. targets: spatially consistent)
        do_lr = tf.random.uniform(()) > 0.5
        x = tf.cond(do_lr, lambda: tf.image.flip_left_right(x), lambda: x)
        do_ud = tf.random.uniform(()) > 0.5
        x = tf.cond(do_ud, lambda: tf.image.flip_up_down(x), lambda: x)
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        x = tf.image.rot90(x, k)

        if training and cfg.augment:
            # FIX F7: photometric ops hit OPTICAL bands only, via symbolic offsets
            delta = tf.random.uniform((), -cfg.aug_brightness, cfg.aug_brightness)
            contra = tf.random.uniform((), 1 - cfg.aug_contrast, 1 + cfg.aug_contrast)

            def photo(t, lo, hi):
                pre, mid, post = t[..., :lo], t[..., lo:hi], t[..., hi:]
                mid = tf.image.adjust_contrast(tf.image.adjust_brightness(mid, delta), contra)
                return tf.concat([pre, mid, post], axis=-1)

            lo1, hi1 = LAY["s2"]
            lo2, hi2 = LAY["s2t2"]
            x = photo(x, lo1, hi1) if hi1 < lo2 else x
            # optical T2 sits later in the tensor; adjust with a rebuild
            if hi1 < lo2:
                pre = x[..., :lo2]
                mid = x[..., lo2:hi2]
                post = x[..., hi2:]
                mid = tf.image.adjust_contrast(
                    tf.image.adjust_brightness(mid, delta), contra)
                x = tf.concat([pre, mid, post], axis=-1)

            # FIX F8: noise on INPUT bands only (never on packed targets)
            n_inputs_end = LAY["auxt2"][1]
            noise = tf.random.normal(tf.shape(x[..., :n_inputs_end]),
                                     stddev=cfg.aug_noise)
            x = tf.concat([x[..., :n_inputs_end] + noise, x[..., n_inputs_end:]],
                          axis=-1)

            # modality dropout: same modality choice for T1 and T2
            p = cfg.modality_dropout_p
            if p > 0:
                drop_opt = tf.random.uniform(()) < p
                drop_sar = tf.logical_and(tf.random.uniform(()) < p,
                                          tf.logical_not(drop_opt))
                one = tf.ones((), tf.float32)
                zero = tf.zeros((), tf.float32)
                g_opt = tf.where(drop_opt, zero, one)     # zeroes S2 bands T1+T2
                g_sar = tf.where(drop_sar, zero, one)     # zeroes S1 bands T1+T2
                s1a, s1b = LAY["s1"]; s2a, s2b = LAY["s2"]
                t1a, t1b = LAY["s1t2"]; t2a, t2b = LAY["s2t2"]
                parts = [
                    x[..., :s1a],
                    x[..., s1a:s1b] * g_sar,
                    x[..., s1b:s2a],
                    x[..., s2a:s2b] * g_opt,
                    x[..., s2b:t1a],
                    x[..., t1a:t1b] * g_sar,
                    x[..., t1b:t2a],
                    x[..., t2a:t2b] * g_opt,
                    x[..., t2b:],
                ]
                x = tf.concat(parts, axis=-1)
        return x

    def unpack_to_xy(x):
        g = lambda n: x[..., LAY[n][0]:LAY[n][1]]
        s1, s2, scl, aux = g("s1"), g("s2"), g("scl"), g("aux")
        s1t2, s2t2, sclt2, auxt2 = g("s1t2"), g("s2t2"), g("sclt2"), g("auxt2")
        seg, carbon, eco, change, risk = g("seg"), g("carbon"), g("eco"), \
            g("change"), g("risk")
        dt = g("dt")

        carbon_y = tf.where(carbon > -900.0, carbon / carbon_scale, -999.0)
        eco_y = tf.where(eco > -900.0, eco / eco_scale, -999.0)
        risk_y = tf.where(risk > -900.0, risk, -999.0)

        # --- free-supervision targets derived on the fly ---
        # quality: 1 = usable pixel (SCL not invalid). Always has labels.
        scl_i = tf.cast(scl, tf.int32)
        bad = tf.zeros(tf.shape(scl_i), tf.bool)
        for c in list(cfg.invalid_scl):
            bad = tf.logical_or(bad, tf.equal(scl_i, c))
        quality_y = 1.0 - tf.cast(bad, tf.float32)

        # boundary: morphological gradient of seg one-hot (valid only where seg is)
        valid_seg = tf.cast(tf.reduce_sum(seg, -1, keepdims=True) > 0, tf.float32)
        mx = tf.nn.max_pool2d(seg, 3, 1, "SAME")
        mn = -tf.nn.max_pool2d(-seg, 3, 1, "SAME")
        edge = tf.reduce_max(tf.clip_by_value(mx - mn, 0.0, 1.0), -1,
                             keepdims=True)
        boundary_y = tf.where(valid_seg > 0, edge, -1.0)

        inputs = {"s1_input": s1, "s2_input": s2, "scl_input": scl,
                  "aux_input": aux, "s1_t2_input": s1t2, "s2_t2_input": s2t2,
                  "scl_t2_input": sclt2, "aux_t2_input": auxt2, "dt_input": dt}
        targets = {"seg_head": seg, "carbon_head": carbon_y, "eco_head": eco_y,
                   "change_head": change, "risk_head": risk_y,
                   "quality_head": quality_y, "boundary_head": boundary_y}
        if deep:
            targets["seg_deep_head"] = tf.nn.avg_pool2d(seg, 4, 4, "VALID")
        return inputs, targets

    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=output_signature(cfg))
    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(cfg.shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(random_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(unpack_to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(cfg.deterministic)
    ds = ds.with_options(opts)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 8. SWIN-V2 COMPONENTS  (all buffers are non-trainable WEIGHTS: F1 fixed)
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
    """Swin-V2 attention: scaled-cosine attention (learnable per-head logit
    scale, clamped) + log-spaced Continuous Position Bias via meta-MLP.
    All coordinate buffers stored as non-trainable weights (Keras-3 safe)."""

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
        # learnable logit scale: init log(10), clamped to <= log(100)
        self.logit_scale = self.add_weight(
            name="logit_scale", shape=(self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(math.log(10.0)),
            trainable=True)
        # CPB meta-network on log-spaced relative coordinates
        self.cpb1 = layers.Dense(self.cpb_hidden, activation="relu", name="cpb1")
        self.cpb2 = layers.Dense(self.num_heads, use_bias=False, name="cpb2")

        # ---- buffers as non-trainable WEIGHTS (FIX F1) ----
        # pairwise relative coords for each of the N x N token pairs, log-spaced
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws),
                                      indexing="ij"))            # (2, ws, ws)
        cf = coords.reshape(2, -1).astype(np.float32)            # (2, N)
        rel = cf[:, :, None] - cf[:, None, :]                    # (2, N, N)
        rel = rel.transpose(1, 2, 0)                             # (N, N, 2)
        rel = rel / max(ws - 1, 1) * 8.0                         # normalize to [-8, 8]
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

        # cosine attention, fp32 for stability
        qn = tf.math.l2_normalize(tf.cast(q, tf.float32), axis=-1)
        kn = tf.math.l2_normalize(tf.cast(k, tf.float32), axis=-1)
        scale = tf.exp(tf.minimum(self.logit_scale, math.log(100.0)))
        attn = tf.matmul(qn, kn, transpose_b=True) * tf.cast(scale, tf.float32)

        # continuous position bias, squashed to [-16, 16] like Swin-V2
        bias = self.cpb2(self.cpb1(self.rel_coords))             # (N, N, heads)
        bias = 16.0 * tf.sigmoid(bias)
        attn += tf.transpose(bias, [2, 0, 1])[None]

        if mask is not None:                                     # (nW, 1, N, N)
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
    """Res-post-norm Swin-V2 block:
       x = norm1(x + dp(attn(x)));  x = norm2(x + dp(mlp(x)))."""

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
            layers.Dense(hidden, activation="gelu"),
            layers.Dropout(self.drop),
            layers.Dense(self.dim),
            layers.Dropout(self.drop)], name="mlp")
        # shifted-window mask as non-trainable WEIGHT (FIX F1, same bug class)
        if self.shift_size > 0:
            mask_np = self._build_mask_np()                      # (nW, 1, N, N)
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
            h = tf.roll(h, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        w = window_partition(h, self.window_size)
        w = self.attn(w, mask=self.attn_mask, training=training)
        h = window_reverse(w, self.window_size, self.H, self.W)
        if self.shift_size > 0:
            h = tf.roll(h, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        x = self.norm1(x + self.dp1(h, training=training))       # post-norm
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
    """Cheap fusion for high-res levels: opt + tanh(gate) * W[opt, sar].
    Gate initialized at 0 -> identity at init (stable)."""

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
    """Bidirectional token cross-attention (deep levels only, small N)."""

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


@tf.keras.utils.register_keras_serializable(package="mangrove")
class TimeEmbedding(layers.Layer):
    """Sinusoidal embedding of the absolute time gap (years) + MLP."""

    def __init__(self, dim, freqs=16, max_years=20.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.freqs = int(freqs)
        self.max_years = float(max_years)

    def build(self, input_shape):
        f = np.arange(self.freqs, dtype=np.float32) / self.freqs
        self.freq_table = self.add_weight(
            name="freq_table", shape=(self.freqs,), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(
                1.0 / (10000.0 ** f)), trainable=False)
        self.d1 = layers.Dense(self.dim, activation="gelu", name="t1")
        self.d2 = layers.Dense(self.dim, name="t2")
        super().build(input_shape)

    def call(self, dt):
        dt = tf.cast(tf.reshape(dt, [-1, 1]), tf.float32) / self.max_years
        ang = dt * self.freq_table * 2.0 * math.pi
        emb = tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)
        return self.d2(self.d1(emb))                               # (B, dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "freqs": self.freqs,
                    "max_years": self.max_years})
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class FiLM(layers.Layer):
    """Feature-wise modulation of a feature map by a conditioning vector."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)

    def build(self, input_shape):
        self.dense = layers.Dense(2 * self.channels, name="film")
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        gb = self.dense(cond)                                      # (B, 2C)
        g, b = tf.split(gb, 2, axis=-1)
        g = tf.reshape(g, [-1, 1, 1, self.channels])
        b = tf.reshape(b, [-1, 1, 1, self.channels])
        return x * (1.0 + tf.cast(g, x.dtype)) + tf.cast(b, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg["channels"] = self.channels
        return cfg


# ==============================================================================
# 10. ENCODERS / DECODER BLOCKS
# ==============================================================================
def _heads_for(dim):
    return max(2, min(12, dim // 16))


def make_swin_encoder(prefix, dims, depths, cfg, dp_rates):
    """Returns apply(x) -> (bottleneck, skips). dp_rates consumed in order."""
    ws = cfg.window_size
    if True:  # hybrid CNN stem (more stable than a single strided conv)
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
# 11. FULL MODEL
# ==============================================================================
def build_model(cfg, s1_stats, s2_stats):
    C, ps, A = cfg.num_classes, cfg.patch_size, cfg.aux_channels
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)
    ws = cfg.window_size
    active = set(cfg.active_heads)

    s1_in = layers.Input((ps, ps, 4), name="s1_input")
    s2_in = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1), name="scl_input")
    aux_in = layers.Input((ps, ps, A), name="aux_input")
    s1t2_in = layers.Input((ps, ps, 4), name="s1_t2_input")
    s2t2_in = layers.Input((ps, ps, 10), name="s2_t2_input")
    sclt2_in = layers.Input((ps, ps, 1), name="scl_t2_input")
    auxt2_in = layers.Input((ps, ps, A), name="aux_t2_input")
    dt_in = layers.Input((1,), name="dt_input")

    # shared feature engineering (siamese: ONE layer instance for both dates)
    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    f1, f2 = fe([s1_in, s2_in, scl_in, aux_in])
    g1, g2 = fe([s1t2_in, s2t2_in, sclt2_in, auxt2_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]), variance=list(s1_stats[1]),
                              name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]), variance=list(s2_stats[1]),
                              name="s2_norm")
    f1 = n1(f1); f2 = n2(f2)
    g1 = n1(g1); g2 = n2(g2)

    # linear stochastic-depth schedule across all encoder blocks
    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])

    b1, sk1 = enc_s1(f1)                       # SAR T1
    b2, sk2 = enc_s2(f2)                       # OPT T1
    c1, _ = enc_s1(g1)                         # SAR T2 (shared weights)
    c2, sk2_t2 = enc_s2(g2)                    # OPT T2

    # per-level fusion (opt stream fused with sar context; siamese-shared)
    fused_t1, fused_t2 = [], []
    sar_sk_t2 = []
    for i, d in enumerate(dims):
        s_t1, s_t2 = sk1[i], None
        o_t1, o_t2 = sk2[i], sk2_t2[i]
        # project sar to opt dims where needed
        if s_t1.shape[-1] != d:
            s_t1 = layers.Conv2D(d, 1, name=f"sar_proj{i}")(s_t1)
        s_t2 = layers.Conv2D(d, 1, name=f"sar_t2_proj{i}")(sk1[i]) \
            if sk1[i].shape[-1] != d else sk1[i]
        if i in tuple(cfg.cross_attn_levels):
            fuse = CrossAttentionFusion(d, cfg.fusion_heads,
                                        name=f"cross_attn_fusion{i}")
        else:
            fuse = GatedFusion(d, name=f"gated_fusion{i}")
        fused_t1.append(fuse([o_t1, s_t1]))
        fused_t2.append(fuse([o_t2, s_t2]))    # same instance: siamese
        sar_sk_t2.append(s_t2)
    fused_b, fused_b_t2 = fused_t1[-1], fused_t2[-1]

    # ---- main decoder (T1) ----
    x = fused_b
    deep_out = None
    for i, wd in enumerate(cfg.dec_widths):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"dec_up{i}")(x)
        li = len(dims) - 2 - i                 # fuse levels 2,1,0 as skips
        if li >= 0:
            x = layers.concatenate([x, fused_t1[li], sk1[li]], name=f"dec_cat{i}")
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

    # ---- time-aware change decoder: per-level diffs + FiLM(dt) ----
    temb = TimeEmbedding(cfg.time_embed_dim, name="time_embed")(dt_in)
    diffs = []
    for i in range(len(dims)):
        diffs.append(layers.concatenate(
            [AbsDiff(name=f"diff_opt{i}")([fused_t1[i], fused_t2[i]]),
             AbsDiff(name=f"diff_sar{i}")([sk1[i], sar_sk_t2[i]])],
            name=f"diff_cat{i}"))
    d = conv_block(diffs[-1], cfg.change_widths[0], use_se=cfg.use_se,
                   name="change_bottleneck")
    d = FiLM(cfg.change_widths[0], name="film_bottleneck")([d, temb])
    for i, wd in enumerate(cfg.change_widths[1:]):
        d = layers.UpSampling2D(interpolation="bilinear", name=f"change_up{i}")(d)
        li = len(dims) - 2 - i
        if li >= 0:
            d = layers.concatenate([d, diffs[li]], name=f"change_cat{i}")
        d = conv_block(d, wd, use_se=cfg.use_se, name=f"change_c{i}")
        d = FiLM(wd, name=f"film_{i}")([d, temb])

    # ---- heads (pluggable) ----
    outputs = {}
    if "seg" in active:
        outputs["seg_head"] = layers.Activation(
            "softmax", dtype="float32", name="seg_head")(
            layers.Conv2D(C, 1, padding="same", name="seg_logits")(x))
    if cfg.deep_supervision and deep_out is not None:
        outputs["seg_deep_head"] = deep_out
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
    if "change" in active:
        outputs["change_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="change_head")(
            layers.Conv2D(1, 1, padding="same", name="change_logits")(d))
    if "quality" in active:
        outputs["quality_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="quality_head")(
            layers.Conv2D(1, 1, padding="same", name="quality_logits")(x))
    if "boundary" in active:
        outputs["boundary_head"] = layers.Activation(
            "sigmoid", dtype="float32", name="boundary_head")(
            layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x))

    return models.Model(
        inputs=[s1_in, s2_in, scl_in, aux_in,
                s1t2_in, s2t2_in, sclt2_in, auxt2_in, dt_in],
        outputs=outputs, name=f"MangroveEcoMonitor_v{VERSION}")


# ==============================================================================
# 12. LOSSES & METRICS
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="mangrove")
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=1.0,
                 label_smoothing=0.0, **kwargs):
        super().__init__(name="focal_dice", **kwargs)
        self.alpha = None if alpha is None else [float(a) for a in alpha]
        self.gamma = float(gamma)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        valid = tf.cast(tf.reduce_sum(y_true, -1, keepdims=True) > 0, tf.float32)
        if self.label_smoothing > 0:
            C = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / C
        a = tf.ones_like(y_pred) if self.alpha is None \
            else tf.constant(self.alpha, tf.float32)
        ce = -y_true * tf.math.log(y_pred)
        fl = tf.reduce_sum(a * tf.pow(1.0 - y_pred, self.gamma) * ce,
                           axis=-1, keepdims=True) * valid
        focal = tf.reduce_sum(fl, axis=[1, 2, 3]) / \
            (tf.reduce_sum(valid, axis=[1, 2, 3]) + eps)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
        dice = tf.reduce_mean(1.0 - (2.0 * inter + 1.0) / (denom + 1.0), axis=-1)
        return self.focal_weight * focal + self.dice_weight * dice

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma,
                    "focal_weight": self.focal_weight,
                    "dice_weight": self.dice_weight,
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
class MaskedHuber(tf.keras.losses.Loss):
    """Huber on valid pixels (more robust than MSE for noisy eco/risk labels)."""
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(name="masked_huber", **kwargs)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        mask = tf.cast(y > -900.0, tf.float32)
        err = tf.abs(tf.where(mask > 0, y, 0.0) - p)
        hub = tf.where(err <= self.delta, 0.5 * err ** 2,
                       self.delta * (err - 0.5 * self.delta))
        return tf.reduce_sum(hub * mask) / (tf.reduce_sum(mask) + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg["delta"] = self.delta
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MaskedBinaryFocal(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(name="masked_binary_focal", **kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
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
class MaskedBCE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(name="masked_bce", **kwargs)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
        mask = tf.cast(y >= 0.0, tf.float32)
        y = tf.clip_by_value(y, 0.0, 1.0)
        bce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-6)


@tf.keras.utils.register_keras_serializable(package="mangrove")
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.total_cm = self.add_weight(
            name="cm", shape=(self.num_classes, self.num_classes),
            initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.reshape(tf.reduce_sum(y_true, -1) > 0, [-1])
        yt = tf.boolean_mask(
            tf.reshape(tf.argmax(y_true, -1, output_type=tf.int32), [-1]), valid)
        yp = tf.boolean_mask(
            tf.reshape(tf.argmax(y_pred, -1, output_type=tf.int32), [-1]), valid)
        cm = tf.math.confusion_matrix(yt, yp, num_classes=self.num_classes,
                                      dtype=tf.float32)
        self.total_cm.assign_add(cm)

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
        cfg = super().get_config()
        cfg["num_classes"] = self.num_classes
        return cfg


@tf.keras.utils.register_keras_serializable(package="mangrove")
class CarbonMAE(tf.keras.metrics.Metric):
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


@tf.keras.utils.register_keras_serializable(package="mangrove")
class ChangeF1(tf.keras.metrics.Metric):
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
        self.fp.assign_add(tf.reduce_sum((1 - y) * p_bin * mask))
        self.fn.assign_add(tf.reduce_sum(y * (1 - p_bin) * mask))

    def result(self):
        prec = self.tp / (self.tp + self.fp + 1e-6)
        rec = self.tp / (self.tp + self.fn + 1e-6)
        return 2 * prec * rec / (prec + rec + 1e-6)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["threshold"] = self.threshold
        return cfg


# ==============================================================================
# 13. TRAINER: custom train_step, LEARNABLE task weighting, single grad path
# ==============================================================================
HEAD_LOSS_MAP = {
    "seg": "focal_dice", "change": "binary_focal", "quality": "bce",
    "boundary": "bce", "carbon": "carbon_nll", "eco": "huber", "risk": "huber",
}


class EcoMonitorTrainer(tf.keras.Model):
    """Wraps the functional net. Owns:
       - per-task losses + learnable log-variance task weighting (Kendall 2018)
       - explicit metrics per active head
       Gradient accumulation: NATIVE optimizer path only (FIX F4). If
       cfg.grad_accum > 1 we require optimizer gradient_accumulation_steps
       support and never run a second, custom accumulator."""

    def __init__(self, cfg, net, alpha, **kwargs):
        super().__init__(**kwargs)
        self.cfg_obj = cfg
        self.net = net
        self.active = list(cfg.active_heads)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.loss_fns = {}
        for h in self.active:
            kind = HEAD_LOSS_MAP[h]
            if kind == "focal_dice":
                self.loss_fns[h] = FocalDiceLoss(
                    alpha=alpha, gamma=cfg.focal_gamma,
                    label_smoothing=cfg.label_smoothing)
            elif kind == "binary_focal":
                self.loss_fns[h] = MaskedBinaryFocal()
            elif kind == "bce":
                self.loss_fns[h] = MaskedBCE()
            elif kind == "carbon_nll":
                self.loss_fns[h] = CarbonNLLLoss()
            else:
                self.loss_fns[h] = MaskedHuber()
        if cfg.deep_supervision:
            self.loss_fns["seg_deep"] = FocalDiceLoss(
                alpha=alpha, gamma=cfg.focal_gamma,
                label_smoothing=cfg.label_smoothing)

        # learnable per-task log-variances (trainable weights of the wrapper)
        self.use_learned_w = cfg.task_weighting == "learned"
        self.task_log_vars = {}
        for name in list(self.loss_fns.keys()):
            self.task_log_vars[name] = self.add_weight(
                name=f"task_logvar_{name}", shape=(), dtype=tf.float32,
                initializer="zeros", trainable=self.use_learned_w)

        # metrics
        self.per_head_loss = {n: tf.keras.metrics.Mean(name=f"{n}_loss")
                              for n in self.loss_fns}
        self.metric_objs = {}
        if "seg" in self.active:
            self.metric_objs["seg"] = MeanIoU(cfg.num_classes)
        if "change" in self.active:
            self.metric_objs["change"] = ChangeF1()
        if "carbon" in self.active:
            self.metric_objs["carbon"] = CarbonMAE()
        for h in ("quality", "boundary"):
            if h in self.active:
                self.metric_objs[h] = tf.keras.metrics.Mean(name=f"{h}_acc")

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    @property
    def metrics(self):
        ms = [self.loss_tracker] + list(self.per_head_loss.values()) + \
             list(self.metric_objs.values())
        return ms

    def _per_task_losses(self, y, p):
        out = {}
        for name, fn in self.loss_fns.items():
            key = name if name != "seg_deep" else "seg_deep_head"
            tgt = y.get(key)
            prd = p.get(key if key in p else None)
            if tgt is None or prd is None:
                continue
            out[name] = fn(tgt, prd)
        return out

    def _total(self, per_task):
        total = 0.0
        for name, l in per_task.items():
            if self.use_learned_w:
                lv = self.task_log_vars[name]
                w_extra = self.cfg_obj.deep_supervision_weight \
                    if name == "seg_deep" else 1.0
                total = total + w_extra * (l * tf.exp(-lv) + lv)
            else:
                w = self.cfg_obj.fixed_loss_weights.get(
                    name.replace("_deep", ""), 1.0)
                if name == "seg_deep":
                    w = self.cfg_obj.deep_supervision_weight
                total = total + w * l
        return total

    def _update_metrics(self, per_task, y, p, total):
        self.loss_tracker.update_state(total)
        for name, l in per_task.items():
            self.per_head_loss[name].update_state(l)
        if "seg" in self.metric_objs and "seg_head" in p:
            self.metric_objs["seg"].update_state(y["seg_head"], p["seg_head"])
        if "change" in self.metric_objs and "change_head" in p:
            self.metric_objs["change"].update_state(y["change_head"],
                                                    p["change_head"])
        if "carbon" in self.metric_objs and "carbon_head" in p:
            self.metric_objs["carbon"].update_state(y["carbon_head"],
                                                    p["carbon_head"])
        for h in ("quality", "boundary"):
            key = f"{h}_head"
            if h in self.metric_objs and key in p:
                valid = tf.cast(y[key] >= 0, tf.float32)
                acc = tf.cast(tf.equal(tf.cast(p[key] > 0.5, tf.float32),
                                       y[key]), tf.float32)
                denom = tf.reduce_sum(valid)
                val = tf.reduce_sum(acc * valid) / (denom + 1e-6)
                self.metric_objs[h].update_state(val)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            p = self.net(x, training=True)
            per_task = self._per_task_losses(y, p)
            total = self._total(per_task) + sum(self.net.losses)
        # mixed-precision safe scaling (Keras-3 LossScaleOptimizer path)
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
    """Single accumulation mechanism: native optimizer support or hard error."""
    kwargs = dict(learning_rate=lr, weight_decay=cfg.weight_decay,
                  clipnorm=cfg.grad_clip)
    if cfg.grad_accum > 1:
        try:
            return optimizers.AdamW(
                gradient_accumulation_steps=int(cfg.grad_accum), **kwargs)
        except TypeError as e:
            raise RuntimeError(
                "cfg.grad_accum > 1 requires Keras>=3.4 native "
                "gradient_accumulation_steps. Either upgrade TF/Keras or set "
                "grad_accum=1 (v5 does NOT run a second custom accumulator "
                "- see FIX F4).") from e
    return optimizers.AdamW(**kwargs)


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
# 15. LR SCHEDULE / PLOTS / EVAL
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
    C = cfg.num_classes
    cm = np.zeros((C, C), np.int64)
    for inputs, targets in val_ds:
        out = net(inputs, training=False)
        p = _preds_to_dict(net, out)["seg_head"].numpy().argmax(-1)
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
    metrics = {n: {"iou": float(iou[i]), "pixels": int(cm[i].sum())}
               for i, n in enumerate(CLASS_NAMES)}
    metrics["mIoU"] = float(np.nanmean(iou))
    with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return cm, iou


def calibrate_temperature(net, val_ds, cfg, max_batches=10):
    """Post-hoc temperature scaling on validation seg probs (grid search)."""
    if "seg" not in cfg.active_heads:
        return 1.0
    probs, labels = [], []
    for i, (inputs, targets) in enumerate(val_ds):
        if i >= max_batches:
            break
        out = _preds_to_dict(net, net(inputs, training=False))
        p = out["seg_head"].numpy().reshape(-1, cfg.num_classes)
        y = targets["seg_head"].numpy().reshape(-1, cfg.num_classes)
        v = y.sum(-1) > 0
        probs.append(p[v]); labels.append(y[v].argmax(-1))
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
    print(f"  [calibration] seg temperature = {best_t:.2f} "
          f"(NLL {best_nll:.4f})")
    cfg.seg_temperature = float(best_t)
    # persist alongside norm stats
    if os.path.exists(STATS_PATH):
        z = dict(np.load(STATS_PATH))
        z["seg_temperature"] = np.float32(best_t)
        np.savez(STATS_PATH, **z)
    return best_t


def plot_training_history(history):
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
    heads = ["seg_head", "change_head", "quality_head", "boundary_head",
             "carbon_head", "risk_head"]
    have = [h for h in heads if h in [o for o in net.output_names]]
    ncols = 3 + len([h for h in have if h != "seg_head"])
    fig, axes = plt.subplots(n_samples, ncols, figsize=(4 * ncols,
                                                        4 * n_samples))
    axes = np.atleast_2d(axes)
    titles = ["S2 RGB", "GT Seg", "Pred Seg"] + \
             [h.replace("_head", "") for h in have if h != "seg_head"]
    for a, t in zip(axes[0], titles):
        a.set_title(t, fontsize=10)
    count = 0
    for inputs, targets in val_ds.take(2):
        preds = _preds_to_dict(net, net(inputs, training=False))
        preds = {k: v.numpy() for k, v in preds.items()}
        bs = len(targets["seg_head"])
        for i in range(min(bs, n_samples - count)):
            s2 = inputs["s2_input"][i].numpy()
            axes[count, 0].imshow(_stretch(s2[:, :, [2, 1, 0]]))
            y = targets["seg_head"][i].numpy().argmax(-1)
            axes[count, 1].imshow(y, cmap=cmap, vmin=0, vmax=4)
            if "seg_head" in preds:
                axes[count, 2].imshow(preds["seg_head"][i].argmax(-1),
                                      cmap=cmap, vmin=0, vmax=4)
            col = 3
            for h in have:
                if h == "seg_head":
                    continue
                arr = preds[h][i]
                if h == "carbon_head":
                    arr = arr[..., 0] * cfg.carbon_scale
                    axes[count, col].imshow(arr, cmap="YlGn")
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
# 16. SELF-SUPERVISED PRETRAINING (SimMIM-style masked reconstruction)
# ==============================================================================
def build_pretrain_model(cfg, s1_stats, s2_stats):
    """Masked-block reconstruction of S1 feats + S2 bands from the fused
    bottleneck. Encoder layer names match the main model -> by-name transfer."""
    ps, A = cfg.patch_size, cfg.aux_channels
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)

    s1_in = layers.Input((ps, ps, 4), name="s1_input")
    s2_in = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1), name="scl_input")
    aux_in = layers.Input((ps, ps, A), name="aux_input")

    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    f1, f2 = fe([s1_in, s2_in, scl_in, aux_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]),
                              variance=list(s1_stats[1]), name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]),
                              variance=list(s2_stats[1]), name="s2_norm")
    f1n, f2n = n1(f1), n2(f2)

    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])
    b1, _ = enc_s1(f1n)
    b2, _ = enc_s2(f2n)

    if b1.shape[-1] != dims[-1]:
        b1 = layers.Conv2D(dims[-1], 1, name="sar_proj3")(b1)
    fuse = CrossAttentionFusion(dims[-1], cfg.fusion_heads,
                                name="cross_attn_fusion3")
    z = fuse([b2, b1])

    x = z
    for i, wd in enumerate([192, 96]):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"pt_up{i}")(x)
        x = conv_block(x, wd, use_se=False, name=f"pt_c{i}")
    rec = layers.Conv2D(C1 + 10, 1, padding="same", dtype="float32",
                        name="recon_head")(x)
    return models.Model([s1_in, s2_in, scl_in, aux_in], rec,
                        name="MangrovePretrainer_v5"), (f1, f2, f1n, f2n)


def _masked_block(x_raw, ratio, block):
    """Zero random square blocks of an image tensor. Returns (x_masked, keep)."""
    ps = tf.shape(x_raw)[1]
    g = tf.maximum(ps // block, 1)
    cells = tf.random.uniform((tf.shape(x_raw)[0], g, g, 1)) < ratio
    mask = tf.image.resize(tf.cast(cells, tf.float32), [ps, ps],
                           method="nearest")                    # 1 = masked
    keep = 1.0 - mask
    return x_raw * keep, keep


def pretrain(cfg=CFG):
    """SimMIM-style: reconstruct normalized S1 feats + raw S2 bands at 1/4 res
    under heavy block masking. No labels needed."""
    print("\n=== v5 SELF-SUPERVISED PRETRAINING (SimMIM-style) ===")
    files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.npz")))
    assert files, "pretrain: set cfg.train_dir to a folder of NPZ scenes"
    s1m, s1v, s2m, s2v, _ = load_or_compute_stats(files, cfg)

    def gen():
        yield from patch_generator(files, cfg, False, cfg.max_train_patches)

    ps = cfg.patch_size
    sig = output_signature(cfg)

    def to_inputs(*vals):
        s1, s2, scl, aux = vals[0], vals[1], vals[2], vals[3]
        return (s1, s2, scl, aux)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig) \
        .map(lambda *v: to_inputs(*v), num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(cfg.shuffle_buffer) \
        .batch(cfg.batch_per_replica * N_REPLICAS, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    s1_mean = tf.constant(s1m, tf.float32)
    s1_std = tf.sqrt(tf.constant(s1v, tf.float32))

    with STRATEGY.scope():
        pt, refs = build_pretrain_model(cfg, (s1m, s1v), (s2m, s2v))
        f1_raw, f2_raw, f1n, f2n = refs
        opt = optimizers.AdamW(learning_rate=cfg.pretrain_lr,
                               weight_decay=cfg.weight_decay)

        @tf.function
        def step(s1, s2, scl, aux):
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block(s1, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                s2_m, keep2 = _masked_block(s2, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                rec = pt({"s1_input": s1_m, "s2_input": s2_m,
                          "scl_input": scl, "aux_input": aux}, training=True)
                # targets at 1/4 res: normalized s1 feats + raw s2 bands
                t_s1 = (feats_from(s1, scl) - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1, 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2, 4, 4, "VALID")
                m = tf.concat([tf.repeat(m1, C1, -1), tf.repeat(m2, 10, -1)], -1)
                m = tf.cast(m > 0.5, tf.float32)               # 1 = masked cell
                loss = tf.reduce_sum(tf.abs(rec - tgt_q) * m) / \
                    (tf.reduce_sum(m) + 1e-6)
            grads = tape.gradient(loss, pt.trainable_variables)
            opt.apply_gradients([(g, v) for g, v in
                                 zip(grads, pt.trainable_variables)
                                 if g is not None])
            return loss

        # small helper: recompute S1 features OUTSIDE the model for targets
        def feats_from(s1, scl):
            vv, vh, vv_std, vh_std = tf.split(s1, 4, axis=-1)
            lin_vv = tf.pow(10.0, vv / 10.0)
            lin_vh = tf.pow(10.0, vh / 10.0)
            rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
            return tf.concat([vv, vh, vh - vv, rvi, vv_std, vh_std], -1)

        # make feats_from visible to the tf.function closure
        step.__globals__["feats_from"] = feats_from if hasattr(step, "__globals__") else None

        step_fn = step
        # NOTE: nested def ordering - rebuild step with helper bound
        @tf.function
        def train_one(s1, s2, scl, aux):
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block(s1, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                s2_m, keep2 = _masked_block(s2, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                rec = pt({"s1_input": s1_m, "s2_input": s2_m,
                          "scl_input": scl, "aux_input": aux}, training=True)
                vv, vh, vvs, vhs = tf.split(s1, 4, axis=-1)
                lin_vv = tf.pow(10.0, vv / 10.0)
                lin_vh = tf.pow(10.0, vh / 10.0)
                rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
                t_s1 = tf.concat([vv, vh, vh - vv, rvi, vvs, vhs], -1)
                t_s1 = (t_s1 - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1, 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2, 4, 4, "VALID")
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
            for s1, s2, scl, aux in ds:
                losses.append(float(train_one(s1, s2, scl, aux)))
            print(f"  pretrain {ep + 1}/{cfg.pretrain_epochs} "
                  f"loss={np.mean(losses):.4f} ({time.time() - t0:.0f}s)")

        out = os.path.join(WORK_DIR, "pretrain_encoder_v5.weights.h5")
        pt.save_weights(out)
        print(f"Pretrained encoder weights saved: {out}")
        print("Fine-tune with: cfg.pretrain_weights = "
              f"'{out}' before train()")
        return out


# ==============================================================================
# 17. BATCH PREFLIGHT + OOM-RESILIENT TRAINING
# ==============================================================================
def preflight_batch(trainer, ds):
    """One train_on_batch to verify memory; returns True/False."""
    try:
        x, y = next(iter(ds))
        trainer.train_on_batch(x, y)
        return True
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        print(f"  [preflight] OOM at current batch: {str(e)[:160]}")
        return False


def train(cfg=CFG):
    print(f"\n=== MANGROVE ECO-MONITOR v{VERSION} - TRAINING ===\n")
    train_files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.npz")))
    val_files = sorted(glob.glob(os.path.join(cfg.val_dir, "*.npz")))
    assert train_files and val_files, "Set cfg.train_dir / cfg.val_dir"
    validate_heads_vs_data(train_files, cfg)
    print(f"train scenes: {len(train_files)} | val scenes: {len(val_files)}")

    s1m, s1v, s2m, s2v, alpha = load_or_compute_stats(train_files, cfg)

    if cfg.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    attempt, oom_left = 0, cfg.oom_retries
    while True:
        batch = cfg.batch_per_replica * N_REPLICAS
        ds_train = make_dataset(
            lambda: patch_generator(train_files, cfg, True,
                                    cfg.max_train_patches), cfg, True, batch)
        ds_val = make_dataset(
            lambda: patch_generator(val_files, cfg, False,
                                    cfg.max_val_patches), cfg, False, batch)

        with STRATEGY.scope():
            net = build_model(cfg, (s1m, s1v), (s2m, s2v))
            if cfg.pretrain_weights and os.path.exists(cfg.pretrain_weights):
                try:
                    net.load_weights(cfg.pretrain_weights, by_name=True,
                                     skip_mismatch=True)
                    print(f"  [pretrain] loaded encoder weights "
                          f"(by_name, skip_mismatch) from {cfg.pretrain_weights}")
                except Exception as e:
                    print(f"  [pretrain] weight load failed ({e}); training "
                          f"from scratch.")
            trainer = EcoMonitorTrainer(cfg, net, alpha)
            est_steps = max(1, cfg.max_train_patches // batch)
            lr_sched = WarmUpCosineDecay(
                cfg.lr, cfg.epochs * est_steps,
                cfg.lr_warmup_epochs * est_steps)
            trainer.compile(optimizer=make_optimizer(cfg, lr_sched))

        if cfg.auto_batch and attempt == 0:
            ok = preflight_batch(trainer, ds_train)
            while not ok and cfg.batch_per_replica > 1:
                cfg.batch_per_replica //= 2
                tf.keras.backend.clear_session()
                gc.collect()
                batch = cfg.batch_per_replica * N_REPLICAS
                ds_train = make_dataset(
                    lambda: patch_generator(train_files, cfg, True,
                                            cfg.max_train_patches),
                    cfg, True, batch)
                with STRATEGY.scope():
                    net = build_model(cfg, (s1m, s1v), (s2m, s2v))
                    trainer = EcoMonitorTrainer(cfg, net, alpha)
                    trainer.compile(optimizer=make_optimizer(cfg, lr_sched))
                ok = preflight_batch(trainer, ds_train)
            print(f"  [auto_batch] settled on batch/replica="
                  f"{cfg.batch_per_replica}")

        monitor = "val_seg_mean_iou" if "seg" in cfg.active_heads else "val_loss"
        mode = "max" if "iou" in monitor else "min"
        ema = EMACallback(cfg.ema_decay)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=15,
                                             restore_best_weights=True,
                                             verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(WORK_DIR, "best_v5.weights.h5"), monitor,
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
        except (tf.errors.ResourceExhaustedError,) as e:
            if oom_left <= 0 or cfg.batch_per_replica <= 1:
                raise
            oom_left -= 1
            attempt += 1
            cfg.batch_per_replica = max(1, cfg.batch_per_replica // 2)
            print(f"  [OOM-RETRY] halving batch to {cfg.batch_per_replica}/replica "
                  f"and restarting from backup ({oom_left} retries left). "
                  f"Detail: {str(e)[:120]}")
            tf.keras.backend.clear_session()
            gc.collect()

    # ---- finalize on EMA weights ----
    print("\nSwapping in EMA weights for final eval + export...")
    ema.swap_in()
    calibrate_temperature(net, ds_val, cfg)
    model_path = os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras")
    net.save(model_path)
    cfg.save(os.path.join(WORK_DIR, "config_v5.json"))
    plot_training_history(history)
    if "seg" in cfg.active_heads:
        evaluate_per_class(net, ds_val, cfg)
    visualize_predictions(net, ds_val, cfg)
    print(f"\nModel saved: {model_path}")
    print(f"Config saved: {os.path.join(WORK_DIR, 'config_v5.json')}")
    return net


# ==============================================================================
# 18. TILED INFERENCE (Hann blending fixed, TTA fixed, MC-dropout, SAR-only)
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
    """FIX F10: proper centre-peaked window (v4's was inverted)."""
    h = np.hanning(ps).astype(np.float32)
    h = np.maximum(h, floor)
    return np.outer(h, h)


def predict_scene(net, s1, s2, scl=None, aux=None, t2=None, cfg=CFG,
                  batch_size=None, tta=None, overlap=None, sar_only=False,
                  dt_years=None):
    """Production inference: Hann-blended tiling, correct TTA averaging
    (FIX F9), optional MC-dropout epistemic variance, SAR-only fallback."""
    batch_size = batch_size or cfg.infer_batch
    tta = cfg.infer_tta if tta is None else tta
    overlap = cfg.infer_overlap if overlap is None else overlap
    ps, C, A = cfg.patch_size, cfg.num_classes, cfg.aux_channels
    h, w = s2.shape[:2]
    temp = float(getattr(cfg, "seg_temperature", 1.0))

    if s1.shape[-1] == 2:
        s1 = np.concatenate([s1, np.zeros_like(s1)], -1)
    scl = np.full((h, w, 1), 4.0, np.float32) if scl is None \
        else scl[..., None].astype(np.float32)
    aux = np.zeros((h, w, A), np.float32) if aux is None \
        else aux.astype(np.float32)
    if t2 is None:
        s1t2, s2t2, sclt2, auxt2 = s1, s2, scl, aux
        dt_val = 0.0
    else:
        s1t2 = t2["s1"]
        if s1t2.shape[-1] == 2:
            s1t2 = np.concatenate([s1t2, np.zeros_like(s1t2)], -1)
        s2t2 = t2["s2"]
        sclt2 = t2["scl"][..., None].astype(np.float32) \
            if t2.get("scl") is not None else scl
        auxt2 = t2["aux"].astype(np.float32) if t2.get("aux") is not None else aux
        dt_val = float(dt_years if dt_years is not None else 1.0)

    if sar_only:   # optical branch zeroed: model was trained for this (U8)
        s2 = np.zeros_like(s2)
        s2t2 = np.zeros_like(s2t2)

    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        pad = ((0, pad_h), (0, pad_w), (0, 0))
        s1, s2, aux = np.pad(s1, pad), np.pad(s2, pad), np.pad(aux, pad)
        s1t2, s2t2, auxt2 = np.pad(s1t2, pad), np.pad(s2t2, pad), np.pad(auxt2, pad)
        scl = np.pad(scl, pad, constant_values=4)
        sclt2 = np.pad(sclt2, pad, constant_values=4)
    H, W = s2.shape[:2]

    out_names = list(net.output_names)
    shapes = {"seg_head": (H, W, C), "carbon_head": (H, W, 2),
              "eco_head": (H, W, 2), "change_head": (H, W, 1),
              "risk_head": (H, W, 1), "quality_head": (H, W, 1),
              "boundary_head": (H, W, 1)}
    acc = {n: np.zeros(shapes[n], np.float32) for n in out_names if n in shapes}
    weight = np.zeros((H, W), np.float32)
    kernel = _hann_kernel(ps)
    mc = max(0, int(cfg.mc_samples))
    mc_sq = {n: np.zeros(shapes[n], np.float32)
             for n in acc} if mc > 1 else None

    variants = [(lambda x: x, lambda p: p),
                (lambda x: x[:, :, ::-1], lambda p: p[:, :, ::-1]),
                (lambda x: x[:, ::-1], lambda p: p[:, ::-1]),
                (lambda x: np.rot90(x, 1, (1, 2)),
                 lambda p: np.rot90(p, -1, (1, 2)))] if tta else \
               [(lambda x: x, lambda p: p)]

    coords = [(i, j) for i in _tile_starts(H, ps, overlap)
              for j in _tile_starts(W, ps, overlap)]
    print(f"Inference: {h}x{w} -> padded {H}x{W} | tiles={len(coords)} | "
          f"batch={batch_size} | TTA={tta} | MC={mc} | sar_only={sar_only} "
          f"| dt={dt_val}y")

    n_passes = 0
    for fwd, inv in variants:
        for k in range(0, len(coords), batch_size):
            chunk = coords[k:k + batch_size]

            def stack(arr):
                return np.stack([arr[i:i + ps, j:j + ps] for i, j in chunk])

            batch = {"s1_input": stack(s1), "s2_input": stack(s2),
                     "scl_input": stack(scl), "aux_input": stack(aux),
                     "s1_t2_input": stack(s1t2), "s2_t2_input": stack(s2t2),
                     "scl_t2_input": stack(sclt2), "aux_t2_input": stack(auxt2),
                     "dt_input": np.full((len(chunk), 1), dt_val, np.float32)}
            fb = {k2: fwd(v) for k2, v in batch.items()}
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
            for i, j in chunk:          # FIX F9: weight added PER PASS
                weight[i:i + ps, j:j + ps] += kernel
            n_passes += 1

    weight = np.maximum(weight[..., None], 1e-6)
    out = {}
    for name in acc:
        out[name] = acc[name] / weight
    res = {}
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
    if "change_head" in out:
        res["change_prob"] = out["change_head"][..., 0].astype(np.float32)[:h, :w]
        res["change_alert"] = (res["change_prob"] > cfg.change_threshold) \
            .astype(np.uint8)
    if "risk_head" in out:
        res["risk_index"] = (out["risk_head"][..., 0] * 100.0) \
            .astype(np.float32)[:h, :w]
    if "quality_head" in out:
        res["quality_prob"] = out["quality_head"][..., 0] \
            .astype(np.float32)[:h, :w]
    if mc_sq is not None and "seg_head" in out:
        var = np.maximum(mc_sq["seg_head"] / weight -
                         out["seg_head"] ** 2, 0.0)
        res["seg_epistemic_var"] = var.mean(-1).astype(np.float32)[:h, :w]
    return res


# ==============================================================================
# 19. EXPORT: COG GeoTIFFs + STAC item + GeoJSON alerts + MRV summary
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
    """Polygonize change alerts -> GeoJSON with area + confidence (MRV alerts)."""
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
                                     "confidence": cfg.change_threshold},
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
        "collection": "mangrove-ecomonitor-v5",
        "properties": {
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processing:software": f"mangrove-ecomonitor v{VERSION}",
            "mangrove:classes": CLASS_NAMES,
            "mangrove:active_heads": list(cfg.active_heads),
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
    """VM0033-aligned reporting primitives: class areas (ha) and carbon stock
    with 95% CI in tCO2e (sum of means, root-sum-of-squares of stds)."""
    pa_ha = _pixel_area_ha(reference_tif, cfg)
    rows = [("metric", "value", "unit")]
    if "classes" in products:
        for i, n in enumerate(CLASS_NAMES):
            area = float((products["classes"] == i).sum()) * pa_ha
            rows.append((f"area_class_{i}_{n.replace(' ', '_').replace('/', '_')}",
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
                     f"{float(products['change_alert'].sum()) * pa_ha:.2f}", "ha"))
    path = os.path.join(out_dir, "mrv_summary.csv")
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  saved {path}")
    return rows


def export_all_products(net, scene_npz_path, ref_tif_path, out_dir, cfg=CFG,
                        dt_years=None, sar_only=False):
    os.makedirs(out_dir, exist_ok=True)
    data = load_npz(scene_npz_path, cfg)
    s1, s2, scl, aux = data[0], data[1], data[2], data[3]
    has_t2 = not np.array_equal(data[4], data[0])
    t2 = {"s1": data[4], "s2": data[5], "scl": data[6], "aux": data[7]} \
        if has_t2 else None
    if dt_years is None and t2 is not None:
        dt_years = float(abs(data[14] - data[13])) or 1.0

    products = predict_scene(net, s1, s2, scl, aux, t2, cfg,
                             sar_only=sar_only, dt_years=dt_years)

    meta = []
    plan = [
        ("classes.tif", "classes", "uint8", 255, "Land-cover class (GMW-aligned)"),
        ("seg_probs.tif", "seg_probs", "float32", None, "Per-class probability"),
        ("carbon_mean.tif", "carbon_mean", "float32", -999, "AGB t/ha"),
        ("carbon_std.tif", "carbon_std", "float32", -999, "AGB uncertainty t/ha"),
        ("change_prob.tif", "change_prob", "float32", -999, "Change probability"),
        ("change_alert.tif", "change_alert", "uint8", 255, "Change alert mask"),
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

    export_change_geojson(products, ref_tif_path,
                          os.path.join(out_dir, "change_alerts.geojson"),
                          cfg, period=f"dt={dt_years}y")
    write_stac_item(out_dir, meta, ref_tif_path,
                    os.path.splitext(os.path.basename(scene_npz_path))[0], cfg)
    mrv_summary(products, ref_tif_path, out_dir, cfg)
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_npz_path, "reference": ref_tif_path,
                   "version": VERSION, "active_heads": list(cfg.active_heads),
                   "sar_only": sar_only, "dt_years": dt_years,
                   "seg_temperature": cfg.seg_temperature}, f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products


# ==============================================================================
# 20. SMOKE TEST
# ==============================================================================
def _smoke_generator(cfg, n=24):
    ps, A = cfg.patch_size, cfg.aux_channels
    rng = np.random.default_rng(42)
    for _ in range(n):
        s1 = np.concatenate([rng.uniform(-30, 0, (ps, ps, 2)),
                             rng.uniform(0, 3, (ps, ps, 2))], -1) \
            .astype(np.float32)
        s2 = rng.uniform(0, 0.5, (ps, ps, 10)).astype(np.float32)
        scl = rng.choice([4, 5, 6, 8, 9], (ps, ps, 1)).astype(np.float32)
        aux = rng.uniform(-2, 2, (ps, ps, A)).astype(np.float32)
        s1t2 = np.clip(s1 + rng.normal(0, 0.5, s1.shape), -35, 5) \
            .astype(np.float32)
        s2t2 = np.clip(s2 + rng.normal(0, 0.02, s2.shape), 0, 1) \
            .astype(np.float32)
        label = rng.choice([0, 1, 2, 3, 4, 255], (ps, ps)).astype(np.int32)
        carbon = rng.uniform(0, 250, (ps, ps)).astype(np.float32)
        eco = np.stack([rng.uniform(-5, 45, (ps, ps)),
                        rng.uniform(0, 1, (ps, ps))], -1).astype(np.float32)
        change = rng.choice([0, 0, 0, 1, 255], (ps, ps)).astype(np.int32)
        risk = rng.uniform(0, 1, (ps, ps)).astype(np.float32)
        yield (s1, s2, scl, aux, s1t2, s2t2, scl.copy(), aux.copy(),
               label, carbon, eco, change, risk,
               np.float32(2023.5), np.float32(2024.5))


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
    n_params = net.count_params()
    print(f"  model params: {n_params / 1e6:.1f}M | outputs: {net.output_names}")
    trainer.fit(ds_tr, validation_data=ds_va, epochs=2)

    model_path = os.path.join(WORK_DIR, "smoke_v5.keras")
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
                             batch_size=2, tta=True, dt_years=1.0)
    products_sar = predict_scene(m2, s1, s2, aux=aux, t2=t2, cfg=cfg,
                                 batch_size=2, tta=False, sar_only=True)
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
                                     "infer", "export"], nargs="?",
                    default="smoke")
    ap.add_argument("scene", nargs="?", help="scene NPZ (infer/export)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (infer/export)")
    ap.add_argument("out", nargs="?", help="output dir (infer/export)")
    ap.add_argument("--config", help="path to config_v5.json")
    ap.add_argument("--sar-only", action="store_true",
                    help="optical-free inference (cloudy scenes)")
    ap.add_argument("--dt", type=float, default=None,
                    help="time gap in years for the change decoder")
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
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras"))
        files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.npz")))
        ds = make_dataset(lambda: patch_generator(files, cfg, False,
                                                  cfg.max_val_patches),
                          cfg, False, cfg.batch_per_replica * N_REPLICAS)
        if "seg" in cfg.active_heads:
            evaluate_per_class(net, ds, cfg)
    elif args.mode in ("infer", "export"):
        if not (args.scene and args.ref and args.out):
            ap.error(f"{args.mode} requires <scene.npz> <ref.tif> <out_dir>")
        net = tf.keras.models.load_model(
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras"))
        export_all_products(net, args.scene, args.ref, args.out, cfg,
                            dt_years=args.dt, sar_only=args.sar_only)


if __name__ == "__main__":
    main()if self._step % self.update_every:
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
# 15. LR SCHEDULE / PLOTS / EVAL
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
    C = cfg.num_classes
    cm = np.zeros((C, C), np.int64)
    for inputs, targets in val_ds:
        out = net(inputs, training=False)
        p = _preds_to_dict(net, out)["seg_head"].numpy().argmax(-1)
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
    metrics = {n: {"iou": float(iou[i]), "pixels": int(cm[i].sum())}
               for i, n in enumerate(CLASS_NAMES)}
    metrics["mIoU"] = float(np.nanmean(iou))
    with open(os.path.join(WORK_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return cm, iou


def calibrate_temperature(net, val_ds, cfg, max_batches=10):
    """Post-hoc temperature scaling on validation seg probs (grid search)."""
    if "seg" not in cfg.active_heads:
        return 1.0
    probs, labels = [], []
    for i, (inputs, targets) in enumerate(val_ds):
        if i >= max_batches:
            break
        out = _preds_to_dict(net, net(inputs, training=False))
        p = out["seg_head"].numpy().reshape(-1, cfg.num_classes)
        y = targets["seg_head"].numpy().reshape(-1, cfg.num_classes)
        v = y.sum(-1) > 0
        probs.append(p[v]); labels.append(y[v].argmax(-1))
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
    print(f"  [calibration] seg temperature = {best_t:.2f} "
          f"(NLL {best_nll:.4f})")
    cfg.seg_temperature = float(best_t)
    # persist alongside norm stats
    if os.path.exists(STATS_PATH):
        z = dict(np.load(STATS_PATH))
        z["seg_temperature"] = np.float32(best_t)
        np.savez(STATS_PATH, **z)
    return best_t


def plot_training_history(history):
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
    heads = ["seg_head", "change_head", "quality_head", "boundary_head",
             "carbon_head", "risk_head"]
    have = [h for h in heads if h in [o for o in net.output_names]]
    ncols = 3 + len([h for h in have if h != "seg_head"])
    fig, axes = plt.subplots(n_samples, ncols, figsize=(4 * ncols,
                                                        4 * n_samples))
    axes = np.atleast_2d(axes)
    titles = ["S2 RGB", "GT Seg", "Pred Seg"] + \
             [h.replace("_head", "") for h in have if h != "seg_head"]
    for a, t in zip(axes[0], titles):
        a.set_title(t, fontsize=10)
    count = 0
    for inputs, targets in val_ds.take(2):
        preds = _preds_to_dict(net, net(inputs, training=False))
        preds = {k: v.numpy() for k, v in preds.items()}
        bs = len(targets["seg_head"])
        for i in range(min(bs, n_samples - count)):
            s2 = inputs["s2_input"][i].numpy()
            axes[count, 0].imshow(_stretch(s2[:, :, [2, 1, 0]]))
            y = targets["seg_head"][i].numpy().argmax(-1)
            axes[count, 1].imshow(y, cmap=cmap, vmin=0, vmax=4)
            if "seg_head" in preds:
                axes[count, 2].imshow(preds["seg_head"][i].argmax(-1),
                                      cmap=cmap, vmin=0, vmax=4)
            col = 3
            for h in have:
                if h == "seg_head":
                    continue
                arr = preds[h][i]
                if h == "carbon_head":
                    arr = arr[..., 0] * cfg.carbon_scale
                    axes[count, col].imshow(arr, cmap="YlGn")
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
# 16. SELF-SUPERVISED PRETRAINING (SimMIM-style masked reconstruction)
# ==============================================================================
def build_pretrain_model(cfg, s1_stats, s2_stats):
    """Masked-block reconstruction of S1 feats + S2 bands from the fused
    bottleneck. Encoder layer names match the main model -> by-name transfer."""
    ps, A = cfg.patch_size, cfg.aux_channels
    dims = list(cfg.embed_dims)
    s1_dims = [max(16, d // 2) for d in dims]
    depths = list(cfg.depths)

    s1_in = layers.Input((ps, ps, 4), name="s1_input")
    s2_in = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1), name="scl_input")
    aux_in = layers.Input((ps, ps, A), name="aux_input")

    fe = FeatureEngineer(cfg.invalid_scl, cfg.cloud_dilate, A,
                         name="feature_engineer")
    f1, f2 = fe([s1_in, s2_in, scl_in, aux_in])
    n1 = layers.Normalization(mean=list(s1_stats[0]),
                              variance=list(s1_stats[1]), name="s1_norm")
    n2 = layers.Normalization(mean=list(s2_stats[0]),
                              variance=list(s2_stats[1]), name="s2_norm")
    f1n, f2n = n1(f1), n2(f2)

    total_blocks = sum(depths) * 2
    dp_rates = np.linspace(0.0, cfg.drop_path, total_blocks).tolist()
    half = sum(depths)
    enc_s1 = make_swin_encoder("s1", s1_dims, depths, cfg, dp_rates[:half])
    enc_s2 = make_swin_encoder("s2", dims, depths, cfg, dp_rates[half:])
    b1, _ = enc_s1(f1n)
    b2, _ = enc_s2(f2n)

    if b1.shape[-1] != dims[-1]:
        b1 = layers.Conv2D(dims[-1], 1, name="sar_proj3")(b1)
    fuse = CrossAttentionFusion(dims[-1], cfg.fusion_heads,
                                name="cross_attn_fusion3")
    z = fuse([b2, b1])

    x = z
    for i, wd in enumerate([192, 96]):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"pt_up{i}")(x)
        x = conv_block(x, wd, use_se=False, name=f"pt_c{i}")
    rec = layers.Conv2D(C1 + 10, 1, padding="same", dtype="float32",
                        name="recon_head")(x)
    return models.Model([s1_in, s2_in, scl_in, aux_in], rec,
                        name="MangrovePretrainer_v5"), (f1, f2, f1n, f2n)


def _masked_block(x_raw, ratio, block):
    """Zero random square blocks of an image tensor. Returns (x_masked, keep)."""
    ps = tf.shape(x_raw)[1]
    g = tf.maximum(ps // block, 1)
    cells = tf.random.uniform((tf.shape(x_raw)[0], g, g, 1)) < ratio
    mask = tf.image.resize(tf.cast(cells, tf.float32), [ps, ps],
                           method="nearest")                    # 1 = masked
    keep = 1.0 - mask
    return x_raw * keep, keep


def pretrain(cfg=CFG):
    """SimMIM-style: reconstruct normalized S1 feats + raw S2 bands at 1/4 res
    under heavy block masking. No labels needed."""
    print("\n=== v5 SELF-SUPERVISED PRETRAINING (SimMIM-style) ===")
    files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.npz")))
    assert files, "pretrain: set cfg.train_dir to a folder of NPZ scenes"
    s1m, s1v, s2m, s2v, _ = load_or_compute_stats(files, cfg)

    def gen():
        yield from patch_generator(files, cfg, False, cfg.max_train_patches)

    ps = cfg.patch_size
    sig = output_signature(cfg)

    def to_inputs(*vals):
        s1, s2, scl, aux = vals[0], vals[1], vals[2], vals[3]
        return (s1, s2, scl, aux)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig) \
        .map(lambda *v: to_inputs(*v), num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(cfg.shuffle_buffer) \
        .batch(cfg.batch_per_replica * N_REPLICAS, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    s1_mean = tf.constant(s1m, tf.float32)
    s1_std = tf.sqrt(tf.constant(s1v, tf.float32))

    with STRATEGY.scope():
        pt, refs = build_pretrain_model(cfg, (s1m, s1v), (s2m, s2v))
        f1_raw, f2_raw, f1n, f2n = refs
        opt = optimizers.AdamW(learning_rate=cfg.pretrain_lr,
                               weight_decay=cfg.weight_decay)

        @tf.function
        def step(s1, s2, scl, aux):
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block(s1, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                s2_m, keep2 = _masked_block(s2, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                rec = pt({"s1_input": s1_m, "s2_input": s2_m,
                          "scl_input": scl, "aux_input": aux}, training=True)
                # targets at 1/4 res: normalized s1 feats + raw s2 bands
                t_s1 = (feats_from(s1, scl) - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1, 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2, 4, 4, "VALID")
                m = tf.concat([tf.repeat(m1, C1, -1), tf.repeat(m2, 10, -1)], -1)
                m = tf.cast(m > 0.5, tf.float32)               # 1 = masked cell
                loss = tf.reduce_sum(tf.abs(rec - tgt_q) * m) / \
                    (tf.reduce_sum(m) + 1e-6)
            grads = tape.gradient(loss, pt.trainable_variables)
            opt.apply_gradients([(g, v) for g, v in
                                 zip(grads, pt.trainable_variables)
                                 if g is not None])
            return loss

        # small helper: recompute S1 features OUTSIDE the model for targets
        def feats_from(s1, scl):
            vv, vh, vv_std, vh_std = tf.split(s1, 4, axis=-1)
            lin_vv = tf.pow(10.0, vv / 10.0)
            lin_vh = tf.pow(10.0, vh / 10.0)
            rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
            return tf.concat([vv, vh, vh - vv, rvi, vv_std, vh_std], -1)

        # make feats_from visible to the tf.function closure
        step.__globals__["feats_from"] = feats_from if hasattr(step, "__globals__") else None

        step_fn = step
        # NOTE: nested def ordering - rebuild step with helper bound
        @tf.function
        def train_one(s1, s2, scl, aux):
            with tf.GradientTape() as tape:
                s1_m, keep1 = _masked_block(s1, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                s2_m, keep2 = _masked_block(s2, cfg.pretrain_mask_ratio,
                                            cfg.pretrain_block)
                rec = pt({"s1_input": s1_m, "s2_input": s2_m,
                          "scl_input": scl, "aux_input": aux}, training=True)
                vv, vh, vvs, vhs = tf.split(s1, 4, axis=-1)
                lin_vv = tf.pow(10.0, vv / 10.0)
                lin_vh = tf.pow(10.0, vh / 10.0)
                rvi = 4.0 * lin_vh / (lin_vv + lin_vh + 1e-5)
                t_s1 = tf.concat([vv, vh, vh - vv, rvi, vvs, vhs], -1)
                t_s1 = (t_s1 - s1_mean) / (s1_std + 1e-6)
                tgt = tf.concat([t_s1, s2], -1)
                tgt_q = tf.nn.avg_pool2d(tgt, 4, 4, "VALID")
                m1 = tf.nn.avg_pool2d(1.0 - keep1, 4, 4, "VALID")
                m2 = tf.nn.avg_pool2d(1.0 - keep2, 4, 4, "VALID")
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
            for s1, s2, scl, aux in ds:
                losses.append(float(train_one(s1, s2, scl, aux)))
            print(f"  pretrain {ep + 1}/{cfg.pretrain_epochs} "
                  f"loss={np.mean(losses):.4f} ({time.time() - t0:.0f}s)")

        out = os.path.join(WORK_DIR, "pretrain_encoder_v5.weights.h5")
        pt.save_weights(out)
        print(f"Pretrained encoder weights saved: {out}")
        print("Fine-tune with: cfg.pretrain_weights = "
              f"'{out}' before train()")
        return out


# ==============================================================================
# 17. BATCH PREFLIGHT + OOM-RESILIENT TRAINING
# ==============================================================================
def preflight_batch(trainer, ds):
    """One train_on_batch to verify memory; returns True/False."""
    try:
        x, y = next(iter(ds))
        trainer.train_on_batch(x, y)
        return True
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        print(f"  [preflight] OOM at current batch: {str(e)[:160]}")
        return False


def train(cfg=CFG):
    print(f"\n=== MANGROVE ECO-MONITOR v{VERSION} - TRAINING ===\n")
    train_files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.npz")))
    val_files = sorted(glob.glob(os.path.join(cfg.val_dir, "*.npz")))
    assert train_files and val_files, "Set cfg.train_dir / cfg.val_dir"
    validate_heads_vs_data(train_files, cfg)
    print(f"train scenes: {len(train_files)} | val scenes: {len(val_files)}")

    s1m, s1v, s2m, s2v, alpha = load_or_compute_stats(train_files, cfg)

    if cfg.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    attempt, oom_left = 0, cfg.oom_retries
    while True:
        batch = cfg.batch_per_replica * N_REPLICAS
        ds_train = make_dataset(
            lambda: patch_generator(train_files, cfg, True,
                                    cfg.max_train_patches), cfg, True, batch)
        ds_val = make_dataset(
            lambda: patch_generator(val_files, cfg, False,
                                    cfg.max_val_patches), cfg, False, batch)

        with STRATEGY.scope():
            net = build_model(cfg, (s1m, s1v), (s2m, s2v))
            if cfg.pretrain_weights and os.path.exists(cfg.pretrain_weights):
                try:
                    net.load_weights(cfg.pretrain_weights, by_name=True,
                                     skip_mismatch=True)
                    print(f"  [pretrain] loaded encoder weights "
                          f"(by_name, skip_mismatch) from {cfg.pretrain_weights}")
                except Exception as e:
                    print(f"  [pretrain] weight load failed ({e}); training "
                          f"from scratch.")
            trainer = EcoMonitorTrainer(cfg, net, alpha)
            est_steps = max(1, cfg.max_train_patches // batch)
            lr_sched = WarmUpCosineDecay(
                cfg.lr, cfg.epochs * est_steps,
                cfg.lr_warmup_epochs * est_steps)
            trainer.compile(optimizer=make_optimizer(cfg, lr_sched))

        if cfg.auto_batch and attempt == 0:
            ok = preflight_batch(trainer, ds_train)
            while not ok and cfg.batch_per_replica > 1:
                cfg.batch_per_replica //= 2
                tf.keras.backend.clear_session()
                gc.collect()
                batch = cfg.batch_per_replica * N_REPLICAS
                ds_train = make_dataset(
                    lambda: patch_generator(train_files, cfg, True,
                                            cfg.max_train_patches),
                    cfg, True, batch)
                with STRATEGY.scope():
                    net = build_model(cfg, (s1m, s1v), (s2m, s2v))
                    trainer = EcoMonitorTrainer(cfg, net, alpha)
                    trainer.compile(optimizer=make_optimizer(cfg, lr_sched))
                ok = preflight_batch(trainer, ds_train)
            print(f"  [auto_batch] settled on batch/replica="
                  f"{cfg.batch_per_replica}")

        monitor = "val_seg_mean_iou" if "seg" in cfg.active_heads else "val_loss"
        mode = "max" if "iou" in monitor else "min"
        ema = EMACallback(cfg.ema_decay)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=15,
                                             restore_best_weights=True,
                                             verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(WORK_DIR, "best_v5.weights.h5"), monitor,
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
        except (tf.errors.ResourceExhaustedError,) as e:
            if oom_left <= 0 or cfg.batch_per_replica <= 1:
                raise
            oom_left -= 1
            attempt += 1
            cfg.batch_per_replica = max(1, cfg.batch_per_replica // 2)
            print(f"  [OOM-RETRY] halving batch to {cfg.batch_per_replica}/replica "
                  f"and restarting from backup ({oom_left} retries left). "
                  f"Detail: {str(e)[:120]}")
            tf.keras.backend.clear_session()
            gc.collect()

    # ---- finalize on EMA weights ----
    print("\nSwapping in EMA weights for final eval + export...")
    ema.swap_in()
    calibrate_temperature(net, ds_val, cfg)
    model_path = os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras")
    net.save(model_path)
    cfg.save(os.path.join(WORK_DIR, "config_v5.json"))
    plot_training_history(history)
    if "seg" in cfg.active_heads:
        evaluate_per_class(net, ds_val, cfg)
    visualize_predictions(net, ds_val, cfg)
    print(f"\nModel saved: {model_path}")
    print(f"Config saved: {os.path.join(WORK_DIR, 'config_v5.json')}")
    return net


# ==============================================================================
# 18. TILED INFERENCE (Hann blending fixed, TTA fixed, MC-dropout, SAR-only)
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
    """FIX F10: proper centre-peaked window (v4's was inverted)."""
    h = np.hanning(ps).astype(np.float32)
    h = np.maximum(h, floor)
    return np.outer(h, h)


def predict_scene(net, s1, s2, scl=None, aux=None, t2=None, cfg=CFG,
                  batch_size=None, tta=None, overlap=None, sar_only=False,
                  dt_years=None):
    """Production inference: Hann-blended tiling, correct TTA averaging
    (FIX F9), optional MC-dropout epistemic variance, SAR-only fallback."""
    batch_size = batch_size or cfg.infer_batch
    tta = cfg.infer_tta if tta is None else tta
    overlap = cfg.infer_overlap if overlap is None else overlap
    ps, C, A = cfg.patch_size, cfg.num_classes, cfg.aux_channels
    h, w = s2.shape[:2]
    temp = float(getattr(cfg, "seg_temperature", 1.0))

    if s1.shape[-1] == 2:
        s1 = np.concatenate([s1, np.zeros_like(s1)], -1)
    scl = np.full((h, w, 1), 4.0, np.float32) if scl is None \
        else scl[..., None].astype(np.float32)
    aux = np.zeros((h, w, A), np.float32) if aux is None \
        else aux.astype(np.float32)
    if t2 is None:
        s1t2, s2t2, sclt2, auxt2 = s1, s2, scl, aux
        dt_val = 0.0
    else:
        s1t2 = t2["s1"]
        if s1t2.shape[-1] == 2:
            s1t2 = np.concatenate([s1t2, np.zeros_like(s1t2)], -1)
        s2t2 = t2["s2"]
        sclt2 = t2["scl"][..., None].astype(np.float32) \
            if t2.get("scl") is not None else scl
        auxt2 = t2["aux"].astype(np.float32) if t2.get("aux") is not None else aux
        dt_val = float(dt_years if dt_years is not None else 1.0)

    if sar_only:   # optical branch zeroed: model was trained for this (U8)
        s2 = np.zeros_like(s2)
        s2t2 = np.zeros_like(s2t2)

    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        pad = ((0, pad_h), (0, pad_w), (0, 0))
        s1, s2, aux = np.pad(s1, pad), np.pad(s2, pad), np.pad(aux, pad)
        s1t2, s2t2, auxt2 = np.pad(s1t2, pad), np.pad(s2t2, pad), np.pad(auxt2, pad)
        scl = np.pad(scl, pad, constant_values=4)
        sclt2 = np.pad(sclt2, pad, constant_values=4)
    H, W = s2.shape[:2]

    out_names = list(net.output_names)
    shapes = {"seg_head": (H, W, C), "carbon_head": (H, W, 2),
              "eco_head": (H, W, 2), "change_head": (H, W, 1),
              "risk_head": (H, W, 1), "quality_head": (H, W, 1),
              "boundary_head": (H, W, 1)}
    acc = {n: np.zeros(shapes[n], np.float32) for n in out_names if n in shapes}
    weight = np.zeros((H, W), np.float32)
    kernel = _hann_kernel(ps)
    mc = max(0, int(cfg.mc_samples))
    mc_sq = {n: np.zeros(shapes[n], np.float32)
             for n in acc} if mc > 1 else None

    variants = [(lambda x: x, lambda p: p),
                (lambda x: x[:, :, ::-1], lambda p: p[:, :, ::-1]),
                (lambda x: x[:, ::-1], lambda p: p[:, ::-1]),
                (lambda x: np.rot90(x, 1, (1, 2)),
                 lambda p: np.rot90(p, -1, (1, 2)))] if tta else \
               [(lambda x: x, lambda p: p)]

    coords = [(i, j) for i in _tile_starts(H, ps, overlap)
              for j in _tile_starts(W, ps, overlap)]
    print(f"Inference: {h}x{w} -> padded {H}x{W} | tiles={len(coords)} | "
          f"batch={batch_size} | TTA={tta} | MC={mc} | sar_only={sar_only} "
          f"| dt={dt_val}y")

    n_passes = 0
    for fwd, inv in variants:
        for k in range(0, len(coords), batch_size):
            chunk = coords[k:k + batch_size]

            def stack(arr):
                return np.stack([arr[i:i + ps, j:j + ps] for i, j in chunk])

            batch = {"s1_input": stack(s1), "s2_input": stack(s2),
                     "scl_input": stack(scl), "aux_input": stack(aux),
                     "s1_t2_input": stack(s1t2), "s2_t2_input": stack(s2t2),
                     "scl_t2_input": stack(sclt2), "aux_t2_input": stack(auxt2),
                     "dt_input": np.full((len(chunk), 1), dt_val, np.float32)}
            fb = {k2: fwd(v) for k2, v in batch.items()}
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
            for i, j in chunk:          # FIX F9: weight added PER PASS
                weight[i:i + ps, j:j + ps] += kernel
            n_passes += 1

    weight = np.maximum(weight[..., None], 1e-6)
    out = {}
    for name in acc:
        out[name] = acc[name] / weight
    res = {}
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
    if "change_head" in out:
        res["change_prob"] = out["change_head"][..., 0].astype(np.float32)[:h, :w]
        res["change_alert"] = (res["change_prob"] > cfg.change_threshold) \
            .astype(np.uint8)
    if "risk_head" in out:
        res["risk_index"] = (out["risk_head"][..., 0] * 100.0) \
            .astype(np.float32)[:h, :w]
    if "quality_head" in out:
        res["quality_prob"] = out["quality_head"][..., 0] \
            .astype(np.float32)[:h, :w]
    if mc_sq is not None and "seg_head" in out:
        var = np.maximum(mc_sq["seg_head"] / weight -
                         out["seg_head"] ** 2, 0.0)
        res["seg_epistemic_var"] = var.mean(-1).astype(np.float32)[:h, :w]
    return res


# ==============================================================================
# 19. EXPORT: COG GeoTIFFs + STAC item + GeoJSON alerts + MRV summary
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
    """Polygonize change alerts -> GeoJSON with area + confidence (MRV alerts)."""
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
                                     "confidence": cfg.change_threshold},
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
        "collection": "mangrove-ecomonitor-v5",
        "properties": {
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processing:software": f"mangrove-ecomonitor v{VERSION}",
            "mangrove:classes": CLASS_NAMES,
            "mangrove:active_heads": list(cfg.active_heads),
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
    """VM0033-aligned reporting primitives: class areas (ha) and carbon stock
    with 95% CI in tCO2e (sum of means, root-sum-of-squares of stds)."""
    pa_ha = _pixel_area_ha(reference_tif, cfg)
    rows = [("metric", "value", "unit")]
    if "classes" in products:
        for i, n in enumerate(CLASS_NAMES):
            area = float((products["classes"] == i).sum()) * pa_ha
            rows.append((f"area_class_{i}_{n.replace(' ', '_').replace('/', '_')}",
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
                     f"{float(products['change_alert'].sum()) * pa_ha:.2f}", "ha"))
    path = os.path.join(out_dir, "mrv_summary.csv")
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  saved {path}")
    return rows


def export_all_products(net, scene_npz_path, ref_tif_path, out_dir, cfg=CFG,
                        dt_years=None, sar_only=False):
    os.makedirs(out_dir, exist_ok=True)
    data = load_npz(scene_npz_path, cfg)
    s1, s2, scl, aux = data[0], data[1], data[2], data[3]
    has_t2 = not np.array_equal(data[4], data[0])
    t2 = {"s1": data[4], "s2": data[5], "scl": data[6], "aux": data[7]} \
        if has_t2 else None
    if dt_years is None and t2 is not None:
        dt_years = float(abs(data[14] - data[13])) or 1.0

    products = predict_scene(net, s1, s2, scl, aux, t2, cfg,
                             sar_only=sar_only, dt_years=dt_years)

    meta = []
    plan = [
        ("classes.tif", "classes", "uint8", 255, "Land-cover class (GMW-aligned)"),
        ("seg_probs.tif", "seg_probs", "float32", None, "Per-class probability"),
        ("carbon_mean.tif", "carbon_mean", "float32", -999, "AGB t/ha"),
        ("carbon_std.tif", "carbon_std", "float32", -999, "AGB uncertainty t/ha"),
        ("change_prob.tif", "change_prob", "float32", -999, "Change probability"),
        ("change_alert.tif", "change_alert", "uint8", 255, "Change alert mask"),
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

    export_change_geojson(products, ref_tif_path,
                          os.path.join(out_dir, "change_alerts.geojson"),
                          cfg, period=f"dt={dt_years}y")
    write_stac_item(out_dir, meta, ref_tif_path,
                    os.path.splitext(os.path.basename(scene_npz_path))[0], cfg)
    mrv_summary(products, ref_tif_path, out_dir, cfg)
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_npz_path, "reference": ref_tif_path,
                   "version": VERSION, "active_heads": list(cfg.active_heads),
                   "sar_only": sar_only, "dt_years": dt_years,
                   "seg_temperature": cfg.seg_temperature}, f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products


# ==============================================================================
# 20. SMOKE TEST
# ==============================================================================
def _smoke_generator(cfg, n=24):
    ps, A = cfg.patch_size, cfg.aux_channels
    rng = np.random.default_rng(42)
    for _ in range(n):
        s1 = np.concatenate([rng.uniform(-30, 0, (ps, ps, 2)),
                             rng.uniform(0, 3, (ps, ps, 2))], -1) \
            .astype(np.float32)
        s2 = rng.uniform(0, 0.5, (ps, ps, 10)).astype(np.float32)
        scl = rng.choice([4, 5, 6, 8, 9], (ps, ps, 1)).astype(np.float32)
        aux = rng.uniform(-2, 2, (ps, ps, A)).astype(np.float32)
        s1t2 = np.clip(s1 + rng.normal(0, 0.5, s1.shape), -35, 5) \
            .astype(np.float32)
        s2t2 = np.clip(s2 + rng.normal(0, 0.02, s2.shape), 0, 1) \
            .astype(np.float32)
        label = rng.choice([0, 1, 2, 3, 4, 255], (ps, ps)).astype(np.int32)
        carbon = rng.uniform(0, 250, (ps, ps)).astype(np.float32)
        eco = np.stack([rng.uniform(-5, 45, (ps, ps)),
                        rng.uniform(0, 1, (ps, ps))], -1).astype(np.float32)
        change = rng.choice([0, 0, 0, 1, 255], (ps, ps)).astype(np.int32)
        risk = rng.uniform(0, 1, (ps, ps)).astype(np.float32)
        yield (s1, s2, scl, aux, s1t2, s2t2, scl.copy(), aux.copy(),
               label, carbon, eco, change, risk,
               np.float32(2023.5), np.float32(2024.5))


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
    n_params = net.count_params()
    print(f"  model params: {n_params / 1e6:.1f}M | outputs: {net.output_names}")
    trainer.fit(ds_tr, validation_data=ds_va, epochs=2)

    model_path = os.path.join(WORK_DIR, "smoke_v5.keras")
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
                             batch_size=2, tta=True, dt_years=1.0)
    products_sar = predict_scene(m2, s1, s2, aux=aux, t2=t2, cfg=cfg,
                                 batch_size=2, tta=False, sar_only=True)
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
                                     "infer", "export"], nargs="?",
                    default="smoke")
    ap.add_argument("scene", nargs="?", help="scene NPZ (infer/export)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (infer/export)")
    ap.add_argument("out", nargs="?", help="output dir (infer/export)")
    ap.add_argument("--config", help="path to config_v5.json")
    ap.add_argument("--sar-only", action="store_true",
                    help="optical-free inference (cloudy scenes)")
    ap.add_argument("--dt", type=float, default=None,
                    help="time gap in years for the change decoder")
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
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras"))
        files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.npz")))
        ds = make_dataset(lambda: patch_generator(files, cfg, False,
                                                  cfg.max_val_patches),
                          cfg, False, cfg.batch_per_replica * N_REPLICAS)
        if "seg" in cfg.active_heads:
            evaluate_per_class(net, ds, cfg)
    elif args.mode in ("infer", "export"):
        if not (args.scene and args.ref and args.out):
            ap.error(f"{args.mode} requires <scene.npz> <ref.tif> <out_dir>")
        net = tf.keras.models.load_model(
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras"))
        export_all_products(net, args.scene, args.ref, args.out, cfg,
                            dt_years=args.dt, sar_only=args.sar_only)


if __name__ == "__main__":
    main()
            attempt += 1
            cfg.batch_per_replica = max(1, cfg.batch_per_replica // 2)
            print(f"  [OOM-RETRY] halving batch to {cfg.batch_per_replica}/replica "
                  f"and restarting from backup ({oom_left} retries left). "
                  f"Detail: {str(e)[:120]}")
            tf.keras.backend.clear_session()
            gc.collect()

    # ---- finalize on EMA weights ----
    print("\nSwapping in EMA weights for final eval + export...")
    ema.swap_in()
    calibrate_temperature(net, ds_val, cfg)
    model_path = os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras")
    net.save(model_path)
    cfg.save(os.path.join(WORK_DIR, "config_v5.json"))
    plot_training_history(history)
    if "seg" in cfg.active_heads:
        evaluate_per_class(net, ds_val, cfg)
    visualize_predictions(net, ds_val, cfg)
    print(f"\nModel saved: {model_path}")
    print(f"Config saved: {os.path.join(WORK_DIR, 'config_v5.json')}")
    return net


# ==============================================================================
# 18. TILED INFERENCE (Hann blending fixed, TTA fixed, MC-dropout, SAR-only)
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
    """FIX F10: proper centre-peaked window (v4's was inverted)."""
    h = np.hanning(ps).astype(np.float32)
    h = np.maximum(h, floor)
    return np.outer(h, h)


def predict_scene(net, s1, s2, scl=None, aux=None, t2=None, cfg=CFG,
                  batch_size=None, tta=None, overlap=None, sar_only=False,
                  dt_years=None):
    """Production inference: Hann-blended tiling, correct TTA averaging
    (FIX F9), optional MC-dropout epistemic variance, SAR-only fallback."""
    batch_size = batch_size or cfg.infer_batch
    tta = cfg.infer_tta if tta is None else tta
    overlap = cfg.infer_overlap if overlap is None else overlap
    ps, C, A = cfg.patch_size, cfg.num_classes, cfg.aux_channels
    h, w = s2.shape[:2]
    temp = float(getattr(cfg, "seg_temperature", 1.0))

    if s1.shape[-1] == 2:
        s1 = np.concatenate([s1, np.zeros_like(s1)], -1)
    scl = np.full((h, w, 1), 4.0, np.float32) if scl is None \
        else scl[..., None].astype(np.float32)
    aux = np.zeros((h, w, A), np.float32) if aux is None \
        else aux.astype(np.float32)
    if t2 is None:
        s1t2, s2t2, sclt2, auxt2 = s1, s2, scl, aux
        dt_val = 0.0
    else:
        s1t2 = t2["s1"]
        if s1t2.shape[-1] == 2:
            s1t2 = np.concatenate([s1t2, np.zeros_like(s1t2)], -1)
        s2t2 = t2["s2"]
        sclt2 = t2["scl"][..., None].astype(np.float32) \
            if t2.get("scl") is not None else scl
        auxt2 = t2["aux"].astype(np.float32) if t2.get("aux") is not None else aux
        dt_val = float(dt_years if dt_years is not None else 1.0)

    if sar_only:   # optical branch zeroed: model was trained for this (U8)
        s2 = np.zeros_like(s2)
        s2t2 = np.zeros_like(s2t2)

    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        pad = ((0, pad_h), (0, pad_w), (0, 0))
        s1, s2, aux = np.pad(s1, pad), np.pad(s2, pad), np.pad(aux, pad)
        s1t2, s2t2, auxt2 = np.pad(s1t2, pad), np.pad(s2t2, pad), np.pad(auxt2, pad)
        scl = np.pad(scl, pad, constant_values=4)
        sclt2 = np.pad(sclt2, pad, constant_values=4)
    H, W = s2.shape[:2]

    out_names = list(net.output_names)
    shapes = {"seg_head": (H, W, C), "carbon_head": (H, W, 2),
              "eco_head": (H, W, 2), "change_head": (H, W, 1),
              "risk_head": (H, W, 1), "quality_head": (H, W, 1),
              "boundary_head": (H, W, 1)}
    acc = {n: np.zeros(shapes[n], np.float32) for n in out_names if n in shapes}
    weight = np.zeros((H, W), np.float32)
    kernel = _hann_kernel(ps)
    mc = max(0, int(cfg.mc_samples))
    mc_sq = {n: np.zeros(shapes[n], np.float32)
             for n in acc} if mc > 1 else None

    variants = [(lambda x: x, lambda p: p),
                (lambda x: x[:, :, ::-1], lambda p: p[:, :, ::-1]),
                (lambda x: x[:, ::-1], lambda p: p[:, ::-1]),
                (lambda x: np.rot90(x, 1, (1, 2)),
                 lambda p: np.rot90(p, -1, (1, 2)))] if tta else \
               [(lambda x: x, lambda p: p)]

    coords = [(i, j) for i in _tile_starts(H, ps, overlap)
              for j in _tile_starts(W, ps, overlap)]
    print(f"Inference: {h}x{w} -> padded {H}x{W} | tiles={len(coords)} | "
          f"batch={batch_size} | TTA={tta} | MC={mc} | sar_only={sar_only} "
          f"| dt={dt_val}y")

    n_passes = 0
    for fwd, inv in variants:
        for k in range(0, len(coords), batch_size):
            chunk = coords[k:k + batch_size]

            def stack(arr):
                return np.stack([arr[i:i + ps, j:j + ps] for i, j in chunk])

            batch = {"s1_input": stack(s1), "s2_input": stack(s2),
                     "scl_input": stack(scl), "aux_input": stack(aux),
                     "s1_t2_input": stack(s1t2), "s2_t2_input": stack(s2t2),
                     "scl_t2_input": stack(sclt2), "aux_t2_input": stack(auxt2),
                     "dt_input": np.full((len(chunk), 1), dt_val, np.float32)}
            fb = {k2: fwd(v) for k2, v in batch.items()}
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
            for i, j in chunk:          # FIX F9: weight added PER PASS
                weight[i:i + ps, j:j + ps] += kernel
            n_passes += 1

    weight = np.maximum(weight[..., None], 1e-6)
    out = {}
    for name in acc:
        out[name] = acc[name] / weight
    res = {}
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
    if "change_head" in out:
        res["change_prob"] = out["change_head"][..., 0].astype(np.float32)[:h, :w]
        res["change_alert"] = (res["change_prob"] > cfg.change_threshold) \
            .astype(np.uint8)
    if "risk_head" in out:
        res["risk_index"] = (out["risk_head"][..., 0] * 100.0) \
            .astype(np.float32)[:h, :w]
    if "quality_head" in out:
        res["quality_prob"] = out["quality_head"][..., 0] \
            .astype(np.float32)[:h, :w]
    if mc_sq is not None and "seg_head" in out:
        var = np.maximum(mc_sq["seg_head"] / weight -
                         out["seg_head"] ** 2, 0.0)
        res["seg_epistemic_var"] = var.mean(-1).astype(np.float32)[:h, :w]
    return res


# ==============================================================================
# 19. EXPORT: COG GeoTIFFs + STAC item + GeoJSON alerts + MRV summary
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
    """Polygonize change alerts -> GeoJSON with area + confidence (MRV alerts)."""
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
                                     "confidence": cfg.change_threshold},
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
        "collection": "mangrove-ecomonitor-v5",
        "properties": {
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processing:software": f"mangrove-ecomonitor v{VERSION}",
            "mangrove:classes": CLASS_NAMES,
            "mangrove:active_heads": list(cfg.active_heads),
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
    """VM0033-aligned reporting primitives: class areas (ha) and carbon stock
    with 95% CI in tCO2e (sum of means, root-sum-of-squares of stds)."""
    pa_ha = _pixel_area_ha(reference_tif, cfg)
    rows = [("metric", "value", "unit")]
    if "classes" in products:
        for i, n in enumerate(CLASS_NAMES):
            area = float((products["classes"] == i).sum()) * pa_ha
            rows.append((f"area_class_{i}_{n.replace(' ', '_').replace('/', '_')}",
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
                     f"{float(products['change_alert'].sum()) * pa_ha:.2f}", "ha"))
    path = os.path.join(out_dir, "mrv_summary.csv")
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  saved {path}")
    return rows


def export_all_products(net, scene_npz_path, ref_tif_path, out_dir, cfg=CFG,
                        dt_years=None, sar_only=False):
    os.makedirs(out_dir, exist_ok=True)
    data = load_npz(scene_npz_path, cfg)
    s1, s2, scl, aux = data[0], data[1], data[2], data[3]
    has_t2 = not np.array_equal(data[4], data[0])
    t2 = {"s1": data[4], "s2": data[5], "scl": data[6], "aux": data[7]} \
        if has_t2 else None
    if dt_years is None and t2 is not None:
        dt_years = float(abs(data[14] - data[13])) or 1.0

    products = predict_scene(net, s1, s2, scl, aux, t2, cfg,
                             sar_only=sar_only, dt_years=dt_years)

    meta = []
    plan = [
        ("classes.tif", "classes", "uint8", 255, "Land-cover class (GMW-aligned)"),
        ("seg_probs.tif", "seg_probs", "float32", None, "Per-class probability"),
        ("carbon_mean.tif", "carbon_mean", "float32", -999, "AGB t/ha"),
        ("carbon_std.tif", "carbon_std", "float32", -999, "AGB uncertainty t/ha"),
        ("change_prob.tif", "change_prob", "float32", -999, "Change probability"),
        ("change_alert.tif", "change_alert", "uint8", 255, "Change alert mask"),
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

    export_change_geojson(products, ref_tif_path,
                          os.path.join(out_dir, "change_alerts.geojson"),
                          cfg, period=f"dt={dt_years}y")
    write_stac_item(out_dir, meta, ref_tif_path,
                    os.path.splitext(os.path.basename(scene_npz_path))[0], cfg)
    mrv_summary(products, ref_tif_path, out_dir, cfg)
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump({"scene": scene_npz_path, "reference": ref_tif_path,
                   "version": VERSION, "active_heads": list(cfg.active_heads),
                   "sar_only": sar_only, "dt_years": dt_years,
                   "seg_temperature": cfg.seg_temperature}, f, indent=2)
    print(f"\nAll products exported to: {out_dir}")
    return products


# ==============================================================================
# 20. SMOKE TEST
# ==============================================================================
def _smoke_generator(cfg, n=24):
    ps, A = cfg.patch_size, cfg.aux_channels
    rng = np.random.default_rng(42)
    for _ in range(n):
        s1 = np.concatenate([rng.uniform(-30, 0, (ps, ps, 2)),
                             rng.uniform(0, 3, (ps, ps, 2))], -1) \
            .astype(np.float32)
        s2 = rng.uniform(0, 0.5, (ps, ps, 10)).astype(np.float32)
        scl = rng.choice([4, 5, 6, 8, 9], (ps, ps, 1)).astype(np.float32)
        aux = rng.uniform(-2, 2, (ps, ps, A)).astype(np.float32)
        s1t2 = np.clip(s1 + rng.normal(0, 0.5, s1.shape), -35, 5) \
            .astype(np.float32)
        s2t2 = np.clip(s2 + rng.normal(0, 0.02, s2.shape), 0, 1) \
            .astype(np.float32)
        label = rng.choice([0, 1, 2, 3, 4, 255], (ps, ps)).astype(np.int32)
        carbon = rng.uniform(0, 250, (ps, ps)).astype(np.float32)
        eco = np.stack([rng.uniform(-5, 45, (ps, ps)),
                        rng.uniform(0, 1, (ps, ps))], -1).astype(np.float32)
        change = rng.choice([0, 0, 0, 1, 255], (ps, ps)).astype(np.int32)
        risk = rng.uniform(0, 1, (ps, ps)).astype(np.float32)
        yield (s1, s2, scl, aux, s1t2, s2t2, scl.copy(), aux.copy(),
               label, carbon, eco, change, risk,
               np.float32(2023.5), np.float32(2024.5))


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
    n_params = net.count_params()
    print(f"  model params: {n_params / 1e6:.1f}M | outputs: {net.output_names}")
    trainer.fit(ds_tr, validation_data=ds_va, epochs=2)

    model_path = os.path.join(WORK_DIR, "smoke_v5.keras")
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
                             batch_size=2, tta=True, dt_years=1.0)
    products_sar = predict_scene(m2, s1, s2, aux=aux, t2=t2, cfg=cfg,
                                 batch_size=2, tta=False, sar_only=True)
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
                                     "infer", "export"], nargs="?",
                    default="smoke")
    ap.add_argument("scene", nargs="?", help="scene NPZ (infer/export)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (infer/export)")
    ap.add_argument("out", nargs="?", help="output dir (infer/export)")
    ap.add_argument("--config", help="path to config_v5.json")
    ap.add_argument("--sar-only", action="store_true",
                    help="optical-free inference (cloudy scenes)")
    ap.add_argument("--dt", type=float, default=None,
                    help="time gap in years for the change decoder")
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
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras"))
        files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.npz")))
        ds = make_dataset(lambda: patch_generator(files, cfg, False,
                                                  cfg.max_val_patches),
                          cfg, False, cfg.batch_per_replica * N_REPLICAS)
        if "seg" in cfg.active_heads:
            evaluate_per_class(net, ds, cfg)
    elif args.mode in ("infer", "export"):
        if not (args.scene and args.ref and args.out):
            ap.error(f"{args.mode} requires <scene.npz> <ref.tif> <out_dir>")
        net = tf.keras.models.load_model(
            os.path.join(WORK_DIR, "mangrove_ecomonitor_v5.keras"))
        export_all_products(net, args.scene, args.ref, args.out, cfg,
                            dt_years=args.dt, sar_only=args.sar_only)


if __name__ == "__main__":
    main()sum(y_true + y_pred, axis=[1,2])
        dice  = tf.reduce_mean(1.0 - (2.0*inter + 1.0) / (denom + 1.0), axis=-1)
        return self.focal_weight * focal + self.dice_weight * dice

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma,
                    "focal_weight": self.focal_weight, "dice_weight": self.dice_weight})
        return cfg

@tf.keras.utils.register_keras_serializable(package="mangrove")
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.total_cm = self.add_weight(name="cm", shape=(self.num_classes, self.num_classes),
                                        initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.reshape(tf.reduce_sum(y_true, -1) > 0, [-1])          # drop ignored px
        yt = tf.boolean_mask(tf.reshape(tf.argmax(y_true, -1, output_type=tf.int32), [-1]), valid)
        yp = tf.boolean_mask(tf.reshape(tf.argmax(y_pred, -1, output_type=tf.int32), [-1]), valid)
        cm = tf.math.confusion_matrix(yt, yp, num_classes=self.num_classes, dtype=tf.float32)
        return self.total_cm.assign_add(cm)
    def result(self):
        cm = self.total_cm
        diag = tf.linalg.diag_part(cm)
        union = tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - diag
        present = tf.cast(union > 0, tf.float32)                          # absent classes excluded
        iou = tf.where(union > 0, diag / (union + 1e-7), tf.zeros_like(union))
        return tf.reduce_sum(iou) / (tf.reduce_sum(present) + 1e-7)
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))
    def get_config(self):
        cfg = super().get_config(); cfg["num_classes"] = self.num_classes; return cfg

# ─────────────────────────────────────────────
# 8. MODEL: dual-branch, 4 levels, SE channel-attention (learns NIR/SWIR importance)
# ─────────────────────────────────────────────
def se_block(x, ratio=8, name=None):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    s = layers.Dense(max(c // ratio, 8), activation="relu", name=f"{name}_fc1")(s)
    s = layers.Dense(c, activation="sigmoid", name=f"{name}_fc2")(s)
    s = layers.Reshape((1, 1, c), name=f"{name}_rs")(s)
    return layers.multiply([x, s], name=f"{name}_scale")

def conv_block(x, filters, dropout=0.1, name=None):
    shortcut = layers.Conv2D(filters, 1, padding="same", name=f"{name}_proj")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.add([x, shortcut], name=f"{name}_add")
    x = layers.Activation("relu", name=f"{name}_out")(x)
    return se_block(x, name=f"{name}_se")

def encoder(x, widths, prefix):
    skips = []
    for i, f in enumerate(widths):
        x = conv_block(x, f, name=f"{prefix}_e{i}")
        skips.append(x)
        x = layers.MaxPooling2D(name=f"{prefix}_p{i}")(x)
    return x, skips

def build_model(cfg, s1_stats, s2_stats, num_classes=None, head_name="logits"):
    C  = num_classes or cfg["num_classes"]
    ps = cfg["patch_size"]
    s1_in  = layers.Input((ps, ps, 4),  name="s1_input")
    s2_in  = layers.Input((ps, ps, 10), name="s2_input")
    scl_in = layers.Input((ps, ps, 1),  name="scl_input")

    s1f, s2f = FeatureEngineer(cfg["invalid_scl"], cfg["cloud_dilate"],
                               name="feature_engineer")([s1_in, s2_in, scl_in])
    s1f = layers.Normalization(mean=list(s1_stats[0]), variance=list(s1_stats[1]), name="s1_norm")(s1f)
    s2f = layers.Normalization(mean=list(s2_stats[0]), variance=list(s2_stats[1]), name="s2_norm")(s2f)

    x1, sk1 = encoder(s1f, cfg["s1_widths"], "s1")
    x2, sk2 = encoder(s2f, cfg["s2_widths"], "s2")

    b = layers.concatenate([x1, x2], name="fuse_bottleneck")
    b = conv_block(b, cfg["bottleneck"], dropout=0.3, name="bottleneck")

    x = b
    for i, f in enumerate(cfg["dec_widths"]):
        x = layers.UpSampling2D(interpolation="bilinear", name=f"dec_up{i}")(x)
        x = layers.concatenate([x, sk1[-(i+1)], sk2[-(i+1)]], name=f"dec_cat{i}")
        x = conv_block(x, f, name=f"dec_c{i}")

    out = layers.Conv2D(C, 1, padding="same", dtype="float32", name=head_name)(x)
    out = layers.Activation("softmax", dtype="float32", name="softmax")(out)
    return models.Model([s1_in, s2_in, scl_in], out, name=f"MangroveFusionV3_{C}cls")

# ─────────────────────────────────────────────
# 9. CALLBACKS / EVAL
# ─────────────────────────────────────────────
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def evaluate_per_class(model, val_ds, cfg):
    from sklearn.metrics import confusion_matrix
    C = cfg["num_classes"]; cm = np.zeros((C, C), np.int64)
    for inputs, y in val_ds:
        p = model.predict(inputs, verbose=0).argmax(-1)
        t = y.numpy().argmax(-1); v = y.numpy().sum(-1) > 0
        cm += confusion_matrix(t[v].ravel(), p[v].ravel(), labels=list(range(C)))
    diag = np.diag(cm); union = cm.sum(0) + cm.sum(1) - diag
    iou = np.where(union > 0, diag / np.maximum(union, 1), np.nan)
    print("\n── Per-class results ──")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:18s} IoU={iou[i]:.3f}  px={cm[i].sum():,}")
    print(f"  mIoU={np.nanmean(iou):.3f}")
    np.savetxt(os.path.join(WORK_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    return cm, iou

# ─────────────────────────────────────────────
# 10. VISUALIZATION
# ─────────────────────────────────────────────
def _stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history.get("loss", []), label="train"); ax1.plot(history.history.get("val_loss", []), label="val")
    ax1.set_title("Focal+Dice Loss"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(history.history.get("mean_iou", []), label="train"); ax2.plot(history.history.get("val_mean_iou", []), label="val")
    ax2.set_title("Mean IoU"); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "training_history.png"), dpi=150); plt.show()

def visualize_predictions(model, val_ds, n_samples=4):
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
    titles = ["S2 RGB (masked)", "SAR (VV,VH,VH-VV)", "Ground Truth", "Prediction"]
    for a, t in zip(axes[0], titles): a.set_title(t)
    cmap = mcolors.ListedColormap(CLASS_COLORS)
    count = 0
    for inputs, y_batch in val_ds.take(3):
        preds = model.predict(inputs, verbose=0)
        for i in range(min(len(y_batch), n_samples - count)):
            s2 = inputs["s2_input"][i].numpy(); s1 = inputs["s1_input"][i].numpy()
            y = y_batch[i].numpy().argmax(-1); p = preds[i].argmax(-1)
            axes[count,0].imshow(_stretch(s2[:,:,[2,1,0]]))                       # B4,B3,B2
            sar = np.stack([_stretch(s1[:,:,0]), _stretch(s1[:,:,1]), _stretch(s1[:,:,2])], -1)
            axes[count,1].imshow(sar)
            axes[count,2].imshow(y, cmap=cmap, vmin=0, vmax=4)
            axes[count,3].imshow(p, cmap=cmap, vmin=0, vmax=4)
            for a in axes[count]: a.axis("off")
            count += 1
            if count >= n_samples: break
        if count >= n_samples: break
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "predictions_preview.png"),
                                    dpi=150, bbox_inches="tight"); plt.show()

# ─────────────────────────────────────────────
# 11. TILED INFERENCE (edge-safe, batched, optional TTA) + FLOOD RISK EXPORT
# ─────────────────────────────────────────────
def _tile_starts(dim, ps, stride):
    if dim <= ps: return [0]
    starts = list(range(0, dim - ps + 1, stride))
    if starts[-1] != dim - ps: starts.append(dim - ps)     # FIX: always cover far edge
    return starts

def predict_tiled(model, s1, s2, scl=None, cfg=CONFIG, batch_size=None, tta=None):
    """s1:(H,W,2|4) dB, s2:(H,W,10) 0-1, scl:(H,W)|None -> class map, prob map"""
    batch_size = batch_size or cfg["infer_batch"]; tta = cfg["infer_tta"] if tta is None else tta
    ps, C = cfg["patch_size"], cfg["num_classes"]
    h, w = s2.shape[:2]
    if s1.shape[-1] == 2:
        s1 = np.concatenate([s1, np.zeros_like(s1)], -1)
    scl = np.full((h, w, 1), 4.0, np.float32) if scl is None else scl[..., None].astype(np.float32)

    pad_h, pad_w = max(0, ps - h), max(0, ps - w)
    if pad_h or pad_w:
        s1  = np.pad(s1,  ((0,pad_h),(0,pad_w),(0,0)))
        s2  = np.pad(s2,  ((0,pad_h),(0,pad_w),(0,0)))
        scl = np.pad(scl, ((0,pad_h),(0,pad_w),(0,0)), constant_values=4)
    H, W = s2.shape[:2]
    acc = np.zeros((H, W, C), np.float32); cnt = np.zeros((H, W), np.float32)

    # TTA variants: (forward, inverse)
    variants = [(lambda x: x, lambda p: p),
                (lambda x: x[:, :, ::-1], lambda p: p[:, :, ::-1]),
                (lambda x: x[:, ::-1],    lambda p: p[:, ::-1]),
                (lambda x: np.rot90(x, 1, (1, 2)), lambda p: np.rot90(p, -1, (1, 2)))] if tta \
                else [(lambda x: x, lambda p: p)]

    coords = [(i, j) for i in _tile_starts(H, ps, ps//2) for j in _tile_starts(W, ps, ps//2)]
    for k in range(0, len(coords), batch_size):
        chunk = coords[k:k+batch_size]
        b1 = np.stack([s1[i:i+ps, j:j+ps]  for i, j in chunk])
        b2 = np.stack([s2[i:i+ps, j:j+ps]  for i, j in chunk])
        bs = np.stack([scl[i:i+ps, j:j+ps] for i, j in chunk])
        for fwd, inv in variants:
            pr = model.predict({"s1_input": fwd(b1), "s2_input": fwd(b2), "scl_input": fwd(bs)},
                               verbose=0, batch_size=len(chunk))
            pr = inv(pr)
            for n, (i, j) in enumerate(chunk):
                acc[i:i+ps, j:j+ps] += pr[n]; cnt[i:i+ps, j:j+ps] += 1
    probs = acc / np.maximum(cnt[..., None], 1.0)
    return probs.argmax(-1).astype(np.uint8)[:h, :w], probs[:h, :w]

def flood_risk_map(probs):
    """Flood/tidal-prone probability = P(Tidal Water) + P(Permanent Flood)."""
    return (probs[..., 1] + probs[..., 2]).astype(np.float32)

def export_geotiff(reference_tif, array, out_path, dtype="uint8", nodata=None):
    import rasterio
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate")
    if nodata is not None: profile.update(nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    print(f"Saved: {out_path}")

# ─────────────────────────────────────────────
# 12. SEN1FLOODS11 PRETRAINING (Bonafilia et al., 2020) + WEIGHT TRANSFER
# ─────────────────────────────────────────────
def convert_sen1floods11(root, out_dir):
    """root must contain S1Hand/, S2Hand/, LabelHand/ (official Sen1Floods11 layout)."""
    import rasterio
    os.makedirs(out_dir, exist_ok=True)
    s2_keep = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]     # 13-band chip -> our 10 bands
    n = 0
    for s1_path in sorted(glob.glob(os.path.join(root, "S1Hand", "*.tif"))):
        chip = os.path.basename(s1_path).replace("_S1Hand.tif", "")
        s2_path  = os.path.join(root, "S2Hand",    chip + "_S2Hand.tif")
        lab_path = os.path.join(root, "LabelHand", chip + "_LabelHand.tif")
        if not (os.path.exists(s2_path) and os.path.exists(lab_path)): continue
        with rasterio.open(s1_path)  as f: s1  = f.read().transpose(1, 2, 0)
        with rasterio.open(s2_path)  as f: s2  = f.read()[s2_keep].transpose(1, 2, 0)
        with rasterio.open(lab_path) as f: lab = f.read(1)
        lab = np.where(lab < 0, 255, lab).astype(np.uint8)          # -1 invalid -> 255
        np.savez_compressed(os.path.join(out_dir, chip + ".npz"),
                            s1=s1.astype(np.float32), s2=s2.astype(np.float32), label=lab)
        n += 1
    print(f"Converted {n} Sen1Floods11 chips -> {out_dir}")

def transfer_trunk_weights(src_model, dst_model):
    src = {l.name: l for l in src_model.layers}
    moved = 0
    for l in dst_model.layers:
        if l.name.endswith("_norm") or not l.weights:      # keep target-domain norm stats
            continue
        s = src.get(l.name)
        if s and [w.shape for w in s.weights] == [w.shape for w in l.weights]:
            l.set_weights(s.get_weights()); moved += 1
    print(f"Transferred weights for {moved} layers (head excluded).")

def pretrain_sen1floods11(cfg=CONFIG):
    files = sorted(glob.glob(os.path.join(cfg["pretrain_dir"], "*.npz")))
    assert files, f"No npz in {cfg['pretrain_dir']} — run convert_sen1floods11() first"
    n_val = max(1, int(0.1 * len(files)))
    cfg2 = dict(cfg, num_classes=2)
    s1m, s1v, s2m, s2v, _ = compute_stats(files[:-n_val], cfg2, max_patches=200)
    ds_tr = make_dataset(files[:-n_val], cfg2, True,  8000)
    ds_va = make_dataset(files[-n_val:], cfg2, False, 1500)
    with strategy.scope():
        model = build_model(cfg2, (s1m, s1v), (s2m, s2v), num_classes=2, head_name="pre_logits")
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
                      loss=FocalDiceLoss(gamma=2.0), metrics=[MeanIoU(2)])
    model.fit(ds_tr, validation_data=ds_va, epochs=25, callbacks=[
        tf.keras.callbacks.EarlyStopping("val_mean_iou", mode="max", patience=5, restore_best_weights=True),
        MemoryCleanupCallback()])
    model.save(cfg["pretrained_path"])
    print(f"Pretrained water model saved: {cfg['pretrained_path']}")

# ─────────────────────────────────────────────
# 13. TRAINING
# ─────────────────────────────────────────────
def train(cfg=CONFIG):
    print("\n=== MANGROVE FUSION v3 — TRAINING ===\n")
    train_files = sorted(glob.glob(os.path.join(cfg["train_dir"], "*.npz")))
    val_files   = sorted(glob.glob(os.path.join(cfg["val_dir"], "*.npz")))
    assert train_files and val_files, "Set train_dir/val_dir in CONFIG"
    print(f"train scenes: {len(train_files)} | val scenes: {len(val_files)} | global batch: {GLOBAL_BATCH}")

    s1m, s1v, s2m, s2v, alpha = load_or_compute_stats(train_files, cfg)

    ds_train = make_dataset(train_files, cfg, True,  cfg["max_train_patches"],
                            cache_path=os.path.join(WORK_DIR, "cache_tr") if cfg["use_disk_cache"] else None)
    ds_val   = make_dataset(val_files,   cfg, False, cfg["max_val_patches"],
                            cache_path=os.path.join(WORK_DIR, "cache_va") if cfg["use_disk_cache"] else None)

    with strategy.scope():
        model = build_model(cfg, (s1m, s1v), (s2m, s2v))
        if os.path.exists(cfg["pretrained_path"]):
            print("Loading Sen1Floods11 pretrained trunk...")
            pre = tf.keras.models.load_model(cfg["pretrained_path"])
            transfer_trunk_weights(pre, model)
            del pre; gc.collect()

        est_steps = max(1, cfg["max_train_patches"] // GLOBAL_BATCH)
        lr_sched = tf.keras.optimizers.schedules.CosineDecay(cfg["lr"], cfg["epochs"] * est_steps, alpha=0.01)
        opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=cfg["weight_decay"])
        # NOTE: no LossScaleOptimizer — Keras 3 scales automatically under mixed_float16
        model.compile(optimizer=opt,
                      loss=FocalDiceLoss(alpha=alpha, gamma=cfg["focal_gamma"],
                                         focal_weight=cfg["focal_weight"], dice_weight=cfg["dice_weight"]),
                      metrics=[MeanIoU(cfg["num_classes"])], jit_compile=cfg["jit_compile"])
    model.summary(print_fn=lambda s: print(s) if "model" not in s.lower() else None)

    callbacks = [
        tf.keras.callbacks.EarlyStopping("val_mean_iou", mode="max", patience=12,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, "val_mean_iou", mode="max",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.BackupAndRestore(os.path.join(WORK_DIR, "backup")),  # Kaggle preemption
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(os.path.join(WORK_DIR, "training_log.csv")),
        tf.keras.callbacks.TensorBoard(os.path.join(WORK_DIR, "tb")),
        MemoryCleanupCallback(),
    ]
    history = model.fit(ds_train, validation_data=ds_val, epochs=cfg["epochs"], callbacks=callbacks)

    model.save(MODEL_PATH)   # final (best weights restored by EarlyStopping)
    plot_training_history(history)
    evaluate_per_class(model, ds_val, cfg)
    visualize_predictions(model, ds_val)
    print(f"\nModel saved: {MODEL_PATH}")
    return model

# ─────────────────────────────────────────────
# 14. SMOKE TEST (dummy data — verify the graph end-to-end on Kaggle first)
# ─────────────────────────────────────────────
def smoke_test(cfg=CONFIG):
    print("\n=== SMOKE TEST (dummy data) ===")
    ps = cfg["patch_size"]
    def gen():
        for _ in range(64):
            yield (np.random.uniform(-30, 0, (ps, ps, 4)).astype(np.float32),
                   np.random.uniform(0, 0.5, (ps, ps, 10)).astype(np.float32),
                   np.random.choice([4,5,6,8,9], (ps, ps, 1)).astype(np.float32),
                   np.random.choice([0,1,2,3,4,255], (ps, ps)).astype(np.int32))
    sig = (tf.TensorSpec((ps,ps,4), tf.float32), tf.TensorSpec((ps,ps,10), tf.float32),
           tf.TensorSpec((ps,ps,1), tf.float32), tf.TensorSpec((ps,ps), tf.int32))
    def mk():
        return (tf.data.Dataset.from_generator(gen, output_signature=sig)
                .map(lambda a,b,c,d: ({"s1_input":a,"s2_input":b,"scl_input":c},
                                      tf.one_hot(tf.where(d==255,-1,d), cfg["num_classes"])))
                .batch(GLOBAL_BATCH, drop_remainder=True).prefetch(1))
    s1m = np.zeros(6); s1v = np.ones(6); s2m = np.zeros(17); s2v = np.ones(17)
    with strategy.scope():
        model = build_model(cfg, (s1m, s1v), (s2m, s2v))
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3),
                      loss=FocalDiceLoss(), metrics=[MeanIoU(cfg["num_classes"])])
    model.fit(mk(), validation_data=mk(), epochs=2)
    model.save(MODEL_PATH)
    m2 = tf.keras.models.load_model(MODEL_PATH)   # verify serialization round-trip
    print("Smoke test OK — model trains, saves, and reloads with custom objects.")

# ─────────────────────────────────────────────
# 15. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    MODE = "smoke"        # "smoke" | "pretrain" | "train" | "infer"

    if MODE == "smoke":
        smoke_test()
    elif MODE == "pretrain":
        # convert_sen1floods11("/path/to/Sen1Floods11_root", CONFIG["pretrain_dir"])  # run once
        pretrain_sen1floods11()
    elif MODE == "train":
        train()
    elif MODE == "infer":
        # Example (local i7 machine is fine — CPU, batch 8):
        # import rasterio
        # with rasterio.open("scene_s2.tif") as f: s2 = f.read().transpose(1,2,0) / 1e4
        # with rasterio.open("scene_s1.tif") as f: s1 = f.read().transpose(1,2,0)
        # model = tf.keras.models.load_model(MODEL_PATH)
        # cls, probs = predict_tiled(model, s1, s2)
        # export_geotiff("scene_s2.tif", cls, "classes.tif", "uint8", nodata=255)
        # export_geotiff("scene_s2.tif", flood_risk_map(probs), "flood_risk.tif", "float32")
        pass