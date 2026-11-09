"""
GEE 38-band T1/T2 GeoTIFF -> v8 pair-NPZ converter (Windowed, Resumable, Seamless Stride)

Fixes Applied:
  1. Window Overlap: Pads READ_WINDOW by (PATCH_SIZE - STRIDE) to preserve 
     continuous 50% overlap across window boundaries.
  2. Resume Logic: Checks os.path.exists() FIRST before performing array operations.
  3. Edge Safeguard: Clamps window reads at far edges (H, W) so non-multiple 
     dimensions are never silently dropped.
  4. Channels-First Schema: Keeps multi-channel arrays in (C, H, W) format for direct
     ingestion into PyTorch tensors without runtime transposition.
"""
import os
os.environ["GDAL_CACHEMAX"] = "512"
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

import rasterio
from rasterio.windows import Window
import numpy as np
import glob
import gc
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TIF_FOLDER = "/media/ayush/Chocolate/CANDIES/ULTIMATRIX/RAW/"
OUT_DIR = "kaggle_npz_v8/"

PATCH_SIZE = 256
STRIDE = 128                  # 50% overlap between saved chips
READ_WINDOW = 2048            # Base window size for chunked I/O

DATE_T1 = np.float32(2018.5)
DATE_T2 = np.float32(2023.5)

MAX_IGNORE_FRAC = 0.60        # Discard chip if >60% of label pixels are 255

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# BAND SLICES (Indices 0-37)
# ==============================================================================
S1_T1, S1STD_T1 = slice(0, 2), slice(2, 4)
S2_T1_SLICE, SCL_T1_IDX = slice(4, 14), 14
S1_T2, S1STD_T2 = slice(15, 17), slice(17, 19)
S2_T2_SLICE, SCL_T2_IDX = slice(19, 29), 29
AUX_SLICE = slice(30, 36)
LABEL_IDX, CHANGE_IDX = 36, 37


def _tile_starts(dim, size, stride):
    """Generates continuous start coordinates covering full dimension including far edge."""
    if dim <= size:
        return [0]
    starts = list(range(0, dim - size + 1, stride))
    if starts[-1] != dim - size:
        starts.append(dim - size)
    return starts


def process_tif(tif_path, out_dir, stem):
    saved, skipped_exist, skipped_ignore = 0, 0, 0
    
    with rasterio.open(tif_path) as src:
        H, W, n_bands = src.height, src.width, src.count
        print(f"  dims: {W}x{H}, bands: {n_bands}, "
              f"tiled: {src.profile.get('tiled')}, compress: {src.profile.get('compress')}")
        
        if n_bands != 38:
            print(f"  WARNING: expected 38 bands, got {n_bands} -- skipping file")
            return 0, 0, 0

        # Calculate global tile starts across full raster to ensure seamless stride
        y_starts = _tile_starts(H, PATCH_SIZE, STRIDE)
        x_starts = _tile_starts(W, PATCH_SIZE, STRIDE)
        
        # Group global tile starts into READ_WINDOW chunks
        for wy in range(0, H, READ_WINDOW):
            wy_end = min(wy + READ_WINDOW, H)
            # Find all tiles that start within this Y window
            y_tiles = [y for y in y_starts if wy <= y < wy_end]
            if not y_tiles:
                continue

            for wx in range(0, W, READ_WINDOW):
                wx_end = min(wx + READ_WINDOW, W)
                x_tiles = [x for x in x_starts if wx <= x < wx_end]
                if not x_tiles:
                    continue

                # Read window padded to cover the full width/height of contained tiles
                read_y2 = min(H, max(y + PATCH_SIZE for y in y_tiles))
                read_x2 = min(W, max(x + PATCH_SIZE for x in x_tiles))
                read_h = read_y2 - wy
                read_w = read_x2 - wx

                t0 = time.time()
                window = Window(wx, wy, read_w, read_h)
                # Keep in Channels-First format: (38, read_h, read_w)
                block = src.read(window=window)
                read_s = time.time() - t0

                n_chips_this_window = 0
                for y in y_tiles:
                    iy = y - wy  # Local Y inside block
                    for x in x_tiles:
                        ix = x - wx  # Local X inside block
                        
                        out_path = os.path.join(out_dir, f"{stem}_w{y}_{x}.npz")
                        
                        # FIX: Check existing file BEFORE computing array operations
                        if os.path.exists(out_path):
                            skipped_exist += 1
                            continue

                        # Extract raw chip slice (38, 256, 256)
                        chip = block[:, iy:iy + PATCH_SIZE, ix:ix + PATCH_SIZE]

                        # 1. Segmentation Label: Fill NaN with 255 (Ignore Index) BEFORE uint8 cast
                        raw_label = chip[LABEL_IDX, :, :]
                        label = np.nan_to_num(raw_label, nan=255.0).astype(np.uint8)

                        # Check ignore fraction using cleaned label
                        if np.mean(label == 255) > MAX_IGNORE_FRAC:
                            skipped_ignore += 1
                            continue

                        # 2. Change Label: Fill NaN with 0 (No Change) BEFORE uint8 cast
                        raw_change = chip[CHANGE_IDX, :, :]
                        change = np.nan_to_num(raw_change, nan=0.0).astype(np.uint8)

                        # 3. Scene Classification Layers (SCL): Fill NaN with 0 (NoData)
                        scl_t1 = np.nan_to_num(chip[SCL_T1_IDX, :, :], nan=0.0).astype(np.uint8)
                        scl_t2 = np.nan_to_num(chip[SCL_T2_IDX, :, :], nan=0.0).astype(np.uint8)

                        # 4. Feature Bands: Replace any NaN with 0.0 to prevent NaN loss in PyTorch
                        s1_t1 = np.nan_to_num(chip[S1_T1, :, :], nan=0.0)
                        s1_t1_std = np.nan_to_num(chip[S1STD_T1, :, :], nan=0.0)
                        s2_t1 = np.nan_to_num(chip[S2_T1_SLICE, :, :], nan=0.0)
                        
                        s1_t2 = np.nan_to_num(chip[S1_T2, :, :], nan=0.0)
                        s1_t2_std = np.nan_to_num(chip[S1STD_T2, :, :], nan=0.0)
                        s2_t2 = np.nan_to_num(chip[S2_T2_SLICE, :, :], nan=0.0)
                        
                        aux = np.nan_to_num(chip[AUX_SLICE, :, :], nan=0.0)

                        # Save clean Channels-First arrays (C, H, W)
                        np.savez_compressed(
                            out_path,
                            s1=s1_t1.astype(np.float32),
                            s1_std=s1_t1_std.astype(np.float32),
                            s2=s2_t1.astype(np.float32),
                            scl=scl_t1,
                            s1_t2=s1_t2.astype(np.float32),
                            s1_t2_std=s1_t2_std.astype(np.float32),
                            s2_t2=s2_t2.astype(np.float32),
                            scl_t2=scl_t2,
                            aux=aux.astype(np.float32),
                            aux_t2=aux.copy().astype(np.float32),
                            label=label,
                            change=change,
                            date_t1=DATE_T1,
                            date_t2=DATE_T2,
                        )
                        saved += 1
                        n_chips_this_window += 1

                print(f"  window (x={wx},y={wy}) read {read_s:.2f}s -> "
                      f"{n_chips_this_window} saved (total new: {saved})", flush=True)
                del block
        gc.collect()
        
    return saved, skipped_exist, skipped_ignore


def process_tifs():
    tif_files = sorted(glob.glob(os.path.join(TIF_FOLDER, "*.tif")))
    if not tif_files:
        print(f"[ERROR] No .tif files found in {TIF_FOLDER}")
        return

    print(f"Found {len(tif_files)} GeoTIFF files to process.")
    total_saved = 0
    for tif_path in tif_files:
        stem = os.path.splitext(os.path.basename(tif_path))[0]
        print(f"\nProcessing {os.path.basename(tif_path)}...")
        saved, skip_exist, skip_ignore = process_tif(tif_path, OUT_DIR, stem)
        total_saved += saved
        print(f"  file summary: saved={saved}, "
              f"skipped(exists)={skip_exist}, skipped(ignore)={skip_ignore}")

    n_out = len(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    print(f"\n[SUCCESS] {n_out} .npz files total in {OUT_DIR} "
          f"({total_saved} newly written this run)")


if __name__ == "__main__":
    process_tifs()