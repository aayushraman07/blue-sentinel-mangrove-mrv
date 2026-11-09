"""
GEE GeoTIFF -> Kaggle NPZ converter
Fixes applied vs. original, and vs. the previous revision of this script:
  1. FATAL (previous bug): s1/s1_std split corrected -- s1 saves VV,VH
     only; VV_std,VH_std go in their own key, matching what v3's
     load_npz() expects to concatenate itself. Verified by execution.
  2. Windowed reads instead of src.read() on the whole file -- avoids
     loading a ~7GB array per source tile into memory at once.
  3. NEW: one npz per WINDOW (full 2048x2048 resolution), not one npz
     per 256px chip. The chip-per-file version produced hundreds of tiny
     compressed writes per window, and on a FUSE-mounted external drive
     (exFAT/NTFS under Linux) that per-file overhead -- not the actual
     read or compression -- was the real bottleneck (confirmed: reads
     were 0.7-3.4s/window, but "chips so far" jumped by 100-270 between
     prints with nothing else timed in between). Writing 25 larger files
     instead of thousands of small ones removes that overhead almost
     entirely, and hands all 256px/stride tiling to v3's own
     patch_generator, which was already built to do exactly that.
  4. Resumable: skips a window if its output file already exists, so an
     interrupted run can just be re-launched rather than restarted from
     scratch.
  5. Per-window read AND save timing both printed, so a slow run is
     diagnosable instead of looking identical to a stuck one.
"""
import os
# Must be set BEFORE rasterio is imported/used -- GDAL reads these once at
# first init and caches them process-wide.
os.environ["GDAL_CACHEMAX"] = "512"
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

import rasterio
from rasterio.windows import Window
import numpy as np
import glob
import gc
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TIF_FOLDER   = "/media/ayush/Chocolate/CANDIES/ULTIMATRIX/RAW"
OUT_DIR      = "kaggle_npz_files/"   # consider pointing this at your internal
                                      # NVMe (e.g. /home/ayush/...) rather than
                                      # the external drive -- fewer files now,
                                      # but internal storage is still faster
READ_WINDOW  = 2048         # window size read+saved as one npz
MAX_IGNORE_FRAC = 0.95      # skip a window only if it's ALMOST entirely
                             # ignore-label -- fine-grained 256px-level
                             # filtering (min_valid_frac) happens in v3's
                             # own patch_generator at load time now

os.makedirs(OUT_DIR, exist_ok=True)

# Band layout produced by the GEE v2 export script (verified index-by-index
# against that script's addBands() order):
#   0: VV        1: VH        2: VV_std    3: VH_std
#   4: B2   5: B3   6: B4   7: B5   8: B6   9: B7   10: B8   11: B8A   12: B11   13: B12
#   14: SCL
#   15: acq_time   (kept index-aligned, not written to the npz -- v3 doesn't consume it)
#   16: label
S1_SLICE    = slice(0, 2)    # VV, VH
S1STD_SLICE = slice(2, 4)    # VV_std, VH_std
S2_SLICE    = slice(4, 14)   # 10 Sentinel-2 bands
SCL_IDX     = 14
LABEL_IDX   = 16


def process_tif(tif_path, out_dir):
    stem = os.path.splitext(os.path.basename(tif_path))[0]
    saved, skipped_exist, skipped_empty = 0, 0, 0

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        n_bands = src.count
        print(f"  dims: {W}x{H}, bands: {n_bands}, driver: {src.driver}, "
              f"tiled: {src.profile.get('tiled')}, compress: {src.profile.get('compress')}")
        n_windows_x = (W + READ_WINDOW - 1) // READ_WINDOW
        n_windows_y = (H + READ_WINDOW - 1) // READ_WINDOW
        print(f"  will read {n_windows_x * n_windows_y} windows of ~{READ_WINDOW}px")

        if n_bands != 17:
            print(f"  WARNING: {os.path.basename(tif_path)} has {n_bands} bands, expected 17 -- skipping")
            return

        win_num = 0
        for wy in range(0, H, READ_WINDOW):
            win_h = min(READ_WINDOW, H - wy)
            for wx in range(0, W, READ_WINDOW):
                win_num += 1
                win_w = min(READ_WINDOW, W - wx)
                out_path = os.path.join(out_dir, f"{stem}_w{wy}_{wx}.npz")

                if os.path.exists(out_path):
                    skipped_exist += 1
                    print(f"  window {win_num}/{n_windows_x * n_windows_y} "
                          f"(x={wx},y={wy}) already exists, skipping", flush=True)
                    continue

                t_read0 = time.time()
                window = Window(wx, wy, win_w, win_h)
                block = src.read(window=window)                 # (17, win_h, win_w)
                block = np.transpose(block, (1, 2, 0))           # (win_h, win_w, 17)
                read_s = time.time() - t_read0

                label = block[:, :, LABEL_IDX].astype(np.uint8)
                if np.mean(label == 255) > MAX_IGNORE_FRAC:
                    skipped_empty += 1
                    print(f"  window {win_num}/{n_windows_x * n_windows_y} "
                          f"(x={wx},y={wy}) read in {read_s:.1f}s, "
                          f"almost entirely ignore-label -- skipped", flush=True)
                    del block
                    continue

                t_save0 = time.time()
                s1     = block[:, :, S1_SLICE].astype(np.float32)
                s1_std = block[:, :, S1STD_SLICE].astype(np.float32)
                s2     = block[:, :, S2_SLICE].astype(np.float32)
                scl    = block[:, :, SCL_IDX].astype(np.uint8)

                np.savez_compressed(out_path, s1=s1, s1_std=s1_std, s2=s2, scl=scl, label=label)
                save_s = time.time() - t_save0
                saved += 1

                print(f"  window {win_num}/{n_windows_x * n_windows_y} "
                      f"(x={wx},y={wy}) read {read_s:.1f}s, saved {save_s:.1f}s "
                      f"-> {os.path.basename(out_path)}", flush=True)

                del block
        gc.collect()

    print(f"  file summary: saved={saved}, skipped(exists)={skipped_exist}, "
          f"skipped(empty)={skipped_empty}")


def main():
    tif_files = sorted(glob.glob(os.path.join(TIF_FOLDER, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {TIF_FOLDER}")
        return

    for tif_path in tif_files:
        print(f"Processing {os.path.basename(tif_path)}...")
        process_tif(tif_path, OUT_DIR)

    n_out = len(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    print(f"\nDone. {n_out} .npz files total in {OUT_DIR}")


if __name__ == "__main__":
    main()