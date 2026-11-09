"""
================================================================================
 BLUE SENTINEL APEX v11.6.2  "MANGROVE ECO-MONITOR — UNIFIED SPATIO-TEMPORAL SSM"
 KAGGLE EDITION of blue-sentinel-v11.py — single self-contained file.
 Defaults already point at /kaggle/input/mangrove-v11/{train,val,test}/ and
 outputs land in /kaggle/working/. Tuned for a 16 GB T4 / P100 session
 (batch_size=2, grad-checkpointing, ~12h session cap with --resume).

--------------------------------------------------------------------------------
 HOW TO RUN ON KAGGLE (pick ONE)
--------------------------------------------------------------------------------
 1) PASTE this whole file into a notebook cell. It self-detects the notebook,
    prints this quick-start and defines everything (nothing auto-runs).
    Then, in the NEXT cell:

        main("smoke")                       # 2-5 min self-test (recommended!)
        main(["build"])                     # FIRST, if your dataset is the RAW
                                            # GEE export folder (blue_sentinel_
                                            # v11_raw TIFFs): packs it into
                                            # v11 NPZ scenes -> working/data
        main("train")                       # full training -> /kaggle/working
        main(["train", "--resume"])          # continue after a 12h session cut
        main(["train", "--pretrain", "/kaggle/working/pretrain_encoder_v11.pt"])
        main("pretrain")                     # SimMIM SSL pretrain
        main(["eval", "--test-dir", "/kaggle/input/mangrove-v11/test"])
        main(["export", "<scene_dir_or_npz>", "<ref.tif>", "/kaggle/working/out"])
        main(["smooth", "<dir_with_seg_probs_npz>"])

 2) UPLOAD this file as a Kaggle Dataset, then run in a cell:
        !python /kaggle/input/<your-dataset>/blue_sentinel_v11_kaggle.py smoke
        !python /kaggle/input/<your-dataset>/blue_sentinel_v11_kaggle.py train

 3) ADD as a Kaggle "Utility Script" (File -> Add utility script), then:
        from blue_sentinel_v11_kaggle import main
        main("train")

 NOTES
 - 'main' accepts a string or a list of CLI args (see argparse below); it
   ignores unknown args so the notebook kernel's '-f <kernel.json>' can't
   break it.
 - MULTI-GPU (2x T4): v11.4.7 restores DataParallel "mirror" mode across
   both GPUs (32 GB combined VRAM) with the v11.4.2-era bugs fixed at the
   root: per-replica autocast (torch autocast is thread-local, so the old
   code silently ran replica 1 in fp32 -> the 2nd T4 OOM'd), CPU-side
   DropPath RNG, and no forced allocator flags. Default batch is
   VRAM-aware (1 sample/GPU on 16 GB cards -> batch 2, effective 4 with
   grad_accum=2). OOM ladder: halve batch -> drop mirror -> halve
   grad_accum. An illegal memory access aborts with a --no-mirror hint
   (a poisoned CUDA context cannot be retried). --no-mirror = single GPU;
   checkpoints stay single-GPU-portable (no 'module.' prefix).
 - 30 GB RAM: v11.4.6 streams folder scenes through a chunked patcher
   (peak ~1 step array + a budget-sized tile buffer, not whole temporal cubes)
   and auto-runs 2 dataloader workers on Kaggle so decompression overlaps
   GPU compute (GPU stays busy, CPU no longer pegged at 100% alone).
   --workers N overrides.
 - RAW GEE EXPORTS -> main(["build"]): if your dataset is the RAW Drive
   folder from gee_dataset_builder_v11.py Part A ({aoi}_{year}.tif +
   {aoi}_static.tif — however Drive split it across download parts),
   'build' packs everything into v11 NPZ scenes under /kaggle/working/data
   with train/val/test AOI splits (--split "5,2,1" = AOIs per split,
   alphabetical; rasterio pip-installs automatically; re-runs skip packed
   scenes). Then main("train") picks them up automatically.
 - ZIP DATASETS ARE AUTO-HANDLED: Kaggle often uploads datasets as .zip
   archives. Before build/train/eval/pretrain, every *.zip under /kaggle/input
   is extracted to /kaggle/working/data (idempotent; extension-less Drive
   split archives are detected by magic bytes) and train/val/test dirs are
   auto-located wherever they landed (nested zips OK). Layout inside
   the zip must be: train/scene_x/step_YYYY.npz (plus val/, test/).
   Zips that hold scenes directly (no train/ folder) can't be split
   automatically -> re-zip per split or pass --train-dir/--val-dir.
 - Expected dataset (see gee_dataset_builder_v11.py to build it):
     /kaggle/input/mangrove-v11/train/scene_*/step_YYYY.npz
     /kaggle/input/mangrove-v11/val/...   /kaggle/input/mangrove-v11/test/...
   Heads whose label keys are absent auto-disable at startup.
 - Outputs: best_v11.pt (best mIoU), last_v11.pt (per-epoch, for --resume),
   final_calibrated_v11.pt, training_log.csv, config_v11.json, and
   confusion/metrics/... under /kaggle/working/.
================================================================================

 TECH REFERENCE
 REQUIRES: Python >= 3.10, PyTorch >= 2.1 (Kaggle's stock torch works).
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

  v11.5.5 (IPC removed from the training path + leak attribution) [KAGGLE]:
    - The ~71 MB/batch leak survived EVERYTHING: pin ON/OFF, sync/async H2D,
      mamba ON/OFF, full per-iteration del + gc.collect, malloc_trim.
      smaps/statm telemetry shows ANONYMOUS PRIVATE growth at exactly the
      batch transfer size with file-backed mappings flat — leaving the
      worker->main shared-memory IPC as the prime suspect (torch
      multiprocess tensor transport retaining received-segment mappings).
    - num_workers now defaults to 0 (was auto-2 on Kaggle): the training
      loop runs the streaming loader INLINE — no torch shm IPC, no worker
      forks (each fork also inherited ~2-5 GB of mapped mamba/torch
      modules). The resident/chunked loader is fast enough that worker
      overlap was marginal. --workers N re-enables multiprocessing to
      A/B the IPC theory on demand.
    - Heartbeat now prints the smaps_rollup ANONYMOUS total alongside
      shared: if RSS still climbs with workers=0, 'anon' growing means
      heap/driver; any other growth means retained mappings — the next
      run pinpoints the retainer class definitively.
  v11.6.2 (training graphs + dataset visuals) [KAGGLE EDITION]:
    - NEW plot_training_curves(): report-ready 2x2 PNG from
      training_log.csv — loss (log), mIoU with best-epoch star, LR, aux
      heads (change-F1 + carbon/hydro/canopy MAEs), curriculum phases
      shaded. Regenerates automatically after every epoch (one CSV read +
      one PNG) and at train() end; fault-isolated, matplotlib optional.
    - NEW plot_dataset_overview(): renders what the model actually sees
      for up to 4 scenes — Sentinel-2 true colour (B4,B3,B2, 2-98%
      stretch), the class-label map in CLASS_COLORS, and the label-over-
      RGB blend, with a class legend. Runs once at train() start.
    - Both wired into train(); assets land in WORK_DIR
      (training_curves.png / dataset_overview.png) for git/report use.
  v11.6.1 (the agreed weekly plan: 12 epochs, batch=2 single GPU, all-
  resident loader) [KAGGLE EDITION]:
    - COMPRESSED CURRICULUM (user-agreed): epochs 100 -> 12, phases
      25/60 -> 3/8, ema_decay 0.999 -> 0.99. Sized for the 5-scene dataset
      and the 45 h/week GPU budget: 12 epochs x ~5 h ~ 1.2-1.5 weeks of
      quota. The old 100-epoch schedule needed ~900 GPU-hours (~20 weeks).
    - BATCH: 16 GB cards now default to SINGLE-GPU batch=2, grad_accum=2
      (effective batch 4 - same gradient statistics as the batch=1 run,
      so the epoch-1 checkpoint resumes cleanly). The mamba scan kernel
      is sequential in time but has spare lane capacity -> 2 samples cost
      near 1 sample's kernel time -> ~1.8x faster epochs. DataParallel
      stays OFF (batch=2 through DP was the confirmed leak config, and
      2 samples/GPU under DP OOM'd twice). OOM ladder unchanged as the
      safety net; --batch-size/--mirror still allow experiments.
    - SCENE CACHE 2400 -> 4000 MB: all five scenes (max 3.6 GB) now take
      the resident path -> zero chunked reloads per epoch; the host RAM
      headroom (12-13 GB free of 30) funds it.
    - Context from the epoch-1 checkpoint forensics: train mIoU 0.371 /
      val 0.088 with the EMA only ~27% warmed - the compressed schedule
      + faster EMA warmup make val meaningful from ~epoch 2.
  v11.6.0 (early stopping, EMA-warmup aware) [KAGGLE EDITION]:
    - NEW: early stopping on val mIoU — cfg.early_stop_patience (5) epochs
      without a new best, cfg.early_stop_min_epoch (8) arm point. The arm
      point matters: val is evaluated with EMA weights, which at
      ema_decay=0.999 need ~10 epochs (~4600 steps) to absorb the
      training — a naive patience check fires during that warmup plateau
      and kills the run right before val starts climbing (the epoch-1
      checkpoint forensics showed the EMA carried only ~27% of the
      learning). When it fires, the post-loop calibration still runs and
      best_v11.pt is untouched.
    - Weekly-budget context (45 h/week = 6.6 epochs at 256px): the early
      stop protects the quota from a plateaued model — worst case it
      saves ~3-5 idle sessions.
    - Plan agreed with the user: finish this week's quota on the current
      256px run (phase 1, seg heads — a solid seg-only model by week's
      end), download best/last + training_log.csv at each session wall,
      then next week resume under Path 1 (compressed curriculum 3/8/12 +
      ema_decay 0.99) — the phase schedule self-adjusts on resume because
      phase_for_epoch reads the CURRENT cfg.
  v11.5.9 (loader overlap restored — kill the 26 s/micro-batch wall) [KAGGLE]:
    - MEASURED (user telemetry): 260 steps in 8h22m = ~26 s per
      micro-batch, i.e. ~80% of wall time was the INLINE loader (workers=0
      serializes decompression with training; chunked scenes re-decompress
      their 4 non-last steps for every 36-patch chunk — papua alone did
      ~40 step-loads per epoch).
    - WHY WORKERS ARE SAFE AGAIN: the leak was never IPC per se — it was
      FRESH CPU PAGES used as H2D copy sources (driver pin bookkeeping).
      The v11.5.6 persistent pinned staging buffer intercepts every batch
      into FIXED pages before the H2D, so the copy source is the same page
      range regardless of where the batch came from. num_workers=2 is
      auto-restored on Kaggle (--workers overrides).
    - Scene cache budget 1800 -> 2400 MB (borneo/irrawaddy go resident
      again: ~4-5 decompressions per scene total); chunk tile budget
      1.2 -> 2 GB (papua: 10 -> 6 chunk reloads). Host RAM headroom is
      ~23 GB in the current flat profile.
    - EXPECTED: epoch time from ~2.4 h to ~35-45 min (GPU-bound); 100
      epochs ~50-65 h (~4-5 sessions), realistic val-plateau finish
      (epoch 30-60) in 2-3 sessions. Watch the anon telemetry — if it
      climbs with workers back, paste one heartbeat line.
  v11.5.8 (the leak's final identity: batch=2 under DataParallel) [KAGGLE]:
    - CROSS-SESSION DATA SETTLED IT: v11.5.7 leaked (+66 MB/batch) with the
      v11.5.6 staging code UNCHANGED — so the staging buffer was never the
      fix. Recomparing all runs: EVERY leaking configuration (~66-96 MB/
      batch) had batch=2 under DataParallel (1 sample/GPU); the ONLY flat
      run (yesterday, 10 h, RSS +-3 MB) was batch=1 after the OOM ladder.
      The leak lives in DP's two-replica path (scatter/gather/broadcast),
      not in copies, pinning, IPC, or the scan.
    - DEFAULT FIX: 16 GB cards now start directly in the proven-flat state
      — single GPU, batch=1, grad_accum scaled xN_GPUS (effective batch 4
      preserved) — instead of attempting the doomed batch=2 (which OOM'd
      epoch 1 twice) and falling into it via the ladder. >= 24 GB cards
      keep mirror with 2 samples/GPU. --mirror / --batch-size 2 remain for
      experimentation; the OOM ladder still protects any manual override.
    - Side effects: no more wasted batch=2 epoch attempt at startup, GPU 1
      released (no idle 2.8 GB replica), and the mamba scan runs at the
      batch=1 shapes that are known-good on T4.
  v11.5.7 (daily-session polish from the 10 h survival run) [KAGGLE EDITION]:
    - The v11.5.6 run survived a FULL ~10 h session (first ever): leak dead,
      mid-epoch checkpoints at steps 100/200/300, RSS flat at ~21-23 GB.
    - MIRROR EFFICIENCY: DataParallel with batch_size < N_GPUS is pure
      overhead — after the epoch-1 OOM halved batch to 1, GPU 1 sat at 0%
      the whole session while DP kept gathering. Mirror is now disabled
      automatically whenever batch_size < GPU count (both at startup and
      inside the OOM ladder's healed state).
    - VRAM DEFRAG per epoch: GPU 0's reserved memory crept to 13.4/15 GB
      (allocator caching + scan-workspace fragmentation) which is what
      OOM'd the batch=2 attempt; torch.cuda.empty_cache() now runs after
      every epoch's validation to restore headroom.
    - SCENE CACHE BUDGET 2560 -> 1800 MB: the 2.4-2.5 GB scenes now take
      the (proven) chunked path, trimming ~1.4 GB of live main-process RAM
      (the session sat at 27.6/30 GB cgroup-wide). BS11_SCENE_CACHE_MB
      still overrides.
    - RESUME HINT: when last_v11.pt exists but --resume wasn't passed,
      train() prints its epoch and the exact continue command — no more
      accidentally restarting from scratch after a session cut.
  v11.5.6 (persistent pinned staging — the fresh-pages theory confirmed)
  [KAGGLE EDITION]:
    - UNIFIED ROOT CAUSE: the ~71-96 MB/batch anonymous host growth tracked
      FRESH CPU PAGES used as H2D copy sources, not any Python object:
        * v11.5.0/2 (pin ON): the pin thread allocated a FRESH pinned block
          per batch -> driver pins fresh pages -> leak ~74 MB/batch.
        * v11.5.3/4 (pin OFF): the driver pinned the FRESH pageable source
          pages per cudaMemcpy -> leak ~71 MB/batch.
        * v11.5.5 (workers=0, loader inline in main): even more fresh CPU
          pages per iteration (aug copies + collate all in main) -> leak
          grew to ~96 MB/batch. gc/malloc_trim can NEVER reclaim it
          because the memory is held by the CUDA driver/runtime, not by
          Python — matching every heartbeat (gpu flat, shared flat,
          anon climbing linearly).
    - FIX: the standard high-performance-dataloader pattern — ONE
      persistent pinned staging buffer per key, allocated ONCE and
      overwritten in place (s.copy_(v)) every iteration. The driver sees
      the SAME page range forever: one-time pin, zero fresh ranges, zero
      growth. non_blocking H2D from the fixed pinned buffer is safe (the
      loop's float() sync guarantees the previous copy completed before
      the buffer is overwritten). Shape changes (val tail batch) fall
      back to a one-time reallocation.
    - CPU-only runs skip staging entirely; both train and val loops use it.
  v11.5.5 (IPC removed from the training path + leak attribution) [KAGGLE]:
    - v11.5.3's telemetry isolated the leak to the batch H2D transfer:
      ~68 MB/batch private growth with pin ON (+71.5) AND pin OFF (+67.6),
      with and without mamba — the only common path left was
      non_blocking=True, which routes through torch 2.11's CUDA
      CachingHostAllocator (staging buffers + completion events) and that
      allocator RETAINED every freed block on this setup.
    - Batch transfers are now synchronous (.to(DEVICE) without
      non_blocking): plain cudaMemcpy, no staging, no events, allocator
      bypassed. Costs ~15-25 ms per ~2 s batch (<1.5%); workers still
      overlap GPU compute.
    - Mid-epoch checkpoint interval 200 -> 100 optimizer steps: this
      dataset's epoch is only ~180 steps, so the 200-step interval never
      fired before the OOM point (the v11.5.3 run would have died without
      a single mid-epoch save).
  v11.5.3 (pin_memory leak killed + mid-epoch checkpoints) [KAGGLE EDITION]:
    - v11.5.2's teardown did NOT stop the leak (telemetry: still ~68 MB per
      micro-batch, PRIVATE heap, shared flat). Root cause: the DataLoader
      pin thread pinned a fresh ~74 MB copy of EVERY batch and the CUDA
      CachingHostAllocator never reused the freed blocks. pin_memory is now
      OFF: pageable H2D copies cost ~15 ms per ~2 s batch (<1%) and the
      leak source is gone entirely.
    - CRASH INSURANCE: a resumable last_v11.pt is now saved every 200
      optimizer steps mid-epoch (the 12 h session cap / an OOM previously
      lost the WHOLE epoch — the last two runs died at step ~180 with
      nothing saved). --resume picks up from the mid-epoch point; at most
      ~200 steps of work are lost.
    - Host heap is now trimmed (gc + malloc_trim) at EVERY heartbeat, not
      just past a threshold, so RSS returns to the OS continuously.
  v11.5.2 (host-RAM leak teardown — the ~72 MB/batch killer) [KAGGLE EDITION]:
    - DIAGNOSIS (heartbeat telemetry): the live run showed main-process RSS
      growing PERFECTLY LINEARLY at ~71.5 MB per micro-batch (5.5 -> 28.5 GB
      over 180 optimizer steps) — exactly the size of one received batch
      (x=65 MB + targets ~9 MB) — while GPU memory stayed flat at 0.8 GB.
      The full received batch was being retained in host RAM every
      iteration until the 30 GB cgroup wall killed the kernel mid-epoch.
    - FIX (per-iteration teardown, both run_epoch and run_epoch_accum):
      * the received batch is deleted the moment its GPU copy exists
        (releasing the shared-memory/pinned source tensors immediately
        instead of holding them until the next loop iteration),
      * every per-iteration graph object (out/tg/per_task/weighted/total)
        is explicitly deleted after the metrics block, and gc.collect()
        runs per iteration (the GPU keeps working during the collection —
        effectively free at ~2 s/batch),
      * glibc malloc_trim + gc after every epoch returns freed heap to the
        OS so RSS actually drops between epochs.
    - NEW telemetry: the heartbeat now reports the shared-vs-private RSS
      split (growing 'shared' = retained DataLoader IPC mappings; growing
      private = heap) and a watchdog trims at 22 GB — long before the
      30 GB wall — so the next run either stays flat or pinpoints the
      retainer precisely.
  v11.5.1 (mamba auto-installer: prebuilt-wheel first) [KAGGLE EDITION]:
    - The auto-installer now tries a PREBUILT wheel first: verified
      mamba_ssm-2.3.2.post1+cu.12.8.torch.<major.minor> on the Astral
      community index (wheels.astral.sh/simple/cu128) matches Kaggle's
      torch 2.11.0+cu128 / Python 3.13 exactly — ~1-2 min download instead
      of a 5-8 min nvcc build. Source build remains the stage-2 fallback.
    - causal-conv1d deliberately NOT installed: optional dep (Blue Sentinel
      only uses selective_scan_fn) and no torch-2.11 wheel exists on the
      index — installing it triggers a pointless 10-20 min source build.
    - Manual fast path (run BEFORE the model cell):
        !pip install -q "mamba-ssm==2.3.2.post1+cu.12.8.torch.2.11" \
            --extra-index-url https://wheels.astral.sh/simple/cu128/
  v11.5.0 (scan fallback rebuilt: the 'CPU 174% / GPU 4%' killer) [KAGGLE]:
    - DIAGNOSIS (from live Kaggle telemetry): sessions restarted after a
      kernel crash lose pip packages -> mamba_ssm silently disappears ->
      the pure-torch scan ran its NAIVE per-step Python loop: ~2.4M tiny
      CUDA kernel launches per VSS block, ~20-40 MIN per micro-batch at
      256px/T5 (CPU pegged in both DP replica threads, GPU at 4-5%, no
      heartbeat within 15 min). The first session's banner said
      mamba_ssm=True; later sessions didn't.
    - FIX 1: selective_scan_torch rebuilt as a chunked Hillis-Steele prefix
      scan. h_t = a_t*h_{t-1}+b_t is associative and a_t = exp(dt*A) with
      dt>0 (softplus) and A<0 lies in (0,1) by construction, so the scan
      needs only decaying products — no division, fp32-exact, no overflow.
      512 sequential steps per chunk become 9 vectorized combines (~60-100x
      fewer kernel launches, ~30-60x faster fallback). Forward AND backward
      gradients verified against the naive recurrence in the smoke test.
    - FIX 2: train/pretrain now auto-attempt
      `pip install --no-deps --no-build-isolation mamba-ssm` (8 min timeout)
      when the CUDA kernels are missing — --no-deps/--no-build-isolation so
      pip can never touch torch. Success flips the model to the CUDA scan
      for the whole session; failure falls back with exact manual
      instructions. BS11_NO_AUTO_MAMBA=1 skips.
    - FIX 3: loud warning banner with ETA + remedies when the fallback is
      active at patch>=256; heartbeat now every 10 optimizer steps and
      includes GPU memory (rss + gpu telemetry in one line).
  v11.4.9 (dual-mode loader: zero re-decompression + RAM telemetry) [KAGGLE]:
    - FIX CPU 164% / GPU 0% / RAM 29 GB: v11.4.6-8 re-decompressed the same
      step NPZs for EVERY chunk of a scene (5 scenes x 26 chunks x 4 files
      per epoch = ~500 zlib decompressions of the same data). Loader is now
      DUAL-MODE: scenes whose decompressed steps fit a per-worker budget
      (default 1.8 GB, BS11_SCENE_CACHE_MB) are loaded ONCE and patches
      sliced directly (the user's ~2 GB scenes: 5 loads/scene/epoch, GPU
      duty ~90%); giant AOIs stream in 1.2 GB chunks with a background
      thread prefetching the NEXT chunk while the GPU consumes the current
      (zlib releases the GIL, so decompression overlaps compute).
    - load_step_npz is now in-place (clip/log10/sanitize write into the
      freshly materialized npz arrays; astype->asarray): removes 2-3 full
      array copies per band -> ~half the transient RAM AND CPU per load.
    - TELEMETRY: per-scene loader line prints mode/scene-GB/worker-RSS;
      the training heartbeat now includes host RSS, so a crash log shows
      exactly where the 30 GB went.
  v11.4.8 (GPU duty-cycle fix + training heartbeat) [KAGGLE EDITION]:
    - FIX GPU idle gaps during training: the v11.4.6 streaming loader used a
      fixed 16-patch chunk, re-decompressing the same T step NPZs for every
      chunk (~10x redundant zlib work per scene -> GPU starved between
      chunks; sidebars showed 0.00% GPU during loads). Chunk size is now
      budget-based (~2 GB tile RAM per worker -> 48-96 patches per file
      load on 256px/T5 scenes = 4-6x fewer decompressions per patch) and
      the (T,n,...) x_chunk double buffer was removed (per-patch stacking
      from per-step tiles) -> half the transient RAM per worker.
    - DataLoader prefetch 2 -> 3 per worker to ride out scene-chunk loads.
    - NEW training heartbeat: every 25 optimizer steps prints step/loss so
      "loading data" is distinguishable from "hung" in the notebook.
  v11.4.7 (mirror mode restored — with the real bugs fixed) [KAGGLE EDITION]:
    - ROOT-CAUSE FIX for the v11.4.2 mirror OOM ("Tried to allocate 1.50 GiB"
      in the selective scan) and for why the 2nd T4 always died first:
      torch.autocast is THREAD-LOCAL and does NOT propagate into
      DataParallel's parallel_apply worker threads. The training loop's
      amp_context() reached replica 0 only — every other replica silently
      ran fp32, doubling activation VRAM exactly when the batch was split.
      The model now opens its own autocast INSIDE forward when mirror is
      active (the documented DataParallel+AMP pattern), so all replicas run
      the AMP dtype. fp16 on T4, bf16 on Ampere+.
    - DataParallel mirror mode re-enabled by default on >1 GPU (v11.4.4 had
      removed it outright after the v11.4.2/v11.4.3 crashes; the two actual
      culprits — thread-local autocast above, and the GPU-side DropPath RNG
      (fixed v11.4.4) / forced expandable_segments (removed v11.4.3) — are
      now addressed at the root instead).
    - OOM ladder under mirror: halve batch_size -> drop the DP wrapper ->
      halve grad_accum; each retry rebuilds loaders and zeroes grads.
      Illegal memory access aborts immediately with a --no-mirror hint
      (a poisoned CUDA context cannot be recovered by retrying).
    - VRAM-aware default batch: 1 sample/GPU on 16 GB cards (2x T4 ->
      batch 2), 2 samples/GPU on >=32 GB cards; --batch-size overrides.
    - load_checkpoint wraps the model in mirror for eval/export again;
      smoke test exercises the mirror path on multi-GPU machines so any
      DataParallel regression surfaces in the 2-5 min self-test.
    - NOTE for people upgrading: run the LATEST pasted file — the v11.4.5
      log that showed 100% CPU + 0% GPU + kernel OOM was the pre-streaming
      loader (v11.4.6 fixes that part).
  v11.4.6 (RAM / GPU-starvation hotfix) [KAGGLE EDITION]:
    - FIX 30 GB system-RAM OOM with GPU idle at 0%: patch_generator used to
      load EVERY scene whole (5x25xHxW float32 = 2.5-12+ GB per AOI, plus
      transient copies) with num_workers=0, so training alternated between
      long CPU-bound decompression stalls (GPU starved, CPU pegged) and
      RAM blow-ups. Folder scenes are now STREAMED: only the last step
      (labels/SCL) stays resident; every other step is loaded once per
      16-patch chunk, sliced, and freed immediately. Peak RAM drops from
      O(full temporal cube x copies) to ~1 step + a small tile buffer.
    - DataLoader workers now SHARD scenes (the old patch-modulo filter made
      every worker decompress every scene = N x work). num_workers=2 is
      auto-enabled on Kaggle (4 vCPUs) so decompression overlaps GPU
      compute; --workers overrides.
    - load_step_npz(want_targets=False) skips building ~1 GB of unused
      default-target arrays for non-label steps; one avoidable astype copy
      removed.
  v11.4.5 (T4 bf16 / FIND-engine hotfix) [KAGGLE EDITION]:
    - FIX crash ("FIND was unable to find an engine to execute this
      computation" in the depthwise conv): torch >= 2.7 reports bf16 as
      supported on Turing via new emulation flags, but cuDNN has NO bf16
      conv kernels for sm_75 (T4) -> autocast(bf16) convs died at cuDNN
      engine selection. bf16 AMP now requires compute capability >= 8.0
      (Ampere+); T4/P100 fall back to fp16 + GradScaler (as pre-2.7).
    - Depthwise conv now retries once on the native (cuDNN-disabled) path
      if an engine-lookup error ever recurs (state-dict keys unchanged).
    - FIX crash loading artifacts from v11.4.2/v11.4.3: Config.load /
      load_checkpoint ignore fields removed in newer versions (e.g.
      mirror_gpus) and strip legacy 'module.' state-dict prefixes.
  v11.4.4 (single-GPU + gradient accumulation release) [KAGGLE EDITION]:
    - REMOVED DataParallel (mirror mode) entirely: root cause of the
      "CUDA error: an illegal memory access was encountered" crashes on
      2x T4 (RNG state corruption in stochastic-depth/DropPath under
      DataParallel broadcast churn).
    - NEW grad accumulation: cfg.grad_accum (default 2, --grad-accum N) —
      effective batch = batch_size * grad_accum = 4 on T4, no extra VRAM,
      near-DataParallel throughput with none of its instability.
    - FIX: DropPath now draws its mask on CPU (RNG-state safety), VSS
      blocks are excluded from dynamo tracing, grad checkpointing falls
      back to use_reentrant=True if the non-reentrant path raises.
    - OOM handler now halves batch_size first, then grad_accum.
    - FIX: cfg=CFG default args evaluated at import time -> NameError;
      functions now default to None and resolve CFG at call time.
  v11.4.3 (illegal-access hotfix) [KAGGLE EDITION]:
    - FIX crash ("CUDA error: an illegal memory access was encountered",
      async-reported at a torch.rand allocation during training):
      v11.4.2's forced PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      corrupts the caching allocator under DataParallel's per-forward
      broadcast/free churn on some torch builds (v11.4.1 without it OOM'd
      cleanly). Reverted — set the env var yourself to opt in.
    - NEW --no-mirror: run on cuda:0 only even when several GPUs are
      visible (isolates DataParallel if this ever recurs). If it does,
      also try CUDA_LAUNCH_BLOCKING=1 for a truthful stack.
  v11.4.2 (T4 OOM hotfix) [KAGGLE EDITION]:
    - FIX crash (CUDA OOM on 2x T4, "Tried to allocate 1.50 GiB" in the
      selective scan): mirror mode blindly doubled the batch to 2/GPU, but
      2 samples of 256px x T5 patches exceed a 16 GB T4 during the chunked
      scan. The auto-batch is now VRAM-aware (1/GPU on <32 GB cards,
      2/GPU on bigger) and a training-epoch OOM auto-halves the batch,
      rebuilds the loaders and retries the epoch instead of dying.
      --batch-size still overrides everything.
    - main("train --batch-size 2") now works: a flag-bearing string used to
      arrive as ONE argv element (argparse "invalid choice"); strings are
      shlex-split now
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set (before torch
      boots its allocator) unless the user set it — the chunked scan's
      many shapes otherwise fragment reserved VRAM
  v11.4.1 (NaN-nodata hotfix) [KAGGLE EDITION]:
    - FIX crash ("'list' argument must have no negative elements" from
      np.bincount in compute_stats): GEE static exports carry NaN nodata
      in the label/change/recovery bands; _pack_scene's blind
      astype(int64) turned NaN into INT64_MIN (the "invalid value
      encountered in cast" RuntimeWarning during build). Int targets are
      now cast NaN-aware at pack time AND sanitized at load time
      (non-finite/negative/255+ -> 255 ignore), so NPZs already packed
      with garbage labels are repaired on the fly — no re-build needed
      (build re-runs skip existing scenes anyway).
    - FIX: carbon/eco/change_year nodata now packs as the -999 sentinel so
      the masked losses skip it (eco's 0.0 filler was trained on as if a
      real observation)
    - compute_stats/_patch_weight drop negative labels before bincount
      (was the next crash after the stats one)
  v11.4 (dataset-on-Kaggle release) [KAGGLE EDITION]:
    - NEW 'build' mode: packs RAW GEE Drive exports ({aoi}_{year}.tif +
      {aoi}_static.tif, from gee_dataset_builder_v11.py Part A — the
      'blue_sentinel_v11_raw' folder, however it was uploaded: bare TIFFs,
      *.zip, or extension-less Drive split archives) straight into v11 NPZ
      scenes on Kaggle: main("build") then main("train"). AOIs are
      auto-discovered; splits assigned alphabetically via --split
      (default 5,2,1); rasterio pip-installs automatically; re-runs skip
      already-packed scenes; labels land on the last step per the v11 schema
    - prepare: detects raw-GeoTIFF-only inputs and points to main("build");
      prints a TIFF/NPZ inventory when no scenes are found; extension-less
      ZIP archives are detected by magic bytes and extracted
    - clearer no-scenes assert message in train()
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
    - KAGGLE EDITION: *.zip dataset archives under /kaggle/input are
      auto-extracted to /kaggle/working/data and train/val/test dirs are
      auto-resolved before train/eval/pretrain (zips were previously
      invisible to the scene loader)
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

import os, re, gc, sys, json, glob, math, time, shutil, hashlib, argparse
import platform, warnings, base64, zipfile, shlex
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
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# FIX: Disable torch.compile/inductor for grouped convolutions (depthwise conv)
# "FIND was unable to find an engine" error on T4 with grouped conv
torch._dynamo.config.suppress_errors = True

warnings.filterwarnings("ignore", category=UserWarning)

VERSION = "11.6.2"
MIN_TORCH = (2, 1)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except Exception:
    selective_scan_fn = None
    HAS_MAMBA = False


def _maybe_install_mamba():
    """One-shot best-effort install of the mamba-ssm CUDA scan kernels when
    missing (the pure-torch fallback is ~50x slower). v11.5.1 strategy:
      1) PREFERRED: prebuilt wheel from the Astral community index matched
         to this session's torch major.minor (cu12.8 build, no compile,
         ~1-2 min for the 470 MB wheel — verified to exist for
         torch.2.11/cp313 on Kaggle).
      2) FALLBACK: source build from PyPI (--no-build-isolation so the
         system torch headers resolve; 8 min timeout).
    --no-deps always: pip must NEVER touch torch. causal-conv1d is NOT
    installed (optional dep; Blue Sentinel only uses selective_scan_fn and
    no torch-2.11 wheel exists for it). Skip with BS11_NO_AUTO_MAMBA=1."""
    global HAS_MAMBA, selective_scan_fn
    if HAS_MAMBA or DEVICE.type != "cuda" \
            or os.environ.get("BS11_NO_AUTO_MAMBA"):
        return
    import importlib
    import subprocess

    def _try_import():
        importlib.invalidate_caches()
        from mamba_ssm.ops.selective_scan_interface import \
            selective_scan_fn as _fn
        global selective_scan_fn, HAS_MAMBA
        selective_scan_fn = _fn
        HAS_MAMBA = True

    tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])  # e.g. 2.11
    idx = "https://wheels.astral.sh/simple/cu128/"
    wheel_spec = f"mamba-ssm==2.3.2.post1+cu.12.8.torch.{tv}"
    print(f"[mamba] CUDA scan kernels missing -> trying prebuilt wheel "
          f"({wheel_spec}, ~1-2 min)...", flush=True)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "--no-build-isolation", wheel_spec, "--extra-index-url", idx],
            capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            try:
                _try_import()
                print("[mamba] prebuilt wheel installed + imported OK — "
                      "CUDA scan kernels active (50-100x faster than the "
                      "fallback)", flush=True)
                return
            except Exception as e:
                print(f"[mamba] wheel installed but import failed: "
                      f"{str(e)[:160]}", flush=True)
        else:
            print(f"[mamba] prebuilt wheel unavailable for torch {tv}: "
                  f"{(r.stderr or r.stdout or '').strip()[-160:]}", flush=True)
    except subprocess.TimeoutExpired:
        print("[mamba] prebuilt-wheel install timed out (5 min)", flush=True)
    except Exception as e:
        print(f"[mamba] prebuilt-wheel attempt skipped: {e}", flush=True)
    print("[mamba] falling back to source build from PyPI "
          "(compiles ~5-8 min)...", flush=True)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "--no-build-isolation", "mamba-ssm"],
            capture_output=True, text=True, timeout=480)
        if r.returncode == 0:
            try:
                _try_import()
                print("[mamba] source build OK — CUDA scan kernels active",
                      flush=True)
                return
            except Exception as e:
                print(f"[mamba] built but import failed: "
                      f"{str(e)[:160]}", flush=True)
        else:
            print(f"[mamba] source build failed: "
                  f"{(r.stderr or r.stdout or '').strip()[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print("[mamba] source build timed out (8 min) — continuing with the "
              "vectorized torch fallback", flush=True)
    except Exception as e:
        print(f"[mamba] auto-install skipped: {e}", flush=True)

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
# FIX (v11.4.5, "FIND was unable to find an engine" crash on T4):
# torch >= 2.7 reports bf16 as *supported* on Turing via new emulation flags,
# but cuDNN ships NO bf16 conv kernels for sm_75 — so autocast(bf16) convs
# die at cuDNN engine selection. bf16 AMP now requires a real bf16 GPU
# (compute capability >= 8.0, Ampere+); T4/P100 correctly fall back to
# fp16 + GradScaler.
AMP_BF16 = False
if DEVICE.type == "cuda":
    _CC_MAJOR = torch.cuda.get_device_capability(0)[0]
    AMP_BF16 = _CC_MAJOR >= 8 and torch.cuda.is_bf16_supported()
    del _CC_MAJOR
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

N_GPUS = torch.cuda.device_count() if DEVICE.type == "cuda" else 0


def unwrap(model):
    """Underlying module (strips nn.DataParallel)."""
    return model.module if isinstance(model, nn.DataParallel) else model


# FIX (v11.4.7): DataParallel autocast is THREAD-LOCAL — the outer
# amp_context() in the train loop does NOT reach parallel_apply's worker
# threads, so replica 1..N silently ran fp32 (2x activation VRAM) while
# replica 0 ran fp16. That — not DataParallel itself — is what OOM'd the
# second T4 in v11.4.2. The flag below tells BlueSentinelV11.forward to
# open its own autocast INSIDE the module (the documented DP+AMP pattern),
# so every replica runs fp16 regardless of which thread executes it.
MIRROR_ACTIVE = False


def wrap_mirror(model, enabled=True):
    """Mirror strategy (Keras-speak) == nn.DataParallel: replicate the model
    on every CUDA GPU and split each batch across them. No-op on single-GPU
    / CPU boxes. Enables the in-forward autocast path (MIRROR_ACTIVE)."""
    global MIRROR_ACTIVE
    if enabled and DEVICE.type == "cuda" and N_GPUS > 1 \
            and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
        MIRROR_ACTIVE = True
        print(f"  [gpu] mirror mode ON: DataParallel over {N_GPUS}x "
              f"{torch.cuda.get_device_name(0)} "
              f"({N_GPUS * torch.cuda.get_device_properties(0).total_memory / (1 << 30):.0f} GB "
              f"combined VRAM; per-GPU batch = batch_size / {N_GPUS}; "
              f"in-forward autocast ON)")
    else:
        MIRROR_ACTIVE = False
    return model


print(f"Blue Sentinel v{VERSION} | torch {torch.__version__} | {DEVICE} | "
      f"bf16={AMP_BF16} | mamba_ssm={HAS_MAMBA}")
if DEVICE.type == "cuda":
    # Config class defined later; default grad_accum=2
    print(f"  GPU: {torch.cuda.get_device_name(0)} "
          f"(cc {'.'.join(map(str, torch.cuda.get_device_capability(0)))})"
          + (f" x{N_GPUS} (grad_accum=2)" if N_GPUS > 1 else ""))
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
    epochs: int = 12                      # v11.6.1: compressed curriculum for
    phase1_end: int = 3                   # the 5-scene dataset + 45 h/week
    phase2_end: int = 8                   # GPU budget (was 25/60/100)
    phase2_lambda: float = 0.1
    lr: float = 1e-4
    final_lr: float = 1e-6
    weight_decay: float = 5e-2
    batch_size: int = 2                   # Kaggle T4 16 GB safe
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_mamba: bool = True               # ckpt VSS even with CUDA kernels (VRAM)
    mirror_gpus: bool = True              # DataParallel over >1 CUDA GPU (2x T4);
                                          # per-GPU autocast + OOM-ladder fallback
    precip_clip: float = 6000.0           # mm; monsoon sites exceed 2000
    grad_accum: int = 2                   # gradient accumulation steps (effective batch = batch_size * grad_accum)
    max_scene_pixels: int = 8_000_000     # inference RAM guard (<=30 GB RAM)
    resume: bool = False                  # continue from last_v11.pt
    early_stop_patience: int = 5          # stop after N epochs without a new
    early_stop_min_epoch: int = 8         # best val mIoU (armed AFTER the EMA
                                          # warmup — val is EMA-based and lags)
    ema_decay: float = 0.99               # v11.6.1: warms in ~1 epoch at the
                                          # compressed step counts (0.999 needed
                                          # ~16 epochs — val stayed EMA-lagged)
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
        # FIX (v11.4.5): configs saved by older versions may carry removed
        # fields (e.g. mirror_gpus) — ignore unknown keys instead of crashing.
        known = set(Config.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
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
    # v11.4.9: in-place ops (npz members are freshly materialized + writable)
    # — avoids 2-3 full-copy temporaries per band on large AOIs (CPU + RAM).
    x = np.asarray(x, np.float32)
    if np.nanmin(x) >= 0:
        np.clip(x, 1e-6, None, out=x)
        np.log10(x, out=x)
        x *= 10.0
    np.clip(x, lo, hi, out=x)
    return x


def _process_s2(s2):
    s2 = np.asarray(s2, np.float32)
    mx = np.nanmax(s2) if s2.size else 0.0
    if mx > 20000: s2 *= 1.0 / 65535.0
    elif mx > 100: s2 *= 1e-4
    np.clip(s2, 0.0, 1.5, out=s2)
    return s2


def _sanitize(arr, name, path, max_bad=0.30):
    bad = ~np.isfinite(arr)
    if bad.any():
        frac = bad.sum() / arr.size
        if frac > max_bad:
            raise NPZSchemaError(f"{path}: '{name}' {frac:.0%} non-finite")
        arr[bad] = 0.0
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


def load_step_npz(path, cfg, want_targets=True):
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
        s1_std = (_sanitize(np.asarray(d["s1_std"], np.float32), "s1_std", path)
                  if "s1_std" in files else np.zeros((H, W, 2), np.float32))
        scl = _as_hw(d["scl"] if "scl" in files else None, H, W, fill=4.0)
        lband = (_sanitize(_to_db(d["lband"], -40.0, 5.0), "lband", path)
                 if "lband" in files else np.zeros((H, W, 2), np.float32))
        precip = _sanitize(_as_hw(d.get("precip"), H, W), "precip", path)
        np.clip(precip, 0.0, cfg.precip_clip, out=precip)
        inund = _sanitize(_as_hw(d.get("inundation"), H, W),
                          "inundation", path)
        np.clip(inund, 0.0, 20000.0, out=inund)
        aux = (np.asarray(d["aux"], np.float32) if "aux" in files
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
        # v11.4.6: want_targets=False skips this block entirely for
        # non-label steps (saves ~10 x H*W arrays, ~1 GB on big AOIs).
        t = _default_targets(H, W, cfg) if want_targets else None
        if want_targets:
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
                        axis=-1).astype(np.float32, copy=False)
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


def _rss_mb():
    """Resident set size of THIS process in MB (Linux; -1 elsewhere)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        return -1.0


def _rss_split_mb():
    """(resident, shared) RSS of THIS process in MB. A growing 'shared'
    means DataLoader IPC shm mappings are retained; growing private means
    heap — the split identifies the leak's source."""
    try:
        with open("/proc/self/statm") as f:
            p = [int(x) for x in f.read().split()]
        pg = os.sysconf("SC_PAGE_SIZE") / 1e6
        return p[1] * pg, p[2] * pg
    except Exception:
        return -1.0, -1.0


def _host_trim():
    """Best-effort return of freed heap to the OS: Python gc pass, then
    glibc malloc_trim. Helps when RSS is retained-but-unreferenced."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _mem_privacy_mb():
    """(Private_Dirty, Anonymous) from smaps_rollup in MB. v11.5.5: with
    statm shared flat but RSS growing, this attributes the leak precisely:
    growing Anonymous = heap/driver allocations; growing anything else =
    retained file/shm mappings."""
    try:
        pd = an = -1.0
        with open("/proc/self/smaps_rollup") as f:
            for ln in f:
                if ln.startswith("Private_Dirty:"):
                    pd = float(ln.split()[1]) / 1024.0
                elif ln.startswith("Anonymous:"):
                    an = float(ln.split()[1]) / 1024.0
        return pd, an
    except Exception:
        return -1.0, -1.0


def _patch_coords(label, scl, cfg, training, stride, ps, require_labels,
                  invalid):
    """Valid patch coordinates: label coverage + cloud filters + rare-class
    weight (shared by the full-scene and streaming folder paths)."""
    h, w = label.shape
    coords = []
    if h < ps or w < ps:
        return coords
    for i in range(0, h - ps + 1, stride):
        for j in range(0, w - ps + 1, stride):
            lp = label[i:i + ps, j:j + ps]
            # FIX (v11.1): pretrain/stats on unlabelled scenes must not
            # be starved by the label-coverage filter.
            if require_labels and \
                    float(np.mean(lp != cfg.ignore_label)) < cfg.min_valid_frac:
                continue
            sc = scl[i:i + ps, j:j + ps]
            if float(np.mean(np.isin(sc, invalid))) > cfg.max_cloud_frac:
                continue
            wgt = _patch_weight(lp, cfg) \
                if (training and require_labels) else 1.0
            if wgt > 0:
                coords.append((i, j, wgt))
    return coords


def _emit_full_scene_patches(scene, coords, rng, training, cfg):
    """Yield patches from an already-resident scene dict (cube schema)."""
    ps = cfg.patch_size
    for (i, j, _w) in coords:
        sl = np.s_[i:i + ps, j:j + ps]
        patch = {"x": np.ascontiguousarray(scene["x"][:, :, sl[0], sl[1]]),
                 "dates": scene["dates"]}
        for key in TARGET_KEYS_2D + ("eco",):
            t = scene["targets"][key]
            patch[key] = np.ascontiguousarray(
                t[sl] if t.ndim == 2 else t[sl[0], sl[1], :])
        if training:
            patch = _np_geometric_aug(patch, rng)
            patch = _np_photometric_aug(patch, rng, cfg)
        yield patch


def _iter_folder_patches(path, cfg, training, rng, ps, stride, quota,
                         require_labels, invalid, state, limit):
    """Streaming folder-schema patch source (v11.4.6 RAM fix).

    Per scene, only the LAST step (the one carrying labels) stays resident.
    Earlier steps are loaded one at a time per CHUNK of patches and freed
    right after their tiles are cut, so peak RAM is ~1 step array + a
    16-patch tile buffer instead of a full multi-GB temporal cube."""
    if limit <= 0 or state["n"] >= limit:
        return
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise NPZSchemaError(f"{path}: no step npz files")
    # ---- metadata pass (cheap: no big array decompressed) ----
    dated = []
    for p in files:
        try:
            with np.load(p) as d:
                if "s2" not in d.files or "s1" not in d.files:
                    raise NPZSchemaError("missing s2/s1")
                dt = float(np.asarray(d["date"]).reshape(-1)[0]) \
                    if "date" in d.files else _date_from_name(p)
        except Exception as e:
            print(f"  [data] bad step {os.path.basename(p)}: {str(e)[:100]}")
            continue
        dated.append((dt, p))
    if not dated:
        raise NPZSchemaError(f"{path}: no valid steps")
    dated.sort(key=lambda t: t[0])
    T = cfg.temporal_steps
    if len(dated) > T:
        idx = np.unique(np.linspace(0, len(dated) - 1, T).round().astype(int))
        chosen = [dated[int(i)] for i in idx]
    else:
        chosen = list(dated)
        while len(chosen) < T:
            chosen.append(chosen[-1])
    dates = np.array([d for d, _ in chosen], np.float32)
    last_path = chosen[-1][1]

    # ---- last step: full load (labels + SCL for filtering) ----
    last = load_step_npz(last_path, cfg)
    last_x = last["x"]                                   # (25, H, W)
    tgt = last["targets"]
    label = tgt["label"]
    H, W = label.shape
    if H < ps or W < ps:
        del last_x, last
        return
    coords = _patch_coords(label, last_x[CH_SCL], cfg, training, stride, ps,
                           require_labels, invalid)
    if training and len(coords) > quota:
        wts = np.array([c[2] for c in coords], np.float64)
        wts /= wts.sum()
        sel = rng.choice(len(coords), size=quota, replace=False, p=wts)
        coords = [coords[int(k)] for k in sel]
    if not coords:
        del last_x, last
        return
    if tgt["depth"] is None:                             # v9 pseudo-label fallback
        tgt["depth"] = last_x[CH_INUND].copy()

    # ---- emission: DUAL MODE (v11.4.9) ----
    # resident: scene's decompressed steps fit the per-worker budget (default
    # 1.8 GB, BS11_SCENE_CACHE_MB) -> every step is decompressed EXACTLY ONCE
    # per scene and patches are sliced directly (your ~2 GB scenes take this
    # path: 5 loads/scene/epoch instead of ~100, GPU duty ~90%).
    # chunked: giant AOIs stream in budget-sized chunks with a background
    # thread prefetching the NEXT chunk while the GPU consumes the current
    # one (zlib releases the GIL -> decompression overlaps compute).
    scene_bytes = T * N_CH * H * W * 4
    files_order = [f for _, f in chosen]
    sname = os.path.basename(path.rstrip("/")) or path
    budget_mb = 4000
    try:
        budget_mb = int(os.environ.get("BS11_SCENE_CACHE_MB", "4000"))
    except ValueError:
        pass
    budget = max(0, budget_mb) * 1e6

    if scene_bytes <= budget:
        steps = {}
        for _dt, f in chosen:
            if f in steps:
                continue
            steps[f] = last_x if f == last_path \
                else load_step_npz(f, cfg, want_targets=False)["x"]
        print(f"  [data] {sname}: {len(coords)} patches "
              f"(resident mode, scene {scene_bytes / 1e9:.1f} GB, "
              f"rss {_rss_mb():.0f} MB)")
        for (i, j, _w) in coords:
            if state["n"] >= limit:
                break
            patch = {"x": np.stack([steps[f][:, i:i + ps, j:j + ps]
                                    for f in files_order], 0),
                     "dates": dates}
            for key in TARGET_KEYS_2D + ("eco",):
                t = tgt[key]
                patch[key] = np.ascontiguousarray(
                    t[i:i + ps, j:j + ps] if t.ndim == 2
                    else t[i:i + ps, j:j + ps, :])
            if training:
                patch = _np_geometric_aug(patch, rng)
                patch = _np_photometric_aug(patch, rng, cfg)
            yield patch
        del steps
        return

    tile_bytes = max(1, T * N_CH * ps * ps * 4)
    CHUNK = int(max(8, min(96, 2_000_000_000 // tile_bytes)))
    CHUNK = min(CHUNK, max(1, len(coords)))
    print(f"  [data] {sname}: {len(coords)} patches (chunked mode, chunk="
          f"{CHUNK}, scene {scene_bytes / 1e9:.1f} GB > budget "
          f"{budget_mb} MB, rss {_rss_mb():.0f} MB)")

    def load_chunk_tiles(chunk):
        cache = {}
        for _dt, f in chosen:
            if f in cache:
                continue                       # padding repeats of the last step
            if f == last_path:
                cache[f] = np.stack(
                    [last_x[:, i:i + ps, j:j + ps] for (i, j, _) in chunk], 0)
            else:
                st = load_step_npz(f, cfg, want_targets=False)
                cache[f] = np.stack(
                    [np.ascontiguousarray(st["x"][:, i:i + ps, j:j + ps])
                     for (i, j, _) in chunk], 0)
                del st
        return cache

    chunks = [coords[s:s + CHUNK] for s in range(0, len(coords), CHUNK)]
    ex = ThreadPoolExecutor(max_workers=1)
    fut = None
    try:
        for ci, chunk in enumerate(chunks):
            if state["n"] >= limit:
                break
            cache = fut.result() if fut is not None else load_chunk_tiles(chunk)
            fut = None
            tt = {}
            for key in TARGET_KEYS_2D:
                arr = tgt[key]
                tt[key] = np.stack(
                    [(arr[i:i + ps, j:j + ps] if arr.ndim == 2
                      else arr[i:i + ps, j:j + ps, :]) for (i, j, _) in chunk],
                    0)
            tt["eco"] = np.stack([tgt["eco"][i:i + ps, j:j + ps, :]
                                  for (i, j, _) in chunk], 0)
            if ci + 1 < len(chunks):
                fut = ex.submit(load_chunk_tiles, chunks[ci + 1])
            for n_i in range(len(chunk)):
                if state["n"] >= limit:
                    break
                patch = {"x": np.stack([cache[f][n_i] for _, f in chosen], 0),
                         "dates": dates}
                for key in TARGET_KEYS_2D + ("eco",):
                    patch[key] = tt[key][n_i]
                if training:
                    patch = _np_geometric_aug(patch, rng)
                    patch = _np_photometric_aug(patch, rng, cfg)
                yield patch
            del cache, tt
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


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
            if desc[0] == "npz":
                scene = _load_cube_npz(desc[1], cfg)
                label = scene["targets"]["label"]
                coords = _patch_coords(label, scene["x"][-1, CH_SCL], cfg,
                                       training, stride, ps, require_labels,
                                       invalid)
                if training and len(coords) > quota:
                    wts = np.array([c[2] for c in coords], np.float64)
                    wts /= wts.sum()
                    sel = rng.choice(len(coords), size=quota, replace=False,
                                     p=wts)
                    coords = [coords[int(k)] for k in sel]
                for patch in _emit_full_scene_patches(scene, coords, rng,
                                                      training, cfg):
                    if emitted >= max_patches:
                        break
                    yield patch
                    emitted += 1
                del scene, coords
            else:
                # FIX (v11.4.6, RAM OOM): folder scenes are no longer loaded
                # whole (a 5x25xHxW float32 cube is 2.5-12+ GB). Only the
                # LAST step (labels/SCL) stays resident; every other step is
                # loaded once per CHUNK of patches, sliced, and freed
                # immediately. Peak RAM drops from O(scene) x copies to
                # ~1 step + a 16-patch tile buffer.
                state = {"n": 0}
                for patch in _iter_folder_patches(
                        desc[1], cfg, training, rng, ps, stride, quota,
                        require_labels, invalid, state, max_patches - emitted):
                    yield patch
                    emitted += 1
                    state["n"] += 1
                del state
            gc.collect()
        except Exception as e:
            failed += 1
            print(f"  [data] SKIP {desc[1]}: {str(e)[:140]}")
            if failed >= 25:
                raise RuntimeError("Too many scene failures; check data root.")
            continue
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
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        seed = self.cfg.seed + 1000 * self.epoch_seed + wid
        # FIX (v11.4.6): shard SCENES across workers instead of the old
        # patch-modulo filter, which made every worker load EVERY scene
        # (N x the decompression work and RAM). Each worker now walks a
        # disjoint slice of scenes, so data loading pipelines with the GPU
        # instead of competing with it.
        descs = self.scene_descs
        maxp = self.max_patches
        if nw > 1 and len(descs) > 1:
            descs = [d for k, d in enumerate(descs) if k % nw == wid]
            if descs:
                maxp = max(1, self.max_patches // nw)
        for patch in patch_generator(
                descs, self.cfg, self.training, maxp, seed=seed,
                require_labels=self.require_labels):
            yield {k: torch.from_numpy(np.asarray(v)) for k, v in patch.items()}


def make_loader(scene_descs, cfg, training, max_patches, require_labels=True):
    ds = PatchStream(scene_descs, cfg, training, max_patches,
                     require_labels=require_labels)
    nw = max(0, int(cfg.num_workers))
    kw = {}
    if nw > 0:
        kw["prefetch_factor"] = 3          # keep >=3 batches ready per worker
        kw["persistent_workers"] = False   # respawn per epoch: picks up set_epoch seed
    # FIX (v11.5.3): pin_memory DISABLED. The pin thread pinned a fresh
    # ~74 MB copy of every batch and the CachingHostAllocator never reused
    # the freed blocks — main-process RSS grew ~68 MB/batch, private,
    # linear to the 30 GB OOM (proven by the shared-flat/private-growing
    # heartbeat telemetry). Pageable H2D copies cost ~15 ms extra per
    # ~2 s batch (<1%): irrelevant here, and the leak source is gone.
    return DataLoader(ds, batch_size=cfg.batch_size,
                      num_workers=nw,
                      pin_memory=False,
                      drop_last=training, **kw), ds


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
def _hs_scan_chunk(a, b):
    """Inclusive Hillis-Steele prefix scan of h_t = a_t*h_{t-1} + b_t over the
    chunk axis (dim 1). The recurrence is associative, and a = exp(dt*A) with
    dt>0 (softplus) and A<0 lies in (0,1) by construction — so repeated
    products only DECAY: no division anywhere, fp32-safe, no overflow.
    log2(C) vectorized combine steps replace C sequential Python steps."""
    C = a.shape[1]
    d = 1
    while d < C:
        a_s = F.pad(a[:, :-d], (0, 0, 0, 0, d, 0))   # a[t-d] (zeros at front)
        b_s = F.pad(b[:, :-d], (0, 0, 0, 0, d, 0))
        b = torch.addcmul(b, a, b_s)                 # b_t = a_t*b_{t-d} + b_t
        a = a * a_s                                  # a_t = a_t*a_{t-d}
        d <<= 1
    return b


def _scan_chunk_vec(u_c, dt_c, B_c, C_c, h_prev, A_neg):
    """One chunk of the selective scan, fully vectorized (v11.5.0)."""
    a = torch.exp(dt_c.unsqueeze(-1) * A_neg)                  # (B,C,D,N) in (0,1)
    b = dt_c.unsqueeze(-1) * u_c.unsqueeze(-1) \
        * B_c.unsqueeze(2)                                     # (B,C,D,N)
    if h_prev is not None:                                     # carry from the
        b = torch.cat([b[:, :1] + a[:, :1] * h_prev.unsqueeze(1),   # previous
                       b[:, 1:]], 1)                           # chunk
    h = _hs_scan_chunk(a, b)
    y = torch.einsum("bcdn,bcn->bcd", h, C_c)
    return y, h[:, -1]


def selective_scan_torch(u, dt, A, B, C, D, chunk=512, use_ckpt=True):
    """Pure-PyTorch selective scan (fallback when mamba_ssm is absent).

    v11.5.0: the old per-step Python loop spent ~2.4M tiny kernel launches
    per VSS block (~20-40 min per micro-batch at 256px/T5 — the 'CPU 174% /
    GPU 4%' failure mode). Each 512-step chunk is now a Hillis-Steele prefix
    scan: ~60x fewer kernel launches, identical math (verified against the
    naive recurrence and its gradients in the smoke test)."""
    Bs, L, Dd = u.shape; N = A.shape[-1]
    u32, dt32, B32, C32 = u.float(), dt.float(), B.float(), C.float()
    A_neg = A.float()
    Dskip = D.float()
    h = u32.new_zeros(Bs, Dd, N)
    outs = []
    ckpt = use_ckpt and torch.is_grad_enabled()
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        args = (u32[:, s:e], dt32[:, s:e], B32[:, s:e], C32[:, s:e],
                h, A_neg)
        if ckpt:
            try:
                y_c, h = torch.utils.checkpoint.checkpoint(
                    _scan_chunk_vec, *args, use_reentrant=False)
            except RuntimeError:
                y_c, h = torch.utils.checkpoint.checkpoint(
                    _scan_chunk_vec, *args, use_reentrant=True)
        else:
            y_c, h = _scan_chunk_vec(*args)
        outs.append(y_c)
    y = torch.cat(outs, dim=1) + u32 * Dskip
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
        # FIX (illegal memory access under DataParallel):
        # torch.rand() on CUDA with DataParallel can cause illegal memory access
        # due to per-GPU RNG state corruption. Use CPU RNG then move to device.
        # Also guard against empty tensors.
        if x.numel() == 0:
            return x
        device = x.device
        # Generate mask on CPU to avoid DataParallel RNG issues, then move
        mask = (torch.rand(shape, device="cpu") < keep).to(device, dtype=x.dtype)
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

    def _dw(self, x):
        # FIX (v11.4.5): depthwise/grouped convs can hit cuDNN's
        # "FIND was unable to find an engine" on some driver/cuDNN combos.
        # Retry once on the native (non-cuDNN) path before giving up.
        # Keeps self.dw state-dict keys identical to older checkpoints.
        try:
            return self.dw(x)
        except RuntimeError as e:
            if "engine" in str(e).lower() or "cudnn" in str(e).lower():
                with torch.backends.cudnn.flags(enabled=False):
                    return self.dw(x)
            raise

    @torch._dynamo.disable
    def forward(self, x):                              # (B,T,C,H,W)
        B, T, C, H, W = x.shape
        di = self.d_inner
        x = x.permute(0, 1, 3, 4, 2)
        x = self.ln(x)
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = xi.permute(0, 1, 4, 2, 3).reshape(B * T, di, H, W)
        xi = F.silu(self._dw(xi)).view(B, T, di, H, W)
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

    @torch._dynamo.disable
    def _forward(self, x):
        return x + self.dp(self.ssm(x))

    @torch._dynamo.disable
    def forward(self, x):
        if self.use_ckpt and self.training and torch.is_grad_enabled() \
                and (self.ckpt_mamba or not (HAS_MAMBA and x.is_cuda)):
            try:
                return torch.utils.checkpoint.checkpoint(
                    self._forward, x, use_reentrant=False)
            except RuntimeError:
                # FIX: fallback to reentrant=True for DataParallel compatibility
                return torch.utils.checkpoint.checkpoint(
                    self._forward, x, use_reentrant=True)
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
        FiLM; defaults to integer spacing).

        FIX (v11.4.7): under DataParallel (mirror), parallel_apply runs each
        replica in its own thread, and torch.autocast is thread-local — so
        the training loop's amp_context() never reached replicas 1..N and
        they silently ran fp32 (2x activation VRAM, half speed). When
        MIRROR_ACTIVE, the module opens its own autocast here so EVERY
        replica runs the AMP dtype regardless of the calling thread."""
        if MIRROR_ACTIVE and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16 if AMP_BF16
                                else torch.float16):
                return self._forward_impl(x, dates)
        return self._forward_impl(x, dates)

    def _forward_impl(self, x, dates=None):
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

    # FIX (v11.5.4, host-RAM leak root cause): non_blocking=True H2D copies
    # route through the CUDA CachingHostAllocator (staging + completion
    # events) on torch 2.11/cu128, which RETAINED every freed block on this
    # setup: main-process RSS grew ~68 MB per micro-batch — exactly the
    # batch transfer size — under pin ON (v11.5.2, +71.5) AND pin OFF
    # (v11.5.3, +67.6), with and without mamba. Synchronous copies bypass
    # that allocator entirely (plain cudaMemcpy; the driver's internal
    # staging is reused per-device). Cost: ~15-25 ms per ~2 s batch (<1.5%),
    # and the pipeline still overlaps: the workers keep decoding while the
    # GPU trains on the previous batch.
    # v11.5.6: persistent pinned staging — same fixed pages every batch
    # (see the full FIX comment in run_epoch_accum). Handles the val
    # loader's smaller tail batch via the shape check.
    _stage = {}

    def _stage_batch(batch_cpu):
        if DEVICE.type != "cuda":
            return batch_cpu
        for k, v in batch_cpu.items():
            s = _stage.get(k)
            if s is None or s.shape != v.shape or s.dtype != v.dtype:
                _stage[k] = v.pin_memory()
            else:
                s.copy_(v)
        return _stage

    for batch_cpu in loader:
        batch = {k: v.to(DEVICE, non_blocking=True)
                 for k, v in _stage_batch(batch_cpu).items()}
        del batch_cpu         # v11.5.2: free received tensors NOW
        tg = build_targets(batch, cfg)
        try:
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

        except RuntimeError as e:
            if "illegal memory access" in str(e).lower() or "an illegal memory access" in str(e).lower():
                # Re-raise as AcceleratorError so the training loop catches it
                _IllegalAccess = getattr(torch, "AcceleratorError", RuntimeError)
                raise _IllegalAccess(str(e)) from e
            raise

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
        # v11.5.2: explicit per-iteration teardown — the live run leaked the
        # full ~72 MB received batch per micro-batch in host RAM (perfectly
        # linear to the OOM). Break every reference the moment the batch is
        # consumed, then collect.
        del out, tg, per_task, weighted, batch, total
        gc.collect()

    n = max(agg.pop("steps"), 1)
    res = {k: v / n for k, v in agg.items()}
    res.update({"miou": cm.miou(), "change_f1": f1.value(),
                "carbon_mae": mae_c.value(), "hydro_mae": mae_h.value(),
                "canopy_mae": mae_can.value(), "cm": cm.cm})
    return res


def run_epoch_accum(model, loader, cfg, phase, task_logvars=None, optimizer=None,
                    scaler=None, ema=None, alpha=None, grad_accum=1,
                    ckpt_cb=None):
    """Gradient accumulation version of run_epoch."""
    training = optimizer is not None
    model.train(training)
    active = set(cfg.active_heads) | {"quality", "boundary"}
    if cfg.deep_supervision and "seg" in active:
        active.add("seg_deep")
    tasks = [t for t in ALL_TASKS if t in active]
    if phase == 1:
        tasks = [t for t in tasks if t in PHASE1_TASKS]
    lam = 1.0 if phase == 3 else cfg.phase2_lambda
    fns = _loss_fns(cfg, alpha)
    agg = {"steps": 0}
    cm = ConfusionMatrix(cfg.num_classes)
    f1 = MaskedF1()
    mae_c = MaskedMAEMetric()
    mae_h = MaskedMAEMetric()
    mae_can = MaskedMAEMetric()

    # FIX (v11.5.6, THE leak root cause): the ~71-96 MB/batch anonymous host
    # growth tracked FRESH CPU PAGES used as H2D copy sources — every batch
    # was a brand-new allocation (worker-shm, collate output, aug copies),
    # and the CUDA driver/runtime leaks per fresh page range it pins for a
    # copy (explains pin ON: fresh pinned blocks; pin OFF: driver-pinned
    # pageable sources; workers=0: even more fresh loader pages -> higher
    # rate; and why gc/malloc_trim could never reclaim it — the memory is
    # driver-held, not Python-visible). The fix is the standard
    # high-performance-dataloader pattern: ONE persistent pinned staging
    # buffer per key, allocated once, overwritten in place every iteration.
    # The driver sees the SAME page range forever -> nothing new to leak.
    _stage = {}

    def _stage_batch(batch_cpu):
        if DEVICE.type != "cuda":
            return batch_cpu            # CPU-only: no H2D, no staging
        for k, v in batch_cpu.items():
            s = _stage.get(k)
            if s is None or s.shape != v.shape or s.dtype != v.dtype:
                # one-time alloc, or the rare shape change (val tail batch)
                _stage[k] = v.pin_memory()
            else:
                s.copy_(v)
        return _stage

    accum_step = 0
    opt_step = 0
    for batch_cpu in loader:
        batch = {k: v.to(DEVICE, non_blocking=True)          # pinned source:
                 for k, v in _stage_batch(batch_cpu).items()}  # true async, OK
        del batch_cpu         # v11.5.2: free received tensors NOW
        tg = build_targets(batch, cfg)
        try:
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
                    total = total / grad_accum

        except RuntimeError as e:
            if "illegal memory access" in str(e).lower() or "an illegal memory access" in str(e).lower():
                _IllegalAccess = getattr(torch, "AcceleratorError", RuntimeError)
                raise _IllegalAccess(str(e)) from e
            raise

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total).backward()
            else:
                total.backward()

            accum_step += 1
            if accum_step % grad_accum == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                # v11.4.8: liveness heartbeat; v11.5.0: every 10 steps with
                # GPU memory; v11.5.2: shared/private RSS split + a host-RAM
                # watchdog; v11.5.3: trim every heartbeat + mid-epoch
                # resumable checkpoints every 200 steps (crash insurance).
                opt_step += 1
                if opt_step % 10 == 0:
                    _gm = ""
                    if DEVICE.type == "cuda":
                        _gm = (f" | gpu={torch.cuda.memory_allocated() / 1e9:.1f}"
                               f"GB")
                    _r, _sh = _rss_split_mb()
                    _pd, _an = _mem_privacy_mb()
                    print(f"    [train] step {opt_step} "
                          f"(micro {accum_step}): loss="
                          f"{float(total) * grad_accum:.3f} | "
                          f"rss={_r:.0f} MB (shared {_sh:.0f}, "
                          f"anon {_an:.0f}){_gm}")
                    _host_trim()
                    if ckpt_cb is not None and opt_step % 100 == 0:
                        # v11.5.4: 200->100 — this dataset's epoch is only
                        # ~180 optimizer steps, so a 200-step interval never
                        # fired before the OOM point.
                        ckpt_cb(opt_step)

        agg["steps"] += 1
        for n, l in per_task.items():
            agg[n] = agg.get(n, 0.0) + float(l)
        agg["loss"] = agg.get("loss", 0.0) + float(total * grad_accum if training else total)
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
        # v11.5.2: explicit per-iteration teardown — the live run leaked the
        # full ~72 MB received batch per micro-batch in host RAM (perfectly
        # linear to the OOM). Break every reference the moment the batch is
        # consumed, then collect.
        del out, tg, per_task, weighted, batch, total
        gc.collect()

    # Handle any remaining gradients
    if training and accum_step % grad_accum != 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
    n = max(agg.pop("steps"), 1)
    res = {k: v / n for k, v in agg.items()}
    res.update({"miou": cm.miou(), "change_f1": f1.value(),
                "carbon_mae": mae_c.value(), "hydro_mae": mae_h.value(),
                "canopy_mae": mae_can.value(), "cm": cm.cm})
    return res


def train(cfg=None):
    cfg = cfg or CFG
    cfg.check()
    _maybe_install_mamba()
    print(f"\n=== BLUE SENTINEL v{VERSION} - TRAINING ===\n")
    if DEVICE.type == "cuda" and not HAS_MAMBA and cfg.patch_size >= 256:
        print("!" * 78)
        print("[ssm] WARNING: mamba_ssm kernels UNAVAILABLE — training uses the")
        print("       vectorized torch fallback (~60-100 s per micro-batch at")
        print("       256px/T5). If pip above failed, run in a cell:")
        print("         !pip install --no-deps --no-build-isolation mamba-ssm")
        print("       ...or use --patch-size 128 (~4x faster scans).")
        print("!" * 78 + "\n")
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    train_scenes = discover_scenes(cfg.train_dir)
    val_scenes = discover_scenes(cfg.val_dir)
    assert train_scenes and val_scenes, (
        f"no v11 scenes found (train={len(train_scenes)} from {cfg.train_dir},"
        f" val={len(val_scenes)} from {cfg.val_dir}). If your input holds raw "
        'GEE GeoTIFF exports, run main("build") first; else pass --train-dir '
        "/ --val-dir pointing at scene folders with step_YYYY.npz files.")
    validate_heads_vs_data(train_scenes, cfg)
    fp, nf = dataset_fingerprint(train_scenes)
    print(f"train={len(train_scenes)} val={len(val_scenes)} "
          f"T={cfg.temporal_steps} fingerprint={fp} ({nf} files)")

    mean, var, alpha = load_or_compute_stats(train_scenes, cfg)
    # v11.6.2: dataset overview — what the model sees (RGB / labels / overlay)
    # for up to 3 train scenes + 1 val scene; fault-isolated.
    try:
        plot_dataset_overview(train_scenes[:3] + val_scenes[:1], cfg)
    except Exception as e:
        print(f"  [viz] dataset overview failed (non-fatal): {e}")
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
    # v11.5.7: daily-session hint — an unresumed run next to a checkpoint
    # almost always means the user forgot --resume after a session cut.
    if not resumed and os.path.exists(ckpt_last):
        try:
            _prev_ep = torch.load(ckpt_last, map_location="cpu",
                                  weights_only=False).get("epoch", "?")
        except Exception:
            _prev_ep = "?"
        print(f"  [resume] HINT: {ckpt_last} exists (epoch {_prev_ep}). "
              f"Continue with main(['train', '--resume']); starting fresh "
              f"otherwise.")

    _OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    _IllegalAccess = getattr(torch, "AcceleratorError", RuntimeError)
    grad_accum = max(1, int(getattr(cfg, "grad_accum", 1)))
    _stall = 0            # v11.6.0: epochs since the last new best val mIoU
    for ep1 in range(start_ep + 1, cfg.epochs + 1):
        phase = phase_for_epoch(ep1, cfg)
        core.set_phase(phase)
        train_stream.set_epoch(ep1)
        t0 = time.time()

        def _mid_ckpt(opt_step, _ep=ep1):
            # v11.5.3: crash insurance — a resumable last_v11.pt every 200
            # optimizer steps, so an OOM/timeout/preemption loses at most
            # ~200 steps instead of the whole epoch (a 12 h Kaggle session
            # can die any time; --resume picks up from here).
            try:
                torch.save({"model": core.state_dict(), "ema": ema.shadow,
                            "norm_mean": mean, "norm_var": var,
                            "alpha": alpha, "cfg": asdict(cfg),
                            "epoch": _ep, "val_miou": best,
                            "version": VERSION,
                            "dataset_fingerprint": fp,
                            "seg_temperature": cfg.seg_temperature,
                            "change_conformal_q": cfg.change_conformal_q},
                           ckpt_last)
                print(f"    [ckpt] mid-epoch resume point saved "
                      f"(epoch {_ep}, step {opt_step})")
            except Exception as e:
                print(f"    [ckpt] mid-epoch save failed: {e}")

        # gradient accumulation: accumulate gradients over grad_accum steps
        # effective batch = batch_size * grad_accum
        optimizer.zero_grad(set_to_none=True)
        while True:
            try:
                tr = run_epoch_accum(model, train_loader, cfg, phase, task_logvars,
                                     optimizer, scaler, ema, alpha, grad_accum,
                                     ckpt_cb=_mid_ckpt)
                break
            except (_OOM, _IllegalAccess) as e:
                _msg = str(e).lower()
                # FIX (v11.4.7): an illegal memory access poisons the CUDA
                # context — NOTHING after it can run, so retrying (even with
                # a smaller batch) would just fail in a loop. Abort with an
                # actionable hint instead. (OOM, in contrast, is recoverable.)
                if "illegal memory access" in _msg:
                    if MIRROR_ACTIVE:
                        print("\n  [gpu] FATAL: CUDA illegal memory access "
                              "under DataParallel mirror mode.\n"
                              "        Re-run with --no-mirror (single GPU) "
                              "and report the log.\n"
                              "        NOTE: the CUDA context is corrupted; "
                              "the process cannot continue.\n")
                    else:
                        print("\n  [gpu] FATAL: CUDA illegal memory access "
                              "(single-GPU run).\n"
                              "        The CUDA context is corrupted.\n")
                    raise
                gc.collect(); torch.cuda.empty_cache()
                print(f"  [gpu] CUDA OOM during epoch {ep1} "
                      f"(mirror={'ON' if MIRROR_ACTIVE else 'off'}, "
                      f"batch={cfg.batch_size}, accum={grad_accum})")
                if cfg.batch_size > 1:
                    cfg.batch_size = max(1, cfg.batch_size // 2)
                    print(f"  [gpu] CUDA OOM -> retrying epoch {ep1} with "
                          f"batch_size={cfg.batch_size}")
                    # v11.5.7: a halved batch below the GPU count turns DP
                    # into overhead — shed the wrapper in the healed state.
                    if cfg.batch_size < N_GPUS and MIRROR_ACTIVE:
                        print("  [gpu] batch_size < GPU count -> dropping "
                              "mirror (single GPU)")
                        model = unwrap(model)
                        core = model
                        cfg.mirror_gpus = False
                        wrap_mirror(model, False)
                    train_loader, train_stream = make_loader(
                        train_scenes, cfg, True, cfg.max_train_patches)
                    val_loader, _ = make_loader(val_scenes, cfg, False,
                                                cfg.max_val_patches)
                    train_stream.set_epoch(ep1)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if MIRROR_ACTIVE:
                    # batch=1 already runs a single replica; drop the DP
                    # wrapper (broadcast overhead, scatter edge cases).
                    print("  [gpu] CUDA OOM -> dropping mirror mode "
                          "(single GPU), keeping batch_size=1")
                    model = unwrap(model)
                    core = model
                    cfg.mirror_gpus = False
                    wrap_mirror(model, False)      # resets MIRROR_ACTIVE
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if grad_accum > 1:
                    grad_accum = max(1, grad_accum // 2)
                    print(f"  [gpu] CUDA OOM -> retrying epoch {ep1} with "
                          f"grad_accum={grad_accum}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise
        scheduler.step()
        ema.store(model); ema.copy_to(model)
        va = run_epoch(model, val_loader, cfg, phase=3)
        ema.restore(model)
        _host_trim()          # v11.5.2: return freed heap to the OS per epoch
        torch.cuda.empty_cache()   # v11.5.7: release cached blocks (VRAM
        # fragmentation crept to 13.4/15 GB over yesterday's session and
        # triggered the batch=2 OOM; a per-epoch defrag keeps headroom).
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
        # v11.6.2: report visuals — regenerate the training curves from the
        # CSV after every epoch (one read + one PNG, fault-isolated).
        try:
            plot_training_curves(csv_path, WORK_DIR)
        except Exception:
            pass
        payload = {"model": core.state_dict(), "ema": ema.shadow,
                   "norm_mean": mean, "norm_var": var, "alpha": alpha,
                   "cfg": asdict(cfg), "epoch": ep1, "val_miou": va["miou"],
                   "version": VERSION, "dataset_fingerprint": fp,
                   "seg_temperature": cfg.seg_temperature,
                   "change_conformal_q": cfg.change_conformal_q}
        torch.save(payload, ckpt_last)
        _prev_best = best
        if va["miou"] >= best:
            best = va["miou"]
            torch.save(payload, ckpt_best)
            print(f"  [ckpt] new best val mIoU={best:.3f} -> {ckpt_best}")
        # v11.6.0: early stop — armed only after the EMA warmup (val mIoU
        # is EMA-based and lags ~10 epochs at decay 0.999; an unguarded
        # patience would fire during that plateau and stop prematurely).
        # Stall counts epochs without a STRICT new high (>= ties are not
        # progress); when it fires, the post-loop calibration still runs
        # and best_v11.pt is untouched.
        if ep1 >= cfg.early_stop_min_epoch:
            if va["miou"] > _prev_best:
                _stall = 0
            else:
                _stall += 1
                if _stall >= cfg.early_stop_patience:
                    print(f"\n  [early-stop] no new best val mIoU for "
                          f"{cfg.early_stop_patience} epochs (best {best:.4f}) "
                          f"— stopping at epoch {ep1}. best_v11.pt is intact.")
                    break

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
    # FIX (v11.4.5): ignore fields removed in newer versions (e.g.
    # mirror_gpus from pre-11.4.4 checkpoints) instead of crashing.
    known = set(Config.__dataclass_fields__)
    cfg_d = {k: v for k, v in cfg_d.items() if k in known}
    for k in Config._TUPLES:
        if k in cfg_d:
            cfg_d[k] = tuple(cfg_d[k])
    cfg = Config(**cfg_d)
    model = BlueSentinelV11(cfg, ckpt["norm_mean"], ckpt["norm_var"]).to(device)
    src = ckpt.get("ema", ckpt["model"])
    # FIX (v11.4.5): strip legacy 'module.' prefixes from checkpoints saved
    # under the removed DataParallel wrapper; tolerate missing keys too.
    if isinstance(src, dict):
        src = {(k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
               for k, v in src.items()}
    missing, unexpected = model.load_state_dict(src, strict=False)
    if missing or unexpected:
        print(f"  [ckpt] {len(missing)} missing / {len(unexpected)} unexpected "
              f"keys (left at init)")
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
# 17b. TRAINING GRAPHS + DATASET VISUALS (git/report-ready PNGs)
# ==============================================================================
def _mpl():
    """matplotlib pyplot in Agg mode, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_training_curves(csv_path=None, out_dir=None):
    """Render report-ready training curves from training_log.csv (v11.6.2).

    2x2 panels: (1) train/val loss on a log axis, (2) train/val mIoU with a
    star on the best val epoch, (3) learning rate, (4) aux heads
    (change-F1 + carbon/hydro/canopy MAEs). Curriculum phases are shaded
    green/amber/blue (P1/P2/P3). Regenerates after every epoch — cheap
    (one CSV read + one PNG). Fault-isolated; returns the PNG path or None.
    """
    plt = _mpl()
    if plt is None:
        return None
    csv_path = csv_path or os.path.join(WORK_DIR, "training_log.csv")
    if not os.path.exists(csv_path):
        return None
    rows = []
    try:
        import csv as _csv
        with open(csv_path) as f:
            for r in _csv.DictReader(f):
                try:
                    rows.append({k: float(v) for k, v in r.items()})
                except (TypeError, ValueError):
                    continue
    except Exception:
        return None
    if not rows:
        return None
    out_dir = out_dir or WORK_DIR
    os.makedirs(out_dir, exist_ok=True)

    ep = [r["epoch"] for r in rows]
    phases = [int(r["phase"]) for r in rows]
    bands, s = [], 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[s]:
            bands.append((ep[s], ep[i - 1], phases[s]))
            s = i
    PCOL = {1: "#2e7d32", 2: "#f9a825", 3: "#1565c0"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax in axes.flat:
        for (a, b, p) in bands:
            ax.axvspan(a - 0.5, b + 0.5, color=PCOL.get(p, "#9e9e9e"), alpha=0.10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

    ax = axes[0][0]
    ax.plot(ep, [r["loss"] for r in rows], "-o", ms=3, color="#c62828",
            label="train")
    ax.plot(ep, [r["val_loss"] for r in rows], "--s", ms=3, color="#6a1b9a",
            label="val")
    ax.set_yscale("log")
    ax.set_title("loss (log scale)")
    ax.legend()

    ax = axes[0][1]
    ax.plot(ep, [r["miou"] for r in rows], "-o", ms=3, color="#2e7d32",
            label="train mIoU")
    ax.plot(ep, [r["val_miou"] for r in rows], "--s", ms=3, color="#1565c0",
            label="val mIoU")
    bi = max(range(len(rows)), key=lambda i: rows[i]["val_miou"])
    ax.plot(ep[bi], rows[bi]["val_miou"], "*", ms=17, color="#f9a825",
            markeredgecolor="k", label=f"best {rows[bi]['val_miou']:.3f} (ep {int(ep[bi])})")
    ax.set_ylim(0, 1)
    ax.set_title("mIoU")
    ax.legend()

    ax = axes[1][0]
    ax.plot(ep, [r["lr"] for r in rows], "-o", ms=3, color="#455a64")
    ax.set_yscale("log")
    ax.set_title("learning rate")

    ax = axes[1][1]
    ax.plot(ep, [r["change_f1"] for r in rows], "-o", ms=3, color="#e65100",
            label="change F1")
    for k, c, ls in (("carbon_mae", "#6d4c41", "--"), ("hydro_mae", "#0277bd", "--"),
                     ("canopy_mae", "#33691e", "--")):
        ax.plot(ep, [r[k] for r in rows], ls, lw=1.2, color=c,
                label=k.replace("_", " "))
    ax.set_title("aux heads (change F1 + MAEs)")
    ax.legend(fontsize=8)

    for (a, b, p) in bands:
        axes[0][0].text((a + b) / 2, axes[0][0].get_ylim()[1], f"P{p}",
                        ha="center", va="bottom", fontsize=9, color=PCOL.get(p, "#555"),
                        fontweight="bold")

    fig.suptitle(f"Blue Sentinel v{VERSION} — training curves "
                 f"({len(rows)} epoch{'s' if len(rows) > 1 else ''})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, "training_curves.png")
    try:
        fig.savefig(out, dpi=130)
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    print(f"  [viz] training curves -> {out}")
    return out


def plot_dataset_overview(scene_descs, cfg, out_dir=None, max_scenes=4):
    """Render what the model actually sees (v11.6.2): per scene, the last
    step's Sentinel-2 true colour (B4,B3,B2 with a 2-98% stretch), the
    class-label map in CLASS_COLORS, and the label-over-RGB blend.
    Scenes are downscaled to <=512 px for display. Saves
    dataset_overview.png; fault-isolated; returns the path or None."""
    plt = _mpl()
    if plt is None:
        return None
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    rows = []
    for desc in list(scene_descs)[:max_scenes]:
        try:
            sc = load_scene(desc, cfg)
        except Exception as e:
            print(f"  [viz] skip {os.path.basename(str(desc[1]))}: {str(e)[:80]}")
            continue
        x = sc["x"][-1]                       # (25, H, W) reference step
        lab = sc["targets"]["label"]
        H, W = lab.shape
        st = max(1, int(math.ceil(max(H, W) / 512)))
        rgb = x[[2, 1, 0]][:, ::st, ::st].transpose(1, 2, 0)   # B4,B3,B2
        lo = np.percentile(rgb, 2, axis=(0, 1))
        hi = np.percentile(rgb, 98, axis=(0, 1))
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
        lab_s = np.asarray(lab)[::st, ::st]
        cmap = np.array([mcolors.to_rgb(c) for c in CLASS_COLORS], np.float32)
        lab_rgb = np.full((*lab_s.shape, 3), 0.88, np.float32)
        valid = (lab_s >= 0) & (lab_s < cfg.num_classes)
        lab_rgb[valid] = cmap[lab_s[valid].astype(int)]
        name = os.path.basename(str(desc[1]).rstrip("/")) or "scene"
        dt = float(sc["dates"][-1])
        rows.append((f"{name}  ({int(dt)})", rgb, lab_rgb))
    if not rows:
        return None
    out_dir = out_dir or WORK_DIR
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (title, rgb, lab_rgb) in enumerate(rows):
        axes[r][0].imshow(rgb)
        axes[r][0].set_title(f"{title} — S2 true colour", fontsize=10)
        axes[r][1].imshow(lab_rgb)
        axes[r][1].set_title("class labels", fontsize=10)
        axes[r][2].imshow(rgb)
        axes[r][2].imshow(lab_rgb, alpha=0.45)
        axes[r][2].set_title("overlay", fontsize=10)
        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(cfg.num_classes)]
    handles.append(mpatches.Patch(color="#e0e0e0", label="ignore/unlabelled"))
    fig.legend(handles=handles, loc="lower center",
               ncol=cfg.num_classes + 1, fontsize=9, frameon=False)
    fig.suptitle(f"Blue Sentinel v{VERSION} — dataset overview "
                 f"(inputs -> labels the heads learn)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    out = os.path.join(out_dir, "dataset_overview.png")
    try:
        fig.savefig(out, dpi=125)
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    print(f"  [viz] dataset overview -> {out}")
    return out


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


def pretrain(cfg=None):
    cfg = cfg or CFG
    _maybe_install_mamba()
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


def export_edge(cfg=None, ckpt=None, out_dir=None):
    cfg = cfg or CFG
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
    model = wrap_mirror(model, cfg.mirror_gpus)   # exercise the DP path in smoke
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
    opt = make_optimizer(core, tlv, cfg)
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
# 23b. KAGGLE DATA PREP — auto-extract ZIP datasets + auto-resolve split dirs
# ==============================================================================
# Kaggle dataset uploads often land as .zip archives (e.g. train.zip,
# val.zip, test.zip, one zip per AOI, or one big zip). The loader needs bare
# NPZ scene folders, so before train/eval/pretrain we:
#   1. extract every *.zip found under the Kaggle input mount into
#      <WORK_DIR>/data (idempotent: a size-keyed marker skips re-extraction),
#   2. auto-locate train/val/test split dirs wherever they landed (nested
#      inside zips is fine: <data>/mangrove-v11/train/... resolves too).
# Explicitly configured dirs that already exist on disk always win.

_KG_SPLIT_ALIASES = {
    "train_dir": ("train", "training"),
    "val_dir": ("val", "valid", "validation"),
    "test_dir": ("test", "testing"),
}


def _find_split_dir(root, names):
    """Shallowest directory named like the split that actually holds scenes
    (folder-of-step-NPZs or cube NPZs one level below)."""
    best, best_depth = None, 10 ** 9
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath).lower() in names \
                and discover_scenes(dirpath):
            depth = dirpath.count(os.sep)
            if depth < best_depth:
                best, best_depth = dirpath, depth
            dirnames[:] = []                     # don't descend past a hit
    return best


def prepare_kaggle_data(cfg, work_root=None, input_root=None):
    """Extract ZIP datasets and rewrite cfg.train_dir/val_dir/test_dir.

    Local/no-Kaggle runs are a no-op unless input_root is passed explicitly
    (used by tests). Returns the (possibly updated) cfg.
    """
    if input_root is None and not ON_KAGGLE:
        return cfg
    work_root = work_root or WORK_DIR
    if input_root is None:
        base = os.path.dirname(cfg.train_dir.rstrip("/"))
        input_root = base if os.path.isdir(base) else "/kaggle/input"
    data_root = os.path.join(work_root, "data")

    # ---- collect zips under the input mount ----
    zips = sorted({os.path.join(dp, f)
                   for dp, _, fs in os.walk(input_root)
                   for f in fs if f.lower().endswith(".zip")})
    # Drive split-downloads can land as extension-less archives (e.g.
    # blue_sentinel_v11_raw-...-001 with the .zip stripped by the upload
    # flow); detect real ZIPs by magic bytes so they get extracted too.
    _skip_ext = (".zip", ".npz", ".tif", ".tiff", ".json", ".py", ".pt",
                 ".csv", ".txt")
    for dp, _, fs in os.walk(input_root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in _skip_ext:
                continue
            p = os.path.join(dp, f)
            try:
                with open(p, "rb") as fh:
                    if fh.read(4) == b"PK\x03\x04":
                        zips.append(p)
            except OSError:
                pass
    zips = sorted(set(zips))
    marker = os.path.join(work_root, ".bs11_extracted_zips")
    done = set()
    if os.path.exists(marker):
        with open(marker) as fh:
            done = {ln.strip() for ln in fh if ln.strip()}
    todo, total = [], 0
    for z in zips:
        try:
            key = f"{z}:{os.path.getsize(z)}"
        except OSError:
            continue
        if key not in done:
            todo.append((z, key)); total += os.path.getsize(z)
    if todo:
        os.makedirs(data_root, exist_ok=True)
        print(f"[data] extracting {len(todo)} ZIP archive(s), "
              f"{total / 1e9:.2f} GB -> {data_root}")
        if total > 10e9:
            print("  [data] NOTE: /kaggle/working is capped at ~19.5 GB; "
                  "delete other outputs if extraction runs out of space.")
        for z, key in todo:
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(data_root)     # zipfile sanitizes members
                with open(marker, "a") as fh:
                    fh.write(key + "\n")
                print(f"  [data] extracted {os.path.basename(z)}")
            except Exception as e:
                print(f"  [data] FAILED {os.path.basename(z)}: "
                      f"{str(e)[:120]}")
    elif zips:
        print(f"[data] {len(zips)} ZIP archive(s) already extracted "
              f"(marker: {marker}); delete it to force re-extraction.")

    # ---- resolve split dirs (existing/explicit config wins) ----
    search_roots = [data_root, input_root]
    for attr, names in _KG_SPLIT_ALIASES.items():
        cur = getattr(cfg, attr)
        if os.path.isdir(cur) and discover_scenes(cur):
            continue
        hit = None
        for r in search_roots:
            if os.path.isdir(r):
                hit = _find_split_dir(r, names)
                if hit:
                    break
        if hit:
            setattr(cfg, attr, hit.rstrip("/") + "/")
            print(f"  [data] {attr} -> {getattr(cfg, attr)}")
    if not (os.path.isdir(cfg.train_dir) and discover_scenes(cfg.train_dir)):
        print("  [data] WARNING: no 'train' split found under "
              f"'{input_root}' or '{data_root}'.\n"
              "         Expected zip layout: train/scene_x/step_YYYY.npz "
              "(zipped as train.zip etc.).\n"
              "         If your zips hold scenes directly, re-zip per split "
              "or pass --train-dir / --val-dir / --test-dir explicitly.")
        n_tif = n_npz = 0
        for _root in (input_root, data_root):
            for dp, _, fs in os.walk(_root):
                for f in fs:
                    fl = f.lower()
                    if fl.endswith((".tif", ".tiff")):
                        n_tif += 1
                    elif fl.endswith(".npz"):
                        n_npz += 1
        print(f"  [data] inventory under '{input_root}': {n_tif} GeoTIFF(s), "
              f"{n_npz} NPZ(s)")
        if n_tif and not n_npz:
            print('  [data] raw GEE exports detected ({aoi}_{year}.tif + '
                  '{aoi}_static.tif) -> run main("build") to pack them '
                  'into v11 NPZ scenes first.')
    return cfg


# ==============================================================================
# 23c. KAGGLE RAW-EXPORT BUILDER — GEE GeoTIFFs -> v11 NPZ scenes (Part B)
# ==============================================================================
# gee_dataset_builder_v11.py Part A exports, per AOI, to the Drive folder
# 'blue_sentinel_v11_raw': {aoi}_{year}.tif (19-band STEP stack) and
# {aoi}_static.tif (14-band STATIC stack). Downloading that folder from Drive
# yields split archives like blue_sentinel_v11_raw-...-001..N; uploading them
# to Kaggle lands the RAW TIFFs here — NOT the NPZ scenes the model needs.
# main("build") runs Part B directly on Kaggle: it discovers every
# {aoi}_{year}.tif + {aoi}_static.tif under /kaggle/input (and any extracted
# ZIP parts), packs them into train/val/test NPZ scene folders under
# /kaggle/working/data, and main("train") picks them up from there.

BUILD_STEP_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11",
                    "B12", "SCL", "VV", "VH", "VV_std", "VH_std", "HH", "HV",
                    "precip", "inund")
BUILD_STATIC_BANDS = ("elev", "slope", "ndvi_trend", "flood_freq", "lst_trend",
                     "s1_trend", "label", "change", "recovery", "change_year",
                     "gedi", "carbon", "eco_lst", "eco_ndmi")


def _ensure_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        import subprocess
        print("[build] rasterio missing -> pip install rasterio ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "rasterio"])
        if r.returncode != 0:
            raise RuntimeError("pip install rasterio failed — build mode "
                               "needs it to read the GeoTIFF exports")
        import rasterio
        print(f"[build] rasterio {rasterio.__version__} installed")
        return rasterio


def find_raw_exports(root):
    """{aoi: {year: tif}} + {aoi: static_tif} for all GEE exports under root."""
    step_rx = re.compile(r"^(?P<aoi>.+)_(?P<year>\d{4})\.ti[fF]$")
    static_rx = re.compile(r"^(?P<aoi>.+)_static\.ti[fF]$")
    raw, statics = {}, {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            m = step_rx.match(f)
            if m:
                raw.setdefault(m.group("aoi"), {})[int(m.group("year"))] = \
                    os.path.join(dp, f)
                continue
            m = static_rx.match(f)
            if m:
                statics[m.group("aoi")] = os.path.join(dp, f)
    return raw, statics


def _pack_scene(aoi, year_tifs, static_path, out_dir):
    """{aoi} raw TIFFs -> out_dir/step_<year>.npz (labels on the last step)."""
    import numpy as _np
    rasterio = _ensure_rasterio()
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(static_path) as s:
        if s.count != len(BUILD_STATIC_BANDS):
            raise NPZSchemaError(
                f"{static_path}: {s.count} bands, expected "
                f"{len(BUILD_STATIC_BANDS)} STATIC bands")
        st = {b: s.read(i + 1) for i, b in enumerate(BUILD_STATIC_BANDS)}
        shw = (s.height, s.width)
    aux = _np.stack([st[b] for b in BUILD_STATIC_BANDS[:6]], -1)
    aux = _np.nan_to_num(aux, nan=0.0).astype(_np.float32)
    gedi = st["gedi"].astype(_np.float32)
    gedi = _np.where(gedi > 0, gedi, _np.nan).astype(_np.float32)
    # FIX (v11.4.1): nodata -> -999 sentinel so the masked losses skip it
    # (eco's 0.0 filler used to be treated as a VALID observation).
    carbon = _np.nan_to_num(st["carbon"], nan=-999.0, posinf=-999.0,
                            neginf=-999.0).astype(_np.float32)
    eco = _np.stack([_np.nan_to_num(st["eco_lst"], nan=-999.0),
                     _np.nan_to_num(st["eco_ndmi"], nan=-999.0)], -1) \
        .astype(_np.float32)
    years = sorted(year_tifs)
    for k, y in enumerate(years):
        out_p = os.path.join(out_dir, f"step_{y}.npz")
        if os.path.exists(out_p):
            continue
        with rasterio.open(year_tifs[y]) as d:
            if d.count != len(BUILD_STEP_BANDS):
                raise NPZSchemaError(
                    f"{year_tifs[y]}: {d.count} bands, expected "
                    f"{len(BUILD_STEP_BANDS)} STEP bands")
            if (d.height, d.width) != shw:
                raise NPZSchemaError(
                    f"{year_tifs[y]}: {d.height}x{d.width} vs static "
                    f"{shw[0]}x{shw[1]} — re-export this AOI")
            stp = {b: d.read(i + 1) for i, b in enumerate(BUILD_STEP_BANDS)}
        s2 = _np.stack([stp[b] for b in BUILD_STEP_BANDS[:10]], -1) \
            .astype(_np.float32)
        scl = _np.nan_to_num(stp["SCL"], nan=0.0).astype(_np.float32)
        s1 = _np.stack([_np.nan_to_num(stp["VV"], nan=-30.0),
                       _np.nan_to_num(stp["VH"], nan=-30.0)], -1) \
            .astype(_np.float32)
        s1_std = _np.stack([_np.nan_to_num(stp["VV_std"], nan=0.0),
                            _np.nan_to_num(stp["VH_std"], nan=0.0)], -1) \
            .astype(_np.float32)
        lband = _np.stack([_np.nan_to_num(stp["HH"], nan=-35.0),
                           _np.nan_to_num(stp["HV"], nan=-35.0)], -1) \
            .astype(_np.float32)
        precip = _np.nan_to_num(stp["precip"], nan=0.0).astype(_np.float32)
        inund = _np.nan_to_num(stp["inund"], nan=0.0).astype(_np.float32)
        keys = dict(s2=s2, scl=scl, s1=s1, s1_std=s1_std, lband=lband,
                    precip=precip, inundation=inund, aux=aux,
                    date=_np.float32(float(y)))
        if k == len(years) - 1:      # v11: labels ONLY on the last step
            # FIX (v11.4.1): GEE writes NaN nodata into these bands; a blind
            # astype(int64) turned NaN into INT64_MIN ("invalid value
            # encountered in cast") which crashed bincount in compute_stats.
            # Non-finite / negative -> 255 (ignore) before casting.
            def _safe_int(b):
                b = _np.asarray(b, _np.float64)
                return _np.where(_np.isfinite(b) & (b >= 0), b, 255.0) \
                    .astype(_np.int64)
            keys.update(label=_safe_int(st["label"]),
                        change=_safe_int(st["change"]),
                        recovery=_safe_int(st["recovery"]),
                        change_year=_np.nan_to_num(
                            st["change_year"], nan=-999.0, posinf=-999.0,
                            neginf=-999.0).astype(_np.float32),
                        gedi=gedi, carbon=carbon, eco=eco)
        _np.savez_compressed(out_p, **keys)
        print(f"  [build] packed {aoi}/step_{y}.npz "
              f"({shw[0]}x{shw[1]})")


def build_dataset(cfg, input_root=None, out_root=None, split="5,2,1"):
    """Raw GEE GeoTIFF exports -> v11 NPZ scenes, split train/val/test.

    AOIs are auto-discovered under input_root; 'split' gives the number of
    AOIs per split in alphabetical order (train,val,test). Re-runs skip
    scenes that are already packed.
    """
    input_root = input_root or "/kaggle/input"
    out_root = out_root or os.path.join(WORK_DIR, "data")
    raw, statics = find_raw_exports(input_root)
    # zips extracted by prepare_kaggle_data land in WORK_DIR/data; a custom
    # out_root is scanned too (covers redirected work roots)
    scan_roots = {os.path.realpath(os.path.join(WORK_DIR, "data")),
                  os.path.realpath(out_root)}
    for root in scan_roots:
        if not os.path.isdir(root) or root == os.path.realpath(input_root):
            continue
        r2, s2 = find_raw_exports(root)      # zips extracted earlier
        for a, ys in r2.items():
            raw.setdefault(a, {}).update(ys)
        statics.update({k: v for k, v in s2.items() if k not in statics})
    if not raw:
        scanned = "', '".join(sorted({input_root,
                                       os.path.join(WORK_DIR, "data"),
                                       out_root}))
        print(f"[build] no raw exports (*_YYYY.tif / *_static.tif) found "
              f"under '{scanned}'.\n"
              "       Upload the gee_dataset_builder_v11.py Part A output "
              "(Drive folder 'blue_sentinel_v11_raw'), or point --raw-dir "
              "at it.")
        return None
    try:
        counts = [max(0, int(v)) for v in str(split).split(",")]
        assert len(counts) == 3
    except Exception:
        print('[build] --split expects "train,val,test" counts, e.g. "5,2,1"')
        return None
    aois = sorted(a for a in raw if a in statics)
    missing_static = sorted(set(raw) - set(statics))
    if missing_static:
        yr_info = {a: len(raw[a]) for a in missing_static}
        print(f"  [build] WARNING: no _static.tif for {missing_static} "
              f"-> skipped (yearly steps found: {yr_info})")
    if not aois:
        print("[build] no complete AOI (needs both yearly and static TIFFs)")
        return None
    n_tr, n_va, n_te = counts
    train_a = aois[:n_tr]
    val_a = aois[n_tr:n_tr + n_va]
    test_a = aois[n_tr + n_va:n_tr + n_va + n_te]
    extra = aois[n_tr + n_va + n_te:]
    if extra:
        print(f"  [build] {len(extra)} AOI(s) beyond the split counts -> "
              f"train: {extra}")
        train_a += extra
    if not val_a and len(train_a) >= 2:
        val_a = [train_a.pop()]
        print("  [build] val split was empty -> moved one train AOI to val")
    if not test_a:
        print("  [build] NOTE: no test AOIs (eval mode needs some) — "
              "adjust --split if you want a held-out test region")
    print(f"[build] packing {len(aois)} AOI(s) "
          f"(train={train_a} val={val_a} test={test_a}) -> {out_root}")
    for split_name, group in (("train", train_a), ("val", val_a),
                              ("test", test_a)):
        for a in group:
            years = sorted(raw[a])
            out_dir = os.path.join(out_root, split_name, a)
            if all(os.path.exists(os.path.join(out_dir, f"step_{y}.npz"))
                   for y in years):
                print(f"  [build] {split_name}/{a}: already packed, skipping")
                continue
            _pack_scene(a, raw[a], statics[a], out_dir)
    print(f"[build] done. Scenes under {out_root}. Next: main(\"train\")")
    return out_root


# ==============================================================================
# 24. CLI  (notebook-safe: main("train") / main(["train","--resume"]) / main())
# ==============================================================================
def main(argv=None):
    """Kaggle/notebook-safe entry point.

    argv : str | list[str] | None
        "train"            -> ["train"]
        "train --batch-size 2" -> ["train", "--batch-size", "2"]
        ["train", "--resume"]
        None               -> sys.argv (script mode)
    Unknown args are ignored (the notebook kernel's '-f <kernel.json>' can't
    break it).
    """
    if isinstance(argv, str):
        # FIX (v11.4.2): a string with flags ("train --batch-size 2") used
        # to arrive as ONE argv element -> argparse "invalid choice".
        argv = shlex.split(argv)
    if argv is None:
        # In a notebook sys.argv is kernel noise ("-f /path/kernel.json"),
        # not user intent — don't let it become the 'mode' positional.
        if _in_interactive():
            print('main() needs an explicit mode in a notebook, e.g. '
                  'main("smoke") or main("train")')
            return
        argv = sys.argv[1:]
    ap = argparse.ArgumentParser(description=f"Blue Sentinel v{VERSION}")
    ap.add_argument("mode", choices=["smoke", "build", "pretrain", "train",
                                     "eval", "export", "export-edge",
                                     "smooth"],
                    nargs="?", default="smoke")
    ap.add_argument("scene", nargs="?",
                    help="scene folder/cube NPZ (export) / prob dir (smooth)")
    ap.add_argument("ref", nargs="?", help="reference GeoTIFF (export)")
    ap.add_argument("out", nargs="?", help="output dir (export / build)")
    ap.add_argument("--config")
    ap.add_argument("--ckpt", default=os.path.join(WORK_DIR, "best_v11.pt"))
    ap.add_argument("--pretrain", dest="pretrain_w", default="")
    ap.add_argument("--train-dir"); ap.add_argument("--val-dir")
    ap.add_argument("--test-dir")
    ap.add_argument("--raw-dir", help="input root with raw GEE exports "
                    "(build mode; default: /kaggle/input)")
    ap.add_argument("--split", default="5,2,1",
                    help="build mode: AOIs per split, train,val,test "
                    "(alphabetical order)")
    ap.add_argument("--batch-size", type=int); ap.add_argument("--epochs", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--resume", action="store_true",
                    help="resume training from last_v11.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--mirror", action="store_true",
                    help="force DataParallel mirror mode across GPUs "
                         "(default: ON when >1 GPU is visible)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="run on a single GPU even when several are visible "
                         "(isolates DataParallel issues)")
    ap.add_argument("--grad-accum", type=int, help="gradient accumulation steps (effective batch = batch_size * grad_accum)")
    ap.add_argument("--sar-only", action="store_true")
    ap.add_argument("--optical-only", action="store_true")
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"[cli] ignoring unknown args: {unknown}")

    if args.mode == "smoke":
        smoke_test()
        return

    cfg = Config.load(args.config) if args.config else Config()
    for a, k in (("train_dir", "train_dir"), ("val_dir", "val_dir"),
                 ("test_dir", "test_dir"), ("batch_size", "batch_size"),
                 ("epochs", "epochs"), ("workers", "num_workers"),
                 ("grad_accum", "grad_accum")):
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
    elif args.mirror:
        cfg.mirror_gpus = True
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    # v11.5.9: workers are BACK. They were disabled in v11.5.5 when the
    # leak suspect was the worker->main IPC path — but the v11.5.6 pinned
    # staging buffer intercepts every batch into FIXED pages before any
    # H2D copy, so fresh-page pinning (the actual leak) can no longer
    # happen regardless of where batches come from. With workers=0 the
    # inline loader dominated wall time (~26 s/micro-batch: chunked scenes
    # re-decompress per chunk on the critical path) — 2 loader workers
    # overlap that with GPU compute again. --workers overrides.
    if ON_KAGGLE and args.workers is None and cfg.num_workers == 0 \
            and (multiprocessing.cpu_count() or 1) >= 4:
        cfg.num_workers = 2
        print(f"  [data] num_workers=2 (auto on Kaggle; safe with the "
              f"v11.5.6 pinned staging; --workers to change)")

    # Kaggle datasets uploaded as ZIP archives are extracted and the
    # train/val/test dirs auto-located before build/training/eval/pretrain.
    if args.mode in ("build", "pretrain", "train", "eval"):
        cfg = prepare_kaggle_data(cfg)

    # FIX (v11.4.7, restored from v11.4.2 with per-GPU autocast fixed):
    # DataParallel (mirror) engages when >1 CUDA GPU is visible. The default
    # batch is VRAM-aware: 1 sample/GPU on 16 GB cards (2x T4 -> batch 2),
    # 2 samples/GPU on >=32 GB cards. --batch-size overrides. An epoch OOM
    # auto-halves the batch, then drops mirror, then halves grad_accum.
    if DEVICE.type == "cuda" and cfg.mirror_gpus and N_GPUS > 1:
        if args.batch_size is None:
            _total_gb = torch.cuda.get_device_properties(0).total_memory \
                / (1 << 30)
            if _total_gb >= 24:
                cfg.batch_size = 2 * N_GPUS     # 2 samples/GPU fits comfortably
            else:
                # v11.6.1 (agreed plan): 16 GB cards run SINGLE-GPU batch=2
                # (accum 2 -> effective batch 4, unchanged science). The
                # scan kernel has spare lane capacity, so 2 samples cost
                # ~the same kernel time as 1 -> ~1.8x faster epochs than
                # batch=1. DataParallel stays OFF: batch=2 through DP's
                # two-replica path is the confirmed host-RAM leak config,
                # and 2 samples/GPU under DP OOM'd twice. The OOM ladder
                # (batch -> 1, accum halves) remains the safety net.
                cfg.batch_size = 2
                cfg.grad_accum = 2
                cfg.mirror_gpus = False
        if cfg.batch_size < N_GPUS:
            # v11.5.7/8: DP with batch < GPU count is pure overhead.
            cfg.mirror_gpus = False

    if DEVICE.type == "cuda" and cfg.mirror_gpus and N_GPUS > 1 \
            and cfg.batch_size >= N_GPUS:
        print(f"[gpu] mirror: {N_GPUS}x {torch.cuda.get_device_name(0)} -> "
              f"batch_size={cfg.batch_size} "
              f"({cfg.batch_size // N_GPUS}/GPU); grad_accum={cfg.grad_accum} "
              f"(effective batch={cfg.batch_size * cfg.grad_accum}); "
              f"--batch-size / --no-mirror to override")
    else:
        print(f"[gpu] single GPU: "
              f"{torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}"
              + (f" | batch={cfg.batch_size} grad_accum={cfg.grad_accum} "
                 f"(effective batch={cfg.batch_size * cfg.grad_accum}) "
                 f"| DP off (leak-free path; --mirror to experiment)"
                 if DEVICE.type == "cuda" and N_GPUS > 1 else ""))

    if args.mode == "build":
        build_dataset(cfg, input_root=args.raw_dir,
                      out_root=args.out or None, split=args.split)
    elif args.mode == "pretrain":
        pretrain(cfg)
    elif args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        if not os.path.exists(args.ckpt):
            print(f"[eval] checkpoint not found: {args.ckpt}\n"
                  f"       Train first (main(\"train\")) or pass "
                  f"--ckpt <path/to/best_v11.pt>")
            return
        model, mcfg = load_checkpoint(args.ckpt)
        scenes = discover_scenes(cfg.test_dir)
        assert scenes, "eval: set --test-dir"
        evaluate(model, mcfg, scenes, split="test")
    elif args.mode == "export":
        if not (args.scene and args.ref and args.out):
            print("[export] requires <scene> <ref.tif> <out_dir>")
            return
        if not os.path.exists(args.ckpt):
            print(f"[export] checkpoint not found: {args.ckpt}\n"
                  f"        Train first (main(\"train\")) or pass "
                  f"--ckpt <path/to/best_v11.pt>")
            return
        model, mcfg = load_checkpoint(args.ckpt)
        export_products(model, mcfg, args.scene, args.ref, args.out,
                        sar_only=args.sar_only,
                        optical_only=args.optical_only,
                        gradcam=args.gradcam, review=not args.no_review)
    elif args.mode == "export-edge":
        export_edge(cfg, ckpt=args.ckpt, out_dir=args.out or WORK_DIR)
    elif args.mode == "smooth":
        if not args.scene:
            print("[smooth] requires <prob_dir> with per-date 'seg_probs' NPZs")
            return
        smooth_mode(args.scene, cfg)


def _in_interactive():
    """True when running inside a Jupyter/Kaggle notebook cell."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return "ipykernel" in sys.modules


if __name__ == "__main__":
    if _in_interactive():
        # Pasted into a Kaggle notebook cell: define everything, run nothing.
        print(f"Blue Sentinel v{VERSION} ready (Kaggle edition). "
              "Nothing ran yet. In the next cell:\n"
              '    main("smoke")   # ~2-5 min self-test (recommended first)\n'
              '    main(["build"])  # if your dataset is RAW GEE GeoTIFFs\n'
              '    main("train")   # full training -> /kaggle/working/\n'
              '    main(["train", "--resume"])   # after a 12h session cut\n'
              '    main(["eval", "--test-dir", "/kaggle/input/mangrove-v11/test"])\n'
              "See the file header for export/smooth/pretrain examples.")
    else:
        main()