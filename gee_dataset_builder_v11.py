# -*- coding: utf-8 -*-
"""
================================================================================
 BLUE SENTINEL v11 — DATASET BUILDER (Google Earth Engine -> NPZ -> Kaggle)
================================================================================
No public dataset matches v11's 25-channel scene schema. You must BUILD it.
This script has two parts:

  PART A  (run in Google Colab, internet + Earth Engine access):
      Exports per-AOI / per-year GeoTIFF stacks to your Google Drive.
      Requires:  pip install earthengine-api
                 earthengine authenticate --auth_mode=notebook
                 + a Google Cloud project with the Earth Engine API enabled
                 (free for non-commercial/research use: signup at
                  https://earthengine.google.com/noncommercial/)

  Datasets used (all public in the GEE catalog):
      Sentinel-2 SR Harmonized   COPERNICUS/S2_SR_HARMONIZED (10 bands + SCL)
      Sentinel-1 GRD (dB)        COPERNICUS/S1_GRD           (VV,VH mean+std)
      ALOS-2/PALSAR-2 mosaic     JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH (HH,HV)
      CHIRPS daily precip        UCSB-CHG/CHIRPS/DAILY
      Surface water occurrence   JRC/GSW1_4/GlobalSurfaceWater
      Elevation                  NASA/NASADEM_HGT/001
      MODIS LST 1km              MODIS/061/MOD11A2
      Mangrove extent / change   ESA/WorldCover/v100 (2020), /v200 (2021), cls 95
      GEDI canopy height rh98    LARSE/GEDI/GEDI02_A_002_MONTHLY

  PART B  (run locally):
      rasterio -> per-date step NPZs in exactly the v11 folder schema:
        scene_x/step_2019.npz : s2 (H,W,10) refl 0..1.5 | scl | s1 (H,W,2) dB |
          s1_std | lband HH,HV dB | precip mm | inundation | aux (H,W,6) | date
        LAST step additionally: label (0-4,255 ignore) | change | recovery |
          change_year (-999) | gedi (NaN) | carbon (-999) | eco (H,W,2)

  Then zip and upload as a Kaggle dataset so it lands at /kaggle/input/...,
  matching Config.train_dir/val_dir/test_dir.

 NOTE ON LABELS: v11 auto-disables heads whose label keys are missing
 (validate_heads_vs_data). We emit label/change/recovery/change_year/gedi/
 carbon/eco. species/debris/risk/depth are left absent ON PURPOSE -> those
 heads auto-disable. `depth` SHOULD stay absent only after you apply the
 one-line pseudo-label fix in blue-sentinel-v11.py (see review, Bug #1),
 otherwise the hydro head will CRASH.
================================================================================
"""

# ============================================================================ #
#  PART A — GEE EXPORTS (Colab)                                                #
# ============================================================================ #

AOIS = [  # name, lon, lat, half-size (deg). ~0.1 deg ~= 11 km ~= 1100 px @10m
    ("sundarbans_bd",   89.00, 21.90, 0.12),
    ("irrawaddy_mm",    95.20, 15.90, 0.10),
    ("mekong_vn",      106.60, 10.40, 0.08),
    ("gulf_guayaquil", -80.60, -2.50, 0.08),
    ("rufiji_tz",       38.90, -7.90, 0.08),
    ("zambezi_mz",      36.30, -18.80, 0.10),
    ("borneo_kalimantan", 116.40, -2.90, 0.10),
    ("papua_id",       138.50, -5.60, 0.12),
]
YEARS = [2019, 2020, 2021, 2022, 2023]   # T=5 matches cfg.temporal_steps
SCALE = 10
DRIVE_FOLDER = "blue_sentinel_v11_raw"

# band order in the exported per-year TIFF (Part B depends on this!)
STEP_BANDS = (["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
               "SCL", "VV", "VH", "VV_std", "VH_std", "HH", "HV",
               "precip", "inund"])
STATIC_BANDS = (["elev", "slope", "ndvi_trend", "flood_freq",
                 "lst_trend", "s1_trend",
                 "label", "change", "recovery", "change_year",
                 "gedi", "carbon", "eco_lst", "eco_ndmi"])


def gee_export_all(aois=AOIS, years=YEARS, project=None):
    import ee
    if not project:
        raise SystemExit("gee_export_all(project='YOUR_GCLOUD_PROJECT') "
                         "required — create a free Cloud project with the "
                         "Earth Engine API enabled first.")
    ee.Initialize(project=project)

    # ---- helpers -----------------------------------------------------------
    def s2_masked_col(year, aoi):
        def mask(im):
            scl = im.select("SCL")
            keep = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)))  # drop clouds/shadow
            return im.updateMask(keep)
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .map(mask))

    def s2_step(year, aoi):
        col = s2_masked_col(year, aoi)
        bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        refl = col.select(bands).median().multiply(1e-4)          # -> 0..1.5
        scl = col.select("SCL").mode().unmask(0).rename("SCL")    # 0 = no data
        return refl.rename(bands).addBands(scl), col

    def s1_step(year, aoi):
        col = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(aoi)
               .filterDate(f"{year}-01-01", f"{year}-12-31")
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .select(["VV", "VH"]))           # GEE S1_GRD is already dB
        mean = col.mean().rename(["VV", "VH"])
        std = col.reduce(ee.Reducer.stdDev()).rename(["VV_std", "VH_std"])
        return mean.addBands(std)

    def alos_step(year, aoi):
        # JAXA mosaic covers ~2015-2021; clamp later years to the 2021 mosaic
        y = min(year, 2021)
        col = (ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
               .filterBounds(aoi)
               .filterDate(f"{y}-01-01", f"{y}-12-31").select(["HH", "HV"]))
        # gamma0[dB] = 10*log10(DN^2) - 83  (JAXA standard calibration)
        dn = col.mosaic()
        db = dn.pow(2).log10().multiply(10).subtract(83.0).rename(["HH", "HV"])
        return db

    def chirps_annual(year, aoi):
        return (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .sum().rename("precip").clip(aoi))

    def inund_prior(_year, aoi):
        # JRC occurrence (0-100 %) -> depth-like prior in "mm-equivalents".
        # Honest caveat: this is a static prior, NOT a tide model.
        occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        return occ.multiply(200.0).rename("inund").clip(aoi)

    def utm_epsg(aoi):
        lon = aoi.centroid(1).coordinates().get(0)
        lon = ee.Number(lon)
        zone = lon.add(180).divide(6).floor().add(1)
        south = aoi.centroid(1).coordinates().get(1)
        base = ee.Algorithms.If(ee.Number(south).lt(0), 32700, 32600)
        return ee.Number(base).add(zone).getInfo()

    def stack_with_time(img, year, name):
        return (ee.Image.constant(year).rename("t").toFloat()
                .addBands(img.rename(name).toFloat()))

    def trend(name, imgs):  # imgs: list of (image, year); linear slope / year
        col = ee.ImageCollection([stack_with_time(im, y, name)
                                  for im, y in imgs]).select(["t", name])
        return (col.reduce(ee.Reducer.linearFit())
                .select("scale").rename(name + "_trend"))

    # ---- per-AOI loop ------------------------------------------------------
    tasks = []
    for name, lon, lat, half in aois:
        aoi = ee.Geometry.Rectangle([lon - half, lat - half,
                                     lon + half, lat + half])
        epsg = f"EPSG:{utm_epsg(aoi)}"
        region = aoi.bounds(1)

        # --- yearly dynamic stacks ---
        yearly_s2 = {}
        for y in years:
            s2img, _col = s2_step(y, aoi)
            yearly_s2[y] = s2img
            img = (s2img
                   .addBands(s1_step(y, aoi))
                   .addBands(alos_step(y, aoi))
                   .addBands(chirps_annual(y, aoi))
                   .addBands(inund_prior(y, aoi)))
            img = img.select(STEP_BANDS).toFloat()
            tasks.append(ee.batch.Export.image.toDrive(
                image=img, description=f"{name}_{y}",
                folder=DRIVE_FOLDER, fileNamePrefix=f"{name}_{y}",
                region=region, scale=SCALE, crs=epsg, maxPixels=1e13))

        # --- static / aux ---
        elev = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
        slope = ee.Terrain.slope(elev).rename("slope")
        elev = elev.rename("elev")

        ndvi_trend = trend("ndvi", [
            (s2img.normalizedDifference(["B8", "B4"]), y)
            for y, s2img in yearly_s2.items()])

        flood_freq = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                      .select("occurrence").divide(100.0).rename("flood_freq"))

        lst_an = []
        for y in years:
            lst = (ee.ImageCollection("MODIS/061/MOD11A2")
                   .filterDate(f"{y}-01-01", f"{y}-12-31")
                   .select("LST_Day_1km").mean()
                   .multiply(0.02).subtract(273.15))          # K -> degC
            lst_an.append((lst, y))
        lst_trend = trend("lst", lst_an)

        s1_an = []
        for y in years:
            vv = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(aoi).filterDate(f"{y}-01-01", f"{y}-12-31")
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .select("VV").mean())
            s1_an.append((vv, y))
        s1_trend = trend("s1", s1_an)

        # --- labels (WorldCover 2020 / 2021, class 95 = mangroves) ----------
        wc20 = ee.Image("ESA/WorldCover/v100").select("Map")
        wc21 = ee.Image("ESA/WorldCover/v200").select("Map")
        mang21 = wc21.eq(95)
        ndvi_last = yearly_s2[years[-1]].normalizedDifference(["B8", "B4"])
        occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

        label = (ee.Image(0)                                   # 0 bare/agri
                 .where(occ.gte(50), 1)                        # 1 tidal water
                 .where(occ.gte(90), 2)                        # 2 perm flood
                 .where(mang21.And(ndvi_last.lte(0.45)), 3)    # 3 degraded
                 .where(mang21.And(ndvi_last.gt(0.45)), 4))    # 4 dense
        # 255 = ignore where the last-year composite has no valid obs
        valid = yearly_s2[years[-1]].select("B4").mask()
        label = label.updateMask(valid).unmask(255).rename("label").toInt16()

        loss = wc20.eq(95).And(wc21.neq(95))
        gain = wc20.neq(95).And(wc21.eq(95))
        change = ee.Image(0).where(loss, 1).updateMask(valid) \
                 .unmask(255).rename("change").toInt16()
        recovery = ee.Image(0).where(gain, 1).updateMask(valid) \
                   .unmask(255).rename("recovery").toInt16()
        change_year = (ee.Image.constant(-999.0)
                       .where(loss, float(years[-1]))
                       .updateMask(valid).unmask(-999.0)
                       .rename("change_year").toFloat())

        # --- GEDI canopy proxy ---
        gedi = (ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
                .filterBounds(aoi)
                .filterDate(f"{years[-1]-1}-01-01", f"{years[-1]}-12-31")
                .select("rh98").median().rename("gedi"))

        # --- carbon pseudo-GT (PLACEHOLDER allometry; see README note) ------
        # Replace with a calibrated mangrove biomass layer if you can
        # (e.g. Simard et al. 2019 AGB raster). Kept conservative + capped.
        agb = gedi.multiply(6.0).clamp(0, 250)
        carbon = agb.where(mang21.And(gedi.gt(0)), agb) \
                    .where(mang21.Not().Or(gedi.lte(0)), -999.0) \
                    .rename("carbon").toFloat()

        # --- eco: [LST degC (last year), NDMI (last year)] ------------------
        eco_lst = lst_an[-1][0].rename("eco_lst")
        eco_ndmi = yearly_s2[years[-1]].normalizedDifference(["B8", "B11"]) \
                   .rename("eco_ndmi")

        static = (elev.addBands(slope).addBands(ndvi_trend).addBands(flood_freq)
                  .addBands(lst_trend).addBands(s1_trend)
                  .addBands(label).addBands(change).addBands(recovery)
                  .addBands(change_year).addBands(gedi).addBands(carbon)
                  .addBands(eco_lst).addBands(eco_ndmi))
        static = static.select(STATIC_BANDS).toFloat()
        tasks.append(ee.batch.Export.image.toDrive(
            image=static, description=f"{name}_static",
            folder=DRIVE_FOLDER, fileNamePrefix=f"{name}_static",
            region=region, scale=SCALE, crs=epsg, maxPixels=1e13))

    for t in tasks:
        t.start()
    print(f"started {len(tasks)} Earth Engine export tasks -> "
          f"Drive folder '{DRIVE_FOLDER}'. Watch: https://code.earthengine.google.com/tasks")
    return tasks


# ============================================================================ #
#  PART B — TIFF -> v11 NPZ SCENE PACKER  (local; pip install rasterio numpy)  #
# ============================================================================ #
def pack_scene(raw_dir, aoi_name, years, out_root):
    """raw_dir/{aoi}_{year}.tif + {aoi}_static.tif -> out_root/{aoi}/step_*.npz"""
    import os
    import numpy as np
    import rasterio

    out_dir = os.path.join(out_root, aoi_name)
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(os.path.join(raw_dir, f"{aoi_name}_static.tif")) as s:
        st = {b: s.read(i + 1) for i, b in enumerate(STATIC_BANDS)}
        assert len(STATIC_BANDS) == s.count, "band-order mismatch in static tif"

    aux = np.stack([st["elev"], st["slope"], st["ndvi_trend"],
                    st["flood_freq"], st["lst_trend"], st["s1_trend"]], -1)
    aux = np.nan_to_num(aux, nan=0.0).astype(np.float32)          # (H,W,6)

    gedi = st["gedi"].astype(np.float32)
    gedi = np.where(gedi > 0, gedi, np.nan).astype(np.float32)    # <=0 -> NaN
    carbon = st["carbon"].astype(np.float32)                      # -999 ignored
    eco = np.stack([np.nan_to_num(st["eco_lst"], nan=0.0),
                    np.nan_to_num(st["eco_ndmi"], nan=0.0)], -1).astype(np.float32)

    n = len(years)
    for k, y in enumerate(years):
        with rasterio.open(os.path.join(raw_dir, f"{aoi_name}_{y}.tif")) as d:
            stp = {b: d.read(i + 1) for i, b in enumerate(STEP_BANDS)}
            assert len(STEP_BANDS) == d.count, "band-order mismatch in step tif"

        s2 = np.stack([stp[b] for b in STEP_BANDS[:10]], -1).astype(np.float32)
        scl = np.nan_to_num(stp["SCL"], nan=0.0).astype(np.float32)
        s1 = np.stack([np.nan_to_num(stp["VV"], nan=-30.0),
                       np.nan_to_num(stp["VH"], nan=-30.0)], -1).astype(np.float32)
        s1_std = np.stack([np.nan_to_num(stp["VV_std"], nan=0.0),
                           np.nan_to_num(stp["VH_std"], nan=0.0)], -1) \
            .astype(np.float32)
        lband = np.stack([np.nan_to_num(stp["HH"], nan=-35.0),
                          np.nan_to_num(stp["HV"], nan=-35.0)], -1) \
            .astype(np.float32)
        precip = np.nan_to_num(stp["precip"], nan=0.0).astype(np.float32)
        inund = np.nan_to_num(stp["inund"], nan=0.0).astype(np.float32)

        keys = dict(s2=s2, scl=scl, s1=s1, s1_std=s1_std, lband=lband,
                    precip=precip, inundation=inund, aux=aux,
                    date=np.float32(float(y)))
        if k == n - 1:   # v11 expects labels ONLY on the last step
            keys.update(
                label=st["label"].astype(np.int64),            # 255 = ignore
                change=st["change"].astype(np.int64),
                recovery=st["recovery"].astype(np.int64),
                change_year=st["change_year"].astype(np.float32),  # -999
                gedi=gedi, carbon=carbon, eco=eco)
            # NO 'depth' key -> hydro uses the inundation pseudo-label
            # (only safe after Bug #1 fix in blue-sentinel-v11.py!)
        np.savez_compressed(os.path.join(out_dir, f"step_{y}.npz"), **keys)
        print(f"  packed {aoi_name}/step_{y}.npz")
    return out_dir


def pack_all(raw_dir, out_root, aois=AOIS, years=YEARS):
    for name, *_ in aois:
        pack_scene(raw_dir, name, years, out_root)
    print(f"done -> {out_root}  (split scenes into train/val/test folders, "
          f"zip, then upload to Kaggle)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["gee", "pack"])
    ap.add_argument("--raw-dir", default="./gee_raw")
    ap.add_argument("--out", default="./mangrove-v11")
    args = ap.parse_args()
    if args.part == "gee":
        gee_export_all()
    else:
        pack_all(args.raw_dir, args.out)
