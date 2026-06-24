#!/usr/bin/env python3
"""
train_svm.py -- M-3 SVM Landslide Susceptibility Mapping
Hong Kong validation site (ENTLI + SRTM + ESA WorldCover)

Usage:
    conda activate m3_svm
    cd ~/landslide-toolkit/M3/hongkong
    python ~/M3-SVM-Susceptibility/train_svm.py

Outputs (written to ~/landslide-toolkit/M3/hongkong/model/):
    svm_hk_pipeline.pkl      -- trained sklearn Pipeline (scaler + SVM)
    susceptibility_hk.tif   -- failure probability raster, full HK territory
    roc_curve.csv            -- FPR/TPR/threshold from test set
    feature_importance.csv   -- permutation importance scores
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from shapely.geometry import Point
import joblib, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/landslide-toolkit/M3/hongkong")
OUT  = os.path.join(BASE, "features")
MDL  = os.path.join(BASE, "model")
os.makedirs(MDL, exist_ok=True)

ENTLI_GDB   = os.path.join(BASE, "ENTLI (Up to Year 2022).gdb")
DEM_MERGED  = os.path.join(BASE, "dem/merged.tif")
WC_TILES    = [os.path.join(OUT, f"worldcover_N21E{t}.tif") for t in ["111","114"]]
SRTM_TILES  = [os.path.join(BASE, f"dem/N22E{t}.hgt") for t in ["113","114"]]

# ── Step 1: Build terrain feature rasters (skip if already present) ────────
def build_terrain_features():
    dem_path = os.path.join(OUT, "dem_hk2326.tif")
    if os.path.exists(dem_path):
        print("Terrain features already exist, skipping derivation.")
        return

    print("=== Building terrain features ===")
    # Merge + reproject SRTM
    src_files = [rasterio.open(p) for p in SRTM_TILES]
    mosaic, t = merge(src_files)
    meta = src_files[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=t)
    merged = os.path.join(BASE, "dem/merged.tif")
    with rasterio.open(merged, "w", **meta) as dst:
        dst.write(mosaic)
    for s in src_files: s.close()

    with rasterio.open(merged) as src:
        tf, w, h = calculate_default_transform(
            src.crs, "EPSG:2326", src.width, src.height, *src.bounds, resolution=30)
        m = src.meta.copy()
        m.update(crs="EPSG:2326", transform=tf, width=w, height=h, compress="lzw")
        with rasterio.open(dem_path, "w", **m) as dst:
            reproject(rasterio.band(src,1), rasterio.band(dst,1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tf, dst_crs="EPSG:2326",
                      resampling=Resampling.bilinear)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        res = src.res[0]
        nd  = src.nodata
        if nd: dem[dem == nd] = np.nan
        ref_meta = src.meta.copy()

    def save(arr, name):
        m = ref_meta.copy()
        m.update(dtype="float32", nodata=-9999, compress="lzw")
        with rasterio.open(os.path.join(OUT, f"{name}.tif"), "w", **m) as dst:
            a = arr.astype("float32")
            a[~np.isfinite(a)] = -9999
            dst.write(a, 1)
        print(f"  {name} saved")

    save(np.where(np.isnan(dem), -9999, dem), "elevation")
    dz_dx = np.gradient(dem, res, axis=1)
    dz_dy = np.gradient(dem, res, axis=0)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    save(np.degrees(slope_rad), "slope")
    save(np.degrees(np.arctan2(-dz_dy, dz_dx)) % 360, "aspect")
    d2x = np.gradient(dz_dx, res, axis=1)
    d2y = np.gradient(dz_dy, res, axis=0)
    dxy = np.gradient(dz_dx, res, axis=0)
    denom = dz_dx**2 + dz_dy**2
    with np.errstate(invalid='ignore', divide='ignore'):
        pcurv = np.where(denom > 0,
            -(dz_dx**2*d2x + 2*dz_dx*dz_dy*dxy + dz_dy**2*d2y)
            / (denom * np.sqrt(1 + denom)), 0.0)
    save(pcurv, "profile_curvature")
    rows, cols = dem.shape
    acc = np.ones((rows, cols))
    dx = [0,1,1,1,0,-1,-1,-1]; dy = [-1,-1,0,1,1,1,0,-1]
    flat = np.argsort(dem.flatten())[::-1]
    ri, ci = np.unravel_index(flat, (rows, cols))
    for r, c in zip(ri, ci):
        if np.isnan(dem[r,c]): continue
        mn, bd = dem[r,c], -1
        for d in range(8):
            nr, nc = r+dy[d], c+dx[d]
            if 0<=nr<rows and 0<=nc<cols and not np.isnan(dem[nr,nc]) and dem[nr,nc]<mn:
                mn, bd = dem[nr,nc], d
        if bd >= 0:
            acc[r+dy[bd], c+dx[bd]] += acc[r,c]
    with np.errstate(invalid='ignore', divide='ignore'):
        ts = np.where(np.tan(slope_rad) < 0.001, 0.001, np.tan(slope_rad))
        twi = np.where(np.isnan(dem), np.nan, np.log((acc*res*res)/ts))
    save(twi, "twi")

    # WorldCover
    srcs = [rasterio.open(p) for p in WC_TILES]
    mosaic, t = merge(srcs)
    wm = ref_meta.copy(); wm.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=t)
    tmp = os.path.join(OUT, "worldcover_merged_wgs84.tif")
    with rasterio.open(tmp, "w", **wm) as dst: dst.write(mosaic)
    for s in srcs: s.close()
    with rasterio.open(tmp) as src:
        tf2, w2, h2 = calculate_default_transform(
            src.crs, "EPSG:2326", src.width, src.height, *src.bounds, resolution=30)
        m2 = src.meta.copy()
        m2.update(crs="EPSG:2326", transform=tf2, width=w2, height=h2, compress="lzw")
        with rasterio.open(os.path.join(OUT,"worldcover_hk2326.tif"), "w", **m2) as dst:
            reproject(rasterio.band(src,1), rasterio.band(dst,1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tf2, dst_crs="EPSG:2326",
                      resampling=Resampling.nearest)
    with rasterio.open(os.path.join(OUT,"worldcover_hk2326.tif")) as wc_src, \
         rasterio.open(dem_path) as dem_ref:
        m3 = dem_ref.meta.copy(); m3.update(dtype="uint8", nodata=0, compress="lzw")
        with rasterio.open(os.path.join(OUT,"worldcover_matched.tif"), "w", **m3) as dst:
            reproject(rasterio.band(wc_src,1), rasterio.band(dst,1),
                      src_transform=wc_src.transform, src_crs=wc_src.crs,
                      dst_transform=dem_ref.transform, dst_crs=dem_ref.crs,
                      resampling=Resampling.nearest)

# ── Step 2: Build training samples ────────────────────────────────────────
def build_training_data():
    csv_path = os.path.join(OUT, "training_samples.csv")
    if os.path.exists(csv_path):
        print("Training samples already exist, loading...")
        return pd.read_csv(csv_path)

    print("=== Building training dataset ===")
    crown = gpd.read_file(ENTLI_GDB, layer="ENTLI_Crown")
    clean = crown[(crown['SLOPE'] < 9999) & (crown['HEADELEV'] < 9999)].copy()
    n_pos = 5000
    strata = clean['SLIDE_TYPE'].value_counts(normalize=True)
    positives = []
    for stype, prop in strata.items():
        n = max(1, int(round(prop * n_pos)))
        positives.append(clean[clean['SLIDE_TYPE']==stype].sample(
            min(n, sum(clean['SLIDE_TYPE']==stype)), random_state=42))
    positives = gpd.GeoDataFrame(pd.concat(positives).head(n_pos), crs=clean.crs)
    positives['label'] = 1

    with rasterio.open(os.path.join(OUT,"elevation.tif")) as dem_src:
        bounds = dem_src.bounds
        dem_data = dem_src.read(1).astype(float)
        dem_nd = dem_src.nodata
    all_buf = clean.geometry.buffer(250).union_all()
    negatives = []
    while len(negatives) < n_pos:
        xs = np.random.uniform(bounds.left, bounds.right, 1000)
        ys = np.random.uniform(bounds.bottom, bounds.top, 1000)
        for x, y in zip(xs, ys):
            if len(negatives) >= n_pos: break
            if not all_buf.contains(Point(x, y)):
                try:
                    with rasterio.open(os.path.join(OUT,"elevation.tif")) as s:
                        r, c = s.index(x, y)
                    if 0<=r<dem_data.shape[0] and 0<=c<dem_data.shape[1]:
                        v = dem_data[r,c]
                        if (dem_nd is None or v != dem_nd) and v > 0:
                            negatives.append({'geometry': Point(x,y), 'label': 0})
                except: pass
    negatives_gdf = gpd.GeoDataFrame(negatives, crs="EPSG:2326")

    all_pts = gpd.GeoDataFrame(
        pd.concat([positives[['geometry','label']].reset_index(drop=True),
                   negatives_gdf.reset_index(drop=True)], ignore_index=True),
        geometry='geometry', crs='EPSG:2326')

    feat_files = {'elevation': 'elevation.tif', 'slope': 'slope.tif',
                  'aspect': 'aspect.tif', 'profile_curvature': 'profile_curvature.tif',
                  'twi': 'twi.tif', 'landcover': 'worldcover_matched.tif'}
    for fname, ffile in feat_files.items():
        coords = [(g.x, g.y) for g in all_pts.geometry]
        with rasterio.open(os.path.join(OUT, ffile)) as src:
            all_pts[fname] = np.array([x[0] for x in src.sample(coords)], dtype=float)

    df = all_pts.drop(columns='geometry')
    df.to_csv(csv_path, index=False)
    return df

# ── Step 3: Train and evaluate SVM ────────────────────────────────────────
def train_and_evaluate(df):
    print("=== Training SVM ===")
    features = ['elevation','slope','aspect','profile_curvature','twi','landcover']
    X = df[features].values
    y = df['label'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                                 probability=True, random_state=42))])
    pipe.fit(X_tr, y_tr)

    y_prob = pipe.predict_proba(X_te)[:,1]
    y_pred = pipe.predict(X_te)
    auc = roc_auc_score(y_te, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"rTP: {tp/(tp+fn)*100:.1f}%  rTN: {tn/(tn+fp)*100:.1f}%")
    print(classification_report(y_te, y_pred,
                                target_names=['Non-landslide','Landslide']))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"CV AUC: {cv_aucs.mean():.4f} +/- {cv_aucs.std():.4f}")

    perm = permutation_importance(pipe, X_te, y_te, n_repeats=10,
                                  random_state=42, scoring='roc_auc', n_jobs=-1)
    imp_df = pd.DataFrame({'feature': features,
                           'importance': perm.importances_mean}).sort_values(
        'importance', ascending=False)
    print("\nFeature importance:\n", imp_df.to_string(index=False))

    joblib.dump(pipe, os.path.join(MDL, "svm_hk_pipeline.pkl"))
    fpr, tpr, thr = roc_curve(y_te, y_prob)
    pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thr}).to_csv(
        os.path.join(MDL,"roc_curve.csv"), index=False)
    imp_df.to_csv(os.path.join(MDL,"feature_importance.csv"), index=False)
    return pipe, features

# ── Step 4: Predict susceptibility map ────────────────────────────────────
def predict_map(pipe, features):
    print("=== Predicting susceptibility map ===")
    feat_files = ['elevation.tif','slope.tif','aspect.tif',
                  'profile_curvature.tif','twi.tif','worldcover_matched.tif']
    arrays = []
    with rasterio.open(os.path.join(OUT, feat_files[0])) as ref:
        meta = ref.meta.copy()
        rows, cols = ref.height, ref.width
    for f in feat_files:
        with rasterio.open(os.path.join(OUT, f)) as src:
            arr = src.read(1).astype(float)
            nd = src.nodata
            if nd is not None: arr[arr==nd] = np.nan
        arrays.append(arr)

    X_full = np.stack([a.flatten() for a in arrays], axis=1)
    valid = np.all(np.isfinite(X_full), axis=1) & (X_full[:,0] > 0)
    prob_map = np.full(rows*cols, np.nan)
    idx = np.where(valid)[0]
    for i in range(0, len(idx), 100_000):
        batch = idx[i:i+100_000]
        prob_map[batch] = pipe.predict_proba(X_full[batch])[:,1]
        if (i//100_000+1) % 10 == 0:
            print(f"  {i//100_000+1}/{int(np.ceil(len(idx)/100_000))} batches done")

    prob_map = prob_map.reshape(rows, cols)
    meta.update(dtype='float32', nodata=-9999, compress='lzw')
    out_path = os.path.join(MDL, "susceptibility_hk.tif")
    with rasterio.open(out_path, 'w', **meta) as dst:
        out = prob_map.astype('float32')
        out[np.isnan(prob_map)] = -9999
        dst.write(out, 1)
    vp = prob_map[np.isfinite(prob_map)]
    print(f"Saved: {out_path}")
    print(f"  Pf range: {vp.min():.3f} - {vp.max():.3f}, mean: {vp.mean():.3f}")
    print(f"  % area > 0.5: {(vp>0.5).mean()*100:.1f}%")

if __name__ == "__main__":
    os.chdir(BASE)
    build_terrain_features()
    df = build_training_data()
    pipe, features = train_and_evaluate(df)
    predict_map(pipe, features)
    print("\nDone.")
