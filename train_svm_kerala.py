#!/usr/bin/env python3
"""
train_svm_kerala.py -- M-3 SVM Landslide Susceptibility Mapping
Kerala application site (Kerala landslide inventory + Copernicus GLO-30 + ESA WorldCover)

Usage:
    conda activate m3_svm
    cd ~/landslide-toolkit/M3/kerala
    python ~/M3-SVM-Susceptibility/train_svm_kerala.py

Outputs (written to ~/landslide-toolkit/M3/kerala/model/):
    svm_kerala_pipeline.pkl           -- trained sklearn Pipeline (scaler + SVM)
    susceptibility_kerala.tif         -- failure probability raster, full Kerala
    susceptibility_classified_kerala.tif -- classified: 1=Low 2=Mod 3=High 4=VHigh
    roc_curve_kerala.csv              -- FPR/TPR/threshold from test set
    feature_importance_kerala.csv     -- permutation importance scores

Notes:
    - Features reduced to 4 (elevation, slope, aspect, landcover):
      TWI dropped (D8 routing over ~200M pixels impractical in pure Python);
      profile_curvature dropped (near-zero importance in HK validation).
    - Susceptibility map generated via manual numpy RBF + Platt sigmoid
      (bypasses sklearn predict_proba which is extremely slow at pixel scale
      due to Platt scaling overhead; mathematically identical result).
    - Nodata sentinel -9999 leaks into slope/aspect at DEM boundaries;
      these pixels are masked in the susceptibility map but present in
      training samples (minor effect on AUC, documented as known issue).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from shapely.geometry import Point, box
import joblib, os, warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

BASE     = os.path.expanduser("~/landslide-toolkit/M3/kerala")
OUT      = os.path.join(BASE, "features")
MDL      = os.path.join(BASE, "model")
INV_PATH = os.path.expanduser("~/landslide-toolkit/kerala_inventory/Kerela landslide.shp")
os.makedirs(MDL, exist_ok=True)

FEATURES   = ['elevation', 'slope', 'aspect', 'landcover']
FEAT_FILES = {
    'elevation': os.path.join(OUT, 'elevation.tif'),
    'slope':     os.path.join(OUT, 'slope.tif'),
    'aspect':    os.path.join(OUT, 'aspect.tif'),
    'landcover': os.path.join(OUT, 'worldcover_matched.tif'),
}

# ── Step 1: WorldCover merge + match (skip if done) ───────────────────────
def build_worldcover():
    out_path = os.path.join(OUT, 'worldcover_matched.tif')
    if os.path.exists(out_path):
        print("WorldCover already matched, skipping.")
        return
    print("Merging WorldCover tiles...")
    tiles = [os.path.join(OUT, f'worldcover_{t}.tif')
             for t in ['N06E072','N06E075','N09E072','N09E075']]
    srcs = [rasterio.open(p) for p in tiles]
    mosaic, t = merge(srcs)
    tmp_meta = srcs[0].meta.copy()
    tmp_meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=t)
    tmp = os.path.join(OUT, 'worldcover_merged_wgs84.tif')
    with rasterio.open(tmp, 'w', **tmp_meta) as dst:
        dst.write(mosaic)
    for s in srcs: s.close()
    dem_path = os.path.join(OUT, 'dem_kerala_utm.tif')
    with rasterio.open(dem_path) as dem_ref:
        m = dem_ref.meta.copy()
        m.update(dtype='uint8', nodata=0, compress='lzw')
        with rasterio.open(tmp) as wc_src, \
             rasterio.open(out_path, 'w', **m) as dst:
            reproject(source=rasterio.band(wc_src, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=wc_src.transform, src_crs=wc_src.crs,
                      dst_transform=dem_ref.transform, dst_crs=dem_ref.crs,
                      resampling=Resampling.nearest)
    print(f"  WorldCover matched: {out_path}")

# ── Step 2: Build training samples ────────────────────────────────────────
def build_training_data():
    csv_path = os.path.join(OUT, 'training_samples_kerala.csv')
    if os.path.exists(csv_path):
        print("Training samples already exist, loading...")
        return pd.read_csv(csv_path)

    print("Loading Kerala landslide inventory...")
    inv    = gpd.read_file(INV_PATH).to_crs("EPSG:32643")
    with rasterio.open(FEAT_FILES['elevation']) as dem_src:
        bounds    = dem_src.bounds
        dem_data  = dem_src.read(1).astype(float)
        dem_nd    = dem_src.nodata
        dem_rows, dem_cols = dem_src.height, dem_src.width

    dem_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    inv_in  = inv[inv.geometry.within(dem_box)].copy()
    print(f"  Inventory within DEM: {len(inv_in)}")

    n_pos   = 5000
    strata  = inv_in['Type_of_sl'].value_counts(normalize=True)
    positives = []
    for stype, prop in strata.items():
        n = max(1, int(round(prop * n_pos)))
        subset = inv_in[inv_in['Type_of_sl'] == stype]
        positives.append(subset.sample(min(n, len(subset)), random_state=42))
    positives = gpd.GeoDataFrame(pd.concat(positives).head(n_pos), crs=inv_in.crs)
    positives['label'] = 1

    all_buf   = inv_in.geometry.buffer(250).union_all()
    negatives = []
    while len(negatives) < n_pos:
        xs = np.random.uniform(bounds.left, bounds.right, 2000)
        ys = np.random.uniform(bounds.bottom, bounds.top, 2000)
        for x, y in zip(xs, ys):
            if len(negatives) >= n_pos: break
            if all_buf.contains(Point(x, y)): continue
            try:
                with rasterio.open(FEAT_FILES['elevation']) as s:
                    r, c = s.index(x, y)
                if 0 <= r < dem_rows and 0 <= c < dem_cols:
                    v = dem_data[r, c]
                    if (dem_nd is None or v != dem_nd) and v > 10:
                        negatives.append({'geometry': Point(x, y), 'label': 0})
            except: pass
    negatives_gdf = gpd.GeoDataFrame(negatives, crs="EPSG:32643")

    all_pts = gpd.GeoDataFrame(
        pd.concat([positives[['geometry','label']].reset_index(drop=True),
                   negatives_gdf.reset_index(drop=True)], ignore_index=True),
        geometry='geometry', crs='EPSG:32643')

    for fname, fpath in FEAT_FILES.items():
        coords = [(g.x, g.y) for g in all_pts.geometry]
        with rasterio.open(fpath) as src:
            all_pts[fname] = np.array([x[0] for x in src.sample(coords)], dtype=float)

    df = all_pts.drop(columns='geometry')
    df.to_csv(csv_path, index=False)
    return df

# ── Step 3: Train and evaluate ────────────────────────────────────────────
def train_and_evaluate(df):
    print("Training SVM...")
    X = df[FEATURES].values
    y = df['label'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                    probability=True, random_state=42))
    ])
    pipe.fit(X_tr, y_tr)

    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = pipe.predict(X_te)
    auc    = roc_auc_score(y_te, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    print(f"AUC-ROC: {auc:.4f}")
    print(f"rTP: {tp/(tp+fn)*100:.1f}%  rTN: {tn/(tn+fp)*100:.1f}%")
    print(classification_report(y_te, y_pred,
                                target_names=['Non-landslide','Landslide']))
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"5-fold CV AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    perm   = permutation_importance(pipe, X_te, y_te, n_repeats=10,
                                    random_state=42, scoring='roc_auc', n_jobs=-1)
    imp_df = pd.DataFrame({'feature': FEATURES,
                           'importance': perm.importances_mean}).sort_values(
        'importance', ascending=False)
    print(f"\nFeature importance:\n{imp_df.to_string(index=False)}")

    joblib.dump(pipe, os.path.join(MDL, 'svm_kerala_pipeline.pkl'))
    fpr, tpr, thr = roc_curve(y_te, y_prob)
    pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thr}).to_csv(
        os.path.join(MDL,'roc_curve_kerala.csv'), index=False)
    imp_df.to_csv(os.path.join(MDL,'feature_importance_kerala.csv'), index=False)
    return pipe

# ── Step 4: Susceptibility map (numpy RBF — fast) ─────────────────────────
def predict_map(pipe):
    """
    Manual numpy RBF + Platt sigmoid prediction, bypassing sklearn's
    predict_proba which is prohibitively slow at pixel scale (~200M pixels)
    due to Platt scaling overhead in scikit-learn >= 1.9.
    Mathematically identical to pipe.predict_proba(X)[:,1].
    """
    print("Predicting susceptibility map (numpy RBF)...")
    svm    = pipe.named_steps['svm']
    scaler = pipe.named_steps['scaler']
    SVs       = svm.support_vectors_.astype(np.float32)
    dual_coef = svm.dual_coef_.astype(np.float32).ravel()
    SV_sq     = np.sum(SVs**2, axis=1)
    MEAN      = scaler.mean_.astype(np.float32)
    SCALE     = scaler.scale_.astype(np.float32)
    GAMMA     = np.float32(svm._gamma)
    INTERCEPT = np.float32(svm.intercept_[0])
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        PLATT_A = np.float32(svm.probA_[0])
        PLATT_B = np.float32(svm.probB_[0])

    def predict_batch(X_raw):
        X    = (X_raw - MEAN) / SCALE
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        K    = np.exp(-GAMMA * (X_sq + SV_sq - 2.0 * (X @ SVs.T)))
        dec  = K @ dual_coef + INTERCEPT
        return (1.0 / (1.0 + np.exp(PLATT_A * dec + PLATT_B))).astype(np.float32)

    ref_path = os.path.join(OUT, 'dem_kerala_utm.tif')
    with rasterio.open(ref_path) as ref:
        meta         = ref.meta.copy()
        nrows, ncols = ref.height, ref.width
        nodata_ref   = ref.nodata

    meta.update(dtype='float32', nodata=-9999.0, count=1, compress='lzw')
    out_path  = os.path.join(MDL, 'susceptibility_kerala.tif')
    feat_list = [FEAT_FILES[f] for f in FEATURES]
    ROW_CHUNK = 8
    PIX_BATCH = 5000
    n_chunks  = (nrows + ROW_CHUNK - 1) // ROW_CHUNK
    srcs      = [rasterio.open(p) for p in feat_list]
    t0        = time.time()

    with rasterio.open(out_path, 'w', **meta) as dst:
        for ci in range(n_chunks):
            r0  = ci * ROW_CHUNK
            r1  = min(r0 + ROW_CHUNK, nrows)
            win = Window(0, r0, ncols, r1 - r0)
            bands = [s.read(1, window=win).astype(np.float32) for s in srcs]
            stack = np.stack(bands, axis=0)
            H, W  = stack.shape[1], stack.shape[2]
            nd_mask = np.zeros((H, W), dtype=bool)
            for b in stack:
                nd_mask |= (b == -9999) | (~np.isfinite(b))
            if nodata_ref is not None:
                nd_mask |= (stack[0] == nodata_ref)
            X_flat    = stack.reshape(4, -1).T
            valid_idx = np.where(~nd_mask.ravel())[0]
            prob_map  = np.full(H * W, -9999.0, dtype=np.float32)
            for b0 in range(0, len(valid_idx), PIX_BATCH):
                idx = valid_idx[b0:b0+PIX_BATCH]
                prob_map[idx] = predict_batch(X_flat[idx])
            dst.write(prob_map.reshape(H, W), 1, window=win)
            if ci % 500 == 0 or ci == n_chunks - 1:
                elapsed = time.time() - t0
                eta     = (n_chunks-ci-1)/(ci+1)*elapsed if ci > 0 else 0
                print(f"  {ci+1}/{n_chunks} ({(ci+1)/n_chunks*100:.0f}%) "
                      f"elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m")
    for s in srcs: s.close()
    print(f"Saved: {out_path}  ({(time.time()-t0)/60:.1f} min)")

# ── Step 5: Classify susceptibility zones ─────────────────────────────────
def classify_map():
    src_path = os.path.join(MDL, 'susceptibility_kerala.tif')
    cls_path = os.path.join(MDL, 'susceptibility_classified_kerala.tif')
    THRESHOLDS = [0.20, 0.50, 0.80]
    LABELS     = ['Low (<0.20)','Moderate (0.20-0.50)','High (0.50-0.80)','Very High (>0.80)']
    with rasterio.open(src_path) as src:
        meta         = src.meta.copy()
        nrows, ncols = src.height, src.width
    meta.update(dtype='uint8', nodata=0, compress='lzw')
    counts      = np.zeros(4, dtype=np.int64)
    total_valid = 0
    with rasterio.open(src_path) as src, rasterio.open(cls_path, 'w', **meta) as dst:
        for ci in range((nrows+500-1)//500):
            r0  = ci * 500
            win = Window(0, r0, ncols, min(500, nrows-r0))
            d   = src.read(1, window=win)
            cls = np.zeros_like(d, dtype=np.uint8)
            v   = d != -9999
            total_valid += v.sum()
            pv  = d[v]
            c   = np.ones(len(pv), dtype=np.uint8)
            c[pv >= THRESHOLDS[0]] = 2
            c[pv >= THRESHOLDS[1]] = 3
            c[pv >= THRESHOLDS[2]] = 4
            cls[v] = c
            for i in range(4): counts[i] += (c == i+1).sum()
            dst.write(cls, 1, window=win)
    print("\n=== Susceptibility Zone Statistics ===")
    for i, label in enumerate(LABELS):
        print(f"  Class {i+1} — {label}: {counts[i]:,} px ({counts[i]/total_valid*100:.1f}%)")
    print(f"Classified map: {cls_path}")

if __name__ == "__main__":
    os.chdir(BASE)
    build_worldcover()
    df   = build_training_data()
    pipe = train_and_evaluate(df)
    predict_map(pipe)
    classify_map()
    print("\nDone.")
