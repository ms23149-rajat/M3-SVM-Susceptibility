import numpy as np, pandas as pd, geopandas as gpd, rasterio
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from shapely.geometry import Point, box
import os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

os.chdir(os.path.expanduser("~/landslide-toolkit/M3/kerala"))
FEAT_FILES = {
    'elevation': "features/elevation.tif",
    'slope':     "features/slope.tif",
    'aspect':    "features/aspect.tif",
    'landcover': "features/worldcover_matched.tif",
}
FEATURES = ['elevation','slope','aspect','landcover']

print("="*72)
print("STEP 1 -- Regenerate point dataset (positives + territory-wide negatives)")
print("="*72)

inv = gpd.read_file(os.path.expanduser("~/landslide-toolkit/kerala_inventory/Kerela landslide.shp")).to_crs("EPSG:32643")
with rasterio.open(FEAT_FILES['elevation']) as dem_src:
    bounds = dem_src.bounds
    dem_data = dem_src.read(1).astype(float)
    dem_nd = dem_src.nodata

dem_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
inv_in = inv[inv.geometry.within(dem_box)].copy()

n_pos = 5000
strata = inv_in['Type_of_sl'].value_counts(normalize=True)
positives = []
for stype, prop in strata.items():
    n = max(1, int(round(prop * n_pos)))
    subset = inv_in[inv_in['Type_of_sl']==stype]
    positives.append(subset.sample(min(n, len(subset)), random_state=42))
positives = gpd.GeoDataFrame(pd.concat(positives).head(n_pos), crs=inv_in.crs)
positives['label'] = 1

all_buf = inv_in.geometry.buffer(250).union_all()
buf_gdf = gpd.GeoDataFrame(geometry=[all_buf], crs="EPSG:32643")

def sample_valid_points(n_raw):
    xs = np.random.uniform(bounds.left, bounds.right, n_raw)
    ys = np.random.uniform(bounds.bottom, bounds.top, n_raw)
    gdf = gpd.GeoDataFrame({'geometry':[Point(x,y) for x,y in zip(xs,ys)]}, crs="EPSG:32643")
    with rasterio.open(FEAT_FILES['elevation']) as s:
        coords = [(g.x,g.y) for g in gdf.geometry]
        elev = np.array([v[0] for v in s.sample(coords)], dtype=float)
    gdf['elevation'] = elev
    valid = (elev > 10) & np.isfinite(elev)
    if dem_nd is not None:
        valid &= (elev != dem_nd)
    gdf = gdf[valid].reset_index(drop=True)
    joined = gpd.sjoin(gdf, buf_gdf, how='left', predicate='within')
    gdf = gdf[joined['index_right'].isna()].reset_index(drop=True)
    return gdf

neg_pool = sample_valid_points(20000)
negatives_gdf = neg_pool.sample(n=min(n_pos, len(neg_pool)), random_state=42).reset_index(drop=True)
negatives_gdf['label'] = 0

print(f"Positives: {len(positives)}   Territory-wide negatives: {len(negatives_gdf)}")

all_pts = gpd.GeoDataFrame(
    pd.concat([positives[['geometry','label']].reset_index(drop=True),
               negatives_gdf[['geometry','label']].reset_index(drop=True)], ignore_index=True),
    geometry='geometry', crs='EPSG:32643')

for fname, fpath in FEAT_FILES.items():
    coords = [(g.x, g.y) for g in all_pts.geometry]
    with rasterio.open(fpath) as src:
        all_pts[fname] = np.array([v[0] for v in src.sample(coords)], dtype=float)
all_pts['x'] = all_pts.geometry.x
all_pts['y'] = all_pts.geometry.y

print("\n" + "="*72)
print("STEP 2 -- Baseline sanity check (regenerated data, random split)")
print("="*72)
X = all_pts[FEATURES].values
y = all_pts['label'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
pipe.fit(X_tr, y_tr)
auc_baseline = roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])
print(f"Baseline AUC (should land near the reported 0.921): {auc_baseline:.4f}")

print("\n" + "="*72)
print("STEP 3 -- Spatial-block CV (10 km blocks) vs random CV")
print("="*72)
BLOCK = 10000
all_pts['block_id'] = (all_pts['x']//BLOCK).astype(int).astype(str) + "_" + (all_pts['y']//BLOCK).astype(int).astype(str)
print(f"Unique 10km blocks occupied: {all_pts['block_id'].nunique()}")

gkf = GroupKFold(n_splits=5)
spatial_aucs = []
for tr_idx, te_idx in gkf.split(X, y, groups=all_pts['block_id']):
    p = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
    p.fit(X[tr_idx], y[tr_idx])
    spatial_aucs.append(roc_auc_score(y[te_idx], p.predict_proba(X[te_idx])[:,1]))
spatial_aucs = np.array(spatial_aucs)
random_cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
print(f"Spatial-block 5-fold CV AUC : {spatial_aucs.mean():.4f} +/- {spatial_aucs.std():.4f}")
print(f"Random 5-fold CV AUC        : {random_cv.mean():.4f} +/- {random_cv.std():.4f}")
print(f"(Prof's independent check: spatial 0.923 vs random 0.918)")

print("\n" + "="*72)
print("STEP 4 -- Terrain contrast: positives vs territory-wide negatives")
print("="*72)
pos_df = all_pts[all_pts.label==1].copy()
neg_df = all_pts[all_pts.label==0].copy()
print(f"Positives      : slope median={pos_df.slope.median():.2f} deg, elevation median={pos_df.elevation.median():.1f} m")
print(f"Territory negs : slope median={neg_df.slope.median():.2f} deg, elevation median={neg_df.elevation.median():.1f} m")

print("\n" + "="*72)
print("STEP 5 -- Large candidate pool for terrain-matched negatives")
print("="*72)
cand_gdf = sample_valid_points(150000)
for fname, fpath in FEAT_FILES.items():
    if fname == 'elevation': continue
    coords = [(g.x,g.y) for g in cand_gdf.geometry]
    with rasterio.open(fpath) as src:
        cand_gdf[fname] = np.array([v[0] for v in src.sample(coords)], dtype=float)
print(f"Candidate pool size: {len(cand_gdf)}")

print("\n" + "="*72)
print("STEP 6 -- Stratified matching: resample negatives onto positives' terrain")
print("="*72)
N_BINS = 8
slope_edges = np.quantile(pos_df['slope'], np.linspace(0,1,N_BINS+1)); slope_edges[0]-=1; slope_edges[-1]+=1
elev_edges  = np.quantile(pos_df['elevation'], np.linspace(0,1,N_BINS+1)); elev_edges[0]-=1; elev_edges[-1]+=1

pos_df['bin'] = (pd.cut(pos_df['slope'], slope_edges, labels=False).astype('Int64').astype(str)
                  + '_' + pd.cut(pos_df['elevation'], elev_edges, labels=False).astype('Int64').astype(str))

in_range = (cand_gdf['slope']     > slope_edges[0]) & (cand_gdf['slope']     <= slope_edges[-1]) & \
           (cand_gdf['elevation'] > elev_edges[0])  & (cand_gdf['elevation'] <= elev_edges[-1])
cand_in = cand_gdf[in_range].copy()
cand_in['bin'] = (pd.cut(cand_in['slope'], slope_edges, labels=False).astype('Int64').astype(str)
                   + '_' + pd.cut(cand_in['elevation'], elev_edges, labels=False).astype('Int64').astype(str))
print(f"Candidates within positives' terrain envelope: {len(cand_in)} / {len(cand_gdf)}")

bin_counts = pos_df['bin'].value_counts()

matched_list, shortfall, empty_bins = [], 0, []
for bin_id, count in bin_counts.items():
    pool = cand_in[cand_in['bin']==bin_id]
    if len(pool) == 0:
        empty_bins.append(bin_id)
        continue
    if len(pool) >= count:
        matched_list.append(pool.sample(n=int(count), random_state=42, replace=False))
    else:
        shortfall += 1
        matched_list.append(pool.sample(n=int(count), random_state=42, replace=True))

if not matched_list:
    raise RuntimeError("No bins matched at all -- check slope/elevation units before proceeding.")

matched_neg = pd.concat(matched_list, ignore_index=True)
pos_df = pos_df[~pos_df['bin'].isin(empty_bins)].copy()

print(f"Matched negatives: {len(matched_neg)}  (bins needing replacement: {shortfall}/{len(bin_counts)})")
print(f"Zero-candidate bins: {len(empty_bins)}  ->  {sum(bin_counts[b] for b in empty_bins)} positives excluded (no comparable terrain in the 150k pool)")
print(f"Matched negs   : slope median={matched_neg.slope.median():.2f} deg, elevation median={matched_neg.elevation.median():.1f} m")
print(f"(should now be close to the positives' {pos_df.slope.median():.2f} deg / {pos_df.elevation.median():.1f} m)")

print("\n" + "="*72)
print("STEP 7 -- Retrain + evaluate on the terrain-matched (hard) problem")
print("="*72)
matched_all = pd.concat([pos_df[FEATURES+['label']], matched_neg[FEATURES].assign(label=0)], ignore_index=True)
Xm, ym = matched_all[FEATURES].values, matched_all['label'].values
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.2, random_state=42, stratify=ym)
pipe_m = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
pipe_m.fit(Xm_tr, ym_tr)
auc_matched = roc_auc_score(ym_te, pipe_m.predict_proba(Xm_te)[:,1])
cv_matched = cross_val_score(pipe_m, Xm, ym, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring='roc_auc', n_jobs=-1)
print(f"AUC, terrain-matched, all 4 features: {auc_matched:.4f}  (5-fold CV {cv_matched.mean():.4f} +/- {cv_matched.std():.4f})")
print(f"(Prof's independent number: 0.63)")

perm = permutation_importance(pipe_m, Xm_te, ym_te, n_repeats=10, random_state=42, scoring='roc_auc', n_jobs=-1)
imp_df = pd.DataFrame({'feature':FEATURES,'importance':perm.importances_mean}).sort_values('importance', ascending=False)
print(f"\nPermutation importance, matched problem:\n{imp_df.to_string(index=False)}")

print("\n" + "="*72)
print("STEP 8 -- Drop elevation, re-evaluate on the matched problem")
print("="*72)
FEATS_NOELEV = ['slope','aspect','landcover']
Xm2 = matched_all[FEATS_NOELEV].values
Xm2_tr, Xm2_te, ym_tr2, ym_te2 = train_test_split(Xm2, ym, test_size=0.2, random_state=42, stratify=ym)
pipe_m2 = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
pipe_m2.fit(Xm2_tr, ym_tr2)
auc_noelev = roc_auc_score(ym_te2, pipe_m2.predict_proba(Xm2_te)[:,1])
print(f"AUC, terrain-matched, WITHOUT elevation: {auc_noelev:.4f}  (delta vs 4-feature: {auc_matched-auc_noelev:+.4f})")
print(f"(Prof's independent numbers: 0.627 -> 0.619)")

print("\n" + "="*72)
print("SUMMARY")
print("="*72)
print(f"{'Territory-wide negatives (headline)':42s} {auc_baseline:.4f}")
print(f"{'Spatial-block 5-fold CV':42s} {spatial_aucs.mean():.4f}")
print(f"{'Random 5-fold CV':42s} {random_cv.mean():.4f}")
print(f"{'Terrain-matched negatives, 4 features':42s} {auc_matched:.4f}")
print(f"{'Terrain-matched, elevation dropped':42s} {auc_noelev:.4f}")

matched_all.to_csv("features/matched_terrain_samples.csv", index=False)
print("\nSaved features/matched_terrain_samples.csv for the follow-up profile-curvature test.")
