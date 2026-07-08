import numpy as np, pandas as pd, geopandas as gpd, rasterio
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
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
print("SETUP -- rebuilding positives / territory negatives / candidate pool")
print("="*72)

inv = gpd.read_file(os.path.expanduser("~/landslide-toolkit/kerala_inventory/Kerela landslide.shp")).to_crs("EPSG:32643")
with rasterio.open(FEAT_FILES['elevation']) as dem_src:
    bounds = dem_src.bounds
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

all_pts = gpd.GeoDataFrame(
    pd.concat([positives[['geometry','label']].reset_index(drop=True),
               negatives_gdf[['geometry','label']].reset_index(drop=True)], ignore_index=True),
    geometry='geometry', crs='EPSG:32643')

for fname, fpath in FEAT_FILES.items():
    coords = [(g.x, g.y) for g in all_pts.geometry]
    with rasterio.open(fpath) as src:
        all_pts[fname] = np.array([v[0] for v in src.sample(coords)], dtype=float)

pos_df_base = all_pts[all_pts.label==1].copy()

cand_gdf = sample_valid_points(150000)
for fname, fpath in FEAT_FILES.items():
    if fname == 'elevation': continue
    coords = [(g.x,g.y) for g in cand_gdf.geometry]
    with rasterio.open(fpath) as src:
        cand_gdf[fname] = np.array([v[0] for v in src.sample(coords)], dtype=float)

print(f"Positives: {len(pos_df_base)}   Candidate pool: {len(cand_gdf)}")

def match_and_eval(n_bins):
    pos_df = pos_df_base.copy()
    slope_edges = np.quantile(pos_df['slope'], np.linspace(0,1,n_bins+1)); slope_edges[0]-=1; slope_edges[-1]+=1
    elev_edges  = np.quantile(pos_df['elevation'], np.linspace(0,1,n_bins+1)); elev_edges[0]-=1; elev_edges[-1]+=1

    pos_df['bin'] = (pd.cut(pos_df['slope'], slope_edges, labels=False).astype('Int64').astype(str)
                      + '_' + pd.cut(pos_df['elevation'], elev_edges, labels=False).astype('Int64').astype(str))

    in_range = (cand_gdf['slope']     > slope_edges[0]) & (cand_gdf['slope']     <= slope_edges[-1]) & \
               (cand_gdf['elevation'] > elev_edges[0])  & (cand_gdf['elevation'] <= elev_edges[-1])
    cand_in = cand_gdf[in_range].copy()
    cand_in['bin'] = (pd.cut(cand_in['slope'], slope_edges, labels=False).astype('Int64').astype(str)
                       + '_' + pd.cut(cand_in['elevation'], elev_edges, labels=False).astype('Int64').astype(str))

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
        return dict(n_bins=n_bins, occupied_bins=len(bin_counts), n_matched=0, n_excluded=len(pos_df),
                    shortfall_bins=0, auc=float('nan'), cv_auc=float('nan'), cv_std=float('nan'))

    matched_neg = pd.concat(matched_list, ignore_index=True)
    pos_df_m = pos_df[~pos_df['bin'].isin(empty_bins)].copy()

    matched_all = pd.concat([pos_df_m[FEATURES+['label']], matched_neg[FEATURES].assign(label=0)], ignore_index=True)
    Xm, ym = matched_all[FEATURES].values, matched_all['label'].values
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.2, random_state=42, stratify=ym)
    pipe_m = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
    pipe_m.fit(Xm_tr, ym_tr)
    auc = roc_auc_score(ym_te, pipe_m.predict_proba(Xm_te)[:,1])
    cv = cross_val_score(pipe_m, Xm, ym, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)

    n_excluded = sum(bin_counts[b] for b in empty_bins)
    return dict(n_bins=n_bins, occupied_bins=len(bin_counts), n_matched=len(matched_neg), n_excluded=n_excluded,
                shortfall_bins=shortfall, auc=auc, cv_auc=cv.mean(), cv_std=cv.std())

print("\n" + "="*72)
print("N_BINS SWEEP -- does AUC trend down as the terrain match gets tighter?")
print("="*72)
results = []
for n_bins in [4, 8, 16, 24, 32]:
    r = match_and_eval(n_bins)
    results.append(r)
    print(f"N_BINS={r['n_bins']:3d}  bins_occupied={r['occupied_bins']:4d}  matched_neg={r['n_matched']:5d}  "
          f"excluded_pos={r['n_excluded']:4d}  shortfall_bins={r['shortfall_bins']:4d}  "
          f"AUC={r['auc']:.4f}  CV_AUC={r['cv_auc']:.4f} +/- {r['cv_std']:.4f}")

res_df = pd.DataFrame(results)
res_df.to_csv("nbins_sweep_results.csv", index=False)
print("\nSaved nbins_sweep_results.csv")
print("(Reference: N_BINS=8 single run gave AUC 0.6782 / CV 0.6676 +/- 0.0115; Prof's independent number: 0.63)")
