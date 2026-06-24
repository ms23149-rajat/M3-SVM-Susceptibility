# Notes — M-3 (SVM Susceptibility Mapping)

## Environment

- Workstation: labws2, IP 172.26.248.92
- Python 3.11, scikit-learn 1.9.0, geopandas 1.1.3, rasterio 1.4.4
- Conda env: m3_svm (separate from p7_rslopestability used for GRASS-based models)

## Methodology

Following the required loop: read both source papers first, then implement the methodology from scratch using sklearn.svm.SVC rather than sourcing separate author code (no standalone code release exists for either paper — both describe a standard SVM application, so reimplementation from the methods section is the correct approach here, unlike P-1 and P-7 where authors' own compiled code was available).

## Data issues encountered and resolved

### 1. ENTLI CLASS field caused stratification collapse

Initial sampling plan filtered on CLASS != '-' to use geotechnical material class as a lithology proxy for negative samples. This silently removed all non-R slide types: types O, C, and S either had missing CLASS values or sentinel SLOPE/HEADELEV values, so after both filters only SLIDE_TYPE=R remained. The stratified subsample therefore produced 5000/5000 type-R points.

Fix: dropped CLASS from the feature set entirely. CLASS is an attribute of recorded landslide events, not a spatial raster — it cannot be sampled at arbitrary negative-sample locations anyway. Used ESA WorldCover as a consistent spatial raster for land cover instead, sampling it identically at both positive and negative locations.

### 2. Geology coverage insufficient for territory-wide model

The HK 1:20,000 geological map (2nd edition) download contained only Sheet 2 and Sheet 5 — two tiles covering a small area of NW New Territories (roughly E 805,000-830,000, N 825,000-843,000 in EPSG:2326). The full ENTLI spans E 801,000-860,000, N 801,000-847,000. Merging a partial geology layer would have silently assigned NoData to ~90% of the training points.

Fix: lithology dropped from the feature set, documented as a known limitation. The 1:100,000 territory-wide geology package was not downloaded in time; this is the natural next step for improving the model.

### 3. WorldCover raster grid mismatch

WorldCover was reprojected independently of the terrain rasters, producing a larger grid (11,196 x 20,803) covering a wider area than the DEM extent (3,709 x 6,886). np.stack() failed when trying to combine them.

Fix: added a second reprojection step using rasterio.warp.reproject() with the DEM's exact transform and dimensions as the destination grid, producing worldcover_matched.tif at exactly 3,709 x 6,886 pixels.

### 4. Negative sample quality

Non-landslide points were sampled randomly from land pixels (elevation > 0) outside a 250m buffer around all 107,658 clean ENTLI records. This follows Yao (2008)'s methodology directly. Known limitation: absence from the ENTLI does not guarantee genuine stability — unrecorded events, pre-inventory failures, and susceptible-but-not-yet-failed terrain all appear as stable in this scheme. This is a standard caveat in inventory-based susceptibility modeling.

## Feature engineering notes

All five terrain parameters derived from SRTM 30m DEM using numpy gradient operations:
- Slope/aspect: Sobel-based first derivatives
- Profile curvature: second derivative along steepest descent direction
- TWI: D8 single-flow-direction accumulation, then ln(upslope_area / tan(slope))

Profile curvature showed near-zero permutation importance (0.000012) in the final model — expected at 30m resolution where the second derivative is dominated by DEM noise. Retained for methodological completeness and comparison with Yao, but should be replaced by a higher-resolution source in any future iteration.

## Model choices and honest caveats

- Kernel: RBF (standard default for non-linear tabular data, consistent with both papers)
- C=1.0, gamma='scale': default values, no hyperparameter search performed
- StandardScaler applied before SVM: mandatory for distance-based kernels
- probability=True enables Platt scaling for calibrated probabilities
- Training sample: 5,000 positive + 5,000 negative = 10,000 total, trained in under 2 minutes
- No hyperparameter tuning performed; grid search over C and gamma is the most obvious next step

## AI assistance

Claude was used throughout as an interactive mentor: explaining both source papers in plain language before any coding began, walking through each feature engineering and modeling step, diagnosing and fixing the three data issues above, and helping interpret results. All code was run by hand on labws2; Claude has no direct access to the lab machine and never executed anything on it.
