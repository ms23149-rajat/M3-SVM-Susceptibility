# M-3 — SVM Landslide Susceptibility Mapping

**Status: RUNS** — two-class SVM trained and validated on Hong Kong ENTLI; full territory susceptibility map produced; AUC 0.880 on held-out test set.

**Intern:** Rajat (rajat-surge2026), SURGE/SARIP 2026, Group B
**Supervisor:** Dr. Shyam Nandan
**Workstation:** labws2 (Ubuntu 24.04.4 LTS, Python 3.11, scikit-learn 1.9.0)
**Conda env:** m3_svm

## What this model is

A Support Vector Machine (SVM) binary classifier for landslide susceptibility mapping. Given a set of terrain and environmental predictor rasters, it learns to distinguish landslide-occurrence locations from stable locations, then predicts a failure probability (0–1) at every pixel across the study area. Unlike physically-based models (P-1 TRIGRS, P-7 r.slope.stability), SVM is purely data-driven — it learns the empirical relationship between terrain attributes and observed landslide locations without assuming any specific failure mechanism.

## Source papers

- Yao, X., Tham, L.G., Dai, F.C. (2008). Landslide susceptibility mapping based on support vector machine: A case study on natural slopes of Hong Kong, China. *Geomorphology*, 101(4), 572–582.
- Marjanović, M., Kovačević, M., Bajat, B., Voženilek, V. (2011). Landslide susceptibility assessment using SVM machine learning algorithm. *Engineering Geology*, 123(3), 225–234.

## Validation site: Hong Kong

Data sources:
- **Landslide inventory**: Enhanced Natural Terrain Landslide Inventory (ENTLI, up to year 2022), Geotechnical Engineering Office (GEO), CEDD, HKSAR. Freely available at https://www.ginfo.cedd.gov.hk/geoopendata/eng/ENTLI.aspx. Contains 111,851 crown-point records from 1924–2022. After filtering sentinel values (SLOPE/HEADELEV = 9999), 107,658 clean records remain across four types: R (rotational/planar, 80%), O (open slope, 13%), C (channelized debris flow, 7%), S (sheet failure, <1%).
- **DEM**: SRTM 1-arcsecond (~30m), tiles N22E113 and N22E114, downloaded from AWS S3 public bucket. Merged and reprojected to EPSG:2326 (Hong Kong 1980 Grid).
- **Land cover**: ESA WorldCover 2021 v200 (10m), tiles N21E111 and N21E114. Resampled to 30m to match DEM grid via nearest-neighbour (categorical, must not be interpolated).

## Feature set

Six predictors, following Yao (2008) as closely as available data allows:

| Feature | Source | Notes |
|---|---|---|
| Elevation | SRTM DEM | Derived directly |
| Slope angle | SRTM DEM | Sobel gradient method |
| Slope aspect | SRTM DEM | Sobel gradient method |
| Profile curvature | SRTM DEM | Second derivative along steepest descent |
| TWI | SRTM DEM | D8 flow accumulation; ln(a/tan β) |
| Land cover | ESA WorldCover | Substitutes for Yao's vegetation field |

Yao's seventh predictor (lithology) was not reproduced — the HK 1:20,000 geology download (2nd edition) covered only Sheet 2 and Sheet 5 (two tiles in NW New Territories), insufficient for territory-wide modeling. This is documented as a known limitation.

## Results

### Hong Kong validation

Training: 5,000 landslide points (stratified by type) + 5,000 non-landslide points (sampled outside 250m buffer around all known events, land pixels only). 80/20 train/test split.

| Metric | Value |
|---|---|
| AUC-ROC (test set) | 0.880 |
| AUC-ROC (5-fold CV mean) | 0.874 ± 0.008 |
| rTP / sensitivity | 94.2% |
| rTN / specificity | 72.8% |
| Overall accuracy | 83.4% |

**Feature importance (permutation, AUC-based):**

| Feature | Importance |
|---|---|
| Slope | 0.091 |
| Elevation | 0.087 |
| Land cover | 0.024 |
| TWI | 0.005 |
| Aspect | 0.003 |
| Profile curvature | ~0.000 |

Slope and elevation together account for nearly all discriminatory power, consistent with Yao (2008)'s findings for the same study area. Profile curvature contributes nothing at 30m resolution — consistent with this derivative being noise-dominated at coarser DEM resolutions.

**Comparison with literature:** Marjanović (2011) reported AUC ~0.79–0.82 for SVM on their Serbia (Fruška Gora) dataset. Our HK result (AUC 0.880) exceeds this, though the comparison is indicative rather than definitive given different study areas, feature sets, and training sample sizes.

### Susceptibility map statistics (full HK territory)

| Statistic | Value |
|---|---|
| Valid land pixels | 12,968,656 |
| Min Pf | 0.006 |
| Max Pf | 0.922 |
| Mean Pf | 0.280 |
| % area Pf > 0.5 | 29.7% |
| % area Pf > 0.7 | 24.0% |

## Known limitations

- Lithology predictor absent (geology sheets covered only 2 of ~30 map tiles).
- Profile curvature computed at 30m — too coarse to be informative; included for methodological completeness.
- Non-landslide samples drawn randomly from territory-wide land pixels: genuinely stable vs. unrecorded landslide locations are indistinguishable at this stage.
- No hyperparameter tuning (C, gamma) performed — default values used; a grid search would likely improve AUC further.
- ENTLI CLASS field (geotechnical material) not used in final model due to stratification artifact discovered during exploration; see notes.md.
