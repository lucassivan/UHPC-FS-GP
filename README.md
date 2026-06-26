# UHPC Flexural Strength -- Probabilistic GP Prediction Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange.svg)]()

A probabilistic machine learning framework for predicting **UHPC flexural strength (FS)**, combining Gaussian process regression with automatic relevance determination (GP-ARD-RBF), Boruta feature selection, calibrated uncertainty quantification, and a Jupyter-based interactive prediction dashboard.

---

## Overview

This repository contains the dataset and interactive notebook accompanying the paper:

> **"Probabilistic Flexural Strength Prediction of Ultra-High Performance Concrete Using Gaussian Process Regression with Calibrated Uncertainty and Reliability Assessment"**

Flexural failure in UHPC is driven by fibre bridging mechanisms that compressive-strength models do not capture, and no existing machine learning study provides probabilistic flexural predictions with verified calibration. This framework trains a GP-ARD-RBF model on 321 UHPC beam specimens, delivering a full predictive distribution at each query point rather than a single estimate.

### Key Features

- **GP-ARD-RBF Model**: per-feature length scales across 14 Boruta-selected inputs resolve the competing influence of fibre, binder, and aggregate variables
- **9 Engineered Features**: physics-informed composites including Fiber Reinforcement Index (Fiber\_RI), Steel Fibre Efficiency, and pozzolan-corrected W/B ratio (W\_Beff)
- **Calibrated Prediction Intervals**: GP-ARD-RBF achieves PICP95 = 95.4% at the nominal 95% level; Bootstrap-CatBoost reaches only 75.4% on the same data
- **Ablation Study (A to E)**: five progressive feature-set configurations isolate the ECE cost of collinearity and the benefit of Boruta selection (ECE reduced from 0.131 to 0.062)
- **GP-SHAP with Uncertainty Propagation**: each Shapley value carries a GP posterior standard deviation; SSV leads at 3.23 MPa attribution
- **2D ALE Interactions**: SSV x D and SSV x Age surfaces locate the high-performance design region at SSV 1.5--2.5% with D 0.17--0.19 mm
- **Reliability Index Maps**: beta = Phi^-1(P(FS > d)) computed analytically from GP posteriors, linked to fib Model Code 2020 target (beta = 3.8)
- **External Validation**: 333 independent specimens from two separate laboratories, R2 = 0.827, PICP95 = 93.9%, Roy slope parameters passed

### Model Performance (GP-ARD-RBF, 80/20 held-out test set)

| Metric | Value |
|--------|-------|
| R2 (test) | 0.825 |
| RMSE (test) | 3.96 MPa |
| PICP95 (test) | 95.4% |
| ECE (test) | 0.062 |
| 30-seed CV R2 | 0.718 +/- 0.082 |
| 30-seed CV PICP95 | 90.9% +/- 4.2% |
| Bootstrap-CatBoost PICP95 | 75.4% (same data) |

---

## Repository Contents

This repository provides the **dataset** and **interactive notebook** to support reproducibility and practical use. The full analysis pipeline is described below for transparency.

```
UHPC-FS-GP/
├── README.md
├── uhpc_flexural_strength_1.csv   # 321 UHPC beam specimen records
└── UHPC-FS-GUI.ipynb              # Probabilistic prediction dashboard (ipywidgets)
```

> **Note:** The complete analysis codebase will be made publicly available upon paper acceptance. The dataset and notebook are provided to support reproducibility and practical use. Analysis scripts are available upon reasonable request.

---

## Dataset

**File:** `uhpc_flexural_strength_1.csv`

The dataset comprises **321 UHPC beam specimens** compiled from published experimental literature, covering a wide range of mix designs, fibre types, and curing protocols.

| Property | Value |
|----------|-------|
| Records | 321 |
| FS range | 7.78 -- 55.51 MPa |
| FS mean | 27.50 MPa (SD 9.89 MPa, CV 35.9%) |

### Raw Input Features (17)

| Category | Symbol | Description | Unit | Sparsity (%) |
|----------|--------|-------------|------|--------------|
| Binder | C | Cement content | kg/m3 | 0 |
| Binder | SF | Silica fume | kg/m3 | 5.9 |
| Binder | QP | Quartz powder | kg/m3 | 90.0 |
| Binder | FA | Fly ash | kg/m3 | 72.0 |
| Binder | SL | Slag | kg/m3 | 63.2 |
| Binder | MK | Metakaolin | kg/m3 | 85.0 |
| Aggregate | S | Sand | kg/m3 | 19.6 |
| Aggregate | QS | Quartz sand | kg/m3 | 80.4 |
| Mix | W | Water | kg/m3 | 0 |
| Mix | SP | Superplasticizer | kg/m3 | 0 |
| Curing | Age | Curing age | days | 0 |
| Fibre geom. | L | Fibre length | mm | 15.9 |
| Fibre geom. | D | Fibre diameter | mm | 15.9 |
| Fibre | BV | Basalt fibre | vol.% | 80.1 |
| Fibre | PPV | Polypropylene fibre | vol.% | 93.8 |
| Fibre | GV | Glass fibre | vol.% | 89.7 |
| Fibre | SSV | Steel fibre | vol.% | 52.3 |

### Engineered Features (9, auto-computed in notebook)

| Symbol | Formula | Physical Rationale |
|--------|---------|-------------------|
| W\_C\_ratio | W / C | Hydration water control |
| Total\_Binder | C + SF + FA + SL + MK | Cumulative reactive powder |
| W\_Binder\_ratio | W / Total\_Binder | Effective porosity indicator |
| SF\_C\_ratio | SF / C | Pozzolanic reaction intensity |
| Total\_Fiber | BV + PPV + GV + SSV | Aggregate reinforcement contribution |
| Fiber\_Aspect | L / D | Geometric bridging efficiency |
| Fiber\_RI | Total\_Fiber x (L/D) | Combined volume-geometry effect on crack bridging |
| Steel\_Efficiency | SSV x (L/D) | Steel fibre contribution to post-cracking strength |
| W\_Beff | W / (C + 0.5SF + 0.3FA + 0.4SL + 0.6MK) | Pozzolan-corrected effective water-binder ratio |

### Boruta Selection Result

Applied to 26 variables (17 raw + 9 engineered), Boruta retained **14 features** (11 confirmed + 3 tentative): SF, S, QS, W, SP, Age, D, SSV, SF\_C\_ratio, Fiber\_RI, Steel\_Efficiency, SL, W\_C\_ratio, W\_Beff. Nine raw variables (C, QP, FA, MK, L, BV, PPV, GV) and three engineered variables (Total\_Binder, W\_Binder\_ratio, Total\_Fiber, Fiber\_Aspect) were rejected as uninformative.

---

## Notebook

**File:** `UHPC-FS-GUI.ipynb`

A Jupyter notebook implementing the full GP-ARD-RBF pipeline and an interactive prediction dashboard using `ipywidgets`. No pre-trained model file is required; the notebook trains the GP from the dataset at startup.

### Installation

```bash
pip install boruta ipywidgets scikit-learn pandas numpy matplotlib scipy
```

### Usage

Place `UHPC-FS-GUI.ipynb` and `uhpc_flexural_strength_1.csv` in the same folder, then open the notebook and run all cells from top to bottom. Training (5-fold CV + final GP fit) takes approximately 3--6 minutes.

### Notebook Structure

**Step 1 -- Train the GP model** (cells 1--11)

Runs the full pipeline automatically:
1. Load `uhpc_flexural_strength_1.csv` and compute the 9 engineered features
2. Split 80/20 (seed = 42), run Boruta on the training partition, retain 14 features
3. 5-fold cross-validation with GP-ARD-RBF (5 restarts per fold)
4. Train the final GP on the full 256-sample training set
5. Evaluate on the 65-sample held-out test set

A model performance badge is displayed above the input panels showing CV and test-set metrics.

**Step 2 -- Interactive dashboard** (cells 12--13)

Launches a four-panel prediction interface:

- **Prediction card**: GP posterior mean +/- std and 95% CI for the submitted mix
- **Probability gauge**: P(FS > threshold) for a user-selected design threshold (5--50 MPa slider)
- **Beta-index table**: exceedance probability and reliability index at six structural thresholds (10, 15, 20, 25, 30, 35 MPa)
- **Reliability curve**: continuous P(FS > d) sweep with 95% and 50% reference lines and a marker at the selected threshold

Input fields are organized in three panels:

| Panel | Variables |
|-------|-----------|
| Cementitious | C, SF, QP, FA, SL, MK (kg/m3) |
| Aggregates & Water | S, QS, W, SP (kg/m3), Age (days) |
| Fibre Parameters | L (mm), D (mm), BV, PPV, GV, SSV (vol.%) |

The nine engineered features are computed internally from the 17 raw inputs. An out-of-range check is not implemented in the current version; users are advised to keep inputs within the training ranges listed in the dataset table above.

**Step 3 -- Scripted prediction** (cells 14--15)

Direct function call for batch use without the GUI:

```python
sample = dict(C=950, SF=180, QP=120, FA=0, SL=0, MK=0,
              S=980, QS=0, W=170, SP=28, Age=28,
              L=13.0, D=0.20, BV=0.0, PPV=0.0, GV=0.0, SSV=2.0)

mu, sigma = predict_fs(sample)
# Predicted FS : 37.80 +/- 10.89 MPa
# 95% CI       : [16.4,  59.1] MPa
```

---

## Full Analysis Pipeline

The complete framework comprises the following scripts (available upon request). Scripts in the figure group read the CSV outputs from `uhpc_flexural_gp.py` and do not retrain any model.

### `uhpc_flexural_gp.py` -- Core Analysis

The main entry point. Run this script first; all figure scripts depend on its CSV outputs.

**Steps executed in order:**

1. Load `uhpc_flexural_strength_1.csv` (321 samples, 17 raw features) and compute 9 engineered features; total pool = 26 variables
2. 80/20 stratified split (seed = 42): 256 training, 65 test samples; StandardScaler fitted on training partition only
3. Boruta feature selection on scaled training set (RandomForest, 100 iterations, n\_estimators = auto); confirmed + tentative features retained
4. Define 8 GP kernel configurations (GP\_RBF, GP\_Matern15, GP\_Matern25, GP\_RQ, GP\_RBF\_Matern, GP\_RBF\_Linear, GP\_ARD\_RBF, GP\_Physics) and 7 ML baselines (CatBoost, XGBoost, LightGBM, GradientBoosting, RandomForest, SVR, MLP); all trained on the Boruta-selected feature set
5. Single-split evaluation of all 15 models; GP variants return PICP95, PICP90, MPIW, NLL, ECE alongside R2/RMSE/MAE/MAPE
6. Five-level progressive ablation (A: 17 raw; B: +6 general engineered; C: +Fiber\_RI and Steel\_Efficiency; D: +W\_Beff; E: Boruta-filtered); GP-ARD-RBF only, same 80/20 split
7. ARD inverse length-scale extraction from the fitted GP-ARD-RBF kernel: 1/l\_j for each of the 14 retained features, averaged over 5 restarts
8. GP-SHAP via KernelExplainer (50-sample background, nsamples = 100 per test point); per-feature GP uncertainty component: sigma\_SHAP\_j = sigma\_GP x |phi\_j| / sum\_k|phi\_k|
9. 1D ALE for 6 leading features and 2D ALE for SSV x D and SSV x Age (custom numpy implementation, no external ALE library); GP posterior standard deviation propagated through accumulation
10. Reliability index beta = Phi^-1(P(FS > d)) for five design thresholds on a representative test specimen; computed analytically from GP posterior mean and std

**Outputs:**

| File | Contents |
|------|----------|
| `fs_boruta_results.csv` | Feature name, Boruta decision (Confirmed/Tentative/Rejected), Z-score, RF importance |
| `fs_gp_all_results.csv` | All 15 models: R2, RMSE, MAE, MAPE, PICP95, PICP90, MPIW, NLL, ECE (single split) |
| `fs_gp_predictions.csv` | GP-ARD-RBF test-set predictions: y\_true, y\_pred (mu), y\_std (sigma), standardised residual |
| `fs_gp_ablation.csv` | Ablation levels A-E: feature count, R2, RMSE, PICP95, MPIW, ECE |
| `fs_kernel_comparison.csv` | All 8 GP kernels: same metrics as fs\_gp\_all\_results |
| `fs_gp_reliability.csv` | Design threshold, P(FS > d), beta index for 5 thresholds |
| `fs_ale_1d_results.csv` | 1D ALE: feature, grid value, ALE effect, GP sigma band |
| `fs_ale_2d_ssv_d.csv` | 2D ALE grid: SSV x D interaction values |
| `fs_ale_2d_ssv_age.csv` | 2D ALE grid: SSV x Age interaction values |
| `fs_shap_importance.csv` | Per-feature mean absolute SHAP, cross-sample SD, GP sigma component |

---

### `run_cv30.py` -- 30-Seed Cross-Validation

Standalone script; run after `uhpc_flexural_gp.py` has generated `fs_boruta_results.csv`.

Loads the same data and applies identical feature engineering. Reads Boruta results from cache (or falls back to the hard-coded 17-feature list if the cache is absent). For each of 30 random seeds, a 5-fold stratified split is constructed; GP-ARD-RBF and all 7 ML baselines are trained and evaluated on each fold (150 evaluations per model in total). GP fits use 3 optimizer restarts per fold to keep runtime under 35 minutes on a single CPU. Wilcoxon signed-rank tests (two-sided) compare the 150-value R2 distributions of GP-ARD-RBF against each ML baseline; the test statistic and p-value are reported in the summary CSV. GP-ARD-RBF is the only model for which PICP95 and ECE are computed, since deterministic baselines produce no predictive intervals.

**Inputs:** `uhpc_flexural_strength_1.csv`, `fs_boruta_results.csv`

**Outputs:**

| File | Contents |
|------|----------|
| `fs_gp_cv_raw_30.csv` | One row per (seed, fold, model): R2, RMSE, MAE, MAPE; PICP95 and ECE for GP-ARD-RBF |
| `fs_gp_cv_summary_30.csv` | Mean +/- SD across 150 evaluations per model; Wilcoxon p-value vs GP-ARD-RBF |

---

### `uhpc_fs_visualization_v2.py` -- Fig. 1--3 (EDA)

Reads the raw dataset only; no model outputs required. Generates three publication figures covering the full distributional and structural characterization of the 321-specimen dataset.

- **Fig. 1**: curing-age stratified FS bar chart (5 age groups), FS histogram with fitted normal overlay (skewness = 0.337), Pearson correlation lollipop (top positive/negative predictors), fibre-type ridge plot (steel 31.3 MPa, glass 20.3 MPa), feature sparsity chart
- **Fig. 2**: grouped bubble chart of FS by fibre type and curing age, hexbin density of SSV vs FS (r = +0.36), fibre-type pie chart, fibre-reinforcement-index confidence ellipses
- **Fig. 3**: 2D KDE contour (W/B vs FS), triangular Pearson matrix, parallel coordinate profiles by FS quartile, Q-Q normality check (KS p = confirmed near-Gaussian)

**Inputs:** `uhpc_flexural_strength_1.csv`

**Outputs:** `FS_Figure_x_Distribution_Raincloud_v2.png/svg`, `FS_Figure_y_Fibre_Characterisation_v2.png/svg`, `FS_Figure_z_Statistical_Summary_v2.png/svg`

---

### `fig2_boruta_v4.py` -- Fig. 4 (Boruta Feature Selection)

Reads Boruta results from cache; re-runs Boruta on the training partition if the cache is absent. Three-panel figure with a continuous RdYlGn Z-score bar chart, a 5x6 waffle grid of all 26 features coloured by Confirmed/Tentative/Rejected status, and an RF importance lollipop with shadow-max and shadow-mean reference lines.

**Inputs:** `uhpc_flexural_strength_1.csv`, `fs_boruta_results.csv`

**Outputs:** `fig2_boruta_v4.png/svg`

---

### `fig4_gp_diagnostics_v2.py` -- Fig. 5 (GP Posterior Diagnostics)

Reads pre-computed GP predictions and the full results table. Four-panel diagnostic figure: posterior sigma violin + strip (median, P5-P95 annotated), standardised residual Q-Q with 95% simulation envelope and KS p-value, ARD inverse length-scale horizontal bar chart coloured by feature category (fibre vs composition), RMSE-PICP95 scatter for all 8 kernels with GP-ARD-RBF highlighted as the Pareto-optimal point.

**Inputs:** `fs_gp_predictions.csv`, `fs_gp_all_results.csv`

**Outputs:** `fig4_gp_diagnostics_v3.png/svg`

---

### `fig8_qq_calibration.py` -- Fig. 6 (Kernel Calibration Comparison)

Re-trains all 8 GP kernels on the same 80/20 split (no cached inputs needed beyond the dataset). Three-panel calibration figure: residual Q-Q overlaid for all kernels, accuracy-calibration scatter (RMSE on x, ECE on y) showing the isotropic cluster vs GP-ARD-RBF Pareto position, calibration curves at 19 nominal confidence levels with the perfect-calibration diagonal.

**Inputs:** `uhpc_flexural_strength_1.csv`, `fs_boruta_results.csv`

**Outputs:** `fig8_qq_calibration.png/svg`

---

### `fig5_bootstrap_comparison.py` -- Fig. 7 (Bootstrap vs GP Calibration)

Trains GP-ARD-RBF and a Bootstrap ensemble of 200 CatBoost models on the same 256-sample training set. Three-panel figure: reliability diagrams (nominal vs observed coverage for both methods), coverage deviation curve across the full confidence spectrum (GP tracks zero deviation; Bootstrap deviates by up to -20 pp), and a normalized bar comparison of PICP95, ECE, MPIW, and NLL.

**Inputs:** `uhpc_flexural_strength_1.csv`, `fs_boruta_results.csv`

**Outputs:** `fig5_bootstrap_comparison.png/svg`, `fs_bootstrap_comparison.csv`

---

### `fig7_gp_shap.py` -- Fig. 8 (GP-SHAP with Uncertainty Propagation)

Loads Boruta-selected features, re-trains GP-ARD-RBF on the full training set, computes KernelExplainer SHAP values on the test set, and reads 1D ALE results for the Spearman consistency panel. Four-panel figure: mean |phi| bar chart with GP-uncertainty error bars, custom beeswarm where dot transparency encodes posterior sigma, Spearman rank-consistency scatter between SHAP and ALE orderings, SSV dependence curve with 95% GP confidence bands.

**Inputs:** `uhpc_flexural_strength_1.csv`, `fs_boruta_results.csv`, `fs_ale_1d_results.csv`

**Outputs:** `fig7_gp_shap.png/svg`, `fs_shap_importance.csv`

---

### `fig6_ale.py` -- Fig. 9 (GP-ALE Interaction Surfaces)

Custom ALE implementation (pure numpy, no external library). Re-trains GP-ARD-RBF once and computes 1D ALE for six features and two 2D ALE grids. Three main panels: SSV x D interaction contour (positive synergy +0.27 MPa at high SSV and thin fibres), SSV x Age interaction contour (early-age synergy reaching +4.53 MPa at SSV = 5% and Age = 5 days), GP posterior mean surface in SSV-D space identifying the high-performance region.

**Inputs:** `uhpc_flexural_strength_1.csv`, `fs_boruta_results.csv`

**Outputs:** `fig6_ale_fiber.png/svg` (2D panels), `fig8_ale_1d_panel.png/svg` (1D panel), `fs_ale_1d_results.csv`, `fs_ale_2d_ssv_d.csv`, `fs_ale_2d_ssv_age.csv`

---

### `fig9_learning_curve_v2.py` -- Fig. 10 (Learning Curve)

Plot-only script; all computation is in `uhpc_flexural_gp.py`. Reads the pre-computed learning curve CSV and generates a dual-axis figure: R2 trajectories for GP-ARD-RBF, CatBoost, XGBoost, and RandomForest on the left axis; PICP95 and ECE for GP-ARD-RBF on the right axis. Annotates the calibration-failure zone (n < 50) where PICP95 drops to 57.5% and ECE reaches 0.234.

**Inputs:** `fs_learning_curve.csv`

**Outputs:** `fig9_learning_curve_v2.png/svg`

---

### `fig10_noise_robustness_v2.py` -- Fig. 11 (Noise Robustness)

Plot-only script. Reads the pre-computed noise robustness CSV and plots GP-ARD-RBF and CatBoost R2, PICP95, and ECE across six Gaussian label noise levels (eta = 0, 0.02, 0.05, 0.10, 0.15, 0.20 of sigma\_y). Annotates the WhiteKernel adaptive mechanism: PICP95 stays at 95.4% through eta = 0.15 because the optimised noise\_level absorbs injected variance without inflating the posterior mean error.

**Inputs:** `fs_noise_robustness.csv`

**Outputs:** `fig10_noise_robustness_v2.png/svg`

---

### `gp_external_validation.py` -- Fig. 12--14 (External Validation)

Merges the 321-specimen training corpus (Dataset A) with 333 independently compiled specimens from two laboratories (Dataset B). Aligns 15 common features, converts steel fibre mass to volume fraction using density 7850 kg/m3, runs Boruta on the merged 80% training split (523 samples), and trains GP-ARD-RBF on the 11 retained features. Evaluates on 131 held-out test samples. Computes Roy parameters (k, k', R2, Ro2, R'o2, Rm, |m|, |n|) and generates a six-panel validation figure, a Bland-Altman agreement plot, and a dataset-comparison summary.

**Inputs:** `uhpc_flexural_strength_1.csv`, `validation.xlsx` (Dataset B)

**Outputs:** `gp_ext_predictions.csv`, `gp_ext_metrics_summary.csv`, `gp_ext_validation_parameters.csv`, `gp_ext_boruta_results.csv`, `gp_ext_cv_results.csv`, `fig_gp_ext_comprehensive.png/svg`, `fig_gp_ext_bland_altman.png/svg`, `fig_gp_ext_dataset_comparison.png/svg`

---

## Results Summary

### Single-Split Test Performance (65 held-out specimens)

| Model | R2 | RMSE (MPa) | PICP95 (%) | ECE |
|-------|----|------------|------------|-----|
| CatBoost | 0.908 | 2.88 | -- | -- |
| XGBoost | 0.904 | 2.93 | -- | -- |
| RandomForest | 0.896 | 3.06 | -- | -- |
| **GP-ARD-RBF** | **0.825** | **3.96** | **95.4** | **0.062** |
| Bootstrap-CatBoost | 0.897 | 3.05 | 75.4 | 0.192 |

### Cross-Validation (30 seeds x 5 folds, 150 evaluations)

| Model | R2 (mean +/- SD) | PICP95 (%) | ECE |
|-------|-----------------|------------|-----|
| CatBoost | 0.854 +/- 0.044 | -- | -- |
| XGBoost | 0.817 +/- 0.062 | -- | -- |
| **GP-ARD-RBF** | **0.718 +/- 0.082** | **90.9 +/- 4.2** | **0.049 +/- 0.020** |
| MLP | 0.396 +/- 0.448 | -- | -- |

### External Validation (333 specimens, 2 independent laboratories)

| Metric | Combined | Dataset A | Dataset B |
|--------|----------|-----------|-----------|
| R2 | 0.827 | 0.826 | 0.791 |
| RMSE (MPa) | 3.91 | 4.13 | 3.68 |
| PICP95 (%) | 93.9 | 92.2 | 95.5 |
| ECE | 0.075 | 0.019 | 0.131 |
| Roy k | 0.976 (Pass) | -- | -- |
| Roy k' | 1.006 (Pass) | -- | -- |

### Key Interpretability Findings

- **ARD length scales**: SSV and D carry the shortest per-feature length scales, indicating the steepest local sensitivity in the GP covariance structure
- **GP-SHAP mean attribution**: SSV 3.23 MPa (GP sigma component 0.873 MPa), Age 2.82 MPa (cross-sample SD 3.04 MPa), W/C ratio 2.51 MPa
- **2D ALE (SSV x D)**: thin fibres (D = 0.17--0.19 mm) generate a positive interaction of +0.25 MPa at high SSV; the interaction magnitude is small relative to the SSV main-effect range of 12.6 MPa
- **2D ALE (SSV x Age)**: SSV = 5.0% combined with early curing (Age = 5 days) reaches +4.53 MPa, indicating disproportionate fibre crack-bridging before matrix maturation

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
boruta>=0.3
scipy>=1.7.0
matplotlib>=3.5.0
ipywidgets>=8.0.0
```

---

## Code Availability

> **The complete analysis codebase will be made publicly available upon paper acceptance.**

### Currently Available in This Repository

| File | Description |
|------|-------------|
| `uhpc_flexural_strength_1.csv` | 321 UHPC beam specimen records (17 features + FS target) |
| `UHPC-FS-GUI.ipynb` | Interactive probabilistic prediction dashboard |

### Available Upon Request (Full Release After Acceptance)

| Script | Description |
|--------|-------------|
| `uhpc_flexural_gp.py` | Core analysis: GP/ML training, ablation, SHAP, ALE, reliability |
| `run_cv30.py` | 30-seed x 5-fold cross-validation with Wilcoxon tests |
| `uhpc_fs_visualization_v2.py` | EDA figures (Fig. 1--3) |
| `fig2_boruta_v4.py` | Boruta feature selection figure (Fig. 4) |
| `fig4_gp_diagnostics_v2.py` | GP diagnostic figure (Fig. 5) |
| `fig8_qq_calibration.py` | Kernel calibration comparison (Fig. 6) |
| `fig5_bootstrap_comparison.py` | Bootstrap vs GP comparison (Fig. 7) |
| `fig7_gp_shap.py` | GP-SHAP with uncertainty propagation (Fig. 8) |
| `fig6_ale.py` | GP-ALE interaction surfaces (Fig. 9) |
| `fig9_learning_curve_v2.py` | Learning curve analysis (Fig. 10) |
| `fig10_noise_robustness_v2.py` | Noise robustness analysis (Fig. 11) |
| `gp_external_validation.py` | External validation across two laboratories (Fig. 12--14) |

To request access before the full release, please contact us with your institutional affiliation, research purpose, and the specific scripts needed. We aim to respond within 5 business days.

---

## Experimental Environment

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 |
| Python | 3.10+ |
| scikit-learn | 1.6.1 |
| SHAP | 0.46.0 |

---

## Citation

If you find this work useful, please cite our paper (details will be updated upon publication):

```bibtex
@article{uhpcflexural2026,
  title   = {Probabilistic Flexural Strength Prediction of Ultra-High
             Performance Concrete Using Gaussian Process Regression with
             Calibrated Uncertainty and Reliability Assessment},
  author  = {},
  journal = {},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, dataset requests, or collaboration:

**Email**: lucassivan@163.com

---

## Acknowledgements

We thank the authors of the original UHPC experimental studies whose published data were compiled to form the dataset used in this work. The dataset is compiled from published peer-reviewed literature and is used here solely for academic research purposes.
