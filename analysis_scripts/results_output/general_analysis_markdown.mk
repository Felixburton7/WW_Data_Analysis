# Analysis Report: Protein Flexibility Prediction Models

## 1. Basic Information

```json
{
    "Source Data File": "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv",
    "Analysis Timestamp": "2025-04-17 10:21:01",
    "Total Rows": 287865,
    "Total Columns": 29,
    "Unique Domains": 406,
    "Memory Usage": "139.23 MB"
}
```

### 1.1 Model Key

This table maps the report names to the original keys and the relevant prediction/uncertainty columns in the dataset.

| Report Name         | Original Key   | Prediction Column      | Uncertainty Column               |
|:--------------------|:---------------|:-----------------------|:---------------------------------|
| DeepFlex            | DeepFlex       | `DeepFlex_rmsf`          | `DeepFlex_rmsf_uncertainty`        |
| ESM-Only (Seq+Temp) | ESM_Only       | `ESM_Only_rmsf`          | *(N/A)*                          |
| VoxelFlex-3D        | Voxel          | `voxel_rmsf`             | *(N/A)*                          |
| LGBM (All Features) | LGBM           | `ensembleflex_LGBM_rmsf` | *(N/A)*                          |
| RF (All Features)   | RF             | `ensembleflex_RF_rmsf`   | `ensembleflex_RF_rmsf_uncertainty` |
| RF (No ESM Feats)   | No_ESM_RF      | `No_ESM_RF_prediction`   | *(N/A)*                          |

### 1.2 Primary Model

*   Primary Model for focused analysis: **DeepFlex**

---

## 2. Missing Value Summary

Summary of missing values for key prediction columns.

| Column                    | Missing Count | Missing Percentage |
|:--------------------------|--------------:|-------------------:|
| `instance_key`            |      2360.00 |               0.82 |
| `DeepFlex_rmsf`           |      2360.00 |               0.82 |
| `DeepFlex_rmsf_uncertainty` |      2360.00 |               0.82 |
| `voxel_rmsf`              |      1975.00 |               0.69 |
| `ESM_Only_rmsf`           |      1580.00 |               0.55 |

---

## 3. Overall Descriptive Statistics (Key Variables)

Descriptive statistics for the actual RMSF, model predictions, uncertainty estimates, and key input features across all residues.

| Variable                         |         count |       mean |       std |         min |         25% |        50% |        75% |        max |
|:---------------------------------|--------------:|-----------:|----------:|------------:|------------:|-----------:|-----------:|-----------:|
| `rmsf`                           | 287865.000000 |   0.465560 |  0.414388 |    0.033808 |    0.146926 |   0.301352 |   0.687178 |   2.983779 |
| `DeepFlex_rmsf`                  | 285505.000000 |   0.447640 |  0.382811 |    0.041382 |    0.159424 |   0.290527 |   0.642578 |   2.585938 |
| `ESM_Only_rmsf`                  | 286285.000000 |   0.478303 |  0.346721 |   -0.324951 |    0.207153 |   0.385986 |   0.695801 |   2.421875 |
| `voxel_rmsf`                     | 285890.000000 |   0.421479 |  0.308001 |    0.091187 |    0.180786 |   0.284912 |   0.602539 |   1.715820 |
| `ensembleflex_LGBM_rmsf`         | 287865.000000 |   0.434329 |  0.362487 |    0.030489 |    0.154586 |   0.286970 |   0.634654 |   1.974620 |
| `ensembleflex_RF_rmsf`           | 287865.000000 |   0.456728 |  0.345249 |    0.057051 |    0.186278 |   0.331733 |   0.653896 |   2.367856 |
| `No_ESM_RF_prediction`           | 287865.000000 |   0.444400 |  0.286127 |    0.090449 |    0.235963 |   0.341841 |   0.592547 |   1.720189 |
| `DeepFlex_rmsf_uncertainty`      | 285505.000000 |   0.037716 |  0.027955 |    0.001709 |    0.015671 |   0.030945 |   0.052277 |   0.430908 |
| `ensembleflex_RF_rmsf_uncertainty`| 287865.000000 |   0.111719 |  0.073610 |    0.005804 |    0.054034 |   0.092866 |   0.155081 |   0.581580 |
| `temperature`                    | 287865.000000 | 382.000000 | 46.030505 |  320.000000 |  348.000000 | 379.000000 | 413.000000 | 450.000000 |
| `normalized_resid`               | 287865.000000 |   0.500784 |  0.290775 |    0.000000 |    0.249169 |   0.500000 |   0.752941 |   1.000000 |
| `relative_accessibility`         | 287865.000000 |   0.351337 |  0.282183 |    0.000000 |    0.081218 |   0.323944 |   0.563380 |   1.000000 |
| `protein_size`                   | 287865.000000 | 190.825665 | 99.703603 |   50.000000 |  108.000000 | 171.000000 | 261.000000 | 483.000000 |
| `bfactor_norm`                   | 287865.000000 |  -0.024536 |  0.935034 |   -3.524990 |   -0.623775 |  -0.213914 |   0.349073 |   8.687703 |
| `phi`                            | 287865.000000 | -72.063505 | 62.792884 | -180.000000 | -104.300000 | -71.600000 | -60.500000 | 360.000000 |
| `psi`                            | 287865.000000 |  35.431866 | 92.663742 | -180.000000 |  -40.300000 |  -4.400000 | 130.600000 | 360.000000 |

---

## 4. Data Distributions

Counts and percentages for key categorical or binned features across all rows.

### 4.1 Temperature Distribution

| Temperature (K) | Percent |    Count |
|----------------:|--------:|---------:|
|          320.00 |   20.00 | 57573.00 |
|          348.00 |   20.00 | 57573.00 |
|          379.00 |   20.00 | 57573.00 |
|          413.00 |   20.00 | 57573.00 |
|          450.00 |   20.00 | 57573.00 |

### 4.2 Residue Name (Resname) Distribution

| Resname   | Percent |    Count |
|:----------|--------:|---------:|
| LEU       |    9.56 | 27520.00 |
| ALA       |    7.94 | 22870.00 |
| GLU       |    7.02 | 20195.00 |
| GLY       |    6.95 | 20010.00 |
| VAL       |    6.86 | 19735.00 |
| SER       |    6.15 | 17710.00 |
| ASP       |    5.90 | 16985.00 |
| LYS       |    5.84 | 16820.00 |
| THR       |    5.47 | 15740.00 |
| ILE       |    5.42 | 15595.00 |
| ARG       |    5.25 | 15110.00 |
| PRO       |    4.56 | 13115.00 |
| ASN       |    4.29 | 12360.00 |
| PHE       |    4.18 | 12020.00 |
| GLN       |    3.82 | 10985.00 |
| TYR       |    3.42 |  9840.00 |
| HIS       |    2.43 |  7005.00 |
| MET       |    2.08 |  5975.00 |
| CYS       |    1.46 |  4205.00 |
| TRP       |    1.41 |  4070.00 |

### 4.3 Core/Exterior Distribution

Based on `core_exterior_encoded` feature.

| Core/Exterior   | Percent |     Count |
|:----------------|--------:|----------:|
| Exterior        |   62.08 | 178715.00 |
| Core            |   37.92 | 109150.00 |

### 4.4 Secondary Structure (H/E/L) Distribution

Based on `secondary_structure_encoded` feature.

| Secondary Structure | Percent |     Count |
|:--------------------|--------:|----------:|
| Loop/Other          |   41.40 | 119175.00 |
| Helix               |   38.54 | 110950.00 |
| Sheet               |   20.06 |  57740.00 |

---

## 5. Comprehensive Model Comparison

Comparing performance across all detected models using global metrics.

### 5.1 Overall Performance Metrics

Standard regression metrics comparing predictions against actual RMSF. Lower RMSE/MAE/MedAE and higher PCC/R2 are better.

| Model               |     RMSE |      MAE |    MedAE |      PCC |       R2 |
|:--------------------|---------:|---------:|---------:|---------:|---------:|
| **DeepFlex**        | **0.204**| **0.135**| **0.078**| **0.873**| **0.757**|
| RF (All Features)   | 0.215    | 0.149    | 0.094    | 0.856    | 0.732    |
| LGBM (All Features) | 0.223    | 0.150    | 0.090    | 0.847    | 0.711    |
| ESM-Only (Seq+Temp) | 0.236    | 0.172    | 0.124    | 0.823    | 0.677    |
| RF (No ESM Feats)   | 0.269    | 0.193    | 0.132    | 0.767    | 0.580    |
| VoxelFlex-3D        | 0.277    | 0.191    | 0.117    | 0.752    | 0.553    |

### 5.2 Overall Rank Metrics

Metrics evaluating how well models preserve the relative order of flexibility. Higher values are better.

| Model               | Spearman Rho | Kendall Tau |
|:--------------------|-------------:|------------:|
| **DeepFlex**        |     **0.874**|    **0.691**|
| RF (All Features)   |     0.848    |    0.659    |
| LGBM (All Features) |     0.844    |    0.653    |
| ESM-Only (Seq+Temp) |     0.812    |    0.616    |
| RF (No ESM Feats)   |     0.762    |    0.566    |
| VoxelFlex-3D        |     0.759    |    0.562    |

### 5.3 Performance Metrics by Temperature

Comparing MAE, PCC, Spearman Rho, and R2 for each model, stratified by temperature.

*(Note: Table shows MAE, PCC, Rho, R2 for each model at each temperature. For brevity, only key metrics are shown here; the full table is very wide. Refer to the raw data for all columns if needed.)*

| Temp (K) | Count | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | DeepFlex R2 | RF (All) MAE | RF (All) PCC | RF (All) Rho | RF (All) R2 | ... (Other Models) |
|---------:|------:|-------------:|-------------:|-------------:|------------:|-------------:|-------------:|-------------:|------------:|:------------------:|
|   320.00 | 57573 |      **0.072** |      **0.824** |      **0.783** |     0.669 |       0.087 |       0.759 |       0.719 |      0.575 | ...                |
|   348.00 | 57573 |      **0.089** |      **0.824** |      **0.791** |     0.672 |       0.104 |       0.759 |       0.724 |      0.574 | ...                |
|   379.00 | 57573 |      **0.119** |      **0.792** |      **0.786** |     0.604 |       0.133 |       0.750 |       0.737 |      0.562 | ...                |
|   413.00 | 57573 |      **0.168** |      **0.788** |      **0.774** |     0.616 |       0.184 |       0.754 |       0.747 |      0.568 | ...                |
|   450.00 | 57573 |      **0.228** |      **0.745** |      **0.727** |     0.543 |       0.236 |       0.734 |       0.712 |      0.536 | ...                |

#### 5.3.1 Temperature Performance Summary

Model comparison highlights by temperature:

*   **T=320.0K** (N=57573):
    *   Best MAE: **DeepFlex (0.072)**
    *   Best PCC: **DeepFlex (0.824)**
    *   Best Spearman Rho: **DeepFlex (0.783)**
    *   *DeepFlex Performance:* MAE=0.072, PCC=0.824, Rho=0.783
*   **T=348.0K** (N=57573):
    *   Best MAE: **DeepFlex (0.089)**
    *   Best PCC: **DeepFlex (0.824)**
    *   Best Spearman Rho: **DeepFlex (0.791)**
    *   *DeepFlex Performance:* MAE=0.089, PCC=0.824, Rho=0.791
*   **T=379.0K** (N=57573):
    *   Best MAE: **DeepFlex (0.119)**
    *   Best PCC: **DeepFlex (0.792)**
    *   Best Spearman Rho: **DeepFlex (0.786)**
    *   *DeepFlex Performance:* MAE=0.119, PCC=0.792, Rho=0.786
*   **T=413.0K** (N=57573):
    *   Best MAE: **DeepFlex (0.168)**
    *   Best PCC: **DeepFlex (0.788)**
    *   Best Spearman Rho: **DeepFlex (0.774)**
    *   *DeepFlex Performance:* MAE=0.168, PCC=0.788, Rho=0.774
*   **T=450.0K** (N=57573):
    *   Best MAE: **DeepFlex (0.228)**
    *   Best PCC: **DeepFlex (0.745)**
    *   Best Spearman Rho: **DeepFlex (0.727)**
    *   *DeepFlex Performance:* MAE=0.228, PCC=0.745, Rho=0.727

### 5.4 Prediction R-Squared Matrix (Coefficient of Determination, Incl. Actual)

Shows the squared Pearson correlation (R²) between the predictions of different models and the actual RMSF. High values indicate predictions are similar. Diagonal is 1.

|                     | ActualRMSF | DeepFlex | ESM-Only (Seq+Temp) | VoxelFlex-3D | LGBM (All Features) | RF (All Features) | RF (No ESM Feats) |
|:--------------------|-----------:|---------:|--------------------:|-------------:|--------------------:|------------------:|------------------:|
| ActualRMSF          |     1.000  |   0.767  |              0.678  |       0.565  |              0.721  |            0.736  |            0.588  |
| DeepFlex            |     0.767  |   1.000  |              0.767  |       0.702  |              0.823  |            0.849  |            0.741  |
| ESM-Only (Seq+Temp) |     0.678  |   0.767  |              1.000  |       0.709  |              0.935  |            0.929  |            0.740  |
| VoxelFlex-3D        |     0.565  |   0.702  |              0.709  |       1.000  |              0.770  |            0.767  |            0.843  |
| LGBM (All Features) |     0.721  |   0.823  |              0.935  |       0.770  |              1.000  |            0.974  |            0.772  |
| RF (All Features)   |     0.736  |   0.849  |              0.929  |       0.767  |              0.974  |            1.000  |            0.808  |
| RF (No ESM Feats)   |     0.588  |   0.741  |              0.740  |       0.843  |              0.772  |            0.808  |            1.000  |

### 5.5 Absolute Error R-Squared Matrix (Coefficient of Determination)

Shows the squared Pearson correlation (R²) between the *absolute errors* of different models. High values indicate models tend to make errors on the same samples. The correlation between error and actual RMSF is also shown.

|                            | ActualRMSF | DeepFlex AbsErr | ESM-Only AbsErr | VoxelFlex-3D AbsErr | LGBM AbsErr | RF (All) AbsErr | RF (No ESM) AbsErr |
|:---------------------------|-----------:|----------------:|----------------:|--------------------:|------------:|----------------:|-------------------:|
| ActualRMSF                 |     1.000  |          0.259  |          0.189  |              0.398  |      0.287  |          0.254  |             0.353  |
| DeepFlex_AbsErr            |     0.259  |          1.000  |          0.298  |              0.349  |      0.427  |          0.457  |             0.354  |
| ESM-Only (Seq+Temp)_AbsErr |     0.189  |          0.298  |          1.000  |              0.359  |      0.709  |          0.738  |             0.389  |
| VoxelFlex-3D_AbsErr        |     0.398  |          0.349  |          0.359  |              1.000  |      0.472  |          0.482  |             0.691  |
| LGBM (All Features)_AbsErr |     0.287  |          0.427  |          0.709  |              0.472  |      1.000  |          0.866  |             0.423  |
| RF (All Features)_AbsErr   |     0.254  |          0.457  |          0.738  |              0.482  |      0.866  |          1.000  |             0.524  |
| RF (No ESM Feats)_AbsErr   |     0.353  |          0.354  |          0.389  |              0.691  |      0.423  |          0.524  |             1.000  |

### 5.6 Dihedral Angle (Ramachandran) Analysis

#### 5.6.1 Ramachandran Region Distribution

Distribution of residues based on their phi/psi dihedral angles falling into defined Ramachandran plot regions.

| Ramachandran Region | Percent |     Count |
|:--------------------|--------:|----------:|
| Alpha-Helix         |   51.97 | 149590.00 |
| Beta-Sheet          |   37.46 | 107830.00 |
| Other Allowed/Loop  |    6.20 |  17855.00 |
| L-Alpha             |    3.62 |  10420.00 |
| Disallowed          |    0.75 |   2170.00 |

#### 5.6.2 Performance by Ramachandran Region

Comparing model performance (MAE and PCC) stratified by Ramachandran region.

| Ramachandran Region | Count | DeepFlex MAE | DeepFlex PCC | ESM-Only MAE | ESM-Only PCC | VoxelFlex MAE | VoxelFlex PCC | LGBM MAE | LGBM PCC | RF (All) MAE | RF (All) PCC | RF (No ESM) MAE | RF (No ESM) PCC |
|:--------------------|------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|---------:|---------:|-------------:|-------------:|----------------:|----------------:|
| Beta-Sheet          |107830 |        0.129 |        0.865 |       0.169 |       0.810 |        0.178 |        0.742 |   0.141 |   0.841 |       0.140 |       0.851 |          0.182 |          0.756 |
| Alpha-Helix         |149590 |        0.134 |        0.878 |       0.171 |       0.828 |        0.195 |        0.751 |   0.152 |   0.850 |       0.149 |       0.858 |          0.193 |          0.772 |
| L-Alpha             | 10420 |        0.148 |        0.861 |       0.183 |       0.811 |        0.208 |        0.724 |   0.162 |   0.837 |       0.161 |       0.847 |          0.220 |          0.730 |
| Disallowed          |  2170 |        0.167 |        0.884 |       0.211 |       0.825 |        0.240 |        0.766 |   0.186 |   0.856 |       0.185 |       0.862 |          0.250 |          0.766 |
| Other Allowed/Loop  | 17855 |        0.174 |        0.849 |       0.195 |       0.825 |        0.224 |        0.756 |   0.184 |   0.832 |       0.181 |       0.843 |          0.227 |          0.751 |

---

## 6. Uncertainty Analysis

Comparing uncertainty estimates for models where available (DeepFlex and RF All Features).

### 6.1 Uncertainty Distribution Statistics

| Model             |         Count |     Mean |      Std |      Min |      25% |      50% |      75% |      Max |
|:------------------|--------------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| DeepFlex          | 285505.000000 |   0.0377 |   0.0280 |   0.0017 |   0.0157 |   0.0309 |   0.0523 |   0.4309 |
| RF (All Features) | 287865.000000 |   0.1117 |   0.0736 |   0.0058 |   0.0540 |   0.0929 |   0.1551 |   0.5816 |

### 6.2 Uncertainty vs. Absolute Error Correlation

Pearson Correlation Coefficient (PCC) between predicted uncertainty and the model's absolute error. A positive correlation indicates that higher predicted uncertainty tends to correspond to higher actual error.

| Model             | Uncertainty-Error PCC |
|:------------------|----------------------:|
| DeepFlex          |              0.470390 |
| RF (All Features) |              0.502957 |

### 6.3 Simple Calibration Check

Percentage of residues where the absolute error is less than or equal to the predicted uncertainty. For a perfectly calibrated Gaussian uncertainty (interpreting uncertainty as 1 standard deviation), this should be around 68.2%.

| Model             | % within 1 Uncertainty |
|:------------------|-----------------------:|
| DeepFlex          |                  18.73 |
| RF (All Features) |                  44.77 |

*(Note: This suggests both models' uncertainty estimates might be under-dispersed or not perfectly Gaussian.)*

### 6.4 Mean Absolute Error Binned by Uncertainty Quantile

Shows how the actual Mean Absolute Error (MAE) changes across different levels (deciles) of predicted uncertainty. Ideally, MAE should increase as predicted uncertainty increases.

#### DeepFlex Uncertainty

| Decile | Uncertainty Quantile Range | Mean Abs Error | Median Abs Error | Std Dev Abs Error | Count |
|-------:|:---------------------------|---------------:|-----------------:|------------------:|------:|
|      0 | 0.00171 - 0.00983          |       0.034770 |         0.025126 |          0.038218 | 28582 |
|      1 | 0.00983 - 0.0135           |       0.046894 |         0.032561 |          0.054666 | 28578 |
|      2 | 0.0135 - 0.0182            |       0.064547 |         0.043752 |          0.075666 | 28561 |
|      3 | 0.0182 - 0.0241            |       0.091877 |         0.060761 |          0.103175 | 28524 |
|      4 | 0.0241 - 0.0309            |       0.119458 |         0.083142 |          0.121592 | 28536 |
|      5 | 0.0309 - 0.0385            |       0.144165 |         0.104486 |          0.137154 | 28602 |
|      6 | 0.0385 - 0.0472            |       0.165974 |         0.125165 |          0.148381 | 28503 |
|      7 | 0.0472 - 0.0581            |       0.191583 |         0.150244 |          0.162814 | 28597 |
|      8 | 0.0581 - 0.0748            |       0.221478 |         0.179161 |          0.178064 | 28531 |
|      9 | 0.0748 - 0.431             |       0.271640 |         0.225510 |          0.212619 | 28491 |

#### RF (All Features) Uncertainty

| Decile | Uncertainty Quantile Range | Mean Abs Error | Median Abs Error | Std Dev Abs Error | Count |
|-------:|:---------------------------|---------------:|-----------------:|------------------:|------:|
|      0 | 0.0058 - 0.0356            |       0.035605 |         0.027577 |          0.039293 | 28788 |
|      1 | 0.0356 - 0.048             |       0.061345 |         0.045637 |          0.067645 | 28785 |
|      2 | 0.048 - 0.0604             |       0.083164 |         0.060806 |          0.088754 | 28789 |
|      3 | 0.0604 - 0.0748            |       0.106984 |         0.078554 |          0.107687 | 28786 |
|      4 | 0.0748 - 0.0929            |       0.127444 |         0.096888 |          0.117770 | 28785 |
|      5 | 0.0929 - 0.115             |       0.154102 |         0.121095 |          0.133634 | 28786 |
|      6 | 0.115 - 0.14               |       0.181039 |         0.147994 |          0.145318 | 28787 |
|      7 | 0.14 - 0.171               |       0.206839 |         0.176199 |          0.157563 | 28786 |
|      8 | 0.171 - 0.212              |       0.234465 |         0.205242 |          0.170492 | 28786 |
|      9 | 0.212 - 0.582              |       0.295535 |         0.258056 |          0.215711 | 28787 |

### 6.5 Reliability Diagram Data

Data for plotting reliability diagrams. Residues are binned by predicted uncertainty. For a perfectly calibrated model, the 'Mean Predicted Uncertainty' should match the 'Observed MAE/RMSE in Bin'.

#### DeepFlex

| Bin | Uncertainty Range | Mean Predicted Uncertainty | Observed MAE in Bin | Observed RMSE in Bin | Count |
|----:|:------------------|---------------------------:|--------------------:|---------------------:|------:|
|   0 | 0.00171 - 0.0446  |                   0.022055 |            0.092038 |             0.144173 | 191874|
|   1 | 0.0446 - 0.0875   |                   0.061113 |            0.209689 |             0.272856 |  77424|
|   2 | 0.0875 - 0.13     |                   0.102366 |            0.282232 |             0.355498 |  13624|
|   3 | 0.13 - 0.173      |                   0.145953 |            0.328823 |             0.414251 |   2067|
|   4 | 0.173 - 0.216     |                   0.189619 |            0.353414 |             0.451618 |    365|
|   5 | 0.216 - 0.259     |                   0.233253 |            0.359166 |             0.446407 |     94|
|   6 | 0.259 - 0.302     |                   0.274445 |            0.300673 |             0.340766 |     32|
|   7 | 0.302 - 0.345     |                   0.328701 |            0.252258 |             0.266479 |     14|
|   8 | 0.345 - 0.388     |                   0.358203 |            0.247055 |             0.266304 |     10|
|   9 | 0.388 - 0.431     |                   0.430908 |            0.364572 |             0.364572 |      1|

#### RF (All Features)

| Bin | Uncertainty Range | Mean Predicted Uncertainty | Observed MAE in Bin | Observed RMSE in Bin | Count |
|----:|:------------------|---------------------------:|--------------------:|---------------------:|------:|
|   0 | 0.0058 - 0.0634   |                   0.042281 |            0.062699 |             0.097135 |  92841|
|   1 | 0.0634 - 0.121    |                   0.089337 |            0.135503 |             0.184428 |  87477|
|   2 | 0.121 - 0.179     |                   0.148050 |            0.200098 |             0.253075 |  56067|
|   3 | 0.179 - 0.236     |                   0.203060 |            0.244029 |             0.300831 |  32833|
|   4 | 0.236 - 0.294     |                   0.259803 |            0.289036 |             0.354329 |  11561|
|   5 | 0.294 - 0.351     |                   0.319195 |            0.337105 |             0.412558 |   4454|
|   6 | 0.351 - 0.409     |                   0.374904 |            0.392305 |             0.480964 |   2008|
|   7 | 0.409 - 0.466     |                   0.428865 |            0.413878 |             0.508216 |    530|
|   8 | 0.466 - 0.524     |                   0.485645 |            0.373292 |             0.454814 |     79|
|   9 | 0.524 - 0.582     |                   0.547051 |            0.431198 |             0.517737 |     15|

### 6.6 Expected Calibration Error (ECE)

A summary metric for calibration. Lower ECE indicates better calibration (predicted uncertainty better matches actual error magnitude). Calculated using MAE as the error measure.

| Model             | Expected Calibration Error (ECE-MAE) |
|:------------------|-------------------------------------:|
| DeepFlex          |                               0.0975 |
| RF (All Features) |                               0.0371 |

*(Note: RF appears better calibrated according to this metric and binning scheme.)*

---

## 7. Domain-Level Performance Metrics

Metrics aggregated at the protein domain level.

### 7.1 Mean Metrics Across Domains

Average of performance metrics calculated independently for each of the 406 unique domains. `nan` indicates the metric is not available for that model (e.g., no uncertainty prediction).

| Metric Name                                | Mean Value Across Domains |
|:-------------------------------------------|--------------------------:|
| DeepFlex_mae                               |                  0.141082 |
| DeepFlex_pcc                               |                  0.897185 |
| DeepFlex_rmse                              |                  0.192733 |
| DeepFlex_r2                                |                  0.539260 |
| DeepFlex_avg_uncertainty                   |                  0.038607 |
| DeepFlex_uncertainty_error_corr            |                  0.445335 |
| DeepFlex_within_1std                       |                 18.007883 |
| ... (DeepFlex metrics per temperature) ... |                           |
| RF (All Features)_mae                      |                  0.157365 |
| RF (All Features)_pcc                      |                  0.883980 |
| RF (All Features)_rmse                     |                  0.208995 |
| RF (All Features)_r2                       |                  0.402776 |
| RF (All Features)_avg_uncertainty          |                  0.118085 |
| RF (All Features)_uncertainty_error_corr   |                  0.463348 |
| RF (All Features)_within_1std              |                 44.219420 |
| ... (RF metrics per temperature) ...       |                           |
| LGBM (All Features)_mae                    |                  0.160828 |
| LGBM (All Features)_pcc                    |                  0.872074 |
| ... (LGBM metrics) ...                     |                           |
| ESM-Only (Seq+Temp)_mae                    |                  0.178464 |
| ESM-Only (Seq+Temp)_pcc                    |                  0.846478 |
| ... (ESM-Only metrics) ...                 |                           |
| RF (No ESM Feats)_mae                      |                  0.202619 |
| RF (No ESM Feats)_pcc                      |                  0.838506 |
| ... (No ESM RF metrics) ...                |                           |
| VoxelFlex-3D_mae                           |                  0.203723 |
| VoxelFlex-3D_pcc                           |                  0.818898 |
| ... (VoxelFlex metrics) ...                |                           |
| actual_mean                                |                  0.519597 |
| actual_stddev                              |                  0.356862 |
| count_residues                             |                709.027094 |

*(Note: The full table includes many per-temperature metrics per domain. Refer to raw data for details.)*

---

## 8. Performance by Amino Acid

Comparing model performance (MAE, PCC, Spearman Rho) stratified by residue type.

| Resname | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | RF (All) MAE | RF (All) PCC | RF (All) Rho | ... (Other Models) |
|:--------|------:|-----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------------------:|
| MET     |  5975 |           0.450 |        0.124 |        0.895 |        0.892 |       0.140 |       0.867 |       0.855 | ...                |
| VAL     | 19735 |           0.405 |        0.124 |        0.870 |        0.874 |       0.134 |       0.858 |       0.852 | ...                |
| ILE     | 15595 |           0.408 |        0.124 |        0.875 |        0.876 |       0.134 |       0.859 |       0.852 | ...                |
| TRP     |  4070 |           0.422 |        0.126 |        0.872 |        0.873 |       0.138 |       0.852 |       0.842 | ...                |
| PHE     | 12020 |           0.414 |        0.126 |        0.869 |        0.873 |       0.134 |       0.859 |       0.851 | ...                |
| LEU     | 27520 |           0.420 |        0.126 |        0.872 |        0.873 |       0.136 |       0.859 |       0.850 | ...                |
| TYR     |  9840 |           0.416 |        0.128 |        0.867 |        0.864 |       0.140 |       0.855 |       0.843 | ...                |
| ALA     | 22870 |           0.467 |        0.132 |        0.878 |        0.880 |       0.144 |       0.864 |       0.855 | ...                |
| GLN     | 10985 |           0.504 |        0.136 |        0.882 |        0.882 |       0.154 |       0.856 |       0.842 | ...                |
| ARG     | 15110 |           0.473 |        0.137 |        0.870 |        0.871 |       0.152 |       0.853 |       0.840 | ...                |
| THR     | 15740 |           0.470 |        0.138 |        0.872 |        0.872 |       0.152 |       0.857 |       0.848 | ...                |
| PRO     | 13115 |           0.490 |        0.141 |        0.868 |        0.864 |       0.155 |       0.848 |       0.834 | ...                |
| GLU     | 20195 |           0.500 |        0.141 |        0.871 |        0.869 |       0.159 |       0.847 |       0.838 | ...                |
| SER     | 17710 |           0.507 |        0.142 |        0.879 |        0.877 |       0.158 |       0.860 |       0.851 | ...                |
| ASP     | 16985 |           0.504 |        0.143 |        0.867 |        0.868 |       0.157 |       0.846 |       0.839 | ...                |
| GLY     | 20010 |           0.501 |        0.143 |        0.871 |        0.870 |       0.156 |       0.859 |       0.850 | ...                |
| LYS     | 16820 |           0.501 |        0.143 |        0.873 |        0.871 |       0.161 |       0.852 |       0.838 | ...                |
| CYS     |  4205 |           0.419 |        0.144 |        0.819 |        0.854 |       0.142 |       0.832 |       0.837 | ...                |
| ASN     | 12360 |           0.497 |        0.144 |        0.866 |        0.864 |       0.162 |       0.839 |       0.826 | ...                |
| HIS     |  7005 |           0.499 |        0.145 |        0.872 |        0.873 |       0.155 |       0.861 |       0.856 | ...                |

---

## 9. Performance by Normalized Residue Position

Comparing model performance stratified by the residue's normalized position within its protein chain (0=N-terminus, 1=C-terminus), binned into quintiles.

| Normalized Residue Bin | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | RF (All) MAE | RF (All) PCC | RF (All) Rho | ... (Other Models) |
|:-----------------------|------:|-----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------------------:|
| 0 - 0.199              | 57580 |           0.531 |        0.151 |        0.881 |        0.884 |       0.170 |       0.860 |       0.858 | ...                |
| 0.199 - 0.4            | 57730 |           0.426 |        0.121 |        0.868 |        0.880 |       0.132 |       0.855 |       0.851 | ...                |
| 0.4 - 0.602            | 57410 |           0.412 |        0.121 |        0.858 |        0.868 |       0.132 |       0.844 |       0.842 | ...                |
| 0.602 - 0.803          | 57575 |           0.420 |        0.122 |        0.867 |        0.868 |       0.133 |       0.851 |       0.839 | ...                |
| 0.803 - 1              | 57570 |           0.539 |        0.161 |        0.869 |        0.868 |       0.175 |       0.852 |       0.848 | ...                |

---

## 10. Performance by Core/Exterior

Comparing model performance based on whether a residue is classified as 'Core' or 'Exterior'.

| Core/Exterior | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | RF (All) MAE | RF (All) PCC | RF (All) Rho | ... (Other Models) |
|:--------------|------:|-----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------------------:|
| Core          |109150 |           0.347 |        0.115 |        0.855 |        0.848 |       0.123 |       0.848 |       0.828 | ...                |
| Exterior      |178715 |           0.538 |        0.148 |        0.870 |        0.870 |       0.165 |       0.848 |       0.835 | ...                |

---

## 11. Performance by Secondary Structure (H/E/L)

Comparing model performance based on secondary structure assignment (Helix, Sheet, Loop/Other).

| Secondary Structure | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | RF (All) MAE | RF (All) PCC | RF (All) Rho | ... (Other Models) |
|:--------------------|------:|-----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------------------:|
| Sheet               | 57740 |           0.301 |        0.109 |        0.834 |        0.835 |       0.118 |       0.826 |       0.809 | ...                |
| Helix               |110950 |           0.469 |        0.129 |        0.880 |        0.879 |       0.144 |       0.859 |       0.845 | ...                |
| Loop/Other          |119175 |           0.542 |        0.154 |        0.864 |        0.859 |       0.168 |       0.847 |       0.834 | ...                |

---

## 12. Performance by Actual RMSF Quantile

Performance of the primary model (**DeepFlex**) stratified by the ground truth RMSF value (binned into deciles). This helps identify if the model struggles more at low or high flexibility values.

| Actual RMSF Range | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | DeepFlex RMSE | DeepFlex R2 |
|:------------------|------:|-----------------:|-------------:|-------------:|-------------:|--------------:|------------:|
| 0.0338 - 0.0918   | 28554 |           0.0722 |       0.0477 |       0.2429 |       0.3690 |      0.075547 |  -32.981255 |
| 0.0918 - 0.127    | 28547 |           0.1092 |       0.0540 |       0.1261 |       0.1674 |      0.092346 |  -82.610004 |
| 0.127 - 0.17      | 28551 |           0.1476 |       0.0611 |       0.1135 |       0.1528 |      0.106589 |  -74.486564 |
| 0.17 - 0.225      | 28550 |           0.1959 |       0.0766 |       0.1012 |       0.1317 |      0.124549 |  -60.449513 |
| 0.225 - 0.302     | 28551 |           0.2608 |       0.0993 |       0.1210 |       0.1507 |      0.148339 |  -43.758597 |
| 0.302 - 0.414     | 28550 |           0.3543 |       0.1301 |       0.1529 |       0.1705 |      0.175785 |  -28.099099 |
| 0.414 - 0.581     | 28550 |           0.4924 |       0.1691 |       0.1760 |       0.1838 |      0.213940 |  -18.724679 |
| 0.581 - 0.811     | 28551 |           0.6908 |       0.2109 |       0.2028 |       0.2064 |      0.256864 |  -13.869809 |
| 0.811 - 1.11      | 28550 |           0.9511 |       0.2215 |       0.2489 |       0.2550 |      0.279680 |  -10.006190 |
| 1.11 - 2.98       | 28551 |           1.3852 |       0.2817 |       0.5222 |       0.4391 |      0.365778 |   -1.232642 |

*(Note: Low PCC/Rho and negative R2 within narrow bins are expected, as the range of actual RMSF values is small within each bin, making correlation difficult.)*

---

## 13. Performance by Relative Accessibility Quantile

Comparing model performance stratified by Relative Solvent Accessibility (RSA), binned into quintiles.

| RSA Bin         | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | RF (All) MAE | RF (All) PCC | RF (All) Rho | ... (Other Models) |
|:----------------|------:|-----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------------------:|
| 0 - 0.0462      | 57615 |           0.300 |        0.101 |        0.854 |        0.845 |       0.108 |       0.849 |       0.826 | ...                |
| 0.0462 - 0.225  | 57560 |           0.403 |        0.128 |        0.857 |        0.850 |       0.139 |       0.843 |       0.821 | ...                |
| 0.225 - 0.423   | 57595 |           0.464 |        0.138 |        0.860 |        0.859 |       0.153 |       0.841 |       0.822 | ...                |
| 0.423 - 0.618   | 57530 |           0.521 |        0.150 |        0.856 |        0.848 |       0.165 |       0.837 |       0.814 | ...                |
| 0.618 - 1       | 57565 |           0.639 |        0.159 |        0.875 |        0.877 |       0.179 |       0.848 |       0.837 | ...                |

---

## 14. Performance by Normalized B-Factor Quantile

Comparing model performance stratified by normalized experimental B-factors (`bfactor_norm`), binned into quintiles.

| Bfactor Norm Bin | Count | Mean Actual RMSF | DeepFlex MAE | DeepFlex PCC | DeepFlex Rho | RF (All) MAE | RF (All) PCC | RF (All) Rho | ... (Other Models) |
|:-----------------|------:|-----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:------------------:|
| -3.52 - -0.728   | 57575 |           0.412 |        0.121 |        0.880 |        0.884 |       0.133 |       0.870 |       0.861 | ...                |
| -0.728 - -0.306  | 57575 |           0.398 |        0.122 |        0.868 |        0.870 |       0.134 |       0.856 |       0.845 | ...                |
| -0.306 - -0.0978 | 57570 |           0.512 |        0.149 |        0.860 |        0.859 |       0.159 |       0.844 |       0.832 | ...                |
| -0.0978 - 0.572  | 57575 |           0.458 |        0.134 |        0.871 |        0.862 |       0.149 |       0.854 |       0.836 | ...                |
| 0.572 - 8.69     | 57570 |           0.549 |        0.150 |        0.874 |        0.869 |       0.169 |       0.846 |       0.837 | ...                |

---

## 15. Model Disagreement vs. Error

Analysis of how disagreement between model predictions relates to prediction error and uncertainty. Disagreement is measured by the standard deviation and range of predictions across all available models for each residue.

**Model Prediction Disagreement Statistics:**

| Statistic         | Prediction Std Dev | Prediction Range |
|:------------------|-------------------:|-----------------:|
| Count             |      287865.000000 |     287865.00000 |
| Mean              |           0.095677 |          0.25335 |
| Std Dev           |           0.063331 |          0.16556 |
| Min               |           0.004295 |          0.01010 |
| 25% (Q1)          |           0.048686 |          0.12996 |
| 50% (Median)      |           0.079112 |          0.21128 |
| 75% (Q3)          |           0.126508 |          0.33644 |
| Max               |           0.611054 |          1.71706 |

**Correlations (N = 285505, where DeepFlex data is available):**

*   PCC (Prediction Std Dev vs. DeepFlex Abs Error): **0.384**
*   PCC (Prediction Range vs. DeepFlex Abs Error): **0.377**
*   PCC (Prediction Std Dev vs. DeepFlex Uncertainty): **0.527**
*   PCC (Prediction Range vs. DeepFlex Uncertainty): **0.521**

*(Interpretation: Higher disagreement between models weakly correlates with higher absolute error for DeepFlex. Disagreement correlates more strongly with DeepFlex's predicted uncertainty.)*

---

## 16. Case Study Candidates

Domains selected based on DeepFlex performance metrics, potentially interesting for closer examination. Lists are mutually exclusive based on the order presented. (Max 15 shown per category).

### 16.1 High Accuracy Candidates

> Criteria: Domain-level PCC > 0.93 AND MAE < 0.12 AND Actual RMSF Std Dev > 0.15.
> (Found 103 matching domains)

| Domain ID | Residues | Mean RMSF | Std RMSF | DeepFlex MAE | DeepFlex PCC | MAE 320K | PCC 320K | Rho 320K | MAE 450K | PCC 450K | Rho 450K | ΔMAE | ΔPCC | ΔRho |
|:----------|---------:|----------:|---------:|-------------:|-------------:|---------:|---------:|---------:|---------:|---------:|---------:|-----:|-----:|-----:|
| 1r6bX05   |      495 |     0.461 |    0.399 |        0.055 |        0.983 |    0.034 |    0.965 |    0.884 |    0.084 |    0.886 |    0.702 | 0.05 | -0.08| -0.18|
| 3qiiA00   |      260 |     0.231 |    0.239 |        0.060 |        0.965 |    0.053 |    0.619 |    0.834 |    0.084 |    0.925 |    0.940 | 0.03 |  0.31|  0.11|
| 1sz9C00   |      700 |     0.389 |    0.390 |        0.062 |        0.977 |    0.031 |    0.956 |    0.848 |    0.128 |    0.829 |    0.727 | 0.10 | -0.13| -0.12|
| 1w5sA02   |      950 |     0.231 |    0.231 |        0.067 |        0.931 |    0.029 |    0.810 |    0.742 |    0.170 |    0.859 |    0.752 | 0.14 |  0.05|  0.01|
| 3dsoA00   |      330 |     0.391 |    0.334 |        0.068 |        0.959 |    0.042 |    0.864 |    0.811 |    0.103 |    0.746 |    0.749 | 0.06 | -0.12| -0.06|
| ... (10 more) ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 16.2 Good Temperature Handling Candidates

> Criteria: Domain-level PCC > 0.9 AND Absolute Change in MAE (450K vs 320K) < 0.1 AND Change in PCC (450K vs 320K) > -0.05.
> (Found 74 matching domains, excluding those already in High Accuracy list)

| Domain ID | Residues | Mean RMSF | Std RMSF | DeepFlex MAE | DeepFlex PCC | MAE 320K | PCC 320K | Rho 320K | MAE 450K | PCC 450K | Rho 450K | ΔMAE | ΔPCC | ΔRho |
|:----------|---------:|----------:|---------:|-------------:|-------------:|---------:|---------:|---------:|---------:|---------:|---------:|-----:|-----:|-----:|
| 3abgB03   |      930 |     0.787 |    0.444 |        0.153 |        0.926 |    0.153 |    0.923 |    0.874 |    0.148 |    0.901 |    0.861 | -0.00| -0.02| -0.01|
| 1lp1A00   |      275 |     0.683 |    0.426 |        0.110 |        0.949 |    0.072 |    0.833 |    0.517 |    0.066 |    0.970 |    0.835 | -0.01|  0.14|  0.32|
| 1v66A00   |      325 |     0.409 |    0.378 |        0.080 |        0.979 |    0.063 |    0.847 |    0.789 |    0.056 |    0.941 |    0.895 | -0.01|  0.09|  0.11|
| 3zrgA00   |      330 |     0.484 |    0.428 |        0.240 |        0.917 |    0.115 |    0.858 |    0.799 |    0.122 |    0.873 |    0.909 |  0.01|  0.01|  0.11|
| 2cmpA00   |      280 |     0.550 |    0.443 |        0.101 |        0.975 |    0.127 |    0.875 |    0.761 |    0.119 |    0.980 |    0.965 | -0.01|  0.11|  0.20|
| ... (10 more) ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 16.3 Challenging Candidates

> Criteria: Domain-level PCC < 0.8 OR MAE > 0.25 OR Absolute Change in MAE (450K vs 320K) > 0.20.
> (Found 137 matching domains, excluding those in previous lists)

| Domain ID | Residues | Mean RMSF | Std RMSF | DeepFlex MAE | DeepFlex PCC | MAE 320K | PCC 320K | Rho 320K | MAE 450K | PCC 450K | Rho 450K | ΔMAE | ΔPCC | ΔRho |
|:----------|---------:|----------:|---------:|-------------:|-------------:|---------:|---------:|---------:|---------:|---------:|---------:|-----:|-----:|-----:|
| 3h33A00   |      355 |     1.013 |    0.347 |        0.544 |        0.716 |    0.187 |    0.756 |    0.782 |    0.647 |    0.285 |    0.026 | 0.46 | -0.47| -0.76|
| 1bhuA00   |      510 |     0.933 |    0.422 |        0.439 |        0.724 |    0.176 |    0.745 |    0.733 |    0.218 |    0.641 |    0.312 | 0.04 | -0.10| -0.42|
| 1ruyH02   |     1150 |     0.296 |    0.299 |        0.399 |        0.712 |    0.260 |    0.321 |    0.150 |    0.351 |    0.678 |    0.606 | 0.09 |  0.36|  0.46|
| 2mk5A00   |      655 |     0.910 |    0.420 |        0.382 |        0.573 |    0.500 |    0.873 |    0.855 |    0.456 |    0.908 |    0.891 | -0.04|  0.03|  0.04|
| 2rowA01   |      355 |     0.829 |    0.347 |        0.379 |        0.854 |    0.350 |    0.537 |    0.599 |    0.313 |    0.853 |    0.558 | -0.04|  0.32| -0.04|
| ... (10 more) ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 17. Statistical Significance Tests

Comparing **DeepFlex** against key baseline models using the Wilcoxon signed-rank test on per-domain metrics (MAE and PCC).

> -   **Null Hypothesis (H0):** The median difference between the paired domain-level metrics (e.g., DeepFlex MAE vs. RF MAE) is zero.
> -   **MAE Test Alternative:** DeepFlex MAE is significantly *smaller* than the baseline MAE (one-sided test, `alternative='less'`).
> -   **PCC Test Alternative:** DeepFlex PCC is significantly *larger* than the baseline PCC (one-sided test, `alternative='greater'`).

| Baseline Compared To DeepFlex | Metric | p-value   | N Pairs | Interpretation                                        |
|:------------------------------|:-------|:----------|:--------|:------------------------------------------------------|
| RF (All Features)             | MAE    | 2.391e-11 |     403 | DeepFlex MAE is significantly lower (p < 0.001).    |
| RF (All Features)             | PCC    | 1.468e-14 |     403 | DeepFlex PCC is significantly higher (p < 0.001).   |
| ESM-Only (Seq+Temp)           | MAE    | 2.175e-32 |     403 | DeepFlex MAE is significantly lower (p < 0.001).    |
| ESM-Only (Seq+Temp)           | PCC    | 2.795e-50 |     403 | DeepFlex PCC is significantly higher (p < 0.001).   |

*(Conclusion: DeepFlex shows statistically significant improvement over RF (All Features) and ESM-Only (Seq+Temp) based on median per-domain MAE and PCC.)*

---

## 18. Feature Attribution & Interpretability (Placeholder)

> **Note:** This analysis requires access to the trained **DeepFlex** model object and appropriate interpretability libraries (e.g., SHAP, Captum) or pre-computed attribution scores.
>
> **Potential Steps (if data/model available):**
> 1.  Load the DeepFlex model and a representative data subset.
> 2.  Initialize an explainer tool (e.g., SHAP KernelExplainer, DeepExplainer, Integrated Gradients).
> 3.  Compute feature attributions for inputs like:
>     *   Temperature embedding contribution.
>     *   Aggregate contribution of ESM sequence embedding dimensions.
>     *   Contribution of 1D structural features (SASA, SS, phi/psi, etc.).
>     *   Contribution of 3D structural context (if applicable via VoxelFlex-like inputs).
> 4.  Summarize average absolute attributions globally or stratified by residue type, secondary structure, etc.
> 5.  Visualize mean attributions (e.g., bar plot).
> 6.  If attention weights are available (e.g., from Transformer layers within DeepFlex), analyze and visualize attention patterns.

---

## 19. Temperature-Encoding Ablation (Placeholder)

> **Note:** This analysis requires results from a **DeepFlex** model trained *without* the temperature conditioning mechanism.
>
> **Input Needed:** Predictions from the ablated model (e.g., a column like `DeepFlex_NoTemp_rmsf` in the dataset).
>
> **Analysis Steps (if ablated predictions available):**
> 1.  Add the ablated model to the `MODEL_CONFIG` used for analysis.
> 2.  Re-run the performance comparison (Section 5), focusing on the difference between the full DeepFlex model and the ablated version.
> 3.  Compare overall metrics (MAE, PCC, Rho).
> 4.  Critically, compare per-temperature metrics (Section 5.3) to quantify the performance drop/change at different temperatures when conditioning is removed.
> 5.  Calculate the delta in MAE/PCC between the full and ablated model per domain and analyze the distribution of this delta. This highlights which domains benefited most from temperature conditioning.

---

## 20. Cross-Dataset / External Validation (Placeholder)

> **Note:** This analysis requires an independent dataset with ground truth RMSF values, not used during the training or hyperparameter tuning of **DeepFlex**.
>
> **Input Needed:** Path to an external validation CSV file (must have compatible columns: residue identifiers, temperature, actual RMSF, and necessary features for model input).
>
> **Analysis Steps:**
> 1.  Load the external dataset.
> 2.  Run inference with the trained **DeepFlex** model on the inputs from the external dataset to generate predictions.
> 3.  Merge these predictions with the external ground truth RMSF values.
> 4.  Calculate performance metrics (MAE, PCC, Spearman Rho, R²) on this external dataset.
> 5.  Compare these external validation metrics to the performance observed on the internal test set (e.g., Section 5.1) to assess generalization ability.
> 6.  Optionally, stratify performance on the external set by available features (temperature, protein characteristics, data source, etc.).

---

## 21. Computational Performance & Scalability (Placeholder)

> **Note:** This section requires benchmark data obtained from running model inference and MD simulations separately.
>
> **Information Needed:**
> 1.  **DeepFlex Inference Time:**
>     *   Average time per protein or per residue.
>     *   Hardware specifications (GPU model, CPU type, cores).
>     *   Memory usage (GPU RAM, System RAM).
> 2.  **MD Simulation Time:**
>     *   Typical time for a standard simulation (e.g., 100 ns) of a representative protein.
>     *   Hardware specifications (GPU model) and simulation software (e.g., GROMACS, AMBER).
>     *   System size (number of atoms).
> 3.  **Comparison:**
>     *   Calculate the approximate speedup factor (MD time / Inference time) for obtaining flexibility estimates.
>     *   Discuss scalability: How does DeepFlex inference time change with protein size (sequence length)?
>
> **Example Placeholder Values (Update with actual measurements):**
> *   DeepFlex Inference Time: ~0.5 seconds per protein (Avg Size 190 residues) on Nvidia A100 GPU.
> *   MD Simulation Time (100ns): ~24 hours for a 30k atom system on Nvidia A100 GPU.
> *   Estimated Speedup Factor: > 100,000x (for equivalent data point generation)

---

## 22. Error Analysis vs. Features

Examining the relationship between **DeepFlex** absolute error and specific input features.

### 22.1 Error vs. Normalized B-Factor

How DeepFlex absolute error varies across quintiles of normalized experimental B-factors (`bfactor_norm`).

| Feature Range (`bfactor_norm`) | Mean Feature Value | Mean DeepFlex Abs Error | Median DeepFlex Abs Error | Count |
|:-------------------------------|-------------------:|------------------------:|--------------------------:|------:|
| -3.52 - -0.728                 |          -1.040291 |                0.121206 |                  0.062896 | 57105 |
| -0.728 - -0.307                |          -0.520456 |                0.121994 |                  0.064021 | 57100 |
| -0.307 - -0.0968               |          -0.210043 |                0.148611 |                  0.092109 | 57100 |
| -0.0968 - 0.574                |           0.193933 |                0.134007 |                  0.078372 | 57100 |
| 0.574 - 8.69                   |           1.455031 |                0.150147 |                  0.096114 | 57100 |

### 22.2 Error vs. Contact Number

> Skipped: Feature column `contact_number` not found in the dataset.

### 22.3 Error vs. Co-Evolution Signal

> Skipped: Feature column `coevolution_score` not found in the dataset.

