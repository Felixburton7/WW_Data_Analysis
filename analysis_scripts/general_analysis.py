# general_analysis.py (Robust Version v2)

import time
import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from collections import defaultdict
from tabulate import tabulate
import argparse
import json
import re # Import re for robust sorting key
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple

# --- Configuration ---
# Define models and their corresponding prediction/uncertainty columns
# Matches the keys used in the rest of the script for consistency
MODEL_CONFIG = {
    "DeepFlex": {
        "key": "Attention_ESM",
        "pred_col": "Attention_ESM_rmsf",
        "unc_col": "Attention_ESM_rmsf_uncertainty"
    },
    "ESM-Only (Seq+Temp)": {
        "key": "ESM_Only",
        "pred_col": "ESM_Only_rmsf",
        "unc_col": None # Or provide col name if uncertainty exists
    },
    "VoxelFlex-3D": {
        "key": "Voxel",
        "pred_col": "voxel_rmsf",
        "unc_col": None
    },
    "LGBM (All Features)": {
        "key": "LGBM",
        "pred_col": "ensembleflex_LGBM_rmsf",
        "unc_col": None
    },
    "RF (All Features)": {
        "key": "RF",
        "pred_col": "ensembleflex_RF_rmsf",
        "unc_col": "ensembleflex_RF_rmsf_uncertainty"
    },
    "RF (No ESM Feats)": {
        "key": "No_ESM_RF",
        "pred_col": "No_ESM_RF_prediction",
        "unc_col": None
    }
}
# Key baselines for significance testing against DeepFlex
KEY_BASELINES_FOR_SIG_TEST = ["RF (All Features)", "ESM-Only (Seq+Temp)"]
TARGET_COL = 'rmsf' # Define target column centrally

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def safe_pearsonr(x, y):
    """Calculates Pearson correlation safely, returning NaN on error or insufficient data."""
    try:
        # Ensure numpy arrays and handle potential all-NaN slices after filtering
        x_np = np.asarray(x).astype(np.float64)
        y_np = np.asarray(y).astype(np.float64)
        valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]

        if len(x_clean) < 2:
            # logger.debug("Not enough valid data points (<2) for Pearson correlation.")
            return np.nan, np.nan # Return NaN for p-value too
        # Check for near-zero variance AFTER cleaning NaNs
        if np.std(x_clean) < 1e-9 or np.std(y_clean) < 1e-9:
            # logger.debug("One or both arrays have near-zero standard deviation. Pearson correlation is undefined.")
            return np.nan, np.nan

        # Use nan_policy='omit' if scipy version supports it, otherwise rely on manual cleaning
        try:
            # Scipy pearsonr raises ValueError if inputs are constant
            corr, p_value = pearsonr(x_clean, y_clean)
        except ValueError as ve:
            # Handle constant input case explicitly if needed, though std check above should catch it
            logger.debug(f"Pearson calculation failed (likely constant input): {ve}")
            return np.nan, np.nan
        except TypeError: # Older scipy might not handle NaNs well even after cleaning? Fallback just in case
            logger.warning("Pearson TypeError encountered, retrying calculation.")
            corr, p_value = pearsonr(x_clean, y_clean)

        # Final NaN check, although previous checks should prevent this
        return corr if not np.isnan(corr) else np.nan, p_value
    except (ValueError, FloatingPointError) as e: # Catch specific numeric errors
        logger.debug(f"Pearson calculation failed with numeric error: {e}")
        return np.nan, np.nan
    except Exception as e:
        logger.error(f"Unexpected error during pearsonr calculation: {e}", exc_info=True)
        return np.nan, np.nan

def format_table(data, headers="keys", tablefmt="pipe", floatfmt=".6f", **kwargs):
    """Formats data using tabulate, with improved error handling for empty data."""
    if isinstance(data, pd.DataFrame) and data.empty:
        return " (No data to display) "
    # Also check if data is a list or other iterable and is empty
    if not isinstance(data, pd.DataFrame) and not data:
         return " (No data to display) "
    try:
        return tabulate(data, headers=headers, tablefmt=tablefmt, floatfmt=floatfmt, **kwargs)
    except Exception as e:
        logger.error(f"Tabulate formatting failed: {e}")
        # Provide more context if data is a DataFrame
        if isinstance(data, pd.DataFrame):
             return f"(Error formatting DataFrame: {e})\nColumns: {data.columns.tolist()}\nFirst few rows:\n{data.head().to_string()}"
        return f"(Error formatting data: {e})\nData: {str(data)[:200]}..." # Fallback to string representation

def parse_key_for_sort(key_str):
    """ Parses section keys like '1.1', '5.2.1', '15.' for robust numerical sorting."""
    if not isinstance(key_str, str):
        return (999,) # Handle non-string keys

    # Try to match the pattern: Number(s) followed by optional dot and text
    match = re.match(r"^(\d+(?:\.\d+)*)\.?(.*)", key_str.strip())
    if match:
        num_part = match.group(1)
        # text_part = match.group(2).strip() # Text part if needed later
        try:
            # Split numeric part by '.' and convert to tuple of integers
            return tuple(int(p) for p in num_part.split('.'))
        except ValueError:
            # Fallback if conversion fails (e.g., unexpected characters)
            return (999,)
    else:
        # Fallback for keys that don't start with the expected pattern
        return (999,)

# --- Per-Domain Metrics (Robust Version) ---
def calculate_per_domain_metrics(df, target_col='rmsf'):
    """Calculates MAE and PCC for each model within each domain. Handles missing data."""
    logger.info("Calculating per-domain performance metrics...")
    domain_metrics = {} # Use dict directly: {domain_id: {model_name: {'mae': v, 'pcc': v}}}
    required_cols = ['domain_id', target_col]

    if 'domain_id' not in df.columns:
        logger.error("Input DataFrame missing essential 'domain_id' column for per-domain metrics. Skipping calculation.")
        return {}
    if target_col not in df.columns:
         logger.error(f"Input DataFrame missing essential target column '{target_col}' for per-domain metrics. Skipping calculation.")
         return {}

    # Pre-filter DataFrame for rows where target is not NaN and domain_id is not NaN
    df_filtered = df.dropna(subset=[target_col, 'domain_id']).copy()
    if df_filtered.empty:
        logger.warning(f"No valid rows found after dropping NaNs in target/domain_id columns. Cannot calculate per-domain metrics.")
        return {}

    grouped = df_filtered.groupby('domain_id')
    num_domains = len(grouped)
    processed_domains = 0

    for domain_id, group in tqdm(grouped, desc="Domain Metrics", total=num_domains):
        processed_domains += 1
        # Check if group is empty after potential filtering (shouldn't happen with pre-filtering)
        if group.empty:
            logger.debug(f"Skipping domain {domain_id} as the group is empty.")
            continue

        domain_results = {} # Results for this specific domain
        actual = group[target_col].values # Already filtered NaNs for target

        for report_name, config in MODEL_CONFIG.items():
            pred_col = config['pred_col']
            mae_val, pcc_val = np.nan, np.nan # Default to NaN

            if pred_col in group.columns:
                predicted = group[pred_col].values
                # Ensure lengths match (should if grouped correctly) and drop NaNs pair-wise for this model's prediction
                valid_mask = ~np.isnan(predicted) # Actual is already not NaN
                actual_valid = actual[valid_mask]
                predicted_valid = predicted[valid_mask]

                if len(actual_valid) > 1: # Need at least 2 points
                    try:
                        mae_val = mean_absolute_error(actual_valid, predicted_valid)
                    except Exception as e:
                        logger.debug(f"MAE calculation failed for {domain_id}/{report_name}: {e}")
                        mae_val = np.nan # Ensure NaN on failure
                    try:
                        pcc_val, _ = safe_pearsonr(actual_valid, predicted_valid) # Handles internal checks and returns NaN on failure
                    except Exception as e:
                        logger.debug(f"PCC calculation failed for {domain_id}/{report_name}: {e}")
                        pcc_val = np.nan # Ensure NaN on failure
                # else: logger.debug(f"Less than 2 valid points for {domain_id}/{report_name}")
            # else: logger.debug(f"Prediction column {pred_col} not found for domain {domain_id}")

            # Store results, even if NaN
            domain_results[report_name] = {'mae': mae_val, 'pcc': pcc_val}

        # Only add domain to final dict if it has results (even if NaN)
        if domain_results:
             domain_metrics[domain_id] = domain_results

    logger.info(f"Calculated per-domain metrics for {len(domain_metrics)} domains.")
    return domain_metrics # Return the dict {domain_id: {model_name: {metric: val}}}


# --- Significance Testing (Robust Version) ---
def calculate_significance_tests(per_domain_metrics, primary_model_report_name, baseline_report_names):
    """Performs Wilcoxon tests comparing primary model to baselines on per-domain metrics."""
    logger.info(f"Performing significance tests comparing {primary_model_report_name} against {baseline_report_names}...")
    results = {}
    if not per_domain_metrics:
        logger.warning("Per-domain metrics dictionary is empty. Skipping significance tests.")
        return results

    # Check if primary model exists in the metrics keys
    first_domain_key = next(iter(per_domain_metrics), None)
    if not first_domain_key or primary_model_report_name not in per_domain_metrics[first_domain_key]:
         logger.error(f"Primary model '{primary_model_report_name}' not found in per-domain metrics data. Cannot perform tests.")
         return results

    domain_ids_primary = list(per_domain_metrics.keys())

    # Function to extract paired metric values safely
    def get_paired_values(metric_key):
        primary_values = []
        baseline_values = []
        # Ensure we iterate through the same domains in the same order
        for domain_id in domain_ids_primary:
             # Use .get() with default empty dicts to avoid KeyErrors
             primary_val = per_domain_metrics.get(domain_id, {}).get(primary_model_report_name, {}).get(metric_key, np.nan)
             baseline_val = per_domain_metrics.get(domain_id, {}).get(baseline_name, {}).get(metric_key, np.nan)
             # Only include pairs where BOTH are valid numbers
             if not np.isnan(primary_val) and not np.isnan(baseline_val):
                 primary_values.append(primary_val)
                 baseline_values.append(baseline_val)
             # else: logger.debug(f"Skipping pair for {domain_id}, metric {metric_key}: P={primary_val}, B={baseline_val}")
        return primary_values, baseline_values

    for baseline_name in baseline_report_names:
        # Check if baseline model exists in the metrics keys
        if not first_domain_key or baseline_name not in per_domain_metrics[first_domain_key]:
            logger.warning(f"Baseline model '{baseline_name}' not found in per-domain metrics data. Skipping tests against it.")
            results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': 0}
            results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': 0}
            continue

        primary_mae, baseline_mae = get_paired_values('mae')
        primary_pcc, baseline_pcc = get_paired_values('pcc')

        # Perform MAE test
        n_mae = len(primary_mae)
        if n_mae < 10: # Minimum samples for a somewhat reliable non-parametric test
            logger.warning(f"Too few valid paired MAE samples ({n_mae}) for Wilcoxon test vs {baseline_name}. Requires >= 10. Skipping.")
            results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_mae}
        else:
            try:
                # Test if primary MAE is significantly LESS than baseline MAE
                stat_mae, p_mae = wilcoxon(primary_mae, baseline_mae, alternative='less', zero_method='zsplit')
                results[f"{baseline_name}_vs_MAE"] = {'p_value': p_mae, 'statistic': stat_mae, 'N': n_mae}
            except ValueError as e: # Handle cases like all differences being zero
                 logger.warning(f"Wilcoxon test failed for MAE vs {baseline_name}: {e}")
                 results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_mae}

        # Perform PCC test
        n_pcc = len(primary_pcc)
        if n_pcc < 10:
            logger.warning(f"Too few valid paired PCC samples ({n_pcc}) for Wilcoxon test vs {baseline_name}. Requires >= 10. Skipping.")
            results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_pcc}
        else:
            try:
                # Test if primary PCC is significantly GREATER than baseline PCC
                stat_pcc, p_pcc = wilcoxon(primary_pcc, baseline_pcc, alternative='greater', zero_method='zsplit')
                results[f"{baseline_name}_vs_PCC"] = {'p_value': p_pcc, 'statistic': stat_pcc, 'N': n_pcc}
            except ValueError as e:
                 logger.warning(f"Wilcoxon test failed for PCC vs {baseline_name}: {e}")
                 results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_pcc}

    logger.info("Significance testing complete.")
    return results



# --- Performance Domain level metrics 
# --- Domain Level Analysis (Robust Version) ---
def run_domain_level_analysis(df, analysis_results):
    """Runs domain-level analysis and saves detailed metrics per domain to a CSV file."""
    logger.info("Running domain-level analysis...")
    analysis_results['7. DOMAIN-LEVEL PERFORMANCE METRICS'] = "" # Initialize section header

    # Calculate per-domain metrics (MAE, PCC for now) using the helper
    domain_metrics_dict = calculate_per_domain_metrics(df, target_col=TARGET_COL)

    if not domain_metrics_dict:
        logger.error("Failed to calculate per-domain metrics. Skipping domain-level analysis.")
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: No per-domain metrics calculated."
        # Ensure the internal path is not set if calculation failed
        if '_internal_domain_metrics_path' in analysis_results:
             del analysis_results['_internal_domain_metrics_path']
        return

    # --- Convert dict to DataFrame and add more stats ---
    domain_data_list = []
    all_metric_keys = set() # Track all generated metric columns

    # Add more complex metrics by iterating through domains again (less efficient but clearer)
    if 'domain_id' not in df.columns:
         logger.error("Cannot calculate detailed domain stats: 'domain_id' missing from main DataFrame.")
         return

    grouped_by_domain = df.groupby('domain_id')

    for domain_id, model_metrics in tqdm(domain_metrics_dict.items(), desc="Aggregating Domain Stats", total=len(domain_metrics_dict)):
        row = {'domain_id': domain_id}
        domain_group = grouped_by_domain.get_group(domain_id) # Get original data for this domain

        # Add pre-calculated MAE/PCC
        for model_name, metrics in model_metrics.items():
            row[f"{model_name}_mae"] = metrics.get('mae', np.nan)
            row[f"{model_name}_pcc"] = metrics.get('pcc', np.nan)
            all_metric_keys.add(f"{model_name}_mae")
            all_metric_keys.add(f"{model_name}_pcc")

        # Add basic domain info
        row['count_residues'] = len(domain_group) # Total rows for this domain (residues * temps)
        actual_rmsf_domain = domain_group[TARGET_COL].dropna()
        row['actual_mean'] = actual_rmsf_domain.mean() if not actual_rmsf_domain.empty else np.nan
        row['actual_stddev'] = actual_rmsf_domain.std() if len(actual_rmsf_domain) > 1 else 0.0

        # Calculate additional metrics (RMSE, R2, MedAE, Prediction StdDev) per model
        for report_name, config in MODEL_CONFIG.items():
             pred_col = config['pred_col']
             if pred_col in domain_group.columns:
                  aligned_df = domain_group[[TARGET_COL, pred_col]].dropna()
                  if len(aligned_df) > 1:
                       y_true_dom = aligned_df[TARGET_COL].values
                       y_pred_dom = aligned_df[pred_col].values
                       try: row[f"{report_name}_rmse"] = np.sqrt(mean_squared_error(y_true_dom, y_pred_dom))
                       except: row[f"{report_name}_rmse"] = np.nan
                       try: row[f"{report_name}_r2"] = r2_score(y_true_dom, y_pred_dom) if np.var(y_true_dom) > 1e-9 else np.nan
                       except: row[f"{report_name}_r2"] = np.nan
                       try: row[f"{report_name}_medae"] = median_absolute_error(y_true_dom, y_pred_dom)
                       except: row[f"{report_name}_medae"] = np.nan
                       row[f"{report_name}_pred_stddev"] = y_pred_dom.std() if len(y_pred_dom) > 1 else 0.0
                  else: # Fill with NaN if not enough data
                       for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: row[f"{report_name}_{metric}"] = np.nan
             else: # Fill with NaN if prediction column missing
                  for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: row[f"{report_name}_{metric}"] = np.nan
             # Add keys for potentially calculated metrics
             for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: all_metric_keys.add(f"{report_name}_{metric}")


             # Calculate per-domain uncertainty metrics
             unc_col = config.get('unc_col')
             error_col = f"{report_name}_abs_error" # Need error calculated first
             if error_col not in domain_group.columns and TARGET_COL in domain_group.columns and pred_col in domain_group.columns:
                   # Calculate error on the fly for this domain group
                   domain_group[error_col] = (domain_group[pred_col] - domain_group[TARGET_COL]).abs()


             if unc_col and unc_col in domain_group.columns and error_col in domain_group.columns:
                 unc_valid = domain_group[[unc_col, error_col]].dropna()
                 if not unc_valid.empty:
                      row[f"{report_name}_avg_uncertainty"] = unc_valid[unc_col].mean()
                      if len(unc_valid) > 1:
                           corr_unc_err, _ = safe_pearsonr(unc_valid[unc_col], unc_valid[error_col])
                           row[f"{report_name}_uncertainty_error_corr"] = corr_unc_err
                           try: row[f"{report_name}_within_1std"] = np.mean(unc_valid[error_col] <= unc_valid[unc_col]) * 100
                           except: row[f"{report_name}_within_1std"] = np.nan
                      else: # Fill NaN if not enough data for corr/calibration
                           row[f"{report_name}_uncertainty_error_corr"] = np.nan
                           row[f"{report_name}_within_1std"] = np.nan
                 else: # Fill NaN if no valid uncertainty/error pairs
                      row[f"{report_name}_avg_uncertainty"] = np.nan
                      row[f"{report_name}_uncertainty_error_corr"] = np.nan
                      row[f"{report_name}_within_1std"] = np.nan
             else: # Fill NaN if no uncertainty column for this model
                  row[f"{report_name}_avg_uncertainty"] = np.nan
                  row[f"{report_name}_uncertainty_error_corr"] = np.nan
                  row[f"{report_name}_within_1std"] = np.nan
             # Add uncertainty keys
             for unc_metric in ['avg_uncertainty', 'uncertainty_error_corr', 'within_1std']:
                  all_metric_keys.add(f"{report_name}_{unc_metric}")


             # Calculate temperature-specific metrics per domain
             if 'temperature' in domain_group.columns:
                  for temp, temp_group in domain_group.groupby('temperature'):
                       temp_key_suffix = f"{temp:.1f}K" # Consistent key format
                       mae_t_val, pcc_t_val = np.nan, np.nan
                       if pred_col in temp_group.columns:
                            aligned_temp_df = temp_group[[TARGET_COL, pred_col]].dropna()
                            if len(aligned_temp_df) > 1:
                                 y_true_t = aligned_temp_df[TARGET_COL].values
                                 y_pred_t = aligned_temp_df[pred_col].values
                                 try: mae_t_val = mean_absolute_error(y_true_t, y_pred_t)
                                 except: pass
                                 try: pcc_t_val, _ = safe_pearsonr(y_true_t, y_pred_t)
                                 except: pass
                       row[f"{report_name}_mae_{temp_key_suffix}"] = mae_t_val
                       row[f"{report_name}_pcc_{temp_key_suffix}"] = pcc_t_val
                       all_metric_keys.add(f"{report_name}_mae_{temp_key_suffix}")
                       all_metric_keys.add(f"{report_name}_pcc_{temp_key_suffix}")


        domain_data_list.append(row)

    if not domain_data_list:
        logger.error("No data compiled for domain-level DataFrame.")
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: No domain data compiled."
        # Ensure the internal path is not set if calculation failed
        if '_internal_domain_metrics_path' in analysis_results:
            del analysis_results['_internal_domain_metrics_path']
        return

    try:
        domain_df = pd.DataFrame(domain_data_list)
        # Check if domain_id was added correctly before setting index
        if 'domain_id' in domain_df.columns and domain_df['domain_id'].notna().all():
             domain_df = domain_df.set_index('domain_id')
        else:
             logger.warning("Could not set 'domain_id' as index (missing or contains NaNs). Using default index.")

        # Ensure all potential metric columns exist before calculating mean
        for key in all_metric_keys:
            if key not in domain_df.columns:
                domain_df[key] = np.nan

        # Calculate mean values across domains (skipna=True is default for mean)
        mean_domain_metrics = domain_df.mean().to_frame()
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = format_table(mean_domain_metrics)

        # --- Save domain_df for Case Study Selection ---
        main_output_dir = os.path.dirname(args.output_file) # Use args from main scope
        domain_analysis_dir = os.path.join(main_output_dir, "domain_analysis")
        os.makedirs(domain_analysis_dir, exist_ok=True)
        domain_metrics_path = os.path.join(domain_analysis_dir, "calculated_domain_metrics.csv")
        domain_df.to_csv(domain_metrics_path) # Save the DataFrame
        logger.info(f"Saved detailed domain metrics to: {domain_metrics_path}")
        # Store path for later use by case study selection
        analysis_results['_internal_domain_metrics_path'] = domain_metrics_path

    except Exception as e:
        logger.error(f"Error calculating, formatting, or saving domain metrics: {e}", exc_info=True)
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = f"Error: {e}"
        # Ensure internal path is cleared on error
        if '_internal_domain_metrics_path' in analysis_results:
            del analysis_results['_internal_domain_metrics_path']
            
# --- Performance vs Actual RMSF (Robust Version) ---
def calculate_performance_vs_actual_rmsf(df, target_col='rmsf', model_pred_col='Attention_ESM_rmsf', n_bins=10):
    """Calculates performance metrics binned by actual RMSF magnitude."""
    logger.info(f"Analyzing performance vs actual '{target_col}' magnitude for model prediction '{model_pred_col}'...")
    results = []
    required_cols = [target_col, model_pred_col]

    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"DataFrame missing required columns for RMSF binning analysis: {missing}. Skipping.")
        return pd.DataFrame()

    # Drop rows where target or prediction is NaN before binning
    df_valid = df[required_cols].dropna().copy()
    if df_valid.empty or len(df_valid) < n_bins * 2: # Need enough points for meaningful bins/metrics
        logger.warning(f"Not enough valid data points ({len(df_valid)}) found for RMSF binning analysis (need at least {n_bins*2}). Skipping.")
        return pd.DataFrame()

    try:
        # Create quantiles based on the actual RMSF
        df_valid['rmsf_quantile_bin'], bin_edges = pd.qcut(df_valid[target_col], q=n_bins, labels=False, duplicates='drop', retbins=True)
        actual_num_bins = df_valid['rmsf_quantile_bin'].nunique() # Number of bins actually created

        # Check if the number of bins is sufficient
        if actual_num_bins < 2:
             logger.warning(f"Could not create at least 2 distinct RMSF bins (only {actual_num_bins} created). Skipping analysis.")
             return pd.DataFrame()
        if actual_num_bins < n_bins:
            logger.warning(f"Created only {actual_num_bins} bins due to duplicate edges in actual RMSF distribution (requested {n_bins}).")

        logger.info(f"Created {actual_num_bins} bins based on actual RMSF quantiles.")

        # Use bin centers derived from actual edges for labels
        # Handle potential issues if bin_edges length doesn't match actual_num_bins+1
        if len(bin_edges) != actual_num_bins + 1:
             logger.warning("Length of bin_edges does not match number of bins + 1. Cannot calculate bin centers accurately.")
             bin_centers = [np.nan] * actual_num_bins
             bin_range_labels = {i: "N/A" for i in range(actual_num_bins)}
        else:
             bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(actual_num_bins)]
             # Map bin indices back to the range strings for clarity in the output table
             bin_range_labels = {i: f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(actual_num_bins)}


        grouped = df_valid.groupby('rmsf_quantile_bin')

        for bin_label_float, group in grouped:
             bin_label = int(bin_label_float) # Convert bin label (which is float from NaN check) to int index
             # Check if bin_label is valid index for labels/centers
             if bin_label >= len(bin_centers):
                 logger.warning(f"Invalid bin label {bin_label} encountered. Skipping.")
                 continue
             if len(group) < 2: continue # Need >= 2 points for metrics

             actual_bin = group[target_col].values
             predicted_bin = group[model_pred_col].values

             # Calculate metrics safely within the bin
             mae, pcc, rmse, r2 = np.nan, np.nan, np.nan, np.nan # Default to NaN
             try: mae = mean_absolute_error(actual_bin, predicted_bin)
             except: pass
             try: pcc, _ = safe_pearsonr(actual_bin, predicted_bin)
             except: pass
             try: rmse = np.sqrt(mean_squared_error(actual_bin, predicted_bin))
             except: pass
             try: r2 = r2_score(actual_bin, predicted_bin) if np.var(actual_bin) > 1e-9 else np.nan
             except: pass

             results.append({
                 'RMSF_Quantile_Bin': bin_label,
                 'RMSF_Range': bin_range_labels.get(bin_label, "N/A"), # Get bin range string
                 'Bin_Center_Approx': bin_centers[bin_label], # Add approx center
                 'Count': len(group),
                 'Mean_Actual_RMSF': np.mean(actual_bin),
                 'MAE': mae,
                 'PCC': pcc,
                 'RMSE': rmse,
                 'R2': r2
             })

    except ValueError as ve:
         if "Bin edges must be unique" in str(ve) or "cannot specify fewer than 2 bins" in str(ve):
              logger.warning(f"Could not perform RMSF binning due to data distribution: {ve}. Skipping.")
              return pd.DataFrame()
         else:
              logger.error(f"ValueError during RMSF binning analysis: {ve}", exc_info=True)
              return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error during RMSF binning analysis: {e}", exc_info=True)
        return pd.DataFrame() # Return empty on error

    if not results:
         logger.warning("No results generated from RMSF binning analysis.")

    return pd.DataFrame(results).sort_values('RMSF_Quantile_Bin')


# --- Existing Analysis Functions (Keep as is or adapt slightly) ---
def run_basic_info(df, analysis_results):
    logger.info("Running basic info analysis...")
    analysis_results['1. BASIC INFORMATION'] = {}
    analysis_results['1. BASIC INFORMATION']['Source Data File'] = df.attrs.get('source_file', 'N/A')
    analysis_results['1. BASIC INFORMATION']['Analysis Timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    analysis_results['1. BASIC INFORMATION']['Total Rows'] = len(df)
    analysis_results['1. BASIC INFORMATION']['Total Columns'] = len(df.columns)
    try:
        if 'domain_id' in df.columns:
            analysis_results['1. BASIC INFORMATION']['Unique Domains'] = df['domain_id'].nunique()
        else:
            analysis_results['1. BASIC INFORMATION']['Unique Domains'] = "N/A (column missing)"
    except Exception as e:
        logger.warning(f"Could not count unique domains: {e}")
        analysis_results['1. BASIC INFORMATION']['Unique Domains'] = "Error"
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        if memory_bytes < 1024**2: mem_str = f"{memory_bytes/1024:.2f} KB"
        elif memory_bytes < 1024**3: mem_str = f"{memory_bytes/1024**2:.2f} MB"
        else: mem_str = f"{memory_bytes/1024**3:.2f} GB"
        analysis_results['1. BASIC INFORMATION']['Memory Usage'] = mem_str
    except Exception as e:
        logger.warning(f"Could not estimate memory usage: {e}")
        analysis_results['1. BASIC INFORMATION']['Memory Usage'] = "N/A"

    # Add Model Key Table
    model_key_data = []
    for report_name, config in MODEL_CONFIG.items():
        model_key_data.append({
            "Report Name": report_name,
            "Original Key": config['key'],
            "Prediction Column": config['pred_col'],
            "Uncertainty Column": config.get('unc_col', 'N/A')
        })
    # Use format_table's robustness
    analysis_results['1.1 MODEL KEY'] = format_table(model_key_data, headers="keys", tablefmt="pipe")


def run_missing_values(df, analysis_results):
    logger.info("Running missing value analysis...")
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame({'count': missing.astype(int), 'percentage': (missing / len(df)) * 100})
            analysis_results['2. MISSING VALUE SUMMARY'] = format_table(missing_df.sort_values('count', ascending=False), floatfmt=".2f")
        else:
            analysis_results['2. MISSING VALUE SUMMARY'] = "No missing values found."
    except Exception as e:
         logger.error(f"Error during missing value analysis: {e}")
         analysis_results['2. MISSING VALUE SUMMARY'] = f"Error: {e}"


def run_descriptive_stats(df, analysis_results):
    logger.info("Running descriptive statistics...")
    try:
        # Select only numeric columns for describe()
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
             desc_stats = df[numeric_cols].describe().transpose()
             analysis_results['3. OVERALL DESCRIPTIVE STATISTICS'] = format_table(desc_stats)
        else:
             analysis_results['3. OVERALL DESCRIPTIVE STATISTICS'] = "No numeric columns found for descriptive statistics."
    except Exception as e:
        logger.error(f"Error during descriptive statistics: {e}")
        analysis_results['3. OVERALL DESCRIPTIVE STATISTICS'] = f"Error: {e}"


def run_data_distributions(df, analysis_results):
    logger.info("Analyzing data distributions...")
    analysis_results['4. DATA DISTRIBUTIONS'] = "Counts based on all rows."
    distribution_cols = {
        'temperature': '4.1 TEMPERATURE DISTRIBUTION',
        'resname': '4.2 RESNAME DISTRIBUTION',
        'core_exterior_encoded': '4.3 CORE/EXTERIOR DISTRIBUTION',
        'secondary_structure_encoded': '4.4 SECONDARY STRUCTURE (H/E/L) DISTRIBUTION'
    }
    ss_label_map = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
    core_label_map = {0: 'core', 1: 'exterior'}

    for col, section_title in distribution_cols.items():
        if col in df.columns:
            try:
                counts = df[col].value_counts()
                percent = (counts / len(df)) * 100
                dist_df = pd.DataFrame({'Percent': percent, 'Count': counts.astype(int)})

                # Apply labels if needed
                if col == 'secondary_structure_encoded':
                    dist_df.index = dist_df.index.map(ss_label_map).fillna('Unknown')
                elif col == 'core_exterior_encoded':
                    dist_df.index = dist_df.index.map(core_label_map).fillna('Unknown')

                analysis_results[section_title] = format_table(dist_df.sort_values('Percent', ascending=False), floatfmt=".2f")
            except Exception as e:
                 logger.error(f"Error analyzing distribution for column '{col}': {e}")
                 analysis_results[section_title] = f"Error calculating distribution for '{col}'."
        else:
            analysis_results[section_title] = f"Column '{col}' not found."

def run_model_comparison(df, analysis_results):
    logger.info("Comparing model performance...")
    analysis_results['5. COMPREHENSIVE MODEL COMPARISON'] = "Comparing performance across ALL detected models."
    target_col = TARGET_COL
    metrics_overall = []
    metrics_by_temp_dict = defaultdict(lambda: defaultdict(list)) # Use defaultdict

    model_names_in_df = [cfg['pred_col'] for cfg in MODEL_CONFIG.values() if cfg['pred_col'] in df.columns]
    logger.info(f"Models found for comparison: {[k for k,v in MODEL_CONFIG.items() if v['pred_col'] in model_names_in_df]}")

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found. Skipping model comparison.")
        analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = "Target column missing."
        analysis_results['5.2 PERFORMANCE METRICS BY TEMPERATURE'] = "Target column missing."
        analysis_results['5.3 PREDICTION R-SQUARED MATRIX'] = "Target column missing."
        analysis_results['5.4 ABSOLUTE ERROR R-SQUARED MATRIX'] = "Target column missing."
        return

    df_errors = df[[target_col]].copy() # For error correlation

    for report_name, config in MODEL_CONFIG.items():
        pred_col = config['pred_col']
        if pred_col in df.columns:
            # --- Overall Metrics ---
            df_valid_overall = df[[target_col, pred_col]].dropna()
            if len(df_valid_overall) > 1:
                y_true = df_valid_overall[target_col].values
                y_pred = df_valid_overall[pred_col].values
                try:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae = mean_absolute_error(y_true, y_pred)
                    medae = median_absolute_error(y_true, y_pred)
                    pcc, _ = safe_pearsonr(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 1e-9 else np.nan
                    metrics_overall.append({'Model': report_name, 'rmse': rmse, 'mae': mae, 'medae': medae, 'pcc': pcc, 'r2': r2})
                    # Calculate and store errors for correlation matrix
                    df_errors[f"{report_name}_abs_error"] = (pd.Series(y_pred, index=df_valid_overall.index) - pd.Series(y_true, index=df_valid_overall.index)).abs()
                except Exception as e:
                    logger.warning(f"Could not calculate overall metrics for {report_name}: {e}")
            else: logger.warning(f"Not enough valid data ({len(df_valid_overall)}) for overall metrics for {report_name}")

            # --- Metrics by Temperature ---
            if 'temperature' in df.columns:
                for temp, group in df.groupby('temperature'):
                    df_valid_temp = group[[target_col, pred_col]].dropna()
                    if len(df_valid_temp) > 1:
                         y_true_t = df_valid_temp[target_col].values
                         y_pred_t = df_valid_temp[pred_col].values
                         try:
                             mae_t = mean_absolute_error(y_true_t, y_pred_t)
                             pcc_t, _ = safe_pearsonr(y_true_t, y_pred_t)
                             r2_t = r2_score(y_true_t, y_pred_t) if np.var(y_true_t) > 1e-9 else np.nan
                             metrics_by_temp_dict[temp][f'{report_name}_mae'].append(mae_t)
                             metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(pcc_t)
                             metrics_by_temp_dict[temp][f'{report_name}_r2'].append(r2_t)
                         except Exception as e:
                              logger.warning(f"Could not calculate metrics for {report_name} at T={temp}: {e}")
                              metrics_by_temp_dict[temp][f'{report_name}_mae'].append(np.nan)
                              metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(np.nan)
                              metrics_by_temp_dict[temp][f'{report_name}_r2'].append(np.nan)
                    else: # Append NaNs if not enough data at this temp for this model
                        metrics_by_temp_dict[temp][f'{report_name}_mae'].append(np.nan)
                        metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(np.nan)
                        metrics_by_temp_dict[temp][f'{report_name}_r2'].append(np.nan)
        else:
            logger.warning(f"Prediction column '{pred_col}' for model '{report_name}' not found. Skipping comparison.")


    # Format Overall Metrics Table
    if metrics_overall:
        overall_df = pd.DataFrame(metrics_overall).set_index('Model')
        analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = format_table(overall_df.sort_values('pcc', ascending=False))
    else: analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = "No models found or metrics calculable."

    # Format Metrics by Temperature Table
    if metrics_by_temp_dict:
        temp_metrics_list = []
        all_temp_cols = set() # Track all columns generated across temperatures
        for temp, metrics in sorted(metrics_by_temp_dict.items()):
             # Count valid rows at this temperature *before* dropping NaNs for metrics
            row_count = len(df[np.isclose(df['temperature'], temp)]) if 'temperature' in df.columns else 'N/A'
            row = {'temperature': temp, 'count': row_count}
            for metric_col, metric_list in metrics.items():
                # Average the metric lists (should ideally contain one value per model per temp)
                # Use nanmean to ignore NaNs from failed calculations
                avg_metric = np.nanmean(metric_list)
                row[metric_col] = avg_metric
                all_temp_cols.add(metric_col)
            temp_metrics_list.append(row)

        temp_df = pd.DataFrame(temp_metrics_list).set_index('temperature')
        # Ensure all potential columns exist
        for col in all_temp_cols:
             if col not in temp_df.columns: temp_df[col] = np.nan
        # Sort columns for better readability
        cols_ordered = ['count'] + sorted([col for col in temp_df.columns if col != 'count'], key=lambda x: (MODEL_CONFIG.get(x.split('_')[0], {}).get('key', x.split('_')[0]), x.split('_')[-1])) # Sort by model name then metric
        analysis_results['5.2 PERFORMANCE METRICS BY TEMPERATURE'] = format_table(temp_df[cols_ordered])

        # Add summary comparison highlights
        try:
             temp_summary_lines = ["Model Comparison Highlights by Temperature:"]
             for temp in sorted(metrics_by_temp_dict.keys()):
                 row_data = temp_df.loc[temp]
                 mae_cols = [col for col in row_data.index if col.endswith('_mae') and pd.notna(row_data[col])]
                 pcc_cols = [col for col in row_data.index if col.endswith('_pcc') and pd.notna(row_data[col])]
                 if not mae_cols or not pcc_cols: continue

                 best_mae_model_col = row_data[mae_cols].idxmin()
                 best_pcc_model_col = row_data[pcc_cols].idxmax()
                 # Extract model name from column name robustly
                 best_mae_model = next((name for name, cfg in MODEL_CONFIG.items() if cfg['pred_col'].replace('_rmsf','_mae') == best_mae_model_col), best_mae_model_col.replace('_mae',''))
                 best_pcc_model = next((name for name, cfg in MODEL_CONFIG.items() if cfg['pred_col'].replace('_rmsf','_pcc') == best_pcc_model_col), best_pcc_model_col.replace('_pcc',''))

                 temp_summary_lines.append(f"  T={temp:.1f}K (N={int(row_data.get('count',0))}): Best MAE={best_mae_model}({row_data[best_mae_model_col]:.3f}), Best PCC={best_pcc_model}({row_data[best_pcc_model_col]:.3f})")
             analysis_results['5.2.1 TEMP PERFORMANCE SUMMARY'] = "\n".join(temp_summary_lines)
        except Exception as e:
             logger.error(f"Error generating temperature performance summary: {e}")
             analysis_results['5.2.1 TEMP PERFORMANCE SUMMARY'] = "Error generating summary."


    else: analysis_results['5.2 PERFORMANCE METRICS BY TEMPERATURE'] = "No temperature data or metrics calculable."

    # Prediction Correlation Matrix
    pred_cols_present = [config['pred_col'] for config in MODEL_CONFIG.values() if config['pred_col'] in df.columns]
    if len(pred_cols_present) > 1:
        try:
            # Calculate correlation on valid pairs only
            pred_corr = df[[target_col] + pred_cols_present].dropna().corr(method='pearson')**2 # R-squared
            # Rename columns for readability
            rename_map_pred = {target_col: 'Actual'}
            for report_name, config in MODEL_CONFIG.items():
                 if config['pred_col'] in pred_corr.columns: rename_map_pred[config['pred_col']] = report_name
            pred_corr = pred_corr.rename(columns=rename_map_pred, index=rename_map_pred)
            analysis_results['5.3 PREDICTION R-SQUARED MATRIX (COEFFICIENT OF DETERMINATION, INCL. ACTUAL)'] = format_table(pred_corr)
        except Exception as e:
             logger.error(f"Error calculating prediction correlation: {e}")
             analysis_results['5.3 PREDICTION R-SQUARED MATRIX'] = f"Error calculating prediction correlation: {e}"
    else: analysis_results['5.3 PREDICTION R-SQUARED MATRIX'] = "Not enough models with predictions to correlate."

    # Error Correlation Matrix
    error_cols_present = [col for col in df_errors.columns if col.endswith('_abs_error')]
    if len(error_cols_present) > 1:
        try:
             # Calculate correlation on valid pairs only
            error_corr = df_errors[[target_col] + error_cols_present].dropna().corr(method='pearson')**2 # R-squared
            # Rename columns for readability
            rename_map_err = {target_col: 'rmsf'} # Keep target as rmsf for clarity
            # error_corr = error_corr.rename(columns=rename_map_err, index=rename_map_err) # Apply rename
            analysis_results['5.4 ABSOLUTE ERROR R-SQUARED MATRIX (COEFFICIENT OF DETERMINATION, INCL. ACTUAL)'] = \
                "(Shows squared correlation (R^2) between errors. High value means models tend to make errors on the same samples. R^2 between errors and Actual RMSF is also shown.)\n" + \
                 format_table(error_corr)
        except Exception as e:
             logger.error(f"Error calculating error correlation: {e}")
             analysis_results['5.4 ABSOLUTE ERROR R-SQUARED MATRIX'] = f"Error calculating error correlation: {e}"
    else: analysis_results['5.4 ABSOLUTE ERROR R-SQUARED MATRIX'] = "Not enough models with errors to correlate."


def run_uncertainty_analysis(df, analysis_results):
    logger.info("Running uncertainty analysis...")
    analysis_results['6. UNCERTAINTY ANALYSIS'] = "Comparing uncertainty estimates for models where available."
    uncertainty_stats = []
    uncertainty_error_corr = []
    calibration_check = []
    mae_vs_unc_bins = {}

    target_col = TARGET_COL
    # primary_model_report_name = "DeepFlex" # Defined globally if needed, or pass as arg
    # primary_model_key = MODEL_CONFIG.get(primary_model_report_name, {}).get("key") # Not needed here
    # primary_model_abs_error_col = f"{primary_model_report_name}_abs_error" # Standard name

    # Calculate absolute error columns if they don't exist
    df_copy = df.copy() # Work on a copy
    models_with_uncertainty = []
    for report_name, config in MODEL_CONFIG.items():
        pred_col = config['pred_col']
        unc_col = config.get('unc_col')
        error_col = f"{report_name}_abs_error"

        # Ensure error column exists
        if error_col not in df_copy.columns:
            if target_col in df_copy.columns and pred_col in df_copy.columns:
                logger.debug(f"Calculating {error_col} for uncertainty analysis.")
                df_copy[error_col] = (df_copy[pred_col] - df_copy[target_col]).abs()
            else:
                logger.warning(f"Cannot calculate error for {report_name}, skipping uncertainty analysis for it.")
                continue # Skip this model if error cannot be calculated

        # Check if uncertainty column exists for this model
        if unc_col and unc_col in df_copy.columns:
             models_with_uncertainty.append(report_name)
             df_valid = df_copy[[error_col, unc_col]].dropna() # Use calculated error col

             if not df_valid.empty:
                 # Stats
                 stats = df_valid[unc_col].describe()
                 # Add type check and conversion for safety when adding to list
                 safe_stats = {k: float(v) if isinstance(v, (np.number, int, float)) else str(v) for k, v in stats.to_dict().items()}
                 uncertainty_stats.append({'Model': report_name, **safe_stats})

                 # Correlation with error
                 abs_error = df_valid[error_col].values
                 unc_values = df_valid[unc_col].values
                 if len(abs_error) > 1:
                     corr, _ = safe_pearsonr(unc_values, abs_error) # Handles internal checks
                     uncertainty_error_corr.append({'Model': report_name, 'Uncertainty-Error PCC': corr})
                 else:
                      uncertainty_error_corr.append({'Model': report_name, 'Uncertainty-Error PCC': np.nan})

                 # Calibration Check
                 try:
                     within_1_std = np.mean(abs_error <= unc_values) * 100
                 except TypeError: # Handle potential type issues if NaN comparison fails
                      within_1_std = np.nan
                 calibration_check.append({'Model': report_name, '% within 1 Uncertainty': within_1_std})

                 # MAE vs Uncertainty Bins
                 try:
                     df_valid['unc_quantile'], bin_edges = pd.qcut(df_valid[unc_col], q=10, labels=False, duplicates='drop', retbins=True)
                     actual_num_bins = df_valid['unc_quantile'].nunique()
                     if actual_num_bins >= 2:
                          bin_range_labels = {i: f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)}
                          binned_mae = df_valid.groupby('unc_quantile')['abs_error'].agg(['mean', 'median', 'count']).reset_index()
                          binned_mae['unc_bin_range'] = binned_mae['unc_quantile'].apply(lambda x: bin_range_labels.get(int(x),"N/A"))
                          mae_vs_unc_bins[report_name] = format_table(binned_mae[['unc_bin_range','mean','median','count']].rename(columns={'unc_bin_range':f'{report_name}_Unc_Quantile'}))
                     else: logger.warning(f"Could not create enough bins for MAE vs Uncertainty for {report_name}")

                 except Exception as e:
                     logger.warning(f"Could not calculate MAE vs uncertainty bins for {report_name}: {e}")

             else: logger.warning(f"No valid data found for uncertainty analysis of {report_name}")


    # Format and store results
    if uncertainty_stats:
        analysis_results['6.1 UNCERTAINTY DISTRIBUTION STATISTICS'] = format_table(pd.DataFrame(uncertainty_stats).set_index('Model'))
    else: analysis_results['6.1 UNCERTAINTY DISTRIBUTION STATISTICS'] = "No models with uncertainty data found."

    if uncertainty_error_corr:
        analysis_results['6.2 OVERALL UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION'] = "(Positive correlation is desired.)\n" + format_table(pd.DataFrame(uncertainty_error_corr).set_index('Model'))
    else: analysis_results['6.2 OVERALL UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION'] = "No models with uncertainty data found or correlation calculable."

    if calibration_check:
        analysis_results['6.3 SIMPLE CALIBRATION CHECK'] = "(Expected ~68.2% for well-calibrated Gaussian uncertainty.)\n" + format_table(pd.DataFrame(calibration_check).set_index('Model'), floatfmt=".2f")
    else: analysis_results['6.3 SIMPLE CALIBRATION CHECK'] = "No models with uncertainty data found or calibration calculable."

    for model_name, table_str in mae_vs_unc_bins.items():
        analysis_results[f'6.4 MEAN ABSOLUTE ERROR BINNED BY {model_name.upper()} UNCERTAINTY QUANTILE'] = table_str
    if not mae_vs_unc_bins:
         analysis_results['6.4 MEAN ABSOLUTE ERROR BINNED BY UNCERTAINTY QUANTILE'] = "No results generated (check data/binning)."


def run_stratified_performance(df, analysis_results):
    logger.info("Running stratified performance analysis...")
    target_col = TARGET_COL
    # Define strata columns and their section titles
    strata_definitions = {
        'resname': ('8. AMINO ACID PERFORMANCE', None),
        'normalized_resid': ('9. PERFORMANCE BY NORMALIZED_RESID BIN', 5),
        'core_exterior_encoded': ('10. PERFORMANCE BY CORE/EXTERIOR', None),
        'secondary_structure_encoded': ('11. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)', None),
        'relative_accessibility': ('13. PERFORMANCE BY REL. ACCESSIBILITY QUANTILE', 5),
        'bfactor_norm': ('14. PERFORMANCE BY BFACTOR_NORM QUANTILE', 5)
        # Add 'rmsf' here if desired: 'rmsf': ('12. PERFORMANCE BY ACTUAL RMSF QUANTILE', 10)
    }
    # Map encoded values to readable labels
    label_maps = {
        'secondary_structure_encoded': {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'},
        'core_exterior_encoded': {0: 'core', 1: 'exterior'}
    }

    for strat_col, (section_title, n_bins) in strata_definitions.items():
        if strat_col not in df.columns:
            analysis_results[section_title] = f"Column '{strat_col}' not found."
            continue

        logger.info(f"Stratifying by '{strat_col}'...")
        strata_results = []
        df_copy = df.copy() # Work on a copy

        grouping_col = strat_col
        label_map_to_use = label_maps.get(strat_col)
        sort_by_index = False # Default sort by performance
        final_bin_labels = None # Store final labels for sorting

        # --- Bin continuous data ---
        if n_bins is not None and n_bins > 0:
            bin_col_name = f"{strat_col}_bin"
            grouping_col = bin_col_name
            try:
                df_copy[bin_col_name], bin_edges = pd.qcut(df_copy[strat_col].dropna(), q=n_bins, labels=False, duplicates='drop', retbins=True)
                actual_bins = df_copy[bin_col_name].nunique()
                if actual_bins < 2:
                     logger.warning(f"Could not create at least 2 bins for '{strat_col}'. Skipping stratification.")
                     analysis_results[section_title] = f"Could not create sufficient bins for '{strat_col}'."
                     continue

                # Create string labels from edges
                final_bin_labels = {i: f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)}
                # Apply labels to the column used for grouping
                df_copy[bin_col_name] = df_copy[bin_col_name].map(final_bin_labels)
                label_map_to_use = None # Labels are now directly in the column
                sort_by_index = False # Sort by the bin range string naturally
                logger.debug(f"Binned '{strat_col}' into {actual_bins} bins.")

            except Exception as e:
                logger.error(f"Failed to bin column '{strat_col}': {e}. Skipping.")
                analysis_results[section_title] = f"Error binning column '{strat_col}'."
                continue

        # --- Group and Aggregate ---
        # Drop rows where the grouping column is NaN (can happen after qcut if values were outside range)
        df_copy.dropna(subset=[grouping_col], inplace=True)
        if df_copy.empty:
            logger.warning(f"DataFrame empty after dropping NaN group labels for '{grouping_col}'. Skipping.")
            analysis_results[section_title] = f"No data after creating groups/bins for '{strat_col}'."
            continue

        grouped = df_copy.groupby(grouping_col, observed=False) # Use observed=False for categorical bins

        for group_name, group_df in grouped:
             if len(group_df) < 2: continue # Need min samples for reliable metrics

             row = {grouping_col: group_name, 'count': len(group_df)}

             for report_name, config in MODEL_CONFIG.items():
                  pred_col = config['pred_col']
                  mae_val, pcc_val = np.nan, np.nan # Default to NaN
                  if pred_col in group_df.columns:
                       df_valid = group_df[[target_col, pred_col]].dropna()
                       if len(df_valid) > 1:
                            y_true = df_valid[target_col].values
                            y_pred = df_valid[pred_col].values
                            try: mae_val = mean_absolute_error(y_true, y_pred)
                            except: pass
                            try: pcc_val, _ = safe_pearsonr(y_true, y_pred)
                            except: pass
                  row[f'{report_name}_mae'] = mae_val
                  row[f'{report_name}_pcc'] = pcc_val

             strata_results.append(row)

        # --- Format and Save ---
        if strata_results:
             strata_df = pd.DataFrame(strata_results)
             # Apply label mapping if defined and not already applied via binning
             if label_map_to_use:
                  strata_df[grouping_col] = strata_df[grouping_col].map(label_map_to_use).fillna(strata_df[grouping_col])

             # Set index and sort
             strata_df = strata_df.set_index(grouping_col)
             index_name = section_title.split(" BY ")[-1] # Get descriptive name
             strata_df.index.name = index_name

             # Determine sort order
             if sort_by_index: # Sort by bin index if applicable
                 strata_df = strata_df.sort_index()
             else: # Default: sort by primary model MAE if available
                 primary_mae_col = f'{MODEL_CONFIG["DeepFlex"]["key"]}_mae' # Construct expected MAE col name
                 if primary_mae_col in strata_df.columns:
                      strata_df = strata_df.sort_values(primary_mae_col)

             analysis_results[section_title] = format_table(strata_df)
        else: analysis_results[section_title] = f"No results after grouping by '{strat_col}'."



def run_performance_vs_actual_rmsf(df, analysis_results, primary_model_report_name="DeepFlex"):
    logger.info("Running performance vs actual RMSF analysis...")
    section_title = "12. PERFORMANCE BY ACTUAL RMSF QUANTILE" # Define section title
    primary_model_config = MODEL_CONFIG.get(primary_model_report_name)
    if not primary_model_config:
        logger.error(f"Primary model '{primary_model_report_name}' not found in MODEL_CONFIG.")
        analysis_results[section_title] = "Primary model config missing."
        return

    pred_col = primary_model_config['pred_col']
    binned_df = calculate_performance_vs_actual_rmsf(df, target_col=TARGET_COL, model_pred_col=pred_col, n_bins=10)

    if not binned_df.empty:
        # Rename columns for final report
        binned_df.rename(columns={'MAE': f'{primary_model_report_name}_MAE',
                                 'PCC': f'{primary_model_report_name}_PCC',
                                 'RMSE': f'{primary_model_report_name}_RMSE',
                                 'R2': f'{primary_model_report_name}_R2'}, inplace=True)
        # Set index for better table formatting
        binned_df = binned_df.set_index('RMSF_Range')
        analysis_results[section_title] = format_table(binned_df.drop(columns=['RMSF_Quantile_Bin','Bin_Center_Approx'])) # Drop helper columns
    else:
        analysis_results[section_title] = "No results generated or error during binning."


def run_model_disagreement_analysis(df, analysis_results, primary_model_report_name="DeepFlex"):
    logger.info("Running model disagreement analysis...")
    section_title = "15. MODEL DISAGREEMENT VS. ERROR" # Define section title
    pred_cols = [config['pred_col'] for config in MODEL_CONFIG.values() if config['pred_col'] in df.columns]
    if len(pred_cols) < 2:
        analysis_results[section_title] = "Need at least 2 models with predictions."
        return

    # Calculate std dev across model predictions for each residue/temp
    df_preds_only = df[pred_cols].copy()
    # Convert to numeric, coercing errors, before calculating std dev
    for col in df_preds_only.columns:
        df_preds_only[col] = pd.to_numeric(df_preds_only[col], errors='coerce')

    # Use skipna=True to handle potential missing predictions for some models on some rows
    df['prediction_stddev'] = df_preds_only.std(axis=1, skipna=True)

    # Add stats on the disagreement column
    try:
        stats = df['prediction_stddev'].describe().to_frame()
        analysis_results[section_title] = "MODEL PREDICTION STANDARD DEVIATION STATS\n" + format_table(stats) + "\n\nNote: Std Dev calculated across available model predictions.\n"
    except Exception as e:
         logger.warning(f"Could not calculate disagreement stats: {e}")
         analysis_results[section_title] = "Could not calculate disagreement stats.\n"


    # Correlate disagreement with primary model error and uncertainty
    primary_model_config = MODEL_CONFIG.get(primary_model_report_name)
    if primary_model_config:
         primary_pred_col = primary_model_config['pred_col']
         primary_unc_col = primary_model_config.get('unc_col')
         error_col = f"{primary_model_report_name}_abs_error" # Consistent error col name

         # Ensure error column exists (calculate if needed)
         df_copy = df.copy() # Use copy to avoid modifying original df used by other analyses
         if error_col not in df_copy.columns:
              if TARGET_COL in df_copy.columns and primary_pred_col in df_copy.columns:
                   df_copy[error_col] = (df_copy[primary_pred_col] - df_copy[TARGET_COL]).abs()
              # else: error col remains missing

         corr_lines = []
         # Correlation with Absolute Error
         if error_col in df_copy.columns and 'prediction_stddev' in df_copy.columns:
              df_valid_corr = df_copy[[error_col, 'prediction_stddev']].dropna()
              if len(df_valid_corr) > 1:
                   pcc_err, _ = safe_pearsonr(df_valid_corr['prediction_stddev'], df_valid_corr[error_col])
                   corr_lines.append(f"Correlation between prediction_stddev and {error_col}: {pcc_err:.3f} (N={len(df_valid_corr)})")
              else: corr_lines.append(f"Correlation between prediction_stddev and {error_col}: Not enough data.")
         else: corr_lines.append(f"Correlation with {error_col} skipped (column missing).")


         # Correlation with Uncertainty (if available)
         if primary_unc_col and primary_unc_col in df_copy.columns and 'prediction_stddev' in df_copy.columns:
              df_valid_unc = df_copy[[primary_unc_col, 'prediction_stddev']].dropna()
              if len(df_valid_unc) > 1:
                   pcc_unc, _ = safe_pearsonr(df_valid_unc['prediction_stddev'], df_valid_unc[primary_unc_col])
                   corr_lines.append(f"Correlation between prediction_stddev and {primary_model_report_name} Uncertainty: {pcc_unc:.3f} (N={len(df_valid_unc)})")
              else: corr_lines.append(f"Correlation between prediction_stddev and {primary_model_report_name} Uncertainty: Not enough data.")
         elif primary_unc_col: corr_lines.append(f"Correlation with {primary_model_report_name} Uncertainty skipped (column '{primary_unc_col}' missing).")
         else: corr_lines.append(f"Correlation with {primary_model_report_name} Uncertainty skipped (no uncertainty column defined).")


         analysis_results[section_title] += "\n".join(corr_lines)

    else: analysis_results[section_title] += "\nPrimary model for error correlation not found."


def run_case_study_candidate_selection(analysis_results, primary_model_report_name="DeepFlex", n_candidates=15):
    """Selects and formats candidate domains for case studies based on DeepFlex performance."""
    logger.info("Selecting case study candidates (Focusing on DeepFlex Metrics)...")
    section_title = "16. CASE STUDY CANDIDATES"
    analysis_results[section_title] = f"(Based on {primary_model_report_name} Metrics)"

    # --- Define DeepFlex Specific Columns to Display ---
    # Add more as needed, e.g., _rmse, _r2, _medae if calculated and saved in domain metrics
    esm_flex_cols_to_show = [
        'count_residues', # Renamed from 'count' for clarity if using the full run_domain_level_analysis
        'actual_mean',
        'actual_stddev',
        f'{primary_model_report_name}_mae',
        f'{primary_model_report_name}_pcc',
        f'{primary_model_report_name}_mae_320.0K',
        f'{primary_model_report_name}_pcc_320.0K',
        f'{primary_model_report_name}_mae_450.0K',
        f'{primary_model_report_name}_pcc_450.0K',
        'delta_mae_320_450', # Keep delta columns if calculated and needed for selection/info
        'delta_pcc_320_450',
        'delta_actual_rmsf_320_450' # Keep if calculated and informative
    ]
    # --- End Column Definition ---

    # Get the path saved by run_domain_level_analysis
    domain_metrics_path = analysis_results.get('_internal_domain_metrics_path')

    if not domain_metrics_path or not os.path.exists(domain_metrics_path):
        analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = "Domain metrics file not found, cannot select candidates."
        analysis_results['16.2 TEMP. HANDLING CANDIDATES'] = "Domain metrics file not found, cannot select candidates."
        analysis_results['16.3 CHALLENGING CANDIDATES'] = "Domain metrics file not found, cannot select candidates."
        logger.warning(f"Domain metrics file path not found or file does not exist. Skipping case study selection.")
        return

    try:
        domain_df_full = pd.read_csv(domain_metrics_path, index_col='domain_id')
        logger.info(f"Loaded domain metrics from {domain_metrics_path} for case study selection.")

        # --- Ensure Necessary Columns Exist & Calculate Deltas if Missing ---
        primary_pcc_col = f'{primary_model_report_name}_pcc'
        primary_mae_col = f'{primary_model_report_name}_mae'
        mae_320_col = f'{primary_model_report_name}_mae_320.0K'
        mae_450_col = f'{primary_model_report_name}_mae_450.0K'
        pcc_320_col = f'{primary_model_report_name}_pcc_320.0K'
        pcc_450_col = f'{primary_model_report_name}_pcc_450.0K'
        delta_mae_col = 'delta_mae_320_450'
        delta_pcc_col = 'delta_pcc_320_450'

        required_cols_exist = all(c in domain_df_full.columns for c in [primary_pcc_col, primary_mae_col])
        if not required_cols_exist:
             raise ValueError(f"Domain metrics file missing essential DeepFlex columns: {primary_pcc_col}, {primary_mae_col}")

        # Calculate deltas robustly if needed and possible
        if delta_mae_col not in domain_df_full.columns and mae_320_col in domain_df_full.columns and mae_450_col in domain_df_full.columns:
            domain_df_full[delta_mae_col] = domain_df_full[mae_450_col] - domain_df_full[mae_320_col]
        if delta_pcc_col not in domain_df_full.columns and pcc_320_col in domain_df_full.columns and pcc_450_col in domain_df_full.columns:
            domain_df_full[delta_pcc_col] = domain_df_full[pcc_450_col] - domain_df_full[pcc_320_col]
        # Note: Calculating delta_actual_rmsf here requires merging with original data again, skipped for simplicity unless already present

        # Filter display columns to only those present in the loaded DataFrame
        cols_to_display_final = [col for col in esm_flex_cols_to_show if col in domain_df_full.columns]


        # --- Candidate Selection (using try-except for robustness) ---
        # High Accuracy
        try:
            # Criteria focus on DeepFlex metrics
            high_acc = domain_df_full[ (domain_df_full[primary_pcc_col] > 0.93) &
                                       (domain_df_full[primary_mae_col] < 0.12) ]
                                       # Optional: Add back actual_stddev > 0.2 if desired and column exists
            if 'actual_stddev' in domain_df_full.columns:
                 high_acc = high_acc[high_acc['actual_stddev'] > 0.]

            # Select only DeepFlex relevant columns for the output table
            high_acc_display = high_acc[cols_to_display_final]
            analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = f"Found {len(high_acc)} candidates meeting criteria.\n" + format_table(high_acc_display.nsmallest(n_candidates, primary_mae_col))
        except Exception as e:
             logger.warning(f"Error selecting High Accuracy Candidates: {e}")
             analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = f"Error during selection: {e}"

        # Temp Handling
        try:
             required_temp_hand = [primary_pcc_col, delta_mae_col, delta_pcc_col] # Add more if needed e.g. delta_actual_rmsf
             if all(c in domain_df_full.columns for c in required_temp_hand):
                  temp_handle = domain_df_full[ (domain_df_full[primary_pcc_col] > 0.90) & # High overall PCC
                                                 # (domain_df_full['delta_actual_rmsf_320_450'].abs() > 0.2) & # Uncomment if column exists
                                                 (domain_df_full[delta_mae_col].abs() < 0.1) & # Error change is small
                                                 (domain_df_full[delta_pcc_col] > -0.05) ] # PCC doesn't drop much
                  # Select only DeepFlex relevant columns for the output table
                  temp_handle_display = temp_handle[cols_to_display_final]
                  analysis_results['16.2 TEMP. HANDLING CANDIDATES'] = f"Found {len(temp_handle)} candidates meeting criteria.\n" + format_table(temp_handle_display.nsmallest(n_candidates, delta_mae_col))
             else: analysis_results['16.2 TEMP. HANDLING CANDIDATES'] = f"Missing columns for Temp Handling criteria: {required_temp_hand}"
        except Exception as e:
             logger.warning(f"Error selecting Temp Handling Candidates: {e}")
             analysis_results['16.2 TEMP. HANDLING CANDIDATES'] = f"Error during selection: {e}"


        # Challenging
        try:
             required_challenging = [primary_pcc_col, primary_mae_col]
             if all(c in domain_df_full.columns for c in required_challenging):
                  # Define challenging condition - **ADJUSTED PCC THRESHOLD**
                  is_challenging = (domain_df_full[primary_pcc_col] < 0.80) | \
                                   (domain_df_full[primary_mae_col] > 0.25)
                  # Optional: Add back delta_mae criterion if desired and calculated
                  if delta_mae_col in domain_df_full.columns:
                       is_challenging |= (domain_df_full[delta_mae_col] > 0.2)

                  challenging = domain_df_full[ is_challenging ]
                  # Select only DeepFlex relevant columns for the output table
                  challenging_display = challenging[cols_to_display_final]
                  analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Found {len(challenging)} candidates meeting criteria.\n" + format_table(challenging_display.nlargest(n_candidates, primary_mae_col))
             else: analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Missing columns for Challenging criteria: {required_challenging}"
        except Exception as e:
             logger.warning(f"Error selecting Challenging Candidates: {e}")
             analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Error during selection: {e}"

    except FileNotFoundError:
        logger.error(f"Domain metrics file not found at {domain_metrics_path}. Cannot select candidates.")
        analysis_results['16. CASE STUDY CANDIDATES'] += "\nDomain metrics file not found."
    except Exception as e:
        logger.error(f"Error during case study selection using {domain_metrics_path}: {e}", exc_info=True)
        analysis_results['16. CASE STUDY CANDIDATES'] += "\nError reading/processing domain metrics file."

# --- Main Analysis Runner (Robust Version) ---
def run_analysis(df_path, output_file):
    """Runs the full analysis pipeline with enhanced error handling."""
    start_time = time.time()
    logger.info(f"Starting analysis for file: {df_path}")
    try:
        df = pd.read_csv(df_path)
        df.attrs['source_file'] = df_path # Store source file info
        logger.info(f"Successfully loaded data: {df.shape}")
        if df.empty:
             logger.error("Input CSV file is empty. Aborting analysis.")
             return
    except FileNotFoundError:
        logger.error(f"Data file not found: {df_path}")
        return
    except pd.errors.EmptyDataError:
        logger.error(f"Data file is empty: {df_path}")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    analysis_results = {}
    # Dictionary to store intermediate dataframes if needed
    internal_data = {'original_df': df}

    # Define the order and functions for analysis steps
    # Using lambdas to handle functions that need the results dict later
    analysis_steps = [
        ("1. BASIC INFORMATION", run_basic_info),
        # "1.1 MODEL KEY" handled by run_basic_info
        ("2. MISSING VALUE SUMMARY", run_missing_values),
        ("3. OVERALL DESCRIPTIVE STATISTICS", run_descriptive_stats),
        ("4. DATA DISTRIBUTIONS", run_data_distributions),
        # Sub-distributions handled within run_data_distributions
        ("5. COMPREHENSIVE MODEL COMPARISON", run_model_comparison),
        # Sub-comparisons handled within run_model_comparison
        ("6. UNCERTAINTY ANALYSIS", run_uncertainty_analysis),
        # Sub-uncertainty handled within run_uncertainty_analysis
        # ("7. DOMAIN-LEVEL PERFORMANCE METRICS", run_domain_level_analysis),
        # Individual stratified analyses called below, map to parent sections
        ("8. AMINO ACID PERFORMANCE", lambda df, res: run_stratified_performance(df, res)),
        ("9. PERFORMANCE BY NORMALIZED_RESID BIN", lambda df, res: run_stratified_performance(df, res)),
        ("10. PERFORMANCE BY CORE/EXTERIOR", lambda df, res: run_stratified_performance(df, res)),
        ("11. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)", lambda df, res: run_stratified_performance(df, res)),
        ("12. PERFORMANCE BY ACTUAL RMSF QUANTILE", run_performance_vs_actual_rmsf),
        ("13. PERFORMANCE BY REL. ACCESSIBILITY QUANTILE", lambda df, res: run_stratified_performance(df, res)),
        ("14. PERFORMANCE BY BFACTOR_NORM QUANTILE", lambda df, res: run_stratified_performance(df, res)),
        ("15. MODEL DISAGREEMENT VS. ERROR", run_model_disagreement_analysis),
        ("16. CASE STUDY CANDIDATES", lambda df, res: run_case_study_candidate_selection(res)),
        ("17. STATISTICAL SIGNIFICANCE TESTS", lambda df, res: None) # Placeholder, called explicitly
    ]

    # Run analysis steps sequentially, catching errors per step
    # Create dummy entries for parent headers first
    parent_headers = set(re.match(r"(\d+)\.", k).group(1) for k,v in analysis_steps if re.match(r"(\d+)\.", k))
    for p_header_num in sorted(parent_headers, key=int):
         parent_key = next((k for k,v in analysis_steps if k.startswith(f"{p_header_num}. ")), None)
         if parent_key:
             analysis_results[parent_key] = "" # Initialize parent section header

    # Run actual analyses
    for section_key, analysis_func in analysis_steps:
        # Skip placeholders/parent headers
        if analysis_func is None: continue

        logger.info(f"--- Running Analysis: {section_key} ---")
        try:
            # Pass the *original* dataframe to avoid mutation issues between steps
            # For steps needing results dict (like case studies), pass analysis_results
            if section_key == "16. CASE STUDY CANDIDATES":
                 analysis_func(internal_data['original_df'], analysis_results) # Pass results dict
            else:
                 analysis_func(internal_data['original_df'], analysis_results)
        except Exception as e:
            logger.error(f"Error executing analysis step '{section_key}': {e}", exc_info=True)
            analysis_results[section_key] = f"!!! ERROR DURING ANALYSIS: {e} !!!"

    # Explicitly run significance tests after domain metrics are calculated and stored
    try:
        # Use the potentially saved path from run_domain_level_analysis
        domain_metrics_path = analysis_results.get('_internal_domain_metrics_path')
        if domain_metrics_path and os.path.exists(domain_metrics_path):
             per_domain_metrics_data_df = pd.read_csv(domain_metrics_path, index_col='domain_id')
             # Convert DataFrame back to the expected dict format if needed by calculate_significance_tests
             # Or adapt calculate_significance_tests to take the DataFrame
             per_domain_metrics_data = per_domain_metrics_data_df.apply(lambda row: {
                   col.split('_')[0]: {'mae': row.get(f"{col.split('_')[0]}_mae"), 'pcc': row.get(f"{col.split('_')[0]}_pcc")}
                   for col in domain_df.columns if '_mae' in col or '_pcc' in col
             }, axis=1).to_dict() # This conversion might be inefficient/complex

             # Simpler: Adapt calculate_significance_tests to accept the DataFrame directly
             # For now, we assume the previous implementation using the dict:
             # per_domain_metrics_data = calculate_per_domain_metrics(internal_data['original_df']) # Recalculate if needed

        else: # Fallback: Recalculate if file wasn't saved or path missing
             logger.warning("Recalculating domain metrics for significance testing (saved file not found).")
             per_domain_metrics_data = calculate_per_domain_metrics(internal_data['original_df'])


        if per_domain_metrics_data:
            significance_results = calculate_significance_tests(
                per_domain_metrics_data,
                "DeepFlex",
                KEY_BASELINES_FOR_SIG_TEST
            )
            if significance_results:
                 sig_text = "Comparing DeepFlex against key baselines using Wilcoxon signed-rank test on per-domain metrics.\n"
                 sig_text += "MAE Test (alternative='less'): Is DeepFlex MAE significantly smaller?\n"
                 sig_text += "PCC Test (alternative='greater'): Is DeepFlex PCC significantly larger?\n\n"
                 sig_data_to_format = []
                 for test_name, res in significance_results.items():
                      baseline, metric = test_name.split('_vs_')
                      sig_data_to_format.append({
                          "Baseline": baseline, "Metric": metric,
                          "p-value": res.get('p_value', np.nan),
                          "N_pairs": res.get('N', 0)
                      })
                 sig_text += format_table(sig_data_to_format, floatfmt=".4g")
                 analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = sig_text
            else: analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = "Significance tests could not be calculated (no valid pairs?)."
        else: analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = "Per-domain metrics calculation failed, cannot perform significance tests."
    except Exception as e:
        logger.error(f"Error during significance testing step: {e}", exc_info=True)
        analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = f"!!! ERROR DURING ANALYSIS: {e} !!!"


    # Write results to file using the robust sorting key
    output_dir = os.path.dirname(output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file, 'w') as f:
            # Sort keys numerically using the robust parser
            sorted_keys = sorted(analysis_results.keys(), key=parse_key_for_sort)

            for section_key in sorted_keys:
                # Skip internal keys
                if section_key.startswith('_internal'): continue

                content = analysis_results[section_key]
                # Extract base number and title cleanly
                match = re.match(r"(\d+(?:\.\d+)*)\.?\s*(.*)", section_key)
                if match:
                    section_num_str = match.group(1)
                    section_title_str = match.group(2).strip()
                    # Only print header if title exists (avoid printing for sub-sections like 5.1, 5.2 etc)
                    # Print header only for top-level sections (e.g., '1.', '5.', '17.')
                    if '.' not in section_num_str and section_title_str:
                         f.write("=" * 80 + "\n")
                         f.write(f"## {section_num_str}. {section_title_str} ##\n")
                         f.write("=" * 80 + "\n")
                    elif section_title_str: # Print sub-section header
                         f.write("-" * 80 + "\n")
                         f.write(f"## {section_num_str} {section_title_str} ##\n")
                         f.write("-" * 80 + "\n")

                else: # Fallback for potentially non-standard keys
                     f.write("=" * 80 + "\n")
                     f.write(f"## {section_key} ##\n")
                     f.write("=" * 80 + "\n")

                # Write content only if it's not just the header placeholder
                if content:
                     if isinstance(content, pd.DataFrame):
                         f.write(format_table(content) + "\n\n")
                     elif isinstance(content, dict):
                         try: f.write(json.dumps(content, indent=4, default=str) + "\n\n")
                         except TypeError: f.write(str(content) + "\n\n")
                     else:
                         f.write(str(content) + "\n\n")
            

    except Exception as e:
        logger.error(f"Failed to write analysis results to {output_file}: {e}", exc_info=True)


    end_time = time.time()
    logger.info(f"Analysis complete. Results saved to: {output_file}")
    logger.info(f"Total analysis time: {end_time - start_time:.2f} seconds.")

# --- Command Line Argument Parsing (Keep existing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive analysis on the aggregated flexibility prediction dataset.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="../data/01_final_analysis_dataset.csv",
        help="Path to the aggregated input CSV file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../output/general_analysis.txt",
        help="Path to save the analysis results text file."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__) if __file__ else '.' # Handle case where __file__ is not defined
    input_csv_path = args.input_csv if os.path.isabs(args.input_csv) else os.path.join(script_dir, args.input_csv)
    output_file_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(script_dir, args.output_file)

    run_analysis(input_csv_path, output_file_path)