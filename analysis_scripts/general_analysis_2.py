
import time
import os
import logging
import pandas as pd
import numpy as np
# Added kendalltau, spearmanr was already there
from scipy.stats import wilcoxon, pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
# Added for reliability diagrams
from sklearn.calibration import calibration_curve
from collections import defaultdict
from tabulate import tabulate
import argparse
import json
import re # Import re for robust sorting key
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple

# --- Configuration ---
# Define models and their corresponding prediction/uncertainty columns
# *** Renamed ESM-Flex to DeepFlex ***
MODEL_CONFIG = {
    "DeepFlex": {
        "key": "DeepFlex", # Updated key
        "pred_col": "DeepFlex_rmsf", # Updated column name
        "unc_col": "DeepFlex_rmsf_uncertainty" # Updated column name
    },
    "ESM-Only (Seq+Temp)": {
        "key": "ESM_Only",
        "pred_col": "ESM_Only_rmsf",
        "unc_col": None
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
# *** Ensure these baseline names match the keys in MODEL_CONFIG if testing is desired ***
KEY_BASELINES_FOR_SIG_TEST = ["RF (All Features)", "ESM-Only (Seq+Temp)"]
PRIMARY_MODEL_NAME = "DeepFlex" # Define primary model name centrally
TARGET_COL = 'rmsf' # Define target column centrally

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (safe_pearsonr, format_table, parse_key_for_sort - Keep as is) ---
def safe_pearsonr(x, y):
    """Calculates Pearson correlation safely, returning NaN on error or insufficient data."""
    try:
        x_np = np.asarray(x).astype(np.float64)
        y_np = np.asarray(y).astype(np.float64)
        valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]

        if len(x_clean) < 2: return np.nan, np.nan
        if np.std(x_clean) < 1e-9 or np.std(y_clean) < 1e-9: return np.nan, np.nan

        try: corr, p_value = pearsonr(x_clean, y_clean)
        except (ValueError, TypeError) as ve:
            logger.debug(f"Pearson calculation failed (likely constant input or type issue): {ve}")
            return np.nan, np.nan
        return corr if not np.isnan(corr) else np.nan, p_value
    except Exception as e:
        logger.error(f"Unexpected error during pearsonr calculation: {e}", exc_info=False) # Reduce noise
        return np.nan, np.nan

def safe_spearmanr(x, y):
    """Calculates Spearman correlation safely."""
    try:
        x_np = np.asarray(x).astype(np.float64)
        y_np = np.asarray(y).astype(np.float64)
        valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]
        if len(x_clean) < 2: return np.nan, np.nan
        # Spearman handles ties, less sensitive to std=0 than Pearson? Check needed.
        # Add std check just in case underlying rank calculation has issues
        if np.std(x_clean) < 1e-9 or np.std(y_clean) < 1e-9: return np.nan, np.nan
        corr, p_value = spearmanr(x_clean, y_clean)
        return corr if not np.isnan(corr) else np.nan, p_value
    except Exception as e:
        logger.error(f"Unexpected error during spearmanr calculation: {e}", exc_info=False)
        return np.nan, np.nan

def safe_kendalltau(x, y):
    """Calculates Kendall's Tau safely."""
    try:
        x_np = np.asarray(x).astype(np.float64)
        y_np = np.asarray(y).astype(np.float64)
        valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_clean = x_np[valid_mask]
        y_clean = y_np[valid_mask]
        if len(x_clean) < 2: return np.nan, np.nan
        # Kendall Tau should also handle constant arrays gracefully (results in NaN or zero corr)
        corr, p_value = kendalltau(x_clean, y_clean)
        return corr if not np.isnan(corr) else np.nan, p_value
    except Exception as e:
        logger.error(f"Unexpected error during kendalltau calculation: {e}", exc_info=False)
        return np.nan, np.nan

def format_table(data, headers="keys", tablefmt="pipe", floatfmt=".6f", **kwargs):
    """Formats data using tabulate, with improved error handling for empty data."""
    if isinstance(data, pd.DataFrame) and data.empty: return " (No data to display) "
    if not isinstance(data, pd.DataFrame) and not data: return " (No data to display) "
    try:
        return tabulate(data, headers=headers, tablefmt=tablefmt, floatfmt=floatfmt, **kwargs)
    except Exception as e:
        logger.error(f"Tabulate formatting failed: {e}")
        if isinstance(data, pd.DataFrame):
             return f"(Error formatting DataFrame: {e})\nColumns: {data.columns.tolist()}\nFirst few rows:\n{data.head().to_string()}"
        return f"(Error formatting data: {e})\nData: {str(data)[:200]}..."

def parse_key_for_sort(key_str):
    """ Parses section keys like '1.1', '5.2.1', '15.' for robust numerical sorting."""
    if not isinstance(key_str, str): return (999,)
    match = re.match(r"^(\d+(?:\.\d+)*)\.?(.*)", key_str.strip())
    if match:
        num_part = match.group(1)
        try: return tuple(int(p) for p in num_part.split('.'))
        except ValueError: return (999,)
    else: return (999,)


# --- Existing Analysis Functions (Keep as is or adapt slightly) ---
# run_basic_info, run_missing_values, run_descriptive_stats, run_data_distributions
# calculate_per_domain_metrics, calculate_significance_tests, run_domain_level_analysis
# calculate_performance_vs_actual_rmsf, run_performance_vs_actual_rmsf
# run_model_disagreement_analysis, run_case_study_candidate_selection
# Note: Update PRIMARY_MODEL_NAME usage inside relevant functions if needed.
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
            "Original Key": config['key'], # Shows 'DeepFlex' now
            "Prediction Column": config['pred_col'], # Shows 'DeepFlex_rmsf'
            "Uncertainty Column": config.get('unc_col', 'N/A') # Shows 'DeepFlex_rmsf_uncertainty'
        })
    analysis_results['1.1 MODEL KEY'] = format_table(model_key_data, headers="keys", tablefmt="pipe")
    analysis_results['1.2 PRIMARY MODEL'] = f"Primary Model for focused analysis: {PRIMARY_MODEL_NAME}"

def run_missing_values(df, analysis_results):
    logger.info("Running missing value analysis...")
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame({'count': missing.astype(int), 'percentage': (missing / len(df)) * 100})
            # Explicitly check for DeepFlex columns and add if missing from isnull().sum() output but present in config
            deepflex_pred = MODEL_CONFIG[PRIMARY_MODEL_NAME]['pred_col']
            deepflex_unc = MODEL_CONFIG[PRIMARY_MODEL_NAME]['unc_col']
            if deepflex_pred not in missing.index and deepflex_pred in df.columns:
                 if df[deepflex_pred].isnull().sum() > 0:
                      missing.loc[deepflex_pred] = df[deepflex_pred].isnull().sum()
                      missing_df.loc[deepflex_pred] = {'count': missing.loc[deepflex_pred], 'percentage': (missing.loc[deepflex_pred] / len(df)) * 100}

            if deepflex_unc and deepflex_unc not in missing.index and deepflex_unc in df.columns:
                 if df[deepflex_unc].isnull().sum() > 0:
                      missing.loc[deepflex_unc] = df[deepflex_unc].isnull().sum()
                      missing_df.loc[deepflex_unc] = {'count': missing.loc[deepflex_unc], 'percentage': (missing.loc[deepflex_unc] / len(df)) * 100}

            analysis_results['2. MISSING VALUE SUMMARY'] = format_table(missing_df.sort_values('count', ascending=False), floatfmt=".2f")
        else:
            analysis_results['2. MISSING VALUE SUMMARY'] = "No missing values found (or columns missing)."
    except Exception as e:
         logger.error(f"Error during missing value analysis: {e}")
         analysis_results['2. MISSING VALUE SUMMARY'] = f"Error: {e}"

def run_descriptive_stats(df, analysis_results):
    logger.info("Running descriptive statistics...")
    try:
        # Select only numeric columns for describe()
        numeric_cols = df.select_dtypes(include=np.number).columns
        cols_to_describe = [TARGET_COL] + \
                           [cfg['pred_col'] for cfg in MODEL_CONFIG.values() if cfg['pred_col'] in df.columns] + \
                           [cfg['unc_col'] for cfg in MODEL_CONFIG.values() if cfg.get('unc_col') and cfg['unc_col'] in df.columns] + \
                           ['temperature', 'normalized_resid', 'relative_accessibility', 'protein_size', 'bfactor_norm', 'phi', 'psi', 'contact_number'] # Add potentially relevant features

        # Filter to only columns that actually exist in the dataframe
        cols_to_describe = [col for col in cols_to_describe if col in df.columns]
        cols_to_describe = list(dict.fromkeys(cols_to_describe)) # Remove duplicates

        if cols_to_describe:
             desc_stats = df[cols_to_describe].describe().transpose()
             analysis_results['3. OVERALL DESCRIPTIVE STATISTICS (Key Variables)'] = format_table(desc_stats)
        else:
             analysis_results['3. OVERALL DESCRIPTIVE STATISTICS (Key Variables)'] = "No relevant numeric columns found for descriptive statistics."
    except Exception as e:
        logger.error(f"Error during descriptive statistics: {e}")
        analysis_results['3. OVERALL DESCRIPTIVE STATISTICS (Key Variables)'] = f"Error: {e}"

def run_data_distributions(df, analysis_results):
    logger.info("Analyzing data distributions...")
    analysis_results['4. DATA DISTRIBUTIONS'] = "Counts based on all rows."
    distribution_cols = {
        'temperature': '4.1 TEMPERATURE DISTRIBUTION',
        'resname': '4.2 RESNAME DISTRIBUTION',
        'core_exterior_encoded': '4.3 CORE/EXTERIOR DISTRIBUTION',
        'secondary_structure_encoded': '4.4 SECONDARY STRUCTURE (H/E/L) DISTRIBUTION'
        # Ramachandran distribution handled in its own section if phi/psi exist
    }
    ss_label_map = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
    core_label_map = {0: 'core', 1: 'exterior'}

    for col, section_title in distribution_cols.items():
        if col in df.columns:
            try:
                counts = df[col].value_counts()
                percent = (counts / len(df)) * 100
                dist_df = pd.DataFrame({'Percent': percent, 'Count': counts.astype(int)})

                if col == 'secondary_structure_encoded':
                    dist_df.index = dist_df.index.map(ss_label_map).fillna('Unknown')
                elif col == 'core_exterior_encoded':
                    dist_df.index = dist_df.index.map(core_label_map).fillna('Unknown')
                elif col == 'temperature':
                    dist_df = dist_df.sort_index() # Sort temperatures

                analysis_results[section_title] = format_table(dist_df.sort_values('Count', ascending=False), floatfmt=".2f")
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
    metrics_by_temp_dict = defaultdict(lambda: defaultdict(list)) # Use defaultdict for flexibility
    rank_metrics_overall = [] # For Spearman/Kendall

    model_report_names_found = [name for name, cfg in MODEL_CONFIG.items() if cfg['pred_col'] in df.columns]
    logger.info(f"Models found for comparison: {model_report_names_found}")

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found. Skipping model comparison.")
        analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = "Target column missing."
        analysis_results['5.2 OVERALL RANK METRICS'] = "Target column missing."
        analysis_results['5.3 PERFORMANCE METRICS BY TEMPERATURE'] = "Target column missing."
        analysis_results['5.4 PREDICTION R-SQUARED MATRIX'] = "Target column missing."
        analysis_results['5.5 ABSOLUTE ERROR R-SQUARED MATRIX'] = "Target column missing."
        return

    df_errors = df[[target_col]].copy() # For error correlation

    for report_name, config in MODEL_CONFIG.items():
        pred_col = config['pred_col']
        if pred_col in df.columns:
            # --- Overall Metrics (MAE, RMSE, PCC, R2, MedAE) ---
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

                    # --- Overall Rank Metrics (Spearman, Kendall) ---
                    rho, rho_p = safe_spearmanr(y_true, y_pred)
                    tau, tau_p = safe_kendalltau(y_true, y_pred)
                    rank_metrics_overall.append({'Model': report_name, 'spearman_rho': rho, 'kendall_tau': tau})

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
                             rho_t, _ = safe_spearmanr(y_true_t, y_pred_t) # Spearman by Temp

                             metrics_by_temp_dict[temp][f'{report_name}_mae'].append(mae_t)
                             metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(pcc_t)
                             metrics_by_temp_dict[temp][f'{report_name}_r2'].append(r2_t)
                             metrics_by_temp_dict[temp][f'{report_name}_spearman_rho'].append(rho_t) # Add Spearman
                         except Exception as e:
                              logger.warning(f"Could not calculate metrics for {report_name} at T={temp}: {e}")
                              metrics_by_temp_dict[temp][f'{report_name}_mae'].append(np.nan)
                              metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(np.nan)
                              metrics_by_temp_dict[temp][f'{report_name}_r2'].append(np.nan)
                              metrics_by_temp_dict[temp][f'{report_name}_spearman_rho'].append(np.nan) # Add Spearman NaN
                    else: # Append NaNs if not enough data at this temp for this model
                        metrics_by_temp_dict[temp][f'{report_name}_mae'].append(np.nan)
                        metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(np.nan)
                        metrics_by_temp_dict[temp][f'{report_name}_r2'].append(np.nan)
                        metrics_by_temp_dict[temp][f'{report_name}_spearman_rho'].append(np.nan) # Add Spearman NaN
        else:
            logger.warning(f"Prediction column '{pred_col}' for model '{report_name}' not found. Skipping comparison.")


    # Format Overall Performance Metrics Table
    if metrics_overall:
        overall_df = pd.DataFrame(metrics_overall).set_index('Model')
        # Sort by primary model's key metric, e.g., PCC
        sort_metric = 'pcc'
        analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = format_table(overall_df.sort_values(sort_metric, ascending=False))
    else: analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = "No models found or metrics calculable."

    # Format Overall Rank Metrics Table
    if rank_metrics_overall:
        rank_df = pd.DataFrame(rank_metrics_overall).set_index('Model')
        analysis_results['5.2 OVERALL RANK METRICS'] = "(Higher values indicate better preservation of relative flexibility order)\n" + format_table(rank_df.sort_values('spearman_rho', ascending=False))
    else: analysis_results['5.2 OVERALL RANK METRICS'] = "No models found or rank metrics calculable."


    # Format Metrics by Temperature Table
    if metrics_by_temp_dict:
        temp_metrics_list = []
        all_temp_cols = set() # Track all columns generated across temperatures
        for temp, metrics in sorted(metrics_by_temp_dict.items()):
            row_count = len(df[np.isclose(df['temperature'], temp)]) if 'temperature' in df.columns else 'N/A'
            row = {'temperature': temp, 'count': row_count}
            for metric_col, metric_list in metrics.items():
                # Use nanmean to ignore NaNs from failed calculations
                avg_metric = np.nanmean(metric_list) if metric_list else np.nan
                row[metric_col] = avg_metric
                all_temp_cols.add(metric_col)
            temp_metrics_list.append(row)

        temp_df = pd.DataFrame(temp_metrics_list).set_index('temperature')
        # Ensure all potential columns exist
        for col in all_temp_cols:
             if col not in temp_df.columns: temp_df[col] = np.nan

        # Define desired column order: count, then group by model, then metric type
        cols_ordered = ['count']
        metric_order = ['mae', 'pcc', 'spearman_rho', 'r2'] # Define metric order
        for model_name in model_report_names_found: # Iterate through models found
             for metric_suffix in metric_order:
                  col_name = f'{model_name}_{metric_suffix}'
                  if col_name in temp_df.columns:
                       cols_ordered.append(col_name)
        # Append any remaining columns not caught by the loop (shouldn't happen ideally)
        cols_ordered.extend([col for col in temp_df.columns if col not in cols_ordered])

        analysis_results['5.3 PERFORMANCE METRICS BY TEMPERATURE'] = format_table(temp_df[cols_ordered])

        # Add summary comparison highlights by Temperature
        try:
             temp_summary_lines = ["Model Comparison Highlights by Temperature:"]
             primary_mae_prefix = f"{PRIMARY_MODEL_NAME}_mae"
             primary_pcc_prefix = f"{PRIMARY_MODEL_NAME}_pcc"
             primary_rho_prefix = f"{PRIMARY_MODEL_NAME}_spearman_rho"

             for temp in sorted(metrics_by_temp_dict.keys()):
                 row_data = temp_df.loc[temp]
                 mae_cols = [col for col in row_data.index if col.endswith('_mae') and pd.notna(row_data[col])]
                 pcc_cols = [col for col in row_data.index if col.endswith('_pcc') and pd.notna(row_data[col])]
                 rho_cols = [col for col in row_data.index if col.endswith('_spearman_rho') and pd.notna(row_data[col])]

                 if not mae_cols or not pcc_cols: continue # Need basic metrics

                 best_mae_model_col = row_data[mae_cols].idxmin()
                 best_pcc_model_col = row_data[pcc_cols].idxmax()
                 best_rho_model_col = row_data[rho_cols].idxmax() if rho_cols else "N/A"

                 # Extract model name cleanly
                 best_mae_model = best_mae_model_col.replace('_mae', '')
                 best_pcc_model = best_pcc_model_col.replace('_pcc', '')
                 best_rho_model = best_rho_model_col.replace('_spearman_rho', '') if rho_cols else "N/A"

                 summary_str = f"  T={temp:.1f}K (N={int(row_data.get('count',0))}):"
                 summary_str += f" Best MAE={best_mae_model}({row_data[best_mae_model_col]:.3f}),"
                 summary_str += f" Best PCC={best_pcc_model}({row_data[best_pcc_model_col]:.3f}),"
                 if rho_cols:
                     summary_str += f" Best Spearman Rho={best_rho_model}({row_data[best_rho_model_col]:.3f})"
                 # Optionally add primary model's values for comparison
                 if primary_mae_prefix in row_data and pd.notna(row_data[primary_mae_prefix]):
                      summary_str += f" | {PRIMARY_MODEL_NAME}: MAE={row_data[primary_mae_prefix]:.3f}"
                 if primary_pcc_prefix in row_data and pd.notna(row_data[primary_pcc_prefix]):
                      summary_str += f", PCC={row_data[primary_pcc_prefix]:.3f}"
                 if primary_rho_prefix in row_data and pd.notna(row_data[primary_rho_prefix]):
                     summary_str += f", Rho={row_data[primary_rho_prefix]:.3f}"

                 temp_summary_lines.append(summary_str)

             analysis_results['5.3.1 TEMP PERFORMANCE SUMMARY'] = "\n".join(temp_summary_lines)
        except Exception as e:
             logger.error(f"Error generating temperature performance summary: {e}")
             analysis_results['5.3.1 TEMP PERFORMANCE SUMMARY'] = "Error generating summary."

    else: analysis_results['5.3 PERFORMANCE METRICS BY TEMPERATURE'] = "No temperature data or metrics calculable."

    # Prediction Correlation Matrix
    pred_cols_present = [config['pred_col'] for config in MODEL_CONFIG.values() if config['pred_col'] in df.columns]
    if len(pred_cols_present) > 1:
        try:
            pred_corr = df[[target_col] + pred_cols_present].dropna().corr(method='pearson')**2 # R-squared
            rename_map_pred = {target_col: 'ActualRMSF'}
            for report_name, config in MODEL_CONFIG.items():
                 if config['pred_col'] in pred_corr.columns: rename_map_pred[config['pred_col']] = report_name
            pred_corr = pred_corr.rename(columns=rename_map_pred, index=rename_map_pred)
            analysis_results['5.4 PREDICTION R-SQUARED MATRIX (COEFFICIENT OF DETERMINATION, INCL. ACTUAL)'] = format_table(pred_corr)
        except Exception as e:
             logger.error(f"Error calculating prediction correlation: {e}")
             analysis_results['5.4 PREDICTION R-SQUARED MATRIX'] = f"Error: {e}"
    else: analysis_results['5.4 PREDICTION R-SQUARED MATRIX'] = "Not enough models with predictions to correlate."

    # Error Correlation Matrix
    error_cols_present = [col for col in df_errors.columns if col.endswith('_abs_error')]
    if len(error_cols_present) > 1:
        try:
            error_corr = df_errors[[target_col] + error_cols_present].dropna().corr(method='pearson')**2 # R-squared
            rename_map_err = {target_col: 'ActualRMSF'} # Keep target as rmsf for clarity
            for report_name, config in MODEL_CONFIG.items():
                 err_col_name = f"{report_name}_abs_error"
                 if err_col_name in error_corr.columns: rename_map_err[err_col_name] = f"{report_name}_AbsErr"
            error_corr = error_corr.rename(columns=rename_map_err, index=rename_map_err)

            analysis_results['5.5 ABSOLUTE ERROR R-SQUARED MATRIX (COEFFICIENT OF DETERMINATION)'] = \
                "(Shows squared correlation (R^2) between model errors. High value means models make errors on the same samples. R^2 between errors and Actual RMSF is also shown.)\n" + \
                 format_table(error_corr)
        except Exception as e:
             logger.error(f"Error calculating error correlation: {e}")
             analysis_results['5.5 ABSOLUTE ERROR R-SQUARED MATRIX'] = f"Error: {e}"
    else: analysis_results['5.5 ABSOLUTE ERROR R-SQUARED MATRIX'] = "Not enough models with errors to correlate."

# --- NEW: Dihedral Angle (Ramachandran) Analysis ---
def get_ramachandran_region(phi, psi):
    """Assigns residue to Ramachandran region based on phi/psi angles."""
    # Ensure angles are within -180 to 180 range if needed (data might be 0-360)
    # These boundaries are approximate and can be adjusted
    if pd.isna(phi) or pd.isna(psi):
        return "Undefined"

    # Convert to -180 to 180 if necessary (example check)
    if phi > 180: phi -= 360
    if psi > 180: psi -= 360

    # Define regions (Example boundaries - adjust as needed)
    if (-180 <= phi < -30) and (-70 <= psi < 50): # Broad alpha-helix region
        return "Alpha-Helix"
    if (-180 <= phi < -40) and (90 <= psi <= 180 or -180 <= psi < -150): # Broad beta-sheet region
         return "Beta-Sheet"
    if (30 < phi < 100) and (-20 < psi < 90): # Left-handed alpha region
         return "L-Alpha"
    # Simplified "Disallowed" - regions not covered above (can be refined)
    # Check specific disallowed combinations if needed
    if (0 <= phi < 180 and -90 <= psi < 0): # Example disallowed zone
         return "Disallowed"
    return "Other Allowed/Loop"

def run_dihedral_analysis(df, analysis_results):
    logger.info("Running Dihedral Angle (Ramachandran) analysis...")
    analysis_results['5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS'] = "" # Placeholder title

    phi_col, psi_col = 'phi', 'psi'
    if phi_col not in df.columns or psi_col not in df.columns:
        logger.warning(f"Missing '{phi_col}' or '{psi_col}' columns. Skipping Dihedral Analysis.")
        analysis_results['5.6.1 PERFORMANCE BY RAMACHANDRAN REGION'] = f"Skipped: Missing '{phi_col}' or '{psi_col}' columns."
        return

    try:
        df_copy = df.copy()
        # Apply the region assignment function
        df_copy['rama_region'] = df_copy.apply(lambda row: get_ramachandran_region(row[phi_col], row[psi_col]), axis=1)

        # --- Distribution of Regions ---
        rama_counts = df_copy['rama_region'].value_counts()
        rama_percent = (rama_counts / len(df_copy[df_copy['rama_region'] != "Undefined"])) * 100 # Percent of defined
        rama_dist_df = pd.DataFrame({'Percent': rama_percent, 'Count': rama_counts.astype(int)})
        analysis_results['5.6.1 RAMACHANDRAN REGION DISTRIBUTION'] = "(Based on defined phi/psi pairs)\n" + format_table(rama_dist_df.sort_values('Count', ascending=False), floatfmt=".2f")


        # --- Performance Stratified by Region ---
        logger.info("Stratifying performance by Ramachandran region...")
        strata_results = []
        grouped = df_copy.groupby('rama_region', observed=True) # Use observed=True

        for group_name, group_df in grouped:
             if group_name == "Undefined" or len(group_df) < 5: continue # Skip undefined or small groups

             row = {'rama_region': group_name, 'count': len(group_df)}

             for report_name, config in MODEL_CONFIG.items():
                  pred_col = config['pred_col']
                  mae_val, pcc_val = np.nan, np.nan # Default to NaN
                  if pred_col in group_df.columns:
                       df_valid = group_df[[TARGET_COL, pred_col]].dropna()
                       if len(df_valid) > 1:
                            y_true = df_valid[TARGET_COL].values
                            y_pred = df_valid[pred_col].values
                            try: mae_val = mean_absolute_error(y_true, y_pred)
                            except: pass
                            try: pcc_val, _ = safe_pearsonr(y_true, y_pred)
                            except: pass
                  row[f'{report_name}_mae'] = mae_val
                  row[f'{report_name}_pcc'] = pcc_val
             strata_results.append(row)

        if strata_results:
            strata_df = pd.DataFrame(strata_results).set_index('rama_region')
            strata_df.index.name = "Ramachandran Region"
            # Sort maybe by DeepFlex MAE
            primary_mae_col = f'{PRIMARY_MODEL_NAME}_mae'
            if primary_mae_col in strata_df.columns:
                 strata_df = strata_df.sort_values(primary_mae_col)
            else:
                 strata_df = strata_df.sort_values('count', ascending=False) # Fallback sort

            analysis_results['5.6.2 PERFORMANCE BY RAMACHANDRAN REGION'] = format_table(strata_df)
        else:
            analysis_results['5.6.2 PERFORMANCE BY RAMACHANDRAN REGION'] = "No results generated after grouping by region."

    except Exception as e:
        logger.error(f"Error during Dihedral Analysis: {e}", exc_info=True)
        analysis_results['5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS'] += f"\nError during analysis: {e}"

# --- Uncertainty Analysis (Enhanced) ---
def run_uncertainty_analysis(df, analysis_results):
    logger.info("Running uncertainty analysis...")
    analysis_results['6. UNCERTAINTY ANALYSIS'] = "Comparing uncertainty estimates for models where available."
    uncertainty_stats = []
    uncertainty_error_corr = []
    calibration_check = []
    mae_vs_unc_bins = {}
    reliability_data = {} # Store data for reliability diagrams
    ece_results = [] # Store Expected Calibration Error

    target_col = TARGET_COL

    df_copy = df.copy() # Work on a copy
    models_with_uncertainty = []
    for report_name, config in MODEL_CONFIG.items():
        pred_col = config['pred_col']
        unc_col = config.get('unc_col')
        error_col = f"{report_name}_abs_error"

        if not unc_col or unc_col not in df_copy.columns:
            logger.debug(f"No uncertainty column '{unc_col}' found for model '{report_name}'. Skipping uncertainty analysis.")
            continue # Skip model if no uncertainty column

        # Ensure error column exists
        if error_col not in df_copy.columns:
            if target_col in df_copy.columns and pred_col in df_copy.columns:
                logger.debug(f"Calculating {error_col} for uncertainty analysis.")
                df_copy[error_col] = (df_copy[pred_col] - df_copy[target_col]).abs()
            else:
                logger.warning(f"Cannot calculate error for {report_name}, skipping uncertainty analysis for it.")
                continue # Skip this model if error cannot be calculated

        models_with_uncertainty.append(report_name)
        df_valid = df_copy[[error_col, unc_col]].dropna() # Use calculated error col

        if not df_valid.empty and len(df_valid) > 10: # Need sufficient points
            unc_values = df_valid[unc_col].values
            abs_error = df_valid[error_col].values

            # 6.1 Stats
            stats = df_valid[unc_col].describe()
            safe_stats = {k: float(v) if isinstance(v, (np.number, int, float)) else str(v) for k, v in stats.to_dict().items()}
            uncertainty_stats.append({'Model': report_name, **safe_stats})

            # 6.2 Correlation with error
            corr, _ = safe_pearsonr(unc_values, abs_error) # Handles internal checks
            uncertainty_error_corr.append({'Model': report_name, 'Uncertainty-Error PCC': corr})

            # 6.3 Calibration Check (% within 1 sigma)
            try: within_1_std = np.mean(abs_error <= unc_values) * 100
            except TypeError: within_1_std = np.nan
            calibration_check.append({'Model': report_name, '% within 1 Uncertainty': within_1_std})

            # 6.4 MAE vs Uncertainty Bins (Quantile Bins)
            try:
                n_bins_mae = 10
                df_valid['unc_quantile'], bin_edges_mae = pd.qcut(unc_values, q=n_bins_mae, labels=False, duplicates='drop', retbins=True)
                actual_num_bins_mae = df_valid['unc_quantile'].nunique()
                if actual_num_bins_mae >= 2:
                    bin_range_labels_mae = {i: f"{bin_edges_mae[i]:.3g}-{bin_edges_mae[i+1]:.3g}" for i in range(len(bin_edges_mae)-1)}
                    binned_mae = df_valid.groupby('unc_quantile')[error_col].agg(['mean', 'median', 'std', 'count']).reset_index()
                    binned_mae['unc_bin_range'] = binned_mae['unc_quantile'].map(bin_range_labels_mae) # Map index to range
                    binned_mae = binned_mae.rename(columns={'mean': 'MeanAbsErr', 'median': 'MedianAbsErr', 'std': 'StdAbsErr', 'unc_bin_range': 'UncertaintyQuantile'})
                    mae_vs_unc_bins[report_name] = format_table(binned_mae[['UncertaintyQuantile', 'MeanAbsErr', 'MedianAbsErr', 'StdAbsErr', 'count']])
                else: logger.warning(f"Could not create enough bins for MAE vs Uncertainty for {report_name}")
            except Exception as e:
                logger.warning(f"Could not calculate MAE vs uncertainty bins for {report_name}: {e}")

            # 6.5 Reliability Diagram Data & ECE (Evenly spaced bins)
            try:
                n_bins_rel = 10
                # Use calibration_curve for binning and calculating fraction_of_positives equivalent
                # Here, "positive" means error <= uncertainty
                # We adapt: bin by predicted uncertainty, calculate mean predicted uncertainty and mean observed error in each bin.
                min_unc, max_unc = np.min(unc_values), np.max(unc_values)
                if max_unc <= min_unc: raise ValueError("Uncertainty range is zero or negative.")

                # Define bins based on uncertainty range
                bin_limits = np.linspace(min_unc, max_unc, n_bins_rel + 1)
                # Ensure bins are unique, handle edge cases
                bin_limits = np.unique(bin_limits)
                if len(bin_limits) < 2: raise ValueError("Could not create at least 2 unique bin edges for reliability.")
                actual_n_bins_rel = len(bin_limits) - 1

                # Assign data points to bins based on uncertainty
                # Use pd.cut which handles edges better. include_lowest=True ensures min value is included.
                # labels=False gives integer indices 0 to n_bins-1
                # Need to handle potential NaNs returned by cut if value is outside outer edges (shouldn't happen with linspace)
                df_valid['rel_bin'] = pd.cut(unc_values, bins=bin_limits, labels=False, include_lowest=True, right=True)

                # Calculate metrics per bin
                binned_data = []
                total_count = len(df_valid)
                ece = 0.0

                for i in range(actual_n_bins_rel):
                    bin_mask = (df_valid['rel_bin'] == i)
                    bin_df = df_valid[bin_mask]
                    bin_count = len(bin_df)

                    if bin_count > 0:
                        mean_pred_unc = bin_df[unc_col].mean()
                        # Observed error: Use RMSE of absolute errors within the bin as a measure of deviation
                        # Or simply use Mean Absolute Error in the bin
                        # Using MAE for consistency with other metrics
                        observed_mae_bin = bin_df[error_col].mean()
                        # Optional: Observed RMSE in the bin
                        observed_rmse_bin = np.sqrt(mean_squared_error(np.zeros_like(bin_df[error_col]), bin_df[error_col]))


                        bin_lower = bin_limits[i]
                        bin_upper = bin_limits[i+1]
                        binned_data.append({
                            'Bin Index': i,
                            'Uncertainty Range': f"{bin_lower:.3g}-{bin_upper:.3g}",
                            'Mean Predicted Uncertainty': mean_pred_unc,
                            'Observed MAE in Bin': observed_mae_bin,
                            'Observed RMSE in Bin': observed_rmse_bin, # Added RMSE
                            'Count': bin_count
                        })
                        # ECE calculation using MAE as the observed error measure
                        ece += (np.abs(mean_pred_unc - observed_mae_bin) * bin_count)
                    # else: append empty row? or just skip

                if total_count > 0:
                    ece = ece / total_count
                    ece_results.append({'Model': report_name, 'Expected Calibration Error (ECE-MAE)': ece})

                    reliability_df = pd.DataFrame(binned_data)
                    reliability_data[report_name] = format_table(reliability_df.set_index('Bin Index'))
                else:
                    logger.warning(f"No data points found for reliability analysis for {report_name}")


            except Exception as e:
                logger.warning(f"Could not calculate Reliability/ECE data for {report_name}: {e}", exc_info=False) # Reduce noise

        else: logger.warning(f"Not enough valid data found for uncertainty analysis of {report_name}")


    # Format and store results
    if uncertainty_stats:
        analysis_results['6.1 UNCERTAINTY DISTRIBUTION STATISTICS'] = format_table(pd.DataFrame(uncertainty_stats).set_index('Model'))
    else: analysis_results['6.1 UNCERTAINTY DISTRIBUTION STATISTICS'] = "No models with uncertainty data found."

    if uncertainty_error_corr:
        analysis_results['6.2 UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION'] = "(Positive correlation indicates uncertainty tracks error magnitude.)\n" + format_table(pd.DataFrame(uncertainty_error_corr).set_index('Model'))
    else: analysis_results['6.2 UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION'] = "No models with uncertainty data found or correlation calculable."

    if calibration_check:
        analysis_results['6.3 SIMPLE CALIBRATION CHECK'] = "(% of errors <= predicted uncertainty. Expected ~68.2% for well-calibrated Gaussian uncertainty.)\n" + format_table(pd.DataFrame(calibration_check).set_index('Model'), floatfmt=".2f")
    else: analysis_results['6.3 SIMPLE CALIBRATION CHECK'] = "No models with uncertainty data found or calibration calculable."

    for model_name, table_str in mae_vs_unc_bins.items():
        analysis_results[f'6.4 MEAN ABSOLUTE ERROR BINNED BY {model_name.upper()} UNCERTAINTY QUANTILE'] = table_str
    if not mae_vs_unc_bins:
         analysis_results['6.4 MEAN ABSOLUTE ERROR BINNED BY UNCERTAINTY QUANTILE'] = "No results generated (check data/binning)."

    for model_name, table_str in reliability_data.items():
        analysis_results[f'6.5 RELIABILITY DIAGRAM DATA ({model_name.upper()})'] = "(Binning by predicted uncertainty. Plot 'Mean Predicted Uncertainty' vs 'Observed MAE/RMSE in Bin' for reliability diagram.)\n" + table_str
    if not reliability_data:
         analysis_results['6.5 RELIABILITY DIAGRAM DATA'] = "No results generated (check data/binning)."

    if ece_results:
        analysis_results['6.6 EXPECTED CALIBRATION ERROR (ECE)'] = "(Lower ECE indicates better calibration. Calculated using MAE as observed error measure.)\n" + format_table(pd.DataFrame(ece_results).set_index('Model'), floatfmt=".4f")
    else: analysis_results['6.6 EXPECTED CALIBRATION ERROR (ECE)'] = "ECE not calculated."


# --- Domain Level Analysis (Keep existing function) ---
def calculate_per_domain_metrics(df, target_col='rmsf'):
    """Calculates MAE and PCC for each model within each domain. Handles missing data."""
    logger.info("Calculating per-domain performance metrics...")
    domain_metrics = {} # Use dict directly: {domain_id: {model_name: {'mae': v, 'pcc': v}}}

    if 'domain_id' not in df.columns:
        logger.error("Input DataFrame missing 'domain_id' column for per-domain metrics. Skipping.")
        return {}
    if target_col not in df.columns:
         logger.error(f"Input DataFrame missing target column '{target_col}'. Skipping.")
         return {}

    df_filtered = df.dropna(subset=[target_col, 'domain_id']).copy()
    if df_filtered.empty:
        logger.warning("No valid rows found after dropping NaNs in target/domain_id. Cannot calculate per-domain metrics.")
        return {}

    grouped = df_filtered.groupby('domain_id')
    num_domains = len(grouped)

    for domain_id, group in tqdm(grouped, desc="Domain Metrics", total=num_domains):
        if group.empty: continue
        domain_results = {}
        actual = group[target_col].values

        for report_name, config in MODEL_CONFIG.items():
            pred_col = config['pred_col']
            mae_val, pcc_val = np.nan, np.nan

            if pred_col in group.columns:
                predicted = group[pred_col].values
                valid_mask = ~np.isnan(predicted) # Actual is already not NaN
                actual_valid = actual[valid_mask]
                predicted_valid = predicted[valid_mask]

                if len(actual_valid) > 1:
                    try: mae_val = mean_absolute_error(actual_valid, predicted_valid)
                    except Exception: mae_val = np.nan
                    try: pcc_val, _ = safe_pearsonr(actual_valid, predicted_valid)
                    except Exception: pcc_val = np.nan

            domain_results[report_name] = {'mae': mae_val, 'pcc': pcc_val}

        if domain_results: domain_metrics[domain_id] = domain_results

    logger.info(f"Calculated per-domain metrics for {len(domain_metrics)} domains.")
    return domain_metrics

def run_domain_level_analysis(df, analysis_results):
    """Runs domain-level analysis and saves detailed metrics per domain to a CSV file."""
    logger.info("Running domain-level analysis...")
    analysis_results['7. DOMAIN-LEVEL PERFORMANCE METRICS'] = "" # Initialize section header

    domain_metrics_dict = calculate_per_domain_metrics(df, target_col=TARGET_COL)

    if not domain_metrics_dict:
        logger.error("Failed to calculate per-domain metrics. Skipping domain-level analysis.")
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: No per-domain metrics calculated."
        if '_internal_domain_metrics_path' in analysis_results: del analysis_results['_internal_domain_metrics_path']
        return

    domain_data_list = []
    all_metric_keys = set(['domain_id', 'count_residues', 'actual_mean', 'actual_stddev']) # Base keys

    if 'domain_id' not in df.columns:
         logger.error("Cannot calculate detailed domain stats: 'domain_id' missing.")
         # Still try to proceed if domain_metrics_dict exists, but mean table might be limited
         # Or return early:
         # analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: 'domain_id' missing."
         # return
         pass # Continue but some calculations might fail

    grouped_by_domain = df.groupby('domain_id') if 'domain_id' in df.columns else None

    for domain_id, model_metrics in tqdm(domain_metrics_dict.items(), desc="Aggregating Domain Stats", total=len(domain_metrics_dict)):
        row = {'domain_id': domain_id}
        domain_group = grouped_by_domain.get_group(domain_id).copy() if grouped_by_domain is not None else pd.DataFrame() # Get original data

        # Add pre-calculated MAE/PCC
        for model_name, metrics in model_metrics.items():
            row[f"{model_name}_mae"] = metrics.get('mae', np.nan)
            row[f"{model_name}_pcc"] = metrics.get('pcc', np.nan)
            all_metric_keys.add(f"{model_name}_mae")
            all_metric_keys.add(f"{model_name}_pcc")

        # Add basic domain info
        row['count_residues'] = len(domain_group)
        actual_rmsf_domain = domain_group[TARGET_COL].dropna() if TARGET_COL in domain_group else pd.Series(dtype=float)
        row['actual_mean'] = actual_rmsf_domain.mean() if not actual_rmsf_domain.empty else np.nan
        row['actual_stddev'] = actual_rmsf_domain.std() if len(actual_rmsf_domain) > 1 else 0.0

        # Calculate additional metrics per model if domain_group is available
        if not domain_group.empty:
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
                 for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: all_metric_keys.add(f"{report_name}_{metric}") # Add keys

                 # Calculate per-domain uncertainty metrics
                 unc_col = config.get('unc_col')
                 error_col = f"{report_name}_abs_error"
                 if error_col not in domain_group.columns and TARGET_COL in domain_group.columns and pred_col in domain_group.columns:
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
                          row[f"{report_name}_avg_uncertainty"] = np.nan; row[f"{report_name}_uncertainty_error_corr"] = np.nan; row[f"{report_name}_within_1std"] = np.nan
                 else: # Fill NaN if no uncertainty column for this model
                      row[f"{report_name}_avg_uncertainty"] = np.nan; row[f"{report_name}_uncertainty_error_corr"] = np.nan; row[f"{report_name}_within_1std"] = np.nan
                 for unc_metric in ['avg_uncertainty', 'uncertainty_error_corr', 'within_1std']: all_metric_keys.add(f"{report_name}_{unc_metric}") # Add keys

                 # Calculate temperature-specific metrics per domain
                 if 'temperature' in domain_group.columns:
                      for temp, temp_group in domain_group.groupby('temperature'):
                           temp_key_suffix = f"{temp:.1f}K" # Consistent key format
                           mae_t_val, pcc_t_val, rho_t_val = np.nan, np.nan, np.nan
                           if pred_col in temp_group.columns:
                                aligned_temp_df = temp_group[[TARGET_COL, pred_col]].dropna()
                                if len(aligned_temp_df) > 1:
                                     y_true_t = aligned_temp_df[TARGET_COL].values
                                     y_pred_t = aligned_temp_df[pred_col].values
                                     try: mae_t_val = mean_absolute_error(y_true_t, y_pred_t)
                                     except: pass
                                     try: pcc_t_val, _ = safe_pearsonr(y_true_t, y_pred_t)
                                     except: pass
                                     try: rho_t_val, _ = safe_spearmanr(y_true_t, y_pred_t) # Spearman Rank per Temp per Domain
                                     except: pass
                           row[f"{report_name}_mae_{temp_key_suffix}"] = mae_t_val
                           row[f"{report_name}_pcc_{temp_key_suffix}"] = pcc_t_val
                           row[f"{report_name}_spearman_rho_{temp_key_suffix}"] = rho_t_val # Store Spearman Rank
                           all_metric_keys.add(f"{report_name}_mae_{temp_key_suffix}")
                           all_metric_keys.add(f"{report_name}_pcc_{temp_key_suffix}")
                           all_metric_keys.add(f"{report_name}_spearman_rho_{temp_key_suffix}") # Add key

        domain_data_list.append(row)

    if not domain_data_list:
        logger.error("No data compiled for domain-level DataFrame.")
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: No domain data compiled."
        if '_internal_domain_metrics_path' in analysis_results: del analysis_results['_internal_domain_metrics_path']
        return

    try:
        domain_df = pd.DataFrame(domain_data_list)
        if 'domain_id' in domain_df.columns and domain_df['domain_id'].notna().all():
             domain_df = domain_df.set_index('domain_id')
        else: logger.warning("Could not set 'domain_id' as index. Using default.")

        # Ensure all potential metric columns exist before calculating mean
        logger.debug(f"All metric keys tracked for domain mean calc: {sorted(list(all_metric_keys))}")
        for key in all_metric_keys:
            if key not in domain_df.columns and key != 'domain_id': # Ensure column exists
                domain_df[key] = np.nan

        # Calculate mean values across domains (skipna=True is default)
        # Select only numeric columns for mean calculation
        numeric_domain_cols = domain_df.select_dtypes(include=np.number).columns
        if not numeric_domain_cols.empty:
             mean_domain_metrics = domain_df[numeric_domain_cols].mean().to_frame(name='Mean Value Across Domains')
             # Sort the index for better readability
             mean_domain_metrics = mean_domain_metrics.sort_index()
             analysis_results['7.1 MEAN ACROSS DOMAINS'] = format_table(mean_domain_metrics)
        else:
             analysis_results['7.1 MEAN ACROSS DOMAINS'] = "No numeric domain metrics to average."


        # --- Save domain_df for Case Study Selection ---
        # Get output directory from args passed to run_analysis
        output_dir_for_saving = os.path.dirname(analysis_results['_internal_output_file_path']) # Use stored path
        domain_analysis_dir = os.path.join(output_dir_for_saving, "domain_analysis_data") # Changed subdir name
        os.makedirs(domain_analysis_dir, exist_ok=True)
        domain_metrics_path = os.path.join(domain_analysis_dir, "detailed_domain_metrics.csv") # Changed filename
        domain_df.to_csv(domain_metrics_path)
        logger.info(f"Saved detailed domain metrics CSV to: {domain_metrics_path}")
        analysis_results['_internal_domain_metrics_path'] = domain_metrics_path # Store for later steps

    except Exception as e:
        logger.error(f"Error calculating, formatting, or saving domain metrics: {e}", exc_info=True)
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = f"Error: {e}"
        if '_internal_domain_metrics_path' in analysis_results: del analysis_results['_internal_domain_metrics_path']


# --- Statistical Significance Testing (Keep existing function) ---
def calculate_significance_tests(per_domain_metrics, primary_model_report_name, baseline_report_names):
    """Performs Wilcoxon tests comparing primary model to baselines on per-domain metrics."""
    logger.info(f"Performing significance tests comparing {primary_model_report_name} against {baseline_report_names}...")
    results = {}
    if not per_domain_metrics:
        logger.warning("Per-domain metrics dictionary is empty. Skipping significance tests.")
        return results

    first_domain_key = next(iter(per_domain_metrics), None)
    if not first_domain_key or primary_model_report_name not in per_domain_metrics.get(first_domain_key, {}):
         logger.error(f"Primary model '{primary_model_report_name}' not found in per-domain metrics. Cannot perform tests.")
         return results

    domain_ids_primary = list(per_domain_metrics.keys())

    def get_paired_values(metric_key):
        primary_values = []
        baseline_values = []
        for domain_id in domain_ids_primary:
             primary_val = per_domain_metrics.get(domain_id, {}).get(primary_model_report_name, {}).get(metric_key, np.nan)
             baseline_val = per_domain_metrics.get(domain_id, {}).get(baseline_name, {}).get(metric_key, np.nan)
             if not np.isnan(primary_val) and not np.isnan(baseline_val):
                 primary_values.append(primary_val)
                 baseline_values.append(baseline_val)
        return primary_values, baseline_values

    for baseline_name in baseline_report_names:
        if not first_domain_key or baseline_name not in per_domain_metrics.get(first_domain_key, {}):
            logger.warning(f"Baseline model '{baseline_name}' not found in per-domain metrics. Skipping tests against it.")
            results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': 0}
            results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': 0}
            continue

        primary_mae, baseline_mae = get_paired_values('mae')
        primary_pcc, baseline_pcc = get_paired_values('pcc')

        n_mae = len(primary_mae)
        if n_mae < 10:
            logger.warning(f"Too few MAE pairs ({n_mae}) for Wilcoxon test vs {baseline_name}. Skipping.")
            results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_mae}
        else:
            try:
                stat_mae, p_mae = wilcoxon(primary_mae, baseline_mae, alternative='less', zero_method='zsplit')
                results[f"{baseline_name}_vs_MAE"] = {'p_value': p_mae, 'statistic': stat_mae, 'N': n_mae}
            except ValueError as e:
                 logger.warning(f"Wilcoxon MAE test failed vs {baseline_name}: {e}")
                 results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_mae}

        n_pcc = len(primary_pcc)
        if n_pcc < 10:
            logger.warning(f"Too few PCC pairs ({n_pcc}) for Wilcoxon test vs {baseline_name}. Skipping.")
            results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_pcc}
        else:
            try:
                stat_pcc, p_pcc = wilcoxon(primary_pcc, baseline_pcc, alternative='greater', zero_method='zsplit')
                results[f"{baseline_name}_vs_PCC"] = {'p_value': p_pcc, 'statistic': stat_pcc, 'N': n_pcc}
            except ValueError as e:
                 logger.warning(f"Wilcoxon PCC test failed vs {baseline_name}: {e}")
                 results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': n_pcc}

    logger.info("Significance testing complete.")
    return results


# --- Stratified Performance Analysis (Modified for Flexibility) ---
def run_stratified_performance(df, analysis_results, strat_col, section_title, n_bins=None, label_map=None, bin_col_suffix='_bin', sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae'):
    """Calculates performance metrics stratified by a given column (categorical or binned continuous)."""
    logger.info(f"Running stratified performance analysis by '{strat_col}'...")

    if strat_col not in df.columns:
        analysis_results[section_title] = f"Skipped: Column '{strat_col}' not found."
        return

    target_col = TARGET_COL
    if target_col not in df.columns:
        analysis_results[section_title] = f"Skipped: Target column '{target_col}' not found."
        return

    strata_results = []
    df_copy = df[[target_col, strat_col] + [cfg['pred_col'] for cfg in MODEL_CONFIG.values() if cfg['pred_col'] in df.columns]].copy() # Select only needed columns

    grouping_col = strat_col
    index_name = strat_col # Default index name

    # --- Bin continuous data if n_bins is specified ---
    if n_bins is not None and n_bins > 0 and pd.api.types.is_numeric_dtype(df_copy[strat_col]):
        bin_col_name = f"{strat_col}{bin_col_suffix}"
        grouping_col = bin_col_name
        index_name = f"{strat_col.replace('_', ' ').title()} Bin" # Nicer index name
        try:
            # Drop NaNs *before* binning the stratification column
            df_copy.dropna(subset=[strat_col], inplace=True)
            if df_copy.empty: raise ValueError("DataFrame empty after dropping NaNs in strat_col.")

            df_copy[bin_col_name], bin_edges = pd.qcut(df_copy[strat_col], q=n_bins, labels=False, duplicates='drop', retbins=True)
            actual_bins = df_copy[bin_col_name].nunique()
            if actual_bins < 2:
                 raise ValueError(f"Could not create at least 2 bins for '{strat_col}'.")

            # Create string labels from edges for readability in the table
            final_bin_labels = {i: f"{bin_edges[i]:.3g}-{bin_edges[i+1]:.3g}" for i in range(len(bin_edges)-1)}
            # Apply labels to the column used for grouping, handle potential map errors
            df_copy[bin_col_name] = df_copy[bin_col_name].map(final_bin_labels).fillna("Binning Error") # Map numeric bin index to string range
            logger.debug(f"Binned '{strat_col}' into {actual_bins} bins.")

        except Exception as e:
            logger.error(f"Failed to bin column '{strat_col}': {e}. Skipping stratification.")
            analysis_results[section_title] = f"Skipped: Error binning column '{strat_col}' ({e})."
            return
    elif label_map: # Apply label mapping for categorical encoded columns
         df_copy[grouping_col] = df_copy[grouping_col].map(label_map).fillna('Unknown')
         index_name = strat_col.replace('_encoded', '').replace('_', ' ').title()


    # --- Group and Aggregate ---
    # Drop rows where the grouping column OR target is NaN
    df_copy.dropna(subset=[grouping_col, target_col], inplace=True)
    if df_copy.empty:
        logger.warning(f"DataFrame empty after dropping NaNs in group/target for '{strat_col}'. Skipping.")
        analysis_results[section_title] = f"Skipped: No valid data after creating groups/bins for '{strat_col}'."
        return

    # Use observed=True for categorical data created by binning/mapping
    grouped = df_copy.groupby(grouping_col, observed=True)

    for group_name, group_df in grouped:
         if len(group_df) < 5: continue # Need min samples for reliable metrics

         row = {grouping_col: group_name, 'count': len(group_df)}

         # Calculate mean actual RMSF in the group
         row['mean_actual_rmsf'] = group_df[target_col].mean()

         for report_name, config in MODEL_CONFIG.items():
              pred_col = config['pred_col']
              mae_val, pcc_val, rho_val = np.nan, np.nan, np.nan # Add rho_val
              if pred_col in group_df.columns:
                   df_valid = group_df[[target_col, pred_col]].dropna() # Ensure pairwise valid
                   if len(df_valid) > 1:
                        y_true = df_valid[target_col].values
                        y_pred = df_valid[pred_col].values
                        try: mae_val = mean_absolute_error(y_true, y_pred)
                        except: pass
                        try: pcc_val, _ = safe_pearsonr(y_true, y_pred)
                        except: pass
                        try: rho_val, _ = safe_spearmanr(y_true, y_pred) # Calculate Spearman
                        except: pass
              row[f'{report_name}_mae'] = mae_val
              row[f'{report_name}_pcc'] = pcc_val
              row[f'{report_name}_spearman_rho'] = rho_val # Store Spearman

         strata_results.append(row)

    # --- Format and Save ---
    if strata_results:
         strata_df = pd.DataFrame(strata_results).set_index(grouping_col)
         strata_df.index.name = index_name

         # Sort by specified metric or fallback to count
         try:
             if sort_by_metric and sort_by_metric in strata_df.columns and strata_df[sort_by_metric].notna().any():
                  # Determine ascending based on metric type (lower MAE is better, higher PCC/Rho is better)
                  ascending_sort = 'mae' in sort_by_metric.lower() or 'error' in sort_by_metric.lower()
                  strata_df = strata_df.sort_values(sort_by_metric, ascending=ascending_sort)
             else:
                  logger.debug(f"Sorting stratified results by count descending as metric '{sort_by_metric}' not found or all NaN.")
                  strata_df = strata_df.sort_values('count', ascending=False) # Fallback sort
         except Exception as sort_e:
              logger.warning(f"Could not sort stratified results by '{sort_by_metric}', falling back to count: {sort_e}")
              strata_df = strata_df.sort_values('count', ascending=False)

         analysis_results[section_title] = format_table(strata_df)
    else: analysis_results[section_title] = f"No results after grouping by '{strat_col}' (groups might be too small)."


# --- Specific Stratified Analysis Callers ---
def run_amino_acid_performance(df, analysis_results):
    run_stratified_performance(df, analysis_results,
                               strat_col='resname',
                               section_title='8. PERFORMANCE BY AMINO ACID',
                               sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae')

def run_norm_resid_performance(df, analysis_results):
    run_stratified_performance(df, analysis_results,
                               strat_col='normalized_resid',
                               section_title='9. PERFORMANCE BY NORMALIZED RESIDUE POSITION',
                               n_bins=5,
                               sort_by_metric=None) # Sort by bin index

def run_core_exterior_performance(df, analysis_results):
    core_label_map = {0: 'Core', 1: 'Exterior'}
    run_stratified_performance(df, analysis_results,
                               strat_col='core_exterior_encoded',
                               section_title='10. PERFORMANCE BY CORE/EXTERIOR',
                               label_map=core_label_map,
                               sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae')

def run_secondary_structure_performance(df, analysis_results):
    ss_label_map = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
    run_stratified_performance(df, analysis_results,
                               strat_col='secondary_structure_encoded',
                               section_title='11. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)',
                               label_map=ss_label_map,
                               sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae')

def run_rel_accessibility_performance(df, analysis_results):
    run_stratified_performance(df, analysis_results,
                               strat_col='relative_accessibility',
                               section_title='13. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE',
                               n_bins=5,
                               sort_by_metric=None) # Sort by bin index

def run_bfactor_performance(df, analysis_results):
    # Use bfactor_norm if available, otherwise try bfactor
    bfactor_col = 'bfactor_norm' if 'bfactor_norm' in df.columns else 'bfactor' if 'bfactor' in df.columns else None
    if bfactor_col:
        run_stratified_performance(df, analysis_results,
                                   strat_col=bfactor_col,
                                   section_title=f'14. PERFORMANCE BY {bfactor_col.upper()} QUANTILE',
                                   n_bins=5,
                                   sort_by_metric=None) # Sort by bin index
    else:
        analysis_results['14. PERFORMANCE BY BFACTOR QUANTILE'] = "Skipped: No 'bfactor_norm' or 'bfactor' column found."

# --- Performance vs Actual RMSF (Keep Existing - check name usage) ---
def calculate_performance_vs_actual_rmsf(df, target_col='rmsf', model_pred_col='DeepFlex_rmsf', n_bins=10): # Updated default model_pred_col
    """Calculates performance metrics binned by actual RMSF magnitude."""
    logger.info(f"Analyzing performance vs actual '{target_col}' magnitude for model prediction '{model_pred_col}'...")
    results = []
    required_cols = [target_col, model_pred_col]

    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"DataFrame missing required columns for RMSF binning: {missing}. Skipping.")
        return pd.DataFrame()

    df_valid = df[required_cols].dropna().copy()
    if df_valid.empty or len(df_valid) < n_bins * 2:
        logger.warning(f"Not enough valid points ({len(df_valid)}) for RMSF binning (need >= {n_bins*2}). Skipping.")
        return pd.DataFrame()

    try:
        df_valid['rmsf_quantile_bin'], bin_edges = pd.qcut(df_valid[target_col], q=n_bins, labels=False, duplicates='drop', retbins=True)
        actual_num_bins = df_valid['rmsf_quantile_bin'].nunique()

        if actual_num_bins < 2:
             logger.warning(f"Only {actual_num_bins} distinct RMSF bins created. Skipping.")
             return pd.DataFrame()
        if actual_num_bins < n_bins:
            logger.warning(f"Created only {actual_num_bins} bins (requested {n_bins}) due to duplicate edges.")

        bin_range_labels = {i: f"{bin_edges[i]:.3g}-{bin_edges[i+1]:.3g}" for i in range(len(bin_edges)-1)}

        grouped = df_valid.groupby('rmsf_quantile_bin')

        for bin_label_float, group in grouped:
             bin_label = int(bin_label_float)
             if bin_label >= actual_num_bins: continue # Should not happen with qcut labels=False
             if len(group) < 2: continue

             actual_bin = group[target_col].values
             predicted_bin = group[model_pred_col].values

             mae, pcc, rmse, r2, rho = np.nan, np.nan, np.nan, np.nan, np.nan # Add rho
             try: mae = mean_absolute_error(actual_bin, predicted_bin)
             except: pass
             try: pcc, _ = safe_pearsonr(actual_bin, predicted_bin)
             except: pass
             try: rho, _ = safe_spearmanr(actual_bin, predicted_bin) # Add Spearman
             except: pass
             try: rmse = np.sqrt(mean_squared_error(actual_bin, predicted_bin))
             except: pass
             try: r2 = r2_score(actual_bin, predicted_bin) if np.var(actual_bin) > 1e-9 else np.nan
             except: pass

             results.append({
                 'RMSF_Quantile_Bin': bin_label,
                 'RMSF_Range': bin_range_labels.get(bin_label, "N/A"),
                 'Count': len(group),
                 'Mean_Actual_RMSF': np.mean(actual_bin),
                 'MAE': mae,
                 'PCC': pcc,
                 'SpearmanRho': rho, # Added Spearman
                 'RMSE': rmse,
                 'R2': r2
             })

    except ValueError as ve:
         if "Bin edges must be unique" in str(ve) or "fewer than 2 bins" in str(ve):
              logger.warning(f"RMSF binning failed due to data distribution: {ve}. Skipping.")
              return pd.DataFrame()
         else: logger.error(f"ValueError during RMSF binning: {ve}", exc_info=True); return pd.DataFrame()
    except Exception as e: logger.error(f"Error during RMSF binning: {e}", exc_info=True); return pd.DataFrame()

    if not results: logger.warning("No results generated from RMSF binning.")
    return pd.DataFrame(results).sort_values('RMSF_Quantile_Bin')

def run_performance_vs_actual_rmsf(df, analysis_results, primary_model_report_name=PRIMARY_MODEL_NAME): # Use central primary name
    logger.info("Running performance vs actual RMSF analysis...")
    section_title = f"12. {primary_model_report_name.upper()} PERFORMANCE BY ACTUAL RMSF QUANTILE" # Updated title
    primary_model_config = MODEL_CONFIG.get(primary_model_report_name)
    if not primary_model_config:
        logger.error(f"Primary model '{primary_model_report_name}' not found in config.")
        analysis_results[section_title] = "Primary model config missing."
        return

    pred_col = primary_model_config['pred_col']
    binned_df = calculate_performance_vs_actual_rmsf(df, target_col=TARGET_COL, model_pred_col=pred_col, n_bins=10)

    if not binned_df.empty:
        binned_df.rename(columns={'MAE': f'{primary_model_report_name}_MAE',
                                 'PCC': f'{primary_model_report_name}_PCC',
                                 'SpearmanRho': f'{primary_model_report_name}_SpearmanRho', # Added Spearman
                                 'RMSE': f'{primary_model_report_name}_RMSE',
                                 'R2': f'{primary_model_report_name}_R2'}, inplace=True)
        binned_df = binned_df.set_index('RMSF_Range')
        analysis_results[section_title] = format_table(binned_df.drop(columns=['RMSF_Quantile_Bin']))
    else:
        analysis_results[section_title] = "No results generated or error during binning."


# --- Model Disagreement Analysis (Keep Existing - check name usage) ---
def run_model_disagreement_analysis(df, analysis_results, primary_model_report_name=PRIMARY_MODEL_NAME): # Use central primary name
    logger.info("Running model disagreement analysis...")
    section_title = "15. MODEL DISAGREEMENT VS. ERROR"
    pred_cols = [config['pred_col'] for config in MODEL_CONFIG.values() if config['pred_col'] in df.columns]
    if len(pred_cols) < 2:
        analysis_results[section_title] = "Skipped: Need predictions from at least 2 models."
        return

    df_copy = df.copy() # Work on a copy
    df_preds_only = df_copy[pred_cols]
    for col in df_preds_only.columns: # Ensure numeric
        df_preds_only[col] = pd.to_numeric(df_preds_only[col], errors='coerce')

    df_copy['prediction_stddev'] = df_preds_only.std(axis=1, skipna=True)
    df_copy['prediction_range'] = df_preds_only.max(axis=1, skipna=True) - df_preds_only.min(axis=1, skipna=True) # Add range as alternative measure

    # Add stats on the disagreement metrics
    try:
        stats_std = df_copy['prediction_stddev'].describe().to_frame(name='Prediction StdDev')
        stats_range = df_copy['prediction_range'].describe().to_frame(name='Prediction Range')
        stats_combined = pd.concat([stats_std, stats_range], axis=1)
        analysis_results[section_title] = "MODEL PREDICTION DISAGREEMENT STATS\n" + format_table(stats_combined) + "\n\nNote: Std Dev/Range calculated across available model predictions per residue.\n"
    except Exception as e:
         logger.warning(f"Could not calculate disagreement stats: {e}")
         analysis_results[section_title] = "Could not calculate disagreement stats.\n"

    # Correlate disagreement with primary model error and uncertainty
    primary_model_config = MODEL_CONFIG.get(primary_model_report_name)
    if primary_model_config:
         primary_pred_col = primary_model_config['pred_col']
         primary_unc_col = primary_model_config.get('unc_col')
         error_col = f"{primary_model_report_name}_abs_error" # Consistent error col name

         # Ensure error column exists
         if error_col not in df_copy.columns:
              if TARGET_COL in df_copy.columns and primary_pred_col in df_copy.columns:
                   df_copy[error_col] = (df_copy[primary_pred_col] - df_copy[TARGET_COL]).abs()

         corr_lines = []
         # Correlation with Absolute Error
         if error_col in df_copy.columns and 'prediction_stddev' in df_copy.columns:
              df_valid_corr_std = df_copy[[error_col, 'prediction_stddev']].dropna()
              df_valid_corr_range = df_copy[[error_col, 'prediction_range']].dropna()

              if len(df_valid_corr_std) > 1:
                   pcc_err_std, _ = safe_pearsonr(df_valid_corr_std['prediction_stddev'], df_valid_corr_std[error_col])
                   corr_lines.append(f"PCC(Prediction StdDev vs {error_col}): {pcc_err_std:.3f} (N={len(df_valid_corr_std)})")
              else: corr_lines.append(f"PCC(Prediction StdDev vs {error_col}): Not enough data.")

              if len(df_valid_corr_range) > 1:
                   pcc_err_range, _ = safe_pearsonr(df_valid_corr_range['prediction_range'], df_valid_corr_range[error_col])
                   corr_lines.append(f"PCC(Prediction Range vs {error_col}): {pcc_err_range:.3f} (N={len(df_valid_corr_range)})")
              else: corr_lines.append(f"PCC(Prediction Range vs {error_col}): Not enough data.")

         else: corr_lines.append(f"Correlation with {error_col} skipped (column missing).")

         # Correlation with Uncertainty (if available)
         if primary_unc_col and primary_unc_col in df_copy.columns:
             if 'prediction_stddev' in df_copy.columns:
                  df_valid_unc_std = df_copy[[primary_unc_col, 'prediction_stddev']].dropna()
                  if len(df_valid_unc_std) > 1:
                       pcc_unc_std, _ = safe_pearsonr(df_valid_unc_std['prediction_stddev'], df_valid_unc_std[primary_unc_col])
                       corr_lines.append(f"PCC(Prediction StdDev vs {primary_model_report_name} Uncertainty): {pcc_unc_std:.3f} (N={len(df_valid_unc_std)})")
                  else: corr_lines.append(f"PCC(Prediction StdDev vs {primary_model_report_name} Uncertainty): Not enough data.")

             if 'prediction_range' in df_copy.columns:
                  df_valid_unc_range = df_copy[[primary_unc_col, 'prediction_range']].dropna()
                  if len(df_valid_unc_range) > 1:
                      pcc_unc_range, _ = safe_pearsonr(df_valid_unc_range['prediction_range'], df_valid_unc_range[primary_unc_col])
                      corr_lines.append(f"PCC(Prediction Range vs {primary_model_report_name} Uncertainty): {pcc_unc_range:.3f} (N={len(df_valid_unc_range)})")
                  else: corr_lines.append(f"PCC(Prediction Range vs {primary_model_report_name} Uncertainty): Not enough data.")

         elif primary_unc_col: corr_lines.append(f"Correlation with {primary_model_report_name} Uncertainty skipped (column '{primary_unc_col}' missing).")
         else: corr_lines.append(f"Correlation with {primary_model_report_name} Uncertainty skipped (no uncertainty column defined).")

         analysis_results[section_title] += "\n".join(corr_lines)

    else: analysis_results[section_title] += f"\nPrimary model '{primary_model_report_name}' for error correlation not found."


# --- Case Study Candidate Selection (Keep Existing - check name usage) ---
def run_case_study_candidate_selection(analysis_results, primary_model_report_name=PRIMARY_MODEL_NAME, n_candidates=15): # Keep n_candidates=15
    """Selects and formats candidate domains for case studies based on primary model performance, ensuring lists are mutually exclusive."""
    logger.info(f"Selecting case study candidates (Focusing on {primary_model_report_name} Metrics, n={n_candidates} per category, exclusive lists)...") # Updated log message
    section_title = "16. CASE STUDY CANDIDATES"
    analysis_results[section_title] = f"(Based on {primary_model_report_name} Metrics. Max {n_candidates} per category. Lists are mutually exclusive.)" # Updated description

    # Define Primary Model Specific Columns potentially needed (adjust based on run_domain_level_analysis output)
    primary_mae_col = f'{primary_model_report_name}_mae'
    primary_pcc_col = f'{primary_model_report_name}_pcc'
    primary_rho_col = f'{primary_model_report_name}_spearman_rho' # If added to domain analysis
    # Temperature specific examples (check exact suffix from run_domain_level_analysis)
    mae_320_col = f'{primary_model_report_name}_mae_320.0K'
    mae_450_col = f'{primary_model_report_name}_mae_450.0K'
    pcc_320_col = f'{primary_model_report_name}_pcc_320.0K'
    pcc_450_col = f'{primary_model_report_name}_pcc_450.0K'
    rho_320_col = f'{primary_model_report_name}_spearman_rho_320.0K' # Spearman example
    rho_450_col = f'{primary_model_report_name}_spearman_rho_450.0K' # Spearman example

    base_cols_to_show = ['count_residues', 'actual_mean', 'actual_stddev']
    primary_model_cols_to_show = [primary_mae_col, primary_pcc_col, primary_rho_col,
                                  mae_320_col, pcc_320_col, rho_320_col,
                                  mae_450_col, pcc_450_col, rho_450_col]
    # Deltas can be calculated on the fly if needed
    delta_mae_col = 'delta_mae_320_450'
    delta_pcc_col = 'delta_pcc_320_450'
    delta_rho_col = 'delta_rho_320_450'
    delta_actual_rmsf_col = 'delta_actual_rmsf_320_450' # If added to domain data

    # Get the path saved by run_domain_level_analysis
    domain_metrics_path = analysis_results.get('_internal_domain_metrics_path')

    if not domain_metrics_path or not os.path.exists(domain_metrics_path):
        analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = "Skipped: Domain metrics file not found (expected from Section 7)."
        analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = "Skipped: Domain metrics file not found."
        analysis_results['16.3 CHALLENGING CANDIDATES'] = "Skipped: Domain metrics file not found."
        logger.warning("Domain metrics file path missing or file not found. Skipping case study selection.")
        return

    try:
        domain_df = pd.read_csv(domain_metrics_path, index_col='domain_id')
        logger.info(f"Loaded domain metrics from {domain_metrics_path} for case study selection.")

        # --- Ensure Necessary Columns Exist & Calculate Deltas ---
        required_cols_exist = all(c in domain_df.columns for c in [primary_pcc_col, primary_mae_col])
        if not required_cols_exist:
             raise ValueError(f"Domain metrics file missing essential columns: {primary_pcc_col}, {primary_mae_col}")

        # Calculate deltas robustly if needed and possible
        cols_calculated = []
        if delta_mae_col not in domain_df.columns and mae_320_col in domain_df.columns and mae_450_col in domain_df.columns:
            domain_df[delta_mae_col] = domain_df[mae_450_col] - domain_df[mae_320_col]; cols_calculated.append(delta_mae_col)
        if delta_pcc_col not in domain_df.columns and pcc_320_col in domain_df.columns and pcc_450_col in domain_df.columns:
            domain_df[delta_pcc_col] = domain_df[pcc_450_col] - domain_df[pcc_320_col]; cols_calculated.append(delta_pcc_col)
        if delta_rho_col not in domain_df.columns and rho_320_col in domain_df.columns and rho_450_col in domain_df.columns:
            domain_df[delta_rho_col] = domain_df[rho_450_col] - domain_df[rho_320_col]; cols_calculated.append(delta_rho_col)

        # Filter display columns to only those present or calculated
        all_cols_to_show = base_cols_to_show + primary_model_cols_to_show + cols_calculated
        cols_to_display_final = [col for col in all_cols_to_show if col in domain_df.columns]
        cols_to_display_final = list(dict.fromkeys(cols_to_display_final)) # Unique cols in order

        # Keep track of selected domains to ensure exclusivity
        selected_domain_ids = set()

        # --- Candidate Selection ---

        # 1. High Accuracy Candidates
        try:
            criteria = (domain_df[primary_pcc_col] > 0.93) & (domain_df[primary_mae_col] < 0.12)
            if 'actual_stddev' in domain_df.columns: criteria &= (domain_df['actual_stddev'] > 0.15)

            high_acc_all = domain_df[criteria] # Get all that meet criteria
            # Select top N and add to selected set
            high_acc_display = high_acc_all[cols_to_display_final].nsmallest(n_candidates, primary_mae_col)
            selected_domain_ids.update(high_acc_display.index.tolist()) # Add selected IDs

            analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = f"Criteria: PCC>{0.93}, MAE<{0.12}" + \
                 f"{', ActualStd>0.15' if 'actual_stddev' in domain_df.columns else ''}. Found {len(high_acc_all)} matching, showing top {len(high_acc_display)}.\n" + \
                 format_table(high_acc_display, floatfmt=".3f")
        except Exception as e:
             logger.warning(f"Error selecting High Accuracy Candidates: {e}", exc_info=True)
             analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = f"Error during selection: {e}"

        # 2. Good Temperature Handling Candidates (Exclusive of High Accuracy)
        try:
            required_temp_hand = [primary_pcc_col, delta_mae_col, delta_pcc_col]
            if all(c in domain_df.columns for c in required_temp_hand):
                criteria = (domain_df[primary_pcc_col] > 0.90) & \
                           (domain_df[delta_mae_col].abs() < 0.10) & \
                           (domain_df[delta_pcc_col] > -0.05)

                # Exclude domains already selected in High Accuracy
                temp_handle_pool = domain_df[criteria & ~domain_df.index.isin(selected_domain_ids)].copy()

                if not temp_handle_pool.empty:
                    abs_sort_col = 'abs_' + delta_mae_col
                    temp_handle_pool[abs_sort_col] = temp_handle_pool[delta_mae_col].abs()
                    temp_handle_sorted = temp_handle_pool.nsmallest(n_candidates, abs_sort_col)
                    temp_handle_display = temp_handle_sorted[cols_to_display_final]
                    selected_domain_ids.update(temp_handle_display.index.tolist()) # Add newly selected IDs
                else:
                    temp_handle_display = pd.DataFrame() # Empty DataFrame if none qualify after exclusion

                analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = f"Criteria: PCC>{0.90}, |DeltaMAE|<{0.10}, DeltaPCC>{-0.05}. Found {len(temp_handle_pool)} matching (excl. High Acc), showing top {len(temp_handle_display)}.\n" + \
                     format_table(temp_handle_display, floatfmt=".3f")
            else:
                analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = f"Skipped: Missing columns for Temp Handling criteria: {required_temp_hand}"
        except Exception as e:
             logger.warning(f"Error selecting Temp Handling Candidates: {e}", exc_info=True)
             analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = f"Error during selection: {e}"

        # 3. Challenging Candidates (Exclusive of High Accuracy and Temp Handling)
        try:
            required_challenging = [primary_pcc_col, primary_mae_col]
            if all(c in domain_df.columns for c in required_challenging):
                 is_challenging = (domain_df[primary_pcc_col] < 0.80) | (domain_df[primary_mae_col] > 0.25)
                 if delta_mae_col in domain_df.columns: is_challenging |= (domain_df[delta_mae_col] > 0.20)

                 # Exclude domains selected in *either* previous category
                 challenging_pool = domain_df[is_challenging & ~domain_df.index.isin(selected_domain_ids)]

                 challenging_display = challenging_pool[cols_to_display_final].nlargest(n_candidates, primary_mae_col)
                 # No need to update selected_domain_ids after the last category

                 analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Criteria: PCC<{0.80} OR MAE>{0.25}" + \
                     f"{' OR DeltaMAE>0.20' if delta_mae_col in domain_df.columns else ''}. Found {len(challenging_pool)} matching (excl. previous lists), showing top {len(challenging_display)}.\n" + \
                     format_table(challenging_display, floatfmt=".3f")
            else:
                 analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Skipped: Missing columns for Challenging criteria: {required_challenging}"
        except Exception as e:
             logger.warning(f"Error selecting Challenging Candidates: {e}", exc_info=True)
             analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Error during selection: {e}"

    except FileNotFoundError:
        logger.error(f"Domain metrics file not found at {domain_metrics_path}. Cannot select candidates.")
        analysis_results['16. CASE STUDY CANDIDATES'] += "\nSkipped: Domain metrics file not found."
    except Exception as e:
        logger.error(f"Error during case study selection using {domain_metrics_path}: {e}", exc_info=True)
        analysis_results['16. CASE STUDY CANDIDATES'] += f"\nError reading/processing domain metrics file: {e}"


# --- Placeholder Sections for Analyses Requiring External Data/Models ---

def run_feature_attribution_placeholder(analysis_results):
    logger.info("Placeholder for Feature Attribution analysis.")
    analysis_results['18. FEATURE ATTRIBUTION & INTERPRETABILITY (Placeholder)'] = \
        "This analysis requires access to the trained model object(s) and appropriate libraries (e.g., SHAP, Captum).\n" \
        "Or, it requires pre-computed attribution scores (e.g., SHAP values per feature per residue) or attention weights in the input data.\n" \
        "Potential Steps (if data/model available):\n" \
        "1. Load model and a representative data subset.\n" \
        "2. Initialize explainer (SHAP KernelExplainer, DeepExplainer, Integrated Gradients, etc.).\n" \
        "3. Compute attributions for features like:\n" \
        "   - Temperature embedding contribution.\n" \
        "   - Aggregate contribution of ESM sequence embedding dimensions.\n" \
        "   - Aggregate contribution of VoxelFlex-3D inputs (if applicable).\n" \
        "   - Contribution of 1D structural features (SASA, SS, etc.).\n" \
        "4. Summarize average absolute attributions globally or per residue type/region.\n" \
        "5. Visualize mean attributions (e.g., bar plot).\n" \
        "6. If attention weights are available (e.g., from Transformer layers), analyze and visualize attention patterns for key residues or domains."

def run_temperature_ablation_placeholder(analysis_results):
    logger.info("Placeholder for Temperature-Encoding Ablation analysis.")
    analysis_results['19. TEMPERATURE-ENCODING ABLATION (Placeholder)'] = \
        "This analysis requires results from a model trained *without* the temperature conditioning mechanism.\n" \
        "Input Needed: Predictions from the ablated model (e.g., a column like 'DeepFlex_NoTemp_rmsf' in the dataset).\n" \
        "Analysis Steps (if ablated predictions available):\n" \
        "1. Add the ablated model to `MODEL_CONFIG`.\n" \
        "2. Re-run the performance comparison (Section 5), focusing on the difference between the full model and the ablated model.\n" \
        "3. Compare overall metrics (MAE, PCC, Rho).\n" \
        "4. Compare per-temperature metrics to quantify the performance drop at different temperatures when conditioning is removed.\n" \
        "5. Calculate the delta in MAE/PCC between the full and ablated model per domain and analyze the distribution of this delta."

def run_external_validation_placeholder(analysis_results):
    logger.info("Placeholder for Cross-Dataset / External Validation.")
    analysis_results['20. CROSS-DATASET / EXTERNAL VALIDATION (Placeholder)'] = \
        "This analysis requires an independent dataset with ground truth RMSF values (e.g., from different MD simulations, experimental sources, or a held-out test set not used in training/hyperparameter tuning).\n" \
        "Input Needed: Path to the external validation CSV file (must have compatible columns: residue identifiers, temperature, actual RMSF).\n" \
        "Analysis Steps:\n" \
        "1. Load the external dataset.\n" \
        "2. Generate predictions using the trained DeepFlex model for the structures/residues in the external dataset.\n" \
        "   (This step requires running the model inference, not just analyzing existing predictions).\n" \
        "3. Merge predictions with the external ground truth.\n" \
        "4. Calculate performance metrics (MAE, PCC, Spearman Rho) on this external dataset.\n" \
        "5. Compare these metrics to the performance on the internal test set to assess generalization.\n" \
        "6. Optionally, stratify performance on the external set by available features (temperature, protein type, etc.)."

def run_computational_performance_placeholder(analysis_results):
    logger.info("Placeholder for Computational Performance analysis.")
    analysis_results['21. COMPUTATIONAL PERFORMANCE & SCALABILITY (Placeholder)'] = \
        "This section requires benchmark data obtained separately from running model inference and MD simulations.\n" \
        "Information Needed:\n" \
        "1. DeepFlex Inference Time:\n" \
        "   - Average time per protein or per residue.\n" \
        "   - Specify hardware used (GPU model, CPU type, number of cores).\n" \
        "   - Memory requirements (GPU RAM, System RAM).\n" \
        "2. MD Simulation Time:\n" \
        "   - Time for a standard simulation (e.g., 100 ns) of a representative protein.\n" \
        "   - Specify hardware used (GPU model) and simulation software (e.g., GROMACS, AMBER).\n" \
        "   - Number of atoms in the simulated system.\n" \
        "3. Comparison:\n" \
        "   - Calculate the approximate speedup factor (MD time / Inference time) for obtaining flexibility estimates.\n" \
        "   - Discuss scalability with protein size (how inference time changes with sequence length)." \
        "\nExample Placeholder Values (Update with actual measurements):\n" \
        " - DeepFlex Inference Time: ~X seconds per protein (Avg Size Y residues) on Nvidia Z GPU.\n" \
        " - MD Simulation Time (100ns): ~H hours for Protein P (N atoms) on Nvidia Z GPU.\n" \
        " - Estimated Speedup: ~ Factor S"

# --- NEW: Minor Additional Analyses ---
def run_error_vs_feature_analysis(df, analysis_results):
    logger.info("Running Error vs Feature analyses...")
    analysis_results['22. ERROR ANALYSIS VS FEATURES'] = f"(Error = Absolute Error for {PRIMARY_MODEL_NAME})"

    primary_model_config = MODEL_CONFIG.get(PRIMARY_MODEL_NAME)
    if not primary_model_config:
        logger.error(f"Primary model '{PRIMARY_MODEL_NAME}' not found for error analysis.")
        analysis_results['22. ERROR ANALYSIS VS FEATURES'] = "Skipped: Primary model config missing."
        return

    pred_col = primary_model_config['pred_col']
    error_col = f"{PRIMARY_MODEL_NAME}_abs_error"

    df_copy = df.copy()
    # Calculate error if missing
    if error_col not in df_copy.columns:
        if TARGET_COL in df_copy.columns and pred_col in df_copy.columns:
            df_copy[error_col] = (df_copy[pred_col] - df_copy[TARGET_COL]).abs()
        else:
            logger.error(f"Cannot calculate error '{error_col}'. Skipping Error vs Feature analysis.")
            analysis_results['22. ERROR ANALYSIS VS FEATURES'] = f"Skipped: Cannot calculate error '{error_col}'."
            return

    features_to_analyze = {
        'bfactor_norm': '22.1 ERROR VS. NORMALIZED B-FACTOR',
        'contact_number': '22.2 ERROR VS. CONTACT NUMBER',
        # Add coevolution score column name if available, e.g., 'conservation' or 'coevo_score'
        'coevolution_score': '22.3 ERROR VS. CO-EVOLUTION SIGNAL',
    }

    for feature_col, section_title in features_to_analyze.items():
        if feature_col in df_copy.columns:
            logger.info(f"Analyzing error vs '{feature_col}'...")
            # Use the generic stratification function, but report MAE only
            error_results = []
            df_feature_err = df_copy[[feature_col, error_col]].dropna()
            if df_feature_err.empty or len(df_feature_err) < 50: # Need enough points for binning
                 analysis_results[section_title] = f"Skipped: Not enough valid data points for '{feature_col}' and error."
                 continue

            try:
                 n_bins = 5
                 df_feature_err['feature_bin'], bin_edges = pd.qcut(df_feature_err[feature_col], q=n_bins, labels=False, duplicates='drop', retbins=True)
                 actual_bins = df_feature_err['feature_bin'].nunique()
                 if actual_bins < 2: raise ValueError("Could not create bins.")

                 bin_range_labels = {i: f"{bin_edges[i]:.3g}-{bin_edges[i+1]:.3g}" for i in range(len(bin_edges)-1)}

                 grouped = df_feature_err.groupby('feature_bin')
                 binned_errors = []
                 for bin_idx, group in grouped:
                      if len(group) > 2:
                           binned_errors.append({
                               'Feature Range': bin_range_labels.get(bin_idx, "N/A"),
                               'Mean Feature Value': group[feature_col].mean(),
                               f'Mean {PRIMARY_MODEL_NAME} Abs Error': group[error_col].mean(),
                               f'Median {PRIMARY_MODEL_NAME} Abs Error': group[error_col].median(),
                               'Count': len(group)
                           })

                 if binned_errors:
                      error_df = pd.DataFrame(binned_errors).set_index('Feature Range')
                      analysis_results[section_title] = format_table(error_df)
                 else: analysis_results[section_title] = "No results after binning."

            except Exception as e:
                 logger.warning(f"Could not analyze error vs {feature_col}: {e}")
                 analysis_results[section_title] = f"Error during analysis for {feature_col}: {e}"

        else:
            analysis_results[section_title] = f"Skipped: Feature column '{feature_col}' not found."


# --- Main Analysis Runner ---
def run_analysis(df_path, output_file):
    """Runs the full analysis pipeline with enhanced error handling."""
    start_time = time.time()
    logger.info(f"Starting analysis for file: {df_path}")
    try:
        df = pd.read_csv(df_path)
        # *** Global Rename: Update prediction/uncertainty columns if they exist ***
        old_pred = "Attention_ESM_rmsf"
        new_pred = MODEL_CONFIG[PRIMARY_MODEL_NAME]['pred_col'] # "DeepFlex_rmsf"
        old_unc = "Attention_ESM_rmsf_uncertainty"
        new_unc = MODEL_CONFIG[PRIMARY_MODEL_NAME]['unc_col'] # "DeepFlex_rmsf_uncertainty"

        rename_dict = {}
        if old_pred in df.columns:
            rename_dict[old_pred] = new_pred
            logger.info(f"Renaming column '{old_pred}' to '{new_pred}'")
        if old_unc in df.columns:
            rename_dict[old_unc] = new_unc
            logger.info(f"Renaming column '{old_unc}' to '{new_unc}'")
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)

        df.attrs['source_file'] = df_path
        logger.info(f"Successfully loaded data: {df.shape}")
        if df.empty: logger.error("Input CSV is empty. Aborting."); return
    except FileNotFoundError: logger.error(f"Data file not found: {df_path}"); return
    except pd.errors.EmptyDataError: logger.error(f"Data file is empty: {df_path}"); return
    except Exception as e: logger.error(f"Error loading data: {e}", exc_info=True); return

    analysis_results = {'_internal_output_file_path': output_file} # Store output path for saving domain data
    internal_data = {'original_df': df} # Store potentially modified df

    # Define the order and functions for analysis steps
    analysis_steps = [
        # Setup & Basic Info
        ("1. BASIC INFORMATION", run_basic_info),
        ("2. MISSING VALUE SUMMARY", run_missing_values),
        ("3. OVERALL DESCRIPTIVE STATISTICS (Key Variables)", run_descriptive_stats),
        ("4. DATA DISTRIBUTIONS", run_data_distributions),
        # Overall & Temp Performance
        ("5. COMPREHENSIVE MODEL COMPARISON", run_model_comparison), # Includes Rank Metrics (5.2), Temp Perf (5.3), R2 matrices (5.4, 5.5)
        ("5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS", run_dihedral_analysis), # Added Dihedral
        # Uncertainty & Calibration
        ("6. UNCERTAINTY ANALYSIS", run_uncertainty_analysis), # Includes Reliability (6.5), ECE (6.6)
        # Domain Level
        ("7. DOMAIN-LEVEL PERFORMANCE METRICS", run_domain_level_analysis), # Generates CSV needed for Case Studies
        # Stratified Performance
        ("8. PERFORMANCE BY AMINO ACID", run_amino_acid_performance),
        ("9. PERFORMANCE BY NORMALIZED RESIDUE POSITION", run_norm_resid_performance),
        ("10. PERFORMANCE BY CORE/EXTERIOR", run_core_exterior_performance),
        ("11. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)", run_secondary_structure_performance),
        ("12. PERFORMANCE BY ACTUAL RMSF QUANTILE", lambda d, r: run_performance_vs_actual_rmsf(d, r)), # Use lambda to pass primary model name implicitly
        ("13. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", run_rel_accessibility_performance),
        ("14. PERFORMANCE BY BFACTOR QUANTILE", run_bfactor_performance),
        # Disagreement & Case Studies
        ("15. MODEL DISAGREEMENT VS. ERROR", lambda d, r: run_model_disagreement_analysis(d, r)), # Use lambda
        ("16. CASE STUDY CANDIDATES", lambda d, r: run_case_study_candidate_selection(r)), # Uses results dict
        # Significance (Requires Domain Data)
        ("17. STATISTICAL SIGNIFICANCE TESTS", lambda d, r: None), # Placeholder, called explicitly later
        # Placeholders for Advanced Analyses
        ("18. FEATURE ATTRIBUTION & INTERPRETABILITY (Placeholder)", lambda d, r: run_feature_attribution_placeholder(r)),
        ("19. TEMPERATURE-ENCODING ABLATION (Placeholder)", lambda d, r: run_temperature_ablation_placeholder(r)),
        ("20. CROSS-DATASET / EXTERNAL VALIDATION (Placeholder)", lambda d, r: run_external_validation_placeholder(r)),
        ("21. COMPUTATIONAL PERFORMANCE & SCALABILITY (Placeholder)", lambda d, r: run_computational_performance_placeholder(r)),
        # Minor Error Analyses
        ("22. ERROR ANALYSIS VS FEATURES", run_error_vs_feature_analysis),
    ]

    # Run analysis steps
    parent_headers = set(re.match(r"(\d+)\.", k).group(1) for k,v in analysis_steps if re.match(r"(\d+)\.", k))
    for p_header_num in sorted(parent_headers, key=int):
         parent_key = next((k for k,v in analysis_steps if k.startswith(f"{p_header_num}. ")), None)
         if parent_key: analysis_results[parent_key] = "" # Initialize parent section header

    for section_key, analysis_func in analysis_steps:
        if analysis_func is None and section_key != "17. STATISTICAL SIGNIFICANCE TESTS": continue # Skip true placeholders except sig test

        logger.info(f"--- Running Analysis: {section_key} ---")
        try:
            # Pass the potentially name-updated dataframe
            # Pass analysis_results to functions that need it (placeholders, case studies)
            analysis_func(internal_data['original_df'], analysis_results)
        except Exception as e:
            logger.error(f"Error executing analysis step '{section_key}': {e}", exc_info=True)
            analysis_results[section_key] = f"!!! ERROR DURING ANALYSIS: {e} !!!"

    # Explicitly run significance tests after domain metrics are calculated
    logger.info("--- Running Analysis: 17. STATISTICAL SIGNIFICANCE TESTS ---")
    try:
        per_domain_metrics_data = {} # Reset or ensure it's empty
        domain_metrics_path = analysis_results.get('_internal_domain_metrics_path')
        if domain_metrics_path and os.path.exists(domain_metrics_path):
             try:
                 # Load the CSV and convert back to the required dict format
                 domain_df = pd.read_csv(domain_metrics_path, index_col='domain_id')
                 logger.info(f"Loading domain metrics from {domain_metrics_path} for significance tests.")
                 # Convert DataFrame rows to nested dict {domain_id: {model_name: {'mae': val, 'pcc': val}}}
                 for domain_id, row in domain_df.iterrows():
                      domain_data = {}
                      for model_name in MODEL_CONFIG.keys():
                           mae_col = f"{model_name}_mae"
                           pcc_col = f"{model_name}_pcc"
                           # Check if columns exist and value is not NaN
                           if mae_col in row and pd.notna(row[mae_col]) and pcc_col in row and pd.notna(row[pcc_col]):
                                domain_data[model_name] = {'mae': row[mae_col], 'pcc': row[pcc_col]}
                           # else: include models with NaN? Original did. Let's include them.
                           elif mae_col in row or pcc_col in row: # Include if at least one metric exists (might be NaN)
                                domain_data[model_name] = {'mae': row.get(mae_col, np.nan), 'pcc': row.get(pcc_col, np.nan)}

                      if domain_data: # Only add if data exists for this domain
                           per_domain_metrics_data[domain_id] = domain_data

                 if not per_domain_metrics_data:
                      logger.warning("Loaded domain metrics file, but failed to convert to dict format or it was empty.")
                      raise FileNotFoundError # Trigger recalculation

             except Exception as load_err:
                  logger.warning(f"Failed to load or process domain metrics from {domain_metrics_path}: {load_err}. Recalculating...")
                  per_domain_metrics_data = calculate_per_domain_metrics(internal_data['original_df'])
        else:
             logger.warning("Domain metrics file path not found. Recalculating domain metrics for significance testing.")
             per_domain_metrics_data = calculate_per_domain_metrics(internal_data['original_df'])


        if per_domain_metrics_data:
            significance_results = calculate_significance_tests(
                per_domain_metrics_data,
                PRIMARY_MODEL_NAME,
                KEY_BASELINES_FOR_SIG_TEST
            )
            if significance_results:
                 sig_text = f"Comparing {PRIMARY_MODEL_NAME} against key baselines using Wilcoxon signed-rank test on per-domain metrics.\n"
                 sig_text += "H0: Median difference is zero.\n"
                 sig_text += "MAE Test (alternative='less'): Is {PRIMARY_MODEL_NAME} MAE significantly smaller than baseline?\n"
                 sig_text += "PCC Test (alternative='greater'): Is {PRIMARY_MODEL_NAME} PCC significantly larger than baseline?\n\n"
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
            else: analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = "Significance tests could not be calculated (no valid pairs or other error)."
        else: analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = "Per-domain metrics calculation failed, cannot perform significance tests."
    except Exception as e:
        logger.error(f"Error during significance testing step: {e}", exc_info=True)
        analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = f"!!! ERROR DURING ANALYSIS: {e} !!!"


    # Write results to file
    output_dir = os.path.dirname(output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file, 'w') as f:
            sorted_keys = sorted(analysis_results.keys(), key=parse_key_for_sort)

            for section_key in sorted_keys:
                if section_key.startswith('_internal'): continue # Skip internal keys

                content = analysis_results[section_key]
                match = re.match(r"(\d+(?:\.\d+)*)\.?\s*(.*)", section_key)
                if match:
                    section_num_str = match.group(1)
                    section_title_str = match.group(2).strip().upper() # Uppercase titles
                    # Print header only for top-level sections (e.g., '1.', '5.')
                    if '.' not in section_num_str and section_title_str:
                         f.write("\n" + "=" * 80 + "\n")
                         f.write(f"## {section_num_str}. {section_title_str} ##\n")
                         f.write("=" * 80 + "\n")
                    elif section_title_str: # Print sub-section header
                         f.write("\n" + "-" * 60 + "\n")
                         f.write(f"### {section_num_str} {section_title_str} ###\n")
                         f.write("-" * 60 + "\n")
                    # Else: No header needed if it's just a numbered sub-section without title text

                else: # Fallback for non-standard keys
                     f.write("\n" + "=" * 80 + "\n")
                     f.write(f"## {section_key.upper()} ##\n")
                     f.write("=" * 80 + "\n")

                # Write content only if it's not just the header placeholder or empty
                if content and isinstance(content, str) and content.strip():
                     f.write(content + "\n\n")
                elif content and not isinstance(content, str): # Handle non-string content (e.g., tables)
                     if isinstance(content, pd.DataFrame): f.write(format_table(content) + "\n\n")
                     elif isinstance(content, dict): f.write(json.dumps(content, indent=4, default=str) + "\n\n")
                     else: f.write(str(content) + "\n\n")
                # If content is an empty string (placeholder header), skip writing content.

    except Exception as e:
        logger.error(f"Failed to write analysis results to {output_file}: {e}", exc_info=True)


    end_time = time.time()
    logger.info(f"Analysis complete. Results saved to: {output_file}")
    logger.info(f"Detailed domain metrics saved to CSV in: {os.path.join(os.path.dirname(output_file), 'domain_analysis_data')}")
    logger.info(f"Total analysis time: {end_time - start_time:.2f} seconds.")

# --- Command Line Argument Parsing (Keep existing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive analysis on the aggregated flexibility prediction dataset.")
    parser.add_argument(
        "--input_csv",
        type=str,
        # default="../data/01_final_analysis_dataset.csv", # Adjust default if needed
        default="/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv", # Use user's path
        help="Path to the aggregated input CSV file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        # default="../output/general_analysis_v3.txt", # Adjust default if needed
        default="enhanced_general_analysis_report.txt", # Simpler default name
        help="Path to save the enhanced analysis results text file."
    )
    args = parser.parse_args()

    # Make paths absolute if they are relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    input_csv_path = args.input_csv if os.path.isabs(args.input_csv) else os.path.join(script_dir, args.input_csv)
    output_file_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(script_dir, args.output_file)

    # Ensure output directory exists before running
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_analysis(input_csv_path, output_file_path)