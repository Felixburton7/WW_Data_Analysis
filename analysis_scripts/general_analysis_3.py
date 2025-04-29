# --- Complete Corrected Script ---

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
from tqdm.auto import tqdm # Use tqdm.auto for better environment detection (notebook vs script)
from typing import Dict, Any, Optional, Union, List, Tuple

# --- Configuration ---
# Define models and their corresponding prediction/uncertainty columns
MODEL_CONFIG = {
    "DeepFlex": {
        "key": "DeepFlex",
        "pred_col": "DeepFlex_rmsf",
        "unc_col": "DeepFlex_rmsf_uncertainty"
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
    },
       # --- ADD THIS NEW ENTRY ---
    "DeepFlex (Single Temp Models)": {   # Name for the report sections
        "key": "DeepFlex_Temp",          # Unique key for internal use
        "pred_col": "temperature_DeepFlex_RMSF", # <-- Name of the column you just added
        "unc_col": None                  # Set to None as there's no corresponding uncertainty column
    }
    # --- END OF NEW ENTRY ---
}
# Key baselines for significance testing against DeepFlex
KEY_BASELINES_FOR_SIG_TEST = ["RF (All Features)", "ESM-Only (Seq+Temp)"]
PRIMARY_MODEL_NAME = "DeepFlex" # Define primary model name centrally
TARGET_COL = 'rmsf' # Define target column centrally

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
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
        logger.error(f"Unexpected error during pearsonr calculation: {e}", exc_info=False)
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
        corr, p_value = kendalltau(x_clean, y_clean)
        return corr if not np.isnan(corr) else np.nan, p_value
    except Exception as e:
        logger.error(f"Unexpected error during kendalltau calculation: {e}", exc_info=False)
        return np.nan, np.nan

# *** CORRECTED format_table function ***
def format_table(data, headers="keys", tablefmt="pipe", floatfmt=".6f", **kwargs):
    """Formats data using tabulate, with improved error handling for empty data."""
    if isinstance(data, pd.DataFrame) and data.empty: return " (No data to display) "
    if not isinstance(data, pd.DataFrame) and not data: return " (No data to display) "
    try:
        # Explicitly handle headers for list of dicts
        local_headers = headers # Use a local variable to avoid modifying kwargs implicitly
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # If headers="keys" or headers is not provided explicitly in kwargs, extract from first dict
            if local_headers == "keys" or ('headers' not in kwargs and headers == "keys"): # Check both conditions
                 local_headers = list(data[0].keys()) # Extract keys from first dict as headers

        # Pass explicit headers list or let tabulate handle DataFrame headers
        return tabulate(data, headers=local_headers, tablefmt=tablefmt, floatfmt=floatfmt, **kwargs)
    except Exception as e:
        # Log the specific error and data type that failed
        logger.error(f"Tabulate formatting failed: {e} (Data type: {type(data)}, Headers arg: {headers})")
        if isinstance(data, pd.DataFrame):
             return f"(Error formatting DataFrame: {e})\nColumns: {data.columns.tolist()}\nFirst few rows:\n{data.head().to_string()}"
        elif isinstance(data, list):
             return f"(Error formatting list: {e})\nFirst element type: {type(data[0]) if data else 'N/A'}\nData snippet: {str(data)[:200]}..."
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


# --- Analysis Functions ---

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
    # Pass headers explicitly because model_key_data is list of dicts
    analysis_results['1.1 MODEL KEY'] = format_table(model_key_data, headers=list(model_key_data[0].keys()) if model_key_data else "keys", tablefmt="pipe")
    analysis_results['1.2 PRIMARY MODEL'] = f"Primary Model for focused analysis: {PRIMARY_MODEL_NAME}"

def run_missing_values(df, analysis_results):
    logger.info("Running missing value analysis...")
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame({'count': missing.astype(int), 'percentage': (missing / len(df)) * 100})
            primary_model_config = MODEL_CONFIG.get(PRIMARY_MODEL_NAME, {})
            primary_pred = primary_model_config.get('pred_col')
            primary_unc = primary_model_config.get('unc_col')

            if primary_pred and primary_pred not in missing.index and primary_pred in df.columns:
                 missing_val = df[primary_pred].isnull().sum()
                 if missing_val > 0:
                      missing_df.loc[primary_pred] = {'count': missing_val, 'percentage': (missing_val / len(df)) * 100}
                      # Update base series only if needed for downstream logic (currently not)
                      # missing = missing.append(pd.Series([missing_val], index=[primary_pred]))

            if primary_unc and primary_unc not in missing.index and primary_unc in df.columns:
                 missing_val = df[primary_unc].isnull().sum()
                 if missing_val > 0:
                      missing_df.loc[primary_unc] = {'count': missing_val, 'percentage': (missing_val / len(df)) * 100}
                      # missing = missing.append(pd.Series([missing_val], index=[primary_unc]))

            analysis_results['2. MISSING VALUE SUMMARY'] = format_table(missing_df.sort_values('count', ascending=False), floatfmt=".2f")
        else:
            analysis_results['2. MISSING VALUE SUMMARY'] = "No missing values found."
    except Exception as e:
         logger.error(f"Error during missing value analysis: {e}")
         analysis_results['2. MISSING VALUE SUMMARY'] = f"Error: {e}"

def run_descriptive_stats(df, analysis_results):
    logger.info("Running descriptive statistics...")
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns
        cols_to_describe = [TARGET_COL] + \
                           [cfg['pred_col'] for cfg in MODEL_CONFIG.values() if cfg['pred_col'] in df.columns] + \
                           [cfg['unc_col'] for cfg in MODEL_CONFIG.values() if cfg.get('unc_col') and cfg['unc_col'] in df.columns] + \
                           ['temperature', 'normalized_resid', 'relative_accessibility', 'protein_size', 'bfactor_norm', 'phi', 'psi', 'contact_number', 'coevolution_score'] # Added coevolution score

        cols_to_describe = [col for col in cols_to_describe if col in df.columns and col in numeric_cols] # Ensure numeric and exists
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
    }
    ss_label_map = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
    core_label_map = {0: 'core', 1: 'exterior'}

    for col, section_title in distribution_cols.items():
        if col in df.columns:
            try:
                counts = df[col].value_counts()
                percent = (counts / len(df)) * 100
                dist_df = pd.DataFrame({'Percent': percent, 'Count': counts.astype(int)})

                if col == 'secondary_structure_encoded': dist_df.index = dist_df.index.map(ss_label_map).fillna('Unknown')
                elif col == 'core_exterior_encoded': dist_df.index = dist_df.index.map(core_label_map).fillna('Unknown')
                elif col == 'temperature': dist_df = dist_df.sort_index()

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
    metrics_by_temp_dict = defaultdict(lambda: defaultdict(list))
    rank_metrics_overall = []

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

    df_errors = df[[target_col]].copy()

    for report_name, config in MODEL_CONFIG.items():
        pred_col = config['pred_col']
        if pred_col in df.columns:
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

                    rho, rho_p = safe_spearmanr(y_true, y_pred)
                    tau, tau_p = safe_kendalltau(y_true, y_pred)
                    rank_metrics_overall.append({'Model': report_name, 'spearman_rho': rho, 'kendall_tau': tau})

                    df_errors[f"{report_name}_abs_error"] = (pd.Series(y_pred, index=df_valid_overall.index) - pd.Series(y_true, index=df_valid_overall.index)).abs()
                except Exception as e:
                    logger.warning(f"Could not calculate overall metrics for {report_name}: {e}")
            else: logger.warning(f"Not enough valid data ({len(df_valid_overall)}) for overall metrics for {report_name}")

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
                             rho_t, _ = safe_spearmanr(y_true_t, y_pred_t)

                             metrics_by_temp_dict[temp][f'{report_name}_mae'].append(mae_t)
                             metrics_by_temp_dict[temp][f'{report_name}_pcc'].append(pcc_t)
                             metrics_by_temp_dict[temp][f'{report_name}_r2'].append(r2_t)
                             metrics_by_temp_dict[temp][f'{report_name}_spearman_rho'].append(rho_t)
                         except Exception as e:
                              logger.warning(f"Could not calculate metrics for {report_name} at T={temp}: {e}")
                              for suffix in ['mae', 'pcc', 'r2', 'spearman_rho']: metrics_by_temp_dict[temp][f'{report_name}_{suffix}'].append(np.nan)
                    else:
                        for suffix in ['mae', 'pcc', 'r2', 'spearman_rho']: metrics_by_temp_dict[temp][f'{report_name}_{suffix}'].append(np.nan)
        else:
            logger.warning(f"Prediction column '{pred_col}' for model '{report_name}' not found. Skipping comparison.")

    if metrics_overall:
        overall_df = pd.DataFrame(metrics_overall).set_index('Model')
        analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = format_table(overall_df.sort_values('pcc', ascending=False))
    else: analysis_results['5.1 OVERALL PERFORMANCE METRICS'] = "No models found or metrics calculable."

    if rank_metrics_overall:
        rank_df = pd.DataFrame(rank_metrics_overall).set_index('Model')
        analysis_results['5.2 OVERALL RANK METRICS'] = "(Higher values indicate better preservation of relative flexibility order)\n" + format_table(rank_df.sort_values('spearman_rho', ascending=False))
    else: analysis_results['5.2 OVERALL RANK METRICS'] = "No models found or rank metrics calculable."

    if metrics_by_temp_dict:
        temp_metrics_list = []
        all_temp_cols = set()
        for temp, metrics in sorted(metrics_by_temp_dict.items()):
            row_count = df['temperature'].value_counts().get(temp, 0)
            row = {'temperature': temp, 'count': row_count}
            for metric_col, metric_list in metrics.items():
                avg_metric = np.nanmean(metric_list) if metric_list else np.nan
                row[metric_col] = avg_metric
                all_temp_cols.add(metric_col)
            temp_metrics_list.append(row)

        temp_df = pd.DataFrame(temp_metrics_list).set_index('temperature')
        for col in all_temp_cols:
             if col not in temp_df.columns: temp_df[col] = np.nan

        cols_ordered = ['count']
        metric_order = ['mae', 'pcc', 'spearman_rho', 'r2']
        for model_name in model_report_names_found:
             for metric_suffix in metric_order:
                  col_name = f'{model_name}_{metric_suffix}'
                  if col_name in temp_df.columns: cols_ordered.append(col_name)
        cols_ordered.extend([col for col in temp_df.columns if col not in cols_ordered])

        analysis_results['5.3 PERFORMANCE METRICS BY TEMPERATURE'] = format_table(temp_df[cols_ordered])

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

                 if not mae_cols or not pcc_cols: continue

                 best_mae_model_col = row_data[mae_cols].idxmin()
                 best_pcc_model_col = row_data[pcc_cols].idxmax()
                 best_rho_model_col = row_data[rho_cols].idxmax() if rho_cols else "N/A"
                 best_mae_model = best_mae_model_col.replace('_mae', '')
                 best_pcc_model = best_pcc_model_col.replace('_pcc', '')
                 best_rho_model = best_rho_model_col.replace('_spearman_rho', '') if rho_cols else "N/A"

                 summary_str = f"  T={temp:.1f}K (N={int(row_data.get('count',0))}):"
                 summary_str += f" Best MAE={best_mae_model}({row_data[best_mae_model_col]:.3f}),"
                 summary_str += f" Best PCC={best_pcc_model}({row_data[best_pcc_model_col]:.3f}),"
                 if rho_cols: summary_str += f" Best Spearman Rho={best_rho_model}({row_data[best_rho_model_col]:.3f})"

                 if PRIMARY_MODEL_NAME != best_mae_model or PRIMARY_MODEL_NAME != best_pcc_model:
                     summary_str += f" | {PRIMARY_MODEL_NAME}:"
                     if primary_mae_prefix in row_data and pd.notna(row_data[primary_mae_prefix]): summary_str += f" MAE={row_data[primary_mae_prefix]:.3f}"
                     if primary_pcc_prefix in row_data and pd.notna(row_data[primary_pcc_prefix]): summary_str += f", PCC={row_data[primary_pcc_prefix]:.3f}"
                     if primary_rho_prefix in row_data and pd.notna(row_data[primary_rho_prefix]): summary_str += f", Rho={row_data[primary_rho_prefix]:.3f}"

                 temp_summary_lines.append(summary_str)

             analysis_results['5.3.1 TEMP PERFORMANCE SUMMARY'] = "\n".join(temp_summary_lines)
        except Exception as e:
             logger.error(f"Error generating temperature performance summary: {e}")
             analysis_results['5.3.1 TEMP PERFORMANCE SUMMARY'] = "Error generating summary."
    else: analysis_results['5.3 PERFORMANCE METRICS BY TEMPERATURE'] = "No temperature data or metrics calculable."

    pred_cols_present = [config['pred_col'] for config in MODEL_CONFIG.values() if config['pred_col'] in df.columns]
    if len(pred_cols_present) > 1:
        try:
            pred_corr = df[[target_col] + pred_cols_present].dropna().corr(method='pearson')**2
            rename_map_pred = {target_col: 'ActualRMSF'}
            for report_name, config in MODEL_CONFIG.items():
                 if config['pred_col'] in pred_corr.columns: rename_map_pred[config['pred_col']] = report_name
            pred_corr = pred_corr.rename(columns=rename_map_pred, index=rename_map_pred)
            analysis_results['5.4 PREDICTION R-SQUARED MATRIX (COEFFICIENT OF DETERMINATION, INCL. ACTUAL)'] = format_table(pred_corr)
        except Exception as e:
             logger.error(f"Error calculating prediction correlation: {e}")
             analysis_results['5.4 PREDICTION R-SQUARED MATRIX'] = f"Error: {e}"
    else: analysis_results['5.4 PREDICTION R-SQUARED MATRIX'] = "Not enough models with predictions to correlate."

    error_cols_present = [col for col in df_errors.columns if col.endswith('_abs_error')]
    if len(error_cols_present) > 1:
        try:
            error_corr = df_errors[[target_col] + error_cols_present].dropna().corr(method='pearson')**2
            rename_map_err = {target_col: 'ActualRMSF'}
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

def get_ramachandran_region(phi, psi):
    """Assigns residue to Ramachandran region based on phi/psi angles."""
    if pd.isna(phi) or pd.isna(psi): return "Undefined"
    if phi > 180: phi -= 360
    if psi > 180: psi -= 360
    if (-180 <= phi < -30) and (-70 <= psi < 50): return "Alpha-Helix"
    if (-180 <= phi < -40) and (90 <= psi <= 180 or -180 <= psi < -150): return "Beta-Sheet"
    if (30 < phi < 100) and (-20 < psi < 90): return "L-Alpha"
    if (0 <= phi < 180 and -90 <= psi < 0): return "Disallowed"
    return "Other Allowed/Loop"

def run_dihedral_analysis(df, analysis_results):
    logger.info("Running Dihedral Angle (Ramachandran) analysis...")
    analysis_results['5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS'] = ""

    phi_col, psi_col = 'phi', 'psi'
    if phi_col not in df.columns or psi_col not in df.columns:
        logger.warning(f"Missing '{phi_col}' or '{psi_col}' columns. Skipping Dihedral Analysis.")
        analysis_results['5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS'] = f"Skipped: Missing '{phi_col}' or '{psi_col}' columns."
        return

    try:
        df_copy = df.copy()
        df_copy['rama_region'] = df_copy.apply(lambda row: get_ramachandran_region(row[phi_col], row[psi_col]), axis=1)

        rama_counts = df_copy['rama_region'].value_counts()
        rama_percent = (rama_counts / len(df_copy[df_copy['rama_region'] != "Undefined"])) * 100
        rama_dist_df = pd.DataFrame({'Percent': rama_percent, 'Count': rama_counts.astype(int)})
        analysis_results['5.6.1 RAMACHANDRAN REGION DISTRIBUTION'] = "(Based on defined phi/psi pairs)\n" + format_table(rama_dist_df.sort_values('Count', ascending=False), floatfmt=".2f")

        logger.info("Stratifying performance by Ramachandran region...")
        strata_results = []
        grouped = df_copy.groupby('rama_region', observed=True)

        for group_name, group_df in grouped:
             if group_name == "Undefined" or len(group_df) < 5: continue

             row = {'rama_region': group_name, 'count': len(group_df)}
             for report_name, config in MODEL_CONFIG.items():
                  pred_col = config['pred_col']
                  mae_val, pcc_val = np.nan, np.nan
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
            primary_mae_col = f'{PRIMARY_MODEL_NAME}_mae'
            sort_col = primary_mae_col if primary_mae_col in strata_df.columns else 'count'
            strata_df = strata_df.sort_values(sort_col, ascending=(sort_col == primary_mae_col))
            analysis_results['5.6.2 PERFORMANCE BY RAMACHANDRAN REGION'] = format_table(strata_df)
        else:
            analysis_results['5.6.2 PERFORMANCE BY RAMACHANDRAN REGION'] = "No results generated after grouping by region."

    except Exception as e:
        logger.error(f"Error during Dihedral Analysis: {e}", exc_info=True)
        analysis_results['5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS'] += f"\nError during analysis: {e}"

def run_uncertainty_analysis(df, analysis_results):
    logger.info("Running uncertainty analysis...")
    analysis_results['6. UNCERTAINTY ANALYSIS'] = "Comparing uncertainty estimates for models where available."
    uncertainty_stats, uncertainty_error_corr, calibration_check, ece_results = [], [], [], []
    mae_vs_unc_bins, reliability_data = {}, {}
    target_col = TARGET_COL
    df_copy = df.copy()

    for report_name, config in MODEL_CONFIG.items():
        pred_col = config['pred_col']
        unc_col = config.get('unc_col')
        error_col = f"{report_name}_abs_error"

        if not unc_col or unc_col not in df_copy.columns: continue
        if error_col not in df_copy.columns:
            if target_col in df_copy.columns and pred_col in df_copy.columns:
                df_copy[error_col] = (df_copy[pred_col] - df_copy[target_col]).abs()
            else: continue

        df_valid = df_copy[[error_col, unc_col]].dropna()
        if len(df_valid) < 10: continue

        unc_values = df_valid[unc_col].values
        abs_error = df_valid[error_col].values

        stats = df_valid[unc_col].describe()
        # Convert stats to native python types to avoid potential serialization issues later
        safe_stats = {k: float(v) if isinstance(v, (np.number, int, float)) else str(v) for k, v in stats.to_dict().items()}
        uncertainty_stats.append({'Model': report_name, **safe_stats})


        corr, _ = safe_pearsonr(unc_values, abs_error)
        uncertainty_error_corr.append({'Model': report_name, 'Uncertainty-Error PCC': corr})

        try: within_1_std = np.mean(abs_error <= unc_values) * 100
        except TypeError: within_1_std = np.nan
        calibration_check.append({'Model': report_name, '% within 1 Uncertainty': within_1_std})

        try:
            n_bins_mae = 10
            df_valid['unc_quantile'], bin_edges_mae = pd.qcut(unc_values, q=n_bins_mae, labels=False, duplicates='drop', retbins=True)
            if df_valid['unc_quantile'].nunique() >= 2:
                bin_range_labels_mae = {i: f"{bin_edges_mae[i]:.3g}-{bin_edges_mae[i+1]:.3g}" for i in range(len(bin_edges_mae)-1)}
                binned_mae = df_valid.groupby('unc_quantile')[error_col].agg(['mean', 'median', 'std', 'count']).reset_index()
                binned_mae['unc_bin_range'] = binned_mae['unc_quantile'].map(bin_range_labels_mae)
                binned_mae = binned_mae.rename(columns={'mean': 'MeanAbsErr', 'median': 'MedianAbsErr', 'std': 'StdAbsErr', 'unc_bin_range': 'UncertaintyQuantile'})
                mae_vs_unc_bins[report_name] = format_table(binned_mae[['UncertaintyQuantile', 'MeanAbsErr', 'MedianAbsErr', 'StdAbsErr', 'count']])
        except Exception as e: logger.warning(f"Could not calculate MAE vs uncertainty bins for {report_name}: {e}")

        try:
            n_bins_rel = 10
            min_unc, max_unc = np.min(unc_values), np.max(unc_values)
            if max_unc <= min_unc: raise ValueError("Uncertainty range is zero or negative.")
            bin_limits = np.unique(np.linspace(min_unc, max_unc, n_bins_rel + 1))
            if len(bin_limits) < 2: raise ValueError("Could not create unique bin edges.")
            actual_n_bins_rel = len(bin_limits) - 1
            df_valid['rel_bin'] = pd.cut(unc_values, bins=bin_limits, labels=False, include_lowest=True, right=True)

            binned_data = []
            total_count, ece = len(df_valid), 0.0
            if total_count > 0:
                 for i in range(actual_n_bins_rel):
                      bin_mask = (df_valid['rel_bin'] == i); bin_df = df_valid[bin_mask]; bin_count = len(bin_df)
                      if bin_count > 0:
                          mean_pred_unc = bin_df[unc_col].mean()
                          observed_mae_bin = bin_df[error_col].mean()
                          observed_rmse_bin = np.sqrt(mean_squared_error(np.zeros_like(bin_df[error_col]), bin_df[error_col]))
                          binned_data.append({
                              'Bin Index': i, 'Uncertainty Range': f"{bin_limits[i]:.3g}-{bin_limits[i+1]:.3g}",
                              'Mean Predicted Uncertainty': mean_pred_unc, 'Observed MAE in Bin': observed_mae_bin,
                              'Observed RMSE in Bin': observed_rmse_bin, 'Count': bin_count})
                          ece += (np.abs(mean_pred_unc - observed_mae_bin) * bin_count)

                 ece_results.append({'Model': report_name, 'Expected Calibration Error (ECE-MAE)': ece / total_count})
                 reliability_data[report_name] = format_table(pd.DataFrame(binned_data).set_index('Bin Index'))

        except Exception as e: logger.warning(f"Could not calculate Reliability/ECE data for {report_name}: {e}", exc_info=False)

    if uncertainty_stats: analysis_results['6.1 UNCERTAINTY DISTRIBUTION STATISTICS'] = format_table(pd.DataFrame(uncertainty_stats).set_index('Model'))
    else: analysis_results['6.1 UNCERTAINTY DISTRIBUTION STATISTICS'] = "No models with uncertainty data found."
    if uncertainty_error_corr: analysis_results['6.2 UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION'] = "(Positive correlation indicates uncertainty tracks error magnitude.)\n" + format_table(pd.DataFrame(uncertainty_error_corr).set_index('Model'))
    else: analysis_results['6.2 UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION'] = "No models with uncertainty data found or correlation calculable."
    if calibration_check: analysis_results['6.3 SIMPLE CALIBRATION CHECK'] = "(% of errors <= predicted uncertainty. Expected ~68.2% for well-calibrated Gaussian uncertainty.)\n" + format_table(pd.DataFrame(calibration_check).set_index('Model'), floatfmt=".2f")
    else: analysis_results['6.3 SIMPLE CALIBRATION CHECK'] = "No models with uncertainty data found or calibration calculable."
    for model_name, table_str in mae_vs_unc_bins.items(): analysis_results[f'6.4 MEAN ABSOLUTE ERROR BINNED BY {model_name.upper()} UNCERTAINTY QUANTILE'] = table_str
    if not mae_vs_unc_bins: analysis_results['6.4 MEAN ABSOLUTE ERROR BINNED BY UNCERTAINTY QUANTILE'] = "No results generated (check data/binning)."
    for model_name, table_str in reliability_data.items(): analysis_results[f'6.5 RELIABILITY DIAGRAM DATA ({model_name.upper()})'] = "(Binning by predicted uncertainty. Plot 'Mean Predicted Uncertainty' vs 'Observed MAE/RMSE in Bin' for reliability diagram.)\n" + table_str
    if not reliability_data: analysis_results['6.5 RELIABILITY DIAGRAM DATA'] = "No results generated (check data/binning)."
    if ece_results: analysis_results['6.6 EXPECTED CALIBRATION ERROR (ECE)'] = "(Lower ECE indicates better calibration. Calculated using MAE as observed error measure.)\n" + format_table(pd.DataFrame(ece_results).set_index('Model'), floatfmt=".4f")
    else: analysis_results['6.6 EXPECTED CALIBRATION ERROR (ECE)'] = "ECE not calculated."

def calculate_per_domain_metrics(df, target_col='rmsf'):
    """Calculates MAE and PCC for each model within each domain. Handles missing data."""
    logger.info("Calculating per-domain performance metrics...")
    domain_metrics = {}
    if 'domain_id' not in df.columns or target_col not in df.columns:
        logger.error(f"Input DataFrame missing 'domain_id' or '{target_col}'. Skipping per-domain metrics.")
        return {}
    df_filtered = df.dropna(subset=[target_col, 'domain_id'])
    if df_filtered.empty: return {}

    grouped = df_filtered.groupby('domain_id')
    for domain_id, group in tqdm(grouped, desc="Domain Metrics", total=len(grouped)):
        if group.empty: continue
        domain_results = {}
        actual = group[target_col].values
        for report_name, config in MODEL_CONFIG.items():
            pred_col = config['pred_col']
            mae_val, pcc_val = np.nan, np.nan
            if pred_col in group.columns:
                predicted = group[pred_col].values
                valid_mask = ~np.isnan(predicted)
                actual_valid, predicted_valid = actual[valid_mask], predicted[valid_mask]
                if len(actual_valid) > 1:
                    try: mae_val = mean_absolute_error(actual_valid, predicted_valid)
                    except: pass
                    try: pcc_val, _ = safe_pearsonr(actual_valid, predicted_valid)
                    except: pass
            domain_results[report_name] = {'mae': mae_val, 'pcc': pcc_val}
        if domain_results: domain_metrics[domain_id] = domain_results
    return domain_metrics

def run_domain_level_analysis(df, analysis_results):
    """Runs domain-level analysis and saves detailed metrics per domain to a CSV file."""
    logger.info("Running domain-level analysis...")
    analysis_results['7. DOMAIN-LEVEL PERFORMANCE METRICS'] = ""
    domain_metrics_dict = calculate_per_domain_metrics(df, target_col=TARGET_COL)
    if not domain_metrics_dict:
        logger.error("Failed to calculate per-domain metrics. Skipping domain-level analysis.")
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: No per-domain metrics calculated."
        if '_internal_domain_metrics_path' in analysis_results: del analysis_results['_internal_domain_metrics_path']
        return

    domain_data_list = []
    all_metric_keys = set(['domain_id', 'count_residues', 'actual_mean', 'actual_stddev'])
    grouped_by_domain = df.groupby('domain_id') if 'domain_id' in df.columns else None

    for domain_id, model_metrics in tqdm(domain_metrics_dict.items(), desc="Aggregating Domain Stats", total=len(domain_metrics_dict)):
        row = {'domain_id': domain_id}
        # Use .get() with default empty DataFrame to avoid errors if domain_id missing
        domain_group = grouped_by_domain.get_group(domain_id).copy() if grouped_by_domain is not None and domain_id in grouped_by_domain.groups else pd.DataFrame()

        for model_name, metrics in model_metrics.items():
            row[f"{model_name}_mae"] = metrics.get('mae', np.nan); all_metric_keys.add(f"{model_name}_mae")
            row[f"{model_name}_pcc"] = metrics.get('pcc', np.nan); all_metric_keys.add(f"{model_name}_pcc")

        row['count_residues'] = len(domain_group) # This will be 0 if domain_group is empty
        actual_rmsf_domain = domain_group[TARGET_COL].dropna() if not domain_group.empty and TARGET_COL in domain_group else pd.Series(dtype=float)
        row['actual_mean'] = actual_rmsf_domain.mean() if not actual_rmsf_domain.empty else np.nan
        row['actual_stddev'] = actual_rmsf_domain.std() if len(actual_rmsf_domain) > 1 else 0.0

        if not domain_group.empty:
            for report_name, config in MODEL_CONFIG.items():
                 pred_col = config['pred_col']
                 if pred_col in domain_group.columns:
                      aligned_df = domain_group[[TARGET_COL, pred_col]].dropna()
                      if len(aligned_df) > 1:
                           y_true_dom, y_pred_dom = aligned_df[TARGET_COL].values, aligned_df[pred_col].values
                           try: row[f"{report_name}_rmse"] = np.sqrt(mean_squared_error(y_true_dom, y_pred_dom))
                           except: row[f"{report_name}_rmse"] = np.nan
                           try: row[f"{report_name}_r2"] = r2_score(y_true_dom, y_pred_dom) if np.var(y_true_dom) > 1e-9 else np.nan
                           except: row[f"{report_name}_r2"] = np.nan
                           try: row[f"{report_name}_medae"] = median_absolute_error(y_true_dom, y_pred_dom)
                           except: row[f"{report_name}_medae"] = np.nan
                           row[f"{report_name}_pred_stddev"] = y_pred_dom.std() if len(y_pred_dom) > 1 else 0.0
                      else:
                           for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: row[f"{report_name}_{metric}"] = np.nan
                 else:
                      for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: row[f"{report_name}_{metric}"] = np.nan
                 for metric in ['rmse', 'r2', 'medae', 'pred_stddev']: all_metric_keys.add(f"{report_name}_{metric}")

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
                          else: row[f"{report_name}_uncertainty_error_corr"], row[f"{report_name}_within_1std"] = np.nan, np.nan
                     else: row[f"{report_name}_avg_uncertainty"], row[f"{report_name}_uncertainty_error_corr"], row[f"{report_name}_within_1std"] = np.nan, np.nan, np.nan
                 else: row[f"{report_name}_avg_uncertainty"], row[f"{report_name}_uncertainty_error_corr"], row[f"{report_name}_within_1std"] = np.nan, np.nan, np.nan
                 for unc_metric in ['avg_uncertainty', 'uncertainty_error_corr', 'within_1std']: all_metric_keys.add(f"{report_name}_{unc_metric}")

                 if 'temperature' in domain_group.columns:
                      for temp, temp_group in domain_group.groupby('temperature'):
                           temp_key_suffix = f"{temp:.1f}K"
                           mae_t_val, pcc_t_val, rho_t_val = np.nan, np.nan, np.nan
                           if pred_col in temp_group.columns:
                                aligned_temp_df = temp_group[[TARGET_COL, pred_col]].dropna()
                                if len(aligned_temp_df) > 1:
                                     y_true_t, y_pred_t = aligned_temp_df[TARGET_COL].values, aligned_temp_df[pred_col].values
                                     try: mae_t_val = mean_absolute_error(y_true_t, y_pred_t)
                                     except: pass
                                     try: pcc_t_val, _ = safe_pearsonr(y_true_t, y_pred_t)
                                     except: pass
                                     try: rho_t_val, _ = safe_spearmanr(y_true_t, y_pred_t)
                                     except: pass
                           row[f"{report_name}_mae_{temp_key_suffix}"] = mae_t_val; all_metric_keys.add(f"{report_name}_mae_{temp_key_suffix}")
                           row[f"{report_name}_pcc_{temp_key_suffix}"] = pcc_t_val; all_metric_keys.add(f"{report_name}_pcc_{temp_key_suffix}")
                           row[f"{report_name}_spearman_rho_{temp_key_suffix}"] = rho_t_val; all_metric_keys.add(f"{report_name}_spearman_rho_{temp_key_suffix}")
        domain_data_list.append(row)

    if not domain_data_list:
        logger.error("No data compiled for domain-level DataFrame."); analysis_results['7.1 MEAN ACROSS DOMAINS'] = "Failed: No domain data compiled."
        if '_internal_domain_metrics_path' in analysis_results: del analysis_results['_internal_domain_metrics_path']
        return

    try:
        domain_df = pd.DataFrame(domain_data_list).set_index('domain_id')
        logger.debug(f"All metric keys tracked for domain mean calc: {sorted(list(all_metric_keys))}")
        for key in all_metric_keys:
            if key not in domain_df.columns and key != 'domain_id': domain_df[key] = np.nan

        numeric_domain_cols = domain_df.select_dtypes(include=np.number).columns
        if not numeric_domain_cols.empty:
             mean_domain_metrics = domain_df[numeric_domain_cols].mean().to_frame(name='Mean Value Across Domains').sort_index()
             analysis_results['7.1 MEAN ACROSS DOMAINS'] = format_table(mean_domain_metrics)
        else: analysis_results['7.1 MEAN ACROSS DOMAINS'] = "No numeric domain metrics to average."

        output_dir_for_saving = os.path.dirname(analysis_results['_internal_output_file_path'])
        domain_analysis_dir = os.path.join(output_dir_for_saving, "domain_analysis_data")
        os.makedirs(domain_analysis_dir, exist_ok=True)
        domain_metrics_path = os.path.join(domain_analysis_dir, "detailed_domain_metrics.csv")
        domain_df.to_csv(domain_metrics_path)
        logger.info(f"Saved detailed domain metrics CSV to: {domain_metrics_path}")
        analysis_results['_internal_domain_metrics_path'] = domain_metrics_path

    except Exception as e:
        logger.error(f"Error calculating, formatting, or saving domain metrics: {e}", exc_info=True)
        analysis_results['7.1 MEAN ACROSS DOMAINS'] = f"Error: {e}"
        if '_internal_domain_metrics_path' in analysis_results: del analysis_results['_internal_domain_metrics_path']

def calculate_significance_tests(per_domain_metrics, primary_model_report_name, baseline_report_names):
    """Performs Wilcoxon tests comparing primary model to baselines on per-domain metrics."""
    logger.info(f"Performing significance tests comparing {primary_model_report_name} against {baseline_report_names}...")
    results = {}
    if not per_domain_metrics: return results
    first_domain_key = next(iter(per_domain_metrics), None)
    if not first_domain_key or primary_model_report_name not in per_domain_metrics.get(first_domain_key, {}): return results

    def get_paired_values(metric_key, baseline_name):
        primary_values, baseline_values = [], []
        for domain_id, metrics in per_domain_metrics.items():
             primary_val = metrics.get(primary_model_report_name, {}).get(metric_key, np.nan)
             baseline_val = metrics.get(baseline_name, {}).get(metric_key, np.nan)
             if not np.isnan(primary_val) and not np.isnan(baseline_val):
                 primary_values.append(primary_val); baseline_values.append(baseline_val)
        return primary_values, baseline_values

    for baseline_name in baseline_report_names:
        if baseline_name not in per_domain_metrics.get(first_domain_key, {}):
            results[f"{baseline_name}_vs_MAE"] = {'p_value': np.nan, 'statistic': np.nan, 'N': 0}
            results[f"{baseline_name}_vs_PCC"] = {'p_value': np.nan, 'statistic': np.nan, 'N': 0}
            continue

        p_mae, s_mae, n_mae = np.nan, np.nan, 0
        p_pcc, s_pcc, n_pcc = np.nan, np.nan, 0
        primary_mae, baseline_mae = get_paired_values('mae', baseline_name)
        primary_pcc, baseline_pcc = get_paired_values('pcc', baseline_name)
        n_mae, n_pcc = len(primary_mae), len(primary_pcc)

        if n_mae >= 10:
            try: s_mae, p_mae = wilcoxon(primary_mae, baseline_mae, alternative='less', zero_method='zsplit')
            except ValueError as e: logger.warning(f"Wilcoxon MAE test failed vs {baseline_name}: {e}")
        results[f"{baseline_name}_vs_MAE"] = {'p_value': p_mae, 'statistic': s_mae, 'N': n_mae}

        if n_pcc >= 10:
            try: s_pcc, p_pcc = wilcoxon(primary_pcc, baseline_pcc, alternative='greater', zero_method='zsplit')
            except ValueError as e: logger.warning(f"Wilcoxon PCC test failed vs {baseline_name}: {e}")
        results[f"{baseline_name}_vs_PCC"] = {'p_value': p_pcc, 'statistic': s_pcc, 'N': n_pcc}
    return results

def run_stratified_performance(df, analysis_results, strat_col, section_title, n_bins=None, label_map=None, bin_col_suffix='_bin', sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae'):
    """Calculates performance metrics stratified by a given column (categorical or binned continuous)."""
    logger.info(f"Running stratified performance analysis by '{strat_col}'...")
    if strat_col not in df.columns or TARGET_COL not in df.columns:
        analysis_results[section_title] = f"Skipped: Missing column '{strat_col}' or '{TARGET_COL}'."
        return

    strata_results = []
    df_copy = df[[TARGET_COL, strat_col] + [cfg['pred_col'] for cfg in MODEL_CONFIG.values() if cfg['pred_col'] in df.columns]].copy()
    grouping_col, index_name = strat_col, strat_col

    if n_bins is not None and n_bins > 0 and pd.api.types.is_numeric_dtype(df_copy[strat_col]):
        bin_col_name = f"{strat_col}{bin_col_suffix}"
        grouping_col = bin_col_name
        index_name = f"{strat_col.replace('_', ' ').title()} Bin"
        try:
            df_copy.dropna(subset=[strat_col], inplace=True)
            if df_copy.empty: raise ValueError("DataFrame empty after dropping NaNs in strat_col.")
            df_copy[bin_col_name], bin_edges = pd.qcut(df_copy[strat_col], q=n_bins, labels=False, duplicates='drop', retbins=True)
            if df_copy[bin_col_name].nunique() < 2: raise ValueError(f"Could not create >= 2 bins.")
            final_bin_labels = {i: f"{bin_edges[i]:.3g}-{bin_edges[i+1]:.3g}" for i in range(len(bin_edges)-1)}
            df_copy[bin_col_name] = df_copy[bin_col_name].map(final_bin_labels).fillna("Binning Error")
        except Exception as e:
            logger.error(f"Failed to bin column '{strat_col}': {e}. Skipping stratification.")
            analysis_results[section_title] = f"Skipped: Error binning column '{strat_col}' ({e})."
            return
    elif label_map:
         df_copy[grouping_col] = df_copy[grouping_col].map(label_map).fillna('Unknown')
         index_name = strat_col.replace('_encoded', '').replace('_', ' ').title()

    df_copy.dropna(subset=[grouping_col, TARGET_COL], inplace=True)
    if df_copy.empty:
        analysis_results[section_title] = f"Skipped: No valid data after creating groups/bins for '{strat_col}'."
        return

    grouped = df_copy.groupby(grouping_col, observed=True)
    for group_name, group_df in grouped:
         if len(group_df) < 5: continue
         row = {grouping_col: group_name, 'count': len(group_df), 'mean_actual_rmsf': group_df[TARGET_COL].mean()}
         for report_name, config in MODEL_CONFIG.items():
              pred_col = config['pred_col']
              mae_val, pcc_val, rho_val = np.nan, np.nan, np.nan
              if pred_col in group_df.columns:
                   df_valid = group_df[[TARGET_COL, pred_col]].dropna()
                   if len(df_valid) > 1:
                        y_true, y_pred = df_valid[TARGET_COL].values, df_valid[pred_col].values
                        try: mae_val = mean_absolute_error(y_true, y_pred)
                        except: pass
                        try: pcc_val, _ = safe_pearsonr(y_true, y_pred)
                        except: pass
                        try: rho_val, _ = safe_spearmanr(y_true, y_pred)
                        except: pass
              row[f'{report_name}_mae'] = mae_val; row[f'{report_name}_pcc'] = pcc_val; row[f'{report_name}_spearman_rho'] = rho_val
         strata_results.append(row)

    if strata_results:
         strata_df = pd.DataFrame(strata_results).set_index(grouping_col)
         strata_df.index.name = index_name
         try:
             sort_key = sort_by_metric if (sort_by_metric and sort_by_metric in strata_df.columns and strata_df[sort_by_metric].notna().any()) else 'count'
             ascending_sort = 'mae' in sort_key.lower() or 'error' in sort_key.lower() if sort_key != 'count' else False
             strata_df = strata_df.sort_values(sort_key, ascending=ascending_sort)
         except Exception as sort_e:
              logger.warning(f"Could not sort stratified results by '{sort_by_metric}', falling back to count: {sort_e}")
              strata_df = strata_df.sort_values('count', ascending=False)
         analysis_results[section_title] = format_table(strata_df)
    else: analysis_results[section_title] = f"No results after grouping by '{strat_col}' (groups might be too small)."

# --- Specific Stratified Analysis Callers ---
def run_amino_acid_performance(df, analysis_results):
    run_stratified_performance(df, analysis_results, 'resname', '8. PERFORMANCE BY AMINO ACID', sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae')
def run_norm_resid_performance(df, analysis_results):
    run_stratified_performance(df, analysis_results, 'normalized_resid', '9. PERFORMANCE BY NORMALIZED RESIDUE POSITION', n_bins=5, sort_by_metric=None)
def run_core_exterior_performance(df, analysis_results):
    core_label_map = {0: 'Core', 1: 'Exterior'}
    run_stratified_performance(df, analysis_results, 'core_exterior_encoded', '10. PERFORMANCE BY CORE/EXTERIOR', label_map=core_label_map, sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae')
def run_secondary_structure_performance(df, analysis_results):
    ss_label_map = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
    run_stratified_performance(df, analysis_results, 'secondary_structure_encoded', '11. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)', label_map=ss_label_map, sort_by_metric=f'{PRIMARY_MODEL_NAME}_mae')
def run_rel_accessibility_performance(df, analysis_results):
    run_stratified_performance(df, analysis_results, 'relative_accessibility', '13. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE', n_bins=5, sort_by_metric=None)
def run_bfactor_performance(df, analysis_results):
    bfactor_col = 'bfactor_norm' if 'bfactor_norm' in df.columns else 'bfactor' if 'bfactor' in df.columns else None
    if bfactor_col: run_stratified_performance(df, analysis_results, bfactor_col, f'14. PERFORMANCE BY {bfactor_col.upper()} QUANTILE', n_bins=5, sort_by_metric=None)
    else: analysis_results['14. PERFORMANCE BY BFACTOR QUANTILE'] = "Skipped: No 'bfactor_norm' or 'bfactor' column found."

def calculate_performance_vs_actual_rmsf(df, target_col='rmsf', model_pred_col='DeepFlex_rmsf', n_bins=10):
    logger.info(f"Analyzing performance vs actual '{target_col}' magnitude for model prediction '{model_pred_col}'...")
    results = []
    if not all(col in df.columns for col in [target_col, model_pred_col]): return pd.DataFrame()
    df_valid = df[[target_col, model_pred_col]].dropna()
    if len(df_valid) < n_bins * 2: return pd.DataFrame()
    try:
        df_valid['rmsf_quantile_bin'], bin_edges = pd.qcut(df_valid[target_col], q=n_bins, labels=False, duplicates='drop', retbins=True)
        if df_valid['rmsf_quantile_bin'].nunique() < 2: return pd.DataFrame()
        bin_range_labels = {i: f"{bin_edges[i]:.3g}-{bin_edges[i+1]:.3g}" for i in range(len(bin_edges)-1)}
        for bin_label_float, group in df_valid.groupby('rmsf_quantile_bin'):
             if len(group) < 2: continue
             actual_bin, predicted_bin = group[target_col].values, group[model_pred_col].values
             mae, pcc, rho, rmse, r2 = np.nan, np.nan, np.nan, np.nan, np.nan
             try: mae = mean_absolute_error(actual_bin, predicted_bin)
             except: pass
             try: pcc, _ = safe_pearsonr(actual_bin, predicted_bin)
             except: pass
             try: rho, _ = safe_spearmanr(actual_bin, predicted_bin)
             except: pass
             try: rmse = np.sqrt(mean_squared_error(actual_bin, predicted_bin))
             except: pass
             try: r2 = r2_score(actual_bin, predicted_bin) if np.var(actual_bin) > 1e-9 else np.nan
             except: pass
             results.append({'RMSF_Quantile_Bin': int(bin_label_float), 'RMSF_Range': bin_range_labels.get(int(bin_label_float), "N/A"),
                 'Count': len(group), 'Mean_Actual_RMSF': np.mean(actual_bin),
                 'MAE': mae, 'PCC': pcc, 'SpearmanRho': rho, 'RMSE': rmse, 'R2': r2})
    except Exception as e: logger.error(f"Error during RMSF binning: {e}", exc_info=True); return pd.DataFrame()
    return pd.DataFrame(results).sort_values('RMSF_Quantile_Bin')

def run_performance_vs_actual_rmsf(df, analysis_results, primary_model_report_name=PRIMARY_MODEL_NAME):
    logger.info("Running performance vs actual RMSF analysis...")
    section_title = f"12. {primary_model_report_name.upper()} PERFORMANCE BY ACTUAL RMSF QUANTILE"
    primary_model_config = MODEL_CONFIG.get(primary_model_report_name)
    if not primary_model_config:
        analysis_results[section_title] = "Primary model config missing."; return
    binned_df = calculate_performance_vs_actual_rmsf(df, target_col=TARGET_COL, model_pred_col=primary_model_config['pred_col'], n_bins=10)
    if not binned_df.empty:
        binned_df.rename(columns={'MAE': f'{primary_model_report_name}_MAE', 'PCC': f'{primary_model_report_name}_PCC',
                                 'SpearmanRho': f'{primary_model_report_name}_SpearmanRho', 'RMSE': f'{primary_model_report_name}_RMSE',
                                 'R2': f'{primary_model_report_name}_R2'}, inplace=True)
        analysis_results[section_title] = format_table(binned_df.set_index('RMSF_Range').drop(columns=['RMSF_Quantile_Bin']))
    else: analysis_results[section_title] = "No results generated or error during binning."

def run_model_disagreement_analysis(df, analysis_results, primary_model_report_name=PRIMARY_MODEL_NAME):
    logger.info("Running model disagreement analysis...")
    section_title = "15. MODEL DISAGREEMENT VS. ERROR"
    pred_cols = [config['pred_col'] for config in MODEL_CONFIG.values() if config['pred_col'] in df.columns]
    if len(pred_cols) < 2:
        analysis_results[section_title] = "Skipped: Need predictions from at least 2 models."; return

    df_copy = df.copy()
    df_preds_only = df_copy[pred_cols].apply(pd.to_numeric, errors='coerce')
    df_copy['prediction_stddev'] = df_preds_only.std(axis=1, skipna=True)
    df_copy['prediction_range'] = df_preds_only.max(axis=1, skipna=True) - df_preds_only.min(axis=1, skipna=True)

    try:
        stats_std = df_copy['prediction_stddev'].describe().to_frame(name='Prediction StdDev')
        stats_range = df_copy['prediction_range'].describe().to_frame(name='Prediction Range')
        analysis_results[section_title] = "MODEL PREDICTION DISAGREEMENT STATS\n" + format_table(pd.concat([stats_std, stats_range], axis=1)) + "\n\nNote: Std Dev/Range calculated across available model predictions per residue.\n"
    except Exception as e: analysis_results[section_title] = "Could not calculate disagreement stats.\n"

    primary_model_config = MODEL_CONFIG.get(primary_model_report_name)
    if primary_model_config:
         primary_pred_col = primary_model_config['pred_col']
         primary_unc_col = primary_model_config.get('unc_col')
         error_col = f"{primary_model_report_name}_abs_error"
         if error_col not in df_copy.columns and TARGET_COL in df_copy.columns and primary_pred_col in df_copy.columns:
              df_copy[error_col] = (df_copy[primary_pred_col] - df_copy[TARGET_COL]).abs()

         corr_lines = []
         if error_col in df_copy.columns and 'prediction_stddev' in df_copy.columns:
              df_valid_corr_std = df_copy[[error_col, 'prediction_stddev']].dropna()
              df_valid_corr_range = df_copy[[error_col, 'prediction_range']].dropna()
              if len(df_valid_corr_std) > 1: pcc_err_std, _ = safe_pearsonr(df_valid_corr_std['prediction_stddev'], df_valid_corr_std[error_col]); corr_lines.append(f"PCC(Prediction StdDev vs {error_col}): {pcc_err_std:.3f} (N={len(df_valid_corr_std)})")
              else: corr_lines.append(f"PCC(Prediction StdDev vs {error_col}): Not enough data.")
              if len(df_valid_corr_range) > 1: pcc_err_range, _ = safe_pearsonr(df_valid_corr_range['prediction_range'], df_valid_corr_range[error_col]); corr_lines.append(f"PCC(Prediction Range vs {error_col}): {pcc_err_range:.3f} (N={len(df_valid_corr_range)})")
              else: corr_lines.append(f"PCC(Prediction Range vs {error_col}): Not enough data.")
         else: corr_lines.append(f"Correlation with {error_col} skipped (column missing).")

         if primary_unc_col and primary_unc_col in df_copy.columns:
             if 'prediction_stddev' in df_copy.columns:
                  df_valid_unc_std = df_copy[[primary_unc_col, 'prediction_stddev']].dropna()
                  if len(df_valid_unc_std) > 1: pcc_unc_std, _ = safe_pearsonr(df_valid_unc_std['prediction_stddev'], df_valid_unc_std[primary_unc_col]); corr_lines.append(f"PCC(Prediction StdDev vs {primary_model_report_name} Uncertainty): {pcc_unc_std:.3f} (N={len(df_valid_unc_std)})")
                  else: corr_lines.append(f"PCC(Prediction StdDev vs {primary_model_report_name} Uncertainty): Not enough data.")
             if 'prediction_range' in df_copy.columns:
                  df_valid_unc_range = df_copy[[primary_unc_col, 'prediction_range']].dropna()
                  if len(df_valid_unc_range) > 1: pcc_unc_range, _ = safe_pearsonr(df_valid_unc_range['prediction_range'], df_valid_unc_range[primary_unc_col]); corr_lines.append(f"PCC(Prediction Range vs {primary_model_report_name} Uncertainty): {pcc_unc_range:.3f} (N={len(df_valid_unc_range)})")
                  else: corr_lines.append(f"PCC(Prediction Range vs {primary_model_report_name} Uncertainty): Not enough data.")
         elif primary_unc_col: corr_lines.append(f"Correlation with {primary_model_report_name} Uncertainty skipped (column '{primary_unc_col}' missing).")
         else: corr_lines.append(f"Correlation with {primary_model_report_name} Uncertainty skipped (no uncertainty column defined).")
         analysis_results[section_title] += "\n".join(corr_lines)
    else: analysis_results[section_title] += f"\nPrimary model '{primary_model_report_name}' for error correlation not found."

# --- Helper Function for New Case Study Criteria ---
def check_internal_structure_accuracy(domain_id, df_original, domain_stats, primary_model_pred_col,
                                      target_col='rmsf',
                                      structure_col='secondary_structure_encoded',
                                      norm_resid_col='normalized_resid',
                                      internal_min=0.2, internal_max=0.8,
                                      flexibility_min_threshold=0.7,
                                      mae_subset_max=0.15,
                                      min_residue_subset_count=8):
    """
    Checks if a domain has flexible, internal Alpha or Beta residues
    that are predicted with high accuracy.
    """
    try:
        # Ensure required columns exist in df_original
        required_cols = ['domain_id', norm_resid_col, structure_col, target_col, primary_model_pred_col]
        if not all(col in df_original.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_original.columns]
            logger.debug(f"Domain {domain_id}: Skipping internal check - missing required columns in original df: {missing}")
            return False

        domain_data = df_original[df_original['domain_id'] == domain_id].copy()
        if domain_data.empty: return False

        is_internal = (domain_data[norm_resid_col] > internal_min) & (domain_data[norm_resid_col] < internal_max)
        is_structured = domain_data[structure_col].isin([0, 1]) # 0=Helix, 1=Sheet
        is_flexible = domain_data[target_col] > flexibility_min_threshold
        subset_residues = domain_data[is_internal & is_structured & is_flexible]

        # Check unique residue count
        unique_resid_count = 0
        if 'resid' in subset_residues.columns: unique_resid_count = subset_residues['resid'].nunique()
        elif not subset_residues.empty: # Fallback if resid column missing but subset exists
             num_temps_est = df_original['temperature'].nunique() if 'temperature' in df_original.columns else 1
             unique_resid_count = len(subset_residues) / max(1, num_temps_est)
        if unique_resid_count < min_residue_subset_count: return False

        # Calculate MAE on the subset
        subset_actual = subset_residues[target_col]
        subset_pred = subset_residues[primary_model_pred_col]
        valid_mask = ~np.isnan(subset_actual) & ~np.isnan(subset_pred)
        # Also check count of valid pairs
        if valid_mask.sum() < min_residue_subset_count: return False

        mae_subset = mean_absolute_error(subset_actual[valid_mask], subset_pred[valid_mask])
        return mae_subset < mae_subset_max

    except KeyError as ke: logger.error(f"Domain {domain_id}: Missing column during internal structure check: {ke}"); return False
    except Exception as e: logger.error(f"Domain {domain_id}: Error during internal structure check: {e}", exc_info=True); return False


# --- Case Study Candidate Selection (Includes New Category 16.5 and Corrected Sorting) ---
# --- Helper Function (Assumed Defined Elsewhere) ---
# def check_internal_structure_accuracy(...): ...

def run_case_study_candidate_selection(df_original, analysis_results, primary_model_report_name=PRIMARY_MODEL_NAME, n_candidates=25):
    """Selects and formats candidate domains for case studies, including high potential candidates."""
    logger.info(f"Selecting case study candidates (Focusing on {primary_model_report_name} Metrics, n={n_candidates} per category, exclusive lists)...")
    section_title = "16. CASE STUDY CANDIDATES"
    analysis_results[section_title] = f"(Based on {primary_model_report_name} Metrics. Max {n_candidates} per category. Lists are mutually exclusive.)"

    primary_mae_col = f'{primary_model_report_name}_mae'; primary_pcc_col = f'{primary_model_report_name}_pcc'; primary_rho_col = f'{primary_model_report_name}_spearman_rho'
    primary_pred_col_name = MODEL_CONFIG[primary_model_report_name]['pred_col']
    base_cols_to_show = ['count_residues', 'actual_mean', 'actual_stddev']
    domain_metrics_path = analysis_results.get('_internal_domain_metrics_path')

    if not domain_metrics_path or not os.path.exists(domain_metrics_path):
        for i in range(1, 6): analysis_results[f'16.{i}'] = "Skipped: Domain metrics file not found."
        logger.warning("Domain metrics file path missing or file not found. Skipping case study selection.")
        return

    try:
        domain_df = pd.read_csv(domain_metrics_path, index_col='domain_id')
        required_overall_cols = [primary_pcc_col, primary_mae_col, 'count_residues', 'actual_stddev']
        if not all(c in domain_df.columns for c in required_overall_cols):
             raise ValueError(f"Domain metrics file missing essential overall columns: {[c for c in required_overall_cols if c not in domain_df.columns]}")

        temp_suffixes = set(); temp_col_pattern = re.compile(rf"^{re.escape(primary_model_report_name)}_(mae|pcc|spearman_rho)_(\d+\.\d+K)$")
        for col in domain_df.columns: match = temp_col_pattern.match(col);
        if match: temp_suffixes.add(match.group(2))
        sorted_temp_suffixes = sorted(list(temp_suffixes), key=lambda x: float(x[:-1]))
        primary_model_temp_cols_to_show = [f"{primary_model_report_name}_{metric}_{temp_suffix}" for temp_suffix in sorted_temp_suffixes for metric in ['mae', 'pcc', 'spearman_rho'] if f"{primary_model_report_name}_{metric}_{temp_suffix}" in domain_df.columns]

        delta_cols_calculated, delta_mae_col, delta_pcc_col, delta_rho_col = [], 'delta_mae_range', 'delta_pcc_range', 'delta_rho_range'
        if len(sorted_temp_suffixes) >= 2:
            first_t, last_t = sorted_temp_suffixes[0], sorted_temp_suffixes[-1]
            for metric in ['mae', 'pcc', 'spearman_rho']:
                 first_col, last_col = f'{primary_model_report_name}_{metric}_{first_t}', f'{primary_model_report_name}_{metric}_{last_t}'
                 delta_col = f'delta_{metric}_range'
                 if first_col in domain_df.columns and last_col in domain_df.columns:
                      domain_df[delta_col] = domain_df[last_col] - domain_df[first_col]; delta_cols_calculated.append(delta_col)

        all_cols_to_show = base_cols_to_show + [primary_mae_col, primary_pcc_col, primary_rho_col] + primary_model_temp_cols_to_show + delta_cols_calculated
        cols_to_display_final = list(dict.fromkeys([col for col in all_cols_to_show if col in domain_df.columns]))

        num_temperatures = df_original['temperature'].nunique() if 'temperature' in df_original.columns else 1
        # Size threshold for challenging candidates specifically adjusted
        challenging_size_threshold = 200 * num_temperatures
        # Default size threshold for other categories
        default_size_threshold = 120 * num_temperatures


        selected_domain_ids = set()

        # 16.1 High Accuracy
        try:
            # Original criteria
            criteria = (domain_df[primary_pcc_col] > 0.93) & (domain_df[primary_mae_col] < 0.12) & (domain_df['actual_stddev'] > 0.15)
            high_acc_all = domain_df[criteria]
            high_acc_display = high_acc_all[cols_to_display_final].nsmallest(n_candidates, primary_mae_col)
            selected_domain_ids.update(high_acc_display.index.tolist())
            analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = f"Criteria: Overall PCC>0.93, Overall MAE<0.12, ActualStd>0.15. Found {len(high_acc_all)} matching, showing top {len(high_acc_display)}.\n" + format_table(high_acc_display, floatfmt=".3f")
        except Exception as e: logger.warning(f"Error selecting High Accuracy Candidates: {e}", exc_info=True); analysis_results['16.1 HIGH ACCURACY CANDIDATES'] = f"Error: {e}"

        # 16.2 Good Temperature Handling
        try:
            if all(c in domain_df.columns for c in [primary_pcc_col, delta_mae_col, delta_pcc_col]):
                criteria = (domain_df[primary_pcc_col] > 0.90) & (domain_df[delta_mae_col].abs() < 0.10) & (domain_df[delta_pcc_col] > -0.05)
                temp_handle_pool = domain_df[criteria & ~domain_df.index.isin(selected_domain_ids)].copy()
                if not temp_handle_pool.empty:
                    temp_handle_pool['abs_delta_mae_range'] = temp_handle_pool[delta_mae_col].abs()
                    temp_handle_sorted = temp_handle_pool.nsmallest(n_candidates, 'abs_delta_mae_range')
                    temp_handle_display = temp_handle_sorted[cols_to_display_final]; selected_domain_ids.update(temp_handle_display.index.tolist())
                else: temp_handle_display = pd.DataFrame()
                analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = f"Criteria: Overall PCC>0.90, |DeltaMAE_Range|<0.10, DeltaPCC_Range>-0.05. Found {len(temp_handle_pool)} matching (excl. 16.1), showing top {len(temp_handle_display)}.\n" + format_table(temp_handle_display, floatfmt=".3f")
            else: analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = "Skipped: Missing columns for Temp Handling criteria (range deltas likely not calculated)."
        except Exception as e: logger.warning(f"Error selecting Temp Handling Candidates: {e}", exc_info=True); analysis_results['16.2 GOOD TEMPERATURE HANDLING CANDIDATES'] = f"Error: {e}"

        # --- MODIFIED 16.3 Challenging Candidates ---
        logger.info("Selecting Challenging Candidates (Modified Criteria)...")
        analysis_results['16.3 CHALLENGING CANDIDATES'] = ""
        try:
            # Check for necessary columns
            required_challenging = [primary_pcc_col, 'count_residues']
            if not all(c in domain_df.columns for c in required_challenging):
                 raise ValueError(f"Missing columns for Challenging criteria: {[c for c in required_challenging if c not in domain_df.columns]}")

            # Apply new criteria
            pcc_crit = (domain_df[primary_pcc_col] < 0.6) & (domain_df[primary_pcc_col] > 0.2)
            size_crit = domain_df['count_residues'] > challenging_size_threshold # Use adjusted threshold

            # Combine criteria & exclusivity
            challenging_pool_initial = domain_df[pcc_crit & size_crit & ~domain_df.index.isin(selected_domain_ids)].copy()
            logger.info(f"Found {len(challenging_pool_initial)} candidates after initial challenging filters (PCC 0.4-0.7, Size>{900}res, excl. 16.1-2).")

            # --- Filter for non-random coil (Requires original data) ---
            challenging_filtered_ids = []
            if not challenging_pool_initial.empty:
                 structure_col = 'secondary_structure_encoded'
                 if structure_col not in df_original.columns:
                      logger.warning(f"Cannot filter Challenging Candidates by structure: Column '{structure_col}' missing in original data.")
                      # Proceed without structure filtering if column is missing
                      challenging_filtered_ids = challenging_pool_initial.index.tolist()
                 else:
                      # Iterate through initial candidates and check structure content
                      for domain_id in tqdm(challenging_pool_initial.index, desc="Checking Challenging Structure"):
                           domain_data = df_original[df_original['domain_id'] == domain_id]
                           if domain_data.empty: continue

                           # Calculate percentage of non-loop (Helix=0, Sheet=1)
                           # Group by unique residue if possible (e.g., using resid) then check SS for each residue
                           # Simpler approach: check percentage across all rows for the domain
                           non_loop_count = domain_data[domain_data[structure_col].isin([0, 1])].shape[0]
                           total_count = domain_data.shape[0]
                           if total_count > 0:
                                percent_structured = (non_loop_count / total_count) * 100
                                # Keep if > 30% is Helix or Sheet (adjust threshold as needed)
                                if percent_structured > 20.0:
                                     challenging_filtered_ids.append(domain_id)
                           else:
                                logger.debug(f"Challenging candidate {domain_id} has no rows in original data?")

            logger.info(f"Found {len(challenging_filtered_ids)} challenging candidates after structure filter (>20% H/E).")
            # Select from the filtered IDs
            challenging_pool_final = domain_df.loc[challenging_filtered_ids]

            # Sort by Overall MAE (Ascending - find the 'best' of the challenging ones)
            challenging_display = challenging_pool_final[cols_to_display_final].nsmallest(n_candidates, primary_mae_col)
            selected_domain_ids.update(challenging_display.index.tolist())

            criteria_desc = f"Criteria: 0.5<PCC<0.7, Size>{1000}res, >30% Non-Loop Structure."
            analysis_results['16.3 CHALLENGING CANDIDATES'] = f"{criteria_desc}\nFound {len(challenging_pool_final)} matching (excl. 16.1-2), showing top {len(challenging_display)} sorted by Overall MAE asc.\n" + \
                format_table(challenging_display, floatfmt=".3f")

        except Exception as e:
             logger.warning(f"Error selecting Challenging Candidates: {e}", exc_info=True)
             analysis_results['16.3 CHALLENGING CANDIDATES'] = f"Error during selection: {e}"


        # 16.4 Thermophiles (1h)
        try:
            thermophile_criteria = domain_df.index.str.contains("1h", case=False, na=False) & ~domain_df.index.isin(selected_domain_ids)
            thermophile_pool = domain_df[thermophile_criteria]
            thermophile_display = thermophile_pool[cols_to_display_final].nsmallest(n_candidates, primary_mae_col)
            selected_domain_ids.update(thermophile_display.index.tolist())
            analysis_results['16.4 THERMOPHILES (1h)'] = f"Criteria: '1h' in domain_id. Found {len(thermophile_pool)} matching (excl. 16.1-3), showing top {len(thermophile_display)} sorted by Overall MAE.\n" + format_table(thermophile_display, floatfmt=".3f")
        except Exception as e: logger.warning(f"Error selecting Thermophile Candidates: {e}", exc_info=True); analysis_results['16.4 THERMOPHILES (1h)'] = f"Error: {e}"

        # 16.5 High Potential High Accuracy Candidates
        logger.info("Selecting High Potential High Accuracy Candidates...")
        analysis_results['16.5 HIGH POTENTIAL HIGH ACCURACY CANDIDATES'] = ""
        try:
            # Use default size threshold here unless specifically overridden
            size_crit = domain_df['count_residues'] > default_size_threshold
            pcc_320_col_name = f'{primary_model_report_name}_pcc_320.0K'
            stddev_crit = domain_df['actual_stddev'] > 0.45
            if pcc_320_col_name not in domain_df.columns:
                logger.warning(f"Column '{pcc_320_col_name}' not found. Cannot apply PCC(320K) criteria for 16.5.")
                analysis_results['16.5 HIGH POTENTIAL HIGH ACCURACY CANDIDATES'] = f"Skipped: Missing required column '{pcc_320_col_name}'."
                final_candidates_pool_hp = pd.DataFrame() # Ensure pool is empty
                high_potential_display = pd.DataFrame()
            else:
                pcc_320_crit = domain_df[pcc_320_col_name] > 0.92
                initial_candidate_pool_hp = domain_df[size_crit & pcc_320_crit & stddev_crit & ~domain_df.index.isin(selected_domain_ids)].copy()

                if not initial_candidate_pool_hp.empty:
                     internal_check_params = {'primary_model_pred_col': primary_pred_col_name, 'target_col': TARGET_COL, 'structure_col': 'secondary_structure_encoded', 'norm_resid_col': 'normalized_resid', 'internal_min': 0.2, 'internal_max': 0.8, 'flexibility_min_threshold': 0.7, 'mae_subset_max': 0.15, 'min_residue_subset_count': 8}
                     tqdm.pandas(desc="Checking Internal Structures (HP)")
                     initial_candidate_pool_hp['meets_internal_criteria'] = initial_candidate_pool_hp.progress_apply(
                         lambda row: check_internal_structure_accuracy(row.name, df_original, row, **internal_check_params), axis=1)
                     final_candidates_pool_hp = initial_candidate_pool_hp[initial_candidate_pool_hp['meets_internal_criteria']]

                     sort_columns, ascending_order = [], []
                     if pcc_320_col_name in final_candidates_pool_hp.columns: sort_columns.append(pcc_320_col_name); ascending_order.append(False)
                     if primary_mae_col in final_candidates_pool_hp.columns: sort_columns.append(primary_mae_col); ascending_order.append(True)

                     if not sort_columns:
                          high_potential_display = final_candidates_pool_hp[cols_to_display_final].head(5)
                     else:
                          df_to_sort_hp = final_candidates_pool_hp[cols_to_display_final]
                          valid_sort_columns_hp = [col for col in sort_columns if col in df_to_sort_hp.columns]
                          valid_ascending_order_hp = [asc for col, asc in zip(sort_columns, ascending_order) if col in valid_sort_columns_hp]
                          if not valid_sort_columns_hp: high_potential_display = df_to_sort_hp.head(5)
                          else: high_potential_display = df_to_sort_hp.sort_values(by=valid_sort_columns_hp, ascending=valid_ascending_order_hp).head(5)
                else: high_potential_display, final_candidates_pool_hp = pd.DataFrame(), pd.DataFrame()

                # Use original internal_check_params for description
                internal_check_params_desc = {'flexibility_min_threshold': 0.7, 'mae_subset_max': 0.15, 'min_residue_subset_count': 8}
                criteria_desc = (f"Criteria: Size>{120}res, PCC(320K)>0.92, ActualStd>0.45, " f"+ Internal Flexible (RMSF>{internal_check_params_desc['flexibility_min_threshold']}) " f"Alpha/Beta accuracy (MAE<{internal_check_params_desc['mae_subset_max']}, N>{internal_check_params_desc['min_residue_subset_count']}).")
                analysis_results['16.5 HIGH POTENTIAL HIGH ACCURACY CANDIDATES'] = f"{criteria_desc}\nFound {len(final_candidates_pool_hp)} matching (excl. 16.1-4), showing top {len(high_potential_display)} sorted by PCC(320K) desc, MAE asc.\n" + format_table(high_potential_display, floatfmt=".3f")

        except Exception as e: logger.warning(f"Error selecting High Potential High Accuracy Candidates: {e}", exc_info=True); analysis_results['16.5 HIGH POTENTIAL HIGH ACCURACY CANDIDATES'] = f"Error: {e}"

    except FileNotFoundError: logger.error(f"Domain metrics file not found at {domain_metrics_path}. Cannot select candidates."); analysis_results['16. CASE STUDY CANDIDATES'] += "\nSkipped: Domain metrics file not found."
    except Exception as e: logger.error(f"Error during case study selection using {domain_metrics_path}: {e}", exc_info=True); analysis_results['16. CASE STUDY CANDIDATES'] += f"\nError reading/processing domain metrics file: {e}"
# --- Placeholder Sections ---
def run_feature_attribution_placeholder(df, analysis_results):
    analysis_results['18. FEATURE ATTRIBUTION & INTERPRETABILITY (Placeholder)'] = "Placeholder: Requires model object/precomputed scores (SHAP/Captum/Attention)."
def run_temperature_ablation_placeholder(df, analysis_results):
    analysis_results['19. TEMPERATURE-ENCODING ABLATION (Placeholder)'] = "Placeholder: Requires predictions from model trained without temperature."
def run_external_validation_placeholder(df, analysis_results):
    analysis_results['20. CROSS-DATASET / EXTERNAL VALIDATION (Placeholder)'] = "Placeholder: Requires external dataset and model inference capability."
def run_computational_performance_placeholder(df, analysis_results):
    analysis_results['21. COMPUTATIONAL PERFORMANCE & SCALABILITY (Placeholder)'] = "Placeholder: Requires separate benchmark data (inference vs MD)."

# --- Error Analysis vs Features ---
def run_error_vs_feature_analysis(df, analysis_results):
    logger.info("Running Error vs Feature analyses...")
    analysis_results['22. ERROR ANALYSIS VS FEATURES'] = f"(Error = Absolute Error for {PRIMARY_MODEL_NAME})"
    primary_model_config = MODEL_CONFIG.get(PRIMARY_MODEL_NAME)
    if not primary_model_config: analysis_results['22. ERROR ANALYSIS VS FEATURES'] = "Skipped: Primary model config missing."; return
    pred_col, error_col = primary_model_config['pred_col'], f"{PRIMARY_MODEL_NAME}_abs_error"
    df_copy = df.copy()
    if error_col not in df_copy.columns:
        if TARGET_COL in df_copy.columns and pred_col in df_copy.columns: df_copy[error_col] = (df_copy[pred_col] - df_copy[TARGET_COL]).abs()
        else: analysis_results['22. ERROR ANALYSIS VS FEATURES'] = f"Skipped: Cannot calculate error '{error_col}'."; return

    features_to_analyze = {'bfactor_norm': '22.1 ERROR VS. NORMALIZED B-FACTOR', 'contact_number': '22.2 ERROR VS. CONTACT NUMBER', 'coevolution_score': '22.3 ERROR VS. CO-EVOLUTION SIGNAL'}
    for feature_col, section_title in features_to_analyze.items():
        if feature_col in df_copy.columns:
            logger.info(f"Analyzing error vs '{feature_col}'...")
            df_feature_err = df_copy[[feature_col, error_col]].dropna()
            if len(df_feature_err) < 50: analysis_results[section_title] = f"Skipped: Not enough valid data points."; continue
            try:
                 n_bins = 5
                 # Use dropna on the feature column *before* qcut
                 feature_values_no_na = df_feature_err[feature_col].dropna()
                 if len(feature_values_no_na) < n_bins * 2 : # Check after dropping NaNs
                      raise ValueError(f"Not enough non-NaN values in {feature_col} for binning.")

                 df_feature_err['feature_bin'], bin_edges = pd.qcut(feature_values_no_na, q=n_bins, labels=False, duplicates='drop', retbins=True)
                 if df_feature_err['feature_bin'].nunique() < 2: raise ValueError("Could not create bins.")
                 bin_range_labels = {i: f"{bin_edges[i]:.3g}-{bin_edges[i+1]:.3g}" for i in range(len(bin_edges)-1)}

                 # Group the original df_feature_err using the bin assignments
                 # Need to handle potential index misalignment if NaNs were dropped for qcut
                 # Safer: Re-assign bins to the df_feature_err based on original values
                 df_feature_err['feature_bin'] = pd.cut(df_feature_err[feature_col], bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')


                 binned_errors = []
                 # Group by the newly assigned bins (handle potential NaNs in bin assignment)
                 for bin_idx, group in df_feature_err.dropna(subset=['feature_bin']).groupby('feature_bin'):
                      if len(group) > 2:
                           # Use int(bin_idx) as it should be numeric label from cut/qcut
                           binned_errors.append({'Feature Range': bin_range_labels.get(int(bin_idx), "N/A"),
                                                 'Mean Feature Value': group[feature_col].mean(),
                                                 f'Mean {PRIMARY_MODEL_NAME} Abs Error': group[error_col].mean(),
                                                 f'Median {PRIMARY_MODEL_NAME} Abs Error': group[error_col].median(),
                                                 'Count': len(group)})
                 if binned_errors: analysis_results[section_title] = format_table(pd.DataFrame(binned_errors).set_index('Feature Range'))
                 else: analysis_results[section_title] = "No results after binning."
            except Exception as e: logger.warning(f"Could not analyze error vs {feature_col}: {e}"); analysis_results[section_title] = f"Error: {e}"
        else: analysis_results[section_title] = f"Skipped: Feature column '{feature_col}' not found."


# --- Main Analysis Runner ---
def run_analysis(df_path, output_file):
    """Runs the full analysis pipeline with enhanced error handling."""
    start_time = time.time()
    logger.info(f"Starting analysis for file: {df_path}")
    try:
        df = pd.read_csv(df_path)
        # Apply renaming based on MODEL_CONFIG
        rename_dict = {}
        old_pred_key = "Attention_ESM_rmsf" # Assuming this was the old name for DeepFlex
        if old_pred_key in df.columns and PRIMARY_MODEL_NAME == "DeepFlex":
            rename_dict[old_pred_key] = MODEL_CONFIG["DeepFlex"]['pred_col']
            logger.info(f"Renaming column '{old_pred_key}' to '{rename_dict[old_pred_key]}'")
        old_unc_key = "Attention_ESM_rmsf_uncertainty"
        if old_unc_key in df.columns and PRIMARY_MODEL_NAME == "DeepFlex" and MODEL_CONFIG["DeepFlex"]['unc_col']:
            rename_dict[old_unc_key] = MODEL_CONFIG["DeepFlex"]['unc_col']
            logger.info(f"Renaming column '{old_unc_key}' to '{rename_dict[old_unc_key]}'")
        if rename_dict: df.rename(columns=rename_dict, inplace=True)

        df.attrs['source_file'] = df_path
        logger.info(f"Successfully loaded data: {df.shape}")
        if df.empty: logger.error("Input CSV is empty. Aborting."); return
    except Exception as e: logger.error(f"Error loading data: {e}", exc_info=True); return

    analysis_results = {'_internal_output_file_path': output_file}

    # Define the order and functions for analysis steps
    analysis_steps = [
        ("1. BASIC INFORMATION", run_basic_info),
        ("2. MISSING VALUE SUMMARY", run_missing_values),
        ("3. OVERALL DESCRIPTIVE STATISTICS (Key Variables)", run_descriptive_stats),
        ("4. DATA DISTRIBUTIONS", run_data_distributions),
        ("5. COMPREHENSIVE MODEL COMPARISON", run_model_comparison),
        ("5.6 DIHEDRAL ANGLE (RAMACHANDRAN) ANALYSIS", run_dihedral_analysis),
        ("6. UNCERTAINTY ANALYSIS", run_uncertainty_analysis),
        ("7. DOMAIN-LEVEL PERFORMANCE METRICS", run_domain_level_analysis),
        ("8. PERFORMANCE BY AMINO ACID", run_amino_acid_performance),
        ("9. PERFORMANCE BY NORMALIZED RESIDUE POSITION", run_norm_resid_performance),
        ("10. PERFORMANCE BY CORE/EXTERIOR", run_core_exterior_performance),
        ("11. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)", run_secondary_structure_performance),
        ("12. PERFORMANCE BY ACTUAL RMSF QUANTILE", lambda d, r: run_performance_vs_actual_rmsf(d, r)),
        ("13. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", run_rel_accessibility_performance),
        ("14. PERFORMANCE BY BFACTOR QUANTILE", run_bfactor_performance),
        ("15. MODEL DISAGREEMENT VS. ERROR", lambda d, r: run_model_disagreement_analysis(d, r)),
        ("16. CASE STUDY CANDIDATES", lambda d, r: run_case_study_candidate_selection(d, r)), # Pass df
        ("17. STATISTICAL SIGNIFICANCE TESTS", lambda d, r: None), # Placeholder
        ("18. FEATURE ATTRIBUTION & INTERPRETABILITY (Placeholder)", run_feature_attribution_placeholder),
        ("19. TEMPERATURE-ENCODING ABLATION (Placeholder)", run_temperature_ablation_placeholder),
        ("20. CROSS-DATASET / EXTERNAL VALIDATION (Placeholder)", run_external_validation_placeholder),
        ("21. COMPUTATIONAL PERFORMANCE & SCALABILITY (Placeholder)", run_computational_performance_placeholder),
        ("22. ERROR ANALYSIS VS FEATURES", run_error_vs_feature_analysis),
    ]

    # Run analysis steps
    parent_headers = set(re.match(r"(\d+)\.", k).group(1) for k,v in analysis_steps if re.match(r"(\d+)\.", k))
    for p_header_num in sorted(parent_headers, key=int):
        parent_key = next((k for k,v in analysis_steps if k.startswith(f"{p_header_num}. ")), None)
        if parent_key: analysis_results[parent_key] = ""

    # --- Execute Analysis Steps ---
    # Store df in a way accessible to lambda functions if needed later (though direct pass is better)
    # analysis_results['_internal_df'] = df # Example, not strictly needed with current lambdas

    for section_key, analysis_func in analysis_steps:
        # Skip placeholders explicitly defined as None, except for sig test handled later
        if analysis_func is None and section_key != "17. STATISTICAL SIGNIFICANCE TESTS":
            # Add a placeholder message if the function is truly None
            # analysis_results[section_key] = "Analysis step not implemented."
            continue

        logger.info(f"--- Running Analysis: {section_key} ---")
        try:
            # Call the function, passing df and analysis_results
            analysis_func(df, analysis_results)
        except Exception as e:
            logger.error(f"Error executing analysis step '{section_key}': {e}", exc_info=True)
            analysis_results[section_key] = f"!!! ERROR DURING ANALYSIS: {e} !!!"


    # Explicitly run significance tests after domain metrics are calculated
    logger.info("--- Running Analysis: 17. STATISTICAL SIGNIFICANCE TESTS ---")
    try:
        per_domain_metrics_data = {}
        domain_metrics_path = analysis_results.get('_internal_domain_metrics_path')
        if domain_metrics_path and os.path.exists(domain_metrics_path):
             try:
                 domain_df = pd.read_csv(domain_metrics_path, index_col='domain_id')
                 # Convert DataFrame rows to nested dict {domain_id: {model_name: {'mae': val, 'pcc': val}}}
                 for domain_id, row in domain_df.iterrows():
                      domain_data = {model_name: {'mae': row.get(f"{model_name}_mae", np.nan), 'pcc': row.get(f"{model_name}_pcc", np.nan)} for model_name in MODEL_CONFIG.keys() if f"{model_name}_mae" in row or f"{model_name}_pcc" in row}
                      if domain_data: per_domain_metrics_data[domain_id] = domain_data
                 if not per_domain_metrics_data: raise FileNotFoundError # Trigger recalculation
             except Exception as load_err:
                  logger.warning(f"Failed to load/process domain metrics from {domain_metrics_path}: {load_err}. Recalculating...")
                  per_domain_metrics_data = calculate_per_domain_metrics(df) # Recalculate using original df
        else:
             logger.warning("Domain metrics file path not found. Recalculating domain metrics for significance testing.")
             per_domain_metrics_data = calculate_per_domain_metrics(df) # Recalculate using original df

        if per_domain_metrics_data:
            significance_results = calculate_significance_tests(per_domain_metrics_data, PRIMARY_MODEL_NAME, KEY_BASELINES_FOR_SIG_TEST)
            if significance_results:
                 sig_text = f"Comparing {PRIMARY_MODEL_NAME} against key baselines using Wilcoxon signed-rank test on per-domain metrics.\n" \
                            f"H0: Median difference is zero.\n" \
                            f"MAE Test (alternative='less'): Is {PRIMARY_MODEL_NAME} MAE significantly smaller than baseline?\n" \
                            f"PCC Test (alternative='greater'): Is {PRIMARY_MODEL_NAME} PCC significantly larger than baseline?\n\n"
                 sig_data_to_format = [{"Baseline": bl, "Metric": met, "p-value": res.get('p_value', np.nan), "N_pairs": res.get('N', 0)} for test_name, res in significance_results.items() for bl, met in [test_name.split('_vs_')]]
                 # Pass headers explicitly for list of dicts
                 sig_headers = list(sig_data_to_format[0].keys()) if sig_data_to_format else "keys"
                 sig_text += format_table(sig_data_to_format, headers=sig_headers, floatfmt=".4g")
                 analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = sig_text
            else: analysis_results['17. STATISTICAL SIGNIFICANCE TESTS'] = "Significance tests could not be calculated."
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
                if section_key.startswith('_internal'): continue
                content = analysis_results[section_key]
                match = re.match(r"(\d+(?:\.\d+)*)\.?\s*(.*)", section_key)
                if match:
                    num_str, title_str = match.group(1), match.group(2).strip().upper()
                    if '.' not in num_str and title_str: f.write(f"\n{'='*80}\n## {num_str}. {title_str} ##\n{'='*80}\n")
                    elif title_str: f.write(f"\n{'-'*60}\n### {num_str} {title_str} ###\n{'-'*60}\n")
                else: f.write(f"\n{'='*80}\n## {section_key.upper()} ##\n{'='*80}\n")
                if content and isinstance(content, str) and content.strip(): f.write(content + "\n\n")
                elif content and not isinstance(content, str):
                     if isinstance(content, pd.DataFrame): f.write(format_table(content) + "\n\n")
                     elif isinstance(content, dict): f.write(json.dumps(content, indent=4, default=str) + "\n\n")
                     else: f.write(str(content) + "\n\n")
    except Exception as e: logger.error(f"Failed to write analysis results to {output_file}: {e}", exc_info=True)

    end_time = time.time()
    logger.info(f"Analysis complete. Results saved to: {output_file}")
    if analysis_results.get('_internal_domain_metrics_path'): logger.info(f"Detailed domain metrics saved to CSV in: {os.path.dirname(analysis_results.get('_internal_domain_metrics_path'))}")
    logger.info(f"Total analysis time: {end_time - start_time:.2f} seconds.")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive analysis on the aggregated flexibility prediction dataset.")
    parser.add_argument(
        "--input_csv", type=str,
        default="/home/s_felix/FINAL_PROJECT/Data_Analysis/data/02_final_analysis_dataset.csv",
        help="Path to the aggregated input CSV file."
    )
    parser.add_argument(
        "--output_file", type=str,
        default="enhanced_general_analysis_report_v3_corrected.txt", # Updated output filename
        help="Path to save the enhanced analysis results text file."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    input_csv_path = args.input_csv if os.path.isabs(args.input_csv) else os.path.join(script_dir, args.input_csv)
    output_file_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(script_dir, args.output_file)

    output_dir = os.path.dirname(output_file_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    # Ensure tqdm is available for pandas
    try:
        # from tqdm.auto import tqdm # Moved import to top
        tqdm.pandas()
    except ImportError:
        logger.warning("tqdm not found. Progress bars for pandas apply will not be shown. Install with: pip install tqdm")

    run_analysis(input_csv_path, output_file_path)