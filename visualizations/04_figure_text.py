#!/usr/bin/env python3
"""
Figure 4 Detailed Analysis Data Generation Script (v2 - Fixed Errors)

Reads the final analysis dataset, performs aggregations relevant to each
panel of Figure 4, and writes a detailed CSV file suitable for LLM context
or manual report writing. Does NOT generate plots.

Output: CSV file ('figure4_analysis_detailed.csv') saved to 'figure4_outputs'.
"""

import os
import pandas as pd
import numpy as np
# Matplotlib/Seaborn are not needed for data generation
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats # For aggregation and CI calculations
import logging
import warnings
from pathlib import Path
import csv
# from scipy.ndimage import gaussian_filter # Not needed if Ramachandran is just binned

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
# !!! IMPORTANT: Update this path to your actual data file location !!!
INPUT_CSV = "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "figure4_outputs" # Assumes this dir exists or will be created
OUTPUT_CSV_NAME = "figure4_analysis_detailed.csv"

MODEL_MAP = {
    'DeepFlex': 'Attention_ESM_rmsf',
    'ESM-Only': 'ESM_Only_rmsf',
    'VoxelFlex-3D': 'voxel_rmsf',
}
MODELS_TO_PLOT = ['DeepFlex', 'ESM-Only', 'VoxelFlex-3D']
PRIMARY_MODEL_NAME = 'DeepFlex'

# Ground truth and key feature columns
TARGET_COL = 'rmsf'
DSSP_COL = 'dssp'
RSA_COL = 'relative_accessibility'
NORM_RESID_COL = 'normalized_resid'
BFACTOR_COL = 'bfactor_norm'
PHI_COL = 'phi'
PSI_COL = 'psi'
RESNAME_COL = 'resname'

# --- Logging Setup ---
def setup_logging(log_dir):
    """Configures logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
    log_file = log_dir / "figure4_text_generation.log"
    # Use a distinct logger name if running alongside the plotting script's logger
    logger = logging.getLogger("Figure4TextLogger")
    logger.setLevel(logging.INFO)
    # Remove existing handlers if any to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File Handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Logging initialized for Text Generation. Log file: {log_file}")
    return logger

# --- Helper Functions ---
def calculate_mae(df, model_col, target_col, logger):
    """Safely calculates Mean Absolute Error, logs issues."""
    if model_col not in df.columns or target_col not in df.columns:
        # Ensure logger exists before using it
        try: logger.warning(f"Cannot calculate MAE: Missing {model_col} or {target_col}")
        except NameError: print(f"WARN: Cannot calculate MAE: Missing {model_col} or {target_col}")
        return pd.Series(np.nan, index=df.index)
    pred = pd.to_numeric(df[model_col], errors='coerce')
    target = pd.to_numeric(df[target_col], errors='coerce')
    return np.abs(pred - target)

def map_secondary_structure(dssp_code):
    """Maps DSSP codes to broader categories."""
    if pd.isna(dssp_code): return 'Other'
    code = str(dssp_code).upper()
    if code in ('G', 'H', 'I'): return 'α-Helix'
    if code in ('B', 'E'): return 'β-Sheet'
    return 'Loop/Other'

def map_core_exterior(rsa_value, threshold=0.2):
    """Classifies residue as Core or Exterior based on RSA."""
    if pd.isna(rsa_value): return 'Unknown'
    try:
        return 'Core' if float(rsa_value) <= threshold else 'Exterior'
    except (ValueError, TypeError):
        return 'Unknown'

def create_position_bins(n_bins=5):
    """Creates labels and boundaries for sequence position bins."""
    boundaries = np.linspace(0, 1, n_bins + 1)
    labels = []
    for i in range(n_bins):
        if i == 0: label = f"N-term ({boundaries[i]:.1f}-{boundaries[i+1]:.1f})"
        elif i == n_bins - 1: label = f"C-term ({boundaries[i]:.1f}-{boundaries[i+1]:.1f})"
        else: label = f"Mid ({boundaries[i]:.1f}-{boundaries[i+1]:.1f})"
        labels.append(label)
    return boundaries, labels


def format_value_for_csv(v):
    """Formats values consistently for CSV output. Handles arrays/lists/dicts robustly."""
    # 1. Handle None type explicitly first
    if v is None:
        return "NaN"

    # 2. Handle specific scalar types
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        # Check for scalar NaN before formatting
        if np.isnan(v):
            return "NaN"
        return f"{v:.4f}"
    if isinstance(v, str):
        # Quote strings containing commas, quotes, or newlines
        val_esc = v.replace('"', '""')
        return f'"{val_esc}"' if any(c in val_esc for c in [',', '"', '\n']) else val_esc
    if isinstance(v, bool):
        return str(v)

    # 3. Handle specific iterable types (lists, tuples, dictionaries) recursively
    if isinstance(v, dict):
        return "; ".join([f"{k}:{format_value_for_csv(val)}" for k, val in v.items()])
    if isinstance(v, (list, tuple)):
        if not v: # Handle empty list/tuple
             return ""
        return "; ".join(map(str, [format_value_for_csv(item) for item in v])) # Removed pd.notna check, handled by recursion

    # 4. Handle Numpy arrays and Pandas Series/Index
    if isinstance(v, (np.ndarray, pd.Series, pd.Index)):
        if v.size == 0 or (isinstance(v, pd.Series) and v.empty):
            return "Empty Array/Series"
        # Apply formatting element-wise and join
        # Need to handle potential NaNs within the array/series during formatting
        formatted_elements = [format_value_for_csv(item) for item in v]
        return "; ".join(formatted_elements)

    # 5. Handle pandas Categorical explicitly (extract categories or codes)
    if isinstance(v, pd.Categorical):
         # Example: return categories as string, adjust if needed
         return "; ".join(map(str, v.categories))

    # 6. Final fallback for other types (like pandas Timestamps, etc.)
    try:
        # Check for Pandas NaT (Not a Time) which is tricky
        if pd.isna(v):
            return "NaN"
        # Attempt standard string conversion
        return str(v).replace('"', '""')
    except Exception as e:
        # Log the error and the type that caused it
        try:
            logger.warning(f"Formatting Error for type {type(v)}: {e}. Value: {v}")
        except NameError: # If logger isn't available
            print(f"WARN: Formatting Error for type {type(v)}: {e}. Value: {v}")
        return "Formatting Error"

# --- Data Loading and Preparation for Stats ---
def load_and_prepare_data_for_stats(csv_path, logger):
    """Loads CSV, prepares data needed for statistical aggregations."""
    global analysis_data # Use global only if needed for overall stats initially
    analysis_data = {} # Reset if called multiple times
    logger.info(f"Loading data for stats from: {csv_path}")
    overall_stats = {}
    try:
        # Specify low_memory=False if dtype warnings occur
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows.")
        overall_stats['Total Rows Loaded'] = len(df)
    except FileNotFoundError: logger.critical(f"Input CSV not found: {csv_path}"); return None, overall_stats
    except Exception as e: logger.critical(f"Error loading CSV: {e}", exc_info=True); return None, overall_stats

    # Check essential columns
    required_data_cols = [TARGET_COL, DSSP_COL, RSA_COL, NORM_RESID_COL,
                          BFACTOR_COL, PHI_COL, PSI_COL, RESNAME_COL]
    required_model_cols = [MODEL_MAP.get(m) for m in MODELS_TO_PLOT if MODEL_MAP.get(m)] # Handle missing models
    all_required = required_data_cols + required_model_cols
    missing = [col for col in all_required if col not in df.columns]
    if missing: logger.critical(f"Input CSV is missing required columns: {missing}"); return None, overall_stats

    logger.info("Converting column types...")
    numeric_cols = [TARGET_COL, RSA_COL, NORM_RESID_COL, BFACTOR_COL, PHI_COL, PSI_COL] + required_model_cols
    for col in numeric_cols:
        if col in df.columns: # Check if column exists before conversion
             df[col] = pd.to_numeric(df[col], errors='coerce')
             if df[col].isnull().any(): logger.warning(f"NaNs introduced/present in '{col}' after numeric conversion.")

    logger.info("Calculating MAE...")
    mae_cols_to_check = []
    for model_name in MODELS_TO_PLOT:
        model_col = MODEL_MAP.get(model_name)
        if model_col and model_col in df.columns:
             mae_col = f"{model_name}_MAE"
             df[mae_col] = calculate_mae(df, model_col, TARGET_COL, logger) # Pass logger
             overall_stats[f'{model_name} Overall Mean MAE'] = df[mae_col].mean() # mean() handles NaNs
             overall_stats[f'{model_name} Overall Median MAE'] = df[mae_col].median() # median() handles NaNs
             mae_cols_to_check.append(mae_col)
        else:
             logger.warning(f"Model column '{model_col}' for '{model_name}' not found. Skipping MAE calculation.")


    # Drop rows only if MAE is NaN for ALL plotted models - crucial for stats
    rows_before_drop = len(df)
    df.dropna(subset=mae_cols_to_check, how='all', inplace=True) # Drop if ALL are NaN
    rows_after_drop = len(df)
    if rows_after_drop < rows_before_drop:
        logger.info(f"Dropped {rows_before_drop - rows_after_drop:,} rows due to NaN in ALL MAE columns.")
    overall_stats['Rows after MAE NaN Drop'] = rows_after_drop
    logger.info(f"{rows_after_drop:,} rows remaining after dropping NaN MAE values.")

    logger.info("Mapping categorical features...")
    df['SS_Category'] = df[DSSP_COL].apply(map_secondary_structure)
    df['Core_Exterior'] = df[RSA_COL].apply(map_core_exterior)

    logger.info("Binning continuous features...")
    # --- Position Bin ---
    if NORM_RESID_COL in df.columns and df[NORM_RESID_COL].notna().any():
        df[NORM_RESID_COL] = df[NORM_RESID_COL].clip(0, 1)
        pos_boundaries, pos_labels = create_position_bins(5)
        # Use pd.cut carefully, handle edges
        df['Position_Bin'] = pd.cut(df[NORM_RESID_COL], bins=pos_boundaries, labels=pos_labels, right=False, include_lowest=True)
        # Ensure it's categorical for later grouping, adding Unknown for potential NaNs from cut
        df['Position_Bin'] = pd.Categorical(df['Position_Bin'], categories=pos_labels + ['Unknown'], ordered=True)
        df['Position_Bin'].fillna('Unknown', inplace=True) # Fill NaNs introduced by cut
    else:
        logger.warning(f"'{NORM_RESID_COL}' column missing or empty. Skipping position binning.")
        df['Position_Bin'] = 'Not Available'

    # --- Quantile Labeling ---
    n_quantiles = 5
    for col_name, label_prefix in [(RSA_COL, "RSA"), (BFACTOR_COL, "BFactor")]:
        label_col = f"{label_prefix}_Quantile_Label"
        df[label_col] = 'Not Available' # Default
        if col_name not in df.columns or df[col_name].isnull().all(): continue
        try:
            valid_data_series = df[col_name].dropna()
            if len(valid_data_series) < n_quantiles * 2: raise ValueError("Not enough unique data points.")
            # Use rank to handle duplicates before qcut
            _, q_boundaries = pd.qcut(valid_data_series.rank(method='first'), n_quantiles, retbins=True, duplicates='drop')
            q_boundaries = np.unique(q_boundaries) # Ensure unique boundaries
            if len(q_boundaries) < 2: raise ValueError("Could not determine valid quantile boundaries.")
            # Create labels based on actual boundaries
            labels = [f"{q_boundaries[i]:.2f}-{q_boundaries[i+1]:.2f}" for i in range(len(q_boundaries) - 1)]
            # Apply pd.cut using the determined boundaries
            quantile_labels_series = pd.cut(df[col_name], bins=q_boundaries, labels=labels, include_lowest=True, right=True, duplicates='drop') # Ensure right=True for consistency
            df[label_col] = pd.Categorical(quantile_labels_series, categories=labels, ordered=True)
            df[label_col] = df[label_col].cat.add_categories('Unknown').fillna('Unknown') # Add Unknown and fill NaNs
            logger.info(f"Successfully created string labels for {label_col}.")
        except Exception as e:
            logger.warning(f"Could not generate string labels for {label_col}: {e}. Check data distribution.")
            df[label_col] = 'Error'

    # Final check on categorical columns potentially needed for grouping
    for col in ['SS_Category', 'Core_Exterior', 'Position_Bin', 'RSA_Quantile_Label', 'BFactor_Quantile_Label']:
         if col in df.columns: df[col] = df[col].astype(str).fillna('Unknown') # Ensure string type for grouping

    overall_stats['Rows after Final Processing'] = len(df)
    logger.info("Data preparation finished for stats.")
    return df, overall_stats


# --- Aggregation Functions ---

def aggregate_panel_a_ss(df):
    """Aggregates data for Panel A (Secondary Structure Boxplot)."""
    panel_id = "Panel A"
    plot_type = "Boxplot"
    x_var = "SS_Category"
    y_var = "MAE"
    results = []
    try:
        mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
        plot_data = df[[x_var] + mae_cols].melt(id_vars=[x_var], var_name='Model_MAE', value_name=y_var)
        plot_data['Model'] = plot_data['Model_MAE'].str.replace('_MAE', '')
        plot_data = plot_data[plot_data['Model'].isin(MODELS_TO_PLOT)]
        category_order = ['α-Helix', 'β-Sheet', 'Loop/Other']
        plot_data = plot_data[plot_data[x_var].isin(category_order)]

        if plot_data.empty: logger.warning(f"{panel_id}: No data after filtering."); return results

        grouped = plot_data.groupby([x_var, 'Model'], observed=True)[y_var]
        stats = grouped.agg(['mean', 'median', 'count',
                             ('Q1', lambda x: x.quantile(0.25)),
                             ('Q3', lambda x: x.quantile(0.75))]).reset_index()

        for _, row in stats.iterrows():
            group = row[x_var]; model = row['Model']
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Mean', 'value': row['mean']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Median', 'value': row['median']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Count', 'value': row['count']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Q1', 'value': row['Q1']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Q3', 'value': row['Q3']})
    except Exception as e:
        logger.error(f"Error aggregating Panel A: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results

def aggregate_panel_b_core_exterior(df):
    """Aggregates data for Panel B (Core/Exterior Boxplot)."""
    panel_id = "Panel B"
    plot_type = "Boxplot"
    x_var = "Core_Exterior"
    y_var = "MAE"
    results = []
    try:
        mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
        plot_data = df[[x_var] + mae_cols].melt(id_vars=[x_var], var_name='Model_MAE', value_name=y_var)
        plot_data['Model'] = plot_data['Model_MAE'].str.replace('_MAE', '')
        plot_data = plot_data[plot_data['Model'].isin(MODELS_TO_PLOT)]
        category_order = ['Core', 'Exterior']
        plot_data = plot_data[plot_data[x_var].isin(category_order)]

        if plot_data.empty: logger.warning(f"{panel_id}: No data after filtering."); return results

        grouped = plot_data.groupby([x_var, 'Model'], observed=True)[y_var]
        stats = grouped.agg(['mean', 'median', 'count',
                             ('Q1', lambda x: x.quantile(0.25)),
                             ('Q3', lambda x: x.quantile(0.75))]).reset_index()

        for _, row in stats.iterrows():
            group = row[x_var]; model = row['Model']
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Mean', 'value': row['mean']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Median', 'value': row['median']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Count', 'value': row['count']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Q1', 'value': row['Q1']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Q3', 'value': row['Q3']})
    except Exception as e:
        logger.error(f"Error aggregating Panel B: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results

def aggregate_panel_c_position(df):
    """Aggregates data for Panel C (Position Lineplot+CI)."""
    panel_id = "Panel C"
    plot_type = "Lineplot+CI"
    x_var = "Position_Bin"
    y_var = "MAE"
    results = []
    try:
        mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
        if x_var not in df.columns: raise ValueError(f"Column {x_var} not found")

        _, pos_labels_order = create_position_bins(5)
        pos_labels_order_str = [str(lbl) for lbl in pos_labels_order]

        # Ensure Position_Bin is categorical with the right order
        # Use .loc to avoid SettingWithCopyWarning if df is a slice
        df_filtered = df.loc[df[x_var].isin(pos_labels_order_str)].copy()
        if df_filtered.empty: raise ValueError("No data matching expected position bins")
        df_filtered[x_var] = pd.Categorical(df_filtered[x_var], categories=pos_labels_order_str, ordered=True)

        # Group by the categorical bin
        grouped = df_filtered.groupby(x_var, observed=False) # Use observed=False to keep all bins

        z_score_95 = 1.96
        for model in MODELS_TO_PLOT:
            mae_col = f"{model}_MAE"
            if mae_col not in df_filtered.columns: continue

            # Aggregate stats for this model across bins
            stats = grouped[mae_col].agg(['mean', 'std', 'count']).reset_index()
            stats['sem'] = stats['std'] / np.sqrt(stats['count'])
            stats['ci_lower'] = stats['mean'] - z_score_95 * stats['sem']
            stats['ci_upper'] = stats['mean'] + z_score_95 * stats['sem']

            for _, row in stats.iterrows():
                group = row[x_var]
                results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Mean', 'value': row['mean']})
                results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'StdDev', 'value': row['std']})
                results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Count', 'value': row['count']})
                results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'SEM', 'value': row['sem']})
                results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': '95CI_Lower', 'value': row['ci_lower']})
                results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': '95CI_Upper', 'value': row['ci_upper']})
    except Exception as e:
        logger.error(f"Error aggregating Panel C: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results


def aggregate_panel_d_rsa(df):
    """Aggregates data for Panel D (RSA Lineplot)."""
    panel_id = "Panel D"
    plot_type = "Lineplot"
    x_var = "RSA_Quantile_Label"
    y_var = "MAE"
    results = []
    try:
        mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
        label_col = x_var
        if label_col not in df.columns or df[label_col].astype(str).str.contains('Error|Not Available', na=False).any():
            raise ValueError(f"Column {label_col} missing or has invalid labels.")

        plot_data_melted = df[[label_col] + mae_cols].melt(id_vars=[label_col], var_name='Model_MAE', value_name=y_var)
        plot_data_melted['Model'] = plot_data_melted['Model_MAE'].str.replace('_MAE', '')
        plot_data_melted = plot_data_melted[plot_data_melted['Model'].isin(MODELS_TO_PLOT)]
        plot_data_melted.dropna(subset=[y_var], inplace=True)

        # Determine quantile order based on numerical sort if possible
        try:
             unique_quantiles = plot_data_melted[label_col].unique()
             quantile_order = sorted(unique_quantiles, key=lambda x: float(str(x).split('-')[0]))
             results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': 'N/A', 'model': 'N/A', 'group': 'N/A', 'stat': 'X-axis Sort Order', 'value': quantile_order})
        except:
             quantile_order = sorted(plot_data_melted[label_col].unique()) # Fallback
             results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': 'N/A', 'model': 'N/A', 'group': 'N/A', 'stat': 'X-axis Sort Order', 'value': 'Default sort'})

        # Aggregate mean and count
        grouped = plot_data_melted.groupby([label_col, 'Model'], observed=False)[y_var]
        stats = grouped.agg(['mean', 'count']).reset_index()

        # Reorder stats according to determined quantile order for clarity in CSV
        stats[label_col] = pd.Categorical(stats[label_col], categories=quantile_order, ordered=True)
        stats = stats.sort_values(by=[label_col, 'Model'])

        for _, row in stats.iterrows():
            group = row[label_col]
            model = row['Model']
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Mean', 'value': row['mean']})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': group, 'stat': 'Count', 'value': row['count']})

    except Exception as e:
        logger.error(f"Error aggregating Panel D: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results

def aggregate_panel_e_bfactor(df):
    """Aggregates data for Panel E (BFactor Scatter+LOWESS)."""
    panel_id = "Panel E"
    plot_type = "Scatter+LOWESS"
    x_var = BFACTOR_COL
    y_var = "MAE"
    results = []
    try:
        mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
        if x_var not in df.columns: raise ValueError(f"Column {x_var} not found")

        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Note', 'value': "LOWESS trend with 95% CI shown"})
        for model in MODELS_TO_PLOT:
            mae_col = f"{model}_MAE"
            if mae_col not in df.columns: continue
            plot_data = df[[x_var, mae_col]].dropna()
            count = len(plot_data)
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': 'N/A', 'stat': 'Points Plotted', 'value': count})
            if count >= 2:
                try:
                    # Use Spearman as B-factor relation might not be linear
                    corr, pval = stats.spearmanr(plot_data[x_var], plot_data[mae_col])
                    results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': 'Correlation', 'stat': 'SpearmanRho', 'value': corr})
                    results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': 'Correlation', 'stat': 'P-value', 'value': pval})
                except ValueError:
                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': 'Correlation', 'stat': 'SpearmanRho', 'value': 'Calculation Error'})

    except Exception as e:
        logger.error(f"Error aggregating Panel E: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results

def aggregate_panel_f_ramachandran(df):
    """Aggregates data for Panel F (Ramachandran Heatmap)."""
    # Note: Gaussian smoothing is visual; stats are based on raw bins
    panel_id = "Panel F"
    plot_type = "Heatmap"
    x_var = PHI_COL
    y_var = PSI_COL
    z_var = f"Mean {PRIMARY_MODEL_NAME} MAE"
    results = []
    try:
        primary_mae_col = f"{PRIMARY_MODEL_NAME}_MAE"
        required_cols = [x_var, y_var, primary_mae_col]
        if not all(col in df.columns for col in required_cols): raise ValueError("Missing Phi/Psi/MAE columns")

        plot_data = df[required_cols].dropna()
        n_points = len(plot_data)
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': PRIMARY_MODEL_NAME, 'group': 'N/A', 'stat': 'Total Points Plotted', 'value': n_points})
        if n_points < 100: logger.warning("Panel F: Insufficient data points (<100)."); return results

        n_bins = 72 # Match plotting if possible
        phi_bins = np.linspace(-180, 180, n_bins + 1); psi_bins = np.linspace(-180, 180, n_bins + 1)
        bin_centers_phi = (phi_bins[:-1] + phi_bins[1:]) / 2; bin_centers_psi = (psi_bins[:-1] + psi_bins[1:]) / 2

        mean_mae, _, _, binnumbers = stats.binned_statistic_2d(plot_data[x_var], plot_data[y_var], plot_data[primary_mae_col], statistic='mean', bins=[phi_bins, psi_bins])
        counts, _, _, _ = stats.binned_statistic_2d(plot_data[x_var], plot_data[y_var], None, statistic='count', bins=[phi_bins, psi_bins])

        # Add overall mean MAE in binned data
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': PRIMARY_MODEL_NAME, 'group': 'All Bins', 'stat': 'Mean MAE (Binned)', 'value': np.nanmean(mean_mae)})

        # Add stats for specific regions
        regions = {'α-Helix': {'phi': (-80, -40), 'psi': (-60, -20)}, 'β-Sheet': {'phi': (-150, -90), 'psi': (120, 160)}, 'L-α': {'phi': (40, 80), 'psi': (20, 80)}}
        for label, bounds in regions.items():
            # Find points within the region boundaries
            region_mask = (plot_data[x_var] >= bounds['phi'][0]) & (plot_data[x_var] <= bounds['phi'][1]) & \
                          (plot_data[y_var] >= bounds['psi'][0]) & (plot_data[y_var] <= bounds['psi'][1])
            region_data = plot_data.loc[region_mask, primary_mae_col]
            count_region = region_mask.sum()
            mean_mae_region = region_data.mean() if count_region > 0 else np.nan

            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': PRIMARY_MODEL_NAME, 'group': label, 'stat': 'Mean MAE (Raw Points)', 'value': mean_mae_region})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': label, 'stat': 'Count (Raw Points)', 'value': count_region})

    except Exception as e:
        logger.error(f"Error aggregating Panel F: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results

def aggregate_panel_g_amino_acid(df):
    """Aggregates data for Panel G (Amino Acid Barplot+CI)."""
    panel_id = "Panel G"
    plot_type = "Barplot+CI"
    x_var = RESNAME_COL
    y_var = "MAE"
    results = []
    # --- Amino Acid Full Name Mapping ---
    AMINO_ACID_FULL_NAMES = { 'ALA': 'Alanine', 'ARG': 'Arginine', 'ASN': 'Asparagine', 'ASP': 'Aspartic Acid', 'CYS': 'Cysteine', 'GLN': 'Glutamine', 'GLU': 'Glutamic Acid', 'GLY': 'Glycine', 'HIS': 'Histidine', 'ILE': 'Isoleucine', 'LEU': 'Leucine', 'LYS': 'Lysine', 'MET': 'Methionine', 'PHE': 'Phenylalanine', 'PRO': 'Proline', 'SER': 'Serine', 'THR': 'Threonine', 'TRP': 'Tryptophan', 'TYR': 'Tyrosine', 'VAL': 'Valine' }
    try:
        mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
        if x_var not in df.columns: raise ValueError(f"Column {x_var} not found")

        plot_data_melted = df[[x_var] + mae_cols].melt(id_vars=x_var, var_name='Model_MAE', value_name=y_var)
        plot_data_melted['Model'] = plot_data_melted['Model_MAE'].str.replace('_MAE', '')

        standard_aas = list(AMINO_ACID_FULL_NAMES.keys())
        plot_data_melted = plot_data_melted[plot_data_melted[x_var].isin(standard_aas) & plot_data_melted['Model'].isin(MODELS_TO_PLOT)].dropna(subset=[y_var])
        if plot_data_melted.empty: raise ValueError("No valid data remaining after filtering.")

        # Calculate detailed stats including SEM
        stats_summary = plot_data_melted.groupby([x_var, 'Model'], observed=True)[y_var].agg(['mean', 'std', 'count', 'sem']).reset_index()

        # Determine sort order based on DeepFlex mean
        deepflex_stats = stats_summary[stats_summary['Model'] == PRIMARY_MODEL_NAME].copy()
        if not deepflex_stats.empty:
            aa_order = deepflex_stats.sort_values(by='mean', ascending=True)[x_var].tolist()
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': 'N/A', 'model': 'N/A', 'group': 'N/A', 'stat': 'X-axis Sort Order', 'value': aa_order})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': 'N/A', 'model': PRIMARY_MODEL_NAME, 'group': 'N/A', 'stat': 'Top 3 (Lowest MAE)', 'value': deepflex_stats.nsmallest(3, 'mean').set_index(x_var)['mean'].to_dict()})
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': 'N/A', 'model': PRIMARY_MODEL_NAME, 'group': 'N/A', 'stat': 'Bottom 3 (Highest MAE)', 'value': deepflex_stats.nlargest(3, 'mean').set_index(x_var)['mean'].to_dict()})
        else:
            aa_order = sorted(stats_summary[x_var].unique()) # Fallback order
            results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': 'N/A', 'model': 'N/A', 'group': 'N/A', 'stat': 'X-axis Sort Order', 'value': 'Default alphabetical (sorting failed)'})


        # Add detailed stats per AA and Model
        z_score_95 = 1.96
        # Iterate through sorted order for consistency, but use full stats_summary for data
        stats_summary_indexed = stats_summary.set_index([x_var, 'Model']) # Easier lookup
        for aa in aa_order:
            for model in MODELS_TO_PLOT:
                 try:
                     row = stats_summary_indexed.loc[(aa, model)]
                     mean_val = row['mean']; sem_val = row['sem']; count_val = row['count']; std_val = row['std']
                     ci_lower = mean_val - z_score_95 * sem_val if pd.notna(mean_val) and pd.notna(sem_val) else np.nan
                     ci_upper = mean_val + z_score_95 * sem_val if pd.notna(mean_val) and pd.notna(sem_val) else np.nan

                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': aa, 'stat': 'Mean', 'value': mean_val})
                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': aa, 'stat': 'StdDev', 'value': std_val})
                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': aa, 'stat': 'Count', 'value': count_val})
                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': aa, 'stat': 'SEM', 'value': sem_val})
                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': aa, 'stat': '95CI_Lower', 'value': ci_lower})
                     results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': model, 'group': aa, 'stat': '95CI_Upper', 'value': ci_upper})
                 except KeyError:
                     logger.warning(f"Panel G: No data found for AA '{aa}' and Model '{model}'.")
                     continue # Skip if combo doesn't exist

    except Exception as e:
        logger.error(f"Error aggregating Panel G: {e}")
        results.append({'panel': panel_id, 'plot': plot_type, 'x': x_var, 'y': y_var, 'model': 'N/A', 'group': 'N/A', 'stat': 'Error', 'value': str(e)})
    return results


# --- Detailed Analysis CSV Generation ---
def write_detailed_analysis_csv(aggregated_results, overall_stats, output_path, logger):
    """Writes the collected aggregated analysis data to a structured CSV file."""
    logger.info(f"Generating Detailed Analysis Summary CSV: {output_path}")

    header = ["Panel_ID", "Plot_Type", "X_Axis_Variable", "Y_Axis_Variable", "Model", "Group_Name", "Statistic", "Value"]
    rows_to_write = [header]

    # --- Helper to add rows, ensuring structure ---
    def add_row(data_dict):
        # Ensure all keys exist, defaulting to N/A or Error
        row_data = [
            data_dict.get('panel', 'N/A'),
            data_dict.get('plot', 'N/A'),
            data_dict.get('x', 'N/A'),
            data_dict.get('y', 'N/A'),
            format_value_for_csv(data_dict.get('model', 'N/A')),
            format_value_for_csv(data_dict.get('group', 'N/A')),
            format_value_for_csv(data_dict.get('stat', 'N/A')),
            format_value_for_csv(data_dict.get('value', 'Error'))
        ]
        rows_to_write.append(row_data)

    # Write Overall Stats First
    add_row({'panel': "Overall", 'plot': "Dataset Info", 'x': "N/A", 'y': "N/A", 'model': "N/A", 'group': "N/A", 'stat': "Total Rows Loaded", 'value': overall_stats.get('Total Rows Loaded')})
    add_row({'panel': "Overall", 'plot': "Dataset Info", 'x': "N/A", 'y': "N/A", 'model': "N/A", 'group': "N/A", 'stat': "Rows after MAE NaN Drop", 'value': overall_stats.get('Rows after MAE NaN Drop')})
    add_row({'panel': "Overall", 'plot': "Dataset Info", 'x': "N/A", 'y': "N/A", 'model': "N/A", 'group': "N/A", 'stat': "Rows after Final Processing", 'value': overall_stats.get('Rows after processing')})
    for model in MODELS_TO_PLOT:
        add_row({'panel': "Overall", 'plot': "Performance Metric", 'x': "Across All Data", 'y': "MAE", 'model': model, 'group': "N/A", 'stat': "Mean", 'value': overall_stats.get(f'{model} Overall Mean MAE')})
        add_row({'panel': "Overall", 'plot': "Performance Metric", 'x': "Across All Data", 'y': "MAE", 'model': model, 'group': "N/A", 'stat': "Median", 'value': overall_stats.get(f'{model} Overall Median MAE')})
    rows_to_write.append([]) # Spacer

    # Write Panel-Specific Aggregated Results
    for panel_results in aggregated_results:
        if panel_results: # Check if list is not empty
            for row_dict in panel_results:
                 if isinstance(row_dict, dict): # Ensure it's a dictionary before adding
                     add_row(row_dict)
                 else:
                     logger.warning(f"Skipping non-dictionary item found in aggregated results: {row_dict}")
            rows_to_write.append([]) # Spacer between panels

    # --- Write to CSV ---
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Filter out empty spacer rows just before writing
            writer.writerows([row for row in rows_to_write if row])
        logger.info(f"Successfully wrote DETAILED analysis summary CSV to: {output_path}")
    except IOError as e:
        logger.error(f"Failed to write analysis CSV: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV writing: {e}", exc_info=True)


# --- Main Execution ---
def main():
    """Main function to generate the detailed analysis CSV."""
    global logger # Ensure logger is globally accessible if needed by helpers
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    logger = setup_logging(OUTPUT_DIR)
    logger.info(f"Output directory for CSV: {OUTPUT_DIR}")

    # Load and prepare data
    df, overall_stats = load_and_prepare_data_for_stats(INPUT_CSV, logger)
    if df is None:
        logger.critical("Failed to load or prepare data. Exiting.")
        return

    # Aggregate data for each panel
    logger.info("Aggregating data for Panel A...")
    panel_a_results = aggregate_panel_a_ss(df)
    logger.info("Aggregating data for Panel B...")
    panel_b_results = aggregate_panel_b_core_exterior(df)
    logger.info("Aggregating data for Panel C...")
    panel_c_results = aggregate_panel_c_position(df)
    logger.info("Aggregating data for Panel D...")
    panel_d_results = aggregate_panel_d_rsa(df)
    logger.info("Aggregating data for Panel E...")
    panel_e_results = aggregate_panel_e_bfactor(df)
    logger.info("Aggregating data for Panel F...")
    panel_f_results = aggregate_panel_f_ramachandran(df)
    logger.info("Aggregating data for Panel G...")
    panel_g_results = aggregate_panel_g_amino_acid(df)

    # Combine results
    all_aggregated_results = [
        panel_a_results, panel_b_results, panel_c_results,
        panel_d_results, panel_e_results, panel_f_results,
        panel_g_results
    ]

    # Write the detailed CSV
    output_csv_path = OUTPUT_DIR / OUTPUT_CSV_NAME
    write_detailed_analysis_csv(all_aggregated_results, overall_stats, output_csv_path, logger)

    logger.info(f"--- Detailed Analysis CSV Generation Finished ({OUTPUT_CSV_NAME}) ---")

# --- Amino Acid Full Name Mapping ---
# Defined globally for potential use in helper functions if needed later
AMINO_ACID_FULL_NAMES = {
    'ALA': 'Alanine', 'ARG': 'Arginine', 'ASN': 'Asparagine', 'ASP': 'Aspartic Acid',
    'CYS': 'Cysteine', 'GLN': 'Glutamine', 'GLU': 'Glutamic Acid', 'GLY': 'Glycine',
    'HIS': 'Histidine', 'ILE': 'Isoleucine', 'LEU': 'Leucine', 'LYS': 'Lysine',
    'MET': 'Methionine', 'PHE': 'Phenylalanine', 'PRO': 'Proline', 'SER': 'Serine',
    'THR': 'Threonine', 'TRP': 'Tryptophan', 'TYR': 'Tyrosine', 'VAL': 'Valine'
}

if __name__ == "__main__":
    # Setup logging here if not done globally
    # logger = setup_logging(OUTPUT_DIR) # Ensure logger is defined
    main()