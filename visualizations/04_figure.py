#!/usr/bin/env python3
"""
Figure 4 Generation Script for DeepFlex Manuscript (Final Revision v3 - Fixes)

Generates publication-quality figures visualizing model robustness and an analysis summary CSV.
Output: PNG files (figure4_a_... etc.) and figure4_analysis.csv saved to 'figure4_outputs'.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec # No longer needed for B-factor plot
import seaborn as sns
from scipy import stats
# from scipy.ndimage import gaussian_filter # Keep if needed for other plots
import logging
import warnings
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import csv
from scipy import stats
from scipy.ndimage import gaussian_filter

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_CSV = "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "figure4_outputs"

MODEL_MAP = {
    'DeepFlex': 'Attention_ESM_rmsf',
    'ESM-Only': 'ESM_Only_rmsf',
    'VoxelFlex-3D': 'voxel_rmsf',
}
MODELS_TO_PLOT = ['DeepFlex', 'ESM-Only', 'VoxelFlex-3D']
MODEL_COLORS = {
    'DeepFlex': '#677FB0',      # Specified Purple/Blue
    'ESM-Only': '#CC7980',      # Specified Teal/Green
    'VoxelFlex-3D': '#F2D8AC',  # Specified Red/Pink
}
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

# Global dictionary to store analysis data for CSV
analysis_data = {}

# --- Plotting Style Setup ---
def setup_plotting_style():
    """Sets global Matplotlib parameters for publication quality."""
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 12,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5, 'legend.title_fontsize': 10,
        'figure.titlesize': 14, 'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1, 'axes.labelpad': 6,
        'xtick.major.pad': 4, 'ytick.major.pad': 4,
        'axes.grid': False, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.facecolor': '#f0f0f0' # Default background color
    })
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
    warnings.filterwarnings("ignore", message="Glyph.*missing from current font")
    warnings.filterwarnings("ignore", category=FutureWarning)

# --- Logging Setup ---
def setup_logging(log_dir):
    """Configures logging to file and console."""
    log_file = log_dir / "figure4_generation.log"
    global logger
    logging.basicConfig(
        level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# --- Helper Functions ---
def calculate_mae(df, model_col, target_col):
    if model_col not in df.columns or target_col not in df.columns:
        try: logger.warning(f"Cannot calculate MAE: Missing {model_col} or {target_col}")
        except NameError: print(f"WARN: Cannot calculate MAE: Missing {model_col} or {target_col}")
        return pd.Series(np.nan, index=df.index)
    pred = pd.to_numeric(df[model_col], errors='coerce')
    target = pd.to_numeric(df[target_col], errors='coerce')
    return np.abs(pred - target)

def map_secondary_structure(dssp_code):
    if pd.isna(dssp_code): return 'Other'
    code = str(dssp_code).upper()
    if code in ('G', 'H', 'I'): return 'α-Helix'
    if code in ('B', 'E'): return 'β-Sheet'
    return 'Loop/Other'

def map_core_exterior(rsa_value, threshold=0.2):
    if pd.isna(rsa_value): return 'Unknown'
    try:
        return 'Core' if float(rsa_value) <= threshold else 'Exterior'
    except (ValueError, TypeError):
        return 'Unknown'

def create_position_bins(n_bins=5):
    boundaries = np.linspace(0, 1, n_bins + 1)
    labels = []
    for i in range(n_bins):
        if i == 0: label = f"N-term ({boundaries[i]:.1f}-{boundaries[i+1]:.1f})"
        elif i == n_bins - 1: label = f"C-term ({boundaries[i]:.1f}-{boundaries[i+1]:.1f})"
        else: label = f"Mid ({boundaries[i]:.1f}-{boundaries[i+1]:.1f})"
        labels.append(label)
    return boundaries, labels



# --- Data Loading and Preparation ---
def load_and_prepare_data(csv_path, logger):
    """Loads CSV, prepares data, fixes position bins and quantile labels."""
    global analysis_data
    logger.info(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} rows.")
        analysis_data['Overall'] = {'Total Rows Loaded': len(df)}
    except FileNotFoundError: logger.critical(f"Input CSV not found: {csv_path}"); return None
    except Exception as e: logger.critical(f"Error loading CSV: {e}", exc_info=True); return None

    required_data_cols = [TARGET_COL, DSSP_COL, RSA_COL, NORM_RESID_COL,
                          BFACTOR_COL, PHI_COL, PSI_COL, RESNAME_COL]
    required_model_cols = [MODEL_MAP[m] for m in MODELS_TO_PLOT]
    all_required = required_data_cols + required_model_cols
    missing = [col for col in all_required if col not in df.columns]
    if missing: logger.critical(f"Input CSV is missing required columns: {missing}"); return None

    logger.info("Converting column types...")
    numeric_cols = [TARGET_COL, RSA_COL, NORM_RESID_COL, BFACTOR_COL, PHI_COL, PSI_COL] + required_model_cols
    for col in numeric_cols:
        orig_nan = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        new_nan = df[col].isnull().sum()
        if new_nan > orig_nan: logger.warning(f"Coerced {new_nan - orig_nan} NaNs in column '{col}'.")

    logger.info("Calculating MAE...")
    for model_name in MODELS_TO_PLOT:
        mae_col = f"{model_name}_MAE"
        df[mae_col] = calculate_mae(df, MODEL_MAP[model_name], TARGET_COL)
        analysis_data.setdefault('Overall', {})[f'{model_name} Overall Mean MAE'] = df[mae_col].mean()
        analysis_data.setdefault('Overall', {})[f'{model_name} Overall Median MAE'] = df[mae_col].median()

    mae_cols_to_check = [f"{m}_MAE" for m in MODELS_TO_PLOT]
    rows_before_drop = len(df)
    df.dropna(subset=mae_cols_to_check, inplace=True)
    rows_after_drop = len(df)
    if rows_after_drop < rows_before_drop:
        logger.info(f"Dropped {rows_before_drop - rows_after_drop:,} rows due to NaN MAE values.")
    analysis_data['Overall']['Rows after MAE calculation (non-NaN)'] = rows_after_drop
    logger.info(f"{rows_after_drop:,} rows remaining after dropping NaN MAE values.")


    logger.info("Mapping categorical features...")
    df['SS_Category'] = df[DSSP_COL].apply(map_secondary_structure)
    df['Core_Exterior'] = df[RSA_COL].apply(map_core_exterior)

    logger.info("Binning continuous features...")

    # --- Position Bin ---
    nan_norm_resid = df[NORM_RESID_COL].isnull().sum()
    if nan_norm_resid > 0:
        logger.warning(f"{nan_norm_resid} NaN values found in '{NORM_RESID_COL}'. Excluding.")
        df.dropna(subset=[NORM_RESID_COL], inplace=True)

    if NORM_RESID_COL in df.columns and not df[NORM_RESID_COL].empty:
        logger.info(f"Clipping '{NORM_RESID_COL}' to [0, 1] range before binning.")
        df[NORM_RESID_COL] = df[NORM_RESID_COL].clip(0, 1)

        pos_boundaries, pos_labels = create_position_bins(5)
        pos_labels_str = [str(lbl) for lbl in pos_labels]
        df['Position_Bin'] = pd.cut(df[NORM_RESID_COL], bins=pos_boundaries, labels=pos_labels_str, right=False, include_lowest=True)
        df['Position_Bin'] = pd.Categorical(df['Position_Bin'], categories=pos_labels_str, ordered=True)

        nan_pos_bin = df['Position_Bin'].isnull().sum()
        if nan_pos_bin > 0:
            logger.warning(f"{nan_pos_bin} NULL values in 'Position_Bin' AFTER cutting. Adding 'Unknown'.")
            if 'Unknown' not in df['Position_Bin'].cat.categories:
                 df['Position_Bin'] = df['Position_Bin'].cat.add_categories('Unknown')
            df['Position_Bin'].fillna('Unknown', inplace=True)
    else:
        logger.warning(f"'{NORM_RESID_COL}' column is empty or missing. Skipping position binning.")
        df['Position_Bin'] = 'Not Available'
    # -----------------------

    # --- Robust Quantile Labeling ---
    n_quantiles = 5
    for col_name, label_prefix in [(RSA_COL, "RSA"), (BFACTOR_COL, "BFactor")]:
        quantile_col = f"{label_prefix}_Quantile"
        label_col = f"{label_prefix}_Quantile_Label"
        df[label_col] = f'Error_{label_prefix}'

        if col_name not in df.columns:
            logger.error(f"Column '{col_name}' not found. Skipping {label_prefix} binning.")
            df[label_col] = 'Not Available'
            continue

        try:
            valid_data_series = df[col_name].dropna()
            if len(valid_data_series) < n_quantiles: raise ValueError("Not enough non-NaN data points.")
            df[quantile_col] = pd.qcut(df[col_name].rank(method='first'), n_quantiles, labels=False, duplicates='drop')
            q_boundaries = valid_data_series.quantile(np.linspace(0, 1, n_quantiles + 1)).tolist()

            if len(set(round(b, 5) for b in q_boundaries)) > n_quantiles:
                 labels = [f"{q_boundaries[i]:.2f}-{q_boundaries[i+1]:.2f}" for i in range(n_quantiles)]
                 quantile_labels_series = pd.qcut(df[col_name].rank(method='first'), n_quantiles, labels=labels, duplicates='drop')
                 # Use .loc[] for safer assignment based on index
                 df.loc[quantile_labels_series.index, label_col] = quantile_labels_series
                 df[label_col] = pd.Categorical(df[label_col], categories=labels, ordered=True)
                 logger.info(f"Successfully created string labels for {label_col}.")
            else: raise ValueError("Not enough unique quantile boundaries.")

        except Exception as e:
            logger.warning(f"Could not generate string labels for {label_col}: {e}. Using numeric quantiles if possible.")
            if quantile_col in df.columns and df[quantile_col].notna().any():
                # Assign numeric labels safely using .loc
                valid_idx = df[quantile_col].notna()
                df.loc[valid_idx, label_col] = df.loc[valid_idx, quantile_col].astype('Int64').astype(str)
                df[label_col] = df[label_col].replace('<NA>', 'Unknown')
            else: df[label_col] = 'Unknown'
        finally:
            if label_col in df.columns and df[label_col].isnull().any():
                 if pd.api.types.is_categorical_dtype(df[label_col]):
                     if 'Unknown' not in df[label_col].cat.categories:
                          logger.info(f"Adding 'Unknown' category to {label_col} before fillna.")
                          df[label_col] = df[label_col].cat.add_categories('Unknown')
                 elif not isinstance(df[label_col].dtype, object):
                      df[label_col] = df[label_col].astype(object)
                 df[label_col].fillna('Unknown', inplace=True)
            elif label_col not in df.columns: df[label_col] = 'Unknown'
    # ---------------------------------

    # Final NaN fill/categorization for primary categoricals
    for col in ['SS_Category', 'Core_Exterior']:
        if col in df.columns:
            if not pd.api.types.is_categorical_dtype(df[col]):
                 known_cats = df[col].dropna().unique().tolist()
                 all_cats = known_cats + ['Unknown'] if 'Unknown' not in known_cats else known_cats
                 df[col] = pd.Categorical(df[col], categories=all_cats)
            elif 'Unknown' not in df[col].cat.categories:
                 df[col] = df[col].cat.add_categories('Unknown')
            df[col].fillna('Unknown', inplace=True)
        else: df[col] = 'Not Available'

    analysis_data['Overall']['Rows after processing'] = len(df)
    logger.info("Data preparation finished.")
    return df


# --- Plotting Functions ---

# Plot A: Secondary Structure (Box Plots) - No Fliers, Y-axis Zoomed
# Plot A: Secondary Structure (Box Plots - Reverted to previous Boxplot Style)
def plot_ss_performance(df, output_path, logger):
    """ Generates box plots of MAE per SS category, styled like reference."""
    global analysis_data
    panel_id = 'Panel A: Secondary Structure'
    logger.info(f"Generating {panel_id} (Box Plot)...")
    analysis_data[panel_id] = {}

    mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
    # Ensure SS_Category exists and handle potential issues
    if 'SS_Category' not in df.columns:
        logger.error("Panel A: SS_Category column missing. Skipping.")
        analysis_data[panel_id]['Error'] = "SS_Category column missing."
        return

    # Prepare data, ensure Model names are clean
    plot_data = df[['SS_Category'] + mae_cols].melt(id_vars=['SS_Category'], var_name='Model', value_name='MAE')
    plot_data['Model'] = plot_data['Model'].str.replace('_MAE', '')
    plot_data = plot_data[plot_data['Model'].isin(MODELS_TO_PLOT)] # Filter relevant models

    category_order = ['α-Helix', 'β-Sheet', 'Loop/Other']
    # Filter data to include only expected categories before setting categorical type
    plot_data = plot_data[plot_data['SS_Category'].isin(category_order)]
    if plot_data.empty:
        logger.error("Panel A: No data remaining after filtering for standard SS categories. Skipping.")
        analysis_data[panel_id]['Error'] = "No data for standard SS categories."
        return
    plot_data['SS_Category'] = pd.Categorical(plot_data['SS_Category'], categories=category_order, ordered=True)

    # Calculate stats for analysis CSV
    stats_summary = plot_data.groupby(['SS_Category', 'Model'], observed=True)['MAE'].agg(['mean', 'median', 'count']).unstack()
    analysis_data[panel_id]['Stats (Mean/Median/Count by SS, Model)'] = stats_summary.to_dict()

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    # Use the boxplot function
    sns.boxplot(
        data=plot_data, x='SS_Category', y='MAE', hue='Model',
        order=category_order, palette=MODEL_COLORS, hue_order=MODELS_TO_PLOT,
        linewidth=1.0,
        showfliers=False, # Hide individual outlier points
        width=0.7,        # Adjusted width slightly
        dodge=True,
        gap=0.1,          # Add small gap between dodged boxes
        ax=ax
    )

    # --- Y-axis zoom (Optional but often good for boxplots) ---
    if not plot_data['MAE'].empty:
        # Consider zooming based on IQR or percentiles if needed
        # Example: Zoom based on 1st and 99th percentile
        lower_limit = plot_data['MAE'].quantile(0.01)
        upper_limit = plot_data['MAE'].quantile(0.99)
        # Add buffer, ensure bottom is >= 0
        y_buffer = (upper_limit - lower_limit) * 0.05
        y_bottom = max(0, lower_limit - y_buffer)
        y_top = upper_limit + y_buffer
        # Apply limits if they provide a reasonable view
        if y_top > y_bottom:
             ax.set_ylim(bottom=y_bottom, top=y_top)
             logger.info(f"Panel A: Setting Y-axis limits for focus: [{y_bottom:.3f}, {y_top:.3f}]")
             analysis_data[panel_id]['Y-axis Limits'] = {'Bottom': y_bottom, 'Top': y_top}
        else:
             ax.set_ylim(bottom=0) # Fallback if calculation fails
             logger.warning("Panel A: Could not determine appropriate Y-axis zoom limits.")
    else:
        logger.warning("Panel A: No MAE data available for Y-axis limits.")
        ax.set_ylim(bottom=0)
    # -----------------------------------------------------

    ax.set_xlabel("Secondary Structure", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
    ax.set_title("Performance by Secondary Structure", fontsize=12, pad=15)
    ax.legend(title='Model', loc='upper left', frameon=False, fontsize=8.5) # Legend upper left
    ax.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally:
        plt.close()

# Plot B: Core/Exterior (Box Plot - No Fliers, Y-axis Zoomed)
# Plot B: Core/Exterior (Box Plot - Reverted to previous Boxplot Style)
def plot_core_exterior_performance(df, output_path, logger):
    """ Generates box plots of MAE per Core/Exterior category, styled like reference."""
    global analysis_data
    panel_id = 'Panel B: Core/Exterior'
    logger.info(f"Generating {panel_id} (Box Plot)...")
    analysis_data[panel_id] = {}

    mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
    # Ensure Core_Exterior exists
    if 'Core_Exterior' not in df.columns:
        logger.error("Panel B: Core_Exterior column missing. Skipping.")
        analysis_data[panel_id]['Error'] = "Core_Exterior column missing."
        return

    # Prepare data
    plot_data = df[['Core_Exterior'] + mae_cols].melt(id_vars=['Core_Exterior'], var_name='Model', value_name='MAE')
    plot_data['Model'] = plot_data['Model'].str.replace('_MAE', '')
    plot_data = plot_data[plot_data['Model'].isin(MODELS_TO_PLOT)] # Filter models

    category_order = ['Core', 'Exterior']
    # Filter data, handle 'Unknown' or other unexpected values
    plot_data = plot_data[plot_data['Core_Exterior'].isin(category_order)]
    if plot_data.empty:
        logger.error("Panel B: No data remaining after filtering for Core/Exterior categories. Skipping.")
        analysis_data[panel_id]['Error'] = "No data for Core/Exterior categories."
        return
    plot_data['Core_Exterior'] = pd.Categorical(plot_data['Core_Exterior'], categories=category_order, ordered=True)

    # Calculate stats for analysis CSV
    stats_summary = plot_data.groupby(['Core_Exterior', 'Model'], observed=True)['MAE'].agg(['mean', 'median', 'count']).unstack()
    analysis_data[panel_id]['Stats (Mean/Median/Count by Location, Model)'] = stats_summary.to_dict()

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    # Use boxplot function
    sns.boxplot(
        data=plot_data, x='Core_Exterior', y='MAE', hue='Model',
        order=category_order, palette=MODEL_COLORS, hue_order=MODELS_TO_PLOT,
        linewidth=1.0,
        showfliers=False, # Hide individual outliers
        width=0.6,
        dodge=True,
        gap=0.1,
        ax=ax
    )

    # --- Y-axis zoom (Optional) ---
    if not plot_data['MAE'].empty:
        lower_limit = plot_data['MAE'].quantile(0.01)
        upper_limit = plot_data['MAE'].quantile(0.99)
        y_buffer = (upper_limit - lower_limit) * 0.05
        y_bottom = max(0, lower_limit - y_buffer)
        y_top = upper_limit + y_buffer
        if y_top > y_bottom:
            ax.set_ylim(bottom=y_bottom, top=y_top)
            logger.info(f"Panel B: Setting Y-axis limits for focus: [{y_bottom:.3f}, {y_top:.3f}]")
            analysis_data[panel_id]['Y-axis Limits'] = {'Bottom': y_bottom, 'Top': y_top}
        else:
            ax.set_ylim(bottom=0)
            logger.warning("Panel B: Could not determine appropriate Y-axis zoom limits.")
    else:
        logger.warning("Panel B: No MAE data available for Y-axis limits.")
        ax.set_ylim(bottom=0)
    # --------------------------------

    ax.set_xlabel("Residue Location (RSA Threshold = 0.2)", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
    ax.set_title("Performance by Core/Exterior Location", fontsize=12, pad=15)
    ax.legend(title='Model', loc='upper left', frameon=False, fontsize=8.5) # Legend upper left
    ax.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally:
        plt.close()

# Plot C: Sequence Position (Line Plot w/ CI) - Reverted Y-axis, Legend inside
def plot_position_performance(df, output_path, logger):
    """
    Generates a line plot of Mean MAE ± 95% CI along sequence position bins,
    with outlined lines, grey background, white grid lines, and increased title spacing.
    """
    global analysis_data
    panel_id = 'Panel C: Sequence Position'
    logger.info(f"Generating {panel_id} (Mean ± 95% CI, Outlined Lines, Grey BG, White Grid)...")
    analysis_data[panel_id] = {}

    mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]

    # Select necessary columns
    required_cols_for_plot = ['Position_Bin'] + mae_cols
    if not all(col in df.columns for col in required_cols_for_plot):
        missing = [c for c in required_cols_for_plot if c not in df.columns]
        logger.error(f"Panel C: Missing required columns: {missing}. Skipping.")
        analysis_data[panel_id]['Error'] = f"Missing required columns: {missing}"
        return
    if 'Position_Bin' not in df.columns or df['Position_Bin'].isnull().all():
        logger.error("Panel C: 'Position_Bin' column is missing or empty. Skipping.")
        analysis_data[panel_id]['Error'] = "'Position_Bin' column missing or empty."
        return

    df_plot = df[required_cols_for_plot].copy()

    # --- Define the desired order explicitly ---
    _, pos_labels_order = create_position_bins(5)
    pos_labels_order_str = [str(lbl) for lbl in pos_labels_order]

    # --- Filter data to ONLY include the expected categories ---
    df_plot['Position_Bin'] = df_plot['Position_Bin'].astype(str)
    initial_rows = len(df_plot)
    df_plot = df_plot[df_plot['Position_Bin'].isin(pos_labels_order_str)]
    filtered_rows = len(df_plot)
    if filtered_rows < initial_rows:
        logger.warning(f"Panel C: Filtered out {initial_rows - filtered_rows} rows with unexpected Position_Bin values.")
    if df_plot.empty:
        logger.error("Panel C: No data left after filtering for expected Position_Bin categories. Skipping.")
        analysis_data[panel_id]['Error'] = 'No data matching expected position bins.'
        return

    # --- Enforce the categorical type with the correct order ---
    df_plot['Position_Bin'] = pd.Categorical(df_plot['Position_Bin'], categories=pos_labels_order_str, ordered=True)

    # --- Group and Aggregate (Mean, Std, Count) ---
    plot_data_agg_grouped = df_plot.groupby('Position_Bin', observed=False)[mae_cols].agg(['mean', 'std', 'count'])
    plot_data_plot = plot_data_agg_grouped # Use this directly

    # --- Store analysis data ---
    analysis_stats_cols = [(f"{m}_MAE", stat) for m in MODELS_TO_PLOT for stat in ['mean', 'count']]
    analysis_stats_cols_present = [col for col in analysis_stats_cols if col in plot_data_plot.columns]
    if analysis_stats_cols_present:
        analysis_stats_df = plot_data_plot[analysis_stats_cols_present]
        analysis_data[panel_id]['Stats (Mean/Count by Position Bin, Model)'] = analysis_stats_df.unstack().to_dict()
    else:
        logger.warning("Panel C: No valid aggregated stats columns found for analysis_data.")

    # --- Prepare labels and positions for plotting ---
    x_labels = plot_data_plot.index.astype(str).tolist()
    x_pos = np.arange(len(x_labels))

    # --- Plotting ---
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0') # SET GREY BACKGROUND

    # --- Define Z-score for 95% CI ---
    z_score_95 = 1.96

    for model_name in MODELS_TO_PLOT:
        model_display_name = model_name
        mae_col = f"{model_name}_MAE"
        mean_col = (mae_col, 'mean'); std_col = (mae_col, 'std'); count_col = (mae_col, 'count')

        if mean_col in plot_data_plot.columns and plot_data_plot[mean_col].notna().any():
            means = plot_data_plot[mean_col].values
            stds = plot_data_plot.get(std_col)
            counts = plot_data_plot.get(count_col)

            sem_val = np.zeros_like(means)
            if stds is not None and counts is not None:
                stds_val = stds.values
                counts_val = counts.values
                valid_mask = (counts_val > 1) & (~np.isnan(stds_val)) & (~np.isnan(counts_val))
                sem_val[valid_mask] = stds_val[valid_mask] / np.sqrt(counts_val[valid_mask])
            else:
                logger.warning(f"Std or Count column missing for {model_name}. Cannot calculate CI.")

            ci_lower = means - z_score_95 * sem_val
            ci_upper = means + z_score_95 * sem_val

            color = MODEL_COLORS.get(model_name)
            if color is None: color = 'grey'; logger.warning(f"Color not found for {model_name}.")

            # 1. Plot Outline
            ax.plot(x_pos, means, marker='', linestyle='-', label='',
                    color='black', linewidth=2.5, zorder=2)

            # 2. Plot Main Line
            ax.plot(x_pos, means, marker='o', linestyle='-', label=model_display_name,
                    color=color, markersize=7, linewidth=2.0, zorder=3)

            # 3. Plot Shaded Region for 95% CI
            ax.fill_between(x_pos, ci_lower, ci_upper,
                            color=color, alpha=0.25)
        else:
            logger.warning(f"No valid data for plotting mean MAE for {model_name} in position plot.")

    # --- Axis and Labels ---
    ax.set_xlabel("Normalized Sequence Position Bin", fontsize=10, labelpad=8)
    ax.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
    ax.set_title("Performance along Sequence Position (Mean ± 95% CI)", fontsize=12, pad=25)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=5))

    # --- Legend Styling ---
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = [h for h, l in zip(handles, labels) if l]
    filtered_labels = [l for l in labels if l]
    ax.legend(filtered_handles, filtered_labels,
              title='Model', loc='upper center',
              bbox_to_anchor=(0.5, 1.15),
              ncol=len(MODELS_TO_PLOT), frameon=False, fontsize=9)

    # --- Spines and Ticks ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    spine_color = '#555555'
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_color(spine_color)
    ax.tick_params(axis='x', colors=spine_color)
    ax.tick_params(axis='y', colors=spine_color)

    # --- ADD GRID LINES ---
    ax.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8, zorder=1) # Add white grid lines
    ax.set_axisbelow(True) # Ensure grid is behind data

    fig = plt.gcf()
    fig.set_facecolor('#FFFFFF')

    # Adjust tight_layout rect
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    try:
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally:
        plt.close()



        
# Plot D: Relative Accessibility (Box Plot + Mean RMSF)
def plot_accessibility_performance(df, output_path, logger):
    """
    Generates a line plot of Mean MAE per RSA quantile for each model,
    styled like the reference image.
    """
    global analysis_data
    panel_id = 'Panel D: Relative Accessibility'
    logger.info(f"Generating {panel_id} (Line Plot by Quantile)...")
    analysis_data[panel_id] = {}

    mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
    label_col = 'RSA_Quantile_Label'
    required_cols = [label_col] + mae_cols

    # Check if required columns exist and if labels were generated correctly
    if not all(c in df.columns for c in required_cols) or \
       df[label_col].isnull().all() or \
       'Not Available' in df[label_col].unique() or \
       'Error' in df[label_col].unique():
         logger.error(f"Panel D: Valid data for RSA analysis not found. Skipping.");
         analysis_data[panel_id]['Error'] = f"Data missing or invalid labels in {label_col}."
         return

    # --- Aggregate Mean MAE per Quantile per Model ---
    plot_data_melted = df[[label_col] + mae_cols].melt(id_vars=[label_col], var_name='Model', value_name='MAE')
    plot_data_melted['Model'] = plot_data_melted['Model'].str.replace('_MAE', '')
    plot_data_melted = plot_data_melted[plot_data_melted['Model'].isin(MODELS_TO_PLOT)] # Filter models
    plot_data_melted.dropna(subset=['MAE'], inplace=True)

    # Calculate mean MAE
    mean_mae_agg = plot_data_melted.groupby([label_col, 'Model'], observed=False)['MAE'].mean().unstack()

    # Attempt numerical sort based on label, fall back to category order
    try:
        mean_mae_agg['sort_val'] = mean_mae_agg.index.str.split('-').str[0].astype(float)
        mean_mae_agg = mean_mae_agg.sort_values('sort_val').drop(columns='sort_val')
        quantile_order = mean_mae_agg.index.tolist()
        logger.info("Sorted RSA quantiles numerically for plotting.")
    except Exception as e:
        logger.warning(f"Could not numerically sort RSA Quantile Labels for Panel D ({e}). Using default categorical order.")
        # Ensure the original categorical order from data prep is used if possible
        if pd.api.types.is_categorical_dtype(df[label_col]):
            quantile_order = df[label_col].cat.categories.tolist()
            # Filter out categories not present in aggregated data
            quantile_order = [q for q in quantile_order if q in mean_mae_agg.index]
            mean_mae_agg = mean_mae_agg.reindex(quantile_order) # Reindex based on original category order
        else:
            quantile_order = sorted(mean_mae_agg.index.unique()) # Fallback: alphabetical sort of existing labels
            mean_mae_agg = mean_mae_agg.reindex(quantile_order)

    # Store stats
    analysis_data[panel_id]['Mean MAE by RSA Quantile'] = mean_mae_agg.to_dict()

    # --- Plotting ---
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0') # Grey background

    x_pos = np.arange(len(quantile_order))

    for model_name in MODELS_TO_PLOT:
        if model_name in mean_mae_agg.columns:
            means = mean_mae_agg[model_name].values
            color = MODEL_COLORS.get(model_name, 'grey')
            ax.plot(x_pos, means, marker='o', linestyle='-', label=model_name,
                    color=color, markersize=7, linewidth=2.0)
        else:
            logger.warning(f"Data for model '{model_name}' not found in aggregated RSA data.")

    # --- Axis and Labels ---
    ax.set_xlabel("Relative Solvent Accessibility (RSA) Quantile", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
    ax.set_title("Performance by Solvent Accessibility", fontsize=12, pad=15) # Match reference title
    ax.set_xticks(x_pos)
    ax.set_xticklabels(quantile_order, rotation=30, ha='right', fontsize=8.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) # Adjust number of y-ticks if needed

    # --- Legend Styling ---
    ax.legend(title='Model', loc='upper left', frameon=False, fontsize=8.5)

    # --- Spines and Ticks ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    spine_color = '#555555' # Match other plots
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_color(spine_color)
    ax.tick_params(axis='x', colors=spine_color)
    ax.tick_params(axis='y', colors=spine_color)

    # --- Grid Lines ---
    ax.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8, zorder=1)
    ax.set_axisbelow(True)

    fig = plt.gcf()
    fig.set_facecolor('#FFFFFF')

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally:
        plt.close()
#     """ Generates box plots of MAE per RSA quantile, with mean actual RMSF overlaid."""
#     global analysis_data
#     panel_id = 'Panel D: Relative Accessibility'
#     logger.info(f"Generating {panel_id}...")
#     analysis_data[panel_id] = {}

#     mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
#     label_col = 'RSA_Quantile_Label'
#     required_cols = [TARGET_COL, RSA_COL, label_col] + mae_cols

#     if not all(c in df.columns for c in required_cols) or \
#        df[label_col].isnull().all() or \
#        df[label_col].astype(str).str.contains('Error|Unknown|Not Available', na=False).any():
#          logger.error(f"Valid data for RSA analysis not found. Skipping."); analysis_data[panel_id]['Error'] = f"Data missing/invalid."; return

#     plot_data_mae = df[[label_col] + mae_cols].melt(id_vars=[label_col], var_name='Model', value_name='MAE')
#     plot_data_mae['Model'] = plot_data_mae['Model'].str.replace('_MAE', '')
#     plot_data_mae.dropna(subset=['MAE'], inplace=True)
#     mean_rmsf_per_quantile = df.groupby(label_col, observed=False)[TARGET_COL].agg(['mean', 'count']).reset_index()

#     try:
#         mean_rmsf_per_quantile['sort_val'] = mean_rmsf_per_quantile[label_col].str.split('-').str[0].astype(float)
#         mean_rmsf_per_quantile = mean_rmsf_per_quantile.sort_values('sort_val')
#         quantile_order = mean_rmsf_per_quantile[label_col].tolist()
#         logger.info("Sorted RSA quantiles numerically.")
#     except Exception as e: logger.warning(f"Could not sort RSA Quantile Labels ({e}). Using default."); quantile_order = sorted(df[label_col].unique())

#     plot_data_mae[label_col] = pd.Categorical(plot_data_mae[label_col], categories=quantile_order, ordered=True)
#     plot_data_mae = plot_data_mae.sort_values(label_col)

#     stats_mae_summary = plot_data_mae.groupby([label_col, 'Model'], observed=False)['MAE'].agg(['mean', 'median', 'count']).unstack()
#     analysis_data[panel_id]['MAE Stats (Mean/Median/Count by RSA Quantile, Model)'] = stats_mae_summary.to_dict()
#     rmsf_stats_dict = mean_rmsf_per_quantile.set_index(label_col)[['mean', 'count']].rename(columns={'mean':'Mean Actual RMSF'}).to_dict()
#     analysis_data[panel_id]['Actual RMSF Stats (Mean/Count by RSA Quantile)'] = rmsf_stats_dict

#     fig, ax1 = plt.subplots(figsize=(7, 5))
#     sns.boxplot(
#         data=plot_data_mae, x=label_col, y='MAE', hue='Model', order=quantile_order,
#         palette=MODEL_COLORS, linewidth=1.0, fliersize=1.0, showfliers=False,
#         width=0.75, dodge=True, ax=ax1 # Ensure dodge for spacing
#     )
#     ax1.set_xlabel("Relative Solvent Accessibility (RSA) Quantile", fontsize=10)
#     ax1.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
#     ax1.tick_params(axis='x', rotation=30, labelsize=8.5)
#     ax1.tick_params(axis='y', labelsize=8.5); ax1.set_ylim(bottom=0)

#     ax2 = ax1.twinx()
#     x_pos = np.arange(len(quantile_order))
#     # ax2.plot(x_pos, mean_rmsf_per_quantile['mean'], color='black', linestyle='--', linewidth=1.5, marker='^', markersize=5, label='Mean Actual RMSF')
#     # ax2.set_ylabel("Mean Actual RMSF (Å)", fontsize=10, color='black')
#     # ax2.tick_params(axis='y', labelcolor='black', labelsize=8.5); ax2.set_ylim(bottom=0)

#     lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
#     valid_lines = [line for line in lines1 + lines2 if line is not None]; valid_labels = [label for i, label in enumerate(labels1 + labels2) if (lines1+lines2)[i] is not None]
#     if valid_lines: ax1.legend(valid_lines, valid_labels, title="Metric", loc='upper left', frameon=False, fontsize=8.5)
#     if ax2.get_legend(): ax2.get_legend().remove()

#     ax1.set_title("Performance vs. Actual Flexibility by RSA Quantile", fontsize=12)
#     ax1.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8); ax1.set_axisbelow(True)
#     ax2.grid(False)
#     ax1.set_xticks(x_pos); ax1.set_xticklabels(quantile_order, rotation=30, ha='right')

#     fig.tight_layout()
#     try: plt.savefig(output_path); logger.info(f"Saved plot to {output_path}")
#     except Exception as e: logger.error(f"Failed to save plot: {e}", exc_info=True)
#     finally: plt.close()


# Plot E: B-factor Quantiles (Box Plots)
def plot_bfactor_boxplot_quantiles(df, output_path, logger):
    """ Generates box plots of MAE per B-factor quantile for each model."""
    global analysis_data
    panel_id = 'Panel E: B-factor Performance'
    logger.info(f"Generating {panel_id} (Box Plot by Quantile)...")
    analysis_data[panel_id] = {}

    mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
    label_col = 'BFactor_Quantile_Label'
    required_cols = [label_col] + mae_cols

    if not all(c in df.columns for c in required_cols) or \
       df[label_col].isnull().all() or \
       df[label_col].astype(str).str.contains('Error|Unknown|Not Available', na=False).any():
         logger.error(f"Valid data for B-factor analysis not found. Skipping."); analysis_data[panel_id]['Error'] = f"Data missing/invalid."; return

    plot_data = df[[label_col] + mae_cols].melt(id_vars=[label_col], var_name='Model', value_name='MAE')
    plot_data['Model'] = plot_data['Model'].str.replace('_MAE', '')
    plot_data.dropna(subset=['MAE'], inplace=True)

    try:
        sorter_df = pd.DataFrame({'label': plot_data[label_col].unique()})
        # Handle potential negative numbers in B-factor labels
        sorter_df['sort_val'] = sorter_df['label'].str.split('-').str[0].str.replace(r'[()]', '', regex=True).astype(float)
        sorter_df = sorter_df.sort_values('sort_val')
        quantile_order = sorter_df['label'].tolist()
        logger.info("Sorted B-Factor quantiles numerically.")
    except Exception as e:
        logger.warning(f"Could not numerically sort B-Factor Quantile Labels ({e}). Using default order.")
        quantile_order = sorted(plot_data[label_col].unique())

    plot_data[label_col] = pd.Categorical(plot_data[label_col], categories=quantile_order, ordered=True)
    plot_data = plot_data.sort_values(label_col)

    stats_summary = plot_data.groupby([label_col, 'Model'], observed=False)['MAE'].agg(['mean', 'median', 'count']).unstack()
    analysis_data[panel_id]['MAE Stats (Mean/Median/Count by B-factor Quantile, Model)'] = stats_summary.to_dict()

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    sns.boxplot(
        data=plot_data, x=label_col, y='MAE', hue='Model', order=quantile_order,
        palette=MODEL_COLORS, linewidth=1.0, fliersize=1.0, showfliers=False,
        width=0.75, dodge=True, ax=ax
    )
    ax.set_xlabel("Normalized B-factor Quantile", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
    ax.set_title("Performance by B-factor Quantile", fontsize=12)
    ax.tick_params(axis='x', rotation=30, labelsize=8.5) # Removed ha='right'
    ax.tick_params(axis='y', labelsize=8.5)
    ax.text(0.01, 0.01, "Note: B-factors Z-score normalized per PDB chain",
            transform=ax.transAxes, fontsize=7, style='italic', alpha=0.7)
    ax.legend(title='Model', loc='upper left', frameon=False)
    ax.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True); ax.set_ylim(bottom=0)
    # Set tick labels explicitly after sorting
    ax.set_xticklabels(quantile_order, rotation=30, ha='right')


    plt.tight_layout()
    try: plt.savefig(output_path); logger.info(f"Saved plot to {output_path}")
    except Exception as e: logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally: plt.close()


# Plot F: Phi/Psi Angles (Ramachandran KDE Plot) - Reverted KDE Version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter # Use this for smoothing
import logging
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.cm as cm

# Assume logger, PRIMARY_MODEL_NAME, PHI_COL, PSI_COL, analysis_data are defined

# Plot F: Phi/Psi Angles (Smoothed Mean MAE Heatmap) - Corrected pcolormesh v2
def plot_ramachandran_error(df, output_path, logger):
    """ Generates a Ramachandran plot showing Smoothed Mean MAE per bin,
        with density contours overlaid for context."""
    global analysis_data
    panel_id = 'Panel F: Ramachandran (Mean MAE)'
    logger.info(f"Generating {panel_id}...")
    analysis_data[panel_id] = {}

    primary_mae_col = f"{PRIMARY_MODEL_NAME}_MAE"
    required_cols = [PHI_COL, PSI_COL, primary_mae_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns: {required_cols}. Skipping."); analysis_data[panel_id]['Error'] = f"Missing columns: {required_cols}"; return

    plot_data = df[required_cols].dropna()
    n_points = len(plot_data)
    analysis_data[panel_id]['Total Points Plotted'] = n_points
    if n_points < 100: logger.warning("Insufficient data points (<100)."); analysis_data[panel_id]['Warning'] = "Insufficient data points (<100)"; return

    # --- Binning and Calculation ---
    n_bins = 72
    phi_bins = np.linspace(-180, 180, n_bins + 1); psi_bins = np.linspace(-180, 180, n_bins + 1)
    bin_centers_phi = (phi_bins[:-1] + phi_bins[1:]) / 2; bin_centers_psi = (psi_bins[:-1] + psi_bins[1:]) / 2

    try:
        # IMPORTANT: binned_statistic_2d expects x first (phi), then y (psi)
        mean_mae, _, _, _ = stats.binned_statistic_2d(
            plot_data[PHI_COL], plot_data[PSI_COL], plot_data[primary_mae_col],
            statistic='mean', bins=[phi_bins, psi_bins])
        # The resulting mean_mae grid has shape (n_phi_bins, n_psi_bins) based on input order
    except ValueError as e: logger.error(f"Error during 2D binning: {e}"); return

    density, _, _ = np.histogram2d(plot_data[PHI_COL], plot_data[PSI_COL], bins=[phi_bins, psi_bins])
    # Density grid also has shape (n_phi_bins, n_psi_bins)

    # --- Smoothing ---
    sigma_smooth = 1.0
    mean_mae_smooth = gaussian_filter(mean_mae, sigma=sigma_smooth)
    mean_mae_smooth = np.ma.masked_where(np.isnan(mean_mae), mean_mae_smooth) # Mask original NaNs

    # --- Plotting ---
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    valid_mae_values = mean_mae_smooth.compressed()
    if len(valid_mae_values) < 2: vmin, vmax = np.nanmin(mean_mae_smooth), np.nanmax(mean_mae_smooth)
    else: vmin = np.percentile(valid_mae_values, 5); vmax = np.percentile(valid_mae_values, 95)
    vmin = max(0, vmin) if vmin is not None else 0

    # --- FIX: Ensure correct dimensions for pcolormesh ---
    # C should be shape (N_psi, N_phi) to match Y_psi_bins (N_psi+1) and X_phi_bins (N_phi+1)
    # Since mean_mae_smooth is (N_phi, N_psi), we DON'T transpose it here.
    pcm = ax.pcolormesh(phi_bins, psi_bins, mean_mae_smooth, # No .T here
                        cmap='viridis',
                        shading='auto', # Let matplotlib decide based on X/Y/C dims
                        vmin=vmin, vmax=vmax)
    # ------------------------------------------------------
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(f"Smoothed Mean {PRIMARY_MODEL_NAME} MAE (Å)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # --- Overlay Density Contours ---
    density_smooth = gaussian_filter(density, sigma=0.8)
    density_levels = np.percentile(density_smooth[density_smooth > 5], [25, 50, 75, 90])
    X_centers, Y_centers = np.meshgrid(bin_centers_phi, bin_centers_psi)
    if X_centers.size > 1 and Y_centers.size > 1 and density_smooth.size > 1:
         # Contour also expects Z array shape (N_psi, N_phi) to match Y_centers, X_centers
         # Since density_smooth is (N_phi, N_psi), we DO transpose it here.
         ax.contour(X_centers, Y_centers, density_smooth.T, levels=density_levels, colors='white',
                   linewidths=0.6, alpha=0.7, linestyles='--')
    else: logger.warning("Skipping density contours due to insufficient data/dimensions.")

    # --- Annotations & Styling ---
    regions = {
        'α-Helix': {'phi': (-80, -40), 'psi': (-60, -20)},
        'β-Sheet': {'phi': (-150, -90), 'psi': (120, 160)},
        'L-α': {'phi': (40, 80), 'psi': (20, 80)}
    }
    analysis_data[panel_id]['Mean MAE in Regions'] = {}
    for label, bounds in regions.items():
        # Indexing needs to match the non-transposed mean_mae shape (N_phi, N_psi)
        phi_idx = (bin_centers_phi >= bounds['phi'][0]) & (bin_centers_phi <= bounds['phi'][1])
        psi_idx = (bin_centers_psi >= bounds['psi'][0]) & (bin_centers_psi <= bounds['psi'][1])
        # Extract relevant part: mean_mae[phi_indices, :][:, psi_indices] -> meshgrid better
        region_mean_mae_values = mean_mae[np.ix_(phi_idx, psi_idx)] # mean_mae is (N_phi, N_psi)
        region_density_values = density[np.ix_(phi_idx, psi_idx)]   # density is (N_phi, N_psi)

        valid_means_in_region = region_mean_mae_values[~np.isnan(region_mean_mae_values)]
        mean_mae_region = np.mean(valid_means_in_region) if len(valid_means_in_region) > 0 else np.nan
        count_region = np.sum(region_density_values)

        analysis_data[panel_id]['Mean MAE in Regions'][label] = {'Mean MAE (Avg Bin)': mean_mae_region, 'Count': count_region}
        ax.text(np.mean(bounds['phi']), np.mean(bounds['psi']), label, color='red', ha='center', va='center', fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    ax.set_xlabel("Phi (φ) Angle (°)", fontsize=10); ax.set_ylabel("Psi (ψ) Angle (°)", fontsize=10)
    ax.set_title(f"{PRIMARY_MODEL_NAME} Mean MAE vs Dihedral Angles", fontsize=12)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60)); ax.set_yticks(np.arange(-180, 181, 60))
    ax.axhline(0, color='white', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.axvline(0, color='white', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.grid(False); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    try: plt.savefig(output_path); logger.info(f"Saved plot to {output_path}")
    except Exception as e: logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally: plt.close()
    
# --- Amino Acid Full Name Mapping ---
AMINO_ACID_FULL_NAMES = {
    'ALA': 'Alanine', 'ARG': 'Arginine', 'ASN': 'Asparagine', 'ASP': 'Aspartic Acid',
    'CYS': 'Cysteine', 'GLN': 'Glutamine', 'GLU': 'Glutamic Acid', 'GLY': 'Glycine',
    'HIS': 'Histidine', 'ILE': 'Isoleucine', 'LEU': 'Leucine', 'LYS': 'Lysine',
    'MET': 'Methionine', 'PHE': 'Phenylalanine', 'PRO': 'Proline', 'SER': 'Serine',
    'THR': 'Threonine', 'TRP': 'Tryptophan', 'TYR': 'Tyrosine', 'VAL': 'Valine'
}

# Assume these are defined globally or passed as arguments
# logger = logging.getLogger(__name__) # Should be initialized externally
# MODEL_COLORS = {...}
# MODELS_TO_PLOT = [...]
# PRIMARY_MODEL_NAME = '...'
# RESNAME_COL = 'resname'
# analysis_data = {} # Global dict for analysis

# Plot G: Grouped Bar Plot of Mean MAE by Amino Acid (Styled) - Abbreviations, Vertical Ticks

def plot_grouped_mae_by_aa(df, output_path, logger):
    """
    Generates grouped bar plot of Mean MAE ± 95% CI by Amino Acid,
    sorted LOWEST to HIGHEST by DeepFlex performance, with legend upper left.
    """
    global analysis_data
    panel_id = 'Panel G: Amino Acid Performance'
    logger.info(f"Generating {panel_id} (Bar Plot - Mean ± 95% CI, Sorted by {PRIMARY_MODEL_NAME}, Legend Left)...")
    analysis_data[panel_id] = {}

    mae_cols = [f"{m}_MAE" for m in MODELS_TO_PLOT]
    required_cols = [RESNAME_COL] + mae_cols
    if not all(col in df.columns for col in required_cols) or RESNAME_COL not in df.columns:
        logger.error(f"Missing required columns for Panel G. Skipping.");
        analysis_data[panel_id]['Error'] = "Missing required columns."
        return

    plot_data_melted = df[[RESNAME_COL] + mae_cols].melt(id_vars=RESNAME_COL, var_name='Model', value_name='MAE')
    plot_data_melted['Model'] = plot_data_melted['Model'].str.replace('_MAE', '')

    # Filter only standard AAs and valid models/MAE
    standard_aas = list(AMINO_ACID_FULL_NAMES.keys())
    plot_data_melted = plot_data_melted[
        plot_data_melted[RESNAME_COL].isin(standard_aas) &
        plot_data_melted['Model'].isin(MODELS_TO_PLOT)
    ].dropna(subset=['MAE'])

    if plot_data_melted.empty:
        logger.error("Panel G: No valid data remaining after filtering. Skipping."); return

    # Calculate stats: Mean, Median, Count
    # Important: Group by RESNAME first, then Model to get stats per AA per Model
    stats_summary = plot_data_melted.groupby([RESNAME_COL, 'Model'], observed=True)['MAE'].agg(['mean', 'std', 'count', 'sem']).reset_index() # Use SEM for CI

    analysis_data[panel_id]['Stats (Mean/Std/Count/SEM by AA, Model)'] = stats_summary.set_index([RESNAME_COL, 'Model']).unstack().to_dict() # Save detailed stats

    # Determine Amino Acid Order based on DeepFlex Mean MAE (Lowest to Highest)
    deepflex_stats = stats_summary[stats_summary['Model'] == PRIMARY_MODEL_NAME].copy()
    if not deepflex_stats.empty:
         # Sort the DeepFlex stats by the 'mean' column (lowest first)
         deepflex_stats_sorted = deepflex_stats.sort_values(by='mean', ascending=True)
         # Get the order of amino acids from the sorted DataFrame
         aa_order = deepflex_stats_sorted[RESNAME_COL].tolist()
         analysis_data[panel_id]['AA Order (Lowest to Highest DeepFlex Mean MAE)'] = aa_order
         logger.info(f"Confirmed: Amino acids ordered by {PRIMARY_MODEL_NAME} Mean MAE (Lowest first).")
         analysis_data[panel_id]['Top 3 AAs (DeepFlex Lowest Mean MAE)'] = deepflex_stats_sorted.nsmallest(3, 'mean').set_index(RESNAME_COL)['mean'].to_dict()
         analysis_data[panel_id]['Bottom 3 AAs (DeepFlex Highest Mean MAE)'] = deepflex_stats_sorted.nlargest(3, 'mean').set_index(RESNAME_COL)['mean'].to_dict()
    else:
         logger.warning(f"Could not find stats for '{PRIMARY_MODEL_NAME}' to sort by. Using alphabetical AA order.")
         aa_order = sorted(plot_data_melted[RESNAME_COL].unique())
         analysis_data[panel_id]['AA Order'] = 'Default alphabetical (sorting failed)'

    # --- Plotting ---
    plt.figure(figsize=(12, 5)) # Adjusted figsize slightly
    ax = plt.gca()
    # Use the calculated aa_order in the plot function
    sns.barplot(
        data=plot_data_melted, x=RESNAME_COL, y='MAE', hue='Model',
        order=aa_order, # Use the CORRECTLY SORTED order
        palette=MODEL_COLORS, hue_order=MODELS_TO_PLOT,
        errorbar=('ci', 95), # Seaborn calculates CI from data directly
        err_kws={'linewidth': 1, 'color': 'black'}, capsize=0.05,
        edgecolor='black', linewidth=0.8, ax=ax
    )

    ax.set_xlabel("Amino Acid", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (Å)", fontsize=10)
    ax.set_title("Performance by Amino Acid", fontsize=12) # Simplified title
    ax.set_xticks(np.arange(len(aa_order)))
    ax.set_xticklabels(aa_order, rotation=90, ha='center', fontsize=9) # Use 3-letter codes
    ax.tick_params(axis='y', labelsize=9.5)
    ax.set_ylim(bottom=0) # Start y-axis at 0

    # --- Auto-adjust y-limits (optional, can fine-tune) ---
    # Calculate approximate max CI value for setting ylim buffer
    if not stats_summary.empty:
         stats_summary['ci_upper'] = stats_summary['mean'] + 1.96 * stats_summary['sem'].fillna(0)
         max_ci_val = stats_summary['ci_upper'].max()
         if pd.notna(max_ci_val):
             ax.set_ylim(bottom=0, top=max_ci_val * 1.05) # Add 5% buffer
             analysis_data[panel_id]['Y-axis Limits'] = {'Bottom': ax.get_ylim()[0], 'Top': ax.get_ylim()[1]}
         else:
             logger.warning("Could not calculate max CI for Y-limit adjustment.")
    # -------------------------------------------------------

    ax.grid(axis='y', linestyle='-', color='white', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    # --- Move Legend to Upper Left ---
    ax.legend(handles=handles, labels=labels, title="Model", loc='upper left', frameon=False, fontsize=7.5)

    plt.tight_layout()
    try: plt.savefig(output_path); logger.info(f"Saved plot to {output_path}")
    except Exception as e: logger.error(f"Failed to save plot: {e}", exc_info=True)
    finally: plt.close()

# --- Analysis CSV Generation (Corrected Formatting & Structure Handling v3) ---
def write_analysis_csv(data_dict, output_path, logger):
    """Writes the collected analysis data to a structured CSV file."""
    logger.info(f"Generating Analysis Summary CSV: {output_path}")
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Panel", "Metric", "Details", "Value"]) # Header

            def format_value(v):
                if isinstance(v, (list, np.ndarray, pd.Series)): return "; ".join(map(str, [item for item in v if pd.notna(item)]))
                elif pd.isna(v) or v is None: return "NaN"
                elif isinstance(v, (float, np.number, np.floating)): return f"{v:.4f}"
                elif isinstance(v, (int, np.integer)): return str(v)
                elif isinstance(v, str):
                     val_esc = v.replace('"', '""'); return f'"{val_esc}"' if ',' in val_esc or '"' in val_esc or '\n' in val_esc else val_esc
                else:
                    try: return str(v).replace('"', '""')
                    except Exception: return "Formatting Error"

            if 'Overall' in data_dict:
                writer.writerow(["Overall Stats", "", "", ""]);
                for key, value in data_dict.get('Overall', {}).items(): writer.writerow(["Overall", key, "", format_value(value)])
                writer.writerow([])

            panel_keys = sorted([k for k in data_dict if k.startswith('Panel')], key=lambda x: x.split(':')[0])

            for panel_id in panel_keys:
                writer.writerow([panel_id, "", "", ""])
                panel_data = data_dict.get(panel_id, {})
                for metric, value in panel_data.items():
                    if not isinstance(value, dict) and not isinstance(value, list) and not metric.startswith('Stats'): writer.writerow([panel_id, metric, "", format_value(value)])

                # Process complex stats dictionaries
                stats_key_1 = next((k for k in panel_data if k.startswith('Stats (Mean/Median/Count by') and 'Model)' in k and ('SS' in k or 'Location' in k or 'AA' in k)), None)
                if stats_key_1 and isinstance(panel_data.get(stats_key_1), dict):
                    stats_dict = panel_data[stats_key_1]
                    for stat_type, category_data in stats_dict.items():
                        if isinstance(category_data, dict):
                            for category_label, model_data in category_data.items():
                                if isinstance(model_data, dict):
                                    for model, val in model_data.items(): writer.writerow([panel_id, f"{stat_type} MAE", f"{category_label} - {model}", format_value(val)])
                                else: writer.writerow([panel_id, f"{stat_type} MAE", f"{category_label} - ???", format_value(model_data)])

                stats_key_2 = next((k for k in panel_data if k.startswith('Stats (Mean/Count by') and ('Position Bin' in k or 'RSA Quantile' in k)), None)
                if stats_key_2 and isinstance(panel_data.get(stats_key_2), dict):
                     stats_dict = panel_data[stats_key_2]
                     for stat_tuple, bin_data in stats_dict.items():
                         if isinstance(stat_tuple, tuple) and len(stat_tuple) == 2 and isinstance(bin_data, dict):
                             model_mae, stat_type = stat_tuple; model = str(model_mae).replace("_MAE", "")
                             for bin_label, val in bin_data.items(): writer.writerow([panel_id, f"{stat_type} MAE", f"{bin_label} - {model}", format_value(val)])

                # Added handling for Panel D RMSF stats
                if 'Actual RMSF Stats (Mean/Count by RSA Quantile)' in panel_data:
                     stats_dict = panel_data['Actual RMSF Stats (Mean/Count by RSA Quantile)']
                     if isinstance(stats_dict, dict):
                          for stat_type, quant_data in stats_dict.items():
                              if isinstance(quant_data, dict):
                                  for quant_label, val in quant_data.items(): writer.writerow([panel_id, f"{stat_type} Actual RMSF", f"{quant_label}", format_value(val)])

                if 'Mean MAE in Regions' in panel_data and isinstance(panel_data['Mean MAE in Regions'], dict):
                    region_dict = panel_data['Mean MAE in Regions']
                    for region, stats in region_dict.items():
                        if isinstance(stats, dict):
                            writer.writerow([panel_id, "Mean MAE", region, format_value(stats.get('Mean MAE'))])
                            writer.writerow([panel_id, "Count", region, format_value(stats.get('Count'))])

                # AA Specific metrics
                if 'AA Order (Best to Worst DeepFlex Mean MAE)' in panel_data: writer.writerow([panel_id, "AA Order", "(Best)", format_value(panel_data['AA Order (Best to Worst DeepFlex Mean MAE)'])])
                if 'Top 3 AAs (DeepFlex Lowest Mean MAE)' in panel_data:
                     writer.writerow([panel_id, "Top 3 AAs (DeepFlex)", "", ""]);
                     if isinstance(panel_data['Top 3 AAs (DeepFlex Lowest Mean MAE)'], dict):
                         for aa, val in panel_data['Top 3 AAs (DeepFlex Lowest Mean MAE)'].items(): writer.writerow(["", "", aa, format_value(val)])
                if 'Bottom 3 AAs (DeepFlex Highest Mean MAE)' in panel_data:
                     writer.writerow([panel_id, "Bottom 3 AAs (DeepFlex)", "", ""]);
                     if isinstance(panel_data['Bottom 3 AAs (DeepFlex Highest Mean MAE)'], dict):
                         for aa, val in panel_data['Bottom 3 AAs (DeepFlex Highest Mean MAE)'].items(): writer.writerow(["", "", aa, format_value(val)])
                # B-factor Correlation specific
                corr_key = f'Pearson Correlation ({PRIMARY_MODEL_NAME} MAE vs Norm B-factor)'
                if corr_key in panel_data:
                    writer.writerow([panel_id, corr_key, "", format_value(panel_data[corr_key])])
                    writer.writerow([panel_id, f'P-value', "", format_value(panel_data.get(f'P-value'))])
                if 'Y-axis Limits' in panel_data:
                    if isinstance(panel_data['Y-axis Limits'], dict):
                        writer.writerow([panel_id, "Y-axis Limit Bottom", "", format_value(panel_data['Y-axis Limits'].get('Bottom'))])
                        writer.writerow([panel_id, "Y-axis Limit Top", "", format_value(panel_data['Y-axis Limits'].get('Top'))])

                writer.writerow([]) # Spacer

        logger.info("Successfully wrote analysis summary CSV.")
    except Exception as e:
        logger.error(f"Failed to write analysis CSV: {e}", exc_info=True)

# --- Main Execution ---
def main():
    """Main function to generate all figure panels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    setup_plotting_style()

    df = load_and_prepare_data(INPUT_CSV, logger)
    if df is None: logger.critical("Failed to load or prepare data. Exiting."); return

    # --- Generate plots with final naming convention ---
    plot_ss_performance(df, OUTPUT_DIR / "figure4_a_ss.png", logger)
    plot_core_exterior_performance(df, OUTPUT_DIR / "figure4_b_core_exterior.png", logger)
    plot_position_performance(df, OUTPUT_DIR / "figure4_c_position.png", logger)
    plot_accessibility_performance(df, OUTPUT_DIR / "figure4_d_accessibility.png", logger)
    plot_bfactor_boxplot_quantiles(df, OUTPUT_DIR / "figure4_e_bfactor.png", logger) # Changed B-factor plot

    if PHI_COL in df.columns and PSI_COL in df.columns:
        plot_ramachandran_error(df, OUTPUT_DIR / "figure4_f_ramachandran.png", logger) # Reverted KDE
    else: logger.warning(f"Skipping Ramachandran plot: Missing '{PHI_COL}' or '{PSI_COL}'.")

    plot_grouped_mae_by_aa(df, OUTPUT_DIR / "figure4_g_amino_acid_grouped.png", logger) # Widened bars

    # --- Write Analysis CSV ---
    write_analysis_csv(analysis_data, OUTPUT_DIR / "figure4_analysis.csv", logger)

    logger.info("--- Figure 4 Generation Script Finished ---")

if __name__ == "__main__":
    main()
    
    
    

# def plot_bfactor_jointplot(df, output_path, logger):
#     """ Generates jointplot of B-factor vs MAE with regression and correlation."""
#     global analysis_data
#     panel_id = 'Panel E: B-factor vs MAE'
#     logger.info(f"Generating {panel_id} (JointPlot)...")
#     analysis_data[panel_id] = {}

#     primary_mae_col = f"{PRIMARY_MODEL_NAME}_MAE"
#     required_cols = [BFACTOR_COL, primary_mae_col]
#     if not all(col in df.columns for col in required_cols):
#         logger.error(f"Missing required columns: {required_cols}. Skipping."); analysis_data[panel_id]['Error'] = f"Missing columns: {required_cols}"; return

#     plot_data = df[required_cols].dropna()
#     n_points = len(plot_data)
#     analysis_data[panel_id]['Total Points Plotted'] = n_points
#     if n_points < 50: logger.warning("Insufficient data points (<50). Skipping."); analysis_data[panel_id]['Warning'] = "Insufficient data points (<50)"; return

#     # Calculate correlation
#     try:
#         corr, pval = stats.pearsonr(plot_data[BFACTOR_COL], plot_data[primary_mae_col])
#         corr_text = f"r = {corr:.3f}{'*' if pval < 0.05 else ''}"
#         analysis_data[panel_id][f'Pearson Correlation ({PRIMARY_MODEL_NAME} MAE vs Norm B-factor)'] = corr
#         analysis_data[panel_id][f'P-value'] = pval
#     except ValueError: corr_text = "r = N/A"; analysis_data[panel_id][f'Pearson Correlation (...)'] = 'N/A'

#     g = sns.JointGrid(data=plot_data, x=BFACTOR_COL, y=primary_mae_col, height=5, space=0.1) # Reduce space
#     for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]: ax.set_facecolor('#f0f0f0'); ax.grid(False)

#     g.plot_joint(sns.scatterplot, s=5, alpha=0.08, color=MODEL_COLORS[PRIMARY_MODEL_NAME], edgecolor='none') # More transparent
#     g.plot_joint(sns.regplot, scatter=False, color='#e41a1c', line_kws={'linewidth': 1.5}) # Use a distinct color (e.g., red)
#     g.plot_joint(sns.kdeplot, levels=5, color='black', linewidths=0.6, linestyles='--', alpha=0.7)
#     g.plot_marginals(sns.histplot, kde=False, color=MODEL_COLORS[PRIMARY_MODEL_NAME], bins=40, alpha=0.6) # More bins

#     # Place correlation text inside
#     g.ax_joint.text(0.05, 0.95, corr_text, transform=g.ax_joint.transAxes, ha='left', va='top', fontsize=9,
#                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

#     g.ax_joint.set_xlabel("Normalized B-factor", fontsize=10); g.ax_joint.set_ylabel(f"{PRIMARY_MODEL_NAME} MAE (Å)", fontsize=10)
#     plt.suptitle(f"{PRIMARY_MODEL_NAME} MAE vs. B-factor Relationship", y=1.02, fontsize=12)

#     y_curr = g.ax_joint.get_ylim(); g.ax_joint.set_ylim(bottom=max(0, y_curr[0] * 0.9), top=y_curr[1] * 1.05)
#     x_curr = g.ax_joint.get_xlim(); g.ax_joint.set_xlim(left=x_curr[0]*0.95, right=x_curr[1]*1.05)

#     g.ax_joint.grid(axis='both', linestyle='-', color='white', linewidth=0.7, alpha=0.8); g.ax_joint.set_axisbelow(True)

#     try: plt.savefig(output_path); logger.info(f"Saved plot to {output_path}")
#     except Exception as e: logger.error(f"Failed to save plot: {e}", exc_info=True)
#     plt.close()
