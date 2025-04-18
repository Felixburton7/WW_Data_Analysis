# /home/s_felix/FINAL_PROJECT/Data_Analysis/analysis_scripts/01_overall_performance.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import os
import logging

# --- Configuration ---
INPUT_CSV = "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv"
OUTPUT_DIR_TABLES = "results_output/tables"
OUTPUT_DIR_SUMMARY = "results_output/summary_data"
METRICS_OUTPUT_CSV = os.path.join(OUTPUT_DIR_TABLES, "table1_overall_metrics.csv")
PLOT_DATA_OUTPUT_CSV = os.path.join(OUTPUT_DIR_SUMMARY, "figure1_plot_data.csv")

TARGET_COLUMN = 'rmsf'

# Map raw CSV columns to desired model names for the output table
MODEL_PREDICTION_COLUMNS = {
    'ESM-Flex': 'Attention_ESM_rmsf',
    'RF': 'ensembleflex_RF_rmsf',
    'NN': 'ensembleflex_NN_rmsf',        # Assuming NN is the feature-based Neural Network baseline
    'LGBM': 'ensembleflex_LGBM_rmsf',
    'ESM-Only': 'esm_rmsf',         # Baseline using only ESM embeddings + Temp
    'VoxelFlex': 'voxel_rmsf',       # Baseline using only VoxelFlex 3D CNN
    'No_ESM_RF': 'No_ESM_RF_prediction' # <<< CHANGED: Replaced 'MLP' with 'No_ESM_RF' and updated column name
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def calculate_metrics(y_true, y_pred):
    """Calculates performance metrics handling potential errors."""
    metrics = {
        'PCC': np.nan,
        'R2': np.nan,
        'MAE': np.nan,
        'RMSE': np.nan,
        'MedAE': np.nan,
        'Count': len(y_true) # Store the number of valid pairs used
    }
    if len(y_true) < 2: # Need at least 2 points for correlation/R2
        logging.warning(f"Skipping PCC/R2 calculation due to insufficient data points ({len(y_true)}).")
        if len(y_true) > 0:
             # Calculate basic error metrics even with few points
             try: metrics['MAE'] = mean_absolute_error(y_true, y_pred)
             except Exception as e: logging.warning(f"MAE calculation failed: {e}")
             try: metrics['RMSE'] = root_mean_squared_error(y_true, y_pred)
             except Exception as e: logging.warning(f"RMSE calculation failed: {e}")
             try: metrics['MedAE'] = np.median(np.abs(y_true - y_pred))
             except Exception as e: logging.warning(f"MedAE calculation failed: {e}")
        return metrics

    # Calculate metrics for sufficient data points
    try:
        # Pearson correlation
        pcc_val, _ = pearsonr(y_true, y_pred)
        metrics['PCC'] = pcc_val if not np.isnan(pcc_val) else 0.0
    except ValueError as e:
        logging.warning(f"Could not calculate PCC: {e}")

    try:
        # R-squared
        r2_val = r2_score(y_true, y_pred)
        metrics['R2'] = r2_val if not np.isnan(r2_val) else np.nan # R2 can be neg/NaN legitimately
    except ValueError as e:
        logging.warning(f"Could not calculate R2: {e}")

    try:
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Could not calculate MAE: {e}")

    try:
        # Use squared=False for direct RMSE
        metrics['RMSE'] = root_mean_squared_error(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Could not calculate RMSE: {e}")

    try:
        metrics['MedAE'] = np.median(np.abs(y_true - y_pred))
    except Exception as e: # Catch potential numpy errors on edge cases
         logging.warning(f"Could not calculate MedAE: {e}")

    return metrics

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting overall performance analysis script.")

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SUMMARY, exist_ok=True)
    logging.info(f"Output directories checked/created: {OUTPUT_DIR_TABLES}, {OUTPUT_DIR_SUMMARY}")

    # Load data
    logging.info(f"Loading data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        # Log columns to verify merge
        logging.info(f"Available columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_CSV}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        exit(1)

    # --- Calculate Overall Metrics for Each Model ---
    results = []
    logging.info("Calculating overall performance metrics for each model...")

    # Check if all specified prediction columns actually exist in the dataframe
    missing_cols = []
    for model_name, pred_col in MODEL_PREDICTION_COLUMNS.items():
        if pred_col not in df.columns:
            logging.error(f"Prediction column '{pred_col}' defined for model '{model_name}' not found in {INPUT_CSV}! Skipping this model.")
            missing_cols.append(pred_col)
    if missing_cols:
        logging.warning(f"Please ensure the following columns exist in your input CSV: {missing_cols}")

    # Iterate through the models defined in the dictionary
    for model_name, pred_col in MODEL_PREDICTION_COLUMNS.items():
        # Skip if column was missing
        if pred_col not in df.columns:
            continue

        logging.info(f"Processing model: {model_name} (column: {pred_col})")

        # Filter for rows where both target and prediction are valid (pairwise)
        # Also convert to float just in case there are strings causing issues
        try:
            temp_df = df[[TARGET_COLUMN, pred_col]].copy()
            temp_df[TARGET_COLUMN] = pd.to_numeric(temp_df[TARGET_COLUMN], errors='coerce')
            temp_df[pred_col] = pd.to_numeric(temp_df[pred_col], errors='coerce')
            valid_df = temp_df.dropna()
        except Exception as e:
            logging.error(f"Error preparing data for model {model_name}: {e}")
            valid_df = pd.DataFrame() # Ensure empty df if error occurs

        if valid_df.empty:
            logging.warning(f"No valid numeric data points found for model {model_name} after dropping NaNs. Check input data and column '{pred_col}'.")
            metrics = {'Model': model_name, 'PCC': np.nan, 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MedAE': np.nan, 'Count': 0}
        else:
            y_true = valid_df[TARGET_COLUMN].values
            y_pred = valid_df[pred_col].values
            logging.info(f"  Found {len(y_true)} valid data points for metric calculation.")

            # Calculate metrics
            model_metrics = calculate_metrics(y_true, y_pred)
            model_metrics['Model'] = model_name # Add model name for the results table
            metrics = model_metrics

        results.append(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # Reorder columns for clarity
    if not results_df.empty:
        results_df = results_df[['Model', 'PCC', 'R2', 'MAE', 'RMSE', 'MedAE', 'Count']]

    # Save metrics table
    try:
        results_df.to_csv(METRICS_OUTPUT_CSV, index=False, float_format='%.6f')
        logging.info(f"Overall performance metrics saved to: {METRICS_OUTPUT_CSV}")
        print("\n--- Overall Performance Metrics ---")
        print(results_df.to_string(index=False))
        print("-----------------------------------\n")
    except Exception as e:
        logging.error(f"Error saving metrics CSV: {e}")

    # --- Prepare Data for Figure 1 (ESM-Flex Scatter Plot) ---
    # This part remains unchanged as it specifically targets 'ESM-Flex'
    logging.info("Preparing data for Figure 1 (ESM-Flex scatter plot)...")
    esm_flex_pred_col = MODEL_PREDICTION_COLUMNS.get('ESM-Flex') # Use .get for safety
    if esm_flex_pred_col and esm_flex_pred_col in df.columns:
        # Ensure numeric conversion and dropna for plotting data too
        try:
             plot_data_df = df[[TARGET_COLUMN, esm_flex_pred_col]].copy()
             plot_data_df[TARGET_COLUMN] = pd.to_numeric(plot_data_df[TARGET_COLUMN], errors='coerce')
             plot_data_df[esm_flex_pred_col] = pd.to_numeric(plot_data_df[esm_flex_pred_col], errors='coerce')
             plot_data_df = plot_data_df.dropna()
        except Exception as e:
             logging.error(f"Error preparing plot data for ESM-Flex: {e}")
             plot_data_df = pd.DataFrame()

        if plot_data_df.empty:
            logging.warning("No valid numeric data points found for ESM-Flex scatter plot data.")
        else:
            # Rename columns for clarity in the output file
            plot_data_df.columns = ['Actual_RMSF', 'Predicted_RMSF_ESM_Flex']
            try:
                plot_data_df.to_csv(PLOT_DATA_OUTPUT_CSV, index=False, float_format='%.6f')
                logging.info(f"Data for Figure 1 saved to: {PLOT_DATA_OUTPUT_CSV} ({len(plot_data_df)} rows)")
            except Exception as e:
                logging.error(f"Error saving plot data CSV: {e}")
    else:
        logging.warning(f"ESM-Flex prediction column ('{esm_flex_pred_col}') not found or not defined. Skipping Figure 1 data export.")

    logging.info("Script finished.")