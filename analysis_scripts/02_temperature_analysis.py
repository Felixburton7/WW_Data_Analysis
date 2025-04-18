# /home/s_felix/FINAL_PROJECT/Data_Analysis/analysis_scripts/02_temperature_analysis.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import os
import logging

# --- Configuration ---
INPUT_CSV = "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv"
OUTPUT_DIR_TABLES = "results_output/tables"
OUTPUT_DIR_SUMMARY = "results_output/summary_data"
METRICS_BY_TEMP_OUTPUT_CSV = os.path.join(OUTPUT_DIR_TABLES, "table2_metrics_by_temp.csv")
PLOT_DATA_OUTPUT_CSV = os.path.join(OUTPUT_DIR_SUMMARY, "figure2_plot_data.csv")

TARGET_COLUMN = 'rmsf'
TEMPERATURE_COLUMN = 'temperature'

# Models to analyze by temperature and their corresponding column names
# Using RF as the feature-matched baseline as requested
MODELS_TO_ANALYZE = {
    'ESM-Flex': 'Attention_ESM_rmsf',
    'RF': 'ensembleflex_RF_rmsf',
    'ESM-Only': 'esm_rmsf'
}

TEMPERATURES_TO_EXPECT = [320.0, 348.0, 379.0, 413.0, 450.0] # Expected temperatures

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function (Simplified for PCC & MAE) ---
def calculate_temp_metrics(y_true, y_pred):
    """Calculates PCC and MAE."""
    metrics = {'PCC': np.nan, 'MAE': np.nan, 'Count': len(y_true)}
    if len(y_true) < 2:
        logging.warning(f"Skipping PCC calculation due to insufficient data points ({len(y_true)}).")
        if len(y_true) > 0:
            metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        return metrics

    try:
        metrics['PCC'] = pearsonr(y_true, y_pred)[0]
    except ValueError as e:
        logging.warning(f"Could not calculate PCC: {e}")

    try:
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Could not calculate MAE: {e}")

    return metrics

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting temperature analysis script.")

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SUMMARY, exist_ok=True)
    logging.info(f"Output directories checked/created: {OUTPUT_DIR_TABLES}, {OUTPUT_DIR_SUMMARY}")

    # Load data
    logging.info(f"Loading data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_CSV}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        exit(1)

    # --- Calculate Metrics per Temperature per Model ---
    all_temp_results = []
    plot_data_list = [] # To store data for Figure 2

    logging.info("Calculating performance metrics for each model at each temperature...")

    # Ensure the temperature column is float for accurate comparison
    if df[TEMPERATURE_COLUMN].dtype != 'float64':
         try:
             df[TEMPERATURE_COLUMN] = df[TEMPERATURE_COLUMN].astype(float)
         except ValueError:
             logging.error(f"Could not convert {TEMPERATURE_COLUMN} column to float. Check data.")
             exit(1)

    grouped = df.groupby(TEMPERATURE_COLUMN)
    actual_temperatures = sorted(grouped.groups.keys())

    if set(actual_temperatures) != set(TEMPERATURES_TO_EXPECT):
        logging.warning(f"Temperatures found in data {actual_temperatures} differ from expected {TEMPERATURES_TO_EXPECT}")

    for temp, group_df in grouped:
        logging.info(f"Processing Temperature: {temp} K")
        temp_metrics_for_plot = {'Temperature': temp}

        for model_name, pred_col in MODELS_TO_ANALYZE.items():
            logging.debug(f"  Model: {model_name}")

            # Filter for rows where both target and prediction are valid (pairwise)
            valid_df = group_df[[TARGET_COLUMN, pred_col]].dropna()

            if valid_df.empty:
                logging.warning(f"  No valid data points found for model {model_name} at {temp}K.")
                metrics = {'PCC': np.nan, 'MAE': np.nan, 'Count': 0}
            else:
                y_true = valid_df[TARGET_COLUMN].values
                y_pred = valid_df[pred_col].values
                logging.debug(f"    Found {len(y_true)} valid data points.")

                # Calculate metrics
                metrics = calculate_temp_metrics(y_true, y_pred)

            # Store results for the main table
            result_row = {
                'Temperature': temp,
                'Model': model_name,
                'PCC': metrics['PCC'],
                'MAE': metrics['MAE'],
                'Count': metrics['Count']
            }
            all_temp_results.append(result_row)

            # Store PCC for the plot data
            temp_metrics_for_plot[f'{model_name}_PCC'] = metrics['PCC']
            # Store MAE for the plot data as well (might be useful for another plot later)
            temp_metrics_for_plot[f'{model_name}_MAE'] = metrics['MAE']

        plot_data_list.append(temp_metrics_for_plot)


    # --- Format and Save Results ---
    # Main metrics table
    results_df = pd.DataFrame(all_temp_results)
    # Pivot for better readability if desired, or keep long format
    # Example pivot:
    try:
        pivot_table = results_df.pivot(index='Temperature', columns='Model', values=['PCC', 'MAE', 'Count'])
        # Flatten multi-index columns for easier CSV saving
        pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]
        pivot_table.reset_index(inplace=True)
        logging.info("Pivoted table created.")
    except Exception as e:
        logging.warning(f"Could not pivot table: {e}. Saving in long format.")
        pivot_table = results_df # Fallback to long format

    try:
        pivot_table.to_csv(METRICS_BY_TEMP_OUTPUT_CSV, index=False, float_format='%.6f')
        logging.info(f"Temperature metrics saved to: {METRICS_BY_TEMP_OUTPUT_CSV}")
        print("\n--- Metrics by Temperature ---")
        print(pivot_table.round(4).to_string(index=False)) # Print rounded version
        print("------------------------------\n")
    except Exception as e:
        logging.error(f"Error saving temperature metrics CSV: {e}")


    # Data for Figure 2 plot
    plot_df = pd.DataFrame(plot_data_list)
    plot_df = plot_df.sort_values(by='Temperature') # Ensure temperature order
    try:
        plot_df.to_csv(PLOT_DATA_OUTPUT_CSV, index=False, float_format='%.6f')
        logging.info(f"Data for Figure 2 saved to: {PLOT_DATA_OUTPUT_CSV}")
    except Exception as e:
        logging.error(f"Error saving plot data CSV: {e}")


    logging.info("Temperature analysis script finished.")