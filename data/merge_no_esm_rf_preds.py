# merge_no_esm_rf_preds.py
import pandas as pd
import logging
import os
import numpy as np
import sys

# --- Configuration: Define Paths Here ---
MAIN_ANALYSIS_CSV = "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv"
PREDICTION_CSV = "/home/s_felix/FINAL_PROJECT/Data_Analysis/data/aggregated_holdout_no_esm_RF_preds.csv"
# Output CSV will overwrite the main analysis file by default
OUTPUT_CSV = MAIN_ANALYSIS_CSV
# Name for the new column to be added
NEW_COLUMN_NAME = "No_ESM_RF_prediction"
# --- End Configuration ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_single_column(
    main_file_path: str,
    predictions_file_path: str,
    output_file_path: str,
    new_col_name: str,
    pred_col_to_merge: str = 'No_ESM_RF_prediction' # Column name in the prediction file
    ):
    """
    Merges a single prediction column from a prediction file into a main
    analysis CSV file based on domain_id, resid, and temperature.

    Args:
        main_file_path (str): Path to the main analysis CSV file to be updated.
        predictions_file_path (str): Path to the prediction CSV file containing the
                                     prediction column to merge.
        output_file_path (str): Path to save the final aggregated CSV file.
        new_col_name (str): Name for the new prediction column in the main file.
        pred_col_to_merge (str): The name of the prediction column in the
                                 predictions_file_path.
    """
    logger.info(f"Loading main analysis file: {main_file_path}")
    try:
        main_df = pd.read_csv(main_file_path)
        logger.info(f"Loaded main dataset with shape: {main_df.shape}")
        # Ensure required columns and types for merging
        req_cols = ['domain_id', 'resid', 'temperature']
        if not all(col in main_df.columns for col in req_cols):
             raise ValueError(f"Main CSV must contain columns: {req_cols}")
        main_df['domain_id'] = main_df['domain_id'].astype(str)
        main_df['resid'] = main_df['resid'].astype(int)
        main_df['temperature'] = main_df['temperature'].astype(float)
        # Set index for efficient update
        main_df = main_df.set_index(['domain_id', 'resid', 'temperature'], drop=False)
        if not main_df.index.is_unique:
             logger.warning("Duplicate domain_id/resid/temperature combinations found in main file! "
                            "Merge will affect first occurrence only when updating.")
    except FileNotFoundError:
        logger.error(f"Main analysis file not found: {main_file_path}")
        sys.exit(1)
    except ValueError as ve:
         logger.error(f"Error in main file structure: {ve}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading main file: {e}")
        sys.exit(1)

    logger.info(f"Loading predictions file: {predictions_file_path}")
    try:
        pred_df = pd.read_csv(predictions_file_path)
        logger.info(f"Loaded predictions dataset with shape: {pred_df.shape}")
        # Ensure required columns and types for merging
        req_cols_pred = ['domain_id', 'resid', 'temperature', pred_col_to_merge]
        if not all(col in pred_df.columns for col in req_cols_pred):
             raise ValueError(f"Prediction CSV must contain columns: {req_cols_pred}")
        pred_df['domain_id'] = pred_df['domain_id'].astype(str)
        pred_df['resid'] = pred_df['resid'].astype(int)
        pred_df['temperature'] = pred_df['temperature'].astype(float)
        # Set index for efficient lookup
        pred_df = pred_df.set_index(['domain_id', 'resid', 'temperature'])
        if not pred_df.index.is_unique:
             logger.warning(f"Duplicate keys found in prediction file '{predictions_file_path}'. Keeping first entry.")
             pred_df = pred_df[~pred_df.index.duplicated(keep='first')]

    except FileNotFoundError:
        logger.error(f"Predictions file not found: {predictions_file_path}")
        sys.exit(1)
    except ValueError as ve:
         logger.error(f"Error in prediction file structure: {ve}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading predictions file: {e}")
        sys.exit(1)

    # --- Perform Merge/Update ---
    logger.info(f"Merging column '{pred_col_to_merge}' from predictions as '{new_col_name}' into main dataset...")

    # Find common indices
    common_index = main_df.index.intersection(pred_df.index)
    num_common_rows = len(common_index)

    if num_common_rows == 0:
        logger.error("No matching entries found between main dataset and prediction file based on domain_id, resid, and temperature. Cannot merge.")
        sys.exit(1)

    logger.info(f"Found {num_common_rows} matching entries to update.")

    # Update the main dataframe's new column using the common index
    # This assigns values from pred_df to main_df only where the multi-index matches
    main_df[new_col_name] = pred_df.loc[common_index, pred_col_to_merge]

    # --- Final Check and Save ---
    # Reset index before saving
    main_df = main_df.reset_index(drop=True)

    final_nan_count = main_df[new_col_name].isna().sum()
    if final_nan_count > 0:
        logger.warning(f"The final merged file has {final_nan_count} rows where '{new_col_name}' is NaN. "
                       "This means some entries in the main file didn't have a corresponding prediction entry.")

    logger.info(f"Saving merged dataset to: {output_file_path}")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        # Save the updated main DataFrame
        main_df.to_csv(output_file_path, index=False, float_format='%.8f') # Use precision
        logger.info("Merge complete and file saved successfully.")
    except Exception as e:
        logger.error(f"Error saving merged CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting script to merge No-ESM RF predictions...")

    # Extract the specific column name from the source prediction file
    # Assuming the column name in aggregated_holdout_no_esm_RF_preds.csv is
    # exactly 'MinimalRF_predicted_rmsf' as generated previously.
    # If it's different, change it here.
    source_prediction_column = "No_ESM_RF_prediction"

    # Call the function directly with the configured paths and column names
    merge_single_column(
        main_file_path=MAIN_ANALYSIS_CSV,
        predictions_file_path=PREDICTION_CSV,
        output_file_path=OUTPUT_CSV, # This will overwrite the main analysis file
        new_col_name=NEW_COLUMN_NAME,
        pred_col_to_merge=source_prediction_column
    )
    logger.info("Script finished.")