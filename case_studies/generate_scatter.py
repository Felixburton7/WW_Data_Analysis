import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors # No longer needed
import os
import glob
from scipy.stats import pearsonr, linregress # gaussian_kde no longer needed
import warnings

# --- Configuration ---
INPUT_DIR = '.' # Run script from the case_studies directory
OUTPUT_DIR = 'scatter_plots'
ACTUAL_COL = 'rmsf'
PRED_COL = 'Attention_ESM_rmsf'
PRED_AXIS_LABEL = 'DeepFlex RMSF Prediction'

# Plotting Style
plt.style.use('seaborn-v0_8-white') # Use a style without gridlines by default
plt.rcParams.update({'font.size': 10,
                     'axes.labelsize': 11,
                     'xtick.labelsize': 9,
                     'ytick.labelsize': 9,
                     'legend.fontsize': 10})

# --- Helper Functions ---

def create_output_dir(dir_path):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created output directory: {dir_path}")
        except OSError as e:
            print(f"ERROR: Could not create directory {dir_path}: {e}", file=sys.stderr)
            sys.exit(1)

# Removed plot_scatter_density function

# --- Main Script ---

def main():
    """Finds CSVs, processes them, and generates scatter plots."""
    create_output_dir(OUTPUT_DIR)

    # Find all csv files in the current directory
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        print(f"ERROR: No CSV files found in '{INPUT_DIR}'. Exiting.", file=sys.stderr)
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    for csv_file in csv_files:
        domain_id = os.path.basename(csv_file).replace('.csv', '')
        print(f"\nProcessing: {domain_id}")
        output_png_path = os.path.join(OUTPUT_DIR, f"{domain_id}_scatter.png")

        try:
            # Read CSV, assuming header is present
            df = pd.read_csv(csv_file, header=0, low_memory=False)

            # Check for required columns
            if ACTUAL_COL not in df.columns or PRED_COL not in df.columns:
                print(f"  ERROR: Missing required columns ('{ACTUAL_COL}', '{PRED_COL}') in {csv_file}. Skipping.", file=sys.stderr)
                continue

            # Convert columns to numeric and drop rows with NaNs in essential columns
            df[ACTUAL_COL] = pd.to_numeric(df[ACTUAL_COL], errors='coerce')
            df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors='coerce')
            df_clean = df.dropna(subset=[ACTUAL_COL, PRED_COL]).copy()

            if len(df_clean) < 3: # Need at least 3 points for correlation/regression
                print(f"  ERROR: Not enough valid data points ({len(df_clean)}) after cleaning in {csv_file}. Skipping.", file=sys.stderr)
                continue

            x_data = df_clean[PRED_COL].values
            y_data = df_clean[ACTUAL_COL].values

            # --- Calculations ---
            # Pearson Correlation
            pcc, p_value = pearsonr(x_data, y_data)

            # Linear Regression for line of best fit
            slope, intercept, r_val_unused, p_val_unused, std_err_unused = linregress(x_data, y_data)
            # Calculate line points based on axis limits for full span
            plot_min_for_line = 0
            plot_max_for_line = max(x_data.max(), y_data.max()) * 1.05
            x_line = np.array([plot_min_for_line, plot_max_for_line])
            y_line = slope * x_line + intercept


            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(5, 5)) # Square aspect ratio

            # Simple Scatter Plot (Dark Purple)
            # Using a specific dark purple hex code, adjust alpha for visibility
            ax.scatter(x_data, y_data, s=8, alpha=0.55, color='#4B0082', # Indigo/Dark Purple
                       edgecolor='none', rasterized=True)

            # Line of Best Fit
            ax.plot(x_line, y_line, color='red', linestyle='-', linewidth=1.5, label='Best Fit')

            # Identity Line (y=x)
            lim_min = 0 # Start from 0
            lim_max = plot_max_for_line # Use calculated max
            ax.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle='--', linewidth=1, label='y=x')

            # Axis Labels
            ax.set_xlabel(f"{PRED_AXIS_LABEL} (Å)")
            ax.set_ylabel("Actual RMSF (Å)")

            # Set Limits
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)

            # Aspect Ratio
            ax.set_aspect('equal', adjustable='box')

            # Turn off gridlines explicitly
            ax.grid(False)

            # Annotations
            # PCC top-left, slightly lower, LARGER font size, NO bounding box
            # ax.text(0.05, 0.90, f"PCC = {pcc:.3f}", transform=ax.transAxes,
            #         fontsize=12, # Increased font size to 12
            #         verticalalignment='top', horizontalalignment='left')

            # Domain ID middle-right, MORE padding in bounding box
            ax.text(0.95, 0.5, domain_id, transform=ax.transAxes,
                    fontsize=12, verticalalignment='center', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)) # pad=0.5

            # --- Save and Close ---
            plt.tight_layout()
            plt.savefig(output_png_path, dpi=300)
            plt.close(fig)
            print(f"  Plot saved to: {output_png_path}")

        except Exception as e:
            print(f"  ERROR: Failed to process or plot {csv_file}: {e}", file=sys.stderr)
            plt.close('all') # Close any potentially open plot

    print("\n--- Script Finished ---")

# --- Run Script ---
if __name__ == "__main__":
    main()