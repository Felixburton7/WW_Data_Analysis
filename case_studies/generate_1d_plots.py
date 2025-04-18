import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import pearsonr
import warnings

# --- Configuration ---
INPUT_DIR = '.' # Run script from the case_studies directory
CSV_FILENAME = '1h3eA03.csv'
OUTPUT_DIR = 'rmsf_1d_plots_refined' # New folder for refined plots

# Column Names from Header
DOMAIN_COL = 'domain_id'
TEMP_COL = 'temperature'
RESID_COL = 'resid'
ACTUAL_COL = 'rmsf'
PRED_COL = 'Attention_ESM_rmsf' # Confirmed name

TARGET_TEMPS = [320.0, 348.0, 379.0, 413.0, 450.0]

# Plotting Settings
PRED_COLOR = 'purple'
# Changed to a brighter red
ACTUAL_COLOR = '#E32D00' # Bright Red (was '#D95319')
# Slightly thicker lines might enhance jaggedness perception
LINEWIDTH = 1.7 # Increased slightly from 1.5
# Slightly reduced alpha might make segments pop a bit more
ALPHA = 0.85 # Increased slightly from 0.8 for visibility

plt.style.use('seaborn-v0_8-white') # Clean style, no grid
plt.rcParams.update({'font.size': 11, # Slightly larger base font
                     'axes.labelsize': 12, # Larger axis labels
                     'xtick.labelsize': 10,
                     'ytick.labelsize': 10,
                     'legend.fontsize': 11, # Larger legend font
                     'axes.linewidth': 1.2}) # Make axes lines slightly thicker

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

# --- Main Script ---

def main():
    """Reads the CSV, filters by temperature, and generates refined 1D plots."""
    create_output_dir(OUTPUT_DIR) # Save to new directory
    csv_path = os.path.join(INPUT_DIR, CSV_FILENAME)
    domain_id_base = CSV_FILENAME.replace('.csv', '')

    if not os.path.exists(csv_path):
        print(f"ERROR: Input CSV file not found: {csv_path}. Exiting.", file=sys.stderr)
        return

    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=0, low_memory=False)
        # Basic validation
        required_cols = [TEMP_COL, RESID_COL, ACTUAL_COL, PRED_COL]
        if not all(col in df.columns for col in required_cols):
            print(f"ERROR: CSV {csv_path} is missing required columns: {required_cols}. Exiting.", file=sys.stderr)
            return
        # Convert necessary columns to numeric
        df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors='coerce')
        df[RESID_COL] = pd.to_numeric(df[RESID_COL], errors='coerce')
        df[ACTUAL_COL] = pd.to_numeric(df[ACTUAL_COL], errors='coerce')
        df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors='coerce')
        df.dropna(subset=required_cols, inplace=True) # Drop rows with NaNs in essential columns
        df[RESID_COL] = df[RESID_COL].astype(int)

    except Exception as e:
        print(f"ERROR: Failed to read or process CSV {csv_path}: {e}", file=sys.stderr)
        return

    print(f"Generating plots for {domain_id_base}...")

    for temp in TARGET_TEMPS:
        print(f"  Processing Temperature: {temp:.1f}K")
        temp_str = f"{int(temp)}K"
        # Save refined plots to the new directory
        output_png_path = os.path.join(OUTPUT_DIR, f"{domain_id_base}_{temp_str}_rmsf_plot_refined.png")

        # Filter data for the current temperature using floating point comparison
        temp_df = df[np.isclose(df[TEMP_COL], temp)].copy()

        if temp_df.empty:
            print(f"  WARN: No data found for temperature {temp:.1f}K. Skipping plot.", file=sys.stderr)
            continue

        # Sort by residue ID to ensure lines connect correctly
        temp_df.sort_values(by=RESID_COL, inplace=True)

        if len(temp_df) < 3:
            print(f"  WARN: Not enough data points ({len(temp_df)}) for {temp:.1f}K after filtering/cleaning. Skipping plot.", file=sys.stderr)
            continue

        resids = temp_df[RESID_COL].values
        actual_rmsf = temp_df[ACTUAL_COL].values
        pred_rmsf = temp_df[PRED_COL].values

        # Calculate PCC for this temperature
        try:
            # Ensure there's variance in both arrays before calculating PCC
            if np.std(pred_rmsf) > 1e-9 and np.std(actual_rmsf) > 1e-9:
                 pcc, _ = pearsonr(pred_rmsf, actual_rmsf)
                 pcc_text = f"PCC={pcc:.3f}"
            else:
                 pcc_text = "PCC=N/A (No variance)"
                 print(f"  WARN: Could not calculate PCC for {temp:.1f}K due to zero variance.", file=sys.stderr)
        except ValueError:
            pcc_text = "PCC=N/A" # Handle other calculation errors
            print(f"  WARN: Could not calculate PCC for {temp:.1f}K.", file=sys.stderr)


        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(6, 3.5)) # Slightly taller to accommodate legend better

        # Plot lines with slightly increased width and adjusted alpha
        ax.plot(resids, pred_rmsf, color=PRED_COLOR, linewidth=LINEWIDTH, label='DeepFlex', alpha=ALPHA)
        ax.plot(resids, actual_rmsf, color=ACTUAL_COLOR, linewidth=LINEWIDTH, label='Actual (MD)', alpha=ALPHA)

        # Labels and Title
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("RMSF (Å)")

        # Y-axis limit (start from 0, add padding to max)
        max_val = max(actual_rmsf.max(), pred_rmsf.max()) * 1.1 # Increased padding slightly
        ax.set_ylim(bottom=-0.05, top=max_val) # Start slightly below 0 for visual clarity
        ax.set_xlim(left=resids.min(), right=resids.max())

        # Legend (Aligned with top axis, centered)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), # Adjusted y anchor
                  ncol=2, frameon=False, fontsize=11) # Match PCC font size

        # PCC Annotation (Top left corner, aligned with legend)
        # ax.text(0.02, 1.05, pcc_text, transform=ax.transAxes, # Positioned top-left relative to axes
        #         fontsize=11, # Match legend font size
        #         verticalalignment='bottom', horizontalalignment='left',
        #         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='none', alpha=0.0)) # Transparent box

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Adjust tick parameters for clarity
        ax.tick_params(axis='both', which='major', length=4, width=1.2)

        # --- Save and Close ---
        plt.tight_layout(rect=[0, 0.0, 1, 0.92]) # Adjust bottom/top to prevent cutoff
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight') # Use bbox_inches='tight'
        plt.close(fig)
        print(f"  Plot saved to: {output_png_path}")


    print("\n--- Script Finished ---")

# --- Run Script ---
if __name__ == "__main__":
    main()