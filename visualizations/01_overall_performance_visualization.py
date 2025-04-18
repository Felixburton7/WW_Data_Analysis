# /home/s_felix/FINAL_PROJECT/Data_Analysis/visualizations/01_overall_performance_visualization.py
# Version 2: Optimized KDE calculation using subsampling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import logging
import time # To measure KDE time

# --- Configuration ---
# Paths relative to the script's location (visualizations/)
PLOT_DATA_CSV = "../results_output/summary_data/figure1_plot_data.csv"
METRICS_CSV = "../results_output/tables/table1_overall_metrics.csv"
OUTPUT_DIR_FIGURES = "../results_output/figures"
FIGURE_OUTPUT_BASENAME = os.path.join(OUTPUT_DIR_FIGURES, "figure1_overall_scatter")

# Plotting settings
CMAP = 'viridis' # Colormap for density ('viridis', 'plasma', 'inferno', 'magma')
POINT_SIZE = 5   # Size of the scatter points
POINT_ALPHA = 0.3 # Transparency of the points (consider slightly lower if KDE fails)
FIG_DPI = 300

# --- KDE Optimization ---
# Calculate KDE on a subset if data is large to improve speed
KDE_SUBSET_SIZE = 30000 # Adjust as needed (e.g., 20000-50000)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting overall performance visualization script (Figure 1 - Optimized KDE).")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR_FIGURES, exist_ok=True)
    logging.info(f"Output directory checked/created: {OUTPUT_DIR_FIGURES}")

    # --- Load Data ---
    logging.info(f"Loading plot data from: {PLOT_DATA_CSV}")
    try:
        plot_df = pd.read_csv(PLOT_DATA_CSV)
        logging.info(f"Plot data loaded. Shape: {plot_df.shape}")
    except FileNotFoundError:
        logging.error(f"Plot data file not found: {PLOT_DATA_CSV}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading plot data CSV: {e}")
        exit(1)

    logging.info(f"Loading metrics from: {METRICS_CSV}")
    try:
        metrics_df = pd.read_csv(METRICS_CSV)
        esm_flex_metrics = metrics_df[metrics_df['Model'] == 'ESM-Flex'].iloc[0]
        logging.info("ESM-Flex metrics loaded.")
    except FileNotFoundError:
        logging.error(f"Metrics file not found: {METRICS_CSV}")
        exit(1)
    except IndexError:
        logging.error(f"Could not find 'ESM-Flex' model in metrics file: {METRICS_CSV}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading metrics CSV: {e}")
        exit(1)

    # Extract data for plotting
    x_all = plot_df['Actual_RMSF'].values
    y_all = plot_df['Predicted_RMSF_ESM_Flex'].values
    total_points = len(x_all)

    # --- Calculate Density using Optimized KDE ---
    logging.info("Calculating point density using Gaussian KDE...")
    z_all = None
    kde_start_time = time.time()
    try:
        xy_all = np.vstack([x_all, y_all])

        if total_points > KDE_SUBSET_SIZE:
            logging.info(f"Data size ({total_points}) > {KDE_SUBSET_SIZE}. Subsampling for KDE calculation.")
            subset_indices = np.random.choice(total_points, KDE_SUBSET_SIZE, replace=False)
            xy_subset = xy_all[:, subset_indices]
            logging.info(f"Fitting KDE on subset of size {KDE_SUBSET_SIZE}...")
            kde = gaussian_kde(xy_subset)
            logging.info("KDE fitted. Evaluating on all points...")
            z_all = kde(xy_all) # Evaluate the fitted KDE on *all* points
        else:
            logging.info("Fitting KDE on all points...")
            kde = gaussian_kde(xy_all)
            z_all = kde(xy_all)

        # Sort points by density, so dense points are plotted last (on top)
        idx = z_all.argsort()
        x_plot, y_plot, z_plot = x_all[idx], y_all[idx], z_all[idx]
        kde_end_time = time.time()
        logging.info(f"Density calculation complete (took {kde_end_time - kde_start_time:.2f} seconds).")

    except Exception as e:
        logging.error(f"Error during KDE calculation: {e}. Proceeding without density coloring.")
        # Fallback: Use original unsorted data for plotting without density
        x_plot, y_plot, z_plot = x_all, y_all, None


    # --- Create Plot ---
    logging.info("Generating plot...")
    plt.style.use('seaborn-v0_8-paper') # Use a style suitable for papers
    fig, ax = plt.subplots(figsize=(5, 5)) # Square figure

    # Density Scatter Plot
    if z_plot is not None:
        scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=POINT_SIZE, alpha=POINT_ALPHA, cmap=CMAP, edgecolors='none', rasterized=True)
        # Optional: Add a colorbar (might make plot busy, consider omitting for final version)
        # cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Point Density', fontsize=10)
        # cbar.ax.tick_params(labelsize=8)
    else:
        # Fallback scatter if KDE fails or wasn't calculated
         logging.warning("Plotting without density coloring due to KDE error or data size.")
         # Plot all points if KDE failed, could still be slow for rendering if many points
         ax.scatter(x_plot, y_plot, s=POINT_SIZE, alpha=POINT_ALPHA, c='blue', edgecolors='none', rasterized=True)


    # Plot y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # Extend limits slightly for better visualization, ensure start at 0
    lims = [0, lims[1] * 1.05]
    ax.plot(lims, lims, '--', color='grey', alpha=0.7, zorder=1, label='y=x')

    # Set labels and title
    ax.set_xlabel("Actual RMSF (Å)", fontsize=12)
    ax.set_ylabel("Predicted RMSF (ESM-Flex) (Å)", fontsize=12)

    # Set limits and aspect ratio based on bulk of data
    # Limit based on 99.5th percentile to avoid extreme outliers dominating scale
    plot_max_lim_x = np.percentile(x_all, 99.5) if len(x_all) > 0 else 3.0
    plot_max_lim_y = np.percentile(y_all, 99.5) if len(y_all) > 0 else 3.0
    plot_max_lim = max(plot_max_lim_x, plot_max_lim_y)
    plot_max_lim = min(plot_max_lim, 3.5) # Cap max limit if needed, adjust if your data has higher RMSF often
    ax.set_xlim([0, plot_max_lim])
    ax.set_ylim([0, plot_max_lim])
    ax.set_aspect('equal', adjustable='box')

    # Add metric annotations (rounded to 2 decimal places)
    pcc_val = esm_flex_metrics['PCC']
    r2_val = esm_flex_metrics['R2']
    mae_val = esm_flex_metrics['MAE']
    rmse_val = esm_flex_metrics['RMSE']
    count_val = int(esm_flex_metrics['Count'])

    # Use standard rounding for 2 decimal places
    pcc_str = f"{pcc_val:.2f}"
    r2_str = f"{r2_val:.2f}"
    mae_str = f"{mae_val:.2f}"
    rmse_str = f"{rmse_val:.2f}"

    annotation_text = (
        f"PCC = {pcc_str}\n"
        f"R² = {r2_str}\n"
        f"MAE = {mae_str} Å\n"
        f"RMSE = {rmse_str} Å\n"
        f"N = {count_val:,}"
    )
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey'))

    # Aesthetics
    ax.tick_params(axis='both', which='major', labelsize=10)
    # Despine (remove top and right borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # --- Save Plot ---
    try:
        png_file = f"{FIGURE_OUTPUT_BASENAME}.png"
        pdf_file = f"{FIGURE_OUTPUT_BASENAME}.pdf"
        fig.savefig(png_file, dpi=FIG_DPI, bbox_inches='tight')
        fig.savefig(pdf_file, bbox_inches='tight')
        logging.info(f"Figure saved successfully to {png_file} and {pdf_file}")
    except Exception as e:
        logging.error(f"Error saving figure: {e}")

    # plt.show() # Uncomment to display plot interactively

    logging.info("Visualization script finished.")