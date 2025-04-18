#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 2: Temperature-specific RMSF Distribution Comparisons and Domain-Level Density Plot
Generates 6 separate files: 5 distribution plots (one per temperature) and 1 density scatter plot
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import logging
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# --- Configuration ---
# Make sure these paths are correct for your system
DATA_PATH = '/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv'
OUTPUT_DIR = '/home/s_felix/FINAL_PROJECT/Data_Analysis/results_output'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
LOG_PATH = os.path.join(OUTPUT_DIR, 'fig2_generation.log')

# Plotting parameters
TEMP_COLORS = {
    320.0: '#313695',  # Cool blue
    348.0: '#4575b4',  # Blue
    379.0: '#74add1',  # Light blue
    413.0: '#f46d43',  # Orange
    450.0: '#d73027'   # Red
}
DIST_COLORS = {
    'actual': '#E41A1C',    # Red for actual
    'predicted': '#377EB8'  # Blue for predicted
}
FIG_DPI = 300
KDE_SUBSET_SIZE = 30000  # For density calculation in scatter plot
NUM_BINS = 100

# --- Logging Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
# Clear previous log file if it exists
if os.path.exists(LOG_PATH):
    try:
        open(LOG_PATH, 'w').close()
    except OSError as e:
        print(f"Warning: Could not clear log file {LOG_PATH}: {e}") # Use print as logging might not be set up
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# Use a publication-quality font that's commonly available
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans'] # Prioritize available ones
except Exception as e:
    logging.warning(f"Could not set preferred sans-serif fonts: {e}. Matplotlib will use defaults.")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
# Note: constrained_layout can sometimes cause issues or warnings.
# If layout problems occur, consider removing these or using fig.tight_layout() explicitly before saving.
# plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['figure.constrained_layout.h_pad'] = 0.1
# plt.rcParams['figure.constrained_layout.w_pad'] = 0.1
# plt.rcParams['figure.constrained_layout.hspace'] = 0.05
# plt.rcParams['figure.constrained_layout.wspace'] = 0.05

def setup_hist_kde_plot(ax, temp):
    """Configure axes for histogram+KDE plots"""
    ax.set_xlabel('RMSF (Å)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cα Residue Count', fontsize=11, fontweight='bold')
    # Format temperature for title (int if whole number, else float)
    temp_str = f"{int(temp)}" if temp == int(temp) else f"{temp}"
    ax.set_title(f'RMSF Distribution at {temp_str}K', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=9)

    # --- Only horizontal (y-axis) grid lines with adjusted alpha ---
    ax.grid(axis='y', linestyle='--', color='grey', alpha=0.25) # Less opaque grid

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend with improved positioning
    ax.legend(loc='upper right', frameon=True, fontsize=9,
             framealpha=0.9, edgecolor='lightgray')

    # Use integer ticks for y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    return ax

def generate_distribution_plots(df, temperatures):
    """Generate histogram+KDE plots for each temperature"""
    # --- Determine axis limits ---
    logging.info("Determining optimal axis limits for distribution plots...")
    present_temps = sorted(df['temperature'].unique())
    if not present_temps:
        logging.error("No temperature data found in the DataFrame for distribution plots.")
        return

    low_temps = [t for t in present_temps if t < 400.0]
    high_temps = [t for t in present_temps if t >= 400.0]

    def safe_quantile(series, q):
        return series.quantile(q) if not series.empty else 0

    x_max_low, x_max_high = 2.0, 2.0 # Initialize defaults

    if low_temps:
        low_temp_df = df[df['temperature'].isin(low_temps)].dropna(subset=['rmsf', 'Attention_ESM_rmsf'])
        if not low_temp_df.empty:
            max_rmsf_low = max(
                safe_quantile(low_temp_df['rmsf'], 0.995),
                safe_quantile(low_temp_df['Attention_ESM_rmsf'], 0.995)
            )
            x_max_low = min(max(max_rmsf_low * 1.05, 0.5), 2.0)
        else:
             logging.warning("No valid data for low temperatures (<400K) to determine x-axis limit.")
    else:
        logging.info("No temperatures below 400K found.")

    if high_temps:
        high_temp_df = df[df['temperature'].isin(high_temps)].dropna(subset=['rmsf', 'Attention_ESM_rmsf'])
        if not high_temp_df.empty:
            max_rmsf_high = max(
                safe_quantile(high_temp_df['rmsf'], 0.995),
                safe_quantile(high_temp_df['Attention_ESM_rmsf'], 0.995)
            )
            x_max_high = min(max(max_rmsf_high * 1.1, x_max_low), 3.0)
        else:
            logging.warning("No valid data for high temperatures (>=400K) to determine x-axis limit.")
            x_max_high = x_max_low
    else:
        logging.info("No temperatures >= 400K found.")
        x_max_high = x_max_low

    logging.info(f"Low temp (<400K) x-max: {x_max_low:.2f}, High temp (>=400K) x-max: {x_max_high:.2f}")

    # --- Calculate y-limits ---
    y_limits = {}
    max_count_overall = 0
    for temp in present_temps:
        temp_df = df[df['temperature'] == temp].dropna(subset=['rmsf', 'Attention_ESM_rmsf'])
        if temp_df.empty:
            y_limits[temp] = 10
            continue

        x_max = x_max_high if temp >= 400.0 else x_max_low
        bins = np.linspace(0, x_max, NUM_BINS + 1)
        actual_counts, _ = np.histogram(temp_df['rmsf'], bins=bins)
        pred_counts, _ = np.histogram(temp_df['Attention_ESM_rmsf'], bins=bins)
        max_count = max(np.max(actual_counts) if actual_counts.size > 0 else 0,
                        np.max(pred_counts) if pred_counts.size > 0 else 0)
        y_limits[temp] = max(max_count * 1.15, 10)
        max_count_overall = max(max_count_overall, max_count)

    logging.info("Finished calculating axis limits for distribution plots.")

    # --- Generate plots ---
    for temp in temperatures:
        if temp not in present_temps:
            logging.warning(f"Skipping plot generation for {temp}K as it's not present in the cleaned data.")
            continue

        temp_df = df[df['temperature'] == temp].dropna(subset=['rmsf', 'Attention_ESM_rmsf'])
        if temp_df.empty:
             logging.warning(f"Skipping plot generation for {temp}K due to lack of valid data points.")
             continue

        logging.info(f"Generating distribution plot for {temp}K with {len(temp_df)} records")
        fig, ax = plt.subplots(figsize=(6.5, 5))

        x_max = x_max_high if temp >= 400.0 else x_max_low
        bins = np.linspace(0, x_max, NUM_BINS + 1)
        actual_counts, actual_bins = np.histogram(temp_df['rmsf'], bins=bins)
        pred_counts, pred_bins = np.histogram(temp_df['Attention_ESM_rmsf'], bins=bins)

        if actual_bins.size < 2:
            logging.warning(f"Could not generate valid bins for histogram at {temp}K. Skipping plot.")
            plt.close(fig)
            continue

        bin_centers = (actual_bins[:-1] + actual_bins[1:]) / 2
        bin_width = actual_bins[1] - actual_bins[0]

                # Plot histograms with less obvious edges
        ax.bar(bin_centers, actual_counts, width=bin_width, alpha=0.45,
               color=DIST_COLORS['actual'], label='Actual MD RMSF',
               edgecolor='lightgrey', linewidth=0.3) # Changed edgecolor to lightgrey, slightly reduced linewidth
        ax.bar(bin_centers, pred_counts, width=bin_width, alpha=0.45,
               color=DIST_COLORS['predicted'], label='DeepFlex Predicted RMSF',
               edgecolor='lightgrey', linewidth=0.3) # Changed edgecolor to lightgrey, slightly reduced linewidth
        if len(temp_df['rmsf']) > 1 and len(temp_df['Attention_ESM_rmsf']) > 1:
            try:
                kde_x = np.linspace(0, x_max, 500)
                if np.std(temp_df['rmsf']) > 1e-9:
                    actual_kde = stats.gaussian_kde(temp_df['rmsf'])
                    actual_kde_density = actual_kde(kde_x)
                    max_actual_count = np.max(actual_counts) if actual_counts.size > 0 else 1
                    max_actual_kde = np.max(actual_kde_density) if np.max(actual_kde_density) > 0 else 1
                    actual_kde_y = actual_kde_density * max_actual_count / max_actual_kde
                    ax.plot(kde_x, actual_kde_y, color=DIST_COLORS['actual'], linewidth=2.0)
                else:
                    logging.warning(f"Skipping Actual KDE for {temp}K due to near-zero variance.")

                if np.std(temp_df['Attention_ESM_rmsf']) > 1e-9:
                    pred_kde = stats.gaussian_kde(temp_df['Attention_ESM_rmsf'])
                    pred_kde_density = pred_kde(kde_x)
                    max_pred_count = np.max(pred_counts) if pred_counts.size > 0 else 1
                    max_pred_kde = np.max(pred_kde_density) if np.max(pred_kde_density) > 0 else 1
                    pred_kde_y = pred_kde_density * max_pred_count / max_pred_kde
                    ax.plot(kde_x, pred_kde_y, color=DIST_COLORS['predicted'], linewidth=2.0)
                else:
                    logging.warning(f"Skipping Predicted KDE for {temp}K due to near-zero variance.")

            except Exception as e:
                logging.warning(f"Could not compute or plot KDE for {temp}K: {e}")
        else:
            logging.warning(f"Skipping KDE for {temp}K due to insufficient data points (< 2) in actual or predicted RMSF.")

        ax.set_xlim(0, x_max)
        if temp in y_limits:
             ax.set_ylim(0, y_limits[temp])
        else:
             ax.set_ylim(0, 10) # Fallback y-limit

        setup_hist_kde_plot(ax, temp) # Pass float temp

        png_file = os.path.join(FIGURES_DIR, f"fig2_temperature_{int(temp)}K_distribution.png")
        try:
            # Use tight_layout for better spacing before saving
            fig.tight_layout(pad=1.0)
            fig.savefig(png_file, dpi=FIG_DPI, bbox_inches='tight')
            logging.info(f"Saved distribution plot to {png_file}")
        except Exception as e:
            logging.error(f"Failed to save plot {png_file}: {e}")

        plt.close(fig)

def generate_domain_level_density_plot(df, temperatures_with_colors):
    """
    Generate density scatter plot of domain-level mean RMSF values.
    Highlights means for temperatures specified in temperatures_with_colors.
    """
    logging.info("Generating domain-level density scatter plot...")

    # --- Aggregate data ---
    # Aggregate using *all* data in the passed df first
    if 'domain_id' not in df.columns:
        logging.error("Missing 'domain_id' column for domain-level aggregation.")
        return
    if df.empty:
        logging.error("Input DataFrame for domain plot is empty.")
        return

    logging.info(f"Aggregating domain-level data from {len(df)} records...")
    domain_df = df.groupby(['domain_id', 'temperature']).agg(
        rmsf=('rmsf', 'mean'),
        Attention_ESM_rmsf=('Attention_ESM_rmsf', 'mean')
    ).reset_index()

    domain_df = domain_df.dropna(subset=['rmsf', 'Attention_ESM_rmsf'])
    domain_df = domain_df[np.isfinite(domain_df['rmsf']) & np.isfinite(domain_df['Attention_ESM_rmsf'])]

    if domain_df.empty:
        logging.error("No valid domain-level data generated after aggregation and cleaning.")
        return
    logging.info(f"Generated valid domain-level data with {len(domain_df)} domain-temperature pairs")

    # --- Setup plot ---
    fig, ax = plt.subplots(figsize=(6.5, 6))

    x_all = domain_df['rmsf'].values
    y_all = domain_df['Attention_ESM_rmsf'].values
    temp_all = domain_df['temperature'].values

    # --- Density Calculation and Plotting ---
    scatter = None
    if len(x_all) > 1:
        logging.info("Calculating point density using Gaussian KDE...")
        try:
            xy_all = np.vstack([x_all, y_all])
            z_all = None

            if np.all(np.isfinite(xy_all)) and xy_all.shape[1] > 1:
                std_devs = np.std(xy_all, axis=1)
                if np.any(std_devs < 1e-9):
                    logging.warning("Data points have near-zero variance; KDE may be ill-defined. Using uniform color.")
                    z_all = np.ones(xy_all.shape[1])
                else:
                    if len(x_all) > KDE_SUBSET_SIZE:
                        logging.info(f"Subsampling for KDE calculation ({len(x_all)} > {KDE_SUBSET_SIZE})")
                        sample_size = min(KDE_SUBSET_SIZE, len(x_all))
                        subset_indices = np.random.choice(len(x_all), sample_size, replace=False)
                        xy_subset = xy_all[:, subset_indices]
                        kde = gaussian_kde(xy_subset)
                        z_all = kde(xy_all)
                    else:
                        kde = gaussian_kde(xy_all)
                        z_all = kde(xy_all)
            else:
                logging.warning("Insufficient data or non-finite values for KDE. Using uniform color.")
                z_all = np.ones(len(x_all))


            idx = z_all.argsort()
            x_plot, y_plot, z_plot = x_all[idx], y_all[idx], z_all[idx]
            temp_plot = temp_all[idx]

            global TEMP_COLORS
            TEMP_COLORS_float = {float(k): v for k, v in TEMP_COLORS.items()}

            scatter = ax.scatter(
                x_plot, y_plot, c=z_plot,
                s=15, alpha=0.7, cmap='viridis',
                edgecolors='none', rasterized=True
            )

            # Add temperature mean highlights for specified temperatures
            plotted_labels = set()
            logging.info(f"Adding highlights for temperatures: {temperatures_with_colors}")
            for temp_float in temperatures_with_colors: # Use the passed list for highlights
                temp_mask = np.isclose(temp_plot, temp_float)
                if np.any(temp_mask):
                    temp_x_mean = np.mean(x_plot[temp_mask])
                    temp_y_mean = np.mean(y_plot[temp_mask])
                    label = f"{int(temp_float)}K"
                    if temp_float in TEMP_COLORS_float and label not in plotted_labels:
                        ax.scatter(temp_x_mean, temp_y_mean, s=120,
                                   color=TEMP_COLORS_float[temp_float],
                                   edgecolors='black', linewidth=1.5, alpha=0.95,
                                   label=label, zorder=5)
                        plotted_labels.add(label)
                    elif temp_float not in TEMP_COLORS_float:
                         # This shouldn't happen if temperatures_with_colors is filtered correctly in main
                         logging.warning(f"Highlight requested for {temp_float} but color not defined.")

        except np.linalg.LinAlgError as e:
             logging.error(f"Linear algebra error during KDE: {e}. Falling back to simple scatter.")
             scatter = ax.scatter(x_all, y_all, s=15, alpha=0.6, c='grey', edgecolors='none')
        except ValueError as e:
             logging.error(f"Value error during KDE: {e}. Falling back to simple scatter.")
             scatter = ax.scatter(x_all, y_all, s=15, alpha=0.6, c='grey', edgecolors='none')
        except Exception as e:
            logging.error(f"Unexpected error during density/scatter plotting: {e}", exc_info=True)
            scatter = ax.scatter(x_all, y_all, s=15, alpha=0.6, c='grey', edgecolors='none')

    elif len(x_all) == 1:
         logging.warning("Only one data point for domain plot. Plotting single point.")
         scatter = ax.scatter(x_all, y_all, s=50, alpha=0.8, c='blue')
    else:
        # This case should have been caught by the check on domain_df earlier
        logging.warning("No valid data points to plot for domain-level scatter.")


    # --- Add plot elements ---
    min_val, max_val = 0, 1.0
    if len(x_all) > 0:
        min_val_data = min(x_all.min(), y_all.min())
        max_val_data = max(x_all.max(), y_all.max())
        buffer = (max_val_data - min_val_data) * 0.05
        min_val = max(0, min_val_data - buffer)
        max_val = max_val_data + buffer
        if np.isclose(min_val, max_val):
            min_val = max(0, min_val - 0.1)
            max_val = max_val + 0.1

    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1.0, zorder=0)
    ax.grid(alpha=0.3, linestyle='--')

    if len(x_all) > 1:
        try:
            pearson_r, p_value = stats.pearsonr(x_all, y_all)
            mae = np.mean(np.abs(x_all - y_all))
            rmse = np.sqrt(np.mean((x_all - y_all)**2))
            metrics_text = (
                f"Pearson r = {pearson_r:.3f}\n"
                f"MAE = {mae:.3f} Å\n"
                f"RMSE = {rmse:.3f} Å\n"
                f"N domains = {domain_df['domain_id'].nunique():,}"
            )
            ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='lightgray'))

        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            metrics_text = f"N domains = {domain_df['domain_id'].nunique():,}\n(Metrics error)"
            ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='lightgray'))

    ax.set_xlabel("Mean Actual RMSF (Å)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Predicted RMSF (Å)", fontsize=11, fontweight='bold')
    ax.set_title("DeepFlex: Domain-Level RMSF Prediction", fontsize=12, fontweight='bold')

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Temp Means", loc='lower right', fontsize=9,
                  framealpha=0.95, edgecolor='lightgray', title_fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    png_file = os.path.join(FIGURES_DIR, f"fig2_domain_level_density_scatter.png")
    try:
        fig.tight_layout(pad=1.0)
        fig.savefig(png_file, dpi=FIG_DPI, bbox_inches='tight')
        logging.info(f"Saved domain-level density plot to {png_file}")
    except Exception as e:
        logging.error(f"Failed to save domain plot {png_file}: {e}")

    plt.close(fig)


def main():
    logging.info(f"Starting Figure 2 generation")

    # --- Load Data ---
    logging.info(f"Loading data from {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        required_cols = ['rmsf', 'Attention_ESM_rmsf', 'temperature', 'domain_id']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             logging.error(f"Missing required columns in data file: {missing}. Exiting.")
             return

        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')

        initial_rows = len(df)
        df = df.dropna(subset=['rmsf', 'Attention_ESM_rmsf', 'temperature'])
        df = df[np.isfinite(df['rmsf']) & np.isfinite(df['Attention_ESM_rmsf']) & np.isfinite(df['temperature'])]
        rows_after_drop = len(df)
        logging.info(f"Loaded initial {initial_rows} rows. Kept {rows_after_drop} rows after removing NaNs/Infs in key columns.")

        if df.empty:
            logging.error("No valid data remaining after cleaning. Exiting.")
            return
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_PATH}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading or processing data: {e}. Exiting.")
        return

    # --- Identify Temperatures ---
    temperatures_in_data = sorted(df['temperature'].unique())
    if not temperatures_in_data:
         logging.error("No unique temperatures found in the cleaned data. Exiting.")
         return
    logging.info(f"Temperatures found in data: {temperatures_in_data}")

    global TEMP_COLORS
    try:
        TEMP_COLORS_float = {float(k): v for k, v in TEMP_COLORS.items()}
    except ValueError as e:
        logging.error(f"Error converting TEMP_COLORS keys to float: {e}. Ensure keys are numeric.")
        return

    # Use all temperatures found in data for distribution plots
    temps_for_dist_plots = temperatures_in_data

    # Identify temperatures with defined colors for the domain plot highlights
    temps_for_domain_plot_highlights = [t for t in temperatures_in_data if t in TEMP_COLORS_float]

    if not temps_for_dist_plots:
        logging.error("No temperatures identified to generate plots for. Exiting.")
        return

    logging.info(f"Will generate distribution plots for temperatures: {temps_for_dist_plots}")
    if temps_for_domain_plot_highlights:
        logging.info(f"Will generate domain plot with highlights for temperatures: {temps_for_domain_plot_highlights}")
    else:
        logging.warning("No temperatures with defined colors found in data; domain plot will have no mean highlights.")

    # --- Generate Plots ---
    try:
        # Generate distribution plots for all found temperatures
        logging.info("--- Generating Distribution Plots ---")
        generate_distribution_plots(df, temps_for_dist_plots)

        # --- Generate Domain Plot ---
        logging.info("--- Generating Domain-Level Plot ---")
        # The necessary checks (if df is empty, if aggregation is empty)
        # are handled *inside* generate_domain_level_density_plot.
        # We just need to ensure the initial df isn't empty before calling it.
        if not df.empty:
             # Pass the list of temperatures that *should* have highlights
             generate_domain_level_density_plot(df, temps_for_domain_plot_highlights)
        else:
            # This case should have been caught earlier, but double-check
            logging.error("Main DataFrame 'df' is unexpectedly empty. Skipping domain-level plot generation.")

    except Exception as e:
        logging.error(f"An error occurred during plot generation: {e}", exc_info=True) # Log traceback


    logging.info("Figure 2 generation process finished.")

if __name__ == "__main__":
    # Basic check for data file existence before starting
    if not os.path.isfile(DATA_PATH):
         # Log to console as logger might not be set up yet
         print(f"ERROR: Data file not found: {DATA_PATH}")
         logging.basicConfig(level=logging.ERROR) # Basic config for logging error
         logging.error(f"Data file not found: {DATA_PATH}")
    else:
        # Ensure output directories exist before setting up logging inside main
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        main()