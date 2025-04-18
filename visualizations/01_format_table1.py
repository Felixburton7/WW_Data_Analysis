# /home/s_felix/FINAL_PROJECT/Data_Analysis/visualizations/01_format_table1.py
# Version 7: Adjusted column widths for better header spacing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # Import main matplotlib library for rcParams
import os

# --- Font Configuration ---
# Attempt to set Times New Roman globally
try:
    # Set default font family to serif, and prioritize Times New Roman
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
    # Specify font properties for different elements explicitly
    FONT_PROPS_STD = {'family': 'Times New Roman', 'size': 10} # Standard cell text
    FONT_PROPS_BOLD = {'family': 'Times New Roman', 'size': 10, 'weight': 'bold'} # Bold cell text
    HEADER_FONT_PROPS = {'family': 'Times New Roman', 'size': 10, 'weight': 'bold'} # Header text
    TITLE_FONT_PROPS = {'family': 'Times New Roman', 'size': 12, 'weight': 'bold'} # Figure Title
    CAPTION_FONT_PROPS = {'family': 'Times New Roman', 'size': 9, 'style': 'italic'} # Caption
    print("Attempted to set font to Times New Roman.")
except Exception as e:
    print(f"Warning: Could not set Times New Roman font ({e}). Using default serif.")
    # Fallback font properties if TNR fails
    FONT_PROPS_STD = {'size': 10}
    FONT_PROPS_BOLD = {'size': 10, 'weight': 'bold'}
    HEADER_FONT_PROPS = {'size': 10, 'weight': 'bold'}
    TITLE_FONT_PROPS = {'size': 12, 'weight': 'bold'}
    CAPTION_FONT_PROPS = {'size': 9, 'style': 'italic'}


# --- Other Configuration ---
METRICS_CSV = "../results_output/tables/table1_overall_metrics.csv" # Relative path
TABLE_OUTPUT_DIR = "../results_output/tables"
TABLE_OUTPUT_PNG = os.path.join(TABLE_OUTPUT_DIR, "table1_overall_metrics_formatted_style.png") # Output name
FIG_DPI = 300 # Resolution for the PNG image
N_DECIMALS = 2 # Standard number of decimals to round to

# --- Column Widths ---
# Adjust these relative widths to control spacing
# You might need to fine-tune these values
COL_WIDTHS = [
    0.22,  # Model name (wider)
    0.13,  # PCC
    0.13,  # R2
    0.15,  # MAE (Å)
    0.15,  # RMSE (Å)
    0.15   # MedAE (Å)
]
# Ensure COL_WIDTHS length matches number of columns in df_table_display
assert len(COL_WIDTHS) == 6, "COL_WIDTHS list length must match number of table columns"

# --- Main Execution ---
if __name__ == "__main__":
    print("Generating PNG image for Table 1 from CSV (Example Style, Adjusted Widths)...")

    # Ensure output directory exists for saving
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)

    # --- Load Data ---
    try:
        df = pd.read_csv(METRICS_CSV)
        print(f"Loaded metrics data from: {METRICS_CSV}")
    except FileNotFoundError:
        print(f"ERROR: Metrics file not found: {METRICS_CSV}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Could not load metrics CSV: {e}")
        exit(1)

    # --- Prepare Data for the Table ---
    # Apply STANDARD rounding and format as string
    cols_to_format = ['PCC', 'R2', 'MAE', 'RMSE', 'MedAE']
    df_display = df.copy() # Create a copy for display formatting

    for col in cols_to_format:
        # Round NaN aware, then format
        df_display[col] = df[col].apply(lambda x: f"{round(x, N_DECIMALS):.{N_DECIMALS}f}" if pd.notna(x) else "NaN")

    # Select and Rename Columns for Table
    df_table_display = df_display[['Model', 'PCC', 'R2', 'MAE', 'RMSE', 'MedAE']].copy()
    df_table_display.rename(columns={
        'Model':'Model', # Keep 'Model' as is for header simplicity
        'PCC': 'PCC (↑)',
        'R2': 'R² (↑)',
        'MAE': 'MAE (Å) (↓)',
        'RMSE': 'RMSE (Å) (↓)',
        'MedAE': 'MedAE (Å) (↓)'
    }, inplace=True)

    # --- Create Figure and Table ---
    # Adjust figsize width slightly if needed based on content
    fig, ax = plt.subplots(figsize=(7.5, 3.0)) # Adjusted width slightly

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table with specified column widths
    tab = plt.table(cellText=df_table_display.values,
                    colLabels=df_table_display.columns,
                    colWidths=COL_WIDTHS, # <--- USE CONFIGURED WIDTHS
                    loc='center',
                    cellLoc='center', # Default center, override first col below
                    edges='open'
                   )

    # --- Style the Table ---
    tab.auto_set_font_size(False)
    # Font size is set within the loop below using FONT_PROPS
    tab.scale(1.0, 1.9) # Adjust scale for padding (increased vertical)

    # Iterate through cells for styling
    esm_flex_row_index = df_table_display[df_table_display['Model'] == 'ESM-Flex'].index[0] + 1 # +1 for header row
    num_rows, num_cols = df_table_display.shape[0] + 1, df_table_display.shape[1]

    for i in range(num_rows):
        for j in range(num_cols):
            cell = tab[i, j]
            cell.set_edgecolor('none') # Ensure no cell borders are drawn by default

            # --- Font and Alignment ---
            is_esm_flex_row = (i == esm_flex_row_index)
            current_font_props = FONT_PROPS_BOLD if is_esm_flex_row else FONT_PROPS_STD

            # Special handling for header
            if i == 0:
                cell.set_text_props(**HEADER_FONT_PROPS)
                # cell.set_facecolor("#EEEEEE") # Optional: Light grey header background
                cell.visible_edges = 'B'
                cell.set_edgecolor('black')
                cell.set_linewidth(1.0) # Adjust thickness as needed
            else:
                 cell.set_text_props(**current_font_props) # Apply standard or bold font

            # Alignment
            if j == 0: # First column (Model names)
                cell.set_text_props(ha='left')
                cell._loc = 'left'
                # Width is now controlled by colWidths list
            else: # Other columns
                cell.set_text_props(ha='center')
                cell._loc = 'center'
                # Width is now controlled by colWidths list

    # --- Add Title (Optional - often better in figure caption) ---
    # plt.suptitle("Table 1: Overall performance on the holdout set", **TITLE_FONT_PROPS, y=0.98)

    # --- Add Caption ---
    caption = ("Table 1: Overall performance on the holdout set. Arrows (↑)/(↓) indicate higher/lower values are better.\n"
               "Performance metrics calculated on the holdout set. Values rounded to two decimal places. Best performance shown in bold.")
    # Place caption below the table area
    fig.text(0.5, -0.05, caption, ha='center', va='bottom', **CAPTION_FONT_PROPS, wrap=True) # Lowered y slightly

    # --- Save PNG Image ---
    # Adjust layout to prevent caption/title overlap before saving
    plt.tight_layout(rect=[0, 0.0, 1, 0.95]) # Adjust rect to fit elements if needed
    try:
        fig.savefig(TABLE_OUTPUT_PNG, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.3) # Added padding
        print(f"Table image saved successfully to: {TABLE_OUTPUT_PNG}")
    except Exception as e:
        print(f"ERROR: Could not save table image: {e}")

    # plt.show() # Uncomment to display plot interactively if needed

    print("Script finished.")